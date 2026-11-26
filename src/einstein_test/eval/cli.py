"""CLI do harness de avaliação do RAG.

Uso:
  uv run python -m einstein_test.eval --backend fake --limit 2 --format md
  uv run python -m einstein_test.eval --backend real --judge llm \
      --format html --out eval/reports/latest.html
  uv run python -m einstein_test.eval --backend langfuse --judge llm

`--backend fake` é determinístico (fakes, sem chave) — smoke/CI.
`--backend real` usa banco + embedder + RAG reais (requer OPENAI_API_KEY e
infra) e é opt-in.
`--backend langfuse` usa o ambiente real e roda um experimento Langfuse
(dataset + task + evaluators), registrando scores no trace de cada item
(requer LANGFUSE_ENABLED=true + infra do Langfuse).
"""

from __future__ import annotations

import argparse
import os
import re
from contextlib import contextmanager
from pathlib import Path

from einstein_test.eval.dataset import EvalQuestion, load_dataset
from einstein_test.eval.judge import FakeJudge, LLMJudge
from einstein_test.eval.report import render_report
from einstein_test.eval.runner import run_eval

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = PROJECT_ROOT / "eval" / "dataset" / "dev.jsonl"
DEFAULT_CORPUS = PROJECT_ROOT / "eval" / "corpus"
DATASET_NAME = os.getenv("EVAL_DATASET_NAME", "einstein-rag-eval")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="eval", description="Harness de avaliação do RAG")
    p.add_argument("--stage", choices=["retrieval", "generation", "full"], default="full")
    p.add_argument("--backend", choices=["fake", "real", "langfuse"], default="fake")
    p.add_argument("--judge", choices=["fake", "llm"], default="fake")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    p.add_argument("--format", choices=["md", "html", "json"], default="md")
    p.add_argument("--out", type=Path, default=None)
    return p


def _build_judge(name: str):
    return LLMJudge() if name == "llm" else FakeJudge()


def _run_fake(questions: list[EvalQuestion], judge, limit: int | None):
    from einstein_test.vector import FakeEmbedder
    from einstein_test.vector.similarity import cosine_similarity

    embedder = FakeEmbedder(dim=8)

    def search_fn(q: EvalQuestion) -> list[int]:
        return list(range(1, len(q.expected_files) + 1))

    def answer_fn(q: EvalQuestion) -> tuple[str, list[str], float, float]:
        if q.edge == "no_context":
            return "FABRICADO — dado não está no corpus", [], 0.05, 0.10
        return q.gold_answer, [f"contexto do {q.id}"], 0.05, 0.10

    def similarity_fn(answer: str, gold: str) -> float:
        if not gold:
            return 0.0
        return cosine_similarity(embedder.embed([answer])[0], embedder.embed([gold])[0])

    return run_eval(
        questions,
        search_fn=search_fn,
        answer_fn=answer_fn,
        similarity_fn=similarity_fn,
        judge=judge,
        limit=limit,
    )


@contextmanager
def _real_env(questions: list[EvalQuestion], corpus_dir: Path):
    """Ambiente real de avaliação: usuário + chats + ingestão dos arquivos.

    Yield `(search_fn, answer_fn, file_id_of, embedder)` e faz hard-delete das
    linhas criadas no banco ao final (chunks/files/chats/user).
    """
    import asyncio
    import uuid

    from sqlalchemy.orm import Session

    from einstein_test.api.dependencies import configure_session_factory
    from einstein_test.db.engine import create_engine_from_config
    from einstein_test.eval.dataset import load_corpus_file
    from einstein_test.eval.generation import run_generation
    from einstein_test.models import Chat, File, FileChunk, User
    from einstein_test.repositories.chat import ChatRepositorySQLAlchemy
    from einstein_test.repositories.file import FileRepositorySQLAlchemy
    from einstein_test.services.chat import ChatService
    from einstein_test.services.file import FileService
    from einstein_test.services.rag.service import get_rag_service
    from einstein_test.vector import VectorRepositorySQLAlchemy, get_embedder

    engine = create_engine_from_config()
    configure_session_factory(engine)
    embedder = get_embedder()
    # Um único event loop para todas as perguntas: `asyncio.run` por pergunta
    # fecha o loop e quebra o cliente async do LLM (httpx) com
    # "RuntimeError: Event loop is closed".
    loop = asyncio.new_event_loop()

    with Session(engine) as s:
        user = User(
            email=f"eval_{uuid.uuid4().hex[:8]}@b.com",
            password_hash="!eval",
            model="azure/gpt-5-mini",
        )
        s.add(user)
        s.commit()
        user_id = user.id

    created_files: list[int] = []
    created_chats: list[int] = []

    def _cleanup() -> None:
        # Hard-delete das linhas criadas (chunks/files/chats/user). Os objetos
        # de storage (S3) podem permanecer órfãos — aceitável para eval local.
        with Session(engine) as s:
            if created_files:
                s.query(FileChunk).filter(FileChunk.file_id.in_(created_files)).delete(
                    synchronize_session=False
                )
                s.query(File).filter(File.id.in_(created_files)).delete(
                    synchronize_session=False
                )
            if created_chats:
                s.query(Chat).filter(Chat.id.in_(created_chats)).delete(
                    synchronize_session=False
                )
            s.query(User).filter(User.id == user_id).delete(synchronize_session=False)
            s.commit()

    envs: dict[str, tuple[int, dict[str, int]]] = {}
    try:
        # Um chat por pergunta: ingere os docs de `expected_files` (create_file
        # já extrai/chunka/embed quando o vector_repository é fornecido) e
        # guarda o mapeamento nome -> file_id.
        for q in questions:
            with Session(engine) as s:
                chat = ChatService(ChatRepositorySQLAlchemy(s)).create_chat(
                    user_id=user_id, name=f"eval-{q.id}"
                )
                file_svc = FileService(
                    FileRepositorySQLAlchemy(s),
                    vector_repository=VectorRepositorySQLAlchemy(s),
                )
                name_to_id: dict[str, int] = {}
                for name in q.expected_files:
                    content = load_corpus_file(corpus_dir, name)
                    created = file_svc.create_file(
                        chat_id=chat.id, name=name, content=content,
                        content_type="text/plain",
                    )
                    created_files.append(created.id)
                    name_to_id[name] = created.id
                created_chats.append(chat.id)
                envs[q.id] = (chat.id, name_to_id)
                s.commit()

        def file_id_of(q: EvalQuestion, name: str) -> int:
            return envs[q.id][1][name]

        def search_fn(q: EvalQuestion) -> list[int]:
            chat_id, _ = envs[q.id]
            embedding = embedder.embed([q.question])[0]
            with Session(engine) as s:
                retriever = VectorRepositorySQLAlchemy(s)
                chunks = retriever.search_similar(
                    embedding=embedding, chat_id=chat_id,
                    chunks_per_file=2, max_chunks=12,
                )
            seen: list[int] = []
            for c in chunks or []:
                if c.file_id not in seen:
                    seen.append(c.file_id)
            return seen

        def answer_fn(q: EvalQuestion) -> tuple[str, list[str], float, float]:
            chat_id, _ = envs[q.id]

            async def _call():
                with Session(engine) as s:
                    retriever = VectorRepositorySQLAlchemy(s)
                    rag = get_rag_service(retriever)
                    return await run_generation(
                        question=q.question,
                        chat_id=chat_id,
                        rag_service=rag,
                        retriever=retriever,
                        embedder=embedder,
                        history=[],
                    )

            res = loop.run_until_complete(_call())
            return res.answer, res.context, res.ttft, res.latency

        yield search_fn, answer_fn, file_id_of, embedder
    finally:
        _cleanup()
        if not loop.is_closed():
            loop.close()


def _run_real(questions: list[EvalQuestion], judge, corpus_dir: Path, limit: int | None):
    if limit is not None:
        questions = questions[:limit]

    from einstein_test.vector.similarity import cosine_similarity

    with _real_env(questions, corpus_dir) as (search_fn, answer_fn, file_id_of, embedder):
        def similarity_fn(answer: str, gold: str) -> float:
            if not gold:
                return 0.0
            return cosine_similarity(
                embedder.embed([answer])[0], embedder.embed([gold])[0]
            )

        return run_eval(
            questions,
            search_fn=search_fn,
            answer_fn=answer_fn,
            similarity_fn=similarity_fn,
            judge=judge,
            file_id_of=file_id_of,
            limit=None,
        )


def _run_langfuse(questions: list[EvalQuestion], judge, corpus_dir: Path, limit: int | None):
    """Backend Langfuse: push do dataset + run_experiment (task = RAG real).

    O task retorna a resposta/contexto/perf do RAG e as métricas de retrieval;
    os evaluators convertem isso em scores Langfuse por item do experimento.
    """
    from einstein_test.eval.metrics import hit_rate_at_1, mrr_at_k, recall_at_k
    from einstein_test.services.rag.eval_langfuse import (
        make_retrieval_evaluator,
        make_verdict_evaluator,
        push_dataset,
        run_experiment,
    )

    if limit is not None:
        questions = questions[:limit]

    with _real_env(questions, corpus_dir) as (search_fn, answer_fn, file_id_of, embedder):
        def task(*, item, **kwargs):
            meta = item.get("metadata") or {}
            q = EvalQuestion(
                id=meta.get("id", ""),
                question=item["input"],
                expected_files=tuple(meta.get("expected_files", [])),
                gold_answer=item.get("expected_output") or "",
                domain=meta.get("domain", "synthetic"),
                edge=meta.get("edge", ""),
            )
            ranked = search_fn(q)
            answer, context, ttft, latency = answer_fn(q)
            expected = {file_id_of(q, name) for name in q.expected_files}
            return {
                "answer": answer,
                "context": context,
                "ttft": ttft,
                "latency": latency,
                "retrieval": {
                    "recall@5": recall_at_k(ranked, expected, 5),
                    "recall@12": recall_at_k(ranked, expected, 12),
                    "mrr@5": mrr_at_k(ranked, expected, 5),
                    "hit@1": hit_rate_at_1(ranked, expected),
                },
            }

        data = [
            {
                "input": q.question,
                "expected_output": q.gold_answer,
                "metadata": {
                    "id": q.id,
                    "expected_files": list(q.expected_files),
                    "domain": q.domain,
                    "edge": q.edge,
                },
            }
            for q in questions
        ]

        push_dataset(name=DATASET_NAME, questions=questions)
        return run_experiment(
            name=DATASET_NAME,
            data=data,
            task=task,
            evaluators=[make_verdict_evaluator(judge), make_retrieval_evaluator()],
        )


def _db_host() -> str:
    url = os.getenv("DATABASE_URL", "")
    match = re.search(r"@([^:/\s]+)", url)
    return match.group(1) if match else "sqlite"


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.dataset.exists():
        print(f"dataset não encontrado: {args.dataset}")
        return 1
    if not args.corpus.is_dir():
        print(f"corpus não encontrado: {args.corpus}")
        return 1
    questions = load_dataset(args.dataset)
    if not questions:
        print("dataset vazio")
        return 1
    if args.backend in ("real", "langfuse"):
        host = _db_host()
        if host not in ("localhost", "127.0.0.1", "sqlite") and os.getenv("EVAL_ALLOW_NON_LOCAL") != "1":
            print(
                f"--backend {args.backend} só roda contra banco local "
                "(localhost/127.0.0.1/sqlite); defina EVAL_ALLOW_NON_LOCAL=1 "
                "para liberar outro host"
            )
            return 1
    if args.backend == "langfuse":
        from einstein_test.services.rag.eval_langfuse import langfuse_eval_enabled

        if not langfuse_eval_enabled():
            print(
                "--backend langfuse requer LANGFUSE_ENABLED=true "
                "(e a infra do Langfuse, ver docker-compose)"
            )
            return 1
    judge = _build_judge(args.judge)

    if args.backend == "langfuse":
        result = _run_langfuse(questions, judge, args.corpus, args.limit)
        print(f"experimento Langfuse '{DATASET_NAME}' concluído: {result}")
        return 0
    if args.backend == "real":
        report = _run_real(questions, judge, args.corpus, args.limit)
    else:
        report = _run_fake(questions, judge, args.limit)

    content = render_report(report, args.format)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(content, encoding="utf-8")
        print(f"relatório em: {args.out}")
    else:
        print(content)
    for key, value in report.aggregates().items():
        print(f"{key}: {value:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())