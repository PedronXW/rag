"""Langfuse como plataforma de avaliação do RAG (J-004 / F-059).

- `register_trace_score`: registra um score (`create_score`) no trace de uma
  execução avaliada (avaliação contínua por trace).
- `push_dataset`: espelha as perguntas gold em um dataset Langfuse
  (`create_dataset` + `create_dataset_item`, idempotente por `q.id`).
- `run_experiment`: wrapper de `run_experiment` do Langfuse (task + evaluators)
  sobre o dataset, com tracing automático por item.
- `make_verdict_evaluator` / `make_retrieval_evaluator`: evaluators nativos do
  Langfuse que convertem o veredito do juiz e as métricas de retrieval
  (recall@5/12, MRR@5, hit@1) em scores por item.
- `run_batched_eval`: avalia traces existentes via `run_batched_evaluation`.
- `langfuse_eval_enabled`: gate por env (desabilitado nos testes).

O client do Langfuse é injetável (stub nos testes; `_get_client` em produção).
"""

from typing import Any, Callable

from einstein_test.eval.dataset import EvalQuestion
from einstein_test.eval.judge import JudgeVerdict
from einstein_test.services.rag.callbacks import _get_client, langfuse_enabled


def langfuse_eval_enabled() -> bool:
    return langfuse_enabled()


def _client():
    return _get_client()


def register_trace_score(
    *,
    client=None,
    trace_id: str | None,
    name: str,
    value: float,
    comment: str | None = None,
) -> bool:
    """Registra um score no trace do Langfuse (create_score).

    Retorna True se registrou; False se desabilitado (ou trace sem id).
    """
    if not langfuse_eval_enabled():
        return False
    if not trace_id:
        return False
    c = client or _client()
    c.create_score(
        name=name,
        value=value,
        trace_id=trace_id,
        comment=comment,
        data_type="NUMERIC",
    )
    return True


def push_dataset(
    client=None, *, name: str, questions: list[EvalQuestion]
) -> int:
    """Espelha as perguntas gold em um dataset Langfuse (idempotente).

    Usa `q.id` como id do item (`create_dataset_item` faz upsert por id), então
    re-executar não duplica. Retorna o número de itens enviados; 0 se
    desabilitado (ou sem perguntas).
    """
    if not langfuse_eval_enabled():
        return 0
    if not questions:
        return 0
    c = client or _client()
    try:
        c.create_dataset(name=name)
    except Exception:
        pass  # dataset já existe -> segue para os itens
    for q in questions:
        c.create_dataset_item(
            dataset_name=name,
            id=str(q.id),
            input=q.question,
            expected_output=q.gold_answer,
            metadata={
                "id": q.id,
                "expected_files": list(q.expected_files),
                "domain": q.domain,
                "edge": q.edge,
            },
        )
    return len(questions)


def run_experiment(
    client=None,
    *,
    name: str,
    data: list[dict],
    task: Callable[..., Any],
    evaluators: list[Callable[..., Any]] | None = None,
    run_name: str | None = None,
) -> Any:
    """Roda um experimento Langfuse (task + evaluators) sobre o dataset."""
    if not langfuse_eval_enabled():
        return None
    c = client or _client()
    return c.run_experiment(
        name=name,
        run_name=run_name,
        data=data,
        task=task,
        evaluators=evaluators or [],
    )


def make_verdict_evaluator(judge):
    """Evaluator Langfuse: converte o JudgeVerdict do juiz em scores (1–5)."""

    def evaluator(*, input, output, expected_output, metadata=None, **kwargs):
        from langfuse.experiment import Evaluation

        verdict: JudgeVerdict = judge.judge(
            question=input,
            gold_answer=expected_output or "",
            answer=output.get("answer", ""),
            context=output.get("context", []),
        )
        return [
            Evaluation(
                name="correctness", value=verdict.correctness, comment=verdict.rationale
            ),
            Evaluation(
                name="faithfulness", value=verdict.faithfulness, comment=verdict.rationale
            ),
            Evaluation(
                name="coverage", value=verdict.coverage, comment=verdict.rationale
            ),
        ]

    return evaluator


def make_retrieval_evaluator():
    """Evaluator Langfuse: scores de retrieval (recall@5/12, MRR@5, hit@1).

    O task deve expor `output["retrieval"]` com essas chaves (ver CLI eval).
    """

    def evaluator(*, output, **kwargs):
        from langfuse.experiment import Evaluation

        r = output.get("retrieval", {})
        return [
            Evaluation(name="recall@5", value=r.get("recall@5", 0.0)),
            Evaluation(name="recall@12", value=r.get("recall@12", 0.0)),
            Evaluation(name="mrr@5", value=r.get("mrr@5", 0.0)),
            Evaluation(name="hit@1", value=r.get("hit@1", 0.0)),
        ]

    return evaluator


def run_batched_eval(
    *,
    client=None,
    mapper: Callable[[dict], Any],
    evaluators: list[Callable[[Any], dict]],
    max_items: int | None = None,
) -> Any:
    """Avalia um dataset offline via run_batched_evaluation do Langfuse."""
    if not langfuse_eval_enabled():
        return None
    c = client or _client()
    return c.run_batched_evaluation(
        mapper=mapper,
        evaluators=evaluators,
        max_items=max_items,
    )


__all__ = [
    "register_trace_score",
    "push_dataset",
    "run_experiment",
    "make_verdict_evaluator",
    "make_retrieval_evaluator",
    "run_batched_eval",
    "langfuse_eval_enabled",
]