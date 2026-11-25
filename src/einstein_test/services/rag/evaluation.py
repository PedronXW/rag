"""Harness de avaliação determinístico do RAG (H-005).

`evaluate_rag` roda o graph clássico sobre casos de teste com stubs e calcula
`relevance` (recuperou chunk esperado), `fidelity` (groundedness) e `accuracy`
(resposta contém o esperado). Sem LLM judge — determinístico.
"""

import argparse
import sys

from einstein_test.services.rag.graph import build_graph


def _run_case(graph, case: dict, retriever) -> dict:
    result = graph.invoke({"question": case["question"], "chat_id": 1, "history": []})
    sources = result.get("sources", [])
    source_ids = {s.get("chunk_id") for s in sources}
    expected_ids = set(case.get("chunk_ids", []))
    answer = result.get("answer", "")

    relevance = 1.0 if (source_ids & expected_ids) else 0.0
    # fidelity determinística: a resposta é grounded se contém trecho do contexto
    context_text = " ".join(s.get("content", "") for s in sources).lower()
    fidelity = 1.0 if answer and context_text and answer.lower() in context_text else 0.0
    expected_text = case.get("expected", "")
    accuracy = 1.0 if expected_text and expected_text in answer else 0.0

    return {
        "question": case["question"],
        "answer": answer,
        "relevance": relevance,
        "fidelity": fidelity,
        "accuracy": accuracy,
    }


def evaluate_rag(cases, *, llm, retriever, embedder) -> list[dict]:
    """Roda o graph clássico (com stubs) sobre os casos e calcula as métricas."""
    graph = build_graph(llm=llm, retriever=retriever, embedder=embedder)
    return [_run_case(graph, c, retriever) for c in cases]


def aggregate(results: list[dict]) -> dict:
    if not results:
        return {"relevance": 0.0, "fidelity": 0.0, "accuracy": 0.0}
    n = len(results)
    return {
        "relevance": round(sum(r["relevance"] for r in results) / n, 3),
        "fidelity": round(sum(r["fidelity"] for r in results) / n, 3),
        "accuracy": round(sum(r["accuracy"] for r in results) / n, 3),
    }


def _stub_llm():
    from einstein_test.services.rag.llm import FakeChatLLM

    return FakeChatLLM(tokens=["resposta"])


def _stub_retriever():
    class _Ret:
        def search_similar(self, embedding, chat_id, chunks_per_file, max_chunks):
            return []

    return _Ret()


def _stub_embedder():
    class _Emb:
        def embed(self, texts):
            return [[0.1] * 8 for _ in texts]

    return _Emb()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Avalia o RAG (harness offline)")
    parser.add_argument("--cases", type=int, default=2, help="número de casos")
    parser.add_argument("--min", type=float, default=0.0, help="limiar por métrica")
    args = parser.parse_args(argv)

    cases = [
        {"question": f"pergunta {i}", "expected": "resposta", "chunk_ids": []}
        for i in range(args.cases)
    ]
    results = evaluate_rag(
        cases,
        llm=_stub_llm(),
        retriever=_stub_retriever(),
        embedder=_stub_embedder(),
    )
    scores = aggregate(results)
    print(f"relevance={scores['relevance']} fidelity={scores['fidelity']} accuracy={scores['accuracy']}")
    for name, value in scores.items():
        if value < args.min:
            print(f"FALHOU: {name}={value} abaixo do limiar {args.min}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())