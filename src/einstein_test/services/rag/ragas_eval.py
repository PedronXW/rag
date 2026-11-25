"""Avaliação LLM-as-judge (RAGAS) do RAG (I-006).

`evaluate_ragas` calcula `faithfulness` e `answer_correctness` usando um LLM
juiz injetável sobre os casos. O juiz é um stub determinístico nos testes; em
produção, pode ser um `BaseChatModel` do langchain ou uma integração RAGAS.
"""

import argparse
import sys
from typing import Any, Protocol

from einstein_test.services.rag.graph import build_graph


class Judge(Protocol):
    def evaluate(self, *, question: str, answer: str, contexts: list[str]) -> dict: ...


def collect_case(case: dict, *, llm, retriever, embedder) -> dict:
    """Roda o graph RAG para obter a resposta e o contexto de um caso."""
    graph = build_graph(llm=llm, retriever=retriever, embedder=embedder)
    result = graph.invoke({"question": case["question"], "chat_id": 1, "history": []})
    sources = result.get("sources", [])
    contexts = [s.get("content", "") for s in sources]
    return {"question": case["question"], "answer": result.get("answer", ""), "contexts": contexts}


def evaluate_ragas(cases, *, judge: Judge, llm=None, retriever=None, embedder=None) -> list[dict]:
    """Avalia os casos com um juiz LLM, calculando faithfulness e correctness.

    Usa `collect_case` para rodar o graph (a menos que o caso já traga
    `answer`/`contexts`); então o juiz avalia cada caso.
    """
    results = []
    for case in cases:
        if "answer" in case and "contexts" in case:
            collected = case
        elif llm is not None and retriever is not None and embedder is not None:
            collected = collect_case(case, llm=llm, retriever=retriever, embedder=embedder)
        else:
            collected = {"question": case["question"], "answer": "", "contexts": []}
        scores = judge.evaluate(
            question=collected["question"],
            answer=collected.get("answer", ""),
            contexts=collected.get("contexts", []),
        )
        results.append(
            {
                "question": collected["question"],
                "answer": collected.get("answer", ""),
                "faithfulness": scores.get("faithfulness", 0.0),
                "answer_correctness": scores.get("answer_correctness", 0.0),
            }
        )
    return results


def aggregate(results: list[dict]) -> dict:
    if not results:
        return {"faithfulness": 0.0, "answer_correctness": 0.0}
    n = len(results)
    return {
        "faithfulness": round(sum(r["faithfulness"] for r in results) / n, 3),
        "answer_correctness": round(sum(r["answer_correctness"] for r in results) / n, 3),
    }


def _stub_judge() -> Judge:
    class _Stub:
        def evaluate(self, *, question, answer, contexts):
            # juiz determinístico: 1.0 se há resposta e contexto não-vazio
            return {
                "faithfulness": 1.0 if (answer and contexts) else 0.0,
                "answer_correctness": 1.0 if answer else 0.0,
            }

    return _Stub()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Avalia o RAG (LLM-as-judge / RAGAS)")
    parser.add_argument("--cases", type=int, default=2, help="número de casos")
    parser.add_argument("--min", type=float, default=0.0, help="limiar por métrica")
    args = parser.parse_args(argv)

    cases = [{"question": f"pergunta {i}", "reference": "resposta"} for i in range(args.cases)]
    results = evaluate_ragas(cases, judge=_stub_judge())
    scores = aggregate(results)
    print(f"faithfulness={scores['faithfulness']} answer_correctness={scores['answer_correctness']}")
    for name, value in scores.items():
        if value < args.min:
            print(f"FALHOU: {name}={value} abaixo do limiar {args.min}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["evaluate_ragas", "collect_case", "aggregate", "main"]