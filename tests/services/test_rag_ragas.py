"""I-006 — Testes da avaliação LLM-as-judge (RAGAS)."""



class _JudgeLLM:
    """Juiz stub determinístico: avaliação fixa."""

    def evaluate(self, *, question, answer, contexts):
        # retorna scores deterministicos
        return {"faithfulness": 1.0, "answer_correctness": 1.0}


def test_ragas_calcula_metricas():
    """AC-001: evaluate_ragas calcula faithfulness e answer_correctness."""
    from einstein_test.services.rag.ragas_eval import evaluate_ragas

    cases = [{"question": "q", "reference": "resposta esperada"}]
    results = evaluate_ragas(cases, judge=_JudgeLLM())
    assert results[0]["faithfulness"] == 1.0
    assert results[0]["answer_correctness"] == 1.0


def test_juiz_injetavel():
    """AC-002: o juiz é injetável (stub)."""
    from einstein_test.services.rag.ragas_eval import evaluate_ragas

    results = evaluate_ragas(
        [{"question": "q", "reference": "r"}],
        judge=_JudgeLLM(),
    )
    assert results, "deve rodar com juiz stub"


def test_roda_graph_para_contexto():
    """AC-003: o harness roda o graph para obter resposta + contexto."""
    from einstein_test.services.rag.ragas_eval import collect_case

    class _Ret:
        def search_similar(self, embedding, chat_id, chunks_per_file, max_chunks):
            return []

    class _Emb:
        def embed(self, texts):
            return [[0.1] * 8 for _ in texts]

    class _LLM:
        def stream(self, messages):
            yield "resposta"

        async def astream(self, messages):
            yield "resposta"

    collected = collect_case(
        {"question": "q", "reference": "r"},
        llm=_LLM(),
        retriever=_Ret(),
        embedder=_Emb(),
    )
    assert collected["answer"]
    assert "contexts" in collected


def test_ragas_cli_exit():
    """AC-004: o harness é executável via CLI (exit 0 no limiar)."""
    from einstein_test.services.rag.ragas_eval import main

    rc = main(["--cases", "1", "--min", "0.0"])
    assert rc == 0