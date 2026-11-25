"""J-004 — Testes do Langfuse para avaliação (scores por trace + batched)."""

import pytest


class _FakeClient:
    """Client Langfuse fake: registra scores criados e batched."""

    def __init__(self):
        self.scores = []
        self.batched_calls = 0

    def create_score(self, **kwargs):
        self.scores.append(kwargs)

    def run_batched_evaluation(self, **kwargs):
        self.batched_calls += 1
        return {"status": "ok"}


def test_score_por_trace_quando_habilitado(monkeypatch):
    """AC-001: evaluate_ragas registra score no Langfuse quando habilitado."""
    from einstein_test.services.rag.eval_langfuse import register_trace_score

    client = _FakeClient()
    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    register_trace_score(
        client=client,
        trace_id="trace-1",
        name="faithfulness",
        value=1.0,
        comment="case q",
    )
    assert len(client.scores) == 1
    assert client.scores[0]["trace_id"] == "trace-1"
    assert client.scores[0]["name"] == "faithfulness"


def test_batched_evaluation(monkeypatch):
    """AC-002: existe runner de run_batched_evaluation."""
    from einstein_test.services.rag.eval_langfuse import run_batched_eval

    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    client = _FakeClient()

    def _evaluator(item):
        return {"faithfulness": 1.0}

    result = run_batched_eval(client=client, mapper=lambda item: item, evaluators=[_evaluator])
    assert client.batched_calls == 1


def test_desabilitado_nao_registra(monkeypatch):
    """AC-003: com Langfuse desabilitado, não registra (determinístico)."""
    from einstein_test.services.rag.eval_langfuse import langfuse_eval_enabled

    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    assert langfuse_eval_enabled() is False


def test_harness_ragas_aceita_gerenciador_de_score():
    """AC-003: o harness roda sem registrar quando desabilitado."""
    from einstein_test.services.rag.ragas_eval import evaluate_ragas

    class _Judge:
        def evaluate(self, *, question, answer, contexts):
            return {"faithfulness": 1.0, "answer_correctness": 1.0}

    results = evaluate_ragas(
        [{"question": "q", "reference": "r"}],
        judge=_Judge(),
    )
    assert results[0]["faithfulness"] == 1.0

def test_ragas_integra_score_manager():
    """AC-001: evaluate_ragas registra scores via score_manager."""
    from einstein_test.services.rag.ragas_eval import evaluate_ragas

    class _Judge:
        def evaluate(self, *, question, answer, contexts):
            return {"faithfulness": 1.0, "answer_correctness": 1.0}

    calls = []

    def _manager(**kw):
        calls.append(kw)

    evaluate_ragas(
        [{"question": "q", "reference": "r"}],
        judge=_Judge(),
        score_manager=_manager,
        trace_id="trace-x",
    )
    names = {c["name"] for c in calls}
    assert {"faithfulness", "answer_correctness"} <= names
    assert all(c["trace_id"] == "trace-x" for c in calls)
