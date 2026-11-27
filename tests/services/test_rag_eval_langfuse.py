"""K-001 — Testes do eval via Langfuse (módulo vivo mantido)."""

import pytest


class _FakeClient:
    def __init__(self):
        self.scores = []
        self.batched_calls = 0

    def create_score(self, **kwargs):
        self.scores.append(kwargs)

    def run_batched_evaluation(self, **kwargs):
        self.batched_calls += 1
        return {"status": "ok"}


def test_score_por_trace_quando_habilitado(monkeypatch):
    from einstein_test.services.rag.eval_langfuse import register_trace_score

    client = _FakeClient()
    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    register_trace_score(client=client, trace_id="trace-1", name="faithfulness", value=1.0)
    assert len(client.scores) == 1
    assert client.scores[0]["trace_id"] == "trace-1"


def test_desabilitado_nao_registra(monkeypatch):
    from einstein_test.services.rag.eval_langfuse import langfuse_eval_enabled

    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    assert langfuse_eval_enabled() is False


def test_batched_evaluation(monkeypatch):
    from einstein_test.services.rag.eval_langfuse import run_batched_eval

    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    client = _FakeClient()

    def _evaluator(item):
        return {"faithfulness": 1.0}

    run_batched_eval(client=client, mapper=lambda item: item, evaluators=[_evaluator])
    assert client.batched_calls == 1