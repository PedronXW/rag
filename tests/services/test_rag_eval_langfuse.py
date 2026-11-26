"""J-004 — Testes do Langfuse para avaliação (scores por trace + batched)."""


class _FakeClient:
    """Client Langfuse fake: registra scores criados e batched."""

    def __init__(self):
        self.scores = []
        self.batched_calls = 0
        self.datasets = []
        self.dataset_items = []
        self.experiments = []

    def create_score(self, **kwargs):
        self.scores.append(kwargs)

    def run_batched_evaluation(self, **kwargs):
        self.batched_calls += 1
        return {"status": "ok"}

    def create_dataset(self, **kwargs):
        self.datasets.append(kwargs)

    def create_dataset_item(self, **kwargs):
        self.dataset_items.append(kwargs)

    def run_experiment(self, **kwargs):
        self.experiments.append(kwargs)
        return {"name": kwargs["name"], "run_id": "run-1"}


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

    run_batched_eval(client=client, mapper=lambda item: item, evaluators=[_evaluator])
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


def _question():
    from einstein_test.eval.dataset import EvalQuestion

    return EvalQuestion(
        id="q1",
        question="Quando foi lançado?",
        expected_files=("produto.md",),
        gold_answer="em 2021",
        domain="synthetic",
        edge="",
    )


def test_push_dataset_espelha_perguntas(monkeypatch):
    """push_dataset cria dataset + itens (upsert por id)."""
    from einstein_test.services.rag.eval_langfuse import push_dataset

    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    client = _FakeClient()

    n = push_dataset(client=client, name="einstein-rag-eval", questions=[_question()])

    assert n == 1
    assert client.datasets == [{"name": "einstein-rag-eval"}]
    item = client.dataset_items[0]
    assert item["id"] == "q1"
    assert item["input"] == "Quando foi lançado?"
    assert item["expected_output"] == "em 2021"
    assert item["metadata"]["expected_files"] == ["produto.md"]


def test_push_dataset_ignora_dataset_existente(monkeypatch):
    """push_dataset segue para os itens quando o dataset já existe."""
    from einstein_test.services.rag.eval_langfuse import push_dataset

    monkeypatch.setenv("LANGFUSE_ENABLED", "true")

    class _JáExiste:
        def __init__(self):
            self.items = []

        def create_dataset(self, **kwargs):
            raise RuntimeError("already exists")

        def create_dataset_item(self, **kwargs):
            self.items.append(kwargs)

    client = _JáExiste()
    n = push_dataset(client=client, name="ds", questions=[_question()])
    assert n == 1
    assert len(client.items) == 1


def test_push_dataset_desabilitado_nao_envia(monkeypatch):
    from einstein_test.services.rag.eval_langfuse import push_dataset

    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    client = _FakeClient()
    assert push_dataset(client=client, name="ds", questions=[_question()]) == 0
    assert client.dataset_items == []


def test_run_experiment_wraps_client(monkeypatch):
    from einstein_test.services.rag.eval_langfuse import run_experiment

    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    client = _FakeClient()

    def _task(*, item, **kwargs):
        return {"answer": "x"}

    result = run_experiment(
        client=client,
        name="einstein-rag-eval",
        data=[{"input": "q", "expected_output": "r"}],
        task=_task,
        evaluators=[],
    )
    assert result["run_id"] == "run-1"
    assert client.experiments[0]["name"] == "einstein-rag-eval"


def test_run_experiment_desabilitado_retorna_none(monkeypatch):
    from einstein_test.services.rag.eval_langfuse import run_experiment

    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    client = _FakeClient()
    assert (
        run_experiment(
            client=client,
            name="ds",
            data=[],
            task=lambda **kw: {},
            evaluators=[],
        )
        is None
    )
    assert client.experiments == []


def test_make_verdict_evaluator_retorna_scores():
    from einstein_test.eval.judge import FakeJudge
    from einstein_test.services.rag.eval_langfuse import make_verdict_evaluator

    evaluator = make_verdict_evaluator(FakeJudge())
    out = evaluator(
        input="pergunta",
        output={"answer": "resposta certa", "context": ["ctx"]},
        expected_output="resposta certa",
        metadata={},
    )
    names = {e.name for e in out}
    assert {"correctness", "faithfulness", "coverage"} <= names
    assert all(e.value >= 1.0 for e in out)


def test_make_retrieval_evaluator_converte_metricas():
    from einstein_test.services.rag.eval_langfuse import make_retrieval_evaluator

    evaluator = make_retrieval_evaluator()
    out = evaluator(output={"retrieval": {"recall@5": 1.0, "recall@12": 0.5, "mrr@5": 0.5, "hit@1": 1.0}})
    by_name = {e.name: e.value for e in out}
    assert by_name == {"recall@5": 1.0, "recall@12": 0.5, "mrr@5": 0.5, "hit@1": 1.0}


def test_cli_backend_langfuse_desabilitado_sai_1(tmp_path, capsys):
    """--backend langfuse sem LANGFUSE_ENABLED=true sai 1 com mensagem."""
    import json

    from einstein_test.eval.cli import main as cli_main

    dataset = tmp_path / "dev.jsonl"
    dataset.write_text(json.dumps({
        "id": "q1", "question": "pergunta",
        "expected_files": ["produto.md"], "gold_answer": "g",
    }) + "\n")
    exit_code = cli_main([
        "--backend", "langfuse", "--dataset", str(dataset), "--corpus", str(tmp_path),
    ])
    assert exit_code == 1
    assert "LANGFUSE_ENABLED=true" in capsys.readouterr().out
