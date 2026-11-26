"""Testes TDD da feature F-010 (AC-001: Embedder).

Valida o Protocol Embedder, o OpenAIEmbedder (config via .env/OPENAI_*,
model cohere/embed-v4.0, base url OpenAI-compatível, dim 1536) e o
FakeEmbedder determinístico.
"""

import pytest

from einstein_test.vector import Embedder, FakeEmbedder, OpenAIEmbedder


def test_ac001_embedder_is_protocol():
    assert hasattr(Embedder, "embed"), "Embedder expõe embed(texts) -> list[list[float]]"


def test_ac001_openai_embedder_model_and_dim():
    assert OpenAIEmbedder.model == "cohere/embed-v4.0"
    assert OpenAIEmbedder.dim == 1536


def test_ac001_openai_embedder_base_url_default():
    assert OpenAIEmbedder.base_url == "https://spillway.ensinoeinstein.com/openai/v1"


def test_ac001_openai_embedder_reads_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://spillway.ensinoeinstein.com/openai/v1")
    monkeypatch.setenv("OPENAI_MODEL", "cohere/embed-v4.0")
    monkeypatch.delenv("OPENAI_EMBEDDING_DIM", raising=False)
    emb = OpenAIEmbedder()
    assert emb.api_key == "sk-test"
    assert emb.model == "cohere/embed-v4.0"
    assert emb.base_url == "https://spillway.ensinoeinstein.com/openai/v1"
    assert emb.dim == 1536


def test_ac001_openai_embedder_requires_api_key(monkeypatch):
    import einstein_test.vector.embedder as embedder_module

    monkeypatch.setattr(embedder_module, "load_dotenv", lambda: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        OpenAIEmbedder()


def test_ac001_fake_embedder_deterministic():
    emb = FakeEmbedder()
    a = emb.embed(["olá mundo"])
    b = emb.embed(["olá mundo"])
    assert a == b, "FakeEmbedder determinístico para o mesmo texto"
    assert len(a) == 1
    assert len(a[0]) == 1536, "vetor de 1536 dims (cohere/embed-v4.0)"


def test_ac001_fake_embedder_distinct_texts():
    emb = FakeEmbedder()
    a = emb.embed(["um texto"])
    b = emb.embed(["outro texto"])
    assert a != b, "textos distintos geram vetores distintos"


def test_ac001_fake_embedder_batch():
    emb = FakeEmbedder()
    vecs = emb.embed(["a", "b", "c"])
    assert len(vecs) == 3
    assert all(len(v) == 1536 for v in vecs)


def test_ac001_fake_embedder_dims_are_floats():
    emb = FakeEmbedder()
    vec = emb.embed(["x"])[0]
    assert all(isinstance(v, float) for v in vec)


def test_ollama_embedder_model_e_base_url_defaults(monkeypatch):
    monkeypatch.delenv("OLLAMA_EMBED_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    from einstein_test.vector.embedder import OllamaEmbedder

    assert OllamaEmbedder.model == "nomic-embed-text"
    assert OllamaEmbedder.base_url == "http://localhost:11434/v1"
    assert OllamaEmbedder().dim == 1536, "alvo é o dim do schema (vector(1536))"


def test_ollama_embedder_nao_exige_chave():
    from einstein_test.vector.embedder import OllamaEmbedder

    emb = OllamaEmbedder()
    assert emb.api_key == "ollama"


def test_ollama_embedder_zero_pad_ate_o_dim(monkeypatch):
    import openai

    import einstein_test.vector.embedder as emb_module

    class _FakeResponse:
        def __init__(self):
            self.data = [type("D", (), {"embedding": [0.1] * 768})()]

    class _FakeEmbeddings:
        def __init__(self):
            self.calls = []

        def create(self, model, input):
            self.calls.append((model, input))
            return _FakeResponse()

    class _FakeClient:
        def __init__(self):
            self.embeddings = _FakeEmbeddings()

    fake = _FakeClient()
    monkeypatch.setattr(openai, "OpenAI", lambda api_key, base_url: fake)
    emb = emb_module.OllamaEmbedder()
    vectors = emb.embed(["texto"])
    assert len(vectors[0]) == 1536
    assert vectors[0][:768] == [0.1] * 768
    assert vectors[0][768:] == [0.0] * (1536 - 768)


def test_get_embedder_roteia_por_env(monkeypatch):
    import einstein_test.vector as vector_module

    monkeypatch.setenv("EMBEDDER", "ollama")
    monkeypatch.setattr(vector_module, "_embedder", None)
    assert isinstance(vector_module.get_embedder(), vector_module.OllamaEmbedder)

    monkeypatch.setenv("EMBEDDER", "openai")
    monkeypatch.setattr(vector_module, "_embedder", None)
    assert isinstance(vector_module.get_embedder(), vector_module.OpenAIEmbedder)
    vector_module.set_embedder(None)