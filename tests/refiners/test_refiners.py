"""F-061 — Refiners determinísticos (dedup, citação, PII)."""

from einstein_test.refiners.refiners import (
    CitationRefiner,
    DeduplicateRefiner,
    PIIMaskRefiner,
    build_refiner_pipeline,
)


def test_deduplicate_remove_linhas_repetidas():
    r = DeduplicateRefiner()
    out = r.refine("linha um\nlinha dois\nlinha dois\nlinha três")
    assert out == "linha um\nlinha dois\nlinha três"


def test_citation_anexa_docs_unicos():
    r = CitationRefiner()
    out = r.refine(
        "resposta do assistente",
        chunks=[{"document_id": 1}, {"document_id": 1}, {"document_id": 2}],
    )
    assert "[doc 1]" in out
    assert "[doc 2]" in out
    assert out.endswith("]")
    # não duplica citação
    out2 = r.refine(out, chunks=[{"document_id": 1}, {"document_id": 2}])
    assert out2.count("**Fontes:**") == 1


def test_citation_sem_chunks_nao_altera():
    r = CitationRefiner()
    assert r.refine("resposta", chunks=[]) == "resposta"


def test_pii_mask_refiner():
    r = PIIMaskRefiner()
    out = r.refine("Meu e-mail é joao@x.com")
    assert "joao@x.com" not in out
    assert "[PII]" in out


def test_pipeline_encadeia():
    p = build_refiner_pipeline(enabled=True)
    out = p.run(
        "A escola foi fundada em 1997.\nA escola foi fundada em 1997.",
        chunks=[{"document_id": 3}],
    )
    assert out.count("A escola foi fundada em 1997.") == 1
    assert "[doc 3]" in out


def test_pipeline_off_nao_altera():
    p = build_refiner_pipeline(enabled=False)
    out = p.run("texto\n\ntexto com repetição\n\ntexto com repetição")
    assert "texto com repetição\n\ntexto com repetição" in out