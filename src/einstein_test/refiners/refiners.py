"""Refiners concretos (determinísticos) e o pipeline padrão."""

from __future__ import annotations

from einstein_test.guardrails.input_guardrails import PIIGuard
from einstein_test.refiners.base import Refiner, RefinerPipeline


class DeduplicateRefiner(Refiner):
    """Remove linhas/parágrafos repetidos consecutivos."""

    def refine(self, text: str, **context) -> str:
        lines = [ln for ln in text.split("\n") if ln.strip()]
        deduped: list[str] = []
        for ln in lines:
            if not deduped or deduped[-1] != ln:
                deduped.append(ln)
        return "\n".join(deduped).strip()


class CitationRefiner(Refiner):
    """Anexa citações `[doc N]` dos chunks recuperados usados.

    O `context["chunks"]` deve ser uma lista de dicts com `document_id`
    (ou `doc_id`). Adiciona uma linha final listando as fontes únicas.
    """

    def refine(self, text: str, **context) -> str:
        chunks = context.get("chunks") or []
        doc_ids: list[int] = []
        seen: set[int] = set()
        for c in chunks:
            if isinstance(c, dict):
                doc = c.get("document_id") or c.get("doc_id")
            else:
                doc = getattr(c, "document_id", None) or getattr(c, "file_id", None)
            if doc is not None:
                did = int(doc)
                if did not in seen:
                    seen.add(did)
                    doc_ids.append(did)
        if not doc_ids:
            return text
        citation = "\n\n**Fontes:** " + ", ".join(f"[doc {d}]" for d in doc_ids)
        if citation in text:
            return text
        return text.rstrip() + citation


class PIIMaskRefiner(Refiner):
    """Mascara dado pessoal na resposta final (reusa a detecção do PIIGuard)."""

    def __init__(self) -> None:
        self._guard = PIIGuard()

    def refine(self, text: str, **context) -> str:
        res = self._guard.check(text)
        return res.redacted or text


def build_refiner_pipeline(
    *, enabled: bool = True, cite: bool = True, mask_pii: bool = True
) -> RefinerPipeline:
    """Pipeline de refiners padrão (gate por `enabled`)."""
    if not enabled:
        return RefinerPipeline()
    refiners: list[Refiner] = [DeduplicateRefiner()]
    if mask_pii:
        refiners.append(PIIMaskRefiner())
    if cite:
        refiners.append(CitationRefiner())
    return RefinerPipeline(refiners)


__all__ = [
    "DeduplicateRefiner",
    "CitationRefiner",
    "PIIMaskRefiner",
    "build_refiner_pipeline",
]