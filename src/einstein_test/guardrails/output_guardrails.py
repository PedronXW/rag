"""Guardrails de saída: validação da resposta antes de persistir.

- `EmptyAnswerGuard`: bloqueia respostas vazias/triviais.
- `HarmfulContentGuard`: bloqueia conteúdo nocivo por blocklist.
- `GroundingGuard`: heurística de grounding — verifica se termos do contexto
  recuperado aparecem na resposta (detecta alucinação grosseira). Opcionalmente
  pode usar LLM/judge (não implementado aqui — deixado como ponto de extensão).

Tudo determinístico (sem LLM) por default para ser testável.
"""

from __future__ import annotations

import re

from einstein_test.guardrails.base import (
    GuardAction,
    Guardrail,
    GuardrailPipeline,
    GuardrailResult,
)

_HARMFUL_MARKS = (
    "receita de bomba",
    "como matar",
    "como fabricar drogas",
    "como roubar",
    "instruções para suicídio",
    "como cometer fraude",
)

_WORD_RE = re.compile(r"[a-zà-ú0-9]{4,}")


def _tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


class EmptyAnswerGuard(Guardrail):
    """Bloqueia respostas vazias ou muito curtas (não respondeu nada útil)."""

    name = "empty_answer"

    def __init__(self, min_chars: int = 3) -> None:
        self.min_chars = min_chars

    def check(self, text: str, **context) -> GuardrailResult:
        if len((text or "").strip()) < self.min_chars:
            return GuardrailResult(
                allowed=False,
                action=GuardAction.block,
                guard=self.name,
                reason="resposta vazia ou trivial",
            )
        return GuardrailResult(allowed=True)


class HarmfulContentGuard(Guardrail):
    """Bloqueia conteúdo nocivo por blocklist (determinístico)."""

    name = "harmful_content"

    def __init__(self, marks: tuple[str, ...] | None = None) -> None:
        self.marks = marks or _HARMFUL_MARKS

    def check(self, text: str, **context) -> GuardrailResult:
        low = text.lower()
        for mark in self.marks:
            if mark in low:
                return GuardrailResult(
                    allowed=False,
                    action=GuardAction.block,
                    guard=self.name,
                    reason=f"conteúdo nocivo detectado: {mark!r}",
                )
        return GuardrailResult(allowed=True)


class GroundingGuard(Guardrail):
    """Heurística de grounding: overlap de termos entre contexto e resposta.

    Se a resposta quase não usa termos do contexto, marca `notify` (possível
    alucinação) — não bloqueia, apenas sinaliza.
    """

    name = "grounding"

    def __init__(self, min_overlap: float = 0.15) -> None:
        self.min_overlap = min_overlap

    def check(self, text: str, **context) -> GuardrailResult:
        context_text = context.get("chunks") or context.get("context") or ""
        if not isinstance(context_text, str):
            context_text = "\n".join(_content(c) for c in (context_text or []))
        answer_tokens = _tokens(text)
        context_tokens = _tokens(context_text)
        if not answer_tokens or not context_tokens:
            return GuardrailResult(allowed=True)
        overlap = len(answer_tokens & context_tokens) / len(answer_tokens)
        if overlap < self.min_overlap:
            return GuardrailResult(
                allowed=True,
                action=GuardAction.notify,
                guard=self.name,
                reason=f"baixo grounding (overlap {overlap:.2f})",
            )
        return GuardrailResult(allowed=True)


def _content(c) -> str:
    if isinstance(c, dict):
        return c.get("content", "") or c.get("text", "")
    return getattr(c, "content", str(c))


def build_output_pipeline(*, enabled: bool = True) -> GuardrailPipeline:
    """Pipeline de saída padrão (gate por `enabled`)."""
    if not enabled:
        return GuardrailPipeline()
    return GuardrailPipeline(
        [EmptyAnswerGuard(), HarmfulContentGuard(), GroundingGuard()]
    )


__all__ = [
    "EmptyAnswerGuard",
    "HarmfulContentGuard",
    "GroundingGuard",
    "build_output_pipeline",
]