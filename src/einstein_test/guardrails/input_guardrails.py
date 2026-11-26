"""Guardrails de entrada: inspeção da pergunta do usuário antes do LLM.

- `PIIGuard`: detecta dado pessoal (CPF, e-mail, telefone, RG) e mascara
  (redact) ou bloqueia conforme a política.
- `PromptInjectionGuard`: heurística de injeção de prompt (instruções para
  ignorar o sistema/contexto) — bloqueia.
- `PolicyGuard`: tópicos proibidos configurados (`BLOCKED_TOPICS`) — bloqueia.
"""

from __future__ import annotations

import re

from einstein_test.guardrails.base import (
    GuardAction,
    Guardrail,
    GuardrailPipeline,
    GuardrailResult,
)

# CPF: 000.000.000-00 ou 00000000000 (11 dígitos)
_CPF_RE = re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")
# e-mail
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
# telefone BR: (00) 00000-0000 / 00 00000-0000 / +55 ...
_PHONE_RE = re.compile(r"(?:\+?55[ ]?)?(?:\(?\d{2}\)?[ ]?)?\d{4,5}-?\d{4}\b")

_INJECTION_MARKS = (
    "ignore as instruções",
    "ignore instruções",
    "ignore as instrucoes",
    "ignore instrucoes",
    "ignore o prompt",
    "ignore o contexto",
    "esqueça as instruções",
    "esqueça o prompt",
    "desconsidere as regras",
    "desconsidere as instruções",
    "ignore tudo o que foi dito",
    "você agora é",
    "act as",
    "ignore all previous",
    "forget all instructions",
    "system prompt",
)


class PIIGuard(Guardrail):
    """Detecta e mascara dado pessoal na entrada (LGPD).

    `action`: `redact` mascara o dado (default); `block` impede o envio.
    """

    name = "pii"

    def __init__(self, action: GuardAction = GuardAction.redact) -> None:
        self.action = action

    def _matches(self, text: str) -> list[str]:
        found: list[str] = []
        for rx in (_CPF_RE, _EMAIL_RE, _PHONE_RE):
            found.extend(rx.findall(text))
        return found

    def _mask(self, text: str, matches: list[str]) -> str:
        for m in matches:
            text = text.replace(m, "[PII]")
        return text

    def check(self, text: str, **context) -> GuardrailResult:
        matches = self._matches(text)
        if not matches:
            return GuardrailResult(allowed=True)
        if self.action == GuardAction.block:
            return GuardrailResult(
                allowed=False,
                action=GuardAction.block,
                guard=self.name,
                reason="dado pessoal detectado (bloqueio por política)",
            )
        return GuardrailResult(
            allowed=True,
            action=GuardAction.redact,
            guard=self.name,
            reason=f"dado pessoal mascarado ({len(matches)})",
            redacted=self._mask(text, matches),
        )


class PromptInjectionGuard(Guardrail):
    """Bloqueia tentativas de injeção de prompt (heurística determinística)."""

    name = "prompt_injection"

    def __init__(self, marks: tuple[str, ...] | None = None) -> None:
        self.marks = marks or _INJECTION_MARKS

    def check(self, text: str, **context) -> GuardrailResult:
        low = text.lower()
        for mark in self.marks:
            if mark in low:
                return GuardrailResult(
                    allowed=False,
                    action=GuardAction.block,
                    guard=self.name,
                    reason=f"marca de injeção de prompt: {mark!r}",
                )
        return GuardrailResult(allowed=True)


class PolicyGuard(Guardrail):
    """Bloqueia perguntas sobre tópicos proibidos (`blocked_topics`)."""

    name = "policy"

    def __init__(self, blocked_topics: list[str] | None = None) -> None:
        self.blocked_topics = [t.lower() for t in (blocked_topics or [])]

    def check(self, text: str, **context) -> GuardrailResult:
        low = text.lower()
        for topic in self.blocked_topics:
            if topic and topic in low:
                return GuardrailResult(
                    allowed=False,
                    action=GuardAction.block,
                    guard=self.name,
                    reason=f"tópico proibido pela política: {topic!r}",
                )
        return GuardrailResult(allowed=True)


def build_input_pipeline(
    *,
    enabled: bool = True,
    blocked_topics: list[str] | None = None,
    pii_action: GuardAction = GuardAction.redact,
) -> GuardrailPipeline:
    """Pipeline de entrada padrão (gate por `enabled`)."""
    guards: list[Guardrail] = []
    if not enabled:
        return GuardrailPipeline(guards)
    guards.append(PromptInjectionGuard())
    guards.append(PolicyGuard(blocked_topics))
    guards.append(PIIGuard(action=pii_action))
    return GuardrailPipeline(guards)


__all__ = [
    "PIIGuard",
    "PromptInjectionGuard",
    "PolicyGuard",
    "build_input_pipeline",
]