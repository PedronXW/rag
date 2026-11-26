"""Base de guardrails: Protocol, resultado e pipeline encadeado."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol


class GuardAction(str, Enum):
    """Ação resultante de um guardrail.

    - `allow`: segue o fluxo normalmente.
    - `block`: impede o processamento (ex.: injeção de prompt, tópico proibido).
    - `redact`: permite, mas com conteúdo sanitizado (ex.: PII mascarada).
    - `notify`: permite, mas registra um aviso (ex.: baixo grounding).
    """

    allow = "allow"
    block = "block"
    redact = "redact"
    notify = "notify"


@dataclass(frozen=True)
class GuardrailResult:
    """Veredito de um guardrail sobre um texto."""

    allowed: bool = True
    action: GuardAction = GuardAction.allow
    guard: str = ""
    reason: str = ""
    redacted: str | None = None


class Guardrail(Protocol):
    """Um guardrail: inspeciona `text` e decide a ação."""

    name: str

    def check(self, text: str, **context) -> GuardrailResult: ...


@dataclass
class GuardrailPipeline:
    """Executa guardrails em série, parando no primeiro `block`."""

    guards: list[Guardrail] = field(default_factory=list)

    def run(self, text: str, **context) -> GuardrailResult:
        """Roda os guardrails; o primeiro `block` encerra e vence.

        `redact` acumula o texto sanitizado e segue; `notify` segue. O texto
        final retornado em `redacted` é o texto com todos os `redact` aplicados.
        """
        current = text
        for g in self.guards:
            res = g.check(current, **context)
            if res.action == GuardAction.block or not res.allowed:
                return res
            if res.action == GuardAction.redact and res.redacted is not None:
                current = res.redacted
        # sem block: devolve o texto (redact aplicado) como redacted
        if current != text:
            return GuardrailResult(
                allowed=True,
                action=GuardAction.redact,
                redacted=current,
            )
        return GuardrailResult(allowed=True, action=GuardAction.allow)

    def __iter__(self):
        return iter(self.guards)