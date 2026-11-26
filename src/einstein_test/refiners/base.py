"""Base de refiners: Protocol e pipeline encadeado."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


class Refiner(Protocol):
    """Um refiner: transforma `text` usando `context` (chunks, etc.)."""

    def refine(self, text: str, **context) -> str: ...


@dataclass
class RefinerPipeline:
    """Aplica refiners em série; cada saída alimenta o próximo."""

    refiners: list[Refiner] = field(default_factory=list)

    def run(self, text: str, **context) -> str:
        current = text
        for r in self.refiners:
            current = r.refine(current, **context)
        return current