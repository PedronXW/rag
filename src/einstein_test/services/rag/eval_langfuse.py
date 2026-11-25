"""Langfuse como plataforma de avaliação do RAG (J-004).

- `register_trace_score`: registra um score (`create_score`) no trace de uma
  execução avaliada (avaliação contínua por trace).
- `run_batched_eval`: avalia um dataset offline via `run_batched_evaluation`.
- `langfuse_eval_enabled`: gate por env (desabilitado nos testes).

O client do Langfuse é injetável (stub nos testes; `_get_client` em produção).
"""

from typing import Any, Callable

from einstein_test.services.rag.callbacks import _get_client, langfuse_enabled


def langfuse_eval_enabled() -> bool:
    return langfuse_enabled()


def _client():
    return _get_client()


def register_trace_score(
    *,
    client=None,
    trace_id: str | None,
    name: str,
    value: float,
    comment: str | None = None,
) -> bool:
    """Registra um score no trace do Langfuse (create_score).

    Retorna True se registrou; False se desabilitado (ou trace sem id).
    """
    if not langfuse_eval_enabled():
        return False
    if not trace_id:
        return False
    c = client or _client()
    c.create_score(
        name=name,
        value=value,
        trace_id=trace_id,
        comment=comment,
        data_type="NUMERIC",
    )
    return True


def run_batched_eval(
    *,
    client=None,
    mapper: Callable[[dict], Any],
    evaluators: list[Callable[[Any], dict]],
    max_items: int | None = None,
) -> Any:
    """Avalia um dataset offline via run_batched_evaluation do Langfuse."""
    if not langfuse_eval_enabled():
        return None
    c = client or _client()
    return c.run_batched_evaluation(
        mapper=mapper,
        evaluators=evaluators,
        max_items=max_items,
    )


__all__ = ["register_trace_score", "run_batched_eval", "langfuse_eval_enabled"]