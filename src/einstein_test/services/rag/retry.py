"""Retry de nós do graph RAG (H-005).

`retry_node` re-executa um nó com backoff em falhas transientes (rede/storage/
timeout), absorvendo a falha se a re-execução tiver sucesso.
"""

import time
from typing import Callable


def _is_transient(exc: BaseException) -> bool:
    return isinstance(exc, (TimeoutError, ConnectionError, OSError))


def retry_node(
    node: Callable,
    *,
    retries: int = 3,
    backoff: float = 0.05,
) -> dict:
    """Executa `node` com retry + backoff em falhas transientes.

    Retorna o resultado do nó; relança a última exceção se os retries esgotarem
    (ou se a exceção não for transiente).
    """
    if retries < 1:
        raise ValueError("retries deve ser >= 1")
    for attempt in range(retries):
        try:
            return node()
        except Exception as exc:  # noqa: BLE001
            if not _is_transient(exc) or attempt == retries - 1:
                raise
            time.sleep(backoff * (2**attempt))
    raise RuntimeError("retry_node terminou sem resultado")


__all__ = ["retry_node"]