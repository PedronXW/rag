"""Refiners: pós-processamento da resposta do LLM antes de persistir.

- `Refiner` (Protocol) + `RefinerPipeline` (base).
- `DeduplicateRefiner`: remove linhas/parágrafos repetidos.
- `CitationRefiner`: anexa citações `[chunk N]`/`[doc N]` dos chunks usados.
- `PIIMaskRefiner`: mascara dado pessoal na resposta (reusa PIIGuard).

Gated por `REFINERS_ENABLED` (default off) — determinístico nos testes.
"""