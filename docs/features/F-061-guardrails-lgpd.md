# F-061 — Guardrails, refiners e LGPD

Camada de segurança e conformidade do RAG: guardrails de entrada/saída,
refiners de resposta e features de LGPD (consentimento, DSR export/erasure).

## Guardrails
- `guardrails/base.py` — `Guardrail` (Protocol), `GuardrailResult`
  (`allow`/`block`/`redact`/`notify`), `GuardrailPipeline` (série, para no 1º block).
- `guardrails/input_guardrails.py` — `PromptInjectionGuard`, `PolicyGuard`
  (`BLOCKED_TOPICS`), `PIIGuard` (mascara/bloqueia CPF/e-mail/telefone).
- `guardrails/output_guardrails.py` — `EmptyAnswerGuard`, `HarmfulContentGuard`,
  `GroundingGuard` (heurística de grounding/fidelidade).
- **Input hook**: rotas RAG (`/messages`, `/agent`, `/multi-query`,
  `/agent/supervisor`, `/agent/hitl`) rodam o pipeline de entrada antes do LLM;
  se `block` → evento SSE `guardrail_blocked` (ou 422 nas rotas dict).
- **Output hook**: nó `generate` do graph aplica o pipeline de saída +
  refiners na resposta (gated por env).

## Refiners
- `refiners/base.py` — `Refiner` (Protocol), `RefinerPipeline`.
- `refiners/refiners.py` — `DeduplicateRefiner`, `CitationRefiner`
  (`[doc N]` das fontes), `PIIMaskRefiner`.

## LGPD
- `User.consent_granted`/`consent_at` (migration `540c2517a30f`).
- `POST /users` aceita `consent`; `PATCH /users/me/consent` concede/revoga.
- `GET /users/me/data` — DSR de exportação (perfil + chats + mensagens + arquivos).
- `DELETE /users/me` — DSR de apagamento (hard-delete de user + chats + messages +
  files + chunks + objetos no storage), satisfazendo `C-DATA-001`.
- Gate `REQUIRE_CONSENT=true`: rotas RAG exigem consentimento (SSE `consent_required`
  / 403), sem chamar o LLM.

## Critérios de aceite
- **AC-001** — Guardrails de entrada bloqueiam injeção de prompt, tópico proibido
  e PII (redact/block); pipeline para no 1º block.
- **AC-002** — Guardrails de saída e refiners aplicam-se na resposta do graph
  (gated por env; default off, determinístico nos testes).
- **AC-003** — Consentimento: registro com consent, PATCH de concessão/revogação,
  e `REQUIRE_CONSENT` bloqueia o RAG sem consentimento.
- **AC-004** — DSR: `GET /users/me/data` exporta; `DELETE /users/me` remove os
  dados (hard delete).
- **AC-005** — Regressão: testes existentes continuam passando.

## Env
| Variável | Default | Descrição |
|---|---|---|
| `GUARDRAILS_ENABLED` | `false` | Habilita guardrails de entrada/saída |
| `REFINERS_ENABLED` | `false` | Habilita refiners de resposta |
| `BLOCKED_TOPICS` | — | Tópicos proibidos (guardrail de política) |
| `REQUIRE_CONSENT` | `false` | Exige consentimento LGPD antes do RAG |