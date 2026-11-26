"""Guardrails: validação de entrada e saída do RAG (F-061).

- `Guardrail` (Protocol) + `GuardrailResult` + `GuardrailPipeline` (base).
- `input_guardrails`: inspeção da pergunta do usuário (injeção de prompt, PII,
  política) antes de chamar o LLM.
- `output_guardrails`: validação da resposta (grounding/faithfulness, conteúdo
  nocivo) antes de persistir.

Tudo é gated por `GUARDRAILS_ENABLED` (default off) — determinístico nos testes.
"""