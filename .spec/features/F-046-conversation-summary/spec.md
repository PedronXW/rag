# F-046 — Sumarização de conversa no RAG

Quando o histórico da conversa excede o limite (mesmo após trim), um nó gera um
resumo dos turnos antigos (`SummaryMessage`) e o usa como contexto inicial no
prompt — em vez de descartar informação. Usa `IsLastStep` para gerar o resumo
na última etapa do graph e o insere antes das mensagens recentes.

## Histórias
- U-001 — Como usuário, quero que conversas longas preservem o contexto via
  resumo, **para** que o RAG não perca informação nem estoure o limite.
- U-002 — Como dev, quero um nó que gere `SummaryMessage` dos turnos antigos
  quando o histórico excede o limite, **para** gestão de contexto em escala.
- U-003 — Como dev, quero usar `IsLastStep` para rodar a sumarização só na
  etapa final, **para** não gastar LLM em cada super-step.

## Critérios
- AC-001 — [U-002] Existe um nó que gera um `SummaryMessage` dos turnos antigos
  quando o histórico excede o limite.
- AC-002 — [U-001] O resumo entra no prompt como contexto inicial (antes das
  mensagens recentes).
- AC-003 — [U-003] A geração do resumo usa `IsLastStep` (roda só na última
  etapa).
- AC-004 — [U-001] Quando o histórico não excede o limite, nenhum resumo é
  gerado (sem custo extra de LLM).
- AC-005 — [U-002] O resumo combina com `trim_history` (resumo + últimas N
  mensagens).

## Suposições (status: aberto|confirmada|rejeitada)
- A-001 — (Resumo quando excede limite, via IsLastStep) — confirmada
- A-002 — (Resumo como contexto inicial no prompt) — confirmada
- A-003 — (Testes com FakeChatLLM determinístico) — confirmada

## Perguntas em aberto
- (nenhuma)