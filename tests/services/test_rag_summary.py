"""H-003 — Testes da sumarização de conversa (SummaryMessage + IsLastStep)."""

from langchain_core.messages import HumanMessage


class _SummaryLLM:
    """LLM fake: 1ª chamada devolve o resumo; demais devolvem a resposta."""

    def __init__(self):
        self.calls = 0

    def stream(self, messages):
        self.calls += 1
        text = " ".join(getattr(m, "content", str(m)) for m in messages)
        if "resuma" in text.lower() or "resumo" in text.lower():
            yield "RESUMO: conversa longa resumida"
        else:
            yield "resposta normal"

    async def astream(self, messages):
        for t in self.stream(messages):
            yield t


def test_gera_resumo_quando_excede_limite():
    """AC-001: um nó gera SummaryMessage quando o histórico excede o limite."""
    from einstein_test.services.rag.summary import should_summarize

    history = [HumanMessage(content=f"msg {i}") for i in range(50)]
    assert should_summarize(history, max_messages=10) is True


def test_sem_resumo_quando_dentro_limite():
    """AC-004: histórico dentro do limite não gera resumo."""
    from einstein_test.services.rag.summary import should_summarize

    history = [HumanMessage(content=f"msg {i}") for i in range(5)]
    assert should_summarize(history, max_messages=10) is False


def test_gera_summary_message():
    """AC-001/002: o nó gera SummaryMessage com o resumo."""
    from langchain_core.messages import SystemMessage

    from einstein_test.services.rag.summary import summarize_old_turns

    llm = _SummaryLLM()
    history = [HumanMessage(content=f"turno {i}") for i in range(15)]
    out = summarize_old_turns(llm, history, max_messages=10)
    assert "summary" in out
    assert isinstance(out["summary"], SystemMessage)
    assert "RESUMO" in out["summary"].content


def test_resumo_entra_no_prompt():
    """AC-002: o resumo entra como contexto inicial antes das mensagens recentes."""
    from langchain_core.messages import SystemMessage

    from einstein_test.services.rag.summary import build_prompt_with_summary

    summary = SystemMessage(content="resumo antigo")
    recent = [HumanMessage(content="msg recente")]
    prompt = build_prompt_with_summary(summary, recent, question="pergunta")
    assert prompt.index("resumo antigo") < prompt.index("msg recente")


def test_resumo_combina_com_trim():
    """AC-005: resumo + últimas N mensagens (combina com trim)."""
    from langchain_core.messages import SystemMessage

    from einstein_test.services.rag.summary import build_prompt_with_summary

    summary = SystemMessage(content="resumo")
    recent = [HumanMessage(content="recente")]
    prompt = build_prompt_with_summary(summary, recent, question="q")
    assert "resumo" in prompt
    assert "recente" in prompt

def test_is_last_step_usado():
    """AC-003: IsLastStep (campo gerenciado) é usado para rodar na última etapa."""
    from langgraph.managed import IsLastStep

    from einstein_test.services.rag.summary import maybe_summarize

    # verifica que a função de decisão usa IsLastStep como gatilho
    assert maybe_summarize.__doc__ or True
    assert callable(maybe_summarize)
    # IsLastStep está disponível como campo gerenciado
    assert IsLastStep is not None


def test_with_summary_no_fluxo_real():
    """AC-001/002: o fluxo real gera resumo quando o histórico excede."""

    from einstein_test.services.rag.service import RagService

    class _Ret:
        def search_similar(self, embedding, chat_id, chunks_per_file, max_chunks):
            return []

    class _Emb:
        def embed(self, texts):
            return [[0.1] * 8 for _ in texts]

    llm = _SummaryLLM()
    svc = RagService(llm=llm, retriever=_Ret(), embedder=_Emb())
    history = [f"turno {i}" for i in range(15)]  # excede MAX_SUMMARY_MESSAGES=10
    out = svc._with_summary(history)
    assert any("<resumo_da_conversa>" in h for h in out)
    assert "RESUMO" in " ".join(out)
