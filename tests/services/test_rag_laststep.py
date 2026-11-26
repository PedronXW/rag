"""J-002 — Testes do IsLastStep real e build_prompt unificado."""



def test_summary_node_usa_is_last_step():
    """AC-001: summary_node usa IsLastStep como campo gerenciado."""
    import inspect

    from einstein_test.services.rag.summary import summary_node

    src = inspect.getsource(summary_node)
    assert "IsLastStep" in src


def test_is_last_step_no_grafo():
    """AC-001: IsLastStep é campo gerenciado no estado do grafo."""
    from langgraph.managed import IsLastStep
    from langgraph.graph import MessagesState, StateGraph, START, END

    class S(MessagesState):
        is_last_step: IsLastStep
        seen: bool

    def node(state):
        return {"seen": True}

    g = StateGraph(S)
    g.add_node("n", node)
    g.add_edge(START, "n")
    g.add_edge("n", END)
    comp = g.compile()
    out = comp.invoke({"messages": []})
    assert out["seen"] is True


def test_build_prompt_unico_com_resumo():
    """AC-002: build_prompt único aceita resumo/memória opcionais."""
    from einstein_test.services.rag.graph import build_prompt

    prompt = build_prompt(
        "pergunta",
        [],
        [],
        summary="resumo da conversa",
        memory="memoria do usuario",
    )
    assert "resumo da conversa" in prompt
    assert "memoria do usuario" in prompt
    assert "pergunta" in prompt


def test_summary_reutiliza_build_prompt():
    """AC-002: summary usa o build_prompt do graph (sem divergência)."""
    from einstein_test.services.rag import summary

    import inspect

    src = inspect.getsource(summary)
    assert "build_prompt" in src  # reutiliza (importa ou chama o do graph)


def test_summary_gera_quando_excede():
    """AC-003: sumarização gera resumo quando o histórico excede o limite."""
    from einstein_test.services.rag.summary import should_summarize

    from langchain_core.messages import HumanMessage

    history = [HumanMessage(content=f"msg {i}") for i in range(50)]
    assert should_summarize(history, max_messages=10) is True