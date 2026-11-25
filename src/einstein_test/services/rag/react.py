"""Agente via create_agent (I-004).

`build_agent` usa `langchain.agents.create_agent` (novo, não deprecado) para
criar um agente react com as tools do chat. Requer um `BaseChatModel` real
(com `bind_tools`). `build_react_agent` é mantido apenas como referência ao
`create_react_agent` legado (deprecado no LangGraph 1.x).
"""

from einstein_test.services.rag.agent import build_tools


def build_agent(llm, retriever, embedder, *, chat_id: int, checkpointer=None):
    """Cria um agente react via `langchain.agents.create_agent`.

    O LLM deve ser um `BaseChatModel` do langchain (com `bind_tools`).
    """
    from langchain.agents import create_agent

    tools = build_tools(retriever, embedder, chat_id)
    return create_agent(
        model=llm,
        tools=tools,
        checkpointer=checkpointer,
        name=f"rag-agent-{chat_id}",
    )


def build_react_agent(llm, retriever, embedder, *, chat_id: int, checkpointer=None):
    """(Legado/referência) Agente via `create_react_agent` — deprecado."""
    from langgraph.prebuilt import create_react_agent

    tools = build_tools(retriever, embedder, chat_id)
    return create_react_agent(model=llm, tools=tools, checkpointer=checkpointer)


def react_config(chat_id: int) -> dict:
    return {"configurable": {"thread_id": f"react:{chat_id}"}}


__all__ = ["build_agent", "build_react_agent", "react_config"]