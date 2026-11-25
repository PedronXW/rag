"""create_react_agent (pre-built) para o RAG (H-004).

Comparação com o StateGraph manual do G-002: `build_react_agent` usa o
`create_react_agent` do langgraph-prebuilt com as tools do chat, multi-tenant.
"""

from langgraph.prebuilt import create_react_agent

from einstein_test.services.rag.agent import build_tools


def build_react_agent(llm, retriever, embedder, *, chat_id: int, checkpointer=None):
    """Cria um agente react via `create_react_agent` (pre-built).

    O LLM precisa expor `bind_tools`. As tools são do chat (multi-tenant).
    """
    tools = build_tools(retriever, embedder, chat_id)
    return create_react_agent(
        model=llm,
        tools=tools,
        checkpointer=checkpointer,
    )


def react_config(chat_id: int) -> dict:
    return {"configurable": {"thread_id": str(chat_id)}}


__all__ = ["build_react_agent", "react_config"]