from langchain_community.vectorstores import Chroma
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from src.rag.tools import rules_context_tool
from src.schemas.graph import State


class Rag:
    @staticmethod
    def ask(state: State):
        llm_with_tools = state.llm.bind_tools([rules_context_tool])
        return {"messages": [llm_with_tools.invoke(state.messages)]}

    @staticmethod
    def generate(state: State):
        recent_tool_messages = []
        for message in reversed(state.messages):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break
        tool_messages = recent_tool_messages[::-1]

        docs_content = "\n\n".join(doc.content for doc in tool_messages)
        system_message_content = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise. Only use the information from the provided context"
            "\n\n"
            f"{docs_content}"
        )

        conversation_messages = [
            message
            for message in state.messages
            if message.type in ("human", "system")
            or (message.type == "ai" and not message.tool_calls)
        ]
        prompt = [SystemMessage(system_message_content)] + conversation_messages
        res = state.llm.invoke(prompt)
        return {"messages": [res]}

    @staticmethod
    def build(vector_store: Chroma):
        tools = ToolNode([rules_context_tool]).with_config(
            {"configurable": {"vector_store": vector_store}}
        )

        graph = StateGraph(state_schema=State)
        graph.add_node(Rag.ask)
        graph.add_node(Rag.generate)
        graph.add_node(tools)

        graph.set_entry_point("ask")
        graph.add_edge("ask", "tools")
        graph.add_edge("tools", "generate")
        graph.add_edge("generate", END)

        memory = MemorySaver()
        compiled_graph = graph.compile(checkpointer=memory)

        # Draw and display the graph
        Rag.draw_graph(compiled_graph)

        return compiled_graph

    @staticmethod
    def draw_graph(compiled_graph):
        """Save the LangGraph visualization to file"""
        try:
            # Get the graph image as PNG bytes
            graph_image = compiled_graph.get_graph().draw_mermaid_png()

            import os

            with open("data/langgraph_visualization.png", "wb") as f:
                f.write(graph_image)
        except Exception:
            pass
