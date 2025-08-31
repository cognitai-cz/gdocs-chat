from typing import TypedDict, Annotated, Sequence

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, trim_messages, \
    SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START
from langgraph.graph import StateGraph, MessagesState, add_messages
from langgraph.graph.state import CompiledStateGraph

load_dotenv()


model = init_chat_model("gpt-4o-mini", model_provider="openai")
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

trimmer = trim_messages(
    max_tokens=1000,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str



def call_model(state: State):
    print(f"Messages before trimming: {len(state['messages'])}")
    trimmed_messages = trimmer.invoke(state["messages"])
    print(f"Messages after trimming: {len(trimmed_messages)}")
    print("Remaining messages:")
    for msg in trimmed_messages:
        print(f"  {type(msg).__name__}: {msg.content}")
    prompt = prompt_template.invoke(
        {"messages": trimmed_messages, "language": state["language"]}
    )
    response = model.invoke(prompt)
    return {"messages": [response]}


def init_app() -> CompiledStateGraph:
    workflow = StateGraph(state_schema=State)

    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)

    # Add memory
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


def ask(question: str, app, language: str):

    config = {"configurable": {"thread_id": "abc123"}}

    input_msg = [HumanMessage(question)]
    for chunk, metadata in app.stream(
            {"messages": input_msg, "language": language},
            config,
            stream_mode="messages",
    ):
        if isinstance(chunk, AIMessage):  # Filter to just model responses
            print(chunk.content, end="|")


def main():
    language = input("choose language: ")
    app = init_app()
    while True:
        ask(input(), app, language)


main()
