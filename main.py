from dotenv import load_dotenv
from pprint import pprint
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph

from src.promts import ASK_PROMPT, GDOCS_PARSE_PROMPT
from src.schemas import ExercisesList, State


load_dotenv()

model = init_chat_model("gpt-4o-mini", model_provider="openai")


def parse_exercises(state: State):
    structured_llm = model.with_structured_output(ExercisesList)
    prompt = PromptTemplate.from_template(GDOCS_PARSE_PROMPT).invoke(
        {
            "raw_string": state.raw_string,
        }
    )
    exercises: ExercisesList = structured_llm.invoke(prompt)
    logs = []
    for exercise in exercises.exercises:
        dump = exercise.model_dump()
        logs.append(dump)
    pprint(logs)
    return {"logs": logs}


def answer(state: State):
    prompt = PromptTemplate.from_template(ASK_PROMPT).invoke(
        {"logs": state.logs, "question": state.question}
    )
    res = model.invoke(prompt)
    return {"answer": res}


def init_graph():
    graph = StateGraph(state_schema=State)
    graph.add_sequence([parse_exercises, answer])

    graph.add_edge(START, "parse_exercises")

    # Add memory
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


def parse_file() -> str:
    raw_string = ""
    with open("table.txt") as f:
        lines = [line for line in f.readlines() if line != "\n"]
        raw_string = "\n".join(lines)
    return raw_string


def main():
    # TODO: Add memmory, try to parse more exercises
    config = {"configurable": {"thread_id": "abc123"}}
    graph = init_graph()

    input_message = HumanMessage("What is my best benchrpess (absolute load)?")
    raw_string = parse_file()

    res = graph.invoke({"raw_string": raw_string, "question": input_message}, config)
    res["answer"].pretty_print()


main()
