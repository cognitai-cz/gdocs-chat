from pprint import pprint

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()


def main():
    model = init_chat_model("gpt-4o-mini", model_provider="openai")

    messages = [
        SystemMessage("Translate the following from English into Italian"),
        HumanMessage("hi!"),
    ]

    pprint(model.invoke(messages))



main()
