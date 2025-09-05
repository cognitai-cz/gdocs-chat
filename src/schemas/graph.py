from typing import Annotated

from langchain_core.language_models import BaseChatModel
from langgraph.graph import add_messages
from pydantic import BaseModel


class State(BaseModel):
    messages: Annotated[list, add_messages]

    llm: BaseChatModel
