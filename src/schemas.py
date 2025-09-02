from decimal import Decimal
from typing import List
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


class Exercise(BaseModel):
    name: str
    repetitions: int
    prescribed_intensity: int = Field(description="Unit is Kg")
    real_intensity: int = Field(
        description="If only presribed inensity is present, defaults to that one, unit is Kg"
    )
    rpe: Decimal = Field(description="Format is 1.5, only use 0.5 increments")
    note: str | None


class ExercisesList(BaseModel):
    exercises: List[Exercise] = Field(
        description="A list of all exercises found in the text."
    )


class State(BaseModel):
    question: BaseMessage | None = None
    raw_string: str | None = None
    logs: list[dict] | None = None
    answer: str | None = None
