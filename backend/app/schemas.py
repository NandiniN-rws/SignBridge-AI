from pydantic import BaseModel, Field
from typing import Literal


class ConversationIn(BaseModel):
    title: str = Field(
        default="Hospital communication",
        max_length=200
    )


class MessageIn(BaseModel):
    role: Literal["patient", "doctor", "system"]

    content: str = Field(
        min_length=1,
        max_length=5000
    )

    confirmed: bool = False
    source: str = "typed"


class TextIn(BaseModel):
    text: str = Field(
        min_length=1,
        max_length=3000
    )

    context: list[str] = Field(default_factory=list)


class PredictIn(BaseModel):
    features: list[list[float]]


class MedicalImageIn(BaseModel):
    image_base64: str = Field(
        min_length=20,
        max_length=14_000_000
    )


class DoctorResponseIn(TextIn):
    language: Literal[
        "English",
        "Kannada",
        "Hindi"
    ] = "English"