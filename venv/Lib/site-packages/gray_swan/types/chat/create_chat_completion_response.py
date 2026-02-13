# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import List, Optional
from typing_extensions import Literal

from ..._models import BaseModel
from .chat_completion_choice import ChatCompletionChoice

__all__ = ["CreateChatCompletionResponse", "Usage"]


class Usage(BaseModel):
    completion_tokens: int

    prompt_tokens: int

    total_tokens: int


class CreateChatCompletionResponse(BaseModel):
    id: str

    choices: List[ChatCompletionChoice]

    created: int

    model: str

    object: Literal["chat.completion"]

    system_fingerprint: Optional[str] = None

    usage: Optional[Usage] = None
