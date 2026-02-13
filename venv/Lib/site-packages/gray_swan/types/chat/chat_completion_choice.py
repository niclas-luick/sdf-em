# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional
from typing_extensions import Literal

from ..._models import BaseModel
from .chat_completion_message import ChatCompletionMessage

__all__ = ["ChatCompletionChoice"]


class ChatCompletionChoice(BaseModel):
    finish_reason: Literal["stop", "length", "content_filter", "tool_calls"]

    index: int

    message: ChatCompletionMessage

    logprobs: Optional[str] = None
