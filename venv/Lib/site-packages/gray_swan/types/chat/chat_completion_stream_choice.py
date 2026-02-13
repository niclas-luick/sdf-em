# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional
from typing_extensions import Literal

from ..._models import BaseModel
from .chat_completion_delta import ChatCompletionDelta

__all__ = ["ChatCompletionStreamChoice"]


class ChatCompletionStreamChoice(BaseModel):
    delta: ChatCompletionDelta

    index: int

    finish_reason: Optional[Literal["stop", "length", "content_filter", "tool_calls"]] = None
