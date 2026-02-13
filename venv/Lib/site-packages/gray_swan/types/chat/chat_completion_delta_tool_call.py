# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional
from typing_extensions import Literal

from ..._models import BaseModel
from .chat_completion_delta_tool_call_function import ChatCompletionDeltaToolCallFunction

__all__ = ["ChatCompletionDeltaToolCall"]


class ChatCompletionDeltaToolCall(BaseModel):
    index: int

    id: Optional[str] = None

    function: Optional[ChatCompletionDeltaToolCallFunction] = None

    type: Optional[Literal["function"]] = None
