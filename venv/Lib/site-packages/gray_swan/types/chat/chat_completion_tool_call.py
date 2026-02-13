# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing_extensions import Literal

from ..._models import BaseModel
from .chat_completion_tool_call_function import ChatCompletionToolCallFunction

__all__ = ["ChatCompletionToolCall"]


class ChatCompletionToolCall(BaseModel):
    id: str

    function: ChatCompletionToolCallFunction

    type: Literal["function"]
