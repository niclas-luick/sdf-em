# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import List, Optional
from typing_extensions import Literal

from ..._models import BaseModel
from .chat_completion_tool_call import ChatCompletionToolCall

__all__ = ["ChatCompletionMessage"]


class ChatCompletionMessage(BaseModel):
    role: Literal["assistant"]

    content: Optional[str] = None

    tool_calls: Optional[List[ChatCompletionToolCall]] = None
