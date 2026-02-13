# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import List, Optional
from typing_extensions import Literal

from ..._models import BaseModel
from .chat_completion_delta_tool_call import ChatCompletionDeltaToolCall

__all__ = ["ChatCompletionDelta"]


class ChatCompletionDelta(BaseModel):
    content: Optional[str] = None

    role: Optional[Literal["system", "user", "assistant", "tool"]] = None

    tool_calls: Optional[List[ChatCompletionDeltaToolCall]] = None
