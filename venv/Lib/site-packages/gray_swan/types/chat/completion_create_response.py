# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Union
from typing_extensions import TypeAlias

from .create_chat_completion_response import CreateChatCompletionResponse
from .create_chat_completion_stream_response import CreateChatCompletionStreamResponse

__all__ = ["CompletionCreateResponse"]

CompletionCreateResponse: TypeAlias = Union[CreateChatCompletionResponse, CreateChatCompletionStreamResponse]
