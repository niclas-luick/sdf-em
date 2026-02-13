# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Dict, Union, Iterable, Optional
from typing_extensions import Literal, Required, TypeAlias, TypedDict

__all__ = [
    "CompletionCreateParams",
    "Message",
    "ToolChoice",
    "ToolChoiceToolChoice",
    "ToolChoiceToolChoiceFunction",
    "Tool",
    "ToolFunction",
]


class CompletionCreateParams(TypedDict, total=False):
    messages: Required[Iterable[Message]]

    model: Required[Literal["cygnet-v0.2"]]

    frequency_penalty: Optional[float]

    max_tokens: Optional[int]

    moderate_before: bool

    n: Optional[int]

    stream: Optional[bool]

    temperature: Optional[float]

    tool_choice: Optional[ToolChoice]

    tools: Optional[Iterable[Tool]]

    top_p: Optional[float]


class Message(TypedDict, total=False):
    content: Required[Optional[str]]

    role: Required[Literal["system", "user", "assistant", "data"]]


class ToolChoiceToolChoiceFunction(TypedDict, total=False):
    name: Required[str]


class ToolChoiceToolChoice(TypedDict, total=False):
    function: Required[ToolChoiceToolChoiceFunction]

    type: Required[str]


ToolChoice: TypeAlias = Union[Literal["none", "auto"], ToolChoiceToolChoice]


class ToolFunction(TypedDict, total=False):
    description: Required[str]

    name: Required[str]

    parameters: Required[Dict[str, object]]


class Tool(TypedDict, total=False):
    function: Required[ToolFunction]

    type: Required[str]
