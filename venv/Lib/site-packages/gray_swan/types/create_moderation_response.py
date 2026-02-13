# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import List, Optional

from .._models import BaseModel

__all__ = ["CreateModerationResponse", "Usage"]


class Usage(BaseModel):
    completion_tokens: int

    prompt_tokens: int

    total_tokens: int


class CreateModerationResponse(BaseModel):
    id: str

    created: int

    detected: float

    flagged: bool

    model: str

    passed: float

    categories: Optional[List[object]] = None

    rewrite: Optional[str] = None

    system_fingerprint: Optional[str] = None

    usage: Optional[Usage] = None
