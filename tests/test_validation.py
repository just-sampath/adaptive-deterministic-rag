from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from models.domain import QueryRequest


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"message": "   "}, "non-whitespace"),
        ({"message": "question", "conversation_id": ""}, "at least 1 character"),
        ({"message": "question", "filters": {"$unsafe": "value"}}, "string_pattern"),
        ({"message": "question", "filters": {"score": math.nan}}, "finite_number"),
    ],
)
def test_query_boundary_rejects_unsafe_values(
    values: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        QueryRequest.model_validate(values)
