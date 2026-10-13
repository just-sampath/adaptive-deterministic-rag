from __future__ import annotations

from fastapi import APIRouter

from routes.traces.controller import get_trace
from routes.traces.validator import TraceResponse

router = APIRouter(prefix="/v1/rag/requests", tags=["debug"])
router.add_api_route(
    "/{request_id}/trace",
    get_trace,
    methods=["GET"],
    response_model=TraceResponse,
)
