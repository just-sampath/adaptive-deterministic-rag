from __future__ import annotations

from fastapi import APIRouter

from routes.search.controller import search
from routes.search.validator import SearchResponse

router = APIRouter(prefix="/v1/rag", tags=["rag"])
router.add_api_route(
    "/query",
    search,
    methods=["POST"],
    response_model=SearchResponse,
)
