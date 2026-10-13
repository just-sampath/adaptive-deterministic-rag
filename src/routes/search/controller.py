from __future__ import annotations

from routes.dependencies import Context, RequestIdentity
from routes.search.validator import SearchRequest, SearchResponse


async def search(
    body: SearchRequest,
    identity: RequestIdentity,
    context: Context,
) -> SearchResponse:
    return await context.query_handler.query(identity, body)
