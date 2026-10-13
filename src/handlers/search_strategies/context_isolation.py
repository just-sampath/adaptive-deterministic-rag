from __future__ import annotations

from handlers.search_strategies.contracts import StrategyPlan
from models.domain import RetrievalPlan, RetrievalStrategy


class ContextIsolationStrategy:
    plan = StrategyPlan(
        kind=RetrievalStrategy.CONTEXT_ISOLATION,
        retrieval=RetrievalPlan(dense_k=0, sparse_k=0, candidate_k=0),
        reuse_previous_context=True,
        isolate_conversation=True,
    )
