from __future__ import annotations

from handlers.search_strategies.contracts import StrategyPlan
from models.domain import RetrievalPlan, RetrievalStrategy
from utils.settings import RetrievalConfig


class QueryRecoveryStrategy:
    def __init__(self, config: RetrievalConfig) -> None:
        self._plan = StrategyPlan(
            kind=RetrievalStrategy.QUERY_RECOVERY,
            retrieval=RetrievalPlan(
                dense_k=config.happy_dense_k,
                sparse_k=config.happy_sparse_k,
                candidate_k=config.happy_candidate_k,
                recover_queries=True,
            ),
        )

    @property
    def plan(self) -> StrategyPlan:
        return self._plan
