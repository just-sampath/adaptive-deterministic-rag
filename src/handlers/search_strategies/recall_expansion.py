from __future__ import annotations

from handlers.search_strategies.contracts import StrategyPlan
from models.domain import RetrievalPlan, RetrievalStrategy
from utils.settings import RetrievalConfig


class RecallExpansionStrategy:
    def __init__(self, config: RetrievalConfig) -> None:
        self._plan = StrategyPlan(
            kind=RetrievalStrategy.RECALL_EXPANSION,
            retrieval=RetrievalPlan(
                dense_k=config.recall_dense_k,
                sparse_k=config.recall_sparse_k,
                candidate_k=config.recall_candidate_k,
            ),
        )

    @property
    def plan(self) -> StrategyPlan:
        return self._plan
