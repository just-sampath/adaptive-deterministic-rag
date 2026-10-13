from __future__ import annotations

from handlers.search_strategies.contracts import StrategyPlan
from models.domain import RetrievalPlan, RetrievalStrategy
from utils.settings import RetrievalConfig


class HappyPathStrategy:
    def __init__(self, config: RetrievalConfig) -> None:
        self._plan = StrategyPlan(
            kind=RetrievalStrategy.HAPPY_PATH,
            retrieval=RetrievalPlan(
                dense_k=config.happy_dense_k,
                sparse_k=config.happy_sparse_k,
                candidate_k=config.happy_candidate_k,
            ),
        )

    @property
    def plan(self) -> StrategyPlan:
        return self._plan
