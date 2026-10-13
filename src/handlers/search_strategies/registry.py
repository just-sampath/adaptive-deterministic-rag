"""Fixed registry for the four bounded RAG tries."""

from __future__ import annotations

from collections.abc import Iterable

from handlers.search_strategies.context_isolation import ContextIsolationStrategy
from handlers.search_strategies.contracts import SearchStrategy, StrategyPlan
from handlers.search_strategies.happy_path import HappyPathStrategy
from handlers.search_strategies.query_recovery import QueryRecoveryStrategy
from handlers.search_strategies.recall_expansion import RecallExpansionStrategy
from models.domain import RetrievalStrategy
from utils.settings import RetrievalConfig


class SearchStrategyRegistry:
    def __init__(self, strategies: Iterable[SearchStrategy]) -> None:
        plans = tuple(strategy.plan for strategy in strategies)
        kinds = [plan.kind for plan in plans]
        if len(set(kinds)) != len(kinds):
            raise ValueError("Each deterministic retrieval strategy must be registered once")
        if set(kinds) != set(RetrievalStrategy):
            raise ValueError("Every deterministic retrieval strategy must be registered")
        self._plans = {plan.kind: plan for plan in plans}

    def get(self, strategy: RetrievalStrategy) -> StrategyPlan:
        return self._plans[strategy]


def build_strategy_registry(config: RetrievalConfig) -> SearchStrategyRegistry:
    return SearchStrategyRegistry(
        (
            HappyPathStrategy(config),
            RecallExpansionStrategy(config),
            QueryRecoveryStrategy(config),
            ContextIsolationStrategy(),
        )
    )
