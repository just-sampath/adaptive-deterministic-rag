"""Rules every deterministic retrieval try must declare."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from models.domain import RetrievalPlan, RetrievalStrategy


@dataclass(frozen=True, slots=True)
class StrategyPlan:
    kind: RetrievalStrategy
    retrieval: RetrievalPlan
    reuse_previous_context: bool = False
    isolate_conversation: bool = False


class SearchStrategy(Protocol):
    @property
    def plan(self) -> StrategyPlan: ...
