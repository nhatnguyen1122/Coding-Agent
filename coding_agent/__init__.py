"""Hierarchical MCTS framework for program discovery."""

from coding_agent._version import __version__
from coding_agent.api import (
    DiscoveryResult,
    discover_solution,
    run_discovery,
)
from coding_agent.runner import Runner

__all__ = [
    "Runner",
    "__version__",
    "run_discovery",
    "discover_solution",
    "DiscoveryResult",
]
