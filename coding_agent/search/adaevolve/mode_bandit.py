"""Lightweight UCB selector for prompt/search modes inside an island."""

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ModeStats:
    visits: int = 0
    value_sum: float = 0.0
    best_reward: float = float("-inf")
    discounted_visits: float = 0.0
    discounted_value_sum: float = 0.0
    last_discount_iteration: Optional[int] = None

    def discounted_projection(
        self, reward_discount: float, iteration: Optional[int]
    ) -> tuple[float, float]:
        if (
            iteration is None
            or self.last_discount_iteration is None
            or iteration <= self.last_discount_iteration
        ):
            return self.discounted_visits, self.discounted_value_sum

        factor = reward_discount ** (iteration - self.last_discount_iteration)
        return self.discounted_visits * factor, self.discounted_value_sum * factor

    def apply_reward(self, reward: float, reward_discount: float, iteration: Optional[int]) -> None:
        discounted_visits, discounted_value_sum = self.discounted_projection(
            reward_discount, iteration
        )
        self.visits += 1
        self.value_sum += reward
        self.best_reward = max(self.best_reward, reward)
        self.discounted_visits = discounted_visits + 1.0
        self.discounted_value_sum = discounted_value_sum + reward
        if iteration is not None:
            self.last_discount_iteration = iteration

    def mean(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0

    def discounted_mean(self, reward_discount: float, iteration: Optional[int]) -> float:
        visits, value_sum = self.discounted_projection(reward_discount, iteration)
        return value_sum / visits if visits > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "visits": self.visits,
            "value_sum": self.value_sum,
            "best_reward": self.best_reward,
            "discounted_visits": self.discounted_visits,
            "discounted_value_sum": self.discounted_value_sum,
            "last_discount_iteration": self.last_discount_iteration,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModeStats":
        return cls(
            visits=int(data.get("visits", 0)),
            value_sum=float(data.get("value_sum", 0.0)),
            best_reward=float(data.get("best_reward", float("-inf"))),
            discounted_visits=float(data.get("discounted_visits", 0.0)),
            discounted_value_sum=float(data.get("discounted_value_sum", 0.0)),
            last_discount_iteration=data.get("last_discount_iteration"),
        )


class SearchModeBandit:
    """Per-island UCB selector for search modes.

    The outer MCTS still chooses the island and the inner MCTS still chooses
    the parent. This selector only chooses which prompt/search guidance mode
    conditions that already-selected island/parent pair.
    """

    def __init__(
        self,
        modes: List[str],
        *,
        exploration_constant: float = 1.0,
        policy: str = "uct",
        reward_discount: float = 0.99,
        random_seed: Optional[int] = None,
    ) -> None:
        if not modes:
            raise ValueError("SearchModeBandit requires at least one mode")
        normalized_policy = policy.lower()
        if normalized_policy not in {"uct", "discounted_uct"}:
            raise ValueError(
                f"Unsupported mode policy '{policy}'. Expected 'uct' or 'discounted_uct'."
            )

        self.modes = list(dict.fromkeys(modes))
        self.exploration_constant = exploration_constant
        self.policy = normalized_policy
        self.reward_discount = reward_discount
        self._rng = random.Random(random_seed)
        self._stats: Dict[int, Dict[str, ModeStats]] = {}

    def ensure_island(self, island_idx: int) -> None:
        if island_idx not in self._stats:
            self._stats[island_idx] = {mode: ModeStats() for mode in self.modes}
        else:
            for mode in self.modes:
                self._stats[island_idx].setdefault(mode, ModeStats())

    def select(self, island_idx: int, iteration: Optional[int] = None) -> Dict[str, Any]:
        self.ensure_island(island_idx)
        stats = self._stats[island_idx]
        unvisited = [mode for mode in self.modes if stats[mode].visits == 0]
        if unvisited:
            mode = self._rng.choice(unvisited)
            return {"island_idx": island_idx, "mode": mode, "score": float("inf")}

        if self.policy == "discounted_uct":
            total_visits = sum(
                max(stats[mode].discounted_projection(self.reward_discount, iteration)[0], 0.0)
                for mode in self.modes
            )
        else:
            total_visits = sum(stats[mode].visits for mode in self.modes)
        total_visits = max(total_visits, 1.0)

        best_mode = self.modes[0]
        best_score = float("-inf")
        for mode in self.modes:
            mode_stats = stats[mode]
            if self.policy == "discounted_uct":
                visits, _ = mode_stats.discounted_projection(self.reward_discount, iteration)
                visits = max(visits, 1e-9)
                exploit = mode_stats.discounted_mean(self.reward_discount, iteration)
            else:
                visits = max(float(mode_stats.visits), 1.0)
                exploit = mode_stats.mean()

            explore = self.exploration_constant * math.sqrt(math.log(total_visits + 1.0) / visits)
            score = exploit + explore
            if score > best_score:
                best_mode = mode
                best_score = score

        return {"island_idx": island_idx, "mode": best_mode, "score": best_score}

    def update(
        self, island_idx: int, mode: str, reward: float, iteration: Optional[int] = None
    ) -> None:
        self.ensure_island(island_idx)
        if mode not in self._stats[island_idx]:
            raise ValueError(f"Unknown search mode '{mode}'")
        self._stats[island_idx][mode].apply_reward(reward, self.reward_discount, iteration)

    def island_summary(self, island_idx: int, iteration: Optional[int] = None) -> Dict[str, Any]:
        self.ensure_island(island_idx)
        mode_items = {}
        for mode, stats in self._stats[island_idx].items():
            discounted_visits, discounted_value_sum = stats.discounted_projection(
                self.reward_discount, iteration
            )
            mode_items[mode] = {
                "visits": stats.visits,
                "mean_reward": stats.mean(),
                "best_reward": None if math.isinf(stats.best_reward) else stats.best_reward,
                "discounted_visits": discounted_visits,
                "discounted_mean_reward": (
                    discounted_value_sum / discounted_visits if discounted_visits > 0 else 0.0
                ),
            }
        return {"policy": self.policy, "modes": mode_items}

    def summary(self, iteration: Optional[int] = None) -> Dict[str, Any]:
        return {
            "policy": self.policy,
            "reward_discount": self.reward_discount,
            "exploration_constant": self.exploration_constant,
            "modes": self.modes,
            "islands": {
                str(island_idx): self.island_summary(island_idx, iteration)
                for island_idx in sorted(self._stats)
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modes": self.modes,
            "exploration_constant": self.exploration_constant,
            "policy": self.policy,
            "reward_discount": self.reward_discount,
            "stats": {
                str(island_idx): {mode: stats.to_dict() for mode, stats in island_stats.items()}
                for island_idx, island_stats in self._stats.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchModeBandit":
        bandit = cls(
            modes=list(
                data.get("modes") or ["local_refine", "novelty_context", "paradigm_context"]
            ),
            exploration_constant=float(data.get("exploration_constant", 1.0)),
            policy=data.get("policy", "uct"),
            reward_discount=float(data.get("reward_discount", 0.99)),
        )
        for island_key, island_stats in data.get("stats", {}).items():
            island_idx = int(island_key)
            bandit._stats[island_idx] = {
                mode: ModeStats.from_dict(stats_data) for mode, stats_data in island_stats.items()
            }
            bandit.ensure_island(island_idx)
        return bandit
