"""Per-island MCTS population manager for AdaEvolve.

This module replaces the flat per-island archive with a lineage tree.  The
AdaEvolve database still decides which island receives the next evaluation; the
selected island's tree decides which program lineage to expand and which nodes
remain active.
"""

from __future__ import annotations

import json
import math
import random
import re
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from coding_agent.search.base_database import Program
from coding_agent.utils.metrics import get_score


@dataclass
class MCTSAddResult:
    """Admission metadata used for MCTS reward and logging."""

    program_id: str
    island_idx: int
    parent_id: Optional[str] = None
    parent_node_id: Optional[str] = None
    child_node_id: Optional[str] = None
    accepted: bool = True
    evaluation_success: bool = True
    fitness: float = 0.0
    parent_fitness: float = 0.0
    feature_key: Optional[str] = None
    added_to_new_signature: bool = False
    improved_over_parent: bool = False
    reward: float = 0.0


@dataclass
class MCTSNode:
    """A program lineage node inside one AdaEvolve island."""

    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    program_id: Optional[str] = None
    parent_node_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0
    best_reward: float = float("-inf")
    best_fitness: float = float("-inf")
    pending_count: int = 0
    depth: int = 0
    feature_key: Optional[str] = None
    active: bool = True
    creation_iteration: int = 0
    last_expanded_iteration: int = 0

    @property
    def mean_reward(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.best_reward == float("-inf"):
            data["best_reward"] = None
        if self.best_fitness == float("-inf"):
            data["best_fitness"] = None
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCTSNode":
        node = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        node.best_reward = (
            float("-inf") if data.get("best_reward") is None else float(data["best_reward"])
        )
        node.best_fitness = (
            float("-inf") if data.get("best_fitness") is None else float(data["best_fitness"])
        )
        node.pending_count = 0
        return node


class MCTSIslandTree:
    """Tree-structured population manager for a single AdaEvolve island."""

    def __init__(
        self,
        island_idx: int,
        *,
        population_size: int = 20,
        exploration_constant: float = 0.7,
        qd_weight: float = 0.5,
        max_tree_depth: int = 20,
        backup: str = "mixed",
        backup_alpha: float = 0.5,
        widening_alpha: float = 0.5,
        w_quality: float = 0.35,
        w_delta: float = 0.25,
        w_novelty: float = 0.2,
        w_pareto: float = 0.0,
        w_invalid: float = 0.2,
        w_cost: float = 0.0,
        random_seed: Optional[int] = None,
    ):
        self.island_idx = island_idx
        self.population_size = population_size
        self.exploration_constant = exploration_constant
        self.qd_weight = qd_weight
        self.max_tree_depth = max_tree_depth
        self.backup = backup
        self.backup_alpha = backup_alpha
        self.widening_alpha = widening_alpha
        self.w_quality = w_quality
        self.w_delta = w_delta
        self.w_novelty = w_novelty
        self.w_pareto = w_pareto
        self.w_invalid = w_invalid
        self.w_cost = w_cost
        self.rng = random.Random(random_seed)

        self.root = MCTSNode(node_id="__root__", depth=0, active=False)
        self.nodes: Dict[str, MCTSNode] = {self.root.node_id: self.root}
        self.program_to_node: Dict[str, str] = {}
        self.active_program_ids: Set[str] = set()
        self.feature_counts: Dict[str, int] = {}
        self.best_program_id: Optional[str] = None
        self.last_selected_node_id: Optional[str] = None
        self.total_expansions = 0
        self.total_backprops = 0

    def register_program(
        self,
        program: Program,
        *,
        parent_program_id: Optional[str] = None,
        iteration: int = 0,
        feature_key: Optional[str] = None,
    ) -> MCTSNode:
        """Register a seed or migrated program as an active tree node."""
        if program.id in self.program_to_node:
            node = self.nodes[self.program_to_node[program.id]]
            node.active = True
            self.active_program_ids.add(program.id)
            return node

        parent_node_id = "__root__"
        parent_depth = 0
        if parent_program_id and parent_program_id in self.program_to_node:
            parent_node_id = self.program_to_node[parent_program_id]
            parent_depth = self.nodes[parent_node_id].depth

        key = feature_key or make_feature_signature(program)
        fitness = get_score(program.metrics or {})
        node = MCTSNode(
            program_id=program.id,
            parent_node_id=parent_node_id,
            depth=parent_depth + 1,
            feature_key=key,
            best_fitness=fitness,
            creation_iteration=iteration,
            active=True,
        )
        self.nodes[node.node_id] = node
        self.nodes[parent_node_id].children.append(node.node_id)
        self.program_to_node[program.id] = node.node_id
        self.active_program_ids.add(program.id)
        self.feature_counts[key] = self.feature_counts.get(key, 0) + 1
        if self.best_program_id is None or fitness > self._program_fitness(self.best_program_id):
            self.best_program_id = program.id
        return node

    def select(self, programs: Dict[str, Program]) -> Optional[MCTSNode]:
        """Select an active node for expansion using UCT plus feature rarity."""
        self._discard_missing_programs(programs)
        if not self.active_program_ids:
            return None

        node = self.root
        while True:
            active_children = [self.nodes[cid] for cid in node.children if self._is_selectable(cid)]
            if not active_children:
                break
            if node.depth >= self.max_tree_depth:
                break
            if node.program_id is not None and self._should_widen(node):
                break
            node = max(active_children, key=lambda child: self._uct_score(node, child))

        if node.program_id is None or not node.active:
            # Root or inactive branch: fall back to the best active child.
            candidates = [self.nodes[self.program_to_node[pid]] for pid in self.active_program_ids]
            node = max(candidates, key=lambda n: (n.best_fitness, n.mean_reward, -n.pending_count))

        node.pending_count += 1
        self.last_selected_node_id = node.node_id
        return node

    def admit(
        self,
        program: Program,
        *,
        selected_node_id: Optional[str],
        iteration: int,
        max_fitness: float,
        evaluation_success: bool = True,
        pareto_bonus: float = 0.0,
        token_cost: float = 0.0,
    ) -> Tuple[MCTSAddResult, List[str]]:
        """Admit an evaluated child and backpropagate reward."""
        parent_node = self.nodes.get(selected_node_id or "")
        parent_id = program.parent_id or (parent_node.program_id if parent_node else None)
        parent_fitness = 0.0
        if parent_node is not None and parent_node.program_id:
            parent_fitness = (
                parent_node.best_fitness if parent_node.best_fitness != float("-inf") else 0.0
            )
        fitness = get_score(program.metrics or {})
        feature_key = make_feature_signature(program)

        result = MCTSAddResult(
            program_id=program.id,
            island_idx=self.island_idx,
            parent_id=parent_id,
            parent_node_id=parent_node.node_id if parent_node else None,
            evaluation_success=evaluation_success,
            fitness=fitness,
            parent_fitness=parent_fitness,
            feature_key=feature_key,
            added_to_new_signature=self.feature_counts.get(feature_key, 0) == 0,
            improved_over_parent=fitness > parent_fitness,
        )

        if parent_node is not None:
            parent_node.pending_count = max(0, parent_node.pending_count - 1)

        if not evaluation_success:
            result.accepted = False
            result.reward = -self.w_invalid
            self._backpropagate(
                parent_node.node_id if parent_node else "__root__", result.reward, fitness
            )
            return result, []

        if program.id in self.program_to_node:
            child = self.nodes[self.program_to_node[program.id]]
            child.active = True
        else:
            parent_node_id = parent_node.node_id if parent_node is not None else "__root__"
            child = MCTSNode(
                program_id=program.id,
                parent_node_id=parent_node_id,
                depth=(parent_node.depth + 1) if parent_node else 1,
                feature_key=feature_key,
                best_fitness=fitness,
                creation_iteration=iteration,
                active=True,
            )
            self.nodes[child.node_id] = child
            self.nodes[parent_node_id].children.append(child.node_id)
            if parent_node is not None:
                parent_node.last_expanded_iteration = iteration
            self.program_to_node[program.id] = child.node_id
            self.total_expansions += 1

        self.active_program_ids.add(program.id)
        self.feature_counts[feature_key] = self.feature_counts.get(feature_key, 0) + 1
        result.child_node_id = child.node_id

        if self.best_program_id is None or fitness > self._program_fitness(self.best_program_id):
            self.best_program_id = program.id

        result.reward = self._compute_reward(result, max_fitness, pareto_bonus, token_cost)
        self._backpropagate(child.node_id, result.reward, fitness)
        removed = self.prune()
        return result, removed

    def release_pending(self, node_id: Optional[str]) -> None:
        if not node_id or node_id not in self.nodes:
            return
        self.nodes[node_id].pending_count = max(0, self.nodes[node_id].pending_count - 1)

    def get_active_programs(self, programs: Dict[str, Program]) -> List[Program]:
        self._discard_missing_programs(programs)
        return [programs[pid] for pid in self.active_program_ids if pid in programs]

    def get_top_programs(
        self, programs: Dict[str, Program], n: Optional[int] = None
    ) -> List[Program]:
        active = self.get_active_programs(programs)
        active.sort(key=lambda p: get_score(p.metrics or {}), reverse=True)
        return active if n is None else active[:n]

    def get_children(
        self, parent_id: str, programs: Dict[str, Program], limit: int = 5
    ) -> List[Program]:
        node_id = self.program_to_node.get(parent_id)
        if not node_id:
            return []
        children = [
            programs[self.nodes[cid].program_id]
            for cid in self.nodes[node_id].children
            if cid in self.nodes
            and self.nodes[cid].program_id in programs
            and self.nodes[cid].active
        ]
        children.sort(key=lambda p: getattr(p, "iteration_found", 0))
        return children[-limit:]

    def prune(self) -> List[str]:
        """Deactivate weak leaves until active population fits capacity."""
        removed: List[str] = []
        while len(self.active_program_ids) > self.population_size:
            candidates = [
                self.nodes[self.program_to_node[pid]]
                for pid in self.active_program_ids
                if pid != self.best_program_id and not self._has_active_children(pid)
            ]
            if not candidates:
                candidates = [
                    self.nodes[self.program_to_node[pid]]
                    for pid in self.active_program_ids
                    if pid != self.best_program_id
                ]
            if not candidates:
                break
            victim = min(
                candidates,
                key=lambda n: (
                    n.mean_reward,
                    n.best_fitness if n.best_fitness != float("-inf") else -1e18,
                    n.visits,
                    -n.creation_iteration,
                ),
            )
            victim.active = False
            if victim.program_id:
                self.active_program_ids.discard(victim.program_id)
                removed.append(victim.program_id)
        return removed

    def summary(self) -> Dict[str, Any]:
        depths = [n.depth for n in self.nodes.values() if n.program_id is not None]
        visits = [n.visits for n in self.nodes.values() if n.visits > 0]
        return {
            "island_idx": self.island_idx,
            "active_programs": len(self.active_program_ids),
            "total_nodes": len(self.nodes),
            "max_depth": max(depths) if depths else 0,
            "root_visits": self.root.visits,
            "avg_visits": sum(visits) / len(visits) if visits else 0.0,
            "total_expansions": self.total_expansions,
            "total_backprops": self.total_backprops,
            "best_program_id": self.best_program_id,
            "last_selected_node_id": self.last_selected_node_id,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "island_idx": self.island_idx,
            "population_size": self.population_size,
            "exploration_constant": self.exploration_constant,
            "qd_weight": self.qd_weight,
            "max_tree_depth": self.max_tree_depth,
            "backup": self.backup,
            "backup_alpha": self.backup_alpha,
            "widening_alpha": self.widening_alpha,
            "w_quality": self.w_quality,
            "w_delta": self.w_delta,
            "w_novelty": self.w_novelty,
            "w_pareto": self.w_pareto,
            "w_invalid": self.w_invalid,
            "w_cost": self.w_cost,
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "program_to_node": dict(self.program_to_node),
            "active_program_ids": list(self.active_program_ids),
            "feature_counts": dict(self.feature_counts),
            "best_program_id": self.best_program_id,
            "last_selected_node_id": self.last_selected_node_id,
            "total_expansions": self.total_expansions,
            "total_backprops": self.total_backprops,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCTSIslandTree":
        tree = cls(
            island_idx=data.get("island_idx", 0),
            population_size=data.get("population_size", 20),
            exploration_constant=data.get("exploration_constant", 0.7),
            qd_weight=data.get("qd_weight", 0.5),
            max_tree_depth=data.get("max_tree_depth", 20),
            backup=data.get("backup", "mixed"),
            backup_alpha=data.get("backup_alpha", 0.5),
            widening_alpha=data.get("widening_alpha", 0.5),
            w_quality=data.get("w_quality", 0.35),
            w_delta=data.get("w_delta", 0.25),
            w_novelty=data.get("w_novelty", 0.2),
            w_pareto=data.get("w_pareto", 0.0),
            w_invalid=data.get("w_invalid", 0.2),
            w_cost=data.get("w_cost", 0.0),
        )
        tree.nodes = {
            node_id: MCTSNode.from_dict(node_data)
            for node_id, node_data in data.get("nodes", {}).items()
        }
        if "__root__" not in tree.nodes:
            tree.nodes["__root__"] = MCTSNode(node_id="__root__", depth=0, active=False)
        tree.root = tree.nodes["__root__"]
        tree.program_to_node = dict(data.get("program_to_node", {}))
        tree.active_program_ids = set(data.get("active_program_ids", []))
        tree.feature_counts = dict(data.get("feature_counts", {}))
        tree.best_program_id = data.get("best_program_id")
        tree.last_selected_node_id = data.get("last_selected_node_id")
        tree.total_expansions = data.get("total_expansions", 0)
        tree.total_backprops = data.get("total_backprops", 0)
        return tree

    def _is_selectable(self, node_id: str) -> bool:
        node = self.nodes.get(node_id)
        return bool(node and node.active and node.program_id in self.active_program_ids)

    def _should_widen(self, node: MCTSNode) -> bool:
        if node.visits == 0:
            return True
        max_children = max(1, math.ceil(node.visits**self.widening_alpha))
        active_children = sum(1 for child_id in node.children if self._is_selectable(child_id))
        return active_children < max_children

    def _uct_score(self, parent: MCTSNode, child: MCTSNode) -> float:
        if self.backup == "max":
            exploit = child.best_reward if child.best_reward != float("-inf") else 0.0
        elif self.backup == "mean":
            exploit = child.mean_reward
        else:
            best = child.best_reward if child.best_reward != float("-inf") else child.mean_reward
            exploit = self.backup_alpha * child.mean_reward + (1.0 - self.backup_alpha) * best

        effective_visits = child.visits + child.pending_count
        parent_visits = max(1, parent.visits + parent.pending_count)
        explore = (
            float("inf")
            if effective_visits == 0
            else self.exploration_constant
            * math.sqrt(math.log(parent_visits + 1) / effective_visits)
        )
        rarity = 0.0
        if child.feature_key:
            rarity = 1.0 / math.sqrt(max(1, self.feature_counts.get(child.feature_key, 1)))
        pending_penalty = 0.05 * child.pending_count
        return exploit + explore + self.qd_weight * rarity - pending_penalty

    def _compute_reward(
        self,
        result: MCTSAddResult,
        max_fitness: float,
        pareto_bonus: float,
        token_cost: float,
    ) -> float:
        if not result.accepted or not result.evaluation_success:
            return -self.w_invalid
        denom = max(abs(max_fitness), abs(result.fitness), 1e-8)
        quality = max(0.0, min(1.0, result.fitness / denom))
        delta = max(0.0, result.fitness - result.parent_fitness) / denom
        novelty = 1.0 if result.added_to_new_signature else 0.0
        improved = 1.0 if result.improved_over_parent else 0.0
        return (
            self.w_quality * quality
            + self.w_delta * min(1.0, delta)
            + self.w_novelty * novelty
            + 0.1 * improved
            + self.w_pareto * pareto_bonus
            - self.w_cost * token_cost
        )

    def _backpropagate(self, start_node_id: str, reward: float, fitness: float) -> None:
        node_id: Optional[str] = start_node_id
        while node_id:
            node = self.nodes.get(node_id)
            if node is None:
                break
            node.visits += 1
            node.value_sum += reward
            node.best_reward = max(node.best_reward, reward)
            node.best_fitness = max(node.best_fitness, fitness)
            node_id = node.parent_node_id
        self.total_backprops += 1

    def _discard_missing_programs(self, programs: Dict[str, Program]) -> None:
        missing = [pid for pid in self.active_program_ids if pid not in programs]
        for pid in missing:
            self.active_program_ids.discard(pid)
            node_id = self.program_to_node.get(pid)
            if node_id in self.nodes:
                self.nodes[node_id].active = False

    def _program_fitness(self, program_id: str) -> float:
        node_id = self.program_to_node.get(program_id)
        if not node_id:
            return float("-inf")
        return self.nodes[node_id].best_fitness

    def _has_active_children(self, program_id: str) -> bool:
        node_id = self.program_to_node.get(program_id)
        if not node_id:
            return False
        return any(self._is_selectable(child_id) for child_id in self.nodes[node_id].children)


def make_feature_signature(program: Program, metric_keys: Optional[Iterable[str]] = None) -> str:
    """Build a compact behavioral signature for QD-style tree rewards."""
    metrics = program.metrics or {}
    keys = list(metric_keys or [])
    if not keys:
        keys = [
            key
            for key, value in sorted(metrics.items())
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ][:4]
    parts: List[str] = []
    for key in keys:
        value = metrics.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            parts.append(f"{key}:{round(float(value), 3)}")

    solution = program.solution or ""
    funcs = len(re.findall(r"\bdef\s+\w+", solution))
    loops = len(re.findall(r"\b(for|while)\b", solution))
    imports = len(re.findall(r"^\s*(import|from)\s+", solution, re.MULTILINE))
    length_bucket = int(len(solution) / 500)
    parts.extend([f"funcs:{funcs}", f"loops:{loops}", f"imports:{imports}", f"len:{length_bucket}"])
    return "|".join(parts)
