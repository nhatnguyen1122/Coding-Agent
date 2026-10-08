"""Outer MCTS selector for AdaEvolve island-lineage management."""

from __future__ import annotations

import math
import random
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class IslandMCTSNode:
    """A search island node in the outer island-lineage tree."""

    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    island_idx: Optional[int] = None
    parent_node_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0
    best_reward: float = float("-inf")
    best_fitness: float = float("-inf")
    discounted_visits: float = 0.0
    discounted_value_sum: float = 0.0
    last_discount_iteration: Optional[int] = None
    creation_iteration: int = 0
    last_selected_iteration: int = 0

    @property
    def mean_reward(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0

    @property
    def discounted_mean_reward(self) -> float:
        if not self.discounted_visits:
            return 0.0
        return self.discounted_value_sum / self.discounted_visits

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.best_reward == float("-inf"):
            data["best_reward"] = None
        if self.best_fitness == float("-inf"):
            data["best_fitness"] = None
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IslandMCTSNode":
        node = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        node.best_reward = (
            float("-inf") if data.get("best_reward") is None else float(data["best_reward"])
        )
        node.best_fitness = (
            float("-inf") if data.get("best_fitness") is None else float(data["best_fitness"])
        )
        return node


@dataclass
class IslandMCTSSelection:
    """Selected island path and expansion metadata."""

    island_idx: int
    node_id: str
    path: List[str]
    expanded: bool = False
    parent_node_id: Optional[str] = None
    parent_island_idx: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class IslandMCTSSelector:
    """MCTS over island lineages.

    The root is abstract. Initial islands are root children; spawned islands
    become children of the island node that was expanded.
    """

    def __init__(
        self,
        num_islands: int,
        *,
        max_islands: int = 12,
        exploration_constant: float = 1.0,
        backup: str = "mixed",
        backup_alpha: float = 0.5,
        selection_policy: str = "uct",
        reward_discount: float = 0.97,
        enable_expansion: bool = True,
        expansion_min_visits: int = 3,
        expansion_c: float = 1.5,
        expansion_alpha: float = 0.5,
        spawn_cooldown_iterations: int = 20,
        spawn_min_quality_ratio: float = 0.8,
        select_internal_nodes: bool = True,
        internal_node_bias: float = 0.0,
        random_seed: Optional[int] = None,
    ):
        self.max_islands = max_islands
        self.exploration_constant = exploration_constant
        self.backup = backup
        self.backup_alpha = backup_alpha
        self.selection_policy = selection_policy
        self.reward_discount = max(0.0, min(1.0, reward_discount))
        self.enable_expansion = enable_expansion
        self.expansion_min_visits = expansion_min_visits
        self.expansion_c = expansion_c
        self.expansion_alpha = expansion_alpha
        self.spawn_cooldown_iterations = spawn_cooldown_iterations
        self.spawn_min_quality_ratio = spawn_min_quality_ratio
        self.select_internal_nodes = select_internal_nodes
        self.internal_node_bias = internal_node_bias
        self.rng = random.Random(random_seed)

        self.root = IslandMCTSNode(node_id="__root__", island_idx=None, creation_iteration=0)
        self.nodes: Dict[str, IslandMCTSNode] = {self.root.node_id: self.root}
        self.island_to_node: Dict[int, str] = {}
        self.last_selection: Optional[IslandMCTSSelection] = None
        self.last_update: Dict[str, Any] = {}
        self.last_spawn_iteration: int = -spawn_cooldown_iterations
        self.total_expansions: int = 0
        self.total_backprops: int = 0
        self.current_iteration: int = 0

        for island_idx in range(num_islands):
            self.register_initial_island(island_idx)

    def register_initial_island(self, island_idx: int) -> str:
        """Register an initial island as a root child."""
        if island_idx in self.island_to_node:
            return self.island_to_node[island_idx]
        node = IslandMCTSNode(
            island_idx=island_idx,
            parent_node_id=self.root.node_id,
            creation_iteration=0,
        )
        self.nodes[node.node_id] = node
        self.root.children.append(node.node_id)
        self.island_to_node[island_idx] = node.node_id
        return node.node_id

    def select(self, iteration: int) -> IslandMCTSSelection:
        """Select an existing island node by UCT.

        When internal-node selection is enabled, a productive parent island can
        keep receiving budget after it spawns children. This makes the outer
        tree an island-lineage budget allocator instead of a strict leaf-only
        search.
        """
        self.current_iteration = iteration
        path = [self.root.node_id]
        node = self.root
        while node.children:
            child = self._select_child_or_self(node)
            if child.node_id == node.node_id:
                break
            path.append(child.node_id)
            node = child
            if node.visits == 0:
                break
        if node.island_idx is None:
            raise ValueError("Island MCTS selected root without any island children")
        node.last_selected_iteration = iteration
        selection = IslandMCTSSelection(
            island_idx=node.island_idx,
            node_id=node.node_id,
            path=path,
        )
        self.last_selection = selection
        return selection

    def select_island(self, island_idx: int, iteration: int) -> IslandMCTSSelection:
        """Force-select an existing island while preserving MCTS backprop path."""
        self.current_iteration = iteration
        node_id = self.ensure_island(island_idx)
        node = self.nodes[node_id]
        node.last_selected_iteration = iteration
        selection = IslandMCTSSelection(
            island_idx=island_idx,
            node_id=node_id,
            path=self._path_to_node(node_id),
        )
        self.last_selection = selection
        return selection

    def should_expand(
        self,
        node_id: str,
        *,
        iteration: int,
        current_num_islands: int,
        global_best_score: float,
    ) -> bool:
        """Return True when MCTS progressive widening should spawn a child island."""
        if not self.enable_expansion:
            return False
        if current_num_islands >= self.max_islands:
            return False
        if iteration - self.last_spawn_iteration < self.spawn_cooldown_iterations:
            return False
        node = self.nodes.get(node_id)
        if node is None or node.island_idx is None:
            return False
        if node.visits < self.expansion_min_visits:
            return False

        allowed_children = max(
            1, int(math.floor(self.expansion_c * (node.visits**self.expansion_alpha)))
        )
        if len(node.children) >= allowed_children:
            return False

        if node.mean_reward > 0.0:
            return True
        if node.best_fitness == float("-inf"):
            return False
        if math.isinf(global_best_score):
            return node.best_fitness > float("-inf")
        if abs(global_best_score) <= 1e-8:
            return node.best_fitness > global_best_score
        return node.best_fitness >= global_best_score * self.spawn_min_quality_ratio

    def register_expansion(
        self,
        *,
        parent_node_id: str,
        island_idx: int,
        iteration: int,
    ) -> IslandMCTSSelection:
        """Register a newly spawned island as a child of the selected island node."""
        self.current_iteration = iteration
        parent = self.nodes[parent_node_id]
        node = IslandMCTSNode(
            island_idx=island_idx,
            parent_node_id=parent_node_id,
            creation_iteration=iteration,
            last_selected_iteration=iteration,
        )
        self.nodes[node.node_id] = node
        parent.children.append(node.node_id)
        self.island_to_node[island_idx] = node.node_id
        self.last_spawn_iteration = iteration
        self.total_expansions += 1

        base_path = self._path_to_node(parent_node_id)
        selection = IslandMCTSSelection(
            island_idx=island_idx,
            node_id=node.node_id,
            path=base_path + [node.node_id],
            expanded=True,
            parent_node_id=parent_node_id,
            parent_island_idx=parent.island_idx,
        )
        self.last_selection = selection
        return selection

    def ensure_island(self, island_idx: int) -> str:
        """Ensure an island has a node, using a root child fallback."""
        if island_idx in self.island_to_node:
            return self.island_to_node[island_idx]
        return self.register_initial_island(island_idx)

    def backpropagate(
        self,
        *,
        reward: float,
        fitness: float,
        island_idx: Optional[int] = None,
        iteration: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Backpropagate reward through the last selected island path."""
        selection = self.last_selection
        if selection is None or (island_idx is not None and selection.island_idx != island_idx):
            if island_idx is None:
                raise ValueError("No island MCTS selection available for backpropagation")
            node_id = self.ensure_island(island_idx)
            selection = IslandMCTSSelection(
                island_idx=island_idx,
                node_id=node_id,
                path=self._path_to_node(node_id),
            )
            self.last_selection = selection

        for node_id in selection.path:
            node = self.nodes.get(node_id)
            if node is None:
                continue
            node.visits += 1
            node.value_sum += reward
            self._update_discounted_stats(node, reward, iteration)
            node.best_reward = max(node.best_reward, reward)
            node.best_fitness = max(node.best_fitness, fitness)

        self.total_backprops += 1
        node = self.nodes[selection.node_id]
        self.last_update = {
            "selected_island": selection.island_idx,
            "selected_node_id": selection.node_id,
            "path": list(selection.path),
            "expanded": selection.expanded,
            "parent_node_id": selection.parent_node_id,
            "parent_island_idx": selection.parent_island_idx,
            "reward": reward,
            "fitness": fitness,
            "visits": node.visits,
            "mean_reward": node.mean_reward,
            "discounted_visits": node.discounted_visits,
            "discounted_mean_reward": node.discounted_mean_reward,
            "best_reward": None if node.best_reward == float("-inf") else node.best_reward,
            "best_fitness": None if node.best_fitness == float("-inf") else node.best_fitness,
            "details": details or {},
        }
        return dict(self.last_update)

    def summary(self) -> Dict[str, Any]:
        """Return compact selector stats."""
        island_nodes = [node for node in self.nodes.values() if node.island_idx is not None]
        return {
            "enabled": True,
            "num_nodes": len(self.nodes),
            "num_island_nodes": len(island_nodes),
            "total_expansions": self.total_expansions,
            "total_backprops": self.total_backprops,
            "current_iteration": self.current_iteration,
            "selection_policy": self.selection_policy,
            "reward_discount": self.reward_discount,
            "select_internal_nodes": self.select_internal_nodes,
            "internal_node_bias": self.internal_node_bias,
            "last_spawn_iteration": self.last_spawn_iteration,
            "last_selection": self.last_selection.to_dict() if self.last_selection else None,
            "last_update": dict(self.last_update),
        }

    def node_summary(self, island_idx: int) -> Dict[str, Any]:
        """Return selector stats for a specific island."""
        node_id = self.island_to_node.get(island_idx)
        if not node_id or node_id not in self.nodes:
            return {}
        node = self.nodes[node_id]
        return {
            "node_id": node.node_id,
            "parent_node_id": node.parent_node_id,
            "children": list(node.children),
            "visits": node.visits,
            "mean_reward": node.mean_reward,
            "discounted_visits": node.discounted_visits,
            "discounted_mean_reward": node.discounted_mean_reward,
            "best_reward": None if node.best_reward == float("-inf") else node.best_reward,
            "best_fitness": None if node.best_fitness == float("-inf") else node.best_fitness,
            "creation_iteration": node.creation_iteration,
            "last_selected_iteration": node.last_selected_iteration,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_islands": self.max_islands,
            "exploration_constant": self.exploration_constant,
            "backup": self.backup,
            "backup_alpha": self.backup_alpha,
            "selection_policy": self.selection_policy,
            "reward_discount": self.reward_discount,
            "enable_expansion": self.enable_expansion,
            "expansion_min_visits": self.expansion_min_visits,
            "expansion_c": self.expansion_c,
            "expansion_alpha": self.expansion_alpha,
            "spawn_cooldown_iterations": self.spawn_cooldown_iterations,
            "spawn_min_quality_ratio": self.spawn_min_quality_ratio,
            "select_internal_nodes": self.select_internal_nodes,
            "internal_node_bias": self.internal_node_bias,
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "island_to_node": dict(self.island_to_node),
            "last_selection": self.last_selection.to_dict() if self.last_selection else None,
            "last_update": dict(self.last_update),
            "last_spawn_iteration": self.last_spawn_iteration,
            "total_expansions": self.total_expansions,
            "total_backprops": self.total_backprops,
            "current_iteration": self.current_iteration,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IslandMCTSSelector":
        selector = cls(
            0,
            max_islands=data.get("max_islands", 12),
            exploration_constant=data.get("exploration_constant", 1.0),
            backup=data.get("backup", "mixed"),
            backup_alpha=data.get("backup_alpha", 0.5),
            selection_policy=data.get("selection_policy", "uct"),
            reward_discount=data.get("reward_discount", 0.97),
            enable_expansion=data.get("enable_expansion", True),
            expansion_min_visits=data.get("expansion_min_visits", 3),
            expansion_c=data.get("expansion_c", 1.5),
            expansion_alpha=data.get("expansion_alpha", 0.5),
            spawn_cooldown_iterations=data.get("spawn_cooldown_iterations", 20),
            spawn_min_quality_ratio=data.get("spawn_min_quality_ratio", 0.8),
            select_internal_nodes=data.get("select_internal_nodes", True),
            internal_node_bias=data.get("internal_node_bias", 0.0),
        )
        selector.nodes = {
            node_id: IslandMCTSNode.from_dict(node_data)
            for node_id, node_data in data.get("nodes", {}).items()
        }
        if "__root__" not in selector.nodes:
            selector.nodes["__root__"] = IslandMCTSNode(
                node_id="__root__", island_idx=None, creation_iteration=0
            )
        selector.root = selector.nodes["__root__"]
        selector.island_to_node = {int(k): v for k, v in data.get("island_to_node", {}).items()}
        last_selection = data.get("last_selection")
        selector.last_selection = (
            IslandMCTSSelection(**last_selection) if isinstance(last_selection, dict) else None
        )
        selector.last_update = dict(data.get("last_update", {}))
        selector.last_spawn_iteration = data.get(
            "last_spawn_iteration", -selector.spawn_cooldown_iterations
        )
        selector.total_expansions = data.get("total_expansions", 0)
        selector.total_backprops = data.get("total_backprops", 0)
        selector.current_iteration = data.get("current_iteration", 0)
        return selector

    def _path_to_node(self, node_id: str) -> List[str]:
        path = []
        current = node_id
        while current:
            path.append(current)
            node = self.nodes.get(current)
            if node is None:
                break
            current = node.parent_node_id
        path.reverse()
        if not path or path[0] != self.root.node_id:
            path.insert(0, self.root.node_id)
        return path

    def _select_child(self, parent: IslandMCTSNode) -> IslandMCTSNode:
        children = [self.nodes[child_id] for child_id in parent.children if child_id in self.nodes]
        unvisited = [child for child in children if child.visits == 0]
        if unvisited:
            return self.rng.choice(unvisited)
        return max(children, key=lambda child: self._uct_score(parent, child))

    def _select_child_or_self(self, parent: IslandMCTSNode) -> IslandMCTSNode:
        if parent.island_idx is None or not self.select_internal_nodes:
            return self._select_child(parent)

        children = [self.nodes[child_id] for child_id in parent.children if child_id in self.nodes]
        unvisited = [child for child in children if child.visits == 0]
        if unvisited:
            return self.rng.choice(unvisited)

        candidates = [parent] + children
        return max(candidates, key=lambda node: self._candidate_score(parent, node))

    def _candidate_score(self, parent: IslandMCTSNode, candidate: IslandMCTSNode) -> float:
        if candidate.node_id == parent.node_id:
            exploit = self._exploit_score(candidate) + self.internal_node_bias
            parent_visits = self._effective_parent_visits(parent)
            candidate_visits = self._effective_node_visits(candidate)
            explore = self.exploration_constant * math.sqrt(
                math.log(parent_visits + 1) / max(1e-9, candidate_visits)
            )
            return exploit + explore
        return self._uct_score(parent, candidate)

    def _uct_score(self, parent: IslandMCTSNode, child: IslandMCTSNode) -> float:
        exploit = self._exploit_score(child)
        parent_visits = self._effective_parent_visits(parent)
        child_visits = self._effective_node_visits(child)
        if child_visits <= 0:
            explore = float("inf")
        else:
            explore = self.exploration_constant * math.sqrt(
                math.log(parent_visits + 1) / child_visits
            )
        return exploit + explore

    def _exploit_score(self, node: IslandMCTSNode) -> float:
        if self.selection_policy == "discounted_uct" and node.discounted_visits > 0:
            visits, value_sum = self._projected_discounted_stats(node)
            if visits > 0:
                return value_sum / visits
        if self.backup == "max":
            return node.best_reward if node.best_reward != float("-inf") else 0.0
        elif self.backup == "mean":
            return node.mean_reward
        else:
            best = node.best_reward if node.best_reward != float("-inf") else node.mean_reward
            return self.backup_alpha * node.mean_reward + (1.0 - self.backup_alpha) * best

    def _effective_node_visits(self, node: IslandMCTSNode) -> float:
        if self.selection_policy == "discounted_uct" and node.discounted_visits > 0:
            visits, _ = self._projected_discounted_stats(node)
            return visits
        return float(node.visits)

    def _effective_parent_visits(self, parent: IslandMCTSNode) -> float:
        if self.selection_policy == "discounted_uct":
            child_visits = sum(
                self._effective_node_visits(self.nodes[child_id])
                for child_id in parent.children
                if child_id in self.nodes
            )
            return max(1.0, self._effective_node_visits(parent), child_visits)
        child_visits = sum(
            self.nodes[child_id].visits for child_id in parent.children if child_id in self.nodes
        )
        return float(max(1, parent.visits, child_visits))

    def _update_discounted_stats(
        self,
        node: IslandMCTSNode,
        reward: float,
        iteration: Optional[int],
    ) -> None:
        effective_iteration = self.current_iteration if iteration is None else iteration
        if effective_iteration is not None:
            if node.last_discount_iteration is not None:
                gap = max(0, effective_iteration - node.last_discount_iteration)
                factor = self.reward_discount**gap
                node.discounted_visits *= factor
                node.discounted_value_sum *= factor
            node.last_discount_iteration = effective_iteration
        node.discounted_visits += 1.0
        node.discounted_value_sum += reward

    def _projected_discounted_stats(self, node: IslandMCTSNode) -> tuple[float, float]:
        visits = node.discounted_visits
        value_sum = node.discounted_value_sum
        if node.last_discount_iteration is None:
            return visits, value_sum
        gap = max(0, self.current_iteration - node.last_discount_iteration)
        if gap <= 0:
            return visits, value_sum
        factor = self.reward_discount**gap
        return visits * factor, value_sum * factor
