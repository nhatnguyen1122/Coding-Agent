"""Tests for configuration defaults."""

from coding_agent.config import Config, EvaluatorConfig, IslandMCTSDatabaseConfig


class TestEvaluatorConfigDefaults:
    def test_default_timeout(self):
        assert EvaluatorConfig().timeout == 360

    def test_default_max_retries(self):
        assert EvaluatorConfig().max_retries == 3


def test_island_mcts_uses_hierarchical_defaults():
    config = Config.from_dict({"search": {"type": "island_mcts"}})

    assert isinstance(config.search.database, IslandMCTSDatabaseConfig)
    assert config.search.database.population_manager == "mcts_tree"
    assert config.search.database.island_selector == "mcts"
    assert config.search.database.island_mcts_enable_modes is True
