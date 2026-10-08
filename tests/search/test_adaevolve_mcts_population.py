import tempfile
import unittest

from coding_agent.config import AdaEvolveDatabaseConfig
from coding_agent.search.adaevolve.database import AdaEvolveDatabase
from coding_agent.search.base_database import Program


def make_config(**overrides):
    config = AdaEvolveDatabaseConfig(
        population_manager="mcts_tree",
        use_unified_archive=False,
        use_dynamic_islands=False,
        use_paradigm_breakthrough=False,
        population_size=3,
        num_islands=2,
        fitness_key="combined_score",
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def make_program(pid, score, parent_id=None):
    return Program(
        id=pid,
        solution=f"def run_packing():\n    return None  # {pid}\n",
        metrics={"combined_score": score, "validity": 1.0},
        parent_id=parent_id,
    )


class TestAdaEvolveMCTSPopulation(unittest.TestCase):
    def test_sample_uses_island_mcts_tree(self):
        db = AdaEvolveDatabase("adaevolve", make_config())
        seed = make_program("seed", 0.5)
        db.add(seed, iteration=0, target_island=0)

        parent_dict, context = db.sample(num_context_programs=2)

        parent = next(iter(parent_dict.values()))
        self.assertEqual(parent.id, "seed")
        self.assertEqual(db._last_sampling_mode, "mcts_tree")
        self.assertIn("seed", db.island_trees[0].active_program_ids)
        self.assertIn("", context)

    def test_child_admission_backprops_and_prunes(self):
        db = AdaEvolveDatabase("adaevolve", make_config(population_size=2))
        seed = make_program("seed", 0.5)
        db.add(seed, iteration=0, target_island=0)
        db.sample(num_context_programs=1)

        child = make_program("child", 0.8, parent_id="seed")
        result = db.add_with_result(child, iteration=1, parent_id="seed")

        self.assertTrue(result.accepted)
        self.assertGreater(result.reward, 0)
        self.assertEqual(db.best_program_id, "child")
        self.assertGreater(db.island_trees[0].root.visits, 0)

        db.sample(num_context_programs=1)
        child2 = make_program("child2", 0.6, parent_id="child")
        db.add_with_result(child2, iteration=2, parent_id="child")
        self.assertLessEqual(len(db.island_trees[0].active_program_ids), 2)

    def test_save_load_preserves_mcts_tree(self):
        db = AdaEvolveDatabase("adaevolve", make_config())
        seed = make_program("seed", 0.5)
        db.add(seed, iteration=0, target_island=0)
        db.sample(num_context_programs=1)
        child = make_program("child", 0.7, parent_id="seed")
        db.add_with_result(child, iteration=1, parent_id="seed")

        with tempfile.TemporaryDirectory() as tmp:
            db.save(tmp, iteration=1)
            loaded = AdaEvolveDatabase("adaevolve", make_config())
            loaded.load(tmp)

            self.assertIn("child", loaded.programs)
            self.assertIn("child", loaded.island_trees[0].active_program_ids)
            self.assertGreaterEqual(len(loaded.island_trees[0].nodes), 2)


if __name__ == "__main__":
    unittest.main()
