"""The 150k restart changes data budgets, not the approved v2 training recipe."""
import json
from pathlib import Path
import unittest

from dev_scripts.prepare_tokenizer_corpus import quotas


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"


class SharedV3PlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous = json.loads((EXPERIMENTS / "shared-v2-main-2026-11-01.json").read_text())
        cls.plan = json.loads((EXPERIMENTS / "shared-v3-main-150k-2026-11-02.json").read_text())

    def test_exact_backbone_file_budget(self):
        recipe = self.plan["data_recipe"]["backbone"]
        self.assertEqual(recipe["files"], 150000)
        allocated = quotas(recipe["files"], self.plan["backbone_versions"])
        self.assertEqual(len(allocated), 10)
        self.assertTrue(all(cell == {"pypi": 13500, "csn": 1500} for cell in allocated.values()))
        self.assertTrue(all(cell == recipe["per_version"] for cell in allocated.values()))

    def test_adapter_training_only_expands(self):
        old = self.previous["data_recipe"]["adapter"]
        new = self.plan["data_recipe"]["adapter"]
        self.assertEqual(new["train_files"], 150000)
        for key in old.keys() - {"train_files"}:
            self.assertEqual(new[key], old[key], key)

    def test_tokenizer_initialization_and_layout_unchanged(self):
        for key in ("tokenizer", "initialization", "lanes", "backbone_versions", "held_out_version"):
            self.assertEqual(self.plan[key], self.previous[key], key)
        for key in ("seed", "rule"):
            self.assertEqual(self.plan["data_recipe"]["backbone"][key], self.previous["data_recipe"]["backbone"][key])

    def test_training_settings_unchanged_and_old_checkpoints_isolated(self):
        self.assertTrue(self.plan["approved"])
        self.assertEqual(self.plan["stages"].keys(), self.previous["stages"].keys())
        for task, old_stage in self.previous["stages"].items():
            stage = self.plan["stages"][task]
            self.assertEqual(stage["hub_repo"], old_stage["hub_repo"].replace("-v2", "-v3"))
            self.assertEqual({k: v for k, v in stage.items() if k != "hub_repo"},
                             {k: v for k, v in old_stage.items() if k != "hub_repo"}, task)
        for key in ("backbone_corpus", "adapter_corpus", "encoded"):
            self.assertEqual(self.plan[key]["repo_id"], self.previous[key]["repo_id"].replace("-v2", "-v3"))
            revision = self.plan[key]["revision"]
            if revision is not None:
                self.assertRegex(revision, r"^[0-9a-f]{40}$")
                self.assertNotEqual(revision, self.previous[key]["revision"])


if __name__ == "__main__":
    unittest.main()
