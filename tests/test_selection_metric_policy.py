from __future__ import annotations

import importlib.util
import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "run_loop.py"
SPEC = importlib.util.spec_from_file_location("mini_run_loop", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
run_loop = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_loop)


class SelectionMetricPolicyTests(unittest.TestCase):
    def test_apply_runtime_tier_does_not_inject_legacy_budget_overrides(self) -> None:
        cfg = {"training": {"epochs": 17, "max_train_batches": 1400, "max_eval_batches": 120}}
        settings = {
            "runtime_tiers": {
                "medium": {
                    "comparison_only": True,
                    "description": "comparison bucket only",
                }
            }
        }
        out = run_loop.apply_runtime_tier(cfg, settings, "medium")
        self.assertEqual(out["training"]["epochs"], 17)
        self.assertEqual(out["training"]["max_train_batches"], 1400)
        self.assertEqual(out["training"]["max_eval_batches"], 120)
        self.assertIsNot(out, cfg)

    def test_apply_selection_metric_overrides_training_metric(self) -> None:
        cfg = {"training": {"selection_metric": "dice_pos"}}
        settings = {"selection_metric": "roc_auc_presence"}
        out = run_loop.apply_selection_metric(cfg, settings)
        self.assertEqual(out["training"]["selection_metric"], "roc_auc_presence")
        self.assertEqual(cfg["training"]["selection_metric"], "dice_pos")

    def test_compare_metric_priority_uses_tiebreakers(self) -> None:
        current = {
            "roc_auc_presence": 0.90,
            "average_precision_presence": 0.61,
            "best_f1_presence": 0.50,
            "dice_pos": 0.20,
        }
        reference = {
            "roc_auc_presence": 0.90,
            "average_precision_presence": 0.60,
            "best_f1_presence": 0.80,
            "dice_pos": 0.70,
        }
        comparison, decisive = run_loop.compare_metric_priority(
            current,
            reference,
            ["roc_auc_presence", "average_precision_presence", "best_f1_presence", "dice_pos"],
            1e-6,
        )
        self.assertEqual(comparison, 1)
        self.assertEqual(decisive, "average_precision_presence")

    def test_is_review_worthy_alternate_uses_primary_gap_or_secondary_gain(self) -> None:
        worthy, signals = run_loop.is_review_worthy_alternate(
            {"roc_auc_presence": 0.8210, "average_precision_presence": 0.50},
            {"roc_auc_presence": 0.8248, "average_precision_presence": 0.90},
            ["roc_auc_presence", "average_precision_presence"],
            0.005,
        )
        self.assertTrue(worthy)
        self.assertIn("primary_gap=0.003800", signals)

        worthy, signals = run_loop.is_review_worthy_alternate(
            {"roc_auc_presence": 0.8180, "average_precision_presence": 0.95},
            {"roc_auc_presence": 0.8248, "average_precision_presence": 0.10},
            ["roc_auc_presence", "average_precision_presence"],
            0.005,
        )
        self.assertTrue(worthy)
        self.assertIn("average_precision_presence_better_by=0.850000", signals)


if __name__ == "__main__":
    unittest.main()
