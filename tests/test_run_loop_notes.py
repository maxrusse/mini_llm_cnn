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


class RunLoopNotesTests(unittest.TestCase):
    def test_annotate_review_alternate_note_marks_llm_review_signal(self) -> None:
        note = run_loop.annotate_review_alternate_note(
            note="manual config from near_best.yaml",
            primary_metric="roc_auc_presence",
            prior_best={"experiment_id": "e0006", "val_metric_value": "0.824825"},
            prior_metrics={"roc_auc_presence": 0.824825},
            review_epsilon=0.01,
            signals=["primary_gap=0.003800", "dice_pos_better_by=0.012000"],
        )
        self.assertIn("review-worthy alternate vs e0006", note)
        self.assertIn("llm_review_epsilon=0.010000", note)
        self.assertIn("dice_pos_better_by=0.012000", note)

    def test_annotate_metric_outcome_note_marks_tie(self) -> None:
        note = run_loop.annotate_metric_outcome_note(
            note="manual config from tie_case.yaml",
            selection_metric="dice_pos",
            metric_value=0.331176,
            prior_best={"experiment_id": "e0001", "val_metric_value": "0.331176"},
            epsilon=1e-6,
        )
        self.assertIn("matched current best e0001 dice_pos=0.331176", note)

    def test_annotate_metric_outcome_note_marks_regression(self) -> None:
        note = run_loop.annotate_metric_outcome_note(
            note="manual config from worse_case.yaml",
            selection_metric="dice_pos",
            metric_value=0.30,
            prior_best={"experiment_id": "e0001", "val_metric_value": "0.331176"},
            epsilon=1e-6,
        )
        self.assertIn("did not improve over e0001 dice_pos=0.331176", note)


if __name__ == "__main__":
    unittest.main()
