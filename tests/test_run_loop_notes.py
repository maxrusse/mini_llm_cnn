from __future__ import annotations

import importlib.util
import pathlib
import shutil
import unittest
from unittest import mock


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

    def test_do_run_config_allows_bootstrap_when_enabled(self) -> None:
        tmp_root = REPO_ROOT / ".mini_loop" / "test_tmp"
        tmp_root.mkdir(parents=True, exist_ok=True)
        tmp_dir = tmp_root / "run_config_bootstrap_case"
        shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            config_path = tmp_dir / "candidate.yaml"
            config_path.write_text("model:\n  name: fpn\n", encoding="utf-8")
            paths = {
                "results_file": tmp_dir / "results.tsv",
                "state_file": tmp_dir / "state.json",
                "generated_config_dir": tmp_dir / "generated_configs",
                "benchmark_repo_root": tmp_dir,
                "benchmark_python_exe": tmp_dir / "python.exe",
                "summary_file": tmp_dir / "summary.tsv",
                "logs_dir": tmp_dir / "logs",
                "runs_dir": tmp_dir / "runs",
            }
            paths["generated_config_dir"].mkdir(parents=True, exist_ok=True)
            settings = {
                "selection_metric": "roc_auc_presence",
                "allow_run_config_bootstrap": True,
            }
            with mock.patch.object(run_loop, "current_best_result", return_value=None), \
                mock.patch.object(run_loop, "load_yaml", return_value={"model": {"name": "fpn"}}), \
                mock.patch.object(run_loop, "apply_runtime_tier", side_effect=lambda cfg, *_: cfg), \
                mock.patch.object(run_loop, "apply_selection_metric", side_effect=lambda cfg, *_: cfg), \
                mock.patch.object(run_loop, "make_experiment_id", return_value="e0001"), \
                mock.patch.object(run_loop, "write_candidate_config", return_value=tmp_dir / "generated_configs" / "e0001.yaml"), \
                mock.patch.object(run_loop, "execute_experiment", return_value={"experiment_id": "e0001", "status": "baseline"}) as mocked_execute:
                rc = run_loop.do_run_config(
                    settings=settings,
                    paths=paths,
                    tier="medium",
                    config_arg=str(config_path),
                    label="seeded_code_run",
                    parent_experiment_id="e0018",
                    dry_run=True,
                )
            self.assertEqual(rc, 0)
            self.assertTrue(mocked_execute.called)
            self.assertTrue(mocked_execute.call_args.kwargs["keep_on_first"])
            self.assertIn(
                "bootstrap first run-config for empty tier ledger",
                mocked_execute.call_args.kwargs["note"],
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
