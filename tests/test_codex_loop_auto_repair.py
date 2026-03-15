from __future__ import annotations

import importlib.util
import pathlib
import shutil
import unittest
from unittest import mock


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "codex_loop.py"
SPEC = importlib.util.spec_from_file_location("mini_codex_loop", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
codex_loop = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(codex_loop)


class AutoRepairHelpersTests(unittest.TestCase):
    def test_build_action_schema_required_matches_properties(self) -> None:
        schema = codex_loop.build_action_schema()
        required = set(schema["required"])
        self.assertTrue(set(schema["properties"]).issubset(required))

    def test_validate_action_allows_training_actions_without_budget_fields(self) -> None:
        issues = codex_loop.validate_action(
            {
                "action": "run_config",
                "rationale": "This is a sufficiently long rationale for a config proposal.",
                "label": "budgetless",
                "runtime_tier": "medium",
                "config_path": "generated_configs/test.yaml",
                "config_yaml": "model:\n  name: simple_unet\n",
                "parent_experiment_id": "",
                "experiment_id": "",
                "packages": [],
                "download_url": "",
                "download_path": "",
                "code_edits": [],
                "notes": "",
            }
        )
        self.assertNotIn("missing_expected_runtime_minutes", issues)
        self.assertNotIn("budget_reasoning_too_short", issues)

    def test_build_runtime_env_resolves_repo_local_cache_dirs(self) -> None:
        env = codex_loop.build_runtime_env(
            REPO_ROOT,
            {
                "runtime_env": {
                    "TORCH_HOME": ".mini_loop/test_env/torch",
                    "HF_HOME": ".mini_loop/test_env/hf",
                    "TMP": ".mini_loop/test_env/tmp",
                    "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
                }
            },
        )
        self.assertTrue(env["TORCH_HOME"].endswith("mini_llm_cnn\\.mini_loop\\test_env\\torch"))
        self.assertTrue(pathlib.Path(env["TORCH_HOME"]).exists())
        self.assertTrue(pathlib.Path(env["HF_HOME"]).exists())
        self.assertTrue(pathlib.Path(env["TMP"]).exists())
        self.assertEqual(env["HF_HUB_DISABLE_SYMLINKS_WARNING"], "1")

    def test_extract_missing_module(self) -> None:
        text = "Traceback\nModuleNotFoundError: No module named 'segmentation_models_pytorch'\n"
        self.assertEqual(codex_loop.extract_missing_module(text), "segmentation_models_pytorch")

    def test_extract_unsupported_model_name(self) -> None:
        text = "ValueError: Unsupported model.name: unet_resnet34"
        self.assertEqual(codex_loop.extract_unsupported_model_name(text), "unet_resnet34")

    def test_normalize_config_yaml_text_unescapes_newlines(self) -> None:
        raw = "model:\\n  name: simple_unet\\ntraining:\\n  epochs: 12"
        normalized = codex_loop.normalize_config_yaml_text(raw)
        self.assertIn("model:\n  name: simple_unet", normalized)
        self.assertNotIn("\\n", normalized)

    def test_sanitize_config_yaml_text_repairs_common_stray_prefix_glitch(self) -> None:
        raw = (
            "loss:\n"
            "  name: bce_dice\n"
            "n  bce_weight: 0.4\n"
            "  dice_weight: 0.6\n"
            "model:\n"
            "  name: fpn\n"
        )
        sanitized = codex_loop.sanitize_config_yaml_text(raw)
        self.assertIn("  bce_weight: 0.4\n", sanitized)
        self.assertNotIn("\nn  bce_weight", sanitized)

    def test_extract_model_name_from_yaml_text(self) -> None:
        yaml_text = "model:\n  name: unet_resnet34\ntraining:\n  epochs: 12\n"
        self.assertEqual(codex_loop.extract_model_name_from_yaml_text(yaml_text), "unet_resnet34")

    def test_infer_direct_module_packages_uses_alias(self) -> None:
        packages = codex_loop.infer_direct_module_packages("cv2", {"cv2": ["opencv-python"]})
        self.assertEqual(packages, ["opencv-python"])

    def test_apply_model_name_fallback_to_run_command(self) -> None:
        tmp_root = REPO_ROOT / ".mini_loop" / "test_tmp"
        tmp_root.mkdir(parents=True, exist_ok=True)
        tmp_dir = tmp_root / "fallback_case"
        shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            config_path = tmp_dir / "candidate.yaml"
            config_path.write_text("model:\n  name: unet_resnet34\n", encoding="utf-8")
            command = ["python", "run_loop.py", "run-config", "--config", str(config_path)]

            result = codex_loop.apply_model_name_fallback_to_run_command(
                repo_root=tmp_dir,
                command=command,
                unsupported_model="unet_resnet34",
                fallback_map={"unet_resnet34": "simple_unet"},
                label="fallback_test",
            )

            self.assertIsNotNone(result)
            patched_command, patched_config = result
            self.assertNotEqual(patched_command[4], str(config_path))
            self.assertTrue(patched_config.exists())
            patched_text = patched_config.read_text(encoding="utf-8")
            self.assertIn("name: simple_unet", patched_text)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_recent_plain_simple_unet_streak_counts_tail(self) -> None:
        tmp_root = REPO_ROOT / ".mini_loop" / "test_tmp"
        tmp_root.mkdir(parents=True, exist_ok=True)
        tmp_dir = tmp_root / "streak_case"
        shutil.rmtree(tmp_dir, ignore_errors=True)
        (tmp_dir / "generated_configs").mkdir(parents=True, exist_ok=True)
        try:
            def write_config(name: str, model_name: str) -> str:
                path = tmp_dir / "generated_configs" / name
                path.write_text(f"model:\n  name: {model_name}\n", encoding="utf-8")
                return str(path.relative_to(tmp_dir))

            baseline_cfg = write_config("baseline.yaml", "simple_unet")
            simple_a = write_config("simple_a.yaml", "simple_unet")
            simple_b = write_config("simple_b.yaml", "simple_unet")
            simple_c = write_config("simple_c.yaml", "simple_unet")
            results_path = tmp_dir / "results.tsv"
            results_path.write_text(
                "\n".join(
                    [
                        "experiment_id\tparent_experiment_id\tstatus\truntime_tier\tconfig_path\tresolved_config_path\tcheckpoint_path\tval_metric_key\tval_metric_value\ttrain_seconds\ttotal_seconds\tnotes",
                        f"e0001\t\tbaseline\tmedium\t{baseline_cfg}\t\t\troc_auc_presence\t0.60\t1\t1\tbaseline",
                        f"e0002\te0001\tdiscard\tmedium\t{simple_a}\t\t\troc_auc_presence\t0.59\t1\t1\tnote",
                        f"e0003\te0001\tdiscard\tmedium\t{simple_b}\t\t\troc_auc_presence\t0.58\t1\t1\tnote",
                        f"e0004\te0001\tcrash\tmedium\t{simple_c}\t\t\troc_auc_presence\t\t1\t1\tnote",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            streak = codex_loop.recent_plain_simple_unet_streak(
                repo_root=tmp_dir,
                results_path=results_path,
                tier="medium",
            )
            self.assertEqual(streak, 3)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_execute_wrapper_action_normalizes_escaped_yaml(self) -> None:
        tmp_root = REPO_ROOT / ".mini_loop" / "test_tmp"
        tmp_root.mkdir(parents=True, exist_ok=True)
        tmp_dir = tmp_root / "normalize_case"
        shutil.rmtree(tmp_dir, ignore_errors=True)
        (tmp_dir / "generated_configs").mkdir(parents=True, exist_ok=True)
        (tmp_dir / "logs").mkdir(parents=True, exist_ok=True)
        (tmp_dir / "benchmark" / "src").mkdir(parents=True, exist_ok=True)
        (tmp_dir / "results.tsv").write_text(
            "experiment_id\tparent_experiment_id\tstatus\truntime_tier\tconfig_path\tresolved_config_path\tcheckpoint_path\tval_metric_key\tval_metric_value\ttrain_seconds\ttotal_seconds\tnotes\n",
            encoding="utf-8",
        )
        try:
            with mock.patch.object(codex_loop, "run_logged_command", return_value={"returncode": 0}):
                outcome = codex_loop.execute_wrapper_action(
                    action={
                        "action": "run_config",
                        "label": "escaped_yaml_case",
                        "runtime_tier": "medium",
                        "config_path": "generated_configs/escaped.yaml",
                        "config_yaml": "model:\\n  name: simple_unet\\ntraining:\\n  epochs: 12",
                        "code_edits": [],
                    },
                    repo_root=tmp_dir,
                    app_cfg={
                        "runtime_tiers": {"medium": {}},
                        "results_file": "results.tsv",
                        "selection_metric": "roc_auc_presence",
                    },
                    benchmark_repo_root=tmp_dir / "benchmark",
                    benchmark_python=pathlib.Path("python"),
                    controller_python=pathlib.Path("python"),
                    default_tier="medium",
                    search_space_name="open",
                    cycle_index=1,
                    logs_dir=tmp_dir / "logs",
                    auto_repair_enabled=False,
                    auto_repair_retry_on_success=False,
                    auto_repair_allow_direct_module_install=False,
                    auto_repair_module_package_map={},
                    auto_repair_module_alias_map={},
                    auto_repair_model_package_map={},
                    auto_repair_model_fallback_map={},
                    runtime_env={},
                    same_family_micro_tweak_streak=6,
                    same_family_broad_jump_min_axes=2,
                )
            written = (tmp_dir / "generated_configs" / "escaped.yaml").read_text(encoding="utf-8")
            self.assertEqual(outcome["status"], "executed")
            self.assertIn("model:\n  name: simple_unet\n", written)
            self.assertNotIn("\\n", written)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_execute_wrapper_action_repairs_common_yaml_glitch(self) -> None:
        tmp_root = REPO_ROOT / ".mini_loop" / "test_tmp"
        tmp_root.mkdir(parents=True, exist_ok=True)
        tmp_dir = tmp_root / "repair_case"
        shutil.rmtree(tmp_dir, ignore_errors=True)
        (tmp_dir / "generated_configs").mkdir(parents=True, exist_ok=True)
        (tmp_dir / "logs").mkdir(parents=True, exist_ok=True)
        (tmp_dir / "benchmark" / "src").mkdir(parents=True, exist_ok=True)
        (tmp_dir / "results.tsv").write_text(
            "experiment_id\tparent_experiment_id\tstatus\truntime_tier\tconfig_path\tresolved_config_path\tcheckpoint_path\tval_metric_key\tval_metric_value\ttrain_seconds\ttotal_seconds\tnotes\n",
            encoding="utf-8",
        )
        try:
            with mock.patch.object(codex_loop, "run_logged_command", return_value={"returncode": 0}):
                outcome = codex_loop.execute_wrapper_action(
                    action={
                        "action": "run_config",
                        "label": "repair_yaml_case",
                        "runtime_tier": "medium",
                        "config_path": "generated_configs/repaired.yaml",
                        "config_yaml": (
                            "loss:\n"
                            "  name: bce_dice\n"
                            "n  bce_weight: 0.4\n"
                            "  dice_weight: 0.6\n"
                            "model:\n"
                            "  name: fpn\n"
                        ),
                        "code_edits": [],
                    },
                    repo_root=tmp_dir,
                    app_cfg={
                        "runtime_tiers": {"medium": {}},
                        "results_file": "results.tsv",
                        "selection_metric": "roc_auc_presence",
                    },
                    benchmark_repo_root=tmp_dir / "benchmark",
                    benchmark_python=pathlib.Path("python"),
                    controller_python=pathlib.Path("python"),
                    default_tier="medium",
                    search_space_name="open",
                    cycle_index=1,
                    logs_dir=tmp_dir / "logs",
                    auto_repair_enabled=False,
                    auto_repair_retry_on_success=False,
                    auto_repair_allow_direct_module_install=False,
                    auto_repair_module_package_map={},
                    auto_repair_module_alias_map={},
                    auto_repair_model_package_map={},
                    auto_repair_model_fallback_map={},
                    runtime_env={},
                    same_family_micro_tweak_streak=6,
                    same_family_broad_jump_min_axes=2,
                )
            written = (tmp_dir / "generated_configs" / "repaired.yaml").read_text(encoding="utf-8")
            self.assertEqual(outcome["status"], "executed")
            self.assertIn("  bce_weight: 0.4\n", written)
            self.assertIn("model:\n  name: fpn\n", written)
            self.assertNotIn("\nn  bce_weight", written)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_execute_wrapper_action_blocks_stale_plain_simple_unet_runs(self) -> None:
        tmp_root = REPO_ROOT / ".mini_loop" / "test_tmp"
        tmp_root.mkdir(parents=True, exist_ok=True)
        tmp_dir = tmp_root / "guard_case"
        shutil.rmtree(tmp_dir, ignore_errors=True)
        (tmp_dir / "generated_configs").mkdir(parents=True, exist_ok=True)
        (tmp_dir / "logs").mkdir(parents=True, exist_ok=True)
        (tmp_dir / "benchmark" / "src").mkdir(parents=True, exist_ok=True)
        try:
            def write_config(name: str, model_name: str) -> str:
                path = tmp_dir / "generated_configs" / name
                path.write_text(f"model:\n  name: {model_name}\n", encoding="utf-8")
                return str(path.relative_to(tmp_dir))

            baseline_cfg = write_config("baseline.yaml", "simple_unet")
            simple_a = write_config("simple_a.yaml", "simple_unet")
            simple_b = write_config("simple_b.yaml", "simple_unet")
            simple_c = write_config("simple_c.yaml", "simple_unet")
            (tmp_dir / "results.tsv").write_text(
                "\n".join(
                    [
                        "experiment_id\tparent_experiment_id\tstatus\truntime_tier\tconfig_path\tresolved_config_path\tcheckpoint_path\tval_metric_key\tval_metric_value\ttrain_seconds\ttotal_seconds\tnotes",
                        f"e0001\t\tbaseline\tmedium\t{baseline_cfg}\t\t\troc_auc_presence\t0.60\t1\t1\tbaseline",
                        f"e0002\te0001\tdiscard\tmedium\t{simple_a}\t\t\troc_auc_presence\t0.59\t1\t1\tnote",
                        f"e0003\te0001\tdiscard\tmedium\t{simple_b}\t\t\troc_auc_presence\t0.58\t1\t1\tnote",
                        f"e0004\te0001\tdiscard\tmedium\t{simple_c}\t\t\troc_auc_presence\t0.57\t1\t1\tnote",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(codex_loop, "run_logged_command") as mocked_run:
                with self.assertRaisesRegex(codex_loop.PolicyRejectError, "plain simple_unet search is stuck"):
                    codex_loop.execute_wrapper_action(
                        action={
                            "action": "run_config",
                            "label": "stale_simple_unet",
                            "runtime_tier": "medium",
                            "config_path": "generated_configs/new_simple.yaml",
                            "config_yaml": "model:\n  name: simple_unet\ntraining:\n  epochs: 12\n",
                            "code_edits": [],
                        },
                        repo_root=tmp_dir,
                        app_cfg={
                            "runtime_tiers": {"medium": {}},
                            "results_file": "results.tsv",
                            "selection_metric": "roc_auc_presence",
                        },
                        benchmark_repo_root=tmp_dir / "benchmark",
                        benchmark_python=pathlib.Path("python"),
                        controller_python=pathlib.Path("python"),
                        default_tier="medium",
                        search_space_name="open",
                        cycle_index=1,
                        logs_dir=tmp_dir / "logs",
                        auto_repair_enabled=False,
                        auto_repair_retry_on_success=False,
                        auto_repair_allow_direct_module_install=False,
                        auto_repair_module_package_map={},
                        auto_repair_module_alias_map={},
                        auto_repair_model_package_map={},
                        auto_repair_model_fallback_map={},
                        runtime_env={},
                        same_family_micro_tweak_streak=6,
                        same_family_broad_jump_min_axes=2,
                    )
                mocked_run.assert_not_called()
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_execute_wrapper_action_rejects_same_family_micro_tweak_after_plateau(self) -> None:
        tmp_root = REPO_ROOT / ".mini_loop" / "test_tmp"
        tmp_root.mkdir(parents=True, exist_ok=True)
        tmp_dir = tmp_root / "family_plateau_case"
        shutil.rmtree(tmp_dir, ignore_errors=True)
        (tmp_dir / "generated_configs").mkdir(parents=True, exist_ok=True)
        (tmp_dir / "logs").mkdir(parents=True, exist_ok=True)
        (tmp_dir / "benchmark" / "src").mkdir(parents=True, exist_ok=True)
        try:
            def write_config(name: str, model_name: str) -> str:
                path = tmp_dir / "generated_configs" / name
                path.write_text(f"model:\n  name: {model_name}\n", encoding="utf-8")
                return str(path.relative_to(tmp_dir))

            baseline_cfg = write_config("baseline.yaml", "simple_unet")
            best_cfg = write_config("best.yaml", "deeplabv3plus")
            rows = [
                "experiment_id\tparent_experiment_id\tstatus\truntime_tier\tconfig_path\tresolved_config_path\tcheckpoint_path\tval_metric_key\tval_metric_value\ttrain_seconds\ttotal_seconds\tnotes",
                f"e0001\t\tbaseline\tmedium\t{baseline_cfg}\t\t\troc_auc_presence\t0.60\t1\t1\tbaseline",
                f"e0002\te0001\tkeep\tmedium\t{best_cfg}\t\t\troc_auc_presence\t0.70\t1\t1\tbest",
            ]
            for idx in range(3, 9):
                cfg = write_config(f"discard_{idx}.yaml", "deeplabv3plus")
                rows.append(
                    f"e{idx:04d}\te0002\tdiscard\tmedium\t{cfg}\t\t\troc_auc_presence\t0.69\t1\t1\tnote"
                )
            (tmp_dir / "results.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
            with mock.patch.object(codex_loop, "run_logged_command") as mocked_run:
                with self.assertRaisesRegex(codex_loop.PolicyRejectError, "micro-tweak"):
                    codex_loop.execute_wrapper_action(
                        action={
                            "action": "run_config",
                            "label": "plateau_same_family",
                            "runtime_tier": "medium",
                            "config_path": "generated_configs/new_same_family.yaml",
                            "config_yaml": "model:\n  name: deeplabv3plus\ntraining:\n  epochs: 18\n",
                            "code_edits": [],
                        },
                        repo_root=tmp_dir,
                        app_cfg={
                            "runtime_tiers": {"medium": {}},
                            "results_file": "results.tsv",
                            "selection_metric": "roc_auc_presence",
                        },
                        benchmark_repo_root=tmp_dir / "benchmark",
                        benchmark_python=pathlib.Path("python"),
                        controller_python=pathlib.Path("python"),
                        default_tier="medium",
                        search_space_name="open",
                        cycle_index=1,
                        logs_dir=tmp_dir / "logs",
                        auto_repair_enabled=False,
                        auto_repair_retry_on_success=False,
                        auto_repair_allow_direct_module_install=False,
                        auto_repair_module_package_map={},
                        auto_repair_module_alias_map={},
                        auto_repair_model_package_map={},
                        auto_repair_model_fallback_map={},
                        runtime_env={},
                        same_family_micro_tweak_streak=6,
                        same_family_broad_jump_min_axes=2,
                    )
                mocked_run.assert_not_called()
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_execute_wrapper_action_allows_same_family_broad_jump_after_plateau(self) -> None:
        tmp_root = REPO_ROOT / ".mini_loop" / "test_tmp"
        tmp_root.mkdir(parents=True, exist_ok=True)
        tmp_dir = tmp_root / "family_broad_jump_case"
        shutil.rmtree(tmp_dir, ignore_errors=True)
        (tmp_dir / "generated_configs").mkdir(parents=True, exist_ok=True)
        (tmp_dir / "logs").mkdir(parents=True, exist_ok=True)
        (tmp_dir / "benchmark" / "src").mkdir(parents=True, exist_ok=True)
        try:
            def write_config(name: str, text: str) -> str:
                path = tmp_dir / "generated_configs" / name
                path.write_text(text, encoding="utf-8")
                return str(path.relative_to(tmp_dir))

            baseline_cfg = write_config("baseline.yaml", "model:\n  name: simple_unet\n")
            best_cfg = write_config(
                "best.yaml",
                "input:\n  image_size: 320\ntraining:\n  epochs: 35\nmodel:\n  name: deeplabv3plus\n  backbone: resnet50\n  pretrained: true\n",
            )
            rows = [
                "experiment_id\tparent_experiment_id\tstatus\truntime_tier\tconfig_path\tresolved_config_path\tcheckpoint_path\tval_metric_key\tval_metric_value\ttrain_seconds\ttotal_seconds\tnotes",
                f"e0001\t\tbaseline\tmedium\t{baseline_cfg}\t\t\troc_auc_presence\t0.60\t1\t1\tbaseline",
                f"e0002\te0001\tkeep\tmedium\t{best_cfg}\t\t\troc_auc_presence\t0.70\t1\t1\tbest",
            ]
            for idx in range(3, 9):
                cfg = write_config(
                    f"discard_{idx}.yaml",
                    "input:\n  image_size: 320\ntraining:\n  epochs: 35\nmodel:\n  name: deeplabv3plus\n  backbone: resnet50\n  pretrained: true\n",
                )
                rows.append(
                    f"e{idx:04d}\te0002\tdiscard\tmedium\t{cfg}\t\t\troc_auc_presence\t0.69\t1\t1\tnote"
                )
            (tmp_dir / "results.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
            with mock.patch.object(codex_loop, "run_logged_command", return_value={"returncode": 0}) as mocked_run:
                outcome = codex_loop.execute_wrapper_action(
                    action={
                        "action": "run_config",
                        "label": "plateau_broad_jump",
                        "runtime_tier": "medium",
                        "config_path": "generated_configs/new_same_family.yaml",
                        "config_yaml": (
                            "input:\n  image_size: 384\n"
                            "training:\n  epochs: 50\n"
                            "model:\n  name: deeplabv3plus\n  backbone: resnet50\n  pretrained: false\n"
                        ),
                        "code_edits": [],
                    },
                    repo_root=tmp_dir,
                    app_cfg={
                        "runtime_tiers": {"medium": {}},
                        "results_file": "results.tsv",
                        "selection_metric": "roc_auc_presence",
                    },
                    benchmark_repo_root=tmp_dir / "benchmark",
                    benchmark_python=pathlib.Path("python"),
                    controller_python=pathlib.Path("python"),
                    default_tier="medium",
                    search_space_name="open",
                    cycle_index=1,
                    logs_dir=tmp_dir / "logs",
                    auto_repair_enabled=False,
                    auto_repair_retry_on_success=False,
                    auto_repair_allow_direct_module_install=False,
                    auto_repair_module_package_map={},
                    auto_repair_module_alias_map={},
                    auto_repair_model_package_map={},
                    auto_repair_model_fallback_map={},
                    runtime_env={},
                    same_family_micro_tweak_streak=6,
                    same_family_broad_jump_min_axes=2,
                )
            self.assertEqual(outcome["status"], "executed")
            self.assertEqual(outcome["proposal_kind"], "same_family_broad_jump")
            self.assertIn("input_scale", outcome["changed_axes"])
            self.assertIn("training_budget", outcome["changed_axes"])
            mocked_run.assert_called_once()
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_should_stop_loop_treats_policy_rejected_as_non_terminal(self) -> None:
        self.assertFalse(codex_loop.should_stop_loop(action_kind="run_config", execution_status="policy_rejected"))
        self.assertTrue(codex_loop.should_stop_loop(action_kind="blocked", execution_status="terminal_blocked"))
        self.assertTrue(codex_loop.should_stop_loop(action_kind="run_config", execution_status="infra_blocked"))


if __name__ == "__main__":
    unittest.main()
