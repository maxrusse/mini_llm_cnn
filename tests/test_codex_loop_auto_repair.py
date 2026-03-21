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

    def test_list_allowed_runtime_tiers_uses_code_search_key(self) -> None:
        tiers = codex_loop.list_allowed_runtime_tiers(
            {
                "code_search_runtime_tiers": ["medium", "long", "ghost"],
                "open_search_runtime_tiers": ["smoke"],
            },
            "code",
            {"runtime_tiers": {"medium": {}, "long": {}, "smoke": {}}},
        )
        self.assertEqual(tiers, ["medium", "long"])

    def test_is_context_overflow_error_detects_codex_overflow(self) -> None:
        self.assertTrue(codex_loop.is_context_overflow_error("context_length_exceeded"))
        self.assertTrue(
            codex_loop.is_context_overflow_error(
                "Your input exceeds the context window of this model. Please adjust your input and try again."
            )
        )
        self.assertFalse(codex_loop.is_context_overflow_error("network timeout"))

    def test_reset_codex_thread_state_clears_file_and_logs_reason(self) -> None:
        tmp_root = REPO_ROOT / ".mini_loop" / "test_tmp"
        tmp_root.mkdir(parents=True, exist_ok=True)
        tmp_dir = tmp_root / "thread_reset_case"
        shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            logs_dir = tmp_dir / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            thread_id_file = tmp_dir / "thread_id.txt"
            thread_id_file.write_text("thread-123", encoding="utf-8")

            previous = codex_loop.reset_codex_thread_state(
                thread_id_file=thread_id_file,
                logs_dir=logs_dir,
                reason="context_length_exceeded",
            )

            self.assertEqual(previous, "thread-123")
            self.assertFalse(thread_id_file.exists())
            log_text = (logs_dir / "codex_thread_reset.log").read_text(encoding="utf-8")
            self.assertIn("context_length_exceeded", log_text)
            self.assertIn("thread-123", log_text)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_call_codex_with_fresh_thread_retry_resets_and_retries_once(self) -> None:
        tmp_root = REPO_ROOT / ".mini_loop" / "test_tmp"
        tmp_root.mkdir(parents=True, exist_ok=True)
        tmp_dir = tmp_root / "fresh_thread_retry_case"
        shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            logs_dir = tmp_dir / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            thread_id_file = tmp_dir / "thread_id.txt"
            thread_id_file.write_text("thread-xyz", encoding="utf-8")

            with mock.patch.object(
                codex_loop,
                "call_codex",
                side_effect=[
                    RuntimeError("codex runner failed: context_length_exceeded"),
                    {"returncode": 0, "thread_id": "fresh-thread", "action_payload": {"action": "done"}, "telemetry": {}},
                ],
            ) as mocked_call:
                result = codex_loop.call_codex_with_fresh_thread_retry(
                    codex_exe="codex",
                    repo_root=tmp_dir,
                    codex_home=tmp_dir / "codex_home",
                    thread_id_file=thread_id_file,
                    logs_dir=logs_dir,
                    model="gpt-5.3-codex-spark",
                    reasoning_effort="high",
                    web_search_mode="live",
                    network_access_enabled=True,
                    sandbox_mode="workspace-write",
                    skip_git_repo_check=False,
                    add_dirs=[],
                    prompt="prompt",
                    env={},
                )

            self.assertEqual(mocked_call.call_count, 2)
            self.assertEqual(result["thread_id"], "fresh-thread")
            self.assertTrue(result["telemetry"]["thread_reset_retry"])
            self.assertEqual(result["telemetry"]["thread_reset_reason"], "context_length_exceeded")
            self.assertEqual(result["telemetry"]["previous_thread_id"], "thread-xyz")
            self.assertFalse(thread_id_file.exists())
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_recent_cycles_without_web_search_counts_tail(self) -> None:
        cycles = [
            {"telemetry": {"used_web_search": True}},
            {"telemetry": {"used_web_search": False}},
            {"telemetry": {"used_web_search": False}},
        ]
        self.assertEqual(codex_loop.recent_cycles_without_web_search(cycles), 2)

    def test_build_progress_snapshot_flags_research_refresh(self) -> None:
        tmp_root = REPO_ROOT / ".mini_loop" / "test_tmp"
        tmp_root.mkdir(parents=True, exist_ok=True)
        tmp_dir = tmp_root / "progress_case"
        shutil.rmtree(tmp_dir, ignore_errors=True)
        try:
            results_path = tmp_dir / "results.tsv"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            results_path.write_text(
                "\n".join(
                    [
                        "experiment_id\tparent_experiment_id\tstatus\truntime_tier\tconfig_path\tresolved_config_path\tcheckpoint_path\tval_metric_key\tval_metric_value\ttrain_seconds\ttotal_seconds\tnotes",
                        "e0001\t\tbaseline\tmedium\t\t\t\troc_auc_presence\t0.80\t1\t1\tbaseline",
                        "e0002\te0001\tdiscard\tmedium\t\t\t\troc_auc_presence\t0.79\t1\t1\tnote",
                        "e0003\te0001\tdiscard\tmedium\t\t\t\troc_auc_presence\t0.78\t1\t1\tnote",
                        "e0004\te0001\tdiscard\tmedium\t\t\t\troc_auc_presence\t0.77\t1\t1\tnote",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            progress = codex_loop.build_progress_snapshot(
                repo_root=tmp_dir,
                results_path=results_path,
                tier="medium",
                metric_key="roc_auc_presence",
                session_cycles=[
                    {"telemetry": {"used_web_search": True}},
                    {"telemetry": {"used_web_search": False}},
                    {"telemetry": {"used_web_search": False}},
                ],
                allowed_runtime_tiers=["medium", "long"],
                finalize_runtime_tiers=["long"],
                code_edit_escalation_streak=8,
                research_refresh_streak=2,
            )
            self.assertTrue(progress["research_refresh_due"])
            self.assertEqual(progress["recent_cycles_without_web_search"], 2)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_extract_missing_module(self) -> None:
        text = "Traceback\nModuleNotFoundError: No module named 'segmentation_models_pytorch'\n"
        self.assertEqual(codex_loop.extract_missing_module(text), "segmentation_models_pytorch")

    def test_build_idea_pool_includes_review_alternates_and_excludes_crashes(self) -> None:
        tmp_root = REPO_ROOT / ".mini_loop" / "test_tmp"
        tmp_root.mkdir(parents=True, exist_ok=True)
        tmp_dir = tmp_root / "idea_pool_case"
        shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            results_path = tmp_dir / "results.tsv"
            summary_path = tmp_dir / "experiment_summary.tsv"
            results_path.write_text(
                "\n".join(
                    [
                        "experiment_id\tparent_experiment_id\tstatus\truntime_tier\tconfig_path\tresolved_config_path\tcheckpoint_path\tval_metric_key\tval_metric_value\ttrain_seconds\ttotal_seconds\tnotes",
                        "e0001\t\tbaseline\tmedium\t\t\t\troc_auc_presence\t0.70\t1\t1\tbaseline",
                        "e0002\te0001\tkeep\tmedium\t\t\t\troc_auc_presence\t0.91\t1\t1\tbest",
                        "e0003\te0002\tdiscard\tmedium\t\t\t\troc_auc_presence\t0.90\t1\t1\treview-worthy alternate vs e0002",
                        "e0004\te0002\tcrash\tmedium\t\t\t\troc_auc_presence\t\t1\t1\tboom",
                        "e0005\te0002\tkeep\tlong\t\t\t\troc_auc_presence\t0.99\t1\t1\twrong tier",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            summary_path.write_text(
                "\n".join(
                    [
                        "experiment_id\tparent_experiment_id\tstatus\truntime_tier\tval_metric_key\tval_metric_value\ttrain_seconds\ttotal_seconds\tmodel_name\tmodel_backbone\tmodel_pretrained\tmodel_pretrained_backbone\tmodel_base_channels\tmodel_cls_hidden\tmodel_cls_dropout\tinput_image_size\tinput_preserve_aspect\tbatch_size\tepochs\tlearning_rate\tmin_lr\tweight_decay\tscheduler\taugment\tbalanced_sampling\tpatch_enabled\tpatch_size\tpatch_positive_prob\tpatch_hard_negative_prob\tpatch_hard_negative_quantile\tloss_name\tbce_weight\tdice_weight\tdice_positive_only\tpresence_bce_weight\tpresence_bce_warmup_epochs\teval_threshold\teval_tta\teval_presence_score_mode\teval_presence_topk_frac\teval_presence_threshold\truntime_device\truntime_amp\tconfig_path\tresolved_config_path\tnotes",
                        "e0001\t\tbaseline\tmedium\troc_auc_presence\t0.70\t1\t1\tsimple_unet\t\t\t\t\t\t\t256\t\t2\t12\t\t\t\t\t\t\tfalse\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tbaseline",
                        "e0002\te0001\tkeep\tmedium\troc_auc_presence\t0.91\t1\t1\tfpn\tresnet50\t\t\t\t\t\t320\t\t2\t35\t\t\t\tcosine\t\t\ttrue\t\t\t\t\t\t\t\t\t0.4\t\t\t\t\t\t\t\t\t\t\tbest",
                        "e0003\te0002\tdiscard\tmedium\troc_auc_presence\t0.90\t1\t1\tpspnet\tresnet101\t\t\t\t\t\t416\t\t1\t50\t\t\t\tcosine\t\t\ttrue\t\t\t\t\t\t\t\t\t0.6\t\t\t\t\t\t\t\t\t\t\treview",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            pool = codex_loop.build_idea_pool(
                results_path=results_path,
                summary_path=summary_path,
                tier="medium",
                metric_key="roc_auc_presence",
                max_items=2,
            )
            self.assertEqual([item["experiment_id"] for item in pool], ["e0002", "e0001"])
            expanded = codex_loop.build_idea_pool(
                results_path=results_path,
                summary_path=summary_path,
                tier="medium",
                metric_key="roc_auc_presence",
                max_items=5,
            )
            self.assertEqual([item["experiment_id"] for item in expanded], ["e0002", "e0001", "e0003"])
            self.assertNotIn("e0004", [item["experiment_id"] for item in expanded])
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_extract_unsupported_model_name(self) -> None:
        text = "ValueError: Unsupported model.name: unet_resnet34"
        self.assertEqual(codex_loop.extract_unsupported_model_name(text), "unet_resnet34")

    def test_normalize_config_yaml_text_unescapes_newlines(self) -> None:
        raw = "model:\\n  name: simple_unet\\ntraining:\\n  epochs: 12"
        normalized = codex_loop.normalize_config_yaml_text(raw)
        self.assertIn("model:\n  name: simple_unet", normalized)
        self.assertNotIn("\\n", normalized)

    def test_canonicalize_generated_config_path_rewrites_open_dir_for_code_flow(self) -> None:
        rewritten = codex_loop.canonicalize_generated_config_path(
            "generated_configs/sample.yaml",
            "generated_configs_code",
        )
        self.assertEqual(rewritten, str(pathlib.Path("generated_configs_code") / "sample.yaml"))

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

    def test_execute_wrapper_action_code_flow_allows_seed_control_without_code_edits(self) -> None:
        tmp_root = REPO_ROOT / ".mini_loop" / "test_tmp"
        tmp_root.mkdir(parents=True, exist_ok=True)
        tmp_dir = tmp_root / "code_seed_case"
        shutil.rmtree(tmp_dir, ignore_errors=True)
        (tmp_dir / "generated_configs_code").mkdir(parents=True, exist_ok=True)
        (tmp_dir / "logs_code").mkdir(parents=True, exist_ok=True)
        (tmp_dir / "benchmark" / "src").mkdir(parents=True, exist_ok=True)
        (tmp_dir / "results_code.tsv").write_text(
            "experiment_id\tparent_experiment_id\tstatus\truntime_tier\tconfig_path\tresolved_config_path\tcheckpoint_path\tval_metric_key\tval_metric_value\ttrain_seconds\ttotal_seconds\tnotes\n",
            encoding="utf-8",
        )
        try:
            with mock.patch.object(codex_loop, "run_logged_command", return_value={"returncode": 0}) as mocked_run:
                outcome = codex_loop.execute_wrapper_action(
                    action={
                        "action": "run_config",
                        "label": "code_seed",
                        "runtime_tier": "medium",
                        "config_path": "generated_configs_code/seed.yaml",
                        "config_yaml": "model:\n  name: fpn\n",
                        "code_edits": [],
                    },
                    repo_root=tmp_dir,
                    app_cfg={
                        "runtime_tiers": {"medium": {}},
                        "results_file": "results_code.tsv",
                        "generated_config_dir": "generated_configs_code",
                        "downloads_dir": "downloads_code",
                        "settings_path": "config_code.json",
                        "selection_metric": "roc_auc_presence",
                    },
                    benchmark_repo_root=tmp_dir / "benchmark",
                    benchmark_python=pathlib.Path("python"),
                    controller_python=pathlib.Path("python"),
                    default_tier="medium",
                    search_space_name="code",
                    cycle_index=1,
                    logs_dir=tmp_dir / "logs_code",
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
            self.assertTrue((tmp_dir / "generated_configs_code" / "seed.yaml").exists())
            command = mocked_run.call_args.kwargs["cmd"]
            self.assertIn(str((tmp_dir / "config_code.json").resolve()), command)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_execute_wrapper_action_code_flow_rewrites_generated_configs_prefix(self) -> None:
        tmp_root = REPO_ROOT / ".mini_loop" / "test_tmp"
        tmp_root.mkdir(parents=True, exist_ok=True)
        tmp_dir = tmp_root / "code_rewrite_case"
        shutil.rmtree(tmp_dir, ignore_errors=True)
        (tmp_dir / "generated_configs_code").mkdir(parents=True, exist_ok=True)
        (tmp_dir / "logs_code").mkdir(parents=True, exist_ok=True)
        (tmp_dir / "benchmark" / "src").mkdir(parents=True, exist_ok=True)
        (tmp_dir / "results_code.tsv").write_text(
            "experiment_id\tparent_experiment_id\tstatus\truntime_tier\tconfig_path\tresolved_config_path\tcheckpoint_path\tval_metric_key\tval_metric_value\ttrain_seconds\ttotal_seconds\tnotes\n",
            encoding="utf-8",
        )
        try:
            with mock.patch.object(codex_loop, "run_logged_command", return_value={"returncode": 0}) as mocked_run:
                outcome = codex_loop.execute_wrapper_action(
                    action={
                        "action": "run_config",
                        "label": "code_seed",
                        "runtime_tier": "medium",
                        "config_path": "generated_configs/seed.yaml",
                        "config_yaml": "model:\n  name: fpn\n",
                        "code_edits": [],
                    },
                    repo_root=tmp_dir,
                    app_cfg={
                        "runtime_tiers": {"medium": {}},
                        "results_file": "results_code.tsv",
                        "generated_config_dir": "generated_configs_code",
                        "downloads_dir": "downloads_code",
                        "settings_path": "config_code.json",
                        "selection_metric": "roc_auc_presence",
                    },
                    benchmark_repo_root=tmp_dir / "benchmark",
                    benchmark_python=pathlib.Path("python"),
                    controller_python=pathlib.Path("python"),
                    default_tier="medium",
                    search_space_name="code",
                    cycle_index=1,
                    logs_dir=tmp_dir / "logs_code",
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
            self.assertTrue((tmp_dir / "generated_configs_code" / "seed.yaml").exists())
            command = mocked_run.call_args.kwargs["cmd"]
            self.assertIn(str((tmp_dir / "generated_configs_code" / "seed.yaml").resolve()), command)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_execute_wrapper_action_code_flow_rejects_config_only_after_seed(self) -> None:
        tmp_root = REPO_ROOT / ".mini_loop" / "test_tmp"
        tmp_root.mkdir(parents=True, exist_ok=True)
        tmp_dir = tmp_root / "code_reject_case"
        shutil.rmtree(tmp_dir, ignore_errors=True)
        (tmp_dir / "generated_configs_code").mkdir(parents=True, exist_ok=True)
        (tmp_dir / "logs_code").mkdir(parents=True, exist_ok=True)
        (tmp_dir / "benchmark" / "src").mkdir(parents=True, exist_ok=True)
        (tmp_dir / "results_code.tsv").write_text(
            "\n".join(
                [
                    "experiment_id\tparent_experiment_id\tstatus\truntime_tier\tconfig_path\tresolved_config_path\tcheckpoint_path\tval_metric_key\tval_metric_value\ttrain_seconds\ttotal_seconds\tnotes",
                    "e0001\t\tbaseline\tmedium\tgenerated_configs_code/seed.yaml\t\t\troc_auc_presence\t0.70\t1\t1\tbaseline",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        try:
            with mock.patch.object(codex_loop, "run_logged_command") as mocked_run:
                with self.assertRaisesRegex(codex_loop.PolicyRejectError, "code-flow is code-first"):
                    codex_loop.execute_wrapper_action(
                        action={
                            "action": "run_config",
                            "label": "code_retry",
                            "runtime_tier": "medium",
                            "config_path": "generated_configs_code/retry.yaml",
                            "config_yaml": "model:\n  name: fpn\n",
                            "code_edits": [],
                        },
                        repo_root=tmp_dir,
                        app_cfg={
                            "runtime_tiers": {"medium": {}},
                            "results_file": "results_code.tsv",
                            "generated_config_dir": "generated_configs_code",
                            "downloads_dir": "downloads_code",
                            "settings_path": "config_code.json",
                            "selection_metric": "roc_auc_presence",
                        },
                        benchmark_repo_root=tmp_dir / "benchmark",
                        benchmark_python=pathlib.Path("python"),
                        controller_python=pathlib.Path("python"),
                        default_tier="medium",
                        search_space_name="code",
                        cycle_index=2,
                        logs_dir=tmp_dir / "logs_code",
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

    def test_execute_wrapper_action_code_flow_accepts_code_edits_after_seed(self) -> None:
        tmp_root = REPO_ROOT / ".mini_loop" / "test_tmp"
        tmp_root.mkdir(parents=True, exist_ok=True)
        tmp_dir = tmp_root / "code_edit_case"
        shutil.rmtree(tmp_dir, ignore_errors=True)
        (tmp_dir / "generated_configs_code").mkdir(parents=True, exist_ok=True)
        (tmp_dir / "logs_code").mkdir(parents=True, exist_ok=True)
        target = tmp_dir / "benchmark" / "src" / "demo.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("VALUE = 1\n", encoding="utf-8")
        (tmp_dir / "results_code.tsv").write_text(
            "\n".join(
                [
                    "experiment_id\tparent_experiment_id\tstatus\truntime_tier\tconfig_path\tresolved_config_path\tcheckpoint_path\tval_metric_key\tval_metric_value\ttrain_seconds\ttotal_seconds\tnotes",
                    "e0001\t\tkeep\tmedium\tgenerated_configs_code/seed.yaml\t\t\troc_auc_presence\t0.80\t1\t1\tseed",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        try:
            with mock.patch.object(codex_loop, "run_logged_command", return_value={"returncode": 0}) as mocked_run:
                outcome = codex_loop.execute_wrapper_action(
                    action={
                        "action": "run_config",
                        "label": "code_edit_followup",
                        "runtime_tier": "medium",
                        "config_path": "generated_configs_code/followup.yaml",
                        "config_yaml": "model:\n  name: fpn\n",
                        "parent_experiment_id": "e0001",
                        "code_edits": [{"path": "src/demo.py", "content": "VALUE = 2\n"}],
                    },
                    repo_root=tmp_dir,
                    app_cfg={
                        "runtime_tiers": {"medium": {}},
                        "results_file": "results_code.tsv",
                        "generated_config_dir": "generated_configs_code",
                        "downloads_dir": "downloads_code",
                        "settings_path": "config_code.json",
                        "selection_metric": "roc_auc_presence",
                    },
                    benchmark_repo_root=tmp_dir / "benchmark",
                    benchmark_python=pathlib.Path("python"),
                    controller_python=pathlib.Path("python"),
                    default_tier="medium",
                    search_space_name="code",
                    cycle_index=2,
                    logs_dir=tmp_dir / "logs_code",
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
            self.assertEqual(target.read_text(encoding="utf-8"), "VALUE = 2\n")
            self.assertEqual(outcome["code_edit_paths"], ["src\\demo.py"])
            mocked_run.assert_called_once()
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_execute_wrapper_action_test_uses_settings_path(self) -> None:
        tmp_root = REPO_ROOT / ".mini_loop" / "test_tmp"
        tmp_root.mkdir(parents=True, exist_ok=True)
        tmp_dir = tmp_root / "test_settings_case"
        shutil.rmtree(tmp_dir, ignore_errors=True)
        (tmp_dir / "logs_code").mkdir(parents=True, exist_ok=True)
        try:
            with mock.patch.object(codex_loop, "run_logged_command", return_value={"returncode": 0}) as mocked_run:
                outcome = codex_loop.execute_wrapper_action(
                    action={
                        "action": "test",
                        "experiment_id": "e0018",
                    },
                    repo_root=tmp_dir,
                    app_cfg={
                        "runtime_tiers": {"medium": {}},
                        "results_file": "results_code.tsv",
                        "generated_config_dir": "generated_configs_code",
                        "downloads_dir": "downloads_code",
                        "settings_path": "config_code.json",
                        "selection_metric": "roc_auc_presence",
                    },
                    benchmark_repo_root=tmp_dir / "benchmark",
                    benchmark_python=pathlib.Path("python"),
                    controller_python=pathlib.Path("python"),
                    default_tier="medium",
                    search_space_name="code",
                    cycle_index=3,
                    logs_dir=tmp_dir / "logs_code",
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
            command = mocked_run.call_args.kwargs["cmd"]
            self.assertEqual(command[2], "--settings-path")
            self.assertIn(str((tmp_dir / "config_code.json").resolve()), command)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_should_stop_loop_treats_policy_rejected_as_non_terminal(self) -> None:
        self.assertFalse(codex_loop.should_stop_loop(action_kind="run_config", execution_status="policy_rejected"))
        self.assertTrue(codex_loop.should_stop_loop(action_kind="blocked", execution_status="terminal_blocked"))
        self.assertTrue(codex_loop.should_stop_loop(action_kind="run_config", execution_status="infra_blocked"))


if __name__ == "__main__":
    unittest.main()
