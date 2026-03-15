from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import pathlib
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any, Callable

try:
    import yaml
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        "PyYAML is required. Run this script with the benchmark interpreter from config.json."
    ) from exc


REPO_ROOT = pathlib.Path(__file__).resolve().parent
CONFIG_PATH = REPO_ROOT / "config.json"
RESULT_FIELDS = [
    "experiment_id",
    "parent_experiment_id",
    "status",
    "runtime_tier",
    "config_path",
    "resolved_config_path",
    "checkpoint_path",
    "val_metric_key",
    "val_metric_value",
    "train_seconds",
    "total_seconds",
    "notes",
]
SUMMARY_FIELDS = [
    "experiment_id",
    "parent_experiment_id",
    "status",
    "runtime_tier",
    "val_metric_key",
    "val_metric_value",
    "train_seconds",
    "total_seconds",
    "model_name",
    "model_backbone",
    "model_pretrained",
    "model_pretrained_backbone",
    "model_base_channels",
    "model_cls_hidden",
    "model_cls_dropout",
    "input_image_size",
    "input_preserve_aspect",
    "batch_size",
    "epochs",
    "learning_rate",
    "min_lr",
    "weight_decay",
    "scheduler",
    "augment",
    "balanced_sampling",
    "patch_enabled",
    "patch_size",
    "patch_positive_prob",
    "patch_hard_negative_prob",
    "patch_hard_negative_quantile",
    "loss_name",
    "bce_weight",
    "dice_weight",
    "dice_positive_only",
    "presence_bce_weight",
    "presence_bce_warmup_epochs",
    "eval_threshold",
    "eval_tta",
    "eval_presence_score_mode",
    "eval_presence_topk_frac",
    "eval_presence_threshold",
    "runtime_device",
    "runtime_amp",
    "config_path",
    "resolved_config_path",
    "notes",
]
KEEP_STATUSES = {"baseline", "keep"}
TESTABLE_STATUSES = {"baseline", "keep", "candidate"}


class LoopError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: pathlib.Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: pathlib.Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def load_yaml(path: pathlib.Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise LoopError(f"Expected mapping YAML: {path}")
    return data


def save_yaml(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    txt = yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)
    path.write_text(txt, encoding="utf-8")


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def slugify(text: str) -> str:
    cleaned = []
    for ch in str(text).strip().lower():
        if ch.isalnum():
            cleaned.append(ch)
        else:
            cleaned.append("_")
    out = "".join(cleaned).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return out or "candidate"


def append_note(base_note: str, extra_note: str) -> str:
    base = str(base_note).strip().strip(";")
    extra = str(extra_note).strip().strip(";")
    if not extra:
        return base
    if not base:
        return extra
    return f"{base}; {extra}"


def selection_metric_priority(settings: dict[str, Any]) -> list[str]:
    configured = settings.get("selection_metric_priority", [])
    metric_keys: list[str] = []
    if isinstance(configured, list):
        for item in configured:
            key = str(item).strip()
            if key and key not in metric_keys:
                metric_keys.append(key)
    primary = str(settings.get("selection_metric", "roc_auc_presence")).strip()
    if primary and primary not in metric_keys:
        metric_keys.insert(0, primary)
    return metric_keys or ["roc_auc_presence"]


def metric_stack_label(metric_keys: list[str]) -> str:
    return " > ".join(metric_keys)


def load_metric_bundle(metrics_path: pathlib.Path, metric_keys: list[str]) -> dict[str, float]:
    metrics = load_json(metrics_path, None)
    if not isinstance(metrics, dict):
        raise LoopError(f"Invalid metrics JSON: {metrics_path}")
    out: dict[str, float] = {}
    for key in metric_keys:
        if key not in metrics:
            raise LoopError(f"Missing metric '{key}' in {metrics_path}")
        value = float(metrics[key])
        if math.isnan(value):
            raise LoopError(f"Metric '{key}' is NaN in {metrics_path}")
        out[key] = value
    return out


def compare_metric_priority(
    current: dict[str, float],
    reference: dict[str, float],
    metric_keys: list[str],
    epsilon: float,
) -> tuple[int, str]:
    for key in metric_keys:
        lhs = float(current[key])
        rhs = float(reference[key])
        if lhs > rhs + epsilon:
            return 1, key
        if lhs < rhs - epsilon:
            return -1, key
    return 0, metric_keys[0] if metric_keys else ""


def metrics_path_for_result_row(row: dict[str, str]) -> pathlib.Path | None:
    checkpoint_txt = str(row.get("checkpoint_path", "")).strip()
    if checkpoint_txt:
        checkpoint_path = resolve_from_repo(checkpoint_txt)
        return checkpoint_path.parent / "validate_metrics.json"
    resolved_cfg_txt = str(row.get("resolved_config_path", "")).strip()
    if resolved_cfg_txt:
        resolved_cfg_path = resolve_from_repo(resolved_cfg_txt)
        return resolved_cfg_path.parent / "validate_metrics.json"
    return None


def annotate_metric_outcome_note(
    *,
    note: str,
    selection_metric: str,
    metric_value: float,
    prior_best: dict[str, str] | None,
    epsilon: float,
) -> str:
    if prior_best is None:
        return note
    best_id = str(prior_best.get("experiment_id", "")).strip() or "current_best"
    try:
        best_value = float(prior_best.get("val_metric_value", "nan"))
    except Exception:
        return append_note(note, f"did not improve over {best_id}")
    if math.isclose(metric_value, best_value, rel_tol=0.0, abs_tol=epsilon):
        return append_note(
            note,
            f"matched current best {best_id} {selection_metric}={best_value:.6f}",
        )
    return append_note(
        note,
        f"did not improve over {best_id} {selection_metric}={best_value:.6f}",
    )


def annotate_ranked_outcome_note(
    *,
    note: str,
    primary_metric: str,
    metric_keys: list[str],
    prior_best: dict[str, str] | None,
    prior_metrics: dict[str, float] | None,
    comparison: int,
) -> str:
    if prior_best is None or prior_metrics is None:
        return note
    best_id = str(prior_best.get("experiment_id", "")).strip() or "current_best"
    primary_best = float(prior_metrics[primary_metric])
    if comparison == 0:
        return append_note(
            note,
            f"matched current best {best_id} on metric stack {metric_stack_label(metric_keys)}; "
            f"{primary_metric}={primary_best:.6f}",
        )
    return append_note(
        note,
        f"did not improve over {best_id} on metric stack {metric_stack_label(metric_keys)}; "
        f"{primary_metric}={primary_best:.6f}",
    )


def is_near_best_candidate(
    current: dict[str, float],
    reference: dict[str, float],
    metric_keys: list[str],
    candidate_epsilon: float,
) -> bool:
    if candidate_epsilon <= 0.0 or not metric_keys:
        return False
    primary = metric_keys[0]
    current_primary = float(current[primary])
    reference_primary = float(reference[primary])
    if current_primary + candidate_epsilon < reference_primary:
        return False
    return True


def annotate_candidate_outcome_note(
    *,
    note: str,
    primary_metric: str,
    prior_best: dict[str, str] | None,
    prior_metrics: dict[str, float] | None,
    candidate_epsilon: float,
) -> str:
    if prior_best is None or prior_metrics is None:
        return note
    best_id = str(prior_best.get("experiment_id", "")).strip() or "current_best"
    best_value = float(prior_metrics[primary_metric])
    return append_note(
        note,
        f"near-best candidate within noise band {candidate_epsilon:.6f} of {best_id}; "
        f"{primary_metric}={best_value:.6f}",
    )


def rel_or_abs(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal config-only loop for xray_fracture_benchmark.")
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status", help="Show current loop status.")
    status.add_argument("--limit", type=int, default=5)

    sub.add_parser("list-mutations", help="List built-in mutation operators.")

    baseline = sub.add_parser("baseline", help="Run baseline for a tier.")
    baseline.add_argument("--tier", default="")
    baseline.add_argument("--note-suffix", default="")
    baseline.add_argument("--dry-run", action="store_true")

    step = sub.add_parser("step", help="Create and run the next auto mutation.")
    step.add_argument("--tier", default="")
    step.add_argument("--mutation", default="auto")
    step.add_argument("--dry-run", action="store_true")

    run_config = sub.add_parser("run-config", help="Run an explicit config file.")
    run_config.add_argument("--config", required=True)
    run_config.add_argument("--tier", default="")
    run_config.add_argument("--label", default="manual")
    run_config.add_argument("--parent-experiment-id", default="")
    run_config.add_argument("--note-suffix", default="")
    run_config.add_argument("--dry-run", action="store_true")

    night = sub.add_parser("night-run", help="Run repeated auto steps for a fixed number of hours.")
    night.add_argument("--tier", default="")
    night.add_argument("--hours", type=float, default=8.0)
    night.add_argument("--sleep-seconds", type=int, default=5)
    night.add_argument("--dry-run", action="store_true")

    test = sub.add_parser("test", help="Run locked test evaluation for a kept or baseline experiment.")
    test.add_argument("--experiment-id", required=True)
    test.add_argument("--dry-run", action="store_true")

    return parser.parse_args()


def load_settings() -> dict[str, Any]:
    data = load_json(CONFIG_PATH, None)
    if not isinstance(data, dict):
        raise LoopError(f"Missing or invalid config.json: {CONFIG_PATH}")
    return data


def resolve_from_repo(raw_path: str) -> pathlib.Path:
    p = pathlib.Path(raw_path)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def ensure_settings_paths(settings: dict[str, Any]) -> dict[str, pathlib.Path]:
    paths = {
        "benchmark_repo_root": resolve_from_repo(str(settings["benchmark_repo_root"])),
        "benchmark_python_exe": resolve_from_repo(str(settings["benchmark_python_exe"])),
        "results_file": resolve_from_repo(str(settings.get("results_file", "results.tsv"))),
        "summary_file": resolve_from_repo(str(settings.get("summary_file", "experiment_summary.tsv"))),
        "state_file": resolve_from_repo(str(settings.get("state_file", ".mini_loop/state.json"))),
        "generated_config_dir": resolve_from_repo(str(settings.get("generated_config_dir", "generated_configs"))),
        "logs_dir": resolve_from_repo(str(settings.get("logs_dir", "logs"))),
        "runs_dir": resolve_from_repo(str(settings.get("runs_dir", "runs"))),
    }
    for key in ("benchmark_repo_root", "benchmark_python_exe"):
        if not paths[key].exists():
            raise LoopError(f"Configured path does not exist for {key}: {paths[key]}")
    paths["generated_config_dir"].mkdir(parents=True, exist_ok=True)
    paths["logs_dir"].mkdir(parents=True, exist_ok=True)
    paths["runs_dir"].mkdir(parents=True, exist_ok=True)
    return paths


def load_state(state_path: pathlib.Path) -> dict[str, Any]:
    default = {
        "created_utc": utc_now(),
        "next_experiment_number": 1,
        "best_by_tier": {},
        "history": [],
        "test_evaluations": {},
    }
    data = load_json(state_path, default)
    if not isinstance(data, dict):
        return default
    for key, value in default.items():
        data.setdefault(key, copy.deepcopy(value))
    return data


def save_state(state_path: pathlib.Path, state: dict[str, Any]) -> None:
    save_json(state_path, state)


def ensure_results_header(path: pathlib.Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS, delimiter="\t")
        writer.writeheader()


def read_results(path: pathlib.Path, *, create: bool = False) -> list[dict[str, str]]:
    if create:
        ensure_results_header(path)
    elif not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return [dict(row) for row in reader]


def append_result(path: pathlib.Path, row: dict[str, Any]) -> None:
    ensure_results_header(path)
    cooked = {field: str(row.get(field, "")) for field in RESULT_FIELDS}
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS, delimiter="\t")
        writer.writerow(cooked)


def _nested_get(mapping: dict[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def extract_summary_row(result_row: dict[str, str], resolved_config: dict[str, Any] | None) -> dict[str, str]:
    cfg = resolved_config if isinstance(resolved_config, dict) else {}
    model_name = _stringify(_nested_get(cfg, "model", "name"))
    model_backbone = _stringify(
        _nested_get(cfg, "model", "backbone")
        or _nested_get(cfg, "model", "encoder_name")
        or _nested_get(cfg, "model", "backbone_name")
    )
    scheduler = _stringify(
        _nested_get(cfg, "training", "scheduler")
        or _nested_get(cfg, "scheduler", "name")
        or _nested_get(cfg, "training", "scheduler_name")
    )
    return {
        "experiment_id": _stringify(result_row.get("experiment_id")),
        "parent_experiment_id": _stringify(result_row.get("parent_experiment_id")),
        "status": _stringify(result_row.get("status")),
        "runtime_tier": _stringify(result_row.get("runtime_tier")),
        "val_metric_key": _stringify(result_row.get("val_metric_key")),
        "val_metric_value": _stringify(result_row.get("val_metric_value")),
        "train_seconds": _stringify(result_row.get("train_seconds")),
        "total_seconds": _stringify(result_row.get("total_seconds")),
        "model_name": model_name,
        "model_backbone": model_backbone,
        "model_pretrained": _stringify(_nested_get(cfg, "model", "pretrained")),
        "model_pretrained_backbone": _stringify(_nested_get(cfg, "model", "pretrained_backbone")),
        "model_base_channels": _stringify(_nested_get(cfg, "model", "base_channels")),
        "model_cls_hidden": _stringify(_nested_get(cfg, "model", "cls_hidden")),
        "model_cls_dropout": _stringify(_nested_get(cfg, "model", "cls_dropout")),
        "input_image_size": _stringify(_nested_get(cfg, "input", "image_size")),
        "input_preserve_aspect": _stringify(_nested_get(cfg, "input", "preserve_aspect")),
        "batch_size": _stringify(_nested_get(cfg, "training", "batch_size")),
        "epochs": _stringify(_nested_get(cfg, "training", "epochs")),
        "learning_rate": _stringify(_nested_get(cfg, "training", "learning_rate")),
        "min_lr": _stringify(_nested_get(cfg, "training", "min_lr")),
        "weight_decay": _stringify(_nested_get(cfg, "training", "weight_decay")),
        "scheduler": scheduler,
        "augment": _stringify(_nested_get(cfg, "training", "augment")),
        "balanced_sampling": _stringify(_nested_get(cfg, "training", "balanced_sampling")),
        "patch_enabled": _stringify(_nested_get(cfg, "training", "patch", "enabled")),
        "patch_size": _stringify(_nested_get(cfg, "training", "patch", "size")),
        "patch_positive_prob": _stringify(_nested_get(cfg, "training", "patch", "positive_prob")),
        "patch_hard_negative_prob": _stringify(_nested_get(cfg, "training", "patch", "hard_negative_prob")),
        "patch_hard_negative_quantile": _stringify(_nested_get(cfg, "training", "patch", "hard_negative_quantile")),
        "loss_name": _stringify(_nested_get(cfg, "loss", "name")),
        "bce_weight": _stringify(_nested_get(cfg, "loss", "bce_weight")),
        "dice_weight": _stringify(_nested_get(cfg, "loss", "dice_weight")),
        "dice_positive_only": _stringify(_nested_get(cfg, "loss", "dice_positive_only")),
        "presence_bce_weight": _stringify(_nested_get(cfg, "loss", "presence_bce_weight")),
        "presence_bce_warmup_epochs": _stringify(_nested_get(cfg, "loss", "presence_bce_warmup_epochs")),
        "eval_threshold": _stringify(_nested_get(cfg, "evaluation", "threshold")),
        "eval_tta": _stringify(_nested_get(cfg, "evaluation", "tta")),
        "eval_presence_score_mode": _stringify(_nested_get(cfg, "evaluation", "presence_score_mode")),
        "eval_presence_topk_frac": _stringify(_nested_get(cfg, "evaluation", "presence_topk_frac")),
        "eval_presence_threshold": _stringify(_nested_get(cfg, "evaluation", "presence_threshold")),
        "runtime_device": _stringify(_nested_get(cfg, "runtime", "device")),
        "runtime_amp": _stringify(_nested_get(cfg, "runtime", "amp")),
        "config_path": _stringify(result_row.get("config_path")),
        "resolved_config_path": _stringify(result_row.get("resolved_config_path")),
        "notes": _stringify(result_row.get("notes")),
    }


def refresh_experiment_summary(paths: dict[str, pathlib.Path]) -> None:
    results = read_results(paths["results_file"])
    summary_path = paths["summary_file"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, delimiter="\t")
        writer.writeheader()
        for row in results:
            resolved_cfg_path_txt = str(row.get("resolved_config_path", "")).strip()
            resolved_cfg: dict[str, Any] | None = None
            if resolved_cfg_path_txt:
                resolved_cfg_path = resolve_from_repo(resolved_cfg_path_txt)
                if resolved_cfg_path.exists():
                    try:
                        resolved_cfg = load_yaml(resolved_cfg_path)
                    except Exception:
                        resolved_cfg = None
            writer.writerow(extract_summary_row(row, resolved_cfg))


def benchmark_config_path(settings: dict[str, Any], bench_root: pathlib.Path) -> pathlib.Path:
    raw = str(settings["baseline_config"])
    p = pathlib.Path(raw)
    if p.is_absolute():
        return p
    return (bench_root / p).resolve()


def make_experiment_id(state: dict[str, Any]) -> str:
    num = int(state.get("next_experiment_number", 1))
    state["next_experiment_number"] = num + 1
    return f"e{num:04d}"


def mutation_history_for_parent(state: dict[str, Any], parent_experiment_id: str, tier: str) -> set[str]:
    out: set[str] = set()
    for item in state.get("history", []):
        if str(item.get("parent_experiment_id", "")) == parent_experiment_id and str(item.get("runtime_tier", "")) == tier:
            label = str(item.get("mutation_label", "")).strip()
            if label:
                out.add(label)
    return out


def current_best_result(
    results: list[dict[str, str]],
    tier: str,
    selection_metric: str,
    *,
    metric_keys: list[str] | None = None,
) -> dict[str, str] | None:
    best_row = None
    best_metrics: dict[str, float] | None = None
    metric_stack = metric_keys or [selection_metric]
    for row in results:
        if row.get("runtime_tier") != tier:
            continue
        if row.get("status") not in KEEP_STATUSES:
            continue
        try:
            metrics_path = metrics_path_for_result_row(row)
            if metrics_path is None or not metrics_path.exists():
                raise LoopError("missing validate metrics")
            row_metrics = load_metric_bundle(metrics_path, metric_stack)
        except Exception:
            if row.get("val_metric_key") != selection_metric:
                continue
            try:
                fallback_value = float(row.get("val_metric_value", "nan"))
            except ValueError:
                continue
            if math.isnan(fallback_value):
                continue
            row_metrics = {selection_metric: fallback_value}
            metric_stack = [selection_metric]
        if best_metrics is None:
            best_row = row
            best_metrics = row_metrics
            continue
        comparison, _ = compare_metric_priority(row_metrics, best_metrics, list(row_metrics.keys()), 0.0)
        if comparison > 0:
            best_row = row
            best_metrics = row_metrics
    return best_row


def load_parent_config(
    *,
    tier: str,
    results: list[dict[str, str]],
    settings: dict[str, Any],
    bench_root: pathlib.Path,
) -> tuple[dict[str, Any], str, pathlib.Path]:
    selection_metric = str(settings["selection_metric"])
    best_row = current_best_result(
        results,
        tier,
        selection_metric,
        metric_keys=selection_metric_priority(settings),
    )
    if best_row is not None:
        config_path = resolve_from_repo(best_row["config_path"])
        return load_yaml(config_path), best_row["experiment_id"], config_path
    baseline_path = benchmark_config_path(settings, bench_root)
    return load_yaml(baseline_path), "", baseline_path


def runtime_tier_name(args_tier: str, settings: dict[str, Any]) -> str:
    tier = str(args_tier or settings.get("default_tier", "medium")).strip().lower()
    if tier not in settings.get("runtime_tiers", {}):
        raise LoopError(f"Unknown tier: {tier}")
    return tier


def apply_runtime_tier(config: dict[str, Any], settings: dict[str, Any], tier: str) -> dict[str, Any]:
    _ = settings
    _ = tier
    return copy.deepcopy(config)


MutationFn = Callable[[dict[str, Any]], tuple[dict[str, Any], str]]


def _round_sig(value: float) -> float:
    return float(f"{value:.6g}")


def ensure_search_defaults(config: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(config)
    training = out.setdefault("training", {})
    training["selection_metric"] = "roc_auc_presence"
    out.setdefault("runtime", {}).setdefault("amp", True)
    out["runtime"].setdefault("device", "cuda")
    return out


def ensure_patch_cfg(config: dict[str, Any]) -> dict[str, Any]:
    out = ensure_search_defaults(config)
    out.setdefault("input", {})["preserve_aspect"] = True
    patch_cfg = out.setdefault("training", {}).setdefault("patch", {})
    patch_cfg.setdefault("enabled", True)
    patch_cfg.setdefault("size", 256)
    patch_cfg.setdefault("positive_prob", 0.90)
    patch_cfg.setdefault("hard_negative_prob", 0.50)
    patch_cfg.setdefault("hard_negative_quantile", 0.92)
    return out


def apply_selection_metric(config: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(config)
    out.setdefault("training", {})["selection_metric"] = str(settings["selection_metric"])
    return out


def mutate_lr_up(config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    out = copy.deepcopy(config)
    current = float(out.setdefault("training", {}).get("learning_rate", 3e-4))
    out["training"]["learning_rate"] = _round_sig(current * 2.0)
    return out, "double learning rate"


def mutate_lr_down(config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    out = copy.deepcopy(config)
    current = float(out.setdefault("training", {}).get("learning_rate", 3e-4))
    out["training"]["learning_rate"] = _round_sig(current * 0.5)
    return out, "halve learning rate"


def mutate_weight_decay_up(config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    out = copy.deepcopy(config)
    current = float(out.setdefault("training", {}).get("weight_decay", 1e-4))
    out["training"]["weight_decay"] = _round_sig(current * 2.0)
    return out, "increase weight decay"


def mutate_input_320(config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    out = copy.deepcopy(config)
    out.setdefault("input", {})["image_size"] = 320
    return out, "raise image size to 320"


def mutate_input_384(config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    out = copy.deepcopy(config)
    out.setdefault("input", {})["image_size"] = 384
    return out, "raise image size to 384"


def mutate_simple_unet_wider(config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    out = copy.deepcopy(config)
    model = out.setdefault("model", {})
    if str(model.get("name", "simple_unet")).lower() != "simple_unet":
        raise LoopError("simple_unet_wider requires model.name == simple_unet")
    base = int(model.get("base_channels", 32))
    model["base_channels"] = 48 if base < 48 else 64
    return out, "widen simple_unet"


def mutate_switch_deeplab50(config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    out = ensure_search_defaults(config)
    out["model"] = {
        "name": "deeplabv3_resnet50",
        "out_channels": 1,
        "pretrained": False,
        "pretrained_backbone": False,
    }
    out.setdefault("training", {})["batch_size"] = 2
    out.setdefault("input", {})["image_size"] = max(int(out.get("input", {}).get("image_size", 256)), 320)
    return out, "switch to deeplabv3_resnet50"


def mutate_switch_deeplab101(config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    out = ensure_search_defaults(config)
    out["model"] = {
        "name": "deeplabv3_resnet101",
        "out_channels": 1,
        "pretrained": False,
        "pretrained_backbone": False,
    }
    out.setdefault("training", {})["batch_size"] = 2
    out.setdefault("input", {})["image_size"] = max(int(out.get("input", {}).get("image_size", 256)), 320)
    return out, "switch to deeplabv3_resnet101"


def mutate_augment_off(config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    out = copy.deepcopy(config)
    out.setdefault("training", {})["augment"] = False
    return out, "disable augmentation"


def mutate_balanced_sampling_off(config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    out = copy.deepcopy(config)
    out.setdefault("training", {})["balanced_sampling"] = False
    return out, "disable balanced sampling"


def mutate_loss_more_dice(config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    out = copy.deepcopy(config)
    loss = out.setdefault("loss", {})
    loss["name"] = "bce_dice"
    loss["bce_weight"] = 0.2
    loss["dice_weight"] = 0.8
    loss["dice_positive_only"] = True
    return out, "bias loss toward dice"


def mutate_threshold_040(config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    out = copy.deepcopy(config)
    out.setdefault("evaluation", {})["threshold"] = 0.4
    return out, "set eval threshold 0.4"


def mutate_threshold_060(config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    out = copy.deepcopy(config)
    out.setdefault("evaluation", {})["threshold"] = 0.6
    return out, "set eval threshold 0.6"


def mutate_scheduler_cosine(config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    out = copy.deepcopy(config)
    training = out.setdefault("training", {})
    training["scheduler"] = "cosine"
    training["min_lr"] = float(training.get("min_lr", 1e-6))
    return out, "enable cosine scheduler"


def mutate_scheduler_onecycle(config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    out = copy.deepcopy(config)
    training = out.setdefault("training", {})
    training["scheduler"] = "onecycle"
    training["max_lr"] = float(training.get("max_lr", training.get("learning_rate", 3e-4)))
    training["pct_start"] = float(training.get("pct_start", 0.15))
    return out, "enable onecycle scheduler"


def mutate_batch_size_2(config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    out = copy.deepcopy(config)
    out.setdefault("training", {})["batch_size"] = 2
    return out, "set batch size 2"


def mutate_batch_size_6(config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    out = copy.deepcopy(config)
    out.setdefault("training", {})["batch_size"] = 6
    return out, "set batch size 6"


def mutate_preserve_aspect_on(config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    out = copy.deepcopy(config)
    out.setdefault("input", {})["preserve_aspect"] = True
    return out, "enable preserve_aspect"


def mutate_patch_on(config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    out = ensure_patch_cfg(config)
    return out, "enable patch sampling"


def mutate_patch_hardneg_055(config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    out = ensure_patch_cfg(config)
    out["training"]["patch"]["hard_negative_prob"] = 0.55
    return out, "raise hard negative patch probability to 0.55"


def mutate_patch_hardneg_065(config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    out = ensure_patch_cfg(config)
    out["training"]["patch"]["hard_negative_prob"] = 0.65
    return out, "raise hard negative patch probability to 0.65"


def mutate_patch_positive_095(config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    out = ensure_patch_cfg(config)
    out["training"]["patch"]["positive_prob"] = 0.95
    return out, "raise positive patch probability to 0.95"


def mutate_pretrained_backbone_on(config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    out = ensure_search_defaults(config)
    model = out.setdefault("model", {})
    if "deeplab" not in str(model.get("name", "")).lower():
        raise LoopError("pretrained_backbone_on requires a deeplab model")
    model["pretrained_backbone"] = True
    return out, "enable pretrained backbone"


def mutate_presence_bce_015(config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    out = ensure_search_defaults(config)
    out.setdefault("loss", {})["presence_bce_weight"] = 0.15
    return out, "set presence_bce_weight to 0.15"


def mutate_presence_bce_030(config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    out = ensure_search_defaults(config)
    loss = out.setdefault("loss", {})
    loss["presence_bce_weight"] = 0.30
    loss["presence_bce_warmup_epochs"] = int(loss.get("presence_bce_warmup_epochs", 4))
    return out, "set presence_bce_weight to 0.30"


def mutate_presence_mode_cls(config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    out = ensure_search_defaults(config)
    out.setdefault("evaluation", {})["presence_score_mode"] = "cls"
    out["evaluation"]["presence_threshold"] = float(out["evaluation"].get("presence_threshold", 0.5))
    return out, "score presence with cls head"


def mutate_simple_unet_dual(config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    out = ensure_search_defaults(config)
    out["model"] = {
        "name": "simple_unet_dual_head",
        "in_channels": 3,
        "out_channels": 1,
        "base_channels": 32,
        "cls_hidden": 128,
        "cls_dropout": 0.2,
    }
    out.setdefault("loss", {})["presence_bce_weight"] = 0.25
    out["loss"]["presence_bce_warmup_epochs"] = 4
    out.setdefault("evaluation", {})["presence_score_mode"] = "cls"
    return out, "switch to simple_unet_dual_head"


def mutate_deeplab50_dual(config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    out = ensure_search_defaults(config)
    out = ensure_patch_cfg(out)
    out["model"] = {
        "name": "deeplabv3_resnet50_dual_head",
        "out_channels": 1,
        "pretrained": False,
        "pretrained_backbone": False,
        "cls_hidden": 256,
        "cls_dropout": 0.2,
    }
    out["training"]["batch_size"] = 2
    out.setdefault("loss", {})["presence_bce_weight"] = 0.30
    out["loss"]["presence_bce_warmup_epochs"] = 4
    out.setdefault("evaluation", {})["presence_score_mode"] = "cls"
    return out, "switch to deeplabv3_resnet50_dual_head"


def mutate_deeplab101_dual(config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    out = ensure_search_defaults(config)
    out = ensure_patch_cfg(out)
    out["model"] = {
        "name": "deeplabv3_resnet101_dual_head",
        "out_channels": 1,
        "pretrained": False,
        "pretrained_backbone": False,
        "cls_hidden": 256,
        "cls_dropout": 0.2,
    }
    out["training"]["batch_size"] = 2
    out.setdefault("loss", {})["presence_bce_weight"] = 0.30
    out["loss"]["presence_bce_warmup_epochs"] = 4
    out.setdefault("evaluation", {})["presence_score_mode"] = "cls"
    return out, "switch to deeplabv3_resnet101_dual_head"


def mutate_focal_dice_tversky(config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    out = ensure_search_defaults(config)
    out["loss"] = {
        "name": "focal_dice_tversky",
        "focal_weight": 0.35,
        "dice_weight": 0.35,
        "tversky_weight": 0.30,
        "focal_gamma": 2.0,
        "dice_positive_only": True,
        "tversky_alpha": 0.3,
        "tversky_beta": 0.7,
        "tversky_gamma": 1.0,
    }
    return out, "switch to focal_dice_tversky loss"


MUTATION_LIBRARY: dict[str, MutationFn] = {
    "lr_up": mutate_lr_up,
    "lr_down": mutate_lr_down,
    "weight_decay_up": mutate_weight_decay_up,
    "input_320": mutate_input_320,
    "input_384": mutate_input_384,
    "simple_unet_wider": mutate_simple_unet_wider,
    "switch_deeplab50": mutate_switch_deeplab50,
    "switch_deeplab101": mutate_switch_deeplab101,
    "augment_off": mutate_augment_off,
    "balanced_sampling_off": mutate_balanced_sampling_off,
    "loss_more_dice": mutate_loss_more_dice,
    "threshold_040": mutate_threshold_040,
    "threshold_060": mutate_threshold_060,
    "scheduler_cosine": mutate_scheduler_cosine,
    "scheduler_onecycle": mutate_scheduler_onecycle,
    "batch_size_2": mutate_batch_size_2,
    "batch_size_6": mutate_batch_size_6,
    "preserve_aspect_on": mutate_preserve_aspect_on,
    "patch_on": mutate_patch_on,
    "patch_hardneg_055": mutate_patch_hardneg_055,
    "patch_hardneg_065": mutate_patch_hardneg_065,
    "patch_positive_095": mutate_patch_positive_095,
    "pretrained_backbone_on": mutate_pretrained_backbone_on,
    "presence_bce_015": mutate_presence_bce_015,
    "presence_bce_030": mutate_presence_bce_030,
    "presence_mode_cls": mutate_presence_mode_cls,
    "simple_unet_dual": mutate_simple_unet_dual,
    "deeplab50_dual": mutate_deeplab50_dual,
    "deeplab101_dual": mutate_deeplab101_dual,
    "focal_dice_tversky": mutate_focal_dice_tversky,
}


def choose_auto_mutation(*, parent_config: dict[str, Any], already_used: set[str]) -> str:
    for name in MUTATION_LIBRARY:
        if name in already_used:
            continue
        try:
            MUTATION_LIBRARY[name](parent_config)
        except LoopError:
            continue
        return name
    raise LoopError("No remaining auto mutations for this parent/tier")


def resolve_mutation_label(*, requested: str, parent_config: dict[str, Any], already_used: set[str]) -> str:
    if requested == "auto":
        return choose_auto_mutation(parent_config=parent_config, already_used=already_used)
    if requested not in MUTATION_LIBRARY:
        raise LoopError(f"Unknown mutation: {requested}")
    return requested


def run_subprocess(*, cmd: list[str], cwd: pathlib.Path, log_path: pathlib.Path, dry_run: bool) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"[{utc_now()}] $ {' '.join(cmd)}\n")
        if dry_run:
            log.write("[dry-run]\n")
            return 0, 0.0
        completed = subprocess.run(cmd, cwd=str(cwd), check=False, stdout=log, stderr=subprocess.STDOUT, text=True)
    return int(completed.returncode), float(time.monotonic() - started)


def record_history(state: dict[str, Any], payload: dict[str, Any]) -> None:
    state.setdefault("history", []).append(payload)


def write_candidate_config(
    *,
    experiment_id: str,
    label: str,
    config_payload: dict[str, Any],
    generated_dir: pathlib.Path,
    tier: str,
) -> pathlib.Path:
    name = f"{experiment_id}_{slugify(label)}_{tier}.yaml"
    path = generated_dir / name
    save_yaml(path, config_payload)
    return path


def validate_metric(metrics_path: pathlib.Path, metric_key: str) -> float:
    return load_metric_bundle(metrics_path, [metric_key])[metric_key]


def execute_experiment(
    *,
    experiment_id: str,
    parent_experiment_id: str,
    mutation_label: str,
    note: str,
    tier: str,
    config_path: pathlib.Path,
    settings: dict[str, Any],
    paths: dict[str, pathlib.Path],
    state: dict[str, Any],
    dry_run: bool,
    keep_on_first: bool,
) -> dict[str, Any]:
    bench_root = paths["benchmark_repo_root"]
    python_exe = paths["benchmark_python_exe"]
    logs_dir = paths["logs_dir"]
    runs_dir = paths["runs_dir"]
    results_path = paths["results_file"]
    selection_metric = str(settings["selection_metric"])
    metric_keys = selection_metric_priority(settings)
    epsilon = float(settings.get("keep_improvement_epsilon", 1e-6))
    candidate_epsilon = float(settings.get("candidate_metric_epsilon", 0.0))

    run_dir = runs_dir / experiment_id
    run_dir.mkdir(parents=True, exist_ok=True)
    train_log = logs_dir / f"{experiment_id}_train.log"
    validate_log = logs_dir / f"{experiment_id}_validate.log"

    train_cmd = [
        str(python_exe),
        str((bench_root / "scripts" / "train.py").resolve()),
        "--config",
        str(config_path),
        "--output-dir",
        str(run_dir),
    ]
    validate_cmd = [
        str(python_exe),
        str((bench_root / "scripts" / "validate.py").resolve()),
        "--config",
        str((run_dir / "resolved_config.yaml").resolve()),
        "--checkpoint",
        str((run_dir / "best_model.pt").resolve()),
        "--output",
        str((run_dir / "validate_metrics.json").resolve()),
    ]

    train_rc, train_seconds = run_subprocess(cmd=train_cmd, cwd=bench_root, log_path=train_log, dry_run=dry_run)
    total_seconds = train_seconds

    status = "crash"
    resolved_config_path = run_dir / "resolved_config.yaml"
    checkpoint_path = run_dir / "best_model.pt"
    metrics_path = run_dir / "validate_metrics.json"
    metric_value_txt = ""

    if train_rc == 0 and not dry_run:
        if not resolved_config_path.exists() or not checkpoint_path.exists():
            status = "crash"
            note = f"{note}; missing resolved config or checkpoint".strip("; ")
        else:
            validate_rc, validate_seconds = run_subprocess(
                cmd=validate_cmd,
                cwd=bench_root,
                log_path=validate_log,
                dry_run=False,
            )
            total_seconds += validate_seconds
            if validate_rc != 0 or not metrics_path.exists():
                status = "crash"
                note = f"{note}; validate failed".strip("; ")
            else:
                metric_bundle = load_metric_bundle(metrics_path, metric_keys)
                metric_value = float(metric_bundle[selection_metric])
                metric_value_txt = f"{metric_value:.6f}"
                prior_best = current_best_result(
                    read_results(results_path),
                    tier,
                    selection_metric,
                    metric_keys=metric_keys,
                )
                prior_metrics = None
                keep = keep_on_first
                if prior_best is not None:
                    prior_metrics_path = metrics_path_for_result_row(prior_best)
                    if prior_metrics_path is None or not prior_metrics_path.exists():
                        raise LoopError(
                            f"Missing validate_metrics.json for current best {prior_best.get('experiment_id', '')}"
                        )
                    prior_metrics = load_metric_bundle(prior_metrics_path, metric_keys)
                    comparison, _ = compare_metric_priority(metric_bundle, prior_metrics, metric_keys, epsilon)
                    keep = comparison > 0
                if keep_on_first and prior_best is None:
                    status = "baseline"
                else:
                    if keep:
                        status = "keep"
                    elif prior_best is not None and prior_metrics is not None and is_near_best_candidate(
                        metric_bundle,
                        prior_metrics,
                        metric_keys,
                        candidate_epsilon,
                    ):
                        status = "candidate"
                        note = annotate_candidate_outcome_note(
                            note=note,
                            primary_metric=selection_metric,
                            prior_best=prior_best,
                            prior_metrics=prior_metrics,
                            candidate_epsilon=candidate_epsilon,
                        )
                    else:
                        status = "discard"
                        note = annotate_ranked_outcome_note(
                            note=note,
                            primary_metric=selection_metric,
                            metric_keys=metric_keys,
                            prior_best=prior_best,
                            prior_metrics=prior_metrics,
                            comparison=comparison if prior_best is not None else 0,
                        )
    elif dry_run:
        status = "dry_run"
        metric_value_txt = ""
    else:
        note = append_note(note, f"train failed rc={train_rc}")

    row = {
        "experiment_id": experiment_id,
        "parent_experiment_id": parent_experiment_id,
        "status": status,
        "runtime_tier": tier,
        "config_path": rel_or_abs(config_path),
        "resolved_config_path": rel_or_abs(resolved_config_path),
        "checkpoint_path": rel_or_abs(checkpoint_path),
        "val_metric_key": selection_metric,
        "val_metric_value": metric_value_txt,
        "train_seconds": f"{train_seconds:.3f}" if not dry_run else "",
        "total_seconds": f"{total_seconds:.3f}" if not dry_run else "",
        "notes": note,
    }
    if not dry_run:
        append_result(results_path, row)
        refresh_experiment_summary(paths)
        record_history(
            state,
            {
                "experiment_id": experiment_id,
                "parent_experiment_id": parent_experiment_id,
                "runtime_tier": tier,
                "mutation_label": mutation_label,
                "status": status,
                "config_path": rel_or_abs(config_path),
                "created_utc": utc_now(),
            },
        )
        if status in KEEP_STATUSES:
            state.setdefault("best_by_tier", {})[tier] = experiment_id
    return row


def print_status(settings: dict[str, Any], paths: dict[str, pathlib.Path], limit: int) -> int:
    refresh_experiment_summary(paths)
    results = read_results(paths["results_file"])
    state = load_state(paths["state_file"])
    tiers = sorted(settings.get("runtime_tiers", {}).keys())
    print("mini_llm_cnn status")
    print(f"benchmark_repo_root: {paths['benchmark_repo_root']}")
    print(f"benchmark_python_exe: {paths['benchmark_python_exe']}")
    for tier in tiers:
        best_id = str(state.get("best_by_tier", {}).get(tier, "")).strip()
        best_row = None
        if best_id:
            for row in reversed(results):
                if row.get("experiment_id") == best_id:
                    best_row = row
                    break
        if best_row is None:
            print(f"- {tier}: no kept baseline yet")
            continue
        print(
            f"- {tier}: {best_id} status={best_row.get('status')} "
            f"{best_row.get('val_metric_key')}={best_row.get('val_metric_value')}"
        )
    tail = results[-max(0, limit) :]
    if tail:
        print("")
        print("recent results:")
        for row in tail:
            print(
                f"  {row['experiment_id']} {row['runtime_tier']} {row['status']} "
                f"{row['val_metric_key']}={row['val_metric_value']} notes={row['notes']}"
            )
    return 0


def do_baseline(
    settings: dict[str, Any],
    paths: dict[str, pathlib.Path],
    tier: str,
    dry_run: bool,
    note_suffix: str = "",
) -> int:
    ensure_results_header(paths["results_file"])
    state = load_state(paths["state_file"])
    results = read_results(paths["results_file"], create=True)
    if current_best_result(
        results,
        tier,
        str(settings["selection_metric"]),
        metric_keys=selection_metric_priority(settings),
    ) is not None:
        raise LoopError(f"Baseline already exists for tier '{tier}'. Use step or run-config.")
    bench_root = paths["benchmark_repo_root"]
    experiment_id = make_experiment_id(state)
    base_cfg = load_yaml(benchmark_config_path(settings, bench_root))
    tier_cfg = apply_selection_metric(apply_runtime_tier(base_cfg, settings, tier), settings)
    label = "baseline"
    note = append_note(f"baseline from {settings['baseline_config']}", note_suffix)
    config_path = write_candidate_config(
        experiment_id=experiment_id,
        label=label,
        config_payload=tier_cfg,
        generated_dir=paths["generated_config_dir"],
        tier=tier,
    )
    row = execute_experiment(
        experiment_id=experiment_id,
        parent_experiment_id="",
        mutation_label=label,
        note=note,
        tier=tier,
        config_path=config_path,
        settings=settings,
        paths=paths,
        state=state,
        dry_run=dry_run,
        keep_on_first=True,
    )
    if not dry_run:
        save_state(paths["state_file"], state)
    print(f"baseline complete: {row['experiment_id']} status={row['status']}")
    return 0


def do_step(settings: dict[str, Any], paths: dict[str, pathlib.Path], tier: str, requested_mutation: str, dry_run: bool) -> int:
    ensure_results_header(paths["results_file"])
    state = load_state(paths["state_file"])
    results = read_results(paths["results_file"], create=True)
    bench_root = paths["benchmark_repo_root"]
    parent_cfg, parent_id, _ = load_parent_config(tier=tier, results=results, settings=settings, bench_root=bench_root)
    if not parent_id:
        raise LoopError(f"No baseline for tier '{tier}'. Run baseline first.")
    used = mutation_history_for_parent(state, parent_id, tier)
    mutation_name = resolve_mutation_label(requested=requested_mutation, parent_config=parent_cfg, already_used=used)
    mutated_cfg, description = MUTATION_LIBRARY[mutation_name](parent_cfg)
    tier_cfg = apply_selection_metric(apply_runtime_tier(mutated_cfg, settings, tier), settings)
    experiment_id = make_experiment_id(state)
    label = mutation_name
    note = f"{description}; parent={parent_id}"
    config_path = write_candidate_config(
        experiment_id=experiment_id,
        label=label,
        config_payload=tier_cfg,
        generated_dir=paths["generated_config_dir"],
        tier=tier,
    )
    row = execute_experiment(
        experiment_id=experiment_id,
        parent_experiment_id=parent_id,
        mutation_label=mutation_name,
        note=note,
        tier=tier,
        config_path=config_path,
        settings=settings,
        paths=paths,
        state=state,
        dry_run=dry_run,
        keep_on_first=False,
    )
    if not dry_run:
        save_state(paths["state_file"], state)
    print(f"step complete: {row['experiment_id']} status={row['status']}")
    return 0


def do_run_config(
    settings: dict[str, Any],
    paths: dict[str, pathlib.Path],
    tier: str,
    config_arg: str,
    label: str,
    parent_experiment_id: str,
    dry_run: bool,
    note_suffix: str = "",
) -> int:
    ensure_results_header(paths["results_file"])
    state = load_state(paths["state_file"])
    results = read_results(paths["results_file"], create=True)
    if current_best_result(
        results,
        tier,
        str(settings["selection_metric"]),
        metric_keys=selection_metric_priority(settings),
    ) is None:
        raise LoopError(f"No baseline for tier '{tier}'. Run baseline first.")
    raw_path = pathlib.Path(config_arg)
    if raw_path.is_absolute():
        candidate_path = raw_path
    else:
        candidate_path = (REPO_ROOT / raw_path).resolve()
    if not candidate_path.exists():
        raise LoopError(f"Config does not exist: {candidate_path}")
    candidate_cfg = load_yaml(candidate_path)
    candidate_cfg = apply_selection_metric(apply_runtime_tier(candidate_cfg, settings, tier), settings)
    experiment_id = make_experiment_id(state)
    generated_path = write_candidate_config(
        experiment_id=experiment_id,
        label=label,
        config_payload=candidate_cfg,
        generated_dir=paths["generated_config_dir"],
        tier=tier,
    )
    note = append_note(f"manual config from {candidate_path.name}", note_suffix)
    row = execute_experiment(
        experiment_id=experiment_id,
        parent_experiment_id=parent_experiment_id,
        mutation_label=slugify(label),
        note=note,
        tier=tier,
        config_path=generated_path,
        settings=settings,
        paths=paths,
        state=state,
        dry_run=dry_run,
        keep_on_first=False,
    )
    if not dry_run:
        save_state(paths["state_file"], state)
    print(f"run-config complete: {row['experiment_id']} status={row['status']}")
    return 0


def append_night_log(log_path: pathlib.Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{utc_now()}] {message}\n")


def do_night_run(
    settings: dict[str, Any],
    paths: dict[str, pathlib.Path],
    tier: str,
    hours: float,
    sleep_seconds: int,
    dry_run: bool,
) -> int:
    if hours <= 0:
        raise LoopError("--hours must be > 0")
    sleep_seconds = max(0, int(sleep_seconds))
    deadline = time.monotonic() + (float(hours) * 3600.0)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    night_log = paths["logs_dir"] / f"night_run_{tier}_{stamp}.log"
    append_night_log(night_log, f"start tier={tier} hours={hours} dry_run={dry_run}")

    results = read_results(paths["results_file"])
    if current_best_result(
        results,
        tier,
        str(settings["selection_metric"]),
        metric_keys=selection_metric_priority(settings),
    ) is None:
        append_night_log(night_log, f"baseline missing for tier={tier}; running baseline")
        do_baseline(settings, paths, tier, dry_run)
    else:
        append_night_log(night_log, f"baseline already present for tier={tier}")

    steps = 0
    while time.monotonic() < deadline:
        try:
            do_step(settings, paths, tier, "auto", dry_run)
            steps += 1
            append_night_log(night_log, f"step_complete index={steps}")
        except LoopError as exc:
            append_night_log(night_log, f"step_error {exc}")
            if "No remaining auto mutations" in str(exc):
                break
            raise
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    append_night_log(night_log, f"end steps={steps}")
    print(f"night-run complete: tier={tier} steps={steps} log={rel_or_abs(night_log)}")
    return 0


def do_test(settings: dict[str, Any], paths: dict[str, pathlib.Path], experiment_id: str, dry_run: bool) -> int:
    results = read_results(paths["results_file"])
    row = None
    for item in results:
        if item.get("experiment_id") == experiment_id:
            row = item
    if row is None:
        raise LoopError(f"Unknown experiment id: {experiment_id}")
    if row.get("status") not in TESTABLE_STATUSES:
        raise LoopError(f"Locked test is only allowed for kept, candidate, or baseline runs: {experiment_id}")

    state = load_state(paths["state_file"])
    bench_root = paths["benchmark_repo_root"]
    python_exe = paths["benchmark_python_exe"]
    run_dir = resolve_from_repo(row["checkpoint_path"]).parent
    test_output = run_dir / "test_metrics.json"
    test_log = paths["logs_dir"] / f"{experiment_id}_test.log"
    cmd = [
        str(python_exe),
        str((bench_root / "scripts" / "test.py").resolve()),
        "--config",
        str(resolve_from_repo(row["resolved_config_path"])),
        "--checkpoint",
        str(resolve_from_repo(row["checkpoint_path"])),
        "--output",
        str(test_output),
    ]
    rc, elapsed = run_subprocess(cmd=cmd, cwd=bench_root, log_path=test_log, dry_run=dry_run)
    state.setdefault("test_evaluations", {})[experiment_id] = {
        "requested_utc": utc_now(),
        "returncode": rc,
        "elapsed_seconds": elapsed,
        "output_path": rel_or_abs(test_output),
        "dry_run": bool(dry_run),
    }
    if not dry_run:
        save_state(paths["state_file"], state)
    if rc != 0 and not dry_run:
        raise LoopError(f"Locked test failed for {experiment_id}; see {test_log}")
    print(f"test complete: {experiment_id} output={rel_or_abs(test_output)}")
    return 0


def main() -> int:
    args = parse_args()
    settings = load_settings()
    paths = ensure_settings_paths(settings)

    try:
        if args.command == "status":
            return print_status(settings, paths, int(args.limit))
        if args.command == "list-mutations":
            for name in MUTATION_LIBRARY:
                print(name)
            return 0
        if args.command == "baseline":
            return do_baseline(
                settings,
                paths,
                runtime_tier_name(args.tier, settings),
                bool(args.dry_run),
                str(args.note_suffix),
            )
        if args.command == "step":
            return do_step(
                settings,
                paths,
                runtime_tier_name(args.tier, settings),
                str(args.mutation).strip().lower(),
                bool(args.dry_run),
            )
        if args.command == "run-config":
            return do_run_config(
                settings,
                paths,
                runtime_tier_name(args.tier, settings),
                str(args.config),
                str(args.label),
                str(args.parent_experiment_id),
                bool(args.dry_run),
                str(args.note_suffix),
            )
        if args.command == "night-run":
            return do_night_run(
                settings,
                paths,
                runtime_tier_name(args.tier, settings),
                float(args.hours),
                int(args.sleep_seconds),
                bool(args.dry_run),
            )
        if args.command == "test":
            return do_test(settings, paths, str(args.experiment_id), bool(args.dry_run))
        raise LoopError(f"Unsupported command: {args.command}")
    except LoopError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
