#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any

import yaml


ACTION_VALUES: tuple[str, ...] = (
    "baseline",
    "run_config",
    "test",
    "install_package",
    "download_file",
    "done",
    "blocked",
)

CODE_EDIT_FIELDS: tuple[str, str] = ("path", "content")
DEFAULT_AUTO_REPAIR_MODULE_PACKAGE_MAP: dict[str, list[str]] = {
    "segmentation_models_pytorch": ["segmentation-models-pytorch", "timm"],
    "timm": ["timm"],
    "einops": ["einops"],
    "transformers": ["transformers"],
}
DEFAULT_AUTO_REPAIR_MODULE_ALIAS_MAP: dict[str, list[str]] = {
    "cv2": ["opencv-python"],
    "pil": ["pillow"],
    "yaml": ["pyyaml"],
    "sklearn": ["scikit-learn"],
}
DEFAULT_AUTO_REPAIR_MODEL_PACKAGE_MAP: dict[str, list[str]] = {
    "deeplabv3": ["segmentation-models-pytorch", "timm"],
    "deeplabv3_dual": ["segmentation-models-pytorch", "timm"],
    "deeplabv3_dual_head": ["segmentation-models-pytorch", "timm"],
    "unet_resnet34": ["segmentation-models-pytorch", "timm"],
    "unet_resnet34_dual": ["segmentation-models-pytorch", "timm"],
    "unet_resnet34_dual_head": ["segmentation-models-pytorch", "timm"],
}
DEFAULT_AUTO_REPAIR_MODEL_FALLBACK_MAP: dict[str, str] = {
    "deeplabv3": "deeplabv3_resnet50",
    "deeplabv3_dual": "deeplabv3_resnet50_dual_head",
    "deeplabv3_dual_head": "deeplabv3_resnet50_dual_head",
    "unet_resnet34": "simple_unet",
    "unet_resnet34_dual": "simple_unet_dual_head",
    "unet_resnet34_dual_head": "simple_unet_dual_head",
}
PATH_LIKE_ENV_VARS: set[str] = {
    "TORCH_HOME",
    "HF_HOME",
    "HUGGINGFACE_HUB_CACHE",
    "PIP_CACHE_DIR",
    "TMP",
    "TEMP",
    "TMPDIR",
}
PLAIN_SIMPLE_UNET_STREAK_LIMIT = 3
DEFAULT_SAME_FAMILY_MICRO_TWEAK_STREAK = 6
DEFAULT_CODE_EDIT_ESCALATION_STREAK = 8
DEFAULT_SAME_FAMILY_BROAD_JUMP_MIN_AXES = 2
DEFAULT_RESEARCH_REFRESH_STREAK = 2


class PolicyRejectError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dir(path: pathlib.Path) -> pathlib.Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def rel_or_abs(path: pathlib.Path, root: pathlib.Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path.resolve())


def resolve_repo_path(repo_root: pathlib.Path, raw: str) -> pathlib.Path:
    path = pathlib.Path(raw)
    return path if path.is_absolute() else (repo_root / path).resolve()


def build_runtime_env(repo_root: pathlib.Path, loop_cfg: dict[str, Any], *, codex_home: pathlib.Path | None = None) -> dict[str, str]:
    env = dict(os.environ)
    overrides = loop_cfg.get("runtime_env", {})
    if not isinstance(overrides, dict):
        overrides = {}
    for key, value in overrides.items():
        key_txt = str(key).strip()
        if not key_txt:
            continue
        value_txt = str(value).strip()
        if key_txt in PATH_LIKE_ENV_VARS:
            target = pathlib.Path(value_txt)
            if not target.is_absolute():
                target = (repo_root / target).resolve()
            target.mkdir(parents=True, exist_ok=True)
            env[key_txt] = str(target)
        else:
            env[key_txt] = value_txt
    if codex_home is not None:
        env["CODEX_HOME"] = str(codex_home)
    return env


def try_read_text(path: pathlib.Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def read_text_limited(path: pathlib.Path, *, max_chars: int = 8000) -> str:
    text = try_read_text(path).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated]"


def latest_results_summary(results_path: pathlib.Path, *, max_rows: int = 8) -> str:
    if not results_path.exists():
        return "results.tsv does not exist yet."
    rows = [line.rstrip("\n") for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return "results.tsv is empty."
    return "\n".join(rows[-max_rows:])


def list_allowed_runtime_tiers(loop_cfg: dict[str, Any], search_space: str, app_cfg: dict[str, Any]) -> list[str]:
    key = "limited_search_runtime_tiers" if search_space == "limited" else "open_search_runtime_tiers"
    configured = loop_cfg.get(key, [])
    tiers: list[str] = []
    if isinstance(configured, list):
        for item in configured:
            name = str(item).strip()
            if name and name in app_cfg.get("runtime_tiers", {}):
                tiers.append(name)
    if tiers:
        return tiers
    return [str(app_cfg.get("default_tier", "medium"))]


def runtime_tier_summary(app_cfg: dict[str, Any], allowed_tiers: list[str]) -> str:
    lines: list[str] = []
    tier_specs = app_cfg.get("runtime_tiers", {})
    for name in allowed_tiers:
        spec = tier_specs.get(name, {})
        description = ""
        if isinstance(spec, dict):
            description = str(spec.get("description", "")).strip()
        suffix = f": {description}" if description else ""
        lines.append(f"- {name}{suffix}")
    lines.append("- No automatic epoch or batch caps are applied by the wrapper. The config itself defines training budget.")
    return "\n".join(lines)


def load_results_rows(results_path: pathlib.Path) -> list[dict[str, str]]:
    if not results_path.exists():
        return []
    with results_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [dict(row) for row in reader]


def current_best_result(results: list[dict[str, str]], tier: str, metric_key: str) -> dict[str, str] | None:
    best_row = None
    best_val = float("-inf")
    for row in results:
        if row.get("runtime_tier") != tier:
            continue
        if row.get("status") not in {"baseline", "keep"}:
            continue
        if row.get("val_metric_key") != metric_key:
            continue
        try:
            score = float(row.get("val_metric_value", "nan"))
        except Exception:
            continue
        if score > best_val:
            best_val = score
            best_row = row
    return best_row


def latest_result_row(results_path: pathlib.Path, *, tier: str | None = None) -> dict[str, str] | None:
    rows = load_results_rows(results_path)
    if tier is None:
        return rows[-1] if rows else None
    for row in reversed(rows):
        if row.get("runtime_tier") == tier:
            return row
    return None


def family_attempt_counts(results: list[dict[str, str]], repo_root: pathlib.Path, tier: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in results:
        if row.get("runtime_tier") != tier:
            continue
        family = model_name_from_result_row(repo_root, row)
        if not family:
            continue
        counts[family] = counts.get(family, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def family_last_keep(results: list[dict[str, str]], repo_root: pathlib.Path, tier: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in results:
        if row.get("runtime_tier") != tier:
            continue
        if str(row.get("status", "")).strip().lower() not in {"baseline", "keep", "candidate"}:
            continue
        family = model_name_from_result_row(repo_root, row)
        if family:
            out[family] = str(row.get("experiment_id", "")).strip()
    return out


def count_session_code_edit_attempts(session_cycles: list[dict[str, Any]]) -> int:
    total = 0
    for cycle in session_cycles:
        action = cycle.get("action", {})
        if isinstance(action, dict) and action.get("code_edits"):
            total += 1
    return total


def count_session_web_search_cycles(session_cycles: list[dict[str, Any]]) -> int:
    total = 0
    for cycle in session_cycles:
        telemetry = cycle.get("telemetry", {})
        if isinstance(telemetry, dict) and telemetry.get("used_web_search"):
            total += 1
    return total


def recent_cycles_without_web_search(session_cycles: list[dict[str, Any]]) -> int:
    streak = 0
    for cycle in reversed(session_cycles):
        telemetry = cycle.get("telemetry", {})
        used_web_search = isinstance(telemetry, dict) and bool(telemetry.get("used_web_search"))
        if used_web_search:
            break
        streak += 1
    return streak


def determine_search_phase(
    *,
    tier: str,
    best_exists: bool,
    non_keep_streak: int,
    allowed_runtime_tiers: list[str],
    finalize_runtime_tiers: list[str],
    code_edit_attempt_count: int,
    code_edit_escalation_streak: int,
) -> str:
    if tier in finalize_runtime_tiers and best_exists:
        return "finalize"
    if not best_exists:
        return "explore"
    if non_keep_streak >= max(1, code_edit_escalation_streak) and "long" in allowed_runtime_tiers and tier != "long":
        return "finalize"
    if non_keep_streak >= 3 or code_edit_attempt_count > 0:
        return "refine"
    return "explore"


def build_progress_snapshot(
    *,
    repo_root: pathlib.Path,
    results_path: pathlib.Path,
    tier: str,
    metric_key: str,
    session_cycles: list[dict[str, Any]],
    allowed_runtime_tiers: list[str],
    finalize_runtime_tiers: list[str],
    code_edit_escalation_streak: int,
    research_refresh_streak: int,
) -> dict[str, Any]:
    results = load_results_rows(results_path)
    best = current_best_result(results, tier, metric_key)
    non_keep = recent_non_keep_streak(results_path=results_path, tier=tier)
    code_edit_attempts = count_session_code_edit_attempts(session_cycles)
    web_search_cycles = count_session_web_search_cycles(session_cycles)
    phase = determine_search_phase(
        tier=tier,
        best_exists=best is not None,
        non_keep_streak=non_keep,
        allowed_runtime_tiers=allowed_runtime_tiers,
        finalize_runtime_tiers=finalize_runtime_tiers,
        code_edit_attempt_count=code_edit_attempts,
        code_edit_escalation_streak=code_edit_escalation_streak,
    )
    recommended_runtime_tier = tier
    recent_without_search = recent_cycles_without_web_search(session_cycles)
    research_refresh_due = phase in {"refine", "finalize"} and recent_without_search >= max(1, research_refresh_streak)
    if phase == "finalize" and "long" in allowed_runtime_tiers and tier != "long":
        recommended_runtime_tier = "long"
    return {
        "phase": phase,
        "recommended_runtime_tier": recommended_runtime_tier,
        "non_keep_streak_by_tier": {tier: non_keep},
        "family_attempt_counts": {tier: family_attempt_counts(results, repo_root, tier)},
        "family_last_keep": {tier: family_last_keep(results, repo_root, tier)},
        "code_edit_attempt_count": code_edit_attempts,
        "web_search_cycles": web_search_cycles,
        "recent_cycles_without_web_search": recent_without_search,
        "research_refresh_due": research_refresh_due,
        "code_edit_escalation_due": non_keep >= max(1, code_edit_escalation_streak) and code_edit_attempts == 0,
    }


def format_progress_snapshot(progress: dict[str, Any], tier: str) -> str:
    family_counts = progress.get("family_attempt_counts", {}).get(tier, {})
    family_count_text = ", ".join(f"{name}:{count}" for name, count in list(family_counts.items())[:6]) or "(none)"
    family_last_keep = progress.get("family_last_keep", {}).get(tier, {})
    last_keep_text = ", ".join(f"{name}->{exp_id}" for name, exp_id in list(family_last_keep.items())[:6]) or "(none)"
    lines = [
        f"- phase={progress.get('phase', 'explore')}",
        f"- recommended_runtime_tier={progress.get('recommended_runtime_tier', tier)}",
        f"- non_keep_streak[{tier}]={progress.get('non_keep_streak_by_tier', {}).get(tier, 0)}",
        f"- family_attempt_counts[{tier}]={family_count_text}",
        f"- family_last_keep[{tier}]={last_keep_text}",
        f"- code_edit_attempt_count={progress.get('code_edit_attempt_count', 0)}",
        f"- web_search_cycles={progress.get('web_search_cycles', 0)}",
        f"- recent_cycles_without_web_search={progress.get('recent_cycles_without_web_search', 0)}",
    ]
    if progress.get("code_edit_escalation_due"):
        lines.append("- escalation_due=plateau has persisted without any code_edit attempts; prefer a code_edit experiment or a clearly broader jump.")
    if progress.get("research_refresh_due"):
        lines.append("- research_refresh_due=true; do web search before the next proposal and use it to justify a broader jump, alternate family, or code-edit direction.")
    return "\n".join(lines)


def should_stop_loop(*, action_kind: str, execution_status: str) -> bool:
    status = str(execution_status).strip().lower()
    if str(action_kind).strip().lower() == "done" and status != "policy_rejected":
        return True
    return status in {"terminal_blocked", "infra_blocked"}


def benchmark_baseline_config_path(repo_root: pathlib.Path, app_cfg: dict[str, Any], benchmark_repo_root: pathlib.Path) -> pathlib.Path:
    raw = str(app_cfg["baseline_config"])
    candidate = pathlib.Path(raw)
    if candidate.is_absolute():
        return candidate
    return (benchmark_repo_root / candidate).resolve()


def extract_best_config_context(
    *,
    repo_root: pathlib.Path,
    results_path: pathlib.Path,
    tier: str,
    metric_key: str,
) -> dict[str, str]:
    rows = load_results_rows(results_path)
    best = current_best_result(rows, tier, metric_key)
    if best is None:
        return {
            "experiment_id": "",
            "metric_key": metric_key,
            "metric": "",
            "config_path": "",
            "config_text": "",
        }

    preferred = str(best.get("resolved_config_path", "")).strip() or str(best.get("config_path", "")).strip()
    config_path = resolve_repo_path(repo_root, preferred) if preferred else pathlib.Path()
    config_text = read_text_limited(config_path, max_chars=6000) if preferred and config_path.exists() else ""
    return {
        "experiment_id": str(best.get("experiment_id", "")).strip(),
        "metric_key": metric_key,
        "metric": str(best.get("val_metric_value", "")).strip(),
        "config_path": preferred,
        "config_text": config_text,
    }


def run_capture(
    *,
    cmd: list[str],
    cwd: pathlib.Path,
    env: dict[str, str] | None = None,
) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    output = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
    return proc.returncode, output.strip()


def run_logged_command(
    *,
    cmd: list[str],
    cwd: pathlib.Path,
    log_path: pathlib.Path,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"[{utc_now()}] COMMAND {' '.join(cmd)}\n")
        handle.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
    return {
        "returncode": int(proc.returncode),
        "elapsed_seconds": round(time.time() - started, 3),
        "log_path": str(log_path),
    }


def tail_text(path: pathlib.Path, *, max_bytes: int = 24000) -> str:
    if not path.exists():
        return ""
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            start = size - max_bytes if size > max_bytes else 0
            handle.seek(start)
            return handle.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""


def normalize_package_map(raw: Any, defaults: dict[str, list[str]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    merged = dict(defaults)
    if isinstance(raw, dict):
        merged.update(raw)
    for key, value in merged.items():
        key_txt = str(key).strip().lower()
        if not key_txt:
            continue
        items = value if isinstance(value, list) else [value]
        packages: list[str] = []
        for item in items:
            package = str(item).strip()
            if package and package not in packages:
                packages.append(package)
        if packages:
            out[key_txt] = packages
    return out


def normalize_fallback_map(raw: Any, defaults: dict[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    merged = dict(defaults)
    if isinstance(raw, dict):
        merged.update(raw)
    for key, value in merged.items():
        key_txt = str(key).strip().lower()
        replacement = str(value).strip()
        if key_txt and replacement:
            out[key_txt] = replacement
    return out


def extract_missing_module(log_text: str) -> str | None:
    if not log_text:
        return None
    match = re.search(r"ModuleNotFoundError:\s+No module named ['\"]([^'\"]+)['\"]", log_text)
    if not match:
        return None
    return str(match.group(1)).strip()


def extract_unsupported_model_name(log_text: str) -> str | None:
    if not log_text:
        return None
    match = re.search(r"Unsupported model\.name:\s*([A-Za-z0-9_\-\.]+)", log_text)
    if not match:
        return None
    return str(match.group(1)).strip()


def infer_direct_module_packages(missing_module: str, alias_map: dict[str, list[str]]) -> list[str]:
    module_txt = str(missing_module).strip()
    if not module_txt:
        return []
    root = module_txt.split(".")[0].strip()
    key = root.lower()
    if key in alias_map:
        return list(alias_map[key])
    packages: list[str] = []
    for item in [module_txt, root, root.replace("_", "-"), module_txt.replace(".", "-")]:
        package = str(item).strip()
        if package and package not in packages:
            packages.append(package)
    return packages


def patch_model_name_in_yaml_config(
    *,
    repo_root: pathlib.Path,
    config_path: pathlib.Path,
    unsupported_model: str,
    replacement_model: str,
    label: str,
) -> pathlib.Path | None:
    if not config_path.exists():
        return None
    original_text = config_path.read_text(encoding="utf-8", errors="ignore")
    pattern = re.compile(
        rf"(?mi)^(\s*name\s*:\s*)(['\"]?){re.escape(unsupported_model)}(['\"]?)\s*$"
    )
    patched_text, changed = pattern.subn(rf"\1\2{replacement_model}\3", original_text, count=1)
    if changed <= 0:
        return None
    out_dir = ensure_dir(repo_root / ".mini_loop" / "autofix_configs")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_name = f"{stamp}_{slugify(label)}_{config_path.stem}_autofix_{slugify(replacement_model)}{config_path.suffix or '.yaml'}"
    out_path = out_dir / out_name
    out_path.write_text(patched_text, encoding="utf-8")
    return out_path


def apply_model_name_fallback_to_run_command(
    *,
    repo_root: pathlib.Path,
    command: list[str],
    unsupported_model: str,
    fallback_map: dict[str, str],
    label: str,
) -> tuple[list[str], pathlib.Path] | None:
    replacement = str(fallback_map.get(unsupported_model.strip().lower(), "")).strip()
    if not replacement:
        return None
    if "--config" not in command:
        return None
    config_index = command.index("--config") + 1
    if config_index >= len(command):
        return None
    config_path = pathlib.Path(command[config_index])
    patched_config = patch_model_name_in_yaml_config(
        repo_root=repo_root,
        config_path=config_path,
        unsupported_model=unsupported_model,
        replacement_model=replacement,
        label=label,
    )
    if not patched_config:
        return None
    patched_command = list(command)
    patched_command[config_index] = str(patched_config)
    return patched_command, patched_config


def install_packages_with_python(
    *,
    python_exe: pathlib.Path,
    cwd: pathlib.Path,
    packages: list[str],
    log_path: pathlib.Path,
    env: dict[str, str],
) -> dict[str, Any]:
    if not packages:
        return {"attempted": False, "success": False, "packages": []}
    command = [str(python_exe), "-m", "pip", "install", *packages]
    outcome = run_logged_command(cmd=command, cwd=cwd, log_path=log_path, env=env)
    return {
        "attempted": True,
        "success": int(outcome.get("returncode", 1)) == 0,
        "packages": packages,
        "command": command,
        "log_path": outcome.get("log_path", str(log_path)),
        "returncode": int(outcome.get("returncode", 1)),
        "elapsed_seconds": outcome.get("elapsed_seconds", 0.0),
    }


def attempt_auto_repair(
    *,
    repo_root: pathlib.Path,
    benchmark_repo_root: pathlib.Path,
    benchmark_python: pathlib.Path,
    command: list[str],
    log_path: pathlib.Path,
    logs_dir: pathlib.Path,
    label: str,
    cycle_slug: str,
    auto_repair_enabled: bool,
    auto_repair_allow_direct_module_install: bool,
    auto_repair_module_package_map: dict[str, list[str]],
    auto_repair_module_alias_map: dict[str, list[str]],
    auto_repair_model_package_map: dict[str, list[str]],
    auto_repair_model_fallback_map: dict[str, str],
    runtime_env: dict[str, str],
) -> dict[str, Any] | None:
    if not auto_repair_enabled:
        return None
    log_tail = tail_text(log_path)
    if not log_tail:
        return None

    missing_module = extract_missing_module(log_tail)
    unsupported_model = extract_unsupported_model_name(log_tail)
    if not missing_module and not unsupported_model:
        return None

    repair: dict[str, Any] = {
        "reason": "",
        "missing_module": missing_module or "",
        "unsupported_model": unsupported_model or "",
        "packages": [],
        "pip": {"attempted": False, "success": False, "packages": []},
        "model_fallback_applied": False,
    }

    retry_command: list[str] | None = None
    if missing_module:
        key = missing_module.strip().lower()
        packages = list(auto_repair_module_package_map.get(key, []))
        if not packages and auto_repair_allow_direct_module_install:
            packages = infer_direct_module_packages(missing_module, auto_repair_module_alias_map)
        repair["reason"] = f"missing_module:{missing_module}"
        repair["packages"] = packages
        if packages:
            install_log = logs_dir / f"{cycle_slug}_{label}_autofix_install.log"
            repair["pip"] = install_packages_with_python(
                python_exe=benchmark_python,
                cwd=benchmark_repo_root,
                packages=packages,
                log_path=install_log,
                env=runtime_env,
            )
            if bool(repair["pip"].get("success", False)):
                retry_command = list(command)
    elif unsupported_model:
        repair["reason"] = f"unsupported_model:{unsupported_model}"
        fallback = apply_model_name_fallback_to_run_command(
            repo_root=repo_root,
            command=command,
            unsupported_model=unsupported_model,
            fallback_map=auto_repair_model_fallback_map,
            label=label,
        )
        if fallback is not None:
            retry_command, patched_config = fallback
            repair["model_fallback_applied"] = True
            repair["patched_config_path"] = str(patched_config)
        else:
            packages = list(auto_repair_model_package_map.get(unsupported_model.strip().lower(), []))
            repair["packages"] = packages
            if packages:
                install_log = logs_dir / f"{cycle_slug}_{label}_autofix_install.log"
                repair["pip"] = install_packages_with_python(
                    python_exe=benchmark_python,
                    cwd=benchmark_repo_root,
                    packages=packages,
                    log_path=install_log,
                    env=runtime_env,
                )
                if bool(repair["pip"].get("success", False)):
                    retry_command = list(command)

    if retry_command is None and not repair["model_fallback_applied"] and not bool(repair["pip"].get("attempted", False)):
        return None
    repair["retry_command"] = retry_command or []
    return repair


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


def build_action_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "action",
            "rationale",
            "label",
            "runtime_tier",
            "config_path",
            "config_yaml",
            "parent_experiment_id",
            "experiment_id",
            "packages",
            "download_url",
            "download_path",
            "code_edits",
            "notes",
        ],
        "properties": {
            "action": {"type": "string", "enum": list(ACTION_VALUES)},
            "rationale": {"type": "string"},
            "label": {"type": "string"},
            "runtime_tier": {"type": "string"},
            "config_path": {"type": "string"},
            "config_yaml": {"type": "string"},
            "parent_experiment_id": {"type": "string"},
            "experiment_id": {"type": "string"},
            "packages": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 8,
            },
            "download_url": {"type": "string"},
            "download_path": {"type": "string"},
            "code_edits": {
                "type": "array",
                "maxItems": 8,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["path", "content"],
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                },
            },
            "notes": {"type": "string"},
        },
    }


def coerce_action(raw_action: Any) -> dict[str, Any]:
    raw = raw_action if isinstance(raw_action, dict) else {}
    action = str(raw.get("action", "blocked")).strip().lower()
    if action not in ACTION_VALUES:
        action = "blocked"
    packages_raw = raw.get("packages", [])
    packages: list[str] = []
    if isinstance(packages_raw, list):
        for item in packages_raw[:8]:
            token = str(item).strip()
            if token:
                packages.append(token)
    code_edits_raw = raw.get("code_edits", [])
    code_edits: list[dict[str, str]] = []
    if isinstance(code_edits_raw, list):
        for item in code_edits_raw[:8]:
            if not isinstance(item, dict):
                continue
            path = str(item.get("path", "")).strip()
            content = str(item.get("content", ""))
            if path:
                code_edits.append({"path": path, "content": content})
    return {
        "action": action,
        "rationale": " ".join(str(raw.get("rationale", "")).strip().split())[:1200],
        "label": slugify(str(raw.get("label", "")).strip())[:80],
        "runtime_tier": str(raw.get("runtime_tier", "")).strip(),
        "config_path": str(raw.get("config_path", "")).strip(),
        "config_yaml": str(raw.get("config_yaml", "")),
        "parent_experiment_id": str(raw.get("parent_experiment_id", "")).strip(),
        "experiment_id": str(raw.get("experiment_id", "")).strip(),
        "packages": packages,
        "download_url": str(raw.get("download_url", "")).strip(),
        "download_path": str(raw.get("download_path", "")).strip(),
        "code_edits": code_edits,
        "notes": " ".join(str(raw.get("notes", "")).strip().split())[:1200],
    }


def validate_action(action: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if len(str(action.get("rationale", ""))) < 20:
        issues.append("rationale_too_short")
    kind = str(action.get("action", ""))
    if kind in {"baseline", "run_config"} and not str(action.get("runtime_tier", "")).strip():
        issues.append("missing_runtime_tier")
    if kind == "run_config":
        if not str(action.get("config_path", "")).strip():
            issues.append("missing_config_path")
        if not str(action.get("config_yaml", "")).strip():
            issues.append("missing_config_yaml")
        if not str(action.get("label", "")).strip():
            issues.append("missing_label")
        if action.get("code_edits") and not isinstance(action.get("code_edits"), list):
            issues.append("invalid_code_edits")
    elif kind == "test":
        if not str(action.get("experiment_id", "")).strip():
            issues.append("missing_experiment_id")
    elif kind == "install_package":
        if not action.get("packages"):
            issues.append("missing_packages")
    elif kind == "download_file":
        if not str(action.get("download_url", "")).strip():
            issues.append("missing_download_url")
        if not str(action.get("download_path", "")).strip():
            issues.append("missing_download_path")
    elif kind == "blocked":
        if not str(action.get("notes", "")).strip():
            issues.append("blocked_requires_notes")
    return issues


def validate_action_policy(
    *,
    action: dict[str, Any],
    search_space_name: str,
    deadline: datetime,
    done_allowed_within_minutes: int,
    stop_flag_exists: bool,
) -> list[str]:
    issues: list[str] = []
    kind = str(action.get("action", "")).strip().lower()
    if kind == "done" and not stop_flag_exists:
        remaining_seconds = (deadline - datetime.now(timezone.utc)).total_seconds()
        if remaining_seconds > max(0, int(done_allowed_within_minutes)) * 60:
            issues.append(
                "done_too_early:"
                f" remaining_hours={round(remaining_seconds / 3600.0, 2)}"
                f" guard_minutes={int(done_allowed_within_minutes)}"
            )
    if kind == "done" and search_space_name == "open":
        note_text = " ".join(
            [
                str(action.get("rationale", "")).strip(),
                str(action.get("notes", "")).strip(),
            ]
        ).lower()
        if "block" in note_text:
            issues.append("use_blocked_not_done_for_external_blockers")
    return issues


def normalize_config_yaml_text(raw_yaml: str) -> str:
    text = str(raw_yaml).strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    if "\\n" in text and text.count("\n") <= 1:
        text = text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def repair_common_yaml_glitches(yaml_text: str) -> str:
    repaired = str(yaml_text)
    # Common streamed-output glitch: a stray leading character gets attached
    # to an indented key, e.g. `n  bce_weight: 0.4`.
    repaired = re.sub(
        r"(?m)^[A-Za-z]([ \t]{2,}[A-Za-z_][A-Za-z0-9_]*\s*:)",
        r"\1",
        repaired,
    )
    return repaired


def sanitize_config_yaml_text(raw_yaml: str) -> str:
    text = normalize_config_yaml_text(raw_yaml)
    candidate_texts = [text]
    repaired = repair_common_yaml_glitches(text)
    if repaired != text:
        candidate_texts.append(repaired)
    last_error: Exception | None = None
    for candidate in candidate_texts:
        try:
            payload = yaml.safe_load(candidate)
        except yaml.YAMLError as exc:
            last_error = exc
            continue
        if not isinstance(payload, dict):
            raise RuntimeError("config_yaml must parse to a YAML mapping")
        return yaml.safe_dump(payload, sort_keys=False, default_flow_style=False).rstrip() + "\n"
    raise RuntimeError(f"config_yaml is not valid YAML: {last_error}")


def load_yaml_mapping_from_text(yaml_text: str) -> dict[str, Any]:
    payload = yaml.safe_load(yaml_text)
    if not isinstance(payload, dict):
        raise RuntimeError("config_yaml must parse to a YAML mapping")
    return payload


def extract_model_name_from_yaml_text(yaml_text: str) -> str:
    if not yaml_text:
        return ""
    block = re.search(r"(?ms)^\s*model\s*:\s*\n(?P<body>(?:^[ \t]+.*\n?)*)", yaml_text)
    if not block:
        return ""
    body = str(block.group("body"))
    match = re.search(r"(?m)^[ \t]+name\s*:\s*['\"]?([A-Za-z0-9_\-\.]+)['\"]?\s*$", body)
    if not match:
        return ""
    return str(match.group(1)).strip().lower()


def nested_get(payload: dict[str, Any], *path: str) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def classify_same_family_change_axes(candidate_cfg: dict[str, Any], best_cfg: dict[str, Any]) -> set[str]:
    axes: set[str] = set()
    if any(
        nested_get(candidate_cfg, "model", key) != nested_get(best_cfg, "model", key)
        for key in ("backbone", "pretrained", "pretrained_backbone", "base_channels", "cls_hidden", "cls_dropout")
    ):
        axes.add("model_capacity")
    if any(
        nested_get(candidate_cfg, "input", key) != nested_get(best_cfg, "input", key)
        for key in ("image_size", "preserve_aspect")
    ):
        axes.add("input_scale")
    if any(
        nested_get(candidate_cfg, "training", key) != nested_get(best_cfg, "training", key)
        for key in ("epochs", "max_train_batches", "max_eval_batches", "batch_size")
    ):
        axes.add("training_budget")
    if any(
        nested_get(candidate_cfg, "training", key) != nested_get(best_cfg, "training", key)
        for key in ("learning_rate", "min_lr", "weight_decay", "scheduler")
    ):
        axes.add("optimization")
    if any(
        nested_get(candidate_cfg, "training", key) != nested_get(best_cfg, "training", key)
        for key in (
            "balanced_sampling",
            "patch_enabled",
            "patch_size",
            "patch_positive_prob",
            "patch_hard_negative_prob",
            "patch_hard_negative_quantile",
        )
    ):
        axes.add("sampling")
    if any(
        nested_get(candidate_cfg, "loss", key) != nested_get(best_cfg, "loss", key)
        for key in (
            "name",
            "bce_weight",
            "dice_weight",
            "dice_positive_only",
            "presence_bce_weight",
            "presence_bce_warmup_epochs",
        )
    ):
        axes.add("loss")
    if any(
        nested_get(candidate_cfg, "runtime", key) != nested_get(best_cfg, "runtime", key)
        for key in ("amp",)
    ):
        axes.add("runtime")
    return axes


def classify_proposal_kind(
    *,
    proposed_model_name: str,
    candidate_cfg: dict[str, Any],
    best_model_name: str,
    best_cfg: dict[str, Any] | None,
    has_code_edits: bool,
    same_family_broad_jump_min_axes: int,
) -> tuple[str, set[str]]:
    if has_code_edits:
        return "code_edit_experiment", set()
    if not best_cfg or not best_model_name or not proposed_model_name:
        return "cross_family_jump", set()
    if proposed_model_name != best_model_name:
        return "cross_family_jump", set()
    axes = classify_same_family_change_axes(candidate_cfg, best_cfg)
    if len(axes) >= max(1, int(same_family_broad_jump_min_axes)):
        return "same_family_broad_jump", axes
    return "same_family_micro_tweak", axes


def model_name_from_result_row(repo_root: pathlib.Path, row: dict[str, str]) -> str:
    for key in ("resolved_config_path", "config_path"):
        raw = str(row.get(key, "")).strip()
        if not raw:
            continue
        path = resolve_repo_path(repo_root, raw)
        text = try_read_text(path)
        name = extract_model_name_from_yaml_text(text)
        if name:
            return name
    return ""


def recent_non_keep_streak(
    *,
    results_path: pathlib.Path,
    tier: str,
) -> int:
    rows = load_results_rows(results_path)
    streak = 0
    for row in reversed(rows):
        if row.get("runtime_tier") != tier:
            continue
        status = str(row.get("status", "")).strip().lower()
        if status in {"baseline", "keep"}:
            break
        streak += 1
    return streak


def recent_plain_simple_unet_streak(
    *,
    repo_root: pathlib.Path,
    results_path: pathlib.Path,
    tier: str,
) -> int:
    rows = load_results_rows(results_path)
    streak = 0
    for row in reversed(rows):
        if row.get("runtime_tier") != tier:
            continue
        status = str(row.get("status", "")).strip().lower()
        if status == "baseline":
            break
        model_name = model_name_from_result_row(repo_root, row)
        if model_name != "simple_unet":
            break
        streak += 1
    return streak


def normalize_generated_config_path(repo_root: pathlib.Path, raw_path: str) -> pathlib.Path:
    candidate = pathlib.Path(str(raw_path).strip())
    if candidate.is_absolute():
        raise ValueError("config_path must be repo-relative")
    resolved = (repo_root / candidate).resolve()
    generated_root = (repo_root / "generated_configs").resolve()
    if generated_root not in resolved.parents and resolved != generated_root:
        raise ValueError("config_path must stay under generated_configs/")
    if resolved.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("config_path must end in .yaml or .yml")
    return resolved


def normalize_benchmark_code_path(benchmark_repo_root: pathlib.Path, raw_path: str) -> pathlib.Path:
    candidate = pathlib.Path(str(raw_path).strip())
    if candidate.is_absolute():
        raise ValueError("code_edit path must be benchmark-repo-relative")
    resolved = (benchmark_repo_root / candidate).resolve()
    src_root = (benchmark_repo_root / "src").resolve()
    if src_root not in resolved.parents and resolved != src_root:
        raise ValueError("code_edit path must stay under benchmark src/")
    suffix = resolved.suffix.lower()
    if suffix not in {".py", ".pyi"}:
        raise ValueError("code_edit path must target a Python source file")
    return resolved


def normalize_download_path(repo_root: pathlib.Path, raw_path: str) -> pathlib.Path:
    candidate = pathlib.Path(str(raw_path).strip())
    if candidate.is_absolute():
        raise ValueError("download_path must be repo-relative")
    resolved = (repo_root / candidate).resolve()
    downloads_root = (repo_root / "downloads").resolve()
    if downloads_root not in resolved.parents and resolved != downloads_root:
        raise ValueError("download_path must stay under downloads/")
    return resolved


def build_initial_prompt(
    *,
    repo_root: pathlib.Path,
    benchmark_repo_root: pathlib.Path,
    benchmark_python: pathlib.Path,
    tier: str,
    deadline_utc: str,
    done_guard_minutes: int,
    search_space_name: str,
    search_space_text: str,
    runtime_tier_text: str,
    baseline_config_path: pathlib.Path,
    baseline_config_text: str,
    status_output: str,
    results_tail: str,
    best_context: dict[str, str],
    progress_text: str,
) -> str:
    best_section = "No kept run exists yet for this tier."
    if best_context.get("experiment_id"):
        best_section = (
            f"Current best experiment: {best_context['experiment_id']}\n"
            f"Current best {best_context.get('metric_key') or 'metric'}: {best_context['metric']}\n"
            f"Current best config path: {best_context['config_path']}\n"
            f"Current best config text:\n{best_context['config_text'] or '(unavailable)'}"
        )

    return f"""You are the single research agent for this repo.

Context:
- repo={repo_root}
- benchmark_repo={benchmark_repo_root}
- benchmark_python={benchmark_python}
- tier={tier}
- deadline_utc={deadline_utc}
- search_space={search_space_name}

Return exactly one JSON action. Do not execute shell commands or edit files directly; the wrapper does that. The wrapper may auto-repair one failed run_config once.

Core rules:
- Optimize the validation metric stack `roc_auc_presence > average_precision_presence > best_f1_presence > dice_pos`.
- `runtime_tier` is a comparison bucket only, not an automatic time cap. The config itself must define the real training budget.
- Compare only within the same runtime tier.
- Aim for 2026+ SOTA-level ideas when feasible.
- Use transdomain transfer when the mapping is credible.
- Do not tune on test.
- Do not edit ../xray_fracture_benchmark/scripts.
- Do not change datasets, labels, manifests, or splits.
- In limited mode, stay config-only.
- In open mode, benchmark src/ code edits are allowed via run_config.code_edits under ../xray_fracture_benchmark/src only, paired with a concrete run and bounded to one hypothesis.
- Treat recoverable crashes as repair targets, not negative metric evidence.
- Matched scores are discard, but they still matter as tie evidence.
- Near-best runs and tradeoff runs should be noted as review-worthy alternates in the ledger so the LLM can decide whether to revisit, promote, or ignore them.
- Longer bounded programming work is allowed if it is the cleanest way to test a strong idea.
- If a model family or kept run looks promising but still limited by the current benchmark surface, prefer extending that promising direction with benchmark src code edits rather than only retuning configs around it.
- If recent attempts in this tier are still plain `simple_unet` config-only runs and none improve, do not propose another plain `simple_unet` config-only run. Switch model family or use benchmark src code edits.
- If a tier has plateaued for several cycles without a keep, do not keep proposing same-family micro-tweaks. Use another family, a same-family broad jump, or benchmark src code edits.
- A same-family broad jump should usually change at least two meaningful axes such as backbone/pretraining, resolution/aspect handling, training budget, sampling regime, or loss structure.
- Treat wrapper policy rejections as feedback and continue searching; they are not terminal blockers.
- Do not choose done early. If more than about {done_guard_minutes} minutes remain before {deadline_utc} and meaningful directions still exist, keep searching.

Search expectations:
- In open search, actively use web search, including promising ideas from other domains.
- If loop progress says `research_refresh_due=true`, do web search before the next proposal and let that research materially affect the experiment choice.
- Broaden if experiment_summary.tsv shows one narrow architecture or hyperparameter basin.
- Architecture changes, model-family changes, new heads, helper modules, and end-to-end benchmark-side components are valid when justified.
- Prefer coherent 1-4 change hypotheses, not random bundles.
- Do not spend the whole run on tiny local retuning unless results clearly justify it.
- Set the actual budget in config_yaml. Choose epochs, max_train_batches, max_eval_batches, scheduler, and other budget controls based on the method change.
- State the budget reasoning briefly in `notes` for every baseline or run_config.
- Longer budgets are expected for heavier changes such as no-pretrain runs, higher resolution, larger backbones, or new code paths when that is methodologically justified.

Actions:
- baseline
- run_config
- test
- install_package
- download_file
- done
- blocked

run_config must include:
- action="run_config"
- label
- runtime_tier
- config_path under generated_configs/
- full config_yaml
- optional parent_experiment_id
- optional code_edits with benchmark-repo-relative src/ file paths and full file content

Search-space policy:
{search_space_text}

Runtime tiers:
{runtime_tier_text}

Loop progress:
{progress_text}

Status:
{status_output}

Recent results:
{results_tail}

Experiment summary:
{read_text_limited(repo_root / "experiment_summary.tsv", max_chars=5000) if (repo_root / "experiment_summary.tsv").exists() else "experiment_summary.tsv does not exist yet."}

Baseline config:
- path={baseline_config_path}
- text:
{baseline_config_text}

Current best:
{best_section}

Return JSON only.
"""


def build_resume_prompt(
    *,
    repo_root: pathlib.Path,
    tier: str,
    deadline_utc: str,
    done_guard_minutes: int,
    search_space_name: str,
    runtime_tier_text: str,
    status_output: str,
    results_tail: str,
    best_context: dict[str, str],
    previous_cycle_summary: str,
    previous_action: dict[str, Any] | None,
    previous_execution: dict[str, Any] | None,
    progress_text: str,
) -> str:
    best_section = "No kept run exists yet for this tier."
    if best_context.get("experiment_id"):
        best_section = (
            f"Current best experiment: {best_context['experiment_id']}\n"
            f"Current best {best_context.get('metric_key') or 'metric'}: {best_context['metric']}\n"
            f"Current best config path: {best_context['config_path']}\n"
            f"Current best config text:\n{best_context['config_text'] or '(unavailable)'}"
        )

    previous_action_json = json.dumps(previous_action or {}, ensure_ascii=True, indent=2)
    previous_execution_json = json.dumps(previous_execution or {}, ensure_ascii=True, indent=2)

    return f"""Continue the same single-agent research thread in {repo_root}.

Context:
- tier={tier}
- deadline_utc={deadline_utc}
- search_space={search_space_name}

Return exactly one JSON action. Do not execute shell commands directly. The wrapper may auto-repair one failed run_config once.

Keep these rules active:
- Optimize the validation metric stack `roc_auc_presence > average_precision_presence > best_f1_presence > dice_pos`.
- `runtime_tier` is a comparison bucket only, not an automatic time cap. The config itself must define the real training budget.
- Compare only within the same runtime tier.
- Aim for 2026+ SOTA-level ideas when feasible.
- Use transdomain transfer when credible.
- In open search, use web search regularly.
- If loop progress says `research_refresh_due=true`, do web search before the next proposal and let that research materially affect the experiment choice.
- Treat recoverable crashes as repair targets.
- Treat matched scores as ties, not improvements.
- Near-best runs and tradeoff runs should be noted as review-worthy alternates in the ledger so the LLM can decide whether to revisit, promote, or ignore them.
- Broaden if the search collapses into one narrow basin.
- If recent attempts in this tier are still plain `simple_unet` config-only runs and none improve, do not propose another plain `simple_unet` config-only run. Switch model family or use benchmark src code edits.
- If a tier has plateaued for several cycles without a keep, do not keep proposing same-family micro-tweaks. Use another family, a same-family broad jump, or benchmark src code edits.
- A same-family broad jump should usually change at least two meaningful axes such as backbone/pretraining, resolution/aspect handling, training budget, sampling regime, or loss structure.
- Benchmark src/ code edits are allowed only under ../xray_fracture_benchmark/src via run_config.code_edits, paired with one concrete hypothesis.
- Longer bounded programming work is allowed when it is the cleanest path to a strong experiment.
- If a model family or kept run looks promising but still limited by the current benchmark surface, prefer extending that promising direction with benchmark src code edits rather than only retuning configs around it.
- Set the actual budget in config_yaml and state the budget reasoning briefly in `notes` for every baseline or run_config.
- Treat wrapper policy rejections as feedback and keep searching; they are not terminal blockers.
- If loop progress says `recommended_runtime_tier=long`, use `long` by default for finalist tie-down unless there is a strong reason to stay in `medium`.
- Do not tune on test, edit benchmark scripts, or change datasets/splits.
- Do not choose done early. If more than about {done_guard_minutes} minutes remain before {deadline_utc} and meaningful directions still exist, keep searching.

Runtime tiers:
{runtime_tier_text}

Loop progress:
{progress_text}

Previous cycle summary:
{previous_cycle_summary or "(none)"}

Previous action:
{previous_action_json}

Previous execution:
{previous_execution_json}

Status:
{status_output}

Recent results:
{results_tail}

Experiment summary:
{read_text_limited(repo_root / "experiment_summary.tsv", max_chars=5000) if (repo_root / "experiment_summary.tsv").exists() else "experiment_summary.tsv does not exist yet."}

Current best:
{best_section}

Return JSON only.
"""


def call_codex(
    *,
    codex_exe: str,
    repo_root: pathlib.Path,
    codex_home: pathlib.Path,
    thread_id_file: pathlib.Path,
    logs_dir: pathlib.Path,
    model: str,
    reasoning_effort: str,
    web_search_mode: str,
    network_access_enabled: bool,
    sandbox_mode: str,
    skip_git_repo_check: bool,
    add_dirs: list[pathlib.Path],
    prompt: str,
    env: dict[str, str],
) -> dict[str, Any]:
    io_dir = ensure_dir(logs_dir / "_codex_io")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    uid = uuid.uuid4().hex[:8]
    schema_path = io_dir / f"schema_{stamp}_{uid}.json"
    stdout_path = logs_dir / f"codex_cycle_{stamp}_{uid}.jsonl"
    stderr_path = logs_dir / f"codex_cycle_{stamp}_{uid}.stderr.log"
    schema_path.write_text(json.dumps(build_action_schema(), ensure_ascii=True), encoding="utf-8")

    cmd = [
        codex_exe,
        "exec",
        "--json",
        "--model",
        model,
        "--sandbox",
        sandbox_mode,
        "--cd",
        str(repo_root),
        "--output-schema",
        str(schema_path),
    ]
    if skip_git_repo_check:
        cmd.append("--skip-git-repo-check")
    cmd.extend(["--config", f'model_reasoning_effort="{reasoning_effort}"'])
    cmd.extend(["--config", f'web_search="{web_search_mode}"'])
    cmd.extend(
        [
            "--config",
            "sandbox_workspace_write.network_access="
            + ("true" if network_access_enabled else "false"),
        ]
    )
    for add_dir in add_dirs:
        cmd.extend(["--add-dir", str(add_dir)])

    thread_id = try_read_text(thread_id_file).strip()
    if thread_id:
        cmd.extend(["resume", thread_id])

    proc = subprocess.run(
        cmd,
        input=prompt,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    stdout_path.write_text(proc.stdout or "", encoding="utf-8")
    stderr_path.write_text(proc.stderr or "", encoding="utf-8")

    parsed_result: dict[str, Any] | None = None
    seen_thread_id = thread_id
    turn_failed = ""
    event_errors: list[str] = []
    event_counts: Counter[str] = Counter()
    item_type_counts: Counter[str] = Counter()
    tool_signals: set[str] = set()
    used_web_search = False
    parsed_event_lines = 0

    for raw_line in (proc.stdout or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except Exception:
            continue
        parsed_event_lines += 1
        event_type = str(event.get("type", "")).strip()
        if event_type:
            event_counts[event_type] += 1
        event_dump = json.dumps(event, ensure_ascii=True).lower()
        if any(token in event_dump for token in ("web_search", "search_query", "image_query", "internet")):
            used_web_search = True
        for field in ("tool_name", "name", "server_name", "connector_name"):
            value = event.get(field)
            if isinstance(value, str) and value.strip():
                tool_signals.add(value.strip())

        if event_type == "thread.started":
            seen_thread_id = str(event.get("thread_id", "")).strip() or seen_thread_id
        elif event_type == "item.completed":
            item = event.get("item")
            if isinstance(item, dict):
                item_type = str(item.get("type", "")).strip()
                if item_type:
                    item_type_counts[item_type] += 1
                item_dump = json.dumps(item, ensure_ascii=True).lower()
                if any(token in item_dump for token in ("web_search", "search_query", "image_query", "internet")):
                    used_web_search = True
                if item.get("type") == "agent_message":
                    msg = item.get("text")
                    if isinstance(msg, str):
                        try:
                            obj = json.loads(msg)
                        except Exception:
                            obj = None
                        if isinstance(obj, dict):
                            parsed_result = obj
        elif event_type == "turn.failed":
            error = event.get("error")
            if isinstance(error, dict):
                turn_failed = str(error.get("message", "")).strip()
        elif event_type == "error":
            msg = str(event.get("message", "")).strip()
            if msg:
                event_errors.append(msg)

    if seen_thread_id:
        thread_id_file.parent.mkdir(parents=True, exist_ok=True)
        thread_id_file.write_text(seen_thread_id, encoding="utf-8")

    telemetry = {
        "trace_file": str(stdout_path),
        "stderr_file": str(stderr_path),
        "event_counts": dict(event_counts),
        "item_type_counts": dict(item_type_counts),
        "parsed_event_lines": parsed_event_lines,
        "used_web_search": used_web_search,
        "tool_signals": sorted(tool_signals)[:24],
    }
    if parsed_result is not None:
        return {
            "returncode": int(proc.returncode),
            "thread_id": seen_thread_id,
            "action_payload": parsed_result,
            "telemetry": telemetry,
        }

    details: list[str] = []
    if turn_failed:
        details.append("turn_failed=" + turn_failed)
    if event_errors:
        details.append("errors=" + " | ".join(event_errors[-3:]))
    if proc.stderr.strip():
        details.append("stderr=" + proc.stderr.strip())
    if proc.returncode != 0:
        details.append("exit_code=" + str(proc.returncode))
    if not details:
        details.append("codex result missing or unparsable")
    fail_log = logs_dir / "codex_last_failure.log"
    fail_log.write_text(
        json.dumps(
            {
                "ts_utc": utc_now(),
                "cmd": cmd,
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "turn_failed": turn_failed,
                "event_errors": event_errors,
                "telemetry": telemetry,
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    raise RuntimeError("codex runner failed: " + "; ".join(details))


def which_codex() -> str:
    if os.name == "nt":
        cmd = shutil.which("codex.cmd")
        if cmd:
            return cmd
    return shutil.which("codex") or ""


def execute_wrapper_action(
    *,
    action: dict[str, Any],
    repo_root: pathlib.Path,
    app_cfg: dict[str, Any],
    benchmark_repo_root: pathlib.Path,
    benchmark_python: pathlib.Path,
    controller_python: pathlib.Path,
    default_tier: str,
    search_space_name: str,
    cycle_index: int,
    logs_dir: pathlib.Path,
    auto_repair_enabled: bool,
    auto_repair_retry_on_success: bool,
    auto_repair_allow_direct_module_install: bool,
    auto_repair_module_package_map: dict[str, list[str]],
    auto_repair_module_alias_map: dict[str, list[str]],
    auto_repair_model_package_map: dict[str, list[str]],
    auto_repair_model_fallback_map: dict[str, str],
    runtime_env: dict[str, str],
    same_family_micro_tweak_streak: int,
    same_family_broad_jump_min_axes: int,
) -> dict[str, Any]:
    kind = str(action["action"])
    run_loop_path = repo_root / "run_loop.py"
    cycle_slug = f"cycle_{cycle_index:04d}_{kind}"
    selected_tier = str(action.get("runtime_tier", "")).strip() or default_tier
    if kind in {"baseline", "run_config"} and selected_tier not in app_cfg.get("runtime_tiers", {}):
        raise PolicyRejectError(f"unknown runtime_tier requested by agent: {selected_tier}")

    if kind == "done":
        return {
            "status": "done",
            "message": str(action.get("notes", "")).strip() or "agent declared work complete",
        }

    if kind == "blocked":
        return {
            "status": "terminal_blocked",
            "message": str(action.get("notes", "")).strip() or "agent reported a blocker",
        }

    if kind == "baseline":
        log_path = logs_dir / f"{cycle_slug}.log"
        command = [
            str(controller_python),
            str(run_loop_path),
            "baseline",
            "--tier",
            selected_tier,
        ]
        outcome = run_logged_command(cmd=command, cwd=repo_root, log_path=log_path, env=runtime_env)
        outcome.update({"status": "executed", "action": kind, "command": command, "runtime_tier": selected_tier})
        return outcome

    if kind == "run_config":
        config_path = normalize_generated_config_path(repo_root, str(action["config_path"]))
        config_path.parent.mkdir(parents=True, exist_ok=True)
        yaml_text = sanitize_config_yaml_text(str(action["config_yaml"]))
        candidate_cfg = load_yaml_mapping_from_text(yaml_text)
        code_edits = action.get("code_edits", [])
        proposed_model_name = extract_model_name_from_yaml_text(yaml_text)
        results_path = resolve_repo_path(repo_root, str(app_cfg["results_file"]))
        results = load_results_rows(results_path)
        selection_metric = str(app_cfg["selection_metric"])
        best_row = current_best_result(results, selected_tier, selection_metric)
        best_model_name = model_name_from_result_row(repo_root, best_row) if best_row else ""
        best_cfg: dict[str, Any] | None = None
        if best_row:
            for key in ("resolved_config_path", "config_path"):
                raw = str(best_row.get(key, "")).strip()
                if not raw:
                    continue
                resolved = resolve_repo_path(repo_root, raw)
                try:
                    best_cfg = load_yaml_mapping_from_text(try_read_text(resolved))
                except Exception:
                    continue
                if best_cfg:
                    break
        proposal_kind, changed_axes = classify_proposal_kind(
            proposed_model_name=proposed_model_name,
            candidate_cfg=candidate_cfg,
            best_model_name=best_model_name,
            best_cfg=best_cfg,
            has_code_edits=bool(code_edits),
            same_family_broad_jump_min_axes=same_family_broad_jump_min_axes,
        )
        if (
            search_space_name == "open"
            and not code_edits
            and proposed_model_name == "simple_unet"
            and recent_plain_simple_unet_streak(
                repo_root=repo_root,
                results_path=results_path,
                tier=selected_tier,
            )
            >= PLAIN_SIMPLE_UNET_STREAK_LIMIT
        ):
            raise PolicyRejectError(
                "plain simple_unet search is stuck in this tier; propose a different model family or use benchmark src code_edits"
            )
        if (
            search_space_name == "open"
            and not code_edits
            and proposed_model_name
            and best_model_name
            and proposed_model_name == best_model_name
            and proposal_kind == "same_family_micro_tweak"
            and recent_non_keep_streak(results_path=results_path, tier=selected_tier) >= max(1, same_family_micro_tweak_streak)
        ):
            raise PolicyRejectError(
                "search plateaued in the current best model family; this proposal is still a micro-tweak. "
                f"Changed axes={sorted(changed_axes)}. Propose another family, a same-family broad jump, or benchmark src code_edits."
            )
        config_path.write_text(yaml_text, encoding="utf-8")
        if code_edits and search_space_name != "open":
            raise PolicyRejectError("code_edits are allowed only in open search")

        label = str(action.get("label", "")).strip() or "candidate"
        command = [
            str(controller_python),
            str(run_loop_path),
            "run-config",
            "--config",
            str(config_path),
            "--tier",
            selected_tier,
            "--label",
            label,
        ]
        parent_experiment_id = str(action.get("parent_experiment_id", "")).strip()
        if parent_experiment_id:
            command.extend(["--parent-experiment-id", parent_experiment_id])
        log_path = logs_dir / f"{cycle_slug}_{label}.log"
        backups: list[tuple[pathlib.Path, bool, str]] = []
        edited_paths: list[str] = []
        if code_edits:
            best_row = current_best_result(results, selected_tier, selection_metric)
            current_best_id = str(best_row.get("experiment_id", "")).strip() if best_row else ""
            if current_best_id and parent_experiment_id and parent_experiment_id != current_best_id:
                raise PolicyRejectError("code_edits currently require parent_experiment_id to match the current kept best for that tier")
            for edit in code_edits:
                target_path = normalize_benchmark_code_path(benchmark_repo_root, str(edit.get("path", "")))
                existed = target_path.exists()
                original_text = target_path.read_text(encoding="utf-8") if existed else ""
                backups.append((target_path, existed, original_text))
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_text(str(edit.get("content", "")), encoding="utf-8")
                edited_paths.append(rel_or_abs(target_path, benchmark_repo_root))
        try:
            outcome = run_logged_command(cmd=command, cwd=repo_root, log_path=log_path, env=runtime_env)
        except Exception:
            for target_path, existed, original_text in reversed(backups):
                if existed:
                    target_path.write_text(original_text, encoding="utf-8")
                elif target_path.exists():
                    target_path.unlink()
            raise
        repair = None
        retry_outcome = None
        final_outcome = outcome
        if (
            auto_repair_enabled
            and auto_repair_retry_on_success
            and int(outcome.get("returncode", 0)) != 0
        ):
            repair = attempt_auto_repair(
                repo_root=repo_root,
                benchmark_repo_root=benchmark_repo_root,
                benchmark_python=benchmark_python,
                command=command,
                log_path=log_path,
                logs_dir=logs_dir,
                label=label,
                cycle_slug=cycle_slug,
                auto_repair_enabled=auto_repair_enabled,
                auto_repair_allow_direct_module_install=auto_repair_allow_direct_module_install,
                auto_repair_module_package_map=auto_repair_module_package_map,
                auto_repair_module_alias_map=auto_repair_module_alias_map,
                auto_repair_model_package_map=auto_repair_model_package_map,
                auto_repair_model_fallback_map=auto_repair_model_fallback_map,
                runtime_env=runtime_env,
            )
            retry_command = repair.get("retry_command", []) if isinstance(repair, dict) else []
            if isinstance(retry_command, list) and retry_command:
                retry_log_path = logs_dir / f"{cycle_slug}_{label}_retry.log"
                retry_outcome = run_logged_command(cmd=retry_command, cwd=repo_root, log_path=retry_log_path, env=runtime_env)
                final_outcome = dict(retry_outcome)
                final_outcome["retry_command"] = retry_command
        if backups:
            results_after = load_results_rows(resolve_repo_path(repo_root, str(app_cfg["results_file"])))
            latest_row = results_after[-1] if results_after else {}
            latest_status = str(latest_row.get("status", "")).strip().lower()
            keep_code = latest_status in {"keep", "baseline", "candidate"}
            if not keep_code:
                for target_path, existed, original_text in reversed(backups):
                    if existed:
                        target_path.write_text(original_text, encoding="utf-8")
                    elif target_path.exists():
                        target_path.unlink()
        final_outcome.update(
            {
                "status": "executed",
                "action": kind,
                "command": command,
                "config_path": str(config_path),
                "runtime_tier": selected_tier,
                "code_edit_paths": edited_paths,
                "code_edit_retained": bool(backups) and latest_status in {"keep", "baseline", "candidate"} if backups else False,
                "proposal_kind": proposal_kind,
                "changed_axes": sorted(changed_axes),
            }
        )
        if repair is not None:
            final_outcome["auto_repair"] = repair
            final_outcome["initial_outcome"] = outcome
        if retry_outcome is not None:
            final_outcome["retry_outcome"] = retry_outcome
        return final_outcome

    if kind == "test":
        experiment_id = str(action["experiment_id"]).strip()
        log_path = logs_dir / f"{cycle_slug}_{slugify(experiment_id)}.log"
        command = [
            str(controller_python),
            str(run_loop_path),
            "test",
            "--experiment-id",
            experiment_id,
        ]
        outcome = run_logged_command(cmd=command, cwd=repo_root, log_path=log_path, env=runtime_env)
        outcome.update({"status": "executed", "action": kind, "command": command})
        return outcome

    if kind == "install_package":
        packages = [str(item).strip() for item in action.get("packages", []) if str(item).strip()]
        log_path = logs_dir / f"{cycle_slug}.log"
        command = [str(benchmark_python), "-m", "pip", "install", *packages]
        outcome = run_logged_command(cmd=command, cwd=benchmark_repo_root, log_path=log_path, env=runtime_env)
        outcome.update({"status": "executed", "action": kind, "command": command, "packages": packages})
        return outcome

    if kind == "download_file":
        url = str(action["download_url"]).strip()
        dest_path = normalize_download_path(repo_root, str(action["download_path"]))
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / f"{cycle_slug}.log"
        started = time.time()
        with log_path.open("w", encoding="utf-8") as handle:
            handle.write(f"[{utc_now()}] DOWNLOAD {url} -> {dest_path}\n")
            handle.flush()
            urllib.request.urlretrieve(url, dest_path)
        return {
            "status": "executed",
            "action": kind,
            "returncode": 0,
            "elapsed_seconds": round(time.time() - started, 3),
            "log_path": str(log_path),
            "download_path": str(dest_path),
            "download_url": url,
        }

    raise RuntimeError(f"unsupported action: {kind}")


def collect_status_snapshot(
    *,
    repo_root: pathlib.Path,
    controller_python: pathlib.Path,
    env: dict[str, str],
) -> str:
    cmd = [str(controller_python), str(repo_root / "run_loop.py"), "status", "--limit", "8"]
    rc, output = run_capture(cmd=cmd, cwd=repo_root, env=env)
    if rc == 0:
        return output or "status produced no output"
    return f"status command failed rc={rc}\n{output}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Single-agent Codex loop for mini_llm_cnn.")
    parser.add_argument("--config-path", default="config/codex_loop.json")
    parser.add_argument("--tier", default="medium")
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--search-space", choices=["open", "limited"], default="open")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    loop_cfg = load_json(resolve_repo_path(repo_root, args.config_path))
    app_cfg = load_json(repo_root / "config.json")

    benchmark_repo_root = resolve_repo_path(repo_root, str(app_cfg["benchmark_repo_root"]))
    benchmark_python = resolve_repo_path(repo_root, str(app_cfg["benchmark_python_exe"]))
    controller_python = resolve_repo_path(repo_root, str(loop_cfg["controller_python_exe"]))
    results_path = resolve_repo_path(repo_root, str(app_cfg["results_file"]))
    logs_dir = ensure_dir(resolve_repo_path(repo_root, str(app_cfg["logs_dir"])))
    ensure_dir(repo_root / "downloads")
    auto_repair_enabled = bool(loop_cfg.get("auto_repair_enabled", True))
    auto_repair_retry_on_success = bool(loop_cfg.get("auto_repair_retry_on_success", True))
    auto_repair_allow_direct_module_install = bool(loop_cfg.get("auto_repair_allow_direct_module_install", True))
    auto_repair_module_package_map = normalize_package_map(
        loop_cfg.get("auto_repair_module_package_map"),
        DEFAULT_AUTO_REPAIR_MODULE_PACKAGE_MAP,
    )
    auto_repair_module_alias_map = normalize_package_map(
        loop_cfg.get("auto_repair_module_alias_map"),
        DEFAULT_AUTO_REPAIR_MODULE_ALIAS_MAP,
    )
    auto_repair_model_package_map = normalize_package_map(
        loop_cfg.get("auto_repair_model_package_map"),
        DEFAULT_AUTO_REPAIR_MODEL_PACKAGE_MAP,
    )
    auto_repair_model_fallback_map = normalize_fallback_map(
        loop_cfg.get("auto_repair_model_fallback_map"),
        DEFAULT_AUTO_REPAIR_MODEL_FALLBACK_MAP,
    )

    codex_home = ensure_dir(resolve_repo_path(repo_root, str(loop_cfg["codex_home_dir"])))
    runtime_env = build_runtime_env(repo_root, loop_cfg, codex_home=codex_home)
    thread_id_file = resolve_repo_path(repo_root, str(loop_cfg["thread_id_file"]))
    session_state_file = resolve_repo_path(repo_root, str(loop_cfg["session_state_file"]))
    stop_flag_file = resolve_repo_path(repo_root, str(loop_cfg["stop_flag_file"]))

    search_space_doc = repo_root / ("search_space_limited.md" if args.search_space == "limited" else "search_space_open.md")
    if not controller_python.exists():
        raise SystemExit(f"controller python not found: {controller_python}")
    if not benchmark_repo_root.exists():
        raise SystemExit(f"benchmark repo root not found: {benchmark_repo_root}")
    if not benchmark_python.exists():
        raise SystemExit(f"benchmark python not found: {benchmark_python}")
    if not search_space_doc.exists():
        raise SystemExit(f"search-space doc not found: {search_space_doc}")

    codex_cmd = which_codex()
    if not codex_cmd:
        raise SystemExit("codex CLI not found in PATH")

    deadline = datetime.now(timezone.utc) + timedelta(hours=float(args.hours))
    deadline_utc = deadline.strftime("%Y-%m-%dT%H:%M:%SZ")
    session_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    session_log = logs_dir / f"codex_loop_{args.tier}_{session_stamp}.log"

    metric_key = str(app_cfg["selection_metric"])
    done_guard_minutes = int(loop_cfg.get("done_allowed_within_minutes", 30))
    allowed_runtime_tiers = list_allowed_runtime_tiers(loop_cfg, args.search_space, app_cfg)
    finalize_runtime_tiers = [
        str(item).strip()
        for item in loop_cfg.get("finalize_runtime_tiers", [])
        if str(item).strip() in app_cfg.get("runtime_tiers", {})
    ]
    same_family_micro_tweak_streak = int(loop_cfg.get("same_family_micro_tweak_streak", DEFAULT_SAME_FAMILY_MICRO_TWEAK_STREAK))
    code_edit_escalation_streak = int(loop_cfg.get("code_edit_escalation_streak", DEFAULT_CODE_EDIT_ESCALATION_STREAK))
    same_family_broad_jump_min_axes = int(loop_cfg.get("same_family_broad_jump_min_axes", DEFAULT_SAME_FAMILY_BROAD_JUMP_MIN_AXES))
    research_refresh_streak = int(loop_cfg.get("research_refresh_streak", DEFAULT_RESEARCH_REFRESH_STREAK))
    runtime_tier_text = runtime_tier_summary(app_cfg, allowed_runtime_tiers)
    baseline_config_path = benchmark_baseline_config_path(repo_root, app_cfg, benchmark_repo_root)
    baseline_config_text = read_text_limited(baseline_config_path, max_chars=6000)
    search_space_text = read_text_limited(search_space_doc, max_chars=8000)
    status_output = collect_status_snapshot(repo_root=repo_root, controller_python=controller_python, env=runtime_env)
    best_context = extract_best_config_context(
        repo_root=repo_root,
        results_path=results_path,
        tier=args.tier,
        metric_key=metric_key,
    )
    progress = build_progress_snapshot(
        repo_root=repo_root,
        results_path=results_path,
        tier=args.tier,
        metric_key=metric_key,
        session_cycles=[],
        allowed_runtime_tiers=allowed_runtime_tiers,
        finalize_runtime_tiers=finalize_runtime_tiers,
        code_edit_escalation_streak=code_edit_escalation_streak,
        research_refresh_streak=research_refresh_streak,
    )
    progress_text = format_progress_snapshot(progress, args.tier)

    session_state: dict[str, Any] = {
        "started_utc": utc_now(),
        "deadline_utc": deadline_utc,
        "tier": args.tier,
        "search_space": args.search_space,
        "search_space_doc": str(search_space_doc),
        "allowed_runtime_tiers": allowed_runtime_tiers,
        "model": str(loop_cfg["model"]),
        "reasoning_effort": str(loop_cfg["reasoning_effort"]),
        "web_search_mode": str(loop_cfg["web_search_mode"]),
        "network_access_enabled": bool(loop_cfg["network_access_enabled"]),
        "thread_id": try_read_text(thread_id_file).strip(),
        "controller_python_exe": str(controller_python),
        "benchmark_python_exe": str(benchmark_python),
        "benchmark_repo_root": str(benchmark_repo_root),
        "auto_repair_enabled": auto_repair_enabled,
        "runtime_env_overrides": loop_cfg.get("runtime_env", {}),
        "progress": progress,
        "session_log": str(session_log),
        "cycles": [],
    }
    session_state_file.write_text(json.dumps(session_state, indent=2), encoding="utf-8")

    prompt = build_initial_prompt(
        repo_root=repo_root,
        benchmark_repo_root=benchmark_repo_root,
        benchmark_python=benchmark_python,
        tier=args.tier,
        deadline_utc=deadline_utc,
        done_guard_minutes=done_guard_minutes,
        search_space_name=args.search_space,
        search_space_text=search_space_text,
        runtime_tier_text=runtime_tier_text,
        baseline_config_path=baseline_config_path,
        baseline_config_text=baseline_config_text,
        status_output=status_output,
        results_tail=latest_results_summary(results_path),
        best_context=best_context,
        progress_text=progress_text,
    )
    if args.dry_run:
        session_log.write_text(prompt, encoding="utf-8")
        print(f"dry-run prompt written to {rel_or_abs(session_log, repo_root)}")
        return 0

    login_probe = subprocess.run(
        [codex_cmd, "login", "status"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        env=runtime_env,
    )
    if login_probe.returncode != 0:
        sys.stderr.write(f"Codex is not logged in for CODEX_HOME={codex_home}\n")
        sys.stderr.write("Run scripts/login_codex.ps1 or scripts/login_codex.sh first.\n")
        return 2

    max_cycles = int(loop_cfg.get("max_cycles", 1000))
    pause_seconds = max(0, int(loop_cfg.get("cycle_pause_seconds", 5)))
    previous_cycle_summary = ""
    previous_action: dict[str, Any] | None = None
    previous_execution: dict[str, Any] | None = None

    for cycle in range(1, max_cycles + 1):
        if datetime.now(timezone.utc) >= deadline:
            break
        if stop_flag_file.exists():
            break

        cycle_started = utc_now()
        try:
            codex_result = call_codex(
                codex_exe=codex_cmd,
                repo_root=repo_root,
                codex_home=codex_home,
                thread_id_file=thread_id_file,
                logs_dir=logs_dir,
                model=str(loop_cfg["model"]),
                reasoning_effort=str(loop_cfg["reasoning_effort"]),
                web_search_mode=str(loop_cfg["web_search_mode"]),
                network_access_enabled=bool(loop_cfg["network_access_enabled"]),
                sandbox_mode=str(loop_cfg.get("sandbox_mode", "workspace-write")),
                skip_git_repo_check=bool(loop_cfg.get("skip_git_repo_check", False)),
                add_dirs=[benchmark_repo_root, benchmark_python.parent.parent],
                prompt=prompt,
                env=runtime_env,
            )
            action = coerce_action(codex_result["action_payload"])
            issues = validate_action(action)
            issues.extend(
                validate_action_policy(
                    action=action,
                    search_space_name=args.search_space,
                    deadline=deadline,
                    done_allowed_within_minutes=done_guard_minutes,
                    stop_flag_exists=stop_flag_file.exists(),
                )
            )
            if issues:
                execution: dict[str, Any] = {
                    "status": "policy_rejected",
                    "message": "action validation failed: " + ", ".join(issues),
                }
            else:
                try:
                    execution = execute_wrapper_action(
                        action=action,
                        repo_root=repo_root,
                        app_cfg=app_cfg,
                        benchmark_repo_root=benchmark_repo_root,
                        benchmark_python=benchmark_python,
                        controller_python=controller_python,
                        default_tier=args.tier,
                        search_space_name=args.search_space,
                        cycle_index=cycle,
                        logs_dir=logs_dir,
                        auto_repair_enabled=auto_repair_enabled,
                        auto_repair_retry_on_success=auto_repair_retry_on_success,
                        auto_repair_allow_direct_module_install=auto_repair_allow_direct_module_install,
                        auto_repair_module_package_map=auto_repair_module_package_map,
                        auto_repair_module_alias_map=auto_repair_module_alias_map,
                        auto_repair_model_package_map=auto_repair_model_package_map,
                        auto_repair_model_fallback_map=auto_repair_model_fallback_map,
                        runtime_env=runtime_env,
                        same_family_micro_tweak_streak=same_family_micro_tweak_streak,
                        same_family_broad_jump_min_axes=same_family_broad_jump_min_axes,
                    )
                except PolicyRejectError as exc:
                    execution = {
                        "status": "policy_rejected",
                        "message": str(exc),
                    }
                except Exception as exc:
                    execution = {
                        "status": "infra_blocked",
                        "message": str(exc),
                    }
        except Exception as exc:
            action = {
                "action": "blocked",
                "rationale": "codex execution failed",
                "label": "",
                "config_path": "",
                "config_yaml": "",
                "parent_experiment_id": "",
                "experiment_id": "",
                "packages": [],
                "download_url": "",
                "download_path": "",
                "notes": str(exc),
            }
            execution = {
                "status": "infra_blocked",
                "message": str(exc),
            }
            codex_result = {
                "returncode": 1,
                "thread_id": try_read_text(thread_id_file).strip(),
                "telemetry": {},
            }

        status_output = collect_status_snapshot(repo_root=repo_root, controller_python=controller_python, env=runtime_env)
        best_context = extract_best_config_context(
            repo_root=repo_root,
            results_path=results_path,
            tier=args.tier,
            metric_key=metric_key,
        )
        previous_cycle_summary = (
            str(execution.get("message", "")).strip()
            or str(action.get("notes", "")).strip()
            or str(action.get("rationale", "")).strip()
        )
        previous_action = action
        previous_execution = execution
        outcome_label = str(execution.get("status", "")).strip() or "unknown"
        if outcome_label == "executed" and action.get("action") in {"baseline", "run_config"}:
            latest_row = latest_result_row(results_path, tier=args.tier)
            latest_status = str((latest_row or {}).get("status", "")).strip().lower()
            if latest_status:
                outcome_label = f"executed_{latest_status}"

        cycle_record = {
            "cycle": cycle,
            "started_utc": cycle_started,
            "ended_utc": utc_now(),
            "thread_id": codex_result.get("thread_id", ""),
            "codex_returncode": int(codex_result.get("returncode", 0)),
            "telemetry": codex_result.get("telemetry", {}),
            "action": action,
            "execution": execution,
            "outcome_label": outcome_label,
        }
        session_state["thread_id"] = codex_result.get("thread_id", "")
        session_state["cycles"].append(cycle_record)
        progress = build_progress_snapshot(
            repo_root=repo_root,
            results_path=results_path,
            tier=args.tier,
            metric_key=metric_key,
            session_cycles=session_state.get("cycles", []),
            allowed_runtime_tiers=allowed_runtime_tiers,
            finalize_runtime_tiers=finalize_runtime_tiers,
            code_edit_escalation_streak=code_edit_escalation_streak,
            research_refresh_streak=research_refresh_streak,
        )
        progress_text = format_progress_snapshot(progress, args.tier)
        session_state["progress"] = progress
        session_state["last_cycle"] = cycle_record
        session_state_file.write_text(json.dumps(session_state, indent=2), encoding="utf-8")
        with session_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(cycle_record, ensure_ascii=True) + "\n")

        terminal_status = str(execution.get("status", "")).strip().lower()
        if should_stop_loop(action_kind=str(action.get("action", "")), execution_status=terminal_status):
            break

        prompt = build_resume_prompt(
            repo_root=repo_root,
            tier=args.tier,
            deadline_utc=deadline_utc,
            done_guard_minutes=done_guard_minutes,
            search_space_name=args.search_space,
            runtime_tier_text=runtime_tier_text,
            status_output=status_output,
            results_tail=latest_results_summary(results_path),
            best_context=best_context,
            previous_cycle_summary=previous_cycle_summary,
            previous_action=previous_action,
            previous_execution=previous_execution,
            progress_text=progress_text,
        )
        if pause_seconds > 0:
            time.sleep(pause_seconds)

    print(
        "codex loop complete: "
        f"log={rel_or_abs(session_log, repo_root)} "
        f"state={rel_or_abs(session_state_file, repo_root)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
