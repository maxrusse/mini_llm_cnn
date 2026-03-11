#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
import urllib.request
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any


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
    finalize_tier = "long"
    for name in allowed_tiers:
        spec = tier_specs.get(name, {})
        approx = ""
        finalize_only = False
        if isinstance(spec, dict):
            approx_raw = spec.get("approx_minutes")
            if approx_raw is not None:
                approx = f"~{approx_raw}m"
            finalize_only = bool(spec.get("finalize_only", False))
        extras: list[str] = []
        if approx:
            extras.append(approx)
        if finalize_only or name == finalize_tier:
            extras.append("finalize-only")
        suffix = f" ({', '.join(extras)})" if extras else ""
        lines.append(f"- {name}{suffix}")
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
            "metric": "",
            "config_path": "",
            "config_text": "",
        }

    preferred = str(best.get("resolved_config_path", "")).strip() or str(best.get("config_path", "")).strip()
    config_path = resolve_repo_path(repo_root, preferred) if preferred else pathlib.Path()
    config_text = read_text_limited(config_path, max_chars=12000) if preferred and config_path.exists() else ""
    return {
        "experiment_id": str(best.get("experiment_id", "")).strip(),
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
                "maxItems": 4,
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
        for item in code_edits_raw[:4]:
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
    search_space_name: str,
    search_space_text: str,
    runtime_tier_text: str,
    baseline_config_path: pathlib.Path,
    baseline_config_text: str,
    status_output: str,
    results_tail: str,
    best_context: dict[str, str],
) -> str:
    best_section = "No kept run exists yet for this tier."
    if best_context.get("experiment_id"):
        best_section = (
            f"Current best experiment: {best_context['experiment_id']}\n"
            f"Current best dice_pos: {best_context['metric']}\n"
            f"Current best config path: {best_context['config_path']}\n"
            f"Current best config text:\n{best_context['config_text'] or '(unavailable)'}"
        )

    return f"""You are the single research agent for this repository.

Repository: {repo_root}
Benchmark repo: {benchmark_repo_root}
Benchmark Python: {benchmark_python}
Tier: {tier}
Hard deadline UTC: {deadline_utc}
Search-space mode: {search_space_name}

You are not allowed to execute shell commands directly in this turn.
You are not allowed to edit files directly in this turn.
Instead, return exactly one JSON action. The wrapper will execute it and resume the same thread with the result.

Hard rules:
- Optimize validation dice_pos only.
- Do not tune on test.
- Do not edit code in ../xray_fracture_benchmark/scripts.
- Do not change datasets, labels, manifests, or split definitions.
- Stay within the selected runtime tier.
- In limited mode, stay config-only.
- In open mode, if existing benchmark methods look exhausted, you may propose benchmark src/ code edits through run_config.code_edits.
- Open-mode code edits must stay under ../xray_fracture_benchmark/src, be tightly scoped, and be paired with a concrete config experiment.

Available wrapper actions:
- baseline: run run_loop.py baseline --tier {tier}
- run_config: write one YAML file under generated_configs/, optionally apply a small set of benchmark src/ code edits, and run it with run_loop.py run-config --tier {tier}
- test: run locked test for a finalist experiment id
- install_package: install one or more narrowly scoped packages into the benchmark venv
- download_file: download a file into downloads/
- done: stop because the overnight goal is complete
- blocked: stop because of a concrete external blocker

Search-space policy:
{search_space_text}

Runtime tiers available for this search:
{runtime_tier_text}

Budget rule:
- In open search, active exploration should use the 5m to 30m tiers.
- In open search, use web search regularly for architecture, loss, optimization, or training ideas instead of staying fully local.
- Use long only to finalize clearly strong candidates.
- Compare only against runs from the same runtime tier.
- Use experiment_summary.tsv as an exploration audit. If recent runs stay inside one model family or one narrow hyperparameter basin, broaden again.
- In open search, architecture and model-family changes are expected exploration axes, not optional extras.
- In open search, own method ideas are allowed. You may propose small code or math changes inside benchmark src/ when they are data-driven and clearly tied to a hypothesis.
- Do not spend a whole run only retuning loss weights or learning rate around one architecture unless the results clearly justify it.
- Do not underuse the GPU. When benchmark-compatible, consider stronger architectures, pretrained backbones, larger inputs, or larger batches that make meaningful use of the available budget.
- In limited search, prefer one controlled change at a time.
- In open search, you may propose a compact 2-4 change bundle when it is one coherent hypothesis and clearly faster than scalar-only local search.
- If you bundle changes, keep them centered on one idea such as architecture + backbone + batch fit, or patch enable + patch settings, not a random grab bag.

Current status output:
{status_output}

Recent results tail:
{results_tail}

Experiment summary table:
{read_text_limited(repo_root / "experiment_summary.tsv", max_chars=8000) if (repo_root / "experiment_summary.tsv").exists() else "experiment_summary.tsv does not exist yet."}

Baseline config path:
{baseline_config_path}

Baseline config text:
{baseline_config_text}

Current best context:
{best_section}

Output requirements:
- Return JSON only.
- Choose exactly one action.
- For run_config, provide:
  - action="run_config"
  - label
  - runtime_tier
  - config_path like generated_configs/<slug>.yaml
  - full config_yaml content
  - optional parent_experiment_id
  - optional code_edits as an array of benchmark-repo-relative src/ files with full file content
- For baseline, provide runtime_tier for the tier you want to establish.
- For install_package, provide packages as a JSON array.
- For download_file, provide download_url and download_path under downloads/.
- Use notes for a short expected outcome or blocker description.
- Keep changes small and data-driven.
- In open search, do not complete a whole overnight run without using web search.
- In open search, a compact hypothesis-driven bundle is allowed when it accelerates exploration.
- If you use code_edits, keep them tightly scoped and tied to one concrete method hypothesis.
"""


def build_resume_prompt(
    *,
    repo_root: pathlib.Path,
    tier: str,
    deadline_utc: str,
    search_space_name: str,
    runtime_tier_text: str,
    status_output: str,
    results_tail: str,
    best_context: dict[str, str],
    previous_cycle_summary: str,
    previous_action: dict[str, Any] | None,
    previous_execution: dict[str, Any] | None,
) -> str:
    best_section = "No kept run exists yet for this tier."
    if best_context.get("experiment_id"):
        best_section = (
            f"Current best experiment: {best_context['experiment_id']}\n"
            f"Current best dice_pos: {best_context['metric']}\n"
            f"Current best config path: {best_context['config_path']}\n"
            f"Current best config text:\n{best_context['config_text'] or '(unavailable)'}"
        )

    previous_action_json = json.dumps(previous_action or {}, ensure_ascii=True, indent=2)
    previous_execution_json = json.dumps(previous_execution or {}, ensure_ascii=True, indent=2)

    return f"""Continue the same single-agent research thread in {repo_root}.

Tier: {tier}
Hard deadline UTC: {deadline_utc}
Search-space mode: {search_space_name}

You still must not execute shell commands directly. Return exactly one JSON action for the wrapper to execute.

Runtime tiers available for this search:
{runtime_tier_text}

Budget rule:
- Use the 5m to 30m tiers for active search.
- In open search, use web search regularly for architecture, loss, optimization, or training ideas instead of staying fully local.
- Use long only to finalize clearly strong candidates.
- Compare only within the same runtime tier.
- Use experiment_summary.tsv as an exploration audit. If recent runs stay inside one model family or one narrow hyperparameter basin, broaden again.
- In open search, architecture and model-family changes are expected exploration axes, not optional extras.
- In open search, own method ideas are allowed. You may propose small code or math changes inside benchmark src/ when they are data-driven and clearly tied to a hypothesis.
- Do not spend a whole run only retuning loss weights or learning rate around one architecture unless the results clearly justify it.
- Do not underuse the GPU. When benchmark-compatible, consider stronger architectures, pretrained backbones, larger inputs, or larger batches that make meaningful use of the available budget.
- In limited search, prefer one controlled change at a time.
- In open search, you may propose a compact 2-4 change bundle when it is one coherent hypothesis and clearly faster than scalar-only local search.
- If you bundle changes, keep them centered on one idea such as architecture + backbone + batch fit, or patch enable + patch settings, not a random grab bag.
- If you use code_edits, keep them tightly scoped and pair them with a concrete config run.

Previous cycle summary:
{previous_cycle_summary or "(none)"}

Previous chosen action:
{previous_action_json}

Previous execution result:
{previous_execution_json}

Current status output:
{status_output}

Recent results tail:
{results_tail}

Experiment summary table:
{read_text_limited(repo_root / "experiment_summary.tsv", max_chars=8000) if (repo_root / "experiment_summary.tsv").exists() else "experiment_summary.tsv does not exist yet."}

Current best context:
{best_section}

Return the next single JSON action only.
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
        env={**os.environ, "CODEX_HOME": str(codex_home)},
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
) -> dict[str, Any]:
    kind = str(action["action"])
    run_loop_path = repo_root / "run_loop.py"
    cycle_slug = f"cycle_{cycle_index:04d}_{kind}"
    selected_tier = str(action.get("runtime_tier", "")).strip() or default_tier
    if kind in {"baseline", "run_config"} and selected_tier not in app_cfg.get("runtime_tiers", {}):
        raise RuntimeError(f"unknown runtime_tier requested by agent: {selected_tier}")

    if kind == "done":
        return {
            "status": "done",
            "message": str(action.get("notes", "")).strip() or "agent declared work complete",
        }

    if kind == "blocked":
        return {
            "status": "blocked",
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
        outcome = run_logged_command(cmd=command, cwd=repo_root, log_path=log_path)
        outcome.update({"status": "executed", "action": kind, "command": command, "runtime_tier": selected_tier})
        return outcome

    if kind == "run_config":
        config_path = normalize_generated_config_path(repo_root, str(action["config_path"]))
        config_path.parent.mkdir(parents=True, exist_ok=True)
        yaml_text = str(action["config_yaml"]).rstrip() + "\n"
        config_path.write_text(yaml_text, encoding="utf-8")
        code_edits = action.get("code_edits", [])
        if code_edits and search_space_name != "open":
            raise RuntimeError("code_edits are allowed only in open search")

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
            selection_metric = str(app_cfg["selection_metric"])
            results = load_results_rows(resolve_repo_path(repo_root, str(app_cfg["results_file"])))
            best_row = current_best_result(results, selected_tier, selection_metric)
            current_best_id = str(best_row.get("experiment_id", "")).strip() if best_row else ""
            if current_best_id and parent_experiment_id and parent_experiment_id != current_best_id:
                raise RuntimeError("code_edits currently require parent_experiment_id to match the current kept best for that tier")
            for edit in code_edits:
                target_path = normalize_benchmark_code_path(benchmark_repo_root, str(edit.get("path", "")))
                existed = target_path.exists()
                original_text = target_path.read_text(encoding="utf-8") if existed else ""
                backups.append((target_path, existed, original_text))
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_text(str(edit.get("content", "")), encoding="utf-8")
                edited_paths.append(rel_or_abs(target_path, benchmark_repo_root))
        try:
            outcome = run_logged_command(cmd=command, cwd=repo_root, log_path=log_path)
        except Exception:
            for target_path, existed, original_text in reversed(backups):
                if existed:
                    target_path.write_text(original_text, encoding="utf-8")
                elif target_path.exists():
                    target_path.unlink()
            raise
        if backups:
            results_after = load_results_rows(resolve_repo_path(repo_root, str(app_cfg["results_file"])))
            latest_row = results_after[-1] if results_after else {}
            latest_status = str(latest_row.get("status", "")).strip().lower()
            keep_code = latest_status in {"keep", "baseline"}
            if not keep_code:
                for target_path, existed, original_text in reversed(backups):
                    if existed:
                        target_path.write_text(original_text, encoding="utf-8")
                    elif target_path.exists():
                        target_path.unlink()
        outcome.update(
            {
                "status": "executed",
                "action": kind,
                "command": command,
                "config_path": str(config_path),
                "runtime_tier": selected_tier,
                "code_edit_paths": edited_paths,
                "code_edit_retained": bool(backups) and latest_status in {"keep", "baseline"} if backups else False,
            }
        )
        return outcome

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
        outcome = run_logged_command(cmd=command, cwd=repo_root, log_path=log_path)
        outcome.update({"status": "executed", "action": kind, "command": command})
        return outcome

    if kind == "install_package":
        packages = [str(item).strip() for item in action.get("packages", []) if str(item).strip()]
        log_path = logs_dir / f"{cycle_slug}.log"
        command = [str(benchmark_python), "-m", "pip", "install", *packages]
        outcome = run_logged_command(cmd=command, cwd=benchmark_repo_root, log_path=log_path)
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
) -> str:
    cmd = [str(controller_python), str(repo_root / "run_loop.py"), "status", "--limit", "8"]
    rc, output = run_capture(cmd=cmd, cwd=repo_root)
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

    codex_home = ensure_dir(resolve_repo_path(repo_root, str(loop_cfg["codex_home_dir"])))
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
    allowed_runtime_tiers = list_allowed_runtime_tiers(loop_cfg, args.search_space, app_cfg)
    runtime_tier_text = runtime_tier_summary(app_cfg, allowed_runtime_tiers)
    baseline_config_path = benchmark_baseline_config_path(repo_root, app_cfg, benchmark_repo_root)
    baseline_config_text = read_text_limited(baseline_config_path, max_chars=12000)
    search_space_text = read_text_limited(search_space_doc, max_chars=8000)
    status_output = collect_status_snapshot(repo_root=repo_root, controller_python=controller_python)
    best_context = extract_best_config_context(
        repo_root=repo_root,
        results_path=results_path,
        tier=args.tier,
        metric_key=metric_key,
    )

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
        search_space_name=args.search_space,
        search_space_text=search_space_text,
        runtime_tier_text=runtime_tier_text,
        baseline_config_path=baseline_config_path,
        baseline_config_text=baseline_config_text,
        status_output=status_output,
        results_tail=latest_results_summary(results_path),
        best_context=best_context,
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
        env={**os.environ, "CODEX_HOME": str(codex_home)},
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
            )
            action = coerce_action(codex_result["action_payload"])
            issues = validate_action(action)
            if issues:
                execution: dict[str, Any] = {
                    "status": "blocked",
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
                    )
                except Exception as exc:
                    execution = {
                        "status": "blocked",
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
                "status": "blocked",
                "message": str(exc),
            }
            codex_result = {
                "returncode": 1,
                "thread_id": try_read_text(thread_id_file).strip(),
                "telemetry": {},
            }

        status_output = collect_status_snapshot(repo_root=repo_root, controller_python=controller_python)
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

        cycle_record = {
            "cycle": cycle,
            "started_utc": cycle_started,
            "ended_utc": utc_now(),
            "thread_id": codex_result.get("thread_id", ""),
            "codex_returncode": int(codex_result.get("returncode", 0)),
            "telemetry": codex_result.get("telemetry", {}),
            "action": action,
            "execution": execution,
        }
        session_state["thread_id"] = codex_result.get("thread_id", "")
        session_state["last_cycle"] = cycle_record
        session_state["cycles"].append(cycle_record)
        session_state_file.write_text(json.dumps(session_state, indent=2), encoding="utf-8")
        with session_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(cycle_record, ensure_ascii=True) + "\n")

        terminal_status = str(execution.get("status", "")).strip().lower()
        if action["action"] in {"done", "blocked"}:
            break
        if terminal_status == "blocked":
            break

        prompt = build_resume_prompt(
            repo_root=repo_root,
            tier=args.tier,
            deadline_utc=deadline_utc,
            search_space_name=args.search_space,
            runtime_tier_text=runtime_tier_text,
            status_output=status_output,
            results_tail=latest_results_summary(results_path),
            best_context=best_context,
            previous_cycle_summary=previous_cycle_summary,
            previous_action=previous_action,
            previous_execution=previous_execution,
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
