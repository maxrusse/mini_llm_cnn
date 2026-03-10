# mini_llm_cnn

Minimal autonomous config-only research loop for `xray_fracture_benchmark`.

This repo is intentionally much smaller than `llm_driven_cnns`. It reuses the prepared benchmark environment and data by explicit path, but it keeps all loop state, generated configs, logs, and run outputs local to this repo.

## Design
- Config-only experimentation. The loop mutates YAML configs, not benchmark Python code.
- Validation-first selection. The primary metric is validation `dice_pos`.
- Fixed runtime tiers. `smoke`, `medium`, and `long` are defined in `config.json`.
- Locked test. Test evaluation is an explicit separate command for kept or baseline experiments only.
- Overnight mode. `night-run` keeps stepping a tier until a deadline or until the built-in mutation space is exhausted.

## Quick Start
From this repo root:

```powershell
& ..\xray_fracture_benchmark_venv\Scripts\python.exe .\run_loop.py status
& ..\xray_fracture_benchmark_venv\Scripts\python.exe .\run_loop.py baseline --tier smoke
& ..\xray_fracture_benchmark_venv\Scripts\python.exe .\run_loop.py step --tier medium
& ..\xray_fracture_benchmark_venv\Scripts\python.exe .\run_loop.py night-run --tier medium --hours 8
```

To dry-run the commands without launching training:

```powershell
& ..\xray_fracture_benchmark_venv\Scripts\python.exe .\run_loop.py baseline --tier smoke --dry-run
& ..\xray_fracture_benchmark_venv\Scripts\python.exe .\run_loop.py step --tier medium --dry-run
```

## Files
- `config.json`: local loop configuration and runtime tiers
- `program.md`: agent instructions for Codex or a similar coding agent
- `search_space_open.md`: open, data-driven search-space policy
- `search_space_limited.md`: stricter limited search-space policy
- `run_loop.py`: loop entrypoint
- `scripts/start_night_run.ps1`: convenience launcher for unattended night runs
- `scripts/start_night_run.sh`: bash launcher with the same mode selection
- `generated_configs/`: generated experiment YAMLs
- `logs/`: captured train/validate/test logs
- `runs/`: checkpoints and metrics produced by the benchmark scripts
- `results.tsv`: append-only experiment ledger, created on first run
- `.mini_loop/state.json`: local runtime state, created on first run

## Search-Space Modes
- `open`: wider, data-driven exploration inside the benchmark contract
- `limited`: narrower, more conservative config search
- PowerShell launcher: `& .\scripts\start_night_run.ps1 -Tier medium -Hours 8 -SearchSpace open`
- Bash launcher: `bash ./scripts/start_night_run.sh --tier medium --hours 8 --search-space limited`

## CLI
```powershell
& ..\xray_fracture_benchmark_venv\Scripts\python.exe .\run_loop.py status
& ..\xray_fracture_benchmark_venv\Scripts\python.exe .\run_loop.py list-mutations
& ..\xray_fracture_benchmark_venv\Scripts\python.exe .\run_loop.py baseline --tier smoke
& ..\xray_fracture_benchmark_venv\Scripts\python.exe .\run_loop.py step --tier medium [--mutation auto|lr_up|...]
& ..\xray_fracture_benchmark_venv\Scripts\python.exe .\run_loop.py night-run --tier medium --hours 8
& ..\xray_fracture_benchmark_venv\Scripts\python.exe .\run_loop.py run-config --config .\generated_configs\my_try.yaml --tier medium --label my_try
& ..\xray_fracture_benchmark_venv\Scripts\python.exe .\run_loop.py test --experiment-id e0001
& .\scripts\start_night_run.ps1 -Tier medium -Hours 8 -SearchSpace open -StartInNewWindow
bash ./scripts/start_night_run.sh --tier medium --hours 8 --search-space limited
```

## Keep/Discard Policy
- Baseline is recorded first and becomes the initial best run for its tier.
- A candidate is `keep` only if its validation `dice_pos` is strictly better than the current best for the same tier.
- Non-improving successful runs are logged as `discard`.
- Failed runs are logged as `crash`.

## Notes
- The benchmark repo remains the source of truth for metric semantics, data splits, and train/validate/test behavior.
- Runtime tiers are enforced by config overrides, not by ad hoc agent choices.
- Generated configs and run outputs are ignored by git so the repo can stay clean during autonomous search.
- If you are using Codex or another agent interactively in this repo, it is allowed to web-search for benchmark-relevant ideas, install narrowly scoped packages into `xray_fracture_benchmark_venv`, and download pretrained weights or papers. Do not let it pull new datasets or anything that changes the train/val/test contract.
