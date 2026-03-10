# mini_llm_cnn

`mini_llm_cnn` is a small autonomous research loop for `xray_fracture_benchmark`.

It is designed to be much simpler than `llm_driven_cnns`: the benchmark code stays fixed, the loop only generates and runs config variants, and all search state stays local to this repo.

## What It Does
- reuses the prepared benchmark environment and dataset by explicit path
- searches by changing YAML configs, not benchmark Python code
- selects candidates by validation `dice_pos`
- keeps `smoke`, `medium`, and `long` runtime tiers comparable
- supports unattended overnight search

## What It Does Not Do
- it does not edit benchmark training code
- it does not tune on `test`
- it does not change dataset splits or evaluation semantics
- it does not pull in new datasets

## Quick Start
From this repo root:

```powershell
& ..\xray_fracture_benchmark_venv\Scripts\python.exe .\run_loop.py status
& ..\xray_fracture_benchmark_venv\Scripts\python.exe .\run_loop.py baseline --tier smoke
& ..\xray_fracture_benchmark_venv\Scripts\python.exe .\run_loop.py step --tier medium
& ..\xray_fracture_benchmark_venv\Scripts\python.exe .\run_loop.py night-run --tier medium --hours 8
```

## Search Modes
- `open`: broader, data-driven search inside the benchmark contract
- `limited`: narrower, more conservative search

Launchers:

```powershell
& .\scripts\start_night_run.ps1 -Tier medium -Hours 8 -SearchSpace open -StartInNewWindow
```

```bash
bash ./scripts/start_night_run.sh --tier medium --hours 8 --search-space limited
```

## Main Files
- `run_loop.py`: main loop entrypoint
- `config.json`: interpreter paths, benchmark paths, runtime tiers
- `program.md`: agent operating instructions
- `search_space_open.md`: open search policy
- `search_space_limited.md`: limited search policy

## Outputs
- `generated_configs/`: generated YAML configs
- `logs/`: train/validate/test logs
- `runs/`: checkpoints and metrics
- `results.tsv`: experiment ledger, created on first run
- `.mini_loop/state.json`: local loop state, created on first run

## Rules
- baseline first
- compare only within the same runtime tier
- `keep` only when validation `dice_pos` improves
- run locked test only on selected kept or baseline experiments

See [program.md](/C:/Users/Max/code/mini_llm_cnn/program.md), [search_space_open.md](/C:/Users/Max/code/mini_llm_cnn/search_space_open.md), and [search_space_limited.md](/C:/Users/Max/code/mini_llm_cnn/search_space_limited.md) for the operating contract.
