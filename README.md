# mini_llm_cnn

`mini_llm_cnn` is a small single-agent Codex research loop for `xray_fracture_benchmark`.

It is designed to be much simpler than `llm_driven_cnns`: the benchmark code stays fixed, a single Codex agent owns the search policy, and all search state stays local to this repo.

## What It Does
- reuses the prepared benchmark environment and dataset by explicit path
- uses a resumed `codex exec` session as the search agent
- lets Codex choose one structured action per cycle while the local wrapper executes it
- searches by changing YAML configs, not benchmark Python code
- selects candidates by validation `dice_pos`
- lets open search choose between real 5m, 10m, 20m, and 30m screening tiers
- reserves `long` for finalist runs up to about 2 hours
- supports unattended overnight search

## What It Does Not Do
- it does not edit benchmark training code
- it does not tune on `test`
- it does not change dataset splits or evaluation semantics
- it does not pull in new datasets

## Quick Start
From this repo root:

```powershell
& .\scripts\login_codex.ps1
& .\scripts\start_codex_loop.ps1 -Tier medium -Hours 8 -SearchSpace open -StartInNewWindow
```

## Search Modes
- `open`: broader, data-driven search inside the benchmark contract
- `limited`: narrower, more conservative search

## Runtime Tiers
- `medium_5m`: short screening run, about 5 minutes
- `medium`: default search tier, about 10 minutes
- `medium_20m`: slower follow-up search tier, about 20 minutes
- `medium_30m`: heavier search tier, about 30 minutes
- `long`: finalist tier, up to about 2 hours
- `smoke`: debug-only sanity tier

Launchers:

```powershell
& .\scripts\start_codex_loop.ps1 -Tier medium -Hours 8 -SearchSpace open -StartInNewWindow
```

```bash
bash ./scripts/start_codex_loop.sh --tier medium --hours 8 --search-space limited
```

## Main Files
- `run_loop.py`: main loop entrypoint
- `scripts/codex_loop.py`: action-wrapper that resumes one Codex thread and executes chosen actions
- `config.json`: interpreter paths, benchmark paths, runtime tiers
- `config/codex_loop.json`: Codex model and loop settings
- `program.md`: agent operating instructions
- `search_space_open.md`: open search policy
- `search_space_limited.md`: limited search policy

## Outputs
- `generated_configs/`: generated YAML configs
- `logs/`: train/validate/test logs
- `runs/`: checkpoints and metrics
- `results.tsv`: experiment ledger, created on first run
- `.mini_loop/state.json`: local loop state, created on first run
- `.mini_loop/codex_home/`: repo-local Codex login state
- `.mini_loop/codex_session.json`: current Codex loop session state
- `downloads/`: optional downloaded papers or weights

## Rules
- baseline first
- compare only within the same runtime tier
- `keep` only when validation `dice_pos` improves
- run locked test only on selected kept or baseline experiments
- create `.mini_loop/STOP_CODEX_LOOP` if you want the loop to stop after the current cycle

See [program.md](/C:/Users/Max/code/mini_llm_cnn/program.md), [search_space_open.md](/C:/Users/Max/code/mini_llm_cnn/search_space_open.md), and [search_space_limited.md](/C:/Users/Max/code/mini_llm_cnn/search_space_limited.md) for the operating contract.
