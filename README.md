# mini_llm_cnn

`mini_llm_cnn` is a small single-agent Codex research loop for `xray_fracture_benchmark`.

It is designed to be much simpler than `llm_driven_cnns`: a single Codex agent owns the search policy, the wrapper executes the chosen action, and all search state stays local to this repo.

## What It Does
- reuses the prepared benchmark environment and dataset by explicit path
- uses a resumed `codex exec` session as the search agent
- lets Codex choose one structured action per cycle while the local wrapper executes it
- primarily searches by changing YAML configs
- can also test benchmark `src/` code ideas in `open` mode, from small helper changes up to new heads, model classes, or pipeline components
- can auto-repair one failed `run_config` cycle by installing mapped packages or retrying with a configured model-name fallback
- selects candidates by the validation metric stack `roc_auc_presence > average_precision_presence > best_f1_presence > dice_pos`
- uses `runtime_tier` as a comparison bucket only; actual training budget is set per experiment in the config itself
- expects open search to actually use web search for outside ideas
- explicitly allows importing strong methods from other domains when the transfer case to fracture segmentation is plausible
- lets open search use a small coherent bundle of changes when that is faster than one-scalar-at-a-time search
- supports unattended overnight search
- keeps Torch, Hugging Face, pip, and temp caches under `.mini_loop/` by default so model downloads and temp files stay inside the repo workspace

## What It Does Not Do
- it does not edit benchmark runner scripts automatically or arbitrarily
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
- `smoke`: debug-only comparison bucket
- `medium`: main exploration and evaluation bucket
- `long`: final tie-down bucket for strongest candidates

These are bookkeeping buckets for fair comparison, not hidden epoch or batch presets. The actual training budget must be stated in each experiment config and justified by the model.

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
- `config.json`: interpreter paths, benchmark paths, comparison-bucket labels
- `config/codex_loop.json`: Codex model and loop settings
  - includes wrapper auto-repair maps for missing modules and unsupported model aliases
- `program.md`: agent operating instructions
- `search_space_open.md`: open search policy
- `search_space_limited.md`: limited search policy

## Outputs
- `generated_configs/`: generated YAML configs
- `logs/`: train/validate/test logs
- `runs/`: checkpoints and metrics
- `results.tsv`: experiment ledger, created on first run
- `experiment_summary.tsv`: compact per-experiment model, architecture, augmentation, sampling, and hyperparameter summary
- `.mini_loop/state.json`: local loop state, created on first run
- `.mini_loop/codex_home/`: repo-local Codex login state
- `.mini_loop/codex_session.json`: current Codex loop session state
- `downloads/`: optional downloaded papers or weights
- `.mini_loop/autofix_configs/`: wrapper-generated retry configs when model-name fallback is applied

## Rules
- baseline first
- compare only within the same runtime tier
- runtime tiers are comparison buckets only; the wrapper no longer injects legacy time-based epoch or batch caps
- each baseline or candidate run should state its own expected runtime and budget reasoning
- `medium` is the main exploration/evaluation bucket, while `long` is for final tie-down of the strongest candidates rather than a preset training length
- `keep` only when the validation metric stack `roc_auc_presence > average_precision_presence > best_f1_presence > dice_pos` improves
- near-best runs within the configured noise band are retained as `candidate` instead of being discarded outright, so faster or otherwise attractive alternatives stay visible for finalist selection
- recoverable crashes should trigger repair work when the direction still looks promising; they are not the same as measured negative results
- matched scores stay `discard`, but should remain visible and be noted explicitly as ties in the ledger
- when a tier plateaus for several cycles without a new keep, the loop should broaden rather than keep making same-family config-only tweaks
- run locked test only on selected kept or baseline experiments
- create `.mini_loop/STOP_CODEX_LOOP` if you want the loop to stop after the current cycle

See [program.md](/C:/Users/Max/code/mini_llm_cnn/program.md), [search_space_open.md](/C:/Users/Max/code/mini_llm_cnn/search_space_open.md), and [search_space_limited.md](/C:/Users/Max/code/mini_llm_cnn/search_space_limited.md) for the operating contract.
