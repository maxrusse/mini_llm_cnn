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
- treats wrapper policy rejections as non-terminal feedback, so the session can reprompt and continue instead of stopping early
- selects candidates by the validation metric stack `roc_auc_presence > average_precision_presence > best_f1_presence > dice_pos`
- uses `runtime_tier` as a comparison bucket only; actual training budget is set per experiment in the config itself
- expects open search to actually use web search for outside ideas
- explicitly allows importing strong methods from other domains when the transfer case to fracture segmentation is plausible
- lets open search use a small coherent bundle of changes when that is faster than one-scalar-at-a-time search
- supports unattended overnight search
- keeps Torch, Hugging Face, pip, and temp caches under `.mini_loop/` by default so model downloads and temp files stay inside the repo workspace
- auto-recovers once from a remote Codex thread context overflow by clearing the saved thread id, starting a fresh thread, and continuing from local repo state

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

## Required Training Venv
`mini_llm_cnn` does not own or rebuild its training stack locally. It runs against the benchmark interpreter at `..\xray_fracture_benchmark_venv\Scripts\python.exe`, as configured in `config.json`.

The rebuild source of truth lives in `..\xray_fracture_benchmark`:
- `requirements-cu128.txt`
- `requirements.txt`
- `scripts\setup_env.ps1`

For other users, the expected training environment is documented in `venv_req.txt` in this repo. The preferred rebuild path is still to run the benchmark repo's setup script:

```powershell
cd ..\xray_fracture_benchmark
.\scripts\setup_env.ps1
```

## Search Modes
- `open`: broader, data-driven search inside the benchmark contract
- `limited`: narrower, more conservative search
- `code`: implementation-focused flow that reads top kept runs plus review-worthy alternates, then prefers benchmark `src/` code experiments over more config-only search
- `aggressive`: implementation-heavy flow with its own ledger that pushes iterative benchmark `src/` build-out for a stronger next-generation approach family

## Runtime Tiers
- `smoke`: debug-only comparison bucket
- `medium`: main exploration and evaluation bucket
- `long`: final tie-down bucket for strongest candidates

These are bookkeeping buckets for fair comparison, not hidden epoch or batch presets. The actual training budget must be stated in each experiment config and justified by the model.

Launchers:

```powershell
& .\scripts\start_codex_loop.ps1 -Tier medium -Hours 8 -SearchSpace open -StartInNewWindow
```

```powershell
& .\scripts\start_codex_code_fresh.ps1 -Tier medium -Hours 8
```

```powershell
& .\scripts\start_codex_aggressive_fresh.ps1 -Tier medium -Hours 8
```

If `code-flow` says Codex is not logged in for `.mini_loop\codex_home_code`, run:

```powershell
& .\scripts\login_codex.ps1 -SearchSpace code
```

```powershell
& .\scripts\start_codex_loop.ps1 -Tier medium -Hours 8 -SearchSpace code -SeedParentExperimentId e0018
```

```powershell
& .\scripts\start_codex_loop.ps1 -Tier medium -Hours 8 -SearchSpace code -StartFromScratch
```

If `aggressive-flow` says Codex is not logged in for `.mini_loop\codex_home_aggressive`, run:

```powershell
& .\scripts\login_codex.ps1 -SearchSpace aggressive
```

```powershell
& .\scripts\start_codex_loop.ps1 -Tier medium -Hours 8 -SearchSpace aggressive -SeedParentExperimentId e0018
```

```powershell
& .\scripts\start_codex_loop.ps1 -Tier medium -Hours 8 -SearchSpace aggressive -StartFromScratch
```

```bash
bash ./scripts/start_codex_loop.sh --tier medium --hours 8 --search-space limited
```

## Main Files
- `run_loop.py`: main loop entrypoint
- `scripts/codex_loop.py`: action-wrapper that resumes one Codex thread and executes chosen actions
- `config.json`: interpreter paths, benchmark paths, comparison-bucket labels
- `config_code.json`: separate executor paths and ledgers for code-flow
  - also enables first-run `run-config` bootstrap for an empty code-flow ledger
- `config_aggressive.json`: separate executor paths and ledgers for aggressive code-flow
- `config/codex_loop.json`: Codex model and loop settings
  - includes wrapper auto-repair maps for missing modules and unsupported model aliases
- `config/codex_loop_code.json`: code-flow controller settings
- `config/codex_loop_aggressive.json`: aggressive code-flow controller settings
- `program.md`: agent operating instructions
- `search_space_open.md`: open search policy
- `search_space_limited.md`: limited search policy
- `search_space_code.md`: code-flow policy
- `search_space_aggressive.md`: aggressive code-flow policy

## General Build Shape
This repo is intentionally file-based and small, but the core platform pieces are already separated:

- `search policy`: `program.md` plus the search-space documents decide what Codex should propose next
- `execution wrapper`: `scripts/codex_loop.py` handles thread lifecycle, prompt assembly, retries, and action dispatch
- `experiment runner`: `run_loop.py` executes baseline, run-config, install, download, status, and test actions
- `artifact store`: configs, logs, checkpoints, metrics, and ledgers live in predictable repo-local folders
- `session state`: `.mini_loop/` holds Codex login state, thread ids, session files, stop files, and cache roots
- `policy layer`: wrapper checks enforce runtime-bucket comparison, plateau broadening, and explicit finalist-only test use

The key design choice is that remote LLM state is disposable, but local experiment state is not. The ledger, artifacts, and local session files are the durable source of truth.

## Upcoming Modularization
If this gets rebuilt into a real platform, the clean split is:

- `planner service`: assembles context and produces the next structured action
- `execution service`: runs approved actions, benchmark commands, and bounded repair steps
- `experiment registry`: owns experiment rows, parent-child links, keep/discard decisions, and summaries
- `artifact service`: owns configs, metrics, logs, checkpoints, archive exports, and reproducibility bundles
- `session service`: owns Codex login state, thread resumption, compaction failures, and fresh-thread rotation
- `policy engine`: owns allowed actions, plateau rules, evaluation restrictions, and finalist gates

The new overflow retry behavior matters here: session restart is an infrastructure concern, not a search-policy concern. That should stay modular when this gets platformized.

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
- `generated_configs_code/`, `logs_code/`, `runs_code/`, `downloads_code/`: code-flow artifacts
- `results_code.tsv`: code-flow experiment ledger
- `experiment_summary_code.tsv`: code-flow summary ledger
- `generated_configs_aggressive/`, `logs_aggressive/`, `runs_aggressive/`, `downloads_aggressive/`: aggressive-flow artifacts
- `results_aggressive.tsv`: aggressive-flow experiment ledger
- `experiment_summary_aggressive.tsv`: aggressive-flow summary ledger
- `.mini_loop/autofix_configs/`: wrapper-generated retry configs when model-name fallback is applied
- `logs*/codex_thread_reset.log`: thread-reset telemetry when the wrapper rotates to a fresh Codex thread after a context overflow

## Rules
- baseline first
- compare only within the same runtime tier
- runtime tiers are comparison buckets only; the wrapper no longer injects legacy time-based epoch or batch caps
- each baseline or promising alternate run should state its own expected runtime and budget reasoning
- `medium` is the main exploration/evaluation bucket, while `long` is for final tie-down of the strongest candidates rather than a preset training length
- `keep` only when the validation metric stack `roc_auc_presence > average_precision_presence > best_f1_presence > dice_pos` improves
- near-best or tradeoff runs are annotated as review-worthy alternates in the ledger, so the LLM can decide whether they deserve follow-up despite losing on the primary stack
- recoverable crashes should trigger repair work when the direction still looks promising; they are not the same as measured negative results
- matched scores stay `discard`, but should remain visible and be noted explicitly as ties in the ledger
- when a tier plateaus for several cycles without a new keep, the loop should broaden rather than keep making same-family config-only tweaks
- same-family broad jumps remain allowed after plateau when they change multiple meaningful axes such as pretraining, resolution, training budget, sampling, or loss structure
- if plateau persists without any benchmark `src/` experiments, the loop should escalate toward `code_edits` instead of ending the session
- policy-rejected proposals should be logged and fed back into the next prompt, not treated as terminal blockers
- a remote Codex context overflow is an infra event; the wrapper may clear the saved thread id and retry once on a fresh thread while preserving local experiment state
- run locked test only on selected kept or baseline experiments
- create `.mini_loop/STOP_CODEX_LOOP` if you want the loop to stop after the current cycle

See [program.md](/C:/Users/Max/code/mini_llm_cnn/program.md), [search_space_open.md](/C:/Users/Max/code/mini_llm_cnn/search_space_open.md), and [search_space_limited.md](/C:/Users/Max/code/mini_llm_cnn/search_space_limited.md) for the operating contract.
