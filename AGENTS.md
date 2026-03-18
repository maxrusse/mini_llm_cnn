# AGENTS.md - mini_llm_cnn

## Mission and Scope
- Build a minimal autonomous research loop for `xray_fracture_benchmark`.
- Optimize the validation metric stack `roc_auc_presence > average_precision_presence > best_f1_presence > dice_pos` through config-first experimentation, with optional open-mode method/code experiments when config-only search looks too narrow.
- Keep the mini loop lightweight, reproducible, and easy for a single Codex agent to operate.

## Hard Constraints and Safety Boundaries
- Do not edit code inside `../xray_fracture_benchmark/scripts`.
- In `limited` mode, stay config-only.
- In `open` mode, benchmark code edits under `../xray_fracture_benchmark/src` are allowed only through the wrapper-owned `run_config.code_edits` path.
- Open-mode code edits may range from small targeted fixes to larger benchmark-side implementations such as new helper functions, new heads, new model classes, or end-to-end training/pipeline components when they are the most feasible way to test a concrete method idea.
- `code-flow` is a separate implementation-focused mode that reads the strongest kept runs plus review-worthy alternates as an idea pool and then prefers benchmark-side `src/` implementations over more config-only search.
- Code edits must stay data-driven, tied to one concrete hypothesis, and paired with a real config run.
- The wrapper currently supports hill-climbing code edits on top of the current kept best for a tier; do not assume arbitrary branching between old code states.
- Do not tune on `test`; locked test evaluation is explicit and finalist-only.
- Treat the benchmark repo as the source of truth for data splits, metric definitions, and train/validate/test semantics.
- Use explicit interpreter and repo paths from `config.json`; do not rely on shell activation state.
- Keep experiment outputs local to this repo.
- The search logic belongs to the Codex agent, while the local wrapper executes the chosen action.
- If interactive agent work is needed overnight, web research is allowed for benchmark-relevant ideas, but downloaded datasets, labels, or alternate test data are not.
- Targeted package installs into `xray_fracture_benchmark_venv` are allowed only when they are necessary for a concrete supported experiment direction and are logged in the session summary.
- Downloading pretrained weights or papers is allowed; downloading new training datasets is out of scope.

## Data and Evaluation Policy
- Primary selection stack is validation `roc_auc_presence > average_precision_presence > best_f1_presence > dice_pos`.
- Compare only runs produced with the same runtime tier.
- `runtime_tier` is a comparison bucket only. Actual training budget must be declared in the config for each run.
- Use `medium` as the main exploration/evaluation bucket and `long` for final tie-down of the strongest candidates, not as hardcoded training-time presets.
- Test evaluation is allowed only for a kept or baseline experiment and must use the run's resolved config.
- Do not modify dataset manifests, split files, or threshold-selection policy in the benchmark repo.

## Code-Level Quality Bar
- Keep the loop config-driven and deterministic where possible.
- Log every attempted experiment to `results.tsv`.
- Each run must preserve its generated config, logs, checkpoint path, and validation metrics path.
- For each baseline or promising alternate run, the agent should state the expected runtime and why that budget matches the method change.
- Near-best or tradeoff runs should be recorded as review-worthy alternates for the LLM to judge, rather than auto-promoted by a fixed wrapper rule.
- If several cycles plateau without a new keep, the search should broaden to another family, a coherent same-family broad jump, or benchmark `src/` code edits instead of continuing same-family micro-tuning.
- If a kept direction looks strong but obviously benchmark-limited, prefer a bounded benchmark `src/` implementation on top of that promising direction instead of only more config retuning.
- Wrapper policy rejections are non-terminal loop feedback and should trigger a broader follow-up proposal, not session termination.
- If the loop behavior or CLI changes, update `README.md` and `program.md`.
- Keep Codex loop state minimal: one repo-local `CODEX_HOME`, one thread id file, one session state file, and log files.
- Before starting a fresh round, archive old run artifacts outside the repo so a new session starts clean without local-result leakage.

## Operational Commands
- Login Codex for this repo: `& .\scripts\login_codex.ps1` or `& .\scripts\login_codex.ps1 -SearchSpace code`
- Start single-agent Codex loop: `& .\scripts\start_codex_loop.ps1 -Tier medium -Hours 8 -SearchSpace open -StartInNewWindow`
- Start coding-focused flow: `& .\scripts\start_codex_code_fresh.ps1 -Tier medium -Hours 8`
- Stop after the current cycle: `New-Item -ItemType File -Path .\.mini_loop\STOP_CODEX_LOOP -Force | Out-Null`
- Baseline: `& ..\xray_fracture_benchmark_venv\Scripts\python.exe .\run_loop.py baseline --tier smoke`
- Run explicit config: `& ..\xray_fracture_benchmark_venv\Scripts\python.exe .\run_loop.py run-config --config .\generated_configs\custom.yaml --tier medium --label custom_try`
- Status: `& ..\xray_fracture_benchmark_venv\Scripts\python.exe .\run_loop.py status`
- Locked test on finalist: `& ..\xray_fracture_benchmark_venv\Scripts\python.exe .\run_loop.py test --experiment-id e0007`
