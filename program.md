# mini_llm_cnn

This repo is an autonomous config-only research loop for `xray_fracture_benchmark`.

## Setup
1. Read `README.md` and `config.json`.
2. Confirm the benchmark repo path and benchmark Python path in `config.json` are correct.
3. The Codex wrapper will provide current status, recent results, and best-config context every cycle.
4. If no baseline exists for the target tier, choose a baseline action first.

## What you can change
- You may propose one wrapper action per cycle.
- You may propose configs under `generated_configs/`.
- You may choose among existing benchmark-supported model families and supported config knobs.
- You may use web search for benchmark-relevant ideas and references.
- In `open` mode, you should actually use web search during the run instead of staying fully local.
- You may request narrowly scoped package installs into `xray_fracture_benchmark_venv` and downloads of pretrained weights or papers if they are directly tied to a concrete experiment path.

## What you cannot change
- Do not modify code in `../xray_fracture_benchmark/src` or `../xray_fracture_benchmark/scripts`.
- Do not modify dataset files, manifests, or split definitions.
- Do not tune on `test`.
- Do not bypass `config.json` by hardcoding different interpreter or repo paths in commands.
- Do not download alternate datasets, labels, or anything that changes the benchmark data contract.
- Do not try to execute shell commands directly; the wrapper executes approved actions for you.

## Goal
Get the highest validation `dice_pos` within a fixed runtime tier while keeping changes simple and reviewable.

## Operating rules
- Baseline first for each runtime tier you want to use.
- Compare only against the current best run in the same tier.
- In `open` mode, use the real search tiers flexibly:
  - `medium_5m`, `medium`, `medium_20m`, `medium_30m` for active exploration
  - `long` only for clear finalist candidates
- In `open` mode, architecture and model-family exploration is a first-class search axis.
- Use `experiment_summary.tsv` to audit what has actually been explored. If recent runs cluster inside one model family or one narrow hyperparameter basin, broaden again.
- When local tuning goes flat, broaden with benchmark-supported architecture moves such as:
  - model family or backbone changes
  - width or capacity changes
  - auxiliary or dual-head variants
  - input-size or aspect-handling changes
  - scheduler or sampling changes
- Do not spend a whole night only retuning BCE/Dice weights or learning rate around one architecture unless the results clearly justify that focus.
- In `open` mode, do not underuse the available GPU budget. Consider stronger architectures, pretrained backbones, larger inputs, or larger batches when they fit the benchmark contract and the current tier.
- Prefer config changes that are understandable and tied to evidence from prior results.
- Keep candidate configs small and specific. Avoid giant bundles unless simpler changes are exhausted.
- Use `test` only for explicit finalist evaluation after validation success.

## Loop
1. Inspect the wrapper-provided status, result tail, best-config context, and `experiment_summary.tsv`.
2. Return exactly one JSON action.
3. If baseline is missing for the runtime tier you want to use, choose `baseline` for that tier.
4. Otherwise choose a small `run_config` candidate and set its `runtime_tier`, or request `install_package` / `download_file` only when directly justified.
5. Review the next cycle's result summary and continue until manually stopped.
