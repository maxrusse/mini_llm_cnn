# mini_llm_cnn

This repo is an autonomous config-only research loop for `xray_fracture_benchmark`.

## Setup
1. Read `README.md` and `config.json`.
2. Confirm the benchmark repo path and benchmark Python path in `config.json` are correct.
3. Start by checking current state:
   - `& ..\xray_fracture_benchmark_venv\Scripts\python.exe .\run_loop.py status`
4. If no baseline exists for the target tier, run baseline first.

## What you can change
- You may create or edit configs under `generated_configs/`.
- You may call `run_loop.py baseline`, `step`, `run-config`, `status`, `list-mutations`, and `test`.
- You may choose among existing benchmark-supported model families and supported config knobs.
- When used interactively, you may use web search for benchmark-relevant ideas and references.
- When used interactively, you may install narrowly scoped packages into `xray_fracture_benchmark_venv` and download pretrained weights or papers if they are directly tied to a concrete experiment path.

## What you cannot change
- Do not modify code in `../xray_fracture_benchmark/src` or `../xray_fracture_benchmark/scripts`.
- Do not modify dataset files, manifests, or split definitions.
- Do not tune on `test`.
- Do not bypass `config.json` by hardcoding different interpreter or repo paths in commands.
- Do not download alternate datasets, labels, or anything that changes the benchmark data contract.

## Goal
Get the highest validation `dice_pos` within a fixed runtime tier while keeping changes simple and reviewable.

## Operating rules
- Baseline first for each tier.
- Compare only against the current best run in the same tier.
- Prefer config changes that are understandable and tied to evidence from prior results.
- Keep candidate configs small and specific. Avoid giant bundles unless simpler changes are exhausted.
- Use `test` only for explicit finalist evaluation after validation success.

## Loop
1. Check status.
2. If baseline is missing for the tier, run it.
3. Otherwise run `step --tier <tier>` or create a config and run it with `run-config`.
4. Review the result in `results.tsv` and the run logs.
5. Continue until manually stopped.
