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
- You may choose among existing benchmark-supported model families and supported config knobs, or extend the benchmark in `open` mode when the current support surface is too narrow for a strong experiment.
- You may use web search for benchmark-relevant ideas and references.
- You may use web search to pull in fitting ideas from other domains when the transfer case is plausible, including methods from general segmentation, detection, dense prediction, remote sensing, document vision, industrial inspection, or foundation-model literature.
- In `open` mode, you should actually use web search during the run instead of staying fully local.
- In `open` mode, you may include benchmark `src/` code edits through `run_config.code_edits` when that is the best data-driven way to test a method idea.
- Those code edits may be larger than a tiny tweak when justified: new helper functions, new heads, new model classes, or full benchmark-side pipeline components are allowed if they stay within the benchmark contract and are paired with a concrete run.
- Longer programming work is allowed when it is the most feasible route to a strong experiment. The loop does not need to restrict itself to only tiny edits if a larger bounded implementation is what the hypothesis requires.
- You may request narrowly scoped package installs into `xray_fracture_benchmark_venv` and downloads of pretrained weights or papers if they are directly tied to a concrete experiment path.
- The wrapper may auto-repair one failed `run_config` cycle by installing mapped packages or retrying once with a configured model-name fallback.
- Recoverable crashes are valid work items. If a promising direction fails because of a missing package, unsupported config surface, or benchmark-side implementation gap, prefer repairing it over treating the crash as evidence that the idea is bad.
- The wrapper runs benchmark commands with repo-local cache and temp directories under `.mini_loop/` so pretrained weights and transient files stay inside the workspace when possible.

## What you cannot change
- Do not modify code in `../xray_fracture_benchmark/scripts`.
- Do not modify dataset files, manifests, or split definitions.
- Do not tune on `test`.
- Do not bypass `config.json` by hardcoding different interpreter or repo paths in commands.
- Do not download alternate datasets, labels, or anything that changes the benchmark data contract.
- Do not try to execute shell commands directly; the wrapper executes approved actions for you.

## Goal
Get the best validation metric stack within a chosen comparison bucket while keeping changes simple and reviewable.
Primary ordering is `roc_auc_presence > average_precision_presence > best_f1_presence > dice_pos`.
Target 2026-and-beyond SOTA-level ideas when feasible within the benchmark contract; do not cap the search at the repo's current default methods.
Integrate transdomain knowledge when useful: if another domain has a method that plausibly transfers to sparse fracture segmentation, it is in scope to test or adapt it.

## Operating rules
- Baseline first for each runtime tier you want to use.
- Compare only against the current best run in the same tier.
- `runtime_tier` is a comparison bucket, not an automatic runtime cap.
- Use `medium` as the main exploration/evaluation bucket and `long` for final tie-down of the strongest candidates, not as a built-in time preset.
- The config itself must set training budget explicitly: epochs, `max_train_batches`, `max_eval_batches`, scheduler, and other controls belong in the YAML.
- For every `baseline` or `run_config`, state the expected runtime and why that budget fits the method change.
- In `open` mode, if a change plausibly needs longer training because of no pretraining, higher resolution, larger backbones, or bigger code changes, allocate that budget explicitly instead of pretending everything fits a fixed preset.
- In `open` mode, architecture and model-family exploration is a first-class search axis.
- In `open` mode, original method ideas are allowed. Code or math changes inside benchmark `src/` are valid when they are evidence-driven and paired with a concrete run.
- Cross-domain transfer is encouraged. If another domain has a method with a credible mapping to this benchmark's failure modes, bring it in and adapt it rather than staying confined to x-ray-specific precedent.
- Treat crashes separately from metric evidence. A crash is a repair target when feasible, not a reason by itself to abandon an architecture or method direction.
- Use `experiment_summary.tsv` to audit what has actually been explored. If recent runs cluster inside one model family or one narrow hyperparameter basin, broaden again.
- If a tier has plateaued for several cycles without a keep, do not keep proposing same-family micro-tweaks. Broaden with a different family, a same-family broad jump, or benchmark `src/` code edits.
- A same-family broad jump should usually change at least two meaningful axes such as backbone/pretraining, resolution/aspect handling, training budget, sampling regime, or loss structure.
- Treat wrapper policy rejections as feedback, not session-ending blockers.
- If plateau persists with zero `code_edits` attempts, the next rounds should bias toward a benchmark `src/` implementation experiment instead of another local config-only nudge.
- When local tuning goes flat, broaden with benchmark-supported architecture moves such as:
  - model family or backbone changes
  - width or capacity changes
  - auxiliary or dual-head variants
  - input-size or aspect-handling changes
  - scheduler or sampling changes
- Do not spend a whole night only retuning BCE/Dice weights or learning rate around one architecture unless the results clearly justify that focus.
- In `open` mode, do not underuse the available GPU budget. Consider stronger architectures, pretrained backbones, larger inputs, or larger batches when they fit the benchmark contract and the current tier.
- Prefer config changes that are understandable and tied to evidence from prior results.
- If a run lands close enough to the best that it may be within noise, keep it as a `candidate` instead of discarding it, especially when it is faster, simpler, or otherwise attractive for final comparison.
- In `limited` mode, keep candidate configs small and specific.
- In `open` mode, one controlled change is preferred for clean attribution, but you may bundle 2-4 tightly related changes when that is the faster data-driven path.
- Open-mode bundles must be coherent, not random:
  - example: pretrained backbone + matching model-family switch + batch-size adjustment
  - example: patch-sampling enable + patch size + hard-negative settings
  - example: dual-head architecture + presence loss settings + presence threshold mode
- Avoid giant unfocused bundles. If you bundle changes, keep the rationale explicit and the bundle centered on one hypothesis.
- If you use `code_edits`, keep them bounded to one method hypothesis and attach them to the current kept best for that tier instead of assuming arbitrary branching between old code states.
- Use `test` only for explicit finalist evaluation after validation success.

## Loop
1. Inspect the wrapper-provided status, result tail, best-config context, and `experiment_summary.tsv`.
2. Return exactly one JSON action.
3. If baseline is missing for the runtime tier you want to use, choose `baseline` for that tier.
4. Otherwise choose a `run_config` candidate, set its comparison bucket in `runtime_tier`, set the real training budget directly in `config_yaml`, and explain the budget choice, or request `install_package` / `download_file` only when directly justified.
5. Review the next cycle's result summary and continue until manually stopped.
6. If a run crashes because of a missing module, unsupported model/config surface, or another bounded implementation gap, prefer a repair path such as `install_package`, `download_file`, or `run_config.code_edits` before abandoning the direction.
7. If the wrapper rejects a proposal as a plateau micro-tweak, use that rejection to broaden the next proposal rather than stopping the session.
