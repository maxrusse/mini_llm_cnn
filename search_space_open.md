# SEARCH_SPACE_OPEN

Open search-space policy for `mini_llm_cnn` night runs.

## Goal
- Primary goal: improve the validation metric stack `roc_auc_presence > average_precision_presence > best_f1_presence > dice_pos`.
- Compare only within the same runtime tier.
- `runtime_tier` is a comparison bucket only. The config itself sets the real training budget.
- Use `medium` as the main exploration/evaluation bucket and `long` to tie down the strongest candidates at the end of exploration.
- `test` stays locked until there is an explicit finalist selection.
- Open search should use real GPU budgets, not sub-minute proxy runs.
- The quality bar is 2026-and-beyond SOTA-level thinking within the benchmark contract, not just conservative tuning of the current repo defaults.
- Transdomain knowledge transfer is in scope. Strong ideas from other domains should be explored when there is a plausible mapping to this benchmark's structure or failure modes.

## Fixed Contract
- Training, validation, and test semantics come from `xray_fracture_benchmark`.
- No edits in `../xray_fracture_benchmark/scripts`.
- Edits in `../xray_fracture_benchmark/src` are allowed only in `open` mode through `run_config.code_edits`, and they must stay tied to one concrete experiment.
- No new datasets, no new labels, and no split changes.
- Generated experiments stay in `generated_configs/`, `runs/`, and `logs/`.

## Open Search Space
- The search space is intentionally not limited to a rigid whitelist.
- Any idea is allowed if it stays within the existing benchmark contract and can be expressed as a testable experiment, whether that is config-only or requires bounded benchmark-side code.
- Typical directions include model choice, backbone choice, architecture variants, optimization, input size, aspect preservation, sampling, patch strategies, loss design, presence auxiliary signals, and evaluation thresholds.
- Relevant outside sources include general semantic segmentation, dense prediction, object detection, remote sensing, document AI, industrial anomaly inspection, weak supervision, self-supervision, multimodal vision, and foundation-model adaptation.
- Existing benchmark-compatible model families and config patterns are preferred starting points, but they are not a hard ceiling.
- The benchmark builder now accepts generic family names with `model.backbone`, including `deeplabv3`, `deeplabv3plus`, `unet`, `unetplusplus`, `fpn`, `pspnet`, `linknet`, `manet`, and `pan`, in addition to legacy concrete names like `deeplabv3_resnet50`.
- Architecture exploration is explicitly in scope. Open search should not collapse into pure loss and learning-rate tuning around one model family.
- Use `experiment_summary.tsv` as an exploration audit. If the recent table shows only one architecture family, widen the search.
- Strong open-search candidates include benchmark-supported:
  - model-family switches
  - pretrained backbone changes
  - dual-head or auxiliary-head variants
  - width or capacity changes
  - input-size and aspect-handling changes
  - sampling or scheduler pivots
- Open mode may also test original method ideas through `src/` code edits when config-only search is too limiting.
- Open mode may also strengthen a promising kept direction through `src/` code edits when the current benchmark surface is the limiting factor.
- Valid code-edit directions include loss refinements, sampling logic improvements, architectural tweaks, better heads, new helper modules, new model classes, or full benchmark-side training/inference pipeline changes that still stay within the benchmark contract.
- Longer bounded programming work is allowed when needed to make a strong method runnable. Open search does not need to restrict itself to only tiny edits if a larger implementation is the cleanest way to test the idea.
- Cross-domain adaptation is explicitly valid: if a method from another domain plausibly addresses sparse targets, fine boundaries, class imbalance, calibration, or limited-data transfer, it is worth testing here.
- Open mode may use a small coherent bundle of changes when that is the fastest way to test a real hypothesis. It is not restricted to exactly one knob at a time.
- Good bundles are centered on one idea:
  - architecture family + backbone + matching batch-size change
  - patch enable + patch size + hard-negative settings
  - dual-head model + presence loss + presence evaluation settings
- Bad bundles are unfocused mixes across unrelated axes.

## Data-Driven Rule
- Every new direction should have a data-driven reason: baseline outcome, repeated failure pattern, observed weakness, or external evidence from papers or web search.
- External evidence may come from outside radiography if the transfer argument is technically coherent.
- Weak directions should be discarded instead of being pursued on intuition alone.
- Recoverable crashes are not weak metric evidence. If a direction fails because of a missing dependency, unsupported config surface, or bounded benchmark implementation gap, prefer repairing the path before abandoning the idea.
- If one parent is clearly stronger, search locally around that parent; if local search goes flat, broaden again.
- If a tier has plateaued for several cycles without a new keep, do not keep proposing same-family micro-tweaks. Broaden to a different family, a coherent same-family broad jump, or benchmark `src/` code edits.
- A same-family broad jump should usually change at least two meaningful axes such as backbone/pretraining, resolution/aspect handling, training budget, sampling regime, or loss structure.
- Wrapper policy rejections are feedback, not terminal blockers. Use them to redirect the next proposal.
- Added complexity must be justified by evidence. More heads, more sampling logic, or more structure only when results support it.
- Open search should mix breadth and depth: first establish a reasonable spread of architectures and training regimes, then exploit the strongest basin.
- Before moving to `long`, there should usually be evidence that more than one architecture or model direction was considered, unless the benchmark support makes that impossible.
- If a method idea needs code, keep the code change narrow enough that it can still be attributed and reverted if the run is worse.

## Allowed External Help
- Web search for benchmark-relevant architecture, sampling, loss, optimization, or training ideas.
- In open mode, web search is expected, not optional. The agent should actively bring in outside ideas instead of staying fully local for the whole run.
- Download of papers or pretrained weights.
- Narrow package installs into `xray_fracture_benchmark_venv` when they are directly needed for a concrete experiment path.

## Not Allowed
- Downloading new datasets.
- Using the test set for optimization.
- Changing benchmark metrics or split files.
- Expanding the repo arbitrarily without a clear link to the existing benchmark.

## Overnight Strategy
- Fresh start per tier: baseline first, then broad search inside the chosen comparison bucket.
- The agent must set the real budget directly in config and state why it fits the change. Higher resolution, larger backbones, no-pretrain runs, or larger code changes should usually get longer budgets.
- Use `long` to tie down the strongest candidates at the end of exploration.
- Express new ideas as small, reviewable config changes; avoid uncontrolled bundled jumps at the start.
- In open mode, if a single-change search is clearly too slow, use a compact hypothesis-driven bundle instead of inching forward one scalar at a time.
- If a strong parent already exists, continue searching from that parent; otherwise start from baseline.
- Do not underuse the GPU. In open mode, it is acceptable to spend real budget on stronger architectures, pretrained backbones, larger inputs, and other heavier but benchmark-compatible variants when they have a concrete rationale.
