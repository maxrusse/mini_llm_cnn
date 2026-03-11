# SEARCH_SPACE_OPEN

Open search-space policy for `mini_llm_cnn` night runs.

## Goal
- Primary goal: improve validation `dice_pos`.
- Compare only within the same runtime tier.
- `test` stays locked until there is an explicit finalist selection.
- Open search should use real GPU budgets, not sub-minute proxy runs.

## Fixed Contract
- Training, validation, and test semantics come from `xray_fracture_benchmark`.
- No edits in `../xray_fracture_benchmark/src` or `../xray_fracture_benchmark/scripts`.
- No new datasets, no new labels, and no split changes.
- Generated experiments stay in `generated_configs/`, `runs/`, and `logs/`.

## Open Search Space
- The search space is intentionally not limited to a rigid whitelist.
- Any idea is allowed if it stays within the existing benchmark contract and can be expressed as a small, testable config change.
- Typical directions include model choice, backbone choice, architecture variants, optimization, input size, aspect preservation, sampling, patch strategies, loss design, presence auxiliary signals, and evaluation thresholds.
- Existing benchmark-compatible model families and config patterns are preferred starting points, but they are not a hard ceiling.
- Architecture exploration is explicitly in scope. Open search should not collapse into pure loss and learning-rate tuning around one model family.
- Use `experiment_summary.tsv` as an exploration audit. If the recent table shows only one architecture family, widen the search.
- Strong open-search candidates include benchmark-supported:
  - model-family switches
  - pretrained backbone changes
  - dual-head or auxiliary-head variants
  - width or capacity changes
  - input-size and aspect-handling changes
  - sampling or scheduler pivots

## Data-Driven Rule
- Every new direction should have a data-driven reason: baseline outcome, repeated failure pattern, observed weakness, or external evidence from papers or web search.
- Weak directions should be discarded instead of being pursued on intuition alone.
- If one parent is clearly stronger, search locally around that parent; if local search goes flat, broaden again.
- Added complexity must be justified by evidence. More heads, more sampling logic, or more structure only when results support it.
- Open search should mix breadth and depth: first establish a reasonable spread of architectures and training regimes, then exploit the strongest basin.
- Before moving to `long`, there should usually be evidence that more than one architecture or model direction was considered, unless the benchmark support makes that impossible.

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
- Fresh start per tier: baseline first, then broad search across the 5m to 30m tiers.
- Use `medium_5m`, `medium`, `medium_20m`, and `medium_30m` flexibly based on signal quality and experiment cost.
- Use `long` only for candidates that are clearly better than baseline in the shorter search tiers and are worth up to about 2 hours of GPU time.
- Express new ideas as small, reviewable config changes; avoid uncontrolled bundled jumps at the start.
- If a strong parent already exists, continue searching from that parent; otherwise start from baseline.
- Do not underuse the GPU. In open mode, it is acceptable to spend real budget on stronger architectures, pretrained backbones, larger inputs, and other heavier but benchmark-compatible variants when they have a concrete rationale.
