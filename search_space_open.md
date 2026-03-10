# SEARCH_SPACE_OPEN

Open search-space policy for `mini_llm_cnn` night runs.

## Goal
- Primary goal: improve validation `dice_pos`.
- Compare only within the same runtime tier.
- `test` stays locked until there is an explicit finalist selection.

## Fixed Contract
- Training, validation, and test semantics come from `xray_fracture_benchmark`.
- No edits in `../xray_fracture_benchmark/src` or `../xray_fracture_benchmark/scripts`.
- No new datasets, no new labels, and no split changes.
- Generated experiments stay in `generated_configs/`, `runs/`, and `logs/`.

## Open Search Space
- The search space is intentionally not limited to a rigid whitelist.
- Any idea is allowed if it stays within the existing benchmark contract and can be expressed as a small, testable config change.
- Typical directions include model choice, optimization, input size, aspect preservation, sampling, patch strategies, loss design, presence auxiliary signals, and evaluation thresholds.
- Existing benchmark-compatible model families and config patterns are preferred starting points, but they are not a hard ceiling.

## Data-Driven Rule
- Every new direction should have a data-driven reason: baseline outcome, repeated failure pattern, observed weakness, or external evidence from papers or web search.
- Weak directions should be discarded instead of being pursued on intuition alone.
- If one parent is clearly stronger, search locally around that parent; if local search goes flat, broaden again.
- Added complexity must be justified by evidence. More heads, more sampling logic, or more structure only when results support it.

## Allowed External Help
- Web search for benchmark-relevant architecture, sampling, loss, optimization, or training ideas.
- Download of papers or pretrained weights.
- Narrow package installs into `xray_fracture_benchmark_venv` when they are directly needed for a concrete experiment path.

## Not Allowed
- Downloading new datasets.
- Using the test set for optimization.
- Changing benchmark metrics or split files.
- Expanding the repo arbitrarily without a clear link to the existing benchmark.

## Overnight Strategy
- Fresh start per tier: baseline first, then broad `medium` search.
- Use `long` only for candidates that are clearly better than baseline in `medium`.
- Express new ideas as small, reviewable config changes; avoid uncontrolled bundled jumps at the start.
- If a strong parent already exists, continue searching from that parent; otherwise start from baseline.
