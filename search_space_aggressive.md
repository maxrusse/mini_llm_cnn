# Aggressive Code Search-Space Policy

Aggressive code search is the next-step implementation track for building a stronger fracture-presence system iteratively.

## Purpose
- Build a next-generation approach family, not just small local improvements.
- Use benchmark-side `src/` changes to expand model capability in bounded, testable steps.
- Keep each run tied to one coherent hypothesis while allowing multi-file implementation when the hypothesis needs it.

## Core Behavior
- `aggressive-flow` keeps its own results, logs, configs, and session state.
- It may read `open-flow` and `code-flow` outputs as source material, but it must not overwrite them.
- Default seed is the strongest source run in the chosen tier unless launched with scratch mode or an explicit parent experiment id.
- On an empty aggressive ledger, the first seeded `run_config` may bootstrap the local tier instead of requiring a prior local baseline.
- Every non-bootstrap `run_config` should include benchmark `src/` code edits.
- Prefer iterative build-out of one promising family across several cycles instead of unrelated one-off experiments.

## Aggressive Expectations
- Prefer broader benchmark `src/` work over loss-only edits.
- Typical acceptable edit scopes include:
  - new model heads or auxiliary branches
  - multi-scale evidence aggregation
  - explicit presence-classification branches tied to segmentation features
  - feature fusion or decoder/context modules
  - ranking or calibration components paired with architectural support
  - bounded training/pipeline components under `src/`
- Pure loss-only changes are insufficient unless they are clearly part of a larger ongoing implementation line.
- If several cycles still stay narrow, escalate again rather than orbiting around the same small tweak class.

## Suggested Next-Gen Direction
- Favor an iterative presence-evidence architecture:
  - aggregate top-k or learned evidence from multi-scale segmentation features
  - expose an explicit presence head instead of relying only on pooled mask logits
  - keep segmentation supervision, but let presence ranking/classification use dedicated learned evidence
- Build this in bounded steps:
  - step 1: explicit presence head fed by pooled decoder features
  - step 2: learned multi-scale evidence fusion or gated pooling
  - step 3: stronger ranking/calibration objective on the explicit presence logits

## Allowed Edits
- Edits are allowed only under `../xray_fracture_benchmark/src` through `run_config.code_edits`.
- Edits may span multiple `src/` files when needed for one coherent hypothesis.
- The best kept aggressive code state becomes the parent code state for later aggressive runs.

## Source Idea Pool
- Use up to about 10 source experiments from the same runtime tier.
- Source pool should include baseline, kept runs, and review-worthy alternates.
- Exclude crashes from the idea pool.
- Prefer mixing strong source ideas into an implementation step rather than replaying them unchanged.

## Boundaries
- Do not edit `../xray_fracture_benchmark/scripts`.
- Do not change datasets, manifests, labels, or train/val/test split semantics.
- Do not tune on test.
- Keep each experiment bounded enough that its outcome is still interpretable.
