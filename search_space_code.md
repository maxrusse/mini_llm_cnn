# Code Search-Space Policy

Code search is the implementation-focused companion to open search.

## Purpose
- Start from the strongest current idea pool and improve it by coding.
- Combine, intensify, or re-express promising ideas through benchmark-side `src/` changes.
- Hill-climb from the best current code state instead of reopening a broad architecture sweep.

## Core Behavior
- `code-flow` keeps its own results, logs, configs, and session state.
- It may read `open-flow` results and summaries as source material, but it must not overwrite them.
- Default seed is the current best kept run in the chosen tier unless launched with scratch mode or an explicit parent experiment id.
- On an empty code-flow ledger, the first seeded `run_config` may bootstrap the local tier instead of requiring a prior local baseline.
- After the initial seed/control step, most `run_config` actions should include `code_edits`.

## Allowed Edits
- Edits are allowed only under `../xray_fracture_benchmark/src` through `run_config.code_edits`.
- Edits may include new helper functions, new heads, decoder/context changes, sampling logic, feature fusion helpers, auxiliary branches, or other bounded pipeline components tied to one hypothesis.
- The best kept code state in code-flow becomes the parent code state for later code-flow runs.

## Source Idea Pool
- Use up to about 10 source experiments from the same runtime tier.
- Source pool should include baseline, kept runs, and review-worthy alternates.
- Exclude crashes from the idea pool.
- Prefer mixing ideas from source runs into one code-level hypothesis rather than re-running them unchanged.

## Boundaries
- Do not edit `../xray_fracture_benchmark/scripts`.
- Do not change datasets, manifests, labels, or train/val/test split semantics.
- Do not tune on test.
- Keep each code experiment bounded to one coherent hypothesis, even if it combines multiple source ideas.
