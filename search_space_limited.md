# SEARCH_SPACE_LIMITED

Limited search-space policy for `mini_llm_cnn` night runs.

## Goal
- Primary goal: improve the validation metric stack `roc_auc_presence > average_precision_presence > best_f1_presence > dice_pos`.
- Compare only within the same runtime tier.
- `runtime_tier` is a comparison bucket only. The config itself sets the real training budget.
- Use `medium` as the main exploration/evaluation bucket and `long` to tie down the strongest candidates at the end of exploration.
- Wrapper policy rejections are non-terminal feedback; if a proposal is rejected as a plateau micro-tweak, the next step should broaden rather than stop the run.
- `test` stays locked until there is an explicit finalist selection.
- Limited search should still use real GPU budgets, not ultra-short proxy runs.

## Fixed Contract
- Training, validation, and test semantics come from `xray_fracture_benchmark`.
- No edits in `../xray_fracture_benchmark/src` or `../xray_fracture_benchmark/scripts`.
- No new datasets, no new labels, and no split changes.
- Generated experiments stay in `generated_configs/`, `runs/`, and `logs/`.

## Limited Search Space
### Model families
- `simple_unet`
- `simple_unet_dual_head`
- `deeplabv3` with `backbone: resnet50|resnet101`
- `deeplabv3_resnet50`
- `deeplabv3_resnet101`
- `deeplabv3_resnet50_dual_head`
- `deeplabv3_resnet101_dual_head`

### Optimization
- `learning_rate`
- `weight_decay`
- `batch_size`
- `scheduler`: `none`, `cosine`, `onecycle`
- `min_lr`, `max_lr`, `pct_start`

### Input / Preprocessing
- `input.image_size`
- `input.preserve_aspect`

### Sampling
- `training.balanced_sampling`
- `training.patch.enabled`
- `training.patch.size`
- `training.patch.positive_prob`
- `training.patch.hard_negative_prob`
- `training.patch.hard_negative_quantile`

### Loss
- `bce_dice`
- `focal_dice_tversky`
- `bce_weight`, `dice_weight`
- `presence_bce_weight`
- `presence_bce_warmup_epochs`

### Evaluation
- `evaluation.threshold`
- `evaluation.presence_score_mode`
- `evaluation.presence_threshold`

## Allowed External Help
- Web search for benchmark-relevant architecture, sampling, loss, or training ideas.
- Download of papers or pretrained weights.
- Narrow package installs into `xray_fracture_benchmark_venv` when they are directly needed for a concrete experiment path.

## Not Allowed
- Downloading new datasets.
- Using the test set for optimization.
- Changing benchmark metrics or split files.
- Arbitrary repo expansion without a clear link to the existing benchmark.

## Overnight Strategy
- Fresh start per tier: baseline first, then conservative search inside the chosen comparison bucket.
- The agent must set the real budget directly in config and state why it fits the change.
- Use `long` to tie down the strongest candidates at the end of exploration.
- Express new ideas as small, reviewable config changes; avoid uncontrolled bundled jumps at the start.
- If a strong parent already exists, continue searching from that parent; otherwise start from baseline.
