# Changelog

## [Unreleased] — 2026-06-09

### Fixed

#### `Evo_1/scripts/train.py`

- **`init_swanlab()` did not respect `--disable_wandb`**
  - The function always called `swanlab.init()` regardless of the flag, attempting to authenticate
    with the SwanLab cloud and crashing with `KeyFileError: api key not configured` when no API
    key was present.
  - Added an early-return guard: `if config.disable_wandb: return`.
  - Also added `mode='local'` to `swanlab.init()` as a safe offline fallback for when SwanLab *is*
    enabled but no cloud key is configured.

- **`swanlab.log()` called unconditionally in training loop**
  - `swanlab.log(wandb_log_dict)` was called every log step regardless of whether SwanLab was
    initialised (i.e. when `--disable_wandb` caused the early return above, SwanLab was never
    `init()`-ed, so the subsequent `.log()` call would raise).
  - Wrapped with `if not config.disable_wandb:` guard.
