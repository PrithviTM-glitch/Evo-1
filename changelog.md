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

## [Unreleased] — 2026-06-10

### Added

#### `Evo_1/scripts/Evo1_server_eval.py` *(new file — duplicate of `Evo1_server.py`)*

- Accepts CLI arguments via `argparse` so multiple server instances can run simultaneously:
  - `--ckpt_dir` (required) — path to checkpoint directory
  - `--port` (default: `9000`) — WebSocket port to listen on
  - `--arm_key` (default: `metaworld_sawyer`) — arm key in `norm_stats.json`
  - `--dataset_key` (default: `Evo1_MetaWorld`) — dataset key in `norm_stats.json`
- Fixed checkpoint loading to match the format produced by the training script:
  - Original targeted DeepSpeed format: `mp_rank_00_model_states.pt` / `checkpoint["module"]`
  - Fixed to standard torch format: `checkpoint.pt` / `checkpoint["model_state_dict"]`
  - Root cause: training used `accelerate launch --deepspeed_config_file` without `--use_deepspeed`,
    so `accelerator.distributed_type != DEEPSPEED` and `train.py` took the standard save path.

#### `MetaWorld_evaluation/mt50_evo1_client_eval.py` *(new file — duplicate of `mt50_evo1_client_prompt.py`)*

- Accepts CLI arguments via `argparse` for isolated parallel runs:
  - `--port` (default: `9000`) — WebSocket port to connect to
  - `--out_dir` (default: `.`) — root output directory; overrides `LOG_DIR`, `LOG_PATH`,
    `VIDEO_SAVE_DIR`, and `INSPECT_DIR` so concurrent runs never write to the same paths

#### `MetaWorld_evaluation/eval_queue.py` *(new file)*

- Orchestrator script that evaluates multiple checkpoints in parallel with a configurable
  concurrency limit.
- Discovers checkpoints automatically:
  - `baseline/step_5000` and `baseline/step_10000`
  - All subdirectories under `baseline/stage2/`
- Maintains a port pool (`base_port + slot_id`) so each concurrent server gets its own port.
- Uses `ThreadPoolExecutor` + `queue.Queue` port pool: remaining jobs queue up and grab a slot
  as soon as a running evaluation finishes.
- CLI arguments:
  - `--parallel` (default: `3`) — number of simultaneous evaluations
  - `--base_port` (default: `9000`) — first port in the pool
  - `--out_root` (default: `MetaWorld_evaluation/eval_outputs`) — root dir for all outputs
  - `--arm_key` / `--dataset_key` — forwarded to each server instance
- Each checkpoint writes to its own subdirectory under `--out_root` (`baseline_step_5000/`,
  `stage2_step_10000/`, etc.) containing `server.log`, `client.log`, `logs/`, `episode_videos/`.

## [Unreleased] — 2026-06-10 (bugfixes)

### Fixed

#### `MetaWorld_evaluation/mt50_evo1_client_eval.py`

- **`SHOW_WINDOW = True` caused SIGABRT on headless servers**
  - `render_single_bgr()` called `cv2.imshow()` every frame, which caused OpenCV's Qt backend
    to attempt loading the `xcb` platform plugin. On a headless server with no `$DISPLAY`,
    this triggered an OS-level `abort()` (signal 6 / SIGABRT), killing the client process
    immediately with `exit=-6`.
  - The `except Exception: pass` guard around `cv2.imshow()` provided **no protection** because
    SIGABRT is raised at the C library level — it never surfaces as a Python exception.
  - **Fix:** Changed `SHOW_WINDOW = True` → `SHOW_WINDOW = False` (line 32) with comment
    `# headless server — cv2.imshow aborts without a display`.
  - Note: `MUJOCO_GL=egl` (already set via `os.environ.setdefault`) provides correct offscreen
    rendering on this machine — no Xvfb needed.

#### `MetaWorld_evaluation/eval_queue.py`

- **Checkpoint discovery included non-checkpoint directories (e.g. `wandb/`)**
  - `get_checkpoints()` iterated all subdirectories under `baseline/stage2/` using
    `not d.name.startswith(".")` as the only filter. The `wandb/` directory passes that test,
    gets scheduled as a fake checkpoint, and wastes a server slot (server startup times out
    after 180 s or fails to load the directory as a model).
  - **Fix:** Changed filter from `not d.name.startswith(".")` to `d.name.startswith("step")`,
    so only valid checkpoint directories (`step_2500`, `step_5000`, …, `step_best`) are picked up.
