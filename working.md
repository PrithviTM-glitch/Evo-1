# Working Scripts & Configurations

## Stage 1 — MetaWorld Fine-tuning (FlowMatching Action Head)

### Overview

Fine-tunes the Evo-1 action head on the MetaWorld Sawyer dataset using a FlowMatching policy.
The VLM backbone (InternVL3-1B) is kept **frozen**; only the action head is trained.
W&B runs in **offline** mode — sync manually after training completes.

### Environment

| Item | Detail |
|------|--------|
| GPU | NVIDIA A100-SXM4-80GB |
| CUDA | 13.0 (driver 580.159.03) |
| torch | 2.5.1+cu124 |
| flash-attn | 2.8.3 |
| deepspeed | 0.18.9 |
| accelerate | latest |
| Working directory | `/home/tmprithvi/Evo-1/Evo_1` |

### Dataset

- **Path:** `/home/tmprithvi/Evo1_training_dataset/Metaworld`
- **Config:** `dataset/metaworld_config.yaml`
- **Cache:** `/home/tmprithvi/Evo1_training_dataset/cache/metaworld`
- **Samples:** 190,989

### Launch Command

Run from `/home/tmprithvi/Evo-1/Evo_1` (recommended: inside a `tmux` session):

```bash
accelerate launch \
  --num_processes 1 \
  --num_machines 1 \
  --deepspeed_config_file ds_config.json \
  scripts/train.py \
  --wandb_project evo1_metaworld \
  --run_name Evo1_flash_metaworld_stage1 \
  --action_head flowmatching \
  --use_augmentation \
  --lr 1e-5 \
  --dropout 0.2 \
  --weight_decay 1e-3 \
  --batch_size 16 \
  --image_size 448 \
  --max_steps 20000 \
  --log_interval 10 \
  --ckpt_interval 5000 \
  --warmup_steps 1000 \
  --grad_clip_norm 1.0 \
  --num_layers 8 \
  --horizon 50 \
  --finetune_action_head \
  --disable_wandb \
  --prefetch_factor 2 \
  --video_backend av \
  --cache_dir /home/tmprithvi/Evo1_training_dataset/cache/metaworld \
  --vlm_name OpenGVLab/InternVL3-1B \
  --dataset_config_path dataset/metaworld_config.yaml \
  --per_action_dim 24 \
  --state_dim 24 \
  --save_dir /home/tmprithvi/baseline
```

### Key Flags Explained

| Flag | Value | Why |
|------|-------|-----|
| `--action_head flowmatching` | flowmatching | Flow-matching policy head (only supported head type) |
| `--finetune_action_head` | flag | Freeze VLM; train action head only |
| `--disable_wandb` | flag | W&B logs offline to `save_dir/wandb/`; no cloud key needed |
| `--vlm_name` | OpenGVLab/InternVL3-1B | VLM backbone — must match pre-cached model in HuggingFace cache |
| `--cache_dir` | `.../cache/metaworld` | Pre-processed dataset cache (must exist and be populated) |
| `--image_size 448` | 448 | Input resolution for InternVL3 |
| `--per_action_dim 24` | 24 | Must match `max_action_dim` in `metaworld_config.yaml` |
| `--state_dim 24` | 24 | Must match `max_state_dim` in `metaworld_config.yaml` |
| `--horizon 50` | 50 | Action prediction horizon |
| `--num_layers 8` | 8 | Number of transformer layers in the action head |

### Observed Performance (A100-SXM4-80GB)

| Metric | Value |
|--------|-------|
| Throughput | ~1.3 it/s (~21 samples/s) |
| GPU memory (peak) | ~6 GB allocated / 7.34 GB reserved |
| Loss at step 700 | ~0.378 |
| Loss at step 1300 | ~0.319 (decreasing normally) |
| Best checkpoint loss | 0.3048 (step 1273) |
| Estimated total time | ~4.3 hours for 20,000 steps |

### Output Structure

```
/home/tmprithvi/baseline/
├── train_log_<timestamp>.log     # Full training log
├── step_best/                    # Best checkpoint (lowest loss)
├── step_5000/                    # Periodic checkpoint
├── step_10000/                   # Periodic checkpoint
├── step_15000/                   # Periodic checkpoint
├── step_20000/                   # Final checkpoint
└── wandb/
    └── offline-run-<id>/         # W&B offline run data
```

### Syncing W&B After Training

```bash
wandb sync /home/tmprithvi/baseline/wandb/offline-run-*
```

### Known Issues / Gotchas

1. **`--cache_dir` must point to the pre-populated cache** — pointing to a non-existent path
   causes the script to re-export all 2,500 parquet windows from scratch (~2 min overhead).
   The correct path is `/home/tmprithvi/Evo1_training_dataset/cache/metaworld`.

2. **SwanLab requires `--disable_wandb` guard** — a bug fix was applied to `scripts/train.py`
   so that `init_swanlab()` and `swanlab.log()` are skipped when `--disable_wandb` is passed.
   Without this fix the script crashes immediately with `KeyFileError`.

3. **Accelerate warnings** — `--mixed_precision` and `--dynamo_backend` default to `no`.
   These are just warnings; run `accelerate config` once to suppress them permanently.

---

## Stage 2 — MetaWorld Fine-tuning (VLM + Action Head)

### Overview

Continues from a Stage 1 checkpoint, now unfreezing the full VLM backbone (InternVL3-1B) in
addition to the action head. Uses `--resume_pretrain` to load Stage 1 weights as an
initialization point while resetting the step counter and optimizer to zero (fresh warmup).

### Environment

Same as Stage 1 (see above).

### Launch Command

Run from `/home/tmprithvi/Evo-1/Evo_1` (recommended: inside a `tmux` session):

```bash
accelerate launch \
  --num_processes 1 \
  --num_machines 1 \
  --deepspeed_config_file ds_config.json \
  scripts/train.py \
  --wandb_project evo1_metaworld \
  --run_name Evo1_metaworld_stage2 \
  --disable_gradient_checkpointing \
  --action_head flowmatching \
  --use_augmentation \
  --lr 1e-5 \
  --dropout 0.2 \
  --weight_decay 1e-3 \
  --batch_size 16 \
  --image_size 448 \
  --max_steps 80000 \
  --log_interval 10 \
  --ckpt_interval 2500 \
  --warmup_steps 1000 \
  --grad_clip_norm 1.0 \
  --num_layers 8 \
  --horizon 50 \
  --finetune_vlm \
  --finetune_action_head \
  --disable_wandb \
  --prefetch_factor 2 \
  --video_backend av \
  --cache_dir /home/tmprithvi/Evo1_training_dataset/cache/metaworld \
  --vlm_name OpenGVLab/InternVL3-1B \
  --dataset_config_path dataset/metaworld_config.yaml \
  --per_action_dim 24 \
  --state_dim 24 \
  --save_dir /home/tmprithvi/baseline/stage2 \
  --resume \
  --resume_pretrain \
  --resume_path /home/tmprithvi/baseline/step_10000
```

### Resuming an Interrupted Stage 2 Run

If training is interrupted mid-run (crash, preemption, etc.), do **NOT** use `--resume_pretrain`
again — that would reset the step counter to 0 and discard all progress. Instead use:

```bash
accelerate launch \
  --num_processes 1 \
  --num_machines 1 \
  --deepspeed_config_file ds_config.json \
  scripts/train.py \
  --wandb_project evo1_metaworld \
  --run_name Evo1_metaworld_stage2 \
  --disable_gradient_checkpointing \
  --action_head flowmatching \
  --use_augmentation \
  --lr 1e-5 \
  --dropout 0.2 \
  --weight_decay 1e-3 \
  --batch_size 16 \
  --image_size 448 \
  --max_steps 80000 \
  --log_interval 10 \
  --ckpt_interval 2500 \
  --warmup_steps 1000 \
  --grad_clip_norm 1.0 \
  --num_layers 8 \
  --horizon 50 \
  --finetune_vlm \
  --finetune_action_head \
  --disable_wandb \
  --prefetch_factor 2 \
  --video_backend av \
  --cache_dir /home/tmprithvi/Evo1_training_dataset/cache/metaworld \
  --vlm_name OpenGVLab/InternVL3-1B \
  --dataset_config_path dataset/metaworld_config.yaml \
  --per_action_dim 24 \
  --state_dim 24 \
  --save_dir /home/tmprithvi/baseline/stage2 \
  --resume \
  --resume_path /home/tmprithvi/baseline/stage2/step_10000
```

Replace `step_XXXXX` with the latest checkpoint in `/home/tmprithvi/stage2/` — checkpoints are
saved every 2500 steps. To find the latest:

```bash
ls -d /home/tmprithvi/stage2/step_*/ | sort -t_ -k2 -n | tail -1
```

Key differences from the initial launch:
- `--resume_pretrain` is **removed** — preserves step counter and optimizer/scheduler state
- `--resume_path` points to a **Stage 2** checkpoint, not the Stage 1 baseline

### Key Flags Explained

| Flag | Value | Why |
|------|-------|-----|
| `--finetune_vlm` | flag | Unfreeze VLM backbone; all parameters trainable |
| `--finetune_action_head` | flag | Also train action head (both components unfrozen) |
| `--resume_pretrain` | flag | Load weights from Stage 1 checkpoint, reset step to 0 |
| `--resume` | flag | Must be set together with `--resume_path` |
| `--resume_path` | `.../baseline/step_10000` | Stage 1 checkpoint used as weight initialization |
| `--save_dir` | `/home/tmprithvi/stage2` | Separate output dir from Stage 1 |

### Observed Performance (A100-SXM4-80GB)

| Metric | Value |
|--------|-------|
| Throughput | ~0.70 it/s (~11 samples/s) |
| GPU memory (peak) | ~12.19 GiB allocated / 14.14 GiB reserved |
| Loss at step 10 | ~0.035 (low — initialized from trained Stage 1 weights) |
| Estimated total time | ~32 hours for 80,000 steps |

### Output Structure

```
/home/tmprithvi/stage2/
├── train_log_<timestamp>.log     # Full training log
├── best_checkpoint.pt            # Tracker for best checkpoint path
├── step_2500/                    # Periodic checkpoint
├── step_5000/                    # Periodic checkpoint
├── ...
├── step_80000/                   # Final checkpoint (or step_final/)
└── wandb/
    └── offline-run-<id>/         # W&B offline run data
```

### Syncing W&B After Training

```bash
wandb sync /home/tmprithvi/stage2/wandb/offline-run-*
```

### Known Issues / Gotchas

1. **`--resume_pretrain` resets step to 0** — this is intentional for starting Stage 2 fresh
   from Stage 1 weights, but will destroy in-progress Stage 2 training if misused on a resume.
   Always check which command variant you're using before launching.

2. **Higher GPU memory than Stage 1** — ~14 GiB reserved vs ~7.3 GiB in Stage 1 because VLM
   gradients are now active. Still well within A100-SXM4-80GB limits.

3. **Lower throughput than Stage 1** — ~0.70 it/s vs ~1.3 it/s because backprop now flows
   through the full VLM backbone.

---

## Parallel MetaWorld Evaluation

### Overview

Evaluates multiple checkpoints (baseline stage 1 + all stage 2) against MT50 MetaWorld tasks
in parallel. Each evaluation spins up its own server instance on a separate port, so N checkpoints
run simultaneously with the rest queued.

Uses three new files (originals untouched):
- `Evo_1/scripts/Evo1_server_eval.py` — server with CLI args + correct checkpoint loading
- `MetaWorld_evaluation/mt50_evo1_client_eval.py` — client with CLI args for port + output dir
- `MetaWorld_evaluation/eval_queue.py` — queue orchestrator

### Launch Command

Run from `/home/tmprithvi/Evo-1` (recommended: inside a `tmux` session):

```bash
python MetaWorld_evaluation/eval_queue.py --parallel 3
```

### Common Variants

```bash
# Run 4 evals in parallel
python MetaWorld_evaluation/eval_queue.py --parallel 4

# Custom port range (e.g. if 9000-9002 are in use)
python MetaWorld_evaluation/eval_queue.py --parallel 3 --base_port 9100

# Custom output directory
python MetaWorld_evaluation/eval_queue.py --parallel 3 --out_root /tmp/eval_results
```

### What Gets Evaluated

| Name | Checkpoint Path |
|------|----------------|
| `baseline_step_5000` | `/home/tmprithvi/baseline/step_5000` |
| `baseline_step_10000` | `/home/tmprithvi/baseline/step_10000` |
| `stage2_step_2500` … `stage2_step_best` | `/home/tmprithvi/baseline/stage2/step_*` |

### Output Structure

```
MetaWorld_evaluation/eval_outputs/
├── baseline_step_5000/
│   ├── server.log          # Server stdout/stderr
│   ├── client.log          # Client stdout/stderr
│   ├── logs/               # mt50_YYYYMMDD_HHMMSS.txt (per-task success rates)
│   └── episode_videos/     # task##_<slug>_ep###.mp4
├── baseline_step_10000/
│   └── ...
└── stage2_step_*/
    └── ...
```

### Running a Single Evaluation Manually

```bash
# Terminal 1 — start server
cd /home/tmprithvi/Evo-1/Evo_1/scripts
python Evo1_server_eval.py \
  --ckpt_dir /home/tmprithvi/baseline/step_5000 \
  --port 9000 \
  --arm_key metaworld_sawyer \
  --dataset_key Evo1_MetaWorld

# Terminal 2 — start client (once server prints "running at ws://0.0.0.0:9000")
cd /home/tmprithvi/Evo-1/MetaWorld_evaluation
python mt50_evo1_client_eval.py \
  --port 9000 \
  --out_dir /tmp/eval_baseline_step5000
```

### Key Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--parallel` | `3` | Number of simultaneous evaluations |
| `--base_port` | `9000` | First port; slot `i` uses `base_port + i` |
| `--out_root` | `MetaWorld_evaluation/eval_outputs` | Root dir for all per-checkpoint outputs |
| `--arm_key` | `metaworld_sawyer` | Arm key in `norm_stats.json` |
| `--dataset_key` | `Evo1_MetaWorld` | Dataset key in `norm_stats.json` |
