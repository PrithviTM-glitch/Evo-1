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
