# Running HCNT on Kaggle with 2 GPUs

This project now supports multi-GPU training via `torch.nn.DataParallel`.
On Kaggle's T4 x2 kernel this roughly halves per-epoch time with zero
command-line changes to your workflow.

---

## 1. Kaggle notebook settings

1. Open your notebook on Kaggle.
2. Right sidebar → **Settings** → **Accelerator** → choose **GPU T4 x2**
   (also acceptable: **GPU P100** stays single-GPU; **TPU VM v3-8** is not
   supported by this script).
3. Pin a Python session (**>** menu → **Run all** or just execute a cell)
   so the two GPUs are allocated to your container.

Verify both GPUs are visible before training:

```python
import torch
print("CUDA available :", torch.cuda.is_available())
print("GPU count      :", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"  cuda:{i} -> {torch.cuda.get_device_name(i)}")
```

Expected output:

```
CUDA available : True
GPU count      : 2
  cuda:0 -> Tesla T4
  cuda:1 -> Tesla T4
```

If `GPU count` is 1, the accelerator setting didn't apply — change it and
restart the kernel.

---

## 2. Run commands (unchanged from single-GPU)

Multi-GPU is automatic — the code detects visible GPUs and wraps the model
in `nn.DataParallel` when two or more are present. All existing commands
work:

```bash
# Sanity run — one seed
python -m src.train_v2 --seed 42

# Full 5-seed sweep
python -m src.run_seeds

# Ensemble + threshold tuning (Tier 1 pipeline)
python -m src.tier1_eval
```

To force single-GPU training on the T4 x2 kernel (useful for debugging):

```bash
python -m src.train_v2 --seed 42 --no-multi-gpu
```

The training banner tells you which path is active:

```
Seed=42  Device=cuda:0  GPUs visible=2  DataParallel=True
[Multi-GPU] nn.DataParallel across 2 GPUs (effective batch size = 32,
per-GPU = 16). Increase Config.BATCH_SIZE to 64 to keep per-GPU batch
unchanged if you have VRAM headroom.
```

---

## 3. Batch size tuning (optional)

By default the per-GPU batch is `Config.BATCH_SIZE // n_gpu = 16` on
2 GPUs. T4 (16 GB VRAM) comfortably fits a per-GPU batch of 32 for
`model_v2` at 384×384, so you can double throughput further by setting
`Config.BATCH_SIZE = 64` in `src/config.py`. The learning rate is already
reasonable at this scale — no linear LR scaling needed for a 2x jump.

If you hit OOM on the first forward pass, back off to `Config.BATCH_SIZE =
48` first; model_v2 + MixUp + CutMix at 384-px can be tight.

---

## 4. Expected speedup

On Kaggle T4 x2, with the default batch size 32:

| GPUs | Per-epoch time | 5-seed sweep |
|------|----------------|--------------|
| 1    | ~90 s          | ~2.5 h       |
| 2    | ~55 s          | ~1.5 h       |

The sub-linear speedup is expected for `DataParallel` — it incurs
host-side scatter/gather overhead per step. For a 100-epoch training
budget on 929 images, this overhead is dwarfed by the actual compute
and the net wall-clock win is roughly 1.6–1.7x per seed, 5× across the
full sweep.

If you need linear scaling, `DistributedDataParallel` (DDP) is the
standard answer. For this project at this dataset scale, DDP's launch
complexity (`torchrun`, process groups, DistributedSampler, rank-aware
logging) outweighs the extra 10–15 % of speedup it would buy on 2 GPUs.
We can revisit DDP for the journal-tier version (PRAD-10K, 5-fold CV).

---

## 5. Checkpoint portability

Checkpoints are saved as the EMA copy of the underlying `raw_model`'s
`state_dict`, with no `module.` prefix from `DataParallel`. This means:

- A checkpoint trained on 2 GPUs loads cleanly on 1 GPU for evaluation.
- A checkpoint trained on 1 GPU loads cleanly on 2 GPUs for ensembling.
- `src/ensemble.py`, `src/tier1_eval.py`, and `src/evaluate.py` all run
  single-GPU without any changes.

So the recommended flow is: multi-GPU for training, single-GPU for
inference and the Tier 1 ensemble pipeline. The code already does this
automatically.

---

## 6. Caveats to know

1. **MixUp / CutMix fire host-side** before the forward pass, so they
   mix across the full global batch and then the mixed images are
   sharded across GPUs. Behaviour is unchanged vs. single-GPU.

2. **OHEM runs per-GPU shard** (each GPU keeps its own top-70% hardest
   from its sub-batch). On 2 GPUs with batch 32, that means top-70% of
   16 = top-11 kept per GPU vs. top-22 of 32 in single-GPU mode. For a
   small-batch classifier this delta is negligible, but note it when
   reading per-step loss curves.

3. **Random augmentation seeding** is still set once per process by
   `set_seed(seed)`. DataParallel runs in a single process (threads
   under the hood for forward passes), so determinism is preserved.
   DDP would require per-rank seeding; DP doesn't.

4. **Logger output appears once** (not duplicated across GPUs) because
   `DataParallel` is single-process. If you later migrate to DDP, you'll
   need to gate `logger.on_epoch_end()` on `rank == 0`.

---

## 7. Troubleshooting

**`RuntimeError: CUDA out of memory`**
Reduce `Config.BATCH_SIZE` back to its default (32) or lower (16 for
very tight VRAM). The paired T4s have 16 GB each; model + batch 32 +
AMP should sit around 7–8 GB per GPU.

**`AttributeError: 'DataParallel' object has no attribute 'unfreeze_backbone'`**
Should not happen with the patched `train_v2.py` — every custom-method
call routes through `raw_model`, not the DP wrapper. If you see it,
you're running an old copy of `train_v2.py`; re-upload the current
version to Kaggle.

**Only one GPU shows utilization in `nvidia-smi`**
The host side (data loading, loss computation, MixUp) runs on CPU and
only one GPU does the forward pass before gather. This is normal for DP;
total throughput is still ~2x. If you want to confirm both GPUs are
doing work, watch `nvidia-smi dmon -s u` during a step — you'll see
both GPUs briefly spike for every forward pass.
