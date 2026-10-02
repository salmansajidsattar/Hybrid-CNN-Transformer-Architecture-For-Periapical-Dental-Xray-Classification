"""
train_v2.py — Training script for HybridCNNTransformerV2
=========================================================
Improvements over train.py:
  • Focal Loss (gamma=2)  — focuses training on hard/borderline samples
  • OHEM — Online Hard Example Mining (top-k loss reweighting per batch)
  • Differential LRs: backbone 10× lower than Transformer
  • Warmup-cosine LR schedule
  • Backbone unfreeze at epoch Config.UNFREEZE_EPOCH
  • TTA at evaluation / test time
  • Full classification report + confusion matrix on test set
  • EMA (exponential moving average) of weights for stable evaluation
  • Structured logging via src.logger.TrainingLogger
  • Val-AUC based checkpoint selection (instead of val_acc)

Usage:
    python -m src.train_v2              # train from scratch
    python -m src.train_v2 --eval-only  # evaluate best checkpoint
    python -m src.train_v2 --seed 42    # set specific random seed
"""

import sys
import math
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, f1_score)
import matplotlib
matplotlib.use('Agg')  # headless-safe: no GUI window on a training server/Kaggle
import matplotlib.pyplot as plt
import seaborn as sns

from .config import Config
from .dataset import create_dataloaders
from .model_v2 import (create_model_v2, build_optimizer_v2,
                        build_scheduler_v2, predict_with_tta)
from .utils import plot_training_history, save_metrics, AverageMeter
from .logger import TrainingLogger


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Loss functions ────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss — down-weights easy examples so the model focuses on
    hard/borderline cases (the 16 misclassified samples in the test set).

    gamma=2.0 means a correctly classified sample at p=0.9 contributes
    only 1% of the loss weight compared to a sample at p=0.5.

    Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25,
                 label_smoothing: float = 0.05):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ls    = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_cls = logits.size(1)
        # Label-smoothed one-hot
        with torch.no_grad():
            one_hot = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1)
            one_hot = one_hot * (1 - self.ls) + self.ls / n_cls

        log_p = F.log_softmax(logits, dim=1)
        p     = log_p.exp()

        # Focal weight
        focal_w = self.alpha * (1 - p) ** self.gamma

        loss = -(focal_w * one_hot * log_p).sum(dim=1).mean()
        return loss


class OHEMLoss(nn.Module):
    """Online Hard Example Mining: keep top-k hardest per batch."""
    def __init__(self, base_loss, keep_ratio: float = 0.7):
        super().__init__()
        self.base   = base_loss
        self.keep   = keep_ratio

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B = logits.size(0)
        k = max(1, int(B * self.keep))

        # Per-sample loss (without reduction)
        per_sample = F.cross_entropy(logits, targets, reduction='none')

        # Keep top-k hardest
        _, idx = per_sample.topk(k, largest=True)
        hard_logits  = logits[idx]
        hard_targets = targets[idx]

        return self.base(hard_logits, hard_targets)


# ── MixUp ─────────────────────────────────────────────────────────────────────

def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def mixup_criterion(criterion, pred, ya, yb, lam):
    return lam * criterion(pred, ya) + (1 - lam) * criterion(pred, yb)


def cutmix_data(x, y, alpha=1.0):
    # CutMix (Yun et al., 2019): paste a random box from one image in the
    # batch onto another; lam is corrected to the actual pasted-area ratio
    # afterwards (loss mixing reuses mixup_criterion with this lam).
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    H, W = x.shape[-2:]
    cut_rat = (1.0 - lam) ** 0.5
    cut_h, cut_w = int(H * cut_rat), int(W * cut_rat)
    cy, cx = np.random.randint(H), np.random.randint(W)
    y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, H)
    x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, W)
    x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam_adjusted = 1.0 - ((x2 - x1) * (y2 - y1) / (H * W))
    return x, y, y[idx], lam_adjusted


# ── EMA (Exponential Moving Average of weights) ──────────────────────────────

class ModelEMA:
    """
    Keeps a shadow copy of the model with weights updated as
        ema_w = decay * ema_w + (1 - decay) * model_w

    The EFFECTIVE decay ramps up from ~0 to `decay` over the first few
    hundred updates (d_t = min(decay, (1+t)/(10+t)) -- the standard EMA
    warmup used by torchvision/timm/YOLOv5's reference EMA implementations)
    instead of using the fixed `decay` from update 1.

    Without this ramp, a small dataset with few steps/epoch never
    accumulates enough updates for the EMA to escape its near-random
    initial weights within a normal training budget: e.g. with
    decay=0.9995 (time constant ~2000 steps) and ~21 steps/epoch, a
    100-epoch run is only ~2100 total steps, so ~35% of the EMA's weight
    is STILL the initial (untrained) state at the very end of training.
    Since validate()/full_evaluate() always evaluate `ema.module` (not the
    raw model), and checkpoint selection is based on that same val_auc,
    every reported metric and the saved checkpoint end up reflecting this
    lagging shadow copy rather than the actually-trained network -- val
    accuracy can even sit below chance while the raw model's train
    accuracy is high. The warmup ramp fixes this: after ~1 epoch (21
    updates) effective decay is already ~0.71, and by ~10 epochs (210
    updates) it is ~0.96, so the EMA tracks the real model closely for
    the whole run instead of lagging behind it throughout.

    Evaluate on the EMA copy instead of the raw model — noticeably more
    stable on small datasets, once it has actually converged.
    """
    def __init__(self, model, decay: float = 0.9995):
        self.decay = decay
        self.updates = 0
        self.module = deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])


# ── Train epoch ───────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device, epoch,
                scaler=None, logger: TrainingLogger = None):
    model.train()
    meter = AverageMeter()
    correct = total = 0
    use_mixup   = getattr(Config, 'USE_MIXUP', True)
    use_cutmix  = getattr(Config, 'USE_CUTMIX', False)
    cutmix_prob = getattr(Config, 'CUTMIX_PROB', 0.5)

    for step, (imgs, labels) in enumerate(
            tqdm(loader, desc=f"Train E{epoch:03d}", leave=False)):
        imgs, labels = imgs.to(device), labels.to(device)

        mixed = False
        if (use_mixup or use_cutmix) and random.random() < 0.5:
            # Apply exactly one of MixUp / CutMix to this batch (not both).
            # When both are enabled, CutMix fires with probability
            # Config.CUTMIX_PROB and MixUp otherwise.
            do_cutmix = use_cutmix and (not use_mixup or random.random() < cutmix_prob)
            aug_fn = cutmix_data if do_cutmix else mixup_data
            imgs, ya, yb, lam = aug_fn(imgs, labels,
                                        alpha=getattr(Config, 'MIXUP_ALPHA', 0.2))
            mixed = True

        optimizer.zero_grad()
        with autocast(enabled=(scaler is not None)):
            logits = model(imgs)
            loss   = (mixup_criterion(criterion, logits, ya, yb, lam)
                      if mixed else criterion(logits, labels))

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), Config.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), Config.GRAD_CLIP)
            optimizer.step()

        meter.update(loss.item(), imgs.size(0))
        if not mixed:
            correct += (logits.argmax(1) == labels).sum().item()
            total   += labels.size(0)

        if logger is not None:
            logger.on_train_step(
                epoch=epoch, step=step,
                loss=loss.item(),
                lr=optimizer.param_groups[-1]['lr'],   # transformer LR
                grad_norm=float(grad_norm),
                preds=(None if mixed else logits.argmax(1)),
                labels=(None if mixed else labels),
            )

    acc = correct / total * 100 if total > 0 else 0.0
    return meter.avg, acc


# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, criterion, device, use_tta=False,
             logger: TrainingLogger = None):
    """
    Returns:
        loss  : mean validation loss over the epoch
        acc   : validation accuracy (%)
        auc   : ROC-AUC (0.0 if not computable)
        f1    : macro-F1 (%)
    """
    model.eval()
    meter = AverageMeter()
    all_probs, all_labels_np = [], []

    for imgs, labels in tqdm(loader, desc="  Val", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss   = criterion(logits, labels)
        meter.update(loss.item(), imgs.size(0))

        if use_tta:
            probs_batch = torch.stack([
                predict_with_tta(model, imgs[i:i+1], device,
                                  getattr(Config, 'TTA_N_AUG', 5))
                for i in range(imgs.size(0))
            ])
        else:
            probs_batch = F.softmax(logits, dim=1)

        all_probs.append(probs_batch.cpu())
        all_labels_np.append(labels.cpu())

        if logger is not None:
            logger.on_val_step(loss=loss.item(),
                               probs=probs_batch, labels=labels)

    probs  = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels_np).numpy()
    preds  = probs.argmax(axis=1)
    acc    = (preds == labels).mean() * 100
    try:
        auc = roc_auc_score(labels, probs[:, 1])
    except ValueError:
        auc = float('nan')
    f1 = f1_score(labels, preds, average='macro', zero_division=0) * 100
    # Cast off numpy.float64 (from .mean()/sklearn) -- these get pickled
    # straight into checkpoints, and PyTorch >=2.6's torch.load(weights_only=
    # True default) refuses to unpickle numpy scalar globals back out.
    return meter.avg, float(acc), float(auc), float(f1)


# ── Full evaluation ───────────────────────────────────────────────────────────

@torch.no_grad()
def full_evaluate(model, loader, device, class_names, save_dir: Path,
                  use_tta=False, tag='v2'):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for imgs, labels in tqdm(loader, desc="  Test eval", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        if use_tta:
            for i in range(imgs.size(0)):
                probs = predict_with_tta(model, imgs[i:i+1], device,
                                          getattr(Config, 'TTA_N_AUG', 5))
                all_probs.append(probs.cpu())
                all_preds.append(probs.argmax().item())
        else:
            logits = model(imgs)
            probs  = F.softmax(logits, dim=1)
            all_probs.extend(probs.cpu())
            all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    preds  = np.array(all_preds)
    labels = np.array(all_labels)
    probs  = torch.stack(all_probs).numpy()

    print("\n" + "="*60)
    print(f"CLASSIFICATION REPORT ({tag})")
    print("="*60)
    print(classification_report(labels, preds, target_names=class_names,
                                 digits=4))

    # AUC (only meaningful for binary)
    auc = None
    if probs.shape[1] == 2:
        auc = roc_auc_score(labels, probs[:, 1])
        print(f"AUC-ROC: {auc:.4f}")

    # Confusion matrix
    save_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.title(f'Confusion Matrix ({tag})')
    plt.tight_layout()
    cm_path = save_dir / f'confusion_matrix_{tag}.png'
    plt.savefig(cm_path, dpi=150); plt.close()
    print(f"Confusion matrix -> {cm_path}")

    acc = float((preds == labels).mean() * 100)
    return {'accuracy': acc,
            'auc': float(auc) if auc is not None else None,
            'preds': preds, 'labels': labels, 'probs': probs}


# ── Main training loop ────────────────────────────────────────────────────────

def train_v2(seed: int = 42, save_suffix: str = '', multi_gpu: bool = True):
    """
    multi_gpu : when True (default) and torch.cuda.device_count() >= 2, wrap
                the model in nn.DataParallel to split each mini-batch across
                all visible GPUs. On Kaggle's T4 x2 kernel this roughly halves
                epoch time at zero code-change cost. Set multi_gpu=False to
                force single-GPU training even when 2 GPUs are visible.
    """
    set_seed(seed)
    device = Config.DEVICE
    n_gpu  = torch.cuda.device_count() if device.type == 'cuda' else 0
    use_dp = bool(multi_gpu and n_gpu > 1)
    print(f"\n{'='*60}\nSeed={seed}  Device={device}  "
          f"GPUs visible={n_gpu}  DataParallel={use_dp}\n{'='*60}")

    train_loader, val_loader, test_loader, class_names = create_dataloaders(Config.DATA_DIR)

    # raw_model keeps the HybridCNNTransformerV2 interface intact (custom
    # methods like unfreeze_backbone / get_backbone_params that nn.DataParallel
    # does NOT forward through its wrapper). We only wrap it in DataParallel
    # for the training forward pass; EMA, optimizer, checkpoints and the
    # custom-method calls all keep operating on raw_model.
    raw_model = create_model_v2(Config.NUM_CLASSES).to(device)
    model = nn.DataParallel(raw_model) if use_dp else raw_model
    if use_dp:
        print(f"[Multi-GPU] nn.DataParallel across {n_gpu} GPUs "
              f"(effective batch size = {Config.BATCH_SIZE}, per-GPU = "
              f"{Config.BATCH_SIZE // n_gpu}). Increase Config.BATCH_SIZE "
              f"to {Config.BATCH_SIZE * n_gpu} to keep per-GPU batch "
              f"unchanged if you have VRAM headroom.")

    # ── Loss: Focal + OHEM (or plain cross-entropy for the ablation study) ──
    if getattr(Config, 'USE_FOCAL_OHEM', True):
        focal    = FocalLoss(gamma=getattr(Config, 'FOCAL_GAMMA', 2.0),
                             alpha=getattr(Config, 'FOCAL_ALPHA', 0.25),
                             label_smoothing=getattr(Config, 'LABEL_SMOOTHING', 0.05))
        criterion = OHEMLoss(focal, keep_ratio=getattr(Config, 'OHEM_KEEP_RATIO', 0.7))
        print(f"Loss: FocalLoss(gamma={getattr(Config,'FOCAL_GAMMA',2.0)}) + OHEM")
    else:
        criterion = nn.CrossEntropyLoss(
            label_smoothing=getattr(Config, 'LABEL_SMOOTHING', 0.05))
        print("Loss: plain CrossEntropyLoss (Focal+OHEM ablated)")

    # Optimizer reads raw_model.get_backbone_params() / get_transformer_params(),
    # which DataParallel does NOT expose on its wrapper -- always use raw_model
    # here regardless of use_dp.
    optimizer = build_optimizer_v2(
        raw_model,
        base_lr            = Config.LEARNING_RATE,
        backbone_lr_scale  = getattr(Config, 'BACKBONE_LR_SCALE', 0.1),
        weight_decay       = Config.WEIGHT_DECAY,
    )
    scheduler = build_scheduler_v2(
        optimizer,
        num_epochs    = Config.NUM_EPOCHS,
        warmup_epochs = getattr(Config, 'WARMUP_EPOCHS', 5),
    )
    scaler = GradScaler() if getattr(Config, 'USE_AMP', True) else None

    unfreeze_ep = getattr(Config, 'UNFREEZE_EPOCH', 10)
    use_tta     = getattr(Config, 'USE_TTA', False)
    ckpt_name   = f'best_model_v2{save_suffix}.pth'

    # ── Structured logger + EMA weights ────────────────────────────────────
    logger = TrainingLogger(
        run_name  = f'v2_s{seed}',
        save_dir  = Config.LOG_DIR,
        use_tensorboard = True,
        class_names = class_names,
    )
    # EMA shadows the UNDERLYING model so its state_dict keys match
    # raw_model's (no "module." prefix from DataParallel) -- otherwise every
    # ema.update() would see mismatched keys and silently drift from the
    # trained weights.
    ema = ModelEMA(raw_model, decay=getattr(Config, 'EMA_DECAY', 0.9995))

    best_val_auc = -1.0
    best_val_acc = -1.0
    best_val_f1  = -1.0

    # Real per-epoch history for plot_training_history() / training_history.png
    # (previously imported but never populated or called -- the figure the
    # paper references was never actually regenerated from real training).
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(1, Config.NUM_EPOCHS + 1):
        if epoch == unfreeze_ep:
            # unfreeze_backbone is a custom method on HybridCNNTransformerV2
            # that nn.DataParallel does not expose through its wrapper -- call
            # it on raw_model to work in both single- and multi-GPU paths.
            raw_model.unfreeze_backbone()

        logger.on_epoch_start(epoch)

        # Train — model may be DataParallel-wrapped; logger captures per-step
        # loss/lr/grad_norm from GPU 0 (DP returns gathered logits there).
        tr_loss, tr_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            scaler=scaler, logger=logger,
        )
        # EMA shadows raw_model (no "module." prefix); feed it raw_model so
        # the state_dict keys line up on both single- and multi-GPU paths.
        ema.update(raw_model)

        # Validate the EMA copy (more stable than raw weights)
        va_loss, va_acc, va_auc, va_f1 = validate(
            ema.module, val_loader, criterion, device,
            use_tta=use_tta, logger=logger,
        )

        # --- TEMPORARY DIAGNOSTIC (see docstring at top of this patch) ---
        # Compares the raw (just-trained) model's own val performance against
        # the EMA copy's, to tell an EMA-specific artifact apart from a real
        # collapse in the trained network itself. Remove once resolved.
        # Validate against raw_model (not the DP wrapper) so output shapes
        # don't depend on how many GPUs are visible.
        raw_va_loss, raw_va_acc, raw_va_auc, raw_va_f1 = validate(
            raw_model, val_loader, criterion, device, use_tta=False, logger=None)
        print(f"    [debug-raw] raw_val_acc={raw_va_acc:.2f}%  raw_val_auc={raw_va_auc:.4f}  "
              f"raw_val_f1={raw_va_f1:.2f}%   (EMA: val_acc={va_acc:.2f}% val_auc={va_auc:.4f} val_f1={va_f1:.2f}%)")

        scheduler.step()

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(va_loss)
        history['val_acc'].append(va_acc)

        train_metrics = logger.finish_train_epoch()
        val_metrics   = logger.finish_val_epoch()

        # Report both LRs (backbone and transformer)
        lr_bb = optimizer.param_groups[0]['lr']
        lr_tf = optimizer.param_groups[-1]['lr']
        logger.on_epoch_end(
            train_metrics, val_metrics,
            lrs={'backbone': lr_bb, 'transformer': lr_tf},
        )

        # AUC-based checkpoint selection (was: val_acc)
        if va_auc > best_val_auc:
            best_val_auc = va_auc
            best_val_acc = va_acc
            best_val_f1  = va_f1
            torch.save({
                'epoch': epoch, 'seed': seed,
                'model_state_dict': ema.module.state_dict(),
                'ema_state_dict':   ema.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_auc': best_val_auc,
                'best_val_acc': best_val_acc,
                'best_val_f1' : best_val_f1,
            }, Config.CHECKPOINT_DIR / ckpt_name)
            logger.on_checkpoint_saved(
                epoch, va_auc, va_acc,
                path=Config.CHECKPOINT_DIR / ckpt_name,
                reason='best_val_auc',
            )

    logger.finalize(extra={
        'seed': seed,
        'best_val_auc': best_val_auc,
        'best_val_acc': best_val_acc,
        'best_val_f1' : best_val_f1,
    })
    logger.close()
    print(f"\nBest val AUC: {best_val_auc:.4f}  "
          f"(val_acc={best_val_acc:.2f}%  val_f1={best_val_f1:.2f}%)")

    # Regenerate training_history.{json,png} from the real per-epoch data
    # collected above (main.tex's Fig. \ref{fig:training} includes
    # figures/training_history.png -- copy the chosen run's copy from
    # results/ into figures/ when finalising the paper).
    save_metrics(history, Config.RESULTS_DIR / f'training_history{save_suffix}.json')
    plot_training_history(
        history, save_path=Config.RESULTS_DIR / f'training_history{save_suffix}.png')

    # Test-set evaluation with best checkpoint.
    # The checkpoint stores ema.module.state_dict() (clean keys, no "module."
    # prefix), so load into raw_model and run test-time evaluation on it
    # directly -- avoids both the DP overhead at inference time and any
    # key-mismatch when a run started with 2 GPUs is evaluated on 1 or vice
    # versa.
    ckpt = torch.load(Config.CHECKPOINT_DIR / ckpt_name, map_location=device)
    raw_model.load_state_dict(ckpt['model_state_dict'])
    results = full_evaluate(raw_model, test_loader, device, class_names,
                             Config.RESULTS_DIR,
                             use_tta=True, tag=f'v2{save_suffix}')
    results['best_val_auc'] = best_val_auc
    results['best_val_acc'] = best_val_acc
    results['best_val_f1']  = best_val_f1
    results['seed'] = seed
    return results


# ── Entry points ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-multi-gpu', dest='multi_gpu', action='store_false',
                        help='Disable DataParallel; force single-GPU training '
                             'even when 2+ GPUs are visible (default: use all '
                             'visible GPUs).')
    parser.set_defaults(multi_gpu=True)
    args = parser.parse_args()

    if args.eval_only:
        # Single-GPU inference is sufficient here; test-set evaluation is
        # cheap and DP overhead at inference-only outweighs its benefit.
        device = Config.DEVICE
        _, _, test_loader, class_names = create_dataloaders(Config.DATA_DIR)
        model = create_model_v2(Config.NUM_CLASSES).to(device)
        ckpt  = torch.load(Config.CHECKPOINT_DIR / 'best_model_v2.pth', map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        full_evaluate(model, test_loader, device, class_names,
                      Config.RESULTS_DIR, use_tta=True)
    else:
        train_v2(seed=args.seed, multi_gpu=args.multi_gpu)