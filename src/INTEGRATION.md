# Integration Patches for train_v2.py

Five edits to wire up `TrainingLogger`, val-AUC checkpoint selection, and (optionally) EMA weights.

Apply them top-to-bottom. Each patch shows a small block of the current file (BEFORE) and the replacement (AFTER). Use your editor's search to locate the `BEFORE` block, then replace with `AFTER`.

---

## Patch 1 — Add imports

**Find near the top of `train_v2.py`, after the other `from .` imports (around line 38):**

### BEFORE
```python
from .config import Config
from .dataset import create_dataloaders
from .model_v2 import (create_model_v2, build_optimizer_v2,
                        build_scheduler_v2, predict_with_tta)
from .utils import plot_training_history, save_metrics, AverageMeter
```

### AFTER
```python
from .config import Config
from .dataset import create_dataloaders
from .model_v2 import (create_model_v2, build_optimizer_v2,
                        build_scheduler_v2, predict_with_tta)
from .utils import plot_training_history, save_metrics, AverageMeter
from .logger import TrainingLogger
from copy import deepcopy
```

---

## Patch 2 — Give `train_epoch` access to the logger

**Find `train_epoch` (around line 131):**

### BEFORE
```python
def train_epoch(model, loader, criterion, optimizer, device, epoch, scaler=None):
    model.train()
    meter = AverageMeter()
    correct = total = 0
    use_mixup = getattr(Config, 'USE_MIXUP', True)

    for imgs, labels in tqdm(loader, desc=f"Train E{epoch:03d}", leave=False):
```

### AFTER
```python
def train_epoch(model, loader, criterion, optimizer, device, epoch,
                scaler=None, logger: TrainingLogger = None):
    model.train()
    meter = AverageMeter()
    correct = total = 0
    use_mixup = getattr(Config, 'USE_MIXUP', True)

    for step, (imgs, labels) in enumerate(
            tqdm(loader, desc=f"Train E{epoch:03d}", leave=False)):
```

**Then inside the same function, immediately after `optimizer.step()` (both branches — the AMP one and the non-AMP one), add step logging.** Find:

### BEFORE
```python
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)
            optimizer.step()

        meter.update(loss.item(), imgs.size(0))
        if not mixed:
            correct += (logits.argmax(1) == labels).sum().item()
            total   += labels.size(0)
```

### AFTER
```python
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
```

---

## Patch 3 — Give `validate` access to the logger, and return probs + AUC

The current `validate` returns only `(val_loss, val_acc)` — we need `val_auc` too for checkpoint selection.

**Find `validate` (around line 176):**

### BEFORE
```python
@torch.no_grad()
def validate(model, loader, criterion, device, use_tta=False):
    model.eval()
    meter = AverageMeter()
    correct = total = 0

    for imgs, labels in tqdm(loader, desc="  Val", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss   = criterion(logits, labels)
        meter.update(loss.item(), imgs.size(0))

        if use_tta:
            preds = torch.stack([
                predict_with_tta(model, imgs[i:i+1], device,
                                  getattr(Config, 'TTA_N_AUG', 5)).argmax()
                for i in range(imgs.size(0))
            ])
        else:
            preds = logits.argmax(1)

        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    return meter.avg, correct / total * 100
```

### AFTER
```python
@torch.no_grad()
def validate(model, loader, criterion, device, use_tta=False,
             logger: TrainingLogger = None):
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
    return meter.avg, acc, auc, f1
```

Note: the return signature changed from 2 to 4 values. The main loop needs updating (Patch 5).

---

## Patch 4 — Add an EMA helper (small class near the top of the file)

Insert this right after the FocalLoss / OHEMLoss classes (around line 130, before `train_epoch`):

```python
# ── EMA (Exponential Moving Average of weights) ──────────────────────────────

class ModelEMA:
    """
    Keeps a shadow copy of the model with weights updated as
        ema_w = decay * ema_w + (1 - decay) * model_w
    Evaluate on the EMA copy instead of the raw model — noticeably more
    stable on small datasets.
    """
    def __init__(self, model, decay: float = 0.9995):
        self.decay = decay
        self.module = deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1 - self.decay)
            else:
                v.copy_(msd[k])
```

---

## Patch 5 — Rewire the main `train_v2` loop

**Find `train_v2` (around line 261) — replace the whole loop body:**

### BEFORE (the entire body from `for epoch in range` through `print(f"\nBest val acc...")`)
```python
    history      = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    unfreeze_ep  = getattr(Config, 'UNFREEZE_EPOCH', 10)
    use_tta      = getattr(Config, 'USE_TTA', False)
    ckpt_name    = f'best_model_v2{save_suffix}.pth'

    for epoch in range(1, Config.NUM_EPOCHS + 1):
        if epoch == unfreeze_ep:
            model.unfreeze_backbone()

        tr_loss, tr_acc = train_epoch(model, train_loader, criterion,
                                       optimizer, device, epoch, scaler)
        va_loss, va_acc = validate(model, val_loader, criterion, device,
                                    use_tta=use_tta)
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(va_loss)
        history['val_acc'].append(va_acc)

        marker = ' ◀ best' if va_acc > best_val_acc else ''
        print(f"Ep {epoch:3d}/{Config.NUM_EPOCHS}  "
              f"train {tr_acc:.1f}% ({tr_loss:.4f})  "
              f"val {va_acc:.1f}% ({va_loss:.4f})  "
              f"lr {lr:.2e}{marker}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({
                'epoch': epoch, 'seed': seed,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
            }, Config.CHECKPOINT_DIR / ckpt_name)

    print(f"\nBest val acc: {best_val_acc:.2f}%")
```

### AFTER
```python
    unfreeze_ep = getattr(Config, 'UNFREEZE_EPOCH', 10)
    use_tta     = getattr(Config, 'USE_TTA', False)
    ckpt_name   = f'best_model_v2{save_suffix}.pth'

    # Logger and EMA
    logger = TrainingLogger(
        run_name  = f'v2_s{seed}',
        save_dir  = Config.LOG_DIR,
        use_tensorboard = True,
        class_names = class_names,
    )
    ema = ModelEMA(model, decay=getattr(Config, 'EMA_DECAY', 0.9995))

    best_val_auc = -1.0
    best_val_acc = -1.0

    for epoch in range(1, Config.NUM_EPOCHS + 1):
        if epoch == unfreeze_ep:
            model.unfreeze_backbone()

        logger.on_epoch_start(epoch)

        # Train — logger is passed so it captures per-step loss/lr/grad_norm
        tr_loss, tr_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            scaler=scaler, logger=logger,
        )
        ema.update(model)

        # Validate the EMA copy (more stable than raw weights)
        va_loss, va_acc, va_auc, va_f1 = validate(
            ema.module, val_loader, criterion, device,
            use_tta=use_tta, logger=logger,
        )
        scheduler.step()

        train_metrics = logger.finish_train_epoch()
        val_metrics   = logger.finish_val_epoch()

        # Report both LRs (backbone and transformer)
        lr_bb = optimizer.param_groups[0]['lr']
        lr_tf = optimizer.param_groups[-1]['lr']
        logger.on_epoch_end(
            train_metrics, val_metrics,
            lrs={'backbone': lr_bb, 'transformer': lr_tf},
        )

        # ── AUC-based checkpoint selection (was: val_acc) ─────────────────
        if va_auc > best_val_auc:
            best_val_auc = va_auc
            best_val_acc = va_acc
            torch.save({
                'epoch': epoch, 'seed': seed,
                'model_state_dict': ema.module.state_dict(),
                'ema_state_dict':   ema.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_auc': best_val_auc,
                'best_val_acc': best_val_acc,
                'best_val_f1' : va_f1,
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
    })
    logger.close()
    print(f"\nBest val AUC: {best_val_auc:.4f}  (val_acc={best_val_acc:.2f}%)")
```

Note: also delete or comment-out the two lines below the loop that reference `history`:

```python
save_metrics(history, Config.RESULTS_DIR / f'training_history_v2{save_suffix}.json')
plot_training_history(history, Config.RESULTS_DIR / f'training_history_v2{save_suffix}.png')
```

The logger now handles both — its `training_curves.png` supersedes the old plot, and its `epoch_log.csv` supersedes the old JSON.

Also update the final test-eval block just below to use `best_val_auc`/`best_val_acc`:

```python
    results['best_val_auc'] = best_val_auc
    results['best_val_acc'] = best_val_acc
    results['seed'] = seed
    return results
```

---

## Optional: add EMA_DECAY to `config.py`

Not strictly required — the default 0.9995 in `ModelEMA.__init__` is a safe value. But if you want it configurable, add to `Config`:

```python
EMA_DECAY = 0.9995
```

---

## What you get per run after these patches

For every training run, a folder appears under `logs/`:

```
logs/v2_s42_20260710-093015/
├── train.log                # human-readable log
├── epoch_log.csv            # per-epoch aggregate → plot directly with pandas
├── step_log.jsonl           # per-batch loss/lr/grad_norm → detailed diagnostics
├── summary.json             # final best metrics
├── training_curves.png      # 4-panel plot (loss, acc, AUC+F1, LR)
└── tb/                      # TensorBoard event files
    └── events.out.tfevents.*
```

Live TensorBoard: `tensorboard --logdir=logs` in another terminal, open http://localhost:6006.

---

## Quick sanity check before the full 5-seed run

Run one short seed first to confirm everything integrates cleanly:

```
python -m src.train_v2 --seed 42
```

Then check:
- `logs/v2_s42_*/epoch_log.csv` — should have 100 rows with sensible values
- `logs/v2_s42_*/training_curves.png` — should show clean curves
- `checkpoints/best_model_v2_s42.pth` — should contain `best_val_auc` key
- Console should show the new `[epoch NNN] tr_loss=... val_auc=...` format

Once that works, kick off `python -m src.run_seeds` for the full sweep.
