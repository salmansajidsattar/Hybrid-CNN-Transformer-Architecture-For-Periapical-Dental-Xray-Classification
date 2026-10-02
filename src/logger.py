"""
logger.py — Structured training logger for HCNT
================================================
One class (TrainingLogger) that handles ALL logging in one place:

    • Per-step (batch)   → JSONL   (step_log.jsonl)     — machine-parsable
    • Per-epoch aggregate→ CSV     (epoch_log.csv)      — easy to plot
    • Per-epoch aggregate→ TensorBoard scalars          — live curves
    • Checkpoint events  → text log (train.log)         — best-model diffs
    • End of training    → summary JSON + PNG plots     — paper-ready

Also captures:
    • Wall time per step and per epoch
    • Learning rates (backbone + transformer, differential LR)
    • Gradient norm (for stability diagnostics)
    • Val AUC (needed for AUC-based checkpoint selection)

Usage inside train_v2.py:

    logger = TrainingLogger(run_name=f'v2_s{seed}', save_dir=Config.LOG_DIR)

    for epoch in range(num_epochs):
        logger.on_epoch_start(epoch)
        for step, (imgs, labels) in enumerate(train_loader):
            # ... forward, loss, backward ...
            logger.on_train_step(
                epoch=epoch, step=step,
                loss=loss.item(), lr=optimizer.param_groups[0]['lr'],
                grad_norm=grad_norm.item(),
            )
        train_metrics = logger.finish_train_epoch()

        for step, (imgs, labels) in enumerate(val_loader):
            # ... forward, loss ...
            logger.on_val_step(loss=loss.item(), probs=probs, labels=labels)
        val_metrics = logger.finish_val_epoch()

        logger.on_epoch_end(train_metrics, val_metrics,
                            lrs={'backbone': ..., 'transformer': ...})

        if val_metrics['auc'] > best_val_auc:
            logger.on_checkpoint_saved(epoch, val_metrics['auc'], path=...)

    logger.close()
"""

import csv
import json
import time
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              precision_recall_fscore_support)

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_OK = True
except ImportError:
    _TB_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────────────

class _Accumulator:
    """Rolling mean over a stream of values."""
    def __init__(self):
        self.sum = 0.0
        self.n   = 0
    def add(self, v, w=1):
        self.sum += float(v) * w
        self.n   += w
    @property
    def mean(self):
        return self.sum / max(1, self.n)


def _mkdir(p: Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ─────────────────────────────────────────────────────────────────────────────
# TrainingLogger
# ─────────────────────────────────────────────────────────────────────────────

class TrainingLogger:
    def __init__(self,
                 run_name: str,
                 save_dir,
                 use_tensorboard: bool = True,
                 log_every_n_steps: int = 10,
                 class_names=None):
        """
        Args:
            run_name          : identifier included in filenames (e.g. 'v2_s42')
            save_dir          : parent directory for logs (typically Config.LOG_DIR)
            use_tensorboard   : write TensorBoard scalar summaries
            log_every_n_steps : how often to print step-level progress to console
            class_names       : list of class names for classification report
        """
        self.run_name = run_name
        self.timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.run_dir = _mkdir(Path(save_dir) / f'{run_name}_{self.timestamp}')

        self.log_every_n_steps = log_every_n_steps
        self.class_names = class_names or ['non_periapical', 'periapical']

        # ── File logger ───────────────────────────────────────────────────────
        self.log_file = self.run_dir / 'train.log'
        self.logger = logging.getLogger(f'HCNT.{run_name}')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []
        fh = logging.FileHandler(self.log_file, encoding='utf-8')
        fh.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s'))
        self.logger.addHandler(fh)
        self.logger.propagate = False

        # ── Step-level JSONL sink ─────────────────────────────────────────────
        self.step_log_path = self.run_dir / 'step_log.jsonl'
        self._step_fp = open(self.step_log_path, 'w', encoding='utf-8')

        # ── Epoch-level CSV sink ──────────────────────────────────────────────
        self.epoch_csv_path = self.run_dir / 'epoch_log.csv'
        self._epoch_fields = [
            'epoch', 'train_loss', 'train_acc',
            'val_loss', 'val_acc', 'val_auc', 'val_f1_macro',
            'lr_backbone', 'lr_transformer',
            'grad_norm_mean', 'epoch_time_s', 'timestamp',
        ]
        self._epoch_fp = open(self.epoch_csv_path, 'w', newline='',
                              encoding='utf-8')
        self._epoch_writer = csv.DictWriter(self._epoch_fp,
                                            fieldnames=self._epoch_fields)
        self._epoch_writer.writeheader()

        # ── TensorBoard ───────────────────────────────────────────────────────
        self.tb = None
        if use_tensorboard and _TB_OK:
            self.tb = SummaryWriter(log_dir=str(self.run_dir / 'tb'))

        # ── State for the current epoch ───────────────────────────────────────
        self._reset_epoch_state()
        self.best_val_auc = -1.0
        self.best_val_acc = -1.0
        self.best_epoch   = -1
        self.history = []           # list of per-epoch dicts

        self._banner()

    # ─────────────────────────────────────────────────────────────────────────
    # Banner + close
    # ─────────────────────────────────────────────────────────────────────────

    def _banner(self):
        msg = (f'\n{"="*72}\n'
               f' RUN: {self.run_name}\n'
               f' Dir: {self.run_dir}\n'
               f' Started: {self.timestamp}\n'
               f' TensorBoard: {"ON" if self.tb else "OFF"}\n'
               f'{"="*72}')
        print(msg)
        self.logger.info(msg)

    def close(self):
        self._step_fp.close()
        self._epoch_fp.close()
        if self.tb is not None:
            self.tb.close()
        for h in self.logger.handlers[:]:
            h.close()
            self.logger.removeHandler(h)

    # ─────────────────────────────────────────────────────────────────────────
    # Epoch lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    def _reset_epoch_state(self):
        self._train_loss = _Accumulator()
        self._grad_norm  = _Accumulator()
        self._train_correct = 0
        self._train_total   = 0

        self._val_loss = _Accumulator()
        self._val_probs  = []
        self._val_labels = []

        self._epoch_start_time = None
        self._epoch_step_count = 0

    def on_epoch_start(self, epoch: int):
        self._reset_epoch_state()
        self._epoch_start_time = time.time()
        self.current_epoch = epoch

    # ─────────────────────────────────────────────────────────────────────────
    # Train-step logging
    # ─────────────────────────────────────────────────────────────────────────

    def on_train_step(self, epoch: int, step: int,
                      loss: float, lr: float,
                      preds=None, labels=None,
                      grad_norm: float = None,
                      step_time_s: float = None,
                      extra: dict = None):
        """
        Called after each training batch. `preds` and `labels` are optional;
        if provided, running train accuracy is tracked.
        """
        self._epoch_step_count += 1
        self._train_loss.add(loss)
        if grad_norm is not None:
            self._grad_norm.add(grad_norm)

        if preds is not None and labels is not None:
            if isinstance(preds, torch.Tensor):
                preds = preds.detach().cpu().numpy()
            if isinstance(labels, torch.Tensor):
                labels = labels.detach().cpu().numpy()
            self._train_correct += int((preds == labels).sum())
            self._train_total   += int(len(labels))

        # Persist per-step JSONL
        row = {
            'phase'    : 'train',
            'epoch'    : int(epoch),
            'step'     : int(step),
            'loss'     : float(loss),
            'lr'       : float(lr),
            'grad_norm': float(grad_norm) if grad_norm is not None else None,
            'ts'       : time.time(),
        }
        if extra:
            row.update(extra)
        self._step_fp.write(json.dumps(row) + '\n')

        # Console throttling
        if (step % self.log_every_n_steps) == 0:
            msg = (f'  ep{epoch:03d} step{step:04d} '
                   f'loss={loss:.4f} lr={lr:.2e}')
            if grad_norm is not None:
                msg += f' grad={grad_norm:.3f}'
            print(msg)

        # TensorBoard: log every step
        if self.tb is not None:
            global_step = epoch * 10000 + step   # ~unique across epochs
            self.tb.add_scalar('train_step/loss', loss, global_step)
            self.tb.add_scalar('train_step/lr', lr, global_step)
            if grad_norm is not None:
                self.tb.add_scalar('train_step/grad_norm', grad_norm, global_step)

    # ─────────────────────────────────────────────────────────────────────────
    # Val-step logging
    # ─────────────────────────────────────────────────────────────────────────

    def on_val_step(self, loss: float,
                    probs=None, labels=None,
                    extra: dict = None):
        """
        Called after each validation batch. `probs` is (B, num_classes) softmax
        and `labels` is (B,) integer targets - both required for epoch aggregation.
        """
        self._val_loss.add(loss)
        if probs is not None and labels is not None:
            if isinstance(probs, torch.Tensor):
                probs = probs.detach().cpu().numpy()
            if isinstance(labels, torch.Tensor):
                labels = labels.detach().cpu().numpy()
            self._val_probs.append(probs)
            self._val_labels.append(labels)

        # JSONL
        row = {
            'phase' : 'val',
            'epoch' : int(getattr(self, 'current_epoch', -1)),
            'loss'  : float(loss),
            'ts'    : time.time(),
        }
        if extra:
            row.update(extra)
        self._step_fp.write(json.dumps(row) + '\n')

    # ─────────────────────────────────────────────────────────────────────────
    # Epoch aggregation
    # ─────────────────────────────────────────────────────────────────────────

    def finish_train_epoch(self) -> dict:
        train_acc = (self._train_correct / self._train_total
                     if self._train_total > 0 else float('nan'))
        return {
            'loss'      : self._train_loss.mean,
            'acc'       : train_acc,
            'grad_norm' : self._grad_norm.mean if self._grad_norm.n > 0 else 0.0,
        }

    def finish_val_epoch(self) -> dict:
        if self._val_probs:
            probs  = np.concatenate(self._val_probs)
            labels = np.concatenate(self._val_labels)
            preds  = probs.argmax(axis=1)
            acc    = accuracy_score(labels, preds)
            try:
                auc = roc_auc_score(labels, probs[:, 1])
            except ValueError:
                auc = float('nan')
            f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
            prec, rec, _, _ = precision_recall_fscore_support(
                labels, preds, average='macro', zero_division=0)
        else:
            acc = auc = f1_macro = prec = rec = float('nan')

        return {
            'loss'     : self._val_loss.mean,
            'acc'      : acc,
            'auc'      : auc,
            'f1_macro' : f1_macro,
            'precision': prec,
            'recall'   : rec,
        }

    def on_epoch_end(self,
                     train_metrics: dict,
                     val_metrics: dict,
                     lrs: dict = None):
        """
        Called after the epoch's train + val loops are done.
        Writes CSV + TensorBoard + console table row.

        Args:
            train_metrics : output of finish_train_epoch()
            val_metrics   : output of finish_val_epoch()
            lrs           : {'backbone': ..., 'transformer': ...}
        """
        elapsed = time.time() - self._epoch_start_time
        lrs = lrs or {}
        epoch = self.current_epoch

        # Persist per-epoch CSV row
        row = {
            'epoch'          : epoch,
            'train_loss'     : round(train_metrics['loss'], 6),
            'train_acc'      : round(train_metrics.get('acc', float('nan')), 6),
            'val_loss'       : round(val_metrics['loss'], 6),
            'val_acc'        : round(val_metrics['acc'], 6),
            'val_auc'        : round(val_metrics['auc'], 6),
            'val_f1_macro'   : round(val_metrics['f1_macro'], 6),
            'lr_backbone'    : lrs.get('backbone'),
            'lr_transformer' : lrs.get('transformer'),
            'grad_norm_mean' : round(train_metrics.get('grad_norm', 0.0), 6),
            'epoch_time_s'   : round(elapsed, 2),
            'timestamp'      : datetime.now().isoformat(timespec='seconds'),
        }
        self._epoch_writer.writerow(row)
        self._epoch_fp.flush()
        self.history.append(row)

        # TensorBoard aggregates
        if self.tb is not None:
            self.tb.add_scalar('epoch/train_loss', train_metrics['loss'], epoch)
            self.tb.add_scalar('epoch/train_acc',
                               train_metrics.get('acc', 0.0), epoch)
            self.tb.add_scalar('epoch/val_loss', val_metrics['loss'], epoch)
            self.tb.add_scalar('epoch/val_acc', val_metrics['acc'], epoch)
            self.tb.add_scalar('epoch/val_auc', val_metrics['auc'], epoch)
            self.tb.add_scalar('epoch/val_f1_macro',
                               val_metrics['f1_macro'], epoch)
            for k, v in lrs.items():
                if v is not None:
                    self.tb.add_scalar(f'epoch/lr_{k}', v, epoch)
            self.tb.add_scalar('epoch/time_s', elapsed, epoch)

        # Console + file log summary
        msg = (f'[epoch {epoch:03d}] '
               f'tr_loss={train_metrics["loss"]:.4f} '
               f'tr_acc={train_metrics.get("acc", float("nan")):.4f}  |  '
               f'val_loss={val_metrics["loss"]:.4f} '
               f'val_acc={val_metrics["acc"]:.4f} '
               f'val_auc={val_metrics["auc"]:.4f} '
               f'val_f1={val_metrics["f1_macro"]:.4f}  '
               f'({elapsed:.1f}s)')
        print(msg)
        self.logger.info(msg)

    # ─────────────────────────────────────────────────────────────────────────
    # Checkpoint events
    # ─────────────────────────────────────────────────────────────────────────

    def on_checkpoint_saved(self, epoch: int, val_auc: float, val_acc: float,
                            path, reason: str = 'best_val_auc'):
        """
        Call whenever a new best-model checkpoint is saved.
        """
        if val_auc > self.best_val_auc:
            self.best_val_auc = val_auc
            self.best_val_acc = val_acc
            self.best_epoch   = epoch
        msg = (f'  [ckpt] epoch={epoch:03d} '
               f'val_auc={val_auc:.4f} val_acc={val_acc:.4f} '
               f'reason={reason}  -> {path}')
        print(msg)
        self.logger.info(msg)
        if self.tb is not None:
            self.tb.add_scalar('checkpoint/val_auc_at_save', val_auc, epoch)
            self.tb.add_scalar('checkpoint/val_acc_at_save', val_acc, epoch)

    # ─────────────────────────────────────────────────────────────────────────
    # End of training: summary + plots
    # ─────────────────────────────────────────────────────────────────────────

    def finalize(self, extra: dict = None):
        """
        Write summary.json and produce training-curve plots.
        Call once after the training loop exits.
        """
        summary = {
            'run_name'      : self.run_name,
            'timestamp'     : self.timestamp,
            'best_epoch'    : self.best_epoch,
            'best_val_auc'  : self.best_val_auc,
            'best_val_acc'  : self.best_val_acc,
            'n_epochs_run'  : len(self.history),
            'epoch_log_csv' : str(self.epoch_csv_path),
            'step_log_jsonl': str(self.step_log_path),
        }
        if extra:
            summary.update(extra)

        with open(self.run_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        self._plot_curves()

        msg = (f'\n{"="*72}\n'
               f' TRAINING DONE  |  best epoch={self.best_epoch}  '
               f'val_auc={self.best_val_auc:.4f}  '
               f'val_acc={self.best_val_acc:.4f}\n'
               f' Outputs: {self.run_dir}\n'
               f'{"="*72}\n')
        print(msg)
        self.logger.info(msg)

    # ─────────────────────────────────────────────────────────────────────────
    # Plotting
    # ─────────────────────────────────────────────────────────────────────────

    def _plot_curves(self):
        if not self.history:
            return
        H = self.history
        epochs = [r['epoch'] for r in H]

        fig, axes = plt.subplots(2, 2, figsize=(11, 8))

        # Loss
        axes[0, 0].plot(epochs, [r['train_loss'] for r in H], label='train')
        axes[0, 0].plot(epochs, [r['val_loss'] for r in H], label='val')
        axes[0, 0].set_title('Loss'); axes[0, 0].set_xlabel('epoch')
        axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

        # Accuracy
        axes[0, 1].plot(epochs, [r['train_acc'] for r in H], label='train')
        axes[0, 1].plot(epochs, [r['val_acc'] for r in H], label='val')
        axes[0, 1].set_title('Accuracy'); axes[0, 1].set_xlabel('epoch')
        axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

        # AUC + macro F1
        axes[1, 0].plot(epochs, [r['val_auc'] for r in H], label='val AUC')
        axes[1, 0].plot(epochs, [r['val_f1_macro'] for r in H],
                        label='val macro-F1')
        if self.best_epoch >= 0:
            axes[1, 0].axvline(self.best_epoch, color='r', linestyle='--',
                               label=f'best (ep {self.best_epoch})')
        axes[1, 0].set_title('Val AUC & Macro-F1'); axes[1, 0].set_xlabel('epoch')
        axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

        # LR schedule
        lr_bb = [r.get('lr_backbone') for r in H]
        lr_tf = [r.get('lr_transformer') for r in H]
        if any(x is not None for x in lr_bb):
            axes[1, 1].plot(epochs, lr_bb, label='backbone')
        if any(x is not None for x in lr_tf):
            axes[1, 1].plot(epochs, lr_tf, label='transformer')
        axes[1, 1].set_title('Learning rate'); axes[1, 1].set_xlabel('epoch')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        plot_path = self.run_dir / 'training_curves.png'
        plt.savefig(plot_path, dpi=150)
        plt.close()
        self.logger.info(f'Training curves -> {plot_path}')
