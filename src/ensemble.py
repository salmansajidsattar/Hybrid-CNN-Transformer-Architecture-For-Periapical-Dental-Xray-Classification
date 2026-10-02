"""
ensemble.py — Multi-model ensemble inference
============================================
Loads N checkpoint files and combines their predictions via:
  • Soft voting  : average softmax probabilities  (default, best accuracy)
  • Hard voting  : majority vote on argmax labels

Usage (after training 5 seeds with run_seeds.py):
    python -m src.ensemble
    python -m src.ensemble --method hard
    python -m src.ensemble --ckpts checkpoints/best_model_v2_s42.pth checkpoints/best_model_v2_s7.pth ...

Why ensemble works:
  Each model makes different errors (different random init → different
  loss landscape). Averaging their softmax outputs cancels out individual
  errors and pushes borderline cases past the 0.5 decision boundary.
  Typical gain over a single model: +1 to +2 pp.
"""

import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, accuracy_score)
import matplotlib.pyplot as plt
import seaborn as sns

from .config import Config
from .dataset import create_dataloaders
from .model_v2 import create_model_v2, predict_with_tta


# ── Load a single checkpoint ──────────────────────────────────────────────────

def load_model(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    model = create_model_v2(Config.NUM_CLASSES).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    val_acc = ckpt.get('best_val_acc', '?')
    print(f"  Loaded {Path(ckpt_path).name}  (best_val_acc={val_acc})")
    return model


# ── Collect softmax probabilities for one model ───────────────────────────────

@torch.no_grad()
def get_probs(model, loader, device, use_tta: bool = True, n_aug: int = 5):
    model.eval()
    all_probs, all_labels = [], []

    for imgs, labels in tqdm(loader, desc="  Probs", leave=False):
        imgs = imgs.to(device)
        if use_tta:
            batch_probs = []
            for i in range(imgs.size(0)):
                p = predict_with_tta(model, imgs[i:i+1], device, n_aug)
                batch_probs.append(p.cpu())
            all_probs.extend(batch_probs)
        else:
            logits = model(imgs)
            all_probs.extend(F.softmax(logits, dim=1).cpu())
        all_labels.extend(labels.numpy())

    return torch.stack(all_probs).numpy(), np.array(all_labels)


# ── Ensemble ──────────────────────────────────────────────────────────────────

def ensemble_predict(ckpt_paths: list, loader, device, class_names: list,
                     method: str = 'soft', use_tta: bool = True,
                     save_dir: Path = None):
    """
    Args:
        ckpt_paths : list of checkpoint file paths
        method     : 'soft' (average probs) or 'hard' (majority vote)
    """
    print(f"\nEnsemble method: {method.upper()}  |  TTA: {use_tta}")
    print(f"Loading {len(ckpt_paths)} models...")

    all_model_probs = []   # shape: (n_models, n_samples, n_classes)
    labels = None

    for path in ckpt_paths:
        model = load_model(path, device)
        probs, labs = get_probs(model, loader, device, use_tta)
        all_model_probs.append(probs)
        labels = labs
        del model
        torch.cuda.empty_cache()

    stack = np.stack(all_model_probs)  # (n_models, n_samples, n_classes)

    if method == 'soft':
        avg_probs = stack.mean(axis=0)             # (n_samples, n_classes)
        preds     = avg_probs.argmax(axis=1)
        auc_probs = avg_probs[:, 1]
    else:  # hard voting
        hard_preds = stack.argmax(axis=2)          # (n_models, n_samples)
        # Majority vote
        from scipy import stats
        preds, _ = stats.mode(hard_preds, axis=0)
        preds     = preds.squeeze()
        avg_probs = stack.mean(axis=0)
        auc_probs = avg_probs[:, 1]

    acc = accuracy_score(labels, preds) * 100
    auc = roc_auc_score(labels, auc_probs)

    print(f"\n{'='*60}")
    print(f"ENSEMBLE RESULTS ({method.upper()} voting, {len(ckpt_paths)} models)")
    print(f"{'='*60}")
    print(classification_report(labels, preds, target_names=class_names, digits=4))
    print(f"Accuracy : {acc:.2f}%")
    print(f"AUC-ROC  : {auc:.4f}")

    # Save confusion matrix
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted'); plt.ylabel('Actual')
        plt.title(f'Ensemble Confusion Matrix ({method})')
        plt.tight_layout()
        path = save_dir / f'confusion_matrix_ensemble_{method}.png'
        plt.savefig(path, dpi=150); plt.close()
        print(f"Confusion matrix → {path}")

        results = {'accuracy': acc, 'auc': auc, 'method': method,
                   'n_models': len(ckpt_paths)}
        with open(save_dir / 'ensemble_results.json', 'w') as f:
            json.dump(results, f, indent=2)

    return acc, auc, preds


# ── Auto-discover checkpoints ─────────────────────────────────────────────────

def find_seed_checkpoints(ckpt_dir: Path) -> list:
    """Find all best_model_v2_s*.pth files (produced by run_seeds.py)."""
    paths = sorted(ckpt_dir.glob('best_model_v2_s*.pth'))
    if not paths:
        # Fall back to a single best model
        single = ckpt_dir / 'best_model_v2.pth'
        if single.exists():
            paths = [single]
    return [str(p) for p in paths]


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpts', nargs='+', default=None,
                        help='Checkpoint paths. If omitted, auto-discovers best_model_v2_s*.pth')
    parser.add_argument('--method', choices=['soft', 'hard'], default='soft')
    parser.add_argument('--no-tta', action='store_true')
    args = parser.parse_args()

    device = Config.DEVICE
    _, _, test_loader, class_names = create_dataloaders(Config.DATA_DIR)

    ckpt_paths = args.ckpts or find_seed_checkpoints(Config.CHECKPOINT_DIR)
    if not ckpt_paths:
        raise FileNotFoundError(
            "No checkpoints found. Run `python -m src.run_seeds` first.")

    print(f"Found {len(ckpt_paths)} checkpoints: {ckpt_paths}")

    ensemble_predict(
        ckpt_paths  = ckpt_paths,
        loader      = test_loader,
        device      = device,
        class_names = class_names,
        method      = args.method,
        use_tta     = not args.no_tta,
        save_dir    = Config.RESULTS_DIR,
    )
