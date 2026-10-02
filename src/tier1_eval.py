"""
tier1_eval.py — Full Week-1 evaluation pipeline (ensemble + threshold + audit)
=============================================================================
Implements the three Tier-1 improvements over the multi-seed baseline:

  Step 1: Soft-vote ensemble of the 5 seed checkpoints (uses existing
          ensemble.py helpers - averages softmax over N models with TTA).
  Step 2: Threshold tuning on the validation set for macro-F1, then
          applied to the test set. Corrects the model's positive-class
          bias (seed 99 showed Non-Periapical recall of only 0.76).
  Step 3: Exports the ensemble's test-set misclassifications to CSV
          for clinician label audit. Includes image path, true label,
          predicted label, and per-class probability.

Usage:
    python -m src.tier1_eval                    # runs all three steps
    python -m src.tier1_eval --no-tta           # skip TTA (~5x faster)
    python -m src.tier1_eval --ckpts A B C ...  # override checkpoints

Outputs (in results/):
    tier1_results.json               summary metrics (before/after threshold)
    tier1_misclassifications.csv     for M.D.A.A. to review
    tier1_confusion_matrix.png       final ensemble + threshold confusion
    tier1_threshold_sweep.png        val macro-F1 vs threshold plot

Expected impact vs. single-seed baseline (88.71 +/- 3.14 %):
    Step 1 (ensemble):        ~90-91 % test acc, std drops to ~1 pp
    Step 2 (threshold tune):  +1.0 to +1.5 pp
    Step 3 (label audit):     +1-2 pp if genuine label errors exist
    Combined:                 ~92-94 % test accuracy
"""

import argparse
import csv
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              classification_report, f1_score, roc_auc_score,
                              precision_recall_fscore_support)

from .config    import Config
from .dataset   import create_dataloaders
from .ensemble  import load_model, get_probs, find_seed_checkpoints


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 helper: run all models and stack per-sample softmax probabilities
# ─────────────────────────────────────────────────────────────────────────────

def collect_ensemble_probs(ckpt_paths, loader, device, use_tta=True):
    """
    Returns:
        avg_probs : (N, 2) numpy array - mean softmax across all models
        labels    : (N,)   numpy array - ground-truth labels
    """
    all_model_probs = []
    labels = None
    for i, path in enumerate(ckpt_paths):
        print(f"[{i+1}/{len(ckpt_paths)}] {Path(path).name}")
        model = load_model(path, device)
        probs, labs = get_probs(model, loader, device, use_tta=use_tta)
        all_model_probs.append(probs)
        labels = labs
        del model
        torch.cuda.empty_cache()

    stack = np.stack(all_model_probs)              # (M, N, 2)
    avg   = stack.mean(axis=0)                     # (N, 2)
    return avg, labels


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: sweep decision threshold on val for max macro-F1
# ─────────────────────────────────────────────────────────────────────────────

def tune_threshold(val_probs, val_labels, thresholds=None):
    """
    Sweep the positive-class threshold on the val set and pick the one that
    maximises macro-F1.

    Returns:
        best_t     : float - threshold that maximises macro-F1
        best_f1    : float - macro-F1 at that threshold
        sweep      : list of (t, macro_f1) pairs for plotting
    """
    if thresholds is None:
        thresholds = np.linspace(0.30, 0.70, 41)

    sweep = []
    for t in thresholds:
        preds = (val_probs[:, 1] > t).astype(int)
        macro_f1 = f1_score(val_labels, preds, average='macro')
        sweep.append((float(t), float(macro_f1)))

    best_t, best_f1 = max(sweep, key=lambda x: x[1])
    return best_t, best_f1, sweep


def apply_threshold(probs, threshold):
    """Convert (N, 2) softmax → (N,) predictions using positive-class threshold."""
    return (probs[:, 1] > threshold).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: export misclassifications for dentist audit
# ─────────────────────────────────────────────────────────────────────────────

def get_test_image_paths(loader):
    """
    Extract per-sample image paths from the test loader's Dataset.
    Assumes create_dataloaders sets `image_paths` on the Dataset object.
    """
    ds = loader.dataset
    if hasattr(ds, 'image_paths'):
        return list(ds.image_paths)
    # Fallback: numeric indices if paths aren't exposed
    return [f'test_idx_{i}' for i in range(len(ds))]


def export_misclassifications(preds, labels, probs, image_paths,
                              class_names, save_path):
    """
    Write the misclassified test samples to CSV in a form that a dentist
    can review directly.
    """
    rows = []
    for i, (p, y, path) in enumerate(zip(preds, labels, image_paths)):
        if p != y:
            rows.append({
                'index'         : i,
                'image_path'    : str(path),
                'true_label'    : class_names[int(y)],
                'predicted'     : class_names[int(p)],
                'prob_non_peri' : float(probs[i, 0]),
                'prob_peri'     : float(probs[i, 1]),
                'confidence'    : float(max(probs[i])),
                'dentist_verdict': '',     # to be filled: correct / label_error / borderline
                'notes'         : '',
            })

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        else:
            f.write('# No misclassifications found.\n')

    print(f"\nMisclassifications ({len(rows)}) exported to:")
    print(f"  {save_path}")
    print("Have your dentist fill the 'dentist_verdict' column with one of:")
    print("  correct       - model is wrong, ground truth is right")
    print("  label_error   - ground truth is wrong, model is right")
    print("  borderline    - genuinely ambiguous case")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def print_metrics(name, labels, preds, probs, class_names):
    acc = accuracy_score(labels, preds) * 100
    auc = roc_auc_score(labels, probs[:, 1])
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average='macro', zero_division=0
    )
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(classification_report(labels, preds, target_names=class_names,
                                digits=4, zero_division=0))
    print(f"  Accuracy       : {acc:.2f} %")
    print(f"  Macro-F1       : {f1*100:.2f} %")
    print(f"  AUC-ROC        : {auc:.4f}")
    return {'accuracy': acc, 'macro_f1': f1*100, 'auc': auc,
            'macro_precision': prec*100, 'macro_recall': rec*100}


def plot_threshold_sweep(sweep, best_t, save_path):
    ts, f1s = zip(*sweep)
    plt.figure(figsize=(6, 4))
    plt.plot(ts, f1s, marker='.')
    plt.axvline(best_t, color='r', linestyle='--',
                label=f'best t = {best_t:.2f}')
    plt.axvline(0.50, color='grey', linestyle=':', label='default t = 0.50')
    plt.xlabel('Decision threshold (positive class)')
    plt.ylabel('Val macro-F1')
    plt.title('Threshold sweep on validation set')
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(save_path, dpi=150); plt.close()
    print(f"  Threshold sweep plot -> {save_path}")


def plot_confusion(labels, preds, class_names, save_path, title):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150); plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpts', nargs='+', default=None,
                        help='Checkpoint paths (default: auto-discover)')
    parser.add_argument('--no-tta', action='store_true',
                        help='Disable test-time augmentation (~5x faster)')
    parser.add_argument('--outdir', type=str, default=None,
                        help='Output directory (default: Config.RESULTS_DIR)')
    args = parser.parse_args()

    device      = Config.DEVICE
    use_tta     = not args.no_tta
    outdir      = Path(args.outdir) if args.outdir else Config.RESULTS_DIR
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"# TIER 1 EVALUATION PIPELINE")
    print(f"# Device: {device}   TTA: {use_tta}   Outdir: {outdir}")
    print(f"{'#'*60}")

    # Auto-discover checkpoints
    ckpt_paths = args.ckpts or find_seed_checkpoints(Config.CHECKPOINT_DIR)
    if not ckpt_paths:
        raise FileNotFoundError(
            "No checkpoints found. Run `python -m src.run_seeds` first.")
    print(f"\nUsing {len(ckpt_paths)} checkpoints:")
    for p in ckpt_paths:
        print(f"  {p}")

    # Load val and test loaders
    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        Config.DATA_DIR
    )
    print(f"\nClasses: {class_names}")

    # Step 1: soft-vote ensemble on VAL and TEST
    print(f"\n{'-'*60}")
    print("STEP 1: ensemble softmax on VAL set")
    print(f"{'-'*60}")
    val_probs, val_labels = collect_ensemble_probs(
        ckpt_paths, val_loader, device, use_tta=use_tta
    )

    print(f"\n{'-'*60}")
    print("STEP 1: ensemble softmax on TEST set")
    print(f"{'-'*60}")
    test_probs, test_labels = collect_ensemble_probs(
        ckpt_paths, test_loader, device, use_tta=use_tta
    )

    # Report ensemble @ default threshold 0.5
    ensemble_preds_default = apply_threshold(test_probs, 0.50)
    metrics_ensemble = print_metrics(
        'Ensemble (5 seeds, t=0.50)',
        test_labels, ensemble_preds_default, test_probs, class_names
    )

    # Step 2: tune threshold on VAL, apply to TEST
    print(f"\n{'-'*60}")
    print("STEP 2: threshold tuning on val")
    print(f"{'-'*60}")
    best_t, best_val_f1, sweep = tune_threshold(val_probs, val_labels)
    print(f"  Best val macro-F1: {best_val_f1*100:.2f} % at threshold t={best_t:.2f}")
    plot_threshold_sweep(sweep, best_t, outdir / 'tier1_threshold_sweep.png')

    ensemble_preds_tuned = apply_threshold(test_probs, best_t)
    metrics_tuned = print_metrics(
        f'Ensemble (5 seeds, t={best_t:.2f})',
        test_labels, ensemble_preds_tuned, test_probs, class_names
    )

    # Save the tuned confusion matrix
    plot_confusion(
        test_labels, ensemble_preds_tuned, class_names,
        outdir / 'tier1_confusion_matrix.png',
        f'Ensemble + Threshold (t={best_t:.2f}) Confusion Matrix'
    )

    # Step 3: export misclassifications for label audit
    print(f"\n{'-'*60}")
    print("STEP 3: export misclassifications for dentist audit")
    print(f"{'-'*60}")
    image_paths = get_test_image_paths(test_loader)
    misclass = export_misclassifications(
        ensemble_preds_tuned, test_labels, test_probs, image_paths,
        class_names, outdir / 'tier1_misclassifications.csv'
    )

    # Persist all results
    results = {
        'n_models'            : len(ckpt_paths),
        'checkpoints'         : ckpt_paths,
        'use_tta'             : use_tta,
        'best_threshold'      : float(best_t),
        'best_val_macro_f1'   : float(best_val_f1 * 100),
        'ensemble_default'    : metrics_ensemble,
        'ensemble_tuned'      : metrics_tuned,
        'n_misclassified'     : len(misclass),
        'test_set_size'       : int(len(test_labels)),
    }
    with open(outdir / 'tier1_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Summary
    print(f"\n{'#'*60}")
    print(f"# TIER 1 SUMMARY")
    print(f"{'#'*60}")
    print(f"  Ensemble (default t=0.50) : {metrics_ensemble['accuracy']:.2f} %"
          f"  AUC {metrics_ensemble['auc']:.4f}")
    print(f"  Ensemble (tuned t={best_t:.2f}) : {metrics_tuned['accuracy']:.2f} %"
          f"  AUC {metrics_tuned['auc']:.4f}")
    print(f"  Misclassifications         : {len(misclass)} / {len(test_labels)}")
    print(f"\nAll outputs -> {outdir}")


if __name__ == '__main__':
    main()
