"""
recover_test_eval.py -- one-off recovery script.

Use case: train_v2()/run_seeds.py finished a full training run and saved a
good checkpoint, but then crashed trying to reload that SAME checkpoint for
the final test-set evaluation, because PyTorch >=2.6's torch.load() default
(weights_only=True) refuses to unpickle the numpy.float64 metadata values
that older code saved into the checkpoint (best_val_auc/acc/f1). That numpy
issue is fixed going forward in train_v2.py (validate()/full_evaluate() now
return plain floats), but a checkpoint saved by the OLD code before that fix
still has the numpy values baked in.

This script does two things:
  1. Loads the existing checkpoint with weights_only=False (safe here
     specifically because it's a file YOU trained on YOUR OWN Kaggle session
     moments ago, not something downloaded from an untrusted source) and
     runs the pending test-set evaluation, so you don't have to re-run the
     full (expensive) training loop again just to get real numbers.
  2. HEALS the checkpoint in place: re-saves it with the numpy metadata
     cast to plain floats, so every OTHER script that loads this exact file
     from now on -- ensemble.py, explainability.py, tsne_embeddings.py, or
     a future `train_v2.py --eval-only` run -- can use the normal, safe
     default torch.load() with no override, and won't hit this crash again.

Usage:
    python -m src.recover_test_eval --checkpoint checkpoints/best_model_v2_s42.pth --tag v2_s42
"""
import argparse
import torch

from .config import Config
from .dataset import create_dataloaders
from .model_v2 import create_model_v2
from .train_v2 import full_evaluate


def _to_native(value):
    """Cast a numpy scalar (or anything numeric) to a plain Python float;
    leave anything else (None, etc.) untouched."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--tag', type=str, default='v2')
    args = parser.parse_args()

    device = Config.DEVICE
    print(f"Loading {args.checkpoint} (weights_only=False -- trusted, self-generated file)...")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    model = create_model_v2(Config.NUM_CLASSES).to(device)
    model.load_state_dict(ckpt['model_state_dict'])

    print(f"Checkpoint epoch={ckpt.get('epoch')}  "
          f"best_val_auc={ckpt.get('best_val_auc')}  "
          f"best_val_acc={ckpt.get('best_val_acc')}  "
          f"best_val_f1={ckpt.get('best_val_f1')}")

    # --- Heal the checkpoint in place (see docstring point 2 above) ---
    for key in ('best_val_auc', 'best_val_acc', 'best_val_f1'):
        if key in ckpt:
            ckpt[key] = _to_native(ckpt[key])
    torch.save(ckpt, args.checkpoint)
    print(f"Healed checkpoint in place (now safe under torch.load's default "
          f"weights_only=True): {args.checkpoint}")

    _, _, test_loader, class_names = create_dataloaders(Config.DATA_DIR)
    results = full_evaluate(model, test_loader, device, class_names,
                             Config.RESULTS_DIR, use_tta=True, tag=args.tag)

    print(f"\nTest accuracy: {results['accuracy']:.2f}%")
    if results['auc'] is not None:
        print(f"Test AUC: {results['auc']:.4f}")


if __name__ == '__main__':
    main()
