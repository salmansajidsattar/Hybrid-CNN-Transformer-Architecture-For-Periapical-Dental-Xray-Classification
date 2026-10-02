"""
run_seeds.py — Multi-seed training for statistical reliability
==============================================================
Trains the model with 5 fixed seeds, collects per-seed metrics, and
saves mean ± std — the standard format required by IEEE reviewers.

Also produces the ablation table data by toggling individual components.

Usage:
    python -m src.run_seeds                 # 5-seed run with full model
    python -m src.run_seeds --seeds 42 7 13 # custom seeds
    python -m src.run_seeds --ablation      # run ablation study too

Output files:
    results/multi_seed_results.json    — per-seed + aggregate stats
    results/ablation_results.json      — ablation component contributions
    results/multi_seed_summary.txt     — human-readable table for the paper
"""

import argparse
import json
import copy
import numpy as np
import torch
from pathlib import Path

from .config import Config
from .train_v2 import train_v2


# ── Aggregate stats ───────────────────────────────────────────────────────────

def compute_stats(values: list) -> dict:
    arr = np.array(values)
    return {
        'mean': float(arr.mean()),
        'std':  float(arr.std()),
        'min':  float(arr.min()),
        'max':  float(arr.max()),
        'values': [float(v) for v in values],
    }


# ── Multi-seed run ────────────────────────────────────────────────────────────

def run_multi_seed(seeds: list, multi_gpu: bool = True) -> dict:
    print(f"\n{'='*60}")
    print(f"MULTI-SEED TRAINING  |  Seeds: {seeds}  |  multi_gpu={multi_gpu}")
    print(f"{'='*60}\n")

    per_seed = []

    for seed in seeds:
        print(f"\n{'─'*60}")
        print(f"  Seed {seed}")
        print(f"{'─'*60}")
        results = train_v2(seed=seed, save_suffix=f'_s{seed}',
                           multi_gpu=multi_gpu)
        per_seed.append({
            'seed':     seed,
            'accuracy': results['accuracy'],
            'auc':      results.get('auc'),
            'best_val': results['best_val_acc'],
        })
        print(f"  Seed {seed} done — test acc: {results['accuracy']:.2f}%")

    accs = [r['accuracy'] for r in per_seed]
    aucs = [r['auc']      for r in per_seed if r['auc'] is not None]
    vals = [r['best_val'] for r in per_seed]

    aggregate = {
        'accuracy':     compute_stats(accs),
        'val_accuracy': compute_stats(vals),
    }
    if aucs:
        aggregate['auc'] = compute_stats(aucs)

    summary = {
        'seeds':     seeds,
        'per_seed':  per_seed,
        'aggregate': aggregate,
    }

    out_path = Config.RESULTS_DIR / 'multi_seed_results.json'
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved → {out_path}")

    _print_summary(summary)
    _save_txt_table(summary, Config.RESULTS_DIR / 'multi_seed_summary.txt')

    return summary


def _print_summary(s: dict):
    print(f"\n{'='*60}")
    print("MULTI-SEED SUMMARY")
    print(f"{'='*60}")
    print(f"{'Seed':<8} {'Test Acc':>10} {'Val Acc':>10} {'AUC':>8}")
    print(f"{'─'*40}")
    for r in s['per_seed']:
        auc_str = f"{r['auc']:.4f}" if r['auc'] else '  N/A '
        print(f"{r['seed']:<8} {r['accuracy']:>9.2f}% {r['best_val']:>9.2f}% {auc_str:>8}")
    print(f"{'─'*40}")
    ag = s['aggregate']
    acc = ag['accuracy']
    auc = ag.get('auc', {})
    print(f"{'Mean':8} {acc['mean']:>9.2f}% "
          f"{ag['val_accuracy']['mean']:>9.2f}% "
          f"{auc.get('mean', 0):>8.4f}")
    print(f"{'Std':8} {acc['std']:>9.2f}% "
          f"{ag['val_accuracy']['std']:>9.2f}% "
          f"{auc.get('std', 0):>8.4f}")
    print(f"{'='*60}")


def _save_txt_table(s: dict, path: Path):
    ag  = s['aggregate']
    acc = ag['accuracy']
    auc = ag.get('auc', {})
    lines = [
        "Multi-seed results (for paper Table V)",
        "",
        f"Seeds tested : {s['seeds']}",
        f"Test accuracy: {acc['mean']:.2f} ± {acc['std']:.2f} %",
        f"AUC-ROC      : {auc.get('mean', 0):.4f} ± {auc.get('std', 0):.4f}",
        "",
        "Per-seed breakdown:",
    ]
    for r in s['per_seed']:
        lines.append(f"  Seed {r['seed']:3d}: acc={r['accuracy']:.2f}%  "
                     f"val={r['best_val']:.2f}%  "
                     f"auc={r['auc']:.4f}" if r['auc'] else
                     f"  Seed {r['seed']:3d}: acc={r['accuracy']:.2f}%  "
                     f"val={r['best_val']:.2f}%")
    path.write_text('\n'.join(lines))
    print(f"Text table → {path}")


# ── Ablation study ────────────────────────────────────────────────────────────

ABLATION_CONFIGS = [
    # name, description, config_overrides -- each row removes exactly ONE
    # component from the full HCNT configuration (matches paper Table III).
    ('Full HCNT (baseline)', 'Complete configuration, no ablation',
     {}),

    ('ImageNet init instead of RadImageNet', 'Ablate domain-specific pretraining',
     {'BACKBONE': 'resnet50'}),

    ('No Transformer (GAP + linear head)', 'Ablate the Transformer encoder entirely',
     {'USE_TRANSFORMER': False}),

    ('Transformer depth 2 (vs 6)', 'Shallower Transformer encoder',
     {'NUM_LAYERS_V2': 2}),

    ('No positional encoding', 'Ablate learnable positional embeddings',
     {'USE_POS_EMBED': False}),

    ('Frozen backbone throughout', 'Backbone never unfrozen (no epoch-10 fine-tuning)',
     {'UNFREEZE_EPOCH': 99999}),

    ('Cross-entropy instead of Focal+OHEM', 'Ablate the hard-example-focused loss',
     {'USE_FOCAL_OHEM': False}),

    ('No DropPath + no LayerScale', 'Ablate both small-data regularisers',
     {'DROP_PATH_RATE': 0.0, 'USE_LAYER_SCALE': False}),

    ('No MixUp + no CutMix', 'Ablate batch-level augmentation',
     {'USE_MIXUP': False, 'USE_CUTMIX': False}),

    ('No test-time augmentation', 'Ablate TTA at inference',
     {'USE_TTA': False}),
]


def run_ablation(seeds: list = None, multi_gpu: bool = True,
                 ablation_epochs: int = None) -> dict:
    """
    Train each ablation config across multiple seeds (default: 3, matching
    the paper's Table III "3 Seeds per row") and aggregate mean +/- std.
    Each row removes exactly one component from the full HCNT config; the
    first row is the unablated baseline, included for reference/sanity
    checking against the corresponding row of the main results table.

    ablation_epochs : when not None, temporarily sets Config.NUM_EPOCHS to
                      this value for the ENTIRE ablation phase (restored
                      afterwards). Ablation deltas usually emerge well
                      before full convergence, so a shorter budget (e.g.
                      50 epochs instead of 100) halves the ablation wall
                      clock while preserving the relative component
                      ordering that the ablation table is actually
                      reporting. Multi-seed phase is unaffected.
    """
    if seeds is None:
        seeds = [42, 7, 13]

    # Short-budget override for the ablation phase only. Captured here and
    # restored in the matching finally-block below so an exception in the
    # middle of the sweep does not leak the shortened epoch count into any
    # follow-on code.
    orig_num_epochs = getattr(Config, 'NUM_EPOCHS', None)
    if ablation_epochs is not None:
        setattr(Config, 'NUM_EPOCHS', int(ablation_epochs))
        print(f"[Ablation] Overriding Config.NUM_EPOCHS "
              f"{orig_num_epochs} -> {ablation_epochs} for this phase only.")

    print(f"\n{'='*60}")
    print(f"ABLATION STUDY  |  Seeds={seeds}  "
          f"({len(ABLATION_CONFIGS)} configs)  |  multi_gpu={multi_gpu}  "
          f"|  epochs={getattr(Config, 'NUM_EPOCHS', '?')}")
    print(f"{'='*60}\n")

    ablation_results = []

    for name, desc, overrides in ABLATION_CONFIGS:
        print(f"\n── {name} ──")
        print(f"   {desc}")

        # Apply config overrides
        original = {}
        for k, v in overrides.items():
            original[k] = getattr(Config, k, None)
            setattr(Config, k, v)

        accs, aucs = [], []
        try:
            tag_base = name[:24].replace(' ', '_').replace('(', '').replace(')', '')
            for seed in seeds:
                results = train_v2(seed=seed,
                                   save_suffix=f'_abl_{tag_base}_s{seed}',
                                   multi_gpu=multi_gpu)
                accs.append(results['accuracy'])
                if results.get('auc') is not None:
                    aucs.append(results['auc'])
                print(f"  seed={seed} → acc={results['accuracy']:.2f}%")
            ablation_results.append({
                'name':     name,
                'desc':     desc,
                'seeds':    seeds,
                'accuracy': compute_stats(accs),
                'auc':      compute_stats(aucs) if aucs else None,
            })
        except Exception as e:
            print(f"  ⚠ Failed: {e}")
            ablation_results.append({'name': name, 'desc': desc,
                                     'accuracy': None, 'auc': None})
        finally:
            # Restore original config
            for k, v in original.items():
                if v is not None:
                    setattr(Config, k, v)
                else:
                    try:
                        delattr(Config, k)
                    except AttributeError:
                        pass

    # Restore the original NUM_EPOCHS before writing the result file so the
    # saved summary records the budget that WAS actually used (via
    # ablation_epochs below), not the one that will be in force for any
    # follow-on code.
    used_epochs = getattr(Config, 'NUM_EPOCHS', None)
    if ablation_epochs is not None and orig_num_epochs is not None:
        setattr(Config, 'NUM_EPOCHS', orig_num_epochs)

    out = {'seeds': seeds, 'configs': ablation_results,
           'epochs_used': used_epochs}
    path = Config.RESULTS_DIR / 'ablation_results.json'
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nAblation results → {path}")

    print(f"\n{'='*60}")
    print(f"ABLATION TABLE (mean over {len(seeds)} seeds)")
    print(f"{'─'*70}")
    print(f"{'Configuration':<40} {'Test Acc':>16} {'AUC':>10}")
    print(f"{'─'*70}")
    for r in ablation_results:
        acc = f"{r['accuracy']['mean']:.2f}±{r['accuracy']['std']:.2f}%" if r['accuracy'] else "  FAIL"
        auc = f"{r['auc']['mean']:.4f}" if r.get('auc') else "  N/A "
        print(f"{r['name']:<40} {acc:>16} {auc:>10}")
    print(f"{'='*70}")

    return out


# ── Combined summary ──────────────────────────────────────────────────────────

def _write_combined_summary(ms: dict, abl: dict, path: Path):
    """Write a single human-readable summary + JSON covering both studies."""
    lines = [
        "=" * 70,
        " HCNT -- Combined Multi-Seed + Ablation Summary",
        "=" * 70,
        "",
    ]

    if ms is not None:
        ag  = ms['aggregate']
        acc = ag['accuracy']
        auc = ag.get('auc', {})
        lines += [
            "MULTI-SEED (full HCNT configuration)",
            "-" * 70,
            f"  Seeds        : {ms['seeds']}",
            f"  Test accuracy: {acc['mean']:.2f} +/- {acc['std']:.2f} %",
            f"  AUC-ROC      : {auc.get('mean', 0):.4f} +/- {auc.get('std', 0):.4f}",
            "",
        ]

    if abl is not None:
        lines += [
            "ABLATION TABLE (one component removed per row)",
            "-" * 70,
            f"  {'Configuration':<40} {'Test Acc':>14} {'AUC':>10}",
            f"  {'-'*40} {'-'*14} {'-'*10}",
        ]
        for r in abl['configs']:
            acc = (f"{r['accuracy']['mean']:.2f}+/-{r['accuracy']['std']:.2f}%"
                   if r['accuracy'] else "   FAIL   ")
            auc = (f"{r['auc']['mean']:.4f}" if r.get('auc') else "   N/A ")
            lines.append(f"  {r['name']:<40} {acc:>14} {auc:>10}")
        lines.append("")

    lines.append("=" * 70)
    path.write_text('\n'.join(lines))
    print(f"\nCombined summary -> {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Default behaviour: run BOTH the 5-seed sweep and the ablation study in
    # a single invocation. Use --skip-ablation or --skip-seeds to opt out of
    # either half; the legacy --ablation / --ablation-only flags are kept
    # alive for backwards compatibility with older notebook cells.
    parser = argparse.ArgumentParser(
        description='Run the multi-seed sweep AND the ablation study. '
                    'Both run by default; pass --skip-ablation or --skip-seeds '
                    'to opt out of either.')
    parser.add_argument('--seeds', nargs='+', type=int,
                        default=[42, 7, 13, 21, 99],
                        help='Seeds for the full multi-seed sweep (default: '
                             '42 7 13 21 99 -- the five fixed seeds reported '
                             'in the paper).')
    parser.add_argument('--ablation-seeds', nargs='+', type=int, default=None,
                        help='Seeds for each ablation row (default: first 3 '
                             'of --seeds, matching the paper Table III).')
    parser.add_argument('--skip-seeds', action='store_true',
                        help='Skip the multi-seed sweep; run only the '
                             'ablation study.')
    parser.add_argument('--skip-ablation', action='store_true',
                        help='Skip the ablation study; run only the '
                             'multi-seed sweep.')
    parser.add_argument('--no-multi-gpu', dest='multi_gpu', action='store_false',
                        help='Force single-GPU training even when 2+ GPUs '
                             'are visible (default: use DataParallel across '
                             'all visible GPUs).')
    parser.set_defaults(multi_gpu=True)
    parser.add_argument('--ablation-epochs', type=int, default=None,
                        help='Shorter per-config budget for the ablation '
                             'phase only (multi-seed phase keeps the full '
                             'Config.NUM_EPOCHS). Recommended: 50 on Kaggle '
                             'T4 x2 to fit multi-seed + ablation inside a '
                             'single 12 h session. Ablation deltas emerge '
                             'well before full convergence, so the relative '
                             'component ordering survives the shorter '
                             'schedule.')

    # Legacy flags (do NOT remove -- previous Kaggle notebook cells rely on
    # them). --ablation now has no effect because ablation runs by default,
    # but accepting it means older cells won't error. --ablation-only is
    # equivalent to the new --skip-seeds.
    parser.add_argument('--ablation', action='store_true',
                        help=argparse.SUPPRESS)
    parser.add_argument('--ablation-only', action='store_true',
                        help=argparse.SUPPRESS)

    args = parser.parse_args()

    # Resolve the two "run this half" flags from the mix of new and legacy
    # options. Multi-seed runs unless --skip-seeds OR legacy --ablation-only.
    # Ablation runs unless --skip-ablation.
    do_seeds    = not (args.skip_seeds or args.ablation_only)
    do_ablation = not args.skip_ablation

    ms_summary  = None
    abl_summary = None

    if do_seeds:
        ms_summary = run_multi_seed(args.seeds, multi_gpu=args.multi_gpu)

    if do_ablation:
        abl_seeds = args.ablation_seeds or args.seeds[:3]
        abl_summary = run_ablation(seeds=abl_seeds,
                                   multi_gpu=args.multi_gpu,
                                   ablation_epochs=args.ablation_epochs)

    # Combined summary file, written whenever at least one half ran.
    if ms_summary is not None or abl_summary is not None:
        _write_combined_summary(
            ms_summary, abl_summary,
            Config.RESULTS_DIR / 'combined_summary.txt',
        )
