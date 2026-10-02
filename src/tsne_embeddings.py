"""
tsne_embeddings.py -- Fig. 5 ("Feature Space Analysis") generator.
====================================================================
Produces a t-SNE projection of the REAL [CLS]-token embeddings that the
TRAINED HybridCNNTransformerV2 (HCNT) extracts from all 929 images,
coloured by true class (Periapical / Non-Periapical). This is exactly
what main.tex's Fig.~\ref{fig:tsne} caption claims: "t-SNE projection of
the [CLS]-token embeddings extracted from all 929 images after HCNT
training."

Why this is a NEW file rather than a fix to src/auto_cluster_periapical.py:
    auto_cluster_periapical.py is a different, still-useful tool: an
    UNSUPERVISED raw-dataset auto-labeller that sorts a folder of not-yet-
    classified X-rays into periapical/ and non_periapical/ subfolders
    using a generic, untrained ImageNet ResNet-50 feature extractor plus
    heuristic aspect-ratio/intensity/width rules. It never loads a
    trained HCNT checkpoint, is not one of the paper's six modular
    pipeline components, and its own "clustering_visualization.png" is a
    coincidentally-identical filename, not a real feature-space
    visualisation of the trained classifier. Patching trained-model
    embedding extraction into that class would conflate two unrelated
    tools, so it is left exactly as-is (it may still be useful for
    organising newly collected, unlabeled X-rays), and this script is
    the one that actually feeds figures/clustering_visualization.png.

Usage:
    python -m src.tsne_embeddings --checkpoint checkpoints/best_model_v2.pth
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # headless-safe: no GUI window on a training server/Kaggle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from .config import Config
from .model_v2 import create_model_v2
from .dataset import DentalXrayDataset, load_dataset, get_transforms


def load_model(checkpoint_path, device):
    model = create_model_v2()
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def extract_cls_embeddings(model, loader, device):
    """
    Runs every image in `loader` through the trained HCNT's backbone +
    Transformer encoder, replicating HybridCNNTransformerV2.forward() up
    to (and including) `self.norm(tokens)[:, 0]` -- the exact [CLS]-token
    vector the model's own classification head consumes -- WITHOUT the
    final nn.Linear head, since we want the embedding, not the logits.

    Requires a checkpoint trained with the Transformer enabled: the "no
    Transformer" ablation (USE_TRANSFORMER=False) has no [CLS] token.
    """
    if not getattr(model, 'use_transformer', True):
        raise RuntimeError(
            "This checkpoint was trained with USE_TRANSFORMER=False (the "
            "'no Transformer' ablation) -- there is no [CLS] token to "
            "extract embeddings from. Point --checkpoint at a full-HCNT "
            "run (e.g. checkpoints/best_model_v2.pth) for Fig. 5.")

    embeddings, labels = [], []
    for imgs, y in loader:
        imgs = imgs.to(device)
        feat = model.backbone(imgs)                          # (B, D, 24, 24)
        tokens = model.patch_embed(feat)                     # (B, 576, D)
        cls = model.cls_token.expand(imgs.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)              # (B, 577, D)
        tokens = model.pos_drop(tokens + model.pos_embed) if model.use_pos_embed \
                 else model.pos_drop(tokens)
        for blk in model.blocks:
            tokens = blk(tokens)
        tokens = model.norm(tokens)
        cls_embed = tokens[:, 0]                              # (B, D) -- the real [CLS] embedding
        embeddings.append(cls_embed.cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(embeddings, axis=0), np.concatenate(labels, axis=0)


def plot_tsne(embeddings, labels, class_names, save_path, seed=42):
    n = embeddings.shape[0]
    tsne = TSNE(n_components=2, random_state=seed,
                perplexity=min(30, max(5, n // 4)), init='pca')
    emb2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    colors = ['red', 'blue']
    for idx, name in enumerate(class_names):
        mask = labels == idx
        plt.scatter(emb2d[mask, 0], emb2d[mask, 1],
                    label=f"{name} (n={int(mask.sum())})",
                    alpha=0.65, s=40, c=colors[idx % len(colors)])
    plt.xlabel('t-SNE Component 1', fontsize=12, fontweight='bold')
    plt.ylabel('t-SNE Component 2', fontsize=12, fontweight='bold')
    plt.title('t-SNE of Trained HCNT [CLS]-Token Embeddings (929 images)',
              fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved t-SNE plot -> {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Fig. 5: t-SNE of the trained HCNT's real [CLS] embeddings "
                     "over all 929 images")
    parser.add_argument('--checkpoint', type=str,
                         default=str(Config.CHECKPOINT_DIR / 'best_model_v2.pth'))
    parser.add_argument('--out', type=str,
                         default=str(Config.RESULTS_DIR / 'clustering_visualization.png'))
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = Config.DEVICE
    model = load_model(args.checkpoint, device)

    # All 929 images, no train/val/test split -- matches the paper's "all
    # 929 images" wording exactly -- with the deterministic (non-augmented)
    # eval transform, so embeddings are not contaminated by random augmentation.
    image_paths, labels, class_names = load_dataset(Config.DATA_DIR)
    dataset = DentalXrayDataset(image_paths, labels, transform=get_transforms(augment=False))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                         num_workers=getattr(Config, 'NUM_WORKERS', 0),
                         pin_memory=getattr(Config, 'PIN_MEMORY', False))

    embeddings, labels_arr = extract_cls_embeddings(model, loader, device)
    print(f"Extracted {embeddings.shape[0]} [CLS] embeddings of dim {embeddings.shape[1]}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_tsne(embeddings, labels_arr, class_names, out_path, seed=args.seed)

    np.savez(out_path.with_suffix('.npz'), embeddings=embeddings, labels=labels_arr)
    print(f"Saved raw embeddings -> {out_path.with_suffix('.npz')}")


if __name__ == '__main__':
    main()
