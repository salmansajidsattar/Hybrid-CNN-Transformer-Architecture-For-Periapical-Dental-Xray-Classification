"""
explainability.py — Grad-CAM (CNN backbone) + Attention Rollout (Transformer)
for HybridCNNTransformerV2 (HCNT).

Both methods are implemented purely as external forward/backward hooks on
existing submodules -- NO changes to model_v2.py were needed, so this file
cannot affect training in any way and works with any checkpoint already
produced by train_v2.py / run_seeds.py.

  * Grad-CAM targets `model.backbone.proj`, the (B, embed_dim, 24, 24)
    feature map the ResNet-50 backbone hands off to the Transformer. The
    predicted class logit is backpropagated through the FULL model (the
    Transformer encoder and classification head included), so the CAM
    reflects the whole hybrid model's decision, not the CNN in isolation.

  * Attention Rollout (Abnar & Zuidema, 2020) captures the post-softmax
    attention matrix from every one of the 6 TransformerBlocks via a
    forward hook on `block.attn.attn_drop`, averages heads, folds in the
    residual connection, and multiplies across all blocks to trace how the
    [CLS] token's final representation traces back to the 576 input patch
    tokens (reshaped to the 24x24 spatial grid they came from).

Usage:
    # Auto-sample from the real held-out test split (recommended for the paper):
    python -m src.explainability --checkpoint checkpoints/best_model_v2.pth \
        --n_correct 3 --n_wrong 3

    # Or explain specific images:
    python -m src.explainability --checkpoint checkpoints/best_model_v2.pth \
        --images path/to/img1.png path/to/img2.png

Output (default results/explainability/):
    <idx>_<correct|WRONG>_pred-<class>_explain.png   -- one 3-panel figure per
        sample: input | Grad-CAM overlay | Attention-Rollout overlay
    summary.json  -- predictions, true labels, and softmax probabilities
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .config import Config
from .model_v2 import create_model_v2
from .dataset import create_dataloaders, get_transforms


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denormalize(tensor_img: torch.Tensor) -> np.ndarray:
    """(3,H,W) ImageNet-normalised tensor -> (H,W,3) uint8 numpy for display."""
    img = tensor_img.detach().cpu() * IMAGENET_STD + IMAGENET_MEAN
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


# ── Grad-CAM on the CNN backbone's last spatial feature map ─────────────────

class GradCAM:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer = model.backbone.proj
        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def __call__(self, x: torch.Tensor, class_idx: int = None):
        """
        x: (1, 3, H, W) preprocessed tensor.
        Returns: cam (H, W) numpy in [0,1], predicted class index, softmax probs.
        """
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)                       # (1, num_classes)
        probs = F.softmax(logits, dim=-1)[0]
        if class_idx is None:
            class_idx = int(probs.argmax().item())
        logits[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)             # (1,C,1,1)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))  # (1,1,h,w)
        cam = F.interpolate(cam, size=x.shape[-2:], mode='bilinear', align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam -= cam.min()
        if cam.max() > 1e-8:
            cam /= cam.max()
        return cam, class_idx, probs.detach().cpu().numpy()


# ── Attention Rollout across all Transformer blocks ──────────────────────────

def attention_rollout(model, x: torch.Tensor) -> np.ndarray:
    """
    Returns the CLS-token rollout map reshaped to (24, 24) numpy in [0,1].
    Uses forward hooks on each block's `attn.attn_drop` -- the module that
    receives the post-softmax, pre-dropout attention weights as its input --
    so no changes to model_v2.py's Attention/TransformerBlock classes were
    required.
    """
    attn_maps = []

    def make_hook():
        def hook(module, inp, out):
            attn_maps.append(inp[0].detach())   # (1, heads, N, N) post-softmax
        return hook

    handles = [blk.attn.attn_drop.register_forward_hook(make_hook())
               for blk in model.blocks]
    try:
        with torch.no_grad():
            model(x)
    finally:
        for h in handles:
            h.remove()

    N = attn_maps[0].shape[-1]
    device = attn_maps[0].device
    rollout = torch.eye(N, device=device).unsqueeze(0)
    for attn in attn_maps:
        attn_heads_avg = attn.mean(dim=1)                                    # (1, N, N)
        attn_res = 0.5 * attn_heads_avg + 0.5 * torch.eye(N, device=device)
        attn_res = attn_res / attn_res.sum(dim=-1, keepdim=True)
        rollout = attn_res @ rollout

    cls_attn = rollout[0, 0, 1:]                 # CLS row, drop CLS-to-CLS entry
    grid = int(round(cls_attn.numel() ** 0.5))    # 576 -> 24
    cam = cls_attn.reshape(grid, grid).cpu().numpy()
    cam -= cam.min()
    if cam.max() > 1e-8:
        cam /= cam.max()
    return cam


def overlay_heatmap(cam_small: np.ndarray, img_size: int, base_img_uint8: np.ndarray,
                     colormap: str = 'jet', alpha: float = 0.45) -> np.ndarray:
    """cam_small: a small (e.g. 24x24) or full-size [0,1] map; resized + blended."""
    cam_img = Image.fromarray((cam_small * 255).astype(np.uint8)).resize(
        (img_size, img_size), resample=Image.BILINEAR)
    cam_resized = np.asarray(cam_img).astype(np.float32) / 255.0
    cmap = matplotlib.colormaps[colormap]
    heat = (cmap(cam_resized)[:, :, :3] * 255).astype(np.uint8)
    overlay = (alpha * heat + (1 - alpha) * base_img_uint8).astype(np.uint8)
    return overlay


# ── Driver ────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path, device):
    model = create_model_v2()
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def explain_image(model, gradcam, image_tensor, device, true_label=None, class_names=None):
    x = image_tensor.unsqueeze(0).to(device)
    cam, pred_idx, probs = gradcam(x, class_idx=None)
    rollout_map = attention_rollout(model, x)

    base = denormalize(image_tensor)
    img_size = base.shape[0]
    gradcam_overlay = overlay_heatmap(cam, img_size, base)
    rollout_overlay = overlay_heatmap(rollout_map, img_size, base)

    result = {
        'pred_idx': pred_idx,
        'pred_class': class_names[pred_idx] if class_names else str(pred_idx),
        'probs': [float(p) for p in probs],
        'true_label': int(true_label) if true_label is not None else None,
        'true_class': class_names[true_label] if (class_names is not None and true_label is not None) else None,
    }
    return base, gradcam_overlay, rollout_overlay, result


def save_panel(out_path, base, gradcam_overlay, rollout_overlay, result):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    for ax, img, title in zip(
        axes,
        [base, gradcam_overlay, rollout_overlay],
        ['Input', 'Grad-CAM (CNN backbone)', 'Attention Rollout (Transformer)']
    ):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    pred = result['pred_class']
    true = result.get('true_class')
    conf = max(result['probs']) * 100
    suptitle = f"Pred: {pred} ({conf:.1f}%)"
    if true is not None:
        tag = "correct" if pred == true else f"WRONG (true: {true})"
        suptitle += f"   |  {tag}"
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM + Attention Rollout for HCNT")
    parser.add_argument('--checkpoint', type=str,
                         default=str(Config.CHECKPOINT_DIR / 'best_model_v2.pth'))
    parser.add_argument('--n_correct', type=int, default=3,
                         help="# correctly classified test images to visualise")
    parser.add_argument('--n_wrong', type=int, default=3,
                         help="# misclassified test images to visualise")
    parser.add_argument('--images', type=str, nargs='*', default=None,
                         help="explicit image paths instead of sampling the test set")
    parser.add_argument('--out_dir', type=str, default=str(Config.RESULTS_DIR / 'explainability'))
    args = parser.parse_args()

    device = Config.DEVICE
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint, device)
    gradcam = GradCAM(model)
    summary = []

    if args.images:
        transform = get_transforms(augment=False)
        for i, p in enumerate(args.images):
            img = Image.open(p).convert('RGB')
            tensor = transform(img)
            base, gc, ro, res = explain_image(model, gradcam, tensor, device)
            fname = out_dir / f"{i:02d}_{Path(p).stem}_explain.png"
            save_panel(fname, base, gc, ro, res)
            res['image'] = str(p)
            summary.append(res)
    else:
        _, _, test_loader, class_names = create_dataloaders(Config.DATA_DIR, batch_size=1)
        correct_saved, wrong_saved = 0, 0
        for idx in range(len(test_loader.dataset)):
            if correct_saved >= args.n_correct and wrong_saved >= args.n_wrong:
                break
            img, label = test_loader.dataset[idx]
            x = img.unsqueeze(0).to(device)
            with torch.no_grad():
                pred_idx = int(model(x).argmax(dim=-1).item())
            is_correct = (pred_idx == label)
            if is_correct and correct_saved >= args.n_correct:
                continue
            if not is_correct and wrong_saved >= args.n_wrong:
                continue
            base, gc, ro, res = explain_image(model, gradcam, img, device,
                                               true_label=label, class_names=class_names)
            tag = "correct" if is_correct else "WRONG"
            fname = out_dir / f"{idx:03d}_{tag}_pred-{res['pred_class']}_explain.png"
            save_panel(fname, base, gc, ro, res)
            res['index'] = idx
            summary.append(res)
            if is_correct:
                correct_saved += 1
            else:
                wrong_saved += 1

    gradcam.remove()
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {len(summary)} explanation panel(s) to {out_dir}")


if __name__ == '__main__':
    main()
