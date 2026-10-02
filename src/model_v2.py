"""
model_v2.py — Improved Hybrid CNN-Transformer
==============================================
Supports three backbone options (set via Config.BACKBONE):
  'resnet50'       — ResNet-50, ImageNet V2 weights  (default, ~91-94%)
  'radimagen'      — ResNet-50, RadImageNet weights   (best for radiology, ~94-97%)
  'efficientnet'   — EfficientNet-V2-M, ImageNet     (strong alternative, ~93-96%)

Other improvements over model.py:
  • DropPath (Stochastic Depth) in every Transformer block
  • LayerScale for stable small-dataset training
  • Differential learning rates (backbone 10× lower than Transformer)
  • Warmup-cosine LR schedule
  • TTA at inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from pathlib import Path
from src.config import Config


# ── DropPath ─────────────────────────────────────────────────────────────────

class DropPath(nn.Module):
    """Stochastic Depth — more effective than uniform dropout in ViT blocks."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, dtype=x.dtype, device=x.device).floor_() + keep
        return x / keep * mask


# ── Backbone factory ─────────────────────────────────────────────────────────

def _remap_radimagen_resnet50(sd: dict) -> dict:
    """
    RadImageNet ResNet-50 checkpoints use Sequential-index keys
    (backbone.0/1/4/5/6/7) instead of torchvision's named keys.

    Confirmed mapping from checkpoint inspection:
        backbone.0.*  →  conv1.*        (7×7 stem conv)
        backbone.1.*  →  bn1.*          (stem BatchNorm)
        backbone.2    →  relu           (no params)
        backbone.3    →  maxpool        (no params)
        backbone.4.*  →  layer1.*
        backbone.5.*  →  layer2.*
        backbone.6.*  →  layer3.*
        backbone.7.*  →  layer4.*
        backbone.8    →  avgpool        (no params)
    """
    INDEX_MAP = {
        'backbone.0': 'conv1',
        'backbone.1': 'bn1',
        'backbone.4': 'layer1',
        'backbone.5': 'layer2',
        'backbone.6': 'layer3',
        'backbone.7': 'layer4',
    }
    remapped = {}
    skipped  = 0
    for k, v in sd.items():
        new_k = None
        for prefix, target in INDEX_MAP.items():
            if k.startswith(prefix + '.'):
                new_k = target + k[len(prefix):]
                break
        if new_k is not None:
            remapped[new_k] = v
        else:
            skipped += 1          # backbone.2/3/8 etc. — no params, skip silently
    print(f"  Key remap: {len(remapped)} mapped, {skipped} skipped (relu/pool/etc.)")
    return remapped


class ResNet50Backbone(nn.Module):
    """
    ResNet-50 up to layer3 → (B, 1024, 24, 24) for 384-px input.
    A 1×1 conv projects to embed_dim.

    Backbone options (set via radimagen_ckpt arg):
      None              → ImageNet V2 pretrained weights (torchvision)
      path/ResNet50.pt  → RadImageNet pretrained weights (radiology domain)
    """
    OUT_CHANNELS = 1024

    def __init__(self, embed_dim: int = 512, freeze_stages: int = 1,
                 radimagen_ckpt: str = None):
        super().__init__()
        base = models.resnet50(weights=None)   # architecture only, fill weights below

        if radimagen_ckpt and Path(radimagen_ckpt).exists():
            print(f"[Backbone] Loading RadImageNet weights → {Path(radimagen_ckpt).name}")
            raw   = torch.load(radimagen_ckpt, map_location='cpu')
            # The checkpoint IS the state_dict (OrderedDict), no wrapper key
            sd    = raw if isinstance(raw, dict) else raw.state_dict()
            # Strip DataParallel 'module.' prefix if present
            sd    = {k.replace('module.', ''): v for k, v in sd.items()}
            # Remap backbone.N.* → torchvision layer names
            sd    = _remap_radimagen_resnet50(sd)
            miss, unexp = base.load_state_dict(sd, strict=False)
            # Expected missing: fc.weight, fc.bias (we don't use fc)
            fc_keys = {k for k in miss if k.startswith('fc.')}
            real_miss = [k for k in miss if k not in fc_keys]
            print(f"  Missing (non-fc): {real_miss[:5] or 'none ✓'}")
            print(f"  Unexpected      : {unexp[:5] or 'none ✓'}")
            if real_miss:
                print(f"  ⚠ WARNING: {len(real_miss)} unexpected missing keys — "
                      f"check checkpoint path is ResNet50.pt not DenseNet121.pt")
        else:
            print("[Backbone] Loading ImageNet V2 ResNet-50 weights")
            base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        self.stem   = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1   # (B, 256, 96, 96) for 384-in
        self.layer2 = base.layer2   # (B, 512, 48, 48)
        self.layer3 = base.layer3   # (B, 1024, 24, 24)  ← output grid

        self.proj = nn.Sequential(
            nn.Conv2d(self.OUT_CHANNELS, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
        )
        self._freeze(freeze_stages)

    def _freeze(self, n: int):
        parts = [self.stem, self.layer1, self.layer2]
        for part in parts[:n]:
            for p in part.parameters():
                p.requires_grad_(False)

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad_(True)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.proj(x)          # (B, embed_dim, 24, 24)


class EfficientNetBackbone(nn.Module):
    """
    EfficientNet-V2-M backbone → projects to (B, embed_dim, 24, 24).
    For 384-px input EfficientNet-V2-M outputs (B, 160, 24, 24) at stage 5.
    """
    def __init__(self, embed_dim: int = 512, freeze_stages: int = 1):
        super().__init__()
        print("[Backbone] Loading EfficientNet-V2-M with ImageNet weights")
        base = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.IMAGENET1K_V1)

        # Keep features[0..5] (through stage 5, ~24×24 spatial at 384-in)
        self.features = nn.Sequential(*list(base.features.children())[:6])

        # Probe output channels
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 384, 384)
            out_ch = self.features(dummy).shape[1]
        print(f"  EfficientNet-V2-M stage-5 channels: {out_ch}")

        self.proj = nn.Sequential(
            nn.Conv2d(out_ch, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
        )
        if freeze_stages >= 1:
            for p in list(self.features.children())[0].parameters():
                p.requires_grad_(False)

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad_(True)

    def forward(self, x):
        x = self.features(x)
        return self.proj(x)          # (B, embed_dim, H, W)


def _remap_radimagen_densenet121(sd: dict) -> dict:
    """
    RadImageNet DenseNet-121 checkpoints also use backbone.N.* keys.
    DenseNet-121 torchvision naming: features.conv0, features.norm0,
    features.denseblock1/2/3/4, features.transition1/2/3, features.norm5.

    backbone.0  → features.conv0
    backbone.1  → features.norm0
    backbone.3  → features.denseblock1    (backbone.2 = relu, no params)
    backbone.4  → features.transition1
    backbone.5  → features.denseblock2
    backbone.6  → features.transition2
    backbone.7  → features.denseblock3
    backbone.8  → features.transition3
    backbone.9  → features.denseblock4
    backbone.10 → features.norm5
    """
    INDEX_MAP = {
        'backbone.0':  'features.conv0',
        'backbone.1':  'features.norm0',
        'backbone.3':  'features.denseblock1',
        'backbone.4':  'features.transition1',
        'backbone.5':  'features.denseblock2',
        'backbone.6':  'features.transition2',
        'backbone.7':  'features.denseblock3',
        'backbone.8':  'features.transition3',
        'backbone.9':  'features.denseblock4',
        'backbone.10': 'features.norm5',
    }
    remapped = {}
    for k, v in sd.items():
        new_k = None
        for prefix, target in INDEX_MAP.items():
            if k.startswith(prefix + '.') or k == prefix:
                new_k = target + k[len(prefix):]
                break
        if new_k is not None:
            remapped[new_k] = v
    print(f"  DenseNet key remap: {len(remapped)} mapped")
    return remapped


class DenseNet121Backbone(nn.Module):
    """
    DenseNet-121 backbone with RadImageNet weights.
    Outputs (B, 1024, 24, 24) for a 384-px input (after denseblock3).
    A 1×1 conv projects to embed_dim.

    Use Config.BACKBONE = 'densenet121' and Config.RADIMAGEN_CKPT pointing
    to DenseNet121.pt from your RadImageNet_pytorch folder.
    """
    OUT_CHANNELS = 1024   # after denseblock3

    def __init__(self, embed_dim: int = 512, freeze_stages: int = 1,
                 radimagen_ckpt: str = None):
        super().__init__()
        base = models.densenet121(weights=None)

        if radimagen_ckpt and Path(radimagen_ckpt).exists():
            print(f"[Backbone] Loading RadImageNet DenseNet-121 → {Path(radimagen_ckpt).name}")
            raw = torch.load(radimagen_ckpt, map_location='cpu')
            sd  = raw if isinstance(raw, dict) else raw.state_dict()
            sd  = {k.replace('module.', ''): v for k, v in sd.items()}
            sd  = _remap_radimagen_densenet121(sd)
            miss, unexp = base.load_state_dict(sd, strict=False)
            cls_miss = [k for k in miss if 'classifier' not in k]
            print(f"  Missing (non-cls): {cls_miss[:5] or 'none ✓'}")
        else:
            print("[Backbone] Loading ImageNet DenseNet-121 weights")
            base = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

        # Use features up through denseblock3 (gives 24×24 at 384-in)
        feats = base.features
        self.stem        = nn.Sequential(feats.conv0, feats.norm0, feats.relu0, feats.pool0)
        self.denseblock1 = feats.denseblock1
        self.transition1 = feats.transition1
        self.denseblock2 = feats.denseblock2
        self.transition2 = feats.transition2
        self.denseblock3 = feats.denseblock3    # output: (B, 1024, 24, 24)

        self.proj = nn.Sequential(
            nn.Conv2d(self.OUT_CHANNELS, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
        )
        self._freeze(freeze_stages)

    def _freeze(self, n: int):
        parts = [self.stem, self.denseblock1, self.transition1]
        for part in parts[:n]:
            for p in part.parameters():
                p.requires_grad_(False)

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad_(True)

    def forward(self, x):
        x = self.stem(x)
        x = self.denseblock1(x)
        x = self.transition1(x)
        x = self.denseblock2(x)
        x = self.transition2(x)
        x = self.denseblock3(x)
        return self.proj(x)     # (B, embed_dim, 24, 24)


def build_backbone(embed_dim: int = 512, freeze_stages: int = 1,
                   backbone: str = 'resnet50', radimagen_ckpt: str = None):
    """
    backbone options:
      'resnet50'    — ResNet-50, ImageNet V2 pretrained
      'radimagen'   — ResNet-50, RadImageNet pretrained  ← best for radiology
      'densenet121' — DenseNet-121, RadImageNet pretrained (set RADIMAGEN_CKPT to DenseNet121.pt)
      'efficientnet'— EfficientNet-V2-M, ImageNet pretrained
    """
    if backbone == 'efficientnet':
        return EfficientNetBackbone(embed_dim=embed_dim, freeze_stages=freeze_stages)
    elif backbone == 'densenet121':
        return DenseNet121Backbone(embed_dim=embed_dim, freeze_stages=freeze_stages,
                                   radimagen_ckpt=radimagen_ckpt)
    else:  # 'resnet50' or 'radimagen'
        return ResNet50Backbone(embed_dim=embed_dim, freeze_stages=freeze_stages,
                                radimagen_ckpt=radimagen_ckpt)


# ── Patch embedding ──────────────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    def __init__(self, embed_dim: int = 512):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):              # x: (B, embed_dim, H, W)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)   # (B, H*W, C)
        return self.norm(x)


# ── Transformer block ────────────────────────────────────────────────────────

class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, attn_drop: float = 0.0,
                 proj_drop: float = 0.0):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.qkv  = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim).permute(2,0,3,1,4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1,2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0,
                 drop: float = 0.0, attn_drop: float = 0.0,
                 drop_path: float = 0.0, layer_scale: float = 1e-5,
                 use_layer_scale: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn  = Attention(dim, heads, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(drop),
            nn.Linear(mlp_dim, dim), nn.Dropout(drop),
        )
        self.dp = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        # use_layer_scale=False genuinely ablates LayerScale (fixed unit
        # gain) rather than merely re-initialising g1/g2 near zero -- since
        # they are learnable parameters they would otherwise drift away
        # from zero during training and the ablation would not be clean.
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.g1 = nn.Parameter(layer_scale * torch.ones(dim))
            self.g2 = nn.Parameter(layer_scale * torch.ones(dim))

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.dp(self.g1 * self.attn(self.norm1(x)))
            x = x + self.dp(self.g2 * self.mlp(self.norm2(x)))
        else:
            x = x + self.dp(self.attn(self.norm1(x)))
            x = x + self.dp(self.mlp(self.norm2(x)))
        return x


# ── Full hybrid model ────────────────────────────────────────────────────────

class HybridCNNTransformerV2(nn.Module):
    """
    Hybrid CNN-Transformer with configurable pretrained backbone.
    Set Config.BACKBONE to 'resnet50', 'radimagen', or 'efficientnet'.
    Set Config.RADIMAGEN_CKPT to path of RadImageNet .pth file when using 'radimagen'.
    """
    def __init__(self, num_classes: int = 2, embed_dim: int = 512,
                 num_heads: int = 8, num_layers: int = 6, mlp_ratio: float = 4.0,
                 drop_rate: float = 0.2, attn_drop: float = 0.0,
                 drop_path_rate: float = 0.1, freeze_stages: int = 1,
                 layer_scale: float = 1e-5, backbone: str = 'resnet50',
                 radimagen_ckpt: str = None, use_transformer: bool = True,
                 use_pos_embed: bool = True, use_layer_scale: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        # Ablation toggles -- all default True, reproducing the exact
        # published \HCNT architecture. See src/run_seeds.py ABLATION_CONFIGS.
        self.use_transformer = use_transformer
        self.use_pos_embed   = use_pos_embed

        self.backbone = build_backbone(embed_dim, freeze_stages, backbone, radimagen_ckpt)

        if use_transformer:
            self.patch_embed = PatchEmbedding(embed_dim)

            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, 577, embed_dim))  # 576+1
            self.pos_drop  = nn.Dropout(drop_rate)
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
            self.blocks = nn.ModuleList([
                TransformerBlock(embed_dim, num_heads, mlp_ratio, drop_rate,
                                 attn_drop, dpr[i], layer_scale, use_layer_scale)
                for i in range(num_layers)
            ])
            self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        else:
            # Ablation: "no Transformer" -- global-average-pool the CNN
            # feature map directly, matching the paper's "GAP + linear head"
            # description of this ablation row.
            self.gap = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02)

        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Model] Params total={total:,}  trainable={trainable:,}")

    def forward(self, x):
        B = x.shape[0]
        feat = self.backbone(x)                           # (B, D, 24, 24)

        if not self.use_transformer:
            pooled = self.gap(feat).flatten(1)             # (B, D)
            return self.head(pooled)

        tokens = self.patch_embed(feat)                  # (B, 576, D)
        cls    = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)         # (B, 577, D)
        tokens = self.pos_drop(tokens + self.pos_embed) if self.use_pos_embed \
                 else self.pos_drop(tokens)
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        return self.head(tokens[:, 0])

    def unfreeze_backbone(self):
        self.backbone.unfreeze_all()
        print("[Model] Backbone fully unfrozen for fine-tuning")

    def get_backbone_params(self):
        return list(self.backbone.parameters())

    def get_transformer_params(self):
        bb_ids = {id(p) for p in self.backbone.parameters()}
        return [p for p in self.parameters() if id(p) not in bb_ids]


# ── Optimizer: differential LR ───────────────────────────────────────────────

def build_optimizer_v2(model: HybridCNNTransformerV2, base_lr: float = 3e-4,
                        backbone_lr_scale: float = 0.1, weight_decay: float = 1e-4):
    groups = [
        {"params": model.get_backbone_params(),
         "lr": base_lr * backbone_lr_scale, "weight_decay": weight_decay, "name": "backbone"},
        {"params": model.get_transformer_params(),
         "lr": base_lr, "weight_decay": weight_decay, "name": "transformer"},
    ]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999))
    print(f"[Optimizer] backbone_lr={base_lr*backbone_lr_scale:.2e}  transformer_lr={base_lr:.2e}")
    return opt


# ── Warmup-cosine LR ─────────────────────────────────────────────────────────

def build_scheduler_v2(optimizer, num_epochs: int = 100, warmup_epochs: int = 5,
                        min_lr_ratio: float = 1e-2):
    import math
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        prog = (epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * prog))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── TTA ──────────────────────────────────────────────────────────────────────

def predict_with_tta(model: nn.Module, img: torch.Tensor,
                     device: torch.device, n_aug: int = 5) -> torch.Tensor:
    """Average softmax over n_aug augmented views (original + flips + rotations)."""
    model.eval()
    views = [
        img,
        torch.flip(img, dims=[-1]),
        torch.flip(img, dims=[-2]),
        torch.rot90(img, 1, [-2, -1]),
        torch.rot90(img, 3, [-2, -1]),
    ]
    batch = torch.cat(views[:n_aug], dim=0).to(device)
    with torch.no_grad():
        logits = model(batch)
    return F.softmax(logits, dim=-1).mean(dim=0)


# ── Factory ──────────────────────────────────────────────────────────────────

def create_model_v2(num_classes: int = None) -> HybridCNNTransformerV2:
    if num_classes is None:
        num_classes = Config.NUM_CLASSES
    return HybridCNNTransformerV2(
        num_classes      = num_classes,
        embed_dim        = getattr(Config, 'EMBED_DIM', 512),
        num_heads        = getattr(Config, 'NUM_HEADS_V2', 8),
        num_layers       = getattr(Config, 'NUM_LAYERS_V2', 6),
        mlp_ratio        = getattr(Config, 'MLP_RATIO', 4.0),
        drop_rate        = getattr(Config, 'DROPOUT', 0.2),
        attn_drop        = getattr(Config, 'ATTN_DROP', 0.0),
        drop_path_rate   = getattr(Config, 'DROP_PATH_RATE', 0.1),
        freeze_stages    = getattr(Config, 'FREEZE_STAGES', 1),
        layer_scale      = getattr(Config, 'LAYER_SCALE_INIT', 1e-5),
        backbone         = getattr(Config, 'BACKBONE', 'resnet50'),
        radimagen_ckpt   = getattr(Config, 'RADIMAGEN_CKPT', None),
        use_transformer  = getattr(Config, 'USE_TRANSFORMER', True),
        use_pos_embed    = getattr(Config, 'USE_POS_EMBED', True),
        use_layer_scale  = getattr(Config, 'USE_LAYER_SCALE', True),
    )


if __name__ == '__main__':
    model = create_model_v2(2)
    x = torch.randn(2, 3, 384, 384)
    with torch.no_grad():
        print(model(x).shape)
    print("OK")
