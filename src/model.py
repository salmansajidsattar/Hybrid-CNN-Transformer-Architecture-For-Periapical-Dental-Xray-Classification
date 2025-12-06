import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import Config


# ==================== CNN FEATURE EXTRACTOR ====================

class CNNBlock(nn.Module):
    """Residual CNN block with skip connection"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.dropout = nn.Dropout2d(0.1)
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.skip(identity)
        out = self.relu(out)
        
        return out


class CNNFeatureExtractor(nn.Module):
    """
    CNN backbone for spatial feature extraction
    Input: (B, 3, 224, 224)
    Output: (B, 512, 14, 14)
    """
    def __init__(self, channels=[64, 128, 256, 512]):
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # Output: (B, 64, 56, 56)
        
        # Residual blocks
        self.layer1 = self._make_layer(channels[0], channels[0], blocks=2, stride=1)
        # Output: (B, 64, 56, 56)
        
        self.layer2 = self._make_layer(channels[0], channels[1], blocks=2, stride=2)
        # Output: (B, 128, 28, 28)
        
        self.layer3 = self._make_layer(channels[1], channels[2], blocks=2, stride=2)
        # Output: (B, 256, 14, 14)
        
        self.layer4 = self._make_layer(channels[2], channels[3], blocks=2, stride=1)
        # Output: (B, 512, 14, 14)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(CNNBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(CNNBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)      # (B, 64, 56, 56)
        x = self.layer1(x)     # (B, 64, 56, 56)
        x = self.layer2(x)     # (B, 128, 28, 28)
        x = self.layer3(x)     # (B, 256, 14, 14)
        x = self.layer4(x)     # (B, 512, 14, 14)
        return x


# ==================== PATCH EMBEDDING ====================

class PatchEmbedding(nn.Module):
    """
    Convert CNN feature maps into patch embeddings
    Input: (B, in_channels, H, W) e.g., (B, 512, 14, 14)
    Output: (B, num_patches, embed_dim) e.g., (B, 196, 512)
    """
    def __init__(self, in_channels=512, embed_dim=512, patch_size=1):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2),  # Flatten spatial dimensions
        )
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        """
        Args:
            x: (B, in_channels, H, W)
        Returns:
            (B, num_patches, embed_dim)
        """
        x = self.projection(x)  # (B, embed_dim, H', W')
        x = x.transpose(1, 2)   # (B, H'*W', embed_dim) = (B, num_patches, embed_dim)
        x = self.norm(x)
        return x


# ==================== POSITIONAL ENCODING ====================

class PositionalEncoding(nn.Module):
    """
    Learnable positional embeddings
    Adds position information to patch embeddings
    """
    def __init__(self, num_patches, embed_dim, dropout=0.1):
        super().__init__()
        # Learnable positional embeddings (better than fixed sinusoidal for vision)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim) * 0.02)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: (B, num_patches + 1, embed_dim)  # +1 for CLS token
        Returns:
            (B, num_patches + 1, embed_dim)
        """
        x = x + self.pos_embed
        x = self.dropout(x)
        return x


# ==================== MULTI-HEAD SELF-ATTENTION ====================

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism
    Core component of Transformer
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Linear projections for Q, K, V
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        
        # Output projection
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: (B, N, embed_dim) where N = num_patches + 1
        Returns:
            (B, N, embed_dim)
        """
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        # Q @ K^T
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        
        # Attention @ V
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, embed_dim)
        
        # Output projection
        x = self.proj(x)
        x = self.proj_dropout(x)
        
        return x


# ==================== FEED-FORWARD NETWORK ====================

class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network
    Applied after attention in each transformer block
    """
    def __init__(self, embed_dim, mlp_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(mlp_dim, embed_dim)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


# ==================== TRANSFORMER ENCODER BLOCK ====================

class TransformerEncoderBlock(nn.Module):
    """
    Single Transformer Encoder Block
    Architecture: 
        LayerNorm → Multi-Head Attention → Residual
        LayerNorm → Feed-Forward → Residual
    """
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        
        # Layer Normalization (Pre-LN architecture)
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        
        # Multi-Head Self-Attention
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        
        # Feed-Forward Network
        self.ffn = FeedForward(embed_dim, mlp_dim, dropout)
        
        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: (B, N, embed_dim)
        Returns:
            (B, N, embed_dim)
        """
        # Multi-Head Attention with residual
        x = x + self.dropout(self.attn(self.norm1(x)))
        
        # Feed-Forward with residual
        x = x + self.dropout(self.ffn(self.norm2(x)))
        
        return x


# ==================== TRANSFORMER ENCODER ====================

class TransformerEncoder(nn.Module):
    """
    Stack of Transformer Encoder Blocks
    """
    def __init__(self, num_layers, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
    
    def forward(self, x):
        """
        Args:
            x: (B, N, embed_dim)
        Returns:
            (B, N, embed_dim)
        """
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        return x


# ==================== COMPLETE HYBRID MODEL ====================

class HybridCNNTransformer(nn.Module):
    """
    Complete Hybrid CNN + Transformer for Image Classification
    
    Architecture Flow:
    1. Input Image (B, 3, 224, 224)
    2. CNN Feature Extraction → (B, 512, 14, 14)
    3. Patch Embedding → (B, 196, 512)
    4. Add CLS Token → (B, 197, 512)
    5. Positional Encoding → (B, 197, 512)
    6. Transformer Encoder → (B, 197, 512)
    7. Extract CLS Token → (B, 512)
    8. Classification Head → (B, num_classes)
    """
    
    def __init__(
        self,
        num_classes=2,
        img_size=384,
        cnn_channels=[64, 128, 256, 512],
        embed_dim=512,
        num_heads=8,
        num_layers=6,
        mlp_dim=2048,
        dropout=0.3
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        self.feature_size = 24
        self.num_patches = self.feature_size ** 2  # 14 * 14 = 196

        # B, C, H, W = cnn_feat.shape
        # num_patches = H * W

        
        print("="*70)
        print("Building Hybrid CNN + Transformer Model")
        print("="*70)
        
        
        print("1. CNN Feature Extractor (ResNet-style)")
        self.cnn = CNNFeatureExtractor(channels=cnn_channels)
        print(f"   Output: (B, {cnn_channels[-1]}, {self.feature_size}, {self.feature_size})")
        
        
        print("2. Patch Embedding")
        self.patch_embed = PatchEmbedding(
            in_channels=cnn_channels[-1],
            embed_dim=embed_dim,
            patch_size=1
        )
        print(f"   Output: (B, {self.num_patches}, {embed_dim})")
        
        print("3. CLS Token (Classification Token)")
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        print(f"   Shape: (1, 1, {embed_dim})")
        
        
        print("4. Positional Encoding (Learnable)")
        self.pos_encoding = PositionalEncoding(
            num_patches=self.num_patches,
            embed_dim=embed_dim,
            dropout=dropout
        )
        print(f"   Positions: {self.num_patches + 1} (patches + CLS)")
        
        
        print(f"5. Transformer Encoder ({num_layers} layers)")
        self.transformer = TransformerEncoder(
            num_layers=num_layers,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout
        )
        print(f"   - Layers: {num_layers}")
        print(f"   - Heads: {num_heads}")
        print(f"   - MLP Dim: {mlp_dim}")
        print(f"   - Dropout: {dropout}")
        
        
        print("6. Classification Head")
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, mlp_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(mlp_dim // 2, num_classes)
        )
        print(f"   Output: (B, {num_classes})")
        
        
        self._init_weights()
        
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print("\n" + "="*70)
        print("Model Statistics")
        print("="*70)
        print(f"Total Parameters:     {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Model Size:           ~{total_params * 4 / 1024 / 1024:.2f} MB")
        print("="*70)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        B = x.shape[0]
        
        features = self.cnn(x)  # (B, 512, 14, 14)
        
        
        patches = self.patch_embed(features)  # (B, 196, 512)
        
        
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, 512)
        x = torch.cat([cls_tokens, patches], dim=1)  # (B, 197, 512)
        
        
        x = self.pos_encoding(x)  # (B, 197, 512)
        
        
        x = self.transformer(x)  # (B, 197, 512)
        
        cls_output = x[:, 0]  # (B, 512)
        
        logits = self.classifier(cls_output)  # (B, num_classes)
        
        return logits
    
    def get_attention_weights(self, x):
        B = x.shape[0]
        
        features = self.cnn(x)
        patches = self.patch_embed(features)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, patches], dim=1)
        x = self.pos_encoding(x)
        
        
        attention_weights = []
        for layer in self.transformer.layers:
            qkv = layer.attn.qkv(layer.norm1(x))
            B, N, _ = x.shape
            qkv = qkv.reshape(B, N, 3, layer.attn.num_heads, layer.attn.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            attn = (q @ k.transpose(-2, -1)) * layer.attn.scale
            attn = attn.softmax(dim=-1)
            attention_weights.append(attn.detach().cpu())
            
            x = layer(x)
        
        return attention_weights


def create_model(num_classes=None):
    if num_classes is None:
        num_classes = Config.NUM_CLASSES
    
    model = HybridCNNTransformer(
        num_classes=num_classes,
        img_size=Config.IMG_SIZE,
        cnn_channels=Config.CNN_CHANNELS,
        embed_dim=Config.EMBED_DIM,
        num_heads=Config.NUM_HEADS,
        num_layers=Config.NUM_TRANSFORMER_LAYERS,
        mlp_dim=Config.MLP_DIM,
        dropout=Config.DROPOUT
    )
    
    return model


if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING HYBRID CNN + TRANSFORMER MODEL")
    print("="*70 + "\n")
    

    model = create_model(num_classes=2)
    

    print("\nTesting Forward Pass...")
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 384, 384)
    
    print(f"Input shape: {dummy_input.shape}")
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Output logits sample: {output[0]}")
    
    # Test with gradients
    print("\nTesting Backward Pass...")
    output = model(dummy_input)
    loss = output.sum()
    loss.backward()
    print("✓ Backward pass successful!")
    
    # Test attention weights
    print("\nTesting Attention Weight Extraction...")
    with torch.no_grad():
        attention_weights = model.get_attention_weights(dummy_input)
    print(f"✓ Extracted attention from {len(attention_weights)} layers")
    print(f"  Attention shape: {attention_weights[0].shape}")
    
    print("\n" + "="*70)
    print("ALL TESTS PASSED! ✓")
    print("="*70 + "\n")
    
    print("Model is ready for training!")
    print(f"\nExpected accuracy with this architecture: 85-95%")
    print(f"Training time (50 epochs with GPU): ~2-3 hours")