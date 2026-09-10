"""Vision Transformer for MNIST classification.

A clean, educational implementation following the original ViT paper:
  Dosovitskiy et al. "An Image is Worth 16x16 Words." ICLR 2021.

Architecture (ViT-10):
  - Image size: 28x28 (grayscale)
  - Patch size: 7x7 -> 4x4 = 16 patches
  - Embedding dim: 256
  - Depth: 10 Transformer blocks
  - Attention heads: 8
  - MLP ratio: 2.0 (hidden dim = 512)
  - Dropout: 0.1 (attention + MLP)
  - DropPath: stochastic depth (prob. increases with depth)
  - Parameters: ~5.3M
  - Expected test accuracy: > 99.5%
"""
from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class DropPath(nn.Module):
    """Stochastic depth: randomly drop residual paths during training.

    Used in modern ViT variants (DeiT, Swin) as a regularizer.
    Each sample in a batch is dropped independently.
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        # Generate random mask per sample: shape (B, 1, 1, ...) — broadcastable
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        # Scale by 1/keep_prob to maintain expected value (inverted dropout style)
        return x * mask.div(keep_prob)


def drop_path_grid(depth: int, drop_path_rate: float = 0.1) -> List[float]:
    """Compute DropPath probability for each layer with linear increase.

    The first block gets very little drop, the last block gets `drop_path_rate`.
    Mimics the scaling strategy in DeiT / Swin Transformer.
    """
    if depth == 1:
        return [0.0]
    return [drop_path_rate * i / (depth - 1) for i in range(depth)]


# ---------------------------------------------------------------------------
# Patch Embedding
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Split image into non-overlapping patches and embed them via Conv2d.

    A Conv2d with stride = kernel_size is mathematically equivalent to:
       展平 -> Linear 投影，但更简洁高效。

    Parameters
    ----------
    img_size : int
        Height and width of the square input image.
    patch_size : int
        Height and width of each square patch.
    in_ch : int
        Number of input channels (1 for grayscale, 3 for RGB).
    embed_dim : int
        Output embedding dimension per patch.
    """

    def __init__(
        self,
        img_size: int = 28,
        patch_size: int = 7,
        in_ch: int = 1,
        embed_dim: int = 256,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_ch, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C, H, W)

        Returns
        -------
        patches : (B, N, D) where N = n_patches, D = embed_dim
        """
        x = self.proj(x)          # (B, D, H/P, W/P)
        x = x.flatten(2)          # (B, D, N)
        x = x.transpose(1, 2)     # (B, N, D)
        return x


# ---------------------------------------------------------------------------
# Multi-Head Self-Attention
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """Standard scaled dot-product multi-head self-attention.

    q,k,v are projected from the input, split into multiple heads,
    attention is computed in parallel, then heads are concatenated
    and projected back to the embedding dimension.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        qkv_bias: bool = True,
        attn_p: float = 0.0,
        proj_p: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % n_heads == 0, f"dim ({dim}) must be divisible by n_heads ({n_heads})"
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim**-0.5  # 1/sqrt(d_k)

        # Single Linear to project x -> (q, k, v) concatenated
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : (B, N, D)

        Returns
        -------
        out : (B, N, D)
        attn : (B, n_heads, N, N)  — stored for visualization if needed
        """
        B, N, D = x.shape
        # (B, N, 3*D) -> (B, N, 3, n_heads, head_dim) -> (3, B, n_heads, N, head_dim)
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.n_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, n_heads, N, head_dim)

        # Scaled dot-product attention: Q·K^T / sqrt(d_k)
        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, n_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, D)  # (B, N, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn  # return attn for visualization


# ---------------------------------------------------------------------------
# Feed-Forward Network (MLP)
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Two-layer feed-forward network with GELU activation."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None,
        out_features: int,
        p: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(p)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# ---------------------------------------------------------------------------
# Transformer Encoder Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """One Transformer encoder block: Pre-LN -> Attention -> Pre-LN -> MLP.

    Each block adds a residual connection around Attention and MLP.
    Optionally includes stochastic depth (DropPath) for regularization.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        p: float = 0.0,
        attn_p: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = MultiHeadAttention(
            dim, n_heads=n_heads, qkv_bias=qkv_bias, attn_p=attn_p, proj_p=p
        )
        self.drop_path1 = DropPath(drop_path)

        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=hidden_dim,
            out_features=dim,
            p=p,
        )
        self.drop_path2 = DropPath(drop_path)

    def forward(
        self, x: torch.Tensor, return_attn: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # Attention branch
        y, attn = self.attn(self.norm1(x))
        x = x + self.drop_path1(y)

        # MLP branch
        x = x + self.drop_path2(self.mlp(self.norm2(x)))

        if return_attn:
            return x, attn
        return x


# ---------------------------------------------------------------------------
# Vision Transformer (ViT)
# ---------------------------------------------------------------------------

class ViT(nn.Module):
    """Vision Transformer for image classification.

    Architecture flow:
        Image -> PatchEmbed -> [CLS] + PosEmbed -> Transformer×N -> LayerNorm -> Linear

    Supports:
        - Feature extraction (w/o classification head)
        - Attention map extraction from all layers
        - Flexible image / patch sizes, channel counts, class numbers
    """

    def __init__(
        self,
        img_size: int = 28,
        patch_size: int = 7,
        in_ch: int = 1,
        n_classes: int = 10,
        embed_dim: int = 256,
        depth: int = 10,
        n_heads: int = 8,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        p: float = 0.1,
        attn_p: float = 0.0,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_ch=in_ch,
            embed_dim=embed_dim,
        )

        # [CLS] token: learnable vector prepended to patch sequence
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embedding: one vector per token (N_patches + 1 cls)
        n_tokens = self.patch_embed.n_patches + 1
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, embed_dim))

        self.pos_drop = nn.Dropout(p=p)

        # Transformer blocks with increasing drop path probability
        drop_path_probs = drop_path_grid(depth, drop_path_rate)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p,
                    drop_path=drop_path_probs[i],
                )
                for i in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        # Classification head
        self.head = nn.Linear(embed_dim, n_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights following the original ViT paper."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward_features(
        self, x: torch.Tensor, return_attn: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Extract features without the classification head.

        Returns the final [CLS] token representation, optionally with
        all attention maps for visualization.
        """
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)                     # (B, N, D)

        # Prepend [CLS] token
        cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat((cls_token, x), dim=1)        # (B, N+1, D)

        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        attn_maps = []
        for block in self.blocks:
            if return_attn:
                x, attn = block(x, return_attn=True)
                attn_maps.append(attn)
            else:
                x = block(x)

        x = self.norm(x)

        if return_attn:
            return x, attn_maps
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass: image -> class logits.

        Parameters
        ----------
        x : (B, C, H, W)

        Returns
        -------
        logits : (B, n_classes)
        """
        x = self.forward_features(x)
        cls_out = x[:, 0]                           # (B, D)
        out = self.head(cls_out)                    # (B, n_classes)
        return out

    def get_attention_maps(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract attention maps from all layers for visualization.

        Returns
        -------
        attn_maps : list of (B, n_heads, N+1, N+1) tensors, one per layer.
        """
        _, attn_maps = self.forward_features(x, return_attn=True)
        return attn_maps

    def model_summary(self) -> str:
        """Return a formatted summary of model architecture and parameters."""
        lines = ["=" * 60, "  ViT Model Summary", "=" * 60]
        lines.append(f"  {'Image size:':<25} {self.patch_embed.img_size} x {self.patch_embed.img_size}")
        lines.append(f"  {'Patch size:':<25} {self.patch_embed.patch_size} x {self.patch_embed.patch_size}")
        lines.append(f"  {'# patches (+ cls):':<25} {self.patch_embed.n_patches} + 1 = {self.patch_embed.n_patches + 1}")
        lines.append(f"  {'Embedding dim:':<25} {self.pos_embed.shape[-1]}")
        lines.append(f"  {'# blocks (depth):':<25} {len(self.blocks)}")
        lines.append(f"  {'# heads:':<25} {self.blocks[0].attn.n_heads}")
        lines.append(f"  {'MLP ratio:':<25} 2.0")
        lines.append(f"  {'# classes:':<25} {self.head.out_features}")
        lines.append("")

        # Parameter counts per module
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        lines.append(f"  {'Total parameters:':<25} {total:,}")
        lines.append(f"  {'Trainable parameters:':<25} {trainable:,}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = ViT(
        img_size=28,
        patch_size=7,
        in_ch=1,
        n_classes=10,
        embed_dim=256,
        depth=10,
        n_heads=8,
        mlp_ratio=2.0,
        p=0.1,
        attn_p=0.0,
        drop_path_rate=0.1,
    )

    print(model.model_summary())
    print()

    # Forward pass test
    dummy = torch.randn(4, 1, 28, 28)
    out = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape}")  # (4, 10)

    # Attention map test
    attn_maps = model.get_attention_maps(dummy)
    print(f"Attention maps: {len(attn_maps)} layers")
    print(f"  Layer 0 shape: {attn_maps[0].shape}")  # (4, 8, 17, 17)