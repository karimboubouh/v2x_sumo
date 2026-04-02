"""
dl/models.py — Neural network architectures for personalized DPL
===============================================================
Supports five architectures across four datasets.

Factory function:
    model = build_model(dataset, arch)

Input shapes:
    MNIST   -> (1, 28, 28)  — grayscale 10-class
    FEMNIST -> (1, 28, 28)  — grayscale 62-class
    CIFAR10 -> (3, 32, 32)  — RGB 10-class
    CIFAR100-> (3, 32, 32)  — RGB 100-class

Adapted from v2x_sim/fl_models.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Dataset metadata ──────────────────────────────────────────────────────────

DATASET_META = {
    "MNIST": {"in_ch": 1, "img": 28, "n_cls": 10},
    "FEMNIST": {"in_ch": 1, "img": 28, "n_cls": 62},
    "CIFAR10": {"in_ch": 3, "img": 32, "n_cls": 10},
    "CIFAR100": {"in_ch": 3, "img": 32, "n_cls": 100},
}


# ─────────────────────────────────────────────────────────────────────────────
# DNN — Deep fully-connected network (flattened input)
# ─────────────────────────────────────────────────────────────────────────────

class DNN(nn.Module):
    """Multi-layer perceptron: flat → 200 → n_cls.

    200 hidden units is the community standard for MNIST FL/DPL benchmarks
    (McMahan et al. 2017 FedAvg, Li et al. FedProx, etc.).
    Dropout(0.2) regularizes against overfitting to non-IID local distributions.
    """

    def __init__(self, in_ch: int, img: int, n_cls: int):
        super().__init__()
        flat = in_ch * img * img
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 200), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(200, n_cls),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# CNN — Convolutional Network
# ─────────────────────────────────────────────────────────────────────────────

class CNN(nn.Module):
    """Single-block convolutional network."""

    def __init__(self, in_ch: int, img: int, n_cls: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # img/2
        )
        reduced = (img // 2) ** 2 * 16
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(reduced, 64), nn.ReLU(),
            nn.Linear(64, n_cls),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ─────────────────────────────────────────────────────────────────────────────
# LSTM — Recurrent model treating image rows as time steps
# ─────────────────────────────────────────────────────────────────────────────

class LSTMModel(nn.Module):
    """Treats each image as a sequence of rows."""

    def __init__(self, in_ch: int, img: int, n_cls: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_ch * img,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
        )
        self.fc = nn.Linear(128, n_cls)
        self.in_ch = in_ch
        self.img = img

    def forward(self, x):
        B = x.size(0)
        x = x.permute(0, 2, 1, 3).reshape(B, self.img, self.in_ch * self.img)
        out, (h, _) = self.lstm(x)
        return self.fc(h[-1])


# ─────────────────────────────────────────────────────────────────────────────
# Transformer — Patch-based Vision Transformer (lite)
# ─────────────────────────────────────────────────────────────────────────────

class PatchEmbed(nn.Module):
    """Splits image into patches and linearly embeds each patch."""

    def __init__(self, in_ch, img, patch_size, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, patch_size, stride=patch_size)
        self.n_patches = (img // patch_size) ** 2

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class TransformerModel(nn.Module):
    """Tiny Vision Transformer (ViT-style)."""

    def __init__(self, in_ch: int, img: int, n_cls: int,
                 patch_size: int = 4, embed_dim: int = 64, n_heads: int = 4):
        super().__init__()
        self.patch_embed = PatchEmbed(in_ch, img, patch_size, embed_dim)
        n_patches = (img // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=128,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_cls)

    def forward(self, x):
        B = x.size(0)
        patches = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, patches], dim=1)
        x = x + self.pos_embed
        x = self.transformer(x)
        x = self.norm(x[:, 0])
        return self.head(x)


# ─────────────────────────────────────────────────────────────────────────────
# ResNet — Small residual network (ResNet-style)
# ─────────────────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """Standard residual block: F(x) + x with optional downsample shortcut."""

    def __init__(self, ch_in: int, ch_out: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(ch_out), nn.ReLU(),
            nn.Conv2d(ch_out, ch_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(ch_in, ch_out, 1, stride=stride, bias=False),
                nn.BatchNorm2d(ch_out),
            ) if stride != 1 or ch_in != ch_out else nn.Identity()
        )

    def forward(self, x):
        return F.relu(self.conv(x) + self.shortcut(x))


class ResNet(nn.Module):
    """Small ResNet: stem -> 3 stages -> global avg pool -> FC."""

    def __init__(self, in_ch: int, img: int, n_cls: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(),
        )
        self.layer1 = ResBlock(32, 64, stride=2)
        self.layer2 = ResBlock(64, 128, stride=2)
        self.layer3 = ResBlock(128, 128, stride=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, n_cls)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(dataset: str, arch: str) -> nn.Module:
    """Instantiate the requested model architecture for the given dataset."""
    meta = DATASET_META[dataset]
    in_ch = meta["in_ch"]
    img = meta["img"]
    n_cls = meta["n_cls"]

    dispatch = {
        "DNN": DNN,
        "CNN": CNN,
        "LSTM": LSTMModel,
        "Transformer": TransformerModel,
        "ResNet": ResNet,
    }
    if arch not in dispatch:
        raise ValueError(f"Unknown arch '{arch}'. Choose from {list(dispatch)}")
    return dispatch[arch](in_ch, img, n_cls)
