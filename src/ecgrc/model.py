"""CNN-GRU beat classifier with a causal temporal model.

Input:  x (B, L, W) beat windows, rr (B, L, 1) causal RR features, mask (B, L) True for real beats.
Output: logits (B, L, C). Position L-1 is the newest beat; only causal context is used.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BeatEncoder(nn.Module):
    def __init__(self, window: int, width: int = 64, emb: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, width, 5, padding=2), nn.BatchNorm1d(width), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(width, width, 3, padding=1), nn.BatchNorm1d(width), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(width, emb)

    def forward(self, x):                       # x: (N, W)
        h = self.conv(x.unsqueeze(1)).squeeze(-1)
        return self.proj(h)


class CNNGRU(nn.Module):
    def __init__(self, window: int = 100, n_classes: int = 4, emb: int = 64, hidden: int = 64, n_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.window = window
        self.n_classes = n_classes
        self.encoder = BeatEncoder(window, emb=emb)
        self.rr = nn.Sequential(nn.Linear(1, 16), nn.ReLU())
        self.fuse = nn.Sequential(nn.Linear(emb + 16, emb), nn.ReLU(), nn.Dropout(dropout))
        self.gru = nn.GRU(emb, hidden, num_layers=n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0.0)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x, rr, mask):
        B, L, W = x.shape
        if W != self.window:
            raise ValueError(f"window length {W} != model window {self.window}")
        e = self.encoder(x.reshape(B * L, W)).reshape(B, L, -1)
        z = self.fuse(torch.cat([e, self.rr(rr)], dim=-1))
        z = z * mask.unsqueeze(-1).to(z.dtype)          # padding carries no information
        h, _ = self.gru(z)                               # unidirectional: causal
        return self.head(h)                              # (B, L, C)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
