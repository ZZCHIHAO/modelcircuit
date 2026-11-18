"""Independent PyTorch references for the finite Transformer templates."""
import math

import torch
from torch import nn


class AttentionReference(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.0, *, causal_length=None):
        super().__init__()
        self.heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        if causal_length is not None:
            indices = torch.arange(causal_length)
            self.register_buffer('causal_keep', (indices[:, None] >= indices[None, :])[None, None])
        else:
            self.causal_keep = None

    def forward(self, x, keep=None):
        batch, tokens, features = x.shape
        q = self.query(x).reshape(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
        k = self.key(x).reshape(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
        v = self.value(x).reshape(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
        if keep is not None:
            keep = keep[:, None, None, :]
        elif self.causal_keep is not None:
            keep = self.causal_keep
        if keep is not None:
            if not bool(keep.any(dim=-1).all()):
                raise ValueError('Every attention row needs one visible key')
            scores = scores.masked_fill(~keep, -torch.inf)
        weights = scores.softmax(dim=-1)
        context = (self.dropout(weights) @ v).transpose(1, 2).reshape(batch, tokens, features)
        return {'features': self.output(context), 'attention': weights}


class EncoderBlockReference(nn.Module):
    def __init__(self, hidden_dim, num_heads, mlp_dim, dropout, attention_dropout):
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.self_attn = AttentionReference(hidden_dim, num_heads, attention_dropout)
        self.attn_dropout = nn.Dropout(dropout)
        self.mlp_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.up = nn.Linear(hidden_dim, mlp_dim)
        self.act = nn.GELU()
        self.hidden_dropout = nn.Dropout(dropout)
        self.down = nn.Linear(mlp_dim, hidden_dim)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.attn_dropout(self.self_attn(self.attn_norm(x))['features'])
        hidden = self.hidden_dropout(self.act(self.up(self.mlp_norm(x))))
        return x + self.residual_dropout(self.down(hidden))


class EncoderReference(nn.Module):
    def __init__(self, hidden_dim, num_heads, mlp_dim, depth, dropout=0.0, attention_dropout=0.0):
        super().__init__()
        self.blocks = nn.ModuleList(EncoderBlockReference(hidden_dim, num_heads, mlp_dim, dropout, attention_dropout)
                                    for _ in range(depth))
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-6)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return {'features': self.norm(x)}


class SmallViTReference(nn.Module):
    def __init__(self, image_size=32, patch_size=4, channels=3, hidden_dim=64, num_heads=4,
                 mlp_dim=128, depth=2, num_classes=2, dropout=0.0, attention_dropout=0.0):
        super().__init__()
        tokens = (image_size // patch_size) ** 2 + 1
        self.patch = nn.Conv2d(channels, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.position = nn.Parameter(torch.empty(1, tokens, hidden_dim))
        nn.init.normal_(self.position, mean=0.0, std=0.02)
        self.encoder = EncoderReference(hidden_dim, num_heads, mlp_dim, depth, dropout, attention_dropout)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        patches = self.patch(x).flatten(2).transpose(1, 2)
        tokens = torch.cat((self.class_token.expand(x.shape[0], -1, -1), patches), dim=1)
        features = self.encoder(tokens + self.position)['features']
        return {'logits': self.head(features[:, 0, :])}
