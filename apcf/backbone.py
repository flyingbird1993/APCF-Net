"""ViT blocks adapted from CA_CTI_APC.py; CAM follows MSA before residual."""
import math

import torch
from torch import nn

from .cam import CoordAtt


class Embeddings(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, p, size = cfg['hidden_size'], cfg['patch_size'], cfg['image_size']
        self.patch_embeddings = nn.Conv2d(3, d, kernel_size=p, stride=p)
        self.position_embeddings = nn.Parameter(torch.zeros(1, (size // p)**2 + 1, d))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        self.dropout = nn.Dropout(cfg['dropout'])

    def forward(self, x):
        patches = self.patch_embeddings(x).flatten(2).transpose(1, 2)
        tokens = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), patches], dim=1)
        return self.dropout(tokens + self.position_embeddings)


class Mlp(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg['hidden_size'], cfg['mlp_dim'])
        self.fc2 = nn.Linear(cfg['mlp_dim'], cfg['hidden_size'])
        self.dropout = nn.Dropout(cfg['dropout'])
        for layer in (self.fc1, self.fc2):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.normal_(layer.bias, std=1e-6)

    def forward(self, x):
        return self.dropout(self.fc2(self.dropout(torch.nn.functional.gelu(self.fc1(x)))))


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg['hidden_size']
        self.heads = cfg['num_heads']
        self.head_dim = d // self.heads
        self.query, self.key, self.value = (nn.Linear(d, d) for _ in range(3))
        self.out = nn.Linear(d, d)
        self.attn_dropout = nn.Dropout(cfg['attention_dropout'])
        self.proj_dropout = nn.Dropout(cfg['attention_dropout'])

    def forward(self, x):
        batch, count, d = x.shape
        q, k, v = [layer(x).reshape(batch, count, self.heads, self.head_dim).transpose(1, 2)
                   for layer in (self.query, self.key, self.value)]
        weights = ((q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)).softmax(-1)
        context = (self.attn_dropout(weights) @ v).transpose(1, 2).reshape(batch, count, d)
        return self.proj_dropout(self.out(context)), weights


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg['hidden_size']
        self.attention_norm = nn.LayerNorm(d, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(d, eps=1e-6)
        self.ffn = Mlp(cfg)
        self.attn = Attention(cfg)
        self.coord_att = CoordAtt(d, cfg['cam_reduction']) if cfg['use_cam'] else None

    def forward(self, x):
        update, attention = self.attn(self.attention_norm(x))
        if self.coord_att is not None:
            batch, count, channels = update.shape
            side = math.isqrt(count - 1)
            patches = update[:, 1:].transpose(1, 2).reshape(batch, channels, side, side)
            patches = self.coord_att(patches).flatten(2).transpose(1, 2)
            update = torch.cat([update[:, :1], patches], dim=1)
        x = x + update
        return x + self.ffn(self.ffn_norm(x)), attention
