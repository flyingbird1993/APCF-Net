"""APCF-Net; all public forwards return ModelOutput with stable fields."""
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import nn

from .aaps import complement_patches, rollout_step
from .backbone import Embeddings, Block


@dataclass
class ModelOutput:
    logits: List[torch.Tensor]
    mixed_images: Optional[torch.Tensor] = None
    lam_a: Optional[torch.Tensor] = None
    lam_b: Optional[torch.Tensor] = None
    partners: Optional[torch.Tensor] = None


class APCFNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = dict(cfg)
        self.embeddings = Embeddings(cfg)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg['num_layers'])])
        self.head_layers = cfg['head_layers']
        self.norms = nn.ModuleDict({str(i): nn.LayerNorm(cfg['hidden_size'], eps=1e-6) for i in self.head_layers})
        self.heads = nn.ModuleDict({str(i): nn.Linear(cfg['hidden_size'], cfg['num_classes']) for i in self.head_layers})

    def forward(self, images, augment=False):
        size = self.cfg['image_size']
        if tuple(images.shape[1:]) != (3, size, size):
            raise ValueError('Input must have shape [batch, 3, image_size, image_size]')
        if augment and not self.training:
            raise ValueError('AAPS is available only in training mode')
        hidden = self.embeddings(images)
        logits, rollout = {}, None
        for index, block in enumerate(self.blocks, 1):
            hidden, attention = block(hidden)
            if augment and (self.cfg['rollout_scope'] == 'all' or index in self.head_layers):
                rollout = rollout_step(rollout, attention)
            if index in self.head_layers:
                logits[index] = self.heads[str(index)](self.norms[str(index)](hidden[:, 0]))
        result = ModelOutput([logits[i] for i in self.head_layers])
        if augment:
            result.mixed_images, result.lam_a, result.lam_b, result.partners = complement_patches(
                images, rollout[:, 0, 1:], self.cfg['aaps_grid'],
                normalize=self.cfg['normalize_mix_weights'], distinct_pairs=self.cfg['distinct_pairs'])
        return result
