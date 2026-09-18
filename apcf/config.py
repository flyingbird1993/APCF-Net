"""JSON configuration without import-time I/O or cloud platform dependencies."""
import json
from pathlib import Path


def validate_config(cfg):
    m, t = cfg['model'], cfg['training']
    for key in ('image_size', 'patch_size', 'hidden_size', 'mlp_dim', 'num_heads', 'num_layers', 'num_classes', 'cam_reduction'):
        if not isinstance(m[key], int) or m[key] <= 0:
            raise ValueError('model.' + key + ' must be a positive integer')
    if m['image_size'] % m['patch_size'] or m['hidden_size'] % m['num_heads']:
        raise ValueError('image/patch and hidden/head sizes must divide evenly')
    if m['aaps_grid'] < 2 or m['aaps_grid'] % 2 or m['image_size'] % m['aaps_grid']:
        raise ValueError('aaps_grid must be even and divide image_size')
    layers = m['head_layers']
    if not layers or layers != sorted(set(layers)) or min(layers) < 1 or max(layers) > m['num_layers']:
        raise ValueError('head_layers must be unique ascending layer numbers')
    for key in ('loss_weights', 'fusion_weights'):
        values = t[key]
        if len(values) != len(layers) or any(w < 0 for w in values) or sum(values) <= 0:
            raise ValueError(key + ' must match head_layers and have positive total')
    if abs(sum(t['fusion_weights']) - 1) > 1e-6:
        raise ValueError('fusion_weights must sum to one')
    if m['rollout_scope'] not in ('all', 'heads') or t['fusion_mode'] not in ('probabilities', 'logits'):
        raise ValueError('Unknown rollout_scope or fusion_mode')
    if not 0 <= t['aaps_probability'] <= 1:
        raise ValueError('aaps_probability must be in [0, 1]')
    if cfg['data']['resize_size'] < m['image_size']:
        raise ValueError('resize_size cannot be smaller than image_size')
    if t['batch_size'] < 1 or t['epochs'] < 1 or t['lr'] <= 0 or t['num_workers'] < 0:
        raise ValueError('Invalid training sizes or learning rate')
    if cfg['data']['label_base'] not in (0, 1):
        raise ValueError('label_base must be 0 or 1')
    return cfg


def load_config(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        return validate_config(json.load(f))
