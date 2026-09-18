"""Attention rollout and complementary patches (paper section 2.1)."""
import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def rollout_step(previous, attention):
    attention = attention.detach().mean(dim=1)
    eye = torch.eye(attention.shape[-1], device=attention.device, dtype=attention.dtype)
    current = (attention + eye) / 2
    current = current / current.sum(dim=-1, keepdim=True)
    return current if previous is None else current @ previous


@torch.no_grad()
def complement_patches(images, cls_attention, grid_size, normalize=False,
                       distinct_pairs=True):
    """Keep anchor top half, fill bottom half with donor top half in rank order.

    Eq. (6) and the supplied code use raw retained attention masses by default;
    they need not sum to one. normalize=True is an explicit optional variant.
    A singleton batch returns the unchanged image with weights (1, 0).
    """
    batch, channels, height, width = images.shape
    p = grid_size
    if p < 2 or p % 2 or height % p or width % p:
        raise ValueError('AAPS requires an even grid dividing both image dimensions')
    side = math.isqrt(cls_attention.shape[-1])
    if side * side != cls_attention.shape[-1]:
        raise ValueError('AAPS attention must describe a square patch grid')
    weights = F.adaptive_avg_pool2d(cls_attention.reshape(batch, 1, side, side), (p, p)).flatten(1)
    weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-12)
    scores, indices = weights.sort(dim=1, descending=True)
    if batch == 1:
        return images.detach().clone(), images.new_ones(1), images.new_zeros(1), torch.zeros(1, dtype=torch.long, device=images.device)
    if distinct_pairs:
        order = torch.randperm(batch, device=images.device)
        partners = torch.empty_like(order)
        partners[order] = order.roll(1)
    else:
        partners = torch.randperm(batch, device=images.device)
    half = p * p // 2
    ph, pw = height // p, width // p
    patches = images.reshape(batch, channels, p, ph, p, pw).permute(0, 2, 4, 1, 3, 5).reshape(batch, p*p, channels, ph, pw)
    mixed = patches.detach().clone()
    rows = torch.arange(batch, device=images.device)[:, None]
    mixed[rows, indices[:, half:]] = patches[partners[:, None], indices[partners, :half]]
    mixed = mixed.reshape(batch, p, p, channels, ph, pw).permute(0, 3, 1, 4, 2, 5).reshape_as(images)
    lam_a = scores[:, :half].sum(1)
    lam_b = scores[partners, :half].sum(1)
    if normalize:
        total = (lam_a + lam_b).clamp_min(1e-12)
        lam_a, lam_b = lam_a / total, lam_b / total
    return mixed, lam_a, lam_b, partners
