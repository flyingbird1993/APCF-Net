"""Independent heads, deep supervision, and probability fusion (section 2.2)."""
import torch
import torch.nn.functional as F


def fused_probabilities(logits, weights, mode='probabilities'):
    if mode == 'logits':
        return sum(w * x for w, x in zip(weights, logits)).softmax(-1)
    return sum(w * x.softmax(-1) for w, x in zip(weights, logits))


def classification_loss(logits, labels, weights, donor_labels=None, lam_a=None, lam_b=None):
    loss = logits[0].new_zeros(())
    for weight, scores in zip(weights, logits):
        ce = F.cross_entropy(scores, labels, reduction='none')
        if donor_labels is not None:
            ce = lam_a * ce + lam_b * F.cross_entropy(scores, donor_labels, reduction='none')
        loss = loss + weight * ce.mean()
    return loss
