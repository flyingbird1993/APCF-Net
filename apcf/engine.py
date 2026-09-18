"""Training, validation metrics, and epoch-boundary checkpoint recovery."""
import json
import platform
import random
from pathlib import Path

import numpy as np
import torch

from .mcff import classification_loss, fused_probabilities


def setup_runtime(cfg):
    t = cfg['training']
    random.seed(t['seed'])
    np.random.seed(t['seed'])
    torch.manual_seed(t['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(t['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(t['cpu_threads'])
    name = t['device']
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if name.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use --device cpu or install CUDA-enabled PyTorch')
    return torch.device(name)


def make_optimizer(model, cfg):
    t = cfg['training']
    if t['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=t['lr'], weight_decay=t['weight_decay'])
    elif t['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=t['lr'], momentum=t['momentum'], weight_decay=t['weight_decay'])
    else:
        raise ValueError('Unknown optimizer: ' + t['optimizer'])
    if t['scheduler'] == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, t['step_size'], t['gamma'])
    elif t['scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t['epochs'])
    else:
        raise ValueError('Unknown scheduler: ' + t['scheduler'])
    return optimizer, scheduler


def train_epoch(model, loader, optimizer, cfg, device):
    model.train()
    t = cfg['training']
    total, samples, augmented = 0., 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        use_aaps = len(labels) > 1 and random.random() < t['aaps_probability']
        optimizer.zero_grad(set_to_none=True)
        original = model(images, augment=use_aaps)
        loss = classification_loss(original.logits, labels, t['loss_weights'])
        if use_aaps:
            mixed = model(original.mixed_images)
            loss = loss + t['mixed_loss_weight'] * classification_loss(
                mixed.logits, labels, t['loss_weights'], labels[original.partners], original.lam_a, original.lam_b)
            augmented += 1
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite training loss')
        loss.backward()
        optimizer.step()
        total += float(loss.detach()) * len(labels)
        samples += len(labels)
    return {'loss': total / samples, 'samples': samples, 'aaps_batches': augmented}


def metrics_from_confusion(matrix):
    cm = matrix.double()
    true_positive = cm.diag()
    precision = true_positive / cm.sum(0).clamp_min(1)
    recall = true_positive / cm.sum(1).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    return {'accuracy': float(true_positive.sum() / cm.sum().clamp_min(1)),
            'macro_precision': float(precision.mean()), 'macro_recall': float(recall.mean()),
            'macro_f1': float(f1.mean()), 'samples': int(cm.sum()),
            'per_class': [{'label': i, 'precision': float(precision[i]), 'recall': float(recall[i]),
                           'f1': float(f1[i]), 'support': int(cm[i].sum())} for i in range(len(cm))],
            'confusion_matrix': matrix.tolist()}


@torch.no_grad()
def evaluate(model, loader, cfg, device):
    model.eval()
    classes, t = cfg['model']['num_classes'], cfg['training']
    cm = torch.zeros(classes, classes, dtype=torch.long)
    total, samples = 0., 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        loss = classification_loss(output.logits, labels, t['loss_weights'])
        prediction = fused_probabilities(output.logits, t['fusion_weights'], t['fusion_mode']).argmax(1)
        cm += torch.bincount((labels * classes + prediction).cpu(), minlength=classes**2).reshape(classes, classes)
        total += float(loss) * len(labels)
        samples += len(labels)
    metrics = metrics_from_confusion(cm)
    metrics['loss'] = total / samples
    return metrics


def rng_state(loader):
    state = {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
             'loader': loader.generator.get_state()}
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state, loader):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    loader.generator.set_state(state['loader'].cpu())
    if 'cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in state['cuda']])


def save_checkpoint(path, model, optimizer, scheduler, epoch, best, cfg, loader, provenance):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    state = {'format_version': 1, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'epoch': epoch, 'best_accuracy': best,
                'config': cfg, 'rng': rng_state(loader), 'data_provenance': provenance,
                'runtime': {'python': platform.python_version(), 'torch': str(torch.__version__),
                            'cuda': torch.version.cuda}}
    # Python handles Unicode paths correctly on older Windows PyTorch builds.
    with temporary.open('wb') as stream:
        torch.save(state, stream)
    temporary.replace(path)


def load_checkpoint(path):
    # Full training checkpoints contain RNG state. Load only trusted local files.
    with Path(path).open('rb') as stream:
        return torch.load(stream, map_location='cpu', weights_only=False)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
