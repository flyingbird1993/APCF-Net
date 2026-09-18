"""Train with a validation split; the test split is never opened here."""
import argparse
import hashlib
import json
from pathlib import Path

from apcf.config import load_config, validate_config
from apcf.data import build_dataset, make_loader, check_disjoint
from apcf.engine import (setup_runtime, make_optimizer, train_epoch, evaluate, save_checkpoint,
                         load_checkpoint, restore_rng, write_json)
from apcf.model import APCFNet
from apcf.pretrained import load_vit_npz


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/paper.json')
    parser.add_argument('--data-root')
    parser.add_argument('--train-csv')
    parser.add_argument('--val-csv')
    parser.add_argument('--pretrained')
    parser.add_argument('--scratch', action='store_true', help='Explicitly train without ImageNet initialization')
    parser.add_argument('--resume', help='Trusted last.pt checkpoint from this package')
    parser.add_argument('--device')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--num-workers', type=int)
    parser.add_argument('--output-dir')
    parser.add_argument('--stop-after-epoch', type=int, help='Stop early without changing the configured LR schedule')
    parser.add_argument('--check-data', action='store_true', help='Validate train/val manifests and files, then exit')
    args = parser.parse_args()
    cfg = load_config(args.config)
    for arg, key in (('data_root', 'root'), ('train_csv', 'train_csv'), ('val_csv', 'val_csv')):
        if getattr(args, arg) is not None:
            cfg['data'][key] = getattr(args, arg)
    for key in ('device', 'batch_size', 'num_workers', 'output_dir', 'pretrained'):
        if getattr(args, key) is not None:
            cfg['training'][key] = getattr(args, key)
    if args.scratch:
        cfg['training']['pretrained'] = None
    if args.resume and (args.scratch or args.pretrained):
        parser.error('--resume cannot be combined with --scratch or --pretrained')
    validate_config(cfg)
    device = setup_runtime(cfg)
    train_data, val_data = build_dataset(cfg, 'train'), build_dataset(cfg, 'val')
    check_disjoint(train_data, val_data)
    provenance = {split: {'sha256': hashlib.sha256(Path(cfg['data'][split + '_csv']).read_bytes()).hexdigest(),
                          'samples': len(dataset)} for split, dataset in [('train', train_data), ('val', val_data)]}
    print(json.dumps({'data': provenance, 'device': str(device)}, indent=2))
    if args.check_data:
        return
    t = cfg['training']
    output_dir = Path(t['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / 'last.pt').exists() and not args.resume:
        raise FileExistsError('Output already contains last.pt; use --resume or a new --output-dir')
    loader = make_loader(train_data, t['batch_size'], True, t['num_workers'], t['seed'])
    val_loader = make_loader(val_data, t['batch_size'], False, t['num_workers'], t['seed'])
    checkpoint = load_checkpoint(args.resume) if args.resume else None
    if checkpoint:
        old = checkpoint['config']
        allowed_changes = {'device', 'cpu_threads', 'output_dir', 'pretrained'}
        if old['model'] != cfg['model'] or any(old['training'][k] != v for k, v in t.items() if k not in allowed_changes):
            raise ValueError('Resume requires unchanged model/training configuration (including total epochs)')
        if checkpoint['data_provenance'] != provenance or old['data']['label_base'] != cfg['data']['label_base'] or old['data']['resize_size'] != cfg['data']['resize_size']:
            raise ValueError('Resume dataset manifests or preprocessing differ from checkpoint')
    model = APCFNet(cfg['model'])
    if checkpoint:
        model.load_state_dict(checkpoint['model'])
    elif t['pretrained']:
        if not Path(t['pretrained']).is_file():
            raise FileNotFoundError('Pretrained .npz missing; supply --pretrained PATH or explicitly use --scratch')
        print(json.dumps(load_vit_npz(model, t['pretrained']), ensure_ascii=False))
    model.to(device)
    optimizer, scheduler = make_optimizer(model, cfg)
    start, best = 0, -1.
    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        restore_rng(checkpoint['rng'], loader)
        start, best = checkpoint['epoch'] + 1, checkpoint['best_accuracy']
    end = t['epochs'] if args.stop_after_epoch is None else min(t['epochs'], args.stop_after_epoch)
    if end <= start:
        raise ValueError('No epochs remain; stop-after-epoch must exceed the completed epoch count')
    write_json(output_dir / 'config.json', cfg)
    write_json(output_dir / 'data_provenance.json', provenance)
    for epoch in range(start, end):
        lr = optimizer.param_groups[0]['lr']
        train_metrics = train_epoch(model, loader, optimizer, cfg, device)
        val_metrics = evaluate(model, val_loader, cfg, device)
        scheduler.step()
        improved = val_metrics['accuracy'] > best
        best = max(best, val_metrics['accuracy'])
        record = {'epoch': epoch + 1, 'lr': lr, 'train': train_metrics, 'val': val_metrics}
        with (output_dir / 'history.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
        save_checkpoint(output_dir / 'last.pt', model, optimizer, scheduler, epoch, best, cfg, loader, provenance)
        if improved:
            save_checkpoint(output_dir / 'best.pt', model, optimizer, scheduler, epoch, best, cfg, loader, provenance)
        print('epoch={}/{} loss={:.6f} val_accuracy={:.4f} val_macro_f1={:.4f} aaps_batches={}'.format(
            epoch + 1, t['epochs'], train_metrics['loss'], val_metrics['accuracy'], val_metrics['macro_f1'], train_metrics['aaps_batches']), flush=True)


if __name__ == '__main__':
    main()
