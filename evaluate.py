"""Evaluate a trusted checkpoint on a held-out CSV with no AAPS augmentation."""
import argparse
import json

from apcf.data import build_dataset, make_loader
from apcf.engine import setup_runtime, load_checkpoint, evaluate, write_json
from apcf.model import APCFNet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--csv', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--output', default='outputs/test_metrics.json')
    args = parser.parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    cfg = checkpoint['config']
    cfg['data']['root'], cfg['data']['test_csv'] = args.data_root, args.csv
    cfg['training']['device'] = args.device
    if args.batch_size is not None:
        cfg['training']['batch_size'] = args.batch_size
    device = setup_runtime(cfg)
    dataset = build_dataset(cfg, 'test')
    loader = make_loader(dataset, cfg['training']['batch_size'], False, cfg['training']['num_workers'], cfg['training']['seed'])
    model = APCFNet(cfg['model']).to(device)
    model.load_state_dict(checkpoint['model'])
    metrics = evaluate(model, loader, cfg, device)
    metrics.update({'checkpoint': args.checkpoint, 'checkpoint_epoch': checkpoint['epoch'] + 1,
                    'csv': args.csv, 'fusion_mode': cfg['training']['fusion_mode']})
    write_json(args.output, metrics)
    print(json.dumps({k: v for k, v in metrics.items() if k not in ('per_class', 'confusion_matrix')}, indent=2))


if __name__ == '__main__':
    main()
