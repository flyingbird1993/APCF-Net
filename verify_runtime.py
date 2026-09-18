"""Run real CLI smoke training/resume/evaluation; optionally test full ViT-B/16.

Generated artifacts stay under outputs/verification. No research images are used.
"""
import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pretrained', help='Optional local ViT-B_16.npz for a full 224px train step')
    parser.add_argument('--output-dir', default='outputs/verification')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    out = (root / args.output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Choose an empty --output-dir')
    out.mkdir(parents=True, exist_ok=True)
    data = out / 'synthetic'
    run = out / 'run'
    cfg = json.loads((root / 'configs/smoke.json').read_text(encoding='utf-8'))
    cfg['data'].update(root=str(data / 'images'), **{k + '_csv': str(data / (k + '.csv')) for k in ('train', 'val', 'test')})
    cfg['training']['output_dir'] = str(run)
    config_path = out / 'smoke.json'
    config_path.write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    report = {'python': platform.python_version(), 'checks': [], 'full_training': False}
    commands = [
        ['-m', 'unittest', 'discover', '-s', 'tests', '-v'],
        ['tobacco.py', '--output-dir', str(data)],
        ['train.py', '--config', str(config_path), '--check-data'],
        ['train.py', '--config', str(config_path), '--stop-after-epoch', '1'],
        ['train.py', '--config', str(config_path), '--resume', str(run / 'last.pt')],
        ['evaluate.py', '--checkpoint', str(run / 'best.pt'), '--csv', str(data / 'test.csv'),
         '--data-root', str(data / 'images'), '--device', 'cpu', '--output', str(out / 'test_metrics.json')]
    ]
    for command in commands:
        print('RUN ' + ' '.join(command), flush=True)
        completed = subprocess.run([sys.executable, '-X', 'utf8'] + command, cwd=root, capture_output=True, text=True, encoding='utf-8')
        log = completed.stdout + completed.stderr
        print(log, flush=True)
        (out / ('check_{}.log'.format(len(report['checks']) + 1))).write_text(log, encoding='utf-8')
        report['checks'].append({'entry': command[0], 'exit_code': completed.returncode})
        if completed.returncode:
            raise RuntimeError('Verification failed: ' + command[0])
    if args.pretrained:
        import torch
        from apcf.config import load_config
        from apcf.engine import setup_runtime, make_optimizer
        from apcf.mcff import classification_loss
        from apcf.model import APCFNet
        from apcf.pretrained import load_vit_npz
        cfg = load_config(root / 'configs/paper.json')
        cfg['training']['device'] = 'cpu'
        setup_runtime(cfg)
        started = time.perf_counter()
        model = APCFNet(cfg['model'])
        initialization = load_vit_npz(model, args.pretrained)
        model.train()
        inputs, labels = torch.randn(2, 3, 224, 224), torch.tensor([0, 16])
        optimizer, _ = make_optimizer(model, cfg)
        outputs = model(inputs, augment=True)
        mixed = model(outputs.mixed_images)
        loss = classification_loss(outputs.logits, labels, cfg['training']['loss_weights'])
        loss += classification_loss(mixed.logits, labels, cfg['training']['loss_weights'],
                                    labels[outputs.partners], outputs.lam_a, outputs.lam_b)
        loss.backward()
        if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError('Invalid full-model gradients')
        before = model.heads['12'].weight.detach().clone()
        optimizer.step()
        if torch.equal(before, model.heads['12'].weight):
            raise RuntimeError('Optimizer failed to change model weights')
        model.eval()
        with torch.no_grad():
            scores = model(inputs).logits
        report['full_model'] = {'torch': str(torch.__version__), 'cuda_available': torch.cuda.is_available(),
                                'device': 'cpu', 'batch_size': 2, 'input_size': 224,
                                'parameter_count': sum(p.numel() for p in model.parameters()),
                                'npz_tensors_loaded': initialization['loaded_tensors'],
                                'loss': float(loss.detach()), 'optimizer': 'Adam',
                                'output_shapes': [list(x.shape) for x in scores],
                                'seconds': round(time.perf_counter() - started, 2)}
        print(json.dumps(report['full_model'], indent=2), flush=True)
    (out / 'verification.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('Verification report: ' + str(out / 'verification.json'))


if __name__ == '__main__':
    main()
