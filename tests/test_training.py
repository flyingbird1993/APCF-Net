import copy
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import TensorDataset

from apcf.config import load_config
from apcf.data import make_loader
from apcf.engine import (setup_runtime, make_optimizer, train_epoch, evaluate, save_checkpoint,
                         load_checkpoint, restore_rng, metrics_from_confusion)
from apcf.model import APCFNet


class TrainingTests(unittest.TestCase):
    def test_epoch_checkpoint_resume_matches_uninterrupted_training(self):
        cfg = load_config(Path(__file__).resolve().parents[1] / 'configs/smoke.json')
        setup_runtime(cfg)
        model = APCFNet(cfg['model'])
        dataset = TensorDataset(torch.randn(4, 3, 32, 32), torch.tensor([0, 1, 2, 0]))
        loader = make_loader(dataset, 2, True, 0, 1)
        optimizer, scheduler = make_optimizer(model, cfg)
        train_epoch(model, loader, optimizer, cfg, torch.device('cpu'))
        scheduler.step()
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / '中文检查点'
            folder.mkdir()
            path = folder / 'last.pt'
            save_checkpoint(path, model, optimizer, scheduler, 0, .5, cfg, loader, {})
            train_epoch(model, loader, optimizer, cfg, torch.device('cpu'))
            expected = copy.deepcopy(model.state_dict())
            checkpoint = load_checkpoint(path)
            restored = APCFNet(cfg['model'])
            restored.load_state_dict(checkpoint['model'])
            opt, sched = make_optimizer(restored, cfg)
            opt.load_state_dict(checkpoint['optimizer'])
            sched.load_state_dict(checkpoint['scheduler'])
            restore_rng(checkpoint['rng'], loader)
            train_epoch(restored, loader, opt, cfg, torch.device('cpu'))
            for name, value in restored.state_dict().items():
                torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
            metrics = evaluate(restored, make_loader(dataset, 3, False, 0, 1), cfg, torch.device('cpu'))
            self.assertEqual(metrics['samples'], 4)

    def test_metrics_macro_average_includes_empty_classes(self):
        metrics = metrics_from_confusion(torch.tensor([[2, 1, 0], [0, 1, 0], [0, 0, 0]]))
        self.assertAlmostEqual(metrics['accuracy'], .75)
        self.assertAlmostEqual(metrics['macro_precision'], .5)
        self.assertAlmostEqual(metrics['macro_recall'], (2/3 + 1)/3)


if __name__ == '__main__':
    unittest.main()
