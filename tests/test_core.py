import csv
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from apcf.aaps import complement_patches, rollout_step
from apcf.config import load_config
from apcf.data import ImageCSVDataset, make_loader, check_disjoint
from apcf.mcff import fused_probabilities, classification_loss
from apcf.model import APCFNet


torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[1]


class ModelTests(unittest.TestCase):
    def test_cpu_training_has_finite_gradients_and_three_heads(self):
        cfg = load_config(ROOT / 'configs/smoke.json')
        model = APCFNet(cfg['model'])
        x = torch.randn(2, 3, 32, 32)
        y = torch.tensor([0, 1])
        output = model(x, augment=True)
        self.assertEqual(len(output.logits), 3)
        self.assertEqual(tuple(output.mixed_images.shape), tuple(x.shape))
        mixed = model(output.mixed_images)
        loss = classification_loss(output.logits, y, [1/3]*3)
        loss += classification_loss(mixed.logits, y, [1/3]*3,
                                    y[output.partners], output.lam_a, output.lam_b)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad, name)
                self.assertTrue(torch.isfinite(p.grad).all(), name)
        self.assertFalse(output.lam_a.requires_grad)
        self.assertIsNone(model(x).mixed_images)

    def test_probability_fusion_not_logit_fusion(self):
        heads = [torch.tensor([[10., 0.]]), torch.tensor([[0., 2.]]), torch.tensor([[0., 2.]])]
        probs = fused_probabilities(heads, [1/3]*3)
        self.assertEqual(probs.argmax(1).item(), 1)
        torch.testing.assert_close(probs.sum(1), torch.ones(1))

    def test_rollout_order_matches_recursive_matrix_product(self):
        a = torch.tensor([[[.8, .2], [.3, .7]]])
        b = torch.tensor([[[.1, .9], [.6, .4]]])
        eye = torch.eye(2).unsqueeze(0)
        result = rollout_step(rollout_step(None, a[:, None]), b[:, None])
        torch.testing.assert_close(result, ((b + eye)/2) @ ((a + eye)/2))
        self.assertFalse(result.requires_grad)

    def test_patch_replacement_retains_top_half_and_weights(self):
        x = torch.arange(8.).reshape(2, 1, 2, 2)
        attention = torch.tensor([[.4, .3, .2, .1], [.1, .2, .3, .4]])
        mixed, a, b, partners = complement_patches(x, attention, 2, normalize=False)
        torch.testing.assert_close(mixed, torch.tensor([[[[0., 1.], [7., 6.]]], [[[1., 0.], [6., 7.]]]]))
        torch.testing.assert_close(a, torch.tensor([.7, .7]))
        torch.testing.assert_close(b, torch.tensor([.7, .7]))
        self.assertEqual(partners.tolist(), [1, 0])
        _, a, b, _ = complement_patches(x, attention, 2, normalize=True)
        torch.testing.assert_close(a + b, torch.ones(2))

    def test_singleton_batch_is_noop_for_augmentation(self):
        x = torch.randn(1, 3, 4, 4)
        mixed, a, b, partners = complement_patches(x, torch.ones(1, 4), 2)
        torch.testing.assert_close(mixed, x)
        self.assertEqual((a.item(), b.item(), partners.item()), (1., 0., 0))


class DataTests(unittest.TestCase):
    def make_data(self, root, rows):
        manifest = root / 'samples.csv'
        with manifest.open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['', 'img_name', 'label'])
            for i, (name, label) in enumerate(rows):
                Image.new('RGB', (40, 48)).save(root / name)
                writer.writerow([i, name, label])
        return manifest

    def test_named_columns_one_based_labels_and_no_dropped_eval_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = self.make_data(root, [('a.png', 1), ('b.png', 2), ('c.png', 1)])
            ds = ImageCSVDataset(csv_path, root, 2, 1, 32, 36, False)
            loader = make_loader(ds, 2, False, 0, 1)
            self.assertEqual(sum(len(y) for _, y in loader), 3)
            self.assertEqual(ds[0][1], 0)
            self.assertEqual(tuple(ds[0][0].shape), (3, 32, 32))
            with self.assertRaisesRegex(ValueError, 'overlap'):
                check_disjoint(ds, ds)

    def test_rejects_out_of_range_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = self.make_data(root, [('a.png', 3)])
            with self.assertRaisesRegex(ValueError, 'label'):
                ImageCSVDataset(csv_path, root, 2, 1, 32, 36, False)


if __name__ == '__main__':
    unittest.main()
