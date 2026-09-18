"""CSV data loading and PIL transforms; no torchvision dependency is required."""
import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader


class ImageCSVDataset(Dataset):
    def __init__(self, csv_path, root, num_classes, label_base, image_size, resize_size, training):
        self.root = Path(root).resolve()
        self.image_size, self.resize_size, self.training = image_size, resize_size, training
        self.samples, seen = [], set()
        with Path(csv_path).open(encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            if not {'img_name', 'label'}.issubset(reader.fieldnames or []):
                raise ValueError('CSV requires named img_name and label columns: ' + str(csv_path))
            for row in reader:
                name = row['img_name'].replace('\\', '/')
                path = (self.root / name).resolve()
                if Path(name).is_absolute() or self.root not in path.parents:
                    raise ValueError('img_name must be relative and stay inside data root: ' + name)
                try:
                    label = int(row['label']) - label_base
                except ValueError:
                    raise ValueError('Invalid integer label: ' + row['label'])
                if not 0 <= label < num_classes:
                    raise ValueError('label out of range: ' + row['label'])
                if path in seen:
                    raise ValueError('Duplicate image in CSV: ' + name)
                if not path.is_file():
                    raise FileNotFoundError('Image missing: ' + str(path))
                seen.add(path)
                self.samples.append((path, label))
        if not self.samples:
            raise ValueError('Empty dataset: ' + str(csv_path))
        self.paths = seen

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as source:
            image = source.convert('RGB')
        width, height = image.size
        scale = self.resize_size / min(width, height)
        image = image.resize((int(width * scale), int(height * scale)), Image.Resampling.BILINEAR)
        width, height = image.size
        size = self.image_size
        if self.training:
            left, top = random.randint(0, width - size), random.randint(0, height - size)
        else:
            left, top = round((width - size) / 2), round((height - size) / 2)
        image = image.crop((left, top, left + size, top + size))
        if self.training and random.random() < .5:
            image = ImageOps.mirror(image)
        tensor = torch.from_numpy(np.array(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255
        mean = tensor.new_tensor([.485, .456, .406])[:, None, None]
        std = tensor.new_tensor([.229, .224, .225])[:, None, None]
        return (tensor - mean) / std, label


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def make_loader(dataset, batch_size, training, num_workers, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=training, drop_last=False,
                      num_workers=num_workers, worker_init_fn=seed_worker, generator=generator)


def check_disjoint(*datasets):
    for i, left in enumerate(datasets):
        for right in datasets[i+1:]:
            overlap = left.paths & right.paths
            if overlap:
                raise ValueError('Dataset overlap: ' + str(next(iter(overlap))))


def build_dataset(cfg, split):
    d, m = cfg['data'], cfg['model']
    return ImageCSVDataset(d[split + '_csv'], d['root'], m['num_classes'], d['label_base'],
                           m['image_size'], d['resize_size'], split == 'train')
