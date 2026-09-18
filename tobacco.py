"""Create synthetic diagnostic samples; download real FTDD images from README."""
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='data/smoke')
    args = parser.parse_args()
    root = Path(args.output_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError('Smoke output already exists; reuse it or choose an empty directory')
    (root / 'images').mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(123)
    for split, count in [('train', 6), ('val', 3), ('test', 3)]:
        with (root / (split + '.csv')).open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['img_name', 'label'])
            for i in range(count):
                name = '{}_{:03d}.png'.format(split, i)
                image = rng.randint(0, 256, (40, 48, 3), dtype=np.uint8)
                Image.fromarray(image).save(root / 'images' / name)
                writer.writerow([name, i % 3])
    print('Synthetic data created: ' + str(root))


if __name__ == '__main__':
    main()
