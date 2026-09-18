# Code provenance and third-party notices

The immediate source of this package is the author-supplied `J_fine-grained` directory. File hashes and extraction decisions are recorded in `metadata/source_inventory.json`. This package is an adapted and reorganized implementation, not a verbatim archive of all experimental files.

## Fine-grained ViT source

The supplied README carries the title “A Vision Transformer for Fine-grained Classification by Reducing Noise and Enhancing Discrimnative Information”. A public repository with the same README and structure is [Hector0426/fine-grained-image-classification-with-vit](https://github.com/Hector0426/fine-grained-image-classification-with-vit). This is a traceable likely upstream reference; an exact commit-to-file lineage has not been established. Its visible root listing did not include a license file when checked on 2026-09-18. This package does not declare that source to be MIT/Apache licensed or replace its authorship.

## Coordinate Attention

The supplied `coordatt.py` corresponds to the Coordinate Attention module by Qibin Hou, Daquan Zhou and Jiashi Feng, CVPR 2021. See [official implementation](https://github.com/houqb/CoordAttention) and [upstream MIT license](https://github.com/houqb/CoordAttention/blob/main/LICENSE). Its copyright and permission notice are retained in `licenses/CoordAttention-MIT.txt`. The module is adapted in `apcf/cam.py`.

## ViT and pretrained weights

ViT/JAX archive structure and backbone loading follow the supplied implementation and [Google Research Vision Transformer](https://github.com/google-research/vision_transformer). Weight files are not redistributed in this package; obtain them from their original source and retain applicable terms.

## Release scope

No overall project license or author identity has been assigned during code extraction. Before choosing a repository license, the code owner should establish the permissions of the supplied/upstream implementation. A citation by itself is not a substitute for a code license. No unpublished PDF, dataset image or trained weight is included.
