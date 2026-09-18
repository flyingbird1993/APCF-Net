"""Google/JAX ViT .npz loader with OS-independent archive keys.

Ported from the supplied load_from routines. New CAM, layer norms and task
heads retain their initialization, matching the provided training code.
"""
import numpy as np
import torch
import torch.nn.functional as F


def load_vit_npz(model, path):
    loaded = []
    with np.load(path, allow_pickle=False) as weights, torch.no_grad():
        def copy(parameter, key, transform=None):
            if key not in weights:
                raise ValueError('Missing ViT NPZ key: ' + key)
            value = torch.from_numpy(np.array(weights[key], copy=True))
            if transform:
                value = transform(value)
            if tuple(value.shape) != tuple(parameter.shape):
                raise ValueError('Pretrained shape mismatch for ' + key + ': ' + str(tuple(value.shape)))
            parameter.copy_(value)
            loaded.append(key)

        embeddings = model.embeddings
        copy(embeddings.patch_embeddings.weight, 'embedding/kernel', lambda x: x.permute(3, 2, 0, 1))
        copy(embeddings.patch_embeddings.bias, 'embedding/bias')
        copy(embeddings.cls_token, 'cls')

        def resize_positions(value):
            if value.shape == embeddings.position_embeddings.shape:
                return value
            count = value.shape[1] - 1
            side = int(count ** .5)
            if side * side != count:
                raise ValueError('Pretrained positional grid is not square')
            new_side = model.cfg['image_size'] // model.cfg['patch_size']
            grid = value[:, 1:].reshape(1, side, side, -1).permute(0, 3, 1, 2)
            # align_corners=True matches source scipy.ndimage.zoom(order=1) endpoints.
            grid = F.interpolate(grid, size=(new_side, new_side), mode='bilinear', align_corners=True)
            return torch.cat([value[:, :1], grid.flatten(2).transpose(1, 2)], dim=1)

        copy(embeddings.position_embeddings, 'Transformer/posembed_input/pos_embedding', resize_positions)
        hidden = model.cfg['hidden_size']
        for i, block in enumerate(model.blocks):
            prefix = 'Transformer/encoderblock_{}/'.format(i)
            for name in ('query', 'key', 'value', 'out'):
                layer = getattr(block.attn, name)
                root = prefix + 'MultiHeadDotProductAttention_1/' + name + '/'
                copy(layer.weight, root + 'kernel', lambda x: x.reshape(hidden, hidden).t())
                copy(layer.bias, root + 'bias', lambda x: x.reshape(-1))
            for number, layer in enumerate((block.ffn.fc1, block.ffn.fc2)):
                root = prefix + 'MlpBlock_3/Dense_{}/'.format(number)
                copy(layer.weight, root + 'kernel', lambda x: x.t())
                copy(layer.bias, root + 'bias')
            for name, layer in (('LayerNorm_0', block.attention_norm), ('LayerNorm_2', block.ffn_norm)):
                copy(layer.weight, prefix + name + '/scale')
                copy(layer.bias, prefix + name + '/bias')
    return {'archive': str(path), 'loaded_tensors': len(loaded),
            'randomly_initialized': ['CAM modules', 'MCFF layer norms', 'classification heads']}
