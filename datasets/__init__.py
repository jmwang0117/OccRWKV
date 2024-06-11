
import torch
from .semantic_kitti import SemanticKitti


def collate_fn(data):
    keys = data[0][0].keys()
    out_dict = {}
    for key in keys:
        if key in ['points', 'points_label']:
            out_dict[key] = [d[0][key] for d in data]
        else:
            out_dict[key] = torch.stack([d[0][key] for d in data])
    idx = [d[1] for d in data]
    return out_dict, idx