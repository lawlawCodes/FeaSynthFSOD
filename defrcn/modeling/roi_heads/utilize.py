from ...data.builtin_meta import *
from ...data.builtin_meta import _get_coco_fewshot_instances_meta

def extract_classnames(dataset):
    if 'voc' in dataset:
        if 'base' in dataset:
            classes = PASCAL_VOC_BASE_CATEGORIES[int(dataset.split('_')[-1][-1])]
        if 'novel' in dataset:
            classes = PASCAL_VOC_NOVEL_CATEGORIES[int(dataset.split('_')[-3][-1])]
        if 'with_all' in dataset:
            classes = PASCAL_VOC_ALL_CATEGORIES[int(dataset.split('_')[-3][-1])]
        dataset= ""

    if 'coco' in dataset:
        ret = _get_coco_fewshot_instances_meta()
        if 'base' in dataset:
            classes = ret["base_classes"]
        if 'novel' in dataset:
            classes = ret["novel_classes"]
        if 'with_all' in dataset:
            classes = ret["thing_classes"]
        dataset = "coco"

    return classes

import torch
import torch.distributed as dist

# @torch.no_grad()
# def concat_all_gather(tensor):
#     tensors_gather = [
#         torch.ones_like(tensor)
#         for _ in range(torch.distributed.get_world_size())
#     ]
#     torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
#     output = torch.cat(tensors_gather, dim=0)
#     return output
@torch.no_grad()
def concat_all_gather(tensor):
    world_size = dist.get_world_size()
    local_size = torch.tensor(tensor.shape[0], device=tensor.device)
    sizes = [torch.empty_like(local_size) for _ in range(world_size)]
    dist.all_gather(sizes, local_size)
    sizes = [int(s.item()) for s in sizes]
    max_size = max(sizes)
    if tensor.shape[0] < max_size:
        padding = torch.zeros((max_size - tensor.shape[0], *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device)
        tensor = torch.cat([tensor, padding], dim=0)
    gather_tensors = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(gather_tensors, tensor)
    return torch.cat([t[:sizes[i]] for i, t in enumerate(gather_tensors)], dim=0)
