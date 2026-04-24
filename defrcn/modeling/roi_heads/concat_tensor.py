import torch
import torch.distributed as dist

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
