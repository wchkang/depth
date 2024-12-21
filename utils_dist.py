import torch.distributed as dist


def get_dist_info():
    initialized = dist.is_available() and dist.is_initialized()
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size

def is_master():
    rank, _ = get_dist_info()
    return rank == 0


def print_at_master(str):
    if is_master():
        print(str)