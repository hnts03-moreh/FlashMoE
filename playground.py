import os
import random

import torch
import torch.distributed as dist

def has_package(name: str):
    import importlib.util
    return importlib.util.find_spec(name) is not None

def get_local_rank() -> int:
    if has_package("torch") and os.environ.get("LOCAL_RANK") is not None:
        return int(os.environ.get("LOCAL_RANK"))
    elif has_package("mpi4py"):
        import mpi4py.MPI as MPI
        import cuda.core.experimental as cuda
        return MPI.COMM_WORLD.Get_rank() % cuda.system.num_devices
    else:
        raise RuntimeError("At least one of {torch, mpi4py} must be available")

def get_rank() -> int:
    if has_package("torch") and os.environ.get("RANK") is not None:
        return int(os.environ.get("RANK"))
    elif has_package("mpi4py"):
        import mpi4py.MPI as MPI
        return MPI.COMM_WORLD.Get_rank()
    else:
        raise RuntimeError("At least one of {torch, mpi4py} must be available")

def get_shared_seed(rank_: int, device_id: int, use_torch: bool) -> int:
    shared_seed = 0
    if rank_ == 0:
        shared_seed = random.randint(1, 2**31 - 1)
    if use_torch:
        import torch.distributed as dist
        seed_tensor = torch.tensor([shared_seed], dtype=torch.int64, device=device_id)
        dist.broadcast(seed_tensor, src=0)
        return int(seed_tensor.item())
    else:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        shared_seed = comm.bcast(shared_seed, root=0)
        return shared_seed

if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    seed = get_shared_seed(rank, local_rank, True)
    print("Rank {} has {}".format(rank, seed))
    dist.destroy_process_group()