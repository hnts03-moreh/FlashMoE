# communication backend
from __future__ import annotations

IS_INITIALIZED = False

def has_package(name: str):
    import importlib.util
    return importlib.util.find_spec(name) is not None


def _is_hip_platform() -> bool:
    """Check if we are running on a HIP/ROCm platform."""
    from .jit import is_hip
    return is_hip()


# ---------------------------------------------------------------------------
# GPU runtime helpers -- thin wrappers so the rest of cb.py stays clean.
# On CUDA: uses ``cuda.core`` (cuda-python).
# On HIP:  uses ``hip`` (hip-python) with a compatibility shim.
# ---------------------------------------------------------------------------

class _GPURuntime:
    """Minimal abstraction over cuda.core / hip-python device & stream APIs."""

    def __init__(self):
        self._hip = _is_hip_platform()

    # -- device count -------------------------------------------------------
    @property
    def num_devices(self) -> int:
        if self._hip:
            from hip import hip as _hip
            err, count = _hip.hipGetDeviceCount()
            if err != 0:
                raise RuntimeError(f"hipGetDeviceCount failed with error {err}")
            return count
        else:
            import cuda.core as cuda
            return cuda.system.num_devices

    # -- Device wrapper -----------------------------------------------------
    class _Device:
        def __init__(self, device_id: int, is_hip: bool):
            self._id = device_id
            self._hip = is_hip

        def set_current(self):
            if self._hip:
                from hip import hip as _hip
                err = _hip.hipSetDevice(self._id)
                if err != 0:
                    raise RuntimeError(f"hipSetDevice({self._id}) failed: {err}")
            else:
                import cuda.core as cuda
                cuda.Device(self._id).set_current()

        def sync(self):
            if self._hip:
                from hip import hip as _hip
                err = _hip.hipDeviceSynchronize()
                if err != 0:
                    raise RuntimeError(f"hipDeviceSynchronize failed: {err}")
            else:
                import cuda.core as cuda
                cuda.Device(self._id).sync()

    def Device(self, device_id: int):  # noqa: N802  -- match cuda.core API
        return self._Device(device_id, self._hip)

    # -- Stream from raw handle --------------------------------------------
    class _StreamHandle:
        """Wraps a raw stream pointer so that nvshmem / rocshmem can use it."""
        def __init__(self, handle: int, is_hip: bool):
            self._handle = handle
            self._hip = is_hip

        @property
        def handle(self):
            return self._handle

    def stream_from_handle(self, handle: int):
        if self._hip:
            return self._StreamHandle(handle, True)
        else:
            import cuda.core as cuda
            return cuda.Stream.from_handle(handle)


# Module-level singleton
_gpu = None

def _get_gpu() -> _GPURuntime:
    global _gpu
    if _gpu is None:
        _gpu = _GPURuntime()
    return _gpu


# ---------------------------------------------------------------------------
# NVSHMEM / ROCSHMEM abstraction
# ---------------------------------------------------------------------------

class _CommBackend:
    """Wraps nvshmem (CUDA) or rocshmem (HIP) Python bindings.

    On HIP, if rocshmem Python bindings are not available, the methods raise
    ``NotImplementedError`` with an installation hint.
    """

    def __init__(self):
        self._hip = _is_hip_platform()
        self._mod = None  # lazy

    def _ensure_loaded(self):
        if self._mod is not None:
            return
        if self._hip:
            if has_package("rocshmem"):
                import rocshmem.core as _mod
                self._mod = _mod
            else:
                raise NotImplementedError(
                    "ROCSHMEM Python bindings are required for distributed FlashMoE on ROCm "
                    "but could not be found.  Please install rocshmem-python or set up "
                    "the ROCSHMEM environment.  Single-GPU operation does not require "
                    "this package."
                )
        else:
            if has_package("nvshmem"):
                import nvshmem.core as _mod
                self._mod = _mod
            else:
                raise ImportError(
                    "nvshmem Python bindings (nvshmem4py) are required for distributed "
                    "FlashMoE on CUDA.  Install via: pip install nvshmem4py-cu12  (or cu13)"
                )

    # -- thin forwarding API ------------------------------------------------
    def init_status(self):
        self._ensure_loaded()
        return self._mod.init_status()

    @property
    def InitStatus(self):
        self._ensure_loaded()
        return self._mod.InitStatus

    def init(self, **kwargs):
        self._ensure_loaded()
        return self._mod.init(**kwargs)

    def get_unique_id(self, **kwargs):
        self._ensure_loaded()
        return self._mod.get_unique_id(**kwargs)

    def sync_all(self, **kwargs):
        self._ensure_loaded()
        return self._mod.sync_all(**kwargs)

    def my_pe(self):
        self._ensure_loaded()
        return self._mod.my_pe()

    def n_pes(self):
        self._ensure_loaded()
        return self._mod.n_pes()

    def finalize(self):
        self._ensure_loaded()
        return self._mod.finalize()

    def team_n_pes(self, team):
        self._ensure_loaded()
        return self._mod.team_n_pes(team)

    @property
    def Teams(self):
        self._ensure_loaded()
        return self._mod.Teams


_comm = None

def _get_comm() -> _CommBackend:
    global _comm
    if _comm is None:
        _comm = _CommBackend()
    return _comm


# ---------------------------------------------------------------------------
# Public API (unchanged signatures)
# ---------------------------------------------------------------------------

def get_local_rank() -> int:
    import os
    gpu = _get_gpu()
    if has_package("torch") and os.environ.get("LOCAL_RANK") is not None:
        return int(os.environ.get("LOCAL_RANK"))
    elif has_package("mpi4py"):
        import mpi4py.MPI as MPI
        return MPI.COMM_WORLD.Get_rank() % gpu.num_devices
    else:
        raise RuntimeError("At least one of {torch, mpi4py} must be available")


def initialize() -> None:
    gpu = _get_gpu()
    comm = _get_comm()
    global IS_INITIALIZED
    if comm.init_status() == comm.InitStatus.STATUS_IS_INITIALIZED:
        IS_INITIALIZED = True
        return
    # attempt to initialize
    initialized = False
    dev = gpu.Device(get_local_rank())
    dev.set_current()
    if has_package("torch"):
        import torch.distributed as dist
        if dist.is_initialized():
            num_ranks = dist.get_world_size()
            rank_id = dist.get_rank()
            uniqueid = comm.get_unique_id(empty=True)
            src_rank = 0
            if rank_id == src_rank:
                # Rank 0 gets a real uniqueid
                uniqueid = comm.get_unique_id()
                broadcast_objects = [uniqueid]
            else:
                broadcast_objects = [None]
            dist.broadcast_object_list(broadcast_objects, src=src_rank)
            dist.barrier()
            comm.init(device=dev, uid=broadcast_objects[0], rank=rank_id, nranks=num_ranks,
                      initializer_method="uid")
            initialized = True
    if not initialized and has_package("mpi4py"):
        import mpi4py.MPI as MPI
        comm.init(device=dev, mpi_comm=MPI.COMM_WORLD, initializer_method="mpi")
        initialized = True
    IS_INITIALIZED = initialized
    if not initialized:
        raise RuntimeError("At least one of {torch, mpi4py} must be initialized")


def get_rank() -> int:
    assert IS_INITIALIZED
    return _get_comm().my_pe()


def get_world_size() -> int:
    assert IS_INITIALIZED
    return _get_comm().n_pes()


def sync_all(stream_ptr: int) -> None:
    assert IS_INITIALIZED
    gpu = _get_gpu()
    comm = _get_comm()
    comm.sync_all(stream=gpu.stream_from_handle(stream_ptr))
