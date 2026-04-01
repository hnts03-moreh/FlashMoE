from __future__ import annotations

import os
import threading
from pathlib import Path
from enum import IntEnum

# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------

class Platform(IntEnum):
    CUDA = 0
    HIP = 1

def detect_platform() -> Platform:
    """Detect the GPU platform.

    Priority:
      1. ``FLASHMOE_PLATFORM`` environment variable (``cuda`` or ``hip``)
      2. ``torch.version.hip`` (set when PyTorch is built for ROCm)
      3. Presence of ``hipcc`` on ``PATH``
      4. Default to CUDA
    """
    env = os.environ.get("FLASHMOE_PLATFORM", "").strip().lower()
    if env in ("hip", "rocm"):
        return Platform.HIP
    if env in ("cuda", "nvidia"):
        return Platform.CUDA

    # Auto-detect via PyTorch
    try:
        import torch
        if hasattr(torch.version, "hip") and torch.version.hip is not None:
            return Platform.HIP
    except ImportError:
        pass

    # Auto-detect via hipcc on PATH
    import shutil
    if shutil.which("hipcc") is not None:
        rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
        if os.path.isdir(rocm_path):
            return Platform.HIP

    return Platform.CUDA


# Module-level cached value
_PLATFORM: Platform | None = None

def get_platform() -> Platform:
    """Return the cached detected platform."""
    global _PLATFORM
    if _PLATFORM is None:
        _PLATFORM = detect_platform()
    return _PLATFORM


def is_hip() -> bool:
    return get_platform() == Platform.HIP


def is_cuda() -> bool:
    return get_platform() == Platform.CUDA


# ---------------------------------------------------------------------------
# Data classes & enums
# ---------------------------------------------------------------------------

class ContextHandle:
    __slots__ = ("_mod", "_ctx")

    def __init__(self, mod, ctx):
        self._mod = mod
        self._ctx = ctx

    @property
    def context(self):
        return self._ctx

    @property
    def mod(self):
        return self._mod

class Topology(IntEnum):
    NVLINK_ONLY = 0
    MIXED = 1

class MLPType(IntEnum):
    GATED = 0
    VANILLA = 1

class DataType(IntEnum):
    BF16 = 0
    FP16 = 1
    FP32 = 2
    FP64 = 3

class ActivationType(IntEnum):
    IDENTITY = 0
    SILU = 1
    GELU = 2
    RELU = 3 # as many as CUTLASS supports

class ForwardArgs:
    # integers below are for addresses to the underlying contiguous memory of the corresponding tensor
    tokens : int # [S, H]
    expert_counts: int # [E]
    local_expert_up: int # [num_local_experts, H, I]
    local_expert_up_v: int # [num_local_experts, H, I]
    local_bias_up: int # [num_local_experts, I]
    local_bias_up_v: int # [num_local_experts, I]
    local_expert_down: int # [num_local_experts, I, H]
    local_bias_down: int # [num_local_experts, H]
    moe_out: int # [S, H]
    swish_alpha: float = 1.0
    swish_beta: float = 1.0
    stream_ptr: int
    def __init__(self,
                 mt: MLPType,
                 tokens: int,
                 expert_counts: int,
                 local_expert_up: int,
                 local_bias_up: int,
                 local_expert_down: int,
                 local_bias_down: int,
                 moe_out: int,
                 stream_ptr: int,
                 *,
                 swish_alpha: float = 1.0,
                 swish_beta: float = 1.0,
                 local_expert_up_v: int = None,
                 local_bias_up_v: int = None):
        # if GatedMLP, then mlp weights should be specified
        assert not mt == MLPType.GATED or local_expert_up_v is not None
        assert not mt == MLPType.GATED or local_bias_up_v is not None
        self.tokens = tokens
        self.expert_counts = expert_counts
        self.local_expert_up = local_expert_up
        self.local_expert_up_v = local_expert_up_v
        self.local_expert_down = local_expert_down
        self.local_bias_up = local_bias_up
        self.local_bias_up_v = local_bias_up_v
        self.local_bias_down = local_bias_down
        self.moe_out = moe_out
        self.stream_ptr = stream_ptr
        self.swish_alpha = swish_alpha
        self.swish_beta = swish_beta

class InitArgs:
    from typing import List
    data_type: DataType
    tokens_per_rank: int
    token_dim: int
    ffn_size: int
    num_experts: int
    top_k: int
    expert_map: List[int]
    rank_map: List[int]
    gpu_arch: int
    topo: Topology = Topology.MIXED
    mlp_type: MLPType
    act_type: ActivationType
    device_id: int
    ep_world: int
    ep_rank: int
    my_pe: int
    num_local_experts: int
    stream_ptr: int
    expert_peer_capacity: int

    def __init__(self,
                 data_type: DataType,
                 tokens_per_rank: int,
                 token_dim: int,
                 ffn_size: int,
                 num_experts: int,
                 top_k: int,
                 gpu_arch: int,
                 mlp_type: MLPType,
                 act_type: ActivationType,
                 stream_ptr: int,
                 device_id: int,
                 *,
                 ep_world: int = None,
                 num_local_experts: int = None,
                 ep_rank: int = None,
                 my_pe: int = None,
                 expert_map: List[int] = None,
                 rank_map: List[int] = None,
                 expert_peer_capacity: int = None) -> None:
        from math import ceil
        # CUDA arch values are numeric (e.g. 900), HIP arch is gfx numeric (e.g. 942)
        # Both should be >= 700 for CUDA, or any positive for HIP
        if is_hip():
            assert gpu_arch > 0, f"gpu_arch must be positive, got {gpu_arch}"
        else:
            assert gpu_arch >= 700, f"gpu_arch must be >= 700 for CUDA, got {gpu_arch}"
        self.data_type = data_type
        self.tokens_per_rank = tokens_per_rank
        self.token_dim = token_dim
        self.ffn_size = ffn_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_map = expert_map
        self.rank_map = rank_map
        self.gpu_arch = gpu_arch
        self.mlp_type = mlp_type
        self.act_type = act_type
        self.device_id=device_id
        self.ep_world = ep_world
        self.ep_rank = ep_rank
        self.my_pe = my_pe
        self.num_local_experts = num_local_experts
        self.stream_ptr = stream_ptr
        if expert_peer_capacity is None:
            self.expert_peer_capacity = ceil(float(tokens_per_rank) / num_experts) * top_k
        else:
            self.expert_peer_capacity = expert_peer_capacity

def _verify_dirs() -> None:
    from pathlib import Path
    root = Path(__file__).resolve().parent

    if not (root / "CMakeLists.txt").exists():
        raise RuntimeError("JIT CMakeLists.txt not found at package root")

def _load_ext(mod_name: str, so_path: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(mod_name, so_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {mod_name} from {so_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def _source_fingerprint() -> str:
    from pathlib import Path
    import hashlib

    root = Path(__file__).resolve().parent.parent
    include_dir = root / "csrc" / "include"

    h = hashlib.sha256()
    h.update(b"flashmoe-jit-v2")

    # Include platform headers in the fingerprint so that changes to
    # platform abstraction also invalidate the JIT cache.
    files = sorted(include_dir.glob("**/*.cuh")) + sorted(include_dir.glob("**/*.h"))

    for path in files:
        h.update(str(path.relative_to(root)).encode())
        h.update(path.read_bytes())

    # Also include the detected platform in the hash
    h.update(f"platform={get_platform().name}".encode())

    return h.hexdigest()[:16]


def _get_compiled(arg: InitArgs, src: str, mod_prefix: str, mod_name: str):
    import os
    import sys
    import time
    import socket
    import shutil
    import hashlib
    import subprocess
    from pathlib import Path

    _verify_dirs()

    platform = get_platform()
    platform_tag = platform.name.lower()  # "cuda" or "hip"

    cache = Path(os.environ.get(
        "FLASHMOE_CACHE_DIR",
        str(Path.home() / ".cache" / "flashmoe_jit")
    ))
    # Separate cache directories per platform to avoid collisions
    cache = cache / platform_tag
    cache.mkdir(parents=True, exist_ok=True)

    key = hashlib.sha256(f"{mod_name}|py{sys.version_info[:2]}|{src}".encode()).hexdigest()[:16]

    build_root = cache / f"{mod_name}_{key}"
    build_root.mkdir(parents=True, exist_ok=True)

    so_path = build_root / f"{mod_name}.so"
    lock_path = build_root / ".build.lock"

    # Fast path
    if so_path.exists():
        return _load_ext(mod_name, so_path)

    # Process-unique tag for temp dirs
    host = socket.gethostname()
    pid = os.getpid()
    tid = threading.get_ident()
    uniq = f"{host}_tid{tid}_pid{pid}"

    gen_dir = build_root / f"gen_{uniq}"
    bdir = build_root / f"build_{uniq}"
    gen_dir.mkdir(exist_ok=True)
    bdir.mkdir(exist_ok=True)

    # HIP source files use .cpp extension; CUDA uses .cu
    if platform == Platform.HIP:
        generated = gen_dir / f"{mod_prefix}_bindings.cpp"
    else:
        generated = gen_dir / f"{mod_prefix}_bindings.cu"
    generated.write_text(src)

    root = Path(__file__).resolve().parent.parent
    cmake_source_dir = root / "flashmoe"  # because root CMakeLists.txt lives here

    def _try_acquire_lock() -> bool:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w") as f:
                f.write(f"host={host}\npid={pid}\ntid={tid}\ntime={time.time()}\n")
            return True
        except FileExistsError:
            return False

    def _release_lock() -> None:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass

    def _wait_for_artifact(timeout_s: float = 1800.0, poll_s: float = 0.1):
        start = time.time()
        while True:
            if so_path.exists():
                return _load_ext(mod_name, so_path)

            if time.time() - start > timeout_s:
                raise TimeoutError(
                    f"Timed out waiting for JIT artifact {so_path} while another process was building it."
                )

            time.sleep(poll_s)

    # Try to become the builder
    have_lock = _try_acquire_lock()

    if not have_lock:
        # Another process is building. Wait for the final .so to appear.
        return _wait_for_artifact()

    try:
        # Double-check after lock acquisition in case another process finished just before us
        if so_path.exists():
            return _load_ext(mod_name, so_path)

        # Build the cmake command based on platform
        cmake_args = [
            "cmake", "-S", str(cmake_source_dir), "-B", str(bdir), "-G", "Ninja",
            f"-DGENERATED_SRC={generated}",
            f"-DTARGET_MODULE_NAME={mod_name}",
            f"-DCPM_SOURCE_CACHE={Path.home() / '.cache' / 'cpm'}",
            "-DCMAKE_BUILD_TYPE=Release",
        ]

        if platform == Platform.HIP:
            # HIP build: pass platform, arch string
            hip_arch = os.environ.get("FLASHMOE_HIP_ARCH", "gfx942")
            cmake_args += [
                "-DFLASHMOE_PLATFORM=HIP",
                f"-DCMAKE_HIP_ARCHITECTURES={hip_arch}",
                f"-DARCH={hip_arch}",
            ]
        else:
            # CUDA build: traditional numeric arch
            cmake_args += [
                "-DFLASHMOE_PLATFORM=CUDA",
                f"-DCMAKE_CUDA_ARCHITECTURES={arg.gpu_arch // 10}",
                f"-DARCH={arg.gpu_arch // 10}",
            ]

        subprocess.run(cmake_args, check=True)

        subprocess.run([
            "cmake", "--build", str(bdir), "--parallel"
        ], check=True)

        built = next(bdir.glob(mod_name + "*.so"))

        # Copy into a temp path in build_root, then atomically replace final path
        tmp_so = build_root / f".{mod_name}.{uniq}.tmp.so"
        shutil.copy2(built, tmp_so)
        tmp_so.replace(so_path)

    finally:
        _release_lock()

    return _load_ext(mod_name, so_path)
