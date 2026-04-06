"""
rocshmem — Python bindings for ROCSHMEM (AMD ROCm SHMEM)

Drop-in replacement for nvshmem4py on ROCm platforms.
Provides rocshmem.core with init/finalize/my_pe/n_pes/malloc/free/barrier_all/sync_all.
"""

from . import core  # noqa: F401

__version__ = "0.1.0"
