# FlashMoE Build Guide

## Prerequisites

### Common Requirements

- CMake >= 3.28
- Ninja build system (recommended)
- C++20 compiler (GCC 12+ or Clang 16+)
- Python >= 3.10
- pybind11 (fetched automatically via CPM)

### CUDA Build

- CUDA Toolkit >= 12.0
- NVIDIA GPU with compute capability >= 7.0 (Volta+)
- NVSHMEM (with `NVSHMEM_LIB_HOME` environment variable set)
- MathDx SDK (provides cuBLASDx, cuRANDDx)
- MPI implementation (for distributed tests)

### ROCm/HIP Build

- ROCm >= 6.0 (required for MI300X MFMA intrinsics)
- `hipcc` compiler (ships with ROCm)
- hipcub (`find_package(hipcub)`)
- hipRAND (`find_package(hiprand)`)
- rocTX (libroctx64, ships with ROCm)
- ROCSHMEM (optional -- only needed for distributed multi-GPU)
- MPI implementation (optional -- only for distributed tests)

## CUDA Build (Original)

The CUDA build path is unchanged from the original project:

```bash
# Configure
cmake -B build -S csrc \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=90a

# Build
cmake --build build --parallel
```

Required environment variables:
```bash
export NVSHMEM_LIB_HOME=/path/to/nvshmem/lib
export NVSHMEM_HOME=/path/to/nvshmem
```

## ROCm/HIP Build

### Quick Start

```bash
# Configure for MI300X
cmake -B build -S csrc \
  -G Ninja \
  -DFLASHMOE_PLATFORM=HIP \
  -DCMAKE_HIP_ARCHITECTURES=gfx942 \
  -DCMAKE_BUILD_TYPE=Release

# Build all test targets
cmake --build build --parallel
```

### Step-by-Step

1. **Verify ROCm installation:**

```bash
# Check hipcc
hipcc --version

# Verify ROCM_PATH
echo $ROCM_PATH  # Should be /opt/rocm or similar

# Check GPU visibility
rocm-smi --showproductname
```

2. **Configure the build:**

```bash
cmake -B build -S csrc \
  -G Ninja \
  -DFLASHMOE_PLATFORM=HIP \
  -DCMAKE_HIP_ARCHITECTURES=gfx942 \
  -DCMAKE_BUILD_TYPE=Release
```

The `-DFLASHMOE_PLATFORM=HIP` flag forces HIP mode. If omitted, the build system auto-detects based on the presence of `hipcc` and `ROCM_PATH`.

3. **Build:**

```bash
cmake --build build --parallel
```

This produces the following test executables:

| Target | Description |
|--------|-------------|
| `testScheduler` | Scheduler kernel test |
| `testGEMM` | GEMM sweep benchmark |
| `gemmMNK` | Fixed M/N/K GEMM test |
| `testGatedGEMM` | Gated MLP GEMM test |
| `testGate` | Fused gate kernel test |
| `testCombine` | Token combine test |
| `testFlashMoE` | E2E distributed test (stub -- requires ROCSHMEM) |
| `playground` | Minimal test |

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `FLASHMOE_PLATFORM` | `AUTO` | Platform selection: `AUTO`, `CUDA`, or `HIP` |
| `CMAKE_HIP_ARCHITECTURES` | `gfx942` | Target GPU architecture |
| `CMAKE_BUILD_TYPE` | (none) | `Release` or `Debug` (required) |
| `NUM_CUS` | Auto-detected / 304 | Number of compute units (used for `NUM_SMS` define) |

### ROCSHMEM Setup (Optional)

ROCSHMEM enables distributed multi-GPU operation. If not installed, the build succeeds but distributed features are disabled.

```bash
# If ROCSHMEM is installed with CMake config files:
cmake -B build -S csrc \
  -G Ninja \
  -DFLASHMOE_PLATFORM=HIP \
  -DCMAKE_HIP_ARCHITECTURES=gfx942 \
  -DCMAKE_BUILD_TYPE=Release \
  -Drocshmem_DIR=/path/to/rocshmem/lib/cmake/rocshmem
```

When ROCSHMEM is found, the build defines `FLASHMOE_HAS_ROCSHMEM=1` and links `rocshmem::rocshmem`.

### Building the Root Library (Header-Only)

The root `CMakeLists.txt` provides the `flashmoe::flashmoe` INTERFACE library target for consumers:

```bash
cmake -B build_lib -S . \
  -G Ninja \
  -DFLASHMOE_PLATFORM=HIP \
  -DCMAKE_HIP_ARCHITECTURES=gfx942

# No build step needed -- it is header-only
# Consumers link via: target_link_libraries(my_target PRIVATE flashmoe::flashmoe)
```

## Python Package Installation

### For ROCm

```bash
pip install .[rocm]
```

This installs `hip-python>=6.0.0` as a dependency.

### For CUDA 12

```bash
pip install .[cu12]
```

This installs `cuda-core>=0.5.0` and `nvshmem4py-cu12`.

### For CUDA 13

```bash
pip install .[cu13]
```

## JIT Compilation

FlashMoE compiles specialized kernels at runtime via JIT. The Python layer handles this automatically.

### Platform Detection

The JIT system detects the platform in this order:

1. `FLASHMOE_PLATFORM` environment variable (`hip`/`rocm` or `cuda`/`nvidia`)
2. `torch.version.hip` (set when PyTorch is built for ROCm)
3. Presence of `hipcc` on `PATH` with a valid `ROCM_PATH`
4. Default to CUDA

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASHMOE_PLATFORM` | Auto-detect | Force platform: `hip`, `rocm`, `cuda`, or `nvidia` |
| `FLASHMOE_HIP_ARCH` | `gfx942` | HIP target architecture string |
| `FLASHMOE_CACHE_DIR` | `~/.cache/flashmoe_jit` | JIT compilation cache root |

### JIT Cache

Compiled artifacts are cached under platform-specific subdirectories:
- CUDA: `~/.cache/flashmoe_jit/cuda/`
- HIP: `~/.cache/flashmoe_jit/hip/`

The cache key includes a hash of all `.cuh` and `.h` headers under `csrc/include/`, the platform name, and the generated source code. Changing any header invalidates the cache.

To clear the cache:
```bash
rm -rf ~/.cache/flashmoe_jit/hip/
```

### JIT Source Files

On HIP, JIT-generated source files use the `.cpp` extension (required by CMake's HIP language support). On CUDA, they use `.cu`. The JIT build passes `FLASHMOE_PLATFORM_HIP=1` and `FLASHMOE_USE_MATH_COMPAT=1` as compile definitions on HIP.

## Running Tests

### Single-GPU Kernel Tests (ROCm)

```bash
cd build

# GEMM tests
./testGEMM
./gemmMNK
./testGatedGEMM

# Gate and combine
./testGate
./testCombine

# Scheduler
./testScheduler
```

These tests do not require ROCSHMEM or MPI.

### Distributed E2E Test (ROCm)

The E2E test (`testFlashMoE`) is currently a stub on HIP. It prints a message and exits. Full E2E testing requires:
- ROCSHMEM installed and linked
- MPI runtime (`mpirun`)
- Multiple GPUs

### Python Smoke Test

```bash
# Verify platform detection
FLASHMOE_PLATFORM=hip python -c "from flashmoe.jit import get_platform; print(get_platform())"
# Expected output: Platform.HIP
```

## Troubleshooting

### `hipcc` not found

Ensure ROCm is installed and `hipcc` is on your PATH:
```bash
export PATH=/opt/rocm/bin:$PATH
export ROCM_PATH=/opt/rocm
```

### CMake cannot find `hip` package

Set the ROCm cmake module path:
```bash
cmake ... -DCMAKE_PREFIX_PATH=/opt/rocm
```

### `hipcub` or `hiprand` not found

These ship with ROCm. Verify installation:
```bash
ls /opt/rocm/include/hipcub/
ls /opt/rocm/lib/libhiprand*
```

If installed in a non-standard location:
```bash
cmake ... -Dhipcub_DIR=/path/to/hipcub/lib/cmake/hipcub
```

### JIT compilation fails on HIP

1. Check that `hipcc` is accessible from the Python process
2. Verify `FLASHMOE_HIP_ARCH` matches your GPU (default: `gfx942`)
3. Inspect the build log in `~/.cache/flashmoe_jit/hip/<module_name>/build_*/`

### ROCSHMEM link errors

ROCSHMEM is optional. If you see link errors related to `rocshmem`, either:
- Install ROCSHMEM and point CMake to it via `-Drocshmem_DIR=...`
- Or remove the ROCSHMEM dependency (distributed features will be disabled)

### `__grid_constant__` warnings

HIP does not support `__grid_constant__`. The `device.h` header defines it as a no-op. If you see warnings, they are harmless.

### HIP compilation is slow

HIP compilation with `-fgpu-rdc` (relocatable device code) can be slower than non-RDC builds. This is required for device-side function calls across translation units. Using Ninja with `--parallel` helps.
