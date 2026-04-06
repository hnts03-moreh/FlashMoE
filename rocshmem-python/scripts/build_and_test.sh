#!/bin/bash
# Build and test rocshmem-python bindings
# Usage: bash scripts/build_and_test.sh [NUM_GPUS]
set -uo pipefail

NUM_GPUS="${1:-2}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== Building rocshmem-python ==="
cd "$PROJECT_DIR"

# Ensure pybind11 is available
pip install pybind11 2>/dev/null || true

# Build with CMake
cmake -B build -S . \
    -DCMAKE_PREFIX_PATH=/opt/rocm \
    -DCMAKE_BUILD_TYPE=Release 2>&1

cmake --build build --parallel 2>&1

echo ""
echo "=== Build complete ==="
echo ""

# Copy the built module into the package
cp build/core*.so rocshmem/ 2>/dev/null || true

echo "=== Testing rocshmem-python (${NUM_GPUS} GPUs) ==="

# Test 1: Import test (no MPI needed)
echo "--- Test 1: Import ---"
PYTHONPATH="$PROJECT_DIR" python3 -c "
import rocshmem.core as core
print(f'InitStatus enum: {core.InitStatus.STATUS_NOT_INITIALIZED}')
print(f'init_status(): {core.init_status()}')
print('Import OK')
" 2>&1 || echo "FAIL: import test"

echo ""

# Test 2: Init/finalize with MPI
echo "--- Test 2: MPI init/finalize (${NUM_GPUS} PEs) ---"
PYTHONPATH="$PROJECT_DIR" mpirun --allow-run-as-root -np "$NUM_GPUS" \
    --bind-to none \
    -x ROCM_PATH -x LD_LIBRARY_PATH -x PYTHONPATH \
    python3 -c "
import rocshmem.core as core

print(f'PE ?: init_status = {core.init_status()}')
core.init()
pe = core.my_pe()
n = core.n_pes()
print(f'PE {pe}/{n}: initialized')

core.barrier_all()
print(f'PE {pe}/{n}: barrier OK')

# Test malloc/free
ptr = core.malloc(1024)
print(f'PE {pe}/{n}: malloc({1024}) = 0x{ptr:x}')
core.free(ptr)
print(f'PE {pe}/{n}: free OK')

core.barrier_all()
core.finalize()
print(f'PE {pe}/{n}: finalized')
" 2>&1 || echo "FAIL: MPI test"

echo ""
echo "=== Done ==="
