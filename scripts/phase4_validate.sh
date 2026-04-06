#!/bin/bash
# Phase 4 Validation Script — FlashMoE ROCm Porting
# ROCSHMEM integration testing: build, link, and run distributed E2E tests.
#
# Prerequisites:
#   - ROCm 7.0+ installed
#   - MPI (OpenMPI or MPICH) installed
#   - ROCSHMEM built and installed (run build_rocshmem.sh first if needed)
#   - 2+ AMD GPUs available
#
# Usage:
#   bash scripts/phase4_validate.sh [BUILD_DIR] [NUM_GPUS]
#
# Outputs:
#   <BUILD_DIR>/results/phase4_rocshmem_build.log
#   <BUILD_DIR>/results/phase4_flashmoe_build.log
#   <BUILD_DIR>/results/phase4_e2e_2gpu.txt
#   <BUILD_DIR>/results/phase4_python_init.txt
#   <BUILD_DIR>/results/phase4_summary.txt

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${1:-${PROJECT_ROOT}/build}"
NUM_GPUS="${2:-2}"
RESULTS_DIR="${BUILD_DIR}/results"

echo "=== Phase 4 Validation ==="
echo "Project root: ${PROJECT_ROOT}"
echo "Build dir:    ${BUILD_DIR}"
echo "Num GPUs:     ${NUM_GPUS}"
echo ""

mkdir -p "${RESULTS_DIR}"

# -------------------------------------------------------
# Step 0: Check ROCSHMEM availability
# -------------------------------------------------------
echo "[0/5] Checking ROCSHMEM..."
ROCSHMEM_OK=false
if pkg-config --exists rocshmem 2>/dev/null; then
    echo "  ROCSHMEM found via pkg-config"
    ROCSHMEM_OK=true
elif [ -f "/opt/rocm/lib/librocshmem.so" ]; then
    echo "  ROCSHMEM found at /opt/rocm/lib/librocshmem.so"
    ROCSHMEM_OK=true
elif [ -f "/opt/rocm/lib/cmake/rocshmem/rocshmem-config.cmake" ]; then
    echo "  ROCSHMEM CMake config found"
    ROCSHMEM_OK=true
else
    echo "  WARNING: ROCSHMEM not found. Run build_rocshmem.sh first."
    echo "  Will attempt build anyway (CMake will handle missing dependency)."
fi
echo ""

# -------------------------------------------------------
# Step 1: Build ROCSHMEM (if needed)
# -------------------------------------------------------
if [ "$ROCSHMEM_OK" = false ] && [ -f "${SCRIPT_DIR}/build_rocshmem.sh" ]; then
    echo "[1/5] Building ROCSHMEM..."
    bash "${SCRIPT_DIR}/build_rocshmem.sh" 2>&1 | tee "${RESULTS_DIR}/phase4_rocshmem_build.log"
    echo ""
else
    echo "[1/5] ROCSHMEM already available, skipping build."
    echo ""
fi

# -------------------------------------------------------
# Step 2: Build FlashMoE with ROCSHMEM
# -------------------------------------------------------
echo "[2/5] Building FlashMoE with ROCSHMEM..."
{
    echo "=== Build started at $(date) ==="
    cmake -B "${BUILD_DIR}" -S "${PROJECT_ROOT}/csrc" \
        -G Ninja \
        -DFLASHMOE_PLATFORM=HIP \
        -DCMAKE_HIP_ARCHITECTURES=gfx942 \
        -DCMAKE_BUILD_TYPE=Release 2>&1

    cmake --build "${BUILD_DIR}" --parallel 2>&1
    echo "=== Build completed at $(date) ==="
} | tee "${RESULTS_DIR}/phase4_flashmoe_build.log"
echo ""

# -------------------------------------------------------
# Step 3: ROCSHMEM E2E test (multi-GPU)
# -------------------------------------------------------
echo "[3/5] Running testFlashMoE with ${NUM_GPUS} GPUs..."
{
    echo "=== E2E test started at $(date) ==="
    if [ -f "${BUILD_DIR}/testFlashMoE" ]; then
        # Use MPI to launch multi-GPU
        timeout 120 mpirun --allow-run-as-root -np "${NUM_GPUS}" \
            --bind-to none \
            -x ROCM_PATH -x LD_LIBRARY_PATH -x PATH \
            "${BUILD_DIR}/testFlashMoE" 2>&1
        echo "EXIT_CODE=$?"
    else
        echo "SKIP: testFlashMoE binary not found"
    fi
    echo "=== E2E test completed at $(date) ==="
} > "${RESULTS_DIR}/phase4_e2e_${NUM_GPUS}gpu.txt" 2>&1 || true
echo "  -> phase4_e2e_${NUM_GPUS}gpu.txt"
echo ""

# -------------------------------------------------------
# Step 4: Python ROCSHMEM init test
# -------------------------------------------------------
echo "[4/5] Testing Python ROCSHMEM init..."
{
    echo "=== Python init test started at $(date) ==="
    timeout 60 mpirun --allow-run-as-root -np "${NUM_GPUS}" \
        --bind-to none \
        -x ROCM_PATH -x LD_LIBRARY_PATH -x PATH -x PYTHONPATH \
        python3 -c "
import os
os.environ['FLASHMOE_PLATFORM'] = 'HIP'
try:
    from flashmoe.cb import _is_hip_platform, _get_comm, _get_gpu, initialize, get_rank, get_world_size, get_local_rank
    print(f'Platform is HIP: {_is_hip_platform()}')
    print(f'Local rank: {get_local_rank()}')
    initialize()
    print(f'Rank {get_rank()} of {get_world_size()} initialized successfully')
except NotImplementedError as e:
    print(f'Expected: ROCSHMEM Python bindings not available: {e}')
except Exception as e:
    print(f'Error: {type(e).__name__}: {e}')
" 2>&1
    echo "EXIT_CODE=$?"
    echo "=== Python init test completed at $(date) ==="
} > "${RESULTS_DIR}/phase4_python_init.txt" 2>&1 || true
echo "  -> phase4_python_init.txt"
echo ""

# -------------------------------------------------------
# Step 5: Summary
# -------------------------------------------------------
echo "[5/5] Generating summary..."
{
    echo "=== Phase 4 Validation Summary ==="
    echo "Date: $(date)"
    echo "GPUs: ${NUM_GPUS}"
    echo "ROCSHMEM: ${ROCSHMEM_OK}"
    echo ""

    echo "--- Build Results ---"
    if [ -f "${RESULTS_DIR}/phase4_flashmoe_build.log" ]; then
        grep -E "(ROCSHMEM|Build completed|error:)" "${RESULTS_DIR}/phase4_flashmoe_build.log" | tail -10
    fi
    echo ""

    echo "--- E2E Test Results ---"
    if [ -f "${RESULTS_DIR}/phase4_e2e_${NUM_GPUS}gpu.txt" ]; then
        cat "${RESULTS_DIR}/phase4_e2e_${NUM_GPUS}gpu.txt"
    fi
    echo ""

    echo "--- Python Init Test ---"
    if [ -f "${RESULTS_DIR}/phase4_python_init.txt" ]; then
        cat "${RESULTS_DIR}/phase4_python_init.txt"
    fi
    echo ""
    echo "=== Full results in: ${RESULTS_DIR}/ ==="
} | tee "${RESULTS_DIR}/phase4_summary.txt"

echo ""
echo "Done. Results in: ${RESULTS_DIR}/"
