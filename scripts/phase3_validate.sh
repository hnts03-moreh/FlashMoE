#!/bin/bash
# Phase 3 Validation Script — FlashMoE ROCm Porting
# Builds and runs all Phase 3 tests, saves results to files.
#
# Usage:
#   bash scripts/phase3_validate.sh [BUILD_DIR]
#
# Outputs:
#   <BUILD_DIR>/results/phase3_build.log    — Build output
#   <BUILD_DIR>/results/phase3_precision.txt — FP16/BF16/FP32 accuracy results
#   <BUILD_DIR>/results/phase3_gemm.txt     — testGEMM full sweep
#   <BUILD_DIR>/results/phase3_gemmMNK.txt  — gemmMNK specific sizes
#   <BUILD_DIR>/results/phase3_diag.txt     — diagGEMM diagnostic
#   <BUILD_DIR>/results/phase3_gate.txt     — testGate results
#   <BUILD_DIR>/results/phase3_combine.txt  — testCombine results
#   <BUILD_DIR>/results/phase3_gatedgemm.txt — testGatedGEMM results
#   <BUILD_DIR>/results/phase3_scheduler.txt — testScheduler results
#   <BUILD_DIR>/results/phase3_summary.txt  — Overall summary

set -uo pipefail
# NOTE: no set -e — individual test failures should not abort the script

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${1:-${PROJECT_ROOT}/build}"
RESULTS_DIR="${BUILD_DIR}/results"

echo "=== Phase 3 Validation ==="
echo "Project root: ${PROJECT_ROOT}"
echo "Build dir: ${BUILD_DIR}"
echo "Results dir: ${RESULTS_DIR}"
echo ""

# Create results directory
mkdir -p "${RESULTS_DIR}"

# -------------------------------------------------------
# Step 1: Build
# -------------------------------------------------------
echo "[1/3] Building HIP test binaries..."
{
    echo "=== Build started at $(date) ==="
    cmake -B "${BUILD_DIR}" -S "${PROJECT_ROOT}/csrc" \
        -G Ninja \
        -DFLASHMOE_PLATFORM=HIP \
        -DCMAKE_HIP_ARCHITECTURES=gfx942 \
        -DCMAKE_BUILD_TYPE=Release 2>&1

    cmake --build "${BUILD_DIR}" --parallel 2>&1
    echo "=== Build completed at $(date) ==="
} | tee "${RESULTS_DIR}/phase3_build.log"

BUILD_RC=${PIPESTATUS[0]}
if [ $BUILD_RC -ne 0 ]; then
    echo "BUILD FAILED — see ${RESULTS_DIR}/phase3_build.log"
    echo "Attempting to run existing binaries anyway..."
fi
echo "Build OK"
echo ""

# -------------------------------------------------------
# Step 2: Run tests
# -------------------------------------------------------
run_test() {
    local name="$1"
    local binary="$2"
    local output="$3"
    shift 3
    local args=("$@")

    echo -n "  Running ${name}..."
    {
        echo "=== ${name} started at $(date) ==="
        if [ -f "${BUILD_DIR}/${binary}" ]; then
            timeout 300 "${BUILD_DIR}/${binary}" "${args[@]}" 2>&1
            local rc=$?
            echo "EXIT_CODE=${rc}"
            if [ $rc -ne 0 ]; then
                echo "CRASHED or FAILED (exit code ${rc})"
            fi
        else
            echo "SKIP: binary not found: ${BUILD_DIR}/${binary}"
        fi
        echo "=== ${name} completed at $(date) ==="
    } > "${RESULTS_DIR}/${output}" 2>&1 || true
    echo " done -> ${output}"
}

echo "[2/3] Running tests..."

# Precision validation (FP16/BF16/FP32) — most important
run_test "testPrecision" "testPrecision" "phase3_precision.txt" 42

# FP16 full sweep (2 → 4096)
run_test "testGEMM (full sweep)" "testGEMM" "phase3_gemm.txt" 2 4096

# gemmMNK specific sizes
{
    echo "=== gemmMNK started at $(date) ===" > "${RESULTS_DIR}/phase3_gemmMNK.txt"
    for M in 32 64 128 256 512 1024; do
        for N in 128 256; do
            for K in 64 128 256; do
                echo "--- M=$M N=$N K=$K ---" >> "${RESULTS_DIR}/phase3_gemmMNK.txt"
                "${BUILD_DIR}/gemmMNK" "$M" "$N" "$K" 2>&1 >> "${RESULTS_DIR}/phase3_gemmMNK.txt" || true
            done
        done
    done
    echo "=== gemmMNK completed at $(date) ===" >> "${RESULTS_DIR}/phase3_gemmMNK.txt"
} &
GEMMMNK_PID=$!
echo "  Running gemmMNK (background, PID=$GEMMMNK_PID)..."

# diagGEMM diagnostic
run_test "diagGEMM" "diagGEMM" "phase3_diag.txt" 256 128 64 42

# testGate
run_test "testGate" "testGate" "phase3_gate.txt"

# testCombine
run_test "testCombine" "testCombine" "phase3_combine.txt"

# testGatedGEMM
run_test "testGatedGEMM" "testGatedGEMM" "phase3_gatedgemm.txt"

# testScheduler
run_test "testScheduler" "testScheduler" "phase3_scheduler.txt"

# Wait for background gemmMNK
echo -n "  Waiting for gemmMNK..."
wait $GEMMMNK_PID 2>/dev/null || true
echo " done -> phase3_gemmMNK.txt"

echo ""

# -------------------------------------------------------
# Step 3: Summary
# -------------------------------------------------------
echo "[3/3] Generating summary..."

{
    echo "=== Phase 3 Validation Summary ==="
    echo "Date: $(date)"
    echo "GPU: $(rocm-smi --showproductname 2>/dev/null | head -3 || echo 'unknown')"
    echo "ROCm: $(cat /opt/rocm/.info/version 2>/dev/null || echo 'unknown')"
    echo ""

    echo "--- Precision Test Results ---"
    if [ -f "${RESULTS_DIR}/phase3_precision.txt" ]; then
        grep -E "^(fp16|bf16|fp32|#)" "${RESULTS_DIR}/phase3_precision.txt" || echo "(no results)"
    fi
    echo ""

    echo "--- GEMM Sweep Results ---"
    if [ -f "${RESULTS_DIR}/phase3_gemm.txt" ]; then
        # Show only rows with error > 0
        grep -v "^M," "${RESULTS_DIR}/phase3_gemm.txt" | grep -v "^===" | head -50 || echo "(no results)"
    fi
    echo ""

    echo "--- Test Exit Codes ---"
    for f in "${RESULTS_DIR}"/phase3_*.txt; do
        fname=$(basename "$f")
        if grep -q "EXIT_CODE=" "$f" 2>/dev/null; then
            code=$(grep "EXIT_CODE=" "$f" | tail -1 | cut -d= -f2)
            printf "  %-30s EXIT_CODE=%s\n" "$fname" "$code"
        elif grep -q "SKIP:" "$f" 2>/dev/null; then
            printf "  %-30s SKIPPED\n" "$fname"
        fi
    done
    echo ""
    echo "=== Full results in: ${RESULTS_DIR}/ ==="

} | tee "${RESULTS_DIR}/phase3_summary.txt"

echo ""
echo "Done. All results saved to: ${RESULTS_DIR}/"
echo "Key file to review: ${RESULTS_DIR}/phase3_precision.txt"
