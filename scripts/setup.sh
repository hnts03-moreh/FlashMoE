#!/bin/bash
# FlashMoE ROCm full setup script
# Usage:
#   bash scripts/setup.sh            # full setup (C++ build + Python editable install)
#   bash scripts/setup.sh cpp        # C++ test binaries only
#   bash scripts/setup.sh python     # Python package only
#   bash scripts/setup.sh rebuild    # incremental C++ rebuild (after code changes)
#   bash scripts/setup.sh clean      # remove build dir and reinstall everything

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FLASHMOE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${FLASHMOE_DIR}/build"
MODE="${1:-all}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail()  { echo -e "${RED}[FAIL]${NC} $*"; exit 1; }

# ----- Prerequisite checks -----
check_prereqs() {
    info "Checking prerequisites..."

    command -v hipcc >/dev/null 2>&1 || fail "hipcc not found. Install ROCm first."
    info "  hipcc: $(hipcc --version 2>&1 | head -1)"

    ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
    if [ ! -d "$ROCM_PATH" ]; then
        fail "ROCM_PATH=$ROCM_PATH does not exist"
    fi
    info "  ROCM_PATH: $ROCM_PATH"

    command -v cmake >/dev/null 2>&1 || fail "cmake not found"
    CMAKE_VER=$(cmake --version | head -1 | grep -oP '\d+\.\d+')
    info "  cmake: $CMAKE_VER"

    # Ninja is optional but preferred
    if command -v ninja >/dev/null 2>&1; then
        GENERATOR="Ninja"
        info "  generator: Ninja"
    else
        GENERATOR="Unix Makefiles"
        warn "  ninja not found, falling back to make"
    fi

    # Python (optional, for python install)
    if command -v python3 >/dev/null 2>&1; then
        info "  python: $(python3 --version 2>&1)"
    else
        warn "  python3 not found — skipping Python install"
    fi

    # GPU architecture detection
    GPU_ARCH="gfx942"
    if command -v rocminfo >/dev/null 2>&1; then
        DETECTED=$(rocminfo 2>/dev/null | grep -oP 'gfx\w+' | head -1 || true)
        if [ -n "$DETECTED" ]; then
            GPU_ARCH="$DETECTED"
        fi
    fi
    info "  target GPU: $GPU_ARCH"
    echo ""
}

# ----- C++ build -----
build_cpp() {
    info "=== C++ test binaries build ==="

    if [ ! -f "$BUILD_DIR/build.ninja" ] && [ ! -f "$BUILD_DIR/Makefile" ]; then
        info "Configuring CMake..."
        cmake -B "$BUILD_DIR" -S "$FLASHMOE_DIR/csrc" \
            -G "$GENERATOR" \
            -DFLASHMOE_PLATFORM=HIP \
            -DCMAKE_HIP_ARCHITECTURES="$GPU_ARCH" \
            -DCMAKE_BUILD_TYPE=Release
        echo ""
    fi

    info "Building all targets..."
    cmake --build "$BUILD_DIR" --parallel
    echo ""

    info "Build artifacts:"
    for bin in testGEMM gemmMNK testGatedGEMM testGate testCombine testScheduler diagGEMM playground testFlashMoE; do
        if [ -f "$BUILD_DIR/$bin" ]; then
            echo -e "  ${GREEN}✓${NC} $bin"
        else
            echo -e "  ${YELLOW}✗${NC} $bin (not built)"
        fi
    done
    echo ""
}

# ----- Python install -----
install_python() {
    if ! command -v python3 >/dev/null 2>&1; then
        warn "python3 not found, skipping Python install"
        return
    fi

    info "=== Python package install (editable) ==="
    cd "$FLASHMOE_DIR"
    pip install -e ".[rocm]" 2>&1 | tail -3
    echo ""
    info "Verify:"
    python3 -c "import flashmoe; print(f'  flashmoe imported from {flashmoe.__file__}')" 2>&1 || warn "  import failed (non-critical for C++ tests)"
    echo ""
}

# ----- Main -----
echo ""
echo "============================================"
echo " FlashMoE ROCm Setup"
echo " Project: $FLASHMOE_DIR"
echo " Mode:    $MODE"
echo "============================================"
echo ""

case "$MODE" in
    all)
        check_prereqs
        build_cpp
        install_python
        ;;
    cpp)
        check_prereqs
        build_cpp
        ;;
    python)
        check_prereqs
        install_python
        ;;
    rebuild)
        info "Incremental C++ rebuild..."
        if [ ! -d "$BUILD_DIR" ]; then
            fail "No build dir found. Run 'bash scripts/setup.sh cpp' first."
        fi
        cmake --build "$BUILD_DIR" --parallel
        info "Done."
        ;;
    clean)
        info "Cleaning build directory..."
        rm -rf "$BUILD_DIR"
        check_prereqs
        build_cpp
        install_python
        ;;
    *)
        echo "Usage: bash scripts/setup.sh [all|cpp|python|rebuild|clean]"
        exit 1
        ;;
esac

info "=== Setup complete ==="
