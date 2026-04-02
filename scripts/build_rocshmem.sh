#!/bin/bash
set -euo pipefail

ROCSHMEM_SRC="/app/geonwoo/workspace/mif/flashmoe-rocm-porting/rocm-systems/projects/rocshmem"
BUILD_DIR="${ROCSHMEM_SRC}/build"
INSTALL_PREFIX="/opt/rocm"

echo "=== Building ROCSHMEM ==="
echo "Source:  ${ROCSHMEM_SRC}"
echo "Build:   ${BUILD_DIR}"
echo "Install: ${INSTALL_PREFIX}"
echo ""

# Clean previous build
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"

cd "${BUILD_DIR}"

cmake "${ROCSHMEM_SRC}" \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_RO=ON \
    -DBUILD_FUNCTIONAL_TESTS=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_UNIT_TESTS=OFF

echo ""
echo "=== CMake configure done. Building... ==="
make -j$(nproc)

echo ""
echo "=== Build done. Installing to ${INSTALL_PREFIX}... ==="
make install

echo ""
echo "=== ROCSHMEM installed successfully ==="
echo "Verify: ls ${INSTALL_PREFIX}/lib/librocshmem* ${INSTALL_PREFIX}/include/rocshmem/"
ls "${INSTALL_PREFIX}/lib/librocshmem"* 2>/dev/null || echo "WARNING: librocshmem not found in ${INSTALL_PREFIX}/lib"
ls "${INSTALL_PREFIX}/include/rocshmem/" 2>/dev/null | head -5 || echo "WARNING: headers not found"
