cmake_minimum_required(VERSION 3.21)
project(rocblasdx VERSION 0.1.0 LANGUAGES CXX HIP)

# -------------------------------------------------------
# rocBLASDx — Header-only device-side GEMM library
# for AMD MI300X (gfx942) / CDNA3
# -------------------------------------------------------

# Interface library (header-only)
add_library(rocblasdx INTERFACE)
add_library(rocblasdx::rocblasdx ALIAS rocblasdx)

target_include_directories(rocblasdx INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Require HIP and C++20
target_compile_features(rocblasdx INTERFACE cxx_std_20)

# Find HIP (when used standalone, not as a subdirectory)
if(NOT TARGET hip::device)
    find_package(hip REQUIRED)
endif()
target_link_libraries(rocblasdx INTERFACE hip::device)

# Default offload arch for MI300X
if(NOT DEFINED ROCBLASDX_OFFLOAD_ARCH)
    set(ROCBLASDX_OFFLOAD_ARCH "gfx942" CACHE STRING "GPU offload architecture")
endif()

# -------------------------------------------------------
# Tests (optional, only when building standalone)
# -------------------------------------------------------
option(ROCBLASDX_BUILD_TESTS "Build rocBLASDx tests" OFF)

if(ROCBLASDX_BUILD_TESTS)
    enable_testing()

    add_executable(test_rocblasdx_gemm tests/test_gemm.cpp)
    set_source_files_properties(tests/test_gemm.cpp PROPERTIES LANGUAGE HIP)
    target_link_libraries(test_rocblasdx_gemm PRIVATE rocblasdx::rocblasdx)
    target_compile_options(test_rocblasdx_gemm PRIVATE
        --offload-arch=${ROCBLASDX_OFFLOAD_ARCH}
    )
    add_test(NAME rocblasdx_gemm COMMAND test_rocblasdx_gemm)
endif()

# -------------------------------------------------------
# Install
# -------------------------------------------------------
include(GNUInstallDirs)

install(TARGETS rocblasdx
    EXPORT rocblasdx-targets
)

install(DIRECTORY include/rocblasdx
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT rocblasdx-targets
    FILE rocblasdx-targets.cmake
    NAMESPACE rocblasdx::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rocblasdx
)
