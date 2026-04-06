cmake_minimum_required(VERSION 3.24)
project(rocshmem_python LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    # Fallback: use pybind11 from pip
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _pybind11_rc
    )
    if(_pybind11_rc EQUAL 0)
        find_package(pybind11 CONFIG REQUIRED HINTS "${pybind11_DIR}")
    else()
        message(FATAL_ERROR "pybind11 not found. Install: pip install pybind11")
    endif()
endif()

find_package(hip REQUIRED)

# Find ROCSHMEM
list(APPEND CMAKE_PREFIX_PATH /opt/rocm)
find_package(rocshmem QUIET)
if(NOT TARGET roc::rocshmem)
    find_library(ROCSHMEM_LIB rocshmem HINTS ENV ROCM_PATH /opt/rocm PATH_SUFFIXES lib)
    if(NOT ROCSHMEM_LIB)
        message(FATAL_ERROR "ROCSHMEM not found. Build and install ROCSHMEM first.")
    endif()
endif()

# Find MPI (required by ROCSHMEM)
find_package(MPI REQUIRED COMPONENTS CXX)

# Build the Python module
pybind11_add_module(core src/rocshmem_bindings.cpp)

target_include_directories(core PRIVATE
    ${CMAKE_PREFIX_PATH}/include
    /opt/rocm/include
)

# Link ROCSHMEM host lib directly (avoid -fgpu-rdc from roc::rocshmem target)
find_library(ROCSHMEM_LIB rocshmem HINTS ENV ROCM_PATH /opt/rocm PATH_SUFFIXES lib)
if(ROCSHMEM_LIB)
    target_link_libraries(core PRIVATE ${ROCSHMEM_LIB})
elseif(TARGET roc::rocshmem)
    get_target_property(_rocshmem_loc roc::rocshmem IMPORTED_LOCATION)
    target_link_libraries(core PRIVATE ${_rocshmem_loc})
else()
    message(FATAL_ERROR "Cannot find ROCSHMEM library")
endif()

target_link_libraries(core PRIVATE
    hip::host
    MPI::MPI_CXX
)

# Install into rocshmem/ package directory
install(TARGETS core DESTINATION rocshmem)
