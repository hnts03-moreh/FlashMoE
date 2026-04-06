/*
 * rocshmem-python — pybind11 bindings for ROCSHMEM host API
 *
 * Provides the `rocshmem.core` Python module, matching the nvshmem4py
 * interface that FlashMoE's cb.py expects.
 *
 * Host-only API: init, finalize, my_pe, n_pes, malloc, free,
 * barrier_all, sync_all, ptr, init_status.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <cstddef>
#include <stdexcept>

#include <rocshmem/rocshmem.hpp>
#include <hip/hip_runtime.h>

namespace py = pybind11;

// Track init state (ROCSHMEM has no query API)
static bool g_initialized = false;

// ---------------------------------------------------------------------------
// InitStatus enum — matches nvshmem4py convention
// ---------------------------------------------------------------------------
enum class InitStatus {
    STATUS_NOT_INITIALIZED = 0,
    STATUS_IS_INITIALIZED = 1,
};

// ---------------------------------------------------------------------------
// Host API wrappers
// ---------------------------------------------------------------------------

static InitStatus init_status() {
    return g_initialized ? InitStatus::STATUS_IS_INITIALIZED
                         : InitStatus::STATUS_NOT_INITIALIZED;
}

static void rocshmem_py_init() {
    if (g_initialized) return;
    rocshmem::rocshmem_init();
    g_initialized = true;
}

static void rocshmem_py_finalize() {
    if (!g_initialized) return;
    rocshmem::rocshmem_finalize();
    g_initialized = false;
}

static int rocshmem_py_my_pe() {
    if (!g_initialized)
        throw std::runtime_error("rocshmem not initialized");
    return rocshmem::rocshmem_my_pe();
}

static int rocshmem_py_n_pes() {
    if (!g_initialized)
        throw std::runtime_error("rocshmem not initialized");
    return rocshmem::rocshmem_n_pes();
}

static uintptr_t rocshmem_py_malloc(size_t size) {
    if (!g_initialized)
        throw std::runtime_error("rocshmem not initialized");
    void* ptr = rocshmem::rocshmem_malloc(size);
    if (!ptr)
        throw std::runtime_error("rocshmem_malloc failed");
    return reinterpret_cast<uintptr_t>(ptr);
}

static void rocshmem_py_free(uintptr_t ptr) {
    if (!g_initialized)
        throw std::runtime_error("rocshmem not initialized");
    rocshmem::rocshmem_free(reinterpret_cast<void*>(ptr));
}

static uintptr_t rocshmem_py_calloc(size_t count, size_t size) {
    if (!g_initialized)
        throw std::runtime_error("rocshmem not initialized");
    void* ptr = rocshmem::rocshmem_malloc(count * size);
    if (!ptr)
        throw std::runtime_error("rocshmem_calloc failed");
    (void)hipMemset(ptr, 0, count * size);
    return reinterpret_cast<uintptr_t>(ptr);
}

static void rocshmem_py_barrier_all() {
    if (!g_initialized)
        throw std::runtime_error("rocshmem not initialized");
    rocshmem::rocshmem_barrier_all();
}

static void rocshmem_py_sync_all() {
    if (!g_initialized)
        throw std::runtime_error("rocshmem not initialized");
    rocshmem::rocshmem_sync_all();
}

static uintptr_t rocshmem_py_ptr(uintptr_t dest, int pe) {
    if (!g_initialized)
        throw std::runtime_error("rocshmem not initialized");
    void* result = rocshmem::rocshmem_ptr(reinterpret_cast<const void*>(dest), pe);
    return reinterpret_cast<uintptr_t>(result);
}

// ---------------------------------------------------------------------------
// pybind11 module
// ---------------------------------------------------------------------------

PYBIND11_MODULE(core, m) {
    m.doc() = "ROCSHMEM Python bindings (host API) — drop-in replacement for nvshmem4py";

    // InitStatus enum
    py::enum_<InitStatus>(m, "InitStatus")
        .value("STATUS_NOT_INITIALIZED", InitStatus::STATUS_NOT_INITIALIZED)
        .value("STATUS_IS_INITIALIZED", InitStatus::STATUS_IS_INITIALIZED)
        .export_values();

    // Host API
    m.def("init", &rocshmem_py_init,
          "Initialize ROCSHMEM (uses MPI internally)");
    m.def("finalize", &rocshmem_py_finalize,
          "Finalize ROCSHMEM");
    m.def("init_status", &init_status,
          "Query initialization status");
    m.def("my_pe", &rocshmem_py_my_pe,
          "Get this PE's rank");
    m.def("n_pes", &rocshmem_py_n_pes,
          "Get total number of PEs");
    m.def("malloc", &rocshmem_py_malloc, py::arg("size"),
          "Allocate symmetric heap memory, returns device pointer as int");
    m.def("calloc", &rocshmem_py_calloc, py::arg("count"), py::arg("size"),
          "Allocate and zero symmetric heap memory");
    m.def("free", &rocshmem_py_free, py::arg("ptr"),
          "Free symmetric heap memory");
    m.def("barrier_all", &rocshmem_py_barrier_all,
          "Global barrier across all PEs");
    m.def("sync_all", &rocshmem_py_sync_all,
          "Global sync across all PEs");
    m.def("ptr", &rocshmem_py_ptr, py::arg("dest"), py::arg("pe"),
          "Get remote pointer for symmetric address on target PE");
}
