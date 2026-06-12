/**
 * @file onemkl_trait_impl.cpp
 * @brief Intel oneAPI oneMKL GPU BLAS backend implementation
 * 
 * Implements the fb_gpu_backend_trait interface for Intel GPUs using oneMKL.
 * 
 * Supported Hardware:
 * - Intel Arc GPUs (A770, A750, A380)
 * - Intel Data Center Flex/Max GPUs
 * - Intel Integrated Graphics (Iris Xe, UHD 11th gen+)
 * 
 * Requirements:
 * - Intel oneAPI Base Toolkit 2024.0+
 * - oneMKL library
 * - DPC++ compiler (icpx) or standard C++ compiler with SYCL support
 * 
 * Tested on:
 * - Windows 11 with Intel Arc A770 16GB
 * - Ubuntu 22.04 with Intel Data Center GPU Flex 170
 */

#include "faster-blaster/gpu_backend_trait.h"

// oneMKL includes
#include <CL/sycl.hpp>                    // SYCL runtime
#include <oneapi/mkl.hpp>                 // oneMKL BLAS/LAPACK
#include <oneapi/mkl/blas.hpp>
#include <oneapi/mkl/lapack.hpp>

#include <cstdio>
#include <cstring>
#include <vector>

/* ============================================================================
 * Type Conversions and Utilities
 * ========================================================================== */

namespace {

/**
 * @brief Internal context structure for oneMKL backend
 * 
 * Holds SYCL queue and device information for Intel GPU operations
 */
struct onemkl_context_t {
    sycl::queue* sycl_queue;     // SYCL queue for command submission
    sycl::device sycl_device;    // Selected Intel GPU device
    int device_id;               // Device index (for multi-GPU)
    
    onemkl_context_t() : sycl_queue(nullptr), device_id(-1) {}
};

/**
 * @brief Convert char uplo to oneMKL uplo enum
 */
inline oneapi::mkl::uplo convert_uplo(char uplo) {
    return (uplo == 'U' || uplo == 'u') 
        ? oneapi::mkl::uplo::upper 
        : oneapi::mkl::uplo::lower;
}

/**
 * @brief Convert char transpose to oneMKL transpose enum
 */
inline oneapi::mkl::transpose convert_transpose(char trans) {
    switch (trans) {
        case 'N': case 'n': return oneapi::mkl::transpose::nontrans;
        case 'T': case 't': return oneapi::mkl::transpose::trans;
        case 'C': case 'c': return oneapi::mkl::transpose::conjtrans;
        default: return oneapi::mkl::transpose::nontrans;
    }
}

/**
 * @brief Convert char side to oneMKL side enum
 */
inline oneapi::mkl::side convert_side(char side) {
    return (side == 'L' || side == 'l')
        ? oneapi::mkl::side::left
        : oneapi::mkl::side::right;
}

/**
 * @brief Convert char diag to oneMKL diag enum
 */
inline oneapi::mkl::diag convert_diag(char diag) {
    return (diag == 'U' || diag == 'u')
        ? oneapi::mkl::diag::unit
        : oneapi::mkl::diag::nonunit;
}

} // anonymous namespace

/* ============================================================================
 * Lifecycle Operations
 * ========================================================================== */

extern "C" {

static int onemkl_init_impl(int device_id, void** backend_handle) {
    try {
        onemkl_context_t* ctx = new onemkl_context_t();
        ctx->device_id = device_id;
        
        // Get all GPU devices
        auto platforms = sycl::platform::get_platforms();
        std::vector<sycl::device> gpu_devices;
        
        for (const auto& platform : platforms) {
            auto devices = platform.get_devices(sycl::info::device_type::gpu);
            gpu_devices.insert(gpu_devices.end(), devices.begin(), devices.end());
        }
        
        if (gpu_devices.empty()) {
            fprintf(stderr, "oneMKL: No GPU devices found\n");
            delete ctx;
            return -1;
        }
        
        if (device_id < 0 || device_id >= static_cast<int>(gpu_devices.size())) {
            fprintf(stderr, "oneMKL: Invalid device_id %d (found %zu devices)\n", 
                    device_id, gpu_devices.size());
            delete ctx;
            return -1;
        }
        
        // Select the requested device
        ctx->sycl_device = gpu_devices[device_id];
        
        // Create SYCL queue for this device
        ctx->sycl_queue = new sycl::queue(ctx->sycl_device);
        
        *backend_handle = ctx;
        return 0;
        
    } catch (const sycl::exception& e) {
        fprintf(stderr, "oneMKL init error: %s\n", e.what());
        return -1;
    }
}

static void onemkl_shutdown_impl(void* handle) {
    if (!handle) return;
    
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    
    if (ctx->sycl_queue) {
        try {
            ctx->sycl_queue->wait();  // Wait for all operations to complete
        } catch (...) {
            // Ignore errors during shutdown
        }
        delete ctx->sycl_queue;
    }
    
    delete ctx;
}

/* ============================================================================
 * Device Properties
 * ========================================================================== */

static int onemkl_get_device_properties_impl(void* handle, int device_id,
                                              char* name, size_t name_len,
                                              size_t* total_memory) {
    if (!handle) return -1;
    
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    
    try {
        if (name && name_len > 0) {
            std::string device_name = ctx->sycl_device.get_info<sycl::info::device::name>();
            strncpy_s(name, name_len, device_name.c_str(), _TRUNCATE);
        }
        
        if (total_memory) {
            *total_memory = ctx->sycl_device.get_info<sycl::info::device::global_mem_size>();
        }
        
        return 0;
    } catch (const sycl::exception& e) {
        fprintf(stderr, "oneMKL get_device_properties error: %s\n", e.what());
        return -1;
    }
}

/* ============================================================================
 * Memory Management
 * ========================================================================== */

static int onemkl_malloc_impl(void* handle, fb_gpu_ptr_t* ptr, size_t size) {
    if (!handle || !ptr) return -1;
    
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    
    try {
        // Allocate device USM (Unified Shared Memory)
        void* device_ptr = sycl::malloc_device(size, *ctx->sycl_queue);
        if (!device_ptr) return -1;
        
        *ptr = device_ptr;
        return 0;
    } catch (const sycl::exception& e) {
        fprintf(stderr, "oneMKL malloc error: %s\n", e.what());
        return -1;
    }
}

static void onemkl_free_impl(void* handle, fb_gpu_ptr_t ptr) {
    if (!handle || !ptr) return;
    
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    
    try {
        sycl::free(ptr, *ctx->sycl_queue);
    } catch (const sycl::exception& e) {
        fprintf(stderr, "oneMKL free error: %s\n", e.what());
    }
}

static int onemkl_memcpy_h2d_impl(void* handle, fb_gpu_ptr_t dst, 
                                   const void* src, size_t size) {
    if (!handle) return -1;
    
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    
    try {
        ctx->sycl_queue->memcpy(dst, src, size).wait();
        return 0;
    } catch (const sycl::exception& e) {
        fprintf(stderr, "oneMKL memcpy H2D error: %s\n", e.what());
        return -1;
    }
}

static int onemkl_memcpy_d2h_impl(void* handle, void* dst, 
                                   fb_gpu_ptr_t src, size_t size) {
    if (!handle) return -1;
    
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    
    try {
        ctx->sycl_queue->memcpy(dst, src, size).wait();
        return 0;
    } catch (const sycl::exception& e) {
        fprintf(stderr, "oneMKL memcpy D2H error: %s\n", e.what());
        return -1;
    }
}

static int onemkl_memcpy_d2d_impl(void* handle, fb_gpu_ptr_t dst, 
                                   fb_gpu_ptr_t src, size_t size) {
    if (!handle) return -1;
    
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    
    try {
        ctx->sycl_queue->memcpy(dst, src, size).wait();
        return 0;
    } catch (const sycl::exception& e) {
        fprintf(stderr, "oneMKL memcpy D2D error: %s\n", e.what());
        return -1;
    }
}

/* ============================================================================
 * Stream Operations
 * ========================================================================== */

static int onemkl_stream_create_impl(void* handle, fb_gpu_stream_t* stream) {
    if (!handle || !stream) return -1;
    
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    
    try {
        // Create a new SYCL queue (analogous to CUDA stream)
        sycl::queue* new_queue = new sycl::queue(ctx->sycl_device);
        *stream = static_cast<fb_gpu_stream_t>(new_queue);
        return 0;
    } catch (const sycl::exception& e) {
        fprintf(stderr, "oneMKL stream_create error: %s\n", e.what());
        return -1;
    }
}

static void onemkl_stream_destroy_impl(void* handle, fb_gpu_stream_t stream) {
    if (!stream) return;
    
    try {
        sycl::queue* queue = static_cast<sycl::queue*>(stream);
        queue->wait();  // Wait for all operations
        delete queue;
    } catch (const sycl::exception& e) {
        fprintf(stderr, "oneMKL stream_destroy error: %s\n", e.what());
    }
}

static int onemkl_stream_synchronize_impl(void* handle, fb_gpu_stream_t stream) {
    if (!stream) return -1;
    
    try {
        sycl::queue* queue = static_cast<sycl::queue*>(stream);
        queue->wait();
        return 0;
    } catch (const sycl::exception& e) {
        fprintf(stderr, "oneMKL stream_synchronize error: %s\n", e.what());
        return -1;
    }
}

/* ============================================================================
 * BLAS Level 1 Operations - Vector-Vector Operations
 * ========================================================================== */

// SAXPY - Y = alpha*X + Y
static void onemkl_saxpy_impl(void* handle, fb_gpu_stream_t stream,
                               int n, float alpha, fb_gpu_ptr_t x, int incx,
                               fb_gpu_ptr_t y, int incy) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::axpy(*queue, n, alpha,
                                static_cast<const float*>(x), incx,
                                static_cast<float*>(y), incy);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL saxpy error: %s\n", e.what());
    }
}

static void onemkl_daxpy_impl(void* handle, fb_gpu_stream_t stream,
                               int n, double alpha, fb_gpu_ptr_t x, int incx,
                               fb_gpu_ptr_t y, int incy) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::axpy(*queue, n, alpha,
                                static_cast<const double*>(x), incx,
                                static_cast<double*>(y), incy);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL daxpy error: %s\n", e.what());
    }
}

static void onemkl_caxpy_impl(void* handle, fb_gpu_stream_t stream,
                               int n, const void* alpha, fb_gpu_ptr_t x, int incx,
                               fb_gpu_ptr_t y, int incy) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::axpy(*queue, n, *static_cast<const std::complex<float>*>(alpha),
                                static_cast<const std::complex<float>*>(x), incx,
                                static_cast<std::complex<float>*>(y), incy);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL caxpy error: %s\n", e.what());
    }
}

static void onemkl_zaxpy_impl(void* handle, fb_gpu_stream_t stream,
                               int n, const void* alpha, fb_gpu_ptr_t x, int incx,
                               fb_gpu_ptr_t y, int incy) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::axpy(*queue, n, *static_cast<const std::complex<double>*>(alpha),
                                static_cast<const std::complex<double>*>(x), incx,
                                static_cast<std::complex<double>*>(y), incy);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL zaxpy error: %s\n", e.what());
    }
}

/* ============================================================================
 * BLAS Level 1 Operations - Scale (SCAL)
 * ========================================================================== */

static void onemkl_sscal_impl(void* handle, fb_gpu_stream_t stream, int n, float alpha, fb_gpu_ptr_t x, int incx) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::scal(*queue, n, alpha, static_cast<float*>(x), incx);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL sscal error: %s\n", e.what());
    }
}

static void onemkl_dscal_impl(void* handle, fb_gpu_stream_t stream, int n, double alpha, fb_gpu_ptr_t x, int incx) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::scal(*queue, n, alpha, static_cast<double*>(x), incx);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL dscal error: %s\n", e.what());
    }
}

static void onemkl_cscal_impl(void* handle, fb_gpu_stream_t stream, int n, const void* alpha, fb_gpu_ptr_t x, int incx) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::scal(*queue, n, *static_cast<const std::complex<float>*>(alpha), static_cast<std::complex<float>*>(x), incx);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL cscal error: %s\n", e.what());
    }
}

static void onemkl_zscal_impl(void* handle, fb_gpu_stream_t stream, int n, const void* alpha, fb_gpu_ptr_t x, int incx) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::scal(*queue, n, *static_cast<const std::complex<double>*>(alpha), static_cast<std::complex<double>*>(x), incx);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL zscal error: %s\n", e.what());
    }
}

static void onemkl_csscal_impl(void* handle, fb_gpu_stream_t stream, int n, float alpha, fb_gpu_ptr_t x, int incx) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::scal(*queue, n, alpha, static_cast<std::complex<float>*>(x), incx);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL csscal error: %s\n", e.what());
    }
}

static void onemkl_zdscal_impl(void* handle, fb_gpu_stream_t stream, int n, double alpha, fb_gpu_ptr_t x, int incx) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::scal(*queue, n, alpha, static_cast<std::complex<double>*>(x), incx);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL zdscal error: %s\n", e.what());
    }
}

/* ============================================================================
 * BLAS Level 1 Operations - Copy
 * ========================================================================== */

static void onemkl_scopy_impl(void* handle, fb_gpu_stream_t stream, int n, fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::copy(*queue, n, static_cast<const float*>(x), incx, static_cast<float*>(y), incy);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL scopy error: %s\n", e.what());
    }
}

static void onemkl_dcopy_impl(void* handle, fb_gpu_stream_t stream, int n, fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::copy(*queue, n, static_cast<const double*>(x), incx, static_cast<double*>(y), incy);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL dcopy error: %s\n", e.what());
    }
}

static void onemkl_ccopy_impl(void* handle, fb_gpu_stream_t stream, int n, fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::copy(*queue, n, static_cast<const std::complex<float>*>(x), incx, static_cast<std::complex<float>*>(y), incy);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL ccopy error: %s\n", e.what());
    }
}

static void onemkl_zcopy_impl(void* handle, fb_gpu_stream_t stream, int n, fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::copy(*queue, n, static_cast<const std::complex<double>*>(x), incx, static_cast<std::complex<double>*>(y), incy);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL zcopy error: %s\n", e.what());
    }
}

/* ============================================================================
 * BLAS Level 1 Operations - Swap
 * ========================================================================== */

static void onemkl_sswap_impl(void* handle, fb_gpu_stream_t stream, int n, fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::swap(*queue, n, static_cast<float*>(x), incx, static_cast<float*>(y), incy);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL sswap error: %s\n", e.what());
    }
}

static void onemkl_dswap_impl(void* handle, fb_gpu_stream_t stream, int n, fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::swap(*queue, n, static_cast<double*>(x), incx, static_cast<double*>(y), incy);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL dswap error: %s\n", e.what());
    }
}

static void onemkl_cswap_impl(void* handle, fb_gpu_stream_t stream, int n, fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::swap(*queue, n, static_cast<std::complex<float>*>(x), incx, static_cast<std::complex<float>*>(y), incy);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL cswap error: %s\n", e.what());
    }
}

static void onemkl_zswap_impl(void* handle, fb_gpu_stream_t stream, int n, fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::swap(*queue, n, static_cast<std::complex<double>*>(x), incx, static_cast<std::complex<double>*>(y), incy);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL zswap error: %s\n", e.what());
    }
}

/* ============================================================================
 * BLAS Level 1 Operations - Dot Product
 * ========================================================================== */

static float onemkl_sdot_impl(void* handle, fb_gpu_stream_t stream, int n, fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    float result = 0.0f;
    try {
        oneapi::mkl::blas::dot(*queue, n, static_cast<const float*>(x), incx, static_cast<const float*>(y), incy, &result);
        queue->wait();
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL sdot error: %s\n", e.what());
    }
    return result;
}

static double onemkl_ddot_impl(void* handle, fb_gpu_stream_t stream, int n, fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    double result = 0.0;
    try {
        oneapi::mkl::blas::dot(*queue, n, static_cast<const double*>(x), incx, static_cast<const double*>(y), incy, &result);
        queue->wait();
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL ddot error: %s\n", e.what());
    }
    return result;
}

static void onemkl_cdotu_impl(void* handle, fb_gpu_stream_t stream, int n, fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy, fb_gpu_ptr_t result) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::dotu(*queue, n, static_cast<const std::complex<float>*>(x), incx, static_cast<const std::complex<float>*>(y), incy, static_cast<std::complex<float>*>(result));
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL cdotu error: %s\n", e.what());
    }
}

static void onemkl_zdotu_impl(void* handle, fb_gpu_stream_t stream, int n, fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy, fb_gpu_ptr_t result) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::dotu(*queue, n, static_cast<const std::complex<double>*>(x), incx, static_cast<const std::complex<double>*>(y), incy, static_cast<std::complex<double>*>(result));
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL zdotu error: %s\n", e.what());
    }
}

static void onemkl_cdotc_impl(void* handle, fb_gpu_stream_t stream, int n, fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy, fb_gpu_ptr_t result) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::dotc(*queue, n, static_cast<const std::complex<float>*>(x), incx, static_cast<const std::complex<float>*>(y), incy, static_cast<std::complex<float>*>(result));
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL cdotc error: %s\n", e.what());
    }
}

static void onemkl_zdotc_impl(void* handle, fb_gpu_stream_t stream, int n, fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy, fb_gpu_ptr_t result) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::dotc(*queue, n, static_cast<const std::complex<double>*>(x), incx, static_cast<const std::complex<double>*>(y), incy, static_cast<std::complex<double>*>(result));
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL zdotc error: %s\n", e.what());
    }
}

/* ============================================================================
 * BLAS Level 1 Operations - Norms and Sums
 * ========================================================================== */

static float onemkl_snrm2_impl(void* handle, fb_gpu_stream_t stream, int n, fb_gpu_ptr_t x, int incx) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    float result = 0.0f;
    try {
        oneapi::mkl::blas::nrm2(*queue, n, static_cast<const float*>(x), incx, &result);
        queue->wait();
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL snrm2 error: %s\n", e.what());
    }
    return result;
}

static double onemkl_dnrm2_impl(void* handle, fb_gpu_stream_t stream, int n, fb_gpu_ptr_t x, int incx) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    double result = 0.0;
    try {
        oneapi::mkl::blas::nrm2(*queue, n, static_cast<const double*>(x), incx, &result);
        queue->wait();
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL dnrm2 error: %s\n", e.what());
    }
    return result;
}

static float onemkl_scnrm2_impl(void* handle, fb_gpu_stream_t stream, int n, fb_gpu_ptr_t x, int incx) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    float result = 0.0f;
    try {
        oneapi::mkl::blas::nrm2(*queue, n, static_cast<const std::complex<float>*>(x), incx, &result);
        queue->wait();
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL scnrm2 error: %s\n", e.what());
    }
    return result;
}

static double onemkl_dznrm2_impl(void* handle, fb_gpu_stream_t stream, int n, fb_gpu_ptr_t x, int incx) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    double result = 0.0;
    try {
        oneapi::mkl::blas::nrm2(*queue, n, static_cast<const std::complex<double>*>(x), incx, &result);
        queue->wait();
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL dznrm2 error: %s\n", e.what());
    }
    return result;
}

static float onemkl_sasum_impl(void* handle, fb_gpu_stream_t stream, int n, fb_gpu_ptr_t x, int incx) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    float result = 0.0f;
    try {
        oneapi::mkl::blas::asum(*queue, n, static_cast<const float*>(x), incx, &result);
        queue->wait();
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL sasum error: %s\n", e.what());
    }
    return result;
}

static double onemkl_dasum_impl(void* handle, fb_gpu_stream_t stream, int n, fb_gpu_ptr_t x, int incx) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    double result = 0.0;
    try {
        oneapi::mkl::blas::asum(*queue, n, static_cast<const double*>(x), incx, &result);
        queue->wait();
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL dasum error: %s\n", e.what());
    }
    return result;
}

static float onemkl_scasum_impl(void* handle, fb_gpu_stream_t stream, int n, fb_gpu_ptr_t x, int incx) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    float result = 0.0f;
    try {
        oneapi::mkl::blas::asum(*queue, n, static_cast<const std::complex<float>*>(x), incx, &result);
        queue->wait();
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL scasum error: %s\n", e.what());
    }
    return result;
}

static double onemkl_dzasum_impl(void* handle, fb_gpu_stream_t stream, int n, fb_gpu_ptr_t x, int incx) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    double result = 0.0;
    try {
        oneapi::mkl::blas::asum(*queue, n, static_cast<const std::complex<double>*>(x), incx, &result);
        queue->wait();
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL dzasum error: %s\n", e.what());
    }
    return result;
}

static int onemkl_isamax_impl(void* handle, fb_gpu_stream_t stream, int n, fb_gpu_ptr_t x, int incx) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    int64_t result = 0;
    try {
        oneapi::mkl::blas::iamax(*queue, n, static_cast<const float*>(x), incx, &result);
        queue->wait();
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL isamax error: %s\n", e.what());
    }
    return static_cast<int>(result);
}

static int onemkl_idamax_impl(void* handle, fb_gpu_stream_t stream, int n, fb_gpu_ptr_t x, int incx) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    int64_t result = 0;
    try {
        oneapi::mkl::blas::iamax(*queue, n, static_cast<const double*>(x), incx, &result);
        queue->wait();
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL idamax error: %s\n", e.what());
    }
    return static_cast<int>(result);
}

static int onemkl_icamax_impl(void* handle, fb_gpu_stream_t stream, int n, fb_gpu_ptr_t x, int incx) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    int64_t result = 0;
    try {
        oneapi::mkl::blas::iamax(*queue, n, static_cast<const std::complex<float>*>(x), incx, &result);
        queue->wait();
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL icamax error: %s\n", e.what());
    }
    return static_cast<int>(result);
}

static int onemkl_izamax_impl(void* handle, fb_gpu_stream_t stream, int n, fb_gpu_ptr_t x, int incx) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    int64_t result = 0;
    try {
        oneapi::mkl::blas::iamax(*queue, n, static_cast<const std::complex<double>*>(x), incx, &result);
        queue->wait();
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL izamax error: %s\n", e.what());
    }
    return static_cast<int>(result);
}

/* ============================================================================
 * BLAS Level 1 Operations - Rotation Operations
 * ========================================================================== */

static void onemkl_srotg_impl(void* handle, fb_gpu_stream_t stream, fb_gpu_ptr_t a, fb_gpu_ptr_t b, fb_gpu_ptr_t c, fb_gpu_ptr_t s) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::rotg(*queue, static_cast<float*>(a), static_cast<float*>(b), static_cast<float*>(c), static_cast<float*>(s));
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL srotg error: %s\n", e.what());
    }
}

static void onemkl_drotg_impl(void* handle, fb_gpu_stream_t stream, fb_gpu_ptr_t a, fb_gpu_ptr_t b, fb_gpu_ptr_t c, fb_gpu_ptr_t s) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::rotg(*queue, static_cast<double*>(a), static_cast<double*>(b), static_cast<double*>(c), static_cast<double*>(s));
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL drotg error: %s\n", e.what());
    }
}

static void onemkl_srot_impl(void* handle, fb_gpu_stream_t stream, int n, fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy, float c, float s) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::rot(*queue, n, static_cast<float*>(x), incx, static_cast<float*>(y), incy, c, s);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL srot error: %s\n", e.what());
    }
}

static void onemkl_drot_impl(void* handle, fb_gpu_stream_t stream, int n, fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy, double c, double s) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::rot(*queue, n, static_cast<double*>(x), incx, static_cast<double*>(y), incy, c, s);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL drot error: %s\n", e.what());
    }
}

static void onemkl_srotm_impl(void* handle, fb_gpu_stream_t stream, int n, fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy, fb_gpu_ptr_t param) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::rotm(*queue, n, static_cast<float*>(x), incx, static_cast<float*>(y), incy, static_cast<const float*>(param));
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL srotm error: %s\n", e.what());
    }
}

static void onemkl_drotm_impl(void* handle, fb_gpu_stream_t stream, int n, fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy, fb_gpu_ptr_t param) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::rotm(*queue, n, static_cast<double*>(x), incx, static_cast<double*>(y), incy, static_cast<const double*>(param));
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL drotm error: %s\n", e.what());
    }
}

static void onemkl_srotmg_impl(void* handle, fb_gpu_stream_t stream, fb_gpu_ptr_t d1, fb_gpu_ptr_t d2, fb_gpu_ptr_t x1, fb_gpu_ptr_t y1, fb_gpu_ptr_t param) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::rotmg(*queue, static_cast<float*>(d1), static_cast<float*>(d2), static_cast<float*>(x1), static_cast<const float*>(y1), static_cast<float*>(param));
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL srotmg error: %s\n", e.what());
    }
}

static void onemkl_drotmg_impl(void* handle, fb_gpu_stream_t stream, fb_gpu_ptr_t d1, fb_gpu_ptr_t d2, fb_gpu_ptr_t x1, fb_gpu_ptr_t y1, fb_gpu_ptr_t param) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::rotmg(*queue, static_cast<double*>(d1), static_cast<double*>(d2), static_cast<double*>(x1), static_cast<const double*>(y1), static_cast<double*>(param));
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL drotmg error: %s\n", e.what());
    }
}

/* ============================================================================
 * BLAS Level 2 Operations - Matrix-Vector Operations
 * ========================================================================== */

// Example: SGEMV - Single precision matrix-vector multiply
static void onemkl_sgemv_impl(void* handle, fb_gpu_stream_t stream,
                               char trans, int m, int n, float alpha,
                               fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx,
                               float beta, fb_gpu_ptr_t y, int incy) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    
    try {
        oneapi::mkl::blas::gemv(*queue, convert_transpose(trans), m, n,
                                alpha,
                                static_cast<const float*>(a), lda,
                                static_cast<const float*>(x), incx,
                                beta,
                                static_cast<float*>(y), incy);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL sgemv error: %s\n", e.what());
    }
}

static void onemkl_dgemv_impl(void* handle, fb_gpu_stream_t stream, char trans, int m, int n, double alpha, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx, double beta, fb_gpu_ptr_t y, int incy) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::gemv(*queue, convert_transpose(trans), m, n, alpha, static_cast<const double*>(a), lda, static_cast<const double*>(x), incx, beta, static_cast<double*>(y), incy);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL dgemv error: %s\n", e.what());
    }
}

static void onemkl_cgemv_impl(void* handle, fb_gpu_stream_t stream, char trans, int m, int n, const void* alpha, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx, const void* beta, fb_gpu_ptr_t y, int incy) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::gemv(*queue, convert_transpose(trans), m, n, *static_cast<const std::complex<float>*>(alpha), static_cast<const std::complex<float>*>(a), lda, static_cast<const std::complex<float>*>(x), incx, *static_cast<const std::complex<float>*>(beta), static_cast<std::complex<float>*>(y), incy);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL cgemv error: %s\n", e.what());
    }
}

static void onemkl_zgemv_impl(void* handle, fb_gpu_stream_t stream, char trans, int m, int n, const void* alpha, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx, const void* beta, fb_gpu_ptr_t y, int incy) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::gemv(*queue, convert_transpose(trans), m, n, *static_cast<const std::complex<double>*>(alpha), static_cast<const std::complex<double>*>(a), lda, static_cast<const std::complex<double>*>(x), incx, *static_cast<const std::complex<double>*>(beta), static_cast<std::complex<double>*>(y), incy);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL zgemv error: %s\n", e.what());
    }
}

/* ============================================================================
 * BLAS Level 2 Operations - Banded Matrix-Vector Operations (GBMV)
 * ========================================================================== */

static void onemkl_sgbmv_impl(void* handle, fb_gpu_stream_t stream, char trans, int m, int n, int kl, int ku, float alpha, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx, float beta, fb_gpu_ptr_t y, int incy) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::gbmv(*queue, convert_transpose(trans), m, n, kl, ku, alpha, static_cast<const float*>(a), lda, static_cast<const float*>(x), incx, beta, static_cast<float*>(y), incy);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL sgbmv error: %s\n", e.what());
    }
}

static void onemkl_dgbmv_impl(void* handle, fb_gpu_stream_t stream, char trans, int m, int n, int kl, int ku, double alpha, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx, double beta, fb_gpu_ptr_t y, int incy) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::gbmv(*queue, convert_transpose(trans), m, n, kl, ku, alpha, static_cast<const double*>(a), lda, static_cast<const double*>(x), incx, beta, static_cast<double*>(y), incy);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL dgbmv error: %s\n", e.what());
    }
}

static void onemkl_cgbmv_impl(void* handle, fb_gpu_stream_t stream, char trans, int m, int n, int kl, int ku, const void* alpha, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx, const void* beta, fb_gpu_ptr_t y, int incy) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::gbmv(*queue, convert_transpose(trans), m, n, kl, ku, *static_cast<const std::complex<float>*>(alpha), static_cast<const std::complex<float>*>(a), lda, static_cast<const std::complex<float>*>(x), incx, *static_cast<const std::complex<float>*>(beta), static_cast<std::complex<float>*>(y), incy);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL cgbmv error: %s\n", e.what());
    }
}

static void onemkl_zgbmv_impl(void* handle, fb_gpu_stream_t stream, char trans, int m, int n, int kl, int ku, const void* alpha, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx, const void* beta, fb_gpu_ptr_t y, int incy) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::gbmv(*queue, convert_transpose(trans), m, n, kl, ku, *static_cast<const std::complex<double>*>(alpha), static_cast<const std::complex<double>*>(a), lda, static_cast<const std::complex<double>*>(x), incx, *static_cast<const std::complex<double>*>(beta), static_cast<std::complex<double>*>(y), incy);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL zgbmv error: %s\n", e.what());
    }
}

/* ============================================================================
 * BLAS Level 2 Operations - Symmetric/Hermitian Matrix-Vector Operations
 * ========================================================================== */

static void onemkl_ssymv_impl(void* handle, fb_gpu_stream_t stream, char uplo, int n, float alpha, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx, float beta, fb_gpu_ptr_t y, int incy) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::symv(*queue, convert_uplo(uplo), n, alpha, static_cast<const float*>(a), lda, static_cast<const float*>(x), incx, beta, static_cast<float*>(y), incy);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL ssymv error: %s\n", e.what());
    }
}

static void onemkl_dsymv_impl(void* handle, fb_gpu_stream_t stream, char uplo, int n, double alpha, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx, double beta, fb_gpu_ptr_t y, int incy) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::symv(*queue, convert_uplo(uplo), n, alpha, static_cast<const double*>(a), lda, static_cast<const double*>(x), incx, beta, static_cast<double*>(y), incy);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL dsymv error: %s\n", e.what());
    }
}

static void onemkl_csymv_impl(void* handle, fb_gpu_stream_t stream, char uplo, int n, const void* alpha, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx, const void* beta, fb_gpu_ptr_t y, int incy) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::symv(*queue, convert_uplo(uplo), n, *static_cast<const std::complex<float>*>(alpha), static_cast<const std::complex<float>*>(a), lda, static_cast<const std::complex<float>*>(x), incx, *static_cast<const std::complex<float>*>(beta), static_cast<std::complex<float>*>(y), incy);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL csymv error: %s\n", e.what());
    }
}

static void onemkl_zsymv_impl(void* handle, fb_gpu_stream_t stream, char uplo, int n, const void* alpha, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx, const void* beta, fb_gpu_ptr_t y, int incy) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::symv(*queue, convert_uplo(uplo), n, *static_cast<const std::complex<double>*>(alpha), static_cast<const std::complex<double>*>(a), lda, static_cast<const std::complex<double>*>(x), incx, *static_cast<const std::complex<double>*>(beta), static_cast<std::complex<double>*>(y), incy);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL zsymv error: %s\n", e.what());
    }
}

static void onemkl_chemv_impl(void* handle, fb_gpu_stream_t stream, char uplo, int n, const void* alpha, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx, const void* beta, fb_gpu_ptr_t y, int incy) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::hemv(*queue, convert_uplo(uplo), n, *static_cast<const std::complex<float>*>(alpha), static_cast<const std::complex<float>*>(a), lda, static_cast<const std::complex<float>*>(x), incx, *static_cast<const std::complex<float>*>(beta), static_cast<std::complex<float>*>(y), incy);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL chemv error: %s\n", e.what());
    }
}

static void onemkl_zhemv_impl(void* handle, fb_gpu_stream_t stream, char uplo, int n, const void* alpha, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx, const void* beta, fb_gpu_ptr_t y, int incy) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::hemv(*queue, convert_uplo(uplo), n, *static_cast<const std::complex<double>*>(alpha), static_cast<const std::complex<double>*>(a), lda, static_cast<const std::complex<double>*>(x), incx, *static_cast<const std::complex<double>*>(beta), static_cast<std::complex<double>*>(y), incy);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL zhemv error: %s\n", e.what());
    }
}

/* ============================================================================
 * BLAS Level 2 Operations - Triangular Matrix-Vector Operations
 * ========================================================================== */

static void onemkl_strmv_impl(void* handle, fb_gpu_stream_t stream, char uplo, char trans, char diag, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::trmv(*queue, convert_uplo(uplo), convert_transpose(trans), convert_diag(diag), n, static_cast<const float*>(a), lda, static_cast<float*>(x), incx);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL strmv error: %s\n", e.what());
    }
}

static void onemkl_dtrmv_impl(void* handle, fb_gpu_stream_t stream, char uplo, char trans, char diag, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::trmv(*queue, convert_uplo(uplo), convert_transpose(trans), convert_diag(diag), n, static_cast<const double*>(a), lda, static_cast<double*>(x), incx);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL dtrmv error: %s\n", e.what());
    }
}

static void onemkl_ctrmv_impl(void* handle, fb_gpu_stream_t stream, char uplo, char trans, char diag, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::trmv(*queue, convert_uplo(uplo), convert_transpose(trans), convert_diag(diag), n, static_cast<const std::complex<float>*>(a), lda, static_cast<std::complex<float>*>(x), incx);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL ctrmv error: %s\n", e.what());
    }
}

static void onemkl_ztrmv_impl(void* handle, fb_gpu_stream_t stream, char uplo, char trans, char diag, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::trmv(*queue, convert_uplo(uplo), convert_transpose(trans), convert_diag(diag), n, static_cast<const std::complex<double>*>(a), lda, static_cast<std::complex<double>*>(x), incx);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL ztrmv error: %s\n", e.what());
    }
}

static void onemkl_strsv_impl(void* handle, fb_gpu_stream_t stream, char uplo, char trans, char diag, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::trsv(*queue, convert_uplo(uplo), convert_transpose(trans), convert_diag(diag), n, static_cast<const float*>(a), lda, static_cast<float*>(x), incx);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL strsv error: %s\n", e.what());
    }
}

static void onemkl_dtrsv_impl(void* handle, fb_gpu_stream_t stream, char uplo, char trans, char diag, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::trsv(*queue, convert_uplo(uplo), convert_transpose(trans), convert_diag(diag), n, static_cast<const double*>(a), lda, static_cast<double*>(x), incx);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL dtrsv error: %s\n", e.what());
    }
}

static void onemkl_ctrsv_impl(void* handle, fb_gpu_stream_t stream, char uplo, char trans, char diag, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::trsv(*queue, convert_uplo(uplo), convert_transpose(trans), convert_diag(diag), n, static_cast<const std::complex<float>*>(a), lda, static_cast<std::complex<float>*>(x), incx);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL ctrsv error: %s\n", e.what());
    }
}

static void onemkl_ztrsv_impl(void* handle, fb_gpu_stream_t stream, char uplo, char trans, char diag, int n, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::trsv(*queue, convert_uplo(uplo), convert_transpose(trans), convert_diag(diag), n, static_cast<const std::complex<double>*>(a), lda, static_cast<std::complex<double>*>(x), incx);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL ztrsv error: %s\n", e.what());
    }
}

/* ============================================================================
 * BLAS Level 2 Operations - Rank-1 Update (GER)
 * ========================================================================== */

static void onemkl_sger_impl(void* handle, fb_gpu_stream_t stream, int m, int n, float alpha, fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy, fb_gpu_ptr_t a, int lda) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::ger(*queue, m, n, alpha, static_cast<const float*>(x), incx, static_cast<const float*>(y), incy, static_cast<float*>(a), lda);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL sger error: %s\n", e.what());
    }
}

static void onemkl_dger_impl(void* handle, fb_gpu_stream_t stream, int m, int n, double alpha, fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy, fb_gpu_ptr_t a, int lda) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::ger(*queue, m, n, alpha, static_cast<const double*>(x), incx, static_cast<const double*>(y), incy, static_cast<double*>(a), lda);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL dger error: %s\n", e.what());
    }
}

static void onemkl_cgeru_impl(void* handle, fb_gpu_stream_t stream, int m, int n, const void* alpha, fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy, fb_gpu_ptr_t a, int lda) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::geru(*queue, m, n, *static_cast<const std::complex<float>*>(alpha), static_cast<const std::complex<float>*>(x), incx, static_cast<const std::complex<float>*>(y), incy, static_cast<std::complex<float>*>(a), lda);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL cgeru error: %s\n", e.what());
    }
}

static void onemkl_zgeru_impl(void* handle, fb_gpu_stream_t stream, int m, int n, const void* alpha, fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy, fb_gpu_ptr_t a, int lda) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::geru(*queue, m, n, *static_cast<const std::complex<double>*>(alpha), static_cast<const std::complex<double>*>(x), incx, static_cast<const std::complex<double>*>(y), incy, static_cast<std::complex<double>*>(a), lda);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL zgeru error: %s\n", e.what());
    }
}

static void onemkl_cgerc_impl(void* handle, fb_gpu_stream_t stream, int m, int n, const void* alpha, fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy, fb_gpu_ptr_t a, int lda) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::gerc(*queue, m, n, *static_cast<const std::complex<float>*>(alpha), static_cast<const std::complex<float>*>(x), incx, static_cast<const std::complex<float>*>(y), incy, static_cast<std::complex<float>*>(a), lda);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL cgerc error: %s\n", e.what());
    }
}

static void onemkl_zgerc_impl(void* handle, fb_gpu_stream_t stream, int m, int n, const void* alpha, fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy, fb_gpu_ptr_t a, int lda) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::gerc(*queue, m, n, *static_cast<const std::complex<double>*>(alpha), static_cast<const std::complex<double>*>(x), incx, static_cast<const std::complex<double>*>(y), incy, static_cast<std::complex<double>*>(a), lda);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL zgerc error: %s\n", e.what());
    }
}

/* ============================================================================
 * BLAS Level 2 Operations - Symmetric/Hermitian Rank-1 Update
 * ========================================================================== */

static void onemkl_ssyr_impl(void* handle, fb_gpu_stream_t stream, char uplo, int n, float alpha, fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t a, int lda) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::syr(*queue, convert_uplo(uplo), n, alpha, static_cast<const float*>(x), incx, static_cast<float*>(a), lda);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL ssyr error: %s\n", e.what());
    }
}

static void onemkl_dsyr_impl(void* handle, fb_gpu_stream_t stream, char uplo, int n, double alpha, fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t a, int lda) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::syr(*queue, convert_uplo(uplo), n, alpha, static_cast<const double*>(x), incx, static_cast<double*>(a), lda);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL dsyr error: %s\n", e.what());
    }
}

static void onemkl_csyr_impl(void* handle, fb_gpu_stream_t stream, char uplo, int n, const void* alpha, fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t a, int lda) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::syr(*queue, convert_uplo(uplo), n, *static_cast<const std::complex<float>*>(alpha), static_cast<const std::complex<float>*>(x), incx, static_cast<std::complex<float>*>(a), lda);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL csyr error: %s\n", e.what());
    }
}

static void onemkl_zsyr_impl(void* handle, fb_gpu_stream_t stream, char uplo, int n, const void* alpha, fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t a, int lda) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::syr(*queue, convert_uplo(uplo), n, *static_cast<const std::complex<double>*>(alpha), static_cast<const std::complex<double>*>(x), incx, static_cast<std::complex<double>*>(a), lda);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL zsyr error: %s\n", e.what());
    }
}

static void onemkl_cher_impl(void* handle, fb_gpu_stream_t stream, char uplo, int n, float alpha, fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t a, int lda) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::her(*queue, convert_uplo(uplo), n, alpha, static_cast<const std::complex<float>*>(x), incx, static_cast<std::complex<float>*>(a), lda);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL cher error: %s\n", e.what());
    }
}

static void onemkl_zher_impl(void* handle, fb_gpu_stream_t stream, char uplo, int n, double alpha, fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t a, int lda) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::her(*queue, convert_uplo(uplo), n, alpha, static_cast<const std::complex<double>*>(x), incx, static_cast<std::complex<double>*>(a), lda);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL zher error: %s\n", e.what());
    }
}

/* ============================================================================
 * BLAS Level 2 Operations - Symmetric/Hermitian Rank-2 Update
 * ========================================================================== */

static void onemkl_ssyr2_impl(void* handle, fb_gpu_stream_t stream, char uplo, int n, float alpha, fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy, fb_gpu_ptr_t a, int lda) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::syr2(*queue, convert_uplo(uplo), n, alpha, static_cast<const float*>(x), incx, static_cast<const float*>(y), incy, static_cast<float*>(a), lda);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL ssyr2 error: %s\n", e.what());
    }
}

static void onemkl_dsyr2_impl(void* handle, fb_gpu_stream_t stream, char uplo, int n, double alpha, fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy, fb_gpu_ptr_t a, int lda) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::syr2(*queue, convert_uplo(uplo), n, alpha, static_cast<const double*>(x), incx, static_cast<const double*>(y), incy, static_cast<double*>(a), lda);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL dsyr2 error: %s\n", e.what());
    }
}

static void onemkl_csyr2_impl(void* handle, fb_gpu_stream_t stream, char uplo, int n, const void* alpha, fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy, fb_gpu_ptr_t a, int lda) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::syr2(*queue, convert_uplo(uplo), n, *static_cast<const std::complex<float>*>(alpha), static_cast<const std::complex<float>*>(x), incx, static_cast<const std::complex<float>*>(y), incy, static_cast<std::complex<float>*>(a), lda);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL csyr2 error: %s\n", e.what());
    }
}

static void onemkl_zsyr2_impl(void* handle, fb_gpu_stream_t stream, char uplo, int n, const void* alpha, fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy, fb_gpu_ptr_t a, int lda) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::syr2(*queue, convert_uplo(uplo), n, *static_cast<const std::complex<double>*>(alpha), static_cast<const std::complex<double>*>(x), incx, static_cast<const std::complex<double>*>(y), incy, static_cast<std::complex<double>*>(a), lda);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL zsyr2 error: %s\n", e.what());
    }
}

static void onemkl_cher2_impl(void* handle, fb_gpu_stream_t stream, char uplo, int n, const void* alpha, fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy, fb_gpu_ptr_t a, int lda) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::her2(*queue, convert_uplo(uplo), n, *static_cast<const std::complex<float>*>(alpha), static_cast<const std::complex<float>*>(x), incx, static_cast<const std::complex<float>*>(y), incy, static_cast<std::complex<float>*>(a), lda);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL cher2 error: %s\n", e.what());
    }
}

static void onemkl_zher2_impl(void* handle, fb_gpu_stream_t stream, char uplo, int n, const void* alpha, fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy, fb_gpu_ptr_t a, int lda) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::her2(*queue, convert_uplo(uplo), n, *static_cast<const std::complex<double>*>(alpha), static_cast<const std::complex<double>*>(x), incx, static_cast<const std::complex<double>*>(y), incy, static_cast<std::complex<double>*>(a), lda);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL zher2 error: %s\n", e.what());
    }
}

/* ============================================================================
 * BLAS Level 2 Operations - Banded Symmetric/Hermitian Matrix-Vector Ops
 * ========================================================================== */

static void onemkl_ssbmv_impl(void* handle, fb_gpu_stream_t stream, char uplo, int n, int k, float alpha, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx, float beta, fb_gpu_ptr_t y, int incy) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::sbmv(*queue, convert_uplo(uplo), n, k, alpha, static_cast<const float*>(a), lda, static_cast<const float*>(x), incx, beta, static_cast<float*>(y), incy);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL ssbmv error: %s\n", e.what());
    }
}

static void onemkl_dsbmv_impl(void* handle, fb_gpu_stream_t stream, char uplo, int n, int k, double alpha, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx, double beta, fb_gpu_ptr_t y, int incy) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::sbmv(*queue, convert_uplo(uplo), n, k, alpha, static_cast<const double*>(a), lda, static_cast<const double*>(x), incx, beta, static_cast<double*>(y), incy);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL dsbmv error: %s\n", e.what());
    }
}

static void onemkl_chbmv_impl(void* handle, fb_gpu_stream_t stream, char uplo, int n, int k, const void* alpha, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx, const void* beta, fb_gpu_ptr_t y, int incy) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::hbmv(*queue, convert_uplo(uplo), n, k, *static_cast<const std::complex<float>*>(alpha), static_cast<const std::complex<float>*>(a), lda, static_cast<const std::complex<float>*>(x), incx, *static_cast<const std::complex<float>*>(beta), static_cast<std::complex<float>*>(y), incy);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL chbmv error: %s\n", e.what());
    }
}

static void onemkl_zhbmv_impl(void* handle, fb_gpu_stream_t stream, char uplo, int n, int k, const void* alpha, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx, const void* beta, fb_gpu_ptr_t y, int incy) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::hbmv(*queue, convert_uplo(uplo), n, k, *static_cast<const std::complex<double>*>(alpha), static_cast<const std::complex<double>*>(a), lda, static_cast<const std::complex<double>*>(x), incx, *static_cast<const std::complex<double>*>(beta), static_cast<std::complex<double>*>(y), incy);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL zhbmv error: %s\n", e.what());
    }
}

/* ============================================================================
 * BLAS Level 2 Operations - Packed Symmetric/Hermitian Matrix-Vector Ops
 * ========================================================================== */

static void onemkl_sspmv_impl(void* handle, fb_gpu_stream_t stream, char uplo, int n, float alpha, fb_gpu_ptr_t ap, fb_gpu_ptr_t x, int incx, float beta, fb_gpu_ptr_t y, int incy) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::spmv(*queue, convert_uplo(uplo), n, alpha, static_cast<const float*>(ap), static_cast<const float*>(x), incx, beta, static_cast<float*>(y), incy);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL sspmv error: %s\n", e.what());
    }
}

static void onemkl_dspmv_impl(void* handle, fb_gpu_stream_t stream, char uplo, int n, double alpha, fb_gpu_ptr_t ap, fb_gpu_ptr_t x, int incx, double beta, fb_gpu_ptr_t y, int incy) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::spmv(*queue, convert_uplo(uplo), n, alpha, static_cast<const double*>(ap), static_cast<const double*>(x), incx, beta, static_cast<double*>(y), incy);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL dspmv error: %s\n", e.what());
    }
}

static void onemkl_chpmv_impl(void* handle, fb_gpu_stream_t stream, char uplo, int n, const void* alpha, fb_gpu_ptr_t ap, fb_gpu_ptr_t x, int incx, const void* beta, fb_gpu_ptr_t y, int incy) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::hpmv(*queue, convert_uplo(uplo), n, *static_cast<const std::complex<float>*>(alpha), static_cast<const std::complex<float>*>(ap), static_cast<const std::complex<float>*>(x), incx, *static_cast<const std::complex<float>*>(beta), static_cast<std::complex<float>*>(y), incy);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL chpmv error: %s\n", e.what());
    }
}

static void onemkl_zhpmv_impl(void* handle, fb_gpu_stream_t stream, char uplo, int n, const void* alpha, fb_gpu_ptr_t ap, fb_gpu_ptr_t x, int incx, const void* beta, fb_gpu_ptr_t y, int incy) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::hpmv(*queue, convert_uplo(uplo), n, *static_cast<const std::complex<double>*>(alpha), static_cast<const std::complex<double>*>(ap), static_cast<const std::complex<double>*>(x), incx, *static_cast<const std::complex<double>*>(beta), static_cast<std::complex<double>*>(y), incy);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL zhpmv error: %s\n", e.what());
    }
}

/* ============================================================================
 * BLAS Level 2 Operations - Banded Triangular Matrix-Vector Ops
 * ========================================================================== */

static void onemkl_stbmv_impl(void* handle, fb_gpu_stream_t stream, char uplo, char trans, char diag, int n, int k, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::tbmv(*queue, convert_uplo(uplo), convert_transpose(trans), convert_diag(diag), n, k, static_cast<const float*>(a), lda, static_cast<float*>(x), incx);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL stbmv error: %s\n", e.what());
    }
}

static void onemkl_dtbmv_impl(void* handle, fb_gpu_stream_t stream, char uplo, char trans, char diag, int n, int k, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::tbmv(*queue, convert_uplo(uplo), convert_transpose(trans), convert_diag(diag), n, k, static_cast<const double*>(a), lda, static_cast<double*>(x), incx);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL dtbmv error: %s\n", e.what());
    }
}

static void onemkl_ctbmv_impl(void* handle, fb_gpu_stream_t stream, char uplo, char trans, char diag, int n, int k, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::tbmv(*queue, convert_uplo(uplo), convert_transpose(trans), convert_diag(diag), n, k, static_cast<const std::complex<float>*>(a), lda, static_cast<std::complex<float>*>(x), incx);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL ctbmv error: %s\n", e.what());
    }
}

static void onemkl_ztbmv_impl(void* handle, fb_gpu_stream_t stream, char uplo, char trans, char diag, int n, int k, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::tbmv(*queue, convert_uplo(uplo), convert_transpose(trans), convert_diag(diag), n, k, static_cast<const std::complex<double>*>(a), lda, static_cast<std::complex<double>*>(x), incx);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL ztbmv error: %s\n", e.what());
    }
}

static void onemkl_stbsv_impl(void* handle, fb_gpu_stream_t stream, char uplo, char trans, char diag, int n, int k, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::tbsv(*queue, convert_uplo(uplo), convert_transpose(trans), convert_diag(diag), n, k, static_cast<const float*>(a), lda, static_cast<float*>(x), incx);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL stbsv error: %s\n", e.what());
    }
}

static void onemkl_dtbsv_impl(void* handle, fb_gpu_stream_t stream, char uplo, char trans, char diag, int n, int k, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::tbsv(*queue, convert_uplo(uplo), convert_transpose(trans), convert_diag(diag), n, k, static_cast<const double*>(a), lda, static_cast<double*>(x), incx);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL dtbsv error: %s\n", e.what());
    }
}

static void onemkl_ctbsv_impl(void* handle, fb_gpu_stream_t stream, char uplo, char trans, char diag, int n, int k, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::tbsv(*queue, convert_uplo(uplo), convert_transpose(trans), convert_diag(diag), n, k, static_cast<const std::complex<float>*>(a), lda, static_cast<std::complex<float>*>(x), incx);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL ctbsv error: %s\n", e.what());
    }
}

static void onemkl_ztbsv_impl(void* handle, fb_gpu_stream_t stream, char uplo, char trans, char diag, int n, int k, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t x, int incx) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::tbsv(*queue, convert_uplo(uplo), convert_transpose(trans), convert_diag(diag), n, k, static_cast<const std::complex<double>*>(a), lda, static_cast<std::complex<double>*>(x), incx);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL ztbsv error: %s\n", e.what());
    }
}

/* ============================================================================
 * BLAS Level 2 Operations - Packed Triangular Matrix-Vector Ops
 * ========================================================================== */

static void onemkl_stpmv_impl(void* handle, fb_gpu_stream_t stream, char uplo, char trans, char diag, int n, fb_gpu_ptr_t ap, fb_gpu_ptr_t x, int incx) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::tpmv(*queue, convert_uplo(uplo), convert_transpose(trans), convert_diag(diag), n, static_cast<const float*>(ap), static_cast<float*>(x), incx);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL stpmv error: %s\n", e.what());
    }
}

static void onemkl_dtpmv_impl(void* handle, fb_gpu_stream_t stream, char uplo, char trans, char diag, int n, fb_gpu_ptr_t ap, fb_gpu_ptr_t x, int incx) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::tpmv(*queue, convert_uplo(uplo), convert_transpose(trans), convert_diag(diag), n, static_cast<const double*>(ap), static_cast<double*>(x), incx);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL dtpmv error: %s\n", e.what());
    }
}

static void onemkl_ctpmv_impl(void* handle, fb_gpu_stream_t stream, char uplo, char trans, char diag, int n, fb_gpu_ptr_t ap, fb_gpu_ptr_t x, int incx) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::tpmv(*queue, convert_uplo(uplo), convert_transpose(trans), convert_diag(diag), n, static_cast<const std::complex<float>*>(ap), static_cast<std::complex<float>*>(x), incx);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL ctpmv error: %s\n", e.what());
    }
}

static void onemkl_ztpmv_impl(void* handle, fb_gpu_stream_t stream, char uplo, char trans, char diag, int n, fb_gpu_ptr_t ap, fb_gpu_ptr_t x, int incx) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::tpmv(*queue, convert_uplo(uplo), convert_transpose(trans), convert_diag(diag), n, static_cast<const std::complex<double>*>(ap), static_cast<std::complex<double>*>(x), incx);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL ztpmv error: %s\n", e.what());
    }
}

static void onemkl_stpsv_impl(void* handle, fb_gpu_stream_t stream, char uplo, char trans, char diag, int n, fb_gpu_ptr_t ap, fb_gpu_ptr_t x, int incx) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::tpsv(*queue, convert_uplo(uplo), convert_transpose(trans), convert_diag(diag), n, static_cast<const float*>(ap), static_cast<float*>(x), incx);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL stpsv error: %s\n", e.what());
    }
}

static void onemkl_dtpsv_impl(void* handle, fb_gpu_stream_t stream, char uplo, char trans, char diag, int n, fb_gpu_ptr_t ap, fb_gpu_ptr_t x, int incx) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::tpsv(*queue, convert_uplo(uplo), convert_transpose(trans), convert_diag(diag), n, static_cast<const double*>(ap), static_cast<double*>(x), incx);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL dtpsv error: %s\n", e.what());
    }
}

static void onemkl_ctpsv_impl(void* handle, fb_gpu_stream_t stream, char uplo, char trans, char diag, int n, fb_gpu_ptr_t ap, fb_gpu_ptr_t x, int incx) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::tpsv(*queue, convert_uplo(uplo), convert_transpose(trans), convert_diag(diag), n, static_cast<const std::complex<float>*>(ap), static_cast<std::complex<float>*>(x), incx);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL ctpsv error: %s\n", e.what());
    }
}

static void onemkl_ztpsv_impl(void* handle, fb_gpu_stream_t stream, char uplo, char trans, char diag, int n, fb_gpu_ptr_t ap, fb_gpu_ptr_t x, int incx) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::tpsv(*queue, convert_uplo(uplo), convert_transpose(trans), convert_diag(diag), n, static_cast<const std::complex<double>*>(ap), static_cast<std::complex<double>*>(x), incx);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL ztpsv error: %s\n", e.what());
    }
}

/* ============================================================================
 * BLAS Level 2 Operations - Packed Symmetric/Hermitian Rank Updates
 * ========================================================================== */

static void onemkl_sspr_impl(void* handle, fb_gpu_stream_t stream, char uplo, int n, float alpha, fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t ap) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::spr(*queue, convert_uplo(uplo), n, alpha, static_cast<const float*>(x), incx, static_cast<float*>(ap));
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL sspr error: %s\n", e.what());
    }
}

static void onemkl_dspr_impl(void* handle, fb_gpu_stream_t stream, char uplo, int n, double alpha, fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t ap) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::spr(*queue, convert_uplo(uplo), n, alpha, static_cast<const double*>(x), incx, static_cast<double*>(ap));
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL dspr error: %s\n", e.what());
    }
}

static void onemkl_chpr_impl(void* handle, fb_gpu_stream_t stream, char uplo, int n, float alpha, fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t ap) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::hpr(*queue, convert_uplo(uplo), n, alpha, static_cast<const std::complex<float>*>(x), incx, static_cast<std::complex<float>*>(ap));
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL chpr error: %s\n", e.what());
    }
}

static void onemkl_zhpr_impl(void* handle, fb_gpu_stream_t stream, char uplo, int n, double alpha, fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t ap) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::hpr(*queue, convert_uplo(uplo), n, alpha, static_cast<const std::complex<double>*>(x), incx, static_cast<std::complex<double>*>(ap));
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL zhpr error: %s\n", e.what());
    }
}

static void onemkl_sspr2_impl(void* handle, fb_gpu_stream_t stream, char uplo, int n, float alpha, fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy, fb_gpu_ptr_t ap) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::spr2(*queue, convert_uplo(uplo), n, alpha, static_cast<const float*>(x), incx, static_cast<const float*>(y), incy, static_cast<float*>(ap));
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL sspr2 error: %s\n", e.what());
    }
}

static void onemkl_dspr2_impl(void* handle, fb_gpu_stream_t stream, char uplo, int n, double alpha, fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy, fb_gpu_ptr_t ap) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::spr2(*queue, convert_uplo(uplo), n, alpha, static_cast<const double*>(x), incx, static_cast<const double*>(y), incy, static_cast<double*>(ap));
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL dspr2 error: %s\n", e.what());
    }
}

static void onemkl_chpr2_impl(void* handle, fb_gpu_stream_t stream, char uplo, int n, const void* alpha, fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy, fb_gpu_ptr_t ap) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::hpr2(*queue, convert_uplo(uplo), n, *static_cast<const std::complex<float>*>(alpha), static_cast<const std::complex<float>*>(x), incx, static_cast<const std::complex<float>*>(y), incy, static_cast<std::complex<float>*>(ap));
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL chpr2 error: %s\n", e.what());
    }
}

static void onemkl_zhpr2_impl(void* handle, fb_gpu_stream_t stream, char uplo, int n, const void* alpha, fb_gpu_ptr_t x, int incx, fb_gpu_ptr_t y, int incy, fb_gpu_ptr_t ap) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::hpr2(*queue, convert_uplo(uplo), n, *static_cast<const std::complex<double>*>(alpha), static_cast<const std::complex<double>*>(x), incx, static_cast<const std::complex<double>*>(y), incy, static_cast<std::complex<double>*>(ap));
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL zhpr2 error: %s\n", e.what());
    }
}

// TODO: Implement remaining ~91 Level 2 BLAS operations
// dgemv, cgemv, zgemv, sgbmv, dgbmv, cgbmv, zgbmv,
// ssymv, dsymv, chemv, zhemv, ssbmv, dsbmv, chbmv, zhbmv,
// sspmv, dspmv, chpmv, zhpmv (oneMKL has these! cuBLAS/rocBLAS don't),
// strmv, dtrmv, ctrmv, ztrmv, stbmv, dtbmv, ctbmv, ztbmv,
// stpmv, dtpmv, ctpmv, ztpmv, strsv, dtrsv, ctrsv, ztrsv,
// stbsv, dtbsv, ctbsv, ztbsv, stpsv, dtpsv, ctpsv, ztpsv,
// sger, dger, cgeru, zgeru, cgerc, zgerc,
// ssyr, dsyr, cher, zher, sspr, dspr, chpr, zhpr,
// ssyr2, dsyr2, cher2, zher2, sspr2, dspr2, chpr2, zhpr2,
// csymv, zsymv, csyr, zsyr, csyr2, zsyr2,
// cspmv, zspmv (AVAILABLE IN ONEMKL!), cspr, zspr, cspr2, zspr2

/* ============================================================================
 * BLAS Level 3 Operations - Matrix-Matrix Operations
 * ========================================================================== */

// Example: SGEMM - Single precision matrix-matrix multiply
static void onemkl_sgemm_impl(void* handle, fb_gpu_stream_t stream,
                               char transa, char transb, int m, int n, int k,
                               float alpha, fb_gpu_ptr_t a, int lda,
                               fb_gpu_ptr_t b, int ldb,
                               float beta, fb_gpu_ptr_t c, int ldc) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    
    try {
        oneapi::mkl::blas::gemm(*queue, 
                                convert_transpose(transa), convert_transpose(transb),
                                m, n, k,
                                alpha,
                                static_cast<const float*>(a), lda,
                                static_cast<const float*>(b), ldb,
                                beta,
                                static_cast<float*>(c), ldc);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL sgemm error: %s\n", e.what());
    }
}

static void onemkl_dgemm_impl(void* handle, fb_gpu_stream_t stream, char transa, char transb, int m, int n, int k, double alpha, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb, double beta, fb_gpu_ptr_t c, int ldc) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::gemm(*queue, convert_transpose(transa), convert_transpose(transb), m, n, k, alpha, static_cast<const double*>(a), lda, static_cast<const double*>(b), ldb, beta, static_cast<double*>(c), ldc);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL dgemm error: %s\n", e.what());
    }
}

static void onemkl_cgemm_impl(void* handle, fb_gpu_stream_t stream, char transa, char transb, int m, int n, int k, const void* alpha, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb, const void* beta, fb_gpu_ptr_t c, int ldc) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::gemm(*queue, convert_transpose(transa), convert_transpose(transb), m, n, k, *static_cast<const std::complex<float>*>(alpha), static_cast<const std::complex<float>*>(a), lda, static_cast<const std::complex<float>*>(b), ldb, *static_cast<const std::complex<float>*>(beta), static_cast<std::complex<float>*>(c), ldc);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL cgemm error: %s\n", e.what());
    }
}

static void onemkl_zgemm_impl(void* handle, fb_gpu_stream_t stream, char transa, char transb, int m, int n, int k, const void* alpha, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb, const void* beta, fb_gpu_ptr_t c, int ldc) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::gemm(*queue, convert_transpose(transa), convert_transpose(transb), m, n, k, *static_cast<const std::complex<double>*>(alpha), static_cast<const std::complex<double>*>(a), lda, static_cast<const std::complex<double>*>(b), ldb, *static_cast<const std::complex<double>*>(beta), static_cast<std::complex<double>*>(c), ldc);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL zgemm error: %s\n", e.what());
    }
}

/* ============================================================================
 * BLAS Level 3 Operations - Symmetric/Hermitian Matrix Multiply
 * ========================================================================== */

static void onemkl_ssymm_impl(void* handle, fb_gpu_stream_t stream, char side, char uplo, int m, int n, float alpha, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb, float beta, fb_gpu_ptr_t c, int ldc) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::symm(*queue, convert_side(side), convert_uplo(uplo), m, n, alpha, static_cast<const float*>(a), lda, static_cast<const float*>(b), ldb, beta, static_cast<float*>(c), ldc);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL ssymm error: %s\n", e.what());
    }
}

static void onemkl_dsymm_impl(void* handle, fb_gpu_stream_t stream, char side, char uplo, int m, int n, double alpha, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb, double beta, fb_gpu_ptr_t c, int ldc) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::symm(*queue, convert_side(side), convert_uplo(uplo), m, n, alpha, static_cast<const double*>(a), lda, static_cast<const double*>(b), ldb, beta, static_cast<double*>(c), ldc);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL dsymm error: %s\n", e.what());
    }
}

static void onemkl_csymm_impl(void* handle, fb_gpu_stream_t stream, char side, char uplo, int m, int n, const void* alpha, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb, const void* beta, fb_gpu_ptr_t c, int ldc) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::symm(*queue, convert_side(side), convert_uplo(uplo), m, n, *static_cast<const std::complex<float>*>(alpha), static_cast<const std::complex<float>*>(a), lda, static_cast<const std::complex<float>*>(b), ldb, *static_cast<const std::complex<float>*>(beta), static_cast<std::complex<float>*>(c), ldc);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL csymm error: %s\n", e.what());
    }
}

static void onemkl_zsymm_impl(void* handle, fb_gpu_stream_t stream, char side, char uplo, int m, int n, const void* alpha, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb, const void* beta, fb_gpu_ptr_t c, int ldc) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::symm(*queue, convert_side(side), convert_uplo(uplo), m, n, *static_cast<const std::complex<double>*>(alpha), static_cast<const std::complex<double>*>(a), lda, static_cast<const std::complex<double>*>(b), ldb, *static_cast<const std::complex<double>*>(beta), static_cast<std::complex<double>*>(c), ldc);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL zsymm error: %s\n", e.what());
    }
}

static void onemkl_chemm_impl(void* handle, fb_gpu_stream_t stream, char side, char uplo, int m, int n, const void* alpha, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb, const void* beta, fb_gpu_ptr_t c, int ldc) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::hemm(*queue, convert_side(side), convert_uplo(uplo), m, n, *static_cast<const std::complex<float>*>(alpha), static_cast<const std::complex<float>*>(a), lda, static_cast<const std::complex<float>*>(b), ldb, *static_cast<const std::complex<float>*>(beta), static_cast<std::complex<float>*>(c), ldc);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL chemm error: %s\n", e.what());
    }
}

static void onemkl_zhemm_impl(void* handle, fb_gpu_stream_t stream, char side, char uplo, int m, int n, const void* alpha, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb, const void* beta, fb_gpu_ptr_t c, int ldc) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::hemm(*queue, convert_side(side), convert_uplo(uplo), m, n, *static_cast<const std::complex<double>*>(alpha), static_cast<const std::complex<double>*>(a), lda, static_cast<const std::complex<double>*>(b), ldb, *static_cast<const std::complex<double>*>(beta), static_cast<std::complex<double>*>(c), ldc);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL zhemm error: %s\n", e.what());
    }
}

/* ============================================================================
 * BLAS Level 3 Operations - Triangular Matrix Multiply/Solve
 * ========================================================================== */

static void onemkl_strmm_impl(void* handle, fb_gpu_stream_t stream, char side, char uplo, char trans, char diag, int m, int n, float alpha, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::trmm(*queue, convert_side(side), convert_uplo(uplo), convert_transpose(trans), convert_diag(diag), m, n, alpha, static_cast<const float*>(a), lda, static_cast<float*>(b), ldb);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL strmm error: %s\n", e.what());
    }
}

static void onemkl_dtrmm_impl(void* handle, fb_gpu_stream_t stream, char side, char uplo, char trans, char diag, int m, int n, double alpha, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::trmm(*queue, convert_side(side), convert_uplo(uplo), convert_transpose(trans), convert_diag(diag), m, n, alpha, static_cast<const double*>(a), lda, static_cast<double*>(b), ldb);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL dtrmm error: %s\n", e.what());
    }
}

static void onemkl_ctrmm_impl(void* handle, fb_gpu_stream_t stream, char side, char uplo, char trans, char diag, int m, int n, const void* alpha, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::trmm(*queue, convert_side(side), convert_uplo(uplo), convert_transpose(trans), convert_diag(diag), m, n, *static_cast<const std::complex<float>*>(alpha), static_cast<const std::complex<float>*>(a), lda, static_cast<std::complex<float>*>(b), ldb);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL ctrmm error: %s\n", e.what());
    }
}

static void onemkl_ztrmm_impl(void* handle, fb_gpu_stream_t stream, char side, char uplo, char trans, char diag, int m, int n, const void* alpha, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::trmm(*queue, convert_side(side), convert_uplo(uplo), convert_transpose(trans), convert_diag(diag), m, n, *static_cast<const std::complex<double>*>(alpha), static_cast<const std::complex<double>*>(a), lda, static_cast<std::complex<double>*>(b), ldb);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL ztrmm error: %s\n", e.what());
    }
}

static void onemkl_strsm_impl(void* handle, fb_gpu_stream_t stream, char side, char uplo, char trans, char diag, int m, int n, float alpha, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::trsm(*queue, convert_side(side), convert_uplo(uplo), convert_transpose(trans), convert_diag(diag), m, n, alpha, static_cast<const float*>(a), lda, static_cast<float*>(b), ldb);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL strsm error: %s\n", e.what());
    }
}

static void onemkl_dtrsm_impl(void* handle, fb_gpu_stream_t stream, char side, char uplo, char trans, char diag, int m, int n, double alpha, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::trsm(*queue, convert_side(side), convert_uplo(uplo), convert_transpose(trans), convert_diag(diag), m, n, alpha, static_cast<const double*>(a), lda, static_cast<double*>(b), ldb);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL dtrsm error: %s\n", e.what());
    }
}

static void onemkl_ctrsm_impl(void* handle, fb_gpu_stream_t stream, char side, char uplo, char trans, char diag, int m, int n, const void* alpha, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::trsm(*queue, convert_side(side), convert_uplo(uplo), convert_transpose(trans), convert_diag(diag), m, n, *static_cast<const std::complex<float>*>(alpha), static_cast<const std::complex<float>*>(a), lda, static_cast<std::complex<float>*>(b), ldb);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL ctrsm error: %s\n", e.what());
    }
}

static void onemkl_ztrsm_impl(void* handle, fb_gpu_stream_t stream, char side, char uplo, char trans, char diag, int m, int n, const void* alpha, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::trsm(*queue, convert_side(side), convert_uplo(uplo), convert_transpose(trans), convert_diag(diag), m, n, *static_cast<const std::complex<double>*>(alpha), static_cast<const std::complex<double>*>(a), lda, static_cast<std::complex<double>*>(b), ldb);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL ztrsm error: %s\n", e.what());
    }
}

/* ============================================================================
 * BLAS Level 3 Operations - Symmetric/Hermitian Rank-k Updates
 * ========================================================================== */

static void onemkl_ssyrk_impl(void* handle, fb_gpu_stream_t stream, char uplo, char trans, int n, int k, float alpha, fb_gpu_ptr_t a, int lda, float beta, fb_gpu_ptr_t c, int ldc) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::syrk(*queue, convert_uplo(uplo), convert_transpose(trans), n, k, alpha, static_cast<const float*>(a), lda, beta, static_cast<float*>(c), ldc);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL ssyrk error: %s\n", e.what());
    }
}

static void onemkl_dsyrk_impl(void* handle, fb_gpu_stream_t stream, char uplo, char trans, int n, int k, double alpha, fb_gpu_ptr_t a, int lda, double beta, fb_gpu_ptr_t c, int ldc) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::syrk(*queue, convert_uplo(uplo), convert_transpose(trans), n, k, alpha, static_cast<const double*>(a), lda, beta, static_cast<double*>(c), ldc);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL dsyrk error: %s\n", e.what());
    }
}

static void onemkl_csyrk_impl(void* handle, fb_gpu_stream_t stream, char uplo, char trans, int n, int k, const void* alpha, fb_gpu_ptr_t a, int lda, const void* beta, fb_gpu_ptr_t c, int ldc) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::syrk(*queue, convert_uplo(uplo), convert_transpose(trans), n, k, *static_cast<const std::complex<float>*>(alpha), static_cast<const std::complex<float>*>(a), lda, *static_cast<const std::complex<float>*>(beta), static_cast<std::complex<float>*>(c), ldc);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL csyrk error: %s\n", e.what());
    }
}

static void onemkl_zsyrk_impl(void* handle, fb_gpu_stream_t stream, char uplo, char trans, int n, int k, const void* alpha, fb_gpu_ptr_t a, int lda, const void* beta, fb_gpu_ptr_t c, int ldc) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::syrk(*queue, convert_uplo(uplo), convert_transpose(trans), n, k, *static_cast<const std::complex<double>*>(alpha), static_cast<const std::complex<double>*>(a), lda, *static_cast<const std::complex<double>*>(beta), static_cast<std::complex<double>*>(c), ldc);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL zsyrk error: %s\n", e.what());
    }
}

static void onemkl_cherk_impl(void* handle, fb_gpu_stream_t stream, char uplo, char trans, int n, int k, float alpha, fb_gpu_ptr_t a, int lda, float beta, fb_gpu_ptr_t c, int ldc) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::herk(*queue, convert_uplo(uplo), convert_transpose(trans), n, k, alpha, static_cast<const std::complex<float>*>(a), lda, beta, static_cast<std::complex<float>*>(c), ldc);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL cherk error: %s\n", e.what());
    }
}

static void onemkl_zherk_impl(void* handle, fb_gpu_stream_t stream, char uplo, char trans, int n, int k, double alpha, fb_gpu_ptr_t a, int lda, double beta, fb_gpu_ptr_t c, int ldc) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::herk(*queue, convert_uplo(uplo), convert_transpose(trans), n, k, alpha, static_cast<const std::complex<double>*>(a), lda, beta, static_cast<std::complex<double>*>(c), ldc);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL zherk error: %s\n", e.what());
    }
}

static void onemkl_ssyr2k_impl(void* handle, fb_gpu_stream_t stream, char uplo, char trans, int n, int k, float alpha, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb, float beta, fb_gpu_ptr_t c, int ldc) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::syr2k(*queue, convert_uplo(uplo), convert_transpose(trans), n, k, alpha, static_cast<const float*>(a), lda, static_cast<const float*>(b), ldb, beta, static_cast<float*>(c), ldc);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL ssyr2k error: %s\n", e.what());
    }
}

static void onemkl_dsyr2k_impl(void* handle, fb_gpu_stream_t stream, char uplo, char trans, int n, int k, double alpha, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb, double beta, fb_gpu_ptr_t c, int ldc) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::syr2k(*queue, convert_uplo(uplo), convert_transpose(trans), n, k, alpha, static_cast<const double*>(a), lda, static_cast<const double*>(b), ldb, beta, static_cast<double*>(c), ldc);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL dsyr2k error: %s\n", e.what());
    }
}

static void onemkl_csyr2k_impl(void* handle, fb_gpu_stream_t stream, char uplo, char trans, int n, int k, const void* alpha, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb, const void* beta, fb_gpu_ptr_t c, int ldc) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::syr2k(*queue, convert_uplo(uplo), convert_transpose(trans), n, k, *static_cast<const std::complex<float>*>(alpha), static_cast<const std::complex<float>*>(a), lda, static_cast<const std::complex<float>*>(b), ldb, *static_cast<const std::complex<float>*>(beta), static_cast<std::complex<float>*>(c), ldc);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL csyr2k error: %s\n", e.what());
    }
}

static void onemkl_zsyr2k_impl(void* handle, fb_gpu_stream_t stream, char uplo, char trans, int n, int k, const void* alpha, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb, const void* beta, fb_gpu_ptr_t c, int ldc) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::syr2k(*queue, convert_uplo(uplo), convert_transpose(trans), n, k, *static_cast<const std::complex<double>*>(alpha), static_cast<const std::complex<double>*>(a), lda, static_cast<const std::complex<double>*>(b), ldb, *static_cast<const std::complex<double>*>(beta), static_cast<std::complex<double>*>(c), ldc);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL zsyr2k error: %s\n", e.what());
    }
}

static void onemkl_cher2k_impl(void* handle, fb_gpu_stream_t stream, char uplo, char trans, int n, int k, const void* alpha, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb, float beta, fb_gpu_ptr_t c, int ldc) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::her2k(*queue, convert_uplo(uplo), convert_transpose(trans), n, k, *static_cast<const std::complex<float>*>(alpha), static_cast<const std::complex<float>*>(a), lda, static_cast<const std::complex<float>*>(b), ldb, beta, static_cast<std::complex<float>*>(c), ldc);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL cher2k error: %s\n", e.what());
    }
}

static void onemkl_zher2k_impl(void* handle, fb_gpu_stream_t stream, char uplo, char trans, int n, int k, const void* alpha, fb_gpu_ptr_t a, int lda, fb_gpu_ptr_t b, int ldb, double beta, fb_gpu_ptr_t c, int ldc) {
    onemkl_context_t* ctx = static_cast<onemkl_context_t*>(handle);
    sycl::queue* queue = stream ? static_cast<sycl::queue*>(stream) : ctx->sycl_queue;
    try {
        oneapi::mkl::blas::her2k(*queue, convert_uplo(uplo), convert_transpose(trans), n, k, *static_cast<const std::complex<double>*>(alpha), static_cast<const std::complex<double>*>(a), lda, static_cast<const std::complex<double>*>(b), ldb, beta, static_cast<std::complex<double>*>(c), ldc);
    } catch (const oneapi::mkl::exception& e) {
        fprintf(stderr, "oneMKL zher2k error: %s\n", e.what());
    }
}

/* ============================================================================
 * LAPACK Operations (subset) - Available in separate onemkl_lapack_impl.cpp
 * ========================================================================== */

// Note: LAPACK operations are in a separate file for clarity
// See onemkl_lapack_impl.cpp for getrf, getrs, potrf, potrs, geqrf, gesvd, syev/heev, etc.

/* ============================================================================
 * Enum Conversion Functions (for vtable)
 * ========================================================================== */

static int onemkl_convert_transpose_impl(char trans) {
    switch (trans) {
        case 'N': case 'n': return 0;  // nontrans
        case 'T': case 't': return 1;  // trans
        case 'C': case 'c': return 2;  // conjtrans
        default: return 0;
    }
}

static int onemkl_convert_uplo_impl(char uplo) {
    return (uplo == 'U' || uplo == 'u') ? 0 : 1;  // upper : lower
}

static int onemkl_convert_diag_impl(char diag) {
    return (diag == 'U' || diag == 'u') ? 1 : 0;  // unit : nonunit
}

static int onemkl_convert_side_impl(char side) {
    return (side == 'L' || side == 'l') ? 0 : 1;  // left : right
}

/* ============================================================================
 * Virtual Function Table (Trait Implementation)
 * ========================================================================== */

static const fb_gpu_backend_trait onemkl_trait = {
    .name = "Intel oneAPI oneMKL",
    .type = FB_GPU_BACKEND_ONEMKL,
    
    // Lifecycle
    .init = onemkl_init_impl,
    .shutdown = onemkl_shutdown_impl,
    .get_device_properties = onemkl_get_device_properties_impl,
    
    // Memory
    .malloc = onemkl_malloc_impl,
    .free = onemkl_free_impl,
    .memcpy_h2d = onemkl_memcpy_h2d_impl,
    .memcpy_d2h = onemkl_memcpy_d2h_impl,
    .memcpy_d2d = onemkl_memcpy_d2d_impl,
    
    // Streams
    .stream_create = onemkl_stream_create_impl,
    .stream_destroy = onemkl_stream_destroy_impl,
    .stream_synchronize = onemkl_stream_synchronize_impl,
    
    // Enum conversions
    .convert_transpose = onemkl_convert_transpose_impl,
    .convert_uplo = onemkl_convert_uplo_impl,
    .convert_diag = onemkl_convert_diag_impl,
    .convert_side = onemkl_convert_side_impl,
    
    // Level 1 BLAS - Real
    .saxpy = onemkl_saxpy_impl,
    .daxpy = onemkl_daxpy_impl,
    .sscal = onemkl_sscal_impl,
    .dscal = onemkl_dscal_impl,
    .scopy = onemkl_scopy_impl,
    .dcopy = onemkl_dcopy_impl,
    .sswap = onemkl_sswap_impl,
    .dswap = onemkl_dswap_impl,
    .sdot = onemkl_sdot_impl,
    .ddot = onemkl_ddot_impl,
    .snrm2 = onemkl_snrm2_impl,
    .dnrm2 = onemkl_dnrm2_impl,
    .sasum = onemkl_sasum_impl,
    .dasum = onemkl_dasum_impl,
    .isamax = onemkl_isamax_impl,
    .idamax = onemkl_idamax_impl,
    
    // Level 1 BLAS - Complex
    .caxpy = onemkl_caxpy_impl,
    .zaxpy = onemkl_zaxpy_impl,
    .cscal = onemkl_cscal_impl,
    .zscal = onemkl_zscal_impl,
    .csscal = onemkl_csscal_impl,
    .zdscal = onemkl_zdscal_impl,
    .ccopy = onemkl_ccopy_impl,
    .zcopy = onemkl_zcopy_impl,
    .cswap = onemkl_cswap_impl,
    .zswap = onemkl_zswap_impl,
    .cdotu = onemkl_cdotu_impl,
    .zdotu = onemkl_zdotu_impl,
    .cdotc = onemkl_cdotc_impl,
    .zdotc = onemkl_zdotc_impl,
    .scnrm2 = onemkl_scnrm2_impl,
    .dznrm2 = onemkl_dznrm2_impl,
    .scasum = onemkl_scasum_impl,
    .dzasum = onemkl_dzasum_impl,
    .icamax = onemkl_icamax_impl,
    .izamax = onemkl_izamax_impl,
    
    // Level 1 BLAS - Rotation
    .srotg = onemkl_srotg_impl,
    .drotg = onemkl_drotg_impl,
    .srot = onemkl_srot_impl,
    .drot = onemkl_drot_impl,
    .srotm = onemkl_srotm_impl,
    .drotm = onemkl_drotm_impl,
    .srotmg = onemkl_srotmg_impl,
    .drotmg = onemkl_drotmg_impl,
    
    // Level 2 BLAS - Matrix-vector operations
    .sgemv = onemkl_sgemv_impl,
    .dgemv = onemkl_dgemv_impl,
    .cgemv = onemkl_cgemv_impl,
    .zgemv = onemkl_zgemv_impl,
    .chemv = onemkl_chemv_impl,
    .zhemv = onemkl_zhemv_impl,
    .ssymv = onemkl_ssymv_impl,
    .dsymv = onemkl_dsymv_impl,
    .strmv = onemkl_strmv_impl,
    .dtrmv = onemkl_dtrmv_impl,
    .ctrmv = onemkl_ctrmv_impl,
    .ztrmv = onemkl_ztrmv_impl,
    .strsv = onemkl_strsv_impl,
    .dtrsv = onemkl_dtrsv_impl,
    .ctrsv = onemkl_ctrsv_impl,
    .ztrsv = onemkl_ztrsv_impl,
    
    // Level 2 BLAS - Rank updates
    .sger = onemkl_sger_impl,
    .dger = onemkl_dger_impl,
    .cgeru = onemkl_cgeru_impl,
    .zgeru = onemkl_zgeru_impl,
    .cgerc = onemkl_cgerc_impl,
    .zgerc = onemkl_zgerc_impl,
    .cher = onemkl_cher_impl,
    .zher = onemkl_zher_impl,
    .ssyr = onemkl_ssyr_impl,
    .dsyr = onemkl_dsyr_impl,
    .cher2 = onemkl_cher2_impl,
    .zher2 = onemkl_zher2_impl,
    .ssyr2 = onemkl_ssyr2_impl,
    .dsyr2 = onemkl_dsyr2_impl,
    
    // Level 2 BLAS - Banded matrix operations
    .sgbmv = onemkl_sgbmv_impl,
    .dgbmv = onemkl_dgbmv_impl,
    .cgbmv = onemkl_cgbmv_impl,
    .zgbmv = onemkl_zgbmv_impl,
    .ssbmv = onemkl_ssbmv_impl,
    .dsbmv = onemkl_dsbmv_impl,
    .chbmv = onemkl_chbmv_impl,
    .zhbmv = onemkl_zhbmv_impl,
    .stbmv = onemkl_stbmv_impl,
    .dtbmv = onemkl_dtbmv_impl,
    .ctbmv = onemkl_ctbmv_impl,
    .ztbmv = onemkl_ztbmv_impl,
    .stbsv = onemkl_stbsv_impl,
    .dtbsv = onemkl_dtbsv_impl,
    .ctbsv = onemkl_ctbsv_impl,
    .ztbsv = onemkl_ztbsv_impl,
    
    // Level 2 BLAS - Packed matrix operations
    .sspmv = onemkl_sspmv_impl,
    .dspmv = onemkl_dspmv_impl,
    .chpmv = onemkl_chpmv_impl,
    .zhpmv = onemkl_zhpmv_impl,
    .stpmv = onemkl_stpmv_impl,
    .dtpmv = onemkl_dtpmv_impl,
    .ctpmv = onemkl_ctpmv_impl,
    .ztpmv = onemkl_ztpmv_impl,
    .stpsv = onemkl_stpsv_impl,
    .dtpsv = onemkl_dtpsv_impl,
    .ctpsv = onemkl_ctpsv_impl,
    .ztpsv = onemkl_ztpsv_impl,
    
    // Level 2 BLAS - Packed rank updates
    .sspr = onemkl_sspr_impl,
    .dspr = onemkl_dspr_impl,
    .chpr = onemkl_chpr_impl,
    .zhpr = onemkl_zhpr_impl,
    .sspr2 = onemkl_sspr2_impl,
    .dspr2 = onemkl_dspr2_impl,
    .chpr2 = onemkl_chpr2_impl,
    .zhpr2 = onemkl_zhpr2_impl,
    
    // Level 2 BLAS - Complex symmetric operations
    .csymv = onemkl_csymv_impl,
    .zsymv = onemkl_zsymv_impl,
    .csyr = onemkl_csyr_impl,
    .zsyr = onemkl_zsyr_impl,
    .csyr2 = onemkl_csyr2_impl,
    .zsyr2 = onemkl_zsyr2_impl,
    
    // Complex symmetric packed operations (NOT supported by cuBLAS - rocBLAS only)
    .cspmv = NULL,
    .zspmv = NULL,
    .cspr = NULL,
    .zspr = NULL,
    .cspr2 = NULL,
    .zspr2 = NULL,
    
    // Level 3 BLAS - Matrix-matrix operations
    .sgemm = onemkl_sgemm_impl,
    .dgemm = onemkl_dgemm_impl,
    .cgemm = onemkl_cgemm_impl,
    .zgemm = onemkl_zgemm_impl,
    .ssymm = onemkl_ssymm_impl,
    .dsymm = onemkl_dsymm_impl,
    .csymm = onemkl_csymm_impl,
    .zsymm = onemkl_zsymm_impl,
    .chemm = onemkl_chemm_impl,
    .zhemm = onemkl_zhemm_impl,
    .strmm = onemkl_strmm_impl,
    .dtrmm = onemkl_dtrmm_impl,
    .ctrmm = onemkl_ctrmm_impl,
    .ztrmm = onemkl_ztrmm_impl,
    .strsm = onemkl_strsm_impl,
    .dtrsm = onemkl_dtrsm_impl,
    .ctrsm = onemkl_ctrsm_impl,
    .ztrsm = onemkl_ztrsm_impl,
    .ssyrk = onemkl_ssyrk_impl,
    .dsyrk = onemkl_dsyrk_impl,
    .csyrk = onemkl_csyrk_impl,
    .zsyrk = onemkl_zsyrk_impl,
    .cherk = onemkl_cherk_impl,
    .zherk = onemkl_zherk_impl,
    .ssyr2k = onemkl_ssyr2k_impl,
    .dsyr2k = onemkl_dsyr2k_impl,
    .csyr2k = onemkl_csyr2k_impl,
    .zsyr2k = onemkl_zsyr2k_impl,
    .cher2k = onemkl_cher2k_impl,
    .zher2k = onemkl_zher2k_impl,
    
    // LAPACK operations (available in separate onemkl_lapack_impl.cpp backend)
    // Note: These are NULL here - use the separate LAPACK backend for these operations
    .sgetrf = NULL,
    .dgetrf = NULL,
    .cgetrf = NULL,
    .zgetrf = NULL,
    .sgetrs = NULL,
    .dgetrs = NULL,
    .cgetrs = NULL,
    .zgetrs = NULL,
    .spotrf = NULL,
    .dpotrf = NULL,
    .cpotrf = NULL,
    .zpotrf = NULL,
    .spotrs = NULL,
    .dpotrs = NULL,
    .cpotrs = NULL,
    .zpotrs = NULL,
    .sgeqrf = NULL,
    .dgeqrf = NULL,
    .cgeqrf = NULL,
    .zgeqrf = NULL,
    .sgesvd = NULL,
    .dgesvd = NULL,
    .cgesvd = NULL,
    .zgesvd = NULL,
    .ssyev = NULL,
    .dsyev = NULL,
    .cheev = NULL,
    .zheev = NULL,
};

/* ============================================================================
 * Public Interface
 * ========================================================================== */

const fb_gpu_backend_trait* fb_onemkl_get_trait(void) {
    return &onemkl_trait;
}

} // extern "C"
