/**
 * @file clblast_vtable_wrappers.c
 * @brief Wrapper functions to adapt CLBlast GPU trait to unified vtable
 * 
 * Bridges the gap between:
 * - GPU trait signature: (void* handle, stream, int, float, fb_gpu_ptr_t, ...)
 * - Unified vtable:      (int64_t, float, float*, ...)
 * 
 * Also handles layout translation since CLBlast only supports column-major.
 */

#include "faster-blaster/backend_plugin.h"
#include "faster-blaster/gpu_backend_trait.h"
#include "../backends/backend_interface.h"
#include "layout_translation.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef FB_ENABLE_OPENCL
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

extern const fb_gpu_backend_trait_t fb_clblast_trait;

/* CLBlast context structure (from clblast_trait_impl.c) */
typedef struct {
    cl_context context;
    cl_command_queue queue;
    cl_device_id device;
    int device_id;
    void* lib_handle;
    /* ... function pointers ... */
} clblast_context_t;

/* ============================================================================
 * Multi-Instance Support - Vtable Registry
 * 
 * Each CLBlast instance has its own vtable and backend_handle.
 * We maintain a mapping: vtable pointer -> backend_handle
 * This allows multiple concurrent instances without global/TLS state.
 * ========================================================================== */

#define MAX_CLBLAST_INSTANCES 16

typedef struct {
    const fb_backend_vtable_t* vtable;
    void* backend_handle;
    int active;
} clblast_instance_registry_entry_t;

static clblast_instance_registry_entry_t g_clblast_registry[MAX_CLBLAST_INSTANCES] = {{0}};
static int g_registry_initialized = 0;

/* Thread-local variable to track which vtable is currently being used.
 * Since Windows DLL TLS can be unreliable, we use a simple static variable.
 * This assumes single-threaded access or proper external locking. */
static const fb_backend_vtable_t* g_current_vtable = NULL;

/* Mark which vtable should be used for subsequent operations */
void clblast_set_active_instance(const fb_backend_vtable_t* vtable) {
    g_current_vtable = vtable;
    fprintf(stderr, "[CLBlast] set_active_instance: vtable=%p\n", vtable);
}

/* Register a vtable->handle mapping */
void clblast_register_instance(const fb_backend_vtable_t* vtable, void* backend_handle) {
    fprintf(stderr, "[CLBlast] register_instance called: vtable=%p, handle=%p\n", vtable, backend_handle);
    if (!vtable || !backend_handle) {
        fprintf(stderr, "[CLBlast] register_instance: NULL vtable or handle!\n");
        return;
    }
    
    for (int i = 0; i < MAX_CLBLAST_INSTANCES; i++) {
        if (!g_clblast_registry[i].active) {
            g_clblast_registry[i].vtable = vtable;
            g_clblast_registry[i].backend_handle = backend_handle;
            g_clblast_registry[i].active = 1;
            fprintf(stderr, "[CLBlast] Registered at slot %d\n", i);
            return;
        }
    }
    fprintf(stderr, "[CLBlast] ERROR: Registry full!\n");
}

/* Unregister a vtable */
void clblast_unregister_instance(const fb_backend_vtable_t* vtable) {
    for (int i = 0; i < MAX_CLBLAST_INSTANCES; i++) {
        if (g_clblast_registry[i].active && g_clblast_registry[i].vtable == vtable) {
            fprintf(stderr, "[CLBlast] Unregistered instance: vtable=%p at slot %d\n", vtable, i);
            g_clblast_registry[i].active = 0;
            g_clblast_registry[i].vtable = NULL;
            g_clblast_registry[i].backend_handle = NULL;
            return;
        }
    }
}

/* Lookup backend handle for the currently active vtable */
static void* clblast_get_handle_for_vtable(const void* vtable_ptr) {
    (void)vtable_ptr;  /* Not used, we use g_current_vtable instead */
    
    /* Look up the handle for the currently active vtable */
    if (!g_current_vtable) {
        fprintf(stderr, "[CLBlast] ERROR: No active vtable set!\n");
        return NULL;
    }
    
    for (int i = 0; i < MAX_CLBLAST_INSTANCES; i++) {
        if (g_clblast_registry[i].active && g_clblast_registry[i].vtable == g_current_vtable) {
            fprintf(stderr, "[CLBlast] Found handle for vtable=%p at slot %d, handle=%p\n", 
                   g_current_vtable, i, g_clblast_registry[i].backend_handle);
            return g_clblast_registry[i].backend_handle;
        }
    }
    
    fprintf(stderr, "[CLBlast] ERROR: No handle found for active vtable=%p\n", g_current_vtable);
    return NULL;
}

/* ============================================================================
 * GPU Buffer Management Helpers
 * 
 * These functions handle automatic host-to-device memory transfers so that
 * the unified vtable can accept host pointers (like CPU backends do).
 * ========================================================================== */

/* Allocate GPU buffer and copy data from host */
static cl_mem clblast_alloc_and_copy_to_device(clblast_context_t* ctx, 
                                                const void* host_ptr, 
                                                size_t size_bytes) {
    cl_int err;
    cl_mem device_buffer = clCreateBuffer(ctx->context, CL_MEM_READ_WRITE, 
                                         size_bytes, NULL, &err);
    if (err != CL_SUCCESS || !device_buffer) {
        fprintf(stderr, "[CLBlast] ERROR: Failed to allocate GPU buffer: %d\n", err);
        return NULL;
    }
    
    err = clEnqueueWriteBuffer(ctx->queue, device_buffer, CL_TRUE, 0, 
                              size_bytes, host_ptr, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "[CLBlast] ERROR: Failed to copy to device: %d\n", err);
        clReleaseMemObject(device_buffer);
        return NULL;
    }
    
    return device_buffer;
}

/* Copy data from device to host */
static int clblast_copy_from_device(clblast_context_t* ctx,
                                    void* host_ptr,
                                    cl_mem device_buffer,
                                    size_t size_bytes) {
    cl_int err = clEnqueueReadBuffer(ctx->queue, device_buffer, CL_TRUE, 0,
                                    size_bytes, host_ptr, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "[CLBlast] ERROR: Failed to copy from device: %d\n", err);
        return -1;
    }
    return 0;
}

/* ============================================================================
 * Level 1 BLAS Wrappers
 * ========================================================================== */

static void clblast_saxpy_wrapper(const int64_t n, const float alpha,
                                  const float *x, const int64_t incx,
                                  float *y, const int64_t incy) {
    void* backend_handle = clblast_get_handle_for_vtable(NULL);
    if (!backend_handle || !fb_clblast_trait.saxpy) {
        return;
    }
    
    clblast_context_t* ctx = (clblast_context_t*)backend_handle;
    
    /* Allocate GPU buffers and copy input data */
    size_t x_size = n * incx * sizeof(float);
    size_t y_size = n * incy * sizeof(float);
    
    cl_mem x_device = clblast_alloc_and_copy_to_device(ctx, x, x_size);
    cl_mem y_device = clblast_alloc_and_copy_to_device(ctx, y, y_size);
    
    if (!x_device || !y_device) {
        fprintf(stderr, "[CLBlast] ERROR: Failed to allocate device buffers\n");
        if (x_device) clReleaseMemObject(x_device);
        if (y_device) clReleaseMemObject(y_device);
        return;
    }
    
    fprintf(stderr, "[CLBlast] saxpy: About to call CLBlast with n=%lld, alpha=%f\n", n, alpha);
    fprintf(stderr, "[CLBlast] saxpy: x_device=%p, y_device=%p\n", x_device, y_device);
    fprintf(stderr, "[CLBlast] saxpy: fb_clblast_trait.saxpy=%p\n", fb_clblast_trait.saxpy);
    fprintf(stderr, "[CLBlast] saxpy: backend_handle=%p\n", backend_handle);
    
    /* Call CLBlast */
    fb_clblast_trait.saxpy(backend_handle, NULL,
                          (int)n, alpha, (fb_gpu_ptr_t)x_device, (int)incx,
                          (fb_gpu_ptr_t)y_device, (int)incy);
    
    fprintf(stderr, "[CLBlast] saxpy: CLBlast call returned\n");
    
    /* Wait for operation to complete */
    clFinish(ctx->queue);
    
    fprintf(stderr, "[CLBlast] saxpy: clFinish completed\n");
    
    /* Copy result back to host */
    clblast_copy_from_device(ctx, y, y_device, y_size);
    
    /* Free GPU buffers */
    clReleaseMemObject(x_device);
    clReleaseMemObject(y_device);
}

static void clblast_daxpy_wrapper(const int64_t n, const double alpha,
                                  const double *x, const int64_t incx,
                                  double *y, const int64_t incy) {
    void* backend_handle = clblast_get_handle_for_vtable(NULL);
    if (!backend_handle || !fb_clblast_trait.daxpy) {
        return;
    }
    
    clblast_context_t* ctx = (clblast_context_t*)backend_handle;
    
    /* Allocate GPU buffers and copy input data */
    size_t x_size = n * incx * sizeof(double);
    size_t y_size = n * incy * sizeof(double);
    
    cl_mem x_device = clblast_alloc_and_copy_to_device(ctx, x, x_size);
    cl_mem y_device = clblast_alloc_and_copy_to_device(ctx, y, y_size);
    
    if (!x_device || !y_device) {
        if (x_device) clReleaseMemObject(x_device);
        if (y_device) clReleaseMemObject(y_device);
        return;
    }
    
    /* Call CLBlast */
    fb_clblast_trait.daxpy(backend_handle, NULL,
                          (int)n, alpha, (fb_gpu_ptr_t)x_device, (int)incx,
                          (fb_gpu_ptr_t)y_device, (int)incy);
    
    /* Wait for operation to complete */
    clFinish(ctx->queue);
    
    /* Copy result back to host */
    clblast_copy_from_device(ctx, y, y_device, y_size);
    
    /* Free GPU buffers */
    clReleaseMemObject(x_device);
    clReleaseMemObject(y_device);
}

static void clblast_sscal_wrapper(const int64_t n, const float alpha,
                                  float *x, const int64_t incx) {
    void* backend_handle = clblast_get_handle_for_vtable(NULL);
    if (!fb_clblast_trait.sscal || !backend_handle) return;
    
    clblast_context_t* ctx = (clblast_context_t*)backend_handle;
    
    // Allocate GPU buffer and copy input
    size_t x_size = n * abs((int)incx) * sizeof(float);
    cl_mem x_device = clblast_alloc_and_copy_to_device(ctx, x, x_size);
    if (!x_device) return;
    
    // Call trait function with device pointer
    fb_clblast_trait.sscal(backend_handle, NULL,
                          (int)n, alpha, (fb_gpu_ptr_t)x_device, (int)incx);
    
    // Copy result back and free
    clblast_copy_from_device(ctx, x, x_device, x_size);
    clReleaseMemObject(x_device);
}

static void clblast_dscal_wrapper(const int64_t n, const double alpha,
                                  double *x, const int64_t incx) {
    void* backend_handle = clblast_get_handle_for_vtable(NULL);
    if (!fb_clblast_trait.dscal || !backend_handle) return;
    
    clblast_context_t* ctx = (clblast_context_t*)backend_handle;
    
    // Allocate GPU buffer and copy input
    size_t x_size = n * abs((int)incx) * sizeof(double);
    cl_mem x_device = clblast_alloc_and_copy_to_device(ctx, x, x_size);
    if (!x_device) return;
    
    // Call trait function with device pointer
    fb_clblast_trait.dscal(backend_handle, NULL,
                          (int)n, alpha, (fb_gpu_ptr_t)x_device, (int)incx);
    
    // Copy result back and free
    clblast_copy_from_device(ctx, x, x_device, x_size);
    clReleaseMemObject(x_device);
}

static void clblast_scopy_wrapper(const int64_t n, const float *x, const int64_t incx,
                                  float *y, const int64_t incy) {
    void* backend_handle = clblast_get_handle_for_vtable(NULL);
    if (!fb_clblast_trait.scopy || !backend_handle) return;
    
    clblast_context_t* ctx = (clblast_context_t*)backend_handle;
    
    // Allocate GPU buffers and copy inputs
    size_t x_size = n * abs((int)incx) * sizeof(float);
    size_t y_size = n * abs((int)incy) * sizeof(float);
    cl_mem x_device = clblast_alloc_and_copy_to_device(ctx, x, x_size);
    cl_mem y_device = clblast_alloc_and_copy_to_device(ctx, y, y_size);
    if (!x_device || !y_device) {
        if (x_device) clReleaseMemObject(x_device);
        if (y_device) clReleaseMemObject(y_device);
        return;
    }
    
    // Call trait function with device pointers
    fb_clblast_trait.scopy(backend_handle, NULL,
                          (int)n, (fb_gpu_ptr_t)x_device, (int)incx,
                          (fb_gpu_ptr_t)y_device, (int)incy);
    
    // Copy result back and free
    clblast_copy_from_device(ctx, y, y_device, y_size);
    clReleaseMemObject(x_device);
    clReleaseMemObject(y_device);
}

static void clblast_dcopy_wrapper(const int64_t n, const double *x, const int64_t incx,
                                  double *y, const int64_t incy) {
    void* backend_handle = clblast_get_handle_for_vtable(NULL);
    if (!fb_clblast_trait.dcopy || !backend_handle) return;
    
    clblast_context_t* ctx = (clblast_context_t*)backend_handle;
    
    // Allocate GPU buffers and copy inputs
    size_t x_size = n * abs((int)incx) * sizeof(double);
    size_t y_size = n * abs((int)incy) * sizeof(double);
    cl_mem x_device = clblast_alloc_and_copy_to_device(ctx, x, x_size);
    cl_mem y_device = clblast_alloc_and_copy_to_device(ctx, y, y_size);
    if (!x_device || !y_device) {
        if (x_device) clReleaseMemObject(x_device);
        if (y_device) clReleaseMemObject(y_device);
        return;
    }
    
    // Call trait function with device pointers
    fb_clblast_trait.dcopy(backend_handle, NULL,
                          (int)n, (fb_gpu_ptr_t)x_device, (int)incx,
                          (fb_gpu_ptr_t)y_device, (int)incy);
    
    // Copy result back and free
    clblast_copy_from_device(ctx, y, y_device, y_size);
    clReleaseMemObject(x_device);
    clReleaseMemObject(y_device);
}

/* ============================================================================
 * Level 2 BLAS Wrappers
 * ========================================================================== */

static void clblast_sgemv_wrapper(const fb_layout_t layout, const fb_transpose_t trans,
                                  const int64_t m, const int64_t n,
                                  const float alpha, const float *a, const int64_t lda,
                                  const float *x, const int64_t incx,
                                  const float beta, float *y, const int64_t incy) {
    void* backend_handle = clblast_get_handle_for_vtable(NULL);
    if (!fb_clblast_trait.sgemv || !backend_handle) return;
    
    clblast_context_t* ctx = (clblast_context_t*)backend_handle;
    
    /* Copy parameters for translation */
    fb_layout_t layout_work = layout;
    fb_transpose_t trans_work = trans;
    int64_t m_work = m, n_work = n, lda_work = lda;
    const float* a_work = a;
    
    /* Translate row-major to column-major if needed */
    translate_gemv_layout(&layout_work, &trans_work, &m_work, &n_work, &a_work, &lda_work);
    
    // Allocate GPU buffers and copy inputs
    size_t a_size = m_work * n_work * sizeof(float);  // Conservative estimate
    size_t x_size = n * abs((int)incx) * sizeof(float);
    size_t y_size = m * abs((int)incy) * sizeof(float);
    
    cl_mem a_device = clblast_alloc_and_copy_to_device(ctx, a_work, a_size);
    cl_mem x_device = clblast_alloc_and_copy_to_device(ctx, x, x_size);
    cl_mem y_device = clblast_alloc_and_copy_to_device(ctx, y, y_size);
    
    if (!a_device || !x_device || !y_device) {
        if (a_device) clReleaseMemObject(a_device);
        if (x_device) clReleaseMemObject(x_device);
        if (y_device) clReleaseMemObject(y_device);
        return;
    }
    
    // Call trait function with device pointers
    char trans_char = (trans_work == FB_NO_TRANS) ? 'N' : (trans_work == FB_TRANS) ? 'T' : 'C';
    fb_clblast_trait.sgemv(backend_handle, NULL, trans_char,
                          (int)m_work, (int)n_work, alpha,
                          (fb_gpu_ptr_t)a_device, (int)lda_work,
                          (fb_gpu_ptr_t)x_device, (int)incx,
                          beta, (fb_gpu_ptr_t)y_device, (int)incy);
    
    // Copy result back and free
    clblast_copy_from_device(ctx, y, y_device, y_size);
    clReleaseMemObject(a_device);
    clReleaseMemObject(x_device);
    clReleaseMemObject(y_device);
}

static void clblast_dgemv_wrapper(const fb_layout_t layout, const fb_transpose_t trans,
                                  const int64_t m, const int64_t n,
                                  const double alpha, const double *a, const int64_t lda,
                                  const double *x, const int64_t incx,
                                  const double beta, double *y, const int64_t incy) {
    void* backend_handle = clblast_get_handle_for_vtable(NULL);
    if (!fb_clblast_trait.dgemv || !backend_handle) return;
    
    clblast_context_t* ctx = (clblast_context_t*)backend_handle;
    
    /* Copy parameters for translation */
    fb_layout_t layout_work = layout;
    fb_transpose_t trans_work = trans;
    int64_t m_work = m, n_work = n, lda_work = lda;
    const double* a_work = a;
    
    /* Translate row-major to column-major if needed */
    translate_gemv_layout_double(&layout_work, &trans_work, &m_work, &n_work, &a_work, &lda_work);
    
    // Allocate GPU buffers and copy inputs
    size_t a_size = m_work * n_work * sizeof(double);  // Conservative estimate
    size_t x_size = n * abs((int)incx) * sizeof(double);
    size_t y_size = m * abs((int)incy) * sizeof(double);
    
    cl_mem a_device = clblast_alloc_and_copy_to_device(ctx, a_work, a_size);
    cl_mem x_device = clblast_alloc_and_copy_to_device(ctx, x, x_size);
    cl_mem y_device = clblast_alloc_and_copy_to_device(ctx, y, y_size);
    
    if (!a_device || !x_device || !y_device) {
        if (a_device) clReleaseMemObject(a_device);
        if (x_device) clReleaseMemObject(x_device);
        if (y_device) clReleaseMemObject(y_device);
        return;
    }
    
    // Call trait function with device pointers
    char trans_char = (trans_work == FB_NO_TRANS) ? 'N' : (trans_work == FB_TRANS) ? 'T' : 'C';
    fb_clblast_trait.dgemv(backend_handle, NULL, trans_char,
                          (int)m_work, (int)n_work, alpha,
                          (fb_gpu_ptr_t)a_device, (int)lda_work,
                          (fb_gpu_ptr_t)x_device, (int)incx,
                          beta, (fb_gpu_ptr_t)y_device, (int)incy);
    
    // Copy result back and free
    clblast_copy_from_device(ctx, y, y_device, y_size);
    clReleaseMemObject(a_device);
    clReleaseMemObject(x_device);
    clReleaseMemObject(y_device);
}

/* ============================================================================
 * Level 3 BLAS Wrappers
 * ========================================================================== */

static void clblast_sgemm_wrapper(const fb_layout_t layout,
                                  const fb_transpose_t transa, const fb_transpose_t transb,
                                  const int64_t m, const int64_t n, const int64_t k,
                                  const float alpha, const float *a, const int64_t lda,
                                  const float *b, const int64_t ldb,
                                  const float beta, float *c, const int64_t ldc) {
    void* backend_handle = clblast_get_handle_for_vtable(NULL);
    if (!fb_clblast_trait.sgemm || !backend_handle) return;
    
    clblast_context_t* ctx = (clblast_context_t*)backend_handle;
    
    /* Copy parameters for translation */
    fb_layout_t layout_work = layout;
    fb_transpose_t transa_work = transa, transb_work = transb;
    int64_t m_work = m, n_work = n, k_work = k;
    int64_t lda_work = lda, ldb_work = ldb, ldc_work = ldc;
    const float *a_work = a, *b_work = b;
    float *c_work = c;
    
    /* Translate row-major to column-major if needed */
    translate_gemm_layout(&layout_work, &transa_work, &transb_work,
                         &m_work, &n_work, &k_work,
                         &a_work, &lda_work, &b_work, &ldb_work,
                         &c_work, &ldc_work);
    
    // Allocate GPU buffers and copy inputs
    size_t a_size = m_work * k_work * sizeof(float);  // Conservative estimate
    size_t b_size = k_work * n_work * sizeof(float);
    size_t c_size = m_work * n_work * sizeof(float);
    
    cl_mem a_device = clblast_alloc_and_copy_to_device(ctx, a_work, a_size);
    cl_mem b_device = clblast_alloc_and_copy_to_device(ctx, b_work, b_size);
    cl_mem c_device = clblast_alloc_and_copy_to_device(ctx, c_work, c_size);
    
    if (!a_device || !b_device || !c_device) {
        if (a_device) clReleaseMemObject(a_device);
        if (b_device) clReleaseMemObject(b_device);
        if (c_device) clReleaseMemObject(c_device);
        return;
    }
    
    // Call trait function with device pointers
    char transa_char = (transa_work == FB_NO_TRANS) ? 'N' : (transa_work == FB_TRANS) ? 'T' : 'C';
    char transb_char = (transb_work == FB_NO_TRANS) ? 'N' : (transb_work == FB_TRANS) ? 'T' : 'C';
    fb_clblast_trait.sgemm(backend_handle, NULL, transa_char, transb_char,
                          (int)m_work, (int)n_work, (int)k_work, alpha,
                          (fb_gpu_ptr_t)a_device, (int)lda_work,
                          (fb_gpu_ptr_t)b_device, (int)ldb_work,
                          beta, (fb_gpu_ptr_t)c_device, (int)ldc_work);
    
    // Copy result back and free
    clblast_copy_from_device(ctx, c_work, c_device, c_size);
    clReleaseMemObject(a_device);
    clReleaseMemObject(b_device);
    clReleaseMemObject(c_device);
}

static void clblast_dgemm_wrapper(const fb_layout_t layout,
                                  const fb_transpose_t transa, const fb_transpose_t transb,
                                  const int64_t m, const int64_t n, const int64_t k,
                                  const double alpha, const double *a, const int64_t lda,
                                  const double *b, const int64_t ldb,
                                  const double beta, double *c, const int64_t ldc) {
    void* backend_handle = clblast_get_handle_for_vtable(NULL);
    if (!fb_clblast_trait.dgemm || !backend_handle) return;
    
    clblast_context_t* ctx = (clblast_context_t*)backend_handle;
    
    /* Copy parameters for translation */
    fb_layout_t layout_work = layout;
    fb_transpose_t transa_work = transa, transb_work = transb;
    int64_t m_work = m, n_work = n, k_work = k;
    int64_t lda_work = lda, ldb_work = ldb, ldc_work = ldc;
    const double *a_work = a, *b_work = b;
    double *c_work = c;
    
    /* Translate row-major to column-major if needed */
    translate_gemm_layout_double(&layout_work, &transa_work, &transb_work,
                                &m_work, &n_work, &k_work,
                                &a_work, &lda_work, &b_work, &ldb_work,
                                &c_work, &ldc_work);
    
    // Allocate GPU buffers and copy inputs
    size_t a_size = m_work * k_work * sizeof(double);  // Conservative estimate
    size_t b_size = k_work * n_work * sizeof(double);
    size_t c_size = m_work * n_work * sizeof(double);
    
    cl_mem a_device = clblast_alloc_and_copy_to_device(ctx, a_work, a_size);
    cl_mem b_device = clblast_alloc_and_copy_to_device(ctx, b_work, b_size);
    cl_mem c_device = clblast_alloc_and_copy_to_device(ctx, c_work, c_size);
    
    if (!a_device || !b_device || !c_device) {
        if (a_device) clReleaseMemObject(a_device);
        if (b_device) clReleaseMemObject(b_device);
        if (c_device) clReleaseMemObject(c_device);
        return;
    }
    
    // Call trait function with device pointers
    char transa_char = (transa_work == FB_NO_TRANS) ? 'N' : (transa_work == FB_TRANS) ? 'T' : 'C';
    char transb_char = (transb_work == FB_NO_TRANS) ? 'N' : (transb_work == FB_TRANS) ? 'T' : 'C';
    fb_clblast_trait.dgemm(backend_handle, NULL, transa_char, transb_char,
                          (int)m_work, (int)n_work, (int)k_work, alpha,
                          (fb_gpu_ptr_t)a_device, (int)lda_work,
                          (fb_gpu_ptr_t)b_device, (int)ldb_work,
                          beta, (fb_gpu_ptr_t)c_device, (int)ldc_work);
    
    // Copy result back and free
    clblast_copy_from_device(ctx, c_work, c_device, c_size);
    clReleaseMemObject(a_device);
    clReleaseMemObject(b_device);
    clReleaseMemObject(c_device);
}

/* ============================================================================
 * Unified CPU/GPU Interface - Memory Management
 * ========================================================================== */

static int clblast_mem_alloc(void* handle, void** ptr, size_t size) {
    if (!fb_clblast_trait.malloc || !handle) return -1;
    fb_gpu_ptr_t gpu_ptr;
    int result = fb_clblast_trait.malloc(handle, &gpu_ptr, size);
    if (result == 0) {
        *ptr = (void*)gpu_ptr;
    }
    return result;
}

static void clblast_mem_free(void* handle, void* ptr) {
    if (fb_clblast_trait.free && handle) {
        fb_clblast_trait.free(handle, (fb_gpu_ptr_t)ptr);
    }
}

static int clblast_mem_upload(void* handle, void* dst, const void* src, size_t size) {
    if (!fb_clblast_trait.memcpy_h2d || !handle) return -1;
    return fb_clblast_trait.memcpy_h2d(handle, (fb_gpu_ptr_t)dst, src, size);
}

static int clblast_mem_download(void* handle, void* dst, const void* src, size_t size) {
    if (!fb_clblast_trait.memcpy_d2h || !handle) return -1;
    return fb_clblast_trait.memcpy_d2h(handle, dst, (fb_gpu_ptr_t)src, size);
}

static int clblast_mem_copy(void* handle, void* dst, const void* src, size_t size) {
    if (!fb_clblast_trait.memcpy_d2d || !handle) return -1;
    return fb_clblast_trait.memcpy_d2d(handle, (fb_gpu_ptr_t)dst, (fb_gpu_ptr_t)src, size);
}

/* ============================================================================
 * Unified CPU/GPU Interface - Stream Management
 * ========================================================================== */

static int clblast_stream_create(void* handle, void** stream) {
    if (!fb_clblast_trait.stream_create || !handle) return -1;
    fb_gpu_stream_t gpu_stream;
    int result = fb_clblast_trait.stream_create(handle, &gpu_stream);
    if (result == 0) {
        *stream = (void*)gpu_stream;
    }
    return result;
}

static void clblast_stream_destroy(void* handle, void* stream) {
    if (fb_clblast_trait.stream_destroy && handle) {
        fb_clblast_trait.stream_destroy(handle, (fb_gpu_stream_t)stream);
    }
}

static int clblast_stream_sync(void* handle, void* stream) {
    if (!fb_clblast_trait.stream_synchronize || !handle) return -1;
    return fb_clblast_trait.stream_synchronize(handle, (fb_gpu_stream_t)stream);
}

static int clblast_stream_set(void* handle, void* stream) {
    /* CLBlast may not have explicit stream setting - would need to track active stream */
    (void)handle; (void)stream;
    return 0;  /* TODO: Implement if CLBlast supports it */
}

/* ============================================================================
 * Unified CPU/GPU Interface - Backend Properties
 * ========================================================================== */

static uint32_t clblast_get_capabilities(void* handle) {
    (void)handle;
    return FB_PLUGIN_CAP_GPU | FB_PLUGIN_CAP_LEVEL1 | FB_PLUGIN_CAP_LEVEL2 | FB_PLUGIN_CAP_LEVEL3 |
           FB_PLUGIN_CAP_SINGLE_PREC | FB_PLUGIN_CAP_DOUBLE_PREC | FB_PLUGIN_CAP_COMPLEX;
}

static int clblast_get_num_threads(void* handle) {
    (void)handle;
    return -1;  /* Not applicable for GPU */
}

static void clblast_set_num_threads(void* handle, int num_threads) {
    (void)handle; (void)num_threads;
    /* Not applicable for GPU */
}

/* ============================================================================
 * Vtable Population
 * ========================================================================== */

void clblast_populate_vtable(fb_backend_vtable_t* vtable) {
    if (!vtable) {
        fprintf(stderr, "[CLBlast] ERROR: vtable is NULL\n");
        return;
    }
    
    memset(vtable, 0, sizeof(fb_backend_vtable_t));
    
    /* Level 1 BLAS */
    vtable->saxpy = clblast_saxpy_wrapper;
    vtable->daxpy = clblast_daxpy_wrapper;
    vtable->sscal = clblast_sscal_wrapper;
    vtable->dscal = clblast_dscal_wrapper;
    vtable->scopy = clblast_scopy_wrapper;
    vtable->dcopy = clblast_dcopy_wrapper;
    
    /* Level 2 BLAS */
    vtable->sgemv = clblast_sgemv_wrapper;
    vtable->dgemv = clblast_dgemv_wrapper;
    
    /* Level 3 BLAS */
    vtable->sgemm = clblast_sgemm_wrapper;
    vtable->dgemm = clblast_dgemm_wrapper;
    
    /* Unified CPU/GPU Interface - Memory Management */
    vtable->mem_alloc = clblast_mem_alloc;
    vtable->mem_free = clblast_mem_free;
    fprintf(stderr, "[CLBlast] vtable->mem_free = %p\n", (void*)vtable->mem_free);
    vtable->mem_upload = clblast_mem_upload;
    vtable->mem_download = clblast_mem_download;
    vtable->mem_copy = clblast_mem_copy;
    
    /* Unified CPU/GPU Interface - Stream Management */
    vtable->stream_create = clblast_stream_create;
    vtable->stream_destroy = clblast_stream_destroy;
    fprintf(stderr, "[CLBlast] vtable->stream_destroy = %p\n", (void*)vtable->stream_destroy);
    vtable->stream_sync = clblast_stream_sync;
    vtable->stream_set = clblast_stream_set;
    
    /* Unified CPU/GPU Interface - Backend Properties */
    vtable->get_capabilities = clblast_get_capabilities;
    vtable->get_num_threads = clblast_get_num_threads;
    vtable->set_num_threads = clblast_set_num_threads;
}

#endif /* FB_ENABLE_OPENCL */
