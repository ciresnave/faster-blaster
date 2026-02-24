/**
 * @file cublas_complete_smart_wrappers.c
 * @brief Complete smart wrappers for BLAS Level 1, 2, and 3 operations
 * 
 * This file provides complete implementations of smart wrappers for all
 * Phase 1 operations, properly calling the GPU trait functions.
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "faster-blaster/gpu_backend_trait.h"
#include "faster-blaster/device_memory_manager.h"
#include "faster_blaster.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* External cuBLAS trait and handle */
extern const fb_gpu_backend_trait_t fb_cublas_trait;
extern void* g_cublas_handle;  /* Shared with cublas_smart_wrappers.c */
static const int CUBLAS_DEVICE_ID = 0;

/* Helper function prototypes */
static fb_device_memory_manager_t* get_device_manager(void);
static size_t calculate_vector_size(int n, int inc);
static size_t calculate_matrix_size_col_major(int rows, int cols, int ld);
static int get_device_buffer_input(const void* host_ptr, size_t size, fb_gpu_ptr_t* device_ptr_out);
static int get_device_buffer_output(void* host_ptr, size_t size, int is_inout, fb_gpu_ptr_t* device_ptr_out);
static void mark_output_dirty(void* host_ptr);
static void release_device_buffer(const void* host_ptr);

/* ============================================================================
 * Helper Functions (Implementation)
 * ========================================================================== */

static fb_device_memory_manager_t* get_device_manager(void) {
    return fb_device_memory_get_manager(CUBLAS_DEVICE_ID);
}

static size_t calculate_vector_size(int n, int inc) {
    if (n <= 0 || inc == 0) return 0;
    return (size_t)(1 + (n - 1) * abs(inc));
}

static size_t calculate_matrix_size_col_major(int rows, int cols, int ld) {
    return (size_t)ld * cols;
}

static int get_device_buffer_input(const void* host_ptr, size_t size, fb_gpu_ptr_t* device_ptr_out) {
    fb_device_memory_manager_t* manager = get_device_manager();
    if (!manager) return -1;
    
    return fb_device_memory_get_or_alloc(manager, &fb_cublas_trait, g_cublas_handle,
                                        host_ptr, size, FB_GPU_BACKEND_CUBLAS, device_ptr_out);
}

static int get_device_buffer_output(void* host_ptr, size_t size, int is_inout, fb_gpu_ptr_t* device_ptr_out) {
    return get_device_buffer_input(host_ptr, size, device_ptr_out);
}

static void mark_output_dirty(void* host_ptr) {
    fb_device_memory_manager_t* manager = get_device_manager();
    if (manager) {
        fb_device_memory_mark_dirty(manager, host_ptr, FB_GPU_BACKEND_CUBLAS);
    }
}

static void release_device_buffer(const void* host_ptr) {
    fb_device_memory_manager_t* manager = get_device_manager();
    if (manager) {
        fb_device_memory_release(manager, host_ptr);
    }
}

/* ============================================================================
 * BLAS Level 1: Vector Operations
 * ========================================================================== */

/** saxpy: y = alpha*x + y */
void cublas_saxpy_smart_wrapper(int n, float alpha, const float* x, int incx, float* y, int incy) {
    if (!x || !y || n <= 0 || !g_cublas_handle) return;
    
    size_t size_x = calculate_vector_size(n, incx) * sizeof(float);
    size_t size_y = calculate_vector_size(n, incy) * sizeof(float);
    
    fb_gpu_ptr_t d_x = NULL, d_y = NULL;
    
    if (get_device_buffer_input(x, size_x, &d_x) != 0 ||
        get_device_buffer_output(y, size_y, 1, &d_y) != 0) {
        return;
    }
    
    if (fb_cublas_trait.saxpy) {
        fb_cublas_trait.saxpy(g_cublas_handle, NULL, n, alpha, d_x, incx, d_y, incy);
        mark_output_dirty(y);
    }
    
    release_device_buffer(x);
    release_device_buffer(y);
}

/** daxpy: y = alpha*x + y */
void cublas_daxpy_smart_wrapper(int n, double alpha, const double* x, int incx, double* y, int incy) {
    if (!x || !y || n <= 0 || !g_cublas_handle) return;
    
    size_t size_x = calculate_vector_size(n, incx) * sizeof(double);
    size_t size_y = calculate_vector_size(n, incy) * sizeof(double);
    
    fb_gpu_ptr_t d_x = NULL, d_y = NULL;
    
    if (get_device_buffer_input(x, size_x, &d_x) != 0 ||
        get_device_buffer_output(y, size_y, 1, &d_y) != 0) {
        return;
    }
    
    if (fb_cublas_trait.daxpy) {
        fb_cublas_trait.daxpy(g_cublas_handle, NULL, n, alpha, d_x, incx, d_y, incy);
        mark_output_dirty(y);
    }
    
    release_device_buffer(x);
    release_device_buffer(y);
}

/** sscal: x = alpha*x */
void cublas_sscal_smart_wrapper(int n, float alpha, float* x, int incx) {
    if (!x || n <= 0 || !g_cublas_handle) return;
    
    size_t size_x = calculate_vector_size(n, incx) * sizeof(float);
    fb_gpu_ptr_t d_x = NULL;
    
    if (get_device_buffer_output(x, size_x, 1, &d_x) != 0) return;
    
    if (fb_cublas_trait.sscal) {
        fb_cublas_trait.sscal(g_cublas_handle, NULL, n, alpha, d_x, incx);
        mark_output_dirty(x);
    }
    
    release_device_buffer(x);
}

/** dscal: x = alpha*x */
void cublas_dscal_smart_wrapper(int n, double alpha, double* x, int incx) {
    if (!x || n <= 0 || !g_cublas_handle) return;
    
    size_t size_x = calculate_vector_size(n, incx) * sizeof(double);
    fb_gpu_ptr_t d_x = NULL;
    
    if (get_device_buffer_output(x, size_x, 1, &d_x) != 0) return;
    
    if (fb_cublas_trait.dscal) {
        fb_cublas_trait.dscal(g_cublas_handle, NULL, n, alpha, d_x, incx);
        mark_output_dirty(x);
    }
    
    release_device_buffer(x);
}

/** scopy: y = x */
void cublas_scopy_smart_wrapper(int n, const float* x, int incx, float* y, int incy) {
    if (!x || !y || n <= 0 || !g_cublas_handle) return;
    
    size_t size_x = calculate_vector_size(n, incx) * sizeof(float);
    size_t size_y = calculate_vector_size(n, incy) * sizeof(float);
    
    fb_gpu_ptr_t d_x = NULL, d_y = NULL;
    
    if (get_device_buffer_input(x, size_x, &d_x) != 0 ||
        get_device_buffer_output(y, size_y, 0, &d_y) != 0) {
        return;
    }
    
    if (fb_cublas_trait.scopy) {
        fb_cublas_trait.scopy(g_cublas_handle, NULL, n, d_x, incx, d_y, incy);
        mark_output_dirty(y);
    }
    
    release_device_buffer(x);
    release_device_buffer(y);
}

/** dcopy: y = x */
void cublas_dcopy_smart_wrapper(int n, const double* x, int incx, double* y, int incy) {
    if (!x || !y || n <= 0 || !g_cublas_handle) return;
    
    size_t size_x = calculate_vector_size(n, incx) * sizeof(double);
    size_t size_y = calculate_vector_size(n, incy) * sizeof(double);
    
    fb_gpu_ptr_t d_x = NULL, d_y = NULL;
    
    if (get_device_buffer_input(x, size_x, &d_x) != 0 ||
        get_device_buffer_output(y, size_y, 0, &d_y) != 0) {
        return;
    }
    
    if (fb_cublas_trait.dcopy) {
        fb_cublas_trait.dcopy(g_cublas_handle, NULL, n, d_x, incx, d_y, incy);
        mark_output_dirty(y);
    }
    
    release_device_buffer(x);
    release_device_buffer(y);
}

/** sswap: swap x and y */
void cublas_sswap_smart_wrapper(int n, float* x, int incx, float* y, int incy) {
    if (!x || !y || n <= 0 || !g_cublas_handle) return;
    
    size_t size_x = calculate_vector_size(n, incx) * sizeof(float);
    size_t size_y = calculate_vector_size(n, incy) * sizeof(float);
    
    fb_gpu_ptr_t d_x = NULL, d_y = NULL;
    
    if (get_device_buffer_output(x, size_x, 1, &d_x) != 0 ||
        get_device_buffer_output(y, size_y, 1, &d_y) != 0) {
        return;
    }
    
    if (fb_cublas_trait.sswap) {
        fb_cublas_trait.sswap(g_cublas_handle, NULL, n, d_x, incx, d_y, incy);
        mark_output_dirty(x);
        mark_output_dirty(y);
    }
    
    release_device_buffer(x);
    release_device_buffer(y);
}

/** dswap: swap x and y */
void cublas_dswap_smart_wrapper(int n, double* x, int incx, double* y, int incy) {
    if (!x || !y || n <= 0 || !g_cublas_handle) return;
    
    size_t size_x = calculate_vector_size(n, incx) * sizeof(double);
    size_t size_y = calculate_vector_size(n, incy) * sizeof(double);
    
    fb_gpu_ptr_t d_x = NULL, d_y = NULL;
    
    if (get_device_buffer_output(x, size_x, 1, &d_x) != 0 ||
        get_device_buffer_output(y, size_y, 1, &d_y) != 0) {
        return;
    }
    
    if (fb_cublas_trait.dswap) {
        fb_cublas_trait.dswap(g_cublas_handle, NULL, n, d_x, incx, d_y, incy);
        mark_output_dirty(x);
        mark_output_dirty(y);
    }
    
    release_device_buffer(x);
    release_device_buffer(y);
}

/** sdot: dot product */
float cublas_sdot_smart_wrapper(int n, const float* x, int incx, const float* y, int incy) {
    if (!x || !y || n <= 0 || !g_cublas_handle) return 0.0f;
    
    size_t size_x = calculate_vector_size(n, incx) * sizeof(float);
    size_t size_y = calculate_vector_size(n, incy) * sizeof(float);
    
    fb_gpu_ptr_t d_x = NULL, d_y = NULL;
    float result = 0.0f;
    
    if (get_device_buffer_input(x, size_x, &d_x) != 0 ||
        get_device_buffer_input(y, size_y, &d_y) != 0) {
        return 0.0f;
    }
    
    if (fb_cublas_trait.sdot) {
        result = fb_cublas_trait.sdot(g_cublas_handle, NULL, n, d_x, incx, d_y, incy);
    }
    
    release_device_buffer(x);
    release_device_buffer(y);
    
    return result;
}

/** ddot: dot product */
double cublas_ddot_smart_wrapper(int n, const double* x, int incx, const double* y, int incy) {
    if (!x || !y || n <= 0 || !g_cublas_handle) return 0.0;
    
    size_t size_x = calculate_vector_size(n, incx) * sizeof(double);
    size_t size_y = calculate_vector_size(n, incy) * sizeof(double);
    
    fb_gpu_ptr_t d_x = NULL, d_y = NULL;
    double result = 0.0;
    
    if (get_device_buffer_input(x, size_x, &d_x) != 0 ||
        get_device_buffer_input(y, size_y, &d_y) != 0) {
        return 0.0;
    }
    
    if (fb_cublas_trait.ddot) {
        result = fb_cublas_trait.ddot(g_cublas_handle, NULL, n, d_x, incx, d_y, incy);
    }
    
    release_device_buffer(x);
    release_device_buffer(y);
    
    return result;
}

/** snrm2: Euclidean norm */
float cublas_snrm2_smart_wrapper(int n, const float* x, int incx) {
    if (!x || n <= 0 || !g_cublas_handle) return 0.0f;
    
    size_t size_x = calculate_vector_size(n, incx) * sizeof(float);
    fb_gpu_ptr_t d_x = NULL;
    float result = 0.0f;
    
    if (get_device_buffer_input(x, size_x, &d_x) != 0) return 0.0f;
    
    if (fb_cublas_trait.snrm2) {
        result = fb_cublas_trait.snrm2(g_cublas_handle, NULL, n, d_x, incx);
    }
    
    release_device_buffer(x);
    return result;
}

/** dnrm2: Euclidean norm */
double cublas_dnrm2_smart_wrapper(int n, const double* x, int incx) {
    if (!x || n <= 0 || !g_cublas_handle) return 0.0;
    
    size_t size_x = calculate_vector_size(n, incx) * sizeof(double);
    fb_gpu_ptr_t d_x = NULL;
    double result = 0.0;
    
    if (get_device_buffer_input(x, size_x, &d_x) != 0) return 0.0;
    
    if (fb_cublas_trait.dnrm2) {
        result = fb_cublas_trait.dnrm2(g_cublas_handle, NULL, n, d_x, incx);
    }
    
    release_device_buffer(x);
    return result;
}

/** sasum: sum of absolute values */
float cublas_sasum_smart_wrapper(int n, const float* x, int incx) {
    if (!x || n <= 0 || !g_cublas_handle) return 0.0f;
    
    size_t size_x = calculate_vector_size(n, incx) * sizeof(float);
    fb_gpu_ptr_t d_x = NULL;
    float result = 0.0f;
    
    if (get_device_buffer_input(x, size_x, &d_x) != 0) return 0.0f;
    
    if (fb_cublas_trait.sasum) {
        result = fb_cublas_trait.sasum(g_cublas_handle, NULL, n, d_x, incx);
    }
    
    release_device_buffer(x);
    return result;
}

/** dasum: sum of absolute values */
double cublas_dasum_smart_wrapper(int n, const double* x, int incx) {
    if (!x || n <= 0 || !g_cublas_handle) return 0.0;
    
    size_t size_x = calculate_vector_size(n, incx) * sizeof(double);
    fb_gpu_ptr_t d_x = NULL;
    double result = 0.0;
    
    if (get_device_buffer_input(x, size_x, &d_x) != 0) return 0.0;
    
    if (fb_cublas_trait.dasum) {
        result = fb_cublas_trait.dasum(g_cublas_handle, NULL, n, d_x, incx);
    }
    
    release_device_buffer(x);
    return result;
}

/** isamax: index of maximum absolute value */
int cublas_isamax_smart_wrapper(int n, const float* x, int incx) {
    if (!x || n <= 0 || !g_cublas_handle) return 0;
    
    size_t size_x = calculate_vector_size(n, incx) * sizeof(float);
    fb_gpu_ptr_t d_x = NULL;
    int result = 0;
    
    if (get_device_buffer_input(x, size_x, &d_x) != 0) return 0;
    
    if (fb_cublas_trait.isamax) {
        result = fb_cublas_trait.isamax(g_cublas_handle, NULL, n, d_x, incx);
    }
    
    release_device_buffer(x);
    return result;
}

/** idamax: index of maximum absolute value */
int cublas_idamax_smart_wrapper(int n, const double* x, int incx) {
    if (!x || n <= 0 || !g_cublas_handle) return 0;
    
    size_t size_x = calculate_vector_size(n, incx) * sizeof(double);
    fb_gpu_ptr_t d_x = NULL;
    int result = 0;
    
    if (get_device_buffer_input(x, size_x, &d_x) != 0) return 0;
    
    if (fb_cublas_trait.idamax) {
        result = fb_cublas_trait.idamax(g_cublas_handle, NULL, n, d_x, incx);
    }
    
    release_device_buffer(x);
    return result;
}

/* ============================================================================
 * BLAS Level 2: Matrix-Vector Operations
 * ========================================================================== */

/** sger: rank-1 update A = alpha*x*y' + A */
void cublas_sger_smart_wrapper(int m, int n, float alpha,
                                const float* x, int incx,
                                const float* y, int incy,
                                float* a, int lda) {
    if (!x || !y || !a || m <= 0 || n <= 0 || !g_cublas_handle) return;
    
    size_t size_x = calculate_vector_size(m, incx) * sizeof(float);
    size_t size_y = calculate_vector_size(n, incy) * sizeof(float);
    size_t size_a = calculate_matrix_size_col_major(m, n, lda) * sizeof(float);
    
    fb_gpu_ptr_t d_x = NULL, d_y = NULL, d_a = NULL;
    
    if (get_device_buffer_input(x, size_x, &d_x) != 0 ||
        get_device_buffer_input(y, size_y, &d_y) != 0 ||
        get_device_buffer_output(a, size_a, 1, &d_a) != 0) {
        return;
    }
    
    if (fb_cublas_trait.sger) {
        fb_cublas_trait.sger(g_cublas_handle, NULL, m, n, alpha,
                             d_x, incx, d_y, incy, d_a, lda);
        mark_output_dirty(a);
    }
    
    release_device_buffer(x);
    release_device_buffer(y);
    release_device_buffer(a);
}

/** dger: rank-1 update A = alpha*x*y' + A */
void cublas_dger_smart_wrapper(int m, int n, double alpha,
                                const double* x, int incx,
                                const double* y, int incy,
                                double* a, int lda) {
    if (!x || !y || !a || m <= 0 || n <= 0 || !g_cublas_handle) return;
    
    size_t size_x = calculate_vector_size(m, incx) * sizeof(double);
    size_t size_y = calculate_vector_size(n, incy) * sizeof(double);
    size_t size_a = calculate_matrix_size_col_major(m, n, lda) * sizeof(double);
    
    fb_gpu_ptr_t d_x = NULL, d_y = NULL, d_a = NULL;
    
    if (get_device_buffer_input(x, size_x, &d_x) != 0 ||
        get_device_buffer_input(y, size_y, &d_y) != 0 ||
        get_device_buffer_output(a, size_a, 1, &d_a) != 0) {
        return;
    }
    
    if (fb_cublas_trait.dger) {
        fb_cublas_trait.dger(g_cublas_handle, NULL, m, n, alpha,
                             d_x, incx, d_y, incy, d_a, lda);
        mark_output_dirty(a);
    }
    
    release_device_buffer(x);
    release_device_buffer(y);
    release_device_buffer(a);
}

/* ============================================================================
 * BLAS Level 3: Matrix-Matrix Operations
 * ========================================================================== */

/** sgemm: C = alpha*op(A)*op(B) + beta*C */
void cublas_sgemm_smart_wrapper(FB_LAYOUT layout, FB_TRANSPOSE transa, FB_TRANSPOSE transb,
                                 int m, int n, int k, float alpha,
                                 const float* a, int lda,
                                 const float* b, int ldb,
                                 float beta, float* c, int ldc) {
    if (!a || !b || !c || m <= 0 || n <= 0 || k <= 0 || !g_cublas_handle) return;
    
    /* Calculate sizes based on transpose flags */
    int rows_a = (transa == FbNoTrans) ? m : k;
    int cols_a = (transa == FbNoTrans) ? k : m;
    int rows_b = (transb == FbNoTrans) ? k : n;
    int cols_b = (transb == FbNoTrans) ? n : k;
    
    size_t size_a = calculate_matrix_size_col_major(rows_a, cols_a, lda) * sizeof(float);
    size_t size_b = calculate_matrix_size_col_major(rows_b, cols_b, ldb) * sizeof(float);
    size_t size_c = calculate_matrix_size_col_major(m, n, ldc) * sizeof(float);
    
    fb_gpu_ptr_t d_a = NULL, d_b = NULL, d_c = NULL;
    
    if (get_device_buffer_input(a, size_a, &d_a) != 0 ||
        get_device_buffer_input(b, size_b, &d_b) != 0 ||
        get_device_buffer_output(c, size_c, (beta != 0.0f), &d_c) != 0) {
        return;
    }
    
    char trans_a = (transa == FbNoTrans) ? 'N' : (transa == FbTrans) ? 'T' : 'C';
    char trans_b = (transb == FbNoTrans) ? 'N' : (transb == FbTrans) ? 'T' : 'C';
    
    if (fb_cublas_trait.sgemm) {
        fb_cublas_trait.sgemm(g_cublas_handle, NULL, trans_a, trans_b,
                              m, n, k, alpha, d_a, lda, d_b, ldb,
                              beta, d_c, ldc);
        mark_output_dirty(c);
    }
    
    release_device_buffer(a);
    release_device_buffer(b);
    release_device_buffer(c);
}

/** dgemm: C = alpha*op(A)*op(B) + beta*C */
void cublas_dgemm_smart_wrapper(FB_LAYOUT layout, FB_TRANSPOSE transa, FB_TRANSPOSE transb,
                                 int m, int n, int k, double alpha,
                                 const double* a, int lda,
                                 const double* b, int ldb,
                                 double beta, double* c, int ldc) {
    if (!a || !b || !c || m <= 0 || n <= 0 || k <= 0 || !g_cublas_handle) return;
    
    int rows_a = (transa == FbNoTrans) ? m : k;
    int cols_a = (transa == FbNoTrans) ? k : m;
    int rows_b = (transb == FbNoTrans) ? k : n;
    int cols_b = (transb == FbNoTrans) ? n : k;
    
    size_t size_a = calculate_matrix_size_col_major(rows_a, cols_a, lda) * sizeof(double);
    size_t size_b = calculate_matrix_size_col_major(rows_b, cols_b, ldb) * sizeof(double);
    size_t size_c = calculate_matrix_size_col_major(m, n, ldc) * sizeof(double);
    
    fb_gpu_ptr_t d_a = NULL, d_b = NULL, d_c = NULL;
    
    if (get_device_buffer_input(a, size_a, &d_a) != 0 ||
        get_device_buffer_input(b, size_b, &d_b) != 0 ||
        get_device_buffer_output(c, size_c, (beta != 0.0), &d_c) != 0) {
        return;
    }
    
    char trans_a = (transa == FbNoTrans) ? 'N' : (transa == FbTrans) ? 'T' : 'C';
    char trans_b = (transb == FbNoTrans) ? 'N' : (transb == FbTrans) ? 'T' : 'C';
    
    if (fb_cublas_trait.dgemm) {
        fb_cublas_trait.dgemm(g_cublas_handle, NULL, trans_a, trans_b,
                              m, n, k, alpha, d_a, lda, d_b, ldb,
                              beta, d_c, ldc);
        mark_output_dirty(c);
    }
    
    release_device_buffer(a);
    release_device_buffer(b);
    release_device_buffer(c);
}
