/**
 * @file cublas_blas_level2_smart.c
 * @brief Smart wrappers for BLAS Level 2 operations (Phase 1 - Real-valued)
 * 
 * This file provides smart wrappers for Level 2 operations:
 * - Matrix-vector: gbmv (banded), symv (symmetric)
 * - Triangular: trmv, trsv
 * - Rank updates: syr, syr2
 * - Banded: sbmv, tbmv, tbsv
 * - Packed: spmv, tpmv, tpsv, spr, spr2
 * 
 * All operations use the device memory manager for intelligent caching.
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
extern void* g_cublas_handle;
static const int CUBLAS_DEVICE_ID = 0;

/* Helper function prototypes */
static fb_device_memory_manager_t* get_device_manager(void);
static size_t calculate_vector_size(int n, int inc);
static size_t calculate_matrix_size_col_major(int rows, int cols, int ld);
static size_t calculate_banded_size(int n, int k, int ld);
static size_t calculate_packed_size(int n);
static int get_device_buffer_input(const void* host_ptr, size_t size, fb_gpu_ptr_t* device_ptr_out);
static int get_device_buffer_inout(void* host_ptr, size_t size, fb_gpu_ptr_t* device_ptr_out);
static void mark_output_dirty(void* host_ptr);
static void release_device_buffer(const void* host_ptr);

/* ============================================================================
 * Helper Functions
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

static size_t calculate_banded_size(int n, int k, int ld) {
    /* Banded matrix storage: ld rows × n columns */
    return (size_t)ld * n;
}

static size_t calculate_packed_size(int n) {
    /* Packed storage: n*(n+1)/2 elements */
    return (size_t)(n * (n + 1) / 2);
}

static int get_device_buffer_input(const void* host_ptr, size_t size, fb_gpu_ptr_t* device_ptr_out) {
    fb_device_memory_manager_t* manager = get_device_manager();
    if (!manager) return -1;
    return fb_device_memory_get_or_alloc(manager, &fb_cublas_trait, g_cublas_handle,
                                         host_ptr, size, FB_GPU_BACKEND_CUBLAS, device_ptr_out);
}

static int get_device_buffer_inout(void* host_ptr, size_t size, fb_gpu_ptr_t* device_ptr_out) {
    fb_device_memory_manager_t* manager = get_device_manager();
    if (!manager) return -1;
    return fb_device_memory_get_or_alloc(manager, &fb_cublas_trait, g_cublas_handle,
                                         host_ptr, size, FB_GPU_BACKEND_CUBLAS, device_ptr_out);
}

static void mark_output_dirty(void* host_ptr) {
    /* Mark dirty first (device has authoritative data from GPU op), THEN sync.
     * Without marking dirty first, sync_to_host returns early (dirty==0 guard).
     * This ensures callers immediately see GPU results without an explicit sync. */
    fb_device_memory_manager_t* manager = get_device_manager();
    if (!manager) return;
    fb_device_memory_mark_dirty(manager, host_ptr, FB_GPU_BACKEND_CUBLAS);
    fb_device_memory_sync_to_host(manager, &fb_cublas_trait, g_cublas_handle, host_ptr);
}

static void release_device_buffer(const void* host_ptr) {
    fb_device_memory_manager_t* manager = get_device_manager();
    if (manager) fb_device_memory_release(manager, host_ptr);
}

/* ============================================================================
 * BLAS Level 2: Banded Matrix-Vector Multiplication (2 ops)
 * ========================================================================== */

/**
 * @brief General banded matrix-vector multiply (single precision)
 * y := alpha*A*x + beta*y  or  y := alpha*A^T*x + beta*y
 * A is m×n with kl sub-diagonals and ku super-diagonals
 */
void cublas_sgbmv_smart_wrapper(FB_TRANSPOSE trans, int m, int n, int kl, int ku,
                                 float alpha, const float* a, int lda,
                                 const float* x, int incx, float beta,
                                 float* y, int incy) {
    if (m <= 0 || n <= 0) return;
    
    /* Determine dimensions based on transpose */
    int x_len = (trans == FbNoTrans) ? n : m;
    int y_len = (trans == FbNoTrans) ? m : n;
    
    size_t size_a = calculate_banded_size(n, kl + ku + 1, lda) * sizeof(float);
    size_t size_x = calculate_vector_size(x_len, incx) * sizeof(float);
    size_t size_y = calculate_vector_size(y_len, incy) * sizeof(float);
    
    /* Get device buffers */
    fb_gpu_ptr_t d_a, d_x, d_y;
    if (get_device_buffer_input(a, size_a, &d_a) != 0) return;
    if (get_device_buffer_input(x, size_x, &d_x) != 0) {
        release_device_buffer(a);
        return;
    }
    if (get_device_buffer_inout(y, size_y, &d_y) != 0) {
        release_device_buffer(a);
        release_device_buffer(x);
        return;
    }
    
    /* Convert transpose */
    char trans_char = (trans == FbNoTrans) ? 'N' : 
                      (trans == FbTrans) ? 'T' : 'C';
    
    /* Execute GPU operation */
    if (fb_cublas_trait.sgbmv) {
        fb_cublas_trait.sgbmv(g_cublas_handle, NULL, trans_char, m, n, kl, ku,
                              alpha, d_a, lda, d_x, incx, beta, d_y, incy);
    }
    
    mark_output_dirty(y);
    release_device_buffer(a);
    release_device_buffer(x);
    release_device_buffer(y);
}

/**
 * @brief General banded matrix-vector multiply (double precision)
 */
void cublas_dgbmv_smart_wrapper(FB_TRANSPOSE trans, int m, int n, int kl, int ku,
                                 double alpha, const double* a, int lda,
                                 const double* x, int incx, double beta,
                                 double* y, int incy) {
    if (m <= 0 || n <= 0) return;
    
    int x_len = (trans == FbNoTrans) ? n : m;
    int y_len = (trans == FbNoTrans) ? m : n;
    
    size_t size_a = calculate_banded_size(n, kl + ku + 1, lda) * sizeof(double);
    size_t size_x = calculate_vector_size(x_len, incx) * sizeof(double);
    size_t size_y = calculate_vector_size(y_len, incy) * sizeof(double);
    
    fb_gpu_ptr_t d_a, d_x, d_y;
    if (get_device_buffer_input(a, size_a, &d_a) != 0) return;
    if (get_device_buffer_input(x, size_x, &d_x) != 0) {
        release_device_buffer(a);
        return;
    }
    if (get_device_buffer_inout(y, size_y, &d_y) != 0) {
        release_device_buffer(a);
        release_device_buffer(x);
        return;
    }
    
    char trans_char = (trans == FbNoTrans) ? 'N' : 
                      (trans == FbTrans) ? 'T' : 'C';
    
    if (fb_cublas_trait.dgbmv) {
        fb_cublas_trait.dgbmv(g_cublas_handle, NULL, trans_char, m, n, kl, ku,
                              alpha, d_a, lda, d_x, incx, beta, d_y, incy);
    }
    
    mark_output_dirty(y);
    release_device_buffer(a);
    release_device_buffer(x);
    release_device_buffer(y);
}

/* ============================================================================
 * BLAS Level 2: Symmetric Matrix-Vector Multiplication (2 ops)
 * ========================================================================== */

/**
 * @brief Symmetric matrix-vector multiply (single precision)
 * y := alpha*A*x + beta*y, where A is n×n symmetric
 */
void cublas_ssymv_smart_wrapper(char uplo, int n, float alpha,
                                 const float* a, int lda, const float* x, int incx,
                                 float beta, float* y, int incy) {
    if (n <= 0) return;
    
    size_t size_a = calculate_matrix_size_col_major(lda, n, lda) * sizeof(float);
    size_t size_x = calculate_vector_size(n, incx) * sizeof(float);
    size_t size_y = calculate_vector_size(n, incy) * sizeof(float);
    
    fb_gpu_ptr_t d_a, d_x, d_y;
    if (get_device_buffer_input(a, size_a, &d_a) != 0) return;
    if (get_device_buffer_input(x, size_x, &d_x) != 0) {
        release_device_buffer(a);
        return;
    }
    if (get_device_buffer_inout(y, size_y, &d_y) != 0) {
        release_device_buffer(a);
        release_device_buffer(x);
        return;
    }
    
    if (fb_cublas_trait.ssymv) {
        fb_cublas_trait.ssymv(g_cublas_handle, NULL, uplo, n, alpha,
                              d_a, lda, d_x, incx, beta, d_y, incy);
    }
    
    mark_output_dirty(y);
    release_device_buffer(a);
    release_device_buffer(x);
    release_device_buffer(y);
}

/**
 * @brief Symmetric matrix-vector multiply (double precision)
 */
void cublas_dsymv_smart_wrapper(char uplo, int n, double alpha,
                                 const double* a, int lda, const double* x, int incx,
                                 double beta, double* y, int incy) {
    if (n <= 0) return;
    
    size_t size_a = calculate_matrix_size_col_major(lda, n, lda) * sizeof(double);
    size_t size_x = calculate_vector_size(n, incx) * sizeof(double);
    size_t size_y = calculate_vector_size(n, incy) * sizeof(double);
    
    fb_gpu_ptr_t d_a, d_x, d_y;
    if (get_device_buffer_input(a, size_a, &d_a) != 0) return;
    if (get_device_buffer_input(x, size_x, &d_x) != 0) {
        release_device_buffer(a);
        return;
    }
    if (get_device_buffer_inout(y, size_y, &d_y) != 0) {
        release_device_buffer(a);
        release_device_buffer(x);
        return;
    }
    
    if (fb_cublas_trait.dsymv) {
        fb_cublas_trait.dsymv(g_cublas_handle, NULL, uplo, n, alpha,
                              d_a, lda, d_x, incx, beta, d_y, incy);
    }
    
    mark_output_dirty(y);
    release_device_buffer(a);
    release_device_buffer(x);
    release_device_buffer(y);
}

/* ============================================================================
 * BLAS Level 2: Triangular Matrix-Vector Operations (4 ops)
 * ========================================================================== */

/**
 * @brief Triangular matrix-vector multiply (single precision)
 * x := A*x  or  x := A^T*x, where A is n×n triangular
 */
void cublas_strmv_smart_wrapper(char uplo, FB_TRANSPOSE trans, char diag,
                                 int n, const float* a, int lda,
                                 float* x, int incx) {
    if (n <= 0) return;
    
    size_t size_a = calculate_matrix_size_col_major(lda, n, lda) * sizeof(float);
    size_t size_x = calculate_vector_size(n, incx) * sizeof(float);
    
    fb_gpu_ptr_t d_a, d_x;
    if (get_device_buffer_input(a, size_a, &d_a) != 0) return;
    if (get_device_buffer_inout(x, size_x, &d_x) != 0) {
        release_device_buffer(a);
        return;
    }
    
    char trans_char = (trans == FbNoTrans) ? 'N' : 
                      (trans == FbTrans) ? 'T' : 'C';
    
    if (fb_cublas_trait.strmv) {
        fb_cublas_trait.strmv(g_cublas_handle, NULL, uplo, trans_char, diag,
                              n, d_a, lda, d_x, incx);
    }
    
    mark_output_dirty(x);
    release_device_buffer(a);
    release_device_buffer(x);
}

/**
 * @brief Triangular matrix-vector multiply (double precision)
 */
void cublas_dtrmv_smart_wrapper(char uplo, FB_TRANSPOSE trans, char diag,
                                 int n, const double* a, int lda,
                                 double* x, int incx) {
    if (n <= 0) return;
    
    size_t size_a = calculate_matrix_size_col_major(lda, n, lda) * sizeof(double);
    size_t size_x = calculate_vector_size(n, incx) * sizeof(double);
    
    fb_gpu_ptr_t d_a, d_x;
    if (get_device_buffer_input(a, size_a, &d_a) != 0) return;
    if (get_device_buffer_inout(x, size_x, &d_x) != 0) {
        release_device_buffer(a);
        return;
    }
    
    char trans_char = (trans == FbNoTrans) ? 'N' : 
                      (trans == FbTrans) ? 'T' : 'C';
    
    if (fb_cublas_trait.dtrmv) {
        fb_cublas_trait.dtrmv(g_cublas_handle, NULL, uplo, trans_char, diag,
                              n, d_a, lda, d_x, incx);
    }
    
    mark_output_dirty(x);
    release_device_buffer(a);
    release_device_buffer(x);
}

/**
 * @brief Triangular solve (single precision)
 * Solves A*x = b  or  A^T*x = b, where A is n×n triangular
 */
void cublas_strsv_smart_wrapper(char uplo, FB_TRANSPOSE trans, char diag,
                                 int n, const float* a, int lda,
                                 float* x, int incx) {
    if (n <= 0) return;
    
    size_t size_a = calculate_matrix_size_col_major(lda, n, lda) * sizeof(float);
    size_t size_x = calculate_vector_size(n, incx) * sizeof(float);
    
    fb_gpu_ptr_t d_a, d_x;
    if (get_device_buffer_input(a, size_a, &d_a) != 0) return;
    if (get_device_buffer_inout(x, size_x, &d_x) != 0) {
        release_device_buffer(a);
        return;
    }
    
    char trans_char = (trans == FbNoTrans) ? 'N' : 
                      (trans == FbTrans) ? 'T' : 'C';
    
    if (fb_cublas_trait.strsv) {
        fb_cublas_trait.strsv(g_cublas_handle, NULL, uplo, trans_char, diag,
                              n, d_a, lda, d_x, incx);
    }
    
    mark_output_dirty(x);
    release_device_buffer(a);
    release_device_buffer(x);
}

/**
 * @brief Triangular solve (double precision)
 */
void cublas_dtrsv_smart_wrapper(char uplo, FB_TRANSPOSE trans, char diag,
                                 int n, const double* a, int lda,
                                 double* x, int incx) {
    if (n <= 0) return;
    
    size_t size_a = calculate_matrix_size_col_major(lda, n, lda) * sizeof(double);
    size_t size_x = calculate_vector_size(n, incx) * sizeof(double);
    
    fb_gpu_ptr_t d_a, d_x;
    if (get_device_buffer_input(a, size_a, &d_a) != 0) return;
    if (get_device_buffer_inout(x, size_x, &d_x) != 0) {
        release_device_buffer(a);
        return;
    }
    
    char trans_char = (trans == FbNoTrans) ? 'N' : 
                      (trans == FbTrans) ? 'T' : 'C';
    
    if (fb_cublas_trait.dtrsv) {
        fb_cublas_trait.dtrsv(g_cublas_handle, NULL, uplo, trans_char, diag,
                              n, d_a, lda, d_x, incx);
    }
    
    mark_output_dirty(x);
    release_device_buffer(a);
    release_device_buffer(x);
}

/* ============================================================================
 * BLAS Level 2: Symmetric Rank Updates (4 ops)
 * ========================================================================== */

/**
 * @brief Symmetric rank-1 update (single precision)
 * A := alpha*x*x^T + A, where A is n×n symmetric
 */
void cublas_ssyr_smart_wrapper(char uplo, int n, float alpha,
                                const float* x, int incx, float* a, int lda) {
    if (n <= 0) return;
    
    size_t size_x = calculate_vector_size(n, incx) * sizeof(float);
    size_t size_a = calculate_matrix_size_col_major(lda, n, lda) * sizeof(float);
    
    fb_gpu_ptr_t d_x, d_a;
    if (get_device_buffer_input(x, size_x, &d_x) != 0) return;
    if (get_device_buffer_inout(a, size_a, &d_a) != 0) {
        release_device_buffer(x);
        return;
    }
    
    if (fb_cublas_trait.ssyr) {
        fb_cublas_trait.ssyr(g_cublas_handle, NULL, uplo, n, alpha, d_x, incx, d_a, lda);
    }
    
    mark_output_dirty(a);
    release_device_buffer(x);
    release_device_buffer(a);
}

/**
 * @brief Symmetric rank-1 update (double precision)
 */
void cublas_dsyr_smart_wrapper(char uplo, int n, double alpha,
                                const double* x, int incx, double* a, int lda) {
    if (n <= 0) return;
    
    size_t size_x = calculate_vector_size(n, incx) * sizeof(double);
    size_t size_a = calculate_matrix_size_col_major(lda, n, lda) * sizeof(double);
    
    fb_gpu_ptr_t d_x, d_a;
    if (get_device_buffer_input(x, size_x, &d_x) != 0) return;
    if (get_device_buffer_inout(a, size_a, &d_a) != 0) {
        release_device_buffer(x);
        return;
    }
    
    if (fb_cublas_trait.dsyr) {
        fb_cublas_trait.dsyr(g_cublas_handle, NULL, uplo, n, alpha, d_x, incx, d_a, lda);
    }
    
    mark_output_dirty(a);
    release_device_buffer(x);
    release_device_buffer(a);
}

/**
 * @brief Symmetric rank-2 update (single precision)
 * A := alpha*x*y^T + alpha*y*x^T + A, where A is n×n symmetric
 */
void cublas_ssyr2_smart_wrapper(char uplo, int n, float alpha,
                                 const float* x, int incx, const float* y, int incy,
                                 float* a, int lda) {
    if (n <= 0) return;
    
    size_t size_x = calculate_vector_size(n, incx) * sizeof(float);
    size_t size_y = calculate_vector_size(n, incy) * sizeof(float);
    size_t size_a = calculate_matrix_size_col_major(lda, n, lda) * sizeof(float);
    
    fb_gpu_ptr_t d_x, d_y, d_a;
    if (get_device_buffer_input(x, size_x, &d_x) != 0) return;
    if (get_device_buffer_input(y, size_y, &d_y) != 0) {
        release_device_buffer(x);
        return;
    }
    if (get_device_buffer_inout(a, size_a, &d_a) != 0) {
        release_device_buffer(x);
        release_device_buffer(y);
        return;
    }
    
    if (fb_cublas_trait.ssyr2) {
        fb_cublas_trait.ssyr2(g_cublas_handle, NULL, uplo, n, alpha,
                              d_x, incx, d_y, incy, d_a, lda);
    }
    
    mark_output_dirty(a);
    release_device_buffer(x);
    release_device_buffer(y);
    release_device_buffer(a);
}

/**
 * @brief Symmetric rank-2 update (double precision)
 */
void cublas_dsyr2_smart_wrapper(char uplo, int n, double alpha,
                                 const double* x, int incx, const double* y, int incy,
                                 double* a, int lda) {
    if (n <= 0) return;
    
    size_t size_x = calculate_vector_size(n, incx) * sizeof(double);
    size_t size_y = calculate_vector_size(n, incy) * sizeof(double);
    size_t size_a = calculate_matrix_size_col_major(lda, n, lda) * sizeof(double);
    
    fb_gpu_ptr_t d_x, d_y, d_a;
    if (get_device_buffer_input(x, size_x, &d_x) != 0) return;
    if (get_device_buffer_input(y, size_y, &d_y) != 0) {
        release_device_buffer(x);
        return;
    }
    if (get_device_buffer_inout(a, size_a, &d_a) != 0) {
        release_device_buffer(x);
        release_device_buffer(y);
        return;
    }
    
    if (fb_cublas_trait.dsyr2) {
        fb_cublas_trait.dsyr2(g_cublas_handle, NULL, uplo, n, alpha,
                              d_x, incx, d_y, incy, d_a, lda);
    }
    
    mark_output_dirty(a);
    release_device_buffer(x);
    release_device_buffer(y);
    release_device_buffer(a);
}

/* ============================================================================
 * BLAS Level 2: Banded Operations (6 ops)
 * ========================================================================== */

/**
 * @brief Symmetric banded matrix-vector multiply (single precision)
 * y := alpha*A*x + beta*y, where A is n×n symmetric banded with k super-diagonals
 */
void cublas_ssbmv_smart_wrapper(char uplo, int n, int k, float alpha,
                                 const float* a, int lda, const float* x, int incx,
                                 float beta, float* y, int incy) {
    if (n <= 0) return;
    
    size_t size_a = calculate_banded_size(n, k + 1, lda) * sizeof(float);
    size_t size_x = calculate_vector_size(n, incx) * sizeof(float);
    size_t size_y = calculate_vector_size(n, incy) * sizeof(float);
    
    fb_gpu_ptr_t d_a, d_x, d_y;
    if (get_device_buffer_input(a, size_a, &d_a) != 0) return;
    if (get_device_buffer_input(x, size_x, &d_x) != 0) {
        release_device_buffer(a);
        return;
    }
    if (get_device_buffer_inout(y, size_y, &d_y) != 0) {
        release_device_buffer(a);
        release_device_buffer(x);
        return;
    }
    
    if (fb_cublas_trait.ssbmv) {
        fb_cublas_trait.ssbmv(g_cublas_handle, NULL, uplo, n, k, alpha,
                              d_a, lda, d_x, incx, beta, d_y, incy);
    }
    
    mark_output_dirty(y);
    release_device_buffer(a);
    release_device_buffer(x);
    release_device_buffer(y);
}

/**
 * @brief Symmetric banded matrix-vector multiply (double precision)
 */
void cublas_dsbmv_smart_wrapper(char uplo, int n, int k, double alpha,
                                 const double* a, int lda, const double* x, int incx,
                                 double beta, double* y, int incy) {
    if (n <= 0) return;
    
    size_t size_a = calculate_banded_size(n, k + 1, lda) * sizeof(double);
    size_t size_x = calculate_vector_size(n, incx) * sizeof(double);
    size_t size_y = calculate_vector_size(n, incy) * sizeof(double);
    
    fb_gpu_ptr_t d_a, d_x, d_y;
    if (get_device_buffer_input(a, size_a, &d_a) != 0) return;
    if (get_device_buffer_input(x, size_x, &d_x) != 0) {
        release_device_buffer(a);
        return;
    }
    if (get_device_buffer_inout(y, size_y, &d_y) != 0) {
        release_device_buffer(a);
        release_device_buffer(x);
        return;
    }
    
    if (fb_cublas_trait.dsbmv) {
        fb_cublas_trait.dsbmv(g_cublas_handle, NULL, uplo, n, k, alpha,
                              d_a, lda, d_x, incx, beta, d_y, incy);
    }
    
    mark_output_dirty(y);
    release_device_buffer(a);
    release_device_buffer(x);
    release_device_buffer(y);
}

/**
 * @brief Triangular banded matrix-vector multiply (single precision)
 */
void cublas_stbmv_smart_wrapper(char uplo, FB_TRANSPOSE trans, char diag,
                                 int n, int k, const float* a, int lda,
                                 float* x, int incx) {
    if (n <= 0) return;
    
    size_t size_a = calculate_banded_size(n, k + 1, lda) * sizeof(float);
    size_t size_x = calculate_vector_size(n, incx) * sizeof(float);
    
    fb_gpu_ptr_t d_a, d_x;
    if (get_device_buffer_input(a, size_a, &d_a) != 0) return;
    if (get_device_buffer_inout(x, size_x, &d_x) != 0) {
        release_device_buffer(a);
        return;
    }
    
    char trans_char = (trans == FbNoTrans) ? 'N' : 
                      (trans == FbTrans) ? 'T' : 'C';
    
    if (fb_cublas_trait.stbmv) {
        fb_cublas_trait.stbmv(g_cublas_handle, NULL, uplo, trans_char, diag,
                              n, k, d_a, lda, d_x, incx);
    }
    
    mark_output_dirty(x);
    release_device_buffer(a);
    release_device_buffer(x);
}

/**
 * @brief Triangular banded matrix-vector multiply (double precision)
 */
void cublas_dtbmv_smart_wrapper(char uplo, FB_TRANSPOSE trans, char diag,
                                 int n, int k, const double* a, int lda,
                                 double* x, int incx) {
    if (n <= 0) return;
    
    size_t size_a = calculate_banded_size(n, k + 1, lda) * sizeof(double);
    size_t size_x = calculate_vector_size(n, incx) * sizeof(double);
    
    fb_gpu_ptr_t d_a, d_x;
    if (get_device_buffer_input(a, size_a, &d_a) != 0) return;
    if (get_device_buffer_inout(x, size_x, &d_x) != 0) {
        release_device_buffer(a);
        return;
    }
    
    char trans_char = (trans == FbNoTrans) ? 'N' : 
                      (trans == FbTrans) ? 'T' : 'C';
    
    if (fb_cublas_trait.dtbmv) {
        fb_cublas_trait.dtbmv(g_cublas_handle, NULL, uplo, trans_char, diag,
                              n, k, d_a, lda, d_x, incx);
    }
    
    mark_output_dirty(x);
    release_device_buffer(a);
    release_device_buffer(x);
}

/**
 * @brief Triangular banded solve (single precision)
 */
void cublas_stbsv_smart_wrapper(char uplo, FB_TRANSPOSE trans, char diag,
                                 int n, int k, const float* a, int lda,
                                 float* x, int incx) {
    if (n <= 0) return;
    
    size_t size_a = calculate_banded_size(n, k + 1, lda) * sizeof(float);
    size_t size_x = calculate_vector_size(n, incx) * sizeof(float);
    
    fb_gpu_ptr_t d_a, d_x;
    if (get_device_buffer_input(a, size_a, &d_a) != 0) return;
    if (get_device_buffer_inout(x, size_x, &d_x) != 0) {
        release_device_buffer(a);
        return;
    }
    
    char trans_char = (trans == FbNoTrans) ? 'N' : 
                      (trans == FbTrans) ? 'T' : 'C';
    
    if (fb_cublas_trait.stbsv) {
        fb_cublas_trait.stbsv(g_cublas_handle, NULL, uplo, trans_char, diag,
                              n, k, d_a, lda, d_x, incx);
    }
    
    mark_output_dirty(x);
    release_device_buffer(a);
    release_device_buffer(x);
}

/**
 * @brief Triangular banded solve (double precision)
 */
void cublas_dtbsv_smart_wrapper(char uplo, FB_TRANSPOSE trans, char diag,
                                 int n, int k, const double* a, int lda,
                                 double* x, int incx) {
    if (n <= 0) return;
    
    size_t size_a = calculate_banded_size(n, k + 1, lda) * sizeof(double);
    size_t size_x = calculate_vector_size(n, incx) * sizeof(double);
    
    fb_gpu_ptr_t d_a, d_x;
    if (get_device_buffer_input(a, size_a, &d_a) != 0) return;
    if (get_device_buffer_inout(x, size_x, &d_x) != 0) {
        release_device_buffer(a);
        return;
    }
    
    char trans_char = (trans == FbNoTrans) ? 'N' : 
                      (trans == FbTrans) ? 'T' : 'C';
    
    if (fb_cublas_trait.dtbsv) {
        fb_cublas_trait.dtbsv(g_cublas_handle, NULL, uplo, trans_char, diag,
                              n, k, d_a, lda, d_x, incx);
    }
    
    mark_output_dirty(x);
    release_device_buffer(a);
    release_device_buffer(x);
}

/* ============================================================================
 * BLAS Level 2: Packed Storage Operations (10 ops)
 * ========================================================================== */

/**
 * @brief Symmetric packed matrix-vector multiply (single precision)
 * y := alpha*A*x + beta*y, where A is n×n symmetric in packed storage
 */
void cublas_sspmv_smart_wrapper(char uplo, int n, float alpha,
                                 const float* ap, const float* x, int incx,
                                 float beta, float* y, int incy) {
    if (n <= 0) return;
    
    size_t size_ap = calculate_packed_size(n) * sizeof(float);
    size_t size_x = calculate_vector_size(n, incx) * sizeof(float);
    size_t size_y = calculate_vector_size(n, incy) * sizeof(float);
    
    fb_gpu_ptr_t d_ap, d_x, d_y;
    if (get_device_buffer_input(ap, size_ap, &d_ap) != 0) return;
    if (get_device_buffer_input(x, size_x, &d_x) != 0) {
        release_device_buffer(ap);
        return;
    }
    if (get_device_buffer_inout(y, size_y, &d_y) != 0) {
        release_device_buffer(ap);
        release_device_buffer(x);
        return;
    }
    
    if (fb_cublas_trait.sspmv) {
        fb_cublas_trait.sspmv(g_cublas_handle, NULL, uplo, n, alpha,
                              d_ap, d_x, incx, beta, d_y, incy);
    }
    
    mark_output_dirty(y);
    release_device_buffer(ap);
    release_device_buffer(x);
    release_device_buffer(y);
}

/**
 * @brief Symmetric packed matrix-vector multiply (double precision)
 */
void cublas_dspmv_smart_wrapper(char uplo, int n, double alpha,
                                 const double* ap, const double* x, int incx,
                                 double beta, double* y, int incy) {
    if (n <= 0) return;
    
    size_t size_ap = calculate_packed_size(n) * sizeof(double);
    size_t size_x = calculate_vector_size(n, incx) * sizeof(double);
    size_t size_y = calculate_vector_size(n, incy) * sizeof(double);
    
    fb_gpu_ptr_t d_ap, d_x, d_y;
    if (get_device_buffer_input(ap, size_ap, &d_ap) != 0) return;
    if (get_device_buffer_input(x, size_x, &d_x) != 0) {
        release_device_buffer(ap);
        return;
    }
    if (get_device_buffer_inout(y, size_y, &d_y) != 0) {
        release_device_buffer(ap);
        release_device_buffer(x);
        return;
    }
    
    if (fb_cublas_trait.dspmv) {
        fb_cublas_trait.dspmv(g_cublas_handle, NULL, uplo, n, alpha,
                              d_ap, d_x, incx, beta, d_y, incy);
    }
    
    mark_output_dirty(y);
    release_device_buffer(ap);
    release_device_buffer(x);
    release_device_buffer(y);
}

/**
 * @brief Triangular packed matrix-vector multiply (single precision)
 */
void cublas_stpmv_smart_wrapper(char uplo, FB_TRANSPOSE trans, char diag,
                                 int n, const float* ap, float* x, int incx) {
    if (n <= 0) return;
    
    size_t size_ap = calculate_packed_size(n) * sizeof(float);
    size_t size_x = calculate_vector_size(n, incx) * sizeof(float);
    
    fb_gpu_ptr_t d_ap, d_x;
    if (get_device_buffer_input(ap, size_ap, &d_ap) != 0) return;
    if (get_device_buffer_inout(x, size_x, &d_x) != 0) {
        release_device_buffer(ap);
        return;
    }
    
    char trans_char = (trans == FbNoTrans) ? 'N' : 
                      (trans == FbTrans) ? 'T' : 'C';
    
    if (fb_cublas_trait.stpmv) {
        fb_cublas_trait.stpmv(g_cublas_handle, NULL, uplo, trans_char, diag,
                              n, d_ap, d_x, incx);
    }
    
    mark_output_dirty(x);
    release_device_buffer(ap);
    release_device_buffer(x);
}

/**
 * @brief Triangular packed matrix-vector multiply (double precision)
 */
void cublas_dtpmv_smart_wrapper(char uplo, FB_TRANSPOSE trans, char diag,
                                 int n, const double* ap, double* x, int incx) {
    if (n <= 0) return;
    
    size_t size_ap = calculate_packed_size(n) * sizeof(double);
    size_t size_x = calculate_vector_size(n, incx) * sizeof(double);
    
    fb_gpu_ptr_t d_ap, d_x;
    if (get_device_buffer_input(ap, size_ap, &d_ap) != 0) return;
    if (get_device_buffer_inout(x, size_x, &d_x) != 0) {
        release_device_buffer(ap);
        return;
    }
    
    char trans_char = (trans == FbNoTrans) ? 'N' : 
                      (trans == FbTrans) ? 'T' : 'C';
    
    if (fb_cublas_trait.dtpmv) {
        fb_cublas_trait.dtpmv(g_cublas_handle, NULL, uplo, trans_char, diag,
                              n, d_ap, d_x, incx);
    }
    
    mark_output_dirty(x);
    release_device_buffer(ap);
    release_device_buffer(x);
}

/**
 * @brief Triangular packed solve (single precision)
 */
void cublas_stpsv_smart_wrapper(char uplo, FB_TRANSPOSE trans, char diag,
                                 int n, const float* ap, float* x, int incx) {
    if (n <= 0) return;
    
    size_t size_ap = calculate_packed_size(n) * sizeof(float);
    size_t size_x = calculate_vector_size(n, incx) * sizeof(float);
    
    fb_gpu_ptr_t d_ap, d_x;
    if (get_device_buffer_input(ap, size_ap, &d_ap) != 0) return;
    if (get_device_buffer_inout(x, size_x, &d_x) != 0) {
        release_device_buffer(ap);
        return;
    }
    
    char trans_char = (trans == FbNoTrans) ? 'N' : 
                      (trans == FbTrans) ? 'T' : 'C';
    
    if (fb_cublas_trait.stpsv) {
        fb_cublas_trait.stpsv(g_cublas_handle, NULL, uplo, trans_char, diag,
                              n, d_ap, d_x, incx);
    }
    
    mark_output_dirty(x);
    release_device_buffer(ap);
    release_device_buffer(x);
}

/**
 * @brief Triangular packed solve (double precision)
 */
void cublas_dtpsv_smart_wrapper(char uplo, FB_TRANSPOSE trans, char diag,
                                 int n, const double* ap, double* x, int incx) {
    if (n <= 0) return;
    
    size_t size_ap = calculate_packed_size(n) * sizeof(double);
    size_t size_x = calculate_vector_size(n, incx) * sizeof(double);
    
    fb_gpu_ptr_t d_ap, d_x;
    if (get_device_buffer_input(ap, size_ap, &d_ap) != 0) return;
    if (get_device_buffer_inout(x, size_x, &d_x) != 0) {
        release_device_buffer(ap);
        return;
    }
    
    char trans_char = (trans == FbNoTrans) ? 'N' : 
                      (trans == FbTrans) ? 'T' : 'C';
    
    if (fb_cublas_trait.dtpsv) {
        fb_cublas_trait.dtpsv(g_cublas_handle, NULL, uplo, trans_char, diag,
                              n, d_ap, d_x, incx);
    }
    
    mark_output_dirty(x);
    release_device_buffer(ap);
    release_device_buffer(x);
}

/**
 * @brief Symmetric packed rank-1 update (single precision)
 * A := alpha*x*x^T + A, where A is n×n symmetric in packed storage
 */
void cublas_sspr_smart_wrapper(char uplo, int n, float alpha,
                                const float* x, int incx, float* ap) {
    if (n <= 0) return;
    
    size_t size_x = calculate_vector_size(n, incx) * sizeof(float);
    size_t size_ap = calculate_packed_size(n) * sizeof(float);
    
    fb_gpu_ptr_t d_x, d_ap;
    if (get_device_buffer_input(x, size_x, &d_x) != 0) return;
    if (get_device_buffer_inout(ap, size_ap, &d_ap) != 0) {
        release_device_buffer(x);
        return;
    }
    
    if (fb_cublas_trait.sspr) {
        fb_cublas_trait.sspr(g_cublas_handle, NULL, uplo, n, alpha, d_x, incx, d_ap);
    }
    
    mark_output_dirty(ap);
    release_device_buffer(x);
    release_device_buffer(ap);
}

/**
 * @brief Symmetric packed rank-1 update (double precision)
 */
void cublas_dspr_smart_wrapper(char uplo, int n, double alpha,
                                const double* x, int incx, double* ap) {
    if (n <= 0) return;
    
    size_t size_x = calculate_vector_size(n, incx) * sizeof(double);
    size_t size_ap = calculate_packed_size(n) * sizeof(double);
    
    fb_gpu_ptr_t d_x, d_ap;
    if (get_device_buffer_input(x, size_x, &d_x) != 0) return;
    if (get_device_buffer_inout(ap, size_ap, &d_ap) != 0) {
        release_device_buffer(x);
        return;
    }
    
    if (fb_cublas_trait.dspr) {
        fb_cublas_trait.dspr(g_cublas_handle, NULL, uplo, n, alpha, d_x, incx, d_ap);
    }
    
    mark_output_dirty(ap);
    release_device_buffer(x);
    release_device_buffer(ap);
}

/**
 * @brief Symmetric packed rank-2 update (single precision)
 * A := alpha*x*y^T + alpha*y*x^T + A, where A is n×n symmetric in packed storage
 */
void cublas_sspr2_smart_wrapper(char uplo, int n, float alpha,
                                 const float* x, int incx, const float* y, int incy,
                                 float* ap) {
    if (n <= 0) return;
    
    size_t size_x = calculate_vector_size(n, incx) * sizeof(float);
    size_t size_y = calculate_vector_size(n, incy) * sizeof(float);
    size_t size_ap = calculate_packed_size(n) * sizeof(float);
    
    fb_gpu_ptr_t d_x, d_y, d_ap;
    if (get_device_buffer_input(x, size_x, &d_x) != 0) return;
    if (get_device_buffer_input(y, size_y, &d_y) != 0) {
        release_device_buffer(x);
        return;
    }
    if (get_device_buffer_inout(ap, size_ap, &d_ap) != 0) {
        release_device_buffer(x);
        release_device_buffer(y);
        return;
    }
    
    if (fb_cublas_trait.sspr2) {
        fb_cublas_trait.sspr2(g_cublas_handle, NULL, uplo, n, alpha,
                              d_x, incx, d_y, incy, d_ap);
    }
    
    mark_output_dirty(ap);
    release_device_buffer(x);
    release_device_buffer(y);
    release_device_buffer(ap);
}

/**
 * @brief Symmetric packed rank-2 update (double precision)
 */
void cublas_dspr2_smart_wrapper(char uplo, int n, double alpha,
                                 const double* x, int incx, const double* y, int incy,
                                 double* ap) {
    if (n <= 0) return;
    
    size_t size_x = calculate_vector_size(n, incx) * sizeof(double);
    size_t size_y = calculate_vector_size(n, incy) * sizeof(double);
    size_t size_ap = calculate_packed_size(n) * sizeof(double);
    
    fb_gpu_ptr_t d_x, d_y, d_ap;
    if (get_device_buffer_input(x, size_x, &d_x) != 0) return;
    if (get_device_buffer_input(y, size_y, &d_y) != 0) {
        release_device_buffer(x);
        return;
    }
    if (get_device_buffer_inout(ap, size_ap, &d_ap) != 0) {
        release_device_buffer(x);
        release_device_buffer(y);
        return;
    }
    
    if (fb_cublas_trait.dspr2) {
        fb_cublas_trait.dspr2(g_cublas_handle, NULL, uplo, n, alpha,
                              d_x, incx, d_y, incy, d_ap);
    }
    
    mark_output_dirty(ap);
    release_device_buffer(x);
    release_device_buffer(y);
    release_device_buffer(ap);
}
