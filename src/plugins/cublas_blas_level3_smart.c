/**
 * @file cublas_blas_level3_smart.c
 * @brief Smart wrappers for BLAS Level 3 operations (Phase 1 - Real-valued)
 * 
 * This file provides smart wrappers for Level 3 matrix-matrix operations:
 * - symm: Symmetric matrix-matrix multiply
 * - trmm: Triangular matrix-matrix multiply
 * - trsm: Triangular solve with multiple RHS
 * - syrk: Symmetric rank-k update
 * - syr2k: Symmetric rank-2k update
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

/* External cuBLAS trait and handle */
extern const fb_gpu_backend_trait_t fb_cublas_trait;
extern void* g_cublas_handle;
static const int CUBLAS_DEVICE_ID = 0;

/* Helper function prototypes */
static fb_device_memory_manager_t* get_device_manager(void);
static size_t calculate_matrix_size_col_major(int rows, int cols, int ld);
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

static size_t calculate_matrix_size_col_major(int rows, int cols, int ld) {
    return (size_t)ld * cols;
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
 * BLAS Level 3: Symmetric Matrix-Matrix Multiplication (2 ops)
 * ========================================================================== */

/**
 * @brief Symmetric matrix-matrix multiply (single precision)
 * C := alpha*A*B + beta*C  or  C := alpha*B*A + beta*C
 * where A is m×m or n×n symmetric
 */
void cublas_ssymm_smart_wrapper(char side, char uplo, int m, int n,
                                 float alpha, const float* a, int lda,
                                 const float* b, int ldb, float beta,
                                 float* c, int ldc) {
    if (m <= 0 || n <= 0) return;
    
    /* Matrix dimensions based on side */
    int ka = (side == 'L' || side == 'l') ? m : n;
    
    size_t size_a = calculate_matrix_size_col_major(lda, ka, lda) * sizeof(float);
    size_t size_b = calculate_matrix_size_col_major(ldb, n, ldb) * sizeof(float);
    size_t size_c = calculate_matrix_size_col_major(ldc, n, ldc) * sizeof(float);
    
    fb_gpu_ptr_t d_a, d_b, d_c;
    if (get_device_buffer_input(a, size_a, &d_a) != 0) return;
    if (get_device_buffer_input(b, size_b, &d_b) != 0) {
        release_device_buffer(a);
        return;
    }
    if (get_device_buffer_inout(c, size_c, &d_c) != 0) {
        release_device_buffer(a);
        release_device_buffer(b);
        return;
    }
    
    if (fb_cublas_trait.ssymm) {
        fb_cublas_trait.ssymm(g_cublas_handle, NULL, side, uplo, m, n,
                              alpha, d_a, lda, d_b, ldb, beta, d_c, ldc);
    }
    
    mark_output_dirty(c);
    release_device_buffer(a);
    release_device_buffer(b);
    release_device_buffer(c);
}

/**
 * @brief Symmetric matrix-matrix multiply (double precision)
 */
void cublas_dsymm_smart_wrapper(char side, char uplo, int m, int n,
                                 double alpha, const double* a, int lda,
                                 const double* b, int ldb, double beta,
                                 double* c, int ldc) {
    if (m <= 0 || n <= 0) return;
    
    int ka = (side == 'L' || side == 'l') ? m : n;
    
    size_t size_a = calculate_matrix_size_col_major(lda, ka, lda) * sizeof(double);
    size_t size_b = calculate_matrix_size_col_major(ldb, n, ldb) * sizeof(double);
    size_t size_c = calculate_matrix_size_col_major(ldc, n, ldc) * sizeof(double);
    
    fb_gpu_ptr_t d_a, d_b, d_c;
    if (get_device_buffer_input(a, size_a, &d_a) != 0) return;
    if (get_device_buffer_input(b, size_b, &d_b) != 0) {
        release_device_buffer(a);
        return;
    }
    if (get_device_buffer_inout(c, size_c, &d_c) != 0) {
        release_device_buffer(a);
        release_device_buffer(b);
        return;
    }
    
    if (fb_cublas_trait.dsymm) {
        fb_cublas_trait.dsymm(g_cublas_handle, NULL, side, uplo, m, n,
                              alpha, d_a, lda, d_b, ldb, beta, d_c, ldc);
    }
    
    mark_output_dirty(c);
    release_device_buffer(a);
    release_device_buffer(b);
    release_device_buffer(c);
}

/* ============================================================================
 * BLAS Level 3: Triangular Matrix-Matrix Multiplication (2 ops)
 * ========================================================================== */

/**
 * @brief Triangular matrix-matrix multiply (single precision)
 * B := alpha*op(A)*B  or  B := alpha*B*op(A)
 * where A is triangular and op(A) = A, A^T, or A^H
 */
void cublas_strmm_smart_wrapper(char side, char uplo, FB_TRANSPOSE transa, char diag,
                                 int m, int n, float alpha,
                                 const float* a, int lda, float* b, int ldb) {
    if (m <= 0 || n <= 0) return;
    
    int ka = (side == 'L' || side == 'l') ? m : n;
    
    size_t size_a = calculate_matrix_size_col_major(lda, ka, lda) * sizeof(float);
    size_t size_b = calculate_matrix_size_col_major(ldb, n, ldb) * sizeof(float);
    
    fb_gpu_ptr_t d_a, d_b;
    if (get_device_buffer_input(a, size_a, &d_a) != 0) return;
    if (get_device_buffer_inout(b, size_b, &d_b) != 0) {
        release_device_buffer(a);
        return;
    }
    
    char trans_char = (transa == FbNoTrans) ? 'N' : 
                      (transa == FbTrans) ? 'T' : 'C';
    
    if (fb_cublas_trait.strmm) {
        fb_cublas_trait.strmm(g_cublas_handle, NULL, side, uplo, trans_char, diag,
                              m, n, alpha, d_a, lda, d_b, ldb);
    }
    
    mark_output_dirty(b);
    release_device_buffer(a);
    release_device_buffer(b);
}

/**
 * @brief Triangular matrix-matrix multiply (double precision)
 */
void cublas_dtrmm_smart_wrapper(char side, char uplo, FB_TRANSPOSE transa, char diag,
                                 int m, int n, double alpha,
                                 const double* a, int lda, double* b, int ldb) {
    if (m <= 0 || n <= 0) return;
    
    int ka = (side == 'L' || side == 'l') ? m : n;
    
    size_t size_a = calculate_matrix_size_col_major(lda, ka, lda) * sizeof(double);
    size_t size_b = calculate_matrix_size_col_major(ldb, n, ldb) * sizeof(double);
    
    fb_gpu_ptr_t d_a, d_b;
    if (get_device_buffer_input(a, size_a, &d_a) != 0) return;
    if (get_device_buffer_inout(b, size_b, &d_b) != 0) {
        release_device_buffer(a);
        return;
    }
    
    char trans_char = (transa == FbNoTrans) ? 'N' : 
                      (transa == FbTrans) ? 'T' : 'C';
    
    if (fb_cublas_trait.dtrmm) {
        fb_cublas_trait.dtrmm(g_cublas_handle, NULL, side, uplo, trans_char, diag,
                              m, n, alpha, d_a, lda, d_b, ldb);
    }
    
    mark_output_dirty(b);
    release_device_buffer(a);
    release_device_buffer(b);
}

/* ============================================================================
 * BLAS Level 3: Triangular Solve (2 ops)
 * ========================================================================== */

/**
 * @brief Triangular solve with multiple RHS (single precision)
 * Solves op(A)*X = alpha*B  or  X*op(A) = alpha*B
 * where A is triangular
 */
void cublas_strsm_smart_wrapper(char side, char uplo, FB_TRANSPOSE transa, char diag,
                                 int m, int n, float alpha,
                                 const float* a, int lda, float* b, int ldb) {
    if (m <= 0 || n <= 0) return;
    
    int ka = (side == 'L' || side == 'l') ? m : n;
    
    size_t size_a = calculate_matrix_size_col_major(lda, ka, lda) * sizeof(float);
    size_t size_b = calculate_matrix_size_col_major(ldb, n, ldb) * sizeof(float);
    
    fb_gpu_ptr_t d_a, d_b;
    if (get_device_buffer_input(a, size_a, &d_a) != 0) return;
    if (get_device_buffer_inout(b, size_b, &d_b) != 0) {
        release_device_buffer(a);
        return;
    }
    
    char trans_char = (transa == FbNoTrans) ? 'N' : 
                      (transa == FbTrans) ? 'T' : 'C';
    
    if (fb_cublas_trait.strsm) {
        fb_cublas_trait.strsm(g_cublas_handle, NULL, side, uplo, trans_char, diag,
                              m, n, alpha, d_a, lda, d_b, ldb);
    }
    
    mark_output_dirty(b);
    release_device_buffer(a);
    release_device_buffer(b);
}

/**
 * @brief Triangular solve with multiple RHS (double precision)
 */
void cublas_dtrsm_smart_wrapper(char side, char uplo, FB_TRANSPOSE transa, char diag,
                                 int m, int n, double alpha,
                                 const double* a, int lda, double* b, int ldb) {
    if (m <= 0 || n <= 0) return;
    
    int ka = (side == 'L' || side == 'l') ? m : n;
    
    size_t size_a = calculate_matrix_size_col_major(lda, ka, lda) * sizeof(double);
    size_t size_b = calculate_matrix_size_col_major(ldb, n, ldb) * sizeof(double);
    
    fb_gpu_ptr_t d_a, d_b;
    if (get_device_buffer_input(a, size_a, &d_a) != 0) return;
    if (get_device_buffer_inout(b, size_b, &d_b) != 0) {
        release_device_buffer(a);
        return;
    }
    
    char trans_char = (transa == FbNoTrans) ? 'N' : 
                      (transa == FbTrans) ? 'T' : 'C';
    
    if (fb_cublas_trait.dtrsm) {
        fb_cublas_trait.dtrsm(g_cublas_handle, NULL, side, uplo, trans_char, diag,
                              m, n, alpha, d_a, lda, d_b, ldb);
    }
    
    mark_output_dirty(b);
    release_device_buffer(a);
    release_device_buffer(b);
}

/* ============================================================================
 * BLAS Level 3: Symmetric Rank-k Update (2 ops)
 * ========================================================================== */

/**
 * @brief Symmetric rank-k update (single precision)
 * C := alpha*A*A^T + beta*C  or  C := alpha*A^T*A + beta*C
 * where C is n×n symmetric
 */
void cublas_ssyrk_smart_wrapper(char uplo, FB_TRANSPOSE trans,
                                 int n, int k, float alpha,
                                 const float* a, int lda, float beta,
                                 float* c, int ldc) {
    if (n <= 0 || k < 0) return;
    
    /* Matrix A dimensions based on transpose */
    int a_rows = (trans == FbNoTrans) ? n : k;
    int a_cols = (trans == FbNoTrans) ? k : n;
    
    size_t size_a = calculate_matrix_size_col_major(lda, a_cols, lda) * sizeof(float);
    size_t size_c = calculate_matrix_size_col_major(ldc, n, ldc) * sizeof(float);
    
    fb_gpu_ptr_t d_a, d_c;
    if (get_device_buffer_input(a, size_a, &d_a) != 0) return;
    if (get_device_buffer_inout(c, size_c, &d_c) != 0) {
        release_device_buffer(a);
        return;
    }
    
    char trans_char = (trans == FbNoTrans) ? 'N' : 
                      (trans == FbTrans) ? 'T' : 'C';
    
    if (fb_cublas_trait.ssyrk) {
        fb_cublas_trait.ssyrk(g_cublas_handle, NULL, uplo, trans_char,
                              n, k, alpha, d_a, lda, beta, d_c, ldc);
    }
    
    mark_output_dirty(c);
    release_device_buffer(a);
    release_device_buffer(c);
}

/**
 * @brief Symmetric rank-k update (double precision)
 */
void cublas_dsyrk_smart_wrapper(char uplo, FB_TRANSPOSE trans,
                                 int n, int k, double alpha,
                                 const double* a, int lda, double beta,
                                 double* c, int ldc) {
    if (n <= 0 || k < 0) return;
    
    int a_rows = (trans == FbNoTrans) ? n : k;
    int a_cols = (trans == FbNoTrans) ? k : n;
    
    size_t size_a = calculate_matrix_size_col_major(lda, a_cols, lda) * sizeof(double);
    size_t size_c = calculate_matrix_size_col_major(ldc, n, ldc) * sizeof(double);
    
    fb_gpu_ptr_t d_a, d_c;
    if (get_device_buffer_input(a, size_a, &d_a) != 0) return;
    if (get_device_buffer_inout(c, size_c, &d_c) != 0) {
        release_device_buffer(a);
        return;
    }
    
    char trans_char = (trans == FbNoTrans) ? 'N' : 
                      (trans == FbTrans) ? 'T' : 'C';
    
    if (fb_cublas_trait.dsyrk) {
        fb_cublas_trait.dsyrk(g_cublas_handle, NULL, uplo, trans_char,
                              n, k, alpha, d_a, lda, beta, d_c, ldc);
    }
    
    mark_output_dirty(c);
    release_device_buffer(a);
    release_device_buffer(c);
}

/* ============================================================================
 * BLAS Level 3: Symmetric Rank-2k Update (2 ops)
 * ========================================================================== */

/**
 * @brief Symmetric rank-2k update (single precision)
 * C := alpha*A*B^T + alpha*B*A^T + beta*C  or
 * C := alpha*A^T*B + alpha*B^T*A + beta*C
 * where C is n×n symmetric
 */
void cublas_ssyr2k_smart_wrapper(char uplo, FB_TRANSPOSE trans,
                                  int n, int k, float alpha,
                                  const float* a, int lda, const float* b, int ldb,
                                  float beta, float* c, int ldc) {
    if (n <= 0 || k < 0) return;
    
    /* Matrix dimensions based on transpose */
    int a_cols = (trans == FbNoTrans) ? k : n;
    int b_cols = (trans == FbNoTrans) ? k : n;
    
    size_t size_a = calculate_matrix_size_col_major(lda, a_cols, lda) * sizeof(float);
    size_t size_b = calculate_matrix_size_col_major(ldb, b_cols, ldb) * sizeof(float);
    size_t size_c = calculate_matrix_size_col_major(ldc, n, ldc) * sizeof(float);
    
    fb_gpu_ptr_t d_a, d_b, d_c;
    if (get_device_buffer_input(a, size_a, &d_a) != 0) return;
    if (get_device_buffer_input(b, size_b, &d_b) != 0) {
        release_device_buffer(a);
        return;
    }
    if (get_device_buffer_inout(c, size_c, &d_c) != 0) {
        release_device_buffer(a);
        release_device_buffer(b);
        return;
    }
    
    char trans_char = (trans == FbNoTrans) ? 'N' : 
                      (trans == FbTrans) ? 'T' : 'C';
    
    if (fb_cublas_trait.ssyr2k) {
        fb_cublas_trait.ssyr2k(g_cublas_handle, NULL, uplo, trans_char,
                               n, k, alpha, d_a, lda, d_b, ldb, beta, d_c, ldc);
    }
    
    mark_output_dirty(c);
    release_device_buffer(a);
    release_device_buffer(b);
    release_device_buffer(c);
}

/**
 * @brief Symmetric rank-2k update (double precision)
 */
void cublas_dsyr2k_smart_wrapper(char uplo, FB_TRANSPOSE trans,
                                  int n, int k, double alpha,
                                  const double* a, int lda, const double* b, int ldb,
                                  double beta, double* c, int ldc) {
    if (n <= 0 || k < 0) return;
    
    int a_cols = (trans == FbNoTrans) ? k : n;
    int b_cols = (trans == FbNoTrans) ? k : n;
    
    size_t size_a = calculate_matrix_size_col_major(lda, a_cols, lda) * sizeof(double);
    size_t size_b = calculate_matrix_size_col_major(ldb, b_cols, ldb) * sizeof(double);
    size_t size_c = calculate_matrix_size_col_major(ldc, n, ldc) * sizeof(double);
    
    fb_gpu_ptr_t d_a, d_b, d_c;
    if (get_device_buffer_input(a, size_a, &d_a) != 0) return;
    if (get_device_buffer_input(b, size_b, &d_b) != 0) {
        release_device_buffer(a);
        return;
    }
    if (get_device_buffer_inout(c, size_c, &d_c) != 0) {
        release_device_buffer(a);
        release_device_buffer(b);
        return;
    }
    
    char trans_char = (trans == FbNoTrans) ? 'N' : 
                      (trans == FbTrans) ? 'T' : 'C';
    
    if (fb_cublas_trait.dsyr2k) {
        fb_cublas_trait.dsyr2k(g_cublas_handle, NULL, uplo, trans_char,
                               n, k, alpha, d_a, lda, d_b, ldb, beta, d_c, ldc);
    }
    
    mark_output_dirty(c);
    release_device_buffer(a);
    release_device_buffer(b);
    release_device_buffer(c);
}
