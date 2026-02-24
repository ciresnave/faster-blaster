/**
 * @file backend_interface.h
 * @brief Minimal CBLAS-style interface for compilation
 * 
 * This is a simplified version to get the project compiling.
 * Full 160+ operations will be added incrementally.
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#ifndef FASTER_BLASTER_BACKEND_INTERFACE_H
#define FASTER_BLASTER_BACKEND_INTERFACE_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Type Definitions
 * ========================================================================= */

typedef enum {
    FB_LAYOUT_ROW_MAJOR = 101,
    FB_LAYOUT_COL_MAJOR = 102
} fb_layout_t;

typedef enum {
    FB_NO_TRANS = 111,
    FB_TRANS = 112,
    FB_CONJ_TRANS = 113
} fb_transpose_t;

typedef enum {
    FB_UPPER = 121,
    FB_LOWER = 122
} fb_uplo_t;

typedef enum {
    FB_NON_UNIT = 131,
    FB_UNIT = 132
} fb_diag_t;

typedef enum {
    FB_LEFT = 141,
    FB_RIGHT = 142
} fb_side_t;

typedef enum {
    FB_CAP_LEVEL1       = 1 << 0,
    FB_CAP_LEVEL2       = 1 << 1,
    FB_CAP_LEVEL3       = 1 << 2,
    FB_CAP_BATCHED      = 1 << 3,
    FB_CAP_STRIDED      = 1 << 4,
    FB_CAP_MIXED_PREC   = 1 << 5,
    FB_CAP_GPU          = 1 << 6,
    FB_CAP_CPU          = 1 << 7,
    FB_CAP_SPARSE       = 1 << 8,
    FB_CAP_EXTENDED     = 1 << 9,
    FB_CAP_ASYNC        = 1 << 10
} fb_capability_t;

typedef enum {
    FB_HW_UNKNOWN = 0,
    FB_HW_CPU_INTEL,
    FB_HW_CPU_AMD,
    FB_HW_CPU_ARM,
    FB_HW_CPU_APPLE_SILICON,
    FB_HW_GPU_NVIDIA,
    FB_HW_GPU_AMD,
    FB_HW_GPU_INTEL,
    FB_HW_GPU_APPLE
} fb_hardware_type_t;

typedef struct {
    const char *name;
    const char *version;
    const char *vendor;
    uint32_t capabilities;
    fb_hardware_type_t hw_type;
    bool thread_safe;
    size_t min_efficient_size;
    size_t max_matrix_size;
} fb_backend_info_t;

/* ============================================================================
 * Function Pointer Types - Level 1 (Float/Double only for now)
 * ========================================================================= */

/* AXPY */
typedef void (*fb_saxpy_fn)(const int64_t n, const float alpha, const float *x, const int64_t incx, float *y, const int64_t incy);
typedef void (*fb_daxpy_fn)(const int64_t n, const double alpha, const double *x, const int64_t incx, double *y, const int64_t incy);

/* COPY */
typedef void (*fb_scopy_fn)(const int64_t n, const float *x, const int64_t incx, float *y, const int64_t incy);
typedef void (*fb_dcopy_fn)(const int64_t n, const double *x, const int64_t incx, double *y, const int64_t incy);

/* DOT */
typedef float (*fb_sdot_fn)(const int64_t n, const float *x, const int64_t incx, const float *y, const int64_t incy);
typedef double (*fb_ddot_fn)(const int64_t n, const double *x, const int64_t incx, const double *y, const int64_t incy);

/* NRM2 */
typedef float (*fb_snrm2_fn)(const int64_t n, const float *x, const int64_t incx);
typedef double (*fb_dnrm2_fn)(const int64_t n, const double *x, const int64_t incx);

/* SCAL */
typedef void (*fb_sscal_fn)(const int64_t n, const float alpha, float *x, const int64_t incx);
typedef void (*fb_dscal_fn)(const int64_t n, const double alpha, double *x, const int64_t incx);

/* ============================================================================
 * Function Pointer Types - Level 2
 * ========================================================================= */

/* GEMV */
typedef void (*fb_sgemv_fn)(const fb_layout_t layout, const fb_transpose_t trans, const int64_t m, const int64_t n, const float alpha, const float *A, const int64_t lda, const float *x, const int64_t incx, const float beta, float *y, const int64_t incy);
typedef void (*fb_dgemv_fn)(const fb_layout_t layout, const fb_transpose_t trans, const int64_t m, const int64_t n, const double alpha, const double *A, const int64_t lda, const double *x, const int64_t incx, const double beta, double *y, const int64_t incy);

/* ============================================================================
 * Function Pointer Types - Level 3
 * ========================================================================= */

/* GEMM */
typedef void (*fb_sgemm_fn)(const fb_layout_t layout, const fb_transpose_t transA, const fb_transpose_t transB, const int64_t m, const int64_t n, const int64_t k, const float alpha, const float *A, const int64_t lda, const float *B, const int64_t ldb, const float beta, float *C, const int64_t ldc);
typedef void (*fb_dgemm_fn)(const fb_layout_t layout, const fb_transpose_t transA, const fb_transpose_t transB, const int64_t m, const int64_t n, const int64_t k, const double alpha, const double *A, const int64_t lda, const double *B, const int64_t ldb, const double beta, double *C, const int64_t ldc);

/* ============================================================================
 * Backend Virtual Table
 * ========================================================================= */

typedef struct {
    /* Backend metadata */
    fb_backend_info_t info;
    
    /* Level 1 */
    fb_saxpy_fn saxpy;
    fb_daxpy_fn daxpy;
    fb_scopy_fn scopy;
    fb_dcopy_fn dcopy;
    fb_sdot_fn sdot;
    fb_ddot_fn ddot;
    fb_snrm2_fn snrm2;
    fb_dnrm2_fn dnrm2;
    fb_sscal_fn sscal;
    fb_dscal_fn dscal;
    
    /* Level 2 */
    fb_sgemv_fn sgemv;
    fb_dgemv_fn dgemv;
    
    /* Level 3 */
    fb_sgemm_fn sgemm;
    fb_dgemm_fn dgemm;
    
} fb_backend_vtable_t;

#ifdef __cplusplus
}
#endif

#endif /* FASTER_BLASTER_BACKEND_INTERFACE_H */
