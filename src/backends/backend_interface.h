/**
 * @file backend_interface.h
 * @brief CBLAS-style type-specific interface for BLAS backend implementations
 *
 * Complete interface for BLAS/LAPACK backends with 212 operations:
 * - BLAS Level 1: 52 functions (vector-vector)
 * - BLAS Level 2: 70 functions (matrix-vector)
 * - BLAS Level 3: 30 functions (matrix-matrix)
 * - LAPACK subset: 60 functions (linear algebra)
 *
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#ifndef FASTER_BLASTER_BACKEND_INTERFACE_H
#define FASTER_BLASTER_BACKEND_INTERFACE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Complex scalar types and status codes — canonical definitions */
#include <faster-blaster/fb_types.h>

/* Full operation-ID namespace (FB_OP_* constants, FB_JUDGE_MAX_OPERATIONS) */
/* NOTE: Do NOT include judge_op_ids.h here — it defines FB_OP_* as preprocessor
 * macros that conflict with the FB_OP_* enum in src/core/dispatch.h.
 * We only need FB_JUDGE_MAX_OPERATIONS for the ext_ops[] array size. */
#include <faster-blaster/judge.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Enum Definitions
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

typedef enum { FB_UPPER = 121, FB_LOWER = 122 } fb_uplo_t;

typedef enum { FB_NON_UNIT = 131, FB_UNIT = 132 } fb_diag_t;

typedef enum { FB_LEFT = 141, FB_RIGHT = 142 } fb_side_t;

typedef enum {
  FB_CAP_LEVEL1 = 1 << 0,
  FB_CAP_LEVEL2 = 1 << 1,
  FB_CAP_LEVEL3 = 1 << 2,
  FB_CAP_LAPACK = 1 << 3,
  FB_CAP_BATCHED = 1 << 4,
  FB_CAP_STRIDED = 1 << 5,
  FB_CAP_MIXED_PREC = 1 << 6,
  FB_CAP_GPU = 1 << 7,
  FB_CAP_CPU = 1 << 8,
  FB_CAP_SPARSE = 1 << 9,
  FB_CAP_EXTENDED = 1 << 10,
  FB_CAP_ASYNC = 1 << 11,
  FB_CAP_DOUBLE = 1 << 12,
  FB_CAP_SINGLE = 1 << 13,
  FB_CAP_COMPLEX = 1 << 14,
  FB_CAP_THREADSAFE = 1 << 15
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

/* ============================================================================
 * Unified Operation Types (for advanced features)
 * ========================================================================= */

/* fb_status_t and fb_precision_t are defined in fb_types.h (included above) */

typedef enum {
  FB_BATCH_SINGLE, /* Single operation (no batching) */
  FB_BATCH_ARRAY,  /* Array of pointers (A[], B[], C[]) */
  FB_BATCH_STRIDED /* Strided (A + i*strideA) */
} fb_batch_mode_t;

typedef enum {
  FB_FUSION_NONE,      /* No fusion */
  FB_FUSION_BIAS,      /* C = op(A, B) + bias */
  FB_FUSION_RELU,      /* C = relu(op(A, B)) */
  FB_FUSION_GELU,      /* C = gelu(op(A, B)) */
  FB_FUSION_BIAS_RELU, /* C = relu(op(A, B) + bias) */
  FB_FUSION_BIAS_GELU  /* C = gelu(op(A, B) + bias) */
} fb_fusion_t;

typedef enum {
  FB_ACT_NONE,
  FB_ACT_RELU,
  FB_ACT_GELU,
  FB_ACT_SIGMOID,
  FB_ACT_TANH
} fb_activation_t;

typedef enum {
  FB_NORM_Z_SCORE,      /* Standard z-score normalization */
  FB_NORM_BATCH,        /* Batch normalization (DNN) */
  FB_NORM_LAYER,        /* Layer normalization (Transformers) */
  FB_NORM_INSTANCE,     /* Instance normalization */
  FB_NORM_GROUP,        /* Group normalization */
  FB_NORM_L1,           /* L1 normalization */
  FB_NORM_L2,           /* L2 normalization */
  FB_NORM_MAX,          /* Max normalization */
  FB_NORM_MIN_MAX_SCALE /* Min-max scaling */
} fb_norm_mode_t;

typedef enum { FB_NORM_TRAINING, FB_NORM_INFERENCE } fb_norm_phase_t;

typedef enum {
  FB_REDUCE_SUM,
  FB_REDUCE_MIN,
  FB_REDUCE_MAX,
  FB_REDUCE_PRODUCT,
  FB_REDUCE_MEAN,
  FB_REDUCE_VARIANCE,
  FB_REDUCE_STD_DEV,
  FB_REDUCE_NORM_L1,
  FB_REDUCE_NORM_L2,
  FB_REDUCE_NORM_MAX
} fb_reduce_op_t;

typedef enum {
  FB_REDUCE_LOCAL,     /* Single device reduction */
  FB_REDUCE_COLLECTIVE /* Multi-device reduction (NCCL/RCCL) */
} fb_reduce_scope_t;

typedef enum {
  FB_SCAN_NONE,      /* Pure reduction (no scan) */
  FB_SCAN_INCLUSIVE, /* Inclusive scan (prefix sum) */
  FB_SCAN_EXCLUSIVE  /* Exclusive scan */
} fb_scan_mode_t;

/* Opaque communicator handle for collective operations */
typedef void *fb_comm_t;

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
 * BLAS Level 1: Vector-Vector Operations (52 functions)
 * ========================================================================= */

/* ROTG - Generate plane rotation */
typedef void (*fb_srotg_fn)(float *a, float *b, float *c, float *s);
typedef void (*fb_drotg_fn)(double *a, double *b, double *c, double *s);
typedef void (*fb_crotg_fn)(fb_complex_float_t *a, fb_complex_float_t *b,
                            float *c, fb_complex_float_t *s);
typedef void (*fb_zrotg_fn)(fb_complex_double_t *a, fb_complex_double_t *b,
                            double *c, fb_complex_double_t *s);

/* ROTMG - Generate modified plane rotation */
typedef void (*fb_srotmg_fn)(float *d1, float *d2, float *x1, const float y1,
                             float *param);
typedef void (*fb_drotmg_fn)(double *d1, double *d2, double *x1,
                             const double y1, double *param);

/* ROT - Apply plane rotation */
typedef void (*fb_srot_fn)(const int n, float *x, const int incx,
                           float *y, const int incy, const float c,
                           const float s);
typedef void (*fb_drot_fn)(const int n, double *x, const int incx,
                           double *y, const int incy, const double c,
                           const double s);
typedef void (*fb_crot_fn)(const int n, fb_complex_float_t *x,
                           const int incx, fb_complex_float_t *y,
                           const int incy, const float c,
                           const fb_complex_float_t s);
typedef void (*fb_zrot_fn)(const int n, fb_complex_double_t *x,
                           const int incx, fb_complex_double_t *y,
                           const int incy, const double c,
                           const fb_complex_double_t s);
typedef void (*fb_zdrot_fn)(const int n, fb_complex_double_t *x,
                            const int incx, fb_complex_double_t *y,
                            const int incy, const double c, const double s);

/* ROTM - Apply modified plane rotation */
typedef void (*fb_srotm_fn)(const int n, float *x, const int incx,
                            float *y, const int incy, const float *param);
typedef void (*fb_drotm_fn)(const int n, double *x, const int incx,
                            double *y, const int incy, const double *param);

/* SWAP - Exchange vectors */
typedef void (*fb_sswap_fn)(const int n, float *x, const int incx,
                            float *y, const int incy);
typedef void (*fb_dswap_fn)(const int n, double *x, const int incx,
                            double *y, const int incy);
typedef void (*fb_cswap_fn)(const int n, fb_complex_float_t *x,
                            const int incx, fb_complex_float_t *y,
                            const int incy);
typedef void (*fb_zswap_fn)(const int n, fb_complex_double_t *x,
                            const int incx, fb_complex_double_t *y,
                            const int incy);

/* SCAL - Scale vector */
typedef void (*fb_sscal_fn)(const int n, const float alpha, float *x,
                            const int incx);
typedef void (*fb_dscal_fn)(const int n, const double alpha, double *x,
                            const int incx);
typedef void (*fb_cscal_fn)(const int n, const fb_complex_float_t alpha,
                            fb_complex_float_t *x, const int incx);
typedef void (*fb_zscal_fn)(const int n, const fb_complex_double_t alpha,
                            fb_complex_double_t *x, const int incx);
typedef void (*fb_csscal_fn)(const int n, const float alpha,
                             fb_complex_float_t *x, const int incx);
typedef void (*fb_zdscal_fn)(const int n, const double alpha,
                             fb_complex_double_t *x, const int incx);

/* COPY - Copy vector */
typedef void (*fb_scopy_fn)(const int n, const float *x, const int incx,
                            float *y, const int incy);
typedef void (*fb_dcopy_fn)(const int n, const double *x,
                            const int incx, double *y, const int incy);
typedef void (*fb_ccopy_fn)(const int n, const fb_complex_float_t *x,
                            const int incx, fb_complex_float_t *y,
                            const int incy);
typedef void (*fb_zcopy_fn)(const int n, const fb_complex_double_t *x,
                            const int incx, fb_complex_double_t *y,
                            const int incy);

/* AXPY - y = alpha*x + y */
typedef void (*fb_saxpy_fn)(const int n, const float alpha, const float *x,
                            const int incx, float *y, const int incy);
typedef void (*fb_daxpy_fn)(const int n, const double alpha,
                            const double *x, const int incx, double *y,
                            const int incy);
typedef void (*fb_caxpy_fn)(const int n, const fb_complex_float_t alpha,
                            const fb_complex_float_t *x, const int incx,
                            fb_complex_float_t *y, const int incy);
typedef void (*fb_zaxpy_fn)(const int n, const fb_complex_double_t alpha,
                            const fb_complex_double_t *x, const int incx,
                            fb_complex_double_t *y, const int incy);

/* DOT - Dot product */
typedef float (*fb_sdot_fn)(const int n, const float *x, const int incx,
                            const float *y, const int incy);
typedef double (*fb_ddot_fn)(const int n, const double *x,
                             const int incx, const double *y,
                             const int incy);
typedef float (*fb_sdsdot_fn)(const int n, const float sb, const float *x,
                              const int incx, const float *y,
                              const int incy);
typedef double (*fb_dsdot_fn)(const int n, const float *x,
                              const int incx, const float *y,
                              const int incy);

/* DOTU - Unconjugated dot product (complex) */
typedef void (*fb_cdotu_fn)(fb_complex_float_t *result, const int n,
                            const fb_complex_float_t *x, const int incx,
                            const fb_complex_float_t *y, const int incy);
typedef void (*fb_zdotu_fn)(fb_complex_double_t *result, const int n,
                            const fb_complex_double_t *x, const int incx,
                            const fb_complex_double_t *y, const int incy);

/* DOTC - Conjugated dot product (complex) */
typedef void (*fb_cdotc_fn)(fb_complex_float_t *result, const int n,
                            const fb_complex_float_t *x, const int incx,
                            const fb_complex_float_t *y, const int incy);
typedef void (*fb_zdotc_fn)(fb_complex_double_t *result, const int n,
                            const fb_complex_double_t *x, const int incx,
                            const fb_complex_double_t *y, const int incy);

/* NRM2 - Euclidean norm */
typedef float (*fb_snrm2_fn)(const int n, const float *x,
                             const int incx);
typedef double (*fb_dnrm2_fn)(const int n, const double *x,
                              const int incx);
typedef float (*fb_scnrm2_fn)(const int n, const fb_complex_float_t *x,
                              const int incx);
typedef double (*fb_dznrm2_fn)(const int n, const fb_complex_double_t *x,
                               const int incx);

/* ASUM - Sum of absolute values */
typedef float (*fb_sasum_fn)(const int n, const float *x,
                             const int incx);
typedef double (*fb_dasum_fn)(const int n, const double *x,
                              const int incx);
typedef float (*fb_scasum_fn)(const int n, const fb_complex_float_t *x,
                              const int incx);
typedef double (*fb_dzasum_fn)(const int n, const fb_complex_double_t *x,
                               const int incx);

/* IAMAX - Index of maximum absolute value */
typedef int (*fb_isamax_fn)(const int n, const float *x,
                                const int incx);
typedef int (*fb_idamax_fn)(const int n, const double *x,
                                const int incx);
typedef int (*fb_icamax_fn)(const int n, const fb_complex_float_t *x,
                                const int incx);
typedef int (*fb_izamax_fn)(const int n, const fb_complex_double_t *x,
                                const int incx);

/* ============================================================================
 * BLAS Level 2: Matrix-Vector Operations (70 functions)
 * ========================================================================= */

/* GEMV - General matrix-vector multiply: y = alpha*op(A)*x + beta*y */
typedef void (*fb_sgemv_fn)(const fb_layout_t layout,
                            const fb_transpose_t trans, const int m,
                            const int n, const float alpha, const float *A,
                            const int lda, const float *x,
                            const int incx, const float beta, float *y,
                            const int incy);
typedef void (*fb_dgemv_fn)(const fb_layout_t layout,
                            const fb_transpose_t trans, const int m,
                            const int n, const double alpha,
                            const double *A, const int lda, const double *x,
                            const int incx, const double beta, double *y,
                            const int incy);
typedef void (*fb_cgemv_fn)(const fb_layout_t layout,
                            const fb_transpose_t trans, const int m,
                            const int n, const fb_complex_float_t alpha,
                            const fb_complex_float_t *A, const int lda,
                            const fb_complex_float_t *x, const int incx,
                            const fb_complex_float_t beta,
                            fb_complex_float_t *y, const int incy);
typedef void (*fb_zgemv_fn)(const fb_layout_t layout,
                            const fb_transpose_t trans, const int m,
                            const int n, const fb_complex_double_t alpha,
                            const fb_complex_double_t *A, const int lda,
                            const fb_complex_double_t *x, const int incx,
                            const fb_complex_double_t beta,
                            fb_complex_double_t *y, const int incy);

/* GBMV - General banded matrix-vector multiply */
typedef void (*fb_sgbmv_fn)(const fb_layout_t layout,
                            const fb_transpose_t trans, const int m,
                            const int n, const int kl, const int ku,
                            const float alpha, const float *A,
                            const int lda, const float *x,
                            const int incx, const float beta, float *y,
                            const int incy);
typedef void (*fb_dgbmv_fn)(const fb_layout_t layout,
                            const fb_transpose_t trans, const int m,
                            const int n, const int kl, const int ku,
                            const double alpha, const double *A,
                            const int lda, const double *x,
                            const int incx, const double beta, double *y,
                            const int incy);
typedef void (*fb_cgbmv_fn)(const fb_layout_t layout,
                            const fb_transpose_t trans, const int m,
                            const int n, const int kl, const int ku,
                            const fb_complex_float_t alpha,
                            const fb_complex_float_t *A, const int lda,
                            const fb_complex_float_t *x, const int incx,
                            const fb_complex_float_t beta,
                            fb_complex_float_t *y, const int incy);
typedef void (*fb_zgbmv_fn)(const fb_layout_t layout,
                            const fb_transpose_t trans, const int m,
                            const int n, const int kl, const int ku,
                            const fb_complex_double_t alpha,
                            const fb_complex_double_t *A, const int lda,
                            const fb_complex_double_t *x, const int incx,
                            const fb_complex_double_t beta,
                            fb_complex_double_t *y, const int incy);

/* HEMV - Hermitian matrix-vector multiply (complex only) */
typedef void (*fb_chemv_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                            const int n, const fb_complex_float_t alpha,
                            const fb_complex_float_t *A, const int lda,
                            const fb_complex_float_t *x, const int incx,
                            const fb_complex_float_t beta,
                            fb_complex_float_t *y, const int incy);
typedef void (*fb_zhemv_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                            const int n, const fb_complex_double_t alpha,
                            const fb_complex_double_t *A, const int lda,
                            const fb_complex_double_t *x, const int incx,
                            const fb_complex_double_t beta,
                            fb_complex_double_t *y, const int incy);

/* HBMV - Hermitian banded matrix-vector multiply (complex only) */
typedef void (*fb_chbmv_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                            const int n, const int k,
                            const fb_complex_float_t alpha,
                            const fb_complex_float_t *A, const int lda,
                            const fb_complex_float_t *x, const int incx,
                            const fb_complex_float_t beta,
                            fb_complex_float_t *y, const int incy);
typedef void (*fb_zhbmv_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                            const int n, const int k,
                            const fb_complex_double_t alpha,
                            const fb_complex_double_t *A, const int lda,
                            const fb_complex_double_t *x, const int incx,
                            const fb_complex_double_t beta,
                            fb_complex_double_t *y, const int incy);

/* HPMV - Hermitian packed matrix-vector multiply (complex only) */
typedef void (*fb_chpmv_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                            const int n, const fb_complex_float_t alpha,
                            const fb_complex_float_t *AP,
                            const fb_complex_float_t *x, const int incx,
                            const fb_complex_float_t beta,
                            fb_complex_float_t *y, const int incy);
typedef void (*fb_zhpmv_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                            const int n, const fb_complex_double_t alpha,
                            const fb_complex_double_t *AP,
                            const fb_complex_double_t *x, const int incx,
                            const fb_complex_double_t beta,
                            fb_complex_double_t *y, const int incy);

/* SYMV - Symmetric matrix-vector multiply */
typedef void (*fb_ssymv_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                            const int n, const float alpha, const float *A,
                            const int lda, const float *x,
                            const int incx, const float beta, float *y,
                            const int incy);
typedef void (*fb_dsymv_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                            const int n, const double alpha,
                            const double *A, const int lda, const double *x,
                            const int incx, const double beta, double *y,
                            const int incy);
typedef void (*fb_csymv_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                            const int n, const void *alpha, const void *A,
                            const int lda, const void *x,
                            const int incx, const void *beta, void *y,
                            const int incy);
typedef void (*fb_zsymv_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                            const int n, const void *alpha, const void *A,
                            const int lda, const void *x,
                            const int incx, const void *beta, void *y,
                            const int incy);

/* SBMV - Symmetric banded matrix-vector multiply */
typedef void (*fb_ssbmv_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                            const int n, const int k, const float alpha,
                            const float *A, const int lda, const float *x,
                            const int incx, const float beta, float *y,
                            const int incy);
typedef void (*fb_dsbmv_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                            const int n, const int k,
                            const double alpha, const double *A,
                            const int lda, const double *x,
                            const int incx, const double beta, double *y,
                            const int incy);

/* SPMV - Symmetric packed matrix-vector multiply */
typedef void (*fb_sspmv_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                            const int n, const float alpha, const float *AP,
                            const float *x, const int incx,
                            const float beta, float *y, const int incy);
typedef void (*fb_dspmv_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                            const int n, const double alpha,
                            const double *AP, const double *x,
                            const int incx, const double beta, double *y,
                            const int incy);

/* TRMV - Triangular matrix-vector multiply */
typedef void (*fb_strmv_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                            const fb_transpose_t trans, const fb_diag_t diag,
                            const int n, const float *A, const int lda,
                            float *x, const int incx);
typedef void (*fb_dtrmv_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                            const fb_transpose_t trans, const fb_diag_t diag,
                            const int n, const double *A, const int lda,
                            double *x, const int incx);
typedef void (*fb_ctrmv_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                            const fb_transpose_t trans, const fb_diag_t diag,
                            const int n, const fb_complex_float_t *A,
                            const int lda, fb_complex_float_t *x,
                            const int incx);
typedef void (*fb_ztrmv_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                            const fb_transpose_t trans, const fb_diag_t diag,
                            const int n, const fb_complex_double_t *A,
                            const int lda, fb_complex_double_t *x,
                            const int incx);

/* TBMV - Triangular banded matrix-vector multiply */
typedef void (*fb_stbmv_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                            const fb_transpose_t trans, const fb_diag_t diag,
                            const int n, const int k, const float *A,
                            const int lda, float *x, const int incx);
typedef void (*fb_dtbmv_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                            const fb_transpose_t trans, const fb_diag_t diag,
                            const int n, const int k, const double *A,
                            const int lda, double *x, const int incx);
typedef void (*fb_ctbmv_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                            const fb_transpose_t trans, const fb_diag_t diag,
                            const int n, const int k,
                            const fb_complex_float_t *A, const int lda,
                            fb_complex_float_t *x, const int incx);
typedef void (*fb_ztbmv_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                            const fb_transpose_t trans, const fb_diag_t diag,
                            const int n, const int k,
                            const fb_complex_double_t *A, const int lda,
                            fb_complex_double_t *x, const int incx);

/* TPMV - Triangular packed matrix-vector multiply */
typedef void (*fb_stpmv_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                            const fb_transpose_t trans, const fb_diag_t diag,
                            const int n, const float *AP, float *x,
                            const int incx);
typedef void (*fb_dtpmv_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                            const fb_transpose_t trans, const fb_diag_t diag,
                            const int n, const double *AP, double *x,
                            const int incx);
typedef void (*fb_ctpmv_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                            const fb_transpose_t trans, const fb_diag_t diag,
                            const int n, const fb_complex_float_t *AP,
                            fb_complex_float_t *x, const int incx);
typedef void (*fb_ztpmv_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                            const fb_transpose_t trans, const fb_diag_t diag,
                            const int n, const fb_complex_double_t *AP,
                            fb_complex_double_t *x, const int incx);

/* TRSV - Triangular system solve */
typedef void (*fb_strsv_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                            const fb_transpose_t trans, const fb_diag_t diag,
                            const int n, const float *A, const int lda,
                            float *x, const int incx);
typedef void (*fb_dtrsv_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                            const fb_transpose_t trans, const fb_diag_t diag,
                            const int n, const double *A, const int lda,
                            double *x, const int incx);
typedef void (*fb_ctrsv_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                            const fb_transpose_t trans, const fb_diag_t diag,
                            const int n, const fb_complex_float_t *A,
                            const int lda, fb_complex_float_t *x,
                            const int incx);
typedef void (*fb_ztrsv_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                            const fb_transpose_t trans, const fb_diag_t diag,
                            const int n, const fb_complex_double_t *A,
                            const int lda, fb_complex_double_t *x,
                            const int incx);

/* TBSV - Triangular banded system solve */
typedef void (*fb_stbsv_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                            const fb_transpose_t trans, const fb_diag_t diag,
                            const int n, const int k, const float *A,
                            const int lda, float *x, const int incx);
typedef void (*fb_dtbsv_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                            const fb_transpose_t trans, const fb_diag_t diag,
                            const int n, const int k, const double *A,
                            const int lda, double *x, const int incx);
typedef void (*fb_ctbsv_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                            const fb_transpose_t trans, const fb_diag_t diag,
                            const int n, const int k,
                            const fb_complex_float_t *A, const int lda,
                            fb_complex_float_t *x, const int incx);
typedef void (*fb_ztbsv_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                            const fb_transpose_t trans, const fb_diag_t diag,
                            const int n, const int k,
                            const fb_complex_double_t *A, const int lda,
                            fb_complex_double_t *x, const int incx);

/* TPSV - Triangular packed system solve */
typedef void (*fb_stpsv_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                            const fb_transpose_t trans, const fb_diag_t diag,
                            const int n, const float *AP, float *x,
                            const int incx);
typedef void (*fb_dtpsv_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                            const fb_transpose_t trans, const fb_diag_t diag,
                            const int n, const double *AP, double *x,
                            const int incx);
typedef void (*fb_ctpsv_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                            const fb_transpose_t trans, const fb_diag_t diag,
                            const int n, const fb_complex_float_t *AP,
                            fb_complex_float_t *x, const int incx);
typedef void (*fb_ztpsv_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                            const fb_transpose_t trans, const fb_diag_t diag,
                            const int n, const fb_complex_double_t *AP,
                            fb_complex_double_t *x, const int incx);

/* GER - General rank-1 update: A = alpha*x*y^T + A (real only) */
typedef void (*fb_sger_fn)(const fb_layout_t layout, const int m,
                           const int n, const float alpha, const float *x,
                           const int incx, const float *y,
                           const int incy, float *A, const int lda);
typedef void (*fb_dger_fn)(const fb_layout_t layout, const int m,
                           const int n, const double alpha, const double *x,
                           const int incx, const double *y,
                           const int incy, double *A, const int lda);

/* GERU - General rank-1 update unconjugated: A = alpha*x*y^T + A (complex) */
typedef void (*fb_cgeru_fn)(const fb_layout_t layout, const int m,
                            const int n, const fb_complex_float_t alpha,
                            const fb_complex_float_t *x, const int incx,
                            const fb_complex_float_t *y, const int incy,
                            fb_complex_float_t *A, const int lda);
typedef void (*fb_zgeru_fn)(const fb_layout_t layout, const int m,
                            const int n, const fb_complex_double_t alpha,
                            const fb_complex_double_t *x, const int incx,
                            const fb_complex_double_t *y, const int incy,
                            fb_complex_double_t *A, const int lda);

/* GERC - General rank-1 update conjugated: A = alpha*x*y^H + A (complex) */
typedef void (*fb_cgerc_fn)(const fb_layout_t layout, const int m,
                            const int n, const fb_complex_float_t alpha,
                            const fb_complex_float_t *x, const int incx,
                            const fb_complex_float_t *y, const int incy,
                            fb_complex_float_t *A, const int lda);
typedef void (*fb_zgerc_fn)(const fb_layout_t layout, const int m,
                            const int n, const fb_complex_double_t alpha,
                            const fb_complex_double_t *x, const int incx,
                            const fb_complex_double_t *y, const int incy,
                            fb_complex_double_t *A, const int lda);

/* HER - Hermitian rank-1 update: A = alpha*x*x^H + A (complex only) */
typedef void (*fb_cher_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                           const int n, const float alpha,
                           const fb_complex_float_t *x, const int incx,
                           fb_complex_float_t *A, const int lda);
typedef void (*fb_zher_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                           const int n, const double alpha,
                           const fb_complex_double_t *x, const int incx,
                           fb_complex_double_t *A, const int lda);

/* HPR - Hermitian packed rank-1 update (complex only) */
typedef void (*fb_chpr_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                           const int n, const float alpha,
                           const fb_complex_float_t *x, const int incx,
                           fb_complex_float_t *AP);
typedef void (*fb_zhpr_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                           const int n, const double alpha,
                           const fb_complex_double_t *x, const int incx,
                           fb_complex_double_t *AP);

/* HER2 - Hermitian rank-2 update: A = alpha*x*y^H + conj(alpha)*y*x^H + A
 * (complex only) */
typedef void (*fb_cher2_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                            const int n, const fb_complex_float_t alpha,
                            const fb_complex_float_t *x, const int incx,
                            const fb_complex_float_t *y, const int incy,
                            fb_complex_float_t *A, const int lda);
typedef void (*fb_zher2_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                            const int n, const fb_complex_double_t alpha,
                            const fb_complex_double_t *x, const int incx,
                            const fb_complex_double_t *y, const int incy,
                            fb_complex_double_t *A, const int lda);

/* HPR2 - Hermitian packed rank-2 update (complex only) */
typedef void (*fb_chpr2_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                            const int n, const fb_complex_float_t alpha,
                            const fb_complex_float_t *x, const int incx,
                            const fb_complex_float_t *y, const int incy,
                            fb_complex_float_t *AP);
typedef void (*fb_zhpr2_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                            const int n, const fb_complex_double_t alpha,
                            const fb_complex_double_t *x, const int incx,
                            const fb_complex_double_t *y, const int incy,
                            fb_complex_double_t *AP);

/* SYR - Symmetric rank-1 update: A = alpha*x*x^T + A */
typedef void (*fb_ssyr_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                           const int n, const float alpha, const float *x,
                           const int incx, float *A, const int lda);
typedef void (*fb_dsyr_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                           const int n, const double alpha, const double *x,
                           const int incx, double *A, const int lda);
typedef void (*fb_csyr_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                           const int n, const void *alpha, const void *x,
                           const int incx, void *A, const int lda);
typedef void (*fb_zsyr_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                           const int n, const void *alpha, const void *x,
                           const int incx, void *A, const int lda);

/* SPR - Symmetric packed rank-1 update (real only) */
typedef void (*fb_sspr_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                           const int n, const float alpha, const float *x,
                           const int incx, float *AP);
typedef void (*fb_dspr_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                           const int n, const double alpha, const double *x,
                           const int incx, double *AP);

/* SYR2 - Symmetric rank-2 update: A = alpha*x*y^T + alpha*y*x^T + A (real only)
 */
typedef void (*fb_ssyr2_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                            const int n, const float alpha, const float *x,
                            const int incx, const float *y,
                            const int incy, float *A, const int lda);
typedef void (*fb_dsyr2_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                            const int n, const double alpha,
                            const double *x, const int incx,
                            const double *y, const int incy, double *A,
                            const int lda);

/* SPR2 - Symmetric packed rank-2 update (real only) */
typedef void (*fb_sspr2_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                            const int n, const float alpha, const float *x,
                            const int incx, const float *y,
                            const int incy, float *AP);
typedef void (*fb_dspr2_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                            const int n, const double alpha,
                            const double *x, const int incx,
                            const double *y, const int incy, double *AP);

/* ============================================================================
 * BLAS Level 3: Matrix-Matrix Operations (30 functions)
 * ========================================================================= */

/* GEMM - General matrix-matrix multiply: C = alpha*op(A)*op(B) + beta*C */
typedef void (*fb_sgemm_fn)(const fb_layout_t layout,
                            const fb_transpose_t transA,
                            const fb_transpose_t transB, const int m,
                            const int n, const int k, const float alpha,
                            const float *A, const int lda, const float *B,
                            const int ldb, const float beta, float *C,
                            const int ldc);
typedef void (*fb_dgemm_fn)(const fb_layout_t layout,
                            const fb_transpose_t transA,
                            const fb_transpose_t transB, const int m,
                            const int n, const int k,
                            const double alpha, const double *A,
                            const int lda, const double *B,
                            const int ldb, const double beta, double *C,
                            const int ldc);
typedef void (*fb_cgemm_fn)(const fb_layout_t layout,
                            const fb_transpose_t transA,
                            const fb_transpose_t transB, const int m,
                            const int n, const int k,
                            const fb_complex_float_t alpha,
                            const fb_complex_float_t *A, const int lda,
                            const fb_complex_float_t *B, const int ldb,
                            const fb_complex_float_t beta,
                            fb_complex_float_t *C, const int ldc);
typedef void (*fb_zgemm_fn)(const fb_layout_t layout,
                            const fb_transpose_t transA,
                            const fb_transpose_t transB, const int m,
                            const int n, const int k,
                            const fb_complex_double_t alpha,
                            const fb_complex_double_t *A, const int lda,
                            const fb_complex_double_t *B, const int ldb,
                            const fb_complex_double_t beta,
                            fb_complex_double_t *C, const int ldc);

/* SYMM - Symmetric matrix-matrix multiply: C = alpha*A*B + beta*C or C =
 * alpha*B*A + beta*C */
typedef void (*fb_ssymm_fn)(const fb_layout_t layout, const fb_side_t side,
                            const fb_uplo_t uplo, const int m,
                            const int n, const float alpha, const float *A,
                            const int lda, const float *B,
                            const int ldb, const float beta, float *C,
                            const int ldc);
typedef void (*fb_dsymm_fn)(const fb_layout_t layout, const fb_side_t side,
                            const fb_uplo_t uplo, const int m,
                            const int n, const double alpha,
                            const double *A, const int lda, const double *B,
                            const int ldb, const double beta, double *C,
                            const int ldc);
typedef void (*fb_csymm_fn)(const fb_layout_t layout, const fb_side_t side,
                            const fb_uplo_t uplo, const int m,
                            const int n, const fb_complex_float_t alpha,
                            const fb_complex_float_t *A, const int lda,
                            const fb_complex_float_t *B, const int ldb,
                            const fb_complex_float_t beta,
                            fb_complex_float_t *C, const int ldc);
typedef void (*fb_zsymm_fn)(const fb_layout_t layout, const fb_side_t side,
                            const fb_uplo_t uplo, const int m,
                            const int n, const fb_complex_double_t alpha,
                            const fb_complex_double_t *A, const int lda,
                            const fb_complex_double_t *B, const int ldb,
                            const fb_complex_double_t beta,
                            fb_complex_double_t *C, const int ldc);

/* HEMM - Hermitian matrix-matrix multiply (complex only) */
typedef void (*fb_chemm_fn)(const fb_layout_t layout, const fb_side_t side,
                            const fb_uplo_t uplo, const int m,
                            const int n, const fb_complex_float_t alpha,
                            const fb_complex_float_t *A, const int lda,
                            const fb_complex_float_t *B, const int ldb,
                            const fb_complex_float_t beta,
                            fb_complex_float_t *C, const int ldc);
typedef void (*fb_zhemm_fn)(const fb_layout_t layout, const fb_side_t side,
                            const fb_uplo_t uplo, const int m,
                            const int n, const fb_complex_double_t alpha,
                            const fb_complex_double_t *A, const int lda,
                            const fb_complex_double_t *B, const int ldb,
                            const fb_complex_double_t beta,
                            fb_complex_double_t *C, const int ldc);

/* SYRK - Symmetric rank-k update: C = alpha*A*A^T + beta*C or C = alpha*A^T*A +
 * beta*C */
typedef void (*fb_ssyrk_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                            const fb_transpose_t trans, const int n,
                            const int k, const float alpha, const float *A,
                            const int lda, const float beta, float *C,
                            const int ldc);
typedef void (*fb_dsyrk_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                            const fb_transpose_t trans, const int n,
                            const int k, const double alpha,
                            const double *A, const int lda,
                            const double beta, double *C, const int ldc);
typedef void (*fb_csyrk_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                            const fb_transpose_t trans, const int n,
                            const int k, const fb_complex_float_t alpha,
                            const fb_complex_float_t *A, const int lda,
                            const fb_complex_float_t beta,
                            fb_complex_float_t *C, const int ldc);
typedef void (*fb_zsyrk_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                            const fb_transpose_t trans, const int n,
                            const int k, const fb_complex_double_t alpha,
                            const fb_complex_double_t *A, const int lda,
                            const fb_complex_double_t beta,
                            fb_complex_double_t *C, const int ldc);

/* HERK - Hermitian rank-k update (complex only) */
typedef void (*fb_cherk_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                            const fb_transpose_t trans, const int n,
                            const int k, const float alpha,
                            const fb_complex_float_t *A, const int lda,
                            const float beta, fb_complex_float_t *C,
                            const int ldc);
typedef void (*fb_zherk_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                            const fb_transpose_t trans, const int n,
                            const int k, const double alpha,
                            const fb_complex_double_t *A, const int lda,
                            const double beta, fb_complex_double_t *C,
                            const int ldc);

/* SYR2K - Symmetric rank-2k update: C = alpha*A*B^T + alpha*B*A^T + beta*C */
typedef void (*fb_ssyr2k_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                             const fb_transpose_t trans, const int n,
                             const int k, const float alpha, const float *A,
                             const int lda, const float *B,
                             const int ldb, const float beta, float *C,
                             const int ldc);
typedef void (*fb_dsyr2k_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                             const fb_transpose_t trans, const int n,
                             const int k, const double alpha,
                             const double *A, const int lda,
                             const double *B, const int ldb,
                             const double beta, double *C, const int ldc);
typedef void (*fb_csyr2k_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                             const fb_transpose_t trans, const int n,
                             const int k, const fb_complex_float_t alpha,
                             const fb_complex_float_t *A, const int lda,
                             const fb_complex_float_t *B, const int ldb,
                             const fb_complex_float_t beta,
                             fb_complex_float_t *C, const int ldc);
typedef void (*fb_zsyr2k_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                             const fb_transpose_t trans, const int n,
                             const int k, const fb_complex_double_t alpha,
                             const fb_complex_double_t *A, const int lda,
                             const fb_complex_double_t *B, const int ldb,
                             const fb_complex_double_t beta,
                             fb_complex_double_t *C, const int ldc);

/* HER2K - Hermitian rank-2k update (complex only) */
typedef void (*fb_cher2k_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                             const fb_transpose_t trans, const int n,
                             const int k, const fb_complex_float_t alpha,
                             const fb_complex_float_t *A, const int lda,
                             const fb_complex_float_t *B, const int ldb,
                             const float beta, fb_complex_float_t *C,
                             const int ldc);
typedef void (*fb_zher2k_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                             const fb_transpose_t trans, const int n,
                             const int k, const fb_complex_double_t alpha,
                             const fb_complex_double_t *A, const int lda,
                             const fb_complex_double_t *B, const int ldb,
                             const double beta, fb_complex_double_t *C,
                             const int ldc);

/* TRMM - Triangular matrix-matrix multiply: B = alpha*op(A)*B or B =
 * alpha*B*op(A) */
typedef void (*fb_strmm_fn)(const fb_layout_t layout, const fb_side_t side,
                            const fb_uplo_t uplo, const fb_transpose_t trans,
                            const fb_diag_t diag, const int m,
                            const int n, const float alpha, const float *A,
                            const int lda, float *B, const int ldb);
typedef void (*fb_dtrmm_fn)(const fb_layout_t layout, const fb_side_t side,
                            const fb_uplo_t uplo, const fb_transpose_t trans,
                            const fb_diag_t diag, const int m,
                            const int n, const double alpha,
                            const double *A, const int lda, double *B,
                            const int ldb);
typedef void (*fb_ctrmm_fn)(const fb_layout_t layout, const fb_side_t side,
                            const fb_uplo_t uplo, const fb_transpose_t trans,
                            const fb_diag_t diag, const int m,
                            const int n, const fb_complex_float_t alpha,
                            const fb_complex_float_t *A, const int lda,
                            fb_complex_float_t *B, const int ldb);
typedef void (*fb_ztrmm_fn)(const fb_layout_t layout, const fb_side_t side,
                            const fb_uplo_t uplo, const fb_transpose_t trans,
                            const fb_diag_t diag, const int m,
                            const int n, const fb_complex_double_t alpha,
                            const fb_complex_double_t *A, const int lda,
                            fb_complex_double_t *B, const int ldb);

/* TRSM - Triangular system solve with multiple RHS: op(A)*X = alpha*B or
 * X*op(A) = alpha*B */
typedef void (*fb_strsm_fn)(const fb_layout_t layout, const fb_side_t side,
                            const fb_uplo_t uplo, const fb_transpose_t trans,
                            const fb_diag_t diag, const int m,
                            const int n, const float alpha, const float *A,
                            const int lda, float *B, const int ldb);
typedef void (*fb_dtrsm_fn)(const fb_layout_t layout, const fb_side_t side,
                            const fb_uplo_t uplo, const fb_transpose_t trans,
                            const fb_diag_t diag, const int m,
                            const int n, const double alpha,
                            const double *A, const int lda, double *B,
                            const int ldb);
typedef void (*fb_ctrsm_fn)(const fb_layout_t layout, const fb_side_t side,
                            const fb_uplo_t uplo, const fb_transpose_t trans,
                            const fb_diag_t diag, const int m,
                            const int n, const fb_complex_float_t alpha,
                            const fb_complex_float_t *A, const int lda,
                            fb_complex_float_t *B, const int ldb);
typedef void (*fb_ztrsm_fn)(const fb_layout_t layout, const fb_side_t side,
                            const fb_uplo_t uplo, const fb_transpose_t trans,
                            const fb_diag_t diag, const int m,
                            const int n, const fb_complex_double_t alpha,
                            const fb_complex_double_t *A, const int lda,
                            fb_complex_double_t *B, const int ldb);

/* ============================================================================
 * LAPACK Subset: Linear Algebra Operations (60 functions)
 * ========================================================================= */

/* GESV - General linear system solve: A*X = B */
typedef int (*fb_sgesv_fn)(const fb_layout_t layout, const int n,
                               const int nrhs, float *A, const int lda,
                               int *ipiv, float *B, const int ldb);
typedef int (*fb_dgesv_fn)(const fb_layout_t layout, const int n,
                               const int nrhs, double *A, const int lda,
                               int *ipiv, double *B, const int ldb);
typedef int (*fb_cgesv_fn)(const fb_layout_t layout, const int n,
                               const int nrhs, fb_complex_float_t *A,
                               const int lda, int *ipiv,
                               fb_complex_float_t *B, const int ldb);
typedef int (*fb_zgesv_fn)(const fb_layout_t layout, const int n,
                               const int nrhs, fb_complex_double_t *A,
                               const int lda, int *ipiv,
                               fb_complex_double_t *B, const int ldb);

/* POSV - Positive-definite linear system solve: A*X = B */
typedef int (*fb_sposv_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                               const int n, const int nrhs, float *A,
                               const int lda, float *B, const int ldb);
typedef int (*fb_dposv_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                               const int n, const int nrhs, double *A,
                               const int lda, double *B, const int ldb);
typedef int (*fb_cposv_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                               const int n, const int nrhs,
                               fb_complex_float_t *A, const int lda,
                               fb_complex_float_t *B, const int ldb);
typedef int (*fb_zposv_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                               const int n, const int nrhs,
                               fb_complex_double_t *A, const int lda,
                               fb_complex_double_t *B, const int ldb);

/* SYSV - Symmetric indefinite linear system solve */
typedef int (*fb_ssysv_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                               const int n, const int nrhs, float *A,
                               const int lda, int *ipiv, float *B,
                               const int ldb);
typedef int (*fb_dsysv_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                               const int n, const int nrhs, double *A,
                               const int lda, int *ipiv, double *B,
                               const int ldb);
typedef int (*fb_csysv_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                               const int n, const int nrhs,
                               fb_complex_float_t *A, const int lda,
                               int *ipiv, fb_complex_float_t *B,
                               const int ldb);
typedef int (*fb_zsysv_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                               const int n, const int nrhs,
                               fb_complex_double_t *A, const int lda,
                               int *ipiv, fb_complex_double_t *B,
                               const int ldb);

/* HESV - Hermitian indefinite linear system solve (complex only) */
typedef int (*fb_chesv_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                               const int n, const int nrhs,
                               fb_complex_float_t *A, const int lda,
                               int *ipiv, fb_complex_float_t *B,
                               const int ldb);
typedef int (*fb_zhesv_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                               const int n, const int nrhs,
                               fb_complex_double_t *A, const int lda,
                               int *ipiv, fb_complex_double_t *B,
                               const int ldb);

/* GETRF - LU factorization */
typedef int (*fb_sgetrf_fn)(const fb_layout_t layout, const int m,
                                const int n, float *A, const int lda,
                                int *ipiv);
typedef int (*fb_dgetrf_fn)(const fb_layout_t layout, const int m,
                                const int n, double *A, const int lda,
                                int *ipiv);
typedef int (*fb_cgetrf_fn)(const fb_layout_t layout, const int m,
                                const int n, fb_complex_float_t *A,
                                const int lda, int *ipiv);
typedef int (*fb_zgetrf_fn)(const fb_layout_t layout, const int m,
                                const int n, fb_complex_double_t *A,
                                const int lda, int *ipiv);

/* POTRF - Cholesky factorization */
typedef int (*fb_spotrf_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                                const int n, float *A, const int lda);
typedef int (*fb_dpotrf_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                                const int n, double *A, const int lda);
typedef int (*fb_cpotrf_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                                const int n, fb_complex_float_t *A,
                                const int lda);
typedef int (*fb_zpotrf_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                                const int n, fb_complex_double_t *A,
                                const int lda);

/* GETRS - Solve using LU factorization */
typedef int (*fb_sgetrs_fn)(const fb_layout_t layout,
                                const fb_transpose_t trans, const int n,
                                const int nrhs, const float *A,
                                const int lda, const int *ipiv,
                                float *B, const int ldb);
typedef int (*fb_dgetrs_fn)(const fb_layout_t layout,
                                const fb_transpose_t trans, const int n,
                                const int nrhs, const double *A,
                                const int lda, const int *ipiv,
                                double *B, const int ldb);
typedef int (*fb_cgetrs_fn)(const fb_layout_t layout,
                                const fb_transpose_t trans, const int n,
                                const int nrhs, const fb_complex_float_t *A,
                                const int lda, const int *ipiv,
                                fb_complex_float_t *B, const int ldb);
typedef int (*fb_zgetrs_fn)(const fb_layout_t layout,
                                const fb_transpose_t trans, const int n,
                                const int nrhs,
                                const fb_complex_double_t *A, const int lda,
                                const int *ipiv, fb_complex_double_t *B,
                                const int ldb);

/* POTRS - Solve using Cholesky factorization */
typedef int (*fb_spotrs_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                                const int n, const int nrhs,
                                const float *A, const int lda, float *B,
                                const int ldb);
typedef int (*fb_dpotrs_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                                const int n, const int nrhs,
                                const double *A, const int lda, double *B,
                                const int ldb);
typedef int (*fb_cpotrs_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                                const int n, const int nrhs,
                                const fb_complex_float_t *A, const int lda,
                                fb_complex_float_t *B, const int ldb);
typedef int (*fb_zpotrs_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                                const int n, const int nrhs,
                                const fb_complex_double_t *A, const int lda,
                                fb_complex_double_t *B, const int ldb);

/* GETRI - Matrix inversion using LU factorization */
typedef int (*fb_sgetri_fn)(const fb_layout_t layout, const int n,
                                float *A, const int lda,
                                const int *ipiv);
typedef int (*fb_dgetri_fn)(const fb_layout_t layout, const int n,
                                double *A, const int lda,
                                const int *ipiv);
typedef int (*fb_cgetri_fn)(const fb_layout_t layout, const int n,
                                fb_complex_float_t *A, const int lda,
                                const int *ipiv);
typedef int (*fb_zgetri_fn)(const fb_layout_t layout, const int n,
                                fb_complex_double_t *A, const int lda,
                                const int *ipiv);

/* POTRI - Matrix inversion using Cholesky factorization */
typedef int (*fb_spotri_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                                const int n, float *A, const int lda);
typedef int (*fb_dpotri_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                                const int n, double *A, const int lda);
typedef int (*fb_cpotri_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                                const int n, fb_complex_float_t *A,
                                const int lda);
typedef int (*fb_zpotri_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                                const int n, fb_complex_double_t *A,
                                const int lda);

/* TRTRI - Triangular matrix inversion */
typedef int (*fb_strtri_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                                const fb_diag_t diag, const int n, float *A,
                                const int lda);
typedef int (*fb_dtrtri_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                                const fb_diag_t diag, const int n,
                                double *A, const int lda);
typedef int (*fb_ctrtri_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                                const fb_diag_t diag, const int n,
                                fb_complex_float_t *A, const int lda);
typedef int (*fb_ztrtri_fn)(const fb_layout_t layout, const fb_uplo_t uplo,
                                const fb_diag_t diag, const int n,
                                fb_complex_double_t *A, const int lda);

/* GEEV - Eigenvalue decomposition (general matrix) */
typedef int (*fb_sgeev_fn)(const fb_layout_t layout, const char jobvl,
                               const char jobvr, const int n, float *A,
                               const int lda, float *wr, float *wi,
                               float *VL, const int ldvl, float *VR,
                               const int ldvr);
typedef int (*fb_dgeev_fn)(const fb_layout_t layout, const char jobvl,
                               const char jobvr, const int n, double *A,
                               const int lda, double *wr, double *wi,
                               double *VL, const int ldvl, double *VR,
                               const int ldvr);
typedef int (*fb_cgeev_fn)(const fb_layout_t layout, const char jobvl,
                               const char jobvr, const int n,
                               fb_complex_float_t *A, const int lda,
                               fb_complex_float_t *w, fb_complex_float_t *VL,
                               const int ldvl, fb_complex_float_t *VR,
                               const int ldvr);
typedef int (*fb_zgeev_fn)(const fb_layout_t layout, const char jobvl,
                               const char jobvr, const int n,
                               fb_complex_double_t *A, const int lda,
                               fb_complex_double_t *w, fb_complex_double_t *VL,
                               const int ldvl, fb_complex_double_t *VR,
                               const int ldvr);

/* SYEV - Eigenvalue decomposition (symmetric real matrix) */
typedef int (*fb_ssyev_fn)(const fb_layout_t layout, const char jobz,
                               const fb_uplo_t uplo, const int n, float *A,
                               const int lda, float *w);
typedef int (*fb_dsyev_fn)(const fb_layout_t layout, const char jobz,
                               const fb_uplo_t uplo, const int n, double *A,
                               const int lda, double *w);

/* HEEV - Eigenvalue decomposition (Hermitian complex matrix) */
typedef int (*fb_cheev_fn)(const fb_layout_t layout, const char jobz,
                               const fb_uplo_t uplo, const int n,
                               fb_complex_float_t *A, const int lda,
                               float *w);
typedef int (*fb_zheev_fn)(const fb_layout_t layout, const char jobz,
                               const fb_uplo_t uplo, const int n,
                               fb_complex_double_t *A, const int lda,
                               double *w);

/* GESVD - Singular value decomposition */
typedef int (*fb_sgesvd_fn)(const fb_layout_t layout, const char jobu,
                                const char jobvt, const int m,
                                const int n, float *A, const int lda,
                                float *s, float *U, const int ldu,
                                float *VT, const int ldvt, float *superb);
typedef int (*fb_dgesvd_fn)(const fb_layout_t layout, const char jobu,
                                const char jobvt, const int m,
                                const int n, double *A, const int lda,
                                double *s, double *U, const int ldu,
                                double *VT, const int ldvt, double *superb);
typedef int (*fb_cgesvd_fn)(const fb_layout_t layout, const char jobu,
                                const char jobvt, const int m,
                                const int n, fb_complex_float_t *A,
                                const int lda, float *s,
                                fb_complex_float_t *U, const int ldu,
                                fb_complex_float_t *VT, const int ldvt,
                                float *superb);
typedef int (*fb_zgesvd_fn)(const fb_layout_t layout, const char jobu,
                                const char jobvt, const int m,
                                const int n, fb_complex_double_t *A,
                                const int lda, double *s,
                                fb_complex_double_t *U, const int ldu,
                                fb_complex_double_t *VT, const int ldvt,
                                double *superb);

/* GEQRF - QR factorization */
typedef int (*fb_sgeqrf_fn)(const fb_layout_t layout, const int m,
                                const int n, float *A, const int lda,
                                float *tau);
typedef int (*fb_dgeqrf_fn)(const fb_layout_t layout, const int m,
                                const int n, double *A, const int lda,
                                double *tau);
typedef int (*fb_cgeqrf_fn)(const fb_layout_t layout, const int m,
                                const int n, fb_complex_float_t *A,
                                const int lda, fb_complex_float_t *tau);
typedef int (*fb_zgeqrf_fn)(const fb_layout_t layout, const int m,
                                const int n, fb_complex_double_t *A,
                                const int lda, fb_complex_double_t *tau);

/* ORGQR - Generate Q from QR factorization (real) */
typedef int (*fb_sorgqr_fn)(const fb_layout_t layout, const int m,
                                const int n, const int k, float *A,
                                const int lda, const float *tau);
typedef int (*fb_dorgqr_fn)(const fb_layout_t layout, const int m,
                                const int n, const int k, double *A,
                                const int lda, const double *tau);

/* UNGQR - Generate Q from QR factorization (complex) */
typedef int (*fb_cungqr_fn)(const fb_layout_t layout, const int m,
                                const int n, const int k,
                                fb_complex_float_t *A, const int lda,
                                const fb_complex_float_t *tau);
typedef int (*fb_zungqr_fn)(const fb_layout_t layout, const int m,
                                const int n, const int k,
                                fb_complex_double_t *A, const int lda,
                                const fb_complex_double_t *tau);

/* GELS - Solve overdetermined/underdetermined linear systems using QR or LQ */
typedef int (*fb_sgels_fn)(const fb_layout_t layout,
                               const fb_transpose_t trans, const int m,
                               const int n, const int nrhs, float *A,
                               const int lda, float *B, const int ldb);
typedef int (*fb_dgels_fn)(const fb_layout_t layout,
                               const fb_transpose_t trans, const int m,
                               const int n, const int nrhs, double *A,
                               const int lda, double *B, const int ldb);
typedef int (*fb_cgels_fn)(const fb_layout_t layout,
                               const fb_transpose_t trans, const int m,
                               const int n, const int nrhs,
                               fb_complex_float_t *A, const int lda,
                               fb_complex_float_t *B, const int ldb);
typedef int (*fb_zgels_fn)(const fb_layout_t layout,
                               const fb_transpose_t trans, const int m,
                               const int n, const int nrhs,
                               fb_complex_double_t *A, const int lda,
                               fb_complex_double_t *B, const int ldb);

/* ORMQR - Apply Q from QR factorization (real) */
typedef int (*fb_sormqr_fn)(const fb_layout_t layout, const fb_side_t side,
                                const fb_transpose_t trans, const int m,
                                const int n, const int k,
                                const float *A, const int lda,
                                const float *tau, float *C, const int ldc);
typedef int (*fb_dormqr_fn)(const fb_layout_t layout, const fb_side_t side,
                                const fb_transpose_t trans, const int m,
                                const int n, const int k,
                                const double *A, const int lda,
                                const double *tau, double *C,
                                const int ldc);

/* UNMQR - Apply Q from QR factorization (complex) */
typedef int (*fb_cunmqr_fn)(const fb_layout_t layout, const fb_side_t side,
                                const fb_transpose_t trans, const int m,
                                const int n, const int k,
                                const fb_complex_float_t *A, const int lda,
                                const fb_complex_float_t *tau,
                                fb_complex_float_t *C, const int ldc);
typedef int (*fb_zunmqr_fn)(const fb_layout_t layout, const fb_side_t side,
                                const fb_transpose_t trans, const int m,
                                const int n, const int k,
                                const fb_complex_double_t *A, const int lda,
                                const fb_complex_double_t *tau,
                                fb_complex_double_t *C, const int ldc);

/* GELSD - Least squares with SVD divide-and-conquer */
typedef int (*fb_sgelsd_fn)(const fb_layout_t layout, const int m,
                                const int n, const int nrhs, float *A,
                                const int lda, float *B, const int ldb,
                                float *s, float rcond, int *rank);
typedef int (*fb_dgelsd_fn)(const fb_layout_t layout, const int m,
                                const int n, const int nrhs, double *A,
                                const int lda, double *B, const int ldb,
                                double *s, double rcond, int *rank);
typedef int (*fb_cgelsd_fn)(const fb_layout_t layout, const int m,
                                const int n, const int nrhs,
                                fb_complex_float_t *A, const int lda,
                                fb_complex_float_t *B, const int ldb,
                                float *s, float rcond, int *rank);
typedef int (*fb_zgelsd_fn)(const fb_layout_t layout, const int m,
                                const int n, const int nrhs,
                                fb_complex_double_t *A, const int lda,
                                fb_complex_double_t *B, const int ldb,
                                double *s, double rcond, int *rank);

/* GESDD - SVD with divide-and-conquer */
typedef int (*fb_sgesdd_fn)(const fb_layout_t layout, char jobz,
                                const int m, const int n, float *A,
                                const int lda, float *s, float *U,
                                const int ldu, float *VT,
                                const int ldvt);
typedef int (*fb_dgesdd_fn)(const fb_layout_t layout, char jobz,
                                const int m, const int n, double *A,
                                const int lda, double *s, double *U,
                                const int ldu, double *VT,
                                const int ldvt);
typedef int (*fb_cgesdd_fn)(const fb_layout_t layout, char jobz,
                                const int m, const int n,
                                fb_complex_float_t *A, const int lda,
                                float *s, fb_complex_float_t *U,
                                const int ldu, fb_complex_float_t *VT,
                                const int ldvt);
typedef int (*fb_zgesdd_fn)(const fb_layout_t layout, char jobz,
                                const int m, const int n,
                                fb_complex_double_t *A, const int lda,
                                double *s, fb_complex_double_t *U,
                                const int ldu, fb_complex_double_t *VT,
                                const int ldvt);

/* SYGV - Generalized symmetric eigenvalue problem */
typedef int (*fb_ssygv_fn)(const fb_layout_t layout, const int itype,
                               char jobz, const fb_uplo_t uplo, const int n,
                               float *A, const int lda, float *B,
                               const int ldb, float *w);
typedef int (*fb_dsygv_fn)(const fb_layout_t layout, const int itype,
                               char jobz, const fb_uplo_t uplo, const int n,
                               double *A, const int lda, double *B,
                               const int ldb, double *w);

/* HEGV - Generalized hermitian eigenvalue problem */
typedef int (*fb_chegv_fn)(const fb_layout_t layout, const int itype,
                               char jobz, const fb_uplo_t uplo, const int n,
                               fb_complex_float_t *A, const int lda,
                               fb_complex_float_t *B, const int ldb,
                               float *w);
typedef int (*fb_zhegv_fn)(const fb_layout_t layout, const int itype,
                               char jobz, const fb_uplo_t uplo, const int n,
                               fb_complex_double_t *A, const int lda,
                               fb_complex_double_t *B, const int ldb,
                               double *w);

/* GELSY - Least squares with QR and pivoting */
typedef int (*fb_sgelsy_fn)(const fb_layout_t layout, const int m,
                                const int n, const int nrhs, float *A,
                                const int lda, float *B, const int ldb,
                                int *jpvt, float rcond, int *rank);
typedef int (*fb_dgelsy_fn)(const fb_layout_t layout, const int m,
                                const int n, const int nrhs, double *A,
                                const int lda, double *B, const int ldb,
                                int *jpvt, double rcond, int *rank);
typedef int (*fb_cgelsy_fn)(const fb_layout_t layout, const int m,
                                const int n, const int nrhs,
                                fb_complex_float_t *A, const int lda,
                                fb_complex_float_t *B, const int ldb,
                                int *jpvt, float rcond, int *rank);
typedef int (*fb_zgelsy_fn)(const fb_layout_t layout, const int m,
                                const int n, const int nrhs,
                                fb_complex_double_t *A, const int lda,
                                fb_complex_double_t *B, const int ldb,
                                int *jpvt, double rcond, int *rank);

/* ============================================================================
 * Backend Virtual Table
 *   252 named operation fields  (BLAS L1/L2/L3 + LAPACK core)
 * + 15  infrastructure fields  (unified, memory, streams, thread control)
 * + ext_ops[] array covering the full 2266-op superset (FB_JUDGE_MAX_OPERATIONS)
 * ========================================================================= */

typedef struct fb_backend_vtable {
  /* Backend metadata */
  fb_backend_info_t info;

  /* ===== BLAS Level 1 (52 operations) ===== */

  /* Rotation generators */
  fb_srotg_fn srotg;
  fb_drotg_fn drotg;
  fb_crotg_fn crotg;
  fb_zrotg_fn zrotg;
  fb_srotmg_fn srotmg;
  fb_drotmg_fn drotmg;

  /* Rotations */
  fb_srot_fn srot;
  fb_drot_fn drot;
  fb_crot_fn crot;
  fb_zrot_fn zrot;
  fb_zdrot_fn zdrot;
  fb_srotm_fn srotm;
  fb_drotm_fn drotm;

  /* Swaps */
  fb_sswap_fn sswap;
  fb_dswap_fn dswap;
  fb_cswap_fn cswap;
  fb_zswap_fn zswap;

  /* Scales */
  fb_sscal_fn sscal;
  fb_dscal_fn dscal;
  fb_cscal_fn cscal;
  fb_zscal_fn zscal;
  fb_csscal_fn csscal;
  fb_zdscal_fn zdscal;

  /* Copies */
  fb_scopy_fn scopy;
  fb_dcopy_fn dcopy;
  fb_ccopy_fn ccopy;
  fb_zcopy_fn zcopy;

  /* AXPY */
  fb_saxpy_fn saxpy;
  fb_daxpy_fn daxpy;
  fb_caxpy_fn caxpy;
  fb_zaxpy_fn zaxpy;

  /* Dot products */
  fb_sdot_fn sdot;
  fb_ddot_fn ddot;
  fb_sdsdot_fn sdsdot;
  fb_dsdot_fn dsdot;
  fb_cdotu_fn cdotu;
  fb_zdotu_fn zdotu;
  fb_cdotc_fn cdotc;
  fb_zdotc_fn zdotc;

  /* Norms */
  fb_snrm2_fn snrm2;
  fb_dnrm2_fn dnrm2;
  fb_scnrm2_fn scnrm2;
  fb_dznrm2_fn dznrm2;
  fb_sasum_fn sasum;
  fb_dasum_fn dasum;
  fb_scasum_fn scasum;
  fb_dzasum_fn dzasum;

  /* Index of maximum */
  fb_isamax_fn isamax;
  fb_idamax_fn idamax;
  fb_icamax_fn icamax;
  fb_izamax_fn izamax;

  /* ===== BLAS Level 2 (70 operations) ===== */

  /* GEMV */
  fb_sgemv_fn sgemv;
  fb_dgemv_fn dgemv;
  fb_cgemv_fn cgemv;
  fb_zgemv_fn zgemv;

  /* GBMV */
  fb_sgbmv_fn sgbmv;
  fb_dgbmv_fn dgbmv;
  fb_cgbmv_fn cgbmv;
  fb_zgbmv_fn zgbmv;

  /* HEMV */
  fb_chemv_fn chemv;
  fb_zhemv_fn zhemv;

  /* HBMV */
  fb_chbmv_fn chbmv;
  fb_zhbmv_fn zhbmv;

  /* HPMV */
  fb_chpmv_fn chpmv;
  fb_zhpmv_fn zhpmv;

  /* SYMV */
  fb_ssymv_fn ssymv;
  fb_dsymv_fn dsymv;
  fb_csymv_fn csymv;
  fb_zsymv_fn zsymv;

  /* SBMV */
  fb_ssbmv_fn ssbmv;
  fb_dsbmv_fn dsbmv;

  /* SPMV */
  fb_sspmv_fn sspmv;
  fb_dspmv_fn dspmv;

  /* TRMV */
  fb_strmv_fn strmv;
  fb_dtrmv_fn dtrmv;
  fb_ctrmv_fn ctrmv;
  fb_ztrmv_fn ztrmv;

  /* TBMV */
  fb_stbmv_fn stbmv;
  fb_dtbmv_fn dtbmv;
  fb_ctbmv_fn ctbmv;
  fb_ztbmv_fn ztbmv;

  /* TPMV */
  fb_stpmv_fn stpmv;
  fb_dtpmv_fn dtpmv;
  fb_ctpmv_fn ctpmv;
  fb_ztpmv_fn ztpmv;

  /* TRSV */
  fb_strsv_fn strsv;
  fb_dtrsv_fn dtrsv;
  fb_ctrsv_fn ctrsv;
  fb_ztrsv_fn ztrsv;

  /* TBSV */
  fb_stbsv_fn stbsv;
  fb_dtbsv_fn dtbsv;
  fb_ctbsv_fn ctbsv;
  fb_ztbsv_fn ztbsv;

  /* TPSV */
  fb_stpsv_fn stpsv;
  fb_dtpsv_fn dtpsv;
  fb_ctpsv_fn ctpsv;
  fb_ztpsv_fn ztpsv;

  /* GER */
  fb_sger_fn sger;
  fb_dger_fn dger;

  /* GERU/GERC */
  fb_cgeru_fn cgeru;
  fb_zgeru_fn zgeru;
  fb_cgerc_fn cgerc;
  fb_zgerc_fn zgerc;

  /* HER */
  fb_cher_fn cher;
  fb_zher_fn zher;

  /* HPR */
  fb_chpr_fn chpr;
  fb_zhpr_fn zhpr;

  /* HER2 */
  fb_cher2_fn cher2;
  fb_zher2_fn zher2;

  /* HPR2 */
  fb_chpr2_fn chpr2;
  fb_zhpr2_fn zhpr2;

  /* SYR */
  fb_ssyr_fn ssyr;
  fb_dsyr_fn dsyr;
  fb_csyr_fn csyr;
  fb_zsyr_fn zsyr;

  /* SPR */
  fb_sspr_fn sspr;
  fb_dspr_fn dspr;

  /* SYR2 */
  fb_ssyr2_fn ssyr2;
  fb_dsyr2_fn dsyr2;

  /* SPR2 */
  fb_sspr2_fn sspr2;
  fb_dspr2_fn dspr2;

  /* ===== BLAS Level 3 (30 operations) ===== */

  /* GEMM */
  fb_sgemm_fn sgemm;
  fb_dgemm_fn dgemm;
  fb_cgemm_fn cgemm;
  fb_zgemm_fn zgemm;

  /* SYMM */
  fb_ssymm_fn ssymm;
  fb_dsymm_fn dsymm;
  fb_csymm_fn csymm;
  fb_zsymm_fn zsymm;

  /* HEMM */
  fb_chemm_fn chemm;
  fb_zhemm_fn zhemm;

  /* SYRK */
  fb_ssyrk_fn ssyrk;
  fb_dsyrk_fn dsyrk;
  fb_csyrk_fn csyrk;
  fb_zsyrk_fn zsyrk;

  /* HERK */
  fb_cherk_fn cherk;
  fb_zherk_fn zherk;

  /* SYR2K */
  fb_ssyr2k_fn ssyr2k;
  fb_dsyr2k_fn dsyr2k;
  fb_csyr2k_fn csyr2k;
  fb_zsyr2k_fn zsyr2k;

  /* HER2K */
  fb_cher2k_fn cher2k;
  fb_zher2k_fn zher2k;

  /* TRMM */
  fb_strmm_fn strmm;
  fb_dtrmm_fn dtrmm;
  fb_ctrmm_fn ctrmm;
  fb_ztrmm_fn ztrmm;

  /* TRSM */
  fb_strsm_fn strsm;
  fb_dtrsm_fn dtrsm;
  fb_ctrsm_fn ctrsm;
  fb_ztrsm_fn ztrsm;

  /* ===== LAPACK Subset (60 operations) ===== */

  /* GESV */
  fb_sgesv_fn sgesv;
  fb_dgesv_fn dgesv;
  fb_cgesv_fn cgesv;
  fb_zgesv_fn zgesv;

  /* POSV */
  fb_sposv_fn sposv;
  fb_dposv_fn dposv;
  fb_cposv_fn cposv;
  fb_zposv_fn zposv;

  /* SYSV */
  fb_ssysv_fn ssysv;
  fb_dsysv_fn dsysv;
  fb_csysv_fn csysv;
  fb_zsysv_fn zsysv;

  /* HESV */
  fb_chesv_fn chesv;
  fb_zhesv_fn zhesv;

  /* GETRF */
  fb_sgetrf_fn sgetrf;
  fb_dgetrf_fn dgetrf;
  fb_cgetrf_fn cgetrf;
  fb_zgetrf_fn zgetrf;

  /* GETRS */
  fb_sgetrs_fn sgetrs;
  fb_dgetrs_fn dgetrs;
  fb_cgetrs_fn cgetrs;
  fb_zgetrs_fn zgetrs;

  /* POTRF */
  fb_spotrf_fn spotrf;
  fb_dpotrf_fn dpotrf;
  fb_cpotrf_fn cpotrf;
  fb_zpotrf_fn zpotrf;

  /* POTRS */
  fb_spotrs_fn spotrs;
  fb_dpotrs_fn dpotrs;
  fb_cpotrs_fn cpotrs;
  fb_zpotrs_fn zpotrs;

  /* GETRI */
  fb_sgetri_fn sgetri;
  fb_dgetri_fn dgetri;
  fb_cgetri_fn cgetri;
  fb_zgetri_fn zgetri;

  /* POTRI */
  fb_spotri_fn spotri;
  fb_dpotri_fn dpotri;
  fb_cpotri_fn cpotri;
  fb_zpotri_fn zpotri;

  /* TRTRI */
  fb_strtri_fn strtri;
  fb_dtrtri_fn dtrtri;
  fb_ctrtri_fn ctrtri;
  fb_ztrtri_fn ztrtri;

  /* GEEV */
  fb_sgeev_fn sgeev;
  fb_dgeev_fn dgeev;
  fb_cgeev_fn cgeev;
  fb_zgeev_fn zgeev;

  /* SYEV */
  fb_ssyev_fn ssyev;
  fb_dsyev_fn dsyev;

  /* HEEV */
  fb_cheev_fn cheev;
  fb_zheev_fn zheev;

  /* GESVD */
  fb_sgesvd_fn sgesvd;
  fb_dgesvd_fn dgesvd;
  fb_cgesvd_fn cgesvd;
  fb_zgesvd_fn zgesvd;

  /* GEQRF */
  fb_sgeqrf_fn sgeqrf;
  fb_dgeqrf_fn dgeqrf;
  fb_cgeqrf_fn cgeqrf;
  fb_zgeqrf_fn zgeqrf;

  /* ORGQR */
  fb_sorgqr_fn sorgqr;
  fb_dorgqr_fn dorgqr;

  /* UNGQR */
  fb_cungqr_fn cungqr;
  fb_zungqr_fn zungqr;

  /* GELS */
  fb_sgels_fn sgels;
  fb_dgels_fn dgels;
  fb_cgels_fn cgels;
  fb_zgels_fn zgels;

  /* ORMQR */
  fb_sormqr_fn sormqr;
  fb_dormqr_fn dormqr;

  /* UNMQR */
  fb_cunmqr_fn cunmqr;
  fb_zunmqr_fn zunmqr;

  /* GELSD */
  fb_sgelsd_fn sgelsd;
  fb_dgelsd_fn dgelsd;
  fb_cgelsd_fn cgelsd;
  fb_zgelsd_fn zgelsd;

  /* GESDD */
  fb_sgesdd_fn sgesdd;
  fb_dgesdd_fn dgesdd;
  fb_cgesdd_fn cgesdd;
  fb_zgesdd_fn zgesdd;

  /* SYGV */
  fb_ssygv_fn ssygv;
  fb_dsygv_fn dsygv;

  /* HEGV */
  fb_chegv_fn chegv;
  fb_zhegv_fn zhegv;

  /* GELSY */
  fb_sgelsy_fn sgelsy;
  fb_dgelsy_fn dgelsy;
  fb_cgelsy_fn cgelsy;
  fb_zgelsy_fn zgelsy;

  /* ========================================================================
   * UNIFIED OPERATIONS (Advanced Features)
   * ======================================================================== */

  /**
   * @brief Unified GEMM - covers all GEMM variants with single implementation
   * Replaces 82 separate GEMM operations (standard, batched, mixed-precision,
   * fused) See FASTER-BLASTER-OPERATIONS-SUPERSET.md Section 6.1 for details
   */
  fb_status_t (*gemm_unified)(
      fb_precision_t precision, fb_batch_mode_t batch_mode, fb_fusion_t fusion,
      const char transa, const char transb, const size_t m, const size_t n,
      const size_t k, const void *alpha, const void *A, const size_t lda,
      const size_t strideA, const void *B, const size_t ldb,
      const size_t strideB, const void *beta, void *C, const size_t ldc,
      const size_t strideC, const void *bias, fb_activation_t activation,
      const size_t batch_count, void *stream);

  /**
   * @brief Unified Normalization - covers all normalization variants
   * Replaces 17 separate normalization operations (batch/layer/instance/group
   * norm, z-score, scaling) See FASTER-BLASTER-OPERATIONS-SUPERSET.md
   * Section 6.2 for details
   */
  fb_status_t (*normalize_unified)(
      fb_norm_mode_t mode, fb_norm_phase_t phase, const size_t *dims,
      size_t ndims, const int *normalize_axes, size_t num_axes,
      const void *input, void *output, const void *scale, const void *bias,
      const void *running_mean, const void *running_var, float momentum,
      float epsilon, int num_groups, float scale_min, float scale_max,
      const void *grad_output, void *grad_input, void *grad_scale,
      void *grad_bias);

  /**
   * @brief Unified Reduction - covers all reduction/scan/collective variants
   * Replaces 24 separate reduction operations (tensor, parallel primitives,
   * collective communications) See FASTER-BLASTER-OPERATIONS-SUPERSET.md
   * Section 6.3 for details
   */
  fb_status_t (*reduce_unified)(
      fb_reduce_op_t operation, fb_reduce_scope_t scope, const void *input,
      void *output, const size_t *input_dims, size_t ndims,
      const int *reduce_axes, size_t num_axes, bool keepdims,
      fb_scan_mode_t scan_mode, const void *scan_init, fb_comm_t communicator,
      int root_rank, void *scratch_buffer, size_t scratch_size, void *stream);

  /* ========================================================================
   * UNIFIED BACKEND EXTENSIONS (CPU/GPU Interoperability)
   * ======================================================================== */

  /**
   * @brief Allocate backend-specific memory
   * CPU: Regular malloc or no-op if data already in RAM
   * GPU: cudaMalloc/hipMalloc/clCreateBuffer
   * @return 0 on success, non-zero on error
   */
  int (*mem_alloc)(void *backend_handle, void **ptr, size_t size);

  /**
   * @brief Free backend-specific memory
   * CPU: free() or no-op
   * GPU: cudaFree/hipFree/clReleaseMemObject
   */
  void (*mem_free)(void *backend_handle, void *ptr);

  /**
   * @brief Transfer data to device (host → device)
   * CPU: no-op (data already accessible in system RAM)
   * GPU: cudaMemcpy H2D/hipMemcpy H2D
   * @return 0 on success, non-zero on error
   */
  int (*mem_upload)(void *backend_handle, void *dst, const void *src,
                    size_t size);

  /**
   * @brief Transfer data from device (device → host)
   * CPU: no-op (data already accessible in system RAM)
   * GPU: cudaMemcpy D2H/hipMemcpy D2H
   * @return 0 on success, non-zero on error
   */
  int (*mem_download)(void *backend_handle, void *dst, const void *src,
                      size_t size);

  /**
   * @brief Copy within device memory
   * CPU: memcpy()
   * GPU: cudaMemcpy D2D/hipMemcpy D2D
   * @return 0 on success, non-zero on error
   */
  int (*mem_copy)(void *backend_handle, void *dst, const void *src,
                  size_t size);

  /**
   * @brief Create execution stream/queue
   * CPU: Allocate operation queue structure (for future CPU queuing)
   * GPU: cudaStreamCreate/hipStreamCreate/clCreateCommandQueue
   * @return 0 on success, non-zero on error
   */
  int (*stream_create)(void *backend_handle, void **stream);

  /**
   * @brief Destroy execution stream/queue
   * CPU: Free queue structure
   * GPU: cudaStreamDestroy/hipStreamDestroy
   */
  void (*stream_destroy)(void *backend_handle, void *stream);

  /**
   * @brief Synchronize stream (wait for all queued operations to complete)
   * CPU: Flush operation queue or no-op (ops are synchronous)
   * GPU: cudaStreamSynchronize/hipStreamSynchronize
   * @return 0 on success, non-zero on error
   */
  int (*stream_sync)(void *backend_handle, void *stream);

  /**
   * @brief Set active stream for subsequent operations
   * CPU: Set current queue for this thread
   * GPU: cublasSetStream/hipblasSetStream
   * @return 0 on success, non-zero on error
   */
  int (*stream_set)(void *backend_handle, void *stream);

  /**
   * @brief Query backend capabilities
   * @return Bitmask of FB_CAP_* flags
   */
  uint32_t (*get_capabilities)(void *backend_handle);

  /**
   * @brief Get number of threads used by backend (CPU only)
   * GPU backends should return 0 or -1
   * @return Thread count, or -1 if not applicable
   */
  int (*get_num_threads)(void *backend_handle);

  /**
   * @brief Set number of threads used by backend (CPU only)
   * GPU backends should ignore this
   */
  void (*set_num_threads)(void *backend_handle, int num_threads);

  /* =========================================================================
   * Extended operation dispatch table (full 2266-op superset)
   * =========================================================================
   *
   * Indexed by FB_OP_* constants from judge_op_ids.h.
   *
   * For the 252 named operations above, fb_vtable_sync_ext_ops() mirrors
   * each named field into the corresponding ext_ops[FB_OP_XXX] slot so that
   * callers can index any operation uniformly:
   *
   *   fb_generic_fn fn = vtable->ext_ops[op_id];           // O(1)
   *   if (fn) ((fb_saxpy_fn)fn)(n, alpha, x, incx, y, incy);
   *
   * For operations not covered by a named field (LAPACK supplement,
   * ScaLAPACK, DNN, FFT, Sparse, Tensor, etc.) backends populate
   * ext_ops[] directly before registration.
   *
   * NULL  →  operation not supported by this backend.
   */
  fb_generic_fn ext_ops[FB_JUDGE_MAX_OPERATIONS];

} fb_backend_vtable_t;

#ifdef __cplusplus
}
#endif

#endif /* FASTER_BLASTER_BACKEND_INTERFACE_H */
