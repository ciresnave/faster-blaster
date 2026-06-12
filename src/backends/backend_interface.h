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

/**
 * Calling convention tags used by the 2-D ext_ops table.
 *
 * FB_CONV_CBLAS   — cblas_saxpy(n, alpha, x, incx, y, incy)  (scalars by value)
 * FB_CONV_FORTRAN — saxpy_(&n, &alpha, x, &incx, y, &incy)   (everything by ptr, trailing _)
 *
 * CBLAS and vtable enum values are ABI-identical (FB_NO_TRANS==CBLAS_NO_TRANS==111),
 * so FB_CONV_CBLAS slots can be direct-cast from CBLAS function pointers without thunks.
 * Fortran↔CBLAS thunks are generated per-operation in src/core/conv_thunks.c.
 *
 * Note: FB_CONV_REF was removed.  Reference implementations export standard
 * cblas_* names; handle-scoped dlsym provides isolation between backends.
 */
typedef enum {
  FB_CONV_CBLAS   = 0, /* cblas_* prefix, scalars pass-by-value          */
  FB_CONV_FORTRAN = 1, /* trailing underscore, all args pass-by-pointer   */
  FB_CONV_COUNT   = 2
} fb_conv_t;

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
typedef void (*fb_cscal_fn)(const int n, const fb_complex_float_t *alpha,
                            fb_complex_float_t *x, const int incx);
typedef void (*fb_zscal_fn)(const int n, const fb_complex_double_t *alpha,
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
typedef void (*fb_caxpy_fn)(const int n, const fb_complex_float_t *alpha,
                            const fb_complex_float_t *x, const int incx,
                            fb_complex_float_t *y, const int incy);
typedef void (*fb_zaxpy_fn)(const int n, const fb_complex_double_t *alpha,
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
typedef int (*fb_ssysv_fn)(const fb_layout_t layout, char uplo, const int n,
                           const int nrhs, float *A, const int lda, int *ipiv,
                           float *B, const int ldb);
typedef int (*fb_dsysv_fn)(const fb_layout_t layout, char uplo, const int n,
                           const int nrhs, double *A, const int lda, int *ipiv,
                           double *B, const int ldb);
typedef int (*fb_csysv_fn)(const fb_layout_t layout, char uplo, const int n,
                           const int nrhs, fb_complex_float_t *A, const int lda,
                           int *ipiv, fb_complex_float_t *B, const int ldb);
typedef int (*fb_zsysv_fn)(const fb_layout_t layout, char uplo, const int n,
                           const int nrhs, fb_complex_double_t *A,
                           const int lda, int *ipiv, fb_complex_double_t *B,
                           const int ldb);

/* HESV - Hermitian indefinite linear system solve (complex only) */
typedef int (*fb_chesv_fn)(const fb_layout_t layout, char uplo, const int n,
                           const int nrhs, fb_complex_float_t *A, const int lda,
                           int *ipiv, fb_complex_float_t *B, const int ldb);
typedef int (*fb_zhesv_fn)(const fb_layout_t layout, char uplo, const int n,
                           const int nrhs, fb_complex_double_t *A,
                           const int lda, int *ipiv, fb_complex_double_t *B,
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
 * + ext_ops[] array covering the full 3054-op superset
 * (FB_JUDGE_MAX_OPERATIONS)
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

  fb_generic_fn sgbmvx;  /* FB_OP_SGBMVX = 53 */
  fb_generic_fn dgbmvx;  /* FB_OP_DGBMVX = 54 */
  fb_generic_fn cgbmvx;  /* FB_OP_CGBMVX = 55 */
  fb_generic_fn zgbmvx;  /* FB_OP_ZGBMVX = 56 */
  fb_generic_fn sgemm_batch;  /* FB_OP_SGEMM_BATCH = 123 */
  fb_generic_fn dgemm_batch;  /* FB_OP_DGEMM_BATCH = 124 */
  fb_generic_fn cgemm_batch;  /* FB_OP_CGEMM_BATCH = 125 */
  fb_generic_fn zgemm_batch;  /* FB_OP_ZGEMM_BATCH = 126 */
  fb_generic_fn sgemm_strided;  /* FB_OP_SGEMM_STRIDED = 127 */
  fb_generic_fn dgemm_strided;  /* FB_OP_DGEMM_STRIDED = 128 */
  fb_generic_fn cgemm_strided;  /* FB_OP_CGEMM_STRIDED = 129 */
  fb_generic_fn zgemm_strided;  /* FB_OP_ZGEMM_STRIDED = 130 */

  /* ========================================================================
   * LAPACK — standard s/d/c/z routines
   * ========================================================================
   * fb_generic_fn fields: backends assign  vtable->op = (fb_generic_fn)impl;
   * fb_vtable_sync_ext_ops() mirrors them into ext_ops[] automatically. */

  fb_generic_fn sbdsdc;  /* FB_OP_SBDSDC = 157 */
  fb_generic_fn dbdsdc;  /* FB_OP_DBDSDC = 158 */
  fb_generic_fn cbdsdc;  /* FB_OP_CBDSDC = 159 */
  fb_generic_fn zbdsdc;  /* FB_OP_ZBDSDC = 160 */
  fb_generic_fn sbdsqr;  /* FB_OP_SBDSQR = 161 */
  fb_generic_fn dbdsqr;  /* FB_OP_DBDSQR = 162 */
  fb_generic_fn cbdsqr;  /* FB_OP_CBDSQR = 163 */
  fb_generic_fn zbdsqr;  /* FB_OP_ZBDSQR = 164 */
  fb_generic_fn sbmv;  /* FB_OP_SBMV = 165 */
  fb_generic_fn dbmv;  /* FB_OP_DBMV = 166 */
  fb_generic_fn scsum1;  /* FB_OP_SCSUM1 = 167 */
  fb_generic_fn zdrscl;  /* FB_OP_ZDRSCL = 168 */
  fb_generic_fn sgbcon;  /* FB_OP_SGBCON = 169 */
  fb_generic_fn dgbcon;  /* FB_OP_DGBCON = 170 */
  fb_generic_fn cgbcon;  /* FB_OP_CGBCON = 171 */
  fb_generic_fn zgbcon;  /* FB_OP_ZGBCON = 172 */
  fb_generic_fn sgbequ;  /* FB_OP_SGBEQU = 173 */
  fb_generic_fn dgbequ;  /* FB_OP_DGBEQU = 174 */
  fb_generic_fn cgbequ;  /* FB_OP_CGBEQU = 175 */
  fb_generic_fn zgbequ;  /* FB_OP_ZGBEQU = 176 */
  fb_generic_fn sgbrfs;  /* FB_OP_SGBRFS = 177 */
  fb_generic_fn dgbrfs;  /* FB_OP_DGBRFS = 178 */
  fb_generic_fn cgbrfs;  /* FB_OP_CGBRFS = 179 */
  fb_generic_fn zgbrfs;  /* FB_OP_ZGBRFS = 180 */
  fb_generic_fn sgbsv;  /* FB_OP_SGBSV = 181 */
  fb_generic_fn dgbsv;  /* FB_OP_DGBSV = 182 */
  fb_generic_fn cgbsv;  /* FB_OP_CGBSV = 183 */
  fb_generic_fn zgbsv;  /* FB_OP_ZGBSV = 184 */
  fb_generic_fn sgbsvx;  /* FB_OP_SGBSVX = 185 */
  fb_generic_fn dgbsvx;  /* FB_OP_DGBSVX = 186 */
  fb_generic_fn cgbsvx;  /* FB_OP_CGBSVX = 187 */
  fb_generic_fn zgbsvx;  /* FB_OP_ZGBSVX = 188 */
  fb_generic_fn sgbtrf;  /* FB_OP_SGBTRF = 189 */
  fb_generic_fn dgbtrf;  /* FB_OP_DGBTRF = 190 */
  fb_generic_fn cgbtrf;  /* FB_OP_CGBTRF = 191 */
  fb_generic_fn zgbtrf;  /* FB_OP_ZGBTRF = 192 */
  fb_generic_fn sgbtrs;  /* FB_OP_SGBTRS = 193 */
  fb_generic_fn dgbtrs;  /* FB_OP_DGBTRS = 194 */
  fb_generic_fn cgbtrs;  /* FB_OP_CGBTRS = 195 */
  fb_generic_fn zgbtrs;  /* FB_OP_ZGBTRS = 196 */
  fb_generic_fn sgebak;  /* FB_OP_SGEBAK = 197 */
  fb_generic_fn dgebak;  /* FB_OP_DGEBAK = 198 */
  fb_generic_fn cgebak;  /* FB_OP_CGEBAK = 199 */
  fb_generic_fn zgebak;  /* FB_OP_ZGEBAK = 200 */
  fb_generic_fn sgebal;  /* FB_OP_SGEBAL = 201 */
  fb_generic_fn dgebal;  /* FB_OP_DGEBAL = 202 */
  fb_generic_fn cgebal;  /* FB_OP_CGEBAL = 203 */
  fb_generic_fn zgebal;  /* FB_OP_ZGEBAL = 204 */
  fb_generic_fn sgebd2;  /* FB_OP_SGEBD2 = 205 */
  fb_generic_fn dgebd2;  /* FB_OP_DGEBD2 = 206 */
  fb_generic_fn cgebd2;  /* FB_OP_CGEBD2 = 207 */
  fb_generic_fn zgebd2;  /* FB_OP_ZGEBD2 = 208 */
  fb_generic_fn sgebrd;  /* FB_OP_SGEBRD = 209 */
  fb_generic_fn dgebrd;  /* FB_OP_DGEBRD = 210 */
  fb_generic_fn cgebrd;  /* FB_OP_CGEBRD = 211 */
  fb_generic_fn zgebrd;  /* FB_OP_ZGEBRD = 212 */
  fb_generic_fn sgecon;  /* FB_OP_SGECON = 213 */
  fb_generic_fn dgecon;  /* FB_OP_DGECON = 214 */
  fb_generic_fn cgecon;  /* FB_OP_CGECON = 215 */
  fb_generic_fn zgecon;  /* FB_OP_ZGECON = 216 */
  fb_generic_fn sgeequ;  /* FB_OP_SGEEQU = 217 */
  fb_generic_fn dgeequ;  /* FB_OP_DGEEQU = 218 */
  fb_generic_fn cgeequ;  /* FB_OP_CGEEQU = 219 */
  fb_generic_fn zgeequ;  /* FB_OP_ZGEEQU = 220 */
  fb_generic_fn sgees;  /* FB_OP_SGEES = 221 */
  fb_generic_fn dgees;  /* FB_OP_DGEES = 222 */
  fb_generic_fn cgees;  /* FB_OP_CGEES = 223 */
  fb_generic_fn zgees;  /* FB_OP_ZGEES = 224 */
  fb_generic_fn sgeesx;  /* FB_OP_SGEESX = 225 */
  fb_generic_fn dgeesx;  /* FB_OP_DGEESX = 226 */
  fb_generic_fn cgeesx;  /* FB_OP_CGEESX = 227 */
  fb_generic_fn zgeesx;  /* FB_OP_ZGEESX = 228 */
  fb_generic_fn sgeevx;  /* FB_OP_SGEEVX = 233 */
  fb_generic_fn dgeevx;  /* FB_OP_DGEEVX = 234 */
  fb_generic_fn cgeevx;  /* FB_OP_CGEEVX = 235 */
  fb_generic_fn zgeevx;  /* FB_OP_ZGEEVX = 236 */
  fb_generic_fn sgehd2;  /* FB_OP_SGEHD2 = 237 */
  fb_generic_fn dgehd2;  /* FB_OP_DGEHD2 = 238 */
  fb_generic_fn cgehd2;  /* FB_OP_CGEHD2 = 239 */
  fb_generic_fn zgehd2;  /* FB_OP_ZGEHD2 = 240 */
  fb_generic_fn sgehrd;  /* FB_OP_SGEHRD = 241 */
  fb_generic_fn dgehrd;  /* FB_OP_DGEHRD = 242 */
  fb_generic_fn cgehrd;  /* FB_OP_CGEHRD = 243 */
  fb_generic_fn zgehrd;  /* FB_OP_ZGEHRD = 244 */
  fb_generic_fn sgelq2;  /* FB_OP_SGELQ2 = 245 */
  fb_generic_fn dgelq2;  /* FB_OP_DGELQ2 = 246 */
  fb_generic_fn cgelq2;  /* FB_OP_CGELQ2 = 247 */
  fb_generic_fn zgelq2;  /* FB_OP_ZGELQ2 = 248 */
  fb_generic_fn sgelqf;  /* FB_OP_SGELQF = 249 */
  fb_generic_fn dgelqf;  /* FB_OP_DGELQF = 250 */
  fb_generic_fn cgelqf;  /* FB_OP_CGELQF = 251 */
  fb_generic_fn zgelqf;  /* FB_OP_ZGELQF = 252 */
  fb_generic_fn sgelss;  /* FB_OP_SGELSS = 261 */
  fb_generic_fn dgelss;  /* FB_OP_DGELSS = 262 */
  fb_generic_fn cgelss;  /* FB_OP_CGELSS = 263 */
  fb_generic_fn zgelss;  /* FB_OP_ZGELSS = 264 */
  fb_generic_fn sgemmt;  /* FB_OP_SGEMMT = 269 */
  fb_generic_fn dgemmt;  /* FB_OP_DGEMMT = 270 */
  fb_generic_fn cgemmt;  /* FB_OP_CGEMMT = 271 */
  fb_generic_fn zgemmt;  /* FB_OP_ZGEMMT = 272 */
  fb_generic_fn sgemm_compute;  /* FB_OP_SGEMM_COMPUTE = 273 */
  fb_generic_fn dgemm_compute;  /* FB_OP_DGEMM_COMPUTE = 274 */
  fb_generic_fn cgemm_compute;  /* FB_OP_CGEMM_COMPUTE = 275 */
  fb_generic_fn zgemm_compute;  /* FB_OP_ZGEMM_COMPUTE = 276 */
  fb_generic_fn sgemm_pack;  /* FB_OP_SGEMM_PACK = 277 */
  fb_generic_fn dgemm_pack;  /* FB_OP_DGEMM_PACK = 278 */
  fb_generic_fn cgemm_pack;  /* FB_OP_CGEMM_PACK = 279 */
  fb_generic_fn zgemm_pack;  /* FB_OP_ZGEMM_PACK = 280 */
  fb_generic_fn sgemm_pack_get_size;  /* FB_OP_SGEMM_PACK_GET_SIZE = 281 */
  fb_generic_fn dgemm_pack_get_size;  /* FB_OP_DGEMM_PACK_GET_SIZE = 282 */
  fb_generic_fn cgemm_pack_get_size;  /* FB_OP_CGEMM_PACK_GET_SIZE = 283 */
  fb_generic_fn zgemm_pack_get_size;  /* FB_OP_ZGEMM_PACK_GET_SIZE = 284 */
  fb_generic_fn sgemm_ptr;  /* FB_OP_SGEMM_PTR = 285 */
  fb_generic_fn dgemm_ptr;  /* FB_OP_DGEMM_PTR = 286 */
  fb_generic_fn cgemm_ptr;  /* FB_OP_CGEMM_PTR = 287 */
  fb_generic_fn zgemm_ptr;  /* FB_OP_ZGEMM_PTR = 288 */
  fb_generic_fn sgeql2;  /* FB_OP_SGEQL2 = 289 */
  fb_generic_fn dgeql2;  /* FB_OP_DGEQL2 = 290 */
  fb_generic_fn cgeql2;  /* FB_OP_CGEQL2 = 291 */
  fb_generic_fn zgeql2;  /* FB_OP_ZGEQL2 = 292 */
  fb_generic_fn sgeqlf;  /* FB_OP_SGEQLF = 293 */
  fb_generic_fn dgeqlf;  /* FB_OP_DGEQLF = 294 */
  fb_generic_fn cgeqlf;  /* FB_OP_CGEQLF = 295 */
  fb_generic_fn zgeqlf;  /* FB_OP_ZGEQLF = 296 */
  fb_generic_fn sgeqp3;  /* FB_OP_SGEQP3 = 297 */
  fb_generic_fn dgeqp3;  /* FB_OP_DGEQP3 = 298 */
  fb_generic_fn cgeqp3;  /* FB_OP_CGEQP3 = 299 */
  fb_generic_fn zgeqp3;  /* FB_OP_ZGEQP3 = 300 */
  fb_generic_fn sgeqpf;  /* FB_OP_SGEQPF = 301 */
  fb_generic_fn dgeqpf;  /* FB_OP_DGEQPF = 302 */
  fb_generic_fn cgeqpf;  /* FB_OP_CGEQPF = 303 */
  fb_generic_fn zgeqpf;  /* FB_OP_ZGEQPF = 304 */
  fb_generic_fn sgerfs;  /* FB_OP_SGERFS = 309 */
  fb_generic_fn dgerfs;  /* FB_OP_DGERFS = 310 */
  fb_generic_fn cgerfs;  /* FB_OP_CGERFS = 311 */
  fb_generic_fn zgerfs;  /* FB_OP_ZGERFS = 312 */
  fb_generic_fn sgerq2;  /* FB_OP_SGERQ2 = 313 */
  fb_generic_fn dgerq2;  /* FB_OP_DGERQ2 = 314 */
  fb_generic_fn cgerq2;  /* FB_OP_CGERQ2 = 315 */
  fb_generic_fn zgerq2;  /* FB_OP_ZGERQ2 = 316 */
  fb_generic_fn sgerqf;  /* FB_OP_SGERQF = 317 */
  fb_generic_fn dgerqf;  /* FB_OP_DGERQF = 318 */
  fb_generic_fn cgerqf;  /* FB_OP_CGERQF = 319 */
  fb_generic_fn zgerqf;  /* FB_OP_ZGERQF = 320 */
  fb_generic_fn sgesvx;  /* FB_OP_SGESVX = 333 */
  fb_generic_fn dgesvx;  /* FB_OP_DGESVX = 334 */
  fb_generic_fn cgesvx;  /* FB_OP_CGESVX = 335 */
  fb_generic_fn zgesvx;  /* FB_OP_ZGESVX = 336 */
  fb_generic_fn sgesvxx;  /* FB_OP_SGESVXX = 337 */
  fb_generic_fn dgesvxx;  /* FB_OP_DGESVXX = 338 */
  fb_generic_fn cgesvxx;  /* FB_OP_CGESVXX = 339 */
  fb_generic_fn zgesvxx;  /* FB_OP_ZGESVXX = 340 */
  fb_generic_fn sgges;  /* FB_OP_SGGES = 353 */
  fb_generic_fn dgges;  /* FB_OP_DGGES = 354 */
  fb_generic_fn cgges;  /* FB_OP_CGGES = 355 */
  fb_generic_fn zgges;  /* FB_OP_ZGGES = 356 */
  fb_generic_fn sggesx;  /* FB_OP_SGGESX = 357 */
  fb_generic_fn dggesx;  /* FB_OP_DGGESX = 358 */
  fb_generic_fn cggesx;  /* FB_OP_CGGESX = 359 */
  fb_generic_fn zggesx;  /* FB_OP_ZGGESX = 360 */
  fb_generic_fn sggev;  /* FB_OP_SGGEV = 361 */
  fb_generic_fn dggev;  /* FB_OP_DGGEV = 362 */
  fb_generic_fn cggev;  /* FB_OP_CGGEV = 363 */
  fb_generic_fn zggev;  /* FB_OP_ZGGEV = 364 */
  fb_generic_fn sggevx;  /* FB_OP_SGGEVX = 365 */
  fb_generic_fn dggevx;  /* FB_OP_DGGEVX = 366 */
  fb_generic_fn cggevx;  /* FB_OP_CGGEVX = 367 */
  fb_generic_fn zggevx;  /* FB_OP_ZGGEVX = 368 */
  fb_generic_fn sggglm;  /* FB_OP_SGGGLM = 369 */
  fb_generic_fn dggglm;  /* FB_OP_DGGGLM = 370 */
  fb_generic_fn cggglm;  /* FB_OP_CGGGLM = 371 */
  fb_generic_fn zggglm;  /* FB_OP_ZGGGLM = 372 */
  fb_generic_fn sgghrd;  /* FB_OP_SGGHRD = 373 */
  fb_generic_fn dgghrd;  /* FB_OP_DGGHRD = 374 */
  fb_generic_fn cgghrd;  /* FB_OP_CGGHRD = 375 */
  fb_generic_fn zgghrd;  /* FB_OP_ZGGHRD = 376 */
  fb_generic_fn sgglse;  /* FB_OP_SGGLSE = 377 */
  fb_generic_fn dgglse;  /* FB_OP_DGGLSE = 378 */
  fb_generic_fn cgglse;  /* FB_OP_CGGLSE = 379 */
  fb_generic_fn zgglse;  /* FB_OP_ZGGLSE = 380 */
  fb_generic_fn sggsvd;  /* FB_OP_SGGSVD = 381 */
  fb_generic_fn dggsvd;  /* FB_OP_DGGSVD = 382 */
  fb_generic_fn cggsvd;  /* FB_OP_CGGSVD = 383 */
  fb_generic_fn zggsvd;  /* FB_OP_ZGGSVD = 384 */
  fb_generic_fn sgtcon;  /* FB_OP_SGTCON = 385 */
  fb_generic_fn dgtcon;  /* FB_OP_DGTCON = 386 */
  fb_generic_fn cgtcon;  /* FB_OP_CGTCON = 387 */
  fb_generic_fn zgtcon;  /* FB_OP_ZGTCON = 388 */
  fb_generic_fn sgtrfs;  /* FB_OP_SGTRFS = 389 */
  fb_generic_fn dgtrfs;  /* FB_OP_DGTRFS = 390 */
  fb_generic_fn cgtrfs;  /* FB_OP_CGTRFS = 391 */
  fb_generic_fn zgtrfs;  /* FB_OP_ZGTRFS = 392 */
  fb_generic_fn sgtsv;  /* FB_OP_SGTSV = 393 */
  fb_generic_fn dgtsv;  /* FB_OP_DGTSV = 394 */
  fb_generic_fn cgtsv;  /* FB_OP_CGTSV = 395 */
  fb_generic_fn zgtsv;  /* FB_OP_ZGTSV = 396 */
  fb_generic_fn sgtsvx;  /* FB_OP_SGTSVX = 397 */
  fb_generic_fn dgtsvx;  /* FB_OP_DGTSVX = 398 */
  fb_generic_fn cgtsvx;  /* FB_OP_CGTSVX = 399 */
  fb_generic_fn zgtsvx;  /* FB_OP_ZGTSVX = 400 */
  fb_generic_fn sgttrf;  /* FB_OP_SGTTRF = 401 */
  fb_generic_fn dgttrf;  /* FB_OP_DGTTRF = 402 */
  fb_generic_fn cgttrf;  /* FB_OP_CGTTRF = 403 */
  fb_generic_fn zgttrf;  /* FB_OP_ZGTTRF = 404 */
  fb_generic_fn sgttrs;  /* FB_OP_SGTTRS = 405 */
  fb_generic_fn dgttrs;  /* FB_OP_DGTTRS = 406 */
  fb_generic_fn cgttrs;  /* FB_OP_CGTTRS = 407 */
  fb_generic_fn zgttrs;  /* FB_OP_ZGTTRS = 408 */
  fb_generic_fn chbev;  /* FB_OP_CHBEV = 409 */
  fb_generic_fn zhbev;  /* FB_OP_ZHBEV = 410 */
  fb_generic_fn chbevd;  /* FB_OP_CHBEVD = 411 */
  fb_generic_fn zhbevd;  /* FB_OP_ZHBEVD = 412 */
  fb_generic_fn chbevx;  /* FB_OP_CHBEVX = 413 */
  fb_generic_fn zhbevx;  /* FB_OP_ZHBEVX = 414 */
  fb_generic_fn chbgv;  /* FB_OP_CHBGV = 415 */
  fb_generic_fn zhbgv;  /* FB_OP_ZHBGV = 416 */
  fb_generic_fn chbgvd;  /* FB_OP_CHBGVD = 417 */
  fb_generic_fn zhbgvd;  /* FB_OP_ZHBGVD = 418 */
  fb_generic_fn chbgvx;  /* FB_OP_CHBGVX = 419 */
  fb_generic_fn zhbgvx;  /* FB_OP_ZHBGVX = 420 */
  fb_generic_fn shbmv;  /* FB_OP_SHBMV = 421 */
  fb_generic_fn dhbmv;  /* FB_OP_DHBMV = 422 */
  fb_generic_fn chbtrf;  /* FB_OP_CHBTRF = 423 */
  fb_generic_fn zhbtrf;  /* FB_OP_ZHBTRF = 424 */
  fb_generic_fn checon;  /* FB_OP_CHECON = 425 */
  fb_generic_fn zhecon;  /* FB_OP_ZHECON = 426 */
  fb_generic_fn cheevd;  /* FB_OP_CHEEVD = 429 */
  fb_generic_fn zheevd;  /* FB_OP_ZHEEVD = 430 */
  fb_generic_fn cheevr;  /* FB_OP_CHEEVR = 431 */
  fb_generic_fn zheevr;  /* FB_OP_ZHEEVR = 432 */
  fb_generic_fn cheevx;  /* FB_OP_CHEEVX = 433 */
  fb_generic_fn zheevx;  /* FB_OP_ZHEEVX = 434 */
  fb_generic_fn chegst;  /* FB_OP_CHEGST = 435 */
  fb_generic_fn zhegst;  /* FB_OP_ZHEGST = 436 */
  fb_generic_fn chegvd;  /* FB_OP_CHEGVD = 439 */
  fb_generic_fn zhegvd;  /* FB_OP_ZHEGVD = 440 */
  fb_generic_fn chegvx;  /* FB_OP_CHEGVX = 441 */
  fb_generic_fn zhegvx;  /* FB_OP_ZHEGVX = 442 */
  fb_generic_fn sher;  /* FB_OP_SHER = 443 */
  fb_generic_fn dher;  /* FB_OP_DHER = 444 */
  fb_generic_fn sher2;  /* FB_OP_SHER2 = 445 */
  fb_generic_fn dher2;  /* FB_OP_DHER2 = 446 */
  fb_generic_fn cherfs;  /* FB_OP_CHERFS = 447 */
  fb_generic_fn zherfs;  /* FB_OP_ZHERFS = 448 */
  fb_generic_fn chesvx;  /* FB_OP_CHESVX = 451 */
  fb_generic_fn zhesvx;  /* FB_OP_ZHESVX = 452 */
  fb_generic_fn chesvxx;  /* FB_OP_CHESVXX = 453 */
  fb_generic_fn zhesvxx;  /* FB_OP_ZHESVXX = 454 */
  fb_generic_fn shetd2;  /* FB_OP_SHETD2 = 455 */
  fb_generic_fn dhetd2;  /* FB_OP_DHETD2 = 456 */
  fb_generic_fn chetrd;  /* FB_OP_CHETRD = 457 */
  fb_generic_fn zhetrd;  /* FB_OP_ZHETRD = 458 */
  fb_generic_fn chetrf;  /* FB_OP_CHETRF = 459 */
  fb_generic_fn zhetrf;  /* FB_OP_ZHETRF = 460 */
  fb_generic_fn chetrf_rook;  /* FB_OP_CHETRF_ROOK = 461 */
  fb_generic_fn zhetrf_rook;  /* FB_OP_ZHETRF_ROOK = 462 */
  fb_generic_fn chetri;  /* FB_OP_CHETRI = 463 */
  fb_generic_fn zhetri;  /* FB_OP_ZHETRI = 464 */
  fb_generic_fn chetrs;  /* FB_OP_CHETRS = 465 */
  fb_generic_fn zhetrs;  /* FB_OP_ZHETRS = 466 */
  fb_generic_fn shgeqz;  /* FB_OP_SHGEQZ = 467 */
  fb_generic_fn dhgeqz;  /* FB_OP_DHGEQZ = 468 */
  fb_generic_fn chgeqz;  /* FB_OP_CHGEQZ = 469 */
  fb_generic_fn zhgeqz;  /* FB_OP_ZHGEQZ = 470 */
  fb_generic_fn chpcon;  /* FB_OP_CHPCON = 471 */
  fb_generic_fn zhpcon;  /* FB_OP_ZHPCON = 472 */
  fb_generic_fn chpev;  /* FB_OP_CHPEV = 473 */
  fb_generic_fn zhpev;  /* FB_OP_ZHPEV = 474 */
  fb_generic_fn chpevd;  /* FB_OP_CHPEVD = 475 */
  fb_generic_fn zhpevd;  /* FB_OP_ZHPEVD = 476 */
  fb_generic_fn chpevx;  /* FB_OP_CHPEVX = 477 */
  fb_generic_fn zhpevx;  /* FB_OP_ZHPEVX = 478 */
  fb_generic_fn chpgv;  /* FB_OP_CHPGV = 479 */
  fb_generic_fn zhpgv;  /* FB_OP_ZHPGV = 480 */
  fb_generic_fn chpgvd;  /* FB_OP_CHPGVD = 481 */
  fb_generic_fn zhpgvd;  /* FB_OP_ZHPGVD = 482 */
  fb_generic_fn chpgvx;  /* FB_OP_CHPGVX = 483 */
  fb_generic_fn zhpgvx;  /* FB_OP_ZHPGVX = 484 */
  fb_generic_fn shpmv;  /* FB_OP_SHPMV = 485 */
  fb_generic_fn dhpmv;  /* FB_OP_DHPMV = 486 */
  fb_generic_fn chprfs;  /* FB_OP_CHPRFS = 487 */
  fb_generic_fn zhprfs;  /* FB_OP_ZHPRFS = 488 */
  fb_generic_fn chpsv;  /* FB_OP_CHPSV = 489 */
  fb_generic_fn zhpsv;  /* FB_OP_ZHPSV = 490 */
  fb_generic_fn chptrf;  /* FB_OP_CHPTRF = 491 */
  fb_generic_fn zhptrf;  /* FB_OP_ZHPTRF = 492 */
  fb_generic_fn chptri;  /* FB_OP_CHPTRI = 493 */
  fb_generic_fn zhptri;  /* FB_OP_ZHPTRI = 494 */
  fb_generic_fn chptrs;  /* FB_OP_CHPTRS = 495 */
  fb_generic_fn zhptrs;  /* FB_OP_ZHPTRS = 496 */
  fb_generic_fn shseqr;  /* FB_OP_SHSEQR = 497 */
  fb_generic_fn dhseqr;  /* FB_OP_DHSEQR = 498 */
  fb_generic_fn chseqr;  /* FB_OP_CHSEQR = 499 */
  fb_generic_fn zhseqr;  /* FB_OP_ZHSEQR = 500 */
  fb_generic_fn simatcopy;  /* FB_OP_SIMATCOPY = 501 */
  fb_generic_fn dimatcopy;  /* FB_OP_DIMATCOPY = 502 */
  fb_generic_fn cimatcopy;  /* FB_OP_CIMATCOPY = 503 */
  fb_generic_fn zimatcopy;  /* FB_OP_ZIMATCOPY = 504 */
  fb_generic_fn slabad;  /* FB_OP_SLABAD = 505 */
  fb_generic_fn dlabad;  /* FB_OP_DLABAD = 506 */
  fb_generic_fn slabrd;  /* FB_OP_SLABRD = 507 */
  fb_generic_fn dlabrd;  /* FB_OP_DLABRD = 508 */
  fb_generic_fn clabrd;  /* FB_OP_CLABRD = 509 */
  fb_generic_fn zlabrd;  /* FB_OP_ZLABRD = 510 */
  fb_generic_fn clacgv;  /* FB_OP_CLACGV = 511 */
  fb_generic_fn zlacgv;  /* FB_OP_ZLACGV = 512 */
  fb_generic_fn slacon;  /* FB_OP_SLACON = 513 */
  fb_generic_fn dlacon;  /* FB_OP_DLACON = 514 */
  fb_generic_fn clacon;  /* FB_OP_CLACON = 515 */
  fb_generic_fn zlacon;  /* FB_OP_ZLACON = 516 */
  fb_generic_fn slacpy;  /* FB_OP_SLACPY = 517 */
  fb_generic_fn dlacpy;  /* FB_OP_DLACPY = 518 */
  fb_generic_fn clacpy;  /* FB_OP_CLACPY = 519 */
  fb_generic_fn zlacpy;  /* FB_OP_ZLACPY = 520 */
  fb_generic_fn clacrm;  /* FB_OP_CLACRM = 521 */
  fb_generic_fn zlacrm;  /* FB_OP_ZLACRM = 522 */
  fb_generic_fn clacrt;  /* FB_OP_CLACRT = 523 */
  fb_generic_fn zlacrt;  /* FB_OP_ZLACRT = 524 */
  fb_generic_fn sladiv;  /* FB_OP_SLADIV = 525 */
  fb_generic_fn dladiv;  /* FB_OP_DLADIV = 526 */
  fb_generic_fn slaebz;  /* FB_OP_SLAEBZ = 527 */
  fb_generic_fn dlaebz;  /* FB_OP_DLAEBZ = 528 */
  fb_generic_fn slaed0;  /* FB_OP_SLAED0 = 529 */
  fb_generic_fn dlaed0;  /* FB_OP_DLAED0 = 530 */
  fb_generic_fn slaed1;  /* FB_OP_SLAED1 = 531 */
  fb_generic_fn dlaed1;  /* FB_OP_DLAED1 = 532 */
  fb_generic_fn slaed2;  /* FB_OP_SLAED2 = 533 */
  fb_generic_fn dlaed2;  /* FB_OP_DLAED2 = 534 */
  fb_generic_fn slaed3;  /* FB_OP_SLAED3 = 535 */
  fb_generic_fn dlaed3;  /* FB_OP_DLAED3 = 536 */
  fb_generic_fn slaed4;  /* FB_OP_SLAED4 = 537 */
  fb_generic_fn dlaed4;  /* FB_OP_DLAED4 = 538 */
  fb_generic_fn slaed5;  /* FB_OP_SLAED5 = 539 */
  fb_generic_fn dlaed5;  /* FB_OP_DLAED5 = 540 */
  fb_generic_fn slaed6;  /* FB_OP_SLAED6 = 541 */
  fb_generic_fn dlaed6;  /* FB_OP_DLAED6 = 542 */
  fb_generic_fn slaed7;  /* FB_OP_SLAED7 = 543 */
  fb_generic_fn dlaed7;  /* FB_OP_DLAED7 = 544 */
  fb_generic_fn slaed8;  /* FB_OP_SLAED8 = 545 */
  fb_generic_fn dlaed8;  /* FB_OP_DLAED8 = 546 */
  fb_generic_fn slaed9;  /* FB_OP_SLAED9 = 547 */
  fb_generic_fn dlaed9;  /* FB_OP_DLAED9 = 548 */
  fb_generic_fn slaeda;  /* FB_OP_SLAEDA = 549 */
  fb_generic_fn dlaeda;  /* FB_OP_DLAEDA = 550 */
  fb_generic_fn slaein;  /* FB_OP_SLAEIN = 551 */
  fb_generic_fn dlaein;  /* FB_OP_DLAEIN = 552 */
  fb_generic_fn claein;  /* FB_OP_CLAEIN = 553 */
  fb_generic_fn zlaein;  /* FB_OP_ZLAEIN = 554 */
  fb_generic_fn claesy;  /* FB_OP_CLAESY = 555 */
  fb_generic_fn zlaesy;  /* FB_OP_ZLAESY = 556 */
  fb_generic_fn slaev2;  /* FB_OP_SLAEV2 = 557 */
  fb_generic_fn dlaev2;  /* FB_OP_DLAEV2 = 558 */
  fb_generic_fn slaexc;  /* FB_OP_SLAEXC = 559 */
  fb_generic_fn dlaexc;  /* FB_OP_DLAEXC = 560 */
  fb_generic_fn claexc;  /* FB_OP_CLAEXC = 561 */
  fb_generic_fn zlaexc;  /* FB_OP_ZLAEXC = 562 */
  fb_generic_fn slags2;  /* FB_OP_SLAGS2 = 563 */
  fb_generic_fn dlags2;  /* FB_OP_DLAGS2 = 564 */
  fb_generic_fn clags2;  /* FB_OP_CLAGS2 = 565 */
  fb_generic_fn zlags2;  /* FB_OP_ZLAGS2 = 566 */
  fb_generic_fn slagtf;  /* FB_OP_SLAGTF = 567 */
  fb_generic_fn dlagtf;  /* FB_OP_DLAGTF = 568 */
  fb_generic_fn clagtf;  /* FB_OP_CLAGTF = 569 */
  fb_generic_fn zlagtf;  /* FB_OP_ZLAGTF = 570 */
  fb_generic_fn slagtm;  /* FB_OP_SLAGTM = 571 */
  fb_generic_fn dlagtm;  /* FB_OP_DLAGTM = 572 */
  fb_generic_fn clagtm;  /* FB_OP_CLAGTM = 573 */
  fb_generic_fn zlagtm;  /* FB_OP_ZLAGTM = 574 */
  fb_generic_fn slagts;  /* FB_OP_SLAGTS = 575 */
  fb_generic_fn dlagts;  /* FB_OP_DLAGTS = 576 */
  fb_generic_fn clagts;  /* FB_OP_CLAGTS = 577 */
  fb_generic_fn zlagts;  /* FB_OP_ZLAGTS = 578 */
  fb_generic_fn slagv2;  /* FB_OP_SLAGV2 = 579 */
  fb_generic_fn dlagv2;  /* FB_OP_DLAGV2 = 580 */
  fb_generic_fn clagv2;  /* FB_OP_CLAGV2 = 581 */
  fb_generic_fn zlagv2;  /* FB_OP_ZLAGV2 = 582 */
  fb_generic_fn slahqr;  /* FB_OP_SLAHQR = 583 */
  fb_generic_fn dlahqr;  /* FB_OP_DLAHQR = 584 */
  fb_generic_fn clahqr;  /* FB_OP_CLAHQR = 585 */
  fb_generic_fn zlahqr;  /* FB_OP_ZLAHQR = 586 */
  fb_generic_fn slahrd;  /* FB_OP_SLAHRD = 587 */
  fb_generic_fn dlahrd;  /* FB_OP_DLAHRD = 588 */
  fb_generic_fn clahrd;  /* FB_OP_CLAHRD = 589 */
  fb_generic_fn zlahrd;  /* FB_OP_ZLAHRD = 590 */
  fb_generic_fn slaic1;  /* FB_OP_SLAIC1 = 591 */
  fb_generic_fn dlaic1;  /* FB_OP_DLAIC1 = 592 */
  fb_generic_fn claic1;  /* FB_OP_CLAIC1 = 593 */
  fb_generic_fn zlaic1;  /* FB_OP_ZLAIC1 = 594 */
  fb_generic_fn slaln2;  /* FB_OP_SLALN2 = 595 */
  fb_generic_fn dlaln2;  /* FB_OP_DLALN2 = 596 */
  fb_generic_fn slamch;  /* FB_OP_SLAMCH = 597 */
  fb_generic_fn dlamch;  /* FB_OP_DLAMCH = 598 */
  fb_generic_fn slangb;  /* FB_OP_SLANGB = 599 */
  fb_generic_fn dlangb;  /* FB_OP_DLANGB = 600 */
  fb_generic_fn clangb;  /* FB_OP_CLANGB = 601 */
  fb_generic_fn zlangb;  /* FB_OP_ZLANGB = 602 */
  fb_generic_fn slange;  /* FB_OP_SLANGE = 603 */
  fb_generic_fn dlange;  /* FB_OP_DLANGE = 604 */
  fb_generic_fn clange;  /* FB_OP_CLANGE = 605 */
  fb_generic_fn zlange;  /* FB_OP_ZLANGE = 606 */
  fb_generic_fn slangt;  /* FB_OP_SLANGT = 607 */
  fb_generic_fn dlangt;  /* FB_OP_DLANGT = 608 */
  fb_generic_fn clangt;  /* FB_OP_CLANGT = 609 */
  fb_generic_fn zlangt;  /* FB_OP_ZLANGT = 610 */
  fb_generic_fn clanhb;  /* FB_OP_CLANHB = 611 */
  fb_generic_fn zlanhb;  /* FB_OP_ZLANHB = 612 */
  fb_generic_fn clanhe;  /* FB_OP_CLANHE = 613 */
  fb_generic_fn zlanhe;  /* FB_OP_ZLANHE = 614 */
  fb_generic_fn clanhp;  /* FB_OP_CLANHP = 615 */
  fb_generic_fn zlanhp;  /* FB_OP_ZLANHP = 616 */
  fb_generic_fn slanhs;  /* FB_OP_SLANHS = 617 */
  fb_generic_fn dlanhs;  /* FB_OP_DLANHS = 618 */
  fb_generic_fn clanhs;  /* FB_OP_CLANHS = 619 */
  fb_generic_fn zlanhs;  /* FB_OP_ZLANHS = 620 */
  fb_generic_fn slansb;  /* FB_OP_SLANSB = 621 */
  fb_generic_fn dlansb;  /* FB_OP_DLANSB = 622 */
  fb_generic_fn clansb;  /* FB_OP_CLANSB = 623 */
  fb_generic_fn zlansb;  /* FB_OP_ZLANSB = 624 */
  fb_generic_fn slansp;  /* FB_OP_SLANSP = 625 */
  fb_generic_fn dlansp;  /* FB_OP_DLANSP = 626 */
  fb_generic_fn clansp;  /* FB_OP_CLANSP = 627 */
  fb_generic_fn zlansp;  /* FB_OP_ZLANSP = 628 */
  fb_generic_fn slanst;  /* FB_OP_SLANST = 629 */
  fb_generic_fn dlanst;  /* FB_OP_DLANST = 630 */
  fb_generic_fn slansy;  /* FB_OP_SLANSY = 631 */
  fb_generic_fn dlansy;  /* FB_OP_DLANSY = 632 */
  fb_generic_fn clansy;  /* FB_OP_CLANSY = 633 */
  fb_generic_fn zlansy;  /* FB_OP_ZLANSY = 634 */
  fb_generic_fn slantb;  /* FB_OP_SLANTB = 635 */
  fb_generic_fn dlantb;  /* FB_OP_DLANTB = 636 */
  fb_generic_fn clantb;  /* FB_OP_CLANTB = 637 */
  fb_generic_fn zlantb;  /* FB_OP_ZLANTB = 638 */
  fb_generic_fn slantp;  /* FB_OP_SLANTP = 639 */
  fb_generic_fn dlantp;  /* FB_OP_DLANTP = 640 */
  fb_generic_fn clantp;  /* FB_OP_CLANTP = 641 */
  fb_generic_fn zlantp;  /* FB_OP_ZLANTP = 642 */
  fb_generic_fn slantr;  /* FB_OP_SLANTR = 643 */
  fb_generic_fn dlantr;  /* FB_OP_DLANTR = 644 */
  fb_generic_fn clantr;  /* FB_OP_CLANTR = 645 */
  fb_generic_fn zlantr;  /* FB_OP_ZLANTR = 646 */
  fb_generic_fn slapmt;  /* FB_OP_SLAPMT = 647 */
  fb_generic_fn dlapmt;  /* FB_OP_DLAPMT = 648 */
  fb_generic_fn clapmt;  /* FB_OP_CLAPMT = 649 */
  fb_generic_fn zlapmt;  /* FB_OP_ZLAPMT = 650 */
  fb_generic_fn slapy2;  /* FB_OP_SLAPY2 = 651 */
  fb_generic_fn dlapy2;  /* FB_OP_DLAPY2 = 652 */
  fb_generic_fn slapy3;  /* FB_OP_SLAPY3 = 653 */
  fb_generic_fn dlapy3;  /* FB_OP_DLAPY3 = 654 */
  fb_generic_fn slaqgb;  /* FB_OP_SLAQGB = 655 */
  fb_generic_fn dlaqgb;  /* FB_OP_DLAQGB = 656 */
  fb_generic_fn claqgb;  /* FB_OP_CLAQGB = 657 */
  fb_generic_fn zlaqgb;  /* FB_OP_ZLAQGB = 658 */
  fb_generic_fn slaqge;  /* FB_OP_SLAQGE = 659 */
  fb_generic_fn dlaqge;  /* FB_OP_DLAQGE = 660 */
  fb_generic_fn claqge;  /* FB_OP_CLAQGE = 661 */
  fb_generic_fn zlaqge;  /* FB_OP_ZLAQGE = 662 */
  fb_generic_fn claqhe;  /* FB_OP_CLAQHE = 663 */
  fb_generic_fn zlaqhe;  /* FB_OP_ZLAQHE = 664 */
  fb_generic_fn claqhp;  /* FB_OP_CLAQHP = 665 */
  fb_generic_fn zlaqhp;  /* FB_OP_ZLAQHP = 666 */
  fb_generic_fn slaqp2;  /* FB_OP_SLAQP2 = 667 */
  fb_generic_fn dlaqp2;  /* FB_OP_DLAQP2 = 668 */
  fb_generic_fn claqp2;  /* FB_OP_CLAQP2 = 669 */
  fb_generic_fn zlaqp2;  /* FB_OP_ZLAQP2 = 670 */
  fb_generic_fn slaqps;  /* FB_OP_SLAQPS = 671 */
  fb_generic_fn dlaqps;  /* FB_OP_DLAQPS = 672 */
  fb_generic_fn claqps;  /* FB_OP_CLAQPS = 673 */
  fb_generic_fn zlaqps;  /* FB_OP_ZLAQPS = 674 */
  fb_generic_fn slaqr0;  /* FB_OP_SLAQR0 = 675 */
  fb_generic_fn dlaqr0;  /* FB_OP_DLAQR0 = 676 */
  fb_generic_fn slaqr1;  /* FB_OP_SLAQR1 = 677 */
  fb_generic_fn dlaqr1;  /* FB_OP_DLAQR1 = 678 */
  fb_generic_fn slaqr2;  /* FB_OP_SLAQR2 = 679 */
  fb_generic_fn dlaqr2;  /* FB_OP_DLAQR2 = 680 */
  fb_generic_fn slaqsb;  /* FB_OP_SLAQSB = 681 */
  fb_generic_fn dlaqsb;  /* FB_OP_DLAQSB = 682 */
  fb_generic_fn claqsb;  /* FB_OP_CLAQSB = 683 */
  fb_generic_fn zlaqsb;  /* FB_OP_ZLAQSB = 684 */
  fb_generic_fn slaqsp;  /* FB_OP_SLAQSP = 685 */
  fb_generic_fn dlaqsp;  /* FB_OP_DLAQSP = 686 */
  fb_generic_fn claqsp;  /* FB_OP_CLAQSP = 687 */
  fb_generic_fn zlaqsp;  /* FB_OP_ZLAQSP = 688 */
  fb_generic_fn slaqsy;  /* FB_OP_SLAQSY = 689 */
  fb_generic_fn dlaqsy;  /* FB_OP_DLAQSY = 690 */
  fb_generic_fn claqsy;  /* FB_OP_CLAQSY = 691 */
  fb_generic_fn zlaqsy;  /* FB_OP_ZLAQSY = 692 */
  fb_generic_fn slar1v;  /* FB_OP_SLAR1V = 693 */
  fb_generic_fn dlar1v;  /* FB_OP_DLAR1V = 694 */
  fb_generic_fn slar2v;  /* FB_OP_SLAR2V = 695 */
  fb_generic_fn dlar2v;  /* FB_OP_DLAR2V = 696 */
  fb_generic_fn slarf;  /* FB_OP_SLARF = 697 */
  fb_generic_fn dlarf;  /* FB_OP_DLARF = 698 */
  fb_generic_fn clarf;  /* FB_OP_CLARF = 699 */
  fb_generic_fn zlarf;  /* FB_OP_ZLARF = 700 */
  fb_generic_fn slarfb;  /* FB_OP_SLARFB = 701 */
  fb_generic_fn dlarfb;  /* FB_OP_DLARFB = 702 */
  fb_generic_fn clarfb;  /* FB_OP_CLARFB = 703 */
  fb_generic_fn zlarfb;  /* FB_OP_ZLARFB = 704 */
  fb_generic_fn slarfg;  /* FB_OP_SLARFG = 705 */
  fb_generic_fn dlarfg;  /* FB_OP_DLARFG = 706 */
  fb_generic_fn clarfg;  /* FB_OP_CLARFG = 707 */
  fb_generic_fn zlarfg;  /* FB_OP_ZLARFG = 708 */
  fb_generic_fn slarft;  /* FB_OP_SLARFT = 709 */
  fb_generic_fn dlarft;  /* FB_OP_DLARFT = 710 */
  fb_generic_fn clarft;  /* FB_OP_CLARFT = 711 */
  fb_generic_fn zlarft;  /* FB_OP_ZLARFT = 712 */
  fb_generic_fn slarfx;  /* FB_OP_SLARFX = 713 */
  fb_generic_fn dlarfx;  /* FB_OP_DLARFX = 714 */
  fb_generic_fn clarfx;  /* FB_OP_CLARFX = 715 */
  fb_generic_fn zlarfx;  /* FB_OP_ZLARFX = 716 */
  fb_generic_fn slarge;  /* FB_OP_SLARGE = 717 */
  fb_generic_fn dlarge;  /* FB_OP_DLARGE = 718 */
  fb_generic_fn clargv;  /* FB_OP_CLARGV = 719 */
  fb_generic_fn zlargv;  /* FB_OP_ZLARGV = 720 */
  fb_generic_fn slarnv;  /* FB_OP_SLARNV = 721 */
  fb_generic_fn dlarnv;  /* FB_OP_DLARNV = 722 */
  fb_generic_fn clarnv;  /* FB_OP_CLARNV = 723 */
  fb_generic_fn zlarnv;  /* FB_OP_ZLARNV = 724 */
  fb_generic_fn slarra;  /* FB_OP_SLARRA = 725 */
  fb_generic_fn dlarra;  /* FB_OP_DLARRA = 726 */
  fb_generic_fn slarrb;  /* FB_OP_SLARRB = 727 */
  fb_generic_fn dlarrb;  /* FB_OP_DLARRB = 728 */
  fb_generic_fn slarrc;  /* FB_OP_SLARRC = 729 */
  fb_generic_fn dlarrc;  /* FB_OP_DLARRC = 730 */
  fb_generic_fn slarrd;  /* FB_OP_SLARRD = 731 */
  fb_generic_fn dlarrd;  /* FB_OP_DLARRD = 732 */
  fb_generic_fn slarre;  /* FB_OP_SLARRE = 733 */
  fb_generic_fn dlarre;  /* FB_OP_DLARRE = 734 */
  fb_generic_fn slarrf;  /* FB_OP_SLARRF = 735 */
  fb_generic_fn dlarrf;  /* FB_OP_DLARRF = 736 */
  fb_generic_fn slarrj;  /* FB_OP_SLARRJ = 737 */
  fb_generic_fn dlarrj;  /* FB_OP_DLARRJ = 738 */
  fb_generic_fn slarrk;  /* FB_OP_SLARRK = 739 */
  fb_generic_fn dlarrk;  /* FB_OP_DLARRK = 740 */
  fb_generic_fn slarrr;  /* FB_OP_SLARRR = 741 */
  fb_generic_fn dlarrr;  /* FB_OP_DLARRR = 742 */
  fb_generic_fn slarrv;  /* FB_OP_SLARRV = 743 */
  fb_generic_fn dlarrv;  /* FB_OP_DLARRV = 744 */
  fb_generic_fn slartg;  /* FB_OP_SLARTG = 745 */
  fb_generic_fn dlartg;  /* FB_OP_DLARTG = 746 */
  fb_generic_fn clartg;  /* FB_OP_CLARTG = 747 */
  fb_generic_fn zlartg;  /* FB_OP_ZLARTG = 748 */
  fb_generic_fn slartv;  /* FB_OP_SLARTV = 749 */
  fb_generic_fn dlartv;  /* FB_OP_DLARTV = 750 */
  fb_generic_fn clartv;  /* FB_OP_CLARTV = 751 */
  fb_generic_fn zlartv;  /* FB_OP_ZLARTV = 752 */
  fb_generic_fn slarz;  /* FB_OP_SLARZ = 753 */
  fb_generic_fn dlarz;  /* FB_OP_DLARZ = 754 */
  fb_generic_fn clarz;  /* FB_OP_CLARZ = 755 */
  fb_generic_fn zlarz;  /* FB_OP_ZLARZ = 756 */
  fb_generic_fn slarzb;  /* FB_OP_SLARZB = 757 */
  fb_generic_fn dlarzb;  /* FB_OP_DLARZB = 758 */
  fb_generic_fn clarzb;  /* FB_OP_CLARZB = 759 */
  fb_generic_fn zlarzb;  /* FB_OP_ZLARZB = 760 */
  fb_generic_fn slarzt;  /* FB_OP_SLARZT = 761 */
  fb_generic_fn dlarzt;  /* FB_OP_DLARZT = 762 */
  fb_generic_fn clarzt;  /* FB_OP_CLARZT = 763 */
  fb_generic_fn zlarzt;  /* FB_OP_ZLARZT = 764 */
  fb_generic_fn slas2;  /* FB_OP_SLAS2 = 765 */
  fb_generic_fn dlas2;  /* FB_OP_DLAS2 = 766 */
  fb_generic_fn slascl;  /* FB_OP_SLASCL = 767 */
  fb_generic_fn dlascl;  /* FB_OP_DLASCL = 768 */
  fb_generic_fn clascl;  /* FB_OP_CLASCL = 769 */
  fb_generic_fn zlascl;  /* FB_OP_ZLASCL = 770 */
  fb_generic_fn slasd0;  /* FB_OP_SLASD0 = 771 */
  fb_generic_fn dlasd0;  /* FB_OP_DLASD0 = 772 */
  fb_generic_fn slasd1;  /* FB_OP_SLASD1 = 773 */
  fb_generic_fn dlasd1;  /* FB_OP_DLASD1 = 774 */
  fb_generic_fn slasd2;  /* FB_OP_SLASD2 = 775 */
  fb_generic_fn dlasd2;  /* FB_OP_DLASD2 = 776 */
  fb_generic_fn slasd3;  /* FB_OP_SLASD3 = 777 */
  fb_generic_fn dlasd3;  /* FB_OP_DLASD3 = 778 */
  fb_generic_fn slasd4;  /* FB_OP_SLASD4 = 779 */
  fb_generic_fn dlasd4;  /* FB_OP_DLASD4 = 780 */
  fb_generic_fn slasd5;  /* FB_OP_SLASD5 = 781 */
  fb_generic_fn dlasd5;  /* FB_OP_DLASD5 = 782 */
  fb_generic_fn slasd6;  /* FB_OP_SLASD6 = 783 */
  fb_generic_fn dlasd6;  /* FB_OP_DLASD6 = 784 */
  fb_generic_fn slasd7;  /* FB_OP_SLASD7 = 785 */
  fb_generic_fn dlasd7;  /* FB_OP_DLASD7 = 786 */
  fb_generic_fn slasd8;  /* FB_OP_SLASD8 = 787 */
  fb_generic_fn dlasd8;  /* FB_OP_DLASD8 = 788 */
  fb_generic_fn slasda;  /* FB_OP_SLASDA = 789 */
  fb_generic_fn dlasda;  /* FB_OP_DLASDA = 790 */
  fb_generic_fn slasdq;  /* FB_OP_SLASDQ = 791 */
  fb_generic_fn dlasdq;  /* FB_OP_DLASDQ = 792 */
  fb_generic_fn slasdt;  /* FB_OP_SLASDT = 793 */
  fb_generic_fn dlasdt;  /* FB_OP_DLASDT = 794 */
  fb_generic_fn slaset;  /* FB_OP_SLASET = 795 */
  fb_generic_fn dlaset;  /* FB_OP_DLASET = 796 */
  fb_generic_fn claset;  /* FB_OP_CLASET = 797 */
  fb_generic_fn zlaset;  /* FB_OP_ZLASET = 798 */
  fb_generic_fn slasq1;  /* FB_OP_SLASQ1 = 799 */
  fb_generic_fn dlasq1;  /* FB_OP_DLASQ1 = 800 */
  fb_generic_fn slasq2;  /* FB_OP_SLASQ2 = 801 */
  fb_generic_fn dlasq2;  /* FB_OP_DLASQ2 = 802 */
  fb_generic_fn slasq3;  /* FB_OP_SLASQ3 = 803 */
  fb_generic_fn dlasq3;  /* FB_OP_DLASQ3 = 804 */
  fb_generic_fn slasq4;  /* FB_OP_SLASQ4 = 805 */
  fb_generic_fn dlasq4;  /* FB_OP_DLASQ4 = 806 */
  fb_generic_fn slasq5;  /* FB_OP_SLASQ5 = 807 */
  fb_generic_fn dlasq5;  /* FB_OP_DLASQ5 = 808 */
  fb_generic_fn slasq6;  /* FB_OP_SLASQ6 = 809 */
  fb_generic_fn dlasq6;  /* FB_OP_DLASQ6 = 810 */
  fb_generic_fn slasr;  /* FB_OP_SLASR = 811 */
  fb_generic_fn dlasr;  /* FB_OP_DLASR = 812 */
  fb_generic_fn slasrt;  /* FB_OP_SLASRT = 813 */
  fb_generic_fn dlasrt;  /* FB_OP_DLASRT = 814 */
  fb_generic_fn slassq;  /* FB_OP_SLASSQ = 815 */
  fb_generic_fn dlassq;  /* FB_OP_DLASSQ = 816 */
  fb_generic_fn classq;  /* FB_OP_CLASSQ = 817 */
  fb_generic_fn zlassq;  /* FB_OP_ZLASSQ = 818 */
  fb_generic_fn slasv2;  /* FB_OP_SLASV2 = 819 */
  fb_generic_fn dlasv2;  /* FB_OP_DLASV2 = 820 */
  fb_generic_fn slaswp;  /* FB_OP_SLASWP = 821 */
  fb_generic_fn dlaswp;  /* FB_OP_DLASWP = 822 */
  fb_generic_fn claswp;  /* FB_OP_CLASWP = 823 */
  fb_generic_fn zlaswp;  /* FB_OP_ZLASWP = 824 */
  fb_generic_fn slatbs;  /* FB_OP_SLATBS = 825 */
  fb_generic_fn dlatbs;  /* FB_OP_DLATBS = 826 */
  fb_generic_fn clatbs;  /* FB_OP_CLATBS = 827 */
  fb_generic_fn zlatbs;  /* FB_OP_ZLATBS = 828 */
  fb_generic_fn slatdf;  /* FB_OP_SLATDF = 829 */
  fb_generic_fn dlatdf;  /* FB_OP_DLATDF = 830 */
  fb_generic_fn clatdf;  /* FB_OP_CLATDF = 831 */
  fb_generic_fn zlatdf;  /* FB_OP_ZLATDF = 832 */
  fb_generic_fn slatps;  /* FB_OP_SLATPS = 833 */
  fb_generic_fn dlatps;  /* FB_OP_DLATPS = 834 */
  fb_generic_fn clatps;  /* FB_OP_CLATPS = 835 */
  fb_generic_fn zlatps;  /* FB_OP_ZLATPS = 836 */
  fb_generic_fn slatrd;  /* FB_OP_SLATRD = 837 */
  fb_generic_fn dlatrd;  /* FB_OP_DLATRD = 838 */
  fb_generic_fn clatrd;  /* FB_OP_CLATRD = 839 */
  fb_generic_fn zlatrd;  /* FB_OP_ZLATRD = 840 */
  fb_generic_fn slatrs;  /* FB_OP_SLATRS = 841 */
  fb_generic_fn dlatrs;  /* FB_OP_DLATRS = 842 */
  fb_generic_fn clatrs;  /* FB_OP_CLATRS = 843 */
  fb_generic_fn zlatrs;  /* FB_OP_ZLATRS = 844 */
  fb_generic_fn slatrz;  /* FB_OP_SLATRZ = 845 */
  fb_generic_fn dlatrz;  /* FB_OP_DLATRZ = 846 */
  fb_generic_fn clatrz;  /* FB_OP_CLATRZ = 847 */
  fb_generic_fn zlatrz;  /* FB_OP_ZLATRZ = 848 */
  fb_generic_fn slauu2;  /* FB_OP_SLAUU2 = 849 */
  fb_generic_fn dlauu2;  /* FB_OP_DLAUU2 = 850 */
  fb_generic_fn clauu2;  /* FB_OP_CLAUU2 = 851 */
  fb_generic_fn zlauu2;  /* FB_OP_ZLAUU2 = 852 */
  fb_generic_fn slauum;  /* FB_OP_SLAUUM = 853 */
  fb_generic_fn dlauum;  /* FB_OP_DLAUUM = 854 */
  fb_generic_fn clauum;  /* FB_OP_CLAUUM = 855 */
  fb_generic_fn zlauum;  /* FB_OP_ZLAUUM = 856 */
  fb_generic_fn sla_gbamv;  /* FB_OP_SLA_GBAMV = 857 */
  fb_generic_fn dla_gbamv;  /* FB_OP_DLA_GBAMV = 858 */
  fb_generic_fn cla_gbamv;  /* FB_OP_CLA_GBAMV = 859 */
  fb_generic_fn zla_gbamv;  /* FB_OP_ZLA_GBAMV = 860 */
  fb_generic_fn cnrm2;  /* FB_OP_CNRM2 = 861 */
  fb_generic_fn znrm2;  /* FB_OP_ZNRM2 = 862 */
  fb_generic_fn somatadd;  /* FB_OP_SOMATADD = 863 */
  fb_generic_fn domatadd;  /* FB_OP_DOMATADD = 864 */
  fb_generic_fn comatadd;  /* FB_OP_COMATADD = 865 */
  fb_generic_fn zomatadd;  /* FB_OP_ZOMATADD = 866 */
  fb_generic_fn somatcopy;  /* FB_OP_SOMATCOPY = 867 */
  fb_generic_fn domatcopy;  /* FB_OP_DOMATCOPY = 868 */
  fb_generic_fn comatcopy;  /* FB_OP_COMATCOPY = 869 */
  fb_generic_fn zomatcopy;  /* FB_OP_ZOMATCOPY = 870 */
  fb_generic_fn somatcopy2;  /* FB_OP_SOMATCOPY2 = 871 */
  fb_generic_fn domatcopy2;  /* FB_OP_DOMATCOPY2 = 872 */
  fb_generic_fn comatcopy2;  /* FB_OP_COMATCOPY2 = 873 */
  fb_generic_fn zomatcopy2;  /* FB_OP_ZOMATCOPY2 = 874 */
  fb_generic_fn sorg2l;  /* FB_OP_SORG2L = 875 */
  fb_generic_fn dorg2l;  /* FB_OP_DORG2L = 876 */
  fb_generic_fn sorgbr;  /* FB_OP_SORGBR = 877 */
  fb_generic_fn dorgbr;  /* FB_OP_DORGBR = 878 */
  fb_generic_fn sorghr;  /* FB_OP_SORGHR = 879 */
  fb_generic_fn dorghr;  /* FB_OP_DORGHR = 880 */
  fb_generic_fn sorgl2;  /* FB_OP_SORGL2 = 881 */
  fb_generic_fn dorgl2;  /* FB_OP_DORGL2 = 882 */
  fb_generic_fn sorglq;  /* FB_OP_SORGLQ = 883 */
  fb_generic_fn dorglq;  /* FB_OP_DORGLQ = 884 */
  fb_generic_fn sorgql;  /* FB_OP_SORGQL = 885 */
  fb_generic_fn dorgql;  /* FB_OP_DORGQL = 886 */
  fb_generic_fn sorgr2;  /* FB_OP_SORGR2 = 889 */
  fb_generic_fn dorgr2;  /* FB_OP_DORGR2 = 890 */
  fb_generic_fn sorgrq;  /* FB_OP_SORGRQ = 891 */
  fb_generic_fn dorgrq;  /* FB_OP_DORGRQ = 892 */
  fb_generic_fn sorgtr;  /* FB_OP_SORGTR = 893 */
  fb_generic_fn dorgtr;  /* FB_OP_DORGTR = 894 */
  fb_generic_fn sormbr;  /* FB_OP_SORMBR = 895 */
  fb_generic_fn dormbr;  /* FB_OP_DORMBR = 896 */
  fb_generic_fn sormhr;  /* FB_OP_SORMHR = 897 */
  fb_generic_fn dormhr;  /* FB_OP_DORMHR = 898 */
  fb_generic_fn sormlq;  /* FB_OP_SORMLQ = 899 */
  fb_generic_fn dormlq;  /* FB_OP_DORMLQ = 900 */
  fb_generic_fn sormql;  /* FB_OP_SORMQL = 901 */
  fb_generic_fn dormql;  /* FB_OP_DORMQL = 902 */
  fb_generic_fn sormr3;  /* FB_OP_SORMR3 = 905 */
  fb_generic_fn dormr3;  /* FB_OP_DORMR3 = 906 */
  fb_generic_fn sormrq;  /* FB_OP_SORMRQ = 907 */
  fb_generic_fn dormrq;  /* FB_OP_DORMRQ = 908 */
  fb_generic_fn sormrz;  /* FB_OP_SORMRZ = 909 */
  fb_generic_fn dormrz;  /* FB_OP_DORMRZ = 910 */
  fb_generic_fn sparse;  /* FB_OP_SPARSE = 911 */
  fb_generic_fn spbcon;  /* FB_OP_SPBCON = 912 */
  fb_generic_fn dpbcon;  /* FB_OP_DPBCON = 913 */
  fb_generic_fn cpbcon;  /* FB_OP_CPBCON = 914 */
  fb_generic_fn zpbcon;  /* FB_OP_ZPBCON = 915 */
  fb_generic_fn spbequ;  /* FB_OP_SPBEQU = 916 */
  fb_generic_fn dpbequ;  /* FB_OP_DPBEQU = 917 */
  fb_generic_fn cpbequ;  /* FB_OP_CPBEQU = 918 */
  fb_generic_fn zpbequ;  /* FB_OP_ZPBEQU = 919 */
  fb_generic_fn spbrfs;  /* FB_OP_SPBRFS = 920 */
  fb_generic_fn dpbrfs;  /* FB_OP_DPBRFS = 921 */
  fb_generic_fn cpbrfs;  /* FB_OP_CPBRFS = 922 */
  fb_generic_fn zpbrfs;  /* FB_OP_ZPBRFS = 923 */
  fb_generic_fn spbsv;  /* FB_OP_SPBSV = 924 */
  fb_generic_fn dpbsv;  /* FB_OP_DPBSV = 925 */
  fb_generic_fn cpbsv;  /* FB_OP_CPBSV = 926 */
  fb_generic_fn zpbsv;  /* FB_OP_ZPBSV = 927 */
  fb_generic_fn spbsvx;  /* FB_OP_SPBSVX = 928 */
  fb_generic_fn dpbsvx;  /* FB_OP_DPBSVX = 929 */
  fb_generic_fn cpbsvx;  /* FB_OP_CPBSVX = 930 */
  fb_generic_fn zpbsvx;  /* FB_OP_ZPBSVX = 931 */
  fb_generic_fn spbtrf;  /* FB_OP_SPBTRF = 932 */
  fb_generic_fn dpbtrf;  /* FB_OP_DPBTRF = 933 */
  fb_generic_fn cpbtrf;  /* FB_OP_CPBTRF = 934 */
  fb_generic_fn zpbtrf;  /* FB_OP_ZPBTRF = 935 */
  fb_generic_fn spbtrs;  /* FB_OP_SPBTRS = 936 */
  fb_generic_fn dpbtrs;  /* FB_OP_DPBTRS = 937 */
  fb_generic_fn cpbtrs;  /* FB_OP_CPBTRS = 938 */
  fb_generic_fn zpbtrs;  /* FB_OP_ZPBTRS = 939 */
  fb_generic_fn spocon;  /* FB_OP_SPOCON = 940 */
  fb_generic_fn dpocon;  /* FB_OP_DPOCON = 941 */
  fb_generic_fn cpocon;  /* FB_OP_CPOCON = 942 */
  fb_generic_fn zpocon;  /* FB_OP_ZPOCON = 943 */
  fb_generic_fn spoequ;  /* FB_OP_SPOEQU = 944 */
  fb_generic_fn dpoequ;  /* FB_OP_DPOEQU = 945 */
  fb_generic_fn cpoequ;  /* FB_OP_CPOEQU = 946 */
  fb_generic_fn zpoequ;  /* FB_OP_ZPOEQU = 947 */
  fb_generic_fn sporfs;  /* FB_OP_SPORFS = 948 */
  fb_generic_fn dporfs;  /* FB_OP_DPORFS = 949 */
  fb_generic_fn cporfs;  /* FB_OP_CPORFS = 950 */
  fb_generic_fn zporfs;  /* FB_OP_ZPORFS = 951 */
  fb_generic_fn sposvx;  /* FB_OP_SPOSVX = 956 */
  fb_generic_fn dposvx;  /* FB_OP_DPOSVX = 957 */
  fb_generic_fn cposvx;  /* FB_OP_CPOSVX = 958 */
  fb_generic_fn zposvx;  /* FB_OP_ZPOSVX = 959 */
  fb_generic_fn sposvxx;  /* FB_OP_SPOSVXX = 960 */
  fb_generic_fn dposvxx;  /* FB_OP_DPOSVXX = 961 */
  fb_generic_fn cposvxx;  /* FB_OP_CPOSVXX = 962 */
  fb_generic_fn zposvxx;  /* FB_OP_ZPOSVXX = 963 */
  fb_generic_fn sppcon;  /* FB_OP_SPPCON = 976 */
  fb_generic_fn dppcon;  /* FB_OP_DPPCON = 977 */
  fb_generic_fn cppcon;  /* FB_OP_CPPCON = 978 */
  fb_generic_fn zppcon;  /* FB_OP_ZPPCON = 979 */
  fb_generic_fn sppequ;  /* FB_OP_SPPEQU = 980 */
  fb_generic_fn dppequ;  /* FB_OP_DPPEQU = 981 */
  fb_generic_fn cppequ;  /* FB_OP_CPPEQU = 982 */
  fb_generic_fn zppequ;  /* FB_OP_ZPPEQU = 983 */
  fb_generic_fn spprfs;  /* FB_OP_SPPRFS = 984 */
  fb_generic_fn dpprfs;  /* FB_OP_DPPRFS = 985 */
  fb_generic_fn cpprfs;  /* FB_OP_CPPRFS = 986 */
  fb_generic_fn zpprfs;  /* FB_OP_ZPPRFS = 987 */
  fb_generic_fn sppsv;  /* FB_OP_SPPSV = 988 */
  fb_generic_fn dppsv;  /* FB_OP_DPPSV = 989 */
  fb_generic_fn cppsv;  /* FB_OP_CPPSV = 990 */
  fb_generic_fn zppsv;  /* FB_OP_ZPPSV = 991 */
  fb_generic_fn sppsvx;  /* FB_OP_SPPSVX = 992 */
  fb_generic_fn dppsvx;  /* FB_OP_DPPSVX = 993 */
  fb_generic_fn cppsvx;  /* FB_OP_CPPSVX = 994 */
  fb_generic_fn zppsvx;  /* FB_OP_ZPPSVX = 995 */
  fb_generic_fn spptrf;  /* FB_OP_SPPTRF = 996 */
  fb_generic_fn dpptrf;  /* FB_OP_DPPTRF = 997 */
  fb_generic_fn cpptrf;  /* FB_OP_CPPTRF = 998 */
  fb_generic_fn zpptrf;  /* FB_OP_ZPPTRF = 999 */
  fb_generic_fn spptri;  /* FB_OP_SPPTRI = 1000 */
  fb_generic_fn dpptri;  /* FB_OP_DPPTRI = 1001 */
  fb_generic_fn cpptri;  /* FB_OP_CPPTRI = 1002 */
  fb_generic_fn zpptri;  /* FB_OP_ZPPTRI = 1003 */
  fb_generic_fn spptrs;  /* FB_OP_SPPTRS = 1004 */
  fb_generic_fn dpptrs;  /* FB_OP_DPPTRS = 1005 */
  fb_generic_fn cpptrs;  /* FB_OP_CPPTRS = 1006 */
  fb_generic_fn zpptrs;  /* FB_OP_ZPPTRS = 1007 */
  fb_generic_fn sptcon;  /* FB_OP_SPTCON = 1008 */
  fb_generic_fn dptcon;  /* FB_OP_DPTCON = 1009 */
  fb_generic_fn cptcon;  /* FB_OP_CPTCON = 1010 */
  fb_generic_fn zptcon;  /* FB_OP_ZPTCON = 1011 */
  fb_generic_fn spteqr;  /* FB_OP_SPTEQR = 1012 */
  fb_generic_fn dpteqr;  /* FB_OP_DPTEQR = 3054 */
  fb_generic_fn cpteqr;  /* FB_OP_CPTEQR = 3055 */
  fb_generic_fn zpteqr;  /* FB_OP_ZPTEQR = 3056 */
  fb_generic_fn sptrfs;  /* FB_OP_SPTRFS = 1013 */
  fb_generic_fn dptrfs;  /* FB_OP_DPTRFS = 1014 */
  fb_generic_fn cptrfs;  /* FB_OP_CPTRFS = 1015 */
  fb_generic_fn zptrfs;  /* FB_OP_ZPTRFS = 1016 */
  fb_generic_fn sptsv;  /* FB_OP_SPTSV = 1017 */
  fb_generic_fn dptsv;  /* FB_OP_DPTSV = 1018 */
  fb_generic_fn cptsv;  /* FB_OP_CPTSV = 1019 */
  fb_generic_fn zptsv;  /* FB_OP_ZPTSV = 1020 */
  fb_generic_fn sptsvx;  /* FB_OP_SPTSVX = 1021 */
  fb_generic_fn dptsvx;  /* FB_OP_DPTSVX = 1022 */
  fb_generic_fn cptsvx;  /* FB_OP_CPTSVX = 1023 */
  fb_generic_fn zptsvx;  /* FB_OP_ZPTSVX = 1024 */
  fb_generic_fn spttrf;  /* FB_OP_SPTTRF = 1025 */
  fb_generic_fn dpttrf;  /* FB_OP_DPTTRF = 1026 */
  fb_generic_fn cpttrf;  /* FB_OP_CPTTRF = 1027 */
  fb_generic_fn zpttrf;  /* FB_OP_ZPTTRF = 1028 */
  fb_generic_fn spttrs;  /* FB_OP_SPTTRS = 1029 */
  fb_generic_fn dpttrs;  /* FB_OP_DPTTRS = 1030 */
  fb_generic_fn cpttrs;  /* FB_OP_CPTTRS = 1031 */
  fb_generic_fn zpttrs;  /* FB_OP_ZPTTRS = 1032 */
  fb_generic_fn srscl;  /* FB_OP_SRSCL = 1035 */
  fb_generic_fn drscl;  /* FB_OP_DRSCL = 1036 */
  fb_generic_fn ssbev;  /* FB_OP_SSBEV = 1037 */
  fb_generic_fn dsbev;  /* FB_OP_DSBEV = 1038 */
  fb_generic_fn ssbevd;  /* FB_OP_SSBEVD = 1039 */
  fb_generic_fn dsbevd;  /* FB_OP_DSBEVD = 1040 */
  fb_generic_fn ssbevx;  /* FB_OP_SSBEVX = 1041 */
  fb_generic_fn dsbevx;  /* FB_OP_DSBEVX = 1042 */
  fb_generic_fn ssbgv;  /* FB_OP_SSBGV = 1043 */
  fb_generic_fn dsbgv;  /* FB_OP_DSBGV = 1044 */
  fb_generic_fn ssbgvd;  /* FB_OP_SSBGVD = 1045 */
  fb_generic_fn dsbgvd;  /* FB_OP_DSBGVD = 1046 */
  fb_generic_fn ssbgvx;  /* FB_OP_SSBGVX = 1047 */
  fb_generic_fn dsbgvx;  /* FB_OP_DSBGVX = 1048 */
  fb_generic_fn csbmv;  /* FB_OP_CSBMV = 1049 */
  fb_generic_fn zsbmv;  /* FB_OP_ZSBMV = 1050 */
  fb_generic_fn ssbtrf;  /* FB_OP_SSBTRF = 1051 */
  fb_generic_fn dsbtrf;  /* FB_OP_DSBTRF = 1052 */
  fb_generic_fn sspcon;  /* FB_OP_SSPCON = 1053 */
  fb_generic_fn dspcon;  /* FB_OP_DSPCON = 1054 */
  fb_generic_fn cspcon;  /* FB_OP_CSPCON = 1055 */
  fb_generic_fn zspcon;  /* FB_OP_ZSPCON = 1056 */
  fb_generic_fn sspev;  /* FB_OP_SSPEV = 1057 */
  fb_generic_fn dspev;  /* FB_OP_DSPEV = 1058 */
  fb_generic_fn sspevd;  /* FB_OP_SSPEVD = 1059 */
  fb_generic_fn dspevd;  /* FB_OP_DSPEVD = 1060 */
  fb_generic_fn sspevx;  /* FB_OP_SSPEVX = 1061 */
  fb_generic_fn dspevx;  /* FB_OP_DSPEVX = 1062 */
  fb_generic_fn sspgv;  /* FB_OP_SSPGV = 1063 */
  fb_generic_fn dspgv;  /* FB_OP_DSPGV = 1064 */
  fb_generic_fn sspgvd;  /* FB_OP_SSPGVD = 1065 */
  fb_generic_fn dspgvd;  /* FB_OP_DSPGVD = 1066 */
  fb_generic_fn sspgvx;  /* FB_OP_SSPGVX = 1067 */
  fb_generic_fn dspgvx;  /* FB_OP_DSPGVX = 1068 */
  fb_generic_fn cspmv;  /* FB_OP_CSPMV = 1069 */
  fb_generic_fn zspmv;  /* FB_OP_ZSPMV = 1070 */
  fb_generic_fn cspr;  /* FB_OP_CSPR = 1071 */
  fb_generic_fn zspr;  /* FB_OP_ZSPR = 1072 */
  fb_generic_fn cspr2;  /* FB_OP_CSPR2 = 1073 */
  fb_generic_fn zspr2;  /* FB_OP_ZSPR2 = 1074 */
  fb_generic_fn ssprfs;  /* FB_OP_SSPRFS = 1075 */
  fb_generic_fn dsprfs;  /* FB_OP_DSPRFS = 1076 */
  fb_generic_fn csprfs;  /* FB_OP_CSPRFS = 1077 */
  fb_generic_fn zsprfs;  /* FB_OP_ZSPRFS = 1078 */
  fb_generic_fn sspsv;  /* FB_OP_SSPSV = 1079 */
  fb_generic_fn dspsv;  /* FB_OP_DSPSV = 1080 */
  fb_generic_fn cspsv;  /* FB_OP_CSPSV = 1081 */
  fb_generic_fn zspsv;  /* FB_OP_ZSPSV = 1082 */
  fb_generic_fn sspsvx;  /* FB_OP_SSPSVX = 1083 */
  fb_generic_fn dspsvx;  /* FB_OP_DSPSVX = 1084 */
  fb_generic_fn ssptrf;  /* FB_OP_SSPTRF = 1085 */
  fb_generic_fn dsptrf;  /* FB_OP_DSPTRF = 1086 */
  fb_generic_fn csptrf;  /* FB_OP_CSPTRF = 1087 */
  fb_generic_fn zsptrf;  /* FB_OP_ZSPTRF = 1088 */
  fb_generic_fn ssptri;  /* FB_OP_SSPTRI = 1089 */
  fb_generic_fn dsptri;  /* FB_OP_DSPTRI = 1090 */
  fb_generic_fn csptri;  /* FB_OP_CSPTRI = 1091 */
  fb_generic_fn zsptri;  /* FB_OP_ZSPTRI = 1092 */
  fb_generic_fn ssptrs;  /* FB_OP_SSPTRS = 1093 */
  fb_generic_fn dsptrs;  /* FB_OP_DSPTRS = 1094 */
  fb_generic_fn csptrs;  /* FB_OP_CSPTRS = 1095 */
  fb_generic_fn zsptrs;  /* FB_OP_ZSPTRS = 1096 */
  fb_generic_fn csrot;  /* FB_OP_CSROT = 1097 */
  fb_generic_fn csrscl;  /* FB_OP_CSRSCL = 1098 */
  fb_generic_fn sstebz;  /* FB_OP_SSTEBZ = 1099 */
  fb_generic_fn dstebz;  /* FB_OP_DSTEBZ = 1100 */
  fb_generic_fn sstedc;  /* FB_OP_SSTEDC = 1101 */
  fb_generic_fn dstedc;  /* FB_OP_DSTEDC = 1102 */
  fb_generic_fn cstedc;  /* FB_OP_CSTEDC = 1103 */
  fb_generic_fn zstedc;  /* FB_OP_ZSTEDC = 1104 */
  fb_generic_fn sstegr;  /* FB_OP_SSTEGR = 1105 */
  fb_generic_fn dstegr;  /* FB_OP_DSTEGR = 1106 */
  fb_generic_fn cstegr;  /* FB_OP_CSTEGR = 1107 */
  fb_generic_fn zstegr;  /* FB_OP_ZSTEGR = 1108 */
  fb_generic_fn sstein;  /* FB_OP_SSTEIN = 1109 */
  fb_generic_fn dstein;  /* FB_OP_DSTEIN = 1110 */
  fb_generic_fn cstein;  /* FB_OP_CSTEIN = 1111 */
  fb_generic_fn zstein;  /* FB_OP_ZSTEIN = 1112 */
  fb_generic_fn ssteqr;  /* FB_OP_SSTEQR = 1113 */
  fb_generic_fn dsteqr;  /* FB_OP_DSTEQR = 1114 */
  fb_generic_fn csteqr;  /* FB_OP_CSTEQR = 1115 */
  fb_generic_fn zsteqr;  /* FB_OP_ZSTEQR = 1116 */
  fb_generic_fn ssterf;  /* FB_OP_SSTERF = 1117 */
  fb_generic_fn dsterf;  /* FB_OP_DSTERF = 1118 */
  fb_generic_fn sstev;  /* FB_OP_SSTEV = 1119 */
  fb_generic_fn dstev;  /* FB_OP_DSTEV = 1120 */
  fb_generic_fn sstevd;  /* FB_OP_SSTEVD = 1121 */
  fb_generic_fn dstevd;  /* FB_OP_DSTEVD = 1122 */
  fb_generic_fn sstevr;  /* FB_OP_SSTEVR = 1123 */
  fb_generic_fn dstevr;  /* FB_OP_DSTEVR = 1124 */
  fb_generic_fn sstevx;  /* FB_OP_SSTEVX = 1125 */
  fb_generic_fn dstevx;  /* FB_OP_DSTEVX = 1126 */
  fb_generic_fn ssycon;  /* FB_OP_SSYCON = 1127 */
  fb_generic_fn dsycon;  /* FB_OP_DSYCON = 1128 */
  fb_generic_fn csycon;  /* FB_OP_CSYCON = 1129 */
  fb_generic_fn zsycon;  /* FB_OP_ZSYCON = 1130 */
  fb_generic_fn ssyevd;  /* FB_OP_SSYEVD = 1133 */
  fb_generic_fn dsyevd;  /* FB_OP_DSYEVD = 1134 */
  fb_generic_fn ssyevr;  /* FB_OP_SSYEVR = 1135 */
  fb_generic_fn dsyevr;  /* FB_OP_DSYEVR = 1136 */
  fb_generic_fn ssyevx;  /* FB_OP_SSYEVX = 1137 */
  fb_generic_fn dsyevx;  /* FB_OP_DSYEVX = 1138 */
  fb_generic_fn ssygst;  /* FB_OP_SSYGST = 1139 */
  fb_generic_fn dsygst;  /* FB_OP_DSYGST = 1140 */
  fb_generic_fn csygst;  /* FB_OP_CSYGST = 1141 */
  fb_generic_fn zsygst;  /* FB_OP_ZSYGST = 1142 */
  fb_generic_fn csygv;  /* FB_OP_CSYGV = 1145 */
  fb_generic_fn zsygv;  /* FB_OP_ZSYGV = 1146 */
  fb_generic_fn ssygvd;  /* FB_OP_SSYGVD = 1147 */
  fb_generic_fn dsygvd;  /* FB_OP_DSYGVD = 1148 */
  fb_generic_fn csygvd;  /* FB_OP_CSYGVD = 1149 */
  fb_generic_fn zsygvd;  /* FB_OP_ZSYGVD = 1150 */
  fb_generic_fn ssygvx;  /* FB_OP_SSYGVX = 1151 */
  fb_generic_fn dsygvx;  /* FB_OP_DSYGVX = 1152 */
  fb_generic_fn csygvx;  /* FB_OP_CSYGVX = 1153 */
  fb_generic_fn zsygvx;  /* FB_OP_ZSYGVX = 1154 */
  fb_generic_fn csyr2;  /* FB_OP_CSYR2 = 1159 */
  fb_generic_fn zsyr2;  /* FB_OP_ZSYR2 = 1160 */
  fb_generic_fn ssyrfs;  /* FB_OP_SSYRFS = 1161 */
  fb_generic_fn dsyrfs;  /* FB_OP_DSYRFS = 1162 */
  fb_generic_fn csyrfs;  /* FB_OP_CSYRFS = 1163 */
  fb_generic_fn zsyrfs;  /* FB_OP_ZSYRFS = 1164 */
  fb_generic_fn ssysvx;  /* FB_OP_SSYSVX = 1169 */
  fb_generic_fn dsysvx;  /* FB_OP_DSYSVX = 1170 */
  fb_generic_fn csysvx;  /* FB_OP_CSYSVX = 1171 */
  fb_generic_fn zsysvx;  /* FB_OP_ZSYSVX = 1172 */
  fb_generic_fn ssysvxx;  /* FB_OP_SSYSVXX = 1173 */
  fb_generic_fn dsysvxx;  /* FB_OP_DSYSVXX = 1174 */
  fb_generic_fn csysvxx;  /* FB_OP_CSYSVXX = 1175 */
  fb_generic_fn zsysvxx;  /* FB_OP_ZSYSVXX = 1176 */
  fb_generic_fn ssytrd;  /* FB_OP_SSYTRD = 1177 */
  fb_generic_fn dsytrd;  /* FB_OP_DSYTRD = 1178 */
  fb_generic_fn csytrd;  /* FB_OP_CSYTRD = 1179 */
  fb_generic_fn zsytrd;  /* FB_OP_ZSYTRD = 1180 */
  fb_generic_fn ssytrf;  /* FB_OP_SSYTRF = 1181 */
  fb_generic_fn dsytrf;  /* FB_OP_DSYTRF = 1182 */
  fb_generic_fn csytrf;  /* FB_OP_CSYTRF = 1183 */
  fb_generic_fn zsytrf;  /* FB_OP_ZSYTRF = 1184 */
  fb_generic_fn ssytrf_aa;  /* FB_OP_SSYTRF_AA = 1185 */
  fb_generic_fn ssytrf_rook;  /* FB_OP_SSYTRF_ROOK = 1186 */
  fb_generic_fn dsytrf_rook;  /* FB_OP_DSYTRF_ROOK = 1187 */
  fb_generic_fn csytrf_rook;  /* FB_OP_CSYTRF_ROOK = 1188 */
  fb_generic_fn zsytrf_rook;  /* FB_OP_ZSYTRF_ROOK = 1189 */
  fb_generic_fn ssytri;  /* FB_OP_SSYTRI = 1190 */
  fb_generic_fn dsytri;  /* FB_OP_DSYTRI = 1191 */
  fb_generic_fn csytri;  /* FB_OP_CSYTRI = 1192 */
  fb_generic_fn zsytri;  /* FB_OP_ZSYTRI = 1193 */
  fb_generic_fn ssytri2;  /* FB_OP_SSYTRI2 = 1194 */
  fb_generic_fn dsytri2;  /* FB_OP_DSYTRI2 = 1195 */
  fb_generic_fn csytri2;  /* FB_OP_CSYTRI2 = 1196 */
  fb_generic_fn ssytrs;  /* FB_OP_SSYTRS = 1197 */
  fb_generic_fn dsytrs;  /* FB_OP_DSYTRS = 1198 */
  fb_generic_fn csytrs;  /* FB_OP_CSYTRS = 1199 */
  fb_generic_fn zsytrs;  /* FB_OP_ZSYTRS = 1200 */
  fb_generic_fn stbcon;  /* FB_OP_STBCON = 1201 */
  fb_generic_fn dtbcon;  /* FB_OP_DTBCON = 1202 */
  fb_generic_fn ctbcon;  /* FB_OP_CTBCON = 1203 */
  fb_generic_fn ztbcon;  /* FB_OP_ZTBCON = 1204 */
  fb_generic_fn stbrfs;  /* FB_OP_STBRFS = 1205 */
  fb_generic_fn dtbrfs;  /* FB_OP_DTBRFS = 1206 */
  fb_generic_fn ctbrfs;  /* FB_OP_CTBRFS = 1207 */
  fb_generic_fn ztbrfs;  /* FB_OP_ZTBRFS = 1208 */
  fb_generic_fn stbtrs;  /* FB_OP_STBTRS = 1209 */
  fb_generic_fn dtbtrs;  /* FB_OP_DTBTRS = 1210 */
  fb_generic_fn ctbtrs;  /* FB_OP_CTBTRS = 1211 */
  fb_generic_fn ztbtrs;  /* FB_OP_ZTBTRS = 1212 */
  fb_generic_fn stgevc;  /* FB_OP_STGEVC = 1213 */
  fb_generic_fn dtgevc;  /* FB_OP_DTGEVC = 1214 */
  fb_generic_fn ctgevc;  /* FB_OP_CTGEVC = 1215 */
  fb_generic_fn ztgevc;  /* FB_OP_ZTGEVC = 1216 */
  fb_generic_fn stgex2;  /* FB_OP_STGEX2 = 1217 */
  fb_generic_fn dtgex2;  /* FB_OP_DTGEX2 = 1218 */
  fb_generic_fn ctgex2;  /* FB_OP_CTGEX2 = 1219 */
  fb_generic_fn ztgex2;  /* FB_OP_ZTGEX2 = 1220 */
  fb_generic_fn stgexc;  /* FB_OP_STGEXC = 1221 */
  fb_generic_fn dtgexc;  /* FB_OP_DTGEXC = 1222 */
  fb_generic_fn ctgexc;  /* FB_OP_CTGEXC = 1223 */
  fb_generic_fn ztgexc;  /* FB_OP_ZTGEXC = 1224 */
  fb_generic_fn stgsen;  /* FB_OP_STGSEN = 1225 */
  fb_generic_fn dtgsen;  /* FB_OP_DTGSEN = 1226 */
  fb_generic_fn ctgsen;  /* FB_OP_CTGSEN = 1227 */
  fb_generic_fn ztgsen;  /* FB_OP_ZTGSEN = 1228 */
  fb_generic_fn stgsja;  /* FB_OP_STGSJA = 1229 */
  fb_generic_fn dtgsja;  /* FB_OP_DTGSJA = 1230 */
  fb_generic_fn ctgsja;  /* FB_OP_CTGSJA = 1231 */
  fb_generic_fn ztgsja;  /* FB_OP_ZTGSJA = 1232 */
  fb_generic_fn stgsna;  /* FB_OP_STGSNA = 1233 */
  fb_generic_fn dtgsna;  /* FB_OP_DTGSNA = 1234 */
  fb_generic_fn ctgsna;  /* FB_OP_CTGSNA = 1235 */
  fb_generic_fn ztgsna;  /* FB_OP_ZTGSNA = 1236 */
  fb_generic_fn stgsy2;  /* FB_OP_STGSY2 = 1237 */
  fb_generic_fn dtgsy2;  /* FB_OP_DTGSY2 = 1238 */
  fb_generic_fn ctgsy2;  /* FB_OP_CTGSY2 = 1239 */
  fb_generic_fn ztgsy2;  /* FB_OP_ZTGSY2 = 1240 */
  fb_generic_fn stgsyl;  /* FB_OP_STGSYL = 1241 */
  fb_generic_fn dtgsyl;  /* FB_OP_DTGSYL = 1242 */
  fb_generic_fn ctgsyl;  /* FB_OP_CTGSYL = 1243 */
  fb_generic_fn ztgsyl;  /* FB_OP_ZTGSYL = 1244 */
  fb_generic_fn stpcon;  /* FB_OP_STPCON = 1245 */
  fb_generic_fn dtpcon;  /* FB_OP_DTPCON = 1246 */
  fb_generic_fn ctpcon;  /* FB_OP_CTPCON = 1247 */
  fb_generic_fn ztpcon;  /* FB_OP_ZTPCON = 1248 */
  fb_generic_fn stprfs;  /* FB_OP_STPRFS = 1249 */
  fb_generic_fn dtprfs;  /* FB_OP_DTPRFS = 1250 */
  fb_generic_fn ctprfs;  /* FB_OP_CTPRFS = 1251 */
  fb_generic_fn ztprfs;  /* FB_OP_ZTPRFS = 1252 */
  fb_generic_fn stptri;  /* FB_OP_STPTRI = 1253 */
  fb_generic_fn dtptri;  /* FB_OP_DTPTRI = 1254 */
  fb_generic_fn ctptri;  /* FB_OP_CTPTRI = 1255 */
  fb_generic_fn ztptri;  /* FB_OP_ZTPTRI = 1256 */
  fb_generic_fn stptrs;  /* FB_OP_STPTRS = 1257 */
  fb_generic_fn dtptrs;  /* FB_OP_DTPTRS = 1258 */
  fb_generic_fn ctptrs;  /* FB_OP_CTPTRS = 1259 */
  fb_generic_fn ztptrs;  /* FB_OP_ZTPTRS = 1260 */
  fb_generic_fn strcon;  /* FB_OP_STRCON = 1261 */
  fb_generic_fn dtrcon;  /* FB_OP_DTRCON = 1262 */
  fb_generic_fn ctrcon;  /* FB_OP_CTRCON = 1263 */
  fb_generic_fn ztrcon;  /* FB_OP_ZTRCON = 1264 */
  fb_generic_fn strevc;  /* FB_OP_STREVC = 1265 */
  fb_generic_fn dtrevc;  /* FB_OP_DTREVC = 1266 */
  fb_generic_fn ctrevc;  /* FB_OP_CTREVC = 1267 */
  fb_generic_fn ztrevc;  /* FB_OP_ZTREVC = 1268 */
  fb_generic_fn strrfs;  /* FB_OP_STRRFS = 1269 */
  fb_generic_fn dtrrfs;  /* FB_OP_DTRRFS = 1270 */
  fb_generic_fn ctrrfs;  /* FB_OP_CTRRFS = 1271 */
  fb_generic_fn ztrrfs;  /* FB_OP_ZTRRFS = 1272 */
  fb_generic_fn strtrs;  /* FB_OP_STRTRS = 1277 */
  fb_generic_fn dtrtrs;  /* FB_OP_DTRTRS = 1278 */
  fb_generic_fn ctrtrs;  /* FB_OP_CTRTRS = 1279 */
  fb_generic_fn ztrtrs;  /* FB_OP_ZTRTRS = 1280 */
  fb_generic_fn stzrzf;  /* FB_OP_STZRZF = 1281 */
  fb_generic_fn dtzrzf;  /* FB_OP_DTZRZF = 1282 */
  fb_generic_fn ctzrzf;  /* FB_OP_CTZRZF = 1283 */
  fb_generic_fn ztzrzf;  /* FB_OP_ZTZRZF = 1284 */
  fb_generic_fn cung2l;  /* FB_OP_CUNG2L = 1285 */
  fb_generic_fn zung2l;  /* FB_OP_ZUNG2L = 1286 */
  fb_generic_fn cungbr;  /* FB_OP_CUNGBR = 1287 */
  fb_generic_fn zungbr;  /* FB_OP_ZUNGBR = 1288 */
  fb_generic_fn cunghr;  /* FB_OP_CUNGHR = 1289 */
  fb_generic_fn zunghr;  /* FB_OP_ZUNGHR = 1290 */
  fb_generic_fn cungl2;  /* FB_OP_CUNGL2 = 1291 */
  fb_generic_fn zungl2;  /* FB_OP_ZUNGL2 = 1292 */
  fb_generic_fn cunglq;  /* FB_OP_CUNGLQ = 1293 */
  fb_generic_fn zunglq;  /* FB_OP_ZUNGLQ = 1294 */
  fb_generic_fn cungql;  /* FB_OP_CUNGQL = 1295 */
  fb_generic_fn zungql;  /* FB_OP_ZUNGQL = 1296 */
  fb_generic_fn cungr2;  /* FB_OP_CUNGR2 = 1299 */
  fb_generic_fn zungr2;  /* FB_OP_ZUNGR2 = 1300 */
  fb_generic_fn cungrq;  /* FB_OP_CUNGRQ = 1301 */
  fb_generic_fn zungrq;  /* FB_OP_ZUNGRQ = 1302 */
  fb_generic_fn cungtr;  /* FB_OP_CUNGTR = 1303 */
  fb_generic_fn zungtr;  /* FB_OP_ZUNGTR = 1304 */
  fb_generic_fn cunmbr;  /* FB_OP_CUNMBR = 1305 */
  fb_generic_fn zunmbr;  /* FB_OP_ZUNMBR = 1306 */
  fb_generic_fn cunmhr;  /* FB_OP_CUNMHR = 1307 */
  fb_generic_fn zunmhr;  /* FB_OP_ZUNMHR = 1308 */
  fb_generic_fn cunmlq;  /* FB_OP_CUNMLQ = 1309 */
  fb_generic_fn zunmlq;  /* FB_OP_ZUNMLQ = 1310 */
  fb_generic_fn cunmql;  /* FB_OP_CUNMQL = 1311 */
  fb_generic_fn zunmql;  /* FB_OP_ZUNMQL = 1312 */
  fb_generic_fn cunmr3;  /* FB_OP_CUNMR3 = 1315 */
  fb_generic_fn zunmr3;  /* FB_OP_ZUNMR3 = 1316 */
  fb_generic_fn cunmrq;  /* FB_OP_CUNMRQ = 1317 */
  fb_generic_fn zunmrq;  /* FB_OP_ZUNMRQ = 1318 */
  fb_generic_fn cunmrz;  /* FB_OP_CUNMRZ = 1319 */
  fb_generic_fn zunmrz;  /* FB_OP_ZUNMRZ = 1320 */
  fb_generic_fn dzsum1;  /* FB_OP_DZSUM1 = 1321 */

  /* ========================================================================
   * ScaLAPACK — parallel p* distributed routines
   * ========================================================================
   * fb_generic_fn fields: backends assign  vtable->op = (fb_generic_fn)impl;
   * fb_vtable_sync_ext_ops() mirrors them into ext_ops[] automatically. */

  fb_generic_fn pcamax;  /* FB_OP_PCAMAX = 1322 */
  fb_generic_fn pcasum;  /* FB_OP_PCASUM = 1323 */
  fb_generic_fn pcaxpy;  /* FB_OP_PCAXPY = 1324 */
  fb_generic_fn pccopy;  /* FB_OP_PCCOPY = 1325 */
  fb_generic_fn pcdot;  /* FB_OP_PCDOT = 1326 */
  fb_generic_fn pcgbsv;  /* FB_OP_PCGBSV = 1327 */
  fb_generic_fn pcgbtrf;  /* FB_OP_PCGBTRF = 1328 */
  fb_generic_fn pcgbtrs;  /* FB_OP_PCGBTRS = 1329 */
  fb_generic_fn pcgebrd;  /* FB_OP_PCGEBRD = 1330 */
  fb_generic_fn pcgecon;  /* FB_OP_PCGECON = 1331 */
  fb_generic_fn pcgeequ;  /* FB_OP_PCGEEQU = 1332 */
  fb_generic_fn pcgeevx;  /* FB_OP_PCGEEVX = 1333 */
  fb_generic_fn pcgehrd;  /* FB_OP_PCGEHRD = 1334 */
  fb_generic_fn pcgelqf;  /* FB_OP_PCGELQF = 1335 */
  fb_generic_fn pcgels;  /* FB_OP_PCGELS = 1336 */
  fb_generic_fn pcgelss;  /* FB_OP_PCGELSS = 1337 */
  fb_generic_fn pcgelsy;  /* FB_OP_PCGELSY = 1338 */
  fb_generic_fn pcgemm;  /* FB_OP_PCGEMM = 1339 */
  fb_generic_fn pcgemv;  /* FB_OP_PCGEMV = 1340 */
  fb_generic_fn pcgeqlf;  /* FB_OP_PCGEQLF = 1341 */
  fb_generic_fn pcger;  /* FB_OP_PCGER = 1342 */
  fb_generic_fn pcgerfs;  /* FB_OP_PCGERFS = 1343 */
  fb_generic_fn pcgerqf;  /* FB_OP_PCGERQF = 1344 */
  fb_generic_fn pcgesdd;  /* FB_OP_PCGESDD = 1345 */
  fb_generic_fn pcgesv;  /* FB_OP_PCGESV = 1346 */
  fb_generic_fn pcgesvd;  /* FB_OP_PCGESVD = 1347 */
  fb_generic_fn pcgesvx;  /* FB_OP_PCGESVX = 1348 */
  fb_generic_fn pcgetrf;  /* FB_OP_PCGETRF = 1349 */
  fb_generic_fn pcgetri;  /* FB_OP_PCGETRI = 1350 */
  fb_generic_fn pcgetrs;  /* FB_OP_PCGETRS = 1351 */
  fb_generic_fn pcgghrd;  /* FB_OP_PCGGHRD = 1352 */
  fb_generic_fn pcheequb;  /* FB_OP_PCHEEQUB = 1353 */
  fb_generic_fn pcheev;  /* FB_OP_PCHEEV = 1354 */
  fb_generic_fn pcheevd;  /* FB_OP_PCHEEVD = 1355 */
  fb_generic_fn pcheevx;  /* FB_OP_PCHEEVX = 1356 */
  fb_generic_fn pchegst;  /* FB_OP_PCHEGST = 1357 */
  fb_generic_fn pchegv;  /* FB_OP_PCHEGV = 1358 */
  fb_generic_fn pchesv;  /* FB_OP_PCHESV = 1359 */
  fb_generic_fn pchetrd;  /* FB_OP_PCHETRD = 1360 */
  fb_generic_fn pchetrf;  /* FB_OP_PCHETRF = 1361 */
  fb_generic_fn pchetrs;  /* FB_OP_PCHETRS = 1362 */
  fb_generic_fn pcnrm2;  /* FB_OP_PCNRM2 = 1363 */
  fb_generic_fn pcpbsv;  /* FB_OP_PCPBSV = 1364 */
  fb_generic_fn pcpbtrf;  /* FB_OP_PCPBTRF = 1365 */
  fb_generic_fn pcpbtrs;  /* FB_OP_PCPBTRS = 1366 */
  fb_generic_fn pcpocon;  /* FB_OP_PCPOCON = 1367 */
  fb_generic_fn pcpoequ;  /* FB_OP_PCPOEQU = 1368 */
  fb_generic_fn pcporfs;  /* FB_OP_PCPORFS = 1369 */
  fb_generic_fn pcposv;  /* FB_OP_PCPOSV = 1370 */
  fb_generic_fn pcposvx;  /* FB_OP_PCPOSVX = 1371 */
  fb_generic_fn pcpotrf;  /* FB_OP_PCPOTRF = 1372 */
  fb_generic_fn pcpotri;  /* FB_OP_PCPOTRI = 1373 */
  fb_generic_fn pcpotrs;  /* FB_OP_PCPOTRS = 1374 */
  fb_generic_fn pcppsv;  /* FB_OP_PCPPSV = 1375 */
  fb_generic_fn pcptsv;  /* FB_OP_PCPTSV = 1376 */
  fb_generic_fn pcpttrf;  /* FB_OP_PCPTTRF = 1377 */
  fb_generic_fn pcpttrs;  /* FB_OP_PCPTTRS = 1378 */
  fb_generic_fn pcscal;  /* FB_OP_PCSCAL = 1379 */
  fb_generic_fn pcswap;  /* FB_OP_PCSWAP = 1380 */
  fb_generic_fn pcsygst;  /* FB_OP_PCSYGST = 1381 */
  fb_generic_fn pcsygv;  /* FB_OP_PCSYGV = 1382 */
  fb_generic_fn pcsymm;  /* FB_OP_PCSYMM = 1383 */
  fb_generic_fn pcsymv;  /* FB_OP_PCSYMV = 1384 */
  fb_generic_fn pcsyr;  /* FB_OP_PCSYR = 1385 */
  fb_generic_fn pcsyr2;  /* FB_OP_PCSYR2 = 1386 */
  fb_generic_fn pcsyr2k;  /* FB_OP_PCSYR2K = 1387 */
  fb_generic_fn pcsyrk;  /* FB_OP_PCSYRK = 1388 */
  fb_generic_fn pcsysv;  /* FB_OP_PCSYSV = 1389 */
  fb_generic_fn pcsytrd;  /* FB_OP_PCSYTRD = 1390 */
  fb_generic_fn pcsytrf;  /* FB_OP_PCSYTRF = 1391 */
  fb_generic_fn pcsytrs;  /* FB_OP_PCSYTRS = 1392 */
  fb_generic_fn pctrcon;  /* FB_OP_PCTRCON = 1393 */
  fb_generic_fn pctrmm;  /* FB_OP_PCTRMM = 1394 */
  fb_generic_fn pctrmv;  /* FB_OP_PCTRMV = 1395 */
  fb_generic_fn pctrrfs;  /* FB_OP_PCTRRFS = 1396 */
  fb_generic_fn pctrsm;  /* FB_OP_PCTRSM = 1397 */
  fb_generic_fn pctrsv;  /* FB_OP_PCTRSV = 1398 */
  fb_generic_fn pctrtri;  /* FB_OP_PCTRTRI = 1399 */
  fb_generic_fn pctzrzf;  /* FB_OP_PCTZRZF = 1400 */
  fb_generic_fn pcungqr;  /* FB_OP_PCUNGQR = 1401 */
  fb_generic_fn pcunmqr;  /* FB_OP_PCUNMQR = 1402 */
  fb_generic_fn pdamax;  /* FB_OP_PDAMAX = 1403 */
  fb_generic_fn pdasum;  /* FB_OP_PDASUM = 1404 */
  fb_generic_fn pdaxpy;  /* FB_OP_PDAXPY = 1405 */
  fb_generic_fn pdcopy;  /* FB_OP_PDCOPY = 1406 */
  fb_generic_fn pddot;  /* FB_OP_PDDOT = 1407 */
  fb_generic_fn pdgbsv;  /* FB_OP_PDGBSV = 1408 */
  fb_generic_fn pdgbtrf;  /* FB_OP_PDGBTRF = 1409 */
  fb_generic_fn pdgbtrs;  /* FB_OP_PDGBTRS = 1410 */
  fb_generic_fn pdgebrd;  /* FB_OP_PDGEBRD = 1411 */
  fb_generic_fn pdgecon;  /* FB_OP_PDGECON = 1412 */
  fb_generic_fn pdgeequ;  /* FB_OP_PDGEEQU = 1413 */
  fb_generic_fn pdgeevx;  /* FB_OP_PDGEEVX = 1414 */
  fb_generic_fn pdgehrd;  /* FB_OP_PDGEHRD = 1415 */
  fb_generic_fn pdgelqf;  /* FB_OP_PDGELQF = 1416 */
  fb_generic_fn pdgels;  /* FB_OP_PDGELS = 1417 */
  fb_generic_fn pdgelss;  /* FB_OP_PDGELSS = 1418 */
  fb_generic_fn pdgelsy;  /* FB_OP_PDGELSY = 1419 */
  fb_generic_fn pdgemm;  /* FB_OP_PDGEMM = 1420 */
  fb_generic_fn pdgemv;  /* FB_OP_PDGEMV = 1421 */
  fb_generic_fn pdgeqlf;  /* FB_OP_PDGEQLF = 1422 */
  fb_generic_fn pdger;  /* FB_OP_PDGER = 1423 */
  fb_generic_fn pdgerfs;  /* FB_OP_PDGERFS = 1424 */
  fb_generic_fn pdgerqf;  /* FB_OP_PDGERQF = 1425 */
  fb_generic_fn pdgesdd;  /* FB_OP_PDGESDD = 1426 */
  fb_generic_fn pdgesv;  /* FB_OP_PDGESV = 1427 */
  fb_generic_fn pdgesvd;  /* FB_OP_PDGESVD = 1428 */
  fb_generic_fn pdgesvx;  /* FB_OP_PDGESVX = 1429 */
  fb_generic_fn pdgetrf;  /* FB_OP_PDGETRF = 1430 */
  fb_generic_fn pdgetri;  /* FB_OP_PDGETRI = 1431 */
  fb_generic_fn pdgetrs;  /* FB_OP_PDGETRS = 1432 */
  fb_generic_fn pdgghrd;  /* FB_OP_PDGGHRD = 1433 */
  fb_generic_fn pdnrm2;  /* FB_OP_PDNRM2 = 1434 */
  fb_generic_fn pdorgqr;  /* FB_OP_PDORGQR = 1435 */
  fb_generic_fn pdormqr;  /* FB_OP_PDORMQR = 1436 */
  fb_generic_fn pdpbsv;  /* FB_OP_PDPBSV = 1437 */
  fb_generic_fn pdpbtrf;  /* FB_OP_PDPBTRF = 1438 */
  fb_generic_fn pdpbtrs;  /* FB_OP_PDPBTRS = 1439 */
  fb_generic_fn pdpocon;  /* FB_OP_PDPOCON = 1440 */
  fb_generic_fn pdpoequ;  /* FB_OP_PDPOEQU = 1441 */
  fb_generic_fn pdporfs;  /* FB_OP_PDPORFS = 1442 */
  fb_generic_fn pdposv;  /* FB_OP_PDPOSV = 1443 */
  fb_generic_fn pdposvx;  /* FB_OP_PDPOSVX = 1444 */
  fb_generic_fn pdpotrf;  /* FB_OP_PDPOTRF = 1445 */
  fb_generic_fn pdpotri;  /* FB_OP_PDPOTRI = 1446 */
  fb_generic_fn pdpotrs;  /* FB_OP_PDPOTRS = 1447 */
  fb_generic_fn pdppsv;  /* FB_OP_PDPPSV = 1448 */
  fb_generic_fn pdptsv;  /* FB_OP_PDPTSV = 1449 */
  fb_generic_fn pdpttrf;  /* FB_OP_PDPTTRF = 1450 */
  fb_generic_fn pdpttrs;  /* FB_OP_PDPTTRS = 1451 */
  fb_generic_fn pdscal;  /* FB_OP_PDSCAL = 1452 */
  fb_generic_fn pdswap;  /* FB_OP_PDSWAP = 1453 */
  fb_generic_fn pdsyequb;  /* FB_OP_PDSYEQUB = 1454 */
  fb_generic_fn pdsyev;  /* FB_OP_PDSYEV = 1455 */
  fb_generic_fn pdsyevd;  /* FB_OP_PDSYEVD = 1456 */
  fb_generic_fn pdsyevx;  /* FB_OP_PDSYEVX = 1457 */
  fb_generic_fn pdsygst;  /* FB_OP_PDSYGST = 1458 */
  fb_generic_fn pdsygv;  /* FB_OP_PDSYGV = 1459 */
  fb_generic_fn pdsymm;  /* FB_OP_PDSYMM = 1460 */
  fb_generic_fn pdsymv;  /* FB_OP_PDSYMV = 1461 */
  fb_generic_fn pdsyr;  /* FB_OP_PDSYR = 1462 */
  fb_generic_fn pdsyr2;  /* FB_OP_PDSYR2 = 1463 */
  fb_generic_fn pdsyr2k;  /* FB_OP_PDSYR2K = 1464 */
  fb_generic_fn pdsyrk;  /* FB_OP_PDSYRK = 1465 */
  fb_generic_fn pdsysv;  /* FB_OP_PDSYSV = 1466 */
  fb_generic_fn pdsytrd;  /* FB_OP_PDSYTRD = 1467 */
  fb_generic_fn pdsytrf;  /* FB_OP_PDSYTRF = 1468 */
  fb_generic_fn pdsytrs;  /* FB_OP_PDSYTRS = 1469 */
  fb_generic_fn pdtrcon;  /* FB_OP_PDTRCON = 1470 */
  fb_generic_fn pdtrmm;  /* FB_OP_PDTRMM = 1471 */
  fb_generic_fn pdtrmv;  /* FB_OP_PDTRMV = 1472 */
  fb_generic_fn pdtrrfs;  /* FB_OP_PDTRRFS = 1473 */
  fb_generic_fn pdtrsm;  /* FB_OP_PDTRSM = 1474 */
  fb_generic_fn pdtrsv;  /* FB_OP_PDTRSV = 1475 */
  fb_generic_fn pdtrtri;  /* FB_OP_PDTRTRI = 1476 */
  fb_generic_fn pdtzrzf;  /* FB_OP_PDTZRZF = 1477 */
  fb_generic_fn psamax;  /* FB_OP_PSAMAX = 1478 */
  fb_generic_fn psasum;  /* FB_OP_PSASUM = 1479 */
  fb_generic_fn psaxpy;  /* FB_OP_PSAXPY = 1480 */
  fb_generic_fn pscopy;  /* FB_OP_PSCOPY = 1481 */
  fb_generic_fn psdot;  /* FB_OP_PSDOT = 1482 */
  fb_generic_fn psgbsv;  /* FB_OP_PSGBSV = 1483 */
  fb_generic_fn psgbtrf;  /* FB_OP_PSGBTRF = 1484 */
  fb_generic_fn psgbtrs;  /* FB_OP_PSGBTRS = 1485 */
  fb_generic_fn psgebrd;  /* FB_OP_PSGEBRD = 1486 */
  fb_generic_fn psgecon;  /* FB_OP_PSGECON = 1487 */
  fb_generic_fn psgeequ;  /* FB_OP_PSGEEQU = 1488 */
  fb_generic_fn psgeevx;  /* FB_OP_PSGEEVX = 1489 */
  fb_generic_fn psgehrd;  /* FB_OP_PSGEHRD = 1490 */
  fb_generic_fn psgelqf;  /* FB_OP_PSGELQF = 1491 */
  fb_generic_fn psgels;  /* FB_OP_PSGELS = 1492 */
  fb_generic_fn psgelss;  /* FB_OP_PSGELSS = 1493 */
  fb_generic_fn psgelsy;  /* FB_OP_PSGELSY = 1494 */
  fb_generic_fn psgemm;  /* FB_OP_PSGEMM = 1495 */
  fb_generic_fn psgemv;  /* FB_OP_PSGEMV = 1496 */
  fb_generic_fn psgeqlf;  /* FB_OP_PSGEQLF = 1497 */
  fb_generic_fn psger;  /* FB_OP_PSGER = 1498 */
  fb_generic_fn psgerfs;  /* FB_OP_PSGERFS = 1499 */
  fb_generic_fn psgerqf;  /* FB_OP_PSGERQF = 1500 */
  fb_generic_fn psgesdd;  /* FB_OP_PSGESDD = 1501 */
  fb_generic_fn psgesv;  /* FB_OP_PSGESV = 1502 */
  fb_generic_fn psgesvd;  /* FB_OP_PSGESVD = 1503 */
  fb_generic_fn psgesvx;  /* FB_OP_PSGESVX = 1504 */
  fb_generic_fn psgetrf;  /* FB_OP_PSGETRF = 1505 */
  fb_generic_fn psgetri;  /* FB_OP_PSGETRI = 1506 */
  fb_generic_fn psgetrs;  /* FB_OP_PSGETRS = 1507 */
  fb_generic_fn psgghrd;  /* FB_OP_PSGGHRD = 1508 */
  fb_generic_fn psnrm2;  /* FB_OP_PSNRM2 = 1509 */
  fb_generic_fn psorgqr;  /* FB_OP_PSORGQR = 1510 */
  fb_generic_fn psormqr;  /* FB_OP_PSORMQR = 1511 */
  fb_generic_fn pspbsv;  /* FB_OP_PSPBSV = 1512 */
  fb_generic_fn pspbtrf;  /* FB_OP_PSPBTRF = 1513 */
  fb_generic_fn pspbtrs;  /* FB_OP_PSPBTRS = 1514 */
  fb_generic_fn pspocon;  /* FB_OP_PSPOCON = 1515 */
  fb_generic_fn pspoequ;  /* FB_OP_PSPOEQU = 1516 */
  fb_generic_fn psporfs;  /* FB_OP_PSPORFS = 1517 */
  fb_generic_fn psposv;  /* FB_OP_PSPOSV = 1518 */
  fb_generic_fn psposvx;  /* FB_OP_PSPOSVX = 1519 */
  fb_generic_fn pspotrf;  /* FB_OP_PSPOTRF = 1520 */
  fb_generic_fn pspotri;  /* FB_OP_PSPOTRI = 1521 */
  fb_generic_fn pspotrs;  /* FB_OP_PSPOTRS = 1522 */
  fb_generic_fn psppsv;  /* FB_OP_PSPPSV = 1523 */
  fb_generic_fn psptsv;  /* FB_OP_PSPTSV = 1524 */
  fb_generic_fn pspttrf;  /* FB_OP_PSPTTRF = 1525 */
  fb_generic_fn pspttrs;  /* FB_OP_PSPTTRS = 1526 */
  fb_generic_fn psscal;  /* FB_OP_PSSCAL = 1527 */
  fb_generic_fn psswap;  /* FB_OP_PSSWAP = 1528 */
  fb_generic_fn pssyequb;  /* FB_OP_PSSYEQUB = 1529 */
  fb_generic_fn pssyev;  /* FB_OP_PSSYEV = 1530 */
  fb_generic_fn pssyevd;  /* FB_OP_PSSYEVD = 1531 */
  fb_generic_fn pssyevx;  /* FB_OP_PSSYEVX = 1532 */
  fb_generic_fn pssygst;  /* FB_OP_PSSYGST = 1533 */
  fb_generic_fn pssygv;  /* FB_OP_PSSYGV = 1534 */
  fb_generic_fn pssymm;  /* FB_OP_PSSYMM = 1535 */
  fb_generic_fn pssymv;  /* FB_OP_PSSYMV = 1536 */
  fb_generic_fn pssyr;  /* FB_OP_PSSYR = 1537 */
  fb_generic_fn pssyr2;  /* FB_OP_PSSYR2 = 1538 */
  fb_generic_fn pssyr2k;  /* FB_OP_PSSYR2K = 1539 */
  fb_generic_fn pssyrk;  /* FB_OP_PSSYRK = 1540 */
  fb_generic_fn pssysv;  /* FB_OP_PSSYSV = 1541 */
  fb_generic_fn pssytrd;  /* FB_OP_PSSYTRD = 1542 */
  fb_generic_fn pssytrf;  /* FB_OP_PSSYTRF = 1543 */
  fb_generic_fn pssytrs;  /* FB_OP_PSSYTRS = 1544 */
  fb_generic_fn pstrcon;  /* FB_OP_PSTRCON = 1545 */
  fb_generic_fn pstrmm;  /* FB_OP_PSTRMM = 1546 */
  fb_generic_fn pstrmv;  /* FB_OP_PSTRMV = 1547 */
  fb_generic_fn pstrrfs;  /* FB_OP_PSTRRFS = 1548 */
  fb_generic_fn pstrsm;  /* FB_OP_PSTRSM = 1549 */
  fb_generic_fn pstrsv;  /* FB_OP_PSTRSV = 1550 */
  fb_generic_fn pstrtri;  /* FB_OP_PSTRTRI = 1551 */
  fb_generic_fn pstzrzf;  /* FB_OP_PSTZRZF = 1552 */
  fb_generic_fn pzamax;  /* FB_OP_PZAMAX = 1553 */
  fb_generic_fn pzasum;  /* FB_OP_PZASUM = 1554 */
  fb_generic_fn pzaxpy;  /* FB_OP_PZAXPY = 1555 */
  fb_generic_fn pzcopy;  /* FB_OP_PZCOPY = 1556 */
  fb_generic_fn pzdot;  /* FB_OP_PZDOT = 1557 */
  fb_generic_fn pzgbsv;  /* FB_OP_PZGBSV = 1558 */
  fb_generic_fn pzgbtrf;  /* FB_OP_PZGBTRF = 1559 */
  fb_generic_fn pzgbtrs;  /* FB_OP_PZGBTRS = 1560 */
  fb_generic_fn pzgebrd;  /* FB_OP_PZGEBRD = 1561 */
  fb_generic_fn pzgecon;  /* FB_OP_PZGECON = 1562 */
  fb_generic_fn pzgeequ;  /* FB_OP_PZGEEQU = 1563 */
  fb_generic_fn pzgeevx;  /* FB_OP_PZGEEVX = 1564 */
  fb_generic_fn pzgehrd;  /* FB_OP_PZGEHRD = 1565 */
  fb_generic_fn pzgelqf;  /* FB_OP_PZGELQF = 1566 */
  fb_generic_fn pzgels;  /* FB_OP_PZGELS = 1567 */
  fb_generic_fn pzgelss;  /* FB_OP_PZGELSS = 1568 */
  fb_generic_fn pzgelsy;  /* FB_OP_PZGELSY = 1569 */
  fb_generic_fn pzgemm;  /* FB_OP_PZGEMM = 1570 */
  fb_generic_fn pzgemv;  /* FB_OP_PZGEMV = 1571 */
  fb_generic_fn pzgeqlf;  /* FB_OP_PZGEQLF = 1572 */
  fb_generic_fn pzger;  /* FB_OP_PZGER = 1573 */
  fb_generic_fn pzgerfs;  /* FB_OP_PZGERFS = 1574 */
  fb_generic_fn pzgerqf;  /* FB_OP_PZGERQF = 1575 */
  fb_generic_fn pzgesdd;  /* FB_OP_PZGESDD = 1576 */
  fb_generic_fn pzgesv;  /* FB_OP_PZGESV = 1577 */
  fb_generic_fn pzgesvd;  /* FB_OP_PZGESVD = 1578 */
  fb_generic_fn pzgesvx;  /* FB_OP_PZGESVX = 1579 */
  fb_generic_fn pzgetrf;  /* FB_OP_PZGETRF = 1580 */
  fb_generic_fn pzgetri;  /* FB_OP_PZGETRI = 1581 */
  fb_generic_fn pzgetrs;  /* FB_OP_PZGETRS = 1582 */
  fb_generic_fn pzgghrd;  /* FB_OP_PZGGHRD = 1583 */
  fb_generic_fn pzheequb;  /* FB_OP_PZHEEQUB = 1584 */
  fb_generic_fn pzheev;  /* FB_OP_PZHEEV = 1585 */
  fb_generic_fn pzheevd;  /* FB_OP_PZHEEVD = 1586 */
  fb_generic_fn pzheevx;  /* FB_OP_PZHEEVX = 1587 */
  fb_generic_fn pzhegst;  /* FB_OP_PZHEGST = 1588 */
  fb_generic_fn pzhegv;  /* FB_OP_PZHEGV = 1589 */
  fb_generic_fn pzhesv;  /* FB_OP_PZHESV = 1590 */
  fb_generic_fn pzhetrd;  /* FB_OP_PZHETRD = 1591 */
  fb_generic_fn pzhetrf;  /* FB_OP_PZHETRF = 1592 */
  fb_generic_fn pzhetrs;  /* FB_OP_PZHETRS = 1593 */
  fb_generic_fn pznrm2;  /* FB_OP_PZNRM2 = 1594 */
  fb_generic_fn pzpbsv;  /* FB_OP_PZPBSV = 1595 */
  fb_generic_fn pzpbtrf;  /* FB_OP_PZPBTRF = 1596 */
  fb_generic_fn pzpbtrs;  /* FB_OP_PZPBTRS = 1597 */
  fb_generic_fn pzpocon;  /* FB_OP_PZPOCON = 1598 */
  fb_generic_fn pzpoequ;  /* FB_OP_PZPOEQU = 1599 */
  fb_generic_fn pzporfs;  /* FB_OP_PZPORFS = 1600 */
  fb_generic_fn pzposv;  /* FB_OP_PZPOSV = 1601 */
  fb_generic_fn pzposvx;  /* FB_OP_PZPOSVX = 1602 */
  fb_generic_fn pzpotrf;  /* FB_OP_PZPOTRF = 1603 */
  fb_generic_fn pzpotri;  /* FB_OP_PZPOTRI = 1604 */
  fb_generic_fn pzpotrs;  /* FB_OP_PZPOTRS = 1605 */
  fb_generic_fn pzppsv;  /* FB_OP_PZPPSV = 1606 */
  fb_generic_fn pzptsv;  /* FB_OP_PZPTSV = 1607 */
  fb_generic_fn pzpttrf;  /* FB_OP_PZPTTRF = 1608 */
  fb_generic_fn pzpttrs;  /* FB_OP_PZPTTRS = 1609 */
  fb_generic_fn pzscal;  /* FB_OP_PZSCAL = 1610 */
  fb_generic_fn pzswap;  /* FB_OP_PZSWAP = 1611 */
  fb_generic_fn pzsygst;  /* FB_OP_PZSYGST = 1612 */
  fb_generic_fn pzsygv;  /* FB_OP_PZSYGV = 1613 */
  fb_generic_fn pzsymm;  /* FB_OP_PZSYMM = 1614 */
  fb_generic_fn pzsymv;  /* FB_OP_PZSYMV = 1615 */
  fb_generic_fn pzsyr;  /* FB_OP_PZSYR = 1616 */
  fb_generic_fn pzsyr2;  /* FB_OP_PZSYR2 = 1617 */
  fb_generic_fn pzsyr2k;  /* FB_OP_PZSYR2K = 1618 */
  fb_generic_fn pzsyrk;  /* FB_OP_PZSYRK = 1619 */
  fb_generic_fn pzsysv;  /* FB_OP_PZSYSV = 1620 */
  fb_generic_fn pzsytrd;  /* FB_OP_PZSYTRD = 1621 */
  fb_generic_fn pzsytrf;  /* FB_OP_PZSYTRF = 1622 */
  fb_generic_fn pzsytrs;  /* FB_OP_PZSYTRS = 1623 */
  fb_generic_fn pztrcon;  /* FB_OP_PZTRCON = 1624 */
  fb_generic_fn pztrmm;  /* FB_OP_PZTRMM = 1625 */
  fb_generic_fn pztrmv;  /* FB_OP_PZTRMV = 1626 */
  fb_generic_fn pztrrfs;  /* FB_OP_PZTRRFS = 1627 */
  fb_generic_fn pztrsm;  /* FB_OP_PZTRSM = 1628 */
  fb_generic_fn pztrsv;  /* FB_OP_PZTRSV = 1629 */
  fb_generic_fn pztrtri;  /* FB_OP_PZTRTRI = 1630 */
  fb_generic_fn pztzrzf;  /* FB_OP_PZTZRZF = 1631 */
  fb_generic_fn pzungqr;  /* FB_OP_PZUNGQR = 1632 */
  fb_generic_fn pzunmqr;  /* FB_OP_PZUNMQR = 1633 */

  /* ========================================================================
   * Extended BLAS (cblas_batch / cblas_strided / axpby variants)
   * ========================================================================
   * fb_generic_fn fields: backends assign  vtable->op = (fb_generic_fn)impl;
   * fb_vtable_sync_ext_ops() mirrors them into ext_ops[] automatically. */

  fb_generic_fn saxpby;  /* FB_OP_SAXPBY = 1634 */
  fb_generic_fn daxpby;  /* FB_OP_DAXPBY = 1635 */
  fb_generic_fn caxpby;  /* FB_OP_CAXPBY = 1636 */
  fb_generic_fn zaxpby;  /* FB_OP_ZAXPBY = 1637 */
  fb_generic_fn saxpy_batch;  /* FB_OP_SAXPY_BATCH = 1638 */
  fb_generic_fn daxpy_batch;  /* FB_OP_DAXPY_BATCH = 1639 */
  fb_generic_fn caxpy_batch;  /* FB_OP_CAXPY_BATCH = 1640 */
  fb_generic_fn zaxpy_batch;  /* FB_OP_ZAXPY_BATCH = 1641 */
  fb_generic_fn saxpy_batch_strided;  /* FB_OP_SAXPY_BATCH_STRIDED = 1642 */
  fb_generic_fn daxpy_batch_strided;  /* FB_OP_DAXPY_BATCH_STRIDED = 1643 */
  fb_generic_fn caxpy_batch_strided;  /* FB_OP_CAXPY_BATCH_STRIDED = 1644 */
  fb_generic_fn zaxpy_batch_strided;  /* FB_OP_ZAXPY_BATCH_STRIDED = 1645 */
  fb_generic_fn cblas_caxpby;  /* FB_OP_CBLAS_CAXPBY = 1646 */
  fb_generic_fn cblas_caxpy_batch;  /* FB_OP_CBLAS_CAXPY_BATCH = 1647 */
  fb_generic_fn cblas_caxpy_batch_strided;  /* FB_OP_CBLAS_CAXPY_BATCH_STRIDED = 1648 */
  fb_generic_fn cblas_ccopy_batch;  /* FB_OP_CBLAS_CCOPY_BATCH = 1649 */
  fb_generic_fn cblas_ccopy_batch_strided;  /* FB_OP_CBLAS_CCOPY_BATCH_STRIDED = 1650 */
  fb_generic_fn cblas_cdgmm_batch;  /* FB_OP_CBLAS_CDGMM_BATCH = 1651 */
  fb_generic_fn cblas_cdgmm_batch_strided;  /* FB_OP_CBLAS_CDGMM_BATCH_STRIDED = 1652 */
  fb_generic_fn cblas_cgemm3m_batch;  /* FB_OP_CBLAS_CGEMM3M_BATCH = 1653 */
  fb_generic_fn cblas_cgemm3m_batch_strided;  /* FB_OP_CBLAS_CGEMM3M_BATCH_STRIDED = 1654 */
  fb_generic_fn cblas_cgemmt;  /* FB_OP_CBLAS_CGEMMT = 1655 */
  fb_generic_fn cblas_cgemm_batch;  /* FB_OP_CBLAS_CGEMM_BATCH = 1656 */
  fb_generic_fn cblas_cgemm_batch_strided;  /* FB_OP_CBLAS_CGEMM_BATCH_STRIDED = 1657 */
  fb_generic_fn cblas_cgemm_compute;  /* FB_OP_CBLAS_CGEMM_COMPUTE = 1658 */
  fb_generic_fn cblas_cgemm_pack;  /* FB_OP_CBLAS_CGEMM_PACK = 1659 */
  fb_generic_fn cblas_cgemm_pack_get_size;  /* FB_OP_CBLAS_CGEMM_PACK_GET_SIZE = 1660 */
  fb_generic_fn cblas_cgemv_batch;  /* FB_OP_CBLAS_CGEMV_BATCH = 1661 */
  fb_generic_fn cblas_cgemv_batch_strided;  /* FB_OP_CBLAS_CGEMV_BATCH_STRIDED = 1662 */
  fb_generic_fn cblas_csymm_batch;  /* FB_OP_CBLAS_CSYMM_BATCH = 1663 */
  fb_generic_fn cblas_csyr2k_batch;  /* FB_OP_CBLAS_CSYR2K_BATCH = 1664 */
  fb_generic_fn cblas_csyrk_batch;  /* FB_OP_CBLAS_CSYRK_BATCH = 1665 */
  fb_generic_fn cblas_ctrsm_batch;  /* FB_OP_CBLAS_CTRSM_BATCH = 1666 */
  fb_generic_fn cblas_ctrsm_batch_strided;  /* FB_OP_CBLAS_CTRSM_BATCH_STRIDED = 1667 */
  fb_generic_fn cblas_daxpby;  /* FB_OP_CBLAS_DAXPBY = 1668 */
  fb_generic_fn cblas_daxpy_batch;  /* FB_OP_CBLAS_DAXPY_BATCH = 1669 */
  fb_generic_fn cblas_daxpy_batch_strided;  /* FB_OP_CBLAS_DAXPY_BATCH_STRIDED = 1670 */
  fb_generic_fn cblas_dcopy_batch;  /* FB_OP_CBLAS_DCOPY_BATCH = 1671 */
  fb_generic_fn cblas_dcopy_batch_strided;  /* FB_OP_CBLAS_DCOPY_BATCH_STRIDED = 1672 */
  fb_generic_fn cblas_ddgmm_batch;  /* FB_OP_CBLAS_DDGMM_BATCH = 1673 */
  fb_generic_fn cblas_ddgmm_batch_strided;  /* FB_OP_CBLAS_DDGMM_BATCH_STRIDED = 1674 */
  fb_generic_fn cblas_dgemm3m_batch;  /* FB_OP_CBLAS_DGEMM3M_BATCH = 1675 */
  fb_generic_fn cblas_dgemm3m_batch_strided;  /* FB_OP_CBLAS_DGEMM3M_BATCH_STRIDED = 1676 */
  fb_generic_fn cblas_dgemmt;  /* FB_OP_CBLAS_DGEMMT = 1677 */
  fb_generic_fn cblas_dgemm_batch;  /* FB_OP_CBLAS_DGEMM_BATCH = 1678 */
  fb_generic_fn cblas_dgemm_batch_strided;  /* FB_OP_CBLAS_DGEMM_BATCH_STRIDED = 1679 */
  fb_generic_fn cblas_dgemm_compute;  /* FB_OP_CBLAS_DGEMM_COMPUTE = 1680 */
  fb_generic_fn cblas_dgemm_pack;  /* FB_OP_CBLAS_DGEMM_PACK = 1681 */
  fb_generic_fn cblas_dgemm_pack_get_size;  /* FB_OP_CBLAS_DGEMM_PACK_GET_SIZE = 1682 */
  fb_generic_fn cblas_dgemv_batch;  /* FB_OP_CBLAS_DGEMV_BATCH = 1683 */
  fb_generic_fn cblas_dgemv_batch_strided;  /* FB_OP_CBLAS_DGEMV_BATCH_STRIDED = 1684 */
  fb_generic_fn cblas_dsymm_batch;  /* FB_OP_CBLAS_DSYMM_BATCH = 1685 */
  fb_generic_fn cblas_dsyr2k_batch;  /* FB_OP_CBLAS_DSYR2K_BATCH = 1686 */
  fb_generic_fn cblas_dsyrk_batch;  /* FB_OP_CBLAS_DSYRK_BATCH = 1687 */
  fb_generic_fn cblas_dtrsm_batch;  /* FB_OP_CBLAS_DTRSM_BATCH = 1688 */
  fb_generic_fn cblas_dtrsm_batch_strided;  /* FB_OP_CBLAS_DTRSM_BATCH_STRIDED = 1689 */
  fb_generic_fn cblas_gemm_bf16bf16f32;  /* FB_OP_CBLAS_GEMM_BF16BF16F32 = 1690 */
  fb_generic_fn cblas_gemm_e4m3e4m3f32;  /* FB_OP_CBLAS_GEMM_E4M3E4M3F32 = 1691 */
  fb_generic_fn cblas_gemm_e5m2e5m2f32;  /* FB_OP_CBLAS_GEMM_E5M2E5M2F32 = 1692 */
  fb_generic_fn cblas_gemm_f16f16f32;  /* FB_OP_CBLAS_GEMM_F16F16F32 = 1693 */
  fb_generic_fn cblas_gemm_s8s8s32;  /* FB_OP_CBLAS_GEMM_S8S8S32 = 1694 */
  fb_generic_fn cblas_gemm_s8u8s32;  /* FB_OP_CBLAS_GEMM_S8U8S32 = 1695 */
  fb_generic_fn cblas_saxpby;  /* FB_OP_CBLAS_SAXPBY = 1696 */
  fb_generic_fn cblas_saxpy_batch;  /* FB_OP_CBLAS_SAXPY_BATCH = 1697 */
  fb_generic_fn cblas_saxpy_batch_strided;  /* FB_OP_CBLAS_SAXPY_BATCH_STRIDED = 1698 */
  fb_generic_fn cblas_scopy_batch;  /* FB_OP_CBLAS_SCOPY_BATCH = 1699 */
  fb_generic_fn cblas_scopy_batch_strided;  /* FB_OP_CBLAS_SCOPY_BATCH_STRIDED = 1700 */
  fb_generic_fn cblas_sdgmm_batch;  /* FB_OP_CBLAS_SDGMM_BATCH = 1701 */
  fb_generic_fn cblas_sdgmm_batch_strided;  /* FB_OP_CBLAS_SDGMM_BATCH_STRIDED = 1702 */
  fb_generic_fn cblas_sgemm3m_batch;  /* FB_OP_CBLAS_SGEMM3M_BATCH = 1703 */
  fb_generic_fn cblas_sgemm3m_batch_strided;  /* FB_OP_CBLAS_SGEMM3M_BATCH_STRIDED = 1704 */
  fb_generic_fn cblas_sgemmt;  /* FB_OP_CBLAS_SGEMMT = 1705 */
  fb_generic_fn cblas_sgemm_batch;  /* FB_OP_CBLAS_SGEMM_BATCH = 1706 */
  fb_generic_fn cblas_sgemm_batch_strided;  /* FB_OP_CBLAS_SGEMM_BATCH_STRIDED = 1707 */
  fb_generic_fn cblas_sgemm_compute;  /* FB_OP_CBLAS_SGEMM_COMPUTE = 1708 */
  fb_generic_fn cblas_sgemm_pack;  /* FB_OP_CBLAS_SGEMM_PACK = 1709 */
  fb_generic_fn cblas_sgemm_pack_get_size;  /* FB_OP_CBLAS_SGEMM_PACK_GET_SIZE = 1710 */
  fb_generic_fn cblas_sgemv_batch;  /* FB_OP_CBLAS_SGEMV_BATCH = 1711 */
  fb_generic_fn cblas_sgemv_batch_strided;  /* FB_OP_CBLAS_SGEMV_BATCH_STRIDED = 1712 */
  fb_generic_fn cblas_ssymm_batch;  /* FB_OP_CBLAS_SSYMM_BATCH = 1713 */
  fb_generic_fn cblas_ssyr2k_batch;  /* FB_OP_CBLAS_SSYR2K_BATCH = 1714 */
  fb_generic_fn cblas_ssyrk_batch;  /* FB_OP_CBLAS_SSYRK_BATCH = 1715 */
  fb_generic_fn cblas_strsm_batch;  /* FB_OP_CBLAS_STRSM_BATCH = 1716 */
  fb_generic_fn cblas_strsm_batch_strided;  /* FB_OP_CBLAS_STRSM_BATCH_STRIDED = 1717 */
  fb_generic_fn cblas_zaxpby;  /* FB_OP_CBLAS_ZAXPBY = 1718 */
  fb_generic_fn cblas_zaxpy_batch;  /* FB_OP_CBLAS_ZAXPY_BATCH = 1719 */
  fb_generic_fn cblas_zaxpy_batch_strided;  /* FB_OP_CBLAS_ZAXPY_BATCH_STRIDED = 1720 */
  fb_generic_fn cblas_zcopy_batch;  /* FB_OP_CBLAS_ZCOPY_BATCH = 1721 */
  fb_generic_fn cblas_zcopy_batch_strided;  /* FB_OP_CBLAS_ZCOPY_BATCH_STRIDED = 1722 */
  fb_generic_fn cblas_zdgmm_batch;  /* FB_OP_CBLAS_ZDGMM_BATCH = 1723 */
  fb_generic_fn cblas_zdgmm_batch_strided;  /* FB_OP_CBLAS_ZDGMM_BATCH_STRIDED = 1724 */
  fb_generic_fn cblas_zgemm3m_batch;  /* FB_OP_CBLAS_ZGEMM3M_BATCH = 1725 */
  fb_generic_fn cblas_zgemm3m_batch_strided;  /* FB_OP_CBLAS_ZGEMM3M_BATCH_STRIDED = 1726 */
  fb_generic_fn cblas_zgemmt;  /* FB_OP_CBLAS_ZGEMMT = 1727 */
  fb_generic_fn cblas_zgemm_batch;  /* FB_OP_CBLAS_ZGEMM_BATCH = 1728 */
  fb_generic_fn cblas_zgemm_batch_strided;  /* FB_OP_CBLAS_ZGEMM_BATCH_STRIDED = 1729 */
  fb_generic_fn cblas_zgemm_compute;  /* FB_OP_CBLAS_ZGEMM_COMPUTE = 1730 */
  fb_generic_fn cblas_zgemm_pack;  /* FB_OP_CBLAS_ZGEMM_PACK = 1731 */
  fb_generic_fn cblas_zgemm_pack_get_size;  /* FB_OP_CBLAS_ZGEMM_PACK_GET_SIZE = 1732 */
  fb_generic_fn cblas_zgemv_batch;  /* FB_OP_CBLAS_ZGEMV_BATCH = 1733 */
  fb_generic_fn cblas_zgemv_batch_strided;  /* FB_OP_CBLAS_ZGEMV_BATCH_STRIDED = 1734 */
  fb_generic_fn cblas_zsymm_batch;  /* FB_OP_CBLAS_ZSYMM_BATCH = 1735 */
  fb_generic_fn cblas_zsyr2k_batch;  /* FB_OP_CBLAS_ZSYR2K_BATCH = 1736 */
  fb_generic_fn cblas_zsyrk_batch;  /* FB_OP_CBLAS_ZSYRK_BATCH = 1737 */
  fb_generic_fn cblas_ztrsm_batch;  /* FB_OP_CBLAS_ZTRSM_BATCH = 1738 */
  fb_generic_fn cblas_ztrsm_batch_strided;  /* FB_OP_CBLAS_ZTRSM_BATCH_STRIDED = 1739 */
  fb_generic_fn scopy_batch;  /* FB_OP_SCOPY_BATCH = 1740 */
  fb_generic_fn dcopy_batch;  /* FB_OP_DCOPY_BATCH = 1741 */
  fb_generic_fn ccopy_batch;  /* FB_OP_CCOPY_BATCH = 1742 */
  fb_generic_fn zcopy_batch;  /* FB_OP_ZCOPY_BATCH = 1743 */
  fb_generic_fn scopy_batch_strided;  /* FB_OP_SCOPY_BATCH_STRIDED = 1744 */
  fb_generic_fn dcopy_batch_strided;  /* FB_OP_DCOPY_BATCH_STRIDED = 1745 */
  fb_generic_fn ccopy_batch_strided;  /* FB_OP_CCOPY_BATCH_STRIDED = 1746 */
  fb_generic_fn zcopy_batch_strided;  /* FB_OP_ZCOPY_BATCH_STRIDED = 1747 */
  fb_generic_fn sdgmm_batch;  /* FB_OP_SDGMM_BATCH = 1748 */
  fb_generic_fn ddgmm_batch;  /* FB_OP_DDGMM_BATCH = 1749 */
  fb_generic_fn cdgmm_batch;  /* FB_OP_CDGMM_BATCH = 1750 */
  fb_generic_fn zdgmm_batch;  /* FB_OP_ZDGMM_BATCH = 1751 */
  fb_generic_fn sdgmm_batch_strided;  /* FB_OP_SDGMM_BATCH_STRIDED = 1752 */
  fb_generic_fn ddgmm_batch_strided;  /* FB_OP_DDGMM_BATCH_STRIDED = 1753 */
  fb_generic_fn cdgmm_batch_strided;  /* FB_OP_CDGMM_BATCH_STRIDED = 1754 */
  fb_generic_fn zdgmm_batch_strided;  /* FB_OP_ZDGMM_BATCH_STRIDED = 1755 */
  fb_generic_fn sgemm3m_batch;  /* FB_OP_SGEMM3M_BATCH = 1756 */
  fb_generic_fn dgemm3m_batch;  /* FB_OP_DGEMM3M_BATCH = 1757 */
  fb_generic_fn cgemm3m_batch;  /* FB_OP_CGEMM3M_BATCH = 1758 */
  fb_generic_fn zgemm3m_batch;  /* FB_OP_ZGEMM3M_BATCH = 1759 */
  fb_generic_fn sgemm3m_batch_strided;  /* FB_OP_SGEMM3M_BATCH_STRIDED = 1760 */
  fb_generic_fn dgemm3m_batch_strided;  /* FB_OP_DGEMM3M_BATCH_STRIDED = 1761 */
  fb_generic_fn cgemm3m_batch_strided;  /* FB_OP_CGEMM3M_BATCH_STRIDED = 1762 */
  fb_generic_fn zgemm3m_batch_strided;  /* FB_OP_ZGEMM3M_BATCH_STRIDED = 1763 */
  fb_generic_fn sgemv_batch;  /* FB_OP_SGEMV_BATCH = 1764 */
  fb_generic_fn dgemv_batch;  /* FB_OP_DGEMV_BATCH = 1765 */
  fb_generic_fn cgemv_batch;  /* FB_OP_CGEMV_BATCH = 1766 */
  fb_generic_fn zgemv_batch;  /* FB_OP_ZGEMV_BATCH = 1767 */
  fb_generic_fn sgemv_batch_strided;  /* FB_OP_SGEMV_BATCH_STRIDED = 1768 */
  fb_generic_fn dgemv_batch_strided;  /* FB_OP_DGEMV_BATCH_STRIDED = 1769 */
  fb_generic_fn cgemv_batch_strided;  /* FB_OP_CGEMV_BATCH_STRIDED = 1770 */
  fb_generic_fn zgemv_batch_strided;  /* FB_OP_ZGEMV_BATCH_STRIDED = 1771 */
  fb_generic_fn simatcopy_batch;  /* FB_OP_SIMATCOPY_BATCH = 1772 */
  fb_generic_fn dimatcopy_batch;  /* FB_OP_DIMATCOPY_BATCH = 1773 */
  fb_generic_fn cimatcopy_batch;  /* FB_OP_CIMATCOPY_BATCH = 1774 */
  fb_generic_fn zimatcopy_batch;  /* FB_OP_ZIMATCOPY_BATCH = 1775 */
  fb_generic_fn simatcopy_batch_strided;  /* FB_OP_SIMATCOPY_BATCH_STRIDED = 1776 */
  fb_generic_fn dimatcopy_batch_strided;  /* FB_OP_DIMATCOPY_BATCH_STRIDED = 1777 */
  fb_generic_fn cimatcopy_batch_strided;  /* FB_OP_CIMATCOPY_BATCH_STRIDED = 1778 */
  fb_generic_fn zimatcopy_batch_strided;  /* FB_OP_ZIMATCOPY_BATCH_STRIDED = 1779 */
  fb_generic_fn somatcopy_batch;  /* FB_OP_SOMATCOPY_BATCH = 1780 */
  fb_generic_fn domatcopy_batch;  /* FB_OP_DOMATCOPY_BATCH = 1781 */
  fb_generic_fn comatcopy_batch;  /* FB_OP_COMATCOPY_BATCH = 1782 */
  fb_generic_fn zomatcopy_batch;  /* FB_OP_ZOMATCOPY_BATCH = 1783 */
  fb_generic_fn somatcopy_batch_strided;  /* FB_OP_SOMATCOPY_BATCH_STRIDED = 1784 */
  fb_generic_fn domatcopy_batch_strided;  /* FB_OP_DOMATCOPY_BATCH_STRIDED = 1785 */
  fb_generic_fn comatcopy_batch_strided;  /* FB_OP_COMATCOPY_BATCH_STRIDED = 1786 */
  fb_generic_fn zomatcopy_batch_strided;  /* FB_OP_ZOMATCOPY_BATCH_STRIDED = 1787 */
  fb_generic_fn ssymm_batch;  /* FB_OP_SSYMM_BATCH = 1788 */
  fb_generic_fn dsymm_batch;  /* FB_OP_DSYMM_BATCH = 1789 */
  fb_generic_fn csymm_batch;  /* FB_OP_CSYMM_BATCH = 1790 */
  fb_generic_fn zsymm_batch;  /* FB_OP_ZSYMM_BATCH = 1791 */
  fb_generic_fn ssyr2k_batch;  /* FB_OP_SSYR2K_BATCH = 1792 */
  fb_generic_fn dsyr2k_batch;  /* FB_OP_DSYR2K_BATCH = 1793 */
  fb_generic_fn csyr2k_batch;  /* FB_OP_CSYR2K_BATCH = 1794 */
  fb_generic_fn zsyr2k_batch;  /* FB_OP_ZSYR2K_BATCH = 1795 */
  fb_generic_fn ssyrk_batch;  /* FB_OP_SSYRK_BATCH = 1796 */
  fb_generic_fn dsyrk_batch;  /* FB_OP_DSYRK_BATCH = 1797 */
  fb_generic_fn csyrk_batch;  /* FB_OP_CSYRK_BATCH = 1798 */
  fb_generic_fn zsyrk_batch;  /* FB_OP_ZSYRK_BATCH = 1799 */
  fb_generic_fn strsm_batch;  /* FB_OP_STRSM_BATCH = 1800 */
  fb_generic_fn dtrsm_batch;  /* FB_OP_DTRSM_BATCH = 1801 */
  fb_generic_fn ctrsm_batch;  /* FB_OP_CTRSM_BATCH = 1802 */
  fb_generic_fn ztrsm_batch;  /* FB_OP_ZTRSM_BATCH = 1803 */
  fb_generic_fn strsm_batch_strided;  /* FB_OP_STRSM_BATCH_STRIDED = 1804 */
  fb_generic_fn dtrsm_batch_strided;  /* FB_OP_DTRSM_BATCH_STRIDED = 1805 */
  fb_generic_fn ctrsm_batch_strided;  /* FB_OP_CTRSM_BATCH_STRIDED = 1806 */
  fb_generic_fn ztrsm_batch_strided;  /* FB_OP_ZTRSM_BATCH_STRIDED = 1807 */

  /* ========================================================================
   * MKL extensions (mkl_jit, mkl_*omatcopy, mkl_sparse_*)
   * ========================================================================
   * fb_generic_fn fields: backends assign  vtable->op = (fb_generic_fn)impl;
   * fb_vtable_sync_ext_ops() mirrors them into ext_ops[] automatically. */

  fb_generic_fn mkl_cimatcopy;  /* FB_OP_MKL_CIMATCOPY = 1808 */
  fb_generic_fn mkl_cimatcopy_batch;  /* FB_OP_MKL_CIMATCOPY_BATCH = 1809 */
  fb_generic_fn mkl_cimatcopy_batch_strided;  /* FB_OP_MKL_CIMATCOPY_BATCH_STRIDED = 1810 */
  fb_generic_fn mkl_comatadd;  /* FB_OP_MKL_COMATADD = 1811 */
  fb_generic_fn mkl_comatcopy;  /* FB_OP_MKL_COMATCOPY = 1812 */
  fb_generic_fn mkl_comatcopy2;  /* FB_OP_MKL_COMATCOPY2 = 1813 */
  fb_generic_fn mkl_comatcopy_batch;  /* FB_OP_MKL_COMATCOPY_BATCH = 1814 */
  fb_generic_fn mkl_comatcopy_batch_strided;  /* FB_OP_MKL_COMATCOPY_BATCH_STRIDED = 1815 */
  fb_generic_fn mkl_dimatcopy;  /* FB_OP_MKL_DIMATCOPY = 1816 */
  fb_generic_fn mkl_dimatcopy_batch;  /* FB_OP_MKL_DIMATCOPY_BATCH = 1817 */
  fb_generic_fn mkl_dimatcopy_batch_strided;  /* FB_OP_MKL_DIMATCOPY_BATCH_STRIDED = 1818 */
  fb_generic_fn mkl_domatadd;  /* FB_OP_MKL_DOMATADD = 1819 */
  fb_generic_fn mkl_domatcopy;  /* FB_OP_MKL_DOMATCOPY = 1820 */
  fb_generic_fn mkl_domatcopy2;  /* FB_OP_MKL_DOMATCOPY2 = 1821 */
  fb_generic_fn mkl_domatcopy_batch;  /* FB_OP_MKL_DOMATCOPY_BATCH = 1822 */
  fb_generic_fn mkl_domatcopy_batch_strided;  /* FB_OP_MKL_DOMATCOPY_BATCH_STRIDED = 1823 */
  fb_generic_fn mkl_jit_create_cgemm;  /* FB_OP_MKL_JIT_CREATE_CGEMM = 1824 */
  fb_generic_fn mkl_jit_create_dgemm;  /* FB_OP_MKL_JIT_CREATE_DGEMM = 1825 */
  fb_generic_fn mkl_jit_create_sgemm;  /* FB_OP_MKL_JIT_CREATE_SGEMM = 1826 */
  fb_generic_fn mkl_jit_create_zgemm;  /* FB_OP_MKL_JIT_CREATE_ZGEMM = 1827 */
  fb_generic_fn mkl_jit_destroy;  /* FB_OP_MKL_JIT_DESTROY = 1828 */
  fb_generic_fn mkl_jit_get_cgemm_ptr;  /* FB_OP_MKL_JIT_GET_CGEMM_PTR = 1829 */
  fb_generic_fn mkl_jit_get_dgemm_ptr;  /* FB_OP_MKL_JIT_GET_DGEMM_PTR = 1830 */
  fb_generic_fn mkl_jit_get_sgemm_ptr;  /* FB_OP_MKL_JIT_GET_SGEMM_PTR = 1831 */
  fb_generic_fn mkl_jit_get_zgemm_ptr;  /* FB_OP_MKL_JIT_GET_ZGEMM_PTR = 1832 */
  fb_generic_fn mkl_simatcopy;  /* FB_OP_MKL_SIMATCOPY = 1833 */
  fb_generic_fn mkl_simatcopy_batch;  /* FB_OP_MKL_SIMATCOPY_BATCH = 1834 */
  fb_generic_fn mkl_simatcopy_batch_strided;  /* FB_OP_MKL_SIMATCOPY_BATCH_STRIDED = 1835 */
  fb_generic_fn mkl_somatadd;  /* FB_OP_MKL_SOMATADD = 1836 */
  fb_generic_fn mkl_somatcopy;  /* FB_OP_MKL_SOMATCOPY = 1837 */
  fb_generic_fn mkl_somatcopy2;  /* FB_OP_MKL_SOMATCOPY2 = 1838 */
  fb_generic_fn mkl_somatcopy_batch;  /* FB_OP_MKL_SOMATCOPY_BATCH = 1839 */
  fb_generic_fn mkl_somatcopy_batch_strided;  /* FB_OP_MKL_SOMATCOPY_BATCH_STRIDED = 1840 */
  fb_generic_fn mkl_sparse_s_create_csr;  /* FB_OP_MKL_SPARSE_S_CREATE_CSR = 1841 */
  fb_generic_fn mkl_sparse_s_mv;  /* FB_OP_MKL_SPARSE_S_MV = 1842 */
  fb_generic_fn mkl_zimatcopy;  /* FB_OP_MKL_ZIMATCOPY = 1843 */
  fb_generic_fn mkl_zimatcopy_batch;  /* FB_OP_MKL_ZIMATCOPY_BATCH = 1844 */
  fb_generic_fn mkl_zimatcopy_batch_strided;  /* FB_OP_MKL_ZIMATCOPY_BATCH_STRIDED = 1845 */
  fb_generic_fn mkl_zomatadd;  /* FB_OP_MKL_ZOMATADD = 1846 */
  fb_generic_fn mkl_zomatcopy;  /* FB_OP_MKL_ZOMATCOPY = 1847 */
  fb_generic_fn mkl_zomatcopy2;  /* FB_OP_MKL_ZOMATCOPY2 = 1848 */
  fb_generic_fn mkl_zomatcopy_batch;  /* FB_OP_MKL_ZOMATCOPY_BATCH = 1849 */
  fb_generic_fn mkl_zomatcopy_batch_strided;  /* FB_OP_MKL_ZOMATCOPY_BATCH_STRIDED = 1850 */

  /* ========================================================================
   * Deep Neural Network primitives (fb_dnn_*)
   * ========================================================================
   * fb_generic_fn fields: backends assign  vtable->op = (fb_generic_fn)impl;
   * fb_vtable_sync_ext_ops() mirrors them into ext_ops[] automatically. */

  fb_generic_fn fb_dnn_adaptivepool;  /* FB_OP_FB_DNN_ADAPTIVEPOOL = 1851 */
  fb_generic_fn fb_dnn_avgpool_backward;  /* FB_OP_FB_DNN_AVGPOOL_BACKWARD = 1852 */
  fb_generic_fn fb_dnn_avgpool_forward;  /* FB_OP_FB_DNN_AVGPOOL_FORWARD = 1853 */
  fb_generic_fn fb_dnn_batchnorm_backward;  /* FB_OP_FB_DNN_BATCHNORM_BACKWARD = 1854 */
  fb_generic_fn fb_dnn_batchnorm_forward_inference;  /* FB_OP_FB_DNN_BATCHNORM_FORWARD_INFERENCE = 1855 */
  fb_generic_fn fb_dnn_batchnorm_forward_training;  /* FB_OP_FB_DNN_BATCHNORM_FORWARD_TRAINING = 1856 */
  fb_generic_fn fb_dnn_conv2d_forward;  /* FB_OP_FB_DNN_CONV2D_FORWARD = 1857 */
  fb_generic_fn fb_dnn_conv3d_forward;  /* FB_OP_FB_DNN_CONV3D_FORWARD = 1858 */
  fb_generic_fn fb_dnn_conv_backward_bias;  /* FB_OP_FB_DNN_CONV_BACKWARD_BIAS = 1859 */
  fb_generic_fn fb_dnn_conv_backward_data;  /* FB_OP_FB_DNN_CONV_BACKWARD_DATA = 1860 */
  fb_generic_fn fb_dnn_conv_backward_filter;  /* FB_OP_FB_DNN_CONV_BACKWARD_FILTER = 1861 */
  fb_generic_fn fb_dnn_conv_batchnorm_relu;  /* FB_OP_FB_DNN_CONV_BATCHNORM_RELU = 1862 */
  fb_generic_fn fb_dnn_conv_bias_relu;  /* FB_OP_FB_DNN_CONV_BIAS_RELU = 1863 */
  fb_generic_fn fb_dnn_conv_depthwise;  /* FB_OP_FB_DNN_CONV_DEPTHWISE = 1864 */
  fb_generic_fn fb_dnn_conv_dilated;  /* FB_OP_FB_DNN_CONV_DILATED = 1865 */
  fb_generic_fn fb_dnn_conv_find_algorithm;  /* FB_OP_FB_DNN_CONV_FIND_ALGORITHM = 1866 */
  fb_generic_fn fb_dnn_conv_grouped;  /* FB_OP_FB_DNN_CONV_GROUPED = 1867 */
  fb_generic_fn fb_dnn_conv_transposed;  /* FB_OP_FB_DNN_CONV_TRANSPOSED = 1868 */
  fb_generic_fn fb_dnn_dropout_backward;  /* FB_OP_FB_DNN_DROPOUT_BACKWARD = 1869 */
  fb_generic_fn fb_dnn_dropout_forward;  /* FB_OP_FB_DNN_DROPOUT_FORWARD = 1870 */
  fb_generic_fn fb_dnn_elu;  /* FB_OP_FB_DNN_ELU = 1871 */
  fb_generic_fn fb_dnn_flash_attention_backward;  /* FB_OP_FB_DNN_FLASH_ATTENTION_BACKWARD = 1872 */
  fb_generic_fn fb_dnn_flash_attention_forward;  /* FB_OP_FB_DNN_FLASH_ATTENTION_FORWARD = 1873 */
  fb_generic_fn fb_dnn_gelu;  /* FB_OP_FB_DNN_GELU = 1874 */
  fb_generic_fn fb_dnn_globalpool;  /* FB_OP_FB_DNN_GLOBALPOOL = 1875 */
  fb_generic_fn fb_dnn_groupnorm_forward;  /* FB_OP_FB_DNN_GROUPNORM_FORWARD = 1876 */
  fb_generic_fn fb_dnn_gru_forward;  /* FB_OP_FB_DNN_GRU_FORWARD = 1877 */
  fb_generic_fn fb_dnn_instancenorm_forward;  /* FB_OP_FB_DNN_INSTANCENORM_FORWARD = 1878 */
  fb_generic_fn fb_dnn_layernorm_backward;  /* FB_OP_FB_DNN_LAYERNORM_BACKWARD = 1879 */
  fb_generic_fn fb_dnn_layernorm_forward;  /* FB_OP_FB_DNN_LAYERNORM_FORWARD = 1880 */
  fb_generic_fn fb_dnn_leaky_relu;  /* FB_OP_FB_DNN_LEAKY_RELU = 1881 */
  fb_generic_fn fb_dnn_linear_gelu;  /* FB_OP_FB_DNN_LINEAR_GELU = 1882 */
  fb_generic_fn fb_dnn_linear_relu;  /* FB_OP_FB_DNN_LINEAR_RELU = 1883 */
  fb_generic_fn fb_dnn_logsoftmax_backward;  /* FB_OP_FB_DNN_LOGSOFTMAX_BACKWARD = 1884 */
  fb_generic_fn fb_dnn_logsoftmax_forward;  /* FB_OP_FB_DNN_LOGSOFTMAX_FORWARD = 1885 */
  fb_generic_fn fb_dnn_lstm_forward;  /* FB_OP_FB_DNN_LSTM_FORWARD = 1886 */
  fb_generic_fn fb_dnn_maxpool_backward;  /* FB_OP_FB_DNN_MAXPOOL_BACKWARD = 1887 */
  fb_generic_fn fb_dnn_maxpool_forward;  /* FB_OP_FB_DNN_MAXPOOL_FORWARD = 1888 */
  fb_generic_fn fb_dnn_mish;  /* FB_OP_FB_DNN_MISH = 1889 */
  fb_generic_fn fb_dnn_multi_head_attention;  /* FB_OP_FB_DNN_MULTI_HEAD_ATTENTION = 1890 */
  fb_generic_fn fb_dnn_relu;  /* FB_OP_FB_DNN_RELU = 1891 */
  fb_generic_fn fb_dnn_relu_backward;  /* FB_OP_FB_DNN_RELU_BACKWARD = 1892 */
  fb_generic_fn fb_dnn_residual_block;  /* FB_OP_FB_DNN_RESIDUAL_BLOCK = 1893 */
  fb_generic_fn fb_dnn_rnn_backward_data;  /* FB_OP_FB_DNN_RNN_BACKWARD_DATA = 1894 */
  fb_generic_fn fb_dnn_rnn_backward_weights;  /* FB_OP_FB_DNN_RNN_BACKWARD_WEIGHTS = 1895 */
  fb_generic_fn fb_dnn_rnn_forward;  /* FB_OP_FB_DNN_RNN_FORWARD = 1896 */
  fb_generic_fn fb_dnn_scaled_dot_product_attention;  /* FB_OP_FB_DNN_SCALED_DOT_PRODUCT_ATTENTION = 1897 */
  fb_generic_fn fb_dnn_sigmoid;  /* FB_OP_FB_DNN_SIGMOID = 1898 */
  fb_generic_fn fb_dnn_softmax_backward;  /* FB_OP_FB_DNN_SOFTMAX_BACKWARD = 1899 */
  fb_generic_fn fb_dnn_softmax_forward;  /* FB_OP_FB_DNN_SOFTMAX_FORWARD = 1900 */
  fb_generic_fn fb_dnn_softplus;  /* FB_OP_FB_DNN_SOFTPLUS = 1901 */
  fb_generic_fn fb_dnn_swish;  /* FB_OP_FB_DNN_SWISH = 1902 */
  fb_generic_fn fb_dnn_tanh;  /* FB_OP_FB_DNN_TANH = 1903 */

  /* ========================================================================
   * FFT (fb_fft_*)
   * ========================================================================
   * fb_generic_fn fields: backends assign  vtable->op = (fb_generic_fn)impl;
   * fb_vtable_sync_ext_ops() mirrors them into ext_ops[] automatically. */

  fb_generic_fn fb_fft_clear_callbacks;  /* FB_OP_FB_FFT_CLEAR_CALLBACKS = 1904 */
  fb_generic_fn fb_fft_commit;  /* FB_OP_FB_FFT_COMMIT = 1905 */
  fb_generic_fn fb_fft_create_plan_1d;  /* FB_OP_FB_FFT_CREATE_PLAN_1D = 1906 */
  fb_generic_fn fb_fft_create_plan_2d;  /* FB_OP_FB_FFT_CREATE_PLAN_2D = 1907 */
  fb_generic_fn fb_fft_create_plan_3d;  /* FB_OP_FB_FFT_CREATE_PLAN_3D = 1908 */
  fb_generic_fn fb_fft_create_plan_many;  /* FB_OP_FB_FFT_CREATE_PLAN_MANY = 1909 */
  fb_generic_fn fb_fft_destroy;  /* FB_OP_FB_FFT_DESTROY = 1910 */
  fb_generic_fn fb_fft_estimate_workspace_1d;  /* FB_OP_FB_FFT_ESTIMATE_WORKSPACE_1D = 1911 */
  fb_generic_fn fb_fft_execute_backward_c2c;  /* FB_OP_FB_FFT_EXECUTE_BACKWARD_C2C = 1912 */
  fb_generic_fn fb_fft_execute_backward_c2r;  /* FB_OP_FB_FFT_EXECUTE_BACKWARD_C2R = 1913 */
  fb_generic_fn fb_fft_execute_backward_z2d;  /* FB_OP_FB_FFT_EXECUTE_BACKWARD_Z2D = 1914 */
  fb_generic_fn fb_fft_execute_backward_z2z;  /* FB_OP_FB_FFT_EXECUTE_BACKWARD_Z2Z = 1915 */
  fb_generic_fn fb_fft_execute_forward_c2c;  /* FB_OP_FB_FFT_EXECUTE_FORWARD_C2C = 1916 */
  fb_generic_fn fb_fft_execute_forward_d2z;  /* FB_OP_FB_FFT_EXECUTE_FORWARD_D2Z = 1917 */
  fb_generic_fn fb_fft_execute_forward_r2c;  /* FB_OP_FB_FFT_EXECUTE_FORWARD_R2C = 1918 */
  fb_generic_fn fb_fft_execute_forward_z2z;  /* FB_OP_FB_FFT_EXECUTE_FORWARD_Z2Z = 1919 */
  fb_generic_fn fb_fft_execute_multi_gpu;  /* FB_OP_FB_FFT_EXECUTE_MULTI_GPU = 1920 */
  fb_generic_fn fb_fft_execute_round_trip_d2z2d;  /* FB_OP_FB_FFT_EXECUTE_ROUND_TRIP_D2Z2D = 1921 */
  fb_generic_fn fb_fft_execute_round_trip_r2c2r;  /* FB_OP_FB_FFT_EXECUTE_ROUND_TRIP_R2C2R = 1922 */
  fb_generic_fn fb_fft_free_distributed;  /* FB_OP_FB_FFT_FREE_DISTRIBUTED = 1923 */
  fb_generic_fn fb_fft_get_backend_name;  /* FB_OP_FB_FFT_GET_BACKEND_NAME = 1924 */
  fb_generic_fn fb_fft_get_max_dimensions;  /* FB_OP_FB_FFT_GET_MAX_DIMENSIONS = 1925 */
  fb_generic_fn fb_fft_get_optimal_size;  /* FB_OP_FB_FFT_GET_OPTIMAL_SIZE = 1926 */
  fb_generic_fn fb_fft_get_supported_transforms;  /* FB_OP_FB_FFT_GET_SUPPORTED_TRANSFORMS = 1927 */
  fb_generic_fn fb_fft_get_version;  /* FB_OP_FB_FFT_GET_VERSION = 1928 */
  fb_generic_fn fb_fft_get_workspace_size;  /* FB_OP_FB_FFT_GET_WORKSPACE_SIZE = 1929 */
  fb_generic_fn fb_fft_is_size_optimal;  /* FB_OP_FB_FFT_IS_SIZE_OPTIMAL = 1930 */
  fb_generic_fn fb_fft_malloc_distributed;  /* FB_OP_FB_FFT_MALLOC_DISTRIBUTED = 1931 */
  fb_generic_fn fb_fft_plan;  /* FB_OP_FB_FFT_PLAN = 1932 */
  fb_generic_fn fb_fft_set_gpus;  /* FB_OP_FB_FFT_SET_GPUS = 1933 */
  fb_generic_fn fb_fft_set_load_callback;  /* FB_OP_FB_FFT_SET_LOAD_CALLBACK = 1934 */
  fb_generic_fn fb_fft_set_normalization;  /* FB_OP_FB_FFT_SET_NORMALIZATION = 1935 */
  fb_generic_fn fb_fft_set_parameter;  /* FB_OP_FB_FFT_SET_PARAMETER = 1936 */
  fb_generic_fn fb_fft_set_placement;  /* FB_OP_FB_FFT_SET_PLACEMENT = 1937 */
  fb_generic_fn fb_fft_set_scale;  /* FB_OP_FB_FFT_SET_SCALE = 1938 */
  fb_generic_fn fb_fft_set_store_callback;  /* FB_OP_FB_FFT_SET_STORE_CALLBACK = 1939 */
  fb_generic_fn fb_fft_set_stream;  /* FB_OP_FB_FFT_SET_STREAM = 1940 */
  fb_generic_fn fb_fft_set_stride;  /* FB_OP_FB_FFT_SET_STRIDE = 1941 */
  fb_generic_fn fb_fft_wait_stream;  /* FB_OP_FB_FFT_WAIT_STREAM = 1942 */

  /* ========================================================================
   * Sparse linear algebra (fb_sparse_*)
   * ========================================================================
   * fb_generic_fn fields: backends assign  vtable->op = (fb_generic_fn)impl;
   * fb_vtable_sync_ext_ops() mirrors them into ext_ops[] automatically. */

  fb_generic_fn fb_sparse_add;  /* FB_OP_FB_SPARSE_ADD = 1943 */
  fb_generic_fn fb_sparse_bicg;  /* FB_OP_FB_SPARSE_BICG = 1944 */
  fb_generic_fn fb_sparse_bicgstab;  /* FB_OP_FB_SPARSE_BICGSTAB = 1945 */
  fb_generic_fn fb_sparse_cg;  /* FB_OP_FB_SPARSE_CG = 1946 */
  fb_generic_fn fb_sparse_convert_bsr;  /* FB_OP_FB_SPARSE_CONVERT_BSR = 1947 */
  fb_generic_fn fb_sparse_convert_coo;  /* FB_OP_FB_SPARSE_CONVERT_COO = 1948 */
  fb_generic_fn fb_sparse_convert_csr;  /* FB_OP_FB_SPARSE_CONVERT_CSR = 1949 */
  fb_generic_fn fb_sparse_copy;  /* FB_OP_FB_SPARSE_COPY = 1950 */
  fb_generic_fn fb_sparse_create_bsr;  /* FB_OP_FB_SPARSE_CREATE_BSR = 1951 */
  fb_generic_fn fb_sparse_create_coo;  /* FB_OP_FB_SPARSE_CREATE_COO = 1952 */
  fb_generic_fn fb_sparse_create_csc;  /* FB_OP_FB_SPARSE_CREATE_CSC = 1953 */
  fb_generic_fn fb_sparse_create_csr;  /* FB_OP_FB_SPARSE_CREATE_CSR = 1954 */
  fb_generic_fn fb_sparse_destroy;  /* FB_OP_FB_SPARSE_DESTROY = 1955 */
  fb_generic_fn fb_sparse_export;  /* FB_OP_FB_SPARSE_EXPORT = 1956 */
  fb_generic_fn fb_sparse_fgmres;  /* FB_OP_FB_SPARSE_FGMRES = 1957 */
  fb_generic_fn fb_sparse_get_format;  /* FB_OP_FB_SPARSE_GET_FORMAT = 1958 */
  fb_generic_fn fb_sparse_get_size;  /* FB_OP_FB_SPARSE_GET_SIZE = 1959 */
  fb_generic_fn fb_sparse_gmres;  /* FB_OP_FB_SPARSE_GMRES = 1960 */
  fb_generic_fn fb_sparse_hermm;  /* FB_OP_FB_SPARSE_HERMM = 1961 */
  fb_generic_fn fb_sparse_hermv;  /* FB_OP_FB_SPARSE_HERMV = 1962 */
  fb_generic_fn fb_sparse_minres;  /* FB_OP_FB_SPARSE_MINRES = 1963 */
  fb_generic_fn fb_sparse_mm;  /* FB_OP_FB_SPARSE_MM = 1964 */
  fb_generic_fn fb_sparse_mv;  /* FB_OP_FB_SPARSE_MV = 1965 */
  fb_generic_fn fb_sparse_optimize;  /* FB_OP_FB_SPARSE_OPTIMIZE = 1966 */
  fb_generic_fn fb_sparse_precond_apply;  /* FB_OP_FB_SPARSE_PRECOND_APPLY = 1967 */
  fb_generic_fn fb_sparse_precond_gs;  /* FB_OP_FB_SPARSE_PRECOND_GS = 1968 */
  fb_generic_fn fb_sparse_precond_ic0_create;  /* FB_OP_FB_SPARSE_PRECOND_IC0_CREATE = 1969 */
  fb_generic_fn fb_sparse_precond_ilu0_create;  /* FB_OP_FB_SPARSE_PRECOND_ILU0_CREATE = 1970 */
  fb_generic_fn fb_sparse_precond_ilup_create;  /* FB_OP_FB_SPARSE_PRECOND_ILUP_CREATE = 1971 */
  fb_generic_fn fb_sparse_precond_jacobi_create;  /* FB_OP_FB_SPARSE_PRECOND_JACOBI_CREATE = 1972 */
  fb_generic_fn fb_sparse_precond_trsv_lower;  /* FB_OP_FB_SPARSE_PRECOND_TRSV_LOWER = 1973 */
  fb_generic_fn fb_sparse_prune_to_2_4;  /* FB_OP_FB_SPARSE_PRUNE_TO_2_4 = 1974 */
  fb_generic_fn fb_sparse_qmr;  /* FB_OP_FB_SPARSE_QMR = 1975 */
  fb_generic_fn fb_sparse_set_diag_type;  /* FB_OP_FB_SPARSE_SET_DIAG_TYPE = 1976 */
  fb_generic_fn fb_sparse_set_fill_mode;  /* FB_OP_FB_SPARSE_SET_FILL_MODE = 1977 */
  fb_generic_fn fb_sparse_set_matrix_type;  /* FB_OP_FB_SPARSE_SET_MATRIX_TYPE = 1978 */
  fb_generic_fn fb_sparse_set_mm_hint;  /* FB_OP_FB_SPARSE_SET_MM_HINT = 1979 */
  fb_generic_fn fb_sparse_set_mv_hint;  /* FB_OP_FB_SPARSE_SET_MV_HINT = 1980 */
  fb_generic_fn fb_sparse_set_sv_hint;  /* FB_OP_FB_SPARSE_SET_SV_HINT = 1981 */
  fb_generic_fn fb_sparse_solver_destroy;  /* FB_OP_FB_SPARSE_SOLVER_DESTROY = 1982 */
  fb_generic_fn fb_sparse_solver_get_iters;  /* FB_OP_FB_SPARSE_SOLVER_GET_ITERS = 1983 */
  fb_generic_fn fb_sparse_solver_get_residual;  /* FB_OP_FB_SPARSE_SOLVER_GET_RESIDUAL = 1984 */
  fb_generic_fn fb_sparse_solver_init;  /* FB_OP_FB_SPARSE_SOLVER_INIT = 1985 */
  fb_generic_fn fb_sparse_solver_set_maxiter;  /* FB_OP_FB_SPARSE_SOLVER_SET_MAXITER = 1986 */
  fb_generic_fn fb_sparse_solver_set_precond;  /* FB_OP_FB_SPARSE_SOLVER_SET_PRECOND = 1987 */
  fb_generic_fn fb_sparse_solver_set_tol;  /* FB_OP_FB_SPARSE_SOLVER_SET_TOL = 1988 */
  fb_generic_fn fb_sparse_solver_solve;  /* FB_OP_FB_SPARSE_SOLVER_SOLVE = 1989 */
  fb_generic_fn fb_sparse_spmm;  /* FB_OP_FB_SPARSE_SPMM = 1990 */
  fb_generic_fn fb_sparse_spmv;  /* FB_OP_FB_SPARSE_SPMV = 1991 */
  fb_generic_fn fb_sparse_structured_create_2_4;  /* FB_OP_FB_SPARSE_STRUCTURED_CREATE_2_4 = 1992 */
  fb_generic_fn fb_sparse_structured_get_compression_ratio;  /* FB_OP_FB_SPARSE_STRUCTURED_GET_COMPRESSION_RATIO = 1993 */
  fb_generic_fn fb_sparse_symm;  /* FB_OP_FB_SPARSE_SYMM = 1994 */
  fb_generic_fn fb_sparse_symv;  /* FB_OP_FB_SPARSE_SYMV = 1995 */
  fb_generic_fn fb_sparse_syr2k;  /* FB_OP_FB_SPARSE_SYR2K = 1996 */
  fb_generic_fn fb_sparse_syrk;  /* FB_OP_FB_SPARSE_SYRK = 1997 */
  fb_generic_fn fb_sparse_tfqmr;  /* FB_OP_FB_SPARSE_TFQMR = 1998 */
  fb_generic_fn fb_sparse_trsm;  /* FB_OP_FB_SPARSE_TRSM = 1999 */
  fb_generic_fn fb_sparse_trsv;  /* FB_OP_FB_SPARSE_TRSV = 2000 */

  /* ========================================================================
   * Tensor operations + RNG + NCCL/RCCL collectives
   * ========================================================================
   * fb_generic_fn fields: backends assign  vtable->op = (fb_generic_fn)impl;
   * fb_vtable_sync_ext_ops() mirrors them into ext_ops[] automatically. */

  fb_generic_fn fb_batch_gemm_bf16;  /* FB_OP_FB_BATCH_GEMM_BF16 = 2001 */
  fb_generic_fn fb_batch_gemm_fp32;  /* FB_OP_FB_BATCH_GEMM_FP32 = 2002 */
  fb_generic_fn fb_batch_gemm_int8;  /* FB_OP_FB_BATCH_GEMM_INT8 = 2003 */
  fb_generic_fn fb_batch_gemm_strided_bf16;  /* FB_OP_FB_BATCH_GEMM_STRIDED_BF16 = 2004 */
  fb_generic_fn fb_batch_gemm_strided_fp32;  /* FB_OP_FB_BATCH_GEMM_STRIDED_FP32 = 2005 */
  fb_generic_fn fb_batch_gemm_strided_int8;  /* FB_OP_FB_BATCH_GEMM_STRIDED_INT8 = 2006 */
  fb_generic_fn fb_center_of_mass;  /* FB_OP_FB_CENTER_OF_MASS = 2007 */
  fb_generic_fn fb_clebsch_gordan_coefficients;  /* FB_OP_FB_CLEBSCH_GORDAN_COEFFICIENTS = 2008 */
  fb_generic_fn fb_dbscan_fit;  /* FB_OP_FB_DBSCAN_FIT = 2009 */
  fb_generic_fn fb_einsum;  /* FB_OP_FB_EINSUM = 2010 */
  fb_generic_fn fb_elastic_net_fit;  /* FB_OP_FB_ELASTIC_NET_FIT = 2011 */
  fb_generic_fn fb_equivariant_mlp;  /* FB_OP_FB_EQUIVARIANT_MLP = 2012 */
  fb_generic_fn fb_equivariant_pooling;  /* FB_OP_FB_EQUIVARIANT_POOLING = 2013 */
  fb_generic_fn fb_feature_recursive_elimination;  /* FB_OP_FB_FEATURE_RECURSIVE_ELIMINATION = 2014 */
  fb_generic_fn fb_feature_univariate_selection;  /* FB_OP_FB_FEATURE_UNIVARIATE_SELECTION = 2015 */
  fb_generic_fn fb_feature_variance_threshold;  /* FB_OP_FB_FEATURE_VARIANCE_THRESHOLD = 2016 */
  fb_generic_fn fb_geometric_message_passing;  /* FB_OP_FB_GEOMETRIC_MESSAGE_PASSING = 2017 */
  fb_generic_fn fb_group_action;  /* FB_OP_FB_GROUP_ACTION = 2018 */
  fb_generic_fn fb_hierarchical_cut;  /* FB_OP_FB_HIERARCHICAL_CUT = 2019 */
  fb_generic_fn fb_hierarchical_fit;  /* FB_OP_FB_HIERARCHICAL_FIT = 2020 */
  fb_generic_fn fb_impute_knn;  /* FB_OP_FB_IMPUTE_KNN = 2021 */
  fb_generic_fn fb_impute_mean;  /* FB_OP_FB_IMPUTE_MEAN = 2022 */
  fb_generic_fn fb_impute_median;  /* FB_OP_FB_IMPUTE_MEDIAN = 2023 */
  fb_generic_fn fb_irrep_tensor_product;  /* FB_OP_FB_IRREP_TENSOR_PRODUCT = 2024 */
  fb_generic_fn fb_kfold_split;  /* FB_OP_FB_KFOLD_SPLIT = 2025 */
  fb_generic_fn fb_logistic_regression_fit;  /* FB_OP_FB_LOGISTIC_REGRESSION_FIT = 2026 */
  fb_generic_fn fb_logistic_regression_predict;  /* FB_OP_FB_LOGISTIC_REGRESSION_PREDICT = 2027 */
  fb_generic_fn fb_logistic_regression_predict_proba;  /* FB_OP_FB_LOGISTIC_REGRESSION_PREDICT_PROBA = 2028 */
  fb_generic_fn fb_nccl_allgather;  /* FB_OP_FB_NCCL_ALLGATHER = 2029 */
  fb_generic_fn fb_nccl_allreduce;  /* FB_OP_FB_NCCL_ALLREDUCE = 2030 */
  fb_generic_fn fb_nccl_alltoall;  /* FB_OP_FB_NCCL_ALLTOALL = 2031 */
  fb_generic_fn fb_nccl_broadcast;  /* FB_OP_FB_NCCL_BROADCAST = 2032 */
  fb_generic_fn fb_nccl_comm_split;  /* FB_OP_FB_NCCL_COMM_SPLIT = 2033 */
  fb_generic_fn fb_nccl_gather;  /* FB_OP_FB_NCCL_GATHER = 2034 */
  fb_generic_fn fb_nccl_group_end;  /* FB_OP_FB_NCCL_GROUP_END = 2035 */
  fb_generic_fn fb_nccl_group_start;  /* FB_OP_FB_NCCL_GROUP_START = 2036 */
  fb_generic_fn fb_nccl_recv;  /* FB_OP_FB_NCCL_RECV = 2037 */
  fb_generic_fn fb_nccl_reduce;  /* FB_OP_FB_NCCL_REDUCE = 2038 */
  fb_generic_fn fb_nccl_reducescatter;  /* FB_OP_FB_NCCL_REDUCESCATTER = 2039 */
  fb_generic_fn fb_nccl_scatter;  /* FB_OP_FB_NCCL_SCATTER = 2040 */
  fb_generic_fn fb_nccl_send;  /* FB_OP_FB_NCCL_SEND = 2041 */
  fb_generic_fn fb_neighbor_list_allpairs;  /* FB_OP_FB_NEIGHBOR_LIST_ALLPAIRS = 2042 */
  fb_generic_fn fb_neighbor_list_celllist;  /* FB_OP_FB_NEIGHBOR_LIST_CELLLIST = 2043 */
  fb_generic_fn fb_neighbor_list_verlet;  /* FB_OP_FB_NEIGHBOR_LIST_VERLET = 2044 */
  fb_generic_fn fb_parity_transform;  /* FB_OP_FB_PARITY_TRANSFORM = 2045 */
  fb_generic_fn fb_pme_auto_energy;  /* FB_OP_FB_PME_AUTO_ENERGY = 2046 */
  fb_generic_fn fb_pme_auto_forces;  /* FB_OP_FB_PME_AUTO_FORCES = 2047 */
  fb_generic_fn fb_pme_energy;  /* FB_OP_FB_PME_ENERGY = 2048 */
  fb_generic_fn fb_pme_forces;  /* FB_OP_FB_PME_FORCES = 2049 */
  fb_generic_fn fb_radial_basis_functions;  /* FB_OP_FB_RADIAL_BASIS_FUNCTIONS = 2050 */
  fb_generic_fn fb_reduce_unified;  /* FB_OP_FB_REDUCE_UNIFIED = 2051 */
  fb_generic_fn fb_reflect_positions;  /* FB_OP_FB_REFLECT_POSITIONS = 2052 */
  fb_generic_fn fb_rng_bernoulli;  /* FB_OP_FB_RNG_BERNOULLI = 2053 */
  fb_generic_fn fb_rng_beta;  /* FB_OP_FB_RNG_BETA = 2054 */
  fb_generic_fn fb_rng_binomial;  /* FB_OP_FB_RNG_BINOMIAL = 2055 */
  fb_generic_fn fb_rng_cauchy;  /* FB_OP_FB_RNG_CAUCHY = 2056 */
  fb_generic_fn fb_rng_create_philox;  /* FB_OP_FB_RNG_CREATE_PHILOX = 2057 */
  fb_generic_fn fb_rng_destroy;  /* FB_OP_FB_RNG_DESTROY = 2058 */
  fb_generic_fn fb_rng_exponential;  /* FB_OP_FB_RNG_EXPONENTIAL = 2059 */
  fb_generic_fn fb_rng_gamma;  /* FB_OP_FB_RNG_GAMMA = 2060 */
  fb_generic_fn fb_rng_gaussian;  /* FB_OP_FB_RNG_GAUSSIAN = 2061 */
  fb_generic_fn fb_rng_get_state_size;  /* FB_OP_FB_RNG_GET_STATE_SIZE = 2062 */
  fb_generic_fn fb_rng_lognormal;  /* FB_OP_FB_RNG_LOGNORMAL = 2063 */
  fb_generic_fn fb_rng_poisson;  /* FB_OP_FB_RNG_POISSON = 2064 */
  fb_generic_fn fb_rng_set_seed;  /* FB_OP_FB_RNG_SET_SEED = 2065 */
  fb_generic_fn fb_rng_skip_ahead;  /* FB_OP_FB_RNG_SKIP_AHEAD = 2066 */
  fb_generic_fn fb_rng_uniform;  /* FB_OP_FB_RNG_UNIFORM = 2067 */
  fb_generic_fn fb_rotate_irreps;  /* FB_OP_FB_ROTATE_IRREPS = 2068 */
  fb_generic_fn fb_rotate_spherical_harmonics;  /* FB_OP_FB_ROTATE_SPHERICAL_HARMONICS = 2069 */
  fb_generic_fn fb_se3_convolution;  /* FB_OP_FB_SE3_CONVOLUTION = 2070 */
  fb_generic_fn fb_spherical_harmonics;  /* FB_OP_FB_SPHERICAL_HARMONICS = 2071 */
  fb_generic_fn fb_spherical_to_cartesian;  /* FB_OP_FB_SPHERICAL_TO_CARTESIAN = 2072 */
  fb_generic_fn fb_stats_central_moments;  /* FB_OP_FB_STATS_CENTRAL_MOMENTS = 2073 */
  fb_generic_fn fb_stats_correlation;  /* FB_OP_FB_STATS_CORRELATION = 2074 */
  fb_generic_fn fb_stats_correlation_matrix;  /* FB_OP_FB_STATS_CORRELATION_MATRIX = 2075 */
  fb_generic_fn fb_stats_covariance;  /* FB_OP_FB_STATS_COVARIANCE = 2076 */
  fb_generic_fn fb_stats_covariance_matrix;  /* FB_OP_FB_STATS_COVARIANCE_MATRIX = 2077 */
  fb_generic_fn fb_stats_histogram;  /* FB_OP_FB_STATS_HISTOGRAM = 2078 */
  fb_generic_fn fb_stats_iqr_outliers;  /* FB_OP_FB_STATS_IQR_OUTLIERS = 2079 */
  fb_generic_fn fb_stats_kde;  /* FB_OP_FB_STATS_KDE = 2080 */
  fb_generic_fn fb_stats_kurtosis;  /* FB_OP_FB_STATS_KURTOSIS = 2081 */
  fb_generic_fn fb_stats_mean;  /* FB_OP_FB_STATS_MEAN = 2082 */
  fb_generic_fn fb_stats_median;  /* FB_OP_FB_STATS_MEDIAN = 2083 */
  fb_generic_fn fb_stats_min_max;  /* FB_OP_FB_STATS_MIN_MAX = 2084 */
  fb_generic_fn fb_stats_modified_zscore;  /* FB_OP_FB_STATS_MODIFIED_ZSCORE = 2085 */
  fb_generic_fn fb_stats_percentile;  /* FB_OP_FB_STATS_PERCENTILE = 2086 */
  fb_generic_fn fb_stats_quantile;  /* FB_OP_FB_STATS_QUANTILE = 2087 */
  fb_generic_fn fb_stats_quartiles;  /* FB_OP_FB_STATS_QUARTILES = 2088 */
  fb_generic_fn fb_stats_raw_moments;  /* FB_OP_FB_STATS_RAW_MOMENTS = 2089 */
  fb_generic_fn fb_stats_skewness;  /* FB_OP_FB_STATS_SKEWNESS = 2090 */
  fb_generic_fn fb_stats_std;  /* FB_OP_FB_STATS_STD = 2091 */
  fb_generic_fn fb_stats_sum;  /* FB_OP_FB_STATS_SUM = 2092 */
  fb_generic_fn fb_stats_variance;  /* FB_OP_FB_STATS_VARIANCE = 2093 */
  fb_generic_fn fb_stats_zscore;  /* FB_OP_FB_STATS_ZSCORE = 2094 */
  fb_generic_fn fb_svd_truncated;  /* FB_OP_FB_SVD_TRUNCATED = 2095 */
  fb_generic_fn fb_tensor_add;  /* FB_OP_FB_TENSOR_ADD = 2096 */
  fb_generic_fn fb_tensor_contract;  /* FB_OP_FB_TENSOR_CONTRACT = 2097 */
  fb_generic_fn fb_tensor_contract_batched;  /* FB_OP_FB_TENSOR_CONTRACT_BATCHED = 2098 */
  fb_generic_fn fb_tensor_copy;  /* FB_OP_FB_TENSOR_COPY = 2099 */
  fb_generic_fn fb_tensor_gemm;  /* FB_OP_FB_TENSOR_GEMM = 2100 */
  fb_generic_fn fb_tensor_hadamard;  /* FB_OP_FB_TENSOR_HADAMARD = 2101 */
  fb_generic_fn fb_tensor_mttkrp;  /* FB_OP_FB_TENSOR_MTTKRP = 2102 */
  fb_generic_fn fb_tensor_permute;  /* FB_OP_FB_TENSOR_PERMUTE = 2103 */
  fb_generic_fn fb_tensor_reduce_max;  /* FB_OP_FB_TENSOR_REDUCE_MAX = 2104 */
  fb_generic_fn fb_tensor_reduce_min;  /* FB_OP_FB_TENSOR_REDUCE_MIN = 2105 */
  fb_generic_fn fb_tensor_reduce_norm;  /* FB_OP_FB_TENSOR_REDUCE_NORM = 2106 */
  fb_generic_fn fb_tensor_reduce_sum;  /* FB_OP_FB_TENSOR_REDUCE_SUM = 2107 */
  fb_generic_fn fb_tensor_reshape;  /* FB_OP_FB_TENSOR_RESHAPE = 2108 */
  fb_generic_fn fb_tensor_scale;  /* FB_OP_FB_TENSOR_SCALE = 2109 */
  fb_generic_fn fb_tensor_transpose;  /* FB_OP_FB_TENSOR_TRANSPOSE = 2110 */
  fb_generic_fn fb_tensor_transpose_scale;  /* FB_OP_FB_TENSOR_TRANSPOSE_SCALE = 2111 */
  fb_generic_fn fb_tensor_ttm;  /* FB_OP_FB_TENSOR_TTM = 2112 */
  fb_generic_fn fb_tensor_ttv;  /* FB_OP_FB_TENSOR_TTV = 2113 */
  fb_generic_fn fb_tensor_view;  /* FB_OP_FB_TENSOR_VIEW = 2114 */
  fb_generic_fn fb_train_test_split;  /* FB_OP_FB_TRAIN_TEST_SPLIT = 2115 */
  fb_generic_fn fb_translate_positions;  /* FB_OP_FB_TRANSLATE_POSITIONS = 2116 */
  fb_generic_fn fb_wigner_d_matrix;  /* FB_OP_FB_WIGNER_D_MATRIX = 2117 */

  /* ========================================================================
   * Statistics + Machine Learning
   * ========================================================================
   * fb_generic_fn fields: backends assign  vtable->op = (fb_generic_fn)impl;
   * fb_vtable_sync_ext_ops() mirrors them into ext_ops[] automatically. */

  fb_generic_fn fb_agglomerative_fit;  /* FB_OP_FB_AGGLOMERATIVE_FIT = 2118 */
  fb_generic_fn fb_covariance_matrix_equivariant;  /* FB_OP_FB_COVARIANCE_MATRIX_EQUIVARIANT = 2119 */
  fb_generic_fn fb_decision_tree_feature_importance;  /* FB_OP_FB_DECISION_TREE_FEATURE_IMPORTANCE = 2120 */
  fb_generic_fn fb_decision_tree_fit;  /* FB_OP_FB_DECISION_TREE_FIT = 2121 */
  fb_generic_fn fb_decision_tree_predict;  /* FB_OP_FB_DECISION_TREE_PREDICT = 2122 */
  fb_generic_fn fb_kmeans_fit;  /* FB_OP_FB_KMEANS_FIT = 2123 */
  fb_generic_fn fb_kmeans_inertia;  /* FB_OP_FB_KMEANS_INERTIA = 2124 */
  fb_generic_fn fb_kmeans_predict;  /* FB_OP_FB_KMEANS_PREDICT = 2125 */
  fb_generic_fn fb_lasso_regression_fit;  /* FB_OP_FB_LASSO_REGRESSION_FIT = 2126 */
  fb_generic_fn fb_linear_regression_fit;  /* FB_OP_FB_LINEAR_REGRESSION_FIT = 2127 */
  fb_generic_fn fb_linear_regression_predict;  /* FB_OP_FB_LINEAR_REGRESSION_PREDICT = 2128 */
  fb_generic_fn fb_linear_regression_score;  /* FB_OP_FB_LINEAR_REGRESSION_SCORE = 2129 */
  fb_generic_fn fb_min_max_scale;  /* FB_OP_FB_MIN_MAX_SCALE = 2130 */
  fb_generic_fn fb_normalize;  /* FB_OP_FB_NORMALIZE = 2131 */
  fb_generic_fn fb_normalize_unified;  /* FB_OP_FB_NORMALIZE_UNIFIED = 2132 */
  fb_generic_fn fb_pca_explained_variance;  /* FB_OP_FB_PCA_EXPLAINED_VARIANCE = 2133 */
  fb_generic_fn fb_pca_fit;  /* FB_OP_FB_PCA_FIT = 2134 */
  fb_generic_fn fb_pca_fit_transform;  /* FB_OP_FB_PCA_FIT_TRANSFORM = 2135 */
  fb_generic_fn fb_pca_inverse_transform;  /* FB_OP_FB_PCA_INVERSE_TRANSFORM = 2136 */
  fb_generic_fn fb_pca_transform;  /* FB_OP_FB_PCA_TRANSFORM = 2137 */
  fb_generic_fn fb_permutation_equivariant_attention;  /* FB_OP_FB_PERMUTATION_EQUIVARIANT_ATTENTION = 2138 */
  fb_generic_fn fb_permutation_invariant_aggregation;  /* FB_OP_FB_PERMUTATION_INVARIANT_AGGREGATION = 2139 */
  fb_generic_fn fb_ridge_regression_fit;  /* FB_OP_FB_RIDGE_REGRESSION_FIT = 2140 */
  fb_generic_fn fb_standardize;  /* FB_OP_FB_STANDARDIZE = 2141 */

  /* ========================================================================
   * Parallel Primitives (fb_prim_*)
   * ========================================================================
   * fb_generic_fn fields: backends assign  vtable->op = (fb_generic_fn)impl;
   * fb_vtable_sync_ext_ops() mirrors them into ext_ops[] automatically. */

  fb_generic_fn fb_prim_adjacent_difference;  /* FB_OP_FB_PRIM_ADJACENT_DIFFERENCE = 2142 */
  fb_generic_fn fb_prim_copy_if;  /* FB_OP_FB_PRIM_COPY_IF = 2143 */
  fb_generic_fn fb_prim_discontinuity;  /* FB_OP_FB_PRIM_DISCONTINUITY = 2144 */
  fb_generic_fn fb_prim_exchange_block_to_striped;  /* FB_OP_FB_PRIM_EXCHANGE_BLOCK_TO_STRIPED = 2145 */
  fb_generic_fn fb_prim_exchange_striped_to_block;  /* FB_OP_FB_PRIM_EXCHANGE_STRIPED_TO_BLOCK = 2146 */
  fb_generic_fn fb_prim_gather;  /* FB_OP_FB_PRIM_GATHER = 2147 */
  fb_generic_fn fb_prim_histogram_even;  /* FB_OP_FB_PRIM_HISTOGRAM_EVEN = 2148 */
  fb_generic_fn fb_prim_histogram_range;  /* FB_OP_FB_PRIM_HISTOGRAM_RANGE = 2149 */
  fb_generic_fn fb_prim_merge;  /* FB_OP_FB_PRIM_MERGE = 2150 */
  fb_generic_fn fb_prim_merge_by_key;  /* FB_OP_FB_PRIM_MERGE_BY_KEY = 2151 */
  fb_generic_fn fb_prim_merge_sort;  /* FB_OP_FB_PRIM_MERGE_SORT = 2152 */
  fb_generic_fn fb_prim_nth_element;  /* FB_OP_FB_PRIM_NTH_ELEMENT = 2153 */
  fb_generic_fn fb_prim_partial_sort;  /* FB_OP_FB_PRIM_PARTIAL_SORT = 2154 */
  fb_generic_fn fb_prim_partition;  /* FB_OP_FB_PRIM_PARTITION = 2155 */
  fb_generic_fn fb_prim_partition_three_way;  /* FB_OP_FB_PRIM_PARTITION_THREE_WAY = 2156 */
  fb_generic_fn fb_prim_radix_sort;  /* FB_OP_FB_PRIM_RADIX_SORT = 2157 */
  fb_generic_fn fb_prim_reduce;  /* FB_OP_FB_PRIM_REDUCE = 2158 */
  fb_generic_fn fb_prim_reduce_block;  /* FB_OP_FB_PRIM_REDUCE_BLOCK = 2159 */
  fb_generic_fn fb_prim_reduce_by_key;  /* FB_OP_FB_PRIM_REDUCE_BY_KEY = 2160 */
  fb_generic_fn fb_prim_reduce_device;  /* FB_OP_FB_PRIM_REDUCE_DEVICE = 2161 */
  fb_generic_fn fb_prim_reduce_warp;  /* FB_OP_FB_PRIM_REDUCE_WARP = 2162 */
  fb_generic_fn fb_prim_scan_by_key;  /* FB_OP_FB_PRIM_SCAN_BY_KEY = 2163 */
  fb_generic_fn fb_prim_scan_exclusive;  /* FB_OP_FB_PRIM_SCAN_EXCLUSIVE = 2164 */
  fb_generic_fn fb_prim_scan_inclusive;  /* FB_OP_FB_PRIM_SCAN_INCLUSIVE = 2165 */
  fb_generic_fn fb_prim_scatter;  /* FB_OP_FB_PRIM_SCATTER = 2166 */
  fb_generic_fn fb_prim_select_flagged;  /* FB_OP_FB_PRIM_SELECT_FLAGGED = 2167 */
  fb_generic_fn fb_prim_select_if;  /* FB_OP_FB_PRIM_SELECT_IF = 2168 */
  fb_generic_fn fb_prim_shuffle_rotate;  /* FB_OP_FB_PRIM_SHUFFLE_ROTATE = 2169 */
  fb_generic_fn fb_prim_shuffle_up;  /* FB_OP_FB_PRIM_SHUFFLE_UP = 2170 */
  fb_generic_fn fb_prim_sort_keys;  /* FB_OP_FB_PRIM_SORT_KEYS = 2171 */
  fb_generic_fn fb_prim_sort_pairs;  /* FB_OP_FB_PRIM_SORT_PAIRS = 2172 */
  fb_generic_fn fb_prim_transform;  /* FB_OP_FB_PRIM_TRANSFORM = 2173 */
  fb_generic_fn fb_prim_transform_if;  /* FB_OP_FB_PRIM_TRANSFORM_IF = 2174 */
  fb_generic_fn fb_prim_unique;  /* FB_OP_FB_PRIM_UNIQUE = 2175 */
  fb_generic_fn fb_prim_unique_by_key;  /* FB_OP_FB_PRIM_UNIQUE_BY_KEY = 2176 */

  /* ========================================================================
   * Chemistry / Physics extensions
   * ========================================================================
   * fb_generic_fn fields: backends assign  vtable->op = (fb_generic_fn)impl;
   * fb_vtable_sync_ext_ops() mirrors them into ext_ops[] automatically. */

  fb_generic_fn fb_coulomb_direct_energy;  /* FB_OP_FB_COULOMB_DIRECT_ENERGY = 2177 */
  fb_generic_fn fb_coulomb_direct_forces;  /* FB_OP_FB_COULOMB_DIRECT_FORCES = 2178 */
  fb_generic_fn fb_dftd3_dispersion_energy;  /* FB_OP_FB_DFTD3_DISPERSION_ENERGY = 2179 */
  fb_generic_fn fb_dftd3_dispersion_forces;  /* FB_OP_FB_DFTD3_DISPERSION_FORCES = 2180 */
  fb_generic_fn fb_dftd4_dispersion_energy;  /* FB_OP_FB_DFTD4_DISPERSION_ENERGY = 2181 */
  fb_generic_fn fb_dftd4_dispersion_forces;  /* FB_OP_FB_DFTD4_DISPERSION_FORCES = 2182 */
  fb_generic_fn fb_ewald_energy;  /* FB_OP_FB_EWALD_ENERGY = 2183 */
  fb_generic_fn fb_ewald_forces;  /* FB_OP_FB_EWALD_FORCES = 2184 */

  /* ========================================================================
   * Vector math (fb_v*)
   * ========================================================================
   * fb_generic_fn fields: backends assign  vtable->op = (fb_generic_fn)impl;
   * fb_vtable_sync_ext_ops() mirrors them into ext_ops[] automatically. */

  fb_generic_fn fb_vacos;  /* FB_OP_FB_VACOS = 2185 */
  fb_generic_fn fb_vacosh;  /* FB_OP_FB_VACOSH = 2186 */
  fb_generic_fn fb_vasin;  /* FB_OP_FB_VASIN = 2187 */
  fb_generic_fn fb_vasinh;  /* FB_OP_FB_VASINH = 2188 */
  fb_generic_fn fb_vatan;  /* FB_OP_FB_VATAN = 2189 */
  fb_generic_fn fb_vatanh;  /* FB_OP_FB_VATANH = 2190 */
  fb_generic_fn fb_vcbrt;  /* FB_OP_FB_VCBRT = 2191 */
  fb_generic_fn fb_vcos;  /* FB_OP_FB_VCOS = 2192 */
  fb_generic_fn fb_vcosh;  /* FB_OP_FB_VCOSH = 2193 */
  fb_generic_fn fb_verf;  /* FB_OP_FB_VERF = 2194 */
  fb_generic_fn fb_verfc;  /* FB_OP_FB_VERFC = 2195 */
  fb_generic_fn fb_verfinv;  /* FB_OP_FB_VERFINV = 2196 */
  fb_generic_fn fb_vexp;  /* FB_OP_FB_VEXP = 2197 */
  fb_generic_fn fb_vexp2;  /* FB_OP_FB_VEXP2 = 2198 */
  fb_generic_fn fb_vgamma;  /* FB_OP_FB_VGAMMA = 2199 */
  fb_generic_fn fb_vinvcbrt;  /* FB_OP_FB_VINVCBRT = 2200 */
  fb_generic_fn fb_vinvsqrt;  /* FB_OP_FB_VINVSQRT = 2201 */
  fb_generic_fn fb_vj0;  /* FB_OP_FB_VJ0 = 2202 */
  fb_generic_fn fb_vj1;  /* FB_OP_FB_VJ1 = 2203 */
  fb_generic_fn fb_vlgamma;  /* FB_OP_FB_VLGAMMA = 2204 */
  fb_generic_fn fb_vlog;  /* FB_OP_FB_VLOG = 2205 */
  fb_generic_fn fb_vlog10;  /* FB_OP_FB_VLOG10 = 2206 */
  fb_generic_fn fb_vlog2;  /* FB_OP_FB_VLOG2 = 2207 */
  fb_generic_fn fb_vpow;  /* FB_OP_FB_VPOW = 2208 */
  fb_generic_fn fb_vpow2o3;  /* FB_OP_FB_VPOW2O3 = 2209 */
  fb_generic_fn fb_vpow3o2;  /* FB_OP_FB_VPOW3O2 = 2210 */
  fb_generic_fn fb_vsin;  /* FB_OP_FB_VSIN = 2211 */
  fb_generic_fn fb_vsinh;  /* FB_OP_FB_VSINH = 2212 */
  fb_generic_fn fb_vsqrt;  /* FB_OP_FB_VSQRT = 2213 */
  fb_generic_fn fb_vtan;  /* FB_OP_FB_VTAN = 2214 */
  fb_generic_fn fb_vtanh;  /* FB_OP_FB_VTANH = 2215 */
  fb_generic_fn fb_vy0;  /* FB_OP_FB_VY0 = 2216 */
  fb_generic_fn fb_vy1;  /* FB_OP_FB_VY1 = 2217 */

  /* ========================================================================
   * Spline interpolation (fb_spline_*)
   * ========================================================================
   * fb_generic_fn fields: backends assign  vtable->op = (fb_generic_fn)impl;
   * fb_vtable_sync_ext_ops() mirrors them into ext_ops[] automatically. */

  fb_generic_fn fb_spline_create_1d_akima;  /* FB_OP_FB_SPLINE_CREATE_1D_AKIMA = 2218 */
  fb_generic_fn fb_spline_create_1d_bessel;  /* FB_OP_FB_SPLINE_CREATE_1D_BESSEL = 2219 */
  fb_generic_fn fb_spline_create_1d_cubic;  /* FB_OP_FB_SPLINE_CREATE_1D_CUBIC = 2220 */
  fb_generic_fn fb_spline_create_1d_hermite;  /* FB_OP_FB_SPLINE_CREATE_1D_HERMITE = 2221 */
  fb_generic_fn fb_spline_create_1d_linear;  /* FB_OP_FB_SPLINE_CREATE_1D_LINEAR = 2222 */
  fb_generic_fn fb_spline_create_1d_quadratic;  /* FB_OP_FB_SPLINE_CREATE_1D_QUADRATIC = 2223 */
  fb_generic_fn fb_spline_create_2d_cubic;  /* FB_OP_FB_SPLINE_CREATE_2D_CUBIC = 2224 */
  fb_generic_fn fb_spline_create_2d_linear;  /* FB_OP_FB_SPLINE_CREATE_2D_LINEAR = 2225 */
  fb_generic_fn fb_spline_destroy;  /* FB_OP_FB_SPLINE_DESTROY = 2226 */
  fb_generic_fn fb_spline_eval;  /* FB_OP_FB_SPLINE_EVAL = 2227 */
  fb_generic_fn fb_spline_eval_batch;  /* FB_OP_FB_SPLINE_EVAL_BATCH = 2228 */
  fb_generic_fn fb_spline_eval_derivative;  /* FB_OP_FB_SPLINE_EVAL_DERIVATIVE = 2229 */
  fb_generic_fn fb_spline_eval_integral;  /* FB_OP_FB_SPLINE_EVAL_INTEGRAL = 2230 */
  fb_generic_fn fb_spline_get_breakpoints;  /* FB_OP_FB_SPLINE_GET_BREAKPOINTS = 2231 */
  fb_generic_fn fb_spline_get_coefficients;  /* FB_OP_FB_SPLINE_GET_COEFFICIENTS = 2232 */
  fb_generic_fn fb_spline_search_cell;  /* FB_OP_FB_SPLINE_SEARCH_CELL = 2233 */

  /* ========================================================================
   * GEMM / LPGEMM fused extensions
   * ========================================================================
   * fb_generic_fn fields: backends assign  vtable->op = (fb_generic_fn)impl;
   * fb_vtable_sync_ext_ops() mirrors them into ext_ops[] automatically. */

  fb_generic_fn fb_gemm_bias_bf16;  /* FB_OP_FB_GEMM_BIAS_BF16 = 2234 */
  fb_generic_fn fb_gemm_bias_fp32;  /* FB_OP_FB_GEMM_BIAS_FP32 = 2235 */
  fb_generic_fn fb_gemm_bias_gelu_bf16;  /* FB_OP_FB_GEMM_BIAS_GELU_BF16 = 2236 */
  fb_generic_fn fb_gemm_bias_gelu_fp32;  /* FB_OP_FB_GEMM_BIAS_GELU_FP32 = 2237 */
  fb_generic_fn fb_gemm_bias_int8;  /* FB_OP_FB_GEMM_BIAS_INT8 = 2238 */
  fb_generic_fn fb_gemm_bias_layernorm_bf16;  /* FB_OP_FB_GEMM_BIAS_LAYERNORM_BF16 = 2239 */
  fb_generic_fn fb_gemm_bias_layernorm_fp32;  /* FB_OP_FB_GEMM_BIAS_LAYERNORM_FP32 = 2240 */
  fb_generic_fn fb_gemm_bias_relu_bf16;  /* FB_OP_FB_GEMM_BIAS_RELU_BF16 = 2241 */
  fb_generic_fn fb_gemm_bias_relu_fp32;  /* FB_OP_FB_GEMM_BIAS_RELU_FP32 = 2242 */
  fb_generic_fn fb_gemm_bias_residual_relu_bf16;  /* FB_OP_FB_GEMM_BIAS_RESIDUAL_RELU_BF16 = 2243 */
  fb_generic_fn fb_gemm_bias_residual_relu_fp32;  /* FB_OP_FB_GEMM_BIAS_RESIDUAL_RELU_FP32 = 2244 */
  fb_generic_fn fb_gemm_bias_sigmoid_bf16;  /* FB_OP_FB_GEMM_BIAS_SIGMOID_BF16 = 2245 */
  fb_generic_fn fb_gemm_bias_sigmoid_fp32;  /* FB_OP_FB_GEMM_BIAS_SIGMOID_FP32 = 2246 */
  fb_generic_fn fb_gemm_bias_tanh_bf16;  /* FB_OP_FB_GEMM_BIAS_TANH_BF16 = 2247 */
  fb_generic_fn fb_gemm_bias_tanh_fp32;  /* FB_OP_FB_GEMM_BIAS_TANH_FP32 = 2248 */
  fb_generic_fn fb_gemm_fp8_e4m3;  /* FB_OP_FB_GEMM_FP8_E4M3 = 2249 */
  fb_generic_fn fb_gemm_fused_bias_add;  /* FB_OP_FB_GEMM_FUSED_BIAS_ADD = 2250 */
  fb_generic_fn fb_gemm_fused_gelu;  /* FB_OP_FB_GEMM_FUSED_GELU = 2251 */
  fb_generic_fn fb_gemm_fused_relu;  /* FB_OP_FB_GEMM_FUSED_RELU = 2252 */
  fb_generic_fn fb_gemm_fused_scale_act;  /* FB_OP_FB_GEMM_FUSED_SCALE_ACT = 2253 */
  fb_generic_fn fb_gemm_fused_silu;  /* FB_OP_FB_GEMM_FUSED_SILU = 2254 */
  fb_generic_fn fb_gemm_fused_swish;  /* FB_OP_FB_GEMM_FUSED_SWISH = 2255 */
  fb_generic_fn fb_gemm_int4;  /* FB_OP_FB_GEMM_INT4 = 2256 */
  fb_generic_fn fb_gemm_unified;  /* FB_OP_FB_GEMM_UNIFIED = 2257 */
  fb_generic_fn fb_lpgemm_bf16;  /* FB_OP_FB_LPGEMM_BF16 = 2258 */
  fb_generic_fn fb_lpgemm_bf16_bf16;  /* FB_OP_FB_LPGEMM_BF16_BF16 = 2259 */
  fb_generic_fn fb_lpgemm_fp32;  /* FB_OP_FB_LPGEMM_FP32 = 2260 */
  fb_generic_fn fb_lpgemm_int8;  /* FB_OP_FB_LPGEMM_INT8 = 2261 */
  fb_generic_fn fb_lpgemm_int8_fp32;  /* FB_OP_FB_LPGEMM_INT8_FP32 = 2262 */
  fb_generic_fn fb_qgemm_bias_gelu_int8;  /* FB_OP_FB_QGEMM_BIAS_GELU_INT8 = 2263 */
  fb_generic_fn fb_qgemm_bias_relu_int8;  /* FB_OP_FB_QGEMM_BIAS_RELU_INT8 = 2264 */
  fb_generic_fn fb_qgemm_int8_asymmetric;  /* FB_OP_FB_QGEMM_INT8_ASYMMETRIC = 2265 */

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
   *
   * ext_ops[op_id][FB_CONV_CBLAS]   — CBLAS convention slot (direct cast, no thunk)
   * ext_ops[op_id][FB_CONV_FORTRAN] — Fortran convention slot (thunk-generated if absent)
   * ext_ops[op_id][FB_CONV_CBLAS] / [FB_CONV_FORTRAN] — convention slots
   *
   * Use fb_enumerate_and_populate(vtable, handle) to auto-fill all three slots
   * from a DLL/SO's exports.  fb_finalize_plugin_vtable() (Strategy 5) fills any
   * remaining empty slots via static cross-convention thunks from conv_thunks.c.
   */
  fb_generic_fn ext_ops[FB_JUDGE_MAX_OPERATIONS][FB_CONV_COUNT];

} fb_backend_vtable_t;

#ifdef __cplusplus
}
#endif

#endif /* FASTER_BLASTER_BACKEND_INTERFACE_H */
