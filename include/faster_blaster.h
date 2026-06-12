/**
 * @file faster_blaster.h
 * @brief Faster-BLASTER: High-performance multi-backend BLAS abstraction layer
 * 
 * This library provides a complete C interface to BLAS (Basic Linear Algebra
 * Subprograms) that automatically dispatches to the fastest available backend
 * for your hardware. It is a drop-in replacement for standard BLAS libraries.
 * 
 * Key features:
 * - Hardware-specific auto-calibration for optimal performance
 * - Multi-backend support (OpenBLAS, MKL, cuBLAS, BLIS, etc.)
 * - Zero-overhead dispatch after calibration
 * - Thread-safe concurrent execution
 * - CPU and GPU acceleration
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#ifndef FASTER_BLASTER_H
#define FASTER_BLASTER_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Library Version and Initialization
 * ========================================================================= */

#define FASTER_BLASTER_VERSION_MAJOR 0
#define FASTER_BLASTER_VERSION_MINOR 1
#define FASTER_BLASTER_VERSION_PATCH 0

/**
 * @brief Initialize the faster-blaster library
 * 
 * This function must be called before any BLAS operations. It will:
 * 1. Detect hardware configuration
 * 2. Load available backend plugins
 * 3. Load or perform calibration for this hardware
 * 4. Set up optimal function dispatch table
 * 
 * If calibration is needed, this may take several minutes on first run.
 * A progress callback can be provided to monitor calibration progress.
 * 
 * @return 0 on success, negative error code on failure
 */
int fb_init(void);

/**
 * @brief Clean up and shut down the library
 * 
 * Unloads all backends and frees resources. Should be called before program exit.
 */
void fb_finalize(void);

/**
 * @brief Get library version string
 * 
 * @return Version string in format "major.minor.patch"
 */
const char* fb_get_version_string(void);

/**
 * @brief Get library version numbers
 *
 * @param major Output for major version
 * @param minor Output for minor version
 * @param patch Output for patch version
 * @return 0 on success
 */
int fb_get_version(int *major, int *minor, int *patch);

/**
 * @brief Check if library is initialized
 * 
 * @return true if initialized, false otherwise
 */
bool fb_is_initialized(void);

/* ============================================================================
 * BLAS Type Definitions (Standard)
 * ========================================================================= */

/**
 * @brief Matrix layout enumeration
 */
typedef enum {
    FbRowMajor = 101,  /**< Row-major (C) order */
    FbColMajor = 102   /**< Column-major (Fortran) order */
} FB_LAYOUT;

/**
 * @brief Transpose operation enumeration
 */
typedef enum {
    FbNoTrans   = 111,  /**< No transpose */
    FbTrans     = 112,  /**< Transpose */
    FbConjTrans = 113   /**< Conjugate transpose */
} FB_TRANSPOSE;

/**
 * @brief Upper/lower triangle specification
 */
typedef enum {
    FbUpper = 121,  /**< Upper triangular */
    FbLower = 122   /**< Lower triangular */
} FB_UPLO;

/**
 * @brief Diagonal type specification
 */
typedef enum {
    FbNonUnit = 131,  /**< Non-unit diagonal */
    FbUnit    = 132   /**< Unit diagonal */
} FB_DIAG;

/**
 * @brief Side specification for operations
 */
typedef enum {
    FbLeft  = 141,  /**< Operate on left side */
    FbRight = 142   /**< Operate on right side */
} FB_SIDE;

/* ============================================================================
 * BLAS Level 1: Vector-Vector Operations
 * ========================================================================= */

/* Single precision */
void fb_sswap(const int N, float *X, const int incX, float *Y, const int incY);
void fb_sscal(const int N, const float alpha, float *X, const int incX);
void fb_scopy(const int N, const float *X, const int incX, float *Y, const int incY);
void fb_saxpy(const int N, const float alpha, const float *X, const int incX, float *Y, const int incY);
float fb_sdot(const int N, const float *X, const int incX, const float *Y, const int incY);
float fb_snrm2(const int N, const float *X, const int incX);
float fb_sasum(const int N, const float *X, const int incX);
size_t fb_isamax(const int N, const float *X, const int incX);
void fb_srot(const int N, float *X, const int incX, float *Y, const int incY, const float c, const float s);
void fb_srotg(float *a, float *b, float *c, float *s);
void fb_srotm(const int N, float *X, const int incX, float *Y, const int incY, const float *param);
void fb_srotmg(float *d1, float *d2, float *x1, const float y1, float *param);

/* Double precision */
void fb_dswap(const int N, double *X, const int incX, double *Y, const int incY);
void fb_dscal(const int N, const double alpha, double *X, const int incX);
void fb_dcopy(const int N, const double *X, const int incX, double *Y, const int incY);
void fb_daxpy(const int N, const double alpha, const double *X, const int incX, double *Y, const int incY);
double fb_ddot(const int N, const double *X, const int incX, const double *Y, const int incY);
double fb_dnrm2(const int N, const double *X, const int incX);
double fb_dasum(const int N, const double *X, const int incX);
size_t fb_idamax(const int N, const double *X, const int incX);
void fb_drot(const int N, double *X, const int incX, double *Y, const int incY, const double c, const double s);
void fb_drotg(double *a, double *b, double *c, double *s);
void fb_drotm(const int N, double *X, const int incX, double *Y, const int incY, const double *param);
void fb_drotmg(double *d1, double *d2, double *x1, const double y1, double *param);

/* Complex single precision */
void fb_cswap(const int N, void *X, const int incX, void *Y, const int incY);
void fb_cscal(const int N, const void *alpha, void *X, const int incX);
void fb_ccopy(const int N, const void *X, const int incX, void *Y, const int incY);
void fb_caxpy(const int N, const void *alpha, const void *X, const int incX, void *Y, const int incY);
void fb_cdotu(const int N, const void *X, const int incX, const void *Y, const int incY, void *result);
void fb_cdotc(const int N, const void *X, const int incX, const void *Y, const int incY, void *result);
float fb_scnrm2(const int N, const void *X, const int incX);
float fb_scasum(const int N, const void *X, const int incX);
size_t fb_icamax(const int N, const void *X, const int incX);

/* Complex double precision */
void fb_zswap(const int N, void *X, const int incX, void *Y, const int incY);
void fb_zscal(const int N, const void *alpha, void *X, const int incX);
void fb_zcopy(const int N, const void *X, const int incX, void *Y, const int incY);
void fb_zaxpy(const int N, const void *alpha, const void *X, const int incX, void *Y, const int incY);
void fb_zdotu(const int N, const void *X, const int incX, const void *Y, const int incY, void *result);
void fb_zdotc(const int N, const void *X, const int incX, const void *Y, const int incY, void *result);
double fb_dznrm2(const int N, const void *X, const int incX);
double fb_dzasum(const int N, const void *X, const int incX);
size_t fb_izamax(const int N, const void *X, const int incX);

/* ============================================================================
 * BLAS Level 2: Matrix-Vector Operations
 * ========================================================================= */

/* Single precision */
void fb_sgemv(const FB_LAYOUT Layout, const FB_TRANSPOSE TransA,
              const int M, const int N, const float alpha,
              const float *A, const int lda,
              const float *X, const int incX,
              const float beta, float *Y, const int incY);

void fb_ssymv(const FB_LAYOUT Layout, const FB_UPLO Uplo,
              const int N, const float alpha,
              const float *A, const int lda,
              const float *X, const int incX,
              const float beta, float *Y, const int incY);

void fb_sgbmv(const FB_LAYOUT Layout, const FB_TRANSPOSE TransA,
              const int M, const int N, const int kl, const int ku,
              const float alpha, const float *A, const int lda,
              const float *X, const int incX,
              const float beta, float *Y, const int incY);

void fb_ssbmv(const FB_LAYOUT Layout, const FB_UPLO Uplo,
              const int N, const int k,
              const float alpha, const float *A, const int lda,
              const float *X, const int incX,
              const float beta, float *Y, const int incY);

void fb_sspmv(const FB_LAYOUT Layout, const FB_UPLO Uplo,
              const int N, const float alpha,
              const float *Ap, const float *X, const int incX,
              const float beta, float *Y, const int incY);

void fb_strmv(const FB_LAYOUT Layout, const FB_UPLO Uplo,
              const FB_TRANSPOSE TransA, const FB_DIAG Diag,
              const int N, const float *A, const int lda,
              float *X, const int incX);

void fb_stbmv(const FB_LAYOUT Layout, const FB_UPLO Uplo,
              const FB_TRANSPOSE TransA, const FB_DIAG Diag,
              const int N, const int k,
              const float *A, const int lda,
              float *X, const int incX);

void fb_stpmv(const FB_LAYOUT Layout, const FB_UPLO Uplo,
              const FB_TRANSPOSE TransA, const FB_DIAG Diag,
              const int N, const float *Ap,
              float *X, const int incX);

void fb_strsv(const FB_LAYOUT Layout, const FB_UPLO Uplo,
              const FB_TRANSPOSE TransA, const FB_DIAG Diag,
              const int N, const float *A, const int lda,
              float *X, const int incX);

void fb_stbsv(const FB_LAYOUT Layout, const FB_UPLO Uplo,
              const FB_TRANSPOSE TransA, const FB_DIAG Diag,
              const int N, const int k,
              const float *A, const int lda,
              float *X, const int incX);

void fb_stpsv(const FB_LAYOUT Layout, const FB_UPLO Uplo,
              const FB_TRANSPOSE TransA, const FB_DIAG Diag,
              const int N, const float *Ap,
              float *X, const int incX);

void fb_sger(const FB_LAYOUT Layout, const int M, const int N,
             const float alpha, const float *X, const int incX,
             const float *Y, const int incY,
             float *A, const int lda);

void fb_ssyr(const FB_LAYOUT Layout, const FB_UPLO Uplo,
             const int N, const float alpha,
             const float *X, const int incX,
             float *A, const int lda);

void fb_sspr(const FB_LAYOUT Layout, const FB_UPLO Uplo,
             const int N, const float alpha,
             const float *X, const int incX,
             float *Ap);

void fb_ssyr2(const FB_LAYOUT Layout, const FB_UPLO Uplo,
              const int N, const float alpha,
              const float *X, const int incX,
              const float *Y, const int incY,
              float *A, const int lda);

void fb_sspr2(const FB_LAYOUT Layout, const FB_UPLO Uplo,
              const int N, const float alpha,
              const float *X, const int incX,
              const float *Y, const int incY,
              float *Ap);

/* Double precision */
void fb_dgemv(const FB_LAYOUT Layout, const FB_TRANSPOSE TransA,
              const int M, const int N, const double alpha,
              const double *A, const int lda,
              const double *X, const int incX,
              const double beta, double *Y, const int incY);

void fb_dsymv(const FB_LAYOUT Layout, const FB_UPLO Uplo,
              const int N, const double alpha,
              const double *A, const int lda,
              const double *X, const int incX,
              const double beta, double *Y, const int incY);

void fb_dgbmv(const FB_LAYOUT Layout, const FB_TRANSPOSE TransA,
              const int M, const int N, const int kl, const int ku,
              const double alpha, const double *A, const int lda,
              const double *X, const int incX,
              const double beta, double *Y, const int incY);

void fb_dsbmv(const FB_LAYOUT Layout, const FB_UPLO Uplo,
              const int N, const int k,
              const double alpha, const double *A, const int lda,
              const double *X, const int incX,
              const double beta, double *Y, const int incY);

void fb_dspmv(const FB_LAYOUT Layout, const FB_UPLO Uplo,
              const int N, const double alpha,
              const double *Ap, const double *X, const int incX,
              const double beta, double *Y, const int incY);

void fb_dtrmv(const FB_LAYOUT Layout, const FB_UPLO Uplo,
              const FB_TRANSPOSE TransA, const FB_DIAG Diag,
              const int N, const double *A, const int lda,
              double *X, const int incX);

void fb_dtbmv(const FB_LAYOUT Layout, const FB_UPLO Uplo,
              const FB_TRANSPOSE TransA, const FB_DIAG Diag,
              const int N, const int k,
              const double *A, const int lda,
              double *X, const int incX);

void fb_dtpmv(const FB_LAYOUT Layout, const FB_UPLO Uplo,
              const FB_TRANSPOSE TransA, const FB_DIAG Diag,
              const int N, const double *Ap,
              double *X, const int incX);

void fb_dtrsv(const FB_LAYOUT Layout, const FB_UPLO Uplo,
              const FB_TRANSPOSE TransA, const FB_DIAG Diag,
              const int N, const double *A, const int lda,
              double *X, const int incX);

void fb_dtbsv(const FB_LAYOUT Layout, const FB_UPLO Uplo,
              const FB_TRANSPOSE TransA, const FB_DIAG Diag,
              const int N, const int k,
              const double *A, const int lda,
              double *X, const int incX);

void fb_dtpsv(const FB_LAYOUT Layout, const FB_UPLO Uplo,
              const FB_TRANSPOSE TransA, const FB_DIAG Diag,
              const int N, const double *Ap,
              double *X, const int incX);

void fb_dger(const FB_LAYOUT Layout, const int M, const int N,
             const double alpha, const double *X, const int incX,
             const double *Y, const int incY,
             double *A, const int lda);

void fb_dsyr(const FB_LAYOUT Layout, const FB_UPLO Uplo,
             const int N, const double alpha,
             const double *X, const int incX,
             double *A, const int lda);

void fb_dspr(const FB_LAYOUT Layout, const FB_UPLO Uplo,
             const int N, const double alpha,
             const double *X, const int incX,
             double *Ap);

void fb_dsyr2(const FB_LAYOUT Layout, const FB_UPLO Uplo,
              const int N, const double alpha,
              const double *X, const int incX,
              const double *Y, const int incY,
              double *A, const int lda);

void fb_dspr2(const FB_LAYOUT Layout, const FB_UPLO Uplo,
              const int N, const double alpha,
              const double *X, const int incX,
              const double *Y, const int incY,
              double *Ap);

/* Complex single precision */
void fb_cgemv(const FB_LAYOUT Layout, const FB_TRANSPOSE TransA,
              const int M, const int N, const void *alpha,
              const void *A, const int lda,
              const void *X, const int incX,
              const void *beta, void *Y, const int incY);

void fb_chemv(const FB_LAYOUT Layout, const FB_UPLO Uplo,
              const int N, const void *alpha,
              const void *A, const int lda,
              const void *X, const int incX,
              const void *beta, void *Y, const int incY);

void fb_ctrmv(const FB_LAYOUT Layout, const FB_UPLO Uplo,
              const FB_TRANSPOSE TransA, const FB_DIAG Diag,
              const int N, const void *A, const int lda,
              void *X, const int incX);

void fb_ctrsv(const FB_LAYOUT Layout, const FB_UPLO Uplo,
              const FB_TRANSPOSE TransA, const FB_DIAG Diag,
              const int N, const void *A, const int lda,
              void *X, const int incX);

void fb_cgeru(const FB_LAYOUT Layout, const int M, const int N,
              const void *alpha, const void *X, const int incX,
              const void *Y, const int incY,
              void *A, const int lda);

void fb_cgerc(const FB_LAYOUT Layout, const int M, const int N,
              const void *alpha, const void *X, const int incX,
              const void *Y, const int incY,
              void *A, const int lda);

/* Complex double precision */
void fb_zgemv(const FB_LAYOUT Layout, const FB_TRANSPOSE TransA,
              const int M, const int N, const void *alpha,
              const void *A, const int lda,
              const void *X, const int incX,
              const void *beta, void *Y, const int incY);

void fb_zhemv(const FB_LAYOUT Layout, const FB_UPLO Uplo,
              const int N, const void *alpha,
              const void *A, const int lda,
              const void *X, const int incX,
              const void *beta, void *Y, const int incY);

void fb_ztrmv(const FB_LAYOUT Layout, const FB_UPLO Uplo,
              const FB_TRANSPOSE TransA, const FB_DIAG Diag,
              const int N, const void *A, const int lda,
              void *X, const int incX);

void fb_ztrsv(const FB_LAYOUT Layout, const FB_UPLO Uplo,
              const FB_TRANSPOSE TransA, const FB_DIAG Diag,
              const int N, const void *A, const int lda,
              void *X, const int incX);

void fb_zgeru(const FB_LAYOUT Layout, const int M, const int N,
              const void *alpha, const void *X, const int incX,
              const void *Y, const int incY,
              void *A, const int lda);

void fb_zgerc(const FB_LAYOUT Layout, const int M, const int N,
              const void *alpha, const void *X, const int incX,
              const void *Y, const int incY,
              void *A, const int lda);

/* ============================================================================
 * BLAS Level 3: Matrix-Matrix Operations
 * ========================================================================= */

/* Single precision */
void fb_sgemm(const FB_LAYOUT Layout,
              const FB_TRANSPOSE TransA, const FB_TRANSPOSE TransB,
              const int M, const int N, const int K,
              const float alpha,
              const float *A, const int lda,
              const float *B, const int ldb,
              const float beta,
              float *C, const int ldc);

void fb_ssymm(const FB_LAYOUT Layout, const FB_SIDE Side, const FB_UPLO Uplo,
              const int M, const int N,
              const float alpha,
              const float *A, const int lda,
              const float *B, const int ldb,
              const float beta,
              float *C, const int ldc);

void fb_ssyrk(const FB_LAYOUT Layout, const FB_UPLO Uplo, const FB_TRANSPOSE Trans,
              const int N, const int K,
              const float alpha,
              const float *A, const int lda,
              const float beta,
              float *C, const int ldc);

void fb_ssyr2k(const FB_LAYOUT Layout, const FB_UPLO Uplo, const FB_TRANSPOSE Trans,
               const int N, const int K,
               const float alpha,
               const float *A, const int lda,
               const float *B, const int ldb,
               const float beta,
               float *C, const int ldc);

void fb_strmm(const FB_LAYOUT Layout, const FB_SIDE Side,
              const FB_UPLO Uplo, const FB_TRANSPOSE TransA, const FB_DIAG Diag,
              const int M, const int N,
              const float alpha,
              const float *A, const int lda,
              float *B, const int ldb);

void fb_strsm(const FB_LAYOUT Layout, const FB_SIDE Side,
              const FB_UPLO Uplo, const FB_TRANSPOSE TransA, const FB_DIAG Diag,
              const int M, const int N,
              const float alpha,
              const float *A, const int lda,
              float *B, const int ldb);

/* Double precision */
void fb_dgemm(const FB_LAYOUT Layout,
              const FB_TRANSPOSE TransA, const FB_TRANSPOSE TransB,
              const int M, const int N, const int K,
              const double alpha,
              const double *A, const int lda,
              const double *B, const int ldb,
              const double beta,
              double *C, const int ldc);

void fb_dsymm(const FB_LAYOUT Layout, const FB_SIDE Side, const FB_UPLO Uplo,
              const int M, const int N,
              const double alpha,
              const double *A, const int lda,
              const double *B, const int ldb,
              const double beta,
              double *C, const int ldc);

void fb_dsyrk(const FB_LAYOUT Layout, const FB_UPLO Uplo, const FB_TRANSPOSE Trans,
              const int N, const int K,
              const double alpha,
              const double *A, const int lda,
              const double beta,
              double *C, const int ldc);

void fb_dsyr2k(const FB_LAYOUT Layout, const FB_UPLO Uplo, const FB_TRANSPOSE Trans,
               const int N, const int K,
               const double alpha,
               const double *A, const int lda,
               const double *B, const int ldb,
               const double beta,
               double *C, const int ldc);

void fb_dtrmm(const FB_LAYOUT Layout, const FB_SIDE Side,
              const FB_UPLO Uplo, const FB_TRANSPOSE TransA, const FB_DIAG Diag,
              const int M, const int N,
              const double alpha,
              const double *A, const int lda,
              double *B, const int ldb);

void fb_dtrsm(const FB_LAYOUT Layout, const FB_SIDE Side,
              const FB_UPLO Uplo, const FB_TRANSPOSE TransA, const FB_DIAG Diag,
              const int M, const int N,
              const double alpha,
              const double *A, const int lda,
              double *B, const int ldb);

/* Complex single precision */
void fb_cgemm(const FB_LAYOUT Layout,
              const FB_TRANSPOSE TransA, const FB_TRANSPOSE TransB,
              const int M, const int N, const int K,
              const void *alpha,
              const void *A, const int lda,
              const void *B, const int ldb,
              const void *beta,
              void *C, const int ldc);

void fb_csymm(const FB_LAYOUT Layout, const FB_SIDE Side, const FB_UPLO Uplo,
              const int M, const int N,
              const void *alpha,
              const void *A, const int lda,
              const void *B, const int ldb,
              const void *beta,
              void *C, const int ldc);

void fb_chemm(const FB_LAYOUT Layout, const FB_SIDE Side, const FB_UPLO Uplo,
              const int M, const int N,
              const void *alpha,
              const void *A, const int lda,
              const void *B, const int ldb,
              const void *beta,
              void *C, const int ldc);

void fb_csyrk(const FB_LAYOUT Layout, const FB_UPLO Uplo, const FB_TRANSPOSE Trans,
              const int N, const int K,
              const void *alpha,
              const void *A, const int lda,
              const void *beta,
              void *C, const int ldc);

void fb_cherk(const FB_LAYOUT Layout, const FB_UPLO Uplo, const FB_TRANSPOSE Trans,
              const int N, const int K,
              const float alpha,
              const void *A, const int lda,
              const float beta,
              void *C, const int ldc);

void fb_csyr2k(const FB_LAYOUT Layout, const FB_UPLO Uplo, const FB_TRANSPOSE Trans,
               const int N, const int K,
               const void *alpha,
               const void *A, const int lda,
               const void *B, const int ldb,
               const void *beta,
               void *C, const int ldc);

void fb_cher2k(const FB_LAYOUT Layout, const FB_UPLO Uplo, const FB_TRANSPOSE Trans,
               const int N, const int K,
               const void *alpha,
               const void *A, const int lda,
               const void *B, const int ldb,
               const float beta,
               void *C, const int ldc);

void fb_ctrmm(const FB_LAYOUT Layout, const FB_SIDE Side,
              const FB_UPLO Uplo, const FB_TRANSPOSE TransA, const FB_DIAG Diag,
              const int M, const int N,
              const void *alpha,
              const void *A, const int lda,
              void *B, const int ldb);

void fb_ctrsm(const FB_LAYOUT Layout, const FB_SIDE Side,
              const FB_UPLO Uplo, const FB_TRANSPOSE TransA, const FB_DIAG Diag,
              const int M, const int N,
              const void *alpha,
              const void *A, const int lda,
              void *B, const int ldb);

/* Complex double precision */
void fb_zgemm(const FB_LAYOUT Layout,
              const FB_TRANSPOSE TransA, const FB_TRANSPOSE TransB,
              const int M, const int N, const int K,
              const void *alpha,
              const void *A, const int lda,
              const void *B, const int ldb,
              const void *beta,
              void *C, const int ldc);

void fb_zsymm(const FB_LAYOUT Layout, const FB_SIDE Side, const FB_UPLO Uplo,
              const int M, const int N,
              const void *alpha,
              const void *A, const int lda,
              const void *B, const int ldb,
              const void *beta,
              void *C, const int ldc);

void fb_zhemm(const FB_LAYOUT Layout, const FB_SIDE Side, const FB_UPLO Uplo,
              const int M, const int N,
              const void *alpha,
              const void *A, const int lda,
              const void *B, const int ldb,
              const void *beta,
              void *C, const int ldc);

void fb_zsyrk(const FB_LAYOUT Layout, const FB_UPLO Uplo, const FB_TRANSPOSE Trans,
              const int N, const int K,
              const void *alpha,
              const void *A, const int lda,
              const void *beta,
              void *C, const int ldc);

void fb_zherk(const FB_LAYOUT Layout, const FB_UPLO Uplo, const FB_TRANSPOSE Trans,
              const int N, const int K,
              const double alpha,
              const void *A, const int lda,
              const double beta,
              void *C, const int ldc);

void fb_zsyr2k(const FB_LAYOUT Layout, const FB_UPLO Uplo, const FB_TRANSPOSE Trans,
               const int N, const int K,
               const void *alpha,
               const void *A, const int lda,
               const void *B, const int ldb,
               const void *beta,
               void *C, const int ldc);

void fb_zher2k(const FB_LAYOUT Layout, const FB_UPLO Uplo, const FB_TRANSPOSE Trans,
               const int N, const int K,
               const void *alpha,
               const void *A, const int lda,
               const void *B, const int ldb,
               const double beta,
               void *C, const int ldc);

void fb_ztrmm(const FB_LAYOUT Layout, const FB_SIDE Side,
              const FB_UPLO Uplo, const FB_TRANSPOSE TransA, const FB_DIAG Diag,
              const int M, const int N,
              const void *alpha,
              const void *A, const int lda,
              void *B, const int ldb);

void fb_ztrsm(const FB_LAYOUT Layout, const FB_SIDE Side,
              const FB_UPLO Uplo, const FB_TRANSPOSE TransA, const FB_DIAG Diag,
              const int M, const int N,
              const void *alpha,
              const void *A, const int lda,
              void *B, const int ldb);

#ifdef __cplusplus
}
#endif

#endif /* FASTER_BLASTER_H */
