/**
 * @file reference.c
 * @brief Reference BLAS backend — delegates to faster-blaster-reference
 *
 * All BLAS L1 / L2 / L3 implementations are provided by the companion
 * faster-blaster-reference repository: pure-C23, IEEE 754 compliant, fully
 * tested across all 2,265 operations.  There is no reason to maintain a
 * second copy here.
 *
 * LAPACK is still delegated to the in-tree reference_lapack.c because
 * faster-blaster-reference LAPACK coverage is not yet complete.  Once it
 * reaches 100% that include can be replaced by the same pattern.
 *
 * Layout note: the _ref functions use Fortran / column-major convention.
 * This backend therefore assumes column-major input — the convention used
 * throughout faster-blaster's test suite.  Row-major callers must transpose
 * their data before reaching this backend; in practice probe_score=5 ensures
 * a proper CBLAS backend is always preferred over this one.
 *
 * Plugin score: 5 — always available, chosen last.
 * Purpose     : correctness oracle and last-resort fallback.
 *
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "backend_interface.h"
#include "faster-blaster/backend_plugin.h"

/* ---- faster-blaster-reference: canonical BLAS implementations ---------- */
/* Include path is added by CMakeLists.txt:
 *   target_include_directories(faster-blaster PRIVATE
 *       <faster-blaster-reference>/include)
 */
#include "blas_complex.h"       /* float_complex = float _Complex            */
#include "blas_l1_reference.h"  /* saxpy_ref, sdot_ref, cdotu_ref, …         */
#include "blas_l2_reference.h"  /* sgemv_ref, ssymv_ref, …                   */
#include "blas_l3_reference.h"  /* sgemm_ref, ssymm_ref, …                   */

/* float_complex  ≡  fb_complex_float_t  ≡  float _Complex  (same ABI)
 * double_complex ≡  fb_complex_double_t ≡  double _Complex (same ABI)       */

#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ============================================================================
 * Enum → char conversion helpers
 * The _ref functions use single-character Fortran convention ('N','T','C',…)
 * while the vtable uses integer enums.
 * ========================================================================= */

static inline char trans_char(fb_transpose_t t) {
    switch (t) {
        case FB_NO_TRANS:  return 'N';
        case FB_TRANS:     return 'T';
        default:           return 'C'; /* FB_CONJ_TRANS */
    }
}
static inline char uplo_char(fb_uplo_t  u) { return (u == FB_UPPER) ? 'U' : 'L'; }
static inline char diag_char(fb_diag_t  d) { return (d == FB_UNIT)  ? 'U' : 'N'; }
static inline char side_char(fb_side_t  s) { return (s == FB_LEFT)  ? 'L' : 'R'; }

/* Convenience shorthand used in wrapper bodies */
#define FC(t) trans_char(t)
#define FU(u) uplo_char(u)
#define FD(d) diag_char(d)
#define FS(s) side_char(s)

/* ============================================================================
 * BLAS Level 1 — result-pointer wrappers (cdotu/zdotu/cdotc/zdotc)
 *
 * vtable convention  : void fn(result*, n, x, incx, y, incy)
 * _ref convention    : float_complex fn(n, x, incx, y, incy)  [return-by-value]
 * ========================================================================= */

static void w_cdotu(fb_complex_float_t *res, const int n,
                    const fb_complex_float_t *x, const int incx,
                    const fb_complex_float_t *y, const int incy) {
    *res = cdotu_ref(n, (const float_complex *)x, incx, (const float_complex *)y, incy);
}
static void w_zdotu(fb_complex_double_t *res, const int n,
                    const fb_complex_double_t *x, const int incx,
                    const fb_complex_double_t *y, const int incy) {
    *res = zdotu_ref(n, (const double_complex *)x, incx, (const double_complex *)y, incy);
}
static void w_cdotc(fb_complex_float_t *res, const int n,
                    const fb_complex_float_t *x, const int incx,
                    const fb_complex_float_t *y, const int incy) {
    *res = cdotc_ref(n, (const float_complex *)x, incx, (const float_complex *)y, incy);
}
static void w_zdotc(fb_complex_double_t *res, const int n,
                    const fb_complex_double_t *x, const int incx,
                    const fb_complex_double_t *y, const int incy) {
    *res = zdotc_ref(n, (const double_complex *)x, incx, (const double_complex *)y, incy);
}

/* ============================================================================
 * BLAS Level 2 — thin wrappers that strip fb_layout_t and convert enums
 *
 * Pattern: (void)lay; call _ref() with char params derived from the enums.
 * ========================================================================= */

/* --- GEMV ---------------------------------------------------------------- */
static void w_sgemv(fb_layout_t lay, fb_transpose_t tr,
                    int m, int n, float a, const float *A, int lda,
                    const float *x, int incx, float b, float *y, int incy)
    { (void)lay; sgemv_ref(FC(tr), m, n, a, A, lda, x, incx, b, y, incy); }
static void w_dgemv(fb_layout_t lay, fb_transpose_t tr,
                    int m, int n, double a, const double *A, int lda,
                    const double *x, int incx, double b, double *y, int incy)
    { (void)lay; dgemv_ref(FC(tr), m, n, a, A, lda, x, incx, b, y, incy); }
static void w_cgemv(fb_layout_t lay, fb_transpose_t tr,
                    int m, int n, fb_complex_float_t a,
                    const fb_complex_float_t *A, int lda,
                    const fb_complex_float_t *x, int incx,
                    fb_complex_float_t b, fb_complex_float_t *y, int incy) {
    (void)lay;
    cgemv_ref(FC(tr), m, n, (float_complex)a, (const float_complex *)A, lda,
              (const float_complex *)x, incx, (float_complex)b,
              (float_complex *)y, incy);
}
static void w_zgemv(fb_layout_t lay, fb_transpose_t tr,
                    int m, int n, fb_complex_double_t a,
                    const fb_complex_double_t *A, int lda,
                    const fb_complex_double_t *x, int incx,
                    fb_complex_double_t b, fb_complex_double_t *y, int incy) {
    (void)lay;
    zgemv_ref(FC(tr), m, n, (double_complex)a, (const double_complex *)A, lda,
              (const double_complex *)x, incx, (double_complex)b,
              (double_complex *)y, incy);
}

/* --- GBMV ---------------------------------------------------------------- */
static void w_sgbmv(fb_layout_t lay, fb_transpose_t tr,
                    int m, int n, int kl, int ku, float a,
                    const float *A, int lda, const float *x, int incx,
                    float b, float *y, int incy)
    { (void)lay; sgbmv_ref(FC(tr), m, n, kl, ku, a, A, lda, x, incx, b, y, incy); }
static void w_dgbmv(fb_layout_t lay, fb_transpose_t tr,
                    int m, int n, int kl, int ku, double a,
                    const double *A, int lda, const double *x, int incx,
                    double b, double *y, int incy)
    { (void)lay; dgbmv_ref(FC(tr), m, n, kl, ku, a, A, lda, x, incx, b, y, incy); }
static void w_cgbmv(fb_layout_t lay, fb_transpose_t tr,
                    int m, int n, int kl, int ku, fb_complex_float_t a,
                    const fb_complex_float_t *A, int lda,
                    const fb_complex_float_t *x, int incx,
                    fb_complex_float_t b, fb_complex_float_t *y, int incy) {
    (void)lay;
    cgbmv_ref(FC(tr), m, n, kl, ku, (float_complex)a,
              (const float_complex *)A, lda, (const float_complex *)x, incx,
              (float_complex)b, (float_complex *)y, incy);
}
static void w_zgbmv(fb_layout_t lay, fb_transpose_t tr,
                    int m, int n, int kl, int ku, fb_complex_double_t a,
                    const fb_complex_double_t *A, int lda,
                    const fb_complex_double_t *x, int incx,
                    fb_complex_double_t b, fb_complex_double_t *y, int incy) {
    (void)lay;
    zgbmv_ref(FC(tr), m, n, kl, ku, (double_complex)a,
              (const double_complex *)A, lda, (const double_complex *)x, incx,
              (double_complex)b, (double_complex *)y, incy);
}

/* --- SYMV / HEMV --------------------------------------------------------- */
static void w_ssymv(fb_layout_t lay, fb_uplo_t up,
                    int n, float a, const float *A, int lda,
                    const float *x, int incx, float b, float *y, int incy)
    { (void)lay; ssymv_ref(FU(up), n, a, A, lda, x, incx, b, y, incy); }
static void w_dsymv(fb_layout_t lay, fb_uplo_t up,
                    int n, double a, const double *A, int lda,
                    const double *x, int incx, double b, double *y, int incy)
    { (void)lay; dsymv_ref(FU(up), n, a, A, lda, x, incx, b, y, incy); }
static void w_csymv(fb_layout_t lay, fb_uplo_t up,
                    int n, fb_complex_float_t a,
                    const fb_complex_float_t *A, int lda,
                    const fb_complex_float_t *x, int incx,
                    fb_complex_float_t b, fb_complex_float_t *y, int incy) {
    (void)lay;
    csymv_ref(FU(up), n, (float_complex)a, (const float_complex *)A, lda,
              (const float_complex *)x, incx, (float_complex)b,
              (float_complex *)y, incy);
}
static void w_zsymv(fb_layout_t lay, fb_uplo_t up,
                    int n, fb_complex_double_t a,
                    const fb_complex_double_t *A, int lda,
                    const fb_complex_double_t *x, int incx,
                    fb_complex_double_t b, fb_complex_double_t *y, int incy) {
    (void)lay;
    zsymv_ref(FU(up), n, (double_complex)a, (const double_complex *)A, lda,
              (const double_complex *)x, incx, (double_complex)b,
              (double_complex *)y, incy);
}
static void w_chemv(fb_layout_t lay, fb_uplo_t up,
                    int n, fb_complex_float_t a,
                    const fb_complex_float_t *A, int lda,
                    const fb_complex_float_t *x, int incx,
                    fb_complex_float_t b, fb_complex_float_t *y, int incy) {
    (void)lay;
    chemv_ref(FU(up), n, (float_complex)a, (const float_complex *)A, lda,
              (const float_complex *)x, incx, (float_complex)b,
              (float_complex *)y, incy);
}
static void w_zhemv(fb_layout_t lay, fb_uplo_t up,
                    int n, fb_complex_double_t a,
                    const fb_complex_double_t *A, int lda,
                    const fb_complex_double_t *x, int incx,
                    fb_complex_double_t b, fb_complex_double_t *y, int incy) {
    (void)lay;
    zhemv_ref(FU(up), n, (double_complex)a, (const double_complex *)A, lda,
              (const double_complex *)x, incx, (double_complex)b,
              (double_complex *)y, incy);
}

/* --- SBMV / HBMV --------------------------------------------------------- */
static void w_ssbmv(fb_layout_t lay, fb_uplo_t up, int n, int k,
                    float a, const float *A, int lda,
                    const float *x, int incx, float b, float *y, int incy)
    { (void)lay; ssbmv_ref(FU(up), n, k, a, A, lda, x, incx, b, y, incy); }
static void w_dsbmv(fb_layout_t lay, fb_uplo_t up, int n, int k,
                    double a, const double *A, int lda,
                    const double *x, int incx, double b, double *y, int incy)
    { (void)lay; dsbmv_ref(FU(up), n, k, a, A, lda, x, incx, b, y, incy); }
static void w_chbmv(fb_layout_t lay, fb_uplo_t up, int n, int k,
                    fb_complex_float_t a, const fb_complex_float_t *A, int lda,
                    const fb_complex_float_t *x, int incx,
                    fb_complex_float_t b, fb_complex_float_t *y, int incy) {
    (void)lay;
    chbmv_ref(FU(up), n, k, (float_complex)a, (const float_complex *)A, lda,
              (const float_complex *)x, incx, (float_complex)b,
              (float_complex *)y, incy);
}
static void w_zhbmv(fb_layout_t lay, fb_uplo_t up, int n, int k,
                    fb_complex_double_t a, const fb_complex_double_t *A, int lda,
                    const fb_complex_double_t *x, int incx,
                    fb_complex_double_t b, fb_complex_double_t *y, int incy) {
    (void)lay;
    zhbmv_ref(FU(up), n, k, (double_complex)a, (const double_complex *)A, lda,
              (const double_complex *)x, incx, (double_complex)b,
              (double_complex *)y, incy);
}

/* --- SPMV / HPMV --------------------------------------------------------- */
static void w_sspmv(fb_layout_t lay, fb_uplo_t up, int n,
                    float a, const float *ap,
                    const float *x, int incx, float b, float *y, int incy)
    { (void)lay; sspmv_ref(FU(up), n, a, ap, x, incx, b, y, incy); }
static void w_dspmv(fb_layout_t lay, fb_uplo_t up, int n,
                    double a, const double *ap,
                    const double *x, int incx, double b, double *y, int incy)
    { (void)lay; dspmv_ref(FU(up), n, a, ap, x, incx, b, y, incy); }
static void w_chpmv(fb_layout_t lay, fb_uplo_t up, int n,
                    fb_complex_float_t a, const fb_complex_float_t *ap,
                    const fb_complex_float_t *x, int incx,
                    fb_complex_float_t b, fb_complex_float_t *y, int incy) {
    (void)lay;
    chpmv_ref(FU(up), n, (float_complex)a, (const float_complex *)ap,
              (const float_complex *)x, incx, (float_complex)b,
              (float_complex *)y, incy);
}
static void w_zhpmv(fb_layout_t lay, fb_uplo_t up, int n,
                    fb_complex_double_t a, const fb_complex_double_t *ap,
                    const fb_complex_double_t *x, int incx,
                    fb_complex_double_t b, fb_complex_double_t *y, int incy) {
    (void)lay;
    zhpmv_ref(FU(up), n, (double_complex)a, (const double_complex *)ap,
              (const double_complex *)x, incx, (double_complex)b,
              (double_complex *)y, incy);
}

/* --- TRMV ---------------------------------------------------------------- */
static void w_strmv(fb_layout_t lay, fb_uplo_t up, fb_transpose_t tr,
                    fb_diag_t dg, int n, const float *A, int lda,
                    float *x, int incx)
    { (void)lay; strmv_ref(FU(up), FC(tr), FD(dg), n, A, lda, x, incx); }
static void w_dtrmv(fb_layout_t lay, fb_uplo_t up, fb_transpose_t tr,
                    fb_diag_t dg, int n, const double *A, int lda,
                    double *x, int incx)
    { (void)lay; dtrmv_ref(FU(up), FC(tr), FD(dg), n, A, lda, x, incx); }
static void w_ctrmv(fb_layout_t lay, fb_uplo_t up, fb_transpose_t tr,
                    fb_diag_t dg, int n,
                    const fb_complex_float_t *A, int lda,
                    fb_complex_float_t *x, int incx) {
    (void)lay;
    ctrmv_ref(FU(up), FC(tr), FD(dg), n, (const float_complex *)A, lda,
              (float_complex *)x, incx);
}
static void w_ztrmv(fb_layout_t lay, fb_uplo_t up, fb_transpose_t tr,
                    fb_diag_t dg, int n,
                    const fb_complex_double_t *A, int lda,
                    fb_complex_double_t *x, int incx) {
    (void)lay;
    ztrmv_ref(FU(up), FC(tr), FD(dg), n, (const double_complex *)A, lda,
              (double_complex *)x, incx);
}

/* --- TBMV ---------------------------------------------------------------- */
static void w_stbmv(fb_layout_t lay, fb_uplo_t up, fb_transpose_t tr,
                    fb_diag_t dg, int n, int k,
                    const float *A, int lda, float *x, int incx)
    { (void)lay; stbmv_ref(FU(up), FC(tr), FD(dg), n, k, A, lda, x, incx); }
static void w_dtbmv(fb_layout_t lay, fb_uplo_t up, fb_transpose_t tr,
                    fb_diag_t dg, int n, int k,
                    const double *A, int lda, double *x, int incx)
    { (void)lay; dtbmv_ref(FU(up), FC(tr), FD(dg), n, k, A, lda, x, incx); }
static void w_ctbmv(fb_layout_t lay, fb_uplo_t up, fb_transpose_t tr,
                    fb_diag_t dg, int n, int k,
                    const fb_complex_float_t *A, int lda,
                    fb_complex_float_t *x, int incx) {
    (void)lay;
    ctbmv_ref(FU(up), FC(tr), FD(dg), n, k, (const float_complex *)A, lda,
              (float_complex *)x, incx);
}
static void w_ztbmv(fb_layout_t lay, fb_uplo_t up, fb_transpose_t tr,
                    fb_diag_t dg, int n, int k,
                    const fb_complex_double_t *A, int lda,
                    fb_complex_double_t *x, int incx) {
    (void)lay;
    ztbmv_ref(FU(up), FC(tr), FD(dg), n, k, (const double_complex *)A, lda,
              (double_complex *)x, incx);
}

/* --- TPMV ---------------------------------------------------------------- */
static void w_stpmv(fb_layout_t lay, fb_uplo_t up, fb_transpose_t tr,
                    fb_diag_t dg, int n, const float *ap, float *x, int incx)
    { (void)lay; stpmv_ref(FU(up), FC(tr), FD(dg), n, ap, x, incx); }
static void w_dtpmv(fb_layout_t lay, fb_uplo_t up, fb_transpose_t tr,
                    fb_diag_t dg, int n, const double *ap, double *x, int incx)
    { (void)lay; dtpmv_ref(FU(up), FC(tr), FD(dg), n, ap, x, incx); }
static void w_ctpmv(fb_layout_t lay, fb_uplo_t up, fb_transpose_t tr,
                    fb_diag_t dg, int n,
                    const fb_complex_float_t *ap, fb_complex_float_t *x, int incx) {
    (void)lay;
    ctpmv_ref(FU(up), FC(tr), FD(dg), n, (const float_complex *)ap,
              (float_complex *)x, incx);
}
static void w_ztpmv(fb_layout_t lay, fb_uplo_t up, fb_transpose_t tr,
                    fb_diag_t dg, int n,
                    const fb_complex_double_t *ap, fb_complex_double_t *x, int incx) {
    (void)lay;
    ztpmv_ref(FU(up), FC(tr), FD(dg), n, (const double_complex *)ap,
              (double_complex *)x, incx);
}

/* --- TRSV ---------------------------------------------------------------- */
static void w_strsv(fb_layout_t lay, fb_uplo_t up, fb_transpose_t tr,
                    fb_diag_t dg, int n, const float *A, int lda,
                    float *x, int incx)
    { (void)lay; strsv_ref(FU(up), FC(tr), FD(dg), n, A, lda, x, incx); }
static void w_dtrsv(fb_layout_t lay, fb_uplo_t up, fb_transpose_t tr,
                    fb_diag_t dg, int n, const double *A, int lda,
                    double *x, int incx)
    { (void)lay; dtrsv_ref(FU(up), FC(tr), FD(dg), n, A, lda, x, incx); }
static void w_ctrsv(fb_layout_t lay, fb_uplo_t up, fb_transpose_t tr,
                    fb_diag_t dg, int n,
                    const fb_complex_float_t *A, int lda,
                    fb_complex_float_t *x, int incx) {
    (void)lay;
    ctrsv_ref(FU(up), FC(tr), FD(dg), n, (const float_complex *)A, lda,
              (float_complex *)x, incx);
}
static void w_ztrsv(fb_layout_t lay, fb_uplo_t up, fb_transpose_t tr,
                    fb_diag_t dg, int n,
                    const fb_complex_double_t *A, int lda,
                    fb_complex_double_t *x, int incx) {
    (void)lay;
    ztrsv_ref(FU(up), FC(tr), FD(dg), n, (const double_complex *)A, lda,
              (double_complex *)x, incx);
}

/* --- TBSV ---------------------------------------------------------------- */
static void w_stbsv(fb_layout_t lay, fb_uplo_t up, fb_transpose_t tr,
                    fb_diag_t dg, int n, int k,
                    const float *A, int lda, float *x, int incx)
    { (void)lay; stbsv_ref(FU(up), FC(tr), FD(dg), n, k, A, lda, x, incx); }
static void w_dtbsv(fb_layout_t lay, fb_uplo_t up, fb_transpose_t tr,
                    fb_diag_t dg, int n, int k,
                    const double *A, int lda, double *x, int incx)
    { (void)lay; dtbsv_ref(FU(up), FC(tr), FD(dg), n, k, A, lda, x, incx); }
static void w_ctbsv(fb_layout_t lay, fb_uplo_t up, fb_transpose_t tr,
                    fb_diag_t dg, int n, int k,
                    const fb_complex_float_t *A, int lda,
                    fb_complex_float_t *x, int incx) {
    (void)lay;
    ctbsv_ref(FU(up), FC(tr), FD(dg), n, k, (const float_complex *)A, lda,
              (float_complex *)x, incx);
}
static void w_ztbsv(fb_layout_t lay, fb_uplo_t up, fb_transpose_t tr,
                    fb_diag_t dg, int n, int k,
                    const fb_complex_double_t *A, int lda,
                    fb_complex_double_t *x, int incx) {
    (void)lay;
    ztbsv_ref(FU(up), FC(tr), FD(dg), n, k, (const double_complex *)A, lda,
              (double_complex *)x, incx);
}

/* --- TPSV ---------------------------------------------------------------- */
static void w_stpsv(fb_layout_t lay, fb_uplo_t up, fb_transpose_t tr,
                    fb_diag_t dg, int n, const float *ap, float *x, int incx)
    { (void)lay; stpsv_ref(FU(up), FC(tr), FD(dg), n, ap, x, incx); }
static void w_dtpsv(fb_layout_t lay, fb_uplo_t up, fb_transpose_t tr,
                    fb_diag_t dg, int n, const double *ap, double *x, int incx)
    { (void)lay; dtpsv_ref(FU(up), FC(tr), FD(dg), n, ap, x, incx); }
static void w_ctpsv(fb_layout_t lay, fb_uplo_t up, fb_transpose_t tr,
                    fb_diag_t dg, int n,
                    const fb_complex_float_t *ap, fb_complex_float_t *x, int incx) {
    (void)lay;
    ctpsv_ref(FU(up), FC(tr), FD(dg), n, (const float_complex *)ap,
              (float_complex *)x, incx);
}
static void w_ztpsv(fb_layout_t lay, fb_uplo_t up, fb_transpose_t tr,
                    fb_diag_t dg, int n,
                    const fb_complex_double_t *ap, fb_complex_double_t *x, int incx) {
    (void)lay;
    ztpsv_ref(FU(up), FC(tr), FD(dg), n, (const double_complex *)ap,
              (double_complex *)x, incx);
}

/* --- SYR / HER ----------------------------------------------------------- */
static void w_ssyr(fb_layout_t lay, fb_uplo_t up, int n,
                   float a, const float *x, int incx, float *A, int lda)
    { (void)lay; ssyr_ref(FU(up), n, a, x, incx, A, lda); }
static void w_dsyr(fb_layout_t lay, fb_uplo_t up, int n,
                   double a, const double *x, int incx, double *A, int lda)
    { (void)lay; dsyr_ref(FU(up), n, a, x, incx, A, lda); }
static void w_csyr(fb_layout_t lay, fb_uplo_t up, int n,
                   fb_complex_float_t a,
                   const fb_complex_float_t *x, int incx,
                   fb_complex_float_t *A, int lda) {
    (void)lay;
    csyr_ref(FU(up), n, (float_complex)a,
             (const float_complex *)x, incx, (float_complex *)A, lda);
}
static void w_zsyr(fb_layout_t lay, fb_uplo_t up, int n,
                   fb_complex_double_t a,
                   const fb_complex_double_t *x, int incx,
                   fb_complex_double_t *A, int lda) {
    (void)lay;
    zsyr_ref(FU(up), n, (double_complex)a,
             (const double_complex *)x, incx, (double_complex *)A, lda);
}
static void w_cher(fb_layout_t lay, fb_uplo_t up, int n,
                   float a, const fb_complex_float_t *x, int incx,
                   fb_complex_float_t *A, int lda) {
    (void)lay;
    cher_ref(FU(up), n, a, (const float_complex *)x, incx,
             (float_complex *)A, lda);
}
static void w_zher(fb_layout_t lay, fb_uplo_t up, int n,
                   double a, const fb_complex_double_t *x, int incx,
                   fb_complex_double_t *A, int lda) {
    (void)lay;
    zher_ref(FU(up), n, a, (const double_complex *)x, incx,
             (double_complex *)A, lda);
}

/* --- SYR2 / HER2 --------------------------------------------------------- */
static void w_ssyr2(fb_layout_t lay, fb_uplo_t up, int n,
                    float a, const float *x, int incx,
                    const float *y, int incy, float *A, int lda)
    { (void)lay; ssyr2_ref(FU(up), n, a, x, incx, y, incy, A, lda); }
static void w_dsyr2(fb_layout_t lay, fb_uplo_t up, int n,
                    double a, const double *x, int incx,
                    const double *y, int incy, double *A, int lda)
    { (void)lay; dsyr2_ref(FU(up), n, a, x, incx, y, incy, A, lda); }
static void w_cher2(fb_layout_t lay, fb_uplo_t up, int n,
                    fb_complex_float_t a,
                    const fb_complex_float_t *x, int incx,
                    const fb_complex_float_t *y, int incy,
                    fb_complex_float_t *A, int lda) {
    (void)lay;
    cher2_ref(FU(up), n, (float_complex)a,
              (const float_complex *)x, incx, (const float_complex *)y, incy,
              (float_complex *)A, lda);
}
static void w_zher2(fb_layout_t lay, fb_uplo_t up, int n,
                    fb_complex_double_t a,
                    const fb_complex_double_t *x, int incx,
                    const fb_complex_double_t *y, int incy,
                    fb_complex_double_t *A, int lda) {
    (void)lay;
    zher2_ref(FU(up), n, (double_complex)a,
              (const double_complex *)x, incx, (const double_complex *)y, incy,
              (double_complex *)A, lda);
}

/* --- GER / GERU / GERC --------------------------------------------------- */
static void w_sger(fb_layout_t lay, int m, int n,
                   float a, const float *x, int incx,
                   const float *y, int incy, float *A, int lda)
    { (void)lay; sger_ref(m, n, a, x, incx, y, incy, A, lda); }
static void w_dger(fb_layout_t lay, int m, int n,
                   double a, const double *x, int incx,
                   const double *y, int incy, double *A, int lda)
    { (void)lay; dger_ref(m, n, a, x, incx, y, incy, A, lda); }
static void w_cgeru(fb_layout_t lay, int m, int n,
                    fb_complex_float_t a,
                    const fb_complex_float_t *x, int incx,
                    const fb_complex_float_t *y, int incy,
                    fb_complex_float_t *A, int lda) {
    (void)lay;
    cgeru_ref(m, n, (float_complex)a,
              (const float_complex *)x, incx, (const float_complex *)y, incy,
              (float_complex *)A, lda);
}
static void w_zgeru(fb_layout_t lay, int m, int n,
                    fb_complex_double_t a,
                    const fb_complex_double_t *x, int incx,
                    const fb_complex_double_t *y, int incy,
                    fb_complex_double_t *A, int lda) {
    (void)lay;
    zgeru_ref(m, n, (double_complex)a,
              (const double_complex *)x, incx, (const double_complex *)y, incy,
              (double_complex *)A, lda);
}
static void w_cgerc(fb_layout_t lay, int m, int n,
                    fb_complex_float_t a,
                    const fb_complex_float_t *x, int incx,
                    const fb_complex_float_t *y, int incy,
                    fb_complex_float_t *A, int lda) {
    (void)lay;
    cgerc_ref(m, n, (float_complex)a,
              (const float_complex *)x, incx, (const float_complex *)y, incy,
              (float_complex *)A, lda);
}
static void w_zgerc(fb_layout_t lay, int m, int n,
                    fb_complex_double_t a,
                    const fb_complex_double_t *x, int incx,
                    const fb_complex_double_t *y, int incy,
                    fb_complex_double_t *A, int lda) {
    (void)lay;
    zgerc_ref(m, n, (double_complex)a,
              (const double_complex *)x, incx, (const double_complex *)y, incy,
              (double_complex *)A, lda);
}

/* --- SPR / HPR ----------------------------------------------------------- */
static void w_sspr(fb_layout_t lay, fb_uplo_t up, int n,
                   float a, const float *x, int incx, float *ap)
    { (void)lay; sspr_ref(FU(up), n, a, x, incx, ap); }
static void w_dspr(fb_layout_t lay, fb_uplo_t up, int n,
                   double a, const double *x, int incx, double *ap)
    { (void)lay; dspr_ref(FU(up), n, a, x, incx, ap); }
static void w_chpr(fb_layout_t lay, fb_uplo_t up, int n,
                   float a, const fb_complex_float_t *x, int incx,
                   fb_complex_float_t *ap) {
    (void)lay;
    chpr_ref(FU(up), n, a, (const float_complex *)x, incx, (float_complex *)ap);
}
static void w_zhpr(fb_layout_t lay, fb_uplo_t up, int n,
                   double a, const fb_complex_double_t *x, int incx,
                   fb_complex_double_t *ap) {
    (void)lay;
    zhpr_ref(FU(up), n, a, (const double_complex *)x, incx,
             (double_complex *)ap);
}

/* --- SPR2 / HPR2 --------------------------------------------------------- */
static void w_sspr2(fb_layout_t lay, fb_uplo_t up, int n,
                    float a, const float *x, int incx,
                    const float *y, int incy, float *ap)
    { (void)lay; sspr2_ref(FU(up), n, a, x, incx, y, incy, ap); }
static void w_dspr2(fb_layout_t lay, fb_uplo_t up, int n,
                    double a, const double *x, int incx,
                    const double *y, int incy, double *ap)
    { (void)lay; dspr2_ref(FU(up), n, a, x, incx, y, incy, ap); }
static void w_chpr2(fb_layout_t lay, fb_uplo_t up, int n,
                    fb_complex_float_t a,
                    const fb_complex_float_t *x, int incx,
                    const fb_complex_float_t *y, int incy,
                    fb_complex_float_t *ap) {
    (void)lay;
    chpr2_ref(FU(up), n, (float_complex)a,
              (const float_complex *)x, incx, (const float_complex *)y, incy,
              (float_complex *)ap);
}
static void w_zhpr2(fb_layout_t lay, fb_uplo_t up, int n,
                    fb_complex_double_t a,
                    const fb_complex_double_t *x, int incx,
                    const fb_complex_double_t *y, int incy,
                    fb_complex_double_t *ap) {
    (void)lay;
    zhpr2_ref(FU(up), n, (double_complex)a,
              (const double_complex *)x, incx, (const double_complex *)y, incy,
              (double_complex *)ap);
}

/* ============================================================================
 * BLAS Level 3
 * ========================================================================= */

/* --- GEMM ---------------------------------------------------------------- */
static void w_sgemm(fb_layout_t lay,
                    fb_transpose_t ta, fb_transpose_t tb,
                    int m, int n, int k, float a,
                    const float *A, int lda, const float *B, int ldb,
                    float b, float *C, int ldc)
    { (void)lay; sgemm_ref(FC(ta), FC(tb), m, n, k, a, A, lda, B, ldb, b, C, ldc); }
static void w_dgemm(fb_layout_t lay,
                    fb_transpose_t ta, fb_transpose_t tb,
                    int m, int n, int k, double a,
                    const double *A, int lda, const double *B, int ldb,
                    double b, double *C, int ldc)
    { (void)lay; dgemm_ref(FC(ta), FC(tb), m, n, k, a, A, lda, B, ldb, b, C, ldc); }
static void w_cgemm(fb_layout_t lay,
                    fb_transpose_t ta, fb_transpose_t tb,
                    int m, int n, int k,
                    fb_complex_float_t a,
                    const fb_complex_float_t *A, int lda,
                    const fb_complex_float_t *B, int ldb,
                    fb_complex_float_t b,
                    fb_complex_float_t *C, int ldc) {
    (void)lay;
    cgemm_ref(FC(ta), FC(tb), m, n, k, (float_complex)a,
              (const float_complex *)A, lda, (const float_complex *)B, ldb,
              (float_complex)b, (float_complex *)C, ldc);
}
static void w_zgemm(fb_layout_t lay,
                    fb_transpose_t ta, fb_transpose_t tb,
                    int m, int n, int k,
                    fb_complex_double_t a,
                    const fb_complex_double_t *A, int lda,
                    const fb_complex_double_t *B, int ldb,
                    fb_complex_double_t b,
                    fb_complex_double_t *C, int ldc) {
    (void)lay;
    zgemm_ref(FC(ta), FC(tb), m, n, k, (double_complex)a,
              (const double_complex *)A, lda, (const double_complex *)B, ldb,
              (double_complex)b, (double_complex *)C, ldc);
}

/* --- SYMM / HEMM --------------------------------------------------------- */
static void w_ssymm(fb_layout_t lay, fb_side_t si, fb_uplo_t up,
                    int m, int n, float a,
                    const float *A, int lda, const float *B, int ldb,
                    float b, float *C, int ldc)
    { (void)lay; ssymm_ref(FS(si), FU(up), m, n, a, A, lda, B, ldb, b, C, ldc); }
static void w_dsymm(fb_layout_t lay, fb_side_t si, fb_uplo_t up,
                    int m, int n, double a,
                    const double *A, int lda, const double *B, int ldb,
                    double b, double *C, int ldc)
    { (void)lay; dsymm_ref(FS(si), FU(up), m, n, a, A, lda, B, ldb, b, C, ldc); }
static void w_csymm(fb_layout_t lay, fb_side_t si, fb_uplo_t up,
                    int m, int n, fb_complex_float_t a,
                    const fb_complex_float_t *A, int lda,
                    const fb_complex_float_t *B, int ldb,
                    fb_complex_float_t b, fb_complex_float_t *C, int ldc) {
    (void)lay;
    csymm_ref(FS(si), FU(up), m, n, (float_complex)a,
              (const float_complex *)A, lda, (const float_complex *)B, ldb,
              (float_complex)b, (float_complex *)C, ldc);
}
static void w_zsymm(fb_layout_t lay, fb_side_t si, fb_uplo_t up,
                    int m, int n, fb_complex_double_t a,
                    const fb_complex_double_t *A, int lda,
                    const fb_complex_double_t *B, int ldb,
                    fb_complex_double_t b, fb_complex_double_t *C, int ldc) {
    (void)lay;
    zsymm_ref(FS(si), FU(up), m, n, (double_complex)a,
              (const double_complex *)A, lda, (const double_complex *)B, ldb,
              (double_complex)b, (double_complex *)C, ldc);
}
static void w_chemm(fb_layout_t lay, fb_side_t si, fb_uplo_t up,
                    int m, int n, fb_complex_float_t a,
                    const fb_complex_float_t *A, int lda,
                    const fb_complex_float_t *B, int ldb,
                    fb_complex_float_t b, fb_complex_float_t *C, int ldc) {
    (void)lay;
    chemm_ref(FS(si), FU(up), m, n, (float_complex)a,
              (const float_complex *)A, lda, (const float_complex *)B, ldb,
              (float_complex)b, (float_complex *)C, ldc);
}
static void w_zhemm(fb_layout_t lay, fb_side_t si, fb_uplo_t up,
                    int m, int n, fb_complex_double_t a,
                    const fb_complex_double_t *A, int lda,
                    const fb_complex_double_t *B, int ldb,
                    fb_complex_double_t b, fb_complex_double_t *C, int ldc) {
    (void)lay;
    zhemm_ref(FS(si), FU(up), m, n, (double_complex)a,
              (const double_complex *)A, lda, (const double_complex *)B, ldb,
              (double_complex)b, (double_complex *)C, ldc);
}

/* --- SYRK / HERK --------------------------------------------------------- */
static void w_ssyrk(fb_layout_t lay, fb_uplo_t up, fb_transpose_t tr,
                    int n, int k, float a, const float *A, int lda,
                    float b, float *C, int ldc)
    { (void)lay; ssyrk_ref(FU(up), FC(tr), n, k, a, A, lda, b, C, ldc); }
static void w_dsyrk(fb_layout_t lay, fb_uplo_t up, fb_transpose_t tr,
                    int n, int k, double a, const double *A, int lda,
                    double b, double *C, int ldc)
    { (void)lay; dsyrk_ref(FU(up), FC(tr), n, k, a, A, lda, b, C, ldc); }
static void w_csyrk(fb_layout_t lay, fb_uplo_t up, fb_transpose_t tr,
                    int n, int k, fb_complex_float_t a,
                    const fb_complex_float_t *A, int lda,
                    fb_complex_float_t b, fb_complex_float_t *C, int ldc) {
    (void)lay;
    csyrk_ref(FU(up), FC(tr), n, k, (float_complex)a,
              (const float_complex *)A, lda, (float_complex)b,
              (float_complex *)C, ldc);
}
static void w_zsyrk(fb_layout_t lay, fb_uplo_t up, fb_transpose_t tr,
                    int n, int k, fb_complex_double_t a,
                    const fb_complex_double_t *A, int lda,
                    fb_complex_double_t b, fb_complex_double_t *C, int ldc) {
    (void)lay;
    zsyrk_ref(FU(up), FC(tr), n, k, (double_complex)a,
              (const double_complex *)A, lda, (double_complex)b,
              (double_complex *)C, ldc);
}
static void w_cherk(fb_layout_t lay, fb_uplo_t up, fb_transpose_t tr,
                    int n, int k, float a,
                    const fb_complex_float_t *A, int lda,
                    float b, fb_complex_float_t *C, int ldc) {
    (void)lay;
    cherk_ref(FU(up), FC(tr), n, k, a, (const float_complex *)A, lda,
              b, (float_complex *)C, ldc);
}
static void w_zherk(fb_layout_t lay, fb_uplo_t up, fb_transpose_t tr,
                    int n, int k, double a,
                    const fb_complex_double_t *A, int lda,
                    double b, fb_complex_double_t *C, int ldc) {
    (void)lay;
    zherk_ref(FU(up), FC(tr), n, k, a, (const double_complex *)A, lda,
              b, (double_complex *)C, ldc);
}

/* --- SYR2K / HER2K ------------------------------------------------------- */
static void w_ssyr2k(fb_layout_t lay, fb_uplo_t up, fb_transpose_t tr,
                     int n, int k, float a,
                     const float *A, int lda, const float *B, int ldb,
                     float b, float *C, int ldc)
    { (void)lay; ssyr2k_ref(FU(up), FC(tr), n, k, a, A, lda, B, ldb, b, C, ldc); }
static void w_dsyr2k(fb_layout_t lay, fb_uplo_t up, fb_transpose_t tr,
                     int n, int k, double a,
                     const double *A, int lda, const double *B, int ldb,
                     double b, double *C, int ldc)
    { (void)lay; dsyr2k_ref(FU(up), FC(tr), n, k, a, A, lda, B, ldb, b, C, ldc); }
static void w_csyr2k(fb_layout_t lay, fb_uplo_t up, fb_transpose_t tr,
                     int n, int k, fb_complex_float_t a,
                     const fb_complex_float_t *A, int lda,
                     const fb_complex_float_t *B, int ldb,
                     fb_complex_float_t b, fb_complex_float_t *C, int ldc) {
    (void)lay;
    csyr2k_ref(FU(up), FC(tr), n, k, (float_complex)a,
               (const float_complex *)A, lda, (const float_complex *)B, ldb,
               (float_complex)b, (float_complex *)C, ldc);
}
static void w_zsyr2k(fb_layout_t lay, fb_uplo_t up, fb_transpose_t tr,
                     int n, int k, fb_complex_double_t a,
                     const fb_complex_double_t *A, int lda,
                     const fb_complex_double_t *B, int ldb,
                     fb_complex_double_t b, fb_complex_double_t *C, int ldc) {
    (void)lay;
    zsyr2k_ref(FU(up), FC(tr), n, k, (double_complex)a,
               (const double_complex *)A, lda, (const double_complex *)B, ldb,
               (double_complex)b, (double_complex *)C, ldc);
}
static void w_cher2k(fb_layout_t lay, fb_uplo_t up, fb_transpose_t tr,
                     int n, int k, fb_complex_float_t a,
                     const fb_complex_float_t *A, int lda,
                     const fb_complex_float_t *B, int ldb,
                     float b, fb_complex_float_t *C, int ldc) {
    (void)lay;
    cher2k_ref(FU(up), FC(tr), n, k, (float_complex)a,
               (const float_complex *)A, lda, (const float_complex *)B, ldb,
               b, (float_complex *)C, ldc);
}
static void w_zher2k(fb_layout_t lay, fb_uplo_t up, fb_transpose_t tr,
                     int n, int k, fb_complex_double_t a,
                     const fb_complex_double_t *A, int lda,
                     const fb_complex_double_t *B, int ldb,
                     double b, fb_complex_double_t *C, int ldc) {
    (void)lay;
    zher2k_ref(FU(up), FC(tr), n, k, (double_complex)a,
               (const double_complex *)A, lda, (const double_complex *)B, ldb,
               b, (double_complex *)C, ldc);
}

/* --- TRMM ---------------------------------------------------------------- */
static void w_strmm(fb_layout_t lay, fb_side_t si, fb_uplo_t up,
                    fb_transpose_t tr, fb_diag_t dg,
                    int m, int n, float a,
                    const float *A, int lda, float *B, int ldb)
    { (void)lay; strmm_ref(FS(si), FU(up), FC(tr), FD(dg), m, n, a, A, lda, B, ldb); }
static void w_dtrmm(fb_layout_t lay, fb_side_t si, fb_uplo_t up,
                    fb_transpose_t tr, fb_diag_t dg,
                    int m, int n, double a,
                    const double *A, int lda, double *B, int ldb)
    { (void)lay; dtrmm_ref(FS(si), FU(up), FC(tr), FD(dg), m, n, a, A, lda, B, ldb); }
static void w_ctrmm(fb_layout_t lay, fb_side_t si, fb_uplo_t up,
                    fb_transpose_t tr, fb_diag_t dg,
                    int m, int n, fb_complex_float_t a,
                    const fb_complex_float_t *A, int lda,
                    fb_complex_float_t *B, int ldb) {
    (void)lay;
    ctrmm_ref(FS(si), FU(up), FC(tr), FD(dg), m, n, (float_complex)a,
              (const float_complex *)A, lda, (float_complex *)B, ldb);
}
static void w_ztrmm(fb_layout_t lay, fb_side_t si, fb_uplo_t up,
                    fb_transpose_t tr, fb_diag_t dg,
                    int m, int n, fb_complex_double_t a,
                    const fb_complex_double_t *A, int lda,
                    fb_complex_double_t *B, int ldb) {
    (void)lay;
    ztrmm_ref(FS(si), FU(up), FC(tr), FD(dg), m, n, (double_complex)a,
              (const double_complex *)A, lda, (double_complex *)B, ldb);
}

/* --- TRSM ---------------------------------------------------------------- */
static void w_strsm(fb_layout_t lay, fb_side_t si, fb_uplo_t up,
                    fb_transpose_t tr, fb_diag_t dg,
                    int m, int n, float a,
                    const float *A, int lda, float *B, int ldb)
    { (void)lay; strsm_ref(FS(si), FU(up), FC(tr), FD(dg), m, n, a, A, lda, B, ldb); }
static void w_dtrsm(fb_layout_t lay, fb_side_t si, fb_uplo_t up,
                    fb_transpose_t tr, fb_diag_t dg,
                    int m, int n, double a,
                    const double *A, int lda, double *B, int ldb)
    { (void)lay; dtrsm_ref(FS(si), FU(up), FC(tr), FD(dg), m, n, a, A, lda, B, ldb); }
static void w_ctrsm(fb_layout_t lay, fb_side_t si, fb_uplo_t up,
                    fb_transpose_t tr, fb_diag_t dg,
                    int m, int n, fb_complex_float_t a,
                    const fb_complex_float_t *A, int lda,
                    fb_complex_float_t *B, int ldb) {
    (void)lay;
    ctrsm_ref(FS(si), FU(up), FC(tr), FD(dg), m, n, (float_complex)a,
              (const float_complex *)A, lda, (float_complex *)B, ldb);
}
static void w_ztrsm(fb_layout_t lay, fb_side_t si, fb_uplo_t up,
                    fb_transpose_t tr, fb_diag_t dg,
                    int m, int n, fb_complex_double_t a,
                    const fb_complex_double_t *A, int lda,
                    fb_complex_double_t *B, int ldb) {
    (void)lay;
    ztrsm_ref(FS(si), FU(up), FC(tr), FD(dg), m, n, (double_complex)a,
              (const double_complex *)A, lda, (double_complex *)B, ldb);
}

/* ============================================================================
 * LAPACK — in-tree implementations until faster-blaster-reference reaches
 * 100% LAPACK coverage.  The _ref functions here already match the vtable
 * signatures (they take fb_layout_t directly).
 *
 * TODO: replace this include once faster-blaster-reference has all LAPACK ops
 * ========================================================================= */
#include "reference_lapack.c"

/* ============================================================================
 * Misc backend extension stubs (CPU reference needs no GPU memory management)
 * ========================================================================= */

static uint32_t reference_get_capabilities(void *h) {
    (void)h;
    return FB_PLUGIN_CAP_CPU | FB_PLUGIN_CAP_LEVEL1 | FB_PLUGIN_CAP_LEVEL2
         | FB_PLUGIN_CAP_LEVEL3 | FB_PLUGIN_CAP_SINGLE_PREC
         | FB_PLUGIN_CAP_DOUBLE_PREC | FB_PLUGIN_CAP_COMPLEX;
}
static int  reference_get_num_threads(void *h) { (void)h; return 1; }
static void reference_set_num_threads(void *h, int n) { (void)h; (void)n; }

/* ============================================================================
 * Backend vtable
 * ========================================================================= */

const fb_backend_vtable_t *fb_reference_backend(void) {
    static fb_backend_vtable_t vt = {0};
    static int ready = 0;
    if (ready) return &vt;

    vt.info.name              = "reference";
    vt.info.version           = "2.0.0";
    vt.info.vendor            = "faster-blaster-reference";
    vt.info.capabilities      = FB_CAP_LEVEL1 | FB_CAP_LEVEL2 | FB_CAP_LEVEL3;
    vt.info.hw_type           = FB_HW_CPU_INTEL;
    vt.info.thread_safe       = true;
    vt.info.min_efficient_size = 0;

    /* ---- Level 1: direct casts (identical signatures) ---- */
    vt.scopy   = (fb_scopy_fn)scopy_ref;
    vt.dcopy   = (fb_dcopy_fn)dcopy_ref;
    vt.ccopy   = (fb_ccopy_fn)ccopy_ref;
    vt.zcopy   = (fb_zcopy_fn)zcopy_ref;

    vt.sswap   = (fb_sswap_fn)sswap_ref;
    vt.dswap   = (fb_dswap_fn)dswap_ref;
    vt.cswap   = (fb_cswap_fn)cswap_ref;
    vt.zswap   = (fb_zswap_fn)zswap_ref;

    vt.sscal   = (fb_sscal_fn)sscal_ref;
    vt.dscal   = (fb_dscal_fn)dscal_ref;
    vt.cscal   = (fb_cscal_fn)cscal_ref;
    vt.zscal   = (fb_zscal_fn)zscal_ref;
    vt.csscal  = (fb_csscal_fn)csscal_ref;
    vt.zdscal  = (fb_zdscal_fn)zdscal_ref;

    vt.saxpy   = (fb_saxpy_fn)saxpy_ref;
    vt.daxpy   = (fb_daxpy_fn)daxpy_ref;
    vt.caxpy   = (fb_caxpy_fn)caxpy_ref;
    vt.zaxpy   = (fb_zaxpy_fn)zaxpy_ref;

    vt.sdot    = (fb_sdot_fn)sdot_ref;
    vt.ddot    = (fb_ddot_fn)ddot_ref;
    /* complex dot: result-pointer wrappers needed */
    vt.cdotu   = (fb_cdotu_fn)w_cdotu;
    vt.zdotu   = (fb_zdotu_fn)w_zdotu;
    vt.cdotc   = (fb_cdotc_fn)w_cdotc;
    vt.zdotc   = (fb_zdotc_fn)w_zdotc;

    vt.snrm2   = (fb_snrm2_fn)snrm2_ref;
    vt.dnrm2   = (fb_dnrm2_fn)dnrm2_ref;
    vt.scnrm2  = (fb_scnrm2_fn)scnrm2_ref;
    vt.dznrm2  = (fb_dznrm2_fn)cblas_dznrm2; /* dznrm2_ref is static inline; use exported CBLAS wrapper */

    vt.sasum   = (fb_sasum_fn)sasum_ref;
    vt.dasum   = (fb_dasum_fn)dasum_ref;
    vt.scasum  = (fb_scasum_fn)scasum_ref;
    vt.dzasum  = (fb_dzasum_fn)dzasum_ref;

    vt.isamax  = (fb_isamax_fn)isamax_ref;
    vt.idamax  = (fb_idamax_fn)idamax_ref;
    vt.icamax  = (fb_icamax_fn)icamax_ref;
    vt.izamax  = (fb_izamax_fn)izamax_ref;

    vt.srotg   = (fb_srotg_fn)srotg_ref;
    vt.drotg   = (fb_drotg_fn)drotg_ref;
    vt.crotg   = (fb_crotg_fn)crotg_ref;
    vt.zrotg   = (fb_zrotg_fn)zrotg_ref;

    vt.srot    = (fb_srot_fn)srot_ref;
    vt.drot    = (fb_drot_fn)drot_ref;
    vt.crot    = (fb_crot_fn)cblas_crot;  /* crot_ref is static inline; use exported CBLAS wrapper */
    vt.zrot    = (fb_zrot_fn)cblas_zrot;  /* zrot_ref is static inline; use exported CBLAS wrapper */

    vt.srotmg  = (fb_srotmg_fn)srotmg_ref;
    vt.drotmg  = (fb_drotmg_fn)drotmg_ref;
    vt.srotm   = (fb_srotm_fn)srotm_ref;
    vt.drotm   = (fb_drotm_fn)drotm_ref;

    /* ---- Level 2: enum-conversion wrappers ---- */
    vt.sgemv   = (fb_sgemv_fn)w_sgemv;
    vt.dgemv   = (fb_dgemv_fn)w_dgemv;
    vt.cgemv   = (fb_cgemv_fn)w_cgemv;
    vt.zgemv   = (fb_zgemv_fn)w_zgemv;

    vt.sgbmv   = (fb_sgbmv_fn)w_sgbmv;
    vt.dgbmv   = (fb_dgbmv_fn)w_dgbmv;
    vt.cgbmv   = (fb_cgbmv_fn)w_cgbmv;
    vt.zgbmv   = (fb_zgbmv_fn)w_zgbmv;

    vt.ssymv   = (fb_ssymv_fn)w_ssymv;
    vt.dsymv   = (fb_dsymv_fn)w_dsymv;
    vt.csymv   = (fb_csymv_fn)w_csymv;
    vt.zsymv   = (fb_zsymv_fn)w_zsymv;
    vt.chemv   = (fb_chemv_fn)w_chemv;
    vt.zhemv   = (fb_zhemv_fn)w_zhemv;

    vt.ssbmv   = (fb_ssbmv_fn)w_ssbmv;
    vt.dsbmv   = (fb_dsbmv_fn)w_dsbmv;
    vt.chbmv   = (fb_chbmv_fn)w_chbmv;
    vt.zhbmv   = (fb_zhbmv_fn)w_zhbmv;

    vt.sspmv   = (fb_sspmv_fn)w_sspmv;
    vt.dspmv   = (fb_dspmv_fn)w_dspmv;
    vt.chpmv   = (fb_chpmv_fn)w_chpmv;
    vt.zhpmv   = (fb_zhpmv_fn)w_zhpmv;

    vt.strmv   = (fb_strmv_fn)w_strmv;
    vt.dtrmv   = (fb_dtrmv_fn)w_dtrmv;
    vt.ctrmv   = (fb_ctrmv_fn)w_ctrmv;
    vt.ztrmv   = (fb_ztrmv_fn)w_ztrmv;

    vt.stbmv   = (fb_stbmv_fn)w_stbmv;
    vt.dtbmv   = (fb_dtbmv_fn)w_dtbmv;
    vt.ctbmv   = (fb_ctbmv_fn)w_ctbmv;
    vt.ztbmv   = (fb_ztbmv_fn)w_ztbmv;

    vt.stpmv   = (fb_stpmv_fn)w_stpmv;
    vt.dtpmv   = (fb_dtpmv_fn)w_dtpmv;
    vt.ctpmv   = (fb_ctpmv_fn)w_ctpmv;
    vt.ztpmv   = (fb_ztpmv_fn)w_ztpmv;

    vt.strsv   = (fb_strsv_fn)w_strsv;
    vt.dtrsv   = (fb_dtrsv_fn)w_dtrsv;
    vt.ctrsv   = (fb_ctrsv_fn)w_ctrsv;
    vt.ztrsv   = (fb_ztrsv_fn)w_ztrsv;

    vt.stbsv   = (fb_stbsv_fn)w_stbsv;
    vt.dtbsv   = (fb_dtbsv_fn)w_dtbsv;
    vt.ctbsv   = (fb_ctbsv_fn)w_ctbsv;
    vt.ztbsv   = (fb_ztbsv_fn)w_ztbsv;

    vt.stpsv   = (fb_stpsv_fn)w_stpsv;
    vt.dtpsv   = (fb_dtpsv_fn)w_dtpsv;
    vt.ctpsv   = (fb_ctpsv_fn)w_ctpsv;
    vt.ztpsv   = (fb_ztpsv_fn)w_ztpsv;

    vt.sger    = (fb_sger_fn)w_sger;
    vt.dger    = (fb_dger_fn)w_dger;
    vt.cgeru   = (fb_cgeru_fn)w_cgeru;
    vt.zgeru   = (fb_zgeru_fn)w_zgeru;
    vt.cgerc   = (fb_cgerc_fn)w_cgerc;
    vt.zgerc   = (fb_zgerc_fn)w_zgerc;

    vt.ssyr    = (fb_ssyr_fn)w_ssyr;
    vt.dsyr    = (fb_dsyr_fn)w_dsyr;
    vt.csyr    = (fb_csyr_fn)w_csyr;
    vt.zsyr    = (fb_zsyr_fn)w_zsyr;
    vt.cher    = (fb_cher_fn)w_cher;
    vt.zher    = (fb_zher_fn)w_zher;

    vt.ssyr2   = (fb_ssyr2_fn)w_ssyr2;
    vt.dsyr2   = (fb_dsyr2_fn)w_dsyr2;
    vt.cher2   = (fb_cher2_fn)w_cher2;
    vt.zher2   = (fb_zher2_fn)w_zher2;

    vt.sspr    = (fb_sspr_fn)w_sspr;
    vt.dspr    = (fb_dspr_fn)w_dspr;
    vt.chpr    = (fb_chpr_fn)w_chpr;
    vt.zhpr    = (fb_zhpr_fn)w_zhpr;

    vt.sspr2   = (fb_sspr2_fn)w_sspr2;
    vt.dspr2   = (fb_dspr2_fn)w_dspr2;
    vt.chpr2   = (fb_chpr2_fn)w_chpr2;
    vt.zhpr2   = (fb_zhpr2_fn)w_zhpr2;

    /* ---- Level 3: enum-conversion wrappers ---- */
    vt.sgemm   = (fb_sgemm_fn)w_sgemm;
    vt.dgemm   = (fb_dgemm_fn)w_dgemm;
    vt.cgemm   = (fb_cgemm_fn)w_cgemm;
    vt.zgemm   = (fb_zgemm_fn)w_zgemm;

    vt.ssymm   = (fb_ssymm_fn)w_ssymm;
    vt.dsymm   = (fb_dsymm_fn)w_dsymm;
    vt.csymm   = (fb_csymm_fn)w_csymm;
    vt.zsymm   = (fb_zsymm_fn)w_zsymm;
    vt.chemm   = (fb_chemm_fn)w_chemm;
    vt.zhemm   = (fb_zhemm_fn)w_zhemm;

    vt.ssyrk   = (fb_ssyrk_fn)w_ssyrk;
    vt.dsyrk   = (fb_dsyrk_fn)w_dsyrk;
    vt.csyrk   = (fb_csyrk_fn)w_csyrk;
    vt.zsyrk   = (fb_zsyrk_fn)w_zsyrk;
    vt.cherk   = (fb_cherk_fn)w_cherk;
    vt.zherk   = (fb_zherk_fn)w_zherk;

    vt.ssyr2k  = (fb_ssyr2k_fn)w_ssyr2k;
    vt.dsyr2k  = (fb_dsyr2k_fn)w_dsyr2k;
    vt.csyr2k  = (fb_csyr2k_fn)w_csyr2k;
    vt.zsyr2k  = (fb_zsyr2k_fn)w_zsyr2k;
    vt.cher2k  = (fb_cher2k_fn)w_cher2k;
    vt.zher2k  = (fb_zher2k_fn)w_zher2k;

    vt.strmm   = (fb_strmm_fn)w_strmm;
    vt.dtrmm   = (fb_dtrmm_fn)w_dtrmm;
    vt.ctrmm   = (fb_ctrmm_fn)w_ctrmm;
    vt.ztrmm   = (fb_ztrmm_fn)w_ztrmm;

    vt.strsm   = (fb_strsm_fn)w_strsm;
    vt.dtrsm   = (fb_dtrsm_fn)w_dtrsm;
    vt.ctrsm   = (fb_ctrsm_fn)w_ctrsm;
    vt.ztrsm   = (fb_ztrsm_fn)w_ztrsm;

    /* ---- LAPACK (from in-tree reference_lapack.c) ---- */
    vt.sgetrf  = (fb_sgetrf_fn)fb_ref_sgetrf;
    vt.dgetrf  = (fb_dgetrf_fn)fb_ref_dgetrf;
    vt.cgetrf  = (fb_cgetrf_fn)fb_ref_cgetrf;
    vt.zgetrf  = (fb_zgetrf_fn)fb_ref_zgetrf;

    vt.sgetrs  = (fb_sgetrs_fn)fb_ref_sgetrs;
    vt.dgetrs  = (fb_dgetrs_fn)fb_ref_dgetrs;
    vt.cgetrs  = (fb_cgetrs_fn)fb_ref_cgetrs;
    vt.zgetrs  = (fb_zgetrs_fn)fb_ref_zgetrs;

    vt.spotrf  = (fb_spotrf_fn)fb_ref_spotrf;
    vt.dpotrf  = (fb_dpotrf_fn)fb_ref_dpotrf;
    vt.cpotrf  = (fb_cpotrf_fn)fb_ref_cpotrf;
    vt.zpotrf  = (fb_zpotrf_fn)fb_ref_zpotrf;

    vt.spotrs  = (fb_spotrs_fn)fb_ref_spotrs;
    vt.dpotrs  = (fb_dpotrs_fn)fb_ref_dpotrs;
    vt.cpotrs  = (fb_cpotrs_fn)fb_ref_cpotrs;
    vt.zpotrs  = (fb_zpotrs_fn)fb_ref_zpotrs;

    vt.sgeqrf  = (fb_sgeqrf_fn)fb_ref_sgeqrf;
    vt.dgeqrf  = (fb_dgeqrf_fn)fb_ref_dgeqrf;
    vt.cgeqrf  = (fb_cgeqrf_fn)fb_ref_cgeqrf;
    vt.zgeqrf  = (fb_zgeqrf_fn)fb_ref_zgeqrf;

    vt.sorgqr  = (fb_sorgqr_fn)fb_ref_sorgqr;
    vt.dorgqr  = (fb_dorgqr_fn)fb_ref_dorgqr;
    vt.cungqr  = (fb_cungqr_fn)fb_ref_cungqr;
    vt.zungqr  = (fb_zungqr_fn)fb_ref_zungqr;

    vt.sgels   = (fb_sgels_fn)fb_ref_sgels;
    vt.dgels   = (fb_dgels_fn)fb_ref_dgels;
    vt.cgels   = (fb_cgels_fn)fb_ref_cgels;
    vt.zgels   = (fb_zgels_fn)fb_ref_zgels;

    vt.sormqr  = (fb_sormqr_fn)fb_ref_sormqr;
    vt.dormqr  = (fb_dormqr_fn)fb_ref_dormqr;
    vt.cunmqr  = (fb_cunmqr_fn)fb_ref_cunmqr;
    vt.zunmqr  = (fb_zunmqr_fn)fb_ref_zunmqr;

    vt.sgelsd  = (fb_sgelsd_fn)fb_ref_sgelsd;
    vt.dgelsd  = (fb_dgelsd_fn)fb_ref_dgelsd;
    vt.cgelsd  = (fb_cgelsd_fn)fb_ref_cgelsd;
    vt.zgelsd  = (fb_zgelsd_fn)fb_ref_zgelsd;

    vt.ssyev   = (fb_ssyev_fn)fb_ref_ssyev;
    vt.dsyev   = (fb_dsyev_fn)fb_ref_dsyev;
    vt.cheev   = (fb_cheev_fn)fb_ref_cheev;
    vt.zheev   = (fb_zheev_fn)fb_ref_zheev;

    vt.sgesvd  = (fb_sgesvd_fn)fb_ref_sgesvd;
    vt.dgesvd  = (fb_dgesvd_fn)fb_ref_dgesvd;
    vt.cgesvd  = (fb_cgesvd_fn)fb_ref_cgesvd;
    vt.zgesvd  = (fb_zgesvd_fn)fb_ref_zgesvd;

    vt.strtri  = (fb_strtri_fn)fb_ref_strtri;
    vt.dtrtri  = (fb_dtrtri_fn)fb_ref_dtrtri;
    vt.ctrtri  = (fb_ctrtri_fn)fb_ref_ctrtri;
    vt.ztrtri  = (fb_ztrtri_fn)fb_ref_ztrtri;

    vt.sgesdd  = (fb_sgesdd_fn)fb_ref_sgesdd;
    vt.dgesdd  = (fb_dgesdd_fn)fb_ref_dgesdd;
    vt.cgesdd  = (fb_cgesdd_fn)fb_ref_cgesdd;
    vt.zgesdd  = (fb_zgesdd_fn)fb_ref_zgesdd;

    vt.ssygv   = (fb_ssygv_fn)fb_ref_ssygv;
    vt.dsygv   = (fb_dsygv_fn)fb_ref_dsygv;
    vt.chegv   = (fb_chegv_fn)fb_ref_chegv;
    vt.zhegv   = (fb_zhegv_fn)fb_ref_zhegv;

    vt.sgelsy  = (fb_sgelsy_fn)fb_ref_sgelsy;
    vt.dgelsy  = (fb_dgelsy_fn)fb_ref_dgelsy;
    vt.cgelsy  = (fb_cgelsy_fn)fb_ref_cgelsy;
    vt.zgelsy  = (fb_zgelsy_fn)fb_ref_zgelsy;

    /* ---- GPU memory management (not needed for CPU reference) ---- */
    vt.mem_alloc      = NULL;
    vt.mem_free       = NULL;
    vt.mem_upload     = NULL;
    vt.mem_download   = NULL;
    vt.mem_copy       = NULL;
    vt.stream_create  = NULL;
    vt.stream_destroy = NULL;
    vt.stream_sync    = NULL;
    vt.stream_set     = NULL;
    vt.get_capabilities  = reference_get_capabilities;
    vt.get_num_threads   = reference_get_num_threads;
    vt.set_num_threads   = reference_set_num_threads;

    ready = 1;
    return &vt;
}

/* Convenience function for backend_instance.c */
const fb_backend_vtable_t *fb_reference_get_vtable(void) {
    return fb_reference_backend();
}


