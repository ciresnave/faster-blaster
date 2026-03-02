/**
 * @file reference_lapack.c
 * @brief LAPACK vtable wrappers delegating to faster-blaster-reference *_ref
 *        implementations.
 *
 * This file is #include'd directly into reference.c so it can use the
 * helpers already defined there (trans_char, uplo_char, FC, FU, …) and
 * share the same translation unit, avoiding duplicate symbol issues.
 *
 * ABI notes:
 *  - fb_complex_float_t  == float _Complex  (same ABI, typedef in fb_types.h)
 *  - fb_complex_double_t == double _Complex (same ABI)
 *  - trans_char() / uplo_char() are defined in reference.c before this include
 *
 * Signature mismatches handled here:
 *  1. spotrs/dpotrs/cpotrs/zpotrs  — *_ref puts nrhs LAST; vtable has it 4th
 *  2. ssyev/dsyev/cheev/zheev      — *_ref uses 'E' (not 'N') for eigs-only;
 *                                     *_ref takes extra z/ldz (pass A,lda)
 *  3. ?gesvd_ref                   — no superb param; vtable has it; ignored
 *  4. ssyev/cheev jobz + uplo      — *_ref takes const char* (pointer);
 *                                     vtable passes char by value
 *
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

/* ---- Forward declarations — faster-blaster-reference *_ref functions ---
 * These are compiled into libfaster_blaster_reference.a.  No headers exist
 * yet for the LAPACK layer, so we declare them manually.
 * ========================================================================= */

/* GETRF */
int sgetrf_ref(int m, int n, float *a, int lda, int *ipiv);
int dgetrf_ref(int m, int n, double *a, int lda, int *ipiv);
int cgetrf_ref(int m, int n, float _Complex *a, int lda, int *ipiv);
int zgetrf_ref(int m, int n, double _Complex *a, int lda, int *ipiv);

/* GETRS */
int sgetrs_ref(char trans, int n, int nrhs,
               const float *a, int lda, const int *ipiv,
               float *b, int ldb);
int dgetrs_ref(char trans, int n, int nrhs,
               const double *a, int lda, const int *ipiv,
               double *b, int ldb);
int cgetrs_ref(char trans, int n, int nrhs,
               const float _Complex *a, int lda, const int *ipiv,
               float _Complex *b, int ldb);
int zgetrs_ref(char trans, int n, int nrhs,
               const double _Complex *a, int lda, const int *ipiv,
               double _Complex *b, int ldb);

/* POTRF */
int spotrf_ref(char uplo, int n, float *a, int lda);
int dpotrf_ref(char uplo, int n, double *a, int lda);
int cpotrf_ref(char uplo, int n, float _Complex *a, int lda);
int zpotrf_ref(char uplo, int n, double _Complex *a, int lda);

/* POTRS — NOTE: nrhs comes LAST in *_ref */
int spotrs_ref(char uplo, int n,
               const float *a, int lda, float *b, int ldb, int nrhs);
int dpotrs_ref(char uplo, int n,
               const double *a, int lda, double *b, int ldb, int nrhs);
int cpotrs_ref(char uplo, int n,
               const float _Complex *a, int lda,
               float _Complex *b, int ldb, int nrhs);
int zpotrs_ref(char uplo, int n,
               const double _Complex *a, int lda,
               double _Complex *b, int ldb, int nrhs);

/* GEQRF */
int sgeqrf_ref(int m, int n, float *a, int lda, float *tau);
int dgeqrf_ref(int m, int n, double *a, int lda, double *tau);
int cgeqrf_ref(int m, int n, float _Complex *a, int lda, float _Complex *tau);
int zgeqrf_ref(int m, int n, double _Complex *a, int lda, double _Complex *tau);

/* ORGQR (real) / UNGQR (complex) */
int sorgqr_ref(int m, int n, int k, float *a, int lda, const float *tau);
int dorgqr_ref(int m, int n, int k, double *a, int lda, const double *tau);
int cungqr_ref(int m, int n, int k,
               float _Complex *a, int lda, const float _Complex *tau);
int zungqr_ref(int m, int n, int k,
               double _Complex *a, int lda, const double _Complex *tau);

/* =========================================================================
 * SYEV / HEEV / GESVD / GEEV — NOT WIRED
 *
 * faster-blaster-reference implements ssyev_ref, cheev_ref, sgesvd_ref,
 * cgesvd_ref, sgeev_ref, cgeev_ref etc. but they depend on LAPACK
 * auxiliary routines (sgehrd_ref, shseqr_ref, strevc_ref, sgebrd_ref,
 * sbdsqr_ref, chetrd_ref, steqr_ref, …) that are not yet implemented in
 * faster-blaster-reference.  Linking them causes undefined-symbol errors.
 *
 * Remove these forward declarations and wrapper bodies until the auxiliary
 * routines are available.  The vtable entries remain NULL in reference.c.
 * ========================================================================= */

/* (syev_jobz helper removed — SYEV/HEEV/GESVD/GEEV are not wired; see comment above) */

/* =========================================================================
 * GETRF wrappers
 * ========================================================================= */
static int ref_sgetrf(fb_layout_t l, int m, int n, float *A, int lda, int *ipiv)
    { (void)l; return sgetrf_ref(m, n, A, lda, ipiv); }
static int ref_dgetrf(fb_layout_t l, int m, int n, double *A, int lda, int *ipiv)
    { (void)l; return dgetrf_ref(m, n, A, lda, ipiv); }
static int ref_cgetrf(fb_layout_t l, int m, int n,
                      float _Complex *A, int lda, int *ipiv)
    { (void)l; return cgetrf_ref(m, n, A, lda, ipiv); }
static int ref_zgetrf(fb_layout_t l, int m, int n,
                      double _Complex *A, int lda, int *ipiv)
    { (void)l; return zgetrf_ref(m, n, A, lda, ipiv); }

/* =========================================================================
 * GETRS wrappers
 * ========================================================================= */
static int ref_sgetrs(fb_layout_t l, fb_transpose_t tr, int n, int nrhs,
                      const float *A, int lda, const int *ipiv, float *B, int ldb)
    { (void)l; return sgetrs_ref(FC(tr), n, nrhs, A, lda, ipiv, B, ldb); }
static int ref_dgetrs(fb_layout_t l, fb_transpose_t tr, int n, int nrhs,
                      const double *A, int lda, const int *ipiv, double *B, int ldb)
    { (void)l; return dgetrs_ref(FC(tr), n, nrhs, A, lda, ipiv, B, ldb); }
static int ref_cgetrs(fb_layout_t l, fb_transpose_t tr, int n, int nrhs,
                      const float _Complex *A, int lda, const int *ipiv,
                      float _Complex *B, int ldb)
    { (void)l; return cgetrs_ref(FC(tr), n, nrhs, A, lda, ipiv, B, ldb); }
static int ref_zgetrs(fb_layout_t l, fb_transpose_t tr, int n, int nrhs,
                      const double _Complex *A, int lda, const int *ipiv,
                      double _Complex *B, int ldb)
    { (void)l; return zgetrs_ref(FC(tr), n, nrhs, A, lda, ipiv, B, ldb); }

/* =========================================================================
 * POTRF wrappers
 * ========================================================================= */
static int ref_spotrf(fb_layout_t l, fb_uplo_t uplo, int n, float *A, int lda)
    { (void)l; return spotrf_ref(FU(uplo), n, A, lda); }
static int ref_dpotrf(fb_layout_t l, fb_uplo_t uplo, int n, double *A, int lda)
    { (void)l; return dpotrf_ref(FU(uplo), n, A, lda); }
static int ref_cpotrf(fb_layout_t l, fb_uplo_t uplo, int n,
                      float _Complex *A, int lda)
    { (void)l; return cpotrf_ref(FU(uplo), n, A, lda); }
static int ref_zpotrf(fb_layout_t l, fb_uplo_t uplo, int n,
                      double _Complex *A, int lda)
    { (void)l; return zpotrf_ref(FU(uplo), n, A, lda); }

/* =========================================================================
 * POTRS wrappers — nrhs is LAST in *_ref, 4th in vtable
 * ========================================================================= */
static int ref_spotrs(fb_layout_t l, fb_uplo_t uplo, int n, int nrhs,
                      const float *A, int lda, float *B, int ldb)
    { (void)l; return spotrs_ref(FU(uplo), n, A, lda, B, ldb, nrhs); }
static int ref_dpotrs(fb_layout_t l, fb_uplo_t uplo, int n, int nrhs,
                      const double *A, int lda, double *B, int ldb)
    { (void)l; return dpotrs_ref(FU(uplo), n, A, lda, B, ldb, nrhs); }
static int ref_cpotrs(fb_layout_t l, fb_uplo_t uplo, int n, int nrhs,
                      const float _Complex *A, int lda, float _Complex *B, int ldb)
    { (void)l; return cpotrs_ref(FU(uplo), n, A, lda, B, ldb, nrhs); }
static int ref_zpotrs(fb_layout_t l, fb_uplo_t uplo, int n, int nrhs,
                      const double _Complex *A, int lda, double _Complex *B, int ldb)
    { (void)l; return zpotrs_ref(FU(uplo), n, A, lda, B, ldb, nrhs); }

/* =========================================================================
 * GEQRF wrappers
 * ========================================================================= */
static int ref_sgeqrf(fb_layout_t l, int m, int n, float *A, int lda, float *tau)
    { (void)l; return sgeqrf_ref(m, n, A, lda, tau); }
static int ref_dgeqrf(fb_layout_t l, int m, int n, double *A, int lda, double *tau)
    { (void)l; return dgeqrf_ref(m, n, A, lda, tau); }
static int ref_cgeqrf(fb_layout_t l, int m, int n,
                      float _Complex *A, int lda, float _Complex *tau)
    { (void)l; return cgeqrf_ref(m, n, A, lda, tau); }
static int ref_zgeqrf(fb_layout_t l, int m, int n,
                      double _Complex *A, int lda, double _Complex *tau)
    { (void)l; return zgeqrf_ref(m, n, A, lda, tau); }

/* =========================================================================
 * ORGQR / UNGQR wrappers
 * ========================================================================= */
static int ref_sorgqr(fb_layout_t l, int m, int n, int k,
                      float *A, int lda, const float *tau)
    { (void)l; return sorgqr_ref(m, n, k, A, lda, tau); }
static int ref_dorgqr(fb_layout_t l, int m, int n, int k,
                      double *A, int lda, const double *tau)
    { (void)l; return dorgqr_ref(m, n, k, A, lda, tau); }
static int ref_cungqr(fb_layout_t l, int m, int n, int k,
                      float _Complex *A, int lda, const float _Complex *tau)
    { (void)l; return cungqr_ref(m, n, k, A, lda, tau); }
static int ref_zungqr(fb_layout_t l, int m, int n, int k,
                      double _Complex *A, int lda, const double _Complex *tau)
    { (void)l; return zungqr_ref(m, n, k, A, lda, tau); }

/* SSYEV / DSYEV / CHEEV / ZHEEV / SGESVD / DGESVD / CGESVD / ZGESVD /
 * SGEEV / DGEEV / CGEEV / ZGEEV: see comment block above — NOT WIRED
 * because their *_ref implementations call unresolved auxiliary _ref
 * functions (sgehrd_ref, shseqr_ref, sbdsqr_ref, steqr_ref, …).
 * vtable entries remain NULL in reference.c.
 */