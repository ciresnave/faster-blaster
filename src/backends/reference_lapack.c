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

/* =========================================================================
 * SYEV / HEEV / GESVD / GEEV / GESDD / SYGV wrappers
 *
 * All auxiliary *_ref routines (sgehrd_ref, shseqr_ref, strevc_ref,
 * sbdsqr_ref, chetrd_ref, steqr_ref, ...) are present in
 * faster-blaster-reference as of Phase 3.  Wrappers here bridge the
 * vtable signatures (LAPACKE-style, no work array) to the *_ref signatures.
 *
 * ABI notes from file header:
 *  - ssyev/dsyev/cheev/zheev _ref take extra z/ldz; pass A,lda (in-place)
 *  - gesvd_ref has no superb param; vtable's superb output is ignored
 *  - gesdd_ref / ssygv_ref require work arrays; allocated here dynamically
 * ========================================================================= */

/* --- Forward declarations ------------------------------------------------ */

/* SYEV / HEEV */
int ssyev_ref(const char *jobz, const char *uplo, int n, float *a, int lda,
              float *w, float *z, int ldz);
int dsyev_ref(const char *jobz, const char *uplo, int n, double *a, int lda,
              double *w, double *z, int ldz);
int cheev_ref(const char *jobz, const char *uplo, int n, float _Complex *a,
              int lda, float *w, float _Complex *z, int ldz);
int zheev_ref(const char *jobz, const char *uplo, int n, double _Complex *a,
              int lda, double *w, double _Complex *z, int ldz);

/* GESVD (no superb in _ref; vtable's superb is ignored) */
int sgesvd_ref(const char *jobu, const char *jobvt, int m, int n, float *a,
               int lda, float *s, float *u, int ldu, float *vt, int ldvt);
int dgesvd_ref(const char *jobu, const char *jobvt, int m, int n, double *a,
               int lda, double *s, double *u, int ldu, double *vt, int ldvt);
int cgesvd_ref(const char *jobu, const char *jobvt, int m, int n,
               float _Complex *a, int lda, float *s, float _Complex *u, int ldu,
               float _Complex *vh, int ldvh);
int zgesvd_ref(const char *jobu, const char *jobvt, int m, int n,
               double _Complex *a, int lda, double *s, double _Complex *u,
               int ldu, double _Complex *vh, int ldvh);

/* GEEV real (wr + wi outputs) */
int sgeev_ref(const char *jobvl, const char *jobvr, int n, float *a, int lda,
              float *wr, float *wi, float *vl, int ldvl, float *vr, int ldvr);
int dgeev_ref(const char *jobvl, const char *jobvr, int n, double *a, int lda,
              double *wr, double *wi, double *vl, int ldvl, double *vr,
              int ldvr);

/* GEEV complex (single complex w output) */
int cgeev_ref(const char *jobvl, const char *jobvr, int n, float _Complex *a,
              int lda, float _Complex *w, float _Complex *vl, int ldvl,
              float _Complex *vr, int ldvr);
int zgeev_ref(const char *jobvl, const char *jobvr, int n, double _Complex *a,
              int lda, double _Complex *w, double _Complex *vl, int ldvl,
              double _Complex *vr, int ldvr);

/* GESDD (divide-and-conquer SVD; needs work/iwork allocation) */
int sgesdd_ref(char jobz, int m, int n, float *a, int lda, float *s, float *u,
               int ldu, float *vt, int ldvt, float *work, int lwork, int *iwork,
               int *info);
int dgesdd_ref(char jobz, int m, int n, double *a, int lda, double *s,
               double *u, int ldu, double *vt, int ldvt, double *work,
               int lwork, int *iwork, int *info);

/* SYGV / DSYGV (generalized symmetric eigenproblem; needs work allocation) */
int ssygv_ref(int itype, char jobz, char uplo, int n, float *a, int lda,
              float *b, int ldb, float *w, float *work, int lwork, int *info);
int dsygv_ref(int itype, char jobz, char uplo, int n, double *a, int lda,
              double *b, int ldb, double *w, double *work, int lwork,
              int *info);

/* --- Wrapper bodies ------------------------------------------------------ */

/* SYEV: pass A/lda as eigenvector output z/ldz (LAPACK in-place convention) */
static int ref_ssyev(fb_layout_t l, char jobz, fb_uplo_t uplo, int n, float *A,
                     int lda, float *w) {
  (void)l;
  char uo = FU(uplo);
  return ssyev_ref(&jobz, &uo, n, A, lda, w, A, lda);
}
static int ref_dsyev(fb_layout_t l, char jobz, fb_uplo_t uplo, int n, double *A,
                     int lda, double *w) {
  (void)l;
  char uo = FU(uplo);
  return dsyev_ref(&jobz, &uo, n, A, lda, w, A, lda);
}
static int ref_cheev(fb_layout_t l, char jobz, fb_uplo_t uplo, int n,
                     float _Complex *A, int lda, float *w) {
  (void)l;
  char uo = FU(uplo);
  return cheev_ref(&jobz, &uo, n, A, lda, w, A, lda);
}
static int ref_zheev(fb_layout_t l, char jobz, fb_uplo_t uplo, int n,
                     double _Complex *A, int lda, double *w) {
  (void)l;
  char uo = FU(uplo);
  return zheev_ref(&jobz, &uo, n, A, lda, w, A, lda);
}

/* GESVD: superb is trailing vtable output -- not computed by _ref, ignored */
static int ref_sgesvd(fb_layout_t l, char jobu, char jobvt, int m, int n,
                      float *A, int lda, float *s, float *U, int ldu, float *VT,
                      int ldvt, float *superb) {
  (void)l;
  (void)superb;
  return sgesvd_ref(&jobu, &jobvt, m, n, A, lda, s, U, ldu, VT, ldvt);
}
static int ref_dgesvd(fb_layout_t l, char jobu, char jobvt, int m, int n,
                      double *A, int lda, double *s, double *U, int ldu,
                      double *VT, int ldvt, double *superb) {
  (void)l;
  (void)superb;
  return dgesvd_ref(&jobu, &jobvt, m, n, A, lda, s, U, ldu, VT, ldvt);
}
static int ref_cgesvd(fb_layout_t l, char jobu, char jobvt, int m, int n,
                      float _Complex *A, int lda, float *s, float _Complex *U,
                      int ldu, float _Complex *VT, int ldvt, float *superb) {
  (void)l;
  (void)superb;
  return cgesvd_ref(&jobu, &jobvt, m, n, A, lda, s, U, ldu, VT, ldvt);
}
static int ref_zgesvd(fb_layout_t l, char jobu, char jobvt, int m, int n,
                      double _Complex *A, int lda, double *s,
                      double _Complex *U, int ldu, double _Complex *VT,
                      int ldvt, double *superb) {
  (void)l;
  (void)superb;
  return zgesvd_ref(&jobu, &jobvt, m, n, A, lda, s, U, ldu, VT, ldvt);
}

/* GEEV real */
static int ref_sgeev(fb_layout_t l, char jobvl, char jobvr, int n, float *A,
                     int lda, float *wr, float *wi, float *VL, int ldvl,
                     float *VR, int ldvr) {
  (void)l;
  return sgeev_ref(&jobvl, &jobvr, n, A, lda, wr, wi, VL, ldvl, VR, ldvr);
}
static int ref_dgeev(fb_layout_t l, char jobvl, char jobvr, int n, double *A,
                     int lda, double *wr, double *wi, double *VL, int ldvl,
                     double *VR, int ldvr) {
  (void)l;
  return dgeev_ref(&jobvl, &jobvr, n, A, lda, wr, wi, VL, ldvl, VR, ldvr);
}

/* GEEV complex */
static int ref_cgeev(fb_layout_t l, char jobvl, char jobvr, int n,
                     float _Complex *A, int lda, float _Complex *w,
                     float _Complex *VL, int ldvl, float _Complex *VR,
                     int ldvr) {
  (void)l;
  return cgeev_ref(&jobvl, &jobvr, n, A, lda, w, VL, ldvl, VR, ldvr);
}
static int ref_zgeev(fb_layout_t l, char jobvl, char jobvr, int n,
                     double _Complex *A, int lda, double _Complex *w,
                     double _Complex *VL, int ldvl, double _Complex *VR,
                     int ldvr) {
  (void)l;
  return zgeev_ref(&jobvl, &jobvr, n, A, lda, w, VL, ldvl, VR, ldvr);
}

/* GESDD: allocates work + iwork dynamically */
static int ref_sgesdd(fb_layout_t l, char jobz, int m, int n, float *A, int lda,
                      float *s, float *U, int ldu, float *VT, int ldvt) {
  (void)l;
  int minmn = (m < n) ? m : n;
  int lwork = 3 * minmn * minmn + ((m > n) ? (7 * minmn) : (5 * minmn + m));
  if (lwork < 1)
    lwork = 1;
  float *work = (float *)malloc((size_t)lwork * sizeof(float));
  int *iwork =
      (int *)malloc((size_t)(8 * (minmn > 1 ? minmn : 1)) * sizeof(int));
  if (!work || !iwork) {
    free(work);
    free(iwork);
    return -12;
  }
  int info = 0;
  sgesdd_ref(jobz, m, n, A, lda, s, U, ldu, VT, ldvt, work, lwork, iwork,
             &info);
  free(work);
  free(iwork);
  return info;
}
static int ref_dgesdd(fb_layout_t l, char jobz, int m, int n, double *A,
                      int lda, double *s, double *U, int ldu, double *VT,
                      int ldvt) {
  (void)l;
  int minmn = (m < n) ? m : n;
  int lwork = 3 * minmn * minmn + ((m > n) ? (7 * minmn) : (5 * minmn + m));
  if (lwork < 1)
    lwork = 1;
  double *work = (double *)malloc((size_t)lwork * sizeof(double));
  int *iwork =
      (int *)malloc((size_t)(8 * (minmn > 1 ? minmn : 1)) * sizeof(int));
  if (!work || !iwork) {
    free(work);
    free(iwork);
    return -12;
  }
  int info = 0;
  dgesdd_ref(jobz, m, n, A, lda, s, U, ldu, VT, ldvt, work, lwork, iwork,
             &info);
  free(work);
  free(iwork);
  return info;
}

/* SYGV / DSYGV: allocates work dynamically */
static int ref_ssygv(fb_layout_t l, int itype, char jobz, fb_uplo_t uplo, int n,
                     float *A, int lda, float *B, int ldb, float *w) {
  (void)l;
  char uo = FU(uplo);
  int lwork = 3 * n + 64;
  float *work = (float *)malloc((size_t)lwork * sizeof(float));
  if (!work)
    return -11;
  int info = 0;
  ssygv_ref(itype, jobz, uo, n, A, lda, B, ldb, w, work, lwork, &info);
  free(work);
  return info;
}
static int ref_dsygv(fb_layout_t l, int itype, char jobz, fb_uplo_t uplo, int n,
                     double *A, int lda, double *B, int ldb, double *w) {
  (void)l;
  char uo = FU(uplo);
  int lwork = 3 * n + 64;
  double *work = (double *)malloc((size_t)lwork * sizeof(double));
  if (!work)
    return -11;
  int info = 0;
  dsygv_ref(itype, jobz, uo, n, A, lda, B, ldb, w, work, lwork, &info);
  free(work);
  return info;
}
