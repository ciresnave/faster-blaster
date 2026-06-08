#include <stdio.h>
#include <string.h>

#include "../include/faster-blaster/vtable_autofill.h"
#include "../src/backends/backend_interface.h"
#include "../src/judge/judge_op_ids.h"

typedef int (*fb_chbev_cblas_fn)(fb_layout_t layout, char jobz, fb_uplo_t uplo,
                                 int n, int kd, fb_complex_float_t *ab,
                                 int ldab, float *w, fb_complex_float_t *z,
                                 int ldz);

typedef int (*fb_ssyev_cblas_fn)(fb_layout_t layout, char jobz, fb_uplo_t uplo,
                                 int n, float *a, int lda, float *w);

typedef int (*fb_ssygv_cblas_fn)(fb_layout_t layout, int itype, char jobz,
                                 fb_uplo_t uplo, int n, float *a, int lda,
                                 float *b, int ldb, float *w);

typedef int (*fb_ssyevr_cblas_fn)(fb_layout_t layout, char jobz, char range,
                                  fb_uplo_t uplo, int n, float *a, int lda,
                                  float vl, float vu, int il, int iu,
                                  float abstol, int *m, float *w, float *z,
                                  int ldz, int *isuppz);

typedef int (*fb_ssyevx_cblas_fn)(fb_layout_t layout, char jobz, char range,
                                  fb_uplo_t uplo, int n, float *a, int lda,
                                  float vl, float vu, int il, int iu,
                                  float abstol, int *m, float *w, float *z,
                                  int ldz, int *ifail);

typedef int (*fb_dsyevr_cblas_fn)(fb_layout_t layout, char jobz, char range,
                                  fb_uplo_t uplo, int n, double *a, int lda,
                                  double vl, double vu, int il, int iu,
                                  double abstol, int *m, double *w, double *z,
                                  int ldz, int *isuppz);

typedef int (*fb_dsyevx_cblas_fn)(fb_layout_t layout, char jobz, char range,
                                  fb_uplo_t uplo, int n, double *a, int lda,
                                  double vl, double vu, int il, int iu,
                                  double abstol, int *m, double *w, double *z,
                                  int ldz, int *ifail);

typedef int (*fb_sstevr_cblas_fn)(fb_layout_t layout, char jobz, char range,
                                  int n, float *d, float *e, float vl,
                                  float vu, int il, int iu, float abstol,
                                  int *m, float *w, float *z, int ldz,
                                  int *isuppz);

typedef int (*fb_dstevx_cblas_fn)(fb_layout_t layout, char jobz, char range,
                                  int n, double *d, double *e, double vl,
                                  double vu, int il, int iu, double abstol,
                                  int *m, double *w, double *z, int ldz,
                                  int *ifail);

typedef void (*fb_sstev_cblas_fn)(int layout, char jobz, int n, float *d,
                                  float *e, float *z, int ldz, float *work,
                                  int *info);

typedef void (*fb_dstev_cblas_fn)(int layout, char jobz, int n, double *d,
                                  double *e, double *z, int ldz, double *work,
                                  int *info);

typedef int (*fb_sstevd_cblas_fn)(fb_layout_t layout, char compz, int n,
                                  float *d, float *e, float *z, int ldz);

typedef int (*fb_dstevd_cblas_fn)(fb_layout_t layout, char compz, int n,
                                  double *d, double *e, double *z, int ldz);

typedef int (*fb_ssygvx_cblas_fn)(fb_layout_t layout, int itype, char jobz,
                                  char range, fb_uplo_t uplo, int n, float *a,
                                  int lda, float *b, int ldb, float vl,
                                  float vu, int il, int iu, float abstol,
                                  int *m, float *w, float *z, int ldz,
                                  int *ifail);

typedef int (*fb_dsygvx_cblas_fn)(fb_layout_t layout, int itype, char jobz,
                                  char range, fb_uplo_t uplo, int n,
                                  double *a, int lda, double *b, int ldb,
                                  double vl, double vu, int il, int iu,
                                  double abstol, int *m, double *w,
                                  double *z, int ldz, int *ifail);

typedef int (*fb_sgeevx_cblas_fn)(fb_layout_t layout, char balanc, char jobvl,
                                  char jobvr, char sense, int n, float *a,
                                  int lda, float *wr, float *wi, float *vl,
                                  int ldvl, float *vr, int ldvr, int *ilo,
                                  int *ihi, float *scale, float *abnrm,
                                  float *rconde, float *rcondv);

typedef int (*fb_dgeevx_cblas_fn)(fb_layout_t layout, char balanc, char jobvl,
                                  char jobvr, char sense, int n, double *a,
                                  int lda, double *wr, double *wi,
                                  double *vl, int ldvl, double *vr, int ldvr,
                                  int *ilo, int *ihi, double *scale,
                                  double *abnrm, double *rconde,
                                  double *rcondv);

typedef int (*fb_cgeevx_cblas_fn)(fb_layout_t layout, char balanc, char jobvl,
                                  char jobvr, char sense, int n,
                                  fb_complex_float_t *a, int lda,
                                  fb_complex_float_t *w,
                                  fb_complex_float_t *vl, int ldvl,
                                  fb_complex_float_t *vr, int ldvr, int *ilo,
                                  int *ihi, float *scale, float *abnrm,
                                  float *rconde, float *rcondv);

typedef int (*fb_zgeevx_cblas_fn)(fb_layout_t layout, char balanc, char jobvl,
                                  char jobvr, char sense, int n,
                                  fb_complex_double_t *a, int lda,
                                  fb_complex_double_t *w,
                                  fb_complex_double_t *vl, int ldvl,
                                  fb_complex_double_t *vr, int ldvr, int *ilo,
                                  int *ihi, double *scale, double *abnrm,
                                  double *rconde, double *rcondv);

typedef int (*fb_sgesvx_cblas_fn)(fb_layout_t layout, char fact, char trans,
                                  int n, int nrhs, float *a, int lda,
                                  float *af, int ldaf, int *ipiv, char *equed,
                                  float *r, float *c, float *b, int ldb,
                                  float *x, int ldx, float *rcond,
                                  float *ferr, float *berr, float *rpivot);

typedef int (*fb_dgesvx_cblas_fn)(fb_layout_t layout, char fact, char trans,
                                  int n, int nrhs, double *a, int lda,
                                  double *af, int ldaf, int *ipiv,
                                  char *equed, double *r, double *c,
                                  double *b, int ldb, double *x, int ldx,
                                  double *rcond, double *ferr, double *berr,
                                  double *rpivot);

typedef int (*fb_cgesvx_cblas_fn)(fb_layout_t layout, char fact, char trans,
                                  int n, int nrhs, fb_complex_float_t *a,
                                  int lda, fb_complex_float_t *af, int ldaf,
                                  int *ipiv, char *equed, float *r, float *c,
                                  fb_complex_float_t *b, int ldb,
                                  fb_complex_float_t *x, int ldx,
                                  float *rcond, float *ferr, float *berr,
                                  float *rpivot);

typedef int (*fb_zgesvx_cblas_fn)(fb_layout_t layout, char fact, char trans,
                                  int n, int nrhs, fb_complex_double_t *a,
                                  int lda, fb_complex_double_t *af, int ldaf,
                                  int *ipiv, char *equed, double *r,
                                  double *c, fb_complex_double_t *b, int ldb,
                                  fb_complex_double_t *x, int ldx,
                                  double *rcond, double *ferr, double *berr,
                                  double *rpivot);

typedef int (*fb_sposvx_cblas_fn)(fb_layout_t layout, char fact, char uplo,
                                  int n, int nrhs, float *a, int lda,
                                  float *af, int ldaf, char *equed, float *s,
                                  float *b, int ldb, float *x, int ldx,
                                  float *rcond, float *ferr, float *berr);

typedef int (*fb_dposvx_cblas_fn)(fb_layout_t layout, char fact, char uplo,
                                  int n, int nrhs, double *a, int lda,
                                  double *af, int ldaf, char *equed,
                                  double *s, double *b, int ldb, double *x,
                                  int ldx, double *rcond, double *ferr,
                                  double *berr);

typedef int (*fb_cposvx_cblas_fn)(fb_layout_t layout, char fact, char uplo,
                                  int n, int nrhs, fb_complex_float_t *a,
                                  int lda, fb_complex_float_t *af, int ldaf,
                                  char *equed, float *s,
                                  fb_complex_float_t *b, int ldb,
                                  fb_complex_float_t *x, int ldx,
                                  float *rcond, float *ferr, float *berr);

typedef int (*fb_zposvx_cblas_fn)(fb_layout_t layout, char fact, char uplo,
                                  int n, int nrhs, fb_complex_double_t *a,
                                  int lda, fb_complex_double_t *af, int ldaf,
                                  char *equed, double *s,
                                  fb_complex_double_t *b, int ldb,
                                  fb_complex_double_t *x, int ldx,
                                  double *rcond, double *ferr, double *berr);

typedef int (*fb_ssysvx_cblas_fn)(fb_layout_t layout, char fact, char uplo,
                                  int n, int nrhs, const float *a, int lda,
                                  float *af, int ldaf, int *ipiv,
                                  const float *b, int ldb, float *x, int ldx,
                                  float *rcond, float *ferr, float *berr);

typedef int (*fb_dsysvx_cblas_fn)(fb_layout_t layout, char fact, char uplo,
                                  int n, int nrhs, const double *a, int lda,
                                  double *af, int ldaf, int *ipiv,
                                  const double *b, int ldb, double *x,
                                  int ldx, double *rcond, double *ferr,
                                  double *berr);

typedef int (*fb_csysvx_cblas_fn)(fb_layout_t layout, char fact, char uplo,
                                  int n, int nrhs,
                                  const fb_complex_float_t *a, int lda,
                                  fb_complex_float_t *af, int ldaf, int *ipiv,
                                  const fb_complex_float_t *b, int ldb,
                                  fb_complex_float_t *x, int ldx,
                                  float *rcond, float *ferr, float *berr);

typedef int (*fb_zsysvx_cblas_fn)(fb_layout_t layout, char fact, char uplo,
                                  int n, int nrhs,
                                  const fb_complex_double_t *a, int lda,
                                  fb_complex_double_t *af, int ldaf,
                                  int *ipiv, const fb_complex_double_t *b,
                                  int ldb, fb_complex_double_t *x, int ldx,
                                  double *rcond, double *ferr, double *berr);

static int g_base_calls = 0;
static int g_alt_calls = 0;
static int g_dense_calls = 0;
static int g_gen_calls = 0;
static int g_syevr_bridge_calls = 0;
static int g_sygvx_bridge_calls = 0;
static int g_dsyevx_bridge_calls = 0;
static int g_ssyevx_bridge_calls = 0;
static int g_dsyevr_bridge_calls = 0;
static int g_dsygvx_bridge_calls = 0;
static int g_sstevr_bridge_calls = 0;
static int g_dstevx_bridge_calls = 0;
static int g_sstevx_bridge_calls = 0;
static int g_dstevr_bridge_calls = 0;
static int g_sstevd_bridge_calls = 0;
static int g_dstevd_bridge_calls = 0;
static int g_sgeevx_bridge_calls = 0;
static int g_dgeevx_bridge_calls = 0;
static int g_cgeevx_bridge_calls = 0;
static int g_zgeevx_bridge_calls = 0;
static int g_sgeev_existing_calls = 0;
static int g_sgesv_existing_calls = 0;
static int g_dgesv_existing_calls = 0;
static int g_cgesv_existing_calls = 0;
static int g_zgesv_existing_calls = 0;
static int g_sgesvx_bridge_calls = 0;
static int g_dgesvx_bridge_calls = 0;
static int g_cgesvx_bridge_calls = 0;
static int g_zgesvx_bridge_calls = 0;
static int g_sgesvx_isolation_a_calls = 0;
static int g_sgesvx_isolation_b_calls = 0;
static int g_dgesvx_isolation_a_calls = 0;
static int g_dgesvx_isolation_b_calls = 0;
static int g_cgesvx_isolation_a_calls = 0;
static int g_cgesvx_isolation_b_calls = 0;
static int g_zgesvx_isolation_a_calls = 0;
static int g_zgesvx_isolation_b_calls = 0;
static int g_sposvx_bridge_calls = 0;
static int g_dposvx_bridge_calls = 0;
static int g_cposvx_bridge_calls = 0;
static int g_zposvx_bridge_calls = 0;
static int g_sposv_existing_calls = 0;
static int g_dposv_existing_calls = 0;
static int g_cposv_existing_calls = 0;
static int g_zposv_existing_calls = 0;
static int g_sposvx_isolation_a_calls = 0;
static int g_sposvx_isolation_b_calls = 0;
static int g_dposvx_isolation_a_calls = 0;
static int g_dposvx_isolation_b_calls = 0;
static int g_cposvx_isolation_a_calls = 0;
static int g_cposvx_isolation_b_calls = 0;
static int g_zposvx_isolation_a_calls = 0;
static int g_zposvx_isolation_b_calls = 0;
static int g_ssysvx_bridge_calls = 0;
static int g_dsysvx_bridge_calls = 0;
static int g_csysvx_bridge_calls = 0;
static int g_zsysvx_bridge_calls = 0;
static int g_ssysv_existing_calls = 0;
static int g_dsysv_existing_calls = 0;
static int g_csysv_existing_calls = 0;
static int g_zsysv_existing_calls = 0;
static int g_ssysvx_isolation_a_calls = 0;
static int g_ssysvx_isolation_b_calls = 0;
static int g_dsysvx_isolation_a_calls = 0;
static int g_dsysvx_isolation_b_calls = 0;
static int g_csysvx_isolation_a_calls = 0;
static int g_csysvx_isolation_b_calls = 0;
static int g_zsysvx_isolation_a_calls = 0;
static int g_zsysvx_isolation_b_calls = 0;
static int g_ssyev_existing_calls = 0;
static int g_sstev_existing_calls = 0;
static int g_ssyevr_isolation_a_calls = 0;
static int g_ssyevr_isolation_b_calls = 0;

static void stub_fortran_ssygvd_dummy(void) {}

static int stub_ssyevr_base(fb_layout_t layout, char jobz, char range,
                            fb_uplo_t uplo, int n, float *a, int lda, float vl,
                            float vu, int il, int iu, float abstol, int *m,
                            float *w, float *z, int ldz, int *isuppz) {
  (void)layout;
  (void)jobz;
  (void)uplo;
  (void)a;
  (void)lda;
  (void)vl;
  (void)vu;
  (void)abstol;
  (void)w;
  (void)z;
  (void)ldz;
  (void)isuppz;
  g_syevr_bridge_calls += 1;

  if (range != 'A' || il != 1 || iu != n) {
    return -222;
  }
  if (m) {
    *m = n;
  }
  return 0;
}

static int stub_ssyevr_isolation_a(fb_layout_t layout, char jobz, char range,
                                   fb_uplo_t uplo, int n, float *a, int lda,
                                   float vl, float vu, int il, int iu,
                                   float abstol, int *m, float *w, float *z,
                                   int ldz, int *isuppz) {
  (void)layout;
  (void)jobz;
  (void)range;
  (void)uplo;
  (void)n;
  (void)a;
  (void)lda;
  (void)vl;
  (void)vu;
  (void)il;
  (void)iu;
  (void)abstol;
  (void)m;
  (void)w;
  (void)z;
  (void)ldz;
  (void)isuppz;
  g_ssyevr_isolation_a_calls += 1;
  return 101;
}

static int stub_ssyevr_isolation_b(fb_layout_t layout, char jobz, char range,
                                   fb_uplo_t uplo, int n, float *a, int lda,
                                   float vl, float vu, int il, int iu,
                                   float abstol, int *m, float *w, float *z,
                                   int ldz, int *isuppz) {
  (void)layout;
  (void)jobz;
  (void)range;
  (void)uplo;
  (void)n;
  (void)a;
  (void)lda;
  (void)vl;
  (void)vu;
  (void)il;
  (void)iu;
  (void)abstol;
  (void)m;
  (void)w;
  (void)z;
  (void)ldz;
  (void)isuppz;
  g_ssyevr_isolation_b_calls += 1;
  return 202;
}

static int stub_ssygvx_base(fb_layout_t layout, int itype, char jobz,
                            char range, fb_uplo_t uplo, int n, float *a,
                            int lda, float *b, int ldb, float vl, float vu,
                            int il, int iu, float abstol, int *m, float *w,
                            float *z, int ldz, int *ifail) {
  (void)layout;
  (void)itype;
  (void)jobz;
  (void)uplo;
  (void)a;
  (void)lda;
  (void)b;
  (void)ldb;
  (void)vl;
  (void)vu;
  (void)abstol;
  (void)w;
  (void)z;
  (void)ldz;
  (void)ifail;
  g_sygvx_bridge_calls += 1;

  if (range != 'A' || il != 1 || iu != n) {
    return -333;
  }
  if (m) {
    *m = n;
  }
  return 0;
}

static int stub_dsyevx_base(fb_layout_t layout, char jobz, char range,
                            fb_uplo_t uplo, int n, double *a, int lda,
                            double vl, double vu, int il, int iu,
                            double abstol, int *m, double *w, double *z,
                            int ldz, int *ifail) {
  (void)layout;
  (void)jobz;
  (void)uplo;
  (void)a;
  (void)lda;
  (void)vl;
  (void)vu;
  (void)abstol;
  (void)w;
  (void)z;
  (void)ldz;
  (void)ifail;
  g_dsyevx_bridge_calls += 1;

  if (range != 'A' || il != 1 || iu != n) {
    return -434;
  }
  if (m) {
    *m = n;
  }
  return 0;
}

static int stub_ssyevx_base(fb_layout_t layout, char jobz, char range,
                            fb_uplo_t uplo, int n, float *a, int lda,
                            float vl, float vu, int il, int iu,
                            float abstol, int *m, float *w, float *z,
                            int ldz, int *ifail) {
  (void)layout;
  (void)jobz;
  (void)uplo;
  (void)a;
  (void)lda;
  (void)vl;
  (void)vu;
  (void)abstol;
  (void)w;
  (void)z;
  (void)ldz;
  (void)ifail;
  g_ssyevx_bridge_calls += 1;

  if (range != 'A' || il != 1 || iu != n) {
    return -535;
  }
  if (m) {
    *m = n;
  }
  return 0;
}

static int stub_dsyevr_base(fb_layout_t layout, char jobz, char range,
                            fb_uplo_t uplo, int n, double *a, int lda,
                            double vl, double vu, int il, int iu,
                            double abstol, int *m, double *w, double *z,
                            int ldz, int *isuppz) {
  (void)layout;
  (void)jobz;
  (void)uplo;
  (void)a;
  (void)lda;
  (void)vl;
  (void)vu;
  (void)abstol;
  (void)w;
  (void)z;
  (void)ldz;
  (void)isuppz;
  g_dsyevr_bridge_calls += 1;

  if (range != 'A' || il != 1 || iu != n) {
    return -636;
  }
  if (m) {
    *m = n;
  }
  return 0;
}

static int stub_sstevr_base(fb_layout_t layout, char jobz, char range, int n,
                            float *d, float *e, float vl, float vu, int il,
                            int iu, float abstol, int *m, float *w, float *z,
                            int ldz, int *isuppz) {
  (void)layout;
  (void)jobz;
  (void)d;
  (void)e;
  (void)vl;
  (void)vu;
  (void)abstol;
  (void)w;
  (void)z;
  (void)ldz;
  (void)isuppz;
  g_sstevr_bridge_calls += 1;

  if (range != 'A' || il != 1 || iu != n) {
    return -808;
  }
  if (m) {
    *m = n;
  }
  return 0;
}

static int stub_dsygvx_base(fb_layout_t layout, int itype, char jobz,
                            char range, fb_uplo_t uplo, int n, double *a,
                            int lda, double *b, int ldb, double vl, double vu,
                            int il, int iu, double abstol, int *m, double *w,
                            double *z, int ldz, int *ifail) {
  (void)layout;
  (void)itype;
  (void)jobz;
  (void)uplo;
  (void)a;
  (void)lda;
  (void)b;
  (void)ldb;
  (void)vl;
  (void)vu;
  (void)abstol;
  (void)w;
  (void)z;
  (void)ldz;
  (void)ifail;
  g_dsygvx_bridge_calls += 1;

  if (range != 'A' || il != 1 || iu != n) {
    return -737;
  }
  if (m) {
    *m = n;
  }
  return 0;
}

static int stub_dstevx_base(fb_layout_t layout, char jobz, char range, int n,
                            double *d, double *e, double vl, double vu, int il,
                            int iu, double abstol, int *m, double *w,
                            double *z, int ldz, int *ifail) {
  (void)layout;
  (void)jobz;
  (void)d;
  (void)e;
  (void)vl;
  (void)vu;
  (void)abstol;
  (void)w;
  (void)z;
  (void)ldz;
  (void)ifail;
  g_dstevx_bridge_calls += 1;

  if (range != 'A' || il != 1 || iu != n) {
    return -909;
  }
  if (m) {
    *m = n;
  }
  return 0;
}

static int stub_sstevx_base(fb_layout_t layout, char jobz, char range, int n,
                            float *d, float *e, float vl, float vu, int il,
                            int iu, float abstol, int *m, float *w, float *z,
                            int ldz, int *ifail) {
  (void)layout;
  (void)jobz;
  (void)d;
  (void)e;
  (void)vl;
  (void)vu;
  (void)abstol;
  (void)w;
  (void)z;
  (void)ldz;
  (void)ifail;
  g_sstevx_bridge_calls += 1;

  if (range != 'A' || il != 1 || iu != n) {
    return -1008;
  }
  if (m) {
    *m = n;
  }
  return 0;
}

static int stub_dstevr_base(fb_layout_t layout, char jobz, char range, int n,
                            double *d, double *e, double vl, double vu, int il,
                            int iu, double abstol, int *m, double *w,
                            double *z, int ldz, int *isuppz) {
  (void)layout;
  (void)jobz;
  (void)d;
  (void)e;
  (void)vl;
  (void)vu;
  (void)abstol;
  (void)w;
  (void)z;
  (void)ldz;
  (void)isuppz;
  g_dstevr_bridge_calls += 1;

  if (range != 'A' || il != 1 || iu != n) {
    return -1109;
  }
  if (m) {
    *m = n;
  }
  return 0;
}

static int stub_sgeevx_base(fb_layout_t layout, char balanc, char jobvl,
                            char jobvr, char sense, int n, float *a, int lda,
                            float *wr, float *wi, float *vl, int ldvl,
                            float *vr, int ldvr, int *ilo, int *ihi,
                            float *scale, float *abnrm, float *rconde,
                            float *rcondv) {
  (void)layout;
  (void)jobvl;
  (void)jobvr;
  (void)a;
  (void)lda;
  (void)wr;
  (void)wi;
  (void)vl;
  (void)ldvl;
  (void)vr;
  (void)ldvr;
  g_sgeevx_bridge_calls += 1;

  if (balanc != 'N' || sense != 'N' || !ilo || !ihi || !scale || !abnrm ||
      !rconde || !rcondv) {
    return -1112;
  }
  *ilo = 1;
  *ihi = n;
  *abnrm = 1.0f;
  return 0;
}

static int stub_dgeevx_base(fb_layout_t layout, char balanc, char jobvl,
                            char jobvr, char sense, int n, double *a,
                            int lda, double *wr, double *wi, double *vl,
                            int ldvl, double *vr, int ldvr, int *ilo,
                            int *ihi, double *scale, double *abnrm,
                            double *rconde, double *rcondv) {
  (void)layout;
  (void)jobvl;
  (void)jobvr;
  (void)a;
  (void)lda;
  (void)wr;
  (void)wi;
  (void)vl;
  (void)ldvl;
  (void)vr;
  (void)ldvr;
  g_dgeevx_bridge_calls += 1;

  if (balanc != 'N' || sense != 'N' || !ilo || !ihi || !scale || !abnrm ||
      !rconde || !rcondv) {
    return -1213;
  }
  *ilo = 1;
  *ihi = n;
  *abnrm = 1.0;
  return 0;
}

static int stub_cgeevx_base(fb_layout_t layout, char balanc, char jobvl,
                            char jobvr, char sense, int n,
                            fb_complex_float_t *a, int lda,
                            fb_complex_float_t *w, fb_complex_float_t *vl,
                            int ldvl, fb_complex_float_t *vr, int ldvr,
                            int *ilo, int *ihi, float *scale, float *abnrm,
                            float *rconde, float *rcondv) {
  (void)layout;
  (void)jobvl;
  (void)jobvr;
  (void)a;
  (void)lda;
  (void)w;
  (void)vl;
  (void)ldvl;
  (void)vr;
  (void)ldvr;
  g_cgeevx_bridge_calls += 1;

  if (balanc != 'N' || sense != 'N' || !ilo || !ihi || !scale || !abnrm ||
      !rconde || !rcondv) {
    return -1314;
  }
  *ilo = 1;
  *ihi = n;
  *abnrm = 1.0f;
  return 0;
}

static int stub_zgeevx_base(fb_layout_t layout, char balanc, char jobvl,
                            char jobvr, char sense, int n,
                            fb_complex_double_t *a, int lda,
                            fb_complex_double_t *w,
                            fb_complex_double_t *vl, int ldvl,
                            fb_complex_double_t *vr, int ldvr, int *ilo,
                            int *ihi, double *scale, double *abnrm,
                            double *rconde, double *rcondv) {
  (void)layout;
  (void)jobvl;
  (void)jobvr;
  (void)a;
  (void)lda;
  (void)w;
  (void)vl;
  (void)ldvl;
  (void)vr;
  (void)ldvr;
  g_zgeevx_bridge_calls += 1;

  if (balanc != 'N' || sense != 'N' || !ilo || !ihi || !scale || !abnrm ||
      !rconde || !rcondv) {
    return -1415;
  }
  *ilo = 1;
  *ihi = n;
  *abnrm = 1.0;
  return 0;
}

static int stub_sgeev_existing(fb_layout_t layout, char jobvl, char jobvr,
                               int n, float *a, int lda, float *wr,
                               float *wi, float *vl, int ldvl, float *vr,
                               int ldvr) {
  (void)layout;
  (void)jobvl;
  (void)jobvr;
  (void)n;
  (void)a;
  (void)lda;
  (void)wr;
  (void)wi;
  (void)vl;
  (void)ldvl;
  (void)vr;
  (void)ldvr;
  g_sgeev_existing_calls += 1;
  return 909;
}

static int stub_sgesvx_base(fb_layout_t layout, char fact, char trans, int n,
                            int nrhs, float *a, int lda, float *af, int ldaf,
                            int *ipiv, char *equed, float *r, float *c,
                            float *b, int ldb, float *x, int ldx,
                            float *rcond, float *ferr, float *berr,
                            float *rpivot) {
  (void)layout;
  g_sgesvx_bridge_calls += 1;

  if (fact != 'N' || trans != 'N' || !a || !af || !ipiv || !equed || !r || !c ||
      !b || !x || !rcond || !ferr || !berr || !rpivot || lda != ldaf ||
      ldb != ldx) {
    return -1516;
  }

  memcpy(af, a, (size_t)lda * (size_t)((n > 0) ? n : 1) * sizeof(float));
  memcpy(x, b, (size_t)ldb * (size_t)((nrhs > 0) ? nrhs : 1) * sizeof(float));
  if (n > 0) {
    ipiv[0] = 7;
  }
  *equed = 'N';
  *rcond = 1.0f;
  return 0;
}

static int stub_sgesvx_isolation_a(fb_layout_t layout, char fact, char trans,
                                   int n, int nrhs, float *a, int lda,
                                   float *af, int ldaf, int *ipiv,
                                   char *equed, float *r, float *c, float *b,
                                   int ldb, float *x, int ldx, float *rcond,
                                   float *ferr, float *berr, float *rpivot) {
  (void)layout;
  (void)n;
  (void)nrhs;
  (void)a;
  (void)lda;
  (void)af;
  (void)ldaf;
  (void)ipiv;
  (void)equed;
  (void)r;
  (void)c;
  (void)b;
  (void)ldb;
  (void)x;
  (void)ldx;
  (void)rcond;
  (void)ferr;
  (void)berr;
  (void)rpivot;
  g_sgesvx_isolation_a_calls += 1;
  if (fact != 'N' || trans != 'N') {
    return -1;
  }
  return 303;
}

static int stub_sgesvx_isolation_b(fb_layout_t layout, char fact, char trans,
                                   int n, int nrhs, float *a, int lda,
                                   float *af, int ldaf, int *ipiv,
                                   char *equed, float *r, float *c, float *b,
                                   int ldb, float *x, int ldx, float *rcond,
                                   float *ferr, float *berr, float *rpivot) {
  (void)layout;
  (void)n;
  (void)nrhs;
  (void)a;
  (void)lda;
  (void)af;
  (void)ldaf;
  (void)ipiv;
  (void)equed;
  (void)r;
  (void)c;
  (void)b;
  (void)ldb;
  (void)x;
  (void)ldx;
  (void)rcond;
  (void)ferr;
  (void)berr;
  (void)rpivot;
  g_sgesvx_isolation_b_calls += 1;
  if (fact != 'N' || trans != 'N') {
    return -1;
  }
  return 404;
}

static int stub_dgesvx_base(fb_layout_t layout, char fact, char trans, int n,
                            int nrhs, double *a, int lda, double *af,
                            int ldaf, int *ipiv, char *equed, double *r,
                            double *c, double *b, int ldb, double *x,
                            int ldx, double *rcond, double *ferr,
                            double *berr, double *rpivot) {
  (void)layout;
  g_dgesvx_bridge_calls += 1;

  if (fact != 'N' || trans != 'N' || !a || !af || !ipiv || !equed || !r || !c ||
      !b || !x || !rcond || !ferr || !berr || !rpivot || lda != ldaf ||
      ldb != ldx) {
    return -1617;
  }

  memcpy(af, a, (size_t)lda * (size_t)((n > 0) ? n : 1) * sizeof(double));
  memcpy(x, b, (size_t)ldb * (size_t)((nrhs > 0) ? nrhs : 1) * sizeof(double));
  if (n > 0) {
    ipiv[0] = 9;
  }
  *equed = 'N';
  *rcond = 1.0;
  return 0;
}

static int stub_dgesvx_isolation_a(fb_layout_t layout, char fact, char trans,
                                   int n, int nrhs, double *a, int lda,
                                   double *af, int ldaf, int *ipiv,
                                   char *equed, double *r, double *c,
                                   double *b, int ldb, double *x, int ldx,
                                   double *rcond, double *ferr, double *berr,
                                   double *rpivot) {
  (void)layout;
  (void)n;
  (void)nrhs;
  (void)a;
  (void)lda;
  (void)af;
  (void)ldaf;
  (void)ipiv;
  (void)equed;
  (void)r;
  (void)c;
  (void)b;
  (void)ldb;
  (void)x;
  (void)ldx;
  (void)rcond;
  (void)ferr;
  (void)berr;
  (void)rpivot;
  g_dgesvx_isolation_a_calls += 1;
  if (fact != 'N' || trans != 'N') {
    return -1;
  }
  return 505;
}

static int stub_dgesvx_isolation_b(fb_layout_t layout, char fact, char trans,
                                   int n, int nrhs, double *a, int lda,
                                   double *af, int ldaf, int *ipiv,
                                   char *equed, double *r, double *c,
                                   double *b, int ldb, double *x, int ldx,
                                   double *rcond, double *ferr, double *berr,
                                   double *rpivot) {
  (void)layout;
  (void)n;
  (void)nrhs;
  (void)a;
  (void)lda;
  (void)af;
  (void)ldaf;
  (void)ipiv;
  (void)equed;
  (void)r;
  (void)c;
  (void)b;
  (void)ldb;
  (void)x;
  (void)ldx;
  (void)rcond;
  (void)ferr;
  (void)berr;
  (void)rpivot;
  g_dgesvx_isolation_b_calls += 1;
  if (fact != 'N' || trans != 'N') {
    return -1;
  }
  return 606;
}

static int stub_sgesv_existing(fb_layout_t layout, int n, int nrhs, float *a,
                               int lda, int *ipiv, float *b, int ldb) {
  (void)layout;
  (void)n;
  (void)nrhs;
  (void)a;
  (void)lda;
  (void)ipiv;
  (void)b;
  (void)ldb;
  g_sgesv_existing_calls += 1;
  return 915;
}

static int stub_dgesv_existing(fb_layout_t layout, int n, int nrhs, double *a,
                               int lda, int *ipiv, double *b, int ldb) {
  (void)layout;
  (void)n;
  (void)nrhs;
  (void)a;
  (void)lda;
  (void)ipiv;
  (void)b;
  (void)ldb;
  g_dgesv_existing_calls += 1;
  return 925;
}

static int stub_cgesv_existing(fb_layout_t layout, int n, int nrhs,
                               fb_complex_float_t *a, int lda, int *ipiv,
                               fb_complex_float_t *b, int ldb) {
  (void)layout;
  (void)n;
  (void)nrhs;
  (void)a;
  (void)lda;
  (void)ipiv;
  (void)b;
  (void)ldb;
  g_cgesv_existing_calls += 1;
  return 919;
}

static int stub_zgesv_existing(fb_layout_t layout, int n, int nrhs,
                               fb_complex_double_t *a, int lda, int *ipiv,
                               fb_complex_double_t *b, int ldb) {
  (void)layout;
  (void)n;
  (void)nrhs;
  (void)a;
  (void)lda;
  (void)ipiv;
  (void)b;
  (void)ldb;
  g_zgesv_existing_calls += 1;
  return 929;
}

static int stub_cgesvx_base(fb_layout_t layout, char fact, char trans, int n,
                            int nrhs, fb_complex_float_t *a, int lda,
                            fb_complex_float_t *af, int ldaf, int *ipiv,
                            char *equed, float *r, float *c,
                            fb_complex_float_t *b, int ldb,
                            fb_complex_float_t *x, int ldx, float *rcond,
                            float *ferr, float *berr, float *rpivot) {
  (void)layout;
  g_cgesvx_bridge_calls += 1;

  if (fact != 'N' || trans != 'N' || !a || !af || !ipiv || !equed || !r || !c ||
      !b || !x || !rcond || !ferr || !berr || !rpivot || lda != ldaf ||
      ldb != ldx) {
    return -1718;
  }

  memcpy(af, a,
         (size_t)lda * (size_t)((n > 0) ? n : 1) * sizeof(fb_complex_float_t));
  memcpy(x, b,
         (size_t)ldb * (size_t)((nrhs > 0) ? nrhs : 1) * sizeof(fb_complex_float_t));
  if (n > 0) {
    ipiv[0] = 11;
  }
  *equed = 'N';
  *rcond = 1.0f;
  return 0;
}

static int stub_cgesvx_isolation_a(fb_layout_t layout, char fact, char trans,
                                   int n, int nrhs, fb_complex_float_t *a,
                                   int lda, fb_complex_float_t *af, int ldaf,
                                   int *ipiv, char *equed, float *r, float *c,
                                   fb_complex_float_t *b, int ldb,
                                   fb_complex_float_t *x, int ldx,
                                   float *rcond, float *ferr, float *berr,
                                   float *rpivot) {
  (void)layout;
  (void)n;
  (void)nrhs;
  (void)a;
  (void)lda;
  (void)af;
  (void)ldaf;
  (void)ipiv;
  (void)equed;
  (void)r;
  (void)c;
  (void)b;
  (void)ldb;
  (void)x;
  (void)ldx;
  (void)rcond;
  (void)ferr;
  (void)berr;
  (void)rpivot;
  g_cgesvx_isolation_a_calls += 1;
  if (fact != 'N' || trans != 'N') {
    return -1;
  }
  return 707;
}

static int stub_cgesvx_isolation_b(fb_layout_t layout, char fact, char trans,
                                   int n, int nrhs, fb_complex_float_t *a,
                                   int lda, fb_complex_float_t *af, int ldaf,
                                   int *ipiv, char *equed, float *r, float *c,
                                   fb_complex_float_t *b, int ldb,
                                   fb_complex_float_t *x, int ldx,
                                   float *rcond, float *ferr, float *berr,
                                   float *rpivot) {
  (void)layout;
  (void)n;
  (void)nrhs;
  (void)a;
  (void)lda;
  (void)af;
  (void)ldaf;
  (void)ipiv;
  (void)equed;
  (void)r;
  (void)c;
  (void)b;
  (void)ldb;
  (void)x;
  (void)ldx;
  (void)rcond;
  (void)ferr;
  (void)berr;
  (void)rpivot;
  g_cgesvx_isolation_b_calls += 1;
  if (fact != 'N' || trans != 'N') {
    return -1;
  }
  return 808;
}

static int stub_zgesvx_base(fb_layout_t layout, char fact, char trans, int n,
                            int nrhs, fb_complex_double_t *a, int lda,
                            fb_complex_double_t *af, int ldaf, int *ipiv,
                            char *equed, double *r, double *c,
                            fb_complex_double_t *b, int ldb,
                            fb_complex_double_t *x, int ldx, double *rcond,
                            double *ferr, double *berr, double *rpivot) {
  (void)layout;
  g_zgesvx_bridge_calls += 1;

  if (fact != 'N' || trans != 'N' || !a || !af || !ipiv || !equed || !r || !c ||
      !b || !x || !rcond || !ferr || !berr || !rpivot || lda != ldaf ||
      ldb != ldx) {
    return -1819;
  }

  memcpy(af, a,
         (size_t)lda * (size_t)((n > 0) ? n : 1) * sizeof(fb_complex_double_t));
  memcpy(x, b,
         (size_t)ldb * (size_t)((nrhs > 0) ? nrhs : 1) * sizeof(fb_complex_double_t));
  if (n > 0) {
    ipiv[0] = 13;
  }
  *equed = 'N';
  *rcond = 1.0;
  return 0;
}

static int stub_zgesvx_isolation_a(fb_layout_t layout, char fact, char trans,
                                   int n, int nrhs, fb_complex_double_t *a,
                                   int lda, fb_complex_double_t *af, int ldaf,
                                   int *ipiv, char *equed, double *r,
                                   double *c, fb_complex_double_t *b, int ldb,
                                   fb_complex_double_t *x, int ldx,
                                   double *rcond, double *ferr, double *berr,
                                   double *rpivot) {
  (void)layout;
  (void)n;
  (void)nrhs;
  (void)a;
  (void)lda;
  (void)af;
  (void)ldaf;
  (void)ipiv;
  (void)equed;
  (void)r;
  (void)c;
  (void)b;
  (void)ldb;
  (void)x;
  (void)ldx;
  (void)rcond;
  (void)ferr;
  (void)berr;
  (void)rpivot;
  g_zgesvx_isolation_a_calls += 1;
  if (fact != 'N' || trans != 'N') {
    return -1;
  }
  return 1007;
}

static int stub_zgesvx_isolation_b(fb_layout_t layout, char fact, char trans,
                                   int n, int nrhs, fb_complex_double_t *a,
                                   int lda, fb_complex_double_t *af, int ldaf,
                                   int *ipiv, char *equed, double *r,
                                   double *c, fb_complex_double_t *b, int ldb,
                                   fb_complex_double_t *x, int ldx,
                                   double *rcond, double *ferr, double *berr,
                                   double *rpivot) {
  (void)layout;
  (void)n;
  (void)nrhs;
  (void)a;
  (void)lda;
  (void)af;
  (void)ldaf;
  (void)ipiv;
  (void)equed;
  (void)r;
  (void)c;
  (void)b;
  (void)ldb;
  (void)x;
  (void)ldx;
  (void)rcond;
  (void)ferr;
  (void)berr;
  (void)rpivot;
  g_zgesvx_isolation_b_calls += 1;
  if (fact != 'N' || trans != 'N') {
    return -1;
  }
  return 1108;
}

static int stub_sposvx_base(fb_layout_t layout, char fact, char uplo, int n,
                            int nrhs, float *a, int lda, float *af, int ldaf,
                            char *equed, float *s, float *b, int ldb,
                            float *x, int ldx, float *rcond, float *ferr,
                            float *berr) {
  (void)layout;
  g_sposvx_bridge_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER || !a || !af || !equed || !s ||
      !b || !x || !rcond || !ferr || !berr || lda != ldaf || ldb != ldx) {
    return -1910;
  }
  memcpy(af, a, (size_t)lda * (size_t)((n > 0) ? n : 1) * sizeof(float));
  memcpy(x, b, (size_t)ldb * (size_t)((nrhs > 0) ? nrhs : 1) * sizeof(float));
  *equed = 'N';
  *rcond = 1.0f;
  return 0;
}

static int stub_sposvx_isolation_a(fb_layout_t layout, char fact, char uplo,
                                   int n, int nrhs, float *a, int lda,
                                   float *af, int ldaf, char *equed,
                                   float *s, float *b, int ldb, float *x,
                                   int ldx, float *rcond, float *ferr,
                                   float *berr) {
  (void)layout;
  (void)n;
  (void)nrhs;
  (void)a;
  (void)lda;
  (void)af;
  (void)ldaf;
  (void)equed;
  (void)s;
  (void)b;
  (void)ldb;
  (void)x;
  (void)ldx;
  (void)rcond;
  (void)ferr;
  (void)berr;
  g_sposvx_isolation_a_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER) {
    return -1;
  }
  return 1209;
}

static int stub_sposvx_isolation_b(fb_layout_t layout, char fact, char uplo,
                                   int n, int nrhs, float *a, int lda,
                                   float *af, int ldaf, char *equed,
                                   float *s, float *b, int ldb, float *x,
                                   int ldx, float *rcond, float *ferr,
                                   float *berr) {
  (void)layout;
  (void)n;
  (void)nrhs;
  (void)a;
  (void)lda;
  (void)af;
  (void)ldaf;
  (void)equed;
  (void)s;
  (void)b;
  (void)ldb;
  (void)x;
  (void)ldx;
  (void)rcond;
  (void)ferr;
  (void)berr;
  g_sposvx_isolation_b_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER) {
    return -1;
  }
  return 1310;
}

static int stub_dposvx_base(fb_layout_t layout, char fact, char uplo, int n,
                            int nrhs, double *a, int lda, double *af,
                            int ldaf, char *equed, double *s, double *b,
                            int ldb, double *x, int ldx, double *rcond,
                            double *ferr, double *berr) {
  (void)layout;
  g_dposvx_bridge_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER || !a || !af || !equed || !s ||
      !b || !x || !rcond || !ferr || !berr || lda != ldaf || ldb != ldx) {
    return -2011;
  }
  memcpy(af, a, (size_t)lda * (size_t)((n > 0) ? n : 1) * sizeof(double));
  memcpy(x, b,
         (size_t)ldb * (size_t)((nrhs > 0) ? nrhs : 1) * sizeof(double));
  *equed = 'N';
  *rcond = 1.0;
  return 0;
}

static int stub_dposvx_isolation_a(fb_layout_t layout, char fact, char uplo,
                                   int n, int nrhs, double *a, int lda,
                                   double *af, int ldaf, char *equed,
                                   double *s, double *b, int ldb, double *x,
                                   int ldx, double *rcond, double *ferr,
                                   double *berr) {
  (void)layout;
  (void)n;
  (void)nrhs;
  (void)a;
  (void)lda;
  (void)af;
  (void)ldaf;
  (void)equed;
  (void)s;
  (void)b;
  (void)ldb;
  (void)x;
  (void)ldx;
  (void)rcond;
  (void)ferr;
  (void)berr;
  g_dposvx_isolation_a_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER) {
    return -1;
  }
  return 1411;
}

static int stub_dposvx_isolation_b(fb_layout_t layout, char fact, char uplo,
                                   int n, int nrhs, double *a, int lda,
                                   double *af, int ldaf, char *equed,
                                   double *s, double *b, int ldb, double *x,
                                   int ldx, double *rcond, double *ferr,
                                   double *berr) {
  (void)layout;
  (void)n;
  (void)nrhs;
  (void)a;
  (void)lda;
  (void)af;
  (void)ldaf;
  (void)equed;
  (void)s;
  (void)b;
  (void)ldb;
  (void)x;
  (void)ldx;
  (void)rcond;
  (void)ferr;
  (void)berr;
  g_dposvx_isolation_b_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER) {
    return -1;
  }
  return 1512;
}

static int stub_sposv_existing(fb_layout_t layout, fb_uplo_t uplo, int n,
                               int nrhs, float *a, int lda, float *b,
                               int ldb) {
  (void)layout;
  (void)uplo;
  (void)n;
  (void)nrhs;
  (void)a;
  (void)lda;
  (void)b;
  (void)ldb;
  g_sposv_existing_calls += 1;
  return 935;
}

static int stub_dposv_existing(fb_layout_t layout, fb_uplo_t uplo, int n,
                               int nrhs, double *a, int lda, double *b,
                               int ldb) {
  (void)layout;
  (void)uplo;
  (void)n;
  (void)nrhs;
  (void)a;
  (void)lda;
  (void)b;
  (void)ldb;
  g_dposv_existing_calls += 1;
  return 945;
}

static int stub_cposvx_base(fb_layout_t layout, char fact, char uplo, int n,
                            int nrhs, fb_complex_float_t *a, int lda,
                            fb_complex_float_t *af, int ldaf, char *equed,
                            float *s, fb_complex_float_t *b, int ldb,
                            fb_complex_float_t *x, int ldx, float *rcond,
                            float *ferr, float *berr) {
  (void)layout;
  g_cposvx_bridge_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER || !a || !af || !equed || !s ||
      !b || !x || !rcond || !ferr || !berr || lda != ldaf || ldb != ldx) {
    return -2112;
  }
  memcpy(af, a,
         (size_t)lda * (size_t)((n > 0) ? n : 1) * sizeof(fb_complex_float_t));
  memcpy(x, b,
         (size_t)ldb * (size_t)((nrhs > 0) ? nrhs : 1) * sizeof(fb_complex_float_t));
  *equed = 'N';
  *rcond = 1.0f;
  return 0;
}

static int stub_cposvx_isolation_a(fb_layout_t layout, char fact, char uplo,
                                   int n, int nrhs, fb_complex_float_t *a,
                                   int lda, fb_complex_float_t *af, int ldaf,
                                   char *equed, float *s,
                                   fb_complex_float_t *b, int ldb,
                                   fb_complex_float_t *x, int ldx,
                                   float *rcond, float *ferr, float *berr) {
  (void)layout;
  (void)n;
  (void)nrhs;
  (void)a;
  (void)lda;
  (void)af;
  (void)ldaf;
  (void)equed;
  (void)s;
  (void)b;
  (void)ldb;
  (void)x;
  (void)ldx;
  (void)rcond;
  (void)ferr;
  (void)berr;
  g_cposvx_isolation_a_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER) {
    return -1;
  }
  return 1613;
}

static int stub_cposvx_isolation_b(fb_layout_t layout, char fact, char uplo,
                                   int n, int nrhs, fb_complex_float_t *a,
                                   int lda, fb_complex_float_t *af, int ldaf,
                                   char *equed, float *s,
                                   fb_complex_float_t *b, int ldb,
                                   fb_complex_float_t *x, int ldx,
                                   float *rcond, float *ferr, float *berr) {
  (void)layout;
  (void)n;
  (void)nrhs;
  (void)a;
  (void)lda;
  (void)af;
  (void)ldaf;
  (void)equed;
  (void)s;
  (void)b;
  (void)ldb;
  (void)x;
  (void)ldx;
  (void)rcond;
  (void)ferr;
  (void)berr;
  g_cposvx_isolation_b_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER) {
    return -1;
  }
  return 1714;
}

static int stub_zposvx_base(fb_layout_t layout, char fact, char uplo, int n,
                            int nrhs, fb_complex_double_t *a, int lda,
                            fb_complex_double_t *af, int ldaf, char *equed,
                            double *s, fb_complex_double_t *b, int ldb,
                            fb_complex_double_t *x, int ldx, double *rcond,
                            double *ferr, double *berr) {
  (void)layout;
  g_zposvx_bridge_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER || !a || !af || !equed || !s ||
      !b || !x || !rcond || !ferr || !berr || lda != ldaf || ldb != ldx) {
    return -2213;
  }
  memcpy(af, a,
         (size_t)lda * (size_t)((n > 0) ? n : 1) * sizeof(fb_complex_double_t));
  memcpy(x, b,
         (size_t)ldb * (size_t)((nrhs > 0) ? nrhs : 1) * sizeof(fb_complex_double_t));
  *equed = 'N';
  *rcond = 1.0;
  return 0;
}

static int stub_zposvx_isolation_a(fb_layout_t layout, char fact, char uplo,
                                   int n, int nrhs, fb_complex_double_t *a,
                                   int lda, fb_complex_double_t *af, int ldaf,
                                   char *equed, double *s,
                                   fb_complex_double_t *b, int ldb,
                                   fb_complex_double_t *x, int ldx,
                                   double *rcond, double *ferr,
                                   double *berr) {
  (void)layout;
  (void)n;
  (void)nrhs;
  (void)a;
  (void)lda;
  (void)af;
  (void)ldaf;
  (void)equed;
  (void)s;
  (void)b;
  (void)ldb;
  (void)x;
  (void)ldx;
  (void)rcond;
  (void)ferr;
  (void)berr;
  g_zposvx_isolation_a_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER) {
    return -1;
  }
  return 1815;
}

static int stub_zposvx_isolation_b(fb_layout_t layout, char fact, char uplo,
                                   int n, int nrhs, fb_complex_double_t *a,
                                   int lda, fb_complex_double_t *af, int ldaf,
                                   char *equed, double *s,
                                   fb_complex_double_t *b, int ldb,
                                   fb_complex_double_t *x, int ldx,
                                   double *rcond, double *ferr,
                                   double *berr) {
  (void)layout;
  (void)n;
  (void)nrhs;
  (void)a;
  (void)lda;
  (void)af;
  (void)ldaf;
  (void)equed;
  (void)s;
  (void)b;
  (void)ldb;
  (void)x;
  (void)ldx;
  (void)rcond;
  (void)ferr;
  (void)berr;
  g_zposvx_isolation_b_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER) {
    return -1;
  }
  return 1916;
}

static int stub_cposv_existing(fb_layout_t layout, fb_uplo_t uplo, int n,
                               int nrhs, fb_complex_float_t *a, int lda,
                               fb_complex_float_t *b, int ldb) {
  (void)layout;
  (void)uplo;
  (void)n;
  (void)nrhs;
  (void)a;
  (void)lda;
  (void)b;
  (void)ldb;
  g_cposv_existing_calls += 1;
  return 955;
}

static int stub_zposv_existing(fb_layout_t layout, fb_uplo_t uplo, int n,
                               int nrhs, fb_complex_double_t *a, int lda,
                               fb_complex_double_t *b, int ldb) {
  (void)layout;
  (void)uplo;
  (void)n;
  (void)nrhs;
  (void)a;
  (void)lda;
  (void)b;
  (void)ldb;
  g_zposv_existing_calls += 1;
  return 965;
}

static int stub_ssysvx_base(fb_layout_t layout, char fact, char uplo, int n,
                            int nrhs, const float *a, int lda, float *af,
                            int ldaf, int *ipiv, const float *b, int ldb,
                            float *x, int ldx, float *rcond, float *ferr,
                            float *berr) {
  (void)layout;
  g_ssysvx_bridge_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER || !a || !af || !ipiv || !b ||
      !x || !rcond || !ferr || !berr || lda != ldaf || ldb != ldx) {
    return -2314;
  }
  memcpy(af, a, (size_t)lda * (size_t)((n > 0) ? n : 1) * sizeof(float));
  memcpy(x, b, (size_t)ldb * (size_t)((nrhs > 0) ? nrhs : 1) * sizeof(float));
  if (n > 0) {
    ipiv[0] = 3;
  }
  *rcond = 1.0f;
  return 0;
}

static int stub_dsysvx_base(fb_layout_t layout, char fact, char uplo, int n,
                            int nrhs, const double *a, int lda, double *af,
                            int ldaf, int *ipiv, const double *b, int ldb,
                            double *x, int ldx, double *rcond, double *ferr,
                            double *berr) {
  (void)layout;
  g_dsysvx_bridge_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER || !a || !af || !ipiv || !b ||
      !x || !rcond || !ferr || !berr || lda != ldaf || ldb != ldx) {
    return -2415;
  }
  memcpy(af, a, (size_t)lda * (size_t)((n > 0) ? n : 1) * sizeof(double));
  memcpy(x, b,
         (size_t)ldb * (size_t)((nrhs > 0) ? nrhs : 1) * sizeof(double));
  if (n > 0) {
    ipiv[0] = 5;
  }
  *rcond = 1.0;
  return 0;
}

static int stub_csysvx_base(fb_layout_t layout, char fact, char uplo, int n,
                            int nrhs, const fb_complex_float_t *a, int lda,
                            fb_complex_float_t *af, int ldaf, int *ipiv,
                            const fb_complex_float_t *b, int ldb,
                            fb_complex_float_t *x, int ldx, float *rcond,
                            float *ferr, float *berr) {
  (void)layout;
  g_csysvx_bridge_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER || !a || !af || !ipiv || !b ||
      !x || !rcond || !ferr || !berr || lda != ldaf || ldb != ldx) {
    return -2516;
  }
  memcpy(af, a,
         (size_t)lda * (size_t)((n > 0) ? n : 1) * sizeof(fb_complex_float_t));
  memcpy(x, b,
         (size_t)ldb * (size_t)((nrhs > 0) ? nrhs : 1) * sizeof(fb_complex_float_t));
  if (n > 0) {
    ipiv[0] = 7;
  }
  *rcond = 1.0f;
  return 0;
}

static int stub_zsysvx_base(fb_layout_t layout, char fact, char uplo, int n,
                            int nrhs, const fb_complex_double_t *a, int lda,
                            fb_complex_double_t *af, int ldaf, int *ipiv,
                            const fb_complex_double_t *b, int ldb,
                            fb_complex_double_t *x, int ldx, double *rcond,
                            double *ferr, double *berr) {
  (void)layout;
  g_zsysvx_bridge_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER || !a || !af || !ipiv || !b ||
      !x || !rcond || !ferr || !berr || lda != ldaf || ldb != ldx) {
    return -2617;
  }
  memcpy(af, a,
         (size_t)lda * (size_t)((n > 0) ? n : 1) * sizeof(fb_complex_double_t));
  memcpy(x, b,
         (size_t)ldb * (size_t)((nrhs > 0) ? nrhs : 1) * sizeof(fb_complex_double_t));
  if (n > 0) {
    ipiv[0] = 9;
  }
  *rcond = 1.0;
  return 0;
}

static int stub_ssysvx_isolation_a(fb_layout_t layout, char fact, char uplo,
                                   int n, int nrhs, const float *a, int lda,
                                   float *af, int ldaf, int *ipiv,
                                   const float *b, int ldb, float *x, int ldx,
                                   float *rcond, float *ferr, float *berr) {
  (void)layout;
  (void)n;
  (void)nrhs;
  (void)a;
  (void)lda;
  (void)af;
  (void)ldaf;
  (void)ipiv;
  (void)b;
  (void)ldb;
  (void)x;
  (void)ldx;
  (void)rcond;
  (void)ferr;
  (void)berr;
  g_ssysvx_isolation_a_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER) {
    return -1;
  }
  return 2017;
}

static int stub_ssysvx_isolation_b(fb_layout_t layout, char fact, char uplo,
                                   int n, int nrhs, const float *a, int lda,
                                   float *af, int ldaf, int *ipiv,
                                   const float *b, int ldb, float *x, int ldx,
                                   float *rcond, float *ferr, float *berr) {
  (void)layout;
  (void)n;
  (void)nrhs;
  (void)a;
  (void)lda;
  (void)af;
  (void)ldaf;
  (void)ipiv;
  (void)b;
  (void)ldb;
  (void)x;
  (void)ldx;
  (void)rcond;
  (void)ferr;
  (void)berr;
  g_ssysvx_isolation_b_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER) {
    return -1;
  }
  return 2118;
}

static int stub_dsysvx_isolation_a(fb_layout_t layout, char fact, char uplo,
                                   int n, int nrhs, const double *a, int lda,
                                   double *af, int ldaf, int *ipiv,
                                   const double *b, int ldb, double *x,
                                   int ldx, double *rcond, double *ferr,
                                   double *berr) {
  (void)layout;
  (void)n;
  (void)nrhs;
  (void)a;
  (void)lda;
  (void)af;
  (void)ldaf;
  (void)ipiv;
  (void)b;
  (void)ldb;
  (void)x;
  (void)ldx;
  (void)rcond;
  (void)ferr;
  (void)berr;
  g_dsysvx_isolation_a_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER) {
    return -1;
  }
  return 2219;
}

static int stub_dsysvx_isolation_b(fb_layout_t layout, char fact, char uplo,
                                   int n, int nrhs, const double *a, int lda,
                                   double *af, int ldaf, int *ipiv,
                                   const double *b, int ldb, double *x,
                                   int ldx, double *rcond, double *ferr,
                                   double *berr) {
  (void)layout;
  (void)n;
  (void)nrhs;
  (void)a;
  (void)lda;
  (void)af;
  (void)ldaf;
  (void)ipiv;
  (void)b;
  (void)ldb;
  (void)x;
  (void)ldx;
  (void)rcond;
  (void)ferr;
  (void)berr;
  g_dsysvx_isolation_b_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER) {
    return -1;
  }
  return 2320;
}

static int stub_ssysv_existing(fb_layout_t layout, char uplo, int n, int nrhs,
                               float *a, int lda, int *ipiv, float *b,
                               int ldb) {
  (void)layout;
  (void)uplo;
  (void)n;
  (void)nrhs;
  (void)a;
  (void)lda;
  (void)ipiv;
  (void)b;
  (void)ldb;
  g_ssysv_existing_calls += 1;
  return 975;
}

static int stub_dsysv_existing(fb_layout_t layout, char uplo, int n, int nrhs,
                               double *a, int lda, int *ipiv, double *b,
                               int ldb) {
  (void)layout;
  (void)uplo;
  (void)n;
  (void)nrhs;
  (void)a;
  (void)lda;
  (void)ipiv;
  (void)b;
  (void)ldb;
  g_dsysv_existing_calls += 1;
  return 985;
}

static int stub_csysv_existing(fb_layout_t layout, char uplo, int n, int nrhs,
                               fb_complex_float_t *a, int lda, int *ipiv,
                               fb_complex_float_t *b, int ldb) {
  (void)layout;
  (void)uplo;
  (void)n;
  (void)nrhs;
  (void)a;
  (void)lda;
  (void)ipiv;
  (void)b;
  (void)ldb;
  g_csysv_existing_calls += 1;
  return 995;
}

static int stub_zsysv_existing(fb_layout_t layout, char uplo, int n, int nrhs,
                               fb_complex_double_t *a, int lda, int *ipiv,
                               fb_complex_double_t *b, int ldb) {
  (void)layout;
  (void)uplo;
  (void)n;
  (void)nrhs;
  (void)a;
  (void)lda;
  (void)ipiv;
  (void)b;
  (void)ldb;
  g_zsysv_existing_calls += 1;
  return 1005;
}

static int stub_csysvx_isolation_a(
    fb_layout_t layout, char fact, char uplo, int n, int nrhs,
    const fb_complex_float_t *a, int lda, fb_complex_float_t *af, int ldaf,
    int *ipiv, const fb_complex_float_t *b, int ldb, fb_complex_float_t *x,
    int ldx, float *rcond, float *ferr, float *berr) {
  (void)layout;
  (void)n;
  (void)nrhs;
  (void)a;
  (void)lda;
  (void)af;
  (void)ldaf;
  (void)ipiv;
  (void)b;
  (void)ldb;
  (void)x;
  (void)ldx;
  (void)rcond;
  (void)ferr;
  (void)berr;
  g_csysvx_isolation_a_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER) {
    return -1;
  }
  return 2421;
}

static int stub_csysvx_isolation_b(
    fb_layout_t layout, char fact, char uplo, int n, int nrhs,
    const fb_complex_float_t *a, int lda, fb_complex_float_t *af, int ldaf,
    int *ipiv, const fb_complex_float_t *b, int ldb, fb_complex_float_t *x,
    int ldx, float *rcond, float *ferr, float *berr) {
  (void)layout;
  (void)n;
  (void)nrhs;
  (void)a;
  (void)lda;
  (void)af;
  (void)ldaf;
  (void)ipiv;
  (void)b;
  (void)ldb;
  (void)x;
  (void)ldx;
  (void)rcond;
  (void)ferr;
  (void)berr;
  g_csysvx_isolation_b_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER) {
    return -1;
  }
  return 2522;
}

static int stub_zsysvx_isolation_a(
    fb_layout_t layout, char fact, char uplo, int n, int nrhs,
    const fb_complex_double_t *a, int lda, fb_complex_double_t *af, int ldaf,
    int *ipiv, const fb_complex_double_t *b, int ldb, fb_complex_double_t *x,
    int ldx, double *rcond, double *ferr, double *berr) {
  (void)layout;
  (void)n;
  (void)nrhs;
  (void)a;
  (void)lda;
  (void)af;
  (void)ldaf;
  (void)ipiv;
  (void)b;
  (void)ldb;
  (void)x;
  (void)ldx;
  (void)rcond;
  (void)ferr;
  (void)berr;
  g_zsysvx_isolation_a_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER) {
    return -1;
  }
  return 2623;
}

static int stub_zsysvx_isolation_b(
    fb_layout_t layout, char fact, char uplo, int n, int nrhs,
    const fb_complex_double_t *a, int lda, fb_complex_double_t *af, int ldaf,
    int *ipiv, const fb_complex_double_t *b, int ldb, fb_complex_double_t *x,
    int ldx, double *rcond, double *ferr, double *berr) {
  (void)layout;
  (void)n;
  (void)nrhs;
  (void)a;
  (void)lda;
  (void)af;
  (void)ldaf;
  (void)ipiv;
  (void)b;
  (void)ldb;
  (void)x;
  (void)ldx;
  (void)rcond;
  (void)ferr;
  (void)berr;
  g_zsysvx_isolation_b_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER) {
    return -1;
  }
  return 2724;
}

static int stub_sstevd_base(fb_layout_t layout, char compz, int n, float *d,
                            float *e, float *z, int ldz) {
  (void)layout;
  (void)n;
  (void)d;
  (void)e;
  (void)z;
  (void)ldz;
  g_sstevd_bridge_calls += 1;

  if (compz != 'I') {
    return -1210;
  }
  return 0;
}

static int stub_dstevd_base(fb_layout_t layout, char compz, int n, double *d,
                            double *e, double *z, int ldz) {
  (void)layout;
  (void)n;
  (void)d;
  (void)e;
  (void)z;
  (void)ldz;
  g_dstevd_bridge_calls += 1;

  if (compz != 'I') {
    return -1311;
  }
  return 0;
}

static int stub_ssyev_existing(fb_layout_t layout, char jobz, fb_uplo_t uplo,
                               int n, float *a, int lda, float *w) {
  (void)layout;
  (void)jobz;
  (void)uplo;
  (void)n;
  (void)a;
  (void)lda;
  (void)w;
  g_ssyev_existing_calls += 1;
  return 911;
}

static void stub_sstev_existing(int layout, char jobz, int n, float *d,
                                float *e, float *z, int ldz, float *work,
                                int *info) {
  (void)layout;
  (void)jobz;
  (void)n;
  (void)d;
  (void)e;
  (void)z;
  (void)ldz;
  (void)work;
  g_sstev_existing_calls += 1;
  if (info) {
    *info = 777;
  }
}

static int stub_chbevd_base(fb_layout_t layout, char jobz, fb_uplo_t uplo,
                            int n, int kd, fb_complex_float_t *ab, int ldab,
                            float *w, fb_complex_float_t *z, int ldz) {
  (void)layout;
  (void)jobz;
  (void)uplo;
  (void)n;
  (void)kd;
  (void)ab;
  (void)ldab;
  (void)w;
  (void)z;
  (void)ldz;
  g_base_calls += 1;
  return 777;
}

static int stub_chbev_alt(fb_layout_t layout, char jobz, fb_uplo_t uplo,
                          int n, int kd, fb_complex_float_t *ab, int ldab,
                          float *w, fb_complex_float_t *z, int ldz) {
  (void)layout;
  (void)jobz;
  (void)uplo;
  (void)n;
  (void)kd;
  (void)ab;
  (void)ldab;
  (void)w;
  (void)z;
  (void)ldz;
  g_alt_calls += 1;
  return 888;
}

static int test_group_fill_from_chbevd(void) {
  fb_backend_vtable_t vtable;
  fb_chbev_cblas_fn chbev_call = NULL;
  fb_status_t status;

  fb_complex_float_t ab[4] = {0};
  fb_complex_float_t z[4] = {0};
  float w[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_base_calls = 0;
  g_alt_calls = 0;

  vtable.ext_ops[FB_OP_CHBEVD][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_chbevd_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  if (vtable.ext_ops[FB_OP_CHBEV][FB_CONV_CBLAS] !=
      vtable.ext_ops[FB_OP_CHBEVD][FB_CONV_CBLAS]) {
    fprintf(stderr,
            "[FAIL] compatibility autofill did not propagate CHBEVD -> CHBEV\n");
    return 1;
  }

  if (vtable.ext_ops[FB_OP_CHBEV][FB_CONV_FORTRAN] == NULL) {
    fprintf(stderr,
            "[FAIL] conv-thunk install did not expose CHBEV Fortran slot\n");
    return 1;
  }

  if (vtable.chbev == NULL) {
    fprintf(stderr,
            "[FAIL] named field CHBEV not backfilled from ext_ops slot\n");
    return 1;
  }

  chbev_call = (fb_chbev_cblas_fn)vtable.ext_ops[FB_OP_CHBEV][FB_CONV_CBLAS];
  if (!chbev_call) {
    fprintf(stderr, "[FAIL] CHBEV CBLAS slot is NULL after finalize\n");
    return 1;
  }

  if (chbev_call(FB_LAYOUT_COL_MAJOR, 'N', FB_UPPER, 2, 1, ab, 2, w, z, 2) !=
      777) {
    fprintf(stderr, "[FAIL] CHBEV routed call did not invoke base donor\n");
    return 1;
  }

  if (g_base_calls != 1 || g_alt_calls != 0) {
    fprintf(stderr,
            "[FAIL] CHBEV call accounting mismatch (base=%d alt=%d)\n",
            g_base_calls, g_alt_calls);
    return 1;
  }

  printf("[PASS] CHBEVD donor auto-fills CHBEV compatible slot\n");
  return 0;
}

static int test_existing_slot_not_overwritten(void) {
  fb_backend_vtable_t vtable;
  fb_chbev_cblas_fn chbev_call = NULL;
  fb_status_t status;

  fb_complex_float_t ab[4] = {0};
  fb_complex_float_t z[4] = {0};
  float w[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_base_calls = 0;
  g_alt_calls = 0;

  vtable.ext_ops[FB_OP_CHBEVD][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_chbevd_base;
  vtable.ext_ops[FB_OP_CHBEV][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_chbev_alt;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  chbev_call = (fb_chbev_cblas_fn)vtable.ext_ops[FB_OP_CHBEV][FB_CONV_CBLAS];
  if (!chbev_call) {
    fprintf(stderr, "[FAIL] CHBEV slot became NULL\n");
    return 1;
  }

  if (chbev_call(FB_LAYOUT_COL_MAJOR, 'N', FB_UPPER, 2, 1, ab, 2, w, z, 2) !=
      888) {
    fprintf(stderr,
            "[FAIL] compatibility autofill overwrote existing CHBEV slot\n");
    return 1;
  }

  if (g_alt_calls != 1 || g_base_calls != 0) {
    fprintf(stderr,
            "[FAIL] overwrite guard mismatch (base=%d alt=%d)\n",
            g_base_calls, g_alt_calls);
    return 1;
  }

  printf("[PASS] existing compatible slot keeps native implementation\n");
  return 0;
}

static int stub_ssyevd_base(fb_layout_t layout, char jobz, fb_uplo_t uplo,
                            int n, float *a, int lda, float *w) {
  (void)layout;
  (void)jobz;
  (void)uplo;
  (void)n;
  (void)a;
  (void)lda;
  (void)w;
  g_dense_calls += 1;
  return 333;
}

static int test_group_fill_from_ssyevd(void) {
  fb_backend_vtable_t vtable;
  fb_ssyev_cblas_fn ssyev_call = NULL;
  fb_status_t status;

  float a[4] = {0};
  float w[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_dense_calls = 0;

  vtable.ext_ops[FB_OP_SSYEVD][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_ssyevd_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  if (vtable.ext_ops[FB_OP_SSYEV][FB_CONV_CBLAS] !=
      vtable.ext_ops[FB_OP_SSYEVD][FB_CONV_CBLAS]) {
    fprintf(stderr,
            "[FAIL] compatibility autofill did not propagate SSYEVD -> SSYEV\n");
    return 1;
  }

  ssyev_call = (fb_ssyev_cblas_fn)vtable.ext_ops[FB_OP_SSYEV][FB_CONV_CBLAS];
  if (!ssyev_call) {
    fprintf(stderr, "[FAIL] SSYEV CBLAS slot is NULL after finalize\n");
    return 1;
  }

  if (ssyev_call(FB_LAYOUT_COL_MAJOR, 'N', FB_UPPER, 2, a, 2, w) != 333) {
    fprintf(stderr, "[FAIL] SSYEV routed call did not invoke base donor\n");
    return 1;
  }

  if (g_dense_calls != 1) {
    fprintf(stderr, "[FAIL] SSYEV call accounting mismatch (calls=%d)\n",
            g_dense_calls);
    return 1;
  }

  printf("[PASS] SSYEVD donor auto-fills SSYEV compatible slot\n");
  return 0;
}

static int stub_ssygvd_base(fb_layout_t layout, int itype, char jobz,
                            fb_uplo_t uplo, int n, float *a, int lda,
                            float *b, int ldb, float *w) {
  (void)layout;
  (void)itype;
  (void)jobz;
  (void)uplo;
  (void)n;
  (void)a;
  (void)lda;
  (void)b;
  (void)ldb;
  (void)w;
  g_gen_calls += 1;
  return 444;
}

static int test_group_fill_from_ssygvd(void) {
  fb_backend_vtable_t vtable;
  fb_ssygv_cblas_fn ssygv_call = NULL;
  fb_status_t status;

  float a[4] = {0};
  float b[4] = {0};
  float w[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_gen_calls = 0;

  vtable.ext_ops[FB_OP_SSYGVD][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_ssygvd_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  if (vtable.ext_ops[FB_OP_SSYGV][FB_CONV_CBLAS] !=
      vtable.ext_ops[FB_OP_SSYGVD][FB_CONV_CBLAS]) {
    fprintf(stderr,
            "[FAIL] compatibility autofill did not propagate SSYGVD -> SSYGV\n");
    return 1;
  }

  ssygv_call = (fb_ssygv_cblas_fn)vtable.ext_ops[FB_OP_SSYGV][FB_CONV_CBLAS];
  if (!ssygv_call) {
    fprintf(stderr, "[FAIL] SSYGV CBLAS slot is NULL after finalize\n");
    return 1;
  }

  if (ssygv_call(FB_LAYOUT_COL_MAJOR, 1, 'N', FB_UPPER, 2, a, 2, b, 2, w) !=
      444) {
    fprintf(stderr, "[FAIL] SSYGV routed call did not invoke base donor\n");
    return 1;
  }

  if (g_gen_calls != 1) {
    fprintf(stderr, "[FAIL] SSYGV call accounting mismatch (calls=%d)\n",
            g_gen_calls);
    return 1;
  }

  printf("[PASS] SSYGVD donor auto-fills SSYGV compatible slot\n");
  return 0;
}

static int test_fortran_only_donor_propagates_after_conv_thunks(void) {
  fb_backend_vtable_t vtable;
  fb_status_t status;

  memset(&vtable, 0, sizeof(vtable));

  /* No CBLAS donor initially: only Fortran slot is populated. */
  vtable.ext_ops[FB_OP_SSYGVD][FB_CONV_FORTRAN] =
      (fb_generic_fn)(void (*)(void))stub_fortran_ssygvd_dummy;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  if (vtable.ext_ops[FB_OP_SSYGVD][FB_CONV_CBLAS] == NULL) {
    fprintf(stderr,
            "[FAIL] conv thunk pass did not expose SSYGVD CBLAS slot\n");
    return 1;
  }

  if (vtable.ext_ops[FB_OP_SSYGV][FB_CONV_CBLAS] == NULL) {
    fprintf(stderr,
            "[FAIL] post-thunk compatibility pass did not fill SSYGV slot\n");
    return 1;
  }

  if (vtable.ext_ops[FB_OP_SSYGV][FB_CONV_CBLAS] !=
      vtable.ext_ops[FB_OP_SSYGVD][FB_CONV_CBLAS]) {
    fprintf(stderr,
            "[FAIL] SSYGV slot did not inherit Fortran-derived SSYGVD donor\n");
    return 1;
  }

  printf("[PASS] Fortran-only donor propagates after post-thunk compatibility pass\n");
  return 0;
}

static int test_adapter_bridge_ssyev_from_ssyevr(void) {
  fb_backend_vtable_t vtable;
  fb_ssyev_cblas_fn ssyev_call = NULL;
  fb_status_t status;

  float a[4] = {0};
  float w[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_syevr_bridge_calls = 0;

  vtable.ext_ops[FB_OP_SSYEVR][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_ssyevr_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  ssyev_call = (fb_ssyev_cblas_fn)vtable.ext_ops[FB_OP_SSYEV][FB_CONV_CBLAS];
  if (!ssyev_call) {
    fprintf(stderr, "[FAIL] SSYEV adapter slot missing after finalize\n");
    return 1;
  }

  if (ssyev_call(FB_LAYOUT_COL_MAJOR, 'N', FB_UPPER, 2, a, 2, w) != 0) {
    fprintf(stderr, "[FAIL] SSYEV adapter bridge returned error\n");
    return 1;
  }

  if (g_syevr_bridge_calls != 1) {
    fprintf(stderr,
            "[FAIL] SSYEV<-SSYEVR adapter call count mismatch (calls=%d)\n",
            g_syevr_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter bridge SSYEV <- SSYEVR applies default range params\n");
  return 0;
}

static int test_adapter_bridge_ssygv_from_ssygvx(void) {
  fb_backend_vtable_t vtable;
  fb_ssygv_cblas_fn ssygv_call = NULL;
  fb_status_t status;

  float a[4] = {0};
  float b[4] = {0};
  float w[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_sygvx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_SSYGVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_ssygvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  ssygv_call = (fb_ssygv_cblas_fn)vtable.ext_ops[FB_OP_SSYGV][FB_CONV_CBLAS];
  if (!ssygv_call) {
    fprintf(stderr, "[FAIL] SSYGV adapter slot missing after finalize\n");
    return 1;
  }

  if (ssygv_call(FB_LAYOUT_COL_MAJOR, 1, 'N', FB_UPPER, 2, a, 2, b, 2, w) !=
      0) {
    fprintf(stderr, "[FAIL] SSYGV adapter bridge returned error\n");
    return 1;
  }

  if (g_sygvx_bridge_calls != 1) {
    fprintf(stderr,
            "[FAIL] SSYGV<-SSYGVX adapter call count mismatch (calls=%d)\n",
            g_sygvx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter bridge SSYGV <- SSYGVX applies default range params\n");
  return 0;
}

static int test_adapter_bridge_ssyev_from_ssyevx(void) {
  fb_backend_vtable_t vtable;
  fb_ssyev_cblas_fn ssyev_call = NULL;
  fb_status_t status;

  float a[4] = {0};
  float w[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_ssyevx_bridge_calls = 0;

  /* No SSYEVR donor: adapter should use SSYEVX source. */
  vtable.ext_ops[FB_OP_SSYEVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_ssyevx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  ssyev_call = (fb_ssyev_cblas_fn)vtable.ext_ops[FB_OP_SSYEV][FB_CONV_CBLAS];
  if (!ssyev_call) {
    fprintf(stderr, "[FAIL] SSYEV adapter slot missing after finalize\n");
    return 1;
  }

  if (ssyev_call(FB_LAYOUT_COL_MAJOR, 'V', FB_UPPER, 2, a, 2, w) != 0) {
    fprintf(stderr, "[FAIL] SSYEV adapter bridge from SSYEVX returned error\n");
    return 1;
  }

  if (g_ssyevx_bridge_calls != 1) {
    fprintf(stderr,
            "[FAIL] SSYEV<-SSYEVX adapter call count mismatch (calls=%d)\n",
            g_ssyevx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter bridge SSYEV <- SSYEVX applies default range params\n");
  return 0;
}

static int test_adapter_bridge_dsyev_from_dsyevr(void) {
  fb_backend_vtable_t vtable;
  fb_dsyev_fn dsyev_call = NULL;
  fb_status_t status;

  double a[4] = {0};
  double w[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_dsyevr_bridge_calls = 0;

  vtable.ext_ops[FB_OP_DSYEVR][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dsyevr_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  dsyev_call = (fb_dsyev_fn)vtable.ext_ops[FB_OP_DSYEV][FB_CONV_CBLAS];
  if (!dsyev_call) {
    fprintf(stderr, "[FAIL] DSYEV adapter slot missing after finalize\n");
    return 1;
  }

  if (dsyev_call(FB_LAYOUT_COL_MAJOR, 'V', FB_UPPER, 2, a, 2, w) != 0) {
    fprintf(stderr, "[FAIL] DSYEV adapter bridge from DSYEVR returned error\n");
    return 1;
  }

  if (g_dsyevr_bridge_calls != 1) {
    fprintf(stderr,
            "[FAIL] DSYEV<-DSYEVR adapter call count mismatch (calls=%d)\n",
            g_dsyevr_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter bridge DSYEV <- DSYEVR applies default range params\n");
  return 0;
}

static int test_adapter_bridge_dsygv_from_dsygvx(void) {
  fb_backend_vtable_t vtable;
  fb_dsygv_fn dsygv_call = NULL;
  fb_status_t status;

  double a[4] = {0};
  double b[4] = {0};
  double w[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_dsygvx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_DSYGVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dsygvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  dsygv_call = (fb_dsygv_fn)vtable.ext_ops[FB_OP_DSYGV][FB_CONV_CBLAS];
  if (!dsygv_call) {
    fprintf(stderr, "[FAIL] DSYGV adapter slot missing after finalize\n");
    return 1;
  }

  if (dsygv_call(FB_LAYOUT_COL_MAJOR, 1, 'V', FB_UPPER, 2, a, 2, b, 2, w) !=
      0) {
    fprintf(stderr, "[FAIL] DSYGV adapter bridge from DSYGVX returned error\n");
    return 1;
  }

  if (g_dsygvx_bridge_calls != 1) {
    fprintf(stderr,
            "[FAIL] DSYGV<-DSYGVX adapter call count mismatch (calls=%d)\n",
            g_dsygvx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter bridge DSYGV <- DSYGVX applies default range params\n");
  return 0;
}

static int test_adapter_bridge_dsyev_from_dsyevx(void) {
  fb_backend_vtable_t vtable;
  fb_dsyev_fn dsyev_call = NULL;
  fb_status_t status;

  double a[4] = {0};
  double w[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_dsyevx_bridge_calls = 0;

  /* No DSYEVR donor: adapter should use DSYEVX source. */
  vtable.ext_ops[FB_OP_DSYEVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dsyevx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  dsyev_call = (fb_dsyev_fn)vtable.ext_ops[FB_OP_DSYEV][FB_CONV_CBLAS];
  if (!dsyev_call) {
    fprintf(stderr, "[FAIL] DSYEV adapter slot missing after finalize\n");
    return 1;
  }

  if (dsyev_call(FB_LAYOUT_COL_MAJOR, 'V', FB_UPPER, 2, a, 2, w) != 0) {
    fprintf(stderr, "[FAIL] DSYEV adapter bridge from DSYEVX returned error\n");
    return 1;
  }

  if (g_dsyevx_bridge_calls != 1) {
    fprintf(stderr,
            "[FAIL] DSYEV<-DSYEVX adapter call count mismatch (calls=%d)\n",
            g_dsyevx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter bridge DSYEV <- DSYEVX applies default range params\n");
  return 0;
}

static int test_adapter_bridge_sgeev_from_sgeevx(void) {
  fb_backend_vtable_t vtable;
  fb_sgeev_fn sgeev_call = NULL;
  fb_status_t status;

  float a[4] = {0};
  float wr[2] = {0};
  float wi[2] = {0};
  float vl[4] = {0};
  float vr[4] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_sgeevx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_SGEEVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sgeevx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  sgeev_call = (fb_sgeev_fn)vtable.ext_ops[FB_OP_SGEEV][FB_CONV_CBLAS];
  if (!sgeev_call) {
    fprintf(stderr, "[FAIL] SGEEV adapter slot missing after finalize\n");
    return 1;
  }

  if (sgeev_call(FB_LAYOUT_COL_MAJOR, 'V', 'V', 2, a, 2, wr, wi, vl, 2, vr,
                 2) != 0) {
    fprintf(stderr, "[FAIL] SGEEV adapter bridge from SGEEVX returned error\n");
    return 1;
  }

  if (g_sgeevx_bridge_calls != 1) {
    fprintf(stderr,
            "[FAIL] SGEEV<-SGEEVX adapter call count mismatch (calls=%d)\n",
            g_sgeevx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter bridge SGEEV <- SGEEVX applies default balance and sense params\n");
  return 0;
}

static int test_adapter_bridge_dgeev_from_dgeevx(void) {
  fb_backend_vtable_t vtable;
  fb_dgeev_fn dgeev_call = NULL;
  fb_status_t status;

  double a[4] = {0};
  double wr[2] = {0};
  double wi[2] = {0};
  double vl[4] = {0};
  double vr[4] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_dgeevx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_DGEEVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dgeevx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  dgeev_call = (fb_dgeev_fn)vtable.ext_ops[FB_OP_DGEEV][FB_CONV_CBLAS];
  if (!dgeev_call) {
    fprintf(stderr, "[FAIL] DGEEV adapter slot missing after finalize\n");
    return 1;
  }

  if (dgeev_call(FB_LAYOUT_COL_MAJOR, 'N', 'V', 2, a, 2, wr, wi, vl, 2, vr,
                 2) != 0) {
    fprintf(stderr, "[FAIL] DGEEV adapter bridge from DGEEVX returned error\n");
    return 1;
  }

  if (g_dgeevx_bridge_calls != 1) {
    fprintf(stderr,
            "[FAIL] DGEEV<-DGEEVX adapter call count mismatch (calls=%d)\n",
            g_dgeevx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter bridge DGEEV <- DGEEVX applies default balance and sense params\n");
  return 0;
}

static int test_adapter_bridge_cgeev_from_cgeevx(void) {
  fb_backend_vtable_t vtable;
  fb_cgeev_fn cgeev_call = NULL;
  fb_status_t status;

  fb_complex_float_t a[4] = {0};
  fb_complex_float_t w[2] = {0};
  fb_complex_float_t vl[4] = {0};
  fb_complex_float_t vr[4] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_cgeevx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_CGEEVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_cgeevx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  cgeev_call = (fb_cgeev_fn)vtable.ext_ops[FB_OP_CGEEV][FB_CONV_CBLAS];
  if (!cgeev_call) {
    fprintf(stderr, "[FAIL] CGEEV adapter slot missing after finalize\n");
    return 1;
  }

  if (cgeev_call(FB_LAYOUT_COL_MAJOR, 'V', 'N', 2, a, 2, w, vl, 2, vr, 2) !=
      0) {
    fprintf(stderr, "[FAIL] CGEEV adapter bridge from CGEEVX returned error\n");
    return 1;
  }

  if (g_cgeevx_bridge_calls != 1) {
    fprintf(stderr,
            "[FAIL] CGEEV<-CGEEVX adapter call count mismatch (calls=%d)\n",
            g_cgeevx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter bridge CGEEV <- CGEEVX applies default balance and sense params\n");
  return 0;
}

static int test_adapter_bridge_zgeev_from_zgeevx(void) {
  fb_backend_vtable_t vtable;
  fb_zgeev_fn zgeev_call = NULL;
  fb_status_t status;

  fb_complex_double_t a[4] = {0};
  fb_complex_double_t w[2] = {0};
  fb_complex_double_t vl[4] = {0};
  fb_complex_double_t vr[4] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_zgeevx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_ZGEEVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zgeevx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  zgeev_call = (fb_zgeev_fn)vtable.ext_ops[FB_OP_ZGEEV][FB_CONV_CBLAS];
  if (!zgeev_call) {
    fprintf(stderr, "[FAIL] ZGEEV adapter slot missing after finalize\n");
    return 1;
  }

  if (zgeev_call(FB_LAYOUT_COL_MAJOR, 'N', 'V', 2, a, 2, w, vl, 2, vr, 2) !=
      0) {
    fprintf(stderr, "[FAIL] ZGEEV adapter bridge from ZGEEVX returned error\n");
    return 1;
  }

  if (g_zgeevx_bridge_calls != 1) {
    fprintf(stderr,
            "[FAIL] ZGEEV<-ZGEEVX adapter call count mismatch (calls=%d)\n",
            g_zgeevx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter bridge ZGEEV <- ZGEEVX applies default balance and sense params\n");
  return 0;
}

static int test_adapter_does_not_override_existing_sgeev(void) {
  fb_backend_vtable_t vtable;
  fb_sgeev_fn sgeev_call = NULL;
  fb_status_t status;

  float a[4] = {0};
  float wr[2] = {0};
  float wi[2] = {0};
  float vl[4] = {0};
  float vr[4] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_sgeev_existing_calls = 0;
  g_sgeevx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_SGEEV][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sgeev_existing;
  vtable.ext_ops[FB_OP_SGEEVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sgeevx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  sgeev_call = (fb_sgeev_fn)vtable.ext_ops[FB_OP_SGEEV][FB_CONV_CBLAS];
  if (!sgeev_call) {
    fprintf(stderr, "[FAIL] SGEEV slot missing after finalize\n");
    return 1;
  }

  if (sgeev_call(FB_LAYOUT_COL_MAJOR, 'N', 'N', 2, a, 2, wr, wi, vl, 2, vr,
                 2) != 909) {
    fprintf(stderr,
            "[FAIL] adapter unexpectedly replaced existing SGEEV implementation\n");
    return 1;
  }

  if (g_sgeev_existing_calls != 1 || g_sgeevx_bridge_calls != 0) {
    fprintf(stderr,
            "[FAIL] SGEEV existing-preserve mismatch (existing=%d adapter=%d)\n",
            g_sgeev_existing_calls, g_sgeevx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter does not override existing SGEEV implementation\n");
  return 0;
}

static int test_adapter_bridge_sgesv_from_sgesvx(void) {
  fb_backend_vtable_t vtable;
  fb_sgesv_fn sgesv_call = NULL;
  fb_status_t status;

  float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float b[2] = {5.0f, 6.0f};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_sgesvx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_SGESVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sgesvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  sgesv_call = (fb_sgesv_fn)vtable.ext_ops[FB_OP_SGESV][FB_CONV_CBLAS];
  if (!sgesv_call) {
    fprintf(stderr, "[FAIL] SGESV adapter slot missing after finalize\n");
    return 1;
  }

  if (sgesv_call(FB_LAYOUT_COL_MAJOR, 2, 1, a, 2, ipiv, b, 2) != 0) {
    fprintf(stderr, "[FAIL] SGESV adapter bridge from SGESVX returned error\n");
    return 1;
  }

  if (g_sgesvx_bridge_calls != 1 || ipiv[0] != 7 || b[0] != 5.0f ||
      b[1] != 6.0f) {
    fprintf(stderr,
            "[FAIL] SGESV<-SGESVX bridge propagation mismatch (calls=%d ipiv0=%d b0=%g b1=%g)\n",
            g_sgesvx_bridge_calls, ipiv[0], b[0], b[1]);
    return 1;
  }

  printf("[PASS] adapter bridge SGESV <- SGESVX applies neutral expert params\n");
  return 0;
}

static int test_adapter_bridge_dgesv_from_dgesvx(void) {
  fb_backend_vtable_t vtable;
  fb_dgesv_fn dgesv_call = NULL;
  fb_status_t status;

  double a[4] = {1.0, 2.0, 3.0, 4.0};
  double b[2] = {5.0, 6.0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_dgesvx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_DGESVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dgesvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  dgesv_call = (fb_dgesv_fn)vtable.ext_ops[FB_OP_DGESV][FB_CONV_CBLAS];
  if (!dgesv_call) {
    fprintf(stderr, "[FAIL] DGESV adapter slot missing after finalize\n");
    return 1;
  }

  if (dgesv_call(FB_LAYOUT_COL_MAJOR, 2, 1, a, 2, ipiv, b, 2) != 0) {
    fprintf(stderr, "[FAIL] DGESV adapter bridge from DGESVX returned error\n");
    return 1;
  }

  if (g_dgesvx_bridge_calls != 1 || ipiv[0] != 9 || b[0] != 5.0 ||
      b[1] != 6.0) {
    fprintf(stderr,
            "[FAIL] DGESV<-DGESVX bridge propagation mismatch (calls=%d ipiv0=%d b0=%g b1=%g)\n",
            g_dgesvx_bridge_calls, ipiv[0], b[0], b[1]);
    return 1;
  }

  printf("[PASS] adapter bridge DGESV <- DGESVX applies neutral expert params\n");
  return 0;
}

static int test_adapter_bridge_cgesv_from_cgesvx(void) {
  fb_backend_vtable_t vtable;
  fb_cgesv_fn cgesv_call = NULL;
  fb_status_t status;

  fb_complex_float_t a[4] = {0};
  fb_complex_float_t b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_cgesvx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_CGESVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_cgesvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  cgesv_call = (fb_cgesv_fn)vtable.ext_ops[FB_OP_CGESV][FB_CONV_CBLAS];
  if (!cgesv_call) {
    fprintf(stderr, "[FAIL] CGESV adapter slot missing after finalize\n");
    return 1;
  }

  if (cgesv_call(FB_LAYOUT_COL_MAJOR, 2, 1, a, 2, ipiv, b, 2) != 0) {
    fprintf(stderr, "[FAIL] CGESV adapter bridge from CGESVX returned error\n");
    return 1;
  }

  if (g_cgesvx_bridge_calls != 1 || ipiv[0] != 11) {
    fprintf(stderr,
            "[FAIL] CGESV<-CGESVX bridge propagation mismatch (calls=%d ipiv0=%d)\n",
            g_cgesvx_bridge_calls, ipiv[0]);
    return 1;
  }

  printf("[PASS] adapter bridge CGESV <- CGESVX applies neutral expert params\n");
  return 0;
}

static int test_adapter_bridge_zgesv_from_zgesvx(void) {
  fb_backend_vtable_t vtable;
  fb_zgesv_fn zgesv_call = NULL;
  fb_status_t status;

  fb_complex_double_t a[4] = {0};
  fb_complex_double_t b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_zgesvx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_ZGESVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zgesvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  zgesv_call = (fb_zgesv_fn)vtable.ext_ops[FB_OP_ZGESV][FB_CONV_CBLAS];
  if (!zgesv_call) {
    fprintf(stderr, "[FAIL] ZGESV adapter slot missing after finalize\n");
    return 1;
  }

  if (zgesv_call(FB_LAYOUT_COL_MAJOR, 2, 1, a, 2, ipiv, b, 2) != 0) {
    fprintf(stderr, "[FAIL] ZGESV adapter bridge from ZGESVX returned error\n");
    return 1;
  }

  if (g_zgesvx_bridge_calls != 1 || ipiv[0] != 13) {
    fprintf(stderr,
            "[FAIL] ZGESV<-ZGESVX bridge propagation mismatch (calls=%d ipiv0=%d)\n",
            g_zgesvx_bridge_calls, ipiv[0]);
    return 1;
  }

  printf("[PASS] adapter bridge ZGESV <- ZGESVX applies neutral expert params\n");
  return 0;
}

static int test_adapter_bridge_sposv_from_sposvx(void) {
  fb_backend_vtable_t vtable;
  fb_sposv_fn sposv_call = NULL;
  fb_status_t status;

  float a[4] = {1.0f, 0.0f, 0.0f, 1.0f};
  float b[2] = {5.0f, 6.0f};

  memset(&vtable, 0, sizeof(vtable));
  g_sposvx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_SPOSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sposvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  sposv_call = (fb_sposv_fn)vtable.ext_ops[FB_OP_SPOSV][FB_CONV_CBLAS];
  if (!sposv_call) {
    fprintf(stderr, "[FAIL] SPOSV adapter slot missing after finalize\n");
    return 1;
  }

  if (sposv_call(FB_LAYOUT_COL_MAJOR, FB_UPPER, 2, 1, a, 2, b, 2) != 0) {
    fprintf(stderr, "[FAIL] SPOSV adapter bridge from SPOSVX returned error\n");
    return 1;
  }

  if (g_sposvx_bridge_calls != 1 || b[0] != 5.0f || b[1] != 6.0f) {
    fprintf(stderr,
            "[FAIL] SPOSV<-SPOSVX bridge propagation mismatch (calls=%d b0=%g b1=%g)\n",
            g_sposvx_bridge_calls, b[0], b[1]);
    return 1;
  }

  printf("[PASS] adapter bridge SPOSV <- SPOSVX applies neutral expert params\n");
  return 0;
}

static int test_adapter_bridge_dposv_from_dposvx(void) {
  fb_backend_vtable_t vtable;
  fb_dposv_fn dposv_call = NULL;
  fb_status_t status;

  double a[4] = {1.0, 0.0, 0.0, 1.0};
  double b[2] = {5.0, 6.0};

  memset(&vtable, 0, sizeof(vtable));
  g_dposvx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_DPOSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dposvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  dposv_call = (fb_dposv_fn)vtable.ext_ops[FB_OP_DPOSV][FB_CONV_CBLAS];
  if (!dposv_call) {
    fprintf(stderr, "[FAIL] DPOSV adapter slot missing after finalize\n");
    return 1;
  }

  if (dposv_call(FB_LAYOUT_COL_MAJOR, FB_UPPER, 2, 1, a, 2, b, 2) != 0) {
    fprintf(stderr, "[FAIL] DPOSV adapter bridge from DPOSVX returned error\n");
    return 1;
  }

  if (g_dposvx_bridge_calls != 1 || b[0] != 5.0 || b[1] != 6.0) {
    fprintf(stderr,
            "[FAIL] DPOSV<-DPOSVX bridge propagation mismatch (calls=%d b0=%g b1=%g)\n",
            g_dposvx_bridge_calls, b[0], b[1]);
    return 1;
  }

  printf("[PASS] adapter bridge DPOSV <- DPOSVX applies neutral expert params\n");
  return 0;
}

static int test_adapter_bridge_cposv_from_cposvx(void) {
  fb_backend_vtable_t vtable;
  fb_cposv_fn cposv_call = NULL;
  fb_status_t status;

  fb_complex_float_t a[4] = {0};
  fb_complex_float_t b[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_cposvx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_CPOSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_cposvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  cposv_call = (fb_cposv_fn)vtable.ext_ops[FB_OP_CPOSV][FB_CONV_CBLAS];
  if (!cposv_call) {
    fprintf(stderr, "[FAIL] CPOSV adapter slot missing after finalize\n");
    return 1;
  }

  if (cposv_call(FB_LAYOUT_COL_MAJOR, FB_UPPER, 2, 1, a, 2, b, 2) != 0) {
    fprintf(stderr, "[FAIL] CPOSV adapter bridge from CPOSVX returned error\n");
    return 1;
  }

  if (g_cposvx_bridge_calls != 1) {
    fprintf(stderr,
            "[FAIL] CPOSV<-CPOSVX adapter call count mismatch (calls=%d)\n",
            g_cposvx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter bridge CPOSV <- CPOSVX applies neutral expert params\n");
  return 0;
}

static int test_adapter_bridge_zposv_from_zposvx(void) {
  fb_backend_vtable_t vtable;
  fb_zposv_fn zposv_call = NULL;
  fb_status_t status;

  fb_complex_double_t a[4] = {0};
  fb_complex_double_t b[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_zposvx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_ZPOSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zposvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  zposv_call = (fb_zposv_fn)vtable.ext_ops[FB_OP_ZPOSV][FB_CONV_CBLAS];
  if (!zposv_call) {
    fprintf(stderr, "[FAIL] ZPOSV adapter slot missing after finalize\n");
    return 1;
  }

  if (zposv_call(FB_LAYOUT_COL_MAJOR, FB_UPPER, 2, 1, a, 2, b, 2) != 0) {
    fprintf(stderr, "[FAIL] ZPOSV adapter bridge from ZPOSVX returned error\n");
    return 1;
  }

  if (g_zposvx_bridge_calls != 1) {
    fprintf(stderr,
            "[FAIL] ZPOSV<-ZPOSVX adapter call count mismatch (calls=%d)\n",
            g_zposvx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter bridge ZPOSV <- ZPOSVX applies neutral expert params\n");
  return 0;
}

static int test_adapter_bridge_ssysv_from_ssysvx(void) {
  fb_backend_vtable_t vtable;
  fb_ssysv_fn ssysv_call = NULL;
  fb_status_t status;

  float a[4] = {1.0f, 0.0f, 0.0f, 1.0f};
  float b[2] = {5.0f, 6.0f};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_ssysvx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_SSYSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_ssysvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  ssysv_call = (fb_ssysv_fn)vtable.ext_ops[FB_OP_SSYSV][FB_CONV_CBLAS];
  if (!ssysv_call) {
    fprintf(stderr, "[FAIL] SSYSV adapter slot missing after finalize\n");
    return 1;
  }

  if (ssysv_call(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, a, 2, ipiv, b,
                 2) != 0) {
    fprintf(stderr, "[FAIL] SSYSV adapter bridge from SSYSVX returned error\n");
    return 1;
  }

  if (g_ssysvx_bridge_calls != 1 || ipiv[0] != 3) {
    fprintf(stderr,
            "[FAIL] SSYSV<-SSYSVX bridge propagation mismatch (calls=%d ipiv0=%d)\n",
            g_ssysvx_bridge_calls, ipiv[0]);
    return 1;
  }

  printf("[PASS] adapter bridge SSYSV <- SSYSVX applies neutral expert params\n");
  return 0;
}

static int test_adapter_bridge_dsysv_from_dsysvx(void) {
  fb_backend_vtable_t vtable;
  fb_dsysv_fn dsysv_call = NULL;
  fb_status_t status;

  double a[4] = {1.0, 0.0, 0.0, 1.0};
  double b[2] = {5.0, 6.0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_dsysvx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_DSYSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dsysvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  dsysv_call = (fb_dsysv_fn)vtable.ext_ops[FB_OP_DSYSV][FB_CONV_CBLAS];
  if (!dsysv_call) {
    fprintf(stderr, "[FAIL] DSYSV adapter slot missing after finalize\n");
    return 1;
  }

  if (dsysv_call(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, a, 2, ipiv, b,
                 2) != 0) {
    fprintf(stderr, "[FAIL] DSYSV adapter bridge from DSYSVX returned error\n");
    return 1;
  }

  if (g_dsysvx_bridge_calls != 1 || ipiv[0] != 5) {
    fprintf(stderr,
            "[FAIL] DSYSV<-DSYSVX bridge propagation mismatch (calls=%d ipiv0=%d)\n",
            g_dsysvx_bridge_calls, ipiv[0]);
    return 1;
  }

  printf("[PASS] adapter bridge DSYSV <- DSYSVX applies neutral expert params\n");
  return 0;
}

static int test_adapter_bridge_csysv_from_csysvx(void) {
  fb_backend_vtable_t vtable;
  fb_csysv_fn csysv_call = NULL;
  fb_status_t status;

  fb_complex_float_t a[4] = {0};
  fb_complex_float_t b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_csysvx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_CSYSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_csysvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  csysv_call = (fb_csysv_fn)vtable.ext_ops[FB_OP_CSYSV][FB_CONV_CBLAS];
  if (!csysv_call) {
    fprintf(stderr, "[FAIL] CSYSV adapter slot missing after finalize\n");
    return 1;
  }

  if (csysv_call(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, a, 2, ipiv, b,
                 2) != 0) {
    fprintf(stderr, "[FAIL] CSYSV adapter bridge from CSYSVX returned error\n");
    return 1;
  }

  if (g_csysvx_bridge_calls != 1 || ipiv[0] != 7) {
    fprintf(stderr,
            "[FAIL] CSYSV<-CSYSVX bridge propagation mismatch (calls=%d ipiv0=%d)\n",
            g_csysvx_bridge_calls, ipiv[0]);
    return 1;
  }

  printf("[PASS] adapter bridge CSYSV <- CSYSVX applies neutral expert params\n");
  return 0;
}

static int test_adapter_bridge_zsysv_from_zsysvx(void) {
  fb_backend_vtable_t vtable;
  fb_zsysv_fn zsysv_call = NULL;
  fb_status_t status;

  fb_complex_double_t a[4] = {0};
  fb_complex_double_t b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_zsysvx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_ZSYSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zsysvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  zsysv_call = (fb_zsysv_fn)vtable.ext_ops[FB_OP_ZSYSV][FB_CONV_CBLAS];
  if (!zsysv_call) {
    fprintf(stderr, "[FAIL] ZSYSV adapter slot missing after finalize\n");
    return 1;
  }

  if (zsysv_call(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, a, 2, ipiv, b,
                 2) != 0) {
    fprintf(stderr, "[FAIL] ZSYSV adapter bridge from ZSYSVX returned error\n");
    return 1;
  }

  if (g_zsysvx_bridge_calls != 1 || ipiv[0] != 9) {
    fprintf(stderr,
            "[FAIL] ZSYSV<-ZSYSVX bridge propagation mismatch (calls=%d ipiv0=%d)\n",
            g_zsysvx_bridge_calls, ipiv[0]);
    return 1;
  }

  printf("[PASS] adapter bridge ZSYSV <- ZSYSVX applies neutral expert params\n");
  return 0;
}

static int test_adapter_does_not_override_existing_cgesv(void) {
  fb_backend_vtable_t vtable;
  fb_cgesv_fn cgesv_call = NULL;
  fb_status_t status;

  fb_complex_float_t a[4] = {0};
  fb_complex_float_t b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_cgesv_existing_calls = 0;
  g_cgesvx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_CGESV][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_cgesv_existing;
  vtable.ext_ops[FB_OP_CGESVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_cgesvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  cgesv_call = (fb_cgesv_fn)vtable.ext_ops[FB_OP_CGESV][FB_CONV_CBLAS];
  if (!cgesv_call) {
    fprintf(stderr, "[FAIL] CGESV slot missing after finalize\n");
    return 1;
  }

  if (cgesv_call(FB_LAYOUT_COL_MAJOR, 2, 1, a, 2, ipiv, b, 2) != 919) {
    fprintf(stderr,
            "[FAIL] adapter unexpectedly replaced existing CGESV implementation\n");
    return 1;
  }

  if (g_cgesv_existing_calls != 1 || g_cgesvx_bridge_calls != 0) {
    fprintf(stderr,
            "[FAIL] CGESV existing-preserve mismatch (existing=%d adapter=%d)\n",
            g_cgesv_existing_calls, g_cgesvx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter does not override existing CGESV implementation\n");
  return 0;
}

static int test_adapter_does_not_override_existing_sgesv(void) {
  fb_backend_vtable_t vtable;
  fb_sgesv_fn sgesv_call = NULL;
  fb_status_t status;

  float a[4] = {0};
  float b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_sgesv_existing_calls = 0;
  g_sgesvx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_SGESV][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sgesv_existing;
  vtable.ext_ops[FB_OP_SGESVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sgesvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  sgesv_call = (fb_sgesv_fn)vtable.ext_ops[FB_OP_SGESV][FB_CONV_CBLAS];
  if (!sgesv_call) {
    fprintf(stderr, "[FAIL] SGESV slot missing after finalize\n");
    return 1;
  }

  if (sgesv_call(FB_LAYOUT_COL_MAJOR, 2, 1, a, 2, ipiv, b, 2) != 915) {
    fprintf(stderr,
            "[FAIL] adapter unexpectedly replaced existing SGESV implementation\n");
    return 1;
  }

  if (g_sgesv_existing_calls != 1 || g_sgesvx_bridge_calls != 0) {
    fprintf(stderr,
            "[FAIL] SGESV existing-preserve mismatch (existing=%d adapter=%d)\n",
            g_sgesv_existing_calls, g_sgesvx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter does not override existing SGESV implementation\n");
  return 0;
}

static int test_adapter_does_not_override_existing_dgesv(void) {
  fb_backend_vtable_t vtable;
  fb_dgesv_fn dgesv_call = NULL;
  fb_status_t status;

  double a[4] = {0};
  double b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_dgesv_existing_calls = 0;
  g_dgesvx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_DGESV][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dgesv_existing;
  vtable.ext_ops[FB_OP_DGESVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dgesvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  dgesv_call = (fb_dgesv_fn)vtable.ext_ops[FB_OP_DGESV][FB_CONV_CBLAS];
  if (!dgesv_call) {
    fprintf(stderr, "[FAIL] DGESV slot missing after finalize\n");
    return 1;
  }

  if (dgesv_call(FB_LAYOUT_COL_MAJOR, 2, 1, a, 2, ipiv, b, 2) != 925) {
    fprintf(stderr,
            "[FAIL] adapter unexpectedly replaced existing DGESV implementation\n");
    return 1;
  }

  if (g_dgesv_existing_calls != 1 || g_dgesvx_bridge_calls != 0) {
    fprintf(stderr,
            "[FAIL] DGESV existing-preserve mismatch (existing=%d adapter=%d)\n",
            g_dgesv_existing_calls, g_dgesvx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter does not override existing DGESV implementation\n");
  return 0;
}

static int test_adapter_does_not_override_existing_zgesv(void) {
  fb_backend_vtable_t vtable;
  fb_zgesv_fn zgesv_call = NULL;
  fb_status_t status;

  fb_complex_double_t a[4] = {0};
  fb_complex_double_t b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_zgesv_existing_calls = 0;
  g_zgesvx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_ZGESV][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zgesv_existing;
  vtable.ext_ops[FB_OP_ZGESVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zgesvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  zgesv_call = (fb_zgesv_fn)vtable.ext_ops[FB_OP_ZGESV][FB_CONV_CBLAS];
  if (!zgesv_call) {
    fprintf(stderr, "[FAIL] ZGESV slot missing after finalize\n");
    return 1;
  }

  if (zgesv_call(FB_LAYOUT_COL_MAJOR, 2, 1, a, 2, ipiv, b, 2) != 929) {
    fprintf(stderr,
            "[FAIL] adapter unexpectedly replaced existing ZGESV implementation\n");
    return 1;
  }

  if (g_zgesv_existing_calls != 1 || g_zgesvx_bridge_calls != 0) {
    fprintf(stderr,
            "[FAIL] ZGESV existing-preserve mismatch (existing=%d adapter=%d)\n",
            g_zgesv_existing_calls, g_zgesvx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter does not override existing ZGESV implementation\n");
  return 0;
}

static int test_adapter_does_not_override_existing_sposv(void) {
  fb_backend_vtable_t vtable;
  fb_sposv_fn sposv_call = NULL;
  fb_status_t status;

  float a[4] = {0};
  float b[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_sposv_existing_calls = 0;
  g_sposvx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_SPOSV][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sposv_existing;
  vtable.ext_ops[FB_OP_SPOSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sposvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  sposv_call = (fb_sposv_fn)vtable.ext_ops[FB_OP_SPOSV][FB_CONV_CBLAS];
  if (!sposv_call) {
    fprintf(stderr, "[FAIL] SPOSV slot missing after finalize\n");
    return 1;
  }

  if (sposv_call(FB_LAYOUT_COL_MAJOR, FB_UPPER, 2, 1, a, 2, b, 2) != 935) {
    fprintf(stderr,
            "[FAIL] adapter unexpectedly replaced existing SPOSV implementation\n");
    return 1;
  }

  if (g_sposv_existing_calls != 1 || g_sposvx_bridge_calls != 0) {
    fprintf(stderr,
            "[FAIL] SPOSV existing-preserve mismatch (existing=%d adapter=%d)\n",
            g_sposv_existing_calls, g_sposvx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter does not override existing SPOSV implementation\n");
  return 0;
}

static int test_adapter_does_not_override_existing_dposv(void) {
  fb_backend_vtable_t vtable;
  fb_dposv_fn dposv_call = NULL;
  fb_status_t status;

  double a[4] = {0};
  double b[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_dposv_existing_calls = 0;
  g_dposvx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_DPOSV][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dposv_existing;
  vtable.ext_ops[FB_OP_DPOSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dposvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  dposv_call = (fb_dposv_fn)vtable.ext_ops[FB_OP_DPOSV][FB_CONV_CBLAS];
  if (!dposv_call) {
    fprintf(stderr, "[FAIL] DPOSV slot missing after finalize\n");
    return 1;
  }

  if (dposv_call(FB_LAYOUT_COL_MAJOR, FB_UPPER, 2, 1, a, 2, b, 2) != 945) {
    fprintf(stderr,
            "[FAIL] adapter unexpectedly replaced existing DPOSV implementation\n");
    return 1;
  }

  if (g_dposv_existing_calls != 1 || g_dposvx_bridge_calls != 0) {
    fprintf(stderr,
            "[FAIL] DPOSV existing-preserve mismatch (existing=%d adapter=%d)\n",
            g_dposv_existing_calls, g_dposvx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter does not override existing DPOSV implementation\n");
  return 0;
}

static int test_adapter_does_not_override_existing_cposv(void) {
  fb_backend_vtable_t vtable;
  fb_cposv_fn cposv_call = NULL;
  fb_status_t status;

  fb_complex_float_t a[4] = {0};
  fb_complex_float_t b[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_cposv_existing_calls = 0;
  g_cposvx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_CPOSV][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_cposv_existing;
  vtable.ext_ops[FB_OP_CPOSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_cposvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  cposv_call = (fb_cposv_fn)vtable.ext_ops[FB_OP_CPOSV][FB_CONV_CBLAS];
  if (!cposv_call) {
    fprintf(stderr, "[FAIL] CPOSV slot missing after finalize\n");
    return 1;
  }

  if (cposv_call(FB_LAYOUT_COL_MAJOR, FB_UPPER, 2, 1, a, 2, b, 2) != 955) {
    fprintf(stderr,
            "[FAIL] adapter unexpectedly replaced existing CPOSV implementation\n");
    return 1;
  }

  if (g_cposv_existing_calls != 1 || g_cposvx_bridge_calls != 0) {
    fprintf(stderr,
            "[FAIL] CPOSV existing-preserve mismatch (existing=%d adapter=%d)\n",
            g_cposv_existing_calls, g_cposvx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter does not override existing CPOSV implementation\n");
  return 0;
}

static int test_adapter_does_not_override_existing_zposv(void) {
  fb_backend_vtable_t vtable;
  fb_zposv_fn zposv_call = NULL;
  fb_status_t status;

  fb_complex_double_t a[4] = {0};
  fb_complex_double_t b[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_zposv_existing_calls = 0;
  g_zposvx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_ZPOSV][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zposv_existing;
  vtable.ext_ops[FB_OP_ZPOSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zposvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  zposv_call = (fb_zposv_fn)vtable.ext_ops[FB_OP_ZPOSV][FB_CONV_CBLAS];
  if (!zposv_call) {
    fprintf(stderr, "[FAIL] ZPOSV slot missing after finalize\n");
    return 1;
  }

  if (zposv_call(FB_LAYOUT_COL_MAJOR, FB_UPPER, 2, 1, a, 2, b, 2) != 965) {
    fprintf(stderr,
            "[FAIL] adapter unexpectedly replaced existing ZPOSV implementation\n");
    return 1;
  }

  if (g_zposv_existing_calls != 1 || g_zposvx_bridge_calls != 0) {
    fprintf(stderr,
            "[FAIL] ZPOSV existing-preserve mismatch (existing=%d adapter=%d)\n",
            g_zposv_existing_calls, g_zposvx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter does not override existing ZPOSV implementation\n");
  return 0;
}

static int test_adapter_does_not_override_existing_ssysv(void) {
  fb_backend_vtable_t vtable;
  fb_ssysv_fn ssysv_call = NULL;
  fb_status_t status;

  float a[4] = {0};
  float b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_ssysv_existing_calls = 0;
  g_ssysvx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_SSYSV][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_ssysv_existing;
  vtable.ext_ops[FB_OP_SSYSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_ssysvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  ssysv_call = (fb_ssysv_fn)vtable.ext_ops[FB_OP_SSYSV][FB_CONV_CBLAS];
  if (!ssysv_call) {
    fprintf(stderr, "[FAIL] SSYSV slot missing after finalize\n");
    return 1;
  }

  if (ssysv_call(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, a, 2, ipiv, b,
                 2) != 975) {
    fprintf(stderr,
            "[FAIL] adapter unexpectedly replaced existing SSYSV implementation\n");
    return 1;
  }

  if (g_ssysv_existing_calls != 1 || g_ssysvx_bridge_calls != 0) {
    fprintf(stderr,
            "[FAIL] SSYSV existing-preserve mismatch (existing=%d adapter=%d)\n",
            g_ssysv_existing_calls, g_ssysvx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter does not override existing SSYSV implementation\n");
  return 0;
}

static int test_adapter_does_not_override_existing_dsysv(void) {
  fb_backend_vtable_t vtable;
  fb_dsysv_fn dsysv_call = NULL;
  fb_status_t status;

  double a[4] = {0};
  double b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_dsysv_existing_calls = 0;
  g_dsysvx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_DSYSV][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dsysv_existing;
  vtable.ext_ops[FB_OP_DSYSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dsysvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  dsysv_call = (fb_dsysv_fn)vtable.ext_ops[FB_OP_DSYSV][FB_CONV_CBLAS];
  if (!dsysv_call) {
    fprintf(stderr, "[FAIL] DSYSV slot missing after finalize\n");
    return 1;
  }

  if (dsysv_call(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, a, 2, ipiv, b,
                 2) != 985) {
    fprintf(stderr,
            "[FAIL] adapter unexpectedly replaced existing DSYSV implementation\n");
    return 1;
  }

  if (g_dsysv_existing_calls != 1 || g_dsysvx_bridge_calls != 0) {
    fprintf(stderr,
            "[FAIL] DSYSV existing-preserve mismatch (existing=%d adapter=%d)\n",
            g_dsysv_existing_calls, g_dsysvx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter does not override existing DSYSV implementation\n");
  return 0;
}

static int test_adapter_does_not_override_existing_csysv(void) {
  fb_backend_vtable_t vtable;
  fb_csysv_fn csysv_call = NULL;
  fb_status_t status;

  fb_complex_float_t a[4] = {0};
  fb_complex_float_t b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_csysv_existing_calls = 0;
  g_csysvx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_CSYSV][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_csysv_existing;
  vtable.ext_ops[FB_OP_CSYSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_csysvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  csysv_call = (fb_csysv_fn)vtable.ext_ops[FB_OP_CSYSV][FB_CONV_CBLAS];
  if (!csysv_call) {
    fprintf(stderr, "[FAIL] CSYSV slot missing after finalize\n");
    return 1;
  }

  if (csysv_call(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, a, 2, ipiv, b,
                 2) != 995) {
    fprintf(stderr,
            "[FAIL] adapter unexpectedly replaced existing CSYSV implementation\n");
    return 1;
  }

  if (g_csysv_existing_calls != 1 || g_csysvx_bridge_calls != 0) {
    fprintf(stderr,
            "[FAIL] CSYSV existing-preserve mismatch (existing=%d adapter=%d)\n",
            g_csysv_existing_calls, g_csysvx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter does not override existing CSYSV implementation\n");
  return 0;
}

static int test_adapter_does_not_override_existing_zsysv(void) {
  fb_backend_vtable_t vtable;
  fb_zsysv_fn zsysv_call = NULL;
  fb_status_t status;

  fb_complex_double_t a[4] = {0};
  fb_complex_double_t b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_zsysv_existing_calls = 0;
  g_zsysvx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_ZSYSV][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zsysv_existing;
  vtable.ext_ops[FB_OP_ZSYSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zsysvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  zsysv_call = (fb_zsysv_fn)vtable.ext_ops[FB_OP_ZSYSV][FB_CONV_CBLAS];
  if (!zsysv_call) {
    fprintf(stderr, "[FAIL] ZSYSV slot missing after finalize\n");
    return 1;
  }

  if (zsysv_call(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, a, 2, ipiv, b,
                 2) != 1005) {
    fprintf(stderr,
            "[FAIL] adapter unexpectedly replaced existing ZSYSV implementation\n");
    return 1;
  }

  if (g_zsysv_existing_calls != 1 || g_zsysvx_bridge_calls != 0) {
    fprintf(stderr,
            "[FAIL] ZSYSV existing-preserve mismatch (existing=%d adapter=%d)\n",
            g_zsysv_existing_calls, g_zsysvx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter does not override existing ZSYSV implementation\n");
  return 0;
}

static int test_adapter_bridge_sstev_from_sstevr(void) {
  fb_backend_vtable_t vtable;
  fb_sstev_cblas_fn sstev_call = NULL;
  fb_status_t status;

  float d[2] = {0};
  float e[2] = {0};
  float z[4] = {0};
  float work[4] = {0};
  int info = -1;

  memset(&vtable, 0, sizeof(vtable));
  g_sstevr_bridge_calls = 0;

  vtable.ext_ops[FB_OP_SSTEVR][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sstevr_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  sstev_call = (fb_sstev_cblas_fn)vtable.ext_ops[FB_OP_SSTEV][FB_CONV_CBLAS];
  if (!sstev_call) {
    fprintf(stderr, "[FAIL] SSTEV adapter slot missing after finalize\n");
    return 1;
  }

  sstev_call(FB_LAYOUT_COL_MAJOR, 'V', 2, d, e, z, 2, work, &info);
  if (info != 0) {
    fprintf(stderr, "[FAIL] SSTEV adapter bridge returned info=%d\n", info);
    return 1;
  }

  if (g_sstevr_bridge_calls != 1) {
    fprintf(stderr,
            "[FAIL] SSTEV<-SSTEVR adapter call count mismatch (calls=%d)\n",
            g_sstevr_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter bridge SSTEV <- SSTEVR applies default range params\n");
  return 0;
}

static int test_adapter_bridge_dstev_from_dstevx(void) {
  fb_backend_vtable_t vtable;
  fb_dstev_cblas_fn dstev_call = NULL;
  fb_status_t status;

  double d[2] = {0};
  double e[2] = {0};
  double z[4] = {0};
  double work[4] = {0};
  int info = -1;

  memset(&vtable, 0, sizeof(vtable));
  g_dstevx_bridge_calls = 0;

  /* No DSTEVR donor: adapter should fallback to DSTEVX source. */
  vtable.ext_ops[FB_OP_DSTEVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dstevx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  dstev_call = (fb_dstev_cblas_fn)vtable.ext_ops[FB_OP_DSTEV][FB_CONV_CBLAS];
  if (!dstev_call) {
    fprintf(stderr, "[FAIL] DSTEV adapter slot missing after finalize\n");
    return 1;
  }

  dstev_call(FB_LAYOUT_COL_MAJOR, 'V', 2, d, e, z, 2, work, &info);
  if (info != 0) {
    fprintf(stderr, "[FAIL] DSTEV adapter bridge returned info=%d\n", info);
    return 1;
  }

  if (g_dstevx_bridge_calls != 1) {
    fprintf(stderr,
            "[FAIL] DSTEV<-DSTEVX adapter call count mismatch (calls=%d)\n",
            g_dstevx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter bridge DSTEV <- DSTEVX applies default range params\n");
  return 0;
}

static int test_adapter_bridge_sstev_from_sstevx(void) {
  fb_backend_vtable_t vtable;
  fb_sstev_cblas_fn sstev_call = NULL;
  fb_status_t status;

  float d[2] = {0};
  float e[2] = {0};
  float z[4] = {0};
  float work[4] = {0};
  int info = -1;

  memset(&vtable, 0, sizeof(vtable));
  g_sstevx_bridge_calls = 0;

  /* No SSTEVR donor: adapter should fallback to SSTEVX source. */
  vtable.ext_ops[FB_OP_SSTEVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sstevx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  sstev_call = (fb_sstev_cblas_fn)vtable.ext_ops[FB_OP_SSTEV][FB_CONV_CBLAS];
  if (!sstev_call) {
    fprintf(stderr, "[FAIL] SSTEV adapter slot missing after finalize\n");
    return 1;
  }

  sstev_call(FB_LAYOUT_COL_MAJOR, 'V', 2, d, e, z, 2, work, &info);
  if (info != 0) {
    fprintf(stderr, "[FAIL] SSTEV adapter bridge returned info=%d\n", info);
    return 1;
  }

  if (g_sstevx_bridge_calls != 1) {
    fprintf(stderr,
            "[FAIL] SSTEV<-SSTEVX adapter call count mismatch (calls=%d)\n",
            g_sstevx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter bridge SSTEV <- SSTEVX applies default range params\n");
  return 0;
}

static int test_adapter_bridge_dstev_from_dstevr(void) {
  fb_backend_vtable_t vtable;
  fb_dstev_cblas_fn dstev_call = NULL;
  fb_status_t status;

  double d[2] = {0};
  double e[2] = {0};
  double z[4] = {0};
  double work[4] = {0};
  int info = -1;

  memset(&vtable, 0, sizeof(vtable));
  g_dstevr_bridge_calls = 0;

  vtable.ext_ops[FB_OP_DSTEVR][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dstevr_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  dstev_call = (fb_dstev_cblas_fn)vtable.ext_ops[FB_OP_DSTEV][FB_CONV_CBLAS];
  if (!dstev_call) {
    fprintf(stderr, "[FAIL] DSTEV adapter slot missing after finalize\n");
    return 1;
  }

  dstev_call(FB_LAYOUT_COL_MAJOR, 'V', 2, d, e, z, 2, work, &info);
  if (info != 0) {
    fprintf(stderr, "[FAIL] DSTEV adapter bridge returned info=%d\n", info);
    return 1;
  }

  if (g_dstevr_bridge_calls != 1) {
    fprintf(stderr,
            "[FAIL] DSTEV<-DSTEVR adapter call count mismatch (calls=%d)\n",
            g_dstevr_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter bridge DSTEV <- DSTEVR applies default range params\n");
  return 0;
}

static int test_adapter_bridge_sstev_from_sstevd(void) {
  fb_backend_vtable_t vtable;
  fb_sstev_cblas_fn sstev_call = NULL;
  fb_status_t status;

  float d[2] = {0};
  float e[2] = {0};
  float z[4] = {0};
  float work[4] = {0};
  int info = -1;

  memset(&vtable, 0, sizeof(vtable));
  g_sstevd_bridge_calls = 0;

  /* No STEVR/STEVX donor: fallback should use STEVD. */
  vtable.ext_ops[FB_OP_SSTEVD][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sstevd_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  sstev_call = (fb_sstev_cblas_fn)vtable.ext_ops[FB_OP_SSTEV][FB_CONV_CBLAS];
  if (!sstev_call) {
    fprintf(stderr, "[FAIL] SSTEV adapter slot missing after finalize\n");
    return 1;
  }

  sstev_call(FB_LAYOUT_COL_MAJOR, 'V', 2, d, e, z, 2, work, &info);
  if (info != 0) {
    fprintf(stderr, "[FAIL] SSTEV adapter from SSTEVD returned info=%d\n", info);
    return 1;
  }

  if (g_sstevd_bridge_calls != 1) {
    fprintf(stderr,
            "[FAIL] SSTEV<-SSTEVD adapter call count mismatch (calls=%d)\n",
            g_sstevd_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter bridge SSTEV <- SSTEVD maps jobz to compz safely\n");
  return 0;
}

static int test_adapter_bridge_dstev_from_dstevd(void) {
  fb_backend_vtable_t vtable;
  fb_dstev_cblas_fn dstev_call = NULL;
  fb_status_t status;

  double d[2] = {0};
  double e[2] = {0};
  double z[4] = {0};
  double work[4] = {0};
  int info = -1;

  memset(&vtable, 0, sizeof(vtable));
  g_dstevd_bridge_calls = 0;

  /* No STEVR/STEVX donor: fallback should use STEVD. */
  vtable.ext_ops[FB_OP_DSTEVD][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dstevd_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  dstev_call = (fb_dstev_cblas_fn)vtable.ext_ops[FB_OP_DSTEV][FB_CONV_CBLAS];
  if (!dstev_call) {
    fprintf(stderr, "[FAIL] DSTEV adapter slot missing after finalize\n");
    return 1;
  }

  dstev_call(FB_LAYOUT_COL_MAJOR, 'V', 2, d, e, z, 2, work, &info);
  if (info != 0) {
    fprintf(stderr, "[FAIL] DSTEV adapter from DSTEVD returned info=%d\n", info);
    return 1;
  }

  if (g_dstevd_bridge_calls != 1) {
    fprintf(stderr,
            "[FAIL] DSTEV<-DSTEVD adapter call count mismatch (calls=%d)\n",
            g_dstevd_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter bridge DSTEV <- DSTEVD maps jobz to compz safely\n");
  return 0;
}

static int test_adapter_precedence_ssyev_prefers_ssyevr(void) {
  fb_backend_vtable_t vtable;
  fb_ssyev_cblas_fn ssyev_call = NULL;
  fb_status_t status;

  float a[4] = {0};
  float w[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_syevr_bridge_calls = 0;
  g_ssyevx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_SSYEVR][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_ssyevr_base;
  vtable.ext_ops[FB_OP_SSYEVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_ssyevx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  ssyev_call = (fb_ssyev_cblas_fn)vtable.ext_ops[FB_OP_SSYEV][FB_CONV_CBLAS];
  if (!ssyev_call) {
    fprintf(stderr, "[FAIL] SSYEV adapter slot missing after finalize\n");
    return 1;
  }

  if (ssyev_call(FB_LAYOUT_COL_MAJOR, 'V', FB_UPPER, 2, a, 2, w) != 0) {
    fprintf(stderr,
            "[FAIL] SSYEV precedence bridge returned non-zero status\n");
    return 1;
  }

  if (g_syevr_bridge_calls != 1 || g_ssyevx_bridge_calls != 0) {
    fprintf(stderr,
            "[FAIL] SSYEV adapter precedence mismatch (syevr=%d syevx=%d)\n",
            g_syevr_bridge_calls, g_ssyevx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter precedence SSYEV prefers SSYEVR over SSYEVX\n");
  return 0;
}

static int test_adapter_precedence_sstev_prefers_sstevr(void) {
  fb_backend_vtable_t vtable;
  fb_sstev_cblas_fn sstev_call = NULL;
  fb_status_t status;

  float d[2] = {0};
  float e[2] = {0};
  float z[4] = {0};
  float work[4] = {0};
  int info = -1;

  memset(&vtable, 0, sizeof(vtable));
  g_sstevr_bridge_calls = 0;
  g_sstevx_bridge_calls = 0;
  g_sstevd_bridge_calls = 0;

  vtable.ext_ops[FB_OP_SSTEVR][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sstevr_base;
  vtable.ext_ops[FB_OP_SSTEVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sstevx_base;
  vtable.ext_ops[FB_OP_SSTEVD][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sstevd_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  sstev_call = (fb_sstev_cblas_fn)vtable.ext_ops[FB_OP_SSTEV][FB_CONV_CBLAS];
  if (!sstev_call) {
    fprintf(stderr, "[FAIL] SSTEV adapter slot missing after finalize\n");
    return 1;
  }

  sstev_call(FB_LAYOUT_COL_MAJOR, 'V', 2, d, e, z, 2, work, &info);
  if (info != 0) {
    fprintf(stderr, "[FAIL] SSTEV precedence bridge returned info=%d\n", info);
    return 1;
  }

  if (g_sstevr_bridge_calls != 1 || g_sstevx_bridge_calls != 0 ||
      g_sstevd_bridge_calls != 0) {
    fprintf(stderr,
            "[FAIL] SSTEV adapter precedence mismatch (stevr=%d stevx=%d stevd=%d)\n",
            g_sstevr_bridge_calls, g_sstevx_bridge_calls,
            g_sstevd_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter precedence SSTEV prefers STEVR over STEVX/STEVD\n");
  return 0;
}

static int test_adapter_precedence_dsyev_prefers_dsyevr(void) {
  fb_backend_vtable_t vtable;
  fb_dsyev_fn dsyev_call = NULL;
  fb_status_t status;

  double a[4] = {0};
  double w[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_dsyevr_bridge_calls = 0;
  g_dsyevx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_DSYEVR][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dsyevr_base;
  vtable.ext_ops[FB_OP_DSYEVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dsyevx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  dsyev_call = (fb_dsyev_fn)vtable.ext_ops[FB_OP_DSYEV][FB_CONV_CBLAS];
  if (!dsyev_call) {
    fprintf(stderr, "[FAIL] DSYEV adapter slot missing after finalize\n");
    return 1;
  }

  if (dsyev_call(FB_LAYOUT_COL_MAJOR, 'V', FB_UPPER, 2, a, 2, w) != 0) {
    fprintf(stderr, "[FAIL] DSYEV precedence bridge returned non-zero status\n");
    return 1;
  }

  if (g_dsyevr_bridge_calls != 1 || g_dsyevx_bridge_calls != 0) {
    fprintf(stderr,
            "[FAIL] DSYEV adapter precedence mismatch (dsyevr=%d dsyevx=%d)\n",
            g_dsyevr_bridge_calls, g_dsyevx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter precedence DSYEV prefers DSYEVR over DSYEVX\n");
  return 0;
}

static int test_adapter_precedence_dstev_prefers_dstevr(void) {
  fb_backend_vtable_t vtable;
  fb_dstev_cblas_fn dstev_call = NULL;
  fb_status_t status;

  double d[2] = {0};
  double e[2] = {0};
  double z[4] = {0};
  double work[4] = {0};
  int info = -1;

  memset(&vtable, 0, sizeof(vtable));
  g_dstevr_bridge_calls = 0;
  g_dstevx_bridge_calls = 0;
  g_dstevd_bridge_calls = 0;

  vtable.ext_ops[FB_OP_DSTEVR][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dstevr_base;
  vtable.ext_ops[FB_OP_DSTEVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dstevx_base;
  vtable.ext_ops[FB_OP_DSTEVD][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dstevd_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  dstev_call = (fb_dstev_cblas_fn)vtable.ext_ops[FB_OP_DSTEV][FB_CONV_CBLAS];
  if (!dstev_call) {
    fprintf(stderr, "[FAIL] DSTEV adapter slot missing after finalize\n");
    return 1;
  }

  dstev_call(FB_LAYOUT_COL_MAJOR, 'V', 2, d, e, z, 2, work, &info);
  if (info != 0) {
    fprintf(stderr, "[FAIL] DSTEV precedence bridge returned info=%d\n", info);
    return 1;
  }

  if (g_dstevr_bridge_calls != 1 || g_dstevx_bridge_calls != 0 ||
      g_dstevd_bridge_calls != 0) {
    fprintf(stderr,
            "[FAIL] DSTEV adapter precedence mismatch (stevr=%d stevx=%d stevd=%d)\n",
            g_dstevr_bridge_calls, g_dstevx_bridge_calls,
            g_dstevd_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter precedence DSTEV prefers STEVR over STEVX/STEVD\n");
  return 0;
}

static int test_adapter_does_not_override_existing_ssyev(void) {
  fb_backend_vtable_t vtable;
  fb_ssyev_cblas_fn ssyev_call = NULL;
  fb_status_t status;

  float a[4] = {0};
  float w[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_ssyev_existing_calls = 0;
  g_syevr_bridge_calls = 0;

  vtable.ext_ops[FB_OP_SSYEV][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_ssyev_existing;
  vtable.ext_ops[FB_OP_SSYEVR][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_ssyevr_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  ssyev_call = (fb_ssyev_cblas_fn)vtable.ext_ops[FB_OP_SSYEV][FB_CONV_CBLAS];
  if (!ssyev_call) {
    fprintf(stderr, "[FAIL] SSYEV slot missing after finalize\n");
    return 1;
  }

  if (ssyev_call(FB_LAYOUT_COL_MAJOR, 'V', FB_UPPER, 2, a, 2, w) != 911) {
    fprintf(stderr,
            "[FAIL] adapter unexpectedly replaced existing SSYEV implementation\n");
    return 1;
  }

  if (g_ssyev_existing_calls != 1 || g_syevr_bridge_calls != 0) {
    fprintf(stderr,
            "[FAIL] SSYEV existing-preserve mismatch (existing=%d adapter=%d)\n",
            g_ssyev_existing_calls, g_syevr_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter does not override existing SSYEV implementation\n");
  return 0;
}

static int test_adapter_does_not_override_existing_sstev(void) {
  fb_backend_vtable_t vtable;
  fb_sstev_cblas_fn sstev_call = NULL;
  fb_status_t status;

  float d[2] = {0};
  float e[2] = {0};
  float z[4] = {0};
  float work[4] = {0};
  int info = -1;

  memset(&vtable, 0, sizeof(vtable));
  g_sstev_existing_calls = 0;
  g_sstevr_bridge_calls = 0;

  vtable.ext_ops[FB_OP_SSTEV][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sstev_existing;
  vtable.ext_ops[FB_OP_SSTEVR][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sstevr_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  sstev_call = (fb_sstev_cblas_fn)vtable.ext_ops[FB_OP_SSTEV][FB_CONV_CBLAS];
  if (!sstev_call) {
    fprintf(stderr, "[FAIL] SSTEV slot missing after finalize\n");
    return 1;
  }

  sstev_call(FB_LAYOUT_COL_MAJOR, 'V', 2, d, e, z, 2, work, &info);
  if (info != 777) {
    fprintf(stderr,
            "[FAIL] adapter unexpectedly replaced existing SSTEV implementation\n");
    return 1;
  }

  if (g_sstev_existing_calls != 1 || g_sstevr_bridge_calls != 0) {
    fprintf(stderr,
            "[FAIL] SSTEV existing-preserve mismatch (existing=%d adapter=%d)\n",
            g_sstev_existing_calls, g_sstevr_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter does not override existing SSTEV implementation\n");
  return 0;
}

static int test_adapter_finalize_idempotent_ssyev(void) {
  fb_backend_vtable_t vtable;
  fb_ssyev_cblas_fn ssyev_call = NULL;
  fb_generic_fn first_ptr = NULL;
  fb_status_t status;

  float a[4] = {0};
  float w[2] = {0};

  memset(&vtable, 0, sizeof(vtable));

  vtable.ext_ops[FB_OP_SSYEVR][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_ssyevr_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] first finalize status=%d\n", status);
    return 1;
  }

  first_ptr = vtable.ext_ops[FB_OP_SSYEV][FB_CONV_CBLAS];
  if (!first_ptr) {
    fprintf(stderr, "[FAIL] SSYEV slot missing after first finalize\n");
    return 1;
  }

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] second finalize status=%d\n", status);
    return 1;
  }

  if (vtable.ext_ops[FB_OP_SSYEV][FB_CONV_CBLAS] != first_ptr) {
    fprintf(stderr,
            "[FAIL] SSYEV adapter pointer changed after second finalize\n");
    return 1;
  }

  ssyev_call = (fb_ssyev_cblas_fn)vtable.ext_ops[FB_OP_SSYEV][FB_CONV_CBLAS];
  if (!ssyev_call) {
    fprintf(stderr, "[FAIL] SSYEV call pointer missing after second finalize\n");
    return 1;
  }

  if (ssyev_call(FB_LAYOUT_COL_MAJOR, 'V', FB_UPPER, 2, a, 2, w) != 0) {
    fprintf(stderr, "[FAIL] SSYEV call failed after repeated finalize\n");
    return 1;
  }

  printf("[PASS] adapter finalize is idempotent for SSYEV bridge\n");
  return 0;
}

static int test_adapter_finalize_idempotent_sgesv(void) {
  fb_backend_vtable_t vtable;
  fb_sgesv_fn sgesv_call = NULL;
  fb_generic_fn first_ptr = NULL;
  fb_status_t status;

  float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float b[2] = {5.0f, 6.0f};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));

  vtable.ext_ops[FB_OP_SGESVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sgesvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] first finalize status=%d\n", status);
    return 1;
  }

  first_ptr = vtable.ext_ops[FB_OP_SGESV][FB_CONV_CBLAS];
  if (!first_ptr) {
    fprintf(stderr, "[FAIL] SGESV slot missing after first finalize\n");
    return 1;
  }

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] second finalize status=%d\n", status);
    return 1;
  }

  if (vtable.ext_ops[FB_OP_SGESV][FB_CONV_CBLAS] != first_ptr) {
    fprintf(stderr,
            "[FAIL] SGESV adapter pointer changed after second finalize\n");
    return 1;
  }

  sgesv_call = (fb_sgesv_fn)vtable.ext_ops[FB_OP_SGESV][FB_CONV_CBLAS];
  if (!sgesv_call) {
    fprintf(stderr, "[FAIL] SGESV call pointer missing after second finalize\n");
    return 1;
  }

  if (sgesv_call(FB_LAYOUT_COL_MAJOR, 2, 1, a, 2, ipiv, b, 2) != 0) {
    fprintf(stderr, "[FAIL] SGESV call failed after repeated finalize\n");
    return 1;
  }

  printf("[PASS] adapter finalize is idempotent for SGESV bridge\n");
  return 0;
}

static int test_adapter_finalize_idempotent_dgesv(void) {
  fb_backend_vtable_t vtable;
  fb_dgesv_fn dgesv_call = NULL;
  fb_generic_fn first_ptr = NULL;
  fb_status_t status;

  double a[4] = {1.0, 2.0, 3.0, 4.0};
  double b[2] = {5.0, 6.0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));

  vtable.ext_ops[FB_OP_DGESVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dgesvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] first finalize status=%d\n", status);
    return 1;
  }

  first_ptr = vtable.ext_ops[FB_OP_DGESV][FB_CONV_CBLAS];
  if (!first_ptr) {
    fprintf(stderr, "[FAIL] DGESV slot missing after first finalize\n");
    return 1;
  }

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] second finalize status=%d\n", status);
    return 1;
  }

  if (vtable.ext_ops[FB_OP_DGESV][FB_CONV_CBLAS] != first_ptr) {
    fprintf(stderr,
            "[FAIL] DGESV adapter pointer changed after second finalize\n");
    return 1;
  }

  dgesv_call = (fb_dgesv_fn)vtable.ext_ops[FB_OP_DGESV][FB_CONV_CBLAS];
  if (!dgesv_call) {
    fprintf(stderr, "[FAIL] DGESV call pointer missing after second finalize\n");
    return 1;
  }

  if (dgesv_call(FB_LAYOUT_COL_MAJOR, 2, 1, a, 2, ipiv, b, 2) != 0) {
    fprintf(stderr, "[FAIL] DGESV call failed after repeated finalize\n");
    return 1;
  }

  printf("[PASS] adapter finalize is idempotent for DGESV bridge\n");
  return 0;
}

static int test_adapter_finalize_idempotent_cgesv(void) {
  fb_backend_vtable_t vtable;
  fb_cgesv_fn cgesv_call = NULL;
  fb_generic_fn first_ptr = NULL;
  fb_status_t status;

  fb_complex_float_t a[4] = {0};
  fb_complex_float_t b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));

  vtable.ext_ops[FB_OP_CGESVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_cgesvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] first finalize status=%d\n", status);
    return 1;
  }

  first_ptr = vtable.ext_ops[FB_OP_CGESV][FB_CONV_CBLAS];
  if (!first_ptr) {
    fprintf(stderr, "[FAIL] CGESV slot missing after first finalize\n");
    return 1;
  }

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] second finalize status=%d\n", status);
    return 1;
  }

  if (vtable.ext_ops[FB_OP_CGESV][FB_CONV_CBLAS] != first_ptr) {
    fprintf(stderr,
            "[FAIL] CGESV adapter pointer changed after second finalize\n");
    return 1;
  }

  cgesv_call = (fb_cgesv_fn)vtable.ext_ops[FB_OP_CGESV][FB_CONV_CBLAS];
  if (!cgesv_call) {
    fprintf(stderr, "[FAIL] CGESV call pointer missing after second finalize\n");
    return 1;
  }

  if (cgesv_call(FB_LAYOUT_COL_MAJOR, 2, 1, a, 2, ipiv, b, 2) != 0) {
    fprintf(stderr, "[FAIL] CGESV call failed after repeated finalize\n");
    return 1;
  }

  printf("[PASS] adapter finalize is idempotent for CGESV bridge\n");
  return 0;
}

static int test_adapter_finalize_idempotent_zgesv(void) {
  fb_backend_vtable_t vtable;
  fb_zgesv_fn zgesv_call = NULL;
  fb_generic_fn first_ptr = NULL;
  fb_status_t status;

  fb_complex_double_t a[4] = {0};
  fb_complex_double_t b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));

  vtable.ext_ops[FB_OP_ZGESVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zgesvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] first finalize status=%d\n", status);
    return 1;
  }

  first_ptr = vtable.ext_ops[FB_OP_ZGESV][FB_CONV_CBLAS];
  if (!first_ptr) {
    fprintf(stderr, "[FAIL] ZGESV slot missing after first finalize\n");
    return 1;
  }

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] second finalize status=%d\n", status);
    return 1;
  }

  if (vtable.ext_ops[FB_OP_ZGESV][FB_CONV_CBLAS] != first_ptr) {
    fprintf(stderr,
            "[FAIL] ZGESV adapter pointer changed after second finalize\n");
    return 1;
  }

  zgesv_call = (fb_zgesv_fn)vtable.ext_ops[FB_OP_ZGESV][FB_CONV_CBLAS];
  if (!zgesv_call) {
    fprintf(stderr, "[FAIL] ZGESV call pointer missing after second finalize\n");
    return 1;
  }

  if (zgesv_call(FB_LAYOUT_COL_MAJOR, 2, 1, a, 2, ipiv, b, 2) != 0) {
    fprintf(stderr, "[FAIL] ZGESV call failed after repeated finalize\n");
    return 1;
  }

  printf("[PASS] adapter finalize is idempotent for ZGESV bridge\n");
  return 0;
}

static int test_adapter_finalize_idempotent_sposv(void) {
  fb_backend_vtable_t vtable;
  fb_sposv_fn sposv_call = NULL;
  fb_generic_fn first_ptr = NULL;
  fb_status_t status;

  float a[4] = {1.0f, 0.0f, 0.0f, 1.0f};
  float b[2] = {5.0f, 6.0f};

  memset(&vtable, 0, sizeof(vtable));

  vtable.ext_ops[FB_OP_SPOSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sposvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] first finalize status=%d\n", status);
    return 1;
  }

  first_ptr = vtable.ext_ops[FB_OP_SPOSV][FB_CONV_CBLAS];
  if (!first_ptr) {
    fprintf(stderr, "[FAIL] SPOSV slot missing after first finalize\n");
    return 1;
  }

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] second finalize status=%d\n", status);
    return 1;
  }

  if (vtable.ext_ops[FB_OP_SPOSV][FB_CONV_CBLAS] != first_ptr) {
    fprintf(stderr,
            "[FAIL] SPOSV adapter pointer changed after second finalize\n");
    return 1;
  }

  sposv_call = (fb_sposv_fn)vtable.ext_ops[FB_OP_SPOSV][FB_CONV_CBLAS];
  if (!sposv_call) {
    fprintf(stderr, "[FAIL] SPOSV call pointer missing after second finalize\n");
    return 1;
  }

  if (sposv_call(FB_LAYOUT_COL_MAJOR, FB_UPPER, 2, 1, a, 2, b, 2) != 0) {
    fprintf(stderr, "[FAIL] SPOSV call failed after repeated finalize\n");
    return 1;
  }

  printf("[PASS] adapter finalize is idempotent for SPOSV bridge\n");
  return 0;
}

static int test_adapter_finalize_idempotent_dposv(void) {
  fb_backend_vtable_t vtable;
  fb_dposv_fn dposv_call = NULL;
  fb_generic_fn first_ptr = NULL;
  fb_status_t status;

  double a[4] = {1.0, 0.0, 0.0, 1.0};
  double b[2] = {5.0, 6.0};

  memset(&vtable, 0, sizeof(vtable));

  vtable.ext_ops[FB_OP_DPOSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dposvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] first finalize status=%d\n", status);
    return 1;
  }

  first_ptr = vtable.ext_ops[FB_OP_DPOSV][FB_CONV_CBLAS];
  if (!first_ptr) {
    fprintf(stderr, "[FAIL] DPOSV slot missing after first finalize\n");
    return 1;
  }

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] second finalize status=%d\n", status);
    return 1;
  }

  if (vtable.ext_ops[FB_OP_DPOSV][FB_CONV_CBLAS] != first_ptr) {
    fprintf(stderr,
            "[FAIL] DPOSV adapter pointer changed after second finalize\n");
    return 1;
  }

  dposv_call = (fb_dposv_fn)vtable.ext_ops[FB_OP_DPOSV][FB_CONV_CBLAS];
  if (!dposv_call) {
    fprintf(stderr, "[FAIL] DPOSV call pointer missing after second finalize\n");
    return 1;
  }

  if (dposv_call(FB_LAYOUT_COL_MAJOR, FB_UPPER, 2, 1, a, 2, b, 2) != 0) {
    fprintf(stderr, "[FAIL] DPOSV call failed after repeated finalize\n");
    return 1;
  }

  printf("[PASS] adapter finalize is idempotent for DPOSV bridge\n");
  return 0;
}

static int test_adapter_finalize_idempotent_cposv(void) {
  fb_backend_vtable_t vtable;
  fb_cposv_fn cposv_call = NULL;
  fb_generic_fn first_ptr = NULL;
  fb_status_t status;

  fb_complex_float_t a[4] = {0};
  fb_complex_float_t b[2] = {0};

  memset(&vtable, 0, sizeof(vtable));

  vtable.ext_ops[FB_OP_CPOSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_cposvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] first finalize status=%d\n", status);
    return 1;
  }

  first_ptr = vtable.ext_ops[FB_OP_CPOSV][FB_CONV_CBLAS];
  if (!first_ptr) {
    fprintf(stderr, "[FAIL] CPOSV slot missing after first finalize\n");
    return 1;
  }

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] second finalize status=%d\n", status);
    return 1;
  }

  if (vtable.ext_ops[FB_OP_CPOSV][FB_CONV_CBLAS] != first_ptr) {
    fprintf(stderr,
            "[FAIL] CPOSV adapter pointer changed after second finalize\n");
    return 1;
  }

  cposv_call = (fb_cposv_fn)vtable.ext_ops[FB_OP_CPOSV][FB_CONV_CBLAS];
  if (!cposv_call) {
    fprintf(stderr, "[FAIL] CPOSV call pointer missing after second finalize\n");
    return 1;
  }

  if (cposv_call(FB_LAYOUT_COL_MAJOR, FB_UPPER, 2, 1, a, 2, b, 2) != 0) {
    fprintf(stderr, "[FAIL] CPOSV call failed after repeated finalize\n");
    return 1;
  }

  printf("[PASS] adapter finalize is idempotent for CPOSV bridge\n");
  return 0;
}

static int test_adapter_finalize_idempotent_zposv(void) {
  fb_backend_vtable_t vtable;
  fb_zposv_fn zposv_call = NULL;
  fb_generic_fn first_ptr = NULL;
  fb_status_t status;

  fb_complex_double_t a[4] = {0};
  fb_complex_double_t b[2] = {0};

  memset(&vtable, 0, sizeof(vtable));

  vtable.ext_ops[FB_OP_ZPOSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zposvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] first finalize status=%d\n", status);
    return 1;
  }

  first_ptr = vtable.ext_ops[FB_OP_ZPOSV][FB_CONV_CBLAS];
  if (!first_ptr) {
    fprintf(stderr, "[FAIL] ZPOSV slot missing after first finalize\n");
    return 1;
  }

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] second finalize status=%d\n", status);
    return 1;
  }

  if (vtable.ext_ops[FB_OP_ZPOSV][FB_CONV_CBLAS] != first_ptr) {
    fprintf(stderr,
            "[FAIL] ZPOSV adapter pointer changed after second finalize\n");
    return 1;
  }

  zposv_call = (fb_zposv_fn)vtable.ext_ops[FB_OP_ZPOSV][FB_CONV_CBLAS];
  if (!zposv_call) {
    fprintf(stderr, "[FAIL] ZPOSV call pointer missing after second finalize\n");
    return 1;
  }

  if (zposv_call(FB_LAYOUT_COL_MAJOR, FB_UPPER, 2, 1, a, 2, b, 2) != 0) {
    fprintf(stderr, "[FAIL] ZPOSV call failed after repeated finalize\n");
    return 1;
  }

  printf("[PASS] adapter finalize is idempotent for ZPOSV bridge\n");
  return 0;
}

static int test_adapter_finalize_idempotent_ssysv(void) {
  fb_backend_vtable_t vtable;
  fb_ssysv_fn ssysv_call = NULL;
  fb_generic_fn first_ptr = NULL;
  fb_status_t status;

  float a[4] = {1.0f, 0.0f, 0.0f, 1.0f};
  float b[2] = {5.0f, 6.0f};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  vtable.ext_ops[FB_OP_SSYSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_ssysvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] first finalize status=%d\n", status);
    return 1;
  }

  first_ptr = vtable.ext_ops[FB_OP_SSYSV][FB_CONV_CBLAS];
  if (!first_ptr) {
    fprintf(stderr, "[FAIL] SSYSV slot missing after first finalize\n");
    return 1;
  }

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] second finalize status=%d\n", status);
    return 1;
  }

  if (vtable.ext_ops[FB_OP_SSYSV][FB_CONV_CBLAS] != first_ptr) {
    fprintf(stderr,
            "[FAIL] SSYSV adapter pointer changed after second finalize\n");
    return 1;
  }

  ssysv_call = (fb_ssysv_fn)vtable.ext_ops[FB_OP_SSYSV][FB_CONV_CBLAS];
  if (!ssysv_call) {
    fprintf(stderr, "[FAIL] SSYSV call pointer missing after second finalize\n");
    return 1;
  }

  if (ssysv_call(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, a, 2, ipiv, b,
                 2) != 0) {
    fprintf(stderr, "[FAIL] SSYSV call failed after repeated finalize\n");
    return 1;
  }

  printf("[PASS] adapter finalize is idempotent for SSYSV bridge\n");
  return 0;
}

static int test_adapter_finalize_idempotent_dsysv(void) {
  fb_backend_vtable_t vtable;
  fb_dsysv_fn dsysv_call = NULL;
  fb_generic_fn first_ptr = NULL;
  fb_status_t status;

  double a[4] = {1.0, 0.0, 0.0, 1.0};
  double b[2] = {5.0, 6.0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  vtable.ext_ops[FB_OP_DSYSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dsysvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] first finalize status=%d\n", status);
    return 1;
  }

  first_ptr = vtable.ext_ops[FB_OP_DSYSV][FB_CONV_CBLAS];
  if (!first_ptr) {
    fprintf(stderr, "[FAIL] DSYSV slot missing after first finalize\n");
    return 1;
  }

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] second finalize status=%d\n", status);
    return 1;
  }

  if (vtable.ext_ops[FB_OP_DSYSV][FB_CONV_CBLAS] != first_ptr) {
    fprintf(stderr,
            "[FAIL] DSYSV adapter pointer changed after second finalize\n");
    return 1;
  }

  dsysv_call = (fb_dsysv_fn)vtable.ext_ops[FB_OP_DSYSV][FB_CONV_CBLAS];
  if (!dsysv_call) {
    fprintf(stderr, "[FAIL] DSYSV call pointer missing after second finalize\n");
    return 1;
  }

  if (dsysv_call(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, a, 2, ipiv, b,
                 2) != 0) {
    fprintf(stderr, "[FAIL] DSYSV call failed after repeated finalize\n");
    return 1;
  }

  printf("[PASS] adapter finalize is idempotent for DSYSV bridge\n");
  return 0;
}

static int test_adapter_finalize_idempotent_csysv(void) {
  fb_backend_vtable_t vtable;
  fb_csysv_fn csysv_call = NULL;
  fb_generic_fn first_ptr = NULL;
  fb_status_t status;

  fb_complex_float_t a[4] = {1.0f, 0.0f, 0.0f, 1.0f};
  fb_complex_float_t b[2] = {5.0f, 6.0f};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  vtable.ext_ops[FB_OP_CSYSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_csysvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] first finalize status=%d\n", status);
    return 1;
  }

  first_ptr = vtable.ext_ops[FB_OP_CSYSV][FB_CONV_CBLAS];
  if (!first_ptr) {
    fprintf(stderr, "[FAIL] CSYSV slot missing after first finalize\n");
    return 1;
  }

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] second finalize status=%d\n", status);
    return 1;
  }

  if (vtable.ext_ops[FB_OP_CSYSV][FB_CONV_CBLAS] != first_ptr) {
    fprintf(stderr,
            "[FAIL] CSYSV adapter pointer changed after second finalize\n");
    return 1;
  }

  csysv_call = (fb_csysv_fn)vtable.ext_ops[FB_OP_CSYSV][FB_CONV_CBLAS];
  if (!csysv_call) {
    fprintf(stderr, "[FAIL] CSYSV call pointer missing after second finalize\n");
    return 1;
  }

  if (csysv_call(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, a, 2, ipiv, b,
                 2) != 0) {
    fprintf(stderr, "[FAIL] CSYSV call failed after repeated finalize\n");
    return 1;
  }

  printf("[PASS] adapter finalize is idempotent for CSYSV bridge\n");
  return 0;
}

static int test_adapter_finalize_idempotent_zsysv(void) {
  fb_backend_vtable_t vtable;
  fb_zsysv_fn zsysv_call = NULL;
  fb_generic_fn first_ptr = NULL;
  fb_status_t status;

  fb_complex_double_t a[4] = {1.0, 0.0, 0.0, 1.0};
  fb_complex_double_t b[2] = {5.0, 6.0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  vtable.ext_ops[FB_OP_ZSYSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zsysvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] first finalize status=%d\n", status);
    return 1;
  }

  first_ptr = vtable.ext_ops[FB_OP_ZSYSV][FB_CONV_CBLAS];
  if (!first_ptr) {
    fprintf(stderr, "[FAIL] ZSYSV slot missing after first finalize\n");
    return 1;
  }

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] second finalize status=%d\n", status);
    return 1;
  }

  if (vtable.ext_ops[FB_OP_ZSYSV][FB_CONV_CBLAS] != first_ptr) {
    fprintf(stderr,
            "[FAIL] ZSYSV adapter pointer changed after second finalize\n");
    return 1;
  }

  zsysv_call = (fb_zsysv_fn)vtable.ext_ops[FB_OP_ZSYSV][FB_CONV_CBLAS];
  if (!zsysv_call) {
    fprintf(stderr, "[FAIL] ZSYSV call pointer missing after second finalize\n");
    return 1;
  }

  if (zsysv_call(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, a, 2, ipiv, b,
                 2) != 0) {
    fprintf(stderr, "[FAIL] ZSYSV call failed after repeated finalize\n");
    return 1;
  }

  printf("[PASS] adapter finalize is idempotent for ZSYSV bridge\n");
  return 0;
}

static int test_adapter_active_vtable_source_isolation(void) {
  fb_backend_vtable_t vtable_a;
  fb_backend_vtable_t vtable_b;
  fb_ssyev_cblas_fn ssyev_a = NULL;
  fb_ssyev_cblas_fn ssyev_b = NULL;
  fb_status_t status;

  float a[4] = {0};
  float w[2] = {0};

  memset(&vtable_a, 0, sizeof(vtable_a));
  memset(&vtable_b, 0, sizeof(vtable_b));
  g_ssyevr_isolation_a_calls = 0;
  g_ssyevr_isolation_b_calls = 0;

  vtable_a.ext_ops[FB_OP_SSYEVR][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_ssyevr_isolation_a;
  vtable_b.ext_ops[FB_OP_SSYEVR][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_ssyevr_isolation_b;

  status = fb_finalize_plugin_vtable(&vtable_a);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize vtable_a status=%d\n", status);
    return 1;
  }
  status = fb_finalize_plugin_vtable(&vtable_b);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize vtable_b status=%d\n", status);
    return 1;
  }

  ssyev_a = (fb_ssyev_cblas_fn)vtable_a.ext_ops[FB_OP_SSYEV][FB_CONV_CBLAS];
  ssyev_b = (fb_ssyev_cblas_fn)vtable_b.ext_ops[FB_OP_SSYEV][FB_CONV_CBLAS];
  if (!ssyev_a || !ssyev_b) {
    fprintf(stderr, "[FAIL] SSYEV adapter slot missing in isolation test\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_a);
  if (ssyev_a(FB_LAYOUT_COL_MAJOR, 'V', FB_UPPER, 2, a, 2, w) != 101) {
    fprintf(stderr,
            "[FAIL] SSYEV call for active vtable_a did not route to source A\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_b);
  if (ssyev_b(FB_LAYOUT_COL_MAJOR, 'V', FB_UPPER, 2, a, 2, w) != 202) {
    fprintf(stderr,
            "[FAIL] SSYEV call for active vtable_b did not route to source B\n");
    return 1;
  }

  if (g_ssyevr_isolation_a_calls != 1 || g_ssyevr_isolation_b_calls != 1) {
    fprintf(stderr,
            "[FAIL] active-vtable isolation call mismatch (A=%d B=%d)\n",
            g_ssyevr_isolation_a_calls, g_ssyevr_isolation_b_calls);
    return 1;
  }

  printf("[PASS] adapter source isolation follows active vtable selection\n");
  return 0;
}

static int test_adapter_active_vtable_source_isolation_sgesv(void) {
  fb_backend_vtable_t vtable_a;
  fb_backend_vtable_t vtable_b;
  fb_sgesv_fn sgesv_a = NULL;
  fb_sgesv_fn sgesv_b = NULL;
  fb_status_t status;

  float a[4] = {0};
  float b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable_a, 0, sizeof(vtable_a));
  memset(&vtable_b, 0, sizeof(vtable_b));
  g_sgesvx_isolation_a_calls = 0;
  g_sgesvx_isolation_b_calls = 0;

  vtable_a.ext_ops[FB_OP_SGESVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sgesvx_isolation_a;
  vtable_b.ext_ops[FB_OP_SGESVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sgesvx_isolation_b;

  status = fb_finalize_plugin_vtable(&vtable_a);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize sgesv vtable_a status=%d\n", status);
    return 1;
  }
  status = fb_finalize_plugin_vtable(&vtable_b);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize sgesv vtable_b status=%d\n", status);
    return 1;
  }

  sgesv_a = (fb_sgesv_fn)vtable_a.ext_ops[FB_OP_SGESV][FB_CONV_CBLAS];
  sgesv_b = (fb_sgesv_fn)vtable_b.ext_ops[FB_OP_SGESV][FB_CONV_CBLAS];
  if (!sgesv_a || !sgesv_b) {
    fprintf(stderr, "[FAIL] SGESV adapter slot missing in isolation test\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_a);
  if (sgesv_a(FB_LAYOUT_COL_MAJOR, 2, 1, a, 2, ipiv, b, 2) != 303) {
    fprintf(stderr,
            "[FAIL] SGESV call for active vtable_a did not route to source A\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_b);
  if (sgesv_b(FB_LAYOUT_COL_MAJOR, 2, 1, a, 2, ipiv, b, 2) != 404) {
    fprintf(stderr,
            "[FAIL] SGESV call for active vtable_b did not route to source B\n");
    return 1;
  }

  if (g_sgesvx_isolation_a_calls != 1 || g_sgesvx_isolation_b_calls != 1) {
    fprintf(stderr,
            "[FAIL] SGESV active-vtable isolation call mismatch (A=%d B=%d)\n",
            g_sgesvx_isolation_a_calls, g_sgesvx_isolation_b_calls);
    return 1;
  }

  printf("[PASS] SGESV adapter source isolation follows active vtable selection\n");
  return 0;
}

static int test_adapter_active_vtable_source_isolation_dgesv(void) {
  fb_backend_vtable_t vtable_a;
  fb_backend_vtable_t vtable_b;
  fb_dgesv_fn dgesv_a = NULL;
  fb_dgesv_fn dgesv_b = NULL;
  fb_status_t status;

  double a[4] = {0};
  double b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable_a, 0, sizeof(vtable_a));
  memset(&vtable_b, 0, sizeof(vtable_b));
  g_dgesvx_isolation_a_calls = 0;
  g_dgesvx_isolation_b_calls = 0;

  vtable_a.ext_ops[FB_OP_DGESVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dgesvx_isolation_a;
  vtable_b.ext_ops[FB_OP_DGESVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dgesvx_isolation_b;

  status = fb_finalize_plugin_vtable(&vtable_a);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize dgesv vtable_a status=%d\n", status);
    return 1;
  }
  status = fb_finalize_plugin_vtable(&vtable_b);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize dgesv vtable_b status=%d\n", status);
    return 1;
  }

  dgesv_a = (fb_dgesv_fn)vtable_a.ext_ops[FB_OP_DGESV][FB_CONV_CBLAS];
  dgesv_b = (fb_dgesv_fn)vtable_b.ext_ops[FB_OP_DGESV][FB_CONV_CBLAS];
  if (!dgesv_a || !dgesv_b) {
    fprintf(stderr, "[FAIL] DGESV adapter slot missing in isolation test\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_a);
  if (dgesv_a(FB_LAYOUT_COL_MAJOR, 2, 1, a, 2, ipiv, b, 2) != 505) {
    fprintf(stderr,
            "[FAIL] DGESV call for active vtable_a did not route to source A\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_b);
  if (dgesv_b(FB_LAYOUT_COL_MAJOR, 2, 1, a, 2, ipiv, b, 2) != 606) {
    fprintf(stderr,
            "[FAIL] DGESV call for active vtable_b did not route to source B\n");
    return 1;
  }

  if (g_dgesvx_isolation_a_calls != 1 || g_dgesvx_isolation_b_calls != 1) {
    fprintf(stderr,
            "[FAIL] DGESV active-vtable isolation call mismatch (A=%d B=%d)\n",
            g_dgesvx_isolation_a_calls, g_dgesvx_isolation_b_calls);
    return 1;
  }

  printf("[PASS] DGESV adapter source isolation follows active vtable selection\n");
  return 0;
}

static int test_adapter_active_vtable_source_isolation_cgesv(void) {
  fb_backend_vtable_t vtable_a;
  fb_backend_vtable_t vtable_b;
  fb_cgesv_fn cgesv_a = NULL;
  fb_cgesv_fn cgesv_b = NULL;
  fb_status_t status;

  fb_complex_float_t a[4] = {0};
  fb_complex_float_t b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable_a, 0, sizeof(vtable_a));
  memset(&vtable_b, 0, sizeof(vtable_b));
  g_cgesvx_isolation_a_calls = 0;
  g_cgesvx_isolation_b_calls = 0;

  vtable_a.ext_ops[FB_OP_CGESVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_cgesvx_isolation_a;
  vtable_b.ext_ops[FB_OP_CGESVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_cgesvx_isolation_b;

  status = fb_finalize_plugin_vtable(&vtable_a);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize cgesv vtable_a status=%d\n", status);
    return 1;
  }
  status = fb_finalize_plugin_vtable(&vtable_b);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize cgesv vtable_b status=%d\n", status);
    return 1;
  }

  cgesv_a = (fb_cgesv_fn)vtable_a.ext_ops[FB_OP_CGESV][FB_CONV_CBLAS];
  cgesv_b = (fb_cgesv_fn)vtable_b.ext_ops[FB_OP_CGESV][FB_CONV_CBLAS];
  if (!cgesv_a || !cgesv_b) {
    fprintf(stderr, "[FAIL] CGESV adapter slot missing in isolation test\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_a);
  if (cgesv_a(FB_LAYOUT_COL_MAJOR, 2, 1, a, 2, ipiv, b, 2) != 707) {
    fprintf(stderr,
            "[FAIL] CGESV call for active vtable_a did not route to source A\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_b);
  if (cgesv_b(FB_LAYOUT_COL_MAJOR, 2, 1, a, 2, ipiv, b, 2) != 808) {
    fprintf(stderr,
            "[FAIL] CGESV call for active vtable_b did not route to source B\n");
    return 1;
  }

  if (g_cgesvx_isolation_a_calls != 1 || g_cgesvx_isolation_b_calls != 1) {
    fprintf(stderr,
            "[FAIL] CGESV active-vtable isolation call mismatch (A=%d B=%d)\n",
            g_cgesvx_isolation_a_calls, g_cgesvx_isolation_b_calls);
    return 1;
  }

  printf("[PASS] CGESV adapter source isolation follows active vtable selection\n");
  return 0;
}

static int test_adapter_active_vtable_source_isolation_zgesv(void) {
  fb_backend_vtable_t vtable_a;
  fb_backend_vtable_t vtable_b;
  fb_zgesv_fn zgesv_a = NULL;
  fb_zgesv_fn zgesv_b = NULL;
  fb_status_t status;

  fb_complex_double_t a[4] = {0};
  fb_complex_double_t b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable_a, 0, sizeof(vtable_a));
  memset(&vtable_b, 0, sizeof(vtable_b));
  g_zgesvx_isolation_a_calls = 0;
  g_zgesvx_isolation_b_calls = 0;

  vtable_a.ext_ops[FB_OP_ZGESVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zgesvx_isolation_a;
  vtable_b.ext_ops[FB_OP_ZGESVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zgesvx_isolation_b;

  status = fb_finalize_plugin_vtable(&vtable_a);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize zgesv vtable_a status=%d\n", status);
    return 1;
  }
  status = fb_finalize_plugin_vtable(&vtable_b);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize zgesv vtable_b status=%d\n", status);
    return 1;
  }

  zgesv_a = (fb_zgesv_fn)vtable_a.ext_ops[FB_OP_ZGESV][FB_CONV_CBLAS];
  zgesv_b = (fb_zgesv_fn)vtable_b.ext_ops[FB_OP_ZGESV][FB_CONV_CBLAS];
  if (!zgesv_a || !zgesv_b) {
    fprintf(stderr, "[FAIL] ZGESV adapter slot missing in isolation test\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_a);
  if (zgesv_a(FB_LAYOUT_COL_MAJOR, 2, 1, a, 2, ipiv, b, 2) != 1007) {
    fprintf(stderr,
            "[FAIL] ZGESV call for active vtable_a did not route to source A\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_b);
  if (zgesv_b(FB_LAYOUT_COL_MAJOR, 2, 1, a, 2, ipiv, b, 2) != 1108) {
    fprintf(stderr,
            "[FAIL] ZGESV call for active vtable_b did not route to source B\n");
    return 1;
  }

  if (g_zgesvx_isolation_a_calls != 1 || g_zgesvx_isolation_b_calls != 1) {
    fprintf(stderr,
            "[FAIL] ZGESV active-vtable isolation call mismatch (A=%d B=%d)\n",
            g_zgesvx_isolation_a_calls, g_zgesvx_isolation_b_calls);
    return 1;
  }

  printf("[PASS] ZGESV adapter source isolation follows active vtable selection\n");
  return 0;
}

static int test_adapter_active_vtable_source_isolation_sposv(void) {
  fb_backend_vtable_t vtable_a;
  fb_backend_vtable_t vtable_b;
  fb_sposv_fn sposv_a = NULL;
  fb_sposv_fn sposv_b = NULL;
  fb_status_t status;

  float a[4] = {0};
  float b[2] = {0};

  memset(&vtable_a, 0, sizeof(vtable_a));
  memset(&vtable_b, 0, sizeof(vtable_b));
  g_sposvx_isolation_a_calls = 0;
  g_sposvx_isolation_b_calls = 0;

  vtable_a.ext_ops[FB_OP_SPOSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sposvx_isolation_a;
  vtable_b.ext_ops[FB_OP_SPOSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sposvx_isolation_b;

  status = fb_finalize_plugin_vtable(&vtable_a);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize sposv vtable_a status=%d\n", status);
    return 1;
  }
  status = fb_finalize_plugin_vtable(&vtable_b);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize sposv vtable_b status=%d\n", status);
    return 1;
  }

  sposv_a = (fb_sposv_fn)vtable_a.ext_ops[FB_OP_SPOSV][FB_CONV_CBLAS];
  sposv_b = (fb_sposv_fn)vtable_b.ext_ops[FB_OP_SPOSV][FB_CONV_CBLAS];
  if (!sposv_a || !sposv_b) {
    fprintf(stderr, "[FAIL] SPOSV adapter slot missing in isolation test\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_a);
  if (sposv_a(FB_LAYOUT_COL_MAJOR, FB_UPPER, 2, 1, a, 2, b, 2) != 1209) {
    fprintf(stderr,
            "[FAIL] SPOSV call for active vtable_a did not route to source A\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_b);
  if (sposv_b(FB_LAYOUT_COL_MAJOR, FB_UPPER, 2, 1, a, 2, b, 2) != 1310) {
    fprintf(stderr,
            "[FAIL] SPOSV call for active vtable_b did not route to source B\n");
    return 1;
  }

  if (g_sposvx_isolation_a_calls != 1 || g_sposvx_isolation_b_calls != 1) {
    fprintf(stderr,
            "[FAIL] SPOSV active-vtable isolation call mismatch (A=%d B=%d)\n",
            g_sposvx_isolation_a_calls, g_sposvx_isolation_b_calls);
    return 1;
  }

  printf("[PASS] SPOSV adapter source isolation follows active vtable selection\n");
  return 0;
}

static int test_adapter_active_vtable_source_isolation_dposv(void) {
  fb_backend_vtable_t vtable_a;
  fb_backend_vtable_t vtable_b;
  fb_dposv_fn dposv_a = NULL;
  fb_dposv_fn dposv_b = NULL;
  fb_status_t status;

  double a[4] = {0};
  double b[2] = {0};

  memset(&vtable_a, 0, sizeof(vtable_a));
  memset(&vtable_b, 0, sizeof(vtable_b));
  g_dposvx_isolation_a_calls = 0;
  g_dposvx_isolation_b_calls = 0;

  vtable_a.ext_ops[FB_OP_DPOSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dposvx_isolation_a;
  vtable_b.ext_ops[FB_OP_DPOSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dposvx_isolation_b;

  status = fb_finalize_plugin_vtable(&vtable_a);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize dposv vtable_a status=%d\n", status);
    return 1;
  }
  status = fb_finalize_plugin_vtable(&vtable_b);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize dposv vtable_b status=%d\n", status);
    return 1;
  }

  dposv_a = (fb_dposv_fn)vtable_a.ext_ops[FB_OP_DPOSV][FB_CONV_CBLAS];
  dposv_b = (fb_dposv_fn)vtable_b.ext_ops[FB_OP_DPOSV][FB_CONV_CBLAS];
  if (!dposv_a || !dposv_b) {
    fprintf(stderr, "[FAIL] DPOSV adapter slot missing in isolation test\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_a);
  if (dposv_a(FB_LAYOUT_COL_MAJOR, FB_UPPER, 2, 1, a, 2, b, 2) != 1411) {
    fprintf(stderr,
            "[FAIL] DPOSV call for active vtable_a did not route to source A\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_b);
  if (dposv_b(FB_LAYOUT_COL_MAJOR, FB_UPPER, 2, 1, a, 2, b, 2) != 1512) {
    fprintf(stderr,
            "[FAIL] DPOSV call for active vtable_b did not route to source B\n");
    return 1;
  }

  if (g_dposvx_isolation_a_calls != 1 || g_dposvx_isolation_b_calls != 1) {
    fprintf(stderr,
            "[FAIL] DPOSV active-vtable isolation call mismatch (A=%d B=%d)\n",
            g_dposvx_isolation_a_calls, g_dposvx_isolation_b_calls);
    return 1;
  }

  printf("[PASS] DPOSV adapter source isolation follows active vtable selection\n");
  return 0;
}

static int test_adapter_active_vtable_source_isolation_cposv(void) {
  fb_backend_vtable_t vtable_a;
  fb_backend_vtable_t vtable_b;
  fb_cposv_fn cposv_a = NULL;
  fb_cposv_fn cposv_b = NULL;
  fb_status_t status;

  fb_complex_float_t a[4] = {0};
  fb_complex_float_t b[2] = {0};

  memset(&vtable_a, 0, sizeof(vtable_a));
  memset(&vtable_b, 0, sizeof(vtable_b));
  g_cposvx_isolation_a_calls = 0;
  g_cposvx_isolation_b_calls = 0;

  vtable_a.ext_ops[FB_OP_CPOSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_cposvx_isolation_a;
  vtable_b.ext_ops[FB_OP_CPOSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_cposvx_isolation_b;

  status = fb_finalize_plugin_vtable(&vtable_a);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize cposv vtable_a status=%d\n", status);
    return 1;
  }
  status = fb_finalize_plugin_vtable(&vtable_b);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize cposv vtable_b status=%d\n", status);
    return 1;
  }

  cposv_a = (fb_cposv_fn)vtable_a.ext_ops[FB_OP_CPOSV][FB_CONV_CBLAS];
  cposv_b = (fb_cposv_fn)vtable_b.ext_ops[FB_OP_CPOSV][FB_CONV_CBLAS];
  if (!cposv_a || !cposv_b) {
    fprintf(stderr, "[FAIL] CPOSV adapter slot missing in isolation test\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_a);
  if (cposv_a(FB_LAYOUT_COL_MAJOR, FB_UPPER, 2, 1, a, 2, b, 2) != 1613) {
    fprintf(stderr,
            "[FAIL] CPOSV call for active vtable_a did not route to source A\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_b);
  if (cposv_b(FB_LAYOUT_COL_MAJOR, FB_UPPER, 2, 1, a, 2, b, 2) != 1714) {
    fprintf(stderr,
            "[FAIL] CPOSV call for active vtable_b did not route to source B\n");
    return 1;
  }

  if (g_cposvx_isolation_a_calls != 1 || g_cposvx_isolation_b_calls != 1) {
    fprintf(stderr,
            "[FAIL] CPOSV active-vtable isolation call mismatch (A=%d B=%d)\n",
            g_cposvx_isolation_a_calls, g_cposvx_isolation_b_calls);
    return 1;
  }

  printf("[PASS] CPOSV adapter source isolation follows active vtable selection\n");
  return 0;
}

static int test_adapter_active_vtable_source_isolation_zposv(void) {
  fb_backend_vtable_t vtable_a;
  fb_backend_vtable_t vtable_b;
  fb_zposv_fn zposv_a = NULL;
  fb_zposv_fn zposv_b = NULL;
  fb_status_t status;

  fb_complex_double_t a[4] = {0};
  fb_complex_double_t b[2] = {0};

  memset(&vtable_a, 0, sizeof(vtable_a));
  memset(&vtable_b, 0, sizeof(vtable_b));
  g_zposvx_isolation_a_calls = 0;
  g_zposvx_isolation_b_calls = 0;

  vtable_a.ext_ops[FB_OP_ZPOSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zposvx_isolation_a;
  vtable_b.ext_ops[FB_OP_ZPOSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zposvx_isolation_b;

  status = fb_finalize_plugin_vtable(&vtable_a);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize zposv vtable_a status=%d\n", status);
    return 1;
  }
  status = fb_finalize_plugin_vtable(&vtable_b);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize zposv vtable_b status=%d\n", status);
    return 1;
  }

  zposv_a = (fb_zposv_fn)vtable_a.ext_ops[FB_OP_ZPOSV][FB_CONV_CBLAS];
  zposv_b = (fb_zposv_fn)vtable_b.ext_ops[FB_OP_ZPOSV][FB_CONV_CBLAS];
  if (!zposv_a || !zposv_b) {
    fprintf(stderr, "[FAIL] ZPOSV adapter slot missing in isolation test\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_a);
  if (zposv_a(FB_LAYOUT_COL_MAJOR, FB_UPPER, 2, 1, a, 2, b, 2) != 1815) {
    fprintf(stderr,
            "[FAIL] ZPOSV call for active vtable_a did not route to source A\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_b);
  if (zposv_b(FB_LAYOUT_COL_MAJOR, FB_UPPER, 2, 1, a, 2, b, 2) != 1916) {
    fprintf(stderr,
            "[FAIL] ZPOSV call for active vtable_b did not route to source B\n");
    return 1;
  }

  if (g_zposvx_isolation_a_calls != 1 || g_zposvx_isolation_b_calls != 1) {
    fprintf(stderr,
            "[FAIL] ZPOSV active-vtable isolation call mismatch (A=%d B=%d)\n",
            g_zposvx_isolation_a_calls, g_zposvx_isolation_b_calls);
    return 1;
  }

  printf("[PASS] ZPOSV adapter source isolation follows active vtable selection\n");
  return 0;
}

static int test_adapter_active_vtable_source_isolation_ssysv(void) {
  fb_backend_vtable_t vtable_a;
  fb_backend_vtable_t vtable_b;
  fb_ssysv_fn ssysv_a = NULL;
  fb_ssysv_fn ssysv_b = NULL;
  fb_status_t status;

  float a[4] = {0};
  float b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable_a, 0, sizeof(vtable_a));
  memset(&vtable_b, 0, sizeof(vtable_b));
  g_ssysvx_isolation_a_calls = 0;
  g_ssysvx_isolation_b_calls = 0;

  vtable_a.ext_ops[FB_OP_SSYSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_ssysvx_isolation_a;
  vtable_b.ext_ops[FB_OP_SSYSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_ssysvx_isolation_b;

  status = fb_finalize_plugin_vtable(&vtable_a);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize ssysv vtable_a status=%d\n", status);
    return 1;
  }
  status = fb_finalize_plugin_vtable(&vtable_b);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize ssysv vtable_b status=%d\n", status);
    return 1;
  }

  ssysv_a = (fb_ssysv_fn)vtable_a.ext_ops[FB_OP_SSYSV][FB_CONV_CBLAS];
  ssysv_b = (fb_ssysv_fn)vtable_b.ext_ops[FB_OP_SSYSV][FB_CONV_CBLAS];
  if (!ssysv_a || !ssysv_b) {
    fprintf(stderr, "[FAIL] SSYSV adapter slot missing in isolation test\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_a);
  if (ssysv_a(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, a, 2, ipiv, b, 2) !=
      2017) {
    fprintf(stderr,
            "[FAIL] SSYSV call for active vtable_a did not route to source A\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_b);
  if (ssysv_b(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, a, 2, ipiv, b, 2) !=
      2118) {
    fprintf(stderr,
            "[FAIL] SSYSV call for active vtable_b did not route to source B\n");
    return 1;
  }

  if (g_ssysvx_isolation_a_calls != 1 || g_ssysvx_isolation_b_calls != 1) {
    fprintf(stderr,
            "[FAIL] SSYSV active-vtable isolation call mismatch (A=%d B=%d)\n",
            g_ssysvx_isolation_a_calls, g_ssysvx_isolation_b_calls);
    return 1;
  }

  printf("[PASS] SSYSV adapter source isolation follows active vtable selection\n");
  return 0;
}

static int test_adapter_active_vtable_source_isolation_dsysv(void) {
  fb_backend_vtable_t vtable_a;
  fb_backend_vtable_t vtable_b;
  fb_dsysv_fn dsysv_a = NULL;
  fb_dsysv_fn dsysv_b = NULL;
  fb_status_t status;

  double a[4] = {0};
  double b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable_a, 0, sizeof(vtable_a));
  memset(&vtable_b, 0, sizeof(vtable_b));
  g_dsysvx_isolation_a_calls = 0;
  g_dsysvx_isolation_b_calls = 0;

  vtable_a.ext_ops[FB_OP_DSYSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dsysvx_isolation_a;
  vtable_b.ext_ops[FB_OP_DSYSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dsysvx_isolation_b;

  status = fb_finalize_plugin_vtable(&vtable_a);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize dsysv vtable_a status=%d\n", status);
    return 1;
  }
  status = fb_finalize_plugin_vtable(&vtable_b);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize dsysv vtable_b status=%d\n", status);
    return 1;
  }

  dsysv_a = (fb_dsysv_fn)vtable_a.ext_ops[FB_OP_DSYSV][FB_CONV_CBLAS];
  dsysv_b = (fb_dsysv_fn)vtable_b.ext_ops[FB_OP_DSYSV][FB_CONV_CBLAS];
  if (!dsysv_a || !dsysv_b) {
    fprintf(stderr, "[FAIL] DSYSV adapter slot missing in isolation test\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_a);
  if (dsysv_a(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, a, 2, ipiv, b, 2) !=
      2219) {
    fprintf(stderr,
            "[FAIL] DSYSV call for active vtable_a did not route to source A\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_b);
  if (dsysv_b(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, a, 2, ipiv, b, 2) !=
      2320) {
    fprintf(stderr,
            "[FAIL] DSYSV call for active vtable_b did not route to source B\n");
    return 1;
  }

  if (g_dsysvx_isolation_a_calls != 1 || g_dsysvx_isolation_b_calls != 1) {
    fprintf(stderr,
            "[FAIL] DSYSV active-vtable isolation call mismatch (A=%d B=%d)\n",
            g_dsysvx_isolation_a_calls, g_dsysvx_isolation_b_calls);
    return 1;
  }

  printf("[PASS] DSYSV adapter source isolation follows active vtable selection\n");
  return 0;
}

static int test_adapter_active_vtable_source_isolation_csysv(void) {
  fb_backend_vtable_t vtable_a;
  fb_backend_vtable_t vtable_b;
  fb_csysv_fn csysv_a = NULL;
  fb_csysv_fn csysv_b = NULL;
  fb_status_t status;

  fb_complex_float_t a[4] = {0};
  fb_complex_float_t b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable_a, 0, sizeof(vtable_a));
  memset(&vtable_b, 0, sizeof(vtable_b));
  g_csysvx_isolation_a_calls = 0;
  g_csysvx_isolation_b_calls = 0;

  vtable_a.ext_ops[FB_OP_CSYSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_csysvx_isolation_a;
  vtable_b.ext_ops[FB_OP_CSYSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_csysvx_isolation_b;

  status = fb_finalize_plugin_vtable(&vtable_a);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize csysv vtable_a status=%d\n", status);
    return 1;
  }
  status = fb_finalize_plugin_vtable(&vtable_b);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize csysv vtable_b status=%d\n", status);
    return 1;
  }

  csysv_a = (fb_csysv_fn)vtable_a.ext_ops[FB_OP_CSYSV][FB_CONV_CBLAS];
  csysv_b = (fb_csysv_fn)vtable_b.ext_ops[FB_OP_CSYSV][FB_CONV_CBLAS];
  if (!csysv_a || !csysv_b) {
    fprintf(stderr, "[FAIL] CSYSV adapter slot missing in isolation test\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_a);
  if (csysv_a(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, a, 2, ipiv, b, 2) !=
      2421) {
    fprintf(stderr,
            "[FAIL] CSYSV call for active vtable_a did not route to source A\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_b);
  if (csysv_b(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, a, 2, ipiv, b, 2) !=
      2522) {
    fprintf(stderr,
            "[FAIL] CSYSV call for active vtable_b did not route to source B\n");
    return 1;
  }

  if (g_csysvx_isolation_a_calls != 1 || g_csysvx_isolation_b_calls != 1) {
    fprintf(stderr,
            "[FAIL] CSYSV active-vtable isolation call mismatch (A=%d B=%d)\n",
            g_csysvx_isolation_a_calls, g_csysvx_isolation_b_calls);
    return 1;
  }

  printf("[PASS] CSYSV adapter source isolation follows active vtable selection\n");
  return 0;
}

static int test_adapter_active_vtable_source_isolation_zsysv(void) {
  fb_backend_vtable_t vtable_a;
  fb_backend_vtable_t vtable_b;
  fb_zsysv_fn zsysv_a = NULL;
  fb_zsysv_fn zsysv_b = NULL;
  fb_status_t status;

  fb_complex_double_t a[4] = {0};
  fb_complex_double_t b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable_a, 0, sizeof(vtable_a));
  memset(&vtable_b, 0, sizeof(vtable_b));
  g_zsysvx_isolation_a_calls = 0;
  g_zsysvx_isolation_b_calls = 0;

  vtable_a.ext_ops[FB_OP_ZSYSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zsysvx_isolation_a;
  vtable_b.ext_ops[FB_OP_ZSYSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zsysvx_isolation_b;

  status = fb_finalize_plugin_vtable(&vtable_a);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize zsysv vtable_a status=%d\n", status);
    return 1;
  }
  status = fb_finalize_plugin_vtable(&vtable_b);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize zsysv vtable_b status=%d\n", status);
    return 1;
  }

  zsysv_a = (fb_zsysv_fn)vtable_a.ext_ops[FB_OP_ZSYSV][FB_CONV_CBLAS];
  zsysv_b = (fb_zsysv_fn)vtable_b.ext_ops[FB_OP_ZSYSV][FB_CONV_CBLAS];
  if (!zsysv_a || !zsysv_b) {
    fprintf(stderr, "[FAIL] ZSYSV adapter slot missing in isolation test\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_a);
  if (zsysv_a(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, a, 2, ipiv, b, 2) !=
      2623) {
    fprintf(stderr,
            "[FAIL] ZSYSV call for active vtable_a did not route to source A\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_b);
  if (zsysv_b(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, a, 2, ipiv, b, 2) !=
      2724) {
    fprintf(stderr,
            "[FAIL] ZSYSV call for active vtable_b did not route to source B\n");
    return 1;
  }

  if (g_zsysvx_isolation_a_calls != 1 || g_zsysvx_isolation_b_calls != 1) {
    fprintf(stderr,
            "[FAIL] ZSYSV active-vtable isolation call mismatch (A=%d B=%d)\n",
            g_zsysvx_isolation_a_calls, g_zsysvx_isolation_b_calls);
    return 1;
  }

  printf("[PASS] ZSYSV adapter source isolation follows active vtable selection\n");
  return 0;
}

int main(void) {
  int status = 0;

  status |= test_group_fill_from_chbevd();
  status |= test_existing_slot_not_overwritten();
  status |= test_group_fill_from_ssyevd();
  status |= test_group_fill_from_ssygvd();
  status |= test_fortran_only_donor_propagates_after_conv_thunks();
  status |= test_adapter_bridge_ssyev_from_ssyevr();
  status |= test_adapter_bridge_ssyev_from_ssyevx();
  status |= test_adapter_bridge_dsyev_from_dsyevr();
  status |= test_adapter_bridge_ssygv_from_ssygvx();
  status |= test_adapter_bridge_dsygv_from_dsygvx();
  status |= test_adapter_bridge_dsyev_from_dsyevx();
  status |= test_adapter_bridge_sgeev_from_sgeevx();
  status |= test_adapter_bridge_dgeev_from_dgeevx();
  status |= test_adapter_bridge_cgeev_from_cgeevx();
  status |= test_adapter_bridge_zgeev_from_zgeevx();
  status |= test_adapter_bridge_sgesv_from_sgesvx();
  status |= test_adapter_bridge_dgesv_from_dgesvx();
  status |= test_adapter_bridge_cgesv_from_cgesvx();
  status |= test_adapter_bridge_zgesv_from_zgesvx();
  status |= test_adapter_bridge_sposv_from_sposvx();
  status |= test_adapter_bridge_dposv_from_dposvx();
  status |= test_adapter_bridge_cposv_from_cposvx();
  status |= test_adapter_bridge_zposv_from_zposvx();
  status |= test_adapter_bridge_ssysv_from_ssysvx();
  status |= test_adapter_bridge_dsysv_from_dsysvx();
  status |= test_adapter_bridge_csysv_from_csysvx();
  status |= test_adapter_bridge_zsysv_from_zsysvx();
  status |= test_adapter_bridge_sstev_from_sstevr();
  status |= test_adapter_bridge_dstev_from_dstevx();
  status |= test_adapter_bridge_sstev_from_sstevx();
  status |= test_adapter_bridge_dstev_from_dstevr();
  status |= test_adapter_bridge_sstev_from_sstevd();
  status |= test_adapter_bridge_dstev_from_dstevd();
  status |= test_adapter_precedence_ssyev_prefers_ssyevr();
  status |= test_adapter_precedence_dsyev_prefers_dsyevr();
  status |= test_adapter_precedence_sstev_prefers_sstevr();
  status |= test_adapter_precedence_dstev_prefers_dstevr();
  status |= test_adapter_does_not_override_existing_sgeev();
  status |= test_adapter_does_not_override_existing_sgesv();
  status |= test_adapter_does_not_override_existing_dgesv();
  status |= test_adapter_does_not_override_existing_cgesv();
  status |= test_adapter_does_not_override_existing_zgesv();
  status |= test_adapter_does_not_override_existing_sposv();
  status |= test_adapter_does_not_override_existing_dposv();
  status |= test_adapter_does_not_override_existing_cposv();
  status |= test_adapter_does_not_override_existing_zposv();
  status |= test_adapter_does_not_override_existing_ssysv();
  status |= test_adapter_does_not_override_existing_dsysv();
  status |= test_adapter_does_not_override_existing_csysv();
  status |= test_adapter_does_not_override_existing_zsysv();
  status |= test_adapter_does_not_override_existing_ssyev();
  status |= test_adapter_does_not_override_existing_sstev();
  status |= test_adapter_finalize_idempotent_ssyev();
  status |= test_adapter_finalize_idempotent_sgesv();
  status |= test_adapter_finalize_idempotent_dgesv();
  status |= test_adapter_finalize_idempotent_cgesv();
  status |= test_adapter_finalize_idempotent_zgesv();
  status |= test_adapter_finalize_idempotent_sposv();
  status |= test_adapter_finalize_idempotent_dposv();
  status |= test_adapter_finalize_idempotent_cposv();
  status |= test_adapter_finalize_idempotent_zposv();
  status |= test_adapter_finalize_idempotent_ssysv();
  status |= test_adapter_finalize_idempotent_dsysv();
  status |= test_adapter_finalize_idempotent_csysv();
  status |= test_adapter_finalize_idempotent_zsysv();
  status |= test_adapter_active_vtable_source_isolation();
  status |= test_adapter_active_vtable_source_isolation_sgesv();
  status |= test_adapter_active_vtable_source_isolation_dgesv();
  status |= test_adapter_active_vtable_source_isolation_cgesv();
  status |= test_adapter_active_vtable_source_isolation_zgesv();
  status |= test_adapter_active_vtable_source_isolation_sposv();
  status |= test_adapter_active_vtable_source_isolation_dposv();
  status |= test_adapter_active_vtable_source_isolation_cposv();
  status |= test_adapter_active_vtable_source_isolation_zposv();
  status |= test_adapter_active_vtable_source_isolation_ssysv();
  status |= test_adapter_active_vtable_source_isolation_dsysv();
  status |= test_adapter_active_vtable_source_isolation_csysv();
  status |= test_adapter_active_vtable_source_isolation_zsysv();

  if (status != 0) {
    fprintf(stderr, "Result: FAIL\n");
    return 1;
  }

  printf("Result: PASS\n");
  return 0;
}
