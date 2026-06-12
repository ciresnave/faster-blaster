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
typedef int (*fb_sgesvxx_cblas_fn)(fb_layout_t layout, char fact, char trans,
                                   int n, int nrhs, float *a, int lda,
                                   float *af, int ldaf, int *ipiv,
                                   char *equed, float *r, float *c,
                                   float *b, int ldb, float *x, int ldx,
                                   float *rcond, float *rpvgrw, float *berr,
                                   int n_err_bnds, float *err_bnds_norm,
                                   float *err_bnds_comp, int nparams,
                                   float *params);
typedef int (*fb_dgesvxx_cblas_fn)(fb_layout_t layout, char fact, char trans,
                                   int n, int nrhs, double *a, int lda,
                                   double *af, int ldaf, int *ipiv,
                                   char *equed, double *r, double *c,
                                   double *b, int ldb, double *x, int ldx,
                                   double *rcond, double *rpvgrw,
                                   double *berr, int n_err_bnds,
                                   double *err_bnds_norm,
                                   double *err_bnds_comp, int nparams,
                                   double *params);
typedef int (*fb_cgesvxx_cblas_fn)(fb_layout_t layout, char fact, char trans,
                                   int n, int nrhs, fb_complex_float_t *a,
                                   int lda, fb_complex_float_t *af, int ldaf,
                                   int *ipiv, char *equed, float *r,
                                   float *c, fb_complex_float_t *b, int ldb,
                                   fb_complex_float_t *x, int ldx,
                                   float *rcond, float *rpvgrw, float *berr,
                                   int n_err_bnds, float *err_bnds_norm,
                                   float *err_bnds_comp, int nparams,
                                   float *params);
typedef int (*fb_zgesvxx_cblas_fn)(fb_layout_t layout, char fact, char trans,
                                   int n, int nrhs, fb_complex_double_t *a,
                                   int lda, fb_complex_double_t *af,
                                   int ldaf, int *ipiv, char *equed,
                                   double *r, double *c,
                                   fb_complex_double_t *b, int ldb,
                                   fb_complex_double_t *x, int ldx,
                                   double *rcond, double *rpvgrw,
                                   double *berr, int n_err_bnds,
                                   double *err_bnds_norm,
                                   double *err_bnds_comp, int nparams,
                                   double *params);

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

typedef int (*fb_sposvxx_cblas_fn)(fb_layout_t layout, char fact, char uplo,
                                   int n, int nrhs, float *a, int lda,
                                   float *af, int ldaf, char *equed,
                                   float *s, float *b, int ldb, float *x,
                                   int ldx, float *rcond, float *rpvgrw,
                                   float *berr, int n_err_bnds,
                                   float *err_bnds_norm,
                                   float *err_bnds_comp, int nparams,
                                   float *params);
typedef int (*fb_dposvxx_cblas_fn)(fb_layout_t layout, char fact, char uplo,
                                   int n, int nrhs, double *a, int lda,
                                   double *af, int ldaf, char *equed,
                                   double *s, double *b, int ldb, double *x,
                                   int ldx, double *rcond, double *rpvgrw,
                                   double *berr, int n_err_bnds,
                                   double *err_bnds_norm,
                                   double *err_bnds_comp, int nparams,
                                   double *params);
typedef int (*fb_cposvxx_cblas_fn)(fb_layout_t layout, char fact, char uplo,
                                   int n, int nrhs, fb_complex_float_t *a,
                                   int lda, fb_complex_float_t *af,
                                   int ldaf, char *equed, float *s,
                                   fb_complex_float_t *b, int ldb,
                                   fb_complex_float_t *x, int ldx,
                                   float *rcond, float *rpvgrw, float *berr,
                                   int n_err_bnds, float *err_bnds_norm,
                                   float *err_bnds_comp, int nparams,
                                   float *params);
typedef int (*fb_zposvxx_cblas_fn)(fb_layout_t layout, char fact, char uplo,
                                   int n, int nrhs, fb_complex_double_t *a,
                                   int lda, fb_complex_double_t *af,
                                   int ldaf, char *equed, double *s,
                                   fb_complex_double_t *b, int ldb,
                                   fb_complex_double_t *x, int ldx,
                                   double *rcond, double *rpvgrw,
                                   double *berr, int n_err_bnds,
                                   double *err_bnds_norm,
                                   double *err_bnds_comp, int nparams,
                                   double *params);

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

typedef int (*fb_ssysvxx_cblas_fn)(fb_layout_t layout, char fact, char uplo,
                                   int n, int nrhs, float *a, int lda,
                                   float *af, int ldaf, int *ipiv,
                                   char *equed, float *s, float *b, int ldb,
                                   float *x, int ldx, float *rcond,
                                   float *rpvgrw, float *berr,
                                   int n_err_bnds, float *err_bnds_norm,
                                   float *err_bnds_comp, int nparams,
                                   float *params);
typedef int (*fb_dsysvxx_cblas_fn)(fb_layout_t layout, char fact, char uplo,
                                   int n, int nrhs, double *a, int lda,
                                   double *af, int ldaf, int *ipiv,
                                   char *equed, double *s, double *b, int ldb,
                                   double *x, int ldx, double *rcond,
                                   double *rpvgrw, double *berr,
                                   int n_err_bnds, double *err_bnds_norm,
                                   double *err_bnds_comp, int nparams,
                                   double *params);
typedef int (*fb_csysvxx_cblas_fn)(fb_layout_t layout, char fact, char uplo,
                                   int n, int nrhs, fb_complex_float_t *a,
                                   int lda, fb_complex_float_t *af, int ldaf,
                                   int *ipiv, char *equed, float *s,
                                   fb_complex_float_t *b, int ldb,
                                   fb_complex_float_t *x, int ldx,
                                   float *rcond, float *rpvgrw, float *berr,
                                   int n_err_bnds, float *err_bnds_norm,
                                   float *err_bnds_comp, int nparams,
                                   float *params);
typedef int (*fb_zsysvxx_cblas_fn)(fb_layout_t layout, char fact, char uplo,
                                   int n, int nrhs, fb_complex_double_t *a,
                                   int lda, fb_complex_double_t *af, int ldaf,
                                   int *ipiv, char *equed, double *s,
                                   fb_complex_double_t *b, int ldb,
                                   fb_complex_double_t *x, int ldx,
                                   double *rcond, double *rpvgrw,
                                   double *berr, int n_err_bnds,
                                   double *err_bnds_norm,
                                   double *err_bnds_comp, int nparams,
                                   double *params);

typedef int (*fb_sgbsv_cblas_fn)(fb_layout_t layout, int n, int kl, int ku,
                                 int nrhs, float *ab, int ldab, int *ipiv,
                                 float *b, int ldb);

typedef int (*fb_dgbsv_cblas_fn)(fb_layout_t layout, int n, int kl, int ku,
                                 int nrhs, double *ab, int ldab, int *ipiv,
                                 double *b, int ldb);

typedef int (*fb_cgbsv_cblas_fn)(fb_layout_t layout, int n, int kl, int ku,
                                 int nrhs, fb_complex_float_t *ab, int ldab,
                                 int *ipiv, fb_complex_float_t *b, int ldb);

typedef int (*fb_zgbsv_cblas_fn)(fb_layout_t layout, int n, int kl, int ku,
                                 int nrhs, fb_complex_double_t *ab, int ldab,
                                 int *ipiv, fb_complex_double_t *b, int ldb);

typedef int (*fb_sgbsvx_cblas_fn)(fb_layout_t layout, char fact, char trans,
                                  int n, int kl, int ku, int nrhs, float *ab,
                                  int ldab, float *afb, int ldafb, int *ipiv,
                                  char *equed, float *r, float *c, float *b,
                                  int ldb, float *x, int ldx, float *rcond,
                                  float *ferr, float *berr, float *rpivot);

typedef int (*fb_dgbsvx_cblas_fn)(fb_layout_t layout, char fact, char trans,
                                  int n, int kl, int ku, int nrhs,
                                  double *ab, int ldab, double *afb,
                                  int ldafb, int *ipiv, char *equed,
                                  double *r, double *c, double *b, int ldb,
                                  double *x, int ldx, double *rcond,
                                  double *ferr, double *berr,
                                  double *rpivot);

typedef int (*fb_cgbsvx_cblas_fn)(fb_layout_t layout, char fact, char trans,
                                  int n, int kl, int ku, int nrhs,
                                  fb_complex_float_t *ab, int ldab,
                                  fb_complex_float_t *afb, int ldafb,
                                  int *ipiv, char *equed, float *r, float *c,
                                  fb_complex_float_t *b, int ldb,
                                  fb_complex_float_t *x, int ldx,
                                  float *rcond, float *ferr, float *berr,
                                  float *rpivot);

typedef int (*fb_zgbsvx_cblas_fn)(fb_layout_t layout, char fact, char trans,
                                  int n, int kl, int ku, int nrhs,
                                  fb_complex_double_t *ab, int ldab,
                                  fb_complex_double_t *afb, int ldafb,
                                  int *ipiv, char *equed, double *r,
                                  double *c, fb_complex_double_t *b, int ldb,
                                  fb_complex_double_t *x, int ldx,
                                  double *rcond, double *ferr, double *berr,
                                  double *rpivot);

typedef int (*fb_sgtsv_cblas_fn)(fb_layout_t layout, int n, int nrhs,
                                 float *dl, float *d, float *du, float *b,
                                 int ldb);

typedef int (*fb_dgtsv_cblas_fn)(fb_layout_t layout, int n, int nrhs,
                                 double *dl, double *d, double *du,
                                 double *b, int ldb);

typedef int (*fb_cgtsv_cblas_fn)(fb_layout_t layout, int n, int nrhs,
                                 fb_complex_float_t *dl,
                                 fb_complex_float_t *d,
                                 fb_complex_float_t *du,
                                 fb_complex_float_t *b, int ldb);

typedef int (*fb_zgtsv_cblas_fn)(fb_layout_t layout, int n, int nrhs,
                                 fb_complex_double_t *dl,
                                 fb_complex_double_t *d,
                                 fb_complex_double_t *du,
                                 fb_complex_double_t *b, int ldb);

typedef int (*fb_sgtsvx_cblas_fn)(fb_layout_t layout, char fact, char trans,
                                  int n, int nrhs, const float *dl,
                                  const float *d, const float *du, float *dlf,
                                  float *df, float *duf, float *du2, int *ipiv,
                                  const float *b, int ldb, float *x, int ldx,
                                  float *rcond, float *ferr, float *berr);

typedef int (*fb_dgtsvx_cblas_fn)(fb_layout_t layout, char fact, char trans,
                                  int n, int nrhs, const double *dl,
                                  const double *d, const double *du,
                                  double *dlf, double *df, double *duf,
                                  double *du2, int *ipiv, const double *b,
                                  int ldb, double *x, int ldx, double *rcond,
                                  double *ferr, double *berr);

typedef int (*fb_cgtsvx_cblas_fn)(fb_layout_t layout, char fact, char trans,
                                  int n, int nrhs,
                                  const fb_complex_float_t *dl,
                                  const fb_complex_float_t *d,
                                  const fb_complex_float_t *du,
                                  fb_complex_float_t *dlf,
                                  fb_complex_float_t *df,
                                  fb_complex_float_t *duf,
                                  fb_complex_float_t *du2, int *ipiv,
                                  const fb_complex_float_t *b, int ldb,
                                  fb_complex_float_t *x, int ldx, float *rcond,
                                  float *ferr, float *berr);

typedef int (*fb_zgtsvx_cblas_fn)(fb_layout_t layout, char fact, char trans,
                                  int n, int nrhs,
                                  const fb_complex_double_t *dl,
                                  const fb_complex_double_t *d,
                                  const fb_complex_double_t *du,
                                  fb_complex_double_t *dlf,
                                  fb_complex_double_t *df,
                                  fb_complex_double_t *duf,
                                  fb_complex_double_t *du2, int *ipiv,
                                  const fb_complex_double_t *b, int ldb,
                                  fb_complex_double_t *x, int ldx,
                                  double *rcond, double *ferr, double *berr);

typedef int (*fb_sptsv_cblas_fn)(fb_layout_t layout, int n, int nrhs, float *d,
                                 float *e, float *b, int ldb);
typedef int (*fb_dptsv_cblas_fn)(fb_layout_t layout, int n, int nrhs,
                                 double *d, double *e, double *b, int ldb);
typedef int (*fb_cptsv_cblas_fn)(fb_layout_t layout, int n, int nrhs, float *d,
                                 fb_complex_float_t *e,
                                 fb_complex_float_t *b, int ldb);
typedef int (*fb_zptsv_cblas_fn)(fb_layout_t layout, int n, int nrhs,
                                 double *d, fb_complex_double_t *e,
                                 fb_complex_double_t *b, int ldb);

typedef int (*fb_sptsvx_cblas_fn)(fb_layout_t layout, char fact, int n,
                                  int nrhs, const float *d, const float *e,
                                  float *df, float *ef, const float *b,
                                  int ldb, float *x, int ldx, float *rcond,
                                  float *ferr, float *berr);
typedef int (*fb_dptsvx_cblas_fn)(fb_layout_t layout, char fact, int n,
                                  int nrhs, const double *d, const double *e,
                                  double *df, double *ef, const double *b,
                                  int ldb, double *x, int ldx, double *rcond,
                                  double *ferr, double *berr);
typedef int (*fb_cptsvx_cblas_fn)(fb_layout_t layout, char fact, int n,
                                  int nrhs, const float *d,
                                  const fb_complex_float_t *e, float *df,
                                  fb_complex_float_t *ef,
                                  const fb_complex_float_t *b, int ldb,
                                  fb_complex_float_t *x, int ldx, float *rcond,
                                  float *ferr, float *berr);
typedef int (*fb_zptsvx_cblas_fn)(fb_layout_t layout, char fact, int n,
                                  int nrhs, const double *d,
                                  const fb_complex_double_t *e, double *df,
                                  fb_complex_double_t *ef,
                                  const fb_complex_double_t *b, int ldb,
                                  fb_complex_double_t *x, int ldx,
                                  double *rcond, double *ferr, double *berr);

typedef int (*fb_spbsv_cblas_fn)(fb_layout_t layout, char uplo, int n, int kd,
                                 int nrhs, float *ab, int ldab, float *b,
                                 int ldb);
typedef int (*fb_dpbsv_cblas_fn)(fb_layout_t layout, char uplo, int n, int kd,
                                 int nrhs, double *ab, int ldab, double *b,
                                 int ldb);
typedef int (*fb_cpbsv_cblas_fn)(fb_layout_t layout, char uplo, int n, int kd,
                                 int nrhs, fb_complex_float_t *ab, int ldab,
                                 fb_complex_float_t *b, int ldb);
typedef int (*fb_zpbsv_cblas_fn)(fb_layout_t layout, char uplo, int n, int kd,
                                 int nrhs, fb_complex_double_t *ab, int ldab,
                                 fb_complex_double_t *b, int ldb);

typedef int (*fb_spbsvx_cblas_fn)(fb_layout_t layout, char fact, char uplo,
                                  int n, int kd, int nrhs, float *ab,
                                  int ldab, float *afb, int ldafb,
                                  char *equed, float *s, float *b, int ldb,
                                  float *x, int ldx, float *rcond,
                                  float *ferr, float *berr);
typedef int (*fb_dpbsvx_cblas_fn)(fb_layout_t layout, char fact, char uplo,
                                  int n, int kd, int nrhs, double *ab,
                                  int ldab, double *afb, int ldafb,
                                  char *equed, double *s, double *b, int ldb,
                                  double *x, int ldx, double *rcond,
                                  double *ferr, double *berr);
typedef int (*fb_cpbsvx_cblas_fn)(fb_layout_t layout, char fact, char uplo,
                                  int n, int kd, int nrhs,
                                  fb_complex_float_t *ab, int ldab,
                                  fb_complex_float_t *afb, int ldafb,
                                  char *equed, float *s,
                                  fb_complex_float_t *b, int ldb,
                                  fb_complex_float_t *x, int ldx,
                                  float *rcond, float *ferr, float *berr);
typedef int (*fb_zpbsvx_cblas_fn)(fb_layout_t layout, char fact, char uplo,
                                  int n, int kd, int nrhs,
                                  fb_complex_double_t *ab, int ldab,
                                  fb_complex_double_t *afb, int ldafb,
                                  char *equed, double *s,
                                  fb_complex_double_t *b, int ldb,
                                  fb_complex_double_t *x, int ldx,
                                  double *rcond, double *ferr, double *berr);
typedef int (*fb_sppsv_cblas_fn)(fb_layout_t layout, char uplo, int n,
                                 int nrhs, float *ap, float *b, int ldb);
typedef int (*fb_dppsv_cblas_fn)(fb_layout_t layout, char uplo, int n,
                                 int nrhs, double *ap, double *b, int ldb);
typedef int (*fb_cppsv_cblas_fn)(fb_layout_t layout, char uplo, int n,
                                 int nrhs, fb_complex_float_t *ap,
                                 fb_complex_float_t *b, int ldb);
typedef int (*fb_zppsv_cblas_fn)(fb_layout_t layout, char uplo, int n,
                                 int nrhs, fb_complex_double_t *ap,
                                 fb_complex_double_t *b, int ldb);

typedef int (*fb_sppsvx_cblas_fn)(fb_layout_t layout, char fact, char uplo,
                                  int n, int nrhs, float *ap, float *afp,
                                  char *equed, float *s, float *b, int ldb,
                                  float *x, int ldx, float *rcond,
                                  float *ferr, float *berr);
typedef int (*fb_dppsvx_cblas_fn)(fb_layout_t layout, char fact, char uplo,
                                  int n, int nrhs, double *ap, double *afp,
                                  char *equed, double *s, double *b, int ldb,
                                  double *x, int ldx, double *rcond,
                                  double *ferr, double *berr);
typedef int (*fb_cppsvx_cblas_fn)(fb_layout_t layout, char fact, char uplo,
                                  int n, int nrhs, fb_complex_float_t *ap,
                                  fb_complex_float_t *afp, char *equed,
                                  float *s, fb_complex_float_t *b, int ldb,
                                  fb_complex_float_t *x, int ldx,
                                  float *rcond, float *ferr, float *berr);
typedef int (*fb_zppsvx_cblas_fn)(fb_layout_t layout, char fact, char uplo,
                                  int n, int nrhs, fb_complex_double_t *ap,
                                  fb_complex_double_t *afp, char *equed,
                                  double *s, fb_complex_double_t *b, int ldb,
                                  fb_complex_double_t *x, int ldx,
                                  double *rcond, double *ferr, double *berr);
typedef int (*fb_sspsv_cblas_fn)(fb_layout_t layout, char uplo, int n,
                                 int nrhs, float *ap, int *ipiv, float *b,
                                 int ldb);
typedef int (*fb_dspsv_cblas_fn)(fb_layout_t layout, char uplo, int n,
                                 int nrhs, double *ap, int *ipiv, double *b,
                                 int ldb);
typedef int (*fb_cspsv_cblas_fn)(fb_layout_t layout, char uplo, int n,
                                 int nrhs, fb_complex_float_t *ap, int *ipiv,
                                 fb_complex_float_t *b, int ldb);
typedef int (*fb_zspsv_cblas_fn)(fb_layout_t layout, char uplo, int n,
                                 int nrhs, fb_complex_double_t *ap, int *ipiv,
                                 fb_complex_double_t *b, int ldb);

typedef int (*fb_sspsvx_cblas_fn)(fb_layout_t layout, char fact, char uplo,
                                  int n, int nrhs, const float *ap,
                                  float *afp, int *ipiv, const float *b,
                                  int ldb, float *x, int ldx, float *rcond,
                                  float *ferr, float *berr);
typedef int (*fb_dspsvx_cblas_fn)(fb_layout_t layout, char fact, char uplo,
                                  int n, int nrhs, const double *ap,
                                  double *afp, int *ipiv, const double *b,
                                  int ldb, double *x, int ldx, double *rcond,
                                  double *ferr, double *berr);
typedef int (*fb_cspsvx_cblas_fn)(fb_layout_t layout, char fact, char uplo,
                                  int n, int nrhs,
                                  const fb_complex_float_t *ap,
                                  fb_complex_float_t *afp, int *ipiv,
                                  const fb_complex_float_t *b, int ldb,
                                  fb_complex_float_t *x, int ldx,
                                  float *rcond, float *ferr, float *berr);
typedef int (*fb_zspsvx_cblas_fn)(fb_layout_t layout, char fact, char uplo,
                                  int n, int nrhs,
                                  const fb_complex_double_t *ap,
                                  fb_complex_double_t *afp, int *ipiv,
                                  const fb_complex_double_t *b, int ldb,
                                  fb_complex_double_t *x, int ldx,
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
static int g_spbsvx_bridge_calls = 0;
static int g_dpbsvx_bridge_calls = 0;
static int g_cpbsvx_bridge_calls = 0;
static int g_zpbsvx_bridge_calls = 0;
static int g_sppsvx_bridge_calls = 0;
static int g_dppsvx_bridge_calls = 0;
static int g_cppsvx_bridge_calls = 0;
static int g_zppsvx_bridge_calls = 0;
static int g_sspsvx_bridge_calls = 0;
static int g_dspsvx_bridge_calls = 0;
static int g_cspsvx_bridge_calls = 0;
static int g_zspsvx_bridge_calls = 0;
static int g_sstevx_bridge_calls = 0;
static int g_dstevr_bridge_calls = 0;
static int g_sstevd_bridge_calls = 0;
static int g_dstevd_bridge_calls = 0;
static int g_spbsv_existing_calls = 0;
static int g_dpbsv_existing_calls = 0;
static int g_cpbsv_existing_calls = 0;
static int g_zpbsv_existing_calls = 0;
static int g_sppsv_existing_calls = 0;
static int g_dppsv_existing_calls = 0;
static int g_cppsv_existing_calls = 0;
static int g_zppsv_existing_calls = 0;
static int g_sspsv_existing_calls = 0;
static int g_dspsv_existing_calls = 0;
static int g_cspsv_existing_calls = 0;
static int g_zspsv_existing_calls = 0;
static int g_sgeevx_bridge_calls = 0;
static int g_dgeevx_bridge_calls = 0;
static int g_cgeevx_bridge_calls = 0;
static int g_zgeevx_bridge_calls = 0;
static int g_sgeev_existing_calls = 0;
static int g_sgesv_existing_calls = 0;
static int g_dgesv_existing_calls = 0;
static int g_cgesv_existing_calls = 0;
static int g_spbsvx_isolation_a_calls = 0;
static int g_spbsvx_isolation_b_calls = 0;
static int g_dpbsvx_isolation_a_calls = 0;
static int g_dpbsvx_isolation_b_calls = 0;
static int g_cpbsvx_isolation_a_calls = 0;
static int g_cpbsvx_isolation_b_calls = 0;
static int g_zpbsvx_isolation_a_calls = 0;
static int g_zpbsvx_isolation_b_calls = 0;
static int g_sppsvx_isolation_a_calls = 0;
static int g_sppsvx_isolation_b_calls = 0;
static int g_dppsvx_isolation_a_calls = 0;
static int g_dppsvx_isolation_b_calls = 0;
static int g_cppsvx_isolation_a_calls = 0;
static int g_cppsvx_isolation_b_calls = 0;
static int g_zppsvx_isolation_a_calls = 0;
static int g_zppsvx_isolation_b_calls = 0;
static int g_sspsvx_isolation_a_calls = 0;
static int g_sspsvx_isolation_b_calls = 0;
static int g_dspsvx_isolation_a_calls = 0;
static int g_dspsvx_isolation_b_calls = 0;
static int g_cspsvx_isolation_a_calls = 0;
static int g_cspsvx_isolation_b_calls = 0;
static int g_zspsvx_isolation_a_calls = 0;
static int g_zspsvx_isolation_b_calls = 0;
static int g_zgesv_existing_calls = 0;
static int g_sgesvx_bridge_calls = 0;
static int g_dgesvx_bridge_calls = 0;
static int g_cgesvx_bridge_calls = 0;
static int g_zgesvx_bridge_calls = 0;
static int g_sgesvxx_bridge_calls = 0;
static int g_dgesvxx_bridge_calls = 0;
static int g_cgesvxx_bridge_calls = 0;
static int g_zgesvxx_bridge_calls = 0;
static int g_sgesvx_isolation_a_calls = 0;
static int g_sgesvx_isolation_b_calls = 0;
static int g_dgesvx_isolation_a_calls = 0;
static int g_dgesvx_isolation_b_calls = 0;
static int g_sgesvxx_isolation_a_calls = 0;
static int g_sgesvxx_isolation_b_calls = 0;
static int g_dgesvxx_isolation_a_calls = 0;
static int g_dgesvxx_isolation_b_calls = 0;
static int g_cgesvxx_isolation_a_calls = 0;
static int g_cgesvxx_isolation_b_calls = 0;
static int g_zgesvxx_isolation_a_calls = 0;
static int g_zgesvxx_isolation_b_calls = 0;
static int g_cgesvx_isolation_a_calls = 0;
static int g_cgesvx_isolation_b_calls = 0;
static int g_zgesvx_isolation_a_calls = 0;
static int g_zgesvx_isolation_b_calls = 0;
static int g_sposvx_bridge_calls = 0;
static int g_dposvx_bridge_calls = 0;
static int g_cposvx_bridge_calls = 0;
static int g_zposvx_bridge_calls = 0;
static int g_sposvxx_bridge_calls = 0;
static int g_dposvxx_bridge_calls = 0;
static int g_cposvxx_bridge_calls = 0;
static int g_zposvxx_bridge_calls = 0;
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
static int g_sposvxx_isolation_a_calls = 0;
static int g_sposvxx_isolation_b_calls = 0;
static int g_dposvxx_isolation_a_calls = 0;
static int g_dposvxx_isolation_b_calls = 0;
static int g_cposvxx_isolation_a_calls = 0;
static int g_cposvxx_isolation_b_calls = 0;
static int g_zposvxx_isolation_a_calls = 0;
static int g_zposvxx_isolation_b_calls = 0;
static int g_ssysvx_bridge_calls = 0;
static int g_dsysvx_bridge_calls = 0;
static int g_csysvx_bridge_calls = 0;
static int g_zsysvx_bridge_calls = 0;
static int g_ssysvxx_bridge_calls = 0;
static int g_dsysvxx_bridge_calls = 0;
static int g_csysvxx_bridge_calls = 0;
static int g_zsysvxx_bridge_calls = 0;
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
static int g_ssysvxx_isolation_a_calls = 0;
static int g_ssysvxx_isolation_b_calls = 0;
static int g_dsysvxx_isolation_a_calls = 0;
static int g_dsysvxx_isolation_b_calls = 0;
static int g_csysvxx_isolation_a_calls = 0;
static int g_csysvxx_isolation_b_calls = 0;
static int g_zsysvxx_isolation_a_calls = 0;
static int g_zsysvxx_isolation_b_calls = 0;
static int g_sgbsvx_bridge_calls = 0;
static int g_dgbsvx_bridge_calls = 0;
static int g_cgbsvx_bridge_calls = 0;
static int g_zgbsvx_bridge_calls = 0;
static int g_sgtsvx_bridge_calls = 0;
static int g_dgtsvx_bridge_calls = 0;
static int g_cgtsvx_bridge_calls = 0;
static int g_zgtsvx_bridge_calls = 0;
static int g_sptsvx_bridge_calls = 0;
static int g_dptsvx_bridge_calls = 0;
static int g_cptsvx_bridge_calls = 0;
static int g_zptsvx_bridge_calls = 0;
static int g_sptsv_existing_calls = 0;
static int g_dptsv_existing_calls = 0;
static int g_cptsv_existing_calls = 0;
static int g_zptsv_existing_calls = 0;
static int g_sptsvx_isolation_a_calls = 0;
static int g_sptsvx_isolation_b_calls = 0;
static int g_dptsvx_isolation_a_calls = 0;
static int g_dptsvx_isolation_b_calls = 0;
static int g_cptsvx_isolation_a_calls = 0;
static int g_cptsvx_isolation_b_calls = 0;
static int g_zptsvx_isolation_a_calls = 0;
static int g_zptsvx_isolation_b_calls = 0;
static int g_sgtsvx_isolation_a_calls = 0;
static int g_sgtsvx_isolation_b_calls = 0;
static int g_dgtsvx_isolation_a_calls = 0;
static int g_dgtsvx_isolation_b_calls = 0;
static int g_cgtsvx_isolation_a_calls = 0;
static int g_cgtsvx_isolation_b_calls = 0;
static int g_zgtsvx_isolation_a_calls = 0;
static int g_zgtsvx_isolation_b_calls = 0;
static int g_sgbsvx_isolation_a_calls = 0;
static int g_sgbsvx_isolation_b_calls = 0;
static int g_dgbsvx_isolation_a_calls = 0;
static int g_dgbsvx_isolation_b_calls = 0;
static int g_cgbsvx_isolation_a_calls = 0;
static int g_cgbsvx_isolation_b_calls = 0;
static int g_zgbsvx_isolation_a_calls = 0;
static int g_zgbsvx_isolation_b_calls = 0;
static int g_sgbsv_existing_calls = 0;
static int g_dgbsv_existing_calls = 0;
static int g_cgbsv_existing_calls = 0;
static int g_zgbsv_existing_calls = 0;
static int g_sgtsv_existing_calls = 0;
static int g_dgtsv_existing_calls = 0;
static int g_cgtsv_existing_calls = 0;
static int g_zgtsv_existing_calls = 0;
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

static int stub_sgesvxx_base(fb_layout_t layout, char fact, char trans, int n,
                             int nrhs, float *a, int lda, float *af, int ldaf,
                             int *ipiv, char *equed, float *r, float *c,
                             float *b, int ldb, float *x, int ldx,
                             float *rcond, float *rpvgrw, float *berr,
                             int n_err_bnds, float *err_bnds_norm,
                             float *err_bnds_comp, int nparams,
                             float *params) {
  (void)layout;
  (void)err_bnds_norm;
  (void)err_bnds_comp;
  (void)params;
  g_sgesvxx_bridge_calls += 1;

  if (fact != 'N' || trans != 'N' || !a || !af || !ipiv || !equed || !r || !c ||
      !b || !x || !rcond || !rpvgrw || !berr || lda != ldaf || ldb != ldx ||
      n_err_bnds != 0 || nparams != 0) {
    return -1916;
  }

  memcpy(af, a, (size_t)lda * (size_t)((n > 0) ? n : 1) * sizeof(float));
  memcpy(x, b, (size_t)ldb * (size_t)((nrhs > 0) ? nrhs : 1) * sizeof(float));
  if (n > 0) {
    ipiv[0] = 17;
  }
  *equed = 'N';
  *rcond = 1.0f;
  *rpvgrw = 1.0f;
  return 0;
}

static int stub_dgesvxx_base(fb_layout_t layout, char fact, char trans, int n,
                             int nrhs, double *a, int lda, double *af,
                             int ldaf, int *ipiv, char *equed, double *r,
                             double *c, double *b, int ldb, double *x,
                             int ldx, double *rcond, double *rpvgrw,
                             double *berr, int n_err_bnds,
                             double *err_bnds_norm, double *err_bnds_comp,
                             int nparams, double *params) {
  (void)layout;
  (void)err_bnds_norm;
  (void)err_bnds_comp;
  (void)params;
  g_dgesvxx_bridge_calls += 1;

  if (fact != 'N' || trans != 'N' || !a || !af || !ipiv || !equed || !r || !c ||
      !b || !x || !rcond || !rpvgrw || !berr || lda != ldaf || ldb != ldx ||
      n_err_bnds != 0 || nparams != 0) {
    return -2017;
  }

  memcpy(af, a, (size_t)lda * (size_t)((n > 0) ? n : 1) * sizeof(double));
  memcpy(x, b, (size_t)ldb * (size_t)((nrhs > 0) ? nrhs : 1) * sizeof(double));
  if (n > 0) {
    ipiv[0] = 19;
  }
  *equed = 'N';
  *rcond = 1.0;
  *rpvgrw = 1.0;
  return 0;
}

static int stub_sgesvxx_isolation_a(
    fb_layout_t layout, char fact, char trans, int n, int nrhs, float *a,
    int lda, float *af, int ldaf, int *ipiv, char *equed, float *r, float *c,
    float *b, int ldb, float *x, int ldx, float *rcond, float *rpvgrw,
    float *berr, int n_err_bnds, float *err_bnds_norm, float *err_bnds_comp,
    int nparams, float *params) {
  (void)layout; (void)n; (void)nrhs; (void)a; (void)lda; (void)af;
  (void)ldaf; (void)ipiv; (void)equed; (void)r; (void)c; (void)b;
  (void)ldb; (void)x; (void)ldx; (void)rcond; (void)rpvgrw; (void)berr;
  (void)n_err_bnds; (void)err_bnds_norm; (void)err_bnds_comp;
  (void)nparams; (void)params;
  g_sgesvxx_isolation_a_calls += 1;
  if (fact != 'N' || trans != 'N') {
    return -1;
  }
  return 1303;
}

static int stub_sgesvxx_isolation_b(
    fb_layout_t layout, char fact, char trans, int n, int nrhs, float *a,
    int lda, float *af, int ldaf, int *ipiv, char *equed, float *r, float *c,
    float *b, int ldb, float *x, int ldx, float *rcond, float *rpvgrw,
    float *berr, int n_err_bnds, float *err_bnds_norm, float *err_bnds_comp,
    int nparams, float *params) {
  (void)layout; (void)n; (void)nrhs; (void)a; (void)lda; (void)af;
  (void)ldaf; (void)ipiv; (void)equed; (void)r; (void)c; (void)b;
  (void)ldb; (void)x; (void)ldx; (void)rcond; (void)rpvgrw; (void)berr;
  (void)n_err_bnds; (void)err_bnds_norm; (void)err_bnds_comp;
  (void)nparams; (void)params;
  g_sgesvxx_isolation_b_calls += 1;
  if (fact != 'N' || trans != 'N') {
    return -1;
  }
  return 1304;
}

static int stub_dgesvxx_isolation_a(
    fb_layout_t layout, char fact, char trans, int n, int nrhs, double *a,
    int lda, double *af, int ldaf, int *ipiv, char *equed, double *r,
    double *c, double *b, int ldb, double *x, int ldx, double *rcond,
    double *rpvgrw, double *berr, int n_err_bnds, double *err_bnds_norm,
    double *err_bnds_comp, int nparams, double *params) {
  (void)layout; (void)n; (void)nrhs; (void)a; (void)lda; (void)af;
  (void)ldaf; (void)ipiv; (void)equed; (void)r; (void)c; (void)b;
  (void)ldb; (void)x; (void)ldx; (void)rcond; (void)rpvgrw; (void)berr;
  (void)n_err_bnds; (void)err_bnds_norm; (void)err_bnds_comp;
  (void)nparams; (void)params;
  g_dgesvxx_isolation_a_calls += 1;
  if (fact != 'N' || trans != 'N') {
    return -1;
  }
  return 1305;
}

static int stub_dgesvxx_isolation_b(
    fb_layout_t layout, char fact, char trans, int n, int nrhs, double *a,
    int lda, double *af, int ldaf, int *ipiv, char *equed, double *r,
    double *c, double *b, int ldb, double *x, int ldx, double *rcond,
    double *rpvgrw, double *berr, int n_err_bnds, double *err_bnds_norm,
    double *err_bnds_comp, int nparams, double *params) {
  (void)layout; (void)n; (void)nrhs; (void)a; (void)lda; (void)af;
  (void)ldaf; (void)ipiv; (void)equed; (void)r; (void)c; (void)b;
  (void)ldb; (void)x; (void)ldx; (void)rcond; (void)rpvgrw; (void)berr;
  (void)n_err_bnds; (void)err_bnds_norm; (void)err_bnds_comp;
  (void)nparams; (void)params;
  g_dgesvxx_isolation_b_calls += 1;
  if (fact != 'N' || trans != 'N') {
    return -1;
  }
  return 1306;
}

static int stub_cgesvxx_base(
    fb_layout_t layout, char fact, char trans, int n, int nrhs,
    fb_complex_float_t *a, int lda, fb_complex_float_t *af, int ldaf,
    int *ipiv, char *equed, float *r, float *c, fb_complex_float_t *b,
    int ldb, fb_complex_float_t *x, int ldx, float *rcond, float *rpvgrw,
    float *berr, int n_err_bnds, float *err_bnds_norm, float *err_bnds_comp,
    int nparams, float *params) {
  (void)layout;
  (void)err_bnds_norm;
  (void)err_bnds_comp;
  (void)params;
  g_cgesvxx_bridge_calls += 1;

  if (fact != 'N' || trans != 'N' || !a || !af || !ipiv || !equed || !r || !c ||
      !b || !x || !rcond || !rpvgrw || !berr || lda != ldaf || ldb != ldx ||
      n_err_bnds != 0 || nparams != 0) {
    return -2118;
  }

  memcpy(af, a,
         (size_t)lda * (size_t)((n > 0) ? n : 1) * sizeof(fb_complex_float_t));
  memcpy(x, b, (size_t)ldb * (size_t)((nrhs > 0) ? nrhs : 1) *
                   sizeof(fb_complex_float_t));
  if (n > 0) {
    ipiv[0] = 21;
  }
  *equed = 'N';
  *rcond = 1.0f;
  *rpvgrw = 1.0f;
  return 0;
}

static int stub_zgesvxx_base(
    fb_layout_t layout, char fact, char trans, int n, int nrhs,
    fb_complex_double_t *a, int lda, fb_complex_double_t *af, int ldaf,
    int *ipiv, char *equed, double *r, double *c, fb_complex_double_t *b,
    int ldb, fb_complex_double_t *x, int ldx, double *rcond,
    double *rpvgrw, double *berr, int n_err_bnds, double *err_bnds_norm,
    double *err_bnds_comp, int nparams, double *params) {
  (void)layout;
  (void)err_bnds_norm;
  (void)err_bnds_comp;
  (void)params;
  g_zgesvxx_bridge_calls += 1;

  if (fact != 'N' || trans != 'N' || !a || !af || !ipiv || !equed || !r || !c ||
      !b || !x || !rcond || !rpvgrw || !berr || lda != ldaf || ldb != ldx ||
      n_err_bnds != 0 || nparams != 0) {
    return -2219;
  }

  memcpy(af, a, (size_t)lda * (size_t)((n > 0) ? n : 1) *
                   sizeof(fb_complex_double_t));
  memcpy(x, b, (size_t)ldb * (size_t)((nrhs > 0) ? nrhs : 1) *
                   sizeof(fb_complex_double_t));
  if (n > 0) {
    ipiv[0] = 23;
  }
  *equed = 'N';
  *rcond = 1.0;
  *rpvgrw = 1.0;
  return 0;
}

static int stub_cgesvxx_isolation_a(
    fb_layout_t layout, char fact, char trans, int n, int nrhs,
    fb_complex_float_t *a, int lda, fb_complex_float_t *af, int ldaf,
    int *ipiv, char *equed, float *r, float *c, fb_complex_float_t *b,
    int ldb, fb_complex_float_t *x, int ldx, float *rcond, float *rpvgrw,
    float *berr, int n_err_bnds, float *err_bnds_norm, float *err_bnds_comp,
    int nparams, float *params) {
  (void)layout; (void)n; (void)nrhs; (void)a; (void)lda; (void)af;
  (void)ldaf; (void)ipiv; (void)equed; (void)r; (void)c; (void)b;
  (void)ldb; (void)x; (void)ldx; (void)rcond; (void)rpvgrw; (void)berr;
  (void)n_err_bnds; (void)err_bnds_norm; (void)err_bnds_comp;
  (void)nparams; (void)params;
  g_cgesvxx_isolation_a_calls += 1;
  if (fact != 'N' || trans != 'N') {
    return -1;
  }
  return 1307;
}

static int stub_cgesvxx_isolation_b(
    fb_layout_t layout, char fact, char trans, int n, int nrhs,
    fb_complex_float_t *a, int lda, fb_complex_float_t *af, int ldaf,
    int *ipiv, char *equed, float *r, float *c, fb_complex_float_t *b,
    int ldb, fb_complex_float_t *x, int ldx, float *rcond, float *rpvgrw,
    float *berr, int n_err_bnds, float *err_bnds_norm, float *err_bnds_comp,
    int nparams, float *params) {
  (void)layout; (void)n; (void)nrhs; (void)a; (void)lda; (void)af;
  (void)ldaf; (void)ipiv; (void)equed; (void)r; (void)c; (void)b;
  (void)ldb; (void)x; (void)ldx; (void)rcond; (void)rpvgrw; (void)berr;
  (void)n_err_bnds; (void)err_bnds_norm; (void)err_bnds_comp;
  (void)nparams; (void)params;
  g_cgesvxx_isolation_b_calls += 1;
  if (fact != 'N' || trans != 'N') {
    return -1;
  }
  return 1308;
}

static int stub_zgesvxx_isolation_a(
    fb_layout_t layout, char fact, char trans, int n, int nrhs,
    fb_complex_double_t *a, int lda, fb_complex_double_t *af, int ldaf,
    int *ipiv, char *equed, double *r, double *c, fb_complex_double_t *b,
    int ldb, fb_complex_double_t *x, int ldx, double *rcond,
    double *rpvgrw, double *berr, int n_err_bnds, double *err_bnds_norm,
    double *err_bnds_comp, int nparams, double *params) {
  (void)layout; (void)n; (void)nrhs; (void)a; (void)lda; (void)af;
  (void)ldaf; (void)ipiv; (void)equed; (void)r; (void)c; (void)b;
  (void)ldb; (void)x; (void)ldx; (void)rcond; (void)rpvgrw; (void)berr;
  (void)n_err_bnds; (void)err_bnds_norm; (void)err_bnds_comp;
  (void)nparams; (void)params;
  g_zgesvxx_isolation_a_calls += 1;
  if (fact != 'N' || trans != 'N') {
    return -1;
  }
  return 1309;
}

static int stub_zgesvxx_isolation_b(
    fb_layout_t layout, char fact, char trans, int n, int nrhs,
    fb_complex_double_t *a, int lda, fb_complex_double_t *af, int ldaf,
    int *ipiv, char *equed, double *r, double *c, fb_complex_double_t *b,
    int ldb, fb_complex_double_t *x, int ldx, double *rcond,
    double *rpvgrw, double *berr, int n_err_bnds, double *err_bnds_norm,
    double *err_bnds_comp, int nparams, double *params) {
  (void)layout; (void)n; (void)nrhs; (void)a; (void)lda; (void)af;
  (void)ldaf; (void)ipiv; (void)equed; (void)r; (void)c; (void)b;
  (void)ldb; (void)x; (void)ldx; (void)rcond; (void)rpvgrw; (void)berr;
  (void)n_err_bnds; (void)err_bnds_norm; (void)err_bnds_comp;
  (void)nparams; (void)params;
  g_zgesvxx_isolation_b_calls += 1;
  if (fact != 'N' || trans != 'N') {
    return -1;
  }
  return 1310;
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

static int stub_sposvxx_base(fb_layout_t layout, char fact, char uplo, int n,
                             int nrhs, float *a, int lda, float *af,
                             int ldaf, char *equed, float *s, float *b,
                             int ldb, float *x, int ldx, float *rcond,
                             float *rpvgrw, float *berr, int n_err_bnds,
                             float *err_bnds_norm, float *err_bnds_comp,
                             int nparams, float *params) {
  (void)layout;
  g_sposvxx_bridge_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER || !a || !af || !equed || !s ||
      !b || !x || !rcond || !rpvgrw || !berr || lda != ldaf || ldb != ldx ||
      n_err_bnds != 0 || err_bnds_norm != NULL || err_bnds_comp != NULL ||
      nparams != 0 || params != NULL) {
    return -3110;
  }
  memcpy(af, a, (size_t)lda * (size_t)((n > 0) ? n : 1) * sizeof(float));
  memcpy(x, b, (size_t)ldb * (size_t)((nrhs > 0) ? nrhs : 1) * sizeof(float));
  *equed = 'N';
  *rcond = 1.0f;
  *rpvgrw = 1.0f;
  return 0;
}

static int stub_dposvxx_base(fb_layout_t layout, char fact, char uplo, int n,
                             int nrhs, double *a, int lda, double *af,
                             int ldaf, char *equed, double *s, double *b,
                             int ldb, double *x, int ldx, double *rcond,
                             double *rpvgrw, double *berr, int n_err_bnds,
                             double *err_bnds_norm, double *err_bnds_comp,
                             int nparams, double *params) {
  (void)layout;
  g_dposvxx_bridge_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER || !a || !af || !equed || !s ||
      !b || !x || !rcond || !rpvgrw || !berr || lda != ldaf || ldb != ldx ||
      n_err_bnds != 0 || err_bnds_norm != NULL || err_bnds_comp != NULL ||
      nparams != 0 || params != NULL) {
    return -3211;
  }
  memcpy(af, a,
         (size_t)lda * (size_t)((n > 0) ? n : 1) * sizeof(double));
  memcpy(x, b,
         (size_t)ldb * (size_t)((nrhs > 0) ? nrhs : 1) * sizeof(double));
  *equed = 'N';
  *rcond = 1.0;
  *rpvgrw = 1.0;
  return 0;
}

static int stub_cposvxx_base(fb_layout_t layout, char fact, char uplo, int n,
                             int nrhs, fb_complex_float_t *a, int lda,
                             fb_complex_float_t *af, int ldaf, char *equed,
                             float *s, fb_complex_float_t *b, int ldb,
                             fb_complex_float_t *x, int ldx, float *rcond,
                             float *rpvgrw, float *berr, int n_err_bnds,
                             float *err_bnds_norm, float *err_bnds_comp,
                             int nparams, float *params) {
  (void)layout;
  g_cposvxx_bridge_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER || !a || !af || !equed || !s ||
      !b || !x || !rcond || !rpvgrw || !berr || lda != ldaf || ldb != ldx ||
      n_err_bnds != 0 || err_bnds_norm != NULL || err_bnds_comp != NULL ||
      nparams != 0 || params != NULL) {
    return -3312;
  }
  memcpy(af, a,
         (size_t)lda * (size_t)((n > 0) ? n : 1) * sizeof(fb_complex_float_t));
  memcpy(x, b,
         (size_t)ldb * (size_t)((nrhs > 0) ? nrhs : 1) * sizeof(fb_complex_float_t));
  *equed = 'N';
  *rcond = 1.0f;
  *rpvgrw = 1.0f;
  return 0;
}

static int stub_zposvxx_base(fb_layout_t layout, char fact, char uplo, int n,
                             int nrhs, fb_complex_double_t *a, int lda,
                             fb_complex_double_t *af, int ldaf, char *equed,
                             double *s, fb_complex_double_t *b, int ldb,
                             fb_complex_double_t *x, int ldx, double *rcond,
                             double *rpvgrw, double *berr, int n_err_bnds,
                             double *err_bnds_norm, double *err_bnds_comp,
                             int nparams, double *params) {
  (void)layout;
  g_zposvxx_bridge_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER || !a || !af || !equed || !s ||
      !b || !x || !rcond || !rpvgrw || !berr || lda != ldaf || ldb != ldx ||
      n_err_bnds != 0 || err_bnds_norm != NULL || err_bnds_comp != NULL ||
      nparams != 0 || params != NULL) {
    return -3413;
  }
  memcpy(af, a,
         (size_t)lda * (size_t)((n > 0) ? n : 1) * sizeof(fb_complex_double_t));
  memcpy(x, b,
         (size_t)ldb * (size_t)((nrhs > 0) ? nrhs : 1) * sizeof(fb_complex_double_t));
  *equed = 'N';
  *rcond = 1.0;
  *rpvgrw = 1.0;
  return 0;
}

static int stub_sposvxx_isolation_a(
    fb_layout_t layout, char fact, char uplo, int n, int nrhs, float *a,
    int lda, float *af, int ldaf, char *equed, float *s, float *b, int ldb,
    float *x, int ldx, float *rcond, float *rpvgrw, float *berr,
    int n_err_bnds, float *err_bnds_norm, float *err_bnds_comp, int nparams,
    float *params) {
  (void)layout; (void)n; (void)nrhs; (void)a; (void)lda; (void)af; (void)ldaf;
  (void)equed; (void)s; (void)b; (void)ldb; (void)x; (void)ldx; (void)rcond;
  (void)rpvgrw; (void)berr; (void)n_err_bnds; (void)err_bnds_norm;
  (void)err_bnds_comp; (void)nparams; (void)params;
  g_sposvxx_isolation_a_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER) {
    return -1;
  }
  return 1321;
}

static int stub_sposvxx_isolation_b(
    fb_layout_t layout, char fact, char uplo, int n, int nrhs, float *a,
    int lda, float *af, int ldaf, char *equed, float *s, float *b, int ldb,
    float *x, int ldx, float *rcond, float *rpvgrw, float *berr,
    int n_err_bnds, float *err_bnds_norm, float *err_bnds_comp, int nparams,
    float *params) {
  (void)layout; (void)n; (void)nrhs; (void)a; (void)lda; (void)af; (void)ldaf;
  (void)equed; (void)s; (void)b; (void)ldb; (void)x; (void)ldx; (void)rcond;
  (void)rpvgrw; (void)berr; (void)n_err_bnds; (void)err_bnds_norm;
  (void)err_bnds_comp; (void)nparams; (void)params;
  g_sposvxx_isolation_b_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER) {
    return -1;
  }
  return 1322;
}

static int stub_dposvxx_isolation_a(
    fb_layout_t layout, char fact, char uplo, int n, int nrhs, double *a,
    int lda, double *af, int ldaf, char *equed, double *s, double *b, int ldb,
    double *x, int ldx, double *rcond, double *rpvgrw, double *berr,
    int n_err_bnds, double *err_bnds_norm, double *err_bnds_comp, int nparams,
    double *params) {
  (void)layout; (void)n; (void)nrhs; (void)a; (void)lda; (void)af; (void)ldaf;
  (void)equed; (void)s; (void)b; (void)ldb; (void)x; (void)ldx; (void)rcond;
  (void)rpvgrw; (void)berr; (void)n_err_bnds; (void)err_bnds_norm;
  (void)err_bnds_comp; (void)nparams; (void)params;
  g_dposvxx_isolation_a_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER) {
    return -1;
  }
  return 1323;
}

static int stub_dposvxx_isolation_b(
    fb_layout_t layout, char fact, char uplo, int n, int nrhs, double *a,
    int lda, double *af, int ldaf, char *equed, double *s, double *b, int ldb,
    double *x, int ldx, double *rcond, double *rpvgrw, double *berr,
    int n_err_bnds, double *err_bnds_norm, double *err_bnds_comp, int nparams,
    double *params) {
  (void)layout; (void)n; (void)nrhs; (void)a; (void)lda; (void)af; (void)ldaf;
  (void)equed; (void)s; (void)b; (void)ldb; (void)x; (void)ldx; (void)rcond;
  (void)rpvgrw; (void)berr; (void)n_err_bnds; (void)err_bnds_norm;
  (void)err_bnds_comp; (void)nparams; (void)params;
  g_dposvxx_isolation_b_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER) {
    return -1;
  }
  return 1324;
}

static int stub_cposvxx_isolation_a(
    fb_layout_t layout, char fact, char uplo, int n, int nrhs,
    fb_complex_float_t *a, int lda, fb_complex_float_t *af, int ldaf,
    char *equed, float *s, fb_complex_float_t *b, int ldb,
    fb_complex_float_t *x, int ldx, float *rcond, float *rpvgrw, float *berr,
    int n_err_bnds, float *err_bnds_norm, float *err_bnds_comp, int nparams,
    float *params) {
  (void)layout; (void)n; (void)nrhs; (void)a; (void)lda; (void)af; (void)ldaf;
  (void)equed; (void)s; (void)b; (void)ldb; (void)x; (void)ldx; (void)rcond;
  (void)rpvgrw; (void)berr; (void)n_err_bnds; (void)err_bnds_norm;
  (void)err_bnds_comp; (void)nparams; (void)params;
  g_cposvxx_isolation_a_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER) {
    return -1;
  }
  return 1325;
}

static int stub_cposvxx_isolation_b(
    fb_layout_t layout, char fact, char uplo, int n, int nrhs,
    fb_complex_float_t *a, int lda, fb_complex_float_t *af, int ldaf,
    char *equed, float *s, fb_complex_float_t *b, int ldb,
    fb_complex_float_t *x, int ldx, float *rcond, float *rpvgrw, float *berr,
    int n_err_bnds, float *err_bnds_norm, float *err_bnds_comp, int nparams,
    float *params) {
  (void)layout; (void)n; (void)nrhs; (void)a; (void)lda; (void)af; (void)ldaf;
  (void)equed; (void)s; (void)b; (void)ldb; (void)x; (void)ldx; (void)rcond;
  (void)rpvgrw; (void)berr; (void)n_err_bnds; (void)err_bnds_norm;
  (void)err_bnds_comp; (void)nparams; (void)params;
  g_cposvxx_isolation_b_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER) {
    return -1;
  }
  return 1326;
}

static int stub_zposvxx_isolation_a(
    fb_layout_t layout, char fact, char uplo, int n, int nrhs,
    fb_complex_double_t *a, int lda, fb_complex_double_t *af, int ldaf,
    char *equed, double *s, fb_complex_double_t *b, int ldb,
    fb_complex_double_t *x, int ldx, double *rcond, double *rpvgrw,
    double *berr, int n_err_bnds, double *err_bnds_norm,
    double *err_bnds_comp, int nparams, double *params) {
  (void)layout; (void)n; (void)nrhs; (void)a; (void)lda; (void)af; (void)ldaf;
  (void)equed; (void)s; (void)b; (void)ldb; (void)x; (void)ldx; (void)rcond;
  (void)rpvgrw; (void)berr; (void)n_err_bnds; (void)err_bnds_norm;
  (void)err_bnds_comp; (void)nparams; (void)params;
  g_zposvxx_isolation_a_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER) {
    return -1;
  }
  return 1327;
}

static int stub_zposvxx_isolation_b(
    fb_layout_t layout, char fact, char uplo, int n, int nrhs,
    fb_complex_double_t *a, int lda, fb_complex_double_t *af, int ldaf,
    char *equed, double *s, fb_complex_double_t *b, int ldb,
    fb_complex_double_t *x, int ldx, double *rcond, double *rpvgrw,
    double *berr, int n_err_bnds, double *err_bnds_norm,
    double *err_bnds_comp, int nparams, double *params) {
  (void)layout; (void)n; (void)nrhs; (void)a; (void)lda; (void)af; (void)ldaf;
  (void)equed; (void)s; (void)b; (void)ldb; (void)x; (void)ldx; (void)rcond;
  (void)rpvgrw; (void)berr; (void)n_err_bnds; (void)err_bnds_norm;
  (void)err_bnds_comp; (void)nparams; (void)params;
  g_zposvxx_isolation_b_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER) {
    return -1;
  }
  return 1328;
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

static int stub_sgbsvx_base(fb_layout_t layout, char fact, char trans, int n,
                            int kl, int ku, int nrhs, float *ab, int ldab,
                            float *afb, int ldafb, int *ipiv, char *equed,
                            float *r, float *c, float *b, int ldb, float *x,
                            int ldx, float *rcond, float *ferr, float *berr,
                            float *rpivot) {
  (void)layout;
  (void)kl;
  (void)ku;
  g_sgbsvx_bridge_calls += 1;
  if (fact != 'N' || trans != 'N' || !ab || !afb || !ipiv || !equed || !r ||
      !c || !b || !x || !rcond || !ferr || !berr || !rpivot || ldab != ldafb ||
      ldb != ldx) {
    return -2718;
  }
  memcpy(afb, ab, (size_t)ldab * (size_t)((n > 0) ? n : 1) * sizeof(float));
  memcpy(x, b, (size_t)ldb * (size_t)((nrhs > 0) ? nrhs : 1) * sizeof(float));
  if (n > 0) {
    ipiv[0] = 11;
  }
  *equed = 'N';
  *rcond = 1.0f;
  return 0;
}

static int stub_dgbsvx_base(fb_layout_t layout, char fact, char trans, int n,
                            int kl, int ku, int nrhs, double *ab, int ldab,
                            double *afb, int ldafb, int *ipiv, char *equed,
                            double *r, double *c, double *b, int ldb,
                            double *x, int ldx, double *rcond, double *ferr,
                            double *berr, double *rpivot) {
  (void)layout;
  (void)kl;
  (void)ku;
  g_dgbsvx_bridge_calls += 1;
  if (fact != 'N' || trans != 'N' || !ab || !afb || !ipiv || !equed || !r ||
      !c || !b || !x || !rcond || !ferr || !berr || !rpivot || ldab != ldafb ||
      ldb != ldx) {
    return -2819;
  }
  memcpy(afb, ab, (size_t)ldab * (size_t)((n > 0) ? n : 1) * sizeof(double));
  memcpy(x, b,
         (size_t)ldb * (size_t)((nrhs > 0) ? nrhs : 1) * sizeof(double));
  if (n > 0) {
    ipiv[0] = 13;
  }
  *equed = 'N';
  *rcond = 1.0;
  return 0;
}

static int stub_cgbsvx_base(fb_layout_t layout, char fact, char trans, int n,
                            int kl, int ku, int nrhs, fb_complex_float_t *ab,
                            int ldab, fb_complex_float_t *afb, int ldafb,
                            int *ipiv, char *equed, float *r, float *c,
                            fb_complex_float_t *b, int ldb,
                            fb_complex_float_t *x, int ldx, float *rcond,
                            float *ferr, float *berr, float *rpivot) {
  (void)layout;
  (void)kl;
  (void)ku;
  g_cgbsvx_bridge_calls += 1;
  if (fact != 'N' || trans != 'N' || !ab || !afb || !ipiv || !equed || !r ||
      !c || !b || !x || !rcond || !ferr || !berr || !rpivot || ldab != ldafb ||
      ldb != ldx) {
    return -2920;
  }
  memcpy(afb, ab,
         (size_t)ldab * (size_t)((n > 0) ? n : 1) * sizeof(fb_complex_float_t));
  memcpy(x, b,
         (size_t)ldb * (size_t)((nrhs > 0) ? nrhs : 1) * sizeof(fb_complex_float_t));
  if (n > 0) {
    ipiv[0] = 15;
  }
  *equed = 'N';
  *rcond = 1.0f;
  return 0;
}

static int stub_zgbsvx_base(fb_layout_t layout, char fact, char trans, int n,
                            int kl, int ku, int nrhs,
                            fb_complex_double_t *ab, int ldab,
                            fb_complex_double_t *afb, int ldafb, int *ipiv,
                            char *equed, double *r, double *c,
                            fb_complex_double_t *b, int ldb,
                            fb_complex_double_t *x, int ldx, double *rcond,
                            double *ferr, double *berr, double *rpivot) {
  (void)layout;
  (void)kl;
  (void)ku;
  g_zgbsvx_bridge_calls += 1;
  if (fact != 'N' || trans != 'N' || !ab || !afb || !ipiv || !equed || !r ||
      !c || !b || !x || !rcond || !ferr || !berr || !rpivot || ldab != ldafb ||
      ldb != ldx) {
    return -3021;
  }
  memcpy(afb, ab,
         (size_t)ldab * (size_t)((n > 0) ? n : 1) * sizeof(fb_complex_double_t));
  memcpy(x, b,
         (size_t)ldb * (size_t)((nrhs > 0) ? nrhs : 1) * sizeof(fb_complex_double_t));
  if (n > 0) {
    ipiv[0] = 17;
  }
  *equed = 'N';
  *rcond = 1.0;
  return 0;
}

static int stub_sgtsvx_base(fb_layout_t layout, char fact, char trans, int n,
                            int nrhs, const float *dl, const float *d,
                            const float *du, float *dlf, float *df,
                            float *duf, float *du2, int *ipiv, const float *b,
                            int ldb, float *x, int ldx, float *rcond,
                            float *ferr, float *berr) {
  (void)layout;
  g_sgtsvx_bridge_calls += 1;
  if (fact != 'N' || trans != 'N' || !dl || !d || !du || !dlf || !df || !duf ||
      !du2 || !ipiv || !b || !x || !rcond || !ferr || !berr || ldb != ldx) {
    return -3501;
  }
  if (n > 0) {
    memcpy(dlf, dl, (size_t)n * sizeof(float));
    memcpy(df, d, (size_t)n * sizeof(float));
    memcpy(duf, du, (size_t)n * sizeof(float));
    ipiv[0] = 19;
  }
  memset(du2, 0, (size_t)((n > 0) ? n : 1) * sizeof(float));
  memcpy(x, b, (size_t)ldb * (size_t)((nrhs > 0) ? nrhs : 1) * sizeof(float));
  *rcond = 1.0f;
  return 0;
}

static int stub_dgtsvx_base(fb_layout_t layout, char fact, char trans, int n,
                            int nrhs, const double *dl, const double *d,
                            const double *du, double *dlf, double *df,
                            double *duf, double *du2, int *ipiv,
                            const double *b, int ldb, double *x, int ldx,
                            double *rcond, double *ferr, double *berr) {
  (void)layout;
  g_dgtsvx_bridge_calls += 1;
  if (fact != 'N' || trans != 'N' || !dl || !d || !du || !dlf || !df || !duf ||
      !du2 || !ipiv || !b || !x || !rcond || !ferr || !berr || ldb != ldx) {
    return -3502;
  }
  if (n > 0) {
    memcpy(dlf, dl, (size_t)n * sizeof(double));
    memcpy(df, d, (size_t)n * sizeof(double));
    memcpy(duf, du, (size_t)n * sizeof(double));
    ipiv[0] = 21;
  }
  memset(du2, 0, (size_t)((n > 0) ? n : 1) * sizeof(double));
  memcpy(x, b,
         (size_t)ldb * (size_t)((nrhs > 0) ? nrhs : 1) * sizeof(double));
  *rcond = 1.0;
  return 0;
}

static int stub_cgtsvx_base(fb_layout_t layout, char fact, char trans, int n,
                            int nrhs, const fb_complex_float_t *dl,
                            const fb_complex_float_t *d,
                            const fb_complex_float_t *du,
                            fb_complex_float_t *dlf, fb_complex_float_t *df,
                            fb_complex_float_t *duf, fb_complex_float_t *du2,
                            int *ipiv, const fb_complex_float_t *b, int ldb,
                            fb_complex_float_t *x, int ldx, float *rcond,
                            float *ferr, float *berr) {
  (void)layout;
  g_cgtsvx_bridge_calls += 1;
  if (fact != 'N' || trans != 'N' || !dl || !d || !du || !dlf || !df || !duf ||
      !du2 || !ipiv || !b || !x || !rcond || !ferr || !berr || ldb != ldx) {
    return -3503;
  }
  if (n > 0) {
    memcpy(dlf, dl, (size_t)n * sizeof(fb_complex_float_t));
    memcpy(df, d, (size_t)n * sizeof(fb_complex_float_t));
    memcpy(duf, du, (size_t)n * sizeof(fb_complex_float_t));
    ipiv[0] = 23;
  }
  memset(du2, 0, (size_t)((n > 0) ? n : 1) * sizeof(fb_complex_float_t));
  memcpy(x, b, (size_t)ldb * (size_t)((nrhs > 0) ? nrhs : 1) *
                   sizeof(fb_complex_float_t));
  *rcond = 1.0f;
  return 0;
}

static int stub_zgtsvx_base(fb_layout_t layout, char fact, char trans, int n,
                            int nrhs, const fb_complex_double_t *dl,
                            const fb_complex_double_t *d,
                            const fb_complex_double_t *du,
                            fb_complex_double_t *dlf, fb_complex_double_t *df,
                            fb_complex_double_t *duf, fb_complex_double_t *du2,
                            int *ipiv, const fb_complex_double_t *b, int ldb,
                            fb_complex_double_t *x, int ldx, double *rcond,
                            double *ferr, double *berr) {
  (void)layout;
  g_zgtsvx_bridge_calls += 1;
  if (fact != 'N' || trans != 'N' || !dl || !d || !du || !dlf || !df || !duf ||
      !du2 || !ipiv || !b || !x || !rcond || !ferr || !berr || ldb != ldx) {
    return -3504;
  }
  if (n > 0) {
    memcpy(dlf, dl, (size_t)n * sizeof(fb_complex_double_t));
    memcpy(df, d, (size_t)n * sizeof(fb_complex_double_t));
    memcpy(duf, du, (size_t)n * sizeof(fb_complex_double_t));
    ipiv[0] = 25;
  }
  memset(du2, 0, (size_t)((n > 0) ? n : 1) * sizeof(fb_complex_double_t));
  memcpy(x, b, (size_t)ldb * (size_t)((nrhs > 0) ? nrhs : 1) *
                   sizeof(fb_complex_double_t));
  *rcond = 1.0;
  return 0;
}

static int stub_sptsvx_base(fb_layout_t layout, char fact, int n, int nrhs,
                            const float *d, const float *e, float *df,
                            float *ef, const float *b, int ldb, float *x,
                            int ldx, float *rcond, float *ferr, float *berr) {
  (void)layout;
  g_sptsvx_bridge_calls += 1;
  if (fact != 'N' || !d || !e || !df || !ef || !b || !x || !rcond || !ferr ||
      !berr || ldb != ldx) {
    return -3601;
  }
  if (n > 0) {
    memcpy(df, d, (size_t)n * sizeof(float));
    memcpy(ef, e, (size_t)n * sizeof(float));
  }
  memcpy(x, b, (size_t)ldb * (size_t)((nrhs > 0) ? nrhs : 1) * sizeof(float));
  *rcond = 1.0f;
  return 0;
}

static int stub_dptsvx_base(fb_layout_t layout, char fact, int n, int nrhs,
                            const double *d, const double *e, double *df,
                            double *ef, const double *b, int ldb, double *x,
                            int ldx, double *rcond, double *ferr,
                            double *berr) {
  (void)layout;
  g_dptsvx_bridge_calls += 1;
  if (fact != 'N' || !d || !e || !df || !ef || !b || !x || !rcond || !ferr ||
      !berr || ldb != ldx) {
    return -3602;
  }
  if (n > 0) {
    memcpy(df, d, (size_t)n * sizeof(double));
    memcpy(ef, e, (size_t)n * sizeof(double));
  }
  memcpy(x, b,
         (size_t)ldb * (size_t)((nrhs > 0) ? nrhs : 1) * sizeof(double));
  *rcond = 1.0;
  return 0;
}

static int stub_cptsvx_base(fb_layout_t layout, char fact, int n, int nrhs,
                            const float *d, const fb_complex_float_t *e,
                            float *df, fb_complex_float_t *ef,
                            const fb_complex_float_t *b, int ldb,
                            fb_complex_float_t *x, int ldx, float *rcond,
                            float *ferr, float *berr) {
  (void)layout;
  g_cptsvx_bridge_calls += 1;
  if (fact != 'N' || !d || !e || !df || !ef || !b || !x || !rcond || !ferr ||
      !berr || ldb != ldx) {
    return -3603;
  }
  if (n > 0) {
    memcpy(df, d, (size_t)n * sizeof(float));
    memcpy(ef, e, (size_t)n * sizeof(fb_complex_float_t));
  }
  memcpy(x, b, (size_t)ldb * (size_t)((nrhs > 0) ? nrhs : 1) *
                   sizeof(fb_complex_float_t));
  *rcond = 1.0f;
  return 0;
}

static int stub_zptsvx_base(fb_layout_t layout, char fact, int n, int nrhs,
                            const double *d, const fb_complex_double_t *e,
                            double *df, fb_complex_double_t *ef,
                            const fb_complex_double_t *b, int ldb,
                            fb_complex_double_t *x, int ldx, double *rcond,
                            double *ferr, double *berr) {
  (void)layout;
  g_zptsvx_bridge_calls += 1;
  if (fact != 'N' || !d || !e || !df || !ef || !b || !x || !rcond || !ferr ||
      !berr || ldb != ldx) {
    return -3604;
  }
  if (n > 0) {
    memcpy(df, d, (size_t)n * sizeof(double));
    memcpy(ef, e, (size_t)n * sizeof(fb_complex_double_t));
  }
  memcpy(x, b, (size_t)ldb * (size_t)((nrhs > 0) ? nrhs : 1) *
                   sizeof(fb_complex_double_t));
  *rcond = 1.0;
  return 0;
}

static int stub_sptsvx_isolation_a(fb_layout_t layout, char fact, int n,
                                   int nrhs, const float *d, const float *e,
                                   float *df, float *ef, const float *b,
                                   int ldb, float *x, int ldx, float *rcond,
                                   float *ferr, float *berr) {
  (void)layout; (void)n; (void)nrhs; (void)d; (void)e; (void)df; (void)ef;
  (void)b; (void)ldb; (void)x; (void)ldx; (void)rcond; (void)ferr; (void)berr;
  g_sptsvx_isolation_a_calls += 1;
  if (fact != 'N') {
    return -1;
  }
  return 3601;
}

static int stub_sptsvx_isolation_b(fb_layout_t layout, char fact, int n,
                                   int nrhs, const float *d, const float *e,
                                   float *df, float *ef, const float *b,
                                   int ldb, float *x, int ldx, float *rcond,
                                   float *ferr, float *berr) {
  (void)layout; (void)n; (void)nrhs; (void)d; (void)e; (void)df; (void)ef;
  (void)b; (void)ldb; (void)x; (void)ldx; (void)rcond; (void)ferr; (void)berr;
  g_sptsvx_isolation_b_calls += 1;
  if (fact != 'N') {
    return -1;
  }
  return 3602;
}

static int stub_dptsvx_isolation_a(fb_layout_t layout, char fact, int n,
                                   int nrhs, const double *d,
                                   const double *e, double *df, double *ef,
                                   const double *b, int ldb, double *x,
                                   int ldx, double *rcond, double *ferr,
                                   double *berr) {
  (void)layout; (void)n; (void)nrhs; (void)d; (void)e; (void)df; (void)ef;
  (void)b; (void)ldb; (void)x; (void)ldx; (void)rcond; (void)ferr; (void)berr;
  g_dptsvx_isolation_a_calls += 1;
  if (fact != 'N') {
    return -1;
  }
  return 3603;
}

static int stub_dptsvx_isolation_b(fb_layout_t layout, char fact, int n,
                                   int nrhs, const double *d,
                                   const double *e, double *df, double *ef,
                                   const double *b, int ldb, double *x,
                                   int ldx, double *rcond, double *ferr,
                                   double *berr) {
  (void)layout; (void)n; (void)nrhs; (void)d; (void)e; (void)df; (void)ef;
  (void)b; (void)ldb; (void)x; (void)ldx; (void)rcond; (void)ferr; (void)berr;
  g_dptsvx_isolation_b_calls += 1;
  if (fact != 'N') {
    return -1;
  }
  return 3604;
}

static int stub_cptsvx_isolation_a(
    fb_layout_t layout, char fact, int n, int nrhs, const float *d,
    const fb_complex_float_t *e, float *df, fb_complex_float_t *ef,
    const fb_complex_float_t *b, int ldb, fb_complex_float_t *x, int ldx,
    float *rcond, float *ferr, float *berr) {
  (void)layout; (void)n; (void)nrhs; (void)d; (void)e; (void)df; (void)ef;
  (void)b; (void)ldb; (void)x; (void)ldx; (void)rcond; (void)ferr; (void)berr;
  g_cptsvx_isolation_a_calls += 1;
  if (fact != 'N') {
    return -1;
  }
  return 3605;
}

static int stub_cptsvx_isolation_b(
    fb_layout_t layout, char fact, int n, int nrhs, const float *d,
    const fb_complex_float_t *e, float *df, fb_complex_float_t *ef,
    const fb_complex_float_t *b, int ldb, fb_complex_float_t *x, int ldx,
    float *rcond, float *ferr, float *berr) {
  (void)layout; (void)n; (void)nrhs; (void)d; (void)e; (void)df; (void)ef;
  (void)b; (void)ldb; (void)x; (void)ldx; (void)rcond; (void)ferr; (void)berr;
  g_cptsvx_isolation_b_calls += 1;
  if (fact != 'N') {
    return -1;
  }
  return 3606;
}

static int stub_zptsvx_isolation_a(
    fb_layout_t layout, char fact, int n, int nrhs, const double *d,
    const fb_complex_double_t *e, double *df, fb_complex_double_t *ef,
    const fb_complex_double_t *b, int ldb, fb_complex_double_t *x, int ldx,
    double *rcond, double *ferr, double *berr) {
  (void)layout; (void)n; (void)nrhs; (void)d; (void)e; (void)df; (void)ef;
  (void)b; (void)ldb; (void)x; (void)ldx; (void)rcond; (void)ferr; (void)berr;
  g_zptsvx_isolation_a_calls += 1;
  if (fact != 'N') {
    return -1;
  }
  return 3607;
}

static int stub_zptsvx_isolation_b(
    fb_layout_t layout, char fact, int n, int nrhs, const double *d,
    const fb_complex_double_t *e, double *df, fb_complex_double_t *ef,
    const fb_complex_double_t *b, int ldb, fb_complex_double_t *x, int ldx,
    double *rcond, double *ferr, double *berr) {
  (void)layout; (void)n; (void)nrhs; (void)d; (void)e; (void)df; (void)ef;
  (void)b; (void)ldb; (void)x; (void)ldx; (void)rcond; (void)ferr; (void)berr;
  g_zptsvx_isolation_b_calls += 1;
  if (fact != 'N') {
    return -1;
  }
  return 3608;
}

static int stub_sptsv_existing(fb_layout_t layout, int n, int nrhs, float *d,
                               float *e, float *b, int ldb) {
  (void)layout; (void)n; (void)nrhs; (void)d; (void)e; (void)b; (void)ldb;
  g_sptsv_existing_calls += 1;
  return 1135;
}

static int stub_dptsv_existing(fb_layout_t layout, int n, int nrhs,
                               double *d, double *e, double *b, int ldb) {
  (void)layout; (void)n; (void)nrhs; (void)d; (void)e; (void)b; (void)ldb;
  g_dptsv_existing_calls += 1;
  return 1145;
}

static int stub_cptsv_existing(fb_layout_t layout, int n, int nrhs, float *d,
                               fb_complex_float_t *e,
                               fb_complex_float_t *b, int ldb) {
  (void)layout; (void)n; (void)nrhs; (void)d; (void)e; (void)b; (void)ldb;
  g_cptsv_existing_calls += 1;
  return 1155;
}

static int stub_zptsv_existing(fb_layout_t layout, int n, int nrhs,
                               double *d, fb_complex_double_t *e,
                               fb_complex_double_t *b, int ldb) {
  (void)layout; (void)n; (void)nrhs; (void)d; (void)e; (void)b; (void)ldb;
  g_zptsv_existing_calls += 1;
  return 1165;
}

static int stub_spbsvx_base(fb_layout_t layout, char fact, char uplo, int n,
                            int kd, int nrhs, float *ab, int ldab,
                            float *afb, int ldafb, char *equed, float *s,
                            float *b, int ldb, float *x, int ldx,
                            float *rcond, float *ferr, float *berr) {
  (void)layout;
  (void)kd;
  g_spbsvx_bridge_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER || !ab || !afb || !equed || !s ||
      !b || !x || !rcond || !ferr || !berr || ldab != ldafb || ldb != ldx) {
    return -3701;
  }
  if (n > 0) {
    memcpy(afb, ab, (size_t)ldab * (size_t)n * sizeof(float));
    memcpy(x, b, (size_t)ldb * (size_t)((nrhs > 0) ? nrhs : 1) * sizeof(float));
    s[0] = 1.0f;
  }
  *equed = 'N';
  *rcond = 1.0f;
  return 0;
}

static int stub_dpbsvx_base(fb_layout_t layout, char fact, char uplo, int n,
                            int kd, int nrhs, double *ab, int ldab,
                            double *afb, int ldafb, char *equed, double *s,
                            double *b, int ldb, double *x, int ldx,
                            double *rcond, double *ferr, double *berr) {
  (void)layout;
  (void)kd;
  g_dpbsvx_bridge_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER || !ab || !afb || !equed || !s ||
      !b || !x || !rcond || !ferr || !berr || ldab != ldafb || ldb != ldx) {
    return -3702;
  }
  if (n > 0) {
    memcpy(afb, ab, (size_t)ldab * (size_t)n * sizeof(double));
    memcpy(x, b, (size_t)ldb * (size_t)((nrhs > 0) ? nrhs : 1) * sizeof(double));
    s[0] = 1.0;
  }
  *equed = 'N';
  *rcond = 1.0;
  return 0;
}

static int stub_cpbsvx_base(fb_layout_t layout, char fact, char uplo, int n,
                            int kd, int nrhs, fb_complex_float_t *ab,
                            int ldab, fb_complex_float_t *afb, int ldafb,
                            char *equed, float *s, fb_complex_float_t *b,
                            int ldb, fb_complex_float_t *x, int ldx,
                            float *rcond, float *ferr, float *berr) {
  (void)layout;
  (void)kd;
  g_cpbsvx_bridge_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER || !ab || !afb || !equed || !s ||
      !b || !x || !rcond || !ferr || !berr || ldab != ldafb || ldb != ldx) {
    return -3703;
  }
  if (n > 0) {
    memcpy(afb, ab, (size_t)ldab * (size_t)n * sizeof(fb_complex_float_t));
    memcpy(x, b, (size_t)ldb * (size_t)((nrhs > 0) ? nrhs : 1) * sizeof(fb_complex_float_t));
    s[0] = 1.0f;
  }
  *equed = 'N';
  *rcond = 1.0f;
  return 0;
}

static int stub_zpbsvx_base(fb_layout_t layout, char fact, char uplo, int n,
                            int kd, int nrhs, fb_complex_double_t *ab,
                            int ldab, fb_complex_double_t *afb, int ldafb,
                            char *equed, double *s, fb_complex_double_t *b,
                            int ldb, fb_complex_double_t *x, int ldx,
                            double *rcond, double *ferr, double *berr) {
  (void)layout;
  (void)kd;
  g_zpbsvx_bridge_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER || !ab || !afb || !equed || !s ||
      !b || !x || !rcond || !ferr || !berr || ldab != ldafb || ldb != ldx) {
    return -3704;
  }
  if (n > 0) {
    memcpy(afb, ab, (size_t)ldab * (size_t)n * sizeof(fb_complex_double_t));
    memcpy(x, b, (size_t)ldb * (size_t)((nrhs > 0) ? nrhs : 1) * sizeof(fb_complex_double_t));
    s[0] = 1.0;
  }
  *equed = 'N';
  *rcond = 1.0;
  return 0;
}

static int stub_spbsvx_isolation_a(fb_layout_t layout, char fact, char uplo,
                                   int n, int kd, int nrhs, float *ab,
                                   int ldab, float *afb, int ldafb,
                                   char *equed, float *s, float *b, int ldb,
                                   float *x, int ldx, float *rcond,
                                   float *ferr, float *berr) {
  (void)layout; (void)n; (void)kd; (void)nrhs; (void)ab; (void)ldab;
  (void)afb; (void)ldafb; (void)equed; (void)s; (void)b; (void)ldb; (void)x;
  (void)ldx; (void)rcond; (void)ferr; (void)berr;
  g_spbsvx_isolation_a_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER) {
    return -1;
  }
  return 3701;
}

static int stub_spbsvx_isolation_b(fb_layout_t layout, char fact, char uplo,
                                   int n, int kd, int nrhs, float *ab,
                                   int ldab, float *afb, int ldafb,
                                   char *equed, float *s, float *b, int ldb,
                                   float *x, int ldx, float *rcond,
                                   float *ferr, float *berr) {
  (void)layout; (void)n; (void)kd; (void)nrhs; (void)ab; (void)ldab;
  (void)afb; (void)ldafb; (void)equed; (void)s; (void)b; (void)ldb; (void)x;
  (void)ldx; (void)rcond; (void)ferr; (void)berr;
  g_spbsvx_isolation_b_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER) {
    return -1;
  }
  return 3702;
}

static int stub_dpbsvx_isolation_a(fb_layout_t layout, char fact, char uplo,
                                   int n, int kd, int nrhs, double *ab,
                                   int ldab, double *afb, int ldafb,
                                   char *equed, double *s, double *b, int ldb,
                                   double *x, int ldx, double *rcond,
                                   double *ferr, double *berr) {
  (void)layout; (void)n; (void)kd; (void)nrhs; (void)ab; (void)ldab;
  (void)afb; (void)ldafb; (void)equed; (void)s; (void)b; (void)ldb; (void)x;
  (void)ldx; (void)rcond; (void)ferr; (void)berr;
  g_dpbsvx_isolation_a_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER) {
    return -1;
  }
  return 3703;
}

static int stub_dpbsvx_isolation_b(fb_layout_t layout, char fact, char uplo,
                                   int n, int kd, int nrhs, double *ab,
                                   int ldab, double *afb, int ldafb,
                                   char *equed, double *s, double *b, int ldb,
                                   double *x, int ldx, double *rcond,
                                   double *ferr, double *berr) {
  (void)layout; (void)n; (void)kd; (void)nrhs; (void)ab; (void)ldab;
  (void)afb; (void)ldafb; (void)equed; (void)s; (void)b; (void)ldb; (void)x;
  (void)ldx; (void)rcond; (void)ferr; (void)berr;
  g_dpbsvx_isolation_b_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER) {
    return -1;
  }
  return 3704;
}

static int stub_cpbsvx_isolation_a(fb_layout_t layout, char fact, char uplo,
                                   int n, int kd, int nrhs,
                                   fb_complex_float_t *ab, int ldab,
                                   fb_complex_float_t *afb, int ldafb,
                                   char *equed, float *s,
                                   fb_complex_float_t *b, int ldb,
                                   fb_complex_float_t *x, int ldx,
                                   float *rcond, float *ferr, float *berr) {
  (void)layout; (void)n; (void)kd; (void)nrhs; (void)ab; (void)ldab;
  (void)afb; (void)ldafb; (void)equed; (void)s; (void)b; (void)ldb; (void)x;
  (void)ldx; (void)rcond; (void)ferr; (void)berr;
  g_cpbsvx_isolation_a_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER) {
    return -1;
  }
  return 3705;
}

static int stub_cpbsvx_isolation_b(fb_layout_t layout, char fact, char uplo,
                                   int n, int kd, int nrhs,
                                   fb_complex_float_t *ab, int ldab,
                                   fb_complex_float_t *afb, int ldafb,
                                   char *equed, float *s,
                                   fb_complex_float_t *b, int ldb,
                                   fb_complex_float_t *x, int ldx,
                                   float *rcond, float *ferr, float *berr) {
  (void)layout; (void)n; (void)kd; (void)nrhs; (void)ab; (void)ldab;
  (void)afb; (void)ldafb; (void)equed; (void)s; (void)b; (void)ldb; (void)x;
  (void)ldx; (void)rcond; (void)ferr; (void)berr;
  g_cpbsvx_isolation_b_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER) {
    return -1;
  }
  return 3706;
}

static int stub_zpbsvx_isolation_a(fb_layout_t layout, char fact, char uplo,
                                   int n, int kd, int nrhs,
                                   fb_complex_double_t *ab, int ldab,
                                   fb_complex_double_t *afb, int ldafb,
                                   char *equed, double *s,
                                   fb_complex_double_t *b, int ldb,
                                   fb_complex_double_t *x, int ldx,
                                   double *rcond, double *ferr,
                                   double *berr) {
  (void)layout; (void)n; (void)kd; (void)nrhs; (void)ab; (void)ldab;
  (void)afb; (void)ldafb; (void)equed; (void)s; (void)b; (void)ldb; (void)x;
  (void)ldx; (void)rcond; (void)ferr; (void)berr;
  g_zpbsvx_isolation_a_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER) {
    return -1;
  }
  return 3707;
}

static int stub_zpbsvx_isolation_b(fb_layout_t layout, char fact, char uplo,
                                   int n, int kd, int nrhs,
                                   fb_complex_double_t *ab, int ldab,
                                   fb_complex_double_t *afb, int ldafb,
                                   char *equed, double *s,
                                   fb_complex_double_t *b, int ldb,
                                   fb_complex_double_t *x, int ldx,
                                   double *rcond, double *ferr,
                                   double *berr) {
  (void)layout; (void)n; (void)kd; (void)nrhs; (void)ab; (void)ldab;
  (void)afb; (void)ldafb; (void)equed; (void)s; (void)b; (void)ldb; (void)x;
  (void)ldx; (void)rcond; (void)ferr; (void)berr;
  g_zpbsvx_isolation_b_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER) {
    return -1;
  }
  return 3708;
}

static int stub_spbsv_existing(fb_layout_t layout, char uplo, int n, int kd,
                               int nrhs, float *ab, int ldab, float *b,
                               int ldb) {
  (void)layout; (void)uplo; (void)n; (void)kd; (void)nrhs; (void)ab;
  (void)ldab; (void)b; (void)ldb;
  g_spbsv_existing_calls += 1;
  return 1175;
}

static int stub_dpbsv_existing(fb_layout_t layout, char uplo, int n, int kd,
                               int nrhs, double *ab, int ldab, double *b,
                               int ldb) {
  (void)layout; (void)uplo; (void)n; (void)kd; (void)nrhs; (void)ab;
  (void)ldab; (void)b; (void)ldb;
  g_dpbsv_existing_calls += 1;
  return 1185;
}

static int stub_cpbsv_existing(fb_layout_t layout, char uplo, int n, int kd,
                               int nrhs, fb_complex_float_t *ab, int ldab,
                               fb_complex_float_t *b, int ldb) {
  (void)layout; (void)uplo; (void)n; (void)kd; (void)nrhs; (void)ab;
  (void)ldab; (void)b; (void)ldb;
  g_cpbsv_existing_calls += 1;
  return 1195;
}

static int stub_zpbsv_existing(fb_layout_t layout, char uplo, int n, int kd,
                               int nrhs, fb_complex_double_t *ab, int ldab,
                               fb_complex_double_t *b, int ldb) {
  (void)layout; (void)uplo; (void)n; (void)kd; (void)nrhs; (void)ab;
  (void)ldab; (void)b; (void)ldb;
  g_zpbsv_existing_calls += 1;
  return 1205;
}

static int stub_sppsvx_base(fb_layout_t layout, char fact, char uplo, int n,
                            int nrhs, float *ap, float *afp, char *equed,
                            float *s, float *b, int ldb, float *x, int ldx,
                            float *rcond, float *ferr, float *berr) {
  size_t packed_elems = (size_t)((n > 0) ? ((n * (n + 1)) / 2) : 1);
  (void)layout;
  g_sppsvx_bridge_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER || !ap || !afp || !equed || !s ||
      !b || !x || !rcond || !ferr || !berr || ldb != ldx) {
    return -3801;
  }
  memcpy(afp, ap, packed_elems * sizeof(float));
  memcpy(x, b, (size_t)ldb * (size_t)((nrhs > 0) ? nrhs : 1) * sizeof(float));
  s[0] = 1.0f;
  *equed = 'N';
  *rcond = 1.0f;
  return 0;
}

static int stub_dppsvx_base(fb_layout_t layout, char fact, char uplo, int n,
                            int nrhs, double *ap, double *afp, char *equed,
                            double *s, double *b, int ldb, double *x,
                            int ldx, double *rcond, double *ferr,
                            double *berr) {
  size_t packed_elems = (size_t)((n > 0) ? ((n * (n + 1)) / 2) : 1);
  (void)layout;
  g_dppsvx_bridge_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER || !ap || !afp || !equed || !s ||
      !b || !x || !rcond || !ferr || !berr || ldb != ldx) {
    return -3802;
  }
  memcpy(afp, ap, packed_elems * sizeof(double));
  memcpy(x, b, (size_t)ldb * (size_t)((nrhs > 0) ? nrhs : 1) * sizeof(double));
  s[0] = 1.0;
  *equed = 'N';
  *rcond = 1.0;
  return 0;
}

static int stub_cppsvx_base(fb_layout_t layout, char fact, char uplo, int n,
                            int nrhs, fb_complex_float_t *ap,
                            fb_complex_float_t *afp, char *equed, float *s,
                            fb_complex_float_t *b, int ldb,
                            fb_complex_float_t *x, int ldx, float *rcond,
                            float *ferr, float *berr) {
  size_t packed_elems = (size_t)((n > 0) ? ((n * (n + 1)) / 2) : 1);
  (void)layout;
  g_cppsvx_bridge_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER || !ap || !afp || !equed || !s ||
      !b || !x || !rcond || !ferr || !berr || ldb != ldx) {
    return -3803;
  }
  memcpy(afp, ap, packed_elems * sizeof(fb_complex_float_t));
  memcpy(x, b,
         (size_t)ldb * (size_t)((nrhs > 0) ? nrhs : 1) *
             sizeof(fb_complex_float_t));
  s[0] = 1.0f;
  *equed = 'N';
  *rcond = 1.0f;
  return 0;
}

static int stub_zppsvx_base(fb_layout_t layout, char fact, char uplo, int n,
                            int nrhs, fb_complex_double_t *ap,
                            fb_complex_double_t *afp, char *equed, double *s,
                            fb_complex_double_t *b, int ldb,
                            fb_complex_double_t *x, int ldx, double *rcond,
                            double *ferr, double *berr) {
  size_t packed_elems = (size_t)((n > 0) ? ((n * (n + 1)) / 2) : 1);
  (void)layout;
  g_zppsvx_bridge_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER || !ap || !afp || !equed || !s ||
      !b || !x || !rcond || !ferr || !berr || ldb != ldx) {
    return -3804;
  }
  memcpy(afp, ap, packed_elems * sizeof(fb_complex_double_t));
  memcpy(x, b,
         (size_t)ldb * (size_t)((nrhs > 0) ? nrhs : 1) *
             sizeof(fb_complex_double_t));
  s[0] = 1.0;
  *equed = 'N';
  *rcond = 1.0;
  return 0;
}

static int stub_sppsvx_isolation_a(fb_layout_t layout, char fact, char uplo,
                                   int n, int nrhs, float *ap, float *afp,
                                   char *equed, float *s, float *b, int ldb,
                                   float *x, int ldx, float *rcond,
                                   float *ferr, float *berr) {
  (void)layout; (void)n; (void)nrhs; (void)ap; (void)afp; (void)equed;
  (void)s; (void)b; (void)ldb; (void)x; (void)ldx; (void)rcond; (void)ferr;
  (void)berr;
  g_sppsvx_isolation_a_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER) {
    return -1;
  }
  return 3801;
}

static int stub_sppsvx_isolation_b(fb_layout_t layout, char fact, char uplo,
                                   int n, int nrhs, float *ap, float *afp,
                                   char *equed, float *s, float *b, int ldb,
                                   float *x, int ldx, float *rcond,
                                   float *ferr, float *berr) {
  (void)layout; (void)n; (void)nrhs; (void)ap; (void)afp; (void)equed;
  (void)s; (void)b; (void)ldb; (void)x; (void)ldx; (void)rcond; (void)ferr;
  (void)berr;
  g_sppsvx_isolation_b_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER) {
    return -1;
  }
  return 3802;
}

static int stub_dppsvx_isolation_a(fb_layout_t layout, char fact, char uplo,
                                   int n, int nrhs, double *ap, double *afp,
                                   char *equed, double *s, double *b, int ldb,
                                   double *x, int ldx, double *rcond,
                                   double *ferr, double *berr) {
  (void)layout; (void)n; (void)nrhs; (void)ap; (void)afp; (void)equed;
  (void)s; (void)b; (void)ldb; (void)x; (void)ldx; (void)rcond; (void)ferr;
  (void)berr;
  g_dppsvx_isolation_a_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER) {
    return -1;
  }
  return 3803;
}

static int stub_dppsvx_isolation_b(fb_layout_t layout, char fact, char uplo,
                                   int n, int nrhs, double *ap, double *afp,
                                   char *equed, double *s, double *b, int ldb,
                                   double *x, int ldx, double *rcond,
                                   double *ferr, double *berr) {
  (void)layout; (void)n; (void)nrhs; (void)ap; (void)afp; (void)equed;
  (void)s; (void)b; (void)ldb; (void)x; (void)ldx; (void)rcond; (void)ferr;
  (void)berr;
  g_dppsvx_isolation_b_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER) {
    return -1;
  }
  return 3804;
}

static int stub_cppsvx_isolation_a(fb_layout_t layout, char fact, char uplo,
                                   int n, int nrhs, fb_complex_float_t *ap,
                                   fb_complex_float_t *afp, char *equed,
                                   float *s, fb_complex_float_t *b, int ldb,
                                   fb_complex_float_t *x, int ldx,
                                   float *rcond, float *ferr, float *berr) {
  (void)layout; (void)n; (void)nrhs; (void)ap; (void)afp; (void)equed;
  (void)s; (void)b; (void)ldb; (void)x; (void)ldx; (void)rcond; (void)ferr;
  (void)berr;
  g_cppsvx_isolation_a_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER) {
    return -1;
  }
  return 3805;
}

static int stub_cppsvx_isolation_b(fb_layout_t layout, char fact, char uplo,
                                   int n, int nrhs, fb_complex_float_t *ap,
                                   fb_complex_float_t *afp, char *equed,
                                   float *s, fb_complex_float_t *b, int ldb,
                                   fb_complex_float_t *x, int ldx,
                                   float *rcond, float *ferr, float *berr) {
  (void)layout; (void)n; (void)nrhs; (void)ap; (void)afp; (void)equed;
  (void)s; (void)b; (void)ldb; (void)x; (void)ldx; (void)rcond; (void)ferr;
  (void)berr;
  g_cppsvx_isolation_b_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER) {
    return -1;
  }
  return 3806;
}

static int stub_zppsvx_isolation_a(fb_layout_t layout, char fact, char uplo,
                                   int n, int nrhs, fb_complex_double_t *ap,
                                   fb_complex_double_t *afp, char *equed,
                                   double *s, fb_complex_double_t *b, int ldb,
                                   fb_complex_double_t *x, int ldx,
                                   double *rcond, double *ferr,
                                   double *berr) {
  (void)layout; (void)n; (void)nrhs; (void)ap; (void)afp; (void)equed;
  (void)s; (void)b; (void)ldb; (void)x; (void)ldx; (void)rcond; (void)ferr;
  (void)berr;
  g_zppsvx_isolation_a_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER) {
    return -1;
  }
  return 3807;
}

static int stub_zppsvx_isolation_b(fb_layout_t layout, char fact, char uplo,
                                   int n, int nrhs, fb_complex_double_t *ap,
                                   fb_complex_double_t *afp, char *equed,
                                   double *s, fb_complex_double_t *b, int ldb,
                                   fb_complex_double_t *x, int ldx,
                                   double *rcond, double *ferr,
                                   double *berr) {
  (void)layout; (void)n; (void)nrhs; (void)ap; (void)afp; (void)equed;
  (void)s; (void)b; (void)ldb; (void)x; (void)ldx; (void)rcond; (void)ferr;
  (void)berr;
  g_zppsvx_isolation_b_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER) {
    return -1;
  }
  return 3808;
}

static int stub_sppsv_existing(fb_layout_t layout, char uplo, int n, int nrhs,
                               float *ap, float *b, int ldb) {
  (void)layout; (void)uplo; (void)n; (void)nrhs; (void)ap; (void)b;
  (void)ldb;
  g_sppsv_existing_calls += 1;
  return 1215;
}

static int stub_dppsv_existing(fb_layout_t layout, char uplo, int n, int nrhs,
                               double *ap, double *b, int ldb) {
  (void)layout; (void)uplo; (void)n; (void)nrhs; (void)ap; (void)b;
  (void)ldb;
  g_dppsv_existing_calls += 1;
  return 1225;
}

static int stub_cppsv_existing(fb_layout_t layout, char uplo, int n, int nrhs,
                               fb_complex_float_t *ap, fb_complex_float_t *b,
                               int ldb) {
  (void)layout; (void)uplo; (void)n; (void)nrhs; (void)ap; (void)b;
  (void)ldb;
  g_cppsv_existing_calls += 1;
  return 1235;
}

static int stub_zppsv_existing(fb_layout_t layout, char uplo, int n, int nrhs,
                               fb_complex_double_t *ap,
                               fb_complex_double_t *b, int ldb) {
  (void)layout; (void)uplo; (void)n; (void)nrhs; (void)ap; (void)b;
  (void)ldb;
  g_zppsv_existing_calls += 1;
  return 1245;
}

static int stub_sspsvx_base(fb_layout_t layout, char fact, char uplo, int n,
                            int nrhs, const float *ap, float *afp, int *ipiv,
                            const float *b, int ldb, float *x, int ldx,
                            float *rcond, float *ferr, float *berr) {
  size_t packed_elems = (size_t)((n > 0) ? ((n * (n + 1)) / 2) : 1);
  (void)layout;
  g_sspsvx_bridge_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER || !ap || !afp || !ipiv || !b ||
      !x || !rcond || !ferr || !berr || ldb != ldx) {
    return -3901;
  }
  memcpy(afp, ap, packed_elems * sizeof(float));
  memcpy(x, b, (size_t)ldb * (size_t)((nrhs > 0) ? nrhs : 1) * sizeof(float));
  if (n > 0) {
    ipiv[0] = 1;
  }
  *rcond = 1.0f;
  return 0;
}

static int stub_dspsvx_base(fb_layout_t layout, char fact, char uplo, int n,
                            int nrhs, const double *ap, double *afp,
                            int *ipiv, const double *b, int ldb, double *x,
                            int ldx, double *rcond, double *ferr,
                            double *berr) {
  size_t packed_elems = (size_t)((n > 0) ? ((n * (n + 1)) / 2) : 1);
  (void)layout;
  g_dspsvx_bridge_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER || !ap || !afp || !ipiv || !b ||
      !x || !rcond || !ferr || !berr || ldb != ldx) {
    return -3902;
  }
  memcpy(afp, ap, packed_elems * sizeof(double));
  memcpy(x, b, (size_t)ldb * (size_t)((nrhs > 0) ? nrhs : 1) * sizeof(double));
  if (n > 0) {
    ipiv[0] = 1;
  }
  *rcond = 1.0;
  return 0;
}

static int stub_cspsvx_base(fb_layout_t layout, char fact, char uplo, int n,
                            int nrhs, const fb_complex_float_t *ap,
                            fb_complex_float_t *afp, int *ipiv,
                            const fb_complex_float_t *b, int ldb,
                            fb_complex_float_t *x, int ldx, float *rcond,
                            float *ferr, float *berr) {
  size_t packed_elems = (size_t)((n > 0) ? ((n * (n + 1)) / 2) : 1);
  (void)layout;
  g_cspsvx_bridge_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER || !ap || !afp || !ipiv || !b ||
      !x || !rcond || !ferr || !berr || ldb != ldx) {
    return -3903;
  }
  memcpy(afp, ap, packed_elems * sizeof(fb_complex_float_t));
  memcpy(x, b,
         (size_t)ldb * (size_t)((nrhs > 0) ? nrhs : 1) *
             sizeof(fb_complex_float_t));
  if (n > 0) {
    ipiv[0] = 1;
  }
  *rcond = 1.0f;
  return 0;
}

static int stub_zspsvx_base(fb_layout_t layout, char fact, char uplo, int n,
                            int nrhs, const fb_complex_double_t *ap,
                            fb_complex_double_t *afp, int *ipiv,
                            const fb_complex_double_t *b, int ldb,
                            fb_complex_double_t *x, int ldx, double *rcond,
                            double *ferr, double *berr) {
  size_t packed_elems = (size_t)((n > 0) ? ((n * (n + 1)) / 2) : 1);
  (void)layout;
  g_zspsvx_bridge_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER || !ap || !afp || !ipiv || !b ||
      !x || !rcond || !ferr || !berr || ldb != ldx) {
    return -3904;
  }
  memcpy(afp, ap, packed_elems * sizeof(fb_complex_double_t));
  memcpy(x, b,
         (size_t)ldb * (size_t)((nrhs > 0) ? nrhs : 1) *
             sizeof(fb_complex_double_t));
  if (n > 0) {
    ipiv[0] = 1;
  }
  *rcond = 1.0;
  return 0;
}

static int stub_sspsvx_isolation_a(fb_layout_t layout, char fact, char uplo,
                                   int n, int nrhs, const float *ap,
                                   float *afp, int *ipiv, const float *b,
                                   int ldb, float *x, int ldx, float *rcond,
                                   float *ferr, float *berr) {
  (void)layout; (void)n; (void)nrhs; (void)ap; (void)afp; (void)ipiv;
  (void)b; (void)ldb; (void)x; (void)ldx; (void)rcond; (void)ferr; (void)berr;
  g_sspsvx_isolation_a_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER) {
    return -1;
  }
  return 3901;
}

static int stub_sspsvx_isolation_b(fb_layout_t layout, char fact, char uplo,
                                   int n, int nrhs, const float *ap,
                                   float *afp, int *ipiv, const float *b,
                                   int ldb, float *x, int ldx, float *rcond,
                                   float *ferr, float *berr) {
  (void)layout; (void)n; (void)nrhs; (void)ap; (void)afp; (void)ipiv;
  (void)b; (void)ldb; (void)x; (void)ldx; (void)rcond; (void)ferr; (void)berr;
  g_sspsvx_isolation_b_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER) {
    return -1;
  }
  return 3902;
}

static int stub_dspsvx_isolation_a(fb_layout_t layout, char fact, char uplo,
                                   int n, int nrhs, const double *ap,
                                   double *afp, int *ipiv, const double *b,
                                   int ldb, double *x, int ldx,
                                   double *rcond, double *ferr,
                                   double *berr) {
  (void)layout; (void)n; (void)nrhs; (void)ap; (void)afp; (void)ipiv;
  (void)b; (void)ldb; (void)x; (void)ldx; (void)rcond; (void)ferr; (void)berr;
  g_dspsvx_isolation_a_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER) {
    return -1;
  }
  return 3903;
}

static int stub_dspsvx_isolation_b(fb_layout_t layout, char fact, char uplo,
                                   int n, int nrhs, const double *ap,
                                   double *afp, int *ipiv, const double *b,
                                   int ldb, double *x, int ldx,
                                   double *rcond, double *ferr,
                                   double *berr) {
  (void)layout; (void)n; (void)nrhs; (void)ap; (void)afp; (void)ipiv;
  (void)b; (void)ldb; (void)x; (void)ldx; (void)rcond; (void)ferr; (void)berr;
  g_dspsvx_isolation_b_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER) {
    return -1;
  }
  return 3904;
}

static int stub_cspsvx_isolation_a(fb_layout_t layout, char fact, char uplo,
                                   int n, int nrhs,
                                   const fb_complex_float_t *ap,
                                   fb_complex_float_t *afp, int *ipiv,
                                   const fb_complex_float_t *b, int ldb,
                                   fb_complex_float_t *x, int ldx,
                                   float *rcond, float *ferr, float *berr) {
  (void)layout; (void)n; (void)nrhs; (void)ap; (void)afp; (void)ipiv;
  (void)b; (void)ldb; (void)x; (void)ldx; (void)rcond; (void)ferr; (void)berr;
  g_cspsvx_isolation_a_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER) {
    return -1;
  }
  return 3905;
}

static int stub_cspsvx_isolation_b(fb_layout_t layout, char fact, char uplo,
                                   int n, int nrhs,
                                   const fb_complex_float_t *ap,
                                   fb_complex_float_t *afp, int *ipiv,
                                   const fb_complex_float_t *b, int ldb,
                                   fb_complex_float_t *x, int ldx,
                                   float *rcond, float *ferr, float *berr) {
  (void)layout; (void)n; (void)nrhs; (void)ap; (void)afp; (void)ipiv;
  (void)b; (void)ldb; (void)x; (void)ldx; (void)rcond; (void)ferr; (void)berr;
  g_cspsvx_isolation_b_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER) {
    return -1;
  }
  return 3906;
}

static int stub_zspsvx_isolation_a(fb_layout_t layout, char fact, char uplo,
                                   int n, int nrhs,
                                   const fb_complex_double_t *ap,
                                   fb_complex_double_t *afp, int *ipiv,
                                   const fb_complex_double_t *b, int ldb,
                                   fb_complex_double_t *x, int ldx,
                                   double *rcond, double *ferr,
                                   double *berr) {
  (void)layout; (void)n; (void)nrhs; (void)ap; (void)afp; (void)ipiv;
  (void)b; (void)ldb; (void)x; (void)ldx; (void)rcond; (void)ferr; (void)berr;
  g_zspsvx_isolation_a_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER) {
    return -1;
  }
  return 3907;
}

static int stub_zspsvx_isolation_b(fb_layout_t layout, char fact, char uplo,
                                   int n, int nrhs,
                                   const fb_complex_double_t *ap,
                                   fb_complex_double_t *afp, int *ipiv,
                                   const fb_complex_double_t *b, int ldb,
                                   fb_complex_double_t *x, int ldx,
                                   double *rcond, double *ferr,
                                   double *berr) {
  (void)layout; (void)n; (void)nrhs; (void)ap; (void)afp; (void)ipiv;
  (void)b; (void)ldb; (void)x; (void)ldx; (void)rcond; (void)ferr; (void)berr;
  g_zspsvx_isolation_b_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER) {
    return -1;
  }
  return 3908;
}

static int stub_sspsv_existing(fb_layout_t layout, char uplo, int n, int nrhs,
                               float *ap, int *ipiv, float *b, int ldb) {
  (void)layout; (void)uplo; (void)n; (void)nrhs; (void)ap; (void)ipiv;
  (void)b; (void)ldb;
  g_sspsv_existing_calls += 1;
  return 1255;
}

static int stub_dspsv_existing(fb_layout_t layout, char uplo, int n, int nrhs,
                               double *ap, int *ipiv, double *b, int ldb) {
  (void)layout; (void)uplo; (void)n; (void)nrhs; (void)ap; (void)ipiv;
  (void)b; (void)ldb;
  g_dspsv_existing_calls += 1;
  return 1265;
}

static int stub_cspsv_existing(fb_layout_t layout, char uplo, int n, int nrhs,
                               fb_complex_float_t *ap, int *ipiv,
                               fb_complex_float_t *b, int ldb) {
  (void)layout; (void)uplo; (void)n; (void)nrhs; (void)ap; (void)ipiv;
  (void)b; (void)ldb;
  g_cspsv_existing_calls += 1;
  return 1275;
}

static int stub_zspsv_existing(fb_layout_t layout, char uplo, int n, int nrhs,
                               fb_complex_double_t *ap, int *ipiv,
                               fb_complex_double_t *b, int ldb) {
  (void)layout; (void)uplo; (void)n; (void)nrhs; (void)ap; (void)ipiv;
  (void)b; (void)ldb;
  g_zspsv_existing_calls += 1;
  return 1285;
}

static int stub_sgtsvx_isolation_a(fb_layout_t layout, char fact, char trans,
                                   int n, int nrhs, const float *dl,
                                   const float *d, const float *du, float *dlf,
                                   float *df, float *duf, float *du2,
                                   int *ipiv, const float *b, int ldb,
                                   float *x, int ldx, float *rcond,
                                   float *ferr, float *berr) {
  (void)layout; (void)n; (void)nrhs; (void)dl; (void)d; (void)du; (void)dlf;
  (void)df; (void)duf; (void)du2; (void)ipiv; (void)b; (void)ldb; (void)x;
  (void)ldx; (void)rcond; (void)ferr; (void)berr;
  g_sgtsvx_isolation_a_calls += 1;
  if (fact != 'N' || trans != 'N') {
    return -1;
  }
  return 3505;
}

static int stub_sgtsvx_isolation_b(fb_layout_t layout, char fact, char trans,
                                   int n, int nrhs, const float *dl,
                                   const float *d, const float *du, float *dlf,
                                   float *df, float *duf, float *du2,
                                   int *ipiv, const float *b, int ldb,
                                   float *x, int ldx, float *rcond,
                                   float *ferr, float *berr) {
  (void)layout; (void)n; (void)nrhs; (void)dl; (void)d; (void)du; (void)dlf;
  (void)df; (void)duf; (void)du2; (void)ipiv; (void)b; (void)ldb; (void)x;
  (void)ldx; (void)rcond; (void)ferr; (void)berr;
  g_sgtsvx_isolation_b_calls += 1;
  if (fact != 'N' || trans != 'N') {
    return -1;
  }
  return 3506;
}

static int stub_dgtsvx_isolation_a(fb_layout_t layout, char fact, char trans,
                                   int n, int nrhs, const double *dl,
                                   const double *d, const double *du,
                                   double *dlf, double *df, double *duf,
                                   double *du2, int *ipiv, const double *b,
                                   int ldb, double *x, int ldx, double *rcond,
                                   double *ferr, double *berr) {
  (void)layout; (void)n; (void)nrhs; (void)dl; (void)d; (void)du; (void)dlf;
  (void)df; (void)duf; (void)du2; (void)ipiv; (void)b; (void)ldb; (void)x;
  (void)ldx; (void)rcond; (void)ferr; (void)berr;
  g_dgtsvx_isolation_a_calls += 1;
  if (fact != 'N' || trans != 'N') {
    return -1;
  }
  return 3507;
}

static int stub_dgtsvx_isolation_b(fb_layout_t layout, char fact, char trans,
                                   int n, int nrhs, const double *dl,
                                   const double *d, const double *du,
                                   double *dlf, double *df, double *duf,
                                   double *du2, int *ipiv, const double *b,
                                   int ldb, double *x, int ldx, double *rcond,
                                   double *ferr, double *berr) {
  (void)layout; (void)n; (void)nrhs; (void)dl; (void)d; (void)du; (void)dlf;
  (void)df; (void)duf; (void)du2; (void)ipiv; (void)b; (void)ldb; (void)x;
  (void)ldx; (void)rcond; (void)ferr; (void)berr;
  g_dgtsvx_isolation_b_calls += 1;
  if (fact != 'N' || trans != 'N') {
    return -1;
  }
  return 3508;
}

static int stub_cgtsvx_isolation_a(fb_layout_t layout, char fact, char trans,
                                   int n, int nrhs,
                                   const fb_complex_float_t *dl,
                                   const fb_complex_float_t *d,
                                   const fb_complex_float_t *du,
                                   fb_complex_float_t *dlf,
                                   fb_complex_float_t *df,
                                   fb_complex_float_t *duf,
                                   fb_complex_float_t *du2, int *ipiv,
                                   const fb_complex_float_t *b, int ldb,
                                   fb_complex_float_t *x, int ldx,
                                   float *rcond, float *ferr, float *berr) {
  (void)layout; (void)n; (void)nrhs; (void)dl; (void)d; (void)du; (void)dlf;
  (void)df; (void)duf; (void)du2; (void)ipiv; (void)b; (void)ldb; (void)x;
  (void)ldx; (void)rcond; (void)ferr; (void)berr;
  g_cgtsvx_isolation_a_calls += 1;
  if (fact != 'N' || trans != 'N') {
    return -1;
  }
  return 3509;
}

static int stub_cgtsvx_isolation_b(fb_layout_t layout, char fact, char trans,
                                   int n, int nrhs,
                                   const fb_complex_float_t *dl,
                                   const fb_complex_float_t *d,
                                   const fb_complex_float_t *du,
                                   fb_complex_float_t *dlf,
                                   fb_complex_float_t *df,
                                   fb_complex_float_t *duf,
                                   fb_complex_float_t *du2, int *ipiv,
                                   const fb_complex_float_t *b, int ldb,
                                   fb_complex_float_t *x, int ldx,
                                   float *rcond, float *ferr, float *berr) {
  (void)layout; (void)n; (void)nrhs; (void)dl; (void)d; (void)du; (void)dlf;
  (void)df; (void)duf; (void)du2; (void)ipiv; (void)b; (void)ldb; (void)x;
  (void)ldx; (void)rcond; (void)ferr; (void)berr;
  g_cgtsvx_isolation_b_calls += 1;
  if (fact != 'N' || trans != 'N') {
    return -1;
  }
  return 3510;
}

static int stub_zgtsvx_isolation_a(fb_layout_t layout, char fact, char trans,
                                   int n, int nrhs,
                                   const fb_complex_double_t *dl,
                                   const fb_complex_double_t *d,
                                   const fb_complex_double_t *du,
                                   fb_complex_double_t *dlf,
                                   fb_complex_double_t *df,
                                   fb_complex_double_t *duf,
                                   fb_complex_double_t *du2, int *ipiv,
                                   const fb_complex_double_t *b, int ldb,
                                   fb_complex_double_t *x, int ldx,
                                   double *rcond, double *ferr, double *berr) {
  (void)layout; (void)n; (void)nrhs; (void)dl; (void)d; (void)du; (void)dlf;
  (void)df; (void)duf; (void)du2; (void)ipiv; (void)b; (void)ldb; (void)x;
  (void)ldx; (void)rcond; (void)ferr; (void)berr;
  g_zgtsvx_isolation_a_calls += 1;
  if (fact != 'N' || trans != 'N') {
    return -1;
  }
  return 3511;
}

static int stub_zgtsvx_isolation_b(fb_layout_t layout, char fact, char trans,
                                   int n, int nrhs,
                                   const fb_complex_double_t *dl,
                                   const fb_complex_double_t *d,
                                   const fb_complex_double_t *du,
                                   fb_complex_double_t *dlf,
                                   fb_complex_double_t *df,
                                   fb_complex_double_t *duf,
                                   fb_complex_double_t *du2, int *ipiv,
                                   const fb_complex_double_t *b, int ldb,
                                   fb_complex_double_t *x, int ldx,
                                   double *rcond, double *ferr, double *berr) {
  (void)layout; (void)n; (void)nrhs; (void)dl; (void)d; (void)du; (void)dlf;
  (void)df; (void)duf; (void)du2; (void)ipiv; (void)b; (void)ldb; (void)x;
  (void)ldx; (void)rcond; (void)ferr; (void)berr;
  g_zgtsvx_isolation_b_calls += 1;
  if (fact != 'N' || trans != 'N') {
    return -1;
  }
  return 3512;
}

static int stub_sgtsv_existing(fb_layout_t layout, int n, int nrhs, float *dl,
                               float *d, float *du, float *b, int ldb) {
  (void)layout; (void)n; (void)nrhs; (void)dl; (void)d; (void)du; (void)b;
  (void)ldb;
  g_sgtsv_existing_calls += 1;
  return 1095;
}

static int stub_dgtsv_existing(fb_layout_t layout, int n, int nrhs,
                               double *dl, double *d, double *du, double *b,
                               int ldb) {
  (void)layout; (void)n; (void)nrhs; (void)dl; (void)d; (void)du; (void)b;
  (void)ldb;
  g_dgtsv_existing_calls += 1;
  return 1105;
}

static int stub_cgtsv_existing(fb_layout_t layout, int n, int nrhs,
                               fb_complex_float_t *dl, fb_complex_float_t *d,
                               fb_complex_float_t *du, fb_complex_float_t *b,
                               int ldb) {
  (void)layout; (void)n; (void)nrhs; (void)dl; (void)d; (void)du; (void)b;
  (void)ldb;
  g_cgtsv_existing_calls += 1;
  return 1115;
}

static int stub_zgtsv_existing(fb_layout_t layout, int n, int nrhs,
                               fb_complex_double_t *dl, fb_complex_double_t *d,
                               fb_complex_double_t *du, fb_complex_double_t *b,
                               int ldb) {
  (void)layout; (void)n; (void)nrhs; (void)dl; (void)d; (void)du; (void)b;
  (void)ldb;
  g_zgtsv_existing_calls += 1;
  return 1125;
}

static int stub_sgbsvx_isolation_a(
    fb_layout_t layout, char fact, char trans, int n, int kl, int ku, int nrhs,
    float *ab, int ldab, float *afb, int ldafb, int *ipiv, char *equed,
    float *r, float *c, float *b, int ldb, float *x, int ldx, float *rcond,
    float *ferr, float *berr, float *rpivot) {
  (void)layout;
  (void)n;
  (void)kl;
  (void)ku;
  (void)nrhs;
  (void)ab;
  (void)ldab;
  (void)afb;
  (void)ldafb;
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
  g_sgbsvx_isolation_a_calls += 1;
  if (fact != 'N' || trans != 'N') {
    return -1;
  }
  return 3101;
}

static int stub_sgbsvx_isolation_b(
    fb_layout_t layout, char fact, char trans, int n, int kl, int ku, int nrhs,
    float *ab, int ldab, float *afb, int ldafb, int *ipiv, char *equed,
    float *r, float *c, float *b, int ldb, float *x, int ldx, float *rcond,
    float *ferr, float *berr, float *rpivot) {
  (void)layout;
  (void)n;
  (void)kl;
  (void)ku;
  (void)nrhs;
  (void)ab;
  (void)ldab;
  (void)afb;
  (void)ldafb;
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
  g_sgbsvx_isolation_b_calls += 1;
  if (fact != 'N' || trans != 'N') {
    return -1;
  }
  return 3102;
}

static int stub_dgbsvx_isolation_a(
    fb_layout_t layout, char fact, char trans, int n, int kl, int ku, int nrhs,
    double *ab, int ldab, double *afb, int ldafb, int *ipiv, char *equed,
    double *r, double *c, double *b, int ldb, double *x, int ldx,
    double *rcond, double *ferr, double *berr, double *rpivot) {
  (void)layout;
  (void)n;
  (void)kl;
  (void)ku;
  (void)nrhs;
  (void)ab;
  (void)ldab;
  (void)afb;
  (void)ldafb;
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
  g_dgbsvx_isolation_a_calls += 1;
  if (fact != 'N' || trans != 'N') {
    return -1;
  }
  return 3201;
}

static int stub_dgbsvx_isolation_b(
    fb_layout_t layout, char fact, char trans, int n, int kl, int ku, int nrhs,
    double *ab, int ldab, double *afb, int ldafb, int *ipiv, char *equed,
    double *r, double *c, double *b, int ldb, double *x, int ldx,
    double *rcond, double *ferr, double *berr, double *rpivot) {
  (void)layout;
  (void)n;
  (void)kl;
  (void)ku;
  (void)nrhs;
  (void)ab;
  (void)ldab;
  (void)afb;
  (void)ldafb;
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
  g_dgbsvx_isolation_b_calls += 1;
  if (fact != 'N' || trans != 'N') {
    return -1;
  }
  return 3202;
}

static int stub_cgbsvx_isolation_a(
    fb_layout_t layout, char fact, char trans, int n, int kl, int ku, int nrhs,
    fb_complex_float_t *ab, int ldab, fb_complex_float_t *afb, int ldafb,
    int *ipiv, char *equed, float *r, float *c, fb_complex_float_t *b, int ldb,
    fb_complex_float_t *x, int ldx, float *rcond, float *ferr, float *berr,
    float *rpivot) {
  (void)layout;
  (void)n;
  (void)kl;
  (void)ku;
  (void)nrhs;
  (void)ab;
  (void)ldab;
  (void)afb;
  (void)ldafb;
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
  g_cgbsvx_isolation_a_calls += 1;
  if (fact != 'N' || trans != 'N') {
    return -1;
  }
  return 3301;
}

static int stub_cgbsvx_isolation_b(
    fb_layout_t layout, char fact, char trans, int n, int kl, int ku, int nrhs,
    fb_complex_float_t *ab, int ldab, fb_complex_float_t *afb, int ldafb,
    int *ipiv, char *equed, float *r, float *c, fb_complex_float_t *b, int ldb,
    fb_complex_float_t *x, int ldx, float *rcond, float *ferr, float *berr,
    float *rpivot) {
  (void)layout;
  (void)n;
  (void)kl;
  (void)ku;
  (void)nrhs;
  (void)ab;
  (void)ldab;
  (void)afb;
  (void)ldafb;
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
  g_cgbsvx_isolation_b_calls += 1;
  if (fact != 'N' || trans != 'N') {
    return -1;
  }
  return 3302;
}

static int stub_zgbsvx_isolation_a(
    fb_layout_t layout, char fact, char trans, int n, int kl, int ku, int nrhs,
    fb_complex_double_t *ab, int ldab, fb_complex_double_t *afb, int ldafb,
    int *ipiv, char *equed, double *r, double *c, fb_complex_double_t *b,
    int ldb, fb_complex_double_t *x, int ldx, double *rcond, double *ferr,
    double *berr, double *rpivot) {
  (void)layout;
  (void)n;
  (void)kl;
  (void)ku;
  (void)nrhs;
  (void)ab;
  (void)ldab;
  (void)afb;
  (void)ldafb;
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
  g_zgbsvx_isolation_a_calls += 1;
  if (fact != 'N' || trans != 'N') {
    return -1;
  }
  return 3401;
}

static int stub_zgbsvx_isolation_b(
    fb_layout_t layout, char fact, char trans, int n, int kl, int ku, int nrhs,
    fb_complex_double_t *ab, int ldab, fb_complex_double_t *afb, int ldafb,
    int *ipiv, char *equed, double *r, double *c, fb_complex_double_t *b,
    int ldb, fb_complex_double_t *x, int ldx, double *rcond, double *ferr,
    double *berr, double *rpivot) {
  (void)layout;
  (void)n;
  (void)kl;
  (void)ku;
  (void)nrhs;
  (void)ab;
  (void)ldab;
  (void)afb;
  (void)ldafb;
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
  g_zgbsvx_isolation_b_calls += 1;
  if (fact != 'N' || trans != 'N') {
    return -1;
  }
  return 3402;
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

static int stub_sgbsv_existing(fb_layout_t layout, int n, int kl, int ku,
                               int nrhs, float *ab, int ldab, int *ipiv,
                               float *b, int ldb) {
  (void)layout;
  (void)n;
  (void)kl;
  (void)ku;
  (void)nrhs;
  (void)ab;
  (void)ldab;
  (void)ipiv;
  (void)b;
  (void)ldb;
  g_sgbsv_existing_calls += 1;
  return 1055;
}

static int stub_dgbsv_existing(fb_layout_t layout, int n, int kl, int ku,
                               int nrhs, double *ab, int ldab, int *ipiv,
                               double *b, int ldb) {
  (void)layout;
  (void)n;
  (void)kl;
  (void)ku;
  (void)nrhs;
  (void)ab;
  (void)ldab;
  (void)ipiv;
  (void)b;
  (void)ldb;
  g_dgbsv_existing_calls += 1;
  return 1065;
}

static int stub_cgbsv_existing(fb_layout_t layout, int n, int kl, int ku,
                               int nrhs, fb_complex_float_t *ab, int ldab,
                               int *ipiv, fb_complex_float_t *b, int ldb) {
  (void)layout;
  (void)n;
  (void)kl;
  (void)ku;
  (void)nrhs;
  (void)ab;
  (void)ldab;
  (void)ipiv;
  (void)b;
  (void)ldb;
  g_cgbsv_existing_calls += 1;
  return 1075;
}

static int stub_zgbsv_existing(fb_layout_t layout, int n, int kl, int ku,
                               int nrhs, fb_complex_double_t *ab, int ldab,
                               int *ipiv, fb_complex_double_t *b, int ldb) {
  (void)layout;
  (void)n;
  (void)kl;
  (void)ku;
  (void)nrhs;
  (void)ab;
  (void)ldab;
  (void)ipiv;
  (void)b;
  (void)ldb;
  g_zgbsv_existing_calls += 1;
  return 1085;
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

static int stub_ssysvxx_base(
    fb_layout_t layout, char fact, char uplo, int n, int nrhs, float *a,
    int lda, float *af, int ldaf, int *ipiv, char *equed, float *s, float *b,
    int ldb, float *x, int ldx, float *rcond, float *rpvgrw, float *berr,
    int n_err_bnds, float *err_bnds_norm, float *err_bnds_comp, int nparams,
    float *params) {
  (void)layout;
  g_ssysvxx_bridge_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER || !a || !af || !ipiv || !equed ||
      !s || !b || !x || !rcond || !rpvgrw || !berr || lda != ldaf ||
      ldb != ldx || n_err_bnds != 0 || err_bnds_norm != NULL ||
      err_bnds_comp != NULL || nparams != 0 || params != NULL) {
    return -3510;
  }
  memcpy(af, a, (size_t)lda * (size_t)((n > 0) ? n : 1) * sizeof(float));
  memcpy(x, b, (size_t)ldb * (size_t)((nrhs > 0) ? nrhs : 1) * sizeof(float));
  if (n > 0) {
    ipiv[0] = 31;
  }
  *equed = 'N';
  *rcond = 1.0f;
  *rpvgrw = 1.0f;
  return 0;
}

static int stub_dsysvxx_base(
    fb_layout_t layout, char fact, char uplo, int n, int nrhs, double *a,
    int lda, double *af, int ldaf, int *ipiv, char *equed, double *s,
    double *b, int ldb, double *x, int ldx, double *rcond, double *rpvgrw,
    double *berr, int n_err_bnds, double *err_bnds_norm,
    double *err_bnds_comp, int nparams, double *params) {
  (void)layout;
  g_dsysvxx_bridge_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER || !a || !af || !ipiv || !equed ||
      !s || !b || !x || !rcond || !rpvgrw || !berr || lda != ldaf ||
      ldb != ldx || n_err_bnds != 0 || err_bnds_norm != NULL ||
      err_bnds_comp != NULL || nparams != 0 || params != NULL) {
    return -3611;
  }
  memcpy(af, a,
         (size_t)lda * (size_t)((n > 0) ? n : 1) * sizeof(double));
  memcpy(x, b,
         (size_t)ldb * (size_t)((nrhs > 0) ? nrhs : 1) * sizeof(double));
  if (n > 0) {
    ipiv[0] = 33;
  }
  *equed = 'N';
  *rcond = 1.0;
  *rpvgrw = 1.0;
  return 0;
}

static int stub_csysvxx_base(
    fb_layout_t layout, char fact, char uplo, int n, int nrhs,
    fb_complex_float_t *a, int lda, fb_complex_float_t *af, int ldaf,
    int *ipiv, char *equed, float *s, fb_complex_float_t *b, int ldb,
    fb_complex_float_t *x, int ldx, float *rcond, float *rpvgrw, float *berr,
    int n_err_bnds, float *err_bnds_norm, float *err_bnds_comp, int nparams,
    float *params) {
  (void)layout;
  g_csysvxx_bridge_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER || !a || !af || !ipiv || !equed ||
      !s || !b || !x || !rcond || !rpvgrw || !berr || lda != ldaf ||
      ldb != ldx || n_err_bnds != 0 || err_bnds_norm != NULL ||
      err_bnds_comp != NULL || nparams != 0 || params != NULL) {
    return -3712;
  }
  memcpy(af, a,
         (size_t)lda * (size_t)((n > 0) ? n : 1) * sizeof(fb_complex_float_t));
  memcpy(x, b,
         (size_t)ldb * (size_t)((nrhs > 0) ? nrhs : 1) * sizeof(fb_complex_float_t));
  if (n > 0) {
    ipiv[0] = 35;
  }
  *equed = 'N';
  *rcond = 1.0f;
  *rpvgrw = 1.0f;
  return 0;
}

static int stub_zsysvxx_base(
    fb_layout_t layout, char fact, char uplo, int n, int nrhs,
    fb_complex_double_t *a, int lda, fb_complex_double_t *af, int ldaf,
    int *ipiv, char *equed, double *s, fb_complex_double_t *b, int ldb,
    fb_complex_double_t *x, int ldx, double *rcond, double *rpvgrw,
    double *berr, int n_err_bnds, double *err_bnds_norm,
    double *err_bnds_comp, int nparams, double *params) {
  (void)layout;
  g_zsysvxx_bridge_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER || !a || !af || !ipiv || !equed ||
      !s || !b || !x || !rcond || !rpvgrw || !berr || lda != ldaf ||
      ldb != ldx || n_err_bnds != 0 || err_bnds_norm != NULL ||
      err_bnds_comp != NULL || nparams != 0 || params != NULL) {
    return -3813;
  }
  memcpy(af, a,
         (size_t)lda * (size_t)((n > 0) ? n : 1) * sizeof(fb_complex_double_t));
  memcpy(x, b,
         (size_t)ldb * (size_t)((nrhs > 0) ? nrhs : 1) * sizeof(fb_complex_double_t));
  if (n > 0) {
    ipiv[0] = 37;
  }
  *equed = 'N';
  *rcond = 1.0;
  *rpvgrw = 1.0;
  return 0;
}

static int stub_ssysvxx_isolation_a(
    fb_layout_t layout, char fact, char uplo, int n, int nrhs, float *a,
    int lda, float *af, int ldaf, int *ipiv, char *equed, float *s, float *b,
    int ldb, float *x, int ldx, float *rcond, float *rpvgrw, float *berr,
    int n_err_bnds, float *err_bnds_norm, float *err_bnds_comp, int nparams,
    float *params) {
  (void)layout; (void)n; (void)nrhs; (void)a; (void)lda; (void)af; (void)ldaf;
  (void)ipiv; (void)equed; (void)s; (void)b; (void)ldb; (void)x; (void)ldx;
  (void)rcond; (void)rpvgrw; (void)berr; (void)n_err_bnds;
  (void)err_bnds_norm; (void)err_bnds_comp; (void)nparams; (void)params;
  g_ssysvxx_isolation_a_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER) {
    return -1;
  }
  return 2921;
}

static int stub_ssysvxx_isolation_b(
    fb_layout_t layout, char fact, char uplo, int n, int nrhs, float *a,
    int lda, float *af, int ldaf, int *ipiv, char *equed, float *s, float *b,
    int ldb, float *x, int ldx, float *rcond, float *rpvgrw, float *berr,
    int n_err_bnds, float *err_bnds_norm, float *err_bnds_comp, int nparams,
    float *params) {
  (void)layout; (void)n; (void)nrhs; (void)a; (void)lda; (void)af; (void)ldaf;
  (void)ipiv; (void)equed; (void)s; (void)b; (void)ldb; (void)x; (void)ldx;
  (void)rcond; (void)rpvgrw; (void)berr; (void)n_err_bnds;
  (void)err_bnds_norm; (void)err_bnds_comp; (void)nparams; (void)params;
  g_ssysvxx_isolation_b_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER) {
    return -1;
  }
  return 2922;
}

static int stub_dsysvxx_isolation_a(
    fb_layout_t layout, char fact, char uplo, int n, int nrhs, double *a,
    int lda, double *af, int ldaf, int *ipiv, char *equed, double *s,
    double *b, int ldb, double *x, int ldx, double *rcond, double *rpvgrw,
    double *berr, int n_err_bnds, double *err_bnds_norm,
    double *err_bnds_comp, int nparams, double *params) {
  (void)layout; (void)n; (void)nrhs; (void)a; (void)lda; (void)af; (void)ldaf;
  (void)ipiv; (void)equed; (void)s; (void)b; (void)ldb; (void)x; (void)ldx;
  (void)rcond; (void)rpvgrw; (void)berr; (void)n_err_bnds;
  (void)err_bnds_norm; (void)err_bnds_comp; (void)nparams; (void)params;
  g_dsysvxx_isolation_a_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER) {
    return -1;
  }
  return 2923;
}

static int stub_dsysvxx_isolation_b(
    fb_layout_t layout, char fact, char uplo, int n, int nrhs, double *a,
    int lda, double *af, int ldaf, int *ipiv, char *equed, double *s,
    double *b, int ldb, double *x, int ldx, double *rcond, double *rpvgrw,
    double *berr, int n_err_bnds, double *err_bnds_norm,
    double *err_bnds_comp, int nparams, double *params) {
  (void)layout; (void)n; (void)nrhs; (void)a; (void)lda; (void)af; (void)ldaf;
  (void)ipiv; (void)equed; (void)s; (void)b; (void)ldb; (void)x; (void)ldx;
  (void)rcond; (void)rpvgrw; (void)berr; (void)n_err_bnds;
  (void)err_bnds_norm; (void)err_bnds_comp; (void)nparams; (void)params;
  g_dsysvxx_isolation_b_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER) {
    return -1;
  }
  return 2924;
}

static int stub_csysvxx_isolation_a(
    fb_layout_t layout, char fact, char uplo, int n, int nrhs,
    fb_complex_float_t *a, int lda, fb_complex_float_t *af, int ldaf,
    int *ipiv, char *equed, float *s, fb_complex_float_t *b, int ldb,
    fb_complex_float_t *x, int ldx, float *rcond, float *rpvgrw, float *berr,
    int n_err_bnds, float *err_bnds_norm, float *err_bnds_comp, int nparams,
    float *params) {
  (void)layout; (void)n; (void)nrhs; (void)a; (void)lda; (void)af; (void)ldaf;
  (void)ipiv; (void)equed; (void)s; (void)b; (void)ldb; (void)x; (void)ldx;
  (void)rcond; (void)rpvgrw; (void)berr; (void)n_err_bnds;
  (void)err_bnds_norm; (void)err_bnds_comp; (void)nparams; (void)params;
  g_csysvxx_isolation_a_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER) {
    return -1;
  }
  return 2925;
}

static int stub_csysvxx_isolation_b(
    fb_layout_t layout, char fact, char uplo, int n, int nrhs,
    fb_complex_float_t *a, int lda, fb_complex_float_t *af, int ldaf,
    int *ipiv, char *equed, float *s, fb_complex_float_t *b, int ldb,
    fb_complex_float_t *x, int ldx, float *rcond, float *rpvgrw, float *berr,
    int n_err_bnds, float *err_bnds_norm, float *err_bnds_comp, int nparams,
    float *params) {
  (void)layout; (void)n; (void)nrhs; (void)a; (void)lda; (void)af; (void)ldaf;
  (void)ipiv; (void)equed; (void)s; (void)b; (void)ldb; (void)x; (void)ldx;
  (void)rcond; (void)rpvgrw; (void)berr; (void)n_err_bnds;
  (void)err_bnds_norm; (void)err_bnds_comp; (void)nparams; (void)params;
  g_csysvxx_isolation_b_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER) {
    return -1;
  }
  return 2926;
}

static int stub_zsysvxx_isolation_a(
    fb_layout_t layout, char fact, char uplo, int n, int nrhs,
    fb_complex_double_t *a, int lda, fb_complex_double_t *af, int ldaf,
    int *ipiv, char *equed, double *s, fb_complex_double_t *b, int ldb,
    fb_complex_double_t *x, int ldx, double *rcond, double *rpvgrw,
    double *berr, int n_err_bnds, double *err_bnds_norm,
    double *err_bnds_comp, int nparams, double *params) {
  (void)layout; (void)n; (void)nrhs; (void)a; (void)lda; (void)af; (void)ldaf;
  (void)ipiv; (void)equed; (void)s; (void)b; (void)ldb; (void)x; (void)ldx;
  (void)rcond; (void)rpvgrw; (void)berr; (void)n_err_bnds;
  (void)err_bnds_norm; (void)err_bnds_comp; (void)nparams; (void)params;
  g_zsysvxx_isolation_a_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER) {
    return -1;
  }
  return 2927;
}

static int stub_zsysvxx_isolation_b(
    fb_layout_t layout, char fact, char uplo, int n, int nrhs,
    fb_complex_double_t *a, int lda, fb_complex_double_t *af, int ldaf,
    int *ipiv, char *equed, double *s, fb_complex_double_t *b, int ldb,
    fb_complex_double_t *x, int ldx, double *rcond, double *rpvgrw,
    double *berr, int n_err_bnds, double *err_bnds_norm,
    double *err_bnds_comp, int nparams, double *params) {
  (void)layout; (void)n; (void)nrhs; (void)a; (void)lda; (void)af; (void)ldaf;
  (void)ipiv; (void)equed; (void)s; (void)b; (void)ldb; (void)x; (void)ldx;
  (void)rcond; (void)rpvgrw; (void)berr; (void)n_err_bnds;
  (void)err_bnds_norm; (void)err_bnds_comp; (void)nparams; (void)params;
  g_zsysvxx_isolation_b_calls += 1;
  if (fact != 'N' || uplo != (char)FB_UPPER) {
    return -1;
  }
  return 2928;
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

static int test_adapter_bridge_sgesv_from_sgesvxx(void) {
  fb_backend_vtable_t vtable;
  fb_sgesv_fn sgesv_call = NULL;
  fb_status_t status;

  float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float b[2] = {5.0f, 6.0f};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_sgesvxx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_SGESVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sgesvxx_base;

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
    fprintf(stderr, "[FAIL] SGESV adapter bridge from SGESVXX returned error\n");
    return 1;
  }

  if (g_sgesvxx_bridge_calls != 1 || ipiv[0] != 17 || b[0] != 5.0f ||
      b[1] != 6.0f) {
    fprintf(stderr,
            "[FAIL] SGESV<-SGESVXX bridge propagation mismatch (calls=%d ipiv0=%d b0=%g b1=%g)\n",
            g_sgesvxx_bridge_calls, ipiv[0], b[0], b[1]);
    return 1;
  }

  printf("[PASS] adapter bridge SGESV <- SGESVXX applies neutral expert params\n");
  return 0;
}

static int test_adapter_bridge_dgesv_from_dgesvxx(void) {
  fb_backend_vtable_t vtable;
  fb_dgesv_fn dgesv_call = NULL;
  fb_status_t status;

  double a[4] = {1.0, 2.0, 3.0, 4.0};
  double b[2] = {5.0, 6.0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_dgesvxx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_DGESVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dgesvxx_base;

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
    fprintf(stderr, "[FAIL] DGESV adapter bridge from DGESVXX returned error\n");
    return 1;
  }

  if (g_dgesvxx_bridge_calls != 1 || ipiv[0] != 19 || b[0] != 5.0 ||
      b[1] != 6.0) {
    fprintf(stderr,
            "[FAIL] DGESV<-DGESVXX bridge propagation mismatch (calls=%d ipiv0=%d b0=%g b1=%g)\n",
            g_dgesvxx_bridge_calls, ipiv[0], b[0], b[1]);
    return 1;
  }

  printf("[PASS] adapter bridge DGESV <- DGESVXX applies neutral expert params\n");
  return 0;
}

static int test_adapter_bridge_cgesv_from_cgesvxx(void) {
  fb_backend_vtable_t vtable;
  fb_cgesv_fn cgesv_call = NULL;
  fb_status_t status;

  fb_complex_float_t a[4] = {0};
  fb_complex_float_t b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_cgesvxx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_CGESVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_cgesvxx_base;

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
    fprintf(stderr, "[FAIL] CGESV adapter bridge from CGESVXX returned error\n");
    return 1;
  }

  if (g_cgesvxx_bridge_calls != 1 || ipiv[0] != 21) {
    fprintf(stderr,
            "[FAIL] CGESV<-CGESVXX bridge propagation mismatch (calls=%d ipiv0=%d)\n",
            g_cgesvxx_bridge_calls, ipiv[0]);
    return 1;
  }

  printf("[PASS] adapter bridge CGESV <- CGESVXX applies neutral expert params\n");
  return 0;
}

static int test_adapter_bridge_zgesv_from_zgesvxx(void) {
  fb_backend_vtable_t vtable;
  fb_zgesv_fn zgesv_call = NULL;
  fb_status_t status;

  fb_complex_double_t a[4] = {0};
  fb_complex_double_t b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_zgesvxx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_ZGESVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zgesvxx_base;

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
    fprintf(stderr, "[FAIL] ZGESV adapter bridge from ZGESVXX returned error\n");
    return 1;
  }

  if (g_zgesvxx_bridge_calls != 1 || ipiv[0] != 23) {
    fprintf(stderr,
            "[FAIL] ZGESV<-ZGESVXX bridge propagation mismatch (calls=%d ipiv0=%d)\n",
            g_zgesvxx_bridge_calls, ipiv[0]);
    return 1;
  }

  printf("[PASS] adapter bridge ZGESV <- ZGESVXX applies neutral expert params\n");
  return 0;
}

static int test_adapter_bridge_sposv_from_sposvxx(void) {
  fb_backend_vtable_t vtable;
  fb_sposv_fn sposv_call = NULL;
  fb_status_t status;

  float a[4] = {1.0f, 0.0f, 0.0f, 1.0f};
  float b[2] = {5.0f, 6.0f};

  memset(&vtable, 0, sizeof(vtable));
  g_sposvxx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_SPOSVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sposvxx_base;

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
    fprintf(stderr, "[FAIL] SPOSV adapter bridge from SPOSVXX returned error\n");
    return 1;
  }

  if (g_sposvxx_bridge_calls != 1 || b[0] != 5.0f || b[1] != 6.0f) {
    fprintf(stderr,
            "[FAIL] SPOSV<-SPOSVXX bridge propagation mismatch (calls=%d b0=%g b1=%g)\n",
            g_sposvxx_bridge_calls, b[0], b[1]);
    return 1;
  }

  printf("[PASS] adapter bridge SPOSV <- SPOSVXX applies neutral expert params\n");
  return 0;
}

static int test_adapter_bridge_dposv_from_dposvxx(void) {
  fb_backend_vtable_t vtable;
  fb_dposv_fn dposv_call = NULL;
  fb_status_t status;

  double a[4] = {1.0, 0.0, 0.0, 1.0};
  double b[2] = {5.0, 6.0};

  memset(&vtable, 0, sizeof(vtable));
  g_dposvxx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_DPOSVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dposvxx_base;

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
    fprintf(stderr, "[FAIL] DPOSV adapter bridge from DPOSVXX returned error\n");
    return 1;
  }

  if (g_dposvxx_bridge_calls != 1 || b[0] != 5.0 || b[1] != 6.0) {
    fprintf(stderr,
            "[FAIL] DPOSV<-DPOSVXX bridge propagation mismatch (calls=%d b0=%g b1=%g)\n",
            g_dposvxx_bridge_calls, b[0], b[1]);
    return 1;
  }

  printf("[PASS] adapter bridge DPOSV <- DPOSVXX applies neutral expert params\n");
  return 0;
}

static int test_adapter_bridge_cposv_from_cposvxx(void) {
  fb_backend_vtable_t vtable;
  fb_cposv_fn cposv_call = NULL;
  fb_status_t status;

  fb_complex_float_t a[4] = {0};
  fb_complex_float_t b[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_cposvxx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_CPOSVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_cposvxx_base;

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
    fprintf(stderr, "[FAIL] CPOSV adapter bridge from CPOSVXX returned error\n");
    return 1;
  }

  if (g_cposvxx_bridge_calls != 1) {
    fprintf(stderr,
            "[FAIL] CPOSV<-CPOSVXX adapter call count mismatch (calls=%d)\n",
            g_cposvxx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter bridge CPOSV <- CPOSVXX applies neutral expert params\n");
  return 0;
}

static int test_adapter_bridge_zposv_from_zposvxx(void) {
  fb_backend_vtable_t vtable;
  fb_zposv_fn zposv_call = NULL;
  fb_status_t status;

  fb_complex_double_t a[4] = {0};
  fb_complex_double_t b[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_zposvxx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_ZPOSVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zposvxx_base;

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
    fprintf(stderr, "[FAIL] ZPOSV adapter bridge from ZPOSVXX returned error\n");
    return 1;
  }

  if (g_zposvxx_bridge_calls != 1) {
    fprintf(stderr,
            "[FAIL] ZPOSV<-ZPOSVXX adapter call count mismatch (calls=%d)\n",
            g_zposvxx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter bridge ZPOSV <- ZPOSVXX applies neutral expert params\n");
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

static int test_adapter_bridge_chesv_from_chesvx(void) {
  fb_backend_vtable_t vtable;
  fb_csysv_fn chesv_call = NULL;
  fb_status_t status;

  fb_complex_float_t a[4] = {0};
  fb_complex_float_t b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_csysvx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_CHESVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_csysvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  chesv_call = (fb_csysv_fn)vtable.ext_ops[FB_OP_CHESV][FB_CONV_CBLAS];
  if (!chesv_call) {
    fprintf(stderr, "[FAIL] CHESV adapter slot missing after finalize\n");
    return 1;
  }

  if (chesv_call(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, a, 2, ipiv, b,
                 2) != 0) {
    fprintf(stderr, "[FAIL] CHESV adapter bridge from CHESVX returned error\n");
    return 1;
  }

  if (g_csysvx_bridge_calls != 1 || ipiv[0] != 7) {
    fprintf(stderr,
            "[FAIL] CHESV<-CHESVX bridge propagation mismatch (calls=%d ipiv0=%d)\n",
            g_csysvx_bridge_calls, ipiv[0]);
    return 1;
  }

  printf("[PASS] adapter bridge CHESV <- CHESVX applies neutral expert params\n");
  return 0;
}

static int test_adapter_bridge_zhesv_from_zhesvx(void) {
  fb_backend_vtable_t vtable;
  fb_zsysv_fn zhesv_call = NULL;
  fb_status_t status;

  fb_complex_double_t a[4] = {0};
  fb_complex_double_t b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_zsysvx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_ZHESVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zsysvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  zhesv_call = (fb_zsysv_fn)vtable.ext_ops[FB_OP_ZHESV][FB_CONV_CBLAS];
  if (!zhesv_call) {
    fprintf(stderr, "[FAIL] ZHESV adapter slot missing after finalize\n");
    return 1;
  }

  if (zhesv_call(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, a, 2, ipiv, b,
                 2) != 0) {
    fprintf(stderr, "[FAIL] ZHESV adapter bridge from ZHESVX returned error\n");
    return 1;
  }

  if (g_zsysvx_bridge_calls != 1 || ipiv[0] != 9) {
    fprintf(stderr,
            "[FAIL] ZHESV<-ZHESVX bridge propagation mismatch (calls=%d ipiv0=%d)\n",
            g_zsysvx_bridge_calls, ipiv[0]);
    return 1;
  }

  printf("[PASS] adapter bridge ZHESV <- ZHESVX applies neutral expert params\n");
  return 0;
}

static int test_adapter_bridge_ssysv_from_ssysvxx(void) {
  fb_backend_vtable_t vtable;
  fb_ssysv_fn ssysv_call = NULL;
  fb_status_t status;

  float a[4] = {1.0f, 0.0f, 0.0f, 1.0f};
  float b[2] = {5.0f, 6.0f};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_ssysvxx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_SSYSVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_ssysvxx_base;

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
    fprintf(stderr, "[FAIL] SSYSV adapter bridge from SSYSVXX returned error\n");
    return 1;
  }

  if (g_ssysvxx_bridge_calls != 1 || ipiv[0] != 31) {
    fprintf(stderr,
            "[FAIL] SSYSV<-SSYSVXX bridge propagation mismatch (calls=%d ipiv0=%d)\n",
            g_ssysvxx_bridge_calls, ipiv[0]);
    return 1;
  }

  printf("[PASS] adapter bridge SSYSV <- SSYSVXX applies neutral expert params\n");
  return 0;
}

static int test_adapter_bridge_dsysv_from_dsysvxx(void) {
  fb_backend_vtable_t vtable;
  fb_dsysv_fn dsysv_call = NULL;
  fb_status_t status;

  double a[4] = {1.0, 0.0, 0.0, 1.0};
  double b[2] = {5.0, 6.0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_dsysvxx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_DSYSVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dsysvxx_base;

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
    fprintf(stderr, "[FAIL] DSYSV adapter bridge from DSYSVXX returned error\n");
    return 1;
  }

  if (g_dsysvxx_bridge_calls != 1 || ipiv[0] != 33) {
    fprintf(stderr,
            "[FAIL] DSYSV<-DSYSVXX bridge propagation mismatch (calls=%d ipiv0=%d)\n",
            g_dsysvxx_bridge_calls, ipiv[0]);
    return 1;
  }

  printf("[PASS] adapter bridge DSYSV <- DSYSVXX applies neutral expert params\n");
  return 0;
}

static int test_adapter_bridge_csysv_from_csysvxx(void) {
  fb_backend_vtable_t vtable;
  fb_csysv_fn csysv_call = NULL;
  fb_status_t status;

  fb_complex_float_t a[4] = {0};
  fb_complex_float_t b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_csysvxx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_CSYSVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_csysvxx_base;

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
    fprintf(stderr, "[FAIL] CSYSV adapter bridge from CSYSVXX returned error\n");
    return 1;
  }

  if (g_csysvxx_bridge_calls != 1 || ipiv[0] != 35) {
    fprintf(stderr,
            "[FAIL] CSYSV<-CSYSVXX bridge propagation mismatch (calls=%d ipiv0=%d)\n",
            g_csysvxx_bridge_calls, ipiv[0]);
    return 1;
  }

  printf("[PASS] adapter bridge CSYSV <- CSYSVXX applies neutral expert params\n");
  return 0;
}

static int test_adapter_bridge_zsysv_from_zsysvxx(void) {
  fb_backend_vtable_t vtable;
  fb_zsysv_fn zsysv_call = NULL;
  fb_status_t status;

  fb_complex_double_t a[4] = {0};
  fb_complex_double_t b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_zsysvxx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_ZSYSVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zsysvxx_base;

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
    fprintf(stderr, "[FAIL] ZSYSV adapter bridge from ZSYSVXX returned error\n");
    return 1;
  }

  if (g_zsysvxx_bridge_calls != 1 || ipiv[0] != 37) {
    fprintf(stderr,
            "[FAIL] ZSYSV<-ZSYSVXX bridge propagation mismatch (calls=%d ipiv0=%d)\n",
            g_zsysvxx_bridge_calls, ipiv[0]);
    return 1;
  }

  printf("[PASS] adapter bridge ZSYSV <- ZSYSVXX applies neutral expert params\n");
  return 0;
}

static int test_adapter_bridge_chesv_from_chesvxx(void) {
  fb_backend_vtable_t vtable;
  fb_csysv_fn chesv_call = NULL;
  fb_status_t status;

  fb_complex_float_t a[4] = {0};
  fb_complex_float_t b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_csysvxx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_CHESVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_csysvxx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  chesv_call = (fb_csysv_fn)vtable.ext_ops[FB_OP_CHESV][FB_CONV_CBLAS];
  if (!chesv_call) {
    fprintf(stderr, "[FAIL] CHESV adapter slot missing after finalize\n");
    return 1;
  }

  if (chesv_call(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, a, 2, ipiv, b,
                 2) != 0) {
    fprintf(stderr, "[FAIL] CHESV adapter bridge from CHESVXX returned error\n");
    return 1;
  }

  if (g_csysvxx_bridge_calls != 1 || ipiv[0] != 35) {
    fprintf(stderr,
            "[FAIL] CHESV<-CHESVXX bridge propagation mismatch (calls=%d ipiv0=%d)\n",
            g_csysvxx_bridge_calls, ipiv[0]);
    return 1;
  }

  printf("[PASS] adapter bridge CHESV <- CHESVXX applies neutral expert params\n");
  return 0;
}

static int test_adapter_bridge_zhesv_from_zhesvxx(void) {
  fb_backend_vtable_t vtable;
  fb_zsysv_fn zhesv_call = NULL;
  fb_status_t status;

  fb_complex_double_t a[4] = {0};
  fb_complex_double_t b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_zsysvxx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_ZHESVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zsysvxx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  zhesv_call = (fb_zsysv_fn)vtable.ext_ops[FB_OP_ZHESV][FB_CONV_CBLAS];
  if (!zhesv_call) {
    fprintf(stderr, "[FAIL] ZHESV adapter slot missing after finalize\n");
    return 1;
  }

  if (zhesv_call(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, a, 2, ipiv, b,
                 2) != 0) {
    fprintf(stderr, "[FAIL] ZHESV adapter bridge from ZHESVXX returned error\n");
    return 1;
  }

  if (g_zsysvxx_bridge_calls != 1 || ipiv[0] != 37) {
    fprintf(stderr,
            "[FAIL] ZHESV<-ZHESVXX bridge propagation mismatch (calls=%d ipiv0=%d)\n",
            g_zsysvxx_bridge_calls, ipiv[0]);
    return 1;
  }

  printf("[PASS] adapter bridge ZHESV <- ZHESVXX applies neutral expert params\n");
  return 0;
}

static int test_adapter_bridge_sgbsv_from_sgbsvx(void) {
  fb_backend_vtable_t vtable;
  fb_sgbsv_cblas_fn sgbsv_call = NULL;
  fb_status_t status;

  float ab[8] = {0};
  float b[2] = {5.0f, 6.0f};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_sgbsvx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_SGBSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sgbsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  sgbsv_call = (fb_sgbsv_cblas_fn)vtable.ext_ops[FB_OP_SGBSV][FB_CONV_CBLAS];
  if (!sgbsv_call) {
    fprintf(stderr, "[FAIL] SGBSV adapter slot missing after finalize\n");
    return 1;
  }

  if (sgbsv_call(FB_LAYOUT_COL_MAJOR, 2, 1, 1, 1, ab, 4, ipiv, b, 2) != 0) {
    fprintf(stderr, "[FAIL] SGBSV adapter bridge from SGBSVX returned error\n");
    return 1;
  }

  if (g_sgbsvx_bridge_calls != 1 || ipiv[0] != 11) {
    fprintf(stderr,
            "[FAIL] SGBSV<-SGBSVX bridge propagation mismatch (calls=%d ipiv0=%d)\n",
            g_sgbsvx_bridge_calls, ipiv[0]);
    return 1;
  }

  printf("[PASS] adapter bridge SGBSV <- SGBSVX applies neutral expert params\n");
  return 0;
}

static int test_adapter_bridge_dgbsv_from_dgbsvx(void) {
  fb_backend_vtable_t vtable;
  fb_dgbsv_cblas_fn dgbsv_call = NULL;
  fb_status_t status;

  double ab[8] = {0};
  double b[2] = {5.0, 6.0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_dgbsvx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_DGBSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dgbsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  dgbsv_call = (fb_dgbsv_cblas_fn)vtable.ext_ops[FB_OP_DGBSV][FB_CONV_CBLAS];
  if (!dgbsv_call) {
    fprintf(stderr, "[FAIL] DGBSV adapter slot missing after finalize\n");
    return 1;
  }

  if (dgbsv_call(FB_LAYOUT_COL_MAJOR, 2, 1, 1, 1, ab, 4, ipiv, b, 2) != 0) {
    fprintf(stderr, "[FAIL] DGBSV adapter bridge from DGBSVX returned error\n");
    return 1;
  }

  if (g_dgbsvx_bridge_calls != 1 || ipiv[0] != 13) {
    fprintf(stderr,
            "[FAIL] DGBSV<-DGBSVX bridge propagation mismatch (calls=%d ipiv0=%d)\n",
            g_dgbsvx_bridge_calls, ipiv[0]);
    return 1;
  }

  printf("[PASS] adapter bridge DGBSV <- DGBSVX applies neutral expert params\n");
  return 0;
}

static int test_adapter_bridge_cgbsv_from_cgbsvx(void) {
  fb_backend_vtable_t vtable;
  fb_cgbsv_cblas_fn cgbsv_call = NULL;
  fb_status_t status;

  fb_complex_float_t ab[8] = {0};
  fb_complex_float_t b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_cgbsvx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_CGBSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_cgbsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  cgbsv_call = (fb_cgbsv_cblas_fn)vtable.ext_ops[FB_OP_CGBSV][FB_CONV_CBLAS];
  if (!cgbsv_call) {
    fprintf(stderr, "[FAIL] CGBSV adapter slot missing after finalize\n");
    return 1;
  }

  if (cgbsv_call(FB_LAYOUT_COL_MAJOR, 2, 1, 1, 1, ab, 4, ipiv, b, 2) != 0) {
    fprintf(stderr, "[FAIL] CGBSV adapter bridge from CGBSVX returned error\n");
    return 1;
  }

  if (g_cgbsvx_bridge_calls != 1 || ipiv[0] != 15) {
    fprintf(stderr,
            "[FAIL] CGBSV<-CGBSVX bridge propagation mismatch (calls=%d ipiv0=%d)\n",
            g_cgbsvx_bridge_calls, ipiv[0]);
    return 1;
  }

  printf("[PASS] adapter bridge CGBSV <- CGBSVX applies neutral expert params\n");
  return 0;
}

static int test_adapter_bridge_zgbsv_from_zgbsvx(void) {
  fb_backend_vtable_t vtable;
  fb_zgbsv_cblas_fn zgbsv_call = NULL;
  fb_status_t status;

  fb_complex_double_t ab[8] = {0};
  fb_complex_double_t b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_zgbsvx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_ZGBSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zgbsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  zgbsv_call = (fb_zgbsv_cblas_fn)vtable.ext_ops[FB_OP_ZGBSV][FB_CONV_CBLAS];
  if (!zgbsv_call) {
    fprintf(stderr, "[FAIL] ZGBSV adapter slot missing after finalize\n");
    return 1;
  }

  if (zgbsv_call(FB_LAYOUT_COL_MAJOR, 2, 1, 1, 1, ab, 4, ipiv, b, 2) != 0) {
    fprintf(stderr, "[FAIL] ZGBSV adapter bridge from ZGBSVX returned error\n");
    return 1;
  }

  if (g_zgbsvx_bridge_calls != 1 || ipiv[0] != 17) {
    fprintf(stderr,
            "[FAIL] ZGBSV<-ZGBSVX bridge propagation mismatch (calls=%d ipiv0=%d)\n",
            g_zgbsvx_bridge_calls, ipiv[0]);
    return 1;
  }

  printf("[PASS] adapter bridge ZGBSV <- ZGBSVX applies neutral expert params\n");
  return 0;
}

static int test_adapter_bridge_sgtsv_from_sgtsvx(void) {
  fb_backend_vtable_t vtable;
  fb_sgtsv_cblas_fn sgtsv_call = NULL;
  fb_status_t status;

  float dl[2] = {1.0f, 1.0f};
  float d[2] = {2.0f, 2.0f};
  float du[2] = {1.0f, 1.0f};
  float b[2] = {5.0f, 6.0f};

  memset(&vtable, 0, sizeof(vtable));
  g_sgtsvx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_SGTSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sgtsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  sgtsv_call = (fb_sgtsv_cblas_fn)vtable.ext_ops[FB_OP_SGTSV][FB_CONV_CBLAS];
  if (!sgtsv_call) {
    fprintf(stderr, "[FAIL] SGTSV adapter slot missing after finalize\n");
    return 1;
  }

  if (sgtsv_call(FB_LAYOUT_COL_MAJOR, 2, 1, dl, d, du, b, 2) != 0) {
    fprintf(stderr, "[FAIL] SGTSV adapter bridge from SGTSVX returned error\n");
    return 1;
  }

  if (g_sgtsvx_bridge_calls != 1 || dl[0] != 1.0f || d[0] != 2.0f || du[0] != 1.0f) {
    fprintf(stderr,
            "[FAIL] SGTSV<-SGTSVX bridge propagation mismatch (calls=%d)\n",
            g_sgtsvx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter bridge SGTSV <- SGTSVX applies neutral expert params\n");
  return 0;
}

static int test_adapter_bridge_dgtsv_from_dgtsvx(void) {
  fb_backend_vtable_t vtable;
  fb_dgtsv_cblas_fn dgtsv_call = NULL;
  fb_status_t status;

  double dl[2] = {1.0, 1.0};
  double d[2] = {2.0, 2.0};
  double du[2] = {1.0, 1.0};
  double b[2] = {5.0, 6.0};

  memset(&vtable, 0, sizeof(vtable));
  g_dgtsvx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_DGTSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dgtsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  dgtsv_call = (fb_dgtsv_cblas_fn)vtable.ext_ops[FB_OP_DGTSV][FB_CONV_CBLAS];
  if (!dgtsv_call) {
    fprintf(stderr, "[FAIL] DGTSV adapter slot missing after finalize\n");
    return 1;
  }

  if (dgtsv_call(FB_LAYOUT_COL_MAJOR, 2, 1, dl, d, du, b, 2) != 0) {
    fprintf(stderr, "[FAIL] DGTSV adapter bridge from DGTSVX returned error\n");
    return 1;
  }

  if (g_dgtsvx_bridge_calls != 1 || dl[0] != 1.0 || d[0] != 2.0 || du[0] != 1.0) {
    fprintf(stderr,
            "[FAIL] DGTSV<-DGTSVX bridge propagation mismatch (calls=%d)\n",
            g_dgtsvx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter bridge DGTSV <- DGTSVX applies neutral expert params\n");
  return 0;
}

static int test_adapter_bridge_cgtsv_from_cgtsvx(void) {
  fb_backend_vtable_t vtable;
  fb_cgtsv_cblas_fn cgtsv_call = NULL;
  fb_status_t status;

  fb_complex_float_t dl[2] = {0};
  fb_complex_float_t d[2] = {0};
  fb_complex_float_t du[2] = {0};
  fb_complex_float_t b[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_cgtsvx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_CGTSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_cgtsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  cgtsv_call = (fb_cgtsv_cblas_fn)vtable.ext_ops[FB_OP_CGTSV][FB_CONV_CBLAS];
  if (!cgtsv_call) {
    fprintf(stderr, "[FAIL] CGTSV adapter slot missing after finalize\n");
    return 1;
  }

  if (cgtsv_call(FB_LAYOUT_COL_MAJOR, 2, 1, dl, d, du, b, 2) != 0) {
    fprintf(stderr, "[FAIL] CGTSV adapter bridge from CGTSVX returned error\n");
    return 1;
  }

  if (g_cgtsvx_bridge_calls != 1) {
    fprintf(stderr,
            "[FAIL] CGTSV<-CGTSVX bridge propagation mismatch (calls=%d)\n",
            g_cgtsvx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter bridge CGTSV <- CGTSVX applies neutral expert params\n");
  return 0;
}

static int test_adapter_bridge_zgtsv_from_zgtsvx(void) {
  fb_backend_vtable_t vtable;
  fb_zgtsv_cblas_fn zgtsv_call = NULL;
  fb_status_t status;

  fb_complex_double_t dl[2] = {0};
  fb_complex_double_t d[2] = {0};
  fb_complex_double_t du[2] = {0};
  fb_complex_double_t b[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_zgtsvx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_ZGTSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zgtsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  zgtsv_call = (fb_zgtsv_cblas_fn)vtable.ext_ops[FB_OP_ZGTSV][FB_CONV_CBLAS];
  if (!zgtsv_call) {
    fprintf(stderr, "[FAIL] ZGTSV adapter slot missing after finalize\n");
    return 1;
  }

  if (zgtsv_call(FB_LAYOUT_COL_MAJOR, 2, 1, dl, d, du, b, 2) != 0) {
    fprintf(stderr, "[FAIL] ZGTSV adapter bridge from ZGTSVX returned error\n");
    return 1;
  }

  if (g_zgtsvx_bridge_calls != 1) {
    fprintf(stderr,
            "[FAIL] ZGTSV<-ZGTSVX bridge propagation mismatch (calls=%d)\n",
            g_zgtsvx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter bridge ZGTSV <- ZGTSVX applies neutral expert params\n");
  return 0;
}

static int test_adapter_bridge_sptsv_from_sptsvx(void) {
  fb_backend_vtable_t vtable;
  fb_sptsv_cblas_fn sptsv_call = NULL;
  fb_status_t status;
  float d[2] = {1.0f, 2.0f};
  float e[2] = {3.0f, 4.0f};
  float b[2] = {5.0f, 6.0f};

  memset(&vtable, 0, sizeof(vtable));
  g_sptsvx_bridge_calls = 0;
  vtable.ext_ops[FB_OP_SPTSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sptsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  sptsv_call = (fb_sptsv_cblas_fn)vtable.ext_ops[FB_OP_SPTSV][FB_CONV_CBLAS];
  if (!sptsv_call) {
    fprintf(stderr, "[FAIL] SPTSV adapter slot missing after finalize\n");
    return 1;
  }

  if (sptsv_call(FB_LAYOUT_COL_MAJOR, 2, 1, d, e, b, 2) != 0) {
    fprintf(stderr, "[FAIL] SPTSV adapter bridge from SPTSVX returned error\n");
    return 1;
  }

  if (g_sptsvx_bridge_calls != 1) {
    fprintf(stderr,
            "[FAIL] SPTSV<-SPTSVX bridge propagation mismatch (calls=%d)\n",
            g_sptsvx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter bridge SPTSV <- SPTSVX applies neutral expert params\n");
  return 0;
}

static int test_adapter_bridge_dptsv_from_dptsvx(void) {
  fb_backend_vtable_t vtable;
  fb_dptsv_cblas_fn dptsv_call = NULL;
  fb_status_t status;
  double d[2] = {1.0, 2.0};
  double e[2] = {3.0, 4.0};
  double b[2] = {5.0, 6.0};

  memset(&vtable, 0, sizeof(vtable));
  g_dptsvx_bridge_calls = 0;
  vtable.ext_ops[FB_OP_DPTSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dptsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  dptsv_call = (fb_dptsv_cblas_fn)vtable.ext_ops[FB_OP_DPTSV][FB_CONV_CBLAS];
  if (!dptsv_call) {
    fprintf(stderr, "[FAIL] DPTSV adapter slot missing after finalize\n");
    return 1;
  }

  if (dptsv_call(FB_LAYOUT_COL_MAJOR, 2, 1, d, e, b, 2) != 0) {
    fprintf(stderr, "[FAIL] DPTSV adapter bridge from DPTSVX returned error\n");
    return 1;
  }

  if (g_dptsvx_bridge_calls != 1) {
    fprintf(stderr,
            "[FAIL] DPTSV<-DPTSVX bridge propagation mismatch (calls=%d)\n",
            g_dptsvx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter bridge DPTSV <- DPTSVX applies neutral expert params\n");
  return 0;
}

static int test_adapter_bridge_cptsv_from_cptsvx(void) {
  fb_backend_vtable_t vtable;
  fb_cptsv_cblas_fn cptsv_call = NULL;
  fb_status_t status;
  float d[2] = {1.0f, 2.0f};
  fb_complex_float_t e[2] = {0};
  fb_complex_float_t b[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_cptsvx_bridge_calls = 0;
  vtable.ext_ops[FB_OP_CPTSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_cptsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  cptsv_call = (fb_cptsv_cblas_fn)vtable.ext_ops[FB_OP_CPTSV][FB_CONV_CBLAS];
  if (!cptsv_call) {
    fprintf(stderr, "[FAIL] CPTSV adapter slot missing after finalize\n");
    return 1;
  }

  if (cptsv_call(FB_LAYOUT_COL_MAJOR, 2, 1, d, e, b, 2) != 0) {
    fprintf(stderr, "[FAIL] CPTSV adapter bridge from CPTSVX returned error\n");
    return 1;
  }

  if (g_cptsvx_bridge_calls != 1) {
    fprintf(stderr,
            "[FAIL] CPTSV<-CPTSVX bridge propagation mismatch (calls=%d)\n",
            g_cptsvx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter bridge CPTSV <- CPTSVX applies neutral expert params\n");
  return 0;
}

static int test_adapter_bridge_zptsv_from_zptsvx(void) {
  fb_backend_vtable_t vtable;
  fb_zptsv_cblas_fn zptsv_call = NULL;
  fb_status_t status;
  double d[2] = {1.0, 2.0};
  fb_complex_double_t e[2] = {0};
  fb_complex_double_t b[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_zptsvx_bridge_calls = 0;
  vtable.ext_ops[FB_OP_ZPTSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zptsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  zptsv_call = (fb_zptsv_cblas_fn)vtable.ext_ops[FB_OP_ZPTSV][FB_CONV_CBLAS];
  if (!zptsv_call) {
    fprintf(stderr, "[FAIL] ZPTSV adapter slot missing after finalize\n");
    return 1;
  }

  if (zptsv_call(FB_LAYOUT_COL_MAJOR, 2, 1, d, e, b, 2) != 0) {
    fprintf(stderr, "[FAIL] ZPTSV adapter bridge from ZPTSVX returned error\n");
    return 1;
  }

  if (g_zptsvx_bridge_calls != 1) {
    fprintf(stderr,
            "[FAIL] ZPTSV<-ZPTSVX bridge propagation mismatch (calls=%d)\n",
            g_zptsvx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter bridge ZPTSV <- ZPTSVX applies neutral expert params\n");
  return 0;
}

static int test_adapter_bridge_spbsv_from_spbsvx(void) {
  fb_backend_vtable_t vtable;
  fb_spbsv_cblas_fn spbsv_call = NULL;
  fb_status_t status;
  float ab[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float b[2] = {5.0f, 6.0f};

  memset(&vtable, 0, sizeof(vtable));
  g_spbsvx_bridge_calls = 0;
  vtable.ext_ops[FB_OP_SPBSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_spbsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  spbsv_call = (fb_spbsv_cblas_fn)vtable.ext_ops[FB_OP_SPBSV][FB_CONV_CBLAS];
  if (!spbsv_call) {
    fprintf(stderr, "[FAIL] SPBSV adapter slot missing after finalize\n");
    return 1;
  }

  if (spbsv_call(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, 1, ab, 2, b, 2) != 0) {
    fprintf(stderr, "[FAIL] SPBSV adapter bridge from SPBSVX returned error\n");
    return 1;
  }

  if (g_spbsvx_bridge_calls != 1) {
    fprintf(stderr,
            "[FAIL] SPBSV<-SPBSVX bridge propagation mismatch (calls=%d)\n",
            g_spbsvx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter bridge SPBSV <- SPBSVX applies neutral expert params\n");
  return 0;
}

static int test_adapter_bridge_dpbsv_from_dpbsvx(void) {
  fb_backend_vtable_t vtable;
  fb_dpbsv_cblas_fn dpbsv_call = NULL;
  fb_status_t status;
  double ab[4] = {1.0, 2.0, 3.0, 4.0};
  double b[2] = {5.0, 6.0};

  memset(&vtable, 0, sizeof(vtable));
  g_dpbsvx_bridge_calls = 0;
  vtable.ext_ops[FB_OP_DPBSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dpbsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  dpbsv_call = (fb_dpbsv_cblas_fn)vtable.ext_ops[FB_OP_DPBSV][FB_CONV_CBLAS];
  if (!dpbsv_call) {
    fprintf(stderr, "[FAIL] DPBSV adapter slot missing after finalize\n");
    return 1;
  }

  if (dpbsv_call(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, 1, ab, 2, b, 2) != 0) {
    fprintf(stderr, "[FAIL] DPBSV adapter bridge from DPBSVX returned error\n");
    return 1;
  }

  if (g_dpbsvx_bridge_calls != 1) {
    fprintf(stderr,
            "[FAIL] DPBSV<-DPBSVX bridge propagation mismatch (calls=%d)\n",
            g_dpbsvx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter bridge DPBSV <- DPBSVX applies neutral expert params\n");
  return 0;
}

static int test_adapter_bridge_cpbsv_from_cpbsvx(void) {
  fb_backend_vtable_t vtable;
  fb_cpbsv_cblas_fn cpbsv_call = NULL;
  fb_status_t status;
  fb_complex_float_t ab[4] = {0};
  fb_complex_float_t b[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_cpbsvx_bridge_calls = 0;
  vtable.ext_ops[FB_OP_CPBSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_cpbsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  cpbsv_call = (fb_cpbsv_cblas_fn)vtable.ext_ops[FB_OP_CPBSV][FB_CONV_CBLAS];
  if (!cpbsv_call) {
    fprintf(stderr, "[FAIL] CPBSV adapter slot missing after finalize\n");
    return 1;
  }

  if (cpbsv_call(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, 1, ab, 2, b, 2) != 0) {
    fprintf(stderr, "[FAIL] CPBSV adapter bridge from CPBSVX returned error\n");
    return 1;
  }

  if (g_cpbsvx_bridge_calls != 1) {
    fprintf(stderr,
            "[FAIL] CPBSV<-CPBSVX bridge propagation mismatch (calls=%d)\n",
            g_cpbsvx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter bridge CPBSV <- CPBSVX applies neutral expert params\n");
  return 0;
}

static int test_adapter_bridge_zpbsv_from_zpbsvx(void) {
  fb_backend_vtable_t vtable;
  fb_zpbsv_cblas_fn zpbsv_call = NULL;
  fb_status_t status;
  fb_complex_double_t ab[4] = {0};
  fb_complex_double_t b[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_zpbsvx_bridge_calls = 0;
  vtable.ext_ops[FB_OP_ZPBSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zpbsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  zpbsv_call = (fb_zpbsv_cblas_fn)vtable.ext_ops[FB_OP_ZPBSV][FB_CONV_CBLAS];
  if (!zpbsv_call) {
    fprintf(stderr, "[FAIL] ZPBSV adapter slot missing after finalize\n");
    return 1;
  }

  if (zpbsv_call(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, 1, ab, 2, b, 2) != 0) {
    fprintf(stderr, "[FAIL] ZPBSV adapter bridge from ZPBSVX returned error\n");
    return 1;
  }

  if (g_zpbsvx_bridge_calls != 1) {
    fprintf(stderr,
            "[FAIL] ZPBSV<-ZPBSVX bridge propagation mismatch (calls=%d)\n",
            g_zpbsvx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter bridge ZPBSV <- ZPBSVX applies neutral expert params\n");
  return 0;
}

static int test_adapter_bridge_sppsv_from_sppsvx(void) {
  fb_backend_vtable_t vtable;
  fb_sppsv_cblas_fn sppsv_call = NULL;
  fb_status_t status;
  float ap[3] = {1.0f, 2.0f, 3.0f};
  float b[2] = {4.0f, 5.0f};

  memset(&vtable, 0, sizeof(vtable));
  g_sppsvx_bridge_calls = 0;
  vtable.ext_ops[FB_OP_SPPSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sppsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  sppsv_call = (fb_sppsv_cblas_fn)vtable.ext_ops[FB_OP_SPPSV][FB_CONV_CBLAS];
  if (!sppsv_call) {
    fprintf(stderr, "[FAIL] SPPSV adapter slot missing after finalize\n");
    return 1;
  }

  if (sppsv_call(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, ap, b, 2) != 0) {
    fprintf(stderr, "[FAIL] SPPSV adapter bridge from SPPSVX returned error\n");
    return 1;
  }

  if (g_sppsvx_bridge_calls != 1) {
    fprintf(stderr,
            "[FAIL] SPPSV<-SPPSVX bridge propagation mismatch (calls=%d)\n",
            g_sppsvx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter bridge SPPSV <- SPPSVX applies neutral expert params\n");
  return 0;
}

static int test_adapter_bridge_dppsv_from_dppsvx(void) {
  fb_backend_vtable_t vtable;
  fb_dppsv_cblas_fn dppsv_call = NULL;
  fb_status_t status;
  double ap[3] = {1.0, 2.0, 3.0};
  double b[2] = {4.0, 5.0};

  memset(&vtable, 0, sizeof(vtable));
  g_dppsvx_bridge_calls = 0;
  vtable.ext_ops[FB_OP_DPPSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dppsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  dppsv_call = (fb_dppsv_cblas_fn)vtable.ext_ops[FB_OP_DPPSV][FB_CONV_CBLAS];
  if (!dppsv_call) {
    fprintf(stderr, "[FAIL] DPPSV adapter slot missing after finalize\n");
    return 1;
  }

  if (dppsv_call(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, ap, b, 2) != 0) {
    fprintf(stderr, "[FAIL] DPPSV adapter bridge from DPPSVX returned error\n");
    return 1;
  }

  if (g_dppsvx_bridge_calls != 1) {
    fprintf(stderr,
            "[FAIL] DPPSV<-DPPSVX bridge propagation mismatch (calls=%d)\n",
            g_dppsvx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter bridge DPPSV <- DPPSVX applies neutral expert params\n");
  return 0;
}

static int test_adapter_bridge_cppsv_from_cppsvx(void) {
  fb_backend_vtable_t vtable;
  fb_cppsv_cblas_fn cppsv_call = NULL;
  fb_status_t status;
  fb_complex_float_t ap[3] = {0};
  fb_complex_float_t b[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_cppsvx_bridge_calls = 0;
  vtable.ext_ops[FB_OP_CPPSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_cppsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  cppsv_call = (fb_cppsv_cblas_fn)vtable.ext_ops[FB_OP_CPPSV][FB_CONV_CBLAS];
  if (!cppsv_call) {
    fprintf(stderr, "[FAIL] CPPSV adapter slot missing after finalize\n");
    return 1;
  }

  if (cppsv_call(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, ap, b, 2) != 0) {
    fprintf(stderr, "[FAIL] CPPSV adapter bridge from CPPSVX returned error\n");
    return 1;
  }

  if (g_cppsvx_bridge_calls != 1) {
    fprintf(stderr,
            "[FAIL] CPPSV<-CPPSVX bridge propagation mismatch (calls=%d)\n",
            g_cppsvx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter bridge CPPSV <- CPPSVX applies neutral expert params\n");
  return 0;
}

static int test_adapter_bridge_zppsv_from_zppsvx(void) {
  fb_backend_vtable_t vtable;
  fb_zppsv_cblas_fn zppsv_call = NULL;
  fb_status_t status;
  fb_complex_double_t ap[3] = {0};
  fb_complex_double_t b[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_zppsvx_bridge_calls = 0;
  vtable.ext_ops[FB_OP_ZPPSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zppsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  zppsv_call = (fb_zppsv_cblas_fn)vtable.ext_ops[FB_OP_ZPPSV][FB_CONV_CBLAS];
  if (!zppsv_call) {
    fprintf(stderr, "[FAIL] ZPPSV adapter slot missing after finalize\n");
    return 1;
  }

  if (zppsv_call(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, ap, b, 2) != 0) {
    fprintf(stderr, "[FAIL] ZPPSV adapter bridge from ZPPSVX returned error\n");
    return 1;
  }

  if (g_zppsvx_bridge_calls != 1) {
    fprintf(stderr,
            "[FAIL] ZPPSV<-ZPPSVX bridge propagation mismatch (calls=%d)\n",
            g_zppsvx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter bridge ZPPSV <- ZPPSVX applies neutral expert params\n");
  return 0;
}

static int test_adapter_bridge_sspsv_from_sspsvx(void) {
  fb_backend_vtable_t vtable;
  fb_sspsv_cblas_fn sspsv_call = NULL;
  fb_status_t status;
  float ap[3] = {1.0f, 2.0f, 3.0f};
  float b[2] = {4.0f, 5.0f};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_sspsvx_bridge_calls = 0;
  vtable.ext_ops[FB_OP_SSPSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sspsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  sspsv_call = (fb_sspsv_cblas_fn)vtable.ext_ops[FB_OP_SSPSV][FB_CONV_CBLAS];
  if (!sspsv_call) {
    fprintf(stderr, "[FAIL] SSPSV adapter slot missing after finalize\n");
    return 1;
  }

  if (sspsv_call(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, ap, ipiv, b, 2) !=
      0) {
    fprintf(stderr, "[FAIL] SSPSV adapter bridge from SSPSVX returned error\n");
    return 1;
  }

  if (g_sspsvx_bridge_calls != 1) {
    fprintf(stderr,
            "[FAIL] SSPSV<-SSPSVX bridge propagation mismatch (calls=%d)\n",
            g_sspsvx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter bridge SSPSV <- SSPSVX applies neutral expert params\n");
  return 0;
}

static int test_adapter_bridge_dspsv_from_dspsvx(void) {
  fb_backend_vtable_t vtable;
  fb_dspsv_cblas_fn dspsv_call = NULL;
  fb_status_t status;
  double ap[3] = {1.0, 2.0, 3.0};
  double b[2] = {4.0, 5.0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_dspsvx_bridge_calls = 0;
  vtable.ext_ops[FB_OP_DSPSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dspsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  dspsv_call = (fb_dspsv_cblas_fn)vtable.ext_ops[FB_OP_DSPSV][FB_CONV_CBLAS];
  if (!dspsv_call) {
    fprintf(stderr, "[FAIL] DSPSV adapter slot missing after finalize\n");
    return 1;
  }

  if (dspsv_call(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, ap, ipiv, b, 2) !=
      0) {
    fprintf(stderr, "[FAIL] DSPSV adapter bridge from DSPSVX returned error\n");
    return 1;
  }

  if (g_dspsvx_bridge_calls != 1) {
    fprintf(stderr,
            "[FAIL] DSPSV<-DSPSVX bridge propagation mismatch (calls=%d)\n",
            g_dspsvx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter bridge DSPSV <- DSPSVX applies neutral expert params\n");
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

static int test_adapter_does_not_override_existing_sgesv_from_sgesvxx(void) {
  fb_backend_vtable_t vtable;
  fb_sgesv_fn sgesv_call = NULL;
  fb_status_t status;

  float a[4] = {0};
  float b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_sgesv_existing_calls = 0;
  g_sgesvxx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_SGESV][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sgesv_existing;
  vtable.ext_ops[FB_OP_SGESVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sgesvxx_base;

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

  if (g_sgesv_existing_calls != 1 || g_sgesvxx_bridge_calls != 0) {
    fprintf(stderr,
            "[FAIL] SGESV existing-preserve mismatch via SGESVXX (existing=%d adapter=%d)\n",
            g_sgesv_existing_calls, g_sgesvxx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter does not override existing SGESV with SGESVXX donor\n");
  return 0;
}

static int test_adapter_does_not_override_existing_dgesv_from_dgesvxx(void) {
  fb_backend_vtable_t vtable;
  fb_dgesv_fn dgesv_call = NULL;
  fb_status_t status;

  double a[4] = {0};
  double b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_dgesv_existing_calls = 0;
  g_dgesvxx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_DGESV][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dgesv_existing;
  vtable.ext_ops[FB_OP_DGESVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dgesvxx_base;

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

  if (g_dgesv_existing_calls != 1 || g_dgesvxx_bridge_calls != 0) {
    fprintf(stderr,
            "[FAIL] DGESV existing-preserve mismatch via DGESVXX (existing=%d adapter=%d)\n",
            g_dgesv_existing_calls, g_dgesvxx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter does not override existing DGESV with DGESVXX donor\n");
  return 0;
}

static int test_adapter_does_not_override_existing_cgesv_from_cgesvxx(void) {
  fb_backend_vtable_t vtable;
  fb_cgesv_fn cgesv_call = NULL;
  fb_status_t status;

  fb_complex_float_t a[4] = {0};
  fb_complex_float_t b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_cgesv_existing_calls = 0;
  g_cgesvxx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_CGESV][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_cgesv_existing;
  vtable.ext_ops[FB_OP_CGESVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_cgesvxx_base;

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

  if (g_cgesv_existing_calls != 1 || g_cgesvxx_bridge_calls != 0) {
    fprintf(stderr,
            "[FAIL] CGESV existing-preserve mismatch via CGESVXX (existing=%d adapter=%d)\n",
            g_cgesv_existing_calls, g_cgesvxx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter does not override existing CGESV with CGESVXX donor\n");
  return 0;
}

static int test_adapter_does_not_override_existing_zgesv_from_zgesvxx(void) {
  fb_backend_vtable_t vtable;
  fb_zgesv_fn zgesv_call = NULL;
  fb_status_t status;

  fb_complex_double_t a[4] = {0};
  fb_complex_double_t b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_zgesv_existing_calls = 0;
  g_zgesvxx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_ZGESV][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zgesv_existing;
  vtable.ext_ops[FB_OP_ZGESVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zgesvxx_base;

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

  if (g_zgesv_existing_calls != 1 || g_zgesvxx_bridge_calls != 0) {
    fprintf(stderr,
            "[FAIL] ZGESV existing-preserve mismatch via ZGESVXX (existing=%d adapter=%d)\n",
            g_zgesv_existing_calls, g_zgesvxx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter does not override existing ZGESV with ZGESVXX donor\n");
  return 0;
}

static int test_adapter_does_not_override_existing_sposv_from_sposvxx(void) {
  fb_backend_vtable_t vtable;
  fb_sposv_fn sposv_call = NULL;
  fb_status_t status;

  float a[4] = {0};
  float b[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_sposv_existing_calls = 0;
  g_sposvxx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_SPOSV][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sposv_existing;
  vtable.ext_ops[FB_OP_SPOSVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sposvxx_base;

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

  if (g_sposv_existing_calls != 1 || g_sposvxx_bridge_calls != 0) {
    fprintf(stderr,
            "[FAIL] SPOSV existing-preserve mismatch via SPOSVXX (existing=%d adapter=%d)\n",
            g_sposv_existing_calls, g_sposvxx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter does not override existing SPOSV with SPOSVXX donor\n");
  return 0;
}

static int test_adapter_does_not_override_existing_dposv_from_dposvxx(void) {
  fb_backend_vtable_t vtable;
  fb_dposv_fn dposv_call = NULL;
  fb_status_t status;

  double a[4] = {0};
  double b[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_dposv_existing_calls = 0;
  g_dposvxx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_DPOSV][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dposv_existing;
  vtable.ext_ops[FB_OP_DPOSVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dposvxx_base;

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

  if (g_dposv_existing_calls != 1 || g_dposvxx_bridge_calls != 0) {
    fprintf(stderr,
            "[FAIL] DPOSV existing-preserve mismatch via DPOSVXX (existing=%d adapter=%d)\n",
            g_dposv_existing_calls, g_dposvxx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter does not override existing DPOSV with DPOSVXX donor\n");
  return 0;
}

static int test_adapter_does_not_override_existing_cposv_from_cposvxx(void) {
  fb_backend_vtable_t vtable;
  fb_cposv_fn cposv_call = NULL;
  fb_status_t status;

  fb_complex_float_t a[4] = {0};
  fb_complex_float_t b[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_cposv_existing_calls = 0;
  g_cposvxx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_CPOSV][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_cposv_existing;
  vtable.ext_ops[FB_OP_CPOSVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_cposvxx_base;

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

  if (g_cposv_existing_calls != 1 || g_cposvxx_bridge_calls != 0) {
    fprintf(stderr,
            "[FAIL] CPOSV existing-preserve mismatch via CPOSVXX (existing=%d adapter=%d)\n",
            g_cposv_existing_calls, g_cposvxx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter does not override existing CPOSV with CPOSVXX donor\n");
  return 0;
}

static int test_adapter_does_not_override_existing_zposv_from_zposvxx(void) {
  fb_backend_vtable_t vtable;
  fb_zposv_fn zposv_call = NULL;
  fb_status_t status;

  fb_complex_double_t a[4] = {0};
  fb_complex_double_t b[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_zposv_existing_calls = 0;
  g_zposvxx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_ZPOSV][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zposv_existing;
  vtable.ext_ops[FB_OP_ZPOSVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zposvxx_base;

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

  if (g_zposv_existing_calls != 1 || g_zposvxx_bridge_calls != 0) {
    fprintf(stderr,
            "[FAIL] ZPOSV existing-preserve mismatch via ZPOSVXX (existing=%d adapter=%d)\n",
            g_zposv_existing_calls, g_zposvxx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter does not override existing ZPOSV with ZPOSVXX donor\n");
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

static int test_adapter_does_not_override_existing_chesv_from_chesvx(void) {
  fb_backend_vtable_t vtable;
  fb_csysv_fn chesv_call = NULL;
  fb_status_t status;

  fb_complex_float_t a[4] = {0};
  fb_complex_float_t b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_csysv_existing_calls = 0;
  g_csysvx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_CHESV][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_csysv_existing;
  vtable.ext_ops[FB_OP_CHESVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_csysvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  chesv_call = (fb_csysv_fn)vtable.ext_ops[FB_OP_CHESV][FB_CONV_CBLAS];
  if (!chesv_call) {
    fprintf(stderr, "[FAIL] CHESV slot missing after finalize\n");
    return 1;
  }

  if (chesv_call(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, a, 2, ipiv, b,
                 2) != 995) {
    fprintf(stderr,
            "[FAIL] adapter unexpectedly replaced existing CHESV implementation\n");
    return 1;
  }

  if (g_csysv_existing_calls != 1 || g_csysvx_bridge_calls != 0) {
    fprintf(stderr,
            "[FAIL] CHESV existing-preserve mismatch via CHESVX (existing=%d adapter=%d)\n",
            g_csysv_existing_calls, g_csysvx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter does not override existing CHESV with CHESVX donor\n");
  return 0;
}

static int test_adapter_does_not_override_existing_zhesv_from_zhesvx(void) {
  fb_backend_vtable_t vtable;
  fb_zsysv_fn zhesv_call = NULL;
  fb_status_t status;

  fb_complex_double_t a[4] = {0};
  fb_complex_double_t b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_zsysv_existing_calls = 0;
  g_zsysvx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_ZHESV][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zsysv_existing;
  vtable.ext_ops[FB_OP_ZHESVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zsysvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  zhesv_call = (fb_zsysv_fn)vtable.ext_ops[FB_OP_ZHESV][FB_CONV_CBLAS];
  if (!zhesv_call) {
    fprintf(stderr, "[FAIL] ZHESV slot missing after finalize\n");
    return 1;
  }

  if (zhesv_call(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, a, 2, ipiv, b,
                 2) != 1005) {
    fprintf(stderr,
            "[FAIL] adapter unexpectedly replaced existing ZHESV implementation\n");
    return 1;
  }

  if (g_zsysv_existing_calls != 1 || g_zsysvx_bridge_calls != 0) {
    fprintf(stderr,
            "[FAIL] ZHESV existing-preserve mismatch via ZHESVX (existing=%d adapter=%d)\n",
            g_zsysv_existing_calls, g_zsysvx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter does not override existing ZHESV with ZHESVX donor\n");
  return 0;
}

static int test_adapter_does_not_override_existing_ssysv_from_ssysvxx(void) {
  fb_backend_vtable_t vtable;
  fb_ssysv_fn ssysv_call = NULL;
  fb_status_t status;

  float a[4] = {0};
  float b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_ssysv_existing_calls = 0;
  g_ssysvxx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_SSYSV][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_ssysv_existing;
  vtable.ext_ops[FB_OP_SSYSVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_ssysvxx_base;

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

  if (g_ssysv_existing_calls != 1 || g_ssysvxx_bridge_calls != 0) {
    fprintf(stderr,
            "[FAIL] SSYSV existing-preserve mismatch via SSYSVXX (existing=%d adapter=%d)\n",
            g_ssysv_existing_calls, g_ssysvxx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter does not override existing SSYSV with SSYSVXX donor\n");
  return 0;
}

static int test_adapter_does_not_override_existing_dsysv_from_dsysvxx(void) {
  fb_backend_vtable_t vtable;
  fb_dsysv_fn dsysv_call = NULL;
  fb_status_t status;

  double a[4] = {0};
  double b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_dsysv_existing_calls = 0;
  g_dsysvxx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_DSYSV][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dsysv_existing;
  vtable.ext_ops[FB_OP_DSYSVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dsysvxx_base;

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

  if (g_dsysv_existing_calls != 1 || g_dsysvxx_bridge_calls != 0) {
    fprintf(stderr,
            "[FAIL] DSYSV existing-preserve mismatch via DSYSVXX (existing=%d adapter=%d)\n",
            g_dsysv_existing_calls, g_dsysvxx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter does not override existing DSYSV with DSYSVXX donor\n");
  return 0;
}

static int test_adapter_does_not_override_existing_csysv_from_csysvxx(void) {
  fb_backend_vtable_t vtable;
  fb_csysv_fn csysv_call = NULL;
  fb_status_t status;

  fb_complex_float_t a[4] = {0};
  fb_complex_float_t b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_csysv_existing_calls = 0;
  g_csysvxx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_CSYSV][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_csysv_existing;
  vtable.ext_ops[FB_OP_CSYSVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_csysvxx_base;

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

  if (g_csysv_existing_calls != 1 || g_csysvxx_bridge_calls != 0) {
    fprintf(stderr,
            "[FAIL] CSYSV existing-preserve mismatch via CSYSVXX (existing=%d adapter=%d)\n",
            g_csysv_existing_calls, g_csysvxx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter does not override existing CSYSV with CSYSVXX donor\n");
  return 0;
}

static int test_adapter_does_not_override_existing_zsysv_from_zsysvxx(void) {
  fb_backend_vtable_t vtable;
  fb_zsysv_fn zsysv_call = NULL;
  fb_status_t status;

  fb_complex_double_t a[4] = {0};
  fb_complex_double_t b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_zsysv_existing_calls = 0;
  g_zsysvxx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_ZSYSV][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zsysv_existing;
  vtable.ext_ops[FB_OP_ZSYSVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zsysvxx_base;

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

  if (g_zsysv_existing_calls != 1 || g_zsysvxx_bridge_calls != 0) {
    fprintf(stderr,
            "[FAIL] ZSYSV existing-preserve mismatch via ZSYSVXX (existing=%d adapter=%d)\n",
            g_zsysv_existing_calls, g_zsysvxx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter does not override existing ZSYSV with ZSYSVXX donor\n");
  return 0;
}

static int test_adapter_does_not_override_existing_chesv_from_chesvxx(void) {
  fb_backend_vtable_t vtable;
  fb_csysv_fn chesv_call = NULL;
  fb_status_t status;

  fb_complex_float_t a[4] = {0};
  fb_complex_float_t b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_csysv_existing_calls = 0;
  g_csysvxx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_CHESV][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_csysv_existing;
  vtable.ext_ops[FB_OP_CHESVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_csysvxx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  chesv_call = (fb_csysv_fn)vtable.ext_ops[FB_OP_CHESV][FB_CONV_CBLAS];
  if (!chesv_call) {
    fprintf(stderr, "[FAIL] CHESV slot missing after finalize\n");
    return 1;
  }

  if (chesv_call(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, a, 2, ipiv, b,
                 2) != 995) {
    fprintf(stderr,
            "[FAIL] adapter unexpectedly replaced existing CHESV implementation\n");
    return 1;
  }

  if (g_csysv_existing_calls != 1 || g_csysvxx_bridge_calls != 0) {
    fprintf(stderr,
            "[FAIL] CHESV existing-preserve mismatch via CHESVXX (existing=%d adapter=%d)\n",
            g_csysv_existing_calls, g_csysvxx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter does not override existing CHESV with CHESVXX donor\n");
  return 0;
}

static int test_adapter_does_not_override_existing_zhesv_from_zhesvxx(void) {
  fb_backend_vtable_t vtable;
  fb_zsysv_fn zhesv_call = NULL;
  fb_status_t status;

  fb_complex_double_t a[4] = {0};
  fb_complex_double_t b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_zsysv_existing_calls = 0;
  g_zsysvxx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_ZHESV][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zsysv_existing;
  vtable.ext_ops[FB_OP_ZHESVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zsysvxx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  zhesv_call = (fb_zsysv_fn)vtable.ext_ops[FB_OP_ZHESV][FB_CONV_CBLAS];
  if (!zhesv_call) {
    fprintf(stderr, "[FAIL] ZHESV slot missing after finalize\n");
    return 1;
  }

  if (zhesv_call(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, a, 2, ipiv, b,
                 2) != 1005) {
    fprintf(stderr,
            "[FAIL] adapter unexpectedly replaced existing ZHESV implementation\n");
    return 1;
  }

  if (g_zsysv_existing_calls != 1 || g_zsysvxx_bridge_calls != 0) {
    fprintf(stderr,
            "[FAIL] ZHESV existing-preserve mismatch via ZHESVXX (existing=%d adapter=%d)\n",
            g_zsysv_existing_calls, g_zsysvxx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter does not override existing ZHESV with ZHESVXX donor\n");
  return 0;
}

static int test_adapter_does_not_override_existing_sgbsv(void) {
  fb_backend_vtable_t vtable;
  fb_sgbsv_cblas_fn sgbsv_call = NULL;
  fb_status_t status;

  float ab[8] = {0};
  float b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_sgbsv_existing_calls = 0;
  g_sgbsvx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_SGBSV][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sgbsv_existing;
  vtable.ext_ops[FB_OP_SGBSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sgbsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  sgbsv_call = (fb_sgbsv_cblas_fn)vtable.ext_ops[FB_OP_SGBSV][FB_CONV_CBLAS];
  if (!sgbsv_call) {
    fprintf(stderr, "[FAIL] SGBSV slot missing after finalize\n");
    return 1;
  }

  if (sgbsv_call(FB_LAYOUT_COL_MAJOR, 2, 1, 1, 1, ab, 4, ipiv, b, 2) != 1055) {
    fprintf(stderr,
            "[FAIL] adapter unexpectedly replaced existing SGBSV implementation\n");
    return 1;
  }

  if (g_sgbsv_existing_calls != 1 || g_sgbsvx_bridge_calls != 0) {
    fprintf(stderr,
            "[FAIL] SGBSV existing-preserve mismatch (existing=%d adapter=%d)\n",
            g_sgbsv_existing_calls, g_sgbsvx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter does not override existing SGBSV implementation\n");
  return 0;
}

static int test_adapter_does_not_override_existing_dgbsv(void) {
  fb_backend_vtable_t vtable;
  fb_dgbsv_cblas_fn dgbsv_call = NULL;
  fb_status_t status;

  double ab[8] = {0};
  double b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_dgbsv_existing_calls = 0;
  g_dgbsvx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_DGBSV][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dgbsv_existing;
  vtable.ext_ops[FB_OP_DGBSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dgbsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  dgbsv_call = (fb_dgbsv_cblas_fn)vtable.ext_ops[FB_OP_DGBSV][FB_CONV_CBLAS];
  if (!dgbsv_call) {
    fprintf(stderr, "[FAIL] DGBSV slot missing after finalize\n");
    return 1;
  }

  if (dgbsv_call(FB_LAYOUT_COL_MAJOR, 2, 1, 1, 1, ab, 4, ipiv, b, 2) != 1065) {
    fprintf(stderr,
            "[FAIL] adapter unexpectedly replaced existing DGBSV implementation\n");
    return 1;
  }

  if (g_dgbsv_existing_calls != 1 || g_dgbsvx_bridge_calls != 0) {
    fprintf(stderr,
            "[FAIL] DGBSV existing-preserve mismatch (existing=%d adapter=%d)\n",
            g_dgbsv_existing_calls, g_dgbsvx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter does not override existing DGBSV implementation\n");
  return 0;
}

static int test_adapter_does_not_override_existing_cgbsv(void) {
  fb_backend_vtable_t vtable;
  fb_cgbsv_cblas_fn cgbsv_call = NULL;
  fb_status_t status;

  fb_complex_float_t ab[8] = {0};
  fb_complex_float_t b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_cgbsv_existing_calls = 0;
  g_cgbsvx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_CGBSV][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_cgbsv_existing;
  vtable.ext_ops[FB_OP_CGBSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_cgbsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  cgbsv_call = (fb_cgbsv_cblas_fn)vtable.ext_ops[FB_OP_CGBSV][FB_CONV_CBLAS];
  if (!cgbsv_call) {
    fprintf(stderr, "[FAIL] CGBSV slot missing after finalize\n");
    return 1;
  }

  if (cgbsv_call(FB_LAYOUT_COL_MAJOR, 2, 1, 1, 1, ab, 4, ipiv, b, 2) != 1075) {
    fprintf(stderr,
            "[FAIL] adapter unexpectedly replaced existing CGBSV implementation\n");
    return 1;
  }

  if (g_cgbsv_existing_calls != 1 || g_cgbsvx_bridge_calls != 0) {
    fprintf(stderr,
            "[FAIL] CGBSV existing-preserve mismatch (existing=%d adapter=%d)\n",
            g_cgbsv_existing_calls, g_cgbsvx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter does not override existing CGBSV implementation\n");
  return 0;
}

static int test_adapter_does_not_override_existing_zgbsv(void) {
  fb_backend_vtable_t vtable;
  fb_zgbsv_cblas_fn zgbsv_call = NULL;
  fb_status_t status;

  fb_complex_double_t ab[8] = {0};
  fb_complex_double_t b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_zgbsv_existing_calls = 0;
  g_zgbsvx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_ZGBSV][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zgbsv_existing;
  vtable.ext_ops[FB_OP_ZGBSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zgbsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  zgbsv_call = (fb_zgbsv_cblas_fn)vtable.ext_ops[FB_OP_ZGBSV][FB_CONV_CBLAS];
  if (!zgbsv_call) {
    fprintf(stderr, "[FAIL] ZGBSV slot missing after finalize\n");
    return 1;
  }

  if (zgbsv_call(FB_LAYOUT_COL_MAJOR, 2, 1, 1, 1, ab, 4, ipiv, b, 2) != 1085) {
    fprintf(stderr,
            "[FAIL] adapter unexpectedly replaced existing ZGBSV implementation\n");
    return 1;
  }

  if (g_zgbsv_existing_calls != 1 || g_zgbsvx_bridge_calls != 0) {
    fprintf(stderr,
            "[FAIL] ZGBSV existing-preserve mismatch (existing=%d adapter=%d)\n",
            g_zgbsv_existing_calls, g_zgbsvx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter does not override existing ZGBSV implementation\n");
  return 0;
}

static int test_adapter_does_not_override_existing_sgtsv(void) {
  fb_backend_vtable_t vtable;
  fb_sgtsv_cblas_fn sgtsv_call = NULL;
  fb_status_t status;

  float dl[2] = {0};
  float d[2] = {0};
  float du[2] = {0};
  float b[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_sgtsv_existing_calls = 0;
  g_sgtsvx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_SGTSV][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sgtsv_existing;
  vtable.ext_ops[FB_OP_SGTSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sgtsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  sgtsv_call = (fb_sgtsv_cblas_fn)vtable.ext_ops[FB_OP_SGTSV][FB_CONV_CBLAS];
  if (!sgtsv_call) {
    fprintf(stderr, "[FAIL] SGTSV slot missing after finalize\n");
    return 1;
  }

  if (sgtsv_call(FB_LAYOUT_COL_MAJOR, 2, 1, dl, d, du, b, 2) != 1095) {
    fprintf(stderr,
            "[FAIL] adapter unexpectedly replaced existing SGTSV implementation\n");
    return 1;
  }

  if (g_sgtsv_existing_calls != 1 || g_sgtsvx_bridge_calls != 0) {
    fprintf(stderr,
            "[FAIL] SGTSV existing-preserve mismatch (existing=%d adapter=%d)\n",
            g_sgtsv_existing_calls, g_sgtsvx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter does not override existing SGTSV implementation\n");
  return 0;
}

static int test_adapter_does_not_override_existing_dgtsv(void) {
  fb_backend_vtable_t vtable;
  fb_dgtsv_cblas_fn dgtsv_call = NULL;
  fb_status_t status;

  double dl[2] = {0};
  double d[2] = {0};
  double du[2] = {0};
  double b[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_dgtsv_existing_calls = 0;
  g_dgtsvx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_DGTSV][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dgtsv_existing;
  vtable.ext_ops[FB_OP_DGTSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dgtsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  dgtsv_call = (fb_dgtsv_cblas_fn)vtable.ext_ops[FB_OP_DGTSV][FB_CONV_CBLAS];
  if (!dgtsv_call) {
    fprintf(stderr, "[FAIL] DGTSV slot missing after finalize\n");
    return 1;
  }

  if (dgtsv_call(FB_LAYOUT_COL_MAJOR, 2, 1, dl, d, du, b, 2) != 1105) {
    fprintf(stderr,
            "[FAIL] adapter unexpectedly replaced existing DGTSV implementation\n");
    return 1;
  }

  if (g_dgtsv_existing_calls != 1 || g_dgtsvx_bridge_calls != 0) {
    fprintf(stderr,
            "[FAIL] DGTSV existing-preserve mismatch (existing=%d adapter=%d)\n",
            g_dgtsv_existing_calls, g_dgtsvx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter does not override existing DGTSV implementation\n");
  return 0;
}

static int test_adapter_does_not_override_existing_cgtsv(void) {
  fb_backend_vtable_t vtable;
  fb_cgtsv_cblas_fn cgtsv_call = NULL;
  fb_status_t status;

  fb_complex_float_t dl[2] = {0};
  fb_complex_float_t d[2] = {0};
  fb_complex_float_t du[2] = {0};
  fb_complex_float_t b[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_cgtsv_existing_calls = 0;
  g_cgtsvx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_CGTSV][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_cgtsv_existing;
  vtable.ext_ops[FB_OP_CGTSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_cgtsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  cgtsv_call = (fb_cgtsv_cblas_fn)vtable.ext_ops[FB_OP_CGTSV][FB_CONV_CBLAS];
  if (!cgtsv_call) {
    fprintf(stderr, "[FAIL] CGTSV slot missing after finalize\n");
    return 1;
  }

  if (cgtsv_call(FB_LAYOUT_COL_MAJOR, 2, 1, dl, d, du, b, 2) != 1115) {
    fprintf(stderr,
            "[FAIL] adapter unexpectedly replaced existing CGTSV implementation\n");
    return 1;
  }

  if (g_cgtsv_existing_calls != 1 || g_cgtsvx_bridge_calls != 0) {
    fprintf(stderr,
            "[FAIL] CGTSV existing-preserve mismatch (existing=%d adapter=%d)\n",
            g_cgtsv_existing_calls, g_cgtsvx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter does not override existing CGTSV implementation\n");
  return 0;
}

static int test_adapter_does_not_override_existing_zgtsv(void) {
  fb_backend_vtable_t vtable;
  fb_zgtsv_cblas_fn zgtsv_call = NULL;
  fb_status_t status;

  fb_complex_double_t dl[2] = {0};
  fb_complex_double_t d[2] = {0};
  fb_complex_double_t du[2] = {0};
  fb_complex_double_t b[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_zgtsv_existing_calls = 0;
  g_zgtsvx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_ZGTSV][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zgtsv_existing;
  vtable.ext_ops[FB_OP_ZGTSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zgtsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  zgtsv_call = (fb_zgtsv_cblas_fn)vtable.ext_ops[FB_OP_ZGTSV][FB_CONV_CBLAS];
  if (!zgtsv_call) {
    fprintf(stderr, "[FAIL] ZGTSV slot missing after finalize\n");
    return 1;
  }

  if (zgtsv_call(FB_LAYOUT_COL_MAJOR, 2, 1, dl, d, du, b, 2) != 1125) {
    fprintf(stderr,
            "[FAIL] adapter unexpectedly replaced existing ZGTSV implementation\n");
    return 1;
  }

  if (g_zgtsv_existing_calls != 1 || g_zgtsvx_bridge_calls != 0) {
    fprintf(stderr,
            "[FAIL] ZGTSV existing-preserve mismatch (existing=%d adapter=%d)\n",
            g_zgtsv_existing_calls, g_zgtsvx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter does not override existing ZGTSV implementation\n");
  return 0;
}

static int test_adapter_does_not_override_existing_sptsv(void) {
  fb_backend_vtable_t vtable;
  fb_sptsv_cblas_fn sptsv_call = NULL;
  fb_status_t status;

  float d[2] = {0};
  float e[2] = {0};
  float b[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_sptsv_existing_calls = 0;
  g_sptsvx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_SPTSV][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sptsv_existing;
  vtable.ext_ops[FB_OP_SPTSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sptsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  sptsv_call = (fb_sptsv_cblas_fn)vtable.ext_ops[FB_OP_SPTSV][FB_CONV_CBLAS];
  if (!sptsv_call) {
    fprintf(stderr, "[FAIL] SPTSV slot missing after finalize\n");
    return 1;
  }

  if (sptsv_call(FB_LAYOUT_COL_MAJOR, 2, 1, d, e, b, 2) != 1135) {
    fprintf(stderr,
            "[FAIL] adapter unexpectedly replaced existing SPTSV implementation\n");
    return 1;
  }

  if (g_sptsv_existing_calls != 1 || g_sptsvx_bridge_calls != 0) {
    fprintf(stderr,
            "[FAIL] SPTSV existing-preserve mismatch (existing=%d adapter=%d)\n",
            g_sptsv_existing_calls, g_sptsvx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter does not override existing SPTSV implementation\n");
  return 0;
}

static int test_adapter_does_not_override_existing_dptsv(void) {
  fb_backend_vtable_t vtable;
  fb_dptsv_cblas_fn dptsv_call = NULL;
  fb_status_t status;

  double d[2] = {0};
  double e[2] = {0};
  double b[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_dptsv_existing_calls = 0;
  g_dptsvx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_DPTSV][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dptsv_existing;
  vtable.ext_ops[FB_OP_DPTSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dptsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  dptsv_call = (fb_dptsv_cblas_fn)vtable.ext_ops[FB_OP_DPTSV][FB_CONV_CBLAS];
  if (!dptsv_call) {
    fprintf(stderr, "[FAIL] DPTSV slot missing after finalize\n");
    return 1;
  }

  if (dptsv_call(FB_LAYOUT_COL_MAJOR, 2, 1, d, e, b, 2) != 1145) {
    fprintf(stderr,
            "[FAIL] adapter unexpectedly replaced existing DPTSV implementation\n");
    return 1;
  }

  if (g_dptsv_existing_calls != 1 || g_dptsvx_bridge_calls != 0) {
    fprintf(stderr,
            "[FAIL] DPTSV existing-preserve mismatch (existing=%d adapter=%d)\n",
            g_dptsv_existing_calls, g_dptsvx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter does not override existing DPTSV implementation\n");
  return 0;
}

static int test_adapter_does_not_override_existing_cptsv(void) {
  fb_backend_vtable_t vtable;
  fb_cptsv_cblas_fn cptsv_call = NULL;
  fb_status_t status;

  float d[2] = {0};
  fb_complex_float_t e[2] = {0};
  fb_complex_float_t b[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_cptsv_existing_calls = 0;
  g_cptsvx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_CPTSV][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_cptsv_existing;
  vtable.ext_ops[FB_OP_CPTSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_cptsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  cptsv_call = (fb_cptsv_cblas_fn)vtable.ext_ops[FB_OP_CPTSV][FB_CONV_CBLAS];
  if (!cptsv_call) {
    fprintf(stderr, "[FAIL] CPTSV slot missing after finalize\n");
    return 1;
  }

  if (cptsv_call(FB_LAYOUT_COL_MAJOR, 2, 1, d, e, b, 2) != 1155) {
    fprintf(stderr,
            "[FAIL] adapter unexpectedly replaced existing CPTSV implementation\n");
    return 1;
  }

  if (g_cptsv_existing_calls != 1 || g_cptsvx_bridge_calls != 0) {
    fprintf(stderr,
            "[FAIL] CPTSV existing-preserve mismatch (existing=%d adapter=%d)\n",
            g_cptsv_existing_calls, g_cptsvx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter does not override existing CPTSV implementation\n");
  return 0;
}

static int test_adapter_does_not_override_existing_zptsv(void) {
  fb_backend_vtable_t vtable;
  fb_zptsv_cblas_fn zptsv_call = NULL;
  fb_status_t status;

  double d[2] = {0};
  fb_complex_double_t e[2] = {0};
  fb_complex_double_t b[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_zptsv_existing_calls = 0;
  g_zptsvx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_ZPTSV][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zptsv_existing;
  vtable.ext_ops[FB_OP_ZPTSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zptsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  zptsv_call = (fb_zptsv_cblas_fn)vtable.ext_ops[FB_OP_ZPTSV][FB_CONV_CBLAS];
  if (!zptsv_call) {
    fprintf(stderr, "[FAIL] ZPTSV slot missing after finalize\n");
    return 1;
  }

  if (zptsv_call(FB_LAYOUT_COL_MAJOR, 2, 1, d, e, b, 2) != 1165) {
    fprintf(stderr,
            "[FAIL] adapter unexpectedly replaced existing ZPTSV implementation\n");
    return 1;
  }

  if (g_zptsv_existing_calls != 1 || g_zptsvx_bridge_calls != 0) {
    fprintf(stderr,
            "[FAIL] ZPTSV existing-preserve mismatch (existing=%d adapter=%d)\n",
            g_zptsv_existing_calls, g_zptsvx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter does not override existing ZPTSV implementation\n");
  return 0;
}

static int test_adapter_does_not_override_existing_spbsv(void) {
  fb_backend_vtable_t vtable;
  fb_spbsv_cblas_fn spbsv_call = NULL;
  fb_status_t status;
  float ab[4] = {0};
  float b[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_spbsv_existing_calls = 0;
  g_spbsvx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_SPBSV][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_spbsv_existing;
  vtable.ext_ops[FB_OP_SPBSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_spbsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  spbsv_call = (fb_spbsv_cblas_fn)vtable.ext_ops[FB_OP_SPBSV][FB_CONV_CBLAS];
  if (!spbsv_call) {
    fprintf(stderr, "[FAIL] SPBSV slot missing after finalize\n");
    return 1;
  }

  if (spbsv_call(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, 1, ab, 2, b, 2) !=
      1175) {
    fprintf(stderr,
            "[FAIL] adapter unexpectedly replaced existing SPBSV implementation\n");
    return 1;
  }

  if (g_spbsv_existing_calls != 1 || g_spbsvx_bridge_calls != 0) {
    fprintf(stderr,
            "[FAIL] SPBSV existing-preserve mismatch (existing=%d adapter=%d)\n",
            g_spbsv_existing_calls, g_spbsvx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter does not override existing SPBSV implementation\n");
  return 0;
}

static int test_adapter_does_not_override_existing_dpbsv(void) {
  fb_backend_vtable_t vtable;
  fb_dpbsv_cblas_fn dpbsv_call = NULL;
  fb_status_t status;
  double ab[4] = {0};
  double b[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_dpbsv_existing_calls = 0;
  g_dpbsvx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_DPBSV][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dpbsv_existing;
  vtable.ext_ops[FB_OP_DPBSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dpbsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  dpbsv_call = (fb_dpbsv_cblas_fn)vtable.ext_ops[FB_OP_DPBSV][FB_CONV_CBLAS];
  if (!dpbsv_call) {
    fprintf(stderr, "[FAIL] DPBSV slot missing after finalize\n");
    return 1;
  }

  if (dpbsv_call(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, 1, ab, 2, b, 2) !=
      1185) {
    fprintf(stderr,
            "[FAIL] adapter unexpectedly replaced existing DPBSV implementation\n");
    return 1;
  }

  if (g_dpbsv_existing_calls != 1 || g_dpbsvx_bridge_calls != 0) {
    fprintf(stderr,
            "[FAIL] DPBSV existing-preserve mismatch (existing=%d adapter=%d)\n",
            g_dpbsv_existing_calls, g_dpbsvx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter does not override existing DPBSV implementation\n");
  return 0;
}

static int test_adapter_does_not_override_existing_cpbsv(void) {
  fb_backend_vtable_t vtable;
  fb_cpbsv_cblas_fn cpbsv_call = NULL;
  fb_status_t status;
  fb_complex_float_t ab[4] = {0};
  fb_complex_float_t b[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_cpbsv_existing_calls = 0;
  g_cpbsvx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_CPBSV][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_cpbsv_existing;
  vtable.ext_ops[FB_OP_CPBSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_cpbsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  cpbsv_call = (fb_cpbsv_cblas_fn)vtable.ext_ops[FB_OP_CPBSV][FB_CONV_CBLAS];
  if (!cpbsv_call) {
    fprintf(stderr, "[FAIL] CPBSV slot missing after finalize\n");
    return 1;
  }

  if (cpbsv_call(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, 1, ab, 2, b, 2) !=
      1195) {
    fprintf(stderr,
            "[FAIL] adapter unexpectedly replaced existing CPBSV implementation\n");
    return 1;
  }

  if (g_cpbsv_existing_calls != 1 || g_cpbsvx_bridge_calls != 0) {
    fprintf(stderr,
            "[FAIL] CPBSV existing-preserve mismatch (existing=%d adapter=%d)\n",
            g_cpbsv_existing_calls, g_cpbsvx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter does not override existing CPBSV implementation\n");
  return 0;
}

static int test_adapter_does_not_override_existing_zpbsv(void) {
  fb_backend_vtable_t vtable;
  fb_zpbsv_cblas_fn zpbsv_call = NULL;
  fb_status_t status;
  fb_complex_double_t ab[4] = {0};
  fb_complex_double_t b[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_zpbsv_existing_calls = 0;
  g_zpbsvx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_ZPBSV][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zpbsv_existing;
  vtable.ext_ops[FB_OP_ZPBSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zpbsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  zpbsv_call = (fb_zpbsv_cblas_fn)vtable.ext_ops[FB_OP_ZPBSV][FB_CONV_CBLAS];
  if (!zpbsv_call) {
    fprintf(stderr, "[FAIL] ZPBSV slot missing after finalize\n");
    return 1;
  }

  if (zpbsv_call(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, 1, ab, 2, b, 2) !=
      1205) {
    fprintf(stderr,
            "[FAIL] adapter unexpectedly replaced existing ZPBSV implementation\n");
    return 1;
  }

  if (g_zpbsv_existing_calls != 1 || g_zpbsvx_bridge_calls != 0) {
    fprintf(stderr,
            "[FAIL] ZPBSV existing-preserve mismatch (existing=%d adapter=%d)\n",
            g_zpbsv_existing_calls, g_zpbsvx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter does not override existing ZPBSV implementation\n");
  return 0;
}

static int test_adapter_does_not_override_existing_sppsv(void) {
  fb_backend_vtable_t vtable;
  fb_sppsv_cblas_fn sppsv_call = NULL;
  fb_status_t status;
  float ap[3] = {0};
  float b[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_sppsv_existing_calls = 0;
  g_sppsvx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_SPPSV][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sppsv_existing;
  vtable.ext_ops[FB_OP_SPPSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sppsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  sppsv_call = (fb_sppsv_cblas_fn)vtable.ext_ops[FB_OP_SPPSV][FB_CONV_CBLAS];
  if (!sppsv_call) {
    fprintf(stderr, "[FAIL] SPPSV slot missing after finalize\n");
    return 1;
  }

  if (sppsv_call(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, ap, b, 2) !=
      1215) {
    fprintf(stderr,
            "[FAIL] adapter unexpectedly replaced existing SPPSV implementation\n");
    return 1;
  }

  if (g_sppsv_existing_calls != 1 || g_sppsvx_bridge_calls != 0) {
    fprintf(stderr,
            "[FAIL] SPPSV existing-preserve mismatch (existing=%d adapter=%d)\n",
            g_sppsv_existing_calls, g_sppsvx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter does not override existing SPPSV implementation\n");
  return 0;
}

static int test_adapter_does_not_override_existing_dppsv(void) {
  fb_backend_vtable_t vtable;
  fb_dppsv_cblas_fn dppsv_call = NULL;
  fb_status_t status;
  double ap[3] = {0};
  double b[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_dppsv_existing_calls = 0;
  g_dppsvx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_DPPSV][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dppsv_existing;
  vtable.ext_ops[FB_OP_DPPSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dppsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  dppsv_call = (fb_dppsv_cblas_fn)vtable.ext_ops[FB_OP_DPPSV][FB_CONV_CBLAS];
  if (!dppsv_call) {
    fprintf(stderr, "[FAIL] DPPSV slot missing after finalize\n");
    return 1;
  }

  if (dppsv_call(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, ap, b, 2) !=
      1225) {
    fprintf(stderr,
            "[FAIL] adapter unexpectedly replaced existing DPPSV implementation\n");
    return 1;
  }

  if (g_dppsv_existing_calls != 1 || g_dppsvx_bridge_calls != 0) {
    fprintf(stderr,
            "[FAIL] DPPSV existing-preserve mismatch (existing=%d adapter=%d)\n",
            g_dppsv_existing_calls, g_dppsvx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter does not override existing DPPSV implementation\n");
  return 0;
}

static int test_adapter_does_not_override_existing_cppsv(void) {
  fb_backend_vtable_t vtable;
  fb_cppsv_cblas_fn cppsv_call = NULL;
  fb_status_t status;
  fb_complex_float_t ap[3] = {0};
  fb_complex_float_t b[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_cppsv_existing_calls = 0;
  g_cppsvx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_CPPSV][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_cppsv_existing;
  vtable.ext_ops[FB_OP_CPPSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_cppsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  cppsv_call = (fb_cppsv_cblas_fn)vtable.ext_ops[FB_OP_CPPSV][FB_CONV_CBLAS];
  if (!cppsv_call) {
    fprintf(stderr, "[FAIL] CPPSV slot missing after finalize\n");
    return 1;
  }

  if (cppsv_call(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, ap, b, 2) !=
      1235) {
    fprintf(stderr,
            "[FAIL] adapter unexpectedly replaced existing CPPSV implementation\n");
    return 1;
  }

  if (g_cppsv_existing_calls != 1 || g_cppsvx_bridge_calls != 0) {
    fprintf(stderr,
            "[FAIL] CPPSV existing-preserve mismatch (existing=%d adapter=%d)\n",
            g_cppsv_existing_calls, g_cppsvx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter does not override existing CPPSV implementation\n");
  return 0;
}

static int test_adapter_does_not_override_existing_zppsv(void) {
  fb_backend_vtable_t vtable;
  fb_zppsv_cblas_fn zppsv_call = NULL;
  fb_status_t status;
  fb_complex_double_t ap[3] = {0};
  fb_complex_double_t b[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_zppsv_existing_calls = 0;
  g_zppsvx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_ZPPSV][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zppsv_existing;
  vtable.ext_ops[FB_OP_ZPPSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zppsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  zppsv_call = (fb_zppsv_cblas_fn)vtable.ext_ops[FB_OP_ZPPSV][FB_CONV_CBLAS];
  if (!zppsv_call) {
    fprintf(stderr, "[FAIL] ZPPSV slot missing after finalize\n");
    return 1;
  }

  if (zppsv_call(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, ap, b, 2) !=
      1245) {
    fprintf(stderr,
            "[FAIL] adapter unexpectedly replaced existing ZPPSV implementation\n");
    return 1;
  }

  if (g_zppsv_existing_calls != 1 || g_zppsvx_bridge_calls != 0) {
    fprintf(stderr,
            "[FAIL] ZPPSV existing-preserve mismatch (existing=%d adapter=%d)\n",
            g_zppsv_existing_calls, g_zppsvx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter does not override existing ZPPSV implementation\n");
  return 0;
}

static int test_adapter_does_not_override_existing_sspsv(void) {
  fb_backend_vtable_t vtable;
  fb_sspsv_cblas_fn sspsv_call = NULL;
  fb_status_t status;
  float ap[3] = {0};
  float b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_sspsv_existing_calls = 0;
  g_sspsvx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_SSPSV][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sspsv_existing;
  vtable.ext_ops[FB_OP_SSPSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sspsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  sspsv_call = (fb_sspsv_cblas_fn)vtable.ext_ops[FB_OP_SSPSV][FB_CONV_CBLAS];
  if (!sspsv_call) {
    fprintf(stderr, "[FAIL] SSPSV slot missing after finalize\n");
    return 1;
  }

  if (sspsv_call(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, ap, ipiv, b, 2) !=
      1255) {
    fprintf(stderr,
            "[FAIL] adapter unexpectedly replaced existing SSPSV implementation\n");
    return 1;
  }

  if (g_sspsv_existing_calls != 1 || g_sspsvx_bridge_calls != 0) {
    fprintf(stderr,
            "[FAIL] SSPSV existing-preserve mismatch (existing=%d adapter=%d)\n",
            g_sspsv_existing_calls, g_sspsvx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter does not override existing SSPSV implementation\n");
  return 0;
}

static int test_adapter_does_not_override_existing_dspsv(void) {
  fb_backend_vtable_t vtable;
  fb_dspsv_cblas_fn dspsv_call = NULL;
  fb_status_t status;
  double ap[3] = {0};
  double b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_dspsv_existing_calls = 0;
  g_dspsvx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_DSPSV][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dspsv_existing;
  vtable.ext_ops[FB_OP_DSPSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dspsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  dspsv_call = (fb_dspsv_cblas_fn)vtable.ext_ops[FB_OP_DSPSV][FB_CONV_CBLAS];
  if (!dspsv_call) {
    fprintf(stderr, "[FAIL] DSPSV slot missing after finalize\n");
    return 1;
  }

  if (dspsv_call(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, ap, ipiv, b, 2) !=
      1265) {
    fprintf(stderr,
            "[FAIL] adapter unexpectedly replaced existing DSPSV implementation\n");
    return 1;
  }

  if (g_dspsv_existing_calls != 1 || g_dspsvx_bridge_calls != 0) {
    fprintf(stderr,
            "[FAIL] DSPSV existing-preserve mismatch (existing=%d adapter=%d)\n",
            g_dspsv_existing_calls, g_dspsvx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter does not override existing DSPSV implementation\n");
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

static int test_adapter_finalize_idempotent_sgesv_from_sgesvxx(void) {
  fb_backend_vtable_t vtable;
  fb_sgesv_fn sgesv_call = NULL;
  fb_generic_fn first_ptr = NULL;
  fb_status_t status;
  float a[4] = {0};
  float b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  vtable.ext_ops[FB_OP_SGESVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sgesvxx_base;

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

  printf("[PASS] adapter finalize is idempotent for SGESV <- SGESVXX bridge\n");
  return 0;
}

static int test_adapter_finalize_idempotent_dgesv_from_dgesvxx(void) {
  fb_backend_vtable_t vtable;
  fb_dgesv_fn dgesv_call = NULL;
  fb_generic_fn first_ptr = NULL;
  fb_status_t status;
  double a[4] = {0};
  double b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  vtable.ext_ops[FB_OP_DGESVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dgesvxx_base;

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

  printf("[PASS] adapter finalize is idempotent for DGESV <- DGESVXX bridge\n");
  return 0;
}

static int test_adapter_finalize_idempotent_cgesv_from_cgesvxx(void) {
  fb_backend_vtable_t vtable;
  fb_cgesv_fn cgesv_call = NULL;
  fb_generic_fn first_ptr = NULL;
  fb_status_t status;
  fb_complex_float_t a[4] = {0};
  fb_complex_float_t b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  vtable.ext_ops[FB_OP_CGESVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_cgesvxx_base;

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

  printf("[PASS] adapter finalize is idempotent for CGESV <- CGESVXX bridge\n");
  return 0;
}

static int test_adapter_finalize_idempotent_zgesv_from_zgesvxx(void) {
  fb_backend_vtable_t vtable;
  fb_zgesv_fn zgesv_call = NULL;
  fb_generic_fn first_ptr = NULL;
  fb_status_t status;
  fb_complex_double_t a[4] = {0};
  fb_complex_double_t b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  vtable.ext_ops[FB_OP_ZGESVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zgesvxx_base;

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

  printf("[PASS] adapter finalize is idempotent for ZGESV <- ZGESVXX bridge\n");
  return 0;
}

static int test_adapter_finalize_idempotent_sposv_from_sposvxx(void) {
  fb_backend_vtable_t vtable;
  fb_sposv_fn sposv_call = NULL;
  fb_generic_fn first_ptr = NULL;
  fb_status_t status;
  float a[4] = {0};
  float b[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  vtable.ext_ops[FB_OP_SPOSVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sposvxx_base;

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

  printf("[PASS] adapter finalize is idempotent for SPOSV <- SPOSVXX bridge\n");
  return 0;
}

static int test_adapter_finalize_idempotent_dposv_from_dposvxx(void) {
  fb_backend_vtable_t vtable;
  fb_dposv_fn dposv_call = NULL;
  fb_generic_fn first_ptr = NULL;
  fb_status_t status;
  double a[4] = {0};
  double b[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  vtable.ext_ops[FB_OP_DPOSVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dposvxx_base;

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

  printf("[PASS] adapter finalize is idempotent for DPOSV <- DPOSVXX bridge\n");
  return 0;
}

static int test_adapter_finalize_idempotent_cposv_from_cposvxx(void) {
  fb_backend_vtable_t vtable;
  fb_cposv_fn cposv_call = NULL;
  fb_generic_fn first_ptr = NULL;
  fb_status_t status;
  fb_complex_float_t a[4] = {0};
  fb_complex_float_t b[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  vtable.ext_ops[FB_OP_CPOSVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_cposvxx_base;

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

  printf("[PASS] adapter finalize is idempotent for CPOSV <- CPOSVXX bridge\n");
  return 0;
}

static int test_adapter_finalize_idempotent_zposv_from_zposvxx(void) {
  fb_backend_vtable_t vtable;
  fb_zposv_fn zposv_call = NULL;
  fb_generic_fn first_ptr = NULL;
  fb_status_t status;
  fb_complex_double_t a[4] = {0};
  fb_complex_double_t b[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  vtable.ext_ops[FB_OP_ZPOSVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zposvxx_base;

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

  printf("[PASS] adapter finalize is idempotent for ZPOSV <- ZPOSVXX bridge\n");
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

static int test_adapter_finalize_idempotent_chesv_from_chesvx(void) {
  fb_backend_vtable_t vtable;
  fb_csysv_fn chesv_call = NULL;
  fb_generic_fn first_ptr = NULL;
  fb_status_t status;
  fb_complex_float_t a[4] = {0};
  fb_complex_float_t b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  vtable.ext_ops[FB_OP_CHESVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_csysvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] first finalize status=%d\n", status);
    return 1;
  }

  first_ptr = vtable.ext_ops[FB_OP_CHESV][FB_CONV_CBLAS];
  if (!first_ptr) {
    fprintf(stderr, "[FAIL] CHESV slot missing after first finalize\n");
    return 1;
  }

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] second finalize status=%d\n", status);
    return 1;
  }

  if (vtable.ext_ops[FB_OP_CHESV][FB_CONV_CBLAS] != first_ptr) {
    fprintf(stderr,
            "[FAIL] CHESV adapter pointer changed after second finalize\n");
    return 1;
  }

  chesv_call = (fb_csysv_fn)vtable.ext_ops[FB_OP_CHESV][FB_CONV_CBLAS];
  if (!chesv_call) {
    fprintf(stderr, "[FAIL] CHESV call pointer missing after second finalize\n");
    return 1;
  }

  if (chesv_call(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, a, 2, ipiv, b,
                 2) != 0) {
    fprintf(stderr, "[FAIL] CHESV call failed after repeated finalize\n");
    return 1;
  }

  printf("[PASS] adapter finalize is idempotent for CHESV <- CHESVX bridge\n");
  return 0;
}

static int test_adapter_finalize_idempotent_zhesv_from_zhesvx(void) {
  fb_backend_vtable_t vtable;
  fb_zsysv_fn zhesv_call = NULL;
  fb_generic_fn first_ptr = NULL;
  fb_status_t status;
  fb_complex_double_t a[4] = {0};
  fb_complex_double_t b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  vtable.ext_ops[FB_OP_ZHESVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zsysvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] first finalize status=%d\n", status);
    return 1;
  }

  first_ptr = vtable.ext_ops[FB_OP_ZHESV][FB_CONV_CBLAS];
  if (!first_ptr) {
    fprintf(stderr, "[FAIL] ZHESV slot missing after first finalize\n");
    return 1;
  }

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] second finalize status=%d\n", status);
    return 1;
  }

  if (vtable.ext_ops[FB_OP_ZHESV][FB_CONV_CBLAS] != first_ptr) {
    fprintf(stderr,
            "[FAIL] ZHESV adapter pointer changed after second finalize\n");
    return 1;
  }

  zhesv_call = (fb_zsysv_fn)vtable.ext_ops[FB_OP_ZHESV][FB_CONV_CBLAS];
  if (!zhesv_call) {
    fprintf(stderr, "[FAIL] ZHESV call pointer missing after second finalize\n");
    return 1;
  }

  if (zhesv_call(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, a, 2, ipiv, b,
                 2) != 0) {
    fprintf(stderr, "[FAIL] ZHESV call failed after repeated finalize\n");
    return 1;
  }

  printf("[PASS] adapter finalize is idempotent for ZHESV <- ZHESVX bridge\n");
  return 0;
}

static int test_adapter_finalize_idempotent_ssysv_from_ssysvxx(void) {
  fb_backend_vtable_t vtable;
  fb_ssysv_fn ssysv_call = NULL;
  fb_generic_fn first_ptr = NULL;
  fb_status_t status;
  float a[4] = {0};
  float b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  vtable.ext_ops[FB_OP_SSYSVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_ssysvxx_base;

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

  printf("[PASS] adapter finalize is idempotent for SSYSV <- SSYSVXX bridge\n");
  return 0;
}

static int test_adapter_finalize_idempotent_dsysv_from_dsysvxx(void) {
  fb_backend_vtable_t vtable;
  fb_dsysv_fn dsysv_call = NULL;
  fb_generic_fn first_ptr = NULL;
  fb_status_t status;
  double a[4] = {0};
  double b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  vtable.ext_ops[FB_OP_DSYSVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dsysvxx_base;

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

  printf("[PASS] adapter finalize is idempotent for DSYSV <- DSYSVXX bridge\n");
  return 0;
}

static int test_adapter_finalize_idempotent_csysv_from_csysvxx(void) {
  fb_backend_vtable_t vtable;
  fb_csysv_fn csysv_call = NULL;
  fb_generic_fn first_ptr = NULL;
  fb_status_t status;
  fb_complex_float_t a[4] = {0};
  fb_complex_float_t b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  vtable.ext_ops[FB_OP_CSYSVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_csysvxx_base;

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

  printf("[PASS] adapter finalize is idempotent for CSYSV <- CSYSVXX bridge\n");
  return 0;
}

static int test_adapter_finalize_idempotent_zsysv_from_zsysvxx(void) {
  fb_backend_vtable_t vtable;
  fb_zsysv_fn zsysv_call = NULL;
  fb_generic_fn first_ptr = NULL;
  fb_status_t status;
  fb_complex_double_t a[4] = {0};
  fb_complex_double_t b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  vtable.ext_ops[FB_OP_ZSYSVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zsysvxx_base;

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

  printf("[PASS] adapter finalize is idempotent for ZSYSV <- ZSYSVXX bridge\n");
  return 0;
}

static int test_adapter_finalize_idempotent_chesv_from_chesvxx(void) {
  fb_backend_vtable_t vtable;
  fb_csysv_fn chesv_call = NULL;
  fb_generic_fn first_ptr = NULL;
  fb_status_t status;
  fb_complex_float_t a[4] = {0};
  fb_complex_float_t b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  vtable.ext_ops[FB_OP_CHESVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_csysvxx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] first finalize status=%d\n", status);
    return 1;
  }

  first_ptr = vtable.ext_ops[FB_OP_CHESV][FB_CONV_CBLAS];
  if (!first_ptr) {
    fprintf(stderr, "[FAIL] CHESV slot missing after first finalize\n");
    return 1;
  }

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] second finalize status=%d\n", status);
    return 1;
  }

  if (vtable.ext_ops[FB_OP_CHESV][FB_CONV_CBLAS] != first_ptr) {
    fprintf(stderr,
            "[FAIL] CHESV adapter pointer changed after second finalize\n");
    return 1;
  }

  chesv_call = (fb_csysv_fn)vtable.ext_ops[FB_OP_CHESV][FB_CONV_CBLAS];
  if (!chesv_call) {
    fprintf(stderr, "[FAIL] CHESV call pointer missing after second finalize\n");
    return 1;
  }

  if (chesv_call(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, a, 2, ipiv, b,
                 2) != 0) {
    fprintf(stderr, "[FAIL] CHESV call failed after repeated finalize\n");
    return 1;
  }

  printf("[PASS] adapter finalize is idempotent for CHESV <- CHESVXX bridge\n");
  return 0;
}

static int test_adapter_finalize_idempotent_zhesv_from_zhesvxx(void) {
  fb_backend_vtable_t vtable;
  fb_zsysv_fn zhesv_call = NULL;
  fb_generic_fn first_ptr = NULL;
  fb_status_t status;
  fb_complex_double_t a[4] = {0};
  fb_complex_double_t b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  vtable.ext_ops[FB_OP_ZHESVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zsysvxx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] first finalize status=%d\n", status);
    return 1;
  }

  first_ptr = vtable.ext_ops[FB_OP_ZHESV][FB_CONV_CBLAS];
  if (!first_ptr) {
    fprintf(stderr, "[FAIL] ZHESV slot missing after first finalize\n");
    return 1;
  }

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] second finalize status=%d\n", status);
    return 1;
  }

  if (vtable.ext_ops[FB_OP_ZHESV][FB_CONV_CBLAS] != first_ptr) {
    fprintf(stderr,
            "[FAIL] ZHESV adapter pointer changed after second finalize\n");
    return 1;
  }

  zhesv_call = (fb_zsysv_fn)vtable.ext_ops[FB_OP_ZHESV][FB_CONV_CBLAS];
  if (!zhesv_call) {
    fprintf(stderr, "[FAIL] ZHESV call pointer missing after second finalize\n");
    return 1;
  }

  if (zhesv_call(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, a, 2, ipiv, b,
                 2) != 0) {
    fprintf(stderr, "[FAIL] ZHESV call failed after repeated finalize\n");
    return 1;
  }

  printf("[PASS] adapter finalize is idempotent for ZHESV <- ZHESVXX bridge\n");
  return 0;
}

static int test_adapter_finalize_idempotent_sgbsv(void) {
  fb_backend_vtable_t vtable;
  fb_sgbsv_cblas_fn sgbsv_call = NULL;
  fb_generic_fn first_ptr = NULL;
  fb_status_t status;

  float ab[8] = {0};
  float b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));

  vtable.ext_ops[FB_OP_SGBSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sgbsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] first finalize status=%d\n", status);
    return 1;
  }

  first_ptr = vtable.ext_ops[FB_OP_SGBSV][FB_CONV_CBLAS];
  if (!first_ptr) {
    fprintf(stderr, "[FAIL] SGBSV slot missing after first finalize\n");
    return 1;
  }

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] second finalize status=%d\n", status);
    return 1;
  }

  if (vtable.ext_ops[FB_OP_SGBSV][FB_CONV_CBLAS] != first_ptr) {
    fprintf(stderr,
            "[FAIL] SGBSV adapter pointer changed after second finalize\n");
    return 1;
  }

  sgbsv_call = (fb_sgbsv_cblas_fn)vtable.ext_ops[FB_OP_SGBSV][FB_CONV_CBLAS];
  if (!sgbsv_call) {
    fprintf(stderr, "[FAIL] SGBSV call pointer missing after second finalize\n");
    return 1;
  }

  if (sgbsv_call(FB_LAYOUT_COL_MAJOR, 2, 1, 1, 1, ab, 4, ipiv, b, 2) != 0) {
    fprintf(stderr, "[FAIL] SGBSV call failed after repeated finalize\n");
    return 1;
  }

  printf("[PASS] adapter finalize is idempotent for SGBSV bridge\n");
  return 0;
}

static int test_adapter_finalize_idempotent_dgbsv(void) {
  fb_backend_vtable_t vtable;
  fb_dgbsv_cblas_fn dgbsv_call = NULL;
  fb_generic_fn first_ptr = NULL;
  fb_status_t status;

  double ab[8] = {0};
  double b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));

  vtable.ext_ops[FB_OP_DGBSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dgbsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] first finalize status=%d\n", status);
    return 1;
  }

  first_ptr = vtable.ext_ops[FB_OP_DGBSV][FB_CONV_CBLAS];
  if (!first_ptr) {
    fprintf(stderr, "[FAIL] DGBSV slot missing after first finalize\n");
    return 1;
  }

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] second finalize status=%d\n", status);
    return 1;
  }

  if (vtable.ext_ops[FB_OP_DGBSV][FB_CONV_CBLAS] != first_ptr) {
    fprintf(stderr,
            "[FAIL] DGBSV adapter pointer changed after second finalize\n");
    return 1;
  }

  dgbsv_call = (fb_dgbsv_cblas_fn)vtable.ext_ops[FB_OP_DGBSV][FB_CONV_CBLAS];
  if (!dgbsv_call) {
    fprintf(stderr, "[FAIL] DGBSV call pointer missing after second finalize\n");
    return 1;
  }

  if (dgbsv_call(FB_LAYOUT_COL_MAJOR, 2, 1, 1, 1, ab, 4, ipiv, b, 2) != 0) {
    fprintf(stderr, "[FAIL] DGBSV call failed after repeated finalize\n");
    return 1;
  }

  printf("[PASS] adapter finalize is idempotent for DGBSV bridge\n");
  return 0;
}

static int test_adapter_finalize_idempotent_cgbsv(void) {
  fb_backend_vtable_t vtable;
  fb_cgbsv_cblas_fn cgbsv_call = NULL;
  fb_generic_fn first_ptr = NULL;
  fb_status_t status;

  fb_complex_float_t ab[8] = {0};
  fb_complex_float_t b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));

  vtable.ext_ops[FB_OP_CGBSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_cgbsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] first finalize status=%d\n", status);
    return 1;
  }

  first_ptr = vtable.ext_ops[FB_OP_CGBSV][FB_CONV_CBLAS];
  if (!first_ptr) {
    fprintf(stderr, "[FAIL] CGBSV slot missing after first finalize\n");
    return 1;
  }

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] second finalize status=%d\n", status);
    return 1;
  }

  if (vtable.ext_ops[FB_OP_CGBSV][FB_CONV_CBLAS] != first_ptr) {
    fprintf(stderr,
            "[FAIL] CGBSV adapter pointer changed after second finalize\n");
    return 1;
  }

  cgbsv_call = (fb_cgbsv_cblas_fn)vtable.ext_ops[FB_OP_CGBSV][FB_CONV_CBLAS];
  if (!cgbsv_call) {
    fprintf(stderr, "[FAIL] CGBSV call pointer missing after second finalize\n");
    return 1;
  }

  if (cgbsv_call(FB_LAYOUT_COL_MAJOR, 2, 1, 1, 1, ab, 4, ipiv, b, 2) != 0) {
    fprintf(stderr, "[FAIL] CGBSV call failed after repeated finalize\n");
    return 1;
  }

  printf("[PASS] adapter finalize is idempotent for CGBSV bridge\n");
  return 0;
}

static int test_adapter_finalize_idempotent_zgbsv(void) {
  fb_backend_vtable_t vtable;
  fb_zgbsv_cblas_fn zgbsv_call = NULL;
  fb_generic_fn first_ptr = NULL;
  fb_status_t status;

  fb_complex_double_t ab[8] = {0};
  fb_complex_double_t b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));

  vtable.ext_ops[FB_OP_ZGBSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zgbsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] first finalize status=%d\n", status);
    return 1;
  }

  first_ptr = vtable.ext_ops[FB_OP_ZGBSV][FB_CONV_CBLAS];
  if (!first_ptr) {
    fprintf(stderr, "[FAIL] ZGBSV slot missing after first finalize\n");
    return 1;
  }

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] second finalize status=%d\n", status);
    return 1;
  }

  if (vtable.ext_ops[FB_OP_ZGBSV][FB_CONV_CBLAS] != first_ptr) {
    fprintf(stderr,
            "[FAIL] ZGBSV adapter pointer changed after second finalize\n");
    return 1;
  }

  zgbsv_call = (fb_zgbsv_cblas_fn)vtable.ext_ops[FB_OP_ZGBSV][FB_CONV_CBLAS];
  if (!zgbsv_call) {
    fprintf(stderr, "[FAIL] ZGBSV call pointer missing after second finalize\n");
    return 1;
  }

  if (zgbsv_call(FB_LAYOUT_COL_MAJOR, 2, 1, 1, 1, ab, 4, ipiv, b, 2) != 0) {
    fprintf(stderr, "[FAIL] ZGBSV call failed after repeated finalize\n");
    return 1;
  }

  printf("[PASS] adapter finalize is idempotent for ZGBSV bridge\n");
  return 0;
}

static int test_adapter_finalize_idempotent_sgtsv(void) {
  fb_backend_vtable_t vtable;
  fb_sgtsv_cblas_fn sgtsv_call = NULL;
  fb_generic_fn first_ptr = NULL;
  fb_status_t status;

  float dl[2] = {0};
  float d[2] = {0};
  float du[2] = {0};
  float b[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  vtable.ext_ops[FB_OP_SGTSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sgtsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] first finalize status=%d\n", status);
    return 1;
  }

  first_ptr = vtable.ext_ops[FB_OP_SGTSV][FB_CONV_CBLAS];
  if (!first_ptr) {
    fprintf(stderr, "[FAIL] SGTSV slot missing after first finalize\n");
    return 1;
  }

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] second finalize status=%d\n", status);
    return 1;
  }

  if (vtable.ext_ops[FB_OP_SGTSV][FB_CONV_CBLAS] != first_ptr) {
    fprintf(stderr,
            "[FAIL] SGTSV adapter pointer changed after second finalize\n");
    return 1;
  }

  sgtsv_call = (fb_sgtsv_cblas_fn)vtable.ext_ops[FB_OP_SGTSV][FB_CONV_CBLAS];
  if (!sgtsv_call) {
    fprintf(stderr, "[FAIL] SGTSV call pointer missing after second finalize\n");
    return 1;
  }

  if (sgtsv_call(FB_LAYOUT_COL_MAJOR, 2, 1, dl, d, du, b, 2) != 0) {
    fprintf(stderr, "[FAIL] SGTSV call failed after repeated finalize\n");
    return 1;
  }

  printf("[PASS] adapter finalize is idempotent for SGTSV bridge\n");
  return 0;
}

static int test_adapter_finalize_idempotent_dgtsv(void) {
  fb_backend_vtable_t vtable;
  fb_dgtsv_cblas_fn dgtsv_call = NULL;
  fb_generic_fn first_ptr = NULL;
  fb_status_t status;

  double dl[2] = {0};
  double d[2] = {0};
  double du[2] = {0};
  double b[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  vtable.ext_ops[FB_OP_DGTSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dgtsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] first finalize status=%d\n", status);
    return 1;
  }

  first_ptr = vtable.ext_ops[FB_OP_DGTSV][FB_CONV_CBLAS];
  if (!first_ptr) {
    fprintf(stderr, "[FAIL] DGTSV slot missing after first finalize\n");
    return 1;
  }

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] second finalize status=%d\n", status);
    return 1;
  }

  if (vtable.ext_ops[FB_OP_DGTSV][FB_CONV_CBLAS] != first_ptr) {
    fprintf(stderr,
            "[FAIL] DGTSV adapter pointer changed after second finalize\n");
    return 1;
  }

  dgtsv_call = (fb_dgtsv_cblas_fn)vtable.ext_ops[FB_OP_DGTSV][FB_CONV_CBLAS];
  if (!dgtsv_call) {
    fprintf(stderr, "[FAIL] DGTSV call pointer missing after second finalize\n");
    return 1;
  }

  if (dgtsv_call(FB_LAYOUT_COL_MAJOR, 2, 1, dl, d, du, b, 2) != 0) {
    fprintf(stderr, "[FAIL] DGTSV call failed after repeated finalize\n");
    return 1;
  }

  printf("[PASS] adapter finalize is idempotent for DGTSV bridge\n");
  return 0;
}

static int test_adapter_finalize_idempotent_cgtsv(void) {
  fb_backend_vtable_t vtable;
  fb_cgtsv_cblas_fn cgtsv_call = NULL;
  fb_generic_fn first_ptr = NULL;
  fb_status_t status;

  fb_complex_float_t dl[2] = {0};
  fb_complex_float_t d[2] = {0};
  fb_complex_float_t du[2] = {0};
  fb_complex_float_t b[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  vtable.ext_ops[FB_OP_CGTSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_cgtsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] first finalize status=%d\n", status);
    return 1;
  }

  first_ptr = vtable.ext_ops[FB_OP_CGTSV][FB_CONV_CBLAS];
  if (!first_ptr) {
    fprintf(stderr, "[FAIL] CGTSV slot missing after first finalize\n");
    return 1;
  }

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] second finalize status=%d\n", status);
    return 1;
  }

  if (vtable.ext_ops[FB_OP_CGTSV][FB_CONV_CBLAS] != first_ptr) {
    fprintf(stderr,
            "[FAIL] CGTSV adapter pointer changed after second finalize\n");
    return 1;
  }

  cgtsv_call = (fb_cgtsv_cblas_fn)vtable.ext_ops[FB_OP_CGTSV][FB_CONV_CBLAS];
  if (!cgtsv_call) {
    fprintf(stderr, "[FAIL] CGTSV call pointer missing after second finalize\n");
    return 1;
  }

  if (cgtsv_call(FB_LAYOUT_COL_MAJOR, 2, 1, dl, d, du, b, 2) != 0) {
    fprintf(stderr, "[FAIL] CGTSV call failed after repeated finalize\n");
    return 1;
  }

  printf("[PASS] adapter finalize is idempotent for CGTSV bridge\n");
  return 0;
}

static int test_adapter_finalize_idempotent_zgtsv(void) {
  fb_backend_vtable_t vtable;
  fb_zgtsv_cblas_fn zgtsv_call = NULL;
  fb_generic_fn first_ptr = NULL;
  fb_status_t status;

  fb_complex_double_t dl[2] = {0};
  fb_complex_double_t d[2] = {0};
  fb_complex_double_t du[2] = {0};
  fb_complex_double_t b[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  vtable.ext_ops[FB_OP_ZGTSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zgtsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] first finalize status=%d\n", status);
    return 1;
  }

  first_ptr = vtable.ext_ops[FB_OP_ZGTSV][FB_CONV_CBLAS];
  if (!first_ptr) {
    fprintf(stderr, "[FAIL] ZGTSV slot missing after first finalize\n");
    return 1;
  }

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] second finalize status=%d\n", status);
    return 1;
  }

  if (vtable.ext_ops[FB_OP_ZGTSV][FB_CONV_CBLAS] != first_ptr) {
    fprintf(stderr,
            "[FAIL] ZGTSV adapter pointer changed after second finalize\n");
    return 1;
  }

  zgtsv_call = (fb_zgtsv_cblas_fn)vtable.ext_ops[FB_OP_ZGTSV][FB_CONV_CBLAS];
  if (!zgtsv_call) {
    fprintf(stderr, "[FAIL] ZGTSV call pointer missing after second finalize\n");
    return 1;
  }

  if (zgtsv_call(FB_LAYOUT_COL_MAJOR, 2, 1, dl, d, du, b, 2) != 0) {
    fprintf(stderr, "[FAIL] ZGTSV call failed after repeated finalize\n");
    return 1;
  }

  printf("[PASS] adapter finalize is idempotent for ZGTSV bridge\n");
  return 0;
}

static int test_adapter_finalize_idempotent_sptsv(void) {
  fb_backend_vtable_t vtable;
  fb_sptsv_cblas_fn sptsv_call = NULL;
  fb_generic_fn first_ptr = NULL;
  fb_status_t status;

  float d[2] = {0};
  float e[2] = {0};
  float b[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  vtable.ext_ops[FB_OP_SPTSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sptsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] first finalize status=%d\n", status);
    return 1;
  }

  first_ptr = vtable.ext_ops[FB_OP_SPTSV][FB_CONV_CBLAS];
  if (!first_ptr) {
    fprintf(stderr, "[FAIL] SPTSV slot missing after first finalize\n");
    return 1;
  }

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] second finalize status=%d\n", status);
    return 1;
  }

  if (vtable.ext_ops[FB_OP_SPTSV][FB_CONV_CBLAS] != first_ptr) {
    fprintf(stderr,
            "[FAIL] SPTSV adapter pointer changed after second finalize\n");
    return 1;
  }

  sptsv_call = (fb_sptsv_cblas_fn)vtable.ext_ops[FB_OP_SPTSV][FB_CONV_CBLAS];
  if (!sptsv_call) {
    fprintf(stderr, "[FAIL] SPTSV call pointer missing after second finalize\n");
    return 1;
  }

  if (sptsv_call(FB_LAYOUT_COL_MAJOR, 2, 1, d, e, b, 2) != 0) {
    fprintf(stderr, "[FAIL] SPTSV call failed after repeated finalize\n");
    return 1;
  }

  printf("[PASS] adapter finalize is idempotent for SPTSV bridge\n");
  return 0;
}

static int test_adapter_finalize_idempotent_dptsv(void) {
  fb_backend_vtable_t vtable;
  fb_dptsv_cblas_fn dptsv_call = NULL;
  fb_generic_fn first_ptr = NULL;
  fb_status_t status;

  double d[2] = {0};
  double e[2] = {0};
  double b[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  vtable.ext_ops[FB_OP_DPTSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dptsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] first finalize status=%d\n", status);
    return 1;
  }

  first_ptr = vtable.ext_ops[FB_OP_DPTSV][FB_CONV_CBLAS];
  if (!first_ptr) {
    fprintf(stderr, "[FAIL] DPTSV slot missing after first finalize\n");
    return 1;
  }

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] second finalize status=%d\n", status);
    return 1;
  }

  if (vtable.ext_ops[FB_OP_DPTSV][FB_CONV_CBLAS] != first_ptr) {
    fprintf(stderr,
            "[FAIL] DPTSV adapter pointer changed after second finalize\n");
    return 1;
  }

  dptsv_call = (fb_dptsv_cblas_fn)vtable.ext_ops[FB_OP_DPTSV][FB_CONV_CBLAS];
  if (!dptsv_call) {
    fprintf(stderr, "[FAIL] DPTSV call pointer missing after second finalize\n");
    return 1;
  }

  if (dptsv_call(FB_LAYOUT_COL_MAJOR, 2, 1, d, e, b, 2) != 0) {
    fprintf(stderr, "[FAIL] DPTSV call failed after repeated finalize\n");
    return 1;
  }

  printf("[PASS] adapter finalize is idempotent for DPTSV bridge\n");
  return 0;
}

static int test_adapter_finalize_idempotent_cptsv(void) {
  fb_backend_vtable_t vtable;
  fb_cptsv_cblas_fn cptsv_call = NULL;
  fb_generic_fn first_ptr = NULL;
  fb_status_t status;

  float d[2] = {0};
  fb_complex_float_t e[2] = {0};
  fb_complex_float_t b[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  vtable.ext_ops[FB_OP_CPTSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_cptsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] first finalize status=%d\n", status);
    return 1;
  }

  first_ptr = vtable.ext_ops[FB_OP_CPTSV][FB_CONV_CBLAS];
  if (!first_ptr) {
    fprintf(stderr, "[FAIL] CPTSV slot missing after first finalize\n");
    return 1;
  }

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] second finalize status=%d\n", status);
    return 1;
  }

  if (vtable.ext_ops[FB_OP_CPTSV][FB_CONV_CBLAS] != first_ptr) {
    fprintf(stderr,
            "[FAIL] CPTSV adapter pointer changed after second finalize\n");
    return 1;
  }

  cptsv_call = (fb_cptsv_cblas_fn)vtable.ext_ops[FB_OP_CPTSV][FB_CONV_CBLAS];
  if (!cptsv_call) {
    fprintf(stderr, "[FAIL] CPTSV call pointer missing after second finalize\n");
    return 1;
  }

  if (cptsv_call(FB_LAYOUT_COL_MAJOR, 2, 1, d, e, b, 2) != 0) {
    fprintf(stderr, "[FAIL] CPTSV call failed after repeated finalize\n");
    return 1;
  }

  printf("[PASS] adapter finalize is idempotent for CPTSV bridge\n");
  return 0;
}

static int test_adapter_finalize_idempotent_zptsv(void) {
  fb_backend_vtable_t vtable;
  fb_zptsv_cblas_fn zptsv_call = NULL;
  fb_generic_fn first_ptr = NULL;
  fb_status_t status;

  double d[2] = {0};
  fb_complex_double_t e[2] = {0};
  fb_complex_double_t b[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  vtable.ext_ops[FB_OP_ZPTSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zptsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] first finalize status=%d\n", status);
    return 1;
  }

  first_ptr = vtable.ext_ops[FB_OP_ZPTSV][FB_CONV_CBLAS];
  if (!first_ptr) {
    fprintf(stderr, "[FAIL] ZPTSV slot missing after first finalize\n");
    return 1;
  }

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] second finalize status=%d\n", status);
    return 1;
  }

  if (vtable.ext_ops[FB_OP_ZPTSV][FB_CONV_CBLAS] != first_ptr) {
    fprintf(stderr,
            "[FAIL] ZPTSV adapter pointer changed after second finalize\n");
    return 1;
  }

  zptsv_call = (fb_zptsv_cblas_fn)vtable.ext_ops[FB_OP_ZPTSV][FB_CONV_CBLAS];
  if (!zptsv_call) {
    fprintf(stderr, "[FAIL] ZPTSV call pointer missing after second finalize\n");
    return 1;
  }

  if (zptsv_call(FB_LAYOUT_COL_MAJOR, 2, 1, d, e, b, 2) != 0) {
    fprintf(stderr, "[FAIL] ZPTSV call failed after repeated finalize\n");
    return 1;
  }

  printf("[PASS] adapter finalize is idempotent for ZPTSV bridge\n");
  return 0;
}

static int test_adapter_finalize_idempotent_spbsv(void) {
  fb_backend_vtable_t vtable;
  fb_spbsv_cblas_fn spbsv_call = NULL;
  fb_generic_fn first_ptr = NULL;
  fb_status_t status;
  float ab[4] = {0};
  float b[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  vtable.ext_ops[FB_OP_SPBSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_spbsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] first finalize status=%d\n", status);
    return 1;
  }

  first_ptr = vtable.ext_ops[FB_OP_SPBSV][FB_CONV_CBLAS];
  if (!first_ptr) {
    fprintf(stderr, "[FAIL] SPBSV slot missing after first finalize\n");
    return 1;
  }

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] second finalize status=%d\n", status);
    return 1;
  }

  if (vtable.ext_ops[FB_OP_SPBSV][FB_CONV_CBLAS] != first_ptr) {
    fprintf(stderr,
            "[FAIL] SPBSV adapter pointer changed after second finalize\n");
    return 1;
  }

  spbsv_call = (fb_spbsv_cblas_fn)vtable.ext_ops[FB_OP_SPBSV][FB_CONV_CBLAS];
  if (!spbsv_call) {
    fprintf(stderr, "[FAIL] SPBSV call pointer missing after second finalize\n");
    return 1;
  }

  if (spbsv_call(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, 1, ab, 2, b, 2) !=
      0) {
    fprintf(stderr, "[FAIL] SPBSV call failed after repeated finalize\n");
    return 1;
  }

  printf("[PASS] adapter finalize is idempotent for SPBSV bridge\n");
  return 0;
}

static int test_adapter_finalize_idempotent_dpbsv(void) {
  fb_backend_vtable_t vtable;
  fb_dpbsv_cblas_fn dpbsv_call = NULL;
  fb_generic_fn first_ptr = NULL;
  fb_status_t status;
  double ab[4] = {0};
  double b[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  vtable.ext_ops[FB_OP_DPBSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dpbsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] first finalize status=%d\n", status);
    return 1;
  }

  first_ptr = vtable.ext_ops[FB_OP_DPBSV][FB_CONV_CBLAS];
  if (!first_ptr) {
    fprintf(stderr, "[FAIL] DPBSV slot missing after first finalize\n");
    return 1;
  }

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] second finalize status=%d\n", status);
    return 1;
  }

  if (vtable.ext_ops[FB_OP_DPBSV][FB_CONV_CBLAS] != first_ptr) {
    fprintf(stderr,
            "[FAIL] DPBSV adapter pointer changed after second finalize\n");
    return 1;
  }

  dpbsv_call = (fb_dpbsv_cblas_fn)vtable.ext_ops[FB_OP_DPBSV][FB_CONV_CBLAS];
  if (!dpbsv_call) {
    fprintf(stderr, "[FAIL] DPBSV call pointer missing after second finalize\n");
    return 1;
  }

  if (dpbsv_call(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, 1, ab, 2, b, 2) !=
      0) {
    fprintf(stderr, "[FAIL] DPBSV call failed after repeated finalize\n");
    return 1;
  }

  printf("[PASS] adapter finalize is idempotent for DPBSV bridge\n");
  return 0;
}

static int test_adapter_finalize_idempotent_cpbsv(void) {
  fb_backend_vtable_t vtable;
  fb_cpbsv_cblas_fn cpbsv_call = NULL;
  fb_generic_fn first_ptr = NULL;
  fb_status_t status;
  fb_complex_float_t ab[4] = {0};
  fb_complex_float_t b[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  vtable.ext_ops[FB_OP_CPBSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_cpbsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] first finalize status=%d\n", status);
    return 1;
  }

  first_ptr = vtable.ext_ops[FB_OP_CPBSV][FB_CONV_CBLAS];
  if (!first_ptr) {
    fprintf(stderr, "[FAIL] CPBSV slot missing after first finalize\n");
    return 1;
  }

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] second finalize status=%d\n", status);
    return 1;
  }

  if (vtable.ext_ops[FB_OP_CPBSV][FB_CONV_CBLAS] != first_ptr) {
    fprintf(stderr,
            "[FAIL] CPBSV adapter pointer changed after second finalize\n");
    return 1;
  }

  cpbsv_call = (fb_cpbsv_cblas_fn)vtable.ext_ops[FB_OP_CPBSV][FB_CONV_CBLAS];
  if (!cpbsv_call) {
    fprintf(stderr, "[FAIL] CPBSV call pointer missing after second finalize\n");
    return 1;
  }

  if (cpbsv_call(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, 1, ab, 2, b, 2) !=
      0) {
    fprintf(stderr, "[FAIL] CPBSV call failed after repeated finalize\n");
    return 1;
  }

  printf("[PASS] adapter finalize is idempotent for CPBSV bridge\n");
  return 0;
}

static int test_adapter_finalize_idempotent_zpbsv(void) {
  fb_backend_vtable_t vtable;
  fb_zpbsv_cblas_fn zpbsv_call = NULL;
  fb_generic_fn first_ptr = NULL;
  fb_status_t status;
  fb_complex_double_t ab[4] = {0};
  fb_complex_double_t b[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  vtable.ext_ops[FB_OP_ZPBSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zpbsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] first finalize status=%d\n", status);
    return 1;
  }

  first_ptr = vtable.ext_ops[FB_OP_ZPBSV][FB_CONV_CBLAS];
  if (!first_ptr) {
    fprintf(stderr, "[FAIL] ZPBSV slot missing after first finalize\n");
    return 1;
  }

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] second finalize status=%d\n", status);
    return 1;
  }

  if (vtable.ext_ops[FB_OP_ZPBSV][FB_CONV_CBLAS] != first_ptr) {
    fprintf(stderr,
            "[FAIL] ZPBSV adapter pointer changed after second finalize\n");
    return 1;
  }

  zpbsv_call = (fb_zpbsv_cblas_fn)vtable.ext_ops[FB_OP_ZPBSV][FB_CONV_CBLAS];
  if (!zpbsv_call) {
    fprintf(stderr, "[FAIL] ZPBSV call pointer missing after second finalize\n");
    return 1;
  }

  if (zpbsv_call(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, 1, ab, 2, b, 2) !=
      0) {
    fprintf(stderr, "[FAIL] ZPBSV call failed after repeated finalize\n");
    return 1;
  }

  printf("[PASS] adapter finalize is idempotent for ZPBSV bridge\n");
  return 0;
}

static int test_adapter_finalize_idempotent_sppsv(void) {
  fb_backend_vtable_t vtable;
  fb_sppsv_cblas_fn sppsv_call = NULL;
  fb_generic_fn first_ptr = NULL;
  fb_status_t status;
  float ap[3] = {0};
  float b[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  vtable.ext_ops[FB_OP_SPPSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sppsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] first finalize status=%d\n", status);
    return 1;
  }

  first_ptr = vtable.ext_ops[FB_OP_SPPSV][FB_CONV_CBLAS];
  if (!first_ptr) {
    fprintf(stderr, "[FAIL] SPPSV slot missing after first finalize\n");
    return 1;
  }

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] second finalize status=%d\n", status);
    return 1;
  }

  if (vtable.ext_ops[FB_OP_SPPSV][FB_CONV_CBLAS] != first_ptr) {
    fprintf(stderr,
            "[FAIL] SPPSV adapter pointer changed after second finalize\n");
    return 1;
  }

  sppsv_call = (fb_sppsv_cblas_fn)vtable.ext_ops[FB_OP_SPPSV][FB_CONV_CBLAS];
  if (!sppsv_call) {
    fprintf(stderr, "[FAIL] SPPSV call pointer missing after second finalize\n");
    return 1;
  }

  if (sppsv_call(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, ap, b, 2) != 0) {
    fprintf(stderr, "[FAIL] SPPSV call failed after repeated finalize\n");
    return 1;
  }

  printf("[PASS] adapter finalize is idempotent for SPPSV bridge\n");
  return 0;
}

static int test_adapter_finalize_idempotent_dppsv(void) {
  fb_backend_vtable_t vtable;
  fb_dppsv_cblas_fn dppsv_call = NULL;
  fb_generic_fn first_ptr = NULL;
  fb_status_t status;
  double ap[3] = {0};
  double b[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  vtable.ext_ops[FB_OP_DPPSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dppsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] first finalize status=%d\n", status);
    return 1;
  }

  first_ptr = vtable.ext_ops[FB_OP_DPPSV][FB_CONV_CBLAS];
  if (!first_ptr) {
    fprintf(stderr, "[FAIL] DPPSV slot missing after first finalize\n");
    return 1;
  }

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] second finalize status=%d\n", status);
    return 1;
  }

  if (vtable.ext_ops[FB_OP_DPPSV][FB_CONV_CBLAS] != first_ptr) {
    fprintf(stderr,
            "[FAIL] DPPSV adapter pointer changed after second finalize\n");
    return 1;
  }

  dppsv_call = (fb_dppsv_cblas_fn)vtable.ext_ops[FB_OP_DPPSV][FB_CONV_CBLAS];
  if (!dppsv_call) {
    fprintf(stderr, "[FAIL] DPPSV call pointer missing after second finalize\n");
    return 1;
  }

  if (dppsv_call(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, ap, b, 2) != 0) {
    fprintf(stderr, "[FAIL] DPPSV call failed after repeated finalize\n");
    return 1;
  }

  printf("[PASS] adapter finalize is idempotent for DPPSV bridge\n");
  return 0;
}

static int test_adapter_finalize_idempotent_cppsv(void) {
  fb_backend_vtable_t vtable;
  fb_cppsv_cblas_fn cppsv_call = NULL;
  fb_generic_fn first_ptr = NULL;
  fb_status_t status;
  fb_complex_float_t ap[3] = {0};
  fb_complex_float_t b[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  vtable.ext_ops[FB_OP_CPPSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_cppsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] first finalize status=%d\n", status);
    return 1;
  }

  first_ptr = vtable.ext_ops[FB_OP_CPPSV][FB_CONV_CBLAS];
  if (!first_ptr) {
    fprintf(stderr, "[FAIL] CPPSV slot missing after first finalize\n");
    return 1;
  }

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] second finalize status=%d\n", status);
    return 1;
  }

  if (vtable.ext_ops[FB_OP_CPPSV][FB_CONV_CBLAS] != first_ptr) {
    fprintf(stderr,
            "[FAIL] CPPSV adapter pointer changed after second finalize\n");
    return 1;
  }

  cppsv_call = (fb_cppsv_cblas_fn)vtable.ext_ops[FB_OP_CPPSV][FB_CONV_CBLAS];
  if (!cppsv_call) {
    fprintf(stderr, "[FAIL] CPPSV call pointer missing after second finalize\n");
    return 1;
  }

  if (cppsv_call(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, ap, b, 2) != 0) {
    fprintf(stderr, "[FAIL] CPPSV call failed after repeated finalize\n");
    return 1;
  }

  printf("[PASS] adapter finalize is idempotent for CPPSV bridge\n");
  return 0;
}

static int test_adapter_finalize_idempotent_zppsv(void) {
  fb_backend_vtable_t vtable;
  fb_zppsv_cblas_fn zppsv_call = NULL;
  fb_generic_fn first_ptr = NULL;
  fb_status_t status;
  fb_complex_double_t ap[3] = {0};
  fb_complex_double_t b[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  vtable.ext_ops[FB_OP_ZPPSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zppsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] first finalize status=%d\n", status);
    return 1;
  }

  first_ptr = vtable.ext_ops[FB_OP_ZPPSV][FB_CONV_CBLAS];
  if (!first_ptr) {
    fprintf(stderr, "[FAIL] ZPPSV slot missing after first finalize\n");
    return 1;
  }

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] second finalize status=%d\n", status);
    return 1;
  }

  if (vtable.ext_ops[FB_OP_ZPPSV][FB_CONV_CBLAS] != first_ptr) {
    fprintf(stderr,
            "[FAIL] ZPPSV adapter pointer changed after second finalize\n");
    return 1;
  }

  zppsv_call = (fb_zppsv_cblas_fn)vtable.ext_ops[FB_OP_ZPPSV][FB_CONV_CBLAS];
  if (!zppsv_call) {
    fprintf(stderr, "[FAIL] ZPPSV call pointer missing after second finalize\n");
    return 1;
  }

  if (zppsv_call(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, ap, b, 2) != 0) {
    fprintf(stderr, "[FAIL] ZPPSV call failed after repeated finalize\n");
    return 1;
  }

  printf("[PASS] adapter finalize is idempotent for ZPPSV bridge\n");
  return 0;
}

static int test_adapter_finalize_idempotent_sspsv(void) {
  fb_backend_vtable_t vtable;
  fb_sspsv_cblas_fn sspsv_call = NULL;
  fb_generic_fn first_ptr = NULL;
  fb_status_t status;
  float ap[3] = {0};
  float b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  vtable.ext_ops[FB_OP_SSPSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sspsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] first finalize status=%d\n", status);
    return 1;
  }

  first_ptr = vtable.ext_ops[FB_OP_SSPSV][FB_CONV_CBLAS];
  if (!first_ptr) {
    fprintf(stderr, "[FAIL] SSPSV slot missing after first finalize\n");
    return 1;
  }

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] second finalize status=%d\n", status);
    return 1;
  }

  if (vtable.ext_ops[FB_OP_SSPSV][FB_CONV_CBLAS] != first_ptr) {
    fprintf(stderr,
            "[FAIL] SSPSV adapter pointer changed after second finalize\n");
    return 1;
  }

  sspsv_call = (fb_sspsv_cblas_fn)vtable.ext_ops[FB_OP_SSPSV][FB_CONV_CBLAS];
  if (!sspsv_call) {
    fprintf(stderr, "[FAIL] SSPSV call pointer missing after second finalize\n");
    return 1;
  }

  if (sspsv_call(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, ap, ipiv, b, 2) !=
      0) {
    fprintf(stderr, "[FAIL] SSPSV call failed after repeated finalize\n");
    return 1;
  }

  printf("[PASS] adapter finalize is idempotent for SSPSV bridge\n");
  return 0;
}

static int test_adapter_finalize_idempotent_dspsv(void) {
  fb_backend_vtable_t vtable;
  fb_dspsv_cblas_fn dspsv_call = NULL;
  fb_generic_fn first_ptr = NULL;
  fb_status_t status;
  double ap[3] = {0};
  double b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  vtable.ext_ops[FB_OP_DSPSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dspsvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] first finalize status=%d\n", status);
    return 1;
  }

  first_ptr = vtable.ext_ops[FB_OP_DSPSV][FB_CONV_CBLAS];
  if (!first_ptr) {
    fprintf(stderr, "[FAIL] DSPSV slot missing after first finalize\n");
    return 1;
  }

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] second finalize status=%d\n", status);
    return 1;
  }

  if (vtable.ext_ops[FB_OP_DSPSV][FB_CONV_CBLAS] != first_ptr) {
    fprintf(stderr,
            "[FAIL] DSPSV adapter pointer changed after second finalize\n");
    return 1;
  }

  dspsv_call = (fb_dspsv_cblas_fn)vtable.ext_ops[FB_OP_DSPSV][FB_CONV_CBLAS];
  if (!dspsv_call) {
    fprintf(stderr, "[FAIL] DSPSV call pointer missing after second finalize\n");
    return 1;
  }

  if (dspsv_call(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, ap, ipiv, b, 2) !=
      0) {
    fprintf(stderr, "[FAIL] DSPSV call failed after repeated finalize\n");
    return 1;
  }

  printf("[PASS] adapter finalize is idempotent for DSPSV bridge\n");
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

static int test_adapter_active_vtable_source_isolation_sgesv_from_sgesvxx(void) {
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
  g_sgesvxx_isolation_a_calls = 0;
  g_sgesvxx_isolation_b_calls = 0;

  vtable_a.ext_ops[FB_OP_SGESVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sgesvxx_isolation_a;
  vtable_b.ext_ops[FB_OP_SGESVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sgesvxx_isolation_b;

  status = fb_finalize_plugin_vtable(&vtable_a);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize sgesvxx vtable_a status=%d\n", status);
    return 1;
  }
  status = fb_finalize_plugin_vtable(&vtable_b);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize sgesvxx vtable_b status=%d\n", status);
    return 1;
  }

  sgesv_a = (fb_sgesv_fn)vtable_a.ext_ops[FB_OP_SGESV][FB_CONV_CBLAS];
  sgesv_b = (fb_sgesv_fn)vtable_b.ext_ops[FB_OP_SGESV][FB_CONV_CBLAS];
  if (!sgesv_a || !sgesv_b) {
    fprintf(stderr, "[FAIL] SGESV adapter slot missing in SGESVXX isolation test\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_a);
  if (sgesv_a(FB_LAYOUT_COL_MAJOR, 2, 1, a, 2, ipiv, b, 2) != 1303) {
    fprintf(stderr,
            "[FAIL] SGESV(SGESVXX) call for active vtable_a did not route to source A\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_b);
  if (sgesv_b(FB_LAYOUT_COL_MAJOR, 2, 1, a, 2, ipiv, b, 2) != 1304) {
    fprintf(stderr,
            "[FAIL] SGESV(SGESVXX) call for active vtable_b did not route to source B\n");
    return 1;
  }

  if (g_sgesvxx_isolation_a_calls != 1 || g_sgesvxx_isolation_b_calls != 1) {
    fprintf(stderr,
            "[FAIL] SGESV(SGESVXX) active-vtable isolation mismatch (A=%d B=%d)\n",
            g_sgesvxx_isolation_a_calls, g_sgesvxx_isolation_b_calls);
    return 1;
  }

  printf("[PASS] SGESV adapter source isolation follows active SGESVXX donor selection\n");
  return 0;
}

static int test_adapter_active_vtable_source_isolation_dgesv_from_dgesvxx(void) {
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
  g_dgesvxx_isolation_a_calls = 0;
  g_dgesvxx_isolation_b_calls = 0;

  vtable_a.ext_ops[FB_OP_DGESVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dgesvxx_isolation_a;
  vtable_b.ext_ops[FB_OP_DGESVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dgesvxx_isolation_b;

  status = fb_finalize_plugin_vtable(&vtable_a);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize dgesvxx vtable_a status=%d\n", status);
    return 1;
  }
  status = fb_finalize_plugin_vtable(&vtable_b);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize dgesvxx vtable_b status=%d\n", status);
    return 1;
  }

  dgesv_a = (fb_dgesv_fn)vtable_a.ext_ops[FB_OP_DGESV][FB_CONV_CBLAS];
  dgesv_b = (fb_dgesv_fn)vtable_b.ext_ops[FB_OP_DGESV][FB_CONV_CBLAS];
  if (!dgesv_a || !dgesv_b) {
    fprintf(stderr, "[FAIL] DGESV adapter slot missing in DGESVXX isolation test\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_a);
  if (dgesv_a(FB_LAYOUT_COL_MAJOR, 2, 1, a, 2, ipiv, b, 2) != 1305) {
    fprintf(stderr,
            "[FAIL] DGESV(DGESVXX) call for active vtable_a did not route to source A\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_b);
  if (dgesv_b(FB_LAYOUT_COL_MAJOR, 2, 1, a, 2, ipiv, b, 2) != 1306) {
    fprintf(stderr,
            "[FAIL] DGESV(DGESVXX) call for active vtable_b did not route to source B\n");
    return 1;
  }

  if (g_dgesvxx_isolation_a_calls != 1 || g_dgesvxx_isolation_b_calls != 1) {
    fprintf(stderr,
            "[FAIL] DGESV(DGESVXX) active-vtable isolation mismatch (A=%d B=%d)\n",
            g_dgesvxx_isolation_a_calls, g_dgesvxx_isolation_b_calls);
    return 1;
  }

  printf("[PASS] DGESV adapter source isolation follows active DGESVXX donor selection\n");
  return 0;
}

static int test_adapter_active_vtable_source_isolation_cgesv_from_cgesvxx(void) {
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
  g_cgesvxx_isolation_a_calls = 0;
  g_cgesvxx_isolation_b_calls = 0;

  vtable_a.ext_ops[FB_OP_CGESVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_cgesvxx_isolation_a;
  vtable_b.ext_ops[FB_OP_CGESVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_cgesvxx_isolation_b;

  status = fb_finalize_plugin_vtable(&vtable_a);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize cgesvxx vtable_a status=%d\n", status);
    return 1;
  }
  status = fb_finalize_plugin_vtable(&vtable_b);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize cgesvxx vtable_b status=%d\n", status);
    return 1;
  }

  cgesv_a = (fb_cgesv_fn)vtable_a.ext_ops[FB_OP_CGESV][FB_CONV_CBLAS];
  cgesv_b = (fb_cgesv_fn)vtable_b.ext_ops[FB_OP_CGESV][FB_CONV_CBLAS];
  if (!cgesv_a || !cgesv_b) {
    fprintf(stderr, "[FAIL] CGESV adapter slot missing in CGESVXX isolation test\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_a);
  if (cgesv_a(FB_LAYOUT_COL_MAJOR, 2, 1, a, 2, ipiv, b, 2) != 1307) {
    fprintf(stderr,
            "[FAIL] CGESV(CGESVXX) call for active vtable_a did not route to source A\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_b);
  if (cgesv_b(FB_LAYOUT_COL_MAJOR, 2, 1, a, 2, ipiv, b, 2) != 1308) {
    fprintf(stderr,
            "[FAIL] CGESV(CGESVXX) call for active vtable_b did not route to source B\n");
    return 1;
  }

  if (g_cgesvxx_isolation_a_calls != 1 || g_cgesvxx_isolation_b_calls != 1) {
    fprintf(stderr,
            "[FAIL] CGESV(CGESVXX) active-vtable isolation mismatch (A=%d B=%d)\n",
            g_cgesvxx_isolation_a_calls, g_cgesvxx_isolation_b_calls);
    return 1;
  }

  printf("[PASS] CGESV adapter source isolation follows active CGESVXX donor selection\n");
  return 0;
}

static int test_adapter_active_vtable_source_isolation_zgesv_from_zgesvxx(void) {
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
  g_zgesvxx_isolation_a_calls = 0;
  g_zgesvxx_isolation_b_calls = 0;

  vtable_a.ext_ops[FB_OP_ZGESVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zgesvxx_isolation_a;
  vtable_b.ext_ops[FB_OP_ZGESVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zgesvxx_isolation_b;

  status = fb_finalize_plugin_vtable(&vtable_a);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize zgesvxx vtable_a status=%d\n", status);
    return 1;
  }
  status = fb_finalize_plugin_vtable(&vtable_b);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize zgesvxx vtable_b status=%d\n", status);
    return 1;
  }

  zgesv_a = (fb_zgesv_fn)vtable_a.ext_ops[FB_OP_ZGESV][FB_CONV_CBLAS];
  zgesv_b = (fb_zgesv_fn)vtable_b.ext_ops[FB_OP_ZGESV][FB_CONV_CBLAS];
  if (!zgesv_a || !zgesv_b) {
    fprintf(stderr, "[FAIL] ZGESV adapter slot missing in ZGESVXX isolation test\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_a);
  if (zgesv_a(FB_LAYOUT_COL_MAJOR, 2, 1, a, 2, ipiv, b, 2) != 1309) {
    fprintf(stderr,
            "[FAIL] ZGESV(ZGESVXX) call for active vtable_a did not route to source A\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_b);
  if (zgesv_b(FB_LAYOUT_COL_MAJOR, 2, 1, a, 2, ipiv, b, 2) != 1310) {
    fprintf(stderr,
            "[FAIL] ZGESV(ZGESVXX) call for active vtable_b did not route to source B\n");
    return 1;
  }

  if (g_zgesvxx_isolation_a_calls != 1 || g_zgesvxx_isolation_b_calls != 1) {
    fprintf(stderr,
            "[FAIL] ZGESV(ZGESVXX) active-vtable isolation mismatch (A=%d B=%d)\n",
            g_zgesvxx_isolation_a_calls, g_zgesvxx_isolation_b_calls);
    return 1;
  }

  printf("[PASS] ZGESV adapter source isolation follows active ZGESVXX donor selection\n");
  return 0;
}

static int test_adapter_active_vtable_source_isolation_sposv_from_sposvxx(void) {
  fb_backend_vtable_t vtable_a;
  fb_backend_vtable_t vtable_b;
  fb_sposv_fn sposv_a = NULL;
  fb_sposv_fn sposv_b = NULL;
  fb_status_t status;

  float a[4] = {0};
  float b[2] = {0};

  memset(&vtable_a, 0, sizeof(vtable_a));
  memset(&vtable_b, 0, sizeof(vtable_b));
  g_sposvxx_isolation_a_calls = 0;
  g_sposvxx_isolation_b_calls = 0;

  vtable_a.ext_ops[FB_OP_SPOSVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sposvxx_isolation_a;
  vtable_b.ext_ops[FB_OP_SPOSVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sposvxx_isolation_b;

  status = fb_finalize_plugin_vtable(&vtable_a);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize sposvxx vtable_a status=%d\n", status);
    return 1;
  }
  status = fb_finalize_plugin_vtable(&vtable_b);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize sposvxx vtable_b status=%d\n", status);
    return 1;
  }

  sposv_a = (fb_sposv_fn)vtable_a.ext_ops[FB_OP_SPOSV][FB_CONV_CBLAS];
  sposv_b = (fb_sposv_fn)vtable_b.ext_ops[FB_OP_SPOSV][FB_CONV_CBLAS];
  if (!sposv_a || !sposv_b) {
    fprintf(stderr, "[FAIL] SPOSV adapter slot missing in SPOSVXX isolation test\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_a);
  if (sposv_a(FB_LAYOUT_COL_MAJOR, FB_UPPER, 2, 1, a, 2, b, 2) != 1321) {
    fprintf(stderr,
            "[FAIL] SPOSV(SPOSVXX) call for active vtable_a did not route to source A\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_b);
  if (sposv_b(FB_LAYOUT_COL_MAJOR, FB_UPPER, 2, 1, a, 2, b, 2) != 1322) {
    fprintf(stderr,
            "[FAIL] SPOSV(SPOSVXX) call for active vtable_b did not route to source B\n");
    return 1;
  }

  if (g_sposvxx_isolation_a_calls != 1 || g_sposvxx_isolation_b_calls != 1) {
    fprintf(stderr,
            "[FAIL] SPOSV(SPOSVXX) active-vtable isolation mismatch (A=%d B=%d)\n",
            g_sposvxx_isolation_a_calls, g_sposvxx_isolation_b_calls);
    return 1;
  }

  printf("[PASS] SPOSV adapter source isolation follows active SPOSVXX donor selection\n");
  return 0;
}

static int test_adapter_active_vtable_source_isolation_dposv_from_dposvxx(void) {
  fb_backend_vtable_t vtable_a;
  fb_backend_vtable_t vtable_b;
  fb_dposv_fn dposv_a = NULL;
  fb_dposv_fn dposv_b = NULL;
  fb_status_t status;

  double a[4] = {0};
  double b[2] = {0};

  memset(&vtable_a, 0, sizeof(vtable_a));
  memset(&vtable_b, 0, sizeof(vtable_b));
  g_dposvxx_isolation_a_calls = 0;
  g_dposvxx_isolation_b_calls = 0;

  vtable_a.ext_ops[FB_OP_DPOSVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dposvxx_isolation_a;
  vtable_b.ext_ops[FB_OP_DPOSVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dposvxx_isolation_b;

  status = fb_finalize_plugin_vtable(&vtable_a);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize dposvxx vtable_a status=%d\n", status);
    return 1;
  }
  status = fb_finalize_plugin_vtable(&vtable_b);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize dposvxx vtable_b status=%d\n", status);
    return 1;
  }

  dposv_a = (fb_dposv_fn)vtable_a.ext_ops[FB_OP_DPOSV][FB_CONV_CBLAS];
  dposv_b = (fb_dposv_fn)vtable_b.ext_ops[FB_OP_DPOSV][FB_CONV_CBLAS];
  if (!dposv_a || !dposv_b) {
    fprintf(stderr, "[FAIL] DPOSV adapter slot missing in DPOSVXX isolation test\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_a);
  if (dposv_a(FB_LAYOUT_COL_MAJOR, FB_UPPER, 2, 1, a, 2, b, 2) != 1323) {
    fprintf(stderr,
            "[FAIL] DPOSV(DPOSVXX) call for active vtable_a did not route to source A\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_b);
  if (dposv_b(FB_LAYOUT_COL_MAJOR, FB_UPPER, 2, 1, a, 2, b, 2) != 1324) {
    fprintf(stderr,
            "[FAIL] DPOSV(DPOSVXX) call for active vtable_b did not route to source B\n");
    return 1;
  }

  if (g_dposvxx_isolation_a_calls != 1 || g_dposvxx_isolation_b_calls != 1) {
    fprintf(stderr,
            "[FAIL] DPOSV(DPOSVXX) active-vtable isolation mismatch (A=%d B=%d)\n",
            g_dposvxx_isolation_a_calls, g_dposvxx_isolation_b_calls);
    return 1;
  }

  printf("[PASS] DPOSV adapter source isolation follows active DPOSVXX donor selection\n");
  return 0;
}

static int test_adapter_active_vtable_source_isolation_cposv_from_cposvxx(void) {
  fb_backend_vtable_t vtable_a;
  fb_backend_vtable_t vtable_b;
  fb_cposv_fn cposv_a = NULL;
  fb_cposv_fn cposv_b = NULL;
  fb_status_t status;

  fb_complex_float_t a[4] = {0};
  fb_complex_float_t b[2] = {0};

  memset(&vtable_a, 0, sizeof(vtable_a));
  memset(&vtable_b, 0, sizeof(vtable_b));
  g_cposvxx_isolation_a_calls = 0;
  g_cposvxx_isolation_b_calls = 0;

  vtable_a.ext_ops[FB_OP_CPOSVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_cposvxx_isolation_a;
  vtable_b.ext_ops[FB_OP_CPOSVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_cposvxx_isolation_b;

  status = fb_finalize_plugin_vtable(&vtable_a);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize cposvxx vtable_a status=%d\n", status);
    return 1;
  }
  status = fb_finalize_plugin_vtable(&vtable_b);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize cposvxx vtable_b status=%d\n", status);
    return 1;
  }

  cposv_a = (fb_cposv_fn)vtable_a.ext_ops[FB_OP_CPOSV][FB_CONV_CBLAS];
  cposv_b = (fb_cposv_fn)vtable_b.ext_ops[FB_OP_CPOSV][FB_CONV_CBLAS];
  if (!cposv_a || !cposv_b) {
    fprintf(stderr, "[FAIL] CPOSV adapter slot missing in CPOSVXX isolation test\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_a);
  if (cposv_a(FB_LAYOUT_COL_MAJOR, FB_UPPER, 2, 1, a, 2, b, 2) != 1325) {
    fprintf(stderr,
            "[FAIL] CPOSV(CPOSVXX) call for active vtable_a did not route to source A\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_b);
  if (cposv_b(FB_LAYOUT_COL_MAJOR, FB_UPPER, 2, 1, a, 2, b, 2) != 1326) {
    fprintf(stderr,
            "[FAIL] CPOSV(CPOSVXX) call for active vtable_b did not route to source B\n");
    return 1;
  }

  if (g_cposvxx_isolation_a_calls != 1 || g_cposvxx_isolation_b_calls != 1) {
    fprintf(stderr,
            "[FAIL] CPOSV(CPOSVXX) active-vtable isolation mismatch (A=%d B=%d)\n",
            g_cposvxx_isolation_a_calls, g_cposvxx_isolation_b_calls);
    return 1;
  }

  printf("[PASS] CPOSV adapter source isolation follows active CPOSVXX donor selection\n");
  return 0;
}

static int test_adapter_active_vtable_source_isolation_zposv_from_zposvxx(void) {
  fb_backend_vtable_t vtable_a;
  fb_backend_vtable_t vtable_b;
  fb_zposv_fn zposv_a = NULL;
  fb_zposv_fn zposv_b = NULL;
  fb_status_t status;

  fb_complex_double_t a[4] = {0};
  fb_complex_double_t b[2] = {0};

  memset(&vtable_a, 0, sizeof(vtable_a));
  memset(&vtable_b, 0, sizeof(vtable_b));
  g_zposvxx_isolation_a_calls = 0;
  g_zposvxx_isolation_b_calls = 0;

  vtable_a.ext_ops[FB_OP_ZPOSVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zposvxx_isolation_a;
  vtable_b.ext_ops[FB_OP_ZPOSVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zposvxx_isolation_b;

  status = fb_finalize_plugin_vtable(&vtable_a);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize zposvxx vtable_a status=%d\n", status);
    return 1;
  }
  status = fb_finalize_plugin_vtable(&vtable_b);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize zposvxx vtable_b status=%d\n", status);
    return 1;
  }

  zposv_a = (fb_zposv_fn)vtable_a.ext_ops[FB_OP_ZPOSV][FB_CONV_CBLAS];
  zposv_b = (fb_zposv_fn)vtable_b.ext_ops[FB_OP_ZPOSV][FB_CONV_CBLAS];
  if (!zposv_a || !zposv_b) {
    fprintf(stderr, "[FAIL] ZPOSV adapter slot missing in ZPOSVXX isolation test\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_a);
  if (zposv_a(FB_LAYOUT_COL_MAJOR, FB_UPPER, 2, 1, a, 2, b, 2) != 1327) {
    fprintf(stderr,
            "[FAIL] ZPOSV(ZPOSVXX) call for active vtable_a did not route to source A\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_b);
  if (zposv_b(FB_LAYOUT_COL_MAJOR, FB_UPPER, 2, 1, a, 2, b, 2) != 1328) {
    fprintf(stderr,
            "[FAIL] ZPOSV(ZPOSVXX) call for active vtable_b did not route to source B\n");
    return 1;
  }

  if (g_zposvxx_isolation_a_calls != 1 || g_zposvxx_isolation_b_calls != 1) {
    fprintf(stderr,
            "[FAIL] ZPOSV(ZPOSVXX) active-vtable isolation mismatch (A=%d B=%d)\n",
            g_zposvxx_isolation_a_calls, g_zposvxx_isolation_b_calls);
    return 1;
  }

  printf("[PASS] ZPOSV adapter source isolation follows active ZPOSVXX donor selection\n");
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

static int test_adapter_active_vtable_source_isolation_chesv_from_chesvx(void) {
  fb_backend_vtable_t vtable_a;
  fb_backend_vtable_t vtable_b;
  fb_csysv_fn chesv_a = NULL;
  fb_csysv_fn chesv_b = NULL;
  fb_status_t status;

  fb_complex_float_t a[4] = {0};
  fb_complex_float_t b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable_a, 0, sizeof(vtable_a));
  memset(&vtable_b, 0, sizeof(vtable_b));
  g_csysvx_isolation_a_calls = 0;
  g_csysvx_isolation_b_calls = 0;

  vtable_a.ext_ops[FB_OP_CHESVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_csysvx_isolation_a;
  vtable_b.ext_ops[FB_OP_CHESVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_csysvx_isolation_b;

  status = fb_finalize_plugin_vtable(&vtable_a);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize chesvx vtable_a status=%d\n", status);
    return 1;
  }
  status = fb_finalize_plugin_vtable(&vtable_b);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize chesvx vtable_b status=%d\n", status);
    return 1;
  }

  chesv_a = (fb_csysv_fn)vtable_a.ext_ops[FB_OP_CHESV][FB_CONV_CBLAS];
  chesv_b = (fb_csysv_fn)vtable_b.ext_ops[FB_OP_CHESV][FB_CONV_CBLAS];
  if (!chesv_a || !chesv_b) {
    fprintf(stderr, "[FAIL] CHESV adapter slot missing in CHESVX isolation test\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_a);
  if (chesv_a(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, a, 2, ipiv, b, 2) !=
      2421) {
    fprintf(stderr,
            "[FAIL] CHESV(CHESVX) call for active vtable_a did not route to source A\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_b);
  if (chesv_b(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, a, 2, ipiv, b, 2) !=
      2522) {
    fprintf(stderr,
            "[FAIL] CHESV(CHESVX) call for active vtable_b did not route to source B\n");
    return 1;
  }

  if (g_csysvx_isolation_a_calls != 1 || g_csysvx_isolation_b_calls != 1) {
    fprintf(stderr,
            "[FAIL] CHESV(CHESVX) active-vtable isolation mismatch (A=%d B=%d)\n",
            g_csysvx_isolation_a_calls, g_csysvx_isolation_b_calls);
    return 1;
  }

  printf("[PASS] CHESV adapter source isolation follows active CHESVX donor selection\n");
  return 0;
}

static int test_adapter_active_vtable_source_isolation_zhesv_from_zhesvx(void) {
  fb_backend_vtable_t vtable_a;
  fb_backend_vtable_t vtable_b;
  fb_zsysv_fn zhesv_a = NULL;
  fb_zsysv_fn zhesv_b = NULL;
  fb_status_t status;

  fb_complex_double_t a[4] = {0};
  fb_complex_double_t b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable_a, 0, sizeof(vtable_a));
  memset(&vtable_b, 0, sizeof(vtable_b));
  g_zsysvx_isolation_a_calls = 0;
  g_zsysvx_isolation_b_calls = 0;

  vtable_a.ext_ops[FB_OP_ZHESVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zsysvx_isolation_a;
  vtable_b.ext_ops[FB_OP_ZHESVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zsysvx_isolation_b;

  status = fb_finalize_plugin_vtable(&vtable_a);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize zhesvx vtable_a status=%d\n", status);
    return 1;
  }
  status = fb_finalize_plugin_vtable(&vtable_b);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize zhesvx vtable_b status=%d\n", status);
    return 1;
  }

  zhesv_a = (fb_zsysv_fn)vtable_a.ext_ops[FB_OP_ZHESV][FB_CONV_CBLAS];
  zhesv_b = (fb_zsysv_fn)vtable_b.ext_ops[FB_OP_ZHESV][FB_CONV_CBLAS];
  if (!zhesv_a || !zhesv_b) {
    fprintf(stderr, "[FAIL] ZHESV adapter slot missing in ZHESVX isolation test\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_a);
  if (zhesv_a(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, a, 2, ipiv, b, 2) !=
      2623) {
    fprintf(stderr,
            "[FAIL] ZHESV(ZHESVX) call for active vtable_a did not route to source A\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_b);
  if (zhesv_b(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, a, 2, ipiv, b, 2) !=
      2724) {
    fprintf(stderr,
            "[FAIL] ZHESV(ZHESVX) call for active vtable_b did not route to source B\n");
    return 1;
  }

  if (g_zsysvx_isolation_a_calls != 1 || g_zsysvx_isolation_b_calls != 1) {
    fprintf(stderr,
            "[FAIL] ZHESV(ZHESVX) active-vtable isolation mismatch (A=%d B=%d)\n",
            g_zsysvx_isolation_a_calls, g_zsysvx_isolation_b_calls);
    return 1;
  }

  printf("[PASS] ZHESV adapter source isolation follows active ZHESVX donor selection\n");
  return 0;
}

static int test_adapter_active_vtable_source_isolation_ssysv_from_ssysvxx(void) {
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
  g_ssysvxx_isolation_a_calls = 0;
  g_ssysvxx_isolation_b_calls = 0;

  vtable_a.ext_ops[FB_OP_SSYSVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_ssysvxx_isolation_a;
  vtable_b.ext_ops[FB_OP_SSYSVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_ssysvxx_isolation_b;

  status = fb_finalize_plugin_vtable(&vtable_a);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize ssysvxx vtable_a status=%d\n", status);
    return 1;
  }
  status = fb_finalize_plugin_vtable(&vtable_b);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize ssysvxx vtable_b status=%d\n", status);
    return 1;
  }

  ssysv_a = (fb_ssysv_fn)vtable_a.ext_ops[FB_OP_SSYSV][FB_CONV_CBLAS];
  ssysv_b = (fb_ssysv_fn)vtable_b.ext_ops[FB_OP_SSYSV][FB_CONV_CBLAS];
  if (!ssysv_a || !ssysv_b) {
    fprintf(stderr, "[FAIL] SSYSV adapter slot missing in SSYSVXX isolation test\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_a);
  if (ssysv_a(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, a, 2, ipiv, b, 2) !=
      2921) {
    fprintf(stderr,
            "[FAIL] SSYSV(SSYSVXX) call for active vtable_a did not route to source A\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_b);
  if (ssysv_b(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, a, 2, ipiv, b, 2) !=
      2922) {
    fprintf(stderr,
            "[FAIL] SSYSV(SSYSVXX) call for active vtable_b did not route to source B\n");
    return 1;
  }

  if (g_ssysvxx_isolation_a_calls != 1 || g_ssysvxx_isolation_b_calls != 1) {
    fprintf(stderr,
            "[FAIL] SSYSV(SSYSVXX) active-vtable isolation mismatch (A=%d B=%d)\n",
            g_ssysvxx_isolation_a_calls, g_ssysvxx_isolation_b_calls);
    return 1;
  }

  printf("[PASS] SSYSV adapter source isolation follows active SSYSVXX donor selection\n");
  return 0;
}

static int test_adapter_active_vtable_source_isolation_dsysv_from_dsysvxx(void) {
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
  g_dsysvxx_isolation_a_calls = 0;
  g_dsysvxx_isolation_b_calls = 0;

  vtable_a.ext_ops[FB_OP_DSYSVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dsysvxx_isolation_a;
  vtable_b.ext_ops[FB_OP_DSYSVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dsysvxx_isolation_b;

  status = fb_finalize_plugin_vtable(&vtable_a);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize dsysvxx vtable_a status=%d\n", status);
    return 1;
  }
  status = fb_finalize_plugin_vtable(&vtable_b);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize dsysvxx vtable_b status=%d\n", status);
    return 1;
  }

  dsysv_a = (fb_dsysv_fn)vtable_a.ext_ops[FB_OP_DSYSV][FB_CONV_CBLAS];
  dsysv_b = (fb_dsysv_fn)vtable_b.ext_ops[FB_OP_DSYSV][FB_CONV_CBLAS];
  if (!dsysv_a || !dsysv_b) {
    fprintf(stderr, "[FAIL] DSYSV adapter slot missing in DSYSVXX isolation test\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_a);
  if (dsysv_a(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, a, 2, ipiv, b, 2) !=
      2923) {
    fprintf(stderr,
            "[FAIL] DSYSV(DSYSVXX) call for active vtable_a did not route to source A\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_b);
  if (dsysv_b(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, a, 2, ipiv, b, 2) !=
      2924) {
    fprintf(stderr,
            "[FAIL] DSYSV(DSYSVXX) call for active vtable_b did not route to source B\n");
    return 1;
  }

  if (g_dsysvxx_isolation_a_calls != 1 || g_dsysvxx_isolation_b_calls != 1) {
    fprintf(stderr,
            "[FAIL] DSYSV(DSYSVXX) active-vtable isolation mismatch (A=%d B=%d)\n",
            g_dsysvxx_isolation_a_calls, g_dsysvxx_isolation_b_calls);
    return 1;
  }

  printf("[PASS] DSYSV adapter source isolation follows active DSYSVXX donor selection\n");
  return 0;
}

static int test_adapter_active_vtable_source_isolation_csysv_from_csysvxx(void) {
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
  g_csysvxx_isolation_a_calls = 0;
  g_csysvxx_isolation_b_calls = 0;

  vtable_a.ext_ops[FB_OP_CSYSVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_csysvxx_isolation_a;
  vtable_b.ext_ops[FB_OP_CSYSVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_csysvxx_isolation_b;

  status = fb_finalize_plugin_vtable(&vtable_a);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize csysvxx vtable_a status=%d\n", status);
    return 1;
  }
  status = fb_finalize_plugin_vtable(&vtable_b);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize csysvxx vtable_b status=%d\n", status);
    return 1;
  }

  csysv_a = (fb_csysv_fn)vtable_a.ext_ops[FB_OP_CSYSV][FB_CONV_CBLAS];
  csysv_b = (fb_csysv_fn)vtable_b.ext_ops[FB_OP_CSYSV][FB_CONV_CBLAS];
  if (!csysv_a || !csysv_b) {
    fprintf(stderr, "[FAIL] CSYSV adapter slot missing in CSYSVXX isolation test\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_a);
  if (csysv_a(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, a, 2, ipiv, b, 2) !=
      2925) {
    fprintf(stderr,
            "[FAIL] CSYSV(CSYSVXX) call for active vtable_a did not route to source A\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_b);
  if (csysv_b(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, a, 2, ipiv, b, 2) !=
      2926) {
    fprintf(stderr,
            "[FAIL] CSYSV(CSYSVXX) call for active vtable_b did not route to source B\n");
    return 1;
  }

  if (g_csysvxx_isolation_a_calls != 1 || g_csysvxx_isolation_b_calls != 1) {
    fprintf(stderr,
            "[FAIL] CSYSV(CSYSVXX) active-vtable isolation mismatch (A=%d B=%d)\n",
            g_csysvxx_isolation_a_calls, g_csysvxx_isolation_b_calls);
    return 1;
  }

  printf("[PASS] CSYSV adapter source isolation follows active CSYSVXX donor selection\n");
  return 0;
}

static int test_adapter_active_vtable_source_isolation_zsysv_from_zsysvxx(void) {
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
  g_zsysvxx_isolation_a_calls = 0;
  g_zsysvxx_isolation_b_calls = 0;

  vtable_a.ext_ops[FB_OP_ZSYSVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zsysvxx_isolation_a;
  vtable_b.ext_ops[FB_OP_ZSYSVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zsysvxx_isolation_b;

  status = fb_finalize_plugin_vtable(&vtable_a);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize zsysvxx vtable_a status=%d\n", status);
    return 1;
  }
  status = fb_finalize_plugin_vtable(&vtable_b);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize zsysvxx vtable_b status=%d\n", status);
    return 1;
  }

  zsysv_a = (fb_zsysv_fn)vtable_a.ext_ops[FB_OP_ZSYSV][FB_CONV_CBLAS];
  zsysv_b = (fb_zsysv_fn)vtable_b.ext_ops[FB_OP_ZSYSV][FB_CONV_CBLAS];
  if (!zsysv_a || !zsysv_b) {
    fprintf(stderr, "[FAIL] ZSYSV adapter slot missing in ZSYSVXX isolation test\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_a);
  if (zsysv_a(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, a, 2, ipiv, b, 2) !=
      2927) {
    fprintf(stderr,
            "[FAIL] ZSYSV(ZSYSVXX) call for active vtable_a did not route to source A\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_b);
  if (zsysv_b(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, a, 2, ipiv, b, 2) !=
      2928) {
    fprintf(stderr,
            "[FAIL] ZSYSV(ZSYSVXX) call for active vtable_b did not route to source B\n");
    return 1;
  }

  if (g_zsysvxx_isolation_a_calls != 1 || g_zsysvxx_isolation_b_calls != 1) {
    fprintf(stderr,
            "[FAIL] ZSYSV(ZSYSVXX) active-vtable isolation mismatch (A=%d B=%d)\n",
            g_zsysvxx_isolation_a_calls, g_zsysvxx_isolation_b_calls);
    return 1;
  }

  printf("[PASS] ZSYSV adapter source isolation follows active ZSYSVXX donor selection\n");
  return 0;
}

static int test_adapter_active_vtable_source_isolation_chesv_from_chesvxx(void) {
  fb_backend_vtable_t vtable_a;
  fb_backend_vtable_t vtable_b;
  fb_csysv_fn chesv_a = NULL;
  fb_csysv_fn chesv_b = NULL;
  fb_status_t status;

  fb_complex_float_t a[4] = {0};
  fb_complex_float_t b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable_a, 0, sizeof(vtable_a));
  memset(&vtable_b, 0, sizeof(vtable_b));
  g_csysvxx_isolation_a_calls = 0;
  g_csysvxx_isolation_b_calls = 0;

  vtable_a.ext_ops[FB_OP_CHESVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_csysvxx_isolation_a;
  vtable_b.ext_ops[FB_OP_CHESVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_csysvxx_isolation_b;

  status = fb_finalize_plugin_vtable(&vtable_a);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize chesvxx vtable_a status=%d\n", status);
    return 1;
  }
  status = fb_finalize_plugin_vtable(&vtable_b);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize chesvxx vtable_b status=%d\n", status);
    return 1;
  }

  chesv_a = (fb_csysv_fn)vtable_a.ext_ops[FB_OP_CHESV][FB_CONV_CBLAS];
  chesv_b = (fb_csysv_fn)vtable_b.ext_ops[FB_OP_CHESV][FB_CONV_CBLAS];
  if (!chesv_a || !chesv_b) {
    fprintf(stderr, "[FAIL] CHESV adapter slot missing in CHESVXX isolation test\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_a);
  if (chesv_a(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, a, 2, ipiv, b, 2) !=
      2925) {
    fprintf(stderr,
            "[FAIL] CHESV(CHESVXX) call for active vtable_a did not route to source A\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_b);
  if (chesv_b(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, a, 2, ipiv, b, 2) !=
      2926) {
    fprintf(stderr,
            "[FAIL] CHESV(CHESVXX) call for active vtable_b did not route to source B\n");
    return 1;
  }

  if (g_csysvxx_isolation_a_calls != 1 || g_csysvxx_isolation_b_calls != 1) {
    fprintf(stderr,
            "[FAIL] CHESV(CHESVXX) active-vtable isolation mismatch (A=%d B=%d)\n",
            g_csysvxx_isolation_a_calls, g_csysvxx_isolation_b_calls);
    return 1;
  }

  printf("[PASS] CHESV adapter source isolation follows active CHESVXX donor selection\n");
  return 0;
}

static int test_adapter_active_vtable_source_isolation_zhesv_from_zhesvxx(void) {
  fb_backend_vtable_t vtable_a;
  fb_backend_vtable_t vtable_b;
  fb_zsysv_fn zhesv_a = NULL;
  fb_zsysv_fn zhesv_b = NULL;
  fb_status_t status;

  fb_complex_double_t a[4] = {0};
  fb_complex_double_t b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable_a, 0, sizeof(vtable_a));
  memset(&vtable_b, 0, sizeof(vtable_b));
  g_zsysvxx_isolation_a_calls = 0;
  g_zsysvxx_isolation_b_calls = 0;

  vtable_a.ext_ops[FB_OP_ZHESVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zsysvxx_isolation_a;
  vtable_b.ext_ops[FB_OP_ZHESVXX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zsysvxx_isolation_b;

  status = fb_finalize_plugin_vtable(&vtable_a);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize zhesvxx vtable_a status=%d\n", status);
    return 1;
  }
  status = fb_finalize_plugin_vtable(&vtable_b);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize zhesvxx vtable_b status=%d\n", status);
    return 1;
  }

  zhesv_a = (fb_zsysv_fn)vtable_a.ext_ops[FB_OP_ZHESV][FB_CONV_CBLAS];
  zhesv_b = (fb_zsysv_fn)vtable_b.ext_ops[FB_OP_ZHESV][FB_CONV_CBLAS];
  if (!zhesv_a || !zhesv_b) {
    fprintf(stderr, "[FAIL] ZHESV adapter slot missing in ZHESVXX isolation test\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_a);
  if (zhesv_a(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, a, 2, ipiv, b, 2) !=
      2927) {
    fprintf(stderr,
            "[FAIL] ZHESV(ZHESVXX) call for active vtable_a did not route to source A\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_b);
  if (zhesv_b(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, a, 2, ipiv, b, 2) !=
      2928) {
    fprintf(stderr,
            "[FAIL] ZHESV(ZHESVXX) call for active vtable_b did not route to source B\n");
    return 1;
  }

  if (g_zsysvxx_isolation_a_calls != 1 || g_zsysvxx_isolation_b_calls != 1) {
    fprintf(stderr,
            "[FAIL] ZHESV(ZHESVXX) active-vtable isolation mismatch (A=%d B=%d)\n",
            g_zsysvxx_isolation_a_calls, g_zsysvxx_isolation_b_calls);
    return 1;
  }

  printf("[PASS] ZHESV adapter source isolation follows active ZHESVXX donor selection\n");
  return 0;
}

static int test_adapter_active_vtable_source_isolation_sgbsv(void) {
  fb_backend_vtable_t vtable_a;
  fb_backend_vtable_t vtable_b;
  fb_sgbsv_cblas_fn sgbsv_a = NULL;
  fb_sgbsv_cblas_fn sgbsv_b = NULL;
  fb_status_t status;

  float ab[8] = {0};
  float b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable_a, 0, sizeof(vtable_a));
  memset(&vtable_b, 0, sizeof(vtable_b));
  g_sgbsvx_isolation_a_calls = 0;
  g_sgbsvx_isolation_b_calls = 0;

  vtable_a.ext_ops[FB_OP_SGBSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sgbsvx_isolation_a;
  vtable_b.ext_ops[FB_OP_SGBSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sgbsvx_isolation_b;

  status = fb_finalize_plugin_vtable(&vtable_a);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize sgbsv vtable_a status=%d\n", status);
    return 1;
  }
  status = fb_finalize_plugin_vtable(&vtable_b);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize sgbsv vtable_b status=%d\n", status);
    return 1;
  }

  sgbsv_a = (fb_sgbsv_cblas_fn)vtable_a.ext_ops[FB_OP_SGBSV][FB_CONV_CBLAS];
  sgbsv_b = (fb_sgbsv_cblas_fn)vtable_b.ext_ops[FB_OP_SGBSV][FB_CONV_CBLAS];
  if (!sgbsv_a || !sgbsv_b) {
    fprintf(stderr, "[FAIL] SGBSV adapter slot missing in isolation test\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_a);
  if (sgbsv_a(FB_LAYOUT_COL_MAJOR, 2, 1, 1, 1, ab, 4, ipiv, b, 2) != 3101) {
    fprintf(stderr,
            "[FAIL] SGBSV call for active vtable_a did not route to source A\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_b);
  if (sgbsv_b(FB_LAYOUT_COL_MAJOR, 2, 1, 1, 1, ab, 4, ipiv, b, 2) != 3102) {
    fprintf(stderr,
            "[FAIL] SGBSV call for active vtable_b did not route to source B\n");
    return 1;
  }

  if (g_sgbsvx_isolation_a_calls != 1 || g_sgbsvx_isolation_b_calls != 1) {
    fprintf(stderr,
            "[FAIL] SGBSV active-vtable isolation call mismatch (A=%d B=%d)\n",
            g_sgbsvx_isolation_a_calls, g_sgbsvx_isolation_b_calls);
    return 1;
  }

  printf("[PASS] SGBSV adapter source isolation follows active vtable selection\n");
  return 0;
}

static int test_adapter_active_vtable_source_isolation_dgbsv(void) {
  fb_backend_vtable_t vtable_a;
  fb_backend_vtable_t vtable_b;
  fb_dgbsv_cblas_fn dgbsv_a = NULL;
  fb_dgbsv_cblas_fn dgbsv_b = NULL;
  fb_status_t status;

  double ab[8] = {0};
  double b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable_a, 0, sizeof(vtable_a));
  memset(&vtable_b, 0, sizeof(vtable_b));
  g_dgbsvx_isolation_a_calls = 0;
  g_dgbsvx_isolation_b_calls = 0;

  vtable_a.ext_ops[FB_OP_DGBSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dgbsvx_isolation_a;
  vtable_b.ext_ops[FB_OP_DGBSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dgbsvx_isolation_b;

  status = fb_finalize_plugin_vtable(&vtable_a);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize dgbsv vtable_a status=%d\n", status);
    return 1;
  }
  status = fb_finalize_plugin_vtable(&vtable_b);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize dgbsv vtable_b status=%d\n", status);
    return 1;
  }

  dgbsv_a = (fb_dgbsv_cblas_fn)vtable_a.ext_ops[FB_OP_DGBSV][FB_CONV_CBLAS];
  dgbsv_b = (fb_dgbsv_cblas_fn)vtable_b.ext_ops[FB_OP_DGBSV][FB_CONV_CBLAS];
  if (!dgbsv_a || !dgbsv_b) {
    fprintf(stderr, "[FAIL] DGBSV adapter slot missing in isolation test\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_a);
  if (dgbsv_a(FB_LAYOUT_COL_MAJOR, 2, 1, 1, 1, ab, 4, ipiv, b, 2) != 3201) {
    fprintf(stderr,
            "[FAIL] DGBSV call for active vtable_a did not route to source A\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_b);
  if (dgbsv_b(FB_LAYOUT_COL_MAJOR, 2, 1, 1, 1, ab, 4, ipiv, b, 2) != 3202) {
    fprintf(stderr,
            "[FAIL] DGBSV call for active vtable_b did not route to source B\n");
    return 1;
  }

  if (g_dgbsvx_isolation_a_calls != 1 || g_dgbsvx_isolation_b_calls != 1) {
    fprintf(stderr,
            "[FAIL] DGBSV active-vtable isolation call mismatch (A=%d B=%d)\n",
            g_dgbsvx_isolation_a_calls, g_dgbsvx_isolation_b_calls);
    return 1;
  }

  printf("[PASS] DGBSV adapter source isolation follows active vtable selection\n");
  return 0;
}

static int test_adapter_active_vtable_source_isolation_sgtsv(void) {
  fb_backend_vtable_t vtable_a;
  fb_backend_vtable_t vtable_b;
  fb_sgtsv_cblas_fn sgtsv_a = NULL;
  fb_sgtsv_cblas_fn sgtsv_b = NULL;
  fb_status_t status;

  float dl[2] = {0};
  float d[2] = {0};
  float du[2] = {0};
  float b[2] = {0};

  memset(&vtable_a, 0, sizeof(vtable_a));
  memset(&vtable_b, 0, sizeof(vtable_b));
  g_sgtsvx_isolation_a_calls = 0;
  g_sgtsvx_isolation_b_calls = 0;

  vtable_a.ext_ops[FB_OP_SGTSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sgtsvx_isolation_a;
  vtable_b.ext_ops[FB_OP_SGTSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sgtsvx_isolation_b;

  status = fb_finalize_plugin_vtable(&vtable_a);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize sgtsv vtable_a status=%d\n", status);
    return 1;
  }
  status = fb_finalize_plugin_vtable(&vtable_b);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize sgtsv vtable_b status=%d\n", status);
    return 1;
  }

  sgtsv_a = (fb_sgtsv_cblas_fn)vtable_a.ext_ops[FB_OP_SGTSV][FB_CONV_CBLAS];
  sgtsv_b = (fb_sgtsv_cblas_fn)vtable_b.ext_ops[FB_OP_SGTSV][FB_CONV_CBLAS];
  if (!sgtsv_a || !sgtsv_b) {
    fprintf(stderr, "[FAIL] SGTSV adapter slot missing in isolation test\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_a);
  if (sgtsv_a(FB_LAYOUT_COL_MAJOR, 2, 1, dl, d, du, b, 2) != 3505) {
    fprintf(stderr,
            "[FAIL] SGTSV call for active vtable_a did not route to source A\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_b);
  if (sgtsv_b(FB_LAYOUT_COL_MAJOR, 2, 1, dl, d, du, b, 2) != 3506) {
    fprintf(stderr,
            "[FAIL] SGTSV call for active vtable_b did not route to source B\n");
    return 1;
  }

  if (g_sgtsvx_isolation_a_calls != 1 || g_sgtsvx_isolation_b_calls != 1) {
    fprintf(stderr,
            "[FAIL] SGTSV active-vtable isolation call mismatch (A=%d B=%d)\n",
            g_sgtsvx_isolation_a_calls, g_sgtsvx_isolation_b_calls);
    return 1;
  }

  printf("[PASS] SGTSV adapter source isolation follows active vtable selection\n");
  return 0;
}

static int test_adapter_active_vtable_source_isolation_dgtsv(void) {
  fb_backend_vtable_t vtable_a;
  fb_backend_vtable_t vtable_b;
  fb_dgtsv_cblas_fn dgtsv_a = NULL;
  fb_dgtsv_cblas_fn dgtsv_b = NULL;
  fb_status_t status;

  double dl[2] = {0};
  double d[2] = {0};
  double du[2] = {0};
  double b[2] = {0};

  memset(&vtable_a, 0, sizeof(vtable_a));
  memset(&vtable_b, 0, sizeof(vtable_b));
  g_dgtsvx_isolation_a_calls = 0;
  g_dgtsvx_isolation_b_calls = 0;

  vtable_a.ext_ops[FB_OP_DGTSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dgtsvx_isolation_a;
  vtable_b.ext_ops[FB_OP_DGTSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dgtsvx_isolation_b;

  status = fb_finalize_plugin_vtable(&vtable_a);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize dgtsv vtable_a status=%d\n", status);
    return 1;
  }
  status = fb_finalize_plugin_vtable(&vtable_b);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize dgtsv vtable_b status=%d\n", status);
    return 1;
  }

  dgtsv_a = (fb_dgtsv_cblas_fn)vtable_a.ext_ops[FB_OP_DGTSV][FB_CONV_CBLAS];
  dgtsv_b = (fb_dgtsv_cblas_fn)vtable_b.ext_ops[FB_OP_DGTSV][FB_CONV_CBLAS];
  if (!dgtsv_a || !dgtsv_b) {
    fprintf(stderr, "[FAIL] DGTSV adapter slot missing in isolation test\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_a);
  if (dgtsv_a(FB_LAYOUT_COL_MAJOR, 2, 1, dl, d, du, b, 2) != 3507) {
    fprintf(stderr,
            "[FAIL] DGTSV call for active vtable_a did not route to source A\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_b);
  if (dgtsv_b(FB_LAYOUT_COL_MAJOR, 2, 1, dl, d, du, b, 2) != 3508) {
    fprintf(stderr,
            "[FAIL] DGTSV call for active vtable_b did not route to source B\n");
    return 1;
  }

  if (g_dgtsvx_isolation_a_calls != 1 || g_dgtsvx_isolation_b_calls != 1) {
    fprintf(stderr,
            "[FAIL] DGTSV active-vtable isolation call mismatch (A=%d B=%d)\n",
            g_dgtsvx_isolation_a_calls, g_dgtsvx_isolation_b_calls);
    return 1;
  }

  printf("[PASS] DGTSV adapter source isolation follows active vtable selection\n");
  return 0;
}

static int test_adapter_active_vtable_source_isolation_cgtsv(void) {
  fb_backend_vtable_t vtable_a;
  fb_backend_vtable_t vtable_b;
  fb_cgtsv_cblas_fn cgtsv_a = NULL;
  fb_cgtsv_cblas_fn cgtsv_b = NULL;
  fb_status_t status;

  fb_complex_float_t dl[2] = {0};
  fb_complex_float_t d[2] = {0};
  fb_complex_float_t du[2] = {0};
  fb_complex_float_t b[2] = {0};

  memset(&vtable_a, 0, sizeof(vtable_a));
  memset(&vtable_b, 0, sizeof(vtable_b));
  g_cgtsvx_isolation_a_calls = 0;
  g_cgtsvx_isolation_b_calls = 0;

  vtable_a.ext_ops[FB_OP_CGTSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_cgtsvx_isolation_a;
  vtable_b.ext_ops[FB_OP_CGTSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_cgtsvx_isolation_b;

  status = fb_finalize_plugin_vtable(&vtable_a);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize cgtsv vtable_a status=%d\n", status);
    return 1;
  }
  status = fb_finalize_plugin_vtable(&vtable_b);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize cgtsv vtable_b status=%d\n", status);
    return 1;
  }

  cgtsv_a = (fb_cgtsv_cblas_fn)vtable_a.ext_ops[FB_OP_CGTSV][FB_CONV_CBLAS];
  cgtsv_b = (fb_cgtsv_cblas_fn)vtable_b.ext_ops[FB_OP_CGTSV][FB_CONV_CBLAS];
  if (!cgtsv_a || !cgtsv_b) {
    fprintf(stderr, "[FAIL] CGTSV adapter slot missing in isolation test\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_a);
  if (cgtsv_a(FB_LAYOUT_COL_MAJOR, 2, 1, dl, d, du, b, 2) != 3509) {
    fprintf(stderr,
            "[FAIL] CGTSV call for active vtable_a did not route to source A\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_b);
  if (cgtsv_b(FB_LAYOUT_COL_MAJOR, 2, 1, dl, d, du, b, 2) != 3510) {
    fprintf(stderr,
            "[FAIL] CGTSV call for active vtable_b did not route to source B\n");
    return 1;
  }

  if (g_cgtsvx_isolation_a_calls != 1 || g_cgtsvx_isolation_b_calls != 1) {
    fprintf(stderr,
            "[FAIL] CGTSV active-vtable isolation call mismatch (A=%d B=%d)\n",
            g_cgtsvx_isolation_a_calls, g_cgtsvx_isolation_b_calls);
    return 1;
  }

  printf("[PASS] CGTSV adapter source isolation follows active vtable selection\n");
  return 0;
}

static int test_adapter_active_vtable_source_isolation_zgtsv(void) {
  fb_backend_vtable_t vtable_a;
  fb_backend_vtable_t vtable_b;
  fb_zgtsv_cblas_fn zgtsv_a = NULL;
  fb_zgtsv_cblas_fn zgtsv_b = NULL;
  fb_status_t status;

  fb_complex_double_t dl[2] = {0};
  fb_complex_double_t d[2] = {0};
  fb_complex_double_t du[2] = {0};
  fb_complex_double_t b[2] = {0};

  memset(&vtable_a, 0, sizeof(vtable_a));
  memset(&vtable_b, 0, sizeof(vtable_b));
  g_zgtsvx_isolation_a_calls = 0;
  g_zgtsvx_isolation_b_calls = 0;

  vtable_a.ext_ops[FB_OP_ZGTSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zgtsvx_isolation_a;
  vtable_b.ext_ops[FB_OP_ZGTSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zgtsvx_isolation_b;

  status = fb_finalize_plugin_vtable(&vtable_a);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize zgtsv vtable_a status=%d\n", status);
    return 1;
  }
  status = fb_finalize_plugin_vtable(&vtable_b);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize zgtsv vtable_b status=%d\n", status);
    return 1;
  }

  zgtsv_a = (fb_zgtsv_cblas_fn)vtable_a.ext_ops[FB_OP_ZGTSV][FB_CONV_CBLAS];
  zgtsv_b = (fb_zgtsv_cblas_fn)vtable_b.ext_ops[FB_OP_ZGTSV][FB_CONV_CBLAS];
  if (!zgtsv_a || !zgtsv_b) {
    fprintf(stderr, "[FAIL] ZGTSV adapter slot missing in isolation test\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_a);
  if (zgtsv_a(FB_LAYOUT_COL_MAJOR, 2, 1, dl, d, du, b, 2) != 3511) {
    fprintf(stderr,
            "[FAIL] ZGTSV call for active vtable_a did not route to source A\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_b);
  if (zgtsv_b(FB_LAYOUT_COL_MAJOR, 2, 1, dl, d, du, b, 2) != 3512) {
    fprintf(stderr,
            "[FAIL] ZGTSV call for active vtable_b did not route to source B\n");
    return 1;
  }

  if (g_zgtsvx_isolation_a_calls != 1 || g_zgtsvx_isolation_b_calls != 1) {
    fprintf(stderr,
            "[FAIL] ZGTSV active-vtable isolation call mismatch (A=%d B=%d)\n",
            g_zgtsvx_isolation_a_calls, g_zgtsvx_isolation_b_calls);
    return 1;
  }

  printf("[PASS] ZGTSV adapter source isolation follows active vtable selection\n");
  return 0;
}

static int test_adapter_active_vtable_source_isolation_sptsv(void) {
  fb_backend_vtable_t vtable_a;
  fb_backend_vtable_t vtable_b;
  fb_sptsv_cblas_fn sptsv_a = NULL;
  fb_sptsv_cblas_fn sptsv_b = NULL;
  fb_status_t status;

  float d[2] = {0};
  float e[2] = {0};
  float b[2] = {0};

  memset(&vtable_a, 0, sizeof(vtable_a));
  memset(&vtable_b, 0, sizeof(vtable_b));
  g_sptsvx_isolation_a_calls = 0;
  g_sptsvx_isolation_b_calls = 0;

  vtable_a.ext_ops[FB_OP_SPTSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sptsvx_isolation_a;
  vtable_b.ext_ops[FB_OP_SPTSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sptsvx_isolation_b;

  status = fb_finalize_plugin_vtable(&vtable_a);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize sptsv vtable_a status=%d\n", status);
    return 1;
  }
  status = fb_finalize_plugin_vtable(&vtable_b);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize sptsv vtable_b status=%d\n", status);
    return 1;
  }

  sptsv_a = (fb_sptsv_cblas_fn)vtable_a.ext_ops[FB_OP_SPTSV][FB_CONV_CBLAS];
  sptsv_b = (fb_sptsv_cblas_fn)vtable_b.ext_ops[FB_OP_SPTSV][FB_CONV_CBLAS];
  if (!sptsv_a || !sptsv_b) {
    fprintf(stderr, "[FAIL] SPTSV adapter slot missing in isolation test\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_a);
  if (sptsv_a(FB_LAYOUT_COL_MAJOR, 2, 1, d, e, b, 2) != 3601) {
    fprintf(stderr,
            "[FAIL] SPTSV call for active vtable_a did not route to source A\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_b);
  if (sptsv_b(FB_LAYOUT_COL_MAJOR, 2, 1, d, e, b, 2) != 3602) {
    fprintf(stderr,
            "[FAIL] SPTSV call for active vtable_b did not route to source B\n");
    return 1;
  }

  if (g_sptsvx_isolation_a_calls != 1 || g_sptsvx_isolation_b_calls != 1) {
    fprintf(stderr,
            "[FAIL] SPTSV active-vtable isolation call mismatch (A=%d B=%d)\n",
            g_sptsvx_isolation_a_calls, g_sptsvx_isolation_b_calls);
    return 1;
  }

  printf("[PASS] SPTSV adapter source isolation follows active vtable selection\n");
  return 0;
}

static int test_adapter_active_vtable_source_isolation_dptsv(void) {
  fb_backend_vtable_t vtable_a;
  fb_backend_vtable_t vtable_b;
  fb_dptsv_cblas_fn dptsv_a = NULL;
  fb_dptsv_cblas_fn dptsv_b = NULL;
  fb_status_t status;

  double d[2] = {0};
  double e[2] = {0};
  double b[2] = {0};

  memset(&vtable_a, 0, sizeof(vtable_a));
  memset(&vtable_b, 0, sizeof(vtable_b));
  g_dptsvx_isolation_a_calls = 0;
  g_dptsvx_isolation_b_calls = 0;

  vtable_a.ext_ops[FB_OP_DPTSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dptsvx_isolation_a;
  vtable_b.ext_ops[FB_OP_DPTSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dptsvx_isolation_b;

  status = fb_finalize_plugin_vtable(&vtable_a);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize dptsv vtable_a status=%d\n", status);
    return 1;
  }
  status = fb_finalize_plugin_vtable(&vtable_b);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize dptsv vtable_b status=%d\n", status);
    return 1;
  }

  dptsv_a = (fb_dptsv_cblas_fn)vtable_a.ext_ops[FB_OP_DPTSV][FB_CONV_CBLAS];
  dptsv_b = (fb_dptsv_cblas_fn)vtable_b.ext_ops[FB_OP_DPTSV][FB_CONV_CBLAS];
  if (!dptsv_a || !dptsv_b) {
    fprintf(stderr, "[FAIL] DPTSV adapter slot missing in isolation test\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_a);
  if (dptsv_a(FB_LAYOUT_COL_MAJOR, 2, 1, d, e, b, 2) != 3603) {
    fprintf(stderr,
            "[FAIL] DPTSV call for active vtable_a did not route to source A\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_b);
  if (dptsv_b(FB_LAYOUT_COL_MAJOR, 2, 1, d, e, b, 2) != 3604) {
    fprintf(stderr,
            "[FAIL] DPTSV call for active vtable_b did not route to source B\n");
    return 1;
  }

  if (g_dptsvx_isolation_a_calls != 1 || g_dptsvx_isolation_b_calls != 1) {
    fprintf(stderr,
            "[FAIL] DPTSV active-vtable isolation call mismatch (A=%d B=%d)\n",
            g_dptsvx_isolation_a_calls, g_dptsvx_isolation_b_calls);
    return 1;
  }

  printf("[PASS] DPTSV adapter source isolation follows active vtable selection\n");
  return 0;
}

static int test_adapter_active_vtable_source_isolation_cptsv(void) {
  fb_backend_vtable_t vtable_a;
  fb_backend_vtable_t vtable_b;
  fb_cptsv_cblas_fn cptsv_a = NULL;
  fb_cptsv_cblas_fn cptsv_b = NULL;
  fb_status_t status;

  float d[2] = {0};
  fb_complex_float_t e[2] = {0};
  fb_complex_float_t b[2] = {0};

  memset(&vtable_a, 0, sizeof(vtable_a));
  memset(&vtable_b, 0, sizeof(vtable_b));
  g_cptsvx_isolation_a_calls = 0;
  g_cptsvx_isolation_b_calls = 0;

  vtable_a.ext_ops[FB_OP_CPTSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_cptsvx_isolation_a;
  vtable_b.ext_ops[FB_OP_CPTSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_cptsvx_isolation_b;

  status = fb_finalize_plugin_vtable(&vtable_a);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize cptsv vtable_a status=%d\n", status);
    return 1;
  }
  status = fb_finalize_plugin_vtable(&vtable_b);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize cptsv vtable_b status=%d\n", status);
    return 1;
  }

  cptsv_a = (fb_cptsv_cblas_fn)vtable_a.ext_ops[FB_OP_CPTSV][FB_CONV_CBLAS];
  cptsv_b = (fb_cptsv_cblas_fn)vtable_b.ext_ops[FB_OP_CPTSV][FB_CONV_CBLAS];
  if (!cptsv_a || !cptsv_b) {
    fprintf(stderr, "[FAIL] CPTSV adapter slot missing in isolation test\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_a);
  if (cptsv_a(FB_LAYOUT_COL_MAJOR, 2, 1, d, e, b, 2) != 3605) {
    fprintf(stderr,
            "[FAIL] CPTSV call for active vtable_a did not route to source A\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_b);
  if (cptsv_b(FB_LAYOUT_COL_MAJOR, 2, 1, d, e, b, 2) != 3606) {
    fprintf(stderr,
            "[FAIL] CPTSV call for active vtable_b did not route to source B\n");
    return 1;
  }

  if (g_cptsvx_isolation_a_calls != 1 || g_cptsvx_isolation_b_calls != 1) {
    fprintf(stderr,
            "[FAIL] CPTSV active-vtable isolation call mismatch (A=%d B=%d)\n",
            g_cptsvx_isolation_a_calls, g_cptsvx_isolation_b_calls);
    return 1;
  }

  printf("[PASS] CPTSV adapter source isolation follows active vtable selection\n");
  return 0;
}

static int test_adapter_active_vtable_source_isolation_zptsv(void) {
  fb_backend_vtable_t vtable_a;
  fb_backend_vtable_t vtable_b;
  fb_zptsv_cblas_fn zptsv_a = NULL;
  fb_zptsv_cblas_fn zptsv_b = NULL;
  fb_status_t status;

  double d[2] = {0};
  fb_complex_double_t e[2] = {0};
  fb_complex_double_t b[2] = {0};

  memset(&vtable_a, 0, sizeof(vtable_a));
  memset(&vtable_b, 0, sizeof(vtable_b));
  g_zptsvx_isolation_a_calls = 0;
  g_zptsvx_isolation_b_calls = 0;

  vtable_a.ext_ops[FB_OP_ZPTSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zptsvx_isolation_a;
  vtable_b.ext_ops[FB_OP_ZPTSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zptsvx_isolation_b;

  status = fb_finalize_plugin_vtable(&vtable_a);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize zptsv vtable_a status=%d\n", status);
    return 1;
  }
  status = fb_finalize_plugin_vtable(&vtable_b);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize zptsv vtable_b status=%d\n", status);
    return 1;
  }

  zptsv_a = (fb_zptsv_cblas_fn)vtable_a.ext_ops[FB_OP_ZPTSV][FB_CONV_CBLAS];
  zptsv_b = (fb_zptsv_cblas_fn)vtable_b.ext_ops[FB_OP_ZPTSV][FB_CONV_CBLAS];
  if (!zptsv_a || !zptsv_b) {
    fprintf(stderr, "[FAIL] ZPTSV adapter slot missing in isolation test\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_a);
  if (zptsv_a(FB_LAYOUT_COL_MAJOR, 2, 1, d, e, b, 2) != 3607) {
    fprintf(stderr,
            "[FAIL] ZPTSV call for active vtable_a did not route to source A\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_b);
  if (zptsv_b(FB_LAYOUT_COL_MAJOR, 2, 1, d, e, b, 2) != 3608) {
    fprintf(stderr,
            "[FAIL] ZPTSV call for active vtable_b did not route to source B\n");
    return 1;
  }

  if (g_zptsvx_isolation_a_calls != 1 || g_zptsvx_isolation_b_calls != 1) {
    fprintf(stderr,
            "[FAIL] ZPTSV active-vtable isolation call mismatch (A=%d B=%d)\n",
            g_zptsvx_isolation_a_calls, g_zptsvx_isolation_b_calls);
    return 1;
  }

  printf("[PASS] ZPTSV adapter source isolation follows active vtable selection\n");
  return 0;
}

static int test_adapter_active_vtable_source_isolation_spbsv(void) {
  fb_backend_vtable_t vtable_a;
  fb_backend_vtable_t vtable_b;
  fb_spbsv_cblas_fn spbsv_a = NULL;
  fb_spbsv_cblas_fn spbsv_b = NULL;
  fb_status_t status;
  float ab[4] = {0};
  float b[2] = {0};

  memset(&vtable_a, 0, sizeof(vtable_a));
  memset(&vtable_b, 0, sizeof(vtable_b));
  g_spbsvx_isolation_a_calls = 0;
  g_spbsvx_isolation_b_calls = 0;

  vtable_a.ext_ops[FB_OP_SPBSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_spbsvx_isolation_a;
  vtable_b.ext_ops[FB_OP_SPBSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_spbsvx_isolation_b;

  status = fb_finalize_plugin_vtable(&vtable_a);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize spbsv vtable_a status=%d\n", status);
    return 1;
  }
  status = fb_finalize_plugin_vtable(&vtable_b);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize spbsv vtable_b status=%d\n", status);
    return 1;
  }

  spbsv_a = (fb_spbsv_cblas_fn)vtable_a.ext_ops[FB_OP_SPBSV][FB_CONV_CBLAS];
  spbsv_b = (fb_spbsv_cblas_fn)vtable_b.ext_ops[FB_OP_SPBSV][FB_CONV_CBLAS];
  if (!spbsv_a || !spbsv_b) {
    fprintf(stderr, "[FAIL] SPBSV adapter slot missing in isolation test\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_a);
  if (spbsv_a(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, 1, ab, 2, b, 2) !=
      3701) {
    fprintf(stderr,
            "[FAIL] SPBSV call for active vtable_a did not route to source A\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_b);
  if (spbsv_b(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, 1, ab, 2, b, 2) !=
      3702) {
    fprintf(stderr,
            "[FAIL] SPBSV call for active vtable_b did not route to source B\n");
    return 1;
  }

  if (g_spbsvx_isolation_a_calls != 1 || g_spbsvx_isolation_b_calls != 1) {
    fprintf(stderr,
            "[FAIL] SPBSV active-vtable isolation call mismatch (A=%d B=%d)\n",
            g_spbsvx_isolation_a_calls, g_spbsvx_isolation_b_calls);
    return 1;
  }

  printf("[PASS] SPBSV adapter source isolation follows active vtable selection\n");
  return 0;
}

static int test_adapter_active_vtable_source_isolation_dpbsv(void) {
  fb_backend_vtable_t vtable_a;
  fb_backend_vtable_t vtable_b;
  fb_dpbsv_cblas_fn dpbsv_a = NULL;
  fb_dpbsv_cblas_fn dpbsv_b = NULL;
  fb_status_t status;
  double ab[4] = {0};
  double b[2] = {0};

  memset(&vtable_a, 0, sizeof(vtable_a));
  memset(&vtable_b, 0, sizeof(vtable_b));
  g_dpbsvx_isolation_a_calls = 0;
  g_dpbsvx_isolation_b_calls = 0;

  vtable_a.ext_ops[FB_OP_DPBSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dpbsvx_isolation_a;
  vtable_b.ext_ops[FB_OP_DPBSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dpbsvx_isolation_b;

  status = fb_finalize_plugin_vtable(&vtable_a);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize dpbsv vtable_a status=%d\n", status);
    return 1;
  }
  status = fb_finalize_plugin_vtable(&vtable_b);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize dpbsv vtable_b status=%d\n", status);
    return 1;
  }

  dpbsv_a = (fb_dpbsv_cblas_fn)vtable_a.ext_ops[FB_OP_DPBSV][FB_CONV_CBLAS];
  dpbsv_b = (fb_dpbsv_cblas_fn)vtable_b.ext_ops[FB_OP_DPBSV][FB_CONV_CBLAS];
  if (!dpbsv_a || !dpbsv_b) {
    fprintf(stderr, "[FAIL] DPBSV adapter slot missing in isolation test\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_a);
  if (dpbsv_a(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, 1, ab, 2, b, 2) !=
      3703) {
    fprintf(stderr,
            "[FAIL] DPBSV call for active vtable_a did not route to source A\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_b);
  if (dpbsv_b(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, 1, ab, 2, b, 2) !=
      3704) {
    fprintf(stderr,
            "[FAIL] DPBSV call for active vtable_b did not route to source B\n");
    return 1;
  }

  if (g_dpbsvx_isolation_a_calls != 1 || g_dpbsvx_isolation_b_calls != 1) {
    fprintf(stderr,
            "[FAIL] DPBSV active-vtable isolation call mismatch (A=%d B=%d)\n",
            g_dpbsvx_isolation_a_calls, g_dpbsvx_isolation_b_calls);
    return 1;
  }

  printf("[PASS] DPBSV adapter source isolation follows active vtable selection\n");
  return 0;
}

static int test_adapter_active_vtable_source_isolation_cpbsv(void) {
  fb_backend_vtable_t vtable_a;
  fb_backend_vtable_t vtable_b;
  fb_cpbsv_cblas_fn cpbsv_a = NULL;
  fb_cpbsv_cblas_fn cpbsv_b = NULL;
  fb_status_t status;
  fb_complex_float_t ab[4] = {0};
  fb_complex_float_t b[2] = {0};

  memset(&vtable_a, 0, sizeof(vtable_a));
  memset(&vtable_b, 0, sizeof(vtable_b));
  g_cpbsvx_isolation_a_calls = 0;
  g_cpbsvx_isolation_b_calls = 0;

  vtable_a.ext_ops[FB_OP_CPBSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_cpbsvx_isolation_a;
  vtable_b.ext_ops[FB_OP_CPBSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_cpbsvx_isolation_b;

  status = fb_finalize_plugin_vtable(&vtable_a);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize cpbsv vtable_a status=%d\n", status);
    return 1;
  }
  status = fb_finalize_plugin_vtable(&vtable_b);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize cpbsv vtable_b status=%d\n", status);
    return 1;
  }

  cpbsv_a = (fb_cpbsv_cblas_fn)vtable_a.ext_ops[FB_OP_CPBSV][FB_CONV_CBLAS];
  cpbsv_b = (fb_cpbsv_cblas_fn)vtable_b.ext_ops[FB_OP_CPBSV][FB_CONV_CBLAS];
  if (!cpbsv_a || !cpbsv_b) {
    fprintf(stderr, "[FAIL] CPBSV adapter slot missing in isolation test\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_a);
  if (cpbsv_a(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, 1, ab, 2, b, 2) !=
      3705) {
    fprintf(stderr,
            "[FAIL] CPBSV call for active vtable_a did not route to source A\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_b);
  if (cpbsv_b(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, 1, ab, 2, b, 2) !=
      3706) {
    fprintf(stderr,
            "[FAIL] CPBSV call for active vtable_b did not route to source B\n");
    return 1;
  }

  if (g_cpbsvx_isolation_a_calls != 1 || g_cpbsvx_isolation_b_calls != 1) {
    fprintf(stderr,
            "[FAIL] CPBSV active-vtable isolation call mismatch (A=%d B=%d)\n",
            g_cpbsvx_isolation_a_calls, g_cpbsvx_isolation_b_calls);
    return 1;
  }

  printf("[PASS] CPBSV adapter source isolation follows active vtable selection\n");
  return 0;
}

static int test_adapter_active_vtable_source_isolation_zpbsv(void) {
  fb_backend_vtable_t vtable_a;
  fb_backend_vtable_t vtable_b;
  fb_zpbsv_cblas_fn zpbsv_a = NULL;
  fb_zpbsv_cblas_fn zpbsv_b = NULL;
  fb_status_t status;
  fb_complex_double_t ab[4] = {0};
  fb_complex_double_t b[2] = {0};

  memset(&vtable_a, 0, sizeof(vtable_a));
  memset(&vtable_b, 0, sizeof(vtable_b));
  g_zpbsvx_isolation_a_calls = 0;
  g_zpbsvx_isolation_b_calls = 0;

  vtable_a.ext_ops[FB_OP_ZPBSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zpbsvx_isolation_a;
  vtable_b.ext_ops[FB_OP_ZPBSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zpbsvx_isolation_b;

  status = fb_finalize_plugin_vtable(&vtable_a);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize zpbsv vtable_a status=%d\n", status);
    return 1;
  }
  status = fb_finalize_plugin_vtable(&vtable_b);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize zpbsv vtable_b status=%d\n", status);
    return 1;
  }

  zpbsv_a = (fb_zpbsv_cblas_fn)vtable_a.ext_ops[FB_OP_ZPBSV][FB_CONV_CBLAS];
  zpbsv_b = (fb_zpbsv_cblas_fn)vtable_b.ext_ops[FB_OP_ZPBSV][FB_CONV_CBLAS];
  if (!zpbsv_a || !zpbsv_b) {
    fprintf(stderr, "[FAIL] ZPBSV adapter slot missing in isolation test\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_a);
  if (zpbsv_a(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, 1, ab, 2, b, 2) !=
      3707) {
    fprintf(stderr,
            "[FAIL] ZPBSV call for active vtable_a did not route to source A\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_b);
  if (zpbsv_b(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, 1, ab, 2, b, 2) !=
      3708) {
    fprintf(stderr,
            "[FAIL] ZPBSV call for active vtable_b did not route to source B\n");
    return 1;
  }

  if (g_zpbsvx_isolation_a_calls != 1 || g_zpbsvx_isolation_b_calls != 1) {
    fprintf(stderr,
            "[FAIL] ZPBSV active-vtable isolation call mismatch (A=%d B=%d)\n",
            g_zpbsvx_isolation_a_calls, g_zpbsvx_isolation_b_calls);
    return 1;
  }

  printf("[PASS] ZPBSV adapter source isolation follows active vtable selection\n");
  return 0;
}

static int test_adapter_active_vtable_source_isolation_cgbsv(void) {
  fb_backend_vtable_t vtable_a;
  fb_backend_vtable_t vtable_b;
  fb_cgbsv_cblas_fn cgbsv_a = NULL;
  fb_cgbsv_cblas_fn cgbsv_b = NULL;
  fb_status_t status;

  fb_complex_float_t ab[8] = {0};
  fb_complex_float_t b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable_a, 0, sizeof(vtable_a));
  memset(&vtable_b, 0, sizeof(vtable_b));
  g_cgbsvx_isolation_a_calls = 0;
  g_cgbsvx_isolation_b_calls = 0;

  vtable_a.ext_ops[FB_OP_CGBSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_cgbsvx_isolation_a;
  vtable_b.ext_ops[FB_OP_CGBSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_cgbsvx_isolation_b;

  status = fb_finalize_plugin_vtable(&vtable_a);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize cgbsv vtable_a status=%d\n", status);
    return 1;
  }
  status = fb_finalize_plugin_vtable(&vtable_b);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize cgbsv vtable_b status=%d\n", status);
    return 1;
  }

  cgbsv_a = (fb_cgbsv_cblas_fn)vtable_a.ext_ops[FB_OP_CGBSV][FB_CONV_CBLAS];
  cgbsv_b = (fb_cgbsv_cblas_fn)vtable_b.ext_ops[FB_OP_CGBSV][FB_CONV_CBLAS];
  if (!cgbsv_a || !cgbsv_b) {
    fprintf(stderr, "[FAIL] CGBSV adapter slot missing in isolation test\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_a);
  if (cgbsv_a(FB_LAYOUT_COL_MAJOR, 2, 1, 1, 1, ab, 4, ipiv, b, 2) != 3301) {
    fprintf(stderr,
            "[FAIL] CGBSV call for active vtable_a did not route to source A\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_b);
  if (cgbsv_b(FB_LAYOUT_COL_MAJOR, 2, 1, 1, 1, ab, 4, ipiv, b, 2) != 3302) {
    fprintf(stderr,
            "[FAIL] CGBSV call for active vtable_b did not route to source B\n");
    return 1;
  }

  if (g_cgbsvx_isolation_a_calls != 1 || g_cgbsvx_isolation_b_calls != 1) {
    fprintf(stderr,
            "[FAIL] CGBSV active-vtable isolation call mismatch (A=%d B=%d)\n",
            g_cgbsvx_isolation_a_calls, g_cgbsvx_isolation_b_calls);
    return 1;
  }

  printf("[PASS] CGBSV adapter source isolation follows active vtable selection\n");
  return 0;
}

static int test_adapter_active_vtable_source_isolation_zgbsv(void) {
  fb_backend_vtable_t vtable_a;
  fb_backend_vtable_t vtable_b;
  fb_zgbsv_cblas_fn zgbsv_a = NULL;
  fb_zgbsv_cblas_fn zgbsv_b = NULL;
  fb_status_t status;

  fb_complex_double_t ab[8] = {0};
  fb_complex_double_t b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable_a, 0, sizeof(vtable_a));
  memset(&vtable_b, 0, sizeof(vtable_b));
  g_zgbsvx_isolation_a_calls = 0;
  g_zgbsvx_isolation_b_calls = 0;

  vtable_a.ext_ops[FB_OP_ZGBSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zgbsvx_isolation_a;
  vtable_b.ext_ops[FB_OP_ZGBSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zgbsvx_isolation_b;

  status = fb_finalize_plugin_vtable(&vtable_a);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize zgbsv vtable_a status=%d\n", status);
    return 1;
  }
  status = fb_finalize_plugin_vtable(&vtable_b);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize zgbsv vtable_b status=%d\n", status);
    return 1;
  }

  zgbsv_a = (fb_zgbsv_cblas_fn)vtable_a.ext_ops[FB_OP_ZGBSV][FB_CONV_CBLAS];
  zgbsv_b = (fb_zgbsv_cblas_fn)vtable_b.ext_ops[FB_OP_ZGBSV][FB_CONV_CBLAS];
  if (!zgbsv_a || !zgbsv_b) {
    fprintf(stderr, "[FAIL] ZGBSV adapter slot missing in isolation test\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_a);
  if (zgbsv_a(FB_LAYOUT_COL_MAJOR, 2, 1, 1, 1, ab, 4, ipiv, b, 2) != 3401) {
    fprintf(stderr,
            "[FAIL] ZGBSV call for active vtable_a did not route to source A\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_b);
  if (zgbsv_b(FB_LAYOUT_COL_MAJOR, 2, 1, 1, 1, ab, 4, ipiv, b, 2) != 3402) {
    fprintf(stderr,
            "[FAIL] ZGBSV call for active vtable_b did not route to source B\n");
    return 1;
  }

  if (g_zgbsvx_isolation_a_calls != 1 || g_zgbsvx_isolation_b_calls != 1) {
    fprintf(stderr,
            "[FAIL] ZGBSV active-vtable isolation call mismatch (A=%d B=%d)\n",
            g_zgbsvx_isolation_a_calls, g_zgbsvx_isolation_b_calls);
    return 1;
  }

  printf("[PASS] ZGBSV adapter source isolation follows active vtable selection\n");
  return 0;
}

static int test_adapter_active_vtable_source_isolation_sppsv(void) {
  fb_backend_vtable_t vtable_a;
  fb_backend_vtable_t vtable_b;
  fb_sppsv_cblas_fn sppsv_a = NULL;
  fb_sppsv_cblas_fn sppsv_b = NULL;
  fb_status_t status;
  float ap[3] = {0};
  float b[2] = {0};

  memset(&vtable_a, 0, sizeof(vtable_a));
  memset(&vtable_b, 0, sizeof(vtable_b));
  g_sppsvx_isolation_a_calls = 0;
  g_sppsvx_isolation_b_calls = 0;

  vtable_a.ext_ops[FB_OP_SPPSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sppsvx_isolation_a;
  vtable_b.ext_ops[FB_OP_SPPSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sppsvx_isolation_b;

  status = fb_finalize_plugin_vtable(&vtable_a);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize sppsv vtable_a status=%d\n", status);
    return 1;
  }
  status = fb_finalize_plugin_vtable(&vtable_b);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize sppsv vtable_b status=%d\n", status);
    return 1;
  }

  sppsv_a = (fb_sppsv_cblas_fn)vtable_a.ext_ops[FB_OP_SPPSV][FB_CONV_CBLAS];
  sppsv_b = (fb_sppsv_cblas_fn)vtable_b.ext_ops[FB_OP_SPPSV][FB_CONV_CBLAS];
  if (!sppsv_a || !sppsv_b) {
    fprintf(stderr, "[FAIL] SPPSV adapter slot missing in isolation test\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_a);
  if (sppsv_a(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, ap, b, 2) != 3801) {
    fprintf(stderr,
            "[FAIL] SPPSV call for active vtable_a did not route to source A\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_b);
  if (sppsv_b(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, ap, b, 2) != 3802) {
    fprintf(stderr,
            "[FAIL] SPPSV call for active vtable_b did not route to source B\n");
    return 1;
  }

  if (g_sppsvx_isolation_a_calls != 1 || g_sppsvx_isolation_b_calls != 1) {
    fprintf(stderr,
            "[FAIL] SPPSV active-vtable isolation call mismatch (A=%d B=%d)\n",
            g_sppsvx_isolation_a_calls, g_sppsvx_isolation_b_calls);
    return 1;
  }

  printf("[PASS] SPPSV adapter source isolation follows active vtable selection\n");
  return 0;
}

static int test_adapter_active_vtable_source_isolation_dppsv(void) {
  fb_backend_vtable_t vtable_a;
  fb_backend_vtable_t vtable_b;
  fb_dppsv_cblas_fn dppsv_a = NULL;
  fb_dppsv_cblas_fn dppsv_b = NULL;
  fb_status_t status;
  double ap[3] = {0};
  double b[2] = {0};

  memset(&vtable_a, 0, sizeof(vtable_a));
  memset(&vtable_b, 0, sizeof(vtable_b));
  g_dppsvx_isolation_a_calls = 0;
  g_dppsvx_isolation_b_calls = 0;

  vtable_a.ext_ops[FB_OP_DPPSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dppsvx_isolation_a;
  vtable_b.ext_ops[FB_OP_DPPSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dppsvx_isolation_b;

  status = fb_finalize_plugin_vtable(&vtable_a);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize dppsv vtable_a status=%d\n", status);
    return 1;
  }
  status = fb_finalize_plugin_vtable(&vtable_b);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize dppsv vtable_b status=%d\n", status);
    return 1;
  }

  dppsv_a = (fb_dppsv_cblas_fn)vtable_a.ext_ops[FB_OP_DPPSV][FB_CONV_CBLAS];
  dppsv_b = (fb_dppsv_cblas_fn)vtable_b.ext_ops[FB_OP_DPPSV][FB_CONV_CBLAS];
  if (!dppsv_a || !dppsv_b) {
    fprintf(stderr, "[FAIL] DPPSV adapter slot missing in isolation test\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_a);
  if (dppsv_a(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, ap, b, 2) != 3803) {
    fprintf(stderr,
            "[FAIL] DPPSV call for active vtable_a did not route to source A\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_b);
  if (dppsv_b(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, ap, b, 2) != 3804) {
    fprintf(stderr,
            "[FAIL] DPPSV call for active vtable_b did not route to source B\n");
    return 1;
  }

  if (g_dppsvx_isolation_a_calls != 1 || g_dppsvx_isolation_b_calls != 1) {
    fprintf(stderr,
            "[FAIL] DPPSV active-vtable isolation call mismatch (A=%d B=%d)\n",
            g_dppsvx_isolation_a_calls, g_dppsvx_isolation_b_calls);
    return 1;
  }

  printf("[PASS] DPPSV adapter source isolation follows active vtable selection\n");
  return 0;
}

static int test_adapter_active_vtable_source_isolation_cppsv(void) {
  fb_backend_vtable_t vtable_a;
  fb_backend_vtable_t vtable_b;
  fb_cppsv_cblas_fn cppsv_a = NULL;
  fb_cppsv_cblas_fn cppsv_b = NULL;
  fb_status_t status;
  fb_complex_float_t ap[3] = {0};
  fb_complex_float_t b[2] = {0};

  memset(&vtable_a, 0, sizeof(vtable_a));
  memset(&vtable_b, 0, sizeof(vtable_b));
  g_cppsvx_isolation_a_calls = 0;
  g_cppsvx_isolation_b_calls = 0;

  vtable_a.ext_ops[FB_OP_CPPSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_cppsvx_isolation_a;
  vtable_b.ext_ops[FB_OP_CPPSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_cppsvx_isolation_b;

  status = fb_finalize_plugin_vtable(&vtable_a);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize cppsv vtable_a status=%d\n", status);
    return 1;
  }
  status = fb_finalize_plugin_vtable(&vtable_b);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize cppsv vtable_b status=%d\n", status);
    return 1;
  }

  cppsv_a = (fb_cppsv_cblas_fn)vtable_a.ext_ops[FB_OP_CPPSV][FB_CONV_CBLAS];
  cppsv_b = (fb_cppsv_cblas_fn)vtable_b.ext_ops[FB_OP_CPPSV][FB_CONV_CBLAS];
  if (!cppsv_a || !cppsv_b) {
    fprintf(stderr, "[FAIL] CPPSV adapter slot missing in isolation test\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_a);
  if (cppsv_a(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, ap, b, 2) != 3805) {
    fprintf(stderr,
            "[FAIL] CPPSV call for active vtable_a did not route to source A\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_b);
  if (cppsv_b(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, ap, b, 2) != 3806) {
    fprintf(stderr,
            "[FAIL] CPPSV call for active vtable_b did not route to source B\n");
    return 1;
  }

  if (g_cppsvx_isolation_a_calls != 1 || g_cppsvx_isolation_b_calls != 1) {
    fprintf(stderr,
            "[FAIL] CPPSV active-vtable isolation call mismatch (A=%d B=%d)\n",
            g_cppsvx_isolation_a_calls, g_cppsvx_isolation_b_calls);
    return 1;
  }

  printf("[PASS] CPPSV adapter source isolation follows active vtable selection\n");
  return 0;
}

static int test_adapter_active_vtable_source_isolation_zppsv(void) {
  fb_backend_vtable_t vtable_a;
  fb_backend_vtable_t vtable_b;
  fb_zppsv_cblas_fn zppsv_a = NULL;
  fb_zppsv_cblas_fn zppsv_b = NULL;
  fb_status_t status;
  fb_complex_double_t ap[3] = {0};
  fb_complex_double_t b[2] = {0};

  memset(&vtable_a, 0, sizeof(vtable_a));
  memset(&vtable_b, 0, sizeof(vtable_b));
  g_zppsvx_isolation_a_calls = 0;
  g_zppsvx_isolation_b_calls = 0;

  vtable_a.ext_ops[FB_OP_ZPPSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zppsvx_isolation_a;
  vtable_b.ext_ops[FB_OP_ZPPSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zppsvx_isolation_b;

  status = fb_finalize_plugin_vtable(&vtable_a);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize zppsv vtable_a status=%d\n", status);
    return 1;
  }
  status = fb_finalize_plugin_vtable(&vtable_b);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize zppsv vtable_b status=%d\n", status);
    return 1;
  }

  zppsv_a = (fb_zppsv_cblas_fn)vtable_a.ext_ops[FB_OP_ZPPSV][FB_CONV_CBLAS];
  zppsv_b = (fb_zppsv_cblas_fn)vtable_b.ext_ops[FB_OP_ZPPSV][FB_CONV_CBLAS];
  if (!zppsv_a || !zppsv_b) {
    fprintf(stderr, "[FAIL] ZPPSV adapter slot missing in isolation test\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_a);
  if (zppsv_a(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, ap, b, 2) != 3807) {
    fprintf(stderr,
            "[FAIL] ZPPSV call for active vtable_a did not route to source A\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_b);
  if (zppsv_b(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, ap, b, 2) != 3808) {
    fprintf(stderr,
            "[FAIL] ZPPSV call for active vtable_b did not route to source B\n");
    return 1;
  }

  if (g_zppsvx_isolation_a_calls != 1 || g_zppsvx_isolation_b_calls != 1) {
    fprintf(stderr,
            "[FAIL] ZPPSV active-vtable isolation call mismatch (A=%d B=%d)\n",
            g_zppsvx_isolation_a_calls, g_zppsvx_isolation_b_calls);
    return 1;
  }

  printf("[PASS] ZPPSV adapter source isolation follows active vtable selection\n");
  return 0;
}

static int test_adapter_active_vtable_source_isolation_sspsv(void) {
  fb_backend_vtable_t vtable_a;
  fb_backend_vtable_t vtable_b;
  fb_sspsv_cblas_fn sspsv_a = NULL;
  fb_sspsv_cblas_fn sspsv_b = NULL;
  fb_status_t status;
  float ap[3] = {0};
  float b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable_a, 0, sizeof(vtable_a));
  memset(&vtable_b, 0, sizeof(vtable_b));
  g_sspsvx_isolation_a_calls = 0;
  g_sspsvx_isolation_b_calls = 0;

  vtable_a.ext_ops[FB_OP_SSPSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sspsvx_isolation_a;
  vtable_b.ext_ops[FB_OP_SSPSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sspsvx_isolation_b;

  status = fb_finalize_plugin_vtable(&vtable_a);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize sspsv vtable_a status=%d\n", status);
    return 1;
  }
  status = fb_finalize_plugin_vtable(&vtable_b);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize sspsv vtable_b status=%d\n", status);
    return 1;
  }

  sspsv_a = (fb_sspsv_cblas_fn)vtable_a.ext_ops[FB_OP_SSPSV][FB_CONV_CBLAS];
  sspsv_b = (fb_sspsv_cblas_fn)vtable_b.ext_ops[FB_OP_SSPSV][FB_CONV_CBLAS];
  if (!sspsv_a || !sspsv_b) {
    fprintf(stderr, "[FAIL] SSPSV adapter slot missing in isolation test\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_a);
  if (sspsv_a(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, ap, ipiv, b, 2) !=
      3901) {
    fprintf(stderr,
            "[FAIL] SSPSV call for active vtable_a did not route to source A\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_b);
  if (sspsv_b(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, ap, ipiv, b, 2) !=
      3902) {
    fprintf(stderr,
            "[FAIL] SSPSV call for active vtable_b did not route to source B\n");
    return 1;
  }

  if (g_sspsvx_isolation_a_calls != 1 || g_sspsvx_isolation_b_calls != 1) {
    fprintf(stderr,
            "[FAIL] SSPSV active-vtable isolation call mismatch (A=%d B=%d)\n",
            g_sspsvx_isolation_a_calls, g_sspsvx_isolation_b_calls);
    return 1;
  }

  printf("[PASS] SSPSV adapter source isolation follows active vtable selection\n");
  return 0;
}

static int test_adapter_active_vtable_source_isolation_dspsv(void) {
  fb_backend_vtable_t vtable_a;
  fb_backend_vtable_t vtable_b;
  fb_dspsv_cblas_fn dspsv_a = NULL;
  fb_dspsv_cblas_fn dspsv_b = NULL;
  fb_status_t status;
  double ap[3] = {0};
  double b[2] = {0};
  int ipiv[2] = {0};

  memset(&vtable_a, 0, sizeof(vtable_a));
  memset(&vtable_b, 0, sizeof(vtable_b));
  g_dspsvx_isolation_a_calls = 0;
  g_dspsvx_isolation_b_calls = 0;

  vtable_a.ext_ops[FB_OP_DSPSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dspsvx_isolation_a;
  vtable_b.ext_ops[FB_OP_DSPSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dspsvx_isolation_b;

  status = fb_finalize_plugin_vtable(&vtable_a);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize dspsv vtable_a status=%d\n", status);
    return 1;
  }
  status = fb_finalize_plugin_vtable(&vtable_b);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize dspsv vtable_b status=%d\n", status);
    return 1;
  }

  dspsv_a = (fb_dspsv_cblas_fn)vtable_a.ext_ops[FB_OP_DSPSV][FB_CONV_CBLAS];
  dspsv_b = (fb_dspsv_cblas_fn)vtable_b.ext_ops[FB_OP_DSPSV][FB_CONV_CBLAS];
  if (!dspsv_a || !dspsv_b) {
    fprintf(stderr, "[FAIL] DSPSV adapter slot missing in isolation test\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_a);
  if (dspsv_a(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, ap, ipiv, b, 2) !=
      3903) {
    fprintf(stderr,
            "[FAIL] DSPSV call for active vtable_a did not route to source A\n");
    return 1;
  }

  fb_set_active_vtable(&vtable_b);
  if (dspsv_b(FB_LAYOUT_COL_MAJOR, (char)FB_UPPER, 2, 1, ap, ipiv, b, 2) !=
      3904) {
    fprintf(stderr,
            "[FAIL] DSPSV call for active vtable_b did not route to source B\n");
    return 1;
  }

  if (g_dspsvx_isolation_a_calls != 1 || g_dspsvx_isolation_b_calls != 1) {
    fprintf(stderr,
            "[FAIL] DSPSV active-vtable isolation call mismatch (A=%d B=%d)\n",
            g_dspsvx_isolation_a_calls, g_dspsvx_isolation_b_calls);
    return 1;
  }

  printf("[PASS] DSPSV adapter source isolation follows active vtable selection\n");
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
  status |= test_adapter_bridge_sgesv_from_sgesvxx();
  status |= test_adapter_bridge_dgesv_from_dgesvxx();
  status |= test_adapter_bridge_cgesv_from_cgesvxx();
  status |= test_adapter_bridge_zgesv_from_zgesvxx();
  status |= test_adapter_bridge_sposv_from_sposvxx();
  status |= test_adapter_bridge_dposv_from_dposvxx();
  status |= test_adapter_bridge_cposv_from_cposvxx();
  status |= test_adapter_bridge_zposv_from_zposvxx();
  status |= test_adapter_bridge_sposv_from_sposvx();
  status |= test_adapter_bridge_dposv_from_dposvx();
  status |= test_adapter_bridge_cposv_from_cposvx();
  status |= test_adapter_bridge_zposv_from_zposvx();
  status |= test_adapter_bridge_ssysv_from_ssysvx();
  status |= test_adapter_bridge_dsysv_from_dsysvx();
  status |= test_adapter_bridge_csysv_from_csysvx();
  status |= test_adapter_bridge_zsysv_from_zsysvx();
  status |= test_adapter_bridge_chesv_from_chesvx();
  status |= test_adapter_bridge_zhesv_from_zhesvx();
  status |= test_adapter_bridge_ssysv_from_ssysvxx();
  status |= test_adapter_bridge_dsysv_from_dsysvxx();
  status |= test_adapter_bridge_csysv_from_csysvxx();
  status |= test_adapter_bridge_zsysv_from_zsysvxx();
  status |= test_adapter_bridge_chesv_from_chesvxx();
  status |= test_adapter_bridge_zhesv_from_zhesvxx();
  status |= test_adapter_bridge_sgbsv_from_sgbsvx();
  status |= test_adapter_bridge_dgbsv_from_dgbsvx();
  status |= test_adapter_bridge_cgbsv_from_cgbsvx();
  status |= test_adapter_bridge_zgbsv_from_zgbsvx();
  status |= test_adapter_bridge_sgtsv_from_sgtsvx();
  status |= test_adapter_bridge_dgtsv_from_dgtsvx();
  status |= test_adapter_bridge_cgtsv_from_cgtsvx();
  status |= test_adapter_bridge_zgtsv_from_zgtsvx();
  status |= test_adapter_bridge_sptsv_from_sptsvx();
  status |= test_adapter_bridge_dptsv_from_dptsvx();
  status |= test_adapter_bridge_cptsv_from_cptsvx();
  status |= test_adapter_bridge_zptsv_from_zptsvx();
  status |= test_adapter_bridge_spbsv_from_spbsvx();
  status |= test_adapter_bridge_dpbsv_from_dpbsvx();
  status |= test_adapter_bridge_cpbsv_from_cpbsvx();
  status |= test_adapter_bridge_zpbsv_from_zpbsvx();
  status |= test_adapter_bridge_sppsv_from_sppsvx();
  status |= test_adapter_bridge_dppsv_from_dppsvx();
  status |= test_adapter_bridge_cppsv_from_cppsvx();
  status |= test_adapter_bridge_zppsv_from_zppsvx();
  status |= test_adapter_bridge_sspsv_from_sspsvx();
  status |= test_adapter_bridge_dspsv_from_dspsvx();
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
  status |= test_adapter_does_not_override_existing_sgesv_from_sgesvxx();
  status |= test_adapter_does_not_override_existing_dgesv_from_dgesvxx();
  status |= test_adapter_does_not_override_existing_cgesv_from_cgesvxx();
  status |= test_adapter_does_not_override_existing_zgesv_from_zgesvxx();
  status |= test_adapter_does_not_override_existing_sposv_from_sposvxx();
  status |= test_adapter_does_not_override_existing_dposv_from_dposvxx();
  status |= test_adapter_does_not_override_existing_cposv_from_cposvxx();
  status |= test_adapter_does_not_override_existing_zposv_from_zposvxx();
  status |= test_adapter_does_not_override_existing_sposv();
  status |= test_adapter_does_not_override_existing_dposv();
  status |= test_adapter_does_not_override_existing_cposv();
  status |= test_adapter_does_not_override_existing_zposv();
  status |= test_adapter_does_not_override_existing_ssysv();
  status |= test_adapter_does_not_override_existing_dsysv();
  status |= test_adapter_does_not_override_existing_csysv();
  status |= test_adapter_does_not_override_existing_zsysv();
  status |= test_adapter_does_not_override_existing_chesv_from_chesvx();
  status |= test_adapter_does_not_override_existing_zhesv_from_zhesvx();
  status |= test_adapter_does_not_override_existing_ssysv_from_ssysvxx();
  status |= test_adapter_does_not_override_existing_dsysv_from_dsysvxx();
  status |= test_adapter_does_not_override_existing_csysv_from_csysvxx();
  status |= test_adapter_does_not_override_existing_zsysv_from_zsysvxx();
  status |= test_adapter_does_not_override_existing_chesv_from_chesvxx();
  status |= test_adapter_does_not_override_existing_zhesv_from_zhesvxx();
  status |= test_adapter_does_not_override_existing_sgbsv();
  status |= test_adapter_does_not_override_existing_dgbsv();
  status |= test_adapter_does_not_override_existing_cgbsv();
  status |= test_adapter_does_not_override_existing_zgbsv();
  status |= test_adapter_does_not_override_existing_sgtsv();
  status |= test_adapter_does_not_override_existing_dgtsv();
  status |= test_adapter_does_not_override_existing_cgtsv();
  status |= test_adapter_does_not_override_existing_zgtsv();
  status |= test_adapter_does_not_override_existing_sptsv();
  status |= test_adapter_does_not_override_existing_dptsv();
  status |= test_adapter_does_not_override_existing_cptsv();
  status |= test_adapter_does_not_override_existing_zptsv();
  status |= test_adapter_does_not_override_existing_spbsv();
  status |= test_adapter_does_not_override_existing_dpbsv();
  status |= test_adapter_does_not_override_existing_cpbsv();
  status |= test_adapter_does_not_override_existing_zpbsv();
  status |= test_adapter_does_not_override_existing_sppsv();
  status |= test_adapter_does_not_override_existing_dppsv();
  status |= test_adapter_does_not_override_existing_cppsv();
  status |= test_adapter_does_not_override_existing_zppsv();
  status |= test_adapter_does_not_override_existing_sspsv();
  status |= test_adapter_does_not_override_existing_dspsv();
  status |= test_adapter_does_not_override_existing_ssyev();
  status |= test_adapter_does_not_override_existing_sstev();
  status |= test_adapter_finalize_idempotent_ssyev();
  status |= test_adapter_finalize_idempotent_sgesv();
  status |= test_adapter_finalize_idempotent_dgesv();
  status |= test_adapter_finalize_idempotent_cgesv();
  status |= test_adapter_finalize_idempotent_zgesv();
  status |= test_adapter_finalize_idempotent_sgesv_from_sgesvxx();
  status |= test_adapter_finalize_idempotent_dgesv_from_dgesvxx();
  status |= test_adapter_finalize_idempotent_cgesv_from_cgesvxx();
  status |= test_adapter_finalize_idempotent_zgesv_from_zgesvxx();
  status |= test_adapter_finalize_idempotent_sposv_from_sposvxx();
  status |= test_adapter_finalize_idempotent_dposv_from_dposvxx();
  status |= test_adapter_finalize_idempotent_cposv_from_cposvxx();
  status |= test_adapter_finalize_idempotent_zposv_from_zposvxx();
  status |= test_adapter_finalize_idempotent_sposv();
  status |= test_adapter_finalize_idempotent_dposv();
  status |= test_adapter_finalize_idempotent_cposv();
  status |= test_adapter_finalize_idempotent_zposv();
  status |= test_adapter_finalize_idempotent_ssysv();
  status |= test_adapter_finalize_idempotent_dsysv();
  status |= test_adapter_finalize_idempotent_csysv();
  status |= test_adapter_finalize_idempotent_zsysv();
  status |= test_adapter_finalize_idempotent_chesv_from_chesvx();
  status |= test_adapter_finalize_idempotent_zhesv_from_zhesvx();
  status |= test_adapter_finalize_idempotent_ssysv_from_ssysvxx();
  status |= test_adapter_finalize_idempotent_dsysv_from_dsysvxx();
  status |= test_adapter_finalize_idempotent_csysv_from_csysvxx();
  status |= test_adapter_finalize_idempotent_zsysv_from_zsysvxx();
  status |= test_adapter_finalize_idempotent_chesv_from_chesvxx();
  status |= test_adapter_finalize_idempotent_zhesv_from_zhesvxx();
  status |= test_adapter_finalize_idempotent_sgbsv();
  status |= test_adapter_finalize_idempotent_dgbsv();
  status |= test_adapter_finalize_idempotent_cgbsv();
  status |= test_adapter_finalize_idempotent_zgbsv();
  status |= test_adapter_finalize_idempotent_sgtsv();
  status |= test_adapter_finalize_idempotent_dgtsv();
  status |= test_adapter_finalize_idempotent_cgtsv();
  status |= test_adapter_finalize_idempotent_zgtsv();
  status |= test_adapter_finalize_idempotent_sptsv();
  status |= test_adapter_finalize_idempotent_dptsv();
  status |= test_adapter_finalize_idempotent_cptsv();
  status |= test_adapter_finalize_idempotent_zptsv();
  status |= test_adapter_finalize_idempotent_spbsv();
  status |= test_adapter_finalize_idempotent_dpbsv();
  status |= test_adapter_finalize_idempotent_cpbsv();
  status |= test_adapter_finalize_idempotent_zpbsv();
  status |= test_adapter_finalize_idempotent_sppsv();
  status |= test_adapter_finalize_idempotent_dppsv();
  status |= test_adapter_finalize_idempotent_cppsv();
  status |= test_adapter_finalize_idempotent_zppsv();
  status |= test_adapter_finalize_idempotent_sspsv();
  status |= test_adapter_finalize_idempotent_dspsv();
  status |= test_adapter_active_vtable_source_isolation();
  status |= test_adapter_active_vtable_source_isolation_sgesv();
  status |= test_adapter_active_vtable_source_isolation_dgesv();
  status |= test_adapter_active_vtable_source_isolation_sgesv_from_sgesvxx();
  status |= test_adapter_active_vtable_source_isolation_dgesv_from_dgesvxx();
  status |= test_adapter_active_vtable_source_isolation_cgesv_from_cgesvxx();
  status |= test_adapter_active_vtable_source_isolation_zgesv_from_zgesvxx();
  status |= test_adapter_active_vtable_source_isolation_sposv_from_sposvxx();
  status |= test_adapter_active_vtable_source_isolation_dposv_from_dposvxx();
  status |= test_adapter_active_vtable_source_isolation_cposv_from_cposvxx();
  status |= test_adapter_active_vtable_source_isolation_zposv_from_zposvxx();
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
  status |= test_adapter_active_vtable_source_isolation_chesv_from_chesvx();
  status |= test_adapter_active_vtable_source_isolation_zhesv_from_zhesvx();
  status |= test_adapter_active_vtable_source_isolation_ssysv_from_ssysvxx();
  status |= test_adapter_active_vtable_source_isolation_dsysv_from_dsysvxx();
  status |= test_adapter_active_vtable_source_isolation_csysv_from_csysvxx();
  status |= test_adapter_active_vtable_source_isolation_zsysv_from_zsysvxx();
  status |= test_adapter_active_vtable_source_isolation_chesv_from_chesvxx();
  status |= test_adapter_active_vtable_source_isolation_zhesv_from_zhesvxx();
  status |= test_adapter_active_vtable_source_isolation_sgbsv();
  status |= test_adapter_active_vtable_source_isolation_dgbsv();
  status |= test_adapter_active_vtable_source_isolation_cgbsv();
  status |= test_adapter_active_vtable_source_isolation_zgbsv();
  status |= test_adapter_active_vtable_source_isolation_sgtsv();
  status |= test_adapter_active_vtable_source_isolation_dgtsv();
  status |= test_adapter_active_vtable_source_isolation_cgtsv();
  status |= test_adapter_active_vtable_source_isolation_zgtsv();
  status |= test_adapter_active_vtable_source_isolation_sptsv();
  status |= test_adapter_active_vtable_source_isolation_dptsv();
  status |= test_adapter_active_vtable_source_isolation_cptsv();
  status |= test_adapter_active_vtable_source_isolation_zptsv();
  status |= test_adapter_active_vtable_source_isolation_spbsv();
  status |= test_adapter_active_vtable_source_isolation_dpbsv();
  status |= test_adapter_active_vtable_source_isolation_cpbsv();
  status |= test_adapter_active_vtable_source_isolation_zpbsv();
  status |= test_adapter_active_vtable_source_isolation_sppsv();
  status |= test_adapter_active_vtable_source_isolation_dppsv();
  status |= test_adapter_active_vtable_source_isolation_cppsv();
  status |= test_adapter_active_vtable_source_isolation_zppsv();
  status |= test_adapter_active_vtable_source_isolation_sspsv();
  status |= test_adapter_active_vtable_source_isolation_dspsv();

  if (status != 0) {
    fprintf(stderr, "Result: FAIL\n");
    return 1;
  }

  printf("Result: PASS\n");
  return 0;
}
