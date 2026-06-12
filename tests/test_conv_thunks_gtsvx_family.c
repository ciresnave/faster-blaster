#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef int (*fb_sgtsvx_cblas_fn)(fb_layout_t layout, char fact, char trans,
                                  int n, int nrhs, const float *dl,
                                  const float *d, const float *du, float *dlf,
                                  float *df, float *duf, float *du2, int *ipiv,
                                  const float *b, int ldb, float *x, int ldx,
                                  float *rcond, float *ferr, float *berr);
typedef void (*fb_sgtsvx_fortran_slot_fn)(char *fact, char *trans, int *n,
                                          int *nrhs, float *dl, float *d,
                                          float *du, float *dlf, float *df,
                                          float *duf, float *du2, int *ipiv,
                                          float *b, int *ldb, float *x,
                                          int *ldx, float *rcond, float *ferr,
                                          float *berr, float *work, int *iwork,
                                          int *info);

typedef int (*fb_dgtsvx_cblas_fn)(fb_layout_t layout, char fact, char trans,
                                  int n, int nrhs, const double *dl,
                                  const double *d, const double *du,
                                  double *dlf, double *df, double *duf,
                                  double *du2, int *ipiv, const double *b,
                                  int ldb, double *x, int ldx, double *rcond,
                                  double *ferr, double *berr);
typedef void (*fb_dgtsvx_fortran_slot_fn)(char *fact, char *trans, int *n,
                                          int *nrhs, double *dl, double *d,
                                          double *du, double *dlf, double *df,
                                          double *duf, double *du2, int *ipiv,
                                          double *b, int *ldb, double *x,
                                          int *ldx, double *rcond,
                                          double *ferr, double *berr,
                                          double *work, int *iwork,
                                          int *info);

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
typedef void (*fb_cgtsvx_fortran_slot_fn)(char *fact, char *trans, int *n,
                                          int *nrhs, fb_complex_float_t *dl,
                                          fb_complex_float_t *d,
                                          fb_complex_float_t *du,
                                          fb_complex_float_t *dlf,
                                          fb_complex_float_t *df,
                                          fb_complex_float_t *duf,
                                          fb_complex_float_t *du2, int *ipiv,
                                          fb_complex_float_t *b, int *ldb,
                                          fb_complex_float_t *x, int *ldx,
                                          float *rcond, float *ferr,
                                          float *berr, fb_complex_float_t *work,
                                          float *rwork, int *info);

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
typedef void (*fb_zgtsvx_fortran_slot_fn)(char *fact, char *trans, int *n,
                                          int *nrhs, fb_complex_double_t *dl,
                                          fb_complex_double_t *d,
                                          fb_complex_double_t *du,
                                          fb_complex_double_t *dlf,
                                          fb_complex_double_t *df,
                                          fb_complex_double_t *duf,
                                          fb_complex_double_t *du2, int *ipiv,
                                          fb_complex_double_t *b, int *ldb,
                                          fb_complex_double_t *x, int *ldx,
                                          double *rcond, double *ferr,
                                          double *berr,
                                          fb_complex_double_t *work,
                                          double *rwork, int *info);

static struct {
  int calls;
  char fact;
  char trans;
  int n;
  int nrhs;
  int ldb;
  int ldx;
  int work_seen;
  int aux_seen;
} g_sgtsvx_fortran_call;

static struct {
  int calls;
  char fact;
  char trans;
  int n;
  int nrhs;
  int ldb;
  int ldx;
  int work_seen;
  int aux_seen;
} g_dgtsvx_fortran_call;

static struct {
  int calls;
  fb_layout_t layout;
  char fact;
  char trans;
  int n;
  int nrhs;
  int ldb;
  int ldx;
} g_sgtsvx_cblas_call;

static struct {
  int calls;
  fb_layout_t layout;
  char fact;
  char trans;
  int n;
  int nrhs;
  int ldb;
  int ldx;
} g_dgtsvx_cblas_call;

static struct {
  int calls;
  char fact;
  char trans;
  int n;
  int nrhs;
  int ldb;
  int ldx;
  int work_seen;
  int aux_seen;
} g_cgtsvx_fortran_call;

static struct {
  int calls;
  char fact;
  char trans;
  int n;
  int nrhs;
  int ldb;
  int ldx;
  int work_seen;
  int aux_seen;
} g_zgtsvx_fortran_call;

static struct {
  int calls;
  fb_layout_t layout;
  char fact;
  char trans;
  int n;
  int nrhs;
  int ldb;
  int ldx;
} g_cgtsvx_cblas_call;

static struct {
  int calls;
  fb_layout_t layout;
  char fact;
  char trans;
  int n;
  int nrhs;
  int ldb;
  int ldx;
} g_zgtsvx_cblas_call;

static fb_complex_float_t make_cf32(float real_part, float imag_part) {
  fb_complex_float_t value;
  __real__ value = real_part;
  __imag__ value = imag_part;
  return value;
}

static fb_complex_double_t make_cd64(double real_part, double imag_part) {
  fb_complex_double_t value;
  __real__ value = real_part;
  __imag__ value = imag_part;
  return value;
}

static void stub_sgtsvx_fortran(char *fact, char *trans, int *n, int *nrhs,
                                float *dl, float *d, float *du, float *dlf,
                                float *df, float *duf, float *du2, int *ipiv,
                                float *b, int *ldb, float *x, int *ldx,
                                float *rcond, float *ferr, float *berr,
                                float *work, int *iwork, int *info) {
  (void)dl;
  (void)d;
  (void)du;
  (void)dlf;
  (void)df;
  (void)duf;
  (void)du2;
  (void)ipiv;
  (void)b;
  (void)x;
  (void)rcond;
  (void)ferr;
  (void)berr;
  g_sgtsvx_fortran_call.calls += 1;
  g_sgtsvx_fortran_call.fact = *fact;
  g_sgtsvx_fortran_call.trans = *trans;
  g_sgtsvx_fortran_call.n = *n;
  g_sgtsvx_fortran_call.nrhs = *nrhs;
  g_sgtsvx_fortran_call.ldb = *ldb;
  g_sgtsvx_fortran_call.ldx = *ldx;
  g_sgtsvx_fortran_call.work_seen = (work != NULL);
  g_sgtsvx_fortran_call.aux_seen = (iwork != NULL);
  *info = 123;
}

static int stub_sgtsvx_cblas(fb_layout_t layout, char fact, char trans, int n,
                             int nrhs, const float *dl, const float *d,
                             const float *du, float *dlf, float *df,
                             float *duf, float *du2, int *ipiv, const float *b,
                             int ldb, float *x, int ldx, float *rcond,
                             float *ferr, float *berr) {
  (void)dl;
  (void)d;
  (void)du;
  (void)dlf;
  (void)df;
  (void)duf;
  (void)du2;
  (void)ipiv;
  (void)b;
  (void)x;
  (void)rcond;
  (void)ferr;
  (void)berr;
  g_sgtsvx_cblas_call.calls += 1;
  g_sgtsvx_cblas_call.layout = layout;
  g_sgtsvx_cblas_call.fact = fact;
  g_sgtsvx_cblas_call.trans = trans;
  g_sgtsvx_cblas_call.n = n;
  g_sgtsvx_cblas_call.nrhs = nrhs;
  g_sgtsvx_cblas_call.ldb = ldb;
  g_sgtsvx_cblas_call.ldx = ldx;
  return 321;
}

static void stub_dgtsvx_fortran(char *fact, char *trans, int *n, int *nrhs,
                                double *dl, double *d, double *du,
                                double *dlf, double *df, double *duf,
                                double *du2, int *ipiv, double *b, int *ldb,
                                double *x, int *ldx, double *rcond,
                                double *ferr, double *berr, double *work,
                                int *iwork, int *info) {
  (void)dl;
  (void)d;
  (void)du;
  (void)dlf;
  (void)df;
  (void)duf;
  (void)du2;
  (void)ipiv;
  (void)b;
  (void)x;
  (void)rcond;
  (void)ferr;
  (void)berr;
  g_dgtsvx_fortran_call.calls += 1;
  g_dgtsvx_fortran_call.fact = *fact;
  g_dgtsvx_fortran_call.trans = *trans;
  g_dgtsvx_fortran_call.n = *n;
  g_dgtsvx_fortran_call.nrhs = *nrhs;
  g_dgtsvx_fortran_call.ldb = *ldb;
  g_dgtsvx_fortran_call.ldx = *ldx;
  g_dgtsvx_fortran_call.work_seen = (work != NULL);
  g_dgtsvx_fortran_call.aux_seen = (iwork != NULL);
  *info = 223;
}

static int stub_dgtsvx_cblas(fb_layout_t layout, char fact, char trans, int n,
                             int nrhs, const double *dl, const double *d,
                             const double *du, double *dlf, double *df,
                             double *duf, double *du2, int *ipiv,
                             const double *b, int ldb, double *x, int ldx,
                             double *rcond, double *ferr, double *berr) {
  (void)dl;
  (void)d;
  (void)du;
  (void)dlf;
  (void)df;
  (void)duf;
  (void)du2;
  (void)ipiv;
  (void)b;
  (void)x;
  (void)rcond;
  (void)ferr;
  (void)berr;
  g_dgtsvx_cblas_call.calls += 1;
  g_dgtsvx_cblas_call.layout = layout;
  g_dgtsvx_cblas_call.fact = fact;
  g_dgtsvx_cblas_call.trans = trans;
  g_dgtsvx_cblas_call.n = n;
  g_dgtsvx_cblas_call.nrhs = nrhs;
  g_dgtsvx_cblas_call.ldb = ldb;
  g_dgtsvx_cblas_call.ldx = ldx;
  return 421;
}

static void stub_cgtsvx_fortran(char *fact, char *trans, int *n, int *nrhs,
                                fb_complex_float_t *dl, fb_complex_float_t *d,
                                fb_complex_float_t *du, fb_complex_float_t *dlf,
                                fb_complex_float_t *df,
                                fb_complex_float_t *duf,
                                fb_complex_float_t *du2, int *ipiv,
                                fb_complex_float_t *b, int *ldb,
                                fb_complex_float_t *x, int *ldx, float *rcond,
                                float *ferr, float *berr,
                                fb_complex_float_t *work, float *rwork,
                                int *info) {
  (void)dl;
  (void)d;
  (void)du;
  (void)dlf;
  (void)df;
  (void)duf;
  (void)du2;
  (void)ipiv;
  (void)b;
  (void)x;
  (void)rcond;
  (void)ferr;
  (void)berr;
  g_cgtsvx_fortran_call.calls += 1;
  g_cgtsvx_fortran_call.fact = *fact;
  g_cgtsvx_fortran_call.trans = *trans;
  g_cgtsvx_fortran_call.n = *n;
  g_cgtsvx_fortran_call.nrhs = *nrhs;
  g_cgtsvx_fortran_call.ldb = *ldb;
  g_cgtsvx_fortran_call.ldx = *ldx;
  g_cgtsvx_fortran_call.work_seen = (work != NULL);
  g_cgtsvx_fortran_call.aux_seen = (rwork != NULL);
  *info = 456;
}

static int stub_cgtsvx_cblas(fb_layout_t layout, char fact, char trans, int n,
                             int nrhs, const fb_complex_float_t *dl,
                             const fb_complex_float_t *d,
                             const fb_complex_float_t *du,
                             fb_complex_float_t *dlf, fb_complex_float_t *df,
                             fb_complex_float_t *duf,
                             fb_complex_float_t *du2, int *ipiv,
                             const fb_complex_float_t *b, int ldb,
                             fb_complex_float_t *x, int ldx, float *rcond,
                             float *ferr, float *berr) {
  (void)dl;
  (void)d;
  (void)du;
  (void)dlf;
  (void)df;
  (void)duf;
  (void)du2;
  (void)ipiv;
  (void)b;
  (void)x;
  (void)rcond;
  (void)ferr;
  (void)berr;
  g_cgtsvx_cblas_call.calls += 1;
  g_cgtsvx_cblas_call.layout = layout;
  g_cgtsvx_cblas_call.fact = fact;
  g_cgtsvx_cblas_call.trans = trans;
  g_cgtsvx_cblas_call.n = n;
  g_cgtsvx_cblas_call.nrhs = nrhs;
  g_cgtsvx_cblas_call.ldb = ldb;
  g_cgtsvx_cblas_call.ldx = ldx;
  return 654;
}

static void stub_zgtsvx_fortran(char *fact, char *trans, int *n, int *nrhs,
                                fb_complex_double_t *dl,
                                fb_complex_double_t *d,
                                fb_complex_double_t *du,
                                fb_complex_double_t *dlf,
                                fb_complex_double_t *df,
                                fb_complex_double_t *duf,
                                fb_complex_double_t *du2, int *ipiv,
                                fb_complex_double_t *b, int *ldb,
                                fb_complex_double_t *x, int *ldx,
                                double *rcond, double *ferr, double *berr,
                                fb_complex_double_t *work, double *rwork,
                                int *info) {
  (void)dl;
  (void)d;
  (void)du;
  (void)dlf;
  (void)df;
  (void)duf;
  (void)du2;
  (void)ipiv;
  (void)b;
  (void)x;
  (void)rcond;
  (void)ferr;
  (void)berr;
  g_zgtsvx_fortran_call.calls += 1;
  g_zgtsvx_fortran_call.fact = *fact;
  g_zgtsvx_fortran_call.trans = *trans;
  g_zgtsvx_fortran_call.n = *n;
  g_zgtsvx_fortran_call.nrhs = *nrhs;
  g_zgtsvx_fortran_call.ldb = *ldb;
  g_zgtsvx_fortran_call.ldx = *ldx;
  g_zgtsvx_fortran_call.work_seen = (work != NULL);
  g_zgtsvx_fortran_call.aux_seen = (rwork != NULL);
  *info = 756;
}

static int stub_zgtsvx_cblas(fb_layout_t layout, char fact, char trans, int n,
                             int nrhs, const fb_complex_double_t *dl,
                             const fb_complex_double_t *d,
                             const fb_complex_double_t *du,
                             fb_complex_double_t *dlf,
                             fb_complex_double_t *df,
                             fb_complex_double_t *duf,
                             fb_complex_double_t *du2, int *ipiv,
                             const fb_complex_double_t *b, int ldb,
                             fb_complex_double_t *x, int ldx, double *rcond,
                             double *ferr, double *berr) {
  (void)dl;
  (void)d;
  (void)du;
  (void)dlf;
  (void)df;
  (void)duf;
  (void)du2;
  (void)ipiv;
  (void)b;
  (void)x;
  (void)rcond;
  (void)ferr;
  (void)berr;
  g_zgtsvx_cblas_call.calls += 1;
  g_zgtsvx_cblas_call.layout = layout;
  g_zgtsvx_cblas_call.fact = fact;
  g_zgtsvx_cblas_call.trans = trans;
  g_zgtsvx_cblas_call.n = n;
  g_zgtsvx_cblas_call.nrhs = nrhs;
  g_zgtsvx_cblas_call.ldb = ldb;
  g_zgtsvx_cblas_call.ldx = ldx;
  return 854;
}

static int check_sgtsvx_fortran_to_cblas(void) {
  fb_backend_vtable_t vtable;
  fb_sgtsvx_cblas_fn thunk;
  float dl[2] = {1.0f, 2.0f};
  float d[2] = {3.0f, 4.0f};
  float du[2] = {5.0f, 6.0f};
  float dlf[2] = {0.0f, 0.0f};
  float df[2] = {0.0f, 0.0f};
  float duf[2] = {0.0f, 0.0f};
  float du2[2] = {0.0f, 0.0f};
  int ipiv[2] = {0, 0};
  float b[4] = {7.0f, 8.0f, 9.0f, 10.0f};
  float x[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float rcond = 0.0f;
  float ferr[2] = {0.0f, 0.0f};
  float berr[2] = {0.0f, 0.0f};

  memset(&vtable, 0, sizeof(vtable));
  memset(&g_sgtsvx_fortran_call, 0, sizeof(g_sgtsvx_fortran_call));
  vtable.ext_ops[FB_OP_SGTSVX][FB_CONV_FORTRAN] =
      (fb_generic_fn)(void (*)(void))stub_sgtsvx_fortran;
  fb_install_conv_thunks(&vtable, FB_OP_SGTSVX);

  thunk = (fb_sgtsvx_cblas_fn)vtable.ext_ops[FB_OP_SGTSVX][FB_CONV_CBLAS];
  if (!thunk) {
    fprintf(stderr, "[FAIL] SGTSVX F2C thunk was not installed\n");
    return 1;
  }

  if (thunk(FB_LAYOUT_COL_MAJOR, 'N', 'T', 2, 2, dl, d, du, dlf, df, duf, du2,
            ipiv, b, 2, x, 2, &rcond, ferr, berr) != 123) {
    fprintf(stderr, "[FAIL] SGTSVX F2C thunk returned unexpected info\n");
    return 1;
  }

  if (g_sgtsvx_fortran_call.calls != 1 || g_sgtsvx_fortran_call.fact != 'N' ||
      g_sgtsvx_fortran_call.trans != 'T' || g_sgtsvx_fortran_call.n != 2 ||
      g_sgtsvx_fortran_call.nrhs != 2 || g_sgtsvx_fortran_call.ldb != 2 ||
      g_sgtsvx_fortran_call.ldx != 2 || !g_sgtsvx_fortran_call.work_seen ||
      !g_sgtsvx_fortran_call.aux_seen) {
    fprintf(stderr, "[FAIL] SGTSVX F2C thunk forwarded incorrect ABI\n");
    return 1;
  }

  printf("[PASS] SGTSVX Fortran->CBLAS thunk uses the compact LAPACKE ABI and allocates work/iwork\n");
  return 0;
}

static int check_sgtsvx_cblas_to_fortran(void) {
  fb_backend_vtable_t vtable;
  fb_sgtsvx_fortran_slot_fn thunk;
  char fact = 'F';
  char trans = 'N';
  int n = 2;
  int nrhs = 1;
  float dl[2] = {0.0f, 0.0f};
  float d[2] = {0.0f, 0.0f};
  float du[2] = {0.0f, 0.0f};
  float dlf[2] = {0.0f, 0.0f};
  float df[2] = {0.0f, 0.0f};
  float duf[2] = {0.0f, 0.0f};
  float du2[2] = {0.0f, 0.0f};
  int ipiv[2] = {0, 0};
  float b[2] = {0.0f, 0.0f};
  int ldb = 2;
  float x[2] = {0.0f, 0.0f};
  int ldx = 2;
  float rcond = 0.0f;
  float ferr[1] = {0.0f};
  float berr[1] = {0.0f};
  float work[6] = {0.0f};
  int iwork[2] = {0, 0};
  int info = 0;

  memset(&vtable, 0, sizeof(vtable));
  memset(&g_sgtsvx_cblas_call, 0, sizeof(g_sgtsvx_cblas_call));
  vtable.ext_ops[FB_OP_SGTSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_sgtsvx_cblas;
  fb_install_conv_thunks(&vtable, FB_OP_SGTSVX);

  thunk = (fb_sgtsvx_fortran_slot_fn)vtable.ext_ops[FB_OP_SGTSVX][FB_CONV_FORTRAN];
  if (!thunk) {
    fprintf(stderr, "[FAIL] SGTSVX C2F thunk was not installed\n");
    return 1;
  }

  thunk(&fact, &trans, &n, &nrhs, dl, d, du, dlf, df, duf, du2, ipiv, b, &ldb,
        x, &ldx, &rcond, ferr, berr, work, iwork, &info);

  if (info != 321 || g_sgtsvx_cblas_call.calls != 1 ||
      g_sgtsvx_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
      g_sgtsvx_cblas_call.fact != 'F' || g_sgtsvx_cblas_call.trans != 'N' ||
      g_sgtsvx_cblas_call.n != 2 || g_sgtsvx_cblas_call.nrhs != 1 ||
      g_sgtsvx_cblas_call.ldb != 2 || g_sgtsvx_cblas_call.ldx != 2) {
    fprintf(stderr, "[FAIL] SGTSVX C2F thunk forwarded incorrect ABI\n");
    return 1;
  }

  printf("[PASS] SGTSVX CBLAS->Fortran thunk hides work/iwork and forwards the compact column-major ABI\n");
  return 0;
}

static int check_dgtsvx_fortran_to_cblas(void) {
  fb_backend_vtable_t vtable;
  fb_dgtsvx_cblas_fn thunk;
  double dl[2] = {1.0, 2.0};
  double d[2] = {3.0, 4.0};
  double du[2] = {5.0, 6.0};
  double dlf[2] = {0.0, 0.0};
  double df[2] = {0.0, 0.0};
  double duf[2] = {0.0, 0.0};
  double du2[2] = {0.0, 0.0};
  int ipiv[2] = {0, 0};
  double b[4] = {7.0, 8.0, 9.0, 10.0};
  double x[4] = {0.0, 0.0, 0.0, 0.0};
  double rcond = 0.0;
  double ferr[2] = {0.0, 0.0};
  double berr[2] = {0.0, 0.0};

  memset(&vtable, 0, sizeof(vtable));
  memset(&g_dgtsvx_fortran_call, 0, sizeof(g_dgtsvx_fortran_call));
  vtable.ext_ops[FB_OP_DGTSVX][FB_CONV_FORTRAN] =
      (fb_generic_fn)(void (*)(void))stub_dgtsvx_fortran;
  fb_install_conv_thunks(&vtable, FB_OP_DGTSVX);

  thunk = (fb_dgtsvx_cblas_fn)vtable.ext_ops[FB_OP_DGTSVX][FB_CONV_CBLAS];
  if (!thunk) {
    fprintf(stderr, "[FAIL] DGTSVX F2C thunk was not installed\n");
    return 1;
  }

  if (thunk(FB_LAYOUT_COL_MAJOR, 'N', 'T', 2, 2, dl, d, du, dlf, df, duf, du2,
            ipiv, b, 2, x, 2, &rcond, ferr, berr) != 223) {
    fprintf(stderr, "[FAIL] DGTSVX F2C thunk returned unexpected info\n");
    return 1;
  }

  if (g_dgtsvx_fortran_call.calls != 1 || g_dgtsvx_fortran_call.fact != 'N' ||
      g_dgtsvx_fortran_call.trans != 'T' || g_dgtsvx_fortran_call.n != 2 ||
      g_dgtsvx_fortran_call.nrhs != 2 || g_dgtsvx_fortran_call.ldb != 2 ||
      g_dgtsvx_fortran_call.ldx != 2 || !g_dgtsvx_fortran_call.work_seen ||
      !g_dgtsvx_fortran_call.aux_seen) {
    fprintf(stderr, "[FAIL] DGTSVX F2C thunk forwarded incorrect ABI\n");
    return 1;
  }

  printf("[PASS] DGTSVX Fortran->CBLAS thunk uses the compact LAPACKE ABI and allocates WORK/IWORK\n");
  return 0;
}

static int check_dgtsvx_cblas_to_fortran(void) {
  fb_backend_vtable_t vtable;
  fb_dgtsvx_fortran_slot_fn thunk;
  char fact = 'F';
  char trans = 'N';
  int n = 2;
  int nrhs = 1;
  double dl[2] = {0.0, 0.0};
  double d[2] = {0.0, 0.0};
  double du[2] = {0.0, 0.0};
  double dlf[2] = {0.0, 0.0};
  double df[2] = {0.0, 0.0};
  double duf[2] = {0.0, 0.0};
  double du2[2] = {0.0, 0.0};
  int ipiv[2] = {0, 0};
  double b[2] = {0.0, 0.0};
  int ldb = 2;
  double x[2] = {0.0, 0.0};
  int ldx = 2;
  double rcond = 0.0;
  double ferr[1] = {0.0};
  double berr[1] = {0.0};
  double work[6] = {0.0};
  int iwork[2] = {0, 0};
  int info = 0;

  memset(&vtable, 0, sizeof(vtable));
  memset(&g_dgtsvx_cblas_call, 0, sizeof(g_dgtsvx_cblas_call));
  vtable.ext_ops[FB_OP_DGTSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_dgtsvx_cblas;
  fb_install_conv_thunks(&vtable, FB_OP_DGTSVX);

  thunk = (fb_dgtsvx_fortran_slot_fn)vtable.ext_ops[FB_OP_DGTSVX][FB_CONV_FORTRAN];
  if (!thunk) {
    fprintf(stderr, "[FAIL] DGTSVX C2F thunk was not installed\n");
    return 1;
  }

  thunk(&fact, &trans, &n, &nrhs, dl, d, du, dlf, df, duf, du2, ipiv, b, &ldb,
        x, &ldx, &rcond, ferr, berr, work, iwork, &info);

  if (info != 421 || g_dgtsvx_cblas_call.calls != 1 ||
      g_dgtsvx_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
      g_dgtsvx_cblas_call.fact != 'F' || g_dgtsvx_cblas_call.trans != 'N' ||
      g_dgtsvx_cblas_call.n != 2 || g_dgtsvx_cblas_call.nrhs != 1 ||
      g_dgtsvx_cblas_call.ldb != 2 || g_dgtsvx_cblas_call.ldx != 2) {
    fprintf(stderr, "[FAIL] DGTSVX C2F thunk forwarded incorrect ABI\n");
    return 1;
  }

  printf("[PASS] DGTSVX CBLAS->Fortran thunk hides WORK/IWORK and forwards the compact column-major ABI\n");
  return 0;
}

static int check_cgtsvx_fortran_to_cblas(void) {
  fb_backend_vtable_t vtable;
  fb_cgtsvx_cblas_fn thunk;
  fb_complex_float_t dl[2] = {0};
  fb_complex_float_t d[2] = {0};
  fb_complex_float_t du[2] = {0};
  fb_complex_float_t dlf[2] = {0};
  fb_complex_float_t df[2] = {0};
  fb_complex_float_t duf[2] = {0};
  fb_complex_float_t du2[2] = {0};
  int ipiv[2] = {0, 0};
  fb_complex_float_t b[4] = {0};
  fb_complex_float_t x[4] = {0};
  float rcond = 0.0f;
  float ferr[2] = {0.0f, 0.0f};
  float berr[2] = {0.0f, 0.0f};

  dl[0] = make_cf32(1.0f, 2.0f);
  d[0] = make_cf32(3.0f, 4.0f);
  du[0] = make_cf32(5.0f, 6.0f);

  memset(&vtable, 0, sizeof(vtable));
  memset(&g_cgtsvx_fortran_call, 0, sizeof(g_cgtsvx_fortran_call));
  vtable.ext_ops[FB_OP_CGTSVX][FB_CONV_FORTRAN] =
      (fb_generic_fn)(void (*)(void))stub_cgtsvx_fortran;
  fb_install_conv_thunks(&vtable, FB_OP_CGTSVX);

  thunk = (fb_cgtsvx_cblas_fn)vtable.ext_ops[FB_OP_CGTSVX][FB_CONV_CBLAS];
  if (!thunk) {
    fprintf(stderr, "[FAIL] CGTSVX F2C thunk was not installed\n");
    return 1;
  }

  if (thunk(FB_LAYOUT_COL_MAJOR, 'N', 'C', 2, 2, dl, d, du, dlf, df, duf, du2,
            ipiv, b, 2, x, 2, &rcond, ferr, berr) != 456) {
    fprintf(stderr, "[FAIL] CGTSVX F2C thunk returned unexpected info\n");
    return 1;
  }

  if (g_cgtsvx_fortran_call.calls != 1 || g_cgtsvx_fortran_call.fact != 'N' ||
      g_cgtsvx_fortran_call.trans != 'C' || g_cgtsvx_fortran_call.n != 2 ||
      g_cgtsvx_fortran_call.nrhs != 2 || g_cgtsvx_fortran_call.ldb != 2 ||
      g_cgtsvx_fortran_call.ldx != 2 || !g_cgtsvx_fortran_call.work_seen ||
      !g_cgtsvx_fortran_call.aux_seen) {
    fprintf(stderr, "[FAIL] CGTSVX F2C thunk forwarded incorrect complex ABI\n");
    return 1;
  }

  printf("[PASS] CGTSVX Fortran->CBLAS thunk uses the compact LAPACKE ABI and allocates WORK/RWORK\n");
  return 0;
}

static int check_cgtsvx_cblas_to_fortran(void) {
  fb_backend_vtable_t vtable;
  fb_cgtsvx_fortran_slot_fn thunk;
  char fact = 'F';
  char trans = 'N';
  int n = 2;
  int nrhs = 1;
  fb_complex_float_t dl[2] = {0};
  fb_complex_float_t d[2] = {0};
  fb_complex_float_t du[2] = {0};
  fb_complex_float_t dlf[2] = {0};
  fb_complex_float_t df[2] = {0};
  fb_complex_float_t duf[2] = {0};
  fb_complex_float_t du2[2] = {0};
  int ipiv[2] = {0, 0};
  fb_complex_float_t b[2] = {0};
  int ldb = 2;
  fb_complex_float_t x[2] = {0};
  int ldx = 2;
  float rcond = 0.0f;
  float ferr[1] = {0.0f};
  float berr[1] = {0.0f};
  fb_complex_float_t work[4] = {0};
  float rwork[2] = {0.0f, 0.0f};
  int info = 0;

  memset(&vtable, 0, sizeof(vtable));
  memset(&g_cgtsvx_cblas_call, 0, sizeof(g_cgtsvx_cblas_call));
  vtable.ext_ops[FB_OP_CGTSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_cgtsvx_cblas;
  fb_install_conv_thunks(&vtable, FB_OP_CGTSVX);

  thunk = (fb_cgtsvx_fortran_slot_fn)vtable.ext_ops[FB_OP_CGTSVX][FB_CONV_FORTRAN];
  if (!thunk) {
    fprintf(stderr, "[FAIL] CGTSVX C2F thunk was not installed\n");
    return 1;
  }

  thunk(&fact, &trans, &n, &nrhs, dl, d, du, dlf, df, duf, du2, ipiv, b, &ldb,
        x, &ldx, &rcond, ferr, berr, work, rwork, &info);

  if (info != 654 || g_cgtsvx_cblas_call.calls != 1 ||
      g_cgtsvx_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
      g_cgtsvx_cblas_call.fact != 'F' || g_cgtsvx_cblas_call.trans != 'N' ||
      g_cgtsvx_cblas_call.n != 2 || g_cgtsvx_cblas_call.nrhs != 1 ||
      g_cgtsvx_cblas_call.ldb != 2 || g_cgtsvx_cblas_call.ldx != 2) {
    fprintf(stderr, "[FAIL] CGTSVX C2F thunk forwarded incorrect complex ABI\n");
    return 1;
  }

  printf("[PASS] CGTSVX CBLAS->Fortran thunk hides WORK/RWORK and forwards the compact column-major ABI\n");
  return 0;
}

static int check_zgtsvx_fortran_to_cblas(void) {
  fb_backend_vtable_t vtable;
  fb_zgtsvx_cblas_fn thunk;
  fb_complex_double_t dl[2] = {0};
  fb_complex_double_t d[2] = {0};
  fb_complex_double_t du[2] = {0};
  fb_complex_double_t dlf[2] = {0};
  fb_complex_double_t df[2] = {0};
  fb_complex_double_t duf[2] = {0};
  fb_complex_double_t du2[2] = {0};
  int ipiv[2] = {0, 0};
  fb_complex_double_t b[4] = {0};
  fb_complex_double_t x[4] = {0};
  double rcond = 0.0;
  double ferr[2] = {0.0, 0.0};
  double berr[2] = {0.0, 0.0};

  dl[0] = make_cd64(1.0, 2.0);
  d[0] = make_cd64(3.0, 4.0);
  du[0] = make_cd64(5.0, 6.0);

  memset(&vtable, 0, sizeof(vtable));
  memset(&g_zgtsvx_fortran_call, 0, sizeof(g_zgtsvx_fortran_call));
  vtable.ext_ops[FB_OP_ZGTSVX][FB_CONV_FORTRAN] =
      (fb_generic_fn)(void (*)(void))stub_zgtsvx_fortran;
  fb_install_conv_thunks(&vtable, FB_OP_ZGTSVX);

  thunk = (fb_zgtsvx_cblas_fn)vtable.ext_ops[FB_OP_ZGTSVX][FB_CONV_CBLAS];
  if (!thunk) {
    fprintf(stderr, "[FAIL] ZGTSVX F2C thunk was not installed\n");
    return 1;
  }

  if (thunk(FB_LAYOUT_COL_MAJOR, 'N', 'C', 2, 2, dl, d, du, dlf, df, duf, du2,
            ipiv, b, 2, x, 2, &rcond, ferr, berr) != 756) {
    fprintf(stderr, "[FAIL] ZGTSVX F2C thunk returned unexpected info\n");
    return 1;
  }

  if (g_zgtsvx_fortran_call.calls != 1 || g_zgtsvx_fortran_call.fact != 'N' ||
      g_zgtsvx_fortran_call.trans != 'C' || g_zgtsvx_fortran_call.n != 2 ||
      g_zgtsvx_fortran_call.nrhs != 2 || g_zgtsvx_fortran_call.ldb != 2 ||
      g_zgtsvx_fortran_call.ldx != 2 || !g_zgtsvx_fortran_call.work_seen ||
      !g_zgtsvx_fortran_call.aux_seen) {
    fprintf(stderr, "[FAIL] ZGTSVX F2C thunk forwarded incorrect complex ABI\n");
    return 1;
  }

  printf("[PASS] ZGTSVX Fortran->CBLAS thunk uses the compact LAPACKE ABI and allocates WORK/RWORK\n");
  return 0;
}

static int check_zgtsvx_cblas_to_fortran(void) {
  fb_backend_vtable_t vtable;
  fb_zgtsvx_fortran_slot_fn thunk;
  char fact = 'F';
  char trans = 'N';
  int n = 2;
  int nrhs = 1;
  fb_complex_double_t dl[2] = {0};
  fb_complex_double_t d[2] = {0};
  fb_complex_double_t du[2] = {0};
  fb_complex_double_t dlf[2] = {0};
  fb_complex_double_t df[2] = {0};
  fb_complex_double_t duf[2] = {0};
  fb_complex_double_t du2[2] = {0};
  int ipiv[2] = {0, 0};
  fb_complex_double_t b[2] = {0};
  int ldb = 2;
  fb_complex_double_t x[2] = {0};
  int ldx = 2;
  double rcond = 0.0;
  double ferr[1] = {0.0};
  double berr[1] = {0.0};
  fb_complex_double_t work[4] = {0};
  double rwork[2] = {0.0, 0.0};
  int info = 0;

  memset(&vtable, 0, sizeof(vtable));
  memset(&g_zgtsvx_cblas_call, 0, sizeof(g_zgtsvx_cblas_call));
  vtable.ext_ops[FB_OP_ZGTSVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zgtsvx_cblas;
  fb_install_conv_thunks(&vtable, FB_OP_ZGTSVX);

  thunk = (fb_zgtsvx_fortran_slot_fn)vtable.ext_ops[FB_OP_ZGTSVX][FB_CONV_FORTRAN];
  if (!thunk) {
    fprintf(stderr, "[FAIL] ZGTSVX C2F thunk was not installed\n");
    return 1;
  }

  thunk(&fact, &trans, &n, &nrhs, dl, d, du, dlf, df, duf, du2, ipiv, b, &ldb,
        x, &ldx, &rcond, ferr, berr, work, rwork, &info);

  if (info != 854 || g_zgtsvx_cblas_call.calls != 1 ||
      g_zgtsvx_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
      g_zgtsvx_cblas_call.fact != 'F' || g_zgtsvx_cblas_call.trans != 'N' ||
      g_zgtsvx_cblas_call.n != 2 || g_zgtsvx_cblas_call.nrhs != 1 ||
      g_zgtsvx_cblas_call.ldb != 2 || g_zgtsvx_cblas_call.ldx != 2) {
    fprintf(stderr, "[FAIL] ZGTSVX C2F thunk forwarded incorrect complex ABI\n");
    return 1;
  }

  printf("[PASS] ZGTSVX CBLAS->Fortran thunk hides WORK/RWORK and forwards the compact column-major ABI\n");
  return 0;
}

int main(void) {
  if (check_sgtsvx_fortran_to_cblas() != 0) return 1;
  if (check_sgtsvx_cblas_to_fortran() != 0) return 1;
  if (check_dgtsvx_fortran_to_cblas() != 0) return 1;
  if (check_dgtsvx_cblas_to_fortran() != 0) return 1;
  if (check_cgtsvx_fortran_to_cblas() != 0) return 1;
  if (check_cgtsvx_cblas_to_fortran() != 0) return 1;
  if (check_zgtsvx_fortran_to_cblas() != 0) return 1;
  if (check_zgtsvx_cblas_to_fortran() != 0) return 1;

  printf("Result: PASS\n");
  return 0;
}