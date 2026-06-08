#include <stdio.h>
#include <string.h>

#include "../include/faster-blaster/vtable_autofill.h"
#include "../src/backends/backend_interface.h"
#include "../src/judge/judge_op_ids.h"

typedef int (*fb_chegv_cblas_fn)(fb_layout_t layout, int itype, char jobz,
                                 fb_uplo_t uplo, int n,
                                 fb_complex_float_t *a, int lda,
                                 fb_complex_float_t *b, int ldb, float *w);

typedef int (*fb_zhegv_cblas_fn)(fb_layout_t layout, int itype, char jobz,
                                 fb_uplo_t uplo, int n,
                                 fb_complex_double_t *a, int lda,
                                 fb_complex_double_t *b, int ldb, double *w);

typedef int (*fb_cheevr_cblas_fn)(fb_layout_t layout, char jobz, char range,
                                  fb_uplo_t uplo, int n,
                                  fb_complex_float_t *a, int lda, float vl,
                                  float vu, int il, int iu, float abstol,
                                  int *m, float *w, fb_complex_float_t *z,
                                  int ldz, int *isuppz);

typedef int (*fb_cheevx_cblas_fn)(fb_layout_t layout, char jobz, char range,
                                  fb_uplo_t uplo, int n,
                                  fb_complex_float_t *a, int lda, float vl,
                                  float vu, int il, int iu, float abstol,
                                  int *m, float *w, fb_complex_float_t *z,
                                  int ldz, int *ifail);

typedef int (*fb_zheevr_cblas_fn)(fb_layout_t layout, char jobz, char range,
                                  fb_uplo_t uplo, int n,
                                  fb_complex_double_t *a, int lda, double vl,
                                  double vu, int il, int iu, double abstol,
                                  int *m, double *w, fb_complex_double_t *z,
                                  int ldz, int *isuppz);

typedef int (*fb_zheevx_cblas_fn)(fb_layout_t layout, char jobz, char range,
                                  fb_uplo_t uplo, int n,
                                  fb_complex_double_t *a, int lda, double vl,
                                  double vu, int il, int iu, double abstol,
                                  int *m, double *w, fb_complex_double_t *z,
                                  int ldz, int *ifail);

typedef int (*fb_csygvx_cblas_fn)(fb_layout_t layout, int itype, char jobz,
                                  char range, fb_uplo_t uplo, int n,
                                  fb_complex_float_t *a, int lda,
                                  fb_complex_float_t *b, int ldb, float vl,
                                  float vu, int il, int iu, float abstol,
                                  int *m, float *w, fb_complex_float_t *z,
                                  int ldz, int *ifail);

typedef int (*fb_zsygvx_cblas_fn)(fb_layout_t layout, int itype, char jobz,
                                  char range, fb_uplo_t uplo, int n,
                                  fb_complex_double_t *a, int lda,
                                  fb_complex_double_t *b, int ldb, double vl,
                                  double vu, int il, int iu, double abstol,
                                  int *m, double *w, fb_complex_double_t *z,
                                  int ldz, int *ifail);

typedef int (*fb_chesv_cblas_fn)(fb_layout_t layout, fb_uplo_t uplo, int n,
                                 int nrhs, fb_complex_float_t *a, int lda,
                                 int *ipiv, fb_complex_float_t *b, int ldb);

static int g_c_calls = 0;
static int g_z_calls = 0;
static int g_c_prec_primary_calls = 0;
static int g_c_prec_secondary_calls = 0;
static int g_solver_alias_calls = 0;
static int g_cheevr_bridge_calls = 0;
static int g_zheevx_bridge_calls = 0;
static int g_csygvx_bridge_calls = 0;
static int g_zsygvx_bridge_calls = 0;
static int g_cheevx_bridge_calls = 0;
static int g_zheevr_bridge_calls = 0;
static int g_chegvx_bridge_calls = 0;
static int g_zhegvx_bridge_calls = 0;

static int stub_chegvd_base(fb_layout_t layout, int itype, char jobz,
                            fb_uplo_t uplo, int n,
                            fb_complex_float_t *a, int lda,
                            fb_complex_float_t *b, int ldb, float *w) {
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
  g_c_calls += 1;
  return 515;
}

static int stub_zhegvd_base(fb_layout_t layout, int itype, char jobz,
                            fb_uplo_t uplo, int n,
                            fb_complex_double_t *a, int lda,
                            fb_complex_double_t *b, int ldb, double *w) {
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
  g_z_calls += 1;
  return 616;
}

static int stub_cheevr_base(fb_layout_t layout, char jobz, char range,
                            fb_uplo_t uplo, int n,
                            fb_complex_float_t *a, int lda, float vl, float vu,
                            int il, int iu, float abstol, int *m, float *w,
                            fb_complex_float_t *z, int ldz, int *isuppz) {
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
  g_cheevr_bridge_calls += 1;

  if (range != 'A' || il != 1 || iu != n) {
    return -424;
  }
  if (m) {
    *m = n;
  }
  return 0;
}

static int stub_zheevx_base(fb_layout_t layout, char jobz, char range,
                            fb_uplo_t uplo, int n,
                            fb_complex_double_t *a, int lda, double vl,
                            double vu, int il, int iu, double abstol, int *m,
                            double *w, fb_complex_double_t *z, int ldz,
                            int *ifail) {
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
  g_zheevx_bridge_calls += 1;

  if (range != 'A' || il != 1 || iu != n) {
    return -525;
  }
  if (m) {
    *m = n;
  }
  return 0;
}

static int stub_cheevx_base(fb_layout_t layout, char jobz, char range,
                            fb_uplo_t uplo, int n,
                            fb_complex_float_t *a, int lda, float vl,
                            float vu, int il, int iu, float abstol, int *m,
                            float *w, fb_complex_float_t *z, int ldz,
                            int *ifail) {
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
  g_cheevx_bridge_calls += 1;

  if (range != 'A' || il != 1 || iu != n) {
    return -526;
  }
  if (m) {
    *m = n;
  }
  return 0;
}

static int stub_zheevr_base(fb_layout_t layout, char jobz, char range,
                            fb_uplo_t uplo, int n,
                            fb_complex_double_t *a, int lda, double vl,
                            double vu, int il, int iu, double abstol, int *m,
                            double *w, fb_complex_double_t *z, int ldz,
                            int *isuppz) {
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
  g_zheevr_bridge_calls += 1;

  if (range != 'A' || il != 1 || iu != n) {
    return -527;
  }
  if (m) {
    *m = n;
  }
  return 0;
}

static int stub_csygvx_base(fb_layout_t layout, int itype, char jobz,
                            char range, fb_uplo_t uplo, int n,
                            fb_complex_float_t *a, int lda,
                            fb_complex_float_t *b, int ldb, float vl, float vu,
                            int il, int iu, float abstol, int *m, float *w,
                            fb_complex_float_t *z, int ldz, int *ifail) {
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
  g_csygvx_bridge_calls += 1;

  if (range != 'A' || il != 1 || iu != n) {
    return -626;
  }
  if (m) {
    *m = n;
  }
  return 0;
}

static int stub_zsygvx_base(fb_layout_t layout, int itype, char jobz,
                            char range, fb_uplo_t uplo, int n,
                            fb_complex_double_t *a, int lda,
                            fb_complex_double_t *b, int ldb, double vl,
                            double vu, int il, int iu, double abstol, int *m,
                            double *w, fb_complex_double_t *z, int ldz,
                            int *ifail) {
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
  g_zsygvx_bridge_calls += 1;

  if (range != 'A' || il != 1 || iu != n) {
    return -727;
  }
  if (m) {
    *m = n;
  }
  return 0;
}

static int stub_chegvx_base(fb_layout_t layout, int itype, char jobz,
                            char range, fb_uplo_t uplo, int n,
                            fb_complex_float_t *a, int lda,
                            fb_complex_float_t *b, int ldb, float vl, float vu,
                            int il, int iu, float abstol, int *m, float *w,
                            fb_complex_float_t *z, int ldz, int *ifail) {
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
  g_chegvx_bridge_calls += 1;

  if (range != 'A' || il != 1 || iu != n) {
    return -728;
  }
  if (m) {
    *m = n;
  }
  return 0;
}

static int stub_zhegvx_base(fb_layout_t layout, int itype, char jobz,
                            char range, fb_uplo_t uplo, int n,
                            fb_complex_double_t *a, int lda,
                            fb_complex_double_t *b, int ldb, double vl,
                            double vu, int il, int iu, double abstol, int *m,
                            double *w, fb_complex_double_t *z, int ldz,
                            int *ifail) {
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
  g_zhegvx_bridge_calls += 1;

  if (range != 'A' || il != 1 || iu != n) {
    return -729;
  }
  if (m) {
    *m = n;
  }
  return 0;
}

static int stub_chegv_primary(fb_layout_t layout, int itype, char jobz,
                              fb_uplo_t uplo, int n,
                              fb_complex_float_t *a, int lda,
                              fb_complex_float_t *b, int ldb, float *w) {
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
  g_c_prec_primary_calls += 1;
  return 717;
}

static int stub_csygvd_secondary(fb_layout_t layout, int itype, char jobz,
                                 fb_uplo_t uplo, int n,
                                 fb_complex_float_t *a, int lda,
                                 fb_complex_float_t *b, int ldb, float *w) {
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
  g_c_prec_secondary_calls += 1;
  return 818;
}

static int test_chegvd_donor_fills_alias_family(void) {
  fb_backend_vtable_t vtable;
  fb_chegv_cblas_fn hegv_call = NULL;
  fb_chegv_cblas_fn sygv_call = NULL;
  fb_status_t status;

  fb_complex_float_t a[4] = {0};
  fb_complex_float_t b[4] = {0};
  float w[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_c_calls = 0;

  vtable.ext_ops[FB_OP_CHEGVD][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_chegvd_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  if (vtable.ext_ops[FB_OP_CHEGV][FB_CONV_CBLAS] !=
      vtable.ext_ops[FB_OP_CHEGVD][FB_CONV_CBLAS] ||
      vtable.ext_ops[FB_OP_CSYGV][FB_CONV_CBLAS] !=
          vtable.ext_ops[FB_OP_CHEGVD][FB_CONV_CBLAS] ||
      vtable.ext_ops[FB_OP_CSYGVD][FB_CONV_CBLAS] !=
          vtable.ext_ops[FB_OP_CHEGVD][FB_CONV_CBLAS]) {
    fprintf(stderr,
            "[FAIL] CHEGVD donor did not fill full complex-single alias family\n");
    return 1;
  }

  hegv_call = (fb_chegv_cblas_fn)vtable.ext_ops[FB_OP_CHEGV][FB_CONV_CBLAS];
  sygv_call = (fb_chegv_cblas_fn)vtable.ext_ops[FB_OP_CSYGV][FB_CONV_CBLAS];
  if (!hegv_call || !sygv_call) {
    fprintf(stderr, "[FAIL] CHEGV/CSYGV CBLAS slots missing\n");
    return 1;
  }

  if (hegv_call(FB_LAYOUT_COL_MAJOR, 1, 'N', FB_UPPER, 2, a, 2, b, 2, w) !=
      515) {
    fprintf(stderr, "[FAIL] CHEGV routed call did not invoke donor\n");
    return 1;
  }

  if (sygv_call(FB_LAYOUT_COL_MAJOR, 1, 'N', FB_UPPER, 2, a, 2, b, 2, w) !=
      515) {
    fprintf(stderr, "[FAIL] CSYGV routed call did not invoke donor\n");
    return 1;
  }

  if (g_c_calls != 2) {
    fprintf(stderr, "[FAIL] CHEGV/CSYGV call accounting mismatch (calls=%d)\n",
            g_c_calls);
    return 1;
  }

  printf("[PASS] CHEGVD donor auto-fills CHEGV/CSYGV complex-single alias family\n");
  return 0;
}

static int test_zhegvd_donor_fills_alias_family(void) {
  fb_backend_vtable_t vtable;
  fb_zhegv_cblas_fn hegv_call = NULL;
  fb_zhegv_cblas_fn sygv_call = NULL;
  fb_status_t status;

  fb_complex_double_t a[4] = {0};
  fb_complex_double_t b[4] = {0};
  double w[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_z_calls = 0;

  vtable.ext_ops[FB_OP_ZHEGVD][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zhegvd_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  if (vtable.ext_ops[FB_OP_ZHEGV][FB_CONV_CBLAS] !=
      vtable.ext_ops[FB_OP_ZHEGVD][FB_CONV_CBLAS] ||
      vtable.ext_ops[FB_OP_ZSYGV][FB_CONV_CBLAS] !=
          vtable.ext_ops[FB_OP_ZHEGVD][FB_CONV_CBLAS] ||
      vtable.ext_ops[FB_OP_ZSYGVD][FB_CONV_CBLAS] !=
          vtable.ext_ops[FB_OP_ZHEGVD][FB_CONV_CBLAS]) {
    fprintf(stderr,
            "[FAIL] ZHEGVD donor did not fill full complex-double alias family\n");
    return 1;
  }

  hegv_call = (fb_zhegv_cblas_fn)vtable.ext_ops[FB_OP_ZHEGV][FB_CONV_CBLAS];
  sygv_call = (fb_zhegv_cblas_fn)vtable.ext_ops[FB_OP_ZSYGV][FB_CONV_CBLAS];
  if (!hegv_call || !sygv_call) {
    fprintf(stderr, "[FAIL] ZHEGV/ZSYGV CBLAS slots missing\n");
    return 1;
  }

  if (hegv_call(FB_LAYOUT_COL_MAJOR, 1, 'N', FB_UPPER, 2, a, 2, b, 2, w) !=
      616) {
    fprintf(stderr, "[FAIL] ZHEGV routed call did not invoke donor\n");
    return 1;
  }

  if (sygv_call(FB_LAYOUT_COL_MAJOR, 1, 'N', FB_UPPER, 2, a, 2, b, 2, w) !=
      616) {
    fprintf(stderr, "[FAIL] ZSYGV routed call did not invoke donor\n");
    return 1;
  }

  if (g_z_calls != 2) {
    fprintf(stderr, "[FAIL] ZHEGV/ZSYGV call accounting mismatch (calls=%d)\n",
            g_z_calls);
    return 1;
  }

  printf("[PASS] ZHEGVD donor auto-fills ZHEGV/ZSYGV complex-double alias family\n");
  return 0;
}

static int test_complex_alias_precedence_prefers_first_present_slot(void) {
  fb_backend_vtable_t vtable;
  fb_chegv_cblas_fn chegv_call = NULL;
  fb_chegv_cblas_fn csygv_call = NULL;
  fb_status_t status;

  fb_complex_float_t a[4] = {0};
  fb_complex_float_t b[4] = {0};
  float w[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_c_prec_primary_calls = 0;
  g_c_prec_secondary_calls = 0;

  /* Group order is {CHEGV, CSYGV, CHEGVD, CSYGVD}; donor should be CHEGV. */
  vtable.ext_ops[FB_OP_CHEGV][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_chegv_primary;
  vtable.ext_ops[FB_OP_CSYGVD][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_csygvd_secondary;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  if (vtable.ext_ops[FB_OP_CSYGV][FB_CONV_CBLAS] !=
      vtable.ext_ops[FB_OP_CHEGV][FB_CONV_CBLAS] ||
      vtable.ext_ops[FB_OP_CHEGVD][FB_CONV_CBLAS] !=
          vtable.ext_ops[FB_OP_CHEGV][FB_CONV_CBLAS]) {
    fprintf(stderr,
            "[FAIL] precedence fill did not use first compatible donor slot\n");
    return 1;
  }

  chegv_call = (fb_chegv_cblas_fn)vtable.ext_ops[FB_OP_CHEGV][FB_CONV_CBLAS];
  csygv_call = (fb_chegv_cblas_fn)vtable.ext_ops[FB_OP_CSYGV][FB_CONV_CBLAS];
  if (!chegv_call || !csygv_call) {
    fprintf(stderr, "[FAIL] expected CHEGV/CSYGV slots missing\n");
    return 1;
  }

  if (chegv_call(FB_LAYOUT_COL_MAJOR, 1, 'N', FB_UPPER, 2, a, 2, b, 2, w) !=
      717) {
    fprintf(stderr, "[FAIL] CHEGV call did not use primary donor\n");
    return 1;
  }
  if (csygv_call(FB_LAYOUT_COL_MAJOR, 1, 'N', FB_UPPER, 2, a, 2, b, 2, w) !=
      717) {
    fprintf(stderr, "[FAIL] CSYGV call did not use primary donor\n");
    return 1;
  }

  if (g_c_prec_primary_calls != 2 || g_c_prec_secondary_calls != 0) {
    fprintf(stderr,
            "[FAIL] precedence accounting mismatch (primary=%d secondary=%d)\n",
            g_c_prec_primary_calls, g_c_prec_secondary_calls);
    return 1;
  }

  printf("[PASS] alias precedence prefers first compatible populated slot\n");
  return 0;
}

static int stub_chesv_base(fb_layout_t layout, fb_uplo_t uplo, int n, int nrhs,
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
  g_solver_alias_calls += 1;
  return 919;
}

static int test_solver_alias_chesv_to_csysv(void) {
  fb_backend_vtable_t vtable;
  fb_chesv_cblas_fn csysv_call = NULL;
  fb_status_t status;

  fb_complex_float_t a[4] = {0};
  fb_complex_float_t b[4] = {0};
  int ipiv[2] = {0, 0};

  memset(&vtable, 0, sizeof(vtable));
  g_solver_alias_calls = 0;

  vtable.ext_ops[FB_OP_CHESV][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_chesv_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  if (vtable.ext_ops[FB_OP_CSYSV][FB_CONV_CBLAS] !=
      vtable.ext_ops[FB_OP_CHESV][FB_CONV_CBLAS]) {
    fprintf(stderr, "[FAIL] CHESV donor did not fill CSYSV alias slot\n");
    return 1;
  }

  csysv_call = (fb_chesv_cblas_fn)vtable.ext_ops[FB_OP_CSYSV][FB_CONV_CBLAS];
  if (!csysv_call) {
    fprintf(stderr, "[FAIL] CSYSV slot missing after compatibility fill\n");
    return 1;
  }

  if (csysv_call(FB_LAYOUT_COL_MAJOR, FB_UPPER, 2, 2, a, 2, ipiv, b, 2) !=
      919) {
    fprintf(stderr, "[FAIL] CSYSV routed call did not invoke CHESV donor\n");
    return 1;
  }

  if (g_solver_alias_calls != 1) {
    fprintf(stderr,
            "[FAIL] solver alias call accounting mismatch (calls=%d)\n",
            g_solver_alias_calls);
    return 1;
  }

  printf("[PASS] CHESV donor auto-fills CSYSV solver alias slot\n");
  return 0;
}

static int test_adapter_bridge_cheev_from_cheevr(void) {
  fb_backend_vtable_t vtable;
  fb_cheev_fn cheev_call = NULL;
  fb_status_t status;

  fb_complex_float_t a[4] = {0};
  float w[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_cheevr_bridge_calls = 0;

  vtable.ext_ops[FB_OP_CHEEVR][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_cheevr_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  cheev_call = (fb_cheev_fn)vtable.ext_ops[FB_OP_CHEEV][FB_CONV_CBLAS];
  if (!cheev_call) {
    fprintf(stderr, "[FAIL] CHEEV adapter slot missing after finalize\n");
    return 1;
  }

  if (cheev_call(FB_LAYOUT_COL_MAJOR, 'V', FB_UPPER, 2, a, 2, w) != 0) {
    fprintf(stderr, "[FAIL] CHEEV adapter bridge returned error\n");
    return 1;
  }

  if (g_cheevr_bridge_calls != 1) {
    fprintf(stderr,
            "[FAIL] CHEEV<-CHEEVR adapter call count mismatch (calls=%d)\n",
            g_cheevr_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter bridge CHEEV <- CHEEVR applies default spectrum params\n");
  return 0;
}

static int test_adapter_bridge_zheev_from_zheevx(void) {
  fb_backend_vtable_t vtable;
  fb_zheev_fn zheev_call = NULL;
  fb_status_t status;

  fb_complex_double_t a[4] = {0};
  double w[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_zheevx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_ZHEEVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zheevx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  zheev_call = (fb_zheev_fn)vtable.ext_ops[FB_OP_ZHEEV][FB_CONV_CBLAS];
  if (!zheev_call) {
    fprintf(stderr, "[FAIL] ZHEEV adapter slot missing after finalize\n");
    return 1;
  }

  if (zheev_call(FB_LAYOUT_COL_MAJOR, 'V', FB_UPPER, 2, a, 2, w) != 0) {
    fprintf(stderr, "[FAIL] ZHEEV adapter bridge returned error\n");
    return 1;
  }

  if (g_zheevx_bridge_calls != 1) {
    fprintf(stderr,
            "[FAIL] ZHEEV<-ZHEEVX adapter call count mismatch (calls=%d)\n",
            g_zheevx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter bridge ZHEEV <- ZHEEVX applies default spectrum params\n");
  return 0;
}

static int test_adapter_bridge_cheev_from_cheevx(void) {
  fb_backend_vtable_t vtable;
  fb_cheev_fn cheev_call = NULL;
  fb_status_t status;

  fb_complex_float_t a[4] = {0};
  float w[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_cheevx_bridge_calls = 0;

  /* No CHEEVR donor: adapter should fallback to CHEEVX source. */
  vtable.ext_ops[FB_OP_CHEEVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_cheevx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  cheev_call = (fb_cheev_fn)vtable.ext_ops[FB_OP_CHEEV][FB_CONV_CBLAS];
  if (!cheev_call) {
    fprintf(stderr, "[FAIL] CHEEV adapter slot missing after finalize\n");
    return 1;
  }

  if (cheev_call(FB_LAYOUT_COL_MAJOR, 'V', FB_UPPER, 2, a, 2, w) != 0) {
    fprintf(stderr, "[FAIL] CHEEV adapter bridge from CHEEVX returned error\n");
    return 1;
  }

  if (g_cheevx_bridge_calls != 1) {
    fprintf(stderr,
            "[FAIL] CHEEV<-CHEEVX adapter call count mismatch (calls=%d)\n",
            g_cheevx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter bridge CHEEV <- CHEEVX applies default spectrum params\n");
  return 0;
}

static int test_adapter_bridge_zheev_from_zheevr(void) {
  fb_backend_vtable_t vtable;
  fb_zheev_fn zheev_call = NULL;
  fb_status_t status;

  fb_complex_double_t a[4] = {0};
  double w[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_zheevr_bridge_calls = 0;

  vtable.ext_ops[FB_OP_ZHEEVR][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zheevr_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  zheev_call = (fb_zheev_fn)vtable.ext_ops[FB_OP_ZHEEV][FB_CONV_CBLAS];
  if (!zheev_call) {
    fprintf(stderr, "[FAIL] ZHEEV adapter slot missing after finalize\n");
    return 1;
  }

  if (zheev_call(FB_LAYOUT_COL_MAJOR, 'V', FB_UPPER, 2, a, 2, w) != 0) {
    fprintf(stderr, "[FAIL] ZHEEV adapter bridge from ZHEEVR returned error\n");
    return 1;
  }

  if (g_zheevr_bridge_calls != 1) {
    fprintf(stderr,
            "[FAIL] ZHEEV<-ZHEEVR adapter call count mismatch (calls=%d)\n",
            g_zheevr_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter bridge ZHEEV <- ZHEEVR applies default spectrum params\n");
  return 0;
}

static int test_adapter_bridge_chegv_from_csygvx_alias(void) {
  fb_backend_vtable_t vtable;
  fb_chegv_cblas_fn chegv_call = NULL;
  fb_status_t status;

  fb_complex_float_t a[4] = {0};
  fb_complex_float_t b[4] = {0};
  float w[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_csygvx_bridge_calls = 0;

  /* Only CSYGVX donor provided; CHEGVX should be filled by alias map first,
   * then adapter bridge should synthesize CHEGV. */
  vtable.ext_ops[FB_OP_CSYGVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_csygvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  chegv_call = (fb_chegv_cblas_fn)vtable.ext_ops[FB_OP_CHEGV][FB_CONV_CBLAS];
  if (!chegv_call) {
    fprintf(stderr, "[FAIL] CHEGV adapter slot missing after finalize\n");
    return 1;
  }

  if (chegv_call(FB_LAYOUT_COL_MAJOR, 1, 'V', FB_UPPER, 2, a, 2, b, 2, w) !=
      0) {
    fprintf(stderr, "[FAIL] CHEGV adapter bridge from CSYGVX returned error\n");
    return 1;
  }

  if (g_csygvx_bridge_calls != 1) {
    fprintf(stderr,
            "[FAIL] CHEGV<-CSYGVX adapter call mismatch (calls=%d)\n",
            g_csygvx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter bridge CHEGV <- CSYGVX applies default range params\n");
  return 0;
}

static int test_adapter_bridge_zhegv_from_zsygvx_alias(void) {
  fb_backend_vtable_t vtable;
  fb_zhegv_cblas_fn zhegv_call = NULL;
  fb_status_t status;

  fb_complex_double_t a[4] = {0};
  fb_complex_double_t b[4] = {0};
  double w[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_zsygvx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_ZSYGVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zsygvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  zhegv_call = (fb_zhegv_cblas_fn)vtable.ext_ops[FB_OP_ZHEGV][FB_CONV_CBLAS];
  if (!zhegv_call) {
    fprintf(stderr, "[FAIL] ZHEGV adapter slot missing after finalize\n");
    return 1;
  }

  if (zhegv_call(FB_LAYOUT_COL_MAJOR, 1, 'V', FB_UPPER, 2, a, 2, b, 2, w) !=
      0) {
    fprintf(stderr, "[FAIL] ZHEGV adapter bridge from ZSYGVX returned error\n");
    return 1;
  }

  if (g_zsygvx_bridge_calls != 1) {
    fprintf(stderr,
            "[FAIL] ZHEGV<-ZSYGVX adapter call mismatch (calls=%d)\n",
            g_zsygvx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter bridge ZHEGV <- ZSYGVX applies default range params\n");
  return 0;
}

static int test_adapter_bridge_chegv_from_chegvx_preferred(void) {
  fb_backend_vtable_t vtable;
  fb_chegv_cblas_fn chegv_call = NULL;
  fb_status_t status;

  fb_complex_float_t a[4] = {0};
  fb_complex_float_t b[4] = {0};
  float w[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_chegvx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_CHEGVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_chegvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  chegv_call = (fb_chegv_cblas_fn)vtable.ext_ops[FB_OP_CHEGV][FB_CONV_CBLAS];
  if (!chegv_call) {
    fprintf(stderr, "[FAIL] CHEGV adapter slot missing after finalize\n");
    return 1;
  }

  if (chegv_call(FB_LAYOUT_COL_MAJOR, 1, 'V', FB_UPPER, 2, a, 2, b, 2, w) !=
      0) {
    fprintf(stderr, "[FAIL] CHEGV adapter bridge from CHEGVX returned error\n");
    return 1;
  }

  if (g_chegvx_bridge_calls != 1) {
    fprintf(stderr,
            "[FAIL] CHEGV<-CHEGVX adapter call mismatch (calls=%d)\n",
            g_chegvx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter bridge CHEGV <- CHEGVX applies default range params\n");
  return 0;
}

static int test_adapter_bridge_zhegv_from_zhegvx_preferred(void) {
  fb_backend_vtable_t vtable;
  fb_zhegv_cblas_fn zhegv_call = NULL;
  fb_status_t status;

  fb_complex_double_t a[4] = {0};
  fb_complex_double_t b[4] = {0};
  double w[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_zhegvx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_ZHEGVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zhegvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  zhegv_call = (fb_zhegv_cblas_fn)vtable.ext_ops[FB_OP_ZHEGV][FB_CONV_CBLAS];
  if (!zhegv_call) {
    fprintf(stderr, "[FAIL] ZHEGV adapter slot missing after finalize\n");
    return 1;
  }

  if (zhegv_call(FB_LAYOUT_COL_MAJOR, 1, 'V', FB_UPPER, 2, a, 2, b, 2, w) !=
      0) {
    fprintf(stderr, "[FAIL] ZHEGV adapter bridge from ZHEGVX returned error\n");
    return 1;
  }

  if (g_zhegvx_bridge_calls != 1) {
    fprintf(stderr,
            "[FAIL] ZHEGV<-ZHEGVX adapter call mismatch (calls=%d)\n",
            g_zhegvx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter bridge ZHEGV <- ZHEGVX applies default range params\n");
  return 0;
}

static int test_adapter_precedence_cheev_prefers_cheevr(void) {
  fb_backend_vtable_t vtable;
  fb_cheev_fn cheev_call = NULL;
  fb_status_t status;

  fb_complex_float_t a[4] = {0};
  float w[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_cheevr_bridge_calls = 0;
  g_cheevx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_CHEEVR][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_cheevr_base;
  vtable.ext_ops[FB_OP_CHEEVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_cheevx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  cheev_call = (fb_cheev_fn)vtable.ext_ops[FB_OP_CHEEV][FB_CONV_CBLAS];
  if (!cheev_call) {
    fprintf(stderr, "[FAIL] CHEEV adapter slot missing after finalize\n");
    return 1;
  }

  if (cheev_call(FB_LAYOUT_COL_MAJOR, 'V', FB_UPPER, 2, a, 2, w) != 0) {
    fprintf(stderr, "[FAIL] CHEEV precedence bridge returned error\n");
    return 1;
  }

  if (g_cheevr_bridge_calls != 1 || g_cheevx_bridge_calls != 0) {
    fprintf(stderr,
            "[FAIL] CHEEV adapter precedence mismatch (cheevr=%d cheevx=%d)\n",
            g_cheevr_bridge_calls, g_cheevx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter precedence CHEEV prefers CHEEVR over CHEEVX\n");
  return 0;
}

static int test_adapter_precedence_zhegv_prefers_zhegvx(void) {
  fb_backend_vtable_t vtable;
  fb_zhegv_cblas_fn zhegv_call = NULL;
  fb_status_t status;

  fb_complex_double_t a[4] = {0};
  fb_complex_double_t b[4] = {0};
  double w[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_zhegvx_bridge_calls = 0;
  g_zsygvx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_ZHEGVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zhegvx_base;
  vtable.ext_ops[FB_OP_ZSYGVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zsygvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  zhegv_call = (fb_zhegv_cblas_fn)vtable.ext_ops[FB_OP_ZHEGV][FB_CONV_CBLAS];
  if (!zhegv_call) {
    fprintf(stderr, "[FAIL] ZHEGV adapter slot missing after finalize\n");
    return 1;
  }

  if (zhegv_call(FB_LAYOUT_COL_MAJOR, 1, 'V', FB_UPPER, 2, a, 2, b, 2, w) !=
      0) {
    fprintf(stderr, "[FAIL] ZHEGV precedence bridge returned error\n");
    return 1;
  }

  if (g_zhegvx_bridge_calls != 1 || g_zsygvx_bridge_calls != 0) {
    fprintf(stderr,
            "[FAIL] ZHEGV adapter precedence mismatch (zhegvx=%d zsygvx=%d)\n",
            g_zhegvx_bridge_calls, g_zsygvx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter precedence ZHEGV prefers ZHEGVX over ZSYGVX\n");
  return 0;
}

static int test_adapter_precedence_zheev_prefers_zheevr(void) {
  fb_backend_vtable_t vtable;
  fb_zheev_fn zheev_call = NULL;
  fb_status_t status;

  fb_complex_double_t a[4] = {0};
  double w[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_zheevr_bridge_calls = 0;
  g_zheevx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_ZHEEVR][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zheevr_base;
  vtable.ext_ops[FB_OP_ZHEEVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_zheevx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  zheev_call = (fb_zheev_fn)vtable.ext_ops[FB_OP_ZHEEV][FB_CONV_CBLAS];
  if (!zheev_call) {
    fprintf(stderr, "[FAIL] ZHEEV adapter slot missing after finalize\n");
    return 1;
  }

  if (zheev_call(FB_LAYOUT_COL_MAJOR, 'V', FB_UPPER, 2, a, 2, w) != 0) {
    fprintf(stderr, "[FAIL] ZHEEV precedence bridge returned error\n");
    return 1;
  }

  if (g_zheevr_bridge_calls != 1 || g_zheevx_bridge_calls != 0) {
    fprintf(stderr,
            "[FAIL] ZHEEV adapter precedence mismatch (zheevr=%d zheevx=%d)\n",
            g_zheevr_bridge_calls, g_zheevx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter precedence ZHEEV prefers ZHEEVR over ZHEEVX\n");
  return 0;
}

static int test_adapter_precedence_chegv_prefers_chegvx(void) {
  fb_backend_vtable_t vtable;
  fb_chegv_cblas_fn chegv_call = NULL;
  fb_status_t status;

  fb_complex_float_t a[4] = {0};
  fb_complex_float_t b[4] = {0};
  float w[2] = {0};

  memset(&vtable, 0, sizeof(vtable));
  g_chegvx_bridge_calls = 0;
  g_csygvx_bridge_calls = 0;

  vtable.ext_ops[FB_OP_CHEGVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_chegvx_base;
  vtable.ext_ops[FB_OP_CSYGVX][FB_CONV_CBLAS] =
      (fb_generic_fn)(void (*)(void))stub_csygvx_base;

  status = fb_finalize_plugin_vtable(&vtable);
  if (status != FB_STATUS_SUCCESS) {
    fprintf(stderr, "[FAIL] finalize status=%d\n", status);
    return 1;
  }

  chegv_call = (fb_chegv_cblas_fn)vtable.ext_ops[FB_OP_CHEGV][FB_CONV_CBLAS];
  if (!chegv_call) {
    fprintf(stderr, "[FAIL] CHEGV adapter slot missing after finalize\n");
    return 1;
  }

  if (chegv_call(FB_LAYOUT_COL_MAJOR, 1, 'V', FB_UPPER, 2, a, 2, b, 2, w) !=
      0) {
    fprintf(stderr, "[FAIL] CHEGV precedence bridge returned error\n");
    return 1;
  }

  if (g_chegvx_bridge_calls != 1 || g_csygvx_bridge_calls != 0) {
    fprintf(stderr,
            "[FAIL] CHEGV adapter precedence mismatch (chegvx=%d csygvx=%d)\n",
            g_chegvx_bridge_calls, g_csygvx_bridge_calls);
    return 1;
  }

  printf("[PASS] adapter precedence CHEGV prefers CHEGVX over CSYGVX\n");
  return 0;
}

int main(void) {
  int status = 0;

  status |= test_chegvd_donor_fills_alias_family();
  status |= test_zhegvd_donor_fills_alias_family();
  status |= test_complex_alias_precedence_prefers_first_present_slot();
  status |= test_solver_alias_chesv_to_csysv();
  status |= test_adapter_bridge_cheev_from_cheevr();
  status |= test_adapter_bridge_cheev_from_cheevx();
  status |= test_adapter_bridge_zheev_from_zheevx();
  status |= test_adapter_bridge_zheev_from_zheevr();
  status |= test_adapter_bridge_chegv_from_csygvx_alias();
  status |= test_adapter_bridge_chegv_from_chegvx_preferred();
  status |= test_adapter_bridge_zhegv_from_zsygvx_alias();
  status |= test_adapter_bridge_zhegv_from_zhegvx_preferred();
  status |= test_adapter_precedence_cheev_prefers_cheevr();
  status |= test_adapter_precedence_zheev_prefers_zheevr();
  status |= test_adapter_precedence_chegv_prefers_chegvx();
  status |= test_adapter_precedence_zhegv_prefers_zhegvx();

  if (status != 0) {
    fprintf(stderr, "Result: FAIL\n");
    return 1;
  }

  printf("Result: PASS\n");
  return 0;
}
