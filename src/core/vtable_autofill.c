/**
 * @file vtable_autofill.c
 * @brief Implementation of automatic vtable completion system
 *
 * @copyright Copyright (c) 2025-2026
 * @license MIT OR Apache-2.0
 */

#include "../../include/faster-blaster/vtable_autofill.h"
#include "../backends/backend_interface.h"
#include "conv_thunks.h"
#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Global active vtable pointer */
static const fb_backend_vtable_t *g_active_vtable = NULL;

/* ============================================================================
 * Strategy 0: Compatibility-Group Autofill (pilot)
 * ========================================================================== */

typedef struct fb_compat_group_entry {
  uint32_t op_ids[8];
  size_t count;
} fb_compat_group_entry_t;

#define FB_COMPAT_GROUP2(a, b) {{(a), (b)}, 2}
#define FB_COMPAT_GROUP4(a, b, c, d) {{(a), (b), (c), (d)}, 4}

static void fb_autofill_compat_group_ext_ops(fb_backend_vtable_t *vtable,
                                             const fb_compat_group_entry_t *g) {
  fb_generic_fn donor = NULL;

  for (size_t i = 0; i < g->count; i++) {
    uint32_t op = g->op_ids[i];
    fb_generic_fn fn = vtable->ext_ops[op][FB_CONV_CBLAS];
    if (fn != NULL) {
      donor = fn;
      break;
    }
  }

  if (donor == NULL) {
    return;
  }

  for (size_t i = 0; i < g->count; i++) {
    uint32_t op = g->op_ids[i];
    if (vtable->ext_ops[op][FB_CONV_CBLAS] == NULL) {
      vtable->ext_ops[op][FB_CONV_CBLAS] = donor;
    }
  }
}

static fb_status_t fb_autofill_compatibility_groups(fb_backend_vtable_t *vtable) {
  static const fb_compat_group_entry_t k_groups[] = {
      /* Dense symmetric/Hermitian eigensolvers (same compact CBLAS signature). */
      FB_COMPAT_GROUP2(FB_OP_SSYEV, FB_OP_SSYEVD),
      FB_COMPAT_GROUP2(FB_OP_DSYEV, FB_OP_DSYEVD),
      FB_COMPAT_GROUP2(FB_OP_CHEEV, FB_OP_CHEEVD),
      FB_COMPAT_GROUP2(FB_OP_ZHEEV, FB_OP_ZHEEVD),

      /* Packed symmetric/Hermitian eigensolvers. */
      FB_COMPAT_GROUP2(FB_OP_SSPEV, FB_OP_SSPEVD),
      FB_COMPAT_GROUP2(FB_OP_DSPEV, FB_OP_DSPEVD),
      FB_COMPAT_GROUP2(FB_OP_CHPEV, FB_OP_CHPEVD),
      FB_COMPAT_GROUP2(FB_OP_ZHPEV, FB_OP_ZHPEVD),

      /* Banded symmetric/Hermitian eigensolvers. */
      FB_COMPAT_GROUP2(FB_OP_SSBEV, FB_OP_SSBEVD),
      FB_COMPAT_GROUP2(FB_OP_DSBEV, FB_OP_DSBEVD),
      FB_COMPAT_GROUP2(FB_OP_CHBEV, FB_OP_CHBEVD),
      FB_COMPAT_GROUP2(FB_OP_ZHBEV, FB_OP_ZHBEVD),

      /* Generalized packed symmetric/Hermitian eigensolvers. */
      FB_COMPAT_GROUP2(FB_OP_SSPGV, FB_OP_SSPGVD),
      FB_COMPAT_GROUP2(FB_OP_DSPGV, FB_OP_DSPGVD),
      FB_COMPAT_GROUP2(FB_OP_CHPGV, FB_OP_CHPGVD),
      FB_COMPAT_GROUP2(FB_OP_ZHPGV, FB_OP_ZHPGVD),

      /* Generalized banded symmetric/Hermitian eigensolvers. */
      FB_COMPAT_GROUP2(FB_OP_SSBGV, FB_OP_SSBGVD),
      FB_COMPAT_GROUP2(FB_OP_DSBGV, FB_OP_DSBGVD),
      FB_COMPAT_GROUP2(FB_OP_CHBGV, FB_OP_CHBGVD),
      FB_COMPAT_GROUP2(FB_OP_ZHBGV, FB_OP_ZHBGVD),

      /* Dense generalized symmetric/Hermitian eigensolvers. */
      FB_COMPAT_GROUP2(FB_OP_SSYGV, FB_OP_SSYGVD),
      FB_COMPAT_GROUP2(FB_OP_DSYGV, FB_OP_DSYGVD),

          /* Complex generalized aliases (HE* and SY* names are interchangeable in
           * this API): allow a single impl to satisfy all compatible slots. */
      FB_COMPAT_GROUP4(FB_OP_CHEGV, FB_OP_CSYGV, FB_OP_CHEGVD, FB_OP_CSYGVD),
      FB_COMPAT_GROUP4(FB_OP_ZHEGV, FB_OP_ZSYGV, FB_OP_ZHEGVD, FB_OP_ZSYGVD),

          /* EVR/EVX-style dense eigensolver variants (same selected-spectrum
           * interface, algorithm differs). */
      FB_COMPAT_GROUP2(FB_OP_SSYEVR, FB_OP_SSYEVX),
      FB_COMPAT_GROUP2(FB_OP_DSYEVR, FB_OP_DSYEVX),
      FB_COMPAT_GROUP2(FB_OP_CHEEVR, FB_OP_CHEEVX),
      FB_COMPAT_GROUP2(FB_OP_ZHEEVR, FB_OP_ZHEEVX),

          /* Generalized EVX alias families. */
      FB_COMPAT_GROUP2(FB_OP_CHEGVX, FB_OP_CSYGVX),
      FB_COMPAT_GROUP2(FB_OP_ZHEGVX, FB_OP_ZSYGVX),

            /* Hermitian/symmetric complex solver-condition aliases. */
            FB_COMPAT_GROUP2(FB_OP_CHECON, FB_OP_CSYCON),
            FB_COMPAT_GROUP2(FB_OP_ZHECON, FB_OP_ZSYCON),
            FB_COMPAT_GROUP2(FB_OP_CHESV, FB_OP_CSYSV),
            FB_COMPAT_GROUP2(FB_OP_ZHESV, FB_OP_ZSYSV),
            FB_COMPAT_GROUP2(FB_OP_CHESVX, FB_OP_CSYSVX),
            FB_COMPAT_GROUP2(FB_OP_ZHESVX, FB_OP_ZSYSVX),
            FB_COMPAT_GROUP2(FB_OP_CHESVXX, FB_OP_CSYSVXX),
            FB_COMPAT_GROUP2(FB_OP_ZHESVXX, FB_OP_ZSYSVXX),
  };

  if (!vtable) {
    return FB_STATUS_INVALID_ARGUMENT;
  }

  for (size_t i = 0; i < sizeof(k_groups) / sizeof(k_groups[0]); i++) {
    fb_autofill_compat_group_ext_ops(vtable, &k_groups[i]);
  }

  return FB_STATUS_SUCCESS;
}

/* ============================================================================
 * Strategy 0c: Superset/Subset Adapters (default-parameter bridges)
 * ========================================================================== */

typedef int (*fb_adapter_ssyev_fn)(fb_layout_t, char, fb_uplo_t, int, float *,
                                   int, float *);
typedef int (*fb_adapter_dsyev_fn)(fb_layout_t, char, fb_uplo_t, int, double *,
                                   int, double *);
typedef int (*fb_adapter_cheev_fn)(fb_layout_t, char, fb_uplo_t, int,
                                   fb_complex_float_t *, int, float *);
typedef int (*fb_adapter_zheev_fn)(fb_layout_t, char, fb_uplo_t, int,
                                   fb_complex_double_t *, int, double *);

typedef int (*fb_adapter_ssyevr_fn)(fb_layout_t, char, char, fb_uplo_t, int,
                                    float *, int, float, float, int, int,
                                    float, int *, float *, float *, int,
                                    int *);
typedef int (*fb_adapter_dsyevr_fn)(fb_layout_t, char, char, fb_uplo_t, int,
                                    double *, int, double, double, int, int,
                                    double, int *, double *, double *, int,
                                    int *);
typedef int (*fb_adapter_ssyevx_fn)(fb_layout_t, char, char, fb_uplo_t, int,
                                    float *, int, float, float, int, int,
                                    float, int *, float *, float *, int,
                                    int *);
typedef int (*fb_adapter_dsyevx_fn)(fb_layout_t, char, char, fb_uplo_t, int,
                                    double *, int, double, double, int, int,
                                    double, int *, double *, double *, int,
                                    int *);
typedef int (*fb_adapter_cheevr_fn)(fb_layout_t, char, char, fb_uplo_t, int,
                                    fb_complex_float_t *, int, float, float,
                                    int, int, float, int *, float *,
                                    fb_complex_float_t *, int, int *);
typedef int (*fb_adapter_zheevr_fn)(fb_layout_t, char, char, fb_uplo_t, int,
                                    fb_complex_double_t *, int, double, double,
                                    int, int, double, int *, double *,
                                    fb_complex_double_t *, int, int *);
typedef int (*fb_adapter_cheevx_fn)(fb_layout_t, char, char, fb_uplo_t, int,
                                    fb_complex_float_t *, int, float, float,
                                    int, int, float, int *, float *,
                                    fb_complex_float_t *, int, int *);
typedef int (*fb_adapter_zheevx_fn)(fb_layout_t, char, char, fb_uplo_t, int,
                                    fb_complex_double_t *, int, double, double,
                                    int, int, double, int *, double *,
                                    fb_complex_double_t *, int, int *);

typedef int (*fb_adapter_ssygv_fn)(fb_layout_t, int, char, fb_uplo_t, int,
                                   float *, int, float *, int, float *);
typedef int (*fb_adapter_dsygv_fn)(fb_layout_t, int, char, fb_uplo_t, int,
                                   double *, int, double *, int, double *);
typedef int (*fb_adapter_chegv_fn)(fb_layout_t, int, char, fb_uplo_t, int,
                                   fb_complex_float_t *, int,
                                   fb_complex_float_t *, int, float *);
typedef int (*fb_adapter_zhegv_fn)(fb_layout_t, int, char, fb_uplo_t, int,
                                   fb_complex_double_t *, int,
                                   fb_complex_double_t *, int, double *);

typedef int (*fb_adapter_ssygvx_fn)(fb_layout_t, int, char, char, fb_uplo_t,
                                    int, float *, int, float *, int, float,
                                    float, int, int, float, int *, float *,
                                    float *, int, int *);
typedef int (*fb_adapter_dsygvx_fn)(fb_layout_t, int, char, char, fb_uplo_t,
                                    int, double *, int, double *, int, double,
                                    double, int, int, double, int *, double *,
                                    double *, int, int *);
typedef int (*fb_adapter_chegvx_fn)(fb_layout_t, int, char, char, fb_uplo_t,
                                    int, fb_complex_float_t *, int,
                                    fb_complex_float_t *, int, float, float,
                                    int, int, float, int *, float *,
                                    fb_complex_float_t *, int, int *);
typedef int (*fb_adapter_zhegvx_fn)(fb_layout_t, int, char, char, fb_uplo_t,
                                    int, fb_complex_double_t *, int,
                                    fb_complex_double_t *, int, double, double,
                                    int, int, double, int *, double *,
                                    fb_complex_double_t *, int, int *);

typedef void (*fb_adapter_sstev_fn)(int, char, int, float *, float *, float *,
                                    int, float *, int *);
typedef void (*fb_adapter_dstev_fn)(int, char, int, double *, double *,
                                    double *, int, double *, int *);

typedef int (*fb_adapter_sstevr_fn)(fb_layout_t, char, char, int, float *,
                                    float *, float, float, int, int, float,
                                    int *, float *, float *, int, int *);
typedef int (*fb_adapter_dstevr_fn)(fb_layout_t, char, char, int, double *,
                                    double *, double, double, int, int,
                                    double, int *, double *, double *, int,
                                    int *);
typedef int (*fb_adapter_sstevx_fn)(fb_layout_t, char, char, int, float *,
                                    float *, float, float, int, int, float,
                                    int *, float *, float *, int, int *);
typedef int (*fb_adapter_dstevx_fn)(fb_layout_t, char, char, int, double *,
                                    double *, double, double, int, int,
                                    double, int *, double *, double *, int,
                                    int *);
typedef int (*fb_adapter_sstedc_fn)(fb_layout_t, char, int, float *, float *,
                                    float *, int);
typedef int (*fb_adapter_dstedc_fn)(fb_layout_t, char, int, double *, double *,
                                    double *, int);
typedef int (*fb_adapter_sgeev_fn)(fb_layout_t, char, char, int, float *, int,
                                   float *, float *, float *, int, float *,
                                   int);
typedef int (*fb_adapter_dgeev_fn)(fb_layout_t, char, char, int, double *, int,
                                   double *, double *, double *, int, double *,
                                   int);
typedef int (*fb_adapter_cgeev_fn)(fb_layout_t, char, char, int,
                                   fb_complex_float_t *, int,
                                   fb_complex_float_t *,
                                   fb_complex_float_t *, int,
                                   fb_complex_float_t *, int);
typedef int (*fb_adapter_zgeev_fn)(fb_layout_t, char, char, int,
                                   fb_complex_double_t *, int,
                                   fb_complex_double_t *,
                                   fb_complex_double_t *, int,
                                   fb_complex_double_t *, int);
typedef int (*fb_adapter_sgeevx_fn)(fb_layout_t, char, char, char, char, int,
                                    float *, int, float *, float *, float *,
                                    int, float *, int, int *, int *, float *,
                                    float *, float *, float *);
typedef int (*fb_adapter_dgeevx_fn)(fb_layout_t, char, char, char, char, int,
                                    double *, int, double *, double *,
                                    double *, int, double *, int, int *,
                                    int *, double *, double *, double *,
                                    double *);
typedef int (*fb_adapter_sgesvx_fn)(fb_layout_t, char, char, int, int, float *,
                                    int, float *, int, int *, char *, float *,
                                    float *, float *, int, float *, int,
                                    float *, float *, float *, float *);
typedef int (*fb_adapter_dgesvx_fn)(fb_layout_t, char, char, int, int,
                                    double *, int, double *, int, int *,
                                    char *, double *, double *, double *, int,
                                    double *, int, double *, double *, double *,
                                    double *);
typedef int (*fb_adapter_cgesvx_fn)(fb_layout_t, char, char, int, int,
                                    fb_complex_float_t *, int,
                                    fb_complex_float_t *, int, int *, char *,
                                    float *, float *, fb_complex_float_t *,
                                    int, fb_complex_float_t *, int, float *,
                                    float *, float *, float *);
typedef int (*fb_adapter_zgesvx_fn)(fb_layout_t, char, char, int, int,
                                    fb_complex_double_t *, int,
                                    fb_complex_double_t *, int, int *, char *,
                                    double *, double *, fb_complex_double_t *,
                                    int, fb_complex_double_t *, int, double *,
                                    double *, double *, double *);
typedef int (*fb_adapter_sposvx_fn)(fb_layout_t, char, char, int, int,
                                    float *, int, float *, int, char *,
                                    float *, float *, int, float *, int,
                                    float *, float *, float *);
typedef int (*fb_adapter_dposvx_fn)(fb_layout_t, char, char, int, int,
                                    double *, int, double *, int, char *,
                                    double *, double *, int, double *, int,
                                    double *, double *, double *);
typedef int (*fb_adapter_cposvx_fn)(fb_layout_t, char, char, int, int,
                                    fb_complex_float_t *, int,
                                    fb_complex_float_t *, int, char *, float *,
                                    fb_complex_float_t *, int,
                                    fb_complex_float_t *, int, float *,
                                    float *, float *);
typedef int (*fb_adapter_zposvx_fn)(fb_layout_t, char, char, int, int,
                                    fb_complex_double_t *, int,
                                    fb_complex_double_t *, int, char *,
                                    double *, fb_complex_double_t *, int,
                                    fb_complex_double_t *, int, double *,
                                    double *, double *);
typedef int (*fb_adapter_ssysvx_fn)(fb_layout_t, char, char, int, int,
                                    const float *, int, float *, int, int *,
                                    const float *, int, float *, int, float *,
                                    float *, float *);
typedef int (*fb_adapter_dsysvx_fn)(fb_layout_t, char, char, int, int,
                                    const double *, int, double *, int, int *,
                                    const double *, int, double *, int,
                                    double *, double *, double *);
typedef int (*fb_adapter_csysvx_fn)(fb_layout_t, char, char, int, int,
                                    const fb_complex_float_t *, int,
                                    fb_complex_float_t *, int, int *,
                                    const fb_complex_float_t *, int,
                                    fb_complex_float_t *, int, float *,
                                    float *, float *);
typedef int (*fb_adapter_zsysvx_fn)(fb_layout_t, char, char, int, int,
                                    const fb_complex_double_t *, int,
                                    fb_complex_double_t *, int, int *,
                                    const fb_complex_double_t *, int,
                                    fb_complex_double_t *, int, double *,
                                    double *, double *);
typedef int (*fb_adapter_cgeevx_fn)(fb_layout_t, char, char, char, char, int,
                                    fb_complex_float_t *, int,
                                    fb_complex_float_t *,
                                    fb_complex_float_t *, int,
                                    fb_complex_float_t *, int, int *, int *,
                                    float *, float *, float *, float *);
typedef int (*fb_adapter_zgeevx_fn)(fb_layout_t, char, char, char, char, int,
                                    fb_complex_double_t *, int,
                                    fb_complex_double_t *,
                                    fb_complex_double_t *, int,
                                    fb_complex_double_t *, int, int *, int *,
                                    double *, double *, double *, double *);

/* Single-active-backend model (same limitation pattern as conv_thunks). */
static fb_adapter_ssyevr_fn g_adapter_ssyevr = NULL;
static fb_adapter_dsyevr_fn g_adapter_dsyevr = NULL;
static fb_adapter_ssyevx_fn g_adapter_ssyevx = NULL;
static fb_adapter_dsyevx_fn g_adapter_dsyevx = NULL;
static fb_adapter_cheevr_fn g_adapter_cheevr = NULL;
static fb_adapter_zheevr_fn g_adapter_zheevr = NULL;
static fb_adapter_cheevx_fn g_adapter_cheevx = NULL;
static fb_adapter_zheevx_fn g_adapter_zheevx = NULL;

static fb_adapter_ssygvx_fn g_adapter_ssygvx = NULL;
static fb_adapter_dsygvx_fn g_adapter_dsygvx = NULL;
static fb_adapter_chegvx_fn g_adapter_chegvx = NULL;
static fb_adapter_zhegvx_fn g_adapter_zhegvx = NULL;

static fb_adapter_sstevr_fn g_adapter_sstevr = NULL;
static fb_adapter_dstevr_fn g_adapter_dstevr = NULL;
static fb_adapter_sstevx_fn g_adapter_sstevx = NULL;
static fb_adapter_dstevx_fn g_adapter_dstevx = NULL;
static fb_adapter_sstedc_fn g_adapter_sstedc = NULL;
static fb_adapter_dstedc_fn g_adapter_dstedc = NULL;
static fb_adapter_sgeevx_fn g_adapter_sgeevx = NULL;
static fb_adapter_dgeevx_fn g_adapter_dgeevx = NULL;
static fb_adapter_sgesvx_fn g_adapter_sgesvx = NULL;
static fb_adapter_dgesvx_fn g_adapter_dgesvx = NULL;
static fb_adapter_cgesvx_fn g_adapter_cgesvx = NULL;
static fb_adapter_zgesvx_fn g_adapter_zgesvx = NULL;
static fb_adapter_sposvx_fn g_adapter_sposvx = NULL;
static fb_adapter_dposvx_fn g_adapter_dposvx = NULL;
static fb_adapter_cposvx_fn g_adapter_cposvx = NULL;
static fb_adapter_zposvx_fn g_adapter_zposvx = NULL;
static fb_adapter_ssysvx_fn g_adapter_ssysvx = NULL;
static fb_adapter_dsysvx_fn g_adapter_dsysvx = NULL;
static fb_adapter_csysvx_fn g_adapter_csysvx = NULL;
static fb_adapter_zsysvx_fn g_adapter_zsysvx = NULL;
static fb_adapter_cgeevx_fn g_adapter_cgeevx = NULL;
static fb_adapter_zgeevx_fn g_adapter_zgeevx = NULL;

static int fb_adapter_ssyev_from_ssyevr(fb_layout_t layout, char jobz,
                                        fb_uplo_t uplo, int n, float *A,
                                        int lda, float *w) {
  const fb_backend_vtable_t *ctx = fb_get_active_vtable();
  fb_adapter_ssyevr_fn src =
      ctx ? (fb_adapter_ssyevr_fn)ctx->ext_ops[FB_OP_SSYEVR][FB_CONV_CBLAS]
          : NULL;
  int m = 0;
  int *isuppz = (int *)malloc((size_t)((n > 0) ? (2 * n) : 2) * sizeof(int));
  int rc;
  if (!src || !w || (n > 0 && !A)) {
    free(isuppz);
    return -1;
  }
  if (!isuppz) {
    return -101;
  }
  rc = src(layout, jobz, 'A', uplo, n, A, lda, 0.0f, 0.0f, 1, n, 0.0f, &m, w,
           A, lda, isuppz);
  free(isuppz);
  return rc;
}

static int fb_adapter_dsyev_from_dsyevr(fb_layout_t layout, char jobz,
                                        fb_uplo_t uplo, int n, double *A,
                                        int lda, double *w) {
  const fb_backend_vtable_t *ctx = fb_get_active_vtable();
  fb_adapter_dsyevr_fn src =
      ctx ? (fb_adapter_dsyevr_fn)ctx->ext_ops[FB_OP_DSYEVR][FB_CONV_CBLAS]
          : NULL;
  int m = 0;
  int *isuppz = (int *)malloc((size_t)((n > 0) ? (2 * n) : 2) * sizeof(int));
  int rc;
  if (!src || !w || (n > 0 && !A)) {
    free(isuppz);
    return -1;
  }
  if (!isuppz) {
    return -101;
  }
  rc = src(layout, jobz, 'A', uplo, n, A, lda, 0.0, 0.0, 1, n, 0.0, &m, w, A,
           lda, isuppz);
  free(isuppz);
  return rc;
}

static int fb_adapter_ssyev_from_ssyevx(fb_layout_t layout, char jobz,
                                        fb_uplo_t uplo, int n, float *A,
                                        int lda, float *w) {
  const fb_backend_vtable_t *ctx = fb_get_active_vtable();
  fb_adapter_ssyevx_fn src =
      ctx ? (fb_adapter_ssyevx_fn)ctx->ext_ops[FB_OP_SSYEVX][FB_CONV_CBLAS]
          : NULL;
  int m = 0;
  int *ifail = (int *)malloc((size_t)((n > 0) ? n : 1) * sizeof(int));
  int rc;
  if (!src || !w || (n > 0 && !A)) {
    free(ifail);
    return -1;
  }
  if (!ifail) {
    return -101;
  }
  rc = src(layout, jobz, 'A', uplo, n, A, lda, 0.0f, 0.0f, 1, n, 0.0f, &m, w,
           A, lda, ifail);
  free(ifail);
  return rc;
}

static int fb_adapter_dsyev_from_dsyevx(fb_layout_t layout, char jobz,
                                        fb_uplo_t uplo, int n, double *A,
                                        int lda, double *w) {
  const fb_backend_vtable_t *ctx = fb_get_active_vtable();
  fb_adapter_dsyevx_fn src =
      ctx ? (fb_adapter_dsyevx_fn)ctx->ext_ops[FB_OP_DSYEVX][FB_CONV_CBLAS]
          : NULL;
  int m = 0;
  int *ifail = (int *)malloc((size_t)((n > 0) ? n : 1) * sizeof(int));
  int rc;
  if (!src || !w || (n > 0 && !A)) {
    free(ifail);
    return -1;
  }
  if (!ifail) {
    return -101;
  }
  rc = src(layout, jobz, 'A', uplo, n, A, lda, 0.0, 0.0, 1, n, 0.0, &m, w, A,
           lda, ifail);
  free(ifail);
  return rc;
}

static int fb_adapter_cheev_from_cheevr(fb_layout_t layout, char jobz,
                                        fb_uplo_t uplo, int n,
                                        fb_complex_float_t *A, int lda,
                                        float *w) {
  const fb_backend_vtable_t *ctx = fb_get_active_vtable();
  fb_adapter_cheevr_fn src =
      ctx ? (fb_adapter_cheevr_fn)ctx->ext_ops[FB_OP_CHEEVR][FB_CONV_CBLAS]
          : NULL;
  int m = 0;
  int *isuppz = (int *)malloc((size_t)((n > 0) ? (2 * n) : 2) * sizeof(int));
  int rc;
  if (!src || !w || (n > 0 && !A)) {
    free(isuppz);
    return -1;
  }
  if (!isuppz) {
    return -101;
  }
  rc = src(layout, jobz, 'A', uplo, n, A, lda, 0.0f, 0.0f, 1, n, 0.0f, &m, w,
           A, lda, isuppz);
  free(isuppz);
  return rc;
}

static int fb_adapter_zheev_from_zheevr(fb_layout_t layout, char jobz,
                                        fb_uplo_t uplo, int n,
                                        fb_complex_double_t *A, int lda,
                                        double *w) {
  const fb_backend_vtable_t *ctx = fb_get_active_vtable();
  fb_adapter_zheevr_fn src =
      ctx ? (fb_adapter_zheevr_fn)ctx->ext_ops[FB_OP_ZHEEVR][FB_CONV_CBLAS]
          : NULL;
  int m = 0;
  int *isuppz = (int *)malloc((size_t)((n > 0) ? (2 * n) : 2) * sizeof(int));
  int rc;
  if (!src || !w || (n > 0 && !A)) {
    free(isuppz);
    return -1;
  }
  if (!isuppz) {
    return -101;
  }
  rc = src(layout, jobz, 'A', uplo, n, A, lda, 0.0, 0.0, 1, n, 0.0, &m, w, A,
           lda, isuppz);
  free(isuppz);
  return rc;
}

static int fb_adapter_cheev_from_cheevx(fb_layout_t layout, char jobz,
                                        fb_uplo_t uplo, int n,
                                        fb_complex_float_t *A, int lda,
                                        float *w) {
  const fb_backend_vtable_t *ctx = fb_get_active_vtable();
  fb_adapter_cheevx_fn src =
      ctx ? (fb_adapter_cheevx_fn)ctx->ext_ops[FB_OP_CHEEVX][FB_CONV_CBLAS]
          : NULL;
  int m = 0;
  int *ifail = (int *)malloc((size_t)((n > 0) ? n : 1) * sizeof(int));
  int rc;
  if (!src || !w || (n > 0 && !A)) {
    free(ifail);
    return -1;
  }
  if (!ifail) {
    return -101;
  }
  rc = src(layout, jobz, 'A', uplo, n, A, lda, 0.0f, 0.0f, 1, n, 0.0f, &m, w,
           A, lda, ifail);
  free(ifail);
  return rc;
}

static int fb_adapter_zheev_from_zheevx(fb_layout_t layout, char jobz,
                                        fb_uplo_t uplo, int n,
                                        fb_complex_double_t *A, int lda,
                                        double *w) {
  const fb_backend_vtable_t *ctx = fb_get_active_vtable();
  fb_adapter_zheevx_fn src =
      ctx ? (fb_adapter_zheevx_fn)ctx->ext_ops[FB_OP_ZHEEVX][FB_CONV_CBLAS]
          : NULL;
  int m = 0;
  int *ifail = (int *)malloc((size_t)((n > 0) ? n : 1) * sizeof(int));
  int rc;
  if (!src || !w || (n > 0 && !A)) {
    free(ifail);
    return -1;
  }
  if (!ifail) {
    return -101;
  }
  rc = src(layout, jobz, 'A', uplo, n, A, lda, 0.0, 0.0, 1, n, 0.0, &m, w, A,
           lda, ifail);
  free(ifail);
  return rc;
}

static int fb_adapter_ssygv_from_ssygvx(fb_layout_t layout, int itype,
                                        char jobz, fb_uplo_t uplo, int n,
                                        float *A, int lda, float *B, int ldb,
                                        float *w) {
  const fb_backend_vtable_t *ctx = fb_get_active_vtable();
  fb_adapter_ssygvx_fn src =
      ctx ? (fb_adapter_ssygvx_fn)ctx->ext_ops[FB_OP_SSYGVX][FB_CONV_CBLAS]
          : NULL;
  int m = 0;
  int *ifail = (int *)malloc((size_t)((n > 0) ? n : 1) * sizeof(int));
  int rc;
  if (!src || !w || (n > 0 && (!A || !B))) {
    free(ifail);
    return -1;
  }
  if (!ifail) {
    return -101;
  }
  rc = src(layout, itype, jobz, 'A', uplo, n, A, lda, B, ldb, 0.0f, 0.0f, 1,
           n, 0.0f, &m, w, A, lda, ifail);
  free(ifail);
  return rc;
}

static int fb_adapter_dsygv_from_dsygvx(fb_layout_t layout, int itype,
                                        char jobz, fb_uplo_t uplo, int n,
                                        double *A, int lda, double *B, int ldb,
                                        double *w) {
  const fb_backend_vtable_t *ctx = fb_get_active_vtable();
  fb_adapter_dsygvx_fn src =
      ctx ? (fb_adapter_dsygvx_fn)ctx->ext_ops[FB_OP_DSYGVX][FB_CONV_CBLAS]
          : NULL;
  int m = 0;
  int *ifail = (int *)malloc((size_t)((n > 0) ? n : 1) * sizeof(int));
  int rc;
  if (!src || !w || (n > 0 && (!A || !B))) {
    free(ifail);
    return -1;
  }
  if (!ifail) {
    return -101;
  }
  rc = src(layout, itype, jobz, 'A', uplo, n, A, lda, B, ldb, 0.0, 0.0, 1, n,
           0.0, &m, w, A, lda, ifail);
  free(ifail);
  return rc;
}

static int fb_adapter_chegv_from_chegvx(fb_layout_t layout, int itype,
                                        char jobz, fb_uplo_t uplo, int n,
                                        fb_complex_float_t *A, int lda,
                                        fb_complex_float_t *B, int ldb,
                                        float *w) {
  const fb_backend_vtable_t *ctx = fb_get_active_vtable();
  fb_adapter_chegvx_fn src =
      ctx ? (fb_adapter_chegvx_fn)ctx->ext_ops[FB_OP_CHEGVX][FB_CONV_CBLAS]
          : NULL;
  int m = 0;
  int *ifail = (int *)malloc((size_t)((n > 0) ? n : 1) * sizeof(int));
  int rc;
  if (!src || !w || (n > 0 && (!A || !B))) {
    free(ifail);
    return -1;
  }
  if (!ifail) {
    return -101;
  }
  rc = src(layout, itype, jobz, 'A', uplo, n, A, lda, B, ldb, 0.0f, 0.0f, 1,
           n, 0.0f, &m, w, A, lda, ifail);
  free(ifail);
  return rc;
}

static int fb_adapter_zhegv_from_zhegvx(fb_layout_t layout, int itype,
                                        char jobz, fb_uplo_t uplo, int n,
                                        fb_complex_double_t *A, int lda,
                                        fb_complex_double_t *B, int ldb,
                                        double *w) {
  const fb_backend_vtable_t *ctx = fb_get_active_vtable();
  fb_adapter_zhegvx_fn src =
      ctx ? (fb_adapter_zhegvx_fn)ctx->ext_ops[FB_OP_ZHEGVX][FB_CONV_CBLAS]
          : NULL;
  int m = 0;
  int *ifail = (int *)malloc((size_t)((n > 0) ? n : 1) * sizeof(int));
  int rc;
  if (!src || !w || (n > 0 && (!A || !B))) {
    free(ifail);
    return -1;
  }
  if (!ifail) {
    return -101;
  }
  rc = src(layout, itype, jobz, 'A', uplo, n, A, lda, B, ldb, 0.0, 0.0, 1, n,
           0.0, &m, w, A, lda, ifail);
  free(ifail);
  return rc;
}

static void fb_adapter_sstev_from_sstevr(int layout, char jobz, int n,
                                         float *d, float *e, float *z, int ldz,
                                         float *work, int *info) {
  const fb_backend_vtable_t *ctx = fb_get_active_vtable();
  fb_adapter_sstevr_fn src =
      ctx ? (fb_adapter_sstevr_fn)ctx->ext_ops[FB_OP_SSTEVR][FB_CONV_CBLAS]
          : NULL;
  int m = 0;
  int *isuppz = (int *)malloc((size_t)((n > 0) ? (2 * n) : 2) * sizeof(int));
  int rc;
  (void)work;
  if (!info || !src || !d || !e) {
    if (info) {
      *info = -1;
    }
    free(isuppz);
    return;
  }
  if (!isuppz) {
    *info = -101;
    return;
  }
  rc = src((fb_layout_t)layout, jobz, 'A', n, d, e, 0.0f, 0.0f, 1, n, 0.0f,
           &m, d, z, ldz, isuppz);
  *info = rc;
  free(isuppz);
}

static void fb_adapter_dstev_from_dstevr(int layout, char jobz, int n,
                                         double *d, double *e, double *z,
                                         int ldz, double *work, int *info) {
  const fb_backend_vtable_t *ctx = fb_get_active_vtable();
  fb_adapter_dstevr_fn src =
      ctx ? (fb_adapter_dstevr_fn)ctx->ext_ops[FB_OP_DSTEVR][FB_CONV_CBLAS]
          : NULL;
  int m = 0;
  int *isuppz = (int *)malloc((size_t)((n > 0) ? (2 * n) : 2) * sizeof(int));
  int rc;
  (void)work;
  if (!info || !src || !d || !e) {
    if (info) {
      *info = -1;
    }
    free(isuppz);
    return;
  }
  if (!isuppz) {
    *info = -101;
    return;
  }
  rc = src((fb_layout_t)layout, jobz, 'A', n, d, e, 0.0, 0.0, 1, n, 0.0, &m,
           d, z, ldz, isuppz);
  *info = rc;
  free(isuppz);
}

static void fb_adapter_sstev_from_sstevx(int layout, char jobz, int n,
                                         float *d, float *e, float *z, int ldz,
                                         float *work, int *info) {
  const fb_backend_vtable_t *ctx = fb_get_active_vtable();
  fb_adapter_sstevx_fn src =
      ctx ? (fb_adapter_sstevx_fn)ctx->ext_ops[FB_OP_SSTEVX][FB_CONV_CBLAS]
          : NULL;
  int m = 0;
  int *ifail = (int *)malloc((size_t)((n > 0) ? n : 1) * sizeof(int));
  int rc;
  (void)work;
  if (!info || !src || !d || !e) {
    if (info) {
      *info = -1;
    }
    free(ifail);
    return;
  }
  if (!ifail) {
    *info = -101;
    return;
  }
  rc = src((fb_layout_t)layout, jobz, 'A', n, d, e, 0.0f, 0.0f, 1, n, 0.0f,
           &m, d, z, ldz, ifail);
  *info = rc;
  free(ifail);
}

static void fb_adapter_dstev_from_dstevx(int layout, char jobz, int n,
                                         double *d, double *e, double *z,
                                         int ldz, double *work, int *info) {
  const fb_backend_vtable_t *ctx = fb_get_active_vtable();
  fb_adapter_dstevx_fn src =
      ctx ? (fb_adapter_dstevx_fn)ctx->ext_ops[FB_OP_DSTEVX][FB_CONV_CBLAS]
          : NULL;
  int m = 0;
  int *ifail = (int *)malloc((size_t)((n > 0) ? n : 1) * sizeof(int));
  int rc;
  (void)work;
  if (!info || !src || !d || !e) {
    if (info) {
      *info = -1;
    }
    free(ifail);
    return;
  }
  if (!ifail) {
    *info = -101;
    return;
  }
  rc = src((fb_layout_t)layout, jobz, 'A', n, d, e, 0.0, 0.0, 1, n, 0.0, &m,
           d, z, ldz, ifail);
  *info = rc;
  free(ifail);
}

static void fb_adapter_sstev_from_sstedc(int layout, char jobz, int n,
                                         float *d, float *e, float *z, int ldz,
                                         float *work, int *info) {
  const fb_backend_vtable_t *ctx = fb_get_active_vtable();
  fb_adapter_sstedc_fn src =
      ctx ? (fb_adapter_sstedc_fn)ctx->ext_ops[FB_OP_SSTEVD][FB_CONV_CBLAS]
          : NULL;
  char compz;
  int rc;
  (void)work;
  if (!info || !src || !d || !e) {
    if (info) {
      *info = -1;
    }
    return;
  }
  compz = (jobz == 'V') ? 'I' : jobz;
  rc = src((fb_layout_t)layout, compz, n, d, e, z, ldz);
  *info = rc;
}

static void fb_adapter_dstev_from_dstedc(int layout, char jobz, int n,
                                         double *d, double *e, double *z,
                                         int ldz, double *work, int *info) {
  const fb_backend_vtable_t *ctx = fb_get_active_vtable();
  fb_adapter_dstedc_fn src =
      ctx ? (fb_adapter_dstedc_fn)ctx->ext_ops[FB_OP_DSTEVD][FB_CONV_CBLAS]
          : NULL;
  char compz;
  int rc;
  (void)work;
  if (!info || !src || !d || !e) {
    if (info) {
      *info = -1;
    }
    return;
  }
  compz = (jobz == 'V') ? 'I' : jobz;
  rc = src((fb_layout_t)layout, compz, n, d, e, z, ldz);
  *info = rc;
}

static int fb_adapter_sgeev_from_sgeevx(fb_layout_t layout, char jobvl,
                                        char jobvr, int n, float *A, int lda,
                                        float *wr, float *wi, float *VL,
                                        int ldvl, float *VR, int ldvr) {
  const fb_backend_vtable_t *ctx = fb_get_active_vtable();
  fb_adapter_sgeevx_fn src =
      ctx ? (fb_adapter_sgeevx_fn)ctx->ext_ops[FB_OP_SGEEVX][FB_CONV_CBLAS]
          : NULL;
  size_t count = (size_t)((n > 0) ? n : 1);
  int ilo = 0;
  int ihi = 0;
  float abnrm = 0.0f;
  float *scale = (float *)malloc(count * sizeof(float));
  float *rconde = (float *)malloc(count * sizeof(float));
  float *rcondv = (float *)malloc(count * sizeof(float));
  int rc;

  if (!src || !wr || !wi || (n > 0 && !A)) {
    free(scale);
    free(rconde);
    free(rcondv);
    return -1;
  }
  if (!scale || !rconde || !rcondv) {
    free(scale);
    free(rconde);
    free(rcondv);
    return -101;
  }

  rc = src(layout, 'N', jobvl, jobvr, 'N', n, A, lda, wr, wi, VL, ldvl, VR,
           ldvr, &ilo, &ihi, scale, &abnrm, rconde, rcondv);

  free(scale);
  free(rconde);
  free(rcondv);
  return rc;
}

static int fb_adapter_dgeev_from_dgeevx(fb_layout_t layout, char jobvl,
                                        char jobvr, int n, double *A,
                                        int lda, double *wr, double *wi,
                                        double *VL, int ldvl, double *VR,
                                        int ldvr) {
  const fb_backend_vtable_t *ctx = fb_get_active_vtable();
  fb_adapter_dgeevx_fn src =
      ctx ? (fb_adapter_dgeevx_fn)ctx->ext_ops[FB_OP_DGEEVX][FB_CONV_CBLAS]
          : NULL;
  size_t count = (size_t)((n > 0) ? n : 1);
  int ilo = 0;
  int ihi = 0;
  double abnrm = 0.0;
  double *scale = (double *)malloc(count * sizeof(double));
  double *rconde = (double *)malloc(count * sizeof(double));
  double *rcondv = (double *)malloc(count * sizeof(double));
  int rc;

  if (!src || !wr || !wi || (n > 0 && !A)) {
    free(scale);
    free(rconde);
    free(rcondv);
    return -1;
  }
  if (!scale || !rconde || !rcondv) {
    free(scale);
    free(rconde);
    free(rcondv);
    return -101;
  }

  rc = src(layout, 'N', jobvl, jobvr, 'N', n, A, lda, wr, wi, VL, ldvl, VR,
           ldvr, &ilo, &ihi, scale, &abnrm, rconde, rcondv);

  free(scale);
  free(rconde);
  free(rcondv);
  return rc;
}

static int fb_adapter_cgeev_from_cgeevx(fb_layout_t layout, char jobvl,
                                        char jobvr, int n,
                                        fb_complex_float_t *A, int lda,
                                        fb_complex_float_t *w,
                                        fb_complex_float_t *VL, int ldvl,
                                        fb_complex_float_t *VR, int ldvr) {
  const fb_backend_vtable_t *ctx = fb_get_active_vtable();
  fb_adapter_cgeevx_fn src =
      ctx ? (fb_adapter_cgeevx_fn)ctx->ext_ops[FB_OP_CGEEVX][FB_CONV_CBLAS]
          : NULL;
  size_t count = (size_t)((n > 0) ? n : 1);
  int ilo = 0;
  int ihi = 0;
  float abnrm = 0.0f;
  float *scale = (float *)malloc(count * sizeof(float));
  float *rconde = (float *)malloc(count * sizeof(float));
  float *rcondv = (float *)malloc(count * sizeof(float));
  int rc;

  if (!src || !w || (n > 0 && !A)) {
    free(scale);
    free(rconde);
    free(rcondv);
    return -1;
  }
  if (!scale || !rconde || !rcondv) {
    free(scale);
    free(rconde);
    free(rcondv);
    return -101;
  }

  rc = src(layout, 'N', jobvl, jobvr, 'N', n, A, lda, w, VL, ldvl, VR, ldvr,
           &ilo, &ihi, scale, &abnrm, rconde, rcondv);

  free(scale);
  free(rconde);
  free(rcondv);
  return rc;
}

static int fb_adapter_zgeev_from_zgeevx(fb_layout_t layout, char jobvl,
                                        char jobvr, int n,
                                        fb_complex_double_t *A, int lda,
                                        fb_complex_double_t *w,
                                        fb_complex_double_t *VL, int ldvl,
                                        fb_complex_double_t *VR, int ldvr) {
  const fb_backend_vtable_t *ctx = fb_get_active_vtable();
  fb_adapter_zgeevx_fn src =
      ctx ? (fb_adapter_zgeevx_fn)ctx->ext_ops[FB_OP_ZGEEVX][FB_CONV_CBLAS]
          : NULL;
  size_t count = (size_t)((n > 0) ? n : 1);
  int ilo = 0;
  int ihi = 0;
  double abnrm = 0.0;
  double *scale = (double *)malloc(count * sizeof(double));
  double *rconde = (double *)malloc(count * sizeof(double));
  double *rcondv = (double *)malloc(count * sizeof(double));
  int rc;

  if (!src || !w || (n > 0 && !A)) {
    free(scale);
    free(rconde);
    free(rcondv);
    return -1;
  }
  if (!scale || !rconde || !rcondv) {
    free(scale);
    free(rconde);
    free(rcondv);
    return -101;
  }

  rc = src(layout, 'N', jobvl, jobvr, 'N', n, A, lda, w, VL, ldvl, VR, ldvr,
           &ilo, &ihi, scale, &abnrm, rconde, rcondv);

  free(scale);
  free(rconde);
  free(rcondv);
  return rc;
}

static int fb_adapter_sgesv_from_sgesvx(fb_layout_t layout, int n, int nrhs,
                                        float *A, int lda, int *ipiv,
                                        float *B, int ldb) {
  const fb_backend_vtable_t *ctx = fb_get_active_vtable();
  fb_adapter_sgesvx_fn src =
      ctx ? (fb_adapter_sgesvx_fn)ctx->ext_ops[FB_OP_SGESVX][FB_CONV_CBLAS]
          : NULL;
  size_t order = (size_t)((n > 0) ? n : 1);
  size_t rhs_count = (size_t)((nrhs > 0) ? nrhs : 1);
  float *af = (float *)calloc((size_t)((lda > 0) ? lda : 1) * order,
                              sizeof(float));
  float *x = (float *)calloc((size_t)((ldb > 0) ? ldb : 1) * rhs_count,
                             sizeof(float));
  float *r = (float *)calloc(order, sizeof(float));
  float *c = (float *)calloc(order, sizeof(float));
  float *ferr = (float *)calloc(rhs_count, sizeof(float));
  float *berr = (float *)calloc(rhs_count, sizeof(float));
  float *rpivot = (float *)calloc(order, sizeof(float));
  char equed = 'N';
  float rcond = 0.0f;
  int rc;

  if (!src || !ipiv || (n > 0 && (!A || !B))) {
    free(af);
    free(x);
    free(r);
    free(c);
    free(ferr);
    free(berr);
    free(rpivot);
    return -1;
  }
  if (!af || !x || !r || !c || !ferr || !berr || !rpivot) {
    free(af);
    free(x);
    free(r);
    free(c);
    free(ferr);
    free(berr);
    free(rpivot);
    return -101;
  }

  rc = src(layout, 'N', 'N', n, nrhs, A, lda, af, lda, ipiv, &equed, r, c, B,
           ldb, x, ldb, &rcond, ferr, berr, rpivot);
  if (n > 0 && lda > 0) {
    memcpy(A, af, (size_t)lda * order * sizeof(float));
  }
  if (rc == 0 && nrhs > 0 && ldb > 0) {
    memcpy(B, x, (size_t)ldb * rhs_count * sizeof(float));
  }

  free(af);
  free(x);
  free(r);
  free(c);
  free(ferr);
  free(berr);
  free(rpivot);
  return rc;
}

static int fb_adapter_dgesv_from_dgesvx(fb_layout_t layout, int n, int nrhs,
                                        double *A, int lda, int *ipiv,
                                        double *B, int ldb) {
  const fb_backend_vtable_t *ctx = fb_get_active_vtable();
  fb_adapter_dgesvx_fn src =
      ctx ? (fb_adapter_dgesvx_fn)ctx->ext_ops[FB_OP_DGESVX][FB_CONV_CBLAS]
          : NULL;
  size_t order = (size_t)((n > 0) ? n : 1);
  size_t rhs_count = (size_t)((nrhs > 0) ? nrhs : 1);
  double *af = (double *)calloc((size_t)((lda > 0) ? lda : 1) * order,
                                sizeof(double));
  double *x = (double *)calloc((size_t)((ldb > 0) ? ldb : 1) * rhs_count,
                               sizeof(double));
  double *r = (double *)calloc(order, sizeof(double));
  double *c = (double *)calloc(order, sizeof(double));
  double *ferr = (double *)calloc(rhs_count, sizeof(double));
  double *berr = (double *)calloc(rhs_count, sizeof(double));
  double *rpivot = (double *)calloc(order, sizeof(double));
  char equed = 'N';
  double rcond = 0.0;
  int rc;

  if (!src || !ipiv || (n > 0 && (!A || !B))) {
    free(af);
    free(x);
    free(r);
    free(c);
    free(ferr);
    free(berr);
    free(rpivot);
    return -1;
  }
  if (!af || !x || !r || !c || !ferr || !berr || !rpivot) {
    free(af);
    free(x);
    free(r);
    free(c);
    free(ferr);
    free(berr);
    free(rpivot);
    return -101;
  }

  rc = src(layout, 'N', 'N', n, nrhs, A, lda, af, lda, ipiv, &equed, r, c, B,
           ldb, x, ldb, &rcond, ferr, berr, rpivot);
  if (n > 0 && lda > 0) {
    memcpy(A, af, (size_t)lda * order * sizeof(double));
  }
  if (rc == 0 && nrhs > 0 && ldb > 0) {
    memcpy(B, x, (size_t)ldb * rhs_count * sizeof(double));
  }

  free(af);
  free(x);
  free(r);
  free(c);
  free(ferr);
  free(berr);
  free(rpivot);
  return rc;
}

static int fb_adapter_cgesv_from_cgesvx(fb_layout_t layout, int n, int nrhs,
                                        fb_complex_float_t *A, int lda,
                                        int *ipiv, fb_complex_float_t *B,
                                        int ldb) {
  const fb_backend_vtable_t *ctx = fb_get_active_vtable();
  fb_adapter_cgesvx_fn src =
      ctx ? (fb_adapter_cgesvx_fn)ctx->ext_ops[FB_OP_CGESVX][FB_CONV_CBLAS]
          : NULL;
  size_t order = (size_t)((n > 0) ? n : 1);
  size_t rhs_count = (size_t)((nrhs > 0) ? nrhs : 1);
  fb_complex_float_t *af =
      (fb_complex_float_t *)calloc((size_t)((lda > 0) ? lda : 1) * order,
                                   sizeof(fb_complex_float_t));
  fb_complex_float_t *x =
      (fb_complex_float_t *)calloc((size_t)((ldb > 0) ? ldb : 1) * rhs_count,
                                   sizeof(fb_complex_float_t));
  float *r = (float *)calloc(order, sizeof(float));
  float *c = (float *)calloc(order, sizeof(float));
  float *ferr = (float *)calloc(rhs_count, sizeof(float));
  float *berr = (float *)calloc(rhs_count, sizeof(float));
  float *rpivot = (float *)calloc(order, sizeof(float));
  char equed = 'N';
  float rcond = 0.0f;
  int rc;

  if (!src || !ipiv || (n > 0 && (!A || !B))) {
    free(af);
    free(x);
    free(r);
    free(c);
    free(ferr);
    free(berr);
    free(rpivot);
    return -1;
  }
  if (!af || !x || !r || !c || !ferr || !berr || !rpivot) {
    free(af);
    free(x);
    free(r);
    free(c);
    free(ferr);
    free(berr);
    free(rpivot);
    return -101;
  }

  rc = src(layout, 'N', 'N', n, nrhs, A, lda, af, lda, ipiv, &equed, r, c, B,
           ldb, x, ldb, &rcond, ferr, berr, rpivot);
  if (n > 0 && lda > 0) {
    memcpy(A, af, (size_t)lda * order * sizeof(fb_complex_float_t));
  }
  if (rc == 0 && nrhs > 0 && ldb > 0) {
    memcpy(B, x, (size_t)ldb * rhs_count * sizeof(fb_complex_float_t));
  }

  free(af);
  free(x);
  free(r);
  free(c);
  free(ferr);
  free(berr);
  free(rpivot);
  return rc;
}

static int fb_adapter_zgesv_from_zgesvx(fb_layout_t layout, int n, int nrhs,
                                        fb_complex_double_t *A, int lda,
                                        int *ipiv, fb_complex_double_t *B,
                                        int ldb) {
  const fb_backend_vtable_t *ctx = fb_get_active_vtable();
  fb_adapter_zgesvx_fn src =
      ctx ? (fb_adapter_zgesvx_fn)ctx->ext_ops[FB_OP_ZGESVX][FB_CONV_CBLAS]
          : NULL;
  size_t order = (size_t)((n > 0) ? n : 1);
  size_t rhs_count = (size_t)((nrhs > 0) ? nrhs : 1);
  fb_complex_double_t *af =
      (fb_complex_double_t *)calloc((size_t)((lda > 0) ? lda : 1) * order,
                                    sizeof(fb_complex_double_t));
  fb_complex_double_t *x =
      (fb_complex_double_t *)calloc((size_t)((ldb > 0) ? ldb : 1) * rhs_count,
                                    sizeof(fb_complex_double_t));
  double *r = (double *)calloc(order, sizeof(double));
  double *c = (double *)calloc(order, sizeof(double));
  double *ferr = (double *)calloc(rhs_count, sizeof(double));
  double *berr = (double *)calloc(rhs_count, sizeof(double));
  double *rpivot = (double *)calloc(order, sizeof(double));
  char equed = 'N';
  double rcond = 0.0;
  int rc;

  if (!src || !ipiv || (n > 0 && (!A || !B))) {
    free(af);
    free(x);
    free(r);
    free(c);
    free(ferr);
    free(berr);
    free(rpivot);
    return -1;
  }
  if (!af || !x || !r || !c || !ferr || !berr || !rpivot) {
    free(af);
    free(x);
    free(r);
    free(c);
    free(ferr);
    free(berr);
    free(rpivot);
    return -101;
  }

  rc = src(layout, 'N', 'N', n, nrhs, A, lda, af, lda, ipiv, &equed, r, c, B,
           ldb, x, ldb, &rcond, ferr, berr, rpivot);
  if (n > 0 && lda > 0) {
    memcpy(A, af, (size_t)lda * order * sizeof(fb_complex_double_t));
  }
  if (rc == 0 && nrhs > 0 && ldb > 0) {
    memcpy(B, x, (size_t)ldb * rhs_count * sizeof(fb_complex_double_t));
  }

  free(af);
  free(x);
  free(r);
  free(c);
  free(ferr);
  free(berr);
  free(rpivot);
  return rc;
}

static int fb_adapter_sposv_from_sposvx(fb_layout_t layout, fb_uplo_t uplo,
                                        int n, int nrhs, float *A, int lda,
                                        float *B, int ldb) {
  const fb_backend_vtable_t *ctx = fb_get_active_vtable();
  fb_adapter_sposvx_fn src =
      ctx ? (fb_adapter_sposvx_fn)ctx->ext_ops[FB_OP_SPOSVX][FB_CONV_CBLAS]
          : NULL;
  size_t order = (size_t)((n > 0) ? n : 1);
  size_t rhs_count = (size_t)((nrhs > 0) ? nrhs : 1);
  float *af = (float *)calloc((size_t)((lda > 0) ? lda : 1) * order,
                              sizeof(float));
  float *x = (float *)calloc((size_t)((ldb > 0) ? ldb : 1) * rhs_count,
                             sizeof(float));
  float *s = (float *)calloc(order, sizeof(float));
  float *ferr = (float *)calloc(rhs_count, sizeof(float));
  float *berr = (float *)calloc(rhs_count, sizeof(float));
  char equed = 'N';
  float rcond = 0.0f;
  int rc;

  if (!src || (n > 0 && (!A || !B))) {
    free(af);
    free(x);
    free(s);
    free(ferr);
    free(berr);
    return -1;
  }
  if (!af || !x || !s || !ferr || !berr) {
    free(af);
    free(x);
    free(s);
    free(ferr);
    free(berr);
    return -101;
  }

  rc = src(layout, 'N', (char)uplo, n, nrhs, A, lda, af, lda, &equed, s, B,
           ldb, x, ldb, &rcond, ferr, berr);
  if (n > 0 && lda > 0) {
    memcpy(A, af, (size_t)lda * order * sizeof(float));
  }
  if (rc == 0 && nrhs > 0 && ldb > 0) {
    memcpy(B, x, (size_t)ldb * rhs_count * sizeof(float));
  }

  free(af);
  free(x);
  free(s);
  free(ferr);
  free(berr);
  return rc;
}

static int fb_adapter_dposv_from_dposvx(fb_layout_t layout, fb_uplo_t uplo,
                                        int n, int nrhs, double *A, int lda,
                                        double *B, int ldb) {
  const fb_backend_vtable_t *ctx = fb_get_active_vtable();
  fb_adapter_dposvx_fn src =
      ctx ? (fb_adapter_dposvx_fn)ctx->ext_ops[FB_OP_DPOSVX][FB_CONV_CBLAS]
          : NULL;
  size_t order = (size_t)((n > 0) ? n : 1);
  size_t rhs_count = (size_t)((nrhs > 0) ? nrhs : 1);
  double *af = (double *)calloc((size_t)((lda > 0) ? lda : 1) * order,
                                sizeof(double));
  double *x = (double *)calloc((size_t)((ldb > 0) ? ldb : 1) * rhs_count,
                               sizeof(double));
  double *s = (double *)calloc(order, sizeof(double));
  double *ferr = (double *)calloc(rhs_count, sizeof(double));
  double *berr = (double *)calloc(rhs_count, sizeof(double));
  char equed = 'N';
  double rcond = 0.0;
  int rc;

  if (!src || (n > 0 && (!A || !B))) {
    free(af);
    free(x);
    free(s);
    free(ferr);
    free(berr);
    return -1;
  }
  if (!af || !x || !s || !ferr || !berr) {
    free(af);
    free(x);
    free(s);
    free(ferr);
    free(berr);
    return -101;
  }

  rc = src(layout, 'N', (char)uplo, n, nrhs, A, lda, af, lda, &equed, s, B,
           ldb, x, ldb, &rcond, ferr, berr);
  if (n > 0 && lda > 0) {
    memcpy(A, af, (size_t)lda * order * sizeof(double));
  }
  if (rc == 0 && nrhs > 0 && ldb > 0) {
    memcpy(B, x, (size_t)ldb * rhs_count * sizeof(double));
  }

  free(af);
  free(x);
  free(s);
  free(ferr);
  free(berr);
  return rc;
}

static int fb_adapter_cposv_from_cposvx(fb_layout_t layout, fb_uplo_t uplo,
                                        int n, int nrhs,
                                        fb_complex_float_t *A, int lda,
                                        fb_complex_float_t *B, int ldb) {
  const fb_backend_vtable_t *ctx = fb_get_active_vtable();
  fb_adapter_cposvx_fn src =
      ctx ? (fb_adapter_cposvx_fn)ctx->ext_ops[FB_OP_CPOSVX][FB_CONV_CBLAS]
          : NULL;
  size_t order = (size_t)((n > 0) ? n : 1);
  size_t rhs_count = (size_t)((nrhs > 0) ? nrhs : 1);
  fb_complex_float_t *af =
      (fb_complex_float_t *)calloc((size_t)((lda > 0) ? lda : 1) * order,
                                   sizeof(fb_complex_float_t));
  fb_complex_float_t *x =
      (fb_complex_float_t *)calloc((size_t)((ldb > 0) ? ldb : 1) * rhs_count,
                                   sizeof(fb_complex_float_t));
  float *s = (float *)calloc(order, sizeof(float));
  float *ferr = (float *)calloc(rhs_count, sizeof(float));
  float *berr = (float *)calloc(rhs_count, sizeof(float));
  char equed = 'N';
  float rcond = 0.0f;
  int rc;

  if (!src || (n > 0 && (!A || !B))) {
    free(af);
    free(x);
    free(s);
    free(ferr);
    free(berr);
    return -1;
  }
  if (!af || !x || !s || !ferr || !berr) {
    free(af);
    free(x);
    free(s);
    free(ferr);
    free(berr);
    return -101;
  }

  rc = src(layout, 'N', (char)uplo, n, nrhs, A, lda, af, lda, &equed, s, B,
           ldb, x, ldb, &rcond, ferr, berr);
  if (n > 0 && lda > 0) {
    memcpy(A, af, (size_t)lda * order * sizeof(fb_complex_float_t));
  }
  if (rc == 0 && nrhs > 0 && ldb > 0) {
    memcpy(B, x, (size_t)ldb * rhs_count * sizeof(fb_complex_float_t));
  }

  free(af);
  free(x);
  free(s);
  free(ferr);
  free(berr);
  return rc;
}

static int fb_adapter_zposv_from_zposvx(fb_layout_t layout, fb_uplo_t uplo,
                                        int n, int nrhs,
                                        fb_complex_double_t *A, int lda,
                                        fb_complex_double_t *B, int ldb) {
  const fb_backend_vtable_t *ctx = fb_get_active_vtable();
  fb_adapter_zposvx_fn src =
      ctx ? (fb_adapter_zposvx_fn)ctx->ext_ops[FB_OP_ZPOSVX][FB_CONV_CBLAS]
          : NULL;
  size_t order = (size_t)((n > 0) ? n : 1);
  size_t rhs_count = (size_t)((nrhs > 0) ? nrhs : 1);
  fb_complex_double_t *af =
      (fb_complex_double_t *)calloc((size_t)((lda > 0) ? lda : 1) * order,
                                    sizeof(fb_complex_double_t));
  fb_complex_double_t *x =
      (fb_complex_double_t *)calloc((size_t)((ldb > 0) ? ldb : 1) * rhs_count,
                                    sizeof(fb_complex_double_t));
  double *s = (double *)calloc(order, sizeof(double));
  double *ferr = (double *)calloc(rhs_count, sizeof(double));
  double *berr = (double *)calloc(rhs_count, sizeof(double));
  char equed = 'N';
  double rcond = 0.0;
  int rc;

  if (!src || (n > 0 && (!A || !B))) {
    free(af);
    free(x);
    free(s);
    free(ferr);
    free(berr);
    return -1;
  }
  if (!af || !x || !s || !ferr || !berr) {
    free(af);
    free(x);
    free(s);
    free(ferr);
    free(berr);
    return -101;
  }

  rc = src(layout, 'N', (char)uplo, n, nrhs, A, lda, af, lda, &equed, s, B,
           ldb, x, ldb, &rcond, ferr, berr);
  if (n > 0 && lda > 0) {
    memcpy(A, af, (size_t)lda * order * sizeof(fb_complex_double_t));
  }
  if (rc == 0 && nrhs > 0 && ldb > 0) {
    memcpy(B, x, (size_t)ldb * rhs_count * sizeof(fb_complex_double_t));
  }

  free(af);
  free(x);
  free(s);
  free(ferr);
  free(berr);
  return rc;
}

static int fb_adapter_ssysv_from_ssysvx(fb_layout_t layout, char uplo, int n,
                                        int nrhs, float *A, int lda, int *ipiv,
                                        float *B, int ldb) {
  const fb_backend_vtable_t *ctx = fb_get_active_vtable();
  fb_adapter_ssysvx_fn src =
      ctx ? (fb_adapter_ssysvx_fn)ctx->ext_ops[FB_OP_SSYSVX][FB_CONV_CBLAS]
          : NULL;
  size_t order = (size_t)((n > 0) ? n : 1);
  size_t rhs_count = (size_t)((nrhs > 0) ? nrhs : 1);
  float *af = (float *)calloc((size_t)((lda > 0) ? lda : 1) * order,
                              sizeof(float));
  float *x = (float *)calloc((size_t)((ldb > 0) ? ldb : 1) * rhs_count,
                             sizeof(float));
  float *ferr = (float *)calloc(rhs_count, sizeof(float));
  float *berr = (float *)calloc(rhs_count, sizeof(float));
  float rcond = 0.0f;
  int rc;

  if (!src || !ipiv || (n > 0 && (!A || !B))) {
    free(af);
    free(x);
    free(ferr);
    free(berr);
    return -1;
  }
  if (!af || !x || !ferr || !berr) {
    free(af);
    free(x);
    free(ferr);
    free(berr);
    return -101;
  }

  rc = src(layout, 'N', uplo, n, nrhs, A, lda, af, lda, ipiv, B, ldb, x, ldb,
           &rcond, ferr, berr);
  if (n > 0 && lda > 0) {
    memcpy(A, af, (size_t)lda * order * sizeof(float));
  }
  if (rc == 0 && nrhs > 0 && ldb > 0) {
    memcpy(B, x, (size_t)ldb * rhs_count * sizeof(float));
  }

  free(af);
  free(x);
  free(ferr);
  free(berr);
  return rc;
}

static int fb_adapter_dsysv_from_dsysvx(fb_layout_t layout, char uplo, int n,
                                        int nrhs, double *A, int lda,
                                        int *ipiv, double *B, int ldb) {
  const fb_backend_vtable_t *ctx = fb_get_active_vtable();
  fb_adapter_dsysvx_fn src =
      ctx ? (fb_adapter_dsysvx_fn)ctx->ext_ops[FB_OP_DSYSVX][FB_CONV_CBLAS]
          : NULL;
  size_t order = (size_t)((n > 0) ? n : 1);
  size_t rhs_count = (size_t)((nrhs > 0) ? nrhs : 1);
  double *af = (double *)calloc((size_t)((lda > 0) ? lda : 1) * order,
                                sizeof(double));
  double *x = (double *)calloc((size_t)((ldb > 0) ? ldb : 1) * rhs_count,
                               sizeof(double));
  double *ferr = (double *)calloc(rhs_count, sizeof(double));
  double *berr = (double *)calloc(rhs_count, sizeof(double));
  double rcond = 0.0;
  int rc;

  if (!src || !ipiv || (n > 0 && (!A || !B))) {
    free(af);
    free(x);
    free(ferr);
    free(berr);
    return -1;
  }
  if (!af || !x || !ferr || !berr) {
    free(af);
    free(x);
    free(ferr);
    free(berr);
    return -101;
  }

  rc = src(layout, 'N', uplo, n, nrhs, A, lda, af, lda, ipiv, B, ldb, x, ldb,
           &rcond, ferr, berr);
  if (n > 0 && lda > 0) {
    memcpy(A, af, (size_t)lda * order * sizeof(double));
  }
  if (rc == 0 && nrhs > 0 && ldb > 0) {
    memcpy(B, x, (size_t)ldb * rhs_count * sizeof(double));
  }

  free(af);
  free(x);
  free(ferr);
  free(berr);
  return rc;
}

static int fb_adapter_csysv_from_csysvx(fb_layout_t layout, char uplo, int n,
                                        int nrhs, fb_complex_float_t *A,
                                        int lda, int *ipiv,
                                        fb_complex_float_t *B, int ldb) {
  const fb_backend_vtable_t *ctx = fb_get_active_vtable();
  fb_adapter_csysvx_fn src =
      ctx ? (fb_adapter_csysvx_fn)ctx->ext_ops[FB_OP_CSYSVX][FB_CONV_CBLAS]
          : NULL;
  size_t order = (size_t)((n > 0) ? n : 1);
  size_t rhs_count = (size_t)((nrhs > 0) ? nrhs : 1);
  fb_complex_float_t *af =
      (fb_complex_float_t *)calloc((size_t)((lda > 0) ? lda : 1) * order,
                                   sizeof(fb_complex_float_t));
  fb_complex_float_t *x =
      (fb_complex_float_t *)calloc((size_t)((ldb > 0) ? ldb : 1) * rhs_count,
                                   sizeof(fb_complex_float_t));
  float *ferr = (float *)calloc(rhs_count, sizeof(float));
  float *berr = (float *)calloc(rhs_count, sizeof(float));
  float rcond = 0.0f;
  int rc;

  if (!src || !ipiv || (n > 0 && (!A || !B))) {
    free(af);
    free(x);
    free(ferr);
    free(berr);
    return -1;
  }
  if (!af || !x || !ferr || !berr) {
    free(af);
    free(x);
    free(ferr);
    free(berr);
    return -101;
  }

  rc = src(layout, 'N', uplo, n, nrhs, A, lda, af, lda, ipiv, B, ldb, x, ldb,
           &rcond, ferr, berr);
  if (n > 0 && lda > 0) {
    memcpy(A, af, (size_t)lda * order * sizeof(fb_complex_float_t));
  }
  if (rc == 0 && nrhs > 0 && ldb > 0) {
    memcpy(B, x, (size_t)ldb * rhs_count * sizeof(fb_complex_float_t));
  }

  free(af);
  free(x);
  free(ferr);
  free(berr);
  return rc;
}

static int fb_adapter_zsysv_from_zsysvx(fb_layout_t layout, char uplo, int n,
                                        int nrhs, fb_complex_double_t *A,
                                        int lda, int *ipiv,
                                        fb_complex_double_t *B, int ldb) {
  const fb_backend_vtable_t *ctx = fb_get_active_vtable();
  fb_adapter_zsysvx_fn src =
      ctx ? (fb_adapter_zsysvx_fn)ctx->ext_ops[FB_OP_ZSYSVX][FB_CONV_CBLAS]
          : NULL;
  size_t order = (size_t)((n > 0) ? n : 1);
  size_t rhs_count = (size_t)((nrhs > 0) ? nrhs : 1);
  fb_complex_double_t *af =
      (fb_complex_double_t *)calloc((size_t)((lda > 0) ? lda : 1) * order,
                                    sizeof(fb_complex_double_t));
  fb_complex_double_t *x =
      (fb_complex_double_t *)calloc((size_t)((ldb > 0) ? ldb : 1) * rhs_count,
                                    sizeof(fb_complex_double_t));
  double *ferr = (double *)calloc(rhs_count, sizeof(double));
  double *berr = (double *)calloc(rhs_count, sizeof(double));
  double rcond = 0.0;
  int rc;

  if (!src || !ipiv || (n > 0 && (!A || !B))) {
    free(af);
    free(x);
    free(ferr);
    free(berr);
    return -1;
  }
  if (!af || !x || !ferr || !berr) {
    free(af);
    free(x);
    free(ferr);
    free(berr);
    return -101;
  }

  rc = src(layout, 'N', uplo, n, nrhs, A, lda, af, lda, ipiv, B, ldb, x, ldb,
           &rcond, ferr, berr);
  if (n > 0 && lda > 0) {
    memcpy(A, af, (size_t)lda * order * sizeof(fb_complex_double_t));
  }
  if (rc == 0 && nrhs > 0 && ldb > 0) {
    memcpy(B, x, (size_t)ldb * rhs_count * sizeof(fb_complex_double_t));
  }

  free(af);
  free(x);
  free(ferr);
  free(berr);
  return rc;
}

static void fb_install_adapter_if_missing(fb_backend_vtable_t *vtable,
                                          uint32_t dst_op, uint32_t src_op,
                                          fb_generic_fn adapter_fn,
                                          fb_generic_fn *src_storage) {
  if (vtable->ext_ops[dst_op][FB_CONV_CBLAS] != NULL) {
    return;
  }
  if (vtable->ext_ops[src_op][FB_CONV_CBLAS] == NULL) {
    return;
  }
  *src_storage = vtable->ext_ops[src_op][FB_CONV_CBLAS];
  vtable->ext_ops[dst_op][FB_CONV_CBLAS] = adapter_fn;
}

static fb_status_t fb_autofill_superset_subset_adapters(
    fb_backend_vtable_t *vtable) {
  if (!vtable) {
    return FB_STATUS_INVALID_ARGUMENT;
  }

  /* Family A: SYEV/HEEV from SYEVR/HEEVR (range='A', il=1, iu=n). */
  if (vtable->ext_ops[FB_OP_SSYEV][FB_CONV_CBLAS] == NULL) {
    if (vtable->ext_ops[FB_OP_SSYEVR][FB_CONV_CBLAS] != NULL) {
      g_adapter_ssyevr =
          (fb_adapter_ssyevr_fn)vtable->ext_ops[FB_OP_SSYEVR][FB_CONV_CBLAS];
      vtable->ext_ops[FB_OP_SSYEV][FB_CONV_CBLAS] =
          (fb_generic_fn)(void (*)(void))fb_adapter_ssyev_from_ssyevr;
    } else if (vtable->ext_ops[FB_OP_SSYEVX][FB_CONV_CBLAS] != NULL) {
      g_adapter_ssyevx =
          (fb_adapter_ssyevx_fn)vtable->ext_ops[FB_OP_SSYEVX][FB_CONV_CBLAS];
      vtable->ext_ops[FB_OP_SSYEV][FB_CONV_CBLAS] =
          (fb_generic_fn)(void (*)(void))fb_adapter_ssyev_from_ssyevx;
    }
  }
  if (vtable->ext_ops[FB_OP_DSYEV][FB_CONV_CBLAS] == NULL) {
    if (vtable->ext_ops[FB_OP_DSYEVR][FB_CONV_CBLAS] != NULL) {
      g_adapter_dsyevr =
          (fb_adapter_dsyevr_fn)vtable->ext_ops[FB_OP_DSYEVR][FB_CONV_CBLAS];
      vtable->ext_ops[FB_OP_DSYEV][FB_CONV_CBLAS] =
          (fb_generic_fn)(void (*)(void))fb_adapter_dsyev_from_dsyevr;
    } else if (vtable->ext_ops[FB_OP_DSYEVX][FB_CONV_CBLAS] != NULL) {
      g_adapter_dsyevx =
          (fb_adapter_dsyevx_fn)vtable->ext_ops[FB_OP_DSYEVX][FB_CONV_CBLAS];
      vtable->ext_ops[FB_OP_DSYEV][FB_CONV_CBLAS] =
          (fb_generic_fn)(void (*)(void))fb_adapter_dsyev_from_dsyevx;
    }
  }
  if (vtable->ext_ops[FB_OP_CHEEV][FB_CONV_CBLAS] == NULL) {
    if (vtable->ext_ops[FB_OP_CHEEVR][FB_CONV_CBLAS] != NULL) {
      g_adapter_cheevr = (fb_adapter_cheevr_fn)vtable->ext_ops[FB_OP_CHEEVR][FB_CONV_CBLAS];
      vtable->ext_ops[FB_OP_CHEEV][FB_CONV_CBLAS] =
          (fb_generic_fn)(void (*)(void))fb_adapter_cheev_from_cheevr;
    } else if (vtable->ext_ops[FB_OP_CHEEVX][FB_CONV_CBLAS] != NULL) {
      g_adapter_cheevx = (fb_adapter_cheevx_fn)vtable->ext_ops[FB_OP_CHEEVX][FB_CONV_CBLAS];
      vtable->ext_ops[FB_OP_CHEEV][FB_CONV_CBLAS] =
          (fb_generic_fn)(void (*)(void))fb_adapter_cheev_from_cheevx;
    }
  }
  if (vtable->ext_ops[FB_OP_ZHEEV][FB_CONV_CBLAS] == NULL) {
    if (vtable->ext_ops[FB_OP_ZHEEVR][FB_CONV_CBLAS] != NULL) {
      g_adapter_zheevr = (fb_adapter_zheevr_fn)vtable->ext_ops[FB_OP_ZHEEVR][FB_CONV_CBLAS];
      vtable->ext_ops[FB_OP_ZHEEV][FB_CONV_CBLAS] =
          (fb_generic_fn)(void (*)(void))fb_adapter_zheev_from_zheevr;
    } else if (vtable->ext_ops[FB_OP_ZHEEVX][FB_CONV_CBLAS] != NULL) {
      g_adapter_zheevx = (fb_adapter_zheevx_fn)vtable->ext_ops[FB_OP_ZHEEVX][FB_CONV_CBLAS];
      vtable->ext_ops[FB_OP_ZHEEV][FB_CONV_CBLAS] =
          (fb_generic_fn)(void (*)(void))fb_adapter_zheev_from_zheevx;
    }
  }

  /* Family B: SYGV/HEGV from SYGVX/HEGVX (range='A', il=1, iu=n). */
  fb_install_adapter_if_missing(vtable, FB_OP_SSYGV, FB_OP_SSYGVX,
                                (fb_generic_fn)(void (*)(void))
                                    fb_adapter_ssygv_from_ssygvx,
                                (fb_generic_fn *)&g_adapter_ssygvx);
  fb_install_adapter_if_missing(vtable, FB_OP_DSYGV, FB_OP_DSYGVX,
                                (fb_generic_fn)(void (*)(void))
                                    fb_adapter_dsygv_from_dsygvx,
                                (fb_generic_fn *)&g_adapter_dsygvx);
  fb_install_adapter_if_missing(vtable, FB_OP_CHEGV, FB_OP_CHEGVX,
                                (fb_generic_fn)(void (*)(void))
                                    fb_adapter_chegv_from_chegvx,
                                (fb_generic_fn *)&g_adapter_chegvx);
  fb_install_adapter_if_missing(vtable, FB_OP_ZHEGV, FB_OP_ZHEGVX,
                                (fb_generic_fn)(void (*)(void))
                                    fb_adapter_zhegv_from_zhegvx,
                                (fb_generic_fn *)&g_adapter_zhegvx);

  /* Family C: STEV from STEVR/STEVX (range='A', il=1, iu=n). */
  if (vtable->ext_ops[FB_OP_SSTEV][FB_CONV_CBLAS] == NULL) {
    if (vtable->ext_ops[FB_OP_SSTEVR][FB_CONV_CBLAS] != NULL) {
      g_adapter_sstevr =
          (fb_adapter_sstevr_fn)vtable->ext_ops[FB_OP_SSTEVR][FB_CONV_CBLAS];
      vtable->ext_ops[FB_OP_SSTEV][FB_CONV_CBLAS] =
          (fb_generic_fn)(void (*)(void))fb_adapter_sstev_from_sstevr;
    } else if (vtable->ext_ops[FB_OP_SSTEVX][FB_CONV_CBLAS] != NULL) {
      g_adapter_sstevx =
          (fb_adapter_sstevx_fn)vtable->ext_ops[FB_OP_SSTEVX][FB_CONV_CBLAS];
      vtable->ext_ops[FB_OP_SSTEV][FB_CONV_CBLAS] =
          (fb_generic_fn)(void (*)(void))fb_adapter_sstev_from_sstevx;
    } else if (vtable->ext_ops[FB_OP_SSTEVD][FB_CONV_CBLAS] != NULL) {
      g_adapter_sstedc =
          (fb_adapter_sstedc_fn)vtable->ext_ops[FB_OP_SSTEVD][FB_CONV_CBLAS];
      vtable->ext_ops[FB_OP_SSTEV][FB_CONV_CBLAS] =
          (fb_generic_fn)(void (*)(void))fb_adapter_sstev_from_sstedc;
    }
  }
  if (vtable->ext_ops[FB_OP_DSTEV][FB_CONV_CBLAS] == NULL) {
    if (vtable->ext_ops[FB_OP_DSTEVR][FB_CONV_CBLAS] != NULL) {
      g_adapter_dstevr =
          (fb_adapter_dstevr_fn)vtable->ext_ops[FB_OP_DSTEVR][FB_CONV_CBLAS];
      vtable->ext_ops[FB_OP_DSTEV][FB_CONV_CBLAS] =
          (fb_generic_fn)(void (*)(void))fb_adapter_dstev_from_dstevr;
    } else if (vtable->ext_ops[FB_OP_DSTEVX][FB_CONV_CBLAS] != NULL) {
      g_adapter_dstevx =
          (fb_adapter_dstevx_fn)vtable->ext_ops[FB_OP_DSTEVX][FB_CONV_CBLAS];
      vtable->ext_ops[FB_OP_DSTEV][FB_CONV_CBLAS] =
          (fb_generic_fn)(void (*)(void))fb_adapter_dstev_from_dstevx;
    } else if (vtable->ext_ops[FB_OP_DSTEVD][FB_CONV_CBLAS] != NULL) {
      g_adapter_dstedc =
          (fb_adapter_dstedc_fn)vtable->ext_ops[FB_OP_DSTEVD][FB_CONV_CBLAS];
      vtable->ext_ops[FB_OP_DSTEV][FB_CONV_CBLAS] =
          (fb_generic_fn)(void (*)(void))fb_adapter_dstev_from_dstedc;
    }
  }

  /* Family D: GEEV from GEEVX (balanc='N', sense='N'). */
  fb_install_adapter_if_missing(vtable, FB_OP_SGEEV, FB_OP_SGEEVX,
                                (fb_generic_fn)(void (*)(void))
                                    fb_adapter_sgeev_from_sgeevx,
                                (fb_generic_fn *)&g_adapter_sgeevx);
  fb_install_adapter_if_missing(vtable, FB_OP_DGEEV, FB_OP_DGEEVX,
                                (fb_generic_fn)(void (*)(void))
                                    fb_adapter_dgeev_from_dgeevx,
                                (fb_generic_fn *)&g_adapter_dgeevx);
  fb_install_adapter_if_missing(vtable, FB_OP_CGEEV, FB_OP_CGEEVX,
                                (fb_generic_fn)(void (*)(void))
                                    fb_adapter_cgeev_from_cgeevx,
                                (fb_generic_fn *)&g_adapter_cgeevx);
  fb_install_adapter_if_missing(vtable, FB_OP_ZGEEV, FB_OP_ZGEEVX,
                                (fb_generic_fn)(void (*)(void))
                                    fb_adapter_zgeev_from_zgeevx,
                                (fb_generic_fn *)&g_adapter_zgeevx);

  /* Family E: GESV from GESVX (fact='N', trans='N'). */
  fb_install_adapter_if_missing(vtable, FB_OP_SGESV, FB_OP_SGESVX,
                                (fb_generic_fn)(void (*)(void))
                                    fb_adapter_sgesv_from_sgesvx,
                                (fb_generic_fn *)&g_adapter_sgesvx);
  fb_install_adapter_if_missing(vtable, FB_OP_DGESV, FB_OP_DGESVX,
                                (fb_generic_fn)(void (*)(void))
                                    fb_adapter_dgesv_from_dgesvx,
                                (fb_generic_fn *)&g_adapter_dgesvx);
    fb_install_adapter_if_missing(vtable, FB_OP_CGESV, FB_OP_CGESVX,
                  (fb_generic_fn)(void (*)(void))
                    fb_adapter_cgesv_from_cgesvx,
                  (fb_generic_fn *)&g_adapter_cgesvx);
    fb_install_adapter_if_missing(vtable, FB_OP_ZGESV, FB_OP_ZGESVX,
                  (fb_generic_fn)(void (*)(void))
                    fb_adapter_zgesv_from_zgesvx,
                  (fb_generic_fn *)&g_adapter_zgesvx);

      /* Family F: POSV from POSVX (fact='N'). */
      fb_install_adapter_if_missing(vtable, FB_OP_SPOSV, FB_OP_SPOSVX,
                    (fb_generic_fn)(void (*)(void))
                      fb_adapter_sposv_from_sposvx,
                    (fb_generic_fn *)&g_adapter_sposvx);
      fb_install_adapter_if_missing(vtable, FB_OP_DPOSV, FB_OP_DPOSVX,
                    (fb_generic_fn)(void (*)(void))
                      fb_adapter_dposv_from_dposvx,
                    (fb_generic_fn *)&g_adapter_dposvx);
      fb_install_adapter_if_missing(vtable, FB_OP_CPOSV, FB_OP_CPOSVX,
                    (fb_generic_fn)(void (*)(void))
                      fb_adapter_cposv_from_cposvx,
                    (fb_generic_fn *)&g_adapter_cposvx);
      fb_install_adapter_if_missing(vtable, FB_OP_ZPOSV, FB_OP_ZPOSVX,
                    (fb_generic_fn)(void (*)(void))
                      fb_adapter_zposv_from_zposvx,
                    (fb_generic_fn *)&g_adapter_zposvx);

        /* Family G: SYSV from SYSVX (fact='N'). */
        fb_install_adapter_if_missing(vtable, FB_OP_SSYSV, FB_OP_SSYSVX,
                      (fb_generic_fn)(void (*)(void))
                        fb_adapter_ssysv_from_ssysvx,
                      (fb_generic_fn *)&g_adapter_ssysvx);
        fb_install_adapter_if_missing(vtable, FB_OP_DSYSV, FB_OP_DSYSVX,
                      (fb_generic_fn)(void (*)(void))
                        fb_adapter_dsysv_from_dsysvx,
                      (fb_generic_fn *)&g_adapter_dsysvx);
        fb_install_adapter_if_missing(vtable, FB_OP_CSYSV, FB_OP_CSYSVX,
                      (fb_generic_fn)(void (*)(void))
                        fb_adapter_csysv_from_csysvx,
                      (fb_generic_fn *)&g_adapter_csysvx);
        fb_install_adapter_if_missing(vtable, FB_OP_ZSYSV, FB_OP_ZSYSVX,
                      (fb_generic_fn)(void (*)(void))
                        fb_adapter_zsysv_from_zsysvx,
                      (fb_generic_fn *)&g_adapter_zsysvx);

  return FB_STATUS_SUCCESS;
}

#undef FB_COMPAT_GROUP2
#undef FB_COMPAT_GROUP4

/* ============================================================================
 * Active Vtable Management
 * ========================================================================= */

const fb_backend_vtable_t *fb_get_active_vtable(void) {
  return g_active_vtable;
}

void fb_set_active_vtable(const fb_backend_vtable_t *vtable) {
  g_active_vtable = vtable;
}

/* ============================================================================
 * Core Auto-Fill Entry Point
 * ========================================================================= */

fb_status_t fb_finalize_plugin_vtable(fb_backend_vtable_t *vtable) {
  if (!vtable) {
    return FB_STATUS_INVALID_ARGUMENT;
  }

  fb_status_t status;

  /* Strategy 1: Unified ↔ Specific (zero overhead) */
  status = fb_autofill_unified_to_specific(vtable);
  if (status != FB_STATUS_SUCCESS) {
    printf("[faster-blaster] Warning: Unified→Specific autofill failed\n");
  }

  status = fb_autofill_specific_to_unified(vtable);
  if (status != FB_STATUS_SUCCESS) {
    printf("[faster-blaster] Warning: Specific→Unified autofill failed\n");
  }

  /* Strategy 2: Batched → Single (zero overhead) */
  status = fb_autofill_batched_to_single(vtable);
  if (status != FB_STATUS_SUCCESS) {
    printf("[faster-blaster] Warning: Batched→Single autofill failed\n");
  }

  /* Strategy 3: Array → Strided (lightweight) */
  status = fb_autofill_array_to_strided(vtable);
  if (status != FB_STATUS_SUCCESS) {
    printf("[faster-blaster] Warning: Array→Strided autofill failed\n");
  }

  /* Strategy 4: Precision promotion (last resort) */
  status = fb_autofill_precision_promotion(vtable);
  if (status != FB_STATUS_SUCCESS) {
    printf("[faster-blaster] Warning: Precision promotion autofill failed\n");
  }

  /* Sync named fields into ext_ops before compatibility propagation. */
  fb_vtable_sync_ext_ops(vtable);

  /* Strategy 0a: compatible algorithm families share one implementation. */
  status = fb_autofill_compatibility_groups(vtable);
  if (status != FB_STATUS_SUCCESS) {
    printf("[faster-blaster] Warning: Compatibility-group autofill failed\n");
  }

  /* Strategy 0c-a: install explicit superset/subset adapters where defaults
   * can make a richer API equivalent to a simpler API. */
  status = fb_autofill_superset_subset_adapters(vtable);
  if (status != FB_STATUS_SUCCESS) {
    printf("[faster-blaster] Warning: Superset/subset adapter autofill failed\n");
  }

  /* Strategy 5: install cross-convention thunks for Fortran-only exports. */
  for (uint32_t op_id = 0; op_id < (uint32_t)FB_JUDGE_MAX_OPERATIONS; op_id++) {
    fb_install_conv_thunks(vtable, op_id);
  }

  /* Strategy 0b: re-run compatibility after convention thunks.
   * This lets Fortran-only donor slots (now exposed via generated CBLAS thunks)
   * propagate into compatible CBLAS operation slots as well. */
  status = fb_autofill_compatibility_groups(vtable);
  if (status != FB_STATUS_SUCCESS) {
    printf("[faster-blaster] Warning: Post-thunk compatibility autofill failed\n");
  }

  /* Strategy 0c-b: re-run adapters after thunk installation as well. */
  status = fb_autofill_superset_subset_adapters(vtable);
  if (status != FB_STATUS_SUCCESS) {
    printf("[faster-blaster] Warning: Post-thunk adapter autofill failed\n");
  }

  /* Fill typed named fields from ext_ops for newly group-filled operations. */
  fb_vtable_fill_named_from_ext_ops(vtable);

  /* Final step: mirror named fields → ext_ops[] for uniform op-level dispatch */
  fb_vtable_sync_ext_ops(vtable);

  /* Keep adapter wrappers bound to the just-finalized vtable by default. */
  fb_set_active_vtable(vtable);

  return FB_STATUS_SUCCESS;
}

/* ============================================================================
 * Strategy 1: Unified ↔ Specific
 * ========================================================================= */

/* External wrapper implementations */
extern void fb_autofill_gemm_from_unified(fb_backend_vtable_t *vtable);
#ifdef FB_EXTENDED_VTABLE_SUPPORT
extern void fb_autofill_normalization_from_unified(fb_backend_vtable_t *vtable);
extern void fb_autofill_reduction_from_unified(fb_backend_vtable_t *vtable);
#endif

fb_status_t fb_autofill_unified_to_specific(fb_backend_vtable_t *vtable) {
  if (!vtable) {
    return FB_STATUS_INVALID_ARGUMENT;
  }

  /* GEMM wrappers: sgemm, dgemm, cgemm, zgemm from gemm_unified */
  fb_autofill_gemm_from_unified(vtable);

#ifdef FB_EXTENDED_VTABLE_SUPPORT
  /* Normalization wrappers: batch_norm, layer_norm, etc. from normalize_unified
   */
  fb_autofill_normalization_from_unified(vtable);

  /* Reduction wrappers: tensor/parallel/stats/collective reductions from
   * reduce_unified */
  fb_autofill_reduction_from_unified(vtable);
#endif

  return FB_STATUS_SUCCESS;
}

/* External dispatcher implementations */
extern void fb_autofill_gemm_unified_from_specific(fb_backend_vtable_t *vtable);

fb_status_t fb_autofill_specific_to_unified(fb_backend_vtable_t *vtable) {
  if (!vtable) {
    return FB_STATUS_INVALID_ARGUMENT;
  }

  /* GEMM unified dispatcher: gemm_unified from sgemm/dgemm/cgemm/zgemm */
  fb_autofill_gemm_unified_from_specific(vtable);

  /* TODO: Phase 2.5 - Normalize unified dispatcher from specific normalization
   * ops */
  /* TODO: Phase 2.6 - Reduce unified dispatcher from specific reduction ops */

  return FB_STATUS_SUCCESS;
}

/* ============================================================================
 * Strategy 2: Batched → Single (TODO: Implement in Phase 3)
 * ========================================================================= */

#ifdef FB_EXTENDED_VTABLE_SUPPORT
/* External wrapper implementations */
extern void fb_autofill_all_from_batched(fb_backend_vtable_t *vtable);
#endif

fb_status_t fb_autofill_batched_to_single(fb_backend_vtable_t *vtable) {
  if (!vtable) {
    return FB_STATUS_INVALID_ARGUMENT;
  }

#ifdef FB_EXTENDED_VTABLE_SUPPORT
  /* Install single wrappers from batched operations */
  fb_autofill_all_from_batched(vtable);
#endif

  return FB_STATUS_SUCCESS;
}

/* ============================================================================
 * Strategy 3: Array → Strided (TODO: Implement in Phase 3)
 * ========================================================================= */

#ifdef FB_EXTENDED_VTABLE_SUPPORT
/* External wrapper implementations */
extern void fb_autofill_all_strided_from_array(fb_backend_vtable_t *vtable);
#endif

fb_status_t fb_autofill_array_to_strided(fb_backend_vtable_t *vtable) {
  if (!vtable) {
    return FB_STATUS_INVALID_ARGUMENT;
  }

#ifdef FB_EXTENDED_VTABLE_SUPPORT
  /* Install strided batched wrappers from array-based operations */
  fb_autofill_all_strided_from_array(vtable);
#endif

  return FB_STATUS_SUCCESS;
}

/* ============================================================================
 * Strategy 4: Precision Promotion (TODO: Implement in Phase 5)
 * ========================================================================= */

/* External wrapper implementations */
extern void fb_autofill_all_precision_promotion(fb_backend_vtable_t *vtable);

fb_status_t fb_autofill_precision_promotion(fb_backend_vtable_t *vtable) {
  if (!vtable) {
    return FB_STATUS_INVALID_ARGUMENT;
  }

  /* Install precision promotion fallback wrappers (last resort for correctness)
   */
  fb_autofill_all_precision_promotion(vtable);

  return FB_STATUS_SUCCESS;
}

/* ============================================================================
 * Utility Functions: Array Promotion/Demotion
 * ========================================================================= */

double *fb_promote_array_f32_to_f64(const float *input, size_t count) {
  if (!input || count == 0) {
    return NULL;
  }

  double *output = (double *)malloc(count * sizeof(double));
  if (!output) {
    return NULL;
  }

  for (size_t i = 0; i < count; i++) {
    output[i] = (double)input[i];
  }

  return output;
}

fb_status_t fb_demote_array_f64_to_f32_checked(const double *input,
                                               float *output, size_t count) {
  if (!input || !output || count == 0) {
    return FB_STATUS_INVALID_ARGUMENT;
  }

  fb_status_t status = FB_STATUS_SUCCESS;

  for (size_t i = 0; i < count; i++) {
    double val = input[i];

    /* Check for overflow */
    if (val > (double)FLT_MAX) {
      output[i] = FLT_MAX;
      status = FB_STATUS_OVERFLOW;
    } else if (val < -(double)FLT_MAX) {
      output[i] = -FLT_MAX;
      status = FB_STATUS_OVERFLOW;
    } else {
      output[i] = (float)val;
    }
  }

  if (status == FB_STATUS_OVERFLOW) {
    printf("[faster-blaster] Warning: FP64→FP32 demotion overflow detected, "
           "values clamped to ±FLT_MAX\n");
  }

  return status;
}

fb_complex_double_t *
fb_promote_array_c64_to_c128(const fb_complex_float_t *input, size_t count) {
  if (!input || count == 0) {
    return NULL;
  }

  fb_complex_double_t *output =
      (fb_complex_double_t *)malloc(count * sizeof(fb_complex_double_t));
  if (!output) {
    return NULL;
  }

  for (size_t i = 0; i < count; i++) {
    __real__(output[i]) = (double)__real__(input[i]);
    __imag__(output[i]) = (double)__imag__(input[i]);
  }

  return output;
}

fb_status_t
fb_demote_array_c128_to_c64_checked(const fb_complex_double_t *input,
                                    fb_complex_float_t *output, size_t count) {
  if (!input || !output || count == 0) {
    return FB_STATUS_INVALID_ARGUMENT;
  }

  fb_status_t status = FB_STATUS_SUCCESS;

  for (size_t i = 0; i < count; i++) {
    double real = __real__(input[i]);
    double imag = __imag__(input[i]);

    /* Check for overflow in real part */
    if (real > (double)FLT_MAX) {
      __real__(output[i]) = FLT_MAX;
      status = FB_STATUS_OVERFLOW;
    } else if (real < -(double)FLT_MAX) {
      __real__(output[i]) = -FLT_MAX;
      status = FB_STATUS_OVERFLOW;
    } else {
      __real__(output[i]) = (float)real;
    }

    /* Check for overflow in imaginary part */
    if (imag > (double)FLT_MAX) {
      __imag__(output[i]) = FLT_MAX;
      status = FB_STATUS_OVERFLOW;
    } else if (imag < -(double)FLT_MAX) {
      __imag__(output[i]) = -FLT_MAX;
      status = FB_STATUS_OVERFLOW;
    } else {
      __imag__(output[i]) = (float)imag;
    }
  }

  if (status == FB_STATUS_OVERFLOW) {
    printf("[faster-blaster] Warning: C128→C64 demotion overflow detected\n");
  }

  return status;
}

int32_t *fb_promote_array_i8_to_i32(const int8_t *input, size_t count) {
  if (!input || count == 0) {
    return NULL;
  }

  int32_t *output = (int32_t *)malloc(count * sizeof(int32_t));
  if (!output) {
    return NULL;
  }

  for (size_t i = 0; i < count; i++) {
    output[i] = (int32_t)input[i];
  }

  return output;
}

fb_status_t fb_demote_array_i32_to_i8_checked(const int32_t *input,
                                              int8_t *output, size_t count) {
  if (!input || !output || count == 0) {
    return FB_STATUS_INVALID_ARGUMENT;
  }

  fb_status_t status = FB_STATUS_SUCCESS;

  for (size_t i = 0; i < count; i++) {
    int32_t val = input[i];

    /* Check for overflow */
    if (val > 127) {
      output[i] = 127;
      status = FB_STATUS_OVERFLOW;
    } else if (val < -128) {
      output[i] = -128;
      status = FB_STATUS_OVERFLOW;
    } else {
      output[i] = (int8_t)val;
    }
  }

  if (status == FB_STATUS_OVERFLOW) {
    printf("[faster-blaster] Warning: INT32→INT8 demotion overflow detected, "
           "values clamped to [-128, 127]\n");
  }

  return status;
}

/* ============================================================================
 * Utility Functions: Temporary Buffer Management
 * ========================================================================= */

void *fb_allocate_temp_buffer(size_t size) {
  if (size == 0) {
    return NULL;
  }

  /* Use standard malloc for now - could use aligned_alloc on modern platforms
   */
  return malloc(size);
}

void fb_free_temp_buffer(void *buffer) {
  if (buffer) {
    free(buffer);
  }
}
