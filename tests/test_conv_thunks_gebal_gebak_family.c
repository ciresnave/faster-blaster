#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef int (*fb_sgebal_cblas_fn)(fb_layout_t layout, char job, int n,
                                  float *a, int lda, int *ilo, int *ihi,
                                  float *scale);
typedef int (*fb_dgebal_cblas_fn)(fb_layout_t layout, char job, int n,
                                  double *a, int lda, int *ilo, int *ihi,
                                  double *scale);
typedef int (*fb_cgebal_cblas_fn)(fb_layout_t layout, char job, int n,
                                  fb_complex_float_t *a, int lda, int *ilo,
                                  int *ihi, float *scale);
typedef int (*fb_zgebal_cblas_fn)(fb_layout_t layout, char job, int n,
                                  fb_complex_double_t *a, int lda, int *ilo,
                                  int *ihi, double *scale);

typedef void (*fb_sgebal_fortran_fn)(char *job, int *n, float *a, int *lda,
                                     int *ilo, int *ihi, float *scale,
                                     int *info);
typedef void (*fb_dgebal_fortran_fn)(char *job, int *n, double *a, int *lda,
                                     int *ilo, int *ihi, double *scale,
                                     int *info);
typedef void (*fb_cgebal_fortran_fn)(char *job, int *n, fb_complex_float_t *a,
                                     int *lda, int *ilo, int *ihi,
                                     float *scale, int *info);
typedef void (*fb_zgebal_fortran_fn)(char *job, int *n, fb_complex_double_t *a,
                                     int *lda, int *ilo, int *ihi,
                                     double *scale, int *info);

typedef int (*fb_sgebak_cblas_fn)(fb_layout_t layout, char job, char side,
                                  int n, int ilo, int ihi,
                                  const float *scale, int m, float *v,
                                  int ldv);
typedef int (*fb_dgebak_cblas_fn)(fb_layout_t layout, char job, char side,
                                  int n, int ilo, int ihi,
                                  const double *scale, int m, double *v,
                                  int ldv);
typedef int (*fb_cgebak_cblas_fn)(fb_layout_t layout, char job, char side,
                                  int n, int ilo, int ihi,
                                  const float *scale, int m,
                                  fb_complex_float_t *v, int ldv);
typedef int (*fb_zgebak_cblas_fn)(fb_layout_t layout, char job, char side,
                                  int n, int ilo, int ihi,
                                  const double *scale, int m,
                                  fb_complex_double_t *v, int ldv);

typedef void (*fb_sgebak_fortran_fn)(char *job, char *side, int *n, int *ilo,
                                     int *ihi, float *scale, int *m,
                                     float *v, int *ldv, int *info);
typedef void (*fb_dgebak_fortran_fn)(char *job, char *side, int *n, int *ilo,
                                     int *ihi, double *scale, int *m,
                                     double *v, int *ldv, int *info);
typedef void (*fb_cgebak_fortran_fn)(char *job, char *side, int *n, int *ilo,
                                     int *ihi, float *scale, int *m,
                                     fb_complex_float_t *v, int *ldv,
                                     int *info);
typedef void (*fb_zgebak_fortran_fn)(char *job, char *side, int *n, int *ilo,
                                     int *ihi, double *scale, int *m,
                                     fb_complex_double_t *v, int *ldv,
                                     int *info);

typedef struct {
    int called;
    int layout;
    char job;
    char side;
    int n;
    int lda;
    int ilo;
    int ihi;
    int m;
    int ldv;
    const void *scale;
    void *matrix;
} balance_call_t;

static balance_call_t g_gebal_fortran_call;
static balance_call_t g_gebal_cblas_call;
static balance_call_t g_gebak_fortran_call;
static balance_call_t g_gebak_cblas_call;

static fb_complex_float_t make_cf32(float real_value, float imag_value)
{
    fb_complex_float_t value;

    __real__ value = real_value;
    __imag__ value = imag_value;
    return value;
}

static fb_complex_double_t make_cf64(double real_value, double imag_value)
{
    fb_complex_double_t value;

    __real__ value = real_value;
    __imag__ value = imag_value;
    return value;
}

static int cf32_eq(fb_complex_float_t lhs, fb_complex_float_t rhs)
{
    return __real__ lhs == __real__ rhs && __imag__ lhs == __imag__ rhs;
}

static int cf64_eq(fb_complex_double_t lhs, fb_complex_double_t rhs)
{
    return __real__ lhs == __real__ rhs && __imag__ lhs == __imag__ rhs;
}

static void stub_sgebal_fortran(char *job, int *n, float *a, int *lda,
                                int *ilo, int *ihi, float *scale, int *info)
{
    g_gebal_fortran_call.called += 1;
    g_gebal_fortran_call.job = *job;
    g_gebal_fortran_call.n = *n;
    g_gebal_fortran_call.lda = *lda;
    g_gebal_fortran_call.matrix = a;
    g_gebal_fortran_call.scale = scale;
    *ilo = 1;
    *ihi = 2;
    scale[0] = 11.0f;
    scale[1] = 12.0f;
    a[0] = 101.0f;
    *info = 13;
}

static void stub_dgebal_fortran(char *job, int *n, double *a, int *lda,
                                int *ilo, int *ihi, double *scale, int *info)
{
    g_gebal_fortran_call.called += 1;
    g_gebal_fortran_call.job = *job;
    g_gebal_fortran_call.n = *n;
    g_gebal_fortran_call.lda = *lda;
    g_gebal_fortran_call.matrix = a;
    g_gebal_fortran_call.scale = scale;
    *ilo = 2;
    *ihi = 3;
    scale[0] = 21.0;
    scale[1] = 22.0;
    a[0] = 201.0;
    *info = 23;
}

static void stub_cgebal_fortran(char *job, int *n, fb_complex_float_t *a,
                                int *lda, int *ilo, int *ihi, float *scale,
                                int *info)
{
    g_gebal_fortran_call.called += 1;
    g_gebal_fortran_call.job = *job;
    g_gebal_fortran_call.n = *n;
    g_gebal_fortran_call.lda = *lda;
    g_gebal_fortran_call.matrix = a;
    g_gebal_fortran_call.scale = scale;
    *ilo = 3;
    *ihi = 4;
    scale[0] = 31.0f;
    scale[1] = 32.0f;
    a[0] = make_cf32(301.0f, -1.0f);
    *info = 33;
}

static void stub_zgebal_fortran(char *job, int *n, fb_complex_double_t *a,
                                int *lda, int *ilo, int *ihi, double *scale,
                                int *info)
{
    g_gebal_fortran_call.called += 1;
    g_gebal_fortran_call.job = *job;
    g_gebal_fortran_call.n = *n;
    g_gebal_fortran_call.lda = *lda;
    g_gebal_fortran_call.matrix = a;
    g_gebal_fortran_call.scale = scale;
    *ilo = 4;
    *ihi = 5;
    scale[0] = 41.0;
    scale[1] = 42.0;
    a[0] = make_cf64(401.0, -2.0);
    *info = 43;
}

static int stub_sgebal_cblas(fb_layout_t layout, char job, int n, float *a,
                             int lda, int *ilo, int *ihi, float *scale)
{
    g_gebal_cblas_call.called += 1;
    g_gebal_cblas_call.layout = layout;
    g_gebal_cblas_call.job = job;
    g_gebal_cblas_call.n = n;
    g_gebal_cblas_call.lda = lda;
    g_gebal_cblas_call.matrix = a;
    g_gebal_cblas_call.scale = scale;
    *ilo = 5;
    *ihi = 6;
    scale[0] = 51.0f;
    scale[1] = 52.0f;
    a[0] = 501.0f;
    return 53;
}

static int stub_dgebal_cblas(fb_layout_t layout, char job, int n, double *a,
                             int lda, int *ilo, int *ihi, double *scale)
{
    g_gebal_cblas_call.called += 1;
    g_gebal_cblas_call.layout = layout;
    g_gebal_cblas_call.job = job;
    g_gebal_cblas_call.n = n;
    g_gebal_cblas_call.lda = lda;
    g_gebal_cblas_call.matrix = a;
    g_gebal_cblas_call.scale = scale;
    *ilo = 6;
    *ihi = 7;
    scale[0] = 61.0;
    scale[1] = 62.0;
    a[0] = 601.0;
    return 63;
}

static int stub_cgebal_cblas(fb_layout_t layout, char job, int n,
                             fb_complex_float_t *a, int lda, int *ilo,
                             int *ihi, float *scale)
{
    g_gebal_cblas_call.called += 1;
    g_gebal_cblas_call.layout = layout;
    g_gebal_cblas_call.job = job;
    g_gebal_cblas_call.n = n;
    g_gebal_cblas_call.lda = lda;
    g_gebal_cblas_call.matrix = a;
    g_gebal_cblas_call.scale = scale;
    *ilo = 7;
    *ihi = 8;
    scale[0] = 71.0f;
    scale[1] = 72.0f;
    a[0] = make_cf32(701.0f, 3.0f);
    return 73;
}

static int stub_zgebal_cblas(fb_layout_t layout, char job, int n,
                             fb_complex_double_t *a, int lda, int *ilo,
                             int *ihi, double *scale)
{
    g_gebal_cblas_call.called += 1;
    g_gebal_cblas_call.layout = layout;
    g_gebal_cblas_call.job = job;
    g_gebal_cblas_call.n = n;
    g_gebal_cblas_call.lda = lda;
    g_gebal_cblas_call.matrix = a;
    g_gebal_cblas_call.scale = scale;
    *ilo = 8;
    *ihi = 9;
    scale[0] = 81.0;
    scale[1] = 82.0;
    a[0] = make_cf64(801.0, 4.0);
    return 83;
}

static void stub_sgebak_fortran(char *job, char *side, int *n, int *ilo,
                                int *ihi, float *scale, int *m, float *v,
                                int *ldv, int *info)
{
    g_gebak_fortran_call.called += 1;
    g_gebak_fortran_call.job = *job;
    g_gebak_fortran_call.side = *side;
    g_gebak_fortran_call.n = *n;
    g_gebak_fortran_call.ilo = *ilo;
    g_gebak_fortran_call.ihi = *ihi;
    g_gebak_fortran_call.m = *m;
    g_gebak_fortran_call.ldv = *ldv;
    g_gebak_fortran_call.scale = scale;
    g_gebak_fortran_call.matrix = v;
    v[0] = 901.0f;
    v[1] = 902.0f;
    *info = 93;
}

static void stub_dgebak_fortran(char *job, char *side, int *n, int *ilo,
                                int *ihi, double *scale, int *m, double *v,
                                int *ldv, int *info)
{
    g_gebak_fortran_call.called += 1;
    g_gebak_fortran_call.job = *job;
    g_gebak_fortran_call.side = *side;
    g_gebak_fortran_call.n = *n;
    g_gebak_fortran_call.ilo = *ilo;
    g_gebak_fortran_call.ihi = *ihi;
    g_gebak_fortran_call.m = *m;
    g_gebak_fortran_call.ldv = *ldv;
    g_gebak_fortran_call.scale = scale;
    g_gebak_fortran_call.matrix = v;
    v[0] = 1001.0;
    v[1] = 1002.0;
    *info = 103;
}

static void stub_cgebak_fortran(char *job, char *side, int *n, int *ilo,
                                int *ihi, float *scale, int *m,
                                fb_complex_float_t *v, int *ldv, int *info)
{
    g_gebak_fortran_call.called += 1;
    g_gebak_fortran_call.job = *job;
    g_gebak_fortran_call.side = *side;
    g_gebak_fortran_call.n = *n;
    g_gebak_fortran_call.ilo = *ilo;
    g_gebak_fortran_call.ihi = *ihi;
    g_gebak_fortran_call.m = *m;
    g_gebak_fortran_call.ldv = *ldv;
    g_gebak_fortran_call.scale = scale;
    g_gebak_fortran_call.matrix = v;
    v[0] = make_cf32(1101.0f, -5.0f);
    v[1] = make_cf32(1102.0f, -6.0f);
    *info = 113;
}

static void stub_zgebak_fortran(char *job, char *side, int *n, int *ilo,
                                int *ihi, double *scale, int *m,
                                fb_complex_double_t *v, int *ldv, int *info)
{
    g_gebak_fortran_call.called += 1;
    g_gebak_fortran_call.job = *job;
    g_gebak_fortran_call.side = *side;
    g_gebak_fortran_call.n = *n;
    g_gebak_fortran_call.ilo = *ilo;
    g_gebak_fortran_call.ihi = *ihi;
    g_gebak_fortran_call.m = *m;
    g_gebak_fortran_call.ldv = *ldv;
    g_gebak_fortran_call.scale = scale;
    g_gebak_fortran_call.matrix = v;
    v[0] = make_cf64(1201.0, -7.0);
    v[1] = make_cf64(1202.0, -8.0);
    *info = 123;
}

static int stub_sgebak_cblas(fb_layout_t layout, char job, char side, int n,
                             int ilo, int ihi, const float *scale, int m,
                             float *v, int ldv)
{
    g_gebak_cblas_call.called += 1;
    g_gebak_cblas_call.layout = layout;
    g_gebak_cblas_call.job = job;
    g_gebak_cblas_call.side = side;
    g_gebak_cblas_call.n = n;
    g_gebak_cblas_call.ilo = ilo;
    g_gebak_cblas_call.ihi = ihi;
    g_gebak_cblas_call.m = m;
    g_gebak_cblas_call.ldv = ldv;
    g_gebak_cblas_call.scale = scale;
    g_gebak_cblas_call.matrix = v;
    v[0] = 1301.0f;
    v[1] = 1302.0f;
    return 133;
}

static int stub_dgebak_cblas(fb_layout_t layout, char job, char side, int n,
                             int ilo, int ihi, const double *scale, int m,
                             double *v, int ldv)
{
    g_gebak_cblas_call.called += 1;
    g_gebak_cblas_call.layout = layout;
    g_gebak_cblas_call.job = job;
    g_gebak_cblas_call.side = side;
    g_gebak_cblas_call.n = n;
    g_gebak_cblas_call.ilo = ilo;
    g_gebak_cblas_call.ihi = ihi;
    g_gebak_cblas_call.m = m;
    g_gebak_cblas_call.ldv = ldv;
    g_gebak_cblas_call.scale = scale;
    g_gebak_cblas_call.matrix = v;
    v[0] = 1401.0;
    v[1] = 1402.0;
    return 143;
}

static int stub_cgebak_cblas(fb_layout_t layout, char job, char side, int n,
                             int ilo, int ihi, const float *scale, int m,
                             fb_complex_float_t *v, int ldv)
{
    g_gebak_cblas_call.called += 1;
    g_gebak_cblas_call.layout = layout;
    g_gebak_cblas_call.job = job;
    g_gebak_cblas_call.side = side;
    g_gebak_cblas_call.n = n;
    g_gebak_cblas_call.ilo = ilo;
    g_gebak_cblas_call.ihi = ihi;
    g_gebak_cblas_call.m = m;
    g_gebak_cblas_call.ldv = ldv;
    g_gebak_cblas_call.scale = scale;
    g_gebak_cblas_call.matrix = v;
    v[0] = make_cf32(1501.0f, 5.0f);
    v[1] = make_cf32(1502.0f, 6.0f);
    return 153;
}

static int stub_zgebak_cblas(fb_layout_t layout, char job, char side, int n,
                             int ilo, int ihi, const double *scale, int m,
                             fb_complex_double_t *v, int ldv)
{
    g_gebak_cblas_call.called += 1;
    g_gebak_cblas_call.layout = layout;
    g_gebak_cblas_call.job = job;
    g_gebak_cblas_call.side = side;
    g_gebak_cblas_call.n = n;
    g_gebak_cblas_call.ilo = ilo;
    g_gebak_cblas_call.ihi = ihi;
    g_gebak_cblas_call.m = m;
    g_gebak_cblas_call.ldv = ldv;
    g_gebak_cblas_call.scale = scale;
    g_gebak_cblas_call.matrix = v;
    v[0] = make_cf64(1601.0, 7.0);
    v[1] = make_cf64(1602.0, 8.0);
    return 163;
}

static int check_sgebal_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_sgebal_cblas_fn thunk = NULL;
    int ilo = -1;
    int ihi = -1;
    float a[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    float scale[2] = { 0.0f, 0.0f };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_gebal_fortran_call, 0, sizeof(g_gebal_fortran_call));
    vtable.ext_ops[FB_OP_SGEBAL][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_sgebal_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SGEBAL);
    thunk = (fb_sgebal_cblas_fn)vtable.ext_ops[FB_OP_SGEBAL][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SGEBAL Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    if (thunk(FB_LAYOUT_COL_MAJOR, 'B', 2, a, 2, &ilo, &ihi, scale) != 13 ||
        g_gebal_fortran_call.called != 1 ||
        g_gebal_fortran_call.job != 'B' ||
        g_gebal_fortran_call.n != 2 ||
        g_gebal_fortran_call.lda != 2 ||
        g_gebal_fortran_call.matrix != a ||
        g_gebal_fortran_call.scale != scale ||
        ilo != 1 || ihi != 2 ||
        scale[0] != 11.0f || scale[1] != 12.0f || a[0] != 101.0f) {
        fprintf(stderr, "[FAIL] SGEBAL Fortran->CBLAS thunk did not forward balance arguments and outputs correctly\n");
        return 1;
    }

    printf("[PASS] SGEBAL Fortran->CBLAS thunk forwards balance arguments and outputs\n");
    return 0;
}

static int check_sgebal_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_sgebal_fortran_fn thunk = NULL;
    char job = 'P';
    int n = 2;
    int lda = 2;
    int ilo = -1;
    int ihi = -1;
    int info = -1;
    float a[4] = { 5.0f, 6.0f, 7.0f, 8.0f };
    float scale[2] = { 0.0f, 0.0f };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_gebal_cblas_call, 0, sizeof(g_gebal_cblas_call));
    vtable.ext_ops[FB_OP_SGEBAL][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_sgebal_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SGEBAL);
    thunk = (fb_sgebal_fortran_fn)vtable.ext_ops[FB_OP_SGEBAL][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SGEBAL CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&job, &n, a, &lda, &ilo, &ihi, scale, &info);
    if (g_gebal_cblas_call.called != 1 ||
        g_gebal_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_gebal_cblas_call.job != 'P' ||
        g_gebal_cblas_call.n != 2 ||
        g_gebal_cblas_call.lda != 2 ||
        g_gebal_cblas_call.matrix != a ||
        g_gebal_cblas_call.scale != scale ||
        ilo != 5 || ihi != 6 || info != 53 ||
        scale[0] != 51.0f || scale[1] != 52.0f || a[0] != 501.0f) {
        fprintf(stderr, "[FAIL] SGEBAL CBLAS->Fortran thunk did not map balance arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] SGEBAL CBLAS->Fortran thunk maps balance arguments into the C entry\n");
    return 0;
}

static int check_dgebal_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_dgebal_cblas_fn thunk = NULL;
    int ilo = -1;
    int ihi = -1;
    double a[4] = { 9.0, 10.0, 11.0, 12.0 };
    double scale[2] = { 0.0, 0.0 };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_gebal_fortran_call, 0, sizeof(g_gebal_fortran_call));
    vtable.ext_ops[FB_OP_DGEBAL][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_dgebal_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_DGEBAL);
    thunk = (fb_dgebal_cblas_fn)vtable.ext_ops[FB_OP_DGEBAL][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DGEBAL Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    if (thunk(FB_LAYOUT_COL_MAJOR, 'S', 2, a, 2, &ilo, &ihi, scale) != 23 ||
        g_gebal_fortran_call.called != 1 ||
        g_gebal_fortran_call.job != 'S' ||
        g_gebal_fortran_call.n != 2 ||
        g_gebal_fortran_call.lda != 2 ||
        g_gebal_fortran_call.matrix != a ||
        g_gebal_fortran_call.scale != scale ||
        ilo != 2 || ihi != 3 ||
        scale[0] != 21.0 || scale[1] != 22.0 || a[0] != 201.0) {
        fprintf(stderr, "[FAIL] DGEBAL Fortran->CBLAS thunk did not forward double balance arguments and outputs correctly\n");
        return 1;
    }

    printf("[PASS] DGEBAL Fortran->CBLAS thunk forwards double balance arguments and outputs\n");
    return 0;
}

static int check_dgebal_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_dgebal_fortran_fn thunk = NULL;
    char job = 'B';
    int n = 2;
    int lda = 2;
    int ilo = -1;
    int ihi = -1;
    int info = -1;
    double a[4] = { 13.0, 14.0, 15.0, 16.0 };
    double scale[2] = { 0.0, 0.0 };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_gebal_cblas_call, 0, sizeof(g_gebal_cblas_call));
    vtable.ext_ops[FB_OP_DGEBAL][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_dgebal_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_DGEBAL);
    thunk = (fb_dgebal_fortran_fn)vtable.ext_ops[FB_OP_DGEBAL][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DGEBAL CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&job, &n, a, &lda, &ilo, &ihi, scale, &info);
    if (g_gebal_cblas_call.called != 1 ||
        g_gebal_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_gebal_cblas_call.job != 'B' ||
        g_gebal_cblas_call.n != 2 ||
        g_gebal_cblas_call.lda != 2 ||
        g_gebal_cblas_call.matrix != a ||
        g_gebal_cblas_call.scale != scale ||
        ilo != 6 || ihi != 7 || info != 63 ||
        scale[0] != 61.0 || scale[1] != 62.0 || a[0] != 601.0) {
        fprintf(stderr, "[FAIL] DGEBAL CBLAS->Fortran thunk did not map double balance arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] DGEBAL CBLAS->Fortran thunk maps double balance arguments into the C entry\n");
    return 0;
}

static int check_cgebal_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_cgebal_cblas_fn thunk = NULL;
    int ilo = -1;
    int ihi = -1;
    fb_complex_float_t a[4] = {
        make_cf32(17.0f, 1.0f), make_cf32(18.0f, 2.0f),
        make_cf32(19.0f, 3.0f), make_cf32(20.0f, 4.0f)
    };
    float scale[2] = { 0.0f, 0.0f };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_gebal_fortran_call, 0, sizeof(g_gebal_fortran_call));
    vtable.ext_ops[FB_OP_CGEBAL][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_cgebal_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CGEBAL);
    thunk = (fb_cgebal_cblas_fn)vtable.ext_ops[FB_OP_CGEBAL][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CGEBAL Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    if (thunk(FB_LAYOUT_COL_MAJOR, 'P', 2, a, 2, &ilo, &ihi, scale) != 33 ||
        g_gebal_fortran_call.called != 1 ||
        g_gebal_fortran_call.job != 'P' ||
        g_gebal_fortran_call.n != 2 ||
        g_gebal_fortran_call.lda != 2 ||
        g_gebal_fortran_call.matrix != a ||
        g_gebal_fortran_call.scale != scale ||
        ilo != 3 || ihi != 4 ||
        scale[0] != 31.0f || scale[1] != 32.0f ||
        !cf32_eq(a[0], make_cf32(301.0f, -1.0f))) {
        fprintf(stderr, "[FAIL] CGEBAL Fortran->CBLAS thunk did not forward complex balance arguments and outputs correctly\n");
        return 1;
    }

    printf("[PASS] CGEBAL Fortran->CBLAS thunk forwards complex balance arguments and outputs\n");
    return 0;
}

static int check_cgebal_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_cgebal_fortran_fn thunk = NULL;
    char job = 'B';
    int n = 2;
    int lda = 2;
    int ilo = -1;
    int ihi = -1;
    int info = -1;
    fb_complex_float_t a[4] = {
        make_cf32(21.0f, 1.0f), make_cf32(22.0f, 2.0f),
        make_cf32(23.0f, 3.0f), make_cf32(24.0f, 4.0f)
    };
    float scale[2] = { 0.0f, 0.0f };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_gebal_cblas_call, 0, sizeof(g_gebal_cblas_call));
    vtable.ext_ops[FB_OP_CGEBAL][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_cgebal_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CGEBAL);
    thunk = (fb_cgebal_fortran_fn)vtable.ext_ops[FB_OP_CGEBAL][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CGEBAL CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&job, &n, a, &lda, &ilo, &ihi, scale, &info);
    if (g_gebal_cblas_call.called != 1 ||
        g_gebal_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_gebal_cblas_call.job != 'B' ||
        g_gebal_cblas_call.n != 2 ||
        g_gebal_cblas_call.lda != 2 ||
        g_gebal_cblas_call.matrix != a ||
        g_gebal_cblas_call.scale != scale ||
        ilo != 7 || ihi != 8 || info != 73 ||
        scale[0] != 71.0f || scale[1] != 72.0f ||
        !cf32_eq(a[0], make_cf32(701.0f, 3.0f))) {
        fprintf(stderr, "[FAIL] CGEBAL CBLAS->Fortran thunk did not map complex balance arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] CGEBAL CBLAS->Fortran thunk maps complex balance arguments into the C entry\n");
    return 0;
}

static int check_zgebal_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_zgebal_cblas_fn thunk = NULL;
    int ilo = -1;
    int ihi = -1;
    fb_complex_double_t a[4] = {
        make_cf64(25.0, 1.0), make_cf64(26.0, 2.0),
        make_cf64(27.0, 3.0), make_cf64(28.0, 4.0)
    };
    double scale[2] = { 0.0, 0.0 };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_gebal_fortran_call, 0, sizeof(g_gebal_fortran_call));
    vtable.ext_ops[FB_OP_ZGEBAL][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_zgebal_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_ZGEBAL);
    thunk = (fb_zgebal_cblas_fn)vtable.ext_ops[FB_OP_ZGEBAL][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZGEBAL Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    if (thunk(FB_LAYOUT_COL_MAJOR, 'S', 2, a, 2, &ilo, &ihi, scale) != 43 ||
        g_gebal_fortran_call.called != 1 ||
        g_gebal_fortran_call.job != 'S' ||
        g_gebal_fortran_call.n != 2 ||
        g_gebal_fortran_call.lda != 2 ||
        g_gebal_fortran_call.matrix != a ||
        g_gebal_fortran_call.scale != scale ||
        ilo != 4 || ihi != 5 ||
        scale[0] != 41.0 || scale[1] != 42.0 ||
        !cf64_eq(a[0], make_cf64(401.0, -2.0))) {
        fprintf(stderr, "[FAIL] ZGEBAL Fortran->CBLAS thunk did not forward double-complex balance arguments and outputs correctly\n");
        return 1;
    }

    printf("[PASS] ZGEBAL Fortran->CBLAS thunk forwards double-complex balance arguments and outputs\n");
    return 0;
}

static int check_zgebal_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_zgebal_fortran_fn thunk = NULL;
    char job = 'P';
    int n = 2;
    int lda = 2;
    int ilo = -1;
    int ihi = -1;
    int info = -1;
    fb_complex_double_t a[4] = {
        make_cf64(29.0, 1.0), make_cf64(30.0, 2.0),
        make_cf64(31.0, 3.0), make_cf64(32.0, 4.0)
    };
    double scale[2] = { 0.0, 0.0 };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_gebal_cblas_call, 0, sizeof(g_gebal_cblas_call));
    vtable.ext_ops[FB_OP_ZGEBAL][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_zgebal_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_ZGEBAL);
    thunk = (fb_zgebal_fortran_fn)vtable.ext_ops[FB_OP_ZGEBAL][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZGEBAL CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&job, &n, a, &lda, &ilo, &ihi, scale, &info);
    if (g_gebal_cblas_call.called != 1 ||
        g_gebal_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_gebal_cblas_call.job != 'P' ||
        g_gebal_cblas_call.n != 2 ||
        g_gebal_cblas_call.lda != 2 ||
        g_gebal_cblas_call.matrix != a ||
        g_gebal_cblas_call.scale != scale ||
        ilo != 8 || ihi != 9 || info != 83 ||
        scale[0] != 81.0 || scale[1] != 82.0 ||
        !cf64_eq(a[0], make_cf64(801.0, 4.0))) {
        fprintf(stderr, "[FAIL] ZGEBAL CBLAS->Fortran thunk did not map double-complex balance arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] ZGEBAL CBLAS->Fortran thunk maps double-complex balance arguments into the C entry\n");
    return 0;
}

static int check_sgebak_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_sgebak_cblas_fn thunk = NULL;
    float scale[2] = { 1.0f, 2.0f };
    float v[4] = { 33.0f, 34.0f, 35.0f, 36.0f };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_gebak_fortran_call, 0, sizeof(g_gebak_fortran_call));
    vtable.ext_ops[FB_OP_SGEBAK][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_sgebak_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SGEBAK);
    thunk = (fb_sgebak_cblas_fn)vtable.ext_ops[FB_OP_SGEBAK][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SGEBAK Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    if (thunk(FB_LAYOUT_COL_MAJOR, 'B', 'R', 4, 1, 3, scale, 2, v, 2) != 93 ||
        g_gebak_fortran_call.called != 1 ||
        g_gebak_fortran_call.job != 'B' ||
        g_gebak_fortran_call.side != 'R' ||
        g_gebak_fortran_call.n != 4 ||
        g_gebak_fortran_call.ilo != 1 ||
        g_gebak_fortran_call.ihi != 3 ||
        g_gebak_fortran_call.m != 2 ||
        g_gebak_fortran_call.ldv != 2 ||
        g_gebak_fortran_call.scale != scale ||
        g_gebak_fortran_call.matrix != v ||
        v[0] != 901.0f || v[1] != 902.0f) {
        fprintf(stderr, "[FAIL] SGEBAK Fortran->CBLAS thunk did not forward back-transform arguments and outputs correctly\n");
        return 1;
    }

    printf("[PASS] SGEBAK Fortran->CBLAS thunk forwards back-transform arguments and outputs\n");
    return 0;
}

static int check_sgebak_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_sgebak_fortran_fn thunk = NULL;
    char job = 'S';
    char side = 'L';
    int n = 4;
    int ilo = 2;
    int ihi = 4;
    int m = 2;
    int ldv = 2;
    int info = -1;
    float scale[2] = { 3.0f, 4.0f };
    float v[4] = { 37.0f, 38.0f, 39.0f, 40.0f };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_gebak_cblas_call, 0, sizeof(g_gebak_cblas_call));
    vtable.ext_ops[FB_OP_SGEBAK][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_sgebak_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SGEBAK);
    thunk = (fb_sgebak_fortran_fn)vtable.ext_ops[FB_OP_SGEBAK][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SGEBAK CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&job, &side, &n, &ilo, &ihi, scale, &m, v, &ldv, &info);
    if (g_gebak_cblas_call.called != 1 ||
        g_gebak_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_gebak_cblas_call.job != 'S' ||
        g_gebak_cblas_call.side != 'L' ||
        g_gebak_cblas_call.n != 4 ||
        g_gebak_cblas_call.ilo != 2 ||
        g_gebak_cblas_call.ihi != 4 ||
        g_gebak_cblas_call.m != 2 ||
        g_gebak_cblas_call.ldv != 2 ||
        g_gebak_cblas_call.scale != scale ||
        g_gebak_cblas_call.matrix != v ||
        info != 133 || v[0] != 1301.0f || v[1] != 1302.0f) {
        fprintf(stderr, "[FAIL] SGEBAK CBLAS->Fortran thunk did not map back-transform arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] SGEBAK CBLAS->Fortran thunk maps back-transform arguments into the C entry\n");
    return 0;
}

static int check_dgebak_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_dgebak_cblas_fn thunk = NULL;
    double scale[2] = { 5.0, 6.0 };
    double v[4] = { 41.0, 42.0, 43.0, 44.0 };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_gebak_fortran_call, 0, sizeof(g_gebak_fortran_call));
    vtable.ext_ops[FB_OP_DGEBAK][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_dgebak_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_DGEBAK);
    thunk = (fb_dgebak_cblas_fn)vtable.ext_ops[FB_OP_DGEBAK][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DGEBAK Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    if (thunk(FB_LAYOUT_COL_MAJOR, 'P', 'R', 4, 1, 4, scale, 2, v, 2) != 103 ||
        g_gebak_fortran_call.called != 1 ||
        g_gebak_fortran_call.job != 'P' ||
        g_gebak_fortran_call.side != 'R' ||
        g_gebak_fortran_call.n != 4 ||
        g_gebak_fortran_call.ilo != 1 ||
        g_gebak_fortran_call.ihi != 4 ||
        g_gebak_fortran_call.m != 2 ||
        g_gebak_fortran_call.ldv != 2 ||
        g_gebak_fortran_call.scale != scale ||
        g_gebak_fortran_call.matrix != v ||
        v[0] != 1001.0 || v[1] != 1002.0) {
        fprintf(stderr, "[FAIL] DGEBAK Fortran->CBLAS thunk did not forward double back-transform arguments and outputs correctly\n");
        return 1;
    }

    printf("[PASS] DGEBAK Fortran->CBLAS thunk forwards double back-transform arguments and outputs\n");
    return 0;
}

static int check_dgebak_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_dgebak_fortran_fn thunk = NULL;
    char job = 'B';
    char side = 'L';
    int n = 4;
    int ilo = 2;
    int ihi = 3;
    int m = 2;
    int ldv = 2;
    int info = -1;
    double scale[2] = { 7.0, 8.0 };
    double v[4] = { 45.0, 46.0, 47.0, 48.0 };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_gebak_cblas_call, 0, sizeof(g_gebak_cblas_call));
    vtable.ext_ops[FB_OP_DGEBAK][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_dgebak_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_DGEBAK);
    thunk = (fb_dgebak_fortran_fn)vtable.ext_ops[FB_OP_DGEBAK][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DGEBAK CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&job, &side, &n, &ilo, &ihi, scale, &m, v, &ldv, &info);
    if (g_gebak_cblas_call.called != 1 ||
        g_gebak_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_gebak_cblas_call.job != 'B' ||
        g_gebak_cblas_call.side != 'L' ||
        g_gebak_cblas_call.n != 4 ||
        g_gebak_cblas_call.ilo != 2 ||
        g_gebak_cblas_call.ihi != 3 ||
        g_gebak_cblas_call.m != 2 ||
        g_gebak_cblas_call.ldv != 2 ||
        g_gebak_cblas_call.scale != scale ||
        g_gebak_cblas_call.matrix != v ||
        info != 143 || v[0] != 1401.0 || v[1] != 1402.0) {
        fprintf(stderr, "[FAIL] DGEBAK CBLAS->Fortran thunk did not map double back-transform arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] DGEBAK CBLAS->Fortran thunk maps double back-transform arguments into the C entry\n");
    return 0;
}

static int check_cgebak_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_cgebak_cblas_fn thunk = NULL;
    float scale[2] = { 9.0f, 10.0f };
    fb_complex_float_t v[4] = {
        make_cf32(49.0f, 1.0f), make_cf32(50.0f, 2.0f),
        make_cf32(51.0f, 3.0f), make_cf32(52.0f, 4.0f)
    };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_gebak_fortran_call, 0, sizeof(g_gebak_fortran_call));
    vtable.ext_ops[FB_OP_CGEBAK][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_cgebak_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CGEBAK);
    thunk = (fb_cgebak_cblas_fn)vtable.ext_ops[FB_OP_CGEBAK][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CGEBAK Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    if (thunk(FB_LAYOUT_COL_MAJOR, 'S', 'R', 4, 1, 3, scale, 2, v, 2) != 113 ||
        g_gebak_fortran_call.called != 1 ||
        g_gebak_fortran_call.job != 'S' ||
        g_gebak_fortran_call.side != 'R' ||
        g_gebak_fortran_call.n != 4 ||
        g_gebak_fortran_call.ilo != 1 ||
        g_gebak_fortran_call.ihi != 3 ||
        g_gebak_fortran_call.m != 2 ||
        g_gebak_fortran_call.ldv != 2 ||
        g_gebak_fortran_call.scale != scale ||
        g_gebak_fortran_call.matrix != v ||
        !cf32_eq(v[0], make_cf32(1101.0f, -5.0f)) ) {
        fprintf(stderr, "[FAIL] CGEBAK Fortran->CBLAS thunk did not forward complex back-transform arguments and outputs correctly\n");
        return 1;
    }

    printf("[PASS] CGEBAK Fortran->CBLAS thunk forwards complex back-transform arguments and outputs\n");
    return 0;
}

static int check_cgebak_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_cgebak_fortran_fn thunk = NULL;
    char job = 'B';
    char side = 'L';
    int n = 4;
    int ilo = 2;
    int ihi = 4;
    int m = 2;
    int ldv = 2;
    int info = -1;
    float scale[2] = { 11.0f, 12.0f };
    fb_complex_float_t v[4] = {
        make_cf32(53.0f, 1.0f), make_cf32(54.0f, 2.0f),
        make_cf32(55.0f, 3.0f), make_cf32(56.0f, 4.0f)
    };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_gebak_cblas_call, 0, sizeof(g_gebak_cblas_call));
    vtable.ext_ops[FB_OP_CGEBAK][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_cgebak_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CGEBAK);
    thunk = (fb_cgebak_fortran_fn)vtable.ext_ops[FB_OP_CGEBAK][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CGEBAK CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&job, &side, &n, &ilo, &ihi, scale, &m, v, &ldv, &info);
    if (g_gebak_cblas_call.called != 1 ||
        g_gebak_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_gebak_cblas_call.job != 'B' ||
        g_gebak_cblas_call.side != 'L' ||
        g_gebak_cblas_call.n != 4 ||
        g_gebak_cblas_call.ilo != 2 ||
        g_gebak_cblas_call.ihi != 4 ||
        g_gebak_cblas_call.m != 2 ||
        g_gebak_cblas_call.ldv != 2 ||
        g_gebak_cblas_call.scale != scale ||
        g_gebak_cblas_call.matrix != v ||
        info != 153 || !cf32_eq(v[0], make_cf32(1501.0f, 5.0f))) {
        fprintf(stderr, "[FAIL] CGEBAK CBLAS->Fortran thunk did not map complex back-transform arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] CGEBAK CBLAS->Fortran thunk maps complex back-transform arguments into the C entry\n");
    return 0;
}

static int check_zgebak_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_zgebak_cblas_fn thunk = NULL;
    double scale[2] = { 13.0, 14.0 };
    fb_complex_double_t v[4] = {
        make_cf64(57.0, 1.0), make_cf64(58.0, 2.0),
        make_cf64(59.0, 3.0), make_cf64(60.0, 4.0)
    };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_gebak_fortran_call, 0, sizeof(g_gebak_fortran_call));
    vtable.ext_ops[FB_OP_ZGEBAK][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_zgebak_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_ZGEBAK);
    thunk = (fb_zgebak_cblas_fn)vtable.ext_ops[FB_OP_ZGEBAK][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZGEBAK Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    if (thunk(FB_LAYOUT_COL_MAJOR, 'P', 'R', 4, 1, 4, scale, 2, v, 2) != 123 ||
        g_gebak_fortran_call.called != 1 ||
        g_gebak_fortran_call.job != 'P' ||
        g_gebak_fortran_call.side != 'R' ||
        g_gebak_fortran_call.n != 4 ||
        g_gebak_fortran_call.ilo != 1 ||
        g_gebak_fortran_call.ihi != 4 ||
        g_gebak_fortran_call.m != 2 ||
        g_gebak_fortran_call.ldv != 2 ||
        g_gebak_fortran_call.scale != scale ||
        g_gebak_fortran_call.matrix != v ||
        !cf64_eq(v[0], make_cf64(1201.0, -7.0))) {
        fprintf(stderr, "[FAIL] ZGEBAK Fortran->CBLAS thunk did not forward double-complex back-transform arguments and outputs correctly\n");
        return 1;
    }

    printf("[PASS] ZGEBAK Fortran->CBLAS thunk forwards double-complex back-transform arguments and outputs\n");
    return 0;
}

static int check_zgebak_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_zgebak_fortran_fn thunk = NULL;
    char job = 'S';
    char side = 'L';
    int n = 4;
    int ilo = 2;
    int ihi = 3;
    int m = 2;
    int ldv = 2;
    int info = -1;
    double scale[2] = { 15.0, 16.0 };
    fb_complex_double_t v[4] = {
        make_cf64(61.0, 1.0), make_cf64(62.0, 2.0),
        make_cf64(63.0, 3.0), make_cf64(64.0, 4.0)
    };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_gebak_cblas_call, 0, sizeof(g_gebak_cblas_call));
    vtable.ext_ops[FB_OP_ZGEBAK][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_zgebak_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_ZGEBAK);
    thunk = (fb_zgebak_fortran_fn)vtable.ext_ops[FB_OP_ZGEBAK][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZGEBAK CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&job, &side, &n, &ilo, &ihi, scale, &m, v, &ldv, &info);
    if (g_gebak_cblas_call.called != 1 ||
        g_gebak_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_gebak_cblas_call.job != 'S' ||
        g_gebak_cblas_call.side != 'L' ||
        g_gebak_cblas_call.n != 4 ||
        g_gebak_cblas_call.ilo != 2 ||
        g_gebak_cblas_call.ihi != 3 ||
        g_gebak_cblas_call.m != 2 ||
        g_gebak_cblas_call.ldv != 2 ||
        g_gebak_cblas_call.scale != scale ||
        g_gebak_cblas_call.matrix != v ||
        info != 163 || !cf64_eq(v[0], make_cf64(1601.0, 7.0))) {
        fprintf(stderr, "[FAIL] ZGEBAK CBLAS->Fortran thunk did not map double-complex back-transform arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] ZGEBAK CBLAS->Fortran thunk maps double-complex back-transform arguments into the C entry\n");
    return 0;
}

int main(void)
{
    int status = 0;

    status |= check_sgebal_fortran_to_cblas();
    status |= check_sgebal_cblas_to_fortran();
    status |= check_dgebal_fortran_to_cblas();
    status |= check_dgebal_cblas_to_fortran();
    status |= check_cgebal_fortran_to_cblas();
    status |= check_cgebal_cblas_to_fortran();
    status |= check_zgebal_fortran_to_cblas();
    status |= check_zgebal_cblas_to_fortran();
    status |= check_sgebak_fortran_to_cblas();
    status |= check_sgebak_cblas_to_fortran();
    status |= check_dgebak_fortran_to_cblas();
    status |= check_dgebak_cblas_to_fortran();
    status |= check_cgebak_fortran_to_cblas();
    status |= check_cgebak_cblas_to_fortran();
    status |= check_zgebak_fortran_to_cblas();
    status |= check_zgebak_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}