#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef int (*fb_sgebd2_cblas_fn)(fb_layout_t layout, int m, int n, float *a,
                                  int lda, float *d, float *e, float *tauq,
                                  float *taup);
typedef int (*fb_dgebd2_cblas_fn)(fb_layout_t layout, int m, int n, double *a,
                                  int lda, double *d, double *e,
                                  double *tauq, double *taup);
typedef int (*fb_cgebd2_cblas_fn)(fb_layout_t layout, int m, int n,
                                  fb_complex_float_t *a, int lda, float *d,
                                  float *e, fb_complex_float_t *tauq,
                                  fb_complex_float_t *taup);
typedef int (*fb_zgebd2_cblas_fn)(fb_layout_t layout, int m, int n,
                                  fb_complex_double_t *a, int lda, double *d,
                                  double *e, fb_complex_double_t *tauq,
                                  fb_complex_double_t *taup);

typedef void (*fb_sgebd2_fortran_fn)(int *m, int *n, float *a, int *lda,
                                     float *d, float *e, float *tauq,
                                     float *taup, float *work, int *info);
typedef void (*fb_dgebd2_fortran_fn)(int *m, int *n, double *a, int *lda,
                                     double *d, double *e, double *tauq,
                                     double *taup, double *work, int *info);
typedef void (*fb_cgebd2_fortran_fn)(int *m, int *n, fb_complex_float_t *a,
                                     int *lda, float *d, float *e,
                                     fb_complex_float_t *tauq,
                                     fb_complex_float_t *taup,
                                     fb_complex_float_t *work, int *info);
typedef void (*fb_zgebd2_fortran_fn)(int *m, int *n, fb_complex_double_t *a,
                                     int *lda, double *d, double *e,
                                     fb_complex_double_t *tauq,
                                     fb_complex_double_t *taup,
                                     fb_complex_double_t *work, int *info);

typedef struct {
    int called;
    int m;
    int n;
    int lda;
    int work_present;
    void *a;
    void *d;
    void *e;
    void *tauq;
    void *taup;
} gebd2_fortran_call_t;

typedef struct {
    int called;
    fb_layout_t layout;
    int m;
    int n;
    int lda;
    void *a;
    void *d;
    void *e;
    void *tauq;
    void *taup;
} gebd2_cblas_call_t;

static gebd2_fortran_call_t g_sgebd2_fortran_call;
static gebd2_fortran_call_t g_dgebd2_fortran_call;
static gebd2_fortran_call_t g_cgebd2_fortran_call;
static gebd2_fortran_call_t g_zgebd2_fortran_call;
static gebd2_cblas_call_t g_sgebd2_cblas_call;
static gebd2_cblas_call_t g_dgebd2_cblas_call;
static gebd2_cblas_call_t g_cgebd2_cblas_call;
static gebd2_cblas_call_t g_zgebd2_cblas_call;
static int g_sgebd2_cblas_rc = 0;
static int g_dgebd2_cblas_rc = 0;
static int g_cgebd2_cblas_rc = 0;
static int g_zgebd2_cblas_rc = 0;

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

static void stub_sgebd2_fortran(int *m, int *n, float *a, int *lda, float *d,
                                float *e, float *tauq, float *taup,
                                float *work, int *info)
{
    g_sgebd2_fortran_call.called += 1;
    g_sgebd2_fortran_call.m = *m;
    g_sgebd2_fortran_call.n = *n;
    g_sgebd2_fortran_call.lda = *lda;
    g_sgebd2_fortran_call.work_present = (work != NULL);
    g_sgebd2_fortran_call.a = a;
    g_sgebd2_fortran_call.d = d;
    g_sgebd2_fortran_call.e = e;
    g_sgebd2_fortran_call.tauq = tauq;
    g_sgebd2_fortran_call.taup = taup;
    d[0] = 101.0f;
    d[1] = 102.0f;
    e[0] = 103.0f;
    tauq[0] = 104.0f;
    tauq[1] = 105.0f;
    taup[0] = 106.0f;
    taup[1] = 107.0f;
    *info = 0;
}

static void stub_dgebd2_fortran(int *m, int *n, double *a, int *lda,
                                double *d, double *e, double *tauq,
                                double *taup, double *work, int *info)
{
    g_dgebd2_fortran_call.called += 1;
    g_dgebd2_fortran_call.m = *m;
    g_dgebd2_fortran_call.n = *n;
    g_dgebd2_fortran_call.lda = *lda;
    g_dgebd2_fortran_call.work_present = (work != NULL);
    g_dgebd2_fortran_call.a = a;
    g_dgebd2_fortran_call.d = d;
    g_dgebd2_fortran_call.e = e;
    g_dgebd2_fortran_call.tauq = tauq;
    g_dgebd2_fortran_call.taup = taup;
    d[0] = 201.0;
    d[1] = 202.0;
    e[0] = 203.0;
    tauq[0] = 204.0;
    tauq[1] = 205.0;
    taup[0] = 206.0;
    taup[1] = 207.0;
    *info = 0;
}

static void stub_cgebd2_fortran(int *m, int *n, fb_complex_float_t *a,
                                int *lda, float *d, float *e,
                                fb_complex_float_t *tauq,
                                fb_complex_float_t *taup,
                                fb_complex_float_t *work, int *info)
{
    g_cgebd2_fortran_call.called += 1;
    g_cgebd2_fortran_call.m = *m;
    g_cgebd2_fortran_call.n = *n;
    g_cgebd2_fortran_call.lda = *lda;
    g_cgebd2_fortran_call.work_present = (work != NULL);
    g_cgebd2_fortran_call.a = a;
    g_cgebd2_fortran_call.d = d;
    g_cgebd2_fortran_call.e = e;
    g_cgebd2_fortran_call.tauq = tauq;
    g_cgebd2_fortran_call.taup = taup;
    d[0] = 301.0f;
    d[1] = 302.0f;
    e[0] = 303.0f;
    tauq[0] = make_cf32(304.0f, 4.0f);
    tauq[1] = make_cf32(305.0f, 5.0f);
    taup[0] = make_cf32(306.0f, 6.0f);
    taup[1] = make_cf32(307.0f, 7.0f);
    *info = 0;
}

static void stub_zgebd2_fortran(int *m, int *n, fb_complex_double_t *a,
                                int *lda, double *d, double *e,
                                fb_complex_double_t *tauq,
                                fb_complex_double_t *taup,
                                fb_complex_double_t *work, int *info)
{
    g_zgebd2_fortran_call.called += 1;
    g_zgebd2_fortran_call.m = *m;
    g_zgebd2_fortran_call.n = *n;
    g_zgebd2_fortran_call.lda = *lda;
    g_zgebd2_fortran_call.work_present = (work != NULL);
    g_zgebd2_fortran_call.a = a;
    g_zgebd2_fortran_call.d = d;
    g_zgebd2_fortran_call.e = e;
    g_zgebd2_fortran_call.tauq = tauq;
    g_zgebd2_fortran_call.taup = taup;
    d[0] = 401.0;
    d[1] = 402.0;
    e[0] = 403.0;
    tauq[0] = make_cf64(404.0, 4.0);
    tauq[1] = make_cf64(405.0, 5.0);
    taup[0] = make_cf64(406.0, 6.0);
    taup[1] = make_cf64(407.0, 7.0);
    *info = 0;
}

static int stub_sgebd2_cblas(fb_layout_t layout, int m, int n, float *a,
                             int lda, float *d, float *e, float *tauq,
                             float *taup)
{
    g_sgebd2_cblas_call.called += 1;
    g_sgebd2_cblas_call.layout = layout;
    g_sgebd2_cblas_call.m = m;
    g_sgebd2_cblas_call.n = n;
    g_sgebd2_cblas_call.lda = lda;
    g_sgebd2_cblas_call.a = a;
    g_sgebd2_cblas_call.d = d;
    g_sgebd2_cblas_call.e = e;
    g_sgebd2_cblas_call.tauq = tauq;
    g_sgebd2_cblas_call.taup = taup;
    d[0] = 111.0f;
    d[1] = 112.0f;
    e[0] = 113.0f;
    tauq[0] = 114.0f;
    tauq[1] = 115.0f;
    taup[0] = 116.0f;
    taup[1] = 117.0f;
    return g_sgebd2_cblas_rc;
}

static int stub_dgebd2_cblas(fb_layout_t layout, int m, int n, double *a,
                             int lda, double *d, double *e, double *tauq,
                             double *taup)
{
    g_dgebd2_cblas_call.called += 1;
    g_dgebd2_cblas_call.layout = layout;
    g_dgebd2_cblas_call.m = m;
    g_dgebd2_cblas_call.n = n;
    g_dgebd2_cblas_call.lda = lda;
    g_dgebd2_cblas_call.a = a;
    g_dgebd2_cblas_call.d = d;
    g_dgebd2_cblas_call.e = e;
    g_dgebd2_cblas_call.tauq = tauq;
    g_dgebd2_cblas_call.taup = taup;
    d[0] = 211.0;
    d[1] = 212.0;
    e[0] = 213.0;
    tauq[0] = 214.0;
    tauq[1] = 215.0;
    taup[0] = 216.0;
    taup[1] = 217.0;
    return g_dgebd2_cblas_rc;
}

static int stub_cgebd2_cblas(fb_layout_t layout, int m, int n,
                             fb_complex_float_t *a, int lda, float *d,
                             float *e, fb_complex_float_t *tauq,
                             fb_complex_float_t *taup)
{
    g_cgebd2_cblas_call.called += 1;
    g_cgebd2_cblas_call.layout = layout;
    g_cgebd2_cblas_call.m = m;
    g_cgebd2_cblas_call.n = n;
    g_cgebd2_cblas_call.lda = lda;
    g_cgebd2_cblas_call.a = a;
    g_cgebd2_cblas_call.d = d;
    g_cgebd2_cblas_call.e = e;
    g_cgebd2_cblas_call.tauq = tauq;
    g_cgebd2_cblas_call.taup = taup;
    d[0] = 311.0f;
    d[1] = 312.0f;
    e[0] = 313.0f;
    tauq[0] = make_cf32(314.0f, 14.0f);
    tauq[1] = make_cf32(315.0f, 15.0f);
    taup[0] = make_cf32(316.0f, 16.0f);
    taup[1] = make_cf32(317.0f, 17.0f);
    return g_cgebd2_cblas_rc;
}

static int stub_zgebd2_cblas(fb_layout_t layout, int m, int n,
                             fb_complex_double_t *a, int lda, double *d,
                             double *e, fb_complex_double_t *tauq,
                             fb_complex_double_t *taup)
{
    g_zgebd2_cblas_call.called += 1;
    g_zgebd2_cblas_call.layout = layout;
    g_zgebd2_cblas_call.m = m;
    g_zgebd2_cblas_call.n = n;
    g_zgebd2_cblas_call.lda = lda;
    g_zgebd2_cblas_call.a = a;
    g_zgebd2_cblas_call.d = d;
    g_zgebd2_cblas_call.e = e;
    g_zgebd2_cblas_call.tauq = tauq;
    g_zgebd2_cblas_call.taup = taup;
    d[0] = 411.0;
    d[1] = 412.0;
    e[0] = 413.0;
    tauq[0] = make_cf64(414.0, 14.0);
    tauq[1] = make_cf64(415.0, 15.0);
    taup[0] = make_cf64(416.0, 16.0);
    taup[1] = make_cf64(417.0, 17.0);
    return g_zgebd2_cblas_rc;
}

static int check_sgebd2_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_sgebd2_cblas_fn thunk = NULL;
    float a[6] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
    float d[2] = { 0.0f, 0.0f };
    float e[1] = { 0.0f };
    float tauq[2] = { 0.0f, 0.0f };
    float taup[2] = { 0.0f, 0.0f };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sgebd2_fortran_call, 0, sizeof(g_sgebd2_fortran_call));
    vtable.ext_ops[FB_OP_SGEBD2][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_sgebd2_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SGEBD2);
    thunk = (fb_sgebd2_cblas_fn)vtable.ext_ops[FB_OP_SGEBD2][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SGEBD2 Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 3, 2, a, 3, d, e, tauq, taup);
    if (info != 0 || g_sgebd2_fortran_call.called != 1 ||
        g_sgebd2_fortran_call.m != 3 || g_sgebd2_fortran_call.n != 2 ||
        g_sgebd2_fortran_call.lda != 3 ||
        !g_sgebd2_fortran_call.work_present ||
        g_sgebd2_fortran_call.a != a || g_sgebd2_fortran_call.d != d ||
        g_sgebd2_fortran_call.e != e || g_sgebd2_fortran_call.tauq != tauq ||
        g_sgebd2_fortran_call.taup != taup ||
        d[0] != 101.0f || d[1] != 102.0f || e[0] != 103.0f ||
        tauq[0] != 104.0f || tauq[1] != 105.0f ||
        taup[0] != 106.0f || taup[1] != 107.0f) {
        fprintf(stderr, "[FAIL] SGEBD2 Fortran->CBLAS thunk did not forward unblocked bidiagonal outputs correctly\n");
        return 1;
    }

    printf("[PASS] SGEBD2 Fortran->CBLAS thunk forwards unblocked bidiagonal outputs\n");
    return 0;
}

static int check_sgebd2_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_sgebd2_fortran_fn thunk = NULL;
    int m = 3;
    int n = 2;
    int lda = 3;
    int info = -1;
    float a[6] = { 0.0f };
    float d[2] = { 0.0f, 0.0f };
    float e[1] = { 0.0f };
    float tauq[2] = { 0.0f, 0.0f };
    float taup[2] = { 0.0f, 0.0f };
    float work[3] = { 0.0f, 0.0f, 0.0f };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sgebd2_cblas_call, 0, sizeof(g_sgebd2_cblas_call));
    g_sgebd2_cblas_rc = 121;
    vtable.ext_ops[FB_OP_SGEBD2][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_sgebd2_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SGEBD2);
    thunk = (fb_sgebd2_fortran_fn)vtable.ext_ops[FB_OP_SGEBD2][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SGEBD2 CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&m, &n, a, &lda, d, e, tauq, taup, work, &info);
    if (g_sgebd2_cblas_call.called != 1 ||
        g_sgebd2_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_sgebd2_cblas_call.m != 3 || g_sgebd2_cblas_call.n != 2 ||
        g_sgebd2_cblas_call.lda != 3 || g_sgebd2_cblas_call.a != a ||
        g_sgebd2_cblas_call.d != d || g_sgebd2_cblas_call.e != e ||
        g_sgebd2_cblas_call.tauq != tauq || g_sgebd2_cblas_call.taup != taup ||
        info != 121 || d[0] != 111.0f || d[1] != 112.0f || e[0] != 113.0f ||
        tauq[0] != 114.0f || tauq[1] != 115.0f ||
        taup[0] != 116.0f || taup[1] != 117.0f) {
        fprintf(stderr, "[FAIL] SGEBD2 CBLAS->Fortran thunk did not map unblocked bidiagonal arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] SGEBD2 CBLAS->Fortran thunk maps unblocked bidiagonal arguments into the C entry\n");
    return 0;
}

static int check_dgebd2_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_dgebd2_cblas_fn thunk = NULL;
    double a[6] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    double d[2] = { 0.0, 0.0 };
    double e[1] = { 0.0 };
    double tauq[2] = { 0.0, 0.0 };
    double taup[2] = { 0.0, 0.0 };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dgebd2_fortran_call, 0, sizeof(g_dgebd2_fortran_call));
    vtable.ext_ops[FB_OP_DGEBD2][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_dgebd2_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_DGEBD2);
    thunk = (fb_dgebd2_cblas_fn)vtable.ext_ops[FB_OP_DGEBD2][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DGEBD2 Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 3, 2, a, 3, d, e, tauq, taup);
    if (info != 0 || g_dgebd2_fortran_call.called != 1 ||
        g_dgebd2_fortran_call.m != 3 || g_dgebd2_fortran_call.n != 2 ||
        g_dgebd2_fortran_call.lda != 3 ||
        !g_dgebd2_fortran_call.work_present ||
        g_dgebd2_fortran_call.a != a || g_dgebd2_fortran_call.d != d ||
        g_dgebd2_fortran_call.e != e || g_dgebd2_fortran_call.tauq != tauq ||
        g_dgebd2_fortran_call.taup != taup ||
        d[0] != 201.0 || d[1] != 202.0 || e[0] != 203.0 ||
        tauq[0] != 204.0 || tauq[1] != 205.0 ||
        taup[0] != 206.0 || taup[1] != 207.0) {
        fprintf(stderr, "[FAIL] DGEBD2 Fortran->CBLAS thunk did not forward double unblocked bidiagonal outputs correctly\n");
        return 1;
    }

    printf("[PASS] DGEBD2 Fortran->CBLAS thunk forwards double unblocked bidiagonal outputs\n");
    return 0;
}

static int check_dgebd2_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_dgebd2_fortran_fn thunk = NULL;
    int m = 3;
    int n = 2;
    int lda = 3;
    int info = -1;
    double a[6] = { 0.0 };
    double d[2] = { 0.0, 0.0 };
    double e[1] = { 0.0 };
    double tauq[2] = { 0.0, 0.0 };
    double taup[2] = { 0.0, 0.0 };
    double work[3] = { 0.0, 0.0, 0.0 };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dgebd2_cblas_call, 0, sizeof(g_dgebd2_cblas_call));
    g_dgebd2_cblas_rc = 221;
    vtable.ext_ops[FB_OP_DGEBD2][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_dgebd2_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_DGEBD2);
    thunk = (fb_dgebd2_fortran_fn)vtable.ext_ops[FB_OP_DGEBD2][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DGEBD2 CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&m, &n, a, &lda, d, e, tauq, taup, work, &info);
    if (g_dgebd2_cblas_call.called != 1 ||
        g_dgebd2_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_dgebd2_cblas_call.m != 3 || g_dgebd2_cblas_call.n != 2 ||
        g_dgebd2_cblas_call.lda != 3 || g_dgebd2_cblas_call.a != a ||
        g_dgebd2_cblas_call.d != d || g_dgebd2_cblas_call.e != e ||
        g_dgebd2_cblas_call.tauq != tauq || g_dgebd2_cblas_call.taup != taup ||
        info != 221 || d[0] != 211.0 || d[1] != 212.0 || e[0] != 213.0 ||
        tauq[0] != 214.0 || tauq[1] != 215.0 ||
        taup[0] != 216.0 || taup[1] != 217.0) {
        fprintf(stderr, "[FAIL] DGEBD2 CBLAS->Fortran thunk did not map double unblocked bidiagonal arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] DGEBD2 CBLAS->Fortran thunk maps double unblocked bidiagonal arguments into the C entry\n");
    return 0;
}

static int check_cgebd2_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_cgebd2_cblas_fn thunk = NULL;
    fb_complex_float_t a[6] = {
        make_cf32(1.0f, 1.0f), make_cf32(2.0f, 2.0f),
        make_cf32(3.0f, 3.0f), make_cf32(4.0f, 4.0f),
        make_cf32(5.0f, 5.0f), make_cf32(6.0f, 6.0f)
    };
    float d[2] = { 0.0f, 0.0f };
    float e[1] = { 0.0f };
    fb_complex_float_t tauq[2] = { 0 };
    fb_complex_float_t taup[2] = { 0 };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cgebd2_fortran_call, 0, sizeof(g_cgebd2_fortran_call));
    vtable.ext_ops[FB_OP_CGEBD2][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_cgebd2_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CGEBD2);
    thunk = (fb_cgebd2_cblas_fn)vtable.ext_ops[FB_OP_CGEBD2][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CGEBD2 Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 3, 2, a, 3, d, e, tauq, taup);
    if (info != 0 || g_cgebd2_fortran_call.called != 1 ||
        g_cgebd2_fortran_call.m != 3 || g_cgebd2_fortran_call.n != 2 ||
        g_cgebd2_fortran_call.lda != 3 ||
        !g_cgebd2_fortran_call.work_present ||
        g_cgebd2_fortran_call.a != a || g_cgebd2_fortran_call.d != d ||
        g_cgebd2_fortran_call.e != e || g_cgebd2_fortran_call.tauq != tauq ||
        g_cgebd2_fortran_call.taup != taup ||
        d[0] != 301.0f || d[1] != 302.0f || e[0] != 303.0f ||
        !cf32_eq(tauq[0], make_cf32(304.0f, 4.0f)) ||
        !cf32_eq(tauq[1], make_cf32(305.0f, 5.0f)) ||
        !cf32_eq(taup[0], make_cf32(306.0f, 6.0f)) ||
        !cf32_eq(taup[1], make_cf32(307.0f, 7.0f))) {
        fprintf(stderr, "[FAIL] CGEBD2 Fortran->CBLAS thunk did not forward complex unblocked bidiagonal outputs correctly\n");
        return 1;
    }

    printf("[PASS] CGEBD2 Fortran->CBLAS thunk forwards complex unblocked bidiagonal outputs\n");
    return 0;
}

static int check_cgebd2_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_cgebd2_fortran_fn thunk = NULL;
    int m = 3;
    int n = 2;
    int lda = 3;
    int info = -1;
    fb_complex_float_t a[6] = { 0 };
    float d[2] = { 0.0f, 0.0f };
    float e[1] = { 0.0f };
    fb_complex_float_t tauq[2] = { 0 };
    fb_complex_float_t taup[2] = { 0 };
    fb_complex_float_t work[3] = { 0 };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cgebd2_cblas_call, 0, sizeof(g_cgebd2_cblas_call));
    g_cgebd2_cblas_rc = 321;
    vtable.ext_ops[FB_OP_CGEBD2][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_cgebd2_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CGEBD2);
    thunk = (fb_cgebd2_fortran_fn)vtable.ext_ops[FB_OP_CGEBD2][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CGEBD2 CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&m, &n, a, &lda, d, e, tauq, taup, work, &info);
    if (g_cgebd2_cblas_call.called != 1 ||
        g_cgebd2_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_cgebd2_cblas_call.m != 3 || g_cgebd2_cblas_call.n != 2 ||
        g_cgebd2_cblas_call.lda != 3 || g_cgebd2_cblas_call.a != a ||
        g_cgebd2_cblas_call.d != d || g_cgebd2_cblas_call.e != e ||
        g_cgebd2_cblas_call.tauq != tauq || g_cgebd2_cblas_call.taup != taup ||
        info != 321 || d[0] != 311.0f || d[1] != 312.0f || e[0] != 313.0f ||
        !cf32_eq(tauq[0], make_cf32(314.0f, 14.0f)) ||
        !cf32_eq(tauq[1], make_cf32(315.0f, 15.0f)) ||
        !cf32_eq(taup[0], make_cf32(316.0f, 16.0f)) ||
        !cf32_eq(taup[1], make_cf32(317.0f, 17.0f))) {
        fprintf(stderr, "[FAIL] CGEBD2 CBLAS->Fortran thunk did not map complex unblocked bidiagonal arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] CGEBD2 CBLAS->Fortran thunk maps complex unblocked bidiagonal arguments into the C entry\n");
    return 0;
}

static int check_zgebd2_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_zgebd2_cblas_fn thunk = NULL;
    fb_complex_double_t a[6] = {
        make_cf64(1.0, 1.0), make_cf64(2.0, 2.0),
        make_cf64(3.0, 3.0), make_cf64(4.0, 4.0),
        make_cf64(5.0, 5.0), make_cf64(6.0, 6.0)
    };
    double d[2] = { 0.0, 0.0 };
    double e[1] = { 0.0 };
    fb_complex_double_t tauq[2] = { 0 };
    fb_complex_double_t taup[2] = { 0 };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zgebd2_fortran_call, 0, sizeof(g_zgebd2_fortran_call));
    vtable.ext_ops[FB_OP_ZGEBD2][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_zgebd2_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_ZGEBD2);
    thunk = (fb_zgebd2_cblas_fn)vtable.ext_ops[FB_OP_ZGEBD2][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZGEBD2 Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 3, 2, a, 3, d, e, tauq, taup);
    if (info != 0 || g_zgebd2_fortran_call.called != 1 ||
        g_zgebd2_fortran_call.m != 3 || g_zgebd2_fortran_call.n != 2 ||
        g_zgebd2_fortran_call.lda != 3 ||
        !g_zgebd2_fortran_call.work_present ||
        g_zgebd2_fortran_call.a != a || g_zgebd2_fortran_call.d != d ||
        g_zgebd2_fortran_call.e != e || g_zgebd2_fortran_call.tauq != tauq ||
        g_zgebd2_fortran_call.taup != taup ||
        d[0] != 401.0 || d[1] != 402.0 || e[0] != 403.0 ||
        !cf64_eq(tauq[0], make_cf64(404.0, 4.0)) ||
        !cf64_eq(tauq[1], make_cf64(405.0, 5.0)) ||
        !cf64_eq(taup[0], make_cf64(406.0, 6.0)) ||
        !cf64_eq(taup[1], make_cf64(407.0, 7.0))) {
        fprintf(stderr, "[FAIL] ZGEBD2 Fortran->CBLAS thunk did not forward double-complex unblocked bidiagonal outputs correctly\n");
        return 1;
    }

    printf("[PASS] ZGEBD2 Fortran->CBLAS thunk forwards double-complex unblocked bidiagonal outputs\n");
    return 0;
}

static int check_zgebd2_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_zgebd2_fortran_fn thunk = NULL;
    int m = 3;
    int n = 2;
    int lda = 3;
    int info = -1;
    fb_complex_double_t a[6] = { 0 };
    double d[2] = { 0.0, 0.0 };
    double e[1] = { 0.0 };
    fb_complex_double_t tauq[2] = { 0 };
    fb_complex_double_t taup[2] = { 0 };
    fb_complex_double_t work[3] = { 0 };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zgebd2_cblas_call, 0, sizeof(g_zgebd2_cblas_call));
    g_zgebd2_cblas_rc = 421;
    vtable.ext_ops[FB_OP_ZGEBD2][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_zgebd2_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_ZGEBD2);
    thunk = (fb_zgebd2_fortran_fn)vtable.ext_ops[FB_OP_ZGEBD2][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZGEBD2 CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&m, &n, a, &lda, d, e, tauq, taup, work, &info);
    if (g_zgebd2_cblas_call.called != 1 ||
        g_zgebd2_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_zgebd2_cblas_call.m != 3 || g_zgebd2_cblas_call.n != 2 ||
        g_zgebd2_cblas_call.lda != 3 || g_zgebd2_cblas_call.a != a ||
        g_zgebd2_cblas_call.d != d || g_zgebd2_cblas_call.e != e ||
        g_zgebd2_cblas_call.tauq != tauq || g_zgebd2_cblas_call.taup != taup ||
        info != 421 || d[0] != 411.0 || d[1] != 412.0 || e[0] != 413.0 ||
        !cf64_eq(tauq[0], make_cf64(414.0, 14.0)) ||
        !cf64_eq(tauq[1], make_cf64(415.0, 15.0)) ||
        !cf64_eq(taup[0], make_cf64(416.0, 16.0)) ||
        !cf64_eq(taup[1], make_cf64(417.0, 17.0))) {
        fprintf(stderr, "[FAIL] ZGEBD2 CBLAS->Fortran thunk did not map double-complex unblocked bidiagonal arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] ZGEBD2 CBLAS->Fortran thunk maps double-complex unblocked bidiagonal arguments into the C entry\n");
    return 0;
}

int main(void)
{
    int status = 0;

    status |= check_sgebd2_fortran_to_cblas();
    status |= check_sgebd2_cblas_to_fortran();
    status |= check_dgebd2_fortran_to_cblas();
    status |= check_dgebd2_cblas_to_fortran();
    status |= check_cgebd2_fortran_to_cblas();
    status |= check_cgebd2_cblas_to_fortran();
    status |= check_zgebd2_fortran_to_cblas();
    status |= check_zgebd2_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}