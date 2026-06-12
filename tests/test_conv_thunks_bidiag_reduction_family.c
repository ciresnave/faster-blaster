#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef int (*fb_sgebrd_fn)(fb_layout_t layout, int m, int n, float *a, int lda,
                            float *d, float *e, float *tauq, float *taup);
typedef int (*fb_dgebrd_fn)(fb_layout_t layout, int m, int n, double *a, int lda,
                            double *d, double *e, double *tauq, double *taup);
typedef int (*fb_cgebrd_fn)(fb_layout_t layout, int m, int n,
                            fb_complex_float_t *a, int lda,
                            float *d, float *e,
                            fb_complex_float_t *tauq,
                            fb_complex_float_t *taup);
typedef int (*fb_zgebrd_fn)(fb_layout_t layout, int m, int n,
                            fb_complex_double_t *a, int lda,
                            double *d, double *e,
                            fb_complex_double_t *tauq,
                            fb_complex_double_t *taup);

typedef void (*fb_sgebrd_fortran_slot_fn)(int *m, int *n, float *a, int *lda,
                                          float *d, float *e, float *tauq,
                                          float *taup, float *work, int *lwork,
                                          int *info);
typedef void (*fb_dgebrd_fortran_slot_fn)(int *m, int *n, double *a, int *lda,
                                          double *d, double *e, double *tauq,
                                          double *taup, double *work, int *lwork,
                                          int *info);
typedef void (*fb_cgebrd_fortran_slot_fn)(int *m, int *n,
                                          fb_complex_float_t *a, int *lda,
                                          float *d, float *e,
                                          fb_complex_float_t *tauq,
                                          fb_complex_float_t *taup,
                                          fb_complex_float_t *work,
                                          int *lwork, int *info);
typedef void (*fb_zgebrd_fortran_slot_fn)(int *m, int *n,
                                          fb_complex_double_t *a, int *lda,
                                          double *d, double *e,
                                          fb_complex_double_t *tauq,
                                          fb_complex_double_t *taup,
                                          fb_complex_double_t *work,
                                          int *lwork, int *info);

static struct {
    int query_calls;
    int exec_calls;
    int m;
    int n;
    int lda;
    int query_lwork;
    int exec_lwork;
    float a_snapshot[6];
} g_sgebrd_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    int m;
    int n;
    int lda;
    float *a;
    float *d;
    float *e;
    float *tauq;
    float *taup;
} g_sgebrd_cblas_call;

static struct {
    int query_calls;
    int exec_calls;
    int m;
    int n;
    int lda;
    int query_lwork;
    int exec_lwork;
    double a_snapshot[6];
} g_dgebrd_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    int m;
    int n;
    int lda;
    double *a;
    double *d;
    double *e;
    double *tauq;
    double *taup;
} g_dgebrd_cblas_call;

static struct {
    int query_calls;
    int exec_calls;
    int m;
    int n;
    int lda;
    int query_lwork;
    int exec_lwork;
    float a_real_snapshot[6];
} g_cgebrd_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    int m;
    int n;
    int lda;
    fb_complex_float_t *a;
    float *d;
    float *e;
    fb_complex_float_t *tauq;
    fb_complex_float_t *taup;
} g_cgebrd_cblas_call;

static struct {
    int query_calls;
    int exec_calls;
    int m;
    int n;
    int lda;
    int query_lwork;
    int exec_lwork;
    double a_real_snapshot[6];
} g_zgebrd_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    int m;
    int n;
    int lda;
    fb_complex_double_t *a;
    double *d;
    double *e;
    fb_complex_double_t *tauq;
    fb_complex_double_t *taup;
} g_zgebrd_cblas_call;

static int g_sgebrd_cblas_rc = 0;
static int g_dgebrd_cblas_rc = 0;
static int g_cgebrd_cblas_rc = 0;
static int g_zgebrd_cblas_rc = 0;

static fb_complex_float_t make_cfloat(float real_value)
{
    fb_complex_float_t value = (fb_complex_float_t)0;
    memcpy(&value, &real_value, sizeof(real_value));
    return value;
}

static float cfloat_real(fb_complex_float_t value)
{
    float real_value = 0.0f;
    memcpy(&real_value, &value, sizeof(real_value));
    return real_value;
}

static fb_complex_double_t make_cdouble(double real_value)
{
    fb_complex_double_t value = (fb_complex_double_t)0;
    memcpy(&value, &real_value, sizeof(real_value));
    return value;
}

static double cdouble_real(fb_complex_double_t value)
{
    double real_value = 0.0;
    memcpy(&real_value, &value, sizeof(real_value));
    return real_value;
}

static void stub_sgebrd_fortran(int *m, int *n, float *a, int *lda, float *d,
                                float *e, float *tauq, float *taup,
                                float *work, int *lwork, int *info)
{
    int row = 0;
    int col = 0;

    if (*lwork == -1) {
        g_sgebrd_fortran_call.query_calls += 1;
        g_sgebrd_fortran_call.query_lwork = *lwork;
        if (work) {
            work[0] = 17.0f;
        }
        *info = 0;
        return;
    }

    g_sgebrd_fortran_call.exec_calls += 1;
    g_sgebrd_fortran_call.m = *m;
    g_sgebrd_fortran_call.n = *n;
    g_sgebrd_fortran_call.lda = *lda;
    g_sgebrd_fortran_call.exec_lwork = *lwork;
    for (col = 0; col < *n; ++col) {
        for (row = 0; row < *m; ++row) {
            g_sgebrd_fortran_call.a_snapshot[(col * (*m)) + row] =
                a[(col * (*lda)) + row];
        }
    }
    d[0] = 501.0f;
    d[1] = 502.0f;
    e[0] = 503.0f;
    tauq[0] = 504.0f;
    tauq[1] = 505.0f;
    taup[0] = 506.0f;
    taup[1] = 507.0f;
    *info = 0;
}

static int stub_sgebrd_cblas(fb_layout_t layout, int m, int n, float *a, int lda,
                             float *d, float *e, float *tauq, float *taup)
{
    g_sgebrd_cblas_call.called += 1;
    g_sgebrd_cblas_call.layout = layout;
    g_sgebrd_cblas_call.m = m;
    g_sgebrd_cblas_call.n = n;
    g_sgebrd_cblas_call.lda = lda;
    g_sgebrd_cblas_call.a = a;
    g_sgebrd_cblas_call.d = d;
    g_sgebrd_cblas_call.e = e;
    g_sgebrd_cblas_call.tauq = tauq;
    g_sgebrd_cblas_call.taup = taup;
    return g_sgebrd_cblas_rc;
}

static void stub_dgebrd_fortran(int *m, int *n, double *a, int *lda, double *d,
                                double *e, double *tauq, double *taup,
                                double *work, int *lwork, int *info)
{
    int row = 0;
    int col = 0;

    if (*lwork == -1) {
        g_dgebrd_fortran_call.query_calls += 1;
        g_dgebrd_fortran_call.query_lwork = *lwork;
        if (work) {
            work[0] = 27.0;
        }
        *info = 0;
        return;
    }

    g_dgebrd_fortran_call.exec_calls += 1;
    g_dgebrd_fortran_call.m = *m;
    g_dgebrd_fortran_call.n = *n;
    g_dgebrd_fortran_call.lda = *lda;
    g_dgebrd_fortran_call.exec_lwork = *lwork;
    for (col = 0; col < *n; ++col) {
        for (row = 0; row < *m; ++row) {
            g_dgebrd_fortran_call.a_snapshot[(col * (*m)) + row] =
                a[(col * (*lda)) + row];
        }
    }
    d[0] = 701.0;
    d[1] = 702.0;
    e[0] = 703.0;
    tauq[0] = 704.0;
    tauq[1] = 705.0;
    taup[0] = 706.0;
    taup[1] = 707.0;
    *info = 0;
}

static int stub_dgebrd_cblas(fb_layout_t layout, int m, int n, double *a, int lda,
                             double *d, double *e, double *tauq, double *taup)
{
    g_dgebrd_cblas_call.called += 1;
    g_dgebrd_cblas_call.layout = layout;
    g_dgebrd_cblas_call.m = m;
    g_dgebrd_cblas_call.n = n;
    g_dgebrd_cblas_call.lda = lda;
    g_dgebrd_cblas_call.a = a;
    g_dgebrd_cblas_call.d = d;
    g_dgebrd_cblas_call.e = e;
    g_dgebrd_cblas_call.tauq = tauq;
    g_dgebrd_cblas_call.taup = taup;
    return g_dgebrd_cblas_rc;
}

static void stub_cgebrd_fortran(int *m, int *n, fb_complex_float_t *a,
                                int *lda, float *d, float *e,
                                fb_complex_float_t *tauq,
                                fb_complex_float_t *taup,
                                fb_complex_float_t *work, int *lwork,
                                int *info)
{
    int row = 0;
    int col = 0;

    if (*lwork == -1) {
        g_cgebrd_fortran_call.query_calls += 1;
        g_cgebrd_fortran_call.query_lwork = *lwork;
        if (work) {
            work[0] = make_cfloat(18.0f);
        }
        *info = 0;
        return;
    }

    g_cgebrd_fortran_call.exec_calls += 1;
    g_cgebrd_fortran_call.m = *m;
    g_cgebrd_fortran_call.n = *n;
    g_cgebrd_fortran_call.lda = *lda;
    g_cgebrd_fortran_call.exec_lwork = *lwork;
    for (col = 0; col < *n; ++col) {
        for (row = 0; row < *m; ++row) {
            g_cgebrd_fortran_call.a_real_snapshot[(col * (*m)) + row] =
                cfloat_real(a[(col * (*lda)) + row]);
        }
    }
    d[0] = 601.0f;
    d[1] = 602.0f;
    e[0] = 603.0f;
    tauq[0] = make_cfloat(604.0f);
    tauq[1] = make_cfloat(605.0f);
    taup[0] = make_cfloat(606.0f);
    taup[1] = make_cfloat(607.0f);
    *info = 0;
}

static int stub_cgebrd_cblas(fb_layout_t layout, int m, int n,
                             fb_complex_float_t *a, int lda,
                             float *d, float *e,
                             fb_complex_float_t *tauq,
                             fb_complex_float_t *taup)
{
    g_cgebrd_cblas_call.called += 1;
    g_cgebrd_cblas_call.layout = layout;
    g_cgebrd_cblas_call.m = m;
    g_cgebrd_cblas_call.n = n;
    g_cgebrd_cblas_call.lda = lda;
    g_cgebrd_cblas_call.a = a;
    g_cgebrd_cblas_call.d = d;
    g_cgebrd_cblas_call.e = e;
    g_cgebrd_cblas_call.tauq = tauq;
    g_cgebrd_cblas_call.taup = taup;
    return g_cgebrd_cblas_rc;
}

static void stub_zgebrd_fortran(int *m, int *n, fb_complex_double_t *a,
                                int *lda, double *d, double *e,
                                fb_complex_double_t *tauq,
                                fb_complex_double_t *taup,
                                fb_complex_double_t *work, int *lwork,
                                int *info)
{
    int row = 0;
    int col = 0;

    if (*lwork == -1) {
        g_zgebrd_fortran_call.query_calls += 1;
        g_zgebrd_fortran_call.query_lwork = *lwork;
        if (work) {
            work[0] = make_cdouble(28.0);
        }
        *info = 0;
        return;
    }

    g_zgebrd_fortran_call.exec_calls += 1;
    g_zgebrd_fortran_call.m = *m;
    g_zgebrd_fortran_call.n = *n;
    g_zgebrd_fortran_call.lda = *lda;
    g_zgebrd_fortran_call.exec_lwork = *lwork;
    for (col = 0; col < *n; ++col) {
        for (row = 0; row < *m; ++row) {
            g_zgebrd_fortran_call.a_real_snapshot[(col * (*m)) + row] =
                cdouble_real(a[(col * (*lda)) + row]);
        }
    }
    d[0] = 801.0;
    d[1] = 802.0;
    e[0] = 803.0;
    tauq[0] = make_cdouble(804.0);
    tauq[1] = make_cdouble(805.0);
    taup[0] = make_cdouble(806.0);
    taup[1] = make_cdouble(807.0);
    *info = 0;
}

static int stub_zgebrd_cblas(fb_layout_t layout, int m, int n,
                             fb_complex_double_t *a, int lda,
                             double *d, double *e,
                             fb_complex_double_t *tauq,
                             fb_complex_double_t *taup)
{
    g_zgebrd_cblas_call.called += 1;
    g_zgebrd_cblas_call.layout = layout;
    g_zgebrd_cblas_call.m = m;
    g_zgebrd_cblas_call.n = n;
    g_zgebrd_cblas_call.lda = lda;
    g_zgebrd_cblas_call.a = a;
    g_zgebrd_cblas_call.d = d;
    g_zgebrd_cblas_call.e = e;
    g_zgebrd_cblas_call.tauq = tauq;
    g_zgebrd_cblas_call.taup = taup;
    return g_zgebrd_cblas_rc;
}

static int check_sgebrd_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_sgebrd_fn thunk = NULL;
    float a[6] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
    float expected_a[6] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
    float d[2] = { 0.0f, 0.0f };
    float e[1] = { 0.0f };
    float tauq[2] = { 0.0f, 0.0f };
    float taup[2] = { 0.0f, 0.0f };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sgebrd_fortran_call, 0, sizeof(g_sgebrd_fortran_call));

    vtable.ext_ops[FB_OP_SGEBRD][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_sgebrd_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SGEBRD);

    thunk = (fb_sgebrd_fn)vtable.ext_ops[FB_OP_SGEBRD][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SGEBRD Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 3, 2, a, 3, d, e, tauq, taup);
    if (info != 0 || g_sgebrd_fortran_call.query_calls != 1 ||
        g_sgebrd_fortran_call.exec_calls != 1 ||
        g_sgebrd_fortran_call.query_lwork != -1 ||
        g_sgebrd_fortran_call.exec_lwork != 17 ||
        g_sgebrd_fortran_call.m != 3 || g_sgebrd_fortran_call.n != 2 ||
        g_sgebrd_fortran_call.lda != 3 ||
        memcmp(g_sgebrd_fortran_call.a_snapshot, expected_a, sizeof(expected_a)) != 0 ||
        d[0] != 501.0f || d[1] != 502.0f || e[0] != 503.0f ||
        tauq[0] != 504.0f || tauq[1] != 505.0f ||
        taup[0] != 506.0f || taup[1] != 507.0f) {
        fprintf(stderr, "[FAIL] SGEBRD Fortran->CBLAS thunk did not preserve bidiagonal reduction query semantics\n");
        return 1;
    }

    printf("[PASS] SGEBRD Fortran->CBLAS thunk performs the workspace query and forwards D/E/TAU outputs\n");
    return 0;
}

static int check_sgebrd_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_sgebrd_fortran_slot_fn thunk = NULL;
    float a[6] = { 0.0f };
    float d[2] = { 0.0f, 0.0f };
    float e[1] = { 0.0f };
    float tauq[2] = { 0.0f, 0.0f };
    float taup[2] = { 0.0f, 0.0f };
    float work[4] = { 0.0f };
    int m = 3;
    int n = 2;
    int lda = 3;
    int lwork = 4;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sgebrd_cblas_call, 0, sizeof(g_sgebrd_cblas_call));
    g_sgebrd_cblas_rc = 171;

    vtable.ext_ops[FB_OP_SGEBRD][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_sgebrd_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SGEBRD);

    thunk = (fb_sgebrd_fortran_slot_fn)vtable.ext_ops[FB_OP_SGEBRD][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SGEBRD CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&m, &n, a, &lda, d, e, tauq, taup, work, &lwork, &info);
    if (info != 171 || g_sgebrd_cblas_call.called != 1 ||
        g_sgebrd_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_sgebrd_cblas_call.m != 3 || g_sgebrd_cblas_call.n != 2 ||
        g_sgebrd_cblas_call.lda != 3 || g_sgebrd_cblas_call.a != a ||
        g_sgebrd_cblas_call.d != d || g_sgebrd_cblas_call.e != e ||
        g_sgebrd_cblas_call.tauq != tauq || g_sgebrd_cblas_call.taup != taup) {
        fprintf(stderr, "[FAIL] SGEBRD CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] SGEBRD CBLAS->Fortran thunk maps the all-pointer ABI into the generic C bidiagonal reduction entry\n");
    return 0;
}

static int check_dgebrd_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_dgebrd_fn thunk = NULL;
    double a[6] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    double expected_a[6] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    double d[2] = { 0.0, 0.0 };
    double e[1] = { 0.0 };
    double tauq[2] = { 0.0, 0.0 };
    double taup[2] = { 0.0, 0.0 };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dgebrd_fortran_call, 0, sizeof(g_dgebrd_fortran_call));

    vtable.ext_ops[FB_OP_DGEBRD][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_dgebrd_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_DGEBRD);

    thunk = (fb_dgebrd_fn)vtable.ext_ops[FB_OP_DGEBRD][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DGEBRD Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 3, 2, a, 3, d, e, tauq, taup);
    if (info != 0 || g_dgebrd_fortran_call.query_calls != 1 ||
        g_dgebrd_fortran_call.exec_calls != 1 ||
        g_dgebrd_fortran_call.query_lwork != -1 ||
        g_dgebrd_fortran_call.exec_lwork != 27 ||
        g_dgebrd_fortran_call.m != 3 || g_dgebrd_fortran_call.n != 2 ||
        g_dgebrd_fortran_call.lda != 3 ||
        memcmp(g_dgebrd_fortran_call.a_snapshot, expected_a, sizeof(expected_a)) != 0 ||
        d[0] != 701.0 || d[1] != 702.0 || e[0] != 703.0 ||
        tauq[0] != 704.0 || tauq[1] != 705.0 ||
        taup[0] != 706.0 || taup[1] != 707.0) {
        fprintf(stderr, "[FAIL] DGEBRD Fortran->CBLAS thunk did not preserve bidiagonal reduction query semantics\n");
        return 1;
    }

    printf("[PASS] DGEBRD Fortran->CBLAS thunk performs the workspace query and forwards double D/E/TAU outputs\n");
    return 0;
}

static int check_dgebrd_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_dgebrd_fortran_slot_fn thunk = NULL;
    double a[6] = { 0.0 };
    double d[2] = { 0.0, 0.0 };
    double e[1] = { 0.0 };
    double tauq[2] = { 0.0, 0.0 };
    double taup[2] = { 0.0, 0.0 };
    double work[4] = { 0.0 };
    int m = 3;
    int n = 2;
    int lda = 3;
    int lwork = 4;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dgebrd_cblas_call, 0, sizeof(g_dgebrd_cblas_call));
    g_dgebrd_cblas_rc = 172;

    vtable.ext_ops[FB_OP_DGEBRD][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_dgebrd_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_DGEBRD);

    thunk = (fb_dgebrd_fortran_slot_fn)vtable.ext_ops[FB_OP_DGEBRD][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DGEBRD CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&m, &n, a, &lda, d, e, tauq, taup, work, &lwork, &info);
    if (info != 172 || g_dgebrd_cblas_call.called != 1 ||
        g_dgebrd_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_dgebrd_cblas_call.m != 3 || g_dgebrd_cblas_call.n != 2 ||
        g_dgebrd_cblas_call.lda != 3 || g_dgebrd_cblas_call.a != a ||
        g_dgebrd_cblas_call.d != d || g_dgebrd_cblas_call.e != e ||
        g_dgebrd_cblas_call.tauq != tauq || g_dgebrd_cblas_call.taup != taup) {
        fprintf(stderr, "[FAIL] DGEBRD CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] DGEBRD CBLAS->Fortran thunk maps the all-pointer ABI into the generic double bidiagonal reduction entry\n");
    return 0;
}

static int check_cgebrd_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_cgebrd_fn thunk = NULL;
    fb_complex_float_t a[6] = {
        make_cfloat(1.0f), make_cfloat(2.0f), make_cfloat(3.0f),
        make_cfloat(4.0f), make_cfloat(5.0f), make_cfloat(6.0f)
    };
    float expected_a[6] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
    float d[2] = { 0.0f, 0.0f };
    float e[1] = { 0.0f };
    fb_complex_float_t tauq[2] = { 0 };
    fb_complex_float_t taup[2] = { 0 };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cgebrd_fortran_call, 0, sizeof(g_cgebrd_fortran_call));

    vtable.ext_ops[FB_OP_CGEBRD][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_cgebrd_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CGEBRD);

    thunk = (fb_cgebrd_fn)vtable.ext_ops[FB_OP_CGEBRD][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CGEBRD Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 3, 2, a, 3, d, e, tauq, taup);
    if (info != 0 || g_cgebrd_fortran_call.query_calls != 1 ||
        g_cgebrd_fortran_call.exec_calls != 1 ||
        g_cgebrd_fortran_call.query_lwork != -1 ||
        g_cgebrd_fortran_call.exec_lwork != 18 ||
        g_cgebrd_fortran_call.m != 3 || g_cgebrd_fortran_call.n != 2 ||
        g_cgebrd_fortran_call.lda != 3 ||
        memcmp(g_cgebrd_fortran_call.a_real_snapshot, expected_a, sizeof(expected_a)) != 0 ||
        d[0] != 601.0f || d[1] != 602.0f || e[0] != 603.0f ||
        cfloat_real(tauq[0]) != 604.0f || cfloat_real(tauq[1]) != 605.0f ||
        cfloat_real(taup[0]) != 606.0f || cfloat_real(taup[1]) != 607.0f) {
        fprintf(stderr, "[FAIL] CGEBRD Fortran->CBLAS thunk did not preserve complex bidiagonal reduction query semantics\n");
        return 1;
    }

    printf("[PASS] CGEBRD Fortran->CBLAS thunk performs the workspace query and forwards complex D/E/TAU outputs\n");
    return 0;
}

static int check_cgebrd_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_cgebrd_fortran_slot_fn thunk = NULL;
    fb_complex_float_t a[6] = { 0 };
    float d[2] = { 0.0f, 0.0f };
    float e[1] = { 0.0f };
    fb_complex_float_t tauq[2] = { 0 };
    fb_complex_float_t taup[2] = { 0 };
    fb_complex_float_t work[4] = { 0 };
    int m = 3;
    int n = 2;
    int lda = 3;
    int lwork = 4;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cgebrd_cblas_call, 0, sizeof(g_cgebrd_cblas_call));
    g_cgebrd_cblas_rc = 173;

    vtable.ext_ops[FB_OP_CGEBRD][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_cgebrd_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CGEBRD);

    thunk = (fb_cgebrd_fortran_slot_fn)vtable.ext_ops[FB_OP_CGEBRD][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CGEBRD CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&m, &n, a, &lda, d, e, tauq, taup, work, &lwork, &info);
    if (info != 173 || g_cgebrd_cblas_call.called != 1 ||
        g_cgebrd_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_cgebrd_cblas_call.m != 3 || g_cgebrd_cblas_call.n != 2 ||
        g_cgebrd_cblas_call.lda != 3 || g_cgebrd_cblas_call.a != a ||
        g_cgebrd_cblas_call.d != d || g_cgebrd_cblas_call.e != e ||
        g_cgebrd_cblas_call.tauq != tauq || g_cgebrd_cblas_call.taup != taup) {
        fprintf(stderr, "[FAIL] CGEBRD CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] CGEBRD CBLAS->Fortran thunk maps the all-pointer ABI into the generic complex bidiagonal reduction entry\n");
    return 0;
}

static int check_zgebrd_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_zgebrd_fn thunk = NULL;
    fb_complex_double_t a[6] = {
        make_cdouble(1.0), make_cdouble(2.0), make_cdouble(3.0),
        make_cdouble(4.0), make_cdouble(5.0), make_cdouble(6.0)
    };
    double expected_a[6] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    double d[2] = { 0.0, 0.0 };
    double e[1] = { 0.0 };
    fb_complex_double_t tauq[2] = { 0 };
    fb_complex_double_t taup[2] = { 0 };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zgebrd_fortran_call, 0, sizeof(g_zgebrd_fortran_call));

    vtable.ext_ops[FB_OP_ZGEBRD][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_zgebrd_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_ZGEBRD);

    thunk = (fb_zgebrd_fn)vtable.ext_ops[FB_OP_ZGEBRD][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZGEBRD Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 3, 2, a, 3, d, e, tauq, taup);
    if (info != 0 || g_zgebrd_fortran_call.query_calls != 1 ||
        g_zgebrd_fortran_call.exec_calls != 1 ||
        g_zgebrd_fortran_call.query_lwork != -1 ||
        g_zgebrd_fortran_call.exec_lwork != 28 ||
        g_zgebrd_fortran_call.m != 3 || g_zgebrd_fortran_call.n != 2 ||
        g_zgebrd_fortran_call.lda != 3 ||
        memcmp(g_zgebrd_fortran_call.a_real_snapshot, expected_a, sizeof(expected_a)) != 0 ||
        d[0] != 801.0 || d[1] != 802.0 || e[0] != 803.0 ||
        cdouble_real(tauq[0]) != 804.0 || cdouble_real(tauq[1]) != 805.0 ||
        cdouble_real(taup[0]) != 806.0 || cdouble_real(taup[1]) != 807.0) {
        fprintf(stderr, "[FAIL] ZGEBRD Fortran->CBLAS thunk did not preserve complex-double bidiagonal reduction query semantics\n");
        return 1;
    }

    printf("[PASS] ZGEBRD Fortran->CBLAS thunk performs the workspace query and forwards complex-double D/E/TAU outputs\n");
    return 0;
}

static int check_zgebrd_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_zgebrd_fortran_slot_fn thunk = NULL;
    fb_complex_double_t a[6] = { 0 };
    double d[2] = { 0.0, 0.0 };
    double e[1] = { 0.0 };
    fb_complex_double_t tauq[2] = { 0 };
    fb_complex_double_t taup[2] = { 0 };
    fb_complex_double_t work[4] = { 0 };
    int m = 3;
    int n = 2;
    int lda = 3;
    int lwork = 4;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zgebrd_cblas_call, 0, sizeof(g_zgebrd_cblas_call));
    g_zgebrd_cblas_rc = 174;

    vtable.ext_ops[FB_OP_ZGEBRD][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_zgebrd_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_ZGEBRD);

    thunk = (fb_zgebrd_fortran_slot_fn)vtable.ext_ops[FB_OP_ZGEBRD][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZGEBRD CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&m, &n, a, &lda, d, e, tauq, taup, work, &lwork, &info);
    if (info != 174 || g_zgebrd_cblas_call.called != 1 ||
        g_zgebrd_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_zgebrd_cblas_call.m != 3 || g_zgebrd_cblas_call.n != 2 ||
        g_zgebrd_cblas_call.lda != 3 || g_zgebrd_cblas_call.a != a ||
        g_zgebrd_cblas_call.d != d || g_zgebrd_cblas_call.e != e ||
        g_zgebrd_cblas_call.tauq != tauq || g_zgebrd_cblas_call.taup != taup) {
        fprintf(stderr, "[FAIL] ZGEBRD CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] ZGEBRD CBLAS->Fortran thunk maps the all-pointer ABI into the generic complex-double bidiagonal reduction entry\n");
    return 0;
}

int main(void)
{
    if (check_sgebrd_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_sgebrd_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_dgebrd_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_dgebrd_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_cgebrd_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_cgebrd_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_zgebrd_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_zgebrd_cblas_to_fortran() != 0) {
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}