#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef void (*fb_sorgqr_fortran_slot_fn)(int *m, int *n, int *k, float *a,
                                          int *lda, float *tau, float *work,
                                          int *lwork, int *info);
typedef void (*fb_dorgqr_fortran_slot_fn)(int *m, int *n, int *k, double *a,
                                          int *lda, double *tau, double *work,
                                          int *lwork, int *info);
typedef void (*fb_cungqr_fortran_slot_fn)(int *m, int *n, int *k,
                                          fb_complex_float_t *a, int *lda,
                                          fb_complex_float_t *tau,
                                          fb_complex_float_t *work,
                                          int *lwork, int *info);
typedef void (*fb_zungqr_fortran_slot_fn)(int *m, int *n, int *k,
                                          fb_complex_double_t *a, int *lda,
                                          fb_complex_double_t *tau,
                                          fb_complex_double_t *work,
                                          int *lwork, int *info);

static struct {
    int query_calls;
    int solve_calls;
    int m;
    int n;
    int k;
    int lda;
    int lwork_query;
    int lwork_solve;
    float a_snapshot[6];
    float tau_snapshot[2];
} g_sorgqr_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    int m;
    int n;
    int k;
    int lda;
    float *a;
    const float *tau;
} g_sorgqr_cblas_call;

static struct {
    int query_calls;
    int solve_calls;
    int m;
    int n;
    int k;
    int lda;
    int lwork_query;
    int lwork_solve;
    double a_snapshot[6];
    double tau_snapshot[2];
} g_dorgqr_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    int m;
    int n;
    int k;
    int lda;
    double *a;
    const double *tau;
} g_dorgqr_cblas_call;

static struct {
    int query_calls;
    int solve_calls;
    int m;
    int n;
    int k;
    int lda;
    int lwork_query;
    int lwork_solve;
    float a_real_snapshot[6];
    float tau_real_snapshot[2];
} g_cungqr_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    int m;
    int n;
    int k;
    int lda;
    fb_complex_float_t *a;
    const fb_complex_float_t *tau;
} g_cungqr_cblas_call;

static struct {
    int query_calls;
    int solve_calls;
    int m;
    int n;
    int k;
    int lda;
    int lwork_query;
    int lwork_solve;
    double a_real_snapshot[6];
    double tau_real_snapshot[2];
} g_zungqr_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    int m;
    int n;
    int k;
    int lda;
    fb_complex_double_t *a;
    const fb_complex_double_t *tau;
} g_zungqr_cblas_call;

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

static void stub_sorgqr_fortran(int *m, int *n, int *k, float *a, int *lda,
                                float *tau, float *work, int *lwork,
                                int *info)
{
    int row = 0;
    int col = 0;

    if (*lwork == -1) {
        g_sorgqr_fortran_call.query_calls += 1;
        g_sorgqr_fortran_call.m = *m;
        g_sorgqr_fortran_call.n = *n;
        g_sorgqr_fortran_call.k = *k;
        g_sorgqr_fortran_call.lda = *lda;
        g_sorgqr_fortran_call.lwork_query = *lwork;
        a[0] = -999.0f;
        tau[0] = -777.0f;
        work[0] = 5.0f;
        *info = 0;
        return;
    }

    g_sorgqr_fortran_call.solve_calls += 1;
    g_sorgqr_fortran_call.m = *m;
    g_sorgqr_fortran_call.n = *n;
    g_sorgqr_fortran_call.k = *k;
    g_sorgqr_fortran_call.lda = *lda;
    g_sorgqr_fortran_call.lwork_solve = *lwork;
    for (col = 0; col < *n; ++col) {
        for (row = 0; row < *m; ++row) {
            g_sorgqr_fortran_call.a_snapshot[(col * (*m)) + row] =
                a[(col * (*lda)) + row];
            a[(col * (*lda)) + row] = (float)(100 + (10 * row) + col);
        }
    }
    for (row = 0; row < *k; ++row) {
        g_sorgqr_fortran_call.tau_snapshot[row] = tau[row];
    }

    *info = 0;
}

static int stub_sorgqr_cblas(const fb_layout_t layout, const int m,
                             const int n, const int k, float *a,
                             const int lda, const float *tau)
{
    g_sorgqr_cblas_call.called += 1;
    g_sorgqr_cblas_call.layout = layout;
    g_sorgqr_cblas_call.m = m;
    g_sorgqr_cblas_call.n = n;
    g_sorgqr_cblas_call.k = k;
    g_sorgqr_cblas_call.lda = lda;
    g_sorgqr_cblas_call.a = a;
    g_sorgqr_cblas_call.tau = tau;
    return 61;
}

static void stub_dorgqr_fortran(int *m, int *n, int *k, double *a, int *lda,
                                double *tau, double *work, int *lwork,
                                int *info)
{
    int row = 0;
    int col = 0;

    if (*lwork == -1) {
        g_dorgqr_fortran_call.query_calls += 1;
        g_dorgqr_fortran_call.m = *m;
        g_dorgqr_fortran_call.n = *n;
        g_dorgqr_fortran_call.k = *k;
        g_dorgqr_fortran_call.lda = *lda;
        g_dorgqr_fortran_call.lwork_query = *lwork;
        a[0] = -999.0;
        tau[0] = -777.0;
        work[0] = 7.0;
        *info = 0;
        return;
    }

    g_dorgqr_fortran_call.solve_calls += 1;
    g_dorgqr_fortran_call.m = *m;
    g_dorgqr_fortran_call.n = *n;
    g_dorgqr_fortran_call.k = *k;
    g_dorgqr_fortran_call.lda = *lda;
    g_dorgqr_fortran_call.lwork_solve = *lwork;
    for (col = 0; col < *n; ++col) {
        for (row = 0; row < *m; ++row) {
            g_dorgqr_fortran_call.a_snapshot[(col * (*m)) + row] =
                a[(col * (*lda)) + row];
            a[(col * (*lda)) + row] = (double)(300 + (10 * row) + col);
        }
    }
    for (row = 0; row < *k; ++row) {
        g_dorgqr_fortran_call.tau_snapshot[row] = tau[row];
    }

    *info = 0;
}

static int stub_dorgqr_cblas(const fb_layout_t layout, const int m,
                             const int n, const int k, double *a,
                             const int lda, const double *tau)
{
    g_dorgqr_cblas_call.called += 1;
    g_dorgqr_cblas_call.layout = layout;
    g_dorgqr_cblas_call.m = m;
    g_dorgqr_cblas_call.n = n;
    g_dorgqr_cblas_call.k = k;
    g_dorgqr_cblas_call.lda = lda;
    g_dorgqr_cblas_call.a = a;
    g_dorgqr_cblas_call.tau = tau;
    return 62;
}

static void stub_cungqr_fortran(int *m, int *n, int *k,
                                fb_complex_float_t *a, int *lda,
                                fb_complex_float_t *tau,
                                fb_complex_float_t *work, int *lwork,
                                int *info)
{
    int row = 0;
    int col = 0;

    if (*lwork == -1) {
        g_cungqr_fortran_call.query_calls += 1;
        g_cungqr_fortran_call.m = *m;
        g_cungqr_fortran_call.n = *n;
        g_cungqr_fortran_call.k = *k;
        g_cungqr_fortran_call.lda = *lda;
        g_cungqr_fortran_call.lwork_query = *lwork;
        a[0] = make_cfloat(-999.0f);
        tau[0] = make_cfloat(-777.0f);
        work[0] = make_cfloat(6.0f);
        *info = 0;
        return;
    }

    g_cungqr_fortran_call.solve_calls += 1;
    g_cungqr_fortran_call.m = *m;
    g_cungqr_fortran_call.n = *n;
    g_cungqr_fortran_call.k = *k;
    g_cungqr_fortran_call.lda = *lda;
    g_cungqr_fortran_call.lwork_solve = *lwork;
    for (col = 0; col < *n; ++col) {
        for (row = 0; row < *m; ++row) {
            g_cungqr_fortran_call.a_real_snapshot[(col * (*m)) + row] =
                cfloat_real(a[(col * (*lda)) + row]);
            a[(col * (*lda)) + row] = make_cfloat((float)(200 + (10 * row) + col));
        }
    }
    for (row = 0; row < *k; ++row) {
        g_cungqr_fortran_call.tau_real_snapshot[row] = cfloat_real(tau[row]);
    }

    *info = 0;
}

static int stub_cungqr_cblas(const fb_layout_t layout, const int m,
                             const int n, const int k,
                             fb_complex_float_t *a, const int lda,
                             const fb_complex_float_t *tau)
{
    g_cungqr_cblas_call.called += 1;
    g_cungqr_cblas_call.layout = layout;
    g_cungqr_cblas_call.m = m;
    g_cungqr_cblas_call.n = n;
    g_cungqr_cblas_call.k = k;
    g_cungqr_cblas_call.lda = lda;
    g_cungqr_cblas_call.a = a;
    g_cungqr_cblas_call.tau = tau;
    return 63;
}

static void stub_zungqr_fortran(int *m, int *n, int *k,
                                fb_complex_double_t *a, int *lda,
                                fb_complex_double_t *tau,
                                fb_complex_double_t *work, int *lwork,
                                int *info)
{
    int row = 0;
    int col = 0;

    if (*lwork == -1) {
        g_zungqr_fortran_call.query_calls += 1;
        g_zungqr_fortran_call.m = *m;
        g_zungqr_fortran_call.n = *n;
        g_zungqr_fortran_call.k = *k;
        g_zungqr_fortran_call.lda = *lda;
        g_zungqr_fortran_call.lwork_query = *lwork;
        a[0] = make_cdouble(-999.0);
        tau[0] = make_cdouble(-777.0);
        work[0] = make_cdouble(8.0);
        *info = 0;
        return;
    }

    g_zungqr_fortran_call.solve_calls += 1;
    g_zungqr_fortran_call.m = *m;
    g_zungqr_fortran_call.n = *n;
    g_zungqr_fortran_call.k = *k;
    g_zungqr_fortran_call.lda = *lda;
    g_zungqr_fortran_call.lwork_solve = *lwork;
    for (col = 0; col < *n; ++col) {
        for (row = 0; row < *m; ++row) {
            g_zungqr_fortran_call.a_real_snapshot[(col * (*m)) + row] =
                cdouble_real(a[(col * (*lda)) + row]);
            a[(col * (*lda)) + row] = make_cdouble((double)(400 + (10 * row) + col));
        }
    }
    for (row = 0; row < *k; ++row) {
        g_zungqr_fortran_call.tau_real_snapshot[row] = cdouble_real(tau[row]);
    }

    *info = 0;
}

static int stub_zungqr_cblas(const fb_layout_t layout, const int m,
                             const int n, const int k,
                             fb_complex_double_t *a, const int lda,
                             const fb_complex_double_t *tau)
{
    g_zungqr_cblas_call.called += 1;
    g_zungqr_cblas_call.layout = layout;
    g_zungqr_cblas_call.m = m;
    g_zungqr_cblas_call.n = n;
    g_zungqr_cblas_call.k = k;
    g_zungqr_cblas_call.lda = lda;
    g_zungqr_cblas_call.a = a;
    g_zungqr_cblas_call.tau = tau;
    return 64;
}

static int check_sorgqr_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_sorgqr_fn thunk = NULL;
    float a[6] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
    float tau[2] = { 7.0f, 8.0f };
    float expected_a_snapshot[6] = { 1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f };
    float expected_a_out[6] = { 100.0f, 101.0f, 102.0f, 110.0f, 111.0f, 112.0f };
    float expected_tau_in[2] = { 7.0f, 8.0f };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sorgqr_fortran_call, 0, sizeof(g_sorgqr_fortran_call));

    vtable.ext_ops[FB_OP_SORGQR][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_sorgqr_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SORGQR);

    thunk = (fb_sorgqr_fn)vtable.ext_ops[FB_OP_SORGQR][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SORGQR Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 2, 3, 2, a, 3, tau);
    if (info != 0 || g_sorgqr_fortran_call.query_calls != 1 ||
        g_sorgqr_fortran_call.solve_calls != 1 ||
        g_sorgqr_fortran_call.m != 2 || g_sorgqr_fortran_call.n != 3 ||
        g_sorgqr_fortran_call.k != 2 || g_sorgqr_fortran_call.lda != 2 ||
        g_sorgqr_fortran_call.lwork_query != -1 ||
        g_sorgqr_fortran_call.lwork_solve != 5 ||
        memcmp(g_sorgqr_fortran_call.a_snapshot, expected_a_snapshot,
               sizeof(expected_a_snapshot)) != 0 ||
        memcmp(g_sorgqr_fortran_call.tau_snapshot, expected_tau_in,
               sizeof(expected_tau_in)) != 0 ||
        memcmp(a, expected_a_out, sizeof(expected_a_out)) != 0 ||
        memcmp(tau, expected_tau_in, sizeof(expected_tau_in)) != 0) {
        fprintf(stderr, "[FAIL] SORGQR Fortran->CBLAS thunk did not preserve row-major generator semantics\n");
        return 1;
    }

    printf("[PASS] SORGQR Fortran->CBLAS thunk translates row-major matrices and preserves TAU input across lwork query\n");
    return 0;
}

static int check_sorgqr_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_sorgqr_fortran_slot_fn thunk = NULL;
    float a[6] = { 0.0f };
    float tau[2] = { 7.0f, 8.0f };
    float work[4] = { 0.0f };
    int m = 2;
    int n = 3;
    int k = 2;
    int lda = 2;
    int lwork = 4;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sorgqr_cblas_call, 0, sizeof(g_sorgqr_cblas_call));

    vtable.ext_ops[FB_OP_SORGQR][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_sorgqr_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SORGQR);

    thunk = (fb_sorgqr_fortran_slot_fn)vtable.ext_ops[FB_OP_SORGQR][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SORGQR CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&m, &n, &k, a, &lda, tau, work, &lwork, &info);
    if (info != 61 || g_sorgqr_cblas_call.called != 1 ||
        g_sorgqr_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_sorgqr_cblas_call.m != 2 || g_sorgqr_cblas_call.n != 3 ||
        g_sorgqr_cblas_call.k != 2 || g_sorgqr_cblas_call.lda != 2 ||
        g_sorgqr_cblas_call.a != a || g_sorgqr_cblas_call.tau != tau) {
        fprintf(stderr, "[FAIL] SORGQR CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] SORGQR CBLAS->Fortran thunk maps the all-pointer ABI into the C generator entry\n");
    return 0;
}

static int check_dorgqr_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_dorgqr_fn thunk = NULL;
    double a[6] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    double tau[2] = { 7.0, 8.0 };
    double expected_a_snapshot[6] = { 1.0, 4.0, 2.0, 5.0, 3.0, 6.0 };
    double expected_a_out[6] = { 300.0, 301.0, 302.0, 310.0, 311.0, 312.0 };
    double expected_tau_in[2] = { 7.0, 8.0 };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dorgqr_fortran_call, 0, sizeof(g_dorgqr_fortran_call));

    vtable.ext_ops[FB_OP_DORGQR][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_dorgqr_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_DORGQR);

    thunk = (fb_dorgqr_fn)vtable.ext_ops[FB_OP_DORGQR][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DORGQR Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 2, 3, 2, a, 3, tau);
    if (info != 0 || g_dorgqr_fortran_call.query_calls != 1 ||
        g_dorgqr_fortran_call.solve_calls != 1 ||
        g_dorgqr_fortran_call.m != 2 || g_dorgqr_fortran_call.n != 3 ||
        g_dorgqr_fortran_call.k != 2 || g_dorgqr_fortran_call.lda != 2 ||
        g_dorgqr_fortran_call.lwork_query != -1 ||
        g_dorgqr_fortran_call.lwork_solve != 7 ||
        memcmp(g_dorgqr_fortran_call.a_snapshot, expected_a_snapshot,
               sizeof(expected_a_snapshot)) != 0 ||
        memcmp(g_dorgqr_fortran_call.tau_snapshot, expected_tau_in,
               sizeof(expected_tau_in)) != 0 ||
        memcmp(a, expected_a_out, sizeof(expected_a_out)) != 0 ||
        memcmp(tau, expected_tau_in, sizeof(expected_tau_in)) != 0) {
        fprintf(stderr, "[FAIL] DORGQR Fortran->CBLAS thunk did not preserve row-major generator semantics\n");
        return 1;
    }

    printf("[PASS] DORGQR Fortran->CBLAS thunk translates row-major matrices and preserves TAU input across lwork query\n");
    return 0;
}

static int check_dorgqr_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_dorgqr_fortran_slot_fn thunk = NULL;
    double a[6] = { 0.0 };
    double tau[2] = { 7.0, 8.0 };
    double work[4] = { 0.0 };
    int m = 2;
    int n = 3;
    int k = 2;
    int lda = 2;
    int lwork = 4;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dorgqr_cblas_call, 0, sizeof(g_dorgqr_cblas_call));

    vtable.ext_ops[FB_OP_DORGQR][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_dorgqr_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_DORGQR);

    thunk = (fb_dorgqr_fortran_slot_fn)vtable.ext_ops[FB_OP_DORGQR][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DORGQR CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&m, &n, &k, a, &lda, tau, work, &lwork, &info);
    if (info != 62 || g_dorgqr_cblas_call.called != 1 ||
        g_dorgqr_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_dorgqr_cblas_call.m != 2 || g_dorgqr_cblas_call.n != 3 ||
        g_dorgqr_cblas_call.k != 2 || g_dorgqr_cblas_call.lda != 2 ||
        g_dorgqr_cblas_call.a != a || g_dorgqr_cblas_call.tau != tau) {
        fprintf(stderr, "[FAIL] DORGQR CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] DORGQR CBLAS->Fortran thunk maps the all-pointer ABI into the double C generator entry\n");
    return 0;
}

static int check_cungqr_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_cungqr_fn thunk = NULL;
    fb_complex_float_t a[6] = {
        make_cfloat(1.0f), make_cfloat(2.0f), make_cfloat(3.0f),
        make_cfloat(4.0f), make_cfloat(5.0f), make_cfloat(6.0f)
    };
    fb_complex_float_t tau[2] = { make_cfloat(9.0f), make_cfloat(10.0f) };
    float expected_a_snapshot[6] = { 1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f };
    float expected_tau_in[2] = { 9.0f, 10.0f };
    float expected_a_out[6] = { 200.0f, 201.0f, 202.0f, 210.0f, 211.0f, 212.0f };
    int info = 0;
    int idx = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cungqr_fortran_call, 0, sizeof(g_cungqr_fortran_call));

    vtable.ext_ops[FB_OP_CUNGQR][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_cungqr_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CUNGQR);

    thunk = (fb_cungqr_fn)vtable.ext_ops[FB_OP_CUNGQR][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CUNGQR Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 2, 3, 2, a, 3, tau);
    if (info != 0 || g_cungqr_fortran_call.query_calls != 1 ||
        g_cungqr_fortran_call.solve_calls != 1 ||
        g_cungqr_fortran_call.m != 2 || g_cungqr_fortran_call.n != 3 ||
        g_cungqr_fortran_call.k != 2 || g_cungqr_fortran_call.lda != 2 ||
        g_cungqr_fortran_call.lwork_query != -1 ||
        g_cungqr_fortran_call.lwork_solve != 6 ||
        memcmp(g_cungqr_fortran_call.a_real_snapshot, expected_a_snapshot,
               sizeof(expected_a_snapshot)) != 0 ||
        memcmp(g_cungqr_fortran_call.tau_real_snapshot, expected_tau_in,
               sizeof(expected_tau_in)) != 0) {
        fprintf(stderr, "[FAIL] CUNGQR Fortran->CBLAS thunk did not preserve complex generator semantics\n");
        return 1;
    }

    for (idx = 0; idx < 6; ++idx) {
        if (cfloat_real(a[idx]) != expected_a_out[idx]) {
            fprintf(stderr, "[FAIL] CUNGQR Fortran->CBLAS thunk did not copy row-major matrix output back correctly\n");
            return 1;
        }
    }
    for (idx = 0; idx < 2; ++idx) {
        if (cfloat_real(tau[idx]) != expected_tau_in[idx]) {
            fprintf(stderr, "[FAIL] CUNGQR Fortran->CBLAS thunk did not preserve complex TAU input\n");
            return 1;
        }
    }

    printf("[PASS] CUNGQR Fortran->CBLAS thunk translates row-major matrices and preserves complex TAU input across lwork query\n");
    return 0;
}

static int check_cungqr_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_cungqr_fortran_slot_fn thunk = NULL;
    fb_complex_float_t a[6];
    fb_complex_float_t tau[2] = { make_cfloat(9.0f), make_cfloat(10.0f) };
    fb_complex_float_t work[4];
    int m = 2;
    int n = 3;
    int k = 2;
    int lda = 2;
    int lwork = 4;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cungqr_cblas_call, 0, sizeof(g_cungqr_cblas_call));
    memset(a, 0, sizeof(a));
    memset(work, 0, sizeof(work));

    vtable.ext_ops[FB_OP_CUNGQR][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_cungqr_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CUNGQR);

    thunk = (fb_cungqr_fortran_slot_fn)vtable.ext_ops[FB_OP_CUNGQR][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CUNGQR CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&m, &n, &k, a, &lda, tau, work, &lwork, &info);
    if (info != 63 || g_cungqr_cblas_call.called != 1 ||
        g_cungqr_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_cungqr_cblas_call.m != 2 || g_cungqr_cblas_call.n != 3 ||
        g_cungqr_cblas_call.k != 2 || g_cungqr_cblas_call.lda != 2 ||
        g_cungqr_cblas_call.a != a || g_cungqr_cblas_call.tau != tau) {
        fprintf(stderr, "[FAIL] CUNGQR CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] CUNGQR CBLAS->Fortran thunk maps the all-pointer ABI into the complex generator entry\n");
    return 0;
}

static int check_zungqr_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_zungqr_fn thunk = NULL;
    fb_complex_double_t a[6] = {
        make_cdouble(1.0), make_cdouble(2.0), make_cdouble(3.0),
        make_cdouble(4.0), make_cdouble(5.0), make_cdouble(6.0)
    };
    fb_complex_double_t tau[2] = { make_cdouble(9.0), make_cdouble(10.0) };
    double expected_a_snapshot[6] = { 1.0, 4.0, 2.0, 5.0, 3.0, 6.0 };
    double expected_tau_in[2] = { 9.0, 10.0 };
    double expected_a_out[6] = { 400.0, 401.0, 402.0, 410.0, 411.0, 412.0 };
    int info = 0;
    int idx = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zungqr_fortran_call, 0, sizeof(g_zungqr_fortran_call));

    vtable.ext_ops[FB_OP_ZUNGQR][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_zungqr_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_ZUNGQR);

    thunk = (fb_zungqr_fn)vtable.ext_ops[FB_OP_ZUNGQR][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZUNGQR Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 2, 3, 2, a, 3, tau);
    if (info != 0 || g_zungqr_fortran_call.query_calls != 1 ||
        g_zungqr_fortran_call.solve_calls != 1 ||
        g_zungqr_fortran_call.m != 2 || g_zungqr_fortran_call.n != 3 ||
        g_zungqr_fortran_call.k != 2 || g_zungqr_fortran_call.lda != 2 ||
        g_zungqr_fortran_call.lwork_query != -1 ||
        g_zungqr_fortran_call.lwork_solve != 8 ||
        memcmp(g_zungqr_fortran_call.a_real_snapshot, expected_a_snapshot,
               sizeof(expected_a_snapshot)) != 0 ||
        memcmp(g_zungqr_fortran_call.tau_real_snapshot, expected_tau_in,
               sizeof(expected_tau_in)) != 0) {
        fprintf(stderr, "[FAIL] ZUNGQR Fortran->CBLAS thunk did not preserve complex-double generator semantics\n");
        return 1;
    }

    for (idx = 0; idx < 6; ++idx) {
        if (cdouble_real(a[idx]) != expected_a_out[idx]) {
            fprintf(stderr, "[FAIL] ZUNGQR Fortran->CBLAS thunk did not copy row-major matrix output back correctly\n");
            return 1;
        }
    }
    for (idx = 0; idx < 2; ++idx) {
        if (cdouble_real(tau[idx]) != expected_tau_in[idx]) {
            fprintf(stderr, "[FAIL] ZUNGQR Fortran->CBLAS thunk did not preserve complex-double TAU input\n");
            return 1;
        }
    }

    printf("[PASS] ZUNGQR Fortran->CBLAS thunk translates row-major matrices and preserves complex-double TAU input across lwork query\n");
    return 0;
}

static int check_zungqr_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_zungqr_fortran_slot_fn thunk = NULL;
    fb_complex_double_t a[6];
    fb_complex_double_t tau[2] = { make_cdouble(9.0), make_cdouble(10.0) };
    fb_complex_double_t work[4];
    int m = 2;
    int n = 3;
    int k = 2;
    int lda = 2;
    int lwork = 4;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zungqr_cblas_call, 0, sizeof(g_zungqr_cblas_call));
    memset(a, 0, sizeof(a));
    memset(work, 0, sizeof(work));

    vtable.ext_ops[FB_OP_ZUNGQR][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_zungqr_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_ZUNGQR);

    thunk = (fb_zungqr_fortran_slot_fn)vtable.ext_ops[FB_OP_ZUNGQR][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZUNGQR CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&m, &n, &k, a, &lda, tau, work, &lwork, &info);
    if (info != 64 || g_zungqr_cblas_call.called != 1 ||
        g_zungqr_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_zungqr_cblas_call.m != 2 || g_zungqr_cblas_call.n != 3 ||
        g_zungqr_cblas_call.k != 2 || g_zungqr_cblas_call.lda != 2 ||
        g_zungqr_cblas_call.a != a || g_zungqr_cblas_call.tau != tau) {
        fprintf(stderr, "[FAIL] ZUNGQR CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] ZUNGQR CBLAS->Fortran thunk maps the all-pointer ABI into the complex-double generator entry\n");
    return 0;
}

int main(void)
{
    if (check_sorgqr_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_sorgqr_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_dorgqr_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_dorgqr_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_cungqr_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_cungqr_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_zungqr_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_zungqr_cblas_to_fortran() != 0) {
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}