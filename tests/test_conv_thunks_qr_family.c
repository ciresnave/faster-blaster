#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef void (*fb_sgeqrf_fortran_slot_fn)(int *m, int *n, float *a, int *lda,
                                          float *tau, float *work, int *lwork,
                                          int *info);
typedef int (*test_dgeqrf_cblas_fn)(const fb_layout_t layout, const int m,
                                    const int n, double *a, const int lda,
                                    double *tau);
typedef void (*fb_dgeqrf_fortran_slot_fn)(int *m, int *n, double *a, int *lda,
                                          double *tau, double *work,
                                          int *lwork, int *info);
typedef int (*fb_cgerqf_cblas_fn)(const fb_layout_t layout, const int m,
                                  const int n, fb_complex_float_t *a,
                                  const int lda, fb_complex_float_t *tau);
typedef void (*fb_cgerqf_fortran_slot_fn)(int *m, int *n,
                                          fb_complex_float_t *a, int *lda,
                                          fb_complex_float_t *tau,
                                          fb_complex_float_t *work,
                                          int *lwork, int *info);
typedef int (*test_zgerqf_cblas_fn)(const fb_layout_t layout, const int m,
                                    const int n, fb_complex_double_t *a,
                                    const int lda, fb_complex_double_t *tau);
typedef void (*fb_zgerqf_fortran_slot_fn)(int *m, int *n,
                                          fb_complex_double_t *a, int *lda,
                                          fb_complex_double_t *tau,
                                          fb_complex_double_t *work,
                                          int *lwork, int *info);

static struct {
    int query_calls;
    int solve_calls;
    int m;
    int n;
    int lda;
    int lwork_query;
    int lwork_solve;
    float a_snapshot[6];
    float tau_snapshot[2];
} g_sgeqrf_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    int m;
    int n;
    int lda;
    float *a;
    float *tau;
} g_sgeqrf_cblas_call;

static struct {
    int query_calls;
    int solve_calls;
    int m;
    int n;
    int lda;
    int lwork_query;
    int lwork_solve;
    double a_snapshot[6];
    double tau_snapshot[2];
} g_dgeqrf_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    int m;
    int n;
    int lda;
    double *a;
    double *tau;
} g_dgeqrf_cblas_call;

static struct {
    int query_calls;
    int solve_calls;
    int m;
    int n;
    int lda;
    int lwork_query;
    int lwork_solve;
    float a_real_snapshot[6];
    float tau_real_snapshot[2];
} g_cgerqf_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    int m;
    int n;
    int lda;
    fb_complex_float_t *a;
    fb_complex_float_t *tau;
} g_cgerqf_cblas_call;

static struct {
    int query_calls;
    int solve_calls;
    int m;
    int n;
    int lda;
    int lwork_query;
    int lwork_solve;
    double a_real_snapshot[6];
    double tau_real_snapshot[2];
} g_zgerqf_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    int m;
    int n;
    int lda;
    fb_complex_double_t *a;
    fb_complex_double_t *tau;
} g_zgerqf_cblas_call;

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

static void stub_sgeqrf_fortran(int *m, int *n, float *a, int *lda,
                                float *tau, float *work, int *lwork,
                                int *info)
{
    int row = 0;
    int col = 0;
    int k = (*m < *n) ? *m : *n;

    if (*lwork == -1) {
        g_sgeqrf_fortran_call.query_calls += 1;
        g_sgeqrf_fortran_call.m = *m;
        g_sgeqrf_fortran_call.n = *n;
        g_sgeqrf_fortran_call.lda = *lda;
        g_sgeqrf_fortran_call.lwork_query = *lwork;
        a[0] = -999.0f;
        if (k > 0) {
            tau[0] = -777.0f;
        }
        work[0] = 5.0f;
        *info = 0;
        return;
    }

    g_sgeqrf_fortran_call.solve_calls += 1;
    g_sgeqrf_fortran_call.m = *m;
    g_sgeqrf_fortran_call.n = *n;
    g_sgeqrf_fortran_call.lda = *lda;
    g_sgeqrf_fortran_call.lwork_solve = *lwork;
    for (col = 0; col < *n; ++col) {
        for (row = 0; row < *m; ++row) {
            g_sgeqrf_fortran_call.a_snapshot[(col * (*m)) + row] =
                a[(col * (*lda)) + row];
            a[(col * (*lda)) + row] = (float)(100 + (10 * row) + col);
        }
    }
    for (row = 0; row < k; ++row) {
        g_sgeqrf_fortran_call.tau_snapshot[row] = tau[row];
        tau[row] = (float)(31 + row);
    }

    *info = 0;
}

static int stub_sgeqrf_cblas(const fb_layout_t layout, const int m,
                             const int n, float *a, const int lda,
                             float *tau)
{
    g_sgeqrf_cblas_call.called += 1;
    g_sgeqrf_cblas_call.layout = layout;
    g_sgeqrf_cblas_call.m = m;
    g_sgeqrf_cblas_call.n = n;
    g_sgeqrf_cblas_call.lda = lda;
    g_sgeqrf_cblas_call.a = a;
    g_sgeqrf_cblas_call.tau = tau;
    return 51;
}

static void stub_dgeqrf_fortran(int *m, int *n, double *a, int *lda,
                                double *tau, double *work, int *lwork,
                                int *info)
{
    int row = 0;
    int col = 0;
    int k = (*m < *n) ? *m : *n;

    if (*lwork == -1) {
        g_dgeqrf_fortran_call.query_calls += 1;
        g_dgeqrf_fortran_call.m = *m;
        g_dgeqrf_fortran_call.n = *n;
        g_dgeqrf_fortran_call.lda = *lda;
        g_dgeqrf_fortran_call.lwork_query = *lwork;
        a[0] = -999.0;
        if (k > 0) {
            tau[0] = -777.0;
        }
        work[0] = 7.0;
        *info = 0;
        return;
    }

    g_dgeqrf_fortran_call.solve_calls += 1;
    g_dgeqrf_fortran_call.m = *m;
    g_dgeqrf_fortran_call.n = *n;
    g_dgeqrf_fortran_call.lda = *lda;
    g_dgeqrf_fortran_call.lwork_solve = *lwork;
    for (col = 0; col < *n; ++col) {
        for (row = 0; row < *m; ++row) {
            g_dgeqrf_fortran_call.a_snapshot[(col * (*m)) + row] =
                a[(col * (*lda)) + row];
            a[(col * (*lda)) + row] = (double)(300 + (10 * row) + col);
        }
    }
    for (row = 0; row < k; ++row) {
        g_dgeqrf_fortran_call.tau_snapshot[row] = tau[row];
        tau[row] = (double)(61 + row);
    }

    *info = 0;
}

static int stub_dgeqrf_cblas(const fb_layout_t layout, const int m,
                             const int n, double *a, const int lda,
                             double *tau)
{
    g_dgeqrf_cblas_call.called += 1;
    g_dgeqrf_cblas_call.layout = layout;
    g_dgeqrf_cblas_call.m = m;
    g_dgeqrf_cblas_call.n = n;
    g_dgeqrf_cblas_call.lda = lda;
    g_dgeqrf_cblas_call.a = a;
    g_dgeqrf_cblas_call.tau = tau;
    return 52;
}

static void stub_cgerqf_fortran(int *m, int *n, fb_complex_float_t *a,
                                int *lda, fb_complex_float_t *tau,
                                fb_complex_float_t *work, int *lwork,
                                int *info)
{
    int row = 0;
    int col = 0;
    int k = (*m < *n) ? *m : *n;

    if (*lwork == -1) {
        g_cgerqf_fortran_call.query_calls += 1;
        g_cgerqf_fortran_call.m = *m;
        g_cgerqf_fortran_call.n = *n;
        g_cgerqf_fortran_call.lda = *lda;
        g_cgerqf_fortran_call.lwork_query = *lwork;
        a[0] = make_cfloat(-999.0f);
        if (k > 0) {
            tau[0] = make_cfloat(-777.0f);
        }
        work[0] = make_cfloat(6.0f);
        *info = 0;
        return;
    }

    g_cgerqf_fortran_call.solve_calls += 1;
    g_cgerqf_fortran_call.m = *m;
    g_cgerqf_fortran_call.n = *n;
    g_cgerqf_fortran_call.lda = *lda;
    g_cgerqf_fortran_call.lwork_solve = *lwork;
    for (col = 0; col < *n; ++col) {
        for (row = 0; row < *m; ++row) {
            g_cgerqf_fortran_call.a_real_snapshot[(col * (*m)) + row] =
                cfloat_real(a[(col * (*lda)) + row]);
            a[(col * (*lda)) + row] = make_cfloat((float)(200 + (10 * row) + col));
        }
    }
    for (row = 0; row < k; ++row) {
        g_cgerqf_fortran_call.tau_real_snapshot[row] = cfloat_real(tau[row]);
        tau[row] = make_cfloat((float)(41 + row));
    }

    *info = 0;
}

static int stub_cgerqf_cblas(const fb_layout_t layout, const int m,
                             const int n, fb_complex_float_t *a,
                             const int lda, fb_complex_float_t *tau)
{
    g_cgerqf_cblas_call.called += 1;
    g_cgerqf_cblas_call.layout = layout;
    g_cgerqf_cblas_call.m = m;
    g_cgerqf_cblas_call.n = n;
    g_cgerqf_cblas_call.lda = lda;
    g_cgerqf_cblas_call.a = a;
    g_cgerqf_cblas_call.tau = tau;
    return 53;
}

static void stub_zgerqf_fortran(int *m, int *n, fb_complex_double_t *a,
                                int *lda, fb_complex_double_t *tau,
                                fb_complex_double_t *work, int *lwork,
                                int *info)
{
    int row = 0;
    int col = 0;
    int k = (*m < *n) ? *m : *n;

    if (*lwork == -1) {
        g_zgerqf_fortran_call.query_calls += 1;
        g_zgerqf_fortran_call.m = *m;
        g_zgerqf_fortran_call.n = *n;
        g_zgerqf_fortran_call.lda = *lda;
        g_zgerqf_fortran_call.lwork_query = *lwork;
        a[0] = make_cdouble(-999.0);
        if (k > 0) {
            tau[0] = make_cdouble(-777.0);
        }
        work[0] = make_cdouble(8.0);
        *info = 0;
        return;
    }

    g_zgerqf_fortran_call.solve_calls += 1;
    g_zgerqf_fortran_call.m = *m;
    g_zgerqf_fortran_call.n = *n;
    g_zgerqf_fortran_call.lda = *lda;
    g_zgerqf_fortran_call.lwork_solve = *lwork;
    for (col = 0; col < *n; ++col) {
        for (row = 0; row < *m; ++row) {
            g_zgerqf_fortran_call.a_real_snapshot[(col * (*m)) + row] =
                cdouble_real(a[(col * (*lda)) + row]);
            a[(col * (*lda)) + row] = make_cdouble((double)(400 + (10 * row) + col));
        }
    }
    for (row = 0; row < k; ++row) {
        g_zgerqf_fortran_call.tau_real_snapshot[row] = cdouble_real(tau[row]);
        tau[row] = make_cdouble((double)(71 + row));
    }

    *info = 0;
}

static int stub_zgerqf_cblas(const fb_layout_t layout, const int m,
                             const int n, fb_complex_double_t *a,
                             const int lda, fb_complex_double_t *tau)
{
    g_zgerqf_cblas_call.called += 1;
    g_zgerqf_cblas_call.layout = layout;
    g_zgerqf_cblas_call.m = m;
    g_zgerqf_cblas_call.n = n;
    g_zgerqf_cblas_call.lda = lda;
    g_zgerqf_cblas_call.a = a;
    g_zgerqf_cblas_call.tau = tau;
    return 54;
}

static int check_sgeqrf_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_sgeqrf_fn thunk = NULL;
    float a[6] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
    float tau[2] = { 0.0f, 0.0f };
    float expected_a_snapshot[6] = { 1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f };
    float expected_a_out[6] = { 100.0f, 101.0f, 102.0f, 110.0f, 111.0f, 112.0f };
    float expected_tau_out[2] = { 31.0f, 32.0f };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sgeqrf_fortran_call, 0, sizeof(g_sgeqrf_fortran_call));

    vtable.ext_ops[FB_OP_SGEQRF][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_sgeqrf_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SGEQRF);

    thunk = (fb_sgeqrf_fn)vtable.ext_ops[FB_OP_SGEQRF][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SGEQRF Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 2, 3, a, 3, tau);
    if (info != 0 || g_sgeqrf_fortran_call.query_calls != 1 ||
        g_sgeqrf_fortran_call.solve_calls != 1 ||
        g_sgeqrf_fortran_call.m != 2 || g_sgeqrf_fortran_call.n != 3 ||
        g_sgeqrf_fortran_call.lda != 2 ||
        g_sgeqrf_fortran_call.lwork_query != -1 ||
        g_sgeqrf_fortran_call.lwork_solve != 5 ||
        memcmp(g_sgeqrf_fortran_call.a_snapshot, expected_a_snapshot,
               sizeof(expected_a_snapshot)) != 0 ||
        memcmp(a, expected_a_out, sizeof(expected_a_out)) != 0 ||
        memcmp(tau, expected_tau_out, sizeof(expected_tau_out)) != 0) {
        fprintf(stderr, "[FAIL] SGEQRF Fortran->CBLAS thunk did not preserve row-major workspace-query semantics\n");
        return 1;
    }

    printf("[PASS] SGEQRF Fortran->CBLAS thunk translates row-major matrices and reruns after lwork query\n");
    return 0;
}

static int check_sgeqrf_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_sgeqrf_fortran_slot_fn thunk = NULL;
    float a[6] = { 0.0f };
    float tau[2] = { 0.0f, 0.0f };
    float work[4] = { 0.0f };
    int m = 2;
    int n = 3;
    int lda = 2;
    int lwork = 4;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sgeqrf_cblas_call, 0, sizeof(g_sgeqrf_cblas_call));

    vtable.ext_ops[FB_OP_SGEQRF][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_sgeqrf_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SGEQRF);

    thunk = (fb_sgeqrf_fortran_slot_fn)vtable.ext_ops[FB_OP_SGEQRF][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SGEQRF CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&m, &n, a, &lda, tau, work, &lwork, &info);
    if (info != 51 || g_sgeqrf_cblas_call.called != 1 ||
        g_sgeqrf_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_sgeqrf_cblas_call.m != 2 || g_sgeqrf_cblas_call.n != 3 ||
        g_sgeqrf_cblas_call.lda != 2 ||
        g_sgeqrf_cblas_call.a != a || g_sgeqrf_cblas_call.tau != tau) {
        fprintf(stderr, "[FAIL] SGEQRF CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] SGEQRF CBLAS->Fortran thunk maps the all-pointer ABI into the C factorization entry\n");
    return 0;
}

static int check_dgeqrf_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    test_dgeqrf_cblas_fn thunk = NULL;
    double a[6] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    double tau[2] = { 0.0, 0.0 };
    double expected_a_snapshot[6] = { 1.0, 4.0, 2.0, 5.0, 3.0, 6.0 };
    double expected_a_out[6] = { 300.0, 301.0, 302.0, 310.0, 311.0, 312.0 };
    double expected_tau_out[2] = { 61.0, 62.0 };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dgeqrf_fortran_call, 0, sizeof(g_dgeqrf_fortran_call));

    vtable.ext_ops[FB_OP_DGEQRF][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_dgeqrf_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_DGEQRF);

    thunk = (test_dgeqrf_cblas_fn)vtable.ext_ops[FB_OP_DGEQRF][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DGEQRF Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 2, 3, a, 3, tau);
    if (info != 0 || g_dgeqrf_fortran_call.query_calls != 1 ||
        g_dgeqrf_fortran_call.solve_calls != 1 ||
        g_dgeqrf_fortran_call.m != 2 || g_dgeqrf_fortran_call.n != 3 ||
        g_dgeqrf_fortran_call.lda != 2 ||
        g_dgeqrf_fortran_call.lwork_query != -1 ||
        g_dgeqrf_fortran_call.lwork_solve != 7 ||
        memcmp(g_dgeqrf_fortran_call.a_snapshot, expected_a_snapshot,
               sizeof(expected_a_snapshot)) != 0 ||
        memcmp(a, expected_a_out, sizeof(expected_a_out)) != 0 ||
        memcmp(tau, expected_tau_out, sizeof(expected_tau_out)) != 0) {
        fprintf(stderr, "[FAIL] DGEQRF Fortran->CBLAS thunk did not preserve row-major workspace-query semantics\n");
        return 1;
    }

    printf("[PASS] DGEQRF Fortran->CBLAS thunk translates row-major matrices and reruns after lwork query\n");
    return 0;
}

static int check_dgeqrf_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_dgeqrf_fortran_slot_fn thunk = NULL;
    double a[6] = { 0.0 };
    double tau[2] = { 0.0, 0.0 };
    double work[4] = { 0.0 };
    int m = 2;
    int n = 3;
    int lda = 2;
    int lwork = 4;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dgeqrf_cblas_call, 0, sizeof(g_dgeqrf_cblas_call));

    vtable.ext_ops[FB_OP_DGEQRF][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_dgeqrf_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_DGEQRF);

    thunk = (fb_dgeqrf_fortran_slot_fn)vtable.ext_ops[FB_OP_DGEQRF][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DGEQRF CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&m, &n, a, &lda, tau, work, &lwork, &info);
    if (info != 52 || g_dgeqrf_cblas_call.called != 1 ||
        g_dgeqrf_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_dgeqrf_cblas_call.m != 2 || g_dgeqrf_cblas_call.n != 3 ||
        g_dgeqrf_cblas_call.lda != 2 ||
        g_dgeqrf_cblas_call.a != a || g_dgeqrf_cblas_call.tau != tau) {
        fprintf(stderr, "[FAIL] DGEQRF CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] DGEQRF CBLAS->Fortran thunk maps the all-pointer ABI into the double C factorization entry\n");
    return 0;
}

static int check_cgerqf_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_cgerqf_cblas_fn thunk = NULL;
    fb_complex_float_t a[6] = {
        make_cfloat(1.0f), make_cfloat(2.0f), make_cfloat(3.0f),
        make_cfloat(4.0f), make_cfloat(5.0f), make_cfloat(6.0f)
    };
    fb_complex_float_t tau[2] = { make_cfloat(0.0f), make_cfloat(0.0f) };
    float expected_a_snapshot[6] = { 1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f };
    float expected_a_out[6] = { 200.0f, 201.0f, 202.0f, 210.0f, 211.0f, 212.0f };
    float expected_tau_out[2] = { 41.0f, 42.0f };
    int info = 0;
    int idx = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cgerqf_fortran_call, 0, sizeof(g_cgerqf_fortran_call));

    vtable.ext_ops[FB_OP_CGERQF][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_cgerqf_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CGERQF);

    thunk = (fb_cgerqf_cblas_fn)vtable.ext_ops[FB_OP_CGERQF][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CGERQF Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 2, 3, a, 3, tau);
    if (info != 0 || g_cgerqf_fortran_call.query_calls != 1 ||
        g_cgerqf_fortran_call.solve_calls != 1 ||
        g_cgerqf_fortran_call.m != 2 || g_cgerqf_fortran_call.n != 3 ||
        g_cgerqf_fortran_call.lda != 2 ||
        g_cgerqf_fortran_call.lwork_query != -1 ||
        g_cgerqf_fortran_call.lwork_solve != 6 ||
        memcmp(g_cgerqf_fortran_call.a_real_snapshot, expected_a_snapshot,
               sizeof(expected_a_snapshot)) != 0) {
        fprintf(stderr, "[FAIL] CGERQF Fortran->CBLAS thunk did not preserve complex row-major workspace-query semantics\n");
        return 1;
    }

    for (idx = 0; idx < 6; ++idx) {
        if (cfloat_real(a[idx]) != expected_a_out[idx]) {
            fprintf(stderr, "[FAIL] CGERQF Fortran->CBLAS thunk did not copy row-major matrix output back correctly\n");
            return 1;
        }
    }
    for (idx = 0; idx < 2; ++idx) {
        if (cfloat_real(tau[idx]) != expected_tau_out[idx]) {
            fprintf(stderr, "[FAIL] CGERQF Fortran->CBLAS thunk did not copy complex TAU output back correctly\n");
            return 1;
        }
    }

    printf("[PASS] CGERQF Fortran->CBLAS thunk translates row-major matrices and extracts complex lwork query size\n");
    return 0;
}

static int check_cgerqf_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_cgerqf_fortran_slot_fn thunk = NULL;
    fb_complex_float_t a[6];
    fb_complex_float_t tau[2];
    fb_complex_float_t work[4];
    int m = 2;
    int n = 3;
    int lda = 2;
    int lwork = 4;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cgerqf_cblas_call, 0, sizeof(g_cgerqf_cblas_call));
    memset(a, 0, sizeof(a));
    memset(tau, 0, sizeof(tau));
    memset(work, 0, sizeof(work));

    vtable.ext_ops[FB_OP_CGERQF][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_cgerqf_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CGERQF);

    thunk = (fb_cgerqf_fortran_slot_fn)vtable.ext_ops[FB_OP_CGERQF][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CGERQF CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&m, &n, a, &lda, tau, work, &lwork, &info);
    if (info != 53 || g_cgerqf_cblas_call.called != 1 ||
        g_cgerqf_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_cgerqf_cblas_call.m != 2 || g_cgerqf_cblas_call.n != 3 ||
        g_cgerqf_cblas_call.lda != 2 ||
        g_cgerqf_cblas_call.a != a || g_cgerqf_cblas_call.tau != tau) {
        fprintf(stderr, "[FAIL] CGERQF CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] CGERQF CBLAS->Fortran thunk maps the all-pointer ABI into the complex C factorization entry\n");
    return 0;
}

static int check_zgerqf_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    test_zgerqf_cblas_fn thunk = NULL;
    fb_complex_double_t a[6] = {
        make_cdouble(1.0), make_cdouble(2.0), make_cdouble(3.0),
        make_cdouble(4.0), make_cdouble(5.0), make_cdouble(6.0)
    };
    fb_complex_double_t tau[2] = { make_cdouble(0.0), make_cdouble(0.0) };
    double expected_a_snapshot[6] = { 1.0, 4.0, 2.0, 5.0, 3.0, 6.0 };
    double expected_a_out[6] = { 400.0, 401.0, 402.0, 410.0, 411.0, 412.0 };
    double expected_tau_out[2] = { 71.0, 72.0 };
    int info = 0;
    int idx = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zgerqf_fortran_call, 0, sizeof(g_zgerqf_fortran_call));

    vtable.ext_ops[FB_OP_ZGERQF][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_zgerqf_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_ZGERQF);

    thunk = (test_zgerqf_cblas_fn)vtable.ext_ops[FB_OP_ZGERQF][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZGERQF Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 2, 3, a, 3, tau);
    if (info != 0 || g_zgerqf_fortran_call.query_calls != 1 ||
        g_zgerqf_fortran_call.solve_calls != 1 ||
        g_zgerqf_fortran_call.m != 2 || g_zgerqf_fortran_call.n != 3 ||
        g_zgerqf_fortran_call.lda != 2 ||
        g_zgerqf_fortran_call.lwork_query != -1 ||
        g_zgerqf_fortran_call.lwork_solve != 8 ||
        memcmp(g_zgerqf_fortran_call.a_real_snapshot, expected_a_snapshot,
               sizeof(expected_a_snapshot)) != 0) {
        fprintf(stderr, "[FAIL] ZGERQF Fortran->CBLAS thunk did not preserve complex-double row-major workspace-query semantics\n");
        return 1;
    }

    for (idx = 0; idx < 6; ++idx) {
        if (cdouble_real(a[idx]) != expected_a_out[idx]) {
            fprintf(stderr, "[FAIL] ZGERQF Fortran->CBLAS thunk did not copy row-major matrix output back correctly\n");
            return 1;
        }
    }
    for (idx = 0; idx < 2; ++idx) {
        if (cdouble_real(tau[idx]) != expected_tau_out[idx]) {
            fprintf(stderr, "[FAIL] ZGERQF Fortran->CBLAS thunk did not copy complex-double TAU output back correctly\n");
            return 1;
        }
    }

    printf("[PASS] ZGERQF Fortran->CBLAS thunk translates row-major matrices and extracts complex-double lwork query size\n");
    return 0;
}

static int check_zgerqf_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_zgerqf_fortran_slot_fn thunk = NULL;
    fb_complex_double_t a[6];
    fb_complex_double_t tau[2];
    fb_complex_double_t work[4];
    int m = 2;
    int n = 3;
    int lda = 2;
    int lwork = 4;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zgerqf_cblas_call, 0, sizeof(g_zgerqf_cblas_call));
    memset(a, 0, sizeof(a));
    memset(tau, 0, sizeof(tau));
    memset(work, 0, sizeof(work));

    vtable.ext_ops[FB_OP_ZGERQF][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_zgerqf_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_ZGERQF);

    thunk = (fb_zgerqf_fortran_slot_fn)vtable.ext_ops[FB_OP_ZGERQF][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZGERQF CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&m, &n, a, &lda, tau, work, &lwork, &info);
    if (info != 54 || g_zgerqf_cblas_call.called != 1 ||
        g_zgerqf_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_zgerqf_cblas_call.m != 2 || g_zgerqf_cblas_call.n != 3 ||
        g_zgerqf_cblas_call.lda != 2 ||
        g_zgerqf_cblas_call.a != a || g_zgerqf_cblas_call.tau != tau) {
        fprintf(stderr, "[FAIL] ZGERQF CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] ZGERQF CBLAS->Fortran thunk maps the all-pointer ABI into the complex-double C factorization entry\n");
    return 0;
}

int main(void)
{
    if (check_sgeqrf_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_sgeqrf_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_dgeqrf_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_dgeqrf_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_cgerqf_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_cgerqf_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_zgerqf_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_zgerqf_cblas_to_fortran() != 0) {
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}