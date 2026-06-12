#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef void (*fb_sormqr_fortran_slot_fn)(char *side, char *trans, int *m,
                                          int *n, int *k, float *a, int *lda,
                                          float *tau, float *c, int *ldc,
                                          float *work, int *lwork, int *info);
typedef void (*fb_dormqr_fortran_slot_fn)(char *side, char *trans, int *m,
                                          int *n, int *k, double *a, int *lda,
                                          double *tau, double *c, int *ldc,
                                          double *work, int *lwork, int *info);
typedef void (*fb_cunmqr_fortran_slot_fn)(char *side, char *trans, int *m,
                                          int *n, int *k,
                                          fb_complex_float_t *a, int *lda,
                                          fb_complex_float_t *tau,
                                          fb_complex_float_t *c, int *ldc,
                                          fb_complex_float_t *work,
                                          int *lwork, int *info);
typedef void (*fb_zunmqr_fortran_slot_fn)(char *side, char *trans, int *m,
                                          int *n, int *k,
                                          fb_complex_double_t *a, int *lda,
                                          fb_complex_double_t *tau,
                                          fb_complex_double_t *c, int *ldc,
                                          fb_complex_double_t *work,
                                          int *lwork, int *info);

static struct {
    int query_calls;
    int solve_calls;
    char side;
    char trans;
    int m;
    int n;
    int k;
    int lda;
    int ldc;
    int lwork_query;
    int lwork_solve;
    float a_snapshot[4];
    float c_snapshot[6];
    float tau_snapshot[2];
} g_sormqr_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    fb_side_t side;
    fb_transpose_t trans;
    int m;
    int n;
    int k;
    int lda;
    int ldc;
    const float *a;
    const float *tau;
    float *c;
} g_sormqr_cblas_call;

static struct {
    int query_calls;
    int solve_calls;
    char side;
    char trans;
    int m;
    int n;
    int k;
    int lda;
    int ldc;
    int lwork_query;
    int lwork_solve;
    double a_snapshot[4];
    double c_snapshot[6];
    double tau_snapshot[2];
} g_dormqr_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    fb_side_t side;
    fb_transpose_t trans;
    int m;
    int n;
    int k;
    int lda;
    int ldc;
    const double *a;
    const double *tau;
    double *c;
} g_dormqr_cblas_call;

static struct {
    int query_calls;
    int solve_calls;
    char side;
    char trans;
    int m;
    int n;
    int k;
    int lda;
    int ldc;
    int lwork_query;
    int lwork_solve;
    float a_real_snapshot[6];
    float c_real_snapshot[6];
    float tau_real_snapshot[2];
} g_cunmqr_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    fb_side_t side;
    fb_transpose_t trans;
    int m;
    int n;
    int k;
    int lda;
    int ldc;
    const fb_complex_float_t *a;
    const fb_complex_float_t *tau;
    fb_complex_float_t *c;
} g_cunmqr_cblas_call;

static struct {
    int query_calls;
    int solve_calls;
    char side;
    char trans;
    int m;
    int n;
    int k;
    int lda;
    int ldc;
    int lwork_query;
    int lwork_solve;
    double a_real_snapshot[6];
    double c_real_snapshot[6];
    double tau_real_snapshot[2];
} g_zunmqr_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    fb_side_t side;
    fb_transpose_t trans;
    int m;
    int n;
    int k;
    int lda;
    int ldc;
    const fb_complex_double_t *a;
    const fb_complex_double_t *tau;
    fb_complex_double_t *c;
} g_zunmqr_cblas_call;

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

static void stub_sormqr_fortran(char *side, char *trans, int *m, int *n,
                                int *k, float *a, int *lda, float *tau,
                                float *c, int *ldc, float *work, int *lwork,
                                int *info)
{
    int row = 0;
    int col = 0;

    if (*lwork == -1) {
        g_sormqr_fortran_call.query_calls += 1;
        g_sormqr_fortran_call.side = *side;
        g_sormqr_fortran_call.trans = *trans;
        g_sormqr_fortran_call.m = *m;
        g_sormqr_fortran_call.n = *n;
        g_sormqr_fortran_call.k = *k;
        g_sormqr_fortran_call.lda = *lda;
        g_sormqr_fortran_call.ldc = *ldc;
        g_sormqr_fortran_call.lwork_query = *lwork;
        a[0] = -999.0f;
        tau[0] = -777.0f;
        c[0] = -555.0f;
        work[0] = 7.0f;
        *info = 0;
        return;
    }

    g_sormqr_fortran_call.solve_calls += 1;
    g_sormqr_fortran_call.side = *side;
    g_sormqr_fortran_call.trans = *trans;
    g_sormqr_fortran_call.m = *m;
    g_sormqr_fortran_call.n = *n;
    g_sormqr_fortran_call.k = *k;
    g_sormqr_fortran_call.lda = *lda;
    g_sormqr_fortran_call.ldc = *ldc;
    g_sormqr_fortran_call.lwork_solve = *lwork;
    for (col = 0; col < *k; ++col) {
        for (row = 0; row < *lda; ++row) {
            g_sormqr_fortran_call.a_snapshot[(col * (*lda)) + row] =
                a[(col * (*lda)) + row];
        }
    }
    for (col = 0; col < *n; ++col) {
        for (row = 0; row < *m; ++row) {
            g_sormqr_fortran_call.c_snapshot[(col * (*m)) + row] =
                c[(col * (*ldc)) + row];
            c[(col * (*ldc)) + row] = (float)(300 + (10 * row) + col);
        }
    }
    for (row = 0; row < *k; ++row) {
        g_sormqr_fortran_call.tau_snapshot[row] = tau[row];
    }
    a[0] = -123.0f;
    tau[0] = -456.0f;

    *info = 0;
}

static int stub_sormqr_cblas(const fb_layout_t layout, const fb_side_t side,
                             const fb_transpose_t trans, const int m,
                             const int n, const int k, const float *a,
                             const int lda, const float *tau, float *c,
                             const int ldc)
{
    g_sormqr_cblas_call.called += 1;
    g_sormqr_cblas_call.layout = layout;
    g_sormqr_cblas_call.side = side;
    g_sormqr_cblas_call.trans = trans;
    g_sormqr_cblas_call.m = m;
    g_sormqr_cblas_call.n = n;
    g_sormqr_cblas_call.k = k;
    g_sormqr_cblas_call.lda = lda;
    g_sormqr_cblas_call.ldc = ldc;
    g_sormqr_cblas_call.a = a;
    g_sormqr_cblas_call.tau = tau;
    g_sormqr_cblas_call.c = c;
    return 71;
}

static void stub_dormqr_fortran(char *side, char *trans, int *m, int *n,
                                int *k, double *a, int *lda, double *tau,
                                double *c, int *ldc, double *work, int *lwork,
                                int *info)
{
    int row = 0;
    int col = 0;

    if (*lwork == -1) {
        g_dormqr_fortran_call.query_calls += 1;
        g_dormqr_fortran_call.side = *side;
        g_dormqr_fortran_call.trans = *trans;
        g_dormqr_fortran_call.m = *m;
        g_dormqr_fortran_call.n = *n;
        g_dormqr_fortran_call.k = *k;
        g_dormqr_fortran_call.lda = *lda;
        g_dormqr_fortran_call.ldc = *ldc;
        g_dormqr_fortran_call.lwork_query = *lwork;
        a[0] = -999.0;
        tau[0] = -777.0;
        c[0] = -555.0;
        work[0] = 17.0;
        *info = 0;
        return;
    }

    g_dormqr_fortran_call.solve_calls += 1;
    g_dormqr_fortran_call.side = *side;
    g_dormqr_fortran_call.trans = *trans;
    g_dormqr_fortran_call.m = *m;
    g_dormqr_fortran_call.n = *n;
    g_dormqr_fortran_call.k = *k;
    g_dormqr_fortran_call.lda = *lda;
    g_dormqr_fortran_call.ldc = *ldc;
    g_dormqr_fortran_call.lwork_solve = *lwork;
    for (col = 0; col < *k; ++col) {
        for (row = 0; row < *lda; ++row) {
            g_dormqr_fortran_call.a_snapshot[(col * (*lda)) + row] =
                a[(col * (*lda)) + row];
        }
    }
    for (col = 0; col < *n; ++col) {
        for (row = 0; row < *m; ++row) {
            g_dormqr_fortran_call.c_snapshot[(col * (*m)) + row] =
                c[(col * (*ldc)) + row];
            c[(col * (*ldc)) + row] = (double)(500 + (10 * row) + col);
        }
    }
    for (row = 0; row < *k; ++row) {
        g_dormqr_fortran_call.tau_snapshot[row] = tau[row];
    }
    a[0] = -223.0;
    tau[0] = -556.0;

    *info = 0;
}

static int stub_dormqr_cblas(const fb_layout_t layout, const fb_side_t side,
                             const fb_transpose_t trans, const int m,
                             const int n, const int k, const double *a,
                             const int lda, const double *tau, double *c,
                             const int ldc)
{
    g_dormqr_cblas_call.called += 1;
    g_dormqr_cblas_call.layout = layout;
    g_dormqr_cblas_call.side = side;
    g_dormqr_cblas_call.trans = trans;
    g_dormqr_cblas_call.m = m;
    g_dormqr_cblas_call.n = n;
    g_dormqr_cblas_call.k = k;
    g_dormqr_cblas_call.lda = lda;
    g_dormqr_cblas_call.ldc = ldc;
    g_dormqr_cblas_call.a = a;
    g_dormqr_cblas_call.tau = tau;
    g_dormqr_cblas_call.c = c;
    return 72;
}

static void stub_cunmqr_fortran(char *side, char *trans, int *m, int *n,
                                int *k, fb_complex_float_t *a, int *lda,
                                fb_complex_float_t *tau,
                                fb_complex_float_t *c, int *ldc,
                                fb_complex_float_t *work, int *lwork,
                                int *info)
{
    int row = 0;
    int col = 0;

    if (*lwork == -1) {
        g_cunmqr_fortran_call.query_calls += 1;
        g_cunmqr_fortran_call.side = *side;
        g_cunmqr_fortran_call.trans = *trans;
        g_cunmqr_fortran_call.m = *m;
        g_cunmqr_fortran_call.n = *n;
        g_cunmqr_fortran_call.k = *k;
        g_cunmqr_fortran_call.lda = *lda;
        g_cunmqr_fortran_call.ldc = *ldc;
        g_cunmqr_fortran_call.lwork_query = *lwork;
        a[0] = make_cfloat(-999.0f);
        tau[0] = make_cfloat(-777.0f);
        c[0] = make_cfloat(-555.0f);
        work[0] = make_cfloat(8.0f);
        *info = 0;
        return;
    }

    g_cunmqr_fortran_call.solve_calls += 1;
    g_cunmqr_fortran_call.side = *side;
    g_cunmqr_fortran_call.trans = *trans;
    g_cunmqr_fortran_call.m = *m;
    g_cunmqr_fortran_call.n = *n;
    g_cunmqr_fortran_call.k = *k;
    g_cunmqr_fortran_call.lda = *lda;
    g_cunmqr_fortran_call.ldc = *ldc;
    g_cunmqr_fortran_call.lwork_solve = *lwork;
    for (col = 0; col < *k; ++col) {
        for (row = 0; row < *lda; ++row) {
            g_cunmqr_fortran_call.a_real_snapshot[(col * (*lda)) + row] =
                cfloat_real(a[(col * (*lda)) + row]);
        }
    }
    for (col = 0; col < *n; ++col) {
        for (row = 0; row < *m; ++row) {
            g_cunmqr_fortran_call.c_real_snapshot[(col * (*m)) + row] =
                cfloat_real(c[(col * (*ldc)) + row]);
            c[(col * (*ldc)) + row] = make_cfloat((float)(400 + (10 * row) + col));
        }
    }
    for (row = 0; row < *k; ++row) {
        g_cunmqr_fortran_call.tau_real_snapshot[row] = cfloat_real(tau[row]);
    }
    a[0] = make_cfloat(-123.0f);
    tau[0] = make_cfloat(-456.0f);

    *info = 0;
}

static int stub_cunmqr_cblas(const fb_layout_t layout, const fb_side_t side,
                             const fb_transpose_t trans, const int m,
                             const int n, const int k,
                             const fb_complex_float_t *a, const int lda,
                             const fb_complex_float_t *tau,
                             fb_complex_float_t *c, const int ldc)
{
    g_cunmqr_cblas_call.called += 1;
    g_cunmqr_cblas_call.layout = layout;
    g_cunmqr_cblas_call.side = side;
    g_cunmqr_cblas_call.trans = trans;
    g_cunmqr_cblas_call.m = m;
    g_cunmqr_cblas_call.n = n;
    g_cunmqr_cblas_call.k = k;
    g_cunmqr_cblas_call.lda = lda;
    g_cunmqr_cblas_call.ldc = ldc;
    g_cunmqr_cblas_call.a = a;
    g_cunmqr_cblas_call.tau = tau;
    g_cunmqr_cblas_call.c = c;
    return 73;
}

static void stub_zunmqr_fortran(char *side, char *trans, int *m, int *n,
                                int *k, fb_complex_double_t *a, int *lda,
                                fb_complex_double_t *tau,
                                fb_complex_double_t *c, int *ldc,
                                fb_complex_double_t *work, int *lwork,
                                int *info)
{
    int row = 0;
    int col = 0;

    if (*lwork == -1) {
        g_zunmqr_fortran_call.query_calls += 1;
        g_zunmqr_fortran_call.side = *side;
        g_zunmqr_fortran_call.trans = *trans;
        g_zunmqr_fortran_call.m = *m;
        g_zunmqr_fortran_call.n = *n;
        g_zunmqr_fortran_call.k = *k;
        g_zunmqr_fortran_call.lda = *lda;
        g_zunmqr_fortran_call.ldc = *ldc;
        g_zunmqr_fortran_call.lwork_query = *lwork;
        a[0] = make_cdouble(-999.0);
        tau[0] = make_cdouble(-777.0);
        c[0] = make_cdouble(-555.0);
        work[0] = make_cdouble(18.0);
        *info = 0;
        return;
    }

    g_zunmqr_fortran_call.solve_calls += 1;
    g_zunmqr_fortran_call.side = *side;
    g_zunmqr_fortran_call.trans = *trans;
    g_zunmqr_fortran_call.m = *m;
    g_zunmqr_fortran_call.n = *n;
    g_zunmqr_fortran_call.k = *k;
    g_zunmqr_fortran_call.lda = *lda;
    g_zunmqr_fortran_call.ldc = *ldc;
    g_zunmqr_fortran_call.lwork_solve = *lwork;
    for (col = 0; col < *k; ++col) {
        for (row = 0; row < *lda; ++row) {
            g_zunmqr_fortran_call.a_real_snapshot[(col * (*lda)) + row] =
                cdouble_real(a[(col * (*lda)) + row]);
        }
    }
    for (col = 0; col < *n; ++col) {
        for (row = 0; row < *m; ++row) {
            g_zunmqr_fortran_call.c_real_snapshot[(col * (*m)) + row] =
                cdouble_real(c[(col * (*ldc)) + row]);
            c[(col * (*ldc)) + row] = make_cdouble((double)(600 + (10 * row) + col));
        }
    }
    for (row = 0; row < *k; ++row) {
        g_zunmqr_fortran_call.tau_real_snapshot[row] = cdouble_real(tau[row]);
    }
    a[0] = make_cdouble(-323.0);
    tau[0] = make_cdouble(-656.0);

    *info = 0;
}

static int stub_zunmqr_cblas(const fb_layout_t layout, const fb_side_t side,
                             const fb_transpose_t trans, const int m,
                             const int n, const int k,
                             const fb_complex_double_t *a, const int lda,
                             const fb_complex_double_t *tau,
                             fb_complex_double_t *c, const int ldc)
{
    g_zunmqr_cblas_call.called += 1;
    g_zunmqr_cblas_call.layout = layout;
    g_zunmqr_cblas_call.side = side;
    g_zunmqr_cblas_call.trans = trans;
    g_zunmqr_cblas_call.m = m;
    g_zunmqr_cblas_call.n = n;
    g_zunmqr_cblas_call.k = k;
    g_zunmqr_cblas_call.lda = lda;
    g_zunmqr_cblas_call.ldc = ldc;
    g_zunmqr_cblas_call.a = a;
    g_zunmqr_cblas_call.tau = tau;
    g_zunmqr_cblas_call.c = c;
    return 74;
}

static int check_sormqr_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_sormqr_fn thunk = NULL;
    float h[6] = { 1.0f, 2.0f, 99.0f, 3.0f, 4.0f, 98.0f };
    float tau[2] = { 5.0f, 6.0f };
    float c[6] = { 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f };
    float expected_a_snapshot[4] = { 1.0f, 3.0f, 2.0f, 4.0f };
    float expected_c_snapshot[6] = { 7.0f, 10.0f, 8.0f, 11.0f, 9.0f, 12.0f };
    float expected_tau_in[2] = { 5.0f, 6.0f };
    float expected_c_out[6] = { 300.0f, 301.0f, 302.0f, 310.0f, 311.0f, 312.0f };
    float expected_h_unchanged[6] = { 1.0f, 2.0f, 99.0f, 3.0f, 4.0f, 98.0f };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sormqr_fortran_call, 0, sizeof(g_sormqr_fortran_call));

    vtable.ext_ops[FB_OP_SORMQR][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_sormqr_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SORMQR);

    thunk = (fb_sormqr_fn)vtable.ext_ops[FB_OP_SORMQR][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SORMQR Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_NO_TRANS, 2, 3, 2, h, 3, tau, c, 3);
    if (info != 0 || g_sormqr_fortran_call.query_calls != 1 ||
        g_sormqr_fortran_call.solve_calls != 1 ||
        g_sormqr_fortran_call.side != 'L' || g_sormqr_fortran_call.trans != 'N' ||
        g_sormqr_fortran_call.m != 2 || g_sormqr_fortran_call.n != 3 ||
        g_sormqr_fortran_call.k != 2 || g_sormqr_fortran_call.lda != 2 ||
        g_sormqr_fortran_call.ldc != 2 ||
        g_sormqr_fortran_call.lwork_query != -1 ||
        g_sormqr_fortran_call.lwork_solve != 7 ||
        memcmp(g_sormqr_fortran_call.a_snapshot, expected_a_snapshot,
               sizeof(expected_a_snapshot)) != 0 ||
        memcmp(g_sormqr_fortran_call.c_snapshot, expected_c_snapshot,
               sizeof(expected_c_snapshot)) != 0 ||
        memcmp(g_sormqr_fortran_call.tau_snapshot, expected_tau_in,
               sizeof(expected_tau_in)) != 0 ||
        memcmp(c, expected_c_out, sizeof(expected_c_out)) != 0 ||
        memcmp(h, expected_h_unchanged, sizeof(expected_h_unchanged)) != 0 ||
        memcmp(tau, expected_tau_in, sizeof(expected_tau_in)) != 0) {
        fprintf(stderr, "[FAIL] SORMQR Fortran->CBLAS thunk did not preserve row-major reflector-application semantics\n");
        return 1;
    }

    printf("[PASS] SORMQR Fortran->CBLAS thunk translates row-major reflector application and preserves A/TAU input across lwork query\n");
    return 0;
}

static int check_sormqr_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_sormqr_fortran_slot_fn thunk = NULL;
    char side = 'L';
    char trans = 'N';
    float h[4] = { 0.0f };
    float tau[2] = { 5.0f, 6.0f };
    float c[6] = { 0.0f };
    float work[4] = { 0.0f };
    int m = 2;
    int n = 3;
    int k = 2;
    int lda = 2;
    int ldc = 2;
    int lwork = 4;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sormqr_cblas_call, 0, sizeof(g_sormqr_cblas_call));

    vtable.ext_ops[FB_OP_SORMQR][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_sormqr_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SORMQR);

    thunk = (fb_sormqr_fortran_slot_fn)vtable.ext_ops[FB_OP_SORMQR][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SORMQR CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&side, &trans, &m, &n, &k, h, &lda, tau, c, &ldc, work, &lwork, &info);
    if (info != 71 || g_sormqr_cblas_call.called != 1 ||
        g_sormqr_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_sormqr_cblas_call.side != FB_LEFT ||
        g_sormqr_cblas_call.trans != FB_NO_TRANS ||
        g_sormqr_cblas_call.m != 2 || g_sormqr_cblas_call.n != 3 ||
        g_sormqr_cblas_call.k != 2 || g_sormqr_cblas_call.lda != 2 ||
        g_sormqr_cblas_call.ldc != 2 ||
        g_sormqr_cblas_call.a != h || g_sormqr_cblas_call.tau != tau ||
        g_sormqr_cblas_call.c != c) {
        fprintf(stderr, "[FAIL] SORMQR CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] SORMQR CBLAS->Fortran thunk maps side/trans chars into the C reflector-application entry\n");
    return 0;
}

static int check_dormqr_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_dormqr_fn thunk = NULL;
    double h[6] = { 1.0, 2.0, 99.0, 3.0, 4.0, 98.0 };
    double tau[2] = { 15.0, 16.0 };
    double c[6] = { 17.0, 18.0, 19.0, 20.0, 21.0, 22.0 };
    double expected_a_snapshot[4] = { 1.0, 3.0, 2.0, 4.0 };
    double expected_c_snapshot[6] = { 17.0, 20.0, 18.0, 21.0, 19.0, 22.0 };
    double expected_tau_in[2] = { 15.0, 16.0 };
    double expected_c_out[6] = { 500.0, 501.0, 502.0, 510.0, 511.0, 512.0 };
    double expected_h_unchanged[6] = { 1.0, 2.0, 99.0, 3.0, 4.0, 98.0 };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dormqr_fortran_call, 0, sizeof(g_dormqr_fortran_call));

    vtable.ext_ops[FB_OP_DORMQR][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_dormqr_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_DORMQR);

    thunk = (fb_dormqr_fn)vtable.ext_ops[FB_OP_DORMQR][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DORMQR Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_NO_TRANS, 2, 3, 2, h, 3, tau, c, 3);
    if (info != 0 || g_dormqr_fortran_call.query_calls != 1 ||
        g_dormqr_fortran_call.solve_calls != 1 ||
        g_dormqr_fortran_call.side != 'L' || g_dormqr_fortran_call.trans != 'N' ||
        g_dormqr_fortran_call.m != 2 || g_dormqr_fortran_call.n != 3 ||
        g_dormqr_fortran_call.k != 2 || g_dormqr_fortran_call.lda != 2 ||
        g_dormqr_fortran_call.ldc != 2 ||
        g_dormqr_fortran_call.lwork_query != -1 ||
        g_dormqr_fortran_call.lwork_solve != 17 ||
        memcmp(g_dormqr_fortran_call.a_snapshot, expected_a_snapshot,
               sizeof(expected_a_snapshot)) != 0 ||
        memcmp(g_dormqr_fortran_call.c_snapshot, expected_c_snapshot,
               sizeof(expected_c_snapshot)) != 0 ||
        memcmp(g_dormqr_fortran_call.tau_snapshot, expected_tau_in,
               sizeof(expected_tau_in)) != 0 ||
        memcmp(c, expected_c_out, sizeof(expected_c_out)) != 0 ||
        memcmp(h, expected_h_unchanged, sizeof(expected_h_unchanged)) != 0 ||
        memcmp(tau, expected_tau_in, sizeof(expected_tau_in)) != 0) {
        fprintf(stderr, "[FAIL] DORMQR Fortran->CBLAS thunk did not preserve double reflector-application semantics\n");
        return 1;
    }

    printf("[PASS] DORMQR Fortran->CBLAS thunk translates row-major reflector application and preserves double A/TAU input across lwork query\n");
    return 0;
}

static int check_dormqr_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_dormqr_fortran_slot_fn thunk = NULL;
    char side = 'L';
    char trans = 'N';
    double h[4] = { 0.0 };
    double tau[2] = { 15.0, 16.0 };
    double c[6] = { 0.0 };
    double work[4] = { 0.0 };
    int m = 2;
    int n = 3;
    int k = 2;
    int lda = 2;
    int ldc = 2;
    int lwork = 4;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dormqr_cblas_call, 0, sizeof(g_dormqr_cblas_call));

    vtable.ext_ops[FB_OP_DORMQR][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_dormqr_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_DORMQR);

    thunk = (fb_dormqr_fortran_slot_fn)vtable.ext_ops[FB_OP_DORMQR][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DORMQR CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&side, &trans, &m, &n, &k, h, &lda, tau, c, &ldc, work, &lwork, &info);
    if (info != 72 || g_dormqr_cblas_call.called != 1 ||
        g_dormqr_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_dormqr_cblas_call.side != FB_LEFT ||
        g_dormqr_cblas_call.trans != FB_NO_TRANS ||
        g_dormqr_cblas_call.m != 2 || g_dormqr_cblas_call.n != 3 ||
        g_dormqr_cblas_call.k != 2 || g_dormqr_cblas_call.lda != 2 ||
        g_dormqr_cblas_call.ldc != 2 ||
        g_dormqr_cblas_call.a != h || g_dormqr_cblas_call.tau != tau ||
        g_dormqr_cblas_call.c != c) {
        fprintf(stderr, "[FAIL] DORMQR CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] DORMQR CBLAS->Fortran thunk maps side/trans chars into the double C reflector-application entry\n");
    return 0;
}

static int check_cunmqr_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_cunmqr_fn thunk = NULL;
    fb_complex_float_t h[6] = {
        make_cfloat(1.0f), make_cfloat(2.0f),
        make_cfloat(3.0f), make_cfloat(4.0f),
        make_cfloat(5.0f), make_cfloat(6.0f)
    };
    fb_complex_float_t tau[2] = { make_cfloat(9.0f), make_cfloat(10.0f) };
    fb_complex_float_t c[6] = {
        make_cfloat(11.0f), make_cfloat(12.0f), make_cfloat(13.0f),
        make_cfloat(14.0f), make_cfloat(15.0f), make_cfloat(16.0f)
    };
    float expected_a_snapshot[6] = { 1.0f, 3.0f, 5.0f, 2.0f, 4.0f, 6.0f };
    float expected_c_snapshot[6] = { 11.0f, 14.0f, 12.0f, 15.0f, 13.0f, 16.0f };
    float expected_tau_in[2] = { 9.0f, 10.0f };
    float expected_c_out[6] = { 400.0f, 401.0f, 402.0f, 410.0f, 411.0f, 412.0f };
    int info = 0;
    int idx = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cunmqr_fortran_call, 0, sizeof(g_cunmqr_fortran_call));

    vtable.ext_ops[FB_OP_CUNMQR][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_cunmqr_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CUNMQR);

    thunk = (fb_cunmqr_fn)vtable.ext_ops[FB_OP_CUNMQR][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CUNMQR Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, FB_RIGHT, FB_CONJ_TRANS,
                 2, 3, 2, h, 2, tau, c, 3);
    if (info != 0 || g_cunmqr_fortran_call.query_calls != 1 ||
        g_cunmqr_fortran_call.solve_calls != 1 ||
        g_cunmqr_fortran_call.side != 'R' || g_cunmqr_fortran_call.trans != 'C' ||
        g_cunmqr_fortran_call.m != 2 || g_cunmqr_fortran_call.n != 3 ||
        g_cunmqr_fortran_call.k != 2 || g_cunmqr_fortran_call.lda != 3 ||
        g_cunmqr_fortran_call.ldc != 2 ||
        g_cunmqr_fortran_call.lwork_query != -1 ||
        g_cunmqr_fortran_call.lwork_solve != 8 ||
        memcmp(g_cunmqr_fortran_call.a_real_snapshot, expected_a_snapshot,
               sizeof(expected_a_snapshot)) != 0 ||
        memcmp(g_cunmqr_fortran_call.c_real_snapshot, expected_c_snapshot,
               sizeof(expected_c_snapshot)) != 0 ||
        memcmp(g_cunmqr_fortran_call.tau_real_snapshot, expected_tau_in,
               sizeof(expected_tau_in)) != 0) {
        fprintf(stderr, "[FAIL] CUNMQR Fortran->CBLAS thunk did not preserve complex reflector-application semantics\n");
        return 1;
    }

    for (idx = 0; idx < 6; ++idx) {
        if (cfloat_real(c[idx]) != expected_c_out[idx]) {
            fprintf(stderr, "[FAIL] CUNMQR Fortran->CBLAS thunk did not copy complex row-major output back correctly\n");
            return 1;
        }
    }
    for (idx = 0; idx < 2; ++idx) {
        if (cfloat_real(tau[idx]) != expected_tau_in[idx]) {
            fprintf(stderr, "[FAIL] CUNMQR Fortran->CBLAS thunk did not preserve complex TAU input\n");
            return 1;
        }
    }

    printf("[PASS] CUNMQR Fortran->CBLAS thunk translates row-major reflector application with side/trans preservation\n");
    return 0;
}

static int check_cunmqr_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_cunmqr_fortran_slot_fn thunk = NULL;
    char side = 'R';
    char trans = 'C';
    fb_complex_float_t h[6];
    fb_complex_float_t tau[2] = { make_cfloat(9.0f), make_cfloat(10.0f) };
    fb_complex_float_t c[6];
    fb_complex_float_t work[4];
    int m = 2;
    int n = 3;
    int k = 2;
    int lda = 3;
    int ldc = 2;
    int lwork = 4;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cunmqr_cblas_call, 0, sizeof(g_cunmqr_cblas_call));
    memset(h, 0, sizeof(h));
    memset(c, 0, sizeof(c));
    memset(work, 0, sizeof(work));

    vtable.ext_ops[FB_OP_CUNMQR][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_cunmqr_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CUNMQR);

    thunk = (fb_cunmqr_fortran_slot_fn)vtable.ext_ops[FB_OP_CUNMQR][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CUNMQR CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&side, &trans, &m, &n, &k, h, &lda, tau, c, &ldc, work, &lwork, &info);
    if (info != 73 || g_cunmqr_cblas_call.called != 1 ||
        g_cunmqr_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_cunmqr_cblas_call.side != FB_RIGHT ||
        g_cunmqr_cblas_call.trans != FB_CONJ_TRANS ||
        g_cunmqr_cblas_call.m != 2 || g_cunmqr_cblas_call.n != 3 ||
        g_cunmqr_cblas_call.k != 2 || g_cunmqr_cblas_call.lda != 3 ||
        g_cunmqr_cblas_call.ldc != 2 ||
        g_cunmqr_cblas_call.a != h || g_cunmqr_cblas_call.tau != tau ||
        g_cunmqr_cblas_call.c != c) {
        fprintf(stderr, "[FAIL] CUNMQR CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] CUNMQR CBLAS->Fortran thunk maps side/trans chars into the complex C reflector-application entry\n");
    return 0;
}

static int check_zunmqr_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_zunmqr_fn thunk = NULL;
    fb_complex_double_t h[6] = {
        make_cdouble(1.0), make_cdouble(2.0),
        make_cdouble(3.0), make_cdouble(4.0),
        make_cdouble(5.0), make_cdouble(6.0)
    };
    fb_complex_double_t tau[2] = { make_cdouble(19.0), make_cdouble(20.0) };
    fb_complex_double_t c[6] = {
        make_cdouble(21.0), make_cdouble(22.0), make_cdouble(23.0),
        make_cdouble(24.0), make_cdouble(25.0), make_cdouble(26.0)
    };
    double expected_a_snapshot[6] = { 1.0, 3.0, 5.0, 2.0, 4.0, 6.0 };
    double expected_c_snapshot[6] = { 21.0, 24.0, 22.0, 25.0, 23.0, 26.0 };
    double expected_tau_in[2] = { 19.0, 20.0 };
    double expected_c_out[6] = { 600.0, 601.0, 602.0, 610.0, 611.0, 612.0 };
    int info = 0;
    int idx = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zunmqr_fortran_call, 0, sizeof(g_zunmqr_fortran_call));

    vtable.ext_ops[FB_OP_ZUNMQR][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_zunmqr_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_ZUNMQR);

    thunk = (fb_zunmqr_fn)vtable.ext_ops[FB_OP_ZUNMQR][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZUNMQR Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, FB_RIGHT, FB_CONJ_TRANS,
                 2, 3, 2, h, 2, tau, c, 3);
    if (info != 0 || g_zunmqr_fortran_call.query_calls != 1 ||
        g_zunmqr_fortran_call.solve_calls != 1 ||
        g_zunmqr_fortran_call.side != 'R' || g_zunmqr_fortran_call.trans != 'C' ||
        g_zunmqr_fortran_call.m != 2 || g_zunmqr_fortran_call.n != 3 ||
        g_zunmqr_fortran_call.k != 2 || g_zunmqr_fortran_call.lda != 3 ||
        g_zunmqr_fortran_call.ldc != 2 ||
        g_zunmqr_fortran_call.lwork_query != -1 ||
        g_zunmqr_fortran_call.lwork_solve != 18 ||
        memcmp(g_zunmqr_fortran_call.a_real_snapshot, expected_a_snapshot,
               sizeof(expected_a_snapshot)) != 0 ||
        memcmp(g_zunmqr_fortran_call.c_real_snapshot, expected_c_snapshot,
               sizeof(expected_c_snapshot)) != 0 ||
        memcmp(g_zunmqr_fortran_call.tau_real_snapshot, expected_tau_in,
               sizeof(expected_tau_in)) != 0) {
        fprintf(stderr, "[FAIL] ZUNMQR Fortran->CBLAS thunk did not preserve complex-double reflector-application semantics\n");
        return 1;
    }

    for (idx = 0; idx < 6; ++idx) {
        if (cdouble_real(c[idx]) != expected_c_out[idx]) {
            fprintf(stderr, "[FAIL] ZUNMQR Fortran->CBLAS thunk did not copy complex-double row-major output back correctly\n");
            return 1;
        }
    }
    for (idx = 0; idx < 2; ++idx) {
        if (cdouble_real(tau[idx]) != expected_tau_in[idx]) {
            fprintf(stderr, "[FAIL] ZUNMQR Fortran->CBLAS thunk did not preserve complex-double TAU input\n");
            return 1;
        }
    }

    printf("[PASS] ZUNMQR Fortran->CBLAS thunk translates row-major reflector application with side/trans preservation\n");
    return 0;
}

static int check_zunmqr_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_zunmqr_fortran_slot_fn thunk = NULL;
    char side = 'R';
    char trans = 'C';
    fb_complex_double_t h[6];
    fb_complex_double_t tau[2] = { make_cdouble(19.0), make_cdouble(20.0) };
    fb_complex_double_t c[6];
    fb_complex_double_t work[4];
    int m = 2;
    int n = 3;
    int k = 2;
    int lda = 3;
    int ldc = 2;
    int lwork = 4;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zunmqr_cblas_call, 0, sizeof(g_zunmqr_cblas_call));
    memset(h, 0, sizeof(h));
    memset(c, 0, sizeof(c));
    memset(work, 0, sizeof(work));

    vtable.ext_ops[FB_OP_ZUNMQR][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_zunmqr_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_ZUNMQR);

    thunk = (fb_zunmqr_fortran_slot_fn)vtable.ext_ops[FB_OP_ZUNMQR][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZUNMQR CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&side, &trans, &m, &n, &k, h, &lda, tau, c, &ldc, work, &lwork, &info);
    if (info != 74 || g_zunmqr_cblas_call.called != 1 ||
        g_zunmqr_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_zunmqr_cblas_call.side != FB_RIGHT ||
        g_zunmqr_cblas_call.trans != FB_CONJ_TRANS ||
        g_zunmqr_cblas_call.m != 2 || g_zunmqr_cblas_call.n != 3 ||
        g_zunmqr_cblas_call.k != 2 || g_zunmqr_cblas_call.lda != 3 ||
        g_zunmqr_cblas_call.ldc != 2 ||
        g_zunmqr_cblas_call.a != h || g_zunmqr_cblas_call.tau != tau ||
        g_zunmqr_cblas_call.c != c) {
        fprintf(stderr, "[FAIL] ZUNMQR CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] ZUNMQR CBLAS->Fortran thunk maps side/trans chars into the complex-double C reflector-application entry\n");
    return 0;
}

int main(void)
{
    if (check_sormqr_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_sormqr_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_dormqr_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_dormqr_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_cunmqr_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_cunmqr_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_zunmqr_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_zunmqr_cblas_to_fortran() != 0) {
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}