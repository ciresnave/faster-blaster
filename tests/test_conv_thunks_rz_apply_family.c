#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef int (*fb_sormr3_fn)(fb_layout_t layout, fb_side_t side,
                            fb_transpose_t trans, int m, int n, int k, int l,
                            const float *a, int lda, const float *tau,
                            float *c, int ldc);
typedef int (*fb_dormr3_fn)(fb_layout_t layout, fb_side_t side,
                            fb_transpose_t trans, int m, int n, int k, int l,
                            const double *a, int lda, const double *tau,
                            double *c, int ldc);
typedef int (*fb_cunmr3_fn)(fb_layout_t layout, fb_side_t side,
                            fb_transpose_t trans, int m, int n, int k, int l,
                            const fb_complex_float_t *a, int lda,
                            const fb_complex_float_t *tau,
                            fb_complex_float_t *c, int ldc);
typedef int (*fb_zunmr3_fn)(fb_layout_t layout, fb_side_t side,
                            fb_transpose_t trans, int m, int n, int k, int l,
                            const fb_complex_double_t *a, int lda,
                            const fb_complex_double_t *tau,
                            fb_complex_double_t *c, int ldc);
typedef int (*fb_sormrz_fn)(fb_layout_t layout, fb_side_t side,
                            fb_transpose_t trans, int m, int n, int k, int l,
                            const float *a, int lda, const float *tau,
                            float *c, int ldc);
typedef int (*fb_dormrz_fn)(fb_layout_t layout, fb_side_t side,
                            fb_transpose_t trans, int m, int n, int k, int l,
                            const double *a, int lda, const double *tau,
                            double *c, int ldc);
typedef int (*fb_cunmrz_fn)(fb_layout_t layout, fb_side_t side,
                            fb_transpose_t trans, int m, int n, int k, int l,
                            const fb_complex_float_t *a, int lda,
                            const fb_complex_float_t *tau,
                            fb_complex_float_t *c, int ldc);
typedef int (*fb_zunmrz_fn)(fb_layout_t layout, fb_side_t side,
                            fb_transpose_t trans, int m, int n, int k, int l,
                            const fb_complex_double_t *a, int lda,
                            const fb_complex_double_t *tau,
                            fb_complex_double_t *c, int ldc);

typedef void (*fb_sormr3_fortran_slot_fn)(char *side, char *trans, int *m,
                                          int *n, int *k, int *l, float *a,
                                          int *lda, float *tau, float *c,
                                          int *ldc, float *work, int *info);
typedef void (*fb_dormr3_fortran_slot_fn)(char *side, char *trans, int *m,
                                          int *n, int *k, int *l, double *a,
                                          int *lda, double *tau, double *c,
                                          int *ldc, double *work, int *info);
typedef void (*fb_cunmr3_fortran_slot_fn)(char *side, char *trans, int *m,
                                          int *n, int *k, int *l,
                                          fb_complex_float_t *a, int *lda,
                                          fb_complex_float_t *tau,
                                          fb_complex_float_t *c, int *ldc,
                                          fb_complex_float_t *work, int *info);
typedef void (*fb_zunmr3_fortran_slot_fn)(char *side, char *trans, int *m,
                                          int *n, int *k, int *l,
                                          fb_complex_double_t *a, int *lda,
                                          fb_complex_double_t *tau,
                                          fb_complex_double_t *c, int *ldc,
                                          fb_complex_double_t *work,
                                          int *info);
typedef void (*fb_sormrz_fortran_slot_fn)(char *side, char *trans, int *m,
                                          int *n, int *k, int *l, float *a,
                                          int *lda, float *tau, float *c,
                                          int *ldc, float *work, int *lwork,
                                          int *info);
typedef void (*fb_dormrz_fortran_slot_fn)(char *side, char *trans, int *m,
                                          int *n, int *k, int *l, double *a,
                                          int *lda, double *tau, double *c,
                                          int *ldc, double *work,
                                          int *lwork, int *info);
typedef void (*fb_cunmrz_fortran_slot_fn)(char *side, char *trans, int *m,
                                          int *n, int *k, int *l,
                                          fb_complex_float_t *a, int *lda,
                                          fb_complex_float_t *tau,
                                          fb_complex_float_t *c, int *ldc,
                                          fb_complex_float_t *work,
                                          int *lwork, int *info);
typedef void (*fb_zunmrz_fortran_slot_fn)(char *side, char *trans, int *m,
                                          int *n, int *k, int *l,
                                          fb_complex_double_t *a, int *lda,
                                          fb_complex_double_t *tau,
                                          fb_complex_double_t *c, int *ldc,
                                          fb_complex_double_t *work,
                                          int *lwork, int *info);

static struct {
    int calls;
    char side;
    char trans;
    int m;
    int n;
    int k;
    int l;
    int lda;
    int ldc;
    float a_snapshot[8];
    float tau_snapshot[2];
    float c_snapshot[8];
} g_sormr3_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    fb_side_t side;
    fb_transpose_t trans;
    int m;
    int n;
    int k;
    int l;
    int lda;
    int ldc;
    const float *a;
    const float *tau;
    float *c;
} g_sormr3_cblas_call;

static struct {
    int calls;
    char side;
    char trans;
    int m;
    int n;
    int k;
    int l;
    int lda;
    int ldc;
    double a_snapshot[8];
    double tau_snapshot[2];
    double c_snapshot[8];
} g_dormr3_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    fb_side_t side;
    fb_transpose_t trans;
    int m;
    int n;
    int k;
    int l;
    int lda;
    int ldc;
    const double *a;
    const double *tau;
    double *c;
} g_dormr3_cblas_call;

static struct {
    int calls;
    char side;
    char trans;
    int m;
    int n;
    int k;
    int l;
    int lda;
    int ldc;
    float a_real_snapshot[8];
    float tau_real_snapshot[2];
    float c_real_snapshot[8];
} g_cunmr3_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    fb_side_t side;
    fb_transpose_t trans;
    int m;
    int n;
    int k;
    int l;
    int lda;
    int ldc;
    const fb_complex_float_t *a;
    const fb_complex_float_t *tau;
    fb_complex_float_t *c;
} g_cunmr3_cblas_call;

static struct {
    int calls;
    char side;
    char trans;
    int m;
    int n;
    int k;
    int l;
    int lda;
    int ldc;
    double a_real_snapshot[8];
    double tau_real_snapshot[2];
    double c_real_snapshot[8];
} g_zunmr3_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    fb_side_t side;
    fb_transpose_t trans;
    int m;
    int n;
    int k;
    int l;
    int lda;
    int ldc;
    const fb_complex_double_t *a;
    const fb_complex_double_t *tau;
    fb_complex_double_t *c;
} g_zunmr3_cblas_call;

static struct {
    int calls;
    int query_calls;
    int exec_calls;
    char side;
    char trans;
    int m;
    int n;
    int k;
    int l;
    int lda;
    int ldc;
    int query_lwork;
    int exec_lwork;
    float a_snapshot[8];
    float tau_snapshot[2];
    float c_snapshot[8];
} g_sormrz_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    fb_side_t side;
    fb_transpose_t trans;
    int m;
    int n;
    int k;
    int l;
    int lda;
    int ldc;
    const float *a;
    const float *tau;
    float *c;
} g_sormrz_cblas_call;

static struct {
    int calls;
    int query_calls;
    int exec_calls;
    char side;
    char trans;
    int m;
    int n;
    int k;
    int l;
    int lda;
    int ldc;
    int query_lwork;
    int exec_lwork;
    double a_snapshot[8];
    double tau_snapshot[2];
    double c_snapshot[8];
} g_dormrz_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    fb_side_t side;
    fb_transpose_t trans;
    int m;
    int n;
    int k;
    int l;
    int lda;
    int ldc;
    const double *a;
    const double *tau;
    double *c;
} g_dormrz_cblas_call;

static struct {
    int calls;
    int query_calls;
    int exec_calls;
    char side;
    char trans;
    int m;
    int n;
    int k;
    int l;
    int lda;
    int ldc;
    int query_lwork;
    int exec_lwork;
    float a_real_snapshot[8];
    float tau_real_snapshot[2];
    float c_real_snapshot[8];
} g_cunmrz_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    fb_side_t side;
    fb_transpose_t trans;
    int m;
    int n;
    int k;
    int l;
    int lda;
    int ldc;
    const fb_complex_float_t *a;
    const fb_complex_float_t *tau;
    fb_complex_float_t *c;
} g_cunmrz_cblas_call;

static struct {
    int calls;
    int query_calls;
    int exec_calls;
    char side;
    char trans;
    int m;
    int n;
    int k;
    int l;
    int lda;
    int ldc;
    int query_lwork;
    int exec_lwork;
    double a_real_snapshot[8];
    double tau_real_snapshot[2];
    double c_real_snapshot[8];
} g_zunmrz_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    fb_side_t side;
    fb_transpose_t trans;
    int m;
    int n;
    int k;
    int l;
    int lda;
    int ldc;
    const fb_complex_double_t *a;
    const fb_complex_double_t *tau;
    fb_complex_double_t *c;
} g_zunmrz_cblas_call;

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

static void stub_sormr3_fortran(char *side, char *trans, int *m, int *n, int *k,
                                int *l, float *a, int *lda, float *tau,
                                float *c, int *ldc, float *work, int *info)
{
    int row = 0;
    int col = 0;

    g_sormr3_fortran_call.calls += 1;
    g_sormr3_fortran_call.side = *side;
    g_sormr3_fortran_call.trans = *trans;
    g_sormr3_fortran_call.m = *m;
    g_sormr3_fortran_call.n = *n;
    g_sormr3_fortran_call.k = *k;
    g_sormr3_fortran_call.l = *l;
    g_sormr3_fortran_call.lda = *lda;
    g_sormr3_fortran_call.ldc = *ldc;
    for (col = 0; col < *m; ++col) {
        for (row = 0; row < *k; ++row) {
            g_sormr3_fortran_call.a_snapshot[(col * (*k)) + row] =
                a[(col * (*lda)) + row];
        }
    }
    for (row = 0; row < *k; ++row) {
        g_sormr3_fortran_call.tau_snapshot[row] = tau[row];
    }
    for (col = 0; col < *n; ++col) {
        for (row = 0; row < *m; ++row) {
            g_sormr3_fortran_call.c_snapshot[(col * (*m)) + row] =
                c[(col * (*ldc)) + row];
            c[(col * (*ldc)) + row] = (float)(2100 + (10 * row) + col);
        }
    }
    if (work) {
        work[0] = 15.0f;
    }
    *info = 0;
}

static int stub_sormr3_cblas(fb_layout_t layout, fb_side_t side,
                             fb_transpose_t trans, int m, int n, int k, int l,
                             const float *a, int lda, const float *tau,
                             float *c, int ldc)
{
    g_sormr3_cblas_call.called += 1;
    g_sormr3_cblas_call.layout = layout;
    g_sormr3_cblas_call.side = side;
    g_sormr3_cblas_call.trans = trans;
    g_sormr3_cblas_call.m = m;
    g_sormr3_cblas_call.n = n;
    g_sormr3_cblas_call.k = k;
    g_sormr3_cblas_call.l = l;
    g_sormr3_cblas_call.lda = lda;
    g_sormr3_cblas_call.ldc = ldc;
    g_sormr3_cblas_call.a = a;
    g_sormr3_cblas_call.tau = tau;
    g_sormr3_cblas_call.c = c;
    return 121;
}

static void stub_dormr3_fortran(char *side, char *trans, int *m, int *n, int *k,
                                int *l, double *a, int *lda, double *tau,
                                double *c, int *ldc, double *work, int *info)
{
    int row = 0;
    int col = 0;

    g_dormr3_fortran_call.calls += 1;
    g_dormr3_fortran_call.side = *side;
    g_dormr3_fortran_call.trans = *trans;
    g_dormr3_fortran_call.m = *m;
    g_dormr3_fortran_call.n = *n;
    g_dormr3_fortran_call.k = *k;
    g_dormr3_fortran_call.l = *l;
    g_dormr3_fortran_call.lda = *lda;
    g_dormr3_fortran_call.ldc = *ldc;
    for (col = 0; col < *m; ++col) {
        for (row = 0; row < *k; ++row) {
            g_dormr3_fortran_call.a_snapshot[(col * (*k)) + row] =
                a[(col * (*lda)) + row];
        }
    }
    for (row = 0; row < *k; ++row) {
        g_dormr3_fortran_call.tau_snapshot[row] = tau[row];
    }
    for (col = 0; col < *n; ++col) {
        for (row = 0; row < *m; ++row) {
            g_dormr3_fortran_call.c_snapshot[(col * (*m)) + row] =
                c[(col * (*ldc)) + row];
            c[(col * (*ldc)) + row] = (double)(2150 + (10 * row) + col);
        }
    }
    if (work) {
        work[0] = 15.5;
    }
    *info = 0;
}

static int stub_dormr3_cblas(fb_layout_t layout, fb_side_t side,
                             fb_transpose_t trans, int m, int n, int k, int l,
                             const double *a, int lda, const double *tau,
                             double *c, int ldc)
{
    g_dormr3_cblas_call.called += 1;
    g_dormr3_cblas_call.layout = layout;
    g_dormr3_cblas_call.side = side;
    g_dormr3_cblas_call.trans = trans;
    g_dormr3_cblas_call.m = m;
    g_dormr3_cblas_call.n = n;
    g_dormr3_cblas_call.k = k;
    g_dormr3_cblas_call.l = l;
    g_dormr3_cblas_call.lda = lda;
    g_dormr3_cblas_call.ldc = ldc;
    g_dormr3_cblas_call.a = a;
    g_dormr3_cblas_call.tau = tau;
    g_dormr3_cblas_call.c = c;
    return 122;
}

static void stub_cunmr3_fortran(char *side, char *trans, int *m, int *n, int *k,
                                int *l, fb_complex_float_t *a, int *lda,
                                fb_complex_float_t *tau,
                                fb_complex_float_t *c, int *ldc,
                                fb_complex_float_t *work, int *info)
{
    int row = 0;
    int col = 0;

    g_cunmr3_fortran_call.calls += 1;
    g_cunmr3_fortran_call.side = *side;
    g_cunmr3_fortran_call.trans = *trans;
    g_cunmr3_fortran_call.m = *m;
    g_cunmr3_fortran_call.n = *n;
    g_cunmr3_fortran_call.k = *k;
    g_cunmr3_fortran_call.l = *l;
    g_cunmr3_fortran_call.lda = *lda;
    g_cunmr3_fortran_call.ldc = *ldc;
    for (col = 0; col < *n; ++col) {
        for (row = 0; row < *k; ++row) {
            g_cunmr3_fortran_call.a_real_snapshot[(col * (*k)) + row] =
                cfloat_real(a[(col * (*lda)) + row]);
        }
    }
    for (row = 0; row < *k; ++row) {
        g_cunmr3_fortran_call.tau_real_snapshot[row] = cfloat_real(tau[row]);
    }
    for (col = 0; col < *n; ++col) {
        for (row = 0; row < *m; ++row) {
            g_cunmr3_fortran_call.c_real_snapshot[(col * (*m)) + row] =
                cfloat_real(c[(col * (*ldc)) + row]);
            c[(col * (*ldc)) + row] = make_cfloat((float)(2200 + (10 * row) + col));
        }
    }
    if (work) {
        work[0] = make_cfloat(16.0f);
    }
    *info = 0;
}

static int stub_cunmr3_cblas(fb_layout_t layout, fb_side_t side,
                             fb_transpose_t trans, int m, int n, int k, int l,
                             const fb_complex_float_t *a, int lda,
                             const fb_complex_float_t *tau,
                             fb_complex_float_t *c, int ldc)
{
    g_cunmr3_cblas_call.called += 1;
    g_cunmr3_cblas_call.layout = layout;
    g_cunmr3_cblas_call.side = side;
    g_cunmr3_cblas_call.trans = trans;
    g_cunmr3_cblas_call.m = m;
    g_cunmr3_cblas_call.n = n;
    g_cunmr3_cblas_call.k = k;
    g_cunmr3_cblas_call.l = l;
    g_cunmr3_cblas_call.lda = lda;
    g_cunmr3_cblas_call.ldc = ldc;
    g_cunmr3_cblas_call.a = a;
    g_cunmr3_cblas_call.tau = tau;
    g_cunmr3_cblas_call.c = c;
    return 123;
}

static void stub_zunmr3_fortran(char *side, char *trans, int *m, int *n, int *k,
                                int *l, fb_complex_double_t *a, int *lda,
                                fb_complex_double_t *tau,
                                fb_complex_double_t *c, int *ldc,
                                fb_complex_double_t *work, int *info)
{
    int row = 0;
    int col = 0;

    g_zunmr3_fortran_call.calls += 1;
    g_zunmr3_fortran_call.side = *side;
    g_zunmr3_fortran_call.trans = *trans;
    g_zunmr3_fortran_call.m = *m;
    g_zunmr3_fortran_call.n = *n;
    g_zunmr3_fortran_call.k = *k;
    g_zunmr3_fortran_call.l = *l;
    g_zunmr3_fortran_call.lda = *lda;
    g_zunmr3_fortran_call.ldc = *ldc;
    for (col = 0; col < *n; ++col) {
        for (row = 0; row < *k; ++row) {
            g_zunmr3_fortran_call.a_real_snapshot[(col * (*k)) + row] =
                cdouble_real(a[(col * (*lda)) + row]);
        }
    }
    for (row = 0; row < *k; ++row) {
        g_zunmr3_fortran_call.tau_real_snapshot[row] = cdouble_real(tau[row]);
    }
    for (col = 0; col < *n; ++col) {
        for (row = 0; row < *m; ++row) {
            g_zunmr3_fortran_call.c_real_snapshot[(col * (*m)) + row] =
                cdouble_real(c[(col * (*ldc)) + row]);
            c[(col * (*ldc)) + row] = make_cdouble((double)(2250 + (10 * row) + col));
        }
    }
    if (work) {
        work[0] = make_cdouble(16.5);
    }
    *info = 0;
}

static int stub_zunmr3_cblas(fb_layout_t layout, fb_side_t side,
                             fb_transpose_t trans, int m, int n, int k, int l,
                             const fb_complex_double_t *a, int lda,
                             const fb_complex_double_t *tau,
                             fb_complex_double_t *c, int ldc)
{
    g_zunmr3_cblas_call.called += 1;
    g_zunmr3_cblas_call.layout = layout;
    g_zunmr3_cblas_call.side = side;
    g_zunmr3_cblas_call.trans = trans;
    g_zunmr3_cblas_call.m = m;
    g_zunmr3_cblas_call.n = n;
    g_zunmr3_cblas_call.k = k;
    g_zunmr3_cblas_call.l = l;
    g_zunmr3_cblas_call.lda = lda;
    g_zunmr3_cblas_call.ldc = ldc;
    g_zunmr3_cblas_call.a = a;
    g_zunmr3_cblas_call.tau = tau;
    g_zunmr3_cblas_call.c = c;
    return 124;
}

static void stub_sormrz_fortran(char *side, char *trans, int *m, int *n, int *k,
                                int *l, float *a, int *lda, float *tau,
                                float *c, int *ldc, float *work, int *lwork,
                                int *info)
{
    int row = 0;
    int col = 0;

    g_sormrz_fortran_call.calls += 1;
    g_sormrz_fortran_call.side = *side;
    g_sormrz_fortran_call.trans = *trans;
    g_sormrz_fortran_call.m = *m;
    g_sormrz_fortran_call.n = *n;
    g_sormrz_fortran_call.k = *k;
    g_sormrz_fortran_call.l = *l;
    g_sormrz_fortran_call.lda = *lda;
    g_sormrz_fortran_call.ldc = *ldc;
    if (*lwork == -1) {
        g_sormrz_fortran_call.query_calls += 1;
        g_sormrz_fortran_call.query_lwork = *lwork;
        if (work) {
            work[0] = 17.0f;
        }
        *info = 0;
        return;
    }

    g_sormrz_fortran_call.exec_calls += 1;
    g_sormrz_fortran_call.exec_lwork = *lwork;
    for (col = 0; col < *m; ++col) {
        for (row = 0; row < *k; ++row) {
            g_sormrz_fortran_call.a_snapshot[(col * (*k)) + row] =
                a[(col * (*lda)) + row];
        }
    }
    for (row = 0; row < *k; ++row) {
        g_sormrz_fortran_call.tau_snapshot[row] = tau[row];
    }
    for (col = 0; col < *n; ++col) {
        for (row = 0; row < *m; ++row) {
            g_sormrz_fortran_call.c_snapshot[(col * (*m)) + row] =
                c[(col * (*ldc)) + row];
            c[(col * (*ldc)) + row] = (float)(2300 + (10 * row) + col);
        }
    }
    *info = 0;
}

static int stub_sormrz_cblas(fb_layout_t layout, fb_side_t side,
                             fb_transpose_t trans, int m, int n, int k, int l,
                             const float *a, int lda, const float *tau,
                             float *c, int ldc)
{
    g_sormrz_cblas_call.called += 1;
    g_sormrz_cblas_call.layout = layout;
    g_sormrz_cblas_call.side = side;
    g_sormrz_cblas_call.trans = trans;
    g_sormrz_cblas_call.m = m;
    g_sormrz_cblas_call.n = n;
    g_sormrz_cblas_call.k = k;
    g_sormrz_cblas_call.l = l;
    g_sormrz_cblas_call.lda = lda;
    g_sormrz_cblas_call.ldc = ldc;
    g_sormrz_cblas_call.a = a;
    g_sormrz_cblas_call.tau = tau;
    g_sormrz_cblas_call.c = c;
    return 125;
}

static void stub_dormrz_fortran(char *side, char *trans, int *m, int *n, int *k,
                                int *l, double *a, int *lda, double *tau,
                                double *c, int *ldc, double *work, int *lwork,
                                int *info)
{
    int row = 0;
    int col = 0;

    g_dormrz_fortran_call.calls += 1;
    g_dormrz_fortran_call.side = *side;
    g_dormrz_fortran_call.trans = *trans;
    g_dormrz_fortran_call.m = *m;
    g_dormrz_fortran_call.n = *n;
    g_dormrz_fortran_call.k = *k;
    g_dormrz_fortran_call.l = *l;
    g_dormrz_fortran_call.lda = *lda;
    g_dormrz_fortran_call.ldc = *ldc;
    if (*lwork == -1) {
        g_dormrz_fortran_call.query_calls += 1;
        g_dormrz_fortran_call.query_lwork = *lwork;
        if (work) {
            work[0] = 19.0;
        }
        *info = 0;
        return;
    }

    g_dormrz_fortran_call.exec_calls += 1;
    g_dormrz_fortran_call.exec_lwork = *lwork;
    for (col = 0; col < *m; ++col) {
        for (row = 0; row < *k; ++row) {
            g_dormrz_fortran_call.a_snapshot[(col * (*k)) + row] =
                a[(col * (*lda)) + row];
        }
    }
    for (row = 0; row < *k; ++row) {
        g_dormrz_fortran_call.tau_snapshot[row] = tau[row];
    }
    for (col = 0; col < *n; ++col) {
        for (row = 0; row < *m; ++row) {
            g_dormrz_fortran_call.c_snapshot[(col * (*m)) + row] =
                c[(col * (*ldc)) + row];
            c[(col * (*ldc)) + row] = (double)(2350 + (10 * row) + col);
        }
    }
    *info = 0;
}

static int stub_dormrz_cblas(fb_layout_t layout, fb_side_t side,
                             fb_transpose_t trans, int m, int n, int k, int l,
                             const double *a, int lda, const double *tau,
                             double *c, int ldc)
{
    g_dormrz_cblas_call.called += 1;
    g_dormrz_cblas_call.layout = layout;
    g_dormrz_cblas_call.side = side;
    g_dormrz_cblas_call.trans = trans;
    g_dormrz_cblas_call.m = m;
    g_dormrz_cblas_call.n = n;
    g_dormrz_cblas_call.k = k;
    g_dormrz_cblas_call.l = l;
    g_dormrz_cblas_call.lda = lda;
    g_dormrz_cblas_call.ldc = ldc;
    g_dormrz_cblas_call.a = a;
    g_dormrz_cblas_call.tau = tau;
    g_dormrz_cblas_call.c = c;
    return 126;
}

static void stub_cunmrz_fortran(char *side, char *trans, int *m, int *n, int *k,
                                int *l, fb_complex_float_t *a, int *lda,
                                fb_complex_float_t *tau,
                                fb_complex_float_t *c, int *ldc,
                                fb_complex_float_t *work, int *lwork,
                                int *info)
{
    int row = 0;
    int col = 0;

    g_cunmrz_fortran_call.calls += 1;
    g_cunmrz_fortran_call.side = *side;
    g_cunmrz_fortran_call.trans = *trans;
    g_cunmrz_fortran_call.m = *m;
    g_cunmrz_fortran_call.n = *n;
    g_cunmrz_fortran_call.k = *k;
    g_cunmrz_fortran_call.l = *l;
    g_cunmrz_fortran_call.lda = *lda;
    g_cunmrz_fortran_call.ldc = *ldc;
    if (*lwork == -1) {
        g_cunmrz_fortran_call.query_calls += 1;
        g_cunmrz_fortran_call.query_lwork = *lwork;
        if (work) {
            work[0] = make_cfloat(18.0f);
        }
        *info = 0;
        return;
    }

    g_cunmrz_fortran_call.exec_calls += 1;
    g_cunmrz_fortran_call.exec_lwork = *lwork;
    for (col = 0; col < *n; ++col) {
        for (row = 0; row < *k; ++row) {
            g_cunmrz_fortran_call.a_real_snapshot[(col * (*k)) + row] =
                cfloat_real(a[(col * (*lda)) + row]);
        }
    }
    for (row = 0; row < *k; ++row) {
        g_cunmrz_fortran_call.tau_real_snapshot[row] = cfloat_real(tau[row]);
    }
    for (col = 0; col < *n; ++col) {
        for (row = 0; row < *m; ++row) {
            g_cunmrz_fortran_call.c_real_snapshot[(col * (*m)) + row] =
                cfloat_real(c[(col * (*ldc)) + row]);
            c[(col * (*ldc)) + row] = make_cfloat((float)(2400 + (10 * row) + col));
        }
    }
    *info = 0;
}

static int stub_cunmrz_cblas(fb_layout_t layout, fb_side_t side,
                             fb_transpose_t trans, int m, int n, int k, int l,
                             const fb_complex_float_t *a, int lda,
                             const fb_complex_float_t *tau,
                             fb_complex_float_t *c, int ldc)
{
    g_cunmrz_cblas_call.called += 1;
    g_cunmrz_cblas_call.layout = layout;
    g_cunmrz_cblas_call.side = side;
    g_cunmrz_cblas_call.trans = trans;
    g_cunmrz_cblas_call.m = m;
    g_cunmrz_cblas_call.n = n;
    g_cunmrz_cblas_call.k = k;
    g_cunmrz_cblas_call.l = l;
    g_cunmrz_cblas_call.lda = lda;
    g_cunmrz_cblas_call.ldc = ldc;
    g_cunmrz_cblas_call.a = a;
    g_cunmrz_cblas_call.tau = tau;
    g_cunmrz_cblas_call.c = c;
    return 127;
}

static void stub_zunmrz_fortran(char *side, char *trans, int *m, int *n, int *k,
                                int *l, fb_complex_double_t *a, int *lda,
                                fb_complex_double_t *tau,
                                fb_complex_double_t *c, int *ldc,
                                fb_complex_double_t *work, int *lwork,
                                int *info)
{
    int row = 0;
    int col = 0;

    g_zunmrz_fortran_call.calls += 1;
    g_zunmrz_fortran_call.side = *side;
    g_zunmrz_fortran_call.trans = *trans;
    g_zunmrz_fortran_call.m = *m;
    g_zunmrz_fortran_call.n = *n;
    g_zunmrz_fortran_call.k = *k;
    g_zunmrz_fortran_call.l = *l;
    g_zunmrz_fortran_call.lda = *lda;
    g_zunmrz_fortran_call.ldc = *ldc;
    if (*lwork == -1) {
        g_zunmrz_fortran_call.query_calls += 1;
        g_zunmrz_fortran_call.query_lwork = *lwork;
        if (work) {
            work[0] = make_cdouble(20.0);
        }
        *info = 0;
        return;
    }

    g_zunmrz_fortran_call.exec_calls += 1;
    g_zunmrz_fortran_call.exec_lwork = *lwork;
    for (col = 0; col < *n; ++col) {
        for (row = 0; row < *k; ++row) {
            g_zunmrz_fortran_call.a_real_snapshot[(col * (*k)) + row] =
                cdouble_real(a[(col * (*lda)) + row]);
        }
    }
    for (row = 0; row < *k; ++row) {
        g_zunmrz_fortran_call.tau_real_snapshot[row] = cdouble_real(tau[row]);
    }
    for (col = 0; col < *n; ++col) {
        for (row = 0; row < *m; ++row) {
            g_zunmrz_fortran_call.c_real_snapshot[(col * (*m)) + row] =
                cdouble_real(c[(col * (*ldc)) + row]);
            c[(col * (*ldc)) + row] = make_cdouble((double)(2450 + (10 * row) + col));
        }
    }
    *info = 0;
}

static int stub_zunmrz_cblas(fb_layout_t layout, fb_side_t side,
                             fb_transpose_t trans, int m, int n, int k, int l,
                             const fb_complex_double_t *a, int lda,
                             const fb_complex_double_t *tau,
                             fb_complex_double_t *c, int ldc)
{
    g_zunmrz_cblas_call.called += 1;
    g_zunmrz_cblas_call.layout = layout;
    g_zunmrz_cblas_call.side = side;
    g_zunmrz_cblas_call.trans = trans;
    g_zunmrz_cblas_call.m = m;
    g_zunmrz_cblas_call.n = n;
    g_zunmrz_cblas_call.k = k;
    g_zunmrz_cblas_call.l = l;
    g_zunmrz_cblas_call.lda = lda;
    g_zunmrz_cblas_call.ldc = ldc;
    g_zunmrz_cblas_call.a = a;
    g_zunmrz_cblas_call.tau = tau;
    g_zunmrz_cblas_call.c = c;
    return 128;
}

static int check_dormr3_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_dormr3_fn thunk = NULL;
    double a[8] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    double tau[2] = { 96.0, 97.0 };
    double c[8] = { 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0 };
    double expected_a_snapshot[8] = { 1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0 };
    double expected_tau[2] = { 96.0, 97.0 };
    double expected_c_snapshot[8] = { 10.0, 12.0, 14.0, 16.0, 11.0, 13.0, 15.0, 17.0 };
    double expected_c_out[8] = { 2150.0, 2151.0, 2160.0, 2161.0, 2170.0, 2171.0, 2180.0, 2181.0 };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dormr3_fortran_call, 0, sizeof(g_dormr3_fortran_call));

    vtable.ext_ops[FB_OP_DORMR3][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_dormr3_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_DORMR3);

    thunk = (fb_dormr3_fn)vtable.ext_ops[FB_OP_DORMR3][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DORMR3 Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_NO_TRANS, 4, 2, 2, 1, a, 4, tau, c, 2);
    if (info != 0 || g_dormr3_fortran_call.calls != 1 ||
        g_dormr3_fortran_call.side != 'L' || g_dormr3_fortran_call.trans != 'N' ||
        g_dormr3_fortran_call.m != 4 || g_dormr3_fortran_call.n != 2 ||
        g_dormr3_fortran_call.k != 2 || g_dormr3_fortran_call.l != 1 ||
        g_dormr3_fortran_call.lda != 2 || g_dormr3_fortran_call.ldc != 4 ||
        memcmp(g_dormr3_fortran_call.a_snapshot, expected_a_snapshot,
               sizeof(expected_a_snapshot)) != 0 ||
        memcmp(g_dormr3_fortran_call.tau_snapshot, expected_tau,
               sizeof(expected_tau)) != 0 ||
        memcmp(g_dormr3_fortran_call.c_snapshot, expected_c_snapshot,
               sizeof(expected_c_snapshot)) != 0 ||
        memcmp(c, expected_c_out, sizeof(expected_c_out)) != 0) {
        fprintf(stderr, "[FAIL] DORMR3 Fortran->CBLAS thunk did not preserve double row-major RZ-apply semantics\n");
        return 1;
    }

    printf("[PASS] DORMR3 Fortran->CBLAS thunk translates row-major double RZ-apply matrices and preserves side/trans/l metadata\n");
    return 0;
}

static int check_dormr3_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_dormr3_fortran_slot_fn thunk = NULL;
    double a[8] = { 0.0 };
    double tau[2] = { 96.0, 97.0 };
    double c[8] = { 0.0 };
    double work[2] = { 0.0 };
    char side = 'L';
    char trans = 'N';
    int m = 4;
    int n = 2;
    int k = 2;
    int l = 1;
    int lda = 2;
    int ldc = 4;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dormr3_cblas_call, 0, sizeof(g_dormr3_cblas_call));

    vtable.ext_ops[FB_OP_DORMR3][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_dormr3_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_DORMR3);

    thunk = (fb_dormr3_fortran_slot_fn)vtable.ext_ops[FB_OP_DORMR3][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DORMR3 CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&side, &trans, &m, &n, &k, &l, a, &lda, tau, c, &ldc, work, &info);
    if (info != 122 || g_dormr3_cblas_call.called != 1 ||
        g_dormr3_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_dormr3_cblas_call.side != FB_LEFT ||
        g_dormr3_cblas_call.trans != FB_NO_TRANS ||
        g_dormr3_cblas_call.m != 4 || g_dormr3_cblas_call.n != 2 ||
        g_dormr3_cblas_call.k != 2 || g_dormr3_cblas_call.l != 1 ||
        g_dormr3_cblas_call.lda != 2 || g_dormr3_cblas_call.ldc != 4 ||
        g_dormr3_cblas_call.a != a || g_dormr3_cblas_call.tau != tau ||
        g_dormr3_cblas_call.c != c) {
        fprintf(stderr, "[FAIL] DORMR3 CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] DORMR3 CBLAS->Fortran thunk maps the all-pointer ABI into the generic double C RZ-apply entry\n");
    return 0;
}

static int check_zunmr3_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_zunmr3_fn thunk = NULL;
    fb_complex_double_t a[8] = {
        make_cdouble(1.0), make_cdouble(2.0), make_cdouble(3.0), make_cdouble(4.0),
        make_cdouble(5.0), make_cdouble(6.0), make_cdouble(7.0), make_cdouble(8.0)
    };
    fb_complex_double_t tau[2] = { make_cdouble(98.0), make_cdouble(99.0) };
    fb_complex_double_t c[8] = {
        make_cdouble(20.0), make_cdouble(21.0), make_cdouble(22.0), make_cdouble(23.0),
        make_cdouble(24.0), make_cdouble(25.0), make_cdouble(26.0), make_cdouble(27.0)
    };
    double expected_a_snapshot[8] = { 1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0 };
    double expected_tau[2] = { 98.0, 99.0 };
    double expected_c_snapshot[8] = { 20.0, 24.0, 21.0, 25.0, 22.0, 26.0, 23.0, 27.0 };
    double expected_c_out[8] = { 2250.0, 2251.0, 2252.0, 2253.0, 2260.0, 2261.0, 2262.0, 2263.0 };
    int info = 0;
    int idx = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zunmr3_fortran_call, 0, sizeof(g_zunmr3_fortran_call));

    vtable.ext_ops[FB_OP_ZUNMR3][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_zunmr3_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_ZUNMR3);

    thunk = (fb_zunmr3_fn)vtable.ext_ops[FB_OP_ZUNMR3][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZUNMR3 Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, FB_RIGHT, FB_CONJ_TRANS, 2, 4, 2, 1,
                 a, 4, tau, c, 4);
    if (info != 0 || g_zunmr3_fortran_call.calls != 1 ||
        g_zunmr3_fortran_call.side != 'R' || g_zunmr3_fortran_call.trans != 'C' ||
        g_zunmr3_fortran_call.m != 2 || g_zunmr3_fortran_call.n != 4 ||
        g_zunmr3_fortran_call.k != 2 || g_zunmr3_fortran_call.l != 1 ||
        g_zunmr3_fortran_call.lda != 2 || g_zunmr3_fortran_call.ldc != 2 ||
        memcmp(g_zunmr3_fortran_call.a_real_snapshot, expected_a_snapshot,
               sizeof(expected_a_snapshot)) != 0 ||
        memcmp(g_zunmr3_fortran_call.tau_real_snapshot, expected_tau,
               sizeof(expected_tau)) != 0 ||
        memcmp(g_zunmr3_fortran_call.c_real_snapshot, expected_c_snapshot,
               sizeof(expected_c_snapshot)) != 0) {
        fprintf(stderr, "[FAIL] ZUNMR3 Fortran->CBLAS thunk did not preserve complex-double row-major RZ-apply semantics\n");
        return 1;
    }
    for (idx = 0; idx < 8; ++idx) {
        if (cdouble_real(c[idx]) != expected_c_out[idx]) {
            fprintf(stderr, "[FAIL] ZUNMR3 Fortran->CBLAS thunk did not copy complex-double RZ-apply output back correctly\n");
            return 1;
        }
    }

    printf("[PASS] ZUNMR3 Fortran->CBLAS thunk translates row-major complex-double RZ-apply matrices and preserves side/trans/l metadata\n");
    return 0;
}

static int check_zunmr3_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_zunmr3_fortran_slot_fn thunk = NULL;
    fb_complex_double_t a[8];
    fb_complex_double_t tau[2] = { make_cdouble(98.0), make_cdouble(99.0) };
    fb_complex_double_t c[8];
    fb_complex_double_t work[2];
    char side = 'R';
    char trans = 'C';
    int m = 2;
    int n = 4;
    int k = 2;
    int l = 1;
    int lda = 2;
    int ldc = 2;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zunmr3_cblas_call, 0, sizeof(g_zunmr3_cblas_call));
    memset(a, 0, sizeof(a));
    memset(c, 0, sizeof(c));
    memset(work, 0, sizeof(work));

    vtable.ext_ops[FB_OP_ZUNMR3][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_zunmr3_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_ZUNMR3);

    thunk = (fb_zunmr3_fortran_slot_fn)vtable.ext_ops[FB_OP_ZUNMR3][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZUNMR3 CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&side, &trans, &m, &n, &k, &l, a, &lda, tau, c, &ldc, work, &info);
    if (info != 124 || g_zunmr3_cblas_call.called != 1 ||
        g_zunmr3_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_zunmr3_cblas_call.side != FB_RIGHT ||
        g_zunmr3_cblas_call.trans != FB_CONJ_TRANS ||
        g_zunmr3_cblas_call.m != 2 || g_zunmr3_cblas_call.n != 4 ||
        g_zunmr3_cblas_call.k != 2 || g_zunmr3_cblas_call.l != 1 ||
        g_zunmr3_cblas_call.lda != 2 || g_zunmr3_cblas_call.ldc != 2 ||
        g_zunmr3_cblas_call.a != a || g_zunmr3_cblas_call.tau != tau ||
        g_zunmr3_cblas_call.c != c) {
        fprintf(stderr, "[FAIL] ZUNMR3 CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] ZUNMR3 CBLAS->Fortran thunk maps the all-pointer ABI into the generic complex-double RZ-apply entry\n");
    return 0;
}

static int check_dormrz_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_dormrz_fn thunk = NULL;
    double a[8] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    double tau[2] = { 106.0, 107.0 };
    double c[8] = { 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0 };
    double expected_a_snapshot[8] = { 1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0 };
    double expected_tau[2] = { 106.0, 107.0 };
    double expected_c_snapshot[8] = { 30.0, 32.0, 34.0, 36.0, 31.0, 33.0, 35.0, 37.0 };
    double expected_c_out[8] = { 2350.0, 2351.0, 2360.0, 2361.0, 2370.0, 2371.0, 2380.0, 2381.0 };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dormrz_fortran_call, 0, sizeof(g_dormrz_fortran_call));

    vtable.ext_ops[FB_OP_DORMRZ][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_dormrz_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_DORMRZ);

    thunk = (fb_dormrz_fn)vtable.ext_ops[FB_OP_DORMRZ][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DORMRZ Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_NO_TRANS, 4, 2, 2, 1, a, 4, tau, c, 2);
    if (info != 0 || g_dormrz_fortran_call.calls != 2 ||
        g_dormrz_fortran_call.query_calls != 1 ||
        g_dormrz_fortran_call.exec_calls != 1 ||
        g_dormrz_fortran_call.side != 'L' || g_dormrz_fortran_call.trans != 'N' ||
        g_dormrz_fortran_call.m != 4 || g_dormrz_fortran_call.n != 2 ||
        g_dormrz_fortran_call.k != 2 || g_dormrz_fortran_call.l != 1 ||
        g_dormrz_fortran_call.lda != 2 || g_dormrz_fortran_call.ldc != 4 ||
        g_dormrz_fortran_call.query_lwork != -1 ||
        g_dormrz_fortran_call.exec_lwork != 19 ||
        memcmp(g_dormrz_fortran_call.a_snapshot, expected_a_snapshot,
               sizeof(expected_a_snapshot)) != 0 ||
        memcmp(g_dormrz_fortran_call.tau_snapshot, expected_tau,
               sizeof(expected_tau)) != 0 ||
        memcmp(g_dormrz_fortran_call.c_snapshot, expected_c_snapshot,
               sizeof(expected_c_snapshot)) != 0 ||
        memcmp(c, expected_c_out, sizeof(expected_c_out)) != 0) {
        fprintf(stderr, "[FAIL] DORMRZ Fortran->CBLAS thunk did not preserve double row-major RZ-apply query semantics\n");
        return 1;
    }

    printf("[PASS] DORMRZ Fortran->CBLAS thunk performs the workspace query and preserves double row-major RZ-apply metadata\n");
    return 0;
}

static int check_dormrz_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_dormrz_fortran_slot_fn thunk = NULL;
    double a[8] = { 0.0 };
    double tau[2] = { 106.0, 107.0 };
    double c[8] = { 0.0 };
    double work[2] = { 0.0 };
    char side = 'L';
    char trans = 'N';
    int m = 4;
    int n = 2;
    int k = 2;
    int l = 1;
    int lda = 2;
    int ldc = 4;
    int lwork = 2;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dormrz_cblas_call, 0, sizeof(g_dormrz_cblas_call));

    vtable.ext_ops[FB_OP_DORMRZ][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_dormrz_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_DORMRZ);

    thunk = (fb_dormrz_fortran_slot_fn)vtable.ext_ops[FB_OP_DORMRZ][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DORMRZ CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&side, &trans, &m, &n, &k, &l, a, &lda, tau, c, &ldc, work, &lwork, &info);
    if (info != 126 || g_dormrz_cblas_call.called != 1 ||
        g_dormrz_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_dormrz_cblas_call.side != FB_LEFT ||
        g_dormrz_cblas_call.trans != FB_NO_TRANS ||
        g_dormrz_cblas_call.m != 4 || g_dormrz_cblas_call.n != 2 ||
        g_dormrz_cblas_call.k != 2 || g_dormrz_cblas_call.l != 1 ||
        g_dormrz_cblas_call.lda != 2 || g_dormrz_cblas_call.ldc != 4 ||
        g_dormrz_cblas_call.a != a || g_dormrz_cblas_call.tau != tau ||
        g_dormrz_cblas_call.c != c) {
        fprintf(stderr, "[FAIL] DORMRZ CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] DORMRZ CBLAS->Fortran thunk maps the all-pointer ABI into the generic double C RZ-apply query entry\n");
    return 0;
}

static int check_zunmrz_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_zunmrz_fn thunk = NULL;
    fb_complex_double_t a[8] = {
        make_cdouble(1.0), make_cdouble(2.0), make_cdouble(3.0), make_cdouble(4.0),
        make_cdouble(5.0), make_cdouble(6.0), make_cdouble(7.0), make_cdouble(8.0)
    };
    fb_complex_double_t tau[2] = { make_cdouble(108.0), make_cdouble(109.0) };
    fb_complex_double_t c[8] = {
        make_cdouble(40.0), make_cdouble(41.0), make_cdouble(42.0), make_cdouble(43.0),
        make_cdouble(44.0), make_cdouble(45.0), make_cdouble(46.0), make_cdouble(47.0)
    };
    double expected_a_snapshot[8] = { 1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0 };
    double expected_tau[2] = { 108.0, 109.0 };
    double expected_c_snapshot[8] = { 40.0, 44.0, 41.0, 45.0, 42.0, 46.0, 43.0, 47.0 };
    double expected_c_out[8] = { 2450.0, 2451.0, 2452.0, 2453.0, 2460.0, 2461.0, 2462.0, 2463.0 };
    int info = 0;
    int idx = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zunmrz_fortran_call, 0, sizeof(g_zunmrz_fortran_call));

    vtable.ext_ops[FB_OP_ZUNMRZ][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_zunmrz_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_ZUNMRZ);

    thunk = (fb_zunmrz_fn)vtable.ext_ops[FB_OP_ZUNMRZ][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZUNMRZ Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, FB_RIGHT, FB_CONJ_TRANS, 2, 4, 2, 1,
                 a, 4, tau, c, 4);
    if (info != 0 || g_zunmrz_fortran_call.calls != 2 ||
        g_zunmrz_fortran_call.query_calls != 1 ||
        g_zunmrz_fortran_call.exec_calls != 1 ||
        g_zunmrz_fortran_call.side != 'R' || g_zunmrz_fortran_call.trans != 'C' ||
        g_zunmrz_fortran_call.m != 2 || g_zunmrz_fortran_call.n != 4 ||
        g_zunmrz_fortran_call.k != 2 || g_zunmrz_fortran_call.l != 1 ||
        g_zunmrz_fortran_call.lda != 2 || g_zunmrz_fortran_call.ldc != 2 ||
        g_zunmrz_fortran_call.query_lwork != -1 ||
        g_zunmrz_fortran_call.exec_lwork != 20 ||
        memcmp(g_zunmrz_fortran_call.a_real_snapshot, expected_a_snapshot,
               sizeof(expected_a_snapshot)) != 0 ||
        memcmp(g_zunmrz_fortran_call.tau_real_snapshot, expected_tau,
               sizeof(expected_tau)) != 0 ||
        memcmp(g_zunmrz_fortran_call.c_real_snapshot, expected_c_snapshot,
               sizeof(expected_c_snapshot)) != 0) {
        fprintf(stderr, "[FAIL] ZUNMRZ Fortran->CBLAS thunk did not preserve complex-double row-major RZ-apply query semantics\n");
        return 1;
    }
    for (idx = 0; idx < 8; ++idx) {
        if (cdouble_real(c[idx]) != expected_c_out[idx]) {
            fprintf(stderr, "[FAIL] ZUNMRZ Fortran->CBLAS thunk did not copy complex-double RZ-apply query output back correctly\n");
            return 1;
        }
    }

    printf("[PASS] ZUNMRZ Fortran->CBLAS thunk performs the workspace query and preserves complex-double row-major RZ-apply metadata\n");
    return 0;
}

static int check_zunmrz_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_zunmrz_fortran_slot_fn thunk = NULL;
    fb_complex_double_t a[8];
    fb_complex_double_t tau[2] = { make_cdouble(108.0), make_cdouble(109.0) };
    fb_complex_double_t c[8];
    fb_complex_double_t work[2];
    char side = 'R';
    char trans = 'C';
    int m = 2;
    int n = 4;
    int k = 2;
    int l = 1;
    int lda = 2;
    int ldc = 2;
    int lwork = 2;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zunmrz_cblas_call, 0, sizeof(g_zunmrz_cblas_call));
    memset(a, 0, sizeof(a));
    memset(c, 0, sizeof(c));
    memset(work, 0, sizeof(work));

    vtable.ext_ops[FB_OP_ZUNMRZ][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_zunmrz_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_ZUNMRZ);

    thunk = (fb_zunmrz_fortran_slot_fn)vtable.ext_ops[FB_OP_ZUNMRZ][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZUNMRZ CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&side, &trans, &m, &n, &k, &l, a, &lda, tau, c, &ldc, work, &lwork, &info);
    if (info != 128 || g_zunmrz_cblas_call.called != 1 ||
        g_zunmrz_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_zunmrz_cblas_call.side != FB_RIGHT ||
        g_zunmrz_cblas_call.trans != FB_CONJ_TRANS ||
        g_zunmrz_cblas_call.m != 2 || g_zunmrz_cblas_call.n != 4 ||
        g_zunmrz_cblas_call.k != 2 || g_zunmrz_cblas_call.l != 1 ||
        g_zunmrz_cblas_call.lda != 2 || g_zunmrz_cblas_call.ldc != 2 ||
        g_zunmrz_cblas_call.a != a || g_zunmrz_cblas_call.tau != tau ||
        g_zunmrz_cblas_call.c != c) {
        fprintf(stderr, "[FAIL] ZUNMRZ CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] ZUNMRZ CBLAS->Fortran thunk maps the all-pointer ABI into the generic complex-double RZ-apply query entry\n");
    return 0;
}

static int check_sormr3_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_sormr3_fn thunk = NULL;
    float a[8] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };
    float tau[2] = { 96.0f, 97.0f };
    float c[8] = { 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f };
    float expected_a_snapshot[8] = { 1.0f, 5.0f, 2.0f, 6.0f, 3.0f, 7.0f, 4.0f, 8.0f };
    float expected_tau[2] = { 96.0f, 97.0f };
    float expected_c_snapshot[8] = { 10.0f, 12.0f, 14.0f, 16.0f, 11.0f, 13.0f, 15.0f, 17.0f };
    float expected_c_out[8] = { 2100.0f, 2101.0f, 2110.0f, 2111.0f, 2120.0f, 2121.0f, 2130.0f, 2131.0f };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sormr3_fortran_call, 0, sizeof(g_sormr3_fortran_call));

    vtable.ext_ops[FB_OP_SORMR3][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_sormr3_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SORMR3);

    thunk = (fb_sormr3_fn)vtable.ext_ops[FB_OP_SORMR3][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SORMR3 Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_NO_TRANS, 4, 2, 2, 1, a, 4, tau, c, 2);
    if (info != 0 || g_sormr3_fortran_call.calls != 1 ||
        g_sormr3_fortran_call.side != 'L' || g_sormr3_fortran_call.trans != 'N' ||
        g_sormr3_fortran_call.m != 4 || g_sormr3_fortran_call.n != 2 ||
        g_sormr3_fortran_call.k != 2 || g_sormr3_fortran_call.l != 1 ||
        g_sormr3_fortran_call.lda != 2 || g_sormr3_fortran_call.ldc != 4 ||
        memcmp(g_sormr3_fortran_call.a_snapshot, expected_a_snapshot,
               sizeof(expected_a_snapshot)) != 0 ||
        memcmp(g_sormr3_fortran_call.tau_snapshot, expected_tau,
               sizeof(expected_tau)) != 0 ||
        memcmp(g_sormr3_fortran_call.c_snapshot, expected_c_snapshot,
               sizeof(expected_c_snapshot)) != 0 ||
        memcmp(c, expected_c_out, sizeof(expected_c_out)) != 0) {
        fprintf(stderr, "[FAIL] SORMR3 Fortran->CBLAS thunk did not preserve row-major RZ-apply semantics\n");
        return 1;
    }

    printf("[PASS] SORMR3 Fortran->CBLAS thunk translates row-major RZ-apply matrices and preserves side/trans/l metadata\n");
    return 0;
}

static int check_sormr3_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_sormr3_fortran_slot_fn thunk = NULL;
    float a[8] = { 0.0f };
    float tau[2] = { 96.0f, 97.0f };
    float c[8] = { 0.0f };
    float work[2] = { 0.0f };
    char side = 'L';
    char trans = 'N';
    int m = 4;
    int n = 2;
    int k = 2;
    int l = 1;
    int lda = 2;
    int ldc = 4;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sormr3_cblas_call, 0, sizeof(g_sormr3_cblas_call));

    vtable.ext_ops[FB_OP_SORMR3][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_sormr3_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SORMR3);

    thunk = (fb_sormr3_fortran_slot_fn)vtable.ext_ops[FB_OP_SORMR3][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SORMR3 CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&side, &trans, &m, &n, &k, &l, a, &lda, tau, c, &ldc, work, &info);
    if (info != 121 || g_sormr3_cblas_call.called != 1 ||
        g_sormr3_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_sormr3_cblas_call.side != FB_LEFT ||
        g_sormr3_cblas_call.trans != FB_NO_TRANS ||
        g_sormr3_cblas_call.m != 4 || g_sormr3_cblas_call.n != 2 ||
        g_sormr3_cblas_call.k != 2 || g_sormr3_cblas_call.l != 1 ||
        g_sormr3_cblas_call.lda != 2 || g_sormr3_cblas_call.ldc != 4 ||
        g_sormr3_cblas_call.a != a || g_sormr3_cblas_call.tau != tau ||
        g_sormr3_cblas_call.c != c) {
        fprintf(stderr, "[FAIL] SORMR3 CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] SORMR3 CBLAS->Fortran thunk maps the all-pointer ABI into the generic C RZ-apply entry\n");
    return 0;
}

static int check_cunmr3_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_cunmr3_fn thunk = NULL;
    fb_complex_float_t a[8] = {
        make_cfloat(1.0f), make_cfloat(2.0f), make_cfloat(3.0f), make_cfloat(4.0f),
        make_cfloat(5.0f), make_cfloat(6.0f), make_cfloat(7.0f), make_cfloat(8.0f)
    };
    fb_complex_float_t tau[2] = { make_cfloat(98.0f), make_cfloat(99.0f) };
    fb_complex_float_t c[8] = {
        make_cfloat(20.0f), make_cfloat(21.0f), make_cfloat(22.0f), make_cfloat(23.0f),
        make_cfloat(24.0f), make_cfloat(25.0f), make_cfloat(26.0f), make_cfloat(27.0f)
    };
    float expected_a_snapshot[8] = { 1.0f, 5.0f, 2.0f, 6.0f, 3.0f, 7.0f, 4.0f, 8.0f };
    float expected_tau[2] = { 98.0f, 99.0f };
    float expected_c_snapshot[8] = { 20.0f, 24.0f, 21.0f, 25.0f, 22.0f, 26.0f, 23.0f, 27.0f };
    float expected_c_out[8] = { 2200.0f, 2201.0f, 2202.0f, 2203.0f, 2210.0f, 2211.0f, 2212.0f, 2213.0f };
    int info = 0;
    int idx = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cunmr3_fortran_call, 0, sizeof(g_cunmr3_fortran_call));

    vtable.ext_ops[FB_OP_CUNMR3][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_cunmr3_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CUNMR3);

    thunk = (fb_cunmr3_fn)vtable.ext_ops[FB_OP_CUNMR3][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CUNMR3 Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, FB_RIGHT, FB_CONJ_TRANS, 2, 4, 2, 1,
                 a, 4, tau, c, 4);
    if (info != 0 || g_cunmr3_fortran_call.calls != 1 ||
        g_cunmr3_fortran_call.side != 'R' || g_cunmr3_fortran_call.trans != 'C' ||
        g_cunmr3_fortran_call.m != 2 || g_cunmr3_fortran_call.n != 4 ||
        g_cunmr3_fortran_call.k != 2 || g_cunmr3_fortran_call.l != 1 ||
        g_cunmr3_fortran_call.lda != 2 || g_cunmr3_fortran_call.ldc != 2 ||
        memcmp(g_cunmr3_fortran_call.a_real_snapshot, expected_a_snapshot,
               sizeof(expected_a_snapshot)) != 0 ||
        memcmp(g_cunmr3_fortran_call.tau_real_snapshot, expected_tau,
               sizeof(expected_tau)) != 0 ||
        memcmp(g_cunmr3_fortran_call.c_real_snapshot, expected_c_snapshot,
               sizeof(expected_c_snapshot)) != 0) {
        fprintf(stderr, "[FAIL] CUNMR3 Fortran->CBLAS thunk did not preserve complex row-major RZ-apply semantics\n");
        return 1;
    }
    for (idx = 0; idx < 8; ++idx) {
        if (cfloat_real(c[idx]) != expected_c_out[idx]) {
            fprintf(stderr, "[FAIL] CUNMR3 Fortran->CBLAS thunk did not copy complex RZ-apply output back correctly\n");
            return 1;
        }
    }

    printf("[PASS] CUNMR3 Fortran->CBLAS thunk translates row-major complex RZ-apply matrices and preserves side/trans/l metadata\n");
    return 0;
}

static int check_cunmr3_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_cunmr3_fortran_slot_fn thunk = NULL;
    fb_complex_float_t a[8];
    fb_complex_float_t tau[2] = { make_cfloat(98.0f), make_cfloat(99.0f) };
    fb_complex_float_t c[8];
    fb_complex_float_t work[2];
    char side = 'R';
    char trans = 'C';
    int m = 2;
    int n = 4;
    int k = 2;
    int l = 1;
    int lda = 2;
    int ldc = 2;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cunmr3_cblas_call, 0, sizeof(g_cunmr3_cblas_call));
    memset(a, 0, sizeof(a));
    memset(c, 0, sizeof(c));
    memset(work, 0, sizeof(work));

    vtable.ext_ops[FB_OP_CUNMR3][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_cunmr3_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CUNMR3);

    thunk = (fb_cunmr3_fortran_slot_fn)vtable.ext_ops[FB_OP_CUNMR3][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CUNMR3 CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&side, &trans, &m, &n, &k, &l, a, &lda, tau, c, &ldc, work, &info);
    if (info != 123 || g_cunmr3_cblas_call.called != 1 ||
        g_cunmr3_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_cunmr3_cblas_call.side != FB_RIGHT ||
        g_cunmr3_cblas_call.trans != FB_CONJ_TRANS ||
        g_cunmr3_cblas_call.m != 2 || g_cunmr3_cblas_call.n != 4 ||
        g_cunmr3_cblas_call.k != 2 || g_cunmr3_cblas_call.l != 1 ||
        g_cunmr3_cblas_call.lda != 2 || g_cunmr3_cblas_call.ldc != 2 ||
        g_cunmr3_cblas_call.a != a || g_cunmr3_cblas_call.tau != tau ||
        g_cunmr3_cblas_call.c != c) {
        fprintf(stderr, "[FAIL] CUNMR3 CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] CUNMR3 CBLAS->Fortran thunk maps the all-pointer ABI into the generic complex RZ-apply entry\n");
    return 0;
}

static int check_sormrz_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_sormrz_fn thunk = NULL;
    float a[8] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };
    float tau[2] = { 106.0f, 107.0f };
    float c[8] = { 30.0f, 31.0f, 32.0f, 33.0f, 34.0f, 35.0f, 36.0f, 37.0f };
    float expected_a_snapshot[8] = { 1.0f, 5.0f, 2.0f, 6.0f, 3.0f, 7.0f, 4.0f, 8.0f };
    float expected_tau[2] = { 106.0f, 107.0f };
    float expected_c_snapshot[8] = { 30.0f, 32.0f, 34.0f, 36.0f, 31.0f, 33.0f, 35.0f, 37.0f };
    float expected_c_out[8] = { 2300.0f, 2301.0f, 2310.0f, 2311.0f, 2320.0f, 2321.0f, 2330.0f, 2331.0f };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sormrz_fortran_call, 0, sizeof(g_sormrz_fortran_call));

    vtable.ext_ops[FB_OP_SORMRZ][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_sormrz_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SORMRZ);

    thunk = (fb_sormrz_fn)vtable.ext_ops[FB_OP_SORMRZ][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SORMRZ Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_NO_TRANS, 4, 2, 2, 1, a, 4, tau, c, 2);
    if (info != 0 || g_sormrz_fortran_call.calls != 2 ||
        g_sormrz_fortran_call.query_calls != 1 ||
        g_sormrz_fortran_call.exec_calls != 1 ||
        g_sormrz_fortran_call.side != 'L' || g_sormrz_fortran_call.trans != 'N' ||
        g_sormrz_fortran_call.m != 4 || g_sormrz_fortran_call.n != 2 ||
        g_sormrz_fortran_call.k != 2 || g_sormrz_fortran_call.l != 1 ||
        g_sormrz_fortran_call.lda != 2 || g_sormrz_fortran_call.ldc != 4 ||
        g_sormrz_fortran_call.query_lwork != -1 ||
        g_sormrz_fortran_call.exec_lwork != 17 ||
        memcmp(g_sormrz_fortran_call.a_snapshot, expected_a_snapshot,
               sizeof(expected_a_snapshot)) != 0 ||
        memcmp(g_sormrz_fortran_call.tau_snapshot, expected_tau,
               sizeof(expected_tau)) != 0 ||
        memcmp(g_sormrz_fortran_call.c_snapshot, expected_c_snapshot,
               sizeof(expected_c_snapshot)) != 0 ||
        memcmp(c, expected_c_out, sizeof(expected_c_out)) != 0) {
        fprintf(stderr, "[FAIL] SORMRZ Fortran->CBLAS thunk did not preserve row-major RZ-apply query semantics\n");
        return 1;
    }

    printf("[PASS] SORMRZ Fortran->CBLAS thunk performs the workspace query and preserves row-major RZ-apply metadata\n");
    return 0;
}

static int check_sormrz_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_sormrz_fortran_slot_fn thunk = NULL;
    float a[8] = { 0.0f };
    float tau[2] = { 106.0f, 107.0f };
    float c[8] = { 0.0f };
    float work[2] = { 0.0f };
    char side = 'L';
    char trans = 'N';
    int m = 4;
    int n = 2;
    int k = 2;
    int l = 1;
    int lda = 2;
    int ldc = 4;
    int lwork = 2;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sormrz_cblas_call, 0, sizeof(g_sormrz_cblas_call));

    vtable.ext_ops[FB_OP_SORMRZ][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_sormrz_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SORMRZ);

    thunk = (fb_sormrz_fortran_slot_fn)vtable.ext_ops[FB_OP_SORMRZ][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SORMRZ CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&side, &trans, &m, &n, &k, &l, a, &lda, tau, c, &ldc, work, &lwork, &info);
    if (info != 125 || g_sormrz_cblas_call.called != 1 ||
        g_sormrz_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_sormrz_cblas_call.side != FB_LEFT ||
        g_sormrz_cblas_call.trans != FB_NO_TRANS ||
        g_sormrz_cblas_call.m != 4 || g_sormrz_cblas_call.n != 2 ||
        g_sormrz_cblas_call.k != 2 || g_sormrz_cblas_call.l != 1 ||
        g_sormrz_cblas_call.lda != 2 || g_sormrz_cblas_call.ldc != 4 ||
        g_sormrz_cblas_call.a != a || g_sormrz_cblas_call.tau != tau ||
        g_sormrz_cblas_call.c != c) {
        fprintf(stderr, "[FAIL] SORMRZ CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] SORMRZ CBLAS->Fortran thunk maps the all-pointer ABI into the generic C RZ-apply query entry\n");
    return 0;
}

static int check_cunmrz_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_cunmrz_fn thunk = NULL;
    fb_complex_float_t a[8] = {
        make_cfloat(1.0f), make_cfloat(2.0f), make_cfloat(3.0f), make_cfloat(4.0f),
        make_cfloat(5.0f), make_cfloat(6.0f), make_cfloat(7.0f), make_cfloat(8.0f)
    };
    fb_complex_float_t tau[2] = { make_cfloat(108.0f), make_cfloat(109.0f) };
    fb_complex_float_t c[8] = {
        make_cfloat(40.0f), make_cfloat(41.0f), make_cfloat(42.0f), make_cfloat(43.0f),
        make_cfloat(44.0f), make_cfloat(45.0f), make_cfloat(46.0f), make_cfloat(47.0f)
    };
    float expected_a_snapshot[8] = { 1.0f, 5.0f, 2.0f, 6.0f, 3.0f, 7.0f, 4.0f, 8.0f };
    float expected_tau[2] = { 108.0f, 109.0f };
    float expected_c_snapshot[8] = { 40.0f, 44.0f, 41.0f, 45.0f, 42.0f, 46.0f, 43.0f, 47.0f };
    float expected_c_out[8] = { 2400.0f, 2401.0f, 2402.0f, 2403.0f, 2410.0f, 2411.0f, 2412.0f, 2413.0f };
    int info = 0;
    int idx = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cunmrz_fortran_call, 0, sizeof(g_cunmrz_fortran_call));

    vtable.ext_ops[FB_OP_CUNMRZ][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_cunmrz_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CUNMRZ);

    thunk = (fb_cunmrz_fn)vtable.ext_ops[FB_OP_CUNMRZ][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CUNMRZ Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, FB_RIGHT, FB_CONJ_TRANS, 2, 4, 2, 1,
                 a, 4, tau, c, 4);
    if (info != 0 || g_cunmrz_fortran_call.calls != 2 ||
        g_cunmrz_fortran_call.query_calls != 1 ||
        g_cunmrz_fortran_call.exec_calls != 1 ||
        g_cunmrz_fortran_call.side != 'R' || g_cunmrz_fortran_call.trans != 'C' ||
        g_cunmrz_fortran_call.m != 2 || g_cunmrz_fortran_call.n != 4 ||
        g_cunmrz_fortran_call.k != 2 || g_cunmrz_fortran_call.l != 1 ||
        g_cunmrz_fortran_call.lda != 2 || g_cunmrz_fortran_call.ldc != 2 ||
        g_cunmrz_fortran_call.query_lwork != -1 ||
        g_cunmrz_fortran_call.exec_lwork != 18 ||
        memcmp(g_cunmrz_fortran_call.a_real_snapshot, expected_a_snapshot,
               sizeof(expected_a_snapshot)) != 0 ||
        memcmp(g_cunmrz_fortran_call.tau_real_snapshot, expected_tau,
               sizeof(expected_tau)) != 0 ||
        memcmp(g_cunmrz_fortran_call.c_real_snapshot, expected_c_snapshot,
               sizeof(expected_c_snapshot)) != 0) {
        fprintf(stderr, "[FAIL] CUNMRZ Fortran->CBLAS thunk did not preserve complex row-major RZ-apply query semantics\n");
        return 1;
    }
    for (idx = 0; idx < 8; ++idx) {
        if (cfloat_real(c[idx]) != expected_c_out[idx]) {
            fprintf(stderr, "[FAIL] CUNMRZ Fortran->CBLAS thunk did not copy complex RZ-apply query output back correctly\n");
            return 1;
        }
    }

    printf("[PASS] CUNMRZ Fortran->CBLAS thunk performs the workspace query and preserves complex row-major RZ-apply metadata\n");
    return 0;
}

static int check_cunmrz_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_cunmrz_fortran_slot_fn thunk = NULL;
    fb_complex_float_t a[8];
    fb_complex_float_t tau[2] = { make_cfloat(108.0f), make_cfloat(109.0f) };
    fb_complex_float_t c[8];
    fb_complex_float_t work[2];
    char side = 'R';
    char trans = 'C';
    int m = 2;
    int n = 4;
    int k = 2;
    int l = 1;
    int lda = 2;
    int ldc = 2;
    int lwork = 2;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cunmrz_cblas_call, 0, sizeof(g_cunmrz_cblas_call));
    memset(a, 0, sizeof(a));
    memset(c, 0, sizeof(c));
    memset(work, 0, sizeof(work));

    vtable.ext_ops[FB_OP_CUNMRZ][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_cunmrz_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CUNMRZ);

    thunk = (fb_cunmrz_fortran_slot_fn)vtable.ext_ops[FB_OP_CUNMRZ][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CUNMRZ CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&side, &trans, &m, &n, &k, &l, a, &lda, tau, c, &ldc, work, &lwork, &info);
    if (info != 127 || g_cunmrz_cblas_call.called != 1 ||
        g_cunmrz_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_cunmrz_cblas_call.side != FB_RIGHT ||
        g_cunmrz_cblas_call.trans != FB_CONJ_TRANS ||
        g_cunmrz_cblas_call.m != 2 || g_cunmrz_cblas_call.n != 4 ||
        g_cunmrz_cblas_call.k != 2 || g_cunmrz_cblas_call.l != 1 ||
        g_cunmrz_cblas_call.lda != 2 || g_cunmrz_cblas_call.ldc != 2 ||
        g_cunmrz_cblas_call.a != a || g_cunmrz_cblas_call.tau != tau ||
        g_cunmrz_cblas_call.c != c) {
        fprintf(stderr, "[FAIL] CUNMRZ CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] CUNMRZ CBLAS->Fortran thunk maps the all-pointer ABI into the generic complex RZ-apply query entry\n");
    return 0;
}

int main(void)
{
    if (check_sormr3_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_sormr3_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_dormr3_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_dormr3_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_cunmr3_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_cunmr3_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_zunmr3_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_zunmr3_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_sormrz_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_sormrz_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_dormrz_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_dormrz_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_cunmrz_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_cunmrz_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_zunmrz_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_zunmrz_cblas_to_fortran() != 0) {
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}