#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef int (*fb_sorgbr_fn)(fb_layout_t layout, char vect, int m, int n,
                            int k, float *a, int lda, const float *tau);
typedef int (*fb_dorgbr_fn)(fb_layout_t layout, char vect, int m, int n,
                            int k, double *a, int lda, const double *tau);
typedef int (*fb_cungbr_fn)(fb_layout_t layout, char vect, int m, int n,
                            int k, fb_complex_float_t *a, int lda,
                            const fb_complex_float_t *tau);
typedef int (*fb_zungbr_fn)(fb_layout_t layout, char vect, int m, int n,
                            int k, fb_complex_double_t *a, int lda,
                            const fb_complex_double_t *tau);
typedef int (*fb_sormbr_fn)(fb_layout_t layout, char vect, char side,
                            char trans, int m, int n, int k,
                            const float *a, int lda, const float *tau,
                            float *c, int ldc);
typedef int (*fb_dormbr_fn)(fb_layout_t layout, char vect, char side,
                            char trans, int m, int n, int k,
                            const double *a, int lda, const double *tau,
                            double *c, int ldc);
typedef int (*fb_cunmbr_fn)(fb_layout_t layout, char vect, char side,
                            char trans, int m, int n, int k,
                            const fb_complex_float_t *a, int lda,
                            const fb_complex_float_t *tau,
                            fb_complex_float_t *c, int ldc);
typedef int (*fb_zunmbr_fn)(fb_layout_t layout, char vect, char side,
                            char trans, int m, int n, int k,
                            const fb_complex_double_t *a, int lda,
                            const fb_complex_double_t *tau,
                            fb_complex_double_t *c, int ldc);

typedef void (*fb_sorgbr_fortran_slot_fn)(char *vect, int *m, int *n, int *k,
                                          float *a, int *lda, float *tau,
                                          float *work, int *lwork, int *info);
typedef void (*fb_dorgbr_fortran_slot_fn)(char *vect, int *m, int *n, int *k,
                                          double *a, int *lda, double *tau,
                                          double *work, int *lwork,
                                          int *info);
typedef void (*fb_cungbr_fortran_slot_fn)(char *vect, int *m, int *n, int *k,
                                          fb_complex_float_t *a, int *lda,
                                          fb_complex_float_t *tau,
                                          fb_complex_float_t *work,
                                          int *lwork, int *info);
typedef void (*fb_zungbr_fortran_slot_fn)(char *vect, int *m, int *n, int *k,
                                          fb_complex_double_t *a, int *lda,
                                          fb_complex_double_t *tau,
                                          fb_complex_double_t *work,
                                          int *lwork, int *info);
typedef void (*fb_sormbr_fortran_slot_fn)(char *vect, char *side, char *trans,
                                          int *m, int *n, int *k,
                                          float *a, int *lda, float *tau,
                                          float *c, int *ldc, float *work,
                                          int *lwork, int *info);
typedef void (*fb_dormbr_fortran_slot_fn)(char *vect, char *side, char *trans,
                                          int *m, int *n, int *k,
                                          double *a, int *lda, double *tau,
                                          double *c, int *ldc, double *work,
                                          int *lwork, int *info);
typedef void (*fb_cunmbr_fortran_slot_fn)(char *vect, char *side, char *trans,
                                          int *m, int *n, int *k,
                                          fb_complex_float_t *a, int *lda,
                                          fb_complex_float_t *tau,
                                          fb_complex_float_t *c, int *ldc,
                                          fb_complex_float_t *work,
                                          int *lwork, int *info);
typedef void (*fb_zunmbr_fortran_slot_fn)(char *vect, char *side, char *trans,
                                          int *m, int *n, int *k,
                                          fb_complex_double_t *a, int *lda,
                                          fb_complex_double_t *tau,
                                          fb_complex_double_t *c, int *ldc,
                                          fb_complex_double_t *work,
                                          int *lwork, int *info);

static struct {
    int query_calls;
    int solve_calls;
    char vect;
    int m;
    int n;
    int k;
    int lda;
    int lwork_query;
    int lwork_solve;
    float a_snapshot[6];
    float tau_snapshot[2];
} g_sorgbr_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    char vect;
    int m;
    int n;
    int k;
    int lda;
    float *a;
    const float *tau;
} g_sorgbr_cblas_call;

static struct {
    int query_calls;
    int solve_calls;
    char vect;
    int m;
    int n;
    int k;
    int lda;
    int lwork_query;
    int lwork_solve;
} g_dorgbr_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    char vect;
    int m;
    int n;
    int k;
    int lda;
    double *a;
    const double *tau;
} g_dorgbr_cblas_call;

static struct {
    int query_calls;
    int solve_calls;
    char vect;
    int m;
    int n;
    int k;
    int lda;
    int lwork_query;
    int lwork_solve;
    float a_real_snapshot[6];
    float tau_real_snapshot[2];
} g_cungbr_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    char vect;
    int m;
    int n;
    int k;
    int lda;
    fb_complex_float_t *a;
    const fb_complex_float_t *tau;
} g_cungbr_cblas_call;

static struct {
    int query_calls;
    int solve_calls;
    char vect;
    int m;
    int n;
    int k;
    int lda;
    int lwork_query;
    int lwork_solve;
} g_zungbr_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    char vect;
    int m;
    int n;
    int k;
    int lda;
    fb_complex_double_t *a;
    const fb_complex_double_t *tau;
} g_zungbr_cblas_call;

static struct {
    int query_calls;
    int solve_calls;
    char vect;
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
} g_sormbr_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    char vect;
    char side;
    char trans;
    int m;
    int n;
    int k;
    int lda;
    int ldc;
    const float *a;
    const float *tau;
    float *c;
} g_sormbr_cblas_call;

static struct {
    int query_calls;
    int solve_calls;
    char vect;
    char side;
    char trans;
    int m;
    int n;
    int k;
    int lda;
    int ldc;
    int lwork_query;
    int lwork_solve;
} g_dormbr_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    char vect;
    char side;
    char trans;
    int m;
    int n;
    int k;
    int lda;
    int ldc;
    const double *a;
    const double *tau;
    double *c;
} g_dormbr_cblas_call;

static struct {
    int query_calls;
    int solve_calls;
    char vect;
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
} g_cunmbr_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    char vect;
    char side;
    char trans;
    int m;
    int n;
    int k;
    int lda;
    int ldc;
    const fb_complex_float_t *a;
    const fb_complex_float_t *tau;
    fb_complex_float_t *c;
} g_cunmbr_cblas_call;

static struct {
    int query_calls;
    int solve_calls;
    char vect;
    char side;
    char trans;
    int m;
    int n;
    int k;
    int lda;
    int ldc;
    int lwork_query;
    int lwork_solve;
} g_zunmbr_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    char vect;
    char side;
    char trans;
    int m;
    int n;
    int k;
    int lda;
    int ldc;
    const fb_complex_double_t *a;
    const fb_complex_double_t *tau;
    fb_complex_double_t *c;
} g_zunmbr_cblas_call;

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

static void stub_sorgbr_fortran(char *vect, int *m, int *n, int *k, float *a,
                                int *lda, float *tau, float *work,
                                int *lwork, int *info)
{
    int row = 0;
    int col = 0;

    if (*lwork == -1) {
        g_sorgbr_fortran_call.query_calls += 1;
        g_sorgbr_fortran_call.vect = *vect;
        g_sorgbr_fortran_call.m = *m;
        g_sorgbr_fortran_call.n = *n;
        g_sorgbr_fortran_call.k = *k;
        g_sorgbr_fortran_call.lda = *lda;
        g_sorgbr_fortran_call.lwork_query = *lwork;
        a[0] = -999.0f;
        tau[0] = -777.0f;
        work[0] = 5.0f;
        *info = 0;
        return;
    }

    g_sorgbr_fortran_call.solve_calls += 1;
    g_sorgbr_fortran_call.vect = *vect;
    g_sorgbr_fortran_call.m = *m;
    g_sorgbr_fortran_call.n = *n;
    g_sorgbr_fortran_call.k = *k;
    g_sorgbr_fortran_call.lda = *lda;
    g_sorgbr_fortran_call.lwork_solve = *lwork;
    for (col = 0; col < *n; ++col) {
        for (row = 0; row < *m; ++row) {
            g_sorgbr_fortran_call.a_snapshot[(col * (*m)) + row] =
                a[(col * (*lda)) + row];
            a[(col * (*lda)) + row] = (float)(500 + (10 * row) + col);
        }
    }
    for (row = 0; row < *k; ++row) {
        g_sorgbr_fortran_call.tau_snapshot[row] = tau[row];
    }
    *info = 0;
}

static int stub_sorgbr_cblas(const fb_layout_t layout, char vect,
                             const int m, const int n, const int k,
                             float *a, const int lda, const float *tau)
{
    g_sorgbr_cblas_call.called += 1;
    g_sorgbr_cblas_call.layout = layout;
    g_sorgbr_cblas_call.vect = vect;
    g_sorgbr_cblas_call.m = m;
    g_sorgbr_cblas_call.n = n;
    g_sorgbr_cblas_call.k = k;
    g_sorgbr_cblas_call.lda = lda;
    g_sorgbr_cblas_call.a = a;
    g_sorgbr_cblas_call.tau = tau;
    return 81;
}

static void stub_dorgbr_fortran(char *vect, int *m, int *n, int *k,
                                double *a, int *lda, double *tau,
                                double *work, int *lwork, int *info)
{
    if (*lwork == -1) {
        g_dorgbr_fortran_call.query_calls += 1;
        g_dorgbr_fortran_call.vect = *vect;
        g_dorgbr_fortran_call.m = *m;
        g_dorgbr_fortran_call.n = *n;
        g_dorgbr_fortran_call.k = *k;
        g_dorgbr_fortran_call.lda = *lda;
        g_dorgbr_fortran_call.lwork_query = *lwork;
        work[0] = 9.0;
        *info = 0;
        return;
    }

    g_dorgbr_fortran_call.solve_calls += 1;
    g_dorgbr_fortran_call.vect = *vect;
    g_dorgbr_fortran_call.m = *m;
    g_dorgbr_fortran_call.n = *n;
    g_dorgbr_fortran_call.k = *k;
    g_dorgbr_fortran_call.lda = *lda;
    g_dorgbr_fortran_call.lwork_solve = *lwork;
    a[0] = 901.0;
    tau[0] = 902.0;
    *info = 0;
}

static int stub_dorgbr_cblas(const fb_layout_t layout, char vect,
                             const int m, const int n, const int k,
                             double *a, const int lda, const double *tau)
{
    g_dorgbr_cblas_call.called += 1;
    g_dorgbr_cblas_call.layout = layout;
    g_dorgbr_cblas_call.vect = vect;
    g_dorgbr_cblas_call.m = m;
    g_dorgbr_cblas_call.n = n;
    g_dorgbr_cblas_call.k = k;
    g_dorgbr_cblas_call.lda = lda;
    g_dorgbr_cblas_call.a = a;
    g_dorgbr_cblas_call.tau = tau;
    return 82;
}

static void stub_cungbr_fortran(char *vect, int *m, int *n, int *k,
                                fb_complex_float_t *a, int *lda,
                                fb_complex_float_t *tau,
                                fb_complex_float_t *work, int *lwork,
                                int *info)
{
    int row = 0;
    int col = 0;

    if (*lwork == -1) {
        g_cungbr_fortran_call.query_calls += 1;
        g_cungbr_fortran_call.vect = *vect;
        g_cungbr_fortran_call.m = *m;
        g_cungbr_fortran_call.n = *n;
        g_cungbr_fortran_call.k = *k;
        g_cungbr_fortran_call.lda = *lda;
        g_cungbr_fortran_call.lwork_query = *lwork;
        a[0] = make_cfloat(-999.0f);
        tau[0] = make_cfloat(-777.0f);
        work[0] = make_cfloat(6.0f);
        *info = 0;
        return;
    }

    g_cungbr_fortran_call.solve_calls += 1;
    g_cungbr_fortran_call.vect = *vect;
    g_cungbr_fortran_call.m = *m;
    g_cungbr_fortran_call.n = *n;
    g_cungbr_fortran_call.k = *k;
    g_cungbr_fortran_call.lda = *lda;
    g_cungbr_fortran_call.lwork_solve = *lwork;
    for (col = 0; col < *n; ++col) {
        for (row = 0; row < *m; ++row) {
            g_cungbr_fortran_call.a_real_snapshot[(col * (*m)) + row] =
                cfloat_real(a[(col * (*lda)) + row]);
            a[(col * (*lda)) + row] = make_cfloat((float)(600 + (10 * row) + col));
        }
    }
    for (row = 0; row < *k; ++row) {
        g_cungbr_fortran_call.tau_real_snapshot[row] = cfloat_real(tau[row]);
    }
    *info = 0;
}

static int stub_cungbr_cblas(const fb_layout_t layout, char vect,
                             const int m, const int n, const int k,
                             fb_complex_float_t *a, const int lda,
                             const fb_complex_float_t *tau)
{
    g_cungbr_cblas_call.called += 1;
    g_cungbr_cblas_call.layout = layout;
    g_cungbr_cblas_call.vect = vect;
    g_cungbr_cblas_call.m = m;
    g_cungbr_cblas_call.n = n;
    g_cungbr_cblas_call.k = k;
    g_cungbr_cblas_call.lda = lda;
    g_cungbr_cblas_call.a = a;
    g_cungbr_cblas_call.tau = tau;
    return 83;
}

static void stub_zungbr_fortran(char *vect, int *m, int *n, int *k,
                                fb_complex_double_t *a, int *lda,
                                fb_complex_double_t *tau,
                                fb_complex_double_t *work, int *lwork,
                                int *info)
{
    if (*lwork == -1) {
        g_zungbr_fortran_call.query_calls += 1;
        g_zungbr_fortran_call.vect = *vect;
        g_zungbr_fortran_call.m = *m;
        g_zungbr_fortran_call.n = *n;
        g_zungbr_fortran_call.k = *k;
        g_zungbr_fortran_call.lda = *lda;
        g_zungbr_fortran_call.lwork_query = *lwork;
        work[0] = make_cdouble(10.0);
        *info = 0;
        return;
    }

    g_zungbr_fortran_call.solve_calls += 1;
    g_zungbr_fortran_call.vect = *vect;
    g_zungbr_fortran_call.m = *m;
    g_zungbr_fortran_call.n = *n;
    g_zungbr_fortran_call.k = *k;
    g_zungbr_fortran_call.lda = *lda;
    g_zungbr_fortran_call.lwork_solve = *lwork;
    a[0] = make_cdouble(1001.0);
    tau[0] = make_cdouble(1002.0);
    *info = 0;
}

static int stub_zungbr_cblas(const fb_layout_t layout, char vect,
                             const int m, const int n, const int k,
                             fb_complex_double_t *a, const int lda,
                             const fb_complex_double_t *tau)
{
    g_zungbr_cblas_call.called += 1;
    g_zungbr_cblas_call.layout = layout;
    g_zungbr_cblas_call.vect = vect;
    g_zungbr_cblas_call.m = m;
    g_zungbr_cblas_call.n = n;
    g_zungbr_cblas_call.k = k;
    g_zungbr_cblas_call.lda = lda;
    g_zungbr_cblas_call.a = a;
    g_zungbr_cblas_call.tau = tau;
    return 84;
}

static void stub_sormbr_fortran(char *vect, char *side, char *trans, int *m,
                                int *n, int *k, float *a, int *lda,
                                float *tau, float *c, int *ldc, float *work,
                                int *lwork, int *info)
{
    int row = 0;
    int col = 0;

    if (*lwork == -1) {
        g_sormbr_fortran_call.query_calls += 1;
        g_sormbr_fortran_call.vect = *vect;
        g_sormbr_fortran_call.side = *side;
        g_sormbr_fortran_call.trans = *trans;
        g_sormbr_fortran_call.m = *m;
        g_sormbr_fortran_call.n = *n;
        g_sormbr_fortran_call.k = *k;
        g_sormbr_fortran_call.lda = *lda;
        g_sormbr_fortran_call.ldc = *ldc;
        g_sormbr_fortran_call.lwork_query = *lwork;
        a[0] = -999.0f;
        tau[0] = -777.0f;
        c[0] = -555.0f;
        work[0] = 7.0f;
        *info = 0;
        return;
    }

    g_sormbr_fortran_call.solve_calls += 1;
    g_sormbr_fortran_call.vect = *vect;
    g_sormbr_fortran_call.side = *side;
    g_sormbr_fortran_call.trans = *trans;
    g_sormbr_fortran_call.m = *m;
    g_sormbr_fortran_call.n = *n;
    g_sormbr_fortran_call.k = *k;
    g_sormbr_fortran_call.lda = *lda;
    g_sormbr_fortran_call.ldc = *ldc;
    g_sormbr_fortran_call.lwork_solve = *lwork;
    for (col = 0; col < 2; ++col) {
        for (row = 0; row < 2; ++row) {
            g_sormbr_fortran_call.a_snapshot[(col * 2) + row] =
                a[(col * (*lda)) + row];
        }
    }
    for (col = 0; col < *n; ++col) {
        for (row = 0; row < *m; ++row) {
            g_sormbr_fortran_call.c_snapshot[(col * (*m)) + row] =
                c[(col * (*ldc)) + row];
            c[(col * (*ldc)) + row] = (float)(700 + (10 * row) + col);
        }
    }
    for (row = 0; row < *k; ++row) {
        g_sormbr_fortran_call.tau_snapshot[row] = tau[row];
    }
    a[0] = -123.0f;
    tau[0] = -456.0f;
    *info = 0;
}

static int stub_sormbr_cblas(const fb_layout_t layout, char vect, char side,
                             char trans, const int m, const int n,
                             const int k, const float *a, const int lda,
                             const float *tau, float *c, const int ldc)
{
    g_sormbr_cblas_call.called += 1;
    g_sormbr_cblas_call.layout = layout;
    g_sormbr_cblas_call.vect = vect;
    g_sormbr_cblas_call.side = side;
    g_sormbr_cblas_call.trans = trans;
    g_sormbr_cblas_call.m = m;
    g_sormbr_cblas_call.n = n;
    g_sormbr_cblas_call.k = k;
    g_sormbr_cblas_call.lda = lda;
    g_sormbr_cblas_call.ldc = ldc;
    g_sormbr_cblas_call.a = a;
    g_sormbr_cblas_call.tau = tau;
    g_sormbr_cblas_call.c = c;
    return 85;
}

static void stub_dormbr_fortran(char *vect, char *side, char *trans, int *m,
                                int *n, int *k, double *a, int *lda,
                                double *tau, double *c, int *ldc,
                                double *work, int *lwork, int *info)
{
    if (*lwork == -1) {
        g_dormbr_fortran_call.query_calls += 1;
        g_dormbr_fortran_call.vect = *vect;
        g_dormbr_fortran_call.side = *side;
        g_dormbr_fortran_call.trans = *trans;
        g_dormbr_fortran_call.m = *m;
        g_dormbr_fortran_call.n = *n;
        g_dormbr_fortran_call.k = *k;
        g_dormbr_fortran_call.lda = *lda;
        g_dormbr_fortran_call.ldc = *ldc;
        g_dormbr_fortran_call.lwork_query = *lwork;
        work[0] = 11.0;
        *info = 0;
        return;
    }

    g_dormbr_fortran_call.solve_calls += 1;
    g_dormbr_fortran_call.vect = *vect;
    g_dormbr_fortran_call.side = *side;
    g_dormbr_fortran_call.trans = *trans;
    g_dormbr_fortran_call.m = *m;
    g_dormbr_fortran_call.n = *n;
    g_dormbr_fortran_call.k = *k;
    g_dormbr_fortran_call.lda = *lda;
    g_dormbr_fortran_call.ldc = *ldc;
    g_dormbr_fortran_call.lwork_solve = *lwork;
    c[0] = 1101.0;
    a[0] = -123.0;
    tau[0] = -456.0;
    *info = 0;
}

static int stub_dormbr_cblas(const fb_layout_t layout, char vect, char side,
                             char trans, const int m, const int n,
                             const int k, const double *a, const int lda,
                             const double *tau, double *c, const int ldc)
{
    g_dormbr_cblas_call.called += 1;
    g_dormbr_cblas_call.layout = layout;
    g_dormbr_cblas_call.vect = vect;
    g_dormbr_cblas_call.side = side;
    g_dormbr_cblas_call.trans = trans;
    g_dormbr_cblas_call.m = m;
    g_dormbr_cblas_call.n = n;
    g_dormbr_cblas_call.k = k;
    g_dormbr_cblas_call.lda = lda;
    g_dormbr_cblas_call.ldc = ldc;
    g_dormbr_cblas_call.a = a;
    g_dormbr_cblas_call.tau = tau;
    g_dormbr_cblas_call.c = c;
    return 86;
}

static void stub_cunmbr_fortran(char *vect, char *side, char *trans, int *m,
                                int *n, int *k, fb_complex_float_t *a,
                                int *lda, fb_complex_float_t *tau,
                                fb_complex_float_t *c, int *ldc,
                                fb_complex_float_t *work, int *lwork,
                                int *info)
{
    int row = 0;
    int col = 0;

    if (*lwork == -1) {
        g_cunmbr_fortran_call.query_calls += 1;
        g_cunmbr_fortran_call.vect = *vect;
        g_cunmbr_fortran_call.side = *side;
        g_cunmbr_fortran_call.trans = *trans;
        g_cunmbr_fortran_call.m = *m;
        g_cunmbr_fortran_call.n = *n;
        g_cunmbr_fortran_call.k = *k;
        g_cunmbr_fortran_call.lda = *lda;
        g_cunmbr_fortran_call.ldc = *ldc;
        g_cunmbr_fortran_call.lwork_query = *lwork;
        a[0] = make_cfloat(-999.0f);
        tau[0] = make_cfloat(-777.0f);
        c[0] = make_cfloat(-555.0f);
        work[0] = make_cfloat(8.0f);
        *info = 0;
        return;
    }

    g_cunmbr_fortran_call.solve_calls += 1;
    g_cunmbr_fortran_call.vect = *vect;
    g_cunmbr_fortran_call.side = *side;
    g_cunmbr_fortran_call.trans = *trans;
    g_cunmbr_fortran_call.m = *m;
    g_cunmbr_fortran_call.n = *n;
    g_cunmbr_fortran_call.k = *k;
    g_cunmbr_fortran_call.lda = *lda;
    g_cunmbr_fortran_call.ldc = *ldc;
    g_cunmbr_fortran_call.lwork_solve = *lwork;
    for (col = 0; col < 3; ++col) {
        for (row = 0; row < 2; ++row) {
            g_cunmbr_fortran_call.a_real_snapshot[(col * 2) + row] =
                cfloat_real(a[(col * (*lda)) + row]);
        }
    }
    for (col = 0; col < *n; ++col) {
        for (row = 0; row < *m; ++row) {
            g_cunmbr_fortran_call.c_real_snapshot[(col * (*m)) + row] =
                cfloat_real(c[(col * (*ldc)) + row]);
            c[(col * (*ldc)) + row] = make_cfloat((float)(800 + (10 * row) + col));
        }
    }
    for (row = 0; row < *k; ++row) {
        g_cunmbr_fortran_call.tau_real_snapshot[row] = cfloat_real(tau[row]);
    }
    a[0] = make_cfloat(-123.0f);
    tau[0] = make_cfloat(-456.0f);
    *info = 0;
}

static int stub_cunmbr_cblas(const fb_layout_t layout, char vect, char side,
                             char trans, const int m, const int n,
                             const int k, const fb_complex_float_t *a,
                             const int lda, const fb_complex_float_t *tau,
                             fb_complex_float_t *c, const int ldc)
{
    g_cunmbr_cblas_call.called += 1;
    g_cunmbr_cblas_call.layout = layout;
    g_cunmbr_cblas_call.vect = vect;
    g_cunmbr_cblas_call.side = side;
    g_cunmbr_cblas_call.trans = trans;
    g_cunmbr_cblas_call.m = m;
    g_cunmbr_cblas_call.n = n;
    g_cunmbr_cblas_call.k = k;
    g_cunmbr_cblas_call.lda = lda;
    g_cunmbr_cblas_call.ldc = ldc;
    g_cunmbr_cblas_call.a = a;
    g_cunmbr_cblas_call.tau = tau;
    g_cunmbr_cblas_call.c = c;
    return 87;
}

static void stub_zunmbr_fortran(char *vect, char *side, char *trans, int *m,
                                int *n, int *k, fb_complex_double_t *a,
                                int *lda, fb_complex_double_t *tau,
                                fb_complex_double_t *c, int *ldc,
                                fb_complex_double_t *work, int *lwork,
                                int *info)
{
    if (*lwork == -1) {
        g_zunmbr_fortran_call.query_calls += 1;
        g_zunmbr_fortran_call.vect = *vect;
        g_zunmbr_fortran_call.side = *side;
        g_zunmbr_fortran_call.trans = *trans;
        g_zunmbr_fortran_call.m = *m;
        g_zunmbr_fortran_call.n = *n;
        g_zunmbr_fortran_call.k = *k;
        g_zunmbr_fortran_call.lda = *lda;
        g_zunmbr_fortran_call.ldc = *ldc;
        g_zunmbr_fortran_call.lwork_query = *lwork;
        work[0] = make_cdouble(12.0);
        *info = 0;
        return;
    }

    g_zunmbr_fortran_call.solve_calls += 1;
    g_zunmbr_fortran_call.vect = *vect;
    g_zunmbr_fortran_call.side = *side;
    g_zunmbr_fortran_call.trans = *trans;
    g_zunmbr_fortran_call.m = *m;
    g_zunmbr_fortran_call.n = *n;
    g_zunmbr_fortran_call.k = *k;
    g_zunmbr_fortran_call.lda = *lda;
    g_zunmbr_fortran_call.ldc = *ldc;
    g_zunmbr_fortran_call.lwork_solve = *lwork;
    c[0] = make_cdouble(1201.0);
    a[0] = make_cdouble(-123.0);
    tau[0] = make_cdouble(-456.0);
    *info = 0;
}

static int stub_zunmbr_cblas(const fb_layout_t layout, char vect, char side,
                             char trans, const int m, const int n,
                             const int k, const fb_complex_double_t *a,
                             const int lda, const fb_complex_double_t *tau,
                             fb_complex_double_t *c, const int ldc)
{
    g_zunmbr_cblas_call.called += 1;
    g_zunmbr_cblas_call.layout = layout;
    g_zunmbr_cblas_call.vect = vect;
    g_zunmbr_cblas_call.side = side;
    g_zunmbr_cblas_call.trans = trans;
    g_zunmbr_cblas_call.m = m;
    g_zunmbr_cblas_call.n = n;
    g_zunmbr_cblas_call.k = k;
    g_zunmbr_cblas_call.lda = lda;
    g_zunmbr_cblas_call.ldc = ldc;
    g_zunmbr_cblas_call.a = a;
    g_zunmbr_cblas_call.tau = tau;
    g_zunmbr_cblas_call.c = c;
    return 88;
}

static int check_sorgbr_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_sorgbr_fn thunk = NULL;
    float a[6] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
    float tau[2] = { 7.0f, 8.0f };
    float expected_a_snapshot[6] = { 1.0f, 3.0f, 5.0f, 2.0f, 4.0f, 6.0f };
    float expected_a_out[6] = { 500.0f, 501.0f, 510.0f, 511.0f, 520.0f, 521.0f };
    float expected_tau[2] = { 7.0f, 8.0f };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sorgbr_fortran_call, 0, sizeof(g_sorgbr_fortran_call));

    vtable.ext_ops[FB_OP_SORGBR][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_sorgbr_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SORGBR);

    thunk = (fb_sorgbr_fn)vtable.ext_ops[FB_OP_SORGBR][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SORGBR Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 'Q', 3, 2, 2, a, 2, tau);
    if (info != 0 || g_sorgbr_fortran_call.query_calls != 1 ||
        g_sorgbr_fortran_call.solve_calls != 1 ||
        g_sorgbr_fortran_call.vect != 'Q' ||
        g_sorgbr_fortran_call.m != 3 || g_sorgbr_fortran_call.n != 2 ||
        g_sorgbr_fortran_call.k != 2 || g_sorgbr_fortran_call.lda != 3 ||
        g_sorgbr_fortran_call.lwork_query != -1 ||
        g_sorgbr_fortran_call.lwork_solve != 5 ||
        memcmp(g_sorgbr_fortran_call.a_snapshot, expected_a_snapshot,
               sizeof(expected_a_snapshot)) != 0 ||
        memcmp(g_sorgbr_fortran_call.tau_snapshot, expected_tau,
               sizeof(expected_tau)) != 0 ||
        memcmp(a, expected_a_out, sizeof(expected_a_out)) != 0 ||
        memcmp(tau, expected_tau, sizeof(expected_tau)) != 0) {
        fprintf(stderr, "[FAIL] SORGBR Fortran->CBLAS thunk did not preserve vect-aware generator semantics\n");
        return 1;
    }

    printf("[PASS] SORGBR Fortran->CBLAS thunk translates row-major generator matrices and preserves TAU input across lwork query\n");
    return 0;
}

static int check_sorgbr_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_sorgbr_fortran_slot_fn thunk = NULL;
    char vect = 'Q';
    float a[6] = { 0.0f };
    float tau[2] = { 7.0f, 8.0f };
    float work[4] = { 0.0f };
    int m = 3;
    int n = 2;
    int k = 2;
    int lda = 3;
    int lwork = 4;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sorgbr_cblas_call, 0, sizeof(g_sorgbr_cblas_call));

    vtable.ext_ops[FB_OP_SORGBR][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_sorgbr_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SORGBR);

    thunk = (fb_sorgbr_fortran_slot_fn)vtable.ext_ops[FB_OP_SORGBR][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SORGBR CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&vect, &m, &n, &k, a, &lda, tau, work, &lwork, &info);
    if (info != 81 || g_sorgbr_cblas_call.called != 1 ||
        g_sorgbr_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_sorgbr_cblas_call.vect != 'Q' ||
        g_sorgbr_cblas_call.m != 3 || g_sorgbr_cblas_call.n != 2 ||
        g_sorgbr_cblas_call.k != 2 || g_sorgbr_cblas_call.lda != 3 ||
        g_sorgbr_cblas_call.a != a || g_sorgbr_cblas_call.tau != tau) {
        fprintf(stderr, "[FAIL] SORGBR CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] SORGBR CBLAS->Fortran thunk maps vect and all-pointer ABI into the C generator entry\n");
    return 0;
}

static int check_cungbr_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_cungbr_fn thunk = NULL;
    fb_complex_float_t a[6] = {
        make_cfloat(1.0f), make_cfloat(2.0f), make_cfloat(3.0f),
        make_cfloat(4.0f), make_cfloat(5.0f), make_cfloat(6.0f)
    };
    fb_complex_float_t tau[2] = { make_cfloat(9.0f), make_cfloat(10.0f) };
    float expected_a_snapshot[6] = { 1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f };
    float expected_tau[2] = { 9.0f, 10.0f };
    float expected_a_out[6] = { 600.0f, 601.0f, 602.0f, 610.0f, 611.0f, 612.0f };
    int info = 0;
    int idx = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cungbr_fortran_call, 0, sizeof(g_cungbr_fortran_call));

    vtable.ext_ops[FB_OP_CUNGBR][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_cungbr_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CUNGBR);

    thunk = (fb_cungbr_fn)vtable.ext_ops[FB_OP_CUNGBR][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CUNGBR Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 'P', 2, 3, 2, a, 3, tau);
    if (info != 0 || g_cungbr_fortran_call.query_calls != 1 ||
        g_cungbr_fortran_call.solve_calls != 1 ||
        g_cungbr_fortran_call.vect != 'P' ||
        g_cungbr_fortran_call.m != 2 || g_cungbr_fortran_call.n != 3 ||
        g_cungbr_fortran_call.k != 2 || g_cungbr_fortran_call.lda != 2 ||
        g_cungbr_fortran_call.lwork_query != -1 ||
        g_cungbr_fortran_call.lwork_solve != 6 ||
        memcmp(g_cungbr_fortran_call.a_real_snapshot, expected_a_snapshot,
               sizeof(expected_a_snapshot)) != 0 ||
        memcmp(g_cungbr_fortran_call.tau_real_snapshot, expected_tau,
               sizeof(expected_tau)) != 0) {
        fprintf(stderr, "[FAIL] CUNGBR Fortran->CBLAS thunk did not preserve complex vect-aware generator semantics\n");
        return 1;
    }

    for (idx = 0; idx < 6; ++idx) {
        if (cfloat_real(a[idx]) != expected_a_out[idx]) {
            fprintf(stderr, "[FAIL] CUNGBR Fortran->CBLAS thunk did not copy row-major complex generator output back correctly\n");
            return 1;
        }
    }
    for (idx = 0; idx < 2; ++idx) {
        if (cfloat_real(tau[idx]) != expected_tau[idx]) {
            fprintf(stderr, "[FAIL] CUNGBR Fortran->CBLAS thunk did not preserve complex TAU input\n");
            return 1;
        }
    }

    printf("[PASS] CUNGBR Fortran->CBLAS thunk translates row-major complex generator matrices and preserves TAU input\n");
    return 0;
}

static int check_cungbr_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_cungbr_fortran_slot_fn thunk = NULL;
    char vect = 'P';
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
    memset(&g_cungbr_cblas_call, 0, sizeof(g_cungbr_cblas_call));
    memset(a, 0, sizeof(a));
    memset(work, 0, sizeof(work));

    vtable.ext_ops[FB_OP_CUNGBR][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_cungbr_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CUNGBR);

    thunk = (fb_cungbr_fortran_slot_fn)vtable.ext_ops[FB_OP_CUNGBR][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CUNGBR CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&vect, &m, &n, &k, a, &lda, tau, work, &lwork, &info);
    if (info != 83 || g_cungbr_cblas_call.called != 1 ||
        g_cungbr_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_cungbr_cblas_call.vect != 'P' ||
        g_cungbr_cblas_call.m != 2 || g_cungbr_cblas_call.n != 3 ||
        g_cungbr_cblas_call.k != 2 || g_cungbr_cblas_call.lda != 2 ||
        g_cungbr_cblas_call.a != a || g_cungbr_cblas_call.tau != tau) {
        fprintf(stderr, "[FAIL] CUNGBR CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] CUNGBR CBLAS->Fortran thunk maps vect and all-pointer ABI into the complex C generator entry\n");
    return 0;
}

static int check_sormbr_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_sormbr_fn thunk = NULL;
    float a[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    float tau[2] = { 11.0f, 12.0f };
    float c[6] = { 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f };
    float expected_a_snapshot[4] = { 1.0f, 3.0f, 2.0f, 4.0f };
    float expected_c_snapshot[6] = { 5.0f, 8.0f, 6.0f, 9.0f, 7.0f, 10.0f };
    float expected_tau[2] = { 11.0f, 12.0f };
    float expected_c_out[6] = { 700.0f, 701.0f, 702.0f, 710.0f, 711.0f, 712.0f };
    float expected_a_unchanged[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sormbr_fortran_call, 0, sizeof(g_sormbr_fortran_call));

    vtable.ext_ops[FB_OP_SORMBR][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_sormbr_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SORMBR);

    thunk = (fb_sormbr_fn)vtable.ext_ops[FB_OP_SORMBR][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SORMBR Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 'Q', 'L', 'N', 2, 3, 2, a, 2, tau, c, 3);
    if (info != 0 || g_sormbr_fortran_call.query_calls != 1 ||
        g_sormbr_fortran_call.solve_calls != 1 ||
        g_sormbr_fortran_call.vect != 'Q' ||
        g_sormbr_fortran_call.side != 'L' ||
        g_sormbr_fortran_call.trans != 'N' ||
        g_sormbr_fortran_call.m != 2 || g_sormbr_fortran_call.n != 3 ||
        g_sormbr_fortran_call.k != 2 || g_sormbr_fortran_call.lda != 2 ||
        g_sormbr_fortran_call.ldc != 2 ||
        g_sormbr_fortran_call.lwork_query != -1 ||
        g_sormbr_fortran_call.lwork_solve != 7 ||
        memcmp(g_sormbr_fortran_call.a_snapshot, expected_a_snapshot,
               sizeof(expected_a_snapshot)) != 0 ||
        memcmp(g_sormbr_fortran_call.c_snapshot, expected_c_snapshot,
               sizeof(expected_c_snapshot)) != 0 ||
        memcmp(g_sormbr_fortran_call.tau_snapshot, expected_tau,
               sizeof(expected_tau)) != 0 ||
        memcmp(c, expected_c_out, sizeof(expected_c_out)) != 0 ||
        memcmp(a, expected_a_unchanged, sizeof(expected_a_unchanged)) != 0 ||
        memcmp(tau, expected_tau, sizeof(expected_tau)) != 0) {
        fprintf(stderr, "[FAIL] SORMBR Fortran->CBLAS thunk did not preserve vect-aware reflector-application semantics\n");
        return 1;
    }

    printf("[PASS] SORMBR Fortran->CBLAS thunk translates row-major bidiagonal reflector application and preserves A/TAU input\n");
    return 0;
}

static int check_sormbr_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_sormbr_fortran_slot_fn thunk = NULL;
    char vect = 'Q';
    char side = 'L';
    char trans = 'N';
    float a[4] = { 0.0f };
    float tau[2] = { 11.0f, 12.0f };
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
    memset(&g_sormbr_cblas_call, 0, sizeof(g_sormbr_cblas_call));

    vtable.ext_ops[FB_OP_SORMBR][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_sormbr_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SORMBR);

    thunk = (fb_sormbr_fortran_slot_fn)vtable.ext_ops[FB_OP_SORMBR][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SORMBR CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&vect, &side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work,
          &lwork, &info);
    if (info != 85 || g_sormbr_cblas_call.called != 1 ||
        g_sormbr_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_sormbr_cblas_call.vect != 'Q' ||
        g_sormbr_cblas_call.side != 'L' ||
        g_sormbr_cblas_call.trans != 'N' ||
        g_sormbr_cblas_call.m != 2 || g_sormbr_cblas_call.n != 3 ||
        g_sormbr_cblas_call.k != 2 || g_sormbr_cblas_call.lda != 2 ||
        g_sormbr_cblas_call.ldc != 2 ||
        g_sormbr_cblas_call.a != a || g_sormbr_cblas_call.tau != tau ||
        g_sormbr_cblas_call.c != c) {
        fprintf(stderr, "[FAIL] SORMBR CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] SORMBR CBLAS->Fortran thunk maps vect/side/trans chars into the C reflector-application entry\n");
    return 0;
}

static int check_cunmbr_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_cunmbr_fn thunk = NULL;
    fb_complex_float_t a[6] = {
        make_cfloat(1.0f), make_cfloat(2.0f), make_cfloat(3.0f),
        make_cfloat(4.0f), make_cfloat(5.0f), make_cfloat(6.0f)
    };
    fb_complex_float_t tau[2] = { make_cfloat(13.0f), make_cfloat(14.0f) };
    fb_complex_float_t c[6] = {
        make_cfloat(7.0f), make_cfloat(8.0f), make_cfloat(9.0f),
        make_cfloat(10.0f), make_cfloat(11.0f), make_cfloat(12.0f)
    };
    float expected_a_snapshot[6] = { 1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f };
    float expected_c_snapshot[6] = { 7.0f, 10.0f, 8.0f, 11.0f, 9.0f, 12.0f };
    float expected_tau[2] = { 13.0f, 14.0f };
    float expected_c_out[6] = { 800.0f, 801.0f, 802.0f, 810.0f, 811.0f, 812.0f };
    int info = 0;
    int idx = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cunmbr_fortran_call, 0, sizeof(g_cunmbr_fortran_call));

    vtable.ext_ops[FB_OP_CUNMBR][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_cunmbr_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CUNMBR);

    thunk = (fb_cunmbr_fn)vtable.ext_ops[FB_OP_CUNMBR][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CUNMBR Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 'P', 'R', 'C', 2, 3, 2, a, 3, tau, c, 3);
    if (info != 0 || g_cunmbr_fortran_call.query_calls != 1 ||
        g_cunmbr_fortran_call.solve_calls != 1 ||
        g_cunmbr_fortran_call.vect != 'P' ||
        g_cunmbr_fortran_call.side != 'R' ||
        g_cunmbr_fortran_call.trans != 'C' ||
        g_cunmbr_fortran_call.m != 2 || g_cunmbr_fortran_call.n != 3 ||
        g_cunmbr_fortran_call.k != 2 || g_cunmbr_fortran_call.lda != 2 ||
        g_cunmbr_fortran_call.ldc != 2 ||
        g_cunmbr_fortran_call.lwork_query != -1 ||
        g_cunmbr_fortran_call.lwork_solve != 8 ||
        memcmp(g_cunmbr_fortran_call.a_real_snapshot, expected_a_snapshot,
               sizeof(expected_a_snapshot)) != 0 ||
        memcmp(g_cunmbr_fortran_call.c_real_snapshot, expected_c_snapshot,
               sizeof(expected_c_snapshot)) != 0 ||
        memcmp(g_cunmbr_fortran_call.tau_real_snapshot, expected_tau,
               sizeof(expected_tau)) != 0) {
        fprintf(stderr, "[FAIL] CUNMBR Fortran->CBLAS thunk did not preserve complex vect-aware reflector-application semantics\n");
        return 1;
    }

    for (idx = 0; idx < 6; ++idx) {
        if (cfloat_real(c[idx]) != expected_c_out[idx]) {
            fprintf(stderr, "[FAIL] CUNMBR Fortran->CBLAS thunk did not copy row-major complex output back correctly\n");
            return 1;
        }
    }
    for (idx = 0; idx < 2; ++idx) {
        if (cfloat_real(tau[idx]) != expected_tau[idx]) {
            fprintf(stderr, "[FAIL] CUNMBR Fortran->CBLAS thunk did not preserve complex TAU input\n");
            return 1;
        }
    }

    printf("[PASS] CUNMBR Fortran->CBLAS thunk translates row-major complex bidiagonal reflector application with vect/side/trans preservation\n");
    return 0;
}

static int check_cunmbr_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_cunmbr_fortran_slot_fn thunk = NULL;
    char vect = 'P';
    char side = 'R';
    char trans = 'C';
    fb_complex_float_t a[6];
    fb_complex_float_t tau[2] = { make_cfloat(13.0f), make_cfloat(14.0f) };
    fb_complex_float_t c[6];
    fb_complex_float_t work[4];
    int m = 2;
    int n = 3;
    int k = 2;
    int lda = 2;
    int ldc = 2;
    int lwork = 4;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cunmbr_cblas_call, 0, sizeof(g_cunmbr_cblas_call));
    memset(a, 0, sizeof(a));
    memset(c, 0, sizeof(c));
    memset(work, 0, sizeof(work));

    vtable.ext_ops[FB_OP_CUNMBR][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_cunmbr_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CUNMBR);

    thunk = (fb_cunmbr_fortran_slot_fn)vtable.ext_ops[FB_OP_CUNMBR][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CUNMBR CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&vect, &side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work,
          &lwork, &info);
    if (info != 87 || g_cunmbr_cblas_call.called != 1 ||
        g_cunmbr_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_cunmbr_cblas_call.vect != 'P' ||
        g_cunmbr_cblas_call.side != 'R' ||
        g_cunmbr_cblas_call.trans != 'C' ||
        g_cunmbr_cblas_call.m != 2 || g_cunmbr_cblas_call.n != 3 ||
        g_cunmbr_cblas_call.k != 2 || g_cunmbr_cblas_call.lda != 2 ||
        g_cunmbr_cblas_call.ldc != 2 ||
        g_cunmbr_cblas_call.a != a || g_cunmbr_cblas_call.tau != tau ||
        g_cunmbr_cblas_call.c != c) {
        fprintf(stderr, "[FAIL] CUNMBR CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] CUNMBR CBLAS->Fortran thunk maps vect/side/trans chars into the complex C reflector-application entry\n");
    return 0;
}

static int check_dorgbr_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_dorgbr_fn thunk = NULL;
    double a[6] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    double tau[2] = { 7.0, 8.0 };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dorgbr_fortran_call, 0, sizeof(g_dorgbr_fortran_call));

    vtable.ext_ops[FB_OP_DORGBR][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_dorgbr_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_DORGBR);

    thunk = (fb_dorgbr_fn)vtable.ext_ops[FB_OP_DORGBR][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DORGBR Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 'Q', 3, 2, 2, a, 2, tau);
    if (info != 0 || g_dorgbr_fortran_call.query_calls != 1 ||
        g_dorgbr_fortran_call.solve_calls != 1 ||
        g_dorgbr_fortran_call.vect != 'Q' || a[0] != 901.0 || tau[0] != 7.0) {
        fprintf(stderr, "[FAIL] DORGBR Fortran->CBLAS thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] DORGBR Fortran->CBLAS thunk performs query+solve path and preserves TAU input\n");
    return 0;
}

static int check_dorgbr_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_dorgbr_fortran_slot_fn thunk = NULL;
    char vect = 'P';
    double a[6] = { 0.0 };
    double tau[2] = { 9.0, 10.0 };
    double work[4] = { 0.0 };
    int m = 2;
    int n = 3;
    int k = 2;
    int lda = 2;
    int lwork = 4;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dorgbr_cblas_call, 0, sizeof(g_dorgbr_cblas_call));

    vtable.ext_ops[FB_OP_DORGBR][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_dorgbr_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_DORGBR);

    thunk = (fb_dorgbr_fortran_slot_fn)vtable.ext_ops[FB_OP_DORGBR][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DORGBR CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&vect, &m, &n, &k, a, &lda, tau, work, &lwork, &info);
    if (info != 82 || g_dorgbr_cblas_call.called != 1 ||
        g_dorgbr_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_dorgbr_cblas_call.vect != 'P') {
        fprintf(stderr, "[FAIL] DORGBR CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] DORGBR CBLAS->Fortran thunk maps all-pointer ABI into the double C generator entry\n");
    return 0;
}

static int check_zungbr_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_zungbr_fn thunk = NULL;
    fb_complex_double_t a[6] = { 0 };
    fb_complex_double_t tau[2] = { make_cdouble(9.0), make_cdouble(10.0) };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zungbr_fortran_call, 0, sizeof(g_zungbr_fortran_call));

    vtable.ext_ops[FB_OP_ZUNGBR][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_zungbr_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_ZUNGBR);

    thunk = (fb_zungbr_fn)vtable.ext_ops[FB_OP_ZUNGBR][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZUNGBR Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 'Q', 2, 3, 2, a, 3, tau);
    if (info != 0 || g_zungbr_fortran_call.query_calls != 1 ||
        g_zungbr_fortran_call.solve_calls != 1 ||
        g_zungbr_fortran_call.vect != 'Q' || cdouble_real(a[0]) != 1001.0 ||
        cdouble_real(tau[0]) != 9.0) {
        fprintf(stderr, "[FAIL] ZUNGBR Fortran->CBLAS thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] ZUNGBR Fortran->CBLAS thunk performs query+solve path and preserves complex-double TAU input\n");
    return 0;
}

static int check_zungbr_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_zungbr_fortran_slot_fn thunk = NULL;
    char vect = 'P';
    fb_complex_double_t a[6] = { 0 };
    fb_complex_double_t tau[2] = { make_cdouble(11.0), make_cdouble(12.0) };
    fb_complex_double_t work[4] = { 0 };
    int m = 2;
    int n = 3;
    int k = 2;
    int lda = 2;
    int lwork = 4;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zungbr_cblas_call, 0, sizeof(g_zungbr_cblas_call));

    vtable.ext_ops[FB_OP_ZUNGBR][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_zungbr_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_ZUNGBR);

    thunk = (fb_zungbr_fortran_slot_fn)vtable.ext_ops[FB_OP_ZUNGBR][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZUNGBR CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&vect, &m, &n, &k, a, &lda, tau, work, &lwork, &info);
    if (info != 84 || g_zungbr_cblas_call.called != 1 ||
        g_zungbr_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_zungbr_cblas_call.vect != 'P') {
        fprintf(stderr, "[FAIL] ZUNGBR CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] ZUNGBR CBLAS->Fortran thunk maps all-pointer ABI into the complex-double C generator entry\n");
    return 0;
}

static int check_dormbr_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_dormbr_fn thunk = NULL;
    double a[4] = { 1.0, 2.0, 3.0, 4.0 };
    double tau[2] = { 11.0, 12.0 };
    double c[6] = { 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dormbr_fortran_call, 0, sizeof(g_dormbr_fortran_call));

    vtable.ext_ops[FB_OP_DORMBR][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_dormbr_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_DORMBR);

    thunk = (fb_dormbr_fn)vtable.ext_ops[FB_OP_DORMBR][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DORMBR Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 'Q', 'L', 'N', 2, 3, 2, a, 2, tau, c, 3);
    if (info != 0 || g_dormbr_fortran_call.query_calls != 1 ||
        g_dormbr_fortran_call.solve_calls != 1 ||
        g_dormbr_fortran_call.vect != 'Q' || g_dormbr_fortran_call.side != 'L' ||
        g_dormbr_fortran_call.trans != 'N' || c[0] != 1101.0 ||
        a[0] != 1.0 || tau[0] != 11.0) {
        fprintf(stderr, "[FAIL] DORMBR Fortran->CBLAS thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] DORMBR Fortran->CBLAS thunk performs query+solve path and preserves A/TAU inputs\n");
    return 0;
}

static int check_dormbr_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_dormbr_fortran_slot_fn thunk = NULL;
    char vect = 'Q';
    char side = 'L';
    char trans = 'N';
    double a[4] = { 0.0 };
    double tau[2] = { 11.0, 12.0 };
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
    memset(&g_dormbr_cblas_call, 0, sizeof(g_dormbr_cblas_call));

    vtable.ext_ops[FB_OP_DORMBR][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_dormbr_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_DORMBR);

    thunk = (fb_dormbr_fortran_slot_fn)vtable.ext_ops[FB_OP_DORMBR][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DORMBR CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&vect, &side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work,
          &lwork, &info);
    if (info != 86 || g_dormbr_cblas_call.called != 1 ||
        g_dormbr_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_dormbr_cblas_call.vect != 'Q' || g_dormbr_cblas_call.side != 'L' ||
        g_dormbr_cblas_call.trans != 'N') {
        fprintf(stderr, "[FAIL] DORMBR CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] DORMBR CBLAS->Fortran thunk maps all-pointer ABI into the double C reflector entry\n");
    return 0;
}

static int check_zunmbr_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_zunmbr_fn thunk = NULL;
    fb_complex_double_t a[6] = { 0 };
    fb_complex_double_t tau[2] = { make_cdouble(13.0), make_cdouble(14.0) };
    fb_complex_double_t c[6] = { 0 };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zunmbr_fortran_call, 0, sizeof(g_zunmbr_fortran_call));

    vtable.ext_ops[FB_OP_ZUNMBR][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_zunmbr_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_ZUNMBR);

    thunk = (fb_zunmbr_fn)vtable.ext_ops[FB_OP_ZUNMBR][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZUNMBR Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 'P', 'R', 'C', 2, 3, 2, a, 3, tau, c, 3);
    if (info != 0 || g_zunmbr_fortran_call.query_calls != 1 ||
        g_zunmbr_fortran_call.solve_calls != 1 ||
        g_zunmbr_fortran_call.vect != 'P' || g_zunmbr_fortran_call.side != 'R' ||
        g_zunmbr_fortran_call.trans != 'C' || cdouble_real(c[0]) != 1201.0 ||
        cdouble_real(a[0]) != 0.0 || cdouble_real(tau[0]) != 13.0) {
        fprintf(stderr, "[FAIL] ZUNMBR Fortran->CBLAS thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] ZUNMBR Fortran->CBLAS thunk performs query+solve path and preserves complex-double A/TAU inputs\n");
    return 0;
}

static int check_zunmbr_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_zunmbr_fortran_slot_fn thunk = NULL;
    char vect = 'P';
    char side = 'R';
    char trans = 'C';
    fb_complex_double_t a[6] = { 0 };
    fb_complex_double_t tau[2] = { make_cdouble(13.0), make_cdouble(14.0) };
    fb_complex_double_t c[6] = { 0 };
    fb_complex_double_t work[4] = { 0 };
    int m = 2;
    int n = 3;
    int k = 2;
    int lda = 2;
    int ldc = 2;
    int lwork = 4;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zunmbr_cblas_call, 0, sizeof(g_zunmbr_cblas_call));

    vtable.ext_ops[FB_OP_ZUNMBR][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_zunmbr_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_ZUNMBR);

    thunk = (fb_zunmbr_fortran_slot_fn)vtable.ext_ops[FB_OP_ZUNMBR][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZUNMBR CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&vect, &side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work,
          &lwork, &info);
    if (info != 88 || g_zunmbr_cblas_call.called != 1 ||
        g_zunmbr_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_zunmbr_cblas_call.vect != 'P' || g_zunmbr_cblas_call.side != 'R' ||
        g_zunmbr_cblas_call.trans != 'C') {
        fprintf(stderr, "[FAIL] ZUNMBR CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] ZUNMBR CBLAS->Fortran thunk maps all-pointer ABI into the complex-double C reflector entry\n");
    return 0;
}

int main(void)
{
    if (check_sorgbr_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_sorgbr_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_dorgbr_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_dorgbr_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_cungbr_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_cungbr_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_zungbr_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_zungbr_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_sormbr_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_sormbr_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_dormbr_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_dormbr_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_cunmbr_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_cunmbr_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_zunmbr_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_zunmbr_cblas_to_fortran() != 0) {
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}