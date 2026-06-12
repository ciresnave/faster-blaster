#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef int (*fb_sorg2l_fn)(fb_layout_t layout, int m, int n, int k,
                            float *a, int lda, const float *tau);
typedef int (*fb_dorg2l_fn)(fb_layout_t layout, int m, int n, int k,
                            double *a, int lda, const double *tau);
typedef int (*fb_cung2l_fn)(fb_layout_t layout, int m, int n, int k,
                            fb_complex_float_t *a, int lda,
                            const fb_complex_float_t *tau);
typedef int (*fb_zung2l_fn)(fb_layout_t layout, int m, int n, int k,
                            fb_complex_double_t *a, int lda,
                            const fb_complex_double_t *tau);

typedef void (*fb_sorg2l_fortran_slot_fn)(int *m, int *n, int *k, float *a,
                                          int *lda, float *tau, float *work,
                                          int *info);
typedef void (*fb_dorg2l_fortran_slot_fn)(int *m, int *n, int *k, double *a,
                                          int *lda, double *tau,
                                          double *work, int *info);
typedef void (*fb_cung2l_fortran_slot_fn)(int *m, int *n, int *k,
                                          fb_complex_float_t *a, int *lda,
                                          fb_complex_float_t *tau,
                                          fb_complex_float_t *work,
                                          int *info);
typedef void (*fb_zung2l_fortran_slot_fn)(int *m, int *n, int *k,
                                          fb_complex_double_t *a, int *lda,
                                          fb_complex_double_t *tau,
                                          fb_complex_double_t *work,
                                          int *info);

static struct {
    int calls;
    int m;
    int n;
    int k;
    int lda;
    float a_snapshot[6];
    float tau_snapshot[2];
} g_sorg2l_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    int m;
    int n;
    int k;
    int lda;
    float *a;
    const float *tau;
} g_sorg2l_cblas_call;

static struct {
    int calls;
    int m;
    int n;
    int k;
    int lda;
    double a_snapshot[6];
    double tau_snapshot[2];
} g_dorg2l_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    int m;
    int n;
    int k;
    int lda;
    double *a;
    const double *tau;
} g_dorg2l_cblas_call;

static struct {
    int calls;
    int m;
    int n;
    int k;
    int lda;
    float a_real_snapshot[6];
    float tau_real_snapshot[2];
} g_cung2l_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    int m;
    int n;
    int k;
    int lda;
    fb_complex_float_t *a;
    const fb_complex_float_t *tau;
} g_cung2l_cblas_call;

static struct {
    int calls;
    int m;
    int n;
    int k;
    int lda;
    double a_real_snapshot[6];
    double tau_real_snapshot[2];
} g_zung2l_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    int m;
    int n;
    int k;
    int lda;
    fb_complex_double_t *a;
    const fb_complex_double_t *tau;
} g_zung2l_cblas_call;

static struct {
    int calls;
    int m;
    int n;
    int k;
    int lda;
    float a_snapshot[8];
    float tau_snapshot[2];
} g_sorgl2_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    int m;
    int n;
    int k;
    int lda;
    float *a;
    const float *tau;
} g_sorgl2_cblas_call;

static struct {
    int calls;
    int m;
    int n;
    int k;
    int lda;
    double a_snapshot[8];
    double tau_snapshot[2];
} g_dorgl2_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    int m;
    int n;
    int k;
    int lda;
    double *a;
    const double *tau;
} g_dorgl2_cblas_call;

static struct {
    int calls;
    int m;
    int n;
    int k;
    int lda;
    float a_real_snapshot[8];
    float tau_real_snapshot[2];
} g_cungl2_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    int m;
    int n;
    int k;
    int lda;
    fb_complex_float_t *a;
    const fb_complex_float_t *tau;
} g_cungl2_cblas_call;

static struct {
    int calls;
    int m;
    int n;
    int k;
    int lda;
    double a_real_snapshot[8];
    double tau_real_snapshot[2];
} g_zungl2_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    int m;
    int n;
    int k;
    int lda;
    fb_complex_double_t *a;
    const fb_complex_double_t *tau;
} g_zungl2_cblas_call;

static struct {
    int calls;
    int m;
    int n;
    int k;
    int lda;
    float a_snapshot[8];
    float tau_snapshot[2];
} g_sorgr2_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    int m;
    int n;
    int k;
    int lda;
    float *a;
    const float *tau;
} g_sorgr2_cblas_call;

static struct {
    int calls;
    int m;
    int n;
    int k;
    int lda;
    double a_snapshot[8];
    double tau_snapshot[2];
} g_dorgr2_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    int m;
    int n;
    int k;
    int lda;
    double *a;
    const double *tau;
} g_dorgr2_cblas_call;

static struct {
    int calls;
    int m;
    int n;
    int k;
    int lda;
    float a_real_snapshot[8];
    float tau_real_snapshot[2];
} g_cungr2_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    int m;
    int n;
    int k;
    int lda;
    fb_complex_float_t *a;
    const fb_complex_float_t *tau;
} g_cungr2_cblas_call;

static struct {
    int calls;
    int m;
    int n;
    int k;
    int lda;
    double a_real_snapshot[8];
    double tau_real_snapshot[2];
} g_zungr2_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    int m;
    int n;
    int k;
    int lda;
    fb_complex_double_t *a;
    const fb_complex_double_t *tau;
} g_zungr2_cblas_call;

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

static void stub_sorg2l_fortran(int *m, int *n, int *k, float *a, int *lda,
                                float *tau, float *work, int *info)
{
    int row = 0;
    int col = 0;

    g_sorg2l_fortran_call.calls += 1;
    g_sorg2l_fortran_call.m = *m;
    g_sorg2l_fortran_call.n = *n;
    g_sorg2l_fortran_call.k = *k;
    g_sorg2l_fortran_call.lda = *lda;
    for (col = 0; col < *n; ++col) {
        for (row = 0; row < *m; ++row) {
            g_sorg2l_fortran_call.a_snapshot[(col * (*m)) + row] =
                a[(col * (*lda)) + row];
            a[(col * (*lda)) + row] = (float)(1500 + (10 * row) + col);
        }
    }
    for (row = 0; row < *k; ++row) {
        g_sorg2l_fortran_call.tau_snapshot[row] = tau[row];
    }
    if (work) {
        work[0] = 9.0f;
    }
    *info = 0;
}

static int stub_sorg2l_cblas(fb_layout_t layout, int m, int n, int k, float *a,
                             int lda, const float *tau)
{
    g_sorg2l_cblas_call.called += 1;
    g_sorg2l_cblas_call.layout = layout;
    g_sorg2l_cblas_call.m = m;
    g_sorg2l_cblas_call.n = n;
    g_sorg2l_cblas_call.k = k;
    g_sorg2l_cblas_call.lda = lda;
    g_sorg2l_cblas_call.a = a;
    g_sorg2l_cblas_call.tau = tau;
    return 107;
}

static void stub_dorg2l_fortran(int *m, int *n, int *k, double *a, int *lda,
                                double *tau, double *work, int *info)
{
    int row = 0;
    int col = 0;

    g_dorg2l_fortran_call.calls += 1;
    g_dorg2l_fortran_call.m = *m;
    g_dorg2l_fortran_call.n = *n;
    g_dorg2l_fortran_call.k = *k;
    g_dorg2l_fortran_call.lda = *lda;
    for (col = 0; col < *n; ++col) {
        for (row = 0; row < *m; ++row) {
            g_dorg2l_fortran_call.a_snapshot[(col * (*m)) + row] =
                a[(col * (*lda)) + row];
            a[(col * (*lda)) + row] = (double)(1550 + (10 * row) + col);
        }
    }
    for (row = 0; row < *k; ++row) {
        g_dorg2l_fortran_call.tau_snapshot[row] = tau[row];
    }
    if (work) {
        work[0] = 9.5;
    }
    *info = 0;
}

static int stub_dorg2l_cblas(fb_layout_t layout, int m, int n, int k, double *a,
                             int lda, const double *tau)
{
    g_dorg2l_cblas_call.called += 1;
    g_dorg2l_cblas_call.layout = layout;
    g_dorg2l_cblas_call.m = m;
    g_dorg2l_cblas_call.n = n;
    g_dorg2l_cblas_call.k = k;
    g_dorg2l_cblas_call.lda = lda;
    g_dorg2l_cblas_call.a = a;
    g_dorg2l_cblas_call.tau = tau;
    return 108;
}

static void stub_cung2l_fortran(int *m, int *n, int *k, fb_complex_float_t *a,
                                int *lda, fb_complex_float_t *tau,
                                fb_complex_float_t *work, int *info)
{
    int row = 0;
    int col = 0;

    g_cung2l_fortran_call.calls += 1;
    g_cung2l_fortran_call.m = *m;
    g_cung2l_fortran_call.n = *n;
    g_cung2l_fortran_call.k = *k;
    g_cung2l_fortran_call.lda = *lda;
    for (col = 0; col < *n; ++col) {
        for (row = 0; row < *m; ++row) {
            g_cung2l_fortran_call.a_real_snapshot[(col * (*m)) + row] =
                cfloat_real(a[(col * (*lda)) + row]);
            a[(col * (*lda)) + row] = make_cfloat((float)(1600 + (10 * row) + col));
        }
    }
    for (row = 0; row < *k; ++row) {
        g_cung2l_fortran_call.tau_real_snapshot[row] = cfloat_real(tau[row]);
    }
    if (work) {
        work[0] = make_cfloat(10.0f);
    }
    *info = 0;
}

static int stub_cung2l_cblas(fb_layout_t layout, int m, int n, int k,
                             fb_complex_float_t *a, int lda,
                             const fb_complex_float_t *tau)
{
    g_cung2l_cblas_call.called += 1;
    g_cung2l_cblas_call.layout = layout;
    g_cung2l_cblas_call.m = m;
    g_cung2l_cblas_call.n = n;
    g_cung2l_cblas_call.k = k;
    g_cung2l_cblas_call.lda = lda;
    g_cung2l_cblas_call.a = a;
    g_cung2l_cblas_call.tau = tau;
    return 109;
}

static void stub_zung2l_fortran(int *m, int *n, int *k, fb_complex_double_t *a,
                                int *lda, fb_complex_double_t *tau,
                                fb_complex_double_t *work, int *info)
{
    int row = 0;
    int col = 0;

    g_zung2l_fortran_call.calls += 1;
    g_zung2l_fortran_call.m = *m;
    g_zung2l_fortran_call.n = *n;
    g_zung2l_fortran_call.k = *k;
    g_zung2l_fortran_call.lda = *lda;
    for (col = 0; col < *n; ++col) {
        for (row = 0; row < *m; ++row) {
            g_zung2l_fortran_call.a_real_snapshot[(col * (*m)) + row] =
                cdouble_real(a[(col * (*lda)) + row]);
            a[(col * (*lda)) + row] = make_cdouble((double)(1650 + (10 * row) + col));
        }
    }
    for (row = 0; row < *k; ++row) {
        g_zung2l_fortran_call.tau_real_snapshot[row] = cdouble_real(tau[row]);
    }
    if (work) {
        work[0] = make_cdouble(10.5);
    }
    *info = 0;
}

static int stub_zung2l_cblas(fb_layout_t layout, int m, int n, int k,
                             fb_complex_double_t *a, int lda,
                             const fb_complex_double_t *tau)
{
    g_zung2l_cblas_call.called += 1;
    g_zung2l_cblas_call.layout = layout;
    g_zung2l_cblas_call.m = m;
    g_zung2l_cblas_call.n = n;
    g_zung2l_cblas_call.k = k;
    g_zung2l_cblas_call.lda = lda;
    g_zung2l_cblas_call.a = a;
    g_zung2l_cblas_call.tau = tau;
    return 110;
}

static void stub_sorgl2_fortran(int *m, int *n, int *k, float *a, int *lda,
                                float *tau, float *work, int *info)
{
    int row = 0;
    int col = 0;

    g_sorgl2_fortran_call.calls += 1;
    g_sorgl2_fortran_call.m = *m;
    g_sorgl2_fortran_call.n = *n;
    g_sorgl2_fortran_call.k = *k;
    g_sorgl2_fortran_call.lda = *lda;
    for (col = 0; col < *n; ++col) {
        for (row = 0; row < *m; ++row) {
            g_sorgl2_fortran_call.a_snapshot[(col * (*m)) + row] =
                a[(col * (*lda)) + row];
            a[(col * (*lda)) + row] = (float)(1700 + (10 * row) + col);
        }
    }
    for (row = 0; row < *k; ++row) {
        g_sorgl2_fortran_call.tau_snapshot[row] = tau[row];
    }
    if (work) {
        work[0] = 11.0f;
    }
    *info = 0;
}

static int stub_sorgl2_cblas(fb_layout_t layout, int m, int n, int k, float *a,
                             int lda, const float *tau)
{
    g_sorgl2_cblas_call.called += 1;
    g_sorgl2_cblas_call.layout = layout;
    g_sorgl2_cblas_call.m = m;
    g_sorgl2_cblas_call.n = n;
    g_sorgl2_cblas_call.k = k;
    g_sorgl2_cblas_call.lda = lda;
    g_sorgl2_cblas_call.a = a;
    g_sorgl2_cblas_call.tau = tau;
    return 111;
}

static void stub_dorgl2_fortran(int *m, int *n, int *k, double *a, int *lda,
                                double *tau, double *work, int *info)
{
    int row = 0;
    int col = 0;

    g_dorgl2_fortran_call.calls += 1;
    g_dorgl2_fortran_call.m = *m;
    g_dorgl2_fortran_call.n = *n;
    g_dorgl2_fortran_call.k = *k;
    g_dorgl2_fortran_call.lda = *lda;
    for (col = 0; col < *n; ++col) {
        for (row = 0; row < *m; ++row) {
            g_dorgl2_fortran_call.a_snapshot[(col * (*m)) + row] =
                a[(col * (*lda)) + row];
            a[(col * (*lda)) + row] = (double)(1750 + (10 * row) + col);
        }
    }
    for (row = 0; row < *k; ++row) {
        g_dorgl2_fortran_call.tau_snapshot[row] = tau[row];
    }
    if (work) {
        work[0] = 11.5;
    }
    *info = 0;
}

static int stub_dorgl2_cblas(fb_layout_t layout, int m, int n, int k, double *a,
                             int lda, const double *tau)
{
    g_dorgl2_cblas_call.called += 1;
    g_dorgl2_cblas_call.layout = layout;
    g_dorgl2_cblas_call.m = m;
    g_dorgl2_cblas_call.n = n;
    g_dorgl2_cblas_call.k = k;
    g_dorgl2_cblas_call.lda = lda;
    g_dorgl2_cblas_call.a = a;
    g_dorgl2_cblas_call.tau = tau;
    return 112;
}

static void stub_cungl2_fortran(int *m, int *n, int *k, fb_complex_float_t *a,
                                int *lda, fb_complex_float_t *tau,
                                fb_complex_float_t *work, int *info)
{
    int row = 0;
    int col = 0;

    g_cungl2_fortran_call.calls += 1;
    g_cungl2_fortran_call.m = *m;
    g_cungl2_fortran_call.n = *n;
    g_cungl2_fortran_call.k = *k;
    g_cungl2_fortran_call.lda = *lda;
    for (col = 0; col < *n; ++col) {
        for (row = 0; row < *m; ++row) {
            g_cungl2_fortran_call.a_real_snapshot[(col * (*m)) + row] =
                cfloat_real(a[(col * (*lda)) + row]);
            a[(col * (*lda)) + row] = make_cfloat((float)(1800 + (10 * row) + col));
        }
    }
    for (row = 0; row < *k; ++row) {
        g_cungl2_fortran_call.tau_real_snapshot[row] = cfloat_real(tau[row]);
    }
    if (work) {
        work[0] = make_cfloat(12.0f);
    }
    *info = 0;
}

static int stub_cungl2_cblas(fb_layout_t layout, int m, int n, int k,
                             fb_complex_float_t *a, int lda,
                             const fb_complex_float_t *tau)
{
    g_cungl2_cblas_call.called += 1;
    g_cungl2_cblas_call.layout = layout;
    g_cungl2_cblas_call.m = m;
    g_cungl2_cblas_call.n = n;
    g_cungl2_cblas_call.k = k;
    g_cungl2_cblas_call.lda = lda;
    g_cungl2_cblas_call.a = a;
    g_cungl2_cblas_call.tau = tau;
    return 113;
}

static void stub_zungl2_fortran(int *m, int *n, int *k, fb_complex_double_t *a,
                                int *lda, fb_complex_double_t *tau,
                                fb_complex_double_t *work, int *info)
{
    int row = 0;
    int col = 0;

    g_zungl2_fortran_call.calls += 1;
    g_zungl2_fortran_call.m = *m;
    g_zungl2_fortran_call.n = *n;
    g_zungl2_fortran_call.k = *k;
    g_zungl2_fortran_call.lda = *lda;
    for (col = 0; col < *n; ++col) {
        for (row = 0; row < *m; ++row) {
            g_zungl2_fortran_call.a_real_snapshot[(col * (*m)) + row] =
                cdouble_real(a[(col * (*lda)) + row]);
            a[(col * (*lda)) + row] = make_cdouble((double)(1850 + (10 * row) + col));
        }
    }
    for (row = 0; row < *k; ++row) {
        g_zungl2_fortran_call.tau_real_snapshot[row] = cdouble_real(tau[row]);
    }
    if (work) {
        work[0] = make_cdouble(12.5);
    }
    *info = 0;
}

static int stub_zungl2_cblas(fb_layout_t layout, int m, int n, int k,
                             fb_complex_double_t *a, int lda,
                             const fb_complex_double_t *tau)
{
    g_zungl2_cblas_call.called += 1;
    g_zungl2_cblas_call.layout = layout;
    g_zungl2_cblas_call.m = m;
    g_zungl2_cblas_call.n = n;
    g_zungl2_cblas_call.k = k;
    g_zungl2_cblas_call.lda = lda;
    g_zungl2_cblas_call.a = a;
    g_zungl2_cblas_call.tau = tau;
    return 114;
}

static void stub_sorgr2_fortran(int *m, int *n, int *k, float *a, int *lda,
                                float *tau, float *work, int *info)
{
    int row = 0;
    int col = 0;

    g_sorgr2_fortran_call.calls += 1;
    g_sorgr2_fortran_call.m = *m;
    g_sorgr2_fortran_call.n = *n;
    g_sorgr2_fortran_call.k = *k;
    g_sorgr2_fortran_call.lda = *lda;
    for (col = 0; col < *n; ++col) {
        for (row = 0; row < *m; ++row) {
            g_sorgr2_fortran_call.a_snapshot[(col * (*m)) + row] =
                a[(col * (*lda)) + row];
            a[(col * (*lda)) + row] = (float)(1900 + (10 * row) + col);
        }
    }
    for (row = 0; row < *k; ++row) {
        g_sorgr2_fortran_call.tau_snapshot[row] = tau[row];
    }
    if (work) {
        work[0] = 13.0f;
    }
    *info = 0;
}

static int stub_sorgr2_cblas(fb_layout_t layout, int m, int n, int k, float *a,
                             int lda, const float *tau)
{
    g_sorgr2_cblas_call.called += 1;
    g_sorgr2_cblas_call.layout = layout;
    g_sorgr2_cblas_call.m = m;
    g_sorgr2_cblas_call.n = n;
    g_sorgr2_cblas_call.k = k;
    g_sorgr2_cblas_call.lda = lda;
    g_sorgr2_cblas_call.a = a;
    g_sorgr2_cblas_call.tau = tau;
    return 115;
}

static void stub_dorgr2_fortran(int *m, int *n, int *k, double *a, int *lda,
                                double *tau, double *work, int *info)
{
    int row = 0;
    int col = 0;

    g_dorgr2_fortran_call.calls += 1;
    g_dorgr2_fortran_call.m = *m;
    g_dorgr2_fortran_call.n = *n;
    g_dorgr2_fortran_call.k = *k;
    g_dorgr2_fortran_call.lda = *lda;
    for (col = 0; col < *n; ++col) {
        for (row = 0; row < *m; ++row) {
            g_dorgr2_fortran_call.a_snapshot[(col * (*m)) + row] =
                a[(col * (*lda)) + row];
            a[(col * (*lda)) + row] = (double)(1950 + (10 * row) + col);
        }
    }
    for (row = 0; row < *k; ++row) {
        g_dorgr2_fortran_call.tau_snapshot[row] = tau[row];
    }
    if (work) {
        work[0] = 13.5;
    }
    *info = 0;
}

static int stub_dorgr2_cblas(fb_layout_t layout, int m, int n, int k, double *a,
                             int lda, const double *tau)
{
    g_dorgr2_cblas_call.called += 1;
    g_dorgr2_cblas_call.layout = layout;
    g_dorgr2_cblas_call.m = m;
    g_dorgr2_cblas_call.n = n;
    g_dorgr2_cblas_call.k = k;
    g_dorgr2_cblas_call.lda = lda;
    g_dorgr2_cblas_call.a = a;
    g_dorgr2_cblas_call.tau = tau;
    return 116;
}

static void stub_cungr2_fortran(int *m, int *n, int *k, fb_complex_float_t *a,
                                int *lda, fb_complex_float_t *tau,
                                fb_complex_float_t *work, int *info)
{
    int row = 0;
    int col = 0;

    g_cungr2_fortran_call.calls += 1;
    g_cungr2_fortran_call.m = *m;
    g_cungr2_fortran_call.n = *n;
    g_cungr2_fortran_call.k = *k;
    g_cungr2_fortran_call.lda = *lda;
    for (col = 0; col < *n; ++col) {
        for (row = 0; row < *m; ++row) {
            g_cungr2_fortran_call.a_real_snapshot[(col * (*m)) + row] =
                cfloat_real(a[(col * (*lda)) + row]);
            a[(col * (*lda)) + row] = make_cfloat((float)(2000 + (10 * row) + col));
        }
    }
    for (row = 0; row < *k; ++row) {
        g_cungr2_fortran_call.tau_real_snapshot[row] = cfloat_real(tau[row]);
    }
    if (work) {
        work[0] = make_cfloat(14.0f);
    }
    *info = 0;
}

static int stub_cungr2_cblas(fb_layout_t layout, int m, int n, int k,
                             fb_complex_float_t *a, int lda,
                             const fb_complex_float_t *tau)
{
    g_cungr2_cblas_call.called += 1;
    g_cungr2_cblas_call.layout = layout;
    g_cungr2_cblas_call.m = m;
    g_cungr2_cblas_call.n = n;
    g_cungr2_cblas_call.k = k;
    g_cungr2_cblas_call.lda = lda;
    g_cungr2_cblas_call.a = a;
    g_cungr2_cblas_call.tau = tau;
    return 117;
}

static void stub_zungr2_fortran(int *m, int *n, int *k, fb_complex_double_t *a,
                                int *lda, fb_complex_double_t *tau,
                                fb_complex_double_t *work, int *info)
{
    int row = 0;
    int col = 0;

    g_zungr2_fortran_call.calls += 1;
    g_zungr2_fortran_call.m = *m;
    g_zungr2_fortran_call.n = *n;
    g_zungr2_fortran_call.k = *k;
    g_zungr2_fortran_call.lda = *lda;
    for (col = 0; col < *n; ++col) {
        for (row = 0; row < *m; ++row) {
            g_zungr2_fortran_call.a_real_snapshot[(col * (*m)) + row] =
                cdouble_real(a[(col * (*lda)) + row]);
            a[(col * (*lda)) + row] = make_cdouble((double)(2050 + (10 * row) + col));
        }
    }
    for (row = 0; row < *k; ++row) {
        g_zungr2_fortran_call.tau_real_snapshot[row] = cdouble_real(tau[row]);
    }
    if (work) {
        work[0] = make_cdouble(14.5);
    }
    *info = 0;
}

static int stub_zungr2_cblas(fb_layout_t layout, int m, int n, int k,
                             fb_complex_double_t *a, int lda,
                             const fb_complex_double_t *tau)
{
    g_zungr2_cblas_call.called += 1;
    g_zungr2_cblas_call.layout = layout;
    g_zungr2_cblas_call.m = m;
    g_zungr2_cblas_call.n = n;
    g_zungr2_cblas_call.k = k;
    g_zungr2_cblas_call.lda = lda;
    g_zungr2_cblas_call.a = a;
    g_zungr2_cblas_call.tau = tau;
    return 118;
}

static int check_sorg2l_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_sorg2l_fn thunk = NULL;
    float a[6] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
    float tau[2] = { 80.0f, 81.0f };
    float expected_a_snapshot[6] = { 1.0f, 3.0f, 5.0f, 2.0f, 4.0f, 6.0f };
    float expected_tau[2] = { 80.0f, 81.0f };
    float expected_a_out[6] = { 1500.0f, 1501.0f, 1510.0f, 1511.0f, 1520.0f, 1521.0f };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sorg2l_fortran_call, 0, sizeof(g_sorg2l_fortran_call));

    vtable.ext_ops[FB_OP_SORG2L][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_sorg2l_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SORG2L);

    thunk = (fb_sorg2l_fn)vtable.ext_ops[FB_OP_SORG2L][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SORG2L Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 3, 2, 2, a, 2, tau);
    if (info != 0 || g_sorg2l_fortran_call.calls != 1 ||
        g_sorg2l_fortran_call.m != 3 || g_sorg2l_fortran_call.n != 2 ||
        g_sorg2l_fortran_call.k != 2 || g_sorg2l_fortran_call.lda != 3 ||
        memcmp(g_sorg2l_fortran_call.a_snapshot, expected_a_snapshot,
               sizeof(expected_a_snapshot)) != 0 ||
        memcmp(g_sorg2l_fortran_call.tau_snapshot, expected_tau,
               sizeof(expected_tau)) != 0 ||
        memcmp(a, expected_a_out, sizeof(expected_a_out)) != 0 ||
        memcmp(tau, expected_tau, sizeof(expected_tau)) != 0) {
        fprintf(stderr, "[FAIL] SORG2L Fortran->CBLAS thunk did not preserve unblocked row-major generator semantics\n");
        return 1;
    }

    printf("[PASS] SORG2L Fortran->CBLAS thunk translates row-major unblocked generator matrices and preserves TAU input\n");
    return 0;
}

static int check_sorg2l_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_sorg2l_fortran_slot_fn thunk = NULL;
    float a[6] = { 0.0f };
    float tau[2] = { 80.0f, 81.0f };
    float work[3] = { 0.0f };
    int m = 3;
    int n = 2;
    int k = 2;
    int lda = 3;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sorg2l_cblas_call, 0, sizeof(g_sorg2l_cblas_call));

    vtable.ext_ops[FB_OP_SORG2L][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_sorg2l_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SORG2L);

    thunk = (fb_sorg2l_fortran_slot_fn)vtable.ext_ops[FB_OP_SORG2L][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SORG2L CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&m, &n, &k, a, &lda, tau, work, &info);
    if (info != 107 || g_sorg2l_cblas_call.called != 1 ||
        g_sorg2l_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_sorg2l_cblas_call.m != 3 || g_sorg2l_cblas_call.n != 2 ||
        g_sorg2l_cblas_call.k != 2 || g_sorg2l_cblas_call.lda != 3 ||
        g_sorg2l_cblas_call.a != a || g_sorg2l_cblas_call.tau != tau) {
        fprintf(stderr, "[FAIL] SORG2L CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] SORG2L CBLAS->Fortran thunk maps the all-pointer ABI into the generic C unblocked generator entry\n");
    return 0;
}

static int check_cung2l_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_cung2l_fn thunk = NULL;
    fb_complex_float_t a[6] = {
        make_cfloat(1.0f), make_cfloat(2.0f), make_cfloat(3.0f),
        make_cfloat(4.0f), make_cfloat(5.0f), make_cfloat(6.0f)
    };
    fb_complex_float_t tau[2] = { make_cfloat(90.0f), make_cfloat(91.0f) };
    float expected_a_snapshot[6] = { 1.0f, 3.0f, 5.0f, 2.0f, 4.0f, 6.0f };
    float expected_tau[2] = { 90.0f, 91.0f };
    float expected_a_out[6] = { 1600.0f, 1601.0f, 1610.0f, 1611.0f, 1620.0f, 1621.0f };
    int info = 0;
    int idx = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cung2l_fortran_call, 0, sizeof(g_cung2l_fortran_call));

    vtable.ext_ops[FB_OP_CUNG2L][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_cung2l_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CUNG2L);

    thunk = (fb_cung2l_fn)vtable.ext_ops[FB_OP_CUNG2L][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CUNG2L Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 3, 2, 2, a, 2, tau);
    if (info != 0 || g_cung2l_fortran_call.calls != 1 ||
        g_cung2l_fortran_call.m != 3 || g_cung2l_fortran_call.n != 2 ||
        g_cung2l_fortran_call.k != 2 || g_cung2l_fortran_call.lda != 3 ||
        memcmp(g_cung2l_fortran_call.a_real_snapshot, expected_a_snapshot,
               sizeof(expected_a_snapshot)) != 0 ||
        memcmp(g_cung2l_fortran_call.tau_real_snapshot, expected_tau,
               sizeof(expected_tau)) != 0) {
        fprintf(stderr, "[FAIL] CUNG2L Fortran->CBLAS thunk did not preserve complex unblocked row-major generator semantics\n");
        return 1;
    }

    for (idx = 0; idx < 6; ++idx) {
        if (cfloat_real(a[idx]) != expected_a_out[idx]) {
            fprintf(stderr, "[FAIL] CUNG2L Fortran->CBLAS thunk did not copy complex unblocked generator output back correctly\n");
            return 1;
        }
    }
    for (idx = 0; idx < 2; ++idx) {
        if (cfloat_real(tau[idx]) != expected_tau[idx]) {
            fprintf(stderr, "[FAIL] CUNG2L Fortran->CBLAS thunk did not preserve complex TAU input\n");
            return 1;
        }
    }

    printf("[PASS] CUNG2L Fortran->CBLAS thunk translates row-major complex unblocked generator matrices and preserves TAU input\n");
    return 0;
}

static int check_cung2l_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_cung2l_fortran_slot_fn thunk = NULL;
    fb_complex_float_t a[6];
    fb_complex_float_t tau[2] = { make_cfloat(90.0f), make_cfloat(91.0f) };
    fb_complex_float_t work[3];
    int m = 3;
    int n = 2;
    int k = 2;
    int lda = 3;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cung2l_cblas_call, 0, sizeof(g_cung2l_cblas_call));
    memset(a, 0, sizeof(a));
    memset(work, 0, sizeof(work));

    vtable.ext_ops[FB_OP_CUNG2L][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_cung2l_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CUNG2L);

    thunk = (fb_cung2l_fortran_slot_fn)vtable.ext_ops[FB_OP_CUNG2L][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CUNG2L CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&m, &n, &k, a, &lda, tau, work, &info);
    if (info != 109 || g_cung2l_cblas_call.called != 1 ||
        g_cung2l_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_cung2l_cblas_call.m != 3 || g_cung2l_cblas_call.n != 2 ||
        g_cung2l_cblas_call.k != 2 || g_cung2l_cblas_call.lda != 3 ||
        g_cung2l_cblas_call.a != a || g_cung2l_cblas_call.tau != tau) {
        fprintf(stderr, "[FAIL] CUNG2L CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] CUNG2L CBLAS->Fortran thunk maps the all-pointer ABI into the generic complex unblocked generator entry\n");
    return 0;
}

static int check_dorg2l_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_dorg2l_fn thunk = NULL;
    double a[6] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    double tau[2] = { 82.0, 83.0 };
    double expected_a_snapshot[6] = { 1.0, 3.0, 5.0, 2.0, 4.0, 6.0 };
    double expected_tau[2] = { 82.0, 83.0 };
    double expected_a_out[6] = { 1550.0, 1551.0, 1560.0, 1561.0, 1570.0, 1571.0 };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dorg2l_fortran_call, 0, sizeof(g_dorg2l_fortran_call));

    vtable.ext_ops[FB_OP_DORG2L][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_dorg2l_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_DORG2L);

    thunk = (fb_dorg2l_fn)vtable.ext_ops[FB_OP_DORG2L][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DORG2L Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 3, 2, 2, a, 2, tau);
    if (info != 0 || g_dorg2l_fortran_call.calls != 1 ||
        g_dorg2l_fortran_call.m != 3 || g_dorg2l_fortran_call.n != 2 ||
        g_dorg2l_fortran_call.k != 2 || g_dorg2l_fortran_call.lda != 3 ||
        memcmp(g_dorg2l_fortran_call.a_snapshot, expected_a_snapshot,
               sizeof(expected_a_snapshot)) != 0 ||
        memcmp(g_dorg2l_fortran_call.tau_snapshot, expected_tau,
               sizeof(expected_tau)) != 0 ||
        memcmp(a, expected_a_out, sizeof(expected_a_out)) != 0 ||
        memcmp(tau, expected_tau, sizeof(expected_tau)) != 0) {
        fprintf(stderr, "[FAIL] DORG2L Fortran->CBLAS thunk did not preserve double unblocked row-major generator semantics\n");
        return 1;
    }

    printf("[PASS] DORG2L Fortran->CBLAS thunk translates row-major double unblocked generator matrices and preserves TAU input\n");
    return 0;
}

static int check_dorg2l_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_dorg2l_fortran_slot_fn thunk = NULL;
    double a[6] = { 0.0 };
    double tau[2] = { 82.0, 83.0 };
    double work[3] = { 0.0 };
    int m = 3;
    int n = 2;
    int k = 2;
    int lda = 3;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dorg2l_cblas_call, 0, sizeof(g_dorg2l_cblas_call));

    vtable.ext_ops[FB_OP_DORG2L][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_dorg2l_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_DORG2L);

    thunk = (fb_dorg2l_fortran_slot_fn)vtable.ext_ops[FB_OP_DORG2L][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DORG2L CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&m, &n, &k, a, &lda, tau, work, &info);
    if (info != 108 || g_dorg2l_cblas_call.called != 1 ||
        g_dorg2l_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_dorg2l_cblas_call.m != 3 || g_dorg2l_cblas_call.n != 2 ||
        g_dorg2l_cblas_call.k != 2 || g_dorg2l_cblas_call.lda != 3 ||
        g_dorg2l_cblas_call.a != a || g_dorg2l_cblas_call.tau != tau) {
        fprintf(stderr, "[FAIL] DORG2L CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] DORG2L CBLAS->Fortran thunk maps the all-pointer ABI into the generic double unblocked generator entry\n");
    return 0;
}

static int check_zung2l_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_zung2l_fn thunk = NULL;
    fb_complex_double_t a[6] = {
        make_cdouble(1.0), make_cdouble(2.0), make_cdouble(3.0),
        make_cdouble(4.0), make_cdouble(5.0), make_cdouble(6.0)
    };
    fb_complex_double_t tau[2] = { make_cdouble(92.0), make_cdouble(93.0) };
    double expected_a_snapshot[6] = { 1.0, 3.0, 5.0, 2.0, 4.0, 6.0 };
    double expected_tau[2] = { 92.0, 93.0 };
    double expected_a_out[6] = { 1650.0, 1651.0, 1660.0, 1661.0, 1670.0, 1671.0 };
    int info = 0;
    int idx = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zung2l_fortran_call, 0, sizeof(g_zung2l_fortran_call));

    vtable.ext_ops[FB_OP_ZUNG2L][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_zung2l_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_ZUNG2L);

    thunk = (fb_zung2l_fn)vtable.ext_ops[FB_OP_ZUNG2L][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZUNG2L Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 3, 2, 2, a, 2, tau);
    if (info != 0 || g_zung2l_fortran_call.calls != 1 ||
        g_zung2l_fortran_call.m != 3 || g_zung2l_fortran_call.n != 2 ||
        g_zung2l_fortran_call.k != 2 || g_zung2l_fortran_call.lda != 3 ||
        memcmp(g_zung2l_fortran_call.a_real_snapshot, expected_a_snapshot,
               sizeof(expected_a_snapshot)) != 0 ||
        memcmp(g_zung2l_fortran_call.tau_real_snapshot, expected_tau,
               sizeof(expected_tau)) != 0) {
        fprintf(stderr, "[FAIL] ZUNG2L Fortran->CBLAS thunk did not preserve complex-double unblocked row-major generator semantics\n");
        return 1;
    }

    for (idx = 0; idx < 6; ++idx) {
        if (cdouble_real(a[idx]) != expected_a_out[idx]) {
            fprintf(stderr, "[FAIL] ZUNG2L Fortran->CBLAS thunk did not copy complex-double unblocked generator output back correctly\n");
            return 1;
        }
    }
    for (idx = 0; idx < 2; ++idx) {
        if (cdouble_real(tau[idx]) != expected_tau[idx]) {
            fprintf(stderr, "[FAIL] ZUNG2L Fortran->CBLAS thunk did not preserve complex-double TAU input\n");
            return 1;
        }
    }

    printf("[PASS] ZUNG2L Fortran->CBLAS thunk translates row-major complex-double unblocked generator matrices and preserves TAU input\n");
    return 0;
}

static int check_zung2l_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_zung2l_fortran_slot_fn thunk = NULL;
    fb_complex_double_t a[6];
    fb_complex_double_t tau[2] = { make_cdouble(92.0), make_cdouble(93.0) };
    fb_complex_double_t work[3];
    int m = 3;
    int n = 2;
    int k = 2;
    int lda = 3;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zung2l_cblas_call, 0, sizeof(g_zung2l_cblas_call));
    memset(a, 0, sizeof(a));
    memset(work, 0, sizeof(work));

    vtable.ext_ops[FB_OP_ZUNG2L][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_zung2l_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_ZUNG2L);

    thunk = (fb_zung2l_fortran_slot_fn)vtable.ext_ops[FB_OP_ZUNG2L][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZUNG2L CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&m, &n, &k, a, &lda, tau, work, &info);
    if (info != 110 || g_zung2l_cblas_call.called != 1 ||
        g_zung2l_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_zung2l_cblas_call.m != 3 || g_zung2l_cblas_call.n != 2 ||
        g_zung2l_cblas_call.k != 2 || g_zung2l_cblas_call.lda != 3 ||
        g_zung2l_cblas_call.a != a || g_zung2l_cblas_call.tau != tau) {
        fprintf(stderr, "[FAIL] ZUNG2L CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] ZUNG2L CBLAS->Fortran thunk maps the all-pointer ABI into the generic complex-double unblocked generator entry\n");
    return 0;
}

static int check_sorgl2_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_sorg2l_fn thunk = NULL;
    float a[8] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };
    float tau[2] = { 82.0f, 83.0f };
    float expected_a_snapshot[8] = { 1.0f, 5.0f, 2.0f, 6.0f, 3.0f, 7.0f, 4.0f, 8.0f };
    float expected_tau[2] = { 82.0f, 83.0f };
    float expected_a_out[8] = { 1700.0f, 1701.0f, 1702.0f, 1703.0f, 1710.0f, 1711.0f, 1712.0f, 1713.0f };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sorgl2_fortran_call, 0, sizeof(g_sorgl2_fortran_call));

    vtable.ext_ops[FB_OP_SORGL2][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_sorgl2_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SORGL2);

    thunk = (fb_sorg2l_fn)vtable.ext_ops[FB_OP_SORGL2][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SORGL2 Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 2, 4, 2, a, 4, tau);
    if (info != 0 || g_sorgl2_fortran_call.calls != 1 ||
        g_sorgl2_fortran_call.m != 2 || g_sorgl2_fortran_call.n != 4 ||
        g_sorgl2_fortran_call.k != 2 || g_sorgl2_fortran_call.lda != 2 ||
        memcmp(g_sorgl2_fortran_call.a_snapshot, expected_a_snapshot,
               sizeof(expected_a_snapshot)) != 0 ||
        memcmp(g_sorgl2_fortran_call.tau_snapshot, expected_tau,
               sizeof(expected_tau)) != 0 ||
        memcmp(a, expected_a_out, sizeof(expected_a_out)) != 0 ||
        memcmp(tau, expected_tau, sizeof(expected_tau)) != 0) {
        fprintf(stderr, "[FAIL] SORGL2 Fortran->CBLAS thunk did not preserve row-major unblocked LQ-generator semantics\n");
        return 1;
    }

    printf("[PASS] SORGL2 Fortran->CBLAS thunk translates row-major unblocked LQ-generator matrices and preserves TAU input\n");
    return 0;
}

static int check_sorgl2_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_sorg2l_fortran_slot_fn thunk = NULL;
    float a[8] = { 0.0f };
    float tau[2] = { 82.0f, 83.0f };
    float work[2] = { 0.0f };
    int m = 2;
    int n = 4;
    int k = 2;
    int lda = 2;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sorgl2_cblas_call, 0, sizeof(g_sorgl2_cblas_call));

    vtable.ext_ops[FB_OP_SORGL2][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_sorgl2_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SORGL2);

    thunk = (fb_sorg2l_fortran_slot_fn)vtable.ext_ops[FB_OP_SORGL2][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SORGL2 CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&m, &n, &k, a, &lda, tau, work, &info);
    if (info != 111 || g_sorgl2_cblas_call.called != 1 ||
        g_sorgl2_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_sorgl2_cblas_call.m != 2 || g_sorgl2_cblas_call.n != 4 ||
        g_sorgl2_cblas_call.k != 2 || g_sorgl2_cblas_call.lda != 2 ||
        g_sorgl2_cblas_call.a != a || g_sorgl2_cblas_call.tau != tau) {
        fprintf(stderr, "[FAIL] SORGL2 CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] SORGL2 CBLAS->Fortran thunk maps the all-pointer ABI into the generic C unblocked LQ-generator entry\n");
    return 0;
}

static int check_cungl2_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_cung2l_fn thunk = NULL;
    fb_complex_float_t a[8] = {
        make_cfloat(1.0f), make_cfloat(2.0f), make_cfloat(3.0f), make_cfloat(4.0f),
        make_cfloat(5.0f), make_cfloat(6.0f), make_cfloat(7.0f), make_cfloat(8.0f)
    };
    fb_complex_float_t tau[2] = { make_cfloat(92.0f), make_cfloat(93.0f) };
    float expected_a_snapshot[8] = { 1.0f, 5.0f, 2.0f, 6.0f, 3.0f, 7.0f, 4.0f, 8.0f };
    float expected_tau[2] = { 92.0f, 93.0f };
    float expected_a_out[8] = { 1800.0f, 1801.0f, 1802.0f, 1803.0f, 1810.0f, 1811.0f, 1812.0f, 1813.0f };
    int info = 0;
    int idx = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cungl2_fortran_call, 0, sizeof(g_cungl2_fortran_call));

    vtable.ext_ops[FB_OP_CUNGL2][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_cungl2_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CUNGL2);

    thunk = (fb_cung2l_fn)vtable.ext_ops[FB_OP_CUNGL2][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CUNGL2 Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 2, 4, 2, a, 4, tau);
    if (info != 0 || g_cungl2_fortran_call.calls != 1 ||
        g_cungl2_fortran_call.m != 2 || g_cungl2_fortran_call.n != 4 ||
        g_cungl2_fortran_call.k != 2 || g_cungl2_fortran_call.lda != 2 ||
        memcmp(g_cungl2_fortran_call.a_real_snapshot, expected_a_snapshot,
               sizeof(expected_a_snapshot)) != 0 ||
        memcmp(g_cungl2_fortran_call.tau_real_snapshot, expected_tau,
               sizeof(expected_tau)) != 0) {
        fprintf(stderr, "[FAIL] CUNGL2 Fortran->CBLAS thunk did not preserve complex row-major unblocked LQ-generator semantics\n");
        return 1;
    }

    for (idx = 0; idx < 8; ++idx) {
        if (cfloat_real(a[idx]) != expected_a_out[idx]) {
            fprintf(stderr, "[FAIL] CUNGL2 Fortran->CBLAS thunk did not copy complex unblocked LQ-generator output back correctly\n");
            return 1;
        }
    }
    for (idx = 0; idx < 2; ++idx) {
        if (cfloat_real(tau[idx]) != expected_tau[idx]) {
            fprintf(stderr, "[FAIL] CUNGL2 Fortran->CBLAS thunk did not preserve complex TAU input\n");
            return 1;
        }
    }

    printf("[PASS] CUNGL2 Fortran->CBLAS thunk translates row-major complex unblocked LQ-generator matrices and preserves TAU input\n");
    return 0;
}

static int check_cungl2_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_cung2l_fortran_slot_fn thunk = NULL;
    fb_complex_float_t a[8];
    fb_complex_float_t tau[2] = { make_cfloat(92.0f), make_cfloat(93.0f) };
    fb_complex_float_t work[2];
    int m = 2;
    int n = 4;
    int k = 2;
    int lda = 2;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cungl2_cblas_call, 0, sizeof(g_cungl2_cblas_call));
    memset(a, 0, sizeof(a));
    memset(work, 0, sizeof(work));

    vtable.ext_ops[FB_OP_CUNGL2][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_cungl2_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CUNGL2);

    thunk = (fb_cung2l_fortran_slot_fn)vtable.ext_ops[FB_OP_CUNGL2][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CUNGL2 CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&m, &n, &k, a, &lda, tau, work, &info);
    if (info != 113 || g_cungl2_cblas_call.called != 1 ||
        g_cungl2_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_cungl2_cblas_call.m != 2 || g_cungl2_cblas_call.n != 4 ||
        g_cungl2_cblas_call.k != 2 || g_cungl2_cblas_call.lda != 2 ||
        g_cungl2_cblas_call.a != a || g_cungl2_cblas_call.tau != tau) {
        fprintf(stderr, "[FAIL] CUNGL2 CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] CUNGL2 CBLAS->Fortran thunk maps the all-pointer ABI into the generic complex unblocked LQ-generator entry\n");
    return 0;
}

static int check_dorgl2_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_dorg2l_fn thunk = NULL;
    double a[8] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    double tau[2] = { 84.0, 85.0 };
    double expected_a_snapshot[8] = { 1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0 };
    double expected_tau[2] = { 84.0, 85.0 };
    double expected_a_out[8] = { 1750.0, 1751.0, 1752.0, 1753.0, 1760.0, 1761.0, 1762.0, 1763.0 };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dorgl2_fortran_call, 0, sizeof(g_dorgl2_fortran_call));

    vtable.ext_ops[FB_OP_DORGL2][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_dorgl2_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_DORGL2);

    thunk = (fb_dorg2l_fn)vtable.ext_ops[FB_OP_DORGL2][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DORGL2 Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 2, 4, 2, a, 4, tau);
    if (info != 0 || g_dorgl2_fortran_call.calls != 1 ||
        g_dorgl2_fortran_call.m != 2 || g_dorgl2_fortran_call.n != 4 ||
        g_dorgl2_fortran_call.k != 2 || g_dorgl2_fortran_call.lda != 2 ||
        memcmp(g_dorgl2_fortran_call.a_snapshot, expected_a_snapshot,
               sizeof(expected_a_snapshot)) != 0 ||
        memcmp(g_dorgl2_fortran_call.tau_snapshot, expected_tau,
               sizeof(expected_tau)) != 0 ||
        memcmp(a, expected_a_out, sizeof(expected_a_out)) != 0 ||
        memcmp(tau, expected_tau, sizeof(expected_tau)) != 0) {
        fprintf(stderr, "[FAIL] DORGL2 Fortran->CBLAS thunk did not preserve double row-major unblocked LQ-generator semantics\n");
        return 1;
    }

    printf("[PASS] DORGL2 Fortran->CBLAS thunk translates row-major double unblocked LQ-generator matrices and preserves TAU input\n");
    return 0;
}

static int check_dorgl2_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_dorg2l_fortran_slot_fn thunk = NULL;
    double a[8] = { 0.0 };
    double tau[2] = { 84.0, 85.0 };
    double work[2] = { 0.0 };
    int m = 2;
    int n = 4;
    int k = 2;
    int lda = 2;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dorgl2_cblas_call, 0, sizeof(g_dorgl2_cblas_call));

    vtable.ext_ops[FB_OP_DORGL2][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_dorgl2_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_DORGL2);

    thunk = (fb_dorg2l_fortran_slot_fn)vtable.ext_ops[FB_OP_DORGL2][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DORGL2 CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&m, &n, &k, a, &lda, tau, work, &info);
    if (info != 112 || g_dorgl2_cblas_call.called != 1 ||
        g_dorgl2_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_dorgl2_cblas_call.m != 2 || g_dorgl2_cblas_call.n != 4 ||
        g_dorgl2_cblas_call.k != 2 || g_dorgl2_cblas_call.lda != 2 ||
        g_dorgl2_cblas_call.a != a || g_dorgl2_cblas_call.tau != tau) {
        fprintf(stderr, "[FAIL] DORGL2 CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] DORGL2 CBLAS->Fortran thunk maps the all-pointer ABI into the generic double unblocked LQ-generator entry\n");
    return 0;
}

static int check_zungl2_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_zung2l_fn thunk = NULL;
    fb_complex_double_t a[8] = {
        make_cdouble(1.0), make_cdouble(2.0), make_cdouble(3.0), make_cdouble(4.0),
        make_cdouble(5.0), make_cdouble(6.0), make_cdouble(7.0), make_cdouble(8.0)
    };
    fb_complex_double_t tau[2] = { make_cdouble(94.0), make_cdouble(95.0) };
    double expected_a_snapshot[8] = { 1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0 };
    double expected_tau[2] = { 94.0, 95.0 };
    double expected_a_out[8] = { 1850.0, 1851.0, 1852.0, 1853.0, 1860.0, 1861.0, 1862.0, 1863.0 };
    int info = 0;
    int idx = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zungl2_fortran_call, 0, sizeof(g_zungl2_fortran_call));

    vtable.ext_ops[FB_OP_ZUNGL2][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_zungl2_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_ZUNGL2);

    thunk = (fb_zung2l_fn)vtable.ext_ops[FB_OP_ZUNGL2][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZUNGL2 Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 2, 4, 2, a, 4, tau);
    if (info != 0 || g_zungl2_fortran_call.calls != 1 ||
        g_zungl2_fortran_call.m != 2 || g_zungl2_fortran_call.n != 4 ||
        g_zungl2_fortran_call.k != 2 || g_zungl2_fortran_call.lda != 2 ||
        memcmp(g_zungl2_fortran_call.a_real_snapshot, expected_a_snapshot,
               sizeof(expected_a_snapshot)) != 0 ||
        memcmp(g_zungl2_fortran_call.tau_real_snapshot, expected_tau,
               sizeof(expected_tau)) != 0) {
        fprintf(stderr, "[FAIL] ZUNGL2 Fortran->CBLAS thunk did not preserve complex-double row-major unblocked LQ-generator semantics\n");
        return 1;
    }

    for (idx = 0; idx < 8; ++idx) {
        if (cdouble_real(a[idx]) != expected_a_out[idx]) {
            fprintf(stderr, "[FAIL] ZUNGL2 Fortran->CBLAS thunk did not copy complex-double unblocked LQ-generator output back correctly\n");
            return 1;
        }
    }
    for (idx = 0; idx < 2; ++idx) {
        if (cdouble_real(tau[idx]) != expected_tau[idx]) {
            fprintf(stderr, "[FAIL] ZUNGL2 Fortran->CBLAS thunk did not preserve complex-double TAU input\n");
            return 1;
        }
    }

    printf("[PASS] ZUNGL2 Fortran->CBLAS thunk translates row-major complex-double unblocked LQ-generator matrices and preserves TAU input\n");
    return 0;
}

static int check_zungl2_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_zung2l_fortran_slot_fn thunk = NULL;
    fb_complex_double_t a[8];
    fb_complex_double_t tau[2] = { make_cdouble(94.0), make_cdouble(95.0) };
    fb_complex_double_t work[2];
    int m = 2;
    int n = 4;
    int k = 2;
    int lda = 2;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zungl2_cblas_call, 0, sizeof(g_zungl2_cblas_call));
    memset(a, 0, sizeof(a));
    memset(work, 0, sizeof(work));

    vtable.ext_ops[FB_OP_ZUNGL2][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_zungl2_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_ZUNGL2);

    thunk = (fb_zung2l_fortran_slot_fn)vtable.ext_ops[FB_OP_ZUNGL2][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZUNGL2 CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&m, &n, &k, a, &lda, tau, work, &info);
    if (info != 114 || g_zungl2_cblas_call.called != 1 ||
        g_zungl2_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_zungl2_cblas_call.m != 2 || g_zungl2_cblas_call.n != 4 ||
        g_zungl2_cblas_call.k != 2 || g_zungl2_cblas_call.lda != 2 ||
        g_zungl2_cblas_call.a != a || g_zungl2_cblas_call.tau != tau) {
        fprintf(stderr, "[FAIL] ZUNGL2 CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] ZUNGL2 CBLAS->Fortran thunk maps the all-pointer ABI into the generic complex-double unblocked LQ-generator entry\n");
    return 0;
}

static int check_sorgr2_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_sorg2l_fn thunk = NULL;
    float a[8] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };
    float tau[2] = { 84.0f, 85.0f };
    float expected_a_snapshot[8] = { 1.0f, 5.0f, 2.0f, 6.0f, 3.0f, 7.0f, 4.0f, 8.0f };
    float expected_tau[2] = { 84.0f, 85.0f };
    float expected_a_out[8] = { 1900.0f, 1901.0f, 1902.0f, 1903.0f, 1910.0f, 1911.0f, 1912.0f, 1913.0f };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sorgr2_fortran_call, 0, sizeof(g_sorgr2_fortran_call));

    vtable.ext_ops[FB_OP_SORGR2][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_sorgr2_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SORGR2);

    thunk = (fb_sorg2l_fn)vtable.ext_ops[FB_OP_SORGR2][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SORGR2 Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 2, 4, 2, a, 4, tau);
    if (info != 0 || g_sorgr2_fortran_call.calls != 1 ||
        g_sorgr2_fortran_call.m != 2 || g_sorgr2_fortran_call.n != 4 ||
        g_sorgr2_fortran_call.k != 2 || g_sorgr2_fortran_call.lda != 2 ||
        memcmp(g_sorgr2_fortran_call.a_snapshot, expected_a_snapshot,
               sizeof(expected_a_snapshot)) != 0 ||
        memcmp(g_sorgr2_fortran_call.tau_snapshot, expected_tau,
               sizeof(expected_tau)) != 0 ||
        memcmp(a, expected_a_out, sizeof(expected_a_out)) != 0 ||
        memcmp(tau, expected_tau, sizeof(expected_tau)) != 0) {
        fprintf(stderr, "[FAIL] SORGR2 Fortran->CBLAS thunk did not preserve row-major unblocked RQ-generator semantics\n");
        return 1;
    }

    printf("[PASS] SORGR2 Fortran->CBLAS thunk translates row-major unblocked RQ-generator matrices and preserves TAU input\n");
    return 0;
}

static int check_sorgr2_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_sorg2l_fortran_slot_fn thunk = NULL;
    float a[8] = { 0.0f };
    float tau[2] = { 84.0f, 85.0f };
    float work[2] = { 0.0f };
    int m = 2;
    int n = 4;
    int k = 2;
    int lda = 2;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sorgr2_cblas_call, 0, sizeof(g_sorgr2_cblas_call));

    vtable.ext_ops[FB_OP_SORGR2][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_sorgr2_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SORGR2);

    thunk = (fb_sorg2l_fortran_slot_fn)vtable.ext_ops[FB_OP_SORGR2][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SORGR2 CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&m, &n, &k, a, &lda, tau, work, &info);
    if (info != 115 || g_sorgr2_cblas_call.called != 1 ||
        g_sorgr2_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_sorgr2_cblas_call.m != 2 || g_sorgr2_cblas_call.n != 4 ||
        g_sorgr2_cblas_call.k != 2 || g_sorgr2_cblas_call.lda != 2 ||
        g_sorgr2_cblas_call.a != a || g_sorgr2_cblas_call.tau != tau) {
        fprintf(stderr, "[FAIL] SORGR2 CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] SORGR2 CBLAS->Fortran thunk maps the all-pointer ABI into the generic C unblocked RQ-generator entry\n");
    return 0;
}

static int check_cungr2_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_cung2l_fn thunk = NULL;
    fb_complex_float_t a[8] = {
        make_cfloat(1.0f), make_cfloat(2.0f), make_cfloat(3.0f), make_cfloat(4.0f),
        make_cfloat(5.0f), make_cfloat(6.0f), make_cfloat(7.0f), make_cfloat(8.0f)
    };
    fb_complex_float_t tau[2] = { make_cfloat(94.0f), make_cfloat(95.0f) };
    float expected_a_snapshot[8] = { 1.0f, 5.0f, 2.0f, 6.0f, 3.0f, 7.0f, 4.0f, 8.0f };
    float expected_tau[2] = { 94.0f, 95.0f };
    float expected_a_out[8] = { 2000.0f, 2001.0f, 2002.0f, 2003.0f, 2010.0f, 2011.0f, 2012.0f, 2013.0f };
    int info = 0;
    int idx = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cungr2_fortran_call, 0, sizeof(g_cungr2_fortran_call));

    vtable.ext_ops[FB_OP_CUNGR2][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_cungr2_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CUNGR2);

    thunk = (fb_cung2l_fn)vtable.ext_ops[FB_OP_CUNGR2][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CUNGR2 Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 2, 4, 2, a, 4, tau);
    if (info != 0 || g_cungr2_fortran_call.calls != 1 ||
        g_cungr2_fortran_call.m != 2 || g_cungr2_fortran_call.n != 4 ||
        g_cungr2_fortran_call.k != 2 || g_cungr2_fortran_call.lda != 2 ||
        memcmp(g_cungr2_fortran_call.a_real_snapshot, expected_a_snapshot,
               sizeof(expected_a_snapshot)) != 0 ||
        memcmp(g_cungr2_fortran_call.tau_real_snapshot, expected_tau,
               sizeof(expected_tau)) != 0) {
        fprintf(stderr, "[FAIL] CUNGR2 Fortran->CBLAS thunk did not preserve complex row-major unblocked RQ-generator semantics\n");
        return 1;
    }

    for (idx = 0; idx < 8; ++idx) {
        if (cfloat_real(a[idx]) != expected_a_out[idx]) {
            fprintf(stderr, "[FAIL] CUNGR2 Fortran->CBLAS thunk did not copy complex unblocked RQ-generator output back correctly\n");
            return 1;
        }
    }
    for (idx = 0; idx < 2; ++idx) {
        if (cfloat_real(tau[idx]) != expected_tau[idx]) {
            fprintf(stderr, "[FAIL] CUNGR2 Fortran->CBLAS thunk did not preserve complex TAU input\n");
            return 1;
        }
    }

    printf("[PASS] CUNGR2 Fortran->CBLAS thunk translates row-major complex unblocked RQ-generator matrices and preserves TAU input\n");
    return 0;
}

static int check_cungr2_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_cung2l_fortran_slot_fn thunk = NULL;
    fb_complex_float_t a[8];
    fb_complex_float_t tau[2] = { make_cfloat(94.0f), make_cfloat(95.0f) };
    fb_complex_float_t work[2];
    int m = 2;
    int n = 4;
    int k = 2;
    int lda = 2;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cungr2_cblas_call, 0, sizeof(g_cungr2_cblas_call));
    memset(a, 0, sizeof(a));
    memset(work, 0, sizeof(work));

    vtable.ext_ops[FB_OP_CUNGR2][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_cungr2_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CUNGR2);

    thunk = (fb_cung2l_fortran_slot_fn)vtable.ext_ops[FB_OP_CUNGR2][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CUNGR2 CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&m, &n, &k, a, &lda, tau, work, &info);
    if (info != 117 || g_cungr2_cblas_call.called != 1 ||
        g_cungr2_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_cungr2_cblas_call.m != 2 || g_cungr2_cblas_call.n != 4 ||
        g_cungr2_cblas_call.k != 2 || g_cungr2_cblas_call.lda != 2 ||
        g_cungr2_cblas_call.a != a || g_cungr2_cblas_call.tau != tau) {
        fprintf(stderr, "[FAIL] CUNGR2 CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] CUNGR2 CBLAS->Fortran thunk maps the all-pointer ABI into the generic complex unblocked RQ-generator entry\n");
    return 0;
}

static int check_dorgr2_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_dorg2l_fn thunk = NULL;
    double a[8] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    double tau[2] = { 86.0, 87.0 };
    double expected_a_snapshot[8] = { 1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0 };
    double expected_tau[2] = { 86.0, 87.0 };
    double expected_a_out[8] = { 1950.0, 1951.0, 1952.0, 1953.0, 1960.0, 1961.0, 1962.0, 1963.0 };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dorgr2_fortran_call, 0, sizeof(g_dorgr2_fortran_call));

    vtable.ext_ops[FB_OP_DORGR2][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_dorgr2_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_DORGR2);

    thunk = (fb_dorg2l_fn)vtable.ext_ops[FB_OP_DORGR2][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DORGR2 Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 2, 4, 2, a, 4, tau);
    if (info != 0 || g_dorgr2_fortran_call.calls != 1 ||
        g_dorgr2_fortran_call.m != 2 || g_dorgr2_fortran_call.n != 4 ||
        g_dorgr2_fortran_call.k != 2 || g_dorgr2_fortran_call.lda != 2 ||
        memcmp(g_dorgr2_fortran_call.a_snapshot, expected_a_snapshot,
               sizeof(expected_a_snapshot)) != 0 ||
        memcmp(g_dorgr2_fortran_call.tau_snapshot, expected_tau,
               sizeof(expected_tau)) != 0 ||
        memcmp(a, expected_a_out, sizeof(expected_a_out)) != 0 ||
        memcmp(tau, expected_tau, sizeof(expected_tau)) != 0) {
        fprintf(stderr, "[FAIL] DORGR2 Fortran->CBLAS thunk did not preserve double row-major unblocked RQ-generator semantics\n");
        return 1;
    }

    printf("[PASS] DORGR2 Fortran->CBLAS thunk translates row-major double unblocked RQ-generator matrices and preserves TAU input\n");
    return 0;
}

static int check_dorgr2_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_dorg2l_fortran_slot_fn thunk = NULL;
    double a[8] = { 0.0 };
    double tau[2] = { 86.0, 87.0 };
    double work[2] = { 0.0 };
    int m = 2;
    int n = 4;
    int k = 2;
    int lda = 2;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dorgr2_cblas_call, 0, sizeof(g_dorgr2_cblas_call));

    vtable.ext_ops[FB_OP_DORGR2][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_dorgr2_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_DORGR2);

    thunk = (fb_dorg2l_fortran_slot_fn)vtable.ext_ops[FB_OP_DORGR2][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DORGR2 CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&m, &n, &k, a, &lda, tau, work, &info);
    if (info != 116 || g_dorgr2_cblas_call.called != 1 ||
        g_dorgr2_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_dorgr2_cblas_call.m != 2 || g_dorgr2_cblas_call.n != 4 ||
        g_dorgr2_cblas_call.k != 2 || g_dorgr2_cblas_call.lda != 2 ||
        g_dorgr2_cblas_call.a != a || g_dorgr2_cblas_call.tau != tau) {
        fprintf(stderr, "[FAIL] DORGR2 CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] DORGR2 CBLAS->Fortran thunk maps the all-pointer ABI into the generic double unblocked RQ-generator entry\n");
    return 0;
}

static int check_zungr2_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_zung2l_fn thunk = NULL;
    fb_complex_double_t a[8] = {
        make_cdouble(1.0), make_cdouble(2.0), make_cdouble(3.0), make_cdouble(4.0),
        make_cdouble(5.0), make_cdouble(6.0), make_cdouble(7.0), make_cdouble(8.0)
    };
    fb_complex_double_t tau[2] = { make_cdouble(96.0), make_cdouble(97.0) };
    double expected_a_snapshot[8] = { 1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0 };
    double expected_tau[2] = { 96.0, 97.0 };
    double expected_a_out[8] = { 2050.0, 2051.0, 2052.0, 2053.0, 2060.0, 2061.0, 2062.0, 2063.0 };
    int info = 0;
    int idx = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zungr2_fortran_call, 0, sizeof(g_zungr2_fortran_call));

    vtable.ext_ops[FB_OP_ZUNGR2][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_zungr2_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_ZUNGR2);

    thunk = (fb_zung2l_fn)vtable.ext_ops[FB_OP_ZUNGR2][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZUNGR2 Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 2, 4, 2, a, 4, tau);
    if (info != 0 || g_zungr2_fortran_call.calls != 1 ||
        g_zungr2_fortran_call.m != 2 || g_zungr2_fortran_call.n != 4 ||
        g_zungr2_fortran_call.k != 2 || g_zungr2_fortran_call.lda != 2 ||
        memcmp(g_zungr2_fortran_call.a_real_snapshot, expected_a_snapshot,
               sizeof(expected_a_snapshot)) != 0 ||
        memcmp(g_zungr2_fortran_call.tau_real_snapshot, expected_tau,
               sizeof(expected_tau)) != 0) {
        fprintf(stderr, "[FAIL] ZUNGR2 Fortran->CBLAS thunk did not preserve complex-double row-major unblocked RQ-generator semantics\n");
        return 1;
    }

    for (idx = 0; idx < 8; ++idx) {
        if (cdouble_real(a[idx]) != expected_a_out[idx]) {
            fprintf(stderr, "[FAIL] ZUNGR2 Fortran->CBLAS thunk did not copy complex-double unblocked RQ-generator output back correctly\n");
            return 1;
        }
    }
    for (idx = 0; idx < 2; ++idx) {
        if (cdouble_real(tau[idx]) != expected_tau[idx]) {
            fprintf(stderr, "[FAIL] ZUNGR2 Fortran->CBLAS thunk did not preserve complex-double TAU input\n");
            return 1;
        }
    }

    printf("[PASS] ZUNGR2 Fortran->CBLAS thunk translates row-major complex-double unblocked RQ-generator matrices and preserves TAU input\n");
    return 0;
}

static int check_zungr2_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_zung2l_fortran_slot_fn thunk = NULL;
    fb_complex_double_t a[8];
    fb_complex_double_t tau[2] = { make_cdouble(96.0), make_cdouble(97.0) };
    fb_complex_double_t work[2];
    int m = 2;
    int n = 4;
    int k = 2;
    int lda = 2;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zungr2_cblas_call, 0, sizeof(g_zungr2_cblas_call));
    memset(a, 0, sizeof(a));
    memset(work, 0, sizeof(work));

    vtable.ext_ops[FB_OP_ZUNGR2][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_zungr2_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_ZUNGR2);

    thunk = (fb_zung2l_fortran_slot_fn)vtable.ext_ops[FB_OP_ZUNGR2][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZUNGR2 CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&m, &n, &k, a, &lda, tau, work, &info);
    if (info != 118 || g_zungr2_cblas_call.called != 1 ||
        g_zungr2_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_zungr2_cblas_call.m != 2 || g_zungr2_cblas_call.n != 4 ||
        g_zungr2_cblas_call.k != 2 || g_zungr2_cblas_call.lda != 2 ||
        g_zungr2_cblas_call.a != a || g_zungr2_cblas_call.tau != tau) {
        fprintf(stderr, "[FAIL] ZUNGR2 CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] ZUNGR2 CBLAS->Fortran thunk maps the all-pointer ABI into the generic complex-double unblocked RQ-generator entry\n");
    return 0;
}

int main(void)
{
    if (check_sorg2l_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_sorg2l_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_dorg2l_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_dorg2l_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_cung2l_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_cung2l_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_zung2l_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_zung2l_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_sorgl2_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_sorgl2_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_dorgl2_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_dorgl2_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_cungl2_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_cungl2_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_zungl2_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_zungl2_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_sorgr2_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_sorgr2_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_dorgr2_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_dorgr2_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_cungr2_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_cungr2_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_zungr2_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_zungr2_cblas_to_fortran() != 0) {
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}