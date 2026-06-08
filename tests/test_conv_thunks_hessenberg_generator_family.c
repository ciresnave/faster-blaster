#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef int (*fb_sorghr_fn)(fb_layout_t layout, int n, int ilo, int ihi,
                            float *a, int lda, const float *tau);
typedef int (*fb_cunghr_fn)(fb_layout_t layout, int n, int ilo, int ihi,
                            fb_complex_float_t *a, int lda,
                            const fb_complex_float_t *tau);
typedef int (*fb_sgehrd_fn)(fb_layout_t layout, int n, int ilo, int ihi,
                            float *a, int lda, float *tau);
typedef int (*fb_cgehrd_fn)(fb_layout_t layout, int n, int ilo, int ihi,
                            fb_complex_float_t *a, int lda,
                            fb_complex_float_t *tau);
typedef int (*fb_sgehd2_fn)(fb_layout_t layout, int n, int ilo, int ihi,
                            float *a, int lda, float *tau);
typedef int (*fb_cgehd2_fn)(fb_layout_t layout, int n, int ilo, int ihi,
                            fb_complex_float_t *a, int lda,
                            fb_complex_float_t *tau);

typedef void (*fb_sorghr_fortran_slot_fn)(int *n, int *ilo, int *ihi, float *a,
                                          int *lda, float *tau, float *work,
                                          int *lwork, int *info);
typedef void (*fb_cunghr_fortran_slot_fn)(int *n, int *ilo, int *ihi,
                                          fb_complex_float_t *a, int *lda,
                                          fb_complex_float_t *tau,
                                          fb_complex_float_t *work,
                                          int *lwork, int *info);
typedef void (*fb_sgehrd_fortran_slot_fn)(int *n, int *ilo, int *ihi, float *a,
                                          int *lda, float *tau, float *work,
                                          int *lwork, int *info);
typedef void (*fb_cgehrd_fortran_slot_fn)(int *n, int *ilo, int *ihi,
                                          fb_complex_float_t *a, int *lda,
                                          fb_complex_float_t *tau,
                                          fb_complex_float_t *work,
                                          int *lwork, int *info);
typedef void (*fb_sgehd2_fortran_slot_fn)(int *n, int *ilo, int *ihi, float *a,
                                          int *lda, float *tau, float *work,
                                          int *info);
typedef void (*fb_cgehd2_fortran_slot_fn)(int *n, int *ilo, int *ihi,
                                          fb_complex_float_t *a, int *lda,
                                          fb_complex_float_t *tau,
                                          fb_complex_float_t *work,
                                          int *info);

static struct {
    int query_calls;
    int solve_calls;
    int n;
    int ilo;
    int ihi;
    int lda;
    int lwork_query;
    int lwork_solve;
    float a_snapshot[9];
    float tau_snapshot[2];
} g_sorghr_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    int n;
    int ilo;
    int ihi;
    int lda;
    float *a;
    const float *tau;
} g_sorghr_cblas_call;

static struct {
    int query_calls;
    int solve_calls;
    int n;
    int ilo;
    int ihi;
    int lda;
    int lwork_query;
    int lwork_solve;
    float a_real_snapshot[9];
    float tau_real_snapshot[2];
} g_cunghr_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    int n;
    int ilo;
    int ihi;
    int lda;
    fb_complex_float_t *a;
    const fb_complex_float_t *tau;
} g_cunghr_cblas_call;

static struct {
    int query_calls;
    int solve_calls;
    int n;
    int ilo;
    int ihi;
    int lda;
    int lwork_query;
    int lwork_solve;
    float a_snapshot[9];
    float tau_input_snapshot[2];
} g_sgehrd_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    int n;
    int ilo;
    int ihi;
    int lda;
    float *a;
    float *tau;
} g_sgehrd_cblas_call;

static struct {
    int query_calls;
    int solve_calls;
    int n;
    int ilo;
    int ihi;
    int lda;
    int lwork_query;
    int lwork_solve;
    float a_real_snapshot[9];
    float tau_input_real_snapshot[2];
} g_cgehrd_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    int n;
    int ilo;
    int ihi;
    int lda;
    fb_complex_float_t *a;
    fb_complex_float_t *tau;
} g_cgehrd_cblas_call;

static struct {
    int called;
    int n;
    int ilo;
    int ihi;
    int lda;
    int work_nonnull;
    float a_snapshot[9];
    float tau_input_snapshot[2];
} g_sgehd2_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    int n;
    int ilo;
    int ihi;
    int lda;
    float *a;
    float *tau;
} g_sgehd2_cblas_call;

static struct {
    int called;
    int n;
    int ilo;
    int ihi;
    int lda;
    int work_nonnull;
    float a_real_snapshot[9];
    float tau_input_real_snapshot[2];
} g_cgehd2_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    int n;
    int ilo;
    int ihi;
    int lda;
    fb_complex_float_t *a;
    fb_complex_float_t *tau;
} g_cgehd2_cblas_call;

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

static void stub_sorghr_fortran(int *n, int *ilo, int *ihi, float *a, int *lda,
                                float *tau, float *work, int *lwork, int *info)
{
    int row = 0;
    int col = 0;
    int tau_len = (*ihi > *ilo) ? (*ihi - *ilo) : 0;

    if (*lwork == -1) {
        g_sorghr_fortran_call.query_calls += 1;
        g_sorghr_fortran_call.n = *n;
        g_sorghr_fortran_call.ilo = *ilo;
        g_sorghr_fortran_call.ihi = *ihi;
        g_sorghr_fortran_call.lda = *lda;
        g_sorghr_fortran_call.lwork_query = *lwork;
        a[0] = -999.0f;
        if (tau_len > 0) {
            tau[0] = -777.0f;
        }
        work[0] = 5.0f;
        *info = 0;
        return;
    }

    g_sorghr_fortran_call.solve_calls += 1;
    g_sorghr_fortran_call.n = *n;
    g_sorghr_fortran_call.ilo = *ilo;
    g_sorghr_fortran_call.ihi = *ihi;
    g_sorghr_fortran_call.lda = *lda;
    g_sorghr_fortran_call.lwork_solve = *lwork;
    for (col = 0; col < *n; ++col) {
        for (row = 0; row < *n; ++row) {
            g_sorghr_fortran_call.a_snapshot[(col * (*n)) + row] =
                a[(col * (*lda)) + row];
            a[(col * (*lda)) + row] = (float)(1100 + (10 * col) + row);
        }
    }
    for (row = 0; row < tau_len; ++row) {
        g_sorghr_fortran_call.tau_snapshot[row] = tau[row];
    }
    *info = 0;
}

static int stub_sorghr_cblas(fb_layout_t layout, int n, int ilo, int ihi,
                             float *a, int lda, const float *tau)
{
    g_sorghr_cblas_call.called += 1;
    g_sorghr_cblas_call.layout = layout;
    g_sorghr_cblas_call.n = n;
    g_sorghr_cblas_call.ilo = ilo;
    g_sorghr_cblas_call.ihi = ihi;
    g_sorghr_cblas_call.lda = lda;
    g_sorghr_cblas_call.a = a;
    g_sorghr_cblas_call.tau = tau;
    return 95;
}

static void stub_cunghr_fortran(int *n, int *ilo, int *ihi,
                                fb_complex_float_t *a, int *lda,
                                fb_complex_float_t *tau,
                                fb_complex_float_t *work, int *lwork,
                                int *info)
{
    int row = 0;
    int col = 0;
    int tau_len = (*ihi > *ilo) ? (*ihi - *ilo) : 0;

    if (*lwork == -1) {
        g_cunghr_fortran_call.query_calls += 1;
        g_cunghr_fortran_call.n = *n;
        g_cunghr_fortran_call.ilo = *ilo;
        g_cunghr_fortran_call.ihi = *ihi;
        g_cunghr_fortran_call.lda = *lda;
        g_cunghr_fortran_call.lwork_query = *lwork;
        a[0] = make_cfloat(-999.0f);
        if (tau_len > 0) {
            tau[0] = make_cfloat(-777.0f);
        }
        work[0] = make_cfloat(6.0f);
        *info = 0;
        return;
    }

    g_cunghr_fortran_call.solve_calls += 1;
    g_cunghr_fortran_call.n = *n;
    g_cunghr_fortran_call.ilo = *ilo;
    g_cunghr_fortran_call.ihi = *ihi;
    g_cunghr_fortran_call.lda = *lda;
    g_cunghr_fortran_call.lwork_solve = *lwork;
    for (col = 0; col < *n; ++col) {
        for (row = 0; row < *n; ++row) {
            g_cunghr_fortran_call.a_real_snapshot[(col * (*n)) + row] =
                cfloat_real(a[(col * (*lda)) + row]);
            a[(col * (*lda)) + row] = make_cfloat((float)(1200 + (10 * col) + row));
        }
    }
    for (row = 0; row < tau_len; ++row) {
        g_cunghr_fortran_call.tau_real_snapshot[row] = cfloat_real(tau[row]);
    }
    *info = 0;
}

static int stub_cunghr_cblas(fb_layout_t layout, int n, int ilo, int ihi,
                             fb_complex_float_t *a, int lda,
                             const fb_complex_float_t *tau)
{
    g_cunghr_cblas_call.called += 1;
    g_cunghr_cblas_call.layout = layout;
    g_cunghr_cblas_call.n = n;
    g_cunghr_cblas_call.ilo = ilo;
    g_cunghr_cblas_call.ihi = ihi;
    g_cunghr_cblas_call.lda = lda;
    g_cunghr_cblas_call.a = a;
    g_cunghr_cblas_call.tau = tau;
    return 97;
}

static void stub_sgehd2_fortran(int *n, int *ilo, int *ihi, float *a, int *lda,
                                float *tau, float *work, int *info)
{
    int row = 0;
    int col = 0;
    int tau_len = (*ihi > *ilo) ? (*ihi - *ilo) : 0;

    g_sgehd2_fortran_call.called += 1;
    g_sgehd2_fortran_call.n = *n;
    g_sgehd2_fortran_call.ilo = *ilo;
    g_sgehd2_fortran_call.ihi = *ihi;
    g_sgehd2_fortran_call.lda = *lda;
    g_sgehd2_fortran_call.work_nonnull = (work != NULL);
    for (col = 0; col < *n; ++col) {
        for (row = 0; row < *n; ++row) {
            g_sgehd2_fortran_call.a_snapshot[(col * (*n)) + row] =
                a[(col * (*lda)) + row];
            a[(col * (*lda)) + row] = (float)(1500 + (10 * col) + row);
        }
    }
    for (row = 0; row < tau_len; ++row) {
        g_sgehd2_fortran_call.tau_input_snapshot[row] = tau[row];
        tau[row] = (float)(80 + row);
    }
    if (work) {
        work[0] = 321.0f;
    }
    *info = 0;
}

static int stub_sgehd2_cblas(fb_layout_t layout, int n, int ilo, int ihi,
                             float *a, int lda, float *tau)
{
    g_sgehd2_cblas_call.called += 1;
    g_sgehd2_cblas_call.layout = layout;
    g_sgehd2_cblas_call.n = n;
    g_sgehd2_cblas_call.ilo = ilo;
    g_sgehd2_cblas_call.ihi = ihi;
    g_sgehd2_cblas_call.lda = lda;
    g_sgehd2_cblas_call.a = a;
    g_sgehd2_cblas_call.tau = tau;
    return 109;
}

static void stub_cgehd2_fortran(int *n, int *ilo, int *ihi,
                                fb_complex_float_t *a, int *lda,
                                fb_complex_float_t *tau,
                                fb_complex_float_t *work, int *info)
{
    int row = 0;
    int col = 0;
    int tau_len = (*ihi > *ilo) ? (*ihi - *ilo) : 0;

    g_cgehd2_fortran_call.called += 1;
    g_cgehd2_fortran_call.n = *n;
    g_cgehd2_fortran_call.ilo = *ilo;
    g_cgehd2_fortran_call.ihi = *ihi;
    g_cgehd2_fortran_call.lda = *lda;
    g_cgehd2_fortran_call.work_nonnull = (work != NULL);
    for (col = 0; col < *n; ++col) {
        for (row = 0; row < *n; ++row) {
            g_cgehd2_fortran_call.a_real_snapshot[(col * (*n)) + row] =
                cfloat_real(a[(col * (*lda)) + row]);
            a[(col * (*lda)) + row] = make_cfloat((float)(1600 + (10 * col) + row));
        }
    }
    for (row = 0; row < tau_len; ++row) {
        g_cgehd2_fortran_call.tau_input_real_snapshot[row] = cfloat_real(tau[row]);
        tau[row] = make_cfloat((float)(90 + row));
    }
    if (work) {
        work[0] = make_cfloat(654.0f);
    }
    *info = 0;
}

static int stub_cgehd2_cblas(fb_layout_t layout, int n, int ilo, int ihi,
                             fb_complex_float_t *a, int lda,
                             fb_complex_float_t *tau)
{
    g_cgehd2_cblas_call.called += 1;
    g_cgehd2_cblas_call.layout = layout;
    g_cgehd2_cblas_call.n = n;
    g_cgehd2_cblas_call.ilo = ilo;
    g_cgehd2_cblas_call.ihi = ihi;
    g_cgehd2_cblas_call.lda = lda;
    g_cgehd2_cblas_call.a = a;
    g_cgehd2_cblas_call.tau = tau;
    return 111;
}

static void stub_sgehrd_fortran(int *n, int *ilo, int *ihi, float *a, int *lda,
                                float *tau, float *work, int *lwork, int *info)
{
    int row = 0;
    int col = 0;
    int tau_len = (*ihi > *ilo) ? (*ihi - *ilo) : 0;

    if (*lwork == -1) {
        g_sgehrd_fortran_call.query_calls += 1;
        g_sgehrd_fortran_call.n = *n;
        g_sgehrd_fortran_call.ilo = *ilo;
        g_sgehrd_fortran_call.ihi = *ihi;
        g_sgehrd_fortran_call.lda = *lda;
        g_sgehrd_fortran_call.lwork_query = *lwork;
        a[0] = -1999.0f;
        if (tau_len > 0) {
            tau[0] = -1777.0f;
        }
        work[0] = 7.0f;
        *info = 0;
        return;
    }

    g_sgehrd_fortran_call.solve_calls += 1;
    g_sgehrd_fortran_call.n = *n;
    g_sgehrd_fortran_call.ilo = *ilo;
    g_sgehrd_fortran_call.ihi = *ihi;
    g_sgehrd_fortran_call.lda = *lda;
    g_sgehrd_fortran_call.lwork_solve = *lwork;
    for (col = 0; col < *n; ++col) {
        for (row = 0; row < *n; ++row) {
            g_sgehrd_fortran_call.a_snapshot[(col * (*n)) + row] =
                a[(col * (*lda)) + row];
            a[(col * (*lda)) + row] = (float)(1300 + (10 * col) + row);
        }
    }
    for (row = 0; row < tau_len; ++row) {
        g_sgehrd_fortran_call.tau_input_snapshot[row] = tau[row];
        tau[row] = (float)(60 + row);
    }
    *info = 0;
}

static int stub_sgehrd_cblas(fb_layout_t layout, int n, int ilo, int ihi,
                             float *a, int lda, float *tau)
{
    g_sgehrd_cblas_call.called += 1;
    g_sgehrd_cblas_call.layout = layout;
    g_sgehrd_cblas_call.n = n;
    g_sgehrd_cblas_call.ilo = ilo;
    g_sgehrd_cblas_call.ihi = ihi;
    g_sgehrd_cblas_call.lda = lda;
    g_sgehrd_cblas_call.a = a;
    g_sgehrd_cblas_call.tau = tau;
    return 99;
}

static void stub_cgehrd_fortran(int *n, int *ilo, int *ihi,
                                fb_complex_float_t *a, int *lda,
                                fb_complex_float_t *tau,
                                fb_complex_float_t *work, int *lwork,
                                int *info)
{
    int row = 0;
    int col = 0;
    int tau_len = (*ihi > *ilo) ? (*ihi - *ilo) : 0;

    if (*lwork == -1) {
        g_cgehrd_fortran_call.query_calls += 1;
        g_cgehrd_fortran_call.n = *n;
        g_cgehrd_fortran_call.ilo = *ilo;
        g_cgehrd_fortran_call.ihi = *ihi;
        g_cgehrd_fortran_call.lda = *lda;
        g_cgehrd_fortran_call.lwork_query = *lwork;
        a[0] = make_cfloat(-2999.0f);
        if (tau_len > 0) {
            tau[0] = make_cfloat(-2777.0f);
        }
        work[0] = make_cfloat(8.0f);
        *info = 0;
        return;
    }

    g_cgehrd_fortran_call.solve_calls += 1;
    g_cgehrd_fortran_call.n = *n;
    g_cgehrd_fortran_call.ilo = *ilo;
    g_cgehrd_fortran_call.ihi = *ihi;
    g_cgehrd_fortran_call.lda = *lda;
    g_cgehrd_fortran_call.lwork_solve = *lwork;
    for (col = 0; col < *n; ++col) {
        for (row = 0; row < *n; ++row) {
            g_cgehrd_fortran_call.a_real_snapshot[(col * (*n)) + row] =
                cfloat_real(a[(col * (*lda)) + row]);
            a[(col * (*lda)) + row] = make_cfloat((float)(1400 + (10 * col) + row));
        }
    }
    for (row = 0; row < tau_len; ++row) {
        g_cgehrd_fortran_call.tau_input_real_snapshot[row] = cfloat_real(tau[row]);
        tau[row] = make_cfloat((float)(70 + row));
    }
    *info = 0;
}

static int stub_cgehrd_cblas(fb_layout_t layout, int n, int ilo, int ihi,
                             fb_complex_float_t *a, int lda,
                             fb_complex_float_t *tau)
{
    g_cgehrd_cblas_call.called += 1;
    g_cgehrd_cblas_call.layout = layout;
    g_cgehrd_cblas_call.n = n;
    g_cgehrd_cblas_call.ilo = ilo;
    g_cgehrd_cblas_call.ihi = ihi;
    g_cgehrd_cblas_call.lda = lda;
    g_cgehrd_cblas_call.a = a;
    g_cgehrd_cblas_call.tau = tau;
    return 101;
}

static int check_sorghr_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_sorghr_fn thunk = NULL;
    float a[9] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f };
    float tau[2] = { 40.0f, 41.0f };
    float expected_a_snapshot[9] = { 1.0f, 4.0f, 7.0f, 2.0f, 5.0f, 8.0f, 3.0f, 6.0f, 9.0f };
    float expected_tau[2] = { 40.0f, 41.0f };
    float expected_a_out[9] = { 1100.0f, 1110.0f, 1120.0f, 1101.0f, 1111.0f, 1121.0f, 1102.0f, 1112.0f, 1122.0f };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sorghr_fortran_call, 0, sizeof(g_sorghr_fortran_call));

    vtable.ext_ops[FB_OP_SORGHR][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_sorghr_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SORGHR);

    thunk = (fb_sorghr_fn)vtable.ext_ops[FB_OP_SORGHR][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SORGHR Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 3, 1, 3, a, 3, tau);
    if (info != 0 || g_sorghr_fortran_call.query_calls != 1 ||
        g_sorghr_fortran_call.solve_calls != 1 ||
        g_sorghr_fortran_call.n != 3 || g_sorghr_fortran_call.ilo != 1 ||
        g_sorghr_fortran_call.ihi != 3 || g_sorghr_fortran_call.lda != 3 ||
        g_sorghr_fortran_call.lwork_query != -1 ||
        g_sorghr_fortran_call.lwork_solve != 5 ||
        memcmp(g_sorghr_fortran_call.a_snapshot, expected_a_snapshot,
               sizeof(expected_a_snapshot)) != 0 ||
        memcmp(g_sorghr_fortran_call.tau_snapshot, expected_tau,
               sizeof(expected_tau)) != 0 ||
        memcmp(a, expected_a_out, sizeof(expected_a_out)) != 0 ||
        memcmp(tau, expected_tau, sizeof(expected_tau)) != 0) {
        fprintf(stderr, "[FAIL] SORGHR Fortran->CBLAS thunk did not preserve Hessenberg generator row-major semantics\n");
        return 1;
    }

    printf("[PASS] SORGHR Fortran->CBLAS thunk translates row-major Hessenberg generator matrices and preserves TAU across lwork query\n");
    return 0;
}

static int check_sorghr_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_sorghr_fortran_slot_fn thunk = NULL;
    float a[9] = { 0.0f };
    float tau[2] = { 40.0f, 41.0f };
    float work[4] = { 0.0f };
    int n = 3;
    int ilo = 1;
    int ihi = 3;
    int lda = 3;
    int lwork = 4;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sorghr_cblas_call, 0, sizeof(g_sorghr_cblas_call));

    vtable.ext_ops[FB_OP_SORGHR][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_sorghr_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SORGHR);

    thunk = (fb_sorghr_fortran_slot_fn)vtable.ext_ops[FB_OP_SORGHR][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SORGHR CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&n, &ilo, &ihi, a, &lda, tau, work, &lwork, &info);
    if (info != 95 || g_sorghr_cblas_call.called != 1 ||
        g_sorghr_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_sorghr_cblas_call.n != 3 || g_sorghr_cblas_call.ilo != 1 ||
        g_sorghr_cblas_call.ihi != 3 || g_sorghr_cblas_call.lda != 3 ||
        g_sorghr_cblas_call.a != a || g_sorghr_cblas_call.tau != tau) {
        fprintf(stderr, "[FAIL] SORGHR CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] SORGHR CBLAS->Fortran thunk maps the all-pointer ABI into the generic C Hessenberg generator entry\n");
    return 0;
}

static int check_cunghr_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_cunghr_fn thunk = NULL;
    fb_complex_float_t a[9] = {
        make_cfloat(1.0f), make_cfloat(2.0f), make_cfloat(3.0f),
        make_cfloat(4.0f), make_cfloat(5.0f), make_cfloat(6.0f),
        make_cfloat(7.0f), make_cfloat(8.0f), make_cfloat(9.0f)
    };
    fb_complex_float_t tau[2] = { make_cfloat(50.0f), make_cfloat(51.0f) };
    float expected_a_snapshot[9] = { 1.0f, 4.0f, 7.0f, 2.0f, 5.0f, 8.0f, 3.0f, 6.0f, 9.0f };
    float expected_tau[2] = { 50.0f, 51.0f };
    float expected_a_out[9] = { 1200.0f, 1210.0f, 1220.0f, 1201.0f, 1211.0f, 1221.0f, 1202.0f, 1212.0f, 1222.0f };
    int info = 0;
    int idx = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cunghr_fortran_call, 0, sizeof(g_cunghr_fortran_call));

    vtable.ext_ops[FB_OP_CUNGHR][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_cunghr_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CUNGHR);

    thunk = (fb_cunghr_fn)vtable.ext_ops[FB_OP_CUNGHR][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CUNGHR Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 3, 1, 3, a, 3, tau);
    if (info != 0 || g_cunghr_fortran_call.query_calls != 1 ||
        g_cunghr_fortran_call.solve_calls != 1 ||
        g_cunghr_fortran_call.n != 3 || g_cunghr_fortran_call.ilo != 1 ||
        g_cunghr_fortran_call.ihi != 3 || g_cunghr_fortran_call.lda != 3 ||
        g_cunghr_fortran_call.lwork_query != -1 ||
        g_cunghr_fortran_call.lwork_solve != 6 ||
        memcmp(g_cunghr_fortran_call.a_real_snapshot, expected_a_snapshot,
               sizeof(expected_a_snapshot)) != 0 ||
        memcmp(g_cunghr_fortran_call.tau_real_snapshot, expected_tau,
               sizeof(expected_tau)) != 0) {
        fprintf(stderr, "[FAIL] CUNGHR Fortran->CBLAS thunk did not preserve complex Hessenberg generator row-major semantics\n");
        return 1;
    }

    for (idx = 0; idx < 9; ++idx) {
        if (cfloat_real(a[idx]) != expected_a_out[idx]) {
            fprintf(stderr, "[FAIL] CUNGHR Fortran->CBLAS thunk did not copy complex Hessenberg generator output back correctly\n");
            return 1;
        }
    }
    for (idx = 0; idx < 2; ++idx) {
        if (cfloat_real(tau[idx]) != expected_tau[idx]) {
            fprintf(stderr, "[FAIL] CUNGHR Fortran->CBLAS thunk did not preserve complex TAU input\n");
            return 1;
        }
    }

    printf("[PASS] CUNGHR Fortran->CBLAS thunk translates row-major complex Hessenberg generator matrices and preserves TAU across lwork query\n");
    return 0;
}

static int check_cunghr_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_cunghr_fortran_slot_fn thunk = NULL;
    fb_complex_float_t a[9];
    fb_complex_float_t tau[2] = { make_cfloat(50.0f), make_cfloat(51.0f) };
    fb_complex_float_t work[4];
    int n = 3;
    int ilo = 1;
    int ihi = 3;
    int lda = 3;
    int lwork = 4;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cunghr_cblas_call, 0, sizeof(g_cunghr_cblas_call));
    memset(a, 0, sizeof(a));
    memset(work, 0, sizeof(work));

    vtable.ext_ops[FB_OP_CUNGHR][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_cunghr_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CUNGHR);

    thunk = (fb_cunghr_fortran_slot_fn)vtable.ext_ops[FB_OP_CUNGHR][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CUNGHR CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&n, &ilo, &ihi, a, &lda, tau, work, &lwork, &info);
    if (info != 97 || g_cunghr_cblas_call.called != 1 ||
        g_cunghr_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_cunghr_cblas_call.n != 3 || g_cunghr_cblas_call.ilo != 1 ||
        g_cunghr_cblas_call.ihi != 3 || g_cunghr_cblas_call.lda != 3 ||
        g_cunghr_cblas_call.a != a || g_cunghr_cblas_call.tau != tau) {
        fprintf(stderr, "[FAIL] CUNGHR CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] CUNGHR CBLAS->Fortran thunk maps the all-pointer ABI into the generic complex C Hessenberg generator entry\n");
    return 0;
}

static int check_sgehd2_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_sgehd2_fn thunk = NULL;
    float a[9] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f };
    float tau[2] = { 700.0f, 701.0f };
    float expected_a_snapshot[9] = { 1.0f, 4.0f, 7.0f, 2.0f, 5.0f, 8.0f, 3.0f, 6.0f, 9.0f };
    float expected_tau_input[2] = { 0.0f, 0.0f };
    float expected_tau_out[2] = { 80.0f, 81.0f };
    float expected_a_out[9] = { 1500.0f, 1510.0f, 1520.0f, 1501.0f, 1511.0f, 1521.0f, 1502.0f, 1512.0f, 1522.0f };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sgehd2_fortran_call, 0, sizeof(g_sgehd2_fortran_call));

    vtable.ext_ops[FB_OP_SGEHD2][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_sgehd2_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SGEHD2);

    thunk = (fb_sgehd2_fn)vtable.ext_ops[FB_OP_SGEHD2][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SGEHD2 Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 3, 1, 3, a, 3, tau);
    if (info != 0 || g_sgehd2_fortran_call.called != 1 ||
        g_sgehd2_fortran_call.n != 3 || g_sgehd2_fortran_call.ilo != 1 ||
        g_sgehd2_fortran_call.ihi != 3 || g_sgehd2_fortran_call.lda != 3 ||
        !g_sgehd2_fortran_call.work_nonnull ||
        memcmp(g_sgehd2_fortran_call.a_snapshot, expected_a_snapshot,
               sizeof(expected_a_snapshot)) != 0 ||
        memcmp(g_sgehd2_fortran_call.tau_input_snapshot, expected_tau_input,
               sizeof(expected_tau_input)) != 0 ||
        memcmp(a, expected_a_out, sizeof(expected_a_out)) != 0 ||
        memcmp(tau, expected_tau_out, sizeof(expected_tau_out)) != 0) {
        fprintf(stderr, "[FAIL] SGEHD2 Fortran->CBLAS thunk did not preserve Hessenberg reduction row-major semantics\n");
        return 1;
    }

    printf("[PASS] SGEHD2 Fortran->CBLAS thunk translates row-major unblocked Hessenberg reduction matrices, allocates internal scratch, and copies TAU output back\n");
    return 0;
}

static int check_sgehd2_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_sgehd2_fortran_slot_fn thunk = NULL;
    float a[9] = { 0.0f };
    float tau[2] = { 0.0f };
    float work[4] = { 0.0f };
    int n = 3;
    int ilo = 1;
    int ihi = 3;
    int lda = 3;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sgehd2_cblas_call, 0, sizeof(g_sgehd2_cblas_call));

    vtable.ext_ops[FB_OP_SGEHD2][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_sgehd2_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SGEHD2);

    thunk = (fb_sgehd2_fortran_slot_fn)vtable.ext_ops[FB_OP_SGEHD2][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SGEHD2 CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&n, &ilo, &ihi, a, &lda, tau, work, &info);
    if (info != 109 || g_sgehd2_cblas_call.called != 1 ||
        g_sgehd2_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_sgehd2_cblas_call.n != 3 || g_sgehd2_cblas_call.ilo != 1 ||
        g_sgehd2_cblas_call.ihi != 3 || g_sgehd2_cblas_call.lda != 3 ||
        g_sgehd2_cblas_call.a != a || g_sgehd2_cblas_call.tau != tau) {
        fprintf(stderr, "[FAIL] SGEHD2 CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] SGEHD2 CBLAS->Fortran thunk maps the all-pointer ABI into the generic C unblocked Hessenberg reduction entry\n");
    return 0;
}

static int check_cgehd2_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_cgehd2_fn thunk = NULL;
    fb_complex_float_t a[9] = {
        make_cfloat(1.0f), make_cfloat(2.0f), make_cfloat(3.0f),
        make_cfloat(4.0f), make_cfloat(5.0f), make_cfloat(6.0f),
        make_cfloat(7.0f), make_cfloat(8.0f), make_cfloat(9.0f)
    };
    fb_complex_float_t tau[2] = { make_cfloat(800.0f), make_cfloat(801.0f) };
    float expected_a_snapshot[9] = { 1.0f, 4.0f, 7.0f, 2.0f, 5.0f, 8.0f, 3.0f, 6.0f, 9.0f };
    float expected_tau_input[2] = { 0.0f, 0.0f };
    float expected_tau_out[2] = { 90.0f, 91.0f };
    float expected_a_out[9] = { 1600.0f, 1610.0f, 1620.0f, 1601.0f, 1611.0f, 1621.0f, 1602.0f, 1612.0f, 1622.0f };
    int info = 0;
    int idx = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cgehd2_fortran_call, 0, sizeof(g_cgehd2_fortran_call));

    vtable.ext_ops[FB_OP_CGEHD2][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_cgehd2_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CGEHD2);

    thunk = (fb_cgehd2_fn)vtable.ext_ops[FB_OP_CGEHD2][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CGEHD2 Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 3, 1, 3, a, 3, tau);
    if (info != 0 || g_cgehd2_fortran_call.called != 1 ||
        g_cgehd2_fortran_call.n != 3 || g_cgehd2_fortran_call.ilo != 1 ||
        g_cgehd2_fortran_call.ihi != 3 || g_cgehd2_fortran_call.lda != 3 ||
        !g_cgehd2_fortran_call.work_nonnull ||
        memcmp(g_cgehd2_fortran_call.a_real_snapshot, expected_a_snapshot,
               sizeof(expected_a_snapshot)) != 0 ||
        memcmp(g_cgehd2_fortran_call.tau_input_real_snapshot, expected_tau_input,
               sizeof(expected_tau_input)) != 0) {
        fprintf(stderr, "[FAIL] CGEHD2 Fortran->CBLAS thunk did not preserve complex unblocked Hessenberg reduction row-major semantics\n");
        return 1;
    }

    for (idx = 0; idx < 9; ++idx) {
        if (cfloat_real(a[idx]) != expected_a_out[idx]) {
            fprintf(stderr, "[FAIL] CGEHD2 Fortran->CBLAS thunk did not copy complex unblocked Hessenberg reduction output back correctly\n");
            return 1;
        }
    }
    for (idx = 0; idx < 2; ++idx) {
        if (cfloat_real(tau[idx]) != expected_tau_out[idx]) {
            fprintf(stderr, "[FAIL] CGEHD2 Fortran->CBLAS thunk did not copy complex TAU output back correctly\n");
            return 1;
        }
    }

    printf("[PASS] CGEHD2 Fortran->CBLAS thunk translates row-major complex unblocked Hessenberg reduction matrices, allocates internal scratch, and copies TAU output back\n");
    return 0;
}

static int check_cgehd2_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_cgehd2_fortran_slot_fn thunk = NULL;
    fb_complex_float_t a[9];
    fb_complex_float_t tau[2];
    fb_complex_float_t work[4];
    int n = 3;
    int ilo = 1;
    int ihi = 3;
    int lda = 3;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cgehd2_cblas_call, 0, sizeof(g_cgehd2_cblas_call));
    memset(a, 0, sizeof(a));
    memset(tau, 0, sizeof(tau));
    memset(work, 0, sizeof(work));

    vtable.ext_ops[FB_OP_CGEHD2][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_cgehd2_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CGEHD2);

    thunk = (fb_cgehd2_fortran_slot_fn)vtable.ext_ops[FB_OP_CGEHD2][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CGEHD2 CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&n, &ilo, &ihi, a, &lda, tau, work, &info);
    if (info != 111 || g_cgehd2_cblas_call.called != 1 ||
        g_cgehd2_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_cgehd2_cblas_call.n != 3 || g_cgehd2_cblas_call.ilo != 1 ||
        g_cgehd2_cblas_call.ihi != 3 || g_cgehd2_cblas_call.lda != 3 ||
        g_cgehd2_cblas_call.a != a || g_cgehd2_cblas_call.tau != tau) {
        fprintf(stderr, "[FAIL] CGEHD2 CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] CGEHD2 CBLAS->Fortran thunk maps the all-pointer ABI into the generic complex C unblocked Hessenberg reduction entry\n");
    return 0;
}

static int check_sgehrd_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_sgehrd_fn thunk = NULL;
    float a[9] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f };
    float tau[2] = { 500.0f, 501.0f };
    float expected_a_snapshot[9] = { 1.0f, 4.0f, 7.0f, 2.0f, 5.0f, 8.0f, 3.0f, 6.0f, 9.0f };
    float expected_tau_input[2] = { 0.0f, 0.0f };
    float expected_tau_out[2] = { 60.0f, 61.0f };
    float expected_a_out[9] = { 1300.0f, 1310.0f, 1320.0f, 1301.0f, 1311.0f, 1321.0f, 1302.0f, 1312.0f, 1322.0f };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sgehrd_fortran_call, 0, sizeof(g_sgehrd_fortran_call));

    vtable.ext_ops[FB_OP_SGEHRD][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_sgehrd_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SGEHRD);

    thunk = (fb_sgehrd_fn)vtable.ext_ops[FB_OP_SGEHRD][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SGEHRD Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 3, 1, 3, a, 3, tau);
    if (info != 0 || g_sgehrd_fortran_call.query_calls != 1 ||
        g_sgehrd_fortran_call.solve_calls != 1 ||
        g_sgehrd_fortran_call.n != 3 || g_sgehrd_fortran_call.ilo != 1 ||
        g_sgehrd_fortran_call.ihi != 3 || g_sgehrd_fortran_call.lda != 3 ||
        g_sgehrd_fortran_call.lwork_query != -1 ||
        g_sgehrd_fortran_call.lwork_solve != 7 ||
        memcmp(g_sgehrd_fortran_call.a_snapshot, expected_a_snapshot,
               sizeof(expected_a_snapshot)) != 0 ||
        memcmp(g_sgehrd_fortran_call.tau_input_snapshot, expected_tau_input,
               sizeof(expected_tau_input)) != 0 ||
        memcmp(a, expected_a_out, sizeof(expected_a_out)) != 0 ||
        memcmp(tau, expected_tau_out, sizeof(expected_tau_out)) != 0) {
        fprintf(stderr, "[FAIL] SGEHRD Fortran->CBLAS thunk did not preserve Hessenberg reduction row-major semantics\n");
        return 1;
    }

    printf("[PASS] SGEHRD Fortran->CBLAS thunk translates row-major Hessenberg reduction matrices and copies TAU output back across lwork query\n");
    return 0;
}

static int check_sgehrd_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_sgehrd_fortran_slot_fn thunk = NULL;
    float a[9] = { 0.0f };
    float tau[2] = { 0.0f };
    float work[4] = { 0.0f };
    int n = 3;
    int ilo = 1;
    int ihi = 3;
    int lda = 3;
    int lwork = 4;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sgehrd_cblas_call, 0, sizeof(g_sgehrd_cblas_call));

    vtable.ext_ops[FB_OP_SGEHRD][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_sgehrd_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SGEHRD);

    thunk = (fb_sgehrd_fortran_slot_fn)vtable.ext_ops[FB_OP_SGEHRD][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SGEHRD CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&n, &ilo, &ihi, a, &lda, tau, work, &lwork, &info);
    if (info != 99 || g_sgehrd_cblas_call.called != 1 ||
        g_sgehrd_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_sgehrd_cblas_call.n != 3 || g_sgehrd_cblas_call.ilo != 1 ||
        g_sgehrd_cblas_call.ihi != 3 || g_sgehrd_cblas_call.lda != 3 ||
        g_sgehrd_cblas_call.a != a || g_sgehrd_cblas_call.tau != tau) {
        fprintf(stderr, "[FAIL] SGEHRD CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] SGEHRD CBLAS->Fortran thunk maps the all-pointer ABI into the generic C Hessenberg reduction entry\n");
    return 0;
}

static int check_cgehrd_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_cgehrd_fn thunk = NULL;
    fb_complex_float_t a[9] = {
        make_cfloat(1.0f), make_cfloat(2.0f), make_cfloat(3.0f),
        make_cfloat(4.0f), make_cfloat(5.0f), make_cfloat(6.0f),
        make_cfloat(7.0f), make_cfloat(8.0f), make_cfloat(9.0f)
    };
    fb_complex_float_t tau[2] = { make_cfloat(600.0f), make_cfloat(601.0f) };
    float expected_a_snapshot[9] = { 1.0f, 4.0f, 7.0f, 2.0f, 5.0f, 8.0f, 3.0f, 6.0f, 9.0f };
    float expected_tau_input[2] = { 0.0f, 0.0f };
    float expected_tau_out[2] = { 70.0f, 71.0f };
    float expected_a_out[9] = { 1400.0f, 1410.0f, 1420.0f, 1401.0f, 1411.0f, 1421.0f, 1402.0f, 1412.0f, 1422.0f };
    int info = 0;
    int idx = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cgehrd_fortran_call, 0, sizeof(g_cgehrd_fortran_call));

    vtable.ext_ops[FB_OP_CGEHRD][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_cgehrd_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CGEHRD);

    thunk = (fb_cgehrd_fn)vtable.ext_ops[FB_OP_CGEHRD][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CGEHRD Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 3, 1, 3, a, 3, tau);
    if (info != 0 || g_cgehrd_fortran_call.query_calls != 1 ||
        g_cgehrd_fortran_call.solve_calls != 1 ||
        g_cgehrd_fortran_call.n != 3 || g_cgehrd_fortran_call.ilo != 1 ||
        g_cgehrd_fortran_call.ihi != 3 || g_cgehrd_fortran_call.lda != 3 ||
        g_cgehrd_fortran_call.lwork_query != -1 ||
        g_cgehrd_fortran_call.lwork_solve != 8 ||
        memcmp(g_cgehrd_fortran_call.a_real_snapshot, expected_a_snapshot,
               sizeof(expected_a_snapshot)) != 0 ||
        memcmp(g_cgehrd_fortran_call.tau_input_real_snapshot, expected_tau_input,
               sizeof(expected_tau_input)) != 0) {
        fprintf(stderr, "[FAIL] CGEHRD Fortran->CBLAS thunk did not preserve complex Hessenberg reduction row-major semantics\n");
        return 1;
    }

    for (idx = 0; idx < 9; ++idx) {
        if (cfloat_real(a[idx]) != expected_a_out[idx]) {
            fprintf(stderr, "[FAIL] CGEHRD Fortran->CBLAS thunk did not copy complex Hessenberg reduction output back correctly\n");
            return 1;
        }
    }
    for (idx = 0; idx < 2; ++idx) {
        if (cfloat_real(tau[idx]) != expected_tau_out[idx]) {
            fprintf(stderr, "[FAIL] CGEHRD Fortran->CBLAS thunk did not copy complex TAU output back correctly\n");
            return 1;
        }
    }

    printf("[PASS] CGEHRD Fortran->CBLAS thunk translates row-major complex Hessenberg reduction matrices and copies TAU output back across lwork query\n");
    return 0;
}

static int check_cgehrd_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_cgehrd_fortran_slot_fn thunk = NULL;
    fb_complex_float_t a[9];
    fb_complex_float_t tau[2];
    fb_complex_float_t work[4];
    int n = 3;
    int ilo = 1;
    int ihi = 3;
    int lda = 3;
    int lwork = 4;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cgehrd_cblas_call, 0, sizeof(g_cgehrd_cblas_call));
    memset(a, 0, sizeof(a));
    memset(tau, 0, sizeof(tau));
    memset(work, 0, sizeof(work));

    vtable.ext_ops[FB_OP_CGEHRD][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_cgehrd_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CGEHRD);

    thunk = (fb_cgehrd_fortran_slot_fn)vtable.ext_ops[FB_OP_CGEHRD][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CGEHRD CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&n, &ilo, &ihi, a, &lda, tau, work, &lwork, &info);
    if (info != 101 || g_cgehrd_cblas_call.called != 1 ||
        g_cgehrd_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_cgehrd_cblas_call.n != 3 || g_cgehrd_cblas_call.ilo != 1 ||
        g_cgehrd_cblas_call.ihi != 3 || g_cgehrd_cblas_call.lda != 3 ||
        g_cgehrd_cblas_call.a != a || g_cgehrd_cblas_call.tau != tau) {
        fprintf(stderr, "[FAIL] CGEHRD CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] CGEHRD CBLAS->Fortran thunk maps the all-pointer ABI into the generic complex C Hessenberg reduction entry\n");
    return 0;
}

int main(void)
{
    if (check_sorghr_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_sorghr_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_cunghr_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_cunghr_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_sgehd2_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_sgehd2_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_cgehd2_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_cgehd2_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_sgehrd_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_sgehrd_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_cgehrd_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_cgehrd_cblas_to_fortran() != 0) {
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}