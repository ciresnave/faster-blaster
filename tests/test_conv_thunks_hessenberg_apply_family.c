#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef int (*fb_sormhr_fn)(fb_layout_t layout, char side, char trans, int m,
                            int n, int ilo, int ihi, const float *a, int lda,
                            const float *tau, float *c, int ldc);
typedef int (*fb_cunmhr_fn)(fb_layout_t layout, char side, char trans, int m,
                            int n, int ilo, int ihi,
                            const fb_complex_float_t *a, int lda,
                            const fb_complex_float_t *tau,
                            fb_complex_float_t *c, int ldc);

typedef void (*fb_sormhr_fortran_slot_fn)(char *side, char *trans, int *m,
                                          int *n, int *ilo, int *ihi,
                                          float *a, int *lda, float *tau,
                                          float *c, int *ldc, float *work,
                                          int *lwork, int *info);
typedef void (*fb_cunmhr_fortran_slot_fn)(char *side, char *trans, int *m,
                                          int *n, int *ilo, int *ihi,
                                          fb_complex_float_t *a, int *lda,
                                          fb_complex_float_t *tau,
                                          fb_complex_float_t *c, int *ldc,
                                          fb_complex_float_t *work,
                                          int *lwork, int *info);

static struct {
    int query_calls;
    int solve_calls;
    char side;
    char trans;
    int m;
    int n;
    int ilo;
    int ihi;
    int lda;
    int ldc;
    int lwork_query;
    int lwork_solve;
    float a_snapshot[9];
    float c_snapshot[6];
    float tau_snapshot[2];
} g_sormhr_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    char side;
    char trans;
    int m;
    int n;
    int ilo;
    int ihi;
    int lda;
    int ldc;
    const float *a;
    const float *tau;
    float *c;
} g_sormhr_cblas_call;

static struct {
    int query_calls;
    int solve_calls;
    char side;
    char trans;
    int m;
    int n;
    int ilo;
    int ihi;
    int lda;
    int ldc;
    int lwork_query;
    int lwork_solve;
    float a_real_snapshot[9];
    float c_real_snapshot[6];
    float tau_real_snapshot[2];
} g_cunmhr_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    char side;
    char trans;
    int m;
    int n;
    int ilo;
    int ihi;
    int lda;
    int ldc;
    const fb_complex_float_t *a;
    const fb_complex_float_t *tau;
    fb_complex_float_t *c;
} g_cunmhr_cblas_call;

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

static void stub_sormhr_fortran(char *side, char *trans, int *m, int *n,
                                int *ilo, int *ihi, float *a, int *lda,
                                float *tau, float *c, int *ldc, float *work,
                                int *lwork, int *info)
{
    int row = 0;
    int col = 0;
    int tau_len = (*ihi > *ilo) ? (*ihi - *ilo) : 0;

    if (*lwork == -1) {
        g_sormhr_fortran_call.query_calls += 1;
        g_sormhr_fortran_call.side = *side;
        g_sormhr_fortran_call.trans = *trans;
        g_sormhr_fortran_call.m = *m;
        g_sormhr_fortran_call.n = *n;
        g_sormhr_fortran_call.ilo = *ilo;
        g_sormhr_fortran_call.ihi = *ihi;
        g_sormhr_fortran_call.lda = *lda;
        g_sormhr_fortran_call.ldc = *ldc;
        g_sormhr_fortran_call.lwork_query = *lwork;
        a[0] = -999.0f;
        if (tau_len > 0) {
            tau[0] = -777.0f;
        }
        c[0] = -555.0f;
        work[0] = 6.0f;
        *info = 0;
        return;
    }

    g_sormhr_fortran_call.solve_calls += 1;
    g_sormhr_fortran_call.side = *side;
    g_sormhr_fortran_call.trans = *trans;
    g_sormhr_fortran_call.m = *m;
    g_sormhr_fortran_call.n = *n;
    g_sormhr_fortran_call.ilo = *ilo;
    g_sormhr_fortran_call.ihi = *ihi;
    g_sormhr_fortran_call.lda = *lda;
    g_sormhr_fortran_call.ldc = *ldc;
    g_sormhr_fortran_call.lwork_solve = *lwork;
    for (col = 0; col < *lda; ++col) {
        for (row = 0; row < *lda; ++row) {
            g_sormhr_fortran_call.a_snapshot[(col * (*lda)) + row] =
                a[(col * (*lda)) + row];
        }
    }
    for (col = 0; col < *n; ++col) {
        for (row = 0; row < *m; ++row) {
            g_sormhr_fortran_call.c_snapshot[(col * (*m)) + row] =
                c[(col * (*ldc)) + row];
            c[(col * (*ldc)) + row] = (float)(900 + (10 * col) + row);
        }
    }
    for (row = 0; row < tau_len; ++row) {
        g_sormhr_fortran_call.tau_snapshot[row] = tau[row];
    }
    a[0] = -123.0f;
    if (tau_len > 0) {
        tau[0] = -456.0f;
    }
    *info = 0;
}

static int stub_sormhr_cblas(fb_layout_t layout, char side, char trans, int m,
                             int n, int ilo, int ihi, const float *a, int lda,
                             const float *tau, float *c, int ldc)
{
    g_sormhr_cblas_call.called += 1;
    g_sormhr_cblas_call.layout = layout;
    g_sormhr_cblas_call.side = side;
    g_sormhr_cblas_call.trans = trans;
    g_sormhr_cblas_call.m = m;
    g_sormhr_cblas_call.n = n;
    g_sormhr_cblas_call.ilo = ilo;
    g_sormhr_cblas_call.ihi = ihi;
    g_sormhr_cblas_call.lda = lda;
    g_sormhr_cblas_call.ldc = ldc;
    g_sormhr_cblas_call.a = a;
    g_sormhr_cblas_call.tau = tau;
    g_sormhr_cblas_call.c = c;
    return 91;
}

static void stub_cunmhr_fortran(char *side, char *trans, int *m, int *n,
                                int *ilo, int *ihi, fb_complex_float_t *a,
                                int *lda, fb_complex_float_t *tau,
                                fb_complex_float_t *c, int *ldc,
                                fb_complex_float_t *work, int *lwork,
                                int *info)
{
    int row = 0;
    int col = 0;
    int tau_len = (*ihi > *ilo) ? (*ihi - *ilo) : 0;

    if (*lwork == -1) {
        g_cunmhr_fortran_call.query_calls += 1;
        g_cunmhr_fortran_call.side = *side;
        g_cunmhr_fortran_call.trans = *trans;
        g_cunmhr_fortran_call.m = *m;
        g_cunmhr_fortran_call.n = *n;
        g_cunmhr_fortran_call.ilo = *ilo;
        g_cunmhr_fortran_call.ihi = *ihi;
        g_cunmhr_fortran_call.lda = *lda;
        g_cunmhr_fortran_call.ldc = *ldc;
        g_cunmhr_fortran_call.lwork_query = *lwork;
        a[0] = make_cfloat(-999.0f);
        if (tau_len > 0) {
            tau[0] = make_cfloat(-777.0f);
        }
        c[0] = make_cfloat(-555.0f);
        work[0] = make_cfloat(7.0f);
        *info = 0;
        return;
    }

    g_cunmhr_fortran_call.solve_calls += 1;
    g_cunmhr_fortran_call.side = *side;
    g_cunmhr_fortran_call.trans = *trans;
    g_cunmhr_fortran_call.m = *m;
    g_cunmhr_fortran_call.n = *n;
    g_cunmhr_fortran_call.ilo = *ilo;
    g_cunmhr_fortran_call.ihi = *ihi;
    g_cunmhr_fortran_call.lda = *lda;
    g_cunmhr_fortran_call.ldc = *ldc;
    g_cunmhr_fortran_call.lwork_solve = *lwork;
    for (col = 0; col < *lda; ++col) {
        for (row = 0; row < *lda; ++row) {
            g_cunmhr_fortran_call.a_real_snapshot[(col * (*lda)) + row] =
                cfloat_real(a[(col * (*lda)) + row]);
        }
    }
    for (col = 0; col < *n; ++col) {
        for (row = 0; row < *m; ++row) {
            g_cunmhr_fortran_call.c_real_snapshot[(col * (*m)) + row] =
                cfloat_real(c[(col * (*ldc)) + row]);
            c[(col * (*ldc)) + row] = make_cfloat((float)(1000 + (10 * col) + row));
        }
    }
    for (row = 0; row < tau_len; ++row) {
        g_cunmhr_fortran_call.tau_real_snapshot[row] = cfloat_real(tau[row]);
    }
    a[0] = make_cfloat(-123.0f);
    if (tau_len > 0) {
        tau[0] = make_cfloat(-456.0f);
    }
    *info = 0;
}

static int stub_cunmhr_cblas(fb_layout_t layout, char side, char trans, int m,
                             int n, int ilo, int ihi,
                             const fb_complex_float_t *a, int lda,
                             const fb_complex_float_t *tau,
                             fb_complex_float_t *c, int ldc)
{
    g_cunmhr_cblas_call.called += 1;
    g_cunmhr_cblas_call.layout = layout;
    g_cunmhr_cblas_call.side = side;
    g_cunmhr_cblas_call.trans = trans;
    g_cunmhr_cblas_call.m = m;
    g_cunmhr_cblas_call.n = n;
    g_cunmhr_cblas_call.ilo = ilo;
    g_cunmhr_cblas_call.ihi = ihi;
    g_cunmhr_cblas_call.lda = lda;
    g_cunmhr_cblas_call.ldc = ldc;
    g_cunmhr_cblas_call.a = a;
    g_cunmhr_cblas_call.tau = tau;
    g_cunmhr_cblas_call.c = c;
    return 93;
}

static int check_sormhr_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_sormhr_fn thunk = NULL;
    float a[9] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f };
    float tau[2] = { 20.0f, 21.0f };
    float c[6] = { 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f };
    float expected_a[9] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f };
    float expected_tau[2] = { 20.0f, 21.0f };
    float expected_c_snapshot[6] = { 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f };
    float expected_c_out[6] = { 900.0f, 901.0f, 902.0f, 910.0f, 911.0f, 912.0f };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sormhr_fortran_call, 0, sizeof(g_sormhr_fortran_call));

    vtable.ext_ops[FB_OP_SORMHR][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_sormhr_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SORMHR);

    thunk = (fb_sormhr_fn)vtable.ext_ops[FB_OP_SORMHR][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SORMHR Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 'L', 'N', 3, 2, 1, 3, a, 3, tau, c, 3);
    if (info != 0 || g_sormhr_fortran_call.query_calls != 1 ||
        g_sormhr_fortran_call.solve_calls != 1 ||
        g_sormhr_fortran_call.side != 'L' ||
        g_sormhr_fortran_call.trans != 'N' ||
        g_sormhr_fortran_call.m != 3 || g_sormhr_fortran_call.n != 2 ||
        g_sormhr_fortran_call.ilo != 1 || g_sormhr_fortran_call.ihi != 3 ||
        g_sormhr_fortran_call.lda != 3 || g_sormhr_fortran_call.ldc != 3 ||
        g_sormhr_fortran_call.lwork_query != -1 ||
        g_sormhr_fortran_call.lwork_solve != 6 ||
        memcmp(g_sormhr_fortran_call.a_snapshot, expected_a, sizeof(expected_a)) != 0 ||
        memcmp(g_sormhr_fortran_call.c_snapshot, expected_c_snapshot,
               sizeof(expected_c_snapshot)) != 0 ||
        memcmp(g_sormhr_fortran_call.tau_snapshot, expected_tau,
               sizeof(expected_tau)) != 0 ||
        memcmp(a, expected_a, sizeof(expected_a)) != 0 ||
        memcmp(tau, expected_tau, sizeof(expected_tau)) != 0 ||
        memcmp(c, expected_c_out, sizeof(expected_c_out)) != 0) {
        fprintf(stderr, "[FAIL] SORMHR Fortran->CBLAS thunk did not preserve column-major Hessenberg apply semantics\n");
        return 1;
    }

    printf("[PASS] SORMHR Fortran->CBLAS thunk runs workspace query on scratch buffers and preserves column-major inputs\n");
    return 0;
}

static int check_sormhr_row_major_rejected(void)
{
    fb_backend_vtable_t vtable;
    fb_sormhr_fn thunk = NULL;
    float a[9] = { 0.0f };
    float tau[2] = { 1.0f, 2.0f };
    float c[6] = { 0.0f };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sormhr_fortran_call, 0, sizeof(g_sormhr_fortran_call));

    vtable.ext_ops[FB_OP_SORMHR][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_sormhr_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SORMHR);

    thunk = (fb_sormhr_fn)vtable.ext_ops[FB_OP_SORMHR][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SORMHR row-major rejection thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 'L', 'N', 3, 2, 1, 3, a, 3, tau, c, 2);
    if (info != -1 || g_sormhr_fortran_call.query_calls != 0 ||
        g_sormhr_fortran_call.solve_calls != 0) {
        fprintf(stderr, "[FAIL] SORMHR generated C bridge did not reject unsupported row-major layout\n");
        return 1;
    }

    printf("[PASS] SORMHR Fortran->CBLAS thunk rejects unsupported row-major layout\n");
    return 0;
}

static int check_sormhr_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_sormhr_fortran_slot_fn thunk = NULL;
    char side = 'L';
    char trans = 'T';
    float a[9] = { 0.0f };
    float tau[2] = { 20.0f, 21.0f };
    float c[6] = { 0.0f };
    float work[4] = { 0.0f };
    int m = 3;
    int n = 2;
    int ilo = 1;
    int ihi = 3;
    int lda = 3;
    int ldc = 3;
    int lwork = 4;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sormhr_cblas_call, 0, sizeof(g_sormhr_cblas_call));

    vtable.ext_ops[FB_OP_SORMHR][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_sormhr_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SORMHR);

    thunk = (fb_sormhr_fortran_slot_fn)vtable.ext_ops[FB_OP_SORMHR][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SORMHR CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&side, &trans, &m, &n, &ilo, &ihi, a, &lda, tau, c, &ldc, work,
          &lwork, &info);
    if (info != 91 || g_sormhr_cblas_call.called != 1 ||
        g_sormhr_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_sormhr_cblas_call.side != 'L' || g_sormhr_cblas_call.trans != 'T' ||
        g_sormhr_cblas_call.m != 3 || g_sormhr_cblas_call.n != 2 ||
        g_sormhr_cblas_call.ilo != 1 || g_sormhr_cblas_call.ihi != 3 ||
        g_sormhr_cblas_call.lda != 3 || g_sormhr_cblas_call.ldc != 3 ||
        g_sormhr_cblas_call.a != a || g_sormhr_cblas_call.tau != tau ||
        g_sormhr_cblas_call.c != c) {
        fprintf(stderr, "[FAIL] SORMHR CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] SORMHR CBLAS->Fortran thunk maps the all-pointer ABI into the generic C Hessenberg entry\n");
    return 0;
}

static int check_cunmhr_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_cunmhr_fn thunk = NULL;
    fb_complex_float_t a[9] = {
        make_cfloat(1.0f), make_cfloat(2.0f), make_cfloat(3.0f),
        make_cfloat(4.0f), make_cfloat(5.0f), make_cfloat(6.0f),
        make_cfloat(7.0f), make_cfloat(8.0f), make_cfloat(9.0f)
    };
    fb_complex_float_t tau[2] = { make_cfloat(30.0f), make_cfloat(31.0f) };
    fb_complex_float_t c[6] = {
        make_cfloat(16.0f), make_cfloat(17.0f), make_cfloat(18.0f),
        make_cfloat(19.0f), make_cfloat(20.0f), make_cfloat(21.0f)
    };
    float expected_a[9] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f };
    float expected_tau[2] = { 30.0f, 31.0f };
    float expected_c_snapshot[6] = { 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f };
    float expected_c_out[6] = { 1000.0f, 1001.0f, 1010.0f, 1011.0f, 1020.0f, 1021.0f };
    int info = 0;
    int idx = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cunmhr_fortran_call, 0, sizeof(g_cunmhr_fortran_call));

    vtable.ext_ops[FB_OP_CUNMHR][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_cunmhr_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CUNMHR);

    thunk = (fb_cunmhr_fn)vtable.ext_ops[FB_OP_CUNMHR][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CUNMHR Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 'R', 'C', 2, 3, 1, 3, a, 3, tau, c, 2);
    if (info != 0 || g_cunmhr_fortran_call.query_calls != 1 ||
        g_cunmhr_fortran_call.solve_calls != 1 ||
        g_cunmhr_fortran_call.side != 'R' ||
        g_cunmhr_fortran_call.trans != 'C' ||
        g_cunmhr_fortran_call.m != 2 || g_cunmhr_fortran_call.n != 3 ||
        g_cunmhr_fortran_call.ilo != 1 || g_cunmhr_fortran_call.ihi != 3 ||
        g_cunmhr_fortran_call.lda != 3 || g_cunmhr_fortran_call.ldc != 2 ||
        g_cunmhr_fortran_call.lwork_query != -1 ||
        g_cunmhr_fortran_call.lwork_solve != 7 ||
        memcmp(g_cunmhr_fortran_call.a_real_snapshot, expected_a, sizeof(expected_a)) != 0 ||
        memcmp(g_cunmhr_fortran_call.c_real_snapshot, expected_c_snapshot,
               sizeof(expected_c_snapshot)) != 0 ||
        memcmp(g_cunmhr_fortran_call.tau_real_snapshot, expected_tau,
               sizeof(expected_tau)) != 0) {
        fprintf(stderr, "[FAIL] CUNMHR Fortran->CBLAS thunk did not preserve complex column-major Hessenberg apply semantics\n");
        return 1;
    }

    for (idx = 0; idx < 9; ++idx) {
        if (cfloat_real(a[idx]) != expected_a[idx]) {
            fprintf(stderr, "[FAIL] CUNMHR Fortran->CBLAS thunk did not preserve complex A input across workspace query\n");
            return 1;
        }
    }
    for (idx = 0; idx < 2; ++idx) {
        if (cfloat_real(tau[idx]) != expected_tau[idx]) {
            fprintf(stderr, "[FAIL] CUNMHR Fortran->CBLAS thunk did not preserve complex TAU input\n");
            return 1;
        }
    }
    for (idx = 0; idx < 6; ++idx) {
        if (cfloat_real(c[idx]) != expected_c_out[idx]) {
            fprintf(stderr, "[FAIL] CUNMHR Fortran->CBLAS thunk did not copy complex output back correctly\n");
            return 1;
        }
    }

    printf("[PASS] CUNMHR Fortran->CBLAS thunk runs complex workspace query on scratch buffers and preserves column-major inputs\n");
    return 0;
}

static int check_cunmhr_row_major_rejected(void)
{
    fb_backend_vtable_t vtable;
    fb_cunmhr_fn thunk = NULL;
    fb_complex_float_t a[9];
    fb_complex_float_t tau[2] = { make_cfloat(1.0f), make_cfloat(2.0f) };
    fb_complex_float_t c[6];
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cunmhr_fortran_call, 0, sizeof(g_cunmhr_fortran_call));
    memset(a, 0, sizeof(a));
    memset(c, 0, sizeof(c));

    vtable.ext_ops[FB_OP_CUNMHR][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_cunmhr_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CUNMHR);

    thunk = (fb_cunmhr_fn)vtable.ext_ops[FB_OP_CUNMHR][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CUNMHR row-major rejection thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 'R', 'C', 2, 3, 1, 3, a, 3, tau, c, 3);
    if (info != -1 || g_cunmhr_fortran_call.query_calls != 0 ||
        g_cunmhr_fortran_call.solve_calls != 0) {
        fprintf(stderr, "[FAIL] CUNMHR generated C bridge did not reject unsupported row-major layout\n");
        return 1;
    }

    printf("[PASS] CUNMHR Fortran->CBLAS thunk rejects unsupported row-major layout\n");
    return 0;
}

static int check_cunmhr_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_cunmhr_fortran_slot_fn thunk = NULL;
    char side = 'R';
    char trans = 'N';
    fb_complex_float_t a[9];
    fb_complex_float_t tau[2] = { make_cfloat(30.0f), make_cfloat(31.0f) };
    fb_complex_float_t c[6];
    fb_complex_float_t work[4];
    int m = 2;
    int n = 3;
    int ilo = 1;
    int ihi = 3;
    int lda = 3;
    int ldc = 2;
    int lwork = 4;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cunmhr_cblas_call, 0, sizeof(g_cunmhr_cblas_call));
    memset(a, 0, sizeof(a));
    memset(c, 0, sizeof(c));
    memset(work, 0, sizeof(work));

    vtable.ext_ops[FB_OP_CUNMHR][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_cunmhr_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CUNMHR);

    thunk = (fb_cunmhr_fortran_slot_fn)vtable.ext_ops[FB_OP_CUNMHR][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CUNMHR CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&side, &trans, &m, &n, &ilo, &ihi, a, &lda, tau, c, &ldc, work,
          &lwork, &info);
    if (info != 93 || g_cunmhr_cblas_call.called != 1 ||
        g_cunmhr_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_cunmhr_cblas_call.side != 'R' || g_cunmhr_cblas_call.trans != 'N' ||
        g_cunmhr_cblas_call.m != 2 || g_cunmhr_cblas_call.n != 3 ||
        g_cunmhr_cblas_call.ilo != 1 || g_cunmhr_cblas_call.ihi != 3 ||
        g_cunmhr_cblas_call.lda != 3 || g_cunmhr_cblas_call.ldc != 2 ||
        g_cunmhr_cblas_call.a != a || g_cunmhr_cblas_call.tau != tau ||
        g_cunmhr_cblas_call.c != c) {
        fprintf(stderr, "[FAIL] CUNMHR CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] CUNMHR CBLAS->Fortran thunk maps the all-pointer ABI into the generic complex C Hessenberg entry\n");
    return 0;
}

int main(void)
{
    if (check_sormhr_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_sormhr_row_major_rejected() != 0) {
        return 1;
    }
    if (check_sormhr_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_cunmhr_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_cunmhr_row_major_rejected() != 0) {
        return 1;
    }
    if (check_cunmhr_cblas_to_fortran() != 0) {
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}