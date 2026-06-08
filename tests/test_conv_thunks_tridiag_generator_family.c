#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef int (*fb_sorgtr_fn)(fb_layout_t layout, char uplo, int n, float *a,
                            int lda, const float *tau);
typedef int (*fb_cungtr_fn)(fb_layout_t layout, char uplo, int n,
                            fb_complex_float_t *a, int lda,
                            const fb_complex_float_t *tau);

typedef void (*fb_sorgtr_fortran_slot_fn)(char *uplo, int *n, float *a,
                                          int *lda, float *tau, float *work,
                                          int *lwork, int *info);
typedef void (*fb_cungtr_fortran_slot_fn)(char *uplo, int *n,
                                          fb_complex_float_t *a, int *lda,
                                          fb_complex_float_t *tau,
                                          fb_complex_float_t *work,
                                          int *lwork, int *info);

static struct {
    int query_calls;
    int solve_calls;
    char uplo;
    int n;
    int lda;
    int lwork_query;
    int lwork_solve;
    float a_snapshot[9];
    float tau_snapshot[2];
} g_sorgtr_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    char uplo;
    int n;
    int lda;
    float *a;
    const float *tau;
} g_sorgtr_cblas_call;

static struct {
    int query_calls;
    int solve_calls;
    char uplo;
    int n;
    int lda;
    int lwork_query;
    int lwork_solve;
    float a_real_snapshot[9];
    float tau_real_snapshot[2];
} g_cungtr_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    char uplo;
    int n;
    int lda;
    fb_complex_float_t *a;
    const fb_complex_float_t *tau;
} g_cungtr_cblas_call;

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

static void stub_sorgtr_fortran(char *uplo, int *n, float *a, int *lda,
                                float *tau, float *work, int *lwork,
                                int *info)
{
    int row = 0;
    int col = 0;
    int tau_len = (*n > 1) ? (*n - 1) : 0;

    if (*lwork == -1) {
        g_sorgtr_fortran_call.query_calls += 1;
        g_sorgtr_fortran_call.uplo = *uplo;
        g_sorgtr_fortran_call.n = *n;
        g_sorgtr_fortran_call.lda = *lda;
        g_sorgtr_fortran_call.lwork_query = *lwork;
        a[0] = -999.0f;
        if (tau_len > 0) {
            tau[0] = -777.0f;
        }
        work[0] = 5.0f;
        *info = 0;
        return;
    }

    g_sorgtr_fortran_call.solve_calls += 1;
    g_sorgtr_fortran_call.uplo = *uplo;
    g_sorgtr_fortran_call.n = *n;
    g_sorgtr_fortran_call.lda = *lda;
    g_sorgtr_fortran_call.lwork_solve = *lwork;
    for (col = 0; col < *n; ++col) {
        for (row = 0; row < *n; ++row) {
            g_sorgtr_fortran_call.a_snapshot[(col * (*n)) + row] =
                a[(col * (*lda)) + row];
            a[(col * (*lda)) + row] = (float)(1300 + (10 * col) + row);
        }
    }
    for (row = 0; row < tau_len; ++row) {
        g_sorgtr_fortran_call.tau_snapshot[row] = tau[row];
    }
    *info = 0;
}

static int stub_sorgtr_cblas(fb_layout_t layout, char uplo, int n, float *a,
                             int lda, const float *tau)
{
    g_sorgtr_cblas_call.called += 1;
    g_sorgtr_cblas_call.layout = layout;
    g_sorgtr_cblas_call.uplo = uplo;
    g_sorgtr_cblas_call.n = n;
    g_sorgtr_cblas_call.lda = lda;
    g_sorgtr_cblas_call.a = a;
    g_sorgtr_cblas_call.tau = tau;
    return 101;
}

static void stub_cungtr_fortran(char *uplo, int *n, fb_complex_float_t *a,
                                int *lda, fb_complex_float_t *tau,
                                fb_complex_float_t *work, int *lwork,
                                int *info)
{
    int row = 0;
    int col = 0;
    int tau_len = (*n > 1) ? (*n - 1) : 0;

    if (*lwork == -1) {
        g_cungtr_fortran_call.query_calls += 1;
        g_cungtr_fortran_call.uplo = *uplo;
        g_cungtr_fortran_call.n = *n;
        g_cungtr_fortran_call.lda = *lda;
        g_cungtr_fortran_call.lwork_query = *lwork;
        a[0] = make_cfloat(-999.0f);
        if (tau_len > 0) {
            tau[0] = make_cfloat(-777.0f);
        }
        work[0] = make_cfloat(6.0f);
        *info = 0;
        return;
    }

    g_cungtr_fortran_call.solve_calls += 1;
    g_cungtr_fortran_call.uplo = *uplo;
    g_cungtr_fortran_call.n = *n;
    g_cungtr_fortran_call.lda = *lda;
    g_cungtr_fortran_call.lwork_solve = *lwork;
    for (col = 0; col < *n; ++col) {
        for (row = 0; row < *n; ++row) {
            g_cungtr_fortran_call.a_real_snapshot[(col * (*n)) + row] =
                cfloat_real(a[(col * (*lda)) + row]);
            a[(col * (*lda)) + row] = make_cfloat((float)(1400 + (10 * col) + row));
        }
    }
    for (row = 0; row < tau_len; ++row) {
        g_cungtr_fortran_call.tau_real_snapshot[row] = cfloat_real(tau[row]);
    }
    *info = 0;
}

static int stub_cungtr_cblas(fb_layout_t layout, char uplo, int n,
                             fb_complex_float_t *a, int lda,
                             const fb_complex_float_t *tau)
{
    g_cungtr_cblas_call.called += 1;
    g_cungtr_cblas_call.layout = layout;
    g_cungtr_cblas_call.uplo = uplo;
    g_cungtr_cblas_call.n = n;
    g_cungtr_cblas_call.lda = lda;
    g_cungtr_cblas_call.a = a;
    g_cungtr_cblas_call.tau = tau;
    return 103;
}

static int check_sorgtr_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_sorgtr_fn thunk = NULL;
    float a[9] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f };
    float tau[2] = { 60.0f, 61.0f };
    float expected_a_snapshot[9] = { 1.0f, 4.0f, 7.0f, 2.0f, 5.0f, 8.0f, 3.0f, 6.0f, 9.0f };
    float expected_tau[2] = { 60.0f, 61.0f };
    float expected_a_out[9] = { 1300.0f, 1310.0f, 1320.0f, 1301.0f, 1311.0f, 1321.0f, 1302.0f, 1312.0f, 1322.0f };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sorgtr_fortran_call, 0, sizeof(g_sorgtr_fortran_call));

    vtable.ext_ops[FB_OP_SORGTR][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_sorgtr_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SORGTR);

    thunk = (fb_sorgtr_fn)vtable.ext_ops[FB_OP_SORGTR][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SORGTR Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 'U', 3, a, 3, tau);
    if (info != 0 || g_sorgtr_fortran_call.query_calls != 1 ||
        g_sorgtr_fortran_call.solve_calls != 1 ||
        g_sorgtr_fortran_call.uplo != 'L' ||
        g_sorgtr_fortran_call.n != 3 || g_sorgtr_fortran_call.lda != 3 ||
        g_sorgtr_fortran_call.lwork_query != -1 ||
        g_sorgtr_fortran_call.lwork_solve != 5 ||
        memcmp(g_sorgtr_fortran_call.a_snapshot, expected_a_snapshot,
               sizeof(expected_a_snapshot)) != 0 ||
        memcmp(g_sorgtr_fortran_call.tau_snapshot, expected_tau,
               sizeof(expected_tau)) != 0 ||
        memcmp(a, expected_a_out, sizeof(expected_a_out)) != 0 ||
        memcmp(tau, expected_tau, sizeof(expected_tau)) != 0) {
        fprintf(stderr, "[FAIL] SORGTR Fortran->CBLAS thunk did not preserve UPLO-remapped row-major generator semantics\n");
        return 1;
    }

    printf("[PASS] SORGTR Fortran->CBLAS thunk remaps UPLO for row-major generator matrices and preserves TAU across lwork query\n");
    return 0;
}

static int check_sorgtr_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_sorgtr_fortran_slot_fn thunk = NULL;
    char uplo = 'L';
    float a[9] = { 0.0f };
    float tau[2] = { 60.0f, 61.0f };
    float work[4] = { 0.0f };
    int n = 3;
    int lda = 3;
    int lwork = 4;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sorgtr_cblas_call, 0, sizeof(g_sorgtr_cblas_call));

    vtable.ext_ops[FB_OP_SORGTR][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_sorgtr_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SORGTR);

    thunk = (fb_sorgtr_fortran_slot_fn)vtable.ext_ops[FB_OP_SORGTR][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SORGTR CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&uplo, &n, a, &lda, tau, work, &lwork, &info);
    if (info != 101 || g_sorgtr_cblas_call.called != 1 ||
        g_sorgtr_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_sorgtr_cblas_call.uplo != 'L' ||
        g_sorgtr_cblas_call.n != 3 || g_sorgtr_cblas_call.lda != 3 ||
        g_sorgtr_cblas_call.a != a || g_sorgtr_cblas_call.tau != tau) {
        fprintf(stderr, "[FAIL] SORGTR CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] SORGTR CBLAS->Fortran thunk maps the all-pointer ABI into the generic C tridiagonal generator entry\n");
    return 0;
}

static int check_cungtr_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_cungtr_fn thunk = NULL;
    fb_complex_float_t a[9] = {
        make_cfloat(1.0f), make_cfloat(2.0f), make_cfloat(3.0f),
        make_cfloat(4.0f), make_cfloat(5.0f), make_cfloat(6.0f),
        make_cfloat(7.0f), make_cfloat(8.0f), make_cfloat(9.0f)
    };
    fb_complex_float_t tau[2] = { make_cfloat(70.0f), make_cfloat(71.0f) };
    float expected_a_snapshot[9] = { 1.0f, 4.0f, 7.0f, 2.0f, 5.0f, 8.0f, 3.0f, 6.0f, 9.0f };
    float expected_tau[2] = { 70.0f, 71.0f };
    float expected_a_out[9] = { 1400.0f, 1410.0f, 1420.0f, 1401.0f, 1411.0f, 1421.0f, 1402.0f, 1412.0f, 1422.0f };
    int info = 0;
    int idx = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cungtr_fortran_call, 0, sizeof(g_cungtr_fortran_call));

    vtable.ext_ops[FB_OP_CUNGTR][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_cungtr_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CUNGTR);

    thunk = (fb_cungtr_fn)vtable.ext_ops[FB_OP_CUNGTR][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CUNGTR Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 'L', 3, a, 3, tau);
    if (info != 0 || g_cungtr_fortran_call.query_calls != 1 ||
        g_cungtr_fortran_call.solve_calls != 1 ||
        g_cungtr_fortran_call.uplo != 'U' ||
        g_cungtr_fortran_call.n != 3 || g_cungtr_fortran_call.lda != 3 ||
        g_cungtr_fortran_call.lwork_query != -1 ||
        g_cungtr_fortran_call.lwork_solve != 6 ||
        memcmp(g_cungtr_fortran_call.a_real_snapshot, expected_a_snapshot,
               sizeof(expected_a_snapshot)) != 0 ||
        memcmp(g_cungtr_fortran_call.tau_real_snapshot, expected_tau,
               sizeof(expected_tau)) != 0) {
        fprintf(stderr, "[FAIL] CUNGTR Fortran->CBLAS thunk did not preserve complex UPLO-remapped row-major generator semantics\n");
        return 1;
    }

    for (idx = 0; idx < 9; ++idx) {
        if (cfloat_real(a[idx]) != expected_a_out[idx]) {
            fprintf(stderr, "[FAIL] CUNGTR Fortran->CBLAS thunk did not copy complex tridiagonal generator output back correctly\n");
            return 1;
        }
    }
    for (idx = 0; idx < 2; ++idx) {
        if (cfloat_real(tau[idx]) != expected_tau[idx]) {
            fprintf(stderr, "[FAIL] CUNGTR Fortran->CBLAS thunk did not preserve complex TAU input\n");
            return 1;
        }
    }

    printf("[PASS] CUNGTR Fortran->CBLAS thunk remaps UPLO for row-major complex generator matrices and preserves TAU across lwork query\n");
    return 0;
}

static int check_cungtr_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_cungtr_fortran_slot_fn thunk = NULL;
    char uplo = 'U';
    fb_complex_float_t a[9];
    fb_complex_float_t tau[2] = { make_cfloat(70.0f), make_cfloat(71.0f) };
    fb_complex_float_t work[4];
    int n = 3;
    int lda = 3;
    int lwork = 4;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cungtr_cblas_call, 0, sizeof(g_cungtr_cblas_call));
    memset(a, 0, sizeof(a));
    memset(work, 0, sizeof(work));

    vtable.ext_ops[FB_OP_CUNGTR][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_cungtr_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CUNGTR);

    thunk = (fb_cungtr_fortran_slot_fn)vtable.ext_ops[FB_OP_CUNGTR][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CUNGTR CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&uplo, &n, a, &lda, tau, work, &lwork, &info);
    if (info != 103 || g_cungtr_cblas_call.called != 1 ||
        g_cungtr_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_cungtr_cblas_call.uplo != 'U' ||
        g_cungtr_cblas_call.n != 3 || g_cungtr_cblas_call.lda != 3 ||
        g_cungtr_cblas_call.a != a || g_cungtr_cblas_call.tau != tau) {
        fprintf(stderr, "[FAIL] CUNGTR CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] CUNGTR CBLAS->Fortran thunk maps the all-pointer ABI into the generic complex tridiagonal generator entry\n");
    return 0;
}

int main(void)
{
    if (check_sorgtr_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_sorgtr_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_cungtr_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_cungtr_cblas_to_fortran() != 0) {
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}