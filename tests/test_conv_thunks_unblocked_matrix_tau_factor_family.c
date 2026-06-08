#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef int (*fb_smatrix_tau_factor_fn)(fb_layout_t layout, int m, int n,
                                        float *a, int lda, float *tau);
typedef int (*fb_cmatrix_tau_factor_fn)(fb_layout_t layout, int m, int n,
                                        fb_complex_float_t *a, int lda,
                                        fb_complex_float_t *tau);

typedef void (*fb_smatrix_tau_factor_fortran_slot_fn)(int *m, int *n, float *a,
                                                      int *lda, float *tau,
                                                      float *work, int *info);
typedef void (*fb_cmatrix_tau_factor_fortran_slot_fn)(int *m, int *n,
                                                      fb_complex_float_t *a,
                                                      int *lda,
                                                      fb_complex_float_t *tau,
                                                      fb_complex_float_t *work,
                                                      int *info);

static struct {
    int calls;
    int m;
    int n;
    int lda;
    int tau_len;
    int work_seen;
    float a_snapshot[12];
    float tau_snapshot[4];
} g_real_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    int m;
    int n;
    int lda;
    float *a;
    float *tau;
} g_real_cblas_call;

static struct {
    int calls;
    int m;
    int n;
    int lda;
    int tau_len;
    int work_seen;
    float a_real_snapshot[12];
    float tau_real_snapshot[4];
} g_complex_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    int m;
    int n;
    int lda;
    fb_complex_float_t *a;
    fb_complex_float_t *tau;
} g_complex_cblas_call;

static int g_real_stub_base = 0;
static int g_real_cblas_rc = 0;
static int g_complex_stub_base = 0;
static int g_complex_cblas_rc = 0;

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

static void stub_real_fortran(int *m, int *n, float *a, int *lda, float *tau,
                              float *work, int *info)
{
    int row = 0;
    int col = 0;

    g_real_fortran_call.calls += 1;
    g_real_fortran_call.m = *m;
    g_real_fortran_call.n = *n;
    g_real_fortran_call.lda = *lda;
    g_real_fortran_call.tau_len = (*m < *n) ? *m : *n;
    g_real_fortran_call.work_seen = work != NULL;

    for (col = 0; col < *n; ++col) {
        for (row = 0; row < *m; ++row) {
            g_real_fortran_call.a_snapshot[(col * (*m)) + row] =
                a[(col * (*lda)) + row];
            a[(col * (*lda)) + row] = (float)(g_real_stub_base + (10 * row) + col);
        }
    }
    for (row = 0; row < g_real_fortran_call.tau_len; ++row) {
        g_real_fortran_call.tau_snapshot[row] = tau[row];
        tau[row] = (float)(g_real_stub_base + 200 + row);
    }
    if (work) {
        work[0] = (float)(g_real_stub_base + 99);
    }
    *info = 0;
}

static int stub_real_cblas(fb_layout_t layout, int m, int n, float *a, int lda,
                           float *tau)
{
    g_real_cblas_call.called += 1;
    g_real_cblas_call.layout = layout;
    g_real_cblas_call.m = m;
    g_real_cblas_call.n = n;
    g_real_cblas_call.lda = lda;
    g_real_cblas_call.a = a;
    g_real_cblas_call.tau = tau;
    return g_real_cblas_rc;
}

static void stub_complex_fortran(int *m, int *n, fb_complex_float_t *a,
                                 int *lda, fb_complex_float_t *tau,
                                 fb_complex_float_t *work, int *info)
{
    int row = 0;
    int col = 0;

    g_complex_fortran_call.calls += 1;
    g_complex_fortran_call.m = *m;
    g_complex_fortran_call.n = *n;
    g_complex_fortran_call.lda = *lda;
    g_complex_fortran_call.tau_len = (*m < *n) ? *m : *n;
    g_complex_fortran_call.work_seen = work != NULL;

    for (col = 0; col < *n; ++col) {
        for (row = 0; row < *m; ++row) {
            g_complex_fortran_call.a_real_snapshot[(col * (*m)) + row] =
                cfloat_real(a[(col * (*lda)) + row]);
            a[(col * (*lda)) + row] =
                make_cfloat((float)(g_complex_stub_base + (10 * row) + col));
        }
    }
    for (row = 0; row < g_complex_fortran_call.tau_len; ++row) {
        g_complex_fortran_call.tau_real_snapshot[row] = cfloat_real(tau[row]);
        tau[row] = make_cfloat((float)(g_complex_stub_base + 200 + row));
    }
    if (work) {
        work[0] = make_cfloat((float)(g_complex_stub_base + 99));
    }
    *info = 0;
}

static int stub_complex_cblas(fb_layout_t layout, int m, int n,
                              fb_complex_float_t *a, int lda,
                              fb_complex_float_t *tau)
{
    g_complex_cblas_call.called += 1;
    g_complex_cblas_call.layout = layout;
    g_complex_cblas_call.m = m;
    g_complex_cblas_call.n = n;
    g_complex_cblas_call.lda = lda;
    g_complex_cblas_call.a = a;
    g_complex_cblas_call.tau = tau;
    return g_complex_cblas_rc;
}

static int check_real_fortran_to_cblas(int op_id, const char *name, int m, int n,
                                       const float *a_in,
                                       const float *expected_a_snapshot,
                                       const float *expected_a_out,
                                       int base)
{
    fb_backend_vtable_t vtable;
    fb_smatrix_tau_factor_fn thunk = NULL;
    float a[12] = { 0.0f };
    float tau[4] = { -1.0f, -2.0f, -3.0f, -4.0f };
    int a_len = m * n;
    int tau_len = (m < n) ? m : n;
    int info = 0;

    memcpy(a, a_in, (size_t)a_len * sizeof(float));
    memset(&vtable, 0, sizeof(vtable));
    memset(&g_real_fortran_call, 0, sizeof(g_real_fortran_call));
    g_real_stub_base = base;

    vtable.ext_ops[op_id][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_real_fortran;
    fb_install_conv_thunks(&vtable, op_id);

    thunk = (fb_smatrix_tau_factor_fn)vtable.ext_ops[op_id][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] %s Fortran->CBLAS thunk was not installed\n", name);
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, m, n, a, n, tau);
    if (info != 0 || g_real_fortran_call.calls != 1 ||
        g_real_fortran_call.m != m || g_real_fortran_call.n != n ||
        g_real_fortran_call.lda != m ||
        g_real_fortran_call.tau_len != tau_len ||
        !g_real_fortran_call.work_seen ||
        memcmp(g_real_fortran_call.a_snapshot, expected_a_snapshot,
               (size_t)a_len * sizeof(float)) != 0 ||
        memcmp(a, expected_a_out, (size_t)a_len * sizeof(float)) != 0) {
        fprintf(stderr, "[FAIL] %s Fortran->CBLAS thunk did not preserve row-major unblocked factorization state\n",
                name);
        return 1;
    }
    for (info = 0; info < tau_len; ++info) {
        if (g_real_fortran_call.tau_snapshot[info] != 0.0f ||
            tau[info] != (float)(base + 200 + info)) {
            fprintf(stderr, "[FAIL] %s Fortran->CBLAS thunk did not round-trip tau output correctly\n",
                    name);
            return 1;
        }
    }

    printf("[PASS] %s Fortran->CBLAS thunk translates row-major unblocked factorization state\n",
           name);
    return 0;
}

static int check_real_cblas_to_fortran(int op_id, const char *name, int m, int n,
                                       int rc)
{
    fb_backend_vtable_t vtable;
    fb_smatrix_tau_factor_fortran_slot_fn thunk = NULL;
    float a[12] = { 0.0f };
    float tau[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    float work[4] = { 0.0f };
    int lda = m;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_real_cblas_call, 0, sizeof(g_real_cblas_call));
    g_real_cblas_rc = rc;

    vtable.ext_ops[op_id][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_real_cblas;
    fb_install_conv_thunks(&vtable, op_id);

    thunk = (fb_smatrix_tau_factor_fortran_slot_fn)
        vtable.ext_ops[op_id][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] %s CBLAS->Fortran thunk was not installed\n", name);
        return 1;
    }

    thunk(&m, &n, a, &lda, tau, work, &info);
    if (info != rc || g_real_cblas_call.called != 1 ||
        g_real_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_real_cblas_call.m != m || g_real_cblas_call.n != n ||
        g_real_cblas_call.lda != lda ||
        g_real_cblas_call.a != a || g_real_cblas_call.tau != tau) {
        fprintf(stderr, "[FAIL] %s CBLAS->Fortran thunk delegated incorrectly\n",
                name);
        return 1;
    }

    printf("[PASS] %s CBLAS->Fortran thunk maps the all-pointer ABI into the generic C factorization entry\n",
           name);
    return 0;
}

static int check_complex_fortran_to_cblas(int op_id, const char *name, int m,
                                          int n,
                                          const fb_complex_float_t *a_in,
                                          const float *expected_a_snapshot,
                                          const float *expected_a_out,
                                          int base)
{
    fb_backend_vtable_t vtable;
    fb_cmatrix_tau_factor_fn thunk = NULL;
    fb_complex_float_t a[12];
    fb_complex_float_t tau[4];
    int a_len = m * n;
    int tau_len = (m < n) ? m : n;
    int info = 0;
    int idx = 0;

    memset(a, 0, sizeof(a));
    memset(tau, 0, sizeof(tau));
    memcpy(a, a_in, (size_t)a_len * sizeof(fb_complex_float_t));
    tau[0] = make_cfloat(-1.0f);
    tau[1] = make_cfloat(-2.0f);
    memset(&vtable, 0, sizeof(vtable));
    memset(&g_complex_fortran_call, 0, sizeof(g_complex_fortran_call));
    g_complex_stub_base = base;

    vtable.ext_ops[op_id][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_complex_fortran;
    fb_install_conv_thunks(&vtable, op_id);

    thunk = (fb_cmatrix_tau_factor_fn)vtable.ext_ops[op_id][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] %s Fortran->CBLAS thunk was not installed\n", name);
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, m, n, a, n, tau);
    if (info != 0 || g_complex_fortran_call.calls != 1 ||
        g_complex_fortran_call.m != m || g_complex_fortran_call.n != n ||
        g_complex_fortran_call.lda != m ||
        g_complex_fortran_call.tau_len != tau_len ||
        !g_complex_fortran_call.work_seen ||
        memcmp(g_complex_fortran_call.a_real_snapshot, expected_a_snapshot,
               (size_t)a_len * sizeof(float)) != 0 ||
         memcmp(g_complex_fortran_call.tau_real_snapshot,
             (float[4]) { 0.0f, 0.0f, 0.0f, 0.0f },
             (size_t)tau_len * sizeof(float)) != 0) {
        fprintf(stderr, "[FAIL] %s Fortran->CBLAS thunk did not preserve complex row-major unblocked factorization state\n",
                name);
        return 1;
    }
    for (idx = 0; idx < a_len; ++idx) {
        if (cfloat_real(a[idx]) != expected_a_out[idx]) {
            fprintf(stderr, "[FAIL] %s Fortran->CBLAS thunk did not copy complex factorization output back correctly\n",
                    name);
            return 1;
        }
    }
    for (idx = 0; idx < tau_len; ++idx) {
        if (cfloat_real(tau[idx]) != (float)(base + 200 + idx)) {
            fprintf(stderr, "[FAIL] %s Fortran->CBLAS thunk did not preserve complex tau output\n",
                    name);
            return 1;
        }
    }

    printf("[PASS] %s Fortran->CBLAS thunk translates complex row-major unblocked factorization state\n",
           name);
    return 0;
}

static int check_complex_cblas_to_fortran(int op_id, const char *name, int m,
                                          int n, int rc)
{
    fb_backend_vtable_t vtable;
    fb_cmatrix_tau_factor_fortran_slot_fn thunk = NULL;
    fb_complex_float_t a[12];
    fb_complex_float_t tau[4];
    fb_complex_float_t work[4];
    int lda = m;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_complex_cblas_call, 0, sizeof(g_complex_cblas_call));
    memset(a, 0, sizeof(a));
    memset(tau, 0, sizeof(tau));
    memset(work, 0, sizeof(work));
    g_complex_cblas_rc = rc;

    vtable.ext_ops[op_id][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_complex_cblas;
    fb_install_conv_thunks(&vtable, op_id);

    thunk = (fb_cmatrix_tau_factor_fortran_slot_fn)
        vtable.ext_ops[op_id][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] %s CBLAS->Fortran thunk was not installed\n", name);
        return 1;
    }

    thunk(&m, &n, a, &lda, tau, work, &info);
    if (info != rc || g_complex_cblas_call.called != 1 ||
        g_complex_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_complex_cblas_call.m != m || g_complex_cblas_call.n != n ||
        g_complex_cblas_call.lda != lda ||
        g_complex_cblas_call.a != a || g_complex_cblas_call.tau != tau) {
        fprintf(stderr, "[FAIL] %s CBLAS->Fortran thunk delegated incorrectly\n",
                name);
        return 1;
    }

    printf("[PASS] %s CBLAS->Fortran thunk maps the all-pointer ABI into the generic complex factorization entry\n",
           name);
    return 0;
}

int main(void)
{
    const float gelq2_a[6] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
    const float gelq2_snapshot[6] = { 1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f };
    const float gelq2_out[6] = { 2500.0f, 2501.0f, 2502.0f,
                                 2510.0f, 2511.0f, 2512.0f };

    const fb_complex_float_t cgelq2_a[6] = {
        make_cfloat(1.0f), make_cfloat(2.0f), make_cfloat(3.0f),
        make_cfloat(4.0f), make_cfloat(5.0f), make_cfloat(6.0f)
    };
    const float cgelq2_snapshot[6] = { 1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f };
    const float cgelq2_out[6] = { 2600.0f, 2601.0f, 2602.0f,
                                  2610.0f, 2611.0f, 2612.0f };

    const float geql2_a[6] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
    const float geql2_snapshot[6] = { 1.0f, 3.0f, 5.0f, 2.0f, 4.0f, 6.0f };
    const float geql2_out[6] = { 2700.0f, 2701.0f,
                                 2710.0f, 2711.0f,
                                 2720.0f, 2721.0f };

    const fb_complex_float_t cgeql2_a[6] = {
        make_cfloat(1.0f), make_cfloat(2.0f), make_cfloat(3.0f),
        make_cfloat(4.0f), make_cfloat(5.0f), make_cfloat(6.0f)
    };
    const float cgeql2_snapshot[6] = { 1.0f, 3.0f, 5.0f, 2.0f, 4.0f, 6.0f };
    const float cgeql2_out[6] = { 2800.0f, 2801.0f,
                                  2810.0f, 2811.0f,
                                  2820.0f, 2821.0f };

    const float gerq2_a[8] = { 1.0f, 2.0f, 3.0f, 4.0f,
                               5.0f, 6.0f, 7.0f, 8.0f };
    const float gerq2_snapshot[8] = { 1.0f, 5.0f, 2.0f, 6.0f,
                                      3.0f, 7.0f, 4.0f, 8.0f };
    const float gerq2_out[8] = { 2900.0f, 2901.0f, 2902.0f, 2903.0f,
                                 2910.0f, 2911.0f, 2912.0f, 2913.0f };

    const fb_complex_float_t cgerq2_a[8] = {
        make_cfloat(1.0f), make_cfloat(2.0f), make_cfloat(3.0f), make_cfloat(4.0f),
        make_cfloat(5.0f), make_cfloat(6.0f), make_cfloat(7.0f), make_cfloat(8.0f)
    };
    const float cgerq2_snapshot[8] = { 1.0f, 5.0f, 2.0f, 6.0f,
                                       3.0f, 7.0f, 4.0f, 8.0f };
    const float cgerq2_out[8] = { 3000.0f, 3001.0f, 3002.0f, 3003.0f,
                                  3010.0f, 3011.0f, 3012.0f, 3013.0f };

    if (check_real_fortran_to_cblas(FB_OP_SGELQ2, "SGELQ2", 2, 3,
                                    gelq2_a, gelq2_snapshot,
                                    gelq2_out, 2500) != 0) {
        return 1;
    }
    if (check_real_cblas_to_fortran(FB_OP_SGELQ2, "SGELQ2", 2, 3, 131) != 0) {
        return 1;
    }
    if (check_complex_fortran_to_cblas(FB_OP_CGELQ2, "CGELQ2", 2, 3,
                                       cgelq2_a, cgelq2_snapshot,
                                       cgelq2_out, 2600) != 0) {
        return 1;
    }
    if (check_complex_cblas_to_fortran(FB_OP_CGELQ2, "CGELQ2", 2, 3, 133) != 0) {
        return 1;
    }

    if (check_real_fortran_to_cblas(FB_OP_SGEQL2, "SGEQL2", 3, 2,
                                    geql2_a, geql2_snapshot,
                                    geql2_out, 2700) != 0) {
        return 1;
    }
    if (check_real_cblas_to_fortran(FB_OP_SGEQL2, "SGEQL2", 3, 2, 135) != 0) {
        return 1;
    }
    if (check_complex_fortran_to_cblas(FB_OP_CGEQL2, "CGEQL2", 3, 2,
                                       cgeql2_a, cgeql2_snapshot,
                                       cgeql2_out, 2800) != 0) {
        return 1;
    }
    if (check_complex_cblas_to_fortran(FB_OP_CGEQL2, "CGEQL2", 3, 2, 137) != 0) {
        return 1;
    }

    if (check_real_fortran_to_cblas(FB_OP_SGERQ2, "SGERQ2", 2, 4,
                                    gerq2_a, gerq2_snapshot,
                                    gerq2_out, 2900) != 0) {
        return 1;
    }
    if (check_real_cblas_to_fortran(FB_OP_SGERQ2, "SGERQ2", 2, 4, 139) != 0) {
        return 1;
    }
    if (check_complex_fortran_to_cblas(FB_OP_CGERQ2, "CGERQ2", 2, 4,
                                       cgerq2_a, cgerq2_snapshot,
                                       cgerq2_out, 3000) != 0) {
        return 1;
    }
    if (check_complex_cblas_to_fortran(FB_OP_CGERQ2, "CGERQ2", 2, 4, 141) != 0) {
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}