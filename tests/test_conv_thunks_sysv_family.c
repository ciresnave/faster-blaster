#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef void (*fb_ssysv_fortran_slot_fn)(char *uplo, int *n, int *nrhs,
                                         float *a, int *lda, int *ipiv,
                                         float *b, int *ldb, float *work,
                                         int *lwork, int *info);
typedef void (*fb_dsysv_fortran_slot_fn)(char *uplo, int *n, int *nrhs,
                                         double *a, int *lda, int *ipiv,
                                         double *b, int *ldb, double *work,
                                         int *lwork, int *info);
typedef void (*fb_chesv_fortran_slot_fn)(char *uplo, int *n, int *nrhs,
                                         fb_complex_float_t *a, int *lda,
                                         int *ipiv, fb_complex_float_t *b,
                                         int *ldb, fb_complex_float_t *work,
                                         int *lwork, int *info);
typedef void (*fb_zhesv_fortran_slot_fn)(char *uplo, int *n, int *nrhs,
                                         fb_complex_double_t *a, int *lda,
                                         int *ipiv, fb_complex_double_t *b,
                                         int *ldb, fb_complex_double_t *work,
                                         int *lwork, int *info);

static struct {
    int query_calls;
    int solve_calls;
    char uplo;
    int n;
    int nrhs;
    int lda;
    int ldb;
    int lwork_query;
    int lwork_solve;
    int *ipiv;
    float a_snapshot[9];
    float b_snapshot[6];
} g_ssysv_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    char uplo;
    int n;
    int nrhs;
    int lda;
    int ldb;
    int *ipiv;
    float *a;
    float *b;
} g_ssysv_cblas_call;

static struct {
    int query_calls;
    int solve_calls;
    char uplo;
    int n;
    int nrhs;
    int lda;
    int ldb;
    int lwork_query;
    int lwork_solve;
    int *ipiv;
    double a_snapshot[9];
    double b_snapshot[6];
} g_dsysv_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    char uplo;
    int n;
    int nrhs;
    int lda;
    int ldb;
    int *ipiv;
    double *a;
    double *b;
} g_dsysv_cblas_call;

static struct {
    int query_calls;
    int solve_calls;
    char uplo;
    int n;
    int nrhs;
    int lda;
    int ldb;
    int lwork_query;
    int lwork_solve;
    int *ipiv;
    float a_real_snapshot[9];
    float b_real_snapshot[6];
} g_chesv_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    char uplo;
    int n;
    int nrhs;
    int lda;
    int ldb;
    int *ipiv;
    fb_complex_float_t *a;
    fb_complex_float_t *b;
} g_chesv_cblas_call;

static struct {
    int query_calls;
    int solve_calls;
    char uplo;
    int n;
    int nrhs;
    int lda;
    int ldb;
    int lwork_query;
    int lwork_solve;
    int *ipiv;
    double a_real_snapshot[9];
    double b_real_snapshot[6];
} g_zhesv_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    char uplo;
    int n;
    int nrhs;
    int lda;
    int ldb;
    int *ipiv;
    fb_complex_double_t *a;
    fb_complex_double_t *b;
} g_zhesv_cblas_call;

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

static void stub_ssysv_fortran(char *uplo, int *n, int *nrhs, float *a,
                               int *lda, int *ipiv, float *b, int *ldb,
                               float *work, int *lwork, int *info)
{
    int row = 0;
    int col = 0;

    if (*lwork == -1) {
        g_ssysv_fortran_call.query_calls += 1;
        g_ssysv_fortran_call.uplo = *uplo;
        g_ssysv_fortran_call.n = *n;
        g_ssysv_fortran_call.nrhs = *nrhs;
        g_ssysv_fortran_call.lda = *lda;
        g_ssysv_fortran_call.ldb = *ldb;
        g_ssysv_fortran_call.lwork_query = *lwork;
        g_ssysv_fortran_call.ipiv = ipiv;
        a[0] = -999.0f;
        b[0] = -888.0f;
        work[0] = 7.0f;
        *info = 0;
        return;
    }

    g_ssysv_fortran_call.solve_calls += 1;
    g_ssysv_fortran_call.uplo = *uplo;
    g_ssysv_fortran_call.n = *n;
    g_ssysv_fortran_call.nrhs = *nrhs;
    g_ssysv_fortran_call.lda = *lda;
    g_ssysv_fortran_call.ldb = *ldb;
    g_ssysv_fortran_call.lwork_solve = *lwork;
    g_ssysv_fortran_call.ipiv = ipiv;

    for (col = 0; col < *n; ++col) {
        for (row = 0; row < *n; ++row) {
            g_ssysv_fortran_call.a_snapshot[(col * (*n)) + row] =
                a[(col * (*lda)) + row];
            a[(col * (*lda)) + row] = (float)(100 + (10 * row) + col);
        }
    }
    for (col = 0; col < *nrhs; ++col) {
        for (row = 0; row < *n; ++row) {
            g_ssysv_fortran_call.b_snapshot[(col * (*n)) + row] =
                b[(col * (*ldb)) + row];
            b[(col * (*ldb)) + row] = (float)(200 + (10 * row) + col);
        }
    }

    ipiv[0] = 3;
    ipiv[1] = 2;
    ipiv[2] = 1;
    *info = 0;
}

static int stub_ssysv_cblas(const fb_layout_t layout, char uplo,
                            const int n, const int nrhs, float *a,
                            const int lda, int *ipiv, float *b,
                            const int ldb)
{
    g_ssysv_cblas_call.called += 1;
    g_ssysv_cblas_call.layout = layout;
    g_ssysv_cblas_call.uplo = uplo;
    g_ssysv_cblas_call.n = n;
    g_ssysv_cblas_call.nrhs = nrhs;
    g_ssysv_cblas_call.lda = lda;
    g_ssysv_cblas_call.ldb = ldb;
    g_ssysv_cblas_call.ipiv = ipiv;
    g_ssysv_cblas_call.a = a;
    g_ssysv_cblas_call.b = b;
    return 41;
}

static void stub_dsysv_fortran(char *uplo, int *n, int *nrhs, double *a,
                               int *lda, int *ipiv, double *b, int *ldb,
                               double *work, int *lwork, int *info)
{
    int row = 0;
    int col = 0;

    if (*lwork == -1) {
        g_dsysv_fortran_call.query_calls += 1;
        g_dsysv_fortran_call.uplo = *uplo;
        g_dsysv_fortran_call.n = *n;
        g_dsysv_fortran_call.nrhs = *nrhs;
        g_dsysv_fortran_call.lda = *lda;
        g_dsysv_fortran_call.ldb = *ldb;
        g_dsysv_fortran_call.lwork_query = *lwork;
        g_dsysv_fortran_call.ipiv = ipiv;
        a[0] = -1999.0;
        b[0] = -1888.0;
        work[0] = 11.0;
        *info = 0;
        return;
    }

    g_dsysv_fortran_call.solve_calls += 1;
    g_dsysv_fortran_call.uplo = *uplo;
    g_dsysv_fortran_call.n = *n;
    g_dsysv_fortran_call.nrhs = *nrhs;
    g_dsysv_fortran_call.lda = *lda;
    g_dsysv_fortran_call.ldb = *ldb;
    g_dsysv_fortran_call.lwork_solve = *lwork;
    g_dsysv_fortran_call.ipiv = ipiv;

    for (col = 0; col < *n; ++col) {
        for (row = 0; row < *n; ++row) {
            g_dsysv_fortran_call.a_snapshot[(col * (*n)) + row] =
                a[(col * (*lda)) + row];
            a[(col * (*lda)) + row] = (double)(500 + (10 * row) + col);
        }
    }
    for (col = 0; col < *nrhs; ++col) {
        for (row = 0; row < *n; ++row) {
            g_dsysv_fortran_call.b_snapshot[(col * (*n)) + row] =
                b[(col * (*ldb)) + row];
            b[(col * (*ldb)) + row] = (double)(600 + (10 * row) + col);
        }
    }

    ipiv[0] = 3;
    ipiv[1] = 1;
    ipiv[2] = 2;
    *info = 0;
}

static int stub_dsysv_cblas(const fb_layout_t layout, char uplo,
                            const int n, const int nrhs, double *a,
                            const int lda, int *ipiv, double *b,
                            const int ldb)
{
    g_dsysv_cblas_call.called += 1;
    g_dsysv_cblas_call.layout = layout;
    g_dsysv_cblas_call.uplo = uplo;
    g_dsysv_cblas_call.n = n;
    g_dsysv_cblas_call.nrhs = nrhs;
    g_dsysv_cblas_call.lda = lda;
    g_dsysv_cblas_call.ldb = ldb;
    g_dsysv_cblas_call.ipiv = ipiv;
    g_dsysv_cblas_call.a = a;
    g_dsysv_cblas_call.b = b;
    return 51;
}

static void stub_chesv_fortran(char *uplo, int *n, int *nrhs,
                               fb_complex_float_t *a, int *lda, int *ipiv,
                               fb_complex_float_t *b, int *ldb,
                               fb_complex_float_t *work, int *lwork,
                               int *info)
{
    int row = 0;
    int col = 0;

    if (*lwork == -1) {
        g_chesv_fortran_call.query_calls += 1;
        g_chesv_fortran_call.uplo = *uplo;
        g_chesv_fortran_call.n = *n;
        g_chesv_fortran_call.nrhs = *nrhs;
        g_chesv_fortran_call.lda = *lda;
        g_chesv_fortran_call.ldb = *ldb;
        g_chesv_fortran_call.lwork_query = *lwork;
        g_chesv_fortran_call.ipiv = ipiv;
        a[0] = make_cfloat(-999.0f);
        b[0] = make_cfloat(-888.0f);
        work[0] = make_cfloat(9.0f);
        *info = 0;
        return;
    }

    g_chesv_fortran_call.solve_calls += 1;
    g_chesv_fortran_call.uplo = *uplo;
    g_chesv_fortran_call.n = *n;
    g_chesv_fortran_call.nrhs = *nrhs;
    g_chesv_fortran_call.lda = *lda;
    g_chesv_fortran_call.ldb = *ldb;
    g_chesv_fortran_call.lwork_solve = *lwork;
    g_chesv_fortran_call.ipiv = ipiv;

    for (col = 0; col < *n; ++col) {
        for (row = 0; row < *n; ++row) {
            g_chesv_fortran_call.a_real_snapshot[(col * (*n)) + row] =
                cfloat_real(a[(col * (*lda)) + row]);
            a[(col * (*lda)) + row] = make_cfloat((float)(300 + (10 * row) + col));
        }
    }
    for (col = 0; col < *nrhs; ++col) {
        for (row = 0; row < *n; ++row) {
            g_chesv_fortran_call.b_real_snapshot[(col * (*n)) + row] =
                cfloat_real(b[(col * (*ldb)) + row]);
            b[(col * (*ldb)) + row] = make_cfloat((float)(400 + (10 * row) + col));
        }
    }

    ipiv[0] = 1;
    ipiv[1] = 3;
    ipiv[2] = 2;
    *info = 0;
}

static int stub_chesv_cblas(const fb_layout_t layout, char uplo,
                            const int n, const int nrhs,
                            fb_complex_float_t *a, const int lda, int *ipiv,
                            fb_complex_float_t *b, const int ldb)
{
    g_chesv_cblas_call.called += 1;
    g_chesv_cblas_call.layout = layout;
    g_chesv_cblas_call.uplo = uplo;
    g_chesv_cblas_call.n = n;
    g_chesv_cblas_call.nrhs = nrhs;
    g_chesv_cblas_call.lda = lda;
    g_chesv_cblas_call.ldb = ldb;
    g_chesv_cblas_call.ipiv = ipiv;
    g_chesv_cblas_call.a = a;
    g_chesv_cblas_call.b = b;
    return 43;
}

static void stub_zhesv_fortran(char *uplo, int *n, int *nrhs,
                               fb_complex_double_t *a, int *lda, int *ipiv,
                               fb_complex_double_t *b, int *ldb,
                               fb_complex_double_t *work, int *lwork,
                               int *info)
{
    int row = 0;
    int col = 0;

    if (*lwork == -1) {
        g_zhesv_fortran_call.query_calls += 1;
        g_zhesv_fortran_call.uplo = *uplo;
        g_zhesv_fortran_call.n = *n;
        g_zhesv_fortran_call.nrhs = *nrhs;
        g_zhesv_fortran_call.lda = *lda;
        g_zhesv_fortran_call.ldb = *ldb;
        g_zhesv_fortran_call.lwork_query = *lwork;
        g_zhesv_fortran_call.ipiv = ipiv;
        a[0] = make_cdouble(-2999.0);
        b[0] = make_cdouble(-2888.0);
        work[0] = make_cdouble(13.0);
        *info = 0;
        return;
    }

    g_zhesv_fortran_call.solve_calls += 1;
    g_zhesv_fortran_call.uplo = *uplo;
    g_zhesv_fortran_call.n = *n;
    g_zhesv_fortran_call.nrhs = *nrhs;
    g_zhesv_fortran_call.lda = *lda;
    g_zhesv_fortran_call.ldb = *ldb;
    g_zhesv_fortran_call.lwork_solve = *lwork;
    g_zhesv_fortran_call.ipiv = ipiv;

    for (col = 0; col < *n; ++col) {
        for (row = 0; row < *n; ++row) {
            g_zhesv_fortran_call.a_real_snapshot[(col * (*n)) + row] =
                cdouble_real(a[(col * (*lda)) + row]);
            a[(col * (*lda)) + row] = make_cdouble((double)(700 + (10 * row) + col));
        }
    }
    for (col = 0; col < *nrhs; ++col) {
        for (row = 0; row < *n; ++row) {
            g_zhesv_fortran_call.b_real_snapshot[(col * (*n)) + row] =
                cdouble_real(b[(col * (*ldb)) + row]);
            b[(col * (*ldb)) + row] = make_cdouble((double)(800 + (10 * row) + col));
        }
    }

    ipiv[0] = 2;
    ipiv[1] = 3;
    ipiv[2] = 1;
    *info = 0;
}

static int stub_zhesv_cblas(const fb_layout_t layout, char uplo,
                            const int n, const int nrhs,
                            fb_complex_double_t *a, const int lda, int *ipiv,
                            fb_complex_double_t *b, const int ldb)
{
    g_zhesv_cblas_call.called += 1;
    g_zhesv_cblas_call.layout = layout;
    g_zhesv_cblas_call.uplo = uplo;
    g_zhesv_cblas_call.n = n;
    g_zhesv_cblas_call.nrhs = nrhs;
    g_zhesv_cblas_call.lda = lda;
    g_zhesv_cblas_call.ldb = ldb;
    g_zhesv_cblas_call.ipiv = ipiv;
    g_zhesv_cblas_call.a = a;
    g_zhesv_cblas_call.b = b;
    return 53;
}

static int check_ssysv_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_ssysv_fn thunk = NULL;
    float a[9] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f
    };
    float b[6] = { 11.0f, 12.0f, 21.0f, 22.0f, 31.0f, 32.0f };
    float expected_a_snapshot[9] = { 1.0f, 4.0f, 7.0f, 2.0f, 5.0f, 8.0f, 3.0f, 6.0f, 9.0f };
    float expected_b_snapshot[6] = { 11.0f, 21.0f, 31.0f, 12.0f, 22.0f, 32.0f };
    float expected_a_out[9] = { 100.0f, 101.0f, 102.0f, 110.0f, 111.0f, 112.0f, 120.0f, 121.0f, 122.0f };
    float expected_b_out[6] = { 200.0f, 201.0f, 210.0f, 211.0f, 220.0f, 221.0f };
    int ipiv[3] = { 0, 0, 0 };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_ssysv_fortran_call, 0, sizeof(g_ssysv_fortran_call));

    vtable.ext_ops[FB_OP_SSYSV][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_ssysv_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SSYSV);

    thunk = (fb_ssysv_fn)vtable.ext_ops[FB_OP_SSYSV][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SSYSV Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 'U', 3, 2, a, 3, ipiv, b, 2);
    if (info != 0 || g_ssysv_fortran_call.query_calls != 1 ||
        g_ssysv_fortran_call.solve_calls != 1 || g_ssysv_fortran_call.uplo != 'L' ||
        g_ssysv_fortran_call.n != 3 || g_ssysv_fortran_call.nrhs != 2 ||
        g_ssysv_fortran_call.lda != 3 || g_ssysv_fortran_call.ldb != 3 ||
        g_ssysv_fortran_call.lwork_query != -1 ||
        g_ssysv_fortran_call.lwork_solve != 7 ||
        g_ssysv_fortran_call.ipiv != ipiv ||
        memcmp(g_ssysv_fortran_call.a_snapshot, expected_a_snapshot,
               sizeof(expected_a_snapshot)) != 0 ||
        memcmp(g_ssysv_fortran_call.b_snapshot, expected_b_snapshot,
               sizeof(expected_b_snapshot)) != 0 ||
        memcmp(a, expected_a_out, sizeof(expected_a_out)) != 0 ||
        memcmp(b, expected_b_out, sizeof(expected_b_out)) != 0 ||
        ipiv[0] != 3 || ipiv[1] != 2 || ipiv[2] != 1) {
        fprintf(stderr, "[FAIL] SSYSV Fortran->CBLAS thunk did not preserve workspace-query row-major semantics\n");
        return 1;
    }

    printf("[PASS] SSYSV Fortran->CBLAS thunk translates row-major A/B and reruns after lwork query\n");
    return 0;
}

static int check_ssysv_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_ssysv_fortran_slot_fn thunk = NULL;
    float a[9] = { 0.0f };
    float b[6] = { 0.0f };
    int ipiv[3] = { 0, 0, 0 };
    int n = 3;
    int nrhs = 2;
    int lda = 3;
    int ldb = 3;
    int lwork = 5;
    int info = 0;
    char uplo = 'l';
    float work[5] = { 0.0f };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_ssysv_cblas_call, 0, sizeof(g_ssysv_cblas_call));

    vtable.ext_ops[FB_OP_SSYSV][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_ssysv_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SSYSV);

    thunk = (fb_ssysv_fortran_slot_fn)vtable.ext_ops[FB_OP_SSYSV][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SSYSV CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&uplo, &n, &nrhs, a, &lda, ipiv, b, &ldb, work, &lwork, &info);
    if (info != 41 || g_ssysv_cblas_call.called != 1 ||
        g_ssysv_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_ssysv_cblas_call.uplo != 'L' || g_ssysv_cblas_call.n != 3 ||
        g_ssysv_cblas_call.nrhs != 2 || g_ssysv_cblas_call.lda != 3 ||
        g_ssysv_cblas_call.ldb != 3 || g_ssysv_cblas_call.ipiv != ipiv ||
        g_ssysv_cblas_call.a != a || g_ssysv_cblas_call.b != b) {
        fprintf(stderr, "[FAIL] SSYSV CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] SSYSV CBLAS->Fortran thunk maps the all-pointer ABI into the C solver entry\n");
    return 0;
}

static int check_dsysv_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_dsysv_fn thunk = NULL;
    double a[9] = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0
    };
    double b[6] = { 11.0, 12.0, 21.0, 22.0, 31.0, 32.0 };
    double expected_a_snapshot[9] = { 1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0 };
    double expected_b_snapshot[6] = { 11.0, 21.0, 31.0, 12.0, 22.0, 32.0 };
    double expected_a_out[9] = { 500.0, 501.0, 502.0, 510.0, 511.0, 512.0, 520.0, 521.0, 522.0 };
    double expected_b_out[6] = { 600.0, 601.0, 610.0, 611.0, 620.0, 621.0 };
    int ipiv[3] = { 0, 0, 0 };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dsysv_fortran_call, 0, sizeof(g_dsysv_fortran_call));

    vtable.ext_ops[FB_OP_DSYSV][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_dsysv_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_DSYSV);

    thunk = (fb_dsysv_fn)vtable.ext_ops[FB_OP_DSYSV][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DSYSV Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 'U', 3, 2, a, 3, ipiv, b, 2);
    if (info != 0 || g_dsysv_fortran_call.query_calls != 1 ||
        g_dsysv_fortran_call.solve_calls != 1 || g_dsysv_fortran_call.uplo != 'L' ||
        g_dsysv_fortran_call.n != 3 || g_dsysv_fortran_call.nrhs != 2 ||
        g_dsysv_fortran_call.lda != 3 || g_dsysv_fortran_call.ldb != 3 ||
        g_dsysv_fortran_call.lwork_query != -1 ||
        g_dsysv_fortran_call.lwork_solve != 11 ||
        g_dsysv_fortran_call.ipiv != ipiv ||
        memcmp(g_dsysv_fortran_call.a_snapshot, expected_a_snapshot,
               sizeof(expected_a_snapshot)) != 0 ||
        memcmp(g_dsysv_fortran_call.b_snapshot, expected_b_snapshot,
               sizeof(expected_b_snapshot)) != 0 ||
        memcmp(a, expected_a_out, sizeof(expected_a_out)) != 0 ||
        memcmp(b, expected_b_out, sizeof(expected_b_out)) != 0 ||
        ipiv[0] != 3 || ipiv[1] != 1 || ipiv[2] != 2) {
        fprintf(stderr, "[FAIL] DSYSV Fortran->CBLAS thunk did not preserve workspace-query row-major semantics\n");
        return 1;
    }

    printf("[PASS] DSYSV Fortran->CBLAS thunk translates row-major A/B and reruns after lwork query\n");
    return 0;
}

static int check_dsysv_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_dsysv_fortran_slot_fn thunk = NULL;
    double a[9] = { 0.0 };
    double b[6] = { 0.0 };
    int ipiv[3] = { 0, 0, 0 };
    int n = 3;
    int nrhs = 2;
    int lda = 3;
    int ldb = 3;
    int lwork = 5;
    int info = 0;
    char uplo = 'u';
    double work[5] = { 0.0 };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dsysv_cblas_call, 0, sizeof(g_dsysv_cblas_call));

    vtable.ext_ops[FB_OP_DSYSV][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_dsysv_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_DSYSV);

    thunk = (fb_dsysv_fortran_slot_fn)vtable.ext_ops[FB_OP_DSYSV][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DSYSV CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&uplo, &n, &nrhs, a, &lda, ipiv, b, &ldb, work, &lwork, &info);
    if (info != 51 || g_dsysv_cblas_call.called != 1 ||
        g_dsysv_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_dsysv_cblas_call.uplo != 'U' || g_dsysv_cblas_call.n != 3 ||
        g_dsysv_cblas_call.nrhs != 2 || g_dsysv_cblas_call.lda != 3 ||
        g_dsysv_cblas_call.ldb != 3 || g_dsysv_cblas_call.ipiv != ipiv ||
        g_dsysv_cblas_call.a != a || g_dsysv_cblas_call.b != b) {
        fprintf(stderr, "[FAIL] DSYSV CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] DSYSV CBLAS->Fortran thunk maps the all-pointer ABI into the double C solver entry\n");
    return 0;
}

static int check_chesv_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_chesv_fn thunk = NULL;
    fb_complex_float_t a[9] = {
        make_cfloat(1.0f), make_cfloat(2.0f), make_cfloat(3.0f),
        make_cfloat(4.0f), make_cfloat(5.0f), make_cfloat(6.0f),
        make_cfloat(7.0f), make_cfloat(8.0f), make_cfloat(9.0f)
    };
    fb_complex_float_t b[6] = {
        make_cfloat(11.0f), make_cfloat(12.0f),
        make_cfloat(21.0f), make_cfloat(22.0f),
        make_cfloat(31.0f), make_cfloat(32.0f)
    };
    float expected_a_snapshot[9] = { 1.0f, 4.0f, 7.0f, 2.0f, 5.0f, 8.0f, 3.0f, 6.0f, 9.0f };
    float expected_b_snapshot[6] = { 11.0f, 21.0f, 31.0f, 12.0f, 22.0f, 32.0f };
    float expected_a_out[9] = { 300.0f, 301.0f, 302.0f, 310.0f, 311.0f, 312.0f, 320.0f, 321.0f, 322.0f };
    float expected_b_out[6] = { 400.0f, 401.0f, 410.0f, 411.0f, 420.0f, 421.0f };
    int ipiv[3] = { 0, 0, 0 };
    int info = 0;
    int idx = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_chesv_fortran_call, 0, sizeof(g_chesv_fortran_call));

    vtable.ext_ops[FB_OP_CHESV][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_chesv_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CHESV);

    thunk = (fb_chesv_fn)vtable.ext_ops[FB_OP_CHESV][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CHESV Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 'L', 3, 2, a, 3, ipiv, b, 2);
    if (info != 0 || g_chesv_fortran_call.query_calls != 1 ||
        g_chesv_fortran_call.solve_calls != 1 || g_chesv_fortran_call.uplo != 'U' ||
        g_chesv_fortran_call.n != 3 || g_chesv_fortran_call.nrhs != 2 ||
        g_chesv_fortran_call.lda != 3 || g_chesv_fortran_call.ldb != 3 ||
        g_chesv_fortran_call.lwork_query != -1 ||
        g_chesv_fortran_call.lwork_solve != 9 ||
        g_chesv_fortran_call.ipiv != ipiv ||
        memcmp(g_chesv_fortran_call.a_real_snapshot, expected_a_snapshot,
               sizeof(expected_a_snapshot)) != 0 ||
        memcmp(g_chesv_fortran_call.b_real_snapshot, expected_b_snapshot,
               sizeof(expected_b_snapshot)) != 0 ||
        ipiv[0] != 1 || ipiv[1] != 3 || ipiv[2] != 2) {
        fprintf(stderr, "[FAIL] CHESV Fortran->CBLAS thunk did not preserve complex workspace-query row-major semantics\n");
        return 1;
    }

    for (idx = 0; idx < 9; ++idx) {
        if (cfloat_real(a[idx]) != expected_a_out[idx]) {
            fprintf(stderr, "[FAIL] CHESV Fortran->CBLAS thunk did not copy row-major A output back correctly\n");
            return 1;
        }
    }
    for (idx = 0; idx < 6; ++idx) {
        if (cfloat_real(b[idx]) != expected_b_out[idx]) {
            fprintf(stderr, "[FAIL] CHESV Fortran->CBLAS thunk did not copy row-major B output back correctly\n");
            return 1;
        }
    }

    printf("[PASS] CHESV Fortran->CBLAS thunk translates row-major A/B and extracts complex lwork query size\n");
    return 0;
}

static int check_chesv_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_chesv_fortran_slot_fn thunk = NULL;
    fb_complex_float_t a[9];
    fb_complex_float_t b[6];
    int ipiv[3] = { 0, 0, 0 };
    int n = 3;
    int nrhs = 2;
    int lda = 3;
    int ldb = 3;
    int lwork = 5;
    int info = 0;
    char uplo = 'u';
    fb_complex_float_t work[5];

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_chesv_cblas_call, 0, sizeof(g_chesv_cblas_call));
    memset(a, 0, sizeof(a));
    memset(b, 0, sizeof(b));
    memset(work, 0, sizeof(work));

    vtable.ext_ops[FB_OP_CHESV][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_chesv_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CHESV);

    thunk = (fb_chesv_fortran_slot_fn)vtable.ext_ops[FB_OP_CHESV][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CHESV CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&uplo, &n, &nrhs, a, &lda, ipiv, b, &ldb, work, &lwork, &info);
    if (info != 43 || g_chesv_cblas_call.called != 1 ||
        g_chesv_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_chesv_cblas_call.uplo != 'U' || g_chesv_cblas_call.n != 3 ||
        g_chesv_cblas_call.nrhs != 2 || g_chesv_cblas_call.lda != 3 ||
        g_chesv_cblas_call.ldb != 3 || g_chesv_cblas_call.ipiv != ipiv ||
        g_chesv_cblas_call.a != a || g_chesv_cblas_call.b != b) {
        fprintf(stderr, "[FAIL] CHESV CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] CHESV CBLAS->Fortran thunk maps the all-pointer ABI into the complex C solver entry\n");
    return 0;
}

static int check_zhesv_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_zhesv_fn thunk = NULL;
    fb_complex_double_t a[9] = {
        make_cdouble(1.0), make_cdouble(2.0), make_cdouble(3.0),
        make_cdouble(4.0), make_cdouble(5.0), make_cdouble(6.0),
        make_cdouble(7.0), make_cdouble(8.0), make_cdouble(9.0)
    };
    fb_complex_double_t b[6] = {
        make_cdouble(11.0), make_cdouble(12.0),
        make_cdouble(21.0), make_cdouble(22.0),
        make_cdouble(31.0), make_cdouble(32.0)
    };
    double expected_a_snapshot[9] = { 1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0 };
    double expected_b_snapshot[6] = { 11.0, 21.0, 31.0, 12.0, 22.0, 32.0 };
    double expected_a_out[9] = { 700.0, 701.0, 702.0, 710.0, 711.0, 712.0, 720.0, 721.0, 722.0 };
    double expected_b_out[6] = { 800.0, 801.0, 810.0, 811.0, 820.0, 821.0 };
    int ipiv[3] = { 0, 0, 0 };
    int info = 0;
    int idx = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zhesv_fortran_call, 0, sizeof(g_zhesv_fortran_call));

    vtable.ext_ops[FB_OP_ZHESV][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_zhesv_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_ZHESV);

    thunk = (fb_zhesv_fn)vtable.ext_ops[FB_OP_ZHESV][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZHESV Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 'L', 3, 2, a, 3, ipiv, b, 2);
    if (info != 0 || g_zhesv_fortran_call.query_calls != 1 ||
        g_zhesv_fortran_call.solve_calls != 1 || g_zhesv_fortran_call.uplo != 'U' ||
        g_zhesv_fortran_call.n != 3 || g_zhesv_fortran_call.nrhs != 2 ||
        g_zhesv_fortran_call.lda != 3 || g_zhesv_fortran_call.ldb != 3 ||
        g_zhesv_fortran_call.lwork_query != -1 ||
        g_zhesv_fortran_call.lwork_solve != 13 ||
        g_zhesv_fortran_call.ipiv != ipiv ||
        memcmp(g_zhesv_fortran_call.a_real_snapshot, expected_a_snapshot,
               sizeof(expected_a_snapshot)) != 0 ||
        memcmp(g_zhesv_fortran_call.b_real_snapshot, expected_b_snapshot,
               sizeof(expected_b_snapshot)) != 0 ||
        ipiv[0] != 2 || ipiv[1] != 3 || ipiv[2] != 1) {
        fprintf(stderr, "[FAIL] ZHESV Fortran->CBLAS thunk did not preserve complex-double workspace-query row-major semantics\n");
        return 1;
    }

    for (idx = 0; idx < 9; ++idx) {
        if (cdouble_real(a[idx]) != expected_a_out[idx]) {
            fprintf(stderr, "[FAIL] ZHESV Fortran->CBLAS thunk did not copy row-major A output back correctly\n");
            return 1;
        }
    }
    for (idx = 0; idx < 6; ++idx) {
        if (cdouble_real(b[idx]) != expected_b_out[idx]) {
            fprintf(stderr, "[FAIL] ZHESV Fortran->CBLAS thunk did not copy row-major B output back correctly\n");
            return 1;
        }
    }

    printf("[PASS] ZHESV Fortran->CBLAS thunk translates row-major A/B and extracts complex-double lwork query size\n");
    return 0;
}

static int check_zhesv_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_zhesv_fortran_slot_fn thunk = NULL;
    fb_complex_double_t a[9];
    fb_complex_double_t b[6];
    int ipiv[3] = { 0, 0, 0 };
    int n = 3;
    int nrhs = 2;
    int lda = 3;
    int ldb = 3;
    int lwork = 5;
    int info = 0;
    char uplo = 'l';
    fb_complex_double_t work[5];

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zhesv_cblas_call, 0, sizeof(g_zhesv_cblas_call));
    memset(a, 0, sizeof(a));
    memset(b, 0, sizeof(b));
    memset(work, 0, sizeof(work));

    vtable.ext_ops[FB_OP_ZHESV][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_zhesv_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_ZHESV);

    thunk = (fb_zhesv_fortran_slot_fn)vtable.ext_ops[FB_OP_ZHESV][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZHESV CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&uplo, &n, &nrhs, a, &lda, ipiv, b, &ldb, work, &lwork, &info);
    if (info != 53 || g_zhesv_cblas_call.called != 1 ||
        g_zhesv_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_zhesv_cblas_call.uplo != 'L' || g_zhesv_cblas_call.n != 3 ||
        g_zhesv_cblas_call.nrhs != 2 || g_zhesv_cblas_call.lda != 3 ||
        g_zhesv_cblas_call.ldb != 3 || g_zhesv_cblas_call.ipiv != ipiv ||
        g_zhesv_cblas_call.a != a || g_zhesv_cblas_call.b != b) {
        fprintf(stderr, "[FAIL] ZHESV CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] ZHESV CBLAS->Fortran thunk maps the all-pointer ABI into the complex-double C solver entry\n");
    return 0;
}

int main(void)
{
    if (check_ssysv_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_ssysv_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_dsysv_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_dsysv_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_chesv_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_chesv_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_zhesv_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_zhesv_cblas_to_fortran() != 0) {
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}