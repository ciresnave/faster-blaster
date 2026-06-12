#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef void (*fb_sgesv_fortran_slot_fn)(int *n, int *nrhs, float *a, int *lda,
                                         int *ipiv, float *b, int *ldb,
                                         int *info);
typedef void (*fb_dgesv_fortran_slot_fn)(int *n, int *nrhs, double *a, int *lda,
                                         int *ipiv, double *b, int *ldb,
                                         int *info);
typedef void (*fb_cgesv_fortran_slot_fn)(int *n, int *nrhs,
                                         fb_complex_float_t *a, int *lda,
                                         int *ipiv, fb_complex_float_t *b,
                                         int *ldb, int *info);
typedef void (*fb_zgesv_fortran_slot_fn)(int *n, int *nrhs,
                                         fb_complex_double_t *a, int *lda,
                                         int *ipiv, fb_complex_double_t *b,
                                         int *ldb, int *info);

static struct {
    int called;
    fb_layout_t layout;
    int n;
    int nrhs;
    int lda;
    int ldb;
    float *a;
    int *ipiv;
    float *b;
} g_sgesv_cblas_call;

static struct {
    int called;
    int n;
    int nrhs;
    int lda;
    int ldb;
    float *a;
    int *ipiv;
    float *b;
} g_sgesv_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    int n;
    int nrhs;
    int lda;
    int ldb;
    double *a;
    int *ipiv;
    double *b;
} g_dgesv_cblas_call;

static struct {
    int called;
    int n;
    int nrhs;
    int lda;
    int ldb;
    double *a;
    int *ipiv;
    double *b;
} g_dgesv_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    int n;
    int nrhs;
    int lda;
    int ldb;
    fb_complex_float_t *a;
    int *ipiv;
    fb_complex_float_t *b;
} g_cgesv_cblas_call;

static struct {
    int called;
    int n;
    int nrhs;
    int lda;
    int ldb;
    fb_complex_float_t *a;
    int *ipiv;
    fb_complex_float_t *b;
} g_cgesv_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    int n;
    int nrhs;
    int lda;
    int ldb;
    fb_complex_double_t *a;
    int *ipiv;
    fb_complex_double_t *b;
} g_zgesv_cblas_call;

static struct {
    int called;
    int n;
    int nrhs;
    int lda;
    int ldb;
    fb_complex_double_t *a;
    int *ipiv;
    fb_complex_double_t *b;
} g_zgesv_fortran_call;

static int stub_sgesv_cblas(const fb_layout_t layout, const int n,
                            const int nrhs, float *a, const int lda,
                            int *ipiv, float *b, const int ldb)
{
    g_sgesv_cblas_call.called += 1;
    g_sgesv_cblas_call.layout = layout;
    g_sgesv_cblas_call.n = n;
    g_sgesv_cblas_call.nrhs = nrhs;
    g_sgesv_cblas_call.lda = lda;
    g_sgesv_cblas_call.ldb = ldb;
    g_sgesv_cblas_call.a = a;
    g_sgesv_cblas_call.ipiv = ipiv;
    g_sgesv_cblas_call.b = b;
    return 37;
}

static int stub_dgesv_cblas(const fb_layout_t layout, const int n,
                            const int nrhs, double *a, const int lda,
                            int *ipiv, double *b, const int ldb)
{
    g_dgesv_cblas_call.called += 1;
    g_dgesv_cblas_call.layout = layout;
    g_dgesv_cblas_call.n = n;
    g_dgesv_cblas_call.nrhs = nrhs;
    g_dgesv_cblas_call.lda = lda;
    g_dgesv_cblas_call.ldb = ldb;
    g_dgesv_cblas_call.a = a;
    g_dgesv_cblas_call.ipiv = ipiv;
    g_dgesv_cblas_call.b = b;
    return 137;
}

static void stub_sgesv_fortran(int *n, int *nrhs, float *a, int *lda,
                               int *ipiv, float *b, int *ldb, int *info)
{
    g_sgesv_fortran_call.called += 1;
    g_sgesv_fortran_call.n = *n;
    g_sgesv_fortran_call.nrhs = *nrhs;
    g_sgesv_fortran_call.lda = *lda;
    g_sgesv_fortran_call.ldb = *ldb;
    g_sgesv_fortran_call.a = a;
    g_sgesv_fortran_call.ipiv = ipiv;
    g_sgesv_fortran_call.b = b;
    *info = 53;
}

static void stub_dgesv_fortran(int *n, int *nrhs, double *a, int *lda,
                               int *ipiv, double *b, int *ldb, int *info)
{
    g_dgesv_fortran_call.called += 1;
    g_dgesv_fortran_call.n = *n;
    g_dgesv_fortran_call.nrhs = *nrhs;
    g_dgesv_fortran_call.lda = *lda;
    g_dgesv_fortran_call.ldb = *ldb;
    g_dgesv_fortran_call.a = a;
    g_dgesv_fortran_call.ipiv = ipiv;
    g_dgesv_fortran_call.b = b;
    *info = 153;
}

static int stub_cgesv_cblas(const fb_layout_t layout, const int n,
                            const int nrhs, fb_complex_float_t *a,
                            const int lda, int *ipiv,
                            fb_complex_float_t *b, const int ldb)
{
    g_cgesv_cblas_call.called += 1;
    g_cgesv_cblas_call.layout = layout;
    g_cgesv_cblas_call.n = n;
    g_cgesv_cblas_call.nrhs = nrhs;
    g_cgesv_cblas_call.lda = lda;
    g_cgesv_cblas_call.ldb = ldb;
    g_cgesv_cblas_call.a = a;
    g_cgesv_cblas_call.ipiv = ipiv;
    g_cgesv_cblas_call.b = b;
    return 41;
}

static int stub_zgesv_cblas(const fb_layout_t layout, const int n,
                            const int nrhs, fb_complex_double_t *a,
                            const int lda, int *ipiv,
                            fb_complex_double_t *b, const int ldb)
{
    g_zgesv_cblas_call.called += 1;
    g_zgesv_cblas_call.layout = layout;
    g_zgesv_cblas_call.n = n;
    g_zgesv_cblas_call.nrhs = nrhs;
    g_zgesv_cblas_call.lda = lda;
    g_zgesv_cblas_call.ldb = ldb;
    g_zgesv_cblas_call.a = a;
    g_zgesv_cblas_call.ipiv = ipiv;
    g_zgesv_cblas_call.b = b;
    return 141;
}

static void stub_cgesv_fortran(int *n, int *nrhs, fb_complex_float_t *a,
                               int *lda, int *ipiv, fb_complex_float_t *b,
                               int *ldb, int *info)
{
    g_cgesv_fortran_call.called += 1;
    g_cgesv_fortran_call.n = *n;
    g_cgesv_fortran_call.nrhs = *nrhs;
    g_cgesv_fortran_call.lda = *lda;
    g_cgesv_fortran_call.ldb = *ldb;
    g_cgesv_fortran_call.a = a;
    g_cgesv_fortran_call.ipiv = ipiv;
    g_cgesv_fortran_call.b = b;
    *info = 59;
}

static void stub_zgesv_fortran(int *n, int *nrhs, fb_complex_double_t *a,
                               int *lda, int *ipiv, fb_complex_double_t *b,
                               int *ldb, int *info)
{
    g_zgesv_fortran_call.called += 1;
    g_zgesv_fortran_call.n = *n;
    g_zgesv_fortran_call.nrhs = *nrhs;
    g_zgesv_fortran_call.lda = *lda;
    g_zgesv_fortran_call.ldb = *ldb;
    g_zgesv_fortran_call.a = a;
    g_zgesv_fortran_call.ipiv = ipiv;
    g_zgesv_fortran_call.b = b;
    *info = 159;
}

static int check_sgesv_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_sgesv_fn thunk = NULL;
    float a[16] = { 0.0f };
    float b[8] = { 0.0f };
    int ipiv[4] = { 0 };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sgesv_cblas_call, 0, sizeof(g_sgesv_cblas_call));

    vtable.ext_ops[FB_OP_SGESV][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_sgesv_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SGESV);

    thunk = (fb_sgesv_fn)vtable.ext_ops[FB_OP_SGESV][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SGESV Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 4, 2, a, 4, ipiv, b, 4);
    if (info != 53 || g_sgesv_fortran_call.called != 1 ||
        g_sgesv_fortran_call.n != 4 || g_sgesv_fortran_call.nrhs != 2 ||
        g_sgesv_fortran_call.lda != 4 || g_sgesv_fortran_call.ldb != 4 ||
        g_sgesv_fortran_call.a != a || g_sgesv_fortran_call.ipiv != ipiv ||
        g_sgesv_fortran_call.b != b) {
        fprintf(stderr, "[FAIL] SGESV Fortran->CBLAS thunk delegated incorrectly\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 4, 2, a, 4, ipiv, b, 4);
    if (info != -1 || g_sgesv_fortran_call.called != 1) {
        fprintf(stderr, "[FAIL] SGESV Fortran->CBLAS thunk accepted unsupported row-major layout\n");
        return 1;
    }

    printf("[PASS] SGESV Fortran->CBLAS thunk delegates with C bridge ABI\n");
    return 0;
}

static int check_sgesv_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_sgesv_fortran_slot_fn thunk = NULL;
    float a[16] = { 0.0f };
    float b[8] = { 0.0f };
    int ipiv[4] = { 0 };
    int n = 4;
    int nrhs = 2;
    int lda = 4;
    int ldb = 4;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sgesv_cblas_call, 0, sizeof(g_sgesv_cblas_call));

    vtable.ext_ops[FB_OP_SGESV][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_sgesv_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SGESV);

    thunk = (fb_sgesv_fortran_slot_fn)vtable.ext_ops[FB_OP_SGESV][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SGESV CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
    if (info != 37 || g_sgesv_cblas_call.called != 1 ||
        g_sgesv_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_sgesv_cblas_call.n != 4 || g_sgesv_cblas_call.nrhs != 2 ||
        g_sgesv_cblas_call.lda != 4 || g_sgesv_cblas_call.ldb != 4 ||
        g_sgesv_cblas_call.a != a || g_sgesv_cblas_call.ipiv != ipiv ||
        g_sgesv_cblas_call.b != b) {
        fprintf(stderr, "[FAIL] SGESV CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] SGESV CBLAS->Fortran thunk delegates with pointer ABI\n");
    return 0;
}

static int check_dgesv_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_dgesv_fn thunk = NULL;
    double a[16] = { 0.0 };
    double b[8] = { 0.0 };
    int ipiv[4] = { 0 };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dgesv_fortran_call, 0, sizeof(g_dgesv_fortran_call));

    vtable.ext_ops[FB_OP_DGESV][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_dgesv_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_DGESV);

    thunk = (fb_dgesv_fn)vtable.ext_ops[FB_OP_DGESV][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DGESV Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 4, 2, a, 4, ipiv, b, 4);
    if (info != 153 || g_dgesv_fortran_call.called != 1 ||
        g_dgesv_fortran_call.n != 4 || g_dgesv_fortran_call.nrhs != 2 ||
        g_dgesv_fortran_call.lda != 4 || g_dgesv_fortran_call.ldb != 4 ||
        g_dgesv_fortran_call.a != a || g_dgesv_fortran_call.ipiv != ipiv ||
        g_dgesv_fortran_call.b != b) {
        fprintf(stderr, "[FAIL] DGESV Fortran->CBLAS thunk delegated incorrectly\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 4, 2, a, 4, ipiv, b, 4);
    if (info != -1 || g_dgesv_fortran_call.called != 1) {
        fprintf(stderr, "[FAIL] DGESV Fortran->CBLAS thunk accepted unsupported row-major layout\n");
        return 1;
    }

    printf("[PASS] DGESV Fortran->CBLAS thunk delegates with C bridge ABI\n");
    return 0;
}

static int check_dgesv_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_dgesv_fortran_slot_fn thunk = NULL;
    double a[16] = { 0.0 };
    double b[8] = { 0.0 };
    int ipiv[4] = { 0 };
    int n = 4;
    int nrhs = 2;
    int lda = 4;
    int ldb = 4;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dgesv_cblas_call, 0, sizeof(g_dgesv_cblas_call));

    vtable.ext_ops[FB_OP_DGESV][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_dgesv_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_DGESV);

    thunk = (fb_dgesv_fortran_slot_fn)vtable.ext_ops[FB_OP_DGESV][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DGESV CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
    if (info != 137 || g_dgesv_cblas_call.called != 1 ||
        g_dgesv_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_dgesv_cblas_call.n != 4 || g_dgesv_cblas_call.nrhs != 2 ||
        g_dgesv_cblas_call.lda != 4 || g_dgesv_cblas_call.ldb != 4 ||
        g_dgesv_cblas_call.a != a || g_dgesv_cblas_call.ipiv != ipiv ||
        g_dgesv_cblas_call.b != b) {
        fprintf(stderr, "[FAIL] DGESV CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] DGESV CBLAS->Fortran thunk delegates with pointer ABI\n");
    return 0;
}

static int check_cgesv_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_cgesv_fn thunk = NULL;
    fb_complex_float_t a[16];
    fb_complex_float_t b[8];
    int ipiv[4] = { 0 };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cgesv_fortran_call, 0, sizeof(g_cgesv_fortran_call));
    memset(a, 0, sizeof(a));
    memset(b, 0, sizeof(b));

    vtable.ext_ops[FB_OP_CGESV][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_cgesv_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CGESV);

    thunk = (fb_cgesv_fn)vtable.ext_ops[FB_OP_CGESV][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CGESV Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 4, 2, a, 4, ipiv, b, 4);
    if (info != 59 || g_cgesv_fortran_call.called != 1 ||
        g_cgesv_fortran_call.n != 4 || g_cgesv_fortran_call.nrhs != 2 ||
        g_cgesv_fortran_call.lda != 4 || g_cgesv_fortran_call.ldb != 4 ||
        g_cgesv_fortran_call.a != a || g_cgesv_fortran_call.ipiv != ipiv ||
        g_cgesv_fortran_call.b != b) {
        fprintf(stderr, "[FAIL] CGESV Fortran->CBLAS thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] CGESV Fortran->CBLAS thunk delegates with C bridge ABI\n");
    return 0;
}

static int check_cgesv_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_cgesv_fortran_slot_fn thunk = NULL;
    fb_complex_float_t a[16];
    fb_complex_float_t b[8];
    int ipiv[4] = { 0 };
    int n = 4;
    int nrhs = 2;
    int lda = 4;
    int ldb = 4;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cgesv_cblas_call, 0, sizeof(g_cgesv_cblas_call));
    memset(a, 0, sizeof(a));
    memset(b, 0, sizeof(b));

    vtable.ext_ops[FB_OP_CGESV][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_cgesv_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CGESV);

    thunk = (fb_cgesv_fortran_slot_fn)vtable.ext_ops[FB_OP_CGESV][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CGESV CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
    if (info != 41 || g_cgesv_cblas_call.called != 1 ||
        g_cgesv_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_cgesv_cblas_call.n != 4 || g_cgesv_cblas_call.nrhs != 2 ||
        g_cgesv_cblas_call.lda != 4 || g_cgesv_cblas_call.ldb != 4 ||
        g_cgesv_cblas_call.a != a || g_cgesv_cblas_call.ipiv != ipiv ||
        g_cgesv_cblas_call.b != b) {
        fprintf(stderr, "[FAIL] CGESV CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] CGESV CBLAS->Fortran thunk delegates with pointer ABI\n");
    return 0;
}

static int check_zgesv_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_zgesv_fn thunk = NULL;
    fb_complex_double_t a[16];
    fb_complex_double_t b[8];
    int ipiv[4] = { 0 };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zgesv_fortran_call, 0, sizeof(g_zgesv_fortran_call));
    memset(a, 0, sizeof(a));
    memset(b, 0, sizeof(b));

    vtable.ext_ops[FB_OP_ZGESV][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_zgesv_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_ZGESV);

    thunk = (fb_zgesv_fn)vtable.ext_ops[FB_OP_ZGESV][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZGESV Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 4, 2, a, 4, ipiv, b, 4);
    if (info != 159 || g_zgesv_fortran_call.called != 1 ||
        g_zgesv_fortran_call.n != 4 || g_zgesv_fortran_call.nrhs != 2 ||
        g_zgesv_fortran_call.lda != 4 || g_zgesv_fortran_call.ldb != 4 ||
        g_zgesv_fortran_call.a != a || g_zgesv_fortran_call.ipiv != ipiv ||
        g_zgesv_fortran_call.b != b) {
        fprintf(stderr, "[FAIL] ZGESV Fortran->CBLAS thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] ZGESV Fortran->CBLAS thunk delegates with C bridge ABI\n");
    return 0;
}

static int check_zgesv_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_zgesv_fortran_slot_fn thunk = NULL;
    fb_complex_double_t a[16];
    fb_complex_double_t b[8];
    int ipiv[4] = { 0 };
    int n = 4;
    int nrhs = 2;
    int lda = 4;
    int ldb = 4;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zgesv_cblas_call, 0, sizeof(g_zgesv_cblas_call));
    memset(a, 0, sizeof(a));
    memset(b, 0, sizeof(b));

    vtable.ext_ops[FB_OP_ZGESV][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_zgesv_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_ZGESV);

    thunk = (fb_zgesv_fortran_slot_fn)vtable.ext_ops[FB_OP_ZGESV][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZGESV CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
    if (info != 141 || g_zgesv_cblas_call.called != 1 ||
        g_zgesv_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_zgesv_cblas_call.n != 4 || g_zgesv_cblas_call.nrhs != 2 ||
        g_zgesv_cblas_call.lda != 4 || g_zgesv_cblas_call.ldb != 4 ||
        g_zgesv_cblas_call.a != a || g_zgesv_cblas_call.ipiv != ipiv ||
        g_zgesv_cblas_call.b != b) {
        fprintf(stderr, "[FAIL] ZGESV CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] ZGESV CBLAS->Fortran thunk delegates with pointer ABI\n");
    return 0;
}

int main(void)
{
    if (check_sgesv_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_sgesv_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_dgesv_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_dgesv_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_cgesv_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_cgesv_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_zgesv_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_zgesv_cblas_to_fortran() != 0) {
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}