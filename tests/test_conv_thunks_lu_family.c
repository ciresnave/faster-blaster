#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef void (*fb_sgetrf_fortran_slot_fn)(int *m, int *n, float *a, int *lda,
                                          int *ipiv, int *info);
typedef void (*fb_cgetrf_fortran_slot_fn)(int *m, int *n,
                                          fb_complex_float_t *a, int *lda,
                                          int *ipiv, int *info);
typedef void (*fb_sgetrs_fortran_slot_fn)(char *trans, int *n, int *nrhs,
                                          float *a, int *lda, int *ipiv,
                                          float *b, int *ldb, int *info);
typedef void (*fb_cgetrs_fortran_slot_fn)(char *trans, int *n, int *nrhs,
                                          fb_complex_float_t *a, int *lda,
                                          int *ipiv, fb_complex_float_t *b,
                                          int *ldb, int *info);

static struct {
    int called;
    fb_layout_t layout;
    int m;
    int n;
    int lda;
    float *a;
    int *ipiv;
} g_sgetrf_cblas_call;

static struct {
    int called;
    int m;
    int n;
    int lda;
    float *a;
    int *ipiv;
} g_sgetrf_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    int m;
    int n;
    int lda;
    fb_complex_float_t *a;
    int *ipiv;
} g_cgetrf_cblas_call;

static struct {
    int called;
    int m;
    int n;
    int lda;
    fb_complex_float_t *a;
    int *ipiv;
} g_cgetrf_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    fb_transpose_t trans;
    int n;
    int nrhs;
    int lda;
    int ldb;
    const float *a;
    const int *ipiv;
    float *b;
} g_sgetrs_cblas_call;

static struct {
    int called;
    char trans;
    int n;
    int nrhs;
    int lda;
    int ldb;
    float *a;
    int *ipiv;
    float *b;
} g_sgetrs_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    fb_transpose_t trans;
    int n;
    int nrhs;
    int lda;
    int ldb;
    const fb_complex_float_t *a;
    const int *ipiv;
    fb_complex_float_t *b;
} g_cgetrs_cblas_call;

static struct {
    int called;
    char trans;
    int n;
    int nrhs;
    int lda;
    int ldb;
    fb_complex_float_t *a;
    int *ipiv;
    fb_complex_float_t *b;
} g_cgetrs_fortran_call;

static int stub_sgetrf_cblas(const fb_layout_t layout, const int m,
                             const int n, float *a, const int lda,
                             int *ipiv)
{
    g_sgetrf_cblas_call.called += 1;
    g_sgetrf_cblas_call.layout = layout;
    g_sgetrf_cblas_call.m = m;
    g_sgetrf_cblas_call.n = n;
    g_sgetrf_cblas_call.lda = lda;
    g_sgetrf_cblas_call.a = a;
    g_sgetrf_cblas_call.ipiv = ipiv;
    return 31;
}

static void stub_sgetrf_fortran(int *m, int *n, float *a, int *lda,
                                int *ipiv, int *info)
{
    g_sgetrf_fortran_call.called += 1;
    g_sgetrf_fortran_call.m = *m;
    g_sgetrf_fortran_call.n = *n;
    g_sgetrf_fortran_call.lda = *lda;
    g_sgetrf_fortran_call.a = a;
    g_sgetrf_fortran_call.ipiv = ipiv;
    *info = 37;
}

static int stub_cgetrf_cblas(const fb_layout_t layout, const int m,
                             const int n, fb_complex_float_t *a,
                             const int lda, int *ipiv)
{
    g_cgetrf_cblas_call.called += 1;
    g_cgetrf_cblas_call.layout = layout;
    g_cgetrf_cblas_call.m = m;
    g_cgetrf_cblas_call.n = n;
    g_cgetrf_cblas_call.lda = lda;
    g_cgetrf_cblas_call.a = a;
    g_cgetrf_cblas_call.ipiv = ipiv;
    return 41;
}

static void stub_cgetrf_fortran(int *m, int *n, fb_complex_float_t *a,
                                int *lda, int *ipiv, int *info)
{
    g_cgetrf_fortran_call.called += 1;
    g_cgetrf_fortran_call.m = *m;
    g_cgetrf_fortran_call.n = *n;
    g_cgetrf_fortran_call.lda = *lda;
    g_cgetrf_fortran_call.a = a;
    g_cgetrf_fortran_call.ipiv = ipiv;
    *info = 43;
}

static int stub_sgetrs_cblas(const fb_layout_t layout,
                             const fb_transpose_t trans, const int n,
                             const int nrhs, const float *a, const int lda,
                             const int *ipiv, float *b, const int ldb)
{
    g_sgetrs_cblas_call.called += 1;
    g_sgetrs_cblas_call.layout = layout;
    g_sgetrs_cblas_call.trans = trans;
    g_sgetrs_cblas_call.n = n;
    g_sgetrs_cblas_call.nrhs = nrhs;
    g_sgetrs_cblas_call.lda = lda;
    g_sgetrs_cblas_call.ldb = ldb;
    g_sgetrs_cblas_call.a = a;
    g_sgetrs_cblas_call.ipiv = ipiv;
    g_sgetrs_cblas_call.b = b;
    return 47;
}

static void stub_sgetrs_fortran(char *trans, int *n, int *nrhs, float *a,
                                int *lda, int *ipiv, float *b, int *ldb,
                                int *info)
{
    g_sgetrs_fortran_call.called += 1;
    g_sgetrs_fortran_call.trans = *trans;
    g_sgetrs_fortran_call.n = *n;
    g_sgetrs_fortran_call.nrhs = *nrhs;
    g_sgetrs_fortran_call.lda = *lda;
    g_sgetrs_fortran_call.ldb = *ldb;
    g_sgetrs_fortran_call.a = a;
    g_sgetrs_fortran_call.ipiv = ipiv;
    g_sgetrs_fortran_call.b = b;
    *info = 53;
}

static int stub_cgetrs_cblas(const fb_layout_t layout,
                             const fb_transpose_t trans, const int n,
                             const int nrhs, const fb_complex_float_t *a,
                             const int lda, const int *ipiv,
                             fb_complex_float_t *b, const int ldb)
{
    g_cgetrs_cblas_call.called += 1;
    g_cgetrs_cblas_call.layout = layout;
    g_cgetrs_cblas_call.trans = trans;
    g_cgetrs_cblas_call.n = n;
    g_cgetrs_cblas_call.nrhs = nrhs;
    g_cgetrs_cblas_call.lda = lda;
    g_cgetrs_cblas_call.ldb = ldb;
    g_cgetrs_cblas_call.a = a;
    g_cgetrs_cblas_call.ipiv = ipiv;
    g_cgetrs_cblas_call.b = b;
    return 59;
}

static void stub_cgetrs_fortran(char *trans, int *n, int *nrhs,
                                fb_complex_float_t *a, int *lda, int *ipiv,
                                fb_complex_float_t *b, int *ldb, int *info)
{
    g_cgetrs_fortran_call.called += 1;
    g_cgetrs_fortran_call.trans = *trans;
    g_cgetrs_fortran_call.n = *n;
    g_cgetrs_fortran_call.nrhs = *nrhs;
    g_cgetrs_fortran_call.lda = *lda;
    g_cgetrs_fortran_call.ldb = *ldb;
    g_cgetrs_fortran_call.a = a;
    g_cgetrs_fortran_call.ipiv = ipiv;
    g_cgetrs_fortran_call.b = b;
    *info = 61;
}

static int check_sgetrf_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_sgetrf_fn thunk = NULL;
    float a[20] = { 0.0f };
    int ipiv[4] = { 0 };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sgetrf_fortran_call, 0, sizeof(g_sgetrf_fortran_call));

    vtable.ext_ops[FB_OP_SGETRF][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_sgetrf_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SGETRF);

    thunk = (fb_sgetrf_fn)vtable.ext_ops[FB_OP_SGETRF][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SGETRF Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 5, 4, a, 5, ipiv);
    if (info != 37 || g_sgetrf_fortran_call.called != 1 ||
        g_sgetrf_fortran_call.m != 5 || g_sgetrf_fortran_call.n != 4 ||
        g_sgetrf_fortran_call.lda != 5 || g_sgetrf_fortran_call.a != a ||
        g_sgetrf_fortran_call.ipiv != ipiv) {
        fprintf(stderr, "[FAIL] SGETRF Fortran->CBLAS thunk delegated incorrectly\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 5, 4, a, 5, ipiv);
    if (info != -1 || g_sgetrf_fortran_call.called != 1) {
        fprintf(stderr, "[FAIL] SGETRF Fortran->CBLAS thunk accepted unsupported row-major layout\n");
        return 1;
    }

    printf("[PASS] SGETRF Fortran->CBLAS thunk delegates with LU factor ABI\n");
    return 0;
}

static int check_sgetrf_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_sgetrf_fortran_slot_fn thunk = NULL;
    float a[20] = { 0.0f };
    int ipiv[4] = { 0 };
    int m = 5;
    int n = 4;
    int lda = 5;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sgetrf_cblas_call, 0, sizeof(g_sgetrf_cblas_call));

    vtable.ext_ops[FB_OP_SGETRF][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_sgetrf_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SGETRF);

    thunk = (fb_sgetrf_fortran_slot_fn)vtable.ext_ops[FB_OP_SGETRF][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SGETRF CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&m, &n, a, &lda, ipiv, &info);
    if (info != 31 || g_sgetrf_cblas_call.called != 1 ||
        g_sgetrf_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_sgetrf_cblas_call.m != 5 || g_sgetrf_cblas_call.n != 4 ||
        g_sgetrf_cblas_call.lda != 5 || g_sgetrf_cblas_call.a != a ||
        g_sgetrf_cblas_call.ipiv != ipiv) {
        fprintf(stderr, "[FAIL] SGETRF CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] SGETRF CBLAS->Fortran thunk delegates with pointer ABI\n");
    return 0;
}

static int check_cgetrf_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_cgetrf_fn thunk = NULL;
    fb_complex_float_t a[20];
    int ipiv[4] = { 0 };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cgetrf_fortran_call, 0, sizeof(g_cgetrf_fortran_call));
    memset(a, 0, sizeof(a));

    vtable.ext_ops[FB_OP_CGETRF][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_cgetrf_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CGETRF);

    thunk = (fb_cgetrf_fn)vtable.ext_ops[FB_OP_CGETRF][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CGETRF Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 5, 4, a, 5, ipiv);
    if (info != 43 || g_cgetrf_fortran_call.called != 1 ||
        g_cgetrf_fortran_call.m != 5 || g_cgetrf_fortran_call.n != 4 ||
        g_cgetrf_fortran_call.lda != 5 || g_cgetrf_fortran_call.a != a ||
        g_cgetrf_fortran_call.ipiv != ipiv) {
        fprintf(stderr, "[FAIL] CGETRF Fortran->CBLAS thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] CGETRF Fortran->CBLAS thunk delegates with complex LU factor ABI\n");
    return 0;
}

static int check_cgetrf_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_cgetrf_fortran_slot_fn thunk = NULL;
    fb_complex_float_t a[20];
    int ipiv[4] = { 0 };
    int m = 5;
    int n = 4;
    int lda = 5;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cgetrf_cblas_call, 0, sizeof(g_cgetrf_cblas_call));
    memset(a, 0, sizeof(a));

    vtable.ext_ops[FB_OP_CGETRF][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_cgetrf_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CGETRF);

    thunk = (fb_cgetrf_fortran_slot_fn)vtable.ext_ops[FB_OP_CGETRF][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CGETRF CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&m, &n, a, &lda, ipiv, &info);
    if (info != 41 || g_cgetrf_cblas_call.called != 1 ||
        g_cgetrf_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_cgetrf_cblas_call.m != 5 || g_cgetrf_cblas_call.n != 4 ||
        g_cgetrf_cblas_call.lda != 5 || g_cgetrf_cblas_call.a != a ||
        g_cgetrf_cblas_call.ipiv != ipiv) {
        fprintf(stderr, "[FAIL] CGETRF CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] CGETRF CBLAS->Fortran thunk delegates with complex pointer ABI\n");
    return 0;
}

static int check_sgetrs_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_sgetrs_fn thunk = NULL;
    float a[16] = { 0.0f };
    float b[8] = { 0.0f };
    int ipiv[4] = { 0 };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sgetrs_fortran_call, 0, sizeof(g_sgetrs_fortran_call));

    vtable.ext_ops[FB_OP_SGETRS][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_sgetrs_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SGETRS);

    thunk = (fb_sgetrs_fn)vtable.ext_ops[FB_OP_SGETRS][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SGETRS Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, FB_TRANS, 4, 2, a, 4, ipiv, b, 4);
    if (info != 53 || g_sgetrs_fortran_call.called != 1 ||
        g_sgetrs_fortran_call.trans != 'T' || g_sgetrs_fortran_call.n != 4 ||
        g_sgetrs_fortran_call.nrhs != 2 || g_sgetrs_fortran_call.lda != 4 ||
        g_sgetrs_fortran_call.ldb != 4 || g_sgetrs_fortran_call.a != a ||
        g_sgetrs_fortran_call.ipiv != ipiv || g_sgetrs_fortran_call.b != b) {
        fprintf(stderr, "[FAIL] SGETRS Fortran->CBLAS thunk delegated incorrectly\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, FB_TRANS, 4, 2, a, 4, ipiv, b, 4);
    if (info != -1 || g_sgetrs_fortran_call.called != 1) {
        fprintf(stderr, "[FAIL] SGETRS Fortran->CBLAS thunk accepted unsupported row-major layout\n");
        return 1;
    }

    printf("[PASS] SGETRS Fortran->CBLAS thunk preserves transpose mode\n");
    return 0;
}

static int check_sgetrs_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_sgetrs_fortran_slot_fn thunk = NULL;
    float a[16] = { 0.0f };
    float b[8] = { 0.0f };
    int ipiv[4] = { 0 };
    int n = 4;
    int nrhs = 2;
    int lda = 4;
    int ldb = 4;
    int info = 0;
    char trans = 'C';
    char invalid_trans = 'Q';

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sgetrs_cblas_call, 0, sizeof(g_sgetrs_cblas_call));

    vtable.ext_ops[FB_OP_SGETRS][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_sgetrs_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SGETRS);

    thunk = (fb_sgetrs_fortran_slot_fn)vtable.ext_ops[FB_OP_SGETRS][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SGETRS CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&trans, &n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
    if (info != 47 || g_sgetrs_cblas_call.called != 1 ||
        g_sgetrs_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_sgetrs_cblas_call.trans != FB_CONJ_TRANS ||
        g_sgetrs_cblas_call.n != 4 || g_sgetrs_cblas_call.nrhs != 2 ||
        g_sgetrs_cblas_call.lda != 4 || g_sgetrs_cblas_call.ldb != 4 ||
        g_sgetrs_cblas_call.a != a || g_sgetrs_cblas_call.ipiv != ipiv ||
        g_sgetrs_cblas_call.b != b) {
        fprintf(stderr, "[FAIL] SGETRS CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    thunk(&invalid_trans, &n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
    if (info != -1 || g_sgetrs_cblas_call.called != 1) {
        fprintf(stderr, "[FAIL] SGETRS CBLAS->Fortran thunk accepted invalid transpose mode\n");
        return 1;
    }

    printf("[PASS] SGETRS CBLAS->Fortran thunk preserves transpose mode\n");
    return 0;
}

static int check_cgetrs_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_cgetrs_fn thunk = NULL;
    fb_complex_float_t a[16];
    fb_complex_float_t b[8];
    int ipiv[4] = { 0 };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cgetrs_fortran_call, 0, sizeof(g_cgetrs_fortran_call));
    memset(a, 0, sizeof(a));
    memset(b, 0, sizeof(b));

    vtable.ext_ops[FB_OP_CGETRS][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_cgetrs_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CGETRS);

    thunk = (fb_cgetrs_fn)vtable.ext_ops[FB_OP_CGETRS][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CGETRS Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, FB_CONJ_TRANS, 4, 2, a, 4, ipiv, b, 4);
    if (info != 61 || g_cgetrs_fortran_call.called != 1 ||
        g_cgetrs_fortran_call.trans != 'C' || g_cgetrs_fortran_call.n != 4 ||
        g_cgetrs_fortran_call.nrhs != 2 || g_cgetrs_fortran_call.lda != 4 ||
        g_cgetrs_fortran_call.ldb != 4 || g_cgetrs_fortran_call.a != a ||
        g_cgetrs_fortran_call.ipiv != ipiv || g_cgetrs_fortran_call.b != b) {
        fprintf(stderr, "[FAIL] CGETRS Fortran->CBLAS thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] CGETRS Fortran->CBLAS thunk preserves complex transpose mode\n");
    return 0;
}

static int check_cgetrs_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_cgetrs_fortran_slot_fn thunk = NULL;
    fb_complex_float_t a[16];
    fb_complex_float_t b[8];
    int ipiv[4] = { 0 };
    int n = 4;
    int nrhs = 2;
    int lda = 4;
    int ldb = 4;
    int info = 0;
    char trans = 't';

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cgetrs_cblas_call, 0, sizeof(g_cgetrs_cblas_call));
    memset(a, 0, sizeof(a));
    memset(b, 0, sizeof(b));

    vtable.ext_ops[FB_OP_CGETRS][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_cgetrs_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CGETRS);

    thunk = (fb_cgetrs_fortran_slot_fn)vtable.ext_ops[FB_OP_CGETRS][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CGETRS CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&trans, &n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
    if (info != 59 || g_cgetrs_cblas_call.called != 1 ||
        g_cgetrs_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_cgetrs_cblas_call.trans != FB_TRANS || g_cgetrs_cblas_call.n != 4 ||
        g_cgetrs_cblas_call.nrhs != 2 || g_cgetrs_cblas_call.lda != 4 ||
        g_cgetrs_cblas_call.ldb != 4 || g_cgetrs_cblas_call.a != a ||
        g_cgetrs_cblas_call.ipiv != ipiv || g_cgetrs_cblas_call.b != b) {
        fprintf(stderr, "[FAIL] CGETRS CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] CGETRS CBLAS->Fortran thunk preserves complex transpose mode\n");
    return 0;
}

int main(void)
{
    if (check_sgetrf_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_sgetrf_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_cgetrf_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_cgetrf_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_sgetrs_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_sgetrs_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_cgetrs_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_cgetrs_cblas_to_fortran() != 0) {
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}