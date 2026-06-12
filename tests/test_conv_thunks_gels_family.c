#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef int (*test_sgels_cblas_fn)(fb_layout_t layout, char trans, int m, int n,
                                   int nrhs, float *a, int lda, float *b,
                                   int ldb);
typedef int (*test_dgels_cblas_fn)(fb_layout_t layout, char trans, int m, int n,
                                   int nrhs, double *a, int lda, double *b,
                                   int ldb);
typedef int (*test_cgels_cblas_fn)(fb_layout_t layout, char trans, int m, int n,
                                   int nrhs, fb_complex_float_t *a, int lda,
                                   fb_complex_float_t *b, int ldb);
typedef int (*test_zgels_cblas_fn)(fb_layout_t layout, char trans, int m, int n,
                                   int nrhs, fb_complex_double_t *a, int lda,
                                   fb_complex_double_t *b, int ldb);

typedef void (*fb_sgels_fortran_slot_fn)(char *trans, int *m, int *n,
                                         int *nrhs, float *a, int *lda,
                                         float *b, int *ldb, float *work,
                                         int *lwork, int *info);
typedef void (*fb_dgels_fortran_slot_fn)(char *trans, int *m, int *n,
                                         int *nrhs, double *a, int *lda,
                                         double *b, int *ldb, double *work,
                                         int *lwork, int *info);
typedef void (*fb_cgels_fortran_slot_fn)(char *trans, int *m, int *n,
                                         int *nrhs, fb_complex_float_t *a,
                                         int *lda, fb_complex_float_t *b,
                                         int *ldb, fb_complex_float_t *work,
                                         int *lwork, int *info);
typedef void (*fb_zgels_fortran_slot_fn)(char *trans, int *m, int *n,
                                         int *nrhs, fb_complex_double_t *a,
                                         int *lda, fb_complex_double_t *b,
                                         int *ldb, fb_complex_double_t *work,
                                         int *lwork, int *info);

static struct {
    int query_calls;
    int exec_calls;
    int exec_lwork;
} g_sgels_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    char trans;
    int m;
    int n;
    int nrhs;
    int lda;
    int ldb;
} g_sgels_cblas_call;

static struct {
    int query_calls;
    int exec_calls;
    int exec_lwork;
} g_dgels_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    char trans;
    int m;
    int n;
    int nrhs;
    int lda;
    int ldb;
} g_dgels_cblas_call;

static struct {
    int query_calls;
    int exec_calls;
    int exec_lwork;
} g_cgels_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    char trans;
    int m;
    int n;
    int nrhs;
    int lda;
    int ldb;
} g_cgels_cblas_call;

static struct {
    int query_calls;
    int exec_calls;
    int exec_lwork;
} g_zgels_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    char trans;
    int m;
    int n;
    int nrhs;
    int lda;
    int ldb;
} g_zgels_cblas_call;

static int g_sgels_cblas_rc = 0;
static int g_dgels_cblas_rc = 0;
static int g_cgels_cblas_rc = 0;
static int g_zgels_cblas_rc = 0;

static fb_complex_float_t make_cfloat(float real_value)
{
    fb_complex_float_t value = (fb_complex_float_t)0;
    memcpy(&value, &real_value, sizeof(real_value));
    return value;
}

static fb_complex_double_t make_cdouble(double real_value)
{
    fb_complex_double_t value = (fb_complex_double_t)0;
    memcpy(&value, &real_value, sizeof(real_value));
    return value;
}

static int cfloat_eq(fb_complex_float_t lhs, fb_complex_float_t rhs)
{
    return memcmp(&lhs, &rhs, sizeof(lhs)) == 0;
}

static int cdouble_eq(fb_complex_double_t lhs, fb_complex_double_t rhs)
{
    return memcmp(&lhs, &rhs, sizeof(lhs)) == 0;
}

static void stub_sgels_fortran(char *trans, int *m, int *n, int *nrhs,
                               float *a, int *lda, float *b, int *ldb,
                               float *work, int *lwork, int *info)
{
    (void)trans;
    (void)m;
    (void)n;
    (void)nrhs;
    (void)a;
    (void)lda;
    (void)ldb;
    if (*lwork == -1) {
        g_sgels_fortran_call.query_calls += 1;
        work[0] = 25.0f;
        *info = 0;
        return;
    }

    g_sgels_fortran_call.exec_calls += 1;
    g_sgels_fortran_call.exec_lwork = *lwork;
    b[0] = 1.0f;
    b[1] = 2.0f;
    *info = 0;
}

static int stub_sgels_cblas(fb_layout_t layout, char trans, int m, int n,
                            int nrhs, float *a, int lda, float *b, int ldb)
{
    (void)a;
    (void)b;
    g_sgels_cblas_call.called += 1;
    g_sgels_cblas_call.layout = layout;
    g_sgels_cblas_call.trans = trans;
    g_sgels_cblas_call.m = m;
    g_sgels_cblas_call.n = n;
    g_sgels_cblas_call.nrhs = nrhs;
    g_sgels_cblas_call.lda = lda;
    g_sgels_cblas_call.ldb = ldb;
    return g_sgels_cblas_rc;
}

static void stub_dgels_fortran(char *trans, int *m, int *n, int *nrhs,
                               double *a, int *lda, double *b, int *ldb,
                               double *work, int *lwork, int *info)
{
    (void)trans;
    (void)m;
    (void)n;
    (void)nrhs;
    (void)a;
    (void)lda;
    (void)ldb;
    if (*lwork == -1) {
        g_dgels_fortran_call.query_calls += 1;
        work[0] = 35.0;
        *info = 0;
        return;
    }

    g_dgels_fortran_call.exec_calls += 1;
    g_dgels_fortran_call.exec_lwork = *lwork;
    b[0] = 5.0;
    b[1] = 6.0;
    *info = 0;
}

static int stub_dgels_cblas(fb_layout_t layout, char trans, int m, int n,
                            int nrhs, double *a, int lda, double *b, int ldb)
{
    (void)a;
    (void)b;
    g_dgels_cblas_call.called += 1;
    g_dgels_cblas_call.layout = layout;
    g_dgels_cblas_call.trans = trans;
    g_dgels_cblas_call.m = m;
    g_dgels_cblas_call.n = n;
    g_dgels_cblas_call.nrhs = nrhs;
    g_dgels_cblas_call.lda = lda;
    g_dgels_cblas_call.ldb = ldb;
    return g_dgels_cblas_rc;
}

static void stub_cgels_fortran(char *trans, int *m, int *n, int *nrhs,
                               fb_complex_float_t *a, int *lda,
                               fb_complex_float_t *b, int *ldb,
                               fb_complex_float_t *work, int *lwork,
                               int *info)
{
    (void)trans;
    (void)m;
    (void)n;
    (void)nrhs;
    (void)a;
    (void)lda;
    (void)ldb;
    if (*lwork == -1) {
        g_cgels_fortran_call.query_calls += 1;
        work[0] = make_cfloat(26.0f);
        *info = 0;
        return;
    }

    g_cgels_fortran_call.exec_calls += 1;
    g_cgels_fortran_call.exec_lwork = *lwork;
    b[0] = make_cfloat(3.0f);
    b[1] = make_cfloat(4.0f);
    *info = 0;
}

static int stub_cgels_cblas(fb_layout_t layout, char trans, int m, int n,
                            int nrhs, fb_complex_float_t *a, int lda,
                            fb_complex_float_t *b, int ldb)
{
    (void)a;
    (void)b;
    g_cgels_cblas_call.called += 1;
    g_cgels_cblas_call.layout = layout;
    g_cgels_cblas_call.trans = trans;
    g_cgels_cblas_call.m = m;
    g_cgels_cblas_call.n = n;
    g_cgels_cblas_call.nrhs = nrhs;
    g_cgels_cblas_call.lda = lda;
    g_cgels_cblas_call.ldb = ldb;
    return g_cgels_cblas_rc;
}

static void stub_zgels_fortran(char *trans, int *m, int *n, int *nrhs,
                               fb_complex_double_t *a, int *lda,
                               fb_complex_double_t *b, int *ldb,
                               fb_complex_double_t *work, int *lwork,
                               int *info)
{
    (void)trans;
    (void)m;
    (void)n;
    (void)nrhs;
    (void)a;
    (void)lda;
    (void)ldb;
    if (*lwork == -1) {
        g_zgels_fortran_call.query_calls += 1;
        work[0] = make_cdouble(36.0);
        *info = 0;
        return;
    }

    g_zgels_fortran_call.exec_calls += 1;
    g_zgels_fortran_call.exec_lwork = *lwork;
    b[0] = make_cdouble(7.0);
    b[1] = make_cdouble(8.0);
    *info = 0;
}

static int stub_zgels_cblas(fb_layout_t layout, char trans, int m, int n,
                            int nrhs, fb_complex_double_t *a, int lda,
                            fb_complex_double_t *b, int ldb)
{
    (void)a;
    (void)b;
    g_zgels_cblas_call.called += 1;
    g_zgels_cblas_call.layout = layout;
    g_zgels_cblas_call.trans = trans;
    g_zgels_cblas_call.m = m;
    g_zgels_cblas_call.n = n;
    g_zgels_cblas_call.nrhs = nrhs;
    g_zgels_cblas_call.lda = lda;
    g_zgels_cblas_call.ldb = ldb;
    return g_zgels_cblas_rc;
}

static int check_dgels_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    test_dgels_cblas_fn thunk = NULL;
    double a[8] = { 0.0 };
    double b[4] = { 0.0, 0.0, 0.0, 0.0 };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dgels_fortran_call, 0, sizeof(g_dgels_fortran_call));

    vtable.ext_ops[FB_OP_DGELS][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_dgels_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_DGELS);

    thunk = (test_dgels_cblas_fn)vtable.ext_ops[FB_OP_DGELS][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DGELS Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 'N', 4, 2, 1, a, 4, b, 4);
    if (info != 0 || g_dgels_fortran_call.query_calls != 1 ||
        g_dgels_fortran_call.exec_calls != 1 ||
        g_dgels_fortran_call.exec_lwork != 35 ||
        b[0] != 5.0 || b[1] != 6.0) {
        fprintf(stderr, "[FAIL] DGELS Fortran->CBLAS thunk did not preserve least-squares query semantics\n");
        return 1;
    }

    printf("[PASS] DGELS Fortran->CBLAS thunk performs the workspace query and forwards double least-squares outputs\n");
    return 0;
}

static int check_dgels_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_dgels_fortran_slot_fn thunk = NULL;
    double a[8] = { 0.0 };
    double b[4] = { 0.0, 0.0, 0.0, 0.0 };
    double work[8] = { 0.0 };
    char trans = 'T';
    int m = 4;
    int n = 2;
    int nrhs = 1;
    int lda = 4;
    int ldb = 4;
    int lwork = 8;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dgels_cblas_call, 0, sizeof(g_dgels_cblas_call));
    g_dgels_cblas_rc = 412;

    vtable.ext_ops[FB_OP_DGELS][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_dgels_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_DGELS);

    thunk = (fb_dgels_fortran_slot_fn)vtable.ext_ops[FB_OP_DGELS][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DGELS CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&trans, &m, &n, &nrhs, a, &lda, b, &ldb, work, &lwork, &info);
    if (info != 412 || g_dgels_cblas_call.called != 1 ||
        g_dgels_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_dgels_cblas_call.trans != 'T' || g_dgels_cblas_call.m != 4 ||
        g_dgels_cblas_call.n != 2 || g_dgels_cblas_call.nrhs != 1 ||
        g_dgels_cblas_call.lda != 4 || g_dgels_cblas_call.ldb != 4) {
        fprintf(stderr, "[FAIL] DGELS CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] DGELS CBLAS->Fortran thunk forwards trans/m/n/nrhs double least-squares metadata\n");
    return 0;
}

static int check_sgels_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    test_sgels_cblas_fn thunk = NULL;
    float a[8] = { 0.0f };
    float b[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sgels_fortran_call, 0, sizeof(g_sgels_fortran_call));

    vtable.ext_ops[FB_OP_SGELS][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_sgels_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SGELS);

    thunk = (test_sgels_cblas_fn)vtable.ext_ops[FB_OP_SGELS][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SGELS Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 'N', 4, 2, 1, a, 4, b, 4);
    if (info != 0 || g_sgels_fortran_call.query_calls != 1 ||
        g_sgels_fortran_call.exec_calls != 1 ||
        g_sgels_fortran_call.exec_lwork != 25 ||
        b[0] != 1.0f || b[1] != 2.0f) {
        fprintf(stderr, "[FAIL] SGELS Fortran->CBLAS thunk did not preserve least-squares query semantics\n");
        return 1;
    }

    printf("[PASS] SGELS Fortran->CBLAS thunk performs the workspace query and forwards least-squares outputs\n");
    return 0;
}

static int check_sgels_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_sgels_fortran_slot_fn thunk = NULL;
    float a[8] = { 0.0f };
    float b[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float work[8] = { 0.0f };
    char trans = 'T';
    int m = 4;
    int n = 2;
    int nrhs = 1;
    int lda = 4;
    int ldb = 4;
    int lwork = 8;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sgels_cblas_call, 0, sizeof(g_sgels_cblas_call));
    g_sgels_cblas_rc = 411;

    vtable.ext_ops[FB_OP_SGELS][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_sgels_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SGELS);

    thunk = (fb_sgels_fortran_slot_fn)vtable.ext_ops[FB_OP_SGELS][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SGELS CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&trans, &m, &n, &nrhs, a, &lda, b, &ldb, work, &lwork, &info);
    if (info != 411 || g_sgels_cblas_call.called != 1 ||
        g_sgels_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_sgels_cblas_call.trans != 'T' || g_sgels_cblas_call.m != 4 ||
        g_sgels_cblas_call.n != 2 || g_sgels_cblas_call.nrhs != 1 ||
        g_sgels_cblas_call.lda != 4 || g_sgels_cblas_call.ldb != 4) {
        fprintf(stderr, "[FAIL] SGELS CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] SGELS CBLAS->Fortran thunk forwards trans/m/n/nrhs least-squares metadata\n");
    return 0;
}

static int check_cgels_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    test_cgels_cblas_fn thunk = NULL;
    fb_complex_float_t a[8] = { 0 };
    fb_complex_float_t b[4] = { 0 };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cgels_fortran_call, 0, sizeof(g_cgels_fortran_call));

    vtable.ext_ops[FB_OP_CGELS][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_cgels_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CGELS);

    thunk = (test_cgels_cblas_fn)vtable.ext_ops[FB_OP_CGELS][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CGELS Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 'N', 4, 2, 1, a, 4, b, 4);
    if (info != 0 || g_cgels_fortran_call.query_calls != 1 ||
        g_cgels_fortran_call.exec_calls != 1 ||
        g_cgels_fortran_call.exec_lwork != 26 ||
        !cfloat_eq(b[0], make_cfloat(3.0f)) ||
        !cfloat_eq(b[1], make_cfloat(4.0f))) {
        fprintf(stderr, "[FAIL] CGELS Fortran->CBLAS thunk did not preserve complex least-squares query semantics\n");
        return 1;
    }

    printf("[PASS] CGELS Fortran->CBLAS thunk performs the workspace query and forwards complex least-squares outputs\n");
    return 0;
}

static int check_cgels_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_cgels_fortran_slot_fn thunk = NULL;
    fb_complex_float_t a[8] = { 0 };
    fb_complex_float_t b[4] = { 0 };
    fb_complex_float_t work[8] = { 0 };
    char trans = 'N';
    int m = 4;
    int n = 2;
    int nrhs = 1;
    int lda = 4;
    int ldb = 4;
    int lwork = 8;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cgels_cblas_call, 0, sizeof(g_cgels_cblas_call));
    g_cgels_cblas_rc = 413;

    vtable.ext_ops[FB_OP_CGELS][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_cgels_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CGELS);

    thunk = (fb_cgels_fortran_slot_fn)vtable.ext_ops[FB_OP_CGELS][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CGELS CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&trans, &m, &n, &nrhs, a, &lda, b, &ldb, work, &lwork, &info);
    if (info != 413 || g_cgels_cblas_call.called != 1 ||
        g_cgels_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_cgels_cblas_call.trans != 'N' || g_cgels_cblas_call.m != 4 ||
        g_cgels_cblas_call.n != 2 || g_cgels_cblas_call.nrhs != 1 ||
        g_cgels_cblas_call.lda != 4 || g_cgels_cblas_call.ldb != 4) {
        fprintf(stderr, "[FAIL] CGELS CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] CGELS CBLAS->Fortran thunk forwards complex least-squares metadata\n");
    return 0;
}

static int check_zgels_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    test_zgels_cblas_fn thunk = NULL;
    fb_complex_double_t a[8] = { 0 };
    fb_complex_double_t b[4] = { 0 };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zgels_fortran_call, 0, sizeof(g_zgels_fortran_call));

    vtable.ext_ops[FB_OP_ZGELS][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_zgels_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_ZGELS);

    thunk = (test_zgels_cblas_fn)vtable.ext_ops[FB_OP_ZGELS][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZGELS Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 'N', 4, 2, 1, a, 4, b, 4);
    if (info != 0 || g_zgels_fortran_call.query_calls != 1 ||
        g_zgels_fortran_call.exec_calls != 1 ||
        g_zgels_fortran_call.exec_lwork != 36 ||
        !cdouble_eq(b[0], make_cdouble(7.0)) ||
        !cdouble_eq(b[1], make_cdouble(8.0))) {
        fprintf(stderr, "[FAIL] ZGELS Fortran->CBLAS thunk did not preserve complex-double least-squares query semantics\n");
        return 1;
    }

    printf("[PASS] ZGELS Fortran->CBLAS thunk performs the workspace query and forwards complex-double least-squares outputs\n");
    return 0;
}

static int check_zgels_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_zgels_fortran_slot_fn thunk = NULL;
    fb_complex_double_t a[8] = { 0 };
    fb_complex_double_t b[4] = { 0 };
    fb_complex_double_t work[8] = { 0 };
    char trans = 'N';
    int m = 4;
    int n = 2;
    int nrhs = 1;
    int lda = 4;
    int ldb = 4;
    int lwork = 8;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zgels_cblas_call, 0, sizeof(g_zgels_cblas_call));
    g_zgels_cblas_rc = 414;

    vtable.ext_ops[FB_OP_ZGELS][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_zgels_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_ZGELS);

    thunk = (fb_zgels_fortran_slot_fn)vtable.ext_ops[FB_OP_ZGELS][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZGELS CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&trans, &m, &n, &nrhs, a, &lda, b, &ldb, work, &lwork, &info);
    if (info != 414 || g_zgels_cblas_call.called != 1 ||
        g_zgels_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_zgels_cblas_call.trans != 'N' || g_zgels_cblas_call.m != 4 ||
        g_zgels_cblas_call.n != 2 || g_zgels_cblas_call.nrhs != 1 ||
        g_zgels_cblas_call.lda != 4 || g_zgels_cblas_call.ldb != 4) {
        fprintf(stderr, "[FAIL] ZGELS CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] ZGELS CBLAS->Fortran thunk forwards complex-double least-squares metadata\n");
    return 0;
}

int main(void)
{
    if (check_sgels_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_sgels_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_dgels_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_dgels_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_cgels_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_cgels_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_zgels_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_zgels_cblas_to_fortran() != 0) {
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}