#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef int (*fb_sgeequ_fn)(fb_layout_t layout, int m, int n, const float *a,
                            int lda, float *r, float *c,
                            float *rowcnd, float *colcnd, float *amax);
typedef int (*fb_dgeequ_fn)(fb_layout_t layout, int m, int n, const double *a,
                            int lda, double *r, double *c,
                            double *rowcnd, double *colcnd, double *amax);
typedef int (*fb_cgeequ_fn)(fb_layout_t layout, int m, int n,
                            const fb_complex_float_t *a, int lda,
                            float *r, float *c,
                            float *rowcnd, float *colcnd, float *amax);
typedef int (*fb_zgeequ_fn)(fb_layout_t layout, int m, int n,
                            const fb_complex_double_t *a, int lda,
                            double *r, double *c,
                            double *rowcnd, double *colcnd, double *amax);

typedef void (*fb_sgeequ_fortran_slot_fn)(int *m, int *n, float *a, int *lda,
                                          float *r, float *c,
                                          float *rowcnd, float *colcnd,
                                          float *amax, int *info);
typedef void (*fb_dgeequ_fortran_slot_fn)(int *m, int *n, double *a, int *lda,
                                          double *r, double *c,
                                          double *rowcnd, double *colcnd,
                                          double *amax, int *info);
typedef void (*fb_cgeequ_fortran_slot_fn)(int *m, int *n,
                                          fb_complex_float_t *a, int *lda,
                                          float *r, float *c,
                                          float *rowcnd, float *colcnd,
                                          float *amax, int *info);
typedef void (*fb_zgeequ_fortran_slot_fn)(int *m, int *n,
                                          fb_complex_double_t *a, int *lda,
                                          double *r, double *c,
                                          double *rowcnd, double *colcnd,
                                          double *amax, int *info);

static struct {
    int calls;
    int m;
    int n;
    int lda;
    const float *a;
} g_sgeequ_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    int m;
    int n;
    int lda;
    const float *a;
    float *r;
    float *c;
    float *rowcnd;
    float *colcnd;
    float *amax;
} g_sgeequ_cblas_call;

static struct {
    int calls;
    int m;
    int n;
    int lda;
    const double *a;
} g_dgeequ_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    int m;
    int n;
    int lda;
    const double *a;
    double *r;
    double *c;
    double *rowcnd;
    double *colcnd;
    double *amax;
} g_dgeequ_cblas_call;

static struct {
    int calls;
    int m;
    int n;
    int lda;
    const fb_complex_float_t *a;
} g_cgeequ_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    int m;
    int n;
    int lda;
    const fb_complex_float_t *a;
    float *r;
    float *c;
    float *rowcnd;
    float *colcnd;
    float *amax;
} g_cgeequ_cblas_call;

static struct {
    int calls;
    int m;
    int n;
    int lda;
    const fb_complex_double_t *a;
} g_zgeequ_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    int m;
    int n;
    int lda;
    const fb_complex_double_t *a;
    double *r;
    double *c;
    double *rowcnd;
    double *colcnd;
    double *amax;
} g_zgeequ_cblas_call;

static int g_sgeequ_cblas_rc = 0;
static int g_dgeequ_cblas_rc = 0;
static int g_cgeequ_cblas_rc = 0;
static int g_zgeequ_cblas_rc = 0;

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

static void stub_sgeequ_fortran(int *m, int *n, const float *a, int *lda,
                                float *r, float *c, float *rowcnd,
                                float *colcnd, float *amax, int *info)
{
    g_sgeequ_fortran_call.calls += 1;
    g_sgeequ_fortran_call.m = *m;
    g_sgeequ_fortran_call.n = *n;
    g_sgeequ_fortran_call.lda = *lda;
    g_sgeequ_fortran_call.a = a;
    r[0] = 0.5f;
    r[1] = 0.25f;
    c[0] = 2.0f;
    c[1] = 4.0f;
    *rowcnd = 0.125f;
    *colcnd = 0.0625f;
    *amax = 9.0f;
    *info = 0;
}

static int stub_sgeequ_cblas(fb_layout_t layout, int m, int n, const float *a,
                             int lda, float *r, float *c,
                             float *rowcnd, float *colcnd, float *amax)
{
    g_sgeequ_cblas_call.called += 1;
    g_sgeequ_cblas_call.layout = layout;
    g_sgeequ_cblas_call.m = m;
    g_sgeequ_cblas_call.n = n;
    g_sgeequ_cblas_call.lda = lda;
    g_sgeequ_cblas_call.a = a;
    g_sgeequ_cblas_call.r = r;
    g_sgeequ_cblas_call.c = c;
    g_sgeequ_cblas_call.rowcnd = rowcnd;
    g_sgeequ_cblas_call.colcnd = colcnd;
    g_sgeequ_cblas_call.amax = amax;
    return g_sgeequ_cblas_rc;
}

static void stub_dgeequ_fortran(int *m, int *n, const double *a, int *lda,
                                double *r, double *c, double *rowcnd,
                                double *colcnd, double *amax, int *info)
{
    g_dgeequ_fortran_call.calls += 1;
    g_dgeequ_fortran_call.m = *m;
    g_dgeequ_fortran_call.n = *n;
    g_dgeequ_fortran_call.lda = *lda;
    g_dgeequ_fortran_call.a = a;
    r[0] = 0.55;
    r[1] = 0.35;
    c[0] = 2.5;
    c[1] = 4.5;
    *rowcnd = 0.225;
    *colcnd = 0.1625;
    *amax = 19.0;
    *info = 0;
}

static int stub_dgeequ_cblas(fb_layout_t layout, int m, int n, const double *a,
                             int lda, double *r, double *c,
                             double *rowcnd, double *colcnd, double *amax)
{
    g_dgeequ_cblas_call.called += 1;
    g_dgeequ_cblas_call.layout = layout;
    g_dgeequ_cblas_call.m = m;
    g_dgeequ_cblas_call.n = n;
    g_dgeequ_cblas_call.lda = lda;
    g_dgeequ_cblas_call.a = a;
    g_dgeequ_cblas_call.r = r;
    g_dgeequ_cblas_call.c = c;
    g_dgeequ_cblas_call.rowcnd = rowcnd;
    g_dgeequ_cblas_call.colcnd = colcnd;
    g_dgeequ_cblas_call.amax = amax;
    return g_dgeequ_cblas_rc;
}

static void stub_cgeequ_fortran(int *m, int *n, const fb_complex_float_t *a,
                                int *lda, float *r, float *c,
                                float *rowcnd, float *colcnd,
                                float *amax, int *info)
{
    g_cgeequ_fortran_call.calls += 1;
    g_cgeequ_fortran_call.m = *m;
    g_cgeequ_fortran_call.n = *n;
    g_cgeequ_fortran_call.lda = *lda;
    g_cgeequ_fortran_call.a = a;
    r[0] = 0.75f;
    r[1] = 0.5f;
    c[0] = 1.5f;
    c[1] = 3.0f;
    *rowcnd = 0.2f;
    *colcnd = 0.1f;
    *amax = 11.0f;
    *info = 0;
}

static int stub_cgeequ_cblas(fb_layout_t layout, int m, int n,
                             const fb_complex_float_t *a, int lda,
                             float *r, float *c,
                             float *rowcnd, float *colcnd, float *amax)
{
    g_cgeequ_cblas_call.called += 1;
    g_cgeequ_cblas_call.layout = layout;
    g_cgeequ_cblas_call.m = m;
    g_cgeequ_cblas_call.n = n;
    g_cgeequ_cblas_call.lda = lda;
    g_cgeequ_cblas_call.a = a;
    g_cgeequ_cblas_call.r = r;
    g_cgeequ_cblas_call.c = c;
    g_cgeequ_cblas_call.rowcnd = rowcnd;
    g_cgeequ_cblas_call.colcnd = colcnd;
    g_cgeequ_cblas_call.amax = amax;
    return g_cgeequ_cblas_rc;
}

static void stub_zgeequ_fortran(int *m, int *n, const fb_complex_double_t *a,
                                int *lda, double *r, double *c,
                                double *rowcnd, double *colcnd,
                                double *amax, int *info)
{
    g_zgeequ_fortran_call.calls += 1;
    g_zgeequ_fortran_call.m = *m;
    g_zgeequ_fortran_call.n = *n;
    g_zgeequ_fortran_call.lda = *lda;
    g_zgeequ_fortran_call.a = a;
    r[0] = 0.85;
    r[1] = 0.65;
    c[0] = 1.75;
    c[1] = 3.5;
    *rowcnd = 0.4;
    *colcnd = 0.2;
    *amax = 21.0;
    *info = 0;
}

static int stub_zgeequ_cblas(fb_layout_t layout, int m, int n,
                             const fb_complex_double_t *a, int lda,
                             double *r, double *c,
                             double *rowcnd, double *colcnd, double *amax)
{
    g_zgeequ_cblas_call.called += 1;
    g_zgeequ_cblas_call.layout = layout;
    g_zgeequ_cblas_call.m = m;
    g_zgeequ_cblas_call.n = n;
    g_zgeequ_cblas_call.lda = lda;
    g_zgeequ_cblas_call.a = a;
    g_zgeequ_cblas_call.r = r;
    g_zgeequ_cblas_call.c = c;
    g_zgeequ_cblas_call.rowcnd = rowcnd;
    g_zgeequ_cblas_call.colcnd = colcnd;
    g_zgeequ_cblas_call.amax = amax;
    return g_zgeequ_cblas_rc;
}

static int check_dgeequ_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_dgeequ_fn thunk = NULL;
    double a[4] = { 1.0, 2.0, 3.0, 4.0 };
    double r[2] = { 0.0, 0.0 };
    double c[2] = { 0.0, 0.0 };
    double rowcnd = 0.0;
    double colcnd = 0.0;
    double amax = 0.0;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dgeequ_fortran_call, 0, sizeof(g_dgeequ_fortran_call));

    vtable.ext_ops[FB_OP_DGEEQU][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_dgeequ_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_DGEEQU);

    thunk = (fb_dgeequ_fn)vtable.ext_ops[FB_OP_DGEEQU][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DGEEQU Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 2, 2, a, 2, r, c, &rowcnd, &colcnd, &amax);
    if (info != 0 || g_dgeequ_fortran_call.calls != 1 ||
        g_dgeequ_fortran_call.m != 2 || g_dgeequ_fortran_call.n != 2 ||
        g_dgeequ_fortran_call.lda != 2 || g_dgeequ_fortran_call.a != a ||
        r[0] != 0.55 || r[1] != 0.35 || c[0] != 2.5 || c[1] != 4.5 ||
        rowcnd != 0.225 || colcnd != 0.1625 || amax != 19.0) {
        fprintf(stderr, "[FAIL] DGEEQU Fortran->CBLAS thunk did not preserve double equilibration outputs\n");
        return 1;
    }

    printf("[PASS] DGEEQU Fortran->CBLAS thunk forwards double matrix equilibration outputs\n");
    return 0;
}

static int check_dgeequ_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_dgeequ_fortran_slot_fn thunk = NULL;
    double a[4] = { 1.0, 2.0, 3.0, 4.0 };
    double r[2] = { 0.0, 0.0 };
    double c[2] = { 0.0, 0.0 };
    double rowcnd = 0.0;
    double colcnd = 0.0;
    double amax = 0.0;
    int m = 2;
    int n = 2;
    int lda = 2;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dgeequ_cblas_call, 0, sizeof(g_dgeequ_cblas_call));
    g_dgeequ_cblas_rc = 192;

    vtable.ext_ops[FB_OP_DGEEQU][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_dgeequ_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_DGEEQU);

    thunk = (fb_dgeequ_fortran_slot_fn)vtable.ext_ops[FB_OP_DGEEQU][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DGEEQU CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&m, &n, a, &lda, r, c, &rowcnd, &colcnd, &amax, &info);
    if (info != 192 || g_dgeequ_cblas_call.called != 1 ||
        g_dgeequ_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_dgeequ_cblas_call.m != 2 || g_dgeequ_cblas_call.n != 2 ||
        g_dgeequ_cblas_call.lda != 2 || g_dgeequ_cblas_call.a != a ||
        g_dgeequ_cblas_call.r != r || g_dgeequ_cblas_call.c != c ||
        g_dgeequ_cblas_call.rowcnd != &rowcnd ||
        g_dgeequ_cblas_call.colcnd != &colcnd ||
        g_dgeequ_cblas_call.amax != &amax) {
        fprintf(stderr, "[FAIL] DGEEQU CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] DGEEQU CBLAS->Fortran thunk maps the all-pointer ABI into the double C equilibration entry\n");
    return 0;
}

static int check_sgeequ_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_sgeequ_fn thunk = NULL;
    float a[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    float r[2] = { 0.0f, 0.0f };
    float c[2] = { 0.0f, 0.0f };
    float rowcnd = 0.0f;
    float colcnd = 0.0f;
    float amax = 0.0f;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sgeequ_fortran_call, 0, sizeof(g_sgeequ_fortran_call));

    vtable.ext_ops[FB_OP_SGEEQU][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_sgeequ_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SGEEQU);

    thunk = (fb_sgeequ_fn)vtable.ext_ops[FB_OP_SGEEQU][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SGEEQU Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 2, 2, a, 2, r, c, &rowcnd, &colcnd, &amax);
    if (info != 0 || g_sgeequ_fortran_call.calls != 1 ||
        g_sgeequ_fortran_call.m != 2 || g_sgeequ_fortran_call.n != 2 ||
        g_sgeequ_fortran_call.lda != 2 || g_sgeequ_fortran_call.a != a ||
        r[0] != 0.5f || r[1] != 0.25f || c[0] != 2.0f || c[1] != 4.0f ||
        rowcnd != 0.125f || colcnd != 0.0625f || amax != 9.0f) {
        fprintf(stderr, "[FAIL] SGEEQU Fortran->CBLAS thunk did not preserve equilibration outputs\n");
        return 1;
    }

    printf("[PASS] SGEEQU Fortran->CBLAS thunk forwards matrix equilibration outputs\n");
    return 0;
}

static int check_sgeequ_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_sgeequ_fortran_slot_fn thunk = NULL;
    float a[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    float r[2] = { 0.0f, 0.0f };
    float c[2] = { 0.0f, 0.0f };
    float rowcnd = 0.0f;
    float colcnd = 0.0f;
    float amax = 0.0f;
    int m = 2;
    int n = 2;
    int lda = 2;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sgeequ_cblas_call, 0, sizeof(g_sgeequ_cblas_call));
    g_sgeequ_cblas_rc = 191;

    vtable.ext_ops[FB_OP_SGEEQU][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_sgeequ_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SGEEQU);

    thunk = (fb_sgeequ_fortran_slot_fn)vtable.ext_ops[FB_OP_SGEEQU][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SGEEQU CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&m, &n, a, &lda, r, c, &rowcnd, &colcnd, &amax, &info);
    if (info != 191 || g_sgeequ_cblas_call.called != 1 ||
        g_sgeequ_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_sgeequ_cblas_call.m != 2 || g_sgeequ_cblas_call.n != 2 ||
        g_sgeequ_cblas_call.lda != 2 || g_sgeequ_cblas_call.a != a ||
        g_sgeequ_cblas_call.r != r || g_sgeequ_cblas_call.c != c ||
        g_sgeequ_cblas_call.rowcnd != &rowcnd ||
        g_sgeequ_cblas_call.colcnd != &colcnd ||
        g_sgeequ_cblas_call.amax != &amax) {
        fprintf(stderr, "[FAIL] SGEEQU CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] SGEEQU CBLAS->Fortran thunk maps the all-pointer ABI into the generic C equilibration entry\n");
    return 0;
}

static int check_cgeequ_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_cgeequ_fn thunk = NULL;
    fb_complex_float_t a[4] = {
        make_cfloat(1.0f), make_cfloat(2.0f),
        make_cfloat(3.0f), make_cfloat(4.0f)
    };
    float r[2] = { 0.0f, 0.0f };
    float c[2] = { 0.0f, 0.0f };
    float rowcnd = 0.0f;
    float colcnd = 0.0f;
    float amax = 0.0f;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cgeequ_fortran_call, 0, sizeof(g_cgeequ_fortran_call));

    vtable.ext_ops[FB_OP_CGEEQU][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_cgeequ_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CGEEQU);

    thunk = (fb_cgeequ_fn)vtable.ext_ops[FB_OP_CGEEQU][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CGEEQU Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 2, 2, a, 2, r, c, &rowcnd, &colcnd, &amax);
    if (info != 0 || g_cgeequ_fortran_call.calls != 1 ||
        g_cgeequ_fortran_call.m != 2 || g_cgeequ_fortran_call.n != 2 ||
        g_cgeequ_fortran_call.lda != 2 || g_cgeequ_fortran_call.a != a ||
        r[0] != 0.75f || r[1] != 0.5f || c[0] != 1.5f || c[1] != 3.0f ||
        rowcnd != 0.2f || colcnd != 0.1f || amax != 11.0f) {
        fprintf(stderr, "[FAIL] CGEEQU Fortran->CBLAS thunk did not preserve complex equilibration outputs\n");
        return 1;
    }

    printf("[PASS] CGEEQU Fortran->CBLAS thunk forwards complex matrix equilibration outputs\n");
    return 0;
}

static int check_cgeequ_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_cgeequ_fortran_slot_fn thunk = NULL;
    fb_complex_float_t a[4] = {
        make_cfloat(1.0f), make_cfloat(2.0f),
        make_cfloat(3.0f), make_cfloat(4.0f)
    };
    float r[2] = { 0.0f, 0.0f };
    float c[2] = { 0.0f, 0.0f };
    float rowcnd = 0.0f;
    float colcnd = 0.0f;
    float amax = 0.0f;
    int m = 2;
    int n = 2;
    int lda = 2;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cgeequ_cblas_call, 0, sizeof(g_cgeequ_cblas_call));
    g_cgeequ_cblas_rc = 193;

    vtable.ext_ops[FB_OP_CGEEQU][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_cgeequ_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CGEEQU);

    thunk = (fb_cgeequ_fortran_slot_fn)vtable.ext_ops[FB_OP_CGEEQU][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CGEEQU CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&m, &n, a, &lda, r, c, &rowcnd, &colcnd, &amax, &info);
    if (info != 193 || g_cgeequ_cblas_call.called != 1 ||
        g_cgeequ_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_cgeequ_cblas_call.m != 2 || g_cgeequ_cblas_call.n != 2 ||
        g_cgeequ_cblas_call.lda != 2 || g_cgeequ_cblas_call.a != a ||
        g_cgeequ_cblas_call.r != r || g_cgeequ_cblas_call.c != c ||
        g_cgeequ_cblas_call.rowcnd != &rowcnd ||
        g_cgeequ_cblas_call.colcnd != &colcnd ||
        g_cgeequ_cblas_call.amax != &amax) {
        fprintf(stderr, "[FAIL] CGEEQU CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] CGEEQU CBLAS->Fortran thunk maps the all-pointer ABI into the generic complex equilibration entry\n");
    return 0;
}

static int check_zgeequ_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_zgeequ_fn thunk = NULL;
    fb_complex_double_t a[4] = {
        make_cdouble(1.0), make_cdouble(2.0),
        make_cdouble(3.0), make_cdouble(4.0)
    };
    double r[2] = { 0.0, 0.0 };
    double c[2] = { 0.0, 0.0 };
    double rowcnd = 0.0;
    double colcnd = 0.0;
    double amax = 0.0;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zgeequ_fortran_call, 0, sizeof(g_zgeequ_fortran_call));

    vtable.ext_ops[FB_OP_ZGEEQU][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_zgeequ_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_ZGEEQU);

    thunk = (fb_zgeequ_fn)vtable.ext_ops[FB_OP_ZGEEQU][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZGEEQU Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 2, 2, a, 2, r, c, &rowcnd, &colcnd, &amax);
    if (info != 0 || g_zgeequ_fortran_call.calls != 1 ||
        g_zgeequ_fortran_call.m != 2 || g_zgeequ_fortran_call.n != 2 ||
        g_zgeequ_fortran_call.lda != 2 || g_zgeequ_fortran_call.a != a ||
        r[0] != 0.85 || r[1] != 0.65 || c[0] != 1.75 || c[1] != 3.5 ||
        rowcnd != 0.4 || colcnd != 0.2 || amax != 21.0) {
        fprintf(stderr, "[FAIL] ZGEEQU Fortran->CBLAS thunk did not preserve complex-double equilibration outputs\n");
        return 1;
    }

    printf("[PASS] ZGEEQU Fortran->CBLAS thunk forwards complex-double matrix equilibration outputs\n");
    return 0;
}

static int check_zgeequ_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_zgeequ_fortran_slot_fn thunk = NULL;
    fb_complex_double_t a[4] = {
        make_cdouble(1.0), make_cdouble(2.0),
        make_cdouble(3.0), make_cdouble(4.0)
    };
    double r[2] = { 0.0, 0.0 };
    double c[2] = { 0.0, 0.0 };
    double rowcnd = 0.0;
    double colcnd = 0.0;
    double amax = 0.0;
    int m = 2;
    int n = 2;
    int lda = 2;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zgeequ_cblas_call, 0, sizeof(g_zgeequ_cblas_call));
    g_zgeequ_cblas_rc = 194;

    vtable.ext_ops[FB_OP_ZGEEQU][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_zgeequ_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_ZGEEQU);

    thunk = (fb_zgeequ_fortran_slot_fn)vtable.ext_ops[FB_OP_ZGEEQU][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZGEEQU CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&m, &n, a, &lda, r, c, &rowcnd, &colcnd, &amax, &info);
    if (info != 194 || g_zgeequ_cblas_call.called != 1 ||
        g_zgeequ_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_zgeequ_cblas_call.m != 2 || g_zgeequ_cblas_call.n != 2 ||
        g_zgeequ_cblas_call.lda != 2 || g_zgeequ_cblas_call.a != a ||
        g_zgeequ_cblas_call.r != r || g_zgeequ_cblas_call.c != c ||
        g_zgeequ_cblas_call.rowcnd != &rowcnd ||
        g_zgeequ_cblas_call.colcnd != &colcnd ||
        g_zgeequ_cblas_call.amax != &amax) {
        fprintf(stderr, "[FAIL] ZGEEQU CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] ZGEEQU CBLAS->Fortran thunk maps the all-pointer ABI into the complex-double equilibration entry\n");
    return 0;
}

int main(void)
{
    if (check_sgeequ_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_sgeequ_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_dgeequ_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_dgeequ_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_cgeequ_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_cgeequ_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_zgeequ_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_zgeequ_cblas_to_fortran() != 0) {
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}