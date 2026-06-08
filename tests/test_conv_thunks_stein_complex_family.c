#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef int (*fb_cstein_fn)(fb_layout_t layout, int n, const float *d,
                            const float *e, int m, const float *w,
                            const int *iblock, const int *isplit,
                            fb_complex_float_t *z, int ldz, int *ifailv);
typedef int (*fb_zstein_fn)(fb_layout_t layout, int n, const double *d,
                            const double *e, int m, const double *w,
                            const int *iblock, const int *isplit,
                            fb_complex_double_t *z, int ldz, int *ifailv);

typedef void (*fb_cstein_fortran_slot_fn)(int *n, float *d, float *e, int *m,
                                          float *w, int *iblock, int *isplit,
                                          fb_complex_float_t *z, int *ldz,
                                          float *work, int *iwork,
                                          int *ifailv, int *info);
typedef void (*fb_zstein_fortran_slot_fn)(int *n, double *d, double *e, int *m,
                                          double *w, int *iblock, int *isplit,
                                          fb_complex_double_t *z, int *ldz,
                                          double *work, int *iwork,
                                          int *ifailv, int *info);

static fb_complex_float_t make_cfloat(float real_value)
{
    fb_complex_float_t value;

    __real__ value = real_value;
    __imag__ value = 0.0f;
    return value;
}

static fb_complex_double_t make_cdouble(double real_value)
{
    fb_complex_double_t value;

    __real__ value = real_value;
    __imag__ value = 0.0;
    return value;
}

static float cfloat_real(fb_complex_float_t value)
{
    return __real__ value;
}

static double cdouble_real(fb_complex_double_t value)
{
    return __real__ value;
}

static int cfloat_reals_match(const fb_complex_float_t *values,
                              const float *expected, int count)
{
    int index = 0;

    for (index = 0; index < count; ++index) {
        if (cfloat_real(values[index]) != expected[index]) {
            return 0;
        }
    }
    return 1;
}

static int cdouble_reals_match(const fb_complex_double_t *values,
                               const double *expected, int count)
{
    int index = 0;

    for (index = 0; index < count; ++index) {
        if (cdouble_real(values[index]) != expected[index]) {
            return 0;
        }
    }
    return 1;
}

static struct {
    int exec_calls;
    int saw_work;
    int saw_iwork;
    int exec_ldz;
} g_cstein_fortran_call;

static struct {
    int exec_calls;
    int saw_work;
    int saw_iwork;
    int exec_ldz;
} g_zstein_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    int n;
    int m;
    int ldz;
} g_cstein_cblas_call;

static struct {
    int called;
    fb_layout_t layout;
    int n;
    int m;
    int ldz;
} g_zstein_cblas_call;

static int g_cstein_cblas_rc = 0;
static int g_zstein_cblas_rc = 0;

static void stub_cstein_fortran(int *n, float *d, float *e, int *m, float *w,
                                int *iblock, int *isplit,
                                fb_complex_float_t *z, int *ldz, float *work,
                                int *iwork, int *ifailv, int *info)
{
    (void)n;
    (void)d;
    (void)e;
    (void)m;
    (void)w;
    (void)iblock;
    (void)isplit;
    g_cstein_fortran_call.exec_calls += 1;
    g_cstein_fortran_call.saw_work = (work != NULL);
    g_cstein_fortran_call.saw_iwork = (iwork != NULL);
    g_cstein_fortran_call.exec_ldz = *ldz;
    z[0] = make_cfloat(25.0f);
    z[1] = make_cfloat(26.0f);
    z[2] = make_cfloat(27.0f);
    z[3] = make_cfloat(35.0f);
    z[4] = make_cfloat(36.0f);
    z[5] = make_cfloat(37.0f);
    ifailv[0] = 1;
    ifailv[1] = 0;
    *info = 0;
}

static void stub_zstein_fortran(int *n, double *d, double *e, int *m,
                                double *w, int *iblock, int *isplit,
                                fb_complex_double_t *z, int *ldz,
                                double *work, int *iwork, int *ifailv,
                                int *info)
{
    (void)n;
    (void)d;
    (void)e;
    (void)m;
    (void)w;
    (void)iblock;
    (void)isplit;
    g_zstein_fortran_call.exec_calls += 1;
    g_zstein_fortran_call.saw_work = (work != NULL);
    g_zstein_fortran_call.saw_iwork = (iwork != NULL);
    g_zstein_fortran_call.exec_ldz = *ldz;
    z[0] = make_cdouble(45.0);
    z[1] = make_cdouble(46.0);
    z[2] = make_cdouble(47.0);
    z[3] = make_cdouble(55.0);
    z[4] = make_cdouble(56.0);
    z[5] = make_cdouble(57.0);
    ifailv[0] = 2;
    ifailv[1] = 0;
    *info = 0;
}

static int stub_cstein_cblas(fb_layout_t layout, int n, const float *d,
                             const float *e, int m, const float *w,
                             const int *iblock, const int *isplit,
                             fb_complex_float_t *z, int ldz, int *ifailv)
{
    (void)d;
    (void)e;
    (void)w;
    (void)iblock;
    (void)isplit;
    g_cstein_cblas_call.called += 1;
    g_cstein_cblas_call.layout = layout;
    g_cstein_cblas_call.n = n;
    g_cstein_cblas_call.m = m;
    g_cstein_cblas_call.ldz = ldz;
    z[0] = make_cfloat(31.0f);
    z[1] = make_cfloat(32.0f);
    ifailv[0] = 3;
    ifailv[1] = 0;
    return g_cstein_cblas_rc;
}

static int stub_zstein_cblas(fb_layout_t layout, int n, const double *d,
                             const double *e, int m, const double *w,
                             const int *iblock, const int *isplit,
                             fb_complex_double_t *z, int ldz, int *ifailv)
{
    (void)d;
    (void)e;
    (void)w;
    (void)iblock;
    (void)isplit;
    g_zstein_cblas_call.called += 1;
    g_zstein_cblas_call.layout = layout;
    g_zstein_cblas_call.n = n;
    g_zstein_cblas_call.m = m;
    g_zstein_cblas_call.ldz = ldz;
    z[0] = make_cdouble(41.0);
    z[1] = make_cdouble(42.0);
    ifailv[0] = 4;
    ifailv[1] = 0;
    return g_zstein_cblas_rc;
}

static int check_cstein_fortran_to_cblas(void)
{
    static const float expected_z[6] = {
        25.0f, 35.0f,
        26.0f, 36.0f,
        27.0f, 37.0f
    };
    fb_backend_vtable_t vtable;
    fb_cstein_fn thunk = NULL;
    float d[3] = { 0.0f, 0.0f, 0.0f };
    float e[2] = { 0.0f, 0.0f };
    float w[2] = { 1.0f, 2.0f };
    int iblock[2] = { 1, 1 };
    int isplit[2] = { 3, 0 };
    fb_complex_float_t z[6] = { make_cfloat(0.0f) };
    int ifailv[2] = { 0, 0 };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cstein_fortran_call, 0, sizeof(g_cstein_fortran_call));

    vtable.ext_ops[FB_OP_CSTEIN][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_cstein_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CSTEIN);

    thunk = (fb_cstein_fn)vtable.ext_ops[FB_OP_CSTEIN][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CSTEIN Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 3, d, e, 2, w, iblock, isplit, z, 2,
                 ifailv);
    if (info != 0 || g_cstein_fortran_call.exec_calls != 1 ||
        !g_cstein_fortran_call.saw_work || !g_cstein_fortran_call.saw_iwork ||
        g_cstein_fortran_call.exec_ldz != 3 ||
        !cfloat_reals_match(z, expected_z, 6) || ifailv[0] != 1) {
        fprintf(stderr, "[FAIL] CSTEIN Fortran->CBLAS thunk did not allocate fixed workspaces or export row-major complex inverse-iteration vectors\n");
        return 1;
    }

    printf("[PASS] CSTEIN Fortran->CBLAS thunk allocates fixed workspaces and exports row-major complex inverse-iteration vectors\n");
    return 0;
}

static int check_cstein_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_cstein_fortran_slot_fn thunk = NULL;
    float d[3] = { 0.0f, 0.0f, 0.0f };
    float e[2] = { 0.0f, 0.0f };
    float w[2] = { 1.0f, 2.0f };
    int iblock[2] = { 1, 1 };
    int isplit[2] = { 3, 0 };
    fb_complex_float_t z[6] = { make_cfloat(0.0f) };
    float work[16] = { 0.0f };
    int iwork[16] = { 0 };
    int ifailv[2] = { 0, 0 };
    int n = 3;
    int m = 2;
    int ldz = 3;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cstein_cblas_call, 0, sizeof(g_cstein_cblas_call));
    g_cstein_cblas_rc = 301;

    vtable.ext_ops[FB_OP_CSTEIN][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_cstein_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CSTEIN);

    thunk = (fb_cstein_fortran_slot_fn)vtable.ext_ops[FB_OP_CSTEIN][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CSTEIN CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&n, d, e, &m, w, iblock, isplit, z, &ldz, work, iwork, ifailv,
          &info);
    if (info != 301 || g_cstein_cblas_call.called != 1 ||
        g_cstein_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_cstein_cblas_call.n != 3 || g_cstein_cblas_call.m != 2 ||
        g_cstein_cblas_call.ldz != 3 || cfloat_real(z[0]) != 31.0f ||
        cfloat_real(z[1]) != 32.0f || ifailv[0] != 3) {
        fprintf(stderr, "[FAIL] CSTEIN CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] CSTEIN CBLAS->Fortran thunk maps complex inverse-iteration arguments into the C entry\n");
    return 0;
}

static int check_zstein_fortran_to_cblas(void)
{
    static const double expected_z[6] = {
        45.0, 55.0,
        46.0, 56.0,
        47.0, 57.0
    };
    fb_backend_vtable_t vtable;
    fb_zstein_fn thunk = NULL;
    double d[3] = { 0.0, 0.0, 0.0 };
    double e[2] = { 0.0, 0.0 };
    double w[2] = { 3.0, 4.0 };
    int iblock[2] = { 1, 1 };
    int isplit[2] = { 3, 0 };
    fb_complex_double_t z[6] = { make_cdouble(0.0) };
    int ifailv[2] = { 0, 0 };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zstein_fortran_call, 0, sizeof(g_zstein_fortran_call));

    vtable.ext_ops[FB_OP_ZSTEIN][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_zstein_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_ZSTEIN);

    thunk = (fb_zstein_fn)vtable.ext_ops[FB_OP_ZSTEIN][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZSTEIN Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 3, d, e, 2, w, iblock, isplit, z, 2,
                 ifailv);
    if (info != 0 || g_zstein_fortran_call.exec_calls != 1 ||
        !g_zstein_fortran_call.saw_work || !g_zstein_fortran_call.saw_iwork ||
        g_zstein_fortran_call.exec_ldz != 3 ||
        !cdouble_reals_match(z, expected_z, 6) || ifailv[0] != 2) {
        fprintf(stderr, "[FAIL] ZSTEIN Fortran->CBLAS thunk did not allocate fixed workspaces or export row-major double-precision complex inverse-iteration vectors\n");
        return 1;
    }

    printf("[PASS] ZSTEIN Fortran->CBLAS thunk allocates fixed workspaces and exports row-major double-precision complex inverse-iteration vectors\n");
    return 0;
}

static int check_zstein_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_zstein_fortran_slot_fn thunk = NULL;
    double d[3] = { 0.0, 0.0, 0.0 };
    double e[2] = { 0.0, 0.0 };
    double w[2] = { 3.0, 4.0 };
    int iblock[2] = { 1, 1 };
    int isplit[2] = { 3, 0 };
    fb_complex_double_t z[6] = { make_cdouble(0.0) };
    double work[16] = { 0.0 };
    int iwork[16] = { 0 };
    int ifailv[2] = { 0, 0 };
    int n = 3;
    int m = 2;
    int ldz = 3;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zstein_cblas_call, 0, sizeof(g_zstein_cblas_call));
    g_zstein_cblas_rc = 303;

    vtable.ext_ops[FB_OP_ZSTEIN][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_zstein_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_ZSTEIN);

    thunk = (fb_zstein_fortran_slot_fn)vtable.ext_ops[FB_OP_ZSTEIN][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZSTEIN CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&n, d, e, &m, w, iblock, isplit, z, &ldz, work, iwork, ifailv,
          &info);
    if (info != 303 || g_zstein_cblas_call.called != 1 ||
        g_zstein_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_zstein_cblas_call.n != 3 || g_zstein_cblas_call.m != 2 ||
        g_zstein_cblas_call.ldz != 3 || cdouble_real(z[0]) != 41.0 ||
        cdouble_real(z[1]) != 42.0 || ifailv[0] != 4) {
        fprintf(stderr, "[FAIL] ZSTEIN CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] ZSTEIN CBLAS->Fortran thunk maps double-precision complex inverse-iteration arguments into the C entry\n");
    return 0;
}

int main(void)
{
    if (check_cstein_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_cstein_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_zstein_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_zstein_cblas_to_fortran() != 0) {
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}