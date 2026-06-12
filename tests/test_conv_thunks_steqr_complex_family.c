#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef int (*fb_csteqr_fn)(fb_layout_t layout, char compz, int n, float *d,
                            float *e, fb_complex_float_t *z, int ldz);
typedef int (*fb_zsteqr_fn)(fb_layout_t layout, char compz, int n, double *d,
                            double *e, fb_complex_double_t *z, int ldz);

typedef void (*fb_csteqr_fortran_slot_fn)(char *compz, int *n, float *d,
                                          float *e, fb_complex_float_t *z,
                                          int *ldz, float *work, int *info);
typedef void (*fb_zsteqr_fortran_slot_fn)(char *compz, int *n, double *d,
                                          double *e, fb_complex_double_t *z,
                                          int *ldz, double *work, int *info);

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

static struct {
    int exec_calls;
    char compz;
    int n;
    int ldz;
    int saw_work;
    fb_complex_float_t z_in[6];
} g_csteqr_fortran_call;

static struct {
    int exec_calls;
    char compz;
    int n;
    int ldz;
    int saw_work;
} g_zsteqr_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    char compz;
    int n;
    int ldz;
} g_csteqr_cblas_call;

static struct {
    int called;
    fb_layout_t layout;
    char compz;
    int n;
    int ldz;
} g_zsteqr_cblas_call;

static int g_csteqr_cblas_rc = 0;
static int g_zsteqr_cblas_rc = 0;

static void stub_csteqr_fortran(char *compz, int *n, float *d, float *e,
                                fb_complex_float_t *z, int *ldz, float *work,
                                int *info)
{
    g_csteqr_fortran_call.exec_calls += 1;
    g_csteqr_fortran_call.compz = *compz;
    g_csteqr_fortran_call.n = *n;
    g_csteqr_fortran_call.ldz = *ldz;
    g_csteqr_fortran_call.saw_work = (work != NULL);
    memcpy(g_csteqr_fortran_call.z_in, z, sizeof(g_csteqr_fortran_call.z_in));
    d[0] = 1.5f;
    e[0] = 2.5f;
    z[0] = make_cfloat(101.0f);
    z[1] = make_cfloat(102.0f);
    z[2] = make_cfloat(103.0f);
    z[3] = make_cfloat(201.0f);
    z[4] = make_cfloat(202.0f);
    z[5] = make_cfloat(203.0f);
    z[6] = make_cfloat(301.0f);
    z[7] = make_cfloat(302.0f);
    z[8] = make_cfloat(303.0f);
    *info = 0;
}

static void stub_zsteqr_fortran(char *compz, int *n, double *d, double *e,
                                fb_complex_double_t *z, int *ldz, double *work,
                                int *info)
{
    g_zsteqr_fortran_call.exec_calls += 1;
    g_zsteqr_fortran_call.compz = *compz;
    g_zsteqr_fortran_call.n = *n;
    g_zsteqr_fortran_call.ldz = *ldz;
    g_zsteqr_fortran_call.saw_work = (work != NULL);
    d[0] = 3.5;
    e[0] = 4.5;
    z[0] = make_cdouble(401.0);
    z[1] = make_cdouble(402.0);
    z[2] = make_cdouble(403.0);
    z[3] = make_cdouble(501.0);
    z[4] = make_cdouble(502.0);
    z[5] = make_cdouble(503.0);
    z[6] = make_cdouble(601.0);
    z[7] = make_cdouble(602.0);
    z[8] = make_cdouble(603.0);
    *info = 0;
}

static int stub_csteqr_cblas(fb_layout_t layout, char compz, int n, float *d,
                             float *e, fb_complex_float_t *z, int ldz)
{
    g_csteqr_cblas_call.called += 1;
    g_csteqr_cblas_call.layout = layout;
    g_csteqr_cblas_call.compz = compz;
    g_csteqr_cblas_call.n = n;
    g_csteqr_cblas_call.ldz = ldz;
    d[0] = 5.5f;
    e[0] = 6.5f;
    z[0] = make_cfloat(701.0f);
    return g_csteqr_cblas_rc;
}

static int stub_zsteqr_cblas(fb_layout_t layout, char compz, int n, double *d,
                             double *e, fb_complex_double_t *z, int ldz)
{
    g_zsteqr_cblas_call.called += 1;
    g_zsteqr_cblas_call.layout = layout;
    g_zsteqr_cblas_call.compz = compz;
    g_zsteqr_cblas_call.n = n;
    g_zsteqr_cblas_call.ldz = ldz;
    d[0] = 7.5;
    e[0] = 8.5;
    z[0] = make_cdouble(801.0);
    return g_zsteqr_cblas_rc;
}

static int check_csteqr_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_csteqr_fn thunk = NULL;
    float d[3] = { 0.0f, 0.0f, 0.0f };
    float e[2] = { 0.0f, 0.0f };
    fb_complex_float_t z[9] = {
        make_cfloat(11.0f), make_cfloat(12.0f), make_cfloat(13.0f),
        make_cfloat(21.0f), make_cfloat(22.0f), make_cfloat(23.0f),
        make_cfloat(31.0f), make_cfloat(32.0f), make_cfloat(33.0f)
    };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_csteqr_fortran_call, 0, sizeof(g_csteqr_fortran_call));

    vtable.ext_ops[FB_OP_CSTEQR][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_csteqr_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CSTEQR);

    thunk = (fb_csteqr_fn)vtable.ext_ops[FB_OP_CSTEQR][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CSTEQR Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 'V', 3, d, e, z, 3);
    if (info != 0 || g_csteqr_fortran_call.exec_calls != 1 ||
        g_csteqr_fortran_call.compz != 'V' || g_csteqr_fortran_call.n != 3 ||
        g_csteqr_fortran_call.ldz != 3 || !g_csteqr_fortran_call.saw_work ||
        cfloat_real(g_csteqr_fortran_call.z_in[0]) != 11.0f ||
        cfloat_real(g_csteqr_fortran_call.z_in[1]) != 21.0f ||
        cfloat_real(g_csteqr_fortran_call.z_in[2]) != 31.0f ||
        cfloat_real(g_csteqr_fortran_call.z_in[3]) != 12.0f ||
        cfloat_real(g_csteqr_fortran_call.z_in[4]) != 22.0f ||
        cfloat_real(g_csteqr_fortran_call.z_in[5]) != 32.0f ||
        d[0] != 1.5f || e[0] != 2.5f || cfloat_real(z[0]) != 101.0f ||
        cfloat_real(z[1]) != 201.0f || cfloat_real(z[2]) != 301.0f ||
        cfloat_real(z[3]) != 102.0f || cfloat_real(z[4]) != 202.0f ||
        cfloat_real(z[5]) != 302.0f) {
        fprintf(stderr, "[FAIL] CSTEQR Fortran->CBLAS thunk did not transpose row-major complex vectors correctly\n");
        return 1;
    }

    printf("[PASS] CSTEQR Fortran->CBLAS thunk allocates fixed workspace and round-trips row-major complex eigenvectors\n");
    return 0;
}

static int check_csteqr_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_csteqr_fortran_slot_fn thunk = NULL;
    float d[3] = { 0.0f, 0.0f, 0.0f };
    float e[2] = { 0.0f, 0.0f };
    fb_complex_float_t z[9] = { make_cfloat(0.0f) };
    float work[8] = { 0.0f };
    char compz = 'I';
    int n = 3;
    int ldz = 3;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_csteqr_cblas_call, 0, sizeof(g_csteqr_cblas_call));
    g_csteqr_cblas_rc = 289;

    vtable.ext_ops[FB_OP_CSTEQR][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_csteqr_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CSTEQR);

    thunk = (fb_csteqr_fortran_slot_fn)vtable.ext_ops[FB_OP_CSTEQR][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CSTEQR CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&compz, &n, d, e, z, &ldz, work, &info);
    if (info != 289 || g_csteqr_cblas_call.called != 1 ||
        g_csteqr_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_csteqr_cblas_call.compz != 'I' || g_csteqr_cblas_call.n != 3 ||
        g_csteqr_cblas_call.ldz != 3 || d[0] != 5.5f || e[0] != 6.5f ||
        cfloat_real(z[0]) != 701.0f) {
        fprintf(stderr, "[FAIL] CSTEQR CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] CSTEQR CBLAS->Fortran thunk maps complex QR-iteration arguments into the C entry\n");
    return 0;
}

static int check_zsteqr_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_zsteqr_fn thunk = NULL;
    double d[3] = { 0.0, 0.0, 0.0 };
    double e[2] = { 0.0, 0.0 };
    fb_complex_double_t z[9] = { make_cdouble(0.0) };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zsteqr_fortran_call, 0, sizeof(g_zsteqr_fortran_call));

    vtable.ext_ops[FB_OP_ZSTEQR][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_zsteqr_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_ZSTEQR);

    thunk = (fb_zsteqr_fn)vtable.ext_ops[FB_OP_ZSTEQR][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZSTEQR Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 'I', 3, d, e, z, 3);
    if (info != 0 || g_zsteqr_fortran_call.exec_calls != 1 ||
        g_zsteqr_fortran_call.compz != 'I' || g_zsteqr_fortran_call.n != 3 ||
        g_zsteqr_fortran_call.ldz != 3 || !g_zsteqr_fortran_call.saw_work ||
        d[0] != 3.5 || e[0] != 4.5 || cdouble_real(z[0]) != 401.0 ||
        cdouble_real(z[1]) != 501.0 || cdouble_real(z[2]) != 601.0 ||
        cdouble_real(z[3]) != 402.0 || cdouble_real(z[4]) != 502.0 ||
        cdouble_real(z[5]) != 602.0) {
        fprintf(stderr, "[FAIL] ZSTEQR Fortran->CBLAS thunk did not export generated row-major complex eigenvectors correctly\n");
        return 1;
    }

    printf("[PASS] ZSTEQR Fortran->CBLAS thunk allocates fixed workspace and exports generated row-major complex eigenvectors\n");
    return 0;
}

static int check_zsteqr_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_zsteqr_fortran_slot_fn thunk = NULL;
    double d[3] = { 0.0, 0.0, 0.0 };
    double e[2] = { 0.0, 0.0 };
    fb_complex_double_t z[9] = { make_cdouble(0.0) };
    double work[8] = { 0.0 };
    char compz = 'V';
    int n = 3;
    int ldz = 3;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zsteqr_cblas_call, 0, sizeof(g_zsteqr_cblas_call));
    g_zsteqr_cblas_rc = 291;

    vtable.ext_ops[FB_OP_ZSTEQR][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_zsteqr_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_ZSTEQR);

    thunk = (fb_zsteqr_fortran_slot_fn)vtable.ext_ops[FB_OP_ZSTEQR][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZSTEQR CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&compz, &n, d, e, z, &ldz, work, &info);
    if (info != 291 || g_zsteqr_cblas_call.called != 1 ||
        g_zsteqr_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_zsteqr_cblas_call.compz != 'V' || g_zsteqr_cblas_call.n != 3 ||
        g_zsteqr_cblas_call.ldz != 3 || d[0] != 7.5 || e[0] != 8.5 ||
        cdouble_real(z[0]) != 801.0) {
        fprintf(stderr, "[FAIL] ZSTEQR CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] ZSTEQR CBLAS->Fortran thunk maps double-precision complex QR-iteration arguments into the C entry\n");
    return 0;
}

int main(void)
{
    if (check_csteqr_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_csteqr_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_zsteqr_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_zsteqr_cblas_to_fortran() != 0) {
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}