#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef int (*fb_ssteqr_fn)(fb_layout_t layout, char compz, int n, float *d,
                            float *e, float *z, int ldz);
typedef int (*fb_dsteqr_fn)(fb_layout_t layout, char compz, int n, double *d,
                            double *e, double *z, int ldz);

typedef void (*fb_ssteqr_fortran_slot_fn)(char *compz, int *n, float *d,
                                          float *e, float *z, int *ldz,
                                          float *work, int *info);
typedef void (*fb_dsteqr_fortran_slot_fn)(char *compz, int *n, double *d,
                                          double *e, double *z, int *ldz,
                                          double *work, int *info);

static struct {
    int exec_calls;
    char compz;
    int n;
    int ldz;
    int saw_work;
    float z_in[6];
} g_ssteqr_fortran_call;

static struct {
    int exec_calls;
    char compz;
    int n;
    int ldz;
    int saw_work;
} g_dsteqr_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    char compz;
    int n;
    int ldz;
} g_ssteqr_cblas_call;

static struct {
    int called;
    fb_layout_t layout;
    char compz;
    int n;
    int ldz;
} g_dsteqr_cblas_call;

static int g_ssteqr_cblas_rc = 0;
static int g_dsteqr_cblas_rc = 0;

static void stub_ssteqr_fortran(char *compz, int *n, float *d, float *e,
                                float *z, int *ldz, float *work, int *info)
{
    (void)d;
    (void)e;
    g_ssteqr_fortran_call.exec_calls += 1;
    g_ssteqr_fortran_call.compz = *compz;
    g_ssteqr_fortran_call.n = *n;
    g_ssteqr_fortran_call.ldz = *ldz;
    g_ssteqr_fortran_call.saw_work = (work != NULL);
    memcpy(g_ssteqr_fortran_call.z_in, z, sizeof(g_ssteqr_fortran_call.z_in));
    d[0] = 1.5f;
    e[0] = 2.5f;
    z[0] = 101.0f;
    z[1] = 102.0f;
    z[2] = 103.0f;
    z[3] = 201.0f;
    z[4] = 202.0f;
    z[5] = 203.0f;
    z[6] = 301.0f;
    z[7] = 302.0f;
    z[8] = 303.0f;
    *info = 0;
}

static void stub_dsteqr_fortran(char *compz, int *n, double *d, double *e,
                                double *z, int *ldz, double *work, int *info)
{
    (void)d;
    (void)e;
    g_dsteqr_fortran_call.exec_calls += 1;
    g_dsteqr_fortran_call.compz = *compz;
    g_dsteqr_fortran_call.n = *n;
    g_dsteqr_fortran_call.ldz = *ldz;
    g_dsteqr_fortran_call.saw_work = (work != NULL);
    d[0] = 3.5;
    e[0] = 4.5;
    z[0] = 401.0;
    z[1] = 402.0;
    z[2] = 403.0;
    z[3] = 501.0;
    z[4] = 502.0;
    z[5] = 503.0;
    z[6] = 601.0;
    z[7] = 602.0;
    z[8] = 603.0;
    *info = 0;
}

static int stub_ssteqr_cblas(fb_layout_t layout, char compz, int n, float *d,
                             float *e, float *z, int ldz)
{
    g_ssteqr_cblas_call.called += 1;
    g_ssteqr_cblas_call.layout = layout;
    g_ssteqr_cblas_call.compz = compz;
    g_ssteqr_cblas_call.n = n;
    g_ssteqr_cblas_call.ldz = ldz;
    d[0] = 5.5f;
    e[0] = 6.5f;
    z[0] = 701.0f;
    return g_ssteqr_cblas_rc;
}

static int stub_dsteqr_cblas(fb_layout_t layout, char compz, int n, double *d,
                             double *e, double *z, int ldz)
{
    g_dsteqr_cblas_call.called += 1;
    g_dsteqr_cblas_call.layout = layout;
    g_dsteqr_cblas_call.compz = compz;
    g_dsteqr_cblas_call.n = n;
    g_dsteqr_cblas_call.ldz = ldz;
    d[0] = 7.5;
    e[0] = 8.5;
    z[0] = 801.0;
    return g_dsteqr_cblas_rc;
}

static int check_ssteqr_fortran_to_cblas(void)
{
    static const float expected_z[9] = {
        101.0f, 201.0f, 301.0f,
        102.0f, 202.0f, 302.0f,
        103.0f, 203.0f, 303.0f
    };
    fb_backend_vtable_t vtable;
    fb_ssteqr_fn thunk = NULL;
    float d[3] = { 0.0f, 0.0f, 0.0f };
    float e[2] = { 0.0f, 0.0f };
    float z[9] = {
        11.0f, 12.0f, 13.0f,
        21.0f, 22.0f, 23.0f,
        31.0f, 32.0f, 33.0f
    };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_ssteqr_fortran_call, 0, sizeof(g_ssteqr_fortran_call));

    vtable.ext_ops[FB_OP_SSTEQR][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_ssteqr_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SSTEQR);

    thunk = (fb_ssteqr_fn)vtable.ext_ops[FB_OP_SSTEQR][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SSTEQR Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 'V', 3, d, e, z, 3);
    if (info != 0 || g_ssteqr_fortran_call.exec_calls != 1 ||
        g_ssteqr_fortran_call.compz != 'V' || g_ssteqr_fortran_call.n != 3 ||
        g_ssteqr_fortran_call.ldz != 3 || !g_ssteqr_fortran_call.saw_work ||
        g_ssteqr_fortran_call.z_in[0] != 11.0f ||
        g_ssteqr_fortran_call.z_in[1] != 21.0f ||
        g_ssteqr_fortran_call.z_in[2] != 31.0f ||
        g_ssteqr_fortran_call.z_in[3] != 12.0f ||
        g_ssteqr_fortran_call.z_in[4] != 22.0f ||
        g_ssteqr_fortran_call.z_in[5] != 32.0f || d[0] != 1.5f ||
        e[0] != 2.5f || memcmp(z, expected_z, sizeof(expected_z)) != 0) {
        fprintf(stderr, "[FAIL] SSTEQR Fortran->CBLAS thunk did not transpose row-major vectors correctly\n");
        return 1;
    }

    printf("[PASS] SSTEQR Fortran->CBLAS thunk allocates fixed workspace and round-trips row-major eigenvectors\n");
    return 0;
}

static int check_ssteqr_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_ssteqr_fortran_slot_fn thunk = NULL;
    float d[3] = { 0.0f, 0.0f, 0.0f };
    float e[2] = { 0.0f, 0.0f };
    float z[9] = { 0.0f };
    float work[8] = { 0.0f };
    char compz = 'V';
    int n = 3;
    int ldz = 3;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_ssteqr_cblas_call, 0, sizeof(g_ssteqr_cblas_call));
    g_ssteqr_cblas_rc = 273;

    vtable.ext_ops[FB_OP_SSTEQR][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_ssteqr_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SSTEQR);

    thunk = (fb_ssteqr_fortran_slot_fn)vtable.ext_ops[FB_OP_SSTEQR][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SSTEQR CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&compz, &n, d, e, z, &ldz, work, &info);
    if (info != 273 || g_ssteqr_cblas_call.called != 1 ||
        g_ssteqr_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_ssteqr_cblas_call.compz != 'V' || g_ssteqr_cblas_call.n != 3 ||
        g_ssteqr_cblas_call.ldz != 3 || d[0] != 5.5f || e[0] != 6.5f ||
        z[0] != 701.0f) {
        fprintf(stderr, "[FAIL] SSTEQR CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] SSTEQR CBLAS->Fortran thunk maps QR-iteration arguments into the C entry\n");
    return 0;
}

static int check_dsteqr_fortran_to_cblas(void)
{
    static const double expected_z[9] = {
        401.0, 501.0, 601.0,
        402.0, 502.0, 602.0,
        403.0, 503.0, 603.0
    };
    fb_backend_vtable_t vtable;
    fb_dsteqr_fn thunk = NULL;
    double d[3] = { 0.0, 0.0, 0.0 };
    double e[2] = { 0.0, 0.0 };
    double z[9] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dsteqr_fortran_call, 0, sizeof(g_dsteqr_fortran_call));

    vtable.ext_ops[FB_OP_DSTEQR][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_dsteqr_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_DSTEQR);

    thunk = (fb_dsteqr_fn)vtable.ext_ops[FB_OP_DSTEQR][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DSTEQR Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 'I', 3, d, e, z, 3);
    if (info != 0 || g_dsteqr_fortran_call.exec_calls != 1 ||
        g_dsteqr_fortran_call.compz != 'I' || g_dsteqr_fortran_call.n != 3 ||
        g_dsteqr_fortran_call.ldz != 3 || !g_dsteqr_fortran_call.saw_work ||
        d[0] != 3.5 || e[0] != 4.5 || memcmp(z, expected_z, sizeof(expected_z)) != 0) {
        fprintf(stderr, "[FAIL] DSTEQR Fortran->CBLAS thunk did not handle generated eigenvectors correctly\n");
        return 1;
    }

    printf("[PASS] DSTEQR Fortran->CBLAS thunk allocates fixed workspace and exports generated row-major eigenvectors\n");
    return 0;
}

static int check_dsteqr_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_dsteqr_fortran_slot_fn thunk = NULL;
    double d[3] = { 0.0, 0.0, 0.0 };
    double e[2] = { 0.0, 0.0 };
    double z[9] = { 0.0 };
    double work[8] = { 0.0 };
    char compz = 'I';
    int n = 3;
    int ldz = 3;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dsteqr_cblas_call, 0, sizeof(g_dsteqr_cblas_call));
    g_dsteqr_cblas_rc = 277;

    vtable.ext_ops[FB_OP_DSTEQR][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_dsteqr_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_DSTEQR);

    thunk = (fb_dsteqr_fortran_slot_fn)vtable.ext_ops[FB_OP_DSTEQR][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DSTEQR CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&compz, &n, d, e, z, &ldz, work, &info);
    if (info != 277 || g_dsteqr_cblas_call.called != 1 ||
        g_dsteqr_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_dsteqr_cblas_call.compz != 'I' || g_dsteqr_cblas_call.n != 3 ||
        g_dsteqr_cblas_call.ldz != 3 || d[0] != 7.5 || e[0] != 8.5 ||
        z[0] != 801.0) {
        fprintf(stderr, "[FAIL] DSTEQR CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] DSTEQR CBLAS->Fortran thunk maps double-precision QR-iteration arguments into the C entry\n");
    return 0;
}

int main(void)
{
    if (check_ssteqr_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_ssteqr_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_dsteqr_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_dsteqr_cblas_to_fortran() != 0) {
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}