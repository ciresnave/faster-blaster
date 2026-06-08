#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef int (*fb_ssterf_fn)(int n, float *d, float *e);
typedef int (*fb_dsterf_fn)(int n, double *d, double *e);

typedef void (*fb_ssterf_fortran_slot_fn)(int *n, float *d, float *e, int *info);
typedef void (*fb_dsterf_fortran_slot_fn)(int *n, double *d, double *e, int *info);

static struct {
    int called;
    int n;
} g_ssterf_fortran_call;

static struct {
    int called;
    int n;
} g_dsterf_fortran_call;

static struct {
    int called;
    int n;
} g_ssterf_cblas_call;

static struct {
    int called;
    int n;
} g_dsterf_cblas_call;

static int g_ssterf_cblas_rc = 0;
static int g_dsterf_cblas_rc = 0;

static void stub_ssterf_fortran(int *n, float *d, float *e, int *info)
{
    g_ssterf_fortran_call.called += 1;
    g_ssterf_fortran_call.n = *n;
    d[0] = 41.0f;
    e[0] = 42.0f;
    *info = 0;
}

static void stub_dsterf_fortran(int *n, double *d, double *e, int *info)
{
    g_dsterf_fortran_call.called += 1;
    g_dsterf_fortran_call.n = *n;
    d[0] = 43.0;
    e[0] = 44.0;
    *info = 0;
}

static int stub_ssterf_cblas(int n, float *d, float *e)
{
    g_ssterf_cblas_call.called += 1;
    g_ssterf_cblas_call.n = n;
    d[0] = 45.0f;
    e[0] = 46.0f;
    return g_ssterf_cblas_rc;
}

static int stub_dsterf_cblas(int n, double *d, double *e)
{
    g_dsterf_cblas_call.called += 1;
    g_dsterf_cblas_call.n = n;
    d[0] = 47.0;
    e[0] = 48.0;
    return g_dsterf_cblas_rc;
}

static int check_ssterf_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_ssterf_fn thunk = NULL;
    float d[3] = { 0.0f, 0.0f, 0.0f };
    float e[2] = { 0.0f, 0.0f };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_ssterf_fortran_call, 0, sizeof(g_ssterf_fortran_call));

    vtable.ext_ops[FB_OP_SSTERF][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_ssterf_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SSTERF);

    thunk = (fb_ssterf_fn)vtable.ext_ops[FB_OP_SSTERF][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SSTERF Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(3, d, e);
    if (info != 0 || g_ssterf_fortran_call.called != 1 ||
        g_ssterf_fortran_call.n != 3 || d[0] != 41.0f || e[0] != 42.0f) {
        fprintf(stderr, "[FAIL] SSTERF Fortran->CBLAS thunk did not forward direct tridiagonal eigenvalue arguments\n");
        return 1;
    }

    printf("[PASS] SSTERF Fortran->CBLAS thunk forwards direct tridiagonal eigenvalue arguments\n");
    return 0;
}

static int check_ssterf_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_ssterf_fortran_slot_fn thunk = NULL;
    float d[3] = { 0.0f, 0.0f, 0.0f };
    float e[2] = { 0.0f, 0.0f };
    int n = 3;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_ssterf_cblas_call, 0, sizeof(g_ssterf_cblas_call));
    g_ssterf_cblas_rc = 269;

    vtable.ext_ops[FB_OP_SSTERF][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_ssterf_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SSTERF);

    thunk = (fb_ssterf_fortran_slot_fn)vtable.ext_ops[FB_OP_SSTERF][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SSTERF CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&n, d, e, &info);
    if (info != 269 || g_ssterf_cblas_call.called != 1 ||
        g_ssterf_cblas_call.n != 3 || d[0] != 45.0f || e[0] != 46.0f) {
        fprintf(stderr, "[FAIL] SSTERF CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] SSTERF CBLAS->Fortran thunk maps direct tridiagonal eigenvalue arguments into the C entry\n");
    return 0;
}

static int check_dsterf_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_dsterf_fn thunk = NULL;
    double d[3] = { 0.0, 0.0, 0.0 };
    double e[2] = { 0.0, 0.0 };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dsterf_fortran_call, 0, sizeof(g_dsterf_fortran_call));

    vtable.ext_ops[FB_OP_DSTERF][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_dsterf_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_DSTERF);

    thunk = (fb_dsterf_fn)vtable.ext_ops[FB_OP_DSTERF][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DSTERF Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(3, d, e);
    if (info != 0 || g_dsterf_fortran_call.called != 1 ||
        g_dsterf_fortran_call.n != 3 || d[0] != 43.0 || e[0] != 44.0) {
        fprintf(stderr, "[FAIL] DSTERF Fortran->CBLAS thunk did not forward double-precision tridiagonal eigenvalue arguments\n");
        return 1;
    }

    printf("[PASS] DSTERF Fortran->CBLAS thunk forwards double-precision tridiagonal eigenvalue arguments\n");
    return 0;
}

static int check_dsterf_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_dsterf_fortran_slot_fn thunk = NULL;
    double d[3] = { 0.0, 0.0, 0.0 };
    double e[2] = { 0.0, 0.0 };
    int n = 3;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dsterf_cblas_call, 0, sizeof(g_dsterf_cblas_call));
    g_dsterf_cblas_rc = 271;

    vtable.ext_ops[FB_OP_DSTERF][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_dsterf_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_DSTERF);

    thunk = (fb_dsterf_fortran_slot_fn)vtable.ext_ops[FB_OP_DSTERF][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DSTERF CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&n, d, e, &info);
    if (info != 271 || g_dsterf_cblas_call.called != 1 ||
        g_dsterf_cblas_call.n != 3 || d[0] != 47.0 || e[0] != 48.0) {
        fprintf(stderr, "[FAIL] DSTERF CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] DSTERF CBLAS->Fortran thunk maps double-precision tridiagonal eigenvalue arguments into the C entry\n");
    return 0;
}

int main(void)
{
    if (check_ssterf_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_ssterf_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_dsterf_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_dsterf_cblas_to_fortran() != 0) {
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}