#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef int (*fb_sstein_fn)(fb_layout_t layout, int n, const float *d,
                            const float *e, int m, const float *w,
                            const int *iblock, const int *isplit, float *z,
                            int ldz, int *ifailv);
typedef int (*fb_dstein_fn)(fb_layout_t layout, int n, const double *d,
                            const double *e, int m, const double *w,
                            const int *iblock, const int *isplit, double *z,
                            int ldz, int *ifailv);

typedef void (*fb_sstein_fortran_slot_fn)(int *n, float *d, float *e, int *m,
                                          float *w, int *iblock, int *isplit,
                                          float *z, int *ldz, float *work,
                                          int *iwork, int *ifailv, int *info);
typedef void (*fb_dstein_fortran_slot_fn)(int *n, double *d, double *e, int *m,
                                          double *w, int *iblock, int *isplit,
                                          double *z, int *ldz, double *work,
                                          int *iwork, int *ifailv, int *info);

static struct {
    int exec_calls;
    int saw_work;
    int saw_iwork;
} g_sstein_fortran_call;

static struct {
    int exec_calls;
    int saw_work;
    int saw_iwork;
} g_dstein_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    int n;
    int m;
    int ldz;
} g_sstein_cblas_call;

static struct {
    int called;
    fb_layout_t layout;
    int n;
    int m;
    int ldz;
} g_dstein_cblas_call;

static int g_sstein_cblas_rc = 0;
static int g_dstein_cblas_rc = 0;

static void stub_sstein_fortran(int *n, float *d, float *e, int *m, float *w,
                                int *iblock, int *isplit, float *z, int *ldz,
                                float *work, int *iwork, int *ifailv,
                                int *info)
{
    (void)n;
    (void)d;
    (void)e;
    (void)w;
    (void)iblock;
    (void)isplit;
    (void)ldz;
    g_sstein_fortran_call.exec_calls += 1;
    g_sstein_fortran_call.saw_work = (work != NULL);
    g_sstein_fortran_call.saw_iwork = (iwork != NULL);
    z[0] = 25.0f;
    z[1] = 26.0f;
    ifailv[0] = 1;
    ifailv[1] = 0;
    *info = 0;
}

static void stub_dstein_fortran(int *n, double *d, double *e, int *m, double *w,
                                int *iblock, int *isplit, double *z, int *ldz,
                                double *work, int *iwork, int *ifailv,
                                int *info)
{
    (void)n;
    (void)d;
    (void)e;
    (void)w;
    (void)iblock;
    (void)isplit;
    (void)ldz;
    g_dstein_fortran_call.exec_calls += 1;
    g_dstein_fortran_call.saw_work = (work != NULL);
    g_dstein_fortran_call.saw_iwork = (iwork != NULL);
    z[0] = 27.0;
    z[1] = 28.0;
    ifailv[0] = 2;
    ifailv[1] = 0;
    *info = 0;
}

static int stub_sstein_cblas(fb_layout_t layout, int n, const float *d,
                             const float *e, int m, const float *w,
                             const int *iblock, const int *isplit, float *z,
                             int ldz, int *ifailv)
{
    (void)d;
    (void)e;
    (void)w;
    (void)iblock;
    (void)isplit;
    g_sstein_cblas_call.called += 1;
    g_sstein_cblas_call.layout = layout;
    g_sstein_cblas_call.n = n;
    g_sstein_cblas_call.m = m;
    g_sstein_cblas_call.ldz = ldz;
    z[0] = 31.0f;
    z[1] = 32.0f;
    ifailv[0] = 3;
    ifailv[1] = 0;
    return g_sstein_cblas_rc;
}

static int stub_dstein_cblas(fb_layout_t layout, int n, const double *d,
                             const double *e, int m, const double *w,
                             const int *iblock, const int *isplit, double *z,
                             int ldz, int *ifailv)
{
    (void)d;
    (void)e;
    (void)w;
    (void)iblock;
    (void)isplit;
    g_dstein_cblas_call.called += 1;
    g_dstein_cblas_call.layout = layout;
    g_dstein_cblas_call.n = n;
    g_dstein_cblas_call.m = m;
    g_dstein_cblas_call.ldz = ldz;
    z[0] = 33.0;
    z[1] = 34.0;
    ifailv[0] = 4;
    ifailv[1] = 0;
    return g_dstein_cblas_rc;
}

static int check_sstein_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_sstein_fn thunk = NULL;
    float d[3] = { 0.0f, 0.0f, 0.0f };
    float e[2] = { 0.0f, 0.0f };
    float w[2] = { 1.0f, 2.0f };
    int iblock[2] = { 1, 1 };
    int isplit[2] = { 3, 0 };
    float z[6] = { 0.0f };
    int ifailv[2] = { 0, 0 };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sstein_fortran_call, 0, sizeof(g_sstein_fortran_call));

    vtable.ext_ops[FB_OP_SSTEIN][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_sstein_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SSTEIN);

    thunk = (fb_sstein_fn)vtable.ext_ops[FB_OP_SSTEIN][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SSTEIN Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 3, d, e, 2, w, iblock, isplit, z, 3,
                 ifailv);
    if (info != 0 || g_sstein_fortran_call.exec_calls != 1 ||
        !g_sstein_fortran_call.saw_work || !g_sstein_fortran_call.saw_iwork ||
        z[0] != 25.0f || z[1] != 26.0f || ifailv[0] != 1) {
        fprintf(stderr, "[FAIL] SSTEIN Fortran->CBLAS thunk did not allocate fixed workspaces and forward tridiagonal inverse-iteration outputs\n");
        return 1;
    }

    printf("[PASS] SSTEIN Fortran->CBLAS thunk allocates fixed workspaces and forwards tridiagonal inverse-iteration outputs\n");
    return 0;
}

static int check_sstein_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_sstein_fortran_slot_fn thunk = NULL;
    float d[3] = { 0.0f, 0.0f, 0.0f };
    float e[2] = { 0.0f, 0.0f };
    float w[2] = { 1.0f, 2.0f };
    int iblock[2] = { 1, 1 };
    int isplit[2] = { 3, 0 };
    float z[6] = { 0.0f };
    float work[16] = { 0.0f };
    int iwork[16] = { 0 };
    int ifailv[2] = { 0, 0 };
    int n = 3;
    int m = 2;
    int ldz = 3;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sstein_cblas_call, 0, sizeof(g_sstein_cblas_call));
    g_sstein_cblas_rc = 265;

    vtable.ext_ops[FB_OP_SSTEIN][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_sstein_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SSTEIN);

    thunk = (fb_sstein_fortran_slot_fn)vtable.ext_ops[FB_OP_SSTEIN][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SSTEIN CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&n, d, e, &m, w, iblock, isplit, z, &ldz, work, iwork, ifailv,
          &info);
    if (info != 265 || g_sstein_cblas_call.called != 1 ||
        g_sstein_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_sstein_cblas_call.n != 3 || g_sstein_cblas_call.m != 2 ||
        g_sstein_cblas_call.ldz != 3 || z[0] != 31.0f || z[1] != 32.0f ||
        ifailv[0] != 3) {
        fprintf(stderr, "[FAIL] SSTEIN CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] SSTEIN CBLAS->Fortran thunk maps tridiagonal inverse-iteration arguments into the C entry\n");
    return 0;
}

static int check_dstein_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_dstein_fn thunk = NULL;
    double d[3] = { 0.0, 0.0, 0.0 };
    double e[2] = { 0.0, 0.0 };
    double w[2] = { 3.0, 4.0 };
    int iblock[2] = { 1, 1 };
    int isplit[2] = { 3, 0 };
    double z[6] = { 0.0 };
    int ifailv[2] = { 0, 0 };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dstein_fortran_call, 0, sizeof(g_dstein_fortran_call));

    vtable.ext_ops[FB_OP_DSTEIN][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_dstein_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_DSTEIN);

    thunk = (fb_dstein_fn)vtable.ext_ops[FB_OP_DSTEIN][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DSTEIN Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 3, d, e, 2, w, iblock, isplit, z, 3,
                 ifailv);
    if (info != 0 || g_dstein_fortran_call.exec_calls != 1 ||
        !g_dstein_fortran_call.saw_work || !g_dstein_fortran_call.saw_iwork ||
        z[0] != 27.0 || z[1] != 28.0 || ifailv[0] != 2) {
        fprintf(stderr, "[FAIL] DSTEIN Fortran->CBLAS thunk did not allocate fixed workspaces and forward double-precision tridiagonal inverse-iteration outputs\n");
        return 1;
    }

    printf("[PASS] DSTEIN Fortran->CBLAS thunk allocates fixed workspaces and forwards double-precision tridiagonal inverse-iteration outputs\n");
    return 0;
}

static int check_dstein_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_dstein_fortran_slot_fn thunk = NULL;
    double d[3] = { 0.0, 0.0, 0.0 };
    double e[2] = { 0.0, 0.0 };
    double w[2] = { 3.0, 4.0 };
    int iblock[2] = { 1, 1 };
    int isplit[2] = { 3, 0 };
    double z[6] = { 0.0 };
    double work[16] = { 0.0 };
    int iwork[16] = { 0 };
    int ifailv[2] = { 0, 0 };
    int n = 3;
    int m = 2;
    int ldz = 3;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dstein_cblas_call, 0, sizeof(g_dstein_cblas_call));
    g_dstein_cblas_rc = 267;

    vtable.ext_ops[FB_OP_DSTEIN][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_dstein_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_DSTEIN);

    thunk = (fb_dstein_fortran_slot_fn)vtable.ext_ops[FB_OP_DSTEIN][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DSTEIN CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&n, d, e, &m, w, iblock, isplit, z, &ldz, work, iwork, ifailv,
          &info);
    if (info != 267 || g_dstein_cblas_call.called != 1 ||
        g_dstein_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_dstein_cblas_call.n != 3 || g_dstein_cblas_call.m != 2 ||
        g_dstein_cblas_call.ldz != 3 || z[0] != 33.0 || z[1] != 34.0 ||
        ifailv[0] != 4) {
        fprintf(stderr, "[FAIL] DSTEIN CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] DSTEIN CBLAS->Fortran thunk maps double-precision tridiagonal inverse-iteration arguments into the C entry\n");
    return 0;
}

int main(void)
{
    if (check_sstein_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_sstein_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_dstein_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_dstein_cblas_to_fortran() != 0) {
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}