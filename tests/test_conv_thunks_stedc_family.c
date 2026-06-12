#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef int (*fb_sstedc_fn)(fb_layout_t layout, char compz, int n, float *d,
                            float *e, float *z, int ldz);
typedef int (*fb_dstedc_fn)(fb_layout_t layout, char compz, int n, double *d,
                            double *e, double *z, int ldz);

typedef void (*fb_sstedc_fortran_slot_fn)(char *compz, int *n, float *d,
                                          float *e, float *z, int *ldz,
                                          float *work, int *lwork, int *iwork,
                                          int *liwork, int *info);
typedef void (*fb_dstedc_fortran_slot_fn)(char *compz, int *n, double *d,
                                          double *e, double *z, int *ldz,
                                          double *work, int *lwork,
                                          int *iwork, int *liwork, int *info);

static struct {
    int query_calls;
    int exec_calls;
    int query_lwork;
    int exec_lwork;
    int exec_liwork;
    int exec_ldz;
    float z_in[6];
} g_sstedc_fortran_call;

static struct {
    int query_calls;
    int exec_calls;
    int query_lwork;
    int exec_lwork;
    int exec_liwork;
    int exec_ldz;
} g_dstedc_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    char compz;
    int n;
    int ldz;
} g_sstedc_cblas_call;

static struct {
    int called;
    fb_layout_t layout;
    char compz;
    int n;
    int ldz;
} g_dstedc_cblas_call;

static int g_sstedc_cblas_rc = 0;
static int g_dstedc_cblas_rc = 0;

static void stub_sstedc_fortran(char *compz, int *n, float *d, float *e,
                                float *z, int *ldz, float *work, int *lwork,
                                int *iwork, int *liwork, int *info)
{
    (void)n;
    if (*lwork == -1 || *liwork == -1) {
        g_sstedc_fortran_call.query_calls += 1;
        g_sstedc_fortran_call.query_lwork = *lwork;
        work[0] = 35.0f;
        *iwork = 21;
        *info = 0;
        return;
    }

    g_sstedc_fortran_call.exec_calls += 1;
    g_sstedc_fortran_call.exec_lwork = *lwork;
    g_sstedc_fortran_call.exec_liwork = *liwork;
    g_sstedc_fortran_call.exec_ldz = *ldz;
    memcpy(g_sstedc_fortran_call.z_in, z, sizeof(g_sstedc_fortran_call.z_in));
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
    (void)compz;
}

static void stub_dstedc_fortran(char *compz, int *n, double *d, double *e,
                                double *z, int *ldz, double *work, int *lwork,
                                int *iwork, int *liwork, int *info)
{
    (void)n;
    if (*lwork == -1 || *liwork == -1) {
        g_dstedc_fortran_call.query_calls += 1;
        g_dstedc_fortran_call.query_lwork = *lwork;
        work[0] = 36.0;
        *iwork = 23;
        *info = 0;
        return;
    }

    g_dstedc_fortran_call.exec_calls += 1;
    g_dstedc_fortran_call.exec_lwork = *lwork;
    g_dstedc_fortran_call.exec_liwork = *liwork;
    g_dstedc_fortran_call.exec_ldz = *ldz;
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
    (void)compz;
}

static int stub_sstedc_cblas(fb_layout_t layout, char compz, int n, float *d,
                             float *e, float *z, int ldz)
{
    g_sstedc_cblas_call.called += 1;
    g_sstedc_cblas_call.layout = layout;
    g_sstedc_cblas_call.compz = compz;
    g_sstedc_cblas_call.n = n;
    g_sstedc_cblas_call.ldz = ldz;
    d[0] = 5.5f;
    e[0] = 6.5f;
    z[0] = 701.0f;
    return g_sstedc_cblas_rc;
}

static int stub_dstedc_cblas(fb_layout_t layout, char compz, int n, double *d,
                             double *e, double *z, int ldz)
{
    g_dstedc_cblas_call.called += 1;
    g_dstedc_cblas_call.layout = layout;
    g_dstedc_cblas_call.compz = compz;
    g_dstedc_cblas_call.n = n;
    g_dstedc_cblas_call.ldz = ldz;
    d[0] = 7.5;
    e[0] = 8.5;
    z[0] = 801.0;
    return g_dstedc_cblas_rc;
}

static int check_sstedc_fortran_to_cblas(void)
{
    static const float expected_z[9] = {
        101.0f, 201.0f, 301.0f,
        102.0f, 202.0f, 302.0f,
        103.0f, 203.0f, 303.0f
    };
    fb_backend_vtable_t vtable;
    fb_sstedc_fn thunk = NULL;
    float d[3] = { 0.0f, 0.0f, 0.0f };
    float e[2] = { 0.0f, 0.0f };
    float z[9] = {
        11.0f, 12.0f, 13.0f,
        21.0f, 22.0f, 23.0f,
        31.0f, 32.0f, 33.0f
    };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sstedc_fortran_call, 0, sizeof(g_sstedc_fortran_call));

    vtable.ext_ops[FB_OP_SSTEDC][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_sstedc_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SSTEDC);

    thunk = (fb_sstedc_fn)vtable.ext_ops[FB_OP_SSTEDC][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SSTEDC Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 'V', 3, d, e, z, 3);
    if (info != 0 || g_sstedc_fortran_call.query_calls != 1 ||
        g_sstedc_fortran_call.exec_calls != 1 ||
        g_sstedc_fortran_call.query_lwork != -1 ||
        g_sstedc_fortran_call.exec_lwork != 35 ||
        g_sstedc_fortran_call.exec_liwork != 21 ||
        g_sstedc_fortran_call.exec_ldz != 3 ||
        g_sstedc_fortran_call.z_in[0] != 11.0f ||
        g_sstedc_fortran_call.z_in[1] != 21.0f ||
        g_sstedc_fortran_call.z_in[2] != 31.0f ||
        g_sstedc_fortran_call.z_in[3] != 12.0f ||
        g_sstedc_fortran_call.z_in[4] != 22.0f ||
        g_sstedc_fortran_call.z_in[5] != 32.0f ||
        d[0] != 1.5f || e[0] != 2.5f ||
        memcmp(z, expected_z, sizeof(expected_z)) != 0) {
        fprintf(stderr, "[FAIL] SSTEDC Fortran->CBLAS thunk did not preserve query semantics or row-major vectors\n");
        return 1;
    }

    printf("[PASS] SSTEDC Fortran->CBLAS thunk performs the workspace queries and round-trips row-major eigenvectors\n");
    return 0;
}

static int check_sstedc_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_sstedc_fortran_slot_fn thunk = NULL;
    float d[3] = { 0.0f, 0.0f, 0.0f };
    float e[2] = { 0.0f, 0.0f };
    float z[9] = { 0.0f };
    float work[8] = { 0.0f };
    int iwork[8] = { 0 };
    char compz = 'I';
    int n = 3;
    int ldz = 3;
    int lwork = 8;
    int liwork = 8;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sstedc_cblas_call, 0, sizeof(g_sstedc_cblas_call));
    g_sstedc_cblas_rc = 285;

    vtable.ext_ops[FB_OP_SSTEDC][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_sstedc_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SSTEDC);

    thunk = (fb_sstedc_fortran_slot_fn)vtable.ext_ops[FB_OP_SSTEDC][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SSTEDC CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&compz, &n, d, e, z, &ldz, work, &lwork, iwork, &liwork, &info);
    if (info != 285 || g_sstedc_cblas_call.called != 1 ||
        g_sstedc_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_sstedc_cblas_call.compz != 'I' || g_sstedc_cblas_call.n != 3 ||
        g_sstedc_cblas_call.ldz != 3 || d[0] != 5.5f || e[0] != 6.5f ||
        z[0] != 701.0f) {
        fprintf(stderr, "[FAIL] SSTEDC CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] SSTEDC CBLAS->Fortran thunk maps divide-and-conquer arguments into the C entry\n");
    return 0;
}

static int check_dstedc_fortran_to_cblas(void)
{
    static const double expected_z[9] = {
        401.0, 501.0, 601.0,
        402.0, 502.0, 602.0,
        403.0, 503.0, 603.0
    };
    fb_backend_vtable_t vtable;
    fb_dstedc_fn thunk = NULL;
    double d[3] = { 0.0, 0.0, 0.0 };
    double e[2] = { 0.0, 0.0 };
    double z[9] = { 0.0 };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dstedc_fortran_call, 0, sizeof(g_dstedc_fortran_call));

    vtable.ext_ops[FB_OP_DSTEDC][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_dstedc_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_DSTEDC);

    thunk = (fb_dstedc_fn)vtable.ext_ops[FB_OP_DSTEDC][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DSTEDC Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 'I', 3, d, e, z, 3);
    if (info != 0 || g_dstedc_fortran_call.query_calls != 1 ||
        g_dstedc_fortran_call.exec_calls != 1 ||
        g_dstedc_fortran_call.query_lwork != -1 ||
        g_dstedc_fortran_call.exec_lwork != 36 ||
        g_dstedc_fortran_call.exec_liwork != 23 ||
        g_dstedc_fortran_call.exec_ldz != 3 ||
        d[0] != 3.5 || e[0] != 4.5 ||
        memcmp(z, expected_z, sizeof(expected_z)) != 0) {
        fprintf(stderr, "[FAIL] DSTEDC Fortran->CBLAS thunk did not export generated row-major eigenvectors correctly\n");
        return 1;
    }

    printf("[PASS] DSTEDC Fortran->CBLAS thunk performs the workspace queries and exports generated row-major eigenvectors\n");
    return 0;
}

static int check_dstedc_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_dstedc_fortran_slot_fn thunk = NULL;
    double d[3] = { 0.0, 0.0, 0.0 };
    double e[2] = { 0.0, 0.0 };
    double z[9] = { 0.0 };
    double work[8] = { 0.0 };
    int iwork[8] = { 0 };
    char compz = 'V';
    int n = 3;
    int ldz = 3;
    int lwork = 8;
    int liwork = 8;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dstedc_cblas_call, 0, sizeof(g_dstedc_cblas_call));
    g_dstedc_cblas_rc = 287;

    vtable.ext_ops[FB_OP_DSTEDC][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_dstedc_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_DSTEDC);

    thunk = (fb_dstedc_fortran_slot_fn)vtable.ext_ops[FB_OP_DSTEDC][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DSTEDC CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&compz, &n, d, e, z, &ldz, work, &lwork, iwork, &liwork, &info);
    if (info != 287 || g_dstedc_cblas_call.called != 1 ||
        g_dstedc_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_dstedc_cblas_call.compz != 'V' || g_dstedc_cblas_call.n != 3 ||
        g_dstedc_cblas_call.ldz != 3 || d[0] != 7.5 || e[0] != 8.5 ||
        z[0] != 801.0) {
        fprintf(stderr, "[FAIL] DSTEDC CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] DSTEDC CBLAS->Fortran thunk maps double-precision divide-and-conquer arguments into the C entry\n");
    return 0;
}

int main(void)
{
    if (check_sstedc_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_sstedc_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_dstedc_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_dstedc_cblas_to_fortran() != 0) {
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}