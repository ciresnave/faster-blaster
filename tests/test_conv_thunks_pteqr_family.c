#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef int (*fb_spteqr_fn)(fb_layout_t layout, char compz, int n, float *d,
                            float *e, float *z, int ldz);

typedef void (*fb_spteqr_fortran_slot_fn)(char *compz, int *n, float *d,
                                          float *e, float *z, int *ldz,
                                          float *work, int *info);

static struct {
    int exec_calls;
    char compz;
    int n;
    int ldz;
    int saw_work;
    float z_in[6];
} g_spteqr_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    char compz;
    int n;
    int ldz;
} g_spteqr_cblas_call;

static int g_spteqr_cblas_rc = 0;

static void stub_spteqr_fortran(char *compz, int *n, float *d, float *e,
                                float *z, int *ldz, float *work, int *info)
{
    g_spteqr_fortran_call.exec_calls += 1;
    g_spteqr_fortran_call.compz = *compz;
    g_spteqr_fortran_call.n = *n;
    g_spteqr_fortran_call.ldz = *ldz;
    g_spteqr_fortran_call.saw_work = (work != NULL);
    memcpy(g_spteqr_fortran_call.z_in, z, sizeof(g_spteqr_fortran_call.z_in));
    d[0] = 11.5f;
    e[0] = 12.5f;
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

static int stub_spteqr_cblas(fb_layout_t layout, char compz, int n, float *d,
                             float *e, float *z, int ldz)
{
    g_spteqr_cblas_call.called += 1;
    g_spteqr_cblas_call.layout = layout;
    g_spteqr_cblas_call.compz = compz;
    g_spteqr_cblas_call.n = n;
    g_spteqr_cblas_call.ldz = ldz;
    d[0] = 21.5f;
    e[0] = 22.5f;
    z[0] = 401.0f;
    return g_spteqr_cblas_rc;
}

static int check_spteqr_fortran_to_cblas_vectors(void)
{
    static const float expected_z[9] = {
        101.0f, 201.0f, 301.0f,
        102.0f, 202.0f, 302.0f,
        103.0f, 203.0f, 303.0f
    };
    fb_backend_vtable_t vtable;
    fb_spteqr_fn thunk = NULL;
    float d[3] = { 0.0f, 0.0f, 0.0f };
    float e[2] = { 0.0f, 0.0f };
    float z[9] = {
        11.0f, 12.0f, 13.0f,
        21.0f, 22.0f, 23.0f,
        31.0f, 32.0f, 33.0f
    };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_spteqr_fortran_call, 0, sizeof(g_spteqr_fortran_call));

    vtable.ext_ops[FB_OP_SPTEQR][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_spteqr_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SPTEQR);

    thunk = (fb_spteqr_fn)vtable.ext_ops[FB_OP_SPTEQR][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SPTEQR Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 'V', 3, d, e, z, 3);
    if (info != 0 || g_spteqr_fortran_call.exec_calls != 1 ||
        g_spteqr_fortran_call.compz != 'V' || g_spteqr_fortran_call.n != 3 ||
        g_spteqr_fortran_call.ldz != 3 || !g_spteqr_fortran_call.saw_work ||
        g_spteqr_fortran_call.z_in[0] != 11.0f ||
        g_spteqr_fortran_call.z_in[1] != 21.0f ||
        g_spteqr_fortran_call.z_in[2] != 31.0f ||
        g_spteqr_fortran_call.z_in[3] != 12.0f ||
        g_spteqr_fortran_call.z_in[4] != 22.0f ||
        g_spteqr_fortran_call.z_in[5] != 32.0f ||
        d[0] != 11.5f || e[0] != 12.5f ||
        memcmp(z, expected_z, sizeof(expected_z)) != 0) {
        fprintf(stderr, "[FAIL] SPTEQR Fortran->CBLAS thunk did not transpose row-major vector output correctly\n");
        return 1;
    }

    printf("[PASS] SPTEQR Fortran->CBLAS thunk allocates fixed workspace and round-trips row-major vectors\n");
    return 0;
}

static int check_spteqr_fortran_to_cblas_identity_vectors(void)
{
    fb_backend_vtable_t vtable;
    fb_spteqr_fn thunk = NULL;
    float d[3] = { 0.0f, 0.0f, 0.0f };
    float e[2] = { 0.0f, 0.0f };
    float z[9] = { 0.0f };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_spteqr_fortran_call, 0, sizeof(g_spteqr_fortran_call));

    vtable.ext_ops[FB_OP_SPTEQR][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_spteqr_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SPTEQR);

    thunk = (fb_spteqr_fn)vtable.ext_ops[FB_OP_SPTEQR][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SPTEQR identity-vector thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 'I', 3, d, e, z, 3);
    if (info != 0 || g_spteqr_fortran_call.exec_calls != 1 ||
        g_spteqr_fortran_call.compz != 'I' || g_spteqr_fortran_call.n != 3 ||
        g_spteqr_fortran_call.ldz != 3 || !g_spteqr_fortran_call.saw_work) {
        fprintf(stderr, "[FAIL] SPTEQR Fortran->CBLAS thunk did not handle compz='I' correctly\n");
        return 1;
    }

    printf("[PASS] SPTEQR Fortran->CBLAS thunk exports generated vectors for compz='I'\n");
    return 0;
}

static int check_spteqr_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_spteqr_fortran_slot_fn thunk = NULL;
    float d[3] = { 0.0f, 0.0f, 0.0f };
    float e[2] = { 0.0f, 0.0f };
    float z[9] = { 0.0f };
    float work[8] = { 0.0f };
    char compz = 'V';
    int n = 3;
    int ldz = 3;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_spteqr_cblas_call, 0, sizeof(g_spteqr_cblas_call));
    g_spteqr_cblas_rc = 523;

    vtable.ext_ops[FB_OP_SPTEQR][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_spteqr_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SPTEQR);

    thunk = (fb_spteqr_fortran_slot_fn)vtable.ext_ops[FB_OP_SPTEQR][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SPTEQR CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&compz, &n, d, e, z, &ldz, work, &info);
    if (info != 523 || g_spteqr_cblas_call.called != 1 ||
        g_spteqr_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_spteqr_cblas_call.compz != 'V' || g_spteqr_cblas_call.n != 3 ||
        g_spteqr_cblas_call.ldz != 3 || d[0] != 21.5f ||
        e[0] != 22.5f || z[0] != 401.0f) {
        fprintf(stderr, "[FAIL] SPTEQR CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] SPTEQR CBLAS->Fortran thunk maps QR-iteration arguments into the C entry\n");
    return 0;
}

int main(void)
{
    int status = 0;

    status |= check_spteqr_fortran_to_cblas_vectors();
    status |= check_spteqr_fortran_to_cblas_identity_vectors();
    status |= check_spteqr_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}