#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

static fb_complex_float_t make_cfloat(float real_value, float imag_value)
{
    fb_complex_float_t value = (fb_complex_float_t)0;
    __real__ value = real_value;
    __imag__ value = imag_value;
    return value;
}

static fb_complex_double_t make_cdouble(double real_value, double imag_value)
{
    fb_complex_double_t value = (fb_complex_double_t)0;
    __real__ value = real_value;
    __imag__ value = imag_value;
    return value;
}

#define DEFINE_HPEV_COMPLEX_TESTS(SUFFIX, CTYPE, RTYPE, OP_ID, BASE, MAKE)      \
typedef int (*fb_##SUFFIX##_cblas_fn)(fb_layout_t layout, char jobz,           \
                                      fb_uplo_t uplo, int n, CTYPE *ap,        \
                                      RTYPE *w, CTYPE *z, int ldz);            \
typedef void (*fb_##SUFFIX##_fortran_fn)(char *jobz, char *uplo, int *n,       \
                                         CTYPE *ap, RTYPE *w, CTYPE *z,        \
                                         int *ldz, CTYPE *work, RTYPE *rwork,  \
                                         int *info);                           \
static struct {                                                                   \
    int called;                                                                   \
    char jobz;                                                                    \
    char uplo;                                                                    \
    int n;                                                                        \
    CTYPE ap_snapshot[6];                                                         \
    RTYPE *w;                                                                     \
    CTYPE *z;                                                                     \
    int ldz;                                                                      \
    int work_nonnull;                                                             \
    int rwork_nonnull;                                                            \
} g_##SUFFIX##_fortran_call;                                                      \
static struct {                                                                   \
    int called;                                                                   \
    fb_layout_t layout;                                                           \
    char jobz;                                                                    \
    fb_uplo_t uplo;                                                               \
    int n;                                                                        \
    CTYPE *ap;                                                                    \
    RTYPE *w;                                                                     \
    CTYPE *z;                                                                     \
    int ldz;                                                                      \
} g_##SUFFIX##_cblas_call;                                                        \
static void stub_##SUFFIX##_fortran(char *jobz, char *uplo, int *n, CTYPE *ap, \
                                    RTYPE *w, CTYPE *z, int *ldz, CTYPE *work, \
                                    RTYPE *rwork, int *info)                   \
{                                                                                 \
    size_t elems = (size_t)(*n * (*n + 1)) / 2U;                                 \
    memset(g_##SUFFIX##_fortran_call.ap_snapshot, 0, sizeof(g_##SUFFIX##_fortran_call.ap_snapshot)); \
    if (elems > 0) {                                                              \
        memcpy(g_##SUFFIX##_fortran_call.ap_snapshot, ap, elems * sizeof(CTYPE)); \
    }                                                                             \
    g_##SUFFIX##_fortran_call.called += 1;                                        \
    g_##SUFFIX##_fortran_call.jobz = *jobz;                                       \
    g_##SUFFIX##_fortran_call.uplo = *uplo;                                       \
    g_##SUFFIX##_fortran_call.n = *n;                                             \
    g_##SUFFIX##_fortran_call.w = w;                                              \
    g_##SUFFIX##_fortran_call.z = z;                                              \
    g_##SUFFIX##_fortran_call.ldz = *ldz;                                         \
    g_##SUFFIX##_fortran_call.work_nonnull = (work != NULL);                      \
    g_##SUFFIX##_fortran_call.rwork_nonnull = (rwork != NULL);                    \
    ap[0] = MAKE(41, 11); ap[1] = MAKE(42, 12); ap[2] = MAKE(52, 14);            \
    ap[3] = MAKE(43, 13); ap[4] = MAKE(53, 15); ap[5] = MAKE(63, 16);            \
    w[0] = (RTYPE)((BASE) + 1); w[1] = (RTYPE)((BASE) + 2); w[2] = (RTYPE)((BASE) + 3); \
    z[0] = MAKE(101, 1); z[1] = MAKE(201, 4); z[2] = MAKE(301, 7);               \
    z[3] = MAKE(102, 2); z[4] = MAKE(202, 5); z[5] = MAKE(302, 8);               \
    z[6] = MAKE(103, 3); z[7] = MAKE(203, 6); z[8] = MAKE(303, 9);               \
    *info = 0;                                                                    \
}                                                                                 \
static int stub_##SUFFIX##_cblas(fb_layout_t layout, char jobz, fb_uplo_t uplo, \
                                 int n, CTYPE *ap, RTYPE *w, CTYPE *z, int ldz) \
{                                                                                 \
    g_##SUFFIX##_cblas_call.called += 1;                                          \
    g_##SUFFIX##_cblas_call.layout = layout;                                      \
    g_##SUFFIX##_cblas_call.jobz = jobz;                                          \
    g_##SUFFIX##_cblas_call.uplo = uplo;                                          \
    g_##SUFFIX##_cblas_call.n = n;                                                \
    g_##SUFFIX##_cblas_call.ap = ap;                                              \
    g_##SUFFIX##_cblas_call.w = w;                                                \
    g_##SUFFIX##_cblas_call.z = z;                                                \
    g_##SUFFIX##_cblas_call.ldz = ldz;                                            \
    w[0] = (RTYPE)((BASE) + 4);                                                   \
    z[0] = MAKE((BASE) + 5, (BASE) + 15);                                        \
    return (BASE) + 6;                                                            \
}                                                                                 \
static int check_##SUFFIX##_fortran_to_cblas(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                          \
    CTYPE ap_row[6] = { MAKE(11, 1), MAKE(12, 2), MAKE(13, 3),                   \
                        MAKE(22, 4), MAKE(23, 5), MAKE(33, 6) };                 \
    CTYPE expected_ap_in[6] = { MAKE(11, 1), MAKE(12, 2), MAKE(22, 4),           \
                                MAKE(13, 3), MAKE(23, 5), MAKE(33, 6) };         \
    CTYPE expected_ap_out[6] = { MAKE(41, 11), MAKE(42, 12), MAKE(43, 13),       \
                                 MAKE(52, 14), MAKE(53, 15), MAKE(63, 16) };     \
    RTYPE w[3] = { 0, 0, 0 };                                                     \
    CTYPE z[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };                                  \
    CTYPE expected_z[9] = { MAKE(101, 1), MAKE(102, 2), MAKE(103, 3),            \
                            MAKE(201, 4), MAKE(202, 5), MAKE(203, 6),            \
                            MAKE(301, 7), MAKE(302, 8), MAKE(303, 9) };          \
    int info = 0;                                                                 \
    memset(&vtable, 0, sizeof(vtable));                                           \
    memset(&g_##SUFFIX##_fortran_call, 0, sizeof(g_##SUFFIX##_fortran_call));     \
    vtable.ext_ops[OP_ID][FB_CONV_FORTRAN] =                                      \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_fortran;                   \
    fb_install_conv_thunks(&vtable, OP_ID);                                       \
    thunk = (fb_##SUFFIX##_cblas_fn)vtable.ext_ops[OP_ID][FB_CONV_CBLAS];         \
    if (!thunk) {                                                                 \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk was not installed\n"); \
        return 1;                                                                 \
    }                                                                             \
    info = thunk(FB_LAYOUT_ROW_MAJOR, 'V', FB_UPPER, 3, ap_row, w, z, 3);        \
    if (info != 0 || g_##SUFFIX##_fortran_call.called != 1 ||                    \
        g_##SUFFIX##_fortran_call.jobz != 'V' ||                                  \
        g_##SUFFIX##_fortran_call.uplo != 'U' ||                                  \
        g_##SUFFIX##_fortran_call.n != 3 ||                                       \
        memcmp(g_##SUFFIX##_fortran_call.ap_snapshot, expected_ap_in, sizeof(expected_ap_in)) != 0 || \
        g_##SUFFIX##_fortran_call.w != w ||                                       \
        g_##SUFFIX##_fortran_call.ldz != 3 ||                                     \
        !g_##SUFFIX##_fortran_call.work_nonnull ||                                \
        !g_##SUFFIX##_fortran_call.rwork_nonnull ||                               \
        memcmp(ap_row, expected_ap_out, sizeof(expected_ap_out)) != 0 ||         \
        w[0] != (RTYPE)((BASE) + 1) || w[1] != (RTYPE)((BASE) + 2) ||            \
        w[2] != (RTYPE)((BASE) + 3) ||                                           \
        memcmp(z, expected_z, sizeof(expected_z)) != 0) {                        \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not translate packed HPEV state correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk translates packed Hermitian storage and copies back eigenvectors\n"); \
    return 0;                                                                     \
}                                                                                 \
static int check_##SUFFIX##_cblas_to_fortran(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                        \
    char jobz = 'N';                                                              \
    char uplo = 'L';                                                              \
    int n = 2;                                                                    \
    CTYPE ap[3] = { MAKE(7, 1), MAKE(8, 2), MAKE(9, 3) };                        \
    RTYPE w[2] = { 0, 0 };                                                        \
    CTYPE z[1] = { MAKE(0, 0) };                                                  \
    int ldz = 1;                                                                  \
    CTYPE work[3];                                                                \
    RTYPE rwork[4] = { 0, 0, 0, 0 };                                              \
    int info = -999;                                                              \
    memset(work, 0, sizeof(work));                                                \
    memset(&vtable, 0, sizeof(vtable));                                           \
    memset(&g_##SUFFIX##_cblas_call, 0, sizeof(g_##SUFFIX##_cblas_call));         \
    vtable.ext_ops[OP_ID][FB_CONV_CBLAS] =                                        \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_cblas;                     \
    fb_install_conv_thunks(&vtable, OP_ID);                                       \
    thunk = (fb_##SUFFIX##_fortran_fn)vtable.ext_ops[OP_ID][FB_CONV_FORTRAN];     \
    if (!thunk) {                                                                 \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk was not installed\n"); \
        return 1;                                                                 \
    }                                                                             \
    thunk(&jobz, &uplo, &n, ap, w, z, &ldz, work, rwork, &info);                 \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                    \
        g_##SUFFIX##_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||                  \
        g_##SUFFIX##_cblas_call.jobz != 'N' ||                                    \
        g_##SUFFIX##_cblas_call.uplo != FB_LOWER ||                               \
        g_##SUFFIX##_cblas_call.n != 2 ||                                         \
        g_##SUFFIX##_cblas_call.ap != ap ||                                       \
        g_##SUFFIX##_cblas_call.w != w ||                                         \
        g_##SUFFIX##_cblas_call.z != z ||                                         \
        g_##SUFFIX##_cblas_call.ldz != 1 ||                                       \
        info != (BASE) + 6 ||                                                     \
        w[0] != (RTYPE)((BASE) + 4) ||                                            \
        z[0] != MAKE((BASE) + 5, (BASE) + 15)) {                                 \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not forward HPEV arguments correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk forwards HPEV arguments\n"); \
    return 0;                                                                     \
}

DEFINE_HPEV_COMPLEX_TESTS(chpev, fb_complex_float_t, float, FB_OP_CHPEV, 100, make_cfloat)
DEFINE_HPEV_COMPLEX_TESTS(zhpev, fb_complex_double_t, double, FB_OP_ZHPEV, 300, make_cdouble)

int main(void)
{
    int status = 0;

    status |= check_chpev_fortran_to_cblas();
    status |= check_chpev_cblas_to_fortran();
    status |= check_zhpev_fortran_to_cblas();
    status |= check_zhpev_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}