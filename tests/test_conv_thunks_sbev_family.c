#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

#define DEFINE_SBEV_TESTS(SUFFIX, TYPE, OP_ID, BASE)                             \
typedef int (*fb_##SUFFIX##_cblas_fn)(fb_layout_t layout, char jobz,           \
                                      fb_uplo_t uplo, int n, int kd, TYPE *ab, \
                                      int ldab, TYPE *w, TYPE *z, int ldz);    \
typedef void (*fb_##SUFFIX##_fortran_fn)(char *jobz, char *uplo, int *n, int *kd,\
                                         TYPE *ab, int *ldab, TYPE *w, TYPE *z,\
                                         int *ldz, TYPE *work, int *info);     \
static struct {                                                                   \
    int called;                                                                   \
    char jobz;                                                                    \
    char uplo;                                                                    \
    int n;                                                                        \
    int kd;                                                                       \
    TYPE ab_snapshot[6];                                                          \
    int ldab;                                                                     \
    TYPE *w;                                                                      \
    TYPE *z;                                                                      \
    int ldz;                                                                      \
    int work_nonnull;                                                             \
} g_##SUFFIX##_fortran_call;                                                      \
static struct {                                                                   \
    int called;                                                                   \
    fb_layout_t layout;                                                           \
    char jobz;                                                                    \
    fb_uplo_t uplo;                                                               \
    int n;                                                                        \
    int kd;                                                                       \
    TYPE *ab;                                                                     \
    int ldab;                                                                     \
    TYPE *w;                                                                      \
    TYPE *z;                                                                      \
    int ldz;                                                                      \
} g_##SUFFIX##_cblas_call;                                                        \
static void stub_##SUFFIX##_fortran(char *jobz, char *uplo, int *n, int *kd,   \
                                    TYPE *ab, int *ldab, TYPE *w, TYPE *z,     \
                                    int *ldz, TYPE *work, int *info)           \
{                                                                                 \
    memcpy(g_##SUFFIX##_fortran_call.ab_snapshot, ab, sizeof(g_##SUFFIX##_fortran_call.ab_snapshot)); \
    g_##SUFFIX##_fortran_call.called += 1;                                        \
    g_##SUFFIX##_fortran_call.jobz = *jobz;                                       \
    g_##SUFFIX##_fortran_call.uplo = *uplo;                                       \
    g_##SUFFIX##_fortran_call.n = *n;                                             \
    g_##SUFFIX##_fortran_call.kd = *kd;                                           \
    g_##SUFFIX##_fortran_call.ldab = *ldab;                                       \
    g_##SUFFIX##_fortran_call.w = w;                                              \
    g_##SUFFIX##_fortran_call.z = z;                                              \
    g_##SUFFIX##_fortran_call.ldz = *ldz;                                         \
    g_##SUFFIX##_fortran_call.work_nonnull = (work != NULL);                      \
    w[0] = (TYPE)((BASE) + 1); w[1] = (TYPE)((BASE) + 2); w[2] = (TYPE)((BASE) + 3); \
    z[0] = (TYPE)101; z[1] = (TYPE)201; z[2] = (TYPE)301;                        \
    z[3] = (TYPE)102; z[4] = (TYPE)202; z[5] = (TYPE)302;                        \
    z[6] = (TYPE)103; z[7] = (TYPE)203; z[8] = (TYPE)303;                        \
    *info = 0;                                                                    \
}                                                                                 \
static int stub_##SUFFIX##_cblas(fb_layout_t layout, char jobz, fb_uplo_t uplo, \
                                 int n, int kd, TYPE *ab, int ldab, TYPE *w,   \
                                 TYPE *z, int ldz)                              \
{                                                                                 \
    g_##SUFFIX##_cblas_call.called += 1;                                          \
    g_##SUFFIX##_cblas_call.layout = layout;                                      \
    g_##SUFFIX##_cblas_call.jobz = jobz;                                          \
    g_##SUFFIX##_cblas_call.uplo = uplo;                                          \
    g_##SUFFIX##_cblas_call.n = n;                                                \
    g_##SUFFIX##_cblas_call.kd = kd;                                              \
    g_##SUFFIX##_cblas_call.ab = ab;                                              \
    g_##SUFFIX##_cblas_call.ldab = ldab;                                          \
    g_##SUFFIX##_cblas_call.w = w;                                                \
    g_##SUFFIX##_cblas_call.z = z;                                                \
    g_##SUFFIX##_cblas_call.ldz = ldz;                                            \
    w[0] = (TYPE)((BASE) + 4);                                                    \
    z[0] = (TYPE)((BASE) + 5);                                                    \
    return (BASE) + 6;                                                            \
}                                                                                 \
static int check_##SUFFIX##_fortran_to_cblas(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                          \
    TYPE ab[6] = { (TYPE)0, (TYPE)12, (TYPE)23, (TYPE)11, (TYPE)22, (TYPE)33 }; \
    TYPE expected_ab[6] = { (TYPE)0, (TYPE)11, (TYPE)12, (TYPE)22, (TYPE)23, (TYPE)33 }; \
    TYPE w[3] = { 0, 0, 0 };                                                      \
    TYPE z[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };                                   \
    TYPE expected_z[9] = { (TYPE)101, (TYPE)102, (TYPE)103, (TYPE)201, (TYPE)202, (TYPE)203, (TYPE)301, (TYPE)302, (TYPE)303 }; \
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
    info = thunk(FB_LAYOUT_ROW_MAJOR, 'V', FB_UPPER, 3, 1, ab, 3, w, z, 3);      \
    if (info != 0 || g_##SUFFIX##_fortran_call.called != 1 ||                     \
        g_##SUFFIX##_fortran_call.jobz != 'V' ||                                   \
        g_##SUFFIX##_fortran_call.uplo != 'U' ||                                   \
        g_##SUFFIX##_fortran_call.n != 3 || g_##SUFFIX##_fortran_call.kd != 1 ||  \
        memcmp(g_##SUFFIX##_fortran_call.ab_snapshot, expected_ab, sizeof(expected_ab)) != 0 || \
        g_##SUFFIX##_fortran_call.ldab != 2 ||                                     \
        g_##SUFFIX##_fortran_call.w != w ||                                        \
        g_##SUFFIX##_fortran_call.ldz != 3 ||                                      \
        !g_##SUFFIX##_fortran_call.work_nonnull ||                                 \
        w[0] != (TYPE)((BASE) + 1) || w[1] != (TYPE)((BASE) + 2) ||               \
        w[2] != (TYPE)((BASE) + 3) || memcmp(z, expected_z, sizeof(expected_z)) != 0) { \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not translate SBEV band storage or copy back vectors correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk translates band storage and copies back eigenvectors\n"); \
    return 0;                                                                     \
}                                                                                 \
static int check_##SUFFIX##_cblas_to_fortran(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                        \
    char jobz = 'N';                                                              \
    char uplo = 'L';                                                              \
    int n = 2;                                                                    \
    int kd = 1;                                                                   \
    TYPE ab[4] = { (TYPE)4, (TYPE)5, (TYPE)6, (TYPE)7 };                         \
    int ldab = 2;                                                                 \
    TYPE w[2] = { 0, 0 };                                                         \
    TYPE z[1] = { 0 };                                                            \
    int ldz = 1;                                                                  \
    TYPE work[4] = { 0, 0, 0, 0 };                                                \
    int info = -999;                                                              \
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
    thunk(&jobz, &uplo, &n, &kd, ab, &ldab, w, z, &ldz, work, &info);            \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                    \
        g_##SUFFIX##_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||                  \
        g_##SUFFIX##_cblas_call.jobz != 'N' ||                                    \
        g_##SUFFIX##_cblas_call.uplo != FB_LOWER ||                               \
        g_##SUFFIX##_cblas_call.n != 2 || g_##SUFFIX##_cblas_call.kd != 1 ||      \
        g_##SUFFIX##_cblas_call.ab != ab || g_##SUFFIX##_cblas_call.ldab != 2 ||  \
        g_##SUFFIX##_cblas_call.w != w || g_##SUFFIX##_cblas_call.z != z ||       \
        g_##SUFFIX##_cblas_call.ldz != 1 || info != (BASE) + 6 ||                 \
        w[0] != (TYPE)((BASE) + 4) || z[0] != (TYPE)((BASE) + 5)) {               \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not forward SBEV arguments correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk forwards SBEV arguments\n"); \
    return 0;                                                                     \
}

DEFINE_SBEV_TESTS(ssbev, float, FB_OP_SSBEV, 100)
DEFINE_SBEV_TESTS(dsbev, double, FB_OP_DSBEV, 300)

int main(void)
{
    int status = 0;

    status |= check_ssbev_fortran_to_cblas();
    status |= check_ssbev_cblas_to_fortran();
    status |= check_dsbev_fortran_to_cblas();
    status |= check_dsbev_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}