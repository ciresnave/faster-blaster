#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

#define DEFINE_SBGV_TESTS(SUFFIX, TYPE, OP_ID, BASE)                             \
typedef int (*fb_##SUFFIX##_cblas_fn)(fb_layout_t layout, char jobz,          \
                                      fb_uplo_t uplo, int n, int ka, int kb,  \
                                      TYPE *ab, int ldab, TYPE *bb, int ldbb, \
                                      TYPE *w, TYPE *z, int ldz);             \
typedef void (*fb_##SUFFIX##_fortran_fn)(char *jobz, char *uplo, int *n, int *ka,\
                                         int *kb, TYPE *ab, int *ldab, TYPE *bb,\
                                         int *ldbb, TYPE *w, TYPE *z, int *ldz,\
                                         TYPE *work, int *info);              \
static struct {                                                                   \
    int called;                                                                   \
    char jobz;                                                                    \
    char uplo;                                                                    \
    int n;                                                                        \
    int ka;                                                                       \
    int kb;                                                                       \
    TYPE ab_snapshot[6];                                                          \
    TYPE bb_snapshot[6];                                                          \
    int ldab;                                                                     \
    int ldbb;                                                                     \
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
    int ka;                                                                       \
    int kb;                                                                       \
    TYPE *ab;                                                                     \
    int ldab;                                                                     \
    TYPE *bb;                                                                     \
    int ldbb;                                                                     \
    TYPE *w;                                                                      \
    TYPE *z;                                                                      \
    int ldz;                                                                      \
} g_##SUFFIX##_cblas_call;                                                        \
static void stub_##SUFFIX##_fortran(char *jobz, char *uplo, int *n, int *ka,   \
                                    int *kb, TYPE *ab, int *ldab, TYPE *bb,    \
                                    int *ldbb, TYPE *w, TYPE *z, int *ldz,     \
                                    TYPE *work, int *info)                     \
{                                                                                 \
    memcpy(g_##SUFFIX##_fortran_call.ab_snapshot, ab, sizeof(g_##SUFFIX##_fortran_call.ab_snapshot)); \
    memcpy(g_##SUFFIX##_fortran_call.bb_snapshot, bb, sizeof(g_##SUFFIX##_fortran_call.bb_snapshot)); \
    g_##SUFFIX##_fortran_call.called += 1;                                        \
    g_##SUFFIX##_fortran_call.jobz = *jobz;                                       \
    g_##SUFFIX##_fortran_call.uplo = *uplo;                                       \
    g_##SUFFIX##_fortran_call.n = *n;                                             \
    g_##SUFFIX##_fortran_call.ka = *ka;                                           \
    g_##SUFFIX##_fortran_call.kb = *kb;                                           \
    g_##SUFFIX##_fortran_call.ldab = *ldab;                                       \
    g_##SUFFIX##_fortran_call.ldbb = *ldbb;                                       \
    g_##SUFFIX##_fortran_call.w = w;                                              \
    g_##SUFFIX##_fortran_call.z = z;                                              \
    g_##SUFFIX##_fortran_call.ldz = *ldz;                                         \
    g_##SUFFIX##_fortran_call.work_nonnull = (work != NULL);                      \
    ab[0] = (TYPE)0; ab[1] = (TYPE)41; ab[2] = (TYPE)42;                         \
    ab[3] = (TYPE)52; ab[4] = (TYPE)53; ab[5] = (TYPE)63;                        \
    bb[0] = (TYPE)0; bb[1] = (TYPE)71; bb[2] = (TYPE)72;                         \
    bb[3] = (TYPE)82; bb[4] = (TYPE)83; bb[5] = (TYPE)93;                        \
    w[0] = (TYPE)((BASE) + 1); w[1] = (TYPE)((BASE) + 2); w[2] = (TYPE)((BASE) + 3); \
    z[0] = (TYPE)101; z[1] = (TYPE)201; z[2] = (TYPE)301;                        \
    z[3] = (TYPE)102; z[4] = (TYPE)202; z[5] = (TYPE)302;                        \
    z[6] = (TYPE)103; z[7] = (TYPE)203; z[8] = (TYPE)303;                        \
    *info = 0;                                                                    \
}                                                                                 \
static int stub_##SUFFIX##_cblas(fb_layout_t layout, char jobz, fb_uplo_t uplo,\
                                 int n, int ka, int kb, TYPE *ab, int ldab,    \
                                 TYPE *bb, int ldbb, TYPE *w, TYPE *z, int ldz)\
{                                                                                 \
    g_##SUFFIX##_cblas_call.called += 1;                                          \
    g_##SUFFIX##_cblas_call.layout = layout;                                      \
    g_##SUFFIX##_cblas_call.jobz = jobz;                                          \
    g_##SUFFIX##_cblas_call.uplo = uplo;                                          \
    g_##SUFFIX##_cblas_call.n = n;                                                \
    g_##SUFFIX##_cblas_call.ka = ka;                                              \
    g_##SUFFIX##_cblas_call.kb = kb;                                              \
    g_##SUFFIX##_cblas_call.ab = ab;                                              \
    g_##SUFFIX##_cblas_call.ldab = ldab;                                          \
    g_##SUFFIX##_cblas_call.bb = bb;                                              \
    g_##SUFFIX##_cblas_call.ldbb = ldbb;                                          \
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
    TYPE bb[6] = { (TYPE)0, (TYPE)45, (TYPE)56, (TYPE)44, (TYPE)55, (TYPE)66 }; \
    TYPE expected_ab_in[6] = { (TYPE)0, (TYPE)11, (TYPE)12, (TYPE)22, (TYPE)23, (TYPE)33 }; \
    TYPE expected_bb_in[6] = { (TYPE)0, (TYPE)44, (TYPE)45, (TYPE)55, (TYPE)56, (TYPE)66 }; \
    TYPE expected_ab_out[6] = { (TYPE)0, (TYPE)42, (TYPE)53, (TYPE)41, (TYPE)52, (TYPE)63 }; \
    TYPE expected_bb_out[6] = { (TYPE)0, (TYPE)72, (TYPE)83, (TYPE)71, (TYPE)82, (TYPE)93 }; \
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
    info = thunk(FB_LAYOUT_ROW_MAJOR, 'V', FB_UPPER, 3, 1, 1, ab, 3, bb, 3, w, z, 3); \
    if (info != 0 || g_##SUFFIX##_fortran_call.called != 1 ||                     \
        g_##SUFFIX##_fortran_call.jobz != 'V' ||                                   \
        g_##SUFFIX##_fortran_call.uplo != 'U' ||                                   \
        g_##SUFFIX##_fortran_call.n != 3 || g_##SUFFIX##_fortran_call.ka != 1 ||  \
        g_##SUFFIX##_fortran_call.kb != 1 ||                                       \
        memcmp(g_##SUFFIX##_fortran_call.ab_snapshot, expected_ab_in, sizeof(expected_ab_in)) != 0 || \
        memcmp(g_##SUFFIX##_fortran_call.bb_snapshot, expected_bb_in, sizeof(expected_bb_in)) != 0 || \
        g_##SUFFIX##_fortran_call.ldab != 2 || g_##SUFFIX##_fortran_call.ldbb != 2 || \
        g_##SUFFIX##_fortran_call.w != w || g_##SUFFIX##_fortran_call.ldz != 3 ||  \
        !g_##SUFFIX##_fortran_call.work_nonnull ||                                 \
        memcmp(ab, expected_ab_out, sizeof(expected_ab_out)) != 0 ||              \
        memcmp(bb, expected_bb_out, sizeof(expected_bb_out)) != 0 ||              \
        w[0] != (TYPE)((BASE) + 1) || w[1] != (TYPE)((BASE) + 2) ||               \
        w[2] != (TYPE)((BASE) + 3) || memcmp(z, expected_z, sizeof(expected_z)) != 0) { \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not translate generalized band storage or copy back outputs correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk translates A/B band storage and copies back eigenvectors\n"); \
    return 0;                                                                     \
}                                                                                 \
static int check_##SUFFIX##_cblas_to_fortran(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                        \
    char jobz = 'V';                                                              \
    char uplo = 'L';                                                              \
    int n = 2;                                                                    \
    int ka = 1;                                                                   \
    int kb = 1;                                                                   \
    TYPE ab[4] = { (TYPE)4, (TYPE)5, (TYPE)6, (TYPE)7 };                         \
    TYPE bb[4] = { (TYPE)8, (TYPE)9, (TYPE)10, (TYPE)11 };                       \
    int ldab = 2;                                                                 \
    int ldbb = 2;                                                                 \
    TYPE w[2] = { 0, 0 };                                                         \
    TYPE z[4] = { 0, 0, 0, 0 };                                                   \
    int ldz = 2;                                                                  \
    TYPE work[6] = { 0, 0, 0, 0, 0, 0 };                                          \
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
    thunk(&jobz, &uplo, &n, &ka, &kb, ab, &ldab, bb, &ldbb, w, z, &ldz, work, &info); \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                    \
        g_##SUFFIX##_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||                  \
        g_##SUFFIX##_cblas_call.jobz != 'V' ||                                    \
        g_##SUFFIX##_cblas_call.uplo != FB_LOWER ||                               \
        g_##SUFFIX##_cblas_call.n != 2 || g_##SUFFIX##_cblas_call.ka != 1 ||      \
        g_##SUFFIX##_cblas_call.kb != 1 ||                                        \
        g_##SUFFIX##_cblas_call.ab != ab || g_##SUFFIX##_cblas_call.ldab != 2 ||  \
        g_##SUFFIX##_cblas_call.bb != bb || g_##SUFFIX##_cblas_call.ldbb != 2 ||  \
        g_##SUFFIX##_cblas_call.w != w || g_##SUFFIX##_cblas_call.z != z ||       \
        g_##SUFFIX##_cblas_call.ldz != 2 || info != (BASE) + 6 ||                 \
        w[0] != (TYPE)((BASE) + 4) || z[0] != (TYPE)((BASE) + 5)) {               \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not forward SBGV arguments correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk forwards SBGV arguments\n"); \
    return 0;                                                                     \
}

DEFINE_SBGV_TESTS(ssbgv, float, FB_OP_SSBGV, 100)
DEFINE_SBGV_TESTS(dsbgv, double, FB_OP_DSBGV, 300)

int main(void)
{
    int status = 0;

    status |= check_ssbgv_fortran_to_cblas();
    status |= check_ssbgv_cblas_to_fortran();
    status |= check_dsbgv_fortran_to_cblas();
    status |= check_dsbgv_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}