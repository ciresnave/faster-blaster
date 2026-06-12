#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

#define DEFINE_SBTRF_REAL_TESTS(SUFFIX, TYPE, OP_ID, BASE)                      \
typedef int (*fb_##SUFFIX##_cblas_fn)(fb_layout_t layout, fb_uplo_t uplo,     \
                                      int n, int kd, TYPE *ab, int ldab);     \
typedef void (*fb_##SUFFIX##_fortran_fn)(char *uplo, int *n, int *kd,         \
                                         TYPE *ab, int *ldab, int *info);     \
static struct {                                                                  \
    int called;                                                                  \
    char uplo;                                                                   \
    int n;                                                                       \
    int kd;                                                                      \
    int ldab;                                                                    \
    TYPE ab_snapshot[6];                                                         \
} g_##SUFFIX##_fortran_call;                                                     \
static struct {                                                                  \
    int called;                                                                  \
    fb_layout_t layout;                                                          \
    fb_uplo_t uplo;                                                              \
    int n;                                                                       \
    int kd;                                                                      \
    int ldab;                                                                    \
    TYPE *ab;                                                                    \
} g_##SUFFIX##_cblas_call;                                                       \
static void stub_##SUFFIX##_fortran(char *uplo, int *n, int *kd, TYPE *ab,    \
                                    int *ldab, int *info)                     \
{                                                                                \
    memcpy(g_##SUFFIX##_fortran_call.ab_snapshot, ab, sizeof(g_##SUFFIX##_fortran_call.ab_snapshot)); \
    g_##SUFFIX##_fortran_call.called += 1;                                       \
    g_##SUFFIX##_fortran_call.uplo = *uplo;                                      \
    g_##SUFFIX##_fortran_call.n = *n;                                            \
    g_##SUFFIX##_fortran_call.kd = *kd;                                          \
    g_##SUFFIX##_fortran_call.ldab = *ldab;                                      \
    ab[0] = (TYPE)0; ab[1] = (TYPE)41; ab[2] = (TYPE)42;                        \
    ab[3] = (TYPE)52; ab[4] = (TYPE)53; ab[5] = (TYPE)63;                       \
    *info = 0;                                                                   \
}                                                                                \
static int stub_##SUFFIX##_cblas(fb_layout_t layout, fb_uplo_t uplo, int n,   \
                                 int kd, TYPE *ab, int ldab)                  \
{                                                                                \
    g_##SUFFIX##_cblas_call.called += 1;                                         \
    g_##SUFFIX##_cblas_call.layout = layout;                                     \
    g_##SUFFIX##_cblas_call.uplo = uplo;                                         \
    g_##SUFFIX##_cblas_call.n = n;                                               \
    g_##SUFFIX##_cblas_call.kd = kd;                                             \
    g_##SUFFIX##_cblas_call.ldab = ldab;                                         \
    g_##SUFFIX##_cblas_call.ab = ab;                                             \
    ab[0] = (TYPE)((BASE) + 1);                                                  \
    return (BASE) + 2;                                                           \
}                                                                                \
static int check_##SUFFIX##_fortran_to_cblas(void)                               \
{                                                                                \
    fb_backend_vtable_t vtable;                                                  \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                         \
    TYPE ab[6] = { (TYPE)0, (TYPE)12, (TYPE)23, (TYPE)11, (TYPE)22, (TYPE)33 }; \
    TYPE expected_ab_in[6] = { (TYPE)0, (TYPE)11, (TYPE)12, (TYPE)22, (TYPE)23, (TYPE)33 }; \
    TYPE expected_ab_out[6] = { (TYPE)0, (TYPE)42, (TYPE)53, (TYPE)41, (TYPE)52, (TYPE)63 }; \
    int info = 0;                                                                \
    memset(&vtable, 0, sizeof(vtable));                                          \
    memset(&g_##SUFFIX##_fortran_call, 0, sizeof(g_##SUFFIX##_fortran_call));    \
    vtable.ext_ops[OP_ID][FB_CONV_FORTRAN] =                                     \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_fortran;                  \
    fb_install_conv_thunks(&vtable, OP_ID);                                      \
    thunk = (fb_##SUFFIX##_cblas_fn)vtable.ext_ops[OP_ID][FB_CONV_CBLAS];        \
    if (!thunk) {                                                                \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS factor thunk was not installed\n"); \
        return 1;                                                                \
    }                                                                            \
    info = thunk(FB_LAYOUT_ROW_MAJOR, FB_UPPER, 3, 1, ab, 3);                   \
    if (info != 0 ||                                                             \
        g_##SUFFIX##_fortran_call.called != 1 ||                                 \
        g_##SUFFIX##_fortran_call.uplo != 'U' ||                                 \
        g_##SUFFIX##_fortran_call.n != 3 ||                                      \
        g_##SUFFIX##_fortran_call.kd != 1 ||                                     \
        g_##SUFFIX##_fortran_call.ldab != 2 ||                                   \
        memcmp(g_##SUFFIX##_fortran_call.ab_snapshot, expected_ab_in, sizeof(expected_ab_in)) != 0 || \
        memcmp(ab, expected_ab_out, sizeof(expected_ab_out)) != 0) {            \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS factor thunk did not translate symmetric band storage correctly\n"); \
        return 1;                                                                \
    }                                                                            \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS factor thunk translates symmetric band storage and copies back\n"); \
    return 0;                                                                    \
}                                                                                \
static int check_##SUFFIX##_cblas_to_fortran(void)                               \
{                                                                                \
    fb_backend_vtable_t vtable;                                                  \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                       \
    char uplo = 'L';                                                             \
    int n = 2;                                                                   \
    int kd = 1;                                                                  \
    TYPE ab[4] = { (TYPE)4, (TYPE)5, (TYPE)6, (TYPE)7 };                        \
    int ldab = 2;                                                                \
    int info = -999;                                                             \
    memset(&vtable, 0, sizeof(vtable));                                          \
    memset(&g_##SUFFIX##_cblas_call, 0, sizeof(g_##SUFFIX##_cblas_call));        \
    vtable.ext_ops[OP_ID][FB_CONV_CBLAS] =                                       \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_cblas;                    \
    fb_install_conv_thunks(&vtable, OP_ID);                                      \
    thunk = (fb_##SUFFIX##_fortran_fn)vtable.ext_ops[OP_ID][FB_CONV_FORTRAN];    \
    if (!thunk) {                                                                \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran factor thunk was not installed\n"); \
        return 1;                                                                \
    }                                                                            \
    thunk(&uplo, &n, &kd, ab, &ldab, &info);                                    \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                   \
        g_##SUFFIX##_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||                 \
        g_##SUFFIX##_cblas_call.uplo != FB_LOWER ||                              \
        g_##SUFFIX##_cblas_call.n != 2 ||                                        \
        g_##SUFFIX##_cblas_call.kd != 1 ||                                       \
        g_##SUFFIX##_cblas_call.ldab != 2 ||                                     \
        g_##SUFFIX##_cblas_call.ab != ab ||                                      \
        info != (BASE) + 2 ||                                                    \
        ab[0] != (TYPE)((BASE) + 1)) {                                           \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran factor thunk did not propagate SBTRF info correctly\n"); \
        return 1;                                                                \
    }                                                                            \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran factor thunk propagates SBTRF info\n"); \
    return 0;                                                                    \
}

DEFINE_SBTRF_REAL_TESTS(ssbtrf, float, FB_OP_SSBTRF, 100)
DEFINE_SBTRF_REAL_TESTS(dsbtrf, double, FB_OP_DSBTRF, 300)

int main(void)
{
    int status = 0;

    status |= check_ssbtrf_fortran_to_cblas();
    status |= check_ssbtrf_cblas_to_fortran();
    status |= check_dsbtrf_fortran_to_cblas();
    status |= check_dsbtrf_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}