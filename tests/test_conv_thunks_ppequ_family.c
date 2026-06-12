#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

#define DEFINE_PPEQU_TESTS(SUFFIX, MAT_TYPE, REAL_TYPE, OP_ID, BASE)             \
typedef int (*fb_##SUFFIX##_cblas_fn)(fb_layout_t layout, fb_uplo_t uplo,      \
                                      int n, const MAT_TYPE *ap, REAL_TYPE *s,  \
                                      REAL_TYPE *scond, REAL_TYPE *amax);       \
typedef void (*fb_##SUFFIX##_fortran_fn)(char *uplo, int *n, const MAT_TYPE *ap,\
                                         REAL_TYPE *s, REAL_TYPE *scond,        \
                                         REAL_TYPE *amax, int *info);           \
static struct {                                                                    \
    int called;                                                                    \
    char uplo;                                                                      \
    int n;                                                                          \
    MAT_TYPE ap_snapshot[6];                                                        \
    REAL_TYPE *s;                                                                   \
    REAL_TYPE *scond;                                                               \
    REAL_TYPE *amax;                                                                \
} g_##SUFFIX##_fortran_call;                                                       \
static struct {                                                                    \
    int called;                                                                    \
    fb_layout_t layout;                                                            \
    fb_uplo_t uplo;                                                                \
    int n;                                                                         \
    const MAT_TYPE *ap;                                                            \
    REAL_TYPE *s;                                                                  \
    REAL_TYPE *scond;                                                              \
    REAL_TYPE *amax;                                                               \
} g_##SUFFIX##_cblas_call;                                                         \
static void stub_##SUFFIX##_fortran(char *uplo, int *n, const MAT_TYPE *ap,     \
                                    REAL_TYPE *s, REAL_TYPE *scond,             \
                                    REAL_TYPE *amax, int *info)                 \
{                                                                                 \
    size_t elems = (size_t)(*n * (*n + 1)) / 2U;                                 \
    g_##SUFFIX##_fortran_call.called += 1;                                        \
    g_##SUFFIX##_fortran_call.uplo = *uplo;                                       \
    g_##SUFFIX##_fortran_call.n = *n;                                             \
    memset(g_##SUFFIX##_fortran_call.ap_snapshot, 0, sizeof(g_##SUFFIX##_fortran_call.ap_snapshot)); \
    if (elems > 0) {                                                              \
        memcpy(g_##SUFFIX##_fortran_call.ap_snapshot, ap, elems * sizeof(MAT_TYPE)); \
    }                                                                             \
    g_##SUFFIX##_fortran_call.s = s;                                              \
    g_##SUFFIX##_fortran_call.scond = scond;                                      \
    g_##SUFFIX##_fortran_call.amax = amax;                                        \
    s[0] = (REAL_TYPE)((BASE) + 1);                                               \
    if (*n > 1) s[1] = (REAL_TYPE)((BASE) + 2);                                  \
    *scond = (REAL_TYPE)((BASE) + 3);                                             \
    *amax = (REAL_TYPE)((BASE) + 4);                                              \
    *info = (BASE) + 5;                                                           \
}                                                                                 \
static int stub_##SUFFIX##_cblas(fb_layout_t layout, fb_uplo_t uplo, int n,     \
                                 const MAT_TYPE *ap, REAL_TYPE *s,              \
                                 REAL_TYPE *scond, REAL_TYPE *amax)             \
{                                                                                 \
    g_##SUFFIX##_cblas_call.called += 1;                                          \
    g_##SUFFIX##_cblas_call.layout = layout;                                      \
    g_##SUFFIX##_cblas_call.uplo = uplo;                                          \
    g_##SUFFIX##_cblas_call.n = n;                                                \
    g_##SUFFIX##_cblas_call.ap = ap;                                              \
    g_##SUFFIX##_cblas_call.s = s;                                                \
    g_##SUFFIX##_cblas_call.scond = scond;                                        \
    g_##SUFFIX##_cblas_call.amax = amax;                                          \
    s[0] = (REAL_TYPE)((BASE) + 6);                                               \
    if (n > 1) s[1] = (REAL_TYPE)((BASE) + 7);                                   \
    *scond = (REAL_TYPE)((BASE) + 8);                                             \
    *amax = (REAL_TYPE)((BASE) + 9);                                              \
    return (BASE) + 10;                                                           \
}                                                                                 \
static int check_##SUFFIX##_fortran_to_cblas(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                          \
    MAT_TYPE ap_row[6] = { (MAT_TYPE)11, (MAT_TYPE)12, (MAT_TYPE)13, (MAT_TYPE)22, (MAT_TYPE)23, (MAT_TYPE)33 }; \
    MAT_TYPE expected_col[6] = { (MAT_TYPE)11, (MAT_TYPE)12, (MAT_TYPE)22, (MAT_TYPE)13, (MAT_TYPE)23, (MAT_TYPE)33 }; \
    REAL_TYPE s[3] = { 0, 0, 0 };                                                 \
    REAL_TYPE scond = 0;                                                          \
    REAL_TYPE amax = 0;                                                           \
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
    info = thunk(FB_LAYOUT_ROW_MAJOR, FB_UPPER, 3, ap_row, s, &scond, &amax);    \
    if (info != (BASE) + 5 ||                                                     \
        g_##SUFFIX##_fortran_call.called != 1 ||                                  \
        g_##SUFFIX##_fortran_call.uplo != 'U' ||                                  \
        g_##SUFFIX##_fortran_call.n != 3 ||                                       \
        memcmp(g_##SUFFIX##_fortran_call.ap_snapshot, expected_col, sizeof(expected_col)) != 0 || \
        g_##SUFFIX##_fortran_call.s != s ||                                       \
        g_##SUFFIX##_fortran_call.scond != &scond ||                              \
        g_##SUFFIX##_fortran_call.amax != &amax ||                                \
        s[0] != (REAL_TYPE)((BASE) + 1) || s[1] != (REAL_TYPE)((BASE) + 2) ||    \
        scond != (REAL_TYPE)((BASE) + 3) || amax != (REAL_TYPE)((BASE) + 4)) {   \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not route packed PPEQU inputs correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk routes packed PPEQU inputs\n"); \
    return 0;                                                                     \
}                                                                                 \
static int check_##SUFFIX##_cblas_to_fortran(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                        \
    char uplo = 'L';                                                              \
    int n = 2;                                                                    \
    MAT_TYPE ap[3] = { (MAT_TYPE)7, (MAT_TYPE)8, (MAT_TYPE)9 };                  \
    REAL_TYPE s[2] = { 0, 0 };                                                    \
    REAL_TYPE scond = 0;                                                          \
    REAL_TYPE amax = 0;                                                           \
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
    thunk(&uplo, &n, ap, s, &scond, &amax, &info);                               \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                    \
        g_##SUFFIX##_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||                  \
        g_##SUFFIX##_cblas_call.uplo != FB_LOWER ||                               \
        g_##SUFFIX##_cblas_call.n != 2 ||                                         \
        g_##SUFFIX##_cblas_call.ap != ap ||                                       \
        g_##SUFFIX##_cblas_call.s != s ||                                         \
        g_##SUFFIX##_cblas_call.scond != &scond ||                                \
        g_##SUFFIX##_cblas_call.amax != &amax ||                                  \
        info != (BASE) + 10 ||                                                    \
        s[0] != (REAL_TYPE)((BASE) + 6) || s[1] != (REAL_TYPE)((BASE) + 7) ||    \
        scond != (REAL_TYPE)((BASE) + 8) || amax != (REAL_TYPE)((BASE) + 9)) {   \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not propagate PPEQU info correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk propagates PPEQU info\n"); \
    return 0;                                                                     \
}

DEFINE_PPEQU_TESTS(sppequ, float, float, FB_OP_SPPEQU, 100)
DEFINE_PPEQU_TESTS(dppequ, double, double, FB_OP_DPPEQU, 300)
DEFINE_PPEQU_TESTS(cppequ, fb_complex_float_t, float, FB_OP_CPPEQU, 500)
DEFINE_PPEQU_TESTS(zppequ, fb_complex_double_t, double, FB_OP_ZPPEQU, 700)

int main(void)
{
    int status = 0;

    status |= check_sppequ_fortran_to_cblas();
    status |= check_sppequ_cblas_to_fortran();
    status |= check_dppequ_fortran_to_cblas();
    status |= check_dppequ_cblas_to_fortran();
    status |= check_cppequ_fortran_to_cblas();
    status |= check_cppequ_cblas_to_fortran();
    status |= check_zppequ_fortran_to_cblas();
    status |= check_zppequ_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}