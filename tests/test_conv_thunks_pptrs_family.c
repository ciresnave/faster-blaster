#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

#define DEFINE_PPTRS_TESTS(SUFFIX, TYPE, OP_ID, BASE)                            \
typedef int (*fb_##SUFFIX##_cblas_fn)(fb_layout_t layout, fb_uplo_t uplo,      \
                                      int n, int nrhs, const TYPE *ap, TYPE *b,\
                                      int ldb);                                 \
typedef void (*fb_##SUFFIX##_fortran_fn)(char *uplo, int *n, int *nrhs,        \
                                         const TYPE *ap, TYPE *b, int *ldb,    \
                                         int *info);                            \
static struct {                                                                   \
    int called;                                                                   \
    char uplo;                                                                     \
    int n;                                                                         \
    int nrhs;                                                                      \
    TYPE ap_snapshot[6];                                                           \
    TYPE b_snapshot[6];                                                            \
    int ldb;                                                                       \
} g_##SUFFIX##_fortran_call;                                                      \
static struct {                                                                   \
    int called;                                                                   \
    fb_layout_t layout;                                                           \
    fb_uplo_t uplo;                                                               \
    int n;                                                                         \
    int nrhs;                                                                      \
    const TYPE *ap;                                                                \
    TYPE *b;                                                                       \
    int ldb;                                                                       \
} g_##SUFFIX##_cblas_call;                                                        \
static void stub_##SUFFIX##_fortran(char *uplo, int *n, int *nrhs,              \
                                    const TYPE *ap, TYPE *b, int *ldb,          \
                                    int *info)                                  \
{                                                                                \
    size_t packed_len = (size_t)(*n) * (size_t)((*n) + 1) / 2U;                 \
    size_t mat_elems = (size_t)(*n) * (size_t)(*nrhs);                           \
    g_##SUFFIX##_fortran_call.called += 1;                                       \
    g_##SUFFIX##_fortran_call.uplo = *uplo;                                      \
    g_##SUFFIX##_fortran_call.n = *n;                                            \
    g_##SUFFIX##_fortran_call.nrhs = *nrhs;                                      \
    memset(g_##SUFFIX##_fortran_call.ap_snapshot, 0, sizeof(g_##SUFFIX##_fortran_call.ap_snapshot)); \
    memset(g_##SUFFIX##_fortran_call.b_snapshot, 0, sizeof(g_##SUFFIX##_fortran_call.b_snapshot)); \
    memcpy(g_##SUFFIX##_fortran_call.ap_snapshot, ap, packed_len * sizeof(TYPE));\
    memcpy(g_##SUFFIX##_fortran_call.b_snapshot, b, mat_elems * sizeof(TYPE));   \
    g_##SUFFIX##_fortran_call.ldb = *ldb;                                        \
    b[0] = (TYPE)((BASE) + 1);                                                   \
    b[1] = (TYPE)((BASE) + 2);                                                   \
    b[2] = (TYPE)((BASE) + 3);                                                   \
    b[3] = (TYPE)((BASE) + 4);                                                   \
    b[4] = (TYPE)((BASE) + 5);                                                   \
    b[5] = (TYPE)((BASE) + 6);                                                   \
    *info = (BASE) + 7;                                                          \
}                                                                                \
static int stub_##SUFFIX##_cblas(fb_layout_t layout, fb_uplo_t uplo, int n,    \
                                 int nrhs, const TYPE *ap, TYPE *b, int ldb)   \
{                                                                                \
    g_##SUFFIX##_cblas_call.called += 1;                                         \
    g_##SUFFIX##_cblas_call.layout = layout;                                     \
    g_##SUFFIX##_cblas_call.uplo = uplo;                                         \
    g_##SUFFIX##_cblas_call.n = n;                                               \
    g_##SUFFIX##_cblas_call.nrhs = nrhs;                                         \
    g_##SUFFIX##_cblas_call.ap = ap;                                             \
    g_##SUFFIX##_cblas_call.b = b;                                               \
    g_##SUFFIX##_cblas_call.ldb = ldb;                                           \
    b[0] = (TYPE)((BASE) + 8);                                                   \
    return (BASE) + 9;                                                           \
}                                                                                \
static int check_##SUFFIX##_fortran_to_cblas(void)                               \
{                                                                                \
    fb_backend_vtable_t vtable;                                                  \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                         \
    TYPE ap_row[6] = { (TYPE)11, (TYPE)12, (TYPE)13, (TYPE)22, (TYPE)23, (TYPE)33 }; \
    TYPE ap_original[6] = { (TYPE)11, (TYPE)12, (TYPE)13, (TYPE)22, (TYPE)23, (TYPE)33 }; \
    TYPE expected_ap_in[6] = { (TYPE)11, (TYPE)12, (TYPE)22, (TYPE)13, (TYPE)23, (TYPE)33 }; \
    TYPE b_row[6] = { (TYPE)1, (TYPE)2, (TYPE)3, (TYPE)4, (TYPE)5, (TYPE)6 };  \
    TYPE expected_b_in[6] = { (TYPE)1, (TYPE)3, (TYPE)5, (TYPE)2, (TYPE)4, (TYPE)6 }; \
    TYPE expected_b_out[6] = { (TYPE)((BASE) + 1), (TYPE)((BASE) + 4), (TYPE)((BASE) + 2), (TYPE)((BASE) + 5), (TYPE)((BASE) + 3), (TYPE)((BASE) + 6) }; \
    int info = 0;                                                                \
    memset(&vtable, 0, sizeof(vtable));                                          \
    memset(&g_##SUFFIX##_fortran_call, 0, sizeof(g_##SUFFIX##_fortran_call));    \
    vtable.ext_ops[OP_ID][FB_CONV_FORTRAN] =                                     \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_fortran;                  \
    fb_install_conv_thunks(&vtable, OP_ID);                                      \
    thunk = (fb_##SUFFIX##_cblas_fn)vtable.ext_ops[OP_ID][FB_CONV_CBLAS];        \
    if (!thunk) {                                                                \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk was not installed\n"); \
        return 1;                                                                \
    }                                                                            \
    info = thunk(FB_LAYOUT_ROW_MAJOR, FB_UPPER, 3, 2, ap_row, b_row, 2);        \
    if (info != (BASE) + 7 ||                                                    \
        g_##SUFFIX##_fortran_call.called != 1 ||                                 \
        g_##SUFFIX##_fortran_call.uplo != 'U' ||                                 \
        g_##SUFFIX##_fortran_call.n != 3 ||                                      \
        g_##SUFFIX##_fortran_call.nrhs != 2 ||                                   \
        memcmp(g_##SUFFIX##_fortran_call.ap_snapshot, expected_ap_in, sizeof(expected_ap_in)) != 0 || \
        memcmp(g_##SUFFIX##_fortran_call.b_snapshot, expected_b_in, sizeof(expected_b_in)) != 0 || \
        g_##SUFFIX##_fortran_call.ldb != 3 ||                                    \
        memcmp(ap_row, ap_original, sizeof(ap_original)) != 0 ||                \
        memcmp(b_row, expected_b_out, sizeof(expected_b_out)) != 0) {          \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not route packed PPTRS inputs correctly\n"); \
        return 1;                                                                \
    }                                                                            \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk routes packed PPTRS inputs\n"); \
    return 0;                                                                    \
}                                                                                \
static int check_##SUFFIX##_cblas_to_fortran(void)                               \
{                                                                                \
    fb_backend_vtable_t vtable;                                                  \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                       \
    char uplo = 'L';                                                             \
    int n = 1;                                                                   \
    int nrhs = 1;                                                                \
    TYPE ap[1] = { (TYPE)9 };                                                    \
    TYPE b[1] = { (TYPE)10 };                                                    \
    int ldb = 1;                                                                 \
    int info = -999;                                                             \
    memset(&vtable, 0, sizeof(vtable));                                          \
    memset(&g_##SUFFIX##_cblas_call, 0, sizeof(g_##SUFFIX##_cblas_call));        \
    vtable.ext_ops[OP_ID][FB_CONV_CBLAS] =                                       \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_cblas;                    \
    fb_install_conv_thunks(&vtable, OP_ID);                                      \
    thunk = (fb_##SUFFIX##_fortran_fn)vtable.ext_ops[OP_ID][FB_CONV_FORTRAN];    \
    if (!thunk) {                                                                \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk was not installed\n"); \
        return 1;                                                                \
    }                                                                            \
    thunk(&uplo, &n, &nrhs, ap, b, &ldb, &info);                                 \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                   \
        g_##SUFFIX##_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||                 \
        g_##SUFFIX##_cblas_call.uplo != FB_LOWER ||                              \
        g_##SUFFIX##_cblas_call.n != 1 ||                                        \
        g_##SUFFIX##_cblas_call.nrhs != 1 ||                                     \
        g_##SUFFIX##_cblas_call.ap != ap ||                                      \
        g_##SUFFIX##_cblas_call.b != b ||                                        \
        g_##SUFFIX##_cblas_call.ldb != 1 ||                                      \
        info != (BASE) + 9 || b[0] != (TYPE)((BASE) + 8)) {                     \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not propagate PPTRS info correctly\n"); \
        return 1;                                                                \
    }                                                                            \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk propagates PPTRS info\n"); \
    return 0;                                                                    \
}

DEFINE_PPTRS_TESTS(spptrs, float, FB_OP_SPPTRS, 100)
DEFINE_PPTRS_TESTS(dpptrs, double, FB_OP_DPPTRS, 300)
DEFINE_PPTRS_TESTS(cpptrs, fb_complex_float_t, FB_OP_CPPTRS, 500)
DEFINE_PPTRS_TESTS(zpptrs, fb_complex_double_t, FB_OP_ZPPTRS, 700)

int main(void)
{
    int status = 0;

    status |= check_spptrs_fortran_to_cblas();
    status |= check_spptrs_cblas_to_fortran();
    status |= check_dpptrs_fortran_to_cblas();
    status |= check_dpptrs_cblas_to_fortran();
    status |= check_cpptrs_fortran_to_cblas();
    status |= check_cpptrs_cblas_to_fortran();
    status |= check_zpptrs_fortran_to_cblas();
    status |= check_zpptrs_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}