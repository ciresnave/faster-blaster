#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

static fb_complex_float_t make_cfloat(float real_value)
{
    fb_complex_float_t value = (fb_complex_float_t)0;
    __real__ value = real_value;
    __imag__ value = 0.0f;
    return value;
}

static float cfloat_real(fb_complex_float_t value)
{
    return __real__ value;
}

static fb_complex_double_t make_cdouble(double real_value)
{
    fb_complex_double_t value = (fb_complex_double_t)0;
    __real__ value = real_value;
    __imag__ value = 0.0;
    return value;
}

static double cdouble_real(fb_complex_double_t value)
{
    return __real__ value;
}

#define DEFINE_HPSV_TESTS(SUFFIX, TYPE, REAL_TYPE, OP_ID, BASE, MAKE, REAL_PART) \
typedef int (*fb_##SUFFIX##_cblas_fn)(fb_layout_t layout, fb_uplo_t uplo,      \
                                      int n, int nrhs, TYPE *ap, const int *ipiv, \
                                      TYPE *b, int ldb);                        \
typedef void (*fb_##SUFFIX##_fortran_fn)(char *uplo, int *n, int *nrhs,        \
                                         TYPE *ap, int *ipiv, TYPE *b, int *ldb, \
                                         int *info);                           \
static struct {                                                                   \
    int called;                                                                   \
    char uplo;                                                                    \
    int n;                                                                        \
    int nrhs;                                                                     \
    REAL_TYPE ap_snapshot[6];                                                     \
    int ipiv_vals[3];                                                             \
    REAL_TYPE b_snapshot[6];                                                      \
    int ldb;                                                                      \
} g_##SUFFIX##_fortran_call;                                                      \
static struct {                                                                   \
    int called;                                                                   \
    fb_layout_t layout;                                                           \
    fb_uplo_t uplo;                                                               \
    int n;                                                                        \
    int nrhs;                                                                     \
    TYPE *ap;                                                                     \
    const int *ipiv;                                                              \
    TYPE *b;                                                                      \
    int ldb;                                                                      \
} g_##SUFFIX##_cblas_call;                                                        \
static void stub_##SUFFIX##_fortran(char *uplo, int *n, int *nrhs, TYPE *ap,   \
                                    int *ipiv, TYPE *b, int *ldb, int *info)   \
{                                                                                 \
    size_t packed_len = (size_t)(*n * (*n + 1)) / 2U;                            \
    size_t mat_elems = (size_t)(*n) * (size_t)(*nrhs);                           \
    size_t index = 0;                                                             \
    g_##SUFFIX##_fortran_call.called += 1;                                        \
    g_##SUFFIX##_fortran_call.uplo = *uplo;                                       \
    g_##SUFFIX##_fortran_call.n = *n;                                             \
    g_##SUFFIX##_fortran_call.nrhs = *nrhs;                                       \
    memset(g_##SUFFIX##_fortran_call.ap_snapshot, 0, sizeof(g_##SUFFIX##_fortran_call.ap_snapshot)); \
    memset(g_##SUFFIX##_fortran_call.b_snapshot, 0, sizeof(g_##SUFFIX##_fortran_call.b_snapshot)); \
    for (index = 0; index < packed_len; ++index) {                               \
        g_##SUFFIX##_fortran_call.ap_snapshot[index] = REAL_PART(ap[index]);     \
    }                                                                             \
    for (index = 0; index < mat_elems; ++index) {                                \
        g_##SUFFIX##_fortran_call.b_snapshot[index] = REAL_PART(b[index]);       \
    }                                                                             \
    memset(g_##SUFFIX##_fortran_call.ipiv_vals, 0, sizeof(g_##SUFFIX##_fortran_call.ipiv_vals)); \
    if (*n > 0) {                                                                 \
        memcpy(g_##SUFFIX##_fortran_call.ipiv_vals, ipiv, (size_t)(*n) * sizeof(int)); \
    }                                                                             \
    g_##SUFFIX##_fortran_call.ldb = *ldb;                                         \
    ap[0] = MAKE((REAL_TYPE)((BASE) + 1));                                        \
    ap[1] = MAKE((REAL_TYPE)((BASE) + 2));                                        \
    ap[2] = MAKE((REAL_TYPE)((BASE) + 3));                                        \
    b[0] = MAKE((REAL_TYPE)((BASE) + 4));                                         \
    b[1] = MAKE((REAL_TYPE)((BASE) + 5));                                         \
    b[2] = MAKE((REAL_TYPE)((BASE) + 6));                                         \
    b[3] = MAKE((REAL_TYPE)((BASE) + 7));                                         \
    *info = (BASE) + 8;                                                           \
}                                                                                 \
static int stub_##SUFFIX##_cblas(fb_layout_t layout, fb_uplo_t uplo, int n,     \
                                 int nrhs, TYPE *ap, const int *ipiv, TYPE *b,  \
                                 int ldb)                                       \
{                                                                                 \
    g_##SUFFIX##_cblas_call.called += 1;                                          \
    g_##SUFFIX##_cblas_call.layout = layout;                                      \
    g_##SUFFIX##_cblas_call.uplo = uplo;                                          \
    g_##SUFFIX##_cblas_call.n = n;                                                \
    g_##SUFFIX##_cblas_call.nrhs = nrhs;                                          \
    g_##SUFFIX##_cblas_call.ap = ap;                                              \
    g_##SUFFIX##_cblas_call.ipiv = ipiv;                                          \
    g_##SUFFIX##_cblas_call.b = b;                                                \
    g_##SUFFIX##_cblas_call.ldb = ldb;                                            \
    ap[0] = MAKE((REAL_TYPE)((BASE) + 9));                                        \
    b[0] = MAKE((REAL_TYPE)((BASE) + 10));                                        \
    return (BASE) + 11;                                                           \
}                                                                                 \
static int check_##SUFFIX##_fortran_to_cblas(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                          \
    TYPE ap_row[6] = { MAKE((REAL_TYPE)11), MAKE((REAL_TYPE)12), MAKE((REAL_TYPE)13), MAKE((REAL_TYPE)22), MAKE((REAL_TYPE)23), MAKE((REAL_TYPE)33) }; \
    REAL_TYPE expected_ap[6] = { (REAL_TYPE)11, (REAL_TYPE)12, (REAL_TYPE)13, 0, 0, 0 }; \
    int ipiv[2] = { 1, -2 };                                                      \
    TYPE b_row[4] = { MAKE((REAL_TYPE)1), MAKE((REAL_TYPE)2), MAKE((REAL_TYPE)3), MAKE((REAL_TYPE)4) }; \
    REAL_TYPE expected_b[4] = { (REAL_TYPE)1, (REAL_TYPE)3, (REAL_TYPE)2, (REAL_TYPE)4 }; \
    REAL_TYPE expected_ap_after[6] = { (REAL_TYPE)((BASE) + 1), (REAL_TYPE)((BASE) + 2), (REAL_TYPE)((BASE) + 3), (REAL_TYPE)22, (REAL_TYPE)23, (REAL_TYPE)33 }; \
    REAL_TYPE expected_b_after[4] = { (REAL_TYPE)((BASE) + 4), (REAL_TYPE)((BASE) + 6), (REAL_TYPE)((BASE) + 5), (REAL_TYPE)((BASE) + 7) }; \
    int info = 0;                                                                 \
    size_t index = 0;                                                             \
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
    info = thunk(FB_LAYOUT_ROW_MAJOR, FB_UPPER, 2, 2, ap_row, ipiv, b_row, 2);   \
    if (info != (BASE) + 8 ||                                                     \
        g_##SUFFIX##_fortran_call.called != 1 ||                                  \
        g_##SUFFIX##_fortran_call.uplo != 'U' ||                                  \
        g_##SUFFIX##_fortran_call.n != 2 ||                                       \
        g_##SUFFIX##_fortran_call.nrhs != 2 ||                                    \
        memcmp(g_##SUFFIX##_fortran_call.ipiv_vals, ipiv, sizeof(ipiv)) != 0 ||   \
        g_##SUFFIX##_fortran_call.ldb != 2) {                                    \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not route packed HPSV calls\n"); \
        return 1;                                                                 \
    }                                                                             \
    for (index = 0; index < 6; ++index) {                                        \
        if (g_##SUFFIX##_fortran_call.ap_snapshot[index] != expected_ap[index] || \
            REAL_PART(ap_row[index]) != expected_ap_after[index]) {              \
            fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not translate packed HPSV AP correctly\n"); \
            return 1;                                                             \
        }                                                                         \
    }                                                                             \
    for (index = 0; index < 4; ++index) {                                        \
        if (g_##SUFFIX##_fortran_call.b_snapshot[index] != expected_b[index] ||  \
            REAL_PART(b_row[index]) != expected_b_after[index]) {                \
            fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not translate packed HPSV B correctly\n"); \
            return 1;                                                             \
        }                                                                         \
    }                                                                             \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk routes packed HPSV inputs\n"); \
    return 0;                                                                     \
}                                                                                 \
static int check_##SUFFIX##_cblas_to_fortran(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                        \
    char uplo = 'L';                                                              \
    int n = 1;                                                                    \
    int nrhs = 1;                                                                 \
    TYPE ap[1] = { MAKE((REAL_TYPE)9) };                                          \
    int ipiv[1] = { 1 };                                                          \
    TYPE b[1] = { MAKE((REAL_TYPE)10) };                                          \
    int ldb = 1;                                                                  \
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
    thunk(&uplo, &n, &nrhs, ap, ipiv, b, &ldb, &info);                           \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                    \
        g_##SUFFIX##_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||                  \
        g_##SUFFIX##_cblas_call.uplo != FB_LOWER ||                               \
        g_##SUFFIX##_cblas_call.n != 1 ||                                         \
        g_##SUFFIX##_cblas_call.nrhs != 1 ||                                      \
        g_##SUFFIX##_cblas_call.ap != ap ||                                       \
        g_##SUFFIX##_cblas_call.ipiv != ipiv ||                                   \
        g_##SUFFIX##_cblas_call.b != b ||                                         \
        g_##SUFFIX##_cblas_call.ldb != 1 ||                                       \
        info != (BASE) + 11 || REAL_PART(ap[0]) != (REAL_TYPE)((BASE) + 9) ||    \
        REAL_PART(b[0]) != (REAL_TYPE)((BASE) + 10)) {                           \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not propagate HPSV info correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk propagates HPSV info\n"); \
    return 0;                                                                     \
}

DEFINE_HPSV_TESTS(chpsv, fb_complex_float_t, float, FB_OP_CHPSV, 100, make_cfloat, cfloat_real)
DEFINE_HPSV_TESTS(zhpsv, fb_complex_double_t, double, FB_OP_ZHPSV, 300, make_cdouble, cdouble_real)

int main(void)
{
    int status = 0;

    status |= check_chpsv_fortran_to_cblas();
    status |= check_chpsv_cblas_to_fortran();
    status |= check_zhpsv_fortran_to_cblas();
    status |= check_zhpsv_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}