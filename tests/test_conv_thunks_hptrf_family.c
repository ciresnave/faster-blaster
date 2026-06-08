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

#define DEFINE_HPTRF_COMPLEX_TESTS(SUFFIX, TYPE, REAL_TYPE, OP_ID, BASE, MAKE, REAL_PART) \
typedef int (*fb_##SUFFIX##_cblas_fn)(fb_layout_t layout, fb_uplo_t uplo,     \
                                      int n, TYPE *ap);                        \
typedef void (*fb_##SUFFIX##_fortran_fn)(char *uplo, int *n, TYPE *ap,        \
                                         int *info);                           \
static struct {                                                                  \
    int called;                                                                  \
    char uplo;                                                                   \
    int n;                                                                       \
    REAL_TYPE ap_real_snapshot[6];                                               \
} g_##SUFFIX##_fortran_call;                                                     \
static struct {                                                                  \
    int called;                                                                  \
    fb_layout_t layout;                                                          \
    fb_uplo_t uplo;                                                              \
    int n;                                                                       \
    TYPE *ap;                                                                    \
} g_##SUFFIX##_cblas_call;                                                       \
static void stub_##SUFFIX##_fortran(char *uplo, int *n, TYPE *ap, int *info)   \
{                                                                                \
    size_t index = 0;                                                            \
    g_##SUFFIX##_fortran_call.called += 1;                                       \
    g_##SUFFIX##_fortran_call.uplo = *uplo;                                      \
    g_##SUFFIX##_fortran_call.n = *n;                                            \
    memset(g_##SUFFIX##_fortran_call.ap_real_snapshot, 0, sizeof(g_##SUFFIX##_fortran_call.ap_real_snapshot)); \
    for (index = 0; index < (size_t)(*n) * (size_t)((*n) + 1) / 2U; ++index) {  \
        g_##SUFFIX##_fortran_call.ap_real_snapshot[index] = REAL_PART(ap[index]);\
    }                                                                            \
    ap[0] = MAKE((REAL_TYPE)31); ap[1] = MAKE((REAL_TYPE)32); ap[2] = MAKE((REAL_TYPE)33); \
    ap[3] = MAKE((REAL_TYPE)34); ap[4] = MAKE((REAL_TYPE)35); ap[5] = MAKE((REAL_TYPE)36); \
    *info = (BASE) + 1;                                                          \
}                                                                                \
static int stub_##SUFFIX##_cblas(fb_layout_t layout, fb_uplo_t uplo, int n,    \
                                 TYPE *ap)                                      \
{                                                                                \
    g_##SUFFIX##_cblas_call.called += 1;                                         \
    g_##SUFFIX##_cblas_call.layout = layout;                                     \
    g_##SUFFIX##_cblas_call.uplo = uplo;                                         \
    g_##SUFFIX##_cblas_call.n = n;                                               \
    g_##SUFFIX##_cblas_call.ap = ap;                                             \
    ap[0] = MAKE((REAL_TYPE)((BASE) + 2));                                       \
    return (BASE) + 3;                                                           \
}                                                                                \
static int check_##SUFFIX##_fortran_to_cblas(void)                               \
{                                                                                \
    fb_backend_vtable_t vtable;                                                  \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                         \
    TYPE ap_row[6] = { MAKE((REAL_TYPE)11), MAKE((REAL_TYPE)12), MAKE((REAL_TYPE)13), MAKE((REAL_TYPE)22), MAKE((REAL_TYPE)23), MAKE((REAL_TYPE)33) }; \
    REAL_TYPE expected_in[6] = { (REAL_TYPE)11, (REAL_TYPE)12, (REAL_TYPE)22, (REAL_TYPE)13, (REAL_TYPE)23, (REAL_TYPE)33 }; \
    REAL_TYPE expected_out[6] = { (REAL_TYPE)31, (REAL_TYPE)32, (REAL_TYPE)34, (REAL_TYPE)33, (REAL_TYPE)35, (REAL_TYPE)36 }; \
    int info = 0;                                                                \
    size_t index = 0;                                                            \
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
    info = thunk(FB_LAYOUT_ROW_MAJOR, FB_UPPER, 3, ap_row);                     \
    if (info != (BASE) + 1 ||                                                    \
        g_##SUFFIX##_fortran_call.called != 1 ||                                 \
        g_##SUFFIX##_fortran_call.uplo != 'U' ||                                 \
        g_##SUFFIX##_fortran_call.n != 3) {                                      \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not route complex HPTRF calls\n"); \
        return 1;                                                                \
    }                                                                            \
    for (index = 0; index < 6; ++index) {                                        \
        if (g_##SUFFIX##_fortran_call.ap_real_snapshot[index] != expected_in[index] || \
            REAL_PART(ap_row[index]) != expected_out[index]) {                   \
            fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not translate complex packed HPTRF storage correctly\n"); \
            return 1;                                                            \
        }                                                                        \
    }                                                                            \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk translates complex packed HPTRF storage\n"); \
    return 0;                                                                    \
}                                                                                \
static int check_##SUFFIX##_cblas_to_fortran(void)                               \
{                                                                                \
    fb_backend_vtable_t vtable;                                                  \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                       \
    char uplo = 'L';                                                             \
    int n = 2;                                                                   \
    TYPE ap[3] = { MAKE((REAL_TYPE)7), MAKE((REAL_TYPE)8), MAKE((REAL_TYPE)9) };\
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
    thunk(&uplo, &n, ap, &info);                                                 \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                   \
        g_##SUFFIX##_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||                 \
        g_##SUFFIX##_cblas_call.uplo != FB_LOWER ||                              \
        g_##SUFFIX##_cblas_call.n != 2 ||                                        \
        g_##SUFFIX##_cblas_call.ap != ap ||                                      \
        info != (BASE) + 3 || REAL_PART(ap[0]) != (REAL_TYPE)((BASE) + 2)) {    \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not propagate complex HPTRF info correctly\n"); \
        return 1;                                                                \
    }                                                                            \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk propagates complex HPTRF info\n"); \
    return 0;                                                                    \
}

DEFINE_HPTRF_COMPLEX_TESTS(chptrf, fb_complex_float_t, float, FB_OP_CHPTRF, 100, make_cfloat, cfloat_real)
DEFINE_HPTRF_COMPLEX_TESTS(zhptrf, fb_complex_double_t, double, FB_OP_ZHPTRF, 300, make_cdouble, cdouble_real)

int main(void)
{
    int status = 0;

    status |= check_chptrf_fortran_to_cblas();
    status |= check_chptrf_cblas_to_fortran();
    status |= check_zhptrf_fortran_to_cblas();
    status |= check_zhptrf_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}