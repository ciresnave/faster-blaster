#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

static fb_complex_float_t make_cf32(float real_value, float imag_value)
{
    fb_complex_float_t value;

    __real__ value = real_value;
    __imag__ value = imag_value;
    return value;
}

static fb_complex_double_t make_cf64(double real_value, double imag_value)
{
    fb_complex_double_t value;

    __real__ value = real_value;
    __imag__ value = imag_value;
    return value;
}

static int cf32_eq(fb_complex_float_t lhs, fb_complex_float_t rhs)
{
    return __real__ lhs == __real__ rhs && __imag__ lhs == __imag__ rhs;
}

static int cf64_eq(fb_complex_double_t lhs, fb_complex_double_t rhs)
{
    return __real__ lhs == __real__ rhs && __imag__ lhs == __imag__ rhs;
}

#define DEFINE_LAESY_TESTS(SUFFIX, CTYPE, OP_ID, MAKE_FN, EQ_FN, BASE)         \
typedef int (*fb_##SUFFIX##_cblas_fn)(CTYPE a, CTYPE b, CTYPE c, CTYPE *rt1,   \
                                      CTYPE *rt2, CTYPE *evscal, CTYPE *cs1,   \
                                      CTYPE *sn1);                             \
typedef void (*fb_##SUFFIX##_fortran_fn)(CTYPE *a, CTYPE *b, CTYPE *c,         \
                                         CTYPE *rt1, CTYPE *rt2,               \
                                         CTYPE *evscal, CTYPE *cs1,            \
                                         CTYPE *sn1);                          \
static struct {                                                                 \
    int called;                                                                 \
    CTYPE a;                                                                    \
    CTYPE b;                                                                    \
    CTYPE c;                                                                    \
    CTYPE *rt1;                                                                 \
    CTYPE *rt2;                                                                 \
    CTYPE *evscal;                                                              \
    CTYPE *cs1;                                                                 \
    CTYPE *sn1;                                                                 \
} g_##SUFFIX##_fortran_call;                                                    \
static struct {                                                                 \
    int called;                                                                 \
    CTYPE a;                                                                    \
    CTYPE b;                                                                    \
    CTYPE c;                                                                    \
    CTYPE *rt1;                                                                 \
    CTYPE *rt2;                                                                 \
    CTYPE *evscal;                                                              \
    CTYPE *cs1;                                                                 \
    CTYPE *sn1;                                                                 \
} g_##SUFFIX##_cblas_call;                                                      \
static void stub_##SUFFIX##_fortran(CTYPE *a, CTYPE *b, CTYPE *c, CTYPE *rt1,  \
                                    CTYPE *rt2, CTYPE *evscal, CTYPE *cs1,     \
                                    CTYPE *sn1)                                \
{                                                                               \
    g_##SUFFIX##_fortran_call.called += 1;                                      \
    g_##SUFFIX##_fortran_call.a = *a;                                           \
    g_##SUFFIX##_fortran_call.b = *b;                                           \
    g_##SUFFIX##_fortran_call.c = *c;                                           \
    g_##SUFFIX##_fortran_call.rt1 = rt1;                                        \
    g_##SUFFIX##_fortran_call.rt2 = rt2;                                        \
    g_##SUFFIX##_fortran_call.evscal = evscal;                                  \
    g_##SUFFIX##_fortran_call.cs1 = cs1;                                        \
    g_##SUFFIX##_fortran_call.sn1 = sn1;                                        \
    *rt1 = MAKE_FN((BASE) + 1, (BASE) + 11);                                    \
    *rt2 = MAKE_FN((BASE) + 2, (BASE) + 12);                                    \
    *evscal = MAKE_FN((BASE) + 3, (BASE) + 13);                                 \
    *cs1 = MAKE_FN((BASE) + 4, (BASE) + 14);                                    \
    *sn1 = MAKE_FN((BASE) + 5, (BASE) + 15);                                    \
}                                                                               \
static int stub_##SUFFIX##_cblas(CTYPE a, CTYPE b, CTYPE c, CTYPE *rt1,        \
                                 CTYPE *rt2, CTYPE *evscal, CTYPE *cs1,        \
                                 CTYPE *sn1)                                   \
{                                                                               \
    g_##SUFFIX##_cblas_call.called += 1;                                        \
    g_##SUFFIX##_cblas_call.a = a;                                              \
    g_##SUFFIX##_cblas_call.b = b;                                              \
    g_##SUFFIX##_cblas_call.c = c;                                              \
    g_##SUFFIX##_cblas_call.rt1 = rt1;                                          \
    g_##SUFFIX##_cblas_call.rt2 = rt2;                                          \
    g_##SUFFIX##_cblas_call.evscal = evscal;                                    \
    g_##SUFFIX##_cblas_call.cs1 = cs1;                                          \
    g_##SUFFIX##_cblas_call.sn1 = sn1;                                          \
    *rt1 = MAKE_FN((BASE) + 6, (BASE) + 16);                                    \
    *rt2 = MAKE_FN((BASE) + 7, (BASE) + 17);                                    \
    *evscal = MAKE_FN((BASE) + 8, (BASE) + 18);                                 \
    *cs1 = MAKE_FN((BASE) + 9, (BASE) + 19);                                    \
    *sn1 = MAKE_FN((BASE) + 10, (BASE) + 20);                                   \
    return (BASE) + 99;                                                         \
}                                                                               \
static int check_##SUFFIX##_fortran_to_cblas(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                        \
    CTYPE rt1 = MAKE_FN(0, 0);                                                  \
    CTYPE rt2 = MAKE_FN(0, 0);                                                  \
    CTYPE evscal = MAKE_FN(0, 0);                                               \
    CTYPE cs1 = MAKE_FN(0, 0);                                                  \
    CTYPE sn1 = MAKE_FN(0, 0);                                                  \
    memset(&vtable, 0, sizeof(vtable));                                         \
    memset(&g_##SUFFIX##_fortran_call, 0, sizeof(g_##SUFFIX##_fortran_call));   \
    vtable.ext_ops[OP_ID][FB_CONV_FORTRAN] =                                    \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_fortran;                 \
    fb_install_conv_thunks(&vtable, OP_ID);                                     \
    thunk = (fb_##SUFFIX##_cblas_fn)vtable.ext_ops[OP_ID][FB_CONV_CBLAS];       \
    if (!thunk) {                                                               \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk was not installed\n"); \
        return 1;                                                               \
    }                                                                           \
    if (thunk(MAKE_FN((BASE) + 21, (BASE) + 31),                               \
              MAKE_FN((BASE) + 22, (BASE) + 32),                               \
              MAKE_FN((BASE) + 23, (BASE) + 33),                               \
              &rt1, &rt2, &evscal, &cs1, &sn1) != 0 ||                         \
        g_##SUFFIX##_fortran_call.called != 1 ||                                \
        !EQ_FN(g_##SUFFIX##_fortran_call.a,                                     \
               MAKE_FN((BASE) + 21, (BASE) + 31)) ||                           \
        !EQ_FN(g_##SUFFIX##_fortran_call.b,                                     \
               MAKE_FN((BASE) + 22, (BASE) + 32)) ||                           \
        !EQ_FN(g_##SUFFIX##_fortran_call.c,                                     \
               MAKE_FN((BASE) + 23, (BASE) + 33)) ||                           \
        g_##SUFFIX##_fortran_call.rt1 != &rt1 ||                                \
        g_##SUFFIX##_fortran_call.rt2 != &rt2 ||                                \
        g_##SUFFIX##_fortran_call.evscal != &evscal ||                          \
        g_##SUFFIX##_fortran_call.cs1 != &cs1 ||                                \
        g_##SUFFIX##_fortran_call.sn1 != &sn1 ||                                \
        !EQ_FN(rt1, MAKE_FN((BASE) + 1, (BASE) + 11)) ||                       \
        !EQ_FN(rt2, MAKE_FN((BASE) + 2, (BASE) + 12)) ||                       \
        !EQ_FN(evscal, MAKE_FN((BASE) + 3, (BASE) + 13)) ||                    \
        !EQ_FN(cs1, MAKE_FN((BASE) + 4, (BASE) + 14)) ||                       \
        !EQ_FN(sn1, MAKE_FN((BASE) + 5, (BASE) + 15))) {                       \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward complex inputs or output pointers correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards complex scalars and returns success\n"); \
    return 0;                                                                   \
}                                                                               \
static int check_##SUFFIX##_cblas_to_fortran(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                      \
    CTYPE a = MAKE_FN((BASE) + 41, (BASE) + 51);                               \
    CTYPE b = MAKE_FN((BASE) + 42, (BASE) + 52);                               \
    CTYPE c = MAKE_FN((BASE) + 43, (BASE) + 53);                               \
    CTYPE rt1 = MAKE_FN(0, 0);                                                  \
    CTYPE rt2 = MAKE_FN(0, 0);                                                  \
    CTYPE evscal = MAKE_FN(0, 0);                                               \
    CTYPE cs1 = MAKE_FN(0, 0);                                                  \
    CTYPE sn1 = MAKE_FN(0, 0);                                                  \
    memset(&vtable, 0, sizeof(vtable));                                         \
    memset(&g_##SUFFIX##_cblas_call, 0, sizeof(g_##SUFFIX##_cblas_call));       \
    vtable.ext_ops[OP_ID][FB_CONV_CBLAS] =                                      \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_cblas;                   \
    fb_install_conv_thunks(&vtable, OP_ID);                                     \
    thunk = (fb_##SUFFIX##_fortran_fn)vtable.ext_ops[OP_ID][FB_CONV_FORTRAN];   \
    if (!thunk) {                                                               \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk was not installed\n"); \
        return 1;                                                               \
    }                                                                           \
    thunk(&a, &b, &c, &rt1, &rt2, &evscal, &cs1, &sn1);                        \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                  \
        !EQ_FN(g_##SUFFIX##_cblas_call.a, a) ||                                 \
        !EQ_FN(g_##SUFFIX##_cblas_call.b, b) ||                                 \
        !EQ_FN(g_##SUFFIX##_cblas_call.c, c) ||                                 \
        g_##SUFFIX##_cblas_call.rt1 != &rt1 ||                                  \
        g_##SUFFIX##_cblas_call.rt2 != &rt2 ||                                  \
        g_##SUFFIX##_cblas_call.evscal != &evscal ||                            \
        g_##SUFFIX##_cblas_call.cs1 != &cs1 ||                                  \
        g_##SUFFIX##_cblas_call.sn1 != &sn1 ||                                  \
        !EQ_FN(rt1, MAKE_FN((BASE) + 6, (BASE) + 16)) ||                       \
        !EQ_FN(rt2, MAKE_FN((BASE) + 7, (BASE) + 17)) ||                       \
        !EQ_FN(evscal, MAKE_FN((BASE) + 8, (BASE) + 18)) ||                    \
        !EQ_FN(cs1, MAKE_FN((BASE) + 9, (BASE) + 19)) ||                       \
        !EQ_FN(sn1, MAKE_FN((BASE) + 10, (BASE) + 20))) {                      \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference complex inputs or propagate outputs correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences complex inputs and ignores the C int return\n"); \
    return 0;                                                                   \
}

DEFINE_LAESY_TESTS(claesy, fb_complex_float_t, FB_OP_CLAESY,
                   make_cf32, cf32_eq, 100)
DEFINE_LAESY_TESTS(zlaesy, fb_complex_double_t, FB_OP_ZLAESY,
                   make_cf64, cf64_eq, 300)

int main(void)
{
    int status = 0;

    status |= check_claesy_fortran_to_cblas();
    status |= check_claesy_cblas_to_fortran();
    status |= check_zlaesy_fortran_to_cblas();
    status |= check_zlaesy_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}