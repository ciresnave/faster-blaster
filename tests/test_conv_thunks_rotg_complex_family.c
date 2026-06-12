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

static float cfloat_real(fb_complex_float_t value)
{
    return __real__ value;
}

static float cfloat_imag(fb_complex_float_t value)
{
    return __imag__ value;
}

static fb_complex_double_t make_cdouble(double real_value, double imag_value)
{
    fb_complex_double_t value = (fb_complex_double_t)0;
    __real__ value = real_value;
    __imag__ value = imag_value;
    return value;
}

static double cdouble_real(fb_complex_double_t value)
{
    return __real__ value;
}

static double cdouble_imag(fb_complex_double_t value)
{
    return __imag__ value;
}

#define DEFINE_COMPLEX_ROTG_TESTS(SUFFIX, CTYPE, REAL_TYPE, OP_ID, BASE, MAKE, REAL_PART, IMAG_PART) \
typedef void (*fb_##SUFFIX##_slot_fn)(CTYPE *a, CTYPE *b, REAL_TYPE *c, CTYPE *s); \
static struct {                                                                   \
    int called;                                                                   \
    CTYPE *a;                                                                     \
    CTYPE *b;                                                                     \
    REAL_TYPE *c;                                                                 \
    CTYPE *s;                                                                     \
} g_##SUFFIX##_fortran_call;                                                      \
static struct {                                                                   \
    int called;                                                                   \
    CTYPE *a;                                                                     \
    CTYPE *b;                                                                     \
    REAL_TYPE *c;                                                                 \
    CTYPE *s;                                                                     \
} g_##SUFFIX##_cblas_call;                                                        \
static void stub_##SUFFIX##_fortran(CTYPE *a, CTYPE *b, REAL_TYPE *c, CTYPE *s) \
{                                                                                 \
    g_##SUFFIX##_fortran_call.called += 1;                                        \
    g_##SUFFIX##_fortran_call.a = a;                                              \
    g_##SUFFIX##_fortran_call.b = b;                                              \
    g_##SUFFIX##_fortran_call.c = c;                                              \
    g_##SUFFIX##_fortran_call.s = s;                                              \
    *a = MAKE((REAL_TYPE)((BASE) + 1), (REAL_TYPE)((BASE) + 2));                 \
    *c = (REAL_TYPE)((BASE) + 3);                                                 \
    *s = MAKE((REAL_TYPE)((BASE) + 4), (REAL_TYPE)((BASE) + 5));                 \
}                                                                                 \
static void stub_##SUFFIX##_cblas(CTYPE *a, CTYPE *b, REAL_TYPE *c, CTYPE *s)   \
{                                                                                 \
    g_##SUFFIX##_cblas_call.called += 1;                                          \
    g_##SUFFIX##_cblas_call.a = a;                                                \
    g_##SUFFIX##_cblas_call.b = b;                                                \
    g_##SUFFIX##_cblas_call.c = c;                                                \
    g_##SUFFIX##_cblas_call.s = s;                                                \
    *a = MAKE((REAL_TYPE)((BASE) + 6), (REAL_TYPE)((BASE) + 7));                 \
    *c = (REAL_TYPE)((BASE) + 8);                                                 \
    *s = MAKE((REAL_TYPE)((BASE) + 9), (REAL_TYPE)((BASE) + 10));                \
}                                                                                 \
static int check_##SUFFIX##_fortran_to_cblas(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_slot_fn thunk = NULL;                                           \
    CTYPE a = MAKE((REAL_TYPE)1, (REAL_TYPE)10);                                 \
    CTYPE b = MAKE((REAL_TYPE)2, (REAL_TYPE)20);                                 \
    REAL_TYPE c = (REAL_TYPE)0;                                                   \
    CTYPE s = MAKE((REAL_TYPE)0, (REAL_TYPE)0);                                  \
    memset(&vtable, 0, sizeof(vtable));                                           \
    memset(&g_##SUFFIX##_fortran_call, 0, sizeof(g_##SUFFIX##_fortran_call));     \
    vtable.ext_ops[OP_ID][FB_CONV_FORTRAN] =                                      \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_fortran;                   \
    fb_install_conv_thunks(&vtable, OP_ID);                                       \
    thunk = (fb_##SUFFIX##_slot_fn)vtable.ext_ops[OP_ID][FB_CONV_CBLAS];         \
    if (!thunk) {                                                                 \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk was not installed\n"); \
        return 1;                                                                 \
    }                                                                             \
    thunk(&a, &b, &c, &s);                                                        \
    if (g_##SUFFIX##_fortran_call.called != 1 ||                                  \
        g_##SUFFIX##_fortran_call.a != &a ||                                      \
        g_##SUFFIX##_fortran_call.b != &b ||                                      \
        g_##SUFFIX##_fortran_call.c != &c ||                                      \
        g_##SUFFIX##_fortran_call.s != &s ||                                      \
        REAL_PART(a) != (REAL_TYPE)((BASE) + 1) ||                               \
        IMAG_PART(a) != (REAL_TYPE)((BASE) + 2) ||                               \
        c != (REAL_TYPE)((BASE) + 3) ||                                           \
        REAL_PART(s) != (REAL_TYPE)((BASE) + 4) ||                               \
        IMAG_PART(s) != (REAL_TYPE)((BASE) + 5)) {                               \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward complex ROTG scalars correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards complex ROTG scalars\n"); \
    return 0;                                                                     \
}                                                                                 \
static int check_##SUFFIX##_cblas_to_fortran(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_slot_fn thunk = NULL;                                           \
    CTYPE a = MAKE((REAL_TYPE)1, (REAL_TYPE)10);                                 \
    CTYPE b = MAKE((REAL_TYPE)2, (REAL_TYPE)20);                                 \
    REAL_TYPE c = (REAL_TYPE)0;                                                   \
    CTYPE s = MAKE((REAL_TYPE)0, (REAL_TYPE)0);                                  \
    memset(&vtable, 0, sizeof(vtable));                                           \
    memset(&g_##SUFFIX##_cblas_call, 0, sizeof(g_##SUFFIX##_cblas_call));         \
    vtable.ext_ops[OP_ID][FB_CONV_CBLAS] =                                        \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_cblas;                     \
    fb_install_conv_thunks(&vtable, OP_ID);                                       \
    thunk = (fb_##SUFFIX##_slot_fn)vtable.ext_ops[OP_ID][FB_CONV_FORTRAN];       \
    if (!thunk) {                                                                 \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk was not installed\n"); \
        return 1;                                                                 \
    }                                                                             \
    thunk(&a, &b, &c, &s);                                                        \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                    \
        g_##SUFFIX##_cblas_call.a != &a ||                                        \
        g_##SUFFIX##_cblas_call.b != &b ||                                        \
        g_##SUFFIX##_cblas_call.c != &c ||                                        \
        g_##SUFFIX##_cblas_call.s != &s ||                                        \
        REAL_PART(a) != (REAL_TYPE)((BASE) + 6) ||                               \
        IMAG_PART(a) != (REAL_TYPE)((BASE) + 7) ||                               \
        c != (REAL_TYPE)((BASE) + 8) ||                                           \
        REAL_PART(s) != (REAL_TYPE)((BASE) + 9) ||                               \
        IMAG_PART(s) != (REAL_TYPE)((BASE) + 10)) {                              \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not forward complex ROTG scalars correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk forwards complex ROTG scalars\n"); \
    return 0;                                                                     \
}

DEFINE_COMPLEX_ROTG_TESTS(crotg, fb_complex_float_t, float, FB_OP_CROTG, 100, make_cfloat, cfloat_real, cfloat_imag)
DEFINE_COMPLEX_ROTG_TESTS(zrotg, fb_complex_double_t, double, FB_OP_ZROTG, 300, make_cdouble, cdouble_real, cdouble_imag)

int main(void)
{
    int status = 0;

    status |= check_crotg_fortran_to_cblas();
    status |= check_crotg_cblas_to_fortran();
    status |= check_zrotg_fortran_to_cblas();
    status |= check_zrotg_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}