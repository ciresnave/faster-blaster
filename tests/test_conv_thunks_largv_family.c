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

#define DEFINE_LARGV_COMPLEX_TESTS(SUFFIX, CTYPE, RTYPE, OP_ID, MAKE_FN, EQ_FN, RBASE) \
typedef int (*fb_##SUFFIX##_cblas_fn)(int n, CTYPE *x, int incx, CTYPE *y, int incy, \
                                      RTYPE *c, int incc);                              \
typedef void (*fb_##SUFFIX##_fortran_fn)(int *n, CTYPE *x, int *incx, CTYPE *y, int *incy, \
                                         RTYPE *c, int *incc);                          \
static struct {                                                                          \
    int called;                                                                          \
    int n;                                                                               \
    CTYPE *x;                                                                            \
    int incx;                                                                            \
    CTYPE *y;                                                                            \
    int incy;                                                                            \
    RTYPE *c;                                                                            \
    int incc;                                                                            \
} g_##SUFFIX##_fortran_call;                                                             \
static struct {                                                                          \
    int called;                                                                          \
    int n;                                                                               \
    CTYPE *x;                                                                            \
    int incx;                                                                            \
    CTYPE *y;                                                                            \
    int incy;                                                                            \
    RTYPE *c;                                                                            \
    int incc;                                                                            \
} g_##SUFFIX##_cblas_call;                                                               \
static void stub_##SUFFIX##_fortran(int *n, CTYPE *x, int *incx, CTYPE *y, int *incy,   \
                                    RTYPE *c, int *incc)                                \
{                                                                                        \
    g_##SUFFIX##_fortran_call.called += 1;                                               \
    g_##SUFFIX##_fortran_call.n = *n;                                                    \
    g_##SUFFIX##_fortran_call.x = x;                                                     \
    g_##SUFFIX##_fortran_call.incx = *incx;                                              \
    g_##SUFFIX##_fortran_call.y = y;                                                     \
    g_##SUFFIX##_fortran_call.incy = *incy;                                              \
    g_##SUFFIX##_fortran_call.c = c;                                                     \
    g_##SUFFIX##_fortran_call.incc = *incc;                                              \
    x[0] = MAKE_FN((RBASE) + 1, (RBASE) + 11);                                           \
    y[0] = MAKE_FN((RBASE) + 2, (RBASE) + 12);                                           \
    c[0] = (RTYPE)((RBASE) + 21);                                                        \
    if (*n > 1) {                                                                        \
        x[(size_t)(*incx)] = MAKE_FN((RBASE) + 3, (RBASE) + 13);                         \
        y[(size_t)(*incy)] = MAKE_FN((RBASE) + 4, (RBASE) + 14);                         \
        c[(size_t)(*incc)] = (RTYPE)((RBASE) + 22);                                      \
    }                                                                                    \
}                                                                                        \
static int stub_##SUFFIX##_cblas(int n, CTYPE *x, int incx, CTYPE *y, int incy,         \
                                 RTYPE *c, int incc)                                     \
{                                                                                        \
    g_##SUFFIX##_cblas_call.called += 1;                                                 \
    g_##SUFFIX##_cblas_call.n = n;                                                       \
    g_##SUFFIX##_cblas_call.x = x;                                                       \
    g_##SUFFIX##_cblas_call.incx = incx;                                                 \
    g_##SUFFIX##_cblas_call.y = y;                                                       \
    g_##SUFFIX##_cblas_call.incy = incy;                                                 \
    g_##SUFFIX##_cblas_call.c = c;                                                       \
    g_##SUFFIX##_cblas_call.incc = incc;                                                 \
    x[0] = MAKE_FN((RBASE) + 31, (RBASE) + 41);                                          \
    y[0] = MAKE_FN((RBASE) + 32, (RBASE) + 42);                                          \
    c[0] = (RTYPE)((RBASE) + 51);                                                        \
    if (n > 1) {                                                                         \
        x[(size_t)incx] = MAKE_FN((RBASE) + 33, (RBASE) + 43);                           \
        y[(size_t)incy] = MAKE_FN((RBASE) + 34, (RBASE) + 44);                           \
        c[(size_t)incc] = (RTYPE)((RBASE) + 52);                                         \
    }                                                                                    \
    return 0;                                                                            \
}                                                                                        \
static int check_##SUFFIX##_fortran_to_cblas(void)                                       \
{                                                                                        \
    fb_backend_vtable_t vtable;                                                          \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                                 \
    CTYPE x[3] = { MAKE_FN(1, 2), MAKE_FN(99, 98), MAKE_FN(3, 4) };                      \
    CTYPE y[3] = { MAKE_FN(5, 6), MAKE_FN(97, 96), MAKE_FN(7, 8) };                      \
    RTYPE c[3] = { (RTYPE)9, (RTYPE)95, (RTYPE)10 };                                     \
    memset(&vtable, 0, sizeof(vtable));                                                  \
    memset(&g_##SUFFIX##_fortran_call, 0, sizeof(g_##SUFFIX##_fortran_call));            \
    vtable.ext_ops[OP_ID][FB_CONV_FORTRAN] =                                             \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_fortran;                          \
    fb_install_conv_thunks(&vtable, OP_ID);                                              \
    thunk = (fb_##SUFFIX##_cblas_fn)vtable.ext_ops[OP_ID][FB_CONV_CBLAS];                \
    if (!thunk) {                                                                        \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk was not installed\n"); \
        return 1;                                                                        \
    }                                                                                    \
    if (thunk(2, x, 2, y, 2, c, 2) != 0 ||                                               \
        g_##SUFFIX##_fortran_call.called != 1 ||                                         \
        g_##SUFFIX##_fortran_call.n != 2 ||                                              \
        g_##SUFFIX##_fortran_call.x != x ||                                              \
        g_##SUFFIX##_fortran_call.incx != 2 ||                                           \
        g_##SUFFIX##_fortran_call.y != y ||                                              \
        g_##SUFFIX##_fortran_call.incy != 2 ||                                           \
        g_##SUFFIX##_fortran_call.c != c ||                                              \
        g_##SUFFIX##_fortran_call.incc != 2 ||                                           \
        !EQ_FN(x[0], MAKE_FN((RBASE) + 1, (RBASE) + 11)) ||                              \
        !EQ_FN(y[0], MAKE_FN((RBASE) + 2, (RBASE) + 12)) ||                              \
        !EQ_FN(x[2], MAKE_FN((RBASE) + 3, (RBASE) + 13)) ||                              \
        !EQ_FN(y[2], MAKE_FN((RBASE) + 4, (RBASE) + 14)) ||                              \
        c[0] != (RTYPE)((RBASE) + 21) ||                                                 \
        c[2] != (RTYPE)((RBASE) + 22) ||                                                 \
        !EQ_FN(x[1], MAKE_FN(99, 98)) ||                                                 \
        !EQ_FN(y[1], MAKE_FN(97, 96)) ||                                                 \
        c[1] != (RTYPE)95) {                                                             \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward complex LARGV inputs correctly\n"); \
        return 1;                                                                        \
    }                                                                                    \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards complex LARGV inputs\n"); \
    return 0;                                                                            \
}                                                                                        \
static int check_##SUFFIX##_cblas_to_fortran(void)                                       \
{                                                                                        \
    fb_backend_vtable_t vtable;                                                          \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                               \
    int n = 2;                                                                           \
    int incx = 2;                                                                        \
    int incy = 2;                                                                        \
    int incc = 2;                                                                        \
    CTYPE x[3] = { MAKE_FN(11, 12), MAKE_FN(91, 92), MAKE_FN(13, 14) };                  \
    CTYPE y[3] = { MAKE_FN(15, 16), MAKE_FN(93, 94), MAKE_FN(17, 18) };                  \
    RTYPE c[3] = { (RTYPE)19, (RTYPE)89, (RTYPE)20 };                                    \
    memset(&vtable, 0, sizeof(vtable));                                                  \
    memset(&g_##SUFFIX##_cblas_call, 0, sizeof(g_##SUFFIX##_cblas_call));                \
    vtable.ext_ops[OP_ID][FB_CONV_CBLAS] =                                               \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_cblas;                            \
    fb_install_conv_thunks(&vtable, OP_ID);                                              \
    thunk = (fb_##SUFFIX##_fortran_fn)vtable.ext_ops[OP_ID][FB_CONV_FORTRAN];            \
    if (!thunk) {                                                                        \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk was not installed\n"); \
        return 1;                                                                        \
    }                                                                                    \
    thunk(&n, x, &incx, y, &incy, c, &incc);                                             \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                           \
        g_##SUFFIX##_cblas_call.n != 2 ||                                                \
        g_##SUFFIX##_cblas_call.x != x ||                                                \
        g_##SUFFIX##_cblas_call.incx != 2 ||                                             \
        g_##SUFFIX##_cblas_call.y != y ||                                                \
        g_##SUFFIX##_cblas_call.incy != 2 ||                                             \
        g_##SUFFIX##_cblas_call.c != c ||                                                \
        g_##SUFFIX##_cblas_call.incc != 2 ||                                             \
        !EQ_FN(x[0], MAKE_FN((RBASE) + 31, (RBASE) + 41)) ||                             \
        !EQ_FN(y[0], MAKE_FN((RBASE) + 32, (RBASE) + 42)) ||                             \
        !EQ_FN(x[2], MAKE_FN((RBASE) + 33, (RBASE) + 43)) ||                             \
        !EQ_FN(y[2], MAKE_FN((RBASE) + 34, (RBASE) + 44)) ||                             \
        c[0] != (RTYPE)((RBASE) + 51) ||                                                 \
        c[2] != (RTYPE)((RBASE) + 52) ||                                                 \
        !EQ_FN(x[1], MAKE_FN(91, 92)) ||                                                 \
        !EQ_FN(y[1], MAKE_FN(93, 94)) ||                                                 \
        c[1] != (RTYPE)89) {                                                             \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference complex LARGV inputs correctly\n"); \
        return 1;                                                                        \
    }                                                                                    \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences complex LARGV inputs\n"); \
    return 0;                                                                            \
}

DEFINE_LARGV_COMPLEX_TESTS(clargv, fb_complex_float_t, float, FB_OP_CLARGV,
                           make_cf32, cf32_eq, 100)
DEFINE_LARGV_COMPLEX_TESTS(zlargv, fb_complex_double_t, double, FB_OP_ZLARGV,
                           make_cf64, cf64_eq, 300)

int main(void)
{
    int status = 0;

    status |= check_clargv_fortran_to_cblas();
    status |= check_clargv_cblas_to_fortran();
    status |= check_zlargv_fortran_to_cblas();
    status |= check_zlargv_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}