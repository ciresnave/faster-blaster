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

#define DEFINE_LARTV_REAL_TESTS(SUFFIX, TYPE, OP_ID, BASE)                     \
typedef int (*fb_##SUFFIX##_cblas_fn)(int n, TYPE *x, int incx, TYPE *y, int incy, \
                                      TYPE *c, TYPE *s, int incc);                   \
typedef void (*fb_##SUFFIX##_fortran_fn)(int *n, TYPE *x, int *incx, TYPE *y, int *incy, \
                                         TYPE *c, TYPE *s, int *incc);                \
static struct {                                                                       \
    int called;                                                                       \
    int n;                                                                            \
    TYPE *x;                                                                          \
    int incx;                                                                         \
    TYPE *y;                                                                          \
    int incy;                                                                         \
    TYPE *c;                                                                          \
    TYPE *s;                                                                          \
    int incc;                                                                         \
} g_##SUFFIX##_fortran_call;                                                          \
static struct {                                                                       \
    int called;                                                                       \
    int n;                                                                            \
    TYPE *x;                                                                          \
    int incx;                                                                         \
    TYPE *y;                                                                          \
    int incy;                                                                         \
    TYPE *c;                                                                          \
    TYPE *s;                                                                          \
    int incc;                                                                         \
} g_##SUFFIX##_cblas_call;                                                            \
static void stub_##SUFFIX##_fortran(int *n, TYPE *x, int *incx, TYPE *y, int *incy, \
                                    TYPE *c, TYPE *s, int *incc)                     \
{                                                                                     \
    g_##SUFFIX##_fortran_call.called += 1;                                            \
    g_##SUFFIX##_fortran_call.n = *n;                                                 \
    g_##SUFFIX##_fortran_call.x = x;                                                  \
    g_##SUFFIX##_fortran_call.incx = *incx;                                           \
    g_##SUFFIX##_fortran_call.y = y;                                                  \
    g_##SUFFIX##_fortran_call.incy = *incy;                                           \
    g_##SUFFIX##_fortran_call.c = c;                                                  \
    g_##SUFFIX##_fortran_call.s = s;                                                  \
    g_##SUFFIX##_fortran_call.incc = *incc;                                           \
    x[0] = (TYPE)((BASE) + 1);                                                        \
    y[0] = (TYPE)((BASE) + 2);                                                        \
    if (*n > 1) {                                                                     \
        x[(size_t)(*incx)] = (TYPE)((BASE) + 3);                                      \
        y[(size_t)(*incy)] = (TYPE)((BASE) + 4);                                      \
    }                                                                                 \
}                                                                                     \
static int stub_##SUFFIX##_cblas(int n, TYPE *x, int incx, TYPE *y, int incy, TYPE *c, \
                                 TYPE *s, int incc)                                   \
{                                                                                     \
    g_##SUFFIX##_cblas_call.called += 1;                                              \
    g_##SUFFIX##_cblas_call.n = n;                                                    \
    g_##SUFFIX##_cblas_call.x = x;                                                    \
    g_##SUFFIX##_cblas_call.incx = incx;                                              \
    g_##SUFFIX##_cblas_call.y = y;                                                    \
    g_##SUFFIX##_cblas_call.incy = incy;                                              \
    g_##SUFFIX##_cblas_call.c = c;                                                    \
    g_##SUFFIX##_cblas_call.s = s;                                                    \
    g_##SUFFIX##_cblas_call.incc = incc;                                              \
    x[0] = (TYPE)((BASE) + 11);                                                       \
    y[0] = (TYPE)((BASE) + 12);                                                       \
    if (n > 1) {                                                                      \
        x[(size_t)incx] = (TYPE)((BASE) + 13);                                        \
        y[(size_t)incy] = (TYPE)((BASE) + 14);                                        \
    }                                                                                 \
    return 0;                                                                         \
}                                                                                     \
static int check_##SUFFIX##_fortran_to_cblas(void)                                    \
{                                                                                     \
    fb_backend_vtable_t vtable;                                                       \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                              \
    TYPE x[3] = { (TYPE)1, (TYPE)99, (TYPE)2 };                                       \
    TYPE y[3] = { (TYPE)3, (TYPE)98, (TYPE)4 };                                       \
    TYPE c[3] = { (TYPE)5, (TYPE)97, (TYPE)6 };                                       \
    TYPE s[3] = { (TYPE)7, (TYPE)96, (TYPE)8 };                                       \
    memset(&vtable, 0, sizeof(vtable));                                               \
    memset(&g_##SUFFIX##_fortran_call, 0, sizeof(g_##SUFFIX##_fortran_call));         \
    vtable.ext_ops[OP_ID][FB_CONV_FORTRAN] =                                          \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_fortran;                       \
    fb_install_conv_thunks(&vtable, OP_ID);                                           \
    thunk = (fb_##SUFFIX##_cblas_fn)vtable.ext_ops[OP_ID][FB_CONV_CBLAS];             \
    if (!thunk) {                                                                     \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk was not installed\n"); \
        return 1;                                                                     \
    }                                                                                 \
    if (thunk(2, x, 2, y, 2, c, s, 2) != 0 ||                                         \
        g_##SUFFIX##_fortran_call.called != 1 ||                                      \
        g_##SUFFIX##_fortran_call.n != 2 ||                                           \
        g_##SUFFIX##_fortran_call.x != x ||                                           \
        g_##SUFFIX##_fortran_call.incx != 2 ||                                        \
        g_##SUFFIX##_fortran_call.y != y ||                                           \
        g_##SUFFIX##_fortran_call.incy != 2 ||                                        \
        g_##SUFFIX##_fortran_call.c != c ||                                           \
        g_##SUFFIX##_fortran_call.s != s ||                                           \
        g_##SUFFIX##_fortran_call.incc != 2 ||                                        \
        x[0] != (TYPE)((BASE) + 1) || y[0] != (TYPE)((BASE) + 2) ||                   \
        x[2] != (TYPE)((BASE) + 3) || y[2] != (TYPE)((BASE) + 4) ||                   \
        x[1] != (TYPE)99 || y[1] != (TYPE)98 ||                                       \
        c[0] != (TYPE)5 || c[2] != (TYPE)6 || c[1] != (TYPE)97 ||                     \
        s[0] != (TYPE)7 || s[2] != (TYPE)8 || s[1] != (TYPE)96) {                     \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward LARTV inputs correctly\n"); \
        return 1;                                                                     \
    }                                                                                 \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards LARTV inputs\n");   \
    return 0;                                                                         \
}                                                                                     \
static int check_##SUFFIX##_cblas_to_fortran(void)                                    \
{                                                                                     \
    fb_backend_vtable_t vtable;                                                       \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                            \
    int n = 2;                                                                        \
    int incx = 2;                                                                     \
    int incy = 2;                                                                     \
    int incc = 2;                                                                     \
    TYPE x[3] = { (TYPE)11, (TYPE)95, (TYPE)12 };                                     \
    TYPE y[3] = { (TYPE)13, (TYPE)94, (TYPE)14 };                                     \
    TYPE c[3] = { (TYPE)15, (TYPE)93, (TYPE)16 };                                     \
    TYPE s[3] = { (TYPE)17, (TYPE)92, (TYPE)18 };                                     \
    memset(&vtable, 0, sizeof(vtable));                                               \
    memset(&g_##SUFFIX##_cblas_call, 0, sizeof(g_##SUFFIX##_cblas_call));             \
    vtable.ext_ops[OP_ID][FB_CONV_CBLAS] =                                            \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_cblas;                         \
    fb_install_conv_thunks(&vtable, OP_ID);                                           \
    thunk = (fb_##SUFFIX##_fortran_fn)vtable.ext_ops[OP_ID][FB_CONV_FORTRAN];         \
    if (!thunk) {                                                                     \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk was not installed\n"); \
        return 1;                                                                     \
    }                                                                                 \
    thunk(&n, x, &incx, y, &incy, c, s, &incc);                                       \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                        \
        g_##SUFFIX##_cblas_call.n != 2 ||                                             \
        g_##SUFFIX##_cblas_call.x != x ||                                             \
        g_##SUFFIX##_cblas_call.incx != 2 ||                                          \
        g_##SUFFIX##_cblas_call.y != y ||                                             \
        g_##SUFFIX##_cblas_call.incy != 2 ||                                          \
        g_##SUFFIX##_cblas_call.c != c ||                                             \
        g_##SUFFIX##_cblas_call.s != s ||                                             \
        g_##SUFFIX##_cblas_call.incc != 2 ||                                          \
        x[0] != (TYPE)((BASE) + 11) || y[0] != (TYPE)((BASE) + 12) ||                 \
        x[2] != (TYPE)((BASE) + 13) || y[2] != (TYPE)((BASE) + 14) ||                 \
        x[1] != (TYPE)95 || y[1] != (TYPE)94 ||                                       \
        c[0] != (TYPE)15 || c[2] != (TYPE)16 || c[1] != (TYPE)93 ||                   \
        s[0] != (TYPE)17 || s[2] != (TYPE)18 || s[1] != (TYPE)92) {                   \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference LARTV inputs correctly\n"); \
        return 1;                                                                     \
    }                                                                                 \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences LARTV inputs\n"); \
    return 0;                                                                         \
}

#define DEFINE_LARTV_COMPLEX_TESTS(SUFFIX, CTYPE, RTYPE, OP_ID, MAKE_FN, EQ_FN, RBASE) \
typedef int (*fb_##SUFFIX##_cblas_fn)(int n, CTYPE *x, int incx, CTYPE *y, int incy,    \
                                      RTYPE *c, CTYPE *s, int incc);                      \
typedef void (*fb_##SUFFIX##_fortran_fn)(int *n, CTYPE *x, int *incx, CTYPE *y, int *incy, \
                                         RTYPE *c, CTYPE *s, int *incc);                  \
static struct {                                                                            \
    int called;                                                                            \
    int n;                                                                                 \
    CTYPE *x;                                                                              \
    int incx;                                                                              \
    CTYPE *y;                                                                              \
    int incy;                                                                              \
    RTYPE *c;                                                                              \
    CTYPE *s;                                                                              \
    int incc;                                                                              \
} g_##SUFFIX##_fortran_call;                                                               \
static struct {                                                                            \
    int called;                                                                            \
    int n;                                                                                 \
    CTYPE *x;                                                                              \
    int incx;                                                                              \
    CTYPE *y;                                                                              \
    int incy;                                                                              \
    RTYPE *c;                                                                              \
    CTYPE *s;                                                                              \
    int incc;                                                                              \
} g_##SUFFIX##_cblas_call;                                                                 \
static void stub_##SUFFIX##_fortran(int *n, CTYPE *x, int *incx, CTYPE *y, int *incy,     \
                                    RTYPE *c, CTYPE *s, int *incc)                        \
{                                                                                          \
    g_##SUFFIX##_fortran_call.called += 1;                                                 \
    g_##SUFFIX##_fortran_call.n = *n;                                                      \
    g_##SUFFIX##_fortran_call.x = x;                                                       \
    g_##SUFFIX##_fortran_call.incx = *incx;                                                \
    g_##SUFFIX##_fortran_call.y = y;                                                       \
    g_##SUFFIX##_fortran_call.incy = *incy;                                                \
    g_##SUFFIX##_fortran_call.c = c;                                                       \
    g_##SUFFIX##_fortran_call.s = s;                                                       \
    g_##SUFFIX##_fortran_call.incc = *incc;                                                \
    x[0] = MAKE_FN((RBASE) + 1, (RBASE) + 11);                                             \
    y[0] = MAKE_FN((RBASE) + 2, (RBASE) + 12);                                             \
    if (*n > 1) {                                                                          \
        x[(size_t)(*incx)] = MAKE_FN((RBASE) + 3, (RBASE) + 13);                           \
        y[(size_t)(*incy)] = MAKE_FN((RBASE) + 4, (RBASE) + 14);                           \
    }                                                                                      \
}                                                                                          \
static int stub_##SUFFIX##_cblas(int n, CTYPE *x, int incx, CTYPE *y, int incy, RTYPE *c, \
                                 CTYPE *s, int incc)                                       \
{                                                                                          \
    g_##SUFFIX##_cblas_call.called += 1;                                                   \
    g_##SUFFIX##_cblas_call.n = n;                                                         \
    g_##SUFFIX##_cblas_call.x = x;                                                         \
    g_##SUFFIX##_cblas_call.incx = incx;                                                   \
    g_##SUFFIX##_cblas_call.y = y;                                                         \
    g_##SUFFIX##_cblas_call.incy = incy;                                                   \
    g_##SUFFIX##_cblas_call.c = c;                                                         \
    g_##SUFFIX##_cblas_call.s = s;                                                         \
    g_##SUFFIX##_cblas_call.incc = incc;                                                   \
    x[0] = MAKE_FN((RBASE) + 21, (RBASE) + 31);                                            \
    y[0] = MAKE_FN((RBASE) + 22, (RBASE) + 32);                                            \
    if (n > 1) {                                                                           \
        x[(size_t)incx] = MAKE_FN((RBASE) + 23, (RBASE) + 33);                             \
        y[(size_t)incy] = MAKE_FN((RBASE) + 24, (RBASE) + 34);                             \
    }                                                                                      \
    return 0;                                                                              \
}                                                                                          \
static int check_##SUFFIX##_fortran_to_cblas(void)                                         \
{                                                                                          \
    fb_backend_vtable_t vtable;                                                            \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                                   \
    CTYPE x[3] = { MAKE_FN(1, 2), MAKE_FN(99, 98), MAKE_FN(3, 4) };                        \
    CTYPE y[3] = { MAKE_FN(5, 6), MAKE_FN(97, 96), MAKE_FN(7, 8) };                        \
    RTYPE c[3] = { (RTYPE)9, (RTYPE)95, (RTYPE)10 };                                       \
    CTYPE s[3] = { MAKE_FN(11, 12), MAKE_FN(93, 92), MAKE_FN(13, 14) };                    \
    memset(&vtable, 0, sizeof(vtable));                                                    \
    memset(&g_##SUFFIX##_fortran_call, 0, sizeof(g_##SUFFIX##_fortran_call));              \
    vtable.ext_ops[OP_ID][FB_CONV_FORTRAN] =                                               \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_fortran;                            \
    fb_install_conv_thunks(&vtable, OP_ID);                                                \
    thunk = (fb_##SUFFIX##_cblas_fn)vtable.ext_ops[OP_ID][FB_CONV_CBLAS];                  \
    if (!thunk) {                                                                          \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk was not installed\n"); \
        return 1;                                                                          \
    }                                                                                      \
    if (thunk(2, x, 2, y, 2, c, s, 2) != 0 ||                                              \
        g_##SUFFIX##_fortran_call.called != 1 ||                                           \
        g_##SUFFIX##_fortran_call.n != 2 ||                                                \
        g_##SUFFIX##_fortran_call.x != x ||                                                \
        g_##SUFFIX##_fortran_call.incx != 2 ||                                             \
        g_##SUFFIX##_fortran_call.y != y ||                                                \
        g_##SUFFIX##_fortran_call.incy != 2 ||                                             \
        g_##SUFFIX##_fortran_call.c != c ||                                                \
        g_##SUFFIX##_fortran_call.s != s ||                                                \
        g_##SUFFIX##_fortran_call.incc != 2 ||                                             \
        !EQ_FN(x[0], MAKE_FN((RBASE) + 1, (RBASE) + 11)) ||                                \
        !EQ_FN(y[0], MAKE_FN((RBASE) + 2, (RBASE) + 12)) ||                                \
        !EQ_FN(x[2], MAKE_FN((RBASE) + 3, (RBASE) + 13)) ||                                \
        !EQ_FN(y[2], MAKE_FN((RBASE) + 4, (RBASE) + 14)) ||                                \
        !EQ_FN(x[1], MAKE_FN(99, 98)) ||                                                   \
        !EQ_FN(y[1], MAKE_FN(97, 96)) ||                                                   \
        c[0] != (RTYPE)9 || c[2] != (RTYPE)10 || c[1] != (RTYPE)95 ||                      \
        !EQ_FN(s[0], MAKE_FN(11, 12)) ||                                                   \
        !EQ_FN(s[2], MAKE_FN(13, 14)) ||                                                   \
        !EQ_FN(s[1], MAKE_FN(93, 92))) {                                                   \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward LARTV inputs correctly\n"); \
        return 1;                                                                          \
    }                                                                                      \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards LARTV inputs\n");         \
    return 0;                                                                              \
}                                                                                          \
static int check_##SUFFIX##_cblas_to_fortran(void)                                         \
{                                                                                          \
    fb_backend_vtable_t vtable;                                                            \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                                 \
    int n = 2;                                                                             \
    int incx = 2;                                                                          \
    int incy = 2;                                                                          \
    int incc = 2;                                                                          \
    CTYPE x[3] = { MAKE_FN(21, 22), MAKE_FN(91, 92), MAKE_FN(23, 24) };                   \
    CTYPE y[3] = { MAKE_FN(25, 26), MAKE_FN(89, 88), MAKE_FN(27, 28) };                   \
    RTYPE c[3] = { (RTYPE)29, (RTYPE)87, (RTYPE)30 };                                      \
    CTYPE s[3] = { MAKE_FN(31, 32), MAKE_FN(85, 84), MAKE_FN(33, 34) };                   \
    memset(&vtable, 0, sizeof(vtable));                                                    \
    memset(&g_##SUFFIX##_cblas_call, 0, sizeof(g_##SUFFIX##_cblas_call));                  \
    vtable.ext_ops[OP_ID][FB_CONV_CBLAS] =                                                 \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_cblas;                              \
    fb_install_conv_thunks(&vtable, OP_ID);                                                \
    thunk = (fb_##SUFFIX##_fortran_fn)vtable.ext_ops[OP_ID][FB_CONV_FORTRAN];              \
    if (!thunk) {                                                                          \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk was not installed\n"); \
        return 1;                                                                          \
    }                                                                                      \
    thunk(&n, x, &incx, y, &incy, c, s, &incc);                                            \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                             \
        g_##SUFFIX##_cblas_call.n != 2 ||                                                  \
        g_##SUFFIX##_cblas_call.x != x ||                                                  \
        g_##SUFFIX##_cblas_call.incx != 2 ||                                               \
        g_##SUFFIX##_cblas_call.y != y ||                                                  \
        g_##SUFFIX##_cblas_call.incy != 2 ||                                               \
        g_##SUFFIX##_cblas_call.c != c ||                                                  \
        g_##SUFFIX##_cblas_call.s != s ||                                                  \
        g_##SUFFIX##_cblas_call.incc != 2 ||                                               \
        !EQ_FN(x[0], MAKE_FN((RBASE) + 21, (RBASE) + 31)) ||                               \
        !EQ_FN(y[0], MAKE_FN((RBASE) + 22, (RBASE) + 32)) ||                               \
        !EQ_FN(x[2], MAKE_FN((RBASE) + 23, (RBASE) + 33)) ||                               \
        !EQ_FN(y[2], MAKE_FN((RBASE) + 24, (RBASE) + 34)) ||                               \
        !EQ_FN(x[1], MAKE_FN(91, 92)) ||                                                   \
        !EQ_FN(y[1], MAKE_FN(89, 88)) ||                                                   \
        c[0] != (RTYPE)29 || c[2] != (RTYPE)30 || c[1] != (RTYPE)87 ||                     \
        !EQ_FN(s[0], MAKE_FN(31, 32)) ||                                                   \
        !EQ_FN(s[2], MAKE_FN(33, 34)) ||                                                   \
        !EQ_FN(s[1], MAKE_FN(85, 84))) {                                                   \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference LARTV inputs correctly\n"); \
        return 1;                                                                          \
    }                                                                                      \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences LARTV inputs\n");     \
    return 0;                                                                              \
}

DEFINE_LARTV_REAL_TESTS(slartv, float, FB_OP_SLARTV, 100)
DEFINE_LARTV_REAL_TESTS(dlartv, double, FB_OP_DLARTV, 300)
DEFINE_LARTV_COMPLEX_TESTS(clartv, fb_complex_float_t, float, FB_OP_CLARTV, make_cf32, cf32_eq, 500)
DEFINE_LARTV_COMPLEX_TESTS(zlartv, fb_complex_double_t, double, FB_OP_ZLARTV, make_cf64, cf64_eq, 700)

int main(void)
{
    int status = 0;

    status |= check_slartv_fortran_to_cblas();
    status |= check_slartv_cblas_to_fortran();
    status |= check_dlartv_fortran_to_cblas();
    status |= check_dlartv_cblas_to_fortran();
    status |= check_clartv_fortran_to_cblas();
    status |= check_clartv_cblas_to_fortran();
    status |= check_zlartv_fortran_to_cblas();
    status |= check_zlartv_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}