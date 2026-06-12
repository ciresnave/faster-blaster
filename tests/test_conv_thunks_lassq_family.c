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

#define DEFINE_REAL_LASSQ_TESTS(SUFFIX, TYPE, OP_ID, BASE)                     \
typedef int (*fb_##SUFFIX##_cblas_fn)(int n, TYPE *x, int incx,                \
                                      TYPE *scale, TYPE *sumsq);               \
typedef void (*fb_##SUFFIX##_fortran_fn)(int *n, TYPE *x, int *incx,           \
                                         TYPE *scale, TYPE *sumsq);            \
static struct {                                                                 \
    int called;                                                                 \
    int n;                                                                      \
    int incx;                                                                   \
    TYPE *x;                                                                    \
    TYPE *scale;                                                                \
    TYPE *sumsq;                                                                \
    TYPE x_snapshot[5];                                                         \
} g_##SUFFIX##_fortran_call;                                                    \
static struct {                                                                 \
    int called;                                                                 \
    int n;                                                                      \
    int incx;                                                                   \
    TYPE *x;                                                                    \
    TYPE *scale;                                                                \
    TYPE *sumsq;                                                                \
} g_##SUFFIX##_cblas_call;                                                      \
static void stub_##SUFFIX##_fortran(int *n, TYPE *x, int *incx,                \
                                    TYPE *scale, TYPE *sumsq)                 \
{                                                                               \
    int index;                                                                  \
    g_##SUFFIX##_fortran_call.called += 1;                                      \
    g_##SUFFIX##_fortran_call.n = *n;                                           \
    g_##SUFFIX##_fortran_call.incx = *incx;                                     \
    g_##SUFFIX##_fortran_call.x = x;                                            \
    g_##SUFFIX##_fortran_call.scale = scale;                                    \
    g_##SUFFIX##_fortran_call.sumsq = sumsq;                                    \
    for (index = 0; index < 5; ++index) {                                       \
        g_##SUFFIX##_fortran_call.x_snapshot[index] = x[index];                 \
    }                                                                           \
    *scale = (TYPE)((BASE) + 10);                                               \
    *sumsq = (TYPE)((BASE) + 11);                                               \
}                                                                               \
static int stub_##SUFFIX##_cblas(int n, TYPE *x, int incx,                     \
                                 TYPE *scale, TYPE *sumsq)                    \
{                                                                               \
    g_##SUFFIX##_cblas_call.called += 1;                                        \
    g_##SUFFIX##_cblas_call.n = n;                                              \
    g_##SUFFIX##_cblas_call.incx = incx;                                        \
    g_##SUFFIX##_cblas_call.x = x;                                              \
    g_##SUFFIX##_cblas_call.scale = scale;                                      \
    g_##SUFFIX##_cblas_call.sumsq = sumsq;                                      \
    *scale = (TYPE)((BASE) + 20);                                               \
    *sumsq = (TYPE)((BASE) + 21);                                               \
    return (BASE) + 99;                                                         \
}                                                                               \
static int check_##SUFFIX##_fortran_to_cblas(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                        \
    TYPE x[5] = { (TYPE)1, (TYPE)2, (TYPE)3, (TYPE)4, (TYPE)5 };               \
    TYPE scale = (TYPE)6;                                                       \
    TYPE sumsq = (TYPE)7;                                                       \
    int status = 0;                                                             \
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
    status = thunk(3, x, 2, &scale, &sumsq);                                   \
    if (status != 0 || g_##SUFFIX##_fortran_call.called != 1 ||                \
        g_##SUFFIX##_fortran_call.n != 3 ||                                     \
        g_##SUFFIX##_fortran_call.incx != 2 ||                                  \
        g_##SUFFIX##_fortran_call.x != x ||                                     \
        g_##SUFFIX##_fortran_call.scale != &scale ||                            \
        g_##SUFFIX##_fortran_call.sumsq != &sumsq ||                            \
        g_##SUFFIX##_fortran_call.x_snapshot[0] != (TYPE)1 ||                   \
        g_##SUFFIX##_fortran_call.x_snapshot[1] != (TYPE)2 ||                   \
        g_##SUFFIX##_fortran_call.x_snapshot[2] != (TYPE)3 ||                   \
        g_##SUFFIX##_fortran_call.x_snapshot[3] != (TYPE)4 ||                   \
        g_##SUFFIX##_fortran_call.x_snapshot[4] != (TYPE)5 ||                   \
        scale != (TYPE)((BASE) + 10) ||                                         \
        sumsq != (TYPE)((BASE) + 11)) {                                         \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward vector, stride, or scale/sumsq state correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards vector stride and updates scale/sumsq without layout translation\n"); \
    return 0;                                                                   \
}                                                                               \
static int check_##SUFFIX##_cblas_to_fortran(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                      \
    int n = 3;                                                                  \
    int incx = 2;                                                               \
    TYPE x[5] = { (TYPE)10, (TYPE)11, (TYPE)12, (TYPE)13, (TYPE)14 };          \
    TYPE scale = (TYPE)0;                                                       \
    TYPE sumsq = (TYPE)0;                                                       \
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
    thunk(&n, x, &incx, &scale, &sumsq);                                        \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                  \
        g_##SUFFIX##_cblas_call.n != 3 ||                                       \
        g_##SUFFIX##_cblas_call.incx != 2 ||                                    \
        g_##SUFFIX##_cblas_call.x != x ||                                       \
        g_##SUFFIX##_cblas_call.scale != &scale ||                              \
        g_##SUFFIX##_cblas_call.sumsq != &sumsq ||                              \
        scale != (TYPE)((BASE) + 20) ||                                         \
        sumsq != (TYPE)((BASE) + 21)) {                                         \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not forward vector, stride, or scale/sumsq state correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk forwards vector stride and ignores the C int return as expected\n"); \
    return 0;                                                                   \
}

#define DEFINE_COMPLEX_LASSQ_TESTS(SUFFIX, CTYPE, RTYPE, OP_ID, MAKE_FN, EQ_FN, BASE) \
typedef int (*fb_##SUFFIX##_cblas_fn)(int n, CTYPE *x, int incx,               \
                                      RTYPE *scale, RTYPE *sumsq);             \
typedef void (*fb_##SUFFIX##_fortran_fn)(int *n, CTYPE *x, int *incx,          \
                                         RTYPE *scale, RTYPE *sumsq);          \
static struct {                                                                 \
    int called;                                                                 \
    int n;                                                                      \
    int incx;                                                                   \
    CTYPE *x;                                                                   \
    RTYPE *scale;                                                               \
    RTYPE *sumsq;                                                               \
    CTYPE x_snapshot[5];                                                        \
} g_##SUFFIX##_fortran_call;                                                    \
static struct {                                                                 \
    int called;                                                                 \
    int n;                                                                      \
    int incx;                                                                   \
    CTYPE *x;                                                                   \
    RTYPE *scale;                                                               \
    RTYPE *sumsq;                                                               \
} g_##SUFFIX##_cblas_call;                                                      \
static void stub_##SUFFIX##_fortran(int *n, CTYPE *x, int *incx,               \
                                    RTYPE *scale, RTYPE *sumsq)               \
{                                                                               \
    int index;                                                                  \
    g_##SUFFIX##_fortran_call.called += 1;                                      \
    g_##SUFFIX##_fortran_call.n = *n;                                           \
    g_##SUFFIX##_fortran_call.incx = *incx;                                     \
    g_##SUFFIX##_fortran_call.x = x;                                            \
    g_##SUFFIX##_fortran_call.scale = scale;                                    \
    g_##SUFFIX##_fortran_call.sumsq = sumsq;                                    \
    for (index = 0; index < 5; ++index) {                                       \
        g_##SUFFIX##_fortran_call.x_snapshot[index] = x[index];                 \
    }                                                                           \
    *scale = (RTYPE)((BASE) + 10);                                              \
    *sumsq = (RTYPE)((BASE) + 11);                                              \
}                                                                               \
static int stub_##SUFFIX##_cblas(int n, CTYPE *x, int incx,                    \
                                 RTYPE *scale, RTYPE *sumsq)                  \
{                                                                               \
    g_##SUFFIX##_cblas_call.called += 1;                                        \
    g_##SUFFIX##_cblas_call.n = n;                                              \
    g_##SUFFIX##_cblas_call.incx = incx;                                        \
    g_##SUFFIX##_cblas_call.x = x;                                              \
    g_##SUFFIX##_cblas_call.scale = scale;                                      \
    g_##SUFFIX##_cblas_call.sumsq = sumsq;                                      \
    *scale = (RTYPE)((BASE) + 20);                                              \
    *sumsq = (RTYPE)((BASE) + 21);                                              \
    return (BASE) + 99;                                                         \
}                                                                               \
static int check_##SUFFIX##_fortran_to_cblas(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                        \
    CTYPE x[5];                                                                 \
    RTYPE scale = (RTYPE)6;                                                     \
    RTYPE sumsq = (RTYPE)7;                                                     \
    int status = 0;                                                             \
    x[0] = MAKE_FN((RTYPE)1, (RTYPE)11);                                        \
    x[1] = MAKE_FN((RTYPE)2, (RTYPE)12);                                        \
    x[2] = MAKE_FN((RTYPE)3, (RTYPE)13);                                        \
    x[3] = MAKE_FN((RTYPE)4, (RTYPE)14);                                        \
    x[4] = MAKE_FN((RTYPE)5, (RTYPE)15);                                        \
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
    status = thunk(3, x, 2, &scale, &sumsq);                                   \
    if (status != 0 || g_##SUFFIX##_fortran_call.called != 1 ||                \
        g_##SUFFIX##_fortran_call.n != 3 ||                                     \
        g_##SUFFIX##_fortran_call.incx != 2 ||                                  \
        g_##SUFFIX##_fortran_call.x != x ||                                     \
        g_##SUFFIX##_fortran_call.scale != &scale ||                            \
        g_##SUFFIX##_fortran_call.sumsq != &sumsq ||                            \
        !EQ_FN(g_##SUFFIX##_fortran_call.x_snapshot[0],                         \
               MAKE_FN((RTYPE)1, (RTYPE)11)) ||                                 \
        !EQ_FN(g_##SUFFIX##_fortran_call.x_snapshot[1],                         \
               MAKE_FN((RTYPE)2, (RTYPE)12)) ||                                 \
        !EQ_FN(g_##SUFFIX##_fortran_call.x_snapshot[2],                         \
               MAKE_FN((RTYPE)3, (RTYPE)13)) ||                                 \
        !EQ_FN(g_##SUFFIX##_fortran_call.x_snapshot[3],                         \
               MAKE_FN((RTYPE)4, (RTYPE)14)) ||                                 \
        !EQ_FN(g_##SUFFIX##_fortran_call.x_snapshot[4],                         \
               MAKE_FN((RTYPE)5, (RTYPE)15)) ||                                 \
        scale != (RTYPE)((BASE) + 10) ||                                        \
        sumsq != (RTYPE)((BASE) + 11)) {                                        \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward complex vector, stride, or scale/sumsq state correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards complex vector stride and updates scale/sumsq without layout translation\n"); \
    return 0;                                                                   \
}                                                                               \
static int check_##SUFFIX##_cblas_to_fortran(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                      \
    int n = 3;                                                                  \
    int incx = 2;                                                               \
    CTYPE x[5];                                                                 \
    RTYPE scale = (RTYPE)0;                                                     \
    RTYPE sumsq = (RTYPE)0;                                                     \
    x[0] = MAKE_FN((RTYPE)10, (RTYPE)110);                                      \
    x[1] = MAKE_FN((RTYPE)11, (RTYPE)111);                                      \
    x[2] = MAKE_FN((RTYPE)12, (RTYPE)112);                                      \
    x[3] = MAKE_FN((RTYPE)13, (RTYPE)113);                                      \
    x[4] = MAKE_FN((RTYPE)14, (RTYPE)114);                                      \
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
    thunk(&n, x, &incx, &scale, &sumsq);                                        \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                  \
        g_##SUFFIX##_cblas_call.n != 3 ||                                       \
        g_##SUFFIX##_cblas_call.incx != 2 ||                                    \
        g_##SUFFIX##_cblas_call.x != x ||                                       \
        g_##SUFFIX##_cblas_call.scale != &scale ||                              \
        g_##SUFFIX##_cblas_call.sumsq != &sumsq ||                              \
        scale != (RTYPE)((BASE) + 20) ||                                        \
        sumsq != (RTYPE)((BASE) + 21)) {                                        \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not forward complex vector, stride, or scale/sumsq state correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk forwards complex vector stride and ignores the C int return as expected\n"); \
    return 0;                                                                   \
}

DEFINE_REAL_LASSQ_TESTS(slassq, float, FB_OP_SLASSQ, 100)
DEFINE_REAL_LASSQ_TESTS(dlassq, double, FB_OP_DLASSQ, 300)
DEFINE_COMPLEX_LASSQ_TESTS(classq, fb_complex_float_t, float, FB_OP_CLASSQ,
                           make_cf32, cf32_eq, 500)
DEFINE_COMPLEX_LASSQ_TESTS(zlassq, fb_complex_double_t, double, FB_OP_ZLASSQ,
                           make_cf64, cf64_eq, 700)

int main(void)
{
    int status = 0;

    status |= check_slassq_fortran_to_cblas();
    status |= check_slassq_cblas_to_fortran();
    status |= check_dlassq_fortran_to_cblas();
    status |= check_dlassq_cblas_to_fortran();
    status |= check_classq_fortran_to_cblas();
    status |= check_classq_cblas_to_fortran();
    status |= check_zlassq_fortran_to_cblas();
    status |= check_zlassq_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}