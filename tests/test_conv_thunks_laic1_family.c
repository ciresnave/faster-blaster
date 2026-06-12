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

#define DEFINE_LAIC1_TESTS(SUFFIX, TYPE, OP_ID, BASE)                          \
typedef int (*fb_##SUFFIX##_cblas_fn)(int job, int j, TYPE *x, TYPE sest,      \
                                      TYPE *w, TYPE gamma, TYPE *sestpr,       \
                                      TYPE *s, TYPE *c_out);                   \
typedef void (*fb_##SUFFIX##_fortran_fn)(int *job, int *j, TYPE *x,            \
                                         TYPE *sest, TYPE *w, TYPE *gamma,     \
                                         TYPE *sestpr, TYPE *s, TYPE *c_out);  \
static struct {                                                                 \
    int called;                                                                 \
    int job;                                                                    \
    int j;                                                                      \
    TYPE *x;                                                                    \
    TYPE sest;                                                                  \
    TYPE *w;                                                                    \
    TYPE gamma;                                                                 \
    TYPE *sestpr;                                                               \
    TYPE *s;                                                                    \
    TYPE *c_out;                                                                \
    TYPE x_snapshot[4];                                                         \
    TYPE w_snapshot[4];                                                         \
} g_##SUFFIX##_fortran_call;                                                    \
static struct {                                                                 \
    int called;                                                                 \
    int job;                                                                    \
    int j;                                                                      \
    TYPE *x;                                                                    \
    TYPE sest;                                                                  \
    TYPE *w;                                                                    \
    TYPE gamma;                                                                 \
    TYPE *sestpr;                                                               \
    TYPE *s;                                                                    \
    TYPE *c_out;                                                                \
} g_##SUFFIX##_cblas_call;                                                      \
static void stub_##SUFFIX##_fortran(int *job, int *j, TYPE *x, TYPE *sest,     \
                                    TYPE *w, TYPE *gamma, TYPE *sestpr,        \
                                    TYPE *s, TYPE *c_out)                     \
{                                                                               \
    int index;                                                                  \
    g_##SUFFIX##_fortran_call.called += 1;                                      \
    g_##SUFFIX##_fortran_call.job = *job;                                       \
    g_##SUFFIX##_fortran_call.j = *j;                                           \
    g_##SUFFIX##_fortran_call.x = x;                                            \
    g_##SUFFIX##_fortran_call.sest = *sest;                                     \
    g_##SUFFIX##_fortran_call.w = w;                                            \
    g_##SUFFIX##_fortran_call.gamma = *gamma;                                   \
    g_##SUFFIX##_fortran_call.sestpr = sestpr;                                  \
    g_##SUFFIX##_fortran_call.s = s;                                            \
    g_##SUFFIX##_fortran_call.c_out = c_out;                                    \
    for (index = 0; index < 4; ++index) {                                       \
        g_##SUFFIX##_fortran_call.x_snapshot[index] = x[index];                 \
        g_##SUFFIX##_fortran_call.w_snapshot[index] = w[index];                 \
    }                                                                           \
    *sestpr = (TYPE)((BASE) + 1);                                               \
    *s = (TYPE)((BASE) + 2);                                                    \
    *c_out = (TYPE)((BASE) + 3);                                                \
}                                                                               \
static int stub_##SUFFIX##_cblas(int job, int j, TYPE *x, TYPE sest, TYPE *w,  \
                                 TYPE gamma, TYPE *sestpr, TYPE *s,            \
                                 TYPE *c_out)                                  \
{                                                                               \
    g_##SUFFIX##_cblas_call.called += 1;                                        \
    g_##SUFFIX##_cblas_call.job = job;                                          \
    g_##SUFFIX##_cblas_call.j = j;                                              \
    g_##SUFFIX##_cblas_call.x = x;                                              \
    g_##SUFFIX##_cblas_call.sest = sest;                                        \
    g_##SUFFIX##_cblas_call.w = w;                                              \
    g_##SUFFIX##_cblas_call.gamma = gamma;                                      \
    g_##SUFFIX##_cblas_call.sestpr = sestpr;                                    \
    g_##SUFFIX##_cblas_call.s = s;                                              \
    g_##SUFFIX##_cblas_call.c_out = c_out;                                      \
    *sestpr = (TYPE)((BASE) + 4);                                               \
    *s = (TYPE)((BASE) + 5);                                                    \
    *c_out = (TYPE)((BASE) + 6);                                                \
    return (BASE) + 99;                                                         \
}                                                                               \
static int check_##SUFFIX##_fortran_to_cblas(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                        \
    TYPE x[4] = { (TYPE)1, (TYPE)2, (TYPE)3, (TYPE)4 };                        \
    TYPE w[4] = { (TYPE)11, (TYPE)12, (TYPE)13, (TYPE)14 };                    \
    TYPE sestpr = (TYPE)0;                                                      \
    TYPE s = (TYPE)0;                                                           \
    TYPE c_out = (TYPE)0;                                                       \
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
    if (thunk(2, 3, x, (TYPE)((BASE) + 10), w, (TYPE)((BASE) + 20),            \
              &sestpr, &s, &c_out) != 0 ||                                      \
        g_##SUFFIX##_fortran_call.called != 1 ||                                \
        g_##SUFFIX##_fortran_call.job != 2 ||                                   \
        g_##SUFFIX##_fortran_call.j != 3 ||                                     \
        g_##SUFFIX##_fortran_call.x != x ||                                     \
        g_##SUFFIX##_fortran_call.sest != (TYPE)((BASE) + 10) ||                \
        g_##SUFFIX##_fortran_call.w != w ||                                     \
        g_##SUFFIX##_fortran_call.gamma != (TYPE)((BASE) + 20) ||               \
        g_##SUFFIX##_fortran_call.sestpr != &sestpr ||                          \
        g_##SUFFIX##_fortran_call.s != &s ||                                    \
        g_##SUFFIX##_fortran_call.c_out != &c_out ||                            \
        g_##SUFFIX##_fortran_call.x_snapshot[0] != (TYPE)1 ||                   \
        g_##SUFFIX##_fortran_call.x_snapshot[1] != (TYPE)2 ||                   \
        g_##SUFFIX##_fortran_call.x_snapshot[2] != (TYPE)3 ||                   \
        g_##SUFFIX##_fortran_call.w_snapshot[0] != (TYPE)11 ||                  \
        g_##SUFFIX##_fortran_call.w_snapshot[1] != (TYPE)12 ||                  \
        g_##SUFFIX##_fortran_call.w_snapshot[2] != (TYPE)13 ||                  \
        sestpr != (TYPE)((BASE) + 1) || s != (TYPE)((BASE) + 2) ||              \
        c_out != (TYPE)((BASE) + 3)) {                                          \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward job/j, vectors, or scalar outputs correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards job/j, vectors, and scalar outputs without layout translation\n"); \
    return 0;                                                                   \
}                                                                               \
static int check_##SUFFIX##_cblas_to_fortran(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                      \
    int job = 1;                                                                \
    int j = 3;                                                                  \
    TYPE x[4] = { (TYPE)21, (TYPE)22, (TYPE)23, (TYPE)24 };                    \
    TYPE sest = (TYPE)((BASE) + 30);                                            \
    TYPE w[4] = { (TYPE)31, (TYPE)32, (TYPE)33, (TYPE)34 };                    \
    TYPE gamma = (TYPE)((BASE) + 40);                                           \
    TYPE sestpr = (TYPE)0;                                                      \
    TYPE s = (TYPE)0;                                                           \
    TYPE c_out = (TYPE)0;                                                       \
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
    thunk(&job, &j, x, &sest, w, &gamma, &sestpr, &s, &c_out);                 \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                  \
        g_##SUFFIX##_cblas_call.job != 1 ||                                     \
        g_##SUFFIX##_cblas_call.j != 3 ||                                       \
        g_##SUFFIX##_cblas_call.x != x ||                                       \
        g_##SUFFIX##_cblas_call.sest != (TYPE)((BASE) + 30) ||                  \
        g_##SUFFIX##_cblas_call.w != w ||                                       \
        g_##SUFFIX##_cblas_call.gamma != (TYPE)((BASE) + 40) ||                 \
        g_##SUFFIX##_cblas_call.sestpr != &sestpr ||                            \
        g_##SUFFIX##_cblas_call.s != &s ||                                      \
        g_##SUFFIX##_cblas_call.c_out != &c_out ||                              \
        sestpr != (TYPE)((BASE) + 4) || s != (TYPE)((BASE) + 5) ||              \
        c_out != (TYPE)((BASE) + 6)) {                                          \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference scalar inputs or propagate scalar outputs correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences scalar inputs and ignores the C int return\n"); \
    return 0;                                                                   \
}

#define DEFINE_COMPLEX_LAIC1_TESTS(SUFFIX, CTYPE, RTYPE, OP_ID, MAKE_FN, EQ_FN, BASE) \
typedef int (*fb_##SUFFIX##_cblas_fn)(int job, int j, CTYPE *x, RTYPE sest,    \
                                      CTYPE *w, CTYPE gamma, RTYPE *sestpr,    \
                                      CTYPE *s, CTYPE *c_out);                 \
typedef void (*fb_##SUFFIX##_fortran_fn)(int *job, int *j, CTYPE *x,           \
                                         RTYPE *sest, CTYPE *w, CTYPE *gamma,  \
                                         RTYPE *sestpr, CTYPE *s,              \
                                         CTYPE *c_out);                        \
static struct {                                                                 \
    int called;                                                                 \
    int job;                                                                    \
    int j;                                                                      \
    CTYPE *x;                                                                    \
    RTYPE sest;                                                                  \
    CTYPE *w;                                                                    \
    CTYPE gamma;                                                                 \
    RTYPE *sestpr;                                                               \
    CTYPE *s;                                                                    \
    CTYPE *c_out;                                                                \
    CTYPE x_snapshot[4];                                                         \
    CTYPE w_snapshot[4];                                                         \
} g_##SUFFIX##_fortran_call;                                                    \
static struct {                                                                 \
    int called;                                                                 \
    int job;                                                                    \
    int j;                                                                      \
    CTYPE *x;                                                                    \
    RTYPE sest;                                                                  \
    CTYPE *w;                                                                    \
    CTYPE gamma;                                                                 \
    RTYPE *sestpr;                                                               \
    CTYPE *s;                                                                    \
    CTYPE *c_out;                                                                \
} g_##SUFFIX##_cblas_call;                                                      \
static void stub_##SUFFIX##_fortran(int *job, int *j, CTYPE *x, RTYPE *sest,   \
                                    CTYPE *w, CTYPE *gamma, RTYPE *sestpr,     \
                                    CTYPE *s, CTYPE *c_out)                    \
{                                                                               \
    int index;                                                                  \
    g_##SUFFIX##_fortran_call.called += 1;                                      \
    g_##SUFFIX##_fortran_call.job = *job;                                       \
    g_##SUFFIX##_fortran_call.j = *j;                                           \
    g_##SUFFIX##_fortran_call.x = x;                                            \
    g_##SUFFIX##_fortran_call.sest = *sest;                                     \
    g_##SUFFIX##_fortran_call.w = w;                                            \
    g_##SUFFIX##_fortran_call.gamma = *gamma;                                   \
    g_##SUFFIX##_fortran_call.sestpr = sestpr;                                  \
    g_##SUFFIX##_fortran_call.s = s;                                            \
    g_##SUFFIX##_fortran_call.c_out = c_out;                                    \
    for (index = 0; index < 4; ++index) {                                       \
        g_##SUFFIX##_fortran_call.x_snapshot[index] = x[index];                 \
        g_##SUFFIX##_fortran_call.w_snapshot[index] = w[index];                 \
    }                                                                           \
    *sestpr = (RTYPE)((BASE) + 1);                                              \
    *s = MAKE_FN((BASE) + 2, (BASE) + 12);                                      \
    *c_out = MAKE_FN((BASE) + 3, (BASE) + 13);                                  \
}                                                                               \
static int stub_##SUFFIX##_cblas(int job, int j, CTYPE *x, RTYPE sest,         \
                                 CTYPE *w, CTYPE gamma, RTYPE *sestpr,         \
                                 CTYPE *s, CTYPE *c_out)                       \
{                                                                               \
    g_##SUFFIX##_cblas_call.called += 1;                                        \
    g_##SUFFIX##_cblas_call.job = job;                                          \
    g_##SUFFIX##_cblas_call.j = j;                                              \
    g_##SUFFIX##_cblas_call.x = x;                                              \
    g_##SUFFIX##_cblas_call.sest = sest;                                        \
    g_##SUFFIX##_cblas_call.w = w;                                              \
    g_##SUFFIX##_cblas_call.gamma = gamma;                                      \
    g_##SUFFIX##_cblas_call.sestpr = sestpr;                                    \
    g_##SUFFIX##_cblas_call.s = s;                                              \
    g_##SUFFIX##_cblas_call.c_out = c_out;                                      \
    *sestpr = (RTYPE)((BASE) + 4);                                              \
    *s = MAKE_FN((BASE) + 5, (BASE) + 15);                                      \
    *c_out = MAKE_FN((BASE) + 6, (BASE) + 16);                                  \
    return (BASE) + 99;                                                         \
}                                                                               \
static int check_##SUFFIX##_fortran_to_cblas(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                        \
    CTYPE x[4];                                                                 \
    CTYPE w[4];                                                                 \
    RTYPE sestpr = (RTYPE)0;                                                    \
    CTYPE s = MAKE_FN(0, 0);                                                    \
    CTYPE c_out = MAKE_FN(0, 0);                                                \
    x[0] = MAKE_FN((BASE) + 1, (BASE) + 21);                                    \
    x[1] = MAKE_FN((BASE) + 2, (BASE) + 22);                                    \
    x[2] = MAKE_FN((BASE) + 3, (BASE) + 23);                                    \
    x[3] = MAKE_FN((BASE) + 4, (BASE) + 24);                                    \
    w[0] = MAKE_FN((BASE) + 11, (BASE) + 31);                                   \
    w[1] = MAKE_FN((BASE) + 12, (BASE) + 32);                                   \
    w[2] = MAKE_FN((BASE) + 13, (BASE) + 33);                                   \
    w[3] = MAKE_FN((BASE) + 14, (BASE) + 34);                                   \
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
    if (thunk(2, 3, x, (RTYPE)((BASE) + 10), w,                                \
              MAKE_FN((BASE) + 20, (BASE) + 40), &sestpr, &s, &c_out) != 0 ||  \
        g_##SUFFIX##_fortran_call.called != 1 ||                                \
        g_##SUFFIX##_fortran_call.job != 2 ||                                   \
        g_##SUFFIX##_fortran_call.j != 3 ||                                     \
        g_##SUFFIX##_fortran_call.x != x ||                                     \
        g_##SUFFIX##_fortran_call.sest != (RTYPE)((BASE) + 10) ||               \
        g_##SUFFIX##_fortran_call.w != w ||                                     \
        !EQ_FN(g_##SUFFIX##_fortran_call.gamma,                                 \
               MAKE_FN((BASE) + 20, (BASE) + 40)) ||                           \
        g_##SUFFIX##_fortran_call.sestpr != &sestpr ||                          \
        g_##SUFFIX##_fortran_call.s != &s ||                                    \
        g_##SUFFIX##_fortran_call.c_out != &c_out ||                            \
        !EQ_FN(g_##SUFFIX##_fortran_call.x_snapshot[0],                         \
               MAKE_FN((BASE) + 1, (BASE) + 21)) ||                            \
        !EQ_FN(g_##SUFFIX##_fortran_call.x_snapshot[1],                         \
               MAKE_FN((BASE) + 2, (BASE) + 22)) ||                            \
        !EQ_FN(g_##SUFFIX##_fortran_call.x_snapshot[2],                         \
               MAKE_FN((BASE) + 3, (BASE) + 23)) ||                            \
        !EQ_FN(g_##SUFFIX##_fortran_call.w_snapshot[0],                         \
               MAKE_FN((BASE) + 11, (BASE) + 31)) ||                           \
        !EQ_FN(g_##SUFFIX##_fortran_call.w_snapshot[1],                         \
               MAKE_FN((BASE) + 12, (BASE) + 32)) ||                           \
        !EQ_FN(g_##SUFFIX##_fortran_call.w_snapshot[2],                         \
               MAKE_FN((BASE) + 13, (BASE) + 33)) ||                           \
        sestpr != (RTYPE)((BASE) + 1) ||                                        \
        !EQ_FN(s, MAKE_FN((BASE) + 2, (BASE) + 12)) ||                         \
        !EQ_FN(c_out, MAKE_FN((BASE) + 3, (BASE) + 13))) {                     \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward mixed real/complex LAIC1 state correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards mixed real/complex LAIC1 state without layout translation\n"); \
    return 0;                                                                   \
}                                                                               \
static int check_##SUFFIX##_cblas_to_fortran(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                      \
    int job = 1;                                                                \
    int j = 3;                                                                  \
    CTYPE x[4];                                                                 \
    RTYPE sest = (RTYPE)((BASE) + 30);                                          \
    CTYPE w[4];                                                                 \
    CTYPE gamma = MAKE_FN((BASE) + 40, (BASE) + 60);                            \
    RTYPE sestpr = (RTYPE)0;                                                    \
    CTYPE s = MAKE_FN(0, 0);                                                    \
    CTYPE c_out = MAKE_FN(0, 0);                                                \
    x[0] = MAKE_FN((BASE) + 21, (BASE) + 41);                                   \
    x[1] = MAKE_FN((BASE) + 22, (BASE) + 42);                                   \
    x[2] = MAKE_FN((BASE) + 23, (BASE) + 43);                                   \
    x[3] = MAKE_FN((BASE) + 24, (BASE) + 44);                                   \
    w[0] = MAKE_FN((BASE) + 31, (BASE) + 51);                                   \
    w[1] = MAKE_FN((BASE) + 32, (BASE) + 52);                                   \
    w[2] = MAKE_FN((BASE) + 33, (BASE) + 53);                                   \
    w[3] = MAKE_FN((BASE) + 34, (BASE) + 54);                                   \
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
    thunk(&job, &j, x, &sest, w, &gamma, &sestpr, &s, &c_out);                 \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                  \
        g_##SUFFIX##_cblas_call.job != 1 ||                                     \
        g_##SUFFIX##_cblas_call.j != 3 ||                                       \
        g_##SUFFIX##_cblas_call.x != x ||                                       \
        g_##SUFFIX##_cblas_call.sest != (RTYPE)((BASE) + 30) ||                 \
        g_##SUFFIX##_cblas_call.w != w ||                                       \
        !EQ_FN(g_##SUFFIX##_cblas_call.gamma, gamma) ||                         \
        g_##SUFFIX##_cblas_call.sestpr != &sestpr ||                            \
        g_##SUFFIX##_cblas_call.s != &s ||                                      \
        g_##SUFFIX##_cblas_call.c_out != &c_out ||                              \
        sestpr != (RTYPE)((BASE) + 4) ||                                        \
        !EQ_FN(s, MAKE_FN((BASE) + 5, (BASE) + 15)) ||                         \
        !EQ_FN(c_out, MAKE_FN((BASE) + 6, (BASE) + 16))) {                     \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference mixed real/complex LAIC1 inputs correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences mixed real/complex LAIC1 inputs and ignores the C int return\n"); \
    return 0;                                                                   \
}

DEFINE_LAIC1_TESTS(slaic1, float, FB_OP_SLAIC1, 100)
DEFINE_LAIC1_TESTS(dlaic1, double, FB_OP_DLAIC1, 300)
DEFINE_COMPLEX_LAIC1_TESTS(claic1, fb_complex_float_t, float, FB_OP_CLAIC1,
                           make_cf32, cf32_eq, 500)
DEFINE_COMPLEX_LAIC1_TESTS(zlaic1, fb_complex_double_t, double, FB_OP_ZLAIC1,
                           make_cf64, cf64_eq, 700)

int main(void)
{
    int status = 0;

    status |= check_slaic1_fortran_to_cblas();
    status |= check_slaic1_cblas_to_fortran();
    status |= check_dlaic1_fortran_to_cblas();
    status |= check_dlaic1_cblas_to_fortran();
    status |= check_claic1_fortran_to_cblas();
    status |= check_claic1_cblas_to_fortran();
    status |= check_zlaic1_fortran_to_cblas();
    status |= check_zlaic1_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}