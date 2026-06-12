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

#define DEFINE_LARTG_REAL_TESTS(SUFFIX, TYPE, OP_ID, BASE)                     \
typedef int (*fb_##SUFFIX##_cblas_fn)(TYPE f, TYPE g, TYPE *cs, TYPE *sn,      \
                                      TYPE *r);                                 \
typedef void (*fb_##SUFFIX##_fortran_fn)(TYPE *f, TYPE *g, TYPE *cs, TYPE *sn, \
                                         TYPE *r);                              \
static struct {                                                                 \
    int called;                                                                 \
    TYPE f;                                                                     \
    TYPE g;                                                                     \
    TYPE *cs;                                                                   \
    TYPE *sn;                                                                   \
    TYPE *r;                                                                    \
} g_##SUFFIX##_fortran_call;                                                    \
static struct {                                                                 \
    int called;                                                                 \
    TYPE f;                                                                     \
    TYPE g;                                                                     \
    TYPE *cs;                                                                   \
    TYPE *sn;                                                                   \
    TYPE *r;                                                                    \
} g_##SUFFIX##_cblas_call;                                                      \
static void stub_##SUFFIX##_fortran(TYPE *f, TYPE *g, TYPE *cs, TYPE *sn,      \
                                    TYPE *r)                                    \
{                                                                               \
    g_##SUFFIX##_fortran_call.called += 1;                                      \
    g_##SUFFIX##_fortran_call.f = *f;                                           \
    g_##SUFFIX##_fortran_call.g = *g;                                           \
    g_##SUFFIX##_fortran_call.cs = cs;                                          \
    g_##SUFFIX##_fortran_call.sn = sn;                                          \
    g_##SUFFIX##_fortran_call.r = r;                                            \
    *cs = (TYPE)((BASE) + 1);                                                   \
    *sn = (TYPE)((BASE) + 2);                                                   \
    *r = (TYPE)((BASE) + 3);                                                    \
}                                                                               \
static int stub_##SUFFIX##_cblas(TYPE f, TYPE g, TYPE *cs, TYPE *sn, TYPE *r)  \
{                                                                               \
    g_##SUFFIX##_cblas_call.called += 1;                                        \
    g_##SUFFIX##_cblas_call.f = f;                                              \
    g_##SUFFIX##_cblas_call.g = g;                                              \
    g_##SUFFIX##_cblas_call.cs = cs;                                            \
    g_##SUFFIX##_cblas_call.sn = sn;                                            \
    g_##SUFFIX##_cblas_call.r = r;                                              \
    *cs = (TYPE)((BASE) + 4);                                                   \
    *sn = (TYPE)((BASE) + 5);                                                   \
    *r = (TYPE)((BASE) + 6);                                                    \
    return 0;                                                                   \
}                                                                               \
static int check_##SUFFIX##_fortran_to_cblas(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                        \
    TYPE cs = (TYPE)0;                                                          \
    TYPE sn = (TYPE)0;                                                          \
    TYPE r = (TYPE)0;                                                           \
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
    if (thunk((TYPE)7, (TYPE)8, &cs, &sn, &r) != 0 ||                           \
        g_##SUFFIX##_fortran_call.called != 1 ||                                \
        g_##SUFFIX##_fortran_call.f != (TYPE)7 ||                               \
        g_##SUFFIX##_fortran_call.g != (TYPE)8 ||                               \
        g_##SUFFIX##_fortran_call.cs != &cs ||                                  \
        g_##SUFFIX##_fortran_call.sn != &sn ||                                  \
        g_##SUFFIX##_fortran_call.r != &r ||                                    \
        cs != (TYPE)((BASE) + 1) || sn != (TYPE)((BASE) + 2) ||                 \
        r != (TYPE)((BASE) + 3)) {                                              \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward LARTG inputs correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards LARTG inputs\n"); \
    return 0;                                                                   \
}                                                                               \
static int check_##SUFFIX##_cblas_to_fortran(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                      \
    TYPE f = (TYPE)7;                                                           \
    TYPE g = (TYPE)8;                                                           \
    TYPE cs = (TYPE)0;                                                          \
    TYPE sn = (TYPE)0;                                                          \
    TYPE r = (TYPE)0;                                                           \
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
    thunk(&f, &g, &cs, &sn, &r);                                                \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                  \
        g_##SUFFIX##_cblas_call.f != (TYPE)7 ||                                 \
        g_##SUFFIX##_cblas_call.g != (TYPE)8 ||                                 \
        g_##SUFFIX##_cblas_call.cs != &cs ||                                    \
        g_##SUFFIX##_cblas_call.sn != &sn ||                                    \
        g_##SUFFIX##_cblas_call.r != &r ||                                      \
        cs != (TYPE)((BASE) + 4) || sn != (TYPE)((BASE) + 5) ||                 \
        r != (TYPE)((BASE) + 6)) {                                              \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference LARTG inputs correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences LARTG inputs\n"); \
    return 0;                                                                   \
}

#define DEFINE_LARTG_COMPLEX_TESTS(SUFFIX, CTYPE, RTYPE, OP_ID, MAKE_FN, EQ_FN, RBASE) \
typedef int (*fb_##SUFFIX##_cblas_fn)(CTYPE f, CTYPE g, RTYPE *cs, CTYPE *sn,           \
                                      CTYPE *r);                                          \
typedef void (*fb_##SUFFIX##_fortran_fn)(CTYPE *f, CTYPE *g, RTYPE *cs, CTYPE *sn,       \
                                         CTYPE *r);                                       \
static struct {                                                                           \
    int called;                                                                           \
    CTYPE f;                                                                              \
    CTYPE g;                                                                              \
    RTYPE *cs;                                                                            \
    CTYPE *sn;                                                                            \
    CTYPE *r;                                                                             \
} g_##SUFFIX##_fortran_call;                                                              \
static struct {                                                                           \
    int called;                                                                           \
    CTYPE f;                                                                              \
    CTYPE g;                                                                              \
    RTYPE *cs;                                                                            \
    CTYPE *sn;                                                                            \
    CTYPE *r;                                                                             \
} g_##SUFFIX##_cblas_call;                                                                \
static void stub_##SUFFIX##_fortran(CTYPE *f, CTYPE *g, RTYPE *cs, CTYPE *sn, CTYPE *r)  \
{                                                                                         \
    g_##SUFFIX##_fortran_call.called += 1;                                                \
    g_##SUFFIX##_fortran_call.f = *f;                                                     \
    g_##SUFFIX##_fortran_call.g = *g;                                                     \
    g_##SUFFIX##_fortran_call.cs = cs;                                                    \
    g_##SUFFIX##_fortran_call.sn = sn;                                                    \
    g_##SUFFIX##_fortran_call.r = r;                                                      \
    *cs = (RTYPE)((RBASE) + 1);                                                           \
    *sn = MAKE_FN((RBASE) + 2, (RBASE) + 12);                                             \
    *r = MAKE_FN((RBASE) + 3, (RBASE) + 13);                                              \
}                                                                                         \
static int stub_##SUFFIX##_cblas(CTYPE f, CTYPE g, RTYPE *cs, CTYPE *sn, CTYPE *r)       \
{                                                                                         \
    g_##SUFFIX##_cblas_call.called += 1;                                                  \
    g_##SUFFIX##_cblas_call.f = f;                                                        \
    g_##SUFFIX##_cblas_call.g = g;                                                        \
    g_##SUFFIX##_cblas_call.cs = cs;                                                      \
    g_##SUFFIX##_cblas_call.sn = sn;                                                      \
    g_##SUFFIX##_cblas_call.r = r;                                                        \
    *cs = (RTYPE)((RBASE) + 4);                                                           \
    *sn = MAKE_FN((RBASE) + 5, (RBASE) + 15);                                             \
    *r = MAKE_FN((RBASE) + 6, (RBASE) + 16);                                              \
    return 0;                                                                             \
}                                                                                         \
static int check_##SUFFIX##_fortran_to_cblas(void)                                        \
{                                                                                         \
    fb_backend_vtable_t vtable;                                                           \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                                  \
    RTYPE cs = (RTYPE)0;                                                                  \
    CTYPE sn = MAKE_FN(0, 0);                                                             \
    CTYPE r = MAKE_FN(0, 0);                                                              \
    memset(&vtable, 0, sizeof(vtable));                                                   \
    memset(&g_##SUFFIX##_fortran_call, 0, sizeof(g_##SUFFIX##_fortran_call));             \
    vtable.ext_ops[OP_ID][FB_CONV_FORTRAN] =                                              \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_fortran;                           \
    fb_install_conv_thunks(&vtable, OP_ID);                                               \
    thunk = (fb_##SUFFIX##_cblas_fn)vtable.ext_ops[OP_ID][FB_CONV_CBLAS];                 \
    if (!thunk) {                                                                         \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk was not installed\n"); \
        return 1;                                                                         \
    }                                                                                     \
    if (thunk(MAKE_FN((RBASE) + 7, (RBASE) + 17), MAKE_FN((RBASE) + 8, (RBASE) + 18),    \
              &cs, &sn, &r) != 0 ||                                                       \
        g_##SUFFIX##_fortran_call.called != 1 ||                                          \
        !EQ_FN(g_##SUFFIX##_fortran_call.f, MAKE_FN((RBASE) + 7, (RBASE) + 17)) ||       \
        !EQ_FN(g_##SUFFIX##_fortran_call.g, MAKE_FN((RBASE) + 8, (RBASE) + 18)) ||       \
        g_##SUFFIX##_fortran_call.cs != &cs ||                                            \
        g_##SUFFIX##_fortran_call.sn != &sn ||                                            \
        g_##SUFFIX##_fortran_call.r != &r ||                                              \
        cs != (RTYPE)((RBASE) + 1) ||                                                     \
        !EQ_FN(sn, MAKE_FN((RBASE) + 2, (RBASE) + 12)) ||                                 \
        !EQ_FN(r, MAKE_FN((RBASE) + 3, (RBASE) + 13))) {                                  \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward LARTG inputs correctly\n"); \
        return 1;                                                                         \
    }                                                                                     \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards LARTG inputs\n");        \
    return 0;                                                                             \
}                                                                                         \
static int check_##SUFFIX##_cblas_to_fortran(void)                                        \
{                                                                                         \
    fb_backend_vtable_t vtable;                                                           \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                                \
    CTYPE f = MAKE_FN((RBASE) + 7, (RBASE) + 17);                                         \
    CTYPE g = MAKE_FN((RBASE) + 8, (RBASE) + 18);                                         \
    RTYPE cs = (RTYPE)0;                                                                   \
    CTYPE sn = MAKE_FN(0, 0);                                                              \
    CTYPE r = MAKE_FN(0, 0);                                                               \
    memset(&vtable, 0, sizeof(vtable));                                                   \
    memset(&g_##SUFFIX##_cblas_call, 0, sizeof(g_##SUFFIX##_cblas_call));                 \
    vtable.ext_ops[OP_ID][FB_CONV_CBLAS] =                                                \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_cblas;                             \
    fb_install_conv_thunks(&vtable, OP_ID);                                               \
    thunk = (fb_##SUFFIX##_fortran_fn)vtable.ext_ops[OP_ID][FB_CONV_FORTRAN];             \
    if (!thunk) {                                                                         \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk was not installed\n"); \
        return 1;                                                                         \
    }                                                                                     \
    thunk(&f, &g, &cs, &sn, &r);                                                          \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                            \
        !EQ_FN(g_##SUFFIX##_cblas_call.f, MAKE_FN((RBASE) + 7, (RBASE) + 17)) ||         \
        !EQ_FN(g_##SUFFIX##_cblas_call.g, MAKE_FN((RBASE) + 8, (RBASE) + 18)) ||         \
        g_##SUFFIX##_cblas_call.cs != &cs ||                                              \
        g_##SUFFIX##_cblas_call.sn != &sn ||                                              \
        g_##SUFFIX##_cblas_call.r != &r ||                                                \
        cs != (RTYPE)((RBASE) + 4) ||                                                     \
        !EQ_FN(sn, MAKE_FN((RBASE) + 5, (RBASE) + 15)) ||                                 \
        !EQ_FN(r, MAKE_FN((RBASE) + 6, (RBASE) + 16))) {                                  \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference LARTG inputs correctly\n"); \
        return 1;                                                                         \
    }                                                                                     \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences LARTG inputs\n");    \
    return 0;                                                                             \
}

DEFINE_LARTG_REAL_TESTS(slartg, float, FB_OP_SLARTG, 100)
DEFINE_LARTG_REAL_TESTS(dlartg, double, FB_OP_DLARTG, 300)
DEFINE_LARTG_COMPLEX_TESTS(clartg, fb_complex_float_t, float, FB_OP_CLARTG, make_cf32, cf32_eq, 500)
DEFINE_LARTG_COMPLEX_TESTS(zlartg, fb_complex_double_t, double, FB_OP_ZLARTG, make_cf64, cf64_eq, 700)

int main(void)
{
    int status = 0;

    status |= check_slartg_fortran_to_cblas();
    status |= check_slartg_cblas_to_fortran();
    status |= check_dlartg_fortran_to_cblas();
    status |= check_dlartg_cblas_to_fortran();
    status |= check_clartg_fortran_to_cblas();
    status |= check_clartg_cblas_to_fortran();
    status |= check_zlartg_fortran_to_cblas();
    status |= check_zlartg_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}