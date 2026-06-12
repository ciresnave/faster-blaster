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

#define DEFINE_LARZT_REAL_TESTS(SUFFIX, TYPE, OP_ID, BASE, DIRECT_CHAR, STOREV_CHAR) \
typedef int (*fb_##SUFFIX##_cblas_fn)(char direct, char storev, int n, int k, TYPE *v, int ldv, TYPE *tau, TYPE *t, \
                                      int ldt);                                      \
typedef void (*fb_##SUFFIX##_fortran_fn)(char *direct, char *storev, int *n, int *k, TYPE *v, int *ldv, TYPE *tau, \
                                         TYPE *t, int *ldt);                         \
static struct {                                                                       \
    int called;                                                                       \
    char direct;                                                                      \
    char storev;                                                                      \
    int n;                                                                            \
    int k;                                                                            \
    TYPE *v;                                                                          \
    int ldv;                                                                          \
    TYPE *tau;                                                                        \
    TYPE *t;                                                                          \
    int ldt;                                                                          \
} g_##SUFFIX##_fortran_call;                                                          \
static struct {                                                                       \
    int called;                                                                       \
    char direct;                                                                      \
    char storev;                                                                      \
    int n;                                                                            \
    int k;                                                                            \
    TYPE *v;                                                                          \
    int ldv;                                                                          \
    TYPE *tau;                                                                        \
    TYPE *t;                                                                          \
    int ldt;                                                                          \
} g_##SUFFIX##_cblas_call;                                                            \
static void stub_##SUFFIX##_fortran(char *direct, char *storev, int *n, int *k, TYPE *v, int *ldv, TYPE *tau, TYPE *t, \
                                    int *ldt)                                         \
{                                                                                     \
    g_##SUFFIX##_fortran_call.called += 1;                                            \
    g_##SUFFIX##_fortran_call.direct = *direct;                                       \
    g_##SUFFIX##_fortran_call.storev = *storev;                                       \
    g_##SUFFIX##_fortran_call.n = *n;                                                 \
    g_##SUFFIX##_fortran_call.k = *k;                                                 \
    g_##SUFFIX##_fortran_call.v = v;                                                  \
    g_##SUFFIX##_fortran_call.ldv = *ldv;                                             \
    g_##SUFFIX##_fortran_call.tau = tau;                                              \
    g_##SUFFIX##_fortran_call.t = t;                                                  \
    g_##SUFFIX##_fortran_call.ldt = *ldt;                                             \
    t[0] = (TYPE)((BASE) + 1);                                                        \
    t[1] = (TYPE)((BASE) + 2);                                                        \
    t[3] = (TYPE)((BASE) + 3);                                                        \
}                                                                                     \
static int stub_##SUFFIX##_cblas(char direct, char storev, int n, int k, TYPE *v, int ldv, TYPE *tau, TYPE *t, \
                                 int ldt)                                             \
{                                                                                     \
    g_##SUFFIX##_cblas_call.called += 1;                                              \
    g_##SUFFIX##_cblas_call.direct = direct;                                          \
    g_##SUFFIX##_cblas_call.storev = storev;                                          \
    g_##SUFFIX##_cblas_call.n = n;                                                    \
    g_##SUFFIX##_cblas_call.k = k;                                                    \
    g_##SUFFIX##_cblas_call.v = v;                                                    \
    g_##SUFFIX##_cblas_call.ldv = ldv;                                                \
    g_##SUFFIX##_cblas_call.tau = tau;                                                \
    g_##SUFFIX##_cblas_call.t = t;                                                    \
    g_##SUFFIX##_cblas_call.ldt = ldt;                                                \
    t[0] = (TYPE)((BASE) + 10);                                                       \
    t[1] = (TYPE)((BASE) + 11);                                                       \
    t[3] = (TYPE)((BASE) + 12);                                                       \
    return (BASE) + 99;                                                               \
}                                                                                     \
static int check_##SUFFIX##_fortran_to_cblas(void)                                    \
{                                                                                     \
    fb_backend_vtable_t vtable;                                                       \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                              \
    TYPE v[9] = { (TYPE)1, (TYPE)2, (TYPE)3, (TYPE)4, (TYPE)5, (TYPE)6, (TYPE)7, (TYPE)8, (TYPE)9 }; \
    TYPE tau[3] = { (TYPE)10, (TYPE)11, (TYPE)12 };                                   \
    TYPE t[9] = { (TYPE)0, (TYPE)0, (TYPE)0, (TYPE)0, (TYPE)0, (TYPE)0, (TYPE)0, (TYPE)0, (TYPE)0 }; \
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
    if (thunk((DIRECT_CHAR), (STOREV_CHAR), 3, 2, v, 3, tau, t, 3) != 0 ||           \
        g_##SUFFIX##_fortran_call.called != 1 ||                                      \
        g_##SUFFIX##_fortran_call.direct != (DIRECT_CHAR) ||                          \
        g_##SUFFIX##_fortran_call.storev != (STOREV_CHAR) ||                          \
        g_##SUFFIX##_fortran_call.n != 3 ||                                           \
        g_##SUFFIX##_fortran_call.k != 2 ||                                           \
        g_##SUFFIX##_fortran_call.v != v ||                                           \
        g_##SUFFIX##_fortran_call.ldv != 3 ||                                         \
        g_##SUFFIX##_fortran_call.tau != tau ||                                       \
        g_##SUFFIX##_fortran_call.t != t ||                                           \
        g_##SUFFIX##_fortran_call.ldt != 3 ||                                         \
        t[0] != (TYPE)((BASE) + 1) || t[1] != (TYPE)((BASE) + 2) || t[3] != (TYPE)((BASE) + 3)) { \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward LARZT inputs or outputs correctly\n"); \
        return 1;                                                                     \
    }                                                                                 \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards LARZT inputs and returns success\n"); \
    return 0;                                                                         \
}                                                                                     \
static int check_##SUFFIX##_cblas_to_fortran(void)                                    \
{                                                                                     \
    fb_backend_vtable_t vtable;                                                       \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                            \
    char direct = (DIRECT_CHAR);                                                      \
    char storev = (STOREV_CHAR);                                                      \
    int n = 3;                                                                        \
    int k = 2;                                                                        \
    int ldv = 3;                                                                      \
    int ldt = 3;                                                                      \
    TYPE v[9] = { (TYPE)11, (TYPE)12, (TYPE)13, (TYPE)14, (TYPE)15, (TYPE)16, (TYPE)17, (TYPE)18, (TYPE)19 }; \
    TYPE tau[3] = { (TYPE)20, (TYPE)21, (TYPE)22 };                                  \
    TYPE t[9] = { (TYPE)0, (TYPE)0, (TYPE)0, (TYPE)0, (TYPE)0, (TYPE)0, (TYPE)0, (TYPE)0, (TYPE)0 }; \
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
    thunk(&direct, &storev, &n, &k, v, &ldv, tau, t, &ldt);                          \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                        \
        g_##SUFFIX##_cblas_call.direct != (DIRECT_CHAR) ||                            \
        g_##SUFFIX##_cblas_call.storev != (STOREV_CHAR) ||                            \
        g_##SUFFIX##_cblas_call.n != 3 ||                                             \
        g_##SUFFIX##_cblas_call.k != 2 ||                                             \
        g_##SUFFIX##_cblas_call.v != v ||                                             \
        g_##SUFFIX##_cblas_call.ldv != 3 ||                                           \
        g_##SUFFIX##_cblas_call.tau != tau ||                                         \
        g_##SUFFIX##_cblas_call.t != t ||                                             \
        g_##SUFFIX##_cblas_call.ldt != 3 ||                                           \
        t[0] != (TYPE)((BASE) + 10) || t[1] != (TYPE)((BASE) + 11) || t[3] != (TYPE)((BASE) + 12)) { \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference LARZT inputs or propagate outputs correctly\n"); \
        return 1;                                                                     \
    }                                                                                 \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences LARZT inputs and ignores the C int return\n"); \
    return 0;                                                                         \
}

#define DEFINE_LARZT_COMPLEX_TESTS(SUFFIX, CTYPE, OP_ID, MAKE_FN, EQ_FN, BASE, DIRECT_CHAR, STOREV_CHAR) \
typedef int (*fb_##SUFFIX##_cblas_fn)(char direct, char storev, int n, int k, CTYPE *v, int ldv, CTYPE *tau, CTYPE *t, \
                                      int ldt);                                      \
typedef void (*fb_##SUFFIX##_fortran_fn)(char *direct, char *storev, int *n, int *k, CTYPE *v, int *ldv, CTYPE *tau, \
                                         CTYPE *t, int *ldt);                        \
static struct {                                                                       \
    int called;                                                                       \
    char direct;                                                                      \
    char storev;                                                                      \
    int n;                                                                            \
    int k;                                                                            \
    CTYPE *v;                                                                         \
    int ldv;                                                                          \
    CTYPE *tau;                                                                       \
    CTYPE *t;                                                                         \
    int ldt;                                                                          \
} g_##SUFFIX##_fortran_call;                                                          \
static struct {                                                                       \
    int called;                                                                       \
    char direct;                                                                      \
    char storev;                                                                      \
    int n;                                                                            \
    int k;                                                                            \
    CTYPE *v;                                                                         \
    int ldv;                                                                          \
    CTYPE *tau;                                                                       \
    CTYPE *t;                                                                         \
    int ldt;                                                                          \
} g_##SUFFIX##_cblas_call;                                                            \
static void stub_##SUFFIX##_fortran(char *direct, char *storev, int *n, int *k, CTYPE *v, int *ldv, CTYPE *tau, CTYPE *t, \
                                    int *ldt)                                         \
{                                                                                     \
    g_##SUFFIX##_fortran_call.called += 1;                                            \
    g_##SUFFIX##_fortran_call.direct = *direct;                                       \
    g_##SUFFIX##_fortran_call.storev = *storev;                                       \
    g_##SUFFIX##_fortran_call.n = *n;                                                 \
    g_##SUFFIX##_fortran_call.k = *k;                                                 \
    g_##SUFFIX##_fortran_call.v = v;                                                  \
    g_##SUFFIX##_fortran_call.ldv = *ldv;                                             \
    g_##SUFFIX##_fortran_call.tau = tau;                                              \
    g_##SUFFIX##_fortran_call.t = t;                                                  \
    g_##SUFFIX##_fortran_call.ldt = *ldt;                                             \
    t[0] = MAKE_FN((BASE) + 1, (BASE) + 11);                                          \
    t[1] = MAKE_FN((BASE) + 2, (BASE) + 12);                                          \
    t[3] = MAKE_FN((BASE) + 3, (BASE) + 13);                                          \
}                                                                                     \
static int stub_##SUFFIX##_cblas(char direct, char storev, int n, int k, CTYPE *v, int ldv, CTYPE *tau, CTYPE *t, \
                                 int ldt)                                             \
{                                                                                     \
    g_##SUFFIX##_cblas_call.called += 1;                                              \
    g_##SUFFIX##_cblas_call.direct = direct;                                          \
    g_##SUFFIX##_cblas_call.storev = storev;                                          \
    g_##SUFFIX##_cblas_call.n = n;                                                    \
    g_##SUFFIX##_cblas_call.k = k;                                                    \
    g_##SUFFIX##_cblas_call.v = v;                                                    \
    g_##SUFFIX##_cblas_call.ldv = ldv;                                                \
    g_##SUFFIX##_cblas_call.tau = tau;                                                \
    g_##SUFFIX##_cblas_call.t = t;                                                    \
    g_##SUFFIX##_cblas_call.ldt = ldt;                                                \
    t[0] = MAKE_FN((BASE) + 10, (BASE) + 20);                                         \
    t[1] = MAKE_FN((BASE) + 11, (BASE) + 21);                                         \
    t[3] = MAKE_FN((BASE) + 12, (BASE) + 22);                                         \
    return (BASE) + 99;                                                               \
}                                                                                     \
static int check_##SUFFIX##_fortran_to_cblas(void)                                    \
{                                                                                     \
    fb_backend_vtable_t vtable;                                                       \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                              \
    CTYPE v[6] = { MAKE_FN(1, 11), MAKE_FN(2, 12), MAKE_FN(3, 13), MAKE_FN(4, 14), MAKE_FN(5, 15), MAKE_FN(6, 16) }; \
    CTYPE tau[2] = { MAKE_FN(7, 17), MAKE_FN(8, 18) };                                \
    CTYPE t[4] = { MAKE_FN(0, 0), MAKE_FN(0, 0), MAKE_FN(0, 0), MAKE_FN(0, 0) };      \
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
    if (thunk((DIRECT_CHAR), (STOREV_CHAR), 3, 2, v, 2, tau, t, 2) != 0 ||           \
        g_##SUFFIX##_fortran_call.called != 1 ||                                      \
        g_##SUFFIX##_fortran_call.direct != (DIRECT_CHAR) ||                          \
        g_##SUFFIX##_fortran_call.storev != (STOREV_CHAR) ||                          \
        g_##SUFFIX##_fortran_call.n != 3 ||                                           \
        g_##SUFFIX##_fortran_call.k != 2 ||                                           \
        g_##SUFFIX##_fortran_call.v != v ||                                           \
        g_##SUFFIX##_fortran_call.ldv != 2 ||                                         \
        g_##SUFFIX##_fortran_call.tau != tau ||                                       \
        g_##SUFFIX##_fortran_call.t != t ||                                           \
        g_##SUFFIX##_fortran_call.ldt != 2 ||                                         \
        !EQ_FN(t[0], MAKE_FN((BASE) + 1, (BASE) + 11)) ||                             \
        !EQ_FN(t[1], MAKE_FN((BASE) + 2, (BASE) + 12)) ||                             \
        !EQ_FN(t[3], MAKE_FN((BASE) + 3, (BASE) + 13))) {                             \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward complex LARZT inputs or outputs correctly\n"); \
        return 1;                                                                     \
    }                                                                                 \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards complex LARZT inputs and returns success\n"); \
    return 0;                                                                         \
}                                                                                     \
static int check_##SUFFIX##_cblas_to_fortran(void)                                    \
{                                                                                     \
    fb_backend_vtable_t vtable;                                                       \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                            \
    char direct = (DIRECT_CHAR);                                                      \
    char storev = (STOREV_CHAR);                                                      \
    int n = 3;                                                                        \
    int k = 2;                                                                        \
    int ldv = 2;                                                                      \
    int ldt = 2;                                                                      \
    CTYPE v[6] = { MAKE_FN(11, 21), MAKE_FN(12, 22), MAKE_FN(13, 23), MAKE_FN(14, 24), MAKE_FN(15, 25), MAKE_FN(16, 26) }; \
    CTYPE tau[2] = { MAKE_FN(17, 27), MAKE_FN(18, 28) };                              \
    CTYPE t[4] = { MAKE_FN(0, 0), MAKE_FN(0, 0), MAKE_FN(0, 0), MAKE_FN(0, 0) };      \
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
    thunk(&direct, &storev, &n, &k, v, &ldv, tau, t, &ldt);                          \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                        \
        g_##SUFFIX##_cblas_call.direct != (DIRECT_CHAR) ||                            \
        g_##SUFFIX##_cblas_call.storev != (STOREV_CHAR) ||                            \
        g_##SUFFIX##_cblas_call.n != 3 ||                                             \
        g_##SUFFIX##_cblas_call.k != 2 ||                                             \
        g_##SUFFIX##_cblas_call.v != v ||                                             \
        g_##SUFFIX##_cblas_call.ldv != 2 ||                                           \
        g_##SUFFIX##_cblas_call.tau != tau ||                                         \
        g_##SUFFIX##_cblas_call.t != t ||                                             \
        g_##SUFFIX##_cblas_call.ldt != 2 ||                                           \
        !EQ_FN(t[0], MAKE_FN((BASE) + 10, (BASE) + 20)) ||                            \
        !EQ_FN(t[1], MAKE_FN((BASE) + 11, (BASE) + 21)) ||                            \
        !EQ_FN(t[3], MAKE_FN((BASE) + 12, (BASE) + 22))) {                            \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference complex LARZT inputs or propagate outputs correctly\n"); \
        return 1;                                                                     \
    }                                                                                 \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences complex LARZT inputs and ignores the C int return\n"); \
    return 0;                                                                         \
}

DEFINE_LARZT_REAL_TESTS(slarzt, float, FB_OP_SLARZT, 100, 'B', 'R')
DEFINE_LARZT_REAL_TESTS(dlarzt, double, FB_OP_DLARZT, 300, 'B', 'R')
DEFINE_LARZT_COMPLEX_TESTS(clarzt, fb_complex_float_t, FB_OP_CLARZT, make_cf32,
                           cf32_eq, 500, 'B', 'R')
DEFINE_LARZT_COMPLEX_TESTS(zlarzt, fb_complex_double_t, FB_OP_ZLARZT, make_cf64,
                           cf64_eq, 700, 'B', 'R')

int main(void)
{
    int status = 0;

    status |= check_slarzt_fortran_to_cblas();
    status |= check_slarzt_cblas_to_fortran();
    status |= check_dlarzt_fortran_to_cblas();
    status |= check_dlarzt_cblas_to_fortran();
    status |= check_clarzt_fortran_to_cblas();
    status |= check_clarzt_cblas_to_fortran();
    status |= check_zlarzt_fortran_to_cblas();
    status |= check_zlarzt_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}