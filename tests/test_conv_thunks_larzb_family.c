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

#define DEFINE_LARZB_REAL_TESTS(SUFFIX, TYPE, OP_ID, BASE, SIDE_CHAR, TRANS_CHAR, DIRECT_CHAR, STOREV_CHAR) \
typedef int (*fb_##SUFFIX##_cblas_fn)(char side, char trans, char direct, char storev, int m, int n, int k, \
                                      int l, TYPE *v, int ldv, TYPE *t, int ldt, TYPE *c, int ldc, TYPE *work, \
                                      int ldwork);                                                  \
typedef void (*fb_##SUFFIX##_fortran_fn)(char *side, char *trans, char *direct, char *storev, int *m, int *n, \
                                         int *k, int *l, TYPE *v, int *ldv, TYPE *t, int *ldt, TYPE *c, \
                                         int *ldc, TYPE *work, int *ldwork);                       \
static struct {                                                                                      \
    int called;                                                                                      \
    char side;                                                                                       \
    char trans;                                                                                      \
    char direct;                                                                                     \
    char storev;                                                                                     \
    int m;                                                                                           \
    int n;                                                                                           \
    int k;                                                                                           \
    int l;                                                                                           \
    TYPE *v;                                                                                         \
    int ldv;                                                                                         \
    TYPE *t;                                                                                         \
    int ldt;                                                                                         \
    TYPE *c;                                                                                         \
    int ldc;                                                                                         \
    TYPE *work;                                                                                      \
    int ldwork;                                                                                      \
} g_##SUFFIX##_fortran_call;                                                                         \
static struct {                                                                                      \
    int called;                                                                                      \
    char side;                                                                                       \
    char trans;                                                                                      \
    char direct;                                                                                     \
    char storev;                                                                                     \
    int m;                                                                                           \
    int n;                                                                                           \
    int k;                                                                                           \
    int l;                                                                                           \
    TYPE *v;                                                                                         \
    int ldv;                                                                                         \
    TYPE *t;                                                                                         \
    int ldt;                                                                                         \
    TYPE *c;                                                                                         \
    int ldc;                                                                                         \
    TYPE *work;                                                                                      \
    int ldwork;                                                                                      \
} g_##SUFFIX##_cblas_call;                                                                           \
static void stub_##SUFFIX##_fortran(char *side, char *trans, char *direct, char *storev, int *m,    \
                                    int *n, int *k, int *l, TYPE *v, int *ldv, TYPE *t, int *ldt,   \
                                    TYPE *c, int *ldc, TYPE *work, int *ldwork)                      \
{                                                                                                    \
    g_##SUFFIX##_fortran_call.called += 1;                                                           \
    g_##SUFFIX##_fortran_call.side = *side;                                                          \
    g_##SUFFIX##_fortran_call.trans = *trans;                                                        \
    g_##SUFFIX##_fortran_call.direct = *direct;                                                      \
    g_##SUFFIX##_fortran_call.storev = *storev;                                                      \
    g_##SUFFIX##_fortran_call.m = *m;                                                                \
    g_##SUFFIX##_fortran_call.n = *n;                                                                \
    g_##SUFFIX##_fortran_call.k = *k;                                                                \
    g_##SUFFIX##_fortran_call.l = *l;                                                                \
    g_##SUFFIX##_fortran_call.v = v;                                                                 \
    g_##SUFFIX##_fortran_call.ldv = *ldv;                                                            \
    g_##SUFFIX##_fortran_call.t = t;                                                                 \
    g_##SUFFIX##_fortran_call.ldt = *ldt;                                                            \
    g_##SUFFIX##_fortran_call.c = c;                                                                 \
    g_##SUFFIX##_fortran_call.ldc = *ldc;                                                            \
    g_##SUFFIX##_fortran_call.work = work;                                                           \
    g_##SUFFIX##_fortran_call.ldwork = *ldwork;                                                      \
    c[0] = (TYPE)((BASE) + 1);                                                                       \
    c[3] = (TYPE)((BASE) + 2);                                                                       \
    work[0] = (TYPE)((BASE) + 3);                                                                    \
    work[1] = (TYPE)((BASE) + 4);                                                                    \
}                                                                                                    \
static int stub_##SUFFIX##_cblas(char side, char trans, char direct, char storev, int m, int n,     \
                                 int k, int l, TYPE *v, int ldv, TYPE *t, int ldt, TYPE *c,         \
                                 int ldc, TYPE *work, int ldwork)                                    \
{                                                                                                    \
    g_##SUFFIX##_cblas_call.called += 1;                                                             \
    g_##SUFFIX##_cblas_call.side = side;                                                             \
    g_##SUFFIX##_cblas_call.trans = trans;                                                           \
    g_##SUFFIX##_cblas_call.direct = direct;                                                         \
    g_##SUFFIX##_cblas_call.storev = storev;                                                         \
    g_##SUFFIX##_cblas_call.m = m;                                                                   \
    g_##SUFFIX##_cblas_call.n = n;                                                                   \
    g_##SUFFIX##_cblas_call.k = k;                                                                   \
    g_##SUFFIX##_cblas_call.l = l;                                                                   \
    g_##SUFFIX##_cblas_call.v = v;                                                                   \
    g_##SUFFIX##_cblas_call.ldv = ldv;                                                               \
    g_##SUFFIX##_cblas_call.t = t;                                                                   \
    g_##SUFFIX##_cblas_call.ldt = ldt;                                                               \
    g_##SUFFIX##_cblas_call.c = c;                                                                   \
    g_##SUFFIX##_cblas_call.ldc = ldc;                                                               \
    g_##SUFFIX##_cblas_call.work = work;                                                             \
    g_##SUFFIX##_cblas_call.ldwork = ldwork;                                                         \
    c[0] = (TYPE)((BASE) + 10);                                                                      \
    c[3] = (TYPE)((BASE) + 11);                                                                      \
    work[0] = (TYPE)((BASE) + 12);                                                                   \
    work[1] = (TYPE)((BASE) + 13);                                                                   \
    return (BASE) + 99;                                                                              \
}                                                                                                    \
static int check_##SUFFIX##_fortran_to_cblas(void)                                                   \
{                                                                                                    \
    fb_backend_vtable_t vtable;                                                                      \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                                             \
    TYPE v[4] = { (TYPE)1, (TYPE)2, (TYPE)3, (TYPE)4 };                                              \
    TYPE t[4] = { (TYPE)5, (TYPE)6, (TYPE)7, (TYPE)8 };                                              \
    TYPE c[4] = { (TYPE)0, (TYPE)0, (TYPE)0, (TYPE)0 };                                              \
    TYPE work[4] = { (TYPE)0, (TYPE)0, (TYPE)0, (TYPE)0 };                                           \
    memset(&vtable, 0, sizeof(vtable));                                                              \
    memset(&g_##SUFFIX##_fortran_call, 0, sizeof(g_##SUFFIX##_fortran_call));                        \
    vtable.ext_ops[OP_ID][FB_CONV_FORTRAN] =                                                         \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_fortran;                                      \
    fb_install_conv_thunks(&vtable, OP_ID);                                                          \
    thunk = (fb_##SUFFIX##_cblas_fn)vtable.ext_ops[OP_ID][FB_CONV_CBLAS];                            \
    if (!thunk) {                                                                                    \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk was not installed\n");          \
        return 1;                                                                                    \
    }                                                                                                \
    if (thunk((SIDE_CHAR), (TRANS_CHAR), (DIRECT_CHAR), (STOREV_CHAR), 2, 2, 1, 1, v, 2, t, 2, c, \
              2, work, 2) != 0 ||                                                                    \
        g_##SUFFIX##_fortran_call.called != 1 ||                                                     \
        g_##SUFFIX##_fortran_call.side != (SIDE_CHAR) ||                                             \
        g_##SUFFIX##_fortran_call.trans != (TRANS_CHAR) ||                                           \
        g_##SUFFIX##_fortran_call.direct != (DIRECT_CHAR) ||                                         \
        g_##SUFFIX##_fortran_call.storev != (STOREV_CHAR) ||                                         \
        g_##SUFFIX##_fortran_call.m != 2 ||                                                          \
        g_##SUFFIX##_fortran_call.n != 2 ||                                                          \
        g_##SUFFIX##_fortran_call.k != 1 ||                                                          \
        g_##SUFFIX##_fortran_call.l != 1 ||                                                          \
        g_##SUFFIX##_fortran_call.v != v ||                                                          \
        g_##SUFFIX##_fortran_call.ldv != 2 ||                                                        \
        g_##SUFFIX##_fortran_call.t != t ||                                                          \
        g_##SUFFIX##_fortran_call.ldt != 2 ||                                                        \
        g_##SUFFIX##_fortran_call.c != c ||                                                          \
        g_##SUFFIX##_fortran_call.ldc != 2 ||                                                        \
        g_##SUFFIX##_fortran_call.work != work ||                                                    \
        g_##SUFFIX##_fortran_call.ldwork != 2 ||                                                     \
        c[0] != (TYPE)((BASE) + 1) || c[3] != (TYPE)((BASE) + 2) ||                                 \
        work[0] != (TYPE)((BASE) + 3) || work[1] != (TYPE)((BASE) + 4)) {                           \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward LARZB inputs or outputs correctly\n"); \
        return 1;                                                                                    \
    }                                                                                                \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards LARZB inputs and returns success\n"); \
    return 0;                                                                                        \
}                                                                                                    \
static int check_##SUFFIX##_cblas_to_fortran(void)                                                   \
{                                                                                                    \
    fb_backend_vtable_t vtable;                                                                      \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                                           \
    char side = (SIDE_CHAR);                                                                         \
    char trans = (TRANS_CHAR);                                                                       \
    char direct = (DIRECT_CHAR);                                                                     \
    char storev = (STOREV_CHAR);                                                                     \
    int m = 2;                                                                                       \
    int n = 2;                                                                                       \
    int k = 1;                                                                                       \
    int l = 1;                                                                                       \
    int ldv = 2;                                                                                     \
    int ldt = 2;                                                                                     \
    int ldc = 2;                                                                                     \
    int ldwork = 2;                                                                                  \
    TYPE v[4] = { (TYPE)11, (TYPE)12, (TYPE)13, (TYPE)14 };                                          \
    TYPE t[4] = { (TYPE)15, (TYPE)16, (TYPE)17, (TYPE)18 };                                          \
    TYPE c[4] = { (TYPE)0, (TYPE)0, (TYPE)0, (TYPE)0 };                                              \
    TYPE work[4] = { (TYPE)0, (TYPE)0, (TYPE)0, (TYPE)0 };                                           \
    memset(&vtable, 0, sizeof(vtable));                                                              \
    memset(&g_##SUFFIX##_cblas_call, 0, sizeof(g_##SUFFIX##_cblas_call));                            \
    vtable.ext_ops[OP_ID][FB_CONV_CBLAS] =                                                           \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_cblas;                                        \
    fb_install_conv_thunks(&vtable, OP_ID);                                                          \
    thunk = (fb_##SUFFIX##_fortran_fn)vtable.ext_ops[OP_ID][FB_CONV_FORTRAN];                        \
    if (!thunk) {                                                                                    \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk was not installed\n");          \
        return 1;                                                                                    \
    }                                                                                                \
    thunk(&side, &trans, &direct, &storev, &m, &n, &k, &l, v, &ldv, t, &ldt, c, &ldc, work, &ldwork); \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                                       \
        g_##SUFFIX##_cblas_call.side != (SIDE_CHAR) ||                                               \
        g_##SUFFIX##_cblas_call.trans != (TRANS_CHAR) ||                                             \
        g_##SUFFIX##_cblas_call.direct != (DIRECT_CHAR) ||                                           \
        g_##SUFFIX##_cblas_call.storev != (STOREV_CHAR) ||                                           \
        g_##SUFFIX##_cblas_call.m != 2 ||                                                            \
        g_##SUFFIX##_cblas_call.n != 2 ||                                                            \
        g_##SUFFIX##_cblas_call.k != 1 ||                                                            \
        g_##SUFFIX##_cblas_call.l != 1 ||                                                            \
        g_##SUFFIX##_cblas_call.v != v ||                                                            \
        g_##SUFFIX##_cblas_call.ldv != 2 ||                                                          \
        g_##SUFFIX##_cblas_call.t != t ||                                                            \
        g_##SUFFIX##_cblas_call.ldt != 2 ||                                                          \
        g_##SUFFIX##_cblas_call.c != c ||                                                            \
        g_##SUFFIX##_cblas_call.ldc != 2 ||                                                          \
        g_##SUFFIX##_cblas_call.work != work ||                                                      \
        g_##SUFFIX##_cblas_call.ldwork != 2 ||                                                       \
        c[0] != (TYPE)((BASE) + 10) || c[3] != (TYPE)((BASE) + 11) ||                               \
        work[0] != (TYPE)((BASE) + 12) || work[1] != (TYPE)((BASE) + 13)) {                         \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference LARZB inputs or propagate outputs correctly\n"); \
        return 1;                                                                                    \
    }                                                                                                \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences LARZB inputs and ignores the C int return\n"); \
    return 0;                                                                                        \
}

#define DEFINE_LARZB_COMPLEX_TESTS(SUFFIX, CTYPE, OP_ID, MAKE_FN, EQ_FN, BASE, SIDE_CHAR, TRANS_CHAR, DIRECT_CHAR, STOREV_CHAR) \
typedef int (*fb_##SUFFIX##_cblas_fn)(char side, char trans, char direct, char storev, int m, int n, int k, \
                                      int l, CTYPE *v, int ldv, CTYPE *t, int ldt, CTYPE *c, int ldc, CTYPE *work, \
                                      int ldwork);                                                 \
typedef void (*fb_##SUFFIX##_fortran_fn)(char *side, char *trans, char *direct, char *storev, int *m, int *n, \
                                         int *k, int *l, CTYPE *v, int *ldv, CTYPE *t, int *ldt, CTYPE *c, \
                                         int *ldc, CTYPE *work, int *ldwork);                      \
static struct {                                                                                      \
    int called;                                                                                      \
    char side;                                                                                       \
    char trans;                                                                                      \
    char direct;                                                                                     \
    char storev;                                                                                     \
    int m;                                                                                           \
    int n;                                                                                           \
    int k;                                                                                           \
    int l;                                                                                           \
    CTYPE *v;                                                                                        \
    int ldv;                                                                                         \
    CTYPE *t;                                                                                        \
    int ldt;                                                                                         \
    CTYPE *c;                                                                                        \
    int ldc;                                                                                         \
    CTYPE *work;                                                                                     \
    int ldwork;                                                                                      \
} g_##SUFFIX##_fortran_call;                                                                         \
static struct {                                                                                      \
    int called;                                                                                      \
    char side;                                                                                       \
    char trans;                                                                                      \
    char direct;                                                                                     \
    char storev;                                                                                     \
    int m;                                                                                           \
    int n;                                                                                           \
    int k;                                                                                           \
    int l;                                                                                           \
    CTYPE *v;                                                                                        \
    int ldv;                                                                                         \
    CTYPE *t;                                                                                        \
    int ldt;                                                                                         \
    CTYPE *c;                                                                                        \
    int ldc;                                                                                         \
    CTYPE *work;                                                                                     \
    int ldwork;                                                                                      \
} g_##SUFFIX##_cblas_call;                                                                           \
static void stub_##SUFFIX##_fortran(char *side, char *trans, char *direct, char *storev, int *m,    \
                                    int *n, int *k, int *l, CTYPE *v, int *ldv, CTYPE *t, int *ldt, \
                                    CTYPE *c, int *ldc, CTYPE *work, int *ldwork)                   \
{                                                                                                    \
    g_##SUFFIX##_fortran_call.called += 1;                                                           \
    g_##SUFFIX##_fortran_call.side = *side;                                                          \
    g_##SUFFIX##_fortran_call.trans = *trans;                                                        \
    g_##SUFFIX##_fortran_call.direct = *direct;                                                      \
    g_##SUFFIX##_fortran_call.storev = *storev;                                                      \
    g_##SUFFIX##_fortran_call.m = *m;                                                                \
    g_##SUFFIX##_fortran_call.n = *n;                                                                \
    g_##SUFFIX##_fortran_call.k = *k;                                                                \
    g_##SUFFIX##_fortran_call.l = *l;                                                                \
    g_##SUFFIX##_fortran_call.v = v;                                                                 \
    g_##SUFFIX##_fortran_call.ldv = *ldv;                                                            \
    g_##SUFFIX##_fortran_call.t = t;                                                                 \
    g_##SUFFIX##_fortran_call.ldt = *ldt;                                                            \
    g_##SUFFIX##_fortran_call.c = c;                                                                 \
    g_##SUFFIX##_fortran_call.ldc = *ldc;                                                            \
    g_##SUFFIX##_fortran_call.work = work;                                                           \
    g_##SUFFIX##_fortran_call.ldwork = *ldwork;                                                      \
    c[0] = MAKE_FN((BASE) + 1, (BASE) + 11);                                                         \
    c[3] = MAKE_FN((BASE) + 2, (BASE) + 12);                                                         \
    work[0] = MAKE_FN((BASE) + 3, (BASE) + 13);                                                      \
    work[1] = MAKE_FN((BASE) + 4, (BASE) + 14);                                                      \
}                                                                                                    \
static int stub_##SUFFIX##_cblas(char side, char trans, char direct, char storev, int m, int n,     \
                                 int k, int l, CTYPE *v, int ldv, CTYPE *t, int ldt, CTYPE *c,      \
                                 int ldc, CTYPE *work, int ldwork)                                   \
{                                                                                                    \
    g_##SUFFIX##_cblas_call.called += 1;                                                             \
    g_##SUFFIX##_cblas_call.side = side;                                                             \
    g_##SUFFIX##_cblas_call.trans = trans;                                                           \
    g_##SUFFIX##_cblas_call.direct = direct;                                                         \
    g_##SUFFIX##_cblas_call.storev = storev;                                                         \
    g_##SUFFIX##_cblas_call.m = m;                                                                   \
    g_##SUFFIX##_cblas_call.n = n;                                                                   \
    g_##SUFFIX##_cblas_call.k = k;                                                                   \
    g_##SUFFIX##_cblas_call.l = l;                                                                   \
    g_##SUFFIX##_cblas_call.v = v;                                                                   \
    g_##SUFFIX##_cblas_call.ldv = ldv;                                                               \
    g_##SUFFIX##_cblas_call.t = t;                                                                   \
    g_##SUFFIX##_cblas_call.ldt = ldt;                                                               \
    g_##SUFFIX##_cblas_call.c = c;                                                                   \
    g_##SUFFIX##_cblas_call.ldc = ldc;                                                               \
    g_##SUFFIX##_cblas_call.work = work;                                                             \
    g_##SUFFIX##_cblas_call.ldwork = ldwork;                                                         \
    c[0] = MAKE_FN((BASE) + 10, (BASE) + 20);                                                        \
    c[3] = MAKE_FN((BASE) + 11, (BASE) + 21);                                                        \
    work[0] = MAKE_FN((BASE) + 12, (BASE) + 22);                                                     \
    work[1] = MAKE_FN((BASE) + 13, (BASE) + 23);                                                     \
    return (BASE) + 99;                                                                              \
}                                                                                                    \
static int check_##SUFFIX##_fortran_to_cblas(void)                                                   \
{                                                                                                    \
    fb_backend_vtable_t vtable;                                                                      \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                                             \
    CTYPE v[4] = { MAKE_FN(1, 11), MAKE_FN(2, 12), MAKE_FN(3, 13), MAKE_FN(4, 14) };                \
    CTYPE t[4] = { MAKE_FN(5, 15), MAKE_FN(6, 16), MAKE_FN(7, 17), MAKE_FN(8, 18) };                \
    CTYPE c[4] = { MAKE_FN(0, 0), MAKE_FN(0, 0), MAKE_FN(0, 0), MAKE_FN(0, 0) };                    \
    CTYPE work[4] = { MAKE_FN(0, 0), MAKE_FN(0, 0), MAKE_FN(0, 0), MAKE_FN(0, 0) };                 \
    memset(&vtable, 0, sizeof(vtable));                                                              \
    memset(&g_##SUFFIX##_fortran_call, 0, sizeof(g_##SUFFIX##_fortran_call));                        \
    vtable.ext_ops[OP_ID][FB_CONV_FORTRAN] =                                                         \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_fortran;                                      \
    fb_install_conv_thunks(&vtable, OP_ID);                                                          \
    thunk = (fb_##SUFFIX##_cblas_fn)vtable.ext_ops[OP_ID][FB_CONV_CBLAS];                            \
    if (!thunk) {                                                                                    \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk was not installed\n");          \
        return 1;                                                                                    \
    }                                                                                                \
    if (thunk((SIDE_CHAR), (TRANS_CHAR), (DIRECT_CHAR), (STOREV_CHAR), 2, 2, 1, 1, v, 2, t, 2, c, \
              2, work, 2) != 0 ||                                                                    \
        g_##SUFFIX##_fortran_call.called != 1 ||                                                     \
        g_##SUFFIX##_fortran_call.side != (SIDE_CHAR) ||                                             \
        g_##SUFFIX##_fortran_call.trans != (TRANS_CHAR) ||                                           \
        g_##SUFFIX##_fortran_call.direct != (DIRECT_CHAR) ||                                         \
        g_##SUFFIX##_fortran_call.storev != (STOREV_CHAR) ||                                         \
        g_##SUFFIX##_fortran_call.m != 2 ||                                                          \
        g_##SUFFIX##_fortran_call.n != 2 ||                                                          \
        g_##SUFFIX##_fortran_call.k != 1 ||                                                          \
        g_##SUFFIX##_fortran_call.l != 1 ||                                                          \
        g_##SUFFIX##_fortran_call.v != v ||                                                          \
        g_##SUFFIX##_fortran_call.ldv != 2 ||                                                        \
        g_##SUFFIX##_fortran_call.t != t ||                                                          \
        g_##SUFFIX##_fortran_call.ldt != 2 ||                                                        \
        g_##SUFFIX##_fortran_call.c != c ||                                                          \
        g_##SUFFIX##_fortran_call.ldc != 2 ||                                                        \
        g_##SUFFIX##_fortran_call.work != work ||                                                    \
        g_##SUFFIX##_fortran_call.ldwork != 2 ||                                                     \
        !EQ_FN(c[0], MAKE_FN((BASE) + 1, (BASE) + 11)) ||                                            \
        !EQ_FN(c[3], MAKE_FN((BASE) + 2, (BASE) + 12)) ||                                            \
        !EQ_FN(work[0], MAKE_FN((BASE) + 3, (BASE) + 13)) ||                                         \
        !EQ_FN(work[1], MAKE_FN((BASE) + 4, (BASE) + 14))) {                                         \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward complex LARZB inputs or outputs correctly\n"); \
        return 1;                                                                                    \
    }                                                                                                \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards complex LARZB inputs and returns success\n"); \
    return 0;                                                                                        \
}                                                                                                    \
static int check_##SUFFIX##_cblas_to_fortran(void)                                                   \
{                                                                                                    \
    fb_backend_vtable_t vtable;                                                                      \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                                           \
    char side = (SIDE_CHAR);                                                                         \
    char trans = (TRANS_CHAR);                                                                       \
    char direct = (DIRECT_CHAR);                                                                     \
    char storev = (STOREV_CHAR);                                                                     \
    int m = 2;                                                                                       \
    int n = 2;                                                                                       \
    int k = 1;                                                                                       \
    int l = 1;                                                                                       \
    int ldv = 2;                                                                                     \
    int ldt = 2;                                                                                     \
    int ldc = 2;                                                                                     \
    int ldwork = 2;                                                                                  \
    CTYPE v[4] = { MAKE_FN(11, 21), MAKE_FN(12, 22), MAKE_FN(13, 23), MAKE_FN(14, 24) };            \
    CTYPE t[4] = { MAKE_FN(15, 25), MAKE_FN(16, 26), MAKE_FN(17, 27), MAKE_FN(18, 28) };            \
    CTYPE c[4] = { MAKE_FN(0, 0), MAKE_FN(0, 0), MAKE_FN(0, 0), MAKE_FN(0, 0) };                    \
    CTYPE work[4] = { MAKE_FN(0, 0), MAKE_FN(0, 0), MAKE_FN(0, 0), MAKE_FN(0, 0) };                 \
    memset(&vtable, 0, sizeof(vtable));                                                              \
    memset(&g_##SUFFIX##_cblas_call, 0, sizeof(g_##SUFFIX##_cblas_call));                            \
    vtable.ext_ops[OP_ID][FB_CONV_CBLAS] =                                                           \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_cblas;                                        \
    fb_install_conv_thunks(&vtable, OP_ID);                                                          \
    thunk = (fb_##SUFFIX##_fortran_fn)vtable.ext_ops[OP_ID][FB_CONV_FORTRAN];                        \
    if (!thunk) {                                                                                    \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk was not installed\n");          \
        return 1;                                                                                    \
    }                                                                                                \
    thunk(&side, &trans, &direct, &storev, &m, &n, &k, &l, v, &ldv, t, &ldt, c, &ldc, work, &ldwork); \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                                       \
        g_##SUFFIX##_cblas_call.side != (SIDE_CHAR) ||                                               \
        g_##SUFFIX##_cblas_call.trans != (TRANS_CHAR) ||                                             \
        g_##SUFFIX##_cblas_call.direct != (DIRECT_CHAR) ||                                           \
        g_##SUFFIX##_cblas_call.storev != (STOREV_CHAR) ||                                           \
        g_##SUFFIX##_cblas_call.m != 2 ||                                                            \
        g_##SUFFIX##_cblas_call.n != 2 ||                                                            \
        g_##SUFFIX##_cblas_call.k != 1 ||                                                            \
        g_##SUFFIX##_cblas_call.l != 1 ||                                                            \
        g_##SUFFIX##_cblas_call.v != v ||                                                            \
        g_##SUFFIX##_cblas_call.ldv != 2 ||                                                          \
        g_##SUFFIX##_cblas_call.t != t ||                                                            \
        g_##SUFFIX##_cblas_call.ldt != 2 ||                                                          \
        g_##SUFFIX##_cblas_call.c != c ||                                                            \
        g_##SUFFIX##_cblas_call.ldc != 2 ||                                                          \
        g_##SUFFIX##_cblas_call.work != work ||                                                      \
        g_##SUFFIX##_cblas_call.ldwork != 2 ||                                                       \
        !EQ_FN(c[0], MAKE_FN((BASE) + 10, (BASE) + 20)) ||                                           \
        !EQ_FN(c[3], MAKE_FN((BASE) + 11, (BASE) + 21)) ||                                           \
        !EQ_FN(work[0], MAKE_FN((BASE) + 12, (BASE) + 22)) ||                                        \
        !EQ_FN(work[1], MAKE_FN((BASE) + 13, (BASE) + 23))) {                                        \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference complex LARZB inputs or propagate outputs correctly\n"); \
        return 1;                                                                                    \
    }                                                                                                \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences complex LARZB inputs and ignores the C int return\n"); \
    return 0;                                                                                        \
}

DEFINE_LARZB_REAL_TESTS(slarzb, float, FB_OP_SLARZB, 100, 'L', 'N', 'B', 'R')
DEFINE_LARZB_REAL_TESTS(dlarzb, double, FB_OP_DLARZB, 300, 'R', 'T', 'B', 'R')
DEFINE_LARZB_COMPLEX_TESTS(clarzb, fb_complex_float_t, FB_OP_CLARZB, make_cf32,
                           cf32_eq, 500, 'L', 'C', 'B', 'R')
DEFINE_LARZB_COMPLEX_TESTS(zlarzb, fb_complex_double_t, FB_OP_ZLARZB, make_cf64,
                           cf64_eq, 700, 'R', 'C', 'B', 'R')

int main(void)
{
    int status = 0;

    status |= check_slarzb_fortran_to_cblas();
    status |= check_slarzb_cblas_to_fortran();
    status |= check_dlarzb_fortran_to_cblas();
    status |= check_dlarzb_cblas_to_fortran();
    status |= check_clarzb_fortran_to_cblas();
    status |= check_clarzb_cblas_to_fortran();
    status |= check_zlarzb_fortran_to_cblas();
    status |= check_zlarzb_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}