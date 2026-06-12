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

#define DEFINE_LARFB_REAL_TESTS(SUFFIX, TYPE, OP_ID, BASE)                      \
typedef int (*fb_##SUFFIX##_cblas_fn)(fb_layout_t layout, fb_side_t side,      \
                                      fb_transpose_t trans, char direct,        \
                                      char storev, int m, int n, int k,         \
                                      const TYPE *v, int ldv, const TYPE *t,    \
                                      int ldt, TYPE *c, int ldc);               \
typedef void (*fb_##SUFFIX##_fortran_fn)(char *side, char *trans, char *direct, \
                                         char *storev, int *m, int *n, int *k,  \
                                         TYPE *v, int *ldv, TYPE *t, int *ldt,  \
                                         TYPE *c, int *ldc, TYPE *work,         \
                                         int *ldwork, int *info);               \
static struct {                                                                  \
    int called;                                                                  \
    char side;                                                                   \
    char trans;                                                                  \
    char direct;                                                                 \
    char storev;                                                                 \
    int m;                                                                       \
    int n;                                                                       \
    int k;                                                                       \
    TYPE *v;                                                                     \
    int ldv;                                                                     \
    TYPE *t;                                                                     \
    int ldt;                                                                     \
    TYPE *c;                                                                     \
    int ldc;                                                                     \
    TYPE *work;                                                                  \
    int ldwork;                                                                  \
    TYPE t_snapshot[4];                                                          \
    TYPE c_snapshot[4];                                                          \
} g_##SUFFIX##_fortran_call;                                                     \
static struct {                                                                  \
    int called;                                                                  \
    fb_layout_t layout;                                                          \
    fb_side_t side;                                                              \
    fb_transpose_t trans;                                                        \
    char direct;                                                                 \
    char storev;                                                                 \
    int m;                                                                       \
    int n;                                                                       \
    int k;                                                                       \
    const TYPE *v;                                                               \
    int ldv;                                                                     \
    const TYPE *t;                                                               \
    int ldt;                                                                     \
    TYPE *c;                                                                     \
    int ldc;                                                                     \
} g_##SUFFIX##_cblas_call;                                                       \
static void stub_##SUFFIX##_fortran(char *side, char *trans, char *direct,      \
                                    char *storev, int *m, int *n, int *k,       \
                                    TYPE *v, int *ldv, TYPE *t, int *ldt,       \
                                    TYPE *c, int *ldc, TYPE *work,              \
                                    int *ldwork)                                \
{                                                                                \
    g_##SUFFIX##_fortran_call.called += 1;                                       \
    g_##SUFFIX##_fortran_call.side = *side;                                      \
    g_##SUFFIX##_fortran_call.trans = *trans;                                    \
    g_##SUFFIX##_fortran_call.direct = *direct;                                  \
    g_##SUFFIX##_fortran_call.storev = *storev;                                  \
    g_##SUFFIX##_fortran_call.m = *m;                                            \
    g_##SUFFIX##_fortran_call.n = *n;                                            \
    g_##SUFFIX##_fortran_call.k = *k;                                            \
    g_##SUFFIX##_fortran_call.v = v;                                             \
    g_##SUFFIX##_fortran_call.ldv = *ldv;                                        \
    g_##SUFFIX##_fortran_call.t = t;                                             \
    g_##SUFFIX##_fortran_call.ldt = *ldt;                                        \
    g_##SUFFIX##_fortran_call.c = c;                                             \
    g_##SUFFIX##_fortran_call.ldc = *ldc;                                        \
    g_##SUFFIX##_fortran_call.work = work;                                       \
    g_##SUFFIX##_fortran_call.ldwork = *ldwork;                                  \
    g_##SUFFIX##_fortran_call.t_snapshot[0] = t[0];                              \
    g_##SUFFIX##_fortran_call.t_snapshot[1] = t[1];                              \
    g_##SUFFIX##_fortran_call.t_snapshot[2] = t[2];                              \
    g_##SUFFIX##_fortran_call.t_snapshot[3] = t[3];                              \
    g_##SUFFIX##_fortran_call.c_snapshot[0] = c[0];                              \
    g_##SUFFIX##_fortran_call.c_snapshot[1] = c[1];                              \
    g_##SUFFIX##_fortran_call.c_snapshot[2] = c[2];                              \
    g_##SUFFIX##_fortran_call.c_snapshot[3] = c[3];                              \
    c[0] = (TYPE)((BASE) + 1);                                                   \
    c[1] = (TYPE)((BASE) + 2);                                                   \
    c[2] = (TYPE)((BASE) + 3);                                                   \
    c[3] = (TYPE)((BASE) + 4);                                                   \
}                                                                                \
static int stub_##SUFFIX##_cblas(fb_layout_t layout, fb_side_t side,            \
                                 fb_transpose_t trans, char direct,             \
                                 char storev, int m, int n, int k,             \
                                 const TYPE *v, int ldv, const TYPE *t,        \
                                 int ldt, TYPE *c, int ldc)                    \
{                                                                                \
    g_##SUFFIX##_cblas_call.called += 1;                                         \
    g_##SUFFIX##_cblas_call.layout = layout;                                     \
    g_##SUFFIX##_cblas_call.side = side;                                         \
    g_##SUFFIX##_cblas_call.trans = trans;                                       \
    g_##SUFFIX##_cblas_call.direct = direct;                                     \
    g_##SUFFIX##_cblas_call.storev = storev;                                     \
    g_##SUFFIX##_cblas_call.m = m;                                               \
    g_##SUFFIX##_cblas_call.n = n;                                               \
    g_##SUFFIX##_cblas_call.k = k;                                               \
    g_##SUFFIX##_cblas_call.v = v;                                               \
    g_##SUFFIX##_cblas_call.ldv = ldv;                                           \
    g_##SUFFIX##_cblas_call.t = t;                                               \
    g_##SUFFIX##_cblas_call.ldt = ldt;                                           \
    g_##SUFFIX##_cblas_call.c = c;                                               \
    g_##SUFFIX##_cblas_call.ldc = ldc;                                           \
    c[0] = (TYPE)((BASE) + 10);                                                  \
    c[1] = (TYPE)((BASE) + 11);                                                  \
    c[2] = (TYPE)((BASE) + 12);                                                  \
    c[3] = (TYPE)((BASE) + 13);                                                  \
    return (BASE) + 99;                                                          \
}                                                                                \
static int check_##SUFFIX##_fortran_to_cblas(void)                               \
{                                                                                \
    fb_backend_vtable_t vtable;                                                  \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                         \
    TYPE v_row[4] = { (TYPE)1, (TYPE)2, (TYPE)3, (TYPE)4 };                     \
    TYPE t_row[4] = { (TYPE)5, (TYPE)6, (TYPE)7, (TYPE)8 };                     \
    TYPE c_row[4] = { (TYPE)9, (TYPE)10, (TYPE)11, (TYPE)12 };                  \
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
    if (thunk(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_NO_TRANS, 'F', 'C', 2, 2, 2,     \
              v_row, 2, t_row, 2, c_row, 2) != 0 ||                             \
        g_##SUFFIX##_fortran_call.called != 1 ||                                 \
        g_##SUFFIX##_fortran_call.side != 'L' ||                                 \
        g_##SUFFIX##_fortran_call.trans != 'N' ||                                \
        g_##SUFFIX##_fortran_call.direct != 'F' ||                               \
        g_##SUFFIX##_fortran_call.storev != 'C' ||                               \
        g_##SUFFIX##_fortran_call.m != 2 ||                                      \
        g_##SUFFIX##_fortran_call.n != 2 ||                                      \
        g_##SUFFIX##_fortran_call.k != 2 ||                                      \
        g_##SUFFIX##_fortran_call.v == v_row ||                                  \
        g_##SUFFIX##_fortran_call.t == t_row ||                                  \
        g_##SUFFIX##_fortran_call.c == c_row ||                                  \
        g_##SUFFIX##_fortran_call.ldv != 2 ||                                    \
        g_##SUFFIX##_fortran_call.ldt != 2 ||                                    \
        g_##SUFFIX##_fortran_call.ldc != 2 ||                                    \
        g_##SUFFIX##_fortran_call.ldwork != 2 ||                                 \
        g_##SUFFIX##_fortran_call.work == NULL ||                                \
        g_##SUFFIX##_fortran_call.t_snapshot[0] != (TYPE)5 ||                    \
        g_##SUFFIX##_fortran_call.t_snapshot[1] != (TYPE)7 ||                    \
        g_##SUFFIX##_fortran_call.t_snapshot[2] != (TYPE)6 ||                    \
        g_##SUFFIX##_fortran_call.t_snapshot[3] != (TYPE)8 ||                    \
        g_##SUFFIX##_fortran_call.c_snapshot[0] != (TYPE)9 ||                    \
        g_##SUFFIX##_fortran_call.c_snapshot[1] != (TYPE)11 ||                   \
        g_##SUFFIX##_fortran_call.c_snapshot[2] != (TYPE)10 ||                   \
        g_##SUFFIX##_fortran_call.c_snapshot[3] != (TYPE)12 ||                   \
        c_row[0] != (TYPE)((BASE) + 1) || c_row[1] != (TYPE)((BASE) + 3) ||     \
        c_row[2] != (TYPE)((BASE) + 2) || c_row[3] != (TYPE)((BASE) + 4)) {     \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not stage row-major LARFB inputs or outputs correctly\n"); \
        return 1;                                                                \
    }                                                                            \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk stages row-major LARFB inputs and returns success\n"); \
    return 0;                                                                    \
}                                                                                \
static int check_##SUFFIX##_cblas_to_fortran(void)                               \
{                                                                                \
    fb_backend_vtable_t vtable;                                                  \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                       \
    char side = 'R';                                                             \
    char trans = 'T';                                                            \
    char direct = 'B';                                                           \
    char storev = 'R';                                                           \
    int m = 2;                                                                   \
    int n = 2;                                                                   \
    int k = 2;                                                                   \
    int ldv = 2;                                                                 \
    int ldt = 2;                                                                 \
    int ldc = 2;                                                                 \
    int ldwork = 2;                                                              \
    int info = -1;                                                               \
    TYPE v[4] = { (TYPE)21, (TYPE)22, (TYPE)23, (TYPE)24 };                     \
    TYPE t[4] = { (TYPE)25, (TYPE)26, (TYPE)27, (TYPE)28 };                     \
    TYPE c[4] = { (TYPE)29, (TYPE)30, (TYPE)31, (TYPE)32 };                     \
    TYPE work[4] = { (TYPE)0, (TYPE)0, (TYPE)0, (TYPE)0 };                      \
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
    thunk(&side, &trans, &direct, &storev, &m, &n, &k, v, &ldv, t, &ldt, c, &ldc, work, &ldwork, &info); \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                   \
        g_##SUFFIX##_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||                 \
        g_##SUFFIX##_cblas_call.side != FB_RIGHT ||                              \
        g_##SUFFIX##_cblas_call.trans != FB_TRANS ||                             \
        g_##SUFFIX##_cblas_call.direct != 'B' ||                                 \
        g_##SUFFIX##_cblas_call.storev != 'R' ||                                 \
        g_##SUFFIX##_cblas_call.m != 2 ||                                        \
        g_##SUFFIX##_cblas_call.n != 2 ||                                        \
        g_##SUFFIX##_cblas_call.k != 2 ||                                        \
        g_##SUFFIX##_cblas_call.v != v ||                                        \
        g_##SUFFIX##_cblas_call.ldv != 2 ||                                      \
        g_##SUFFIX##_cblas_call.t != t ||                                        \
        g_##SUFFIX##_cblas_call.ldt != 2 ||                                      \
        g_##SUFFIX##_cblas_call.c != c ||                                        \
        g_##SUFFIX##_cblas_call.ldc != 2 ||                                      \
        info != (BASE) + 99 ||                                                   \
        c[0] != (TYPE)((BASE) + 10) || c[1] != (TYPE)((BASE) + 11) ||           \
        c[2] != (TYPE)((BASE) + 12) || c[3] != (TYPE)((BASE) + 13)) {           \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not forward LARFB inputs or store info correctly\n"); \
        return 1;                                                                \
    }                                                                            \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences LARFB inputs and stores info\n"); \
    return 0;                                                                    \
}

#define DEFINE_LARFB_COMPLEX_TESTS(SUFFIX, CTYPE, OP_ID, MAKE_FN, EQ_FN, BASE) \
typedef int (*fb_##SUFFIX##_cblas_fn)(fb_layout_t layout, fb_side_t side,      \
                                      fb_transpose_t trans, char direct,        \
                                      char storev, int m, int n, int k,         \
                                      const CTYPE *v, int ldv, const CTYPE *t,  \
                                      int ldt, CTYPE *c, int ldc);              \
typedef void (*fb_##SUFFIX##_fortran_fn)(char *side, char *trans, char *direct, \
                                         char *storev, int *m, int *n, int *k,  \
                                         CTYPE *v, int *ldv, CTYPE *t, int *ldt, \
                                         CTYPE *c, int *ldc, CTYPE *work, \
                                         int *ldwork, int *info);               \
static struct {                                                                  \
    int called;                                                                  \
    char side;                                                                   \
    char trans;                                                                  \
    char direct;                                                                 \
    char storev;                                                                 \
    int m;                                                                       \
    int n;                                                                       \
    int k;                                                                       \
    CTYPE *v;                                                                    \
    int ldv;                                                                     \
    CTYPE *t;                                                                    \
    int ldt;                                                                     \
    CTYPE *c;                                                                    \
    int ldc;                                                                     \
    CTYPE *work;                                                                 \
    int ldwork;                                                                  \
    CTYPE t_snapshot[4];                                                         \
    CTYPE c_snapshot[4];                                                         \
} g_##SUFFIX##_fortran_call;                                                     \
static struct {                                                                  \
    int called;                                                                  \
    fb_layout_t layout;                                                          \
    fb_side_t side;                                                              \
    fb_transpose_t trans;                                                        \
    char direct;                                                                 \
    char storev;                                                                 \
    int m;                                                                       \
    int n;                                                                       \
    int k;                                                                       \
    const CTYPE *v;                                                              \
    int ldv;                                                                     \
    const CTYPE *t;                                                              \
    int ldt;                                                                     \
    CTYPE *c;                                                                    \
    int ldc;                                                                     \
} g_##SUFFIX##_cblas_call;                                                       \
static void stub_##SUFFIX##_fortran(char *side, char *trans, char *direct,      \
                                    char *storev, int *m, int *n, int *k,       \
                                    CTYPE *v, int *ldv, CTYPE *t, int *ldt,     \
                                    CTYPE *c, int *ldc, CTYPE *work,            \
                                    int *ldwork)                                \
{                                                                                \
    g_##SUFFIX##_fortran_call.called += 1;                                       \
    g_##SUFFIX##_fortran_call.side = *side;                                      \
    g_##SUFFIX##_fortran_call.trans = *trans;                                    \
    g_##SUFFIX##_fortran_call.direct = *direct;                                  \
    g_##SUFFIX##_fortran_call.storev = *storev;                                  \
    g_##SUFFIX##_fortran_call.m = *m;                                            \
    g_##SUFFIX##_fortran_call.n = *n;                                            \
    g_##SUFFIX##_fortran_call.k = *k;                                            \
    g_##SUFFIX##_fortran_call.v = v;                                             \
    g_##SUFFIX##_fortran_call.ldv = *ldv;                                        \
    g_##SUFFIX##_fortran_call.t = t;                                             \
    g_##SUFFIX##_fortran_call.ldt = *ldt;                                        \
    g_##SUFFIX##_fortran_call.c = c;                                             \
    g_##SUFFIX##_fortran_call.ldc = *ldc;                                        \
    g_##SUFFIX##_fortran_call.work = work;                                       \
    g_##SUFFIX##_fortran_call.ldwork = *ldwork;                                  \
    g_##SUFFIX##_fortran_call.t_snapshot[0] = t[0];                              \
    g_##SUFFIX##_fortran_call.t_snapshot[1] = t[1];                              \
    g_##SUFFIX##_fortran_call.t_snapshot[2] = t[2];                              \
    g_##SUFFIX##_fortran_call.t_snapshot[3] = t[3];                              \
    g_##SUFFIX##_fortran_call.c_snapshot[0] = c[0];                              \
    g_##SUFFIX##_fortran_call.c_snapshot[1] = c[1];                              \
    g_##SUFFIX##_fortran_call.c_snapshot[2] = c[2];                              \
    g_##SUFFIX##_fortran_call.c_snapshot[3] = c[3];                              \
    c[0] = MAKE_FN((BASE) + 1, (BASE) + 11);                                     \
    c[1] = MAKE_FN((BASE) + 2, (BASE) + 12);                                     \
    c[2] = MAKE_FN((BASE) + 3, (BASE) + 13);                                     \
    c[3] = MAKE_FN((BASE) + 4, (BASE) + 14);                                     \
}                                                                                \
static int stub_##SUFFIX##_cblas(fb_layout_t layout, fb_side_t side,            \
                                 fb_transpose_t trans, char direct,             \
                                 char storev, int m, int n, int k,             \
                                 const CTYPE *v, int ldv, const CTYPE *t,      \
                                 int ldt, CTYPE *c, int ldc)                   \
{                                                                                \
    g_##SUFFIX##_cblas_call.called += 1;                                         \
    g_##SUFFIX##_cblas_call.layout = layout;                                     \
    g_##SUFFIX##_cblas_call.side = side;                                         \
    g_##SUFFIX##_cblas_call.trans = trans;                                       \
    g_##SUFFIX##_cblas_call.direct = direct;                                     \
    g_##SUFFIX##_cblas_call.storev = storev;                                     \
    g_##SUFFIX##_cblas_call.m = m;                                               \
    g_##SUFFIX##_cblas_call.n = n;                                               \
    g_##SUFFIX##_cblas_call.k = k;                                               \
    g_##SUFFIX##_cblas_call.v = v;                                               \
    g_##SUFFIX##_cblas_call.ldv = ldv;                                           \
    g_##SUFFIX##_cblas_call.t = t;                                               \
    g_##SUFFIX##_cblas_call.ldt = ldt;                                           \
    g_##SUFFIX##_cblas_call.c = c;                                               \
    g_##SUFFIX##_cblas_call.ldc = ldc;                                           \
    c[0] = MAKE_FN((BASE) + 10, (BASE) + 20);                                    \
    c[1] = MAKE_FN((BASE) + 11, (BASE) + 21);                                    \
    c[2] = MAKE_FN((BASE) + 12, (BASE) + 22);                                    \
    c[3] = MAKE_FN((BASE) + 13, (BASE) + 23);                                    \
    return (BASE) + 99;                                                          \
}                                                                                \
static int check_##SUFFIX##_fortran_to_cblas(void)                               \
{                                                                                \
    fb_backend_vtable_t vtable;                                                  \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                         \
    CTYPE v_row[4] = { MAKE_FN(1, 11), MAKE_FN(2, 12), MAKE_FN(3, 13), MAKE_FN(4, 14) }; \
    CTYPE t_row[4] = { MAKE_FN(5, 15), MAKE_FN(6, 16), MAKE_FN(7, 17), MAKE_FN(8, 18) }; \
    CTYPE c_row[4] = { MAKE_FN(9, 19), MAKE_FN(10, 20), MAKE_FN(11, 21), MAKE_FN(12, 22) }; \
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
    if (thunk(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_CONJ_TRANS, 'F', 'C', 2, 2, 2,   \
              v_row, 2, t_row, 2, c_row, 2) != 0 ||                             \
        g_##SUFFIX##_fortran_call.called != 1 ||                                 \
        g_##SUFFIX##_fortran_call.side != 'L' ||                                 \
        g_##SUFFIX##_fortran_call.trans != 'C' ||                                \
        g_##SUFFIX##_fortran_call.direct != 'F' ||                               \
        g_##SUFFIX##_fortran_call.storev != 'C' ||                               \
        g_##SUFFIX##_fortran_call.m != 2 ||                                      \
        g_##SUFFIX##_fortran_call.n != 2 ||                                      \
        g_##SUFFIX##_fortran_call.k != 2 ||                                      \
        g_##SUFFIX##_fortran_call.v == v_row ||                                  \
        g_##SUFFIX##_fortran_call.t == t_row ||                                  \
        g_##SUFFIX##_fortran_call.c == c_row ||                                  \
        g_##SUFFIX##_fortran_call.ldv != 2 ||                                    \
        g_##SUFFIX##_fortran_call.ldt != 2 ||                                    \
        g_##SUFFIX##_fortran_call.ldc != 2 ||                                    \
        g_##SUFFIX##_fortran_call.ldwork != 2 ||                                 \
        g_##SUFFIX##_fortran_call.work == NULL ||                                \
        !EQ_FN(g_##SUFFIX##_fortran_call.t_snapshot[0], MAKE_FN(5, 15)) ||      \
        !EQ_FN(g_##SUFFIX##_fortran_call.t_snapshot[1], MAKE_FN(7, 17)) ||      \
        !EQ_FN(g_##SUFFIX##_fortran_call.t_snapshot[2], MAKE_FN(6, 16)) ||      \
        !EQ_FN(g_##SUFFIX##_fortran_call.t_snapshot[3], MAKE_FN(8, 18)) ||      \
        !EQ_FN(g_##SUFFIX##_fortran_call.c_snapshot[0], MAKE_FN(9, 19)) ||      \
        !EQ_FN(g_##SUFFIX##_fortran_call.c_snapshot[1], MAKE_FN(11, 21)) ||     \
        !EQ_FN(g_##SUFFIX##_fortran_call.c_snapshot[2], MAKE_FN(10, 20)) ||     \
        !EQ_FN(g_##SUFFIX##_fortran_call.c_snapshot[3], MAKE_FN(12, 22)) ||     \
        !EQ_FN(c_row[0], MAKE_FN((BASE) + 1, (BASE) + 11)) ||                   \
        !EQ_FN(c_row[1], MAKE_FN((BASE) + 3, (BASE) + 13)) ||                   \
        !EQ_FN(c_row[2], MAKE_FN((BASE) + 2, (BASE) + 12)) ||                   \
        !EQ_FN(c_row[3], MAKE_FN((BASE) + 4, (BASE) + 14))) {                   \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not stage complex row-major LARFB inputs or outputs correctly\n"); \
        return 1;                                                                \
    }                                                                            \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk stages complex row-major LARFB inputs and returns success\n"); \
    return 0;                                                                    \
}                                                                                \
static int check_##SUFFIX##_cblas_to_fortran(void)                               \
{                                                                                \
    fb_backend_vtable_t vtable;                                                  \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                       \
    char side = 'R';                                                             \
    char trans = 'C';                                                            \
    char direct = 'B';                                                           \
    char storev = 'R';                                                           \
    int m = 2;                                                                   \
    int n = 2;                                                                   \
    int k = 2;                                                                   \
    int ldv = 2;                                                                 \
    int ldt = 2;                                                                 \
    int ldc = 2;                                                                 \
    int ldwork = 2;                                                              \
    int info = -1;                                                               \
    CTYPE v[4] = { MAKE_FN(21, 31), MAKE_FN(22, 32), MAKE_FN(23, 33), MAKE_FN(24, 34) }; \
    CTYPE t[4] = { MAKE_FN(25, 35), MAKE_FN(26, 36), MAKE_FN(27, 37), MAKE_FN(28, 38) }; \
    CTYPE c[4] = { MAKE_FN(29, 39), MAKE_FN(30, 40), MAKE_FN(31, 41), MAKE_FN(32, 42) }; \
    CTYPE work[4] = { MAKE_FN(0, 0), MAKE_FN(0, 0), MAKE_FN(0, 0), MAKE_FN(0, 0) }; \
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
    thunk(&side, &trans, &direct, &storev, &m, &n, &k, v, &ldv, t, &ldt, c, &ldc, work, &ldwork, &info); \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                   \
        g_##SUFFIX##_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||                 \
        g_##SUFFIX##_cblas_call.side != FB_RIGHT ||                              \
        g_##SUFFIX##_cblas_call.trans != FB_CONJ_TRANS ||                        \
        g_##SUFFIX##_cblas_call.direct != 'B' ||                                 \
        g_##SUFFIX##_cblas_call.storev != 'R' ||                                 \
        g_##SUFFIX##_cblas_call.m != 2 ||                                        \
        g_##SUFFIX##_cblas_call.n != 2 ||                                        \
        g_##SUFFIX##_cblas_call.k != 2 ||                                        \
        g_##SUFFIX##_cblas_call.v != v ||                                        \
        g_##SUFFIX##_cblas_call.ldv != 2 ||                                      \
        g_##SUFFIX##_cblas_call.t != t ||                                        \
        g_##SUFFIX##_cblas_call.ldt != 2 ||                                      \
        g_##SUFFIX##_cblas_call.c != c ||                                        \
        g_##SUFFIX##_cblas_call.ldc != 2 ||                                      \
        info != (BASE) + 99 ||                                                   \
        !EQ_FN(c[0], MAKE_FN((BASE) + 10, (BASE) + 20)) ||                      \
        !EQ_FN(c[1], MAKE_FN((BASE) + 11, (BASE) + 21)) ||                      \
        !EQ_FN(c[2], MAKE_FN((BASE) + 12, (BASE) + 22)) ||                      \
        !EQ_FN(c[3], MAKE_FN((BASE) + 13, (BASE) + 23))) {                      \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not forward complex LARFB inputs or store info correctly\n"); \
        return 1;                                                                \
    }                                                                            \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences complex LARFB inputs and stores info\n"); \
    return 0;                                                                    \
}

DEFINE_LARFB_REAL_TESTS(slarfb, float, FB_OP_SLARFB, 100)
DEFINE_LARFB_REAL_TESTS(dlarfb, double, FB_OP_DLARFB, 300)
DEFINE_LARFB_COMPLEX_TESTS(clarfb, fb_complex_float_t, FB_OP_CLARFB, make_cf32,
                           cf32_eq, 500)
DEFINE_LARFB_COMPLEX_TESTS(zlarfb, fb_complex_double_t, FB_OP_ZLARFB, make_cf64,
                           cf64_eq, 700)

int main(void)
{
    int status = 0;

    status |= check_slarfb_fortran_to_cblas();
    status |= check_slarfb_cblas_to_fortran();
    status |= check_dlarfb_fortran_to_cblas();
    status |= check_dlarfb_cblas_to_fortran();
    status |= check_clarfb_fortran_to_cblas();
    status |= check_clarfb_cblas_to_fortran();
    status |= check_zlarfb_fortran_to_cblas();
    status |= check_zlarfb_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}