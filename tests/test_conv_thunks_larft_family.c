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

#define DEFINE_LARFT_REAL_TESTS(SUFFIX, TYPE, OP_ID, BASE)                     \
typedef int (*fb_##SUFFIX##_cblas_fn)(fb_layout_t layout, char direct,         \
                                      char storev, int n, int k,               \
                                      const TYPE *v, int ldv, const TYPE *tau, \
                                      TYPE *t, int ldt);                       \
typedef void (*fb_##SUFFIX##_fortran_fn)(char *direct, char *storev, int *n,   \
                                         int *k, TYPE *v, int *ldv,            \
                                         TYPE *tau, TYPE *t, int *ldt,         \
                                         int *info);                           \
static struct {                                                                  \
    int called;                                                                  \
    char direct;                                                                  \
    char storev;                                                                  \
    int n;                                                                        \
    int k;                                                                        \
    TYPE *v;                                                                      \
    int ldv;                                                                      \
    TYPE *tau;                                                                    \
    TYPE *t;                                                                      \
    int ldt;                                                                      \
    int *info_ptr;                                                                \
    TYPE v_snapshot[4];                                                           \
    TYPE tau_snapshot[2];                                                         \
} g_##SUFFIX##_fortran_call;                                                      \
static struct {                                                                   \
    int called;                                                                   \
    fb_layout_t layout;                                                           \
    char direct;                                                                  \
    char storev;                                                                  \
    int n;                                                                        \
    int k;                                                                        \
    const TYPE *v;                                                                \
    int ldv;                                                                      \
    const TYPE *tau;                                                              \
    TYPE *t;                                                                      \
    int ldt;                                                                      \
} g_##SUFFIX##_cblas_call;                                                        \
static void stub_##SUFFIX##_fortran(char *direct, char *storev, int *n,         \
                                    int *k, TYPE *v, int *ldv, TYPE *tau,      \
                                    TYPE *t, int *ldt, int *info)              \
{                                                                                 \
    g_##SUFFIX##_fortran_call.called += 1;                                        \
    g_##SUFFIX##_fortran_call.direct = *direct;                                   \
    g_##SUFFIX##_fortran_call.storev = *storev;                                   \
    g_##SUFFIX##_fortran_call.n = *n;                                             \
    g_##SUFFIX##_fortran_call.k = *k;                                             \
    g_##SUFFIX##_fortran_call.v = v;                                              \
    g_##SUFFIX##_fortran_call.ldv = *ldv;                                         \
    g_##SUFFIX##_fortran_call.tau = tau;                                          \
    g_##SUFFIX##_fortran_call.t = t;                                              \
    g_##SUFFIX##_fortran_call.ldt = *ldt;                                         \
    g_##SUFFIX##_fortran_call.info_ptr = info;                                    \
    g_##SUFFIX##_fortran_call.v_snapshot[0] = v[0];                               \
    g_##SUFFIX##_fortran_call.v_snapshot[1] = v[1];                               \
    g_##SUFFIX##_fortran_call.v_snapshot[2] = v[2];                               \
    g_##SUFFIX##_fortran_call.v_snapshot[3] = v[3];                               \
    g_##SUFFIX##_fortran_call.tau_snapshot[0] = tau[0];                           \
    g_##SUFFIX##_fortran_call.tau_snapshot[1] = tau[1];                           \
    t[0] = (TYPE)((BASE) + 1);                                                    \
    t[1] = (TYPE)((BASE) + 2);                                                    \
    t[2] = (TYPE)((BASE) + 3);                                                    \
    t[3] = (TYPE)((BASE) + 4);                                                    \
    *info = 0;                                                                    \
}                                                                                 \
static int stub_##SUFFIX##_cblas(fb_layout_t layout, char direct, char storev,  \
                                 int n, int k, const TYPE *v, int ldv,         \
                                 const TYPE *tau, TYPE *t, int ldt)            \
{                                                                                 \
    g_##SUFFIX##_cblas_call.called += 1;                                          \
    g_##SUFFIX##_cblas_call.layout = layout;                                      \
    g_##SUFFIX##_cblas_call.direct = direct;                                      \
    g_##SUFFIX##_cblas_call.storev = storev;                                      \
    g_##SUFFIX##_cblas_call.n = n;                                                \
    g_##SUFFIX##_cblas_call.k = k;                                                \
    g_##SUFFIX##_cblas_call.v = v;                                                \
    g_##SUFFIX##_cblas_call.ldv = ldv;                                            \
    g_##SUFFIX##_cblas_call.tau = tau;                                            \
    g_##SUFFIX##_cblas_call.t = t;                                                \
    g_##SUFFIX##_cblas_call.ldt = ldt;                                            \
    t[0] = (TYPE)((BASE) + 10);                                                   \
    t[1] = (TYPE)((BASE) + 11);                                                   \
    t[2] = (TYPE)((BASE) + 12);                                                   \
    t[3] = (TYPE)((BASE) + 13);                                                   \
    return (BASE) + 99;                                                           \
}                                                                                 \
static int check_##SUFFIX##_fortran_to_cblas(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                          \
    TYPE v_row[4] = { (TYPE)1, (TYPE)2, (TYPE)3, (TYPE)4 };                      \
    TYPE tau[2] = { (TYPE)5, (TYPE)6 };                                           \
    TYPE t_row[4] = { (TYPE)0, (TYPE)0, (TYPE)0, (TYPE)0 };                      \
    memset(&vtable, 0, sizeof(vtable));                                           \
    memset(&g_##SUFFIX##_fortran_call, 0, sizeof(g_##SUFFIX##_fortran_call));     \
    vtable.ext_ops[OP_ID][FB_CONV_FORTRAN] =                                      \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_fortran;                   \
    fb_install_conv_thunks(&vtable, OP_ID);                                       \
    thunk = (fb_##SUFFIX##_cblas_fn)vtable.ext_ops[OP_ID][FB_CONV_CBLAS];         \
    if (!thunk) {                                                                 \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk was not installed\n"); \
        return 1;                                                                 \
    }                                                                             \
    int rc = thunk(FB_LAYOUT_ROW_MAJOR, 'F', 'C', 2, 2, v_row, 2, tau, t_row, 2); \
    if (rc != 0 ||                                                               \
        g_##SUFFIX##_fortran_call.called != 1 ||                                  \
        g_##SUFFIX##_fortran_call.direct != 'F' ||                                \
        g_##SUFFIX##_fortran_call.storev != 'C' ||                                \
        g_##SUFFIX##_fortran_call.n != 2 ||                                       \
        g_##SUFFIX##_fortran_call.k != 2 ||                                       \
        g_##SUFFIX##_fortran_call.v == v_row ||                                   \
        g_##SUFFIX##_fortran_call.t == t_row ||                                   \
        g_##SUFFIX##_fortran_call.ldv != 2 ||                                     \
        g_##SUFFIX##_fortran_call.ldt != 2 ||                                     \
        g_##SUFFIX##_fortran_call.info_ptr == NULL ||                             \
        g_##SUFFIX##_fortran_call.v_snapshot[0] != (TYPE)1 ||                     \
        g_##SUFFIX##_fortran_call.v_snapshot[1] != (TYPE)3 ||                     \
        g_##SUFFIX##_fortran_call.v_snapshot[2] != (TYPE)2 ||                     \
        g_##SUFFIX##_fortran_call.v_snapshot[3] != (TYPE)4 ||                     \
        g_##SUFFIX##_fortran_call.tau_snapshot[0] != (TYPE)5 ||                   \
        g_##SUFFIX##_fortran_call.tau_snapshot[1] != (TYPE)6 ||                   \
        t_row[0] != (TYPE)((BASE) + 1) || t_row[1] != (TYPE)((BASE) + 3) ||      \
        t_row[2] != (TYPE)((BASE) + 2) || t_row[3] != (TYPE)((BASE) + 4)) {      \
        fprintf(stderr, "[FAIL] " #SUFFIX " rc=%d called=%d ldv=%d ldt=%d v=[%g,%g,%g,%g] t=[%g,%g,%g,%g]\n", \
            rc, g_##SUFFIX##_fortran_call.called,                             \
            g_##SUFFIX##_fortran_call.ldv, g_##SUFFIX##_fortran_call.ldt,     \
            (double)g_##SUFFIX##_fortran_call.v_snapshot[0],                  \
            (double)g_##SUFFIX##_fortran_call.v_snapshot[1],                  \
            (double)g_##SUFFIX##_fortran_call.v_snapshot[2],                  \
            (double)g_##SUFFIX##_fortran_call.v_snapshot[3],                  \
            (double)t_row[0], (double)t_row[1], (double)t_row[2],            \
            (double)t_row[3]);                                               \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk stages row-major LARFT inputs and stores info\n"); \
    return 0;                                                                     \
}                                                                                 \
static int check_##SUFFIX##_cblas_to_fortran(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                        \
    char direct = 'B';                                                            \
    char storev = 'R';                                                            \
    int n = 2;                                                                    \
    int k = 2;                                                                    \
    int ldv = 2;                                                                  \
    int ldt = 2;                                                                  \
    int info = -1;                                                                \
    TYPE v[4] = { (TYPE)21, (TYPE)22, (TYPE)23, (TYPE)24 };                      \
    TYPE tau[2] = { (TYPE)25, (TYPE)26 };                                         \
    TYPE t[4] = { (TYPE)27, (TYPE)28, (TYPE)29, (TYPE)30 };                      \
    memset(&vtable, 0, sizeof(vtable));                                           \
    memset(&g_##SUFFIX##_cblas_call, 0, sizeof(g_##SUFFIX##_cblas_call));         \
    vtable.ext_ops[OP_ID][FB_CONV_CBLAS] =                                        \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_cblas;                     \
    fb_install_conv_thunks(&vtable, OP_ID);                                       \
    thunk = (fb_##SUFFIX##_fortran_fn)vtable.ext_ops[OP_ID][FB_CONV_FORTRAN];     \
    if (!thunk) {                                                                 \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk was not installed\n"); \
        return 1;                                                                 \
    }                                                                             \
    thunk(&direct, &storev, &n, &k, v, &ldv, tau, t, &ldt, &info);               \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                    \
        g_##SUFFIX##_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||                  \
        g_##SUFFIX##_cblas_call.direct != 'B' ||                                  \
        g_##SUFFIX##_cblas_call.storev != 'R' ||                                  \
        g_##SUFFIX##_cblas_call.n != 2 ||                                         \
        g_##SUFFIX##_cblas_call.k != 2 ||                                         \
        g_##SUFFIX##_cblas_call.v != v ||                                         \
        g_##SUFFIX##_cblas_call.ldv != 2 ||                                       \
        g_##SUFFIX##_cblas_call.tau != tau ||                                     \
        g_##SUFFIX##_cblas_call.t != t ||                                         \
        g_##SUFFIX##_cblas_call.ldt != 2 ||                                       \
        info != (BASE) + 99 ||                                                    \
        t[0] != (TYPE)((BASE) + 10) || t[1] != (TYPE)((BASE) + 11) ||            \
        t[2] != (TYPE)((BASE) + 12) || t[3] != (TYPE)((BASE) + 13)) {            \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not forward LARFT inputs or store info correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences LARFT inputs and stores info\n"); \
    return 0;                                                                     \
}

#define DEFINE_LARFT_COMPLEX_TESTS(SUFFIX, CTYPE, OP_ID, MAKE_FN, EQ_FN, BASE) \
typedef int (*fb_##SUFFIX##_cblas_fn)(fb_layout_t layout, char direct,         \
                                      char storev, int n, int k,               \
                                      const CTYPE *v, int ldv, const CTYPE *tau, \
                                      CTYPE *t, int ldt);                      \
typedef void (*fb_##SUFFIX##_fortran_fn)(char *direct, char *storev, int *n,   \
                                         int *k, CTYPE *v, int *ldv,           \
                                         CTYPE *tau, CTYPE *t, int *ldt);      \
static struct {                                                                  \
    int called;                                                                  \
    char direct;                                                                  \
    char storev;                                                                  \
    int n;                                                                        \
    int k;                                                                        \
    CTYPE *v;                                                                     \
    int ldv;                                                                      \
    CTYPE *tau;                                                                   \
    CTYPE *t;                                                                     \
    int ldt;                                                                      \
    CTYPE v_snapshot[4];                                                          \
    CTYPE tau_snapshot[2];                                                        \
} g_##SUFFIX##_fortran_call;                                                      \
static struct {                                                                   \
    int called;                                                                   \
    fb_layout_t layout;                                                           \
    char direct;                                                                  \
    char storev;                                                                  \
    int n;                                                                        \
    int k;                                                                        \
    const CTYPE *v;                                                               \
    int ldv;                                                                      \
    const CTYPE *tau;                                                             \
    CTYPE *t;                                                                     \
    int ldt;                                                                      \
} g_##SUFFIX##_cblas_call;                                                        \
static void stub_##SUFFIX##_fortran(char *direct, char *storev, int *n,         \
                                    int *k, CTYPE *v, int *ldv, CTYPE *tau,    \
                                    CTYPE *t, int *ldt)                        \
{                                                                                 \
    g_##SUFFIX##_fortran_call.called += 1;                                        \
    g_##SUFFIX##_fortran_call.direct = *direct;                                   \
    g_##SUFFIX##_fortran_call.storev = *storev;                                   \
    g_##SUFFIX##_fortran_call.n = *n;                                             \
    g_##SUFFIX##_fortran_call.k = *k;                                             \
    g_##SUFFIX##_fortran_call.v = v;                                              \
    g_##SUFFIX##_fortran_call.ldv = *ldv;                                         \
    g_##SUFFIX##_fortran_call.tau = tau;                                          \
    g_##SUFFIX##_fortran_call.t = t;                                              \
    g_##SUFFIX##_fortran_call.ldt = *ldt;                                         \
    g_##SUFFIX##_fortran_call.v_snapshot[0] = v[0];                               \
    g_##SUFFIX##_fortran_call.v_snapshot[1] = v[1];                               \
    g_##SUFFIX##_fortran_call.v_snapshot[2] = v[2];                               \
    g_##SUFFIX##_fortran_call.v_snapshot[3] = v[3];                               \
    g_##SUFFIX##_fortran_call.tau_snapshot[0] = tau[0];                           \
    g_##SUFFIX##_fortran_call.tau_snapshot[1] = tau[1];                           \
    t[0] = MAKE_FN((BASE) + 1, (BASE) + 11);                                      \
    t[1] = MAKE_FN((BASE) + 2, (BASE) + 12);                                      \
    t[2] = MAKE_FN((BASE) + 3, (BASE) + 13);                                      \
    t[3] = MAKE_FN((BASE) + 4, (BASE) + 14);                                      \
}                                                                                 \
static int stub_##SUFFIX##_cblas(fb_layout_t layout, char direct, char storev,  \
                                 int n, int k, const CTYPE *v, int ldv,        \
                                 const CTYPE *tau, CTYPE *t, int ldt)          \
{                                                                                 \
    g_##SUFFIX##_cblas_call.called += 1;                                          \
    g_##SUFFIX##_cblas_call.layout = layout;                                      \
    g_##SUFFIX##_cblas_call.direct = direct;                                      \
    g_##SUFFIX##_cblas_call.storev = storev;                                      \
    g_##SUFFIX##_cblas_call.n = n;                                                \
    g_##SUFFIX##_cblas_call.k = k;                                                \
    g_##SUFFIX##_cblas_call.v = v;                                                \
    g_##SUFFIX##_cblas_call.ldv = ldv;                                            \
    g_##SUFFIX##_cblas_call.tau = tau;                                            \
    g_##SUFFIX##_cblas_call.t = t;                                                \
    g_##SUFFIX##_cblas_call.ldt = ldt;                                            \
    t[0] = MAKE_FN((BASE) + 10, (BASE) + 20);                                     \
    t[1] = MAKE_FN((BASE) + 11, (BASE) + 21);                                     \
    t[2] = MAKE_FN((BASE) + 12, (BASE) + 22);                                     \
    t[3] = MAKE_FN((BASE) + 13, (BASE) + 23);                                     \
    return (BASE) + 99;                                                           \
}                                                                                 \
static int check_##SUFFIX##_fortran_to_cblas(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                          \
    CTYPE v_row[4] = { MAKE_FN(1, 11), MAKE_FN(2, 12), MAKE_FN(3, 13), MAKE_FN(4, 14) }; \
    CTYPE tau[2] = { MAKE_FN(5, 15), MAKE_FN(6, 16) };                           \
    CTYPE t_row[4] = { MAKE_FN(0, 0), MAKE_FN(0, 0), MAKE_FN(0, 0), MAKE_FN(0, 0) }; \
    memset(&vtable, 0, sizeof(vtable));                                           \
    memset(&g_##SUFFIX##_fortran_call, 0, sizeof(g_##SUFFIX##_fortran_call));     \
    vtable.ext_ops[OP_ID][FB_CONV_FORTRAN] =                                      \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_fortran;                   \
    fb_install_conv_thunks(&vtable, OP_ID);                                       \
    thunk = (fb_##SUFFIX##_cblas_fn)vtable.ext_ops[OP_ID][FB_CONV_CBLAS];         \
    if (!thunk) {                                                                 \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk was not installed\n"); \
        return 1;                                                                 \
    }                                                                             \
    int rc = thunk(FB_LAYOUT_ROW_MAJOR, 'F', 'C', 2, 2, v_row, 2, tau, t_row, 2); \
    if (rc != 0 ||                                                                 \
        g_##SUFFIX##_fortran_call.called != 1 ||                                  \
        g_##SUFFIX##_fortran_call.direct != 'F' ||                                \
        g_##SUFFIX##_fortran_call.storev != 'C' ||                                \
        g_##SUFFIX##_fortran_call.n != 2 ||                                       \
        g_##SUFFIX##_fortran_call.k != 2 ||                                       \
        g_##SUFFIX##_fortran_call.v == v_row ||                                   \
        g_##SUFFIX##_fortran_call.t == t_row ||                                   \
        g_##SUFFIX##_fortran_call.ldv != 2 ||                                     \
        g_##SUFFIX##_fortran_call.ldt != 2 ||                                     \
        !EQ_FN(g_##SUFFIX##_fortran_call.v_snapshot[0], MAKE_FN(1, 11)) ||       \
        !EQ_FN(g_##SUFFIX##_fortran_call.v_snapshot[1], MAKE_FN(3, 13)) ||       \
        !EQ_FN(g_##SUFFIX##_fortran_call.v_snapshot[2], MAKE_FN(2, 12)) ||       \
        !EQ_FN(g_##SUFFIX##_fortran_call.v_snapshot[3], MAKE_FN(4, 14)) ||       \
        !EQ_FN(g_##SUFFIX##_fortran_call.tau_snapshot[0], MAKE_FN(5, 15)) ||     \
        !EQ_FN(g_##SUFFIX##_fortran_call.tau_snapshot[1], MAKE_FN(6, 16)) ||     \
        !EQ_FN(t_row[0], MAKE_FN((BASE) + 1, (BASE) + 11)) ||                    \
        !EQ_FN(t_row[1], MAKE_FN((BASE) + 3, (BASE) + 13)) ||                    \
        !EQ_FN(t_row[2], MAKE_FN((BASE) + 2, (BASE) + 12)) ||                    \
        !EQ_FN(t_row[3], MAKE_FN((BASE) + 4, (BASE) + 14))) {                    \
        fprintf(stderr, "[FAIL] " #SUFFIX " rc=%d called=%d ldv=%d ldt=%d v00=(%g,%g) v01=(%g,%g) v10=(%g,%g) v11=(%g,%g) t00=(%g,%g) t01=(%g,%g) t10=(%g,%g) t11=(%g,%g)\n", \
            rc, g_##SUFFIX##_fortran_call.called,                             \
            g_##SUFFIX##_fortran_call.ldv, g_##SUFFIX##_fortran_call.ldt,     \
            (double)__real__ g_##SUFFIX##_fortran_call.v_snapshot[0],         \
            (double)__imag__ g_##SUFFIX##_fortran_call.v_snapshot[0],         \
            (double)__real__ g_##SUFFIX##_fortran_call.v_snapshot[1],         \
            (double)__imag__ g_##SUFFIX##_fortran_call.v_snapshot[1],         \
            (double)__real__ g_##SUFFIX##_fortran_call.v_snapshot[2],         \
            (double)__imag__ g_##SUFFIX##_fortran_call.v_snapshot[2],         \
            (double)__real__ g_##SUFFIX##_fortran_call.v_snapshot[3],         \
            (double)__imag__ g_##SUFFIX##_fortran_call.v_snapshot[3],         \
            (double)__real__ t_row[0], (double)__imag__ t_row[0],            \
            (double)__real__ t_row[1], (double)__imag__ t_row[1],            \
            (double)__real__ t_row[2], (double)__imag__ t_row[2],            \
            (double)__real__ t_row[3], (double)__imag__ t_row[3]);           \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk stages complex row-major LARFT inputs\n"); \
    return 0;                                                                     \
}                                                                                 \
static int check_##SUFFIX##_cblas_to_fortran(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                        \
    char direct = 'B';                                                            \
    char storev = 'R';                                                            \
    int n = 2;                                                                    \
    int k = 2;                                                                    \
    int ldv = 2;                                                                  \
    int ldt = 2;                                                                  \
    CTYPE v[4] = { MAKE_FN(21, 31), MAKE_FN(22, 32), MAKE_FN(23, 33), MAKE_FN(24, 34) }; \
    CTYPE tau[2] = { MAKE_FN(25, 35), MAKE_FN(26, 36) };                         \
    CTYPE t[4] = { MAKE_FN(27, 37), MAKE_FN(28, 38), MAKE_FN(29, 39), MAKE_FN(30, 40) }; \
    memset(&vtable, 0, sizeof(vtable));                                           \
    memset(&g_##SUFFIX##_cblas_call, 0, sizeof(g_##SUFFIX##_cblas_call));         \
    vtable.ext_ops[OP_ID][FB_CONV_CBLAS] =                                        \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_cblas;                     \
    fb_install_conv_thunks(&vtable, OP_ID);                                       \
    thunk = (fb_##SUFFIX##_fortran_fn)vtable.ext_ops[OP_ID][FB_CONV_FORTRAN];     \
    if (!thunk) {                                                                 \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk was not installed\n"); \
        return 1;                                                                 \
    }                                                                             \
    thunk(&direct, &storev, &n, &k, v, &ldv, tau, t, &ldt);                      \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                    \
        g_##SUFFIX##_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||                  \
        g_##SUFFIX##_cblas_call.direct != 'B' ||                                  \
        g_##SUFFIX##_cblas_call.storev != 'R' ||                                  \
        g_##SUFFIX##_cblas_call.n != 2 ||                                         \
        g_##SUFFIX##_cblas_call.k != 2 ||                                         \
        g_##SUFFIX##_cblas_call.v != v ||                                         \
        g_##SUFFIX##_cblas_call.ldv != 2 ||                                       \
        g_##SUFFIX##_cblas_call.tau != tau ||                                     \
        g_##SUFFIX##_cblas_call.t != t ||                                         \
        g_##SUFFIX##_cblas_call.ldt != 2 ||                                       \
        !EQ_FN(t[0], MAKE_FN((BASE) + 10, (BASE) + 20)) ||                       \
        !EQ_FN(t[1], MAKE_FN((BASE) + 11, (BASE) + 21)) ||                       \
        !EQ_FN(t[2], MAKE_FN((BASE) + 12, (BASE) + 22)) ||                       \
        !EQ_FN(t[3], MAKE_FN((BASE) + 13, (BASE) + 23))) {                       \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not forward complex LARFT inputs correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences complex LARFT inputs\n"); \
    return 0;                                                                     \
}

DEFINE_LARFT_REAL_TESTS(slarft, float, FB_OP_SLARFT, 100)
DEFINE_LARFT_REAL_TESTS(dlarft, double, FB_OP_DLARFT, 300)
DEFINE_LARFT_COMPLEX_TESTS(clarft, fb_complex_float_t, FB_OP_CLARFT, make_cf32,
                           cf32_eq, 500)
DEFINE_LARFT_COMPLEX_TESTS(zlarft, fb_complex_double_t, FB_OP_ZLARFT, make_cf64,
                           cf64_eq, 700)

int main(void)
{
    int status = 0;

    status |= check_slarft_fortran_to_cblas();
    status |= check_slarft_cblas_to_fortran();
    status |= check_dlarft_fortran_to_cblas();
    status |= check_dlarft_cblas_to_fortran();
    status |= check_clarft_fortran_to_cblas();
    status |= check_clarft_cblas_to_fortran();
    status |= check_zlarft_fortran_to_cblas();
    status |= check_zlarft_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}