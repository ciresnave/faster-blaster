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

#define DEFINE_LARZ_REAL_TESTS(SUFFIX, TYPE, OP_ID, BASE, SIDE_CHAR)           \
typedef int (*fb_##SUFFIX##_cblas_fn)(char side, int m, int n, int l,         \
                                      TYPE *v, int incv, TYPE *tau, TYPE *c,  \
                                      int ldc, TYPE *work);                   \
typedef void (*fb_##SUFFIX##_fortran_fn)(char *side, int *m, int *n, int *l,  \
                                         TYPE *v, int *incv, TYPE *tau,       \
                                         TYPE *c, int *ldc, TYPE *work);      \
static struct {                                                                 \
    int called;                                                                 \
    char side;                                                                  \
    int m;                                                                      \
    int n;                                                                      \
    int l;                                                                      \
    TYPE *v;                                                                    \
    int incv;                                                                   \
    TYPE *tau;                                                                  \
    TYPE *c;                                                                    \
    int ldc;                                                                    \
    TYPE *work;                                                                 \
} g_##SUFFIX##_fortran_call;                                                    \
static struct {                                                                 \
    int called;                                                                 \
    char side;                                                                  \
    int m;                                                                      \
    int n;                                                                      \
    int l;                                                                      \
    TYPE *v;                                                                    \
    int incv;                                                                   \
    TYPE *tau;                                                                  \
    TYPE *c;                                                                    \
    int ldc;                                                                    \
    TYPE *work;                                                                 \
} g_##SUFFIX##_cblas_call;                                                      \
static void stub_##SUFFIX##_fortran(char *side, int *m, int *n, int *l,        \
                                    TYPE *v, int *incv, TYPE *tau, TYPE *c,    \
                                    int *ldc, TYPE *work)                      \
{                                                                               \
    g_##SUFFIX##_fortran_call.called += 1;                                      \
    g_##SUFFIX##_fortran_call.side = *side;                                     \
    g_##SUFFIX##_fortran_call.m = *m;                                           \
    g_##SUFFIX##_fortran_call.n = *n;                                           \
    g_##SUFFIX##_fortran_call.l = *l;                                           \
    g_##SUFFIX##_fortran_call.v = v;                                            \
    g_##SUFFIX##_fortran_call.incv = *incv;                                     \
    g_##SUFFIX##_fortran_call.tau = tau;                                        \
    g_##SUFFIX##_fortran_call.c = c;                                            \
    g_##SUFFIX##_fortran_call.ldc = *ldc;                                       \
    g_##SUFFIX##_fortran_call.work = work;                                      \
    c[0] = (TYPE)((BASE) + 1);                                                  \
    c[3] = (TYPE)((BASE) + 2);                                                  \
    work[0] = (TYPE)((BASE) + 3);                                               \
    work[1] = (TYPE)((BASE) + 4);                                               \
}                                                                               \
static int stub_##SUFFIX##_cblas(char side, int m, int n, int l, TYPE *v,      \
                                 int incv, TYPE *tau, TYPE *c, int ldc,        \
                                 TYPE *work)                                    \
{                                                                               \
    g_##SUFFIX##_cblas_call.called += 1;                                        \
    g_##SUFFIX##_cblas_call.side = side;                                        \
    g_##SUFFIX##_cblas_call.m = m;                                              \
    g_##SUFFIX##_cblas_call.n = n;                                              \
    g_##SUFFIX##_cblas_call.l = l;                                              \
    g_##SUFFIX##_cblas_call.v = v;                                              \
    g_##SUFFIX##_cblas_call.incv = incv;                                        \
    g_##SUFFIX##_cblas_call.tau = tau;                                          \
    g_##SUFFIX##_cblas_call.c = c;                                              \
    g_##SUFFIX##_cblas_call.ldc = ldc;                                          \
    g_##SUFFIX##_cblas_call.work = work;                                        \
    c[0] = (TYPE)((BASE) + 10);                                                 \
    c[3] = (TYPE)((BASE) + 11);                                                 \
    work[0] = (TYPE)((BASE) + 12);                                              \
    work[1] = (TYPE)((BASE) + 13);                                              \
    return (BASE) + 99;                                                         \
}                                                                               \
static int check_##SUFFIX##_fortran_to_cblas(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                        \
    TYPE v[1] = { (TYPE)7 };                                                    \
    TYPE tau = (TYPE)9;                                                         \
    TYPE c[4] = { (TYPE)0, (TYPE)0, (TYPE)0, (TYPE)0 };                        \
    TYPE work[2] = { (TYPE)0, (TYPE)0 };                                        \
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
    if (thunk((SIDE_CHAR), 2, 2, 1, v, 1, &tau, c, 2, work) != 0 ||            \
        g_##SUFFIX##_fortran_call.called != 1 ||                                \
        g_##SUFFIX##_fortran_call.side != (SIDE_CHAR) ||                        \
        g_##SUFFIX##_fortran_call.m != 2 ||                                     \
        g_##SUFFIX##_fortran_call.n != 2 ||                                     \
        g_##SUFFIX##_fortran_call.l != 1 ||                                     \
        g_##SUFFIX##_fortran_call.v != v ||                                     \
        g_##SUFFIX##_fortran_call.incv != 1 ||                                  \
        g_##SUFFIX##_fortran_call.tau != &tau ||                                \
        g_##SUFFIX##_fortran_call.c != c ||                                     \
        g_##SUFFIX##_fortran_call.ldc != 2 ||                                   \
        g_##SUFFIX##_fortran_call.work != work ||                               \
        c[0] != (TYPE)((BASE) + 1) || c[3] != (TYPE)((BASE) + 2) ||            \
        work[0] != (TYPE)((BASE) + 3) || work[1] != (TYPE)((BASE) + 4)) {      \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward LARZ inputs or outputs correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards LARZ inputs and returns success\n"); \
    return 0;                                                                   \
}                                                                               \
static int check_##SUFFIX##_cblas_to_fortran(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                      \
    char side = (SIDE_CHAR);                                                    \
    int m = 2;                                                                  \
    int n = 2;                                                                  \
    int l = 1;                                                                  \
    int incv = 1;                                                               \
    int ldc = 2;                                                                \
    TYPE v[1] = { (TYPE)11 };                                                   \
    TYPE tau = (TYPE)13;                                                        \
    TYPE c[4] = { (TYPE)0, (TYPE)0, (TYPE)0, (TYPE)0 };                        \
    TYPE work[2] = { (TYPE)0, (TYPE)0 };                                        \
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
    thunk(&side, &m, &n, &l, v, &incv, &tau, c, &ldc, work);                   \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                  \
        g_##SUFFIX##_cblas_call.side != (SIDE_CHAR) ||                          \
        g_##SUFFIX##_cblas_call.m != 2 ||                                       \
        g_##SUFFIX##_cblas_call.n != 2 ||                                       \
        g_##SUFFIX##_cblas_call.l != 1 ||                                       \
        g_##SUFFIX##_cblas_call.v != v ||                                       \
        g_##SUFFIX##_cblas_call.incv != 1 ||                                    \
        g_##SUFFIX##_cblas_call.tau != &tau ||                                  \
        g_##SUFFIX##_cblas_call.c != c ||                                       \
        g_##SUFFIX##_cblas_call.ldc != 2 ||                                     \
        g_##SUFFIX##_cblas_call.work != work ||                                 \
        c[0] != (TYPE)((BASE) + 10) || c[3] != (TYPE)((BASE) + 11) ||          \
        work[0] != (TYPE)((BASE) + 12) || work[1] != (TYPE)((BASE) + 13)) {    \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference LARZ inputs or propagate outputs correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences LARZ inputs and ignores the C int return\n"); \
    return 0;                                                                   \
}

#define DEFINE_LARZ_COMPLEX_TESTS(SUFFIX, CTYPE, OP_ID, MAKE_FN, EQ_FN, BASE, SIDE_CHAR) \
typedef int (*fb_##SUFFIX##_cblas_fn)(char side, int m, int n, int l,         \
                                      CTYPE *v, int incv, CTYPE *tau,         \
                                      CTYPE *c, int ldc, CTYPE *work);        \
typedef void (*fb_##SUFFIX##_fortran_fn)(char *side, int *m, int *n, int *l,  \
                                         CTYPE *v, int *incv, CTYPE *tau,     \
                                         CTYPE *c, int *ldc, CTYPE *work);    \
static struct {                                                                 \
    int called;                                                                 \
    char side;                                                                  \
    int m;                                                                      \
    int n;                                                                      \
    int l;                                                                      \
    CTYPE *v;                                                                   \
    int incv;                                                                   \
    CTYPE *tau;                                                                 \
    CTYPE *c;                                                                   \
    int ldc;                                                                    \
    CTYPE *work;                                                                \
} g_##SUFFIX##_fortran_call;                                                    \
static struct {                                                                 \
    int called;                                                                 \
    char side;                                                                  \
    int m;                                                                      \
    int n;                                                                      \
    int l;                                                                      \
    CTYPE *v;                                                                   \
    int incv;                                                                   \
    CTYPE *tau;                                                                 \
    CTYPE *c;                                                                   \
    int ldc;                                                                    \
    CTYPE *work;                                                                \
} g_##SUFFIX##_cblas_call;                                                      \
static void stub_##SUFFIX##_fortran(char *side, int *m, int *n, int *l,        \
                                    CTYPE *v, int *incv, CTYPE *tau, CTYPE *c, \
                                    int *ldc, CTYPE *work)                     \
{                                                                               \
    g_##SUFFIX##_fortran_call.called += 1;                                      \
    g_##SUFFIX##_fortran_call.side = *side;                                     \
    g_##SUFFIX##_fortran_call.m = *m;                                           \
    g_##SUFFIX##_fortran_call.n = *n;                                           \
    g_##SUFFIX##_fortran_call.l = *l;                                           \
    g_##SUFFIX##_fortran_call.v = v;                                            \
    g_##SUFFIX##_fortran_call.incv = *incv;                                     \
    g_##SUFFIX##_fortran_call.tau = tau;                                        \
    g_##SUFFIX##_fortran_call.c = c;                                            \
    g_##SUFFIX##_fortran_call.ldc = *ldc;                                       \
    g_##SUFFIX##_fortran_call.work = work;                                      \
    c[0] = MAKE_FN((BASE) + 1, (BASE) + 11);                                    \
    c[3] = MAKE_FN((BASE) + 2, (BASE) + 12);                                    \
    work[0] = MAKE_FN((BASE) + 3, (BASE) + 13);                                 \
    work[1] = MAKE_FN((BASE) + 4, (BASE) + 14);                                 \
}                                                                               \
static int stub_##SUFFIX##_cblas(char side, int m, int n, int l, CTYPE *v,     \
                                 int incv, CTYPE *tau, CTYPE *c, int ldc,      \
                                 CTYPE *work)                                   \
{                                                                               \
    g_##SUFFIX##_cblas_call.called += 1;                                        \
    g_##SUFFIX##_cblas_call.side = side;                                        \
    g_##SUFFIX##_cblas_call.m = m;                                              \
    g_##SUFFIX##_cblas_call.n = n;                                              \
    g_##SUFFIX##_cblas_call.l = l;                                              \
    g_##SUFFIX##_cblas_call.v = v;                                              \
    g_##SUFFIX##_cblas_call.incv = incv;                                        \
    g_##SUFFIX##_cblas_call.tau = tau;                                          \
    g_##SUFFIX##_cblas_call.c = c;                                              \
    g_##SUFFIX##_cblas_call.ldc = ldc;                                          \
    g_##SUFFIX##_cblas_call.work = work;                                        \
    c[0] = MAKE_FN((BASE) + 10, (BASE) + 20);                                   \
    c[3] = MAKE_FN((BASE) + 11, (BASE) + 21);                                   \
    work[0] = MAKE_FN((BASE) + 12, (BASE) + 22);                                \
    work[1] = MAKE_FN((BASE) + 13, (BASE) + 23);                                \
    return (BASE) + 99;                                                         \
}                                                                               \
static int check_##SUFFIX##_fortran_to_cblas(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                        \
    CTYPE v[1] = { MAKE_FN(7, 17) };                                            \
    CTYPE tau = MAKE_FN(9, 19);                                                 \
    CTYPE c[4] = { MAKE_FN(0, 0), MAKE_FN(0, 0), MAKE_FN(0, 0), MAKE_FN(0, 0) }; \
    CTYPE work[2] = { MAKE_FN(0, 0), MAKE_FN(0, 0) };                           \
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
    if (thunk((SIDE_CHAR), 2, 2, 1, v, 1, &tau, c, 2, work) != 0 ||            \
        g_##SUFFIX##_fortran_call.called != 1 ||                                \
        g_##SUFFIX##_fortran_call.side != (SIDE_CHAR) ||                        \
        g_##SUFFIX##_fortran_call.m != 2 ||                                     \
        g_##SUFFIX##_fortran_call.n != 2 ||                                     \
        g_##SUFFIX##_fortran_call.l != 1 ||                                     \
        g_##SUFFIX##_fortran_call.v != v ||                                     \
        g_##SUFFIX##_fortran_call.incv != 1 ||                                  \
        g_##SUFFIX##_fortran_call.tau != &tau ||                                \
        g_##SUFFIX##_fortran_call.c != c ||                                     \
        g_##SUFFIX##_fortran_call.ldc != 2 ||                                   \
        g_##SUFFIX##_fortran_call.work != work ||                               \
        !EQ_FN(c[0], MAKE_FN((BASE) + 1, (BASE) + 11)) ||                      \
        !EQ_FN(c[3], MAKE_FN((BASE) + 2, (BASE) + 12)) ||                      \
        !EQ_FN(work[0], MAKE_FN((BASE) + 3, (BASE) + 13)) ||                   \
        !EQ_FN(work[1], MAKE_FN((BASE) + 4, (BASE) + 14))) {                   \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward complex LARZ inputs or outputs correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards complex LARZ inputs and returns success\n"); \
    return 0;                                                                   \
}                                                                               \
static int check_##SUFFIX##_cblas_to_fortran(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                      \
    char side = (SIDE_CHAR);                                                    \
    int m = 2;                                                                  \
    int n = 2;                                                                  \
    int l = 1;                                                                  \
    int incv = 1;                                                               \
    int ldc = 2;                                                                \
    CTYPE v[1] = { MAKE_FN(11, 21) };                                           \
    CTYPE tau = MAKE_FN(13, 23);                                                \
    CTYPE c[4] = { MAKE_FN(0, 0), MAKE_FN(0, 0), MAKE_FN(0, 0), MAKE_FN(0, 0) }; \
    CTYPE work[2] = { MAKE_FN(0, 0), MAKE_FN(0, 0) };                           \
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
    thunk(&side, &m, &n, &l, v, &incv, &tau, c, &ldc, work);                   \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                  \
        g_##SUFFIX##_cblas_call.side != (SIDE_CHAR) ||                          \
        g_##SUFFIX##_cblas_call.m != 2 ||                                       \
        g_##SUFFIX##_cblas_call.n != 2 ||                                       \
        g_##SUFFIX##_cblas_call.l != 1 ||                                       \
        g_##SUFFIX##_cblas_call.v != v ||                                       \
        g_##SUFFIX##_cblas_call.incv != 1 ||                                    \
        g_##SUFFIX##_cblas_call.tau != &tau ||                                  \
        g_##SUFFIX##_cblas_call.c != c ||                                       \
        g_##SUFFIX##_cblas_call.ldc != 2 ||                                     \
        g_##SUFFIX##_cblas_call.work != work ||                                 \
        !EQ_FN(c[0], MAKE_FN((BASE) + 10, (BASE) + 20)) ||                     \
        !EQ_FN(c[3], MAKE_FN((BASE) + 11, (BASE) + 21)) ||                     \
        !EQ_FN(work[0], MAKE_FN((BASE) + 12, (BASE) + 22)) ||                  \
        !EQ_FN(work[1], MAKE_FN((BASE) + 13, (BASE) + 23))) {                  \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference complex LARZ inputs or propagate outputs correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences complex LARZ inputs and ignores the C int return\n"); \
    return 0;                                                                   \
}

DEFINE_LARZ_REAL_TESTS(slarz, float, FB_OP_SLARZ, 100, 'L')
DEFINE_LARZ_REAL_TESTS(dlarz, double, FB_OP_DLARZ, 300, 'R')
DEFINE_LARZ_COMPLEX_TESTS(clarz, fb_complex_float_t, FB_OP_CLARZ, make_cf32,
                          cf32_eq, 500, 'L')
DEFINE_LARZ_COMPLEX_TESTS(zlarz, fb_complex_double_t, FB_OP_ZLARZ, make_cf64,
                          cf64_eq, 700, 'R')

int main(void)
{
    int status = 0;

    status |= check_slarz_fortran_to_cblas();
    status |= check_slarz_cblas_to_fortran();
    status |= check_dlarz_fortran_to_cblas();
    status |= check_dlarz_cblas_to_fortran();
    status |= check_clarz_fortran_to_cblas();
    status |= check_clarz_cblas_to_fortran();
    status |= check_zlarz_fortran_to_cblas();
    status |= check_zlarz_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}