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

#define DEFINE_LARFX_REAL_TESTS(SUFFIX, TYPE, OP_ID, BASE)                     \
typedef int (*fb_##SUFFIX##_cblas_fn)(fb_layout_t layout, char side, int m,    \
                                      int n, const TYPE *v, TYPE tau, TYPE *c, \
                                      int ldc, TYPE *work);                    \
typedef void (*fb_##SUFFIX##_fortran_fn)(char *side, int *m, int *n, TYPE *v,  \
                                         TYPE *tau, TYPE *c, int *ldc,         \
                                         TYPE *work, int *info);               \
static struct {                                                                  \
    int called;                                                                  \
    char side;                                                                   \
    int m;                                                                       \
    int n;                                                                       \
    TYPE *v;                                                                     \
    TYPE tau_value;                                                              \
    TYPE *c;                                                                     \
    int ldc;                                                                     \
    TYPE *work;                                                                  \
    int *info_ptr;                                                               \
    TYPE c_snapshot[4];                                                          \
} g_##SUFFIX##_fortran_call;                                                     \
static struct {                                                                  \
    int called;                                                                  \
    fb_layout_t layout;                                                          \
    char side;                                                                   \
    int m;                                                                       \
    int n;                                                                       \
    const TYPE *v;                                                               \
    TYPE tau_value;                                                              \
    TYPE *c;                                                                     \
    int ldc;                                                                     \
    TYPE *work;                                                                  \
} g_##SUFFIX##_cblas_call;                                                       \
static void stub_##SUFFIX##_fortran(char *side, int *m, int *n, TYPE *v,       \
                                    TYPE *tau, TYPE *c, int *ldc, TYPE *work,  \
                                    int *info)                                  \
{                                                                                \
    g_##SUFFIX##_fortran_call.called += 1;                                       \
    g_##SUFFIX##_fortran_call.side = *side;                                      \
    g_##SUFFIX##_fortran_call.m = *m;                                            \
    g_##SUFFIX##_fortran_call.n = *n;                                            \
    g_##SUFFIX##_fortran_call.v = v;                                             \
    g_##SUFFIX##_fortran_call.tau_value = *tau;                                  \
    g_##SUFFIX##_fortran_call.c = c;                                             \
    g_##SUFFIX##_fortran_call.ldc = *ldc;                                        \
    g_##SUFFIX##_fortran_call.work = work;                                       \
    g_##SUFFIX##_fortran_call.info_ptr = info;                                   \
    g_##SUFFIX##_fortran_call.c_snapshot[0] = c[0];                              \
    g_##SUFFIX##_fortran_call.c_snapshot[1] = c[1];                              \
    g_##SUFFIX##_fortran_call.c_snapshot[2] = c[2];                              \
    g_##SUFFIX##_fortran_call.c_snapshot[3] = c[3];                              \
    c[0] = (TYPE)((BASE) + 1);                                                   \
    c[1] = (TYPE)((BASE) + 2);                                                   \
    c[2] = (TYPE)((BASE) + 3);                                                   \
    c[3] = (TYPE)((BASE) + 4);                                                   \
    *info = 0;                                                                   \
}                                                                                \
static int stub_##SUFFIX##_cblas(fb_layout_t layout, char side, int m, int n,  \
                                 const TYPE *v, TYPE tau, TYPE *c, int ldc,    \
                                 TYPE *work)                                    \
{                                                                                \
    g_##SUFFIX##_cblas_call.called += 1;                                         \
    g_##SUFFIX##_cblas_call.layout = layout;                                     \
    g_##SUFFIX##_cblas_call.side = side;                                         \
    g_##SUFFIX##_cblas_call.m = m;                                               \
    g_##SUFFIX##_cblas_call.n = n;                                               \
    g_##SUFFIX##_cblas_call.v = v;                                               \
    g_##SUFFIX##_cblas_call.tau_value = tau;                                     \
    g_##SUFFIX##_cblas_call.c = c;                                               \
    g_##SUFFIX##_cblas_call.ldc = ldc;                                           \
    g_##SUFFIX##_cblas_call.work = work;                                         \
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
    TYPE v[2] = { (TYPE)1, (TYPE)2 };                                            \
    TYPE tau = (TYPE)3;                                                          \
    TYPE c_row[4] = { (TYPE)9, (TYPE)10, (TYPE)11, (TYPE)12 };                  \
    TYPE work[2] = { (TYPE)13, (TYPE)14 };                                       \
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
    if (thunk(FB_LAYOUT_ROW_MAJOR, 'L', 2, 2, v, tau, c_row, 2, work) != 0 ||   \
        g_##SUFFIX##_fortran_call.called != 1 ||                                 \
        g_##SUFFIX##_fortran_call.side != 'L' ||                                 \
        g_##SUFFIX##_fortran_call.m != 2 ||                                      \
        g_##SUFFIX##_fortran_call.n != 2 ||                                      \
        g_##SUFFIX##_fortran_call.v != v ||                                      \
        g_##SUFFIX##_fortran_call.tau_value != (TYPE)3 ||                        \
        g_##SUFFIX##_fortran_call.c == c_row ||                                  \
        g_##SUFFIX##_fortran_call.ldc != 2 ||                                    \
        g_##SUFFIX##_fortran_call.work != work ||                                \
        g_##SUFFIX##_fortran_call.info_ptr == NULL ||                            \
        g_##SUFFIX##_fortran_call.c_snapshot[0] != (TYPE)9 ||                    \
        g_##SUFFIX##_fortran_call.c_snapshot[1] != (TYPE)11 ||                   \
        g_##SUFFIX##_fortran_call.c_snapshot[2] != (TYPE)10 ||                   \
        g_##SUFFIX##_fortran_call.c_snapshot[3] != (TYPE)12 ||                   \
        c_row[0] != (TYPE)((BASE) + 1) || c_row[1] != (TYPE)((BASE) + 3) ||     \
        c_row[2] != (TYPE)((BASE) + 2) || c_row[3] != (TYPE)((BASE) + 4)) {     \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not stage row-major LARFX inputs or outputs correctly\n"); \
        return 1;                                                                \
    }                                                                            \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk stages row-major LARFX inputs and stores info\n"); \
    return 0;                                                                    \
}                                                                                \
static int check_##SUFFIX##_cblas_to_fortran(void)                               \
{                                                                                \
    fb_backend_vtable_t vtable;                                                  \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                       \
    char side = 'R';                                                             \
    int m = 2;                                                                   \
    int n = 2;                                                                   \
    int ldc = 2;                                                                 \
    int info = -1;                                                               \
    TYPE v[2] = { (TYPE)21, (TYPE)22 };                                          \
    TYPE tau = (TYPE)23;                                                         \
    TYPE c[4] = { (TYPE)24, (TYPE)25, (TYPE)26, (TYPE)27 };                     \
    TYPE work[2] = { (TYPE)28, (TYPE)29 };                                       \
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
    thunk(&side, &m, &n, v, &tau, c, &ldc, work, &info);                         \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                   \
        g_##SUFFIX##_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||                 \
        g_##SUFFIX##_cblas_call.side != 'R' ||                                   \
        g_##SUFFIX##_cblas_call.m != 2 ||                                        \
        g_##SUFFIX##_cblas_call.n != 2 ||                                        \
        g_##SUFFIX##_cblas_call.v != v ||                                        \
        g_##SUFFIX##_cblas_call.tau_value != (TYPE)23 ||                         \
        g_##SUFFIX##_cblas_call.c != c ||                                        \
        g_##SUFFIX##_cblas_call.ldc != 2 ||                                      \
        g_##SUFFIX##_cblas_call.work != work ||                                  \
        info != (BASE) + 99 ||                                                   \
        c[0] != (TYPE)((BASE) + 10) || c[1] != (TYPE)((BASE) + 11) ||           \
        c[2] != (TYPE)((BASE) + 12) || c[3] != (TYPE)((BASE) + 13)) {           \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not forward LARFX inputs or store info correctly\n"); \
        return 1;                                                                \
    }                                                                            \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences LARFX inputs and stores info\n"); \
    return 0;                                                                    \
}

#define DEFINE_LARFX_COMPLEX_TESTS(SUFFIX, CTYPE, OP_ID, MAKE_FN, EQ_FN, BASE) \
typedef int (*fb_##SUFFIX##_cblas_fn)(fb_layout_t layout, char side, int m,    \
                                      int n, const CTYPE *v, CTYPE tau, CTYPE *c, \
                                      int ldc, CTYPE *work);                   \
typedef void (*fb_##SUFFIX##_fortran_fn)(char *side, char *trans, int *m, int *n, \
                                         int *k, CTYPE *v, int *incv, CTYPE *tau, \
                                         CTYPE *c, int *ldc, CTYPE *work);     \
static struct {                                                                  \
    int called;                                                                  \
    char side;                                                                   \
    char trans;                                                                  \
    int m;                                                                       \
    int n;                                                                       \
    int k;                                                                       \
    CTYPE *v;                                                                    \
    int incv;                                                                    \
    CTYPE tau_value;                                                             \
    CTYPE *c;                                                                    \
    int ldc;                                                                     \
    CTYPE *work;                                                                 \
    CTYPE c_snapshot[4];                                                         \
} g_##SUFFIX##_fortran_call;                                                     \
static struct {                                                                  \
    int called;                                                                  \
    fb_layout_t layout;                                                          \
    char side;                                                                   \
    int m;                                                                       \
    int n;                                                                       \
    const CTYPE *v;                                                              \
    CTYPE v_snapshot[2];                                                         \
    CTYPE tau_value;                                                             \
    CTYPE *c;                                                                    \
    int ldc;                                                                     \
    CTYPE *work;                                                                 \
} g_##SUFFIX##_cblas_call;                                                       \
static void stub_##SUFFIX##_fortran(char *side, char *trans, int *m, int *n,   \
                                    int *k, CTYPE *v, int *incv, CTYPE *tau,   \
                                    CTYPE *c, int *ldc, CTYPE *work)           \
{                                                                                \
    g_##SUFFIX##_fortran_call.called += 1;                                       \
    g_##SUFFIX##_fortran_call.side = *side;                                      \
    g_##SUFFIX##_fortran_call.trans = *trans;                                    \
    g_##SUFFIX##_fortran_call.m = *m;                                            \
    g_##SUFFIX##_fortran_call.n = *n;                                            \
    g_##SUFFIX##_fortran_call.k = *k;                                            \
    g_##SUFFIX##_fortran_call.v = v;                                             \
    g_##SUFFIX##_fortran_call.incv = *incv;                                      \
    g_##SUFFIX##_fortran_call.tau_value = *tau;                                  \
    g_##SUFFIX##_fortran_call.c = c;                                             \
    g_##SUFFIX##_fortran_call.ldc = *ldc;                                        \
    g_##SUFFIX##_fortran_call.work = work;                                       \
    g_##SUFFIX##_fortran_call.c_snapshot[0] = c[0];                              \
    g_##SUFFIX##_fortran_call.c_snapshot[1] = c[1];                              \
    g_##SUFFIX##_fortran_call.c_snapshot[2] = c[2];                              \
    g_##SUFFIX##_fortran_call.c_snapshot[3] = c[3];                              \
    c[0] = MAKE_FN((BASE) + 1, (BASE) + 11);                                     \
    c[1] = MAKE_FN((BASE) + 2, (BASE) + 12);                                     \
    c[2] = MAKE_FN((BASE) + 3, (BASE) + 13);                                     \
    c[3] = MAKE_FN((BASE) + 4, (BASE) + 14);                                     \
}                                                                                \
static int stub_##SUFFIX##_cblas(fb_layout_t layout, char side, int m, int n,  \
                                 const CTYPE *v, CTYPE tau, CTYPE *c, int ldc, \
                                 CTYPE *work)                                   \
{                                                                                \
    g_##SUFFIX##_cblas_call.called += 1;                                         \
    g_##SUFFIX##_cblas_call.layout = layout;                                     \
    g_##SUFFIX##_cblas_call.side = side;                                         \
    g_##SUFFIX##_cblas_call.m = m;                                               \
    g_##SUFFIX##_cblas_call.n = n;                                               \
    g_##SUFFIX##_cblas_call.v = v;                                               \
    g_##SUFFIX##_cblas_call.v_snapshot[0] = v ? v[0] : MAKE_FN(0, 0);           \
    g_##SUFFIX##_cblas_call.v_snapshot[1] = v ? v[1] : MAKE_FN(0, 0);           \
    g_##SUFFIX##_cblas_call.tau_value = tau;                                     \
    g_##SUFFIX##_cblas_call.c = c;                                               \
    g_##SUFFIX##_cblas_call.ldc = ldc;                                           \
    g_##SUFFIX##_cblas_call.work = work;                                         \
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
    CTYPE v[2] = { MAKE_FN(1, 11), MAKE_FN(2, 12) };                             \
    CTYPE tau = MAKE_FN(3, 13);                                                  \
    CTYPE c_row[4] = { MAKE_FN(9, 19), MAKE_FN(10, 20), MAKE_FN(11, 21), MAKE_FN(12, 22) }; \
    CTYPE work[2] = { MAKE_FN(13, 23), MAKE_FN(14, 24) };                        \
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
    if (thunk(FB_LAYOUT_ROW_MAJOR, 'L', 2, 2, v, tau, c_row, 2, work) != 0 ||   \
        g_##SUFFIX##_fortran_call.called != 1 ||                                 \
        g_##SUFFIX##_fortran_call.side != 'L' ||                                 \
        g_##SUFFIX##_fortran_call.trans != 'N' ||                                \
        g_##SUFFIX##_fortran_call.m != 2 ||                                      \
        g_##SUFFIX##_fortran_call.n != 2 ||                                      \
        g_##SUFFIX##_fortran_call.k != 2 ||                                      \
        g_##SUFFIX##_fortran_call.v != v ||                                      \
        g_##SUFFIX##_fortran_call.incv != 1 ||                                   \
        !EQ_FN(g_##SUFFIX##_fortran_call.tau_value, MAKE_FN(3, 13)) ||          \
        g_##SUFFIX##_fortran_call.c == c_row ||                                  \
        g_##SUFFIX##_fortran_call.ldc != 2 ||                                    \
        g_##SUFFIX##_fortran_call.work != work ||                                \
        !EQ_FN(g_##SUFFIX##_fortran_call.c_snapshot[0], MAKE_FN(9, 19)) ||      \
        !EQ_FN(g_##SUFFIX##_fortran_call.c_snapshot[1], MAKE_FN(11, 21)) ||     \
        !EQ_FN(g_##SUFFIX##_fortran_call.c_snapshot[2], MAKE_FN(10, 20)) ||     \
        !EQ_FN(g_##SUFFIX##_fortran_call.c_snapshot[3], MAKE_FN(12, 22)) ||     \
        !EQ_FN(c_row[0], MAKE_FN((BASE) + 1, (BASE) + 11)) ||                   \
        !EQ_FN(c_row[1], MAKE_FN((BASE) + 3, (BASE) + 13)) ||                   \
        !EQ_FN(c_row[2], MAKE_FN((BASE) + 2, (BASE) + 12)) ||                   \
        !EQ_FN(c_row[3], MAKE_FN((BASE) + 4, (BASE) + 14))) {                   \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not stage complex row-major LARFX inputs or outputs correctly\n"); \
        return 1;                                                                \
    }                                                                            \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk stages complex row-major LARFX inputs\n"); \
    return 0;                                                                    \
}                                                                                \
static int check_##SUFFIX##_cblas_to_fortran(void)                               \
{                                                                                \
    fb_backend_vtable_t vtable;                                                  \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                       \
    char side = 'L';                                                             \
    char trans = 'C';                                                            \
    int m = 2;                                                                   \
    int n = 2;                                                                   \
    int k = 2;                                                                   \
    int incv = 2;                                                                \
    int ldc = 2;                                                                 \
    CTYPE v_storage[3] = { MAKE_FN(21, 31), MAKE_FN(99, 99), MAKE_FN(22, 32) }; \
    CTYPE tau[1] = { MAKE_FN(23, 33) };                                          \
    CTYPE c[4] = { MAKE_FN(24, 34), MAKE_FN(25, 35), MAKE_FN(26, 36), MAKE_FN(27, 37) }; \
    CTYPE work[2] = { MAKE_FN(28, 38), MAKE_FN(29, 39) };                       \
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
    thunk(&side, &trans, &m, &n, &k, v_storage, &incv, tau, c, &ldc, work);     \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                   \
        g_##SUFFIX##_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||                 \
        g_##SUFFIX##_cblas_call.side != 'L' ||                                   \
        g_##SUFFIX##_cblas_call.m != 2 ||                                        \
        g_##SUFFIX##_cblas_call.n != 2 ||                                        \
        g_##SUFFIX##_cblas_call.v == v_storage ||                                \
        !EQ_FN(g_##SUFFIX##_cblas_call.v_snapshot[0], MAKE_FN(21, 31)) ||       \
        !EQ_FN(g_##SUFFIX##_cblas_call.v_snapshot[1], MAKE_FN(22, 32)) ||       \
        !EQ_FN(g_##SUFFIX##_cblas_call.tau_value, MAKE_FN(23, -33)) ||          \
        g_##SUFFIX##_cblas_call.c != c ||                                        \
        g_##SUFFIX##_cblas_call.ldc != 2 ||                                      \
        g_##SUFFIX##_cblas_call.work != work ||                                  \
        !EQ_FN(c[0], MAKE_FN((BASE) + 10, (BASE) + 20)) ||                      \
        !EQ_FN(c[1], MAKE_FN((BASE) + 11, (BASE) + 21)) ||                      \
        !EQ_FN(c[2], MAKE_FN((BASE) + 12, (BASE) + 22)) ||                      \
        !EQ_FN(c[3], MAKE_FN((BASE) + 13, (BASE) + 23))) {                      \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not reduce complex LARFX inputs correctly\n"); \
        return 1;                                                                \
    }                                                                            \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk reduces complex LARFX inputs correctly\n"); \
    return 0;                                                                    \
}

DEFINE_LARFX_REAL_TESTS(slarfx, float, FB_OP_SLARFX, 100)
DEFINE_LARFX_REAL_TESTS(dlarfx, double, FB_OP_DLARFX, 300)
DEFINE_LARFX_COMPLEX_TESTS(clarfx, fb_complex_float_t, FB_OP_CLARFX, make_cf32,
                           cf32_eq, 500)
DEFINE_LARFX_COMPLEX_TESTS(zlarfx, fb_complex_double_t, FB_OP_ZLARFX, make_cf64,
                           cf64_eq, 700)

int main(void)
{
    int status = 0;

    status |= check_slarfx_fortran_to_cblas();
    status |= check_slarfx_cblas_to_fortran();
    status |= check_dlarfx_fortran_to_cblas();
    status |= check_dlarfx_cblas_to_fortran();
    status |= check_clarfx_fortran_to_cblas();
    status |= check_clarfx_cblas_to_fortran();
    status |= check_zlarfx_fortran_to_cblas();
    status |= check_zlarfx_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}