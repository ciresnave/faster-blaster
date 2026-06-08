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

#define DEFINE_LAHRD_REAL_TESTS(SUFFIX, TYPE, OP_ID, BASE)                     \
typedef int (*fb_##SUFFIX##_cblas_fn)(int n, int k, int nb, TYPE *a, int lda,  \
                                      TYPE *tau, TYPE *t, int ldt, TYPE *y,    \
                                      int ldy);                                \
typedef void (*fb_##SUFFIX##_fortran_fn)(int *n, int *k, int *nb, TYPE *a,     \
                                         int *lda, TYPE *tau, TYPE *t,         \
                                         int *ldt, TYPE *y, int *ldy,         \
                                         int *info);                          \
static struct {                                                                 \
    int called;                                                                 \
    int n;                                                                      \
    int k;                                                                      \
    int nb;                                                                     \
    TYPE *a;                                                                    \
    int lda;                                                                    \
    TYPE *tau;                                                                  \
    TYPE *t;                                                                    \
    int ldt;                                                                    \
    TYPE *y;                                                                    \
    int ldy;                                                                    \
} g_##SUFFIX##_fortran_call;                                                    \
static struct {                                                                 \
    int called;                                                                 \
    int n;                                                                      \
    int k;                                                                      \
    int nb;                                                                     \
    TYPE *a;                                                                    \
    int lda;                                                                    \
    TYPE *tau;                                                                  \
    TYPE *t;                                                                    \
    int ldt;                                                                    \
    TYPE *y;                                                                    \
    int ldy;                                                                    \
} g_##SUFFIX##_cblas_call;                                                      \
static void stub_##SUFFIX##_fortran(int *n, int *k, int *nb, TYPE *a,          \
                                    int *lda, TYPE *tau, TYPE *t, int *ldt,   \
                                    TYPE *y, int *ldy, int *info)             \
{                                                                               \
    g_##SUFFIX##_fortran_call.called += 1;                                      \
    g_##SUFFIX##_fortran_call.n = *n;                                           \
    g_##SUFFIX##_fortran_call.k = *k;                                           \
    g_##SUFFIX##_fortran_call.nb = *nb;                                         \
    g_##SUFFIX##_fortran_call.a = a;                                            \
    g_##SUFFIX##_fortran_call.lda = *lda;                                       \
    g_##SUFFIX##_fortran_call.tau = tau;                                        \
    g_##SUFFIX##_fortran_call.t = t;                                            \
    g_##SUFFIX##_fortran_call.ldt = *ldt;                                       \
    g_##SUFFIX##_fortran_call.y = y;                                            \
    g_##SUFFIX##_fortran_call.ldy = *ldy;                                       \
    tau[0] = (TYPE)((BASE) + 1);                                                \
    t[0] = (TYPE)((BASE) + 2);                                                  \
    y[0] = (TYPE)((BASE) + 3);                                                  \
    *info = (BASE) + 4;                                                         \
}                                                                               \
static int stub_##SUFFIX##_cblas(int n, int k, int nb, TYPE *a, int lda,       \
                                 TYPE *tau, TYPE *t, int ldt, TYPE *y,         \
                                 int ldy)                                      \
{                                                                               \
    g_##SUFFIX##_cblas_call.called += 1;                                        \
    g_##SUFFIX##_cblas_call.n = n;                                              \
    g_##SUFFIX##_cblas_call.k = k;                                              \
    g_##SUFFIX##_cblas_call.nb = nb;                                            \
    g_##SUFFIX##_cblas_call.a = a;                                              \
    g_##SUFFIX##_cblas_call.lda = lda;                                          \
    g_##SUFFIX##_cblas_call.tau = tau;                                          \
    g_##SUFFIX##_cblas_call.t = t;                                              \
    g_##SUFFIX##_cblas_call.ldt = ldt;                                          \
    g_##SUFFIX##_cblas_call.y = y;                                              \
    g_##SUFFIX##_cblas_call.ldy = ldy;                                          \
    tau[0] = (TYPE)((BASE) + 5);                                                \
    t[0] = (TYPE)((BASE) + 6);                                                  \
    y[0] = (TYPE)((BASE) + 7);                                                  \
    return (BASE) + 8;                                                          \
}                                                                               \
static int check_##SUFFIX##_fortran_to_cblas(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                        \
    TYPE a[4] = { (TYPE)1, (TYPE)2, (TYPE)3, (TYPE)4 };                        \
    TYPE tau[4] = { (TYPE)0, (TYPE)0, (TYPE)0, (TYPE)0 };                      \
    TYPE t[4] = { (TYPE)11, (TYPE)12, (TYPE)13, (TYPE)14 };                    \
    TYPE y[4] = { (TYPE)21, (TYPE)22, (TYPE)23, (TYPE)24 };                    \
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
    if (thunk(3, 1, 2, a, 3, tau, t, 2, y, 3) != (BASE) + 4 ||                 \
        g_##SUFFIX##_fortran_call.called != 1 ||                                \
        g_##SUFFIX##_fortran_call.n != 3 ||                                     \
        g_##SUFFIX##_fortran_call.k != 1 ||                                     \
        g_##SUFFIX##_fortran_call.nb != 2 ||                                    \
        g_##SUFFIX##_fortran_call.a != a ||                                     \
        g_##SUFFIX##_fortran_call.lda != 3 ||                                   \
        g_##SUFFIX##_fortran_call.tau != tau ||                                 \
        g_##SUFFIX##_fortran_call.t != t ||                                     \
        g_##SUFFIX##_fortran_call.ldt != 2 ||                                   \
        g_##SUFFIX##_fortran_call.y != y ||                                     \
        g_##SUFFIX##_fortran_call.ldy != 3 ||                                   \
        tau[0] != (TYPE)((BASE) + 1) ||                                         \
        t[0] != (TYPE)((BASE) + 2) ||                                           \
        y[0] != (TYPE)((BASE) + 3)) {                                           \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward LAHRD real inputs or return info correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards real LAHRD inputs and returns info\n"); \
    return 0;                                                                   \
}                                                                               \
static int check_##SUFFIX##_cblas_to_fortran(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                      \
    int n = 3;                                                                  \
    int k = 1;                                                                  \
    int nb = 2;                                                                 \
    int lda = 3;                                                                \
    int ldt = 2;                                                                \
    int ldy = 3;                                                                \
    int info = -1;                                                              \
    TYPE a[4] = { (TYPE)31, (TYPE)32, (TYPE)33, (TYPE)34 };                    \
    TYPE tau[4] = { (TYPE)0, (TYPE)0, (TYPE)0, (TYPE)0 };                      \
    TYPE t[4] = { (TYPE)41, (TYPE)42, (TYPE)43, (TYPE)44 };                    \
    TYPE y[4] = { (TYPE)51, (TYPE)52, (TYPE)53, (TYPE)54 };                    \
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
    thunk(&n, &k, &nb, a, &lda, tau, t, &ldt, y, &ldy, &info);                 \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                  \
        g_##SUFFIX##_cblas_call.n != 3 ||                                       \
        g_##SUFFIX##_cblas_call.k != 1 ||                                       \
        g_##SUFFIX##_cblas_call.nb != 2 ||                                      \
        g_##SUFFIX##_cblas_call.a != a ||                                       \
        g_##SUFFIX##_cblas_call.lda != 3 ||                                     \
        g_##SUFFIX##_cblas_call.tau != tau ||                                   \
        g_##SUFFIX##_cblas_call.t != t ||                                       \
        g_##SUFFIX##_cblas_call.ldt != 2 ||                                     \
        g_##SUFFIX##_cblas_call.y != y ||                                       \
        g_##SUFFIX##_cblas_call.ldy != 3 ||                                     \
        info != (BASE) + 8 ||                                                   \
        tau[0] != (TYPE)((BASE) + 5) ||                                         \
        t[0] != (TYPE)((BASE) + 6) ||                                           \
        y[0] != (TYPE)((BASE) + 7)) {                                           \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference LAHRD real inputs or store info correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences real LAHRD inputs and stores info\n"); \
    return 0;                                                                   \
}

#define DEFINE_LAHRD_COMPLEX_TESTS(SUFFIX, CTYPE, OP_ID, MAKE_FN, EQ_FN, BASE) \
typedef int (*fb_##SUFFIX##_cblas_fn)(int n, int k, int nb, CTYPE *a, int lda, \
                                      CTYPE *tau, CTYPE *t, int ldt, CTYPE *y, \
                                      int ldy);                                \
typedef void (*fb_##SUFFIX##_fortran_fn)(int *n, int *k, int *nb, CTYPE *a,    \
                                         int *lda, CTYPE *tau, CTYPE *t,       \
                                         int *ldt, CTYPE *y, int *ldy);       \
static struct {                                                                 \
    int called;                                                                 \
    int n;                                                                      \
    int k;                                                                      \
    int nb;                                                                     \
    CTYPE *a;                                                                   \
    int lda;                                                                    \
    CTYPE *tau;                                                                 \
    CTYPE *t;                                                                   \
    int ldt;                                                                    \
    CTYPE *y;                                                                   \
    int ldy;                                                                    \
} g_##SUFFIX##_fortran_call;                                                    \
static struct {                                                                 \
    int called;                                                                 \
    int n;                                                                      \
    int k;                                                                      \
    int nb;                                                                     \
    CTYPE *a;                                                                   \
    int lda;                                                                    \
    CTYPE *tau;                                                                 \
    CTYPE *t;                                                                   \
    int ldt;                                                                    \
    CTYPE *y;                                                                   \
    int ldy;                                                                    \
} g_##SUFFIX##_cblas_call;                                                      \
static void stub_##SUFFIX##_fortran(int *n, int *k, int *nb, CTYPE *a,         \
                                    int *lda, CTYPE *tau, CTYPE *t, int *ldt, \
                                    CTYPE *y, int *ldy)                       \
{                                                                               \
    g_##SUFFIX##_fortran_call.called += 1;                                      \
    g_##SUFFIX##_fortran_call.n = *n;                                           \
    g_##SUFFIX##_fortran_call.k = *k;                                           \
    g_##SUFFIX##_fortran_call.nb = *nb;                                         \
    g_##SUFFIX##_fortran_call.a = a;                                            \
    g_##SUFFIX##_fortran_call.lda = *lda;                                       \
    g_##SUFFIX##_fortran_call.tau = tau;                                        \
    g_##SUFFIX##_fortran_call.t = t;                                            \
    g_##SUFFIX##_fortran_call.ldt = *ldt;                                       \
    g_##SUFFIX##_fortran_call.y = y;                                            \
    g_##SUFFIX##_fortran_call.ldy = *ldy;                                       \
    tau[0] = MAKE_FN((BASE) + 1, (BASE) + 11);                                  \
    t[0] = MAKE_FN((BASE) + 2, (BASE) + 12);                                    \
    y[0] = MAKE_FN((BASE) + 3, (BASE) + 13);                                    \
}                                                                               \
static int stub_##SUFFIX##_cblas(int n, int k, int nb, CTYPE *a, int lda,      \
                                 CTYPE *tau, CTYPE *t, int ldt, CTYPE *y,      \
                                 int ldy)                                      \
{                                                                               \
    g_##SUFFIX##_cblas_call.called += 1;                                        \
    g_##SUFFIX##_cblas_call.n = n;                                              \
    g_##SUFFIX##_cblas_call.k = k;                                              \
    g_##SUFFIX##_cblas_call.nb = nb;                                            \
    g_##SUFFIX##_cblas_call.a = a;                                              \
    g_##SUFFIX##_cblas_call.lda = lda;                                          \
    g_##SUFFIX##_cblas_call.tau = tau;                                          \
    g_##SUFFIX##_cblas_call.t = t;                                              \
    g_##SUFFIX##_cblas_call.ldt = ldt;                                          \
    g_##SUFFIX##_cblas_call.y = y;                                              \
    g_##SUFFIX##_cblas_call.ldy = ldy;                                          \
    tau[0] = MAKE_FN((BASE) + 5, (BASE) + 15);                                  \
    t[0] = MAKE_FN((BASE) + 6, (BASE) + 16);                                    \
    y[0] = MAKE_FN((BASE) + 7, (BASE) + 17);                                    \
    return (BASE) + 8;                                                          \
}                                                                               \
static int check_##SUFFIX##_fortran_to_cblas(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                        \
    CTYPE a[4];                                                                 \
    CTYPE tau[4];                                                               \
    CTYPE t[4];                                                                 \
    CTYPE y[4];                                                                 \
    a[0] = MAKE_FN(1, 21);                                                      \
    a[1] = MAKE_FN(2, 22);                                                      \
    a[2] = MAKE_FN(3, 23);                                                      \
    a[3] = MAKE_FN(4, 24);                                                      \
    tau[0] = MAKE_FN(0, 0);                                                     \
    tau[1] = MAKE_FN(0, 0);                                                     \
    tau[2] = MAKE_FN(0, 0);                                                     \
    tau[3] = MAKE_FN(0, 0);                                                     \
    t[0] = MAKE_FN(11, 31);                                                     \
    t[1] = MAKE_FN(12, 32);                                                     \
    t[2] = MAKE_FN(13, 33);                                                     \
    t[3] = MAKE_FN(14, 34);                                                     \
    y[0] = MAKE_FN(21, 41);                                                     \
    y[1] = MAKE_FN(22, 42);                                                     \
    y[2] = MAKE_FN(23, 43);                                                     \
    y[3] = MAKE_FN(24, 44);                                                     \
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
    if (thunk(3, 1, 2, a, 3, tau, t, 2, y, 3) != 0 ||                          \
        g_##SUFFIX##_fortran_call.called != 1 ||                                \
        g_##SUFFIX##_fortran_call.n != 3 ||                                     \
        g_##SUFFIX##_fortran_call.k != 1 ||                                     \
        g_##SUFFIX##_fortran_call.nb != 2 ||                                    \
        g_##SUFFIX##_fortran_call.a != a ||                                     \
        g_##SUFFIX##_fortran_call.lda != 3 ||                                   \
        g_##SUFFIX##_fortran_call.tau != tau ||                                 \
        g_##SUFFIX##_fortran_call.t != t ||                                     \
        g_##SUFFIX##_fortran_call.ldt != 2 ||                                   \
        g_##SUFFIX##_fortran_call.y != y ||                                     \
        g_##SUFFIX##_fortran_call.ldy != 3 ||                                   \
        !EQ_FN(tau[0], MAKE_FN((BASE) + 1, (BASE) + 11)) ||                    \
        !EQ_FN(t[0], MAKE_FN((BASE) + 2, (BASE) + 12)) ||                      \
        !EQ_FN(y[0], MAKE_FN((BASE) + 3, (BASE) + 13))) {                      \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward LAHRD complex inputs correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards complex LAHRD inputs and returns success\n"); \
    return 0;                                                                   \
}                                                                               \
static int check_##SUFFIX##_cblas_to_fortran(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                      \
    int n = 3;                                                                  \
    int k = 1;                                                                  \
    int nb = 2;                                                                 \
    int lda = 3;                                                                \
    int ldt = 2;                                                                \
    int ldy = 3;                                                                \
    CTYPE a[4];                                                                 \
    CTYPE tau[4];                                                               \
    CTYPE t[4];                                                                 \
    CTYPE y[4];                                                                 \
    a[0] = MAKE_FN(31, 51);                                                     \
    a[1] = MAKE_FN(32, 52);                                                     \
    a[2] = MAKE_FN(33, 53);                                                     \
    a[3] = MAKE_FN(34, 54);                                                     \
    tau[0] = MAKE_FN(0, 0);                                                     \
    tau[1] = MAKE_FN(0, 0);                                                     \
    tau[2] = MAKE_FN(0, 0);                                                     \
    tau[3] = MAKE_FN(0, 0);                                                     \
    t[0] = MAKE_FN(41, 61);                                                     \
    t[1] = MAKE_FN(42, 62);                                                     \
    t[2] = MAKE_FN(43, 63);                                                     \
    t[3] = MAKE_FN(44, 64);                                                     \
    y[0] = MAKE_FN(51, 71);                                                     \
    y[1] = MAKE_FN(52, 72);                                                     \
    y[2] = MAKE_FN(53, 73);                                                     \
    y[3] = MAKE_FN(54, 74);                                                     \
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
    thunk(&n, &k, &nb, a, &lda, tau, t, &ldt, y, &ldy);                        \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                  \
        g_##SUFFIX##_cblas_call.n != 3 ||                                       \
        g_##SUFFIX##_cblas_call.k != 1 ||                                       \
        g_##SUFFIX##_cblas_call.nb != 2 ||                                      \
        g_##SUFFIX##_cblas_call.a != a ||                                       \
        g_##SUFFIX##_cblas_call.lda != 3 ||                                     \
        g_##SUFFIX##_cblas_call.tau != tau ||                                   \
        g_##SUFFIX##_cblas_call.t != t ||                                       \
        g_##SUFFIX##_cblas_call.ldt != 2 ||                                     \
        g_##SUFFIX##_cblas_call.y != y ||                                       \
        g_##SUFFIX##_cblas_call.ldy != 3 ||                                     \
        !EQ_FN(tau[0], MAKE_FN((BASE) + 5, (BASE) + 15)) ||                    \
        !EQ_FN(t[0], MAKE_FN((BASE) + 6, (BASE) + 16)) ||                      \
        !EQ_FN(y[0], MAKE_FN((BASE) + 7, (BASE) + 17))) {                      \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference LAHRD complex inputs correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences complex LAHRD inputs and ignores the C int return\n"); \
    return 0;                                                                   \
}

DEFINE_LAHRD_REAL_TESTS(slahrd, float, FB_OP_SLAHRD, 100)
DEFINE_LAHRD_REAL_TESTS(dlahrd, double, FB_OP_DLAHRD, 300)
DEFINE_LAHRD_COMPLEX_TESTS(clahrd, fb_complex_float_t, FB_OP_CLAHRD,
                           make_cf32, cf32_eq, 500)
DEFINE_LAHRD_COMPLEX_TESTS(zlahrd, fb_complex_double_t, FB_OP_ZLAHRD,
                           make_cf64, cf64_eq, 700)

int main(void)
{
    int status = 0;

    status |= check_slahrd_fortran_to_cblas();
    status |= check_slahrd_cblas_to_fortran();
    status |= check_dlahrd_fortran_to_cblas();
    status |= check_dlahrd_cblas_to_fortran();
    status |= check_clahrd_fortran_to_cblas();
    status |= check_clahrd_cblas_to_fortran();
    status |= check_zlahrd_fortran_to_cblas();
    status |= check_zlahrd_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}