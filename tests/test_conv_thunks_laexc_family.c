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

#define DEFINE_LAEXC_REAL_TESTS(SUFFIX, TYPE, OP_ID, BASE)                     \
typedef int (*fb_##SUFFIX##_cblas_fn)(int wantq, int n, TYPE *t, int ldt,      \
                                      TYPE *q, int ldq, int j1, int n1,       \
                                      int n2, TYPE *work);                     \
typedef void (*fb_##SUFFIX##_fortran_fn)(int *wantq, int *n, TYPE *t,         \
                                         int *ldt, TYPE *q, int *ldq,         \
                                         int *j1, int *n1, int *n2,          \
                                         TYPE *work, int *info);              \
static struct {                                                                 \
    int called;                                                                 \
    int wantq;                                                                  \
    int n;                                                                      \
    TYPE *t;                                                                    \
    int ldt;                                                                    \
    TYPE *q;                                                                    \
    int ldq;                                                                    \
    int j1;                                                                     \
    int n1;                                                                     \
    int n2;                                                                     \
    TYPE *work;                                                                 \
} g_##SUFFIX##_fortran_call;                                                    \
static struct {                                                                 \
    int called;                                                                 \
    int wantq;                                                                  \
    int n;                                                                      \
    TYPE *t;                                                                    \
    int ldt;                                                                    \
    TYPE *q;                                                                    \
    int ldq;                                                                    \
    int j1;                                                                     \
    int n1;                                                                     \
    int n2;                                                                     \
    TYPE *work;                                                                 \
} g_##SUFFIX##_cblas_call;                                                      \
static void stub_##SUFFIX##_fortran(int *wantq, int *n, TYPE *t, int *ldt,    \
                                    TYPE *q, int *ldq, int *j1, int *n1,      \
                                    int *n2, TYPE *work, int *info)           \
{                                                                               \
    g_##SUFFIX##_fortran_call.called += 1;                                      \
    g_##SUFFIX##_fortran_call.wantq = *wantq;                                   \
    g_##SUFFIX##_fortran_call.n = *n;                                           \
    g_##SUFFIX##_fortran_call.t = t;                                            \
    g_##SUFFIX##_fortran_call.ldt = *ldt;                                       \
    g_##SUFFIX##_fortran_call.q = q;                                            \
    g_##SUFFIX##_fortran_call.ldq = *ldq;                                       \
    g_##SUFFIX##_fortran_call.j1 = *j1;                                         \
    g_##SUFFIX##_fortran_call.n1 = *n1;                                         \
    g_##SUFFIX##_fortran_call.n2 = *n2;                                         \
    g_##SUFFIX##_fortran_call.work = work;                                      \
    t[0] = (TYPE)((BASE) + 1);                                                  \
    q[0] = (TYPE)((BASE) + 2);                                                  \
    work[0] = (TYPE)((BASE) + 3);                                               \
    *info = (BASE) + 4;                                                         \
}                                                                               \
static int stub_##SUFFIX##_cblas(int wantq, int n, TYPE *t, int ldt, TYPE *q, \
                                 int ldq, int j1, int n1, int n2,             \
                                 TYPE *work)                                   \
{                                                                               \
    g_##SUFFIX##_cblas_call.called += 1;                                        \
    g_##SUFFIX##_cblas_call.wantq = wantq;                                      \
    g_##SUFFIX##_cblas_call.n = n;                                              \
    g_##SUFFIX##_cblas_call.t = t;                                              \
    g_##SUFFIX##_cblas_call.ldt = ldt;                                          \
    g_##SUFFIX##_cblas_call.q = q;                                              \
    g_##SUFFIX##_cblas_call.ldq = ldq;                                          \
    g_##SUFFIX##_cblas_call.j1 = j1;                                            \
    g_##SUFFIX##_cblas_call.n1 = n1;                                            \
    g_##SUFFIX##_cblas_call.n2 = n2;                                            \
    g_##SUFFIX##_cblas_call.work = work;                                        \
    t[0] = (TYPE)((BASE) + 5);                                                  \
    q[0] = (TYPE)((BASE) + 6);                                                  \
    work[0] = (TYPE)((BASE) + 7);                                               \
    return (BASE) + 8;                                                          \
}                                                                               \
static int check_##SUFFIX##_fortran_to_cblas(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                        \
    TYPE t[4] = { (TYPE)1, (TYPE)2, (TYPE)3, (TYPE)4 };                        \
    TYPE q[4] = { (TYPE)11, (TYPE)12, (TYPE)13, (TYPE)14 };                    \
    TYPE work[4] = { (TYPE)21, (TYPE)22, (TYPE)23, (TYPE)24 };                 \
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
    if (thunk(1, 2, t, 2, q, 2, 1, 1, 1, work) != (BASE) + 4 ||                \
        g_##SUFFIX##_fortran_call.called != 1 ||                                \
        g_##SUFFIX##_fortran_call.wantq != 1 ||                                 \
        g_##SUFFIX##_fortran_call.n != 2 ||                                     \
        g_##SUFFIX##_fortran_call.t != t ||                                     \
        g_##SUFFIX##_fortran_call.ldt != 2 ||                                   \
        g_##SUFFIX##_fortran_call.q != q ||                                     \
        g_##SUFFIX##_fortran_call.ldq != 2 ||                                   \
        g_##SUFFIX##_fortran_call.j1 != 1 ||                                    \
        g_##SUFFIX##_fortran_call.n1 != 1 ||                                    \
        g_##SUFFIX##_fortran_call.n2 != 1 ||                                    \
        g_##SUFFIX##_fortran_call.work != work ||                               \
        t[0] != (TYPE)((BASE) + 1) ||                                           \
        q[0] != (TYPE)((BASE) + 2) ||                                           \
        work[0] != (TYPE)((BASE) + 3)) {                                        \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward LAEXC scalars, matrix pointers, or info correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards no-layout LAEXC inputs and returns info\n"); \
    return 0;                                                                   \
}                                                                               \
static int check_##SUFFIX##_cblas_to_fortran(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                      \
    int wantq = 1;                                                              \
    int n = 2;                                                                  \
    int ldt = 2;                                                                \
    int ldq = 2;                                                                \
    int j1 = 1;                                                                 \
    int n1 = 1;                                                                 \
    int n2 = 1;                                                                 \
    int info = -1;                                                              \
    TYPE t[4] = { (TYPE)31, (TYPE)32, (TYPE)33, (TYPE)34 };                    \
    TYPE q[4] = { (TYPE)41, (TYPE)42, (TYPE)43, (TYPE)44 };                    \
    TYPE work[4] = { (TYPE)51, (TYPE)52, (TYPE)53, (TYPE)54 };                 \
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
    thunk(&wantq, &n, t, &ldt, q, &ldq, &j1, &n1, &n2, work, &info);           \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                  \
        g_##SUFFIX##_cblas_call.wantq != 1 ||                                   \
        g_##SUFFIX##_cblas_call.n != 2 ||                                       \
        g_##SUFFIX##_cblas_call.t != t ||                                       \
        g_##SUFFIX##_cblas_call.ldt != 2 ||                                     \
        g_##SUFFIX##_cblas_call.q != q ||                                       \
        g_##SUFFIX##_cblas_call.ldq != 2 ||                                     \
        g_##SUFFIX##_cblas_call.j1 != 1 ||                                      \
        g_##SUFFIX##_cblas_call.n1 != 1 ||                                      \
        g_##SUFFIX##_cblas_call.n2 != 1 ||                                      \
        g_##SUFFIX##_cblas_call.work != work ||                                 \
        info != (BASE) + 8 ||                                                   \
        t[0] != (TYPE)((BASE) + 5) ||                                           \
        q[0] != (TYPE)((BASE) + 6) ||                                           \
        work[0] != (TYPE)((BASE) + 7)) {                                        \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference LAEXC scalars or propagate info correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences no-layout LAEXC inputs and stores info\n"); \
    return 0;                                                                   \
}

#define DEFINE_LAEXC_COMPLEX_TESTS(SUFFIX, CTYPE, OP_ID, MAKE_FN, EQ_FN, BASE) \
typedef int (*fb_##SUFFIX##_cblas_fn)(int wantq, int n, CTYPE *t, int ldt,     \
                                      CTYPE *q, int ldq, int j1, int n1,      \
                                      int n2, CTYPE *work);                    \
typedef void (*fb_##SUFFIX##_fortran_fn)(int *wantq, int *n, CTYPE *t,        \
                                         int *ldt, CTYPE *q, int *ldq,        \
                                         int *j1, int *n1, int *n2,          \
                                         CTYPE *work, int *info);             \
static struct {                                                                 \
    int called;                                                                 \
    int wantq;                                                                  \
    int n;                                                                      \
    CTYPE *t;                                                                   \
    int ldt;                                                                    \
    CTYPE *q;                                                                   \
    int ldq;                                                                    \
    int j1;                                                                     \
    int n1;                                                                     \
    int n2;                                                                     \
    CTYPE *work;                                                                \
} g_##SUFFIX##_fortran_call;                                                    \
static struct {                                                                 \
    int called;                                                                 \
    int wantq;                                                                  \
    int n;                                                                      \
    CTYPE *t;                                                                   \
    int ldt;                                                                    \
    CTYPE *q;                                                                   \
    int ldq;                                                                    \
    int j1;                                                                     \
    int n1;                                                                     \
    int n2;                                                                     \
    CTYPE *work;                                                                \
} g_##SUFFIX##_cblas_call;                                                      \
static void stub_##SUFFIX##_fortran(int *wantq, int *n, CTYPE *t, int *ldt,   \
                                    CTYPE *q, int *ldq, int *j1, int *n1,     \
                                    int *n2, CTYPE *work, int *info)          \
{                                                                               \
    g_##SUFFIX##_fortran_call.called += 1;                                      \
    g_##SUFFIX##_fortran_call.wantq = *wantq;                                   \
    g_##SUFFIX##_fortran_call.n = *n;                                           \
    g_##SUFFIX##_fortran_call.t = t;                                            \
    g_##SUFFIX##_fortran_call.ldt = *ldt;                                       \
    g_##SUFFIX##_fortran_call.q = q;                                            \
    g_##SUFFIX##_fortran_call.ldq = *ldq;                                       \
    g_##SUFFIX##_fortran_call.j1 = *j1;                                         \
    g_##SUFFIX##_fortran_call.n1 = *n1;                                         \
    g_##SUFFIX##_fortran_call.n2 = *n2;                                         \
    g_##SUFFIX##_fortran_call.work = work;                                      \
    t[0] = MAKE_FN((BASE) + 1, (BASE) + 11);                                    \
    q[0] = MAKE_FN((BASE) + 2, (BASE) + 12);                                    \
    work[0] = MAKE_FN((BASE) + 3, (BASE) + 13);                                 \
    *info = (BASE) + 4;                                                         \
}                                                                               \
static int stub_##SUFFIX##_cblas(int wantq, int n, CTYPE *t, int ldt,         \
                                 CTYPE *q, int ldq, int j1, int n1, int n2,   \
                                 CTYPE *work)                                  \
{                                                                               \
    g_##SUFFIX##_cblas_call.called += 1;                                        \
    g_##SUFFIX##_cblas_call.wantq = wantq;                                      \
    g_##SUFFIX##_cblas_call.n = n;                                              \
    g_##SUFFIX##_cblas_call.t = t;                                              \
    g_##SUFFIX##_cblas_call.ldt = ldt;                                          \
    g_##SUFFIX##_cblas_call.q = q;                                              \
    g_##SUFFIX##_cblas_call.ldq = ldq;                                          \
    g_##SUFFIX##_cblas_call.j1 = j1;                                            \
    g_##SUFFIX##_cblas_call.n1 = n1;                                            \
    g_##SUFFIX##_cblas_call.n2 = n2;                                            \
    g_##SUFFIX##_cblas_call.work = work;                                        \
    t[0] = MAKE_FN((BASE) + 5, (BASE) + 15);                                    \
    q[0] = MAKE_FN((BASE) + 6, (BASE) + 16);                                    \
    work[0] = MAKE_FN((BASE) + 7, (BASE) + 17);                                 \
    return (BASE) + 8;                                                          \
}                                                                               \
static int check_##SUFFIX##_fortran_to_cblas(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                        \
    CTYPE t[4];                                                                 \
    CTYPE q[4];                                                                 \
    CTYPE work[4];                                                              \
    t[0] = MAKE_FN(1, 21);                                                      \
    t[1] = MAKE_FN(2, 22);                                                      \
    t[2] = MAKE_FN(3, 23);                                                      \
    t[3] = MAKE_FN(4, 24);                                                      \
    q[0] = MAKE_FN(11, 31);                                                     \
    q[1] = MAKE_FN(12, 32);                                                     \
    q[2] = MAKE_FN(13, 33);                                                     \
    q[3] = MAKE_FN(14, 34);                                                     \
    work[0] = MAKE_FN(21, 41);                                                  \
    work[1] = MAKE_FN(22, 42);                                                  \
    work[2] = MAKE_FN(23, 43);                                                  \
    work[3] = MAKE_FN(24, 44);                                                  \
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
    if (thunk(1, 2, t, 2, q, 2, 1, 1, 1, work) != (BASE) + 4 ||                \
        g_##SUFFIX##_fortran_call.called != 1 ||                                \
        g_##SUFFIX##_fortran_call.wantq != 1 ||                                 \
        g_##SUFFIX##_fortran_call.n != 2 ||                                     \
        g_##SUFFIX##_fortran_call.t != t ||                                     \
        g_##SUFFIX##_fortran_call.ldt != 2 ||                                   \
        g_##SUFFIX##_fortran_call.q != q ||                                     \
        g_##SUFFIX##_fortran_call.ldq != 2 ||                                   \
        g_##SUFFIX##_fortran_call.j1 != 1 ||                                    \
        g_##SUFFIX##_fortran_call.n1 != 1 ||                                    \
        g_##SUFFIX##_fortran_call.n2 != 1 ||                                    \
        g_##SUFFIX##_fortran_call.work != work ||                               \
        !EQ_FN(t[0], MAKE_FN((BASE) + 1, (BASE) + 11)) ||                      \
        !EQ_FN(q[0], MAKE_FN((BASE) + 2, (BASE) + 12)) ||                      \
        !EQ_FN(work[0], MAKE_FN((BASE) + 3, (BASE) + 13))) {                   \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward complex LAEXC inputs or info correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards complex LAEXC inputs and returns info\n"); \
    return 0;                                                                   \
}                                                                               \
static int check_##SUFFIX##_cblas_to_fortran(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                      \
    int wantq = 1;                                                              \
    int n = 2;                                                                  \
    int ldt = 2;                                                                \
    int ldq = 2;                                                                \
    int j1 = 1;                                                                 \
    int n1 = 1;                                                                 \
    int n2 = 1;                                                                 \
    int info = -1;                                                              \
    CTYPE t[4];                                                                 \
    CTYPE q[4];                                                                 \
    CTYPE work[4];                                                              \
    t[0] = MAKE_FN(31, 51);                                                     \
    t[1] = MAKE_FN(32, 52);                                                     \
    t[2] = MAKE_FN(33, 53);                                                     \
    t[3] = MAKE_FN(34, 54);                                                     \
    q[0] = MAKE_FN(41, 61);                                                     \
    q[1] = MAKE_FN(42, 62);                                                     \
    q[2] = MAKE_FN(43, 63);                                                     \
    q[3] = MAKE_FN(44, 64);                                                     \
    work[0] = MAKE_FN(51, 71);                                                  \
    work[1] = MAKE_FN(52, 72);                                                  \
    work[2] = MAKE_FN(53, 73);                                                  \
    work[3] = MAKE_FN(54, 74);                                                  \
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
    thunk(&wantq, &n, t, &ldt, q, &ldq, &j1, &n1, &n2, work, &info);           \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                  \
        g_##SUFFIX##_cblas_call.wantq != 1 ||                                   \
        g_##SUFFIX##_cblas_call.n != 2 ||                                       \
        g_##SUFFIX##_cblas_call.t != t ||                                       \
        g_##SUFFIX##_cblas_call.ldt != 2 ||                                     \
        g_##SUFFIX##_cblas_call.q != q ||                                       \
        g_##SUFFIX##_cblas_call.ldq != 2 ||                                     \
        g_##SUFFIX##_cblas_call.j1 != 1 ||                                      \
        g_##SUFFIX##_cblas_call.n1 != 1 ||                                      \
        g_##SUFFIX##_cblas_call.n2 != 1 ||                                      \
        g_##SUFFIX##_cblas_call.work != work ||                                 \
        info != (BASE) + 8 ||                                                   \
        !EQ_FN(t[0], MAKE_FN((BASE) + 5, (BASE) + 15)) ||                      \
        !EQ_FN(q[0], MAKE_FN((BASE) + 6, (BASE) + 16)) ||                      \
        !EQ_FN(work[0], MAKE_FN((BASE) + 7, (BASE) + 17))) {                   \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference complex LAEXC inputs or propagate info correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences complex LAEXC inputs and stores info\n"); \
    return 0;                                                                   \
}

DEFINE_LAEXC_REAL_TESTS(slaexc, float, FB_OP_SLAEXC, 100)
DEFINE_LAEXC_REAL_TESTS(dlaexc, double, FB_OP_DLAEXC, 300)
DEFINE_LAEXC_COMPLEX_TESTS(claexc, fb_complex_float_t, FB_OP_CLAEXC,
                           make_cf32, cf32_eq, 500)
DEFINE_LAEXC_COMPLEX_TESTS(zlaexc, fb_complex_double_t, FB_OP_ZLAEXC,
                           make_cf64, cf64_eq, 700)

int main(void)
{
    int status = 0;

    status |= check_slaexc_fortran_to_cblas();
    status |= check_slaexc_cblas_to_fortran();
    status |= check_dlaexc_fortran_to_cblas();
    status |= check_dlaexc_cblas_to_fortran();
    status |= check_claexc_fortran_to_cblas();
    status |= check_claexc_cblas_to_fortran();
    status |= check_zlaexc_fortran_to_cblas();
    status |= check_zlaexc_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}