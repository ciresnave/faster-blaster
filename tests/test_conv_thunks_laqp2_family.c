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

#define DEFINE_LAQP2_REAL_TESTS(SUFFIX, TYPE, OP_ID, BASE)                     \
typedef int (*fb_##SUFFIX##_cblas_fn)(int m, int n, int offset, TYPE *a,       \
                                      int lda, int *jpvt, TYPE *tau,          \
                                      TYPE *vn1, TYPE *vn2, TYPE *work);      \
typedef void (*fb_##SUFFIX##_fortran_fn)(int *m, int *n, int *offset, TYPE *a, \
                                         int *lda, int *jpvt, TYPE *tau,      \
                                         TYPE *vn1, TYPE *vn2, TYPE *work);   \
static struct {                                                                 \
    int called;                                                                 \
    int m;                                                                      \
    int n;                                                                      \
    int offset;                                                                 \
    TYPE *a;                                                                    \
    int lda;                                                                    \
    int *jpvt;                                                                   \
    TYPE *tau;                                                                   \
    TYPE *vn1;                                                                   \
    TYPE *vn2;                                                                   \
    TYPE *work;                                                                  \
} g_##SUFFIX##_fortran_call;                                                    \
static struct {                                                                 \
    int called;                                                                 \
    int m;                                                                      \
    int n;                                                                      \
    int offset;                                                                 \
    TYPE *a;                                                                    \
    int lda;                                                                    \
    int *jpvt;                                                                   \
    TYPE *tau;                                                                   \
    TYPE *vn1;                                                                   \
    TYPE *vn2;                                                                   \
    TYPE *work;                                                                  \
} g_##SUFFIX##_cblas_call;                                                      \
static void stub_##SUFFIX##_fortran(int *m, int *n, int *offset, TYPE *a,      \
                                    int *lda, int *jpvt, TYPE *tau, TYPE *vn1, \
                                    TYPE *vn2, TYPE *work)                     \
{                                                                               \
    g_##SUFFIX##_fortran_call.called += 1;                                      \
    g_##SUFFIX##_fortran_call.m = *m;                                           \
    g_##SUFFIX##_fortran_call.n = *n;                                           \
    g_##SUFFIX##_fortran_call.offset = *offset;                                 \
    g_##SUFFIX##_fortran_call.a = a;                                            \
    g_##SUFFIX##_fortran_call.lda = *lda;                                       \
    g_##SUFFIX##_fortran_call.jpvt = jpvt;                                      \
    g_##SUFFIX##_fortran_call.tau = tau;                                        \
    g_##SUFFIX##_fortran_call.vn1 = vn1;                                        \
    g_##SUFFIX##_fortran_call.vn2 = vn2;                                        \
    g_##SUFFIX##_fortran_call.work = work;                                      \
    jpvt[0] = (BASE) + 1;                                                       \
    tau[0] = (TYPE)((BASE) + 2);                                                \
    vn1[0] = (TYPE)((BASE) + 3);                                                \
    vn2[0] = (TYPE)((BASE) + 4);                                                \
    work[0] = (TYPE)((BASE) + 5);                                               \
}                                                                               \
static int stub_##SUFFIX##_cblas(int m, int n, int offset, TYPE *a, int lda,   \
                                 int *jpvt, TYPE *tau, TYPE *vn1, TYPE *vn2,   \
                                 TYPE *work)                                    \
{                                                                               \
    g_##SUFFIX##_cblas_call.called += 1;                                        \
    g_##SUFFIX##_cblas_call.m = m;                                              \
    g_##SUFFIX##_cblas_call.n = n;                                              \
    g_##SUFFIX##_cblas_call.offset = offset;                                    \
    g_##SUFFIX##_cblas_call.a = a;                                              \
    g_##SUFFIX##_cblas_call.lda = lda;                                          \
    g_##SUFFIX##_cblas_call.jpvt = jpvt;                                        \
    g_##SUFFIX##_cblas_call.tau = tau;                                          \
    g_##SUFFIX##_cblas_call.vn1 = vn1;                                          \
    g_##SUFFIX##_cblas_call.vn2 = vn2;                                          \
    g_##SUFFIX##_cblas_call.work = work;                                        \
    jpvt[0] = (BASE) + 10;                                                      \
    tau[0] = (TYPE)((BASE) + 11);                                               \
    vn1[0] = (TYPE)((BASE) + 12);                                               \
    vn2[0] = (TYPE)((BASE) + 13);                                               \
    work[0] = (TYPE)((BASE) + 14);                                              \
    return (BASE) + 99;                                                         \
}                                                                               \
static int check_##SUFFIX##_fortran_to_cblas(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                        \
    TYPE a[6] = { (TYPE)1, (TYPE)2, (TYPE)3, (TYPE)4, (TYPE)5, (TYPE)6 };      \
    int jpvt[2] = { 1, 2 };                                                     \
    TYPE tau[2] = { (TYPE)0, (TYPE)0 };                                         \
    TYPE vn1[2] = { (TYPE)0, (TYPE)0 };                                         \
    TYPE vn2[2] = { (TYPE)0, (TYPE)0 };                                         \
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
    if (thunk(3, 2, 0, a, 3, jpvt, tau, vn1, vn2, work) != 0 ||                \
        g_##SUFFIX##_fortran_call.called != 1 ||                                \
        g_##SUFFIX##_fortran_call.m != 3 ||                                     \
        g_##SUFFIX##_fortran_call.n != 2 ||                                     \
        g_##SUFFIX##_fortran_call.offset != 0 ||                                \
        g_##SUFFIX##_fortran_call.a != a ||                                     \
        g_##SUFFIX##_fortran_call.lda != 3 ||                                   \
        g_##SUFFIX##_fortran_call.jpvt != jpvt ||                               \
        g_##SUFFIX##_fortran_call.tau != tau ||                                 \
        g_##SUFFIX##_fortran_call.vn1 != vn1 ||                                 \
        g_##SUFFIX##_fortran_call.vn2 != vn2 ||                                 \
        g_##SUFFIX##_fortran_call.work != work ||                               \
        jpvt[0] != (BASE) + 1 ||                                                \
        tau[0] != (TYPE)((BASE) + 2) ||                                         \
        vn1[0] != (TYPE)((BASE) + 3) ||                                         \
        vn2[0] != (TYPE)((BASE) + 4) ||                                         \
        work[0] != (TYPE)((BASE) + 5)) {                                        \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward LAQP2 inputs or outputs correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards LAQP2 inputs and returns success\n"); \
    return 0;                                                                   \
}                                                                               \
static int check_##SUFFIX##_cblas_to_fortran(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                      \
    int m = 3;                                                                  \
    int n = 2;                                                                  \
    int offset = 0;                                                             \
    int lda = 3;                                                                \
    TYPE a[6] = { (TYPE)11, (TYPE)12, (TYPE)13, (TYPE)14, (TYPE)15, (TYPE)16 }; \
    int jpvt[2] = { 1, 2 };                                                     \
    TYPE tau[2] = { (TYPE)0, (TYPE)0 };                                         \
    TYPE vn1[2] = { (TYPE)0, (TYPE)0 };                                         \
    TYPE vn2[2] = { (TYPE)0, (TYPE)0 };                                         \
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
    thunk(&m, &n, &offset, a, &lda, jpvt, tau, vn1, vn2, work);                \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                  \
        g_##SUFFIX##_cblas_call.m != 3 ||                                       \
        g_##SUFFIX##_cblas_call.n != 2 ||                                       \
        g_##SUFFIX##_cblas_call.offset != 0 ||                                  \
        g_##SUFFIX##_cblas_call.a != a ||                                       \
        g_##SUFFIX##_cblas_call.lda != 3 ||                                     \
        g_##SUFFIX##_cblas_call.jpvt != jpvt ||                                 \
        g_##SUFFIX##_cblas_call.tau != tau ||                                   \
        g_##SUFFIX##_cblas_call.vn1 != vn1 ||                                   \
        g_##SUFFIX##_cblas_call.vn2 != vn2 ||                                   \
        g_##SUFFIX##_cblas_call.work != work ||                                 \
        jpvt[0] != (BASE) + 10 ||                                               \
        tau[0] != (TYPE)((BASE) + 11) ||                                        \
        vn1[0] != (TYPE)((BASE) + 12) ||                                        \
        vn2[0] != (TYPE)((BASE) + 13) ||                                        \
        work[0] != (TYPE)((BASE) + 14)) {                                       \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference LAQP2 inputs or propagate outputs correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences LAQP2 inputs and ignores the C int return\n"); \
    return 0;                                                                   \
}

#define DEFINE_LAQP2_COMPLEX_TESTS(SUFFIX, CTYPE, RTYPE, OP_ID, MAKE_FN,       \
                                   EQ_FN, BASE)                                \
typedef int (*fb_##SUFFIX##_cblas_fn)(int m, int n, int offset, CTYPE *a,      \
                                      int lda, int *jpvt, CTYPE *tau,         \
                                      RTYPE *vn1, RTYPE *vn2, CTYPE *work);   \
typedef void (*fb_##SUFFIX##_fortran_fn)(int *m, int *n, int *offset, CTYPE *a, \
                                         int *lda, int *jpvt, CTYPE *tau,     \
                                         RTYPE *vn1, RTYPE *vn2, CTYPE *work); \
static struct {                                                                 \
    int called;                                                                 \
    int m;                                                                      \
    int n;                                                                      \
    int offset;                                                                 \
    CTYPE *a;                                                                   \
    int lda;                                                                    \
    int *jpvt;                                                                   \
    CTYPE *tau;                                                                  \
    RTYPE *vn1;                                                                  \
    RTYPE *vn2;                                                                  \
    CTYPE *work;                                                                 \
} g_##SUFFIX##_fortran_call;                                                    \
static struct {                                                                 \
    int called;                                                                 \
    int m;                                                                      \
    int n;                                                                      \
    int offset;                                                                 \
    CTYPE *a;                                                                   \
    int lda;                                                                    \
    int *jpvt;                                                                   \
    CTYPE *tau;                                                                  \
    RTYPE *vn1;                                                                  \
    RTYPE *vn2;                                                                  \
    CTYPE *work;                                                                 \
} g_##SUFFIX##_cblas_call;                                                      \
static void stub_##SUFFIX##_fortran(int *m, int *n, int *offset, CTYPE *a,     \
                                    int *lda, int *jpvt, CTYPE *tau,          \
                                    RTYPE *vn1, RTYPE *vn2, CTYPE *work)      \
{                                                                               \
    g_##SUFFIX##_fortran_call.called += 1;                                      \
    g_##SUFFIX##_fortran_call.m = *m;                                           \
    g_##SUFFIX##_fortran_call.n = *n;                                           \
    g_##SUFFIX##_fortran_call.offset = *offset;                                 \
    g_##SUFFIX##_fortran_call.a = a;                                            \
    g_##SUFFIX##_fortran_call.lda = *lda;                                       \
    g_##SUFFIX##_fortran_call.jpvt = jpvt;                                      \
    g_##SUFFIX##_fortran_call.tau = tau;                                        \
    g_##SUFFIX##_fortran_call.vn1 = vn1;                                        \
    g_##SUFFIX##_fortran_call.vn2 = vn2;                                        \
    g_##SUFFIX##_fortran_call.work = work;                                      \
    jpvt[0] = (BASE) + 1;                                                       \
    tau[0] = MAKE_FN((BASE) + 2, (BASE) + 12);                                  \
    vn1[0] = (RTYPE)((BASE) + 3);                                               \
    vn2[0] = (RTYPE)((BASE) + 4);                                               \
    work[0] = MAKE_FN((BASE) + 5, (BASE) + 15);                                 \
}                                                                               \
static int stub_##SUFFIX##_cblas(int m, int n, int offset, CTYPE *a, int lda,  \
                                 int *jpvt, CTYPE *tau, RTYPE *vn1, RTYPE *vn2, \
                                 CTYPE *work)                                   \
{                                                                               \
    g_##SUFFIX##_cblas_call.called += 1;                                        \
    g_##SUFFIX##_cblas_call.m = m;                                              \
    g_##SUFFIX##_cblas_call.n = n;                                              \
    g_##SUFFIX##_cblas_call.offset = offset;                                    \
    g_##SUFFIX##_cblas_call.a = a;                                              \
    g_##SUFFIX##_cblas_call.lda = lda;                                          \
    g_##SUFFIX##_cblas_call.jpvt = jpvt;                                        \
    g_##SUFFIX##_cblas_call.tau = tau;                                          \
    g_##SUFFIX##_cblas_call.vn1 = vn1;                                          \
    g_##SUFFIX##_cblas_call.vn2 = vn2;                                          \
    g_##SUFFIX##_cblas_call.work = work;                                        \
    jpvt[0] = (BASE) + 10;                                                      \
    tau[0] = MAKE_FN((BASE) + 11, (BASE) + 21);                                 \
    vn1[0] = (RTYPE)((BASE) + 12);                                              \
    vn2[0] = (RTYPE)((BASE) + 13);                                              \
    work[0] = MAKE_FN((BASE) + 14, (BASE) + 24);                                \
    return (BASE) + 99;                                                         \
}                                                                               \
static int check_##SUFFIX##_fortran_to_cblas(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                        \
    CTYPE a[6] = {                                                              \
        MAKE_FN(1, 11), MAKE_FN(2, 12), MAKE_FN(3, 13),                        \
        MAKE_FN(4, 14), MAKE_FN(5, 15), MAKE_FN(6, 16)                         \
    };                                                                          \
    int jpvt[2] = { 1, 2 };                                                     \
    CTYPE tau[2] = { MAKE_FN(0, 0), MAKE_FN(0, 0) };                           \
    RTYPE vn1[2] = { (RTYPE)0, (RTYPE)0 };                                      \
    RTYPE vn2[2] = { (RTYPE)0, (RTYPE)0 };                                      \
    CTYPE work[2] = { MAKE_FN(0, 0), MAKE_FN(0, 0) };                          \
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
    if (thunk(3, 2, 0, a, 3, jpvt, tau, vn1, vn2, work) != 0 ||                \
        g_##SUFFIX##_fortran_call.called != 1 ||                                \
        g_##SUFFIX##_fortran_call.m != 3 ||                                     \
        g_##SUFFIX##_fortran_call.n != 2 ||                                     \
        g_##SUFFIX##_fortran_call.offset != 0 ||                                \
        g_##SUFFIX##_fortran_call.a != a ||                                     \
        g_##SUFFIX##_fortran_call.lda != 3 ||                                   \
        g_##SUFFIX##_fortran_call.jpvt != jpvt ||                               \
        g_##SUFFIX##_fortran_call.tau != tau ||                                 \
        g_##SUFFIX##_fortran_call.vn1 != vn1 ||                                 \
        g_##SUFFIX##_fortran_call.vn2 != vn2 ||                                 \
        g_##SUFFIX##_fortran_call.work != work ||                               \
        jpvt[0] != (BASE) + 1 ||                                                \
        !EQ_FN(tau[0], MAKE_FN((BASE) + 2, (BASE) + 12)) ||                     \
        vn1[0] != (RTYPE)((BASE) + 3) ||                                        \
        vn2[0] != (RTYPE)((BASE) + 4) ||                                        \
        !EQ_FN(work[0], MAKE_FN((BASE) + 5, (BASE) + 15))) {                    \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward complex LAQP2 inputs or outputs correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards complex LAQP2 inputs and returns success\n"); \
    return 0;                                                                   \
}                                                                               \
static int check_##SUFFIX##_cblas_to_fortran(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                      \
    int m = 3;                                                                  \
    int n = 2;                                                                  \
    int offset = 0;                                                             \
    int lda = 3;                                                                \
    CTYPE a[6] = {                                                              \
        MAKE_FN(21, 31), MAKE_FN(22, 32), MAKE_FN(23, 33),                     \
        MAKE_FN(24, 34), MAKE_FN(25, 35), MAKE_FN(26, 36)                      \
    };                                                                          \
    int jpvt[2] = { 1, 2 };                                                     \
    CTYPE tau[2] = { MAKE_FN(0, 0), MAKE_FN(0, 0) };                           \
    RTYPE vn1[2] = { (RTYPE)0, (RTYPE)0 };                                      \
    RTYPE vn2[2] = { (RTYPE)0, (RTYPE)0 };                                      \
    CTYPE work[2] = { MAKE_FN(0, 0), MAKE_FN(0, 0) };                          \
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
    thunk(&m, &n, &offset, a, &lda, jpvt, tau, vn1, vn2, work);                \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                  \
        g_##SUFFIX##_cblas_call.m != 3 ||                                       \
        g_##SUFFIX##_cblas_call.n != 2 ||                                       \
        g_##SUFFIX##_cblas_call.offset != 0 ||                                  \
        g_##SUFFIX##_cblas_call.a != a ||                                       \
        g_##SUFFIX##_cblas_call.lda != 3 ||                                     \
        g_##SUFFIX##_cblas_call.jpvt != jpvt ||                                 \
        g_##SUFFIX##_cblas_call.tau != tau ||                                   \
        g_##SUFFIX##_cblas_call.vn1 != vn1 ||                                   \
        g_##SUFFIX##_cblas_call.vn2 != vn2 ||                                   \
        g_##SUFFIX##_cblas_call.work != work ||                                 \
        jpvt[0] != (BASE) + 10 ||                                               \
        !EQ_FN(tau[0], MAKE_FN((BASE) + 11, (BASE) + 21)) ||                    \
        vn1[0] != (RTYPE)((BASE) + 12) ||                                       \
        vn2[0] != (RTYPE)((BASE) + 13) ||                                       \
        !EQ_FN(work[0], MAKE_FN((BASE) + 14, (BASE) + 24))) {                   \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference complex LAQP2 inputs or propagate outputs correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences complex LAQP2 inputs and ignores the C int return\n"); \
    return 0;                                                                   \
}

DEFINE_LAQP2_REAL_TESTS(slaqp2, float, FB_OP_SLAQP2, 100)
DEFINE_LAQP2_REAL_TESTS(dlaqp2, double, FB_OP_DLAQP2, 300)
DEFINE_LAQP2_COMPLEX_TESTS(claqp2, fb_complex_float_t, float, FB_OP_CLAQP2,
                           make_cf32, cf32_eq, 500)
DEFINE_LAQP2_COMPLEX_TESTS(zlaqp2, fb_complex_double_t, double, FB_OP_ZLAQP2,
                           make_cf64, cf64_eq, 700)

int main(void)
{
    int status = 0;

    status |= check_slaqp2_fortran_to_cblas();
    status |= check_slaqp2_cblas_to_fortran();
    status |= check_dlaqp2_fortran_to_cblas();
    status |= check_dlaqp2_cblas_to_fortran();
    status |= check_claqp2_fortran_to_cblas();
    status |= check_claqp2_cblas_to_fortran();
    status |= check_zlaqp2_fortran_to_cblas();
    status |= check_zlaqp2_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}