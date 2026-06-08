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

#define DEFINE_LAQPS_REAL_TESTS(SUFFIX, TYPE, OP_ID, BASE)                     \
typedef int (*fb_##SUFFIX##_cblas_fn)(int m, int n, int offset, int nb,        \
                                      int *kb, TYPE *a, int lda, int *jpvt,   \
                                      TYPE *tau, TYPE *vn1, TYPE *vn2,        \
                                      TYPE *auxv, TYPE *f, int ldf);          \
typedef void (*fb_##SUFFIX##_fortran_fn)(int *m, int *n, int *offset, int *nb, \
                                         int *kb, TYPE *a, int *lda,          \
                                         int *jpvt, TYPE *tau, TYPE *vn1,     \
                                         TYPE *vn2, TYPE *auxv, TYPE *f,      \
                                         int *ldf);                            \
static struct {                                                                 \
    int called;                                                                 \
    int m;                                                                      \
    int n;                                                                      \
    int offset;                                                                 \
    int nb;                                                                     \
    int *kb;                                                                    \
    TYPE *a;                                                                    \
    int lda;                                                                    \
    int *jpvt;                                                                   \
    TYPE *tau;                                                                   \
    TYPE *vn1;                                                                   \
    TYPE *vn2;                                                                   \
    TYPE *auxv;                                                                  \
    TYPE *f;                                                                     \
    int ldf;                                                                    \
} g_##SUFFIX##_fortran_call;                                                    \
static struct {                                                                 \
    int called;                                                                 \
    int m;                                                                      \
    int n;                                                                      \
    int offset;                                                                 \
    int nb;                                                                     \
    int *kb;                                                                    \
    TYPE *a;                                                                    \
    int lda;                                                                    \
    int *jpvt;                                                                   \
    TYPE *tau;                                                                   \
    TYPE *vn1;                                                                   \
    TYPE *vn2;                                                                   \
    TYPE *auxv;                                                                  \
    TYPE *f;                                                                     \
    int ldf;                                                                    \
} g_##SUFFIX##_cblas_call;                                                      \
static void stub_##SUFFIX##_fortran(int *m, int *n, int *offset, int *nb,      \
                                    int *kb, TYPE *a, int *lda, int *jpvt,    \
                                    TYPE *tau, TYPE *vn1, TYPE *vn2,          \
                                    TYPE *auxv, TYPE *f, int *ldf)            \
{                                                                               \
    g_##SUFFIX##_fortran_call.called += 1;                                      \
    g_##SUFFIX##_fortran_call.m = *m;                                           \
    g_##SUFFIX##_fortran_call.n = *n;                                           \
    g_##SUFFIX##_fortran_call.offset = *offset;                                 \
    g_##SUFFIX##_fortran_call.nb = *nb;                                         \
    g_##SUFFIX##_fortran_call.kb = kb;                                          \
    g_##SUFFIX##_fortran_call.a = a;                                            \
    g_##SUFFIX##_fortran_call.lda = *lda;                                       \
    g_##SUFFIX##_fortran_call.jpvt = jpvt;                                      \
    g_##SUFFIX##_fortran_call.tau = tau;                                        \
    g_##SUFFIX##_fortran_call.vn1 = vn1;                                        \
    g_##SUFFIX##_fortran_call.vn2 = vn2;                                        \
    g_##SUFFIX##_fortran_call.auxv = auxv;                                      \
    g_##SUFFIX##_fortran_call.f = f;                                            \
    g_##SUFFIX##_fortran_call.ldf = *ldf;                                       \
    *kb = (BASE) + 1;                                                           \
    jpvt[0] = (BASE) + 2;                                                       \
    tau[0] = (TYPE)((BASE) + 3);                                                \
    vn1[0] = (TYPE)((BASE) + 4);                                                \
    vn2[0] = (TYPE)((BASE) + 5);                                                \
    auxv[0] = (TYPE)((BASE) + 6);                                               \
    f[0] = (TYPE)((BASE) + 7);                                                  \
}                                                                               \
static int stub_##SUFFIX##_cblas(int m, int n, int offset, int nb, int *kb,    \
                                 TYPE *a, int lda, int *jpvt, TYPE *tau,      \
                                 TYPE *vn1, TYPE *vn2, TYPE *auxv, TYPE *f,   \
                                 int ldf)                                      \
{                                                                               \
    g_##SUFFIX##_cblas_call.called += 1;                                        \
    g_##SUFFIX##_cblas_call.m = m;                                              \
    g_##SUFFIX##_cblas_call.n = n;                                              \
    g_##SUFFIX##_cblas_call.offset = offset;                                    \
    g_##SUFFIX##_cblas_call.nb = nb;                                            \
    g_##SUFFIX##_cblas_call.kb = kb;                                            \
    g_##SUFFIX##_cblas_call.a = a;                                              \
    g_##SUFFIX##_cblas_call.lda = lda;                                          \
    g_##SUFFIX##_cblas_call.jpvt = jpvt;                                        \
    g_##SUFFIX##_cblas_call.tau = tau;                                          \
    g_##SUFFIX##_cblas_call.vn1 = vn1;                                          \
    g_##SUFFIX##_cblas_call.vn2 = vn2;                                          \
    g_##SUFFIX##_cblas_call.auxv = auxv;                                        \
    g_##SUFFIX##_cblas_call.f = f;                                              \
    g_##SUFFIX##_cblas_call.ldf = ldf;                                          \
    *kb = (BASE) + 10;                                                          \
    jpvt[0] = (BASE) + 11;                                                      \
    tau[0] = (TYPE)((BASE) + 12);                                               \
    vn1[0] = (TYPE)((BASE) + 13);                                               \
    vn2[0] = (TYPE)((BASE) + 14);                                               \
    auxv[0] = (TYPE)((BASE) + 15);                                              \
    f[0] = (TYPE)((BASE) + 16);                                                 \
    return (BASE) + 99;                                                         \
}                                                                               \
static int check_##SUFFIX##_fortran_to_cblas(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                        \
    int kb = -1;                                                                \
    TYPE a[9] = {                                                               \
        (TYPE)1, (TYPE)2, (TYPE)3, (TYPE)4, (TYPE)5,                           \
        (TYPE)6, (TYPE)7, (TYPE)8, (TYPE)9                                     \
    };                                                                          \
    int jpvt[3] = { 1, 2, 3 };                                                  \
    TYPE tau[3] = { (TYPE)0, (TYPE)0, (TYPE)0 };                               \
    TYPE vn1[3] = { (TYPE)0, (TYPE)0, (TYPE)0 };                               \
    TYPE vn2[3] = { (TYPE)0, (TYPE)0, (TYPE)0 };                               \
    TYPE auxv[3] = { (TYPE)0, (TYPE)0, (TYPE)0 };                              \
    TYPE f[3] = { (TYPE)0, (TYPE)0, (TYPE)0 };                                 \
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
    if (thunk(3, 3, 0, 1, &kb, a, 3, jpvt, tau, vn1, vn2, auxv, f, 3) != 0 ||  \
        g_##SUFFIX##_fortran_call.called != 1 ||                                \
        g_##SUFFIX##_fortran_call.m != 3 ||                                     \
        g_##SUFFIX##_fortran_call.n != 3 ||                                     \
        g_##SUFFIX##_fortran_call.offset != 0 ||                                \
        g_##SUFFIX##_fortran_call.nb != 1 ||                                    \
        g_##SUFFIX##_fortran_call.kb != &kb ||                                  \
        g_##SUFFIX##_fortran_call.a != a ||                                     \
        g_##SUFFIX##_fortran_call.lda != 3 ||                                   \
        g_##SUFFIX##_fortran_call.jpvt != jpvt ||                               \
        g_##SUFFIX##_fortran_call.tau != tau ||                                 \
        g_##SUFFIX##_fortran_call.vn1 != vn1 ||                                 \
        g_##SUFFIX##_fortran_call.vn2 != vn2 ||                                 \
        g_##SUFFIX##_fortran_call.auxv != auxv ||                               \
        g_##SUFFIX##_fortran_call.f != f ||                                     \
        g_##SUFFIX##_fortran_call.ldf != 3 ||                                   \
        kb != (BASE) + 1 ||                                                     \
        jpvt[0] != (BASE) + 2 ||                                                \
        tau[0] != (TYPE)((BASE) + 3) ||                                         \
        vn1[0] != (TYPE)((BASE) + 4) ||                                         \
        vn2[0] != (TYPE)((BASE) + 5) ||                                         \
        auxv[0] != (TYPE)((BASE) + 6) ||                                        \
        f[0] != (TYPE)((BASE) + 7)) {                                           \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward LAQPS inputs or outputs correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards LAQPS inputs and returns success\n"); \
    return 0;                                                                   \
}                                                                               \
static int check_##SUFFIX##_cblas_to_fortran(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                      \
    int m = 3;                                                                  \
    int n = 3;                                                                  \
    int offset = 0;                                                             \
    int nb = 1;                                                                 \
    int kb = -1;                                                                \
    int lda = 3;                                                                \
    int ldf = 3;                                                                \
    TYPE a[9] = {                                                               \
        (TYPE)11, (TYPE)12, (TYPE)13, (TYPE)14, (TYPE)15,                      \
        (TYPE)16, (TYPE)17, (TYPE)18, (TYPE)19                                 \
    };                                                                          \
    int jpvt[3] = { 1, 2, 3 };                                                  \
    TYPE tau[3] = { (TYPE)0, (TYPE)0, (TYPE)0 };                               \
    TYPE vn1[3] = { (TYPE)0, (TYPE)0, (TYPE)0 };                               \
    TYPE vn2[3] = { (TYPE)0, (TYPE)0, (TYPE)0 };                               \
    TYPE auxv[3] = { (TYPE)0, (TYPE)0, (TYPE)0 };                              \
    TYPE f[3] = { (TYPE)0, (TYPE)0, (TYPE)0 };                                 \
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
    thunk(&m, &n, &offset, &nb, &kb, a, &lda, jpvt, tau, vn1, vn2, auxv, f, &ldf); \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                  \
        g_##SUFFIX##_cblas_call.m != 3 ||                                       \
        g_##SUFFIX##_cblas_call.n != 3 ||                                       \
        g_##SUFFIX##_cblas_call.offset != 0 ||                                  \
        g_##SUFFIX##_cblas_call.nb != 1 ||                                      \
        g_##SUFFIX##_cblas_call.kb != &kb ||                                    \
        g_##SUFFIX##_cblas_call.a != a ||                                       \
        g_##SUFFIX##_cblas_call.lda != 3 ||                                     \
        g_##SUFFIX##_cblas_call.jpvt != jpvt ||                                 \
        g_##SUFFIX##_cblas_call.tau != tau ||                                   \
        g_##SUFFIX##_cblas_call.vn1 != vn1 ||                                   \
        g_##SUFFIX##_cblas_call.vn2 != vn2 ||                                   \
        g_##SUFFIX##_cblas_call.auxv != auxv ||                                 \
        g_##SUFFIX##_cblas_call.f != f ||                                       \
        g_##SUFFIX##_cblas_call.ldf != 3 ||                                     \
        kb != (BASE) + 10 ||                                                    \
        jpvt[0] != (BASE) + 11 ||                                               \
        tau[0] != (TYPE)((BASE) + 12) ||                                        \
        vn1[0] != (TYPE)((BASE) + 13) ||                                        \
        vn2[0] != (TYPE)((BASE) + 14) ||                                        \
        auxv[0] != (TYPE)((BASE) + 15) ||                                       \
        f[0] != (TYPE)((BASE) + 16)) {                                          \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference LAQPS inputs or propagate outputs correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences LAQPS inputs and ignores the C int return\n"); \
    return 0;                                                                   \
}

#define DEFINE_LAQPS_COMPLEX_TESTS(SUFFIX, CTYPE, RTYPE, OP_ID, MAKE_FN,       \
                                   EQ_FN, BASE)                                \
typedef int (*fb_##SUFFIX##_cblas_fn)(int m, int n, int offset, int nb,        \
                                      int *kb, CTYPE *a, int lda, int *jpvt,  \
                                      CTYPE *tau, RTYPE *vn1, RTYPE *vn2,     \
                                      CTYPE *auxv, CTYPE *f, int ldf);        \
typedef void (*fb_##SUFFIX##_fortran_fn)(int *m, int *n, int *offset, int *nb, \
                                         int *kb, CTYPE *a, int *lda,         \
                                         int *jpvt, CTYPE *tau, RTYPE *vn1,   \
                                         RTYPE *vn2, CTYPE *auxv, CTYPE *f,   \
                                         int *ldf);                            \
static struct {                                                                 \
    int called;                                                                 \
    int m;                                                                      \
    int n;                                                                      \
    int offset;                                                                 \
    int nb;                                                                     \
    int *kb;                                                                    \
    CTYPE *a;                                                                   \
    int lda;                                                                    \
    int *jpvt;                                                                   \
    CTYPE *tau;                                                                  \
    RTYPE *vn1;                                                                  \
    RTYPE *vn2;                                                                  \
    CTYPE *auxv;                                                                 \
    CTYPE *f;                                                                    \
    int ldf;                                                                    \
} g_##SUFFIX##_fortran_call;                                                    \
static struct {                                                                 \
    int called;                                                                 \
    int m;                                                                      \
    int n;                                                                      \
    int offset;                                                                 \
    int nb;                                                                     \
    int *kb;                                                                    \
    CTYPE *a;                                                                   \
    int lda;                                                                    \
    int *jpvt;                                                                   \
    CTYPE *tau;                                                                  \
    RTYPE *vn1;                                                                  \
    RTYPE *vn2;                                                                  \
    CTYPE *auxv;                                                                 \
    CTYPE *f;                                                                    \
    int ldf;                                                                    \
} g_##SUFFIX##_cblas_call;                                                      \
static void stub_##SUFFIX##_fortran(int *m, int *n, int *offset, int *nb,      \
                                    int *kb, CTYPE *a, int *lda, int *jpvt,   \
                                    CTYPE *tau, RTYPE *vn1, RTYPE *vn2,       \
                                    CTYPE *auxv, CTYPE *f, int *ldf)          \
{                                                                               \
    g_##SUFFIX##_fortran_call.called += 1;                                      \
    g_##SUFFIX##_fortran_call.m = *m;                                           \
    g_##SUFFIX##_fortran_call.n = *n;                                           \
    g_##SUFFIX##_fortran_call.offset = *offset;                                 \
    g_##SUFFIX##_fortran_call.nb = *nb;                                         \
    g_##SUFFIX##_fortran_call.kb = kb;                                          \
    g_##SUFFIX##_fortran_call.a = a;                                            \
    g_##SUFFIX##_fortran_call.lda = *lda;                                       \
    g_##SUFFIX##_fortran_call.jpvt = jpvt;                                      \
    g_##SUFFIX##_fortran_call.tau = tau;                                        \
    g_##SUFFIX##_fortran_call.vn1 = vn1;                                        \
    g_##SUFFIX##_fortran_call.vn2 = vn2;                                        \
    g_##SUFFIX##_fortran_call.auxv = auxv;                                      \
    g_##SUFFIX##_fortran_call.f = f;                                            \
    g_##SUFFIX##_fortran_call.ldf = *ldf;                                       \
    *kb = (BASE) + 1;                                                           \
    jpvt[0] = (BASE) + 2;                                                       \
    tau[0] = MAKE_FN((BASE) + 3, (BASE) + 13);                                  \
    vn1[0] = (RTYPE)((BASE) + 4);                                               \
    vn2[0] = (RTYPE)((BASE) + 5);                                               \
    auxv[0] = MAKE_FN((BASE) + 6, (BASE) + 16);                                 \
    f[0] = MAKE_FN((BASE) + 7, (BASE) + 17);                                    \
}                                                                               \
static int stub_##SUFFIX##_cblas(int m, int n, int offset, int nb, int *kb,    \
                                 CTYPE *a, int lda, int *jpvt, CTYPE *tau,    \
                                 RTYPE *vn1, RTYPE *vn2, CTYPE *auxv,         \
                                 CTYPE *f, int ldf)                            \
{                                                                               \
    g_##SUFFIX##_cblas_call.called += 1;                                        \
    g_##SUFFIX##_cblas_call.m = m;                                              \
    g_##SUFFIX##_cblas_call.n = n;                                              \
    g_##SUFFIX##_cblas_call.offset = offset;                                    \
    g_##SUFFIX##_cblas_call.nb = nb;                                            \
    g_##SUFFIX##_cblas_call.kb = kb;                                            \
    g_##SUFFIX##_cblas_call.a = a;                                              \
    g_##SUFFIX##_cblas_call.lda = lda;                                          \
    g_##SUFFIX##_cblas_call.jpvt = jpvt;                                        \
    g_##SUFFIX##_cblas_call.tau = tau;                                          \
    g_##SUFFIX##_cblas_call.vn1 = vn1;                                          \
    g_##SUFFIX##_cblas_call.vn2 = vn2;                                          \
    g_##SUFFIX##_cblas_call.auxv = auxv;                                        \
    g_##SUFFIX##_cblas_call.f = f;                                              \
    g_##SUFFIX##_cblas_call.ldf = ldf;                                          \
    *kb = (BASE) + 10;                                                          \
    jpvt[0] = (BASE) + 11;                                                      \
    tau[0] = MAKE_FN((BASE) + 12, (BASE) + 22);                                 \
    vn1[0] = (RTYPE)((BASE) + 13);                                              \
    vn2[0] = (RTYPE)((BASE) + 14);                                              \
    auxv[0] = MAKE_FN((BASE) + 15, (BASE) + 25);                                \
    f[0] = MAKE_FN((BASE) + 16, (BASE) + 26);                                   \
    return (BASE) + 99;                                                         \
}                                                                               \
static int check_##SUFFIX##_fortran_to_cblas(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                        \
    int kb = -1;                                                                \
    CTYPE a[9] = {                                                              \
        MAKE_FN(1, 11), MAKE_FN(2, 12), MAKE_FN(3, 13),                        \
        MAKE_FN(4, 14), MAKE_FN(5, 15), MAKE_FN(6, 16),                        \
        MAKE_FN(7, 17), MAKE_FN(8, 18), MAKE_FN(9, 19)                         \
    };                                                                          \
    int jpvt[3] = { 1, 2, 3 };                                                  \
    CTYPE tau[3] = { MAKE_FN(0, 0), MAKE_FN(0, 0), MAKE_FN(0, 0) };            \
    RTYPE vn1[3] = { (RTYPE)0, (RTYPE)0, (RTYPE)0 };                           \
    RTYPE vn2[3] = { (RTYPE)0, (RTYPE)0, (RTYPE)0 };                           \
    CTYPE auxv[3] = { MAKE_FN(0, 0), MAKE_FN(0, 0), MAKE_FN(0, 0) };           \
    CTYPE f[3] = { MAKE_FN(0, 0), MAKE_FN(0, 0), MAKE_FN(0, 0) };              \
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
    if (thunk(3, 3, 0, 1, &kb, a, 3, jpvt, tau, vn1, vn2, auxv, f, 3) != 0 ||  \
        g_##SUFFIX##_fortran_call.called != 1 ||                                \
        g_##SUFFIX##_fortran_call.m != 3 ||                                     \
        g_##SUFFIX##_fortran_call.n != 3 ||                                     \
        g_##SUFFIX##_fortran_call.offset != 0 ||                                \
        g_##SUFFIX##_fortran_call.nb != 1 ||                                    \
        g_##SUFFIX##_fortran_call.kb != &kb ||                                  \
        g_##SUFFIX##_fortran_call.a != a ||                                     \
        g_##SUFFIX##_fortran_call.lda != 3 ||                                   \
        g_##SUFFIX##_fortran_call.jpvt != jpvt ||                               \
        g_##SUFFIX##_fortran_call.tau != tau ||                                 \
        g_##SUFFIX##_fortran_call.vn1 != vn1 ||                                 \
        g_##SUFFIX##_fortran_call.vn2 != vn2 ||                                 \
        g_##SUFFIX##_fortran_call.auxv != auxv ||                               \
        g_##SUFFIX##_fortran_call.f != f ||                                     \
        g_##SUFFIX##_fortran_call.ldf != 3 ||                                   \
        kb != (BASE) + 1 ||                                                     \
        jpvt[0] != (BASE) + 2 ||                                                \
        !EQ_FN(tau[0], MAKE_FN((BASE) + 3, (BASE) + 13)) ||                     \
        vn1[0] != (RTYPE)((BASE) + 4) ||                                        \
        vn2[0] != (RTYPE)((BASE) + 5) ||                                        \
        !EQ_FN(auxv[0], MAKE_FN((BASE) + 6, (BASE) + 16)) ||                    \
        !EQ_FN(f[0], MAKE_FN((BASE) + 7, (BASE) + 17))) {                       \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward complex LAQPS inputs or outputs correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards complex LAQPS inputs and returns success\n"); \
    return 0;                                                                   \
}                                                                               \
static int check_##SUFFIX##_cblas_to_fortran(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                      \
    int m = 3;                                                                  \
    int n = 3;                                                                  \
    int offset = 0;                                                             \
    int nb = 1;                                                                 \
    int kb = -1;                                                                \
    int lda = 3;                                                                \
    int ldf = 3;                                                                \
    CTYPE a[9] = {                                                              \
        MAKE_FN(21, 31), MAKE_FN(22, 32), MAKE_FN(23, 33),                     \
        MAKE_FN(24, 34), MAKE_FN(25, 35), MAKE_FN(26, 36),                     \
        MAKE_FN(27, 37), MAKE_FN(28, 38), MAKE_FN(29, 39)                      \
    };                                                                          \
    int jpvt[3] = { 1, 2, 3 };                                                  \
    CTYPE tau[3] = { MAKE_FN(0, 0), MAKE_FN(0, 0), MAKE_FN(0, 0) };            \
    RTYPE vn1[3] = { (RTYPE)0, (RTYPE)0, (RTYPE)0 };                           \
    RTYPE vn2[3] = { (RTYPE)0, (RTYPE)0, (RTYPE)0 };                           \
    CTYPE auxv[3] = { MAKE_FN(0, 0), MAKE_FN(0, 0), MAKE_FN(0, 0) };           \
    CTYPE f[3] = { MAKE_FN(0, 0), MAKE_FN(0, 0), MAKE_FN(0, 0) };              \
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
    thunk(&m, &n, &offset, &nb, &kb, a, &lda, jpvt, tau, vn1, vn2, auxv, f, &ldf); \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                  \
        g_##SUFFIX##_cblas_call.m != 3 ||                                       \
        g_##SUFFIX##_cblas_call.n != 3 ||                                       \
        g_##SUFFIX##_cblas_call.offset != 0 ||                                  \
        g_##SUFFIX##_cblas_call.nb != 1 ||                                      \
        g_##SUFFIX##_cblas_call.kb != &kb ||                                    \
        g_##SUFFIX##_cblas_call.a != a ||                                       \
        g_##SUFFIX##_cblas_call.lda != 3 ||                                     \
        g_##SUFFIX##_cblas_call.jpvt != jpvt ||                                 \
        g_##SUFFIX##_cblas_call.tau != tau ||                                   \
        g_##SUFFIX##_cblas_call.vn1 != vn1 ||                                   \
        g_##SUFFIX##_cblas_call.vn2 != vn2 ||                                   \
        g_##SUFFIX##_cblas_call.auxv != auxv ||                                 \
        g_##SUFFIX##_cblas_call.f != f ||                                       \
        g_##SUFFIX##_cblas_call.ldf != 3 ||                                     \
        kb != (BASE) + 10 ||                                                    \
        jpvt[0] != (BASE) + 11 ||                                               \
        !EQ_FN(tau[0], MAKE_FN((BASE) + 12, (BASE) + 22)) ||                    \
        vn1[0] != (RTYPE)((BASE) + 13) ||                                       \
        vn2[0] != (RTYPE)((BASE) + 14) ||                                       \
        !EQ_FN(auxv[0], MAKE_FN((BASE) + 15, (BASE) + 25)) ||                   \
        !EQ_FN(f[0], MAKE_FN((BASE) + 16, (BASE) + 26))) {                      \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference complex LAQPS inputs or propagate outputs correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences complex LAQPS inputs and ignores the C int return\n"); \
    return 0;                                                                   \
}

DEFINE_LAQPS_REAL_TESTS(slaqps, float, FB_OP_SLAQPS, 100)
DEFINE_LAQPS_REAL_TESTS(dlaqps, double, FB_OP_DLAQPS, 300)
DEFINE_LAQPS_COMPLEX_TESTS(claqps, fb_complex_float_t, float, FB_OP_CLAQPS,
                           make_cf32, cf32_eq, 500)
DEFINE_LAQPS_COMPLEX_TESTS(zlaqps, fb_complex_double_t, double, FB_OP_ZLAQPS,
                           make_cf64, cf64_eq, 700)

int main(void)
{
    int status = 0;

    status |= check_slaqps_fortran_to_cblas();
    status |= check_slaqps_cblas_to_fortran();
    status |= check_dlaqps_fortran_to_cblas();
    status |= check_dlaqps_cblas_to_fortran();
    status |= check_claqps_fortran_to_cblas();
    status |= check_claqps_cblas_to_fortran();
    status |= check_zlaqps_fortran_to_cblas();
    status |= check_zlaqps_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}