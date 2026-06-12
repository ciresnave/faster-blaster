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

#define DEFINE_LAEIN_REAL_TESTS(SUFFIX, TYPE, OP_ID, BASE)                     \
typedef int (*fb_##SUFFIX##_cblas_fn)(int rightv, int noinit, int n, TYPE *h,  \
                                      int ldh, TYPE wr, TYPE wi, TYPE *vr,    \
                                      TYPE *vi, TYPE *b, int ldb, TYPE *work, \
                                      TYPE eps3, TYPE smlnum, TYPE bignum);   \
typedef void (*fb_##SUFFIX##_fortran_fn)(int *rightv, int *noinit, int *n,    \
                                         TYPE *h, int *ldh, TYPE *wr,         \
                                         TYPE *wi, TYPE *vr, TYPE *vi,        \
                                         TYPE *b, int *ldb, TYPE *work,       \
                                         TYPE *eps3, TYPE *smlnum,            \
                                         TYPE *bignum, int *info);            \
static struct {                                                                 \
    int called;                                                                 \
    int rightv;                                                                 \
    int noinit;                                                                 \
    int n;                                                                      \
    TYPE *h;                                                                    \
    int ldh;                                                                    \
    TYPE wr;                                                                    \
    TYPE wi;                                                                    \
    TYPE *vr;                                                                   \
    TYPE *vi;                                                                   \
    TYPE *b;                                                                    \
    int ldb;                                                                    \
    TYPE *work;                                                                 \
    TYPE eps3;                                                                  \
    TYPE smlnum;                                                                \
    TYPE bignum;                                                                \
} g_##SUFFIX##_fortran_call;                                                    \
static struct {                                                                 \
    int called;                                                                 \
    int rightv;                                                                 \
    int noinit;                                                                 \
    int n;                                                                      \
    TYPE *h;                                                                    \
    int ldh;                                                                    \
    TYPE wr;                                                                    \
    TYPE wi;                                                                    \
    TYPE *vr;                                                                   \
    TYPE *vi;                                                                   \
    TYPE *b;                                                                    \
    int ldb;                                                                    \
    TYPE *work;                                                                 \
    TYPE eps3;                                                                  \
    TYPE smlnum;                                                                \
    TYPE bignum;                                                                \
} g_##SUFFIX##_cblas_call;                                                      \
static void stub_##SUFFIX##_fortran(int *rightv, int *noinit, int *n, TYPE *h, \
                                    int *ldh, TYPE *wr, TYPE *wi, TYPE *vr,   \
                                    TYPE *vi, TYPE *b, int *ldb, TYPE *work,  \
                                    TYPE *eps3, TYPE *smlnum, TYPE *bignum,   \
                                    int *info)                                 \
{                                                                               \
    g_##SUFFIX##_fortran_call.called += 1;                                      \
    g_##SUFFIX##_fortran_call.rightv = *rightv;                                 \
    g_##SUFFIX##_fortran_call.noinit = *noinit;                                 \
    g_##SUFFIX##_fortran_call.n = *n;                                           \
    g_##SUFFIX##_fortran_call.h = h;                                            \
    g_##SUFFIX##_fortran_call.ldh = *ldh;                                       \
    g_##SUFFIX##_fortran_call.wr = *wr;                                         \
    g_##SUFFIX##_fortran_call.wi = *wi;                                         \
    g_##SUFFIX##_fortran_call.vr = vr;                                          \
    g_##SUFFIX##_fortran_call.vi = vi;                                          \
    g_##SUFFIX##_fortran_call.b = b;                                            \
    g_##SUFFIX##_fortran_call.ldb = *ldb;                                       \
    g_##SUFFIX##_fortran_call.work = work;                                      \
    g_##SUFFIX##_fortran_call.eps3 = *eps3;                                     \
    g_##SUFFIX##_fortran_call.smlnum = *smlnum;                                 \
    g_##SUFFIX##_fortran_call.bignum = *bignum;                                 \
    vr[0] = (TYPE)((BASE) + 1);                                                 \
    vi[0] = (TYPE)((BASE) + 2);                                                 \
    b[0] = (TYPE)((BASE) + 3);                                                  \
    work[0] = (TYPE)((BASE) + 4);                                               \
    *info = (BASE) + 5;                                                         \
}                                                                               \
static int stub_##SUFFIX##_cblas(int rightv, int noinit, int n, TYPE *h,       \
                                 int ldh, TYPE wr, TYPE wi, TYPE *vr,          \
                                 TYPE *vi, TYPE *b, int ldb, TYPE *work,       \
                                 TYPE eps3, TYPE smlnum, TYPE bignum)          \
{                                                                               \
    g_##SUFFIX##_cblas_call.called += 1;                                        \
    g_##SUFFIX##_cblas_call.rightv = rightv;                                    \
    g_##SUFFIX##_cblas_call.noinit = noinit;                                    \
    g_##SUFFIX##_cblas_call.n = n;                                              \
    g_##SUFFIX##_cblas_call.h = h;                                              \
    g_##SUFFIX##_cblas_call.ldh = ldh;                                          \
    g_##SUFFIX##_cblas_call.wr = wr;                                            \
    g_##SUFFIX##_cblas_call.wi = wi;                                            \
    g_##SUFFIX##_cblas_call.vr = vr;                                            \
    g_##SUFFIX##_cblas_call.vi = vi;                                            \
    g_##SUFFIX##_cblas_call.b = b;                                              \
    g_##SUFFIX##_cblas_call.ldb = ldb;                                          \
    g_##SUFFIX##_cblas_call.work = work;                                        \
    g_##SUFFIX##_cblas_call.eps3 = eps3;                                        \
    g_##SUFFIX##_cblas_call.smlnum = smlnum;                                    \
    g_##SUFFIX##_cblas_call.bignum = bignum;                                    \
    vr[0] = (TYPE)((BASE) + 6);                                                 \
    vi[0] = (TYPE)((BASE) + 7);                                                 \
    b[0] = (TYPE)((BASE) + 8);                                                  \
    work[0] = (TYPE)((BASE) + 9);                                               \
    return (BASE) + 10;                                                         \
}                                                                               \
static int check_##SUFFIX##_fortran_to_cblas(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                        \
    TYPE h[4] = { (TYPE)1, (TYPE)2, (TYPE)3, (TYPE)4 };                        \
    TYPE vr[4] = { (TYPE)0, (TYPE)0, (TYPE)0, (TYPE)0 };                       \
    TYPE vi[4] = { (TYPE)0, (TYPE)0, (TYPE)0, (TYPE)0 };                       \
    TYPE b[4] = { (TYPE)11, (TYPE)12, (TYPE)13, (TYPE)14 };                    \
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
    if (thunk(1, 0, 2, h, 2, (TYPE)((BASE) + 30), (TYPE)((BASE) + 31),         \
              vr, vi, b, 2, work, (TYPE)((BASE) + 32), (TYPE)((BASE) + 33),    \
              (TYPE)((BASE) + 34)) != (BASE) + 5 ||                            \
        g_##SUFFIX##_fortran_call.called != 1 ||                                \
        g_##SUFFIX##_fortran_call.rightv != 1 ||                                \
        g_##SUFFIX##_fortran_call.noinit != 0 ||                                \
        g_##SUFFIX##_fortran_call.n != 2 ||                                     \
        g_##SUFFIX##_fortran_call.h != h ||                                     \
        g_##SUFFIX##_fortran_call.ldh != 2 ||                                   \
        g_##SUFFIX##_fortran_call.wr != (TYPE)((BASE) + 30) ||                  \
        g_##SUFFIX##_fortran_call.wi != (TYPE)((BASE) + 31) ||                  \
        g_##SUFFIX##_fortran_call.vr != vr ||                                   \
        g_##SUFFIX##_fortran_call.vi != vi ||                                   \
        g_##SUFFIX##_fortran_call.b != b ||                                     \
        g_##SUFFIX##_fortran_call.ldb != 2 ||                                   \
        g_##SUFFIX##_fortran_call.work != work ||                               \
        g_##SUFFIX##_fortran_call.eps3 != (TYPE)((BASE) + 32) ||                \
        g_##SUFFIX##_fortran_call.smlnum != (TYPE)((BASE) + 33) ||              \
        g_##SUFFIX##_fortran_call.bignum != (TYPE)((BASE) + 34) ||              \
        vr[0] != (TYPE)((BASE) + 1) ||                                          \
        vi[0] != (TYPE)((BASE) + 2) ||                                          \
        b[0] != (TYPE)((BASE) + 3) ||                                           \
        work[0] != (TYPE)((BASE) + 4)) {                                        \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward LAEIN real inputs or return info correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards real LAEIN inputs and returns info\n"); \
    return 0;                                                                   \
}                                                                               \
static int check_##SUFFIX##_cblas_to_fortran(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                      \
    int rightv = 0;                                                             \
    int noinit = 1;                                                             \
    int n = 2;                                                                  \
    int ldh = 2;                                                                \
    int ldb = 2;                                                                \
    int info = -1;                                                              \
    TYPE wr = (TYPE)((BASE) + 40);                                              \
    TYPE wi = (TYPE)((BASE) + 41);                                              \
    TYPE eps3 = (TYPE)((BASE) + 42);                                            \
    TYPE smlnum = (TYPE)((BASE) + 43);                                          \
    TYPE bignum = (TYPE)((BASE) + 44);                                          \
    TYPE h[4] = { (TYPE)31, (TYPE)32, (TYPE)33, (TYPE)34 };                    \
    TYPE vr[4] = { (TYPE)0, (TYPE)0, (TYPE)0, (TYPE)0 };                       \
    TYPE vi[4] = { (TYPE)0, (TYPE)0, (TYPE)0, (TYPE)0 };                       \
    TYPE b[4] = { (TYPE)41, (TYPE)42, (TYPE)43, (TYPE)44 };                    \
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
    thunk(&rightv, &noinit, &n, h, &ldh, &wr, &wi, vr, vi, b, &ldb, work,      \
          &eps3, &smlnum, &bignum, &info);                                      \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                  \
        g_##SUFFIX##_cblas_call.rightv != 0 ||                                  \
        g_##SUFFIX##_cblas_call.noinit != 1 ||                                  \
        g_##SUFFIX##_cblas_call.n != 2 ||                                       \
        g_##SUFFIX##_cblas_call.h != h ||                                       \
        g_##SUFFIX##_cblas_call.ldh != 2 ||                                     \
        g_##SUFFIX##_cblas_call.wr != (TYPE)((BASE) + 40) ||                    \
        g_##SUFFIX##_cblas_call.wi != (TYPE)((BASE) + 41) ||                    \
        g_##SUFFIX##_cblas_call.vr != vr ||                                     \
        g_##SUFFIX##_cblas_call.vi != vi ||                                     \
        g_##SUFFIX##_cblas_call.b != b ||                                       \
        g_##SUFFIX##_cblas_call.ldb != 2 ||                                     \
        g_##SUFFIX##_cblas_call.work != work ||                                 \
        g_##SUFFIX##_cblas_call.eps3 != (TYPE)((BASE) + 42) ||                  \
        g_##SUFFIX##_cblas_call.smlnum != (TYPE)((BASE) + 43) ||                \
        g_##SUFFIX##_cblas_call.bignum != (TYPE)((BASE) + 44) ||                \
        info != (BASE) + 10 ||                                                  \
        vr[0] != (TYPE)((BASE) + 6) ||                                          \
        vi[0] != (TYPE)((BASE) + 7) ||                                          \
        b[0] != (TYPE)((BASE) + 8) ||                                           \
        work[0] != (TYPE)((BASE) + 9)) {                                        \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference LAEIN real inputs or store info correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences real LAEIN inputs and stores info\n"); \
    return 0;                                                                   \
}

#define DEFINE_LAEIN_COMPLEX_TESTS(SUFFIX, CTYPE, RTYPE, OP_ID, MAKE_FN, EQ_FN, BASE) \
typedef int (*fb_##SUFFIX##_cblas_fn)(int rightv, int noinit, int n, CTYPE *h, \
                                      int ldh, CTYPE w, CTYPE *v, CTYPE *b,   \
                                      int ldb, RTYPE *rwork, RTYPE eps3,      \
                                      RTYPE smlnum);                          \
typedef void (*fb_##SUFFIX##_fortran_fn)(int *rightv, int *noinit, int *n,    \
                                         CTYPE *h, int *ldh, CTYPE *w,        \
                                         CTYPE *v, CTYPE *b, int *ldb,        \
                                         RTYPE *rwork, RTYPE *eps3,           \
                                         RTYPE *smlnum, int *info);           \
static struct {                                                                 \
    int called;                                                                 \
    int rightv;                                                                 \
    int noinit;                                                                 \
    int n;                                                                      \
    CTYPE *h;                                                                   \
    int ldh;                                                                    \
    CTYPE w;                                                                    \
    CTYPE *v;                                                                   \
    CTYPE *b;                                                                   \
    int ldb;                                                                    \
    RTYPE *rwork;                                                               \
    RTYPE eps3;                                                                 \
    RTYPE smlnum;                                                               \
} g_##SUFFIX##_fortran_call;                                                    \
static struct {                                                                 \
    int called;                                                                 \
    int rightv;                                                                 \
    int noinit;                                                                 \
    int n;                                                                      \
    CTYPE *h;                                                                   \
    int ldh;                                                                    \
    CTYPE w;                                                                    \
    CTYPE *v;                                                                   \
    CTYPE *b;                                                                   \
    int ldb;                                                                    \
    RTYPE *rwork;                                                               \
    RTYPE eps3;                                                                 \
    RTYPE smlnum;                                                               \
} g_##SUFFIX##_cblas_call;                                                      \
static void stub_##SUFFIX##_fortran(int *rightv, int *noinit, int *n, CTYPE *h,\
                                    int *ldh, CTYPE *w, CTYPE *v, CTYPE *b,   \
                                    int *ldb, RTYPE *rwork, RTYPE *eps3,      \
                                    RTYPE *smlnum, int *info)                 \
{                                                                               \
    g_##SUFFIX##_fortran_call.called += 1;                                      \
    g_##SUFFIX##_fortran_call.rightv = *rightv;                                 \
    g_##SUFFIX##_fortran_call.noinit = *noinit;                                 \
    g_##SUFFIX##_fortran_call.n = *n;                                           \
    g_##SUFFIX##_fortran_call.h = h;                                            \
    g_##SUFFIX##_fortran_call.ldh = *ldh;                                       \
    g_##SUFFIX##_fortran_call.w = *w;                                           \
    g_##SUFFIX##_fortran_call.v = v;                                            \
    g_##SUFFIX##_fortran_call.b = b;                                            \
    g_##SUFFIX##_fortran_call.ldb = *ldb;                                       \
    g_##SUFFIX##_fortran_call.rwork = rwork;                                    \
    g_##SUFFIX##_fortran_call.eps3 = *eps3;                                     \
    g_##SUFFIX##_fortran_call.smlnum = *smlnum;                                 \
    v[0] = MAKE_FN((BASE) + 1, (BASE) + 11);                                    \
    b[0] = MAKE_FN((BASE) + 2, (BASE) + 12);                                    \
    rwork[0] = (RTYPE)((BASE) + 3);                                             \
    *info = (BASE) + 4;                                                         \
}                                                                               \
static int stub_##SUFFIX##_cblas(int rightv, int noinit, int n, CTYPE *h,      \
                                 int ldh, CTYPE w, CTYPE *v, CTYPE *b,        \
                                 int ldb, RTYPE *rwork, RTYPE eps3,           \
                                 RTYPE smlnum)                                \
{                                                                               \
    g_##SUFFIX##_cblas_call.called += 1;                                        \
    g_##SUFFIX##_cblas_call.rightv = rightv;                                    \
    g_##SUFFIX##_cblas_call.noinit = noinit;                                    \
    g_##SUFFIX##_cblas_call.n = n;                                              \
    g_##SUFFIX##_cblas_call.h = h;                                              \
    g_##SUFFIX##_cblas_call.ldh = ldh;                                          \
    g_##SUFFIX##_cblas_call.w = w;                                              \
    g_##SUFFIX##_cblas_call.v = v;                                              \
    g_##SUFFIX##_cblas_call.b = b;                                              \
    g_##SUFFIX##_cblas_call.ldb = ldb;                                          \
    g_##SUFFIX##_cblas_call.rwork = rwork;                                      \
    g_##SUFFIX##_cblas_call.eps3 = eps3;                                        \
    g_##SUFFIX##_cblas_call.smlnum = smlnum;                                    \
    v[0] = MAKE_FN((BASE) + 5, (BASE) + 15);                                    \
    b[0] = MAKE_FN((BASE) + 6, (BASE) + 16);                                    \
    rwork[0] = (RTYPE)((BASE) + 7);                                             \
    return (BASE) + 8;                                                          \
}                                                                               \
static int check_##SUFFIX##_fortran_to_cblas(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                        \
    CTYPE h[4];                                                                 \
    CTYPE v[4];                                                                 \
    CTYPE b[4];                                                                 \
    RTYPE rwork[4] = { (RTYPE)21, (RTYPE)22, (RTYPE)23, (RTYPE)24 };           \
    h[0] = MAKE_FN(1, 21);                                                      \
    h[1] = MAKE_FN(2, 22);                                                      \
    h[2] = MAKE_FN(3, 23);                                                      \
    h[3] = MAKE_FN(4, 24);                                                      \
    v[0] = MAKE_FN(0, 0);                                                       \
    v[1] = MAKE_FN(0, 0);                                                       \
    v[2] = MAKE_FN(0, 0);                                                       \
    v[3] = MAKE_FN(0, 0);                                                       \
    b[0] = MAKE_FN(11, 31);                                                     \
    b[1] = MAKE_FN(12, 32);                                                     \
    b[2] = MAKE_FN(13, 33);                                                     \
    b[3] = MAKE_FN(14, 34);                                                     \
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
    if (thunk(1, 0, 2, h, 2, MAKE_FN((BASE) + 30, (BASE) + 40), v, b, 2,       \
              rwork, (RTYPE)((BASE) + 31), (RTYPE)((BASE) + 32)) !=            \
            (BASE) + 4 ||                                                      \
        g_##SUFFIX##_fortran_call.called != 1 ||                                \
        g_##SUFFIX##_fortran_call.rightv != 1 ||                                \
        g_##SUFFIX##_fortran_call.noinit != 0 ||                                \
        g_##SUFFIX##_fortran_call.n != 2 ||                                     \
        g_##SUFFIX##_fortran_call.h != h ||                                     \
        g_##SUFFIX##_fortran_call.ldh != 2 ||                                   \
        !EQ_FN(g_##SUFFIX##_fortran_call.w,                                     \
               MAKE_FN((BASE) + 30, (BASE) + 40)) ||                          \
        g_##SUFFIX##_fortran_call.v != v ||                                     \
        g_##SUFFIX##_fortran_call.b != b ||                                     \
        g_##SUFFIX##_fortran_call.ldb != 2 ||                                   \
        g_##SUFFIX##_fortran_call.rwork != rwork ||                             \
        g_##SUFFIX##_fortran_call.eps3 != (RTYPE)((BASE) + 31) ||               \
        g_##SUFFIX##_fortran_call.smlnum != (RTYPE)((BASE) + 32) ||             \
        !EQ_FN(v[0], MAKE_FN((BASE) + 1, (BASE) + 11)) ||                      \
        !EQ_FN(b[0], MAKE_FN((BASE) + 2, (BASE) + 12)) ||                      \
        rwork[0] != (RTYPE)((BASE) + 3)) {                                      \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward LAEIN complex inputs or return info correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards complex LAEIN inputs and returns info\n"); \
    return 0;                                                                   \
}                                                                               \
static int check_##SUFFIX##_cblas_to_fortran(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                      \
    int rightv = 0;                                                             \
    int noinit = 1;                                                             \
    int n = 2;                                                                  \
    int ldh = 2;                                                                \
    int ldb = 2;                                                                \
    int info = -1;                                                              \
    CTYPE w = MAKE_FN((BASE) + 40, (BASE) + 50);                                \
    RTYPE eps3 = (RTYPE)((BASE) + 41);                                          \
    RTYPE smlnum = (RTYPE)((BASE) + 42);                                        \
    CTYPE h[4];                                                                 \
    CTYPE v[4];                                                                 \
    CTYPE b[4];                                                                 \
    RTYPE rwork[4] = { (RTYPE)51, (RTYPE)52, (RTYPE)53, (RTYPE)54 };           \
    h[0] = MAKE_FN(31, 61);                                                     \
    h[1] = MAKE_FN(32, 62);                                                     \
    h[2] = MAKE_FN(33, 63);                                                     \
    h[3] = MAKE_FN(34, 64);                                                     \
    v[0] = MAKE_FN(0, 0);                                                       \
    v[1] = MAKE_FN(0, 0);                                                       \
    v[2] = MAKE_FN(0, 0);                                                       \
    v[3] = MAKE_FN(0, 0);                                                       \
    b[0] = MAKE_FN(41, 71);                                                     \
    b[1] = MAKE_FN(42, 72);                                                     \
    b[2] = MAKE_FN(43, 73);                                                     \
    b[3] = MAKE_FN(44, 74);                                                     \
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
    thunk(&rightv, &noinit, &n, h, &ldh, &w, v, b, &ldb, rwork, &eps3,         \
          &smlnum, &info);                                                      \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                  \
        g_##SUFFIX##_cblas_call.rightv != 0 ||                                  \
        g_##SUFFIX##_cblas_call.noinit != 1 ||                                  \
        g_##SUFFIX##_cblas_call.n != 2 ||                                       \
        g_##SUFFIX##_cblas_call.h != h ||                                       \
        g_##SUFFIX##_cblas_call.ldh != 2 ||                                     \
        !EQ_FN(g_##SUFFIX##_cblas_call.w, w) ||                                 \
        g_##SUFFIX##_cblas_call.v != v ||                                       \
        g_##SUFFIX##_cblas_call.b != b ||                                       \
        g_##SUFFIX##_cblas_call.ldb != 2 ||                                     \
        g_##SUFFIX##_cblas_call.rwork != rwork ||                               \
        g_##SUFFIX##_cblas_call.eps3 != (RTYPE)((BASE) + 41) ||                 \
        g_##SUFFIX##_cblas_call.smlnum != (RTYPE)((BASE) + 42) ||               \
        info != (BASE) + 8 ||                                                   \
        !EQ_FN(v[0], MAKE_FN((BASE) + 5, (BASE) + 15)) ||                      \
        !EQ_FN(b[0], MAKE_FN((BASE) + 6, (BASE) + 16)) ||                      \
        rwork[0] != (RTYPE)((BASE) + 7)) {                                      \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference LAEIN complex inputs or store info correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences complex LAEIN inputs and stores info\n"); \
    return 0;                                                                   \
}

DEFINE_LAEIN_REAL_TESTS(slaein, float, FB_OP_SLAEIN, 100)
DEFINE_LAEIN_REAL_TESTS(dlaein, double, FB_OP_DLAEIN, 300)
DEFINE_LAEIN_COMPLEX_TESTS(claein, fb_complex_float_t, float, FB_OP_CLAEIN,
                           make_cf32, cf32_eq, 500)
DEFINE_LAEIN_COMPLEX_TESTS(zlaein, fb_complex_double_t, double, FB_OP_ZLAEIN,
                           make_cf64, cf64_eq, 700)

int main(void)
{
    int status = 0;

    status |= check_slaein_fortran_to_cblas();
    status |= check_slaein_cblas_to_fortran();
    status |= check_dlaein_fortran_to_cblas();
    status |= check_dlaein_cblas_to_fortran();
    status |= check_claein_fortran_to_cblas();
    status |= check_claein_cblas_to_fortran();
    status |= check_zlaein_fortran_to_cblas();
    status |= check_zlaein_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}