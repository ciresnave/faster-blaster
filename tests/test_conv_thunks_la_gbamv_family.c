#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

#define DEFINE_LA_GBAMV_REAL_TESTS(SUFFIX, TYPE, OP_ID, BASE)                     \
typedef int (*fb_##SUFFIX##_cblas_fn)(int trans, int m, int n, int kl, int ku,  \
                                      TYPE alpha, TYPE *ab, int ldab, TYPE *x,  \
                                      int incx, TYPE beta, TYPE *y, int incy);  \
typedef void (*fb_##SUFFIX##_fortran_fn)(int *trans, int *m, int *n, int *kl,   \
                                         int *ku, TYPE *alpha, TYPE *ab,        \
                                         int *ldab, TYPE *x, int *incx,         \
                                         TYPE *beta, TYPE *y, int *incy);       \
static struct {                                                                    \
    int called;                                                                    \
    int trans;                                                                     \
    int m;                                                                         \
    int n;                                                                         \
    int kl;                                                                        \
    int ku;                                                                        \
    TYPE alpha;                                                                    \
    TYPE *ab;                                                                      \
    int ldab;                                                                      \
    TYPE *x;                                                                       \
    int incx;                                                                      \
    TYPE beta;                                                                     \
    TYPE *y;                                                                       \
    int incy;                                                                      \
} g_##SUFFIX##_fortran_call;                                                       \
static struct {                                                                    \
    int called;                                                                    \
    int trans;                                                                     \
    int m;                                                                         \
    int n;                                                                         \
    int kl;                                                                        \
    int ku;                                                                        \
    TYPE alpha;                                                                    \
    TYPE *ab;                                                                      \
    int ldab;                                                                      \
    TYPE *x;                                                                       \
    int incx;                                                                      \
    TYPE beta;                                                                     \
    TYPE *y;                                                                       \
    int incy;                                                                      \
} g_##SUFFIX##_cblas_call;                                                         \
static void stub_##SUFFIX##_fortran(int *trans, int *m, int *n, int *kl, int *ku,\
                                    TYPE *alpha, TYPE *ab, int *ldab, TYPE *x,  \
                                    int *incx, TYPE *beta, TYPE *y, int *incy)  \
{                                                                                 \
    g_##SUFFIX##_fortran_call.called += 1;                                        \
    g_##SUFFIX##_fortran_call.trans = *trans;                                     \
    g_##SUFFIX##_fortran_call.m = *m;                                             \
    g_##SUFFIX##_fortran_call.n = *n;                                             \
    g_##SUFFIX##_fortran_call.kl = *kl;                                           \
    g_##SUFFIX##_fortran_call.ku = *ku;                                           \
    g_##SUFFIX##_fortran_call.alpha = *alpha;                                     \
    g_##SUFFIX##_fortran_call.ab = ab;                                            \
    g_##SUFFIX##_fortran_call.ldab = *ldab;                                       \
    g_##SUFFIX##_fortran_call.x = x;                                              \
    g_##SUFFIX##_fortran_call.incx = *incx;                                       \
    g_##SUFFIX##_fortran_call.beta = *beta;                                       \
    g_##SUFFIX##_fortran_call.y = y;                                              \
    g_##SUFFIX##_fortran_call.incy = *incy;                                       \
    y[0] = (TYPE)((BASE) + 1);                                                    \
    y[1] = (TYPE)((BASE) + 2);                                                    \
}                                                                                 \
static int stub_##SUFFIX##_cblas(int trans, int m, int n, int kl, int ku,       \
                                 TYPE alpha, TYPE *ab, int ldab, TYPE *x,       \
                                 int incx, TYPE beta, TYPE *y, int incy)        \
{                                                                                 \
    g_##SUFFIX##_cblas_call.called += 1;                                          \
    g_##SUFFIX##_cblas_call.trans = trans;                                        \
    g_##SUFFIX##_cblas_call.m = m;                                                \
    g_##SUFFIX##_cblas_call.n = n;                                                \
    g_##SUFFIX##_cblas_call.kl = kl;                                              \
    g_##SUFFIX##_cblas_call.ku = ku;                                              \
    g_##SUFFIX##_cblas_call.alpha = alpha;                                        \
    g_##SUFFIX##_cblas_call.ab = ab;                                              \
    g_##SUFFIX##_cblas_call.ldab = ldab;                                          \
    g_##SUFFIX##_cblas_call.x = x;                                                \
    g_##SUFFIX##_cblas_call.incx = incx;                                          \
    g_##SUFFIX##_cblas_call.beta = beta;                                          \
    g_##SUFFIX##_cblas_call.y = y;                                                \
    g_##SUFFIX##_cblas_call.incy = incy;                                          \
    y[0] = (TYPE)((BASE) + 3);                                                    \
    y[1] = (TYPE)((BASE) + 4);                                                    \
    return (BASE) + 99;                                                           \
}                                                                                 \
static int check_##SUFFIX##_fortran_to_cblas(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                          \
    TYPE ab[4] = { (TYPE)1, (TYPE)2, (TYPE)3, (TYPE)4 };                         \
    TYPE x[2] = { (TYPE)5, (TYPE)6 };                                             \
    TYPE y[2] = { (TYPE)0, (TYPE)0 };                                             \
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
    if (thunk(111, 2, 2, 1, 0, (TYPE)7, ab, 2, x, 1, (TYPE)8, y, -1) != 0 ||    \
        g_##SUFFIX##_fortran_call.called != 1 ||                                  \
        g_##SUFFIX##_fortran_call.trans != 111 ||                                 \
        g_##SUFFIX##_fortran_call.m != 2 ||                                       \
        g_##SUFFIX##_fortran_call.n != 2 ||                                       \
        g_##SUFFIX##_fortran_call.kl != 1 ||                                      \
        g_##SUFFIX##_fortran_call.ku != 0 ||                                      \
        g_##SUFFIX##_fortran_call.alpha != (TYPE)7 ||                             \
        g_##SUFFIX##_fortran_call.ab != ab ||                                     \
        g_##SUFFIX##_fortran_call.ldab != 2 ||                                    \
        g_##SUFFIX##_fortran_call.x != x ||                                       \
        g_##SUFFIX##_fortran_call.incx != 1 ||                                    \
        g_##SUFFIX##_fortran_call.beta != (TYPE)8 ||                              \
        g_##SUFFIX##_fortran_call.y != y ||                                       \
        g_##SUFFIX##_fortran_call.incy != -1 ||                                   \
        y[0] != (TYPE)((BASE) + 1) || y[1] != (TYPE)((BASE) + 2)) {              \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward LA_GBAMV inputs correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards LA_GBAMV inputs\n"); \
    return 0;                                                                     \
}                                                                                 \
static int check_##SUFFIX##_cblas_to_fortran(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                        \
    int trans = 112;                                                              \
    int m = 2;                                                                    \
    int n = 2;                                                                    \
    int kl = 1;                                                                   \
    int ku = 1;                                                                   \
    TYPE alpha = (TYPE)9;                                                         \
    TYPE ab[4] = { (TYPE)1, (TYPE)2, (TYPE)3, (TYPE)4 };                         \
    int ldab = 2;                                                                 \
    TYPE x[2] = { (TYPE)5, (TYPE)6 };                                             \
    int incx = -1;                                                                \
    TYPE beta = (TYPE)10;                                                         \
    TYPE y[2] = { (TYPE)0, (TYPE)0 };                                             \
    int incy = 1;                                                                 \
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
    thunk(&trans, &m, &n, &kl, &ku, &alpha, ab, &ldab, x, &incx, &beta, y, &incy);\
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                    \
        g_##SUFFIX##_cblas_call.trans != 112 ||                                   \
        g_##SUFFIX##_cblas_call.m != 2 ||                                         \
        g_##SUFFIX##_cblas_call.n != 2 ||                                         \
        g_##SUFFIX##_cblas_call.kl != 1 ||                                        \
        g_##SUFFIX##_cblas_call.ku != 1 ||                                        \
        g_##SUFFIX##_cblas_call.alpha != (TYPE)9 ||                               \
        g_##SUFFIX##_cblas_call.ab != ab ||                                       \
        g_##SUFFIX##_cblas_call.ldab != 2 ||                                      \
        g_##SUFFIX##_cblas_call.x != x ||                                         \
        g_##SUFFIX##_cblas_call.incx != -1 ||                                     \
        g_##SUFFIX##_cblas_call.beta != (TYPE)10 ||                               \
        g_##SUFFIX##_cblas_call.y != y ||                                         \
        g_##SUFFIX##_cblas_call.incy != 1 ||                                      \
        y[0] != (TYPE)((BASE) + 3) || y[1] != (TYPE)((BASE) + 4)) {              \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference LA_GBAMV inputs correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences LA_GBAMV inputs\n"); \
    return 0;                                                                     \
}

#define DEFINE_LA_GBAMV_COMPLEX_TESTS(SUFFIX, CTYPE, RTYPE, OP_ID, BASE)         \
typedef int (*fb_##SUFFIX##_cblas_fn)(int trans, int m, int n, int kl, int ku,  \
                                      RTYPE alpha, CTYPE *ab, int ldab, CTYPE *x,\
                                      int incx, RTYPE beta, RTYPE *y, int incy);\
typedef void (*fb_##SUFFIX##_fortran_fn)(int *trans, int *m, int *n, int *kl,   \
                                         int *ku, RTYPE *alpha, CTYPE *ab,      \
                                         int *ldab, CTYPE *x, int *incx,        \
                                         RTYPE *beta, RTYPE *y, int *incy);     \
static struct {                                                                    \
    int called;                                                                    \
    int trans;                                                                     \
    int m;                                                                         \
    int n;                                                                         \
    int kl;                                                                        \
    int ku;                                                                        \
    RTYPE alpha;                                                                   \
    CTYPE *ab;                                                                     \
    int ldab;                                                                      \
    CTYPE *x;                                                                      \
    int incx;                                                                      \
    RTYPE beta;                                                                    \
    RTYPE *y;                                                                      \
    int incy;                                                                      \
} g_##SUFFIX##_fortran_call;                                                       \
static struct {                                                                    \
    int called;                                                                    \
    int trans;                                                                     \
    int m;                                                                         \
    int n;                                                                         \
    int kl;                                                                        \
    int ku;                                                                        \
    RTYPE alpha;                                                                   \
    CTYPE *ab;                                                                     \
    int ldab;                                                                      \
    CTYPE *x;                                                                      \
    int incx;                                                                      \
    RTYPE beta;                                                                    \
    RTYPE *y;                                                                      \
    int incy;                                                                      \
} g_##SUFFIX##_cblas_call;                                                         \
static void stub_##SUFFIX##_fortran(int *trans, int *m, int *n, int *kl, int *ku,\
                                    RTYPE *alpha, CTYPE *ab, int *ldab, CTYPE *x,\
                                    int *incx, RTYPE *beta, RTYPE *y, int *incy)\
{                                                                                 \
    g_##SUFFIX##_fortran_call.called += 1;                                        \
    g_##SUFFIX##_fortran_call.trans = *trans;                                     \
    g_##SUFFIX##_fortran_call.m = *m;                                             \
    g_##SUFFIX##_fortran_call.n = *n;                                             \
    g_##SUFFIX##_fortran_call.kl = *kl;                                           \
    g_##SUFFIX##_fortran_call.ku = *ku;                                           \
    g_##SUFFIX##_fortran_call.alpha = *alpha;                                     \
    g_##SUFFIX##_fortran_call.ab = ab;                                            \
    g_##SUFFIX##_fortran_call.ldab = *ldab;                                       \
    g_##SUFFIX##_fortran_call.x = x;                                              \
    g_##SUFFIX##_fortran_call.incx = *incx;                                       \
    g_##SUFFIX##_fortran_call.beta = *beta;                                       \
    g_##SUFFIX##_fortran_call.y = y;                                              \
    g_##SUFFIX##_fortran_call.incy = *incy;                                       \
    y[0] = (RTYPE)((BASE) + 1);                                                   \
    y[1] = (RTYPE)((BASE) + 2);                                                   \
}                                                                                 \
static int stub_##SUFFIX##_cblas(int trans, int m, int n, int kl, int ku,       \
                                 RTYPE alpha, CTYPE *ab, int ldab, CTYPE *x,    \
                                 int incx, RTYPE beta, RTYPE *y, int incy)      \
{                                                                                 \
    g_##SUFFIX##_cblas_call.called += 1;                                          \
    g_##SUFFIX##_cblas_call.trans = trans;                                        \
    g_##SUFFIX##_cblas_call.m = m;                                                \
    g_##SUFFIX##_cblas_call.n = n;                                                \
    g_##SUFFIX##_cblas_call.kl = kl;                                              \
    g_##SUFFIX##_cblas_call.ku = ku;                                              \
    g_##SUFFIX##_cblas_call.alpha = alpha;                                        \
    g_##SUFFIX##_cblas_call.ab = ab;                                              \
    g_##SUFFIX##_cblas_call.ldab = ldab;                                          \
    g_##SUFFIX##_cblas_call.x = x;                                                \
    g_##SUFFIX##_cblas_call.incx = incx;                                          \
    g_##SUFFIX##_cblas_call.beta = beta;                                          \
    g_##SUFFIX##_cblas_call.y = y;                                                \
    g_##SUFFIX##_cblas_call.incy = incy;                                          \
    y[0] = (RTYPE)((BASE) + 3);                                                   \
    y[1] = (RTYPE)((BASE) + 4);                                                   \
    return (BASE) + 99;                                                           \
}                                                                                 \
static int check_##SUFFIX##_fortran_to_cblas(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                          \
    CTYPE ab[4] = { 0 };                                                          \
    CTYPE x[2] = { 0 };                                                           \
    RTYPE y[2] = { (RTYPE)0, (RTYPE)0 };                                          \
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
    if (thunk(113, 2, 2, 1, 0, (RTYPE)7, ab, 2, x, -1, (RTYPE)8, y, 1) != 0 ||  \
        g_##SUFFIX##_fortran_call.called != 1 ||                                  \
        g_##SUFFIX##_fortran_call.trans != 113 ||                                 \
        g_##SUFFIX##_fortran_call.m != 2 ||                                       \
        g_##SUFFIX##_fortran_call.n != 2 ||                                       \
        g_##SUFFIX##_fortran_call.kl != 1 ||                                      \
        g_##SUFFIX##_fortran_call.ku != 0 ||                                      \
        g_##SUFFIX##_fortran_call.alpha != (RTYPE)7 ||                            \
        g_##SUFFIX##_fortran_call.ab != ab ||                                     \
        g_##SUFFIX##_fortran_call.ldab != 2 ||                                    \
        g_##SUFFIX##_fortran_call.x != x ||                                       \
        g_##SUFFIX##_fortran_call.incx != -1 ||                                   \
        g_##SUFFIX##_fortran_call.beta != (RTYPE)8 ||                             \
        g_##SUFFIX##_fortran_call.y != y ||                                       \
        g_##SUFFIX##_fortran_call.incy != 1 ||                                    \
        y[0] != (RTYPE)((BASE) + 1) || y[1] != (RTYPE)((BASE) + 2)) {            \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward LA_GBAMV complex inputs correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards LA_GBAMV complex inputs\n"); \
    return 0;                                                                     \
}                                                                                 \
static int check_##SUFFIX##_cblas_to_fortran(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                        \
    int trans = 111;                                                              \
    int m = 2;                                                                    \
    int n = 2;                                                                    \
    int kl = 0;                                                                   \
    int ku = 1;                                                                   \
    RTYPE alpha = (RTYPE)9;                                                       \
    CTYPE ab[4] = { 0 };                                                          \
    int ldab = 2;                                                                 \
    CTYPE x[2] = { 0 };                                                           \
    int incx = 1;                                                                 \
    RTYPE beta = (RTYPE)10;                                                       \
    RTYPE y[2] = { (RTYPE)0, (RTYPE)0 };                                          \
    int incy = -1;                                                                \
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
    thunk(&trans, &m, &n, &kl, &ku, &alpha, ab, &ldab, x, &incx, &beta, y, &incy);\
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                    \
        g_##SUFFIX##_cblas_call.trans != 111 ||                                   \
        g_##SUFFIX##_cblas_call.m != 2 ||                                         \
        g_##SUFFIX##_cblas_call.n != 2 ||                                         \
        g_##SUFFIX##_cblas_call.kl != 0 ||                                        \
        g_##SUFFIX##_cblas_call.ku != 1 ||                                        \
        g_##SUFFIX##_cblas_call.alpha != (RTYPE)9 ||                              \
        g_##SUFFIX##_cblas_call.ab != ab ||                                       \
        g_##SUFFIX##_cblas_call.ldab != 2 ||                                      \
        g_##SUFFIX##_cblas_call.x != x ||                                         \
        g_##SUFFIX##_cblas_call.incx != 1 ||                                      \
        g_##SUFFIX##_cblas_call.beta != (RTYPE)10 ||                              \
        g_##SUFFIX##_cblas_call.y != y ||                                         \
        g_##SUFFIX##_cblas_call.incy != -1 ||                                     \
        y[0] != (RTYPE)((BASE) + 3) || y[1] != (RTYPE)((BASE) + 4)) {            \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference LA_GBAMV complex inputs correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences LA_GBAMV complex inputs\n"); \
    return 0;                                                                     \
}

DEFINE_LA_GBAMV_REAL_TESTS(sla_gbamv, float, FB_OP_SLA_GBAMV, 100)
DEFINE_LA_GBAMV_REAL_TESTS(dla_gbamv, double, FB_OP_DLA_GBAMV, 300)
DEFINE_LA_GBAMV_COMPLEX_TESTS(cla_gbamv, fb_complex_float_t, float, FB_OP_CLA_GBAMV, 500)
DEFINE_LA_GBAMV_COMPLEX_TESTS(zla_gbamv, fb_complex_double_t, double, FB_OP_ZLA_GBAMV, 700)

int main(void)
{
    int status = 0;

    status |= check_sla_gbamv_fortran_to_cblas();
    status |= check_sla_gbamv_cblas_to_fortran();
    status |= check_dla_gbamv_fortran_to_cblas();
    status |= check_dla_gbamv_cblas_to_fortran();
    status |= check_cla_gbamv_fortran_to_cblas();
    status |= check_cla_gbamv_cblas_to_fortran();
    status |= check_zla_gbamv_fortran_to_cblas();
    status |= check_zla_gbamv_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}