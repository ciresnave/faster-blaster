#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

#define DEFINE_LALN2_TESTS(SUFFIX, TYPE, OP_ID, BASE)                          \
typedef int (*fb_##SUFFIX##_cblas_fn)(int ltrans, int na, int nw, TYPE smin,   \
                                      TYPE ca, TYPE *a, int lda, TYPE d1,      \
                                      TYPE d2, TYPE *b, int ldb, TYPE wr,      \
                                      TYPE wi, TYPE *x, int ldx, TYPE *scale,  \
                                      TYPE *xnorm);                            \
typedef void (*fb_##SUFFIX##_fortran_fn)(int *ltrans, int *na, int *nw,        \
                                         TYPE *smin, TYPE *ca, TYPE *a,        \
                                         int *lda, TYPE *d1, TYPE *d2,         \
                                         TYPE *b, int *ldb, TYPE *wr,          \
                                         TYPE *wi, TYPE *x, int *ldx,          \
                                         TYPE *scale, TYPE *xnorm, int *info); \
static struct {                                                                 \
    int called;                                                                 \
    int ltrans;                                                                 \
    int na;                                                                     \
    int nw;                                                                     \
    TYPE smin;                                                                  \
    TYPE ca;                                                                    \
    TYPE *a;                                                                    \
    int lda;                                                                    \
    TYPE d1;                                                                    \
    TYPE d2;                                                                    \
    TYPE *b;                                                                    \
    int ldb;                                                                    \
    TYPE wr;                                                                    \
    TYPE wi;                                                                    \
    TYPE *x;                                                                    \
    int ldx;                                                                    \
    TYPE *scale;                                                                \
    TYPE *xnorm;                                                                \
} g_##SUFFIX##_fortran_call;                                                    \
static struct {                                                                 \
    int called;                                                                 \
    int ltrans;                                                                 \
    int na;                                                                     \
    int nw;                                                                     \
    TYPE smin;                                                                  \
    TYPE ca;                                                                    \
    TYPE *a;                                                                    \
    int lda;                                                                    \
    TYPE d1;                                                                    \
    TYPE d2;                                                                    \
    TYPE *b;                                                                    \
    int ldb;                                                                    \
    TYPE wr;                                                                    \
    TYPE wi;                                                                    \
    TYPE *x;                                                                    \
    int ldx;                                                                    \
    TYPE *scale;                                                                \
    TYPE *xnorm;                                                                \
} g_##SUFFIX##_cblas_call;                                                      \
static void stub_##SUFFIX##_fortran(int *ltrans, int *na, int *nw, TYPE *smin, \
                                    TYPE *ca, TYPE *a, int *lda, TYPE *d1,    \
                                    TYPE *d2, TYPE *b, int *ldb, TYPE *wr,    \
                                    TYPE *wi, TYPE *x, int *ldx, TYPE *scale, \
                                    TYPE *xnorm, int *info)                   \
{                                                                               \
    g_##SUFFIX##_fortran_call.called += 1;                                      \
    g_##SUFFIX##_fortran_call.ltrans = *ltrans;                                 \
    g_##SUFFIX##_fortran_call.na = *na;                                         \
    g_##SUFFIX##_fortran_call.nw = *nw;                                         \
    g_##SUFFIX##_fortran_call.smin = *smin;                                     \
    g_##SUFFIX##_fortran_call.ca = *ca;                                         \
    g_##SUFFIX##_fortran_call.a = a;                                            \
    g_##SUFFIX##_fortran_call.lda = *lda;                                       \
    g_##SUFFIX##_fortran_call.d1 = *d1;                                         \
    g_##SUFFIX##_fortran_call.d2 = *d2;                                         \
    g_##SUFFIX##_fortran_call.b = b;                                            \
    g_##SUFFIX##_fortran_call.ldb = *ldb;                                       \
    g_##SUFFIX##_fortran_call.wr = *wr;                                         \
    g_##SUFFIX##_fortran_call.wi = *wi;                                         \
    g_##SUFFIX##_fortran_call.x = x;                                            \
    g_##SUFFIX##_fortran_call.ldx = *ldx;                                       \
    g_##SUFFIX##_fortran_call.scale = scale;                                    \
    g_##SUFFIX##_fortran_call.xnorm = xnorm;                                    \
    x[0] = (TYPE)((BASE) + 1);                                                  \
    *scale = (TYPE)((BASE) + 2);                                                \
    *xnorm = (TYPE)((BASE) + 3);                                                \
    *info = (BASE) + 4;                                                         \
}                                                                               \
static int stub_##SUFFIX##_cblas(int ltrans, int na, int nw, TYPE smin,       \
                                 TYPE ca, TYPE *a, int lda, TYPE d1, TYPE d2, \
                                 TYPE *b, int ldb, TYPE wr, TYPE wi, TYPE *x, \
                                 int ldx, TYPE *scale, TYPE *xnorm)           \
{                                                                               \
    g_##SUFFIX##_cblas_call.called += 1;                                        \
    g_##SUFFIX##_cblas_call.ltrans = ltrans;                                    \
    g_##SUFFIX##_cblas_call.na = na;                                            \
    g_##SUFFIX##_cblas_call.nw = nw;                                            \
    g_##SUFFIX##_cblas_call.smin = smin;                                        \
    g_##SUFFIX##_cblas_call.ca = ca;                                            \
    g_##SUFFIX##_cblas_call.a = a;                                              \
    g_##SUFFIX##_cblas_call.lda = lda;                                          \
    g_##SUFFIX##_cblas_call.d1 = d1;                                            \
    g_##SUFFIX##_cblas_call.d2 = d2;                                            \
    g_##SUFFIX##_cblas_call.b = b;                                              \
    g_##SUFFIX##_cblas_call.ldb = ldb;                                          \
    g_##SUFFIX##_cblas_call.wr = wr;                                            \
    g_##SUFFIX##_cblas_call.wi = wi;                                            \
    g_##SUFFIX##_cblas_call.x = x;                                              \
    g_##SUFFIX##_cblas_call.ldx = ldx;                                          \
    g_##SUFFIX##_cblas_call.scale = scale;                                      \
    g_##SUFFIX##_cblas_call.xnorm = xnorm;                                      \
    x[0] = (TYPE)((BASE) + 5);                                                  \
    *scale = (TYPE)((BASE) + 6);                                                \
    *xnorm = (TYPE)((BASE) + 7);                                                \
    return (BASE) + 8;                                                          \
}                                                                               \
static int check_##SUFFIX##_fortran_to_cblas(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                        \
    TYPE a[4] = { (TYPE)1, (TYPE)2, (TYPE)3, (TYPE)4 };                        \
    TYPE b[4] = { (TYPE)11, (TYPE)12, (TYPE)13, (TYPE)14 };                    \
    TYPE x[4] = { (TYPE)0, (TYPE)0, (TYPE)0, (TYPE)0 };                        \
    TYPE scale = (TYPE)0;                                                       \
    TYPE xnorm = (TYPE)0;                                                       \
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
    if (thunk(1, 2, 1, (TYPE)((BASE) + 10), (TYPE)((BASE) + 11), a, 2,         \
              (TYPE)((BASE) + 12), (TYPE)((BASE) + 13), b, 2,                  \
              (TYPE)((BASE) + 14), (TYPE)((BASE) + 15), x, 2,                 \
              &scale, &xnorm) != (BASE) + 4 ||                                 \
        g_##SUFFIX##_fortran_call.called != 1 ||                                \
        g_##SUFFIX##_fortran_call.ltrans != 1 ||                                \
        g_##SUFFIX##_fortran_call.na != 2 ||                                    \
        g_##SUFFIX##_fortran_call.nw != 1 ||                                    \
        g_##SUFFIX##_fortran_call.smin != (TYPE)((BASE) + 10) ||                \
        g_##SUFFIX##_fortran_call.ca != (TYPE)((BASE) + 11) ||                  \
        g_##SUFFIX##_fortran_call.a != a ||                                     \
        g_##SUFFIX##_fortran_call.lda != 2 ||                                   \
        g_##SUFFIX##_fortran_call.d1 != (TYPE)((BASE) + 12) ||                  \
        g_##SUFFIX##_fortran_call.d2 != (TYPE)((BASE) + 13) ||                  \
        g_##SUFFIX##_fortran_call.b != b ||                                     \
        g_##SUFFIX##_fortran_call.ldb != 2 ||                                   \
        g_##SUFFIX##_fortran_call.wr != (TYPE)((BASE) + 14) ||                  \
        g_##SUFFIX##_fortran_call.wi != (TYPE)((BASE) + 15) ||                  \
        g_##SUFFIX##_fortran_call.x != x ||                                     \
        g_##SUFFIX##_fortran_call.ldx != 2 ||                                   \
        g_##SUFFIX##_fortran_call.scale != &scale ||                            \
        g_##SUFFIX##_fortran_call.xnorm != &xnorm ||                            \
        x[0] != (TYPE)((BASE) + 1) ||                                           \
        scale != (TYPE)((BASE) + 2) ||                                          \
        xnorm != (TYPE)((BASE) + 3)) {                                          \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward SLALN2 inputs or return info correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards no-layout SLALN2 inputs and returns info\n"); \
    return 0;                                                                   \
}                                                                               \
static int check_##SUFFIX##_cblas_to_fortran(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                      \
    int ltrans = 1;                                                             \
    int na = 2;                                                                 \
    int nw = 1;                                                                 \
    int lda = 2;                                                                \
    int ldb = 2;                                                                \
    int ldx = 2;                                                                \
    int info = -1;                                                              \
    TYPE smin = (TYPE)((BASE) + 20);                                            \
    TYPE ca = (TYPE)((BASE) + 21);                                              \
    TYPE d1 = (TYPE)((BASE) + 22);                                              \
    TYPE d2 = (TYPE)((BASE) + 23);                                              \
    TYPE wr = (TYPE)((BASE) + 24);                                              \
    TYPE wi = (TYPE)((BASE) + 25);                                              \
    TYPE a[4] = { (TYPE)31, (TYPE)32, (TYPE)33, (TYPE)34 };                    \
    TYPE b[4] = { (TYPE)41, (TYPE)42, (TYPE)43, (TYPE)44 };                    \
    TYPE x[4] = { (TYPE)0, (TYPE)0, (TYPE)0, (TYPE)0 };                        \
    TYPE scale = (TYPE)0;                                                       \
    TYPE xnorm = (TYPE)0;                                                       \
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
    thunk(&ltrans, &na, &nw, &smin, &ca, a, &lda, &d1, &d2, b, &ldb, &wr,      \
          &wi, x, &ldx, &scale, &xnorm, &info);                                 \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                  \
        g_##SUFFIX##_cblas_call.ltrans != 1 ||                                  \
        g_##SUFFIX##_cblas_call.na != 2 ||                                      \
        g_##SUFFIX##_cblas_call.nw != 1 ||                                      \
        g_##SUFFIX##_cblas_call.smin != (TYPE)((BASE) + 20) ||                  \
        g_##SUFFIX##_cblas_call.ca != (TYPE)((BASE) + 21) ||                    \
        g_##SUFFIX##_cblas_call.a != a ||                                       \
        g_##SUFFIX##_cblas_call.lda != 2 ||                                     \
        g_##SUFFIX##_cblas_call.d1 != (TYPE)((BASE) + 22) ||                    \
        g_##SUFFIX##_cblas_call.d2 != (TYPE)((BASE) + 23) ||                    \
        g_##SUFFIX##_cblas_call.b != b ||                                       \
        g_##SUFFIX##_cblas_call.ldb != 2 ||                                     \
        g_##SUFFIX##_cblas_call.wr != (TYPE)((BASE) + 24) ||                    \
        g_##SUFFIX##_cblas_call.wi != (TYPE)((BASE) + 25) ||                    \
        g_##SUFFIX##_cblas_call.x != x ||                                       \
        g_##SUFFIX##_cblas_call.ldx != 2 ||                                     \
        g_##SUFFIX##_cblas_call.scale != &scale ||                              \
        g_##SUFFIX##_cblas_call.xnorm != &xnorm ||                              \
        info != (BASE) + 8 ||                                                   \
        x[0] != (TYPE)((BASE) + 5) ||                                           \
        scale != (TYPE)((BASE) + 6) ||                                          \
        xnorm != (TYPE)((BASE) + 7)) {                                          \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference SLALN2 inputs or store info correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences no-layout SLALN2 inputs and stores info\n"); \
    return 0;                                                                   \
}

DEFINE_LALN2_TESTS(slaln2, float, FB_OP_SLALN2, 100)
DEFINE_LALN2_TESTS(dlaln2, double, FB_OP_DLALN2, 300)

int main(void)
{
    int status = 0;

    status |= check_slaln2_fortran_to_cblas();
    status |= check_slaln2_cblas_to_fortran();
    status |= check_dlaln2_fortran_to_cblas();
    status |= check_dlaln2_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}