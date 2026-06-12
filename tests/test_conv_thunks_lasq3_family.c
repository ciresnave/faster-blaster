#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

#define DEFINE_LASQ3_TESTS(SUFFIX, TYPE, OP_ID, BASE)                              \
typedef int (*fb_##SUFFIX##_cblas_fn)(int i0, int n0, TYPE *z, int pp,           \
                                      TYPE *dmin, TYPE *sigma, TYPE *desig,      \
                                      TYPE *qmax, int *nfail, int *iter,         \
                                      int *ndiv, int ieee, int *ttype,           \
                                      TYPE *dmin1, TYPE *dmin2, TYPE dn,         \
                                      TYPE dn1, TYPE dn2, TYPE *g, TYPE tau);    \
typedef void (*fb_##SUFFIX##_fortran_fn)(int *i0, int *n0, TYPE *z, int *pp,    \
                                         TYPE *dmin, TYPE *sigma, TYPE *desig,   \
                                         TYPE *qmax, int *nfail, int *iter,      \
                                         int *ndiv, int *ieee, int *ttype,       \
                                         TYPE *dmin1, TYPE *dmin2, TYPE *dn,     \
                                         TYPE *dn1, TYPE *dn2, TYPE *g,          \
                                         TYPE *tau);                             \
static struct {                                                                    \
    int called;                                                                    \
    int i0;                                                                        \
    int n0;                                                                        \
    TYPE *z;                                                                       \
    int pp;                                                                        \
    TYPE *dmin;                                                                    \
    TYPE *sigma;                                                                   \
    TYPE *desig;                                                                   \
    TYPE *qmax;                                                                    \
    int *nfail;                                                                    \
    int *iter;                                                                     \
    int *ndiv;                                                                     \
    int ieee;                                                                      \
    int *ttype;                                                                    \
    TYPE *dmin1;                                                                   \
    TYPE *dmin2;                                                                   \
    TYPE dn;                                                                       \
    TYPE dn1;                                                                      \
    TYPE dn2;                                                                      \
    TYPE *g;                                                                       \
    TYPE tau;                                                                      \
} g_##SUFFIX##_fortran_call;                                                       \
static struct {                                                                    \
    int called;                                                                    \
    int i0;                                                                        \
    int n0;                                                                        \
    TYPE *z;                                                                       \
    int pp;                                                                        \
    TYPE *dmin;                                                                    \
    TYPE *sigma;                                                                   \
    TYPE *desig;                                                                   \
    TYPE *qmax;                                                                    \
    int *nfail;                                                                    \
    int *iter;                                                                     \
    int *ndiv;                                                                     \
    int ieee;                                                                      \
    int *ttype;                                                                    \
    TYPE *dmin1;                                                                   \
    TYPE *dmin2;                                                                   \
    TYPE dn;                                                                       \
    TYPE dn1;                                                                      \
    TYPE dn2;                                                                      \
    TYPE *g;                                                                       \
    TYPE tau;                                                                      \
} g_##SUFFIX##_cblas_call;                                                         \
static void stub_##SUFFIX##_fortran(int *i0, int *n0, TYPE *z, int *pp,         \
                                    TYPE *dmin, TYPE *sigma, TYPE *desig,       \
                                    TYPE *qmax, int *nfail, int *iter,          \
                                    int *ndiv, int *ieee, int *ttype,           \
                                    TYPE *dmin1, TYPE *dmin2, TYPE *dn,         \
                                    TYPE *dn1, TYPE *dn2, TYPE *g, TYPE *tau) { \
    g_##SUFFIX##_fortran_call.called += 1;                                        \
    g_##SUFFIX##_fortran_call.i0 = *i0;                                           \
    g_##SUFFIX##_fortran_call.n0 = *n0;                                           \
    g_##SUFFIX##_fortran_call.z = z;                                              \
    g_##SUFFIX##_fortran_call.pp = *pp;                                           \
    g_##SUFFIX##_fortran_call.dmin = dmin;                                        \
    g_##SUFFIX##_fortran_call.sigma = sigma;                                      \
    g_##SUFFIX##_fortran_call.desig = desig;                                      \
    g_##SUFFIX##_fortran_call.qmax = qmax;                                        \
    g_##SUFFIX##_fortran_call.nfail = nfail;                                      \
    g_##SUFFIX##_fortran_call.iter = iter;                                        \
    g_##SUFFIX##_fortran_call.ndiv = ndiv;                                        \
    g_##SUFFIX##_fortran_call.ieee = *ieee;                                       \
    g_##SUFFIX##_fortran_call.ttype = ttype;                                      \
    g_##SUFFIX##_fortran_call.dmin1 = dmin1;                                      \
    g_##SUFFIX##_fortran_call.dmin2 = dmin2;                                      \
    g_##SUFFIX##_fortran_call.dn = *dn;                                           \
    g_##SUFFIX##_fortran_call.dn1 = *dn1;                                         \
    g_##SUFFIX##_fortran_call.dn2 = *dn2;                                         \
    g_##SUFFIX##_fortran_call.g = g;                                              \
    g_##SUFFIX##_fortran_call.tau = *tau;                                         \
    *dmin = (TYPE)((BASE) + 1);                                                   \
    *sigma = (TYPE)((BASE) + 2);                                                  \
    *desig = (TYPE)((BASE) + 3);                                                  \
    *qmax = (TYPE)((BASE) + 4);                                                   \
    *nfail = (BASE) + 5;                                                          \
    *iter = (BASE) + 6;                                                           \
    *ndiv = (BASE) + 7;                                                           \
    *ttype = (BASE) + 8;                                                          \
    *dmin1 = (TYPE)((BASE) + 9);                                                  \
    *dmin2 = (TYPE)((BASE) + 10);                                                 \
    *g = (TYPE)((BASE) + 11);                                                     \
}                                                                                 \
static int stub_##SUFFIX##_cblas(int i0, int n0, TYPE *z, int pp, TYPE *dmin,   \
                                 TYPE *sigma, TYPE *desig, TYPE *qmax,          \
                                 int *nfail, int *iter, int *ndiv, int ieee,    \
                                 int *ttype, TYPE *dmin1, TYPE *dmin2, TYPE dn, \
                                 TYPE dn1, TYPE dn2, TYPE *g, TYPE tau) {       \
    g_##SUFFIX##_cblas_call.called += 1;                                         \
    g_##SUFFIX##_cblas_call.i0 = i0;                                             \
    g_##SUFFIX##_cblas_call.n0 = n0;                                             \
    g_##SUFFIX##_cblas_call.z = z;                                               \
    g_##SUFFIX##_cblas_call.pp = pp;                                             \
    g_##SUFFIX##_cblas_call.dmin = dmin;                                         \
    g_##SUFFIX##_cblas_call.sigma = sigma;                                       \
    g_##SUFFIX##_cblas_call.desig = desig;                                       \
    g_##SUFFIX##_cblas_call.qmax = qmax;                                         \
    g_##SUFFIX##_cblas_call.nfail = nfail;                                       \
    g_##SUFFIX##_cblas_call.iter = iter;                                         \
    g_##SUFFIX##_cblas_call.ndiv = ndiv;                                         \
    g_##SUFFIX##_cblas_call.ieee = ieee;                                         \
    g_##SUFFIX##_cblas_call.ttype = ttype;                                       \
    g_##SUFFIX##_cblas_call.dmin1 = dmin1;                                       \
    g_##SUFFIX##_cblas_call.dmin2 = dmin2;                                       \
    g_##SUFFIX##_cblas_call.dn = dn;                                             \
    g_##SUFFIX##_cblas_call.dn1 = dn1;                                           \
    g_##SUFFIX##_cblas_call.dn2 = dn2;                                           \
    g_##SUFFIX##_cblas_call.g = g;                                               \
    g_##SUFFIX##_cblas_call.tau = tau;                                           \
    *dmin = (TYPE)((BASE) + 12);                                                 \
    *sigma = (TYPE)((BASE) + 13);                                                \
    *desig = (TYPE)((BASE) + 14);                                                \
    *qmax = (TYPE)((BASE) + 15);                                                 \
    *nfail = (BASE) + 16;                                                        \
    *iter = (BASE) + 17;                                                         \
    *ndiv = (BASE) + 18;                                                         \
    *ttype = (BASE) + 19;                                                        \
    *dmin1 = (TYPE)((BASE) + 20);                                                \
    *dmin2 = (TYPE)((BASE) + 21);                                                \
    *g = (TYPE)((BASE) + 22);                                                    \
    return 0;                                                                    \
}                                                                                \
static int check_##SUFFIX##_fortran_to_cblas(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                          \
    TYPE z[2] = { (TYPE)1, (TYPE)2 };                                             \
    TYPE dmin = (TYPE)0;                                                          \
    TYPE sigma = (TYPE)0;                                                         \
    TYPE desig = (TYPE)0;                                                         \
    TYPE qmax = (TYPE)0;                                                          \
    int nfail = 0;                                                                \
    int iter = 0;                                                                 \
    int ndiv = 0;                                                                 \
    int ttype = 0;                                                                \
    TYPE dmin1 = (TYPE)0;                                                         \
    TYPE dmin2 = (TYPE)0;                                                         \
    TYPE g = (TYPE)0;                                                             \
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
    if (thunk(1, 3, z, 0, &dmin, &sigma, &desig, &qmax, &nfail, &iter, &ndiv, 1,\
              &ttype, &dmin1, &dmin2, (TYPE)12, (TYPE)13, (TYPE)14, &g,         \
              (TYPE)15) != 0 ||                                                   \
        g_##SUFFIX##_fortran_call.called != 1 ||                                  \
        g_##SUFFIX##_fortran_call.i0 != 1 ||                                      \
        g_##SUFFIX##_fortran_call.n0 != 3 ||                                      \
        g_##SUFFIX##_fortran_call.z != z ||                                       \
        g_##SUFFIX##_fortran_call.pp != 0 ||                                      \
        g_##SUFFIX##_fortran_call.dmin != &dmin ||                                \
        g_##SUFFIX##_fortran_call.sigma != &sigma ||                              \
        g_##SUFFIX##_fortran_call.desig != &desig ||                              \
        g_##SUFFIX##_fortran_call.qmax != &qmax ||                                \
        g_##SUFFIX##_fortran_call.nfail != &nfail ||                              \
        g_##SUFFIX##_fortran_call.iter != &iter ||                                \
        g_##SUFFIX##_fortran_call.ndiv != &ndiv ||                                \
        g_##SUFFIX##_fortran_call.ieee != 1 ||                                    \
        g_##SUFFIX##_fortran_call.ttype != &ttype ||                              \
        g_##SUFFIX##_fortran_call.dmin1 != &dmin1 ||                              \
        g_##SUFFIX##_fortran_call.dmin2 != &dmin2 ||                              \
        g_##SUFFIX##_fortran_call.dn != (TYPE)12 ||                               \
        g_##SUFFIX##_fortran_call.dn1 != (TYPE)13 ||                              \
        g_##SUFFIX##_fortran_call.dn2 != (TYPE)14 ||                              \
        g_##SUFFIX##_fortran_call.g != &g ||                                      \
        g_##SUFFIX##_fortran_call.tau != (TYPE)15 ||                              \
        dmin != (TYPE)((BASE) + 1) || sigma != (TYPE)((BASE) + 2) ||             \
        desig != (TYPE)((BASE) + 3) || qmax != (TYPE)((BASE) + 4) ||             \
        nfail != (BASE) + 5 || iter != (BASE) + 6 || ndiv != (BASE) + 7 ||       \
        ttype != (BASE) + 8 || dmin1 != (TYPE)((BASE) + 9) ||                    \
        dmin2 != (TYPE)((BASE) + 10) || g != (TYPE)((BASE) + 11)) {              \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward LASQ3 inputs correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards LASQ3 inputs\n"); \
    return 0;                                                                     \
}                                                                                 \
static int check_##SUFFIX##_cblas_to_fortran(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                        \
    int i0 = 1;                                                                   \
    int n0 = 3;                                                                   \
    TYPE z[2] = { (TYPE)21, (TYPE)22 };                                           \
    int pp = 0;                                                                   \
    TYPE dmin = (TYPE)0;                                                          \
    TYPE sigma = (TYPE)0;                                                         \
    TYPE desig = (TYPE)0;                                                         \
    TYPE qmax = (TYPE)0;                                                          \
    int nfail = 0;                                                                \
    int iter = 0;                                                                 \
    int ndiv = 0;                                                                 \
    int ieee = 1;                                                                 \
    int ttype = 0;                                                                \
    TYPE dmin1 = (TYPE)0;                                                         \
    TYPE dmin2 = (TYPE)0;                                                         \
    TYPE dn = (TYPE)12;                                                           \
    TYPE dn1 = (TYPE)13;                                                          \
    TYPE dn2 = (TYPE)14;                                                          \
    TYPE g = (TYPE)0;                                                             \
    TYPE tau = (TYPE)15;                                                          \
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
    thunk(&i0, &n0, z, &pp, &dmin, &sigma, &desig, &qmax, &nfail, &iter, &ndiv, \
          &ieee, &ttype, &dmin1, &dmin2, &dn, &dn1, &dn2, &g, &tau);             \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                    \
        g_##SUFFIX##_cblas_call.i0 != 1 ||                                        \
        g_##SUFFIX##_cblas_call.n0 != 3 ||                                        \
        g_##SUFFIX##_cblas_call.z != z ||                                         \
        g_##SUFFIX##_cblas_call.pp != 0 ||                                        \
        g_##SUFFIX##_cblas_call.dmin != &dmin ||                                  \
        g_##SUFFIX##_cblas_call.sigma != &sigma ||                                \
        g_##SUFFIX##_cblas_call.desig != &desig ||                                \
        g_##SUFFIX##_cblas_call.qmax != &qmax ||                                  \
        g_##SUFFIX##_cblas_call.nfail != &nfail ||                                \
        g_##SUFFIX##_cblas_call.iter != &iter ||                                  \
        g_##SUFFIX##_cblas_call.ndiv != &ndiv ||                                  \
        g_##SUFFIX##_cblas_call.ieee != 1 ||                                      \
        g_##SUFFIX##_cblas_call.ttype != &ttype ||                                \
        g_##SUFFIX##_cblas_call.dmin1 != &dmin1 ||                                \
        g_##SUFFIX##_cblas_call.dmin2 != &dmin2 ||                                \
        g_##SUFFIX##_cblas_call.dn != (TYPE)12 ||                                 \
        g_##SUFFIX##_cblas_call.dn1 != (TYPE)13 ||                                \
        g_##SUFFIX##_cblas_call.dn2 != (TYPE)14 ||                                \
        g_##SUFFIX##_cblas_call.g != &g ||                                        \
        g_##SUFFIX##_cblas_call.tau != (TYPE)15 ||                                \
        dmin != (TYPE)((BASE) + 12) || sigma != (TYPE)((BASE) + 13) ||           \
        desig != (TYPE)((BASE) + 14) || qmax != (TYPE)((BASE) + 15) ||           \
        nfail != (BASE) + 16 || iter != (BASE) + 17 || ndiv != (BASE) + 18) {    \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference LASQ3 inputs correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences LASQ3 inputs\n"); \
    return 0;                                                                     \
}

DEFINE_LASQ3_TESTS(slasq3, float, FB_OP_SLASQ3, 100)
DEFINE_LASQ3_TESTS(dlasq3, double, FB_OP_DLASQ3, 300)

int main(void)
{
    int status = 0;

    status |= check_slasq3_fortran_to_cblas();
    status |= check_slasq3_cblas_to_fortran();
    status |= check_dlasq3_fortran_to_cblas();
    status |= check_dlasq3_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}