#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

#define DEFINE_LASQ5_TESTS(SUFFIX, TYPE, OP_ID, BASE)                              \
typedef int (*fb_##SUFFIX##_cblas_fn)(int i0, int n0, TYPE *z, int pp, TYPE tau, \
                                      TYPE *sigma, TYPE *dmin, TYPE *dmin1,      \
                                      TYPE *dmin2, TYPE *dn, TYPE *dnm1,         \
                                      TYPE *dnm2, int ieee);                     \
typedef void (*fb_##SUFFIX##_fortran_fn)(int *i0, int *n0, TYPE *z, int *pp,    \
                                         TYPE *tau, TYPE *sigma, TYPE *dmin,     \
                                         TYPE *dmin1, TYPE *dmin2, TYPE *dn,     \
                                         TYPE *dnm1, TYPE *dnm2, int *ieee);     \
static struct {                                                                    \
    int called;                                                                    \
    int i0;                                                                        \
    int n0;                                                                        \
    TYPE *z;                                                                       \
    int pp;                                                                        \
    TYPE tau;                                                                      \
    TYPE *sigma;                                                                   \
    TYPE *dmin;                                                                    \
    TYPE *dmin1;                                                                   \
    TYPE *dmin2;                                                                   \
    TYPE *dn;                                                                      \
    TYPE *dnm1;                                                                    \
    TYPE *dnm2;                                                                    \
    int ieee;                                                                      \
} g_##SUFFIX##_fortran_call;                                                       \
static struct {                                                                    \
    int called;                                                                    \
    int i0;                                                                        \
    int n0;                                                                        \
    TYPE *z;                                                                       \
    int pp;                                                                        \
    TYPE tau;                                                                      \
    TYPE *sigma;                                                                   \
    TYPE *dmin;                                                                    \
    TYPE *dmin1;                                                                   \
    TYPE *dmin2;                                                                   \
    TYPE *dn;                                                                      \
    TYPE *dnm1;                                                                    \
    TYPE *dnm2;                                                                    \
    int ieee;                                                                      \
} g_##SUFFIX##_cblas_call;                                                         \
static void stub_##SUFFIX##_fortran(int *i0, int *n0, TYPE *z, int *pp,         \
                                    TYPE *tau, TYPE *sigma, TYPE *dmin,         \
                                    TYPE *dmin1, TYPE *dmin2, TYPE *dn,         \
                                    TYPE *dnm1, TYPE *dnm2, int *ieee) {        \
    g_##SUFFIX##_fortran_call.called += 1;                                        \
    g_##SUFFIX##_fortran_call.i0 = *i0;                                           \
    g_##SUFFIX##_fortran_call.n0 = *n0;                                           \
    g_##SUFFIX##_fortran_call.z = z;                                              \
    g_##SUFFIX##_fortran_call.pp = *pp;                                           \
    g_##SUFFIX##_fortran_call.tau = *tau;                                         \
    g_##SUFFIX##_fortran_call.sigma = sigma;                                      \
    g_##SUFFIX##_fortran_call.dmin = dmin;                                        \
    g_##SUFFIX##_fortran_call.dmin1 = dmin1;                                      \
    g_##SUFFIX##_fortran_call.dmin2 = dmin2;                                      \
    g_##SUFFIX##_fortran_call.dn = dn;                                            \
    g_##SUFFIX##_fortran_call.dnm1 = dnm1;                                        \
    g_##SUFFIX##_fortran_call.dnm2 = dnm2;                                        \
    g_##SUFFIX##_fortran_call.ieee = *ieee;                                       \
    z[0] = (TYPE)((BASE) + 1);                                                    \
    *sigma = (TYPE)((BASE) + 2);                                                  \
    *dmin = (TYPE)((BASE) + 3);                                                   \
    *dmin1 = (TYPE)((BASE) + 4);                                                  \
    *dmin2 = (TYPE)((BASE) + 5);                                                  \
    *dn = (TYPE)((BASE) + 6);                                                     \
    *dnm1 = (TYPE)((BASE) + 7);                                                   \
    *dnm2 = (TYPE)((BASE) + 8);                                                   \
}                                                                                 \
static int stub_##SUFFIX##_cblas(int i0, int n0, TYPE *z, int pp, TYPE tau,     \
                                 TYPE *sigma, TYPE *dmin, TYPE *dmin1,          \
                                 TYPE *dmin2, TYPE *dn, TYPE *dnm1,             \
                                 TYPE *dnm2, int ieee) {                         \
    g_##SUFFIX##_cblas_call.called += 1;                                         \
    g_##SUFFIX##_cblas_call.i0 = i0;                                             \
    g_##SUFFIX##_cblas_call.n0 = n0;                                             \
    g_##SUFFIX##_cblas_call.z = z;                                               \
    g_##SUFFIX##_cblas_call.pp = pp;                                             \
    g_##SUFFIX##_cblas_call.tau = tau;                                           \
    g_##SUFFIX##_cblas_call.sigma = sigma;                                       \
    g_##SUFFIX##_cblas_call.dmin = dmin;                                         \
    g_##SUFFIX##_cblas_call.dmin1 = dmin1;                                       \
    g_##SUFFIX##_cblas_call.dmin2 = dmin2;                                       \
    g_##SUFFIX##_cblas_call.dn = dn;                                             \
    g_##SUFFIX##_cblas_call.dnm1 = dnm1;                                         \
    g_##SUFFIX##_cblas_call.dnm2 = dnm2;                                         \
    g_##SUFFIX##_cblas_call.ieee = ieee;                                         \
    z[0] = (TYPE)((BASE) + 9);                                                   \
    *sigma = (TYPE)((BASE) + 10);                                                \
    *dmin = (TYPE)((BASE) + 11);                                                 \
    *dmin1 = (TYPE)((BASE) + 12);                                                \
    *dmin2 = (TYPE)((BASE) + 13);                                                \
    *dn = (TYPE)((BASE) + 14);                                                   \
    *dnm1 = (TYPE)((BASE) + 15);                                                 \
    *dnm2 = (TYPE)((BASE) + 16);                                                 \
    return 0;                                                                    \
}                                                                                \
static int check_##SUFFIX##_fortran_to_cblas(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                          \
    TYPE z[3] = { (TYPE)1, (TYPE)2, (TYPE)3 };                                    \
    TYPE sigma = (TYPE)0;                                                         \
    TYPE dmin = (TYPE)0;                                                          \
    TYPE dmin1 = (TYPE)0;                                                         \
    TYPE dmin2 = (TYPE)0;                                                         \
    TYPE dn = (TYPE)0;                                                            \
    TYPE dnm1 = (TYPE)0;                                                          \
    TYPE dnm2 = (TYPE)0;                                                          \
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
    if (thunk(1, 3, z, 0, (TYPE)5, &sigma, &dmin, &dmin1, &dmin2, &dn, &dnm1,   \
              &dnm2, 1) != 0 ||                                                   \
        g_##SUFFIX##_fortran_call.called != 1 ||                                  \
        g_##SUFFIX##_fortran_call.i0 != 1 ||                                      \
        g_##SUFFIX##_fortran_call.n0 != 3 ||                                      \
        g_##SUFFIX##_fortran_call.z != z ||                                       \
        g_##SUFFIX##_fortran_call.pp != 0 ||                                      \
        g_##SUFFIX##_fortran_call.tau != (TYPE)5 ||                               \
        g_##SUFFIX##_fortran_call.sigma != &sigma ||                              \
        g_##SUFFIX##_fortran_call.dmin != &dmin ||                                \
        g_##SUFFIX##_fortran_call.dmin1 != &dmin1 ||                              \
        g_##SUFFIX##_fortran_call.dmin2 != &dmin2 ||                              \
        g_##SUFFIX##_fortran_call.dn != &dn ||                                    \
        g_##SUFFIX##_fortran_call.dnm1 != &dnm1 ||                                \
        g_##SUFFIX##_fortran_call.dnm2 != &dnm2 ||                                \
        g_##SUFFIX##_fortran_call.ieee != 1 ||                                    \
        z[0] != (TYPE)((BASE) + 1) || sigma != (TYPE)((BASE) + 2) ||             \
        dmin != (TYPE)((BASE) + 3) || dmin1 != (TYPE)((BASE) + 4) ||             \
        dmin2 != (TYPE)((BASE) + 5) || dn != (TYPE)((BASE) + 6) ||               \
        dnm1 != (TYPE)((BASE) + 7) || dnm2 != (TYPE)((BASE) + 8)) {              \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward LASQ5 state correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards LASQ5 state\n"); \
    return 0;                                                                     \
}                                                                                 \
static int check_##SUFFIX##_cblas_to_fortran(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                        \
    int i0 = 1;                                                                   \
    int n0 = 3;                                                                   \
    TYPE z[3] = { (TYPE)1, (TYPE)2, (TYPE)3 };                                    \
    int pp = 0;                                                                   \
    TYPE tau = (TYPE)5;                                                           \
    TYPE sigma = (TYPE)0;                                                         \
    TYPE dmin = (TYPE)0;                                                          \
    TYPE dmin1 = (TYPE)0;                                                         \
    TYPE dmin2 = (TYPE)0;                                                         \
    TYPE dn = (TYPE)0;                                                            \
    TYPE dnm1 = (TYPE)0;                                                          \
    TYPE dnm2 = (TYPE)0;                                                          \
    int ieee = 1;                                                                 \
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
    thunk(&i0, &n0, z, &pp, &tau, &sigma, &dmin, &dmin1, &dmin2, &dn, &dnm1,    \
          &dnm2, &ieee);                                                          \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                    \
        g_##SUFFIX##_cblas_call.i0 != 1 ||                                        \
        g_##SUFFIX##_cblas_call.n0 != 3 ||                                        \
        g_##SUFFIX##_cblas_call.z != z ||                                         \
        g_##SUFFIX##_cblas_call.pp != 0 ||                                        \
        g_##SUFFIX##_cblas_call.tau != (TYPE)5 ||                                 \
        g_##SUFFIX##_cblas_call.sigma != &sigma ||                                \
        g_##SUFFIX##_cblas_call.dmin != &dmin ||                                  \
        g_##SUFFIX##_cblas_call.dmin1 != &dmin1 ||                                \
        g_##SUFFIX##_cblas_call.dmin2 != &dmin2 ||                                \
        g_##SUFFIX##_cblas_call.dn != &dn ||                                      \
        g_##SUFFIX##_cblas_call.dnm1 != &dnm1 ||                                  \
        g_##SUFFIX##_cblas_call.dnm2 != &dnm2 ||                                  \
        g_##SUFFIX##_cblas_call.ieee != 1 ||                                      \
        z[0] != (TYPE)((BASE) + 9) || sigma != (TYPE)((BASE) + 10) ||            \
        dmin != (TYPE)((BASE) + 11) || dmin1 != (TYPE)((BASE) + 12) ||           \
        dmin2 != (TYPE)((BASE) + 13) || dn != (TYPE)((BASE) + 14) ||             \
        dnm1 != (TYPE)((BASE) + 15) || dnm2 != (TYPE)((BASE) + 16)) {            \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference LASQ5 state correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences LASQ5 state\n"); \
    return 0;                                                                     \
}

DEFINE_LASQ5_TESTS(slasq5, float, FB_OP_SLASQ5, 100)
DEFINE_LASQ5_TESTS(dlasq5, double, FB_OP_DLASQ5, 300)

int main(void)
{
    int status = 0;

    status |= check_slasq5_fortran_to_cblas();
    status |= check_slasq5_cblas_to_fortran();
    status |= check_dlasq5_fortran_to_cblas();
    status |= check_dlasq5_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}