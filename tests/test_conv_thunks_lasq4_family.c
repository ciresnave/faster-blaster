#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

#define DEFINE_LASQ4_TESTS(SUFFIX, TYPE, OP_ID, BASE)                              \
typedef int (*fb_##SUFFIX##_cblas_fn)(int i0, int n0, TYPE *z, int pp, int n0in, \
                                      TYPE dmin, TYPE dmin1, TYPE dmin2,         \
                                      TYPE dn, TYPE dn1, TYPE dn2, TYPE *tau,    \
                                      int *ttype, TYPE *g);                      \
typedef void (*fb_##SUFFIX##_fortran_fn)(int *i0, int *n0, TYPE *z, int *pp,    \
                                         int *n0in, TYPE *dmin, TYPE *dmin1,     \
                                         TYPE *dmin2, TYPE *dn, TYPE *dn1,       \
                                         TYPE *dn2, TYPE *tau, int *ttype,       \
                                         TYPE *g);                               \
static struct {                                                                    \
    int called;                                                                    \
    int i0;                                                                        \
    int n0;                                                                        \
    TYPE *z;                                                                       \
    int pp;                                                                        \
    int n0in;                                                                      \
    TYPE dmin;                                                                     \
    TYPE dmin1;                                                                    \
    TYPE dmin2;                                                                    \
    TYPE dn;                                                                       \
    TYPE dn1;                                                                      \
    TYPE dn2;                                                                      \
    TYPE *tau;                                                                     \
    int *ttype;                                                                    \
    TYPE *g;                                                                       \
} g_##SUFFIX##_fortran_call;                                                       \
static struct {                                                                    \
    int called;                                                                    \
    int i0;                                                                        \
    int n0;                                                                        \
    TYPE *z;                                                                       \
    int pp;                                                                        \
    int n0in;                                                                      \
    TYPE dmin;                                                                     \
    TYPE dmin1;                                                                    \
    TYPE dmin2;                                                                    \
    TYPE dn;                                                                       \
    TYPE dn1;                                                                      \
    TYPE dn2;                                                                      \
    TYPE *tau;                                                                     \
    int *ttype;                                                                    \
    TYPE *g;                                                                       \
} g_##SUFFIX##_cblas_call;                                                         \
static void stub_##SUFFIX##_fortran(int *i0, int *n0, TYPE *z, int *pp,         \
                                    int *n0in, TYPE *dmin, TYPE *dmin1,         \
                                    TYPE *dmin2, TYPE *dn, TYPE *dn1, TYPE *dn2,\
                                    TYPE *tau, int *ttype, TYPE *g) {           \
    g_##SUFFIX##_fortran_call.called += 1;                                        \
    g_##SUFFIX##_fortran_call.i0 = *i0;                                           \
    g_##SUFFIX##_fortran_call.n0 = *n0;                                           \
    g_##SUFFIX##_fortran_call.z = z;                                              \
    g_##SUFFIX##_fortran_call.pp = *pp;                                           \
    g_##SUFFIX##_fortran_call.n0in = *n0in;                                       \
    g_##SUFFIX##_fortran_call.dmin = *dmin;                                       \
    g_##SUFFIX##_fortran_call.dmin1 = *dmin1;                                     \
    g_##SUFFIX##_fortran_call.dmin2 = *dmin2;                                     \
    g_##SUFFIX##_fortran_call.dn = *dn;                                           \
    g_##SUFFIX##_fortran_call.dn1 = *dn1;                                         \
    g_##SUFFIX##_fortran_call.dn2 = *dn2;                                         \
    g_##SUFFIX##_fortran_call.tau = tau;                                          \
    g_##SUFFIX##_fortran_call.ttype = ttype;                                      \
    g_##SUFFIX##_fortran_call.g = g;                                              \
    *tau = (TYPE)((BASE) + 1);                                                    \
    *ttype = (BASE) + 2;                                                          \
    *g = (TYPE)((BASE) + 3);                                                      \
}                                                                                 \
static int stub_##SUFFIX##_cblas(int i0, int n0, TYPE *z, int pp, int n0in,     \
                                 TYPE dmin, TYPE dmin1, TYPE dmin2, TYPE dn,    \
                                 TYPE dn1, TYPE dn2, TYPE *tau, int *ttype,     \
                                 TYPE *g) {                                      \
    g_##SUFFIX##_cblas_call.called += 1;                                         \
    g_##SUFFIX##_cblas_call.i0 = i0;                                             \
    g_##SUFFIX##_cblas_call.n0 = n0;                                             \
    g_##SUFFIX##_cblas_call.z = z;                                               \
    g_##SUFFIX##_cblas_call.pp = pp;                                             \
    g_##SUFFIX##_cblas_call.n0in = n0in;                                         \
    g_##SUFFIX##_cblas_call.dmin = dmin;                                         \
    g_##SUFFIX##_cblas_call.dmin1 = dmin1;                                       \
    g_##SUFFIX##_cblas_call.dmin2 = dmin2;                                       \
    g_##SUFFIX##_cblas_call.dn = dn;                                             \
    g_##SUFFIX##_cblas_call.dn1 = dn1;                                           \
    g_##SUFFIX##_cblas_call.dn2 = dn2;                                           \
    g_##SUFFIX##_cblas_call.tau = tau;                                           \
    g_##SUFFIX##_cblas_call.ttype = ttype;                                       \
    g_##SUFFIX##_cblas_call.g = g;                                               \
    *tau = (TYPE)((BASE) + 4);                                                   \
    *ttype = (BASE) + 5;                                                         \
    *g = (TYPE)((BASE) + 6);                                                     \
    return 0;                                                                    \
}                                                                                \
static int check_##SUFFIX##_fortran_to_cblas(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                          \
    TYPE z[1] = { (TYPE)7 };                                                      \
    TYPE tau = (TYPE)0;                                                           \
    int ttype = 0;                                                                \
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
    if (thunk(1, 3, z, 0, 4, (TYPE)10, (TYPE)11, (TYPE)12, (TYPE)13, (TYPE)14,  \
              (TYPE)15, &tau, &ttype, &g) != 0 ||                                \
        g_##SUFFIX##_fortran_call.called != 1 ||                                  \
        g_##SUFFIX##_fortran_call.i0 != 1 ||                                      \
        g_##SUFFIX##_fortran_call.n0 != 3 ||                                      \
        g_##SUFFIX##_fortran_call.z != z ||                                       \
        g_##SUFFIX##_fortran_call.pp != 0 ||                                      \
        g_##SUFFIX##_fortran_call.n0in != 4 ||                                    \
        g_##SUFFIX##_fortran_call.dmin != (TYPE)10 ||                             \
        g_##SUFFIX##_fortran_call.dmin1 != (TYPE)11 ||                            \
        g_##SUFFIX##_fortran_call.dmin2 != (TYPE)12 ||                            \
        g_##SUFFIX##_fortran_call.dn != (TYPE)13 ||                               \
        g_##SUFFIX##_fortran_call.dn1 != (TYPE)14 ||                              \
        g_##SUFFIX##_fortran_call.dn2 != (TYPE)15 ||                              \
        g_##SUFFIX##_fortran_call.tau != &tau ||                                  \
        g_##SUFFIX##_fortran_call.ttype != &ttype ||                              \
        g_##SUFFIX##_fortran_call.g != &g ||                                      \
        tau != (TYPE)((BASE) + 1) || ttype != (BASE) + 2 || g != (TYPE)((BASE) + 3)) { \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not normalize LASQ4 inputs correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk normalizes LASQ4 inputs\n"); \
    return 0;                                                                     \
}                                                                                 \
static int check_##SUFFIX##_cblas_to_fortran(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                        \
    int i0 = 1;                                                                   \
    int n0 = 3;                                                                   \
    TYPE z[1] = { (TYPE)7 };                                                      \
    int pp = 0;                                                                   \
    int n0in = 4;                                                                 \
    TYPE dmin = (TYPE)10;                                                         \
    TYPE dmin1 = (TYPE)11;                                                        \
    TYPE dmin2 = (TYPE)12;                                                        \
    TYPE dn = (TYPE)13;                                                           \
    TYPE dn1 = (TYPE)14;                                                          \
    TYPE dn2 = (TYPE)15;                                                          \
    TYPE tau = (TYPE)0;                                                           \
    int ttype = 0;                                                                \
    TYPE g = (TYPE)0;                                                             \
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
    thunk(&i0, &n0, z, &pp, &n0in, &dmin, &dmin1, &dmin2, &dn, &dn1, &dn2,      \
          &tau, &ttype, &g);                                                      \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                    \
        g_##SUFFIX##_cblas_call.i0 != 1 ||                                        \
        g_##SUFFIX##_cblas_call.n0 != 3 ||                                        \
        g_##SUFFIX##_cblas_call.z != z ||                                         \
        g_##SUFFIX##_cblas_call.pp != 0 ||                                        \
        g_##SUFFIX##_cblas_call.n0in != 4 ||                                      \
        g_##SUFFIX##_cblas_call.dmin != (TYPE)10 ||                               \
        g_##SUFFIX##_cblas_call.dmin1 != (TYPE)11 ||                              \
        g_##SUFFIX##_cblas_call.dmin2 != (TYPE)12 ||                              \
        g_##SUFFIX##_cblas_call.dn != (TYPE)13 ||                                 \
        g_##SUFFIX##_cblas_call.dn1 != (TYPE)14 ||                                \
        g_##SUFFIX##_cblas_call.dn2 != (TYPE)15 ||                                \
        g_##SUFFIX##_cblas_call.tau != &tau ||                                    \
        g_##SUFFIX##_cblas_call.ttype != &ttype ||                                \
        g_##SUFFIX##_cblas_call.g != &g ||                                        \
        tau != (TYPE)((BASE) + 4) || ttype != (BASE) + 5 || g != (TYPE)((BASE) + 6)) { \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference LASQ4 inputs correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences LASQ4 inputs\n"); \
    return 0;                                                                     \
}

DEFINE_LASQ4_TESTS(slasq4, float, FB_OP_SLASQ4, 100)
DEFINE_LASQ4_TESTS(dlasq4, double, FB_OP_DLASQ4, 300)

int main(void)
{
    int status = 0;

    status |= check_slasq4_fortran_to_cblas();
    status |= check_slasq4_cblas_to_fortran();
    status |= check_dlasq4_fortran_to_cblas();
    status |= check_dlasq4_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}