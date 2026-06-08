#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

#define DEFINE_LASD5_TESTS(SUFFIX, TYPE, OP_ID, BASE)                           \
typedef int (*fb_##SUFFIX##_cblas_fn)(int i, TYPE *d, TYPE *z, TYPE *delta,     \
                                      TYPE rho, TYPE *dsigma, TYPE *work);      \
typedef void (*fb_##SUFFIX##_fortran_fn)(int *i, TYPE *d, TYPE *z, TYPE *delta, \
                                         TYPE *rho, TYPE *dsigma, TYPE *work);   \
static struct {                                                                  \
    int called;                                                                  \
    int i;                                                                       \
    TYPE *d;                                                                     \
    TYPE *z;                                                                     \
    TYPE *delta;                                                                 \
    TYPE rho;                                                                    \
    TYPE *dsigma;                                                                \
    TYPE *work;                                                                  \
} g_##SUFFIX##_fortran_call;                                                     \
static struct {                                                                  \
    int called;                                                                  \
    int i;                                                                       \
    TYPE *d;                                                                     \
    TYPE *z;                                                                     \
    TYPE *delta;                                                                 \
    TYPE rho;                                                                    \
    TYPE *dsigma;                                                                \
    TYPE *work;                                                                  \
} g_##SUFFIX##_cblas_call;                                                       \
static void stub_##SUFFIX##_fortran(int *i, TYPE *d, TYPE *z, TYPE *delta,      \
                                    TYPE *rho, TYPE *dsigma, TYPE *work)        \
{                                                                                \
    g_##SUFFIX##_fortran_call.called += 1;                                       \
    g_##SUFFIX##_fortran_call.i = *i;                                            \
    g_##SUFFIX##_fortran_call.d = d;                                             \
    g_##SUFFIX##_fortran_call.z = z;                                             \
    g_##SUFFIX##_fortran_call.delta = delta;                                     \
    g_##SUFFIX##_fortran_call.rho = *rho;                                        \
    g_##SUFFIX##_fortran_call.dsigma = dsigma;                                   \
    g_##SUFFIX##_fortran_call.work = work;                                       \
    *dsigma = (TYPE)((BASE) + 1);                                                \
    delta[0] = (TYPE)((BASE) + 2);                                               \
    work[0] = (TYPE)((BASE) + 3);                                                \
}                                                                                \
static int stub_##SUFFIX##_cblas(int i, TYPE *d, TYPE *z, TYPE *delta, TYPE rho,\
                                 TYPE *dsigma, TYPE *work)                      \
{                                                                                \
    g_##SUFFIX##_cblas_call.called += 1;                                         \
    g_##SUFFIX##_cblas_call.i = i;                                               \
    g_##SUFFIX##_cblas_call.d = d;                                               \
    g_##SUFFIX##_cblas_call.z = z;                                               \
    g_##SUFFIX##_cblas_call.delta = delta;                                       \
    g_##SUFFIX##_cblas_call.rho = rho;                                           \
    g_##SUFFIX##_cblas_call.dsigma = dsigma;                                     \
    g_##SUFFIX##_cblas_call.work = work;                                         \
    *dsigma = (TYPE)((BASE) + 4);                                                \
    delta[0] = (TYPE)((BASE) + 5);                                               \
    work[0] = (TYPE)((BASE) + 6);                                                \
    return 0;                                                                    \
}                                                                                \
static int check_##SUFFIX##_fortran_to_cblas(void)                               \
{                                                                                \
    fb_backend_vtable_t vtable;                                                  \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                         \
    TYPE d[2] = { (TYPE)1, (TYPE)2 };                                            \
    TYPE z[2] = { (TYPE)3, (TYPE)4 };                                            \
    TYPE delta[2] = { (TYPE)5, (TYPE)6 };                                        \
    TYPE dsigma = (TYPE)0;                                                       \
    TYPE work[2] = { (TYPE)7, (TYPE)8 };                                         \
    memset(&vtable, 0, sizeof(vtable));                                          \
    memset(&g_##SUFFIX##_fortran_call, 0, sizeof(g_##SUFFIX##_fortran_call));    \
    vtable.ext_ops[OP_ID][FB_CONV_FORTRAN] =                                     \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_fortran;                  \
    fb_install_conv_thunks(&vtable, OP_ID);                                      \
    thunk = (fb_##SUFFIX##_cblas_fn)vtable.ext_ops[OP_ID][FB_CONV_CBLAS];        \
    if (!thunk) {                                                                \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk was not installed\n"); \
        return 1;                                                                \
    }                                                                            \
    if (thunk(2, d, z, delta, (TYPE)9, &dsigma, work) != 0 ||                   \
        g_##SUFFIX##_fortran_call.called != 1 ||                                 \
        g_##SUFFIX##_fortran_call.i != 2 ||                                      \
        g_##SUFFIX##_fortran_call.d != d ||                                      \
        g_##SUFFIX##_fortran_call.z != z ||                                      \
        g_##SUFFIX##_fortran_call.delta != delta ||                              \
        g_##SUFFIX##_fortran_call.rho != (TYPE)9 ||                              \
        g_##SUFFIX##_fortran_call.dsigma != &dsigma ||                           \
        g_##SUFFIX##_fortran_call.work != work ||                                \
        dsigma != (TYPE)((BASE) + 1) ||                                          \
        delta[0] != (TYPE)((BASE) + 2) ||                                        \
        work[0] != (TYPE)((BASE) + 3)) {                                         \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward LASD5 inputs correctly\n"); \
        return 1;                                                                \
    }                                                                            \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards LASD5 inputs\n"); \
    return 0;                                                                    \
}                                                                                \
static int check_##SUFFIX##_cblas_to_fortran(void)                               \
{                                                                                \
    fb_backend_vtable_t vtable;                                                  \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                       \
    int i = 2;                                                                   \
    TYPE rho = (TYPE)9;                                                          \
    TYPE d[2] = { (TYPE)11, (TYPE)12 };                                          \
    TYPE z[2] = { (TYPE)13, (TYPE)14 };                                          \
    TYPE delta[2] = { (TYPE)15, (TYPE)16 };                                      \
    TYPE dsigma = (TYPE)0;                                                       \
    TYPE work[2] = { (TYPE)17, (TYPE)18 };                                       \
    memset(&vtable, 0, sizeof(vtable));                                          \
    memset(&g_##SUFFIX##_cblas_call, 0, sizeof(g_##SUFFIX##_cblas_call));        \
    vtable.ext_ops[OP_ID][FB_CONV_CBLAS] =                                       \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_cblas;                    \
    fb_install_conv_thunks(&vtable, OP_ID);                                      \
    thunk = (fb_##SUFFIX##_fortran_fn)vtable.ext_ops[OP_ID][FB_CONV_FORTRAN];    \
    if (!thunk) {                                                                \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk was not installed\n"); \
        return 1;                                                                \
    }                                                                            \
    thunk(&i, d, z, delta, &rho, &dsigma, work);                                \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                   \
        g_##SUFFIX##_cblas_call.i != 2 ||                                        \
        g_##SUFFIX##_cblas_call.d != d ||                                        \
        g_##SUFFIX##_cblas_call.z != z ||                                        \
        g_##SUFFIX##_cblas_call.delta != delta ||                                \
        g_##SUFFIX##_cblas_call.rho != (TYPE)9 ||                                \
        g_##SUFFIX##_cblas_call.dsigma != &dsigma ||                             \
        g_##SUFFIX##_cblas_call.work != work ||                                  \
        dsigma != (TYPE)((BASE) + 4) ||                                          \
        delta[0] != (TYPE)((BASE) + 5) ||                                        \
        work[0] != (TYPE)((BASE) + 6)) {                                         \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference LASD5 inputs correctly\n"); \
        return 1;                                                                \
    }                                                                            \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences LASD5 inputs\n"); \
    return 0;                                                                    \
}

DEFINE_LASD5_TESTS(slasd5, float, FB_OP_SLASD5, 100)
DEFINE_LASD5_TESTS(dlasd5, double, FB_OP_DLASD5, 300)

int main(void)
{
    int status = 0;

    status |= check_slasd5_fortran_to_cblas();
    status |= check_slasd5_cblas_to_fortran();
    status |= check_dlasd5_fortran_to_cblas();
    status |= check_dlasd5_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}