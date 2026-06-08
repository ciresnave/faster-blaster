#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

#define DEFINE_LASD4_TESTS(SUFFIX, TYPE, OP_ID, BASE)                           \
typedef int (*fb_##SUFFIX##_cblas_fn)(int n, TYPE *d, TYPE *z, TYPE *delta,     \
                                      TYPE rho, TYPE *sigma, TYPE *work);       \
typedef void (*fb_##SUFFIX##_fortran_fn)(int *n, TYPE *d, TYPE *z, TYPE *delta, \
                                         TYPE *rho, TYPE *sigma, TYPE *work,     \
                                         int *info);                             \
static struct {                                                                  \
    int called;                                                                  \
    int n;                                                                       \
    TYPE *d;                                                                     \
    TYPE *z;                                                                     \
    TYPE *delta;                                                                 \
    TYPE rho;                                                                    \
    TYPE *sigma;                                                                 \
    TYPE *work;                                                                  \
} g_##SUFFIX##_fortran_call;                                                     \
static struct {                                                                  \
    int called;                                                                  \
    int n;                                                                       \
    TYPE *d;                                                                     \
    TYPE *z;                                                                     \
    TYPE *delta;                                                                 \
    TYPE rho;                                                                    \
    TYPE *sigma;                                                                 \
    TYPE *work;                                                                  \
} g_##SUFFIX##_cblas_call;                                                       \
static void stub_##SUFFIX##_fortran(int *n, TYPE *d, TYPE *z, TYPE *delta,      \
                                    TYPE *rho, TYPE *sigma, TYPE *work,         \
                                    int *info)                                  \
{                                                                                \
    g_##SUFFIX##_fortran_call.called += 1;                                       \
    g_##SUFFIX##_fortran_call.n = *n;                                            \
    g_##SUFFIX##_fortran_call.d = d;                                             \
    g_##SUFFIX##_fortran_call.z = z;                                             \
    g_##SUFFIX##_fortran_call.delta = delta;                                     \
    g_##SUFFIX##_fortran_call.rho = *rho;                                        \
    g_##SUFFIX##_fortran_call.sigma = sigma;                                     \
    g_##SUFFIX##_fortran_call.work = work;                                       \
    *sigma = (TYPE)((BASE) + 1);                                                 \
    delta[0] = (TYPE)((BASE) + 2);                                               \
    work[0] = (TYPE)((BASE) + 3);                                                \
    *info = (BASE) + 4;                                                          \
}                                                                                \
static int stub_##SUFFIX##_cblas(int n, TYPE *d, TYPE *z, TYPE *delta, TYPE rho,\
                                 TYPE *sigma, TYPE *work)                       \
{                                                                                \
    g_##SUFFIX##_cblas_call.called += 1;                                         \
    g_##SUFFIX##_cblas_call.n = n;                                               \
    g_##SUFFIX##_cblas_call.d = d;                                               \
    g_##SUFFIX##_cblas_call.z = z;                                               \
    g_##SUFFIX##_cblas_call.delta = delta;                                       \
    g_##SUFFIX##_cblas_call.rho = rho;                                           \
    g_##SUFFIX##_cblas_call.sigma = sigma;                                       \
    g_##SUFFIX##_cblas_call.work = work;                                         \
    *sigma = (TYPE)((BASE) + 5);                                                 \
    delta[0] = (TYPE)((BASE) + 6);                                               \
    work[0] = (TYPE)((BASE) + 7);                                                \
    return (BASE) + 8;                                                           \
}                                                                                \
static int check_##SUFFIX##_fortran_to_cblas(void)                               \
{                                                                                \
    fb_backend_vtable_t vtable;                                                  \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                         \
    TYPE d[3] = { (TYPE)1, (TYPE)2, (TYPE)3 };                                  \
    TYPE z[3] = { (TYPE)4, (TYPE)5, (TYPE)6 };                                  \
    TYPE delta[3] = { (TYPE)7, (TYPE)8, (TYPE)9 };                              \
    TYPE sigma = (TYPE)0;                                                        \
    TYPE work[3] = { (TYPE)10, (TYPE)11, (TYPE)12 };                            \
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
    if (thunk(3, d, z, delta, (TYPE)13, &sigma, work) != (BASE) + 4 ||          \
        g_##SUFFIX##_fortran_call.called != 1 ||                                 \
        g_##SUFFIX##_fortran_call.n != 3 ||                                      \
        g_##SUFFIX##_fortran_call.d != d ||                                      \
        g_##SUFFIX##_fortran_call.z != z ||                                      \
        g_##SUFFIX##_fortran_call.delta != delta ||                              \
        g_##SUFFIX##_fortran_call.rho != (TYPE)13 ||                             \
        g_##SUFFIX##_fortran_call.sigma != &sigma ||                             \
        g_##SUFFIX##_fortran_call.work != work ||                                \
        sigma != (TYPE)((BASE) + 1) ||                                           \
        delta[0] != (TYPE)((BASE) + 2) ||                                        \
        work[0] != (TYPE)((BASE) + 3)) {                                         \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward LASD4 inputs correctly\n"); \
        return 1;                                                                \
    }                                                                            \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards LASD4 inputs\n"); \
    return 0;                                                                    \
}                                                                                \
static int check_##SUFFIX##_cblas_to_fortran(void)                               \
{                                                                                \
    fb_backend_vtable_t vtable;                                                  \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                       \
    int n = 3;                                                                   \
    TYPE rho = (TYPE)13;                                                         \
    TYPE d[3] = { (TYPE)21, (TYPE)22, (TYPE)23 };                               \
    TYPE z[3] = { (TYPE)24, (TYPE)25, (TYPE)26 };                               \
    TYPE delta[3] = { (TYPE)27, (TYPE)28, (TYPE)29 };                           \
    TYPE sigma = (TYPE)0;                                                        \
    TYPE work[3] = { (TYPE)30, (TYPE)31, (TYPE)32 };                            \
    int info = -1;                                                               \
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
    thunk(&n, d, z, delta, &rho, &sigma, work, &info);                          \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                   \
        g_##SUFFIX##_cblas_call.n != 3 ||                                        \
        g_##SUFFIX##_cblas_call.d != d ||                                        \
        g_##SUFFIX##_cblas_call.z != z ||                                        \
        g_##SUFFIX##_cblas_call.delta != delta ||                                \
        g_##SUFFIX##_cblas_call.rho != (TYPE)13 ||                               \
        g_##SUFFIX##_cblas_call.sigma != &sigma ||                               \
        g_##SUFFIX##_cblas_call.work != work ||                                  \
        info != (BASE) + 8 ||                                                    \
        sigma != (TYPE)((BASE) + 5) ||                                           \
        delta[0] != (TYPE)((BASE) + 6) ||                                        \
        work[0] != (TYPE)((BASE) + 7)) {                                         \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference LASD4 inputs correctly\n"); \
        return 1;                                                                \
    }                                                                            \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences LASD4 inputs\n"); \
    return 0;                                                                    \
}

DEFINE_LASD4_TESTS(slasd4, float, FB_OP_SLASD4, 100)
DEFINE_LASD4_TESTS(dlasd4, double, FB_OP_DLASD4, 300)

int main(void)
{
    int status = 0;

    status |= check_slasd4_fortran_to_cblas();
    status |= check_slasd4_cblas_to_fortran();
    status |= check_dlasd4_fortran_to_cblas();
    status |= check_dlasd4_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}