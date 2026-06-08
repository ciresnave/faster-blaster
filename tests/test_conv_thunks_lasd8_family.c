#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

#define DEFINE_LASD8_TESTS(SUFFIX, TYPE, OP_ID, BASE)                           \
typedef int (*fb_##SUFFIX##_cblas_fn)(int icompq, int k, TYPE *d, TYPE *z,      \
                                      TYPE *vf, TYPE *vl, TYPE *difl,           \
                                      TYPE *difr, int lddifr, TYPE *dsigma,     \
                                      TYPE *work);                              \
typedef void (*fb_##SUFFIX##_fortran_fn)(int *icompq, int *k, TYPE *d, TYPE *z,\
                                         TYPE *vf, TYPE *vl, TYPE *difl,        \
                                         TYPE *difr, int *lddifr, TYPE *dsigma, \
                                         TYPE *work, int *info);                \
static struct {                                                                  \
    int called;                                                                  \
    int icompq;                                                                  \
    int k;                                                                       \
    TYPE *d;                                                                     \
    TYPE *z;                                                                     \
    TYPE *vf;                                                                    \
    TYPE *vl;                                                                    \
    TYPE *difl;                                                                  \
    TYPE *difr;                                                                  \
    int lddifr;                                                                  \
    TYPE *dsigma;                                                                \
    TYPE *work;                                                                  \
} g_##SUFFIX##_fortran_call;                                                     \
static struct {                                                                  \
    int called;                                                                  \
    int icompq;                                                                  \
    int k;                                                                       \
    TYPE *d;                                                                     \
    TYPE *z;                                                                     \
    TYPE *vf;                                                                    \
    TYPE *vl;                                                                    \
    TYPE *difl;                                                                  \
    TYPE *difr;                                                                  \
    int lddifr;                                                                  \
    TYPE *dsigma;                                                                \
    TYPE *work;                                                                  \
} g_##SUFFIX##_cblas_call;                                                       \
static void stub_##SUFFIX##_fortran(int *icompq, int *k, TYPE *d, TYPE *z,      \
                                    TYPE *vf, TYPE *vl, TYPE *difl, TYPE *difr, \
                                    int *lddifr, TYPE *dsigma, TYPE *work,      \
                                    int *info)                                  \
{                                                                                \
    g_##SUFFIX##_fortran_call.called += 1;                                       \
    g_##SUFFIX##_fortran_call.icompq = *icompq;                                  \
    g_##SUFFIX##_fortran_call.k = *k;                                            \
    g_##SUFFIX##_fortran_call.d = d;                                             \
    g_##SUFFIX##_fortran_call.z = z;                                             \
    g_##SUFFIX##_fortran_call.vf = vf;                                           \
    g_##SUFFIX##_fortran_call.vl = vl;                                           \
    g_##SUFFIX##_fortran_call.difl = difl;                                       \
    g_##SUFFIX##_fortran_call.difr = difr;                                       \
    g_##SUFFIX##_fortran_call.lddifr = *lddifr;                                  \
    g_##SUFFIX##_fortran_call.dsigma = dsigma;                                   \
    g_##SUFFIX##_fortran_call.work = work;                                       \
    d[0] = (TYPE)((BASE) + 1);                                                   \
    vf[0] = (TYPE)((BASE) + 2);                                                  \
    vl[0] = (TYPE)((BASE) + 3);                                                  \
    difl[0] = (TYPE)((BASE) + 4);                                                \
    difr[0] = (TYPE)((BASE) + 5);                                                \
    dsigma[0] = (TYPE)((BASE) + 6);                                              \
    work[0] = (TYPE)((BASE) + 7);                                                \
    *info = (BASE) + 8;                                                          \
}                                                                                \
static int stub_##SUFFIX##_cblas(int icompq, int k, TYPE *d, TYPE *z, TYPE *vf, \
                                 TYPE *vl, TYPE *difl, TYPE *difr, int lddifr,  \
                                 TYPE *dsigma, TYPE *work)                      \
{                                                                                \
    g_##SUFFIX##_cblas_call.called += 1;                                         \
    g_##SUFFIX##_cblas_call.icompq = icompq;                                     \
    g_##SUFFIX##_cblas_call.k = k;                                               \
    g_##SUFFIX##_cblas_call.d = d;                                               \
    g_##SUFFIX##_cblas_call.z = z;                                               \
    g_##SUFFIX##_cblas_call.vf = vf;                                             \
    g_##SUFFIX##_cblas_call.vl = vl;                                             \
    g_##SUFFIX##_cblas_call.difl = difl;                                         \
    g_##SUFFIX##_cblas_call.difr = difr;                                         \
    g_##SUFFIX##_cblas_call.lddifr = lddifr;                                     \
    g_##SUFFIX##_cblas_call.dsigma = dsigma;                                     \
    g_##SUFFIX##_cblas_call.work = work;                                         \
    d[0] = (TYPE)((BASE) + 9);                                                   \
    vf[0] = (TYPE)((BASE) + 10);                                                 \
    vl[0] = (TYPE)((BASE) + 11);                                                 \
    difl[0] = (TYPE)((BASE) + 12);                                               \
    difr[0] = (TYPE)((BASE) + 13);                                               \
    dsigma[0] = (TYPE)((BASE) + 14);                                             \
    work[0] = (TYPE)((BASE) + 15);                                               \
    return (BASE) + 16;                                                          \
}                                                                                \
static int check_##SUFFIX##_fortran_to_cblas(void)                               \
{                                                                                \
    fb_backend_vtable_t vtable;                                                  \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                         \
    TYPE d[2] = { (TYPE)1, (TYPE)2 };                                            \
    TYPE z[2] = { (TYPE)3, (TYPE)4 };                                            \
    TYPE vf[2] = { (TYPE)5, (TYPE)6 };                                           \
    TYPE vl[2] = { (TYPE)7, (TYPE)8 };                                           \
    TYPE difl[2] = { (TYPE)9, (TYPE)10 };                                        \
    TYPE difr[4] = { (TYPE)11, (TYPE)12, (TYPE)13, (TYPE)14 };                  \
    TYPE dsigma[2] = { (TYPE)15, (TYPE)16 };                                     \
    TYPE work[3] = { (TYPE)17, (TYPE)18, (TYPE)19 };                             \
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
    if (thunk(1, 2, d, z, vf, vl, difl, difr, 2, dsigma, work) != (BASE) + 8 || \
        g_##SUFFIX##_fortran_call.called != 1 ||                                 \
        g_##SUFFIX##_fortran_call.icompq != 1 ||                                 \
        g_##SUFFIX##_fortran_call.k != 2 ||                                      \
        g_##SUFFIX##_fortran_call.d != d ||                                      \
        g_##SUFFIX##_fortran_call.z != z ||                                      \
        g_##SUFFIX##_fortran_call.vf != vf ||                                    \
        g_##SUFFIX##_fortran_call.vl != vl ||                                    \
        g_##SUFFIX##_fortran_call.difl != difl ||                                \
        g_##SUFFIX##_fortran_call.difr != difr ||                                \
        g_##SUFFIX##_fortran_call.lddifr != 2 ||                                 \
        g_##SUFFIX##_fortran_call.dsigma != dsigma ||                            \
        g_##SUFFIX##_fortran_call.work != work ||                                \
        d[0] != (TYPE)((BASE) + 1) ||                                            \
        vf[0] != (TYPE)((BASE) + 2) ||                                           \
        vl[0] != (TYPE)((BASE) + 3) ||                                           \
        difl[0] != (TYPE)((BASE) + 4) ||                                         \
        difr[0] != (TYPE)((BASE) + 5) ||                                         \
        dsigma[0] != (TYPE)((BASE) + 6) ||                                       \
        work[0] != (TYPE)((BASE) + 7)) {                                         \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward LASD8 inputs correctly\n"); \
        return 1;                                                                \
    }                                                                            \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards LASD8 inputs\n"); \
    return 0;                                                                    \
}                                                                                \
static int check_##SUFFIX##_cblas_to_fortran(void)                               \
{                                                                                \
    fb_backend_vtable_t vtable;                                                  \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                       \
    int icompq = 1;                                                              \
    int k = 2;                                                                   \
    int lddifr = 2;                                                              \
    int info = -1;                                                               \
    TYPE d[2] = { (TYPE)21, (TYPE)22 };                                          \
    TYPE z[2] = { (TYPE)23, (TYPE)24 };                                          \
    TYPE vf[2] = { (TYPE)25, (TYPE)26 };                                         \
    TYPE vl[2] = { (TYPE)27, (TYPE)28 };                                         \
    TYPE difl[2] = { (TYPE)29, (TYPE)30 };                                       \
    TYPE difr[4] = { (TYPE)31, (TYPE)32, (TYPE)33, (TYPE)34 };                  \
    TYPE dsigma[2] = { (TYPE)35, (TYPE)36 };                                     \
    TYPE work[3] = { (TYPE)37, (TYPE)38, (TYPE)39 };                             \
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
    thunk(&icompq, &k, d, z, vf, vl, difl, difr, &lddifr, dsigma, work, &info); \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                   \
        g_##SUFFIX##_cblas_call.icompq != 1 ||                                   \
        g_##SUFFIX##_cblas_call.k != 2 ||                                        \
        g_##SUFFIX##_cblas_call.d != d ||                                        \
        g_##SUFFIX##_cblas_call.z != z ||                                        \
        g_##SUFFIX##_cblas_call.vf != vf ||                                      \
        g_##SUFFIX##_cblas_call.vl != vl ||                                      \
        g_##SUFFIX##_cblas_call.difl != difl ||                                  \
        g_##SUFFIX##_cblas_call.difr != difr ||                                  \
        g_##SUFFIX##_cblas_call.lddifr != 2 ||                                   \
        g_##SUFFIX##_cblas_call.dsigma != dsigma ||                              \
        g_##SUFFIX##_cblas_call.work != work ||                                  \
        info != (BASE) + 16 ||                                                   \
        d[0] != (TYPE)((BASE) + 9) ||                                            \
        vf[0] != (TYPE)((BASE) + 10) ||                                          \
        vl[0] != (TYPE)((BASE) + 11) ||                                          \
        difl[0] != (TYPE)((BASE) + 12) ||                                        \
        difr[0] != (TYPE)((BASE) + 13) ||                                        \
        dsigma[0] != (TYPE)((BASE) + 14) ||                                      \
        work[0] != (TYPE)((BASE) + 15)) {                                        \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference LASD8 inputs correctly\n"); \
        return 1;                                                                \
    }                                                                            \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences LASD8 inputs\n"); \
    return 0;                                                                    \
}

DEFINE_LASD8_TESTS(slasd8, float, FB_OP_SLASD8, 100)
DEFINE_LASD8_TESTS(dlasd8, double, FB_OP_DLASD8, 300)

int main(void)
{
    int status = 0;

    status |= check_slasd8_fortran_to_cblas();
    status |= check_slasd8_cblas_to_fortran();
    status |= check_dlasd8_fortran_to_cblas();
    status |= check_dlasd8_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}