#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

#define DEFINE_LASD3_TESTS(SUFFIX, TYPE, OP_ID, BASE)                           \
typedef int (*fb_##SUFFIX##_cblas_fn)(int nl, int nr, int sqre, int k, TYPE *d, \
                                      TYPE *q, int ldq, TYPE *dsigma, TYPE *u,  \
                                      int ldu, TYPE *u2, int ldu2, TYPE *vt,    \
                                      int ldvt, TYPE *vt2, int ldvt2, int *idxc,\
                                      int *ctot, TYPE *z);                      \
typedef void (*fb_##SUFFIX##_fortran_fn)(int *nl, int *nr, int *sqre, int *k,  \
                                         TYPE *d, TYPE *q, int *ldq, TYPE *dsigma,\
                                         TYPE *u, int *ldu, TYPE *u2, int *ldu2,\
                                         TYPE *vt, int *ldvt, TYPE *vt2,        \
                                         int *ldvt2, int *idxc, int *ctot,      \
                                         TYPE *z, int *info);                   \
static struct {                                                                  \
    int called;                                                                  \
    int nl;                                                                      \
    int nr;                                                                      \
    int sqre;                                                                    \
    int k;                                                                       \
    TYPE *d;                                                                     \
    TYPE *q;                                                                     \
    int ldq;                                                                     \
    TYPE *dsigma;                                                                \
    TYPE *u;                                                                     \
    int ldu;                                                                     \
    TYPE *u2;                                                                    \
    int ldu2;                                                                    \
    TYPE *vt;                                                                    \
    int ldvt;                                                                    \
    TYPE *vt2;                                                                   \
    int ldvt2;                                                                   \
    int *idxc;                                                                   \
    int *ctot;                                                                   \
    TYPE *z;                                                                     \
} g_##SUFFIX##_fortran_call;                                                     \
static struct {                                                                  \
    int called;                                                                  \
    int nl;                                                                      \
    int nr;                                                                      \
    int sqre;                                                                    \
    int k;                                                                       \
    TYPE *d;                                                                     \
    TYPE *q;                                                                     \
    int ldq;                                                                     \
    TYPE *dsigma;                                                                \
    TYPE *u;                                                                     \
    int ldu;                                                                     \
    TYPE *u2;                                                                    \
    int ldu2;                                                                    \
    TYPE *vt;                                                                    \
    int ldvt;                                                                    \
    TYPE *vt2;                                                                   \
    int ldvt2;                                                                   \
    int *idxc;                                                                   \
    int *ctot;                                                                   \
    TYPE *z;                                                                     \
} g_##SUFFIX##_cblas_call;                                                       \
static void stub_##SUFFIX##_fortran(int *nl, int *nr, int *sqre, int *k,        \
                                    TYPE *d, TYPE *q, int *ldq, TYPE *dsigma,   \
                                    TYPE *u, int *ldu, TYPE *u2, int *ldu2,     \
                                    TYPE *vt, int *ldvt, TYPE *vt2, int *ldvt2, \
                                    int *idxc, int *ctot, TYPE *z, int *info)   \
{                                                                                \
    g_##SUFFIX##_fortran_call.called += 1;                                       \
    g_##SUFFIX##_fortran_call.nl = *nl;                                          \
    g_##SUFFIX##_fortran_call.nr = *nr;                                          \
    g_##SUFFIX##_fortran_call.sqre = *sqre;                                      \
    g_##SUFFIX##_fortran_call.k = *k;                                            \
    g_##SUFFIX##_fortran_call.d = d;                                             \
    g_##SUFFIX##_fortran_call.q = q;                                             \
    g_##SUFFIX##_fortran_call.ldq = *ldq;                                        \
    g_##SUFFIX##_fortran_call.dsigma = dsigma;                                   \
    g_##SUFFIX##_fortran_call.u = u;                                             \
    g_##SUFFIX##_fortran_call.ldu = *ldu;                                        \
    g_##SUFFIX##_fortran_call.u2 = u2;                                           \
    g_##SUFFIX##_fortran_call.ldu2 = *ldu2;                                      \
    g_##SUFFIX##_fortran_call.vt = vt;                                           \
    g_##SUFFIX##_fortran_call.ldvt = *ldvt;                                      \
    g_##SUFFIX##_fortran_call.vt2 = vt2;                                         \
    g_##SUFFIX##_fortran_call.ldvt2 = *ldvt2;                                    \
    g_##SUFFIX##_fortran_call.idxc = idxc;                                       \
    g_##SUFFIX##_fortran_call.ctot = ctot;                                       \
    g_##SUFFIX##_fortran_call.z = z;                                             \
    d[0] = (TYPE)((BASE) + 1);                                                   \
    u[0] = (TYPE)((BASE) + 2);                                                   \
    *info = (BASE) + 3;                                                          \
}                                                                                \
static int stub_##SUFFIX##_cblas(int nl, int nr, int sqre, int k, TYPE *d,      \
                                 TYPE *q, int ldq, TYPE *dsigma, TYPE *u,       \
                                 int ldu, TYPE *u2, int ldu2, TYPE *vt,         \
                                 int ldvt, TYPE *vt2, int ldvt2, int *idxc,     \
                                 int *ctot, TYPE *z)                            \
{                                                                                \
    g_##SUFFIX##_cblas_call.called += 1;                                         \
    g_##SUFFIX##_cblas_call.nl = nl;                                             \
    g_##SUFFIX##_cblas_call.nr = nr;                                             \
    g_##SUFFIX##_cblas_call.sqre = sqre;                                         \
    g_##SUFFIX##_cblas_call.k = k;                                               \
    g_##SUFFIX##_cblas_call.d = d;                                               \
    g_##SUFFIX##_cblas_call.q = q;                                               \
    g_##SUFFIX##_cblas_call.ldq = ldq;                                           \
    g_##SUFFIX##_cblas_call.dsigma = dsigma;                                     \
    g_##SUFFIX##_cblas_call.u = u;                                               \
    g_##SUFFIX##_cblas_call.ldu = ldu;                                           \
    g_##SUFFIX##_cblas_call.u2 = u2;                                             \
    g_##SUFFIX##_cblas_call.ldu2 = ldu2;                                         \
    g_##SUFFIX##_cblas_call.vt = vt;                                             \
    g_##SUFFIX##_cblas_call.ldvt = ldvt;                                         \
    g_##SUFFIX##_cblas_call.vt2 = vt2;                                           \
    g_##SUFFIX##_cblas_call.ldvt2 = ldvt2;                                       \
    g_##SUFFIX##_cblas_call.idxc = idxc;                                         \
    g_##SUFFIX##_cblas_call.ctot = ctot;                                         \
    g_##SUFFIX##_cblas_call.z = z;                                               \
    d[0] = (TYPE)((BASE) + 4);                                                   \
    u[0] = (TYPE)((BASE) + 5);                                                   \
    return (BASE) + 6;                                                           \
}                                                                                \
static int check_##SUFFIX##_fortran_to_cblas(void)                               \
{                                                                                \
    fb_backend_vtable_t vtable;                                                  \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                         \
    TYPE d[4] = { (TYPE)1, (TYPE)2, (TYPE)3, (TYPE)4 };                         \
    TYPE q[4] = { (TYPE)5, (TYPE)6, (TYPE)7, (TYPE)8 };                         \
    TYPE dsigma[4] = { (TYPE)9, (TYPE)10, (TYPE)11, (TYPE)12 };                 \
    TYPE u[4] = { (TYPE)13, (TYPE)14, (TYPE)15, (TYPE)16 };                     \
    TYPE u2[4] = { (TYPE)17, (TYPE)18, (TYPE)19, (TYPE)20 };                    \
    TYPE vt[4] = { (TYPE)21, (TYPE)22, (TYPE)23, (TYPE)24 };                    \
    TYPE vt2[4] = { (TYPE)25, (TYPE)26, (TYPE)27, (TYPE)28 };                   \
    int idxc[4] = { 29, 30, 31, 32 };                                            \
    int ctot[4] = { 33, 34, 35, 36 };                                            \
    TYPE z[4] = { (TYPE)37, (TYPE)38, (TYPE)39, (TYPE)40 };                     \
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
    if (thunk(1, 0, 0, 2, d, q, 2, dsigma, u, 2, u2, 2, vt, 2, vt2, 2, idxc,    \
              ctot, z) != (BASE) + 3 ||                                          \
        g_##SUFFIX##_fortran_call.called != 1 ||                                 \
        g_##SUFFIX##_fortran_call.nl != 1 ||                                     \
        g_##SUFFIX##_fortran_call.nr != 0 ||                                     \
        g_##SUFFIX##_fortran_call.sqre != 0 ||                                   \
        g_##SUFFIX##_fortran_call.k != 2 ||                                      \
        g_##SUFFIX##_fortran_call.d != d ||                                      \
        g_##SUFFIX##_fortran_call.q != q ||                                      \
        g_##SUFFIX##_fortran_call.ldq != 2 ||                                    \
        g_##SUFFIX##_fortran_call.dsigma != dsigma ||                            \
        g_##SUFFIX##_fortran_call.u != u ||                                      \
        g_##SUFFIX##_fortran_call.ldu != 2 ||                                    \
        g_##SUFFIX##_fortran_call.u2 != u2 ||                                    \
        g_##SUFFIX##_fortran_call.ldu2 != 2 ||                                   \
        g_##SUFFIX##_fortran_call.vt != vt ||                                    \
        g_##SUFFIX##_fortran_call.ldvt != 2 ||                                   \
        g_##SUFFIX##_fortran_call.vt2 != vt2 ||                                  \
        g_##SUFFIX##_fortran_call.ldvt2 != 2 ||                                  \
        g_##SUFFIX##_fortran_call.idxc != idxc ||                                \
        g_##SUFFIX##_fortran_call.ctot != ctot ||                                \
        g_##SUFFIX##_fortran_call.z != z ||                                      \
        d[0] != (TYPE)((BASE) + 1) ||                                            \
        u[0] != (TYPE)((BASE) + 2)) {                                            \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward LASD3 inputs correctly\n"); \
        return 1;                                                                \
    }                                                                            \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards LASD3 inputs\n"); \
    return 0;                                                                    \
}                                                                                \
static int check_##SUFFIX##_cblas_to_fortran(void)                               \
{                                                                                \
    fb_backend_vtable_t vtable;                                                  \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                       \
    int nl = 1;                                                                  \
    int nr = 0;                                                                  \
    int sqre = 0;                                                                \
    int k = 2;                                                                   \
    int ldq = 2;                                                                 \
    int ldu = 2;                                                                 \
    int ldu2 = 2;                                                                \
    int ldvt = 2;                                                                \
    int ldvt2 = 2;                                                               \
    int info = -1;                                                               \
    TYPE d[4] = { (TYPE)41, (TYPE)42, (TYPE)43, (TYPE)44 };                     \
    TYPE q[4] = { (TYPE)45, (TYPE)46, (TYPE)47, (TYPE)48 };                     \
    TYPE dsigma[4] = { (TYPE)49, (TYPE)50, (TYPE)51, (TYPE)52 };                \
    TYPE u[4] = { (TYPE)53, (TYPE)54, (TYPE)55, (TYPE)56 };                     \
    TYPE u2[4] = { (TYPE)57, (TYPE)58, (TYPE)59, (TYPE)60 };                    \
    TYPE vt[4] = { (TYPE)61, (TYPE)62, (TYPE)63, (TYPE)64 };                    \
    TYPE vt2[4] = { (TYPE)65, (TYPE)66, (TYPE)67, (TYPE)68 };                   \
    int idxc[4] = { 69, 70, 71, 72 };                                            \
    int ctot[4] = { 73, 74, 75, 76 };                                            \
    TYPE z[4] = { (TYPE)77, (TYPE)78, (TYPE)79, (TYPE)80 };                     \
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
    thunk(&nl, &nr, &sqre, &k, d, q, &ldq, dsigma, u, &ldu, u2, &ldu2, vt,      \
          &ldvt, vt2, &ldvt2, idxc, ctot, z, &info);                            \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                   \
        g_##SUFFIX##_cblas_call.nl != 1 ||                                       \
        g_##SUFFIX##_cblas_call.nr != 0 ||                                       \
        g_##SUFFIX##_cblas_call.sqre != 0 ||                                     \
        g_##SUFFIX##_cblas_call.k != 2 ||                                        \
        g_##SUFFIX##_cblas_call.d != d ||                                        \
        g_##SUFFIX##_cblas_call.q != q ||                                        \
        g_##SUFFIX##_cblas_call.ldq != 2 ||                                      \
        g_##SUFFIX##_cblas_call.dsigma != dsigma ||                              \
        g_##SUFFIX##_cblas_call.u != u ||                                        \
        g_##SUFFIX##_cblas_call.ldu != 2 ||                                      \
        g_##SUFFIX##_cblas_call.u2 != u2 ||                                      \
        g_##SUFFIX##_cblas_call.ldu2 != 2 ||                                     \
        g_##SUFFIX##_cblas_call.vt != vt ||                                      \
        g_##SUFFIX##_cblas_call.ldvt != 2 ||                                     \
        g_##SUFFIX##_cblas_call.vt2 != vt2 ||                                    \
        g_##SUFFIX##_cblas_call.ldvt2 != 2 ||                                    \
        g_##SUFFIX##_cblas_call.idxc != idxc ||                                  \
        g_##SUFFIX##_cblas_call.ctot != ctot ||                                  \
        g_##SUFFIX##_cblas_call.z != z ||                                        \
        info != (BASE) + 6 ||                                                    \
        d[0] != (TYPE)((BASE) + 4) ||                                            \
        u[0] != (TYPE)((BASE) + 5)) {                                            \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference LASD3 inputs correctly\n"); \
        return 1;                                                                \
    }                                                                            \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences LASD3 inputs\n"); \
    return 0;                                                                    \
}

DEFINE_LASD3_TESTS(slasd3, float, FB_OP_SLASD3, 100)
DEFINE_LASD3_TESTS(dlasd3, double, FB_OP_DLASD3, 300)

int main(void)
{
    int status = 0;

    status |= check_slasd3_fortran_to_cblas();
    status |= check_slasd3_cblas_to_fortran();
    status |= check_dlasd3_fortran_to_cblas();
    status |= check_dlasd3_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}