#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

#define DEFINE_LASD2_TESTS(SUFFIX, TYPE, OP_ID, BASE)                           \
typedef int (*fb_##SUFFIX##_cblas_fn)(int nl, int nr, int sqre, int *k, TYPE *d,\
                                      TYPE *z, TYPE alpha, TYPE beta, TYPE *u,  \
                                      int ldu, TYPE *vt, int ldvt, TYPE *dsigma,\
                                      TYPE *u2, int ldu2, TYPE *vt2, int ldvt2, \
                                      int *idxp, int *idx, int *idxc, int *idxq,\
                                      int *coltyp);                             \
typedef void (*fb_##SUFFIX##_fortran_fn)(int *nl, int *nr, int *sqre, int *k,  \
                                         TYPE *d, TYPE *z, TYPE *alpha,         \
                                         TYPE *beta, TYPE *u, int *ldu,         \
                                         TYPE *vt, int *ldvt, TYPE *dsigma,     \
                                         TYPE *u2, int *ldu2, TYPE *vt2,        \
                                         int *ldvt2, int *idxp, int *idx,       \
                                         int *idxc, int *idxq, int *coltyp,     \
                                         int *info);                            \
static struct {                                                                  \
    int called;                                                                  \
    int nl;                                                                      \
    int nr;                                                                      \
    int sqre;                                                                    \
    int *k;                                                                      \
    TYPE *d;                                                                     \
    TYPE *z;                                                                     \
    TYPE alpha;                                                                  \
    TYPE beta;                                                                   \
    TYPE *u;                                                                     \
    int ldu;                                                                     \
    TYPE *vt;                                                                    \
    int ldvt;                                                                    \
    TYPE *dsigma;                                                                \
    TYPE *u2;                                                                    \
    int ldu2;                                                                    \
    TYPE *vt2;                                                                   \
    int ldvt2;                                                                   \
    int *idxp;                                                                   \
    int *idx;                                                                    \
    int *idxc;                                                                   \
    int *idxq;                                                                   \
    int *coltyp;                                                                 \
} g_##SUFFIX##_fortran_call;                                                     \
static struct {                                                                  \
    int called;                                                                  \
    int nl;                                                                      \
    int nr;                                                                      \
    int sqre;                                                                    \
    int *k;                                                                      \
    TYPE *d;                                                                     \
    TYPE *z;                                                                     \
    TYPE alpha;                                                                  \
    TYPE beta;                                                                   \
    TYPE *u;                                                                     \
    int ldu;                                                                     \
    TYPE *vt;                                                                    \
    int ldvt;                                                                    \
    TYPE *dsigma;                                                                \
    TYPE *u2;                                                                    \
    int ldu2;                                                                    \
    TYPE *vt2;                                                                   \
    int ldvt2;                                                                   \
    int *idxp;                                                                   \
    int *idx;                                                                    \
    int *idxc;                                                                   \
    int *idxq;                                                                   \
    int *coltyp;                                                                 \
} g_##SUFFIX##_cblas_call;                                                       \
static void stub_##SUFFIX##_fortran(int *nl, int *nr, int *sqre, int *k,        \
                                    TYPE *d, TYPE *z, TYPE *alpha, TYPE *beta,  \
                                    TYPE *u, int *ldu, TYPE *vt, int *ldvt,     \
                                    TYPE *dsigma, TYPE *u2, int *ldu2,          \
                                    TYPE *vt2, int *ldvt2, int *idxp, int *idx, \
                                    int *idxc, int *idxq, int *coltyp,          \
                                    int *info)                                  \
{                                                                                \
    g_##SUFFIX##_fortran_call.called += 1;                                       \
    g_##SUFFIX##_fortran_call.nl = *nl;                                          \
    g_##SUFFIX##_fortran_call.nr = *nr;                                          \
    g_##SUFFIX##_fortran_call.sqre = *sqre;                                      \
    g_##SUFFIX##_fortran_call.k = k;                                             \
    g_##SUFFIX##_fortran_call.d = d;                                             \
    g_##SUFFIX##_fortran_call.z = z;                                             \
    g_##SUFFIX##_fortran_call.alpha = *alpha;                                    \
    g_##SUFFIX##_fortran_call.beta = *beta;                                      \
    g_##SUFFIX##_fortran_call.u = u;                                             \
    g_##SUFFIX##_fortran_call.ldu = *ldu;                                        \
    g_##SUFFIX##_fortran_call.vt = vt;                                           \
    g_##SUFFIX##_fortran_call.ldvt = *ldvt;                                      \
    g_##SUFFIX##_fortran_call.dsigma = dsigma;                                   \
    g_##SUFFIX##_fortran_call.u2 = u2;                                           \
    g_##SUFFIX##_fortran_call.ldu2 = *ldu2;                                      \
    g_##SUFFIX##_fortran_call.vt2 = vt2;                                         \
    g_##SUFFIX##_fortran_call.ldvt2 = *ldvt2;                                    \
    g_##SUFFIX##_fortran_call.idxp = idxp;                                       \
    g_##SUFFIX##_fortran_call.idx = idx;                                         \
    g_##SUFFIX##_fortran_call.idxc = idxc;                                       \
    g_##SUFFIX##_fortran_call.idxq = idxq;                                       \
    g_##SUFFIX##_fortran_call.coltyp = coltyp;                                   \
    *k = (BASE) + 1;                                                             \
    d[0] = (TYPE)((BASE) + 2);                                                   \
    dsigma[0] = (TYPE)((BASE) + 3);                                              \
    idx[0] = (BASE) + 4;                                                         \
    coltyp[0] = (BASE) + 5;                                                      \
    *info = (BASE) + 6;                                                          \
}                                                                                \
static int stub_##SUFFIX##_cblas(int nl, int nr, int sqre, int *k, TYPE *d,     \
                                 TYPE *z, TYPE alpha, TYPE beta, TYPE *u,       \
                                 int ldu, TYPE *vt, int ldvt, TYPE *dsigma,     \
                                 TYPE *u2, int ldu2, TYPE *vt2, int ldvt2,      \
                                 int *idxp, int *idx, int *idxc, int *idxq,     \
                                 int *coltyp)                                   \
{                                                                                \
    g_##SUFFIX##_cblas_call.called += 1;                                         \
    g_##SUFFIX##_cblas_call.nl = nl;                                             \
    g_##SUFFIX##_cblas_call.nr = nr;                                             \
    g_##SUFFIX##_cblas_call.sqre = sqre;                                         \
    g_##SUFFIX##_cblas_call.k = k;                                               \
    g_##SUFFIX##_cblas_call.d = d;                                               \
    g_##SUFFIX##_cblas_call.z = z;                                               \
    g_##SUFFIX##_cblas_call.alpha = alpha;                                       \
    g_##SUFFIX##_cblas_call.beta = beta;                                         \
    g_##SUFFIX##_cblas_call.u = u;                                               \
    g_##SUFFIX##_cblas_call.ldu = ldu;                                           \
    g_##SUFFIX##_cblas_call.vt = vt;                                             \
    g_##SUFFIX##_cblas_call.ldvt = ldvt;                                         \
    g_##SUFFIX##_cblas_call.dsigma = dsigma;                                     \
    g_##SUFFIX##_cblas_call.u2 = u2;                                             \
    g_##SUFFIX##_cblas_call.ldu2 = ldu2;                                         \
    g_##SUFFIX##_cblas_call.vt2 = vt2;                                           \
    g_##SUFFIX##_cblas_call.ldvt2 = ldvt2;                                       \
    g_##SUFFIX##_cblas_call.idxp = idxp;                                         \
    g_##SUFFIX##_cblas_call.idx = idx;                                           \
    g_##SUFFIX##_cblas_call.idxc = idxc;                                         \
    g_##SUFFIX##_cblas_call.idxq = idxq;                                         \
    g_##SUFFIX##_cblas_call.coltyp = coltyp;                                     \
    *k = (BASE) + 7;                                                             \
    d[0] = (TYPE)((BASE) + 8);                                                   \
    dsigma[0] = (TYPE)((BASE) + 9);                                              \
    idx[0] = (BASE) + 10;                                                        \
    coltyp[0] = (BASE) + 11;                                                     \
    return (BASE) + 12;                                                          \
}                                                                                \
static int check_##SUFFIX##_fortran_to_cblas(void)                               \
{                                                                                \
    fb_backend_vtable_t vtable;                                                  \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                         \
    int k = 0;                                                                   \
    TYPE d[4] = { (TYPE)1, (TYPE)2, (TYPE)3, (TYPE)4 };                         \
    TYPE z[4] = { (TYPE)5, (TYPE)6, (TYPE)7, (TYPE)8 };                         \
    TYPE u[4] = { (TYPE)9, (TYPE)10, (TYPE)11, (TYPE)12 };                      \
    TYPE vt[4] = { (TYPE)13, (TYPE)14, (TYPE)15, (TYPE)16 };                    \
    TYPE dsigma[4] = { (TYPE)17, (TYPE)18, (TYPE)19, (TYPE)20 };                \
    TYPE u2[4] = { (TYPE)21, (TYPE)22, (TYPE)23, (TYPE)24 };                    \
    TYPE vt2[4] = { (TYPE)25, (TYPE)26, (TYPE)27, (TYPE)28 };                   \
    int idxp[4] = { 29, 30, 31, 32 };                                            \
    int idx[4] = { 33, 34, 35, 36 };                                             \
    int idxc[4] = { 37, 38, 39, 40 };                                            \
    int idxq[4] = { 41, 42, 43, 44 };                                            \
    int coltyp[4] = { 45, 46, 47, 48 };                                          \
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
    if (thunk(1, 1, 0, &k, d, z, (TYPE)49, (TYPE)50, u, 2, vt, 2, dsigma, u2,   \
              2, vt2, 2, idxp, idx, idxc, idxq, coltyp) != (BASE) + 6 ||        \
        g_##SUFFIX##_fortran_call.called != 1 ||                                 \
        g_##SUFFIX##_fortran_call.nl != 1 ||                                     \
        g_##SUFFIX##_fortran_call.nr != 1 ||                                     \
        g_##SUFFIX##_fortran_call.sqre != 0 ||                                   \
        g_##SUFFIX##_fortran_call.k != &k ||                                     \
        g_##SUFFIX##_fortran_call.d != d ||                                      \
        g_##SUFFIX##_fortran_call.z != z ||                                      \
        g_##SUFFIX##_fortran_call.alpha != (TYPE)49 ||                           \
        g_##SUFFIX##_fortran_call.beta != (TYPE)50 ||                            \
        g_##SUFFIX##_fortran_call.u != u ||                                      \
        g_##SUFFIX##_fortran_call.ldu != 2 ||                                    \
        g_##SUFFIX##_fortran_call.vt != vt ||                                    \
        g_##SUFFIX##_fortran_call.ldvt != 2 ||                                   \
        g_##SUFFIX##_fortran_call.dsigma != dsigma ||                            \
        g_##SUFFIX##_fortran_call.u2 != u2 ||                                    \
        g_##SUFFIX##_fortran_call.ldu2 != 2 ||                                   \
        g_##SUFFIX##_fortran_call.vt2 != vt2 ||                                  \
        g_##SUFFIX##_fortran_call.ldvt2 != 2 ||                                  \
        g_##SUFFIX##_fortran_call.idxp != idxp ||                                \
        g_##SUFFIX##_fortran_call.idx != idx ||                                  \
        g_##SUFFIX##_fortran_call.idxc != idxc ||                                \
        g_##SUFFIX##_fortran_call.idxq != idxq ||                                \
        g_##SUFFIX##_fortran_call.coltyp != coltyp ||                            \
        k != (BASE) + 1 ||                                                       \
        d[0] != (TYPE)((BASE) + 2) ||                                            \
        dsigma[0] != (TYPE)((BASE) + 3) ||                                       \
        idx[0] != (BASE) + 4 ||                                                  \
        coltyp[0] != (BASE) + 5) {                                               \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward LASD2 inputs correctly\n"); \
        return 1;                                                                \
    }                                                                            \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards LASD2 inputs\n"); \
    return 0;                                                                    \
}                                                                                \
static int check_##SUFFIX##_cblas_to_fortran(void)                               \
{                                                                                \
    fb_backend_vtable_t vtable;                                                  \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                       \
    int nl = 1;                                                                  \
    int nr = 1;                                                                  \
    int sqre = 0;                                                                \
    int k = 0;                                                                   \
    TYPE alpha = (TYPE)49;                                                       \
    TYPE beta = (TYPE)50;                                                        \
    int ldu = 2;                                                                 \
    int ldvt = 2;                                                                \
    int ldu2 = 2;                                                                \
    int ldvt2 = 2;                                                               \
    int info = -1;                                                               \
    TYPE d[4] = { (TYPE)51, (TYPE)52, (TYPE)53, (TYPE)54 };                     \
    TYPE z[4] = { (TYPE)55, (TYPE)56, (TYPE)57, (TYPE)58 };                     \
    TYPE u[4] = { (TYPE)59, (TYPE)60, (TYPE)61, (TYPE)62 };                     \
    TYPE vt[4] = { (TYPE)63, (TYPE)64, (TYPE)65, (TYPE)66 };                    \
    TYPE dsigma[4] = { (TYPE)67, (TYPE)68, (TYPE)69, (TYPE)70 };                \
    TYPE u2[4] = { (TYPE)71, (TYPE)72, (TYPE)73, (TYPE)74 };                    \
    TYPE vt2[4] = { (TYPE)75, (TYPE)76, (TYPE)77, (TYPE)78 };                   \
    int idxp[4] = { 79, 80, 81, 82 };                                            \
    int idx[4] = { 83, 84, 85, 86 };                                             \
    int idxc[4] = { 87, 88, 89, 90 };                                            \
    int idxq[4] = { 91, 92, 93, 94 };                                            \
    int coltyp[4] = { 95, 96, 97, 98 };                                          \
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
    thunk(&nl, &nr, &sqre, &k, d, z, &alpha, &beta, u, &ldu, vt, &ldvt, dsigma, \
          u2, &ldu2, vt2, &ldvt2, idxp, idx, idxc, idxq, coltyp, &info);        \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                   \
        g_##SUFFIX##_cblas_call.nl != 1 ||                                       \
        g_##SUFFIX##_cblas_call.nr != 1 ||                                       \
        g_##SUFFIX##_cblas_call.sqre != 0 ||                                     \
        g_##SUFFIX##_cblas_call.k != &k ||                                       \
        g_##SUFFIX##_cblas_call.d != d ||                                        \
        g_##SUFFIX##_cblas_call.z != z ||                                        \
        g_##SUFFIX##_cblas_call.alpha != (TYPE)49 ||                             \
        g_##SUFFIX##_cblas_call.beta != (TYPE)50 ||                              \
        g_##SUFFIX##_cblas_call.u != u ||                                        \
        g_##SUFFIX##_cblas_call.ldu != 2 ||                                      \
        g_##SUFFIX##_cblas_call.vt != vt ||                                      \
        g_##SUFFIX##_cblas_call.ldvt != 2 ||                                     \
        g_##SUFFIX##_cblas_call.dsigma != dsigma ||                              \
        g_##SUFFIX##_cblas_call.u2 != u2 ||                                      \
        g_##SUFFIX##_cblas_call.ldu2 != 2 ||                                     \
        g_##SUFFIX##_cblas_call.vt2 != vt2 ||                                    \
        g_##SUFFIX##_cblas_call.ldvt2 != 2 ||                                    \
        g_##SUFFIX##_cblas_call.idxp != idxp ||                                  \
        g_##SUFFIX##_cblas_call.idx != idx ||                                    \
        g_##SUFFIX##_cblas_call.idxc != idxc ||                                  \
        g_##SUFFIX##_cblas_call.idxq != idxq ||                                  \
        g_##SUFFIX##_cblas_call.coltyp != coltyp ||                              \
        info != (BASE) + 12 ||                                                   \
        k != (BASE) + 7 ||                                                       \
        d[0] != (TYPE)((BASE) + 8) ||                                            \
        dsigma[0] != (TYPE)((BASE) + 9) ||                                       \
        idx[0] != (BASE) + 10 ||                                                 \
        coltyp[0] != (BASE) + 11) {                                              \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference LASD2 inputs correctly\n"); \
        return 1;                                                                \
    }                                                                            \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences LASD2 inputs\n"); \
    return 0;                                                                    \
}

DEFINE_LASD2_TESTS(slasd2, float, FB_OP_SLASD2, 100)
DEFINE_LASD2_TESTS(dlasd2, double, FB_OP_DLASD2, 300)

int main(void)
{
    int status = 0;

    status |= check_slasd2_fortran_to_cblas();
    status |= check_slasd2_cblas_to_fortran();
    status |= check_dlasd2_fortran_to_cblas();
    status |= check_dlasd2_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}