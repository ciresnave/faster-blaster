#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

#define DEFINE_LASD7_TESTS(SUFFIX, TYPE, OP_ID, BASE)                             \
typedef int (*fb_##SUFFIX##_cblas_fn)(int icompq, int nl, int nr, int sqre,      \
                                      int *k, TYPE *d, TYPE *z, TYPE *zw,        \
                                      TYPE *vf, TYPE *vfw, TYPE *vl, TYPE *vlw,  \
                                      TYPE alpha, TYPE beta, TYPE *dsigma,       \
                                      int *idx, int *idxp, int *idxq, int *perm, \
                                      int *givptr, int *givcol, int ldgcol,      \
                                      TYPE *givnum, int ldgnum, TYPE *c,         \
                                      TYPE *s);                                  \
typedef void (*fb_##SUFFIX##_fortran_fn)(int *icompq, int *nl, int *nr, int *sqre,\
                                         int *k, TYPE *d, TYPE *z, TYPE *zw,     \
                                         TYPE *vf, TYPE *vfw, TYPE *vl, TYPE *vlw,\
                                         TYPE *alpha, TYPE *beta, TYPE *dsigma,  \
                                         int *idx, int *idxp, int *idxq,         \
                                         int *perm, int *givptr, int *givcol,    \
                                         int *ldgcol, TYPE *givnum, int *ldgnum, \
                                         TYPE *c, TYPE *s, int *info);           \
static struct {                                                                    \
    int called;                                                                    \
    int icompq;                                                                    \
    int nl;                                                                        \
    int nr;                                                                        \
    int sqre;                                                                      \
    int *k;                                                                        \
    TYPE *d;                                                                       \
    TYPE *z;                                                                       \
    TYPE *zw;                                                                      \
    TYPE *vf;                                                                      \
    TYPE *vfw;                                                                     \
    TYPE *vl;                                                                      \
    TYPE *vlw;                                                                     \
    TYPE alpha;                                                                    \
    TYPE beta;                                                                     \
    TYPE *dsigma;                                                                  \
    int *idx;                                                                      \
    int *idxp;                                                                     \
    int *idxq;                                                                     \
    int *perm;                                                                     \
    int *givptr;                                                                   \
    int *givcol;                                                                   \
    int ldgcol;                                                                    \
    TYPE *givnum;                                                                  \
    int ldgnum;                                                                    \
    TYPE *c;                                                                       \
    TYPE *s;                                                                       \
} g_##SUFFIX##_fortran_call;                                                       \
static struct {                                                                    \
    int called;                                                                    \
    int icompq;                                                                    \
    int nl;                                                                        \
    int nr;                                                                        \
    int sqre;                                                                      \
    int *k;                                                                        \
    TYPE *d;                                                                       \
    TYPE *z;                                                                       \
    TYPE *zw;                                                                      \
    TYPE *vf;                                                                      \
    TYPE *vfw;                                                                     \
    TYPE *vl;                                                                      \
    TYPE *vlw;                                                                     \
    TYPE alpha;                                                                    \
    TYPE beta;                                                                     \
    TYPE *dsigma;                                                                  \
    int *idx;                                                                      \
    int *idxp;                                                                     \
    int *idxq;                                                                     \
    int *perm;                                                                     \
    int *givptr;                                                                   \
    int *givcol;                                                                   \
    int ldgcol;                                                                    \
    TYPE *givnum;                                                                  \
    int ldgnum;                                                                    \
    TYPE *c;                                                                       \
    TYPE *s;                                                                       \
} g_##SUFFIX##_cblas_call;                                                         \
static void stub_##SUFFIX##_fortran(int *icompq, int *nl, int *nr, int *sqre,     \
                                    int *k, TYPE *d, TYPE *z, TYPE *zw,         \
                                    TYPE *vf, TYPE *vfw, TYPE *vl, TYPE *vlw,   \
                                    TYPE *alpha, TYPE *beta, TYPE *dsigma,      \
                                    int *idx, int *idxp, int *idxq, int *perm,  \
                                    int *givptr, int *givcol, int *ldgcol,      \
                                    TYPE *givnum, int *ldgnum, TYPE *c, TYPE *s,\
                                    int *info)                                   \
{                                                                                 \
    g_##SUFFIX##_fortran_call.called += 1;                                        \
    g_##SUFFIX##_fortran_call.icompq = *icompq;                                   \
    g_##SUFFIX##_fortran_call.nl = *nl;                                           \
    g_##SUFFIX##_fortran_call.nr = *nr;                                           \
    g_##SUFFIX##_fortran_call.sqre = *sqre;                                       \
    g_##SUFFIX##_fortran_call.k = k;                                              \
    g_##SUFFIX##_fortran_call.d = d;                                              \
    g_##SUFFIX##_fortran_call.z = z;                                              \
    g_##SUFFIX##_fortran_call.zw = zw;                                            \
    g_##SUFFIX##_fortran_call.vf = vf;                                            \
    g_##SUFFIX##_fortran_call.vfw = vfw;                                          \
    g_##SUFFIX##_fortran_call.vl = vl;                                            \
    g_##SUFFIX##_fortran_call.vlw = vlw;                                          \
    g_##SUFFIX##_fortran_call.alpha = *alpha;                                     \
    g_##SUFFIX##_fortran_call.beta = *beta;                                       \
    g_##SUFFIX##_fortran_call.dsigma = dsigma;                                    \
    g_##SUFFIX##_fortran_call.idx = idx;                                          \
    g_##SUFFIX##_fortran_call.idxp = idxp;                                        \
    g_##SUFFIX##_fortran_call.idxq = idxq;                                        \
    g_##SUFFIX##_fortran_call.perm = perm;                                        \
    g_##SUFFIX##_fortran_call.givptr = givptr;                                    \
    g_##SUFFIX##_fortran_call.givcol = givcol;                                    \
    g_##SUFFIX##_fortran_call.ldgcol = *ldgcol;                                   \
    g_##SUFFIX##_fortran_call.givnum = givnum;                                    \
    g_##SUFFIX##_fortran_call.ldgnum = *ldgnum;                                   \
    g_##SUFFIX##_fortran_call.c = c;                                              \
    g_##SUFFIX##_fortran_call.s = s;                                              \
    dsigma[0] = (TYPE)((BASE) + 1);                                               \
    *givptr = (BASE) + 2;                                                         \
    givcol[0] = (BASE) + 3;                                                       \
    givnum[0] = (TYPE)((BASE) + 4);                                               \
    *k = (BASE) + 5;                                                              \
    *c = (TYPE)((BASE) + 6);                                                      \
    *s = (TYPE)((BASE) + 7);                                                      \
    *info = (BASE) + 8;                                                           \
}                                                                                 \
static int stub_##SUFFIX##_cblas(int icompq, int nl, int nr, int sqre, int *k,   \
                                 TYPE *d, TYPE *z, TYPE *zw, TYPE *vf,          \
                                 TYPE *vfw, TYPE *vl, TYPE *vlw, TYPE alpha,    \
                                 TYPE beta, TYPE *dsigma, int *idx, int *idxp,  \
                                 int *idxq, int *perm, int *givptr,             \
                                 int *givcol, int ldgcol, TYPE *givnum,         \
                                 int ldgnum, TYPE *c, TYPE *s)                  \
{                                                                                 \
    g_##SUFFIX##_cblas_call.called += 1;                                          \
    g_##SUFFIX##_cblas_call.icompq = icompq;                                      \
    g_##SUFFIX##_cblas_call.nl = nl;                                              \
    g_##SUFFIX##_cblas_call.nr = nr;                                              \
    g_##SUFFIX##_cblas_call.sqre = sqre;                                          \
    g_##SUFFIX##_cblas_call.k = k;                                                \
    g_##SUFFIX##_cblas_call.d = d;                                                \
    g_##SUFFIX##_cblas_call.z = z;                                                \
    g_##SUFFIX##_cblas_call.zw = zw;                                              \
    g_##SUFFIX##_cblas_call.vf = vf;                                              \
    g_##SUFFIX##_cblas_call.vfw = vfw;                                            \
    g_##SUFFIX##_cblas_call.vl = vl;                                              \
    g_##SUFFIX##_cblas_call.vlw = vlw;                                            \
    g_##SUFFIX##_cblas_call.alpha = alpha;                                        \
    g_##SUFFIX##_cblas_call.beta = beta;                                          \
    g_##SUFFIX##_cblas_call.dsigma = dsigma;                                      \
    g_##SUFFIX##_cblas_call.idx = idx;                                            \
    g_##SUFFIX##_cblas_call.idxp = idxp;                                          \
    g_##SUFFIX##_cblas_call.idxq = idxq;                                          \
    g_##SUFFIX##_cblas_call.perm = perm;                                          \
    g_##SUFFIX##_cblas_call.givptr = givptr;                                      \
    g_##SUFFIX##_cblas_call.givcol = givcol;                                      \
    g_##SUFFIX##_cblas_call.ldgcol = ldgcol;                                      \
    g_##SUFFIX##_cblas_call.givnum = givnum;                                      \
    g_##SUFFIX##_cblas_call.ldgnum = ldgnum;                                      \
    g_##SUFFIX##_cblas_call.c = c;                                                \
    g_##SUFFIX##_cblas_call.s = s;                                                \
    dsigma[0] = (TYPE)((BASE) + 9);                                               \
    *givptr = (BASE) + 10;                                                        \
    givcol[0] = (BASE) + 11;                                                      \
    givnum[0] = (TYPE)((BASE) + 12);                                              \
    *k = (BASE) + 13;                                                             \
    *c = (TYPE)((BASE) + 14);                                                     \
    *s = (TYPE)((BASE) + 15);                                                     \
    return (BASE) + 16;                                                           \
}                                                                                 \
static int check_##SUFFIX##_fortran_to_cblas(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                          \
    int k = 0;                                                                    \
    TYPE d[3] = { (TYPE)1, (TYPE)2, (TYPE)3 };                                   \
    TYPE z[3] = { (TYPE)4, (TYPE)5, (TYPE)6 };                                   \
    TYPE zw[3] = { (TYPE)7, (TYPE)8, (TYPE)9 };                                  \
    TYPE vf[3] = { (TYPE)10, (TYPE)11, (TYPE)12 };                               \
    TYPE vfw[3] = { (TYPE)13, (TYPE)14, (TYPE)15 };                              \
    TYPE vl[3] = { (TYPE)16, (TYPE)17, (TYPE)18 };                               \
    TYPE vlw[3] = { (TYPE)19, (TYPE)20, (TYPE)21 };                              \
    TYPE dsigma[3] = { (TYPE)22, (TYPE)23, (TYPE)24 };                           \
    int idx[3] = { 25, 26, 27 };                                                  \
    int idxp[3] = { 28, 29, 30 };                                                 \
    int idxq[3] = { 31, 32, 33 };                                                 \
    int perm[3] = { 34, 35, 36 };                                                 \
    int givptr = 0;                                                               \
    int givcol[3] = { 37, 38, 39 };                                               \
    TYPE givnum[3] = { (TYPE)40, (TYPE)41, (TYPE)42 };                           \
    TYPE c = (TYPE)0;                                                             \
    TYPE s = (TYPE)0;                                                             \
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
    if (thunk(0, 1, 1, 0, &k, d, z, zw, vf, vfw, vl, vlw, (TYPE)43, (TYPE)44,    \
              dsigma, idx, idxp, idxq, perm, &givptr, givcol, 3, givnum, 3,      \
              &c, &s) != (BASE) + 8 ||                                            \
        g_##SUFFIX##_fortran_call.called != 1 ||                                  \
        g_##SUFFIX##_fortran_call.icompq != 0 ||                                  \
        g_##SUFFIX##_fortran_call.nl != 1 ||                                      \
        g_##SUFFIX##_fortran_call.nr != 1 ||                                      \
        g_##SUFFIX##_fortran_call.sqre != 0 ||                                    \
        g_##SUFFIX##_fortran_call.k != &k ||                                      \
        g_##SUFFIX##_fortran_call.d != d ||                                       \
        g_##SUFFIX##_fortran_call.z != z ||                                       \
        g_##SUFFIX##_fortran_call.zw != zw ||                                     \
        g_##SUFFIX##_fortran_call.vf != vf ||                                     \
        g_##SUFFIX##_fortran_call.vfw != vfw ||                                   \
        g_##SUFFIX##_fortran_call.vl != vl ||                                     \
        g_##SUFFIX##_fortran_call.vlw != vlw ||                                   \
        g_##SUFFIX##_fortran_call.alpha != (TYPE)43 ||                            \
        g_##SUFFIX##_fortran_call.beta != (TYPE)44 ||                             \
        g_##SUFFIX##_fortran_call.dsigma != dsigma ||                             \
        g_##SUFFIX##_fortran_call.idx != idx ||                                   \
        g_##SUFFIX##_fortran_call.idxp != idxp ||                                 \
        g_##SUFFIX##_fortran_call.idxq != idxq ||                                 \
        g_##SUFFIX##_fortran_call.perm != perm ||                                 \
        g_##SUFFIX##_fortran_call.givptr != &givptr ||                            \
        g_##SUFFIX##_fortran_call.givcol != givcol ||                             \
        g_##SUFFIX##_fortran_call.ldgcol != 3 ||                                  \
        g_##SUFFIX##_fortran_call.givnum != givnum ||                             \
        g_##SUFFIX##_fortran_call.ldgnum != 3 ||                                  \
        g_##SUFFIX##_fortran_call.c != &c ||                                      \
        g_##SUFFIX##_fortran_call.s != &s ||                                      \
        dsigma[0] != (TYPE)((BASE) + 1) ||                                        \
        givptr != (BASE) + 2 ||                                                   \
        givcol[0] != (BASE) + 3 ||                                                \
        givnum[0] != (TYPE)((BASE) + 4) ||                                        \
        k != (BASE) + 5 ||                                                        \
        c != (TYPE)((BASE) + 6) ||                                                \
        s != (TYPE)((BASE) + 7)) {                                                \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward LASD7 inputs correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards LASD7 inputs\n"); \
    return 0;                                                                     \
}                                                                                 \
static int check_##SUFFIX##_cblas_to_fortran(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                        \
    int icompq = 0;                                                               \
    int nl = 1;                                                                   \
    int nr = 1;                                                                   \
    int sqre = 0;                                                                 \
    int k = 0;                                                                    \
    TYPE d[3] = { (TYPE)45, (TYPE)46, (TYPE)47 };                                \
    TYPE z[3] = { (TYPE)48, (TYPE)49, (TYPE)50 };                                \
    TYPE zw[3] = { (TYPE)51, (TYPE)52, (TYPE)53 };                               \
    TYPE vf[3] = { (TYPE)54, (TYPE)55, (TYPE)56 };                               \
    TYPE vfw[3] = { (TYPE)57, (TYPE)58, (TYPE)59 };                              \
    TYPE vl[3] = { (TYPE)60, (TYPE)61, (TYPE)62 };                               \
    TYPE vlw[3] = { (TYPE)63, (TYPE)64, (TYPE)65 };                              \
    TYPE dsigma[3] = { (TYPE)66, (TYPE)67, (TYPE)68 };                           \
    int idx[3] = { 69, 70, 71 };                                                  \
    int idxp[3] = { 72, 73, 74 };                                                 \
    int idxq[3] = { 75, 76, 77 };                                                 \
    int perm[3] = { 78, 79, 80 };                                                 \
    int givptr = 0;                                                               \
    int givcol[3] = { 81, 82, 83 };                                               \
    int ldgcol = 3;                                                               \
    TYPE givnum[3] = { (TYPE)84, (TYPE)85, (TYPE)86 };                           \
    int ldgnum = 3;                                                               \
    TYPE c = (TYPE)0;                                                             \
    TYPE s = (TYPE)0;                                                             \
    int info = -1;                                                                \
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
    thunk(&icompq, &nl, &nr, &sqre, &k, d, z, zw, vf, vfw, vl, vlw,              \
          &(TYPE){43}, &(TYPE){44}, dsigma, idx, idxp, idxq, perm, &givptr,      \
          givcol, &ldgcol, givnum, &ldgnum, &c, &s, &info);                      \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                    \
        g_##SUFFIX##_cblas_call.icompq != 0 ||                                    \
        g_##SUFFIX##_cblas_call.nl != 1 ||                                        \
        g_##SUFFIX##_cblas_call.nr != 1 ||                                        \
        g_##SUFFIX##_cblas_call.sqre != 0 ||                                      \
        g_##SUFFIX##_cblas_call.k != &k ||                                        \
        g_##SUFFIX##_cblas_call.d != d ||                                         \
        g_##SUFFIX##_cblas_call.z != z ||                                         \
        g_##SUFFIX##_cblas_call.zw != zw ||                                       \
        g_##SUFFIX##_cblas_call.vf != vf ||                                       \
        g_##SUFFIX##_cblas_call.vfw != vfw ||                                     \
        g_##SUFFIX##_cblas_call.vl != vl ||                                       \
        g_##SUFFIX##_cblas_call.vlw != vlw ||                                     \
        g_##SUFFIX##_cblas_call.alpha != (TYPE)43 ||                              \
        g_##SUFFIX##_cblas_call.beta != (TYPE)44 ||                               \
        g_##SUFFIX##_cblas_call.dsigma != dsigma ||                               \
        g_##SUFFIX##_cblas_call.idx != idx ||                                     \
        g_##SUFFIX##_cblas_call.idxp != idxp ||                                   \
        g_##SUFFIX##_cblas_call.idxq != idxq ||                                   \
        g_##SUFFIX##_cblas_call.perm != perm ||                                   \
        g_##SUFFIX##_cblas_call.givptr != &givptr ||                              \
        g_##SUFFIX##_cblas_call.givcol != givcol ||                               \
        g_##SUFFIX##_cblas_call.ldgcol != 3 ||                                    \
        g_##SUFFIX##_cblas_call.givnum != givnum ||                               \
        g_##SUFFIX##_cblas_call.ldgnum != 3 ||                                    \
        g_##SUFFIX##_cblas_call.c != &c ||                                        \
        g_##SUFFIX##_cblas_call.s != &s ||                                        \
        info != (BASE) + 16 ||                                                    \
        dsigma[0] != (TYPE)((BASE) + 9) ||                                        \
        givptr != (BASE) + 10 ||                                                  \
        givcol[0] != (BASE) + 11 ||                                               \
        givnum[0] != (TYPE)((BASE) + 12) ||                                       \
        k != (BASE) + 13 ||                                                       \
        c != (TYPE)((BASE) + 14) ||                                               \
        s != (TYPE)((BASE) + 15)) {                                               \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference LASD7 inputs correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences LASD7 inputs\n"); \
    return 0;                                                                     \
}

DEFINE_LASD7_TESTS(slasd7, float, FB_OP_SLASD7, 100)
DEFINE_LASD7_TESTS(dlasd7, double, FB_OP_DLASD7, 300)

int main(void)
{
    int status = 0;

    status |= check_slasd7_fortran_to_cblas();
    status |= check_slasd7_cblas_to_fortran();
    status |= check_dlasd7_fortran_to_cblas();
    status |= check_dlasd7_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}