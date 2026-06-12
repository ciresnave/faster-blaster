#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

#define DEFINE_LASD6_TESTS(SUFFIX, TYPE, OP_ID, BASE)                            \
typedef int (*fb_##SUFFIX##_cblas_fn)(int icompq, int nl, int nr, int sqre,     \
                                      TYPE *d, TYPE *vf, TYPE *vl, TYPE *alpha, \
                                      TYPE *beta, int *idxq, int *perm,          \
                                      int *givptr, int *givcol, int ldgcol,      \
                                      TYPE *givnum, int ldgnum, TYPE *poles,     \
                                      TYPE *difl, TYPE *difr, TYPE *z, int *k,   \
                                      TYPE *c, TYPE *s, TYPE *work, int *iwork); \
typedef void (*fb_##SUFFIX##_fortran_fn)(int *icompq, int *nl, int *nr, int *sqre,\
                                         TYPE *d, TYPE *vf, TYPE *vl, TYPE *alpha,\
                                         TYPE *beta, int *idxq, int *perm,       \
                                         int *givptr, int *givcol, int *ldgcol,  \
                                         TYPE *givnum, int *ldgnum, TYPE *poles, \
                                         TYPE *difl, TYPE *difr, TYPE *z, int *k,\
                                         TYPE *c, TYPE *s, TYPE *work,           \
                                         int *iwork, int *info);                 \
static struct {                                                                  \
    int called;                                                                  \
    int icompq;                                                                  \
    int nl;                                                                      \
    int nr;                                                                      \
    int sqre;                                                                    \
    TYPE *d;                                                                     \
    TYPE *vf;                                                                    \
    TYPE *vl;                                                                    \
    TYPE *alpha;                                                                 \
    TYPE *beta;                                                                  \
    int *idxq;                                                                   \
    int *perm;                                                                   \
    int *givptr;                                                                 \
    int *givcol;                                                                 \
    int ldgcol;                                                                  \
    TYPE *givnum;                                                                \
    int ldgnum;                                                                  \
    TYPE *poles;                                                                 \
    TYPE *difl;                                                                  \
    TYPE *difr;                                                                  \
    TYPE *z;                                                                     \
    int *k;                                                                      \
    TYPE *c;                                                                     \
    TYPE *s;                                                                     \
    TYPE *work;                                                                  \
    int *iwork;                                                                  \
} g_##SUFFIX##_fortran_call;                                                     \
static struct {                                                                  \
    int called;                                                                  \
    int icompq;                                                                  \
    int nl;                                                                      \
    int nr;                                                                      \
    int sqre;                                                                    \
    TYPE *d;                                                                     \
    TYPE *vf;                                                                    \
    TYPE *vl;                                                                    \
    TYPE *alpha;                                                                 \
    TYPE *beta;                                                                  \
    int *idxq;                                                                   \
    int *perm;                                                                   \
    int *givptr;                                                                 \
    int *givcol;                                                                 \
    int ldgcol;                                                                  \
    TYPE *givnum;                                                                \
    int ldgnum;                                                                  \
    TYPE *poles;                                                                 \
    TYPE *difl;                                                                  \
    TYPE *difr;                                                                  \
    TYPE *z;                                                                     \
    int *k;                                                                      \
    TYPE *c;                                                                     \
    TYPE *s;                                                                     \
    TYPE *work;                                                                  \
    int *iwork;                                                                  \
} g_##SUFFIX##_cblas_call;                                                       \
static void stub_##SUFFIX##_fortran(int *icompq, int *nl, int *nr, int *sqre,   \
                                    TYPE *d, TYPE *vf, TYPE *vl, TYPE *alpha,   \
                                    TYPE *beta, int *idxq, int *perm,           \
                                    int *givptr, int *givcol, int *ldgcol,      \
                                    TYPE *givnum, int *ldgnum, TYPE *poles,     \
                                    TYPE *difl, TYPE *difr, TYPE *z, int *k,    \
                                    TYPE *c, TYPE *s, TYPE *work, int *iwork,   \
                                    int *info)                                  \
{                                                                                \
    g_##SUFFIX##_fortran_call.called += 1;                                       \
    g_##SUFFIX##_fortran_call.icompq = *icompq;                                  \
    g_##SUFFIX##_fortran_call.nl = *nl;                                          \
    g_##SUFFIX##_fortran_call.nr = *nr;                                          \
    g_##SUFFIX##_fortran_call.sqre = *sqre;                                      \
    g_##SUFFIX##_fortran_call.d = d;                                             \
    g_##SUFFIX##_fortran_call.vf = vf;                                           \
    g_##SUFFIX##_fortran_call.vl = vl;                                           \
    g_##SUFFIX##_fortran_call.alpha = alpha;                                     \
    g_##SUFFIX##_fortran_call.beta = beta;                                       \
    g_##SUFFIX##_fortran_call.idxq = idxq;                                       \
    g_##SUFFIX##_fortran_call.perm = perm;                                       \
    g_##SUFFIX##_fortran_call.givptr = givptr;                                   \
    g_##SUFFIX##_fortran_call.givcol = givcol;                                   \
    g_##SUFFIX##_fortran_call.ldgcol = *ldgcol;                                  \
    g_##SUFFIX##_fortran_call.givnum = givnum;                                   \
    g_##SUFFIX##_fortran_call.ldgnum = *ldgnum;                                  \
    g_##SUFFIX##_fortran_call.poles = poles;                                     \
    g_##SUFFIX##_fortran_call.difl = difl;                                       \
    g_##SUFFIX##_fortran_call.difr = difr;                                       \
    g_##SUFFIX##_fortran_call.z = z;                                             \
    g_##SUFFIX##_fortran_call.k = k;                                             \
    g_##SUFFIX##_fortran_call.c = c;                                             \
    g_##SUFFIX##_fortran_call.s = s;                                             \
    g_##SUFFIX##_fortran_call.work = work;                                       \
    g_##SUFFIX##_fortran_call.iwork = iwork;                                     \
    *alpha = (TYPE)((BASE) + 1);                                                 \
    *beta = (TYPE)((BASE) + 2);                                                  \
    *givptr = (BASE) + 3;                                                        \
    givcol[0] = (BASE) + 4;                                                      \
    givnum[0] = (TYPE)((BASE) + 5);                                              \
    poles[0] = (TYPE)((BASE) + 6);                                               \
    difl[0] = (TYPE)((BASE) + 7);                                                \
    difr[0] = (TYPE)((BASE) + 8);                                                \
    z[0] = (TYPE)((BASE) + 9);                                                   \
    *k = (BASE) + 10;                                                            \
    *c = (TYPE)((BASE) + 11);                                                    \
    *s = (TYPE)((BASE) + 12);                                                    \
    work[0] = (TYPE)((BASE) + 13);                                               \
    iwork[0] = (BASE) + 14;                                                      \
    *info = (BASE) + 15;                                                         \
}                                                                                \
static int stub_##SUFFIX##_cblas(int icompq, int nl, int nr, int sqre, TYPE *d, \
                                 TYPE *vf, TYPE *vl, TYPE *alpha, TYPE *beta,   \
                                 int *idxq, int *perm, int *givptr,             \
                                 int *givcol, int ldgcol, TYPE *givnum,         \
                                 int ldgnum, TYPE *poles, TYPE *difl,           \
                                 TYPE *difr, TYPE *z, int *k, TYPE *c, TYPE *s, \
                                 TYPE *work, int *iwork)                        \
{                                                                                \
    g_##SUFFIX##_cblas_call.called += 1;                                         \
    g_##SUFFIX##_cblas_call.icompq = icompq;                                     \
    g_##SUFFIX##_cblas_call.nl = nl;                                             \
    g_##SUFFIX##_cblas_call.nr = nr;                                             \
    g_##SUFFIX##_cblas_call.sqre = sqre;                                         \
    g_##SUFFIX##_cblas_call.d = d;                                               \
    g_##SUFFIX##_cblas_call.vf = vf;                                             \
    g_##SUFFIX##_cblas_call.vl = vl;                                             \
    g_##SUFFIX##_cblas_call.alpha = alpha;                                       \
    g_##SUFFIX##_cblas_call.beta = beta;                                         \
    g_##SUFFIX##_cblas_call.idxq = idxq;                                         \
    g_##SUFFIX##_cblas_call.perm = perm;                                         \
    g_##SUFFIX##_cblas_call.givptr = givptr;                                     \
    g_##SUFFIX##_cblas_call.givcol = givcol;                                     \
    g_##SUFFIX##_cblas_call.ldgcol = ldgcol;                                     \
    g_##SUFFIX##_cblas_call.givnum = givnum;                                     \
    g_##SUFFIX##_cblas_call.ldgnum = ldgnum;                                     \
    g_##SUFFIX##_cblas_call.poles = poles;                                       \
    g_##SUFFIX##_cblas_call.difl = difl;                                         \
    g_##SUFFIX##_cblas_call.difr = difr;                                         \
    g_##SUFFIX##_cblas_call.z = z;                                               \
    g_##SUFFIX##_cblas_call.k = k;                                               \
    g_##SUFFIX##_cblas_call.c = c;                                               \
    g_##SUFFIX##_cblas_call.s = s;                                               \
    g_##SUFFIX##_cblas_call.work = work;                                         \
    g_##SUFFIX##_cblas_call.iwork = iwork;                                       \
    *alpha = (TYPE)((BASE) + 16);                                                \
    *beta = (TYPE)((BASE) + 17);                                                 \
    *givptr = (BASE) + 18;                                                       \
    givcol[0] = (BASE) + 19;                                                     \
    givnum[0] = (TYPE)((BASE) + 20);                                             \
    poles[0] = (TYPE)((BASE) + 21);                                              \
    difl[0] = (TYPE)((BASE) + 22);                                               \
    difr[0] = (TYPE)((BASE) + 23);                                               \
    z[0] = (TYPE)((BASE) + 24);                                                  \
    *k = (BASE) + 25;                                                            \
    *c = (TYPE)((BASE) + 26);                                                    \
    *s = (TYPE)((BASE) + 27);                                                    \
    work[0] = (TYPE)((BASE) + 28);                                               \
    iwork[0] = (BASE) + 29;                                                      \
    return (BASE) + 30;                                                          \
}                                                                                \
static int check_##SUFFIX##_fortran_to_cblas(void)                               \
{                                                                                \
    fb_backend_vtable_t vtable;                                                  \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                         \
    TYPE d[3] = { (TYPE)1, (TYPE)2, (TYPE)3 };                                  \
    TYPE vf[3] = { (TYPE)4, (TYPE)5, (TYPE)6 };                                 \
    TYPE vl[3] = { (TYPE)7, (TYPE)8, (TYPE)9 };                                 \
    TYPE alpha = (TYPE)10;                                                       \
    TYPE beta = (TYPE)11;                                                        \
    int idxq[3] = { 12, 13, 14 };                                                \
    int perm[3] = { 15, 16, 17 };                                                \
    int givptr = 0;                                                              \
    int givcol[3] = { 18, 19, 20 };                                              \
    TYPE givnum[3] = { (TYPE)21, (TYPE)22, (TYPE)23 };                          \
    TYPE poles[3] = { (TYPE)24, (TYPE)25, (TYPE)26 };                           \
    TYPE difl[3] = { (TYPE)27, (TYPE)28, (TYPE)29 };                            \
    TYPE difr[3] = { (TYPE)30, (TYPE)31, (TYPE)32 };                            \
    TYPE z[3] = { (TYPE)33, (TYPE)34, (TYPE)35 };                               \
    int k = 0;                                                                   \
    TYPE c = (TYPE)0;                                                            \
    TYPE s = (TYPE)0;                                                            \
    TYPE work[3] = { (TYPE)36, (TYPE)37, (TYPE)38 };                            \
    int iwork[3] = { 39, 40, 41 };                                               \
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
    if (thunk(0, 1, 1, 0, d, vf, vl, &alpha, &beta, idxq, perm, &givptr,        \
              givcol, 3, givnum, 3, poles, difl, difr, z, &k, &c, &s, work,     \
              iwork) != (BASE) + 15 ||                                           \
        g_##SUFFIX##_fortran_call.called != 1 ||                                 \
        g_##SUFFIX##_fortran_call.icompq != 0 ||                                 \
        g_##SUFFIX##_fortran_call.nl != 1 ||                                     \
        g_##SUFFIX##_fortran_call.nr != 1 ||                                     \
        g_##SUFFIX##_fortran_call.sqre != 0 ||                                   \
        g_##SUFFIX##_fortran_call.d != d ||                                      \
        g_##SUFFIX##_fortran_call.vf != vf ||                                    \
        g_##SUFFIX##_fortran_call.vl != vl ||                                    \
        g_##SUFFIX##_fortran_call.alpha != &alpha ||                             \
        g_##SUFFIX##_fortran_call.beta != &beta ||                               \
        g_##SUFFIX##_fortran_call.idxq != idxq ||                                \
        g_##SUFFIX##_fortran_call.perm != perm ||                                \
        g_##SUFFIX##_fortran_call.givptr != &givptr ||                           \
        g_##SUFFIX##_fortran_call.givcol != givcol ||                            \
        g_##SUFFIX##_fortran_call.ldgcol != 3 ||                                 \
        g_##SUFFIX##_fortran_call.givnum != givnum ||                            \
        g_##SUFFIX##_fortran_call.ldgnum != 3 ||                                 \
        g_##SUFFIX##_fortran_call.poles != poles ||                              \
        g_##SUFFIX##_fortran_call.difl != difl ||                                \
        g_##SUFFIX##_fortran_call.difr != difr ||                                \
        g_##SUFFIX##_fortran_call.z != z ||                                      \
        g_##SUFFIX##_fortran_call.k != &k ||                                     \
        g_##SUFFIX##_fortran_call.c != &c ||                                     \
        g_##SUFFIX##_fortran_call.s != &s ||                                     \
        g_##SUFFIX##_fortran_call.work != work ||                                \
        g_##SUFFIX##_fortran_call.iwork != iwork ||                              \
        alpha != (TYPE)((BASE) + 1) ||                                           \
        beta != (TYPE)((BASE) + 2) ||                                            \
        givptr != (BASE) + 3 ||                                                  \
        givcol[0] != (BASE) + 4 ||                                               \
        givnum[0] != (TYPE)((BASE) + 5) ||                                       \
        poles[0] != (TYPE)((BASE) + 6) ||                                        \
        difl[0] != (TYPE)((BASE) + 7) ||                                         \
        difr[0] != (TYPE)((BASE) + 8) ||                                         \
        z[0] != (TYPE)((BASE) + 9) ||                                            \
        k != (BASE) + 10 ||                                                      \
        c != (TYPE)((BASE) + 11) ||                                              \
        s != (TYPE)((BASE) + 12) ||                                              \
        work[0] != (TYPE)((BASE) + 13) ||                                        \
        iwork[0] != (BASE) + 14) {                                               \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward LASD6 inputs correctly\n"); \
        return 1;                                                                \
    }                                                                            \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards LASD6 inputs\n"); \
    return 0;                                                                    \
}                                                                                \
static int check_##SUFFIX##_cblas_to_fortran(void)                               \
{                                                                                \
    fb_backend_vtable_t vtable;                                                  \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                       \
    int icompq = 0;                                                              \
    int nl = 1;                                                                  \
    int nr = 1;                                                                  \
    int sqre = 0;                                                                \
    TYPE d[3] = { (TYPE)41, (TYPE)42, (TYPE)43 };                               \
    TYPE vf[3] = { (TYPE)44, (TYPE)45, (TYPE)46 };                              \
    TYPE vl[3] = { (TYPE)47, (TYPE)48, (TYPE)49 };                              \
    TYPE alpha = (TYPE)50;                                                       \
    TYPE beta = (TYPE)51;                                                        \
    int idxq[3] = { 52, 53, 54 };                                                \
    int perm[3] = { 55, 56, 57 };                                                \
    int givptr = 0;                                                              \
    int givcol[3] = { 58, 59, 60 };                                              \
    int ldgcol = 3;                                                              \
    TYPE givnum[3] = { (TYPE)61, (TYPE)62, (TYPE)63 };                          \
    int ldgnum = 3;                                                              \
    TYPE poles[3] = { (TYPE)64, (TYPE)65, (TYPE)66 };                           \
    TYPE difl[3] = { (TYPE)67, (TYPE)68, (TYPE)69 };                            \
    TYPE difr[3] = { (TYPE)70, (TYPE)71, (TYPE)72 };                            \
    TYPE z[3] = { (TYPE)73, (TYPE)74, (TYPE)75 };                               \
    int k = 0;                                                                   \
    TYPE c = (TYPE)0;                                                            \
    TYPE s = (TYPE)0;                                                            \
    TYPE work[3] = { (TYPE)76, (TYPE)77, (TYPE)78 };                            \
    int iwork[3] = { 79, 80, 81 };                                               \
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
    thunk(&icompq, &nl, &nr, &sqre, d, vf, vl, &alpha, &beta, idxq, perm,       \
          &givptr, givcol, &ldgcol, givnum, &ldgnum, poles, difl, difr, z, &k,  \
          &c, &s, work, iwork, &info);                                           \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                   \
        g_##SUFFIX##_cblas_call.icompq != 0 ||                                   \
        g_##SUFFIX##_cblas_call.nl != 1 ||                                       \
        g_##SUFFIX##_cblas_call.nr != 1 ||                                       \
        g_##SUFFIX##_cblas_call.sqre != 0 ||                                     \
        g_##SUFFIX##_cblas_call.d != d ||                                        \
        g_##SUFFIX##_cblas_call.vf != vf ||                                      \
        g_##SUFFIX##_cblas_call.vl != vl ||                                      \
        g_##SUFFIX##_cblas_call.alpha != &alpha ||                               \
        g_##SUFFIX##_cblas_call.beta != &beta ||                                 \
        g_##SUFFIX##_cblas_call.idxq != idxq ||                                  \
        g_##SUFFIX##_cblas_call.perm != perm ||                                  \
        g_##SUFFIX##_cblas_call.givptr != &givptr ||                             \
        g_##SUFFIX##_cblas_call.givcol != givcol ||                              \
        g_##SUFFIX##_cblas_call.ldgcol != 3 ||                                   \
        g_##SUFFIX##_cblas_call.givnum != givnum ||                              \
        g_##SUFFIX##_cblas_call.ldgnum != 3 ||                                   \
        g_##SUFFIX##_cblas_call.poles != poles ||                                \
        g_##SUFFIX##_cblas_call.difl != difl ||                                  \
        g_##SUFFIX##_cblas_call.difr != difr ||                                  \
        g_##SUFFIX##_cblas_call.z != z ||                                        \
        g_##SUFFIX##_cblas_call.k != &k ||                                       \
        g_##SUFFIX##_cblas_call.c != &c ||                                       \
        g_##SUFFIX##_cblas_call.s != &s ||                                       \
        g_##SUFFIX##_cblas_call.work != work ||                                  \
        g_##SUFFIX##_cblas_call.iwork != iwork ||                                \
        info != (BASE) + 30 ||                                                   \
        alpha != (TYPE)((BASE) + 16) ||                                          \
        beta != (TYPE)((BASE) + 17) ||                                           \
        givptr != (BASE) + 18 ||                                                 \
        givcol[0] != (BASE) + 19 ||                                              \
        givnum[0] != (TYPE)((BASE) + 20) ||                                      \
        poles[0] != (TYPE)((BASE) + 21) ||                                       \
        difl[0] != (TYPE)((BASE) + 22) ||                                        \
        difr[0] != (TYPE)((BASE) + 23) ||                                        \
        z[0] != (TYPE)((BASE) + 24) ||                                           \
        k != (BASE) + 25 ||                                                      \
        c != (TYPE)((BASE) + 26) ||                                              \
        s != (TYPE)((BASE) + 27) ||                                              \
        work[0] != (TYPE)((BASE) + 28) ||                                        \
        iwork[0] != (BASE) + 29) {                                               \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference LASD6 inputs correctly\n"); \
        return 1;                                                                \
    }                                                                            \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences LASD6 inputs\n"); \
    return 0;                                                                    \
}

DEFINE_LASD6_TESTS(slasd6, float, FB_OP_SLASD6, 100)
DEFINE_LASD6_TESTS(dlasd6, double, FB_OP_DLASD6, 300)

int main(void)
{
    int status = 0;

    status |= check_slasd6_fortran_to_cblas();
    status |= check_slasd6_cblas_to_fortran();
    status |= check_dlasd6_fortran_to_cblas();
    status |= check_dlasd6_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}