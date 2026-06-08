#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

#define DEFINE_LASD1_TESTS(SUFFIX, TYPE, OP_ID, BASE)                           \
typedef int (*fb_##SUFFIX##_cblas_fn)(int nl, int nr, int sqre, TYPE *d,        \
                                      TYPE alpha, TYPE beta, TYPE *u, int ldu,   \
                                      TYPE *vt, int ldvt, int *idxq,             \
                                      int *iwork, TYPE *work);                   \
typedef void (*fb_##SUFFIX##_fortran_fn)(int *nl, int *nr, int *sqre, TYPE *d,  \
                                         TYPE *alpha, TYPE *beta, TYPE *u,       \
                                         int *ldu, TYPE *vt, int *ldvt,          \
                                         int *idxq, int *iwork, TYPE *work,      \
                                         int *info);                             \
static struct {                                                                  \
    int called;                                                                  \
    int nl;                                                                      \
    int nr;                                                                      \
    int sqre;                                                                    \
    TYPE *d;                                                                     \
    TYPE alpha;                                                                  \
    TYPE beta;                                                                   \
    TYPE *u;                                                                     \
    int ldu;                                                                     \
    TYPE *vt;                                                                    \
    int ldvt;                                                                    \
    int *idxq;                                                                   \
    int *iwork;                                                                  \
    TYPE *work;                                                                  \
} g_##SUFFIX##_fortran_call;                                                     \
static struct {                                                                  \
    int called;                                                                  \
    int nl;                                                                      \
    int nr;                                                                      \
    int sqre;                                                                    \
    TYPE *d;                                                                     \
    TYPE alpha;                                                                  \
    TYPE beta;                                                                   \
    TYPE *u;                                                                     \
    int ldu;                                                                     \
    TYPE *vt;                                                                    \
    int ldvt;                                                                    \
    int *idxq;                                                                   \
    int *iwork;                                                                  \
    TYPE *work;                                                                  \
} g_##SUFFIX##_cblas_call;                                                       \
static void stub_##SUFFIX##_fortran(int *nl, int *nr, int *sqre, TYPE *d,       \
                                    TYPE *alpha, TYPE *beta, TYPE *u, int *ldu, \
                                    TYPE *vt, int *ldvt, int *idxq,             \
                                    int *iwork, TYPE *work, int *info)          \
{                                                                                \
    g_##SUFFIX##_fortran_call.called += 1;                                       \
    g_##SUFFIX##_fortran_call.nl = *nl;                                          \
    g_##SUFFIX##_fortran_call.nr = *nr;                                          \
    g_##SUFFIX##_fortran_call.sqre = *sqre;                                      \
    g_##SUFFIX##_fortran_call.d = d;                                             \
    g_##SUFFIX##_fortran_call.alpha = *alpha;                                    \
    g_##SUFFIX##_fortran_call.beta = *beta;                                      \
    g_##SUFFIX##_fortran_call.u = u;                                             \
    g_##SUFFIX##_fortran_call.ldu = *ldu;                                        \
    g_##SUFFIX##_fortran_call.vt = vt;                                           \
    g_##SUFFIX##_fortran_call.ldvt = *ldvt;                                      \
    g_##SUFFIX##_fortran_call.idxq = idxq;                                       \
    g_##SUFFIX##_fortran_call.iwork = iwork;                                     \
    g_##SUFFIX##_fortran_call.work = work;                                       \
    d[0] = (TYPE)((BASE) + 1);                                                   \
    u[0] = (TYPE)((BASE) + 2);                                                   \
    vt[3] = (TYPE)((BASE) + 3);                                                  \
    idxq[0] = (BASE) + 4;                                                        \
    iwork[0] = (BASE) + 5;                                                       \
    work[0] = (TYPE)((BASE) + 6);                                                \
    *info = (BASE) + 7;                                                          \
}                                                                                \
static int stub_##SUFFIX##_cblas(int nl, int nr, int sqre, TYPE *d, TYPE alpha, \
                                 TYPE beta, TYPE *u, int ldu, TYPE *vt,         \
                                 int ldvt, int *idxq, int *iwork, TYPE *work)   \
{                                                                                \
    g_##SUFFIX##_cblas_call.called += 1;                                         \
    g_##SUFFIX##_cblas_call.nl = nl;                                             \
    g_##SUFFIX##_cblas_call.nr = nr;                                             \
    g_##SUFFIX##_cblas_call.sqre = sqre;                                         \
    g_##SUFFIX##_cblas_call.d = d;                                               \
    g_##SUFFIX##_cblas_call.alpha = alpha;                                       \
    g_##SUFFIX##_cblas_call.beta = beta;                                         \
    g_##SUFFIX##_cblas_call.u = u;                                               \
    g_##SUFFIX##_cblas_call.ldu = ldu;                                           \
    g_##SUFFIX##_cblas_call.vt = vt;                                             \
    g_##SUFFIX##_cblas_call.ldvt = ldvt;                                         \
    g_##SUFFIX##_cblas_call.idxq = idxq;                                         \
    g_##SUFFIX##_cblas_call.iwork = iwork;                                       \
    g_##SUFFIX##_cblas_call.work = work;                                         \
    d[0] = (TYPE)((BASE) + 8);                                                   \
    u[0] = (TYPE)((BASE) + 9);                                                   \
    vt[3] = (TYPE)((BASE) + 10);                                                 \
    idxq[0] = (BASE) + 11;                                                       \
    iwork[0] = (BASE) + 12;                                                      \
    work[0] = (TYPE)((BASE) + 13);                                               \
    return (BASE) + 14;                                                          \
}                                                                                \
static int check_##SUFFIX##_fortran_to_cblas(void)                               \
{                                                                                \
    fb_backend_vtable_t vtable;                                                  \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                         \
    TYPE d[3] = { (TYPE)1, (TYPE)2, (TYPE)3 };                                  \
    TYPE u[4] = { (TYPE)4, (TYPE)5, (TYPE)6, (TYPE)7 };                         \
    TYPE vt[4] = { (TYPE)8, (TYPE)9, (TYPE)10, (TYPE)11 };                      \
    int idxq[4] = { 12, 13, 14, 15 };                                            \
    int iwork[4] = { 16, 17, 18, 19 };                                           \
    TYPE work[4] = { (TYPE)20, (TYPE)21, (TYPE)22, (TYPE)23 };                  \
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
    if (thunk(1, 1, 0, d, (TYPE)24, (TYPE)25, u, 2, vt, 2, idxq, iwork, work)  \
            != (BASE) + 7 ||                                                     \
        g_##SUFFIX##_fortran_call.called != 1 ||                                 \
        g_##SUFFIX##_fortran_call.nl != 1 ||                                     \
        g_##SUFFIX##_fortran_call.nr != 1 ||                                     \
        g_##SUFFIX##_fortran_call.sqre != 0 ||                                   \
        g_##SUFFIX##_fortran_call.d != d ||                                      \
        g_##SUFFIX##_fortran_call.alpha != (TYPE)24 ||                           \
        g_##SUFFIX##_fortran_call.beta != (TYPE)25 ||                            \
        g_##SUFFIX##_fortran_call.u != u ||                                      \
        g_##SUFFIX##_fortran_call.ldu != 2 ||                                    \
        g_##SUFFIX##_fortran_call.vt != vt ||                                    \
        g_##SUFFIX##_fortran_call.ldvt != 2 ||                                   \
        g_##SUFFIX##_fortran_call.idxq != idxq ||                                \
        g_##SUFFIX##_fortran_call.iwork != iwork ||                              \
        g_##SUFFIX##_fortran_call.work != work ||                                \
        d[0] != (TYPE)((BASE) + 1) ||                                            \
        u[0] != (TYPE)((BASE) + 2) ||                                            \
        vt[3] != (TYPE)((BASE) + 3) ||                                           \
        idxq[0] != (BASE) + 4 ||                                                 \
        iwork[0] != (BASE) + 5 ||                                                \
        work[0] != (TYPE)((BASE) + 6)) {                                         \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward LASD1 inputs correctly\n"); \
        return 1;                                                                \
    }                                                                            \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards LASD1 inputs\n"); \
    return 0;                                                                    \
}                                                                                \
static int check_##SUFFIX##_cblas_to_fortran(void)                               \
{                                                                                \
    fb_backend_vtable_t vtable;                                                  \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                       \
    int nl = 1;                                                                  \
    int nr = 1;                                                                  \
    int sqre = 0;                                                                \
    TYPE alpha = (TYPE)24;                                                       \
    TYPE beta = (TYPE)25;                                                        \
    int ldu = 2;                                                                 \
    int ldvt = 2;                                                                \
    int info = -1;                                                               \
    TYPE d[3] = { (TYPE)31, (TYPE)32, (TYPE)33 };                               \
    TYPE u[4] = { (TYPE)34, (TYPE)35, (TYPE)36, (TYPE)37 };                     \
    TYPE vt[4] = { (TYPE)38, (TYPE)39, (TYPE)40, (TYPE)41 };                    \
    int idxq[4] = { 42, 43, 44, 45 };                                            \
    int iwork[4] = { 46, 47, 48, 49 };                                           \
    TYPE work[4] = { (TYPE)50, (TYPE)51, (TYPE)52, (TYPE)53 };                  \
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
    thunk(&nl, &nr, &sqre, d, &alpha, &beta, u, &ldu, vt, &ldvt, idxq, iwork,  \
          work, &info);                                                          \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                   \
        g_##SUFFIX##_cblas_call.nl != 1 ||                                       \
        g_##SUFFIX##_cblas_call.nr != 1 ||                                       \
        g_##SUFFIX##_cblas_call.sqre != 0 ||                                     \
        g_##SUFFIX##_cblas_call.d != d ||                                        \
        g_##SUFFIX##_cblas_call.alpha != (TYPE)24 ||                             \
        g_##SUFFIX##_cblas_call.beta != (TYPE)25 ||                              \
        g_##SUFFIX##_cblas_call.u != u ||                                        \
        g_##SUFFIX##_cblas_call.ldu != 2 ||                                      \
        g_##SUFFIX##_cblas_call.vt != vt ||                                      \
        g_##SUFFIX##_cblas_call.ldvt != 2 ||                                     \
        g_##SUFFIX##_cblas_call.idxq != idxq ||                                  \
        g_##SUFFIX##_cblas_call.iwork != iwork ||                                \
        g_##SUFFIX##_cblas_call.work != work ||                                  \
        info != (BASE) + 14 ||                                                   \
        d[0] != (TYPE)((BASE) + 8) ||                                            \
        u[0] != (TYPE)((BASE) + 9) ||                                            \
        vt[3] != (TYPE)((BASE) + 10) ||                                          \
        idxq[0] != (BASE) + 11 ||                                                \
        iwork[0] != (BASE) + 12 ||                                               \
        work[0] != (TYPE)((BASE) + 13)) {                                        \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference LASD1 inputs correctly\n"); \
        return 1;                                                                \
    }                                                                            \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences LASD1 inputs\n"); \
    return 0;                                                                    \
}

DEFINE_LASD1_TESTS(slasd1, float, FB_OP_SLASD1, 100)
DEFINE_LASD1_TESTS(dlasd1, double, FB_OP_DLASD1, 300)

int main(void)
{
    int status = 0;

    status |= check_slasd1_fortran_to_cblas();
    status |= check_slasd1_cblas_to_fortran();
    status |= check_dlasd1_fortran_to_cblas();
    status |= check_dlasd1_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}