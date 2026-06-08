#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

#define DEFINE_LASD0_TESTS(SUFFIX, TYPE, OP_ID, BASE)                           \
typedef int (*fb_##SUFFIX##_cblas_fn)(int n, int sqre, TYPE *d, const TYPE *e, \
                                      TYPE *u, int ldu, TYPE *vt, int ldvt,     \
                                      int smlsiz, int *iwork, TYPE *work);      \
typedef void (*fb_##SUFFIX##_fortran_fn)(int *n, int *sqre, TYPE *d, TYPE *e,  \
                                         TYPE *u, int *ldu, TYPE *vt, int *ldvt,\
                                         int *smlsiz, int *iwork, TYPE *work,    \
                                         int *info);                             \
static struct {                                                                  \
    int called;                                                                  \
    int n;                                                                       \
    int sqre;                                                                    \
    TYPE *d;                                                                     \
    const TYPE *e;                                                               \
    TYPE *u;                                                                     \
    int ldu;                                                                     \
    TYPE *vt;                                                                    \
    int ldvt;                                                                    \
    int smlsiz;                                                                  \
    int *iwork;                                                                  \
    TYPE *work;                                                                  \
} g_##SUFFIX##_fortran_call;                                                     \
static struct {                                                                  \
    int called;                                                                  \
    int n;                                                                       \
    int sqre;                                                                    \
    TYPE *d;                                                                     \
    const TYPE *e;                                                               \
    TYPE *u;                                                                     \
    int ldu;                                                                     \
    TYPE *vt;                                                                    \
    int ldvt;                                                                    \
    int smlsiz;                                                                  \
    int *iwork;                                                                  \
    TYPE *work;                                                                  \
} g_##SUFFIX##_cblas_call;                                                       \
static void stub_##SUFFIX##_fortran(int *n, int *sqre, TYPE *d, TYPE *e,        \
                                    TYPE *u, int *ldu, TYPE *vt, int *ldvt,     \
                                    int *smlsiz, int *iwork, TYPE *work,         \
                                    int *info)                                   \
{                                                                                \
    g_##SUFFIX##_fortran_call.called += 1;                                       \
    g_##SUFFIX##_fortran_call.n = *n;                                            \
    g_##SUFFIX##_fortran_call.sqre = *sqre;                                      \
    g_##SUFFIX##_fortran_call.d = d;                                             \
    g_##SUFFIX##_fortran_call.e = e;                                             \
    g_##SUFFIX##_fortran_call.u = u;                                             \
    g_##SUFFIX##_fortran_call.ldu = *ldu;                                        \
    g_##SUFFIX##_fortran_call.vt = vt;                                           \
    g_##SUFFIX##_fortran_call.ldvt = *ldvt;                                      \
    g_##SUFFIX##_fortran_call.smlsiz = *smlsiz;                                  \
    g_##SUFFIX##_fortran_call.iwork = iwork;                                     \
    g_##SUFFIX##_fortran_call.work = work;                                       \
    d[0] = (TYPE)((BASE) + 1);                                                   \
    u[0] = (TYPE)((BASE) + 2);                                                   \
    vt[3] = (TYPE)((BASE) + 3);                                                  \
    iwork[0] = (BASE) + 4;                                                       \
    work[0] = (TYPE)((BASE) + 5);                                                \
    *info = (BASE) + 6;                                                          \
}                                                                                \
static int stub_##SUFFIX##_cblas(int n, int sqre, TYPE *d, const TYPE *e,       \
                                 TYPE *u, int ldu, TYPE *vt, int ldvt,           \
                                 int smlsiz, int *iwork, TYPE *work)             \
{                                                                                \
    g_##SUFFIX##_cblas_call.called += 1;                                         \
    g_##SUFFIX##_cblas_call.n = n;                                               \
    g_##SUFFIX##_cblas_call.sqre = sqre;                                         \
    g_##SUFFIX##_cblas_call.d = d;                                               \
    g_##SUFFIX##_cblas_call.e = e;                                               \
    g_##SUFFIX##_cblas_call.u = u;                                               \
    g_##SUFFIX##_cblas_call.ldu = ldu;                                           \
    g_##SUFFIX##_cblas_call.vt = vt;                                             \
    g_##SUFFIX##_cblas_call.ldvt = ldvt;                                         \
    g_##SUFFIX##_cblas_call.smlsiz = smlsiz;                                     \
    g_##SUFFIX##_cblas_call.iwork = iwork;                                       \
    g_##SUFFIX##_cblas_call.work = work;                                         \
    d[0] = (TYPE)((BASE) + 7);                                                   \
    u[0] = (TYPE)((BASE) + 8);                                                   \
    vt[3] = (TYPE)((BASE) + 9);                                                  \
    iwork[0] = (BASE) + 10;                                                      \
    work[0] = (TYPE)((BASE) + 11);                                               \
    return (BASE) + 12;                                                          \
}                                                                                \
static int check_##SUFFIX##_fortran_to_cblas(void)                               \
{                                                                                \
    fb_backend_vtable_t vtable;                                                  \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                         \
    TYPE d[2] = { (TYPE)1, (TYPE)2 };                                            \
    TYPE e[2] = { (TYPE)3, (TYPE)4 };                                            \
    TYPE u[4] = { (TYPE)5, (TYPE)6, (TYPE)7, (TYPE)8 };                         \
    TYPE vt[4] = { (TYPE)9, (TYPE)10, (TYPE)11, (TYPE)12 };                     \
    int iwork[4] = { 13, 14, 15, 16 };                                           \
    TYPE work[4] = { (TYPE)17, (TYPE)18, (TYPE)19, (TYPE)20 };                  \
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
    if (thunk(2, 0, d, e, u, 2, vt, 2, 3, iwork, work) != (BASE) + 6 ||         \
        g_##SUFFIX##_fortran_call.called != 1 ||                                 \
        g_##SUFFIX##_fortran_call.n != 2 ||                                      \
        g_##SUFFIX##_fortran_call.sqre != 0 ||                                   \
        g_##SUFFIX##_fortran_call.d != d ||                                      \
        g_##SUFFIX##_fortran_call.e != e ||                                      \
        g_##SUFFIX##_fortran_call.u != u ||                                      \
        g_##SUFFIX##_fortran_call.ldu != 2 ||                                    \
        g_##SUFFIX##_fortran_call.vt != vt ||                                    \
        g_##SUFFIX##_fortran_call.ldvt != 2 ||                                   \
        g_##SUFFIX##_fortran_call.smlsiz != 3 ||                                 \
        g_##SUFFIX##_fortran_call.iwork != iwork ||                              \
        g_##SUFFIX##_fortran_call.work != work ||                                \
        d[0] != (TYPE)((BASE) + 1) ||                                            \
        u[0] != (TYPE)((BASE) + 2) ||                                            \
        vt[3] != (TYPE)((BASE) + 3) ||                                           \
        iwork[0] != (BASE) + 4 ||                                                \
        work[0] != (TYPE)((BASE) + 5) ||                                         \
        e[0] != (TYPE)3 || e[1] != (TYPE)4) {                                    \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward LASD0 inputs correctly\n"); \
        return 1;                                                                \
    }                                                                            \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards LASD0 inputs\n"); \
    return 0;                                                                    \
}                                                                                \
static int check_##SUFFIX##_cblas_to_fortran(void)                               \
{                                                                                \
    fb_backend_vtable_t vtable;                                                  \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                       \
    int n = 2;                                                                   \
    int sqre = 0;                                                                \
    int ldu = 2;                                                                 \
    int ldvt = 2;                                                                \
    int smlsiz = 3;                                                              \
    int info = -1;                                                               \
    TYPE d[2] = { (TYPE)21, (TYPE)22 };                                          \
    TYPE e[2] = { (TYPE)23, (TYPE)24 };                                          \
    TYPE u[4] = { (TYPE)25, (TYPE)26, (TYPE)27, (TYPE)28 };                     \
    TYPE vt[4] = { (TYPE)29, (TYPE)30, (TYPE)31, (TYPE)32 };                    \
    int iwork[4] = { 33, 34, 35, 36 };                                           \
    TYPE work[4] = { (TYPE)37, (TYPE)38, (TYPE)39, (TYPE)40 };                  \
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
    thunk(&n, &sqre, d, e, u, &ldu, vt, &ldvt, &smlsiz, iwork, work, &info);    \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                   \
        g_##SUFFIX##_cblas_call.n != 2 ||                                        \
        g_##SUFFIX##_cblas_call.sqre != 0 ||                                     \
        g_##SUFFIX##_cblas_call.d != d ||                                        \
        g_##SUFFIX##_cblas_call.e != e ||                                        \
        g_##SUFFIX##_cblas_call.u != u ||                                        \
        g_##SUFFIX##_cblas_call.ldu != 2 ||                                      \
        g_##SUFFIX##_cblas_call.vt != vt ||                                      \
        g_##SUFFIX##_cblas_call.ldvt != 2 ||                                     \
        g_##SUFFIX##_cblas_call.smlsiz != 3 ||                                   \
        g_##SUFFIX##_cblas_call.iwork != iwork ||                                \
        g_##SUFFIX##_cblas_call.work != work ||                                  \
        info != (BASE) + 12 ||                                                   \
        d[0] != (TYPE)((BASE) + 7) ||                                            \
        u[0] != (TYPE)((BASE) + 8) ||                                            \
        vt[3] != (TYPE)((BASE) + 9) ||                                           \
        iwork[0] != (BASE) + 10 ||                                               \
        work[0] != (TYPE)((BASE) + 11) ||                                        \
        e[0] != (TYPE)23 || e[1] != (TYPE)24) {                                  \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference LASD0 inputs correctly\n"); \
        return 1;                                                                \
    }                                                                            \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences LASD0 inputs\n"); \
    return 0;                                                                    \
}

DEFINE_LASD0_TESTS(slasd0, float, FB_OP_SLASD0, 100)
DEFINE_LASD0_TESTS(dlasd0, double, FB_OP_DLASD0, 300)

int main(void)
{
    int status = 0;

    status |= check_slasd0_fortran_to_cblas();
    status |= check_slasd0_cblas_to_fortran();
    status |= check_dlasd0_fortran_to_cblas();
    status |= check_dlasd0_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}