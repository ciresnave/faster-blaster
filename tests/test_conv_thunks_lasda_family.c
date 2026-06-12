#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

#define DEFINE_LASDA_TESTS(SUFFIX, TYPE, OP_ID, BASE)                           \
typedef int (*fb_##SUFFIX##_cblas_fn)(int icompq, int smlsiz, int n, int sqre, \
                                      TYPE *d, TYPE *e, TYPE *u, int ldu,      \
                                      TYPE *vt, int ldvt, int *k, TYPE *difl,  \
                                      TYPE *difr, TYPE *z, TYPE *zw, TYPE *work,\
                                      int *iwork);                              \
typedef void (*fb_##SUFFIX##_fortran_fn)(int *icompq, int *smlsiz, int *n,     \
                                         int *sqre, TYPE *d, TYPE *e, TYPE *u, \
                                         int *ldu, TYPE *vt, int *ldvt, int *k,\
                                         TYPE *difl, TYPE *difr, TYPE *z,      \
                                         TYPE *zw, TYPE *work, int *iwork,     \
                                         int *info);                           \
static struct {                                                                  \
    int called;                                                                  \
    int icompq;                                                                  \
    int smlsiz;                                                                  \
    int n;                                                                       \
    int sqre;                                                                    \
    TYPE *d;                                                                     \
    TYPE *e;                                                                     \
    TYPE *u;                                                                     \
    int ldu;                                                                     \
    TYPE *vt;                                                                    \
    int ldvt;                                                                    \
    int *k;                                                                      \
    TYPE *difl;                                                                  \
    TYPE *difr;                                                                  \
    TYPE *z;                                                                     \
    TYPE *zw;                                                                    \
    TYPE *work;                                                                  \
    int *iwork;                                                                  \
} g_##SUFFIX##_fortran_call;                                                     \
static struct {                                                                  \
    int called;                                                                  \
    int icompq;                                                                  \
    int smlsiz;                                                                  \
    int n;                                                                       \
    int sqre;                                                                    \
    TYPE *d;                                                                     \
    TYPE *e;                                                                     \
    TYPE *u;                                                                     \
    int ldu;                                                                     \
    TYPE *vt;                                                                    \
    int ldvt;                                                                    \
    int *k;                                                                      \
    TYPE *difl;                                                                  \
    TYPE *difr;                                                                  \
    TYPE *z;                                                                     \
    TYPE *zw;                                                                    \
    TYPE *work;                                                                  \
    int *iwork;                                                                  \
} g_##SUFFIX##_cblas_call;                                                       \
static void stub_##SUFFIX##_fortran(int *icompq, int *smlsiz, int *n, int *sqre,\
                                    TYPE *d, TYPE *e, TYPE *u, int *ldu,       \
                                    TYPE *vt, int *ldvt, int *k, TYPE *difl,   \
                                    TYPE *difr, TYPE *z, TYPE *zw, TYPE *work, \
                                    int *iwork, int *info) {                   \
    g_##SUFFIX##_fortran_call.called += 1;                                      \
    g_##SUFFIX##_fortran_call.icompq = *icompq;                                 \
    g_##SUFFIX##_fortran_call.smlsiz = *smlsiz;                                 \
    g_##SUFFIX##_fortran_call.n = *n;                                            \
    g_##SUFFIX##_fortran_call.sqre = *sqre;                                      \
    g_##SUFFIX##_fortran_call.d = d;                                             \
    g_##SUFFIX##_fortran_call.e = e;                                             \
    g_##SUFFIX##_fortran_call.u = u;                                             \
    g_##SUFFIX##_fortran_call.ldu = *ldu;                                        \
    g_##SUFFIX##_fortran_call.vt = vt;                                           \
    g_##SUFFIX##_fortran_call.ldvt = *ldvt;                                      \
    g_##SUFFIX##_fortran_call.k = k;                                             \
    g_##SUFFIX##_fortran_call.difl = difl;                                       \
    g_##SUFFIX##_fortran_call.difr = difr;                                       \
    g_##SUFFIX##_fortran_call.z = z;                                             \
    g_##SUFFIX##_fortran_call.zw = zw;                                           \
    g_##SUFFIX##_fortran_call.work = work;                                       \
    g_##SUFFIX##_fortran_call.iwork = iwork;                                     \
    d[0] = (TYPE)((BASE) + 1);                                                   \
    u[0] = (TYPE)((BASE) + 2);                                                   \
    vt[0] = (TYPE)((BASE) + 3);                                                  \
    difl[0] = (TYPE)((BASE) + 4);                                                \
    difr[0] = (TYPE)((BASE) + 5);                                                \
    z[0] = (TYPE)((BASE) + 6);                                                   \
    zw[0] = (TYPE)((BASE) + 7);                                                  \
    work[0] = (TYPE)((BASE) + 8);                                                \
    iwork[0] = (BASE) + 9;                                                       \
    *k = (BASE) + 10;                                                            \
    *info = (BASE) + 11;                                                         \
}                                                                                \
static int stub_##SUFFIX##_cblas(int icompq, int smlsiz, int n, int sqre,       \
                                 TYPE *d, TYPE *e, TYPE *u, int ldu, TYPE *vt, \
                                 int ldvt, int *k, TYPE *difl, TYPE *difr,     \
                                 TYPE *z, TYPE *zw, TYPE *work, int *iwork) {  \
    g_##SUFFIX##_cblas_call.called += 1;                                        \
    g_##SUFFIX##_cblas_call.icompq = icompq;                                    \
    g_##SUFFIX##_cblas_call.smlsiz = smlsiz;                                    \
    g_##SUFFIX##_cblas_call.n = n;                                              \
    g_##SUFFIX##_cblas_call.sqre = sqre;                                        \
    g_##SUFFIX##_cblas_call.d = d;                                              \
    g_##SUFFIX##_cblas_call.e = e;                                              \
    g_##SUFFIX##_cblas_call.u = u;                                              \
    g_##SUFFIX##_cblas_call.ldu = ldu;                                          \
    g_##SUFFIX##_cblas_call.vt = vt;                                            \
    g_##SUFFIX##_cblas_call.ldvt = ldvt;                                        \
    g_##SUFFIX##_cblas_call.k = k;                                              \
    g_##SUFFIX##_cblas_call.difl = difl;                                        \
    g_##SUFFIX##_cblas_call.difr = difr;                                        \
    g_##SUFFIX##_cblas_call.z = z;                                              \
    g_##SUFFIX##_cblas_call.zw = zw;                                            \
    g_##SUFFIX##_cblas_call.work = work;                                        \
    g_##SUFFIX##_cblas_call.iwork = iwork;                                      \
    d[0] = (TYPE)((BASE) + 12);                                                 \
    u[0] = (TYPE)((BASE) + 13);                                                 \
    vt[0] = (TYPE)((BASE) + 14);                                                \
    difl[0] = (TYPE)((BASE) + 15);                                              \
    difr[0] = (TYPE)((BASE) + 16);                                              \
    z[0] = (TYPE)((BASE) + 17);                                                 \
    zw[0] = (TYPE)((BASE) + 18);                                                \
    work[0] = (TYPE)((BASE) + 19);                                              \
    iwork[0] = (BASE) + 20;                                                     \
    *k = (BASE) + 21;                                                           \
    return (BASE) + 22;                                                         \
}                                                                                \
static int check_##SUFFIX##_fortran_to_cblas(void)                               \
{                                                                                \
    fb_backend_vtable_t vtable;                                                  \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                         \
    TYPE d[3] = { (TYPE)1, (TYPE)2, (TYPE)3 };                                  \
    TYPE e[3] = { (TYPE)4, (TYPE)5, (TYPE)6 };                                  \
    TYPE u[9] = { (TYPE)7, (TYPE)8, (TYPE)9, (TYPE)10, (TYPE)11, (TYPE)12,      \
                   (TYPE)13, (TYPE)14, (TYPE)15 };                              \
    TYPE vt[9] = { (TYPE)16, (TYPE)17, (TYPE)18, (TYPE)19, (TYPE)20, (TYPE)21,  \
                    (TYPE)22, (TYPE)23, (TYPE)24 };                             \
    int k = 0;                                                                   \
    TYPE difl[3] = { (TYPE)25, (TYPE)26, (TYPE)27 };                            \
    TYPE difr[3] = { (TYPE)28, (TYPE)29, (TYPE)30 };                            \
    TYPE z[3] = { (TYPE)31, (TYPE)32, (TYPE)33 };                               \
    TYPE zw[3] = { (TYPE)34, (TYPE)35, (TYPE)36 };                              \
    TYPE work[4] = { (TYPE)37, (TYPE)38, (TYPE)39, (TYPE)40 };                  \
    int iwork[4] = { 41, 42, 43, 44 };                                           \
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
    if (thunk(1, 3, 3, 1, d, e, u, 3, vt, 3, &k, difl, difr, z, zw, work, iwork)\
            != (BASE) + 11 ||                                                    \
        g_##SUFFIX##_fortran_call.called != 1 ||                                 \
        g_##SUFFIX##_fortran_call.icompq != 1 ||                                 \
        g_##SUFFIX##_fortran_call.smlsiz != 3 ||                                 \
        g_##SUFFIX##_fortran_call.n != 3 ||                                      \
        g_##SUFFIX##_fortran_call.sqre != 1 ||                                   \
        g_##SUFFIX##_fortran_call.d != d ||                                      \
        g_##SUFFIX##_fortran_call.e != e ||                                      \
        g_##SUFFIX##_fortran_call.u != u ||                                      \
        g_##SUFFIX##_fortran_call.ldu != 3 ||                                    \
        g_##SUFFIX##_fortran_call.vt != vt ||                                    \
        g_##SUFFIX##_fortran_call.ldvt != 3 ||                                   \
        g_##SUFFIX##_fortran_call.k != &k ||                                     \
        g_##SUFFIX##_fortran_call.difl != difl ||                                \
        g_##SUFFIX##_fortran_call.difr != difr ||                                \
        g_##SUFFIX##_fortran_call.z != z ||                                      \
        g_##SUFFIX##_fortran_call.zw != zw ||                                    \
        g_##SUFFIX##_fortran_call.work != work ||                                \
        g_##SUFFIX##_fortran_call.iwork != iwork ||                              \
        d[0] != (TYPE)((BASE) + 1) ||                                            \
        u[0] != (TYPE)((BASE) + 2) ||                                            \
        vt[0] != (TYPE)((BASE) + 3) ||                                           \
        difl[0] != (TYPE)((BASE) + 4) ||                                         \
        difr[0] != (TYPE)((BASE) + 5) ||                                         \
        z[0] != (TYPE)((BASE) + 6) ||                                            \
        zw[0] != (TYPE)((BASE) + 7) ||                                           \
        work[0] != (TYPE)((BASE) + 8) ||                                         \
        iwork[0] != (BASE) + 9 ||                                                \
        k != (BASE) + 10) {                                                      \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward LASDA inputs correctly\n"); \
        return 1;                                                                \
    }                                                                            \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards LASDA inputs\n"); \
    return 0;                                                                    \
}                                                                                \
static int check_##SUFFIX##_cblas_to_fortran(void)                               \
{                                                                                \
    fb_backend_vtable_t vtable;                                                  \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                       \
    int icompq = 1;                                                              \
    int smlsiz = 3;                                                              \
    int n = 3;                                                                   \
    int sqre = 1;                                                                \
    TYPE d[3] = { (TYPE)45, (TYPE)46, (TYPE)47 };                               \
    TYPE e[3] = { (TYPE)48, (TYPE)49, (TYPE)50 };                               \
    TYPE u[9] = { (TYPE)51, (TYPE)52, (TYPE)53, (TYPE)54, (TYPE)55, (TYPE)56,   \
                   (TYPE)57, (TYPE)58, (TYPE)59 };                              \
    int ldu = 3;                                                                 \
    TYPE vt[9] = { (TYPE)60, (TYPE)61, (TYPE)62, (TYPE)63, (TYPE)64, (TYPE)65,  \
                    (TYPE)66, (TYPE)67, (TYPE)68 };                             \
    int ldvt = 3;                                                                \
    int k = 0;                                                                   \
    TYPE difl[3] = { (TYPE)69, (TYPE)70, (TYPE)71 };                            \
    TYPE difr[3] = { (TYPE)72, (TYPE)73, (TYPE)74 };                            \
    TYPE z[3] = { (TYPE)75, (TYPE)76, (TYPE)77 };                               \
    TYPE zw[3] = { (TYPE)78, (TYPE)79, (TYPE)80 };                              \
    TYPE work[4] = { (TYPE)81, (TYPE)82, (TYPE)83, (TYPE)84 };                  \
    int iwork[4] = { 85, 86, 87, 88 };                                           \
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
    thunk(&icompq, &smlsiz, &n, &sqre, d, e, u, &ldu, vt, &ldvt, &k, difl, difr,\
          z, zw, work, iwork, &info);                                            \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                   \
        g_##SUFFIX##_cblas_call.icompq != 1 ||                                   \
        g_##SUFFIX##_cblas_call.smlsiz != 3 ||                                   \
        g_##SUFFIX##_cblas_call.n != 3 ||                                        \
        g_##SUFFIX##_cblas_call.sqre != 1 ||                                     \
        g_##SUFFIX##_cblas_call.d != d ||                                        \
        g_##SUFFIX##_cblas_call.e != e ||                                        \
        g_##SUFFIX##_cblas_call.u != u ||                                        \
        g_##SUFFIX##_cblas_call.ldu != 3 ||                                      \
        g_##SUFFIX##_cblas_call.vt != vt ||                                      \
        g_##SUFFIX##_cblas_call.ldvt != 3 ||                                     \
        g_##SUFFIX##_cblas_call.k != &k ||                                       \
        g_##SUFFIX##_cblas_call.difl != difl ||                                  \
        g_##SUFFIX##_cblas_call.difr != difr ||                                  \
        g_##SUFFIX##_cblas_call.z != z ||                                        \
        g_##SUFFIX##_cblas_call.zw != zw ||                                      \
        g_##SUFFIX##_cblas_call.work != work ||                                  \
        g_##SUFFIX##_cblas_call.iwork != iwork ||                                \
        info != (BASE) + 22 ||                                                   \
        d[0] != (TYPE)((BASE) + 12) ||                                           \
        u[0] != (TYPE)((BASE) + 13) ||                                           \
        vt[0] != (TYPE)((BASE) + 14) ||                                          \
        difl[0] != (TYPE)((BASE) + 15) ||                                        \
        difr[0] != (TYPE)((BASE) + 16) ||                                        \
        z[0] != (TYPE)((BASE) + 17) ||                                           \
        zw[0] != (TYPE)((BASE) + 18) ||                                          \
        work[0] != (TYPE)((BASE) + 19) ||                                        \
        iwork[0] != (BASE) + 20 ||                                               \
        k != (BASE) + 21) {                                                      \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference LASDA inputs correctly\n"); \
        return 1;                                                                \
    }                                                                            \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences LASDA inputs\n"); \
    return 0;                                                                    \
}

DEFINE_LASDA_TESTS(slasda, float, FB_OP_SLASDA, 100)
DEFINE_LASDA_TESTS(dlasda, double, FB_OP_DLASDA, 300)

int main(void)
{
    int status = 0;

    status |= check_slasda_fortran_to_cblas();
    status |= check_slasda_cblas_to_fortran();
    status |= check_dlasda_fortran_to_cblas();
    status |= check_dlasda_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}