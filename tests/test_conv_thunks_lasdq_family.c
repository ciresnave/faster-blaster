#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

#define DEFINE_LASDQ_TESTS(SUFFIX, TYPE, OP_ID, BASE)                            \
typedef int (*fb_##SUFFIX##_cblas_fn)(char uplo, int sqre, int n, int ncvt,     \
                                      int nru, int ncc, TYPE *d, TYPE *e,       \
                                      TYPE *vt, int ldvt, TYPE *u, int ldu,     \
                                      TYPE *c, int ldc, TYPE *work);            \
typedef void (*fb_##SUFFIX##_fortran_fn)(char *uplo, int *sqre, int *n,         \
                                         int *ncvt, int *nru, int *ncc,         \
                                         TYPE *d, TYPE *e, TYPE *vt, int *ldvt, \
                                         TYPE *u, int *ldu, TYPE *c, int *ldc,  \
                                         TYPE *work, int *info);                \
static struct {                                                                  \
    int called;                                                                  \
    char uplo;                                                                   \
    int sqre;                                                                    \
    int n;                                                                       \
    int ncvt;                                                                    \
    int nru;                                                                     \
    int ncc;                                                                     \
    TYPE *d;                                                                     \
    TYPE *e;                                                                     \
    TYPE *vt;                                                                    \
    int ldvt;                                                                    \
    TYPE *u;                                                                     \
    int ldu;                                                                     \
    TYPE *c;                                                                     \
    int ldc;                                                                     \
    TYPE *work;                                                                  \
} g_##SUFFIX##_fortran_call;                                                     \
static struct {                                                                  \
    int called;                                                                  \
    char uplo;                                                                   \
    int sqre;                                                                    \
    int n;                                                                       \
    int ncvt;                                                                    \
    int nru;                                                                     \
    int ncc;                                                                     \
    TYPE *d;                                                                     \
    TYPE *e;                                                                     \
    TYPE *vt;                                                                    \
    int ldvt;                                                                    \
    TYPE *u;                                                                     \
    int ldu;                                                                     \
    TYPE *c;                                                                     \
    int ldc;                                                                     \
    TYPE *work;                                                                  \
} g_##SUFFIX##_cblas_call;                                                       \
static void stub_##SUFFIX##_fortran(char *uplo, int *sqre, int *n, int *ncvt,   \
                                    int *nru, int *ncc, TYPE *d, TYPE *e,       \
                                    TYPE *vt, int *ldvt, TYPE *u, int *ldu,     \
                                    TYPE *c, int *ldc, TYPE *work, int *info)   \
{                                                                                \
    g_##SUFFIX##_fortran_call.called += 1;                                       \
    g_##SUFFIX##_fortran_call.uplo = *uplo;                                      \
    g_##SUFFIX##_fortran_call.sqre = *sqre;                                      \
    g_##SUFFIX##_fortran_call.n = *n;                                            \
    g_##SUFFIX##_fortran_call.ncvt = *ncvt;                                      \
    g_##SUFFIX##_fortran_call.nru = *nru;                                        \
    g_##SUFFIX##_fortran_call.ncc = *ncc;                                        \
    g_##SUFFIX##_fortran_call.d = d;                                             \
    g_##SUFFIX##_fortran_call.e = e;                                             \
    g_##SUFFIX##_fortran_call.vt = vt;                                           \
    g_##SUFFIX##_fortran_call.ldvt = *ldvt;                                      \
    g_##SUFFIX##_fortran_call.u = u;                                             \
    g_##SUFFIX##_fortran_call.ldu = *ldu;                                        \
    g_##SUFFIX##_fortran_call.c = c;                                             \
    g_##SUFFIX##_fortran_call.ldc = *ldc;                                        \
    g_##SUFFIX##_fortran_call.work = work;                                       \
    d[0] = (TYPE)((BASE) + 1);                                                   \
    vt[0] = (TYPE)((BASE) + 2);                                                  \
    u[0] = (TYPE)((BASE) + 3);                                                   \
    c[0] = (TYPE)((BASE) + 4);                                                   \
    work[0] = (TYPE)((BASE) + 5);                                                \
    *info = (BASE) + 6;                                                          \
}                                                                                \
static int stub_##SUFFIX##_cblas(char uplo, int sqre, int n, int ncvt, int nru, \
                                 int ncc, TYPE *d, TYPE *e, TYPE *vt,          \
                                 int ldvt, TYPE *u, int ldu, TYPE *c, int ldc, \
                                 TYPE *work) {                                  \
    g_##SUFFIX##_cblas_call.called += 1;                                        \
    g_##SUFFIX##_cblas_call.uplo = uplo;                                        \
    g_##SUFFIX##_cblas_call.sqre = sqre;                                        \
    g_##SUFFIX##_cblas_call.n = n;                                              \
    g_##SUFFIX##_cblas_call.ncvt = ncvt;                                        \
    g_##SUFFIX##_cblas_call.nru = nru;                                          \
    g_##SUFFIX##_cblas_call.ncc = ncc;                                          \
    g_##SUFFIX##_cblas_call.d = d;                                              \
    g_##SUFFIX##_cblas_call.e = e;                                              \
    g_##SUFFIX##_cblas_call.vt = vt;                                            \
    g_##SUFFIX##_cblas_call.ldvt = ldvt;                                        \
    g_##SUFFIX##_cblas_call.u = u;                                              \
    g_##SUFFIX##_cblas_call.ldu = ldu;                                          \
    g_##SUFFIX##_cblas_call.c = c;                                              \
    g_##SUFFIX##_cblas_call.ldc = ldc;                                          \
    g_##SUFFIX##_cblas_call.work = work;                                        \
    d[0] = (TYPE)((BASE) + 7);                                                  \
    vt[0] = (TYPE)((BASE) + 8);                                                 \
    u[0] = (TYPE)((BASE) + 9);                                                  \
    c[0] = (TYPE)((BASE) + 10);                                                 \
    work[0] = (TYPE)((BASE) + 11);                                              \
    return (BASE) + 12;                                                         \
}                                                                                \
static int check_##SUFFIX##_fortran_to_cblas(void)                               \
{                                                                                \
    fb_backend_vtable_t vtable;                                                  \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                         \
    TYPE d[2] = { (TYPE)1, (TYPE)2 };                                            \
    TYPE e[2] = { (TYPE)3, (TYPE)4 };                                            \
    TYPE vt[2] = { (TYPE)5, (TYPE)6 };                                           \
    TYPE u[1] = { (TYPE)7 };                                                     \
    TYPE c[2] = { (TYPE)8, (TYPE)9 };                                            \
    TYPE work[4] = { (TYPE)10, (TYPE)11, (TYPE)12, (TYPE)13 };                  \
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
    if (thunk('U', 1, 2, 1, 1, 1, d, e, vt, 2, u, 1, c, 2, work)               \
            != (BASE) + 6 ||                                                     \
        g_##SUFFIX##_fortran_call.called != 1 ||                                 \
        g_##SUFFIX##_fortran_call.uplo != 'U' ||                                 \
        g_##SUFFIX##_fortran_call.sqre != 1 ||                                   \
        g_##SUFFIX##_fortran_call.n != 2 ||                                      \
        g_##SUFFIX##_fortran_call.ncvt != 1 ||                                   \
        g_##SUFFIX##_fortran_call.nru != 1 ||                                    \
        g_##SUFFIX##_fortran_call.ncc != 1 ||                                    \
        g_##SUFFIX##_fortran_call.d != d ||                                      \
        g_##SUFFIX##_fortran_call.e != e ||                                      \
        g_##SUFFIX##_fortran_call.vt != vt ||                                    \
        g_##SUFFIX##_fortran_call.ldvt != 2 ||                                   \
        g_##SUFFIX##_fortran_call.u != u ||                                      \
        g_##SUFFIX##_fortran_call.ldu != 1 ||                                    \
        g_##SUFFIX##_fortran_call.c != c ||                                      \
        g_##SUFFIX##_fortran_call.ldc != 2 ||                                    \
        g_##SUFFIX##_fortran_call.work != work ||                                \
        d[0] != (TYPE)((BASE) + 1) ||                                            \
        vt[0] != (TYPE)((BASE) + 2) ||                                           \
        u[0] != (TYPE)((BASE) + 3) ||                                            \
        c[0] != (TYPE)((BASE) + 4) ||                                            \
        work[0] != (TYPE)((BASE) + 5)) {                                         \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward LASDQ inputs correctly\n"); \
        return 1;                                                                \
    }                                                                            \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards LASDQ inputs\n"); \
    return 0;                                                                    \
}                                                                                \
static int check_##SUFFIX##_cblas_to_fortran(void)                               \
{                                                                                \
    fb_backend_vtable_t vtable;                                                  \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                       \
    char uplo = 'U';                                                             \
    int sqre = 1;                                                                \
    int n = 2;                                                                   \
    int ncvt = 1;                                                                \
    int nru = 1;                                                                 \
    int ncc = 1;                                                                 \
    TYPE d[2] = { (TYPE)14, (TYPE)15 };                                          \
    TYPE e[2] = { (TYPE)16, (TYPE)17 };                                          \
    TYPE vt[2] = { (TYPE)18, (TYPE)19 };                                         \
    int ldvt = 2;                                                                \
    TYPE u[1] = { (TYPE)20 };                                                    \
    int ldu = 1;                                                                 \
    TYPE c[2] = { (TYPE)21, (TYPE)22 };                                          \
    int ldc = 2;                                                                 \
    TYPE work[4] = { (TYPE)23, (TYPE)24, (TYPE)25, (TYPE)26 };                  \
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
    thunk(&uplo, &sqre, &n, &ncvt, &nru, &ncc, d, e, vt, &ldvt, u, &ldu, c,     \
          &ldc, work, &info);                                                    \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                   \
        g_##SUFFIX##_cblas_call.uplo != 'U' ||                                   \
        g_##SUFFIX##_cblas_call.sqre != 1 ||                                     \
        g_##SUFFIX##_cblas_call.n != 2 ||                                        \
        g_##SUFFIX##_cblas_call.ncvt != 1 ||                                     \
        g_##SUFFIX##_cblas_call.nru != 1 ||                                      \
        g_##SUFFIX##_cblas_call.ncc != 1 ||                                      \
        g_##SUFFIX##_cblas_call.d != d ||                                        \
        g_##SUFFIX##_cblas_call.e != e ||                                        \
        g_##SUFFIX##_cblas_call.vt != vt ||                                      \
        g_##SUFFIX##_cblas_call.ldvt != 2 ||                                     \
        g_##SUFFIX##_cblas_call.u != u ||                                        \
        g_##SUFFIX##_cblas_call.ldu != 1 ||                                      \
        g_##SUFFIX##_cblas_call.c != c ||                                        \
        g_##SUFFIX##_cblas_call.ldc != 2 ||                                      \
        g_##SUFFIX##_cblas_call.work != work ||                                  \
        info != (BASE) + 12 ||                                                   \
        d[0] != (TYPE)((BASE) + 7) ||                                            \
        vt[0] != (TYPE)((BASE) + 8) ||                                           \
        u[0] != (TYPE)((BASE) + 9) ||                                            \
        c[0] != (TYPE)((BASE) + 10) ||                                           \
        work[0] != (TYPE)((BASE) + 11)) {                                        \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference LASDQ inputs correctly\n"); \
        return 1;                                                                \
    }                                                                            \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences LASDQ inputs\n"); \
    return 0;                                                                    \
}

DEFINE_LASDQ_TESTS(slasdq, float, FB_OP_SLASDQ, 100)
DEFINE_LASDQ_TESTS(dlasdq, double, FB_OP_DLASDQ, 300)

int main(void)
{
    int status = 0;

    status |= check_slasdq_fortran_to_cblas();
    status |= check_slasdq_cblas_to_fortran();
    status |= check_dlasdq_fortran_to_cblas();
    status |= check_dlasdq_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}