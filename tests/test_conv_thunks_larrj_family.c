#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

#define DEFINE_LARRJ_TESTS(SUFFIX, TYPE, OP_ID, BASE)                          \
typedef int (*fb_##SUFFIX##_cblas_fn)(                                         \
    int n, const TYPE *d, const TYPE *e2, int ifirst, int ilast, TYPE rtol,   \
    int offset, TYPE *w, TYPE *werr, TYPE *work, int *iwork, TYPE pivmin,     \
    TYPE spdiam);                                                               \
typedef void (*fb_##SUFFIX##_fortran_fn)(                                      \
    int *n, TYPE *d, TYPE *e2, int *ifirst, int *ilast, TYPE *rtol,           \
    int *offset, TYPE *w, TYPE *werr, TYPE *work, int *iwork, TYPE *pivmin,   \
    TYPE *spdiam, int *info);                                                   \
static struct {                                                                 \
    int called;                                                                 \
    int n;                                                                      \
    TYPE *d;                                                                    \
    TYPE *e2;                                                                   \
    int ifirst;                                                                 \
    int ilast;                                                                  \
    TYPE rtol;                                                                  \
    int offset;                                                                 \
    TYPE *w;                                                                    \
    TYPE *werr;                                                                 \
    TYPE *work;                                                                 \
    int *iwork;                                                                 \
    TYPE pivmin;                                                                \
    TYPE spdiam;                                                                \
    int *info_ptr;                                                              \
} g_##SUFFIX##_fortran_call;                                                    \
static struct {                                                                 \
    int called;                                                                 \
    int n;                                                                      \
    const TYPE *d;                                                              \
    const TYPE *e2;                                                             \
    int ifirst;                                                                 \
    int ilast;                                                                  \
    TYPE rtol;                                                                  \
    int offset;                                                                 \
    TYPE *w;                                                                    \
    TYPE *werr;                                                                 \
    TYPE *work;                                                                 \
    int *iwork;                                                                 \
    TYPE pivmin;                                                                \
    TYPE spdiam;                                                                \
} g_##SUFFIX##_cblas_call;                                                      \
static void stub_##SUFFIX##_fortran(                                            \
    int *n, TYPE *d, TYPE *e2, int *ifirst, int *ilast, TYPE *rtol,           \
    int *offset, TYPE *w, TYPE *werr, TYPE *work, int *iwork, TYPE *pivmin,   \
    TYPE *spdiam, int *info)                                                    \
{                                                                               \
    g_##SUFFIX##_fortran_call.called += 1;                                      \
    g_##SUFFIX##_fortran_call.n = *n;                                           \
    g_##SUFFIX##_fortran_call.d = d;                                            \
    g_##SUFFIX##_fortran_call.e2 = e2;                                          \
    g_##SUFFIX##_fortran_call.ifirst = *ifirst;                                 \
    g_##SUFFIX##_fortran_call.ilast = *ilast;                                   \
    g_##SUFFIX##_fortran_call.rtol = *rtol;                                     \
    g_##SUFFIX##_fortran_call.offset = *offset;                                 \
    g_##SUFFIX##_fortran_call.w = w;                                            \
    g_##SUFFIX##_fortran_call.werr = werr;                                      \
    g_##SUFFIX##_fortran_call.work = work;                                      \
    g_##SUFFIX##_fortran_call.iwork = iwork;                                    \
    g_##SUFFIX##_fortran_call.pivmin = *pivmin;                                 \
    g_##SUFFIX##_fortran_call.spdiam = *spdiam;                                 \
    g_##SUFFIX##_fortran_call.info_ptr = info;                                  \
    w[0] = (TYPE)((BASE) + 1);                                                  \
    werr[0] = (TYPE)((BASE) + 2);                                               \
    work[0] = (TYPE)((BASE) + 3);                                               \
    iwork[0] = (BASE) + 4;                                                      \
    *info = (BASE) + 5;                                                         \
}                                                                               \
static int stub_##SUFFIX##_cblas(                                               \
    int n, const TYPE *d, const TYPE *e2, int ifirst, int ilast, TYPE rtol,   \
    int offset, TYPE *w, TYPE *werr, TYPE *work, int *iwork, TYPE pivmin,     \
    TYPE spdiam)                                                                \
{                                                                               \
    g_##SUFFIX##_cblas_call.called += 1;                                        \
    g_##SUFFIX##_cblas_call.n = n;                                              \
    g_##SUFFIX##_cblas_call.d = d;                                              \
    g_##SUFFIX##_cblas_call.e2 = e2;                                            \
    g_##SUFFIX##_cblas_call.ifirst = ifirst;                                    \
    g_##SUFFIX##_cblas_call.ilast = ilast;                                      \
    g_##SUFFIX##_cblas_call.rtol = rtol;                                        \
    g_##SUFFIX##_cblas_call.offset = offset;                                    \
    g_##SUFFIX##_cblas_call.w = w;                                              \
    g_##SUFFIX##_cblas_call.werr = werr;                                        \
    g_##SUFFIX##_cblas_call.work = work;                                        \
    g_##SUFFIX##_cblas_call.iwork = iwork;                                      \
    g_##SUFFIX##_cblas_call.pivmin = pivmin;                                    \
    g_##SUFFIX##_cblas_call.spdiam = spdiam;                                    \
    w[0] = (TYPE)((BASE) + 6);                                                  \
    werr[0] = (TYPE)((BASE) + 7);                                               \
    work[0] = (TYPE)((BASE) + 8);                                               \
    iwork[0] = (BASE) + 9;                                                      \
    return (BASE) + 10;                                                         \
}                                                                               \
static int check_##SUFFIX##_fortran_to_cblas(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                        \
    TYPE d[2] = { (TYPE)1, (TYPE)2 };                                           \
    TYPE e2[2] = { (TYPE)3, (TYPE)4 };                                          \
    TYPE w[2] = { (TYPE)5, (TYPE)6 };                                           \
    TYPE werr[2] = { (TYPE)7, (TYPE)8 };                                        \
    TYPE work[2] = { (TYPE)9, (TYPE)10 };                                       \
    int iwork[2] = { 11, 12 };                                                  \
    memset(&vtable, 0, sizeof(vtable));                                         \
    memset(&g_##SUFFIX##_fortran_call, 0, sizeof(g_##SUFFIX##_fortran_call));   \
    vtable.ext_ops[OP_ID][FB_CONV_FORTRAN] =                                    \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_fortran;                 \
    fb_install_conv_thunks(&vtable, OP_ID);                                     \
    thunk = (fb_##SUFFIX##_cblas_fn)vtable.ext_ops[OP_ID][FB_CONV_CBLAS];       \
    if (!thunk) {                                                               \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk was not installed\n"); \
        return 1;                                                               \
    }                                                                           \
    if (thunk(2, d, e2, 1, 2, (TYPE)13, 0, w, werr, work, iwork,              \
              (TYPE)14, (TYPE)15) != (BASE) + 5 ||                             \
        g_##SUFFIX##_fortran_call.called != 1 ||                                \
        g_##SUFFIX##_fortran_call.n != 2 ||                                     \
        g_##SUFFIX##_fortran_call.d != d ||                                     \
        g_##SUFFIX##_fortran_call.e2 != e2 ||                                   \
        g_##SUFFIX##_fortran_call.ifirst != 1 ||                                \
        g_##SUFFIX##_fortran_call.ilast != 2 ||                                 \
        g_##SUFFIX##_fortran_call.rtol != (TYPE)13 ||                           \
        g_##SUFFIX##_fortran_call.offset != 0 ||                                \
        g_##SUFFIX##_fortran_call.w != w ||                                     \
        g_##SUFFIX##_fortran_call.werr != werr ||                               \
        g_##SUFFIX##_fortran_call.work != work ||                               \
        g_##SUFFIX##_fortran_call.iwork != iwork ||                             \
        g_##SUFFIX##_fortran_call.pivmin != (TYPE)14 ||                         \
        g_##SUFFIX##_fortran_call.spdiam != (TYPE)15 ||                         \
        g_##SUFFIX##_fortran_call.info_ptr == NULL ||                           \
        w[0] != (TYPE)((BASE) + 1) || werr[0] != (TYPE)((BASE) + 2) ||          \
        work[0] != (TYPE)((BASE) + 3) || iwork[0] != (BASE) + 4) {              \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward LARRJ inputs correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards LARRJ inputs and returns info\n"); \
    return 0;                                                                   \
}                                                                               \
static int check_##SUFFIX##_cblas_to_fortran(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                      \
    int n = 2;                                                                  \
    int ifirst = 1;                                                             \
    int ilast = 2;                                                              \
    int offset = 0;                                                             \
    TYPE d[2] = { (TYPE)1, (TYPE)2 };                                           \
    TYPE e2[2] = { (TYPE)3, (TYPE)4 };                                          \
    TYPE w[2] = { (TYPE)5, (TYPE)6 };                                           \
    TYPE werr[2] = { (TYPE)7, (TYPE)8 };                                        \
    TYPE work[2] = { (TYPE)9, (TYPE)10 };                                       \
    int iwork[2] = { 11, 12 };                                                  \
    TYPE rtol = (TYPE)13;                                                       \
    TYPE pivmin = (TYPE)14;                                                     \
    TYPE spdiam = (TYPE)15;                                                     \
    int info = -1;                                                              \
    memset(&vtable, 0, sizeof(vtable));                                         \
    memset(&g_##SUFFIX##_cblas_call, 0, sizeof(g_##SUFFIX##_cblas_call));       \
    vtable.ext_ops[OP_ID][FB_CONV_CBLAS] =                                      \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_cblas;                   \
    fb_install_conv_thunks(&vtable, OP_ID);                                     \
    thunk = (fb_##SUFFIX##_fortran_fn)vtable.ext_ops[OP_ID][FB_CONV_FORTRAN];   \
    if (!thunk) {                                                               \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk was not installed\n"); \
        return 1;                                                               \
    }                                                                           \
    thunk(&n, d, e2, &ifirst, &ilast, &rtol, &offset, w, werr, work, iwork,    \
          &pivmin, &spdiam, &info);                                             \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                  \
        g_##SUFFIX##_cblas_call.n != 2 ||                                       \
        g_##SUFFIX##_cblas_call.d != d ||                                       \
        g_##SUFFIX##_cblas_call.e2 != e2 ||                                     \
        g_##SUFFIX##_cblas_call.ifirst != 1 ||                                  \
        g_##SUFFIX##_cblas_call.ilast != 2 ||                                   \
        g_##SUFFIX##_cblas_call.rtol != (TYPE)13 ||                             \
        g_##SUFFIX##_cblas_call.offset != 0 ||                                  \
        g_##SUFFIX##_cblas_call.w != w ||                                       \
        g_##SUFFIX##_cblas_call.werr != werr ||                                 \
        g_##SUFFIX##_cblas_call.work != work ||                                 \
        g_##SUFFIX##_cblas_call.iwork != iwork ||                               \
        g_##SUFFIX##_cblas_call.pivmin != (TYPE)14 ||                           \
        g_##SUFFIX##_cblas_call.spdiam != (TYPE)15 ||                           \
        w[0] != (TYPE)((BASE) + 6) || werr[0] != (TYPE)((BASE) + 7) ||          \
        work[0] != (TYPE)((BASE) + 8) || iwork[0] != (BASE) + 9 ||              \
        info != (BASE) + 10) {                                                  \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference LARRJ inputs correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences LARRJ inputs and stores info\n"); \
    return 0;                                                                   \
}

DEFINE_LARRJ_TESTS(slarrj, float, FB_OP_SLARRJ, 100)
DEFINE_LARRJ_TESTS(dlarrj, double, FB_OP_DLARRJ, 300)

int main(void)
{
    int status = 0;

    status |= check_slarrj_fortran_to_cblas();
    status |= check_slarrj_cblas_to_fortran();
    status |= check_dlarrj_fortran_to_cblas();
    status |= check_dlarrj_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}