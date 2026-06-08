#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

#define DEFINE_LARRB_TESTS(SUFFIX, TYPE, OP_ID, BASE)                          \
typedef int (*fb_##SUFFIX##_cblas_fn)(                                         \
    int n, const TYPE *d, const TYPE *lld, int ifirst, int ilast, TYPE rtol1, \
    TYPE rtol2, int offset, TYPE *w, TYPE *wgap, TYPE *werr, TYPE *work,      \
    int *iwork, TYPE pivmin, TYPE spdiam, int twist);                          \
typedef void (*fb_##SUFFIX##_fortran_fn)(                                      \
    int *n, TYPE *d, TYPE *lld, int *ifirst, int *ilast, TYPE *rtol1,         \
    TYPE *rtol2, int *offset, TYPE *w, TYPE *wgap, TYPE *werr, TYPE *work,    \
    int *iwork, TYPE *pivmin, TYPE *spdiam, int *twist, int *info);           \
static struct {                                                                 \
    int called;                                                                 \
    int n;                                                                      \
    TYPE *d;                                                                    \
    TYPE *lld;                                                                  \
    int ifirst;                                                                 \
    int ilast;                                                                  \
    TYPE rtol1;                                                                 \
    TYPE rtol2;                                                                 \
    int offset;                                                                 \
    TYPE *w;                                                                    \
    TYPE *wgap;                                                                 \
    TYPE *werr;                                                                 \
    TYPE *work;                                                                 \
    int *iwork;                                                                 \
    TYPE pivmin;                                                                \
    TYPE spdiam;                                                                \
    int twist;                                                                  \
    int *info_ptr;                                                              \
} g_##SUFFIX##_fortran_call;                                                    \
static struct {                                                                 \
    int called;                                                                 \
    int n;                                                                      \
    const TYPE *d;                                                              \
    const TYPE *lld;                                                            \
    int ifirst;                                                                 \
    int ilast;                                                                  \
    TYPE rtol1;                                                                 \
    TYPE rtol2;                                                                 \
    int offset;                                                                 \
    TYPE *w;                                                                    \
    TYPE *wgap;                                                                 \
    TYPE *werr;                                                                 \
    TYPE *work;                                                                 \
    int *iwork;                                                                 \
    TYPE pivmin;                                                                \
    TYPE spdiam;                                                                \
    int twist;                                                                  \
} g_##SUFFIX##_cblas_call;                                                      \
static void stub_##SUFFIX##_fortran(                                            \
    int *n, TYPE *d, TYPE *lld, int *ifirst, int *ilast, TYPE *rtol1,         \
    TYPE *rtol2, int *offset, TYPE *w, TYPE *wgap, TYPE *werr, TYPE *work,    \
    int *iwork, TYPE *pivmin, TYPE *spdiam, int *twist, int *info)            \
{                                                                               \
    g_##SUFFIX##_fortran_call.called += 1;                                      \
    g_##SUFFIX##_fortran_call.n = *n;                                           \
    g_##SUFFIX##_fortran_call.d = d;                                            \
    g_##SUFFIX##_fortran_call.lld = lld;                                        \
    g_##SUFFIX##_fortran_call.ifirst = *ifirst;                                 \
    g_##SUFFIX##_fortran_call.ilast = *ilast;                                   \
    g_##SUFFIX##_fortran_call.rtol1 = *rtol1;                                   \
    g_##SUFFIX##_fortran_call.rtol2 = *rtol2;                                   \
    g_##SUFFIX##_fortran_call.offset = *offset;                                 \
    g_##SUFFIX##_fortran_call.w = w;                                            \
    g_##SUFFIX##_fortran_call.wgap = wgap;                                      \
    g_##SUFFIX##_fortran_call.werr = werr;                                      \
    g_##SUFFIX##_fortran_call.work = work;                                      \
    g_##SUFFIX##_fortran_call.iwork = iwork;                                    \
    g_##SUFFIX##_fortran_call.pivmin = *pivmin;                                 \
    g_##SUFFIX##_fortran_call.spdiam = *spdiam;                                 \
    g_##SUFFIX##_fortran_call.twist = *twist;                                   \
    g_##SUFFIX##_fortran_call.info_ptr = info;                                  \
    w[0] = (TYPE)((BASE) + 1);                                                  \
    wgap[0] = (TYPE)((BASE) + 2);                                               \
    werr[0] = (TYPE)((BASE) + 3);                                               \
    work[0] = (TYPE)((BASE) + 4);                                               \
    iwork[0] = (BASE) + 5;                                                      \
    *info = (BASE) + 6;                                                         \
}                                                                               \
static int stub_##SUFFIX##_cblas(                                               \
    int n, const TYPE *d, const TYPE *lld, int ifirst, int ilast, TYPE rtol1, \
    TYPE rtol2, int offset, TYPE *w, TYPE *wgap, TYPE *werr, TYPE *work,      \
    int *iwork, TYPE pivmin, TYPE spdiam, int twist)                           \
{                                                                               \
    g_##SUFFIX##_cblas_call.called += 1;                                        \
    g_##SUFFIX##_cblas_call.n = n;                                              \
    g_##SUFFIX##_cblas_call.d = d;                                              \
    g_##SUFFIX##_cblas_call.lld = lld;                                          \
    g_##SUFFIX##_cblas_call.ifirst = ifirst;                                    \
    g_##SUFFIX##_cblas_call.ilast = ilast;                                      \
    g_##SUFFIX##_cblas_call.rtol1 = rtol1;                                      \
    g_##SUFFIX##_cblas_call.rtol2 = rtol2;                                      \
    g_##SUFFIX##_cblas_call.offset = offset;                                    \
    g_##SUFFIX##_cblas_call.w = w;                                              \
    g_##SUFFIX##_cblas_call.wgap = wgap;                                        \
    g_##SUFFIX##_cblas_call.werr = werr;                                        \
    g_##SUFFIX##_cblas_call.work = work;                                        \
    g_##SUFFIX##_cblas_call.iwork = iwork;                                      \
    g_##SUFFIX##_cblas_call.pivmin = pivmin;                                    \
    g_##SUFFIX##_cblas_call.spdiam = spdiam;                                    \
    g_##SUFFIX##_cblas_call.twist = twist;                                      \
    w[0] = (TYPE)((BASE) + 11);                                                 \
    wgap[0] = (TYPE)((BASE) + 12);                                              \
    werr[0] = (TYPE)((BASE) + 13);                                              \
    work[0] = (TYPE)((BASE) + 14);                                              \
    iwork[0] = (BASE) + 15;                                                     \
    return (BASE) + 16;                                                         \
}                                                                               \
static int check_##SUFFIX##_fortran_to_cblas(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                        \
    TYPE d[2] = { (TYPE)1, (TYPE)2 };                                           \
    TYPE lld[2] = { (TYPE)3, (TYPE)4 };                                         \
    TYPE w[2] = { (TYPE)5, (TYPE)6 };                                           \
    TYPE wgap[2] = { (TYPE)7, (TYPE)8 };                                        \
    TYPE werr[2] = { (TYPE)9, (TYPE)10 };                                       \
    TYPE work[2] = { (TYPE)11, (TYPE)12 };                                      \
    int iwork[2] = { 13, 14 };                                                  \
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
    if (thunk(2, d, lld, 3, 4, (TYPE)15, (TYPE)16, 5, w, wgap, werr, work,     \
              iwork, (TYPE)17, (TYPE)18, 6) != (BASE) + 6 ||                   \
        g_##SUFFIX##_fortran_call.called != 1 ||                                \
        g_##SUFFIX##_fortran_call.n != 2 ||                                     \
        g_##SUFFIX##_fortran_call.d != d ||                                     \
        g_##SUFFIX##_fortran_call.lld != lld ||                                 \
        g_##SUFFIX##_fortran_call.ifirst != 3 ||                                \
        g_##SUFFIX##_fortran_call.ilast != 4 ||                                 \
        g_##SUFFIX##_fortran_call.rtol1 != (TYPE)15 ||                          \
        g_##SUFFIX##_fortran_call.rtol2 != (TYPE)16 ||                          \
        g_##SUFFIX##_fortran_call.offset != 5 ||                                \
        g_##SUFFIX##_fortran_call.w != w ||                                     \
        g_##SUFFIX##_fortran_call.wgap != wgap ||                               \
        g_##SUFFIX##_fortran_call.werr != werr ||                               \
        g_##SUFFIX##_fortran_call.work != work ||                               \
        g_##SUFFIX##_fortran_call.iwork != iwork ||                             \
        g_##SUFFIX##_fortran_call.pivmin != (TYPE)17 ||                         \
        g_##SUFFIX##_fortran_call.spdiam != (TYPE)18 ||                         \
        g_##SUFFIX##_fortran_call.twist != 6 ||                                 \
        g_##SUFFIX##_fortran_call.info_ptr == NULL ||                           \
        w[0] != (TYPE)((BASE) + 1) ||                                           \
        wgap[0] != (TYPE)((BASE) + 2) ||                                        \
        werr[0] != (TYPE)((BASE) + 3) ||                                        \
        work[0] != (TYPE)((BASE) + 4) ||                                        \
        iwork[0] != (BASE) + 5) {                                               \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward LARRB inputs correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards LARRB inputs and returns info\n"); \
    return 0;                                                                   \
}                                                                               \
static int check_##SUFFIX##_cblas_to_fortran(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                      \
    int n = 2;                                                                  \
    int ifirst = 3;                                                             \
    int ilast = 4;                                                              \
    TYPE rtol1 = (TYPE)15;                                                      \
    TYPE rtol2 = (TYPE)16;                                                      \
    int offset = 5;                                                             \
    TYPE d[2] = { (TYPE)21, (TYPE)22 };                                         \
    TYPE lld[2] = { (TYPE)23, (TYPE)24 };                                       \
    TYPE w[2] = { (TYPE)25, (TYPE)26 };                                         \
    TYPE wgap[2] = { (TYPE)27, (TYPE)28 };                                      \
    TYPE werr[2] = { (TYPE)29, (TYPE)30 };                                      \
    TYPE work[2] = { (TYPE)31, (TYPE)32 };                                      \
    int iwork[2] = { 33, 34 };                                                  \
    TYPE pivmin = (TYPE)17;                                                     \
    TYPE spdiam = (TYPE)18;                                                     \
    int twist = 6;                                                              \
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
    thunk(&n, d, lld, &ifirst, &ilast, &rtol1, &rtol2, &offset, w, wgap, werr, \
          work, iwork, &pivmin, &spdiam, &twist, &info);                        \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                  \
        g_##SUFFIX##_cblas_call.n != 2 ||                                       \
        g_##SUFFIX##_cblas_call.d != d ||                                       \
        g_##SUFFIX##_cblas_call.lld != lld ||                                   \
        g_##SUFFIX##_cblas_call.ifirst != 3 ||                                  \
        g_##SUFFIX##_cblas_call.ilast != 4 ||                                   \
        g_##SUFFIX##_cblas_call.rtol1 != (TYPE)15 ||                            \
        g_##SUFFIX##_cblas_call.rtol2 != (TYPE)16 ||                            \
        g_##SUFFIX##_cblas_call.offset != 5 ||                                  \
        g_##SUFFIX##_cblas_call.w != w ||                                       \
        g_##SUFFIX##_cblas_call.wgap != wgap ||                                 \
        g_##SUFFIX##_cblas_call.werr != werr ||                                 \
        g_##SUFFIX##_cblas_call.work != work ||                                 \
        g_##SUFFIX##_cblas_call.iwork != iwork ||                               \
        g_##SUFFIX##_cblas_call.pivmin != (TYPE)17 ||                           \
        g_##SUFFIX##_cblas_call.spdiam != (TYPE)18 ||                           \
        g_##SUFFIX##_cblas_call.twist != 6 ||                                   \
        info != (BASE) + 16 ||                                                  \
        w[0] != (TYPE)((BASE) + 11) ||                                          \
        wgap[0] != (TYPE)((BASE) + 12) ||                                       \
        werr[0] != (TYPE)((BASE) + 13) ||                                       \
        work[0] != (TYPE)((BASE) + 14) ||                                       \
        iwork[0] != (BASE) + 15) {                                              \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference LARRB inputs correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences LARRB inputs and stores info\n"); \
    return 0;                                                                   \
}

DEFINE_LARRB_TESTS(slarrb, float, FB_OP_SLARRB, 100)
DEFINE_LARRB_TESTS(dlarrb, double, FB_OP_DLARRB, 300)

int main(void)
{
    int status = 0;

    status |= check_slarrb_fortran_to_cblas();
    status |= check_slarrb_cblas_to_fortran();
    status |= check_dlarrb_fortran_to_cblas();
    status |= check_dlarrb_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}