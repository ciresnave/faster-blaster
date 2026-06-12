#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

#define DEFINE_LARRF_TESTS(SUFFIX, TYPE, OP_ID, BASE)                          \
typedef int (*fb_##SUFFIX##_cblas_fn)(                                         \
    int n, const TYPE *d, const TYPE *l, const TYPE *ld, int clstrt,          \
    int clend, const TYPE *w, TYPE *wgap, const TYPE *werr, TYPE spdiam,      \
    TYPE clgapl, TYPE clgapr, TYPE pivmin, TYPE *sigma, TYPE *dplus,          \
    TYPE *lplus, TYPE *work);                                                  \
typedef void (*fb_##SUFFIX##_fortran_fn)(                                      \
    int *n, TYPE *d, TYPE *l, TYPE *ld, int *clstrt, int *clend, TYPE *w,     \
    TYPE *wgap, TYPE *werr, TYPE *spdiam, TYPE *clgapl, TYPE *clgapr,         \
    TYPE *pivmin, TYPE *sigma, TYPE *dplus, TYPE *lplus, TYPE *work,          \
    int *info);                                                                \
static struct {                                                                 \
    int called;                                                                 \
    int n;                                                                      \
    TYPE *d;                                                                    \
    TYPE *l;                                                                    \
    TYPE *ld;                                                                   \
    int clstrt;                                                                 \
    int clend;                                                                  \
    TYPE *w;                                                                    \
    TYPE *wgap;                                                                 \
    TYPE *werr;                                                                 \
    TYPE spdiam;                                                                \
    TYPE clgapl;                                                                \
    TYPE clgapr;                                                                \
    TYPE pivmin;                                                                \
    TYPE *sigma;                                                                \
    TYPE *dplus;                                                                \
    TYPE *lplus;                                                                \
    TYPE *work;                                                                 \
    int *info_ptr;                                                              \
} g_##SUFFIX##_fortran_call;                                                    \
static struct {                                                                 \
    int called;                                                                 \
    int n;                                                                      \
    const TYPE *d;                                                              \
    const TYPE *l;                                                              \
    const TYPE *ld;                                                             \
    int clstrt;                                                                 \
    int clend;                                                                  \
    const TYPE *w;                                                              \
    TYPE *wgap;                                                                 \
    const TYPE *werr;                                                           \
    TYPE spdiam;                                                                \
    TYPE clgapl;                                                                \
    TYPE clgapr;                                                                \
    TYPE pivmin;                                                                \
    TYPE *sigma;                                                                \
    TYPE *dplus;                                                                \
    TYPE *lplus;                                                                \
    TYPE *work;                                                                 \
} g_##SUFFIX##_cblas_call;                                                      \
static void stub_##SUFFIX##_fortran(                                            \
    int *n, TYPE *d, TYPE *l, TYPE *ld, int *clstrt, int *clend, TYPE *w,     \
    TYPE *wgap, TYPE *werr, TYPE *spdiam, TYPE *clgapl, TYPE *clgapr,         \
    TYPE *pivmin, TYPE *sigma, TYPE *dplus, TYPE *lplus, TYPE *work,          \
    int *info)                                                                  \
{                                                                               \
    g_##SUFFIX##_fortran_call.called += 1;                                      \
    g_##SUFFIX##_fortran_call.n = *n;                                           \
    g_##SUFFIX##_fortran_call.d = d;                                            \
    g_##SUFFIX##_fortran_call.l = l;                                            \
    g_##SUFFIX##_fortran_call.ld = ld;                                          \
    g_##SUFFIX##_fortran_call.clstrt = *clstrt;                                 \
    g_##SUFFIX##_fortran_call.clend = *clend;                                   \
    g_##SUFFIX##_fortran_call.w = w;                                            \
    g_##SUFFIX##_fortran_call.wgap = wgap;                                      \
    g_##SUFFIX##_fortran_call.werr = werr;                                      \
    g_##SUFFIX##_fortran_call.spdiam = *spdiam;                                 \
    g_##SUFFIX##_fortran_call.clgapl = *clgapl;                                 \
    g_##SUFFIX##_fortran_call.clgapr = *clgapr;                                 \
    g_##SUFFIX##_fortran_call.pivmin = *pivmin;                                 \
    g_##SUFFIX##_fortran_call.sigma = sigma;                                    \
    g_##SUFFIX##_fortran_call.dplus = dplus;                                    \
    g_##SUFFIX##_fortran_call.lplus = lplus;                                    \
    g_##SUFFIX##_fortran_call.work = work;                                      \
    g_##SUFFIX##_fortran_call.info_ptr = info;                                  \
    wgap[0] = (TYPE)((BASE) + 1);                                               \
    *sigma = (TYPE)((BASE) + 2);                                                \
    dplus[0] = (TYPE)((BASE) + 3);                                              \
    lplus[0] = (TYPE)((BASE) + 4);                                              \
    work[0] = (TYPE)((BASE) + 5);                                               \
    *info = (BASE) + 6;                                                         \
}                                                                               \
static int stub_##SUFFIX##_cblas(                                               \
    int n, const TYPE *d, const TYPE *l, const TYPE *ld, int clstrt,          \
    int clend, const TYPE *w, TYPE *wgap, const TYPE *werr, TYPE spdiam,      \
    TYPE clgapl, TYPE clgapr, TYPE pivmin, TYPE *sigma, TYPE *dplus,          \
    TYPE *lplus, TYPE *work)                                                    \
{                                                                               \
    g_##SUFFIX##_cblas_call.called += 1;                                        \
    g_##SUFFIX##_cblas_call.n = n;                                              \
    g_##SUFFIX##_cblas_call.d = d;                                              \
    g_##SUFFIX##_cblas_call.l = l;                                              \
    g_##SUFFIX##_cblas_call.ld = ld;                                            \
    g_##SUFFIX##_cblas_call.clstrt = clstrt;                                    \
    g_##SUFFIX##_cblas_call.clend = clend;                                      \
    g_##SUFFIX##_cblas_call.w = w;                                              \
    g_##SUFFIX##_cblas_call.wgap = wgap;                                        \
    g_##SUFFIX##_cblas_call.werr = werr;                                        \
    g_##SUFFIX##_cblas_call.spdiam = spdiam;                                    \
    g_##SUFFIX##_cblas_call.clgapl = clgapl;                                    \
    g_##SUFFIX##_cblas_call.clgapr = clgapr;                                    \
    g_##SUFFIX##_cblas_call.pivmin = pivmin;                                    \
    g_##SUFFIX##_cblas_call.sigma = sigma;                                      \
    g_##SUFFIX##_cblas_call.dplus = dplus;                                      \
    g_##SUFFIX##_cblas_call.lplus = lplus;                                      \
    g_##SUFFIX##_cblas_call.work = work;                                        \
    wgap[0] = (TYPE)((BASE) + 7);                                               \
    *sigma = (TYPE)((BASE) + 8);                                                \
    dplus[0] = (TYPE)((BASE) + 9);                                              \
    lplus[0] = (TYPE)((BASE) + 10);                                             \
    work[0] = (TYPE)((BASE) + 11);                                              \
    return (BASE) + 12;                                                         \
}                                                                               \
static int check_##SUFFIX##_fortran_to_cblas(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                        \
    TYPE d[3] = { (TYPE)1, (TYPE)2, (TYPE)3 };                                  \
    TYPE l[3] = { (TYPE)4, (TYPE)5, (TYPE)6 };                                  \
    TYPE ld[3] = { (TYPE)7, (TYPE)8, (TYPE)9 };                                 \
    TYPE w[3] = { (TYPE)10, (TYPE)11, (TYPE)12 };                               \
    TYPE wgap[3] = { (TYPE)13, (TYPE)14, (TYPE)15 };                            \
    TYPE werr[3] = { (TYPE)16, (TYPE)17, (TYPE)18 };                            \
    TYPE sigma = (TYPE)0;                                                       \
    TYPE dplus[3] = { (TYPE)19, (TYPE)20, (TYPE)21 };                           \
    TYPE lplus[3] = { (TYPE)22, (TYPE)23, (TYPE)24 };                           \
    TYPE work[3] = { (TYPE)25, (TYPE)26, (TYPE)27 };                            \
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
    if (thunk(3, d, l, ld, 2, 3, w, wgap, werr, (TYPE)28, (TYPE)29,            \
              (TYPE)30, (TYPE)31, &sigma, dplus, lplus, work) !=               \
                (BASE) + 6 ||                                                   \
        g_##SUFFIX##_fortran_call.called != 1 ||                                \
        g_##SUFFIX##_fortran_call.n != 3 ||                                     \
        g_##SUFFIX##_fortran_call.d != d ||                                     \
        g_##SUFFIX##_fortran_call.l != l ||                                     \
        g_##SUFFIX##_fortran_call.ld != ld ||                                   \
        g_##SUFFIX##_fortran_call.clstrt != 2 ||                                \
        g_##SUFFIX##_fortran_call.clend != 3 ||                                 \
        g_##SUFFIX##_fortran_call.w != w ||                                     \
        g_##SUFFIX##_fortran_call.wgap != wgap ||                               \
        g_##SUFFIX##_fortran_call.werr != werr ||                               \
        g_##SUFFIX##_fortran_call.spdiam != (TYPE)28 ||                         \
        g_##SUFFIX##_fortran_call.clgapl != (TYPE)29 ||                         \
        g_##SUFFIX##_fortran_call.clgapr != (TYPE)30 ||                         \
        g_##SUFFIX##_fortran_call.pivmin != (TYPE)31 ||                         \
        g_##SUFFIX##_fortran_call.sigma != &sigma ||                            \
        g_##SUFFIX##_fortran_call.dplus != dplus ||                             \
        g_##SUFFIX##_fortran_call.lplus != lplus ||                             \
        g_##SUFFIX##_fortran_call.work != work ||                               \
        g_##SUFFIX##_fortran_call.info_ptr == NULL ||                           \
        wgap[0] != (TYPE)((BASE) + 1) || sigma != (TYPE)((BASE) + 2) ||         \
        dplus[0] != (TYPE)((BASE) + 3) || lplus[0] != (TYPE)((BASE) + 4) ||     \
        work[0] != (TYPE)((BASE) + 5)) {                                        \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward LARRF inputs correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards LARRF inputs and returns info\n"); \
    return 0;                                                                   \
}                                                                               \
static int check_##SUFFIX##_cblas_to_fortran(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                      \
    int n = 3;                                                                  \
    int clstrt = 2;                                                             \
    int clend = 3;                                                              \
    TYPE d[3] = { (TYPE)1, (TYPE)2, (TYPE)3 };                                  \
    TYPE l[3] = { (TYPE)4, (TYPE)5, (TYPE)6 };                                  \
    TYPE ld[3] = { (TYPE)7, (TYPE)8, (TYPE)9 };                                 \
    TYPE w[3] = { (TYPE)10, (TYPE)11, (TYPE)12 };                               \
    TYPE wgap[3] = { (TYPE)13, (TYPE)14, (TYPE)15 };                            \
    TYPE werr[3] = { (TYPE)16, (TYPE)17, (TYPE)18 };                            \
    TYPE spdiam = (TYPE)28;                                                     \
    TYPE clgapl = (TYPE)29;                                                     \
    TYPE clgapr = (TYPE)30;                                                     \
    TYPE pivmin = (TYPE)31;                                                     \
    TYPE sigma = (TYPE)0;                                                       \
    TYPE dplus[3] = { (TYPE)19, (TYPE)20, (TYPE)21 };                           \
    TYPE lplus[3] = { (TYPE)22, (TYPE)23, (TYPE)24 };                           \
    TYPE work[3] = { (TYPE)25, (TYPE)26, (TYPE)27 };                            \
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
    thunk(&n, d, l, ld, &clstrt, &clend, w, wgap, werr, &spdiam, &clgapl,      \
          &clgapr, &pivmin, &sigma, dplus, lplus, work, &info);                \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                  \
        g_##SUFFIX##_cblas_call.n != 3 ||                                       \
        g_##SUFFIX##_cblas_call.d != d ||                                       \
        g_##SUFFIX##_cblas_call.l != l ||                                       \
        g_##SUFFIX##_cblas_call.ld != ld ||                                     \
        g_##SUFFIX##_cblas_call.clstrt != 2 ||                                  \
        g_##SUFFIX##_cblas_call.clend != 3 ||                                   \
        g_##SUFFIX##_cblas_call.w != w ||                                       \
        g_##SUFFIX##_cblas_call.wgap != wgap ||                                 \
        g_##SUFFIX##_cblas_call.werr != werr ||                                 \
        g_##SUFFIX##_cblas_call.spdiam != (TYPE)28 ||                           \
        g_##SUFFIX##_cblas_call.clgapl != (TYPE)29 ||                           \
        g_##SUFFIX##_cblas_call.clgapr != (TYPE)30 ||                           \
        g_##SUFFIX##_cblas_call.pivmin != (TYPE)31 ||                           \
        g_##SUFFIX##_cblas_call.sigma != &sigma ||                              \
        g_##SUFFIX##_cblas_call.dplus != dplus ||                               \
        g_##SUFFIX##_cblas_call.lplus != lplus ||                               \
        g_##SUFFIX##_cblas_call.work != work ||                                 \
        wgap[0] != (TYPE)((BASE) + 7) || sigma != (TYPE)((BASE) + 8) ||         \
        dplus[0] != (TYPE)((BASE) + 9) || lplus[0] != (TYPE)((BASE) + 10) ||    \
        work[0] != (TYPE)((BASE) + 11) || info != (BASE) + 12) {                \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference LARRF inputs correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences LARRF inputs and stores info\n"); \
    return 0;                                                                   \
}

DEFINE_LARRF_TESTS(slarrf, float, FB_OP_SLARRF, 100)
DEFINE_LARRF_TESTS(dlarrf, double, FB_OP_DLARRF, 300)

int main(void)
{
    int status = 0;

    status |= check_slarrf_fortran_to_cblas();
    status |= check_slarrf_cblas_to_fortran();
    status |= check_dlarrf_fortran_to_cblas();
    status |= check_dlarrf_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}