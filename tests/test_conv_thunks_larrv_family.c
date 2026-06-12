#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

#define DEFINE_LARRV_TESTS(SUFFIX, TYPE, OP_ID, BASE)                          \
typedef int (*fb_##SUFFIX##_cblas_fn)(                                         \
    int n, TYPE vl, TYPE vu, TYPE *d, TYPE *l, TYPE pivmin, const int *isplit, \
    int m, int dol, int dou, TYPE minrgp, TYPE rtol1, TYPE rtol2, TYPE *w,     \
    TYPE *werr, TYPE *wgap, const int *iblock, const int *indexw,              \
    const TYPE *gers, TYPE *z, int ldz, int *isuppz, TYPE *work, int *iwork);  \
typedef void (*fb_##SUFFIX##_fortran_fn)(                                      \
    int *n, TYPE *vl, TYPE *vu, TYPE *d, TYPE *l, TYPE *pivmin, int *isplit,   \
    int *m, int *dol, int *dou, TYPE *minrgp, TYPE *rtol1, TYPE *rtol2,        \
    TYPE *w, TYPE *werr, TYPE *wgap, int *iblock, int *indexw, TYPE *gers,     \
    TYPE *z, int *ldz, int *isuppz, TYPE *work, int *iwork, int *info);        \
static struct {                                                                 \
    int called;                                                                 \
    int n;                                                                      \
    TYPE vl;                                                                    \
    TYPE vu;                                                                    \
    TYPE *d;                                                                    \
    TYPE *l;                                                                    \
    TYPE pivmin;                                                                \
    int *isplit;                                                                \
    int m;                                                                      \
    int dol;                                                                    \
    int dou;                                                                    \
    TYPE minrgp;                                                                \
    TYPE rtol1;                                                                 \
    TYPE rtol2;                                                                 \
    TYPE *w;                                                                    \
    TYPE *werr;                                                                 \
    TYPE *wgap;                                                                 \
    int *iblock;                                                                \
    int *indexw;                                                                \
    TYPE *gers;                                                                 \
    TYPE *z;                                                                    \
    int ldz;                                                                    \
    int *isuppz;                                                                \
    TYPE *work;                                                                 \
    int *iwork;                                                                 \
    int *info_ptr;                                                              \
} g_##SUFFIX##_fortran_call;                                                    \
static struct {                                                                 \
    int called;                                                                 \
    int n;                                                                      \
    TYPE vl;                                                                    \
    TYPE vu;                                                                    \
    TYPE *d;                                                                    \
    TYPE *l;                                                                    \
    TYPE pivmin;                                                                \
    const int *isplit;                                                          \
    int m;                                                                      \
    int dol;                                                                    \
    int dou;                                                                    \
    TYPE minrgp;                                                                \
    TYPE rtol1;                                                                 \
    TYPE rtol2;                                                                 \
    TYPE *w;                                                                    \
    TYPE *werr;                                                                 \
    TYPE *wgap;                                                                 \
    const int *iblock;                                                          \
    const int *indexw;                                                          \
    const TYPE *gers;                                                           \
    TYPE *z;                                                                    \
    int ldz;                                                                    \
    int *isuppz;                                                                \
    TYPE *work;                                                                 \
    int *iwork;                                                                 \
} g_##SUFFIX##_cblas_call;                                                      \
static void stub_##SUFFIX##_fortran(                                            \
    int *n, TYPE *vl, TYPE *vu, TYPE *d, TYPE *l, TYPE *pivmin, int *isplit,   \
    int *m, int *dol, int *dou, TYPE *minrgp, TYPE *rtol1, TYPE *rtol2,        \
    TYPE *w, TYPE *werr, TYPE *wgap, int *iblock, int *indexw, TYPE *gers,     \
    TYPE *z, int *ldz, int *isuppz, TYPE *work, int *iwork, int *info)         \
{                                                                               \
    g_##SUFFIX##_fortran_call.called += 1;                                      \
    g_##SUFFIX##_fortran_call.n = *n;                                           \
    g_##SUFFIX##_fortran_call.vl = *vl;                                         \
    g_##SUFFIX##_fortran_call.vu = *vu;                                         \
    g_##SUFFIX##_fortran_call.d = d;                                            \
    g_##SUFFIX##_fortran_call.l = l;                                            \
    g_##SUFFIX##_fortran_call.pivmin = *pivmin;                                 \
    g_##SUFFIX##_fortran_call.isplit = isplit;                                  \
    g_##SUFFIX##_fortran_call.m = *m;                                           \
    g_##SUFFIX##_fortran_call.dol = *dol;                                       \
    g_##SUFFIX##_fortran_call.dou = *dou;                                       \
    g_##SUFFIX##_fortran_call.minrgp = *minrgp;                                 \
    g_##SUFFIX##_fortran_call.rtol1 = *rtol1;                                   \
    g_##SUFFIX##_fortran_call.rtol2 = *rtol2;                                   \
    g_##SUFFIX##_fortran_call.w = w;                                            \
    g_##SUFFIX##_fortran_call.werr = werr;                                      \
    g_##SUFFIX##_fortran_call.wgap = wgap;                                      \
    g_##SUFFIX##_fortran_call.iblock = iblock;                                  \
    g_##SUFFIX##_fortran_call.indexw = indexw;                                  \
    g_##SUFFIX##_fortran_call.gers = gers;                                      \
    g_##SUFFIX##_fortran_call.z = z;                                            \
    g_##SUFFIX##_fortran_call.ldz = *ldz;                                       \
    g_##SUFFIX##_fortran_call.isuppz = isuppz;                                  \
    g_##SUFFIX##_fortran_call.work = work;                                      \
    g_##SUFFIX##_fortran_call.iwork = iwork;                                    \
    g_##SUFFIX##_fortran_call.info_ptr = info;                                  \
    d[0] = (TYPE)((BASE) + 1);                                                  \
    l[0] = (TYPE)((BASE) + 2);                                                  \
    w[0] = (TYPE)((BASE) + 3);                                                  \
    werr[0] = (TYPE)((BASE) + 4);                                               \
    wgap[0] = (TYPE)((BASE) + 5);                                               \
    z[0] = (TYPE)((BASE) + 6);                                                  \
    isuppz[0] = (BASE) + 7;                                                     \
    work[0] = (TYPE)((BASE) + 8);                                               \
    iwork[0] = (BASE) + 9;                                                      \
    *info = (BASE) + 10;                                                        \
}                                                                               \
static int stub_##SUFFIX##_cblas(                                               \
    int n, TYPE vl, TYPE vu, TYPE *d, TYPE *l, TYPE pivmin, const int *isplit, \
    int m, int dol, int dou, TYPE minrgp, TYPE rtol1, TYPE rtol2, TYPE *w,     \
    TYPE *werr, TYPE *wgap, const int *iblock, const int *indexw,              \
    const TYPE *gers, TYPE *z, int ldz, int *isuppz, TYPE *work, int *iwork)   \
{                                                                               \
    g_##SUFFIX##_cblas_call.called += 1;                                        \
    g_##SUFFIX##_cblas_call.n = n;                                              \
    g_##SUFFIX##_cblas_call.vl = vl;                                            \
    g_##SUFFIX##_cblas_call.vu = vu;                                            \
    g_##SUFFIX##_cblas_call.d = d;                                              \
    g_##SUFFIX##_cblas_call.l = l;                                              \
    g_##SUFFIX##_cblas_call.pivmin = pivmin;                                    \
    g_##SUFFIX##_cblas_call.isplit = isplit;                                    \
    g_##SUFFIX##_cblas_call.m = m;                                              \
    g_##SUFFIX##_cblas_call.dol = dol;                                          \
    g_##SUFFIX##_cblas_call.dou = dou;                                          \
    g_##SUFFIX##_cblas_call.minrgp = minrgp;                                    \
    g_##SUFFIX##_cblas_call.rtol1 = rtol1;                                      \
    g_##SUFFIX##_cblas_call.rtol2 = rtol2;                                      \
    g_##SUFFIX##_cblas_call.w = w;                                              \
    g_##SUFFIX##_cblas_call.werr = werr;                                        \
    g_##SUFFIX##_cblas_call.wgap = wgap;                                        \
    g_##SUFFIX##_cblas_call.iblock = iblock;                                    \
    g_##SUFFIX##_cblas_call.indexw = indexw;                                    \
    g_##SUFFIX##_cblas_call.gers = gers;                                        \
    g_##SUFFIX##_cblas_call.z = z;                                              \
    g_##SUFFIX##_cblas_call.ldz = ldz;                                          \
    g_##SUFFIX##_cblas_call.isuppz = isuppz;                                    \
    g_##SUFFIX##_cblas_call.work = work;                                        \
    g_##SUFFIX##_cblas_call.iwork = iwork;                                      \
    d[0] = (TYPE)((BASE) + 11);                                                 \
    l[0] = (TYPE)((BASE) + 12);                                                 \
    w[0] = (TYPE)((BASE) + 13);                                                 \
    werr[0] = (TYPE)((BASE) + 14);                                              \
    wgap[0] = (TYPE)((BASE) + 15);                                              \
    z[0] = (TYPE)((BASE) + 16);                                                 \
    isuppz[0] = (BASE) + 17;                                                    \
    work[0] = (TYPE)((BASE) + 18);                                              \
    iwork[0] = (BASE) + 19;                                                     \
    return (BASE) + 20;                                                         \
}                                                                               \
static int check_##SUFFIX##_fortran_to_cblas(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                        \
    TYPE d[2] = { (TYPE)1, (TYPE)2 };                                           \
    TYPE l[2] = { (TYPE)3, (TYPE)4 };                                           \
    int isplit[1] = { 2 };                                                      \
    TYPE w[2] = { (TYPE)5, (TYPE)6 };                                           \
    TYPE werr[2] = { (TYPE)7, (TYPE)8 };                                        \
    TYPE wgap[2] = { (TYPE)9, (TYPE)10 };                                       \
    int iblock[2] = { 1, 1 };                                                   \
    int indexw[2] = { 1, 2 };                                                   \
    TYPE gers[4] = { (TYPE)11, (TYPE)12, (TYPE)13, (TYPE)14 };                 \
    TYPE z[4] = { (TYPE)0, (TYPE)0, (TYPE)0, (TYPE)0 };                        \
    int isuppz[4] = { 0, 0, 0, 0 };                                             \
    TYPE work[4] = { (TYPE)0, (TYPE)0, (TYPE)0, (TYPE)0 };                     \
    int iwork[4] = { 0, 0, 0, 0 };                                              \
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
    if (thunk(2, (TYPE)15, (TYPE)16, d, l, (TYPE)17, isplit, 2, 1, 2,          \
              (TYPE)18, (TYPE)19, (TYPE)20, w, werr, wgap, iblock, indexw,     \
              gers, z, 2, isuppz, work, iwork) != (BASE) + 10 ||               \
        g_##SUFFIX##_fortran_call.called != 1 ||                                \
        g_##SUFFIX##_fortran_call.n != 2 ||                                     \
        g_##SUFFIX##_fortran_call.vl != (TYPE)15 ||                             \
        g_##SUFFIX##_fortran_call.vu != (TYPE)16 ||                             \
        g_##SUFFIX##_fortran_call.d != d ||                                     \
        g_##SUFFIX##_fortran_call.l != l ||                                     \
        g_##SUFFIX##_fortran_call.pivmin != (TYPE)17 ||                         \
        g_##SUFFIX##_fortran_call.isplit != isplit ||                           \
        g_##SUFFIX##_fortran_call.m != 2 ||                                     \
        g_##SUFFIX##_fortran_call.dol != 1 ||                                   \
        g_##SUFFIX##_fortran_call.dou != 2 ||                                   \
        g_##SUFFIX##_fortran_call.minrgp != (TYPE)18 ||                         \
        g_##SUFFIX##_fortran_call.rtol1 != (TYPE)19 ||                          \
        g_##SUFFIX##_fortran_call.rtol2 != (TYPE)20 ||                          \
        g_##SUFFIX##_fortran_call.w != w ||                                     \
        g_##SUFFIX##_fortran_call.werr != werr ||                               \
        g_##SUFFIX##_fortran_call.wgap != wgap ||                               \
        g_##SUFFIX##_fortran_call.iblock != iblock ||                           \
        g_##SUFFIX##_fortran_call.indexw != indexw ||                           \
        g_##SUFFIX##_fortran_call.gers != gers ||                               \
        g_##SUFFIX##_fortran_call.z != z ||                                     \
        g_##SUFFIX##_fortran_call.ldz != 2 ||                                   \
        g_##SUFFIX##_fortran_call.isuppz != isuppz ||                           \
        g_##SUFFIX##_fortran_call.work != work ||                               \
        g_##SUFFIX##_fortran_call.iwork != iwork ||                             \
        g_##SUFFIX##_fortran_call.info_ptr == NULL ||                           \
        d[0] != (TYPE)((BASE) + 1) || l[0] != (TYPE)((BASE) + 2) ||             \
        w[0] != (TYPE)((BASE) + 3) || werr[0] != (TYPE)((BASE) + 4) ||          \
        wgap[0] != (TYPE)((BASE) + 5) || z[0] != (TYPE)((BASE) + 6) ||          \
        isuppz[0] != (BASE) + 7 || work[0] != (TYPE)((BASE) + 8) ||             \
        iwork[0] != (BASE) + 9) {                                               \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward LARRV inputs correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards LARRV inputs and returns info\n"); \
    return 0;                                                                   \
}                                                                               \
static int check_##SUFFIX##_cblas_to_fortran(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                      \
    int n = 2;                                                                  \
    TYPE vl = (TYPE)15;                                                         \
    TYPE vu = (TYPE)16;                                                         \
    TYPE d[2] = { (TYPE)1, (TYPE)2 };                                           \
    TYPE l[2] = { (TYPE)3, (TYPE)4 };                                           \
    TYPE pivmin = (TYPE)17;                                                     \
    int isplit[1] = { 2 };                                                      \
    int m = 2;                                                                  \
    int dol = 1;                                                                \
    int dou = 2;                                                                \
    TYPE minrgp = (TYPE)18;                                                     \
    TYPE rtol1 = (TYPE)19;                                                      \
    TYPE rtol2 = (TYPE)20;                                                      \
    TYPE w[2] = { (TYPE)5, (TYPE)6 };                                           \
    TYPE werr[2] = { (TYPE)7, (TYPE)8 };                                        \
    TYPE wgap[2] = { (TYPE)9, (TYPE)10 };                                       \
    int iblock[2] = { 1, 1 };                                                   \
    int indexw[2] = { 1, 2 };                                                   \
    TYPE gers[4] = { (TYPE)11, (TYPE)12, (TYPE)13, (TYPE)14 };                 \
    TYPE z[4] = { (TYPE)0, (TYPE)0, (TYPE)0, (TYPE)0 };                        \
    int ldz = 2;                                                                \
    int isuppz[4] = { 0, 0, 0, 0 };                                             \
    TYPE work[4] = { (TYPE)0, (TYPE)0, (TYPE)0, (TYPE)0 };                     \
    int iwork[4] = { 0, 0, 0, 0 };                                              \
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
    thunk(&n, &vl, &vu, d, l, &pivmin, isplit, &m, &dol, &dou, &minrgp,        \
          &rtol1, &rtol2, w, werr, wgap, iblock, indexw, gers, z, &ldz,        \
          isuppz, work, iwork, &info);                                          \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                  \
        g_##SUFFIX##_cblas_call.n != 2 ||                                       \
        g_##SUFFIX##_cblas_call.vl != (TYPE)15 ||                               \
        g_##SUFFIX##_cblas_call.vu != (TYPE)16 ||                               \
        g_##SUFFIX##_cblas_call.d != d ||                                       \
        g_##SUFFIX##_cblas_call.l != l ||                                       \
        g_##SUFFIX##_cblas_call.pivmin != (TYPE)17 ||                           \
        g_##SUFFIX##_cblas_call.isplit != isplit ||                             \
        g_##SUFFIX##_cblas_call.m != 2 ||                                       \
        g_##SUFFIX##_cblas_call.dol != 1 ||                                     \
        g_##SUFFIX##_cblas_call.dou != 2 ||                                     \
        g_##SUFFIX##_cblas_call.minrgp != (TYPE)18 ||                           \
        g_##SUFFIX##_cblas_call.rtol1 != (TYPE)19 ||                            \
        g_##SUFFIX##_cblas_call.rtol2 != (TYPE)20 ||                            \
        g_##SUFFIX##_cblas_call.w != w ||                                       \
        g_##SUFFIX##_cblas_call.werr != werr ||                                 \
        g_##SUFFIX##_cblas_call.wgap != wgap ||                                 \
        g_##SUFFIX##_cblas_call.iblock != iblock ||                             \
        g_##SUFFIX##_cblas_call.indexw != indexw ||                             \
        g_##SUFFIX##_cblas_call.gers != gers ||                                 \
        g_##SUFFIX##_cblas_call.z != z ||                                       \
        g_##SUFFIX##_cblas_call.ldz != 2 ||                                     \
        g_##SUFFIX##_cblas_call.isuppz != isuppz ||                             \
        g_##SUFFIX##_cblas_call.work != work ||                                 \
        g_##SUFFIX##_cblas_call.iwork != iwork ||                               \
        d[0] != (TYPE)((BASE) + 11) || l[0] != (TYPE)((BASE) + 12) ||           \
        w[0] != (TYPE)((BASE) + 13) || werr[0] != (TYPE)((BASE) + 14) ||        \
        wgap[0] != (TYPE)((BASE) + 15) || z[0] != (TYPE)((BASE) + 16) ||        \
        isuppz[0] != (BASE) + 17 || work[0] != (TYPE)((BASE) + 18) ||           \
        iwork[0] != (BASE) + 19 || info != (BASE) + 20) {                       \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference LARRV inputs correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences LARRV inputs and stores info\n"); \
    return 0;                                                                   \
}

DEFINE_LARRV_TESTS(slarrv, float, FB_OP_SLARRV, 100)
DEFINE_LARRV_TESTS(dlarrv, double, FB_OP_DLARRV, 300)

int main(void)
{
    int status = 0;

    status |= check_slarrv_fortran_to_cblas();
    status |= check_slarrv_cblas_to_fortran();
    status |= check_dlarrv_fortran_to_cblas();
    status |= check_dlarrv_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}