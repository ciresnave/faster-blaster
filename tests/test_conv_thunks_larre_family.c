#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

#define DEFINE_LARRE_TESTS(SUFFIX, TYPE, OP_ID, BASE)                          \
typedef int (*fb_##SUFFIX##_cblas_fn)(                                         \
    char range, int n, TYPE *vl, TYPE *vu, int il, int iu, TYPE *d, TYPE *e,  \
    TYPE *e2, TYPE rtol1, TYPE rtol2, TYPE spltol, int *nsplit, int *isplit,  \
    int *m, TYPE *w, TYPE *werr, TYPE *wgap, int *iblock, int *indexw,        \
    TYPE *gers, TYPE *pivmin, TYPE *work, int *iwork);                         \
typedef void (*fb_##SUFFIX##_fortran_fn)(                                      \
    char *range, int *n, TYPE *vl, TYPE *vu, int *il, int *iu, TYPE *d,       \
    TYPE *e, TYPE *e2, TYPE *rtol1, TYPE *rtol2, TYPE *spltol, int *nsplit,   \
    int *isplit, int *m, TYPE *w, TYPE *werr, TYPE *wgap, int *iblock,        \
    int *indexw, TYPE *gers, TYPE *pivmin, TYPE *work, int *iwork, int *info);\
static struct {                                                                 \
    int called;                                                                 \
    char range;                                                                 \
    int n;                                                                      \
    TYPE *vl;                                                                   \
    TYPE *vu;                                                                   \
    int il;                                                                     \
    int iu;                                                                     \
    TYPE *d;                                                                    \
    TYPE *e;                                                                    \
    TYPE *e2;                                                                   \
    TYPE rtol1;                                                                 \
    TYPE rtol2;                                                                 \
    TYPE spltol;                                                                \
    int *nsplit;                                                                \
    int *isplit;                                                                \
    int *m;                                                                     \
    TYPE *w;                                                                    \
    TYPE *werr;                                                                 \
    TYPE *wgap;                                                                 \
    int *iblock;                                                                \
    int *indexw;                                                                \
    TYPE *gers;                                                                 \
    TYPE *pivmin;                                                               \
    TYPE *work;                                                                 \
    int *iwork;                                                                 \
    int *info_ptr;                                                              \
} g_##SUFFIX##_fortran_call;                                                    \
static struct {                                                                 \
    int called;                                                                 \
    char range;                                                                 \
    int n;                                                                      \
    TYPE *vl;                                                                   \
    TYPE *vu;                                                                   \
    int il;                                                                     \
    int iu;                                                                     \
    TYPE *d;                                                                    \
    TYPE *e;                                                                    \
    TYPE *e2;                                                                   \
    TYPE rtol1;                                                                 \
    TYPE rtol2;                                                                 \
    TYPE spltol;                                                                \
    int *nsplit;                                                                \
    int *isplit;                                                                \
    int *m;                                                                     \
    TYPE *w;                                                                    \
    TYPE *werr;                                                                 \
    TYPE *wgap;                                                                 \
    int *iblock;                                                                \
    int *indexw;                                                                \
    TYPE *gers;                                                                 \
    TYPE *pivmin;                                                               \
    TYPE *work;                                                                 \
    int *iwork;                                                                 \
} g_##SUFFIX##_cblas_call;                                                      \
static void stub_##SUFFIX##_fortran(                                            \
    char *range, int *n, TYPE *vl, TYPE *vu, int *il, int *iu, TYPE *d,       \
    TYPE *e, TYPE *e2, TYPE *rtol1, TYPE *rtol2, TYPE *spltol, int *nsplit,   \
    int *isplit, int *m, TYPE *w, TYPE *werr, TYPE *wgap, int *iblock,        \
    int *indexw, TYPE *gers, TYPE *pivmin, TYPE *work, int *iwork, int *info) \
{                                                                               \
    g_##SUFFIX##_fortran_call.called += 1;                                      \
    g_##SUFFIX##_fortran_call.range = *range;                                   \
    g_##SUFFIX##_fortran_call.n = *n;                                           \
    g_##SUFFIX##_fortran_call.vl = vl;                                          \
    g_##SUFFIX##_fortran_call.vu = vu;                                          \
    g_##SUFFIX##_fortran_call.il = *il;                                         \
    g_##SUFFIX##_fortran_call.iu = *iu;                                         \
    g_##SUFFIX##_fortran_call.d = d;                                            \
    g_##SUFFIX##_fortran_call.e = e;                                            \
    g_##SUFFIX##_fortran_call.e2 = e2;                                          \
    g_##SUFFIX##_fortran_call.rtol1 = *rtol1;                                   \
    g_##SUFFIX##_fortran_call.rtol2 = *rtol2;                                   \
    g_##SUFFIX##_fortran_call.spltol = *spltol;                                 \
    g_##SUFFIX##_fortran_call.nsplit = nsplit;                                  \
    g_##SUFFIX##_fortran_call.isplit = isplit;                                  \
    g_##SUFFIX##_fortran_call.m = m;                                            \
    g_##SUFFIX##_fortran_call.w = w;                                            \
    g_##SUFFIX##_fortran_call.werr = werr;                                      \
    g_##SUFFIX##_fortran_call.wgap = wgap;                                      \
    g_##SUFFIX##_fortran_call.iblock = iblock;                                  \
    g_##SUFFIX##_fortran_call.indexw = indexw;                                  \
    g_##SUFFIX##_fortran_call.gers = gers;                                      \
    g_##SUFFIX##_fortran_call.pivmin = pivmin;                                  \
    g_##SUFFIX##_fortran_call.work = work;                                      \
    g_##SUFFIX##_fortran_call.iwork = iwork;                                    \
    g_##SUFFIX##_fortran_call.info_ptr = info;                                  \
    *vl = (TYPE)((BASE) + 1);                                                   \
    *vu = (TYPE)((BASE) + 2);                                                   \
    d[0] = (TYPE)((BASE) + 3);                                                  \
    e[0] = (TYPE)((BASE) + 4);                                                  \
    e2[0] = (TYPE)((BASE) + 5);                                                 \
    *nsplit = (BASE) + 6;                                                       \
    isplit[0] = (BASE) + 7;                                                     \
    *m = (BASE) + 8;                                                            \
    w[0] = (TYPE)((BASE) + 9);                                                  \
    werr[0] = (TYPE)((BASE) + 10);                                              \
    wgap[0] = (TYPE)((BASE) + 11);                                              \
    iblock[0] = (BASE) + 12;                                                    \
    indexw[0] = (BASE) + 13;                                                    \
    gers[0] = (TYPE)((BASE) + 14);                                              \
    *pivmin = (TYPE)((BASE) + 15);                                              \
    work[0] = (TYPE)((BASE) + 16);                                              \
    iwork[0] = (BASE) + 17;                                                     \
    *info = (BASE) + 18;                                                        \
}                                                                               \
static int stub_##SUFFIX##_cblas(                                               \
    char range, int n, TYPE *vl, TYPE *vu, int il, int iu, TYPE *d, TYPE *e,  \
    TYPE *e2, TYPE rtol1, TYPE rtol2, TYPE spltol, int *nsplit, int *isplit,  \
    int *m, TYPE *w, TYPE *werr, TYPE *wgap, int *iblock, int *indexw,        \
    TYPE *gers, TYPE *pivmin, TYPE *work, int *iwork)                          \
{                                                                               \
    g_##SUFFIX##_cblas_call.called += 1;                                        \
    g_##SUFFIX##_cblas_call.range = range;                                      \
    g_##SUFFIX##_cblas_call.n = n;                                              \
    g_##SUFFIX##_cblas_call.vl = vl;                                            \
    g_##SUFFIX##_cblas_call.vu = vu;                                            \
    g_##SUFFIX##_cblas_call.il = il;                                            \
    g_##SUFFIX##_cblas_call.iu = iu;                                            \
    g_##SUFFIX##_cblas_call.d = d;                                              \
    g_##SUFFIX##_cblas_call.e = e;                                              \
    g_##SUFFIX##_cblas_call.e2 = e2;                                            \
    g_##SUFFIX##_cblas_call.rtol1 = rtol1;                                      \
    g_##SUFFIX##_cblas_call.rtol2 = rtol2;                                      \
    g_##SUFFIX##_cblas_call.spltol = spltol;                                    \
    g_##SUFFIX##_cblas_call.nsplit = nsplit;                                    \
    g_##SUFFIX##_cblas_call.isplit = isplit;                                    \
    g_##SUFFIX##_cblas_call.m = m;                                              \
    g_##SUFFIX##_cblas_call.w = w;                                              \
    g_##SUFFIX##_cblas_call.werr = werr;                                        \
    g_##SUFFIX##_cblas_call.wgap = wgap;                                        \
    g_##SUFFIX##_cblas_call.iblock = iblock;                                    \
    g_##SUFFIX##_cblas_call.indexw = indexw;                                    \
    g_##SUFFIX##_cblas_call.gers = gers;                                        \
    g_##SUFFIX##_cblas_call.pivmin = pivmin;                                    \
    g_##SUFFIX##_cblas_call.work = work;                                        \
    g_##SUFFIX##_cblas_call.iwork = iwork;                                      \
    *vl = (TYPE)((BASE) + 19);                                                  \
    *vu = (TYPE)((BASE) + 20);                                                  \
    d[0] = (TYPE)((BASE) + 21);                                                 \
    e[0] = (TYPE)((BASE) + 22);                                                 \
    e2[0] = (TYPE)((BASE) + 23);                                                \
    *nsplit = (BASE) + 24;                                                      \
    isplit[0] = (BASE) + 25;                                                    \
    *m = (BASE) + 26;                                                           \
    w[0] = (TYPE)((BASE) + 27);                                                 \
    werr[0] = (TYPE)((BASE) + 28);                                              \
    wgap[0] = (TYPE)((BASE) + 29);                                              \
    iblock[0] = (BASE) + 30;                                                    \
    indexw[0] = (BASE) + 31;                                                    \
    gers[0] = (TYPE)((BASE) + 32);                                              \
    *pivmin = (TYPE)((BASE) + 33);                                              \
    work[0] = (TYPE)((BASE) + 34);                                              \
    iwork[0] = (BASE) + 35;                                                     \
    return (BASE) + 36;                                                         \
}                                                                               \
static int check_##SUFFIX##_fortran_to_cblas(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                        \
    TYPE vl = (TYPE)1;                                                          \
    TYPE vu = (TYPE)2;                                                          \
    TYPE d[2] = { (TYPE)3, (TYPE)4 };                                           \
    TYPE e[2] = { (TYPE)5, (TYPE)6 };                                           \
    TYPE e2[2] = { (TYPE)7, (TYPE)8 };                                          \
    int nsplit = 0;                                                             \
    int isplit[2] = { 9, 10 };                                                  \
    int m = 0;                                                                  \
    TYPE w[2] = { (TYPE)11, (TYPE)12 };                                         \
    TYPE werr[2] = { (TYPE)13, (TYPE)14 };                                      \
    TYPE wgap[2] = { (TYPE)15, (TYPE)16 };                                      \
    int iblock[2] = { 17, 18 };                                                 \
    int indexw[2] = { 19, 20 };                                                 \
    TYPE gers[2] = { (TYPE)21, (TYPE)22 };                                      \
    TYPE pivmin = (TYPE)23;                                                     \
    TYPE work[2] = { (TYPE)24, (TYPE)25 };                                      \
    int iwork[2] = { 26, 27 };                                                  \
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
    if (thunk('I', 2, &vl, &vu, 1, 2, d, e, e2, (TYPE)28, (TYPE)29,            \
              (TYPE)30, &nsplit, isplit, &m, w, werr, wgap, iblock, indexw,    \
              gers, &pivmin, work, iwork) != (BASE) + 18 ||                    \
        g_##SUFFIX##_fortran_call.called != 1 ||                                \
        g_##SUFFIX##_fortran_call.range != 'I' ||                               \
        g_##SUFFIX##_fortran_call.n != 2 ||                                     \
        g_##SUFFIX##_fortran_call.vl != &vl ||                                  \
        g_##SUFFIX##_fortran_call.vu != &vu ||                                  \
        g_##SUFFIX##_fortran_call.il != 1 ||                                    \
        g_##SUFFIX##_fortran_call.iu != 2 ||                                    \
        g_##SUFFIX##_fortran_call.d != d ||                                     \
        g_##SUFFIX##_fortran_call.e != e ||                                     \
        g_##SUFFIX##_fortran_call.e2 != e2 ||                                   \
        g_##SUFFIX##_fortran_call.rtol1 != (TYPE)28 ||                          \
        g_##SUFFIX##_fortran_call.rtol2 != (TYPE)29 ||                          \
        g_##SUFFIX##_fortran_call.spltol != (TYPE)30 ||                         \
        g_##SUFFIX##_fortran_call.nsplit != &nsplit ||                          \
        g_##SUFFIX##_fortran_call.isplit != isplit ||                           \
        g_##SUFFIX##_fortran_call.m != &m ||                                    \
        g_##SUFFIX##_fortran_call.w != w ||                                     \
        g_##SUFFIX##_fortran_call.werr != werr ||                               \
        g_##SUFFIX##_fortran_call.wgap != wgap ||                               \
        g_##SUFFIX##_fortran_call.iblock != iblock ||                           \
        g_##SUFFIX##_fortran_call.indexw != indexw ||                           \
        g_##SUFFIX##_fortran_call.gers != gers ||                               \
        g_##SUFFIX##_fortran_call.pivmin != &pivmin ||                          \
        g_##SUFFIX##_fortran_call.work != work ||                               \
        g_##SUFFIX##_fortran_call.iwork != iwork ||                             \
        g_##SUFFIX##_fortran_call.info_ptr == NULL ||                           \
        vl != (TYPE)((BASE) + 1) || vu != (TYPE)((BASE) + 2) ||                 \
        d[0] != (TYPE)((BASE) + 3) || e[0] != (TYPE)((BASE) + 4) ||             \
        e2[0] != (TYPE)((BASE) + 5) || nsplit != (BASE) + 6 ||                  \
        isplit[0] != (BASE) + 7 || m != (BASE) + 8 ||                           \
        w[0] != (TYPE)((BASE) + 9) || werr[0] != (TYPE)((BASE) + 10) ||         \
        wgap[0] != (TYPE)((BASE) + 11) || iblock[0] != (BASE) + 12 ||           \
        indexw[0] != (BASE) + 13 || gers[0] != (TYPE)((BASE) + 14) ||           \
        pivmin != (TYPE)((BASE) + 15) || work[0] != (TYPE)((BASE) + 16) ||      \
        iwork[0] != (BASE) + 17) {                                              \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward LARRE inputs correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards LARRE inputs and returns info\n"); \
    return 0;                                                                   \
}                                                                               \
static int check_##SUFFIX##_cblas_to_fortran(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                      \
    char range = 'I';                                                           \
    int n = 2;                                                                  \
    TYPE vl = (TYPE)1;                                                          \
    TYPE vu = (TYPE)2;                                                          \
    int il = 1;                                                                 \
    int iu = 2;                                                                 \
    TYPE d[2] = { (TYPE)3, (TYPE)4 };                                           \
    TYPE e[2] = { (TYPE)5, (TYPE)6 };                                           \
    TYPE e2[2] = { (TYPE)7, (TYPE)8 };                                          \
    TYPE rtol1 = (TYPE)28;                                                      \
    TYPE rtol2 = (TYPE)29;                                                      \
    TYPE spltol = (TYPE)30;                                                     \
    int nsplit = 0;                                                             \
    int isplit[2] = { 9, 10 };                                                  \
    int m = 0;                                                                  \
    TYPE w[2] = { (TYPE)11, (TYPE)12 };                                         \
    TYPE werr[2] = { (TYPE)13, (TYPE)14 };                                      \
    TYPE wgap[2] = { (TYPE)15, (TYPE)16 };                                      \
    int iblock[2] = { 17, 18 };                                                 \
    int indexw[2] = { 19, 20 };                                                 \
    TYPE gers[2] = { (TYPE)21, (TYPE)22 };                                      \
    TYPE pivmin = (TYPE)23;                                                     \
    TYPE work[2] = { (TYPE)24, (TYPE)25 };                                      \
    int iwork[2] = { 26, 27 };                                                  \
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
    thunk(&range, &n, &vl, &vu, &il, &iu, d, e, e2, &rtol1, &rtol2, &spltol,   \
          &nsplit, isplit, &m, w, werr, wgap, iblock, indexw, gers, &pivmin,   \
          work, iwork, &info);                                                  \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                  \
        g_##SUFFIX##_cblas_call.range != 'I' ||                                 \
        g_##SUFFIX##_cblas_call.n != 2 ||                                       \
        g_##SUFFIX##_cblas_call.vl != &vl ||                                    \
        g_##SUFFIX##_cblas_call.vu != &vu ||                                    \
        g_##SUFFIX##_cblas_call.il != 1 ||                                      \
        g_##SUFFIX##_cblas_call.iu != 2 ||                                      \
        g_##SUFFIX##_cblas_call.d != d ||                                       \
        g_##SUFFIX##_cblas_call.e != e ||                                       \
        g_##SUFFIX##_cblas_call.e2 != e2 ||                                     \
        g_##SUFFIX##_cblas_call.rtol1 != (TYPE)28 ||                            \
        g_##SUFFIX##_cblas_call.rtol2 != (TYPE)29 ||                            \
        g_##SUFFIX##_cblas_call.spltol != (TYPE)30 ||                           \
        g_##SUFFIX##_cblas_call.nsplit != &nsplit ||                            \
        g_##SUFFIX##_cblas_call.isplit != isplit ||                             \
        g_##SUFFIX##_cblas_call.m != &m ||                                      \
        g_##SUFFIX##_cblas_call.w != w ||                                       \
        g_##SUFFIX##_cblas_call.werr != werr ||                                 \
        g_##SUFFIX##_cblas_call.wgap != wgap ||                                 \
        g_##SUFFIX##_cblas_call.iblock != iblock ||                             \
        g_##SUFFIX##_cblas_call.indexw != indexw ||                             \
        g_##SUFFIX##_cblas_call.gers != gers ||                                 \
        g_##SUFFIX##_cblas_call.pivmin != &pivmin ||                            \
        g_##SUFFIX##_cblas_call.work != work ||                                 \
        g_##SUFFIX##_cblas_call.iwork != iwork ||                               \
        vl != (TYPE)((BASE) + 19) || vu != (TYPE)((BASE) + 20) ||               \
        d[0] != (TYPE)((BASE) + 21) || e[0] != (TYPE)((BASE) + 22) ||           \
        e2[0] != (TYPE)((BASE) + 23) || nsplit != (BASE) + 24 ||                \
        isplit[0] != (BASE) + 25 || m != (BASE) + 26 ||                         \
        w[0] != (TYPE)((BASE) + 27) || werr[0] != (TYPE)((BASE) + 28) ||        \
        wgap[0] != (TYPE)((BASE) + 29) || iblock[0] != (BASE) + 30 ||           \
        indexw[0] != (BASE) + 31 || gers[0] != (TYPE)((BASE) + 32) ||           \
        pivmin != (TYPE)((BASE) + 33) || work[0] != (TYPE)((BASE) + 34) ||      \
        iwork[0] != (BASE) + 35 || info != (BASE) + 36) {                       \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference LARRE inputs correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences LARRE inputs and stores info\n"); \
    return 0;                                                                   \
}

DEFINE_LARRE_TESTS(slarre, float, FB_OP_SLARRE, 100)
DEFINE_LARRE_TESTS(dlarre, double, FB_OP_DLARRE, 300)

int main(void)
{
    int status = 0;

    status |= check_slarre_fortran_to_cblas();
    status |= check_slarre_cblas_to_fortran();
    status |= check_dlarre_fortran_to_cblas();
    status |= check_dlarre_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}