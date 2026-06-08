#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

#define DEFINE_LARRD_TESTS(SUFFIX, TYPE, OP_ID, BASE)                          \
typedef int (*fb_##SUFFIX##_cblas_fn)(                                         \
    char range, char order, int n, TYPE vl, TYPE vu, int il, int iu,          \
    TYPE *gers, TYPE reltol, const TYPE *d, const TYPE *e, const TYPE *e2,    \
    TYPE pivmin, int nsplit, const int *isplit, int *m, TYPE *w, TYPE *werr,  \
    TYPE *wl, TYPE *wu, int *iblock, int *indexw, TYPE *work, int *iwork);    \
typedef void (*fb_##SUFFIX##_fortran_fn)(                                      \
    char *range, char *order, int *n, TYPE *vl, TYPE *vu, int *il, int *iu,   \
    TYPE *gers, TYPE *reltol, TYPE *d, TYPE *e, TYPE *e2, TYPE *pivmin,       \
    int *nsplit, int *isplit, int *m, TYPE *w, TYPE *werr, TYPE *wl,          \
    TYPE *wu, int *iblock, int *indexw, TYPE *work, int *iwork, int *info);   \
static struct {                                                                 \
    int called;                                                                 \
    char range;                                                                 \
    char order;                                                                 \
    int n;                                                                      \
    TYPE vl;                                                                    \
    TYPE vu;                                                                    \
    int il;                                                                     \
    int iu;                                                                     \
    TYPE *gers;                                                                 \
    TYPE reltol;                                                                \
    TYPE *d;                                                                    \
    TYPE *e;                                                                    \
    TYPE *e2;                                                                   \
    TYPE pivmin;                                                                \
    int nsplit;                                                                 \
    int *isplit;                                                                \
    int *m;                                                                     \
    TYPE *w;                                                                    \
    TYPE *werr;                                                                 \
    TYPE *wl;                                                                   \
    TYPE *wu;                                                                   \
    int *iblock;                                                                \
    int *indexw;                                                                \
    TYPE *work;                                                                 \
    int *iwork;                                                                 \
    int *info_ptr;                                                              \
} g_##SUFFIX##_fortran_call;                                                    \
static struct {                                                                 \
    int called;                                                                 \
    char range;                                                                 \
    char order;                                                                 \
    int n;                                                                      \
    TYPE vl;                                                                    \
    TYPE vu;                                                                    \
    int il;                                                                     \
    int iu;                                                                     \
    TYPE *gers;                                                                 \
    TYPE reltol;                                                                \
    const TYPE *d;                                                              \
    const TYPE *e;                                                              \
    const TYPE *e2;                                                             \
    TYPE pivmin;                                                                \
    int nsplit;                                                                 \
    const int *isplit;                                                          \
    int *m;                                                                     \
    TYPE *w;                                                                    \
    TYPE *werr;                                                                 \
    TYPE *wl;                                                                   \
    TYPE *wu;                                                                   \
    int *iblock;                                                                \
    int *indexw;                                                                \
    TYPE *work;                                                                 \
    int *iwork;                                                                 \
} g_##SUFFIX##_cblas_call;                                                      \
static void stub_##SUFFIX##_fortran(                                            \
    char *range, char *order, int *n, TYPE *vl, TYPE *vu, int *il, int *iu,   \
    TYPE *gers, TYPE *reltol, TYPE *d, TYPE *e, TYPE *e2, TYPE *pivmin,       \
    int *nsplit, int *isplit, int *m, TYPE *w, TYPE *werr, TYPE *wl,          \
    TYPE *wu, int *iblock, int *indexw, TYPE *work, int *iwork, int *info)    \
{                                                                               \
    g_##SUFFIX##_fortran_call.called += 1;                                      \
    g_##SUFFIX##_fortran_call.range = *range;                                   \
    g_##SUFFIX##_fortran_call.order = *order;                                   \
    g_##SUFFIX##_fortran_call.n = *n;                                           \
    g_##SUFFIX##_fortran_call.vl = *vl;                                         \
    g_##SUFFIX##_fortran_call.vu = *vu;                                         \
    g_##SUFFIX##_fortran_call.il = *il;                                         \
    g_##SUFFIX##_fortran_call.iu = *iu;                                         \
    g_##SUFFIX##_fortran_call.gers = gers;                                      \
    g_##SUFFIX##_fortran_call.reltol = *reltol;                                 \
    g_##SUFFIX##_fortran_call.d = d;                                            \
    g_##SUFFIX##_fortran_call.e = e;                                            \
    g_##SUFFIX##_fortran_call.e2 = e2;                                          \
    g_##SUFFIX##_fortran_call.pivmin = *pivmin;                                 \
    g_##SUFFIX##_fortran_call.nsplit = *nsplit;                                 \
    g_##SUFFIX##_fortran_call.isplit = isplit;                                  \
    g_##SUFFIX##_fortran_call.m = m;                                            \
    g_##SUFFIX##_fortran_call.w = w;                                            \
    g_##SUFFIX##_fortran_call.werr = werr;                                      \
    g_##SUFFIX##_fortran_call.wl = wl;                                          \
    g_##SUFFIX##_fortran_call.wu = wu;                                          \
    g_##SUFFIX##_fortran_call.iblock = iblock;                                  \
    g_##SUFFIX##_fortran_call.indexw = indexw;                                  \
    g_##SUFFIX##_fortran_call.work = work;                                      \
    g_##SUFFIX##_fortran_call.iwork = iwork;                                    \
    g_##SUFFIX##_fortran_call.info_ptr = info;                                  \
    *m = (BASE) + 1;                                                            \
    w[0] = (TYPE)((BASE) + 2);                                                  \
    werr[0] = (TYPE)((BASE) + 3);                                               \
    *wl = (TYPE)((BASE) + 4);                                                   \
    *wu = (TYPE)((BASE) + 5);                                                   \
    iblock[0] = (BASE) + 6;                                                     \
    indexw[0] = (BASE) + 7;                                                     \
    work[0] = (TYPE)((BASE) + 8);                                               \
    iwork[0] = (BASE) + 9;                                                      \
    *info = (BASE) + 10;                                                        \
}                                                                               \
static int stub_##SUFFIX##_cblas(                                               \
    char range, char order, int n, TYPE vl, TYPE vu, int il, int iu,          \
    TYPE *gers, TYPE reltol, const TYPE *d, const TYPE *e, const TYPE *e2,    \
    TYPE pivmin, int nsplit, const int *isplit, int *m, TYPE *w, TYPE *werr,  \
    TYPE *wl, TYPE *wu, int *iblock, int *indexw, TYPE *work, int *iwork)     \
{                                                                               \
    g_##SUFFIX##_cblas_call.called += 1;                                        \
    g_##SUFFIX##_cblas_call.range = range;                                      \
    g_##SUFFIX##_cblas_call.order = order;                                      \
    g_##SUFFIX##_cblas_call.n = n;                                              \
    g_##SUFFIX##_cblas_call.vl = vl;                                            \
    g_##SUFFIX##_cblas_call.vu = vu;                                            \
    g_##SUFFIX##_cblas_call.il = il;                                            \
    g_##SUFFIX##_cblas_call.iu = iu;                                            \
    g_##SUFFIX##_cblas_call.gers = gers;                                        \
    g_##SUFFIX##_cblas_call.reltol = reltol;                                    \
    g_##SUFFIX##_cblas_call.d = d;                                              \
    g_##SUFFIX##_cblas_call.e = e;                                              \
    g_##SUFFIX##_cblas_call.e2 = e2;                                            \
    g_##SUFFIX##_cblas_call.pivmin = pivmin;                                    \
    g_##SUFFIX##_cblas_call.nsplit = nsplit;                                    \
    g_##SUFFIX##_cblas_call.isplit = isplit;                                    \
    g_##SUFFIX##_cblas_call.m = m;                                              \
    g_##SUFFIX##_cblas_call.w = w;                                              \
    g_##SUFFIX##_cblas_call.werr = werr;                                        \
    g_##SUFFIX##_cblas_call.wl = wl;                                            \
    g_##SUFFIX##_cblas_call.wu = wu;                                            \
    g_##SUFFIX##_cblas_call.iblock = iblock;                                    \
    g_##SUFFIX##_cblas_call.indexw = indexw;                                    \
    g_##SUFFIX##_cblas_call.work = work;                                        \
    g_##SUFFIX##_cblas_call.iwork = iwork;                                      \
    *m = (BASE) + 11;                                                           \
    w[0] = (TYPE)((BASE) + 12);                                                 \
    werr[0] = (TYPE)((BASE) + 13);                                              \
    *wl = (TYPE)((BASE) + 14);                                                  \
    *wu = (TYPE)((BASE) + 15);                                                  \
    iblock[0] = (BASE) + 16;                                                    \
    indexw[0] = (BASE) + 17;                                                    \
    work[0] = (TYPE)((BASE) + 18);                                              \
    iwork[0] = (BASE) + 19;                                                     \
    return (BASE) + 20;                                                         \
}                                                                               \
static int check_##SUFFIX##_fortran_to_cblas(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                        \
    TYPE gers[2] = { (TYPE)1, (TYPE)2 };                                        \
    TYPE d[2] = { (TYPE)3, (TYPE)4 };                                           \
    TYPE e[2] = { (TYPE)5, (TYPE)6 };                                           \
    TYPE e2[2] = { (TYPE)7, (TYPE)8 };                                          \
    int isplit[2] = { 1, 2 };                                                   \
    int m = 0;                                                                  \
    TYPE w[2] = { (TYPE)9, (TYPE)10 };                                          \
    TYPE werr[2] = { (TYPE)11, (TYPE)12 };                                      \
    TYPE wl = (TYPE)13;                                                         \
    TYPE wu = (TYPE)14;                                                         \
    int iblock[2] = { 15, 16 };                                                 \
    int indexw[2] = { 17, 18 };                                                 \
    TYPE work[2] = { (TYPE)19, (TYPE)20 };                                      \
    int iwork[2] = { 21, 22 };                                                  \
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
    if (thunk('I', 'E', 2, (TYPE)23, (TYPE)24, 1, 2, gers, (TYPE)25, d, e,     \
              e2, (TYPE)26, 2, isplit, &m, w, werr, &wl, &wu, iblock, indexw,  \
              work, iwork) != (BASE) + 10 ||                                   \
        g_##SUFFIX##_fortran_call.called != 1 ||                                \
        g_##SUFFIX##_fortran_call.range != 'I' ||                               \
        g_##SUFFIX##_fortran_call.order != 'E' ||                               \
        g_##SUFFIX##_fortran_call.n != 2 ||                                     \
        g_##SUFFIX##_fortran_call.vl != (TYPE)23 ||                             \
        g_##SUFFIX##_fortran_call.vu != (TYPE)24 ||                             \
        g_##SUFFIX##_fortran_call.il != 1 ||                                    \
        g_##SUFFIX##_fortran_call.iu != 2 ||                                    \
        g_##SUFFIX##_fortran_call.gers != gers ||                               \
        g_##SUFFIX##_fortran_call.reltol != (TYPE)25 ||                         \
        g_##SUFFIX##_fortran_call.d != d ||                                     \
        g_##SUFFIX##_fortran_call.e != e ||                                     \
        g_##SUFFIX##_fortran_call.e2 != e2 ||                                   \
        g_##SUFFIX##_fortran_call.pivmin != (TYPE)26 ||                         \
        g_##SUFFIX##_fortran_call.nsplit != 2 ||                                \
        g_##SUFFIX##_fortran_call.isplit != isplit ||                           \
        g_##SUFFIX##_fortran_call.m != &m ||                                    \
        g_##SUFFIX##_fortran_call.w != w ||                                     \
        g_##SUFFIX##_fortran_call.werr != werr ||                               \
        g_##SUFFIX##_fortran_call.wl != &wl ||                                  \
        g_##SUFFIX##_fortran_call.wu != &wu ||                                  \
        g_##SUFFIX##_fortran_call.iblock != iblock ||                           \
        g_##SUFFIX##_fortran_call.indexw != indexw ||                           \
        g_##SUFFIX##_fortran_call.work != work ||                               \
        g_##SUFFIX##_fortran_call.iwork != iwork ||                             \
        g_##SUFFIX##_fortran_call.info_ptr == NULL ||                           \
        m != (BASE) + 1 || w[0] != (TYPE)((BASE) + 2) ||                        \
        werr[0] != (TYPE)((BASE) + 3) || wl != (TYPE)((BASE) + 4) ||            \
        wu != (TYPE)((BASE) + 5) || iblock[0] != (BASE) + 6 ||                  \
        indexw[0] != (BASE) + 7 || work[0] != (TYPE)((BASE) + 8) ||             \
        iwork[0] != (BASE) + 9) {                                               \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward LARRD inputs correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards LARRD inputs and returns info\n"); \
    return 0;                                                                   \
}                                                                               \
static int check_##SUFFIX##_cblas_to_fortran(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                      \
    char range = 'I';                                                           \
    char order = 'E';                                                           \
    int n = 2;                                                                  \
    TYPE vl = (TYPE)23;                                                         \
    TYPE vu = (TYPE)24;                                                         \
    int il = 1;                                                                 \
    int iu = 2;                                                                 \
    TYPE gers[2] = { (TYPE)31, (TYPE)32 };                                      \
    TYPE reltol = (TYPE)25;                                                     \
    TYPE d[2] = { (TYPE)33, (TYPE)34 };                                         \
    TYPE e[2] = { (TYPE)35, (TYPE)36 };                                         \
    TYPE e2[2] = { (TYPE)37, (TYPE)38 };                                        \
    TYPE pivmin = (TYPE)26;                                                     \
    int nsplit = 2;                                                             \
    int isplit[2] = { 1, 2 };                                                   \
    int m = 0;                                                                  \
    TYPE w[2] = { (TYPE)39, (TYPE)40 };                                         \
    TYPE werr[2] = { (TYPE)41, (TYPE)42 };                                      \
    TYPE wl = (TYPE)43;                                                         \
    TYPE wu = (TYPE)44;                                                         \
    int iblock[2] = { 45, 46 };                                                 \
    int indexw[2] = { 47, 48 };                                                 \
    TYPE work[2] = { (TYPE)49, (TYPE)50 };                                      \
    int iwork[2] = { 51, 52 };                                                  \
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
    thunk(&range, &order, &n, &vl, &vu, &il, &iu, gers, &reltol, d, e, e2,     \
          &pivmin, &nsplit, isplit, &m, w, werr, &wl, &wu, iblock, indexw,     \
          work, iwork, &info);                                                  \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                  \
        g_##SUFFIX##_cblas_call.range != 'I' ||                                 \
        g_##SUFFIX##_cblas_call.order != 'E' ||                                 \
        g_##SUFFIX##_cblas_call.n != 2 ||                                       \
        g_##SUFFIX##_cblas_call.vl != (TYPE)23 ||                               \
        g_##SUFFIX##_cblas_call.vu != (TYPE)24 ||                               \
        g_##SUFFIX##_cblas_call.il != 1 ||                                      \
        g_##SUFFIX##_cblas_call.iu != 2 ||                                      \
        g_##SUFFIX##_cblas_call.gers != gers ||                                 \
        g_##SUFFIX##_cblas_call.reltol != (TYPE)25 ||                           \
        g_##SUFFIX##_cblas_call.d != d ||                                       \
        g_##SUFFIX##_cblas_call.e != e ||                                       \
        g_##SUFFIX##_cblas_call.e2 != e2 ||                                     \
        g_##SUFFIX##_cblas_call.pivmin != (TYPE)26 ||                           \
        g_##SUFFIX##_cblas_call.nsplit != 2 ||                                  \
        g_##SUFFIX##_cblas_call.isplit != isplit ||                             \
        g_##SUFFIX##_cblas_call.m != &m ||                                      \
        g_##SUFFIX##_cblas_call.w != w ||                                       \
        g_##SUFFIX##_cblas_call.werr != werr ||                                 \
        g_##SUFFIX##_cblas_call.wl != &wl ||                                    \
        g_##SUFFIX##_cblas_call.wu != &wu ||                                    \
        g_##SUFFIX##_cblas_call.iblock != iblock ||                             \
        g_##SUFFIX##_cblas_call.indexw != indexw ||                             \
        g_##SUFFIX##_cblas_call.work != work ||                                 \
        g_##SUFFIX##_cblas_call.iwork != iwork ||                               \
        m != (BASE) + 11 || w[0] != (TYPE)((BASE) + 12) ||                      \
        werr[0] != (TYPE)((BASE) + 13) || wl != (TYPE)((BASE) + 14) ||          \
        wu != (TYPE)((BASE) + 15) || iblock[0] != (BASE) + 16 ||                \
        indexw[0] != (BASE) + 17 || work[0] != (TYPE)((BASE) + 18) ||           \
        iwork[0] != (BASE) + 19 || info != (BASE) + 20) {                       \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference LARRD inputs correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences LARRD inputs and stores info\n"); \
    return 0;                                                                   \
}

DEFINE_LARRD_TESTS(slarrd, float, FB_OP_SLARRD, 100)
DEFINE_LARRD_TESTS(dlarrd, double, FB_OP_DLARRD, 300)

int main(void)
{
    int status = 0;

    status |= check_slarrd_fortran_to_cblas();
    status |= check_slarrd_cblas_to_fortran();
    status |= check_dlarrd_fortran_to_cblas();
    status |= check_dlarrd_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}