#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

#define DEFINE_LARRC_TESTS(SUFFIX, TYPE, OP_ID, BASE)                          \
typedef int (*fb_##SUFFIX##_cblas_fn)(                                         \
    char jobt, int n, TYPE vl, TYPE vu, const TYPE *d, const TYPE *e,          \
    TYPE pivmin, int *eigcnt, int *lcnt, int *rcnt);                           \
typedef void (*fb_##SUFFIX##_fortran_fn)(                                      \
    char *jobt, int *n, TYPE *vl, TYPE *vu, TYPE *d, TYPE *e, TYPE *pivmin,    \
    int *eigcnt, int *lcnt, int *rcnt, int *info);                             \
static struct {                                                                 \
    int called;                                                                 \
    char jobt;                                                                  \
    int n;                                                                      \
    TYPE vl;                                                                    \
    TYPE vu;                                                                    \
    TYPE *d;                                                                    \
    TYPE *e;                                                                    \
    TYPE pivmin;                                                                \
    int *eigcnt;                                                                \
    int *lcnt;                                                                  \
    int *rcnt;                                                                  \
    int *info_ptr;                                                              \
} g_##SUFFIX##_fortran_call;                                                    \
static struct {                                                                 \
    int called;                                                                 \
    char jobt;                                                                  \
    int n;                                                                      \
    TYPE vl;                                                                    \
    TYPE vu;                                                                    \
    const TYPE *d;                                                              \
    const TYPE *e;                                                              \
    TYPE pivmin;                                                                \
    int *eigcnt;                                                                \
    int *lcnt;                                                                  \
    int *rcnt;                                                                  \
} g_##SUFFIX##_cblas_call;                                                      \
static void stub_##SUFFIX##_fortran(                                            \
    char *jobt, int *n, TYPE *vl, TYPE *vu, TYPE *d, TYPE *e, TYPE *pivmin,    \
    int *eigcnt, int *lcnt, int *rcnt, int *info)                               \
{                                                                               \
    g_##SUFFIX##_fortran_call.called += 1;                                      \
    g_##SUFFIX##_fortran_call.jobt = *jobt;                                     \
    g_##SUFFIX##_fortran_call.n = *n;                                           \
    g_##SUFFIX##_fortran_call.vl = *vl;                                         \
    g_##SUFFIX##_fortran_call.vu = *vu;                                         \
    g_##SUFFIX##_fortran_call.d = d;                                            \
    g_##SUFFIX##_fortran_call.e = e;                                            \
    g_##SUFFIX##_fortran_call.pivmin = *pivmin;                                 \
    g_##SUFFIX##_fortran_call.eigcnt = eigcnt;                                  \
    g_##SUFFIX##_fortran_call.lcnt = lcnt;                                      \
    g_##SUFFIX##_fortran_call.rcnt = rcnt;                                      \
    g_##SUFFIX##_fortran_call.info_ptr = info;                                  \
    *eigcnt = (BASE) + 1;                                                       \
    *lcnt = (BASE) + 2;                                                         \
    *rcnt = (BASE) + 3;                                                         \
    *info = (BASE) + 4;                                                         \
}                                                                               \
static int stub_##SUFFIX##_cblas(                                               \
    char jobt, int n, TYPE vl, TYPE vu, const TYPE *d, const TYPE *e,          \
    TYPE pivmin, int *eigcnt, int *lcnt, int *rcnt)                            \
{                                                                               \
    g_##SUFFIX##_cblas_call.called += 1;                                        \
    g_##SUFFIX##_cblas_call.jobt = jobt;                                        \
    g_##SUFFIX##_cblas_call.n = n;                                              \
    g_##SUFFIX##_cblas_call.vl = vl;                                            \
    g_##SUFFIX##_cblas_call.vu = vu;                                            \
    g_##SUFFIX##_cblas_call.d = d;                                              \
    g_##SUFFIX##_cblas_call.e = e;                                              \
    g_##SUFFIX##_cblas_call.pivmin = pivmin;                                    \
    g_##SUFFIX##_cblas_call.eigcnt = eigcnt;                                    \
    g_##SUFFIX##_cblas_call.lcnt = lcnt;                                        \
    g_##SUFFIX##_cblas_call.rcnt = rcnt;                                        \
    *eigcnt = (BASE) + 11;                                                      \
    *lcnt = (BASE) + 12;                                                        \
    *rcnt = (BASE) + 13;                                                        \
    return (BASE) + 14;                                                         \
}                                                                               \
static int check_##SUFFIX##_fortran_to_cblas(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                        \
    TYPE d[2] = { (TYPE)1, (TYPE)2 };                                           \
    TYPE e[2] = { (TYPE)3, (TYPE)4 };                                           \
    int eigcnt = 0;                                                             \
    int lcnt = 0;                                                               \
    int rcnt = 0;                                                               \
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
    if (thunk('T', 2, (TYPE)5, (TYPE)6, d, e, (TYPE)7, &eigcnt, &lcnt, &rcnt)  \
            != (BASE) + 4 ||                                                    \
        g_##SUFFIX##_fortran_call.called != 1 ||                                \
        g_##SUFFIX##_fortran_call.jobt != 'T' ||                                \
        g_##SUFFIX##_fortran_call.n != 2 ||                                     \
        g_##SUFFIX##_fortran_call.vl != (TYPE)5 ||                              \
        g_##SUFFIX##_fortran_call.vu != (TYPE)6 ||                              \
        g_##SUFFIX##_fortran_call.d != d ||                                     \
        g_##SUFFIX##_fortran_call.e != e ||                                     \
        g_##SUFFIX##_fortran_call.pivmin != (TYPE)7 ||                          \
        g_##SUFFIX##_fortran_call.eigcnt != &eigcnt ||                          \
        g_##SUFFIX##_fortran_call.lcnt != &lcnt ||                              \
        g_##SUFFIX##_fortran_call.rcnt != &rcnt ||                              \
        g_##SUFFIX##_fortran_call.info_ptr == NULL ||                           \
        eigcnt != (BASE) + 1 || lcnt != (BASE) + 2 || rcnt != (BASE) + 3) {    \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward LARRC inputs correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards LARRC inputs and returns info\n"); \
    return 0;                                                                   \
}                                                                               \
static int check_##SUFFIX##_cblas_to_fortran(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                      \
    char jobt = 'T';                                                            \
    int n = 2;                                                                  \
    TYPE vl = (TYPE)5;                                                          \
    TYPE vu = (TYPE)6;                                                          \
    TYPE d[2] = { (TYPE)21, (TYPE)22 };                                         \
    TYPE e[2] = { (TYPE)23, (TYPE)24 };                                         \
    TYPE pivmin = (TYPE)7;                                                      \
    int eigcnt = 0;                                                             \
    int lcnt = 0;                                                               \
    int rcnt = 0;                                                               \
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
    thunk(&jobt, &n, &vl, &vu, d, e, &pivmin, &eigcnt, &lcnt, &rcnt, &info);   \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                  \
        g_##SUFFIX##_cblas_call.jobt != 'T' ||                                  \
        g_##SUFFIX##_cblas_call.n != 2 ||                                       \
        g_##SUFFIX##_cblas_call.vl != (TYPE)5 ||                                \
        g_##SUFFIX##_cblas_call.vu != (TYPE)6 ||                                \
        g_##SUFFIX##_cblas_call.d != d ||                                       \
        g_##SUFFIX##_cblas_call.e != e ||                                       \
        g_##SUFFIX##_cblas_call.pivmin != (TYPE)7 ||                            \
        g_##SUFFIX##_cblas_call.eigcnt != &eigcnt ||                            \
        g_##SUFFIX##_cblas_call.lcnt != &lcnt ||                                \
        g_##SUFFIX##_cblas_call.rcnt != &rcnt ||                                \
        eigcnt != (BASE) + 11 || lcnt != (BASE) + 12 || rcnt != (BASE) + 13 || \
        info != (BASE) + 14) {                                                  \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference LARRC inputs correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences LARRC inputs and stores info\n"); \
    return 0;                                                                   \
}

DEFINE_LARRC_TESTS(slarrc, float, FB_OP_SLARRC, 100)
DEFINE_LARRC_TESTS(dlarrc, double, FB_OP_DLARRC, 300)

int main(void)
{
    int status = 0;

    status |= check_slarrc_fortran_to_cblas();
    status |= check_slarrc_cblas_to_fortran();
    status |= check_dlarrc_fortran_to_cblas();
    status |= check_dlarrc_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}