#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

#define DEFINE_LARRA_TESTS(SUFFIX, TYPE, OP_ID, BASE)                          \
typedef int (*fb_##SUFFIX##_cblas_fn)(int n, const TYPE *d, TYPE *e, TYPE *e2, \
                                      int *spltpnt, TYPE rtol1, TYPE rtol2);   \
typedef void (*fb_##SUFFIX##_fortran_fn)(int *n, TYPE *d, TYPE *e, TYPE *e2,   \
                                         int *spltpnt, TYPE *rtol1, TYPE *rtol2); \
static struct {                                                                  \
    int called;                                                                  \
    int n;                                                                       \
    TYPE *d;                                                                     \
    TYPE *e;                                                                     \
    TYPE *e2;                                                                    \
    int *spltpnt;                                                                \
    TYPE rtol1;                                                                  \
    TYPE rtol2;                                                                  \
} g_##SUFFIX##_fortran_call;                                                     \
static struct {                                                                  \
    int called;                                                                  \
    int n;                                                                       \
    const TYPE *d;                                                               \
    TYPE *e;                                                                     \
    TYPE *e2;                                                                    \
    int *spltpnt;                                                                \
    TYPE rtol1;                                                                  \
    TYPE rtol2;                                                                  \
} g_##SUFFIX##_cblas_call;                                                       \
static void stub_##SUFFIX##_fortran(int *n, TYPE *d, TYPE *e, TYPE *e2,        \
                                    int *spltpnt, TYPE *rtol1, TYPE *rtol2)     \
{                                                                                \
    g_##SUFFIX##_fortran_call.called += 1;                                       \
    g_##SUFFIX##_fortran_call.n = *n;                                            \
    g_##SUFFIX##_fortran_call.d = d;                                             \
    g_##SUFFIX##_fortran_call.e = e;                                             \
    g_##SUFFIX##_fortran_call.e2 = e2;                                           \
    g_##SUFFIX##_fortran_call.spltpnt = spltpnt;                                 \
    g_##SUFFIX##_fortran_call.rtol1 = *rtol1;                                    \
    g_##SUFFIX##_fortran_call.rtol2 = *rtol2;                                    \
    e[0] = (TYPE)((BASE) + 1);                                                   \
    e2[0] = (TYPE)((BASE) + 2);                                                  \
    *spltpnt = (BASE) + 3;                                                       \
}                                                                                \
static int stub_##SUFFIX##_cblas(int n, const TYPE *d, TYPE *e, TYPE *e2,      \
                                 int *spltpnt, TYPE rtol1, TYPE rtol2)          \
{                                                                                \
    g_##SUFFIX##_cblas_call.called += 1;                                         \
    g_##SUFFIX##_cblas_call.n = n;                                               \
    g_##SUFFIX##_cblas_call.d = d;                                               \
    g_##SUFFIX##_cblas_call.e = e;                                               \
    g_##SUFFIX##_cblas_call.e2 = e2;                                             \
    g_##SUFFIX##_cblas_call.spltpnt = spltpnt;                                   \
    g_##SUFFIX##_cblas_call.rtol1 = rtol1;                                       \
    g_##SUFFIX##_cblas_call.rtol2 = rtol2;                                       \
    e[0] = (TYPE)((BASE) + 11);                                                  \
    e2[0] = (TYPE)((BASE) + 12);                                                 \
    *spltpnt = (BASE) + 13;                                                      \
    return 0;                                                                    \
}                                                                                \
static int check_##SUFFIX##_fortran_to_cblas(void)                               \
{                                                                                \
    fb_backend_vtable_t vtable;                                                  \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                         \
    TYPE d[2] = { (TYPE)1, (TYPE)2 };                                            \
    TYPE e[2] = { (TYPE)3, (TYPE)4 };                                            \
    TYPE e2[2] = { (TYPE)5, (TYPE)6 };                                           \
    int spltpnt = 0;                                                             \
    TYPE rtol1 = (TYPE)7;                                                        \
    TYPE rtol2 = (TYPE)8;                                                        \
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
    if (thunk(2, d, e, e2, &spltpnt, rtol1, rtol2) != 0 ||                      \
        g_##SUFFIX##_fortran_call.called != 1 ||                                 \
        g_##SUFFIX##_fortran_call.n != 2 ||                                      \
        g_##SUFFIX##_fortran_call.d != d ||                                      \
        g_##SUFFIX##_fortran_call.e != e ||                                      \
        g_##SUFFIX##_fortran_call.e2 != e2 ||                                    \
        g_##SUFFIX##_fortran_call.spltpnt != &spltpnt ||                         \
        g_##SUFFIX##_fortran_call.rtol1 != (TYPE)7 ||                            \
        g_##SUFFIX##_fortran_call.rtol2 != (TYPE)8 ||                            \
        e[0] != (TYPE)((BASE) + 1) ||                                            \
        e2[0] != (TYPE)((BASE) + 2) ||                                           \
        spltpnt != (BASE) + 3 ||                                                 \
        d[0] != (TYPE)1 || d[1] != (TYPE)2 ||                                    \
        e[1] != (TYPE)4 || e2[1] != (TYPE)6) {                                   \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward LARRA inputs correctly\n"); \
        return 1;                                                                \
    }                                                                            \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards LARRA inputs\n"); \
    return 0;                                                                    \
}                                                                                \
static int check_##SUFFIX##_cblas_to_fortran(void)                               \
{                                                                                \
    fb_backend_vtable_t vtable;                                                  \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                       \
    int n = 2;                                                                   \
    TYPE d[2] = { (TYPE)11, (TYPE)12 };                                          \
    TYPE e[2] = { (TYPE)13, (TYPE)14 };                                          \
    TYPE e2[2] = { (TYPE)15, (TYPE)16 };                                         \
    int spltpnt = 0;                                                             \
    TYPE rtol1 = (TYPE)17;                                                       \
    TYPE rtol2 = (TYPE)18;                                                       \
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
    thunk(&n, d, e, e2, &spltpnt, &rtol1, &rtol2);                               \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                   \
        g_##SUFFIX##_cblas_call.n != 2 ||                                        \
        g_##SUFFIX##_cblas_call.d != d ||                                        \
        g_##SUFFIX##_cblas_call.e != e ||                                        \
        g_##SUFFIX##_cblas_call.e2 != e2 ||                                      \
        g_##SUFFIX##_cblas_call.spltpnt != &spltpnt ||                           \
        g_##SUFFIX##_cblas_call.rtol1 != (TYPE)17 ||                             \
        g_##SUFFIX##_cblas_call.rtol2 != (TYPE)18 ||                             \
        e[0] != (TYPE)((BASE) + 11) ||                                           \
        e2[0] != (TYPE)((BASE) + 12) ||                                          \
        spltpnt != (BASE) + 13 ||                                                \
        d[0] != (TYPE)11 || d[1] != (TYPE)12 ||                                  \
        e[1] != (TYPE)14 || e2[1] != (TYPE)16) {                                 \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference LARRA inputs correctly\n"); \
        return 1;                                                                \
    }                                                                            \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences LARRA inputs\n"); \
    return 0;                                                                    \
}

DEFINE_LARRA_TESTS(slarra, float, FB_OP_SLARRA, 100)
DEFINE_LARRA_TESTS(dlarra, double, FB_OP_DLARRA, 300)

int main(void)
{
    int status = 0;

    status |= check_slarra_fortran_to_cblas();
    status |= check_slarra_cblas_to_fortran();
    status |= check_dlarra_fortran_to_cblas();
    status |= check_dlarra_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}