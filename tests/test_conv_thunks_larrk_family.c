#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

#define DEFINE_LARRK_TESTS(SUFFIX, TYPE, OP_ID, BASE)                          \
typedef int (*fb_##SUFFIX##_cblas_fn)(                                         \
    int n, int iw, TYPE gl, TYPE gu, const TYPE *d, const TYPE *e2,           \
    TYPE pivmin, TYPE reltol, TYPE *w, TYPE *werr);                           \
typedef void (*fb_##SUFFIX##_fortran_fn)(                                      \
    int *n, int *iw, TYPE *gl, TYPE *gu, TYPE *d, TYPE *e2, TYPE *pivmin,     \
    TYPE *reltol, TYPE *w, TYPE *werr, int *info);                            \
static struct {                                                                 \
    int called;                                                                 \
    int n;                                                                      \
    int iw;                                                                     \
    TYPE gl;                                                                    \
    TYPE gu;                                                                    \
    TYPE *d;                                                                    \
    TYPE *e2;                                                                   \
    TYPE pivmin;                                                                \
    TYPE reltol;                                                                \
    TYPE *w;                                                                    \
    TYPE *werr;                                                                 \
    int *info_ptr;                                                              \
} g_##SUFFIX##_fortran_call;                                                    \
static struct {                                                                 \
    int called;                                                                 \
    int n;                                                                      \
    int iw;                                                                     \
    TYPE gl;                                                                    \
    TYPE gu;                                                                    \
    const TYPE *d;                                                              \
    const TYPE *e2;                                                             \
    TYPE pivmin;                                                                \
    TYPE reltol;                                                                \
    TYPE *w;                                                                    \
    TYPE *werr;                                                                 \
} g_##SUFFIX##_cblas_call;                                                      \
static void stub_##SUFFIX##_fortran(                                            \
    int *n, int *iw, TYPE *gl, TYPE *gu, TYPE *d, TYPE *e2, TYPE *pivmin,     \
    TYPE *reltol, TYPE *w, TYPE *werr, int *info)                              \
{                                                                               \
    g_##SUFFIX##_fortran_call.called += 1;                                      \
    g_##SUFFIX##_fortran_call.n = *n;                                           \
    g_##SUFFIX##_fortran_call.iw = *iw;                                         \
    g_##SUFFIX##_fortran_call.gl = *gl;                                         \
    g_##SUFFIX##_fortran_call.gu = *gu;                                         \
    g_##SUFFIX##_fortran_call.d = d;                                            \
    g_##SUFFIX##_fortran_call.e2 = e2;                                          \
    g_##SUFFIX##_fortran_call.pivmin = *pivmin;                                 \
    g_##SUFFIX##_fortran_call.reltol = *reltol;                                 \
    g_##SUFFIX##_fortran_call.w = w;                                            \
    g_##SUFFIX##_fortran_call.werr = werr;                                      \
    g_##SUFFIX##_fortran_call.info_ptr = info;                                  \
    *w = (TYPE)((BASE) + 1);                                                    \
    *werr = (TYPE)((BASE) + 2);                                                 \
    *info = (BASE) + 3;                                                         \
}                                                                               \
static int stub_##SUFFIX##_cblas(                                               \
    int n, int iw, TYPE gl, TYPE gu, const TYPE *d, const TYPE *e2,           \
    TYPE pivmin, TYPE reltol, TYPE *w, TYPE *werr)                             \
{                                                                               \
    g_##SUFFIX##_cblas_call.called += 1;                                        \
    g_##SUFFIX##_cblas_call.n = n;                                              \
    g_##SUFFIX##_cblas_call.iw = iw;                                            \
    g_##SUFFIX##_cblas_call.gl = gl;                                            \
    g_##SUFFIX##_cblas_call.gu = gu;                                            \
    g_##SUFFIX##_cblas_call.d = d;                                              \
    g_##SUFFIX##_cblas_call.e2 = e2;                                            \
    g_##SUFFIX##_cblas_call.pivmin = pivmin;                                    \
    g_##SUFFIX##_cblas_call.reltol = reltol;                                    \
    g_##SUFFIX##_cblas_call.w = w;                                              \
    g_##SUFFIX##_cblas_call.werr = werr;                                        \
    *w = (TYPE)((BASE) + 4);                                                    \
    *werr = (TYPE)((BASE) + 5);                                                 \
    return (BASE) + 6;                                                          \
}                                                                               \
static int check_##SUFFIX##_fortran_to_cblas(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                        \
    TYPE d[2] = { (TYPE)1, (TYPE)2 };                                           \
    TYPE e2[2] = { (TYPE)3, (TYPE)4 };                                          \
    TYPE w = (TYPE)0;                                                           \
    TYPE werr = (TYPE)0;                                                        \
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
    if (thunk(2, 1, (TYPE)5, (TYPE)6, d, e2, (TYPE)7, (TYPE)8, &w, &werr) !=  \
            (BASE) + 3 ||                                                       \
        g_##SUFFIX##_fortran_call.called != 1 ||                                \
        g_##SUFFIX##_fortran_call.n != 2 ||                                     \
        g_##SUFFIX##_fortran_call.iw != 1 ||                                    \
        g_##SUFFIX##_fortran_call.gl != (TYPE)5 ||                              \
        g_##SUFFIX##_fortran_call.gu != (TYPE)6 ||                              \
        g_##SUFFIX##_fortran_call.d != d ||                                     \
        g_##SUFFIX##_fortran_call.e2 != e2 ||                                   \
        g_##SUFFIX##_fortran_call.pivmin != (TYPE)7 ||                          \
        g_##SUFFIX##_fortran_call.reltol != (TYPE)8 ||                          \
        g_##SUFFIX##_fortran_call.w != &w ||                                    \
        g_##SUFFIX##_fortran_call.werr != &werr ||                              \
        g_##SUFFIX##_fortran_call.info_ptr == NULL ||                           \
        w != (TYPE)((BASE) + 1) || werr != (TYPE)((BASE) + 2)) {                \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward LARRK inputs correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards LARRK inputs and returns info\n"); \
    return 0;                                                                   \
}                                                                               \
static int check_##SUFFIX##_cblas_to_fortran(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                      \
    int n = 2;                                                                  \
    int iw = 1;                                                                 \
    TYPE gl = (TYPE)5;                                                          \
    TYPE gu = (TYPE)6;                                                          \
    TYPE d[2] = { (TYPE)1, (TYPE)2 };                                           \
    TYPE e2[2] = { (TYPE)3, (TYPE)4 };                                          \
    TYPE pivmin = (TYPE)7;                                                      \
    TYPE reltol = (TYPE)8;                                                      \
    TYPE w = (TYPE)0;                                                           \
    TYPE werr = (TYPE)0;                                                        \
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
    thunk(&n, &iw, &gl, &gu, d, e2, &pivmin, &reltol, &w, &werr, &info);       \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                  \
        g_##SUFFIX##_cblas_call.n != 2 ||                                       \
        g_##SUFFIX##_cblas_call.iw != 1 ||                                      \
        g_##SUFFIX##_cblas_call.gl != (TYPE)5 ||                                \
        g_##SUFFIX##_cblas_call.gu != (TYPE)6 ||                                \
        g_##SUFFIX##_cblas_call.d != d ||                                       \
        g_##SUFFIX##_cblas_call.e2 != e2 ||                                     \
        g_##SUFFIX##_cblas_call.pivmin != (TYPE)7 ||                            \
        g_##SUFFIX##_cblas_call.reltol != (TYPE)8 ||                            \
        g_##SUFFIX##_cblas_call.w != &w ||                                      \
        g_##SUFFIX##_cblas_call.werr != &werr ||                                \
        w != (TYPE)((BASE) + 4) || werr != (TYPE)((BASE) + 5) ||                \
        info != (BASE) + 6) {                                                   \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference LARRK inputs correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences LARRK inputs and stores info\n"); \
    return 0;                                                                   \
}

DEFINE_LARRK_TESTS(slarrk, float, FB_OP_SLARRK, 100)
DEFINE_LARRK_TESTS(dlarrk, double, FB_OP_DLARRK, 300)

int main(void)
{
    int status = 0;

    status |= check_slarrk_fortran_to_cblas();
    status |= check_slarrk_cblas_to_fortran();
    status |= check_dlarrk_fortran_to_cblas();
    status |= check_dlarrk_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}