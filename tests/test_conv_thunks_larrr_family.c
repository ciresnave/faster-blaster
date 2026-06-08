#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

#define DEFINE_LARRR_TESTS(SUFFIX, TYPE, OP_ID, BASE)                          \
typedef int (*fb_##SUFFIX##_cblas_fn)(int n, const TYPE *d, TYPE *e);          \
typedef void (*fb_##SUFFIX##_fortran_fn)(int *n, TYPE *d, TYPE *e, int *info); \
static struct {                                                                 \
    int called;                                                                 \
    int n;                                                                      \
    TYPE *d;                                                                    \
    TYPE *e;                                                                    \
    int *info_ptr;                                                              \
} g_##SUFFIX##_fortran_call;                                                    \
static struct {                                                                 \
    int called;                                                                 \
    int n;                                                                      \
    const TYPE *d;                                                              \
    TYPE *e;                                                                    \
} g_##SUFFIX##_cblas_call;                                                      \
static void stub_##SUFFIX##_fortran(int *n, TYPE *d, TYPE *e, int *info)       \
{                                                                               \
    g_##SUFFIX##_fortran_call.called += 1;                                      \
    g_##SUFFIX##_fortran_call.n = *n;                                           \
    g_##SUFFIX##_fortran_call.d = d;                                            \
    g_##SUFFIX##_fortran_call.e = e;                                            \
    g_##SUFFIX##_fortran_call.info_ptr = info;                                  \
    e[*n - 1] = (TYPE)0;                                                        \
    *info = (BASE) + 1;                                                         \
}                                                                               \
static int stub_##SUFFIX##_cblas(int n, const TYPE *d, TYPE *e)                \
{                                                                               \
    g_##SUFFIX##_cblas_call.called += 1;                                        \
    g_##SUFFIX##_cblas_call.n = n;                                              \
    g_##SUFFIX##_cblas_call.d = d;                                              \
    g_##SUFFIX##_cblas_call.e = e;                                              \
    e[n - 1] = (TYPE)0;                                                         \
    return (BASE) + 2;                                                          \
}                                                                               \
static int check_##SUFFIX##_fortran_to_cblas(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                        \
    TYPE d[3] = { (TYPE)1, (TYPE)2, (TYPE)3 };                                  \
    TYPE e[3] = { (TYPE)4, (TYPE)5, (TYPE)6 };                                  \
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
    if (thunk(3, d, e) != (BASE) + 1 ||                                         \
        g_##SUFFIX##_fortran_call.called != 1 ||                                \
        g_##SUFFIX##_fortran_call.n != 3 ||                                     \
        g_##SUFFIX##_fortran_call.d != d ||                                     \
        g_##SUFFIX##_fortran_call.e != e ||                                     \
        g_##SUFFIX##_fortran_call.info_ptr == NULL ||                           \
        e[2] != (TYPE)0) {                                                      \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward LARRR inputs correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards LARRR inputs and returns info\n"); \
    return 0;                                                                   \
}                                                                               \
static int check_##SUFFIX##_cblas_to_fortran(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                      \
    int n = 3;                                                                  \
    TYPE d[3] = { (TYPE)1, (TYPE)2, (TYPE)3 };                                  \
    TYPE e[3] = { (TYPE)4, (TYPE)5, (TYPE)6 };                                  \
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
    thunk(&n, d, e, &info);                                                     \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                  \
        g_##SUFFIX##_cblas_call.n != 3 ||                                       \
        g_##SUFFIX##_cblas_call.d != d ||                                       \
        g_##SUFFIX##_cblas_call.e != e ||                                       \
        e[2] != (TYPE)0 ||                                                      \
        info != (BASE) + 2) {                                                   \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference LARRR inputs correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences LARRR inputs and stores info\n"); \
    return 0;                                                                   \
}

DEFINE_LARRR_TESTS(slarrr, float, FB_OP_SLARRR, 100)
DEFINE_LARRR_TESTS(dlarrr, double, FB_OP_DLARRR, 300)

int main(void)
{
    int status = 0;

    status |= check_slarrr_fortran_to_cblas();
    status |= check_slarrr_cblas_to_fortran();
    status |= check_dlarrr_fortran_to_cblas();
    status |= check_dlarrr_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}