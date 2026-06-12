#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

#define DEFINE_LASQ6_TESTS(SUFFIX, TYPE, OP_ID, BASE)                              \
typedef int (*fb_##SUFFIX##_cblas_fn)(int i0, int n0, TYPE *z, int pp);          \
typedef int (*fb_##SUFFIX##_fortran_fn)(int *i0, int *n0, TYPE *z, int *pp);     \
static struct {                                                                    \
    int called;                                                                    \
    int i0;                                                                        \
    int n0;                                                                        \
    TYPE *z;                                                                       \
    int pp;                                                                        \
} g_##SUFFIX##_fortran_call;                                                       \
static struct {                                                                    \
    int called;                                                                    \
    int i0;                                                                        \
    int n0;                                                                        \
    TYPE *z;                                                                       \
    int pp;                                                                        \
} g_##SUFFIX##_cblas_call;                                                         \
static int stub_##SUFFIX##_fortran(int *i0, int *n0, TYPE *z, int *pp) {        \
    g_##SUFFIX##_fortran_call.called += 1;                                        \
    g_##SUFFIX##_fortran_call.i0 = *i0;                                           \
    g_##SUFFIX##_fortran_call.n0 = *n0;                                           \
    g_##SUFFIX##_fortran_call.z = z;                                              \
    g_##SUFFIX##_fortran_call.pp = *pp;                                           \
    z[0] = (TYPE)((BASE) + 1);                                                    \
    return (BASE) + 2;                                                            \
}                                                                                 \
static int stub_##SUFFIX##_cblas(int i0, int n0, TYPE *z, int pp) {             \
    g_##SUFFIX##_cblas_call.called += 1;                                         \
    g_##SUFFIX##_cblas_call.i0 = i0;                                             \
    g_##SUFFIX##_cblas_call.n0 = n0;                                             \
    g_##SUFFIX##_cblas_call.z = z;                                               \
    g_##SUFFIX##_cblas_call.pp = pp;                                             \
    z[0] = (TYPE)((BASE) + 3);                                                   \
    return (BASE) + 4;                                                           \
}                                                                                \
static int check_##SUFFIX##_fortran_to_cblas(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                          \
    TYPE z[4] = { (TYPE)1, (TYPE)2, (TYPE)3, (TYPE)4 };                           \
    int rc = 0;                                                                   \
    memset(&vtable, 0, sizeof(vtable));                                           \
    memset(&g_##SUFFIX##_fortran_call, 0, sizeof(g_##SUFFIX##_fortran_call));     \
    vtable.ext_ops[OP_ID][FB_CONV_FORTRAN] =                                      \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_fortran;                   \
    fb_install_conv_thunks(&vtable, OP_ID);                                       \
    thunk = (fb_##SUFFIX##_cblas_fn)vtable.ext_ops[OP_ID][FB_CONV_CBLAS];         \
    if (!thunk) {                                                                 \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk was not installed\n"); \
        return 1;                                                                 \
    }                                                                             \
    rc = thunk(1, 4, z, 1);                                                       \
    if (rc != (BASE) + 2 || g_##SUFFIX##_fortran_call.called != 1 ||             \
        g_##SUFFIX##_fortran_call.i0 != 1 ||                                      \
        g_##SUFFIX##_fortran_call.n0 != 4 ||                                      \
        g_##SUFFIX##_fortran_call.z != z ||                                       \
        g_##SUFFIX##_fortran_call.pp != 1 ||                                      \
        z[0] != (TYPE)((BASE) + 1)) {                                             \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not bridge LASQ6 correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk bridges LASQ6 correctly\n"); \
    return 0;                                                                     \
}                                                                                 \
static int check_##SUFFIX##_cblas_to_fortran(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                        \
    int i0 = 1;                                                                   \
    int n0 = 4;                                                                   \
    TYPE z[4] = { (TYPE)1, (TYPE)2, (TYPE)3, (TYPE)4 };                           \
    int pp = 1;                                                                   \
    int rc = 0;                                                                   \
    memset(&vtable, 0, sizeof(vtable));                                           \
    memset(&g_##SUFFIX##_cblas_call, 0, sizeof(g_##SUFFIX##_cblas_call));         \
    vtable.ext_ops[OP_ID][FB_CONV_CBLAS] =                                        \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_cblas;                     \
    fb_install_conv_thunks(&vtable, OP_ID);                                       \
    thunk = (fb_##SUFFIX##_fortran_fn)vtable.ext_ops[OP_ID][FB_CONV_FORTRAN];     \
    if (!thunk) {                                                                 \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk was not installed\n"); \
        return 1;                                                                 \
    }                                                                             \
    rc = thunk(&i0, &n0, z, &pp);                                                 \
    if (rc != (BASE) + 4 || g_##SUFFIX##_cblas_call.called != 1 ||               \
        g_##SUFFIX##_cblas_call.i0 != 1 ||                                        \
        g_##SUFFIX##_cblas_call.n0 != 4 ||                                        \
        g_##SUFFIX##_cblas_call.z != z ||                                         \
        g_##SUFFIX##_cblas_call.pp != 1 ||                                        \
        z[0] != (TYPE)((BASE) + 3)) {                                             \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not bridge LASQ6 correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk bridges LASQ6 correctly\n"); \
    return 0;                                                                     \
}

DEFINE_LASQ6_TESTS(slasq6, float, FB_OP_SLASQ6, 100)
DEFINE_LASQ6_TESTS(dlasq6, double, FB_OP_DLASQ6, 300)

int main(void)
{
    int status = 0;

    status |= check_slasq6_fortran_to_cblas();
    status |= check_slasq6_cblas_to_fortran();
    status |= check_dlasq6_fortran_to_cblas();
    status |= check_dlasq6_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}