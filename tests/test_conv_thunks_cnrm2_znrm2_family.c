#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

#define DEFINE_COMPLEX_NRM2_ALIAS_TESTS(SUFFIX, RTYPE, CTYPE, OP_ID, BASE)       \
typedef RTYPE (*fb_##SUFFIX##_cblas_fn)(int n, const CTYPE *x, int incx);       \
typedef RTYPE (*fb_##SUFFIX##_fortran_fn)(const int *n, const CTYPE *x,         \
                                          const int *incx);                      \
static struct {                                                                    \
    int called;                                                                    \
    int n;                                                                         \
    const CTYPE *x;                                                                \
    int incx;                                                                      \
} g_##SUFFIX##_fortran_call;                                                       \
static struct {                                                                    \
    int called;                                                                    \
    int n;                                                                         \
    const CTYPE *x;                                                                \
    int incx;                                                                      \
} g_##SUFFIX##_cblas_call;                                                         \
static RTYPE stub_##SUFFIX##_fortran(const int *n, const CTYPE *x,              \
                                     const int *incx)                            \
{                                                                                 \
    g_##SUFFIX##_fortran_call.called += 1;                                        \
    g_##SUFFIX##_fortran_call.n = *n;                                             \
    g_##SUFFIX##_fortran_call.x = x;                                              \
    g_##SUFFIX##_fortran_call.incx = *incx;                                       \
    return (RTYPE)((BASE) + 1);                                                   \
}                                                                                 \
static RTYPE stub_##SUFFIX##_cblas(int n, const CTYPE *x, int incx)             \
{                                                                                 \
    g_##SUFFIX##_cblas_call.called += 1;                                          \
    g_##SUFFIX##_cblas_call.n = n;                                                \
    g_##SUFFIX##_cblas_call.x = x;                                                \
    g_##SUFFIX##_cblas_call.incx = incx;                                          \
    return (RTYPE)((BASE) + 2);                                                   \
}                                                                                 \
static int check_##SUFFIX##_fortran_to_cblas(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                          \
    CTYPE x[2] = { 0 };                                                           \
    RTYPE result;                                                                 \
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
    result = thunk(5, x, -1);                                                     \
    if (result != (RTYPE)((BASE) + 1) ||                                          \
        g_##SUFFIX##_fortran_call.called != 1 ||                                  \
        g_##SUFFIX##_fortran_call.n != 5 ||                                       \
        g_##SUFFIX##_fortran_call.x != x ||                                       \
        g_##SUFFIX##_fortran_call.incx != -1) {                                   \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward complex NRM2 inputs correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards complex NRM2 inputs\n"); \
    return 0;                                                                     \
}                                                                                 \
static int check_##SUFFIX##_cblas_to_fortran(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                        \
    int n = 7;                                                                    \
    int incx = 2;                                                                 \
    CTYPE x[2] = { 0 };                                                           \
    RTYPE result;                                                                 \
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
    result = thunk(&n, x, &incx);                                                 \
    if (result != (RTYPE)((BASE) + 2) ||                                          \
        g_##SUFFIX##_cblas_call.called != 1 ||                                    \
        g_##SUFFIX##_cblas_call.n != 7 ||                                         \
        g_##SUFFIX##_cblas_call.x != x ||                                         \
        g_##SUFFIX##_cblas_call.incx != 2) {                                      \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference complex NRM2 inputs correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences complex NRM2 inputs\n"); \
    return 0;                                                                     \
}

DEFINE_COMPLEX_NRM2_ALIAS_TESTS(cnrm2, float, fb_complex_float_t, FB_OP_CNRM2, 100)
DEFINE_COMPLEX_NRM2_ALIAS_TESTS(znrm2, double, fb_complex_double_t, FB_OP_ZNRM2, 300)

int main(void)
{
    int status = 0;

    status |= check_cnrm2_fortran_to_cblas();
    status |= check_cnrm2_cblas_to_fortran();
    status |= check_znrm2_fortran_to_cblas();
    status |= check_znrm2_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}