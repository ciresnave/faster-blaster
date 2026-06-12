#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

#define DEFINE_LASQ1_TESTS(SUFFIX, TYPE, OP_ID, BASE)                            \
typedef int (*fb_##SUFFIX##_cblas_fn)(int n, TYPE *d, TYPE *e, TYPE *work);     \
typedef void (*fb_##SUFFIX##_fortran_fn)(int *n, TYPE *d, TYPE *e, TYPE *work,  \
                                         int *info);                             \
static struct {                                                                   \
    int called;                                                                   \
    int n;                                                                        \
    TYPE *d;                                                                      \
    TYPE *e;                                                                      \
    TYPE *work;                                                                   \
} g_##SUFFIX##_fortran_call;                                                      \
static struct {                                                                   \
    int called;                                                                   \
    int n;                                                                        \
    TYPE *d;                                                                      \
    TYPE *e;                                                                      \
    TYPE *work;                                                                   \
} g_##SUFFIX##_cblas_call;                                                        \
static void stub_##SUFFIX##_fortran(int *n, TYPE *d, TYPE *e, TYPE *work, int *info) { \
    g_##SUFFIX##_fortran_call.called += 1;                                        \
    g_##SUFFIX##_fortran_call.n = *n;                                             \
    g_##SUFFIX##_fortran_call.d = d;                                              \
    g_##SUFFIX##_fortran_call.e = e;                                              \
    g_##SUFFIX##_fortran_call.work = work;                                        \
    d[0] = (TYPE)((BASE) + 1);                                                    \
    e[0] = (TYPE)((BASE) + 2);                                                    \
    work[0] = (TYPE)((BASE) + 3);                                                 \
    *info = (BASE) + 4;                                                           \
}                                                                                 \
static int stub_##SUFFIX##_cblas(int n, TYPE *d, TYPE *e, TYPE *work) {          \
    g_##SUFFIX##_cblas_call.called += 1;                                          \
    g_##SUFFIX##_cblas_call.n = n;                                                \
    g_##SUFFIX##_cblas_call.d = d;                                                \
    g_##SUFFIX##_cblas_call.e = e;                                                \
    g_##SUFFIX##_cblas_call.work = work;                                          \
    d[0] = (TYPE)((BASE) + 5);                                                    \
    e[0] = (TYPE)((BASE) + 6);                                                    \
    work[0] = (TYPE)((BASE) + 7);                                                 \
    return (BASE) + 8;                                                            \
}                                                                                 \
static int check_##SUFFIX##_fortran_to_cblas(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                          \
    TYPE d[2] = { (TYPE)11, (TYPE)12 };                                           \
    TYPE e[2] = { (TYPE)13, (TYPE)14 };                                           \
    TYPE work[2] = { (TYPE)15, (TYPE)16 };                                        \
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
    if (thunk(2, d, e, work) != (BASE) + 4 ||                                    \
        g_##SUFFIX##_fortran_call.called != 1 ||                                  \
        g_##SUFFIX##_fortran_call.n != 2 ||                                       \
        g_##SUFFIX##_fortran_call.d != d ||                                       \
        g_##SUFFIX##_fortran_call.e != e ||                                       \
        g_##SUFFIX##_fortran_call.work != work ||                                 \
        d[0] != (TYPE)((BASE) + 1) ||                                             \
        e[0] != (TYPE)((BASE) + 2) ||                                             \
        work[0] != (TYPE)((BASE) + 3)) {                                          \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward LASQ1 inputs correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards LASQ1 inputs\n"); \
    return 0;                                                                     \
}                                                                                 \
static int check_##SUFFIX##_cblas_to_fortran(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                        \
    int n = 2;                                                                    \
    TYPE d[2] = { (TYPE)21, (TYPE)22 };                                           \
    TYPE e[2] = { (TYPE)23, (TYPE)24 };                                           \
    TYPE work[2] = { (TYPE)25, (TYPE)26 };                                        \
    int info = -1;                                                                \
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
    thunk(&n, d, e, work, &info);                                                 \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                    \
        g_##SUFFIX##_cblas_call.n != 2 ||                                         \
        g_##SUFFIX##_cblas_call.d != d ||                                         \
        g_##SUFFIX##_cblas_call.e != e ||                                         \
        g_##SUFFIX##_cblas_call.work != work ||                                   \
        info != (BASE) + 8 ||                                                     \
        d[0] != (TYPE)((BASE) + 5) ||                                             \
        e[0] != (TYPE)((BASE) + 6) ||                                             \
        work[0] != (TYPE)((BASE) + 7)) {                                          \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference LASQ1 inputs correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences LASQ1 inputs\n"); \
    return 0;                                                                     \
}

DEFINE_LASQ1_TESTS(slasq1, float, FB_OP_SLASQ1, 100)
DEFINE_LASQ1_TESTS(dlasq1, double, FB_OP_DLASQ1, 300)

int main(void)
{
    int status = 0;

    status |= check_slasq1_fortran_to_cblas();
    status |= check_slasq1_cblas_to_fortran();
    status |= check_dlasq1_fortran_to_cblas();
    status |= check_dlasq1_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}