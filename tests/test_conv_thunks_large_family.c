#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

#define DEFINE_LARGE_TESTS(SUFFIX, TYPE, OP_ID, BASE)                          \
typedef int (*fb_##SUFFIX##_cblas_fn)(int n, TYPE *a, int lda, int *iseed,     \
                                      TYPE *work);                             \
typedef void (*fb_##SUFFIX##_fortran_fn)(int *n, TYPE *a, int *lda, int *iseed, \
                                         TYPE *work, int *info);               \
static struct {                                                                 \
    int called;                                                                 \
    int n;                                                                      \
    TYPE *a;                                                                    \
    int lda;                                                                    \
    int *iseed;                                                                 \
    TYPE *work;                                                                 \
} g_##SUFFIX##_fortran_call;                                                    \
static struct {                                                                 \
    int called;                                                                 \
    int n;                                                                      \
    TYPE *a;                                                                    \
    int lda;                                                                    \
    int *iseed;                                                                 \
    TYPE *work;                                                                 \
} g_##SUFFIX##_cblas_call;                                                      \
static void stub_##SUFFIX##_fortran(int *n, TYPE *a, int *lda, int *iseed,     \
                                    TYPE *work, int *info)                     \
{                                                                               \
    g_##SUFFIX##_fortran_call.called += 1;                                      \
    g_##SUFFIX##_fortran_call.n = *n;                                           \
    g_##SUFFIX##_fortran_call.a = a;                                            \
    g_##SUFFIX##_fortran_call.lda = *lda;                                       \
    g_##SUFFIX##_fortran_call.iseed = iseed;                                    \
    g_##SUFFIX##_fortran_call.work = work;                                      \
    iseed[0] = (BASE) + 1;                                                      \
    a[0] = (TYPE)((BASE) + 2);                                                  \
    work[0] = (TYPE)((BASE) + 3);                                               \
    *info = (BASE) + 4;                                                         \
}                                                                               \
static int stub_##SUFFIX##_cblas(int n, TYPE *a, int lda, int *iseed,          \
                                 TYPE *work)                                    \
{                                                                               \
    g_##SUFFIX##_cblas_call.called += 1;                                        \
    g_##SUFFIX##_cblas_call.n = n;                                              \
    g_##SUFFIX##_cblas_call.a = a;                                              \
    g_##SUFFIX##_cblas_call.lda = lda;                                          \
    g_##SUFFIX##_cblas_call.iseed = iseed;                                      \
    g_##SUFFIX##_cblas_call.work = work;                                        \
    iseed[0] = (BASE) + 10;                                                     \
    a[0] = (TYPE)((BASE) + 11);                                                 \
    work[0] = (TYPE)((BASE) + 12);                                              \
    return (BASE) + 13;                                                         \
}                                                                               \
static int check_##SUFFIX##_fortran_to_cblas(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                        \
    int iseed[4] = { 1, 2, 3, 4 };                                              \
    TYPE a[4] = { (TYPE)0, (TYPE)0, (TYPE)0, (TYPE)0 };                        \
    TYPE work[4] = { (TYPE)0, (TYPE)0, (TYPE)0, (TYPE)0 };                     \
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
    if (thunk(2, a, 2, iseed, work) != (BASE) + 4 ||                           \
        g_##SUFFIX##_fortran_call.called != 1 ||                                \
        g_##SUFFIX##_fortran_call.n != 2 ||                                     \
        g_##SUFFIX##_fortran_call.a != a ||                                     \
        g_##SUFFIX##_fortran_call.lda != 2 ||                                   \
        g_##SUFFIX##_fortran_call.iseed != iseed ||                             \
        g_##SUFFIX##_fortran_call.work != work ||                               \
        iseed[0] != (BASE) + 1 ||                                               \
        a[0] != (TYPE)((BASE) + 2) ||                                           \
        work[0] != (TYPE)((BASE) + 3)) {                                        \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward LARGE inputs or return info correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards LARGE inputs and returns info\n"); \
    return 0;                                                                   \
}                                                                               \
static int check_##SUFFIX##_cblas_to_fortran(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                      \
    int n = 2;                                                                  \
    int lda = 2;                                                                \
    int info = -1;                                                              \
    int iseed[4] = { 5, 6, 7, 8 };                                              \
    TYPE a[4] = { (TYPE)0, (TYPE)0, (TYPE)0, (TYPE)0 };                        \
    TYPE work[4] = { (TYPE)0, (TYPE)0, (TYPE)0, (TYPE)0 };                     \
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
    thunk(&n, a, &lda, iseed, work, &info);                                     \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                  \
        g_##SUFFIX##_cblas_call.n != 2 ||                                       \
        g_##SUFFIX##_cblas_call.a != a ||                                       \
        g_##SUFFIX##_cblas_call.lda != 2 ||                                     \
        g_##SUFFIX##_cblas_call.iseed != iseed ||                               \
        g_##SUFFIX##_cblas_call.work != work ||                                 \
        info != (BASE) + 13 ||                                                  \
        iseed[0] != (BASE) + 10 ||                                              \
        a[0] != (TYPE)((BASE) + 11) ||                                          \
        work[0] != (TYPE)((BASE) + 12)) {                                       \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference LARGE inputs or store info correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences LARGE inputs and stores info\n"); \
    return 0;                                                                   \
}

DEFINE_LARGE_TESTS(slarge, float, FB_OP_SLARGE, 100)
DEFINE_LARGE_TESTS(dlarge, double, FB_OP_DLARGE, 300)

int main(void)
{
    int status = 0;

    status |= check_slarge_fortran_to_cblas();
    status |= check_slarge_cblas_to_fortran();
    status |= check_dlarge_fortran_to_cblas();
    status |= check_dlarge_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}