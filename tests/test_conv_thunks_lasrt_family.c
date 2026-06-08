#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

#define DEFINE_LASRT_TESTS(SUFFIX, TYPE, OP_ID, BASE)                          \
typedef int (*fb_##SUFFIX##_cblas_fn)(char id, int n, TYPE *d);                \
typedef void (*fb_##SUFFIX##_fortran_fn)(char *id, int *n, TYPE *d, int *info);\
static struct {                                                                 \
    int called;                                                                 \
    char id;                                                                    \
    int n;                                                                      \
    TYPE *d;                                                                    \
} g_##SUFFIX##_fortran_call;                                                    \
static struct {                                                                 \
    int called;                                                                 \
    char id;                                                                    \
    int n;                                                                      \
    TYPE *d;                                                                    \
} g_##SUFFIX##_cblas_call;                                                      \
static void stub_##SUFFIX##_fortran(char *id, int *n, TYPE *d, int *info)      \
{                                                                               \
    TYPE sorted[4] = { (TYPE)1, (TYPE)2, (TYPE)3, (TYPE)4 };                   \
    g_##SUFFIX##_fortran_call.called += 1;                                      \
    g_##SUFFIX##_fortran_call.id = *id;                                         \
    g_##SUFFIX##_fortran_call.n = *n;                                           \
    g_##SUFFIX##_fortran_call.d = d;                                            \
    memcpy(d, sorted, sizeof(sorted));                                          \
    *info = (BASE) + 7;                                                         \
}                                                                               \
static int stub_##SUFFIX##_cblas(char id, int n, TYPE *d)                      \
{                                                                               \
    TYPE sorted[4] = { (TYPE)9, (TYPE)8, (TYPE)7, (TYPE)6 };                   \
    g_##SUFFIX##_cblas_call.called += 1;                                        \
    g_##SUFFIX##_cblas_call.id = id;                                            \
    g_##SUFFIX##_cblas_call.n = n;                                              \
    g_##SUFFIX##_cblas_call.d = d;                                              \
    memcpy(d, sorted, sizeof(sorted));                                          \
    return (BASE) + 11;                                                         \
}                                                                               \
static int check_##SUFFIX##_fortran_to_cblas(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                        \
    TYPE d[4] = { (TYPE)4, (TYPE)3, (TYPE)2, (TYPE)1 };                        \
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
    if (thunk('I', 4, d) != (BASE) + 7 ||                                       \
        g_##SUFFIX##_fortran_call.called != 1 ||                                \
        g_##SUFFIX##_fortran_call.id != 'I' ||                                  \
        g_##SUFFIX##_fortran_call.n != 4 ||                                     \
        g_##SUFFIX##_fortran_call.d != d ||                                     \
        d[0] != (TYPE)1 || d[1] != (TYPE)2 || d[2] != (TYPE)3 || d[3] != (TYPE)4) { \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward id/n/vector or return info correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards id/n/vector and returns Fortran info\n"); \
    return 0;                                                                   \
}                                                                               \
static int check_##SUFFIX##_cblas_to_fortran(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                      \
    char id = 'D';                                                              \
    int n = 4;                                                                  \
    int info = -999;                                                            \
    TYPE d[4] = { (TYPE)1, (TYPE)2, (TYPE)3, (TYPE)4 };                        \
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
    thunk(&id, &n, d, &info);                                                   \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                  \
        g_##SUFFIX##_cblas_call.id != 'D' ||                                    \
        g_##SUFFIX##_cblas_call.n != 4 ||                                       \
        g_##SUFFIX##_cblas_call.d != d ||                                       \
        info != (BASE) + 11 ||                                                  \
        d[0] != (TYPE)9 || d[1] != (TYPE)8 || d[2] != (TYPE)7 || d[3] != (TYPE)6) { \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not store the C int return into info correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk forwards id/n/vector and stores the C int return in info\n"); \
    return 0;                                                                   \
}

DEFINE_LASRT_TESTS(slasrt, float, FB_OP_SLASRT, 100)
DEFINE_LASRT_TESTS(dlasrt, double, FB_OP_DLASRT, 300)

int main(void)
{
    int status = 0;

    status |= check_slasrt_fortran_to_cblas();
    status |= check_slasrt_cblas_to_fortran();
    status |= check_dlasrt_fortran_to_cblas();
    status |= check_dlasrt_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}