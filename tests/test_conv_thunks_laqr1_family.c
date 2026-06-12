#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

#define DEFINE_LAQR1_TESTS(SUFFIX, TYPE, OP_ID, BASE)                          \
typedef int (*fb_##SUFFIX##_cblas_fn)(int n, TYPE *h, int ldh, TYPE sr1,       \
                                      TYPE si1, TYPE sr2, TYPE si2, TYPE *v);  \
typedef void (*fb_##SUFFIX##_fortran_fn)(int *n, TYPE *h, int *ldh, TYPE *sr1, \
                                         TYPE *si1, TYPE *sr2, TYPE *si2,      \
                                         TYPE *v);                            \
static struct {                                                                 \
    int called;                                                                 \
    int n;                                                                      \
    TYPE *h;                                                                    \
    int ldh;                                                                    \
    TYPE sr1;                                                                   \
    TYPE si1;                                                                   \
    TYPE sr2;                                                                   \
    TYPE si2;                                                                   \
    TYPE *v;                                                                    \
} g_##SUFFIX##_fortran_call;                                                    \
static struct {                                                                 \
    int called;                                                                 \
    int n;                                                                      \
    TYPE *h;                                                                    \
    int ldh;                                                                    \
    TYPE sr1;                                                                   \
    TYPE si1;                                                                   \
    TYPE sr2;                                                                   \
    TYPE si2;                                                                   \
    TYPE *v;                                                                    \
} g_##SUFFIX##_cblas_call;                                                      \
static void stub_##SUFFIX##_fortran(int *n, TYPE *h, int *ldh, TYPE *sr1,      \
                                    TYPE *si1, TYPE *sr2, TYPE *si2, TYPE *v) \
{                                                                               \
    g_##SUFFIX##_fortran_call.called += 1;                                      \
    g_##SUFFIX##_fortran_call.n = *n;                                           \
    g_##SUFFIX##_fortran_call.h = h;                                            \
    g_##SUFFIX##_fortran_call.ldh = *ldh;                                       \
    g_##SUFFIX##_fortran_call.sr1 = *sr1;                                       \
    g_##SUFFIX##_fortran_call.si1 = *si1;                                       \
    g_##SUFFIX##_fortran_call.sr2 = *sr2;                                       \
    g_##SUFFIX##_fortran_call.si2 = *si2;                                       \
    g_##SUFFIX##_fortran_call.v = v;                                            \
    v[0] = (TYPE)((BASE) + 1);                                                  \
    v[1] = (TYPE)((BASE) + 2);                                                  \
    v[2] = (TYPE)((BASE) + 3);                                                  \
}                                                                               \
static int stub_##SUFFIX##_cblas(int n, TYPE *h, int ldh, TYPE sr1, TYPE si1,  \
                                 TYPE sr2, TYPE si2, TYPE *v)                  \
{                                                                               \
    g_##SUFFIX##_cblas_call.called += 1;                                        \
    g_##SUFFIX##_cblas_call.n = n;                                              \
    g_##SUFFIX##_cblas_call.h = h;                                              \
    g_##SUFFIX##_cblas_call.ldh = ldh;                                          \
    g_##SUFFIX##_cblas_call.sr1 = sr1;                                          \
    g_##SUFFIX##_cblas_call.si1 = si1;                                          \
    g_##SUFFIX##_cblas_call.sr2 = sr2;                                          \
    g_##SUFFIX##_cblas_call.si2 = si2;                                          \
    g_##SUFFIX##_cblas_call.v = v;                                              \
    v[0] = (TYPE)((BASE) + 5);                                                  \
    v[1] = (TYPE)((BASE) + 6);                                                  \
    v[2] = (TYPE)((BASE) + 7);                                                  \
    return (BASE) + 99;                                                         \
}                                                                               \
static int check_##SUFFIX##_fortran_to_cblas(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                        \
    TYPE h[9] = {                                                               \
        (TYPE)1, (TYPE)2, (TYPE)3,                                              \
        (TYPE)4, (TYPE)5, (TYPE)6,                                              \
        (TYPE)7, (TYPE)8, (TYPE)9                                               \
    };                                                                          \
    TYPE v[3] = { (TYPE)0, (TYPE)0, (TYPE)0 };                                 \
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
    if (thunk(3, h, 3, (TYPE)((BASE) + 10), (TYPE)((BASE) + 20),               \
              (TYPE)((BASE) + 30), (TYPE)((BASE) + 40), v) != 0 ||             \
        g_##SUFFIX##_fortran_call.called != 1 ||                                \
        g_##SUFFIX##_fortran_call.n != 3 ||                                     \
        g_##SUFFIX##_fortran_call.h != h ||                                     \
        g_##SUFFIX##_fortran_call.ldh != 3 ||                                   \
        g_##SUFFIX##_fortran_call.sr1 != (TYPE)((BASE) + 10) ||                 \
        g_##SUFFIX##_fortran_call.si1 != (TYPE)((BASE) + 20) ||                 \
        g_##SUFFIX##_fortran_call.sr2 != (TYPE)((BASE) + 30) ||                 \
        g_##SUFFIX##_fortran_call.si2 != (TYPE)((BASE) + 40) ||                 \
        g_##SUFFIX##_fortran_call.v != v ||                                     \
        v[0] != (TYPE)((BASE) + 1) ||                                           \
        v[1] != (TYPE)((BASE) + 2) ||                                           \
        v[2] != (TYPE)((BASE) + 3)) {                                           \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward LAQR1 scalar shifts or output vector correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards LAQR1 inputs and returns success\n"); \
    return 0;                                                                   \
}                                                                               \
static int check_##SUFFIX##_cblas_to_fortran(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                      \
    int n = 3;                                                                  \
    int ldh = 3;                                                                \
    TYPE sr1 = (TYPE)((BASE) + 50);                                             \
    TYPE si1 = (TYPE)((BASE) + 60);                                             \
    TYPE sr2 = (TYPE)((BASE) + 70);                                             \
    TYPE si2 = (TYPE)((BASE) + 80);                                             \
    TYPE h[9] = {                                                               \
        (TYPE)11, (TYPE)12, (TYPE)13,                                           \
        (TYPE)14, (TYPE)15, (TYPE)16,                                           \
        (TYPE)17, (TYPE)18, (TYPE)19                                            \
    };                                                                          \
    TYPE v[3] = { (TYPE)0, (TYPE)0, (TYPE)0 };                                 \
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
    thunk(&n, h, &ldh, &sr1, &si1, &sr2, &si2, v);                              \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                  \
        g_##SUFFIX##_cblas_call.n != 3 ||                                       \
        g_##SUFFIX##_cblas_call.h != h ||                                       \
        g_##SUFFIX##_cblas_call.ldh != 3 ||                                     \
        g_##SUFFIX##_cblas_call.sr1 != (TYPE)((BASE) + 50) ||                   \
        g_##SUFFIX##_cblas_call.si1 != (TYPE)((BASE) + 60) ||                   \
        g_##SUFFIX##_cblas_call.sr2 != (TYPE)((BASE) + 70) ||                   \
        g_##SUFFIX##_cblas_call.si2 != (TYPE)((BASE) + 80) ||                   \
        g_##SUFFIX##_cblas_call.v != v ||                                       \
        v[0] != (TYPE)((BASE) + 5) ||                                           \
        v[1] != (TYPE)((BASE) + 6) ||                                           \
        v[2] != (TYPE)((BASE) + 7)) {                                           \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference LAQR1 scalar shifts or propagate outputs correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences LAQR1 scalar shifts and ignores the C int return\n"); \
    return 0;                                                                   \
}

DEFINE_LAQR1_TESTS(slaqr1, float, FB_OP_SLAQR1, 100)
DEFINE_LAQR1_TESTS(dlaqr1, double, FB_OP_DLAQR1, 300)

int main(void)
{
    int status = 0;

    status |= check_slaqr1_fortran_to_cblas();
    status |= check_slaqr1_cblas_to_fortran();
    status |= check_dlaqr1_fortran_to_cblas();
    status |= check_dlaqr1_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}