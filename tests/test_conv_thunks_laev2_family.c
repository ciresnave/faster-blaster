#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

#define DEFINE_LAEV2_TESTS(SUFFIX, TYPE, OP_ID, BASE)                          \
typedef int (*fb_##SUFFIX##_cblas_fn)(TYPE a, TYPE b, TYPE c, TYPE *rt1,       \
                                      TYPE *rt2, TYPE *cs1, TYPE *sn1);        \
typedef void (*fb_##SUFFIX##_fortran_fn)(TYPE *a, TYPE *b, TYPE *c, TYPE *rt1, \
                                         TYPE *rt2, TYPE *cs1, TYPE *sn1);     \
static struct {                                                                 \
    int called;                                                                 \
    TYPE a;                                                                     \
    TYPE b;                                                                     \
    TYPE c;                                                                     \
    TYPE *rt1;                                                                  \
    TYPE *rt2;                                                                  \
    TYPE *cs1;                                                                  \
    TYPE *sn1;                                                                  \
} g_##SUFFIX##_fortran_call;                                                    \
static struct {                                                                 \
    int called;                                                                 \
    TYPE a;                                                                     \
    TYPE b;                                                                     \
    TYPE c;                                                                     \
    TYPE *rt1;                                                                  \
    TYPE *rt2;                                                                  \
    TYPE *cs1;                                                                  \
    TYPE *sn1;                                                                  \
} g_##SUFFIX##_cblas_call;                                                      \
static void stub_##SUFFIX##_fortran(TYPE *a, TYPE *b, TYPE *c, TYPE *rt1,      \
                                    TYPE *rt2, TYPE *cs1, TYPE *sn1)           \
{                                                                               \
    g_##SUFFIX##_fortran_call.called += 1;                                      \
    g_##SUFFIX##_fortran_call.a = *a;                                           \
    g_##SUFFIX##_fortran_call.b = *b;                                           \
    g_##SUFFIX##_fortran_call.c = *c;                                           \
    g_##SUFFIX##_fortran_call.rt1 = rt1;                                        \
    g_##SUFFIX##_fortran_call.rt2 = rt2;                                        \
    g_##SUFFIX##_fortran_call.cs1 = cs1;                                        \
    g_##SUFFIX##_fortran_call.sn1 = sn1;                                        \
    *rt1 = (TYPE)((BASE) + 1);                                                  \
    *rt2 = (TYPE)((BASE) + 2);                                                  \
    *cs1 = (TYPE)((BASE) + 3);                                                  \
    *sn1 = (TYPE)((BASE) + 4);                                                  \
}                                                                               \
static int stub_##SUFFIX##_cblas(TYPE a, TYPE b, TYPE c, TYPE *rt1, TYPE *rt2, \
                                 TYPE *cs1, TYPE *sn1)                         \
{                                                                               \
    g_##SUFFIX##_cblas_call.called += 1;                                        \
    g_##SUFFIX##_cblas_call.a = a;                                              \
    g_##SUFFIX##_cblas_call.b = b;                                              \
    g_##SUFFIX##_cblas_call.c = c;                                              \
    g_##SUFFIX##_cblas_call.rt1 = rt1;                                          \
    g_##SUFFIX##_cblas_call.rt2 = rt2;                                          \
    g_##SUFFIX##_cblas_call.cs1 = cs1;                                          \
    g_##SUFFIX##_cblas_call.sn1 = sn1;                                          \
    *rt1 = (TYPE)((BASE) + 5);                                                  \
    *rt2 = (TYPE)((BASE) + 6);                                                  \
    *cs1 = (TYPE)((BASE) + 7);                                                  \
    *sn1 = (TYPE)((BASE) + 8);                                                  \
    return (BASE) + 99;                                                         \
}                                                                               \
static int check_##SUFFIX##_fortran_to_cblas(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                        \
    TYPE rt1 = (TYPE)0;                                                         \
    TYPE rt2 = (TYPE)0;                                                         \
    TYPE cs1 = (TYPE)0;                                                         \
    TYPE sn1 = (TYPE)0;                                                         \
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
    if (thunk((TYPE)((BASE) + 10), (TYPE)((BASE) + 20), (TYPE)((BASE) + 30),   \
              &rt1, &rt2, &cs1, &sn1) != 0 ||                                   \
        g_##SUFFIX##_fortran_call.called != 1 ||                                \
        g_##SUFFIX##_fortran_call.a != (TYPE)((BASE) + 10) ||                   \
        g_##SUFFIX##_fortran_call.b != (TYPE)((BASE) + 20) ||                   \
        g_##SUFFIX##_fortran_call.c != (TYPE)((BASE) + 30) ||                   \
        g_##SUFFIX##_fortran_call.rt1 != &rt1 ||                                \
        g_##SUFFIX##_fortran_call.rt2 != &rt2 ||                                \
        g_##SUFFIX##_fortran_call.cs1 != &cs1 ||                                \
        g_##SUFFIX##_fortran_call.sn1 != &sn1 ||                                \
        rt1 != (TYPE)((BASE) + 1) || rt2 != (TYPE)((BASE) + 2) ||               \
        cs1 != (TYPE)((BASE) + 3) || sn1 != (TYPE)((BASE) + 4)) {               \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward scalar inputs or output pointers correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards scalar inputs and returns success\n"); \
    return 0;                                                                   \
}                                                                               \
static int check_##SUFFIX##_cblas_to_fortran(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                      \
    TYPE a = (TYPE)((BASE) + 40);                                               \
    TYPE b = (TYPE)((BASE) + 50);                                               \
    TYPE c = (TYPE)((BASE) + 60);                                               \
    TYPE rt1 = (TYPE)0;                                                         \
    TYPE rt2 = (TYPE)0;                                                         \
    TYPE cs1 = (TYPE)0;                                                         \
    TYPE sn1 = (TYPE)0;                                                         \
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
    thunk(&a, &b, &c, &rt1, &rt2, &cs1, &sn1);                                  \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                  \
        g_##SUFFIX##_cblas_call.a != (TYPE)((BASE) + 40) ||                     \
        g_##SUFFIX##_cblas_call.b != (TYPE)((BASE) + 50) ||                     \
        g_##SUFFIX##_cblas_call.c != (TYPE)((BASE) + 60) ||                     \
        g_##SUFFIX##_cblas_call.rt1 != &rt1 ||                                  \
        g_##SUFFIX##_cblas_call.rt2 != &rt2 ||                                  \
        g_##SUFFIX##_cblas_call.cs1 != &cs1 ||                                  \
        g_##SUFFIX##_cblas_call.sn1 != &sn1 ||                                  \
        rt1 != (TYPE)((BASE) + 5) || rt2 != (TYPE)((BASE) + 6) ||               \
        cs1 != (TYPE)((BASE) + 7) || sn1 != (TYPE)((BASE) + 8)) {               \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference scalar inputs or propagate outputs correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences scalar inputs and ignores the C int return\n"); \
    return 0;                                                                   \
}

DEFINE_LAEV2_TESTS(slaev2, float, FB_OP_SLAEV2, 100)
DEFINE_LAEV2_TESTS(dlaev2, double, FB_OP_DLAEV2, 300)

int main(void)
{
    int status = 0;

    status |= check_slaev2_fortran_to_cblas();
    status |= check_slaev2_cblas_to_fortran();
    status |= check_dlaev2_fortran_to_cblas();
    status |= check_dlaev2_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}