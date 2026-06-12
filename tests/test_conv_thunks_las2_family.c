#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

#define DEFINE_LAS2_TESTS(SUFFIX, TYPE, OP_ID, BASE)                           \
typedef int (*fb_##SUFFIX##_cblas_fn)(TYPE a, TYPE b, TYPE c, TYPE *rt1,       \
                                      TYPE *rt2);                               \
typedef void (*fb_##SUFFIX##_fortran_fn)(TYPE *a, TYPE *b, TYPE *c, TYPE *rt1, \
                                         TYPE *rt2);                            \
static struct {                                                                 \
    int called;                                                                 \
    TYPE a;                                                                     \
    TYPE b;                                                                     \
    TYPE c;                                                                     \
    TYPE *rt1;                                                                  \
    TYPE *rt2;                                                                  \
} g_##SUFFIX##_fortran_call;                                                    \
static struct {                                                                 \
    int called;                                                                 \
    TYPE a;                                                                     \
    TYPE b;                                                                     \
    TYPE c;                                                                     \
    TYPE *rt1;                                                                  \
    TYPE *rt2;                                                                  \
} g_##SUFFIX##_cblas_call;                                                      \
static void stub_##SUFFIX##_fortran(TYPE *a, TYPE *b, TYPE *c, TYPE *rt1,      \
                                    TYPE *rt2)                                  \
{                                                                               \
    g_##SUFFIX##_fortran_call.called += 1;                                      \
    g_##SUFFIX##_fortran_call.a = *a;                                           \
    g_##SUFFIX##_fortran_call.b = *b;                                           \
    g_##SUFFIX##_fortran_call.c = *c;                                           \
    g_##SUFFIX##_fortran_call.rt1 = rt1;                                        \
    g_##SUFFIX##_fortran_call.rt2 = rt2;                                        \
    *rt1 = (TYPE)((BASE) + 1);                                                  \
    *rt2 = (TYPE)((BASE) + 2);                                                  \
}                                                                               \
static int stub_##SUFFIX##_cblas(TYPE a, TYPE b, TYPE c, TYPE *rt1, TYPE *rt2) \
{                                                                               \
    g_##SUFFIX##_cblas_call.called += 1;                                        \
    g_##SUFFIX##_cblas_call.a = a;                                              \
    g_##SUFFIX##_cblas_call.b = b;                                              \
    g_##SUFFIX##_cblas_call.c = c;                                              \
    g_##SUFFIX##_cblas_call.rt1 = rt1;                                          \
    g_##SUFFIX##_cblas_call.rt2 = rt2;                                          \
    *rt1 = (TYPE)((BASE) + 3);                                                  \
    *rt2 = (TYPE)((BASE) + 4);                                                  \
    return 0;                                                                   \
}                                                                               \
static int check_##SUFFIX##_fortran_to_cblas(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                        \
    TYPE rt1 = (TYPE)0;                                                         \
    TYPE rt2 = (TYPE)0;                                                         \
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
    if (thunk((TYPE)7, (TYPE)8, (TYPE)9, &rt1, &rt2) != 0 ||                    \
        g_##SUFFIX##_fortran_call.called != 1 ||                                \
        g_##SUFFIX##_fortran_call.a != (TYPE)7 ||                               \
        g_##SUFFIX##_fortran_call.b != (TYPE)8 ||                               \
        g_##SUFFIX##_fortran_call.c != (TYPE)9 ||                               \
        g_##SUFFIX##_fortran_call.rt1 != &rt1 ||                                \
        g_##SUFFIX##_fortran_call.rt2 != &rt2 ||                                \
        rt1 != (TYPE)((BASE) + 1) || rt2 != (TYPE)((BASE) + 2)) {               \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward LAS2 inputs correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards LAS2 inputs\n"); \
    return 0;                                                                   \
}                                                                               \
static int check_##SUFFIX##_cblas_to_fortran(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                      \
    TYPE a = (TYPE)7;                                                           \
    TYPE b = (TYPE)8;                                                           \
    TYPE c = (TYPE)9;                                                           \
    TYPE rt1 = (TYPE)0;                                                         \
    TYPE rt2 = (TYPE)0;                                                         \
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
    thunk(&a, &b, &c, &rt1, &rt2);                                              \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                  \
        g_##SUFFIX##_cblas_call.a != (TYPE)7 ||                                 \
        g_##SUFFIX##_cblas_call.b != (TYPE)8 ||                                 \
        g_##SUFFIX##_cblas_call.c != (TYPE)9 ||                                 \
        g_##SUFFIX##_cblas_call.rt1 != &rt1 ||                                  \
        g_##SUFFIX##_cblas_call.rt2 != &rt2 ||                                  \
        rt1 != (TYPE)((BASE) + 3) || rt2 != (TYPE)((BASE) + 4)) {               \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference LAS2 inputs correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences LAS2 inputs\n"); \
    return 0;                                                                   \
}

DEFINE_LAS2_TESTS(slas2, float, FB_OP_SLAS2, 100)
DEFINE_LAS2_TESTS(dlas2, double, FB_OP_DLAS2, 300)

int main(void)
{
    int status = 0;

    status |= check_slas2_fortran_to_cblas();
    status |= check_slas2_cblas_to_fortran();
    status |= check_dlas2_fortran_to_cblas();
    status |= check_dlas2_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}