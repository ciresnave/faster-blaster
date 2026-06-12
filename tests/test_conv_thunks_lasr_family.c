#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

#define DEFINE_LASR_TESTS(SUFFIX, TYPE, OP_ID, BASE)                               \
typedef int (*fb_##SUFFIX##_cblas_fn)(char side, char pivot, char direct,        \
                                      int m, int n, const TYPE *c,               \
                                      const TYPE *s, TYPE *a, int lda);          \
typedef void (*fb_##SUFFIX##_fortran_fn)(const char *side, const char *pivot,    \
                                         const char *direct, const int *m,       \
                                         const int *n, const TYPE *c,            \
                                         const TYPE *s, TYPE *a,                 \
                                         const int *lda);                        \
static struct {                                                                    \
    int called;                                                                    \
    char side;                                                                     \
    char pivot;                                                                    \
    char direct;                                                                   \
    int m;                                                                         \
    int n;                                                                         \
    const TYPE *c;                                                                 \
    const TYPE *s;                                                                 \
    TYPE *a;                                                                       \
    int lda;                                                                       \
} g_##SUFFIX##_fortran_call;                                                       \
static struct {                                                                    \
    int called;                                                                    \
    char side;                                                                     \
    char pivot;                                                                    \
    char direct;                                                                   \
    int m;                                                                         \
    int n;                                                                         \
    const TYPE *c;                                                                 \
    const TYPE *s;                                                                 \
    TYPE *a;                                                                       \
    int lda;                                                                       \
} g_##SUFFIX##_cblas_call;                                                         \
static void stub_##SUFFIX##_fortran(const char *side, const char *pivot,         \
                                    const char *direct, const int *m,           \
                                    const int *n, const TYPE *c,                \
                                    const TYPE *s, TYPE *a,                     \
                                    const int *lda)                             \
{                                                                                 \
    g_##SUFFIX##_fortran_call.called += 1;                                        \
    g_##SUFFIX##_fortran_call.side = *side;                                       \
    g_##SUFFIX##_fortran_call.pivot = *pivot;                                     \
    g_##SUFFIX##_fortran_call.direct = *direct;                                   \
    g_##SUFFIX##_fortran_call.m = *m;                                             \
    g_##SUFFIX##_fortran_call.n = *n;                                             \
    g_##SUFFIX##_fortran_call.c = c;                                              \
    g_##SUFFIX##_fortran_call.s = s;                                              \
    g_##SUFFIX##_fortran_call.a = a;                                              \
    g_##SUFFIX##_fortran_call.lda = *lda;                                         \
    a[0] = (TYPE)((BASE) + 1);                                                    \
    a[3] = (TYPE)((BASE) + 2);                                                    \
}                                                                                 \
static int stub_##SUFFIX##_cblas(char side, char pivot, char direct, int m,      \
                                 int n, const TYPE *c, const TYPE *s, TYPE *a,   \
                                 int lda)                                        \
{                                                                                 \
    g_##SUFFIX##_cblas_call.called += 1;                                          \
    g_##SUFFIX##_cblas_call.side = side;                                          \
    g_##SUFFIX##_cblas_call.pivot = pivot;                                        \
    g_##SUFFIX##_cblas_call.direct = direct;                                      \
    g_##SUFFIX##_cblas_call.m = m;                                                \
    g_##SUFFIX##_cblas_call.n = n;                                                \
    g_##SUFFIX##_cblas_call.c = c;                                                \
    g_##SUFFIX##_cblas_call.s = s;                                                \
    g_##SUFFIX##_cblas_call.a = a;                                                \
    g_##SUFFIX##_cblas_call.lda = lda;                                            \
    a[0] = (TYPE)((BASE) + 3);                                                    \
    a[3] = (TYPE)((BASE) + 4);                                                    \
    return (BASE) + 5;                                                            \
}                                                                                 \
static int check_##SUFFIX##_fortran_to_cblas(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                          \
    TYPE c[2] = { (TYPE)1, (TYPE)2 };                                             \
    TYPE s[2] = { (TYPE)3, (TYPE)4 };                                             \
    TYPE a[4] = { (TYPE)10, (TYPE)11, (TYPE)12, (TYPE)13 };                      \
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
    if (thunk('L', 'V', 'F', 2, 2, c, s, a, 2) != 0 ||                           \
        g_##SUFFIX##_fortran_call.called != 1 ||                                  \
        g_##SUFFIX##_fortran_call.side != 'L' ||                                  \
        g_##SUFFIX##_fortran_call.pivot != 'V' ||                                 \
        g_##SUFFIX##_fortran_call.direct != 'F' ||                                \
        g_##SUFFIX##_fortran_call.m != 2 ||                                       \
        g_##SUFFIX##_fortran_call.n != 2 ||                                       \
        g_##SUFFIX##_fortran_call.c != c ||                                       \
        g_##SUFFIX##_fortran_call.s != s ||                                       \
        g_##SUFFIX##_fortran_call.a != a ||                                       \
        g_##SUFFIX##_fortran_call.lda != 2 ||                                     \
        a[0] != (TYPE)((BASE) + 1) || a[3] != (TYPE)((BASE) + 2)) {              \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward LASR inputs correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards LASR inputs\n"); \
    return 0;                                                                     \
}                                                                                 \
static int check_##SUFFIX##_cblas_to_fortran(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                        \
    char side = 'R';                                                              \
    char pivot = 'B';                                                             \
    char direct = 'F';                                                            \
    int m = 2;                                                                    \
    int n = 2;                                                                    \
    int lda = 2;                                                                  \
    TYPE c[2] = { (TYPE)5, (TYPE)6 };                                             \
    TYPE s[2] = { (TYPE)7, (TYPE)8 };                                             \
    TYPE a[4] = { (TYPE)20, (TYPE)21, (TYPE)22, (TYPE)23 };                      \
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
    thunk(&side, &pivot, &direct, &m, &n, c, s, a, &lda);                         \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                    \
        g_##SUFFIX##_cblas_call.side != 'R' ||                                    \
        g_##SUFFIX##_cblas_call.pivot != 'B' ||                                   \
        g_##SUFFIX##_cblas_call.direct != 'F' ||                                  \
        g_##SUFFIX##_cblas_call.m != 2 ||                                         \
        g_##SUFFIX##_cblas_call.n != 2 ||                                         \
        g_##SUFFIX##_cblas_call.c != c ||                                         \
        g_##SUFFIX##_cblas_call.s != s ||                                         \
        g_##SUFFIX##_cblas_call.a != a ||                                         \
        g_##SUFFIX##_cblas_call.lda != 2 ||                                       \
        a[0] != (TYPE)((BASE) + 3) || a[3] != (TYPE)((BASE) + 4)) {              \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference LASR inputs correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences LASR inputs\n"); \
    return 0;                                                                     \
}

DEFINE_LASR_TESTS(slasr, float, FB_OP_SLASR, 100)
DEFINE_LASR_TESTS(dlasr, double, FB_OP_DLASR, 300)

int main(void)
{
    int status = 0;

    status |= check_slasr_fortran_to_cblas();
    status |= check_slasr_cblas_to_fortran();
    status |= check_dlasr_fortran_to_cblas();
    status |= check_dlasr_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}