#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

static fb_complex_float_t make_cf32(float real_value, float imag_value)
{
    fb_complex_float_t value;

    __real__ value = real_value;
    __imag__ value = imag_value;
    return value;
}

static fb_complex_double_t make_cf64(double real_value, double imag_value)
{
    fb_complex_double_t value;

    __real__ value = real_value;
    __imag__ value = imag_value;
    return value;
}

static int cf32_eq(fb_complex_float_t lhs, fb_complex_float_t rhs)
{
    return __real__ lhs == __real__ rhs && __imag__ lhs == __imag__ rhs;
}

static int cf64_eq(fb_complex_double_t lhs, fb_complex_double_t rhs)
{
    return __real__ lhs == __real__ rhs && __imag__ lhs == __imag__ rhs;
}

#define DEFINE_LASET_REAL_TESTS(SUFFIX, TYPE, OP_ID, BASE)                       \
typedef int (*fb_##SUFFIX##_cblas_fn)(char uplo, int m, int n, TYPE alpha,       \
                                      TYPE beta, TYPE *a, int lda);              \
typedef void (*fb_##SUFFIX##_fortran_fn)(char *uplo, int *m, int *n, TYPE *alpha,\
                                         TYPE *beta, TYPE *a, int *lda);         \
static struct {                                                                   \
    int called;                                                                   \
    char uplo;                                                                    \
    int m;                                                                        \
    int n;                                                                        \
    TYPE alpha;                                                                   \
    TYPE beta;                                                                    \
    TYPE *a;                                                                      \
    int lda;                                                                      \
} g_##SUFFIX##_fortran_call;                                                      \
static struct {                                                                   \
    int called;                                                                   \
    char uplo;                                                                    \
    int m;                                                                        \
    int n;                                                                        \
    TYPE alpha;                                                                   \
    TYPE beta;                                                                    \
    TYPE *a;                                                                      \
    int lda;                                                                      \
} g_##SUFFIX##_cblas_call;                                                        \
static void stub_##SUFFIX##_fortran(char *uplo, int *m, int *n, TYPE *alpha,     \
                                    TYPE *beta, TYPE *a, int *lda) {             \
    g_##SUFFIX##_fortran_call.called += 1;                                        \
    g_##SUFFIX##_fortran_call.uplo = *uplo;                                       \
    g_##SUFFIX##_fortran_call.m = *m;                                             \
    g_##SUFFIX##_fortran_call.n = *n;                                             \
    g_##SUFFIX##_fortran_call.alpha = *alpha;                                     \
    g_##SUFFIX##_fortran_call.beta = *beta;                                       \
    g_##SUFFIX##_fortran_call.a = a;                                              \
    g_##SUFFIX##_fortran_call.lda = *lda;                                         \
    a[0] = (TYPE)((BASE) + 1);                                                    \
    a[(size_t)(*lda) + 1] = (TYPE)((BASE) + 2);                                   \
}                                                                                 \
static int stub_##SUFFIX##_cblas(char uplo, int m, int n, TYPE alpha, TYPE beta, \
                                 TYPE *a, int lda) {                              \
    g_##SUFFIX##_cblas_call.called += 1;                                          \
    g_##SUFFIX##_cblas_call.uplo = uplo;                                          \
    g_##SUFFIX##_cblas_call.m = m;                                                \
    g_##SUFFIX##_cblas_call.n = n;                                                \
    g_##SUFFIX##_cblas_call.alpha = alpha;                                        \
    g_##SUFFIX##_cblas_call.beta = beta;                                          \
    g_##SUFFIX##_cblas_call.a = a;                                                \
    g_##SUFFIX##_cblas_call.lda = lda;                                            \
    a[0] = (TYPE)((BASE) + 3);                                                    \
    a[(size_t)lda + 1] = (TYPE)((BASE) + 4);                                      \
    return 0;                                                                     \
}                                                                                 \
static int check_##SUFFIX##_fortran_to_cblas(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                          \
    TYPE a[4] = { (TYPE)11, (TYPE)12, (TYPE)13, (TYPE)14 };                       \
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
    if (thunk('A', 2, 2, (TYPE)7, (TYPE)8, a, 2) != 0 ||                          \
        g_##SUFFIX##_fortran_call.called != 1 ||                                  \
        g_##SUFFIX##_fortran_call.uplo != 'A' ||                                  \
        g_##SUFFIX##_fortran_call.m != 2 ||                                       \
        g_##SUFFIX##_fortran_call.n != 2 ||                                       \
        g_##SUFFIX##_fortran_call.alpha != (TYPE)7 ||                             \
        g_##SUFFIX##_fortran_call.beta != (TYPE)8 ||                              \
        g_##SUFFIX##_fortran_call.a != a ||                                       \
        g_##SUFFIX##_fortran_call.lda != 2 ||                                     \
        a[0] != (TYPE)((BASE) + 1) ||                                             \
        a[3] != (TYPE)((BASE) + 2)) {                                             \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward LASET inputs correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards LASET inputs\n"); \
    return 0;                                                                     \
}                                                                                 \
static int check_##SUFFIX##_cblas_to_fortran(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                        \
    char uplo = 'A';                                                              \
    int m = 2;                                                                    \
    int n = 2;                                                                    \
    TYPE alpha = (TYPE)7;                                                         \
    TYPE beta = (TYPE)8;                                                          \
    TYPE a[4] = { (TYPE)21, (TYPE)22, (TYPE)23, (TYPE)24 };                      \
    int lda = 2;                                                                  \
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
    thunk(&uplo, &m, &n, &alpha, &beta, a, &lda);                                \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                    \
        g_##SUFFIX##_cblas_call.uplo != 'A' ||                                    \
        g_##SUFFIX##_cblas_call.m != 2 ||                                         \
        g_##SUFFIX##_cblas_call.n != 2 ||                                         \
        g_##SUFFIX##_cblas_call.alpha != (TYPE)7 ||                               \
        g_##SUFFIX##_cblas_call.beta != (TYPE)8 ||                                \
        g_##SUFFIX##_cblas_call.a != a ||                                         \
        g_##SUFFIX##_cblas_call.lda != 2 ||                                       \
        a[0] != (TYPE)((BASE) + 3) ||                                             \
        a[3] != (TYPE)((BASE) + 4)) {                                             \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference LASET inputs correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences LASET inputs\n"); \
    return 0;                                                                     \
}

#define DEFINE_LASET_COMPLEX_TESTS(SUFFIX, TYPE, OP_ID, MAKE_FN, EQ_FN, BASE)    \
typedef int (*fb_##SUFFIX##_cblas_fn)(char uplo, int m, int n, TYPE alpha,       \
                                      TYPE beta, TYPE *a, int lda);              \
typedef void (*fb_##SUFFIX##_fortran_fn)(char *uplo, int *m, int *n, TYPE *alpha,\
                                         TYPE *beta, TYPE *a, int *lda);         \
static struct {                                                                   \
    int called;                                                                   \
    char uplo;                                                                    \
    int m;                                                                        \
    int n;                                                                        \
    TYPE alpha;                                                                   \
    TYPE beta;                                                                    \
    TYPE *a;                                                                      \
    int lda;                                                                      \
} g_##SUFFIX##_fortran_call;                                                      \
static struct {                                                                   \
    int called;                                                                   \
    char uplo;                                                                    \
    int m;                                                                        \
    int n;                                                                        \
    TYPE alpha;                                                                   \
    TYPE beta;                                                                    \
    TYPE *a;                                                                      \
    int lda;                                                                      \
} g_##SUFFIX##_cblas_call;                                                        \
static void stub_##SUFFIX##_fortran(char *uplo, int *m, int *n, TYPE *alpha,     \
                                    TYPE *beta, TYPE *a, int *lda) {             \
    g_##SUFFIX##_fortran_call.called += 1;                                        \
    g_##SUFFIX##_fortran_call.uplo = *uplo;                                       \
    g_##SUFFIX##_fortran_call.m = *m;                                             \
    g_##SUFFIX##_fortran_call.n = *n;                                             \
    g_##SUFFIX##_fortran_call.alpha = *alpha;                                     \
    g_##SUFFIX##_fortran_call.beta = *beta;                                       \
    g_##SUFFIX##_fortran_call.a = a;                                              \
    g_##SUFFIX##_fortran_call.lda = *lda;                                         \
    a[0] = MAKE_FN((BASE) + 1, (BASE) + 11);                                      \
    a[(size_t)(*lda) + 1] = MAKE_FN((BASE) + 2, (BASE) + 12);                     \
}                                                                                 \
static int stub_##SUFFIX##_cblas(char uplo, int m, int n, TYPE alpha, TYPE beta, \
                                 TYPE *a, int lda) {                              \
    g_##SUFFIX##_cblas_call.called += 1;                                          \
    g_##SUFFIX##_cblas_call.uplo = uplo;                                          \
    g_##SUFFIX##_cblas_call.m = m;                                                \
    g_##SUFFIX##_cblas_call.n = n;                                                \
    g_##SUFFIX##_cblas_call.alpha = alpha;                                        \
    g_##SUFFIX##_cblas_call.beta = beta;                                          \
    g_##SUFFIX##_cblas_call.a = a;                                                \
    g_##SUFFIX##_cblas_call.lda = lda;                                            \
    a[0] = MAKE_FN((BASE) + 3, (BASE) + 13);                                      \
    a[(size_t)lda + 1] = MAKE_FN((BASE) + 4, (BASE) + 14);                        \
    return 0;                                                                     \
}                                                                                 \
static int check_##SUFFIX##_fortran_to_cblas(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                          \
    TYPE a[4] = { MAKE_FN(1, 2), MAKE_FN(3, 4), MAKE_FN(5, 6), MAKE_FN(7, 8) };  \
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
    if (thunk('A', 2, 2, MAKE_FN((BASE) + 7, (BASE) + 17),                       \
              MAKE_FN((BASE) + 8, (BASE) + 18), a, 2) != 0 ||                    \
        g_##SUFFIX##_fortran_call.called != 1 ||                                  \
        g_##SUFFIX##_fortran_call.uplo != 'A' ||                                  \
        g_##SUFFIX##_fortran_call.m != 2 ||                                       \
        g_##SUFFIX##_fortran_call.n != 2 ||                                       \
        !EQ_FN(g_##SUFFIX##_fortran_call.alpha, MAKE_FN((BASE) + 7, (BASE) + 17)) || \
        !EQ_FN(g_##SUFFIX##_fortran_call.beta, MAKE_FN((BASE) + 8, (BASE) + 18)) ||  \
        g_##SUFFIX##_fortran_call.a != a ||                                       \
        g_##SUFFIX##_fortran_call.lda != 2 ||                                     \
        !EQ_FN(a[0], MAKE_FN((BASE) + 1, (BASE) + 11)) ||                        \
        !EQ_FN(a[3], MAKE_FN((BASE) + 2, (BASE) + 12))) {                        \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward LASET inputs correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards LASET inputs\n"); \
    return 0;                                                                     \
}                                                                                 \
static int check_##SUFFIX##_cblas_to_fortran(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                        \
    char uplo = 'A';                                                              \
    int m = 2;                                                                    \
    int n = 2;                                                                    \
    TYPE alpha = MAKE_FN((BASE) + 7, (BASE) + 17);                               \
    TYPE beta = MAKE_FN((BASE) + 8, (BASE) + 18);                                \
    TYPE a[4] = { MAKE_FN(9, 10), MAKE_FN(11, 12), MAKE_FN(13, 14), MAKE_FN(15, 16) }; \
    int lda = 2;                                                                  \
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
    thunk(&uplo, &m, &n, &alpha, &beta, a, &lda);                                \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                    \
        g_##SUFFIX##_cblas_call.uplo != 'A' ||                                    \
        g_##SUFFIX##_cblas_call.m != 2 ||                                         \
        g_##SUFFIX##_cblas_call.n != 2 ||                                         \
        !EQ_FN(g_##SUFFIX##_cblas_call.alpha, MAKE_FN((BASE) + 7, (BASE) + 17)) || \
        !EQ_FN(g_##SUFFIX##_cblas_call.beta, MAKE_FN((BASE) + 8, (BASE) + 18)) ||  \
        g_##SUFFIX##_cblas_call.a != a ||                                         \
        g_##SUFFIX##_cblas_call.lda != 2 ||                                       \
        !EQ_FN(a[0], MAKE_FN((BASE) + 3, (BASE) + 13)) ||                        \
        !EQ_FN(a[3], MAKE_FN((BASE) + 4, (BASE) + 14))) {                        \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference LASET inputs correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences LASET inputs\n"); \
    return 0;                                                                     \
}

DEFINE_LASET_REAL_TESTS(slaset, float, FB_OP_SLASET, 100)
DEFINE_LASET_REAL_TESTS(dlaset, double, FB_OP_DLASET, 300)
DEFINE_LASET_COMPLEX_TESTS(claset, fb_complex_float_t, FB_OP_CLASET, make_cf32, cf32_eq, 500)
DEFINE_LASET_COMPLEX_TESTS(zlaset, fb_complex_double_t, FB_OP_ZLASET, make_cf64, cf64_eq, 700)

int main(void)
{
    int status = 0;

    status |= check_slaset_fortran_to_cblas();
    status |= check_slaset_cblas_to_fortran();
    status |= check_dlaset_fortran_to_cblas();
    status |= check_dlaset_cblas_to_fortran();
    status |= check_claset_fortran_to_cblas();
    status |= check_claset_cblas_to_fortran();
    status |= check_zlaset_fortran_to_cblas();
    status |= check_zlaset_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}