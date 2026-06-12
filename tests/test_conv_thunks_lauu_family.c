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

#define DEFINE_REAL_LAUU_TESTS(SUFFIX, TYPE, OP_ID, BASE)                      \
typedef int (*fb_##SUFFIX##_cblas_fn)(fb_layout_t layout, fb_uplo_t uplo,      \
                                      int n, TYPE *a, int lda);                \
typedef void (*fb_##SUFFIX##_fortran_fn)(char *uplo, int *n, TYPE *a, int *lda,\
                                         int *info);                           \
static struct {                                                                 \
    int called;                                                                 \
    char uplo;                                                                  \
    int n;                                                                      \
    int lda;                                                                    \
    TYPE *a;                                                                    \
} g_##SUFFIX##_fortran_call;                                                    \
static struct {                                                                 \
    int called;                                                                 \
    fb_layout_t layout;                                                         \
    fb_uplo_t uplo;                                                             \
    int n;                                                                      \
    int lda;                                                                    \
    TYPE *a;                                                                    \
} g_##SUFFIX##_cblas_call;                                                      \
static void stub_##SUFFIX##_fortran(char *uplo, int *n, TYPE *a, int *lda,     \
                                    int *info)                                  \
{                                                                               \
    g_##SUFFIX##_fortran_call.called += 1;                                      \
    g_##SUFFIX##_fortran_call.uplo = *uplo;                                     \
    g_##SUFFIX##_fortran_call.n = *n;                                           \
    g_##SUFFIX##_fortran_call.lda = *lda;                                       \
    g_##SUFFIX##_fortran_call.a = a;                                             \
    a[0] = (TYPE)((BASE) + 10);                                                 \
    a[(*lda) + 1] = (TYPE)((BASE) + 11);                                        \
    *info = (BASE) + 50;                                                        \
}                                                                               \
static int stub_##SUFFIX##_cblas(fb_layout_t layout, fb_uplo_t uplo, int n,    \
                                 TYPE *a, int lda)                              \
{                                                                               \
    g_##SUFFIX##_cblas_call.called += 1;                                        \
    g_##SUFFIX##_cblas_call.layout = layout;                                    \
    g_##SUFFIX##_cblas_call.uplo = uplo;                                        \
    g_##SUFFIX##_cblas_call.n = n;                                              \
    g_##SUFFIX##_cblas_call.lda = lda;                                          \
    g_##SUFFIX##_cblas_call.a = a;                                              \
    a[0] = (TYPE)((BASE) + 300);                                                \
    return (BASE) + 500;                                                        \
}                                                                               \
static int check_##SUFFIX##_fortran_to_cblas(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                        \
    TYPE a[4] = { (TYPE)1, (TYPE)2, (TYPE)3, (TYPE)4 };                        \
    int info = 0;                                                               \
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
    info = thunk(FB_LAYOUT_ROW_MAJOR, FB_UPPER, 2, a, 2);                      \
    if (info != ((BASE) + 50) || g_##SUFFIX##_fortran_call.called != 1 ||       \
        g_##SUFFIX##_fortran_call.uplo != 'L' ||                                \
        g_##SUFFIX##_fortran_call.n != 2 ||                                     \
        g_##SUFFIX##_fortran_call.lda != 2 ||                                   \
        g_##SUFFIX##_fortran_call.a != a ||                                     \
        a[0] != (TYPE)((BASE) + 10) || a[3] != (TYPE)((BASE) + 11)) {          \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not flip UPLO for row-major in-place triangular products\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk flips row-major UPLO and propagates Fortran info\n"); \
    return 0;                                                                   \
}                                                                               \
static int check_##SUFFIX##_cblas_to_fortran(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                      \
    char uplo = 'l';                                                            \
    int n = 2;                                                                  \
    int lda = 2;                                                                \
    int info = 0;                                                               \
    TYPE a[4] = { (TYPE)0, (TYPE)0, (TYPE)0, (TYPE)0 };                        \
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
    thunk(&uplo, &n, a, &lda, &info);                                           \
    if (info != ((BASE) + 500) || g_##SUFFIX##_cblas_call.called != 1 ||        \
        g_##SUFFIX##_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||                \
        g_##SUFFIX##_cblas_call.uplo != FB_LOWER ||                             \
        g_##SUFFIX##_cblas_call.n != 2 ||                                       \
        g_##SUFFIX##_cblas_call.lda != 2 ||                                     \
        g_##SUFFIX##_cblas_call.a != a ||                                       \
        a[0] != (TYPE)((BASE) + 300)) {                                         \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not delegate the LAPACKE triangular product entry correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk maps the all-pointer ABI into the LAPACKE triangular product entry\n"); \
    return 0;                                                                   \
}

#define DEFINE_COMPLEX_LAUU_TESTS(SUFFIX, CTYPE, REALTYPE, OP_ID, MAKE_FN, EQ_FN, BASE) \
typedef int (*fb_##SUFFIX##_cblas_fn)(fb_layout_t layout, fb_uplo_t uplo,      \
                                      int n, CTYPE *a, int lda);               \
typedef void (*fb_##SUFFIX##_fortran_fn)(char *uplo, int *n, CTYPE *a, int *lda,\
                                         int *info);                           \
static struct {                                                                 \
    int called;                                                                 \
    char uplo;                                                                  \
    int n;                                                                      \
    int lda;                                                                    \
    CTYPE *a;                                                                   \
} g_##SUFFIX##_fortran_call;                                                    \
static struct {                                                                 \
    int called;                                                                 \
    fb_layout_t layout;                                                         \
    fb_uplo_t uplo;                                                             \
    int n;                                                                      \
    int lda;                                                                    \
    CTYPE *a;                                                                   \
} g_##SUFFIX##_cblas_call;                                                      \
static void stub_##SUFFIX##_fortran(char *uplo, int *n, CTYPE *a, int *lda,    \
                                    int *info)                                  \
{                                                                               \
    g_##SUFFIX##_fortran_call.called += 1;                                      \
    g_##SUFFIX##_fortran_call.uplo = *uplo;                                     \
    g_##SUFFIX##_fortran_call.n = *n;                                           \
    g_##SUFFIX##_fortran_call.lda = *lda;                                       \
    g_##SUFFIX##_fortran_call.a = a;                                             \
    a[0] = MAKE_FN((REALTYPE)((BASE) + 10), (REALTYPE)((BASE) + 60));          \
    a[(*lda) + 1] = MAKE_FN((REALTYPE)((BASE) + 11),                            \
                            (REALTYPE)((BASE) + 61));                           \
    *info = (BASE) + 50;                                                        \
}                                                                               \
static int stub_##SUFFIX##_cblas(fb_layout_t layout, fb_uplo_t uplo, int n,    \
                                 CTYPE *a, int lda)                             \
{                                                                               \
    g_##SUFFIX##_cblas_call.called += 1;                                        \
    g_##SUFFIX##_cblas_call.layout = layout;                                    \
    g_##SUFFIX##_cblas_call.uplo = uplo;                                        \
    g_##SUFFIX##_cblas_call.n = n;                                              \
    g_##SUFFIX##_cblas_call.lda = lda;                                          \
    g_##SUFFIX##_cblas_call.a = a;                                              \
    a[0] = MAKE_FN((REALTYPE)((BASE) + 300), (REALTYPE)((BASE) + 350));        \
    return (BASE) + 500;                                                        \
}                                                                               \
static int check_##SUFFIX##_fortran_to_cblas(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                        \
    CTYPE a[4];                                                                 \
    int info = 0;                                                               \
    a[0] = MAKE_FN((REALTYPE)1, (REALTYPE)11);                                  \
    a[1] = MAKE_FN((REALTYPE)2, (REALTYPE)12);                                  \
    a[2] = MAKE_FN((REALTYPE)3, (REALTYPE)13);                                  \
    a[3] = MAKE_FN((REALTYPE)4, (REALTYPE)14);                                  \
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
    info = thunk(FB_LAYOUT_ROW_MAJOR, FB_UPPER, 2, a, 2);                      \
    if (info != ((BASE) + 50) || g_##SUFFIX##_fortran_call.called != 1 ||       \
        g_##SUFFIX##_fortran_call.uplo != 'L' ||                                \
        g_##SUFFIX##_fortran_call.n != 2 ||                                     \
        g_##SUFFIX##_fortran_call.lda != 2 ||                                   \
        g_##SUFFIX##_fortran_call.a != a ||                                     \
        !EQ_FN(a[0], MAKE_FN((REALTYPE)((BASE) + 10),                           \
                             (REALTYPE)((BASE) + 60))) ||                       \
        !EQ_FN(a[3], MAKE_FN((REALTYPE)((BASE) + 11),                           \
                             (REALTYPE)((BASE) + 61)))) {                       \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not flip UPLO for row-major complex triangular products\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk flips row-major UPLO and propagates complex Fortran info\n"); \
    return 0;                                                                   \
}                                                                               \
static int check_##SUFFIX##_cblas_to_fortran(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                      \
    char uplo = 'l';                                                            \
    int n = 2;                                                                  \
    int lda = 2;                                                                \
    int info = 0;                                                               \
    CTYPE a[4];                                                                 \
    a[0] = MAKE_FN((REALTYPE)0, (REALTYPE)0);                                   \
    a[1] = MAKE_FN((REALTYPE)0, (REALTYPE)0);                                   \
    a[2] = MAKE_FN((REALTYPE)0, (REALTYPE)0);                                   \
    a[3] = MAKE_FN((REALTYPE)0, (REALTYPE)0);                                   \
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
    thunk(&uplo, &n, a, &lda, &info);                                           \
    if (info != ((BASE) + 500) || g_##SUFFIX##_cblas_call.called != 1 ||        \
        g_##SUFFIX##_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||                \
        g_##SUFFIX##_cblas_call.uplo != FB_LOWER ||                             \
        g_##SUFFIX##_cblas_call.n != 2 ||                                       \
        g_##SUFFIX##_cblas_call.lda != 2 ||                                     \
        g_##SUFFIX##_cblas_call.a != a ||                                       \
        !EQ_FN(a[0], MAKE_FN((REALTYPE)((BASE) + 300),                          \
                             (REALTYPE)((BASE) + 350)))) {                      \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not delegate the complex LAPACKE triangular product entry correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk maps the all-pointer ABI into the complex LAPACKE triangular product entry\n"); \
    return 0;                                                                   \
}

DEFINE_REAL_LAUU_TESTS(slauu2, float, FB_OP_SLAUU2, 100)
DEFINE_REAL_LAUU_TESTS(dlauu2, double, FB_OP_DLAUU2, 300)
DEFINE_COMPLEX_LAUU_TESTS(clauu2, fb_complex_float_t, float, FB_OP_CLAUU2,
                          make_cf32, cf32_eq, 500)
DEFINE_COMPLEX_LAUU_TESTS(zlauu2, fb_complex_double_t, double, FB_OP_ZLAUU2,
                          make_cf64, cf64_eq, 700)
DEFINE_REAL_LAUU_TESTS(slauum, float, FB_OP_SLAUUM, 900)
DEFINE_REAL_LAUU_TESTS(dlauum, double, FB_OP_DLAUUM, 1100)
DEFINE_COMPLEX_LAUU_TESTS(clauum, fb_complex_float_t, float, FB_OP_CLAUUM,
                          make_cf32, cf32_eq, 1300)
DEFINE_COMPLEX_LAUU_TESTS(zlauum, fb_complex_double_t, double, FB_OP_ZLAUUM,
                          make_cf64, cf64_eq, 1500)

int main(void)
{
    int status = 0;

    status |= check_slauu2_fortran_to_cblas();
    status |= check_slauu2_cblas_to_fortran();
    status |= check_dlauu2_fortran_to_cblas();
    status |= check_dlauu2_cblas_to_fortran();
    status |= check_clauu2_fortran_to_cblas();
    status |= check_clauu2_cblas_to_fortran();
    status |= check_zlauu2_fortran_to_cblas();
    status |= check_zlauu2_cblas_to_fortran();
    status |= check_slauum_fortran_to_cblas();
    status |= check_slauum_cblas_to_fortran();
    status |= check_dlauum_fortran_to_cblas();
    status |= check_dlauum_cblas_to_fortran();
    status |= check_clauum_fortran_to_cblas();
    status |= check_clauum_cblas_to_fortran();
    status |= check_zlauum_fortran_to_cblas();
    status |= check_zlauum_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}