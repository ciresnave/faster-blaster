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

#define DEFINE_REAL_LASWP_TESTS(SUFFIX, TYPE, OP_ID, BASE)                     \
typedef int (*fb_##SUFFIX##_cblas_fn)(fb_layout_t layout, int n, TYPE *A,      \
                                      int lda, int k1, int k2,                 \
                                      const int *ipiv, int incx);              \
typedef void (*fb_##SUFFIX##_fortran_fn)(int *n, TYPE *A, int *lda, int *k1,   \
                                         int *k2, int *ipiv, int *incx);       \
static struct {                                                                 \
    int called;                                                                 \
    int n;                                                                      \
    int lda;                                                                    \
    int k1;                                                                     \
    int k2;                                                                     \
    int incx;                                                                   \
    TYPE *A;                                                                    \
    int *ipiv;                                                                  \
    TYPE a_snapshot[6];                                                         \
} g_##SUFFIX##_fortran_call;                                                    \
static struct {                                                                 \
    int called;                                                                 \
    fb_layout_t layout;                                                         \
    int n;                                                                      \
    int lda;                                                                    \
    int k1;                                                                     \
    int k2;                                                                     \
    int incx;                                                                   \
    TYPE *A;                                                                    \
    const int *ipiv;                                                            \
} g_##SUFFIX##_cblas_call;                                                      \
static void stub_##SUFFIX##_fortran(int *n, TYPE *A, int *lda, int *k1,        \
                                    int *k2, int *ipiv, int *incx)            \
{                                                                               \
    TYPE swapped_out[6] = { (TYPE)5, (TYPE)3, (TYPE)1,                         \
                            (TYPE)6, (TYPE)4, (TYPE)2 };                       \
    int index;                                                                  \
    g_##SUFFIX##_fortran_call.called += 1;                                      \
    g_##SUFFIX##_fortran_call.n = *n;                                           \
    g_##SUFFIX##_fortran_call.lda = *lda;                                       \
    g_##SUFFIX##_fortran_call.k1 = *k1;                                         \
    g_##SUFFIX##_fortran_call.k2 = *k2;                                         \
    g_##SUFFIX##_fortran_call.incx = *incx;                                     \
    g_##SUFFIX##_fortran_call.A = A;                                            \
    g_##SUFFIX##_fortran_call.ipiv = ipiv;                                      \
    for (index = 0; index < 6; ++index) {                                       \
        g_##SUFFIX##_fortran_call.a_snapshot[index] = A[index];                 \
    }                                                                           \
    memcpy(A, swapped_out, sizeof(swapped_out));                                \
}                                                                               \
static int stub_##SUFFIX##_cblas(fb_layout_t layout, int n, TYPE *A,           \
                                 int lda, int k1, int k2,                      \
                                 const int *ipiv, int incx)                   \
{                                                                               \
    g_##SUFFIX##_cblas_call.called += 1;                                        \
    g_##SUFFIX##_cblas_call.layout = layout;                                    \
    g_##SUFFIX##_cblas_call.n = n;                                              \
    g_##SUFFIX##_cblas_call.lda = lda;                                          \
    g_##SUFFIX##_cblas_call.k1 = k1;                                            \
    g_##SUFFIX##_cblas_call.k2 = k2;                                            \
    g_##SUFFIX##_cblas_call.incx = incx;                                        \
    g_##SUFFIX##_cblas_call.A = A;                                              \
    g_##SUFFIX##_cblas_call.ipiv = ipiv;                                        \
    A[0] = (TYPE)((BASE) + 10);                                                 \
    A[5] = (TYPE)((BASE) + 11);                                                 \
    return (BASE) + 99;                                                         \
}                                                                               \
static int check_##SUFFIX##_fortran_to_cblas(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                        \
    TYPE A[6] = { (TYPE)1, (TYPE)2, (TYPE)3, (TYPE)4, (TYPE)5, (TYPE)6 };      \
    TYPE expected_in[6] = { (TYPE)1, (TYPE)3, (TYPE)5, (TYPE)2, (TYPE)4,       \
                            (TYPE)6 };                                          \
    TYPE expected_out[6] = { (TYPE)5, (TYPE)6, (TYPE)3, (TYPE)4, (TYPE)1,      \
                             (TYPE)2 };                                         \
    int ipiv[2] = { 3, 2 };                                                     \
    int status = 0;                                                             \
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
    status = thunk(FB_LAYOUT_ROW_MAJOR, 2, A, 2, 1, 2, ipiv, 1);               \
    if (status != 0 || g_##SUFFIX##_fortran_call.called != 1 ||                \
        g_##SUFFIX##_fortran_call.n != 2 ||                                     \
        g_##SUFFIX##_fortran_call.lda != 3 ||                                   \
        g_##SUFFIX##_fortran_call.k1 != 1 ||                                    \
        g_##SUFFIX##_fortran_call.k2 != 2 ||                                    \
        g_##SUFFIX##_fortran_call.incx != 1 ||                                  \
        g_##SUFFIX##_fortran_call.A == A ||                                     \
        g_##SUFFIX##_fortran_call.ipiv != ipiv ||                               \
        memcmp(g_##SUFFIX##_fortran_call.a_snapshot, expected_in,               \
               sizeof(expected_in)) != 0 ||                                     \
        memcmp(A, expected_out, sizeof(expected_out)) != 0) {                   \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not preserve row-major row-swap translation and copy-back\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk transposes row-major matrix input, applies swaps in col-major, and copies results back\n"); \
    return 0;                                                                   \
}                                                                               \
static int check_##SUFFIX##_cblas_to_fortran(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                      \
    int n = 2;                                                                  \
    int lda = 3;                                                                \
    int k1 = 1;                                                                 \
    int k2 = 2;                                                                 \
    int incx = 1;                                                               \
    int ipiv[2] = { 3, 2 };                                                     \
    TYPE A[6] = { (TYPE)1, (TYPE)3, (TYPE)5, (TYPE)2, (TYPE)4, (TYPE)6 };      \
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
    thunk(&n, A, &lda, &k1, &k2, ipiv, &incx);                                  \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                  \
        g_##SUFFIX##_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||                \
        g_##SUFFIX##_cblas_call.n != 2 ||                                       \
        g_##SUFFIX##_cblas_call.lda != 3 ||                                     \
        g_##SUFFIX##_cblas_call.k1 != 1 ||                                      \
        g_##SUFFIX##_cblas_call.k2 != 2 ||                                      \
        g_##SUFFIX##_cblas_call.incx != 1 ||                                    \
        g_##SUFFIX##_cblas_call.A != A ||                                       \
        g_##SUFFIX##_cblas_call.ipiv != ipiv ||                                 \
        A[0] != (TYPE)((BASE) + 10) || A[5] != (TYPE)((BASE) + 11)) {          \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not forward col-major row-swap arguments correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk forwards col-major row-swap arguments and ignores the C int return as expected\n"); \
    return 0;                                                                   \
}

#define DEFINE_COMPLEX_LASWP_TESTS(SUFFIX, CTYPE, OP_ID, MAKE_FN, EQ_FN, BASE) \
typedef int (*fb_##SUFFIX##_cblas_fn)(fb_layout_t layout, int n, CTYPE *A,     \
                                      int lda, int k1, int k2,                 \
                                      const int *ipiv, int incx);              \
typedef void (*fb_##SUFFIX##_fortran_fn)(int *n, CTYPE *A, int *lda, int *k1,  \
                                         int *k2, int *ipiv, int *incx);       \
static struct {                                                                 \
    int called;                                                                 \
    int n;                                                                      \
    int lda;                                                                    \
    int k1;                                                                     \
    int k2;                                                                     \
    int incx;                                                                   \
    CTYPE *A;                                                                   \
    int *ipiv;                                                                  \
    CTYPE a_snapshot[6];                                                        \
} g_##SUFFIX##_fortran_call;                                                    \
static struct {                                                                 \
    int called;                                                                 \
    fb_layout_t layout;                                                         \
    int n;                                                                      \
    int lda;                                                                    \
    int k1;                                                                     \
    int k2;                                                                     \
    int incx;                                                                   \
    CTYPE *A;                                                                   \
    const int *ipiv;                                                            \
} g_##SUFFIX##_cblas_call;                                                      \
static void stub_##SUFFIX##_fortran(int *n, CTYPE *A, int *lda, int *k1,       \
                                    int *k2, int *ipiv, int *incx)            \
{                                                                               \
    CTYPE swapped_out[6];                                                       \
    int index;                                                                  \
    swapped_out[0] = MAKE_FN(5, 105);                                           \
    swapped_out[1] = MAKE_FN(3, 103);                                           \
    swapped_out[2] = MAKE_FN(1, 101);                                           \
    swapped_out[3] = MAKE_FN(6, 106);                                           \
    swapped_out[4] = MAKE_FN(4, 104);                                           \
    swapped_out[5] = MAKE_FN(2, 102);                                           \
    g_##SUFFIX##_fortran_call.called += 1;                                      \
    g_##SUFFIX##_fortran_call.n = *n;                                           \
    g_##SUFFIX##_fortran_call.lda = *lda;                                       \
    g_##SUFFIX##_fortran_call.k1 = *k1;                                         \
    g_##SUFFIX##_fortran_call.k2 = *k2;                                         \
    g_##SUFFIX##_fortran_call.incx = *incx;                                     \
    g_##SUFFIX##_fortran_call.A = A;                                            \
    g_##SUFFIX##_fortran_call.ipiv = ipiv;                                      \
    for (index = 0; index < 6; ++index) {                                       \
        g_##SUFFIX##_fortran_call.a_snapshot[index] = A[index];                 \
    }                                                                           \
    memcpy(A, swapped_out, sizeof(swapped_out));                                \
}                                                                               \
static int stub_##SUFFIX##_cblas(fb_layout_t layout, int n, CTYPE *A,          \
                                 int lda, int k1, int k2,                      \
                                 const int *ipiv, int incx)                   \
{                                                                               \
    g_##SUFFIX##_cblas_call.called += 1;                                        \
    g_##SUFFIX##_cblas_call.layout = layout;                                    \
    g_##SUFFIX##_cblas_call.n = n;                                              \
    g_##SUFFIX##_cblas_call.lda = lda;                                          \
    g_##SUFFIX##_cblas_call.k1 = k1;                                            \
    g_##SUFFIX##_cblas_call.k2 = k2;                                            \
    g_##SUFFIX##_cblas_call.incx = incx;                                        \
    g_##SUFFIX##_cblas_call.A = A;                                              \
    g_##SUFFIX##_cblas_call.ipiv = ipiv;                                        \
    A[0] = MAKE_FN((BASE) + 10, (BASE) + 60);                                   \
    A[5] = MAKE_FN((BASE) + 11, (BASE) + 61);                                   \
    return (BASE) + 99;                                                         \
}                                                                               \
static int check_##SUFFIX##_fortran_to_cblas(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                        \
    CTYPE A[6];                                                                 \
    CTYPE expected_in[6];                                                       \
    CTYPE expected_out[6];                                                      \
    int ipiv[2] = { 3, 2 };                                                     \
    int status = 0;                                                             \
    A[0] = MAKE_FN(1, 101);                                                     \
    A[1] = MAKE_FN(2, 102);                                                     \
    A[2] = MAKE_FN(3, 103);                                                     \
    A[3] = MAKE_FN(4, 104);                                                     \
    A[4] = MAKE_FN(5, 105);                                                     \
    A[5] = MAKE_FN(6, 106);                                                     \
    expected_in[0] = MAKE_FN(1, 101);                                           \
    expected_in[1] = MAKE_FN(3, 103);                                           \
    expected_in[2] = MAKE_FN(5, 105);                                           \
    expected_in[3] = MAKE_FN(2, 102);                                           \
    expected_in[4] = MAKE_FN(4, 104);                                           \
    expected_in[5] = MAKE_FN(6, 106);                                           \
    expected_out[0] = MAKE_FN(5, 105);                                          \
    expected_out[1] = MAKE_FN(6, 106);                                          \
    expected_out[2] = MAKE_FN(3, 103);                                          \
    expected_out[3] = MAKE_FN(4, 104);                                          \
    expected_out[4] = MAKE_FN(1, 101);                                          \
    expected_out[5] = MAKE_FN(2, 102);                                          \
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
    status = thunk(FB_LAYOUT_ROW_MAJOR, 2, A, 2, 1, 2, ipiv, 1);               \
    if (status != 0 || g_##SUFFIX##_fortran_call.called != 1 ||                \
        g_##SUFFIX##_fortran_call.n != 2 ||                                     \
        g_##SUFFIX##_fortran_call.lda != 3 ||                                   \
        g_##SUFFIX##_fortran_call.k1 != 1 ||                                    \
        g_##SUFFIX##_fortran_call.k2 != 2 ||                                    \
        g_##SUFFIX##_fortran_call.incx != 1 ||                                  \
        g_##SUFFIX##_fortran_call.A == A ||                                     \
        g_##SUFFIX##_fortran_call.ipiv != ipiv ||                               \
        !EQ_FN(g_##SUFFIX##_fortran_call.a_snapshot[0], expected_in[0]) ||      \
        !EQ_FN(g_##SUFFIX##_fortran_call.a_snapshot[1], expected_in[1]) ||      \
        !EQ_FN(g_##SUFFIX##_fortran_call.a_snapshot[2], expected_in[2]) ||      \
        !EQ_FN(g_##SUFFIX##_fortran_call.a_snapshot[3], expected_in[3]) ||      \
        !EQ_FN(g_##SUFFIX##_fortran_call.a_snapshot[4], expected_in[4]) ||      \
        !EQ_FN(g_##SUFFIX##_fortran_call.a_snapshot[5], expected_in[5]) ||      \
        !EQ_FN(A[0], expected_out[0]) || !EQ_FN(A[1], expected_out[1]) ||       \
        !EQ_FN(A[2], expected_out[2]) || !EQ_FN(A[3], expected_out[3]) ||       \
        !EQ_FN(A[4], expected_out[4]) || !EQ_FN(A[5], expected_out[5])) {       \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not preserve row-major complex row-swap translation and copy-back\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk transposes row-major complex matrix input, applies swaps in col-major, and copies results back\n"); \
    return 0;                                                                   \
}                                                                               \
static int check_##SUFFIX##_cblas_to_fortran(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                      \
    int n = 2;                                                                  \
    int lda = 3;                                                                \
    int k1 = 1;                                                                 \
    int k2 = 2;                                                                 \
    int incx = 1;                                                               \
    int ipiv[2] = { 3, 2 };                                                     \
    CTYPE A[6];                                                                 \
    A[0] = MAKE_FN(1, 101);                                                     \
    A[1] = MAKE_FN(3, 103);                                                     \
    A[2] = MAKE_FN(5, 105);                                                     \
    A[3] = MAKE_FN(2, 102);                                                     \
    A[4] = MAKE_FN(4, 104);                                                     \
    A[5] = MAKE_FN(6, 106);                                                     \
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
    thunk(&n, A, &lda, &k1, &k2, ipiv, &incx);                                  \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                  \
        g_##SUFFIX##_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||                \
        g_##SUFFIX##_cblas_call.n != 2 ||                                       \
        g_##SUFFIX##_cblas_call.lda != 3 ||                                     \
        g_##SUFFIX##_cblas_call.k1 != 1 ||                                      \
        g_##SUFFIX##_cblas_call.k2 != 2 ||                                      \
        g_##SUFFIX##_cblas_call.incx != 1 ||                                    \
        g_##SUFFIX##_cblas_call.A != A ||                                       \
        g_##SUFFIX##_cblas_call.ipiv != ipiv ||                                 \
        !EQ_FN(A[0], MAKE_FN((BASE) + 10, (BASE) + 60)) ||                      \
        !EQ_FN(A[5], MAKE_FN((BASE) + 11, (BASE) + 61))) {                      \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not forward col-major complex row-swap arguments correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk forwards col-major complex row-swap arguments and ignores the C int return as expected\n"); \
    return 0;                                                                   \
}

DEFINE_REAL_LASWP_TESTS(slaswp, float, FB_OP_SLASWP, 100)
DEFINE_REAL_LASWP_TESTS(dlaswp, double, FB_OP_DLASWP, 300)
DEFINE_COMPLEX_LASWP_TESTS(claswp, fb_complex_float_t, FB_OP_CLASWP,
                           make_cf32, cf32_eq, 500)
DEFINE_COMPLEX_LASWP_TESTS(zlaswp, fb_complex_double_t, FB_OP_ZLASWP,
                           make_cf64, cf64_eq, 700)

int main(void)
{
    int status = 0;

    status |= check_slaswp_fortran_to_cblas();
    status |= check_slaswp_cblas_to_fortran();
    status |= check_dlaswp_fortran_to_cblas();
    status |= check_dlaswp_cblas_to_fortran();
    status |= check_claswp_fortran_to_cblas();
    status |= check_claswp_cblas_to_fortran();
    status |= check_zlaswp_fortran_to_cblas();
    status |= check_zlaswp_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}