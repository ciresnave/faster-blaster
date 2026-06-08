#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef void (*fb_cher2k_cblas_fn)(fb_layout_t layout, fb_uplo_t uplo,
                                   fb_transpose_t trans, int n, int k,
                                   fb_complex_float_t alpha,
                                   const fb_complex_float_t *a, int lda,
                                   const fb_complex_float_t *b, int ldb,
                                   float beta, fb_complex_float_t *c, int ldc);
typedef void (*fb_zher2k_cblas_fn)(fb_layout_t layout, fb_uplo_t uplo,
                                   fb_transpose_t trans, int n, int k,
                                   fb_complex_double_t alpha,
                                   const fb_complex_double_t *a, int lda,
                                   const fb_complex_double_t *b, int ldb,
                                   double beta, fb_complex_double_t *c,
                                   int ldc);

typedef void (*fb_cher2k_fortran_slot_fn)(const int *order, const int *uplo,
                                          const int *trans, const int *n,
                                          const int *k,
                                          const fb_complex_float_t *alpha,
                                          const fb_complex_float_t *a,
                                          const int *lda,
                                          const fb_complex_float_t *b,
                                          const int *ldb, const float *beta,
                                          fb_complex_float_t *c,
                                          const int *ldc);
typedef void (*fb_zher2k_fortran_slot_fn)(const int *order, const int *uplo,
                                          const int *trans, const int *n,
                                          const int *k,
                                          const fb_complex_double_t *alpha,
                                          const fb_complex_double_t *a,
                                          const int *lda,
                                          const fb_complex_double_t *b,
                                          const int *ldb, const double *beta,
                                          fb_complex_double_t *c,
                                          const int *ldc);

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

static struct {
    int called;
    int order;
    int uplo;
    int trans;
    int n;
    int k;
    int lda;
    int ldb;
    int ldc;
    fb_complex_float_t alpha_value;
    float beta_value;
    const fb_complex_float_t *a;
    const fb_complex_float_t *b;
    fb_complex_float_t *c;
} g_cher2k_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    fb_uplo_t uplo;
    fb_transpose_t trans;
    int n;
    int k;
    int lda;
    int ldb;
    int ldc;
    fb_complex_float_t alpha_value;
    float beta_value;
    const fb_complex_float_t *a;
    const fb_complex_float_t *b;
    fb_complex_float_t *c;
} g_cher2k_cblas_call;

static struct {
    int called;
    int order;
    int uplo;
    int trans;
    int n;
    int k;
    int lda;
    int ldb;
    int ldc;
    fb_complex_double_t alpha_value;
    double beta_value;
    const fb_complex_double_t *a;
    const fb_complex_double_t *b;
    fb_complex_double_t *c;
} g_zher2k_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    fb_uplo_t uplo;
    fb_transpose_t trans;
    int n;
    int k;
    int lda;
    int ldb;
    int ldc;
    fb_complex_double_t alpha_value;
    double beta_value;
    const fb_complex_double_t *a;
    const fb_complex_double_t *b;
    fb_complex_double_t *c;
} g_zher2k_cblas_call;

static void stub_cher2k_fortran(const int *order, const int *uplo,
                                const int *trans, const int *n, const int *k,
                                const fb_complex_float_t *alpha,
                                const fb_complex_float_t *a, const int *lda,
                                const fb_complex_float_t *b, const int *ldb,
                                const float *beta, fb_complex_float_t *c,
                                const int *ldc)
{
    g_cher2k_fortran_call.called += 1;
    g_cher2k_fortran_call.order = *order;
    g_cher2k_fortran_call.uplo = *uplo;
    g_cher2k_fortran_call.trans = *trans;
    g_cher2k_fortran_call.n = *n;
    g_cher2k_fortran_call.k = *k;
    g_cher2k_fortran_call.lda = *lda;
    g_cher2k_fortran_call.ldb = *ldb;
    g_cher2k_fortran_call.ldc = *ldc;
    g_cher2k_fortran_call.alpha_value = *alpha;
    g_cher2k_fortran_call.beta_value = *beta;
    g_cher2k_fortran_call.a = a;
    g_cher2k_fortran_call.b = b;
    g_cher2k_fortran_call.c = c;

    c[0] = make_cf32(141.0f, -1.0f);
    c[1] = make_cf32(142.0f, -2.0f);
    c[2] = make_cf32(143.0f, -3.0f);
    c[3] = make_cf32(144.0f, -4.0f);
}

static void stub_cher2k_cblas(fb_layout_t layout, fb_uplo_t uplo,
                              fb_transpose_t trans, int n, int k,
                              fb_complex_float_t alpha,
                              const fb_complex_float_t *a, int lda,
                              const fb_complex_float_t *b, int ldb, float beta,
                              fb_complex_float_t *c, int ldc)
{
    g_cher2k_cblas_call.called += 1;
    g_cher2k_cblas_call.layout = layout;
    g_cher2k_cblas_call.uplo = uplo;
    g_cher2k_cblas_call.trans = trans;
    g_cher2k_cblas_call.n = n;
    g_cher2k_cblas_call.k = k;
    g_cher2k_cblas_call.lda = lda;
    g_cher2k_cblas_call.ldb = ldb;
    g_cher2k_cblas_call.ldc = ldc;
    g_cher2k_cblas_call.alpha_value = alpha;
    g_cher2k_cblas_call.beta_value = beta;
    g_cher2k_cblas_call.a = a;
    g_cher2k_cblas_call.b = b;
    g_cher2k_cblas_call.c = c;

    c[0] = make_cf32(151.0f, 1.0f);
    c[1] = make_cf32(152.0f, 2.0f);
    c[2] = make_cf32(153.0f, 3.0f);
    c[3] = make_cf32(154.0f, 4.0f);
}

static void stub_zher2k_fortran(const int *order, const int *uplo,
                                const int *trans, const int *n, const int *k,
                                const fb_complex_double_t *alpha,
                                const fb_complex_double_t *a, const int *lda,
                                const fb_complex_double_t *b, const int *ldb,
                                const double *beta, fb_complex_double_t *c,
                                const int *ldc)
{
    g_zher2k_fortran_call.called += 1;
    g_zher2k_fortran_call.order = *order;
    g_zher2k_fortran_call.uplo = *uplo;
    g_zher2k_fortran_call.trans = *trans;
    g_zher2k_fortran_call.n = *n;
    g_zher2k_fortran_call.k = *k;
    g_zher2k_fortran_call.lda = *lda;
    g_zher2k_fortran_call.ldb = *ldb;
    g_zher2k_fortran_call.ldc = *ldc;
    g_zher2k_fortran_call.alpha_value = *alpha;
    g_zher2k_fortran_call.beta_value = *beta;
    g_zher2k_fortran_call.a = a;
    g_zher2k_fortran_call.b = b;
    g_zher2k_fortran_call.c = c;

    c[0] = make_cf64(161.0, -1.0);
    c[1] = make_cf64(162.0, -2.0);
    c[2] = make_cf64(163.0, -3.0);
    c[3] = make_cf64(164.0, -4.0);
}

static void stub_zher2k_cblas(fb_layout_t layout, fb_uplo_t uplo,
                              fb_transpose_t trans, int n, int k,
                              fb_complex_double_t alpha,
                              const fb_complex_double_t *a, int lda,
                              const fb_complex_double_t *b, int ldb,
                              double beta, fb_complex_double_t *c, int ldc)
{
    g_zher2k_cblas_call.called += 1;
    g_zher2k_cblas_call.layout = layout;
    g_zher2k_cblas_call.uplo = uplo;
    g_zher2k_cblas_call.trans = trans;
    g_zher2k_cblas_call.n = n;
    g_zher2k_cblas_call.k = k;
    g_zher2k_cblas_call.lda = lda;
    g_zher2k_cblas_call.ldb = ldb;
    g_zher2k_cblas_call.ldc = ldc;
    g_zher2k_cblas_call.alpha_value = alpha;
    g_zher2k_cblas_call.beta_value = beta;
    g_zher2k_cblas_call.a = a;
    g_zher2k_cblas_call.b = b;
    g_zher2k_cblas_call.c = c;

    c[0] = make_cf64(171.0, 1.0);
    c[1] = make_cf64(172.0, 2.0);
    c[2] = make_cf64(173.0, 3.0);
    c[3] = make_cf64(174.0, 4.0);
}

static int check_cher2k_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_cher2k_cblas_fn thunk = NULL;
    fb_complex_float_t alpha = make_cf32(1.5f, -0.5f);
    fb_complex_float_t a[4] = {
        make_cf32(1.0f, -1.0f), make_cf32(2.0f, -2.0f),
        make_cf32(3.0f, -3.0f), make_cf32(4.0f, -4.0f)
    };
    fb_complex_float_t b[4] = {
        make_cf32(5.0f, 1.0f), make_cf32(6.0f, 2.0f),
        make_cf32(7.0f, 3.0f), make_cf32(8.0f, 4.0f)
    };
    fb_complex_float_t c[4] = {
        make_cf32(0.0f, 0.0f), make_cf32(0.0f, 0.0f),
        make_cf32(0.0f, 0.0f), make_cf32(0.0f, 0.0f)
    };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cher2k_fortran_call, 0, sizeof(g_cher2k_fortran_call));

    vtable.ext_ops[FB_OP_CHER2K][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_cher2k_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CHER2K);

    thunk = (fb_cher2k_cblas_fn)vtable.ext_ops[FB_OP_CHER2K][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CHER2K Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    thunk(FB_LAYOUT_ROW_MAJOR, FB_LOWER, FB_CONJ_TRANS, 2, 3, alpha, a, 3, b, 3, 0.75f, c, 2);
    if (g_cher2k_fortran_call.called != 1 ||
        g_cher2k_fortran_call.order != FB_LAYOUT_ROW_MAJOR ||
        g_cher2k_fortran_call.uplo != FB_LOWER ||
        g_cher2k_fortran_call.trans != FB_CONJ_TRANS ||
        g_cher2k_fortran_call.n != 2 ||
        g_cher2k_fortran_call.k != 3 ||
        g_cher2k_fortran_call.lda != 3 ||
        g_cher2k_fortran_call.ldb != 3 ||
        g_cher2k_fortran_call.ldc != 2 ||
        !cf32_eq(g_cher2k_fortran_call.alpha_value, alpha) ||
        g_cher2k_fortran_call.beta_value != 0.75f ||
        g_cher2k_fortran_call.a != a ||
        g_cher2k_fortran_call.b != b ||
        g_cher2k_fortran_call.c != c ||
        !cf32_eq(c[0], make_cf32(141.0f, -1.0f)) ||
        !cf32_eq(c[1], make_cf32(142.0f, -2.0f)) ||
        !cf32_eq(c[2], make_cf32(143.0f, -3.0f)) ||
        !cf32_eq(c[3], make_cf32(144.0f, -4.0f))) {
        fprintf(stderr, "[FAIL] CHER2K Fortran->CBLAS thunk did not forward Hermitian rank-2k arguments correctly\n");
        return 1;
    }

    printf("[PASS] CHER2K Fortran->CBLAS thunk forwards Hermitian rank-2k arguments unchanged\n");
    return 0;
}

static int check_cher2k_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_cher2k_fortran_slot_fn thunk = NULL;
    int order = FB_LAYOUT_COL_MAJOR;
    int uplo = FB_UPPER;
    int trans = FB_NO_TRANS;
    int n = 2;
    int k = 3;
    int lda = 2;
    int ldb = 2;
    int ldc = 2;
    fb_complex_float_t alpha = make_cf32(2.5f, -1.5f);
    float beta = 1.25f;
    fb_complex_float_t a[4] = {
        make_cf32(9.0f, 1.0f), make_cf32(10.0f, 2.0f),
        make_cf32(11.0f, 3.0f), make_cf32(12.0f, 4.0f)
    };
    fb_complex_float_t b[4] = {
        make_cf32(13.0f, -1.0f), make_cf32(14.0f, -2.0f),
        make_cf32(15.0f, -3.0f), make_cf32(16.0f, -4.0f)
    };
    fb_complex_float_t c[4] = {
        make_cf32(0.0f, 0.0f), make_cf32(0.0f, 0.0f),
        make_cf32(0.0f, 0.0f), make_cf32(0.0f, 0.0f)
    };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cher2k_cblas_call, 0, sizeof(g_cher2k_cblas_call));

    vtable.ext_ops[FB_OP_CHER2K][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_cher2k_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CHER2K);

    thunk = (fb_cher2k_fortran_slot_fn)vtable.ext_ops[FB_OP_CHER2K][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CHER2K CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&order, &uplo, &trans, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
    if (g_cher2k_cblas_call.called != 1 ||
        g_cher2k_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_cher2k_cblas_call.uplo != FB_UPPER ||
        g_cher2k_cblas_call.trans != FB_NO_TRANS ||
        g_cher2k_cblas_call.n != 2 ||
        g_cher2k_cblas_call.k != 3 ||
        g_cher2k_cblas_call.lda != 2 ||
        g_cher2k_cblas_call.ldb != 2 ||
        g_cher2k_cblas_call.ldc != 2 ||
        !cf32_eq(g_cher2k_cblas_call.alpha_value, alpha) ||
        g_cher2k_cblas_call.beta_value != 1.25f ||
        g_cher2k_cblas_call.a != a ||
        g_cher2k_cblas_call.b != b ||
        g_cher2k_cblas_call.c != c ||
        !cf32_eq(c[0], make_cf32(151.0f, 1.0f)) ||
        !cf32_eq(c[1], make_cf32(152.0f, 2.0f)) ||
        !cf32_eq(c[2], make_cf32(153.0f, 3.0f)) ||
        !cf32_eq(c[3], make_cf32(154.0f, 4.0f))) {
        fprintf(stderr, "[FAIL] CHER2K CBLAS->Fortran thunk did not map Hermitian rank-2k arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] CHER2K CBLAS->Fortran thunk maps Hermitian rank-2k arguments into the C entry\n");
    return 0;
}

static int check_zher2k_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_zher2k_cblas_fn thunk = NULL;
    fb_complex_double_t alpha = make_cf64(3.5, -0.5);
    fb_complex_double_t a[4] = {
        make_cf64(17.0, -1.0), make_cf64(18.0, -2.0),
        make_cf64(19.0, -3.0), make_cf64(20.0, -4.0)
    };
    fb_complex_double_t b[4] = {
        make_cf64(21.0, 1.0), make_cf64(22.0, 2.0),
        make_cf64(23.0, 3.0), make_cf64(24.0, 4.0)
    };
    fb_complex_double_t c[4] = {
        make_cf64(0.0, 0.0), make_cf64(0.0, 0.0),
        make_cf64(0.0, 0.0), make_cf64(0.0, 0.0)
    };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zher2k_fortran_call, 0, sizeof(g_zher2k_fortran_call));

    vtable.ext_ops[FB_OP_ZHER2K][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_zher2k_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_ZHER2K);

    thunk = (fb_zher2k_cblas_fn)vtable.ext_ops[FB_OP_ZHER2K][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZHER2K Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    thunk(FB_LAYOUT_ROW_MAJOR, FB_LOWER, FB_CONJ_TRANS, 2, 3, alpha, a, 3, b, 3, 2.75, c, 2);
    if (g_zher2k_fortran_call.called != 1 ||
        g_zher2k_fortran_call.order != FB_LAYOUT_ROW_MAJOR ||
        g_zher2k_fortran_call.uplo != FB_LOWER ||
        g_zher2k_fortran_call.trans != FB_CONJ_TRANS ||
        g_zher2k_fortran_call.n != 2 ||
        g_zher2k_fortran_call.k != 3 ||
        g_zher2k_fortran_call.lda != 3 ||
        g_zher2k_fortran_call.ldb != 3 ||
        g_zher2k_fortran_call.ldc != 2 ||
        !cf64_eq(g_zher2k_fortran_call.alpha_value, alpha) ||
        g_zher2k_fortran_call.beta_value != 2.75 ||
        g_zher2k_fortran_call.a != a ||
        g_zher2k_fortran_call.b != b ||
        g_zher2k_fortran_call.c != c ||
        !cf64_eq(c[0], make_cf64(161.0, -1.0)) ||
        !cf64_eq(c[1], make_cf64(162.0, -2.0)) ||
        !cf64_eq(c[2], make_cf64(163.0, -3.0)) ||
        !cf64_eq(c[3], make_cf64(164.0, -4.0))) {
        fprintf(stderr, "[FAIL] ZHER2K Fortran->CBLAS thunk did not forward double-complex Hermitian rank-2k arguments correctly\n");
        return 1;
    }

    printf("[PASS] ZHER2K Fortran->CBLAS thunk forwards double-complex Hermitian rank-2k arguments unchanged\n");
    return 0;
}

static int check_zher2k_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_zher2k_fortran_slot_fn thunk = NULL;
    int order = FB_LAYOUT_COL_MAJOR;
    int uplo = FB_UPPER;
    int trans = FB_NO_TRANS;
    int n = 2;
    int k = 3;
    int lda = 2;
    int ldb = 2;
    int ldc = 2;
    fb_complex_double_t alpha = make_cf64(4.5, -1.5);
    double beta = 3.25;
    fb_complex_double_t a[4] = {
        make_cf64(25.0, 1.0), make_cf64(26.0, 2.0),
        make_cf64(27.0, 3.0), make_cf64(28.0, 4.0)
    };
    fb_complex_double_t b[4] = {
        make_cf64(29.0, -1.0), make_cf64(30.0, -2.0),
        make_cf64(31.0, -3.0), make_cf64(32.0, -4.0)
    };
    fb_complex_double_t c[4] = {
        make_cf64(0.0, 0.0), make_cf64(0.0, 0.0),
        make_cf64(0.0, 0.0), make_cf64(0.0, 0.0)
    };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zher2k_cblas_call, 0, sizeof(g_zher2k_cblas_call));

    vtable.ext_ops[FB_OP_ZHER2K][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_zher2k_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_ZHER2K);

    thunk = (fb_zher2k_fortran_slot_fn)vtable.ext_ops[FB_OP_ZHER2K][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZHER2K CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&order, &uplo, &trans, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
    if (g_zher2k_cblas_call.called != 1 ||
        g_zher2k_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_zher2k_cblas_call.uplo != FB_UPPER ||
        g_zher2k_cblas_call.trans != FB_NO_TRANS ||
        g_zher2k_cblas_call.n != 2 ||
        g_zher2k_cblas_call.k != 3 ||
        g_zher2k_cblas_call.lda != 2 ||
        g_zher2k_cblas_call.ldb != 2 ||
        g_zher2k_cblas_call.ldc != 2 ||
        !cf64_eq(g_zher2k_cblas_call.alpha_value, alpha) ||
        g_zher2k_cblas_call.beta_value != 3.25 ||
        g_zher2k_cblas_call.a != a ||
        g_zher2k_cblas_call.b != b ||
        g_zher2k_cblas_call.c != c ||
        !cf64_eq(c[0], make_cf64(171.0, 1.0)) ||
        !cf64_eq(c[1], make_cf64(172.0, 2.0)) ||
        !cf64_eq(c[2], make_cf64(173.0, 3.0)) ||
        !cf64_eq(c[3], make_cf64(174.0, 4.0))) {
        fprintf(stderr, "[FAIL] ZHER2K CBLAS->Fortran thunk did not map double-complex Hermitian rank-2k arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] ZHER2K CBLAS->Fortran thunk maps double-complex Hermitian rank-2k arguments into the C entry\n");
    return 0;
}

int main(void)
{
    int status = 0;

    status |= check_cher2k_fortran_to_cblas();
    status |= check_cher2k_cblas_to_fortran();
    status |= check_zher2k_fortran_to_cblas();
    status |= check_zher2k_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}