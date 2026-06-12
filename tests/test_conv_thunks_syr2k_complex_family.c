#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef void (*fb_csyr2k_cblas_fn)(fb_layout_t layout, fb_uplo_t uplo,
                                   fb_transpose_t trans, int n, int k,
                                   const void *alpha, const void *a, int lda,
                                   const void *b, int ldb,
                                   const void *beta, void *c, int ldc);
typedef void (*fb_zsyr2k_cblas_fn)(fb_layout_t layout, fb_uplo_t uplo,
                                   fb_transpose_t trans, int n, int k,
                                   const void *alpha, const void *a, int lda,
                                   const void *b, int ldb,
                                   const void *beta, void *c, int ldc);

typedef void (*fb_csyr2k_fortran_slot_fn)(const int *order, const int *uplo,
                                          const int *trans, const int *n,
                                          const int *k, const void *alpha,
                                          const void *a, const int *lda,
                                          const void *b, const int *ldb,
                                          const void *beta, void *c,
                                          const int *ldc);
typedef void (*fb_zsyr2k_fortran_slot_fn)(const int *order, const int *uplo,
                                          const int *trans, const int *n,
                                          const int *k, const void *alpha,
                                          const void *a, const int *lda,
                                          const void *b, const int *ldb,
                                          const void *beta, void *c,
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
    const void *alpha;
    const void *beta;
    fb_complex_float_t alpha_value;
    fb_complex_float_t beta_value;
    const void *a;
    const void *b;
    void *c;
} g_csyr2k_fortran_call;

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
    const void *alpha;
    const void *beta;
    fb_complex_float_t alpha_value;
    fb_complex_float_t beta_value;
    const void *a;
    const void *b;
    void *c;
} g_csyr2k_cblas_call;

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
    const void *alpha;
    const void *beta;
    fb_complex_double_t alpha_value;
    fb_complex_double_t beta_value;
    const void *a;
    const void *b;
    void *c;
} g_zsyr2k_fortran_call;

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
    const void *alpha;
    const void *beta;
    fb_complex_double_t alpha_value;
    fb_complex_double_t beta_value;
    const void *a;
    const void *b;
    void *c;
} g_zsyr2k_cblas_call;

static void stub_csyr2k_fortran(const int *order, const int *uplo,
                                const int *trans, const int *n, const int *k,
                                const void *alpha, const void *a,
                                const int *lda, const void *b,
                                const int *ldb, const void *beta,
                                void *c, const int *ldc)
{
    fb_complex_float_t *c_values = (fb_complex_float_t *)c;

    g_csyr2k_fortran_call.called += 1;
    g_csyr2k_fortran_call.order = *order;
    g_csyr2k_fortran_call.uplo = *uplo;
    g_csyr2k_fortran_call.trans = *trans;
    g_csyr2k_fortran_call.n = *n;
    g_csyr2k_fortran_call.k = *k;
    g_csyr2k_fortran_call.lda = *lda;
    g_csyr2k_fortran_call.ldb = *ldb;
    g_csyr2k_fortran_call.ldc = *ldc;
    g_csyr2k_fortran_call.alpha = alpha;
    g_csyr2k_fortran_call.beta = beta;
    g_csyr2k_fortran_call.alpha_value = *(const fb_complex_float_t *)alpha;
    g_csyr2k_fortran_call.beta_value = *(const fb_complex_float_t *)beta;
    g_csyr2k_fortran_call.a = a;
    g_csyr2k_fortran_call.b = b;
    g_csyr2k_fortran_call.c = c;

    c_values[0] = make_cf32(321.0f, -1.0f);
    c_values[1] = make_cf32(322.0f, -2.0f);
    c_values[2] = make_cf32(323.0f, -3.0f);
    c_values[3] = make_cf32(324.0f, -4.0f);
}

static void stub_csyr2k_cblas(fb_layout_t layout, fb_uplo_t uplo,
                              fb_transpose_t trans, int n, int k,
                              const void *alpha, const void *a, int lda,
                              const void *b, int ldb,
                              const void *beta, void *c, int ldc)
{
    fb_complex_float_t *c_values = (fb_complex_float_t *)c;

    g_csyr2k_cblas_call.called += 1;
    g_csyr2k_cblas_call.layout = layout;
    g_csyr2k_cblas_call.uplo = uplo;
    g_csyr2k_cblas_call.trans = trans;
    g_csyr2k_cblas_call.n = n;
    g_csyr2k_cblas_call.k = k;
    g_csyr2k_cblas_call.lda = lda;
    g_csyr2k_cblas_call.ldb = ldb;
    g_csyr2k_cblas_call.ldc = ldc;
    g_csyr2k_cblas_call.alpha = alpha;
    g_csyr2k_cblas_call.beta = beta;
    g_csyr2k_cblas_call.alpha_value = *(const fb_complex_float_t *)alpha;
    g_csyr2k_cblas_call.beta_value = *(const fb_complex_float_t *)beta;
    g_csyr2k_cblas_call.a = a;
    g_csyr2k_cblas_call.b = b;
    g_csyr2k_cblas_call.c = c;

    c_values[0] = make_cf32(331.0f, 1.0f);
    c_values[1] = make_cf32(332.0f, 2.0f);
    c_values[2] = make_cf32(333.0f, 3.0f);
    c_values[3] = make_cf32(334.0f, 4.0f);
}

static void stub_zsyr2k_fortran(const int *order, const int *uplo,
                                const int *trans, const int *n, const int *k,
                                const void *alpha, const void *a,
                                const int *lda, const void *b,
                                const int *ldb, const void *beta,
                                void *c, const int *ldc)
{
    fb_complex_double_t *c_values = (fb_complex_double_t *)c;

    g_zsyr2k_fortran_call.called += 1;
    g_zsyr2k_fortran_call.order = *order;
    g_zsyr2k_fortran_call.uplo = *uplo;
    g_zsyr2k_fortran_call.trans = *trans;
    g_zsyr2k_fortran_call.n = *n;
    g_zsyr2k_fortran_call.k = *k;
    g_zsyr2k_fortran_call.lda = *lda;
    g_zsyr2k_fortran_call.ldb = *ldb;
    g_zsyr2k_fortran_call.ldc = *ldc;
    g_zsyr2k_fortran_call.alpha = alpha;
    g_zsyr2k_fortran_call.beta = beta;
    g_zsyr2k_fortran_call.alpha_value = *(const fb_complex_double_t *)alpha;
    g_zsyr2k_fortran_call.beta_value = *(const fb_complex_double_t *)beta;
    g_zsyr2k_fortran_call.a = a;
    g_zsyr2k_fortran_call.b = b;
    g_zsyr2k_fortran_call.c = c;

    c_values[0] = make_cf64(341.0, -1.0);
    c_values[1] = make_cf64(342.0, -2.0);
    c_values[2] = make_cf64(343.0, -3.0);
    c_values[3] = make_cf64(344.0, -4.0);
}

static void stub_zsyr2k_cblas(fb_layout_t layout, fb_uplo_t uplo,
                              fb_transpose_t trans, int n, int k,
                              const void *alpha, const void *a, int lda,
                              const void *b, int ldb,
                              const void *beta, void *c, int ldc)
{
    fb_complex_double_t *c_values = (fb_complex_double_t *)c;

    g_zsyr2k_cblas_call.called += 1;
    g_zsyr2k_cblas_call.layout = layout;
    g_zsyr2k_cblas_call.uplo = uplo;
    g_zsyr2k_cblas_call.trans = trans;
    g_zsyr2k_cblas_call.n = n;
    g_zsyr2k_cblas_call.k = k;
    g_zsyr2k_cblas_call.lda = lda;
    g_zsyr2k_cblas_call.ldb = ldb;
    g_zsyr2k_cblas_call.ldc = ldc;
    g_zsyr2k_cblas_call.alpha = alpha;
    g_zsyr2k_cblas_call.beta = beta;
    g_zsyr2k_cblas_call.alpha_value = *(const fb_complex_double_t *)alpha;
    g_zsyr2k_cblas_call.beta_value = *(const fb_complex_double_t *)beta;
    g_zsyr2k_cblas_call.a = a;
    g_zsyr2k_cblas_call.b = b;
    g_zsyr2k_cblas_call.c = c;

    c_values[0] = make_cf64(351.0, 1.0);
    c_values[1] = make_cf64(352.0, 2.0);
    c_values[2] = make_cf64(353.0, 3.0);
    c_values[3] = make_cf64(354.0, 4.0);
}

static int check_csyr2k_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_csyr2k_cblas_fn thunk = NULL;
    fb_complex_float_t alpha = make_cf32(1.5f, -0.5f);
    fb_complex_float_t beta = make_cf32(0.75f, 0.25f);
    fb_complex_float_t a[6] = {
        make_cf32(1.0f, -1.0f), make_cf32(2.0f, -2.0f),
        make_cf32(3.0f, -3.0f), make_cf32(4.0f, -4.0f),
        make_cf32(5.0f, -5.0f), make_cf32(6.0f, -6.0f)
    };
    fb_complex_float_t b[6] = {
        make_cf32(7.0f, 1.0f), make_cf32(8.0f, 2.0f),
        make_cf32(9.0f, 3.0f), make_cf32(10.0f, 4.0f),
        make_cf32(11.0f, 5.0f), make_cf32(12.0f, 6.0f)
    };
    fb_complex_float_t c[4] = {
        make_cf32(0.0f, 0.0f), make_cf32(0.0f, 0.0f),
        make_cf32(0.0f, 0.0f), make_cf32(0.0f, 0.0f)
    };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_csyr2k_fortran_call, 0, sizeof(g_csyr2k_fortran_call));

    vtable.ext_ops[FB_OP_CSYR2K][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_csyr2k_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CSYR2K);

    thunk = (fb_csyr2k_cblas_fn)vtable.ext_ops[FB_OP_CSYR2K][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CSYR2K Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    thunk(FB_LAYOUT_ROW_MAJOR, FB_LOWER, FB_CONJ_TRANS, 2, 3, &alpha, a, 3, b, 3, &beta, c, 2);
    if (g_csyr2k_fortran_call.called != 1 ||
        g_csyr2k_fortran_call.order != FB_LAYOUT_ROW_MAJOR ||
        g_csyr2k_fortran_call.uplo != FB_LOWER ||
        g_csyr2k_fortran_call.trans != FB_CONJ_TRANS ||
        g_csyr2k_fortran_call.n != 2 ||
        g_csyr2k_fortran_call.k != 3 ||
        g_csyr2k_fortran_call.lda != 3 ||
        g_csyr2k_fortran_call.ldb != 3 ||
        g_csyr2k_fortran_call.ldc != 2 ||
        g_csyr2k_fortran_call.alpha != &alpha ||
        g_csyr2k_fortran_call.beta != &beta ||
        !cf32_eq(g_csyr2k_fortran_call.alpha_value, alpha) ||
        !cf32_eq(g_csyr2k_fortran_call.beta_value, beta) ||
        g_csyr2k_fortran_call.a != a ||
        g_csyr2k_fortran_call.b != b ||
        g_csyr2k_fortran_call.c != c ||
        !cf32_eq(c[0], make_cf32(321.0f, -1.0f)) ||
        !cf32_eq(c[1], make_cf32(322.0f, -2.0f)) ||
        !cf32_eq(c[2], make_cf32(323.0f, -3.0f)) ||
        !cf32_eq(c[3], make_cf32(324.0f, -4.0f))) {
        fprintf(stderr, "[FAIL] CSYR2K Fortran->CBLAS thunk did not forward complex symmetric rank-2k arguments correctly\n");
        return 1;
    }

    printf("[PASS] CSYR2K Fortran->CBLAS thunk forwards complex symmetric rank-2k arguments unchanged\n");
    return 0;
}

static int check_csyr2k_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_csyr2k_fortran_slot_fn thunk = NULL;
    int order = FB_LAYOUT_COL_MAJOR;
    int uplo = FB_UPPER;
    int trans = FB_NO_TRANS;
    int n = 2;
    int k = 3;
    int lda = 2;
    int ldb = 2;
    int ldc = 2;
    fb_complex_float_t alpha = make_cf32(2.5f, -1.5f);
    fb_complex_float_t beta = make_cf32(1.25f, -0.75f);
    fb_complex_float_t a[6] = {
        make_cf32(13.0f, 1.0f), make_cf32(14.0f, 2.0f),
        make_cf32(15.0f, 3.0f), make_cf32(16.0f, 4.0f),
        make_cf32(17.0f, 5.0f), make_cf32(18.0f, 6.0f)
    };
    fb_complex_float_t b[6] = {
        make_cf32(19.0f, -1.0f), make_cf32(20.0f, -2.0f),
        make_cf32(21.0f, -3.0f), make_cf32(22.0f, -4.0f),
        make_cf32(23.0f, -5.0f), make_cf32(24.0f, -6.0f)
    };
    fb_complex_float_t c[4] = {
        make_cf32(0.0f, 0.0f), make_cf32(0.0f, 0.0f),
        make_cf32(0.0f, 0.0f), make_cf32(0.0f, 0.0f)
    };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_csyr2k_cblas_call, 0, sizeof(g_csyr2k_cblas_call));

    vtable.ext_ops[FB_OP_CSYR2K][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_csyr2k_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CSYR2K);

    thunk = (fb_csyr2k_fortran_slot_fn)vtable.ext_ops[FB_OP_CSYR2K][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CSYR2K CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&order, &uplo, &trans, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
    if (g_csyr2k_cblas_call.called != 1 ||
        g_csyr2k_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_csyr2k_cblas_call.uplo != FB_UPPER ||
        g_csyr2k_cblas_call.trans != FB_NO_TRANS ||
        g_csyr2k_cblas_call.n != 2 ||
        g_csyr2k_cblas_call.k != 3 ||
        g_csyr2k_cblas_call.lda != 2 ||
        g_csyr2k_cblas_call.ldb != 2 ||
        g_csyr2k_cblas_call.ldc != 2 ||
        g_csyr2k_cblas_call.alpha != &alpha ||
        g_csyr2k_cblas_call.beta != &beta ||
        !cf32_eq(g_csyr2k_cblas_call.alpha_value, alpha) ||
        !cf32_eq(g_csyr2k_cblas_call.beta_value, beta) ||
        g_csyr2k_cblas_call.a != a ||
        g_csyr2k_cblas_call.b != b ||
        g_csyr2k_cblas_call.c != c ||
        !cf32_eq(c[0], make_cf32(331.0f, 1.0f)) ||
        !cf32_eq(c[1], make_cf32(332.0f, 2.0f)) ||
        !cf32_eq(c[2], make_cf32(333.0f, 3.0f)) ||
        !cf32_eq(c[3], make_cf32(334.0f, 4.0f))) {
        fprintf(stderr, "[FAIL] CSYR2K CBLAS->Fortran thunk did not map complex symmetric rank-2k arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] CSYR2K CBLAS->Fortran thunk maps complex symmetric rank-2k arguments into the C entry\n");
    return 0;
}

static int check_zsyr2k_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_zsyr2k_cblas_fn thunk = NULL;
    fb_complex_double_t alpha = make_cf64(3.5, -0.5);
    fb_complex_double_t beta = make_cf64(2.75, 0.5);
    fb_complex_double_t a[6] = {
        make_cf64(25.0, -1.0), make_cf64(26.0, -2.0),
        make_cf64(27.0, -3.0), make_cf64(28.0, -4.0),
        make_cf64(29.0, -5.0), make_cf64(30.0, -6.0)
    };
    fb_complex_double_t b[6] = {
        make_cf64(31.0, 1.0), make_cf64(32.0, 2.0),
        make_cf64(33.0, 3.0), make_cf64(34.0, 4.0),
        make_cf64(35.0, 5.0), make_cf64(36.0, 6.0)
    };
    fb_complex_double_t c[4] = {
        make_cf64(0.0, 0.0), make_cf64(0.0, 0.0),
        make_cf64(0.0, 0.0), make_cf64(0.0, 0.0)
    };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zsyr2k_fortran_call, 0, sizeof(g_zsyr2k_fortran_call));

    vtable.ext_ops[FB_OP_ZSYR2K][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_zsyr2k_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_ZSYR2K);

    thunk = (fb_zsyr2k_cblas_fn)vtable.ext_ops[FB_OP_ZSYR2K][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZSYR2K Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    thunk(FB_LAYOUT_ROW_MAJOR, FB_LOWER, FB_CONJ_TRANS, 2, 3, &alpha, a, 3, b, 3, &beta, c, 2);
    if (g_zsyr2k_fortran_call.called != 1 ||
        g_zsyr2k_fortran_call.order != FB_LAYOUT_ROW_MAJOR ||
        g_zsyr2k_fortran_call.uplo != FB_LOWER ||
        g_zsyr2k_fortran_call.trans != FB_CONJ_TRANS ||
        g_zsyr2k_fortran_call.n != 2 ||
        g_zsyr2k_fortran_call.k != 3 ||
        g_zsyr2k_fortran_call.lda != 3 ||
        g_zsyr2k_fortran_call.ldb != 3 ||
        g_zsyr2k_fortran_call.ldc != 2 ||
        g_zsyr2k_fortran_call.alpha != &alpha ||
        g_zsyr2k_fortran_call.beta != &beta ||
        !cf64_eq(g_zsyr2k_fortran_call.alpha_value, alpha) ||
        !cf64_eq(g_zsyr2k_fortran_call.beta_value, beta) ||
        g_zsyr2k_fortran_call.a != a ||
        g_zsyr2k_fortran_call.b != b ||
        g_zsyr2k_fortran_call.c != c ||
        !cf64_eq(c[0], make_cf64(341.0, -1.0)) ||
        !cf64_eq(c[1], make_cf64(342.0, -2.0)) ||
        !cf64_eq(c[2], make_cf64(343.0, -3.0)) ||
        !cf64_eq(c[3], make_cf64(344.0, -4.0))) {
        fprintf(stderr, "[FAIL] ZSYR2K Fortran->CBLAS thunk did not forward double-complex symmetric rank-2k arguments correctly\n");
        return 1;
    }

    printf("[PASS] ZSYR2K Fortran->CBLAS thunk forwards double-complex symmetric rank-2k arguments unchanged\n");
    return 0;
}

static int check_zsyr2k_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_zsyr2k_fortran_slot_fn thunk = NULL;
    int order = FB_LAYOUT_COL_MAJOR;
    int uplo = FB_UPPER;
    int trans = FB_NO_TRANS;
    int n = 2;
    int k = 3;
    int lda = 2;
    int ldb = 2;
    int ldc = 2;
    fb_complex_double_t alpha = make_cf64(4.5, -1.5);
    fb_complex_double_t beta = make_cf64(3.25, -0.75);
    fb_complex_double_t a[6] = {
        make_cf64(37.0, 1.0), make_cf64(38.0, 2.0),
        make_cf64(39.0, 3.0), make_cf64(40.0, 4.0),
        make_cf64(41.0, 5.0), make_cf64(42.0, 6.0)
    };
    fb_complex_double_t b[6] = {
        make_cf64(43.0, -1.0), make_cf64(44.0, -2.0),
        make_cf64(45.0, -3.0), make_cf64(46.0, -4.0),
        make_cf64(47.0, -5.0), make_cf64(48.0, -6.0)
    };
    fb_complex_double_t c[4] = {
        make_cf64(0.0, 0.0), make_cf64(0.0, 0.0),
        make_cf64(0.0, 0.0), make_cf64(0.0, 0.0)
    };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zsyr2k_cblas_call, 0, sizeof(g_zsyr2k_cblas_call));

    vtable.ext_ops[FB_OP_ZSYR2K][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_zsyr2k_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_ZSYR2K);

    thunk = (fb_zsyr2k_fortran_slot_fn)vtable.ext_ops[FB_OP_ZSYR2K][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZSYR2K CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&order, &uplo, &trans, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
    if (g_zsyr2k_cblas_call.called != 1 ||
        g_zsyr2k_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_zsyr2k_cblas_call.uplo != FB_UPPER ||
        g_zsyr2k_cblas_call.trans != FB_NO_TRANS ||
        g_zsyr2k_cblas_call.n != 2 ||
        g_zsyr2k_cblas_call.k != 3 ||
        g_zsyr2k_cblas_call.lda != 2 ||
        g_zsyr2k_cblas_call.ldb != 2 ||
        g_zsyr2k_cblas_call.ldc != 2 ||
        g_zsyr2k_cblas_call.alpha != &alpha ||
        g_zsyr2k_cblas_call.beta != &beta ||
        !cf64_eq(g_zsyr2k_cblas_call.alpha_value, alpha) ||
        !cf64_eq(g_zsyr2k_cblas_call.beta_value, beta) ||
        g_zsyr2k_cblas_call.a != a ||
        g_zsyr2k_cblas_call.b != b ||
        g_zsyr2k_cblas_call.c != c ||
        !cf64_eq(c[0], make_cf64(351.0, 1.0)) ||
        !cf64_eq(c[1], make_cf64(352.0, 2.0)) ||
        !cf64_eq(c[2], make_cf64(353.0, 3.0)) ||
        !cf64_eq(c[3], make_cf64(354.0, 4.0))) {
        fprintf(stderr, "[FAIL] ZSYR2K CBLAS->Fortran thunk did not map double-complex symmetric rank-2k arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] ZSYR2K CBLAS->Fortran thunk maps double-complex symmetric rank-2k arguments into the C entry\n");
    return 0;
}

int main(void)
{
    int status = 0;

    status |= check_csyr2k_fortran_to_cblas();
    status |= check_csyr2k_cblas_to_fortran();
    status |= check_zsyr2k_fortran_to_cblas();
    status |= check_zsyr2k_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}