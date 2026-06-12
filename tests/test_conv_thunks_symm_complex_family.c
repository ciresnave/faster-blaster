#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef void (*fb_csymm_cblas_fn)(fb_layout_t layout, fb_side_t side,
                                  fb_uplo_t uplo, int m, int n,
                                  const void *alpha, const void *a, int lda,
                                  const void *b, int ldb,
                                  const void *beta, void *c, int ldc);
typedef void (*fb_zsymm_cblas_fn)(fb_layout_t layout, fb_side_t side,
                                  fb_uplo_t uplo, int m, int n,
                                  const void *alpha, const void *a, int lda,
                                  const void *b, int ldb,
                                  const void *beta, void *c, int ldc);

typedef void (*fb_csymm_fortran_slot_fn)(const int *order, const int *side,
                                         const int *uplo, const int *m,
                                         const int *n,
                                         const void *alpha,
                                         const void *a,
                                         const int *lda,
                                         const void *b,
                                         const int *ldb,
                                         const void *beta,
                                         void *c,
                                         const int *ldc);
typedef void (*fb_zsymm_fortran_slot_fn)(const int *order, const int *side,
                                         const int *uplo, const int *m,
                                         const int *n,
                                         const void *alpha,
                                         const void *a,
                                         const int *lda,
                                         const void *b,
                                         const int *ldb,
                                         const void *beta,
                                         void *c,
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
    int side;
    int uplo;
    int m;
    int n;
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
} g_csymm_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    fb_side_t side;
    fb_uplo_t uplo;
    int m;
    int n;
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
} g_csymm_cblas_call;

static struct {
    int called;
    int order;
    int side;
    int uplo;
    int m;
    int n;
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
} g_zsymm_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    fb_side_t side;
    fb_uplo_t uplo;
    int m;
    int n;
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
} g_zsymm_cblas_call;

static void stub_csymm_fortran(const int *order, const int *side,
                               const int *uplo, const int *m, const int *n,
                               const void *alpha, const void *a,
                               const int *lda, const void *b,
                               const int *ldb, const void *beta,
                               void *c, const int *ldc)
{
    fb_complex_float_t *c_values = (fb_complex_float_t *)c;

    g_csymm_fortran_call.called += 1;
    g_csymm_fortran_call.order = *order;
    g_csymm_fortran_call.side = *side;
    g_csymm_fortran_call.uplo = *uplo;
    g_csymm_fortran_call.m = *m;
    g_csymm_fortran_call.n = *n;
    g_csymm_fortran_call.lda = *lda;
    g_csymm_fortran_call.ldb = *ldb;
    g_csymm_fortran_call.ldc = *ldc;
    g_csymm_fortran_call.alpha = alpha;
    g_csymm_fortran_call.beta = beta;
    g_csymm_fortran_call.alpha_value = *(const fb_complex_float_t *)alpha;
    g_csymm_fortran_call.beta_value = *(const fb_complex_float_t *)beta;
    g_csymm_fortran_call.a = a;
    g_csymm_fortran_call.b = b;
    g_csymm_fortran_call.c = c;

    c_values[0] = make_cf32(201.0f, -1.0f);
    c_values[1] = make_cf32(202.0f, -2.0f);
    c_values[2] = make_cf32(203.0f, -3.0f);
    c_values[3] = make_cf32(204.0f, -4.0f);
    c_values[4] = make_cf32(205.0f, -5.0f);
    c_values[5] = make_cf32(206.0f, -6.0f);
}

static void stub_csymm_cblas(fb_layout_t layout, fb_side_t side,
                             fb_uplo_t uplo, int m, int n,
                             const void *alpha, const void *a, int lda,
                             const void *b, int ldb,
                             const void *beta, void *c, int ldc)
{
    fb_complex_float_t *c_values = (fb_complex_float_t *)c;

    g_csymm_cblas_call.called += 1;
    g_csymm_cblas_call.layout = layout;
    g_csymm_cblas_call.side = side;
    g_csymm_cblas_call.uplo = uplo;
    g_csymm_cblas_call.m = m;
    g_csymm_cblas_call.n = n;
    g_csymm_cblas_call.lda = lda;
    g_csymm_cblas_call.ldb = ldb;
    g_csymm_cblas_call.ldc = ldc;
    g_csymm_cblas_call.alpha = alpha;
    g_csymm_cblas_call.beta = beta;
    g_csymm_cblas_call.alpha_value = *(const fb_complex_float_t *)alpha;
    g_csymm_cblas_call.beta_value = *(const fb_complex_float_t *)beta;
    g_csymm_cblas_call.a = a;
    g_csymm_cblas_call.b = b;
    g_csymm_cblas_call.c = c;

    c_values[0] = make_cf32(211.0f, 1.0f);
    c_values[1] = make_cf32(212.0f, 2.0f);
    c_values[2] = make_cf32(213.0f, 3.0f);
    c_values[3] = make_cf32(214.0f, 4.0f);
    c_values[4] = make_cf32(215.0f, 5.0f);
    c_values[5] = make_cf32(216.0f, 6.0f);
}

static void stub_zsymm_fortran(const int *order, const int *side,
                               const int *uplo, const int *m, const int *n,
                               const void *alpha, const void *a,
                               const int *lda, const void *b,
                               const int *ldb, const void *beta,
                               void *c, const int *ldc)
{
    fb_complex_double_t *c_values = (fb_complex_double_t *)c;

    g_zsymm_fortran_call.called += 1;
    g_zsymm_fortran_call.order = *order;
    g_zsymm_fortran_call.side = *side;
    g_zsymm_fortran_call.uplo = *uplo;
    g_zsymm_fortran_call.m = *m;
    g_zsymm_fortran_call.n = *n;
    g_zsymm_fortran_call.lda = *lda;
    g_zsymm_fortran_call.ldb = *ldb;
    g_zsymm_fortran_call.ldc = *ldc;
    g_zsymm_fortran_call.alpha = alpha;
    g_zsymm_fortran_call.beta = beta;
    g_zsymm_fortran_call.alpha_value = *(const fb_complex_double_t *)alpha;
    g_zsymm_fortran_call.beta_value = *(const fb_complex_double_t *)beta;
    g_zsymm_fortran_call.a = a;
    g_zsymm_fortran_call.b = b;
    g_zsymm_fortran_call.c = c;

    c_values[0] = make_cf64(221.0, -1.0);
    c_values[1] = make_cf64(222.0, -2.0);
    c_values[2] = make_cf64(223.0, -3.0);
    c_values[3] = make_cf64(224.0, -4.0);
    c_values[4] = make_cf64(225.0, -5.0);
    c_values[5] = make_cf64(226.0, -6.0);
}

static void stub_zsymm_cblas(fb_layout_t layout, fb_side_t side,
                             fb_uplo_t uplo, int m, int n,
                             const void *alpha, const void *a, int lda,
                             const void *b, int ldb,
                             const void *beta, void *c, int ldc)
{
    fb_complex_double_t *c_values = (fb_complex_double_t *)c;

    g_zsymm_cblas_call.called += 1;
    g_zsymm_cblas_call.layout = layout;
    g_zsymm_cblas_call.side = side;
    g_zsymm_cblas_call.uplo = uplo;
    g_zsymm_cblas_call.m = m;
    g_zsymm_cblas_call.n = n;
    g_zsymm_cblas_call.lda = lda;
    g_zsymm_cblas_call.ldb = ldb;
    g_zsymm_cblas_call.ldc = ldc;
    g_zsymm_cblas_call.alpha = alpha;
    g_zsymm_cblas_call.beta = beta;
    g_zsymm_cblas_call.alpha_value = *(const fb_complex_double_t *)alpha;
    g_zsymm_cblas_call.beta_value = *(const fb_complex_double_t *)beta;
    g_zsymm_cblas_call.a = a;
    g_zsymm_cblas_call.b = b;
    g_zsymm_cblas_call.c = c;

    c_values[0] = make_cf64(231.0, 1.0);
    c_values[1] = make_cf64(232.0, 2.0);
    c_values[2] = make_cf64(233.0, 3.0);
    c_values[3] = make_cf64(234.0, 4.0);
    c_values[4] = make_cf64(235.0, 5.0);
    c_values[5] = make_cf64(236.0, 6.0);
}

static int check_csymm_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_csymm_cblas_fn thunk = NULL;
    fb_complex_float_t alpha = make_cf32(1.5f, -0.5f);
    fb_complex_float_t beta = make_cf32(0.75f, 0.25f);
    fb_complex_float_t a[4] = {
        make_cf32(1.0f, -1.0f), make_cf32(2.0f, -2.0f),
        make_cf32(3.0f, -3.0f), make_cf32(4.0f, -4.0f)
    };
    fb_complex_float_t b[6] = {
        make_cf32(5.0f, 1.0f), make_cf32(6.0f, 2.0f),
        make_cf32(7.0f, 3.0f), make_cf32(8.0f, 4.0f),
        make_cf32(9.0f, 5.0f), make_cf32(10.0f, 6.0f)
    };
    fb_complex_float_t c[6] = {
        make_cf32(0.0f, 0.0f), make_cf32(0.0f, 0.0f),
        make_cf32(0.0f, 0.0f), make_cf32(0.0f, 0.0f),
        make_cf32(0.0f, 0.0f), make_cf32(0.0f, 0.0f)
    };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_csymm_fortran_call, 0, sizeof(g_csymm_fortran_call));

    vtable.ext_ops[FB_OP_CSYMM][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_csymm_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CSYMM);

    thunk = (fb_csymm_cblas_fn)vtable.ext_ops[FB_OP_CSYMM][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CSYMM Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    thunk(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_LOWER, 2, 3, &alpha, a, 2, b, 3, &beta, c, 3);
    if (g_csymm_fortran_call.called != 1 ||
        g_csymm_fortran_call.order != FB_LAYOUT_ROW_MAJOR ||
        g_csymm_fortran_call.side != FB_LEFT ||
        g_csymm_fortran_call.uplo != FB_LOWER ||
        g_csymm_fortran_call.m != 2 ||
        g_csymm_fortran_call.n != 3 ||
        g_csymm_fortran_call.lda != 2 ||
        g_csymm_fortran_call.ldb != 3 ||
        g_csymm_fortran_call.ldc != 3 ||
        g_csymm_fortran_call.alpha != &alpha ||
        g_csymm_fortran_call.beta != &beta ||
        !cf32_eq(g_csymm_fortran_call.alpha_value, alpha) ||
        !cf32_eq(g_csymm_fortran_call.beta_value, beta) ||
        g_csymm_fortran_call.a != a ||
        g_csymm_fortran_call.b != b ||
        g_csymm_fortran_call.c != c ||
        !cf32_eq(c[0], make_cf32(201.0f, -1.0f)) ||
        !cf32_eq(c[1], make_cf32(202.0f, -2.0f)) ||
        !cf32_eq(c[2], make_cf32(203.0f, -3.0f)) ||
        !cf32_eq(c[3], make_cf32(204.0f, -4.0f)) ||
        !cf32_eq(c[4], make_cf32(205.0f, -5.0f)) ||
        !cf32_eq(c[5], make_cf32(206.0f, -6.0f))) {
        fprintf(stderr, "[FAIL] CSYMM Fortran->CBLAS thunk did not forward complex symmetric matrix-matrix arguments correctly\n");
        return 1;
    }

    printf("[PASS] CSYMM Fortran->CBLAS thunk forwards complex symmetric matrix-matrix arguments unchanged\n");
    return 0;
}

static int check_csymm_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_csymm_fortran_slot_fn thunk = NULL;
    int order = FB_LAYOUT_COL_MAJOR;
    int side = FB_RIGHT;
    int uplo = FB_UPPER;
    int m = 2;
    int n = 3;
    int lda = 3;
    int ldb = 3;
    int ldc = 3;
    fb_complex_float_t alpha = make_cf32(2.5f, -1.5f);
    fb_complex_float_t beta = make_cf32(1.25f, -0.75f);
    fb_complex_float_t a[9] = {
        make_cf32(11.0f, 1.0f), make_cf32(12.0f, 2.0f), make_cf32(13.0f, 3.0f),
        make_cf32(14.0f, 4.0f), make_cf32(15.0f, 5.0f), make_cf32(16.0f, 6.0f),
        make_cf32(17.0f, 7.0f), make_cf32(18.0f, 8.0f), make_cf32(19.0f, 9.0f)
    };
    fb_complex_float_t b[6] = {
        make_cf32(20.0f, -1.0f), make_cf32(21.0f, -2.0f),
        make_cf32(22.0f, -3.0f), make_cf32(23.0f, -4.0f),
        make_cf32(24.0f, -5.0f), make_cf32(25.0f, -6.0f)
    };
    fb_complex_float_t c[6] = {
        make_cf32(0.0f, 0.0f), make_cf32(0.0f, 0.0f),
        make_cf32(0.0f, 0.0f), make_cf32(0.0f, 0.0f),
        make_cf32(0.0f, 0.0f), make_cf32(0.0f, 0.0f)
    };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_csymm_cblas_call, 0, sizeof(g_csymm_cblas_call));

    vtable.ext_ops[FB_OP_CSYMM][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_csymm_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CSYMM);

    thunk = (fb_csymm_fortran_slot_fn)vtable.ext_ops[FB_OP_CSYMM][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CSYMM CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&order, &side, &uplo, &m, &n, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
    if (g_csymm_cblas_call.called != 1 ||
        g_csymm_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_csymm_cblas_call.side != FB_RIGHT ||
        g_csymm_cblas_call.uplo != FB_UPPER ||
        g_csymm_cblas_call.m != 2 ||
        g_csymm_cblas_call.n != 3 ||
        g_csymm_cblas_call.lda != 3 ||
        g_csymm_cblas_call.ldb != 3 ||
        g_csymm_cblas_call.ldc != 3 ||
        g_csymm_cblas_call.alpha != &alpha ||
        g_csymm_cblas_call.beta != &beta ||
        !cf32_eq(g_csymm_cblas_call.alpha_value, alpha) ||
        !cf32_eq(g_csymm_cblas_call.beta_value, beta) ||
        g_csymm_cblas_call.a != a ||
        g_csymm_cblas_call.b != b ||
        g_csymm_cblas_call.c != c ||
        !cf32_eq(c[0], make_cf32(211.0f, 1.0f)) ||
        !cf32_eq(c[1], make_cf32(212.0f, 2.0f)) ||
        !cf32_eq(c[2], make_cf32(213.0f, 3.0f)) ||
        !cf32_eq(c[3], make_cf32(214.0f, 4.0f)) ||
        !cf32_eq(c[4], make_cf32(215.0f, 5.0f)) ||
        !cf32_eq(c[5], make_cf32(216.0f, 6.0f))) {
        fprintf(stderr, "[FAIL] CSYMM CBLAS->Fortran thunk did not map complex symmetric matrix-matrix arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] CSYMM CBLAS->Fortran thunk maps complex symmetric matrix-matrix arguments into the C entry\n");
    return 0;
}

static int check_zsymm_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_zsymm_cblas_fn thunk = NULL;
    fb_complex_double_t alpha = make_cf64(3.5, -0.5);
    fb_complex_double_t beta = make_cf64(2.75, 0.5);
    fb_complex_double_t a[4] = {
        make_cf64(26.0, -1.0), make_cf64(27.0, -2.0),
        make_cf64(28.0, -3.0), make_cf64(29.0, -4.0)
    };
    fb_complex_double_t b[6] = {
        make_cf64(30.0, 1.0), make_cf64(31.0, 2.0),
        make_cf64(32.0, 3.0), make_cf64(33.0, 4.0),
        make_cf64(34.0, 5.0), make_cf64(35.0, 6.0)
    };
    fb_complex_double_t c[6] = {
        make_cf64(0.0, 0.0), make_cf64(0.0, 0.0),
        make_cf64(0.0, 0.0), make_cf64(0.0, 0.0),
        make_cf64(0.0, 0.0), make_cf64(0.0, 0.0)
    };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zsymm_fortran_call, 0, sizeof(g_zsymm_fortran_call));

    vtable.ext_ops[FB_OP_ZSYMM][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_zsymm_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_ZSYMM);

    thunk = (fb_zsymm_cblas_fn)vtable.ext_ops[FB_OP_ZSYMM][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZSYMM Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    thunk(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_LOWER, 2, 3, &alpha, a, 2, b, 3, &beta, c, 3);
    if (g_zsymm_fortran_call.called != 1 ||
        g_zsymm_fortran_call.order != FB_LAYOUT_ROW_MAJOR ||
        g_zsymm_fortran_call.side != FB_LEFT ||
        g_zsymm_fortran_call.uplo != FB_LOWER ||
        g_zsymm_fortran_call.m != 2 ||
        g_zsymm_fortran_call.n != 3 ||
        g_zsymm_fortran_call.lda != 2 ||
        g_zsymm_fortran_call.ldb != 3 ||
        g_zsymm_fortran_call.ldc != 3 ||
        g_zsymm_fortran_call.alpha != &alpha ||
        g_zsymm_fortran_call.beta != &beta ||
        !cf64_eq(g_zsymm_fortran_call.alpha_value, alpha) ||
        !cf64_eq(g_zsymm_fortran_call.beta_value, beta) ||
        g_zsymm_fortran_call.a != a ||
        g_zsymm_fortran_call.b != b ||
        g_zsymm_fortran_call.c != c ||
        !cf64_eq(c[0], make_cf64(221.0, -1.0)) ||
        !cf64_eq(c[1], make_cf64(222.0, -2.0)) ||
        !cf64_eq(c[2], make_cf64(223.0, -3.0)) ||
        !cf64_eq(c[3], make_cf64(224.0, -4.0)) ||
        !cf64_eq(c[4], make_cf64(225.0, -5.0)) ||
        !cf64_eq(c[5], make_cf64(226.0, -6.0))) {
        fprintf(stderr, "[FAIL] ZSYMM Fortran->CBLAS thunk did not forward double-complex symmetric matrix-matrix arguments correctly\n");
        return 1;
    }

    printf("[PASS] ZSYMM Fortran->CBLAS thunk forwards double-complex symmetric matrix-matrix arguments unchanged\n");
    return 0;
}

static int check_zsymm_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_zsymm_fortran_slot_fn thunk = NULL;
    int order = FB_LAYOUT_COL_MAJOR;
    int side = FB_RIGHT;
    int uplo = FB_UPPER;
    int m = 2;
    int n = 3;
    int lda = 3;
    int ldb = 3;
    int ldc = 3;
    fb_complex_double_t alpha = make_cf64(4.5, -1.5);
    fb_complex_double_t beta = make_cf64(3.25, -0.75);
    fb_complex_double_t a[9] = {
        make_cf64(36.0, 1.0), make_cf64(37.0, 2.0), make_cf64(38.0, 3.0),
        make_cf64(39.0, 4.0), make_cf64(40.0, 5.0), make_cf64(41.0, 6.0),
        make_cf64(42.0, 7.0), make_cf64(43.0, 8.0), make_cf64(44.0, 9.0)
    };
    fb_complex_double_t b[6] = {
        make_cf64(45.0, -1.0), make_cf64(46.0, -2.0),
        make_cf64(47.0, -3.0), make_cf64(48.0, -4.0),
        make_cf64(49.0, -5.0), make_cf64(50.0, -6.0)
    };
    fb_complex_double_t c[6] = {
        make_cf64(0.0, 0.0), make_cf64(0.0, 0.0),
        make_cf64(0.0, 0.0), make_cf64(0.0, 0.0),
        make_cf64(0.0, 0.0), make_cf64(0.0, 0.0)
    };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zsymm_cblas_call, 0, sizeof(g_zsymm_cblas_call));

    vtable.ext_ops[FB_OP_ZSYMM][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_zsymm_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_ZSYMM);

    thunk = (fb_zsymm_fortran_slot_fn)vtable.ext_ops[FB_OP_ZSYMM][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZSYMM CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&order, &side, &uplo, &m, &n, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
    if (g_zsymm_cblas_call.called != 1 ||
        g_zsymm_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_zsymm_cblas_call.side != FB_RIGHT ||
        g_zsymm_cblas_call.uplo != FB_UPPER ||
        g_zsymm_cblas_call.m != 2 ||
        g_zsymm_cblas_call.n != 3 ||
        g_zsymm_cblas_call.lda != 3 ||
        g_zsymm_cblas_call.ldb != 3 ||
        g_zsymm_cblas_call.ldc != 3 ||
        g_zsymm_cblas_call.alpha != &alpha ||
        g_zsymm_cblas_call.beta != &beta ||
        !cf64_eq(g_zsymm_cblas_call.alpha_value, alpha) ||
        !cf64_eq(g_zsymm_cblas_call.beta_value, beta) ||
        g_zsymm_cblas_call.a != a ||
        g_zsymm_cblas_call.b != b ||
        g_zsymm_cblas_call.c != c ||
        !cf64_eq(c[0], make_cf64(231.0, 1.0)) ||
        !cf64_eq(c[1], make_cf64(232.0, 2.0)) ||
        !cf64_eq(c[2], make_cf64(233.0, 3.0)) ||
        !cf64_eq(c[3], make_cf64(234.0, 4.0)) ||
        !cf64_eq(c[4], make_cf64(235.0, 5.0)) ||
        !cf64_eq(c[5], make_cf64(236.0, 6.0))) {
        fprintf(stderr, "[FAIL] ZSYMM CBLAS->Fortran thunk did not map double-complex symmetric matrix-matrix arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] ZSYMM CBLAS->Fortran thunk maps double-complex symmetric matrix-matrix arguments into the C entry\n");
    return 0;
}

int main(void)
{
    int status = 0;

    status |= check_csymm_fortran_to_cblas();
    status |= check_csymm_cblas_to_fortran();
    status |= check_zsymm_fortran_to_cblas();
    status |= check_zsymm_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}