#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef void (*fb_csyrk_cblas_fn)(fb_layout_t layout, fb_uplo_t uplo,
                                  fb_transpose_t trans, int n, int k,
                                  const void *alpha, const void *a, int lda,
                                  const void *beta, void *c, int ldc);
typedef void (*fb_zsyrk_cblas_fn)(fb_layout_t layout, fb_uplo_t uplo,
                                  fb_transpose_t trans, int n, int k,
                                  const void *alpha, const void *a, int lda,
                                  const void *beta, void *c, int ldc);

typedef void (*fb_csyrk_fortran_slot_fn)(const int *order, const int *uplo,
                                         const int *trans, const int *n,
                                         const int *k, const void *alpha,
                                         const void *a, const int *lda,
                                         const void *beta, void *c,
                                         const int *ldc);
typedef void (*fb_zsyrk_fortran_slot_fn)(const int *order, const int *uplo,
                                         const int *trans, const int *n,
                                         const int *k, const void *alpha,
                                         const void *a, const int *lda,
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
    int ldc;
    const void *alpha;
    const void *beta;
    fb_complex_float_t alpha_value;
    fb_complex_float_t beta_value;
    const void *a;
    void *c;
} g_csyrk_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    fb_uplo_t uplo;
    fb_transpose_t trans;
    int n;
    int k;
    int lda;
    int ldc;
    const void *alpha;
    const void *beta;
    fb_complex_float_t alpha_value;
    fb_complex_float_t beta_value;
    const void *a;
    void *c;
} g_csyrk_cblas_call;

static struct {
    int called;
    int order;
    int uplo;
    int trans;
    int n;
    int k;
    int lda;
    int ldc;
    const void *alpha;
    const void *beta;
    fb_complex_double_t alpha_value;
    fb_complex_double_t beta_value;
    const void *a;
    void *c;
} g_zsyrk_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    fb_uplo_t uplo;
    fb_transpose_t trans;
    int n;
    int k;
    int lda;
    int ldc;
    const void *alpha;
    const void *beta;
    fb_complex_double_t alpha_value;
    fb_complex_double_t beta_value;
    const void *a;
    void *c;
} g_zsyrk_cblas_call;

static void stub_csyrk_fortran(const int *order, const int *uplo,
                               const int *trans, const int *n, const int *k,
                               const void *alpha, const void *a,
                               const int *lda, const void *beta,
                               void *c, const int *ldc)
{
    fb_complex_float_t *c_values = (fb_complex_float_t *)c;

    g_csyrk_fortran_call.called += 1;
    g_csyrk_fortran_call.order = *order;
    g_csyrk_fortran_call.uplo = *uplo;
    g_csyrk_fortran_call.trans = *trans;
    g_csyrk_fortran_call.n = *n;
    g_csyrk_fortran_call.k = *k;
    g_csyrk_fortran_call.lda = *lda;
    g_csyrk_fortran_call.ldc = *ldc;
    g_csyrk_fortran_call.alpha = alpha;
    g_csyrk_fortran_call.beta = beta;
    g_csyrk_fortran_call.alpha_value = *(const fb_complex_float_t *)alpha;
    g_csyrk_fortran_call.beta_value = *(const fb_complex_float_t *)beta;
    g_csyrk_fortran_call.a = a;
    g_csyrk_fortran_call.c = c;

    c_values[0] = make_cf32(281.0f, -1.0f);
    c_values[1] = make_cf32(282.0f, -2.0f);
    c_values[2] = make_cf32(283.0f, -3.0f);
    c_values[3] = make_cf32(284.0f, -4.0f);
}

static void stub_csyrk_cblas(fb_layout_t layout, fb_uplo_t uplo,
                             fb_transpose_t trans, int n, int k,
                             const void *alpha, const void *a, int lda,
                             const void *beta, void *c, int ldc)
{
    fb_complex_float_t *c_values = (fb_complex_float_t *)c;

    g_csyrk_cblas_call.called += 1;
    g_csyrk_cblas_call.layout = layout;
    g_csyrk_cblas_call.uplo = uplo;
    g_csyrk_cblas_call.trans = trans;
    g_csyrk_cblas_call.n = n;
    g_csyrk_cblas_call.k = k;
    g_csyrk_cblas_call.lda = lda;
    g_csyrk_cblas_call.ldc = ldc;
    g_csyrk_cblas_call.alpha = alpha;
    g_csyrk_cblas_call.beta = beta;
    g_csyrk_cblas_call.alpha_value = *(const fb_complex_float_t *)alpha;
    g_csyrk_cblas_call.beta_value = *(const fb_complex_float_t *)beta;
    g_csyrk_cblas_call.a = a;
    g_csyrk_cblas_call.c = c;

    c_values[0] = make_cf32(291.0f, 1.0f);
    c_values[1] = make_cf32(292.0f, 2.0f);
    c_values[2] = make_cf32(293.0f, 3.0f);
    c_values[3] = make_cf32(294.0f, 4.0f);
}

static void stub_zsyrk_fortran(const int *order, const int *uplo,
                               const int *trans, const int *n, const int *k,
                               const void *alpha, const void *a,
                               const int *lda, const void *beta,
                               void *c, const int *ldc)
{
    fb_complex_double_t *c_values = (fb_complex_double_t *)c;

    g_zsyrk_fortran_call.called += 1;
    g_zsyrk_fortran_call.order = *order;
    g_zsyrk_fortran_call.uplo = *uplo;
    g_zsyrk_fortran_call.trans = *trans;
    g_zsyrk_fortran_call.n = *n;
    g_zsyrk_fortran_call.k = *k;
    g_zsyrk_fortran_call.lda = *lda;
    g_zsyrk_fortran_call.ldc = *ldc;
    g_zsyrk_fortran_call.alpha = alpha;
    g_zsyrk_fortran_call.beta = beta;
    g_zsyrk_fortran_call.alpha_value = *(const fb_complex_double_t *)alpha;
    g_zsyrk_fortran_call.beta_value = *(const fb_complex_double_t *)beta;
    g_zsyrk_fortran_call.a = a;
    g_zsyrk_fortran_call.c = c;

    c_values[0] = make_cf64(301.0, -1.0);
    c_values[1] = make_cf64(302.0, -2.0);
    c_values[2] = make_cf64(303.0, -3.0);
    c_values[3] = make_cf64(304.0, -4.0);
}

static void stub_zsyrk_cblas(fb_layout_t layout, fb_uplo_t uplo,
                             fb_transpose_t trans, int n, int k,
                             const void *alpha, const void *a, int lda,
                             const void *beta, void *c, int ldc)
{
    fb_complex_double_t *c_values = (fb_complex_double_t *)c;

    g_zsyrk_cblas_call.called += 1;
    g_zsyrk_cblas_call.layout = layout;
    g_zsyrk_cblas_call.uplo = uplo;
    g_zsyrk_cblas_call.trans = trans;
    g_zsyrk_cblas_call.n = n;
    g_zsyrk_cblas_call.k = k;
    g_zsyrk_cblas_call.lda = lda;
    g_zsyrk_cblas_call.ldc = ldc;
    g_zsyrk_cblas_call.alpha = alpha;
    g_zsyrk_cblas_call.beta = beta;
    g_zsyrk_cblas_call.alpha_value = *(const fb_complex_double_t *)alpha;
    g_zsyrk_cblas_call.beta_value = *(const fb_complex_double_t *)beta;
    g_zsyrk_cblas_call.a = a;
    g_zsyrk_cblas_call.c = c;

    c_values[0] = make_cf64(311.0, 1.0);
    c_values[1] = make_cf64(312.0, 2.0);
    c_values[2] = make_cf64(313.0, 3.0);
    c_values[3] = make_cf64(314.0, 4.0);
}

static int check_csyrk_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_csyrk_cblas_fn thunk = NULL;
    fb_complex_float_t alpha = make_cf32(1.5f, -0.5f);
    fb_complex_float_t beta = make_cf32(0.75f, 0.25f);
    fb_complex_float_t a[6] = {
        make_cf32(1.0f, -1.0f), make_cf32(2.0f, -2.0f),
        make_cf32(3.0f, -3.0f), make_cf32(4.0f, -4.0f),
        make_cf32(5.0f, -5.0f), make_cf32(6.0f, -6.0f)
    };
    fb_complex_float_t c[4] = {
        make_cf32(0.0f, 0.0f), make_cf32(0.0f, 0.0f),
        make_cf32(0.0f, 0.0f), make_cf32(0.0f, 0.0f)
    };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_csyrk_fortran_call, 0, sizeof(g_csyrk_fortran_call));

    vtable.ext_ops[FB_OP_CSYRK][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_csyrk_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CSYRK);

    thunk = (fb_csyrk_cblas_fn)vtable.ext_ops[FB_OP_CSYRK][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CSYRK Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    thunk(FB_LAYOUT_ROW_MAJOR, FB_LOWER, FB_CONJ_TRANS, 2, 3, &alpha, a, 3, &beta, c, 2);
    if (g_csyrk_fortran_call.called != 1 ||
        g_csyrk_fortran_call.order != FB_LAYOUT_ROW_MAJOR ||
        g_csyrk_fortran_call.uplo != FB_LOWER ||
        g_csyrk_fortran_call.trans != FB_CONJ_TRANS ||
        g_csyrk_fortran_call.n != 2 ||
        g_csyrk_fortran_call.k != 3 ||
        g_csyrk_fortran_call.lda != 3 ||
        g_csyrk_fortran_call.ldc != 2 ||
        g_csyrk_fortran_call.alpha != &alpha ||
        g_csyrk_fortran_call.beta != &beta ||
        !cf32_eq(g_csyrk_fortran_call.alpha_value, alpha) ||
        !cf32_eq(g_csyrk_fortran_call.beta_value, beta) ||
        g_csyrk_fortran_call.a != a ||
        g_csyrk_fortran_call.c != c ||
        !cf32_eq(c[0], make_cf32(281.0f, -1.0f)) ||
        !cf32_eq(c[1], make_cf32(282.0f, -2.0f)) ||
        !cf32_eq(c[2], make_cf32(283.0f, -3.0f)) ||
        !cf32_eq(c[3], make_cf32(284.0f, -4.0f))) {
        fprintf(stderr, "[FAIL] CSYRK Fortran->CBLAS thunk did not forward complex symmetric rank-k arguments correctly\n");
        return 1;
    }

    printf("[PASS] CSYRK Fortran->CBLAS thunk forwards complex symmetric rank-k arguments unchanged\n");
    return 0;
}

static int check_csyrk_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_csyrk_fortran_slot_fn thunk = NULL;
    int order = FB_LAYOUT_COL_MAJOR;
    int uplo = FB_UPPER;
    int trans = FB_NO_TRANS;
    int n = 2;
    int k = 3;
    int lda = 2;
    int ldc = 2;
    fb_complex_float_t alpha = make_cf32(2.5f, -1.5f);
    fb_complex_float_t beta = make_cf32(1.25f, -0.75f);
    fb_complex_float_t a[6] = {
        make_cf32(7.0f, 1.0f), make_cf32(8.0f, 2.0f),
        make_cf32(9.0f, 3.0f), make_cf32(10.0f, 4.0f),
        make_cf32(11.0f, 5.0f), make_cf32(12.0f, 6.0f)
    };
    fb_complex_float_t c[4] = {
        make_cf32(0.0f, 0.0f), make_cf32(0.0f, 0.0f),
        make_cf32(0.0f, 0.0f), make_cf32(0.0f, 0.0f)
    };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_csyrk_cblas_call, 0, sizeof(g_csyrk_cblas_call));

    vtable.ext_ops[FB_OP_CSYRK][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_csyrk_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CSYRK);

    thunk = (fb_csyrk_fortran_slot_fn)vtable.ext_ops[FB_OP_CSYRK][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CSYRK CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&order, &uplo, &trans, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
    if (g_csyrk_cblas_call.called != 1 ||
        g_csyrk_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_csyrk_cblas_call.uplo != FB_UPPER ||
        g_csyrk_cblas_call.trans != FB_NO_TRANS ||
        g_csyrk_cblas_call.n != 2 ||
        g_csyrk_cblas_call.k != 3 ||
        g_csyrk_cblas_call.lda != 2 ||
        g_csyrk_cblas_call.ldc != 2 ||
        g_csyrk_cblas_call.alpha != &alpha ||
        g_csyrk_cblas_call.beta != &beta ||
        !cf32_eq(g_csyrk_cblas_call.alpha_value, alpha) ||
        !cf32_eq(g_csyrk_cblas_call.beta_value, beta) ||
        g_csyrk_cblas_call.a != a ||
        g_csyrk_cblas_call.c != c ||
        !cf32_eq(c[0], make_cf32(291.0f, 1.0f)) ||
        !cf32_eq(c[1], make_cf32(292.0f, 2.0f)) ||
        !cf32_eq(c[2], make_cf32(293.0f, 3.0f)) ||
        !cf32_eq(c[3], make_cf32(294.0f, 4.0f))) {
        fprintf(stderr, "[FAIL] CSYRK CBLAS->Fortran thunk did not map complex symmetric rank-k arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] CSYRK CBLAS->Fortran thunk maps complex symmetric rank-k arguments into the C entry\n");
    return 0;
}

static int check_zsyrk_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_zsyrk_cblas_fn thunk = NULL;
    fb_complex_double_t alpha = make_cf64(3.5, -0.5);
    fb_complex_double_t beta = make_cf64(2.75, 0.5);
    fb_complex_double_t a[6] = {
        make_cf64(13.0, -1.0), make_cf64(14.0, -2.0),
        make_cf64(15.0, -3.0), make_cf64(16.0, -4.0),
        make_cf64(17.0, -5.0), make_cf64(18.0, -6.0)
    };
    fb_complex_double_t c[4] = {
        make_cf64(0.0, 0.0), make_cf64(0.0, 0.0),
        make_cf64(0.0, 0.0), make_cf64(0.0, 0.0)
    };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zsyrk_fortran_call, 0, sizeof(g_zsyrk_fortran_call));

    vtable.ext_ops[FB_OP_ZSYRK][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_zsyrk_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_ZSYRK);

    thunk = (fb_zsyrk_cblas_fn)vtable.ext_ops[FB_OP_ZSYRK][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZSYRK Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    thunk(FB_LAYOUT_ROW_MAJOR, FB_LOWER, FB_CONJ_TRANS, 2, 3, &alpha, a, 3, &beta, c, 2);
    if (g_zsyrk_fortran_call.called != 1 ||
        g_zsyrk_fortran_call.order != FB_LAYOUT_ROW_MAJOR ||
        g_zsyrk_fortran_call.uplo != FB_LOWER ||
        g_zsyrk_fortran_call.trans != FB_CONJ_TRANS ||
        g_zsyrk_fortran_call.n != 2 ||
        g_zsyrk_fortran_call.k != 3 ||
        g_zsyrk_fortran_call.lda != 3 ||
        g_zsyrk_fortran_call.ldc != 2 ||
        g_zsyrk_fortran_call.alpha != &alpha ||
        g_zsyrk_fortran_call.beta != &beta ||
        !cf64_eq(g_zsyrk_fortran_call.alpha_value, alpha) ||
        !cf64_eq(g_zsyrk_fortran_call.beta_value, beta) ||
        g_zsyrk_fortran_call.a != a ||
        g_zsyrk_fortran_call.c != c ||
        !cf64_eq(c[0], make_cf64(301.0, -1.0)) ||
        !cf64_eq(c[1], make_cf64(302.0, -2.0)) ||
        !cf64_eq(c[2], make_cf64(303.0, -3.0)) ||
        !cf64_eq(c[3], make_cf64(304.0, -4.0))) {
        fprintf(stderr, "[FAIL] ZSYRK Fortran->CBLAS thunk did not forward double-complex symmetric rank-k arguments correctly\n");
        return 1;
    }

    printf("[PASS] ZSYRK Fortran->CBLAS thunk forwards double-complex symmetric rank-k arguments unchanged\n");
    return 0;
}

static int check_zsyrk_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_zsyrk_fortran_slot_fn thunk = NULL;
    int order = FB_LAYOUT_COL_MAJOR;
    int uplo = FB_UPPER;
    int trans = FB_NO_TRANS;
    int n = 2;
    int k = 3;
    int lda = 2;
    int ldc = 2;
    fb_complex_double_t alpha = make_cf64(4.5, -1.5);
    fb_complex_double_t beta = make_cf64(3.25, -0.75);
    fb_complex_double_t a[6] = {
        make_cf64(19.0, 1.0), make_cf64(20.0, 2.0),
        make_cf64(21.0, 3.0), make_cf64(22.0, 4.0),
        make_cf64(23.0, 5.0), make_cf64(24.0, 6.0)
    };
    fb_complex_double_t c[4] = {
        make_cf64(0.0, 0.0), make_cf64(0.0, 0.0),
        make_cf64(0.0, 0.0), make_cf64(0.0, 0.0)
    };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zsyrk_cblas_call, 0, sizeof(g_zsyrk_cblas_call));

    vtable.ext_ops[FB_OP_ZSYRK][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_zsyrk_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_ZSYRK);

    thunk = (fb_zsyrk_fortran_slot_fn)vtable.ext_ops[FB_OP_ZSYRK][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZSYRK CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&order, &uplo, &trans, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
    if (g_zsyrk_cblas_call.called != 1 ||
        g_zsyrk_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_zsyrk_cblas_call.uplo != FB_UPPER ||
        g_zsyrk_cblas_call.trans != FB_NO_TRANS ||
        g_zsyrk_cblas_call.n != 2 ||
        g_zsyrk_cblas_call.k != 3 ||
        g_zsyrk_cblas_call.lda != 2 ||
        g_zsyrk_cblas_call.ldc != 2 ||
        g_zsyrk_cblas_call.alpha != &alpha ||
        g_zsyrk_cblas_call.beta != &beta ||
        !cf64_eq(g_zsyrk_cblas_call.alpha_value, alpha) ||
        !cf64_eq(g_zsyrk_cblas_call.beta_value, beta) ||
        g_zsyrk_cblas_call.a != a ||
        g_zsyrk_cblas_call.c != c ||
        !cf64_eq(c[0], make_cf64(311.0, 1.0)) ||
        !cf64_eq(c[1], make_cf64(312.0, 2.0)) ||
        !cf64_eq(c[2], make_cf64(313.0, 3.0)) ||
        !cf64_eq(c[3], make_cf64(314.0, 4.0))) {
        fprintf(stderr, "[FAIL] ZSYRK CBLAS->Fortran thunk did not map double-complex symmetric rank-k arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] ZSYRK CBLAS->Fortran thunk maps double-complex symmetric rank-k arguments into the C entry\n");
    return 0;
}

int main(void)
{
    int status = 0;

    status |= check_csyrk_fortran_to_cblas();
    status |= check_csyrk_cblas_to_fortran();
    status |= check_zsyrk_fortran_to_cblas();
    status |= check_zsyrk_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}