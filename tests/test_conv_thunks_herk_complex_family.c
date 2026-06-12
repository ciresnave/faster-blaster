#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef void (*fb_cherk_cblas_fn)(fb_layout_t layout, fb_uplo_t uplo,
                                  fb_transpose_t trans, int n, int k,
                                  float alpha, const fb_complex_float_t *a,
                                  int lda, float beta,
                                  fb_complex_float_t *c, int ldc);
typedef void (*fb_zherk_cblas_fn)(fb_layout_t layout, fb_uplo_t uplo,
                                  fb_transpose_t trans, int n, int k,
                                  double alpha, const fb_complex_double_t *a,
                                  int lda, double beta,
                                  fb_complex_double_t *c, int ldc);

typedef void (*fb_cherk_fortran_slot_fn)(const int *order, const int *uplo,
                                         const int *trans, const int *n,
                                         const int *k, const float *alpha,
                                         const fb_complex_float_t *a,
                                         const int *lda, const float *beta,
                                         fb_complex_float_t *c,
                                         const int *ldc);
typedef void (*fb_zherk_fortran_slot_fn)(const int *order, const int *uplo,
                                         const int *trans, const int *n,
                                         const int *k, const double *alpha,
                                         const fb_complex_double_t *a,
                                         const int *lda, const double *beta,
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
    int ldc;
    float alpha_value;
    float beta_value;
    const fb_complex_float_t *a;
    fb_complex_float_t *c;
} g_cherk_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    fb_uplo_t uplo;
    fb_transpose_t trans;
    int n;
    int k;
    int lda;
    int ldc;
    float alpha_value;
    float beta_value;
    const fb_complex_float_t *a;
    fb_complex_float_t *c;
} g_cherk_cblas_call;

static struct {
    int called;
    int order;
    int uplo;
    int trans;
    int n;
    int k;
    int lda;
    int ldc;
    double alpha_value;
    double beta_value;
    const fb_complex_double_t *a;
    fb_complex_double_t *c;
} g_zherk_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    fb_uplo_t uplo;
    fb_transpose_t trans;
    int n;
    int k;
    int lda;
    int ldc;
    double alpha_value;
    double beta_value;
    const fb_complex_double_t *a;
    fb_complex_double_t *c;
} g_zherk_cblas_call;

static void stub_cherk_fortran(const int *order, const int *uplo,
                               const int *trans, const int *n, const int *k,
                               const float *alpha,
                               const fb_complex_float_t *a, const int *lda,
                               const float *beta, fb_complex_float_t *c,
                               const int *ldc)
{
    g_cherk_fortran_call.called += 1;
    g_cherk_fortran_call.order = *order;
    g_cherk_fortran_call.uplo = *uplo;
    g_cherk_fortran_call.trans = *trans;
    g_cherk_fortran_call.n = *n;
    g_cherk_fortran_call.k = *k;
    g_cherk_fortran_call.lda = *lda;
    g_cherk_fortran_call.ldc = *ldc;
    g_cherk_fortran_call.alpha_value = *alpha;
    g_cherk_fortran_call.beta_value = *beta;
    g_cherk_fortran_call.a = a;
    g_cherk_fortran_call.c = c;

    c[0] = make_cf32(101.0f, -1.0f);
    c[1] = make_cf32(102.0f, -2.0f);
    c[2] = make_cf32(103.0f, -3.0f);
    c[3] = make_cf32(104.0f, -4.0f);
}

static void stub_cherk_cblas(fb_layout_t layout, fb_uplo_t uplo,
                             fb_transpose_t trans, int n, int k, float alpha,
                             const fb_complex_float_t *a, int lda, float beta,
                             fb_complex_float_t *c, int ldc)
{
    g_cherk_cblas_call.called += 1;
    g_cherk_cblas_call.layout = layout;
    g_cherk_cblas_call.uplo = uplo;
    g_cherk_cblas_call.trans = trans;
    g_cherk_cblas_call.n = n;
    g_cherk_cblas_call.k = k;
    g_cherk_cblas_call.lda = lda;
    g_cherk_cblas_call.ldc = ldc;
    g_cherk_cblas_call.alpha_value = alpha;
    g_cherk_cblas_call.beta_value = beta;
    g_cherk_cblas_call.a = a;
    g_cherk_cblas_call.c = c;

    c[0] = make_cf32(111.0f, 1.0f);
    c[1] = make_cf32(112.0f, 2.0f);
    c[2] = make_cf32(113.0f, 3.0f);
    c[3] = make_cf32(114.0f, 4.0f);
}

static void stub_zherk_fortran(const int *order, const int *uplo,
                               const int *trans, const int *n, const int *k,
                               const double *alpha,
                               const fb_complex_double_t *a, const int *lda,
                               const double *beta, fb_complex_double_t *c,
                               const int *ldc)
{
    g_zherk_fortran_call.called += 1;
    g_zherk_fortran_call.order = *order;
    g_zherk_fortran_call.uplo = *uplo;
    g_zherk_fortran_call.trans = *trans;
    g_zherk_fortran_call.n = *n;
    g_zherk_fortran_call.k = *k;
    g_zherk_fortran_call.lda = *lda;
    g_zherk_fortran_call.ldc = *ldc;
    g_zherk_fortran_call.alpha_value = *alpha;
    g_zherk_fortran_call.beta_value = *beta;
    g_zherk_fortran_call.a = a;
    g_zherk_fortran_call.c = c;

    c[0] = make_cf64(121.0, -1.0);
    c[1] = make_cf64(122.0, -2.0);
    c[2] = make_cf64(123.0, -3.0);
    c[3] = make_cf64(124.0, -4.0);
}

static void stub_zherk_cblas(fb_layout_t layout, fb_uplo_t uplo,
                             fb_transpose_t trans, int n, int k, double alpha,
                             const fb_complex_double_t *a, int lda, double beta,
                             fb_complex_double_t *c, int ldc)
{
    g_zherk_cblas_call.called += 1;
    g_zherk_cblas_call.layout = layout;
    g_zherk_cblas_call.uplo = uplo;
    g_zherk_cblas_call.trans = trans;
    g_zherk_cblas_call.n = n;
    g_zherk_cblas_call.k = k;
    g_zherk_cblas_call.lda = lda;
    g_zherk_cblas_call.ldc = ldc;
    g_zherk_cblas_call.alpha_value = alpha;
    g_zherk_cblas_call.beta_value = beta;
    g_zherk_cblas_call.a = a;
    g_zherk_cblas_call.c = c;

    c[0] = make_cf64(131.0, 1.0);
    c[1] = make_cf64(132.0, 2.0);
    c[2] = make_cf64(133.0, 3.0);
    c[3] = make_cf64(134.0, 4.0);
}

static int check_cherk_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_cherk_cblas_fn thunk = NULL;
    fb_complex_float_t a[4] = {
        make_cf32(1.0f, -1.0f), make_cf32(2.0f, -2.0f),
        make_cf32(3.0f, -3.0f), make_cf32(4.0f, -4.0f)
    };
    fb_complex_float_t c[4] = {
        make_cf32(0.0f, 0.0f), make_cf32(0.0f, 0.0f),
        make_cf32(0.0f, 0.0f), make_cf32(0.0f, 0.0f)
    };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cherk_fortran_call, 0, sizeof(g_cherk_fortran_call));

    vtable.ext_ops[FB_OP_CHERK][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_cherk_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CHERK);

    thunk = (fb_cherk_cblas_fn)vtable.ext_ops[FB_OP_CHERK][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CHERK Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    thunk(FB_LAYOUT_ROW_MAJOR, FB_LOWER, FB_CONJ_TRANS, 2, 3, 1.25f, a, 3, 0.5f, c, 2);
    if (g_cherk_fortran_call.called != 1 ||
        g_cherk_fortran_call.order != FB_LAYOUT_ROW_MAJOR ||
        g_cherk_fortran_call.uplo != FB_LOWER ||
        g_cherk_fortran_call.trans != FB_CONJ_TRANS ||
        g_cherk_fortran_call.n != 2 ||
        g_cherk_fortran_call.k != 3 ||
        g_cherk_fortran_call.lda != 3 ||
        g_cherk_fortran_call.ldc != 2 ||
        g_cherk_fortran_call.alpha_value != 1.25f ||
        g_cherk_fortran_call.beta_value != 0.5f ||
        g_cherk_fortran_call.a != a ||
        g_cherk_fortran_call.c != c ||
        !cf32_eq(c[0], make_cf32(101.0f, -1.0f)) ||
        !cf32_eq(c[1], make_cf32(102.0f, -2.0f)) ||
        !cf32_eq(c[2], make_cf32(103.0f, -3.0f)) ||
        !cf32_eq(c[3], make_cf32(104.0f, -4.0f))) {
        fprintf(stderr, "[FAIL] CHERK Fortran->CBLAS thunk did not forward Hermitian rank-k arguments correctly\n");
        return 1;
    }

    printf("[PASS] CHERK Fortran->CBLAS thunk forwards Hermitian rank-k arguments unchanged\n");
    return 0;
}

static int check_cherk_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_cherk_fortran_slot_fn thunk = NULL;
    int order = FB_LAYOUT_COL_MAJOR;
    int uplo = FB_UPPER;
    int trans = FB_NO_TRANS;
    int n = 2;
    int k = 3;
    int lda = 2;
    int ldc = 2;
    float alpha = 2.25f;
    float beta = 1.5f;
    fb_complex_float_t a[4] = {
        make_cf32(5.0f, 1.0f), make_cf32(6.0f, 2.0f),
        make_cf32(7.0f, 3.0f), make_cf32(8.0f, 4.0f)
    };
    fb_complex_float_t c[4] = {
        make_cf32(0.0f, 0.0f), make_cf32(0.0f, 0.0f),
        make_cf32(0.0f, 0.0f), make_cf32(0.0f, 0.0f)
    };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cherk_cblas_call, 0, sizeof(g_cherk_cblas_call));

    vtable.ext_ops[FB_OP_CHERK][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_cherk_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CHERK);

    thunk = (fb_cherk_fortran_slot_fn)vtable.ext_ops[FB_OP_CHERK][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CHERK CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&order, &uplo, &trans, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
    if (g_cherk_cblas_call.called != 1 ||
        g_cherk_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_cherk_cblas_call.uplo != FB_UPPER ||
        g_cherk_cblas_call.trans != FB_NO_TRANS ||
        g_cherk_cblas_call.n != 2 ||
        g_cherk_cblas_call.k != 3 ||
        g_cherk_cblas_call.lda != 2 ||
        g_cherk_cblas_call.ldc != 2 ||
        g_cherk_cblas_call.alpha_value != 2.25f ||
        g_cherk_cblas_call.beta_value != 1.5f ||
        g_cherk_cblas_call.a != a ||
        g_cherk_cblas_call.c != c ||
        !cf32_eq(c[0], make_cf32(111.0f, 1.0f)) ||
        !cf32_eq(c[1], make_cf32(112.0f, 2.0f)) ||
        !cf32_eq(c[2], make_cf32(113.0f, 3.0f)) ||
        !cf32_eq(c[3], make_cf32(114.0f, 4.0f))) {
        fprintf(stderr, "[FAIL] CHERK CBLAS->Fortran thunk did not map Hermitian rank-k arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] CHERK CBLAS->Fortran thunk maps Hermitian rank-k arguments into the C entry\n");
    return 0;
}

static int check_zherk_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_zherk_cblas_fn thunk = NULL;
    fb_complex_double_t a[4] = {
        make_cf64(9.0, -1.0), make_cf64(10.0, -2.0),
        make_cf64(11.0, -3.0), make_cf64(12.0, -4.0)
    };
    fb_complex_double_t c[4] = {
        make_cf64(0.0, 0.0), make_cf64(0.0, 0.0),
        make_cf64(0.0, 0.0), make_cf64(0.0, 0.0)
    };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zherk_fortran_call, 0, sizeof(g_zherk_fortran_call));

    vtable.ext_ops[FB_OP_ZHERK][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_zherk_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_ZHERK);

    thunk = (fb_zherk_cblas_fn)vtable.ext_ops[FB_OP_ZHERK][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZHERK Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    thunk(FB_LAYOUT_ROW_MAJOR, FB_UPPER, FB_CONJ_TRANS, 2, 3, 3.25, a, 3, 2.5, c, 2);
    if (g_zherk_fortran_call.called != 1 ||
        g_zherk_fortran_call.order != FB_LAYOUT_ROW_MAJOR ||
        g_zherk_fortran_call.uplo != FB_UPPER ||
        g_zherk_fortran_call.trans != FB_CONJ_TRANS ||
        g_zherk_fortran_call.n != 2 ||
        g_zherk_fortran_call.k != 3 ||
        g_zherk_fortran_call.lda != 3 ||
        g_zherk_fortran_call.ldc != 2 ||
        g_zherk_fortran_call.alpha_value != 3.25 ||
        g_zherk_fortran_call.beta_value != 2.5 ||
        g_zherk_fortran_call.a != a ||
        g_zherk_fortran_call.c != c ||
        !cf64_eq(c[0], make_cf64(121.0, -1.0)) ||
        !cf64_eq(c[1], make_cf64(122.0, -2.0)) ||
        !cf64_eq(c[2], make_cf64(123.0, -3.0)) ||
        !cf64_eq(c[3], make_cf64(124.0, -4.0))) {
        fprintf(stderr, "[FAIL] ZHERK Fortran->CBLAS thunk did not forward double-complex Hermitian rank-k arguments correctly\n");
        return 1;
    }

    printf("[PASS] ZHERK Fortran->CBLAS thunk forwards double-complex Hermitian rank-k arguments unchanged\n");
    return 0;
}

static int check_zherk_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_zherk_fortran_slot_fn thunk = NULL;
    int order = FB_LAYOUT_COL_MAJOR;
    int uplo = FB_LOWER;
    int trans = FB_NO_TRANS;
    int n = 2;
    int k = 3;
    int lda = 2;
    int ldc = 2;
    double alpha = 4.25;
    double beta = 3.5;
    fb_complex_double_t a[4] = {
        make_cf64(13.0, 1.0), make_cf64(14.0, 2.0),
        make_cf64(15.0, 3.0), make_cf64(16.0, 4.0)
    };
    fb_complex_double_t c[4] = {
        make_cf64(0.0, 0.0), make_cf64(0.0, 0.0),
        make_cf64(0.0, 0.0), make_cf64(0.0, 0.0)
    };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zherk_cblas_call, 0, sizeof(g_zherk_cblas_call));

    vtable.ext_ops[FB_OP_ZHERK][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_zherk_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_ZHERK);

    thunk = (fb_zherk_fortran_slot_fn)vtable.ext_ops[FB_OP_ZHERK][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZHERK CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&order, &uplo, &trans, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
    if (g_zherk_cblas_call.called != 1 ||
        g_zherk_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_zherk_cblas_call.uplo != FB_LOWER ||
        g_zherk_cblas_call.trans != FB_NO_TRANS ||
        g_zherk_cblas_call.n != 2 ||
        g_zherk_cblas_call.k != 3 ||
        g_zherk_cblas_call.lda != 2 ||
        g_zherk_cblas_call.ldc != 2 ||
        g_zherk_cblas_call.alpha_value != 4.25 ||
        g_zherk_cblas_call.beta_value != 3.5 ||
        g_zherk_cblas_call.a != a ||
        g_zherk_cblas_call.c != c ||
        !cf64_eq(c[0], make_cf64(131.0, 1.0)) ||
        !cf64_eq(c[1], make_cf64(132.0, 2.0)) ||
        !cf64_eq(c[2], make_cf64(133.0, 3.0)) ||
        !cf64_eq(c[3], make_cf64(134.0, 4.0))) {
        fprintf(stderr, "[FAIL] ZHERK CBLAS->Fortran thunk did not map double-complex Hermitian rank-k arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] ZHERK CBLAS->Fortran thunk maps double-complex Hermitian rank-k arguments into the C entry\n");
    return 0;
}

int main(void)
{
    int status = 0;

    status |= check_cherk_fortran_to_cblas();
    status |= check_cherk_cblas_to_fortran();
    status |= check_zherk_fortran_to_cblas();
    status |= check_zherk_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}