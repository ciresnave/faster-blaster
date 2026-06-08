#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef void (*fb_csymv_cblas_fn)(fb_layout_t layout, fb_uplo_t uplo, int n,
                                  const void *alpha, const void *A, int lda,
                                  const void *x, int incx, const void *beta,
                                  void *y, int incy);
typedef void (*fb_zsymv_cblas_fn)(fb_layout_t layout, fb_uplo_t uplo, int n,
                                  const void *alpha, const void *A, int lda,
                                  const void *x, int incx, const void *beta,
                                  void *y, int incy);

typedef void (*fb_csymv_fortran_slot_fn)(const int *order, const int *uplo,
                                         const int *n, const void *alpha,
                                         const void *a, const int *lda,
                                         const void *x, const int *incx,
                                         const void *beta, void *y,
                                         const int *incy);
typedef void (*fb_zsymv_fortran_slot_fn)(const int *order, const int *uplo,
                                         const int *n, const void *alpha,
                                         const void *a, const int *lda,
                                         const void *x, const int *incx,
                                         const void *beta, void *y,
                                         const int *incy);

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
    int n;
    int lda;
    int incx;
    int incy;
    const void *alpha;
    const void *a;
    const void *x;
    const void *beta;
    void *y;
    fb_complex_float_t alpha_value;
    fb_complex_float_t beta_value;
} g_csymv_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    fb_uplo_t uplo;
    int n;
    int lda;
    int incx;
    int incy;
    const void *alpha;
    const void *a;
    const void *x;
    const void *beta;
    void *y;
    fb_complex_float_t alpha_value;
    fb_complex_float_t beta_value;
} g_csymv_cblas_call;

static struct {
    int called;
    int order;
    int uplo;
    int n;
    int lda;
    int incx;
    int incy;
    const void *alpha;
    const void *a;
    const void *x;
    const void *beta;
    void *y;
    fb_complex_double_t alpha_value;
    fb_complex_double_t beta_value;
} g_zsymv_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    fb_uplo_t uplo;
    int n;
    int lda;
    int incx;
    int incy;
    const void *alpha;
    const void *a;
    const void *x;
    const void *beta;
    void *y;
    fb_complex_double_t alpha_value;
    fb_complex_double_t beta_value;
} g_zsymv_cblas_call;

static void stub_csymv_fortran(const int *order, const int *uplo, const int *n,
                               const void *alpha, const void *a,
                               const int *lda, const void *x,
                               const int *incx, const void *beta, void *y,
                               const int *incy)
{
    fb_complex_float_t *y_values = (fb_complex_float_t *)y;

    g_csymv_fortran_call.called += 1;
    g_csymv_fortran_call.order = *order;
    g_csymv_fortran_call.uplo = *uplo;
    g_csymv_fortran_call.n = *n;
    g_csymv_fortran_call.lda = *lda;
    g_csymv_fortran_call.incx = *incx;
    g_csymv_fortran_call.incy = *incy;
    g_csymv_fortran_call.alpha = alpha;
    g_csymv_fortran_call.a = a;
    g_csymv_fortran_call.x = x;
    g_csymv_fortran_call.beta = beta;
    g_csymv_fortran_call.y = y;
    g_csymv_fortran_call.alpha_value = *(const fb_complex_float_t *)alpha;
    g_csymv_fortran_call.beta_value = *(const fb_complex_float_t *)beta;

    y_values[0] = make_cf32(41.0f, -5.0f);
    y_values[1] = make_cf32(42.0f, -6.0f);
}

static void stub_csymv_cblas(fb_layout_t layout, fb_uplo_t uplo, int n,
                             const void *alpha, const void *a, int lda,
                             const void *x, int incx, const void *beta,
                             void *y, int incy)
{
    fb_complex_float_t *y_values = (fb_complex_float_t *)y;

    g_csymv_cblas_call.called += 1;
    g_csymv_cblas_call.layout = layout;
    g_csymv_cblas_call.uplo = uplo;
    g_csymv_cblas_call.n = n;
    g_csymv_cblas_call.lda = lda;
    g_csymv_cblas_call.incx = incx;
    g_csymv_cblas_call.incy = incy;
    g_csymv_cblas_call.alpha = alpha;
    g_csymv_cblas_call.a = a;
    g_csymv_cblas_call.x = x;
    g_csymv_cblas_call.beta = beta;
    g_csymv_cblas_call.y = y;
    g_csymv_cblas_call.alpha_value = *(const fb_complex_float_t *)alpha;
    g_csymv_cblas_call.beta_value = *(const fb_complex_float_t *)beta;

    y_values[0] = make_cf32(51.0f, 7.0f);
    y_values[1] = make_cf32(52.0f, 8.0f);
}

static void stub_zsymv_fortran(const int *order, const int *uplo, const int *n,
                               const void *alpha, const void *a,
                               const int *lda, const void *x,
                               const int *incx, const void *beta, void *y,
                               const int *incy)
{
    fb_complex_double_t *y_values = (fb_complex_double_t *)y;

    g_zsymv_fortran_call.called += 1;
    g_zsymv_fortran_call.order = *order;
    g_zsymv_fortran_call.uplo = *uplo;
    g_zsymv_fortran_call.n = *n;
    g_zsymv_fortran_call.lda = *lda;
    g_zsymv_fortran_call.incx = *incx;
    g_zsymv_fortran_call.incy = *incy;
    g_zsymv_fortran_call.alpha = alpha;
    g_zsymv_fortran_call.a = a;
    g_zsymv_fortran_call.x = x;
    g_zsymv_fortran_call.beta = beta;
    g_zsymv_fortran_call.y = y;
    g_zsymv_fortran_call.alpha_value = *(const fb_complex_double_t *)alpha;
    g_zsymv_fortran_call.beta_value = *(const fb_complex_double_t *)beta;

    y_values[0] = make_cf64(61.0, -9.0);
    y_values[1] = make_cf64(62.0, -10.0);
}

static void stub_zsymv_cblas(fb_layout_t layout, fb_uplo_t uplo, int n,
                             const void *alpha, const void *a, int lda,
                             const void *x, int incx, const void *beta,
                             void *y, int incy)
{
    fb_complex_double_t *y_values = (fb_complex_double_t *)y;

    g_zsymv_cblas_call.called += 1;
    g_zsymv_cblas_call.layout = layout;
    g_zsymv_cblas_call.uplo = uplo;
    g_zsymv_cblas_call.n = n;
    g_zsymv_cblas_call.lda = lda;
    g_zsymv_cblas_call.incx = incx;
    g_zsymv_cblas_call.incy = incy;
    g_zsymv_cblas_call.alpha = alpha;
    g_zsymv_cblas_call.a = a;
    g_zsymv_cblas_call.x = x;
    g_zsymv_cblas_call.beta = beta;
    g_zsymv_cblas_call.y = y;
    g_zsymv_cblas_call.alpha_value = *(const fb_complex_double_t *)alpha;
    g_zsymv_cblas_call.beta_value = *(const fb_complex_double_t *)beta;

    y_values[0] = make_cf64(71.0, 11.0);
    y_values[1] = make_cf64(72.0, 12.0);
}

static int check_csymv_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_csymv_cblas_fn thunk = NULL;
    fb_complex_float_t alpha = make_cf32(1.5f, -0.25f);
    fb_complex_float_t beta = make_cf32(-0.5f, 0.75f);
    fb_complex_float_t a[4] = {
        make_cf32(11.0f, 1.0f), make_cf32(12.0f, 2.0f),
        make_cf32(21.0f, 3.0f), make_cf32(22.0f, 4.0f)
    };
    fb_complex_float_t x[2] = { make_cf32(3.0f, -1.0f), make_cf32(4.0f, -2.0f) };
    fb_complex_float_t y[2] = { make_cf32(0.0f, 0.0f), make_cf32(0.0f, 0.0f) };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_csymv_fortran_call, 0, sizeof(g_csymv_fortran_call));

    vtable.ext_ops[FB_OP_CSYMV][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_csymv_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CSYMV);

    thunk = (fb_csymv_cblas_fn)vtable.ext_ops[FB_OP_CSYMV][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CSYMV Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    thunk(FB_LAYOUT_ROW_MAJOR, FB_LOWER, 2, &alpha, a, 2, x, 1, &beta, y, 1);
    if (g_csymv_fortran_call.called != 1 ||
        g_csymv_fortran_call.order != FB_LAYOUT_ROW_MAJOR ||
        g_csymv_fortran_call.uplo != FB_LOWER ||
        g_csymv_fortran_call.n != 2 ||
        g_csymv_fortran_call.lda != 2 ||
        g_csymv_fortran_call.incx != 1 ||
        g_csymv_fortran_call.incy != 1 ||
        g_csymv_fortran_call.alpha != &alpha ||
        g_csymv_fortran_call.a != a ||
        g_csymv_fortran_call.x != x ||
        g_csymv_fortran_call.beta != &beta ||
        g_csymv_fortran_call.y != y ||
        !cf32_eq(g_csymv_fortran_call.alpha_value, alpha) ||
        !cf32_eq(g_csymv_fortran_call.beta_value, beta) ||
        !cf32_eq(y[0], make_cf32(41.0f, -5.0f)) ||
        !cf32_eq(y[1], make_cf32(42.0f, -6.0f))) {
        fprintf(stderr, "[FAIL] CSYMV Fortran->CBLAS thunk did not forward pointer-based complex arguments correctly\n");
        return 1;
    }

    printf("[PASS] CSYMV Fortran->CBLAS thunk forwards pointer-based complex SYMV arguments unchanged\n");
    return 0;
}

static int check_csymv_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_csymv_fortran_slot_fn thunk = NULL;
    int order = FB_LAYOUT_COL_MAJOR;
    int uplo = FB_UPPER;
    int n = 2;
    int lda = 2;
    int incx = 1;
    int incy = 1;
    fb_complex_float_t alpha = make_cf32(2.5f, -1.5f);
    fb_complex_float_t beta = make_cf32(-1.0f, 0.25f);
    fb_complex_float_t a[4] = {
        make_cf32(31.0f, 5.0f), make_cf32(32.0f, 6.0f),
        make_cf32(41.0f, 7.0f), make_cf32(42.0f, 8.0f)
    };
    fb_complex_float_t x[2] = { make_cf32(5.0f, 1.0f), make_cf32(6.0f, 2.0f) };
    fb_complex_float_t y[2] = { make_cf32(0.0f, 0.0f), make_cf32(0.0f, 0.0f) };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_csymv_cblas_call, 0, sizeof(g_csymv_cblas_call));

    vtable.ext_ops[FB_OP_CSYMV][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_csymv_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CSYMV);

    thunk = (fb_csymv_fortran_slot_fn)vtable.ext_ops[FB_OP_CSYMV][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CSYMV CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&order, &uplo, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
    if (g_csymv_cblas_call.called != 1 ||
        g_csymv_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_csymv_cblas_call.uplo != FB_UPPER ||
        g_csymv_cblas_call.n != 2 ||
        g_csymv_cblas_call.lda != 2 ||
        g_csymv_cblas_call.incx != 1 ||
        g_csymv_cblas_call.incy != 1 ||
        g_csymv_cblas_call.alpha != &alpha ||
        g_csymv_cblas_call.a != a ||
        g_csymv_cblas_call.x != x ||
        g_csymv_cblas_call.beta != &beta ||
        g_csymv_cblas_call.y != y ||
        !cf32_eq(g_csymv_cblas_call.alpha_value, alpha) ||
        !cf32_eq(g_csymv_cblas_call.beta_value, beta) ||
        !cf32_eq(y[0], make_cf32(51.0f, 7.0f)) ||
        !cf32_eq(y[1], make_cf32(52.0f, 8.0f))) {
        fprintf(stderr, "[FAIL] CSYMV CBLAS->Fortran thunk did not map pointer-based complex arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] CSYMV CBLAS->Fortran thunk maps pointer-based complex SYMV arguments into the C entry\n");
    return 0;
}

static int check_zsymv_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_zsymv_cblas_fn thunk = NULL;
    fb_complex_double_t alpha = make_cf64(3.5, -0.5);
    fb_complex_double_t beta = make_cf64(-2.0, 1.25);
    fb_complex_double_t a[4] = {
        make_cf64(111.0, 1.0), make_cf64(112.0, 2.0),
        make_cf64(121.0, 3.0), make_cf64(122.0, 4.0)
    };
    fb_complex_double_t x[2] = { make_cf64(7.0, -3.0), make_cf64(8.0, -4.0) };
    fb_complex_double_t y[2] = { make_cf64(0.0, 0.0), make_cf64(0.0, 0.0) };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zsymv_fortran_call, 0, sizeof(g_zsymv_fortran_call));

    vtable.ext_ops[FB_OP_ZSYMV][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_zsymv_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_ZSYMV);

    thunk = (fb_zsymv_cblas_fn)vtable.ext_ops[FB_OP_ZSYMV][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZSYMV Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    thunk(FB_LAYOUT_ROW_MAJOR, FB_UPPER, 2, &alpha, a, 2, x, 1, &beta, y, 1);
    if (g_zsymv_fortran_call.called != 1 ||
        g_zsymv_fortran_call.order != FB_LAYOUT_ROW_MAJOR ||
        g_zsymv_fortran_call.uplo != FB_UPPER ||
        g_zsymv_fortran_call.n != 2 ||
        g_zsymv_fortran_call.lda != 2 ||
        g_zsymv_fortran_call.incx != 1 ||
        g_zsymv_fortran_call.incy != 1 ||
        g_zsymv_fortran_call.alpha != &alpha ||
        g_zsymv_fortran_call.a != a ||
        g_zsymv_fortran_call.x != x ||
        g_zsymv_fortran_call.beta != &beta ||
        g_zsymv_fortran_call.y != y ||
        !cf64_eq(g_zsymv_fortran_call.alpha_value, alpha) ||
        !cf64_eq(g_zsymv_fortran_call.beta_value, beta) ||
        !cf64_eq(y[0], make_cf64(61.0, -9.0)) ||
        !cf64_eq(y[1], make_cf64(62.0, -10.0))) {
        fprintf(stderr, "[FAIL] ZSYMV Fortran->CBLAS thunk did not forward pointer-based double-complex arguments correctly\n");
        return 1;
    }

    printf("[PASS] ZSYMV Fortran->CBLAS thunk forwards pointer-based double-complex SYMV arguments unchanged\n");
    return 0;
}

static int check_zsymv_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_zsymv_fortran_slot_fn thunk = NULL;
    int order = FB_LAYOUT_COL_MAJOR;
    int uplo = FB_LOWER;
    int n = 2;
    int lda = 2;
    int incx = 1;
    int incy = 1;
    fb_complex_double_t alpha = make_cf64(4.5, -2.5);
    fb_complex_double_t beta = make_cf64(-3.0, 0.5);
    fb_complex_double_t a[4] = {
        make_cf64(211.0, 5.0), make_cf64(212.0, 6.0),
        make_cf64(221.0, 7.0), make_cf64(222.0, 8.0)
    };
    fb_complex_double_t x[2] = { make_cf64(9.0, 1.0), make_cf64(10.0, 2.0) };
    fb_complex_double_t y[2] = { make_cf64(0.0, 0.0), make_cf64(0.0, 0.0) };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zsymv_cblas_call, 0, sizeof(g_zsymv_cblas_call));

    vtable.ext_ops[FB_OP_ZSYMV][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_zsymv_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_ZSYMV);

    thunk = (fb_zsymv_fortran_slot_fn)vtable.ext_ops[FB_OP_ZSYMV][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZSYMV CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&order, &uplo, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
    if (g_zsymv_cblas_call.called != 1 ||
        g_zsymv_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_zsymv_cblas_call.uplo != FB_LOWER ||
        g_zsymv_cblas_call.n != 2 ||
        g_zsymv_cblas_call.lda != 2 ||
        g_zsymv_cblas_call.incx != 1 ||
        g_zsymv_cblas_call.incy != 1 ||
        g_zsymv_cblas_call.alpha != &alpha ||
        g_zsymv_cblas_call.a != a ||
        g_zsymv_cblas_call.x != x ||
        g_zsymv_cblas_call.beta != &beta ||
        g_zsymv_cblas_call.y != y ||
        !cf64_eq(g_zsymv_cblas_call.alpha_value, alpha) ||
        !cf64_eq(g_zsymv_cblas_call.beta_value, beta) ||
        !cf64_eq(y[0], make_cf64(71.0, 11.0)) ||
        !cf64_eq(y[1], make_cf64(72.0, 12.0))) {
        fprintf(stderr, "[FAIL] ZSYMV CBLAS->Fortran thunk did not map pointer-based double-complex arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] ZSYMV CBLAS->Fortran thunk maps pointer-based double-complex SYMV arguments into the C entry\n");
    return 0;
}

int main(void)
{
    int status = 0;

    status |= check_csymv_fortran_to_cblas();
    status |= check_csymv_cblas_to_fortran();
    status |= check_zsymv_fortran_to_cblas();
    status |= check_zsymv_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}