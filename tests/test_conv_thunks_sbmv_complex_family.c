#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef void (*fb_csbmv_cblas_fn)(fb_layout_t layout, fb_uplo_t uplo, int n,
                                  int k, fb_complex_float_t alpha,
                                  const fb_complex_float_t *a, int lda,
                                  const fb_complex_float_t *x, int incx,
                                  fb_complex_float_t beta,
                                  fb_complex_float_t *y, int incy);
typedef void (*fb_zsbmv_cblas_fn)(fb_layout_t layout, fb_uplo_t uplo, int n,
                                  int k, fb_complex_double_t alpha,
                                  const fb_complex_double_t *a, int lda,
                                  const fb_complex_double_t *x, int incx,
                                  fb_complex_double_t beta,
                                  fb_complex_double_t *y, int incy);

typedef void (*fb_csbmv_fortran_slot_fn)(const int *order, const int *uplo,
                                         const int *n, const int *k,
                                         const fb_complex_float_t *alpha,
                                         const fb_complex_float_t *a,
                                         const int *lda,
                                         const fb_complex_float_t *x,
                                         const int *incx,
                                         const fb_complex_float_t *beta,
                                         fb_complex_float_t *y,
                                         const int *incy);
typedef void (*fb_zsbmv_fortran_slot_fn)(const int *order, const int *uplo,
                                         const int *n, const int *k,
                                         const fb_complex_double_t *alpha,
                                         const fb_complex_double_t *a,
                                         const int *lda,
                                         const fb_complex_double_t *x,
                                         const int *incx,
                                         const fb_complex_double_t *beta,
                                         fb_complex_double_t *y,
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
    int k;
    int lda;
    int incx;
    int incy;
    fb_complex_float_t alpha_value;
    fb_complex_float_t beta_value;
    const fb_complex_float_t *a;
    const fb_complex_float_t *x;
    fb_complex_float_t *y;
} g_csbmv_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    fb_uplo_t uplo;
    int n;
    int k;
    int lda;
    int incx;
    int incy;
    fb_complex_float_t alpha_value;
    fb_complex_float_t beta_value;
    const fb_complex_float_t *a;
    const fb_complex_float_t *x;
    fb_complex_float_t *y;
} g_csbmv_cblas_call;

static struct {
    int called;
    int order;
    int uplo;
    int n;
    int k;
    int lda;
    int incx;
    int incy;
    fb_complex_double_t alpha_value;
    fb_complex_double_t beta_value;
    const fb_complex_double_t *a;
    const fb_complex_double_t *x;
    fb_complex_double_t *y;
} g_zsbmv_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    fb_uplo_t uplo;
    int n;
    int k;
    int lda;
    int incx;
    int incy;
    fb_complex_double_t alpha_value;
    fb_complex_double_t beta_value;
    const fb_complex_double_t *a;
    const fb_complex_double_t *x;
    fb_complex_double_t *y;
} g_zsbmv_cblas_call;

static void stub_csbmv_fortran(const int *order, const int *uplo, const int *n,
                               const int *k,
                               const fb_complex_float_t *alpha,
                               const fb_complex_float_t *a, const int *lda,
                               const fb_complex_float_t *x, const int *incx,
                               const fb_complex_float_t *beta,
                               fb_complex_float_t *y, const int *incy)
{
    g_csbmv_fortran_call.called += 1;
    g_csbmv_fortran_call.order = *order;
    g_csbmv_fortran_call.uplo = *uplo;
    g_csbmv_fortran_call.n = *n;
    g_csbmv_fortran_call.k = *k;
    g_csbmv_fortran_call.lda = *lda;
    g_csbmv_fortran_call.incx = *incx;
    g_csbmv_fortran_call.incy = *incy;
    g_csbmv_fortran_call.alpha_value = *alpha;
    g_csbmv_fortran_call.beta_value = *beta;
    g_csbmv_fortran_call.a = a;
    g_csbmv_fortran_call.x = x;
    g_csbmv_fortran_call.y = y;

    y[0] = make_cf32(81.0f, -13.0f);
    y[1] = make_cf32(82.0f, -14.0f);
}

static void stub_csbmv_cblas(fb_layout_t layout, fb_uplo_t uplo, int n, int k,
                             fb_complex_float_t alpha,
                             const fb_complex_float_t *a, int lda,
                             const fb_complex_float_t *x, int incx,
                             fb_complex_float_t beta,
                             fb_complex_float_t *y, int incy)
{
    g_csbmv_cblas_call.called += 1;
    g_csbmv_cblas_call.layout = layout;
    g_csbmv_cblas_call.uplo = uplo;
    g_csbmv_cblas_call.n = n;
    g_csbmv_cblas_call.k = k;
    g_csbmv_cblas_call.lda = lda;
    g_csbmv_cblas_call.incx = incx;
    g_csbmv_cblas_call.incy = incy;
    g_csbmv_cblas_call.alpha_value = alpha;
    g_csbmv_cblas_call.beta_value = beta;
    g_csbmv_cblas_call.a = a;
    g_csbmv_cblas_call.x = x;
    g_csbmv_cblas_call.y = y;

    y[0] = make_cf32(91.0f, 15.0f);
    y[1] = make_cf32(92.0f, 16.0f);
}

static void stub_zsbmv_fortran(const int *order, const int *uplo, const int *n,
                               const int *k,
                               const fb_complex_double_t *alpha,
                               const fb_complex_double_t *a, const int *lda,
                               const fb_complex_double_t *x, const int *incx,
                               const fb_complex_double_t *beta,
                               fb_complex_double_t *y, const int *incy)
{
    g_zsbmv_fortran_call.called += 1;
    g_zsbmv_fortran_call.order = *order;
    g_zsbmv_fortran_call.uplo = *uplo;
    g_zsbmv_fortran_call.n = *n;
    g_zsbmv_fortran_call.k = *k;
    g_zsbmv_fortran_call.lda = *lda;
    g_zsbmv_fortran_call.incx = *incx;
    g_zsbmv_fortran_call.incy = *incy;
    g_zsbmv_fortran_call.alpha_value = *alpha;
    g_zsbmv_fortran_call.beta_value = *beta;
    g_zsbmv_fortran_call.a = a;
    g_zsbmv_fortran_call.x = x;
    g_zsbmv_fortran_call.y = y;

    y[0] = make_cf64(101.0, -17.0);
    y[1] = make_cf64(102.0, -18.0);
}

static void stub_zsbmv_cblas(fb_layout_t layout, fb_uplo_t uplo, int n, int k,
                             fb_complex_double_t alpha,
                             const fb_complex_double_t *a, int lda,
                             const fb_complex_double_t *x, int incx,
                             fb_complex_double_t beta,
                             fb_complex_double_t *y, int incy)
{
    g_zsbmv_cblas_call.called += 1;
    g_zsbmv_cblas_call.layout = layout;
    g_zsbmv_cblas_call.uplo = uplo;
    g_zsbmv_cblas_call.n = n;
    g_zsbmv_cblas_call.k = k;
    g_zsbmv_cblas_call.lda = lda;
    g_zsbmv_cblas_call.incx = incx;
    g_zsbmv_cblas_call.incy = incy;
    g_zsbmv_cblas_call.alpha_value = alpha;
    g_zsbmv_cblas_call.beta_value = beta;
    g_zsbmv_cblas_call.a = a;
    g_zsbmv_cblas_call.x = x;
    g_zsbmv_cblas_call.y = y;

    y[0] = make_cf64(111.0, 19.0);
    y[1] = make_cf64(112.0, 20.0);
}

static int check_csbmv_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_csbmv_cblas_fn thunk = NULL;
    fb_complex_float_t alpha = make_cf32(1.25f, -0.5f);
    fb_complex_float_t beta = make_cf32(-0.75f, 0.5f);
    fb_complex_float_t a[4] = {
        make_cf32(11.0f, 1.0f), make_cf32(12.0f, 2.0f),
        make_cf32(21.0f, 3.0f), make_cf32(22.0f, 4.0f)
    };
    fb_complex_float_t x[2] = { make_cf32(3.0f, -1.0f), make_cf32(4.0f, -2.0f) };
    fb_complex_float_t y[2] = { make_cf32(0.0f, 0.0f), make_cf32(0.0f, 0.0f) };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_csbmv_fortran_call, 0, sizeof(g_csbmv_fortran_call));

    vtable.ext_ops[FB_OP_CSBMV][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_csbmv_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CSBMV);

    thunk = (fb_csbmv_cblas_fn)vtable.ext_ops[FB_OP_CSBMV][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CSBMV Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    thunk(FB_LAYOUT_ROW_MAJOR, FB_LOWER, 2, 1, alpha, a, 2, x, 1, beta, y, 1);
    if (g_csbmv_fortran_call.called != 1 ||
        g_csbmv_fortran_call.order != FB_LAYOUT_ROW_MAJOR ||
        g_csbmv_fortran_call.uplo != FB_LOWER ||
        g_csbmv_fortran_call.n != 2 ||
        g_csbmv_fortran_call.k != 1 ||
        g_csbmv_fortran_call.lda != 2 ||
        g_csbmv_fortran_call.incx != 1 ||
        g_csbmv_fortran_call.incy != 1 ||
        !cf32_eq(g_csbmv_fortran_call.alpha_value, alpha) ||
        !cf32_eq(g_csbmv_fortran_call.beta_value, beta) ||
        g_csbmv_fortran_call.a != a ||
        g_csbmv_fortran_call.x != x ||
        g_csbmv_fortran_call.y != y ||
        !cf32_eq(y[0], make_cf32(81.0f, -13.0f)) ||
        !cf32_eq(y[1], make_cf32(82.0f, -14.0f))) {
        fprintf(stderr, "[FAIL] CSBMV Fortran->CBLAS thunk did not forward symmetric band arguments correctly\n");
        return 1;
    }

    printf("[PASS] CSBMV Fortran->CBLAS thunk forwards complex symmetric band arguments unchanged\n");
    return 0;
}

static int check_csbmv_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_csbmv_fortran_slot_fn thunk = NULL;
    int order = FB_LAYOUT_COL_MAJOR;
    int uplo = FB_UPPER;
    int n = 2;
    int k = 1;
    int lda = 2;
    int incx = 1;
    int incy = 1;
    fb_complex_float_t alpha = make_cf32(2.25f, -1.25f);
    fb_complex_float_t beta = make_cf32(-1.5f, 0.75f);
    fb_complex_float_t a[4] = {
        make_cf32(31.0f, 5.0f), make_cf32(32.0f, 6.0f),
        make_cf32(41.0f, 7.0f), make_cf32(42.0f, 8.0f)
    };
    fb_complex_float_t x[2] = { make_cf32(5.0f, 1.0f), make_cf32(6.0f, 2.0f) };
    fb_complex_float_t y[2] = { make_cf32(0.0f, 0.0f), make_cf32(0.0f, 0.0f) };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_csbmv_cblas_call, 0, sizeof(g_csbmv_cblas_call));

    vtable.ext_ops[FB_OP_CSBMV][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_csbmv_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CSBMV);

    thunk = (fb_csbmv_fortran_slot_fn)vtable.ext_ops[FB_OP_CSBMV][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CSBMV CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&order, &uplo, &n, &k, &alpha, a, &lda, x, &incx, &beta, y, &incy);
    if (g_csbmv_cblas_call.called != 1 ||
        g_csbmv_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_csbmv_cblas_call.uplo != FB_UPPER ||
        g_csbmv_cblas_call.n != 2 ||
        g_csbmv_cblas_call.k != 1 ||
        g_csbmv_cblas_call.lda != 2 ||
        g_csbmv_cblas_call.incx != 1 ||
        g_csbmv_cblas_call.incy != 1 ||
        !cf32_eq(g_csbmv_cblas_call.alpha_value, alpha) ||
        !cf32_eq(g_csbmv_cblas_call.beta_value, beta) ||
        g_csbmv_cblas_call.a != a ||
        g_csbmv_cblas_call.x != x ||
        g_csbmv_cblas_call.y != y ||
        !cf32_eq(y[0], make_cf32(91.0f, 15.0f)) ||
        !cf32_eq(y[1], make_cf32(92.0f, 16.0f))) {
        fprintf(stderr, "[FAIL] CSBMV CBLAS->Fortran thunk did not map symmetric band arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] CSBMV CBLAS->Fortran thunk maps complex symmetric band arguments into the C entry\n");
    return 0;
}

static int check_zsbmv_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_zsbmv_cblas_fn thunk = NULL;
    fb_complex_double_t alpha = make_cf64(3.25, -0.75);
    fb_complex_double_t beta = make_cf64(-2.5, 1.5);
    fb_complex_double_t a[4] = {
        make_cf64(111.0, 1.0), make_cf64(112.0, 2.0),
        make_cf64(121.0, 3.0), make_cf64(122.0, 4.0)
    };
    fb_complex_double_t x[2] = { make_cf64(7.0, -3.0), make_cf64(8.0, -4.0) };
    fb_complex_double_t y[2] = { make_cf64(0.0, 0.0), make_cf64(0.0, 0.0) };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zsbmv_fortran_call, 0, sizeof(g_zsbmv_fortran_call));

    vtable.ext_ops[FB_OP_ZSBMV][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_zsbmv_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_ZSBMV);

    thunk = (fb_zsbmv_cblas_fn)vtable.ext_ops[FB_OP_ZSBMV][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZSBMV Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    thunk(FB_LAYOUT_ROW_MAJOR, FB_UPPER, 2, 1, alpha, a, 2, x, 1, beta, y, 1);
    if (g_zsbmv_fortran_call.called != 1 ||
        g_zsbmv_fortran_call.order != FB_LAYOUT_ROW_MAJOR ||
        g_zsbmv_fortran_call.uplo != FB_UPPER ||
        g_zsbmv_fortran_call.n != 2 ||
        g_zsbmv_fortran_call.k != 1 ||
        g_zsbmv_fortran_call.lda != 2 ||
        g_zsbmv_fortran_call.incx != 1 ||
        g_zsbmv_fortran_call.incy != 1 ||
        !cf64_eq(g_zsbmv_fortran_call.alpha_value, alpha) ||
        !cf64_eq(g_zsbmv_fortran_call.beta_value, beta) ||
        g_zsbmv_fortran_call.a != a ||
        g_zsbmv_fortran_call.x != x ||
        g_zsbmv_fortran_call.y != y ||
        !cf64_eq(y[0], make_cf64(101.0, -17.0)) ||
        !cf64_eq(y[1], make_cf64(102.0, -18.0))) {
        fprintf(stderr, "[FAIL] ZSBMV Fortran->CBLAS thunk did not forward double-complex symmetric band arguments correctly\n");
        return 1;
    }

    printf("[PASS] ZSBMV Fortran->CBLAS thunk forwards double-complex symmetric band arguments unchanged\n");
    return 0;
}

static int check_zsbmv_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_zsbmv_fortran_slot_fn thunk = NULL;
    int order = FB_LAYOUT_COL_MAJOR;
    int uplo = FB_LOWER;
    int n = 2;
    int k = 1;
    int lda = 2;
    int incx = 1;
    int incy = 1;
    fb_complex_double_t alpha = make_cf64(4.25, -1.75);
    fb_complex_double_t beta = make_cf64(-3.5, 0.25);
    fb_complex_double_t a[4] = {
        make_cf64(211.0, 5.0), make_cf64(212.0, 6.0),
        make_cf64(221.0, 7.0), make_cf64(222.0, 8.0)
    };
    fb_complex_double_t x[2] = { make_cf64(9.0, 1.0), make_cf64(10.0, 2.0) };
    fb_complex_double_t y[2] = { make_cf64(0.0, 0.0), make_cf64(0.0, 0.0) };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zsbmv_cblas_call, 0, sizeof(g_zsbmv_cblas_call));

    vtable.ext_ops[FB_OP_ZSBMV][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_zsbmv_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_ZSBMV);

    thunk = (fb_zsbmv_fortran_slot_fn)vtable.ext_ops[FB_OP_ZSBMV][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZSBMV CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&order, &uplo, &n, &k, &alpha, a, &lda, x, &incx, &beta, y, &incy);
    if (g_zsbmv_cblas_call.called != 1 ||
        g_zsbmv_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_zsbmv_cblas_call.uplo != FB_LOWER ||
        g_zsbmv_cblas_call.n != 2 ||
        g_zsbmv_cblas_call.k != 1 ||
        g_zsbmv_cblas_call.lda != 2 ||
        g_zsbmv_cblas_call.incx != 1 ||
        g_zsbmv_cblas_call.incy != 1 ||
        !cf64_eq(g_zsbmv_cblas_call.alpha_value, alpha) ||
        !cf64_eq(g_zsbmv_cblas_call.beta_value, beta) ||
        g_zsbmv_cblas_call.a != a ||
        g_zsbmv_cblas_call.x != x ||
        g_zsbmv_cblas_call.y != y ||
        !cf64_eq(y[0], make_cf64(111.0, 19.0)) ||
        !cf64_eq(y[1], make_cf64(112.0, 20.0))) {
        fprintf(stderr, "[FAIL] ZSBMV CBLAS->Fortran thunk did not map double-complex symmetric band arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] ZSBMV CBLAS->Fortran thunk maps double-complex symmetric band arguments into the C entry\n");
    return 0;
}

int main(void)
{
    int status = 0;

    status |= check_csbmv_fortran_to_cblas();
    status |= check_csbmv_cblas_to_fortran();
    status |= check_zsbmv_fortran_to_cblas();
    status |= check_zsbmv_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}