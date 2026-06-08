#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef void (*fb_cher2_cblas_fn)(fb_layout_t layout, fb_uplo_t uplo, int n,
                                  fb_complex_float_t alpha,
                                  const fb_complex_float_t *x, int incx,
                                  const fb_complex_float_t *y, int incy,
                                  fb_complex_float_t *a, int lda);
typedef void (*fb_zher2_cblas_fn)(fb_layout_t layout, fb_uplo_t uplo, int n,
                                  fb_complex_double_t alpha,
                                  const fb_complex_double_t *x, int incx,
                                  const fb_complex_double_t *y, int incy,
                                  fb_complex_double_t *a, int lda);

typedef void (*fb_cher2_fortran_slot_fn)(const int *order, const int *uplo,
                                         const int *n,
                                         const fb_complex_float_t *alpha,
                                         const fb_complex_float_t *x,
                                         const int *incx,
                                         const fb_complex_float_t *y,
                                         const int *incy,
                                         fb_complex_float_t *a,
                                         const int *lda);
typedef void (*fb_zher2_fortran_slot_fn)(const int *order, const int *uplo,
                                         const int *n,
                                         const fb_complex_double_t *alpha,
                                         const fb_complex_double_t *x,
                                         const int *incx,
                                         const fb_complex_double_t *y,
                                         const int *incy,
                                         fb_complex_double_t *a,
                                         const int *lda);

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
    int incx;
    int incy;
    int lda;
    fb_complex_float_t alpha_value;
    const fb_complex_float_t *x;
    const fb_complex_float_t *y;
    fb_complex_float_t *a;
} g_cher2_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    fb_uplo_t uplo;
    int n;
    int incx;
    int incy;
    int lda;
    fb_complex_float_t alpha_value;
    const fb_complex_float_t *x;
    const fb_complex_float_t *y;
    fb_complex_float_t *a;
} g_cher2_cblas_call;

static struct {
    int called;
    int order;
    int uplo;
    int n;
    int incx;
    int incy;
    int lda;
    fb_complex_double_t alpha_value;
    const fb_complex_double_t *x;
    const fb_complex_double_t *y;
    fb_complex_double_t *a;
} g_zher2_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    fb_uplo_t uplo;
    int n;
    int incx;
    int incy;
    int lda;
    fb_complex_double_t alpha_value;
    const fb_complex_double_t *x;
    const fb_complex_double_t *y;
    fb_complex_double_t *a;
} g_zher2_cblas_call;

static void stub_cher2_fortran(const int *order, const int *uplo, const int *n,
                               const fb_complex_float_t *alpha,
                               const fb_complex_float_t *x, const int *incx,
                               const fb_complex_float_t *y, const int *incy,
                               fb_complex_float_t *a, const int *lda)
{
    g_cher2_fortran_call.called += 1;
    g_cher2_fortran_call.order = *order;
    g_cher2_fortran_call.uplo = *uplo;
    g_cher2_fortran_call.n = *n;
    g_cher2_fortran_call.incx = *incx;
    g_cher2_fortran_call.incy = *incy;
    g_cher2_fortran_call.lda = *lda;
    g_cher2_fortran_call.alpha_value = *alpha;
    g_cher2_fortran_call.x = x;
    g_cher2_fortran_call.y = y;
    g_cher2_fortran_call.a = a;

    a[0] = make_cf32(61.0f, -1.0f);
    a[1] = make_cf32(62.0f, -2.0f);
    a[2] = make_cf32(63.0f, -3.0f);
    a[3] = make_cf32(64.0f, -4.0f);
}

static void stub_cher2_cblas(fb_layout_t layout, fb_uplo_t uplo, int n,
                             fb_complex_float_t alpha,
                             const fb_complex_float_t *x, int incx,
                             const fb_complex_float_t *y, int incy,
                             fb_complex_float_t *a, int lda)
{
    g_cher2_cblas_call.called += 1;
    g_cher2_cblas_call.layout = layout;
    g_cher2_cblas_call.uplo = uplo;
    g_cher2_cblas_call.n = n;
    g_cher2_cblas_call.incx = incx;
    g_cher2_cblas_call.incy = incy;
    g_cher2_cblas_call.lda = lda;
    g_cher2_cblas_call.alpha_value = alpha;
    g_cher2_cblas_call.x = x;
    g_cher2_cblas_call.y = y;
    g_cher2_cblas_call.a = a;

    a[0] = make_cf32(71.0f, 1.0f);
    a[1] = make_cf32(72.0f, 2.0f);
    a[2] = make_cf32(73.0f, 3.0f);
    a[3] = make_cf32(74.0f, 4.0f);
}

static void stub_zher2_fortran(const int *order, const int *uplo, const int *n,
                               const fb_complex_double_t *alpha,
                               const fb_complex_double_t *x, const int *incx,
                               const fb_complex_double_t *y, const int *incy,
                               fb_complex_double_t *a, const int *lda)
{
    g_zher2_fortran_call.called += 1;
    g_zher2_fortran_call.order = *order;
    g_zher2_fortran_call.uplo = *uplo;
    g_zher2_fortran_call.n = *n;
    g_zher2_fortran_call.incx = *incx;
    g_zher2_fortran_call.incy = *incy;
    g_zher2_fortran_call.lda = *lda;
    g_zher2_fortran_call.alpha_value = *alpha;
    g_zher2_fortran_call.x = x;
    g_zher2_fortran_call.y = y;
    g_zher2_fortran_call.a = a;

    a[0] = make_cf64(81.0, -1.0);
    a[1] = make_cf64(82.0, -2.0);
    a[2] = make_cf64(83.0, -3.0);
    a[3] = make_cf64(84.0, -4.0);
}

static void stub_zher2_cblas(fb_layout_t layout, fb_uplo_t uplo, int n,
                             fb_complex_double_t alpha,
                             const fb_complex_double_t *x, int incx,
                             const fb_complex_double_t *y, int incy,
                             fb_complex_double_t *a, int lda)
{
    g_zher2_cblas_call.called += 1;
    g_zher2_cblas_call.layout = layout;
    g_zher2_cblas_call.uplo = uplo;
    g_zher2_cblas_call.n = n;
    g_zher2_cblas_call.incx = incx;
    g_zher2_cblas_call.incy = incy;
    g_zher2_cblas_call.lda = lda;
    g_zher2_cblas_call.alpha_value = alpha;
    g_zher2_cblas_call.x = x;
    g_zher2_cblas_call.y = y;
    g_zher2_cblas_call.a = a;

    a[0] = make_cf64(91.0, 1.0);
    a[1] = make_cf64(92.0, 2.0);
    a[2] = make_cf64(93.0, 3.0);
    a[3] = make_cf64(94.0, 4.0);
}

static int check_cher2_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_cher2_cblas_fn thunk = NULL;
    fb_complex_float_t alpha = make_cf32(1.5f, -0.25f);
    fb_complex_float_t x[2] = { make_cf32(1.0f, -1.0f), make_cf32(2.0f, -2.0f) };
    fb_complex_float_t y[2] = { make_cf32(3.0f, 1.0f), make_cf32(4.0f, 2.0f) };
    fb_complex_float_t a[4] = { make_cf32(0.0f, 0.0f), make_cf32(0.0f, 0.0f), make_cf32(0.0f, 0.0f), make_cf32(0.0f, 0.0f) };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cher2_fortran_call, 0, sizeof(g_cher2_fortran_call));

    vtable.ext_ops[FB_OP_CHER2][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_cher2_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CHER2);

    thunk = (fb_cher2_cblas_fn)vtable.ext_ops[FB_OP_CHER2][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CHER2 Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    thunk(FB_LAYOUT_ROW_MAJOR, FB_LOWER, 2, alpha, x, 1, y, 1, a, 2);
    if (g_cher2_fortran_call.called != 1 ||
        g_cher2_fortran_call.order != FB_LAYOUT_ROW_MAJOR ||
        g_cher2_fortran_call.uplo != FB_LOWER ||
        g_cher2_fortran_call.n != 2 ||
        g_cher2_fortran_call.incx != 1 ||
        g_cher2_fortran_call.incy != 1 ||
        g_cher2_fortran_call.lda != 2 ||
        !cf32_eq(g_cher2_fortran_call.alpha_value, alpha) ||
        g_cher2_fortran_call.x != x ||
        g_cher2_fortran_call.y != y ||
        g_cher2_fortran_call.a != a ||
        !cf32_eq(a[0], make_cf32(61.0f, -1.0f)) ||
        !cf32_eq(a[1], make_cf32(62.0f, -2.0f)) ||
        !cf32_eq(a[2], make_cf32(63.0f, -3.0f)) ||
        !cf32_eq(a[3], make_cf32(64.0f, -4.0f))) {
        fprintf(stderr, "[FAIL] CHER2 Fortran->CBLAS thunk did not forward Hermitian rank-2 arguments correctly\n");
        return 1;
    }

    printf("[PASS] CHER2 Fortran->CBLAS thunk forwards Hermitian rank-2 arguments unchanged\n");
    return 0;
}

static int check_cher2_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_cher2_fortran_slot_fn thunk = NULL;
    int order = FB_LAYOUT_COL_MAJOR;
    int uplo = FB_UPPER;
    int n = 2;
    int incx = 1;
    int incy = 1;
    int lda = 2;
    fb_complex_float_t alpha = make_cf32(2.5f, -1.25f);
    fb_complex_float_t x[2] = { make_cf32(5.0f, 1.0f), make_cf32(6.0f, 2.0f) };
    fb_complex_float_t y[2] = { make_cf32(7.0f, -1.0f), make_cf32(8.0f, -2.0f) };
    fb_complex_float_t a[4] = { make_cf32(0.0f, 0.0f), make_cf32(0.0f, 0.0f), make_cf32(0.0f, 0.0f), make_cf32(0.0f, 0.0f) };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cher2_cblas_call, 0, sizeof(g_cher2_cblas_call));

    vtable.ext_ops[FB_OP_CHER2][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_cher2_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CHER2);

    thunk = (fb_cher2_fortran_slot_fn)vtable.ext_ops[FB_OP_CHER2][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CHER2 CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&order, &uplo, &n, &alpha, x, &incx, y, &incy, a, &lda);
    if (g_cher2_cblas_call.called != 1 ||
        g_cher2_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_cher2_cblas_call.uplo != FB_UPPER ||
        g_cher2_cblas_call.n != 2 ||
        g_cher2_cblas_call.incx != 1 ||
        g_cher2_cblas_call.incy != 1 ||
        g_cher2_cblas_call.lda != 2 ||
        !cf32_eq(g_cher2_cblas_call.alpha_value, alpha) ||
        g_cher2_cblas_call.x != x ||
        g_cher2_cblas_call.y != y ||
        g_cher2_cblas_call.a != a ||
        !cf32_eq(a[0], make_cf32(71.0f, 1.0f)) ||
        !cf32_eq(a[1], make_cf32(72.0f, 2.0f)) ||
        !cf32_eq(a[2], make_cf32(73.0f, 3.0f)) ||
        !cf32_eq(a[3], make_cf32(74.0f, 4.0f))) {
        fprintf(stderr, "[FAIL] CHER2 CBLAS->Fortran thunk did not map Hermitian rank-2 arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] CHER2 CBLAS->Fortran thunk maps Hermitian rank-2 arguments into the C entry\n");
    return 0;
}

static int check_zher2_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_zher2_cblas_fn thunk = NULL;
    fb_complex_double_t alpha = make_cf64(3.5, -0.5);
    fb_complex_double_t x[2] = { make_cf64(9.0, -1.0), make_cf64(10.0, -2.0) };
    fb_complex_double_t y[2] = { make_cf64(11.0, 1.0), make_cf64(12.0, 2.0) };
    fb_complex_double_t a[4] = { make_cf64(0.0, 0.0), make_cf64(0.0, 0.0), make_cf64(0.0, 0.0), make_cf64(0.0, 0.0) };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zher2_fortran_call, 0, sizeof(g_zher2_fortran_call));

    vtable.ext_ops[FB_OP_ZHER2][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_zher2_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_ZHER2);

    thunk = (fb_zher2_cblas_fn)vtable.ext_ops[FB_OP_ZHER2][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZHER2 Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    thunk(FB_LAYOUT_ROW_MAJOR, FB_UPPER, 2, alpha, x, 1, y, 1, a, 2);
    if (g_zher2_fortran_call.called != 1 ||
        g_zher2_fortran_call.order != FB_LAYOUT_ROW_MAJOR ||
        g_zher2_fortran_call.uplo != FB_UPPER ||
        g_zher2_fortran_call.n != 2 ||
        g_zher2_fortran_call.incx != 1 ||
        g_zher2_fortran_call.incy != 1 ||
        g_zher2_fortran_call.lda != 2 ||
        !cf64_eq(g_zher2_fortran_call.alpha_value, alpha) ||
        g_zher2_fortran_call.x != x ||
        g_zher2_fortran_call.y != y ||
        g_zher2_fortran_call.a != a ||
        !cf64_eq(a[0], make_cf64(81.0, -1.0)) ||
        !cf64_eq(a[1], make_cf64(82.0, -2.0)) ||
        !cf64_eq(a[2], make_cf64(83.0, -3.0)) ||
        !cf64_eq(a[3], make_cf64(84.0, -4.0))) {
        fprintf(stderr, "[FAIL] ZHER2 Fortran->CBLAS thunk did not forward double-complex Hermitian rank-2 arguments correctly\n");
        return 1;
    }

    printf("[PASS] ZHER2 Fortran->CBLAS thunk forwards double-complex Hermitian rank-2 arguments unchanged\n");
    return 0;
}

static int check_zher2_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_zher2_fortran_slot_fn thunk = NULL;
    int order = FB_LAYOUT_COL_MAJOR;
    int uplo = FB_LOWER;
    int n = 2;
    int incx = 1;
    int incy = 1;
    int lda = 2;
    fb_complex_double_t alpha = make_cf64(4.5, -1.5);
    fb_complex_double_t x[2] = { make_cf64(13.0, 1.0), make_cf64(14.0, 2.0) };
    fb_complex_double_t y[2] = { make_cf64(15.0, -1.0), make_cf64(16.0, -2.0) };
    fb_complex_double_t a[4] = { make_cf64(0.0, 0.0), make_cf64(0.0, 0.0), make_cf64(0.0, 0.0), make_cf64(0.0, 0.0) };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zher2_cblas_call, 0, sizeof(g_zher2_cblas_call));

    vtable.ext_ops[FB_OP_ZHER2][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_zher2_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_ZHER2);

    thunk = (fb_zher2_fortran_slot_fn)vtable.ext_ops[FB_OP_ZHER2][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZHER2 CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&order, &uplo, &n, &alpha, x, &incx, y, &incy, a, &lda);
    if (g_zher2_cblas_call.called != 1 ||
        g_zher2_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_zher2_cblas_call.uplo != FB_LOWER ||
        g_zher2_cblas_call.n != 2 ||
        g_zher2_cblas_call.incx != 1 ||
        g_zher2_cblas_call.incy != 1 ||
        g_zher2_cblas_call.lda != 2 ||
        !cf64_eq(g_zher2_cblas_call.alpha_value, alpha) ||
        g_zher2_cblas_call.x != x ||
        g_zher2_cblas_call.y != y ||
        g_zher2_cblas_call.a != a ||
        !cf64_eq(a[0], make_cf64(91.0, 1.0)) ||
        !cf64_eq(a[1], make_cf64(92.0, 2.0)) ||
        !cf64_eq(a[2], make_cf64(93.0, 3.0)) ||
        !cf64_eq(a[3], make_cf64(94.0, 4.0))) {
        fprintf(stderr, "[FAIL] ZHER2 CBLAS->Fortran thunk did not map double-complex Hermitian rank-2 arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] ZHER2 CBLAS->Fortran thunk maps double-complex Hermitian rank-2 arguments into the C entry\n");
    return 0;
}

int main(void)
{
    int status = 0;

    status |= check_cher2_fortran_to_cblas();
    status |= check_cher2_cblas_to_fortran();
    status |= check_zher2_fortran_to_cblas();
    status |= check_zher2_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}