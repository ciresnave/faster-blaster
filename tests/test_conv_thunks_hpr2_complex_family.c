#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef void (*fb_chpr2_cblas_fn)(fb_layout_t layout, fb_uplo_t uplo, int n,
                                  fb_complex_float_t alpha,
                                  const fb_complex_float_t *x, int incx,
                                  const fb_complex_float_t *y, int incy,
                                  fb_complex_float_t *ap);
typedef void (*fb_zhpr2_cblas_fn)(fb_layout_t layout, fb_uplo_t uplo, int n,
                                  fb_complex_double_t alpha,
                                  const fb_complex_double_t *x, int incx,
                                  const fb_complex_double_t *y, int incy,
                                  fb_complex_double_t *ap);

typedef void (*fb_chpr2_fortran_slot_fn)(const int *order, const int *uplo,
                                         const int *n,
                                         const fb_complex_float_t *alpha,
                                         const fb_complex_float_t *x,
                                         const int *incx,
                                         const fb_complex_float_t *y,
                                         const int *incy,
                                         fb_complex_float_t *ap);
typedef void (*fb_zhpr2_fortran_slot_fn)(const int *order, const int *uplo,
                                         const int *n,
                                         const fb_complex_double_t *alpha,
                                         const fb_complex_double_t *x,
                                         const int *incx,
                                         const fb_complex_double_t *y,
                                         const int *incy,
                                         fb_complex_double_t *ap);

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
    fb_complex_float_t alpha_value;
    const fb_complex_float_t *x;
    const fb_complex_float_t *y;
    fb_complex_float_t *ap;
} g_chpr2_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    fb_uplo_t uplo;
    int n;
    int incx;
    int incy;
    fb_complex_float_t alpha_value;
    const fb_complex_float_t *x;
    const fb_complex_float_t *y;
    fb_complex_float_t *ap;
} g_chpr2_cblas_call;

static struct {
    int called;
    int order;
    int uplo;
    int n;
    int incx;
    int incy;
    fb_complex_double_t alpha_value;
    const fb_complex_double_t *x;
    const fb_complex_double_t *y;
    fb_complex_double_t *ap;
} g_zhpr2_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    fb_uplo_t uplo;
    int n;
    int incx;
    int incy;
    fb_complex_double_t alpha_value;
    const fb_complex_double_t *x;
    const fb_complex_double_t *y;
    fb_complex_double_t *ap;
} g_zhpr2_cblas_call;

static void stub_chpr2_fortran(const int *order, const int *uplo, const int *n,
                               const fb_complex_float_t *alpha,
                               const fb_complex_float_t *x, const int *incx,
                               const fb_complex_float_t *y, const int *incy,
                               fb_complex_float_t *ap)
{
    g_chpr2_fortran_call.called += 1;
    g_chpr2_fortran_call.order = *order;
    g_chpr2_fortran_call.uplo = *uplo;
    g_chpr2_fortran_call.n = *n;
    g_chpr2_fortran_call.incx = *incx;
    g_chpr2_fortran_call.incy = *incy;
    g_chpr2_fortran_call.alpha_value = *alpha;
    g_chpr2_fortran_call.x = x;
    g_chpr2_fortran_call.y = y;
    g_chpr2_fortran_call.ap = ap;

    ap[0] = make_cf32(61.0f, -1.0f);
    ap[1] = make_cf32(62.0f, -2.0f);
    ap[2] = make_cf32(63.0f, -3.0f);
}

static void stub_chpr2_cblas(fb_layout_t layout, fb_uplo_t uplo, int n,
                             fb_complex_float_t alpha,
                             const fb_complex_float_t *x, int incx,
                             const fb_complex_float_t *y, int incy,
                             fb_complex_float_t *ap)
{
    g_chpr2_cblas_call.called += 1;
    g_chpr2_cblas_call.layout = layout;
    g_chpr2_cblas_call.uplo = uplo;
    g_chpr2_cblas_call.n = n;
    g_chpr2_cblas_call.incx = incx;
    g_chpr2_cblas_call.incy = incy;
    g_chpr2_cblas_call.alpha_value = alpha;
    g_chpr2_cblas_call.x = x;
    g_chpr2_cblas_call.y = y;
    g_chpr2_cblas_call.ap = ap;

    ap[0] = make_cf32(71.0f, 1.0f);
    ap[1] = make_cf32(72.0f, 2.0f);
    ap[2] = make_cf32(73.0f, 3.0f);
}

static void stub_zhpr2_fortran(const int *order, const int *uplo, const int *n,
                               const fb_complex_double_t *alpha,
                               const fb_complex_double_t *x, const int *incx,
                               const fb_complex_double_t *y, const int *incy,
                               fb_complex_double_t *ap)
{
    g_zhpr2_fortran_call.called += 1;
    g_zhpr2_fortran_call.order = *order;
    g_zhpr2_fortran_call.uplo = *uplo;
    g_zhpr2_fortran_call.n = *n;
    g_zhpr2_fortran_call.incx = *incx;
    g_zhpr2_fortran_call.incy = *incy;
    g_zhpr2_fortran_call.alpha_value = *alpha;
    g_zhpr2_fortran_call.x = x;
    g_zhpr2_fortran_call.y = y;
    g_zhpr2_fortran_call.ap = ap;

    ap[0] = make_cf64(81.0, -4.0);
    ap[1] = make_cf64(82.0, -5.0);
    ap[2] = make_cf64(83.0, -6.0);
}

static void stub_zhpr2_cblas(fb_layout_t layout, fb_uplo_t uplo, int n,
                             fb_complex_double_t alpha,
                             const fb_complex_double_t *x, int incx,
                             const fb_complex_double_t *y, int incy,
                             fb_complex_double_t *ap)
{
    g_zhpr2_cblas_call.called += 1;
    g_zhpr2_cblas_call.layout = layout;
    g_zhpr2_cblas_call.uplo = uplo;
    g_zhpr2_cblas_call.n = n;
    g_zhpr2_cblas_call.incx = incx;
    g_zhpr2_cblas_call.incy = incy;
    g_zhpr2_cblas_call.alpha_value = alpha;
    g_zhpr2_cblas_call.x = x;
    g_zhpr2_cblas_call.y = y;
    g_zhpr2_cblas_call.ap = ap;

    ap[0] = make_cf64(91.0, 4.0);
    ap[1] = make_cf64(92.0, 5.0);
    ap[2] = make_cf64(93.0, 6.0);
}

static int check_chpr2_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_chpr2_cblas_fn thunk = NULL;
    fb_complex_float_t alpha = make_cf32(1.5f, -0.25f);
    fb_complex_float_t x[2] = { make_cf32(1.0f, -1.0f), make_cf32(2.0f, -2.0f) };
    fb_complex_float_t y[2] = { make_cf32(3.0f, 1.0f), make_cf32(4.0f, 2.0f) };
    fb_complex_float_t ap[3] = { make_cf32(0.0f, 0.0f), make_cf32(0.0f, 0.0f), make_cf32(0.0f, 0.0f) };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_chpr2_fortran_call, 0, sizeof(g_chpr2_fortran_call));

    vtable.ext_ops[FB_OP_CHPR2][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_chpr2_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CHPR2);

    thunk = (fb_chpr2_cblas_fn)vtable.ext_ops[FB_OP_CHPR2][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CHPR2 Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    thunk(FB_LAYOUT_ROW_MAJOR, FB_LOWER, 2, alpha, x, 1, y, 1, ap);
    if (g_chpr2_fortran_call.called != 1 ||
        g_chpr2_fortran_call.order != FB_LAYOUT_ROW_MAJOR ||
        g_chpr2_fortran_call.uplo != FB_LOWER ||
        g_chpr2_fortran_call.n != 2 ||
        g_chpr2_fortran_call.incx != 1 ||
        g_chpr2_fortran_call.incy != 1 ||
        !cf32_eq(g_chpr2_fortran_call.alpha_value, alpha) ||
        g_chpr2_fortran_call.x != x ||
        g_chpr2_fortran_call.y != y ||
        g_chpr2_fortran_call.ap != ap ||
        !cf32_eq(ap[0], make_cf32(61.0f, -1.0f)) ||
        !cf32_eq(ap[1], make_cf32(62.0f, -2.0f)) ||
        !cf32_eq(ap[2], make_cf32(63.0f, -3.0f))) {
        fprintf(stderr, "[FAIL] CHPR2 Fortran->CBLAS thunk did not forward packed Hermitian rank-2 arguments correctly\n");
        return 1;
    }

    printf("[PASS] CHPR2 Fortran->CBLAS thunk forwards packed Hermitian rank-2 arguments unchanged\n");
    return 0;
}

static int check_chpr2_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_chpr2_fortran_slot_fn thunk = NULL;
    int order = FB_LAYOUT_COL_MAJOR;
    int uplo = FB_UPPER;
    int n = 2;
    int incx = 1;
    int incy = 1;
    fb_complex_float_t alpha = make_cf32(2.5f, -1.25f);
    fb_complex_float_t x[2] = { make_cf32(5.0f, 1.0f), make_cf32(6.0f, 2.0f) };
    fb_complex_float_t y[2] = { make_cf32(7.0f, -1.0f), make_cf32(8.0f, -2.0f) };
    fb_complex_float_t ap[3] = { make_cf32(0.0f, 0.0f), make_cf32(0.0f, 0.0f), make_cf32(0.0f, 0.0f) };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_chpr2_cblas_call, 0, sizeof(g_chpr2_cblas_call));

    vtable.ext_ops[FB_OP_CHPR2][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_chpr2_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CHPR2);

    thunk = (fb_chpr2_fortran_slot_fn)vtable.ext_ops[FB_OP_CHPR2][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CHPR2 CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&order, &uplo, &n, &alpha, x, &incx, y, &incy, ap);
    if (g_chpr2_cblas_call.called != 1 ||
        g_chpr2_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_chpr2_cblas_call.uplo != FB_UPPER ||
        g_chpr2_cblas_call.n != 2 ||
        g_chpr2_cblas_call.incx != 1 ||
        g_chpr2_cblas_call.incy != 1 ||
        !cf32_eq(g_chpr2_cblas_call.alpha_value, alpha) ||
        g_chpr2_cblas_call.x != x ||
        g_chpr2_cblas_call.y != y ||
        g_chpr2_cblas_call.ap != ap ||
        !cf32_eq(ap[0], make_cf32(71.0f, 1.0f)) ||
        !cf32_eq(ap[1], make_cf32(72.0f, 2.0f)) ||
        !cf32_eq(ap[2], make_cf32(73.0f, 3.0f))) {
        fprintf(stderr, "[FAIL] CHPR2 CBLAS->Fortran thunk did not map packed Hermitian rank-2 arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] CHPR2 CBLAS->Fortran thunk maps packed Hermitian rank-2 arguments into the C entry\n");
    return 0;
}

static int check_zhpr2_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_zhpr2_cblas_fn thunk = NULL;
    fb_complex_double_t alpha = make_cf64(3.5, -0.5);
    fb_complex_double_t x[2] = { make_cf64(9.0, -1.0), make_cf64(10.0, -2.0) };
    fb_complex_double_t y[2] = { make_cf64(11.0, 1.0), make_cf64(12.0, 2.0) };
    fb_complex_double_t ap[3] = { make_cf64(0.0, 0.0), make_cf64(0.0, 0.0), make_cf64(0.0, 0.0) };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zhpr2_fortran_call, 0, sizeof(g_zhpr2_fortran_call));

    vtable.ext_ops[FB_OP_ZHPR2][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_zhpr2_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_ZHPR2);

    thunk = (fb_zhpr2_cblas_fn)vtable.ext_ops[FB_OP_ZHPR2][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZHPR2 Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    thunk(FB_LAYOUT_ROW_MAJOR, FB_UPPER, 2, alpha, x, 1, y, 1, ap);
    if (g_zhpr2_fortran_call.called != 1 ||
        g_zhpr2_fortran_call.order != FB_LAYOUT_ROW_MAJOR ||
        g_zhpr2_fortran_call.uplo != FB_UPPER ||
        g_zhpr2_fortran_call.n != 2 ||
        g_zhpr2_fortran_call.incx != 1 ||
        g_zhpr2_fortran_call.incy != 1 ||
        !cf64_eq(g_zhpr2_fortran_call.alpha_value, alpha) ||
        g_zhpr2_fortran_call.x != x ||
        g_zhpr2_fortran_call.y != y ||
        g_zhpr2_fortran_call.ap != ap ||
        !cf64_eq(ap[0], make_cf64(81.0, -4.0)) ||
        !cf64_eq(ap[1], make_cf64(82.0, -5.0)) ||
        !cf64_eq(ap[2], make_cf64(83.0, -6.0))) {
        fprintf(stderr, "[FAIL] ZHPR2 Fortran->CBLAS thunk did not forward packed double-complex Hermitian rank-2 arguments correctly\n");
        return 1;
    }

    printf("[PASS] ZHPR2 Fortran->CBLAS thunk forwards packed double-complex Hermitian rank-2 arguments unchanged\n");
    return 0;
}

static int check_zhpr2_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_zhpr2_fortran_slot_fn thunk = NULL;
    int order = FB_LAYOUT_COL_MAJOR;
    int uplo = FB_LOWER;
    int n = 2;
    int incx = 1;
    int incy = 1;
    fb_complex_double_t alpha = make_cf64(4.5, -1.5);
    fb_complex_double_t x[2] = { make_cf64(13.0, 1.0), make_cf64(14.0, 2.0) };
    fb_complex_double_t y[2] = { make_cf64(15.0, -1.0), make_cf64(16.0, -2.0) };
    fb_complex_double_t ap[3] = { make_cf64(0.0, 0.0), make_cf64(0.0, 0.0), make_cf64(0.0, 0.0) };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zhpr2_cblas_call, 0, sizeof(g_zhpr2_cblas_call));

    vtable.ext_ops[FB_OP_ZHPR2][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_zhpr2_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_ZHPR2);

    thunk = (fb_zhpr2_fortran_slot_fn)vtable.ext_ops[FB_OP_ZHPR2][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZHPR2 CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&order, &uplo, &n, &alpha, x, &incx, y, &incy, ap);
    if (g_zhpr2_cblas_call.called != 1 ||
        g_zhpr2_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_zhpr2_cblas_call.uplo != FB_LOWER ||
        g_zhpr2_cblas_call.n != 2 ||
        g_zhpr2_cblas_call.incx != 1 ||
        g_zhpr2_cblas_call.incy != 1 ||
        !cf64_eq(g_zhpr2_cblas_call.alpha_value, alpha) ||
        g_zhpr2_cblas_call.x != x ||
        g_zhpr2_cblas_call.y != y ||
        g_zhpr2_cblas_call.ap != ap ||
        !cf64_eq(ap[0], make_cf64(91.0, 4.0)) ||
        !cf64_eq(ap[1], make_cf64(92.0, 5.0)) ||
        !cf64_eq(ap[2], make_cf64(93.0, 6.0))) {
        fprintf(stderr, "[FAIL] ZHPR2 CBLAS->Fortran thunk did not map packed double-complex Hermitian rank-2 arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] ZHPR2 CBLAS->Fortran thunk maps packed double-complex Hermitian rank-2 arguments into the C entry\n");
    return 0;
}

int main(void)
{
    int status = 0;

    status |= check_chpr2_fortran_to_cblas();
    status |= check_chpr2_cblas_to_fortran();
    status |= check_zhpr2_fortran_to_cblas();
    status |= check_zhpr2_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}