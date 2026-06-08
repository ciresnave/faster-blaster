#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef void (*fb_csyr2_cblas_fn)(fb_layout_t layout, fb_uplo_t uplo, int n,
                                  const void *alpha, const void *x, int incx,
                                  const void *y, int incy, void *a, int lda);
typedef void (*fb_zsyr2_cblas_fn)(fb_layout_t layout, fb_uplo_t uplo, int n,
                                  const void *alpha, const void *x, int incx,
                                  const void *y, int incy, void *a, int lda);

typedef void (*fb_csyr2_fortran_slot_fn)(const int *order, const int *uplo,
                                         const int *n, const void *alpha,
                                         const void *x, const int *incx,
                                         const void *y, const int *incy,
                                         void *a, const int *lda);
typedef void (*fb_zsyr2_fortran_slot_fn)(const int *order, const int *uplo,
                                         const int *n, const void *alpha,
                                         const void *x, const int *incx,
                                         const void *y, const int *incy,
                                         void *a, const int *lda);

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
    const void *alpha;
    const void *x;
    const void *y;
    void *a;
    fb_complex_float_t alpha_value;
} g_csyr2_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    fb_uplo_t uplo;
    int n;
    int incx;
    int incy;
    int lda;
    const void *alpha;
    const void *x;
    const void *y;
    void *a;
    fb_complex_float_t alpha_value;
} g_csyr2_cblas_call;

static struct {
    int called;
    int order;
    int uplo;
    int n;
    int incx;
    int incy;
    int lda;
    const void *alpha;
    const void *x;
    const void *y;
    void *a;
    fb_complex_double_t alpha_value;
} g_zsyr2_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    fb_uplo_t uplo;
    int n;
    int incx;
    int incy;
    int lda;
    const void *alpha;
    const void *x;
    const void *y;
    void *a;
    fb_complex_double_t alpha_value;
} g_zsyr2_cblas_call;

static void stub_csyr2_fortran(const int *order, const int *uplo, const int *n,
                               const void *alpha, const void *x,
                               const int *incx, const void *y,
                               const int *incy, void *a, const int *lda)
{
    fb_complex_float_t *a_values = (fb_complex_float_t *)a;

    g_csyr2_fortran_call.called += 1;
    g_csyr2_fortran_call.order = *order;
    g_csyr2_fortran_call.uplo = *uplo;
    g_csyr2_fortran_call.n = *n;
    g_csyr2_fortran_call.incx = *incx;
    g_csyr2_fortran_call.incy = *incy;
    g_csyr2_fortran_call.lda = *lda;
    g_csyr2_fortran_call.alpha = alpha;
    g_csyr2_fortran_call.x = x;
    g_csyr2_fortran_call.y = y;
    g_csyr2_fortran_call.a = a;
    g_csyr2_fortran_call.alpha_value = *(const fb_complex_float_t *)alpha;

    a_values[0] = make_cf32(41.0f, -5.0f);
    a_values[1] = make_cf32(42.0f, -6.0f);
}

static void stub_csyr2_cblas(fb_layout_t layout, fb_uplo_t uplo, int n,
                             const void *alpha, const void *x, int incx,
                             const void *y, int incy, void *a, int lda)
{
    fb_complex_float_t *a_values = (fb_complex_float_t *)a;

    g_csyr2_cblas_call.called += 1;
    g_csyr2_cblas_call.layout = layout;
    g_csyr2_cblas_call.uplo = uplo;
    g_csyr2_cblas_call.n = n;
    g_csyr2_cblas_call.incx = incx;
    g_csyr2_cblas_call.incy = incy;
    g_csyr2_cblas_call.lda = lda;
    g_csyr2_cblas_call.alpha = alpha;
    g_csyr2_cblas_call.x = x;
    g_csyr2_cblas_call.y = y;
    g_csyr2_cblas_call.a = a;
    g_csyr2_cblas_call.alpha_value = *(const fb_complex_float_t *)alpha;

    a_values[0] = make_cf32(51.0f, 7.0f);
    a_values[1] = make_cf32(52.0f, 8.0f);
}

static void stub_zsyr2_fortran(const int *order, const int *uplo, const int *n,
                               const void *alpha, const void *x,
                               const int *incx, const void *y,
                               const int *incy, void *a, const int *lda)
{
    fb_complex_double_t *a_values = (fb_complex_double_t *)a;

    g_zsyr2_fortran_call.called += 1;
    g_zsyr2_fortran_call.order = *order;
    g_zsyr2_fortran_call.uplo = *uplo;
    g_zsyr2_fortran_call.n = *n;
    g_zsyr2_fortran_call.incx = *incx;
    g_zsyr2_fortran_call.incy = *incy;
    g_zsyr2_fortran_call.lda = *lda;
    g_zsyr2_fortran_call.alpha = alpha;
    g_zsyr2_fortran_call.x = x;
    g_zsyr2_fortran_call.y = y;
    g_zsyr2_fortran_call.a = a;
    g_zsyr2_fortran_call.alpha_value = *(const fb_complex_double_t *)alpha;

    a_values[0] = make_cf64(61.0, -9.0);
    a_values[1] = make_cf64(62.0, -10.0);
}

static void stub_zsyr2_cblas(fb_layout_t layout, fb_uplo_t uplo, int n,
                             const void *alpha, const void *x, int incx,
                             const void *y, int incy, void *a, int lda)
{
    fb_complex_double_t *a_values = (fb_complex_double_t *)a;

    g_zsyr2_cblas_call.called += 1;
    g_zsyr2_cblas_call.layout = layout;
    g_zsyr2_cblas_call.uplo = uplo;
    g_zsyr2_cblas_call.n = n;
    g_zsyr2_cblas_call.incx = incx;
    g_zsyr2_cblas_call.incy = incy;
    g_zsyr2_cblas_call.lda = lda;
    g_zsyr2_cblas_call.alpha = alpha;
    g_zsyr2_cblas_call.x = x;
    g_zsyr2_cblas_call.y = y;
    g_zsyr2_cblas_call.a = a;
    g_zsyr2_cblas_call.alpha_value = *(const fb_complex_double_t *)alpha;

    a_values[0] = make_cf64(71.0, 11.0);
    a_values[1] = make_cf64(72.0, 12.0);
}

static int check_csyr2_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_csyr2_cblas_fn thunk = NULL;
    fb_complex_float_t alpha = make_cf32(1.5f, -0.25f);
    fb_complex_float_t x[2] = { make_cf32(3.0f, -1.0f), make_cf32(4.0f, -2.0f) };
    fb_complex_float_t y[2] = { make_cf32(5.0f, 1.0f), make_cf32(6.0f, 2.0f) };
    fb_complex_float_t a[4] = {
        make_cf32(11.0f, 1.0f), make_cf32(12.0f, 2.0f),
        make_cf32(21.0f, 3.0f), make_cf32(22.0f, 4.0f)
    };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_csyr2_fortran_call, 0, sizeof(g_csyr2_fortran_call));

    vtable.ext_ops[FB_OP_CSYR2][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_csyr2_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CSYR2);

    thunk = (fb_csyr2_cblas_fn)vtable.ext_ops[FB_OP_CSYR2][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CSYR2 Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    thunk(FB_LAYOUT_ROW_MAJOR, FB_LOWER, 2, &alpha, x, 1, y, 1, a, 2);
    if (g_csyr2_fortran_call.called != 1 ||
        g_csyr2_fortran_call.order != FB_LAYOUT_ROW_MAJOR ||
        g_csyr2_fortran_call.uplo != FB_LOWER ||
        g_csyr2_fortran_call.n != 2 ||
        g_csyr2_fortran_call.incx != 1 ||
        g_csyr2_fortran_call.incy != 1 ||
        g_csyr2_fortran_call.lda != 2 ||
        g_csyr2_fortran_call.alpha != &alpha ||
        g_csyr2_fortran_call.x != x ||
        g_csyr2_fortran_call.y != y ||
        g_csyr2_fortran_call.a != a ||
        !cf32_eq(g_csyr2_fortran_call.alpha_value, alpha) ||
        !cf32_eq(a[0], make_cf32(41.0f, -5.0f)) ||
        !cf32_eq(a[1], make_cf32(42.0f, -6.0f))) {
        fprintf(stderr, "[FAIL] CSYR2 Fortran->CBLAS thunk did not forward pointer-based complex rank-2 arguments correctly\n");
        return 1;
    }

    printf("[PASS] CSYR2 Fortran->CBLAS thunk forwards pointer-based complex SYR2 arguments unchanged\n");
    return 0;
}

static int check_csyr2_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_csyr2_fortran_slot_fn thunk = NULL;
    int order = FB_LAYOUT_COL_MAJOR;
    int uplo = FB_UPPER;
    int n = 2;
    int incx = 1;
    int incy = 1;
    int lda = 2;
    fb_complex_float_t alpha = make_cf32(2.5f, -1.5f);
    fb_complex_float_t x[2] = { make_cf32(7.0f, 1.0f), make_cf32(8.0f, 2.0f) };
    fb_complex_float_t y[2] = { make_cf32(9.0f, 3.0f), make_cf32(10.0f, 4.0f) };
    fb_complex_float_t a[4] = {
        make_cf32(31.0f, 5.0f), make_cf32(32.0f, 6.0f),
        make_cf32(41.0f, 7.0f), make_cf32(42.0f, 8.0f)
    };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_csyr2_cblas_call, 0, sizeof(g_csyr2_cblas_call));

    vtable.ext_ops[FB_OP_CSYR2][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_csyr2_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CSYR2);

    thunk = (fb_csyr2_fortran_slot_fn)vtable.ext_ops[FB_OP_CSYR2][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CSYR2 CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&order, &uplo, &n, &alpha, x, &incx, y, &incy, a, &lda);
    if (g_csyr2_cblas_call.called != 1 ||
        g_csyr2_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_csyr2_cblas_call.uplo != FB_UPPER ||
        g_csyr2_cblas_call.n != 2 ||
        g_csyr2_cblas_call.incx != 1 ||
        g_csyr2_cblas_call.incy != 1 ||
        g_csyr2_cblas_call.lda != 2 ||
        g_csyr2_cblas_call.alpha != &alpha ||
        g_csyr2_cblas_call.x != x ||
        g_csyr2_cblas_call.y != y ||
        g_csyr2_cblas_call.a != a ||
        !cf32_eq(g_csyr2_cblas_call.alpha_value, alpha) ||
        !cf32_eq(a[0], make_cf32(51.0f, 7.0f)) ||
        !cf32_eq(a[1], make_cf32(52.0f, 8.0f))) {
        fprintf(stderr, "[FAIL] CSYR2 CBLAS->Fortran thunk did not map pointer-based complex rank-2 arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] CSYR2 CBLAS->Fortran thunk maps pointer-based complex SYR2 arguments into the C entry\n");
    return 0;
}

static int check_zsyr2_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_zsyr2_cblas_fn thunk = NULL;
    fb_complex_double_t alpha = make_cf64(3.5, -0.5);
    fb_complex_double_t x[2] = { make_cf64(11.0, -1.0), make_cf64(12.0, -2.0) };
    fb_complex_double_t y[2] = { make_cf64(13.0, 1.0), make_cf64(14.0, 2.0) };
    fb_complex_double_t a[4] = {
        make_cf64(111.0, 1.0), make_cf64(112.0, 2.0),
        make_cf64(121.0, 3.0), make_cf64(122.0, 4.0)
    };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zsyr2_fortran_call, 0, sizeof(g_zsyr2_fortran_call));

    vtable.ext_ops[FB_OP_ZSYR2][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_zsyr2_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_ZSYR2);

    thunk = (fb_zsyr2_cblas_fn)vtable.ext_ops[FB_OP_ZSYR2][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZSYR2 Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    thunk(FB_LAYOUT_ROW_MAJOR, FB_UPPER, 2, &alpha, x, 1, y, 1, a, 2);
    if (g_zsyr2_fortran_call.called != 1 ||
        g_zsyr2_fortran_call.order != FB_LAYOUT_ROW_MAJOR ||
        g_zsyr2_fortran_call.uplo != FB_UPPER ||
        g_zsyr2_fortran_call.n != 2 ||
        g_zsyr2_fortran_call.incx != 1 ||
        g_zsyr2_fortran_call.incy != 1 ||
        g_zsyr2_fortran_call.lda != 2 ||
        g_zsyr2_fortran_call.alpha != &alpha ||
        g_zsyr2_fortran_call.x != x ||
        g_zsyr2_fortran_call.y != y ||
        g_zsyr2_fortran_call.a != a ||
        !cf64_eq(g_zsyr2_fortran_call.alpha_value, alpha) ||
        !cf64_eq(a[0], make_cf64(61.0, -9.0)) ||
        !cf64_eq(a[1], make_cf64(62.0, -10.0))) {
        fprintf(stderr, "[FAIL] ZSYR2 Fortran->CBLAS thunk did not forward pointer-based double-complex rank-2 arguments correctly\n");
        return 1;
    }

    printf("[PASS] ZSYR2 Fortran->CBLAS thunk forwards pointer-based double-complex SYR2 arguments unchanged\n");
    return 0;
}

static int check_zsyr2_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_zsyr2_fortran_slot_fn thunk = NULL;
    int order = FB_LAYOUT_COL_MAJOR;
    int uplo = FB_LOWER;
    int n = 2;
    int incx = 1;
    int incy = 1;
    int lda = 2;
    fb_complex_double_t alpha = make_cf64(4.5, -2.5);
    fb_complex_double_t x[2] = { make_cf64(15.0, 3.0), make_cf64(16.0, 4.0) };
    fb_complex_double_t y[2] = { make_cf64(17.0, 5.0), make_cf64(18.0, 6.0) };
    fb_complex_double_t a[4] = {
        make_cf64(211.0, 5.0), make_cf64(212.0, 6.0),
        make_cf64(221.0, 7.0), make_cf64(222.0, 8.0)
    };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zsyr2_cblas_call, 0, sizeof(g_zsyr2_cblas_call));

    vtable.ext_ops[FB_OP_ZSYR2][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_zsyr2_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_ZSYR2);

    thunk = (fb_zsyr2_fortran_slot_fn)vtable.ext_ops[FB_OP_ZSYR2][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZSYR2 CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&order, &uplo, &n, &alpha, x, &incx, y, &incy, a, &lda);
    if (g_zsyr2_cblas_call.called != 1 ||
        g_zsyr2_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_zsyr2_cblas_call.uplo != FB_LOWER ||
        g_zsyr2_cblas_call.n != 2 ||
        g_zsyr2_cblas_call.incx != 1 ||
        g_zsyr2_cblas_call.incy != 1 ||
        g_zsyr2_cblas_call.lda != 2 ||
        g_zsyr2_cblas_call.alpha != &alpha ||
        g_zsyr2_cblas_call.x != x ||
        g_zsyr2_cblas_call.y != y ||
        g_zsyr2_cblas_call.a != a ||
        !cf64_eq(g_zsyr2_cblas_call.alpha_value, alpha) ||
        !cf64_eq(a[0], make_cf64(71.0, 11.0)) ||
        !cf64_eq(a[1], make_cf64(72.0, 12.0))) {
        fprintf(stderr, "[FAIL] ZSYR2 CBLAS->Fortran thunk did not map pointer-based double-complex rank-2 arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] ZSYR2 CBLAS->Fortran thunk maps pointer-based double-complex SYR2 arguments into the C entry\n");
    return 0;
}

int main(void)
{
    int status = 0;

    status |= check_csyr2_fortran_to_cblas();
    status |= check_csyr2_cblas_to_fortran();
    status |= check_zsyr2_fortran_to_cblas();
    status |= check_zsyr2_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}