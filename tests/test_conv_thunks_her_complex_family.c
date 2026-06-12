#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef void (*fb_cher_cblas_fn)(fb_layout_t layout, fb_uplo_t uplo, int n,
                                 float alpha, const fb_complex_float_t *x,
                                 int incx, fb_complex_float_t *a, int lda);
typedef void (*fb_zher_cblas_fn)(fb_layout_t layout, fb_uplo_t uplo, int n,
                                 double alpha, const fb_complex_double_t *x,
                                 int incx, fb_complex_double_t *a, int lda);

typedef void (*fb_cher_fortran_slot_fn)(const int *order, const int *uplo,
                                        const int *n, const float *alpha,
                                        const fb_complex_float_t *x,
                                        const int *incx, fb_complex_float_t *a,
                                        const int *lda);
typedef void (*fb_zher_fortran_slot_fn)(const int *order, const int *uplo,
                                        const int *n, const double *alpha,
                                        const fb_complex_double_t *x,
                                        const int *incx,
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
    int lda;
    float alpha_value;
    const fb_complex_float_t *x;
    fb_complex_float_t *a;
} g_cher_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    fb_uplo_t uplo;
    int n;
    int incx;
    int lda;
    float alpha_value;
    const fb_complex_float_t *x;
    fb_complex_float_t *a;
} g_cher_cblas_call;

static struct {
    int called;
    int order;
    int uplo;
    int n;
    int incx;
    int lda;
    double alpha_value;
    const fb_complex_double_t *x;
    fb_complex_double_t *a;
} g_zher_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    fb_uplo_t uplo;
    int n;
    int incx;
    int lda;
    double alpha_value;
    const fb_complex_double_t *x;
    fb_complex_double_t *a;
} g_zher_cblas_call;

static void stub_cher_fortran(const int *order, const int *uplo, const int *n,
                              const float *alpha,
                              const fb_complex_float_t *x, const int *incx,
                              fb_complex_float_t *a, const int *lda)
{
    g_cher_fortran_call.called += 1;
    g_cher_fortran_call.order = *order;
    g_cher_fortran_call.uplo = *uplo;
    g_cher_fortran_call.n = *n;
    g_cher_fortran_call.incx = *incx;
    g_cher_fortran_call.lda = *lda;
    g_cher_fortran_call.alpha_value = *alpha;
    g_cher_fortran_call.x = x;
    g_cher_fortran_call.a = a;

    a[0] = make_cf32(21.0f, -1.0f);
    a[1] = make_cf32(22.0f, -2.0f);
    a[2] = make_cf32(23.0f, -3.0f);
    a[3] = make_cf32(24.0f, -4.0f);
}

static void stub_cher_cblas(fb_layout_t layout, fb_uplo_t uplo, int n,
                            float alpha, const fb_complex_float_t *x, int incx,
                            fb_complex_float_t *a, int lda)
{
    g_cher_cblas_call.called += 1;
    g_cher_cblas_call.layout = layout;
    g_cher_cblas_call.uplo = uplo;
    g_cher_cblas_call.n = n;
    g_cher_cblas_call.incx = incx;
    g_cher_cblas_call.lda = lda;
    g_cher_cblas_call.alpha_value = alpha;
    g_cher_cblas_call.x = x;
    g_cher_cblas_call.a = a;

    a[0] = make_cf32(31.0f, 1.0f);
    a[1] = make_cf32(32.0f, 2.0f);
    a[2] = make_cf32(33.0f, 3.0f);
    a[3] = make_cf32(34.0f, 4.0f);
}

static void stub_zher_fortran(const int *order, const int *uplo, const int *n,
                              const double *alpha,
                              const fb_complex_double_t *x, const int *incx,
                              fb_complex_double_t *a, const int *lda)
{
    g_zher_fortran_call.called += 1;
    g_zher_fortran_call.order = *order;
    g_zher_fortran_call.uplo = *uplo;
    g_zher_fortran_call.n = *n;
    g_zher_fortran_call.incx = *incx;
    g_zher_fortran_call.lda = *lda;
    g_zher_fortran_call.alpha_value = *alpha;
    g_zher_fortran_call.x = x;
    g_zher_fortran_call.a = a;

    a[0] = make_cf64(41.0, -1.0);
    a[1] = make_cf64(42.0, -2.0);
    a[2] = make_cf64(43.0, -3.0);
    a[3] = make_cf64(44.0, -4.0);
}

static void stub_zher_cblas(fb_layout_t layout, fb_uplo_t uplo, int n,
                            double alpha, const fb_complex_double_t *x,
                            int incx, fb_complex_double_t *a, int lda)
{
    g_zher_cblas_call.called += 1;
    g_zher_cblas_call.layout = layout;
    g_zher_cblas_call.uplo = uplo;
    g_zher_cblas_call.n = n;
    g_zher_cblas_call.incx = incx;
    g_zher_cblas_call.lda = lda;
    g_zher_cblas_call.alpha_value = alpha;
    g_zher_cblas_call.x = x;
    g_zher_cblas_call.a = a;

    a[0] = make_cf64(51.0, 1.0);
    a[1] = make_cf64(52.0, 2.0);
    a[2] = make_cf64(53.0, 3.0);
    a[3] = make_cf64(54.0, 4.0);
}

static int check_cher_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_cher_cblas_fn thunk = NULL;
    fb_complex_float_t x[2] = { make_cf32(1.0f, -1.0f), make_cf32(2.0f, -2.0f) };
    fb_complex_float_t a[4] = { make_cf32(0.0f, 0.0f), make_cf32(0.0f, 0.0f), make_cf32(0.0f, 0.0f), make_cf32(0.0f, 0.0f) };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cher_fortran_call, 0, sizeof(g_cher_fortran_call));

    vtable.ext_ops[FB_OP_CHER][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_cher_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CHER);

    thunk = (fb_cher_cblas_fn)vtable.ext_ops[FB_OP_CHER][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CHER Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    thunk(FB_LAYOUT_ROW_MAJOR, FB_LOWER, 2, 1.5f, x, 1, a, 2);
    if (g_cher_fortran_call.called != 1 ||
        g_cher_fortran_call.order != FB_LAYOUT_ROW_MAJOR ||
        g_cher_fortran_call.uplo != FB_LOWER ||
        g_cher_fortran_call.n != 2 ||
        g_cher_fortran_call.incx != 1 ||
        g_cher_fortran_call.lda != 2 ||
        g_cher_fortran_call.alpha_value != 1.5f ||
        g_cher_fortran_call.x != x ||
        g_cher_fortran_call.a != a ||
        !cf32_eq(a[0], make_cf32(21.0f, -1.0f)) ||
        !cf32_eq(a[1], make_cf32(22.0f, -2.0f)) ||
        !cf32_eq(a[2], make_cf32(23.0f, -3.0f)) ||
        !cf32_eq(a[3], make_cf32(24.0f, -4.0f))) {
        fprintf(stderr, "[FAIL] CHER Fortran->CBLAS thunk did not forward Hermitian rank-1 arguments correctly\n");
        return 1;
    }

    printf("[PASS] CHER Fortran->CBLAS thunk forwards Hermitian rank-1 arguments unchanged\n");
    return 0;
}

static int check_cher_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_cher_fortran_slot_fn thunk = NULL;
    int order = FB_LAYOUT_COL_MAJOR;
    int uplo = FB_UPPER;
    int n = 2;
    int incx = 1;
    int lda = 2;
    float alpha = 2.5f;
    fb_complex_float_t x[2] = { make_cf32(3.0f, 1.0f), make_cf32(4.0f, 2.0f) };
    fb_complex_float_t a[4] = { make_cf32(0.0f, 0.0f), make_cf32(0.0f, 0.0f), make_cf32(0.0f, 0.0f), make_cf32(0.0f, 0.0f) };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cher_cblas_call, 0, sizeof(g_cher_cblas_call));

    vtable.ext_ops[FB_OP_CHER][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_cher_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CHER);

    thunk = (fb_cher_fortran_slot_fn)vtable.ext_ops[FB_OP_CHER][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CHER CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&order, &uplo, &n, &alpha, x, &incx, a, &lda);
    if (g_cher_cblas_call.called != 1 ||
        g_cher_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_cher_cblas_call.uplo != FB_UPPER ||
        g_cher_cblas_call.n != 2 ||
        g_cher_cblas_call.incx != 1 ||
        g_cher_cblas_call.lda != 2 ||
        g_cher_cblas_call.alpha_value != 2.5f ||
        g_cher_cblas_call.x != x ||
        g_cher_cblas_call.a != a ||
        !cf32_eq(a[0], make_cf32(31.0f, 1.0f)) ||
        !cf32_eq(a[1], make_cf32(32.0f, 2.0f)) ||
        !cf32_eq(a[2], make_cf32(33.0f, 3.0f)) ||
        !cf32_eq(a[3], make_cf32(34.0f, 4.0f))) {
        fprintf(stderr, "[FAIL] CHER CBLAS->Fortran thunk did not map Hermitian rank-1 arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] CHER CBLAS->Fortran thunk maps Hermitian rank-1 arguments into the C entry\n");
    return 0;
}

static int check_zher_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_zher_cblas_fn thunk = NULL;
    fb_complex_double_t x[2] = { make_cf64(5.0, -1.0), make_cf64(6.0, -2.0) };
    fb_complex_double_t a[4] = { make_cf64(0.0, 0.0), make_cf64(0.0, 0.0), make_cf64(0.0, 0.0), make_cf64(0.0, 0.0) };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zher_fortran_call, 0, sizeof(g_zher_fortran_call));

    vtable.ext_ops[FB_OP_ZHER][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_zher_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_ZHER);

    thunk = (fb_zher_cblas_fn)vtable.ext_ops[FB_OP_ZHER][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZHER Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    thunk(FB_LAYOUT_ROW_MAJOR, FB_UPPER, 2, 3.5, x, 1, a, 2);
    if (g_zher_fortran_call.called != 1 ||
        g_zher_fortran_call.order != FB_LAYOUT_ROW_MAJOR ||
        g_zher_fortran_call.uplo != FB_UPPER ||
        g_zher_fortran_call.n != 2 ||
        g_zher_fortran_call.incx != 1 ||
        g_zher_fortran_call.lda != 2 ||
        g_zher_fortran_call.alpha_value != 3.5 ||
        g_zher_fortran_call.x != x ||
        g_zher_fortran_call.a != a ||
        !cf64_eq(a[0], make_cf64(41.0, -1.0)) ||
        !cf64_eq(a[1], make_cf64(42.0, -2.0)) ||
        !cf64_eq(a[2], make_cf64(43.0, -3.0)) ||
        !cf64_eq(a[3], make_cf64(44.0, -4.0))) {
        fprintf(stderr, "[FAIL] ZHER Fortran->CBLAS thunk did not forward double-complex Hermitian rank-1 arguments correctly\n");
        return 1;
    }

    printf("[PASS] ZHER Fortran->CBLAS thunk forwards double-complex Hermitian rank-1 arguments unchanged\n");
    return 0;
}

static int check_zher_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_zher_fortran_slot_fn thunk = NULL;
    int order = FB_LAYOUT_COL_MAJOR;
    int uplo = FB_LOWER;
    int n = 2;
    int incx = 1;
    int lda = 2;
    double alpha = 4.5;
    fb_complex_double_t x[2] = { make_cf64(7.0, 1.0), make_cf64(8.0, 2.0) };
    fb_complex_double_t a[4] = { make_cf64(0.0, 0.0), make_cf64(0.0, 0.0), make_cf64(0.0, 0.0), make_cf64(0.0, 0.0) };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zher_cblas_call, 0, sizeof(g_zher_cblas_call));

    vtable.ext_ops[FB_OP_ZHER][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_zher_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_ZHER);

    thunk = (fb_zher_fortran_slot_fn)vtable.ext_ops[FB_OP_ZHER][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZHER CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&order, &uplo, &n, &alpha, x, &incx, a, &lda);
    if (g_zher_cblas_call.called != 1 ||
        g_zher_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_zher_cblas_call.uplo != FB_LOWER ||
        g_zher_cblas_call.n != 2 ||
        g_zher_cblas_call.incx != 1 ||
        g_zher_cblas_call.lda != 2 ||
        g_zher_cblas_call.alpha_value != 4.5 ||
        g_zher_cblas_call.x != x ||
        g_zher_cblas_call.a != a ||
        !cf64_eq(a[0], make_cf64(51.0, 1.0)) ||
        !cf64_eq(a[1], make_cf64(52.0, 2.0)) ||
        !cf64_eq(a[2], make_cf64(53.0, 3.0)) ||
        !cf64_eq(a[3], make_cf64(54.0, 4.0))) {
        fprintf(stderr, "[FAIL] ZHER CBLAS->Fortran thunk did not map double-complex Hermitian rank-1 arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] ZHER CBLAS->Fortran thunk maps double-complex Hermitian rank-1 arguments into the C entry\n");
    return 0;
}

int main(void)
{
    int status = 0;

    status |= check_cher_fortran_to_cblas();
    status |= check_cher_cblas_to_fortran();
    status |= check_zher_fortran_to_cblas();
    status |= check_zher_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}