#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef void (*fb_cgemv_cblas_fn)(fb_layout_t layout, fb_transpose_t trans,
                                  int m, int n, const void *alpha,
                                  const void *a, int lda, const void *x,
                                  int incx, const void *beta, void *y,
                                  int incy);
typedef void (*fb_zgemv_cblas_fn)(fb_layout_t layout, fb_transpose_t trans,
                                  int m, int n, const void *alpha,
                                  const void *a, int lda, const void *x,
                                  int incx, const void *beta, void *y,
                                  int incy);
typedef void (*fb_cgemv_fortran_slot_fn)(const int *order, const int *trans,
                                         const int *m, const int *n,
                                         const void *alpha, const void *a,
                                         const int *lda, const void *x,
                                         const int *incx, const void *beta,
                                         void *y, const int *incy);
typedef void (*fb_zgemv_fortran_slot_fn)(const int *order, const int *trans,
                                         const int *m, const int *n,
                                         const void *alpha, const void *a,
                                         const int *lda, const void *x,
                                         const int *incx, const void *beta,
                                         void *y, const int *incy);

typedef void (*fb_cger_cblas_fn)(fb_layout_t layout, int m, int n,
                                 const void *alpha, const void *x, int incx,
                                 const void *y, int incy, void *a, int lda);
typedef void (*fb_zger_cblas_fn)(fb_layout_t layout, int m, int n,
                                 const void *alpha, const void *x, int incx,
                                 const void *y, int incy, void *a, int lda);
typedef void (*fb_cger_fortran_slot_fn)(const int *order, const int *m,
                                        const int *n, const void *alpha,
                                        const void *x, const int *incx,
                                        const void *y, const int *incy,
                                        void *a, const int *lda);
typedef void (*fb_zger_fortran_slot_fn)(const int *order, const int *m,
                                        const int *n, const void *alpha,
                                        const void *x, const int *incx,
                                        const void *y, const int *incy,
                                        void *a, const int *lda);

typedef struct {
    int called;
    int order;
    int trans;
    int m;
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
} cgemv_call_f32_t;

typedef struct {
    int called;
    int order;
    int trans;
    int m;
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
} cgemv_call_f64_t;

typedef struct {
    int called;
    int order;
    int m;
    int n;
    int incx;
    int incy;
    int lda;
    const void *alpha;
    const void *x;
    const void *y;
    void *a;
    fb_complex_float_t alpha_value;
} cger_call_f32_t;

typedef struct {
    int called;
    int order;
    int m;
    int n;
    int incx;
    int incy;
    int lda;
    const void *alpha;
    const void *x;
    const void *y;
    void *a;
    fb_complex_double_t alpha_value;
} cger_call_f64_t;

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

static cgemv_call_f32_t g_cgemv_fortran_call;
static cgemv_call_f32_t g_cgemv_cblas_call;
static cgemv_call_f64_t g_zgemv_fortran_call;
static cgemv_call_f64_t g_zgemv_cblas_call;
static cger_call_f32_t g_cger_fortran_call;
static cger_call_f32_t g_cger_cblas_call;
static cger_call_f64_t g_zger_fortran_call;
static cger_call_f64_t g_zger_cblas_call;

static void stub_cgemv_fortran(const int *order, const int *trans,
                               const int *m, const int *n,
                               const void *alpha, const void *a,
                               const int *lda, const void *x,
                               const int *incx, const void *beta, void *y,
                               const int *incy)
{
    fb_complex_float_t *y_values = (fb_complex_float_t *)y;

    g_cgemv_fortran_call.called += 1;
    g_cgemv_fortran_call.order = *order;
    g_cgemv_fortran_call.trans = *trans;
    g_cgemv_fortran_call.m = *m;
    g_cgemv_fortran_call.n = *n;
    g_cgemv_fortran_call.lda = *lda;
    g_cgemv_fortran_call.incx = *incx;
    g_cgemv_fortran_call.incy = *incy;
    g_cgemv_fortran_call.alpha = alpha;
    g_cgemv_fortran_call.a = a;
    g_cgemv_fortran_call.x = x;
    g_cgemv_fortran_call.beta = beta;
    g_cgemv_fortran_call.y = y;
    g_cgemv_fortran_call.alpha_value = *(const fb_complex_float_t *)alpha;
    g_cgemv_fortran_call.beta_value = *(const fb_complex_float_t *)beta;

    y_values[0] = make_cf32(101.0f, -1.0f);
    y_values[1] = make_cf32(102.0f, -2.0f);
}

static void stub_cgemv_cblas(fb_layout_t layout, fb_transpose_t trans,
                             int m, int n, const void *alpha,
                             const void *a, int lda, const void *x,
                             int incx, const void *beta, void *y, int incy)
{
    fb_complex_float_t *y_values = (fb_complex_float_t *)y;

    g_cgemv_cblas_call.called += 1;
    g_cgemv_cblas_call.order = layout;
    g_cgemv_cblas_call.trans = trans;
    g_cgemv_cblas_call.m = m;
    g_cgemv_cblas_call.n = n;
    g_cgemv_cblas_call.lda = lda;
    g_cgemv_cblas_call.incx = incx;
    g_cgemv_cblas_call.incy = incy;
    g_cgemv_cblas_call.alpha = alpha;
    g_cgemv_cblas_call.a = a;
    g_cgemv_cblas_call.x = x;
    g_cgemv_cblas_call.beta = beta;
    g_cgemv_cblas_call.y = y;
    g_cgemv_cblas_call.alpha_value = *(const fb_complex_float_t *)alpha;
    g_cgemv_cblas_call.beta_value = *(const fb_complex_float_t *)beta;

    y_values[0] = make_cf32(111.0f, 1.0f);
    y_values[1] = make_cf32(112.0f, 2.0f);
}

static void stub_zgemv_fortran(const int *order, const int *trans,
                               const int *m, const int *n,
                               const void *alpha, const void *a,
                               const int *lda, const void *x,
                               const int *incx, const void *beta, void *y,
                               const int *incy)
{
    fb_complex_double_t *y_values = (fb_complex_double_t *)y;

    g_zgemv_fortran_call.called += 1;
    g_zgemv_fortran_call.order = *order;
    g_zgemv_fortran_call.trans = *trans;
    g_zgemv_fortran_call.m = *m;
    g_zgemv_fortran_call.n = *n;
    g_zgemv_fortran_call.lda = *lda;
    g_zgemv_fortran_call.incx = *incx;
    g_zgemv_fortran_call.incy = *incy;
    g_zgemv_fortran_call.alpha = alpha;
    g_zgemv_fortran_call.a = a;
    g_zgemv_fortran_call.x = x;
    g_zgemv_fortran_call.beta = beta;
    g_zgemv_fortran_call.y = y;
    g_zgemv_fortran_call.alpha_value = *(const fb_complex_double_t *)alpha;
    g_zgemv_fortran_call.beta_value = *(const fb_complex_double_t *)beta;

    y_values[0] = make_cf64(121.0, -1.0);
    y_values[1] = make_cf64(122.0, -2.0);
}

static void stub_zgemv_cblas(fb_layout_t layout, fb_transpose_t trans,
                             int m, int n, const void *alpha,
                             const void *a, int lda, const void *x,
                             int incx, const void *beta, void *y, int incy)
{
    fb_complex_double_t *y_values = (fb_complex_double_t *)y;

    g_zgemv_cblas_call.called += 1;
    g_zgemv_cblas_call.order = layout;
    g_zgemv_cblas_call.trans = trans;
    g_zgemv_cblas_call.m = m;
    g_zgemv_cblas_call.n = n;
    g_zgemv_cblas_call.lda = lda;
    g_zgemv_cblas_call.incx = incx;
    g_zgemv_cblas_call.incy = incy;
    g_zgemv_cblas_call.alpha = alpha;
    g_zgemv_cblas_call.a = a;
    g_zgemv_cblas_call.x = x;
    g_zgemv_cblas_call.beta = beta;
    g_zgemv_cblas_call.y = y;
    g_zgemv_cblas_call.alpha_value = *(const fb_complex_double_t *)alpha;
    g_zgemv_cblas_call.beta_value = *(const fb_complex_double_t *)beta;

    y_values[0] = make_cf64(131.0, 1.0);
    y_values[1] = make_cf64(132.0, 2.0);
}

static void stub_cger_fortran(const int *order, const int *m, const int *n,
                              const void *alpha, const void *x,
                              const int *incx, const void *y,
                              const int *incy, void *a, const int *lda)
{
    fb_complex_float_t *a_values = (fb_complex_float_t *)a;

    g_cger_fortran_call.called += 1;
    g_cger_fortran_call.order = *order;
    g_cger_fortran_call.m = *m;
    g_cger_fortran_call.n = *n;
    g_cger_fortran_call.incx = *incx;
    g_cger_fortran_call.incy = *incy;
    g_cger_fortran_call.lda = *lda;
    g_cger_fortran_call.alpha = alpha;
    g_cger_fortran_call.x = x;
    g_cger_fortran_call.y = y;
    g_cger_fortran_call.a = a;
    g_cger_fortran_call.alpha_value = *(const fb_complex_float_t *)alpha;

    a_values[0] = make_cf32(201.0f, -1.0f);
    a_values[1] = make_cf32(202.0f, -2.0f);
}

static void stub_cger_cblas(fb_layout_t layout, int m, int n,
                            const void *alpha, const void *x, int incx,
                            const void *y, int incy, void *a, int lda)
{
    fb_complex_float_t *a_values = (fb_complex_float_t *)a;

    g_cger_cblas_call.called += 1;
    g_cger_cblas_call.order = layout;
    g_cger_cblas_call.m = m;
    g_cger_cblas_call.n = n;
    g_cger_cblas_call.incx = incx;
    g_cger_cblas_call.incy = incy;
    g_cger_cblas_call.lda = lda;
    g_cger_cblas_call.alpha = alpha;
    g_cger_cblas_call.x = x;
    g_cger_cblas_call.y = y;
    g_cger_cblas_call.a = a;
    g_cger_cblas_call.alpha_value = *(const fb_complex_float_t *)alpha;

    a_values[0] = make_cf32(211.0f, 1.0f);
    a_values[1] = make_cf32(212.0f, 2.0f);
}

static void stub_zger_fortran(const int *order, const int *m, const int *n,
                              const void *alpha, const void *x,
                              const int *incx, const void *y,
                              const int *incy, void *a, const int *lda)
{
    fb_complex_double_t *a_values = (fb_complex_double_t *)a;

    g_zger_fortran_call.called += 1;
    g_zger_fortran_call.order = *order;
    g_zger_fortran_call.m = *m;
    g_zger_fortran_call.n = *n;
    g_zger_fortran_call.incx = *incx;
    g_zger_fortran_call.incy = *incy;
    g_zger_fortran_call.lda = *lda;
    g_zger_fortran_call.alpha = alpha;
    g_zger_fortran_call.x = x;
    g_zger_fortran_call.y = y;
    g_zger_fortran_call.a = a;
    g_zger_fortran_call.alpha_value = *(const fb_complex_double_t *)alpha;

    a_values[0] = make_cf64(221.0, -1.0);
    a_values[1] = make_cf64(222.0, -2.0);
}

static void stub_zger_cblas(fb_layout_t layout, int m, int n,
                            const void *alpha, const void *x, int incx,
                            const void *y, int incy, void *a, int lda)
{
    fb_complex_double_t *a_values = (fb_complex_double_t *)a;

    g_zger_cblas_call.called += 1;
    g_zger_cblas_call.order = layout;
    g_zger_cblas_call.m = m;
    g_zger_cblas_call.n = n;
    g_zger_cblas_call.incx = incx;
    g_zger_cblas_call.incy = incy;
    g_zger_cblas_call.lda = lda;
    g_zger_cblas_call.alpha = alpha;
    g_zger_cblas_call.x = x;
    g_zger_cblas_call.y = y;
    g_zger_cblas_call.a = a;
    g_zger_cblas_call.alpha_value = *(const fb_complex_double_t *)alpha;

    a_values[0] = make_cf64(231.0, 1.0);
    a_values[1] = make_cf64(232.0, 2.0);
}

static int check_cgemv_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_cgemv_cblas_fn thunk = NULL;
    fb_complex_float_t alpha = make_cf32(1.0f, 2.0f);
    fb_complex_float_t beta = make_cf32(3.0f, 4.0f);
    fb_complex_float_t a[4] = {
        make_cf32(5.0f, 0.0f), make_cf32(6.0f, 0.0f),
        make_cf32(7.0f, 0.0f), make_cf32(8.0f, 0.0f)
    };
    fb_complex_float_t x[2] = { make_cf32(9.0f, 1.0f), make_cf32(10.0f, 2.0f) };
    fb_complex_float_t y[2] = { make_cf32(11.0f, 3.0f), make_cf32(12.0f, 4.0f) };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cgemv_fortran_call, 0, sizeof(g_cgemv_fortran_call));
    vtable.ext_ops[FB_OP_CGEMV][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_cgemv_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CGEMV);
    thunk = (fb_cgemv_cblas_fn)vtable.ext_ops[FB_OP_CGEMV][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CGEMV Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    thunk(FB_LAYOUT_ROW_MAJOR, FB_TRANS, 2, 2, &alpha, a, 2, x, 1, &beta, y, 1);
    if (g_cgemv_fortran_call.called != 1 ||
        g_cgemv_fortran_call.order != FB_LAYOUT_ROW_MAJOR ||
        g_cgemv_fortran_call.trans != FB_TRANS ||
        g_cgemv_fortran_call.m != 2 ||
        g_cgemv_fortran_call.n != 2 ||
        g_cgemv_fortran_call.lda != 2 ||
        g_cgemv_fortran_call.incx != 1 ||
        g_cgemv_fortran_call.incy != 1 ||
        g_cgemv_fortran_call.alpha != &alpha ||
        g_cgemv_fortran_call.a != a ||
        g_cgemv_fortran_call.x != x ||
        g_cgemv_fortran_call.beta != &beta ||
        g_cgemv_fortran_call.y != y ||
        !cf32_eq(g_cgemv_fortran_call.alpha_value, alpha) ||
        !cf32_eq(g_cgemv_fortran_call.beta_value, beta) ||
        !cf32_eq(y[0], make_cf32(101.0f, -1.0f)) ||
        !cf32_eq(y[1], make_cf32(102.0f, -2.0f))) {
        fprintf(stderr, "[FAIL] CGEMV Fortran->CBLAS thunk did not forward complex general mat-vec arguments correctly\n");
        return 1;
    }

    printf("[PASS] CGEMV Fortran->CBLAS thunk forwards complex general mat-vec arguments unchanged\n");
    return 0;
}

static int check_cgemv_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_cgemv_fortran_slot_fn thunk = NULL;
    int order = FB_LAYOUT_COL_MAJOR;
    int trans = FB_NO_TRANS;
    int m = 2;
    int n = 2;
    int lda = 2;
    int incx = 1;
    int incy = 1;
    fb_complex_float_t alpha = make_cf32(13.0f, 14.0f);
    fb_complex_float_t beta = make_cf32(15.0f, 16.0f);
    fb_complex_float_t a[4] = {
        make_cf32(17.0f, 0.0f), make_cf32(18.0f, 0.0f),
        make_cf32(19.0f, 0.0f), make_cf32(20.0f, 0.0f)
    };
    fb_complex_float_t x[2] = { make_cf32(21.0f, 1.0f), make_cf32(22.0f, 2.0f) };
    fb_complex_float_t y[2] = { make_cf32(23.0f, 3.0f), make_cf32(24.0f, 4.0f) };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cgemv_cblas_call, 0, sizeof(g_cgemv_cblas_call));
    vtable.ext_ops[FB_OP_CGEMV][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_cgemv_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CGEMV);
    thunk = (fb_cgemv_fortran_slot_fn)vtable.ext_ops[FB_OP_CGEMV][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CGEMV CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&order, &trans, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
    if (g_cgemv_cblas_call.called != 1 ||
        g_cgemv_cblas_call.order != FB_LAYOUT_COL_MAJOR ||
        g_cgemv_cblas_call.trans != FB_NO_TRANS ||
        g_cgemv_cblas_call.m != 2 ||
        g_cgemv_cblas_call.n != 2 ||
        g_cgemv_cblas_call.lda != 2 ||
        g_cgemv_cblas_call.incx != 1 ||
        g_cgemv_cblas_call.incy != 1 ||
        g_cgemv_cblas_call.alpha != &alpha ||
        g_cgemv_cblas_call.a != a ||
        g_cgemv_cblas_call.x != x ||
        g_cgemv_cblas_call.beta != &beta ||
        g_cgemv_cblas_call.y != y ||
        !cf32_eq(g_cgemv_cblas_call.alpha_value, alpha) ||
        !cf32_eq(g_cgemv_cblas_call.beta_value, beta) ||
        !cf32_eq(y[0], make_cf32(111.0f, 1.0f)) ||
        !cf32_eq(y[1], make_cf32(112.0f, 2.0f))) {
        fprintf(stderr, "[FAIL] CGEMV CBLAS->Fortran thunk did not map complex general mat-vec arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] CGEMV CBLAS->Fortran thunk maps complex general mat-vec arguments into the C entry\n");
    return 0;
}

static int check_zgemv_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_zgemv_cblas_fn thunk = NULL;
    fb_complex_double_t alpha = make_cf64(25.0, 26.0);
    fb_complex_double_t beta = make_cf64(27.0, 28.0);
    fb_complex_double_t a[4] = {
        make_cf64(29.0, 0.0), make_cf64(30.0, 0.0),
        make_cf64(31.0, 0.0), make_cf64(32.0, 0.0)
    };
    fb_complex_double_t x[2] = { make_cf64(33.0, 1.0), make_cf64(34.0, 2.0) };
    fb_complex_double_t y[2] = { make_cf64(35.0, 3.0), make_cf64(36.0, 4.0) };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zgemv_fortran_call, 0, sizeof(g_zgemv_fortran_call));
    vtable.ext_ops[FB_OP_ZGEMV][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_zgemv_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_ZGEMV);
    thunk = (fb_zgemv_cblas_fn)vtable.ext_ops[FB_OP_ZGEMV][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZGEMV Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    thunk(FB_LAYOUT_ROW_MAJOR, FB_CONJ_TRANS, 2, 2, &alpha, a, 2, x, 1, &beta, y, 1);
    if (g_zgemv_fortran_call.called != 1 ||
        g_zgemv_fortran_call.order != FB_LAYOUT_ROW_MAJOR ||
        g_zgemv_fortran_call.trans != FB_CONJ_TRANS ||
        g_zgemv_fortran_call.m != 2 ||
        g_zgemv_fortran_call.n != 2 ||
        g_zgemv_fortran_call.lda != 2 ||
        g_zgemv_fortran_call.incx != 1 ||
        g_zgemv_fortran_call.incy != 1 ||
        g_zgemv_fortran_call.alpha != &alpha ||
        g_zgemv_fortran_call.a != a ||
        g_zgemv_fortran_call.x != x ||
        g_zgemv_fortran_call.beta != &beta ||
        g_zgemv_fortran_call.y != y ||
        !cf64_eq(g_zgemv_fortran_call.alpha_value, alpha) ||
        !cf64_eq(g_zgemv_fortran_call.beta_value, beta) ||
        !cf64_eq(y[0], make_cf64(121.0, -1.0)) ||
        !cf64_eq(y[1], make_cf64(122.0, -2.0))) {
        fprintf(stderr, "[FAIL] ZGEMV Fortran->CBLAS thunk did not forward double-complex general mat-vec arguments correctly\n");
        return 1;
    }

    printf("[PASS] ZGEMV Fortran->CBLAS thunk forwards double-complex general mat-vec arguments unchanged\n");
    return 0;
}

static int check_zgemv_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_zgemv_fortran_slot_fn thunk = NULL;
    int order = FB_LAYOUT_COL_MAJOR;
    int trans = FB_NO_TRANS;
    int m = 2;
    int n = 2;
    int lda = 2;
    int incx = 1;
    int incy = 1;
    fb_complex_double_t alpha = make_cf64(37.0, 38.0);
    fb_complex_double_t beta = make_cf64(39.0, 40.0);
    fb_complex_double_t a[4] = {
        make_cf64(41.0, 0.0), make_cf64(42.0, 0.0),
        make_cf64(43.0, 0.0), make_cf64(44.0, 0.0)
    };
    fb_complex_double_t x[2] = { make_cf64(45.0, 1.0), make_cf64(46.0, 2.0) };
    fb_complex_double_t y[2] = { make_cf64(47.0, 3.0), make_cf64(48.0, 4.0) };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zgemv_cblas_call, 0, sizeof(g_zgemv_cblas_call));
    vtable.ext_ops[FB_OP_ZGEMV][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_zgemv_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_ZGEMV);
    thunk = (fb_zgemv_fortran_slot_fn)vtable.ext_ops[FB_OP_ZGEMV][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZGEMV CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&order, &trans, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
    if (g_zgemv_cblas_call.called != 1 ||
        g_zgemv_cblas_call.order != FB_LAYOUT_COL_MAJOR ||
        g_zgemv_cblas_call.trans != FB_NO_TRANS ||
        g_zgemv_cblas_call.m != 2 ||
        g_zgemv_cblas_call.n != 2 ||
        g_zgemv_cblas_call.lda != 2 ||
        g_zgemv_cblas_call.incx != 1 ||
        g_zgemv_cblas_call.incy != 1 ||
        g_zgemv_cblas_call.alpha != &alpha ||
        g_zgemv_cblas_call.a != a ||
        g_zgemv_cblas_call.x != x ||
        g_zgemv_cblas_call.beta != &beta ||
        g_zgemv_cblas_call.y != y ||
        !cf64_eq(g_zgemv_cblas_call.alpha_value, alpha) ||
        !cf64_eq(g_zgemv_cblas_call.beta_value, beta) ||
        !cf64_eq(y[0], make_cf64(131.0, 1.0)) ||
        !cf64_eq(y[1], make_cf64(132.0, 2.0))) {
        fprintf(stderr, "[FAIL] ZGEMV CBLAS->Fortran thunk did not map double-complex general mat-vec arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] ZGEMV CBLAS->Fortran thunk maps double-complex general mat-vec arguments into the C entry\n");
    return 0;
}

static int check_cgeru_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_cger_cblas_fn thunk = NULL;
    fb_complex_float_t alpha = make_cf32(49.0f, 50.0f);
    fb_complex_float_t x[2] = { make_cf32(51.0f, 1.0f), make_cf32(52.0f, 2.0f) };
    fb_complex_float_t y[2] = { make_cf32(53.0f, 3.0f), make_cf32(54.0f, 4.0f) };
    fb_complex_float_t a[4] = {
        make_cf32(55.0f, 0.0f), make_cf32(56.0f, 0.0f),
        make_cf32(57.0f, 0.0f), make_cf32(58.0f, 0.0f)
    };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cger_fortran_call, 0, sizeof(g_cger_fortran_call));
    vtable.ext_ops[FB_OP_CGERU][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_cger_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CGERU);
    thunk = (fb_cger_cblas_fn)vtable.ext_ops[FB_OP_CGERU][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CGERU Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    thunk(FB_LAYOUT_ROW_MAJOR, 2, 2, &alpha, x, 1, y, 1, a, 2);
    if (g_cger_fortran_call.called != 1 ||
        g_cger_fortran_call.order != FB_LAYOUT_ROW_MAJOR ||
        g_cger_fortran_call.m != 2 ||
        g_cger_fortran_call.n != 2 ||
        g_cger_fortran_call.incx != 1 ||
        g_cger_fortran_call.incy != 1 ||
        g_cger_fortran_call.lda != 2 ||
        g_cger_fortran_call.alpha != &alpha ||
        g_cger_fortran_call.x != x ||
        g_cger_fortran_call.y != y ||
        g_cger_fortran_call.a != a ||
        !cf32_eq(g_cger_fortran_call.alpha_value, alpha) ||
        !cf32_eq(a[0], make_cf32(201.0f, -1.0f)) ||
        !cf32_eq(a[1], make_cf32(202.0f, -2.0f))) {
        fprintf(stderr, "[FAIL] CGERU Fortran->CBLAS thunk did not forward complex general rank-1 update arguments correctly\n");
        return 1;
    }

    printf("[PASS] CGERU Fortran->CBLAS thunk forwards complex general rank-1 update arguments unchanged\n");
    return 0;
}

static int check_cgeru_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_cger_fortran_slot_fn thunk = NULL;
    int order = FB_LAYOUT_COL_MAJOR;
    int m = 2;
    int n = 2;
    int incx = 1;
    int incy = 1;
    int lda = 2;
    fb_complex_float_t alpha = make_cf32(59.0f, 60.0f);
    fb_complex_float_t x[2] = { make_cf32(61.0f, 1.0f), make_cf32(62.0f, 2.0f) };
    fb_complex_float_t y[2] = { make_cf32(63.0f, 3.0f), make_cf32(64.0f, 4.0f) };
    fb_complex_float_t a[4] = {
        make_cf32(65.0f, 0.0f), make_cf32(66.0f, 0.0f),
        make_cf32(67.0f, 0.0f), make_cf32(68.0f, 0.0f)
    };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cger_cblas_call, 0, sizeof(g_cger_cblas_call));
    vtable.ext_ops[FB_OP_CGERU][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_cger_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CGERU);
    thunk = (fb_cger_fortran_slot_fn)vtable.ext_ops[FB_OP_CGERU][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CGERU CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&order, &m, &n, &alpha, x, &incx, y, &incy, a, &lda);
    if (g_cger_cblas_call.called != 1 ||
        g_cger_cblas_call.order != FB_LAYOUT_COL_MAJOR ||
        g_cger_cblas_call.m != 2 ||
        g_cger_cblas_call.n != 2 ||
        g_cger_cblas_call.incx != 1 ||
        g_cger_cblas_call.incy != 1 ||
        g_cger_cblas_call.lda != 2 ||
        g_cger_cblas_call.alpha != &alpha ||
        g_cger_cblas_call.x != x ||
        g_cger_cblas_call.y != y ||
        g_cger_cblas_call.a != a ||
        !cf32_eq(g_cger_cblas_call.alpha_value, alpha) ||
        !cf32_eq(a[0], make_cf32(211.0f, 1.0f)) ||
        !cf32_eq(a[1], make_cf32(212.0f, 2.0f))) {
        fprintf(stderr, "[FAIL] CGERU CBLAS->Fortran thunk did not map complex general rank-1 update arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] CGERU CBLAS->Fortran thunk maps complex general rank-1 update arguments into the C entry\n");
    return 0;
}

static int check_zgeru_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_zger_cblas_fn thunk = NULL;
    fb_complex_double_t alpha = make_cf64(69.0, 70.0);
    fb_complex_double_t x[2] = { make_cf64(71.0, 1.0), make_cf64(72.0, 2.0) };
    fb_complex_double_t y[2] = { make_cf64(73.0, 3.0), make_cf64(74.0, 4.0) };
    fb_complex_double_t a[4] = {
        make_cf64(75.0, 0.0), make_cf64(76.0, 0.0),
        make_cf64(77.0, 0.0), make_cf64(78.0, 0.0)
    };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zger_fortran_call, 0, sizeof(g_zger_fortran_call));
    vtable.ext_ops[FB_OP_ZGERU][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_zger_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_ZGERU);
    thunk = (fb_zger_cblas_fn)vtable.ext_ops[FB_OP_ZGERU][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZGERU Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    thunk(FB_LAYOUT_ROW_MAJOR, 2, 2, &alpha, x, 1, y, 1, a, 2);
    if (g_zger_fortran_call.called != 1 ||
        g_zger_fortran_call.order != FB_LAYOUT_ROW_MAJOR ||
        g_zger_fortran_call.m != 2 ||
        g_zger_fortran_call.n != 2 ||
        g_zger_fortran_call.incx != 1 ||
        g_zger_fortran_call.incy != 1 ||
        g_zger_fortran_call.lda != 2 ||
        g_zger_fortran_call.alpha != &alpha ||
        g_zger_fortran_call.x != x ||
        g_zger_fortran_call.y != y ||
        g_zger_fortran_call.a != a ||
        !cf64_eq(g_zger_fortran_call.alpha_value, alpha) ||
        !cf64_eq(a[0], make_cf64(221.0, -1.0)) ||
        !cf64_eq(a[1], make_cf64(222.0, -2.0))) {
        fprintf(stderr, "[FAIL] ZGERU Fortran->CBLAS thunk did not forward double-complex general rank-1 update arguments correctly\n");
        return 1;
    }

    printf("[PASS] ZGERU Fortran->CBLAS thunk forwards double-complex general rank-1 update arguments unchanged\n");
    return 0;
}

static int check_zgeru_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_zger_fortran_slot_fn thunk = NULL;
    int order = FB_LAYOUT_COL_MAJOR;
    int m = 2;
    int n = 2;
    int incx = 1;
    int incy = 1;
    int lda = 2;
    fb_complex_double_t alpha = make_cf64(79.0, 80.0);
    fb_complex_double_t x[2] = { make_cf64(81.0, 1.0), make_cf64(82.0, 2.0) };
    fb_complex_double_t y[2] = { make_cf64(83.0, 3.0), make_cf64(84.0, 4.0) };
    fb_complex_double_t a[4] = {
        make_cf64(85.0, 0.0), make_cf64(86.0, 0.0),
        make_cf64(87.0, 0.0), make_cf64(88.0, 0.0)
    };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zger_cblas_call, 0, sizeof(g_zger_cblas_call));
    vtable.ext_ops[FB_OP_ZGERU][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_zger_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_ZGERU);
    thunk = (fb_zger_fortran_slot_fn)vtable.ext_ops[FB_OP_ZGERU][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZGERU CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&order, &m, &n, &alpha, x, &incx, y, &incy, a, &lda);
    if (g_zger_cblas_call.called != 1 ||
        g_zger_cblas_call.order != FB_LAYOUT_COL_MAJOR ||
        g_zger_cblas_call.m != 2 ||
        g_zger_cblas_call.n != 2 ||
        g_zger_cblas_call.incx != 1 ||
        g_zger_cblas_call.incy != 1 ||
        g_zger_cblas_call.lda != 2 ||
        g_zger_cblas_call.alpha != &alpha ||
        g_zger_cblas_call.x != x ||
        g_zger_cblas_call.y != y ||
        g_zger_cblas_call.a != a ||
        !cf64_eq(g_zger_cblas_call.alpha_value, alpha) ||
        !cf64_eq(a[0], make_cf64(231.0, 1.0)) ||
        !cf64_eq(a[1], make_cf64(232.0, 2.0))) {
        fprintf(stderr, "[FAIL] ZGERU CBLAS->Fortran thunk did not map double-complex general rank-1 update arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] ZGERU CBLAS->Fortran thunk maps double-complex general rank-1 update arguments into the C entry\n");
    return 0;
}

static int check_cgerc_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_cger_cblas_fn thunk = NULL;
    fb_complex_float_t alpha = make_cf32(89.0f, 90.0f);
    fb_complex_float_t x[2] = { make_cf32(91.0f, 1.0f), make_cf32(92.0f, 2.0f) };
    fb_complex_float_t y[2] = { make_cf32(93.0f, 3.0f), make_cf32(94.0f, 4.0f) };
    fb_complex_float_t a[4] = {
        make_cf32(95.0f, 0.0f), make_cf32(96.0f, 0.0f),
        make_cf32(97.0f, 0.0f), make_cf32(98.0f, 0.0f)
    };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cger_fortran_call, 0, sizeof(g_cger_fortran_call));
    vtable.ext_ops[FB_OP_CGERC][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_cger_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CGERC);
    thunk = (fb_cger_cblas_fn)vtable.ext_ops[FB_OP_CGERC][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CGERC Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    thunk(FB_LAYOUT_ROW_MAJOR, 2, 2, &alpha, x, 1, y, 1, a, 2);
    if (g_cger_fortran_call.called != 1 ||
        g_cger_fortran_call.order != FB_LAYOUT_ROW_MAJOR ||
        g_cger_fortran_call.m != 2 ||
        g_cger_fortran_call.n != 2 ||
        g_cger_fortran_call.incx != 1 ||
        g_cger_fortran_call.incy != 1 ||
        g_cger_fortran_call.lda != 2 ||
        g_cger_fortran_call.alpha != &alpha ||
        g_cger_fortran_call.x != x ||
        g_cger_fortran_call.y != y ||
        g_cger_fortran_call.a != a ||
        !cf32_eq(g_cger_fortran_call.alpha_value, alpha) ||
        !cf32_eq(a[0], make_cf32(201.0f, -1.0f)) ||
        !cf32_eq(a[1], make_cf32(202.0f, -2.0f))) {
        fprintf(stderr, "[FAIL] CGERC Fortran->CBLAS thunk did not forward complex conjugated rank-1 update arguments correctly\n");
        return 1;
    }

    printf("[PASS] CGERC Fortran->CBLAS thunk forwards complex conjugated rank-1 update arguments unchanged\n");
    return 0;
}

static int check_cgerc_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_cger_fortran_slot_fn thunk = NULL;
    int order = FB_LAYOUT_COL_MAJOR;
    int m = 2;
    int n = 2;
    int incx = 1;
    int incy = 1;
    int lda = 2;
    fb_complex_float_t alpha = make_cf32(99.0f, 100.0f);
    fb_complex_float_t x[2] = { make_cf32(101.0f, 1.0f), make_cf32(102.0f, 2.0f) };
    fb_complex_float_t y[2] = { make_cf32(103.0f, 3.0f), make_cf32(104.0f, 4.0f) };
    fb_complex_float_t a[4] = {
        make_cf32(105.0f, 0.0f), make_cf32(106.0f, 0.0f),
        make_cf32(107.0f, 0.0f), make_cf32(108.0f, 0.0f)
    };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cger_cblas_call, 0, sizeof(g_cger_cblas_call));
    vtable.ext_ops[FB_OP_CGERC][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_cger_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CGERC);
    thunk = (fb_cger_fortran_slot_fn)vtable.ext_ops[FB_OP_CGERC][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CGERC CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&order, &m, &n, &alpha, x, &incx, y, &incy, a, &lda);
    if (g_cger_cblas_call.called != 1 ||
        g_cger_cblas_call.order != FB_LAYOUT_COL_MAJOR ||
        g_cger_cblas_call.m != 2 ||
        g_cger_cblas_call.n != 2 ||
        g_cger_cblas_call.incx != 1 ||
        g_cger_cblas_call.incy != 1 ||
        g_cger_cblas_call.lda != 2 ||
        g_cger_cblas_call.alpha != &alpha ||
        g_cger_cblas_call.x != x ||
        g_cger_cblas_call.y != y ||
        g_cger_cblas_call.a != a ||
        !cf32_eq(g_cger_cblas_call.alpha_value, alpha) ||
        !cf32_eq(a[0], make_cf32(211.0f, 1.0f)) ||
        !cf32_eq(a[1], make_cf32(212.0f, 2.0f))) {
        fprintf(stderr, "[FAIL] CGERC CBLAS->Fortran thunk did not map complex conjugated rank-1 update arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] CGERC CBLAS->Fortran thunk maps complex conjugated rank-1 update arguments into the C entry\n");
    return 0;
}

static int check_zgerc_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_zger_cblas_fn thunk = NULL;
    fb_complex_double_t alpha = make_cf64(109.0, 110.0);
    fb_complex_double_t x[2] = { make_cf64(111.0, 1.0), make_cf64(112.0, 2.0) };
    fb_complex_double_t y[2] = { make_cf64(113.0, 3.0), make_cf64(114.0, 4.0) };
    fb_complex_double_t a[4] = {
        make_cf64(115.0, 0.0), make_cf64(116.0, 0.0),
        make_cf64(117.0, 0.0), make_cf64(118.0, 0.0)
    };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zger_fortran_call, 0, sizeof(g_zger_fortran_call));
    vtable.ext_ops[FB_OP_ZGERC][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_zger_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_ZGERC);
    thunk = (fb_zger_cblas_fn)vtable.ext_ops[FB_OP_ZGERC][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZGERC Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    thunk(FB_LAYOUT_ROW_MAJOR, 2, 2, &alpha, x, 1, y, 1, a, 2);
    if (g_zger_fortran_call.called != 1 ||
        g_zger_fortran_call.order != FB_LAYOUT_ROW_MAJOR ||
        g_zger_fortran_call.m != 2 ||
        g_zger_fortran_call.n != 2 ||
        g_zger_fortran_call.incx != 1 ||
        g_zger_fortran_call.incy != 1 ||
        g_zger_fortran_call.lda != 2 ||
        g_zger_fortran_call.alpha != &alpha ||
        g_zger_fortran_call.x != x ||
        g_zger_fortran_call.y != y ||
        g_zger_fortran_call.a != a ||
        !cf64_eq(g_zger_fortran_call.alpha_value, alpha) ||
        !cf64_eq(a[0], make_cf64(221.0, -1.0)) ||
        !cf64_eq(a[1], make_cf64(222.0, -2.0))) {
        fprintf(stderr, "[FAIL] ZGERC Fortran->CBLAS thunk did not forward double-complex conjugated rank-1 update arguments correctly\n");
        return 1;
    }

    printf("[PASS] ZGERC Fortran->CBLAS thunk forwards double-complex conjugated rank-1 update arguments unchanged\n");
    return 0;
}

static int check_zgerc_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_zger_fortran_slot_fn thunk = NULL;
    int order = FB_LAYOUT_COL_MAJOR;
    int m = 2;
    int n = 2;
    int incx = 1;
    int incy = 1;
    int lda = 2;
    fb_complex_double_t alpha = make_cf64(119.0, 120.0);
    fb_complex_double_t x[2] = { make_cf64(121.0, 1.0), make_cf64(122.0, 2.0) };
    fb_complex_double_t y[2] = { make_cf64(123.0, 3.0), make_cf64(124.0, 4.0) };
    fb_complex_double_t a[4] = {
        make_cf64(125.0, 0.0), make_cf64(126.0, 0.0),
        make_cf64(127.0, 0.0), make_cf64(128.0, 0.0)
    };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zger_cblas_call, 0, sizeof(g_zger_cblas_call));
    vtable.ext_ops[FB_OP_ZGERC][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_zger_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_ZGERC);
    thunk = (fb_zger_fortran_slot_fn)vtable.ext_ops[FB_OP_ZGERC][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZGERC CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&order, &m, &n, &alpha, x, &incx, y, &incy, a, &lda);
    if (g_zger_cblas_call.called != 1 ||
        g_zger_cblas_call.order != FB_LAYOUT_COL_MAJOR ||
        g_zger_cblas_call.m != 2 ||
        g_zger_cblas_call.n != 2 ||
        g_zger_cblas_call.incx != 1 ||
        g_zger_cblas_call.incy != 1 ||
        g_zger_cblas_call.lda != 2 ||
        g_zger_cblas_call.alpha != &alpha ||
        g_zger_cblas_call.x != x ||
        g_zger_cblas_call.y != y ||
        g_zger_cblas_call.a != a ||
        !cf64_eq(g_zger_cblas_call.alpha_value, alpha) ||
        !cf64_eq(a[0], make_cf64(231.0, 1.0)) ||
        !cf64_eq(a[1], make_cf64(232.0, 2.0))) {
        fprintf(stderr, "[FAIL] ZGERC CBLAS->Fortran thunk did not map double-complex conjugated rank-1 update arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] ZGERC CBLAS->Fortran thunk maps double-complex conjugated rank-1 update arguments into the C entry\n");
    return 0;
}

int main(void)
{
    int status = 0;

    status |= check_cgemv_fortran_to_cblas();
    status |= check_cgemv_cblas_to_fortran();
    status |= check_zgemv_fortran_to_cblas();
    status |= check_zgemv_cblas_to_fortran();
    status |= check_cgeru_fortran_to_cblas();
    status |= check_cgeru_cblas_to_fortran();
    status |= check_zgeru_fortran_to_cblas();
    status |= check_zgeru_cblas_to_fortran();
    status |= check_cgerc_fortran_to_cblas();
    status |= check_cgerc_cblas_to_fortran();
    status |= check_zgerc_fortran_to_cblas();
    status |= check_zgerc_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}