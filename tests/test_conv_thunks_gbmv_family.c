#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef void (*fb_sgbmv_cblas_fn)(fb_layout_t layout, fb_transpose_t trans,
                                  int m, int n, int kl, int ku, float alpha,
                                  const float *a, int lda, const float *x,
                                  int incx, float beta, float *y, int incy);
typedef void (*fb_dgbmv_cblas_fn)(fb_layout_t layout, fb_transpose_t trans,
                                  int m, int n, int kl, int ku, double alpha,
                                  const double *a, int lda, const double *x,
                                  int incx, double beta, double *y, int incy);
typedef void (*fb_sgbmv_fortran_slot_fn)(const int *order, const int *trans,
                                         const int *m, const int *n,
                                         const int *kl, const int *ku,
                                         const float *alpha, const float *a,
                                         const int *lda, const float *x,
                                         const int *incx, const float *beta,
                                         float *y, const int *incy);
typedef void (*fb_dgbmv_fortran_slot_fn)(const int *order, const int *trans,
                                         const int *m, const int *n,
                                         const int *kl, const int *ku,
                                         const double *alpha, const double *a,
                                         const int *lda, const double *x,
                                         const int *incx, const double *beta,
                                         double *y, const int *incy);

typedef void (*fb_cgbmv_cblas_fn)(fb_layout_t layout, fb_transpose_t trans,
                                  int m, int n, int kl, int ku,
                                  const void *alpha, const void *a, int lda,
                                  const void *x, int incx, const void *beta,
                                  void *y, int incy);
typedef void (*fb_zgbmv_cblas_fn)(fb_layout_t layout, fb_transpose_t trans,
                                  int m, int n, int kl, int ku,
                                  const void *alpha, const void *a, int lda,
                                  const void *x, int incx, const void *beta,
                                  void *y, int incy);
typedef void (*fb_cgbmv_fortran_slot_fn)(const int *order, const int *trans,
                                         const int *m, const int *n,
                                         const int *kl, const int *ku,
                                         const void *alpha, const void *a,
                                         const int *lda, const void *x,
                                         const int *incx, const void *beta,
                                         void *y, const int *incy);
typedef void (*fb_zgbmv_fortran_slot_fn)(const int *order, const int *trans,
                                         const int *m, const int *n,
                                         const int *kl, const int *ku,
                                         const void *alpha, const void *a,
                                         const int *lda, const void *x,
                                         const int *incx, const void *beta,
                                         void *y, const int *incy);

typedef struct {
    int called;
    int order;
    int trans;
    int m;
    int n;
    int kl;
    int ku;
    int lda;
    int incx;
    int incy;
    const void *a;
    const void *x;
    void *y;
    float alpha_value;
    float beta_value;
} gbmv_call_f32_t;

typedef struct {
    int called;
    int order;
    int trans;
    int m;
    int n;
    int kl;
    int ku;
    int lda;
    int incx;
    int incy;
    const void *a;
    const void *x;
    void *y;
    double alpha_value;
    double beta_value;
} gbmv_call_f64_t;

typedef struct {
    int called;
    int order;
    int trans;
    int m;
    int n;
    int kl;
    int ku;
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
} cgbmv_call_f32_t;

typedef struct {
    int called;
    int order;
    int trans;
    int m;
    int n;
    int kl;
    int ku;
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
} cgbmv_call_f64_t;

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

static gbmv_call_f32_t g_sgbmv_fortran_call;
static gbmv_call_f32_t g_sgbmv_cblas_call;
static gbmv_call_f64_t g_dgbmv_fortran_call;
static gbmv_call_f64_t g_dgbmv_cblas_call;
static cgbmv_call_f32_t g_cgbmv_fortran_call;
static cgbmv_call_f32_t g_cgbmv_cblas_call;
static cgbmv_call_f64_t g_zgbmv_fortran_call;
static cgbmv_call_f64_t g_zgbmv_cblas_call;

static void stub_sgbmv_fortran(const int *order, const int *trans,
                               const int *m, const int *n, const int *kl,
                               const int *ku, const float *alpha,
                               const float *a, const int *lda,
                               const float *x, const int *incx,
                               const float *beta, float *y,
                               const int *incy)
{
    g_sgbmv_fortran_call.called += 1;
    g_sgbmv_fortran_call.order = *order;
    g_sgbmv_fortran_call.trans = *trans;
    g_sgbmv_fortran_call.m = *m;
    g_sgbmv_fortran_call.n = *n;
    g_sgbmv_fortran_call.kl = *kl;
    g_sgbmv_fortran_call.ku = *ku;
    g_sgbmv_fortran_call.lda = *lda;
    g_sgbmv_fortran_call.incx = *incx;
    g_sgbmv_fortran_call.incy = *incy;
    g_sgbmv_fortran_call.a = a;
    g_sgbmv_fortran_call.x = x;
    g_sgbmv_fortran_call.y = y;
    g_sgbmv_fortran_call.alpha_value = *alpha;
    g_sgbmv_fortran_call.beta_value = *beta;
    y[0] = 101.0f;
    y[1] = 102.0f;
}

static void stub_sgbmv_cblas(fb_layout_t layout, fb_transpose_t trans,
                             int m, int n, int kl, int ku, float alpha,
                             const float *a, int lda, const float *x,
                             int incx, float beta, float *y, int incy)
{
    g_sgbmv_cblas_call.called += 1;
    g_sgbmv_cblas_call.order = layout;
    g_sgbmv_cblas_call.trans = trans;
    g_sgbmv_cblas_call.m = m;
    g_sgbmv_cblas_call.n = n;
    g_sgbmv_cblas_call.kl = kl;
    g_sgbmv_cblas_call.ku = ku;
    g_sgbmv_cblas_call.lda = lda;
    g_sgbmv_cblas_call.incx = incx;
    g_sgbmv_cblas_call.incy = incy;
    g_sgbmv_cblas_call.a = a;
    g_sgbmv_cblas_call.x = x;
    g_sgbmv_cblas_call.y = y;
    g_sgbmv_cblas_call.alpha_value = alpha;
    g_sgbmv_cblas_call.beta_value = beta;
    y[0] = 111.0f;
    y[1] = 112.0f;
}

static void stub_dgbmv_fortran(const int *order, const int *trans,
                               const int *m, const int *n, const int *kl,
                               const int *ku, const double *alpha,
                               const double *a, const int *lda,
                               const double *x, const int *incx,
                               const double *beta, double *y,
                               const int *incy)
{
    g_dgbmv_fortran_call.called += 1;
    g_dgbmv_fortran_call.order = *order;
    g_dgbmv_fortran_call.trans = *trans;
    g_dgbmv_fortran_call.m = *m;
    g_dgbmv_fortran_call.n = *n;
    g_dgbmv_fortran_call.kl = *kl;
    g_dgbmv_fortran_call.ku = *ku;
    g_dgbmv_fortran_call.lda = *lda;
    g_dgbmv_fortran_call.incx = *incx;
    g_dgbmv_fortran_call.incy = *incy;
    g_dgbmv_fortran_call.a = a;
    g_dgbmv_fortran_call.x = x;
    g_dgbmv_fortran_call.y = y;
    g_dgbmv_fortran_call.alpha_value = *alpha;
    g_dgbmv_fortran_call.beta_value = *beta;
    y[0] = 121.0;
    y[1] = 122.0;
}

static void stub_dgbmv_cblas(fb_layout_t layout, fb_transpose_t trans,
                             int m, int n, int kl, int ku, double alpha,
                             const double *a, int lda, const double *x,
                             int incx, double beta, double *y, int incy)
{
    g_dgbmv_cblas_call.called += 1;
    g_dgbmv_cblas_call.order = layout;
    g_dgbmv_cblas_call.trans = trans;
    g_dgbmv_cblas_call.m = m;
    g_dgbmv_cblas_call.n = n;
    g_dgbmv_cblas_call.kl = kl;
    g_dgbmv_cblas_call.ku = ku;
    g_dgbmv_cblas_call.lda = lda;
    g_dgbmv_cblas_call.incx = incx;
    g_dgbmv_cblas_call.incy = incy;
    g_dgbmv_cblas_call.a = a;
    g_dgbmv_cblas_call.x = x;
    g_dgbmv_cblas_call.y = y;
    g_dgbmv_cblas_call.alpha_value = alpha;
    g_dgbmv_cblas_call.beta_value = beta;
    y[0] = 131.0;
    y[1] = 132.0;
}

static void stub_cgbmv_fortran(const int *order, const int *trans,
                               const int *m, const int *n, const int *kl,
                               const int *ku, const void *alpha,
                               const void *a, const int *lda,
                               const void *x, const int *incx,
                               const void *beta, void *y,
                               const int *incy)
{
    fb_complex_float_t *y_values = (fb_complex_float_t *)y;

    g_cgbmv_fortran_call.called += 1;
    g_cgbmv_fortran_call.order = *order;
    g_cgbmv_fortran_call.trans = *trans;
    g_cgbmv_fortran_call.m = *m;
    g_cgbmv_fortran_call.n = *n;
    g_cgbmv_fortran_call.kl = *kl;
    g_cgbmv_fortran_call.ku = *ku;
    g_cgbmv_fortran_call.lda = *lda;
    g_cgbmv_fortran_call.incx = *incx;
    g_cgbmv_fortran_call.incy = *incy;
    g_cgbmv_fortran_call.alpha = alpha;
    g_cgbmv_fortran_call.a = a;
    g_cgbmv_fortran_call.x = x;
    g_cgbmv_fortran_call.beta = beta;
    g_cgbmv_fortran_call.y = y;
    g_cgbmv_fortran_call.alpha_value = *(const fb_complex_float_t *)alpha;
    g_cgbmv_fortran_call.beta_value = *(const fb_complex_float_t *)beta;
    y_values[0] = make_cf32(141.0f, -1.0f);
    y_values[1] = make_cf32(142.0f, -2.0f);
}

static void stub_cgbmv_cblas(fb_layout_t layout, fb_transpose_t trans,
                             int m, int n, int kl, int ku,
                             const void *alpha, const void *a, int lda,
                             const void *x, int incx, const void *beta,
                             void *y, int incy)
{
    fb_complex_float_t *y_values = (fb_complex_float_t *)y;

    g_cgbmv_cblas_call.called += 1;
    g_cgbmv_cblas_call.order = layout;
    g_cgbmv_cblas_call.trans = trans;
    g_cgbmv_cblas_call.m = m;
    g_cgbmv_cblas_call.n = n;
    g_cgbmv_cblas_call.kl = kl;
    g_cgbmv_cblas_call.ku = ku;
    g_cgbmv_cblas_call.lda = lda;
    g_cgbmv_cblas_call.incx = incx;
    g_cgbmv_cblas_call.incy = incy;
    g_cgbmv_cblas_call.alpha = alpha;
    g_cgbmv_cblas_call.a = a;
    g_cgbmv_cblas_call.x = x;
    g_cgbmv_cblas_call.beta = beta;
    g_cgbmv_cblas_call.y = y;
    g_cgbmv_cblas_call.alpha_value = *(const fb_complex_float_t *)alpha;
    g_cgbmv_cblas_call.beta_value = *(const fb_complex_float_t *)beta;
    y_values[0] = make_cf32(151.0f, 1.0f);
    y_values[1] = make_cf32(152.0f, 2.0f);
}

static void stub_zgbmv_fortran(const int *order, const int *trans,
                               const int *m, const int *n, const int *kl,
                               const int *ku, const void *alpha,
                               const void *a, const int *lda,
                               const void *x, const int *incx,
                               const void *beta, void *y,
                               const int *incy)
{
    fb_complex_double_t *y_values = (fb_complex_double_t *)y;

    g_zgbmv_fortran_call.called += 1;
    g_zgbmv_fortran_call.order = *order;
    g_zgbmv_fortran_call.trans = *trans;
    g_zgbmv_fortran_call.m = *m;
    g_zgbmv_fortran_call.n = *n;
    g_zgbmv_fortran_call.kl = *kl;
    g_zgbmv_fortran_call.ku = *ku;
    g_zgbmv_fortran_call.lda = *lda;
    g_zgbmv_fortran_call.incx = *incx;
    g_zgbmv_fortran_call.incy = *incy;
    g_zgbmv_fortran_call.alpha = alpha;
    g_zgbmv_fortran_call.a = a;
    g_zgbmv_fortran_call.x = x;
    g_zgbmv_fortran_call.beta = beta;
    g_zgbmv_fortran_call.y = y;
    g_zgbmv_fortran_call.alpha_value = *(const fb_complex_double_t *)alpha;
    g_zgbmv_fortran_call.beta_value = *(const fb_complex_double_t *)beta;
    y_values[0] = make_cf64(161.0, -1.0);
    y_values[1] = make_cf64(162.0, -2.0);
}

static void stub_zgbmv_cblas(fb_layout_t layout, fb_transpose_t trans,
                             int m, int n, int kl, int ku,
                             const void *alpha, const void *a, int lda,
                             const void *x, int incx, const void *beta,
                             void *y, int incy)
{
    fb_complex_double_t *y_values = (fb_complex_double_t *)y;

    g_zgbmv_cblas_call.called += 1;
    g_zgbmv_cblas_call.order = layout;
    g_zgbmv_cblas_call.trans = trans;
    g_zgbmv_cblas_call.m = m;
    g_zgbmv_cblas_call.n = n;
    g_zgbmv_cblas_call.kl = kl;
    g_zgbmv_cblas_call.ku = ku;
    g_zgbmv_cblas_call.lda = lda;
    g_zgbmv_cblas_call.incx = incx;
    g_zgbmv_cblas_call.incy = incy;
    g_zgbmv_cblas_call.alpha = alpha;
    g_zgbmv_cblas_call.a = a;
    g_zgbmv_cblas_call.x = x;
    g_zgbmv_cblas_call.beta = beta;
    g_zgbmv_cblas_call.y = y;
    g_zgbmv_cblas_call.alpha_value = *(const fb_complex_double_t *)alpha;
    g_zgbmv_cblas_call.beta_value = *(const fb_complex_double_t *)beta;
    y_values[0] = make_cf64(171.0, 1.0);
    y_values[1] = make_cf64(172.0, 2.0);
}

static int check_sgbmv_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_sgbmv_cblas_fn thunk = NULL;
    float a[6] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
    float x[2] = { 7.0f, 8.0f };
    float y[2] = { 9.0f, 10.0f };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sgbmv_fortran_call, 0, sizeof(g_sgbmv_fortran_call));
    vtable.ext_ops[FB_OP_SGBMV][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_sgbmv_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SGBMV);
    thunk = (fb_sgbmv_cblas_fn)vtable.ext_ops[FB_OP_SGBMV][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SGBMV Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    thunk(FB_LAYOUT_ROW_MAJOR, FB_TRANS, 2, 2, 1, 1, 1.5f, a, 3, x, 1, 0.5f, y, 1);
    if (g_sgbmv_fortran_call.called != 1 ||
        g_sgbmv_fortran_call.order != FB_LAYOUT_ROW_MAJOR ||
        g_sgbmv_fortran_call.trans != FB_TRANS ||
        g_sgbmv_fortran_call.m != 2 ||
        g_sgbmv_fortran_call.n != 2 ||
        g_sgbmv_fortran_call.kl != 1 ||
        g_sgbmv_fortran_call.ku != 1 ||
        g_sgbmv_fortran_call.lda != 3 ||
        g_sgbmv_fortran_call.incx != 1 ||
        g_sgbmv_fortran_call.incy != 1 ||
        g_sgbmv_fortran_call.a != a ||
        g_sgbmv_fortran_call.x != x ||
        g_sgbmv_fortran_call.y != y ||
        g_sgbmv_fortran_call.alpha_value != 1.5f ||
        g_sgbmv_fortran_call.beta_value != 0.5f ||
        y[0] != 101.0f || y[1] != 102.0f) {
        fprintf(stderr, "[FAIL] SGBMV Fortran->CBLAS thunk did not forward real band mat-vec arguments correctly\n");
        return 1;
    }

    printf("[PASS] SGBMV Fortran->CBLAS thunk forwards real band mat-vec arguments unchanged\n");
    return 0;
}

static int check_sgbmv_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_sgbmv_fortran_slot_fn thunk = NULL;
    int order = FB_LAYOUT_COL_MAJOR;
    int trans = FB_NO_TRANS;
    int m = 2;
    int n = 2;
    int kl = 1;
    int ku = 1;
    int lda = 3;
    int incx = 1;
    int incy = 1;
    float alpha = 2.5f;
    float beta = 1.5f;
    float a[6] = { 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f };
    float x[2] = { 17.0f, 18.0f };
    float y[2] = { 19.0f, 20.0f };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sgbmv_cblas_call, 0, sizeof(g_sgbmv_cblas_call));
    vtable.ext_ops[FB_OP_SGBMV][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_sgbmv_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SGBMV);
    thunk = (fb_sgbmv_fortran_slot_fn)vtable.ext_ops[FB_OP_SGBMV][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SGBMV CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&order, &trans, &m, &n, &kl, &ku, &alpha, a, &lda, x, &incx, &beta, y, &incy);
    if (g_sgbmv_cblas_call.called != 1 ||
        g_sgbmv_cblas_call.order != FB_LAYOUT_COL_MAJOR ||
        g_sgbmv_cblas_call.trans != FB_NO_TRANS ||
        g_sgbmv_cblas_call.m != 2 ||
        g_sgbmv_cblas_call.n != 2 ||
        g_sgbmv_cblas_call.kl != 1 ||
        g_sgbmv_cblas_call.ku != 1 ||
        g_sgbmv_cblas_call.lda != 3 ||
        g_sgbmv_cblas_call.incx != 1 ||
        g_sgbmv_cblas_call.incy != 1 ||
        g_sgbmv_cblas_call.a != a ||
        g_sgbmv_cblas_call.x != x ||
        g_sgbmv_cblas_call.y != y ||
        g_sgbmv_cblas_call.alpha_value != 2.5f ||
        g_sgbmv_cblas_call.beta_value != 1.5f ||
        y[0] != 111.0f || y[1] != 112.0f) {
        fprintf(stderr, "[FAIL] SGBMV CBLAS->Fortran thunk did not map real band mat-vec arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] SGBMV CBLAS->Fortran thunk maps real band mat-vec arguments into the C entry\n");
    return 0;
}

static int check_dgbmv_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_dgbmv_cblas_fn thunk = NULL;
    double a[6] = { 21.0, 22.0, 23.0, 24.0, 25.0, 26.0 };
    double x[2] = { 27.0, 28.0 };
    double y[2] = { 29.0, 30.0 };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dgbmv_fortran_call, 0, sizeof(g_dgbmv_fortran_call));
    vtable.ext_ops[FB_OP_DGBMV][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_dgbmv_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_DGBMV);
    thunk = (fb_dgbmv_cblas_fn)vtable.ext_ops[FB_OP_DGBMV][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DGBMV Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    thunk(FB_LAYOUT_ROW_MAJOR, FB_TRANS, 2, 2, 1, 1, 3.5, a, 3, x, 1, 2.5, y, 1);
    if (g_dgbmv_fortran_call.called != 1 ||
        g_dgbmv_fortran_call.order != FB_LAYOUT_ROW_MAJOR ||
        g_dgbmv_fortran_call.trans != FB_TRANS ||
        g_dgbmv_fortran_call.m != 2 ||
        g_dgbmv_fortran_call.n != 2 ||
        g_dgbmv_fortran_call.kl != 1 ||
        g_dgbmv_fortran_call.ku != 1 ||
        g_dgbmv_fortran_call.lda != 3 ||
        g_dgbmv_fortran_call.incx != 1 ||
        g_dgbmv_fortran_call.incy != 1 ||
        g_dgbmv_fortran_call.a != a ||
        g_dgbmv_fortran_call.x != x ||
        g_dgbmv_fortran_call.y != y ||
        g_dgbmv_fortran_call.alpha_value != 3.5 ||
        g_dgbmv_fortran_call.beta_value != 2.5 ||
        y[0] != 121.0 || y[1] != 122.0) {
        fprintf(stderr, "[FAIL] DGBMV Fortran->CBLAS thunk did not forward double band mat-vec arguments correctly\n");
        return 1;
    }

    printf("[PASS] DGBMV Fortran->CBLAS thunk forwards double band mat-vec arguments unchanged\n");
    return 0;
}

static int check_dgbmv_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_dgbmv_fortran_slot_fn thunk = NULL;
    int order = FB_LAYOUT_COL_MAJOR;
    int trans = FB_NO_TRANS;
    int m = 2;
    int n = 2;
    int kl = 1;
    int ku = 1;
    int lda = 3;
    int incx = 1;
    int incy = 1;
    double alpha = 4.5;
    double beta = 3.5;
    double a[6] = { 31.0, 32.0, 33.0, 34.0, 35.0, 36.0 };
    double x[2] = { 37.0, 38.0 };
    double y[2] = { 39.0, 40.0 };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dgbmv_cblas_call, 0, sizeof(g_dgbmv_cblas_call));
    vtable.ext_ops[FB_OP_DGBMV][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_dgbmv_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_DGBMV);
    thunk = (fb_dgbmv_fortran_slot_fn)vtable.ext_ops[FB_OP_DGBMV][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DGBMV CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&order, &trans, &m, &n, &kl, &ku, &alpha, a, &lda, x, &incx, &beta, y, &incy);
    if (g_dgbmv_cblas_call.called != 1 ||
        g_dgbmv_cblas_call.order != FB_LAYOUT_COL_MAJOR ||
        g_dgbmv_cblas_call.trans != FB_NO_TRANS ||
        g_dgbmv_cblas_call.m != 2 ||
        g_dgbmv_cblas_call.n != 2 ||
        g_dgbmv_cblas_call.kl != 1 ||
        g_dgbmv_cblas_call.ku != 1 ||
        g_dgbmv_cblas_call.lda != 3 ||
        g_dgbmv_cblas_call.incx != 1 ||
        g_dgbmv_cblas_call.incy != 1 ||
        g_dgbmv_cblas_call.a != a ||
        g_dgbmv_cblas_call.x != x ||
        g_dgbmv_cblas_call.y != y ||
        g_dgbmv_cblas_call.alpha_value != 4.5 ||
        g_dgbmv_cblas_call.beta_value != 3.5 ||
        y[0] != 131.0 || y[1] != 132.0) {
        fprintf(stderr, "[FAIL] DGBMV CBLAS->Fortran thunk did not map double band mat-vec arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] DGBMV CBLAS->Fortran thunk maps double band mat-vec arguments into the C entry\n");
    return 0;
}

static int check_cgbmv_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_cgbmv_cblas_fn thunk = NULL;
    fb_complex_float_t alpha = make_cf32(5.0f, 6.0f);
    fb_complex_float_t beta = make_cf32(7.0f, 8.0f);
    fb_complex_float_t a[6] = {
        make_cf32(41.0f, 0.0f), make_cf32(42.0f, 0.0f),
        make_cf32(43.0f, 0.0f), make_cf32(44.0f, 0.0f),
        make_cf32(45.0f, 0.0f), make_cf32(46.0f, 0.0f)
    };
    fb_complex_float_t x[2] = { make_cf32(47.0f, 1.0f), make_cf32(48.0f, 2.0f) };
    fb_complex_float_t y[2] = { make_cf32(49.0f, 3.0f), make_cf32(50.0f, 4.0f) };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cgbmv_fortran_call, 0, sizeof(g_cgbmv_fortran_call));
    vtable.ext_ops[FB_OP_CGBMV][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_cgbmv_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CGBMV);
    thunk = (fb_cgbmv_cblas_fn)vtable.ext_ops[FB_OP_CGBMV][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CGBMV Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    thunk(FB_LAYOUT_ROW_MAJOR, FB_CONJ_TRANS, 2, 2, 1, 1, &alpha, a, 3, x, 1, &beta, y, 1);
    if (g_cgbmv_fortran_call.called != 1 ||
        g_cgbmv_fortran_call.order != FB_LAYOUT_ROW_MAJOR ||
        g_cgbmv_fortran_call.trans != FB_CONJ_TRANS ||
        g_cgbmv_fortran_call.m != 2 ||
        g_cgbmv_fortran_call.n != 2 ||
        g_cgbmv_fortran_call.kl != 1 ||
        g_cgbmv_fortran_call.ku != 1 ||
        g_cgbmv_fortran_call.lda != 3 ||
        g_cgbmv_fortran_call.incx != 1 ||
        g_cgbmv_fortran_call.incy != 1 ||
        g_cgbmv_fortran_call.alpha != &alpha ||
        g_cgbmv_fortran_call.a != a ||
        g_cgbmv_fortran_call.x != x ||
        g_cgbmv_fortran_call.beta != &beta ||
        g_cgbmv_fortran_call.y != y ||
        !cf32_eq(g_cgbmv_fortran_call.alpha_value, alpha) ||
        !cf32_eq(g_cgbmv_fortran_call.beta_value, beta) ||
        !cf32_eq(y[0], make_cf32(141.0f, -1.0f)) ||
        !cf32_eq(y[1], make_cf32(142.0f, -2.0f))) {
        fprintf(stderr, "[FAIL] CGBMV Fortran->CBLAS thunk did not forward complex band mat-vec arguments correctly\n");
        return 1;
    }

    printf("[PASS] CGBMV Fortran->CBLAS thunk forwards complex band mat-vec arguments unchanged\n");
    return 0;
}

static int check_cgbmv_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_cgbmv_fortran_slot_fn thunk = NULL;
    int order = FB_LAYOUT_COL_MAJOR;
    int trans = FB_NO_TRANS;
    int m = 2;
    int n = 2;
    int kl = 1;
    int ku = 1;
    int lda = 3;
    int incx = 1;
    int incy = 1;
    fb_complex_float_t alpha = make_cf32(9.0f, 10.0f);
    fb_complex_float_t beta = make_cf32(11.0f, 12.0f);
    fb_complex_float_t a[6] = {
        make_cf32(51.0f, 0.0f), make_cf32(52.0f, 0.0f),
        make_cf32(53.0f, 0.0f), make_cf32(54.0f, 0.0f),
        make_cf32(55.0f, 0.0f), make_cf32(56.0f, 0.0f)
    };
    fb_complex_float_t x[2] = { make_cf32(57.0f, 1.0f), make_cf32(58.0f, 2.0f) };
    fb_complex_float_t y[2] = { make_cf32(59.0f, 3.0f), make_cf32(60.0f, 4.0f) };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cgbmv_cblas_call, 0, sizeof(g_cgbmv_cblas_call));
    vtable.ext_ops[FB_OP_CGBMV][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_cgbmv_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CGBMV);
    thunk = (fb_cgbmv_fortran_slot_fn)vtable.ext_ops[FB_OP_CGBMV][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CGBMV CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&order, &trans, &m, &n, &kl, &ku, &alpha, a, &lda, x, &incx, &beta, y, &incy);
    if (g_cgbmv_cblas_call.called != 1 ||
        g_cgbmv_cblas_call.order != FB_LAYOUT_COL_MAJOR ||
        g_cgbmv_cblas_call.trans != FB_NO_TRANS ||
        g_cgbmv_cblas_call.m != 2 ||
        g_cgbmv_cblas_call.n != 2 ||
        g_cgbmv_cblas_call.kl != 1 ||
        g_cgbmv_cblas_call.ku != 1 ||
        g_cgbmv_cblas_call.lda != 3 ||
        g_cgbmv_cblas_call.incx != 1 ||
        g_cgbmv_cblas_call.incy != 1 ||
        g_cgbmv_cblas_call.alpha != &alpha ||
        g_cgbmv_cblas_call.a != a ||
        g_cgbmv_cblas_call.x != x ||
        g_cgbmv_cblas_call.beta != &beta ||
        g_cgbmv_cblas_call.y != y ||
        !cf32_eq(g_cgbmv_cblas_call.alpha_value, alpha) ||
        !cf32_eq(g_cgbmv_cblas_call.beta_value, beta) ||
        !cf32_eq(y[0], make_cf32(151.0f, 1.0f)) ||
        !cf32_eq(y[1], make_cf32(152.0f, 2.0f))) {
        fprintf(stderr, "[FAIL] CGBMV CBLAS->Fortran thunk did not map complex band mat-vec arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] CGBMV CBLAS->Fortran thunk maps complex band mat-vec arguments into the C entry\n");
    return 0;
}

static int check_zgbmv_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_zgbmv_cblas_fn thunk = NULL;
    fb_complex_double_t alpha = make_cf64(13.0, 14.0);
    fb_complex_double_t beta = make_cf64(15.0, 16.0);
    fb_complex_double_t a[6] = {
        make_cf64(61.0, 0.0), make_cf64(62.0, 0.0),
        make_cf64(63.0, 0.0), make_cf64(64.0, 0.0),
        make_cf64(65.0, 0.0), make_cf64(66.0, 0.0)
    };
    fb_complex_double_t x[2] = { make_cf64(67.0, 1.0), make_cf64(68.0, 2.0) };
    fb_complex_double_t y[2] = { make_cf64(69.0, 3.0), make_cf64(70.0, 4.0) };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zgbmv_fortran_call, 0, sizeof(g_zgbmv_fortran_call));
    vtable.ext_ops[FB_OP_ZGBMV][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_zgbmv_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_ZGBMV);
    thunk = (fb_zgbmv_cblas_fn)vtable.ext_ops[FB_OP_ZGBMV][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZGBMV Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    thunk(FB_LAYOUT_ROW_MAJOR, FB_CONJ_TRANS, 2, 2, 1, 1, &alpha, a, 3, x, 1, &beta, y, 1);
    if (g_zgbmv_fortran_call.called != 1 ||
        g_zgbmv_fortran_call.order != FB_LAYOUT_ROW_MAJOR ||
        g_zgbmv_fortran_call.trans != FB_CONJ_TRANS ||
        g_zgbmv_fortran_call.m != 2 ||
        g_zgbmv_fortran_call.n != 2 ||
        g_zgbmv_fortran_call.kl != 1 ||
        g_zgbmv_fortran_call.ku != 1 ||
        g_zgbmv_fortran_call.lda != 3 ||
        g_zgbmv_fortran_call.incx != 1 ||
        g_zgbmv_fortran_call.incy != 1 ||
        g_zgbmv_fortran_call.alpha != &alpha ||
        g_zgbmv_fortran_call.a != a ||
        g_zgbmv_fortran_call.x != x ||
        g_zgbmv_fortran_call.beta != &beta ||
        g_zgbmv_fortran_call.y != y ||
        !cf64_eq(g_zgbmv_fortran_call.alpha_value, alpha) ||
        !cf64_eq(g_zgbmv_fortran_call.beta_value, beta) ||
        !cf64_eq(y[0], make_cf64(161.0, -1.0)) ||
        !cf64_eq(y[1], make_cf64(162.0, -2.0))) {
        fprintf(stderr, "[FAIL] ZGBMV Fortran->CBLAS thunk did not forward double-complex band mat-vec arguments correctly\n");
        return 1;
    }

    printf("[PASS] ZGBMV Fortran->CBLAS thunk forwards double-complex band mat-vec arguments unchanged\n");
    return 0;
}

static int check_zgbmv_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_zgbmv_fortran_slot_fn thunk = NULL;
    int order = FB_LAYOUT_COL_MAJOR;
    int trans = FB_NO_TRANS;
    int m = 2;
    int n = 2;
    int kl = 1;
    int ku = 1;
    int lda = 3;
    int incx = 1;
    int incy = 1;
    fb_complex_double_t alpha = make_cf64(17.0, 18.0);
    fb_complex_double_t beta = make_cf64(19.0, 20.0);
    fb_complex_double_t a[6] = {
        make_cf64(71.0, 0.0), make_cf64(72.0, 0.0),
        make_cf64(73.0, 0.0), make_cf64(74.0, 0.0),
        make_cf64(75.0, 0.0), make_cf64(76.0, 0.0)
    };
    fb_complex_double_t x[2] = { make_cf64(77.0, 1.0), make_cf64(78.0, 2.0) };
    fb_complex_double_t y[2] = { make_cf64(79.0, 3.0), make_cf64(80.0, 4.0) };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zgbmv_cblas_call, 0, sizeof(g_zgbmv_cblas_call));
    vtable.ext_ops[FB_OP_ZGBMV][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_zgbmv_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_ZGBMV);
    thunk = (fb_zgbmv_fortran_slot_fn)vtable.ext_ops[FB_OP_ZGBMV][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZGBMV CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&order, &trans, &m, &n, &kl, &ku, &alpha, a, &lda, x, &incx, &beta, y, &incy);
    if (g_zgbmv_cblas_call.called != 1 ||
        g_zgbmv_cblas_call.order != FB_LAYOUT_COL_MAJOR ||
        g_zgbmv_cblas_call.trans != FB_NO_TRANS ||
        g_zgbmv_cblas_call.m != 2 ||
        g_zgbmv_cblas_call.n != 2 ||
        g_zgbmv_cblas_call.kl != 1 ||
        g_zgbmv_cblas_call.ku != 1 ||
        g_zgbmv_cblas_call.lda != 3 ||
        g_zgbmv_cblas_call.incx != 1 ||
        g_zgbmv_cblas_call.incy != 1 ||
        g_zgbmv_cblas_call.alpha != &alpha ||
        g_zgbmv_cblas_call.a != a ||
        g_zgbmv_cblas_call.x != x ||
        g_zgbmv_cblas_call.beta != &beta ||
        g_zgbmv_cblas_call.y != y ||
        !cf64_eq(g_zgbmv_cblas_call.alpha_value, alpha) ||
        !cf64_eq(g_zgbmv_cblas_call.beta_value, beta) ||
        !cf64_eq(y[0], make_cf64(171.0, 1.0)) ||
        !cf64_eq(y[1], make_cf64(172.0, 2.0))) {
        fprintf(stderr, "[FAIL] ZGBMV CBLAS->Fortran thunk did not map double-complex band mat-vec arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] ZGBMV CBLAS->Fortran thunk maps double-complex band mat-vec arguments into the C entry\n");
    return 0;
}

int main(void)
{
    int status = 0;

    status |= check_sgbmv_fortran_to_cblas();
    status |= check_sgbmv_cblas_to_fortran();
    status |= check_dgbmv_fortran_to_cblas();
    status |= check_dgbmv_cblas_to_fortran();
    status |= check_cgbmv_fortran_to_cblas();
    status |= check_cgbmv_cblas_to_fortran();
    status |= check_zgbmv_fortran_to_cblas();
    status |= check_zgbmv_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}