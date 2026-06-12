#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef void (*fb_ssbmv_cblas_fn)(fb_layout_t layout, fb_uplo_t uplo, int n,
                                  int k, float alpha, const float *a, int lda,
                                  const float *x, int incx, float beta,
                                  float *y, int incy);
typedef void (*fb_dsbmv_cblas_fn)(fb_layout_t layout, fb_uplo_t uplo, int n,
                                  int k, double alpha, const double *a,
                                  int lda, const double *x, int incx,
                                  double beta, double *y, int incy);
typedef void (*fb_ssbmv_fortran_slot_fn)(const int *order, const int *uplo,
                                         const int *n, const int *k,
                                         const float *alpha, const float *a,
                                         const int *lda, const float *x,
                                         const int *incx, const float *beta,
                                         float *y, const int *incy);
typedef void (*fb_dsbmv_fortran_slot_fn)(const int *order, const int *uplo,
                                         const int *n, const int *k,
                                         const double *alpha, const double *a,
                                         const int *lda, const double *x,
                                         const int *incx, const double *beta,
                                         double *y, const int *incy);

typedef void (*fb_sspmv_cblas_fn)(fb_layout_t layout, fb_uplo_t uplo, int n,
                                  float alpha, const float *ap,
                                  const float *x, int incx, float beta,
                                  float *y, int incy);
typedef void (*fb_dspmv_cblas_fn)(fb_layout_t layout, fb_uplo_t uplo, int n,
                                  double alpha, const double *ap,
                                  const double *x, int incx, double beta,
                                  double *y, int incy);
typedef void (*fb_sspmv_fortran_slot_fn)(const int *order, const int *uplo,
                                         const int *n, const float *alpha,
                                         const float *ap, const float *x,
                                         const int *incx, const float *beta,
                                         float *y, const int *incy);
typedef void (*fb_dspmv_fortran_slot_fn)(const int *order, const int *uplo,
                                         const int *n, const double *alpha,
                                         const double *ap, const double *x,
                                         const int *incx, const double *beta,
                                         double *y, const int *incy);

typedef struct {
    int called;
    int order;
    int uplo;
    int n;
    int k;
    int lda;
    int incx;
    int incy;
    const void *a;
    const void *x;
    void *y;
    float alpha_value;
    float beta_value;
} sbmv_call_f32_t;

typedef struct {
    int called;
    int order;
    int uplo;
    int n;
    int k;
    int lda;
    int incx;
    int incy;
    const void *a;
    const void *x;
    void *y;
    double alpha_value;
    double beta_value;
} sbmv_call_f64_t;

typedef struct {
    int called;
    int order;
    int uplo;
    int n;
    int incx;
    int incy;
    const void *ap;
    const void *x;
    void *y;
    float alpha_value;
    float beta_value;
} spmv_call_f32_t;

typedef struct {
    int called;
    int order;
    int uplo;
    int n;
    int incx;
    int incy;
    const void *ap;
    const void *x;
    void *y;
    double alpha_value;
    double beta_value;
} spmv_call_f64_t;

static sbmv_call_f32_t g_ssbmv_fortran_call;
static sbmv_call_f32_t g_ssbmv_cblas_call;
static sbmv_call_f64_t g_dsbmv_fortran_call;
static sbmv_call_f64_t g_dsbmv_cblas_call;
static spmv_call_f32_t g_sspmv_fortran_call;
static spmv_call_f32_t g_sspmv_cblas_call;
static spmv_call_f64_t g_dspmv_fortran_call;
static spmv_call_f64_t g_dspmv_cblas_call;

static void stub_ssbmv_fortran(const int *order, const int *uplo,
                               const int *n, const int *k, const float *alpha,
                               const float *a, const int *lda, const float *x,
                               const int *incx, const float *beta, float *y,
                               const int *incy)
{
    g_ssbmv_fortran_call.called += 1;
    g_ssbmv_fortran_call.order = *order;
    g_ssbmv_fortran_call.uplo = *uplo;
    g_ssbmv_fortran_call.n = *n;
    g_ssbmv_fortran_call.k = *k;
    g_ssbmv_fortran_call.lda = *lda;
    g_ssbmv_fortran_call.incx = *incx;
    g_ssbmv_fortran_call.incy = *incy;
    g_ssbmv_fortran_call.a = a;
    g_ssbmv_fortran_call.x = x;
    g_ssbmv_fortran_call.y = y;
    g_ssbmv_fortran_call.alpha_value = *alpha;
    g_ssbmv_fortran_call.beta_value = *beta;
    y[0] = 101.0f;
    y[1] = 102.0f;
}

static void stub_ssbmv_cblas(fb_layout_t layout, fb_uplo_t uplo, int n, int k,
                             float alpha, const float *a, int lda,
                             const float *x, int incx, float beta, float *y,
                             int incy)
{
    g_ssbmv_cblas_call.called += 1;
    g_ssbmv_cblas_call.order = layout;
    g_ssbmv_cblas_call.uplo = uplo;
    g_ssbmv_cblas_call.n = n;
    g_ssbmv_cblas_call.k = k;
    g_ssbmv_cblas_call.lda = lda;
    g_ssbmv_cblas_call.incx = incx;
    g_ssbmv_cblas_call.incy = incy;
    g_ssbmv_cblas_call.a = a;
    g_ssbmv_cblas_call.x = x;
    g_ssbmv_cblas_call.y = y;
    g_ssbmv_cblas_call.alpha_value = alpha;
    g_ssbmv_cblas_call.beta_value = beta;
    y[0] = 111.0f;
    y[1] = 112.0f;
}

static void stub_dsbmv_fortran(const int *order, const int *uplo,
                               const int *n, const int *k,
                               const double *alpha, const double *a,
                               const int *lda, const double *x,
                               const int *incx, const double *beta, double *y,
                               const int *incy)
{
    g_dsbmv_fortran_call.called += 1;
    g_dsbmv_fortran_call.order = *order;
    g_dsbmv_fortran_call.uplo = *uplo;
    g_dsbmv_fortran_call.n = *n;
    g_dsbmv_fortran_call.k = *k;
    g_dsbmv_fortran_call.lda = *lda;
    g_dsbmv_fortran_call.incx = *incx;
    g_dsbmv_fortran_call.incy = *incy;
    g_dsbmv_fortran_call.a = a;
    g_dsbmv_fortran_call.x = x;
    g_dsbmv_fortran_call.y = y;
    g_dsbmv_fortran_call.alpha_value = *alpha;
    g_dsbmv_fortran_call.beta_value = *beta;
    y[0] = 121.0;
    y[1] = 122.0;
}

static void stub_dsbmv_cblas(fb_layout_t layout, fb_uplo_t uplo, int n, int k,
                             double alpha, const double *a, int lda,
                             const double *x, int incx, double beta,
                             double *y, int incy)
{
    g_dsbmv_cblas_call.called += 1;
    g_dsbmv_cblas_call.order = layout;
    g_dsbmv_cblas_call.uplo = uplo;
    g_dsbmv_cblas_call.n = n;
    g_dsbmv_cblas_call.k = k;
    g_dsbmv_cblas_call.lda = lda;
    g_dsbmv_cblas_call.incx = incx;
    g_dsbmv_cblas_call.incy = incy;
    g_dsbmv_cblas_call.a = a;
    g_dsbmv_cblas_call.x = x;
    g_dsbmv_cblas_call.y = y;
    g_dsbmv_cblas_call.alpha_value = alpha;
    g_dsbmv_cblas_call.beta_value = beta;
    y[0] = 131.0;
    y[1] = 132.0;
}

static void stub_sspmv_fortran(const int *order, const int *uplo,
                               const int *n, const float *alpha,
                               const float *ap, const float *x,
                               const int *incx, const float *beta, float *y,
                               const int *incy)
{
    g_sspmv_fortran_call.called += 1;
    g_sspmv_fortran_call.order = *order;
    g_sspmv_fortran_call.uplo = *uplo;
    g_sspmv_fortran_call.n = *n;
    g_sspmv_fortran_call.incx = *incx;
    g_sspmv_fortran_call.incy = *incy;
    g_sspmv_fortran_call.ap = ap;
    g_sspmv_fortran_call.x = x;
    g_sspmv_fortran_call.y = y;
    g_sspmv_fortran_call.alpha_value = *alpha;
    g_sspmv_fortran_call.beta_value = *beta;
    y[0] = 201.0f;
    y[1] = 202.0f;
}

static void stub_sspmv_cblas(fb_layout_t layout, fb_uplo_t uplo, int n,
                             float alpha, const float *ap, const float *x,
                             int incx, float beta, float *y, int incy)
{
    g_sspmv_cblas_call.called += 1;
    g_sspmv_cblas_call.order = layout;
    g_sspmv_cblas_call.uplo = uplo;
    g_sspmv_cblas_call.n = n;
    g_sspmv_cblas_call.incx = incx;
    g_sspmv_cblas_call.incy = incy;
    g_sspmv_cblas_call.ap = ap;
    g_sspmv_cblas_call.x = x;
    g_sspmv_cblas_call.y = y;
    g_sspmv_cblas_call.alpha_value = alpha;
    g_sspmv_cblas_call.beta_value = beta;
    y[0] = 211.0f;
    y[1] = 212.0f;
}

static void stub_dspmv_fortran(const int *order, const int *uplo,
                               const int *n, const double *alpha,
                               const double *ap, const double *x,
                               const int *incx, const double *beta, double *y,
                               const int *incy)
{
    g_dspmv_fortran_call.called += 1;
    g_dspmv_fortran_call.order = *order;
    g_dspmv_fortran_call.uplo = *uplo;
    g_dspmv_fortran_call.n = *n;
    g_dspmv_fortran_call.incx = *incx;
    g_dspmv_fortran_call.incy = *incy;
    g_dspmv_fortran_call.ap = ap;
    g_dspmv_fortran_call.x = x;
    g_dspmv_fortran_call.y = y;
    g_dspmv_fortran_call.alpha_value = *alpha;
    g_dspmv_fortran_call.beta_value = *beta;
    y[0] = 221.0;
    y[1] = 222.0;
}

static void stub_dspmv_cblas(fb_layout_t layout, fb_uplo_t uplo, int n,
                             double alpha, const double *ap, const double *x,
                             int incx, double beta, double *y, int incy)
{
    g_dspmv_cblas_call.called += 1;
    g_dspmv_cblas_call.order = layout;
    g_dspmv_cblas_call.uplo = uplo;
    g_dspmv_cblas_call.n = n;
    g_dspmv_cblas_call.incx = incx;
    g_dspmv_cblas_call.incy = incy;
    g_dspmv_cblas_call.ap = ap;
    g_dspmv_cblas_call.x = x;
    g_dspmv_cblas_call.y = y;
    g_dspmv_cblas_call.alpha_value = alpha;
    g_dspmv_cblas_call.beta_value = beta;
    y[0] = 231.0;
    y[1] = 232.0;
}

static int check_ssbmv_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_ssbmv_cblas_fn thunk = NULL;
    float a[6] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
    float x[2] = { 7.0f, 8.0f };
    float y[2] = { 9.0f, 10.0f };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_ssbmv_fortran_call, 0, sizeof(g_ssbmv_fortran_call));
    vtable.ext_ops[FB_OP_SSBMV][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_ssbmv_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SSBMV);
    thunk = (fb_ssbmv_cblas_fn)vtable.ext_ops[FB_OP_SSBMV][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SSBMV Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    thunk(FB_LAYOUT_ROW_MAJOR, FB_UPPER, 2, 1, 1.5f, a, 2, x, 1, 0.5f, y, 1);
    if (g_ssbmv_fortran_call.called != 1 ||
        g_ssbmv_fortran_call.order != FB_LAYOUT_ROW_MAJOR ||
        g_ssbmv_fortran_call.uplo != FB_UPPER ||
        g_ssbmv_fortran_call.n != 2 ||
        g_ssbmv_fortran_call.k != 1 ||
        g_ssbmv_fortran_call.lda != 2 ||
        g_ssbmv_fortran_call.incx != 1 ||
        g_ssbmv_fortran_call.incy != 1 ||
        g_ssbmv_fortran_call.a != a ||
        g_ssbmv_fortran_call.x != x ||
        g_ssbmv_fortran_call.y != y ||
        g_ssbmv_fortran_call.alpha_value != 1.5f ||
        g_ssbmv_fortran_call.beta_value != 0.5f ||
        y[0] != 101.0f || y[1] != 102.0f) {
        fprintf(stderr, "[FAIL] SSBMV Fortran->CBLAS thunk did not forward symmetric band mat-vec arguments correctly\n");
        return 1;
    }

    printf("[PASS] SSBMV Fortran->CBLAS thunk forwards symmetric band mat-vec arguments unchanged\n");
    return 0;
}

static int check_ssbmv_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_ssbmv_fortran_slot_fn thunk = NULL;
    int order = FB_LAYOUT_COL_MAJOR;
    int uplo = FB_LOWER;
    int n = 2;
    int k = 1;
    int lda = 2;
    int incx = 1;
    int incy = 1;
    float alpha = 2.5f;
    float beta = 1.5f;
    float a[6] = { 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f };
    float x[2] = { 17.0f, 18.0f };
    float y[2] = { 19.0f, 20.0f };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_ssbmv_cblas_call, 0, sizeof(g_ssbmv_cblas_call));
    vtable.ext_ops[FB_OP_SSBMV][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_ssbmv_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SSBMV);
    thunk = (fb_ssbmv_fortran_slot_fn)vtable.ext_ops[FB_OP_SSBMV][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SSBMV CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&order, &uplo, &n, &k, &alpha, a, &lda, x, &incx, &beta, y, &incy);
    if (g_ssbmv_cblas_call.called != 1 ||
        g_ssbmv_cblas_call.order != FB_LAYOUT_COL_MAJOR ||
        g_ssbmv_cblas_call.uplo != FB_LOWER ||
        g_ssbmv_cblas_call.n != 2 ||
        g_ssbmv_cblas_call.k != 1 ||
        g_ssbmv_cblas_call.lda != 2 ||
        g_ssbmv_cblas_call.incx != 1 ||
        g_ssbmv_cblas_call.incy != 1 ||
        g_ssbmv_cblas_call.a != a ||
        g_ssbmv_cblas_call.x != x ||
        g_ssbmv_cblas_call.y != y ||
        g_ssbmv_cblas_call.alpha_value != 2.5f ||
        g_ssbmv_cblas_call.beta_value != 1.5f ||
        y[0] != 111.0f || y[1] != 112.0f) {
        fprintf(stderr, "[FAIL] SSBMV CBLAS->Fortran thunk did not map symmetric band mat-vec arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] SSBMV CBLAS->Fortran thunk maps symmetric band mat-vec arguments into the C entry\n");
    return 0;
}

static int check_dsbmv_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_dsbmv_cblas_fn thunk = NULL;
    double a[6] = { 21.0, 22.0, 23.0, 24.0, 25.0, 26.0 };
    double x[2] = { 27.0, 28.0 };
    double y[2] = { 29.0, 30.0 };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dsbmv_fortran_call, 0, sizeof(g_dsbmv_fortran_call));
    vtable.ext_ops[FB_OP_DSBMV][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_dsbmv_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_DSBMV);
    thunk = (fb_dsbmv_cblas_fn)vtable.ext_ops[FB_OP_DSBMV][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DSBMV Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    thunk(FB_LAYOUT_ROW_MAJOR, FB_UPPER, 2, 1, 3.5, a, 2, x, 1, 2.5, y, 1);
    if (g_dsbmv_fortran_call.called != 1 ||
        g_dsbmv_fortran_call.order != FB_LAYOUT_ROW_MAJOR ||
        g_dsbmv_fortran_call.uplo != FB_UPPER ||
        g_dsbmv_fortran_call.n != 2 ||
        g_dsbmv_fortran_call.k != 1 ||
        g_dsbmv_fortran_call.lda != 2 ||
        g_dsbmv_fortran_call.incx != 1 ||
        g_dsbmv_fortran_call.incy != 1 ||
        g_dsbmv_fortran_call.a != a ||
        g_dsbmv_fortran_call.x != x ||
        g_dsbmv_fortran_call.y != y ||
        g_dsbmv_fortran_call.alpha_value != 3.5 ||
        g_dsbmv_fortran_call.beta_value != 2.5 ||
        y[0] != 121.0 || y[1] != 122.0) {
        fprintf(stderr, "[FAIL] DSBMV Fortran->CBLAS thunk did not forward double symmetric band mat-vec arguments correctly\n");
        return 1;
    }

    printf("[PASS] DSBMV Fortran->CBLAS thunk forwards double symmetric band mat-vec arguments unchanged\n");
    return 0;
}

static int check_dsbmv_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_dsbmv_fortran_slot_fn thunk = NULL;
    int order = FB_LAYOUT_COL_MAJOR;
    int uplo = FB_LOWER;
    int n = 2;
    int k = 1;
    int lda = 2;
    int incx = 1;
    int incy = 1;
    double alpha = 4.5;
    double beta = 3.5;
    double a[6] = { 31.0, 32.0, 33.0, 34.0, 35.0, 36.0 };
    double x[2] = { 37.0, 38.0 };
    double y[2] = { 39.0, 40.0 };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dsbmv_cblas_call, 0, sizeof(g_dsbmv_cblas_call));
    vtable.ext_ops[FB_OP_DSBMV][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_dsbmv_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_DSBMV);
    thunk = (fb_dsbmv_fortran_slot_fn)vtable.ext_ops[FB_OP_DSBMV][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DSBMV CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&order, &uplo, &n, &k, &alpha, a, &lda, x, &incx, &beta, y, &incy);
    if (g_dsbmv_cblas_call.called != 1 ||
        g_dsbmv_cblas_call.order != FB_LAYOUT_COL_MAJOR ||
        g_dsbmv_cblas_call.uplo != FB_LOWER ||
        g_dsbmv_cblas_call.n != 2 ||
        g_dsbmv_cblas_call.k != 1 ||
        g_dsbmv_cblas_call.lda != 2 ||
        g_dsbmv_cblas_call.incx != 1 ||
        g_dsbmv_cblas_call.incy != 1 ||
        g_dsbmv_cblas_call.a != a ||
        g_dsbmv_cblas_call.x != x ||
        g_dsbmv_cblas_call.y != y ||
        g_dsbmv_cblas_call.alpha_value != 4.5 ||
        g_dsbmv_cblas_call.beta_value != 3.5 ||
        y[0] != 131.0 || y[1] != 132.0) {
        fprintf(stderr, "[FAIL] DSBMV CBLAS->Fortran thunk did not map double symmetric band mat-vec arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] DSBMV CBLAS->Fortran thunk maps double symmetric band mat-vec arguments into the C entry\n");
    return 0;
}

static int check_sspmv_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_sspmv_cblas_fn thunk = NULL;
    float ap[3] = { 41.0f, 42.0f, 43.0f };
    float x[2] = { 44.0f, 45.0f };
    float y[2] = { 46.0f, 47.0f };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sspmv_fortran_call, 0, sizeof(g_sspmv_fortran_call));
    vtable.ext_ops[FB_OP_SSPMV][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_sspmv_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SSPMV);
    thunk = (fb_sspmv_cblas_fn)vtable.ext_ops[FB_OP_SSPMV][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SSPMV Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    thunk(FB_LAYOUT_ROW_MAJOR, FB_UPPER, 2, 5.5f, ap, x, 1, 4.5f, y, 1);
    if (g_sspmv_fortran_call.called != 1 ||
        g_sspmv_fortran_call.order != FB_LAYOUT_ROW_MAJOR ||
        g_sspmv_fortran_call.uplo != FB_UPPER ||
        g_sspmv_fortran_call.n != 2 ||
        g_sspmv_fortran_call.incx != 1 ||
        g_sspmv_fortran_call.incy != 1 ||
        g_sspmv_fortran_call.ap != ap ||
        g_sspmv_fortran_call.x != x ||
        g_sspmv_fortran_call.y != y ||
        g_sspmv_fortran_call.alpha_value != 5.5f ||
        g_sspmv_fortran_call.beta_value != 4.5f ||
        y[0] != 201.0f || y[1] != 202.0f) {
        fprintf(stderr, "[FAIL] SSPMV Fortran->CBLAS thunk did not forward symmetric packed mat-vec arguments correctly\n");
        return 1;
    }

    printf("[PASS] SSPMV Fortran->CBLAS thunk forwards symmetric packed mat-vec arguments unchanged\n");
    return 0;
}

static int check_sspmv_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_sspmv_fortran_slot_fn thunk = NULL;
    int order = FB_LAYOUT_COL_MAJOR;
    int uplo = FB_LOWER;
    int n = 2;
    int incx = 1;
    int incy = 1;
    float alpha = 6.5f;
    float beta = 5.5f;
    float ap[3] = { 48.0f, 49.0f, 50.0f };
    float x[2] = { 51.0f, 52.0f };
    float y[2] = { 53.0f, 54.0f };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sspmv_cblas_call, 0, sizeof(g_sspmv_cblas_call));
    vtable.ext_ops[FB_OP_SSPMV][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_sspmv_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SSPMV);
    thunk = (fb_sspmv_fortran_slot_fn)vtable.ext_ops[FB_OP_SSPMV][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SSPMV CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&order, &uplo, &n, &alpha, ap, x, &incx, &beta, y, &incy);
    if (g_sspmv_cblas_call.called != 1 ||
        g_sspmv_cblas_call.order != FB_LAYOUT_COL_MAJOR ||
        g_sspmv_cblas_call.uplo != FB_LOWER ||
        g_sspmv_cblas_call.n != 2 ||
        g_sspmv_cblas_call.incx != 1 ||
        g_sspmv_cblas_call.incy != 1 ||
        g_sspmv_cblas_call.ap != ap ||
        g_sspmv_cblas_call.x != x ||
        g_sspmv_cblas_call.y != y ||
        g_sspmv_cblas_call.alpha_value != 6.5f ||
        g_sspmv_cblas_call.beta_value != 5.5f ||
        y[0] != 211.0f || y[1] != 212.0f) {
        fprintf(stderr, "[FAIL] SSPMV CBLAS->Fortran thunk did not map symmetric packed mat-vec arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] SSPMV CBLAS->Fortran thunk maps symmetric packed mat-vec arguments into the C entry\n");
    return 0;
}

static int check_dspmv_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_dspmv_cblas_fn thunk = NULL;
    double ap[3] = { 55.0, 56.0, 57.0 };
    double x[2] = { 58.0, 59.0 };
    double y[2] = { 60.0, 61.0 };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dspmv_fortran_call, 0, sizeof(g_dspmv_fortran_call));
    vtable.ext_ops[FB_OP_DSPMV][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_dspmv_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_DSPMV);
    thunk = (fb_dspmv_cblas_fn)vtable.ext_ops[FB_OP_DSPMV][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DSPMV Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    thunk(FB_LAYOUT_ROW_MAJOR, FB_UPPER, 2, 7.5, ap, x, 1, 6.5, y, 1);
    if (g_dspmv_fortran_call.called != 1 ||
        g_dspmv_fortran_call.order != FB_LAYOUT_ROW_MAJOR ||
        g_dspmv_fortran_call.uplo != FB_UPPER ||
        g_dspmv_fortran_call.n != 2 ||
        g_dspmv_fortran_call.incx != 1 ||
        g_dspmv_fortran_call.incy != 1 ||
        g_dspmv_fortran_call.ap != ap ||
        g_dspmv_fortran_call.x != x ||
        g_dspmv_fortran_call.y != y ||
        g_dspmv_fortran_call.alpha_value != 7.5 ||
        g_dspmv_fortran_call.beta_value != 6.5 ||
        y[0] != 221.0 || y[1] != 222.0) {
        fprintf(stderr, "[FAIL] DSPMV Fortran->CBLAS thunk did not forward double symmetric packed mat-vec arguments correctly\n");
        return 1;
    }

    printf("[PASS] DSPMV Fortran->CBLAS thunk forwards double symmetric packed mat-vec arguments unchanged\n");
    return 0;
}

static int check_dspmv_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_dspmv_fortran_slot_fn thunk = NULL;
    int order = FB_LAYOUT_COL_MAJOR;
    int uplo = FB_LOWER;
    int n = 2;
    int incx = 1;
    int incy = 1;
    double alpha = 8.5;
    double beta = 7.5;
    double ap[3] = { 62.0, 63.0, 64.0 };
    double x[2] = { 65.0, 66.0 };
    double y[2] = { 67.0, 68.0 };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dspmv_cblas_call, 0, sizeof(g_dspmv_cblas_call));
    vtable.ext_ops[FB_OP_DSPMV][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_dspmv_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_DSPMV);
    thunk = (fb_dspmv_fortran_slot_fn)vtable.ext_ops[FB_OP_DSPMV][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DSPMV CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&order, &uplo, &n, &alpha, ap, x, &incx, &beta, y, &incy);
    if (g_dspmv_cblas_call.called != 1 ||
        g_dspmv_cblas_call.order != FB_LAYOUT_COL_MAJOR ||
        g_dspmv_cblas_call.uplo != FB_LOWER ||
        g_dspmv_cblas_call.n != 2 ||
        g_dspmv_cblas_call.incx != 1 ||
        g_dspmv_cblas_call.incy != 1 ||
        g_dspmv_cblas_call.ap != ap ||
        g_dspmv_cblas_call.x != x ||
        g_dspmv_cblas_call.y != y ||
        g_dspmv_cblas_call.alpha_value != 8.5 ||
        g_dspmv_cblas_call.beta_value != 7.5 ||
        y[0] != 231.0 || y[1] != 232.0) {
        fprintf(stderr, "[FAIL] DSPMV CBLAS->Fortran thunk did not map double symmetric packed mat-vec arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] DSPMV CBLAS->Fortran thunk maps double symmetric packed mat-vec arguments into the C entry\n");
    return 0;
}

int main(void)
{
    int status = 0;

    status |= check_ssbmv_fortran_to_cblas();
    status |= check_ssbmv_cblas_to_fortran();
    status |= check_dsbmv_fortran_to_cblas();
    status |= check_dsbmv_cblas_to_fortran();
    status |= check_sspmv_fortran_to_cblas();
    status |= check_sspmv_cblas_to_fortran();
    status |= check_dspmv_fortran_to_cblas();
    status |= check_dspmv_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}