#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef void (*fb_sspr_cblas_fn)(fb_layout_t layout, fb_uplo_t uplo, int n,
                                 float alpha, const float *x, int incx,
                                 float *ap);
typedef void (*fb_dspr_cblas_fn)(fb_layout_t layout, fb_uplo_t uplo, int n,
                                 double alpha, const double *x, int incx,
                                 double *ap);
typedef void (*fb_sspr_fortran_slot_fn)(const int *order, const int *uplo,
                                        const int *n, const float *alpha,
                                        const float *x, const int *incx,
                                        float *ap);
typedef void (*fb_dspr_fortran_slot_fn)(const int *order, const int *uplo,
                                        const int *n, const double *alpha,
                                        const double *x, const int *incx,
                                        double *ap);

typedef void (*fb_sspr2_cblas_fn)(fb_layout_t layout, fb_uplo_t uplo, int n,
                                  float alpha, const float *x, int incx,
                                  const float *y, int incy, float *ap);
typedef void (*fb_dspr2_cblas_fn)(fb_layout_t layout, fb_uplo_t uplo, int n,
                                  double alpha, const double *x, int incx,
                                  const double *y, int incy, double *ap);
typedef void (*fb_sspr2_fortran_slot_fn)(const int *order, const int *uplo,
                                         const int *n, const float *alpha,
                                         const float *x, const int *incx,
                                         const float *y, const int *incy,
                                         float *ap);
typedef void (*fb_dspr2_fortran_slot_fn)(const int *order, const int *uplo,
                                         const int *n, const double *alpha,
                                         const double *x, const int *incx,
                                         const double *y, const int *incy,
                                         double *ap);

typedef struct {
    int called;
    int order;
    int uplo;
    int n;
    int incx;
    const void *x;
    void *ap;
    float alpha_value;
} spr_call_f32_t;

typedef struct {
    int called;
    int order;
    int uplo;
    int n;
    int incx;
    const void *x;
    void *ap;
    double alpha_value;
} spr_call_f64_t;

typedef struct {
    int called;
    int order;
    int uplo;
    int n;
    int incx;
    int incy;
    const void *x;
    const void *y;
    void *ap;
    float alpha_value;
} spr2_call_f32_t;

typedef struct {
    int called;
    int order;
    int uplo;
    int n;
    int incx;
    int incy;
    const void *x;
    const void *y;
    void *ap;
    double alpha_value;
} spr2_call_f64_t;

static spr_call_f32_t g_sspr_fortran_call;
static spr_call_f32_t g_sspr_cblas_call;
static spr_call_f64_t g_dspr_fortran_call;
static spr_call_f64_t g_dspr_cblas_call;
static spr2_call_f32_t g_sspr2_fortran_call;
static spr2_call_f32_t g_sspr2_cblas_call;
static spr2_call_f64_t g_dspr2_fortran_call;
static spr2_call_f64_t g_dspr2_cblas_call;

static void stub_sspr_fortran(const int *order, const int *uplo, const int *n,
                              const float *alpha, const float *x,
                              const int *incx, float *ap)
{
    g_sspr_fortran_call.called += 1;
    g_sspr_fortran_call.order = *order;
    g_sspr_fortran_call.uplo = *uplo;
    g_sspr_fortran_call.n = *n;
    g_sspr_fortran_call.incx = *incx;
    g_sspr_fortran_call.x = x;
    g_sspr_fortran_call.ap = ap;
    g_sspr_fortran_call.alpha_value = *alpha;
    ap[0] = 101.0f;
    ap[1] = 102.0f;
}

static void stub_sspr_cblas(fb_layout_t layout, fb_uplo_t uplo, int n,
                            float alpha, const float *x, int incx, float *ap)
{
    g_sspr_cblas_call.called += 1;
    g_sspr_cblas_call.order = layout;
    g_sspr_cblas_call.uplo = uplo;
    g_sspr_cblas_call.n = n;
    g_sspr_cblas_call.incx = incx;
    g_sspr_cblas_call.x = x;
    g_sspr_cblas_call.ap = ap;
    g_sspr_cblas_call.alpha_value = alpha;
    ap[0] = 111.0f;
    ap[1] = 112.0f;
}

static void stub_dspr_fortran(const int *order, const int *uplo, const int *n,
                              const double *alpha, const double *x,
                              const int *incx, double *ap)
{
    g_dspr_fortran_call.called += 1;
    g_dspr_fortran_call.order = *order;
    g_dspr_fortran_call.uplo = *uplo;
    g_dspr_fortran_call.n = *n;
    g_dspr_fortran_call.incx = *incx;
    g_dspr_fortran_call.x = x;
    g_dspr_fortran_call.ap = ap;
    g_dspr_fortran_call.alpha_value = *alpha;
    ap[0] = 121.0;
    ap[1] = 122.0;
}

static void stub_dspr_cblas(fb_layout_t layout, fb_uplo_t uplo, int n,
                            double alpha, const double *x, int incx,
                            double *ap)
{
    g_dspr_cblas_call.called += 1;
    g_dspr_cblas_call.order = layout;
    g_dspr_cblas_call.uplo = uplo;
    g_dspr_cblas_call.n = n;
    g_dspr_cblas_call.incx = incx;
    g_dspr_cblas_call.x = x;
    g_dspr_cblas_call.ap = ap;
    g_dspr_cblas_call.alpha_value = alpha;
    ap[0] = 131.0;
    ap[1] = 132.0;
}

static void stub_sspr2_fortran(const int *order, const int *uplo,
                               const int *n, const float *alpha,
                               const float *x, const int *incx,
                               const float *y, const int *incy, float *ap)
{
    g_sspr2_fortran_call.called += 1;
    g_sspr2_fortran_call.order = *order;
    g_sspr2_fortran_call.uplo = *uplo;
    g_sspr2_fortran_call.n = *n;
    g_sspr2_fortran_call.incx = *incx;
    g_sspr2_fortran_call.incy = *incy;
    g_sspr2_fortran_call.x = x;
    g_sspr2_fortran_call.y = y;
    g_sspr2_fortran_call.ap = ap;
    g_sspr2_fortran_call.alpha_value = *alpha;
    ap[0] = 201.0f;
    ap[1] = 202.0f;
}

static void stub_sspr2_cblas(fb_layout_t layout, fb_uplo_t uplo, int n,
                             float alpha, const float *x, int incx,
                             const float *y, int incy, float *ap)
{
    g_sspr2_cblas_call.called += 1;
    g_sspr2_cblas_call.order = layout;
    g_sspr2_cblas_call.uplo = uplo;
    g_sspr2_cblas_call.n = n;
    g_sspr2_cblas_call.incx = incx;
    g_sspr2_cblas_call.incy = incy;
    g_sspr2_cblas_call.x = x;
    g_sspr2_cblas_call.y = y;
    g_sspr2_cblas_call.ap = ap;
    g_sspr2_cblas_call.alpha_value = alpha;
    ap[0] = 211.0f;
    ap[1] = 212.0f;
}

static void stub_dspr2_fortran(const int *order, const int *uplo,
                               const int *n, const double *alpha,
                               const double *x, const int *incx,
                               const double *y, const int *incy, double *ap)
{
    g_dspr2_fortran_call.called += 1;
    g_dspr2_fortran_call.order = *order;
    g_dspr2_fortran_call.uplo = *uplo;
    g_dspr2_fortran_call.n = *n;
    g_dspr2_fortran_call.incx = *incx;
    g_dspr2_fortran_call.incy = *incy;
    g_dspr2_fortran_call.x = x;
    g_dspr2_fortran_call.y = y;
    g_dspr2_fortran_call.ap = ap;
    g_dspr2_fortran_call.alpha_value = *alpha;
    ap[0] = 221.0;
    ap[1] = 222.0;
}

static void stub_dspr2_cblas(fb_layout_t layout, fb_uplo_t uplo, int n,
                             double alpha, const double *x, int incx,
                             const double *y, int incy, double *ap)
{
    g_dspr2_cblas_call.called += 1;
    g_dspr2_cblas_call.order = layout;
    g_dspr2_cblas_call.uplo = uplo;
    g_dspr2_cblas_call.n = n;
    g_dspr2_cblas_call.incx = incx;
    g_dspr2_cblas_call.incy = incy;
    g_dspr2_cblas_call.x = x;
    g_dspr2_cblas_call.y = y;
    g_dspr2_cblas_call.ap = ap;
    g_dspr2_cblas_call.alpha_value = alpha;
    ap[0] = 231.0;
    ap[1] = 232.0;
}

static int check_sspr_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_sspr_cblas_fn thunk = NULL;
    float x[2] = { 1.0f, 2.0f };
    float ap[3] = { 3.0f, 4.0f, 5.0f };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sspr_fortran_call, 0, sizeof(g_sspr_fortran_call));
    vtable.ext_ops[FB_OP_SSPR][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_sspr_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SSPR);
    thunk = (fb_sspr_cblas_fn)vtable.ext_ops[FB_OP_SSPR][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SSPR Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    thunk(FB_LAYOUT_ROW_MAJOR, FB_UPPER, 2, 1.5f, x, 1, ap);
    if (g_sspr_fortran_call.called != 1 ||
        g_sspr_fortran_call.order != FB_LAYOUT_ROW_MAJOR ||
        g_sspr_fortran_call.uplo != FB_UPPER ||
        g_sspr_fortran_call.n != 2 ||
        g_sspr_fortran_call.incx != 1 ||
        g_sspr_fortran_call.x != x ||
        g_sspr_fortran_call.ap != ap ||
        g_sspr_fortran_call.alpha_value != 1.5f ||
        ap[0] != 101.0f || ap[1] != 102.0f) {
        fprintf(stderr, "[FAIL] SSPR Fortran->CBLAS thunk did not forward symmetric packed rank-1 update arguments correctly\n");
        return 1;
    }

    printf("[PASS] SSPR Fortran->CBLAS thunk forwards symmetric packed rank-1 update arguments unchanged\n");
    return 0;
}

static int check_sspr_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_sspr_fortran_slot_fn thunk = NULL;
    int order = FB_LAYOUT_COL_MAJOR;
    int uplo = FB_LOWER;
    int n = 2;
    int incx = 1;
    float alpha = 2.5f;
    float x[2] = { 6.0f, 7.0f };
    float ap[3] = { 8.0f, 9.0f, 10.0f };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sspr_cblas_call, 0, sizeof(g_sspr_cblas_call));
    vtable.ext_ops[FB_OP_SSPR][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_sspr_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SSPR);
    thunk = (fb_sspr_fortran_slot_fn)vtable.ext_ops[FB_OP_SSPR][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SSPR CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&order, &uplo, &n, &alpha, x, &incx, ap);
    if (g_sspr_cblas_call.called != 1 ||
        g_sspr_cblas_call.order != FB_LAYOUT_COL_MAJOR ||
        g_sspr_cblas_call.uplo != FB_LOWER ||
        g_sspr_cblas_call.n != 2 ||
        g_sspr_cblas_call.incx != 1 ||
        g_sspr_cblas_call.x != x ||
        g_sspr_cblas_call.ap != ap ||
        g_sspr_cblas_call.alpha_value != 2.5f ||
        ap[0] != 111.0f || ap[1] != 112.0f) {
        fprintf(stderr, "[FAIL] SSPR CBLAS->Fortran thunk did not map symmetric packed rank-1 update arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] SSPR CBLAS->Fortran thunk maps symmetric packed rank-1 update arguments into the C entry\n");
    return 0;
}

static int check_dspr_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_dspr_cblas_fn thunk = NULL;
    double x[2] = { 11.0, 12.0 };
    double ap[3] = { 13.0, 14.0, 15.0 };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dspr_fortran_call, 0, sizeof(g_dspr_fortran_call));
    vtable.ext_ops[FB_OP_DSPR][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_dspr_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_DSPR);
    thunk = (fb_dspr_cblas_fn)vtable.ext_ops[FB_OP_DSPR][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DSPR Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    thunk(FB_LAYOUT_ROW_MAJOR, FB_UPPER, 2, 3.5, x, 1, ap);
    if (g_dspr_fortran_call.called != 1 ||
        g_dspr_fortran_call.order != FB_LAYOUT_ROW_MAJOR ||
        g_dspr_fortran_call.uplo != FB_UPPER ||
        g_dspr_fortran_call.n != 2 ||
        g_dspr_fortran_call.incx != 1 ||
        g_dspr_fortran_call.x != x ||
        g_dspr_fortran_call.ap != ap ||
        g_dspr_fortran_call.alpha_value != 3.5 ||
        ap[0] != 121.0 || ap[1] != 122.0) {
        fprintf(stderr, "[FAIL] DSPR Fortran->CBLAS thunk did not forward double symmetric packed rank-1 update arguments correctly\n");
        return 1;
    }

    printf("[PASS] DSPR Fortran->CBLAS thunk forwards double symmetric packed rank-1 update arguments unchanged\n");
    return 0;
}

static int check_dspr_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_dspr_fortran_slot_fn thunk = NULL;
    int order = FB_LAYOUT_COL_MAJOR;
    int uplo = FB_LOWER;
    int n = 2;
    int incx = 1;
    double alpha = 4.5;
    double x[2] = { 16.0, 17.0 };
    double ap[3] = { 18.0, 19.0, 20.0 };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dspr_cblas_call, 0, sizeof(g_dspr_cblas_call));
    vtable.ext_ops[FB_OP_DSPR][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_dspr_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_DSPR);
    thunk = (fb_dspr_fortran_slot_fn)vtable.ext_ops[FB_OP_DSPR][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DSPR CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&order, &uplo, &n, &alpha, x, &incx, ap);
    if (g_dspr_cblas_call.called != 1 ||
        g_dspr_cblas_call.order != FB_LAYOUT_COL_MAJOR ||
        g_dspr_cblas_call.uplo != FB_LOWER ||
        g_dspr_cblas_call.n != 2 ||
        g_dspr_cblas_call.incx != 1 ||
        g_dspr_cblas_call.x != x ||
        g_dspr_cblas_call.ap != ap ||
        g_dspr_cblas_call.alpha_value != 4.5 ||
        ap[0] != 131.0 || ap[1] != 132.0) {
        fprintf(stderr, "[FAIL] DSPR CBLAS->Fortran thunk did not map double symmetric packed rank-1 update arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] DSPR CBLAS->Fortran thunk maps double symmetric packed rank-1 update arguments into the C entry\n");
    return 0;
}

static int check_sspr2_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_sspr2_cblas_fn thunk = NULL;
    float x[2] = { 21.0f, 22.0f };
    float y[2] = { 23.0f, 24.0f };
    float ap[3] = { 25.0f, 26.0f, 27.0f };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sspr2_fortran_call, 0, sizeof(g_sspr2_fortran_call));
    vtable.ext_ops[FB_OP_SSPR2][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_sspr2_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SSPR2);
    thunk = (fb_sspr2_cblas_fn)vtable.ext_ops[FB_OP_SSPR2][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SSPR2 Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    thunk(FB_LAYOUT_ROW_MAJOR, FB_UPPER, 2, 5.5f, x, 1, y, 1, ap);
    if (g_sspr2_fortran_call.called != 1 ||
        g_sspr2_fortran_call.order != FB_LAYOUT_ROW_MAJOR ||
        g_sspr2_fortran_call.uplo != FB_UPPER ||
        g_sspr2_fortran_call.n != 2 ||
        g_sspr2_fortran_call.incx != 1 ||
        g_sspr2_fortran_call.incy != 1 ||
        g_sspr2_fortran_call.x != x ||
        g_sspr2_fortran_call.y != y ||
        g_sspr2_fortran_call.ap != ap ||
        g_sspr2_fortran_call.alpha_value != 5.5f ||
        ap[0] != 201.0f || ap[1] != 202.0f) {
        fprintf(stderr, "[FAIL] SSPR2 Fortran->CBLAS thunk did not forward symmetric packed rank-2 update arguments correctly\n");
        return 1;
    }

    printf("[PASS] SSPR2 Fortran->CBLAS thunk forwards symmetric packed rank-2 update arguments unchanged\n");
    return 0;
}

static int check_sspr2_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_sspr2_fortran_slot_fn thunk = NULL;
    int order = FB_LAYOUT_COL_MAJOR;
    int uplo = FB_LOWER;
    int n = 2;
    int incx = 1;
    int incy = 1;
    float alpha = 6.5f;
    float x[2] = { 28.0f, 29.0f };
    float y[2] = { 30.0f, 31.0f };
    float ap[3] = { 32.0f, 33.0f, 34.0f };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sspr2_cblas_call, 0, sizeof(g_sspr2_cblas_call));
    vtable.ext_ops[FB_OP_SSPR2][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_sspr2_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SSPR2);
    thunk = (fb_sspr2_fortran_slot_fn)vtable.ext_ops[FB_OP_SSPR2][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SSPR2 CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&order, &uplo, &n, &alpha, x, &incx, y, &incy, ap);
    if (g_sspr2_cblas_call.called != 1 ||
        g_sspr2_cblas_call.order != FB_LAYOUT_COL_MAJOR ||
        g_sspr2_cblas_call.uplo != FB_LOWER ||
        g_sspr2_cblas_call.n != 2 ||
        g_sspr2_cblas_call.incx != 1 ||
        g_sspr2_cblas_call.incy != 1 ||
        g_sspr2_cblas_call.x != x ||
        g_sspr2_cblas_call.y != y ||
        g_sspr2_cblas_call.ap != ap ||
        g_sspr2_cblas_call.alpha_value != 6.5f ||
        ap[0] != 211.0f || ap[1] != 212.0f) {
        fprintf(stderr, "[FAIL] SSPR2 CBLAS->Fortran thunk did not map symmetric packed rank-2 update arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] SSPR2 CBLAS->Fortran thunk maps symmetric packed rank-2 update arguments into the C entry\n");
    return 0;
}

static int check_dspr2_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_dspr2_cblas_fn thunk = NULL;
    double x[2] = { 35.0, 36.0 };
    double y[2] = { 37.0, 38.0 };
    double ap[3] = { 39.0, 40.0, 41.0 };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dspr2_fortran_call, 0, sizeof(g_dspr2_fortran_call));
    vtable.ext_ops[FB_OP_DSPR2][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_dspr2_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_DSPR2);
    thunk = (fb_dspr2_cblas_fn)vtable.ext_ops[FB_OP_DSPR2][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DSPR2 Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    thunk(FB_LAYOUT_ROW_MAJOR, FB_UPPER, 2, 7.5, x, 1, y, 1, ap);
    if (g_dspr2_fortran_call.called != 1 ||
        g_dspr2_fortran_call.order != FB_LAYOUT_ROW_MAJOR ||
        g_dspr2_fortran_call.uplo != FB_UPPER ||
        g_dspr2_fortran_call.n != 2 ||
        g_dspr2_fortran_call.incx != 1 ||
        g_dspr2_fortran_call.incy != 1 ||
        g_dspr2_fortran_call.x != x ||
        g_dspr2_fortran_call.y != y ||
        g_dspr2_fortran_call.ap != ap ||
        g_dspr2_fortran_call.alpha_value != 7.5 ||
        ap[0] != 221.0 || ap[1] != 222.0) {
        fprintf(stderr, "[FAIL] DSPR2 Fortran->CBLAS thunk did not forward double symmetric packed rank-2 update arguments correctly\n");
        return 1;
    }

    printf("[PASS] DSPR2 Fortran->CBLAS thunk forwards double symmetric packed rank-2 update arguments unchanged\n");
    return 0;
}

static int check_dspr2_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_dspr2_fortran_slot_fn thunk = NULL;
    int order = FB_LAYOUT_COL_MAJOR;
    int uplo = FB_LOWER;
    int n = 2;
    int incx = 1;
    int incy = 1;
    double alpha = 8.5;
    double x[2] = { 42.0, 43.0 };
    double y[2] = { 44.0, 45.0 };
    double ap[3] = { 46.0, 47.0, 48.0 };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dspr2_cblas_call, 0, sizeof(g_dspr2_cblas_call));
    vtable.ext_ops[FB_OP_DSPR2][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_dspr2_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_DSPR2);
    thunk = (fb_dspr2_fortran_slot_fn)vtable.ext_ops[FB_OP_DSPR2][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DSPR2 CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&order, &uplo, &n, &alpha, x, &incx, y, &incy, ap);
    if (g_dspr2_cblas_call.called != 1 ||
        g_dspr2_cblas_call.order != FB_LAYOUT_COL_MAJOR ||
        g_dspr2_cblas_call.uplo != FB_LOWER ||
        g_dspr2_cblas_call.n != 2 ||
        g_dspr2_cblas_call.incx != 1 ||
        g_dspr2_cblas_call.incy != 1 ||
        g_dspr2_cblas_call.x != x ||
        g_dspr2_cblas_call.y != y ||
        g_dspr2_cblas_call.ap != ap ||
        g_dspr2_cblas_call.alpha_value != 8.5 ||
        ap[0] != 231.0 || ap[1] != 232.0) {
        fprintf(stderr, "[FAIL] DSPR2 CBLAS->Fortran thunk did not map double symmetric packed rank-2 update arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] DSPR2 CBLAS->Fortran thunk maps double symmetric packed rank-2 update arguments into the C entry\n");
    return 0;
}

int main(void)
{
    int status = 0;

    status |= check_sspr_fortran_to_cblas();
    status |= check_sspr_cblas_to_fortran();
    status |= check_dspr_fortran_to_cblas();
    status |= check_dspr_cblas_to_fortran();
    status |= check_sspr2_fortran_to_cblas();
    status |= check_sspr2_cblas_to_fortran();
    status |= check_dspr2_fortran_to_cblas();
    status |= check_dspr2_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}