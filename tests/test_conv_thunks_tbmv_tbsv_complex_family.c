#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef void (*fb_ctbmv_cblas_fn)(fb_layout_t layout, fb_uplo_t uplo,
                                  fb_transpose_t trans, fb_diag_t diag,
                                  int n, int k, const void *a, int lda,
                                  void *x, int incx);
typedef void (*fb_ztbmv_cblas_fn)(fb_layout_t layout, fb_uplo_t uplo,
                                  fb_transpose_t trans, fb_diag_t diag,
                                  int n, int k, const void *a, int lda,
                                  void *x, int incx);
typedef void (*fb_ctbmv_fortran_slot_fn)(const int *order, const int *uplo,
                                         const int *trans, const int *diag,
                                         const int *n, const int *k,
                                         const fb_complex_float_t *a,
                                         const int *lda, fb_complex_float_t *x,
                                         const int *incx);
typedef void (*fb_ztbmv_fortran_slot_fn)(const int *order, const int *uplo,
                                         const int *trans, const int *diag,
                                         const int *n, const int *k,
                                         const fb_complex_double_t *a,
                                         const int *lda, fb_complex_double_t *x,
                                         const int *incx);
typedef fb_ctbmv_cblas_fn fb_ctbsv_cblas_fn;
typedef fb_ztbmv_cblas_fn fb_ztbsv_cblas_fn;
typedef fb_ctbmv_fortran_slot_fn fb_ctbsv_fortran_slot_fn;
typedef fb_ztbmv_fortran_slot_fn fb_ztbsv_fortran_slot_fn;

typedef struct {
    int called;
    int order;
    int uplo;
    int trans;
    int diag;
    int n;
    int k;
    int lda;
    int incx;
    const void *a;
    void *x;
} band_call_t;

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

static band_call_t g_ctbmv_fortran_call;
static band_call_t g_ctbmv_cblas_call;
static band_call_t g_ztbmv_fortran_call;
static band_call_t g_ztbmv_cblas_call;
static band_call_t g_ctbsv_fortran_call;
static band_call_t g_ctbsv_cblas_call;
static band_call_t g_ztbsv_fortran_call;
static band_call_t g_ztbsv_cblas_call;

static void stub_ctbmv_fortran(const int *order, const int *uplo,
                               const int *trans, const int *diag,
                               const int *n, const int *k,
                               const fb_complex_float_t *a, const int *lda,
                               fb_complex_float_t *x, const int *incx)
{
    g_ctbmv_fortran_call.called += 1;
    g_ctbmv_fortran_call.order = *order;
    g_ctbmv_fortran_call.uplo = *uplo;
    g_ctbmv_fortran_call.trans = *trans;
    g_ctbmv_fortran_call.diag = *diag;
    g_ctbmv_fortran_call.n = *n;
    g_ctbmv_fortran_call.k = *k;
    g_ctbmv_fortran_call.lda = *lda;
    g_ctbmv_fortran_call.incx = *incx;
    g_ctbmv_fortran_call.a = a;
    g_ctbmv_fortran_call.x = x;
    x[0] = make_cf32(101.0f, -1.0f);
    x[1] = make_cf32(102.0f, -2.0f);
}

static void stub_ctbmv_cblas(fb_layout_t layout, fb_uplo_t uplo,
                             fb_transpose_t trans, fb_diag_t diag,
                             int n, int k, const void *a, int lda,
                             void *x, int incx)
{
    fb_complex_float_t *x_values = (fb_complex_float_t *)x;

    g_ctbmv_cblas_call.called += 1;
    g_ctbmv_cblas_call.order = layout;
    g_ctbmv_cblas_call.uplo = uplo;
    g_ctbmv_cblas_call.trans = trans;
    g_ctbmv_cblas_call.diag = diag;
    g_ctbmv_cblas_call.n = n;
    g_ctbmv_cblas_call.k = k;
    g_ctbmv_cblas_call.lda = lda;
    g_ctbmv_cblas_call.incx = incx;
    g_ctbmv_cblas_call.a = a;
    g_ctbmv_cblas_call.x = x;
    x_values[0] = make_cf32(111.0f, 1.0f);
    x_values[1] = make_cf32(112.0f, 2.0f);
}

static void stub_ztbmv_fortran(const int *order, const int *uplo,
                               const int *trans, const int *diag,
                               const int *n, const int *k,
                               const fb_complex_double_t *a, const int *lda,
                               fb_complex_double_t *x, const int *incx)
{
    g_ztbmv_fortran_call.called += 1;
    g_ztbmv_fortran_call.order = *order;
    g_ztbmv_fortran_call.uplo = *uplo;
    g_ztbmv_fortran_call.trans = *trans;
    g_ztbmv_fortran_call.diag = *diag;
    g_ztbmv_fortran_call.n = *n;
    g_ztbmv_fortran_call.k = *k;
    g_ztbmv_fortran_call.lda = *lda;
    g_ztbmv_fortran_call.incx = *incx;
    g_ztbmv_fortran_call.a = a;
    g_ztbmv_fortran_call.x = x;
    x[0] = make_cf64(121.0, -1.0);
    x[1] = make_cf64(122.0, -2.0);
}

static void stub_ztbmv_cblas(fb_layout_t layout, fb_uplo_t uplo,
                             fb_transpose_t trans, fb_diag_t diag,
                             int n, int k, const void *a, int lda,
                             void *x, int incx)
{
    fb_complex_double_t *x_values = (fb_complex_double_t *)x;

    g_ztbmv_cblas_call.called += 1;
    g_ztbmv_cblas_call.order = layout;
    g_ztbmv_cblas_call.uplo = uplo;
    g_ztbmv_cblas_call.trans = trans;
    g_ztbmv_cblas_call.diag = diag;
    g_ztbmv_cblas_call.n = n;
    g_ztbmv_cblas_call.k = k;
    g_ztbmv_cblas_call.lda = lda;
    g_ztbmv_cblas_call.incx = incx;
    g_ztbmv_cblas_call.a = a;
    g_ztbmv_cblas_call.x = x;
    x_values[0] = make_cf64(131.0, 1.0);
    x_values[1] = make_cf64(132.0, 2.0);
}

static void stub_ctbsv_fortran(const int *order, const int *uplo,
                               const int *trans, const int *diag,
                               const int *n, const int *k,
                               const fb_complex_float_t *a, const int *lda,
                               fb_complex_float_t *x, const int *incx)
{
    g_ctbsv_fortran_call.called += 1;
    g_ctbsv_fortran_call.order = *order;
    g_ctbsv_fortran_call.uplo = *uplo;
    g_ctbsv_fortran_call.trans = *trans;
    g_ctbsv_fortran_call.diag = *diag;
    g_ctbsv_fortran_call.n = *n;
    g_ctbsv_fortran_call.k = *k;
    g_ctbsv_fortran_call.lda = *lda;
    g_ctbsv_fortran_call.incx = *incx;
    g_ctbsv_fortran_call.a = a;
    g_ctbsv_fortran_call.x = x;
    x[0] = make_cf32(201.0f, -1.0f);
    x[1] = make_cf32(202.0f, -2.0f);
}

static void stub_ctbsv_cblas(fb_layout_t layout, fb_uplo_t uplo,
                             fb_transpose_t trans, fb_diag_t diag,
                             int n, int k, const void *a, int lda,
                             void *x, int incx)
{
    fb_complex_float_t *x_values = (fb_complex_float_t *)x;

    g_ctbsv_cblas_call.called += 1;
    g_ctbsv_cblas_call.order = layout;
    g_ctbsv_cblas_call.uplo = uplo;
    g_ctbsv_cblas_call.trans = trans;
    g_ctbsv_cblas_call.diag = diag;
    g_ctbsv_cblas_call.n = n;
    g_ctbsv_cblas_call.k = k;
    g_ctbsv_cblas_call.lda = lda;
    g_ctbsv_cblas_call.incx = incx;
    g_ctbsv_cblas_call.a = a;
    g_ctbsv_cblas_call.x = x;
    x_values[0] = make_cf32(211.0f, 1.0f);
    x_values[1] = make_cf32(212.0f, 2.0f);
}

static void stub_ztbsv_fortran(const int *order, const int *uplo,
                               const int *trans, const int *diag,
                               const int *n, const int *k,
                               const fb_complex_double_t *a, const int *lda,
                               fb_complex_double_t *x, const int *incx)
{
    g_ztbsv_fortran_call.called += 1;
    g_ztbsv_fortran_call.order = *order;
    g_ztbsv_fortran_call.uplo = *uplo;
    g_ztbsv_fortran_call.trans = *trans;
    g_ztbsv_fortran_call.diag = *diag;
    g_ztbsv_fortran_call.n = *n;
    g_ztbsv_fortran_call.k = *k;
    g_ztbsv_fortran_call.lda = *lda;
    g_ztbsv_fortran_call.incx = *incx;
    g_ztbsv_fortran_call.a = a;
    g_ztbsv_fortran_call.x = x;
    x[0] = make_cf64(221.0, -1.0);
    x[1] = make_cf64(222.0, -2.0);
}

static void stub_ztbsv_cblas(fb_layout_t layout, fb_uplo_t uplo,
                             fb_transpose_t trans, fb_diag_t diag,
                             int n, int k, const void *a, int lda,
                             void *x, int incx)
{
    fb_complex_double_t *x_values = (fb_complex_double_t *)x;

    g_ztbsv_cblas_call.called += 1;
    g_ztbsv_cblas_call.order = layout;
    g_ztbsv_cblas_call.uplo = uplo;
    g_ztbsv_cblas_call.trans = trans;
    g_ztbsv_cblas_call.diag = diag;
    g_ztbsv_cblas_call.n = n;
    g_ztbsv_cblas_call.k = k;
    g_ztbsv_cblas_call.lda = lda;
    g_ztbsv_cblas_call.incx = incx;
    g_ztbsv_cblas_call.a = a;
    g_ztbsv_cblas_call.x = x;
    x_values[0] = make_cf64(231.0, 1.0);
    x_values[1] = make_cf64(232.0, 2.0);
}

static int check_ctbmv_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_ctbmv_cblas_fn thunk = NULL;
    fb_complex_float_t a[4] = {
        make_cf32(1.0f, 1.0f), make_cf32(2.0f, 2.0f),
        make_cf32(3.0f, 3.0f), make_cf32(4.0f, 4.0f)
    };
    fb_complex_float_t x[2] = { make_cf32(5.0f, 5.0f), make_cf32(6.0f, 6.0f) };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_ctbmv_fortran_call, 0, sizeof(g_ctbmv_fortran_call));
    vtable.ext_ops[FB_OP_CTBMV][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_ctbmv_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CTBMV);
    thunk = (fb_ctbmv_cblas_fn)vtable.ext_ops[FB_OP_CTBMV][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CTBMV Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    thunk(FB_LAYOUT_ROW_MAJOR, FB_LOWER, FB_CONJ_TRANS, FB_NON_UNIT, 2, 1, a, 2, x, 1);
    if (g_ctbmv_fortran_call.called != 1 ||
        g_ctbmv_fortran_call.order != FB_LAYOUT_ROW_MAJOR ||
        g_ctbmv_fortran_call.uplo != FB_LOWER ||
        g_ctbmv_fortran_call.trans != FB_CONJ_TRANS ||
        g_ctbmv_fortran_call.diag != FB_NON_UNIT ||
        g_ctbmv_fortran_call.n != 2 ||
        g_ctbmv_fortran_call.k != 1 ||
        g_ctbmv_fortran_call.lda != 2 ||
        g_ctbmv_fortran_call.incx != 1 ||
        g_ctbmv_fortran_call.a != a ||
        g_ctbmv_fortran_call.x != x ||
        !cf32_eq(x[0], make_cf32(101.0f, -1.0f)) ||
        !cf32_eq(x[1], make_cf32(102.0f, -2.0f))) {
        fprintf(stderr, "[FAIL] CTBMV Fortran->CBLAS thunk did not forward complex triangular band mat-vec arguments correctly\n");
        return 1;
    }

    printf("[PASS] CTBMV Fortran->CBLAS thunk forwards complex triangular band mat-vec arguments unchanged\n");
    return 0;
}

static int check_ctbmv_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_ctbmv_fortran_slot_fn thunk = NULL;
    int order = FB_LAYOUT_COL_MAJOR;
    int uplo = FB_UPPER;
    int trans = FB_NO_TRANS;
    int diag = FB_UNIT;
    int n = 2;
    int k = 1;
    int lda = 2;
    int incx = 1;
    fb_complex_float_t a[4] = {
        make_cf32(7.0f, 7.0f), make_cf32(8.0f, 8.0f),
        make_cf32(9.0f, 9.0f), make_cf32(10.0f, 10.0f)
    };
    fb_complex_float_t x[2] = { make_cf32(11.0f, 11.0f), make_cf32(12.0f, 12.0f) };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_ctbmv_cblas_call, 0, sizeof(g_ctbmv_cblas_call));
    vtable.ext_ops[FB_OP_CTBMV][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_ctbmv_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CTBMV);
    thunk = (fb_ctbmv_fortran_slot_fn)vtable.ext_ops[FB_OP_CTBMV][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CTBMV CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&order, &uplo, &trans, &diag, &n, &k, a, &lda, x, &incx);
    if (g_ctbmv_cblas_call.called != 1 ||
        g_ctbmv_cblas_call.order != FB_LAYOUT_COL_MAJOR ||
        g_ctbmv_cblas_call.uplo != FB_UPPER ||
        g_ctbmv_cblas_call.trans != FB_NO_TRANS ||
        g_ctbmv_cblas_call.diag != FB_UNIT ||
        g_ctbmv_cblas_call.n != 2 ||
        g_ctbmv_cblas_call.k != 1 ||
        g_ctbmv_cblas_call.lda != 2 ||
        g_ctbmv_cblas_call.incx != 1 ||
        g_ctbmv_cblas_call.a != a ||
        g_ctbmv_cblas_call.x != x ||
        !cf32_eq(x[0], make_cf32(111.0f, 1.0f)) ||
        !cf32_eq(x[1], make_cf32(112.0f, 2.0f))) {
        fprintf(stderr, "[FAIL] CTBMV CBLAS->Fortran thunk did not map complex triangular band mat-vec arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] CTBMV CBLAS->Fortran thunk maps complex triangular band mat-vec arguments into the C entry\n");
    return 0;
}

static int check_ztbmv_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_ztbmv_cblas_fn thunk = NULL;
    fb_complex_double_t a[4] = {
        make_cf64(13.0, 13.0), make_cf64(14.0, 14.0),
        make_cf64(15.0, 15.0), make_cf64(16.0, 16.0)
    };
    fb_complex_double_t x[2] = { make_cf64(17.0, 17.0), make_cf64(18.0, 18.0) };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_ztbmv_fortran_call, 0, sizeof(g_ztbmv_fortran_call));
    vtable.ext_ops[FB_OP_ZTBMV][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_ztbmv_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_ZTBMV);
    thunk = (fb_ztbmv_cblas_fn)vtable.ext_ops[FB_OP_ZTBMV][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZTBMV Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    thunk(FB_LAYOUT_ROW_MAJOR, FB_LOWER, FB_CONJ_TRANS, FB_NON_UNIT, 2, 1, a, 2, x, 1);
    if (g_ztbmv_fortran_call.called != 1 ||
        g_ztbmv_fortran_call.order != FB_LAYOUT_ROW_MAJOR ||
        g_ztbmv_fortran_call.uplo != FB_LOWER ||
        g_ztbmv_fortran_call.trans != FB_CONJ_TRANS ||
        g_ztbmv_fortran_call.diag != FB_NON_UNIT ||
        g_ztbmv_fortran_call.n != 2 ||
        g_ztbmv_fortran_call.k != 1 ||
        g_ztbmv_fortran_call.lda != 2 ||
        g_ztbmv_fortran_call.incx != 1 ||
        g_ztbmv_fortran_call.a != a ||
        g_ztbmv_fortran_call.x != x ||
        !cf64_eq(x[0], make_cf64(121.0, -1.0)) ||
        !cf64_eq(x[1], make_cf64(122.0, -2.0))) {
        fprintf(stderr, "[FAIL] ZTBMV Fortran->CBLAS thunk did not forward double-complex triangular band mat-vec arguments correctly\n");
        return 1;
    }

    printf("[PASS] ZTBMV Fortran->CBLAS thunk forwards double-complex triangular band mat-vec arguments unchanged\n");
    return 0;
}

static int check_ztbmv_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_ztbmv_fortran_slot_fn thunk = NULL;
    int order = FB_LAYOUT_COL_MAJOR;
    int uplo = FB_UPPER;
    int trans = FB_NO_TRANS;
    int diag = FB_UNIT;
    int n = 2;
    int k = 1;
    int lda = 2;
    int incx = 1;
    fb_complex_double_t a[4] = {
        make_cf64(19.0, 19.0), make_cf64(20.0, 20.0),
        make_cf64(21.0, 21.0), make_cf64(22.0, 22.0)
    };
    fb_complex_double_t x[2] = { make_cf64(23.0, 23.0), make_cf64(24.0, 24.0) };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_ztbmv_cblas_call, 0, sizeof(g_ztbmv_cblas_call));
    vtable.ext_ops[FB_OP_ZTBMV][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_ztbmv_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_ZTBMV);
    thunk = (fb_ztbmv_fortran_slot_fn)vtable.ext_ops[FB_OP_ZTBMV][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZTBMV CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&order, &uplo, &trans, &diag, &n, &k, a, &lda, x, &incx);
    if (g_ztbmv_cblas_call.called != 1 ||
        g_ztbmv_cblas_call.order != FB_LAYOUT_COL_MAJOR ||
        g_ztbmv_cblas_call.uplo != FB_UPPER ||
        g_ztbmv_cblas_call.trans != FB_NO_TRANS ||
        g_ztbmv_cblas_call.diag != FB_UNIT ||
        g_ztbmv_cblas_call.n != 2 ||
        g_ztbmv_cblas_call.k != 1 ||
        g_ztbmv_cblas_call.lda != 2 ||
        g_ztbmv_cblas_call.incx != 1 ||
        g_ztbmv_cblas_call.a != a ||
        g_ztbmv_cblas_call.x != x ||
        !cf64_eq(x[0], make_cf64(131.0, 1.0)) ||
        !cf64_eq(x[1], make_cf64(132.0, 2.0))) {
        fprintf(stderr, "[FAIL] ZTBMV CBLAS->Fortran thunk did not map double-complex triangular band mat-vec arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] ZTBMV CBLAS->Fortran thunk maps double-complex triangular band mat-vec arguments into the C entry\n");
    return 0;
}

static int check_ctbsv_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_ctbsv_cblas_fn thunk = NULL;
    fb_complex_float_t a[4] = {
        make_cf32(25.0f, 25.0f), make_cf32(26.0f, 26.0f),
        make_cf32(27.0f, 27.0f), make_cf32(28.0f, 28.0f)
    };
    fb_complex_float_t x[2] = { make_cf32(29.0f, 29.0f), make_cf32(30.0f, 30.0f) };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_ctbsv_fortran_call, 0, sizeof(g_ctbsv_fortran_call));
    vtable.ext_ops[FB_OP_CTBSV][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_ctbsv_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CTBSV);
    thunk = (fb_ctbsv_cblas_fn)vtable.ext_ops[FB_OP_CTBSV][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CTBSV Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    thunk(FB_LAYOUT_ROW_MAJOR, FB_LOWER, FB_CONJ_TRANS, FB_NON_UNIT, 2, 1, a, 2, x, 1);
    if (g_ctbsv_fortran_call.called != 1 ||
        g_ctbsv_fortran_call.order != FB_LAYOUT_ROW_MAJOR ||
        g_ctbsv_fortran_call.uplo != FB_LOWER ||
        g_ctbsv_fortran_call.trans != FB_CONJ_TRANS ||
        g_ctbsv_fortran_call.diag != FB_NON_UNIT ||
        g_ctbsv_fortran_call.n != 2 ||
        g_ctbsv_fortran_call.k != 1 ||
        g_ctbsv_fortran_call.lda != 2 ||
        g_ctbsv_fortran_call.incx != 1 ||
        g_ctbsv_fortran_call.a != a ||
        g_ctbsv_fortran_call.x != x ||
        !cf32_eq(x[0], make_cf32(201.0f, -1.0f)) ||
        !cf32_eq(x[1], make_cf32(202.0f, -2.0f))) {
        fprintf(stderr, "[FAIL] CTBSV Fortran->CBLAS thunk did not forward complex triangular band solve arguments correctly\n");
        return 1;
    }

    printf("[PASS] CTBSV Fortran->CBLAS thunk forwards complex triangular band solve arguments unchanged\n");
    return 0;
}

static int check_ctbsv_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_ctbsv_fortran_slot_fn thunk = NULL;
    int order = FB_LAYOUT_COL_MAJOR;
    int uplo = FB_UPPER;
    int trans = FB_NO_TRANS;
    int diag = FB_UNIT;
    int n = 2;
    int k = 1;
    int lda = 2;
    int incx = 1;
    fb_complex_float_t a[4] = {
        make_cf32(31.0f, 31.0f), make_cf32(32.0f, 32.0f),
        make_cf32(33.0f, 33.0f), make_cf32(34.0f, 34.0f)
    };
    fb_complex_float_t x[2] = { make_cf32(35.0f, 35.0f), make_cf32(36.0f, 36.0f) };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_ctbsv_cblas_call, 0, sizeof(g_ctbsv_cblas_call));
    vtable.ext_ops[FB_OP_CTBSV][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_ctbsv_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CTBSV);
    thunk = (fb_ctbsv_fortran_slot_fn)vtable.ext_ops[FB_OP_CTBSV][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CTBSV CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&order, &uplo, &trans, &diag, &n, &k, a, &lda, x, &incx);
    if (g_ctbsv_cblas_call.called != 1 ||
        g_ctbsv_cblas_call.order != FB_LAYOUT_COL_MAJOR ||
        g_ctbsv_cblas_call.uplo != FB_UPPER ||
        g_ctbsv_cblas_call.trans != FB_NO_TRANS ||
        g_ctbsv_cblas_call.diag != FB_UNIT ||
        g_ctbsv_cblas_call.n != 2 ||
        g_ctbsv_cblas_call.k != 1 ||
        g_ctbsv_cblas_call.lda != 2 ||
        g_ctbsv_cblas_call.incx != 1 ||
        g_ctbsv_cblas_call.a != a ||
        g_ctbsv_cblas_call.x != x ||
        !cf32_eq(x[0], make_cf32(211.0f, 1.0f)) ||
        !cf32_eq(x[1], make_cf32(212.0f, 2.0f))) {
        fprintf(stderr, "[FAIL] CTBSV CBLAS->Fortran thunk did not map complex triangular band solve arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] CTBSV CBLAS->Fortran thunk maps complex triangular band solve arguments into the C entry\n");
    return 0;
}

static int check_ztbsv_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_ztbsv_cblas_fn thunk = NULL;
    fb_complex_double_t a[4] = {
        make_cf64(37.0, 37.0), make_cf64(38.0, 38.0),
        make_cf64(39.0, 39.0), make_cf64(40.0, 40.0)
    };
    fb_complex_double_t x[2] = { make_cf64(41.0, 41.0), make_cf64(42.0, 42.0) };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_ztbsv_fortran_call, 0, sizeof(g_ztbsv_fortran_call));
    vtable.ext_ops[FB_OP_ZTBSV][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_ztbsv_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_ZTBSV);
    thunk = (fb_ztbsv_cblas_fn)vtable.ext_ops[FB_OP_ZTBSV][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZTBSV Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    thunk(FB_LAYOUT_ROW_MAJOR, FB_LOWER, FB_CONJ_TRANS, FB_NON_UNIT, 2, 1, a, 2, x, 1);
    if (g_ztbsv_fortran_call.called != 1 ||
        g_ztbsv_fortran_call.order != FB_LAYOUT_ROW_MAJOR ||
        g_ztbsv_fortran_call.uplo != FB_LOWER ||
        g_ztbsv_fortran_call.trans != FB_CONJ_TRANS ||
        g_ztbsv_fortran_call.diag != FB_NON_UNIT ||
        g_ztbsv_fortran_call.n != 2 ||
        g_ztbsv_fortran_call.k != 1 ||
        g_ztbsv_fortran_call.lda != 2 ||
        g_ztbsv_fortran_call.incx != 1 ||
        g_ztbsv_fortran_call.a != a ||
        g_ztbsv_fortran_call.x != x ||
        !cf64_eq(x[0], make_cf64(221.0, -1.0)) ||
        !cf64_eq(x[1], make_cf64(222.0, -2.0))) {
        fprintf(stderr, "[FAIL] ZTBSV Fortran->CBLAS thunk did not forward double-complex triangular band solve arguments correctly\n");
        return 1;
    }

    printf("[PASS] ZTBSV Fortran->CBLAS thunk forwards double-complex triangular band solve arguments unchanged\n");
    return 0;
}

static int check_ztbsv_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_ztbsv_fortran_slot_fn thunk = NULL;
    int order = FB_LAYOUT_COL_MAJOR;
    int uplo = FB_UPPER;
    int trans = FB_NO_TRANS;
    int diag = FB_UNIT;
    int n = 2;
    int k = 1;
    int lda = 2;
    int incx = 1;
    fb_complex_double_t a[4] = {
        make_cf64(43.0, 43.0), make_cf64(44.0, 44.0),
        make_cf64(45.0, 45.0), make_cf64(46.0, 46.0)
    };
    fb_complex_double_t x[2] = { make_cf64(47.0, 47.0), make_cf64(48.0, 48.0) };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_ztbsv_cblas_call, 0, sizeof(g_ztbsv_cblas_call));
    vtable.ext_ops[FB_OP_ZTBSV][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_ztbsv_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_ZTBSV);
    thunk = (fb_ztbsv_fortran_slot_fn)vtable.ext_ops[FB_OP_ZTBSV][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZTBSV CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&order, &uplo, &trans, &diag, &n, &k, a, &lda, x, &incx);
    if (g_ztbsv_cblas_call.called != 1 ||
        g_ztbsv_cblas_call.order != FB_LAYOUT_COL_MAJOR ||
        g_ztbsv_cblas_call.uplo != FB_UPPER ||
        g_ztbsv_cblas_call.trans != FB_NO_TRANS ||
        g_ztbsv_cblas_call.diag != FB_UNIT ||
        g_ztbsv_cblas_call.n != 2 ||
        g_ztbsv_cblas_call.k != 1 ||
        g_ztbsv_cblas_call.lda != 2 ||
        g_ztbsv_cblas_call.incx != 1 ||
        g_ztbsv_cblas_call.a != a ||
        g_ztbsv_cblas_call.x != x ||
        !cf64_eq(x[0], make_cf64(231.0, 1.0)) ||
        !cf64_eq(x[1], make_cf64(232.0, 2.0))) {
        fprintf(stderr, "[FAIL] ZTBSV CBLAS->Fortran thunk did not map double-complex triangular band solve arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] ZTBSV CBLAS->Fortran thunk maps double-complex triangular band solve arguments into the C entry\n");
    return 0;
}

int main(void)
{
    int status = 0;

    status |= check_ctbmv_fortran_to_cblas();
    status |= check_ctbmv_cblas_to_fortran();
    status |= check_ztbmv_fortran_to_cblas();
    status |= check_ztbmv_cblas_to_fortran();
    status |= check_ctbsv_fortran_to_cblas();
    status |= check_ctbsv_cblas_to_fortran();
    status |= check_ztbsv_fortran_to_cblas();
    status |= check_ztbsv_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}