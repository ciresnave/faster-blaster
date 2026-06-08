#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef void (*fb_ctpmv_cblas_fn)(fb_layout_t layout, fb_uplo_t uplo,
                                  fb_transpose_t trans, fb_diag_t diag,
                                  int n, const void *ap, void *x, int incx);
typedef void (*fb_ztpmv_cblas_fn)(fb_layout_t layout, fb_uplo_t uplo,
                                  fb_transpose_t trans, fb_diag_t diag,
                                  int n, const void *ap, void *x, int incx);
typedef void (*fb_ctpmv_fortran_slot_fn)(const int *order, const int *uplo,
                                         const int *trans, const int *diag,
                                         const int *n,
                                         const fb_complex_float_t *ap,
                                         fb_complex_float_t *x,
                                         const int *incx);
typedef void (*fb_ztpmv_fortran_slot_fn)(const int *order, const int *uplo,
                                         const int *trans, const int *diag,
                                         const int *n,
                                         const fb_complex_double_t *ap,
                                         fb_complex_double_t *x,
                                         const int *incx);
typedef fb_ctpmv_cblas_fn fb_ctpsv_cblas_fn;
typedef fb_ztpmv_cblas_fn fb_ztpsv_cblas_fn;
typedef fb_ctpmv_fortran_slot_fn fb_ctpsv_fortran_slot_fn;
typedef fb_ztpmv_fortran_slot_fn fb_ztpsv_fortran_slot_fn;

typedef struct {
    int called;
    int order;
    int uplo;
    int trans;
    int diag;
    int n;
    int incx;
    const void *ap;
    void *x;
} packed_call_t;

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

static packed_call_t g_ctpmv_fortran_call;
static packed_call_t g_ctpmv_cblas_call;
static packed_call_t g_ztpmv_fortran_call;
static packed_call_t g_ztpmv_cblas_call;
static packed_call_t g_ctpsv_fortran_call;
static packed_call_t g_ctpsv_cblas_call;
static packed_call_t g_ztpsv_fortran_call;
static packed_call_t g_ztpsv_cblas_call;

static void stub_ctpmv_fortran(const int *order, const int *uplo,
                               const int *trans, const int *diag,
                               const int *n, const fb_complex_float_t *ap,
                               fb_complex_float_t *x, const int *incx)
{
    g_ctpmv_fortran_call.called += 1;
    g_ctpmv_fortran_call.order = *order;
    g_ctpmv_fortran_call.uplo = *uplo;
    g_ctpmv_fortran_call.trans = *trans;
    g_ctpmv_fortran_call.diag = *diag;
    g_ctpmv_fortran_call.n = *n;
    g_ctpmv_fortran_call.incx = *incx;
    g_ctpmv_fortran_call.ap = ap;
    g_ctpmv_fortran_call.x = x;
    x[0] = make_cf32(301.0f, -1.0f);
    x[1] = make_cf32(302.0f, -2.0f);
}

static void stub_ctpmv_cblas(fb_layout_t layout, fb_uplo_t uplo,
                             fb_transpose_t trans, fb_diag_t diag,
                             int n, const void *ap, void *x, int incx)
{
    fb_complex_float_t *x_values = (fb_complex_float_t *)x;

    g_ctpmv_cblas_call.called += 1;
    g_ctpmv_cblas_call.order = layout;
    g_ctpmv_cblas_call.uplo = uplo;
    g_ctpmv_cblas_call.trans = trans;
    g_ctpmv_cblas_call.diag = diag;
    g_ctpmv_cblas_call.n = n;
    g_ctpmv_cblas_call.incx = incx;
    g_ctpmv_cblas_call.ap = ap;
    g_ctpmv_cblas_call.x = x;
    x_values[0] = make_cf32(311.0f, 1.0f);
    x_values[1] = make_cf32(312.0f, 2.0f);
}

static void stub_ztpmv_fortran(const int *order, const int *uplo,
                               const int *trans, const int *diag,
                               const int *n, const fb_complex_double_t *ap,
                               fb_complex_double_t *x, const int *incx)
{
    g_ztpmv_fortran_call.called += 1;
    g_ztpmv_fortran_call.order = *order;
    g_ztpmv_fortran_call.uplo = *uplo;
    g_ztpmv_fortran_call.trans = *trans;
    g_ztpmv_fortran_call.diag = *diag;
    g_ztpmv_fortran_call.n = *n;
    g_ztpmv_fortran_call.incx = *incx;
    g_ztpmv_fortran_call.ap = ap;
    g_ztpmv_fortran_call.x = x;
    x[0] = make_cf64(321.0, -1.0);
    x[1] = make_cf64(322.0, -2.0);
}

static void stub_ztpmv_cblas(fb_layout_t layout, fb_uplo_t uplo,
                             fb_transpose_t trans, fb_diag_t diag,
                             int n, const void *ap, void *x, int incx)
{
    fb_complex_double_t *x_values = (fb_complex_double_t *)x;

    g_ztpmv_cblas_call.called += 1;
    g_ztpmv_cblas_call.order = layout;
    g_ztpmv_cblas_call.uplo = uplo;
    g_ztpmv_cblas_call.trans = trans;
    g_ztpmv_cblas_call.diag = diag;
    g_ztpmv_cblas_call.n = n;
    g_ztpmv_cblas_call.incx = incx;
    g_ztpmv_cblas_call.ap = ap;
    g_ztpmv_cblas_call.x = x;
    x_values[0] = make_cf64(331.0, 1.0);
    x_values[1] = make_cf64(332.0, 2.0);
}

static void stub_ctpsv_fortran(const int *order, const int *uplo,
                               const int *trans, const int *diag,
                               const int *n, const fb_complex_float_t *ap,
                               fb_complex_float_t *x, const int *incx)
{
    g_ctpsv_fortran_call.called += 1;
    g_ctpsv_fortran_call.order = *order;
    g_ctpsv_fortran_call.uplo = *uplo;
    g_ctpsv_fortran_call.trans = *trans;
    g_ctpsv_fortran_call.diag = *diag;
    g_ctpsv_fortran_call.n = *n;
    g_ctpsv_fortran_call.incx = *incx;
    g_ctpsv_fortran_call.ap = ap;
    g_ctpsv_fortran_call.x = x;
    x[0] = make_cf32(401.0f, -1.0f);
    x[1] = make_cf32(402.0f, -2.0f);
}

static void stub_ctpsv_cblas(fb_layout_t layout, fb_uplo_t uplo,
                             fb_transpose_t trans, fb_diag_t diag,
                             int n, const void *ap, void *x, int incx)
{
    fb_complex_float_t *x_values = (fb_complex_float_t *)x;

    g_ctpsv_cblas_call.called += 1;
    g_ctpsv_cblas_call.order = layout;
    g_ctpsv_cblas_call.uplo = uplo;
    g_ctpsv_cblas_call.trans = trans;
    g_ctpsv_cblas_call.diag = diag;
    g_ctpsv_cblas_call.n = n;
    g_ctpsv_cblas_call.incx = incx;
    g_ctpsv_cblas_call.ap = ap;
    g_ctpsv_cblas_call.x = x;
    x_values[0] = make_cf32(411.0f, 1.0f);
    x_values[1] = make_cf32(412.0f, 2.0f);
}

static void stub_ztpsv_fortran(const int *order, const int *uplo,
                               const int *trans, const int *diag,
                               const int *n, const fb_complex_double_t *ap,
                               fb_complex_double_t *x, const int *incx)
{
    g_ztpsv_fortran_call.called += 1;
    g_ztpsv_fortran_call.order = *order;
    g_ztpsv_fortran_call.uplo = *uplo;
    g_ztpsv_fortran_call.trans = *trans;
    g_ztpsv_fortran_call.diag = *diag;
    g_ztpsv_fortran_call.n = *n;
    g_ztpsv_fortran_call.incx = *incx;
    g_ztpsv_fortran_call.ap = ap;
    g_ztpsv_fortran_call.x = x;
    x[0] = make_cf64(421.0, -1.0);
    x[1] = make_cf64(422.0, -2.0);
}

static void stub_ztpsv_cblas(fb_layout_t layout, fb_uplo_t uplo,
                             fb_transpose_t trans, fb_diag_t diag,
                             int n, const void *ap, void *x, int incx)
{
    fb_complex_double_t *x_values = (fb_complex_double_t *)x;

    g_ztpsv_cblas_call.called += 1;
    g_ztpsv_cblas_call.order = layout;
    g_ztpsv_cblas_call.uplo = uplo;
    g_ztpsv_cblas_call.trans = trans;
    g_ztpsv_cblas_call.diag = diag;
    g_ztpsv_cblas_call.n = n;
    g_ztpsv_cblas_call.incx = incx;
    g_ztpsv_cblas_call.ap = ap;
    g_ztpsv_cblas_call.x = x;
    x_values[0] = make_cf64(431.0, 1.0);
    x_values[1] = make_cf64(432.0, 2.0);
}

static int check_ctpmv_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_ctpmv_cblas_fn thunk = NULL;
    fb_complex_float_t ap[3] = {
        make_cf32(1.0f, 1.0f), make_cf32(2.0f, 2.0f), make_cf32(3.0f, 3.0f)
    };
    fb_complex_float_t x[2] = { make_cf32(4.0f, 4.0f), make_cf32(5.0f, 5.0f) };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_ctpmv_fortran_call, 0, sizeof(g_ctpmv_fortran_call));
    vtable.ext_ops[FB_OP_CTPMV][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_ctpmv_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CTPMV);
    thunk = (fb_ctpmv_cblas_fn)vtable.ext_ops[FB_OP_CTPMV][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CTPMV Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    thunk(FB_LAYOUT_ROW_MAJOR, FB_LOWER, FB_CONJ_TRANS, FB_NON_UNIT, 2, ap, x, 1);
    if (g_ctpmv_fortran_call.called != 1 ||
        g_ctpmv_fortran_call.order != FB_LAYOUT_ROW_MAJOR ||
        g_ctpmv_fortran_call.uplo != FB_LOWER ||
        g_ctpmv_fortran_call.trans != FB_CONJ_TRANS ||
        g_ctpmv_fortran_call.diag != FB_NON_UNIT ||
        g_ctpmv_fortran_call.n != 2 ||
        g_ctpmv_fortran_call.incx != 1 ||
        g_ctpmv_fortran_call.ap != ap ||
        g_ctpmv_fortran_call.x != x ||
        !cf32_eq(x[0], make_cf32(301.0f, -1.0f)) ||
        !cf32_eq(x[1], make_cf32(302.0f, -2.0f))) {
        fprintf(stderr, "[FAIL] CTPMV Fortran->CBLAS thunk did not forward complex triangular packed mat-vec arguments correctly\n");
        return 1;
    }

    printf("[PASS] CTPMV Fortran->CBLAS thunk forwards complex triangular packed mat-vec arguments unchanged\n");
    return 0;
}

static int check_ctpmv_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_ctpmv_fortran_slot_fn thunk = NULL;
    int order = FB_LAYOUT_COL_MAJOR;
    int uplo = FB_UPPER;
    int trans = FB_NO_TRANS;
    int diag = FB_UNIT;
    int n = 2;
    int incx = 1;
    fb_complex_float_t ap[3] = {
        make_cf32(6.0f, 6.0f), make_cf32(7.0f, 7.0f), make_cf32(8.0f, 8.0f)
    };
    fb_complex_float_t x[2] = { make_cf32(9.0f, 9.0f), make_cf32(10.0f, 10.0f) };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_ctpmv_cblas_call, 0, sizeof(g_ctpmv_cblas_call));
    vtable.ext_ops[FB_OP_CTPMV][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_ctpmv_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CTPMV);
    thunk = (fb_ctpmv_fortran_slot_fn)vtable.ext_ops[FB_OP_CTPMV][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CTPMV CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&order, &uplo, &trans, &diag, &n, ap, x, &incx);
    if (g_ctpmv_cblas_call.called != 1 ||
        g_ctpmv_cblas_call.order != FB_LAYOUT_COL_MAJOR ||
        g_ctpmv_cblas_call.uplo != FB_UPPER ||
        g_ctpmv_cblas_call.trans != FB_NO_TRANS ||
        g_ctpmv_cblas_call.diag != FB_UNIT ||
        g_ctpmv_cblas_call.n != 2 ||
        g_ctpmv_cblas_call.incx != 1 ||
        g_ctpmv_cblas_call.ap != ap ||
        g_ctpmv_cblas_call.x != x ||
        !cf32_eq(x[0], make_cf32(311.0f, 1.0f)) ||
        !cf32_eq(x[1], make_cf32(312.0f, 2.0f))) {
        fprintf(stderr, "[FAIL] CTPMV CBLAS->Fortran thunk did not map complex triangular packed mat-vec arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] CTPMV CBLAS->Fortran thunk maps complex triangular packed mat-vec arguments into the C entry\n");
    return 0;
}

static int check_ztpmv_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_ztpmv_cblas_fn thunk = NULL;
    fb_complex_double_t ap[3] = {
        make_cf64(11.0, 11.0), make_cf64(12.0, 12.0), make_cf64(13.0, 13.0)
    };
    fb_complex_double_t x[2] = { make_cf64(14.0, 14.0), make_cf64(15.0, 15.0) };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_ztpmv_fortran_call, 0, sizeof(g_ztpmv_fortran_call));
    vtable.ext_ops[FB_OP_ZTPMV][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_ztpmv_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_ZTPMV);
    thunk = (fb_ztpmv_cblas_fn)vtable.ext_ops[FB_OP_ZTPMV][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZTPMV Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    thunk(FB_LAYOUT_ROW_MAJOR, FB_LOWER, FB_CONJ_TRANS, FB_NON_UNIT, 2, ap, x, 1);
    if (g_ztpmv_fortran_call.called != 1 ||
        g_ztpmv_fortran_call.order != FB_LAYOUT_ROW_MAJOR ||
        g_ztpmv_fortran_call.uplo != FB_LOWER ||
        g_ztpmv_fortran_call.trans != FB_CONJ_TRANS ||
        g_ztpmv_fortran_call.diag != FB_NON_UNIT ||
        g_ztpmv_fortran_call.n != 2 ||
        g_ztpmv_fortran_call.incx != 1 ||
        g_ztpmv_fortran_call.ap != ap ||
        g_ztpmv_fortran_call.x != x ||
        !cf64_eq(x[0], make_cf64(321.0, -1.0)) ||
        !cf64_eq(x[1], make_cf64(322.0, -2.0))) {
        fprintf(stderr, "[FAIL] ZTPMV Fortran->CBLAS thunk did not forward double-complex triangular packed mat-vec arguments correctly\n");
        return 1;
    }

    printf("[PASS] ZTPMV Fortran->CBLAS thunk forwards double-complex triangular packed mat-vec arguments unchanged\n");
    return 0;
}

static int check_ztpmv_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_ztpmv_fortran_slot_fn thunk = NULL;
    int order = FB_LAYOUT_COL_MAJOR;
    int uplo = FB_UPPER;
    int trans = FB_NO_TRANS;
    int diag = FB_UNIT;
    int n = 2;
    int incx = 1;
    fb_complex_double_t ap[3] = {
        make_cf64(16.0, 16.0), make_cf64(17.0, 17.0), make_cf64(18.0, 18.0)
    };
    fb_complex_double_t x[2] = { make_cf64(19.0, 19.0), make_cf64(20.0, 20.0) };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_ztpmv_cblas_call, 0, sizeof(g_ztpmv_cblas_call));
    vtable.ext_ops[FB_OP_ZTPMV][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_ztpmv_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_ZTPMV);
    thunk = (fb_ztpmv_fortran_slot_fn)vtable.ext_ops[FB_OP_ZTPMV][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZTPMV CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&order, &uplo, &trans, &diag, &n, ap, x, &incx);
    if (g_ztpmv_cblas_call.called != 1 ||
        g_ztpmv_cblas_call.order != FB_LAYOUT_COL_MAJOR ||
        g_ztpmv_cblas_call.uplo != FB_UPPER ||
        g_ztpmv_cblas_call.trans != FB_NO_TRANS ||
        g_ztpmv_cblas_call.diag != FB_UNIT ||
        g_ztpmv_cblas_call.n != 2 ||
        g_ztpmv_cblas_call.incx != 1 ||
        g_ztpmv_cblas_call.ap != ap ||
        g_ztpmv_cblas_call.x != x ||
        !cf64_eq(x[0], make_cf64(331.0, 1.0)) ||
        !cf64_eq(x[1], make_cf64(332.0, 2.0))) {
        fprintf(stderr, "[FAIL] ZTPMV CBLAS->Fortran thunk did not map double-complex triangular packed mat-vec arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] ZTPMV CBLAS->Fortran thunk maps double-complex triangular packed mat-vec arguments into the C entry\n");
    return 0;
}

static int check_ctpsv_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_ctpsv_cblas_fn thunk = NULL;
    fb_complex_float_t ap[3] = {
        make_cf32(21.0f, 21.0f), make_cf32(22.0f, 22.0f), make_cf32(23.0f, 23.0f)
    };
    fb_complex_float_t x[2] = { make_cf32(24.0f, 24.0f), make_cf32(25.0f, 25.0f) };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_ctpsv_fortran_call, 0, sizeof(g_ctpsv_fortran_call));
    vtable.ext_ops[FB_OP_CTPSV][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_ctpsv_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CTPSV);
    thunk = (fb_ctpsv_cblas_fn)vtable.ext_ops[FB_OP_CTPSV][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CTPSV Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    thunk(FB_LAYOUT_ROW_MAJOR, FB_LOWER, FB_CONJ_TRANS, FB_NON_UNIT, 2, ap, x, 1);
    if (g_ctpsv_fortran_call.called != 1 ||
        g_ctpsv_fortran_call.order != FB_LAYOUT_ROW_MAJOR ||
        g_ctpsv_fortran_call.uplo != FB_LOWER ||
        g_ctpsv_fortran_call.trans != FB_CONJ_TRANS ||
        g_ctpsv_fortran_call.diag != FB_NON_UNIT ||
        g_ctpsv_fortran_call.n != 2 ||
        g_ctpsv_fortran_call.incx != 1 ||
        g_ctpsv_fortran_call.ap != ap ||
        g_ctpsv_fortran_call.x != x ||
        !cf32_eq(x[0], make_cf32(401.0f, -1.0f)) ||
        !cf32_eq(x[1], make_cf32(402.0f, -2.0f))) {
        fprintf(stderr, "[FAIL] CTPSV Fortran->CBLAS thunk did not forward complex triangular packed solve arguments correctly\n");
        return 1;
    }

    printf("[PASS] CTPSV Fortran->CBLAS thunk forwards complex triangular packed solve arguments unchanged\n");
    return 0;
}

static int check_ctpsv_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_ctpsv_fortran_slot_fn thunk = NULL;
    int order = FB_LAYOUT_COL_MAJOR;
    int uplo = FB_UPPER;
    int trans = FB_NO_TRANS;
    int diag = FB_UNIT;
    int n = 2;
    int incx = 1;
    fb_complex_float_t ap[3] = {
        make_cf32(26.0f, 26.0f), make_cf32(27.0f, 27.0f), make_cf32(28.0f, 28.0f)
    };
    fb_complex_float_t x[2] = { make_cf32(29.0f, 29.0f), make_cf32(30.0f, 30.0f) };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_ctpsv_cblas_call, 0, sizeof(g_ctpsv_cblas_call));
    vtable.ext_ops[FB_OP_CTPSV][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_ctpsv_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CTPSV);
    thunk = (fb_ctpsv_fortran_slot_fn)vtable.ext_ops[FB_OP_CTPSV][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CTPSV CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&order, &uplo, &trans, &diag, &n, ap, x, &incx);
    if (g_ctpsv_cblas_call.called != 1 ||
        g_ctpsv_cblas_call.order != FB_LAYOUT_COL_MAJOR ||
        g_ctpsv_cblas_call.uplo != FB_UPPER ||
        g_ctpsv_cblas_call.trans != FB_NO_TRANS ||
        g_ctpsv_cblas_call.diag != FB_UNIT ||
        g_ctpsv_cblas_call.n != 2 ||
        g_ctpsv_cblas_call.incx != 1 ||
        g_ctpsv_cblas_call.ap != ap ||
        g_ctpsv_cblas_call.x != x ||
        !cf32_eq(x[0], make_cf32(411.0f, 1.0f)) ||
        !cf32_eq(x[1], make_cf32(412.0f, 2.0f))) {
        fprintf(stderr, "[FAIL] CTPSV CBLAS->Fortran thunk did not map complex triangular packed solve arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] CTPSV CBLAS->Fortran thunk maps complex triangular packed solve arguments into the C entry\n");
    return 0;
}

static int check_ztpsv_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_ztpsv_cblas_fn thunk = NULL;
    fb_complex_double_t ap[3] = {
        make_cf64(31.0, 31.0), make_cf64(32.0, 32.0), make_cf64(33.0, 33.0)
    };
    fb_complex_double_t x[2] = { make_cf64(34.0, 34.0), make_cf64(35.0, 35.0) };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_ztpsv_fortran_call, 0, sizeof(g_ztpsv_fortran_call));
    vtable.ext_ops[FB_OP_ZTPSV][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_ztpsv_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_ZTPSV);
    thunk = (fb_ztpsv_cblas_fn)vtable.ext_ops[FB_OP_ZTPSV][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZTPSV Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    thunk(FB_LAYOUT_ROW_MAJOR, FB_LOWER, FB_CONJ_TRANS, FB_NON_UNIT, 2, ap, x, 1);
    if (g_ztpsv_fortran_call.called != 1 ||
        g_ztpsv_fortran_call.order != FB_LAYOUT_ROW_MAJOR ||
        g_ztpsv_fortran_call.uplo != FB_LOWER ||
        g_ztpsv_fortran_call.trans != FB_CONJ_TRANS ||
        g_ztpsv_fortran_call.diag != FB_NON_UNIT ||
        g_ztpsv_fortran_call.n != 2 ||
        g_ztpsv_fortran_call.incx != 1 ||
        g_ztpsv_fortran_call.ap != ap ||
        g_ztpsv_fortran_call.x != x ||
        !cf64_eq(x[0], make_cf64(421.0, -1.0)) ||
        !cf64_eq(x[1], make_cf64(422.0, -2.0))) {
        fprintf(stderr, "[FAIL] ZTPSV Fortran->CBLAS thunk did not forward double-complex triangular packed solve arguments correctly\n");
        return 1;
    }

    printf("[PASS] ZTPSV Fortran->CBLAS thunk forwards double-complex triangular packed solve arguments unchanged\n");
    return 0;
}

static int check_ztpsv_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_ztpsv_fortran_slot_fn thunk = NULL;
    int order = FB_LAYOUT_COL_MAJOR;
    int uplo = FB_UPPER;
    int trans = FB_NO_TRANS;
    int diag = FB_UNIT;
    int n = 2;
    int incx = 1;
    fb_complex_double_t ap[3] = {
        make_cf64(36.0, 36.0), make_cf64(37.0, 37.0), make_cf64(38.0, 38.0)
    };
    fb_complex_double_t x[2] = { make_cf64(39.0, 39.0), make_cf64(40.0, 40.0) };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_ztpsv_cblas_call, 0, sizeof(g_ztpsv_cblas_call));
    vtable.ext_ops[FB_OP_ZTPSV][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_ztpsv_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_ZTPSV);
    thunk = (fb_ztpsv_fortran_slot_fn)vtable.ext_ops[FB_OP_ZTPSV][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZTPSV CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&order, &uplo, &trans, &diag, &n, ap, x, &incx);
    if (g_ztpsv_cblas_call.called != 1 ||
        g_ztpsv_cblas_call.order != FB_LAYOUT_COL_MAJOR ||
        g_ztpsv_cblas_call.uplo != FB_UPPER ||
        g_ztpsv_cblas_call.trans != FB_NO_TRANS ||
        g_ztpsv_cblas_call.diag != FB_UNIT ||
        g_ztpsv_cblas_call.n != 2 ||
        g_ztpsv_cblas_call.incx != 1 ||
        g_ztpsv_cblas_call.ap != ap ||
        g_ztpsv_cblas_call.x != x ||
        !cf64_eq(x[0], make_cf64(431.0, 1.0)) ||
        !cf64_eq(x[1], make_cf64(432.0, 2.0))) {
        fprintf(stderr, "[FAIL] ZTPSV CBLAS->Fortran thunk did not map double-complex triangular packed solve arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] ZTPSV CBLAS->Fortran thunk maps double-complex triangular packed solve arguments into the C entry\n");
    return 0;
}

int main(void)
{
    int status = 0;

    status |= check_ctpmv_fortran_to_cblas();
    status |= check_ctpmv_cblas_to_fortran();
    status |= check_ztpmv_fortran_to_cblas();
    status |= check_ztpmv_cblas_to_fortran();
    status |= check_ctpsv_fortran_to_cblas();
    status |= check_ctpsv_cblas_to_fortran();
    status |= check_ztpsv_fortran_to_cblas();
    status |= check_ztpsv_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}