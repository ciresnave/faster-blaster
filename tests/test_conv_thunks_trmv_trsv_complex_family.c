#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef void (*fb_ctrmv_cblas_fn)(fb_layout_t layout, fb_uplo_t uplo,
                                  fb_transpose_t trans, fb_diag_t diag,
                                  int n, const void *a, int lda,
                                  void *x, int incx);
typedef void (*fb_ztrmv_cblas_fn)(fb_layout_t layout, fb_uplo_t uplo,
                                  fb_transpose_t trans, fb_diag_t diag,
                                  int n, const void *a, int lda,
                                  void *x, int incx);
typedef void (*fb_ctrmv_fortran_slot_fn)(const int *order, const int *uplo,
                                         const int *trans, const int *diag,
                                         const int *n, const fb_complex_float_t *a,
                                         const int *lda, fb_complex_float_t *x,
                                         const int *incx);
typedef void (*fb_ztrmv_fortran_slot_fn)(const int *order, const int *uplo,
                                         const int *trans, const int *diag,
                                         const int *n, const fb_complex_double_t *a,
                                         const int *lda, fb_complex_double_t *x,
                                         const int *incx);
typedef void (*fb_ctrsv_cblas_fn)(fb_layout_t layout, fb_uplo_t uplo,
                                  fb_transpose_t trans, fb_diag_t diag,
                                  int n, const void *a, int lda,
                                  void *x, int incx);
typedef void (*fb_ztrsv_cblas_fn)(fb_layout_t layout, fb_uplo_t uplo,
                                  fb_transpose_t trans, fb_diag_t diag,
                                  int n, const void *a, int lda,
                                  void *x, int incx);
typedef fb_ctrmv_fortran_slot_fn fb_ctrsv_fortran_slot_fn;
typedef fb_ztrmv_fortran_slot_fn fb_ztrsv_fortran_slot_fn;

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
    int trans;
    int diag;
    int n;
    int lda;
    int incx;
    const fb_complex_float_t *a;
    fb_complex_float_t *x;
} g_ctrmv_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    fb_uplo_t uplo;
    fb_transpose_t trans;
    fb_diag_t diag;
    int n;
    int lda;
    int incx;
    const void *a;
    void *x;
} g_ctrmv_cblas_call;

static struct {
    int called;
    int order;
    int uplo;
    int trans;
    int diag;
    int n;
    int lda;
    int incx;
    const fb_complex_double_t *a;
    fb_complex_double_t *x;
} g_ztrmv_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    fb_uplo_t uplo;
    fb_transpose_t trans;
    fb_diag_t diag;
    int n;
    int lda;
    int incx;
    const void *a;
    void *x;
} g_ztrmv_cblas_call;

static struct {
    int called;
    int order;
    int uplo;
    int trans;
    int diag;
    int n;
    int lda;
    int incx;
    const fb_complex_float_t *a;
    fb_complex_float_t *x;
} g_ctrsv_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    fb_uplo_t uplo;
    fb_transpose_t trans;
    fb_diag_t diag;
    int n;
    int lda;
    int incx;
    const void *a;
    void *x;
} g_ctrsv_cblas_call;

static struct {
    int called;
    int order;
    int uplo;
    int trans;
    int diag;
    int n;
    int lda;
    int incx;
    const fb_complex_double_t *a;
    fb_complex_double_t *x;
} g_ztrsv_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    fb_uplo_t uplo;
    fb_transpose_t trans;
    fb_diag_t diag;
    int n;
    int lda;
    int incx;
    const void *a;
    void *x;
} g_ztrsv_cblas_call;

static void stub_ctrmv_fortran(const int *order, const int *uplo,
                               const int *trans, const int *diag,
                               const int *n, const fb_complex_float_t *a,
                               const int *lda, fb_complex_float_t *x,
                               const int *incx)
{
    g_ctrmv_fortran_call.called += 1;
    g_ctrmv_fortran_call.order = *order;
    g_ctrmv_fortran_call.uplo = *uplo;
    g_ctrmv_fortran_call.trans = *trans;
    g_ctrmv_fortran_call.diag = *diag;
    g_ctrmv_fortran_call.n = *n;
    g_ctrmv_fortran_call.lda = *lda;
    g_ctrmv_fortran_call.incx = *incx;
    g_ctrmv_fortran_call.a = a;
    g_ctrmv_fortran_call.x = x;
    x[0] = make_cf32(101.0f, -1.0f);
    x[1] = make_cf32(102.0f, -2.0f);
}

static void stub_ctrmv_cblas(fb_layout_t layout, fb_uplo_t uplo,
                             fb_transpose_t trans, fb_diag_t diag,
                             int n, const void *a, int lda,
                             void *x, int incx)
{
    fb_complex_float_t *x_values = (fb_complex_float_t *)x;

    g_ctrmv_cblas_call.called += 1;
    g_ctrmv_cblas_call.layout = layout;
    g_ctrmv_cblas_call.uplo = uplo;
    g_ctrmv_cblas_call.trans = trans;
    g_ctrmv_cblas_call.diag = diag;
    g_ctrmv_cblas_call.n = n;
    g_ctrmv_cblas_call.lda = lda;
    g_ctrmv_cblas_call.incx = incx;
    g_ctrmv_cblas_call.a = a;
    g_ctrmv_cblas_call.x = x;
    x_values[0] = make_cf32(111.0f, 1.0f);
    x_values[1] = make_cf32(112.0f, 2.0f);
}

static void stub_ztrmv_fortran(const int *order, const int *uplo,
                               const int *trans, const int *diag,
                               const int *n, const fb_complex_double_t *a,
                               const int *lda, fb_complex_double_t *x,
                               const int *incx)
{
    g_ztrmv_fortran_call.called += 1;
    g_ztrmv_fortran_call.order = *order;
    g_ztrmv_fortran_call.uplo = *uplo;
    g_ztrmv_fortran_call.trans = *trans;
    g_ztrmv_fortran_call.diag = *diag;
    g_ztrmv_fortran_call.n = *n;
    g_ztrmv_fortran_call.lda = *lda;
    g_ztrmv_fortran_call.incx = *incx;
    g_ztrmv_fortran_call.a = a;
    g_ztrmv_fortran_call.x = x;
    x[0] = make_cf64(121.0, -1.0);
    x[1] = make_cf64(122.0, -2.0);
}

static void stub_ztrmv_cblas(fb_layout_t layout, fb_uplo_t uplo,
                             fb_transpose_t trans, fb_diag_t diag,
                             int n, const void *a, int lda,
                             void *x, int incx)
{
    fb_complex_double_t *x_values = (fb_complex_double_t *)x;

    g_ztrmv_cblas_call.called += 1;
    g_ztrmv_cblas_call.layout = layout;
    g_ztrmv_cblas_call.uplo = uplo;
    g_ztrmv_cblas_call.trans = trans;
    g_ztrmv_cblas_call.diag = diag;
    g_ztrmv_cblas_call.n = n;
    g_ztrmv_cblas_call.lda = lda;
    g_ztrmv_cblas_call.incx = incx;
    g_ztrmv_cblas_call.a = a;
    g_ztrmv_cblas_call.x = x;
    x_values[0] = make_cf64(131.0, 1.0);
    x_values[1] = make_cf64(132.0, 2.0);
}

static void stub_ctrsv_fortran(const int *order, const int *uplo,
                               const int *trans, const int *diag,
                               const int *n, const fb_complex_float_t *a,
                               const int *lda, fb_complex_float_t *x,
                               const int *incx)
{
    g_ctrsv_fortran_call.called += 1;
    g_ctrsv_fortran_call.order = *order;
    g_ctrsv_fortran_call.uplo = *uplo;
    g_ctrsv_fortran_call.trans = *trans;
    g_ctrsv_fortran_call.diag = *diag;
    g_ctrsv_fortran_call.n = *n;
    g_ctrsv_fortran_call.lda = *lda;
    g_ctrsv_fortran_call.incx = *incx;
    g_ctrsv_fortran_call.a = a;
    g_ctrsv_fortran_call.x = x;
    x[0] = make_cf32(201.0f, -1.0f);
    x[1] = make_cf32(202.0f, -2.0f);
}

static void stub_ctrsv_cblas(fb_layout_t layout, fb_uplo_t uplo,
                             fb_transpose_t trans, fb_diag_t diag,
                             int n, const void *a, int lda,
                             void *x, int incx)
{
    fb_complex_float_t *x_values = (fb_complex_float_t *)x;

    g_ctrsv_cblas_call.called += 1;
    g_ctrsv_cblas_call.layout = layout;
    g_ctrsv_cblas_call.uplo = uplo;
    g_ctrsv_cblas_call.trans = trans;
    g_ctrsv_cblas_call.diag = diag;
    g_ctrsv_cblas_call.n = n;
    g_ctrsv_cblas_call.lda = lda;
    g_ctrsv_cblas_call.incx = incx;
    g_ctrsv_cblas_call.a = a;
    g_ctrsv_cblas_call.x = x;
    x_values[0] = make_cf32(211.0f, 1.0f);
    x_values[1] = make_cf32(212.0f, 2.0f);
}

static void stub_ztrsv_fortran(const int *order, const int *uplo,
                               const int *trans, const int *diag,
                               const int *n, const fb_complex_double_t *a,
                               const int *lda, fb_complex_double_t *x,
                               const int *incx)
{
    g_ztrsv_fortran_call.called += 1;
    g_ztrsv_fortran_call.order = *order;
    g_ztrsv_fortran_call.uplo = *uplo;
    g_ztrsv_fortran_call.trans = *trans;
    g_ztrsv_fortran_call.diag = *diag;
    g_ztrsv_fortran_call.n = *n;
    g_ztrsv_fortran_call.lda = *lda;
    g_ztrsv_fortran_call.incx = *incx;
    g_ztrsv_fortran_call.a = a;
    g_ztrsv_fortran_call.x = x;
    x[0] = make_cf64(221.0, -1.0);
    x[1] = make_cf64(222.0, -2.0);
}

static void stub_ztrsv_cblas(fb_layout_t layout, fb_uplo_t uplo,
                             fb_transpose_t trans, fb_diag_t diag,
                             int n, const void *a, int lda,
                             void *x, int incx)
{
    fb_complex_double_t *x_values = (fb_complex_double_t *)x;

    g_ztrsv_cblas_call.called += 1;
    g_ztrsv_cblas_call.layout = layout;
    g_ztrsv_cblas_call.uplo = uplo;
    g_ztrsv_cblas_call.trans = trans;
    g_ztrsv_cblas_call.diag = diag;
    g_ztrsv_cblas_call.n = n;
    g_ztrsv_cblas_call.lda = lda;
    g_ztrsv_cblas_call.incx = incx;
    g_ztrsv_cblas_call.a = a;
    g_ztrsv_cblas_call.x = x;
    x_values[0] = make_cf64(231.0, 1.0);
    x_values[1] = make_cf64(232.0, 2.0);
}

static int check_ctrmv_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_ctrmv_cblas_fn thunk = NULL;
    fb_complex_float_t a[4] = {
        make_cf32(1.0f, 1.0f), make_cf32(2.0f, 2.0f),
        make_cf32(3.0f, 3.0f), make_cf32(4.0f, 4.0f)
    };
    fb_complex_float_t x[2] = { make_cf32(5.0f, 5.0f), make_cf32(6.0f, 6.0f) };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_ctrmv_fortran_call, 0, sizeof(g_ctrmv_fortran_call));
    vtable.ext_ops[FB_OP_CTRMV][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_ctrmv_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CTRMV);
    thunk = (fb_ctrmv_cblas_fn)vtable.ext_ops[FB_OP_CTRMV][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CTRMV Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    thunk(FB_LAYOUT_ROW_MAJOR, FB_LOWER, FB_CONJ_TRANS, FB_NON_UNIT, 2, a, 2, x, 1);
    if (g_ctrmv_fortran_call.called != 1 ||
        g_ctrmv_fortran_call.order != FB_LAYOUT_ROW_MAJOR ||
        g_ctrmv_fortran_call.uplo != FB_LOWER ||
        g_ctrmv_fortran_call.trans != FB_CONJ_TRANS ||
        g_ctrmv_fortran_call.diag != FB_NON_UNIT ||
        g_ctrmv_fortran_call.n != 2 ||
        g_ctrmv_fortran_call.lda != 2 ||
        g_ctrmv_fortran_call.incx != 1 ||
        g_ctrmv_fortran_call.a != a ||
        g_ctrmv_fortran_call.x != x ||
        !cf32_eq(x[0], make_cf32(101.0f, -1.0f)) ||
        !cf32_eq(x[1], make_cf32(102.0f, -2.0f))) {
        fprintf(stderr, "[FAIL] CTRMV Fortran->CBLAS thunk did not forward complex triangular mat-vec arguments correctly\n");
        return 1;
    }

    printf("[PASS] CTRMV Fortran->CBLAS thunk forwards complex triangular mat-vec arguments unchanged\n");
    return 0;
}

static int check_ctrmv_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_ctrmv_fortran_slot_fn thunk = NULL;
    int order = FB_LAYOUT_COL_MAJOR;
    int uplo = FB_UPPER;
    int trans = FB_NO_TRANS;
    int diag = FB_UNIT;
    int n = 2;
    int lda = 2;
    int incx = 1;
    fb_complex_float_t a[4] = {
        make_cf32(7.0f, 7.0f), make_cf32(8.0f, 8.0f),
        make_cf32(9.0f, 9.0f), make_cf32(10.0f, 10.0f)
    };
    fb_complex_float_t x[2] = { make_cf32(11.0f, 11.0f), make_cf32(12.0f, 12.0f) };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_ctrmv_cblas_call, 0, sizeof(g_ctrmv_cblas_call));
    vtable.ext_ops[FB_OP_CTRMV][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_ctrmv_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CTRMV);
    thunk = (fb_ctrmv_fortran_slot_fn)vtable.ext_ops[FB_OP_CTRMV][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CTRMV CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&order, &uplo, &trans, &diag, &n, a, &lda, x, &incx);
    if (g_ctrmv_cblas_call.called != 1 ||
        g_ctrmv_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_ctrmv_cblas_call.uplo != FB_UPPER ||
        g_ctrmv_cblas_call.trans != FB_NO_TRANS ||
        g_ctrmv_cblas_call.diag != FB_UNIT ||
        g_ctrmv_cblas_call.n != 2 ||
        g_ctrmv_cblas_call.lda != 2 ||
        g_ctrmv_cblas_call.incx != 1 ||
        g_ctrmv_cblas_call.a != a ||
        g_ctrmv_cblas_call.x != x ||
        !cf32_eq(x[0], make_cf32(111.0f, 1.0f)) ||
        !cf32_eq(x[1], make_cf32(112.0f, 2.0f))) {
        fprintf(stderr, "[FAIL] CTRMV CBLAS->Fortran thunk did not map complex triangular mat-vec arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] CTRMV CBLAS->Fortran thunk maps complex triangular mat-vec arguments into the C entry\n");
    return 0;
}

static int check_ztrmv_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_ztrmv_cblas_fn thunk = NULL;
    fb_complex_double_t a[4] = {
        make_cf64(13.0, 13.0), make_cf64(14.0, 14.0),
        make_cf64(15.0, 15.0), make_cf64(16.0, 16.0)
    };
    fb_complex_double_t x[2] = { make_cf64(17.0, 17.0), make_cf64(18.0, 18.0) };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_ztrmv_fortran_call, 0, sizeof(g_ztrmv_fortran_call));
    vtable.ext_ops[FB_OP_ZTRMV][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_ztrmv_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_ZTRMV);
    thunk = (fb_ztrmv_cblas_fn)vtable.ext_ops[FB_OP_ZTRMV][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZTRMV Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    thunk(FB_LAYOUT_ROW_MAJOR, FB_LOWER, FB_CONJ_TRANS, FB_NON_UNIT, 2, a, 2, x, 1);
    if (g_ztrmv_fortran_call.called != 1 ||
        g_ztrmv_fortran_call.order != FB_LAYOUT_ROW_MAJOR ||
        g_ztrmv_fortran_call.uplo != FB_LOWER ||
        g_ztrmv_fortran_call.trans != FB_CONJ_TRANS ||
        g_ztrmv_fortran_call.diag != FB_NON_UNIT ||
        g_ztrmv_fortran_call.n != 2 ||
        g_ztrmv_fortran_call.lda != 2 ||
        g_ztrmv_fortran_call.incx != 1 ||
        g_ztrmv_fortran_call.a != a ||
        g_ztrmv_fortran_call.x != x ||
        !cf64_eq(x[0], make_cf64(121.0, -1.0)) ||
        !cf64_eq(x[1], make_cf64(122.0, -2.0))) {
        fprintf(stderr, "[FAIL] ZTRMV Fortran->CBLAS thunk did not forward double-complex triangular mat-vec arguments correctly\n");
        return 1;
    }

    printf("[PASS] ZTRMV Fortran->CBLAS thunk forwards double-complex triangular mat-vec arguments unchanged\n");
    return 0;
}

static int check_ztrmv_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_ztrmv_fortran_slot_fn thunk = NULL;
    int order = FB_LAYOUT_COL_MAJOR;
    int uplo = FB_UPPER;
    int trans = FB_NO_TRANS;
    int diag = FB_UNIT;
    int n = 2;
    int lda = 2;
    int incx = 1;
    fb_complex_double_t a[4] = {
        make_cf64(19.0, 19.0), make_cf64(20.0, 20.0),
        make_cf64(21.0, 21.0), make_cf64(22.0, 22.0)
    };
    fb_complex_double_t x[2] = { make_cf64(23.0, 23.0), make_cf64(24.0, 24.0) };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_ztrmv_cblas_call, 0, sizeof(g_ztrmv_cblas_call));
    vtable.ext_ops[FB_OP_ZTRMV][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_ztrmv_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_ZTRMV);
    thunk = (fb_ztrmv_fortran_slot_fn)vtable.ext_ops[FB_OP_ZTRMV][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZTRMV CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&order, &uplo, &trans, &diag, &n, a, &lda, x, &incx);
    if (g_ztrmv_cblas_call.called != 1 ||
        g_ztrmv_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_ztrmv_cblas_call.uplo != FB_UPPER ||
        g_ztrmv_cblas_call.trans != FB_NO_TRANS ||
        g_ztrmv_cblas_call.diag != FB_UNIT ||
        g_ztrmv_cblas_call.n != 2 ||
        g_ztrmv_cblas_call.lda != 2 ||
        g_ztrmv_cblas_call.incx != 1 ||
        g_ztrmv_cblas_call.a != a ||
        g_ztrmv_cblas_call.x != x ||
        !cf64_eq(x[0], make_cf64(131.0, 1.0)) ||
        !cf64_eq(x[1], make_cf64(132.0, 2.0))) {
        fprintf(stderr, "[FAIL] ZTRMV CBLAS->Fortran thunk did not map double-complex triangular mat-vec arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] ZTRMV CBLAS->Fortran thunk maps double-complex triangular mat-vec arguments into the C entry\n");
    return 0;
}

static int check_ctrsv_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_ctrsv_cblas_fn thunk = NULL;
    fb_complex_float_t a[4] = {
        make_cf32(25.0f, 25.0f), make_cf32(26.0f, 26.0f),
        make_cf32(27.0f, 27.0f), make_cf32(28.0f, 28.0f)
    };
    fb_complex_float_t x[2] = { make_cf32(29.0f, 29.0f), make_cf32(30.0f, 30.0f) };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_ctrsv_fortran_call, 0, sizeof(g_ctrsv_fortran_call));
    vtable.ext_ops[FB_OP_CTRSV][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_ctrsv_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CTRSV);
    thunk = (fb_ctrsv_cblas_fn)vtable.ext_ops[FB_OP_CTRSV][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CTRSV Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    thunk(FB_LAYOUT_ROW_MAJOR, FB_LOWER, FB_CONJ_TRANS, FB_NON_UNIT, 2, a, 2, x, 1);
    if (g_ctrsv_fortran_call.called != 1 ||
        g_ctrsv_fortran_call.order != FB_LAYOUT_ROW_MAJOR ||
        g_ctrsv_fortran_call.uplo != FB_LOWER ||
        g_ctrsv_fortran_call.trans != FB_CONJ_TRANS ||
        g_ctrsv_fortran_call.diag != FB_NON_UNIT ||
        g_ctrsv_fortran_call.n != 2 ||
        g_ctrsv_fortran_call.lda != 2 ||
        g_ctrsv_fortran_call.incx != 1 ||
        g_ctrsv_fortran_call.a != a ||
        g_ctrsv_fortran_call.x != x ||
        !cf32_eq(x[0], make_cf32(201.0f, -1.0f)) ||
        !cf32_eq(x[1], make_cf32(202.0f, -2.0f))) {
        fprintf(stderr, "[FAIL] CTRSV Fortran->CBLAS thunk did not forward complex triangular solve-vector arguments correctly\n");
        return 1;
    }

    printf("[PASS] CTRSV Fortran->CBLAS thunk forwards complex triangular solve-vector arguments unchanged\n");
    return 0;
}

static int check_ctrsv_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_ctrsv_fortran_slot_fn thunk = NULL;
    int order = FB_LAYOUT_COL_MAJOR;
    int uplo = FB_UPPER;
    int trans = FB_NO_TRANS;
    int diag = FB_UNIT;
    int n = 2;
    int lda = 2;
    int incx = 1;
    fb_complex_float_t a[4] = {
        make_cf32(31.0f, 31.0f), make_cf32(32.0f, 32.0f),
        make_cf32(33.0f, 33.0f), make_cf32(34.0f, 34.0f)
    };
    fb_complex_float_t x[2] = { make_cf32(35.0f, 35.0f), make_cf32(36.0f, 36.0f) };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_ctrsv_cblas_call, 0, sizeof(g_ctrsv_cblas_call));
    vtable.ext_ops[FB_OP_CTRSV][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_ctrsv_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CTRSV);
    thunk = (fb_ctrsv_fortran_slot_fn)vtable.ext_ops[FB_OP_CTRSV][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CTRSV CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&order, &uplo, &trans, &diag, &n, a, &lda, x, &incx);
    if (g_ctrsv_cblas_call.called != 1 ||
        g_ctrsv_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_ctrsv_cblas_call.uplo != FB_UPPER ||
        g_ctrsv_cblas_call.trans != FB_NO_TRANS ||
        g_ctrsv_cblas_call.diag != FB_UNIT ||
        g_ctrsv_cblas_call.n != 2 ||
        g_ctrsv_cblas_call.lda != 2 ||
        g_ctrsv_cblas_call.incx != 1 ||
        g_ctrsv_cblas_call.a != a ||
        g_ctrsv_cblas_call.x != x ||
        !cf32_eq(x[0], make_cf32(211.0f, 1.0f)) ||
        !cf32_eq(x[1], make_cf32(212.0f, 2.0f))) {
        fprintf(stderr, "[FAIL] CTRSV CBLAS->Fortran thunk did not map complex triangular solve-vector arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] CTRSV CBLAS->Fortran thunk maps complex triangular solve-vector arguments into the C entry\n");
    return 0;
}

static int check_ztrsv_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_ztrsv_cblas_fn thunk = NULL;
    fb_complex_double_t a[4] = {
        make_cf64(37.0, 37.0), make_cf64(38.0, 38.0),
        make_cf64(39.0, 39.0), make_cf64(40.0, 40.0)
    };
    fb_complex_double_t x[2] = { make_cf64(41.0, 41.0), make_cf64(42.0, 42.0) };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_ztrsv_fortran_call, 0, sizeof(g_ztrsv_fortran_call));
    vtable.ext_ops[FB_OP_ZTRSV][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_ztrsv_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_ZTRSV);
    thunk = (fb_ztrsv_cblas_fn)vtable.ext_ops[FB_OP_ZTRSV][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZTRSV Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    thunk(FB_LAYOUT_ROW_MAJOR, FB_LOWER, FB_CONJ_TRANS, FB_NON_UNIT, 2, a, 2, x, 1);
    if (g_ztrsv_fortran_call.called != 1 ||
        g_ztrsv_fortran_call.order != FB_LAYOUT_ROW_MAJOR ||
        g_ztrsv_fortran_call.uplo != FB_LOWER ||
        g_ztrsv_fortran_call.trans != FB_CONJ_TRANS ||
        g_ztrsv_fortran_call.diag != FB_NON_UNIT ||
        g_ztrsv_fortran_call.n != 2 ||
        g_ztrsv_fortran_call.lda != 2 ||
        g_ztrsv_fortran_call.incx != 1 ||
        g_ztrsv_fortran_call.a != a ||
        g_ztrsv_fortran_call.x != x ||
        !cf64_eq(x[0], make_cf64(221.0, -1.0)) ||
        !cf64_eq(x[1], make_cf64(222.0, -2.0))) {
        fprintf(stderr, "[FAIL] ZTRSV Fortran->CBLAS thunk did not forward double-complex triangular solve-vector arguments correctly\n");
        return 1;
    }

    printf("[PASS] ZTRSV Fortran->CBLAS thunk forwards double-complex triangular solve-vector arguments unchanged\n");
    return 0;
}

static int check_ztrsv_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_ztrsv_fortran_slot_fn thunk = NULL;
    int order = FB_LAYOUT_COL_MAJOR;
    int uplo = FB_UPPER;
    int trans = FB_NO_TRANS;
    int diag = FB_UNIT;
    int n = 2;
    int lda = 2;
    int incx = 1;
    fb_complex_double_t a[4] = {
        make_cf64(43.0, 43.0), make_cf64(44.0, 44.0),
        make_cf64(45.0, 45.0), make_cf64(46.0, 46.0)
    };
    fb_complex_double_t x[2] = { make_cf64(47.0, 47.0), make_cf64(48.0, 48.0) };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_ztrsv_cblas_call, 0, sizeof(g_ztrsv_cblas_call));
    vtable.ext_ops[FB_OP_ZTRSV][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_ztrsv_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_ZTRSV);
    thunk = (fb_ztrsv_fortran_slot_fn)vtable.ext_ops[FB_OP_ZTRSV][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZTRSV CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&order, &uplo, &trans, &diag, &n, a, &lda, x, &incx);
    if (g_ztrsv_cblas_call.called != 1 ||
        g_ztrsv_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_ztrsv_cblas_call.uplo != FB_UPPER ||
        g_ztrsv_cblas_call.trans != FB_NO_TRANS ||
        g_ztrsv_cblas_call.diag != FB_UNIT ||
        g_ztrsv_cblas_call.n != 2 ||
        g_ztrsv_cblas_call.lda != 2 ||
        g_ztrsv_cblas_call.incx != 1 ||
        g_ztrsv_cblas_call.a != a ||
        g_ztrsv_cblas_call.x != x ||
        !cf64_eq(x[0], make_cf64(231.0, 1.0)) ||
        !cf64_eq(x[1], make_cf64(232.0, 2.0))) {
        fprintf(stderr, "[FAIL] ZTRSV CBLAS->Fortran thunk did not map double-complex triangular solve-vector arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] ZTRSV CBLAS->Fortran thunk maps double-complex triangular solve-vector arguments into the C entry\n");
    return 0;
}

int main(void)
{
    int status = 0;

    status |= check_ctrmv_fortran_to_cblas();
    status |= check_ctrmv_cblas_to_fortran();
    status |= check_ztrmv_fortran_to_cblas();
    status |= check_ztrmv_cblas_to_fortran();
    status |= check_ctrsv_fortran_to_cblas();
    status |= check_ctrsv_cblas_to_fortran();
    status |= check_ztrsv_fortran_to_cblas();
    status |= check_ztrsv_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}