#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef void (*fb_ctrsm_cblas_fn)(fb_layout_t layout, fb_side_t side,
                                  fb_uplo_t uplo, fb_transpose_t trans,
                                  fb_diag_t diag, int m, int n,
                                  const void *alpha, const void *a, int lda,
                                  void *b, int ldb);
typedef void (*fb_ztrsm_cblas_fn)(fb_layout_t layout, fb_side_t side,
                                  fb_uplo_t uplo, fb_transpose_t trans,
                                  fb_diag_t diag, int m, int n,
                                  const void *alpha, const void *a, int lda,
                                  void *b, int ldb);

typedef void (*fb_ctrsm_fortran_slot_fn)(const int *order, const int *side,
                                         const int *uplo, const int *trans,
                                         const int *diag, const int *m,
                                         const int *n, const void *alpha,
                                         const void *a, const int *lda,
                                         void *b, const int *ldb);
typedef void (*fb_ztrsm_fortran_slot_fn)(const int *order, const int *side,
                                         const int *uplo, const int *trans,
                                         const int *diag, const int *m,
                                         const int *n, const void *alpha,
                                         const void *a, const int *lda,
                                         void *b, const int *ldb);

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
    int side;
    int uplo;
    int trans;
    int diag;
    int m;
    int n;
    int lda;
    int ldb;
    const void *alpha;
    fb_complex_float_t alpha_value;
    const void *a;
    void *b;
} g_ctrsm_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    fb_side_t side;
    fb_uplo_t uplo;
    fb_transpose_t trans;
    fb_diag_t diag;
    int m;
    int n;
    int lda;
    int ldb;
    const void *alpha;
    fb_complex_float_t alpha_value;
    const void *a;
    void *b;
} g_ctrsm_cblas_call;

static struct {
    int called;
    int order;
    int side;
    int uplo;
    int trans;
    int diag;
    int m;
    int n;
    int lda;
    int ldb;
    const void *alpha;
    fb_complex_double_t alpha_value;
    const void *a;
    void *b;
} g_ztrsm_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    fb_side_t side;
    fb_uplo_t uplo;
    fb_transpose_t trans;
    fb_diag_t diag;
    int m;
    int n;
    int lda;
    int ldb;
    const void *alpha;
    fb_complex_double_t alpha_value;
    const void *a;
    void *b;
} g_ztrsm_cblas_call;

static void stub_ctrsm_fortran(const int *order, const int *side,
                               const int *uplo, const int *trans,
                               const int *diag, const int *m, const int *n,
                               const void *alpha, const void *a,
                               const int *lda, void *b, const int *ldb)
{
    fb_complex_float_t *b_values = (fb_complex_float_t *)b;

    g_ctrsm_fortran_call.called += 1;
    g_ctrsm_fortran_call.order = *order;
    g_ctrsm_fortran_call.side = *side;
    g_ctrsm_fortran_call.uplo = *uplo;
    g_ctrsm_fortran_call.trans = *trans;
    g_ctrsm_fortran_call.diag = *diag;
    g_ctrsm_fortran_call.m = *m;
    g_ctrsm_fortran_call.n = *n;
    g_ctrsm_fortran_call.lda = *lda;
    g_ctrsm_fortran_call.ldb = *ldb;
    g_ctrsm_fortran_call.alpha = alpha;
    g_ctrsm_fortran_call.alpha_value = *(const fb_complex_float_t *)alpha;
    g_ctrsm_fortran_call.a = a;
    g_ctrsm_fortran_call.b = b;

    b_values[0] = make_cf32(401.0f, -1.0f);
    b_values[1] = make_cf32(402.0f, -2.0f);
    b_values[2] = make_cf32(403.0f, -3.0f);
    b_values[3] = make_cf32(404.0f, -4.0f);
}

static void stub_ctrsm_cblas(fb_layout_t layout, fb_side_t side,
                             fb_uplo_t uplo, fb_transpose_t trans,
                             fb_diag_t diag, int m, int n,
                             const void *alpha, const void *a, int lda,
                             void *b, int ldb)
{
    fb_complex_float_t *b_values = (fb_complex_float_t *)b;

    g_ctrsm_cblas_call.called += 1;
    g_ctrsm_cblas_call.layout = layout;
    g_ctrsm_cblas_call.side = side;
    g_ctrsm_cblas_call.uplo = uplo;
    g_ctrsm_cblas_call.trans = trans;
    g_ctrsm_cblas_call.diag = diag;
    g_ctrsm_cblas_call.m = m;
    g_ctrsm_cblas_call.n = n;
    g_ctrsm_cblas_call.lda = lda;
    g_ctrsm_cblas_call.ldb = ldb;
    g_ctrsm_cblas_call.alpha = alpha;
    g_ctrsm_cblas_call.alpha_value = *(const fb_complex_float_t *)alpha;
    g_ctrsm_cblas_call.a = a;
    g_ctrsm_cblas_call.b = b;

    b_values[0] = make_cf32(411.0f, 1.0f);
    b_values[1] = make_cf32(412.0f, 2.0f);
    b_values[2] = make_cf32(413.0f, 3.0f);
    b_values[3] = make_cf32(414.0f, 4.0f);
}

static void stub_ztrsm_fortran(const int *order, const int *side,
                               const int *uplo, const int *trans,
                               const int *diag, const int *m, const int *n,
                               const void *alpha, const void *a,
                               const int *lda, void *b, const int *ldb)
{
    fb_complex_double_t *b_values = (fb_complex_double_t *)b;

    g_ztrsm_fortran_call.called += 1;
    g_ztrsm_fortran_call.order = *order;
    g_ztrsm_fortran_call.side = *side;
    g_ztrsm_fortran_call.uplo = *uplo;
    g_ztrsm_fortran_call.trans = *trans;
    g_ztrsm_fortran_call.diag = *diag;
    g_ztrsm_fortran_call.m = *m;
    g_ztrsm_fortran_call.n = *n;
    g_ztrsm_fortran_call.lda = *lda;
    g_ztrsm_fortran_call.ldb = *ldb;
    g_ztrsm_fortran_call.alpha = alpha;
    g_ztrsm_fortran_call.alpha_value = *(const fb_complex_double_t *)alpha;
    g_ztrsm_fortran_call.a = a;
    g_ztrsm_fortran_call.b = b;

    b_values[0] = make_cf64(421.0, -1.0);
    b_values[1] = make_cf64(422.0, -2.0);
    b_values[2] = make_cf64(423.0, -3.0);
    b_values[3] = make_cf64(424.0, -4.0);
}

static void stub_ztrsm_cblas(fb_layout_t layout, fb_side_t side,
                             fb_uplo_t uplo, fb_transpose_t trans,
                             fb_diag_t diag, int m, int n,
                             const void *alpha, const void *a, int lda,
                             void *b, int ldb)
{
    fb_complex_double_t *b_values = (fb_complex_double_t *)b;

    g_ztrsm_cblas_call.called += 1;
    g_ztrsm_cblas_call.layout = layout;
    g_ztrsm_cblas_call.side = side;
    g_ztrsm_cblas_call.uplo = uplo;
    g_ztrsm_cblas_call.trans = trans;
    g_ztrsm_cblas_call.diag = diag;
    g_ztrsm_cblas_call.m = m;
    g_ztrsm_cblas_call.n = n;
    g_ztrsm_cblas_call.lda = lda;
    g_ztrsm_cblas_call.ldb = ldb;
    g_ztrsm_cblas_call.alpha = alpha;
    g_ztrsm_cblas_call.alpha_value = *(const fb_complex_double_t *)alpha;
    g_ztrsm_cblas_call.a = a;
    g_ztrsm_cblas_call.b = b;

    b_values[0] = make_cf64(431.0, 1.0);
    b_values[1] = make_cf64(432.0, 2.0);
    b_values[2] = make_cf64(433.0, 3.0);
    b_values[3] = make_cf64(434.0, 4.0);
}

static int check_ctrsm_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_ctrsm_cblas_fn thunk = NULL;
    fb_complex_float_t alpha = make_cf32(1.5f, -0.5f);
    fb_complex_float_t a[4] = {
        make_cf32(1.0f, -1.0f), make_cf32(2.0f, -2.0f),
        make_cf32(3.0f, -3.0f), make_cf32(4.0f, -4.0f)
    };
    fb_complex_float_t b[4] = {
        make_cf32(5.0f, 1.0f), make_cf32(6.0f, 2.0f),
        make_cf32(7.0f, 3.0f), make_cf32(8.0f, 4.0f)
    };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_ctrsm_fortran_call, 0, sizeof(g_ctrsm_fortran_call));

    vtable.ext_ops[FB_OP_CTRSM][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_ctrsm_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CTRSM);

    thunk = (fb_ctrsm_cblas_fn)vtable.ext_ops[FB_OP_CTRSM][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CTRSM Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    thunk(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_LOWER, FB_CONJ_TRANS, FB_NON_UNIT,
          2, 2, &alpha, a, 2, b, 2);
    if (g_ctrsm_fortran_call.called != 1 ||
        g_ctrsm_fortran_call.order != FB_LAYOUT_ROW_MAJOR ||
        g_ctrsm_fortran_call.side != FB_LEFT ||
        g_ctrsm_fortran_call.uplo != FB_LOWER ||
        g_ctrsm_fortran_call.trans != FB_CONJ_TRANS ||
        g_ctrsm_fortran_call.diag != FB_NON_UNIT ||
        g_ctrsm_fortran_call.m != 2 ||
        g_ctrsm_fortran_call.n != 2 ||
        g_ctrsm_fortran_call.lda != 2 ||
        g_ctrsm_fortran_call.ldb != 2 ||
        g_ctrsm_fortran_call.alpha != &alpha ||
        !cf32_eq(g_ctrsm_fortran_call.alpha_value, alpha) ||
        g_ctrsm_fortran_call.a != a ||
        g_ctrsm_fortran_call.b != b ||
        !cf32_eq(b[0], make_cf32(401.0f, -1.0f)) ||
        !cf32_eq(b[1], make_cf32(402.0f, -2.0f)) ||
        !cf32_eq(b[2], make_cf32(403.0f, -3.0f)) ||
        !cf32_eq(b[3], make_cf32(404.0f, -4.0f))) {
        fprintf(stderr, "[FAIL] CTRSM Fortran->CBLAS thunk did not forward complex triangular solve arguments correctly\n");
        return 1;
    }

    printf("[PASS] CTRSM Fortran->CBLAS thunk forwards complex triangular solve arguments unchanged\n");
    return 0;
}

static int check_ctrsm_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_ctrsm_fortran_slot_fn thunk = NULL;
    int order = FB_LAYOUT_COL_MAJOR;
    int side = FB_RIGHT;
    int uplo = FB_UPPER;
    int trans = FB_NO_TRANS;
    int diag = FB_UNIT;
    int m = 2;
    int n = 2;
    int lda = 2;
    int ldb = 2;
    fb_complex_float_t alpha = make_cf32(2.5f, -1.5f);
    fb_complex_float_t a[4] = {
        make_cf32(9.0f, 1.0f), make_cf32(10.0f, 2.0f),
        make_cf32(11.0f, 3.0f), make_cf32(12.0f, 4.0f)
    };
    fb_complex_float_t b[4] = {
        make_cf32(13.0f, -1.0f), make_cf32(14.0f, -2.0f),
        make_cf32(15.0f, -3.0f), make_cf32(16.0f, -4.0f)
    };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_ctrsm_cblas_call, 0, sizeof(g_ctrsm_cblas_call));

    vtable.ext_ops[FB_OP_CTRSM][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_ctrsm_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CTRSM);

    thunk = (fb_ctrsm_fortran_slot_fn)vtable.ext_ops[FB_OP_CTRSM][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CTRSM CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&order, &side, &uplo, &trans, &diag, &m, &n, &alpha, a, &lda, b, &ldb);
    if (g_ctrsm_cblas_call.called != 1 ||
        g_ctrsm_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_ctrsm_cblas_call.side != FB_RIGHT ||
        g_ctrsm_cblas_call.uplo != FB_UPPER ||
        g_ctrsm_cblas_call.trans != FB_NO_TRANS ||
        g_ctrsm_cblas_call.diag != FB_UNIT ||
        g_ctrsm_cblas_call.m != 2 ||
        g_ctrsm_cblas_call.n != 2 ||
        g_ctrsm_cblas_call.lda != 2 ||
        g_ctrsm_cblas_call.ldb != 2 ||
        g_ctrsm_cblas_call.alpha != &alpha ||
        !cf32_eq(g_ctrsm_cblas_call.alpha_value, alpha) ||
        g_ctrsm_cblas_call.a != a ||
        g_ctrsm_cblas_call.b != b ||
        !cf32_eq(b[0], make_cf32(411.0f, 1.0f)) ||
        !cf32_eq(b[1], make_cf32(412.0f, 2.0f)) ||
        !cf32_eq(b[2], make_cf32(413.0f, 3.0f)) ||
        !cf32_eq(b[3], make_cf32(414.0f, 4.0f))) {
        fprintf(stderr, "[FAIL] CTRSM CBLAS->Fortran thunk did not map complex triangular solve arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] CTRSM CBLAS->Fortran thunk maps complex triangular solve arguments into the C entry\n");
    return 0;
}

static int check_ztrsm_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_ztrsm_cblas_fn thunk = NULL;
    fb_complex_double_t alpha = make_cf64(3.5, -0.5);
    fb_complex_double_t a[4] = {
        make_cf64(17.0, -1.0), make_cf64(18.0, -2.0),
        make_cf64(19.0, -3.0), make_cf64(20.0, -4.0)
    };
    fb_complex_double_t b[4] = {
        make_cf64(21.0, 1.0), make_cf64(22.0, 2.0),
        make_cf64(23.0, 3.0), make_cf64(24.0, 4.0)
    };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_ztrsm_fortran_call, 0, sizeof(g_ztrsm_fortran_call));

    vtable.ext_ops[FB_OP_ZTRSM][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_ztrsm_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_ZTRSM);

    thunk = (fb_ztrsm_cblas_fn)vtable.ext_ops[FB_OP_ZTRSM][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZTRSM Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    thunk(FB_LAYOUT_ROW_MAJOR, FB_LEFT, FB_LOWER, FB_CONJ_TRANS, FB_NON_UNIT,
          2, 2, &alpha, a, 2, b, 2);
    if (g_ztrsm_fortran_call.called != 1 ||
        g_ztrsm_fortran_call.order != FB_LAYOUT_ROW_MAJOR ||
        g_ztrsm_fortran_call.side != FB_LEFT ||
        g_ztrsm_fortran_call.uplo != FB_LOWER ||
        g_ztrsm_fortran_call.trans != FB_CONJ_TRANS ||
        g_ztrsm_fortran_call.diag != FB_NON_UNIT ||
        g_ztrsm_fortran_call.m != 2 ||
        g_ztrsm_fortran_call.n != 2 ||
        g_ztrsm_fortran_call.lda != 2 ||
        g_ztrsm_fortran_call.ldb != 2 ||
        g_ztrsm_fortran_call.alpha != &alpha ||
        !cf64_eq(g_ztrsm_fortran_call.alpha_value, alpha) ||
        g_ztrsm_fortran_call.a != a ||
        g_ztrsm_fortran_call.b != b ||
        !cf64_eq(b[0], make_cf64(421.0, -1.0)) ||
        !cf64_eq(b[1], make_cf64(422.0, -2.0)) ||
        !cf64_eq(b[2], make_cf64(423.0, -3.0)) ||
        !cf64_eq(b[3], make_cf64(424.0, -4.0))) {
        fprintf(stderr, "[FAIL] ZTRSM Fortran->CBLAS thunk did not forward double-complex triangular solve arguments correctly\n");
        return 1;
    }

    printf("[PASS] ZTRSM Fortran->CBLAS thunk forwards double-complex triangular solve arguments unchanged\n");
    return 0;
}

static int check_ztrsm_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_ztrsm_fortran_slot_fn thunk = NULL;
    int order = FB_LAYOUT_COL_MAJOR;
    int side = FB_RIGHT;
    int uplo = FB_UPPER;
    int trans = FB_NO_TRANS;
    int diag = FB_UNIT;
    int m = 2;
    int n = 2;
    int lda = 2;
    int ldb = 2;
    fb_complex_double_t alpha = make_cf64(4.5, -1.5);
    fb_complex_double_t a[4] = {
        make_cf64(25.0, 1.0), make_cf64(26.0, 2.0),
        make_cf64(27.0, 3.0), make_cf64(28.0, 4.0)
    };
    fb_complex_double_t b[4] = {
        make_cf64(29.0, -1.0), make_cf64(30.0, -2.0),
        make_cf64(31.0, -3.0), make_cf64(32.0, -4.0)
    };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_ztrsm_cblas_call, 0, sizeof(g_ztrsm_cblas_call));

    vtable.ext_ops[FB_OP_ZTRSM][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_ztrsm_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_ZTRSM);

    thunk = (fb_ztrsm_fortran_slot_fn)vtable.ext_ops[FB_OP_ZTRSM][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZTRSM CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&order, &side, &uplo, &trans, &diag, &m, &n, &alpha, a, &lda, b, &ldb);
    if (g_ztrsm_cblas_call.called != 1 ||
        g_ztrsm_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_ztrsm_cblas_call.side != FB_RIGHT ||
        g_ztrsm_cblas_call.uplo != FB_UPPER ||
        g_ztrsm_cblas_call.trans != FB_NO_TRANS ||
        g_ztrsm_cblas_call.diag != FB_UNIT ||
        g_ztrsm_cblas_call.m != 2 ||
        g_ztrsm_cblas_call.n != 2 ||
        g_ztrsm_cblas_call.lda != 2 ||
        g_ztrsm_cblas_call.ldb != 2 ||
        g_ztrsm_cblas_call.alpha != &alpha ||
        !cf64_eq(g_ztrsm_cblas_call.alpha_value, alpha) ||
        g_ztrsm_cblas_call.a != a ||
        g_ztrsm_cblas_call.b != b ||
        !cf64_eq(b[0], make_cf64(431.0, 1.0)) ||
        !cf64_eq(b[1], make_cf64(432.0, 2.0)) ||
        !cf64_eq(b[2], make_cf64(433.0, 3.0)) ||
        !cf64_eq(b[3], make_cf64(434.0, 4.0))) {
        fprintf(stderr, "[FAIL] ZTRSM CBLAS->Fortran thunk did not map double-complex triangular solve arguments into the C entry\n");
        return 1;
    }

    printf("[PASS] ZTRSM CBLAS->Fortran thunk maps double-complex triangular solve arguments into the C entry\n");
    return 0;
}

int main(void)
{
    int status = 0;

    status |= check_ctrsm_fortran_to_cblas();
    status |= check_ctrsm_cblas_to_fortran();
    status |= check_ztrsm_fortran_to_cblas();
    status |= check_ztrsm_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}