#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef int (*fb_sgbsvx_cblas_fn)(fb_layout_t layout, char fact, char trans,
                                  int n, int kl, int ku, int nrhs, float *ab,
                                  int ldab, float *afb, int ldafb, int *ipiv,
                                  char *equed, float *r, float *c, float *b,
                                  int ldb, float *x, int ldx, float *rcond,
                                  float *ferr, float *berr, float *rpvgrw);
typedef int (*fb_cgbsvx_cblas_fn)(fb_layout_t layout, char fact, char trans,
                                  int n, int kl, int ku, int nrhs,
                                  fb_complex_float_t *ab, int ldab,
                                  fb_complex_float_t *afb, int ldafb,
                                  int *ipiv, char *equed, float *r, float *c,
                                  fb_complex_float_t *b, int ldb,
                                  fb_complex_float_t *x, int ldx,
                                  float *rcond, float *ferr, float *berr,
                                  float *rpvgrw);

typedef void (*fb_sgbsvx_fortran_slot_fn)(char *fact, char *trans, int *n,
                                          int *kl, int *ku, int *nrhs,
                                          float *ab, int *ldab, float *afb,
                                          int *ldafb, int *ipiv, char *equed,
                                          float *r, float *c, float *b,
                                          int *ldb, float *x, int *ldx,
                                          float *rcond, float *ferr,
                                          float *berr, float *work,
                                          int *iwork, int *info);
typedef void (*fb_cgbsvx_fortran_slot_fn)(char *fact, char *trans, int *n,
                                          int *kl, int *ku, int *nrhs,
                                          fb_complex_float_t *ab, int *ldab,
                                          fb_complex_float_t *afb, int *ldafb,
                                          int *ipiv, char *equed, float *r,
                                          float *c, fb_complex_float_t *b,
                                          int *ldb, fb_complex_float_t *x,
                                          int *ldx, float *rcond, float *ferr,
                                          float *berr,
                                          fb_complex_float_t *work,
                                          float *rwork, int *info);

static struct {
    int calls;
    char fact;
    char trans;
    char equed_before;
    int n;
    int kl;
    int ku;
    int nrhs;
    int ldab;
    int ldafb;
    int ldb;
    int ldx;
    int work_seen;
    int aux_seen;
    int ipiv_snapshot[4];
    float ab_snapshot[16];
    float afb_snapshot[16];
    float b_snapshot[8];
} g_sgbsvx_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    char fact;
    char trans;
    int n;
    int kl;
    int ku;
    int nrhs;
    int ldab;
    int ldafb;
    int ldb;
    int ldx;
    int *ipiv;
    float *ab;
    float *afb;
    char *equed;
    float *r;
    float *c;
    float *b;
    float *x;
    float *rcond;
    float *ferr;
    float *berr;
    float *rpvgrw;
} g_sgbsvx_cblas_call;

static struct {
    int calls;
    char fact;
    char trans;
    char equed_before;
    int n;
    int kl;
    int ku;
    int nrhs;
    int ldab;
    int ldafb;
    int ldb;
    int ldx;
    int work_seen;
    int aux_seen;
    int ipiv_snapshot[4];
    fb_complex_float_t ab_snapshot[16];
    fb_complex_float_t afb_snapshot[16];
    fb_complex_float_t b_snapshot[8];
} g_cgbsvx_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    char fact;
    char trans;
    int n;
    int kl;
    int ku;
    int nrhs;
    int ldab;
    int ldafb;
    int ldb;
    int ldx;
    int *ipiv;
    fb_complex_float_t *ab;
    fb_complex_float_t *afb;
    char *equed;
    float *r;
    float *c;
    fb_complex_float_t *b;
    fb_complex_float_t *x;
    float *rcond;
    float *ferr;
    float *berr;
    float *rpvgrw;
} g_cgbsvx_cblas_call;

static fb_complex_float_t make_cf32(float real_value, float imag_value)
{
    fb_complex_float_t value;
    __real__ value = real_value;
    __imag__ value = imag_value;
    return value;
}

static int cf32_eq(fb_complex_float_t lhs, fb_complex_float_t rhs)
{
    return __real__ lhs == __real__ rhs && __imag__ lhs == __imag__ rhs;
}

static int float_array_eq(const float *lhs, const float *rhs, size_t count)
{
    size_t index = 0;

    for (index = 0; index < count; ++index) {
        if (lhs[index] != rhs[index]) {
            return 0;
        }
    }

    return 1;
}

static int int_array_eq(const int *lhs, const int *rhs, size_t count)
{
    size_t index = 0;

    for (index = 0; index < count; ++index) {
        if (lhs[index] != rhs[index]) {
            return 0;
        }
    }

    return 1;
}

static int cf32_array_eq(const fb_complex_float_t *lhs,
                         const fb_complex_float_t *rhs, size_t count)
{
    size_t index = 0;

    for (index = 0; index < count; ++index) {
        if (!cf32_eq(lhs[index], rhs[index])) {
            return 0;
        }
    }

    return 1;
}

static void stub_sgbsvx_fortran(char *fact, char *trans, int *n, int *kl,
                                int *ku, int *nrhs, float *ab, int *ldab,
                                float *afb, int *ldafb, int *ipiv,
                                char *equed, float *r, float *c, float *b,
                                int *ldb, float *x, int *ldx, float *rcond,
                                float *ferr, float *berr, float *work,
                                int *iwork, int *info)
{
    static const int ipiv_out[4] = { 4, 3, 2, 1 };
    static const float ab_out_col[16] = {
        0.0f, 0.0f, 611.0f, 711.0f,
        0.0f, 511.0f, 612.0f, 712.0f,
        0.0f, 512.0f, 613.0f, 713.0f,
        0.0f, 513.0f, 614.0f, 0.0f
    };
    static const float afb_out_col[16] = {
        0.0f, 0.0f, 631.0f, 641.0f,
        0.0f, 621.0f, 632.0f, 642.0f,
        612.0f, 622.0f, 633.0f, 643.0f,
        613.0f, 623.0f, 634.0f, 0.0f
    };
    static const float b_out_col[8] = {
        810.0f, 830.0f, 850.0f, 870.0f,
        820.0f, 840.0f, 860.0f, 880.0f
    };
    static const float x_out_col[8] = {
        910.0f, 930.0f, 950.0f, 970.0f,
        920.0f, 940.0f, 960.0f, 980.0f
    };

    g_sgbsvx_fortran_call.calls += 1;
    g_sgbsvx_fortran_call.fact = *fact;
    g_sgbsvx_fortran_call.trans = *trans;
    g_sgbsvx_fortran_call.equed_before = *equed;
    g_sgbsvx_fortran_call.n = *n;
    g_sgbsvx_fortran_call.kl = *kl;
    g_sgbsvx_fortran_call.ku = *ku;
    g_sgbsvx_fortran_call.nrhs = *nrhs;
    g_sgbsvx_fortran_call.ldab = *ldab;
    g_sgbsvx_fortran_call.ldafb = *ldafb;
    g_sgbsvx_fortran_call.ldb = *ldb;
    g_sgbsvx_fortran_call.ldx = *ldx;
    g_sgbsvx_fortran_call.work_seen = (work != NULL);
    g_sgbsvx_fortran_call.aux_seen = (iwork != NULL);
    memcpy(g_sgbsvx_fortran_call.ipiv_snapshot, ipiv,
           sizeof(g_sgbsvx_fortran_call.ipiv_snapshot));
    memcpy(g_sgbsvx_fortran_call.ab_snapshot, ab,
           sizeof(g_sgbsvx_fortran_call.ab_snapshot));
    memcpy(g_sgbsvx_fortran_call.afb_snapshot, afb,
           sizeof(g_sgbsvx_fortran_call.afb_snapshot));
    memcpy(g_sgbsvx_fortran_call.b_snapshot, b,
           sizeof(g_sgbsvx_fortran_call.b_snapshot));

    memcpy(ab, ab_out_col, sizeof(ab_out_col));
    memcpy(afb, afb_out_col, sizeof(afb_out_col));
    memcpy(b, b_out_col, sizeof(b_out_col));
    memcpy(x, x_out_col, sizeof(x_out_col));
    memcpy(ipiv, ipiv_out, sizeof(ipiv_out));
    *equed = 'B';
    r[0] = 1.1f;
    c[0] = 2.1f;
    *rcond = 0.25f;
    ferr[0] = 0.5f;
    ferr[1] = 0.6f;
    berr[0] = 0.05f;
    berr[1] = 0.06f;
    work[0] = 0.875f;
    *info = 0;
}

static int stub_sgbsvx_cblas(fb_layout_t layout, char fact, char trans, int n,
                             int kl, int ku, int nrhs, float *ab, int ldab,
                             float *afb, int ldafb, int *ipiv, char *equed,
                             float *r, float *c, float *b, int ldb, float *x,
                             int ldx, float *rcond, float *ferr, float *berr,
                             float *rpvgrw)
{
    g_sgbsvx_cblas_call.called += 1;
    g_sgbsvx_cblas_call.layout = layout;
    g_sgbsvx_cblas_call.fact = fact;
    g_sgbsvx_cblas_call.trans = trans;
    g_sgbsvx_cblas_call.n = n;
    g_sgbsvx_cblas_call.kl = kl;
    g_sgbsvx_cblas_call.ku = ku;
    g_sgbsvx_cblas_call.nrhs = nrhs;
    g_sgbsvx_cblas_call.ldab = ldab;
    g_sgbsvx_cblas_call.ldafb = ldafb;
    g_sgbsvx_cblas_call.ldb = ldb;
    g_sgbsvx_cblas_call.ldx = ldx;
    g_sgbsvx_cblas_call.ipiv = ipiv;
    g_sgbsvx_cblas_call.ab = ab;
    g_sgbsvx_cblas_call.afb = afb;
    g_sgbsvx_cblas_call.equed = equed;
    g_sgbsvx_cblas_call.r = r;
    g_sgbsvx_cblas_call.c = c;
    g_sgbsvx_cblas_call.b = b;
    g_sgbsvx_cblas_call.x = x;
    g_sgbsvx_cblas_call.rcond = rcond;
    g_sgbsvx_cblas_call.ferr = ferr;
    g_sgbsvx_cblas_call.berr = berr;
    g_sgbsvx_cblas_call.rpvgrw = rpvgrw;
    ipiv[0] = 7;
    *equed = 'Y';
    ab[0] = 91.0f;
    afb[0] = 92.0f;
    b[0] = 93.0f;
    x[0] = 94.0f;
    *rcond = 0.75f;
    ferr[0] = 1.5f;
    berr[0] = 0.5f;
    *rpvgrw = 0.33f;
    return 191;
}

static void stub_cgbsvx_fortran(char *fact, char *trans, int *n, int *kl,
                                int *ku, int *nrhs, fb_complex_float_t *ab,
                                int *ldab, fb_complex_float_t *afb,
                                int *ldafb, int *ipiv, char *equed, float *r,
                                float *c, fb_complex_float_t *b, int *ldb,
                                fb_complex_float_t *x, int *ldx, float *rcond,
                                float *ferr, float *berr,
                                fb_complex_float_t *work, float *rwork,
                                int *info)
{
    static const int ipiv_out[4] = { 1, 3, 4, 2 };
    static const fb_complex_float_t b_out_col[8] = {
        (fb_complex_float_t)0, (fb_complex_float_t)0, (fb_complex_float_t)0, (fb_complex_float_t)0,
        (fb_complex_float_t)0, (fb_complex_float_t)0, (fb_complex_float_t)0, (fb_complex_float_t)0
    };
    static const fb_complex_float_t x_out_col[8] = {
        (fb_complex_float_t)0, (fb_complex_float_t)0, (fb_complex_float_t)0, (fb_complex_float_t)0,
        (fb_complex_float_t)0, (fb_complex_float_t)0, (fb_complex_float_t)0, (fb_complex_float_t)0
    };
    fb_complex_float_t local_b_out[8];
    fb_complex_float_t local_x_out[8];
    int index = 0;

    local_b_out[0] = make_cf32(811.0f, 1.0f);
    local_b_out[1] = make_cf32(831.0f, 3.0f);
    local_b_out[2] = make_cf32(851.0f, 5.0f);
    local_b_out[3] = make_cf32(871.0f, 7.0f);
    local_b_out[4] = make_cf32(821.0f, 2.0f);
    local_b_out[5] = make_cf32(841.0f, 4.0f);
    local_b_out[6] = make_cf32(861.0f, 6.0f);
    local_b_out[7] = make_cf32(881.0f, 8.0f);
    local_x_out[0] = make_cf32(911.0f, 11.0f);
    local_x_out[1] = make_cf32(931.0f, 13.0f);
    local_x_out[2] = make_cf32(951.0f, 15.0f);
    local_x_out[3] = make_cf32(971.0f, 17.0f);
    local_x_out[4] = make_cf32(921.0f, 12.0f);
    local_x_out[5] = make_cf32(941.0f, 14.0f);
    local_x_out[6] = make_cf32(961.0f, 16.0f);
    local_x_out[7] = make_cf32(981.0f, 18.0f);

    g_cgbsvx_fortran_call.calls += 1;
    g_cgbsvx_fortran_call.fact = *fact;
    g_cgbsvx_fortran_call.trans = *trans;
    g_cgbsvx_fortran_call.equed_before = *equed;
    g_cgbsvx_fortran_call.n = *n;
    g_cgbsvx_fortran_call.kl = *kl;
    g_cgbsvx_fortran_call.ku = *ku;
    g_cgbsvx_fortran_call.nrhs = *nrhs;
    g_cgbsvx_fortran_call.ldab = *ldab;
    g_cgbsvx_fortran_call.ldafb = *ldafb;
    g_cgbsvx_fortran_call.ldb = *ldb;
    g_cgbsvx_fortran_call.ldx = *ldx;
    g_cgbsvx_fortran_call.work_seen = (work != NULL);
    g_cgbsvx_fortran_call.aux_seen = (rwork != NULL);
    memcpy(g_cgbsvx_fortran_call.ipiv_snapshot, ipiv,
           sizeof(g_cgbsvx_fortran_call.ipiv_snapshot));
    memcpy(g_cgbsvx_fortran_call.ab_snapshot, ab,
           sizeof(g_cgbsvx_fortran_call.ab_snapshot));
    memcpy(g_cgbsvx_fortran_call.afb_snapshot, afb,
           sizeof(g_cgbsvx_fortran_call.afb_snapshot));
    memcpy(g_cgbsvx_fortran_call.b_snapshot, b,
           sizeof(g_cgbsvx_fortran_call.b_snapshot));

    for (index = 0; index < 8; ++index) {
        b[index] = local_b_out[index];
        x[index] = local_x_out[index];
    }
    memcpy(ipiv, ipiv_out, sizeof(ipiv_out));
    *equed = 'B';
    r[0] = 3.1f;
    c[0] = 4.1f;
    *rcond = 0.35f;
    ferr[0] = 0.7f;
    ferr[1] = 0.8f;
    berr[0] = 0.07f;
    berr[1] = 0.08f;
    rwork[1] = 0.625f;
    *info = 0;
}

static int stub_cgbsvx_cblas(fb_layout_t layout, char fact, char trans, int n,
                             int kl, int ku, int nrhs, fb_complex_float_t *ab,
                             int ldab, fb_complex_float_t *afb, int ldafb,
                             int *ipiv, char *equed, float *r, float *c,
                             fb_complex_float_t *b, int ldb,
                             fb_complex_float_t *x, int ldx, float *rcond,
                             float *ferr, float *berr, float *rpvgrw)
{
    g_cgbsvx_cblas_call.called += 1;
    g_cgbsvx_cblas_call.layout = layout;
    g_cgbsvx_cblas_call.fact = fact;
    g_cgbsvx_cblas_call.trans = trans;
    g_cgbsvx_cblas_call.n = n;
    g_cgbsvx_cblas_call.kl = kl;
    g_cgbsvx_cblas_call.ku = ku;
    g_cgbsvx_cblas_call.nrhs = nrhs;
    g_cgbsvx_cblas_call.ldab = ldab;
    g_cgbsvx_cblas_call.ldafb = ldafb;
    g_cgbsvx_cblas_call.ldb = ldb;
    g_cgbsvx_cblas_call.ldx = ldx;
    g_cgbsvx_cblas_call.ipiv = ipiv;
    g_cgbsvx_cblas_call.ab = ab;
    g_cgbsvx_cblas_call.afb = afb;
    g_cgbsvx_cblas_call.equed = equed;
    g_cgbsvx_cblas_call.r = r;
    g_cgbsvx_cblas_call.c = c;
    g_cgbsvx_cblas_call.b = b;
    g_cgbsvx_cblas_call.x = x;
    g_cgbsvx_cblas_call.rcond = rcond;
    g_cgbsvx_cblas_call.ferr = ferr;
    g_cgbsvx_cblas_call.berr = berr;
    g_cgbsvx_cblas_call.rpvgrw = rpvgrw;
    ipiv[0] = 8;
    *equed = 'N';
    ab[0] = make_cf32(191.0f, 1.0f);
    afb[0] = make_cf32(192.0f, 2.0f);
    b[0] = make_cf32(193.0f, 3.0f);
    x[0] = make_cf32(194.0f, 4.0f);
    *rcond = 0.95f;
    ferr[0] = 1.7f;
    berr[0] = 0.9f;
    *rpvgrw = 0.55f;
    return 193;
}

static int check_sgbsvx_fortran_to_cblas_fact_e(void)
{
    fb_backend_vtable_t vtable;
    fb_sgbsvx_cblas_fn thunk = NULL;
    float ab[12] = {
        0.0f, 111.0f, 112.0f, 113.0f,
        211.0f, 212.0f, 213.0f, 214.0f,
        311.0f, 312.0f, 313.0f, 0.0f
    };
    float afb[16] = {
        0.0f, 0.0f, 412.0f, 413.0f,
        0.0f, 521.0f, 522.0f, 523.0f,
        631.0f, 632.0f, 633.0f, 634.0f,
        741.0f, 742.0f, 743.0f, 0.0f
    };
    float b[8] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };
    float x[8] = { 0.0f };
    int ipiv[4] = { 0, 0, 0, 0 };
    char equed = 'N';
    float r[4] = { 0.0f };
    float c[4] = { 0.0f };
    float rcond = 0.0f;
    float ferr[2] = { 0.0f, 0.0f };
    float berr[2] = { 0.0f, 0.0f };
    float rpvgrw = 0.0f;
    static const float expected_ab_in_col[16] = {
        0.0f, 0.0f, 211.0f, 311.0f,
        0.0f, 111.0f, 212.0f, 312.0f,
        0.0f, 112.0f, 213.0f, 313.0f,
        0.0f, 113.0f, 214.0f, 0.0f
    };
    static const float expected_ab_out_row[12] = {
        0.0f, 511.0f, 512.0f, 513.0f,
        611.0f, 612.0f, 613.0f, 614.0f,
        711.0f, 712.0f, 713.0f, 0.0f
    };
    static const float expected_afb_in_col[16] = { 0.0f };
    static const float expected_afb_out_row[16] = {
        0.0f, 0.0f, 612.0f, 613.0f,
        0.0f, 621.0f, 622.0f, 623.0f,
        631.0f, 632.0f, 633.0f, 634.0f,
        641.0f, 642.0f, 643.0f, 0.0f
    };
    static const float expected_x_out_row[8] = {
        910.0f, 920.0f,
        930.0f, 940.0f,
        950.0f, 960.0f,
        970.0f, 980.0f
    };
    static const int expected_ipiv[4] = { 4, 3, 2, 1 };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sgbsvx_fortran_call, 0, sizeof(g_sgbsvx_fortran_call));

    vtable.ext_ops[FB_OP_SGBSVX][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_sgbsvx_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SGBSVX);

    thunk = (fb_sgbsvx_cblas_fn)vtable.ext_ops[FB_OP_SGBSVX][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SGBSVX Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    if (thunk(FB_LAYOUT_ROW_MAJOR, 'E', 'T', 4, 1, 1, 2, ab, 4, afb, 4, ipiv,
              &equed, r, c, b, 2, x, 2, &rcond, ferr, berr, &rpvgrw) != 0) {
        fprintf(stderr, "[FAIL] SGBSVX Fortran->CBLAS thunk returned unexpected info\n");
        return 1;
    }

    if (g_sgbsvx_fortran_call.calls != 1 || g_sgbsvx_fortran_call.fact != 'E' ||
        g_sgbsvx_fortran_call.trans != 'T' ||
        g_sgbsvx_fortran_call.equed_before != 'N' ||
        g_sgbsvx_fortran_call.n != 4 || g_sgbsvx_fortran_call.kl != 1 ||
        g_sgbsvx_fortran_call.ku != 1 || g_sgbsvx_fortran_call.nrhs != 2 ||
        g_sgbsvx_fortran_call.ldab != 4 || g_sgbsvx_fortran_call.ldafb != 4 ||
        g_sgbsvx_fortran_call.ldb != 4 || g_sgbsvx_fortran_call.ldx != 4 ||
        !g_sgbsvx_fortran_call.work_seen || !g_sgbsvx_fortran_call.aux_seen ||
        !float_array_eq(g_sgbsvx_fortran_call.ab_snapshot, expected_ab_in_col, 16) ||
        !float_array_eq(g_sgbsvx_fortran_call.afb_snapshot, expected_afb_in_col, 16) ||
        !float_array_eq(ab, expected_ab_out_row, 12) ||
        !float_array_eq(afb, expected_afb_out_row, 16) ||
        !float_array_eq(x, expected_x_out_row, 8) ||
        !float_array_eq(b, (const float[]){ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f }, 8) ||
        !int_array_eq(ipiv, expected_ipiv, 4) || equed != 'B' || r[0] != 1.1f ||
        c[0] != 2.1f || rcond != 0.25f || ferr[0] != 0.5f || berr[0] != 0.05f ||
        rpvgrw != 0.875f) {
        fprintf(stderr, "[FAIL] SGBSVX Fortran->CBLAS thunk did not preserve the compact-to-expanded band contract\n");
        return 1;
    }

    printf("[PASS] SGBSVX Fortran->CBLAS thunk expands compact AB, copies back row-major outputs, and surfaces work[0] as rpvgrw\n");
    return 0;
}

static int check_sgbsvx_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_sgbsvx_fortran_slot_fn thunk = NULL;
    char fact = 'F';
    char trans = 'N';
    int n = 4;
    int kl = 1;
    int ku = 1;
    int nrhs = 2;
    float ab[16] = { 0.0f };
    int ldab = 4;
    float afb[16] = { 0.0f };
    int ldafb = 4;
    int ipiv[4] = { 0, 0, 0, 0 };
    char equed = 'N';
    float r[4] = { 0.0f };
    float c[4] = { 0.0f };
    float b[8] = { 0.0f };
    int ldb = 4;
    float x[8] = { 0.0f };
    int ldx = 4;
    float rcond = 0.0f;
    float ferr[2] = { 0.0f, 0.0f };
    float berr[2] = { 0.0f, 0.0f };
    float work[12] = { 0.0f };
    int iwork[4] = { 0, 0, 0, 0 };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sgbsvx_cblas_call, 0, sizeof(g_sgbsvx_cblas_call));

    vtable.ext_ops[FB_OP_SGBSVX][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_sgbsvx_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SGBSVX);

    thunk = (fb_sgbsvx_fortran_slot_fn)vtable.ext_ops[FB_OP_SGBSVX][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SGBSVX CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&fact, &trans, &n, &kl, &ku, &nrhs, ab, &ldab, afb, &ldafb, ipiv,
          &equed, r, c, b, &ldb, x, &ldx, &rcond, ferr, berr, work, iwork,
          &info);

    if (g_sgbsvx_cblas_call.called != 1 ||
        g_sgbsvx_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_sgbsvx_cblas_call.fact != 'F' || g_sgbsvx_cblas_call.trans != 'N' ||
        g_sgbsvx_cblas_call.n != 4 || g_sgbsvx_cblas_call.kl != 1 ||
        g_sgbsvx_cblas_call.ku != 1 || g_sgbsvx_cblas_call.nrhs != 2 ||
        g_sgbsvx_cblas_call.ldab != 4 || g_sgbsvx_cblas_call.ldafb != 4 ||
        g_sgbsvx_cblas_call.ldb != 4 || g_sgbsvx_cblas_call.ldx != 4 ||
        g_sgbsvx_cblas_call.ipiv != ipiv || g_sgbsvx_cblas_call.ab != ab ||
        g_sgbsvx_cblas_call.afb != afb || g_sgbsvx_cblas_call.equed != &equed ||
        g_sgbsvx_cblas_call.r != r || g_sgbsvx_cblas_call.c != c ||
        g_sgbsvx_cblas_call.b != b || g_sgbsvx_cblas_call.x != x ||
        g_sgbsvx_cblas_call.rpvgrw == NULL || info != 191 || ipiv[0] != 7 ||
        equed != 'Y' || ab[0] != 91.0f || afb[0] != 92.0f || b[0] != 93.0f ||
        x[0] != 94.0f || rcond != 0.75f || ferr[0] != 1.5f || berr[0] != 0.5f) {
        fprintf(stderr, "[FAIL] SGBSVX CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] SGBSVX CBLAS->Fortran thunk forwards the expanded column-major ABI and hides the extra C rpvgrw parameter\n");
    return 0;
}

static int check_cgbsvx_fortran_to_cblas_fact_f(void)
{
    fb_backend_vtable_t vtable;
    fb_cgbsvx_cblas_fn thunk = NULL;
    fb_complex_float_t ab[12] = {
        make_cf32(0.0f, 0.0f), make_cf32(111.0f, 1.0f), make_cf32(112.0f, 2.0f), make_cf32(113.0f, 3.0f),
        make_cf32(211.0f, -1.0f), make_cf32(212.0f, -2.0f), make_cf32(213.0f, -3.0f), make_cf32(214.0f, -4.0f),
        make_cf32(311.0f, 4.0f), make_cf32(312.0f, 5.0f), make_cf32(313.0f, 6.0f), make_cf32(0.0f, 0.0f)
    };
    fb_complex_float_t afb[16] = {
        make_cf32(0.0f, 0.0f), make_cf32(0.0f, 0.0f), make_cf32(412.0f, 12.0f), make_cf32(413.0f, 13.0f),
        make_cf32(0.0f, 0.0f), make_cf32(521.0f, -21.0f), make_cf32(522.0f, -22.0f), make_cf32(523.0f, -23.0f),
        make_cf32(631.0f, 31.0f), make_cf32(632.0f, 32.0f), make_cf32(633.0f, 33.0f), make_cf32(634.0f, 34.0f),
        make_cf32(741.0f, -41.0f), make_cf32(742.0f, -42.0f), make_cf32(743.0f, -43.0f), make_cf32(0.0f, 0.0f)
    };
    fb_complex_float_t b[8] = {
        make_cf32(1.0f, 1.0f), make_cf32(2.0f, 2.0f), make_cf32(3.0f, 3.0f), make_cf32(4.0f, 4.0f),
        make_cf32(5.0f, 5.0f), make_cf32(6.0f, 6.0f), make_cf32(7.0f, 7.0f), make_cf32(8.0f, 8.0f)
    };
    fb_complex_float_t x[8] = { 0 };
    fb_complex_float_t original_ab[12];
    fb_complex_float_t original_afb[16];
    int ipiv[4] = { 0, 0, 0, 0 };
    char equed = 'B';
    float r[4] = { 0.0f };
    float c[4] = { 0.0f };
    float rcond = 0.0f;
    float ferr[2] = { 0.0f, 0.0f };
    float berr[2] = { 0.0f, 0.0f };
    float rpvgrw = 0.0f;
    fb_complex_float_t expected_ab_in_col[16];
    fb_complex_float_t expected_afb_in_col[16];
    fb_complex_float_t expected_b_out_row[8];
    fb_complex_float_t expected_x_out_row[8];
    static const int expected_ipiv[4] = { 1, 3, 4, 2 };
    int index = 0;

    expected_ab_in_col[0] = make_cf32(0.0f, 0.0f);
    expected_ab_in_col[1] = make_cf32(0.0f, 0.0f);
    expected_ab_in_col[2] = make_cf32(211.0f, -1.0f);
    expected_ab_in_col[3] = make_cf32(311.0f, 4.0f);
    expected_ab_in_col[4] = make_cf32(0.0f, 0.0f);
    expected_ab_in_col[5] = make_cf32(111.0f, 1.0f);
    expected_ab_in_col[6] = make_cf32(212.0f, -2.0f);
    expected_ab_in_col[7] = make_cf32(312.0f, 5.0f);
    expected_ab_in_col[8] = make_cf32(0.0f, 0.0f);
    expected_ab_in_col[9] = make_cf32(112.0f, 2.0f);
    expected_ab_in_col[10] = make_cf32(213.0f, -3.0f);
    expected_ab_in_col[11] = make_cf32(313.0f, 6.0f);
    expected_ab_in_col[12] = make_cf32(0.0f, 0.0f);
    expected_ab_in_col[13] = make_cf32(113.0f, 3.0f);
    expected_ab_in_col[14] = make_cf32(214.0f, -4.0f);
    expected_ab_in_col[15] = make_cf32(0.0f, 0.0f);
    expected_afb_in_col[0] = make_cf32(0.0f, 0.0f);
    expected_afb_in_col[1] = make_cf32(0.0f, 0.0f);
    expected_afb_in_col[2] = make_cf32(631.0f, 31.0f);
    expected_afb_in_col[3] = make_cf32(741.0f, -41.0f);
    expected_afb_in_col[4] = make_cf32(0.0f, 0.0f);
    expected_afb_in_col[5] = make_cf32(521.0f, -21.0f);
    expected_afb_in_col[6] = make_cf32(632.0f, 32.0f);
    expected_afb_in_col[7] = make_cf32(742.0f, -42.0f);
    expected_afb_in_col[8] = make_cf32(412.0f, 12.0f);
    expected_afb_in_col[9] = make_cf32(522.0f, -22.0f);
    expected_afb_in_col[10] = make_cf32(633.0f, 33.0f);
    expected_afb_in_col[11] = make_cf32(743.0f, -43.0f);
    expected_afb_in_col[12] = make_cf32(413.0f, 13.0f);
    expected_afb_in_col[13] = make_cf32(523.0f, -23.0f);
    expected_afb_in_col[14] = make_cf32(634.0f, 34.0f);
    expected_afb_in_col[15] = make_cf32(0.0f, 0.0f);
    expected_b_out_row[0] = make_cf32(811.0f, 1.0f);
    expected_b_out_row[1] = make_cf32(821.0f, 2.0f);
    expected_b_out_row[2] = make_cf32(831.0f, 3.0f);
    expected_b_out_row[3] = make_cf32(841.0f, 4.0f);
    expected_b_out_row[4] = make_cf32(851.0f, 5.0f);
    expected_b_out_row[5] = make_cf32(861.0f, 6.0f);
    expected_b_out_row[6] = make_cf32(871.0f, 7.0f);
    expected_b_out_row[7] = make_cf32(881.0f, 8.0f);
    expected_x_out_row[0] = make_cf32(911.0f, 11.0f);
    expected_x_out_row[1] = make_cf32(921.0f, 12.0f);
    expected_x_out_row[2] = make_cf32(931.0f, 13.0f);
    expected_x_out_row[3] = make_cf32(941.0f, 14.0f);
    expected_x_out_row[4] = make_cf32(951.0f, 15.0f);
    expected_x_out_row[5] = make_cf32(961.0f, 16.0f);
    expected_x_out_row[6] = make_cf32(971.0f, 17.0f);
    expected_x_out_row[7] = make_cf32(981.0f, 18.0f);

    memcpy(original_ab, ab, sizeof(original_ab));
    memcpy(original_afb, afb, sizeof(original_afb));
    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cgbsvx_fortran_call, 0, sizeof(g_cgbsvx_fortran_call));

    vtable.ext_ops[FB_OP_CGBSVX][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_cgbsvx_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CGBSVX);

    thunk = (fb_cgbsvx_cblas_fn)vtable.ext_ops[FB_OP_CGBSVX][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CGBSVX Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    if (thunk(FB_LAYOUT_ROW_MAJOR, 'F', 'C', 4, 1, 1, 2, ab, 4, afb, 4, ipiv,
              &equed, r, c, b, 2, x, 2, &rcond, ferr, berr, &rpvgrw) != 0) {
        fprintf(stderr, "[FAIL] CGBSVX Fortran->CBLAS thunk returned unexpected info\n");
        return 1;
    }

    if (g_cgbsvx_fortran_call.calls != 1 || g_cgbsvx_fortran_call.fact != 'F' ||
        g_cgbsvx_fortran_call.trans != 'C' ||
        g_cgbsvx_fortran_call.equed_before != 'B' ||
        g_cgbsvx_fortran_call.n != 4 || g_cgbsvx_fortran_call.kl != 1 ||
        g_cgbsvx_fortran_call.ku != 1 || g_cgbsvx_fortran_call.nrhs != 2 ||
        g_cgbsvx_fortran_call.ldab != 4 || g_cgbsvx_fortran_call.ldafb != 4 ||
        g_cgbsvx_fortran_call.ldb != 4 || g_cgbsvx_fortran_call.ldx != 4 ||
        !g_cgbsvx_fortran_call.work_seen || !g_cgbsvx_fortran_call.aux_seen ||
        !cf32_array_eq(g_cgbsvx_fortran_call.ab_snapshot, expected_ab_in_col, 16) ||
        !cf32_array_eq(g_cgbsvx_fortran_call.afb_snapshot, expected_afb_in_col, 16) ||
        !cf32_array_eq(ab, original_ab, 12) || !cf32_array_eq(afb, original_afb, 16) ||
        !cf32_array_eq(b, expected_b_out_row, 8) || !cf32_array_eq(x, expected_x_out_row, 8) ||
        !int_array_eq(ipiv, expected_ipiv, 4) || equed != 'B' || r[0] != 3.1f ||
        c[0] != 4.1f || rcond != 0.35f || ferr[0] != 0.7f || berr[0] != 0.07f ||
        rpvgrw != 0.625f) {
        fprintf(stderr, "[FAIL] CGBSVX Fortran->CBLAS thunk did not preserve the factored-band contract\n");
        return 1;
    }

    printf("[PASS] CGBSVX Fortran->CBLAS thunk expands compact AB, translates factored AFB, copies row-major B/X selectively, and surfaces rwork[1] as rpvgrw\n");
    return 0;
}

static int check_cgbsvx_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_cgbsvx_fortran_slot_fn thunk = NULL;
    char fact = 'N';
    char trans = 'T';
    int n = 4;
    int kl = 1;
    int ku = 1;
    int nrhs = 2;
    fb_complex_float_t ab[16] = { 0 };
    int ldab = 4;
    fb_complex_float_t afb[16] = { 0 };
    int ldafb = 4;
    int ipiv[4] = { 0, 0, 0, 0 };
    char equed = 'N';
    float r[4] = { 0.0f };
    float c[4] = { 0.0f };
    fb_complex_float_t b[8] = { 0 };
    int ldb = 4;
    fb_complex_float_t x[8] = { 0 };
    int ldx = 4;
    float rcond = 0.0f;
    float ferr[2] = { 0.0f, 0.0f };
    float berr[2] = { 0.0f, 0.0f };
    fb_complex_float_t work[8] = { 0 };
    float rwork[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cgbsvx_cblas_call, 0, sizeof(g_cgbsvx_cblas_call));

    vtable.ext_ops[FB_OP_CGBSVX][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_cgbsvx_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CGBSVX);

    thunk = (fb_cgbsvx_fortran_slot_fn)vtable.ext_ops[FB_OP_CGBSVX][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CGBSVX CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&fact, &trans, &n, &kl, &ku, &nrhs, ab, &ldab, afb, &ldafb, ipiv,
          &equed, r, c, b, &ldb, x, &ldx, &rcond, ferr, berr, work, rwork,
          &info);

    if (g_cgbsvx_cblas_call.called != 1 ||
        g_cgbsvx_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_cgbsvx_cblas_call.fact != 'N' || g_cgbsvx_cblas_call.trans != 'T' ||
        g_cgbsvx_cblas_call.n != 4 || g_cgbsvx_cblas_call.kl != 1 ||
        g_cgbsvx_cblas_call.ku != 1 || g_cgbsvx_cblas_call.nrhs != 2 ||
        g_cgbsvx_cblas_call.ldab != 4 || g_cgbsvx_cblas_call.ldafb != 4 ||
        g_cgbsvx_cblas_call.ldb != 4 || g_cgbsvx_cblas_call.ldx != 4 ||
        g_cgbsvx_cblas_call.ipiv != ipiv || g_cgbsvx_cblas_call.ab != ab ||
        g_cgbsvx_cblas_call.afb != afb || g_cgbsvx_cblas_call.equed != &equed ||
        g_cgbsvx_cblas_call.r != r || g_cgbsvx_cblas_call.c != c ||
        g_cgbsvx_cblas_call.b != b || g_cgbsvx_cblas_call.x != x ||
        g_cgbsvx_cblas_call.rpvgrw == NULL || info != 193 || ipiv[0] != 8 ||
        equed != 'N' || !cf32_eq(ab[0], make_cf32(191.0f, 1.0f)) ||
        !cf32_eq(afb[0], make_cf32(192.0f, 2.0f)) ||
        !cf32_eq(b[0], make_cf32(193.0f, 3.0f)) ||
        !cf32_eq(x[0], make_cf32(194.0f, 4.0f)) ||
        rcond != 0.95f || ferr[0] != 1.7f || berr[0] != 0.9f) {
        fprintf(stderr, "[FAIL] CGBSVX CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] CGBSVX CBLAS->Fortran thunk forwards the expanded column-major ABI and hides the extra C rpvgrw parameter\n");
    return 0;
}

int main(void)
{
    if (check_sgbsvx_fortran_to_cblas_fact_e() != 0) {
        return 1;
    }
    if (check_sgbsvx_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_cgbsvx_fortran_to_cblas_fact_f() != 0) {
        return 1;
    }
    if (check_cgbsvx_cblas_to_fortran() != 0) {
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}