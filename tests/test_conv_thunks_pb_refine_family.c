#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef int (*fb_spbrfs_fn)(fb_layout_t layout, fb_uplo_t uplo, int n, int kd,
                            int nrhs, const float *ab, int ldab,
                            const float *afb, int ldafb, const float *b,
                            int ldb, float *x, int ldx, float *ferr,
                            float *berr);
typedef int (*fb_dpbrfs_fn)(fb_layout_t layout, fb_uplo_t uplo, int n, int kd,
                            int nrhs, const double *ab, int ldab,
                            const double *afb, int ldafb, const double *b,
                            int ldb, double *x, int ldx, double *ferr,
                            double *berr);
typedef int (*fb_cpbrfs_fn)(fb_layout_t layout, fb_uplo_t uplo, int n, int kd,
                            int nrhs, const fb_complex_float_t *ab, int ldab,
                            const fb_complex_float_t *afb, int ldafb,
                            const fb_complex_float_t *b, int ldb,
                            fb_complex_float_t *x, int ldx, float *ferr,
                            float *berr);
typedef int (*fb_zpbrfs_fn)(fb_layout_t layout, fb_uplo_t uplo, int n, int kd,
                            int nrhs, const fb_complex_double_t *ab, int ldab,
                            const fb_complex_double_t *afb, int ldafb,
                            const fb_complex_double_t *b, int ldb,
                            fb_complex_double_t *x, int ldx, double *ferr,
                            double *berr);

typedef void (*fb_spbrfs_fortran_slot_fn)(char *uplo, int *n, int *kd, int *nrhs,
                                          float *ab, int *ldab, float *afb,
                                          int *ldafb, float *b, int *ldb,
                                          float *x, int *ldx, float *ferr,
                                          float *berr, float *work, int *iwork,
                                          int *info);
typedef void (*fb_dpbrfs_fortran_slot_fn)(char *uplo, int *n, int *kd, int *nrhs,
                                          double *ab, int *ldab, double *afb,
                                          int *ldafb, double *b, int *ldb,
                                          double *x, int *ldx, double *ferr,
                                          double *berr, double *work,
                                          int *iwork, int *info);
typedef void (*fb_cpbrfs_fortran_slot_fn)(char *uplo, int *n, int *kd, int *nrhs,
                                          fb_complex_float_t *ab, int *ldab,
                                          fb_complex_float_t *afb, int *ldafb,
                                          fb_complex_float_t *b, int *ldb,
                                          fb_complex_float_t *x, int *ldx,
                                          float *ferr, float *berr,
                                          fb_complex_float_t *work,
                                          float *rwork, int *info);
typedef void (*fb_zpbrfs_fortran_slot_fn)(char *uplo, int *n, int *kd, int *nrhs,
                                          fb_complex_double_t *ab, int *ldab,
                                          fb_complex_double_t *afb, int *ldafb,
                                          fb_complex_double_t *b, int *ldb,
                                          fb_complex_double_t *x, int *ldx,
                                          double *ferr, double *berr,
                                          fb_complex_double_t *work,
                                          double *rwork, int *info);

static struct {
    int calls;
    char uplo;
    int n;
    int kd;
    int nrhs;
    int ldab;
    int ldafb;
    int ldb;
    int ldx;
    int work_seen;
    int aux_seen;
    float ab_snapshot[8];
    float afb_snapshot[8];
    float b_snapshot[8];
    float x_snapshot[8];
} g_spbrfs_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    fb_uplo_t uplo;
    int n;
    int kd;
    int nrhs;
    int ldab;
    int ldafb;
    int ldb;
    int ldx;
    const float *ab;
    const float *afb;
    const float *b;
    float *x;
    float *ferr;
    float *berr;
} g_spbrfs_cblas_call;

static struct {
    int calls;
    char uplo;
    int n;
    int kd;
    int nrhs;
    int ldab;
    int ldafb;
    int ldb;
    int ldx;
    int work_seen;
    int aux_seen;
    double ab_snapshot[8];
    double afb_snapshot[8];
    double b_snapshot[8];
    double x_snapshot[8];
} g_dpbrfs_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    fb_uplo_t uplo;
    int n;
    int kd;
    int nrhs;
    int ldab;
    int ldafb;
    int ldb;
    int ldx;
    const double *ab;
    const double *afb;
    const double *b;
    double *x;
    double *ferr;
    double *berr;
} g_dpbrfs_cblas_call;

static struct {
    int calls;
    char uplo;
    int n;
    int kd;
    int nrhs;
    int ldab;
    int ldafb;
    int ldb;
    int ldx;
    int work_seen;
    int aux_seen;
    float ab_real_snapshot[8];
    float afb_real_snapshot[8];
    float b_real_snapshot[8];
    float x_real_snapshot[8];
} g_cpbrfs_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    fb_uplo_t uplo;
    int n;
    int kd;
    int nrhs;
    int ldab;
    int ldafb;
    int ldb;
    int ldx;
    const fb_complex_float_t *ab;
    const fb_complex_float_t *afb;
    const fb_complex_float_t *b;
    fb_complex_float_t *x;
    float *ferr;
    float *berr;
} g_cpbrfs_cblas_call;

static struct {
    int calls;
    char uplo;
    int n;
    int kd;
    int nrhs;
    int ldab;
    int ldafb;
    int ldb;
    int ldx;
    int work_seen;
    int aux_seen;
    double ab_real_snapshot[8];
    double afb_real_snapshot[8];
    double b_real_snapshot[8];
    double x_real_snapshot[8];
} g_zpbrfs_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    fb_uplo_t uplo;
    int n;
    int kd;
    int nrhs;
    int ldab;
    int ldafb;
    int ldb;
    int ldx;
    const fb_complex_double_t *ab;
    const fb_complex_double_t *afb;
    const fb_complex_double_t *b;
    fb_complex_double_t *x;
    double *ferr;
    double *berr;
} g_zpbrfs_cblas_call;

static fb_complex_float_t make_cfloat(float real_value)
{
    fb_complex_float_t value = (fb_complex_float_t)0;
    memcpy(&value, &real_value, sizeof(real_value));
    return value;
}

static float cfloat_real(fb_complex_float_t value)
{
    float real_value = 0.0f;
    memcpy(&real_value, &value, sizeof(real_value));
    return real_value;
}

static fb_complex_double_t make_cdouble(double real_value)
{
    fb_complex_double_t value = (fb_complex_double_t)0;
    memcpy(&value, &real_value, sizeof(real_value));
    return value;
}

static double cdouble_real(fb_complex_double_t value)
{
    double real_value = 0.0;
    memcpy(&real_value, &value, sizeof(real_value));
    return real_value;
}

static void stub_spbrfs_fortran(char *uplo, int *n, int *kd, int *nrhs,
                                float *ab, int *ldab, float *afb, int *ldafb,
                                float *b, int *ldb, float *x, int *ldx,
                                float *ferr, float *berr, float *work,
                                int *iwork, int *info)
{
    static const float solution_x_col[8] = { 101.0f, 103.0f, 105.0f, 107.0f,
                                             102.0f, 104.0f, 106.0f, 108.0f };
    int index = 0;

    g_spbrfs_fortran_call.calls += 1;
    g_spbrfs_fortran_call.uplo = *uplo;
    g_spbrfs_fortran_call.n = *n;
    g_spbrfs_fortran_call.kd = *kd;
    g_spbrfs_fortran_call.nrhs = *nrhs;
    g_spbrfs_fortran_call.ldab = *ldab;
    g_spbrfs_fortran_call.ldafb = *ldafb;
    g_spbrfs_fortran_call.ldb = *ldb;
    g_spbrfs_fortran_call.ldx = *ldx;
    g_spbrfs_fortran_call.work_seen = (work != NULL);
    g_spbrfs_fortran_call.aux_seen = (iwork != NULL);
    for (index = 0; index < (*ldab * *n); ++index) {
        g_spbrfs_fortran_call.ab_snapshot[index] = ab[index];
        g_spbrfs_fortran_call.afb_snapshot[index] = afb[index];
    }
    for (index = 0; index < (*ldb * *nrhs); ++index) {
        g_spbrfs_fortran_call.b_snapshot[index] = b[index];
        g_spbrfs_fortran_call.x_snapshot[index] = x[index];
    }
    memcpy(x, solution_x_col, sizeof(solution_x_col));
    ferr[0] = 0.1f;
    ferr[1] = 0.2f;
    berr[0] = 0.01f;
    berr[1] = 0.02f;
    *info = 0;
}

static int stub_spbrfs_cblas(fb_layout_t layout, fb_uplo_t uplo, int n, int kd,
                             int nrhs, const float *ab, int ldab,
                             const float *afb, int ldafb, const float *b,
                             int ldb, float *x, int ldx, float *ferr,
                             float *berr)
{
    g_spbrfs_cblas_call.called += 1;
    g_spbrfs_cblas_call.layout = layout;
    g_spbrfs_cblas_call.uplo = uplo;
    g_spbrfs_cblas_call.n = n;
    g_spbrfs_cblas_call.kd = kd;
    g_spbrfs_cblas_call.nrhs = nrhs;
    g_spbrfs_cblas_call.ldab = ldab;
    g_spbrfs_cblas_call.ldafb = ldafb;
    g_spbrfs_cblas_call.ldb = ldb;
    g_spbrfs_cblas_call.ldx = ldx;
    g_spbrfs_cblas_call.ab = ab;
    g_spbrfs_cblas_call.afb = afb;
    g_spbrfs_cblas_call.b = b;
    g_spbrfs_cblas_call.x = x;
    g_spbrfs_cblas_call.ferr = ferr;
    g_spbrfs_cblas_call.berr = berr;
    x[0] = 299.0f;
    ferr[0] = 1.5f;
    berr[0] = 0.5f;
    return 171;
}

static void stub_dpbrfs_fortran(char *uplo, int *n, int *kd, int *nrhs,
                                double *ab, int *ldab, double *afb,
                                int *ldafb, double *b, int *ldb,
                                double *x, int *ldx, double *ferr,
                                double *berr, double *work, int *iwork,
                                int *info)
{
    static const double solution_x_col[8] = { 201.0, 203.0, 205.0, 207.0,
                                              202.0, 204.0, 206.0, 208.0 };
    int index = 0;

    g_dpbrfs_fortran_call.calls += 1;
    g_dpbrfs_fortran_call.uplo = *uplo;
    g_dpbrfs_fortran_call.n = *n;
    g_dpbrfs_fortran_call.kd = *kd;
    g_dpbrfs_fortran_call.nrhs = *nrhs;
    g_dpbrfs_fortran_call.ldab = *ldab;
    g_dpbrfs_fortran_call.ldafb = *ldafb;
    g_dpbrfs_fortran_call.ldb = *ldb;
    g_dpbrfs_fortran_call.ldx = *ldx;
    g_dpbrfs_fortran_call.work_seen = (work != NULL);
    g_dpbrfs_fortran_call.aux_seen = (iwork != NULL);
    for (index = 0; index < (*ldab * *n); ++index) {
        g_dpbrfs_fortran_call.ab_snapshot[index] = ab[index];
        g_dpbrfs_fortran_call.afb_snapshot[index] = afb[index];
    }
    for (index = 0; index < (*ldb * *nrhs); ++index) {
        g_dpbrfs_fortran_call.b_snapshot[index] = b[index];
        g_dpbrfs_fortran_call.x_snapshot[index] = x[index];
    }
    memcpy(x, solution_x_col, sizeof(solution_x_col));
    ferr[0] = 0.15;
    ferr[1] = 0.25;
    berr[0] = 0.015;
    berr[1] = 0.025;
    *info = 0;
}

static int stub_dpbrfs_cblas(fb_layout_t layout, fb_uplo_t uplo, int n, int kd,
                             int nrhs, const double *ab, int ldab,
                             const double *afb, int ldafb, const double *b,
                             int ldb, double *x, int ldx, double *ferr,
                             double *berr)
{
    g_dpbrfs_cblas_call.called += 1;
    g_dpbrfs_cblas_call.layout = layout;
    g_dpbrfs_cblas_call.uplo = uplo;
    g_dpbrfs_cblas_call.n = n;
    g_dpbrfs_cblas_call.kd = kd;
    g_dpbrfs_cblas_call.nrhs = nrhs;
    g_dpbrfs_cblas_call.ldab = ldab;
    g_dpbrfs_cblas_call.ldafb = ldafb;
    g_dpbrfs_cblas_call.ldb = ldb;
    g_dpbrfs_cblas_call.ldx = ldx;
    g_dpbrfs_cblas_call.ab = ab;
    g_dpbrfs_cblas_call.afb = afb;
    g_dpbrfs_cblas_call.b = b;
    g_dpbrfs_cblas_call.x = x;
    g_dpbrfs_cblas_call.ferr = ferr;
    g_dpbrfs_cblas_call.berr = berr;
    x[0] = 499.0;
    ferr[0] = 1.6;
    berr[0] = 0.6;
    return 172;
}

static void stub_cpbrfs_fortran(char *uplo, int *n, int *kd, int *nrhs,
                                fb_complex_float_t *ab, int *ldab,
                                fb_complex_float_t *afb, int *ldafb,
                                fb_complex_float_t *b, int *ldb,
                                fb_complex_float_t *x, int *ldx, float *ferr,
                                float *berr, fb_complex_float_t *work,
                                float *rwork, int *info)
{
    static const float solution_x_col_real[8] = { 301.0f, 303.0f, 305.0f, 307.0f,
                                                  302.0f, 304.0f, 306.0f, 308.0f };
    int index = 0;

    g_cpbrfs_fortran_call.calls += 1;
    g_cpbrfs_fortran_call.uplo = *uplo;
    g_cpbrfs_fortran_call.n = *n;
    g_cpbrfs_fortran_call.kd = *kd;
    g_cpbrfs_fortran_call.nrhs = *nrhs;
    g_cpbrfs_fortran_call.ldab = *ldab;
    g_cpbrfs_fortran_call.ldafb = *ldafb;
    g_cpbrfs_fortran_call.ldb = *ldb;
    g_cpbrfs_fortran_call.ldx = *ldx;
    g_cpbrfs_fortran_call.work_seen = (work != NULL);
    g_cpbrfs_fortran_call.aux_seen = (rwork != NULL);
    for (index = 0; index < (*ldab * *n); ++index) {
        g_cpbrfs_fortran_call.ab_real_snapshot[index] = cfloat_real(ab[index]);
        g_cpbrfs_fortran_call.afb_real_snapshot[index] = cfloat_real(afb[index]);
    }
    for (index = 0; index < (*ldb * *nrhs); ++index) {
        g_cpbrfs_fortran_call.b_real_snapshot[index] = cfloat_real(b[index]);
        g_cpbrfs_fortran_call.x_real_snapshot[index] = cfloat_real(x[index]);
    }
    for (index = 0; index < 8; ++index) {
        x[index] = make_cfloat(solution_x_col_real[index]);
    }
    ferr[0] = 0.3f;
    ferr[1] = 0.4f;
    berr[0] = 0.03f;
    berr[1] = 0.04f;
    *info = 0;
}

static int stub_cpbrfs_cblas(fb_layout_t layout, fb_uplo_t uplo, int n, int kd,
                             int nrhs, const fb_complex_float_t *ab, int ldab,
                             const fb_complex_float_t *afb, int ldafb,
                             const fb_complex_float_t *b, int ldb,
                             fb_complex_float_t *x, int ldx, float *ferr,
                             float *berr)
{
    g_cpbrfs_cblas_call.called += 1;
    g_cpbrfs_cblas_call.layout = layout;
    g_cpbrfs_cblas_call.uplo = uplo;
    g_cpbrfs_cblas_call.n = n;
    g_cpbrfs_cblas_call.kd = kd;
    g_cpbrfs_cblas_call.nrhs = nrhs;
    g_cpbrfs_cblas_call.ldab = ldab;
    g_cpbrfs_cblas_call.ldafb = ldafb;
    g_cpbrfs_cblas_call.ldb = ldb;
    g_cpbrfs_cblas_call.ldx = ldx;
    g_cpbrfs_cblas_call.ab = ab;
    g_cpbrfs_cblas_call.afb = afb;
    g_cpbrfs_cblas_call.b = b;
    g_cpbrfs_cblas_call.x = x;
    g_cpbrfs_cblas_call.ferr = ferr;
    g_cpbrfs_cblas_call.berr = berr;
    x[0] = make_cfloat(399.0f);
    ferr[0] = 2.5f;
    berr[0] = 0.75f;
    return 173;
}

static void stub_zpbrfs_fortran(char *uplo, int *n, int *kd, int *nrhs,
                                fb_complex_double_t *ab, int *ldab,
                                fb_complex_double_t *afb, int *ldafb,
                                fb_complex_double_t *b, int *ldb,
                                fb_complex_double_t *x, int *ldx,
                                double *ferr, double *berr,
                                fb_complex_double_t *work, double *rwork,
                                int *info)
{
    static const double solution_x_col_real[8] = { 401.0, 403.0, 405.0, 407.0,
                                                   402.0, 404.0, 406.0, 408.0 };
    int index = 0;

    g_zpbrfs_fortran_call.calls += 1;
    g_zpbrfs_fortran_call.uplo = *uplo;
    g_zpbrfs_fortran_call.n = *n;
    g_zpbrfs_fortran_call.kd = *kd;
    g_zpbrfs_fortran_call.nrhs = *nrhs;
    g_zpbrfs_fortran_call.ldab = *ldab;
    g_zpbrfs_fortran_call.ldafb = *ldafb;
    g_zpbrfs_fortran_call.ldb = *ldb;
    g_zpbrfs_fortran_call.ldx = *ldx;
    g_zpbrfs_fortran_call.work_seen = (work != NULL);
    g_zpbrfs_fortran_call.aux_seen = (rwork != NULL);
    for (index = 0; index < (*ldab * *n); ++index) {
        g_zpbrfs_fortran_call.ab_real_snapshot[index] = cdouble_real(ab[index]);
        g_zpbrfs_fortran_call.afb_real_snapshot[index] = cdouble_real(afb[index]);
    }
    for (index = 0; index < (*ldb * *nrhs); ++index) {
        g_zpbrfs_fortran_call.b_real_snapshot[index] = cdouble_real(b[index]);
        g_zpbrfs_fortran_call.x_real_snapshot[index] = cdouble_real(x[index]);
    }
    for (index = 0; index < 8; ++index) {
        x[index] = make_cdouble(solution_x_col_real[index]);
    }
    ferr[0] = 0.35;
    ferr[1] = 0.45;
    berr[0] = 0.035;
    berr[1] = 0.045;
    *info = 0;
}

static int stub_zpbrfs_cblas(fb_layout_t layout, fb_uplo_t uplo, int n, int kd,
                             int nrhs, const fb_complex_double_t *ab, int ldab,
                             const fb_complex_double_t *afb, int ldafb,
                             const fb_complex_double_t *b, int ldb,
                             fb_complex_double_t *x, int ldx, double *ferr,
                             double *berr)
{
    g_zpbrfs_cblas_call.called += 1;
    g_zpbrfs_cblas_call.layout = layout;
    g_zpbrfs_cblas_call.uplo = uplo;
    g_zpbrfs_cblas_call.n = n;
    g_zpbrfs_cblas_call.kd = kd;
    g_zpbrfs_cblas_call.nrhs = nrhs;
    g_zpbrfs_cblas_call.ldab = ldab;
    g_zpbrfs_cblas_call.ldafb = ldafb;
    g_zpbrfs_cblas_call.ldb = ldb;
    g_zpbrfs_cblas_call.ldx = ldx;
    g_zpbrfs_cblas_call.ab = ab;
    g_zpbrfs_cblas_call.afb = afb;
    g_zpbrfs_cblas_call.b = b;
    g_zpbrfs_cblas_call.x = x;
    g_zpbrfs_cblas_call.ferr = ferr;
    g_zpbrfs_cblas_call.berr = berr;
    x[0] = make_cdouble(599.0);
    ferr[0] = 2.6;
    berr[0] = 0.85;
    return 174;
}

static int check_spbrfs_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_spbrfs_fn thunk = NULL;
    float ab[8] = { 10.0f, 11.0f, 12.0f, 13.0f, 20.0f, 21.0f, 22.0f, 23.0f };
    float afb[8] = { 30.0f, 31.0f, 32.0f, 33.0f, 40.0f, 41.0f, 42.0f, 43.0f };
    float b[8] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };
    float x[8] = { 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f };
    float ferr[2] = { 0.0f, 0.0f };
    float berr[2] = { 0.0f, 0.0f };
    float expected_ab_in[8] = { 0.0f, 20.0f, 11.0f, 21.0f, 12.0f, 22.0f, 13.0f, 23.0f };
    float expected_afb_in[8] = { 0.0f, 40.0f, 31.0f, 41.0f, 32.0f, 42.0f, 33.0f, 43.0f };
    float expected_b_in[8] = { 1.0f, 3.0f, 5.0f, 7.0f, 2.0f, 4.0f, 6.0f, 8.0f };
    float expected_x_in[8] = { 9.0f, 11.0f, 13.0f, 15.0f, 10.0f, 12.0f, 14.0f, 16.0f };
    float expected_x_out[8] = { 101.0f, 102.0f, 103.0f, 104.0f, 105.0f, 106.0f, 107.0f, 108.0f };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_spbrfs_fortran_call, 0, sizeof(g_spbrfs_fortran_call));

    vtable.ext_ops[FB_OP_SPBRFS][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_spbrfs_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SPBRFS);

    thunk = (fb_spbrfs_fn)vtable.ext_ops[FB_OP_SPBRFS][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SPBRFS Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, FB_UPPER, 4, 1, 2,
                 ab, 4, afb, 4, b, 2, x, 2, ferr, berr);
    if (info != 0 || g_spbrfs_fortran_call.calls != 1 ||
        g_spbrfs_fortran_call.uplo != 'U' || g_spbrfs_fortran_call.n != 4 ||
        g_spbrfs_fortran_call.kd != 1 || g_spbrfs_fortran_call.nrhs != 2 ||
        g_spbrfs_fortran_call.ldab != 2 || g_spbrfs_fortran_call.ldafb != 2 ||
        g_spbrfs_fortran_call.ldb != 4 || g_spbrfs_fortran_call.ldx != 4 ||
        !g_spbrfs_fortran_call.work_seen || !g_spbrfs_fortran_call.aux_seen ||
        memcmp(g_spbrfs_fortran_call.ab_snapshot, expected_ab_in, sizeof(expected_ab_in)) != 0 ||
        memcmp(g_spbrfs_fortran_call.afb_snapshot, expected_afb_in, sizeof(expected_afb_in)) != 0 ||
        memcmp(g_spbrfs_fortran_call.b_snapshot, expected_b_in, sizeof(expected_b_in)) != 0 ||
        memcmp(g_spbrfs_fortran_call.x_snapshot, expected_x_in, sizeof(expected_x_in)) != 0 ||
        memcmp(x, expected_x_out, sizeof(expected_x_out)) != 0 ||
        ferr[0] != 0.1f || ferr[1] != 0.2f || berr[0] != 0.01f || berr[1] != 0.02f) {
        fprintf(stderr, "[FAIL] SPBRFS Fortran->CBLAS thunk did not preserve row-major band refinement semantics\n");
        return 1;
    }

    printf("[PASS] SPBRFS Fortran->CBLAS thunk transposes banded inputs, allocates fixed real scratch, and copies back only X\n");
    return 0;
}

static int check_spbrfs_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_spbrfs_fortran_slot_fn thunk = NULL;
    float ab[6] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
    float afb[6] = { 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f };
    float b[6] = { 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f };
    float x[6] = { 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f };
    float ferr[2] = { 0.0f, 0.0f };
    float berr[2] = { 0.0f, 0.0f };
    float work[9] = { 0.0f };
    int iwork[3] = { 0, 0, 0 };
    char uplo = 'L';
    int n = 3;
    int kd = 1;
    int nrhs = 2;
    int ldab = 2;
    int ldafb = 2;
    int ldb = 3;
    int ldx = 3;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_spbrfs_cblas_call, 0, sizeof(g_spbrfs_cblas_call));

    vtable.ext_ops[FB_OP_SPBRFS][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_spbrfs_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SPBRFS);

    thunk = (fb_spbrfs_fortran_slot_fn)vtable.ext_ops[FB_OP_SPBRFS][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SPBRFS CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&uplo, &n, &kd, &nrhs, ab, &ldab, afb, &ldafb, b, &ldb, x, &ldx,
          ferr, berr, work, iwork, &info);
    if (info != 171 || g_spbrfs_cblas_call.called != 1 ||
        g_spbrfs_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_spbrfs_cblas_call.uplo != FB_LOWER || g_spbrfs_cblas_call.n != 3 ||
        g_spbrfs_cblas_call.kd != 1 || g_spbrfs_cblas_call.nrhs != 2 ||
        g_spbrfs_cblas_call.ldab != 2 || g_spbrfs_cblas_call.ldafb != 2 ||
        g_spbrfs_cblas_call.ldb != 3 || g_spbrfs_cblas_call.ldx != 3 ||
        g_spbrfs_cblas_call.ab != ab || g_spbrfs_cblas_call.afb != afb ||
        g_spbrfs_cblas_call.b != b || g_spbrfs_cblas_call.x != x ||
        g_spbrfs_cblas_call.ferr != ferr || g_spbrfs_cblas_call.berr != berr ||
        x[0] != 299.0f || ferr[0] != 1.5f || berr[0] != 0.5f) {
        fprintf(stderr, "[FAIL] SPBRFS CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] SPBRFS CBLAS->Fortran thunk maps the all-pointer ABI into the generic band-refinement entry\n");
    return 0;
}

static int check_dpbrfs_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_dpbrfs_fn thunk = NULL;
    double ab[8] = { 10.0, 11.0, 12.0, 13.0, 20.0, 21.0, 22.0, 23.0 };
    double afb[8] = { 30.0, 31.0, 32.0, 33.0, 40.0, 41.0, 42.0, 43.0 };
    double b[8] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    double x[8] = { 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0 };
    double ferr[2] = { 0.0, 0.0 };
    double berr[2] = { 0.0, 0.0 };
    double expected_ab_in[8] = { 0.0, 20.0, 11.0, 21.0, 12.0, 22.0, 13.0, 23.0 };
    double expected_afb_in[8] = { 0.0, 40.0, 31.0, 41.0, 32.0, 42.0, 33.0, 43.0 };
    double expected_b_in[8] = { 1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0 };
    double expected_x_in[8] = { 9.0, 11.0, 13.0, 15.0, 10.0, 12.0, 14.0, 16.0 };
    double expected_x_out[8] = { 201.0, 202.0, 203.0, 204.0, 205.0, 206.0, 207.0, 208.0 };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dpbrfs_fortran_call, 0, sizeof(g_dpbrfs_fortran_call));

    vtable.ext_ops[FB_OP_DPBRFS][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_dpbrfs_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_DPBRFS);

    thunk = (fb_dpbrfs_fn)vtable.ext_ops[FB_OP_DPBRFS][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DPBRFS Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, FB_UPPER, 4, 1, 2,
                 ab, 4, afb, 4, b, 2, x, 2, ferr, berr);
    if (info != 0 || g_dpbrfs_fortran_call.calls != 1 ||
        g_dpbrfs_fortran_call.uplo != 'U' || g_dpbrfs_fortran_call.n != 4 ||
        g_dpbrfs_fortran_call.kd != 1 || g_dpbrfs_fortran_call.nrhs != 2 ||
        g_dpbrfs_fortran_call.ldab != 2 || g_dpbrfs_fortran_call.ldafb != 2 ||
        g_dpbrfs_fortran_call.ldb != 4 || g_dpbrfs_fortran_call.ldx != 4 ||
        !g_dpbrfs_fortran_call.work_seen || !g_dpbrfs_fortran_call.aux_seen ||
        memcmp(g_dpbrfs_fortran_call.ab_snapshot, expected_ab_in, sizeof(expected_ab_in)) != 0 ||
        memcmp(g_dpbrfs_fortran_call.afb_snapshot, expected_afb_in, sizeof(expected_afb_in)) != 0 ||
        memcmp(g_dpbrfs_fortran_call.b_snapshot, expected_b_in, sizeof(expected_b_in)) != 0 ||
        memcmp(g_dpbrfs_fortran_call.x_snapshot, expected_x_in, sizeof(expected_x_in)) != 0 ||
        memcmp(x, expected_x_out, sizeof(expected_x_out)) != 0 ||
        ferr[0] != 0.15 || ferr[1] != 0.25 || berr[0] != 0.015 || berr[1] != 0.025) {
        fprintf(stderr, "[FAIL] DPBRFS Fortran->CBLAS thunk did not preserve row-major band refinement semantics\n");
        return 1;
    }

    printf("[PASS] DPBRFS Fortran->CBLAS thunk transposes banded inputs, allocates fixed real scratch, and copies back only X\n");
    return 0;
}

static int check_dpbrfs_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_dpbrfs_fortran_slot_fn thunk = NULL;
    double ab[6] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    double afb[6] = { 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 };
    double b[6] = { 13.0, 14.0, 15.0, 16.0, 17.0, 18.0 };
    double x[6] = { 19.0, 20.0, 21.0, 22.0, 23.0, 24.0 };
    double ferr[2] = { 0.0, 0.0 };
    double berr[2] = { 0.0, 0.0 };
    double work[9] = { 0.0 };
    int iwork[3] = { 0, 0, 0 };
    char uplo = 'L';
    int n = 3;
    int kd = 1;
    int nrhs = 2;
    int ldab = 2;
    int ldafb = 2;
    int ldb = 3;
    int ldx = 3;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dpbrfs_cblas_call, 0, sizeof(g_dpbrfs_cblas_call));

    vtable.ext_ops[FB_OP_DPBRFS][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_dpbrfs_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_DPBRFS);

    thunk = (fb_dpbrfs_fortran_slot_fn)vtable.ext_ops[FB_OP_DPBRFS][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DPBRFS CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&uplo, &n, &kd, &nrhs, ab, &ldab, afb, &ldafb, b, &ldb, x, &ldx,
          ferr, berr, work, iwork, &info);
    if (info != 172 || g_dpbrfs_cblas_call.called != 1 ||
        g_dpbrfs_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_dpbrfs_cblas_call.uplo != FB_LOWER || g_dpbrfs_cblas_call.n != 3 ||
        g_dpbrfs_cblas_call.kd != 1 || g_dpbrfs_cblas_call.nrhs != 2 ||
        g_dpbrfs_cblas_call.ldab != 2 || g_dpbrfs_cblas_call.ldafb != 2 ||
        g_dpbrfs_cblas_call.ldb != 3 || g_dpbrfs_cblas_call.ldx != 3 ||
        g_dpbrfs_cblas_call.ab != ab || g_dpbrfs_cblas_call.afb != afb ||
        g_dpbrfs_cblas_call.b != b || g_dpbrfs_cblas_call.x != x ||
        g_dpbrfs_cblas_call.ferr != ferr || g_dpbrfs_cblas_call.berr != berr ||
        x[0] != 499.0 || ferr[0] != 1.6 || berr[0] != 0.6) {
        fprintf(stderr, "[FAIL] DPBRFS CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] DPBRFS CBLAS->Fortran thunk maps the all-pointer ABI into the generic double band-refinement entry\n");
    return 0;
}

static int check_cpbrfs_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_cpbrfs_fn thunk = NULL;
    fb_complex_float_t ab[8] = {
        make_cfloat(130.0f), make_cfloat(131.0f), make_cfloat(132.0f), make_cfloat(133.0f),
        make_cfloat(230.0f), make_cfloat(231.0f), make_cfloat(232.0f), make_cfloat(233.0f)
    };
    fb_complex_float_t afb[8] = {
        make_cfloat(330.0f), make_cfloat(331.0f), make_cfloat(332.0f), make_cfloat(333.0f),
        make_cfloat(430.0f), make_cfloat(431.0f), make_cfloat(432.0f), make_cfloat(433.0f)
    };
    fb_complex_float_t b[8] = {
        make_cfloat(1.0f), make_cfloat(2.0f), make_cfloat(3.0f), make_cfloat(4.0f),
        make_cfloat(5.0f), make_cfloat(6.0f), make_cfloat(7.0f), make_cfloat(8.0f)
    };
    fb_complex_float_t x[8] = {
        make_cfloat(9.0f), make_cfloat(10.0f), make_cfloat(11.0f), make_cfloat(12.0f),
        make_cfloat(13.0f), make_cfloat(14.0f), make_cfloat(15.0f), make_cfloat(16.0f)
    };
    float ferr[2] = { 0.0f, 0.0f };
    float berr[2] = { 0.0f, 0.0f };
    float expected_ab_in[8] = { 130.0f, 230.0f, 131.0f, 231.0f, 132.0f, 232.0f, 133.0f, 0.0f };
    float expected_afb_in[8] = { 330.0f, 430.0f, 331.0f, 431.0f, 332.0f, 432.0f, 333.0f, 0.0f };
    float expected_b_in[8] = { 1.0f, 3.0f, 5.0f, 7.0f, 2.0f, 4.0f, 6.0f, 8.0f };
    float expected_x_in[8] = { 9.0f, 11.0f, 13.0f, 15.0f, 10.0f, 12.0f, 14.0f, 16.0f };
    float expected_x_out[8] = { 301.0f, 302.0f, 303.0f, 304.0f, 305.0f, 306.0f, 307.0f, 308.0f };
    float actual_x_out[8] = { 0.0f };
    int index = 0;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cpbrfs_fortran_call, 0, sizeof(g_cpbrfs_fortran_call));

    vtable.ext_ops[FB_OP_CPBRFS][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_cpbrfs_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CPBRFS);

    thunk = (fb_cpbrfs_fn)vtable.ext_ops[FB_OP_CPBRFS][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CPBRFS Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, FB_LOWER, 4, 1, 2,
                 ab, 4, afb, 4, b, 2, x, 2, ferr, berr);
    for (index = 0; index < 8; ++index) {
        actual_x_out[index] = cfloat_real(x[index]);
    }
    if (info != 0 || g_cpbrfs_fortran_call.calls != 1 ||
        g_cpbrfs_fortran_call.uplo != 'L' || g_cpbrfs_fortran_call.n != 4 ||
        g_cpbrfs_fortran_call.kd != 1 || g_cpbrfs_fortran_call.nrhs != 2 ||
        g_cpbrfs_fortran_call.ldab != 2 || g_cpbrfs_fortran_call.ldafb != 2 ||
        g_cpbrfs_fortran_call.ldb != 4 || g_cpbrfs_fortran_call.ldx != 4 ||
        !g_cpbrfs_fortran_call.work_seen || !g_cpbrfs_fortran_call.aux_seen ||
        memcmp(g_cpbrfs_fortran_call.ab_real_snapshot, expected_ab_in, sizeof(expected_ab_in)) != 0 ||
        memcmp(g_cpbrfs_fortran_call.afb_real_snapshot, expected_afb_in, sizeof(expected_afb_in)) != 0 ||
        memcmp(g_cpbrfs_fortran_call.b_real_snapshot, expected_b_in, sizeof(expected_b_in)) != 0 ||
        memcmp(g_cpbrfs_fortran_call.x_real_snapshot, expected_x_in, sizeof(expected_x_in)) != 0 ||
        memcmp(actual_x_out, expected_x_out, sizeof(expected_x_out)) != 0 ||
        ferr[0] != 0.3f || ferr[1] != 0.4f || berr[0] != 0.03f || berr[1] != 0.04f) {
        fprintf(stderr, "[FAIL] CPBRFS Fortran->CBLAS thunk did not preserve complex row-major band refinement semantics\n");
        return 1;
    }

    printf("[PASS] CPBRFS Fortran->CBLAS thunk transposes Hermitian banded inputs, allocates fixed complex scratch, and copies back only X\n");
    return 0;
}

static int check_cpbrfs_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_cpbrfs_fortran_slot_fn thunk = NULL;
    fb_complex_float_t ab[6] = {
        make_cfloat(1.0f), make_cfloat(2.0f), make_cfloat(3.0f),
        make_cfloat(4.0f), make_cfloat(5.0f), make_cfloat(6.0f)
    };
    fb_complex_float_t afb[6] = {
        make_cfloat(7.0f), make_cfloat(8.0f), make_cfloat(9.0f),
        make_cfloat(10.0f), make_cfloat(11.0f), make_cfloat(12.0f)
    };
    fb_complex_float_t b[6] = {
        make_cfloat(13.0f), make_cfloat(14.0f), make_cfloat(15.0f),
        make_cfloat(16.0f), make_cfloat(17.0f), make_cfloat(18.0f)
    };
    fb_complex_float_t x[6] = {
        make_cfloat(19.0f), make_cfloat(20.0f), make_cfloat(21.0f),
        make_cfloat(22.0f), make_cfloat(23.0f), make_cfloat(24.0f)
    };
    float ferr[2] = { 0.0f, 0.0f };
    float berr[2] = { 0.0f, 0.0f };
    fb_complex_float_t work[6];
    float rwork[3] = { 0.0f, 0.0f, 0.0f };
    char uplo = 'U';
    int n = 3;
    int kd = 1;
    int nrhs = 2;
    int ldab = 2;
    int ldafb = 2;
    int ldb = 3;
    int ldx = 3;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cpbrfs_cblas_call, 0, sizeof(g_cpbrfs_cblas_call));
    memset(work, 0, sizeof(work));

    vtable.ext_ops[FB_OP_CPBRFS][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_cpbrfs_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CPBRFS);

    thunk = (fb_cpbrfs_fortran_slot_fn)vtable.ext_ops[FB_OP_CPBRFS][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CPBRFS CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&uplo, &n, &kd, &nrhs, ab, &ldab, afb, &ldafb, b, &ldb, x, &ldx,
          ferr, berr, work, rwork, &info);
    if (info != 173 || g_cpbrfs_cblas_call.called != 1 ||
        g_cpbrfs_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_cpbrfs_cblas_call.uplo != FB_UPPER || g_cpbrfs_cblas_call.n != 3 ||
        g_cpbrfs_cblas_call.kd != 1 || g_cpbrfs_cblas_call.nrhs != 2 ||
        g_cpbrfs_cblas_call.ldab != 2 || g_cpbrfs_cblas_call.ldafb != 2 ||
        g_cpbrfs_cblas_call.ldb != 3 || g_cpbrfs_cblas_call.ldx != 3 ||
        g_cpbrfs_cblas_call.ab != ab || g_cpbrfs_cblas_call.afb != afb ||
        g_cpbrfs_cblas_call.b != b || g_cpbrfs_cblas_call.x != x ||
        g_cpbrfs_cblas_call.ferr != ferr || g_cpbrfs_cblas_call.berr != berr ||
        cfloat_real(x[0]) != 399.0f || ferr[0] != 2.5f || berr[0] != 0.75f) {
        fprintf(stderr, "[FAIL] CPBRFS CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] CPBRFS CBLAS->Fortran thunk maps the all-pointer ABI into the generic complex band-refinement entry\n");
    return 0;
}

static int check_zpbrfs_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_zpbrfs_fn thunk = NULL;
    fb_complex_double_t ab[8] = {
        make_cdouble(130.0), make_cdouble(131.0), make_cdouble(132.0), make_cdouble(133.0),
        make_cdouble(230.0), make_cdouble(231.0), make_cdouble(232.0), make_cdouble(233.0)
    };
    fb_complex_double_t afb[8] = {
        make_cdouble(330.0), make_cdouble(331.0), make_cdouble(332.0), make_cdouble(333.0),
        make_cdouble(430.0), make_cdouble(431.0), make_cdouble(432.0), make_cdouble(433.0)
    };
    fb_complex_double_t b[8] = {
        make_cdouble(1.0), make_cdouble(2.0), make_cdouble(3.0), make_cdouble(4.0),
        make_cdouble(5.0), make_cdouble(6.0), make_cdouble(7.0), make_cdouble(8.0)
    };
    fb_complex_double_t x[8] = {
        make_cdouble(9.0), make_cdouble(10.0), make_cdouble(11.0), make_cdouble(12.0),
        make_cdouble(13.0), make_cdouble(14.0), make_cdouble(15.0), make_cdouble(16.0)
    };
    double ferr[2] = { 0.0, 0.0 };
    double berr[2] = { 0.0, 0.0 };
    double expected_ab_in[8] = { 130.0, 230.0, 131.0, 231.0, 132.0, 232.0, 133.0, 0.0 };
    double expected_afb_in[8] = { 330.0, 430.0, 331.0, 431.0, 332.0, 432.0, 333.0, 0.0 };
    double expected_b_in[8] = { 1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0 };
    double expected_x_in[8] = { 9.0, 11.0, 13.0, 15.0, 10.0, 12.0, 14.0, 16.0 };
    double expected_x_out[8] = { 401.0, 402.0, 403.0, 404.0, 405.0, 406.0, 407.0, 408.0 };
    double actual_x_out[8] = { 0.0 };
    int index = 0;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zpbrfs_fortran_call, 0, sizeof(g_zpbrfs_fortran_call));

    vtable.ext_ops[FB_OP_ZPBRFS][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_zpbrfs_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_ZPBRFS);

    thunk = (fb_zpbrfs_fn)vtable.ext_ops[FB_OP_ZPBRFS][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZPBRFS Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, FB_LOWER, 4, 1, 2,
                 ab, 4, afb, 4, b, 2, x, 2, ferr, berr);
    for (index = 0; index < 8; ++index) {
        actual_x_out[index] = cdouble_real(x[index]);
    }
    if (info != 0 || g_zpbrfs_fortran_call.calls != 1 ||
        g_zpbrfs_fortran_call.uplo != 'L' || g_zpbrfs_fortran_call.n != 4 ||
        g_zpbrfs_fortran_call.kd != 1 || g_zpbrfs_fortran_call.nrhs != 2 ||
        g_zpbrfs_fortran_call.ldab != 2 || g_zpbrfs_fortran_call.ldafb != 2 ||
        g_zpbrfs_fortran_call.ldb != 4 || g_zpbrfs_fortran_call.ldx != 4 ||
        !g_zpbrfs_fortran_call.work_seen || !g_zpbrfs_fortran_call.aux_seen ||
        memcmp(g_zpbrfs_fortran_call.ab_real_snapshot, expected_ab_in, sizeof(expected_ab_in)) != 0 ||
        memcmp(g_zpbrfs_fortran_call.afb_real_snapshot, expected_afb_in, sizeof(expected_afb_in)) != 0 ||
        memcmp(g_zpbrfs_fortran_call.b_real_snapshot, expected_b_in, sizeof(expected_b_in)) != 0 ||
        memcmp(g_zpbrfs_fortran_call.x_real_snapshot, expected_x_in, sizeof(expected_x_in)) != 0 ||
        memcmp(actual_x_out, expected_x_out, sizeof(expected_x_out)) != 0 ||
        ferr[0] != 0.35 || ferr[1] != 0.45 || berr[0] != 0.035 || berr[1] != 0.045) {
        fprintf(stderr, "[FAIL] ZPBRFS Fortran->CBLAS thunk did not preserve complex-double row-major band refinement semantics\n");
        return 1;
    }

    printf("[PASS] ZPBRFS Fortran->CBLAS thunk transposes Hermitian banded inputs, allocates fixed complex scratch, and copies back only X\n");
    return 0;
}

static int check_zpbrfs_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_zpbrfs_fortran_slot_fn thunk = NULL;
    fb_complex_double_t ab[6] = {
        make_cdouble(1.0), make_cdouble(2.0), make_cdouble(3.0),
        make_cdouble(4.0), make_cdouble(5.0), make_cdouble(6.0)
    };
    fb_complex_double_t afb[6] = {
        make_cdouble(7.0), make_cdouble(8.0), make_cdouble(9.0),
        make_cdouble(10.0), make_cdouble(11.0), make_cdouble(12.0)
    };
    fb_complex_double_t b[6] = {
        make_cdouble(13.0), make_cdouble(14.0), make_cdouble(15.0),
        make_cdouble(16.0), make_cdouble(17.0), make_cdouble(18.0)
    };
    fb_complex_double_t x[6] = {
        make_cdouble(19.0), make_cdouble(20.0), make_cdouble(21.0),
        make_cdouble(22.0), make_cdouble(23.0), make_cdouble(24.0)
    };
    double ferr[2] = { 0.0, 0.0 };
    double berr[2] = { 0.0, 0.0 };
    fb_complex_double_t work[6];
    double rwork[3] = { 0.0, 0.0, 0.0 };
    char uplo = 'U';
    int n = 3;
    int kd = 1;
    int nrhs = 2;
    int ldab = 2;
    int ldafb = 2;
    int ldb = 3;
    int ldx = 3;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zpbrfs_cblas_call, 0, sizeof(g_zpbrfs_cblas_call));
    memset(work, 0, sizeof(work));

    vtable.ext_ops[FB_OP_ZPBRFS][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_zpbrfs_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_ZPBRFS);

    thunk = (fb_zpbrfs_fortran_slot_fn)vtable.ext_ops[FB_OP_ZPBRFS][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZPBRFS CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&uplo, &n, &kd, &nrhs, ab, &ldab, afb, &ldafb, b, &ldb, x, &ldx,
          ferr, berr, work, rwork, &info);
    if (info != 174 || g_zpbrfs_cblas_call.called != 1 ||
        g_zpbrfs_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_zpbrfs_cblas_call.uplo != FB_UPPER || g_zpbrfs_cblas_call.n != 3 ||
        g_zpbrfs_cblas_call.kd != 1 || g_zpbrfs_cblas_call.nrhs != 2 ||
        g_zpbrfs_cblas_call.ldab != 2 || g_zpbrfs_cblas_call.ldafb != 2 ||
        g_zpbrfs_cblas_call.ldb != 3 || g_zpbrfs_cblas_call.ldx != 3 ||
        g_zpbrfs_cblas_call.ab != ab || g_zpbrfs_cblas_call.afb != afb ||
        g_zpbrfs_cblas_call.b != b || g_zpbrfs_cblas_call.x != x ||
        g_zpbrfs_cblas_call.ferr != ferr || g_zpbrfs_cblas_call.berr != berr ||
        cdouble_real(x[0]) != 599.0 || ferr[0] != 2.6 || berr[0] != 0.85) {
        fprintf(stderr, "[FAIL] ZPBRFS CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] ZPBRFS CBLAS->Fortran thunk maps the all-pointer ABI into the generic complex-double band-refinement entry\n");
    return 0;
}

int main(void)
{
    if (check_spbrfs_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_spbrfs_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_dpbrfs_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_dpbrfs_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_cpbrfs_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_cpbrfs_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_zpbrfs_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_zpbrfs_cblas_to_fortran() != 0) {
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}