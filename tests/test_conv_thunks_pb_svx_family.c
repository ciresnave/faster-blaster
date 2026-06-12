#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef int (*fb_spbsvx_fn)(fb_layout_t layout, char fact, fb_uplo_t uplo,
                            int n, int kd, int nrhs, float *ab, int ldab,
                            float *afb, int ldafb, char *equed, float *s,
                            float *b, int ldb, float *x, int ldx, float *rcond,
                            float *ferr, float *berr);
typedef int (*fb_dpbsvx_fn)(fb_layout_t layout, char fact, fb_uplo_t uplo,
                            int n, int kd, int nrhs, double *ab, int ldab,
                            double *afb, int ldafb, char *equed, double *s,
                            double *b, int ldb, double *x, int ldx,
                            double *rcond, double *ferr, double *berr);
typedef int (*fb_cpbsvx_fn)(fb_layout_t layout, char fact, fb_uplo_t uplo,
                            int n, int kd, int nrhs, fb_complex_float_t *ab,
                            int ldab, fb_complex_float_t *afb, int ldafb,
                            char *equed, float *s, fb_complex_float_t *b,
                            int ldb, fb_complex_float_t *x, int ldx,
                            float *rcond, float *ferr, float *berr);
typedef int (*fb_zpbsvx_fn)(fb_layout_t layout, char fact, fb_uplo_t uplo,
                            int n, int kd, int nrhs, fb_complex_double_t *ab,
                            int ldab, fb_complex_double_t *afb, int ldafb,
                            char *equed, double *s, fb_complex_double_t *b,
                            int ldb, fb_complex_double_t *x, int ldx,
                            double *rcond, double *ferr, double *berr);

typedef void (*fb_spbsvx_fortran_slot_fn)(char *fact, char *uplo, int *n,
                                          int *kd, int *nrhs, float *ab,
                                          int *ldab, float *afb, int *ldafb,
                                          char *equed, float *s, float *b,
                                          int *ldb, float *x, int *ldx,
                                          float *rcond, float *ferr,
                                          float *berr, float *work,
                                          int *iwork, int *info);
typedef void (*fb_dpbsvx_fortran_slot_fn)(char *fact, char *uplo, int *n,
                                          int *kd, int *nrhs, double *ab,
                                          int *ldab, double *afb, int *ldafb,
                                          char *equed, double *s, double *b,
                                          int *ldb, double *x, int *ldx,
                                          double *rcond, double *ferr,
                                          double *berr, double *work,
                                          int *iwork, int *info);
typedef void (*fb_cpbsvx_fortran_slot_fn)(char *fact, char *uplo, int *n,
                                          int *kd, int *nrhs,
                                          fb_complex_float_t *ab, int *ldab,
                                          fb_complex_float_t *afb, int *ldafb,
                                          char *equed, float *s,
                                          fb_complex_float_t *b, int *ldb,
                                          fb_complex_float_t *x, int *ldx,
                                          float *rcond, float *ferr,
                                          float *berr,
                                          fb_complex_float_t *work,
                                          float *rwork, int *info);
typedef void (*fb_zpbsvx_fortran_slot_fn)(char *fact, char *uplo, int *n,
                                          int *kd, int *nrhs,
                                          fb_complex_double_t *ab, int *ldab,
                                          fb_complex_double_t *afb, int *ldafb,
                                          char *equed, double *s,
                                          fb_complex_double_t *b, int *ldb,
                                          fb_complex_double_t *x, int *ldx,
                                          double *rcond, double *ferr,
                                          double *berr,
                                          fb_complex_double_t *work,
                                          double *rwork, int *info);

static struct {
    int calls;
    char fact;
    char uplo;
    char equed_before;
    int n;
    int kd;
    int nrhs;
    int ldab;
    int ldafb;
    int ldb;
    int ldx;
    int work_seen;
    int aux_seen;
    float s_snapshot[4];
    float ab_snapshot[8];
    float b_snapshot[8];
} g_spbsvx_fortran_call;

static struct {
    int called;
    char fact;
    fb_layout_t layout;
    fb_uplo_t uplo;
    int n;
    int kd;
    int nrhs;
    int ldab;
    int ldafb;
    int ldb;
    int ldx;
    float *ab;
    float *afb;
    char *equed;
    float *s;
    float *b;
    float *x;
    float *rcond;
    float *ferr;
    float *berr;
} g_spbsvx_cblas_call;

static struct {
    int calls;
    char fact;
    char uplo;
    char equed_before;
    int n;
    int kd;
    int nrhs;
    int ldab;
    int ldafb;
    int ldb;
    int ldx;
    int work_seen;
    int aux_seen;
    double s_snapshot[4];
    double ab_snapshot[8];
    double b_snapshot[8];
} g_dpbsvx_fortran_call;

static struct {
    int called;
    char fact;
    fb_layout_t layout;
    fb_uplo_t uplo;
    int n;
    int kd;
    int nrhs;
    int ldab;
    int ldafb;
    int ldb;
    int ldx;
    double *ab;
    double *afb;
    char *equed;
    double *s;
    double *b;
    double *x;
    double *rcond;
    double *ferr;
    double *berr;
} g_dpbsvx_cblas_call;

static struct {
    int calls;
    char fact;
    char uplo;
    char equed_before;
    int n;
    int kd;
    int nrhs;
    int ldab;
    int ldafb;
    int ldb;
    int ldx;
    int work_seen;
    int aux_seen;
    float s_snapshot[4];
    float ab_real_snapshot[8];
    float afb_real_snapshot[8];
    float b_real_snapshot[8];
} g_cpbsvx_fortran_call;

static struct {
    int called;
    char fact;
    fb_layout_t layout;
    fb_uplo_t uplo;
    int n;
    int kd;
    int nrhs;
    int ldab;
    int ldafb;
    int ldb;
    int ldx;
    fb_complex_float_t *ab;
    fb_complex_float_t *afb;
    char *equed;
    float *s;
    fb_complex_float_t *b;
    fb_complex_float_t *x;
    float *rcond;
    float *ferr;
    float *berr;
} g_cpbsvx_cblas_call;

static struct {
    int calls;
    char fact;
    char uplo;
    char equed_before;
    int n;
    int kd;
    int nrhs;
    int ldab;
    int ldafb;
    int ldb;
    int ldx;
    int work_seen;
    int aux_seen;
    double s_snapshot[4];
    double ab_real_snapshot[8];
    double afb_real_snapshot[8];
    double b_real_snapshot[8];
} g_zpbsvx_fortran_call;

static struct {
    int called;
    char fact;
    fb_layout_t layout;
    fb_uplo_t uplo;
    int n;
    int kd;
    int nrhs;
    int ldab;
    int ldafb;
    int ldb;
    int ldx;
    fb_complex_double_t *ab;
    fb_complex_double_t *afb;
    char *equed;
    double *s;
    fb_complex_double_t *b;
    fb_complex_double_t *x;
    double *rcond;
    double *ferr;
    double *berr;
} g_zpbsvx_cblas_call;

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

static void stub_spbsvx_fortran(char *fact, char *uplo, int *n, int *kd,
                                int *nrhs, float *ab, int *ldab, float *afb,
                                int *ldafb, char *equed, float *s, float *b,
                                int *ldb, float *x, int *ldx, float *rcond,
                                float *ferr, float *berr, float *work,
                                int *iwork, int *info)
{
    static const float equilibrated_ab_col[8] = { 0.0f, 620.0f, 611.0f, 621.0f,
                                                  612.0f, 622.0f, 613.0f, 623.0f };
    static const float factor_afb_col[8] = { 0.0f, 720.0f, 711.0f, 721.0f,
                                             712.0f, 722.0f, 713.0f, 723.0f };
    static const float rhs_b_col[8] = { 810.0f, 830.0f, 850.0f, 870.0f,
                                        820.0f, 840.0f, 860.0f, 880.0f };
    static const float solution_x_col[8] = { 910.0f, 930.0f, 950.0f, 970.0f,
                                             920.0f, 940.0f, 960.0f, 980.0f };
    static const float scale_out[4] = { 1.1f, 1.2f, 1.3f, 1.4f };
    int index = 0;

    g_spbsvx_fortran_call.calls += 1;
    g_spbsvx_fortran_call.fact = *fact;
    g_spbsvx_fortran_call.uplo = *uplo;
    g_spbsvx_fortran_call.equed_before = *equed;
    g_spbsvx_fortran_call.n = *n;
    g_spbsvx_fortran_call.kd = *kd;
    g_spbsvx_fortran_call.nrhs = *nrhs;
    g_spbsvx_fortran_call.ldab = *ldab;
    g_spbsvx_fortran_call.ldafb = *ldafb;
    g_spbsvx_fortran_call.ldb = *ldb;
    g_spbsvx_fortran_call.ldx = *ldx;
    g_spbsvx_fortran_call.work_seen = (work != NULL);
    g_spbsvx_fortran_call.aux_seen = (iwork != NULL);
    for (index = 0; index < *n; ++index) {
        g_spbsvx_fortran_call.s_snapshot[index] = s[index];
    }
    for (index = 0; index < (*ldab * *n); ++index) {
        g_spbsvx_fortran_call.ab_snapshot[index] = ab[index];
    }
    for (index = 0; index < (*ldb * *nrhs); ++index) {
        g_spbsvx_fortran_call.b_snapshot[index] = b[index];
    }

    if (*fact == 'E' || *fact == 'e') {
        memcpy(ab, equilibrated_ab_col, sizeof(equilibrated_ab_col));
        memcpy(afb, factor_afb_col, sizeof(factor_afb_col));
        *equed = 'Y';
    } else if (*fact == 'N' || *fact == 'n') {
        memcpy(afb, factor_afb_col, sizeof(factor_afb_col));
        *equed = 'N';
    }

    memcpy(s, scale_out, sizeof(scale_out));
    memcpy(b, rhs_b_col, sizeof(rhs_b_col));
    memcpy(x, solution_x_col, sizeof(solution_x_col));
    *rcond = 0.25f;
    ferr[0] = 0.5f;
    ferr[1] = 0.6f;
    berr[0] = 0.05f;
    berr[1] = 0.06f;
    *info = 0;
}

static int stub_spbsvx_cblas(fb_layout_t layout, char fact, fb_uplo_t uplo,
                             int n, int kd, int nrhs, float *ab, int ldab,
                             float *afb, int ldafb, char *equed, float *s,
                             float *b, int ldb, float *x, int ldx,
                             float *rcond, float *ferr, float *berr)
{
    g_spbsvx_cblas_call.called += 1;
    g_spbsvx_cblas_call.fact = fact;
    g_spbsvx_cblas_call.layout = layout;
    g_spbsvx_cblas_call.uplo = uplo;
    g_spbsvx_cblas_call.n = n;
    g_spbsvx_cblas_call.kd = kd;
    g_spbsvx_cblas_call.nrhs = nrhs;
    g_spbsvx_cblas_call.ldab = ldab;
    g_spbsvx_cblas_call.ldafb = ldafb;
    g_spbsvx_cblas_call.ldb = ldb;
    g_spbsvx_cblas_call.ldx = ldx;
    g_spbsvx_cblas_call.ab = ab;
    g_spbsvx_cblas_call.afb = afb;
    g_spbsvx_cblas_call.equed = equed;
    g_spbsvx_cblas_call.s = s;
    g_spbsvx_cblas_call.b = b;
    g_spbsvx_cblas_call.x = x;
    g_spbsvx_cblas_call.rcond = rcond;
    g_spbsvx_cblas_call.ferr = ferr;
    g_spbsvx_cblas_call.berr = berr;
    *equed = 'Y';
    s[0] = 4.0f;
    ab[0] = 91.0f;
    afb[0] = 92.0f;
    b[0] = 93.0f;
    x[0] = 94.0f;
    *rcond = 0.75f;
    ferr[0] = 1.5f;
    berr[0] = 0.5f;
    return 191;
}

static void stub_dpbsvx_fortran(char *fact, char *uplo, int *n, int *kd,
                                int *nrhs, double *ab, int *ldab, double *afb,
                                int *ldafb, char *equed, double *s, double *b,
                                int *ldb, double *x, int *ldx, double *rcond,
                                double *ferr, double *berr, double *work,
                                int *iwork, int *info)
{
    int index = 0;

    g_dpbsvx_fortran_call.calls += 1;
    g_dpbsvx_fortran_call.fact = *fact;
    g_dpbsvx_fortran_call.uplo = *uplo;
    g_dpbsvx_fortran_call.equed_before = *equed;
    g_dpbsvx_fortran_call.n = *n;
    g_dpbsvx_fortran_call.kd = *kd;
    g_dpbsvx_fortran_call.nrhs = *nrhs;
    g_dpbsvx_fortran_call.ldab = *ldab;
    g_dpbsvx_fortran_call.ldafb = *ldafb;
    g_dpbsvx_fortran_call.ldb = *ldb;
    g_dpbsvx_fortran_call.ldx = *ldx;
    g_dpbsvx_fortran_call.work_seen = (work != NULL);
    g_dpbsvx_fortran_call.aux_seen = (iwork != NULL);
    for (index = 0; index < *n; ++index) {
        g_dpbsvx_fortran_call.s_snapshot[index] = s[index];
    }
    for (index = 0; index < (*ldab * *n); ++index) {
        g_dpbsvx_fortran_call.ab_snapshot[index] = ab[index];
    }
    for (index = 0; index < (*ldb * *nrhs); ++index) {
        g_dpbsvx_fortran_call.b_snapshot[index] = b[index];
    }
    *equed = 'Y';
    s[0] = 7.0;
    ab[0] = 791.0;
    afb[0] = 792.0;
    b[0] = 793.0;
    x[0] = 794.0;
    *rcond = 0.55;
    ferr[0] = 0.65;
    berr[0] = 0.075;
    *info = 0;
}

static int stub_dpbsvx_cblas(fb_layout_t layout, char fact, fb_uplo_t uplo,
                             int n, int kd, int nrhs, double *ab, int ldab,
                             double *afb, int ldafb, char *equed, double *s,
                             double *b, int ldb, double *x, int ldx,
                             double *rcond, double *ferr, double *berr)
{
    g_dpbsvx_cblas_call.called += 1;
    g_dpbsvx_cblas_call.fact = fact;
    g_dpbsvx_cblas_call.layout = layout;
    g_dpbsvx_cblas_call.uplo = uplo;
    g_dpbsvx_cblas_call.n = n;
    g_dpbsvx_cblas_call.kd = kd;
    g_dpbsvx_cblas_call.nrhs = nrhs;
    g_dpbsvx_cblas_call.ldab = ldab;
    g_dpbsvx_cblas_call.ldafb = ldafb;
    g_dpbsvx_cblas_call.ldb = ldb;
    g_dpbsvx_cblas_call.ldx = ldx;
    g_dpbsvx_cblas_call.ab = ab;
    g_dpbsvx_cblas_call.afb = afb;
    g_dpbsvx_cblas_call.equed = equed;
    g_dpbsvx_cblas_call.s = s;
    g_dpbsvx_cblas_call.b = b;
    g_dpbsvx_cblas_call.x = x;
    g_dpbsvx_cblas_call.rcond = rcond;
    g_dpbsvx_cblas_call.ferr = ferr;
    g_dpbsvx_cblas_call.berr = berr;
    *equed = 'N';
    s[0] = 8.0;
    ab[0] = 891.0;
    afb[0] = 892.0;
    b[0] = 893.0;
    x[0] = 894.0;
    *rcond = 0.85;
    ferr[0] = 1.85;
    berr[0] = 0.95;
    return 192;
}

static void stub_cpbsvx_fortran(char *fact, char *uplo, int *n, int *kd,
                                int *nrhs, fb_complex_float_t *ab, int *ldab,
                                fb_complex_float_t *afb, int *ldafb,
                                char *equed, float *s, fb_complex_float_t *b,
                                int *ldb, fb_complex_float_t *x, int *ldx,
                                float *rcond, float *ferr, float *berr,
                                fb_complex_float_t *work, float *rwork,
                                int *info)
{
    static const float mutated_ab_col_real[8] = { 330.0f, 430.0f, 331.0f, 431.0f,
                                                  332.0f, 432.0f, 333.0f, 0.0f };
    static const float mutated_afb_col_real[8] = { 350.0f, 450.0f, 351.0f, 451.0f,
                                                   352.0f, 452.0f, 353.0f, 0.0f };
    static const float equilibrated_ab_col_real[8] = { 530.0f, 630.0f, 531.0f, 631.0f,
                                                       532.0f, 632.0f, 533.0f, 0.0f };
    static const float factor_afb_col_real[8] = { 730.0f, 830.0f, 731.0f, 831.0f,
                                                  732.0f, 832.0f, 733.0f, 0.0f };
    static const float rhs_b_col_real[8] = { 910.0f, 930.0f, 950.0f, 970.0f,
                                             920.0f, 940.0f, 960.0f, 980.0f };
    static const float solution_x_col_real[8] = { 1010.0f, 1030.0f, 1050.0f, 1070.0f,
                                                  1020.0f, 1040.0f, 1060.0f, 1080.0f };
    static const float scale_out[4] = { 2.1f, 2.2f, 2.3f, 2.4f };
    int index = 0;

    g_cpbsvx_fortran_call.calls += 1;
    g_cpbsvx_fortran_call.fact = *fact;
    g_cpbsvx_fortran_call.uplo = *uplo;
    g_cpbsvx_fortran_call.equed_before = *equed;
    g_cpbsvx_fortran_call.n = *n;
    g_cpbsvx_fortran_call.kd = *kd;
    g_cpbsvx_fortran_call.nrhs = *nrhs;
    g_cpbsvx_fortran_call.ldab = *ldab;
    g_cpbsvx_fortran_call.ldafb = *ldafb;
    g_cpbsvx_fortran_call.ldb = *ldb;
    g_cpbsvx_fortran_call.ldx = *ldx;
    g_cpbsvx_fortran_call.work_seen = (work != NULL);
    g_cpbsvx_fortran_call.aux_seen = (rwork != NULL);
    for (index = 0; index < *n; ++index) {
        g_cpbsvx_fortran_call.s_snapshot[index] = s[index];
    }
    for (index = 0; index < (*ldab * *n); ++index) {
        g_cpbsvx_fortran_call.ab_real_snapshot[index] = cfloat_real(ab[index]);
        g_cpbsvx_fortran_call.afb_real_snapshot[index] = cfloat_real(afb[index]);
    }
    for (index = 0; index < (*ldb * *nrhs); ++index) {
        g_cpbsvx_fortran_call.b_real_snapshot[index] = cfloat_real(b[index]);
    }

    if (*fact == 'F' || *fact == 'f') {
        for (index = 0; index < 8; ++index) {
            ab[index] = make_cfloat(mutated_ab_col_real[index]);
            afb[index] = make_cfloat(mutated_afb_col_real[index]);
        }
    } else if (*fact == 'E' || *fact == 'e') {
        for (index = 0; index < 8; ++index) {
            ab[index] = make_cfloat(equilibrated_ab_col_real[index]);
            afb[index] = make_cfloat(factor_afb_col_real[index]);
        }
        memcpy(s, scale_out, sizeof(scale_out));
        *equed = 'Y';
    }

    for (index = 0; index < 8; ++index) {
        b[index] = make_cfloat(rhs_b_col_real[index]);
        x[index] = make_cfloat(solution_x_col_real[index]);
    }
    *rcond = 0.35f;
    ferr[0] = 0.7f;
    ferr[1] = 0.8f;
    berr[0] = 0.07f;
    berr[1] = 0.08f;
    *info = 0;
}

static int stub_cpbsvx_cblas(fb_layout_t layout, char fact, fb_uplo_t uplo,
                             int n, int kd, int nrhs, fb_complex_float_t *ab,
                             int ldab, fb_complex_float_t *afb, int ldafb,
                             char *equed, float *s, fb_complex_float_t *b,
                             int ldb, fb_complex_float_t *x, int ldx,
                             float *rcond, float *ferr, float *berr)
{
    g_cpbsvx_cblas_call.called += 1;
    g_cpbsvx_cblas_call.fact = fact;
    g_cpbsvx_cblas_call.layout = layout;
    g_cpbsvx_cblas_call.uplo = uplo;
    g_cpbsvx_cblas_call.n = n;
    g_cpbsvx_cblas_call.kd = kd;
    g_cpbsvx_cblas_call.nrhs = nrhs;
    g_cpbsvx_cblas_call.ldab = ldab;
    g_cpbsvx_cblas_call.ldafb = ldafb;
    g_cpbsvx_cblas_call.ldb = ldb;
    g_cpbsvx_cblas_call.ldx = ldx;
    g_cpbsvx_cblas_call.ab = ab;
    g_cpbsvx_cblas_call.afb = afb;
    g_cpbsvx_cblas_call.equed = equed;
    g_cpbsvx_cblas_call.s = s;
    g_cpbsvx_cblas_call.b = b;
    g_cpbsvx_cblas_call.x = x;
    g_cpbsvx_cblas_call.rcond = rcond;
    g_cpbsvx_cblas_call.ferr = ferr;
    g_cpbsvx_cblas_call.berr = berr;
    *equed = 'N';
    s[0] = 6.0f;
    ab[0] = make_cfloat(191.0f);
    afb[0] = make_cfloat(192.0f);
    b[0] = make_cfloat(193.0f);
    x[0] = make_cfloat(194.0f);
    *rcond = 0.95f;
    ferr[0] = 1.7f;
    berr[0] = 0.9f;
    return 193;
}

static void stub_zpbsvx_fortran(char *fact, char *uplo, int *n, int *kd,
                                int *nrhs, fb_complex_double_t *ab, int *ldab,
                                fb_complex_double_t *afb, int *ldafb,
                                char *equed, double *s, fb_complex_double_t *b,
                                int *ldb, fb_complex_double_t *x, int *ldx,
                                double *rcond, double *ferr, double *berr,
                                fb_complex_double_t *work, double *rwork,
                                int *info)
{
    int index = 0;

    g_zpbsvx_fortran_call.calls += 1;
    g_zpbsvx_fortran_call.fact = *fact;
    g_zpbsvx_fortran_call.uplo = *uplo;
    g_zpbsvx_fortran_call.equed_before = *equed;
    g_zpbsvx_fortran_call.n = *n;
    g_zpbsvx_fortran_call.kd = *kd;
    g_zpbsvx_fortran_call.nrhs = *nrhs;
    g_zpbsvx_fortran_call.ldab = *ldab;
    g_zpbsvx_fortran_call.ldafb = *ldafb;
    g_zpbsvx_fortran_call.ldb = *ldb;
    g_zpbsvx_fortran_call.ldx = *ldx;
    g_zpbsvx_fortran_call.work_seen = (work != NULL);
    g_zpbsvx_fortran_call.aux_seen = (rwork != NULL);
    for (index = 0; index < *n; ++index) {
        g_zpbsvx_fortran_call.s_snapshot[index] = s[index];
    }
    for (index = 0; index < (*ldab * *n); ++index) {
        g_zpbsvx_fortran_call.ab_real_snapshot[index] = cdouble_real(ab[index]);
        g_zpbsvx_fortran_call.afb_real_snapshot[index] = cdouble_real(afb[index]);
    }
    for (index = 0; index < (*ldb * *nrhs); ++index) {
        g_zpbsvx_fortran_call.b_real_snapshot[index] = cdouble_real(b[index]);
    }
    *equed = 'Y';
    s[0] = 9.0;
    ab[0] = make_cdouble(991.0);
    afb[0] = make_cdouble(992.0);
    b[0] = make_cdouble(993.0);
    x[0] = make_cdouble(994.0);
    *rcond = 0.95;
    ferr[0] = 1.05;
    berr[0] = 0.105;
    *info = 0;
}

static int stub_zpbsvx_cblas(fb_layout_t layout, char fact, fb_uplo_t uplo,
                             int n, int kd, int nrhs, fb_complex_double_t *ab,
                             int ldab, fb_complex_double_t *afb, int ldafb,
                             char *equed, double *s, fb_complex_double_t *b,
                             int ldb, fb_complex_double_t *x, int ldx,
                             double *rcond, double *ferr, double *berr)
{
    g_zpbsvx_cblas_call.called += 1;
    g_zpbsvx_cblas_call.fact = fact;
    g_zpbsvx_cblas_call.layout = layout;
    g_zpbsvx_cblas_call.uplo = uplo;
    g_zpbsvx_cblas_call.n = n;
    g_zpbsvx_cblas_call.kd = kd;
    g_zpbsvx_cblas_call.nrhs = nrhs;
    g_zpbsvx_cblas_call.ldab = ldab;
    g_zpbsvx_cblas_call.ldafb = ldafb;
    g_zpbsvx_cblas_call.ldb = ldb;
    g_zpbsvx_cblas_call.ldx = ldx;
    g_zpbsvx_cblas_call.ab = ab;
    g_zpbsvx_cblas_call.afb = afb;
    g_zpbsvx_cblas_call.equed = equed;
    g_zpbsvx_cblas_call.s = s;
    g_zpbsvx_cblas_call.b = b;
    g_zpbsvx_cblas_call.x = x;
    g_zpbsvx_cblas_call.rcond = rcond;
    g_zpbsvx_cblas_call.ferr = ferr;
    g_zpbsvx_cblas_call.berr = berr;
    *equed = 'N';
    s[0] = 10.0;
    ab[0] = make_cdouble(1091.0);
    afb[0] = make_cdouble(1092.0);
    b[0] = make_cdouble(1093.0);
    x[0] = make_cdouble(1094.0);
    *rcond = 1.15;
    ferr[0] = 2.15;
    berr[0] = 1.15;
    return 194;
}

static int check_spbsvx_fortran_to_cblas_fact_n(void)
{
    fb_backend_vtable_t vtable;
    fb_spbsvx_fn thunk = NULL;
    float ab[8] = { 10.0f, 11.0f, 12.0f, 13.0f, 20.0f, 21.0f, 22.0f, 23.0f };
    float afb[8] = { 100.0f, 101.0f, 102.0f, 103.0f, 110.0f, 111.0f, 112.0f, 113.0f };
    float b[8] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };
    float x[8] = { 51.0f, 52.0f, 53.0f, 54.0f, 55.0f, 56.0f, 57.0f, 58.0f };
    float s[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float original_ab[8] = { 10.0f, 11.0f, 12.0f, 13.0f, 20.0f, 21.0f, 22.0f, 23.0f };
    float expected_ab_in[8] = { 0.0f, 20.0f, 11.0f, 21.0f, 12.0f, 22.0f, 13.0f, 23.0f };
    float expected_b_in[8] = { 1.0f, 3.0f, 5.0f, 7.0f, 2.0f, 4.0f, 6.0f, 8.0f };
    float expected_afb_out[8] = { 0.0f, 711.0f, 712.0f, 713.0f, 720.0f, 721.0f, 722.0f, 723.0f };
    float expected_b_out[8] = { 810.0f, 820.0f, 830.0f, 840.0f, 850.0f, 860.0f, 870.0f, 880.0f };
    float expected_x_out[8] = { 910.0f, 920.0f, 930.0f, 940.0f, 950.0f, 960.0f, 970.0f, 980.0f };
    float expected_s_out[4] = { 1.1f, 1.2f, 1.3f, 1.4f };
    char equed = 'N';
    float rcond = 0.0f;
    float ferr[2] = { 0.0f, 0.0f };
    float berr[2] = { 0.0f, 0.0f };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_spbsvx_fortran_call, 0, sizeof(g_spbsvx_fortran_call));

    vtable.ext_ops[FB_OP_SPBSVX][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_spbsvx_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SPBSVX);

    thunk = (fb_spbsvx_fn)vtable.ext_ops[FB_OP_SPBSVX][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SPBSVX fact=N Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 'N', FB_UPPER, 4, 1, 2, ab, 4, afb, 4,
                 &equed, s, b, 2, x, 2, &rcond, ferr, berr);
    if (info != 0 || g_spbsvx_fortran_call.calls != 1 ||
        g_spbsvx_fortran_call.fact != 'N' || g_spbsvx_fortran_call.uplo != 'U' ||
        g_spbsvx_fortran_call.equed_before != 'N' ||
        g_spbsvx_fortran_call.n != 4 || g_spbsvx_fortran_call.kd != 1 ||
        g_spbsvx_fortran_call.nrhs != 2 || g_spbsvx_fortran_call.ldab != 2 ||
        g_spbsvx_fortran_call.ldafb != 2 || g_spbsvx_fortran_call.ldb != 4 ||
        g_spbsvx_fortran_call.ldx != 4 || !g_spbsvx_fortran_call.work_seen ||
        !g_spbsvx_fortran_call.aux_seen ||
        memcmp(g_spbsvx_fortran_call.ab_snapshot, expected_ab_in, sizeof(expected_ab_in)) != 0 ||
        memcmp(g_spbsvx_fortran_call.b_snapshot, expected_b_in, sizeof(expected_b_in)) != 0 ||
        memcmp(ab, original_ab, sizeof(original_ab)) != 0 ||
        memcmp(afb, expected_afb_out, sizeof(expected_afb_out)) != 0 ||
        memcmp(b, expected_b_out, sizeof(expected_b_out)) != 0 ||
        memcmp(x, expected_x_out, sizeof(expected_x_out)) != 0 ||
        memcmp(s, expected_s_out, sizeof(expected_s_out)) != 0 ||
        equed != 'N' || rcond != 0.25f || ferr[0] != 0.5f || ferr[1] != 0.6f ||
        berr[0] != 0.05f || berr[1] != 0.06f) {
        fprintf(stderr, "[FAIL] SPBSVX fact=N Fortran->CBLAS thunk did not preserve the row-major expert-driver copy policy\n");
        return 1;
    }

    printf("[PASS] SPBSVX fact=N Fortran->CBLAS thunk transposes AB/B in, copies back AFB/B/X, and leaves AB untouched\n");
    return 0;
}

static int check_spbsvx_fortran_to_cblas_fact_e(void)
{
    fb_backend_vtable_t vtable;
    fb_spbsvx_fn thunk = NULL;
    float ab[8] = { 14.0f, 15.0f, 16.0f, 17.0f, 24.0f, 25.0f, 26.0f, 27.0f };
    float afb[8] = { 140.0f, 141.0f, 142.0f, 143.0f, 150.0f, 151.0f, 152.0f, 153.0f };
    float b[8] = { 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f };
    float x[8] = { 61.0f, 62.0f, 63.0f, 64.0f, 65.0f, 66.0f, 67.0f, 68.0f };
    float s[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float expected_ab_in[8] = { 0.0f, 24.0f, 15.0f, 25.0f, 16.0f, 26.0f, 17.0f, 27.0f };
    float expected_b_in[8] = { 11.0f, 13.0f, 15.0f, 17.0f, 12.0f, 14.0f, 16.0f, 18.0f };
    float expected_ab_out[8] = { 0.0f, 611.0f, 612.0f, 613.0f, 620.0f, 621.0f, 622.0f, 623.0f };
    float expected_afb_out[8] = { 0.0f, 711.0f, 712.0f, 713.0f, 720.0f, 721.0f, 722.0f, 723.0f };
    float expected_b_out[8] = { 810.0f, 820.0f, 830.0f, 840.0f, 850.0f, 860.0f, 870.0f, 880.0f };
    float expected_x_out[8] = { 910.0f, 920.0f, 930.0f, 940.0f, 950.0f, 960.0f, 970.0f, 980.0f };
    float expected_s_out[4] = { 1.1f, 1.2f, 1.3f, 1.4f };
    char equed = 'N';
    float rcond = 0.0f;
    float ferr[2] = { 0.0f, 0.0f };
    float berr[2] = { 0.0f, 0.0f };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_spbsvx_fortran_call, 0, sizeof(g_spbsvx_fortran_call));

    vtable.ext_ops[FB_OP_SPBSVX][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_spbsvx_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SPBSVX);

    thunk = (fb_spbsvx_fn)vtable.ext_ops[FB_OP_SPBSVX][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SPBSVX fact=E Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 'E', FB_UPPER, 4, 1, 2, ab, 4, afb, 4,
                 &equed, s, b, 2, x, 2, &rcond, ferr, berr);
    if (info != 0 || g_spbsvx_fortran_call.calls != 1 ||
        g_spbsvx_fortran_call.fact != 'E' || g_spbsvx_fortran_call.uplo != 'U' ||
        g_spbsvx_fortran_call.equed_before != 'N' ||
        memcmp(g_spbsvx_fortran_call.ab_snapshot, expected_ab_in, sizeof(expected_ab_in)) != 0 ||
        memcmp(g_spbsvx_fortran_call.b_snapshot, expected_b_in, sizeof(expected_b_in)) != 0 ||
        memcmp(ab, expected_ab_out, sizeof(expected_ab_out)) != 0 ||
        memcmp(afb, expected_afb_out, sizeof(expected_afb_out)) != 0 ||
        memcmp(b, expected_b_out, sizeof(expected_b_out)) != 0 ||
        memcmp(x, expected_x_out, sizeof(expected_x_out)) != 0 ||
        memcmp(s, expected_s_out, sizeof(expected_s_out)) != 0 ||
        equed != 'Y' || rcond != 0.25f || ferr[0] != 0.5f || ferr[1] != 0.6f ||
        berr[0] != 0.05f || berr[1] != 0.06f) {
        fprintf(stderr, "[FAIL] SPBSVX fact=E Fortran->CBLAS thunk did not copy back equilibrated AB/AFB/B/X as expected\n");
        return 1;
    }

    printf("[PASS] SPBSVX fact=E Fortran->CBLAS thunk copies back equilibrated AB, factored AFB, and row-major RHS/solution matrices\n");
    return 0;
}

static int check_spbsvx_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_spbsvx_fortran_slot_fn thunk = NULL;
    float ab[6] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
    float afb[6] = { 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f };
    float b[6] = { 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f };
    float x[6] = { 31.0f, 32.0f, 33.0f, 34.0f, 35.0f, 36.0f };
    float s[3] = { 0.0f, 0.0f, 0.0f };
    char fact = 'F';
    char uplo = 'L';
    char equed = 'N';
    int n = 3;
    int kd = 1;
    int nrhs = 2;
    int ldab = 2;
    int ldafb = 2;
    int ldb = 3;
    int ldx = 3;
    float rcond = 0.0f;
    float ferr[2] = { 0.0f, 0.0f };
    float berr[2] = { 0.0f, 0.0f };
    float work[9] = { 0.0f };
    int iwork[3] = { 0 };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_spbsvx_cblas_call, 0, sizeof(g_spbsvx_cblas_call));

    vtable.ext_ops[FB_OP_SPBSVX][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_spbsvx_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SPBSVX);

    thunk = (fb_spbsvx_fortran_slot_fn)vtable.ext_ops[FB_OP_SPBSVX][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SPBSVX CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&fact, &uplo, &n, &kd, &nrhs, ab, &ldab, afb, &ldafb, &equed, s, b,
        &ldb, x, &ldx, &rcond, ferr, berr, work, iwork, &info);
    if (info != 191 || g_spbsvx_cblas_call.called != 1 ||
        g_spbsvx_cblas_call.fact != 'F' ||
        g_spbsvx_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_spbsvx_cblas_call.uplo != FB_LOWER ||
        g_spbsvx_cblas_call.n != 3 || g_spbsvx_cblas_call.kd != 1 ||
        g_spbsvx_cblas_call.nrhs != 2 || g_spbsvx_cblas_call.ldab != 2 ||
        g_spbsvx_cblas_call.ldafb != 2 || g_spbsvx_cblas_call.ldb != 3 ||
        g_spbsvx_cblas_call.ldx != 3 || g_spbsvx_cblas_call.ab != ab ||
        g_spbsvx_cblas_call.afb != afb || g_spbsvx_cblas_call.equed != &equed ||
        g_spbsvx_cblas_call.s != s || g_spbsvx_cblas_call.b != b ||
        g_spbsvx_cblas_call.x != x || g_spbsvx_cblas_call.rcond != &rcond ||
        g_spbsvx_cblas_call.ferr != ferr || g_spbsvx_cblas_call.berr != berr ||
        equed != 'Y' || s[0] != 4.0f || ab[0] != 91.0f || afb[0] != 92.0f ||
        b[0] != 93.0f || x[0] != 94.0f || rcond != 0.75f || ferr[0] != 1.5f ||
        berr[0] != 0.5f) {
        fprintf(stderr, "[FAIL] SPBSVX CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] SPBSVX CBLAS->Fortran thunk forwards the pointer ABI into the expert-driver C entry\n");
    return 0;
}

static int check_cpbsvx_fortran_to_cblas_fact_f(void)
{
    fb_backend_vtable_t vtable;
    fb_cpbsvx_fn thunk = NULL;
    fb_complex_float_t ab[8] = {
        make_cfloat(30.0f), make_cfloat(31.0f), make_cfloat(32.0f), make_cfloat(33.0f),
        make_cfloat(40.0f), make_cfloat(41.0f), make_cfloat(42.0f), make_cfloat(43.0f)
    };
    fb_complex_float_t afb[8] = {
        make_cfloat(130.0f), make_cfloat(131.0f), make_cfloat(132.0f), make_cfloat(133.0f),
        make_cfloat(230.0f), make_cfloat(231.0f), make_cfloat(232.0f), make_cfloat(233.0f)
    };
    fb_complex_float_t b[8] = {
        make_cfloat(1.0f), make_cfloat(2.0f), make_cfloat(3.0f), make_cfloat(4.0f),
        make_cfloat(5.0f), make_cfloat(6.0f), make_cfloat(7.0f), make_cfloat(8.0f)
    };
    fb_complex_float_t x[8] = {
        make_cfloat(51.0f), make_cfloat(52.0f), make_cfloat(53.0f), make_cfloat(54.0f),
        make_cfloat(55.0f), make_cfloat(56.0f), make_cfloat(57.0f), make_cfloat(58.0f)
    };
    float s[4] = { 3.1f, 3.2f, 3.3f, 3.4f };
    float expected_ab_in[8] = { 30.0f, 40.0f, 31.0f, 41.0f, 32.0f, 42.0f, 33.0f, 0.0f };
    float expected_afb_in[8] = { 130.0f, 230.0f, 131.0f, 231.0f, 132.0f, 232.0f, 133.0f, 0.0f };
    float expected_b_in[8] = { 1.0f, 3.0f, 5.0f, 7.0f, 2.0f, 4.0f, 6.0f, 8.0f };
    float expected_b_out[8] = { 910.0f, 920.0f, 930.0f, 940.0f, 950.0f, 960.0f, 970.0f, 980.0f };
    float expected_x_out[8] = { 1010.0f, 1020.0f, 1030.0f, 1040.0f, 1050.0f, 1060.0f, 1070.0f, 1080.0f };
    float actual_ab[8] = { 0.0f };
    float actual_afb[8] = { 0.0f };
    float actual_b[8] = { 0.0f };
    float actual_x[8] = { 0.0f };
    char equed = 'Y';
    float rcond = 0.0f;
    float ferr[2] = { 0.0f, 0.0f };
    float berr[2] = { 0.0f, 0.0f };
    int index = 0;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cpbsvx_fortran_call, 0, sizeof(g_cpbsvx_fortran_call));

    vtable.ext_ops[FB_OP_CPBSVX][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_cpbsvx_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CPBSVX);

    thunk = (fb_cpbsvx_fn)vtable.ext_ops[FB_OP_CPBSVX][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CPBSVX fact=F Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 'F', FB_LOWER, 4, 1, 2, ab, 4, afb, 4,
                 &equed, s, b, 2, x, 2, &rcond, ferr, berr);
    for (index = 0; index < 8; ++index) {
        actual_ab[index] = cfloat_real(ab[index]);
        actual_afb[index] = cfloat_real(afb[index]);
        actual_b[index] = cfloat_real(b[index]);
        actual_x[index] = cfloat_real(x[index]);
    }
    if (info != 0 || g_cpbsvx_fortran_call.calls != 1 ||
        g_cpbsvx_fortran_call.fact != 'F' || g_cpbsvx_fortran_call.uplo != 'L' ||
        g_cpbsvx_fortran_call.equed_before != 'Y' || !g_cpbsvx_fortran_call.work_seen ||
        !g_cpbsvx_fortran_call.aux_seen ||
        memcmp(g_cpbsvx_fortran_call.s_snapshot, s, sizeof(s)) != 0 ||
        memcmp(g_cpbsvx_fortran_call.ab_real_snapshot, expected_ab_in, sizeof(expected_ab_in)) != 0 ||
        memcmp(g_cpbsvx_fortran_call.afb_real_snapshot, expected_afb_in, sizeof(expected_afb_in)) != 0 ||
        memcmp(g_cpbsvx_fortran_call.b_real_snapshot, expected_b_in, sizeof(expected_b_in)) != 0 ||
        memcmp(actual_ab, (float[]){ 30.0f, 31.0f, 32.0f, 33.0f, 40.0f, 41.0f, 42.0f, 43.0f }, sizeof(actual_ab)) != 0 ||
        memcmp(actual_afb, (float[]){ 130.0f, 131.0f, 132.0f, 133.0f, 230.0f, 231.0f, 232.0f, 233.0f }, sizeof(actual_afb)) != 0 ||
        memcmp(actual_b, expected_b_out, sizeof(expected_b_out)) != 0 ||
        memcmp(actual_x, expected_x_out, sizeof(expected_x_out)) != 0 ||
        equed != 'Y' || rcond != 0.35f || ferr[0] != 0.7f || ferr[1] != 0.8f ||
        berr[0] != 0.07f || berr[1] != 0.08f) {
        fprintf(stderr, "[FAIL] CPBSVX fact=F Fortran->CBLAS thunk did not preserve the factored-input copy policy\n");
        return 1;
    }

    printf("[PASS] CPBSVX fact=F Fortran->CBLAS thunk transposes AB/AFB/B in, copies back only B/X, and leaves AB/AFB untouched\n");
    return 0;
}

static int check_cpbsvx_fortran_to_cblas_fact_e(void)
{
    fb_backend_vtable_t vtable;
    fb_cpbsvx_fn thunk = NULL;
    fb_complex_float_t ab[8] = {
        make_cfloat(34.0f), make_cfloat(35.0f), make_cfloat(36.0f), make_cfloat(37.0f),
        make_cfloat(44.0f), make_cfloat(45.0f), make_cfloat(46.0f), make_cfloat(47.0f)
    };
    fb_complex_float_t afb[8] = {
        make_cfloat(134.0f), make_cfloat(135.0f), make_cfloat(136.0f), make_cfloat(137.0f),
        make_cfloat(234.0f), make_cfloat(235.0f), make_cfloat(236.0f), make_cfloat(237.0f)
    };
    fb_complex_float_t b[8] = {
        make_cfloat(11.0f), make_cfloat(12.0f), make_cfloat(13.0f), make_cfloat(14.0f),
        make_cfloat(15.0f), make_cfloat(16.0f), make_cfloat(17.0f), make_cfloat(18.0f)
    };
    fb_complex_float_t x[8] = {
        make_cfloat(61.0f), make_cfloat(62.0f), make_cfloat(63.0f), make_cfloat(64.0f),
        make_cfloat(65.0f), make_cfloat(66.0f), make_cfloat(67.0f), make_cfloat(68.0f)
    };
    float s[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float expected_ab_in[8] = { 34.0f, 44.0f, 35.0f, 45.0f, 36.0f, 46.0f, 37.0f, 0.0f };
    float expected_b_in[8] = { 11.0f, 13.0f, 15.0f, 17.0f, 12.0f, 14.0f, 16.0f, 18.0f };
    float expected_ab_out[8] = { 530.0f, 531.0f, 532.0f, 533.0f, 630.0f, 631.0f, 632.0f, 0.0f };
    float expected_afb_out[8] = { 730.0f, 731.0f, 732.0f, 733.0f, 830.0f, 831.0f, 832.0f, 0.0f };
    float expected_b_out[8] = { 910.0f, 920.0f, 930.0f, 940.0f, 950.0f, 960.0f, 970.0f, 980.0f };
    float expected_x_out[8] = { 1010.0f, 1020.0f, 1030.0f, 1040.0f, 1050.0f, 1060.0f, 1070.0f, 1080.0f };
    float expected_s_out[4] = { 2.1f, 2.2f, 2.3f, 2.4f };
    float actual_ab[8] = { 0.0f };
    float actual_afb[8] = { 0.0f };
    float actual_b[8] = { 0.0f };
    float actual_x[8] = { 0.0f };
    char equed = 'N';
    float rcond = 0.0f;
    float ferr[2] = { 0.0f, 0.0f };
    float berr[2] = { 0.0f, 0.0f };
    int index = 0;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cpbsvx_fortran_call, 0, sizeof(g_cpbsvx_fortran_call));

    vtable.ext_ops[FB_OP_CPBSVX][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_cpbsvx_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CPBSVX);

    thunk = (fb_cpbsvx_fn)vtable.ext_ops[FB_OP_CPBSVX][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CPBSVX fact=E Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 'E', FB_LOWER, 4, 1, 2, ab, 4, afb, 4,
                 &equed, s, b, 2, x, 2, &rcond, ferr, berr);
    for (index = 0; index < 8; ++index) {
        actual_ab[index] = cfloat_real(ab[index]);
        actual_afb[index] = cfloat_real(afb[index]);
        actual_b[index] = cfloat_real(b[index]);
        actual_x[index] = cfloat_real(x[index]);
    }
    if (info != 0 || g_cpbsvx_fortran_call.calls != 1 ||
        g_cpbsvx_fortran_call.fact != 'E' || g_cpbsvx_fortran_call.uplo != 'L' ||
        g_cpbsvx_fortran_call.equed_before != 'N' ||
        memcmp(g_cpbsvx_fortran_call.ab_real_snapshot, expected_ab_in, sizeof(expected_ab_in)) != 0 ||
        memcmp(g_cpbsvx_fortran_call.b_real_snapshot, expected_b_in, sizeof(expected_b_in)) != 0 ||
        memcmp(actual_ab, expected_ab_out, sizeof(expected_ab_out)) != 0 ||
        memcmp(actual_afb, expected_afb_out, sizeof(expected_afb_out)) != 0 ||
        memcmp(actual_b, expected_b_out, sizeof(expected_b_out)) != 0 ||
        memcmp(actual_x, expected_x_out, sizeof(expected_x_out)) != 0 ||
        memcmp(s, expected_s_out, sizeof(expected_s_out)) != 0 ||
        equed != 'Y' || rcond != 0.35f || ferr[0] != 0.7f || ferr[1] != 0.8f ||
        berr[0] != 0.07f || berr[1] != 0.08f) {
        fprintf(stderr, "[FAIL] CPBSVX fact=E Fortran->CBLAS thunk did not copy back equilibrated AB/AFB/B/X as expected\n");
        return 1;
    }

    printf("[PASS] CPBSVX fact=E Fortran->CBLAS thunk copies back equilibrated AB, factored AFB, and complex row-major RHS/solution matrices\n");
    return 0;
}

static int check_cpbsvx_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_cpbsvx_fortran_slot_fn thunk = NULL;
    fb_complex_float_t ab[6];
    fb_complex_float_t afb[6];
    fb_complex_float_t b[6];
    fb_complex_float_t x[6];
    float s[3] = { 0.0f, 0.0f, 0.0f };
    char fact = 'N';
    char uplo = 'U';
    char equed = 'Y';
    int n = 3;
    int kd = 1;
    int nrhs = 2;
    int ldab = 2;
    int ldafb = 2;
    int ldb = 3;
    int ldx = 3;
    float rcond = 0.0f;
    float ferr[2] = { 0.0f, 0.0f };
    float berr[2] = { 0.0f, 0.0f };
    fb_complex_float_t work[6];
    float rwork[3] = { 0.0f, 0.0f, 0.0f };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cpbsvx_cblas_call, 0, sizeof(g_cpbsvx_cblas_call));
    memset(ab, 0, sizeof(ab));
    memset(afb, 0, sizeof(afb));
    memset(b, 0, sizeof(b));
    memset(x, 0, sizeof(x));
    memset(work, 0, sizeof(work));

    vtable.ext_ops[FB_OP_CPBSVX][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_cpbsvx_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CPBSVX);

    thunk = (fb_cpbsvx_fortran_slot_fn)vtable.ext_ops[FB_OP_CPBSVX][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CPBSVX CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&fact, &uplo, &n, &kd, &nrhs, ab, &ldab, afb, &ldafb, &equed, s, b,
        &ldb, x, &ldx, &rcond, ferr, berr, work, rwork, &info);
    if (info != 193 || g_cpbsvx_cblas_call.called != 1 ||
        g_cpbsvx_cblas_call.fact != 'N' ||
        g_cpbsvx_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_cpbsvx_cblas_call.uplo != FB_UPPER ||
        g_cpbsvx_cblas_call.n != 3 || g_cpbsvx_cblas_call.kd != 1 ||
        g_cpbsvx_cblas_call.nrhs != 2 || g_cpbsvx_cblas_call.ldab != 2 ||
        g_cpbsvx_cblas_call.ldafb != 2 || g_cpbsvx_cblas_call.ldb != 3 ||
        g_cpbsvx_cblas_call.ldx != 3 || g_cpbsvx_cblas_call.ab != ab ||
        g_cpbsvx_cblas_call.afb != afb || g_cpbsvx_cblas_call.equed != &equed ||
        g_cpbsvx_cblas_call.s != s || g_cpbsvx_cblas_call.b != b ||
        g_cpbsvx_cblas_call.x != x || g_cpbsvx_cblas_call.rcond != &rcond ||
        g_cpbsvx_cblas_call.ferr != ferr || g_cpbsvx_cblas_call.berr != berr ||
        equed != 'N' || s[0] != 6.0f || cfloat_real(ab[0]) != 191.0f ||
        cfloat_real(afb[0]) != 192.0f || cfloat_real(b[0]) != 193.0f ||
        cfloat_real(x[0]) != 194.0f || rcond != 0.95f || ferr[0] != 1.7f ||
        berr[0] != 0.9f) {
        fprintf(stderr, "[FAIL] CPBSVX CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] CPBSVX CBLAS->Fortran thunk forwards the complex pointer ABI into the expert-driver C entry\n");
    return 0;
}

static int check_dpbsvx_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_dpbsvx_fn thunk = NULL;
    double ab[8] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    double afb[8] = { 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0 };
    double b[8] = { 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0 };
    double x[8] = { 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0 };
    double s[4] = { 0.0, 0.0, 0.0, 0.0 };
    char equed = 'N';
    double rcond = 0.0;
    double ferr[2] = { 0.0, 0.0 };
    double berr[2] = { 0.0, 0.0 };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dpbsvx_fortran_call, 0, sizeof(g_dpbsvx_fortran_call));

    vtable.ext_ops[FB_OP_DPBSVX][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_dpbsvx_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_DPBSVX);

    thunk = (fb_dpbsvx_fn)vtable.ext_ops[FB_OP_DPBSVX][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DPBSVX Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 'N', FB_UPPER, 4, 1, 2,
                 ab, 4, afb, 4, &equed, s, b, 2, x, 2, &rcond, ferr, berr);
    if (info != 0 || g_dpbsvx_fortran_call.calls != 1 ||
        g_dpbsvx_fortran_call.fact != 'N' || g_dpbsvx_fortran_call.uplo != 'U' ||
        !g_dpbsvx_fortran_call.work_seen || !g_dpbsvx_fortran_call.aux_seen ||
        rcond != 0.55 || ferr[0] != 0.65 || berr[0] != 0.075) {
        fprintf(stderr, "[FAIL] DPBSVX Fortran->CBLAS thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] DPBSVX Fortran->CBLAS thunk forwards expert-driver arguments and propagates outputs\n");
    return 0;
}

static int check_dpbsvx_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_dpbsvx_fortran_slot_fn thunk = NULL;
    double ab[6] = { 0.0 };
    double afb[6] = { 0.0 };
    double b[6] = { 0.0 };
    double x[6] = { 0.0 };
    double s[3] = { 0.0, 0.0, 0.0 };
    char fact = 'F';
    char uplo = 'L';
    char equed = 'N';
    int n = 3;
    int kd = 1;
    int nrhs = 2;
    int ldab = 2;
    int ldafb = 2;
    int ldb = 3;
    int ldx = 3;
    double rcond = 0.0;
    double ferr[2] = { 0.0, 0.0 };
    double berr[2] = { 0.0, 0.0 };
    double work[9] = { 0.0 };
    int iwork[3] = { 0, 0, 0 };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dpbsvx_cblas_call, 0, sizeof(g_dpbsvx_cblas_call));

    vtable.ext_ops[FB_OP_DPBSVX][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_dpbsvx_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_DPBSVX);

    thunk = (fb_dpbsvx_fortran_slot_fn)vtable.ext_ops[FB_OP_DPBSVX][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DPBSVX CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&fact, &uplo, &n, &kd, &nrhs, ab, &ldab, afb, &ldafb, &equed, s, b,
          &ldb, x, &ldx, &rcond, ferr, berr, work, iwork, &info);
    if (info != 192 || g_dpbsvx_cblas_call.called != 1 ||
        g_dpbsvx_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_dpbsvx_cblas_call.fact != 'F' || g_dpbsvx_cblas_call.uplo != FB_LOWER ||
        equed != 'N' || s[0] != 8.0 || ab[0] != 891.0 || afb[0] != 892.0 ||
        b[0] != 893.0 || x[0] != 894.0 || rcond != 0.85 || ferr[0] != 1.85 ||
        berr[0] != 0.95) {
        fprintf(stderr, "[FAIL] DPBSVX CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] DPBSVX CBLAS->Fortran thunk maps all-pointer ABI into the double expert-driver entry\n");
    return 0;
}

static int check_zpbsvx_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_zpbsvx_fn thunk = NULL;
    fb_complex_double_t ab[8] = { 0 };
    fb_complex_double_t afb[8] = { 0 };
    fb_complex_double_t b[8] = { 0 };
    fb_complex_double_t x[8] = { 0 };
    double s[4] = { 0.0, 0.0, 0.0, 0.0 };
    char equed = 'N';
    double rcond = 0.0;
    double ferr[2] = { 0.0, 0.0 };
    double berr[2] = { 0.0, 0.0 };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zpbsvx_fortran_call, 0, sizeof(g_zpbsvx_fortran_call));

    vtable.ext_ops[FB_OP_ZPBSVX][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_zpbsvx_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_ZPBSVX);

    thunk = (fb_zpbsvx_fn)vtable.ext_ops[FB_OP_ZPBSVX][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZPBSVX Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 'E', FB_UPPER, 4, 1, 2,
                 ab, 4, afb, 4, &equed, s, b, 2, x, 2, &rcond, ferr, berr);
    if (info != 0 || g_zpbsvx_fortran_call.calls != 1 ||
        g_zpbsvx_fortran_call.fact != 'E' || g_zpbsvx_fortran_call.uplo != 'U' ||
        !g_zpbsvx_fortran_call.work_seen || !g_zpbsvx_fortran_call.aux_seen ||
        rcond != 0.95 || ferr[0] != 1.05 || berr[0] != 0.105) {
        fprintf(stderr, "[FAIL] ZPBSVX Fortran->CBLAS thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] ZPBSVX Fortran->CBLAS thunk forwards complex-double expert-driver arguments and propagates outputs\n");
    return 0;
}

static int check_zpbsvx_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_zpbsvx_fortran_slot_fn thunk = NULL;
    fb_complex_double_t ab[6] = { 0 };
    fb_complex_double_t afb[6] = { 0 };
    fb_complex_double_t b[6] = { 0 };
    fb_complex_double_t x[6] = { 0 };
    double s[3] = { 0.0, 0.0, 0.0 };
    char fact = 'N';
    char uplo = 'U';
    char equed = 'Y';
    int n = 3;
    int kd = 1;
    int nrhs = 2;
    int ldab = 2;
    int ldafb = 2;
    int ldb = 3;
    int ldx = 3;
    double rcond = 0.0;
    double ferr[2] = { 0.0, 0.0 };
    double berr[2] = { 0.0, 0.0 };
    fb_complex_double_t work[6] = { 0 };
    double rwork[3] = { 0.0, 0.0, 0.0 };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zpbsvx_cblas_call, 0, sizeof(g_zpbsvx_cblas_call));

    vtable.ext_ops[FB_OP_ZPBSVX][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_zpbsvx_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_ZPBSVX);

    thunk = (fb_zpbsvx_fortran_slot_fn)vtable.ext_ops[FB_OP_ZPBSVX][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZPBSVX CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&fact, &uplo, &n, &kd, &nrhs, ab, &ldab, afb, &ldafb, &equed, s, b,
          &ldb, x, &ldx, &rcond, ferr, berr, work, rwork, &info);
    if (info != 194 || g_zpbsvx_cblas_call.called != 1 ||
        g_zpbsvx_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_zpbsvx_cblas_call.fact != 'N' || g_zpbsvx_cblas_call.uplo != FB_UPPER ||
        equed != 'N' || s[0] != 10.0 || cdouble_real(ab[0]) != 1091.0 ||
        cdouble_real(afb[0]) != 1092.0 || cdouble_real(b[0]) != 1093.0 ||
        cdouble_real(x[0]) != 1094.0 || rcond != 1.15 || ferr[0] != 2.15 ||
        berr[0] != 1.15) {
        fprintf(stderr, "[FAIL] ZPBSVX CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] ZPBSVX CBLAS->Fortran thunk maps all-pointer ABI into the complex-double expert-driver entry\n");
    return 0;
}

int main(void)
{
    if (check_spbsvx_fortran_to_cblas_fact_n() != 0) {
        return 1;
    }
    if (check_spbsvx_fortran_to_cblas_fact_e() != 0) {
        return 1;
    }
    if (check_spbsvx_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_dpbsvx_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_dpbsvx_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_cpbsvx_fortran_to_cblas_fact_f() != 0) {
        return 1;
    }
    if (check_cpbsvx_fortran_to_cblas_fact_e() != 0) {
        return 1;
    }
    if (check_cpbsvx_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_zpbsvx_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_zpbsvx_cblas_to_fortran() != 0) {
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}