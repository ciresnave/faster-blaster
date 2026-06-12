#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef int (*fb_ssyev_fn)(fb_layout_t layout, char jobz, fb_uplo_t uplo, int n,
                           float *a, int lda, float *w);
typedef int (*fb_dsyev_fn)(fb_layout_t layout, char jobz, fb_uplo_t uplo, int n,
                           double *a, int lda, double *w);
typedef int (*fb_cheev_fn)(fb_layout_t layout, char jobz, fb_uplo_t uplo, int n,
                           fb_complex_float_t *a, int lda, float *w);
typedef int (*fb_zheev_fn)(fb_layout_t layout, char jobz, fb_uplo_t uplo, int n,
                           fb_complex_double_t *a, int lda, double *w);
typedef int (*fb_ssyevr_fn)(fb_layout_t layout, char jobz, char range,
                            fb_uplo_t uplo, int n, float *a, int lda, float vl,
                            float vu, int il, int iu, float abstol, int *m,
                            float *w, float *z, int ldz, int *isuppz);
typedef int (*fb_dsyevr_fn)(fb_layout_t layout, char jobz, char range,
                            fb_uplo_t uplo, int n, double *a, int lda,
                            double vl, double vu, int il, int iu,
                            double abstol, int *m, double *w, double *z,
                            int ldz, int *isuppz);
typedef int (*fb_cheevr_fn)(fb_layout_t layout, char jobz, char range,
                            fb_uplo_t uplo, int n, fb_complex_float_t *a,
                            int lda, float vl, float vu, int il, int iu,
                            float abstol, int *m, float *w,
                            fb_complex_float_t *z, int ldz, int *isuppz);
typedef int (*fb_zheevr_fn)(fb_layout_t layout, char jobz, char range,
                            fb_uplo_t uplo, int n, fb_complex_double_t *a,
                            int lda, double vl, double vu, int il, int iu,
                            double abstol, int *m, double *w,
                            fb_complex_double_t *z, int ldz, int *isuppz);
typedef int (*fb_ssyevx_fn)(fb_layout_t layout, char jobz, char range,
                            fb_uplo_t uplo, int n, float *a, int lda, float vl,
                            float vu, int il, int iu, float abstol, int *m,
                            float *w, float *z, int ldz, int *ifail);
typedef int (*fb_dsyevx_fn)(fb_layout_t layout, char jobz, char range,
                            fb_uplo_t uplo, int n, double *a, int lda,
                            double vl, double vu, int il, int iu,
                            double abstol, int *m, double *w, double *z,
                            int ldz, int *ifail);
typedef int (*fb_cheevx_fn)(fb_layout_t layout, char jobz, char range,
                            fb_uplo_t uplo, int n, fb_complex_float_t *a,
                            int lda, float vl, float vu, int il, int iu,
                            float abstol, int *m, float *w,
                            fb_complex_float_t *z, int ldz, int *ifail);
typedef int (*fb_zheevx_fn)(fb_layout_t layout, char jobz, char range,
                            fb_uplo_t uplo, int n, fb_complex_double_t *a,
                            int lda, double vl, double vu, int il, int iu,
                            double abstol, int *m, double *w,
                            fb_complex_double_t *z, int ldz, int *ifail);

typedef void (*fb_ssyev_fortran_slot_fn)(char *jobz, char *uplo, int *n,
                                         float *a, int *lda, float *w,
                                         float *work, int *lwork, int *info);
typedef void (*fb_dsyev_fortran_slot_fn)(char *jobz, char *uplo, int *n,
                                         double *a, int *lda, double *w,
                                         double *work, int *lwork, int *info);
typedef void (*fb_cheev_fortran_slot_fn)(char *jobz, char *uplo, int *n,
                                         fb_complex_float_t *a, int *lda,
                                         float *w, fb_complex_float_t *work,
                                         int *lwork, float *rwork, int *info);
typedef void (*fb_zheev_fortran_slot_fn)(char *jobz, char *uplo, int *n,
                                         fb_complex_double_t *a, int *lda,
                                         double *w, fb_complex_double_t *work,
                                         int *lwork, double *rwork, int *info);
typedef void (*fb_ssyevr_fortran_slot_fn)(char *jobz, char *range, char *uplo,
                                          int *n, float *a, int *lda,
                                          float *vl, float *vu, int *il,
                                          int *iu, float *abstol, int *m,
                                          float *w, float *z, int *ldz,
                                          int *isuppz, float *work,
                                          int *lwork, int *iwork,
                                          int *liwork, int *info);
typedef void (*fb_dsyevr_fortran_slot_fn)(char *jobz, char *range, char *uplo,
                                          int *n, double *a, int *lda,
                                          double *vl, double *vu, int *il,
                                          int *iu, double *abstol, int *m,
                                          double *w, double *z, int *ldz,
                                          int *isuppz, double *work,
                                          int *lwork, int *iwork,
                                          int *liwork, int *info);
typedef void (*fb_cheevr_fortran_slot_fn)(char *jobz, char *range, char *uplo,
                                          int *n, fb_complex_float_t *a,
                                          int *lda, float *vl, float *vu,
                                          int *il, int *iu, float *abstol,
                                          int *m, float *w,
                                          fb_complex_float_t *z, int *ldz,
                                          int *isuppz,
                                          fb_complex_float_t *work, int *lwork,
                                          float *rwork, int *lrwork,
                                          int *iwork, int *liwork, int *info);
typedef void (*fb_zheevr_fortran_slot_fn)(char *jobz, char *range, char *uplo,
                                          int *n, fb_complex_double_t *a,
                                          int *lda, double *vl, double *vu,
                                          int *il, int *iu, double *abstol,
                                          int *m, double *w,
                                          fb_complex_double_t *z, int *ldz,
                                          int *isuppz,
                                          fb_complex_double_t *work,
                                          int *lwork, double *rwork,
                                          int *lrwork, int *iwork,
                                          int *liwork, int *info);
typedef void (*fb_ssyevx_fortran_slot_fn)(char *jobz, char *range, char *uplo,
                                          int *n, float *a, int *lda,
                                          float *vl, float *vu, int *il,
                                          int *iu, float *abstol, int *m,
                                          float *w, float *z, int *ldz,
                                          float *work, int *lwork, int *iwork,
                                          int *ifail, int *info);
typedef void (*fb_dsyevx_fortran_slot_fn)(char *jobz, char *range, char *uplo,
                                          int *n, double *a, int *lda,
                                          double *vl, double *vu, int *il,
                                          int *iu, double *abstol, int *m,
                                          double *w, double *z, int *ldz,
                                          double *work, int *lwork, int *iwork,
                                          int *ifail, int *info);
typedef void (*fb_cheevx_fortran_slot_fn)(char *jobz, char *range, char *uplo,
                                          int *n, fb_complex_float_t *a,
                                          int *lda, float *vl, float *vu,
                                          int *il, int *iu, float *abstol,
                                          int *m, float *w,
                                          fb_complex_float_t *z, int *ldz,
                                          fb_complex_float_t *work, int *lwork,
                                          float *rwork, int *iwork,
                                          int *ifail, int *info);
typedef void (*fb_zheevx_fortran_slot_fn)(char *jobz, char *range, char *uplo,
                                          int *n, fb_complex_double_t *a,
                                          int *lda, double *vl, double *vu,
                                          int *il, int *iu, double *abstol,
                                          int *m, double *w,
                                          fb_complex_double_t *z, int *ldz,
                                          fb_complex_double_t *work,
                                          int *lwork, double *rwork,
                                          int *iwork, int *ifail, int *info);

static struct {
    int query_calls;
    int exec_calls;
    int query_lwork;
    int exec_lwork;
} g_ssyev_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    char jobz;
    fb_uplo_t uplo;
    int n;
    int lda;
} g_ssyev_cblas_call;

static struct {
    int query_calls;
    int exec_calls;
    int query_lwork;
    int exec_lwork;
} g_dsyev_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    char jobz;
    fb_uplo_t uplo;
    int n;
    int lda;
} g_dsyev_cblas_call;

static struct {
    int query_calls;
    int exec_calls;
    int query_lwork;
    int exec_lwork;
} g_cheev_fortran_call;

static struct {
    int query_calls;
    int exec_calls;
    int query_lwork;
    int exec_lwork;
} g_zheev_fortran_call;

static struct {
    int query_calls;
    int exec_calls;
    int query_lwork;
    int exec_lwork;
    int exec_liwork;
} g_ssyevr_fortran_call;

static struct {
    int query_calls;
    int exec_calls;
    int query_lwork;
    int exec_lwork;
    int exec_liwork;
} g_dsyevr_fortran_call;

static struct {
    int query_calls;
    int exec_calls;
    int query_lwork;
    int exec_lwork;
    int exec_lrwork;
    int exec_liwork;
} g_cheevr_fortran_call;

static struct {
    int query_calls;
    int exec_calls;
    int query_lwork;
    int exec_lwork;
    int exec_lrwork;
    int exec_liwork;
} g_zheevr_fortran_call;

static struct {
    int query_calls;
    int exec_calls;
    int query_lwork;
    int exec_lwork;
} g_ssyevx_fortran_call;

static struct {
    int query_calls;
    int exec_calls;
    int query_lwork;
    int exec_lwork;
} g_dsyevx_fortran_call;

static struct {
    int query_calls;
    int exec_calls;
    int query_lwork;
    int exec_lwork;
} g_cheevx_fortran_call;

static struct {
    int query_calls;
    int exec_calls;
    int query_lwork;
    int exec_lwork;
} g_zheevx_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    char jobz;
    fb_uplo_t uplo;
    int n;
    int lda;
} g_cheev_cblas_call;

static struct {
    int called;
    fb_layout_t layout;
    char jobz;
    char range;
    fb_uplo_t uplo;
    int n;
    int lda;
    float vl;
    float vu;
    int il;
    int iu;
    float abstol;
    int ldz;
} g_ssyevr_cblas_call;

static struct {
    int called;
    fb_layout_t layout;
    char jobz;
    char range;
    fb_uplo_t uplo;
    int n;
    int lda;
    double vl;
    double vu;
    int il;
    int iu;
    double abstol;
    int ldz;
} g_dsyevr_cblas_call;

static struct {
    int called;
    fb_layout_t layout;
    char jobz;
    char range;
    fb_uplo_t uplo;
    int n;
    int lda;
    float vl;
    float vu;
    int il;
    int iu;
    float abstol;
    int ldz;
} g_cheevr_cblas_call;

static struct {
    int called;
    fb_layout_t layout;
    char jobz;
    char range;
    fb_uplo_t uplo;
    int n;
    int lda;
    double vl;
    double vu;
    int il;
    int iu;
    double abstol;
    int ldz;
} g_zheevr_cblas_call;

static struct {
    int called;
    fb_layout_t layout;
    char jobz;
    char range;
    fb_uplo_t uplo;
    int n;
    int lda;
    float vl;
    float vu;
    int il;
    int iu;
    float abstol;
    int ldz;
} g_ssyevx_cblas_call;

static struct {
    int called;
    fb_layout_t layout;
    char jobz;
    char range;
    fb_uplo_t uplo;
    int n;
    int lda;
    double vl;
    double vu;
    int il;
    int iu;
    double abstol;
    int ldz;
} g_dsyevx_cblas_call;

static struct {
    int called;
    fb_layout_t layout;
    char jobz;
    char range;
    fb_uplo_t uplo;
    int n;
    int lda;
    float vl;
    float vu;
    int il;
    int iu;
    float abstol;
    int ldz;
} g_cheevx_cblas_call;

static struct {
    int called;
    fb_layout_t layout;
    char jobz;
    char range;
    fb_uplo_t uplo;
    int n;
    int lda;
    double vl;
    double vu;
    int il;
    int iu;
    double abstol;
    int ldz;
} g_zheevx_cblas_call;

static struct {
    int called;
    fb_layout_t layout;
    char jobz;
    fb_uplo_t uplo;
    int n;
    int lda;
} g_zheev_cblas_call;

static int g_ssyev_cblas_rc = 0;
static int g_dsyev_cblas_rc = 0;
static int g_cheev_cblas_rc = 0;
static int g_zheev_cblas_rc = 0;
static int g_ssyevr_cblas_rc = 0;
static int g_dsyevr_cblas_rc = 0;
static int g_cheevr_cblas_rc = 0;
static int g_zheevr_cblas_rc = 0;
static int g_ssyevx_cblas_rc = 0;
static int g_dsyevx_cblas_rc = 0;
static int g_cheevx_cblas_rc = 0;
static int g_zheevx_cblas_rc = 0;

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

static void stub_ssyev_fortran(char *jobz, char *uplo, int *n, float *a,
                               int *lda, float *w, float *work, int *lwork,
                               int *info)
{
    (void)jobz;
    (void)uplo;
    (void)n;
    (void)a;
    (void)lda;
    if (*lwork == -1) {
        g_ssyev_fortran_call.query_calls += 1;
        g_ssyev_fortran_call.query_lwork = *lwork;
        work[0] = 31.0f;
        *info = 0;
        return;
    }

    g_ssyev_fortran_call.exec_calls += 1;
    g_ssyev_fortran_call.exec_lwork = *lwork;
    w[0] = 1.0f;
    w[1] = 2.0f;
    w[2] = 3.0f;
    *info = 0;
}

static int stub_ssyev_cblas(fb_layout_t layout, char jobz, fb_uplo_t uplo, int n,
                            float *a, int lda, float *w)
{
    (void)a;
    (void)w;
    g_ssyev_cblas_call.called += 1;
    g_ssyev_cblas_call.layout = layout;
    g_ssyev_cblas_call.jobz = jobz;
    g_ssyev_cblas_call.uplo = uplo;
    g_ssyev_cblas_call.n = n;
    g_ssyev_cblas_call.lda = lda;
    return g_ssyev_cblas_rc;
}

static void stub_dsyev_fortran(char *jobz, char *uplo, int *n, double *a,
                               int *lda, double *w, double *work, int *lwork,
                               int *info)
{
    (void)jobz;
    (void)uplo;
    (void)n;
    (void)a;
    (void)lda;
    if (*lwork == -1) {
        g_dsyev_fortran_call.query_calls += 1;
        g_dsyev_fortran_call.query_lwork = *lwork;
        work[0] = 37.0;
        *info = 0;
        return;
    }

    g_dsyev_fortran_call.exec_calls += 1;
    g_dsyev_fortran_call.exec_lwork = *lwork;
    w[0] = 1.5;
    w[1] = 2.5;
    w[2] = 3.5;
    *info = 0;
}

static int stub_dsyev_cblas(fb_layout_t layout, char jobz, fb_uplo_t uplo, int n,
                            double *a, int lda, double *w)
{
    (void)a;
    (void)w;
    g_dsyev_cblas_call.called += 1;
    g_dsyev_cblas_call.layout = layout;
    g_dsyev_cblas_call.jobz = jobz;
    g_dsyev_cblas_call.uplo = uplo;
    g_dsyev_cblas_call.n = n;
    g_dsyev_cblas_call.lda = lda;
    return g_dsyev_cblas_rc;
}

static void stub_cheev_fortran(char *jobz, char *uplo, int *n,
                               fb_complex_float_t *a, int *lda, float *w,
                               fb_complex_float_t *work, int *lwork,
                               float *rwork, int *info)
{
    (void)jobz;
    (void)uplo;
    (void)n;
    (void)a;
    (void)lda;
    (void)rwork;
    if (*lwork == -1) {
        g_cheev_fortran_call.query_calls += 1;
        g_cheev_fortran_call.query_lwork = *lwork;
        work[0] = make_cfloat(32.0f);
        *info = 0;
        return;
    }

    g_cheev_fortran_call.exec_calls += 1;
    g_cheev_fortran_call.exec_lwork = *lwork;
    w[0] = 4.0f;
    w[1] = 5.0f;
    w[2] = 6.0f;
    *info = 0;
}

static int stub_cheev_cblas(fb_layout_t layout, char jobz, fb_uplo_t uplo, int n,
                            fb_complex_float_t *a, int lda, float *w)
{
    (void)a;
    (void)w;
    g_cheev_cblas_call.called += 1;
    g_cheev_cblas_call.layout = layout;
    g_cheev_cblas_call.jobz = jobz;
    g_cheev_cblas_call.uplo = uplo;
    g_cheev_cblas_call.n = n;
    g_cheev_cblas_call.lda = lda;
    return g_cheev_cblas_rc;
}

static void stub_zheev_fortran(char *jobz, char *uplo, int *n,
                               fb_complex_double_t *a, int *lda, double *w,
                               fb_complex_double_t *work, int *lwork,
                               double *rwork, int *info)
{
    (void)jobz;
    (void)uplo;
    (void)n;
    (void)a;
    (void)lda;
    (void)rwork;
    if (*lwork == -1) {
        g_zheev_fortran_call.query_calls += 1;
        g_zheev_fortran_call.query_lwork = *lwork;
        work[0] = make_cdouble(38.0);
        *info = 0;
        return;
    }

    g_zheev_fortran_call.exec_calls += 1;
    g_zheev_fortran_call.exec_lwork = *lwork;
    w[0] = 4.5;
    w[1] = 5.5;
    w[2] = 6.5;
    *info = 0;
}

static int stub_zheev_cblas(fb_layout_t layout, char jobz, fb_uplo_t uplo, int n,
                            fb_complex_double_t *a, int lda, double *w)
{
    (void)a;
    (void)w;
    g_zheev_cblas_call.called += 1;
    g_zheev_cblas_call.layout = layout;
    g_zheev_cblas_call.jobz = jobz;
    g_zheev_cblas_call.uplo = uplo;
    g_zheev_cblas_call.n = n;
    g_zheev_cblas_call.lda = lda;
    return g_zheev_cblas_rc;
}

static void stub_ssyevr_fortran(char *jobz, char *range, char *uplo, int *n,
                                float *a, int *lda, float *vl, float *vu,
                                int *il, int *iu, float *abstol, int *m,
                                float *w, float *z, int *ldz, int *isuppz,
                                float *work, int *lwork, int *iwork,
                                int *liwork, int *info)
{
    (void)jobz;
    (void)range;
    (void)uplo;
    (void)n;
    (void)a;
    (void)lda;
    (void)vl;
    (void)vu;
    (void)il;
    (void)iu;
    (void)abstol;
    (void)ldz;
    if (*lwork == -1 || *liwork == -1) {
        g_ssyevr_fortran_call.query_calls += 1;
        g_ssyevr_fortran_call.query_lwork = *lwork;
        work[0] = 33.0f;
        *iwork = 17;
        *info = 0;
        return;
    }

    g_ssyevr_fortran_call.exec_calls += 1;
    g_ssyevr_fortran_call.exec_lwork = *lwork;
    g_ssyevr_fortran_call.exec_liwork = *liwork;
    *m = 2;
    w[0] = 7.0f;
    w[1] = 8.0f;
    z[0] = 41.0f;
    isuppz[0] = 1;
    isuppz[1] = 3;
    *info = 0;
}

static int stub_ssyevr_cblas(fb_layout_t layout, char jobz, char range,
                             fb_uplo_t uplo, int n, float *a, int lda,
                             float vl, float vu, int il, int iu, float abstol,
                             int *m, float *w, float *z, int ldz, int *isuppz)
{
    (void)a;
    g_ssyevr_cblas_call.called += 1;
    g_ssyevr_cblas_call.layout = layout;
    g_ssyevr_cblas_call.jobz = jobz;
    g_ssyevr_cblas_call.range = range;
    g_ssyevr_cblas_call.uplo = uplo;
    g_ssyevr_cblas_call.n = n;
    g_ssyevr_cblas_call.lda = lda;
    g_ssyevr_cblas_call.vl = vl;
    g_ssyevr_cblas_call.vu = vu;
    g_ssyevr_cblas_call.il = il;
    g_ssyevr_cblas_call.iu = iu;
    g_ssyevr_cblas_call.abstol = abstol;
    g_ssyevr_cblas_call.ldz = ldz;
    *m = 2;
    w[0] = 11.0f;
    w[1] = 12.0f;
    z[0] = 43.0f;
    isuppz[0] = 1;
    isuppz[1] = 2;
    return g_ssyevr_cblas_rc;
}

static void stub_dsyevr_fortran(char *jobz, char *range, char *uplo, int *n,
                                double *a, int *lda, double *vl, double *vu,
                                int *il, int *iu, double *abstol, int *m,
                                double *w, double *z, int *ldz, int *isuppz,
                                double *work, int *lwork, int *iwork,
                                int *liwork, int *info)
{
    (void)jobz;
    (void)range;
    (void)uplo;
    (void)n;
    (void)a;
    (void)lda;
    (void)vl;
    (void)vu;
    (void)il;
    (void)iu;
    (void)abstol;
    (void)ldz;
    if (*lwork == -1 || *liwork == -1) {
        g_dsyevr_fortran_call.query_calls += 1;
        g_dsyevr_fortran_call.query_lwork = *lwork;
        work[0] = 53.0;
        *iwork = 27;
        *info = 0;
        return;
    }

    g_dsyevr_fortran_call.exec_calls += 1;
    g_dsyevr_fortran_call.exec_lwork = *lwork;
    g_dsyevr_fortran_call.exec_liwork = *liwork;
    *m = 2;
    w[0] = 17.0;
    w[1] = 18.0;
    z[0] = 141.0;
    isuppz[0] = 1;
    isuppz[1] = 3;
    *info = 0;
}

static int stub_dsyevr_cblas(fb_layout_t layout, char jobz, char range,
                             fb_uplo_t uplo, int n, double *a, int lda,
                             double vl, double vu, int il, int iu,
                             double abstol, int *m, double *w, double *z,
                             int ldz, int *isuppz)
{
    (void)a;
    g_dsyevr_cblas_call.called += 1;
    g_dsyevr_cblas_call.layout = layout;
    g_dsyevr_cblas_call.jobz = jobz;
    g_dsyevr_cblas_call.range = range;
    g_dsyevr_cblas_call.uplo = uplo;
    g_dsyevr_cblas_call.n = n;
    g_dsyevr_cblas_call.lda = lda;
    g_dsyevr_cblas_call.vl = vl;
    g_dsyevr_cblas_call.vu = vu;
    g_dsyevr_cblas_call.il = il;
    g_dsyevr_cblas_call.iu = iu;
    g_dsyevr_cblas_call.abstol = abstol;
    g_dsyevr_cblas_call.ldz = ldz;
    *m = 2;
    w[0] = 21.0;
    w[1] = 22.0;
    z[0] = 143.0;
    isuppz[0] = 1;
    isuppz[1] = 2;
    return g_dsyevr_cblas_rc;
}

static void stub_cheevr_fortran(char *jobz, char *range, char *uplo, int *n,
                                fb_complex_float_t *a, int *lda, float *vl,
                                float *vu, int *il, int *iu, float *abstol,
                                int *m, float *w, fb_complex_float_t *z,
                                int *ldz, int *isuppz,
                                fb_complex_float_t *work, int *lwork,
                                float *rwork, int *lrwork, int *iwork,
                                int *liwork, int *info)
{
    (void)jobz;
    (void)range;
    (void)uplo;
    (void)n;
    (void)a;
    (void)lda;
    (void)vl;
    (void)vu;
    (void)il;
    (void)iu;
    (void)abstol;
    (void)ldz;
    if (*lwork == -1 || *lrwork == -1 || *liwork == -1) {
        g_cheevr_fortran_call.query_calls += 1;
        g_cheevr_fortran_call.query_lwork = *lwork;
        work[0] = make_cfloat(34.0f);
        *rwork = 19.0f;
        *iwork = 21;
        *info = 0;
        return;
    }

    g_cheevr_fortran_call.exec_calls += 1;
    g_cheevr_fortran_call.exec_lwork = *lwork;
    g_cheevr_fortran_call.exec_lrwork = *lrwork;
    g_cheevr_fortran_call.exec_liwork = *liwork;
    *m = 2;
    w[0] = 9.0f;
    w[1] = 10.0f;
    z[0] = make_cfloat(42.0f);
    isuppz[0] = 2;
    isuppz[1] = 3;
    *info = 0;
}

static int stub_cheevr_cblas(fb_layout_t layout, char jobz, char range,
                             fb_uplo_t uplo, int n, fb_complex_float_t *a,
                             int lda, float vl, float vu, int il, int iu,
                             float abstol, int *m, float *w,
                             fb_complex_float_t *z, int ldz, int *isuppz)
{
    (void)a;
    g_cheevr_cblas_call.called += 1;
    g_cheevr_cblas_call.layout = layout;
    g_cheevr_cblas_call.jobz = jobz;
    g_cheevr_cblas_call.range = range;
    g_cheevr_cblas_call.uplo = uplo;
    g_cheevr_cblas_call.n = n;
    g_cheevr_cblas_call.lda = lda;
    g_cheevr_cblas_call.vl = vl;
    g_cheevr_cblas_call.vu = vu;
    g_cheevr_cblas_call.il = il;
    g_cheevr_cblas_call.iu = iu;
    g_cheevr_cblas_call.abstol = abstol;
    g_cheevr_cblas_call.ldz = ldz;
    *m = 2;
    w[0] = 13.0f;
    w[1] = 14.0f;
    z[0] = make_cfloat(44.0f);
    isuppz[0] = 2;
    isuppz[1] = 4;
    return g_cheevr_cblas_rc;
}

static void stub_zheevr_fortran(char *jobz, char *range, char *uplo, int *n,
                                fb_complex_double_t *a, int *lda, double *vl,
                                double *vu, int *il, int *iu, double *abstol,
                                int *m, double *w, fb_complex_double_t *z,
                                int *ldz, int *isuppz,
                                fb_complex_double_t *work, int *lwork,
                                double *rwork, int *lrwork, int *iwork,
                                int *liwork, int *info)
{
    (void)jobz;
    (void)range;
    (void)uplo;
    (void)n;
    (void)a;
    (void)lda;
    (void)vl;
    (void)vu;
    (void)il;
    (void)iu;
    (void)abstol;
    (void)ldz;
    if (*lwork == -1 || *lrwork == -1 || *liwork == -1) {
        g_zheevr_fortran_call.query_calls += 1;
        g_zheevr_fortran_call.query_lwork = *lwork;
        work[0] = make_cdouble(54.0);
        *rwork = 29.0;
        *iwork = 31;
        *info = 0;
        return;
    }

    g_zheevr_fortran_call.exec_calls += 1;
    g_zheevr_fortran_call.exec_lwork = *lwork;
    g_zheevr_fortran_call.exec_lrwork = *lrwork;
    g_zheevr_fortran_call.exec_liwork = *liwork;
    *m = 2;
    w[0] = 19.0;
    w[1] = 20.0;
    z[0] = make_cdouble(142.0);
    isuppz[0] = 2;
    isuppz[1] = 3;
    *info = 0;
}

static int stub_zheevr_cblas(fb_layout_t layout, char jobz, char range,
                             fb_uplo_t uplo, int n, fb_complex_double_t *a,
                             int lda, double vl, double vu, int il, int iu,
                             double abstol, int *m, double *w,
                             fb_complex_double_t *z, int ldz, int *isuppz)
{
    (void)a;
    g_zheevr_cblas_call.called += 1;
    g_zheevr_cblas_call.layout = layout;
    g_zheevr_cblas_call.jobz = jobz;
    g_zheevr_cblas_call.range = range;
    g_zheevr_cblas_call.uplo = uplo;
    g_zheevr_cblas_call.n = n;
    g_zheevr_cblas_call.lda = lda;
    g_zheevr_cblas_call.vl = vl;
    g_zheevr_cblas_call.vu = vu;
    g_zheevr_cblas_call.il = il;
    g_zheevr_cblas_call.iu = iu;
    g_zheevr_cblas_call.abstol = abstol;
    g_zheevr_cblas_call.ldz = ldz;
    *m = 2;
    w[0] = 23.0;
    w[1] = 24.0;
    z[0] = make_cdouble(144.0);
    isuppz[0] = 2;
    isuppz[1] = 4;
    return g_zheevr_cblas_rc;
}

static void stub_ssyevx_fortran(char *jobz, char *range, char *uplo, int *n,
                                float *a, int *lda, float *vl, float *vu,
                                int *il, int *iu, float *abstol, int *m,
                                float *w, float *z, int *ldz, float *work,
                                int *lwork, int *iwork, int *ifail, int *info)
{
    (void)jobz;
    (void)range;
    (void)uplo;
    (void)n;
    (void)a;
    (void)lda;
    (void)vl;
    (void)vu;
    (void)il;
    (void)iu;
    (void)abstol;
    (void)ldz;
    (void)iwork;
    if (*lwork == -1) {
        g_ssyevx_fortran_call.query_calls += 1;
        g_ssyevx_fortran_call.query_lwork = *lwork;
        work[0] = 35.0f;
        *info = 0;
        return;
    }

    g_ssyevx_fortran_call.exec_calls += 1;
    g_ssyevx_fortran_call.exec_lwork = *lwork;
    *m = 2;
    w[0] = 7.0f;
    w[1] = 8.0f;
    z[0] = 105.0f;
    ifail[0] = 1;
    ifail[1] = 0;
    *info = 0;
}

static int stub_ssyevx_cblas(fb_layout_t layout, char jobz, char range,
                             fb_uplo_t uplo, int n, float *a, int lda,
                             float vl, float vu, int il, int iu, float abstol,
                             int *m, float *w, float *z, int ldz, int *ifail)
{
    (void)a;
    g_ssyevx_cblas_call.called += 1;
    g_ssyevx_cblas_call.layout = layout;
    g_ssyevx_cblas_call.jobz = jobz;
    g_ssyevx_cblas_call.range = range;
    g_ssyevx_cblas_call.uplo = uplo;
    g_ssyevx_cblas_call.n = n;
    g_ssyevx_cblas_call.lda = lda;
    g_ssyevx_cblas_call.vl = vl;
    g_ssyevx_cblas_call.vu = vu;
    g_ssyevx_cblas_call.il = il;
    g_ssyevx_cblas_call.iu = iu;
    g_ssyevx_cblas_call.abstol = abstol;
    g_ssyevx_cblas_call.ldz = ldz;
    *m = 2;
    w[0] = 11.0f;
    w[1] = 12.0f;
    z[0] = 107.0f;
    ifail[0] = 3;
    ifail[1] = 0;
    return g_ssyevx_cblas_rc;
}

static void stub_dsyevx_fortran(char *jobz, char *range, char *uplo, int *n,
                                double *a, int *lda, double *vl, double *vu,
                                int *il, int *iu, double *abstol, int *m,
                                double *w, double *z, int *ldz, double *work,
                                int *lwork, int *iwork, int *ifail, int *info)
{
    (void)jobz;
    (void)range;
    (void)uplo;
    (void)n;
    (void)a;
    (void)lda;
    (void)vl;
    (void)vu;
    (void)il;
    (void)iu;
    (void)abstol;
    (void)ldz;
    (void)iwork;
    if (*lwork == -1) {
        g_dsyevx_fortran_call.query_calls += 1;
        g_dsyevx_fortran_call.query_lwork = *lwork;
        work[0] = 55.0;
        *info = 0;
        return;
    }

    g_dsyevx_fortran_call.exec_calls += 1;
    g_dsyevx_fortran_call.exec_lwork = *lwork;
    *m = 2;
    w[0] = 17.0;
    w[1] = 18.0;
    z[0] = 205.0;
    ifail[0] = 1;
    ifail[1] = 0;
    *info = 0;
}

static int stub_dsyevx_cblas(fb_layout_t layout, char jobz, char range,
                             fb_uplo_t uplo, int n, double *a, int lda,
                             double vl, double vu, int il, int iu,
                             double abstol, int *m, double *w, double *z,
                             int ldz, int *ifail)
{
    (void)a;
    g_dsyevx_cblas_call.called += 1;
    g_dsyevx_cblas_call.layout = layout;
    g_dsyevx_cblas_call.jobz = jobz;
    g_dsyevx_cblas_call.range = range;
    g_dsyevx_cblas_call.uplo = uplo;
    g_dsyevx_cblas_call.n = n;
    g_dsyevx_cblas_call.lda = lda;
    g_dsyevx_cblas_call.vl = vl;
    g_dsyevx_cblas_call.vu = vu;
    g_dsyevx_cblas_call.il = il;
    g_dsyevx_cblas_call.iu = iu;
    g_dsyevx_cblas_call.abstol = abstol;
    g_dsyevx_cblas_call.ldz = ldz;
    *m = 2;
    w[0] = 21.0;
    w[1] = 22.0;
    z[0] = 207.0;
    ifail[0] = 3;
    ifail[1] = 0;
    return g_dsyevx_cblas_rc;
}

static void stub_cheevx_fortran(char *jobz, char *range, char *uplo, int *n,
                                fb_complex_float_t *a, int *lda, float *vl,
                                float *vu, int *il, int *iu, float *abstol,
                                int *m, float *w, fb_complex_float_t *z,
                                int *ldz, fb_complex_float_t *work,
                                int *lwork, float *rwork, int *iwork,
                                int *ifail, int *info)
{
    (void)jobz;
    (void)range;
    (void)uplo;
    (void)n;
    (void)a;
    (void)lda;
    (void)vl;
    (void)vu;
    (void)il;
    (void)iu;
    (void)abstol;
    (void)ldz;
    (void)rwork;
    (void)iwork;
    if (*lwork == -1) {
        g_cheevx_fortran_call.query_calls += 1;
        g_cheevx_fortran_call.query_lwork = *lwork;
        work[0] = make_cfloat(36.0f);
        *info = 0;
        return;
    }

    g_cheevx_fortran_call.exec_calls += 1;
    g_cheevx_fortran_call.exec_lwork = *lwork;
    *m = 2;
    w[0] = 9.0f;
    w[1] = 10.0f;
    z[0] = make_cfloat(106.0f);
    ifail[0] = 2;
    ifail[1] = 0;
    *info = 0;
}

static int stub_cheevx_cblas(fb_layout_t layout, char jobz, char range,
                             fb_uplo_t uplo, int n, fb_complex_float_t *a,
                             int lda, float vl, float vu, int il, int iu,
                             float abstol, int *m, float *w,
                             fb_complex_float_t *z, int ldz, int *ifail)
{
    (void)a;
    g_cheevx_cblas_call.called += 1;
    g_cheevx_cblas_call.layout = layout;
    g_cheevx_cblas_call.jobz = jobz;
    g_cheevx_cblas_call.range = range;
    g_cheevx_cblas_call.uplo = uplo;
    g_cheevx_cblas_call.n = n;
    g_cheevx_cblas_call.lda = lda;
    g_cheevx_cblas_call.vl = vl;
    g_cheevx_cblas_call.vu = vu;
    g_cheevx_cblas_call.il = il;
    g_cheevx_cblas_call.iu = iu;
    g_cheevx_cblas_call.abstol = abstol;
    g_cheevx_cblas_call.ldz = ldz;
    *m = 2;
    w[0] = 13.0f;
    w[1] = 14.0f;
    z[0] = make_cfloat(108.0f);
    ifail[0] = 4;
    ifail[1] = 0;
    return g_cheevx_cblas_rc;
}

static void stub_zheevx_fortran(char *jobz, char *range, char *uplo, int *n,
                                fb_complex_double_t *a, int *lda, double *vl,
                                double *vu, int *il, int *iu, double *abstol,
                                int *m, double *w, fb_complex_double_t *z,
                                int *ldz, fb_complex_double_t *work,
                                int *lwork, double *rwork, int *iwork,
                                int *ifail, int *info)
{
    (void)jobz;
    (void)range;
    (void)uplo;
    (void)n;
    (void)a;
    (void)lda;
    (void)vl;
    (void)vu;
    (void)il;
    (void)iu;
    (void)abstol;
    (void)ldz;
    (void)rwork;
    (void)iwork;
    if (*lwork == -1) {
        g_zheevx_fortran_call.query_calls += 1;
        g_zheevx_fortran_call.query_lwork = *lwork;
        work[0] = make_cdouble(56.0);
        *info = 0;
        return;
    }

    g_zheevx_fortran_call.exec_calls += 1;
    g_zheevx_fortran_call.exec_lwork = *lwork;
    *m = 2;
    w[0] = 19.0;
    w[1] = 20.0;
    z[0] = make_cdouble(206.0);
    ifail[0] = 2;
    ifail[1] = 0;
    *info = 0;
}

static int stub_zheevx_cblas(fb_layout_t layout, char jobz, char range,
                             fb_uplo_t uplo, int n, fb_complex_double_t *a,
                             int lda, double vl, double vu, int il, int iu,
                             double abstol, int *m, double *w,
                             fb_complex_double_t *z, int ldz, int *ifail)
{
    (void)a;
    g_zheevx_cblas_call.called += 1;
    g_zheevx_cblas_call.layout = layout;
    g_zheevx_cblas_call.jobz = jobz;
    g_zheevx_cblas_call.range = range;
    g_zheevx_cblas_call.uplo = uplo;
    g_zheevx_cblas_call.n = n;
    g_zheevx_cblas_call.lda = lda;
    g_zheevx_cblas_call.vl = vl;
    g_zheevx_cblas_call.vu = vu;
    g_zheevx_cblas_call.il = il;
    g_zheevx_cblas_call.iu = iu;
    g_zheevx_cblas_call.abstol = abstol;
    g_zheevx_cblas_call.ldz = ldz;
    *m = 2;
    w[0] = 23.0;
    w[1] = 24.0;
    z[0] = make_cdouble(208.0);
    ifail[0] = 4;
    ifail[1] = 0;
    return g_zheevx_cblas_rc;
}

static int check_ssyev_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_ssyev_fn thunk = NULL;
    float a[9] = { 0.0f };
    float w[3] = { 0.0f, 0.0f, 0.0f };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_ssyev_fortran_call, 0, sizeof(g_ssyev_fortran_call));

    vtable.ext_ops[FB_OP_SSYEV][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_ssyev_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SSYEV);

    thunk = (fb_ssyev_fn)vtable.ext_ops[FB_OP_SSYEV][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SSYEV Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 'V', FB_LOWER, 3, a, 3, w);
    if (info != 0 || g_ssyev_fortran_call.query_calls != 1 ||
        g_ssyev_fortran_call.exec_calls != 1 ||
        g_ssyev_fortran_call.query_lwork != -1 ||
        g_ssyev_fortran_call.exec_lwork != 31 ||
        w[0] != 1.0f || w[1] != 2.0f || w[2] != 3.0f) {
        fprintf(stderr, "[FAIL] SSYEV Fortran->CBLAS thunk did not preserve symmetric eigenvalue query semantics\n");
        return 1;
    }

    printf("[PASS] SSYEV Fortran->CBLAS thunk performs the workspace query and forwards eigenvalues\n");
    return 0;
}

static int check_ssyev_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_ssyev_fortran_slot_fn thunk = NULL;
    float a[9] = { 0.0f };
    float w[3] = { 0.0f, 0.0f, 0.0f };
    float work[8] = { 0.0f };
    char jobz = 'V';
    char uplo = 'L';
    int n = 3;
    int lda = 3;
    int lwork = 8;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_ssyev_cblas_call, 0, sizeof(g_ssyev_cblas_call));
    g_ssyev_cblas_rc = 231;

    vtable.ext_ops[FB_OP_SSYEV][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_ssyev_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SSYEV);

    thunk = (fb_ssyev_fortran_slot_fn)vtable.ext_ops[FB_OP_SSYEV][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SSYEV CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&jobz, &uplo, &n, a, &lda, w, work, &lwork, &info);
    if (info != 231 || g_ssyev_cblas_call.called != 1 ||
        g_ssyev_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_ssyev_cblas_call.jobz != 'V' || g_ssyev_cblas_call.uplo != FB_LOWER ||
        g_ssyev_cblas_call.n != 3 || g_ssyev_cblas_call.lda != 3) {
        fprintf(stderr, "[FAIL] SSYEV CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] SSYEV CBLAS->Fortran thunk maps Fortran UPLO chars into the generic C symmetric-eigen entry\n");
    return 0;
}

static int check_dsyev_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_dsyev_fn thunk = NULL;
    double a[9] = { 0.0 };
    double w[3] = { 0.0, 0.0, 0.0 };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dsyev_fortran_call, 0, sizeof(g_dsyev_fortran_call));

    vtable.ext_ops[FB_OP_DSYEV][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_dsyev_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_DSYEV);

    thunk = (fb_dsyev_fn)vtable.ext_ops[FB_OP_DSYEV][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DSYEV Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 'V', FB_UPPER, 3, a, 3, w);
    if (info != 0 || g_dsyev_fortran_call.query_calls != 1 ||
        g_dsyev_fortran_call.exec_calls != 1 ||
        g_dsyev_fortran_call.query_lwork != -1 ||
        g_dsyev_fortran_call.exec_lwork != 37 ||
        w[0] != 1.5 || w[1] != 2.5 || w[2] != 3.5) {
        fprintf(stderr, "[FAIL] DSYEV Fortran->CBLAS thunk did not preserve double symmetric eigenvalue query semantics\n");
        return 1;
    }

    printf("[PASS] DSYEV Fortran->CBLAS thunk performs the workspace query and forwards double eigenvalues\n");
    return 0;
}

static int check_dsyev_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_dsyev_fortran_slot_fn thunk = NULL;
    double a[9] = { 0.0 };
    double w[3] = { 0.0, 0.0, 0.0 };
    double work[8] = { 0.0 };
    char jobz = 'V';
    char uplo = 'U';
    int n = 3;
    int lda = 3;
    int lwork = 8;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dsyev_cblas_call, 0, sizeof(g_dsyev_cblas_call));
    g_dsyev_cblas_rc = 232;

    vtable.ext_ops[FB_OP_DSYEV][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_dsyev_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_DSYEV);

    thunk = (fb_dsyev_fortran_slot_fn)vtable.ext_ops[FB_OP_DSYEV][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DSYEV CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&jobz, &uplo, &n, a, &lda, w, work, &lwork, &info);
    if (info != 232 || g_dsyev_cblas_call.called != 1 ||
        g_dsyev_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_dsyev_cblas_call.jobz != 'V' || g_dsyev_cblas_call.uplo != FB_UPPER ||
        g_dsyev_cblas_call.n != 3 || g_dsyev_cblas_call.lda != 3) {
        fprintf(stderr, "[FAIL] DSYEV CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] DSYEV CBLAS->Fortran thunk maps Fortran UPLO chars into the generic C double symmetric-eigen entry\n");
    return 0;
}

static int check_cheev_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_cheev_fn thunk = NULL;
    fb_complex_float_t a[9] = { 0 };
    float w[3] = { 0.0f, 0.0f, 0.0f };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cheev_fortran_call, 0, sizeof(g_cheev_fortran_call));

    vtable.ext_ops[FB_OP_CHEEV][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_cheev_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CHEEV);

    thunk = (fb_cheev_fn)vtable.ext_ops[FB_OP_CHEEV][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CHEEV Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 'V', FB_UPPER, 3, a, 3, w);
    if (info != 0 || g_cheev_fortran_call.query_calls != 1 ||
        g_cheev_fortran_call.exec_calls != 1 ||
        g_cheev_fortran_call.query_lwork != -1 ||
        g_cheev_fortran_call.exec_lwork != 32 ||
        w[0] != 4.0f || w[1] != 5.0f || w[2] != 6.0f) {
        fprintf(stderr, "[FAIL] CHEEV Fortran->CBLAS thunk did not preserve Hermitian eigenvalue query semantics\n");
        return 1;
    }

    printf("[PASS] CHEEV Fortran->CBLAS thunk performs the workspace query and forwards Hermitian eigenvalues\n");
    return 0;
}

static int check_cheev_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_cheev_fortran_slot_fn thunk = NULL;
    fb_complex_float_t a[9] = { 0 };
    float w[3] = { 0.0f, 0.0f, 0.0f };
    fb_complex_float_t work[8] = { 0 };
    float rwork[16] = { 0.0f };
    char jobz = 'V';
    char uplo = 'U';
    int n = 3;
    int lda = 3;
    int lwork = 8;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cheev_cblas_call, 0, sizeof(g_cheev_cblas_call));
    g_cheev_cblas_rc = 233;

    vtable.ext_ops[FB_OP_CHEEV][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_cheev_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CHEEV);

    thunk = (fb_cheev_fortran_slot_fn)vtable.ext_ops[FB_OP_CHEEV][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CHEEV CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&jobz, &uplo, &n, a, &lda, w, work, &lwork, rwork, &info);
    if (info != 233 || g_cheev_cblas_call.called != 1 ||
        g_cheev_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_cheev_cblas_call.jobz != 'V' || g_cheev_cblas_call.uplo != FB_UPPER ||
        g_cheev_cblas_call.n != 3 || g_cheev_cblas_call.lda != 3) {
        fprintf(stderr, "[FAIL] CHEEV CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] CHEEV CBLAS->Fortran thunk maps Fortran UPLO chars into the generic C Hermitian-eigen entry\n");
    return 0;
}

static int check_zheev_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_zheev_fn thunk = NULL;
    fb_complex_double_t a[9] = { 0 };
    double w[3] = { 0.0, 0.0, 0.0 };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zheev_fortran_call, 0, sizeof(g_zheev_fortran_call));

    vtable.ext_ops[FB_OP_ZHEEV][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_zheev_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_ZHEEV);

    thunk = (fb_zheev_fn)vtable.ext_ops[FB_OP_ZHEEV][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZHEEV Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 'V', FB_LOWER, 3, a, 3, w);
    if (info != 0 || g_zheev_fortran_call.query_calls != 1 ||
        g_zheev_fortran_call.exec_calls != 1 ||
        g_zheev_fortran_call.query_lwork != -1 ||
        g_zheev_fortran_call.exec_lwork != 38 ||
        w[0] != 4.5 || w[1] != 5.5 || w[2] != 6.5) {
        fprintf(stderr, "[FAIL] ZHEEV Fortran->CBLAS thunk did not preserve complex-double Hermitian eigenvalue query semantics\n");
        return 1;
    }

    printf("[PASS] ZHEEV Fortran->CBLAS thunk performs the workspace query and forwards complex-double Hermitian eigenvalues\n");
    return 0;
}

static int check_zheev_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_zheev_fortran_slot_fn thunk = NULL;
    fb_complex_double_t a[9] = { 0 };
    double w[3] = { 0.0, 0.0, 0.0 };
    fb_complex_double_t work[8] = { 0 };
    double rwork[16] = { 0.0 };
    char jobz = 'V';
    char uplo = 'L';
    int n = 3;
    int lda = 3;
    int lwork = 8;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zheev_cblas_call, 0, sizeof(g_zheev_cblas_call));
    g_zheev_cblas_rc = 234;

    vtable.ext_ops[FB_OP_ZHEEV][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_zheev_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_ZHEEV);

    thunk = (fb_zheev_fortran_slot_fn)vtable.ext_ops[FB_OP_ZHEEV][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZHEEV CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&jobz, &uplo, &n, a, &lda, w, work, &lwork, rwork, &info);
    if (info != 234 || g_zheev_cblas_call.called != 1 ||
        g_zheev_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_zheev_cblas_call.jobz != 'V' || g_zheev_cblas_call.uplo != FB_LOWER ||
        g_zheev_cblas_call.n != 3 || g_zheev_cblas_call.lda != 3) {
        fprintf(stderr, "[FAIL] ZHEEV CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] ZHEEV CBLAS->Fortran thunk maps Fortran UPLO chars into the generic C complex-double Hermitian-eigen entry\n");
    return 0;
}

static int check_ssyevr_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_ssyevr_fn thunk = NULL;
    float a[9] = { 0.0f };
    float w[3] = { 0.0f, 0.0f, 0.0f };
    float z[9] = { 0.0f };
    int isuppz[6] = { 0, 0, 0, 0, 0, 0 };
    int m = 0;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_ssyevr_fortran_call, 0, sizeof(g_ssyevr_fortran_call));

    vtable.ext_ops[FB_OP_SSYEVR][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_ssyevr_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SSYEVR);

    thunk = (fb_ssyevr_fn)vtable.ext_ops[FB_OP_SSYEVR][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SSYEVR Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 'V', 'I', FB_LOWER, 3, a, 3, 1.0f, 6.0f,
                 1, 2, 0.25f, &m, w, z, 3, isuppz);
    if (info != 0 || g_ssyevr_fortran_call.query_calls != 1 ||
        g_ssyevr_fortran_call.exec_calls != 1 ||
        g_ssyevr_fortran_call.query_lwork != -1 ||
        g_ssyevr_fortran_call.exec_lwork != 33 ||
        g_ssyevr_fortran_call.exec_liwork != 17 || m != 2 ||
        w[0] != 7.0f || w[1] != 8.0f || z[0] != 41.0f ||
        isuppz[0] != 1 || isuppz[1] != 3) {
        fprintf(stderr, "[FAIL] SSYEVR Fortran->CBLAS thunk did not preserve selective symmetric-eigen RRR query semantics\n");
        return 1;
    }

    printf("[PASS] SSYEVR Fortran->CBLAS thunk performs the workspace queries and forwards selective symmetric-eigen RRR outputs\n");
    return 0;
}

static int check_ssyevr_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_ssyevr_fortran_slot_fn thunk = NULL;
    float a[9] = { 0.0f };
    float w[3] = { 0.0f, 0.0f, 0.0f };
    float z[9] = { 0.0f };
    float work[16] = { 0.0f };
    int iwork[16] = { 0 };
    int isuppz[6] = { 0, 0, 0, 0, 0, 0 };
    char jobz = 'V';
    char range = 'I';
    char uplo = 'L';
    int n = 3;
    int lda = 3;
    float vl = 1.0f;
    float vu = 6.0f;
    int il = 1;
    int iu = 2;
    float abstol = 0.25f;
    int m = 0;
    int ldz = 3;
    int lwork = 16;
    int liwork = 16;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_ssyevr_cblas_call, 0, sizeof(g_ssyevr_cblas_call));
    g_ssyevr_cblas_rc = 241;

    vtable.ext_ops[FB_OP_SSYEVR][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_ssyevr_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SSYEVR);

    thunk = (fb_ssyevr_fortran_slot_fn)vtable.ext_ops[FB_OP_SSYEVR][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SSYEVR CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&jobz, &range, &uplo, &n, a, &lda, &vl, &vu, &il, &iu, &abstol, &m,
          w, z, &ldz, isuppz, work, &lwork, iwork, &liwork, &info);
    if (info != 241 || g_ssyevr_cblas_call.called != 1 ||
        g_ssyevr_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_ssyevr_cblas_call.jobz != 'V' || g_ssyevr_cblas_call.range != 'I' ||
        g_ssyevr_cblas_call.uplo != FB_LOWER || g_ssyevr_cblas_call.n != 3 ||
        g_ssyevr_cblas_call.lda != 3 || g_ssyevr_cblas_call.vl != 1.0f ||
        g_ssyevr_cblas_call.vu != 6.0f || g_ssyevr_cblas_call.il != 1 ||
        g_ssyevr_cblas_call.iu != 2 || g_ssyevr_cblas_call.abstol != 0.25f ||
        g_ssyevr_cblas_call.ldz != 3 || m != 2 || w[0] != 11.0f ||
        w[1] != 12.0f || z[0] != 43.0f || isuppz[0] != 1 ||
        isuppz[1] != 2) {
        fprintf(stderr, "[FAIL] SSYEVR CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] SSYEVR CBLAS->Fortran thunk maps selective symmetric-eigen RRR arguments into the C entry\n");
    return 0;
}

static int check_dsyevr_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_dsyevr_fn thunk = NULL;
    double a[9] = { 0.0 };
    double w[3] = { 0.0, 0.0, 0.0 };
    double z[9] = { 0.0 };
    int isuppz[6] = { 0, 0, 0, 0, 0, 0 };
    int m = 0;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dsyevr_fortran_call, 0, sizeof(g_dsyevr_fortran_call));

    vtable.ext_ops[FB_OP_DSYEVR][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_dsyevr_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_DSYEVR);

    thunk = (fb_dsyevr_fn)vtable.ext_ops[FB_OP_DSYEVR][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DSYEVR Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 'V', 'I', FB_UPPER, 3, a, 3, 1.5, 6.5,
                 1, 2, 0.375, &m, w, z, 3, isuppz);
    if (info != 0 || g_dsyevr_fortran_call.query_calls != 1 ||
        g_dsyevr_fortran_call.exec_calls != 1 ||
        g_dsyevr_fortran_call.query_lwork != -1 ||
        g_dsyevr_fortran_call.exec_lwork != 53 ||
        g_dsyevr_fortran_call.exec_liwork != 27 || m != 2 ||
        w[0] != 17.0 || w[1] != 18.0 || z[0] != 141.0 ||
        isuppz[0] != 1 || isuppz[1] != 3) {
        fprintf(stderr, "[FAIL] DSYEVR Fortran->CBLAS thunk did not preserve selective double symmetric-eigen RRR query semantics\n");
        return 1;
    }

    printf("[PASS] DSYEVR Fortran->CBLAS thunk performs the workspace queries and forwards selective double symmetric-eigen RRR outputs\n");
    return 0;
}

static int check_dsyevr_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_dsyevr_fortran_slot_fn thunk = NULL;
    double a[9] = { 0.0 };
    double w[3] = { 0.0, 0.0, 0.0 };
    double z[9] = { 0.0 };
    double work[16] = { 0.0 };
    int iwork[16] = { 0 };
    int isuppz[6] = { 0, 0, 0, 0, 0, 0 };
    char jobz = 'V';
    char range = 'I';
    char uplo = 'U';
    int n = 3;
    int lda = 3;
    double vl = 1.5;
    double vu = 6.5;
    int il = 1;
    int iu = 2;
    double abstol = 0.375;
    int m = 0;
    int ldz = 3;
    int lwork = 16;
    int liwork = 16;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dsyevr_cblas_call, 0, sizeof(g_dsyevr_cblas_call));
    g_dsyevr_cblas_rc = 341;

    vtable.ext_ops[FB_OP_DSYEVR][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_dsyevr_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_DSYEVR);

    thunk =
        (fb_dsyevr_fortran_slot_fn)vtable.ext_ops[FB_OP_DSYEVR][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DSYEVR CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&jobz, &range, &uplo, &n, a, &lda, &vl, &vu, &il, &iu, &abstol, &m,
          w, z, &ldz, isuppz, work, &lwork, iwork, &liwork, &info);
    if (info != 341 || g_dsyevr_cblas_call.called != 1 ||
        g_dsyevr_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_dsyevr_cblas_call.jobz != 'V' || g_dsyevr_cblas_call.range != 'I' ||
        g_dsyevr_cblas_call.uplo != FB_UPPER || g_dsyevr_cblas_call.n != 3 ||
        g_dsyevr_cblas_call.lda != 3 || g_dsyevr_cblas_call.vl != 1.5 ||
        g_dsyevr_cblas_call.vu != 6.5 || g_dsyevr_cblas_call.il != 1 ||
        g_dsyevr_cblas_call.iu != 2 || g_dsyevr_cblas_call.abstol != 0.375 ||
        g_dsyevr_cblas_call.ldz != 3 || m != 2 || w[0] != 21.0 ||
        w[1] != 22.0 || z[0] != 143.0 || isuppz[0] != 1 ||
        isuppz[1] != 2) {
        fprintf(stderr, "[FAIL] DSYEVR CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] DSYEVR CBLAS->Fortran thunk maps selective double symmetric-eigen RRR arguments into the C entry\n");
    return 0;
}

static int check_cheevr_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_cheevr_fn thunk = NULL;
    fb_complex_float_t a[9] = { 0 };
    fb_complex_float_t z[9] = { 0 };
    float w[3] = { 0.0f, 0.0f, 0.0f };
    int isuppz[6] = { 0, 0, 0, 0, 0, 0 };
    int m = 0;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cheevr_fortran_call, 0, sizeof(g_cheevr_fortran_call));

    vtable.ext_ops[FB_OP_CHEEVR][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_cheevr_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CHEEVR);

    thunk = (fb_cheevr_fn)vtable.ext_ops[FB_OP_CHEEVR][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CHEEVR Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 'V', 'I', FB_UPPER, 3, a, 3, 2.0f, 7.0f,
                 1, 2, 0.5f, &m, w, z, 3, isuppz);
    if (info != 0 || g_cheevr_fortran_call.query_calls != 1 ||
        g_cheevr_fortran_call.exec_calls != 1 ||
        g_cheevr_fortran_call.query_lwork != -1 ||
        g_cheevr_fortran_call.exec_lwork != 34 ||
        g_cheevr_fortran_call.exec_lrwork != 19 ||
        g_cheevr_fortran_call.exec_liwork != 21 || m != 2 ||
        w[0] != 9.0f || w[1] != 10.0f || cfloat_real(z[0]) != 42.0f ||
        isuppz[0] != 2 || isuppz[1] != 3) {
        fprintf(stderr, "[FAIL] CHEEVR Fortran->CBLAS thunk did not preserve selective Hermitian-eigen RRR query semantics\n");
        return 1;
    }

    printf("[PASS] CHEEVR Fortran->CBLAS thunk performs the workspace queries and forwards selective Hermitian-eigen RRR outputs\n");
    return 0;
}

static int check_cheevr_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_cheevr_fortran_slot_fn thunk = NULL;
    fb_complex_float_t a[9] = { 0 };
    fb_complex_float_t z[9] = { 0 };
    fb_complex_float_t work[16] = { 0 };
    float w[3] = { 0.0f, 0.0f, 0.0f };
    float rwork[24] = { 0.0f };
    int iwork[24] = { 0 };
    int isuppz[6] = { 0, 0, 0, 0, 0, 0 };
    char jobz = 'V';
    char range = 'I';
    char uplo = 'U';
    int n = 3;
    int lda = 3;
    float vl = 2.0f;
    float vu = 7.0f;
    int il = 1;
    int iu = 2;
    float abstol = 0.5f;
    int m = 0;
    int ldz = 3;
    int lwork = 16;
    int lrwork = 24;
    int liwork = 24;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cheevr_cblas_call, 0, sizeof(g_cheevr_cblas_call));
    g_cheevr_cblas_rc = 243;

    vtable.ext_ops[FB_OP_CHEEVR][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_cheevr_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CHEEVR);

    thunk = (fb_cheevr_fortran_slot_fn)vtable.ext_ops[FB_OP_CHEEVR][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CHEEVR CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&jobz, &range, &uplo, &n, a, &lda, &vl, &vu, &il, &iu, &abstol, &m,
          w, z, &ldz, isuppz, work, &lwork, rwork, &lrwork, iwork, &liwork,
          &info);
    if (info != 243 || g_cheevr_cblas_call.called != 1 ||
        g_cheevr_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_cheevr_cblas_call.jobz != 'V' || g_cheevr_cblas_call.range != 'I' ||
        g_cheevr_cblas_call.uplo != FB_UPPER || g_cheevr_cblas_call.n != 3 ||
        g_cheevr_cblas_call.lda != 3 || g_cheevr_cblas_call.vl != 2.0f ||
        g_cheevr_cblas_call.vu != 7.0f || g_cheevr_cblas_call.il != 1 ||
        g_cheevr_cblas_call.iu != 2 || g_cheevr_cblas_call.abstol != 0.5f ||
        g_cheevr_cblas_call.ldz != 3 || m != 2 || w[0] != 13.0f ||
        w[1] != 14.0f || cfloat_real(z[0]) != 44.0f || isuppz[0] != 2 ||
        isuppz[1] != 4) {
        fprintf(stderr, "[FAIL] CHEEVR CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] CHEEVR CBLAS->Fortran thunk maps selective Hermitian-eigen RRR arguments into the C entry\n");
    return 0;
}

static int check_zheevr_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_zheevr_fn thunk = NULL;
    fb_complex_double_t a[9] = { 0 };
    fb_complex_double_t z[9] = { 0 };
    double w[3] = { 0.0, 0.0, 0.0 };
    int isuppz[6] = { 0, 0, 0, 0, 0, 0 };
    int m = 0;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zheevr_fortran_call, 0, sizeof(g_zheevr_fortran_call));

    vtable.ext_ops[FB_OP_ZHEEVR][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_zheevr_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_ZHEEVR);

    thunk = (fb_zheevr_fn)vtable.ext_ops[FB_OP_ZHEEVR][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZHEEVR Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 'V', 'I', FB_LOWER, 3, a, 3, 2.5, 7.5,
                 1, 2, 0.625, &m, w, z, 3, isuppz);
    if (info != 0 || g_zheevr_fortran_call.query_calls != 1 ||
        g_zheevr_fortran_call.exec_calls != 1 ||
        g_zheevr_fortran_call.query_lwork != -1 ||
        g_zheevr_fortran_call.exec_lwork != 54 ||
        g_zheevr_fortran_call.exec_lrwork != 29 ||
        g_zheevr_fortran_call.exec_liwork != 31 || m != 2 ||
        w[0] != 19.0 || w[1] != 20.0 || cdouble_real(z[0]) != 142.0 ||
        isuppz[0] != 2 || isuppz[1] != 3) {
        fprintf(stderr, "[FAIL] ZHEEVR Fortran->CBLAS thunk did not preserve selective complex-double Hermitian-eigen RRR query semantics\n");
        return 1;
    }

    printf("[PASS] ZHEEVR Fortran->CBLAS thunk performs the workspace queries and forwards selective complex-double Hermitian-eigen RRR outputs\n");
    return 0;
}

static int check_zheevr_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_zheevr_fortran_slot_fn thunk = NULL;
    fb_complex_double_t a[9] = { 0 };
    fb_complex_double_t z[9] = { 0 };
    fb_complex_double_t work[16] = { 0 };
    double w[3] = { 0.0, 0.0, 0.0 };
    double rwork[24] = { 0.0 };
    int iwork[24] = { 0 };
    int isuppz[6] = { 0, 0, 0, 0, 0, 0 };
    char jobz = 'V';
    char range = 'I';
    char uplo = 'L';
    int n = 3;
    int lda = 3;
    double vl = 2.5;
    double vu = 7.5;
    int il = 1;
    int iu = 2;
    double abstol = 0.625;
    int m = 0;
    int ldz = 3;
    int lwork = 16;
    int lrwork = 24;
    int liwork = 24;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zheevr_cblas_call, 0, sizeof(g_zheevr_cblas_call));
    g_zheevr_cblas_rc = 343;

    vtable.ext_ops[FB_OP_ZHEEVR][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_zheevr_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_ZHEEVR);

    thunk =
        (fb_zheevr_fortran_slot_fn)vtable.ext_ops[FB_OP_ZHEEVR][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZHEEVR CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&jobz, &range, &uplo, &n, a, &lda, &vl, &vu, &il, &iu, &abstol, &m,
          w, z, &ldz, isuppz, work, &lwork, rwork, &lrwork, iwork, &liwork,
          &info);
    if (info != 343 || g_zheevr_cblas_call.called != 1 ||
        g_zheevr_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_zheevr_cblas_call.jobz != 'V' || g_zheevr_cblas_call.range != 'I' ||
        g_zheevr_cblas_call.uplo != FB_LOWER || g_zheevr_cblas_call.n != 3 ||
        g_zheevr_cblas_call.lda != 3 || g_zheevr_cblas_call.vl != 2.5 ||
        g_zheevr_cblas_call.vu != 7.5 || g_zheevr_cblas_call.il != 1 ||
        g_zheevr_cblas_call.iu != 2 || g_zheevr_cblas_call.abstol != 0.625 ||
        g_zheevr_cblas_call.ldz != 3 || m != 2 || w[0] != 23.0 ||
        w[1] != 24.0 || cdouble_real(z[0]) != 144.0 || isuppz[0] != 2 ||
        isuppz[1] != 4) {
        fprintf(stderr, "[FAIL] ZHEEVR CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] ZHEEVR CBLAS->Fortran thunk maps selective complex-double Hermitian-eigen RRR arguments into the C entry\n");
    return 0;
}

static int check_ssyevx_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_ssyevx_fn thunk = NULL;
    float a[9] = { 0.0f };
    float w[3] = { 0.0f, 0.0f, 0.0f };
    float z[9] = { 0.0f };
    int ifail[3] = { 0, 0, 0 };
    int m = 0;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_ssyevx_fortran_call, 0, sizeof(g_ssyevx_fortran_call));
    vtable.ext_ops[FB_OP_SSYEVX][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_ssyevx_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SSYEVX);

    thunk = (fb_ssyevx_fn)vtable.ext_ops[FB_OP_SSYEVX][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SSYEVX Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 'V', 'I', FB_LOWER, 3, a, 3, 1.0f, 6.0f,
                 1, 2, 0.25f, &m, w, z, 3, ifail);
    if (info != 0 || g_ssyevx_fortran_call.query_calls != 1 ||
        g_ssyevx_fortran_call.exec_calls != 1 ||
        g_ssyevx_fortran_call.query_lwork != -1 ||
        g_ssyevx_fortran_call.exec_lwork != 35 || m != 2 ||
        w[0] != 7.0f || w[1] != 8.0f || z[0] != 105.0f || ifail[0] != 1) {
        fprintf(stderr, "[FAIL] SSYEVX Fortran->CBLAS thunk did not preserve selective symmetric-eigen query semantics\n");
        return 1;
    }

    printf("[PASS] SSYEVX Fortran->CBLAS thunk performs the workspace query and forwards selective symmetric eigen outputs\n");
    return 0;
}

static int check_ssyevx_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_ssyevx_fortran_slot_fn thunk = NULL;
    float a[9] = { 0.0f };
    float w[3] = { 0.0f, 0.0f, 0.0f };
    float z[9] = { 0.0f };
    float work[16] = { 0.0f };
    int iwork[16] = { 0 };
    int ifail[3] = { 0, 0, 0 };
    char jobz = 'V';
    char range = 'I';
    char uplo = 'L';
    int n = 3;
    int lda = 3;
    float vl = 1.0f;
    float vu = 6.0f;
    int il = 1;
    int iu = 2;
    float abstol = 0.25f;
    int m = 0;
    int ldz = 3;
    int lwork = 16;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_ssyevx_cblas_call, 0, sizeof(g_ssyevx_cblas_call));
    g_ssyevx_cblas_rc = 235;

    vtable.ext_ops[FB_OP_SSYEVX][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_ssyevx_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SSYEVX);

    thunk = (fb_ssyevx_fortran_slot_fn)vtable.ext_ops[FB_OP_SSYEVX][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SSYEVX CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&jobz, &range, &uplo, &n, a, &lda, &vl, &vu, &il, &iu, &abstol, &m,
          w, z, &ldz, work, &lwork, iwork, ifail, &info);
    if (info != 235 || g_ssyevx_cblas_call.called != 1 ||
        g_ssyevx_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_ssyevx_cblas_call.jobz != 'V' ||
        g_ssyevx_cblas_call.range != 'I' ||
        g_ssyevx_cblas_call.uplo != FB_LOWER || g_ssyevx_cblas_call.n != 3 ||
        g_ssyevx_cblas_call.lda != 3 || g_ssyevx_cblas_call.vl != 1.0f ||
        g_ssyevx_cblas_call.vu != 6.0f || g_ssyevx_cblas_call.il != 1 ||
        g_ssyevx_cblas_call.iu != 2 || g_ssyevx_cblas_call.abstol != 0.25f ||
        g_ssyevx_cblas_call.ldz != 3 || m != 2 || w[0] != 11.0f ||
        w[1] != 12.0f || z[0] != 107.0f || ifail[0] != 3) {
        fprintf(stderr, "[FAIL] SSYEVX CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] SSYEVX CBLAS->Fortran thunk maps selective symmetric-eigen arguments into the C entry\n");
    return 0;
}

static int check_dsyevx_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_dsyevx_fn thunk = NULL;
    double a[9] = { 0.0 };
    double w[3] = { 0.0, 0.0, 0.0 };
    double z[9] = { 0.0 };
    int ifail[3] = { 0, 0, 0 };
    int m = 0;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dsyevx_fortran_call, 0, sizeof(g_dsyevx_fortran_call));
    vtable.ext_ops[FB_OP_DSYEVX][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_dsyevx_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_DSYEVX);

    thunk = (fb_dsyevx_fn)vtable.ext_ops[FB_OP_DSYEVX][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DSYEVX Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 'V', 'I', FB_UPPER, 3, a, 3, 1.5, 6.5,
                 1, 2, 0.375, &m, w, z, 3, ifail);
    if (info != 0 || g_dsyevx_fortran_call.query_calls != 1 ||
        g_dsyevx_fortran_call.exec_calls != 1 ||
        g_dsyevx_fortran_call.query_lwork != -1 ||
        g_dsyevx_fortran_call.exec_lwork != 55 || m != 2 ||
        w[0] != 17.0 || w[1] != 18.0 || z[0] != 205.0 || ifail[0] != 1) {
        fprintf(stderr, "[FAIL] DSYEVX Fortran->CBLAS thunk did not preserve selective double symmetric-eigen query semantics\n");
        return 1;
    }

    printf("[PASS] DSYEVX Fortran->CBLAS thunk performs the workspace query and forwards selective double symmetric eigen outputs\n");
    return 0;
}

static int check_dsyevx_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_dsyevx_fortran_slot_fn thunk = NULL;
    double a[9] = { 0.0 };
    double w[3] = { 0.0, 0.0, 0.0 };
    double z[9] = { 0.0 };
    double work[16] = { 0.0 };
    int iwork[16] = { 0 };
    int ifail[3] = { 0, 0, 0 };
    char jobz = 'V';
    char range = 'I';
    char uplo = 'U';
    int n = 3;
    int lda = 3;
    double vl = 1.5;
    double vu = 6.5;
    int il = 1;
    int iu = 2;
    double abstol = 0.375;
    int m = 0;
    int ldz = 3;
    int lwork = 16;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dsyevx_cblas_call, 0, sizeof(g_dsyevx_cblas_call));
    g_dsyevx_cblas_rc = 345;

    vtable.ext_ops[FB_OP_DSYEVX][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_dsyevx_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_DSYEVX);

    thunk =
        (fb_dsyevx_fortran_slot_fn)vtable.ext_ops[FB_OP_DSYEVX][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DSYEVX CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&jobz, &range, &uplo, &n, a, &lda, &vl, &vu, &il, &iu, &abstol, &m,
          w, z, &ldz, work, &lwork, iwork, ifail, &info);
    if (info != 345 || g_dsyevx_cblas_call.called != 1 ||
        g_dsyevx_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_dsyevx_cblas_call.jobz != 'V' ||
        g_dsyevx_cblas_call.range != 'I' ||
        g_dsyevx_cblas_call.uplo != FB_UPPER || g_dsyevx_cblas_call.n != 3 ||
        g_dsyevx_cblas_call.lda != 3 || g_dsyevx_cblas_call.vl != 1.5 ||
        g_dsyevx_cblas_call.vu != 6.5 || g_dsyevx_cblas_call.il != 1 ||
        g_dsyevx_cblas_call.iu != 2 || g_dsyevx_cblas_call.abstol != 0.375 ||
        g_dsyevx_cblas_call.ldz != 3 || m != 2 || w[0] != 21.0 ||
        w[1] != 22.0 || z[0] != 207.0 || ifail[0] != 3) {
        fprintf(stderr, "[FAIL] DSYEVX CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] DSYEVX CBLAS->Fortran thunk maps selective double symmetric-eigen arguments into the C entry\n");
    return 0;
}

static int check_cheevx_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_cheevx_fn thunk = NULL;
    fb_complex_float_t a[9] = { 0 };
    fb_complex_float_t z[9] = { 0 };
    float w[3] = { 0.0f, 0.0f, 0.0f };
    int ifail[3] = { 0, 0, 0 };
    int m = 0;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cheevx_fortran_call, 0, sizeof(g_cheevx_fortran_call));
    vtable.ext_ops[FB_OP_CHEEVX][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_cheevx_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CHEEVX);

    thunk = (fb_cheevx_fn)vtable.ext_ops[FB_OP_CHEEVX][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CHEEVX Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 'V', 'I', FB_UPPER, 3, a, 3, 2.0f, 7.0f,
                 1, 2, 0.5f, &m, w, z, 3, ifail);
    if (info != 0 || g_cheevx_fortran_call.query_calls != 1 ||
        g_cheevx_fortran_call.exec_calls != 1 ||
        g_cheevx_fortran_call.query_lwork != -1 ||
        g_cheevx_fortran_call.exec_lwork != 36 || m != 2 ||
        w[0] != 9.0f || w[1] != 10.0f || cfloat_real(z[0]) != 106.0f ||
        ifail[0] != 2) {
        fprintf(stderr, "[FAIL] CHEEVX Fortran->CBLAS thunk did not preserve selective Hermitian-eigen query semantics\n");
        return 1;
    }

    printf("[PASS] CHEEVX Fortran->CBLAS thunk performs the workspace query and forwards selective Hermitian eigen outputs\n");
    return 0;
}

static int check_cheevx_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_cheevx_fortran_slot_fn thunk = NULL;
    fb_complex_float_t a[9] = { 0 };
    fb_complex_float_t z[9] = { 0 };
    fb_complex_float_t work[16] = { 0 };
    float w[3] = { 0.0f, 0.0f, 0.0f };
    float rwork[24] = { 0.0f };
    int iwork[16] = { 0 };
    int ifail[3] = { 0, 0, 0 };
    char jobz = 'V';
    char range = 'I';
    char uplo = 'U';
    int n = 3;
    int lda = 3;
    float vl = 2.0f;
    float vu = 7.0f;
    int il = 1;
    int iu = 2;
    float abstol = 0.5f;
    int m = 0;
    int ldz = 3;
    int lwork = 16;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cheevx_cblas_call, 0, sizeof(g_cheevx_cblas_call));
    g_cheevx_cblas_rc = 237;

    vtable.ext_ops[FB_OP_CHEEVX][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_cheevx_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CHEEVX);

    thunk = (fb_cheevx_fortran_slot_fn)vtable.ext_ops[FB_OP_CHEEVX][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CHEEVX CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&jobz, &range, &uplo, &n, a, &lda, &vl, &vu, &il, &iu, &abstol, &m,
          w, z, &ldz, work, &lwork, rwork, iwork, ifail, &info);
    if (info != 237 || g_cheevx_cblas_call.called != 1 ||
        g_cheevx_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_cheevx_cblas_call.jobz != 'V' ||
        g_cheevx_cblas_call.range != 'I' ||
        g_cheevx_cblas_call.uplo != FB_UPPER || g_cheevx_cblas_call.n != 3 ||
        g_cheevx_cblas_call.lda != 3 || g_cheevx_cblas_call.vl != 2.0f ||
        g_cheevx_cblas_call.vu != 7.0f || g_cheevx_cblas_call.il != 1 ||
        g_cheevx_cblas_call.iu != 2 || g_cheevx_cblas_call.abstol != 0.5f ||
        g_cheevx_cblas_call.ldz != 3 || m != 2 || w[0] != 13.0f ||
        w[1] != 14.0f || cfloat_real(z[0]) != 108.0f || ifail[0] != 4) {
        fprintf(stderr, "[FAIL] CHEEVX CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] CHEEVX CBLAS->Fortran thunk maps selective Hermitian-eigen arguments into the C entry\n");
    return 0;
}

static int check_zheevx_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_zheevx_fn thunk = NULL;
    fb_complex_double_t a[9] = { 0 };
    fb_complex_double_t z[9] = { 0 };
    double w[3] = { 0.0, 0.0, 0.0 };
    int ifail[3] = { 0, 0, 0 };
    int m = 0;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zheevx_fortran_call, 0, sizeof(g_zheevx_fortran_call));
    vtable.ext_ops[FB_OP_ZHEEVX][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_zheevx_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_ZHEEVX);

    thunk = (fb_zheevx_fn)vtable.ext_ops[FB_OP_ZHEEVX][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZHEEVX Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 'V', 'I', FB_LOWER, 3, a, 3, 2.5, 7.5,
                 1, 2, 0.625, &m, w, z, 3, ifail);
    if (info != 0 || g_zheevx_fortran_call.query_calls != 1 ||
        g_zheevx_fortran_call.exec_calls != 1 ||
        g_zheevx_fortran_call.query_lwork != -1 ||
        g_zheevx_fortran_call.exec_lwork != 56 || m != 2 ||
        w[0] != 19.0 || w[1] != 20.0 || cdouble_real(z[0]) != 206.0 ||
        ifail[0] != 2) {
        fprintf(stderr, "[FAIL] ZHEEVX Fortran->CBLAS thunk did not preserve selective complex-double Hermitian-eigen query semantics\n");
        return 1;
    }

    printf("[PASS] ZHEEVX Fortran->CBLAS thunk performs the workspace query and forwards selective complex-double Hermitian eigen outputs\n");
    return 0;
}

static int check_zheevx_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_zheevx_fortran_slot_fn thunk = NULL;
    fb_complex_double_t a[9] = { 0 };
    fb_complex_double_t z[9] = { 0 };
    fb_complex_double_t work[16] = { 0 };
    double w[3] = { 0.0, 0.0, 0.0 };
    double rwork[24] = { 0.0 };
    int iwork[16] = { 0 };
    int ifail[3] = { 0, 0, 0 };
    char jobz = 'V';
    char range = 'I';
    char uplo = 'L';
    int n = 3;
    int lda = 3;
    double vl = 2.5;
    double vu = 7.5;
    int il = 1;
    int iu = 2;
    double abstol = 0.625;
    int m = 0;
    int ldz = 3;
    int lwork = 16;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zheevx_cblas_call, 0, sizeof(g_zheevx_cblas_call));
    g_zheevx_cblas_rc = 347;

    vtable.ext_ops[FB_OP_ZHEEVX][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_zheevx_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_ZHEEVX);

    thunk =
        (fb_zheevx_fortran_slot_fn)vtable.ext_ops[FB_OP_ZHEEVX][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZHEEVX CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&jobz, &range, &uplo, &n, a, &lda, &vl, &vu, &il, &iu, &abstol, &m,
          w, z, &ldz, work, &lwork, rwork, iwork, ifail, &info);
    if (info != 347 || g_zheevx_cblas_call.called != 1 ||
        g_zheevx_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_zheevx_cblas_call.jobz != 'V' ||
        g_zheevx_cblas_call.range != 'I' ||
        g_zheevx_cblas_call.uplo != FB_LOWER || g_zheevx_cblas_call.n != 3 ||
        g_zheevx_cblas_call.lda != 3 || g_zheevx_cblas_call.vl != 2.5 ||
        g_zheevx_cblas_call.vu != 7.5 || g_zheevx_cblas_call.il != 1 ||
        g_zheevx_cblas_call.iu != 2 || g_zheevx_cblas_call.abstol != 0.625 ||
        g_zheevx_cblas_call.ldz != 3 || m != 2 || w[0] != 23.0 ||
        w[1] != 24.0 || cdouble_real(z[0]) != 208.0 || ifail[0] != 4) {
        fprintf(stderr, "[FAIL] ZHEEVX CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] ZHEEVX CBLAS->Fortran thunk maps selective complex-double Hermitian-eigen arguments into the C entry\n");
    return 0;
}

int main(void)
{
    if (check_ssyev_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_ssyev_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_dsyev_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_dsyev_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_cheev_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_cheev_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_zheev_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_zheev_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_ssyevr_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_ssyevr_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_dsyevr_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_dsyevr_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_cheevr_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_cheevr_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_zheevr_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_zheevr_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_ssyevx_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_ssyevx_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_dsyevx_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_dsyevx_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_cheevx_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_cheevx_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_zheevx_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_zheevx_cblas_to_fortran() != 0) {
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}