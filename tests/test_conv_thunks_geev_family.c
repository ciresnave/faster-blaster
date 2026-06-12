#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef int (*fb_sgeev_fn)(fb_layout_t layout, char jobvl, char jobvr, int n,
                           float *a, int lda, float *wr, float *wi,
                           float *vl, int ldvl, float *vr, int ldvr);
typedef int (*fb_dgeev_fn)(fb_layout_t layout, char jobvl, char jobvr, int n,
                           double *a, int lda, double *wr, double *wi,
                           double *vl, int ldvl, double *vr, int ldvr);
typedef int (*fb_cgeev_fn)(fb_layout_t layout, char jobvl, char jobvr, int n,
                           fb_complex_float_t *a, int lda,
                           fb_complex_float_t *w,
                           fb_complex_float_t *vl, int ldvl,
                           fb_complex_float_t *vr, int ldvr);
typedef int (*fb_zgeev_fn)(fb_layout_t layout, char jobvl, char jobvr, int n,
                           fb_complex_double_t *a, int lda,
                           fb_complex_double_t *w,
                           fb_complex_double_t *vl, int ldvl,
                           fb_complex_double_t *vr, int ldvr);

typedef void (*fb_sgeev_fortran_slot_fn)(char *jobvl, char *jobvr, int *n,
                                         float *a, int *lda, float *wr,
                                         float *wi, float *vl, int *ldvl,
                                         float *vr, int *ldvr,
                                         float *work, int *lwork, int *info);
typedef void (*fb_dgeev_fortran_slot_fn)(char *jobvl, char *jobvr, int *n,
                                         double *a, int *lda, double *wr,
                                         double *wi, double *vl, int *ldvl,
                                         double *vr, int *ldvr,
                                         double *work, int *lwork,
                                         int *info);
typedef void (*fb_cgeev_fortran_slot_fn)(char *jobvl, char *jobvr, int *n,
                                         fb_complex_float_t *a, int *lda,
                                         fb_complex_float_t *w,
                                         fb_complex_float_t *vl, int *ldvl,
                                         fb_complex_float_t *vr, int *ldvr,
                                         fb_complex_float_t *work, int *lwork,
                                         float *rwork, int *info);
typedef void (*fb_zgeev_fortran_slot_fn)(char *jobvl, char *jobvr, int *n,
                                         fb_complex_double_t *a, int *lda,
                                         fb_complex_double_t *w,
                                         fb_complex_double_t *vl, int *ldvl,
                                         fb_complex_double_t *vr, int *ldvr,
                                         fb_complex_double_t *work,
                                         int *lwork, double *rwork,
                                         int *info);

static struct {
    int query_calls;
    int exec_calls;
    char jobvl;
    char jobvr;
    int n;
    int lda;
    int ldvl;
    int ldvr;
    int query_lwork;
    int exec_lwork;
} g_sgeev_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    char jobvl;
    char jobvr;
    int n;
    int lda;
    int ldvl;
    int ldvr;
    float *a;
    float *wr;
    float *wi;
    float *vl;
    float *vr;
} g_sgeev_cblas_call;

static struct {
    int query_calls;
    int exec_calls;
    char jobvl;
    char jobvr;
    int n;
    int lda;
    int ldvl;
    int ldvr;
    int query_lwork;
    int exec_lwork;
} g_dgeev_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    char jobvl;
    char jobvr;
    int n;
    int lda;
    int ldvl;
    int ldvr;
    double *a;
    double *wr;
    double *wi;
    double *vl;
    double *vr;
} g_dgeev_cblas_call;

static struct {
    int query_calls;
    int exec_calls;
    char jobvl;
    char jobvr;
    int n;
    int lda;
    int ldvl;
    int ldvr;
    int query_lwork;
    int exec_lwork;
} g_cgeev_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    char jobvl;
    char jobvr;
    int n;
    int lda;
    int ldvl;
    int ldvr;
    fb_complex_float_t *a;
    fb_complex_float_t *w;
    fb_complex_float_t *vl;
    fb_complex_float_t *vr;
} g_cgeev_cblas_call;

static struct {
    int query_calls;
    int exec_calls;
    char jobvl;
    char jobvr;
    int n;
    int lda;
    int ldvl;
    int ldvr;
    int query_lwork;
    int exec_lwork;
} g_zgeev_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    char jobvl;
    char jobvr;
    int n;
    int lda;
    int ldvl;
    int ldvr;
    fb_complex_double_t *a;
    fb_complex_double_t *w;
    fb_complex_double_t *vl;
    fb_complex_double_t *vr;
} g_zgeev_cblas_call;

static int g_sgeev_cblas_rc = 0;
static int g_dgeev_cblas_rc = 0;
static int g_cgeev_cblas_rc = 0;
static int g_zgeev_cblas_rc = 0;

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

static void stub_sgeev_fortran(char *jobvl, char *jobvr, int *n, float *a,
                               int *lda, float *wr, float *wi, float *vl,
                               int *ldvl, float *vr, int *ldvr,
                               float *work, int *lwork, int *info)
{
    if (*lwork == -1) {
        g_sgeev_fortran_call.query_calls += 1;
        g_sgeev_fortran_call.query_lwork = *lwork;
        work[0] = 23.0f;
        *info = 0;
        return;
    }

    g_sgeev_fortran_call.exec_calls += 1;
    g_sgeev_fortran_call.jobvl = *jobvl;
    g_sgeev_fortran_call.jobvr = *jobvr;
    g_sgeev_fortran_call.n = *n;
    g_sgeev_fortran_call.lda = *lda;
    g_sgeev_fortran_call.ldvl = *ldvl;
    g_sgeev_fortran_call.ldvr = *ldvr;
    g_sgeev_fortran_call.exec_lwork = *lwork;
    (void)a;
    wr[0] = 1.0f;
    wr[1] = 2.0f;
    wi[0] = 0.0f;
    wi[1] = 0.5f;
    vl[0] = 10.0f;
    vl[3] = 13.0f;
    vr[0] = 20.0f;
    vr[3] = 23.0f;
    *info = 0;
}

static int stub_sgeev_cblas(fb_layout_t layout, char jobvl, char jobvr, int n,
                            float *a, int lda, float *wr, float *wi,
                            float *vl, int ldvl, float *vr, int ldvr)
{
    g_sgeev_cblas_call.called += 1;
    g_sgeev_cblas_call.layout = layout;
    g_sgeev_cblas_call.jobvl = jobvl;
    g_sgeev_cblas_call.jobvr = jobvr;
    g_sgeev_cblas_call.n = n;
    g_sgeev_cblas_call.lda = lda;
    g_sgeev_cblas_call.ldvl = ldvl;
    g_sgeev_cblas_call.ldvr = ldvr;
    g_sgeev_cblas_call.a = a;
    g_sgeev_cblas_call.wr = wr;
    g_sgeev_cblas_call.wi = wi;
    g_sgeev_cblas_call.vl = vl;
    g_sgeev_cblas_call.vr = vr;
    return g_sgeev_cblas_rc;
}

static void stub_dgeev_fortran(char *jobvl, char *jobvr, int *n, double *a,
                               int *lda, double *wr, double *wi, double *vl,
                               int *ldvl, double *vr, int *ldvr,
                               double *work, int *lwork, int *info)
{
    if (*lwork == -1) {
        g_dgeev_fortran_call.query_calls += 1;
        g_dgeev_fortran_call.query_lwork = *lwork;
        work[0] = 33.0;
        *info = 0;
        return;
    }

    g_dgeev_fortran_call.exec_calls += 1;
    g_dgeev_fortran_call.jobvl = *jobvl;
    g_dgeev_fortran_call.jobvr = *jobvr;
    g_dgeev_fortran_call.n = *n;
    g_dgeev_fortran_call.lda = *lda;
    g_dgeev_fortran_call.ldvl = *ldvl;
    g_dgeev_fortran_call.ldvr = *ldvr;
    g_dgeev_fortran_call.exec_lwork = *lwork;
    (void)a;
    wr[0] = 5.0;
    wr[1] = 6.0;
    wi[0] = 0.0;
    wi[1] = 0.25;
    vl[0] = 50.0;
    vl[3] = 53.0;
    vr[0] = 60.0;
    vr[3] = 63.0;
    *info = 0;
}

static int stub_dgeev_cblas(fb_layout_t layout, char jobvl, char jobvr, int n,
                            double *a, int lda, double *wr, double *wi,
                            double *vl, int ldvl, double *vr, int ldvr)
{
    g_dgeev_cblas_call.called += 1;
    g_dgeev_cblas_call.layout = layout;
    g_dgeev_cblas_call.jobvl = jobvl;
    g_dgeev_cblas_call.jobvr = jobvr;
    g_dgeev_cblas_call.n = n;
    g_dgeev_cblas_call.lda = lda;
    g_dgeev_cblas_call.ldvl = ldvl;
    g_dgeev_cblas_call.ldvr = ldvr;
    g_dgeev_cblas_call.a = a;
    g_dgeev_cblas_call.wr = wr;
    g_dgeev_cblas_call.wi = wi;
    g_dgeev_cblas_call.vl = vl;
    g_dgeev_cblas_call.vr = vr;
    return g_dgeev_cblas_rc;
}

static void stub_cgeev_fortran(char *jobvl, char *jobvr, int *n,
                               fb_complex_float_t *a, int *lda,
                               fb_complex_float_t *w,
                               fb_complex_float_t *vl, int *ldvl,
                               fb_complex_float_t *vr, int *ldvr,
                               fb_complex_float_t *work, int *lwork,
                               float *rwork, int *info)
{
    (void)rwork;
    if (*lwork == -1) {
        g_cgeev_fortran_call.query_calls += 1;
        g_cgeev_fortran_call.query_lwork = *lwork;
        work[0] = make_cfloat(24.0f);
        *info = 0;
        return;
    }

    g_cgeev_fortran_call.exec_calls += 1;
    g_cgeev_fortran_call.jobvl = *jobvl;
    g_cgeev_fortran_call.jobvr = *jobvr;
    g_cgeev_fortran_call.n = *n;
    g_cgeev_fortran_call.lda = *lda;
    g_cgeev_fortran_call.ldvl = *ldvl;
    g_cgeev_fortran_call.ldvr = *ldvr;
    g_cgeev_fortran_call.exec_lwork = *lwork;
    (void)a;
    w[0] = make_cfloat(3.0f);
    w[1] = make_cfloat(4.0f);
    vl[0] = make_cfloat(30.0f);
    vl[3] = make_cfloat(33.0f);
    vr[0] = make_cfloat(40.0f);
    vr[3] = make_cfloat(43.0f);
    *info = 0;
}

static int stub_cgeev_cblas(fb_layout_t layout, char jobvl, char jobvr, int n,
                            fb_complex_float_t *a, int lda,
                            fb_complex_float_t *w,
                            fb_complex_float_t *vl, int ldvl,
                            fb_complex_float_t *vr, int ldvr)
{
    g_cgeev_cblas_call.called += 1;
    g_cgeev_cblas_call.layout = layout;
    g_cgeev_cblas_call.jobvl = jobvl;
    g_cgeev_cblas_call.jobvr = jobvr;
    g_cgeev_cblas_call.n = n;
    g_cgeev_cblas_call.lda = lda;
    g_cgeev_cblas_call.ldvl = ldvl;
    g_cgeev_cblas_call.ldvr = ldvr;
    g_cgeev_cblas_call.a = a;
    g_cgeev_cblas_call.w = w;
    g_cgeev_cblas_call.vl = vl;
    g_cgeev_cblas_call.vr = vr;
    return g_cgeev_cblas_rc;
}

static void stub_zgeev_fortran(char *jobvl, char *jobvr, int *n,
                               fb_complex_double_t *a, int *lda,
                               fb_complex_double_t *w,
                               fb_complex_double_t *vl, int *ldvl,
                               fb_complex_double_t *vr, int *ldvr,
                               fb_complex_double_t *work, int *lwork,
                               double *rwork, int *info)
{
    (void)rwork;
    if (*lwork == -1) {
        g_zgeev_fortran_call.query_calls += 1;
        g_zgeev_fortran_call.query_lwork = *lwork;
        work[0] = make_cdouble(34.0);
        *info = 0;
        return;
    }

    g_zgeev_fortran_call.exec_calls += 1;
    g_zgeev_fortran_call.jobvl = *jobvl;
    g_zgeev_fortran_call.jobvr = *jobvr;
    g_zgeev_fortran_call.n = *n;
    g_zgeev_fortran_call.lda = *lda;
    g_zgeev_fortran_call.ldvl = *ldvl;
    g_zgeev_fortran_call.ldvr = *ldvr;
    g_zgeev_fortran_call.exec_lwork = *lwork;
    (void)a;
    w[0] = make_cdouble(7.0);
    w[1] = make_cdouble(8.0);
    vl[0] = make_cdouble(70.0);
    vl[3] = make_cdouble(73.0);
    vr[0] = make_cdouble(80.0);
    vr[3] = make_cdouble(83.0);
    *info = 0;
}

static int stub_zgeev_cblas(fb_layout_t layout, char jobvl, char jobvr, int n,
                            fb_complex_double_t *a, int lda,
                            fb_complex_double_t *w,
                            fb_complex_double_t *vl, int ldvl,
                            fb_complex_double_t *vr, int ldvr)
{
    g_zgeev_cblas_call.called += 1;
    g_zgeev_cblas_call.layout = layout;
    g_zgeev_cblas_call.jobvl = jobvl;
    g_zgeev_cblas_call.jobvr = jobvr;
    g_zgeev_cblas_call.n = n;
    g_zgeev_cblas_call.lda = lda;
    g_zgeev_cblas_call.ldvl = ldvl;
    g_zgeev_cblas_call.ldvr = ldvr;
    g_zgeev_cblas_call.a = a;
    g_zgeev_cblas_call.w = w;
    g_zgeev_cblas_call.vl = vl;
    g_zgeev_cblas_call.vr = vr;
    return g_zgeev_cblas_rc;
}

static int check_dgeev_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_dgeev_fn thunk = NULL;
    double a[4] = { 1.0, 2.0, 3.0, 4.0 };
    double wr[2] = { 0.0, 0.0 };
    double wi[2] = { 0.0, 0.0 };
    double vl[4] = { 0.0, 0.0, 0.0, 0.0 };
    double vr[4] = { 0.0, 0.0, 0.0, 0.0 };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dgeev_fortran_call, 0, sizeof(g_dgeev_fortran_call));

    vtable.ext_ops[FB_OP_DGEEV][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_dgeev_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_DGEEV);

    thunk = (fb_dgeev_fn)vtable.ext_ops[FB_OP_DGEEV][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DGEEV Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 'V', 'V', 2, a, 2, wr, wi, vl, 2, vr, 2);
    if (info != 0 || g_dgeev_fortran_call.query_calls != 1 ||
        g_dgeev_fortran_call.exec_calls != 1 ||
        g_dgeev_fortran_call.query_lwork != -1 ||
        g_dgeev_fortran_call.exec_lwork != 33 ||
        g_dgeev_fortran_call.jobvl != 'V' || g_dgeev_fortran_call.jobvr != 'V' ||
        wr[0] != 5.0 || wr[1] != 6.0 || wi[1] != 0.25 ||
        vl[0] != 50.0 || vl[3] != 53.0 || vr[0] != 60.0 || vr[3] != 63.0) {
        fprintf(stderr, "[FAIL] DGEEV Fortran->CBLAS thunk did not preserve eigenvalue-query semantics\n");
        return 1;
    }

    printf("[PASS] DGEEV Fortran->CBLAS thunk performs the workspace query and forwards double eigenvalue outputs\n");
    return 0;
}

static int check_dgeev_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_dgeev_fortran_slot_fn thunk = NULL;
    double a[4] = { 0.0 };
    double wr[2] = { 0.0, 0.0 };
    double wi[2] = { 0.0, 0.0 };
    double vl[4] = { 0.0 };
    double vr[4] = { 0.0 };
    double work[8] = { 0.0 };
    char jobvl = 'N';
    char jobvr = 'V';
    int n = 2;
    int lda = 2;
    int ldvl = 1;
    int ldvr = 2;
    int lwork = 8;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dgeev_cblas_call, 0, sizeof(g_dgeev_cblas_call));
    g_dgeev_cblas_rc = 202;

    vtable.ext_ops[FB_OP_DGEEV][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_dgeev_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_DGEEV);

    thunk = (fb_dgeev_fortran_slot_fn)vtable.ext_ops[FB_OP_DGEEV][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DGEEV CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&jobvl, &jobvr, &n, a, &lda, wr, wi, vl, &ldvl, vr, &ldvr, work, &lwork, &info);
    if (info != 202 || g_dgeev_cblas_call.called != 1 ||
        g_dgeev_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_dgeev_cblas_call.jobvl != 'N' || g_dgeev_cblas_call.jobvr != 'V' ||
        g_dgeev_cblas_call.n != 2 || g_dgeev_cblas_call.lda != 2 ||
        g_dgeev_cblas_call.ldvl != 1 || g_dgeev_cblas_call.ldvr != 2 ||
        g_dgeev_cblas_call.a != a || g_dgeev_cblas_call.wr != wr ||
        g_dgeev_cblas_call.wi != wi || g_dgeev_cblas_call.vl != vl ||
        g_dgeev_cblas_call.vr != vr) {
        fprintf(stderr, "[FAIL] DGEEV CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] DGEEV CBLAS->Fortran thunk maps the all-pointer ABI into the double C eigenvalue entry\n");
    return 0;
}

static int check_sgeev_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_sgeev_fn thunk = NULL;
    float a[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    float wr[2] = { 0.0f, 0.0f };
    float wi[2] = { 0.0f, 0.0f };
    float vl[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float vr[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sgeev_fortran_call, 0, sizeof(g_sgeev_fortran_call));

    vtable.ext_ops[FB_OP_SGEEV][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_sgeev_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SGEEV);

    thunk = (fb_sgeev_fn)vtable.ext_ops[FB_OP_SGEEV][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SGEEV Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 'V', 'V', 2, a, 2, wr, wi, vl, 2, vr, 2);
    if (info != 0 || g_sgeev_fortran_call.query_calls != 1 ||
        g_sgeev_fortran_call.exec_calls != 1 ||
        g_sgeev_fortran_call.query_lwork != -1 ||
        g_sgeev_fortran_call.exec_lwork != 23 ||
        g_sgeev_fortran_call.jobvl != 'V' || g_sgeev_fortran_call.jobvr != 'V' ||
        g_sgeev_fortran_call.n != 2 || g_sgeev_fortran_call.lda != 2 ||
        g_sgeev_fortran_call.ldvl != 2 || g_sgeev_fortran_call.ldvr != 2 ||
        wr[0] != 1.0f || wr[1] != 2.0f || wi[1] != 0.5f ||
        vl[0] != 10.0f || vl[3] != 13.0f || vr[0] != 20.0f || vr[3] != 23.0f) {
        fprintf(stderr, "[FAIL] SGEEV Fortran->CBLAS thunk did not preserve eigenvalue-query semantics\n");
        return 1;
    }

    printf("[PASS] SGEEV Fortran->CBLAS thunk performs the workspace query and forwards eigenvalue outputs\n");
    return 0;
}

static int check_sgeev_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_sgeev_fortran_slot_fn thunk = NULL;
    float a[4] = { 0.0f };
    float wr[2] = { 0.0f, 0.0f };
    float wi[2] = { 0.0f, 0.0f };
    float vl[4] = { 0.0f };
    float vr[4] = { 0.0f };
    float work[8] = { 0.0f };
    char jobvl = 'N';
    char jobvr = 'V';
    int n = 2;
    int lda = 2;
    int ldvl = 1;
    int ldvr = 2;
    int lwork = 8;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sgeev_cblas_call, 0, sizeof(g_sgeev_cblas_call));
    g_sgeev_cblas_rc = 201;

    vtable.ext_ops[FB_OP_SGEEV][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_sgeev_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SGEEV);

    thunk = (fb_sgeev_fortran_slot_fn)vtable.ext_ops[FB_OP_SGEEV][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SGEEV CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&jobvl, &jobvr, &n, a, &lda, wr, wi, vl, &ldvl, vr, &ldvr, work, &lwork, &info);
    if (info != 201 || g_sgeev_cblas_call.called != 1 ||
        g_sgeev_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_sgeev_cblas_call.jobvl != 'N' || g_sgeev_cblas_call.jobvr != 'V' ||
        g_sgeev_cblas_call.n != 2 || g_sgeev_cblas_call.lda != 2 ||
        g_sgeev_cblas_call.ldvl != 1 || g_sgeev_cblas_call.ldvr != 2 ||
        g_sgeev_cblas_call.a != a || g_sgeev_cblas_call.wr != wr ||
        g_sgeev_cblas_call.wi != wi || g_sgeev_cblas_call.vl != vl ||
        g_sgeev_cblas_call.vr != vr) {
        fprintf(stderr, "[FAIL] SGEEV CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] SGEEV CBLAS->Fortran thunk maps the all-pointer ABI into the generic C eigenvalue entry\n");
    return 0;
}

static int check_cgeev_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_cgeev_fn thunk = NULL;
    fb_complex_float_t a[4] = {
        make_cfloat(1.0f), make_cfloat(2.0f),
        make_cfloat(3.0f), make_cfloat(4.0f)
    };
    fb_complex_float_t w[2] = { 0 };
    fb_complex_float_t vl[4] = { 0 };
    fb_complex_float_t vr[4] = { 0 };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cgeev_fortran_call, 0, sizeof(g_cgeev_fortran_call));

    vtable.ext_ops[FB_OP_CGEEV][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_cgeev_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CGEEV);

    thunk = (fb_cgeev_fn)vtable.ext_ops[FB_OP_CGEEV][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CGEEV Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 'V', 'V', 2, a, 2, w, vl, 2, vr, 2);
    if (info != 0 || g_cgeev_fortran_call.query_calls != 1 ||
        g_cgeev_fortran_call.exec_calls != 1 ||
        g_cgeev_fortran_call.query_lwork != -1 ||
        g_cgeev_fortran_call.exec_lwork != 24 ||
        g_cgeev_fortran_call.jobvl != 'V' || g_cgeev_fortran_call.jobvr != 'V' ||
        g_cgeev_fortran_call.n != 2 || g_cgeev_fortran_call.lda != 2 ||
        g_cgeev_fortran_call.ldvl != 2 || g_cgeev_fortran_call.ldvr != 2 ||
        cfloat_real(w[0]) != 3.0f || cfloat_real(w[1]) != 4.0f ||
        cfloat_real(vl[0]) != 30.0f || cfloat_real(vl[3]) != 33.0f ||
        cfloat_real(vr[0]) != 40.0f || cfloat_real(vr[3]) != 43.0f) {
        fprintf(stderr, "[FAIL] CGEEV Fortran->CBLAS thunk did not preserve complex eigenvalue-query semantics\n");
        return 1;
    }

    printf("[PASS] CGEEV Fortran->CBLAS thunk performs the workspace query and forwards complex eigenvalue outputs\n");
    return 0;
}

static int check_cgeev_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_cgeev_fortran_slot_fn thunk = NULL;
    fb_complex_float_t a[4] = { 0 };
    fb_complex_float_t w[2] = { 0 };
    fb_complex_float_t vl[4] = { 0 };
    fb_complex_float_t vr[4] = { 0 };
    fb_complex_float_t work[8] = { 0 };
    float rwork[8] = { 0.0f };
    char jobvl = 'N';
    char jobvr = 'V';
    int n = 2;
    int lda = 2;
    int ldvl = 1;
    int ldvr = 2;
    int lwork = 8;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cgeev_cblas_call, 0, sizeof(g_cgeev_cblas_call));
    g_cgeev_cblas_rc = 203;

    vtable.ext_ops[FB_OP_CGEEV][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_cgeev_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CGEEV);

    thunk = (fb_cgeev_fortran_slot_fn)vtable.ext_ops[FB_OP_CGEEV][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CGEEV CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&jobvl, &jobvr, &n, a, &lda, w, vl, &ldvl, vr, &ldvr, work, &lwork, rwork, &info);
    if (info != 203 || g_cgeev_cblas_call.called != 1 ||
        g_cgeev_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_cgeev_cblas_call.jobvl != 'N' || g_cgeev_cblas_call.jobvr != 'V' ||
        g_cgeev_cblas_call.n != 2 || g_cgeev_cblas_call.lda != 2 ||
        g_cgeev_cblas_call.ldvl != 1 || g_cgeev_cblas_call.ldvr != 2 ||
        g_cgeev_cblas_call.a != a || g_cgeev_cblas_call.w != w ||
        g_cgeev_cblas_call.vl != vl || g_cgeev_cblas_call.vr != vr) {
        fprintf(stderr, "[FAIL] CGEEV CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] CGEEV CBLAS->Fortran thunk maps the all-pointer ABI into the generic complex eigenvalue entry\n");
    return 0;
}

static int check_zgeev_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_zgeev_fn thunk = NULL;
    fb_complex_double_t a[4] = {
        make_cdouble(1.0), make_cdouble(2.0),
        make_cdouble(3.0), make_cdouble(4.0)
    };
    fb_complex_double_t w[2] = { 0 };
    fb_complex_double_t vl[4] = { 0 };
    fb_complex_double_t vr[4] = { 0 };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zgeev_fortran_call, 0, sizeof(g_zgeev_fortran_call));

    vtable.ext_ops[FB_OP_ZGEEV][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_zgeev_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_ZGEEV);

    thunk = (fb_zgeev_fn)vtable.ext_ops[FB_OP_ZGEEV][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZGEEV Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 'V', 'V', 2, a, 2, w, vl, 2, vr, 2);
    if (info != 0 || g_zgeev_fortran_call.query_calls != 1 ||
        g_zgeev_fortran_call.exec_calls != 1 ||
        g_zgeev_fortran_call.query_lwork != -1 ||
        g_zgeev_fortran_call.exec_lwork != 34 ||
        g_zgeev_fortran_call.jobvl != 'V' || g_zgeev_fortran_call.jobvr != 'V' ||
        cdouble_real(w[0]) != 7.0 || cdouble_real(w[1]) != 8.0 ||
        cdouble_real(vl[0]) != 70.0 || cdouble_real(vl[3]) != 73.0 ||
        cdouble_real(vr[0]) != 80.0 || cdouble_real(vr[3]) != 83.0) {
        fprintf(stderr, "[FAIL] ZGEEV Fortran->CBLAS thunk did not preserve complex-double eigenvalue-query semantics\n");
        return 1;
    }

    printf("[PASS] ZGEEV Fortran->CBLAS thunk performs the workspace query and forwards complex-double eigenvalue outputs\n");
    return 0;
}

static int check_zgeev_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_zgeev_fortran_slot_fn thunk = NULL;
    fb_complex_double_t a[4] = { 0 };
    fb_complex_double_t w[2] = { 0 };
    fb_complex_double_t vl[4] = { 0 };
    fb_complex_double_t vr[4] = { 0 };
    fb_complex_double_t work[8] = { 0 };
    double rwork[8] = { 0.0 };
    char jobvl = 'N';
    char jobvr = 'V';
    int n = 2;
    int lda = 2;
    int ldvl = 1;
    int ldvr = 2;
    int lwork = 8;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zgeev_cblas_call, 0, sizeof(g_zgeev_cblas_call));
    g_zgeev_cblas_rc = 204;

    vtable.ext_ops[FB_OP_ZGEEV][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_zgeev_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_ZGEEV);

    thunk = (fb_zgeev_fortran_slot_fn)vtable.ext_ops[FB_OP_ZGEEV][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZGEEV CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&jobvl, &jobvr, &n, a, &lda, w, vl, &ldvl, vr, &ldvr, work, &lwork, rwork, &info);
    if (info != 204 || g_zgeev_cblas_call.called != 1 ||
        g_zgeev_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_zgeev_cblas_call.jobvl != 'N' || g_zgeev_cblas_call.jobvr != 'V' ||
        g_zgeev_cblas_call.n != 2 || g_zgeev_cblas_call.lda != 2 ||
        g_zgeev_cblas_call.ldvl != 1 || g_zgeev_cblas_call.ldvr != 2 ||
        g_zgeev_cblas_call.a != a || g_zgeev_cblas_call.w != w ||
        g_zgeev_cblas_call.vl != vl || g_zgeev_cblas_call.vr != vr) {
        fprintf(stderr, "[FAIL] ZGEEV CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] ZGEEV CBLAS->Fortran thunk maps the all-pointer ABI into the complex-double eigenvalue entry\n");
    return 0;
}

int main(void)
{
    if (check_sgeev_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_sgeev_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_dgeev_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_dgeev_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_cgeev_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_cgeev_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_zgeev_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_zgeev_cblas_to_fortran() != 0) {
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}