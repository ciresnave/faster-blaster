#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef int (*fb_ssyev_fn)(fb_layout_t layout, char jobz, fb_uplo_t uplo, int n,
                           float *a, int lda, float *w);
typedef int (*fb_cheev_fn)(fb_layout_t layout, char jobz, fb_uplo_t uplo, int n,
                           fb_complex_float_t *a, int lda, float *w);
typedef int (*fb_ssyevr_fn)(fb_layout_t layout, char jobz, char range,
                            fb_uplo_t uplo, int n, float *a, int lda, float vl,
                            float vu, int il, int iu, float abstol, int *m,
                            float *w, float *z, int ldz, int *isuppz);
typedef int (*fb_cheevr_fn)(fb_layout_t layout, char jobz, char range,
                            fb_uplo_t uplo, int n, fb_complex_float_t *a,
                            int lda, float vl, float vu, int il, int iu,
                            float abstol, int *m, float *w,
                            fb_complex_float_t *z, int ldz, int *isuppz);
typedef int (*fb_ssyevx_fn)(fb_layout_t layout, char jobz, char range,
                            fb_uplo_t uplo, int n, float *a, int lda, float vl,
                            float vu, int il, int iu, float abstol, int *m,
                            float *w, float *z, int ldz, int *ifail);
typedef int (*fb_cheevx_fn)(fb_layout_t layout, char jobz, char range,
                            fb_uplo_t uplo, int n, fb_complex_float_t *a,
                            int lda, float vl, float vu, int il, int iu,
                            float abstol, int *m, float *w,
                            fb_complex_float_t *z, int ldz, int *ifail);

typedef void (*fb_ssyev_fortran_slot_fn)(char *jobz, char *uplo, int *n,
                                         float *a, int *lda, float *w,
                                         float *work, int *lwork, int *info);
typedef void (*fb_cheev_fortran_slot_fn)(char *jobz, char *uplo, int *n,
                                         fb_complex_float_t *a, int *lda,
                                         float *w, fb_complex_float_t *work,
                                         int *lwork, float *rwork, int *info);
typedef void (*fb_ssyevr_fortran_slot_fn)(char *jobz, char *range, char *uplo,
                                          int *n, float *a, int *lda,
                                          float *vl, float *vu, int *il,
                                          int *iu, float *abstol, int *m,
                                          float *w, float *z, int *ldz,
                                          int *isuppz, float *work,
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
typedef void (*fb_ssyevx_fortran_slot_fn)(char *jobz, char *range, char *uplo,
                                          int *n, float *a, int *lda,
                                          float *vl, float *vu, int *il,
                                          int *iu, float *abstol, int *m,
                                          float *w, float *z, int *ldz,
                                          float *work, int *lwork, int *iwork,
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
} g_cheev_fortran_call;

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
    int exec_lrwork;
    int exec_liwork;
} g_cheevr_fortran_call;

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
} g_cheevx_fortran_call;

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
    float vl;
    float vu;
    int il;
    int iu;
    float abstol;
    int ldz;
} g_cheevx_cblas_call;

static int g_ssyev_cblas_rc = 0;
static int g_cheev_cblas_rc = 0;
static int g_ssyevr_cblas_rc = 0;
static int g_cheevr_cblas_rc = 0;
static int g_ssyevx_cblas_rc = 0;
static int g_cheevx_cblas_rc = 0;

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

int main(void)
{
    if (check_ssyev_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_ssyev_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_cheev_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_cheev_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_ssyevr_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_ssyevr_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_cheevr_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_cheevr_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_ssyevx_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_ssyevx_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_cheevx_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_cheevx_cblas_to_fortran() != 0) {
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}