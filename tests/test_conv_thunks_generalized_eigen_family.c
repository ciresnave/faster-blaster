#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef int (*fb_ssygv_fn)(fb_layout_t layout, int itype, char jobz, fb_uplo_t uplo,
                           int n, float *a, int lda, float *b, int ldb, float *w);
typedef int (*fb_chegv_fn)(fb_layout_t layout, int itype, char jobz, fb_uplo_t uplo,
                           int n, fb_complex_float_t *a, int lda,
                           fb_complex_float_t *b, int ldb, float *w);
typedef int (*fb_ssygvd_fn)(fb_layout_t layout, int itype, char jobz, fb_uplo_t uplo,
                            int n, float *a, int lda, float *b, int ldb, float *w);
typedef int (*fb_chegvd_fn)(fb_layout_t layout, int itype, char jobz, fb_uplo_t uplo,
                            int n, fb_complex_float_t *a, int lda,
                            fb_complex_float_t *b, int ldb, float *w);
typedef int (*fb_ssygvx_fn)(fb_layout_t layout, int itype, char jobz, char range,
                            fb_uplo_t uplo, int n, float *a, int lda,
                            float *b, int ldb, float vl, float vu, int il,
                            int iu, float abstol, int *m, float *w, float *z,
                            int ldz, int *ifail);
typedef int (*fb_chegvx_fn)(fb_layout_t layout, int itype, char jobz, char range,
                            fb_uplo_t uplo, int n, fb_complex_float_t *a,
                            int lda, fb_complex_float_t *b, int ldb, float vl,
                            float vu, int il, int iu, float abstol, int *m,
                            float *w, fb_complex_float_t *z, int ldz,
                            int *ifail);

typedef void (*fb_ssygv_fortran_slot_fn)(int *itype, char *jobz, char *uplo,
                                         int *n, float *a, int *lda, float *b,
                                         int *ldb, float *w, float *work,
                                         int *lwork, int *info);
typedef void (*fb_chegv_fortran_slot_fn)(int *itype, char *jobz, char *uplo,
                                         int *n, fb_complex_float_t *a,
                                         int *lda, fb_complex_float_t *b,
                                         int *ldb, float *w,
                                         fb_complex_float_t *work, int *lwork,
                                         float *rwork, int *info);
typedef void (*fb_ssygvd_fortran_slot_fn)(int *itype, char *jobz, char *uplo,
                                          int *n, float *a, int *lda, float *b,
                                          int *ldb, float *w, float *work,
                                          int *lwork, int *iwork,
                                          int *liwork, int *info);
typedef void (*fb_chegvd_fortran_slot_fn)(int *itype, char *jobz, char *uplo,
                                          int *n, fb_complex_float_t *a,
                                          int *lda, fb_complex_float_t *b,
                                          int *ldb, float *w,
                                          fb_complex_float_t *work, int *lwork,
                                          float *rwork, int *lrwork,
                                          int *iwork, int *liwork, int *info);
typedef void (*fb_ssygvx_fortran_slot_fn)(int *itype, char *jobz, char *range,
                                          char *uplo, int *n, float *a,
                                          int *lda, float *b, int *ldb,
                                          float *vl, float *vu, int *il,
                                          int *iu, float *abstol, int *m,
                                          float *w, float *z, int *ldz,
                                          float *work, int *lwork, int *iwork,
                                          int *ifail, int *info);
typedef void (*fb_chegvx_fortran_slot_fn)(int *itype, char *jobz, char *range,
                                          char *uplo, int *n,
                                          fb_complex_float_t *a, int *lda,
                                          fb_complex_float_t *b, int *ldb,
                                          float *vl, float *vu, int *il,
                                          int *iu, float *abstol, int *m,
                                          float *w, fb_complex_float_t *z,
                                          int *ldz, fb_complex_float_t *work,
                                          int *lwork, float *rwork,
                                          int *iwork, int *ifail, int *info);

static struct {
    int query_calls;
    int exec_calls;
    int exec_lwork;
} g_ssygv_fortran_call;

static struct {
    int query_calls;
    int exec_calls;
    int exec_lwork;
} g_chegv_fortran_call;

static struct {
    int query_calls;
    int exec_calls;
    int exec_lwork;
    int exec_liwork;
} g_ssygvd_fortran_call;

static struct {
    int query_calls;
    int exec_calls;
    int exec_lwork;
    int exec_lrwork;
    int exec_liwork;
} g_chegvd_fortran_call;

static struct {
    int query_calls;
    int exec_calls;
    int exec_lwork;
} g_ssygvx_fortran_call;

static struct {
    int query_calls;
    int exec_calls;
    int exec_lwork;
} g_chegvx_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    int itype;
    char jobz;
    fb_uplo_t uplo;
    int n;
    int lda;
    int ldb;
} g_ssygv_cblas_call;

static struct {
    int called;
    fb_layout_t layout;
    int itype;
    char jobz;
    fb_uplo_t uplo;
    int n;
    int lda;
    int ldb;
} g_chegv_cblas_call;

static struct {
    int called;
    fb_layout_t layout;
    int itype;
    char jobz;
    fb_uplo_t uplo;
    int n;
    int lda;
    int ldb;
} g_ssygvd_cblas_call;

static struct {
    int called;
    fb_layout_t layout;
    int itype;
    char jobz;
    fb_uplo_t uplo;
    int n;
    int lda;
    int ldb;
} g_chegvd_cblas_call;

static struct {
    int called;
    fb_layout_t layout;
    int itype;
    char jobz;
    char range;
    fb_uplo_t uplo;
    int n;
    int lda;
    int ldb;
    float vl;
    float vu;
    int il;
    int iu;
    float abstol;
    int ldz;
} g_ssygvx_cblas_call;

static struct {
    int called;
    fb_layout_t layout;
    int itype;
    char jobz;
    char range;
    fb_uplo_t uplo;
    int n;
    int lda;
    int ldb;
    float vl;
    float vu;
    int il;
    int iu;
    float abstol;
    int ldz;
} g_chegvx_cblas_call;

static int g_ssygv_cblas_rc = 0;
static int g_chegv_cblas_rc = 0;
static int g_ssygvd_cblas_rc = 0;
static int g_chegvd_cblas_rc = 0;
static int g_ssygvx_cblas_rc = 0;
static int g_chegvx_cblas_rc = 0;

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

static void stub_ssygv_fortran(int *itype, char *jobz, char *uplo, int *n,
                               float *a, int *lda, float *b, int *ldb,
                               float *w, float *work, int *lwork, int *info)
{
    (void)itype;
    (void)jobz;
    (void)uplo;
    (void)n;
    (void)a;
    (void)lda;
    (void)b;
    (void)ldb;
    if (*lwork == -1) {
        g_ssygv_fortran_call.query_calls += 1;
        work[0] = 41.0f;
        *info = 0;
        return;
    }

    g_ssygv_fortran_call.exec_calls += 1;
    g_ssygv_fortran_call.exec_lwork = *lwork;
    w[0] = 1.0f;
    w[1] = 2.0f;
    w[2] = 3.0f;
    *info = 0;
}

static void stub_chegv_fortran(int *itype, char *jobz, char *uplo, int *n,
                               fb_complex_float_t *a, int *lda,
                               fb_complex_float_t *b, int *ldb, float *w,
                               fb_complex_float_t *work, int *lwork,
                               float *rwork, int *info)
{
    (void)itype;
    (void)jobz;
    (void)uplo;
    (void)n;
    (void)a;
    (void)lda;
    (void)b;
    (void)ldb;
    (void)rwork;
    if (*lwork == -1) {
        g_chegv_fortran_call.query_calls += 1;
        work[0] = make_cfloat(42.0f);
        *info = 0;
        return;
    }

    g_chegv_fortran_call.exec_calls += 1;
    g_chegv_fortran_call.exec_lwork = *lwork;
    w[0] = 4.0f;
    w[1] = 5.0f;
    w[2] = 6.0f;
    *info = 0;
}

static void stub_ssygvd_fortran(int *itype, char *jobz, char *uplo, int *n,
                                float *a, int *lda, float *b, int *ldb,
                                float *w, float *work, int *lwork,
                                int *iwork, int *liwork, int *info)
{
    (void)itype;
    (void)jobz;
    (void)uplo;
    (void)n;
    (void)a;
    (void)lda;
    (void)b;
    (void)ldb;
    (void)iwork;
    if (*lwork == -1) {
        g_ssygvd_fortran_call.query_calls += 1;
        work[0] = 43.0f;
        *iwork = 15;
        *info = 0;
        return;
    }

    g_ssygvd_fortran_call.exec_calls += 1;
    g_ssygvd_fortran_call.exec_lwork = *lwork;
    g_ssygvd_fortran_call.exec_liwork = *liwork;
    w[0] = 7.0f;
    w[1] = 8.0f;
    w[2] = 9.0f;
    *info = 0;
}

static void stub_chegvd_fortran(int *itype, char *jobz, char *uplo, int *n,
                                fb_complex_float_t *a, int *lda,
                                fb_complex_float_t *b, int *ldb, float *w,
                                fb_complex_float_t *work, int *lwork,
                                float *rwork, int *lrwork, int *iwork,
                                int *liwork, int *info)
{
    (void)itype;
    (void)jobz;
    (void)uplo;
    (void)n;
    (void)a;
    (void)lda;
    (void)b;
    (void)ldb;
    (void)iwork;
    if (*lwork == -1) {
        g_chegvd_fortran_call.query_calls += 1;
        work[0] = make_cfloat(44.0f);
        *rwork = 21.0f;
        *iwork = 17;
        *info = 0;
        return;
    }

    g_chegvd_fortran_call.exec_calls += 1;
    g_chegvd_fortran_call.exec_lwork = *lwork;
    g_chegvd_fortran_call.exec_lrwork = *lrwork;
    g_chegvd_fortran_call.exec_liwork = *liwork;
    w[0] = 10.0f;
    w[1] = 11.0f;
    w[2] = 12.0f;
    *info = 0;
}

static void stub_ssygvx_fortran(int *itype, char *jobz, char *range,
                                char *uplo, int *n, float *a, int *lda,
                                float *b, int *ldb, float *vl, float *vu,
                                int *il, int *iu, float *abstol, int *m,
                                float *w, float *z, int *ldz, float *work,
                                int *lwork, int *iwork, int *ifail, int *info)
{
    (void)itype;
    (void)jobz;
    (void)range;
    (void)uplo;
    (void)n;
    (void)a;
    (void)lda;
    (void)b;
    (void)ldb;
    (void)vl;
    (void)vu;
    (void)il;
    (void)iu;
    (void)abstol;
    (void)ldz;
    (void)iwork;
    if (*lwork == -1) {
        g_ssygvx_fortran_call.query_calls += 1;
        work[0] = 45.0f;
        *info = 0;
        return;
    }

    g_ssygvx_fortran_call.exec_calls += 1;
    g_ssygvx_fortran_call.exec_lwork = *lwork;
    *m = 2;
    w[0] = 13.0f;
    w[1] = 14.0f;
    z[0] = 101.0f;
    ifail[0] = 2;
    ifail[1] = 0;
    *info = 0;
}

static void stub_chegvx_fortran(int *itype, char *jobz, char *range,
                                char *uplo, int *n, fb_complex_float_t *a,
                                int *lda, fb_complex_float_t *b, int *ldb,
                                float *vl, float *vu, int *il, int *iu,
                                float *abstol, int *m, float *w,
                                fb_complex_float_t *z, int *ldz,
                                fb_complex_float_t *work, int *lwork,
                                float *rwork, int *iwork, int *ifail,
                                int *info)
{
    (void)itype;
    (void)jobz;
    (void)range;
    (void)uplo;
    (void)n;
    (void)a;
    (void)lda;
    (void)b;
    (void)ldb;
    (void)vl;
    (void)vu;
    (void)il;
    (void)iu;
    (void)abstol;
    (void)ldz;
    (void)rwork;
    (void)iwork;
    if (*lwork == -1) {
        g_chegvx_fortran_call.query_calls += 1;
        work[0] = make_cfloat(46.0f);
        *info = 0;
        return;
    }

    g_chegvx_fortran_call.exec_calls += 1;
    g_chegvx_fortran_call.exec_lwork = *lwork;
    *m = 2;
    w[0] = 15.0f;
    w[1] = 16.0f;
    z[0] = make_cfloat(102.0f);
    ifail[0] = 3;
    ifail[1] = 0;
    *info = 0;
}

static int stub_ssygv_cblas(fb_layout_t layout, int itype, char jobz,
                            fb_uplo_t uplo, int n, float *a, int lda, float *b,
                            int ldb, float *w)
{
    (void)a;
    (void)b;
    (void)w;
    g_ssygv_cblas_call.called += 1;
    g_ssygv_cblas_call.layout = layout;
    g_ssygv_cblas_call.itype = itype;
    g_ssygv_cblas_call.jobz = jobz;
    g_ssygv_cblas_call.uplo = uplo;
    g_ssygv_cblas_call.n = n;
    g_ssygv_cblas_call.lda = lda;
    g_ssygv_cblas_call.ldb = ldb;
    return g_ssygv_cblas_rc;
}

static int stub_chegv_cblas(fb_layout_t layout, int itype, char jobz,
                            fb_uplo_t uplo, int n, fb_complex_float_t *a,
                            int lda, fb_complex_float_t *b, int ldb, float *w)
{
    (void)a;
    (void)b;
    (void)w;
    g_chegv_cblas_call.called += 1;
    g_chegv_cblas_call.layout = layout;
    g_chegv_cblas_call.itype = itype;
    g_chegv_cblas_call.jobz = jobz;
    g_chegv_cblas_call.uplo = uplo;
    g_chegv_cblas_call.n = n;
    g_chegv_cblas_call.lda = lda;
    g_chegv_cblas_call.ldb = ldb;
    return g_chegv_cblas_rc;
}

static int stub_ssygvd_cblas(fb_layout_t layout, int itype, char jobz,
                             fb_uplo_t uplo, int n, float *a, int lda,
                             float *b, int ldb, float *w)
{
    (void)a;
    (void)b;
    (void)w;
    g_ssygvd_cblas_call.called += 1;
    g_ssygvd_cblas_call.layout = layout;
    g_ssygvd_cblas_call.itype = itype;
    g_ssygvd_cblas_call.jobz = jobz;
    g_ssygvd_cblas_call.uplo = uplo;
    g_ssygvd_cblas_call.n = n;
    g_ssygvd_cblas_call.lda = lda;
    g_ssygvd_cblas_call.ldb = ldb;
    return g_ssygvd_cblas_rc;
}

static int stub_chegvd_cblas(fb_layout_t layout, int itype, char jobz,
                             fb_uplo_t uplo, int n, fb_complex_float_t *a,
                             int lda, fb_complex_float_t *b, int ldb, float *w)
{
    (void)a;
    (void)b;
    (void)w;
    g_chegvd_cblas_call.called += 1;
    g_chegvd_cblas_call.layout = layout;
    g_chegvd_cblas_call.itype = itype;
    g_chegvd_cblas_call.jobz = jobz;
    g_chegvd_cblas_call.uplo = uplo;
    g_chegvd_cblas_call.n = n;
    g_chegvd_cblas_call.lda = lda;
    g_chegvd_cblas_call.ldb = ldb;
    return g_chegvd_cblas_rc;
}

static int stub_ssygvx_cblas(fb_layout_t layout, int itype, char jobz,
                             char range, fb_uplo_t uplo, int n, float *a,
                             int lda, float *b, int ldb, float vl, float vu,
                             int il, int iu, float abstol, int *m, float *w,
                             float *z, int ldz, int *ifail)
{
    (void)a;
    (void)b;
    g_ssygvx_cblas_call.called += 1;
    g_ssygvx_cblas_call.layout = layout;
    g_ssygvx_cblas_call.itype = itype;
    g_ssygvx_cblas_call.jobz = jobz;
    g_ssygvx_cblas_call.range = range;
    g_ssygvx_cblas_call.uplo = uplo;
    g_ssygvx_cblas_call.n = n;
    g_ssygvx_cblas_call.lda = lda;
    g_ssygvx_cblas_call.ldb = ldb;
    g_ssygvx_cblas_call.vl = vl;
    g_ssygvx_cblas_call.vu = vu;
    g_ssygvx_cblas_call.il = il;
    g_ssygvx_cblas_call.iu = iu;
    g_ssygvx_cblas_call.abstol = abstol;
    g_ssygvx_cblas_call.ldz = ldz;
    *m = 2;
    w[0] = 17.0f;
    w[1] = 18.0f;
    z[0] = 103.0f;
    ifail[0] = 4;
    ifail[1] = 0;
    return g_ssygvx_cblas_rc;
}

static int stub_chegvx_cblas(fb_layout_t layout, int itype, char jobz,
                             char range, fb_uplo_t uplo, int n,
                             fb_complex_float_t *a, int lda,
                             fb_complex_float_t *b, int ldb, float vl,
                             float vu, int il, int iu, float abstol, int *m,
                             float *w, fb_complex_float_t *z, int ldz,
                             int *ifail)
{
    (void)a;
    (void)b;
    g_chegvx_cblas_call.called += 1;
    g_chegvx_cblas_call.layout = layout;
    g_chegvx_cblas_call.itype = itype;
    g_chegvx_cblas_call.jobz = jobz;
    g_chegvx_cblas_call.range = range;
    g_chegvx_cblas_call.uplo = uplo;
    g_chegvx_cblas_call.n = n;
    g_chegvx_cblas_call.lda = lda;
    g_chegvx_cblas_call.ldb = ldb;
    g_chegvx_cblas_call.vl = vl;
    g_chegvx_cblas_call.vu = vu;
    g_chegvx_cblas_call.il = il;
    g_chegvx_cblas_call.iu = iu;
    g_chegvx_cblas_call.abstol = abstol;
    g_chegvx_cblas_call.ldz = ldz;
    *m = 2;
    w[0] = 19.0f;
    w[1] = 20.0f;
    z[0] = make_cfloat(104.0f);
    ifail[0] = 5;
    ifail[1] = 0;
    return g_chegvx_cblas_rc;
}

static int check_ssygv_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_ssygv_fn thunk = NULL;
    float a[9] = { 0.0f };
    float b[9] = { 0.0f };
    float w[3] = { 0.0f, 0.0f, 0.0f };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_ssygv_fortran_call, 0, sizeof(g_ssygv_fortran_call));
    vtable.ext_ops[FB_OP_SSYGV][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_ssygv_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SSYGV);

    thunk = (fb_ssygv_fn)vtable.ext_ops[FB_OP_SSYGV][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SSYGV Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 2, 'V', FB_LOWER, 3, a, 3, b, 3, w);
    if (info != 0 || g_ssygv_fortran_call.query_calls != 1 ||
        g_ssygv_fortran_call.exec_calls != 1 ||
        g_ssygv_fortran_call.exec_lwork != 41 ||
        w[0] != 1.0f || w[1] != 2.0f || w[2] != 3.0f) {
        fprintf(stderr, "[FAIL] SSYGV Fortran->CBLAS thunk did not preserve generalized symmetric-eigen query semantics\n");
        return 1;
    }

    printf("[PASS] SSYGV Fortran->CBLAS thunk performs the workspace query and forwards generalized symmetric eigenvalues\n");
    return 0;
}

static int check_ssygv_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_ssygv_fortran_slot_fn thunk = NULL;
    float a[9] = { 0.0f };
    float b[9] = { 0.0f };
    float w[3] = { 0.0f, 0.0f, 0.0f };
    float work[8] = { 0.0f };
    int itype = 2;
    char jobz = 'V';
    char uplo = 'L';
    int n = 3;
    int lda = 3;
    int ldb = 3;
    int lwork = 8;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_ssygv_cblas_call, 0, sizeof(g_ssygv_cblas_call));
    g_ssygv_cblas_rc = 311;
    vtable.ext_ops[FB_OP_SSYGV][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_ssygv_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SSYGV);

    thunk = (fb_ssygv_fortran_slot_fn)vtable.ext_ops[FB_OP_SSYGV][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SSYGV CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, work, &lwork, &info);
    if (info != 311 || g_ssygv_cblas_call.called != 1 ||
        g_ssygv_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_ssygv_cblas_call.itype != 2 || g_ssygv_cblas_call.jobz != 'V' ||
        g_ssygv_cblas_call.uplo != FB_LOWER || g_ssygv_cblas_call.n != 3 ||
        g_ssygv_cblas_call.lda != 3 || g_ssygv_cblas_call.ldb != 3) {
        fprintf(stderr, "[FAIL] SSYGV CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] SSYGV CBLAS->Fortran thunk maps Fortran UPLO chars into the generalized C symmetric-eigen entry\n");
    return 0;
}

static int check_chegv_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_chegv_fn thunk = NULL;
    fb_complex_float_t a[9] = { 0 };
    fb_complex_float_t b[9] = { 0 };
    float w[3] = { 0.0f, 0.0f, 0.0f };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_chegv_fortran_call, 0, sizeof(g_chegv_fortran_call));
    vtable.ext_ops[FB_OP_CHEGV][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_chegv_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CHEGV);

    thunk = (fb_chegv_fn)vtable.ext_ops[FB_OP_CHEGV][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CHEGV Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 1, 'V', FB_UPPER, 3, a, 3, b, 3, w);
    if (info != 0 || g_chegv_fortran_call.query_calls != 1 ||
        g_chegv_fortran_call.exec_calls != 1 ||
        g_chegv_fortran_call.exec_lwork != 42 ||
        w[0] != 4.0f || w[1] != 5.0f || w[2] != 6.0f) {
        fprintf(stderr, "[FAIL] CHEGV Fortran->CBLAS thunk did not preserve generalized Hermitian-eigen query semantics\n");
        return 1;
    }

    printf("[PASS] CHEGV Fortran->CBLAS thunk performs the workspace query and forwards generalized Hermitian eigenvalues\n");
    return 0;
}

static int check_chegv_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_chegv_fortran_slot_fn thunk = NULL;
    fb_complex_float_t a[9] = { 0 };
    fb_complex_float_t b[9] = { 0 };
    float w[3] = { 0.0f, 0.0f, 0.0f };
    fb_complex_float_t work[8] = { 0 };
    float rwork[16] = { 0.0f };
    int itype = 3;
    char jobz = 'V';
    char uplo = 'U';
    int n = 3;
    int lda = 3;
    int ldb = 3;
    int lwork = 8;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_chegv_cblas_call, 0, sizeof(g_chegv_cblas_call));
    g_chegv_cblas_rc = 313;
    vtable.ext_ops[FB_OP_CHEGV][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_chegv_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CHEGV);

    thunk = (fb_chegv_fortran_slot_fn)vtable.ext_ops[FB_OP_CHEGV][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CHEGV CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, work, &lwork, rwork, &info);
    if (info != 313 || g_chegv_cblas_call.called != 1 ||
        g_chegv_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_chegv_cblas_call.itype != 3 || g_chegv_cblas_call.jobz != 'V' ||
        g_chegv_cblas_call.uplo != FB_UPPER || g_chegv_cblas_call.n != 3 ||
        g_chegv_cblas_call.lda != 3 || g_chegv_cblas_call.ldb != 3) {
        fprintf(stderr, "[FAIL] CHEGV CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] CHEGV CBLAS->Fortran thunk maps Fortran UPLO chars into the generalized C Hermitian-eigen entry\n");
    return 0;
}

static int check_csygv_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_chegv_fn thunk = NULL;
    fb_complex_float_t a[9] = { 0 };
    fb_complex_float_t b[9] = { 0 };
    float w[3] = { 0.0f, 0.0f, 0.0f };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_chegv_fortran_call, 0, sizeof(g_chegv_fortran_call));
    vtable.ext_ops[FB_OP_CSYGV][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_chegv_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CSYGV);

    thunk = (fb_chegv_fn)vtable.ext_ops[FB_OP_CSYGV][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CSYGV Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 1, 'V', FB_UPPER, 3, a, 3, b, 3, w);
    if (info != 0 || g_chegv_fortran_call.query_calls != 1 ||
        g_chegv_fortran_call.exec_calls != 1 ||
        g_chegv_fortran_call.exec_lwork != 42 ||
        w[0] != 4.0f || w[1] != 5.0f || w[2] != 6.0f) {
        fprintf(stderr, "[FAIL] CSYGV Fortran->CBLAS thunk did not preserve generalized complex-symmetric eigen query semantics\n");
        return 1;
    }

    printf("[PASS] CSYGV Fortran->CBLAS thunk performs the workspace query and forwards generalized complex-symmetric eigenvalues\n");
    return 0;
}

static int check_csygv_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_chegv_fortran_slot_fn thunk = NULL;
    fb_complex_float_t a[9] = { 0 };
    fb_complex_float_t b[9] = { 0 };
    float w[3] = { 0.0f, 0.0f, 0.0f };
    fb_complex_float_t work[8] = { 0 };
    float rwork[16] = { 0.0f };
    int itype = 3;
    char jobz = 'V';
    char uplo = 'U';
    int n = 3;
    int lda = 3;
    int ldb = 3;
    int lwork = 8;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_chegv_cblas_call, 0, sizeof(g_chegv_cblas_call));
    g_chegv_cblas_rc = 323;
    vtable.ext_ops[FB_OP_CSYGV][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_chegv_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CSYGV);

    thunk = (fb_chegv_fortran_slot_fn)vtable.ext_ops[FB_OP_CSYGV][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CSYGV CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, work, &lwork, rwork, &info);
    if (info != 323 || g_chegv_cblas_call.called != 1 ||
        g_chegv_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_chegv_cblas_call.itype != 3 || g_chegv_cblas_call.jobz != 'V' ||
        g_chegv_cblas_call.uplo != FB_UPPER || g_chegv_cblas_call.n != 3 ||
        g_chegv_cblas_call.lda != 3 || g_chegv_cblas_call.ldb != 3) {
        fprintf(stderr, "[FAIL] CSYGV CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] CSYGV CBLAS->Fortran thunk maps Fortran UPLO chars into the generalized C complex-symmetric eigen entry\n");
    return 0;
}

static int check_ssygvx_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_ssygvx_fn thunk = NULL;
    float a[9] = { 0.0f };
    float b[9] = { 0.0f };
    float w[3] = { 0.0f, 0.0f, 0.0f };
    float z[9] = { 0.0f };
    int ifail[3] = { 0, 0, 0 };
    int m = 0;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_ssygvx_fortran_call, 0, sizeof(g_ssygvx_fortran_call));
    vtable.ext_ops[FB_OP_SSYGVX][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_ssygvx_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SSYGVX);

    thunk = (fb_ssygvx_fn)vtable.ext_ops[FB_OP_SSYGVX][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SSYGVX Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 1, 'V', 'I', FB_LOWER, 3, a, 3, b, 3,
                 1.25f, 9.5f, 1, 2, 0.25f, &m, w, z, 3, ifail);
    if (info != 0 || g_ssygvx_fortran_call.query_calls != 1 ||
        g_ssygvx_fortran_call.exec_calls != 1 ||
        g_ssygvx_fortran_call.exec_lwork != 45 || m != 2 ||
        w[0] != 13.0f || w[1] != 14.0f || z[0] != 101.0f || ifail[0] != 2) {
        fprintf(stderr, "[FAIL] SSYGVX Fortran->CBLAS thunk did not preserve selective generalized symmetric-eigen query semantics\n");
        return 1;
    }

    printf("[PASS] SSYGVX Fortran->CBLAS thunk performs the workspace query and forwards selective generalized symmetric eigen outputs\n");
    return 0;
}

static int check_ssygvx_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_ssygvx_fortran_slot_fn thunk = NULL;
    float a[9] = { 0.0f };
    float b[9] = { 0.0f };
    float w[3] = { 0.0f, 0.0f, 0.0f };
    float z[9] = { 0.0f };
    float work[16] = { 0.0f };
    int iwork[16] = { 0 };
    int ifail[3] = { 0, 0, 0 };
    int itype = 2;
    char jobz = 'V';
    char range = 'I';
    char uplo = 'L';
    int n = 3;
    int lda = 3;
    int ldb = 3;
    float vl = 1.25f;
    float vu = 9.5f;
    int il = 1;
    int iu = 2;
    float abstol = 0.25f;
    int m = 0;
    int ldz = 3;
    int lwork = 16;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_ssygvx_cblas_call, 0, sizeof(g_ssygvx_cblas_call));
    g_ssygvx_cblas_rc = 331;
    vtable.ext_ops[FB_OP_SSYGVX][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_ssygvx_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SSYGVX);

    thunk = (fb_ssygvx_fortran_slot_fn)vtable.ext_ops[FB_OP_SSYGVX][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SSYGVX CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&itype, &jobz, &range, &uplo, &n, a, &lda, b, &ldb, &vl, &vu, &il,
          &iu, &abstol, &m, w, z, &ldz, work, &lwork, iwork, ifail, &info);
    if (info != 331 || g_ssygvx_cblas_call.called != 1 ||
        g_ssygvx_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_ssygvx_cblas_call.itype != 2 || g_ssygvx_cblas_call.jobz != 'V' ||
        g_ssygvx_cblas_call.range != 'I' ||
        g_ssygvx_cblas_call.uplo != FB_LOWER || g_ssygvx_cblas_call.n != 3 ||
        g_ssygvx_cblas_call.lda != 3 || g_ssygvx_cblas_call.ldb != 3 ||
        g_ssygvx_cblas_call.vl != 1.25f || g_ssygvx_cblas_call.vu != 9.5f ||
        g_ssygvx_cblas_call.il != 1 || g_ssygvx_cblas_call.iu != 2 ||
        g_ssygvx_cblas_call.abstol != 0.25f ||
        g_ssygvx_cblas_call.ldz != 3 || m != 2 || w[0] != 17.0f ||
        w[1] != 18.0f || z[0] != 103.0f || ifail[0] != 4) {
        fprintf(stderr, "[FAIL] SSYGVX CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] SSYGVX CBLAS->Fortran thunk maps selective generalized symmetric-eigen arguments into the C entry\n");
    return 0;
}

static int check_chegvx_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_chegvx_fn thunk = NULL;
    fb_complex_float_t a[9] = { 0 };
    fb_complex_float_t b[9] = { 0 };
    fb_complex_float_t z[9] = { 0 };
    float w[3] = { 0.0f, 0.0f, 0.0f };
    int ifail[3] = { 0, 0, 0 };
    int m = 0;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_chegvx_fortran_call, 0, sizeof(g_chegvx_fortran_call));
    vtable.ext_ops[FB_OP_CHEGVX][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_chegvx_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CHEGVX);

    thunk = (fb_chegvx_fn)vtable.ext_ops[FB_OP_CHEGVX][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CHEGVX Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 3, 'V', 'I', FB_UPPER, 3, a, 3, b, 3,
                 2.0f, 8.0f, 1, 2, 0.5f, &m, w, z, 3, ifail);
    if (info != 0 || g_chegvx_fortran_call.query_calls != 1 ||
        g_chegvx_fortran_call.exec_calls != 1 ||
        g_chegvx_fortran_call.exec_lwork != 46 || m != 2 ||
        w[0] != 15.0f || w[1] != 16.0f ||
        cfloat_real(z[0]) != 102.0f || ifail[0] != 3) {
        fprintf(stderr, "[FAIL] CHEGVX Fortran->CBLAS thunk did not preserve selective generalized Hermitian-eigen query semantics\n");
        return 1;
    }

    printf("[PASS] CHEGVX Fortran->CBLAS thunk performs the workspace query and forwards selective generalized Hermitian eigen outputs\n");
    return 0;
}

static int check_chegvx_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_chegvx_fortran_slot_fn thunk = NULL;
    fb_complex_float_t a[9] = { 0 };
    fb_complex_float_t b[9] = { 0 };
    fb_complex_float_t z[9] = { 0 };
    fb_complex_float_t work[16] = { 0 };
    float w[3] = { 0.0f, 0.0f, 0.0f };
    float rwork[24] = { 0.0f };
    int iwork[16] = { 0 };
    int ifail[3] = { 0, 0, 0 };
    int itype = 3;
    char jobz = 'V';
    char range = 'I';
    char uplo = 'U';
    int n = 3;
    int lda = 3;
    int ldb = 3;
    float vl = 2.0f;
    float vu = 8.0f;
    int il = 1;
    int iu = 2;
    float abstol = 0.5f;
    int m = 0;
    int ldz = 3;
    int lwork = 16;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_chegvx_cblas_call, 0, sizeof(g_chegvx_cblas_call));
    g_chegvx_cblas_rc = 333;
    vtable.ext_ops[FB_OP_CHEGVX][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_chegvx_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CHEGVX);

    thunk = (fb_chegvx_fortran_slot_fn)vtable.ext_ops[FB_OP_CHEGVX][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CHEGVX CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&itype, &jobz, &range, &uplo, &n, a, &lda, b, &ldb, &vl, &vu, &il,
          &iu, &abstol, &m, w, z, &ldz, work, &lwork, rwork, iwork, ifail,
          &info);
    if (info != 333 || g_chegvx_cblas_call.called != 1 ||
        g_chegvx_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_chegvx_cblas_call.itype != 3 || g_chegvx_cblas_call.jobz != 'V' ||
        g_chegvx_cblas_call.range != 'I' ||
        g_chegvx_cblas_call.uplo != FB_UPPER || g_chegvx_cblas_call.n != 3 ||
        g_chegvx_cblas_call.lda != 3 || g_chegvx_cblas_call.ldb != 3 ||
        g_chegvx_cblas_call.vl != 2.0f || g_chegvx_cblas_call.vu != 8.0f ||
        g_chegvx_cblas_call.il != 1 || g_chegvx_cblas_call.iu != 2 ||
        g_chegvx_cblas_call.abstol != 0.5f || g_chegvx_cblas_call.ldz != 3 ||
        m != 2 || w[0] != 19.0f || w[1] != 20.0f ||
        cfloat_real(z[0]) != 104.0f || ifail[0] != 5) {
        fprintf(stderr, "[FAIL] CHEGVX CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] CHEGVX CBLAS->Fortran thunk maps selective generalized Hermitian-eigen arguments into the C entry\n");
    return 0;
}

static int check_csygvx_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_chegvx_fn thunk = NULL;
    fb_complex_float_t a[9] = { 0 };
    fb_complex_float_t b[9] = { 0 };
    fb_complex_float_t z[9] = { 0 };
    float w[3] = { 0.0f, 0.0f, 0.0f };
    int ifail[3] = { 0, 0, 0 };
    int m = 0;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_chegvx_fortran_call, 0, sizeof(g_chegvx_fortran_call));
    vtable.ext_ops[FB_OP_CSYGVX][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_chegvx_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CSYGVX);

    thunk = (fb_chegvx_fn)vtable.ext_ops[FB_OP_CSYGVX][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CSYGVX Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 3, 'V', 'I', FB_UPPER, 3, a, 3, b, 3,
                 2.0f, 8.0f, 1, 2, 0.5f, &m, w, z, 3, ifail);
    if (info != 0 || g_chegvx_fortran_call.query_calls != 1 ||
        g_chegvx_fortran_call.exec_calls != 1 ||
        g_chegvx_fortran_call.exec_lwork != 46 || m != 2 ||
        w[0] != 15.0f || w[1] != 16.0f ||
        cfloat_real(z[0]) != 102.0f || ifail[0] != 3) {
        fprintf(stderr, "[FAIL] CSYGVX Fortran->CBLAS thunk did not preserve selective generalized complex-symmetric eigen query semantics\n");
        return 1;
    }

    printf("[PASS] CSYGVX Fortran->CBLAS thunk performs the workspace query and forwards selective generalized complex-symmetric eigen outputs\n");
    return 0;
}

static int check_csygvx_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_chegvx_fortran_slot_fn thunk = NULL;
    fb_complex_float_t a[9] = { 0 };
    fb_complex_float_t b[9] = { 0 };
    fb_complex_float_t z[9] = { 0 };
    fb_complex_float_t work[16] = { 0 };
    float w[3] = { 0.0f, 0.0f, 0.0f };
    float rwork[24] = { 0.0f };
    int iwork[16] = { 0 };
    int ifail[3] = { 0, 0, 0 };
    int itype = 3;
    char jobz = 'V';
    char range = 'I';
    char uplo = 'U';
    int n = 3;
    int lda = 3;
    int ldb = 3;
    float vl = 2.0f;
    float vu = 8.0f;
    int il = 1;
    int iu = 2;
    float abstol = 0.5f;
    int m = 0;
    int ldz = 3;
    int lwork = 16;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_chegvx_cblas_call, 0, sizeof(g_chegvx_cblas_call));
    g_chegvx_cblas_rc = 341;
    vtable.ext_ops[FB_OP_CSYGVX][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_chegvx_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CSYGVX);

    thunk = (fb_chegvx_fortran_slot_fn)vtable.ext_ops[FB_OP_CSYGVX][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CSYGVX CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&itype, &jobz, &range, &uplo, &n, a, &lda, b, &ldb, &vl, &vu, &il,
          &iu, &abstol, &m, w, z, &ldz, work, &lwork, rwork, iwork, ifail,
          &info);
    if (info != 341 || g_chegvx_cblas_call.called != 1 ||
        g_chegvx_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_chegvx_cblas_call.itype != 3 || g_chegvx_cblas_call.jobz != 'V' ||
        g_chegvx_cblas_call.range != 'I' ||
        g_chegvx_cblas_call.uplo != FB_UPPER || g_chegvx_cblas_call.n != 3 ||
        g_chegvx_cblas_call.lda != 3 || g_chegvx_cblas_call.ldb != 3 ||
        g_chegvx_cblas_call.vl != 2.0f || g_chegvx_cblas_call.vu != 8.0f ||
        g_chegvx_cblas_call.il != 1 || g_chegvx_cblas_call.iu != 2 ||
        g_chegvx_cblas_call.abstol != 0.5f || g_chegvx_cblas_call.ldz != 3 ||
        m != 2 || w[0] != 19.0f || w[1] != 20.0f ||
        cfloat_real(z[0]) != 104.0f || ifail[0] != 5) {
        fprintf(stderr, "[FAIL] CSYGVX CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] CSYGVX CBLAS->Fortran thunk maps selective generalized complex-symmetric eigen arguments into the C entry\n");
    return 0;
}

static int check_ssygvd_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_ssygvd_fn thunk = NULL;
    float a[9] = { 0.0f };
    float b[9] = { 0.0f };
    float w[3] = { 0.0f, 0.0f, 0.0f };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_ssygvd_fortran_call, 0, sizeof(g_ssygvd_fortran_call));
    vtable.ext_ops[FB_OP_SSYGVD][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_ssygvd_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SSYGVD);

    thunk = (fb_ssygvd_fn)vtable.ext_ops[FB_OP_SSYGVD][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SSYGVD Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 1, 'V', FB_LOWER, 3, a, 3, b, 3, w);
    if (info != 0 || g_ssygvd_fortran_call.query_calls != 1 ||
        g_ssygvd_fortran_call.exec_calls != 1 ||
        g_ssygvd_fortran_call.exec_lwork != 43 ||
        g_ssygvd_fortran_call.exec_liwork != 15 ||
        w[0] != 7.0f || w[1] != 8.0f || w[2] != 9.0f) {
        fprintf(stderr, "[FAIL] SSYGVD Fortran->CBLAS thunk did not preserve divide-and-conquer generalized symmetric-eigen query semantics\n");
        return 1;
    }

    printf("[PASS] SSYGVD Fortran->CBLAS thunk performs the workspace query and forwards divide-and-conquer generalized symmetric eigenvalues\n");
    return 0;
}

static int check_ssygvd_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_ssygvd_fortran_slot_fn thunk = NULL;
    float a[9] = { 0.0f };
    float b[9] = { 0.0f };
    float w[3] = { 0.0f, 0.0f, 0.0f };
    float work[8] = { 0.0f };
    int iwork[8] = { 0 };
    int itype = 1;
    char jobz = 'V';
    char uplo = 'L';
    int n = 3;
    int lda = 3;
    int ldb = 3;
    int lwork = 8;
    int liwork = 8;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_ssygvd_cblas_call, 0, sizeof(g_ssygvd_cblas_call));
    g_ssygvd_cblas_rc = 317;
    vtable.ext_ops[FB_OP_SSYGVD][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_ssygvd_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SSYGVD);

    thunk = (fb_ssygvd_fortran_slot_fn)vtable.ext_ops[FB_OP_SSYGVD][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SSYGVD CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, work, &lwork, iwork, &liwork, &info);
    if (info != 317 || g_ssygvd_cblas_call.called != 1 ||
        g_ssygvd_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_ssygvd_cblas_call.itype != 1 || g_ssygvd_cblas_call.jobz != 'V' ||
        g_ssygvd_cblas_call.uplo != FB_LOWER || g_ssygvd_cblas_call.n != 3 ||
        g_ssygvd_cblas_call.lda != 3 || g_ssygvd_cblas_call.ldb != 3) {
        fprintf(stderr, "[FAIL] SSYGVD CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] SSYGVD CBLAS->Fortran thunk maps Fortran UPLO chars into the divide-and-conquer generalized C symmetric-eigen entry\n");
    return 0;
}

static int check_chegvd_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_chegvd_fn thunk = NULL;
    fb_complex_float_t a[9] = { 0 };
    fb_complex_float_t b[9] = { 0 };
    float w[3] = { 0.0f, 0.0f, 0.0f };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_chegvd_fortran_call, 0, sizeof(g_chegvd_fortran_call));
    vtable.ext_ops[FB_OP_CHEGVD][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_chegvd_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CHEGVD);

    thunk = (fb_chegvd_fn)vtable.ext_ops[FB_OP_CHEGVD][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CHEGVD Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 2, 'V', FB_UPPER, 3, a, 3, b, 3, w);
    if (info != 0 || g_chegvd_fortran_call.query_calls != 1 ||
        g_chegvd_fortran_call.exec_calls != 1 ||
        g_chegvd_fortran_call.exec_lwork != 44 ||
        g_chegvd_fortran_call.exec_lrwork != 21 ||
        g_chegvd_fortran_call.exec_liwork != 17 ||
        w[0] != 10.0f || w[1] != 11.0f || w[2] != 12.0f) {
        fprintf(stderr, "[FAIL] CHEGVD Fortran->CBLAS thunk did not preserve divide-and-conquer generalized Hermitian-eigen query semantics\n");
        return 1;
    }

    printf("[PASS] CHEGVD Fortran->CBLAS thunk performs the workspace query and forwards divide-and-conquer generalized Hermitian eigenvalues\n");
    return 0;
}

static int check_chegvd_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_chegvd_fortran_slot_fn thunk = NULL;
    fb_complex_float_t a[9] = { 0 };
    fb_complex_float_t b[9] = { 0 };
    float w[3] = { 0.0f, 0.0f, 0.0f };
    fb_complex_float_t work[8] = { 0 };
    float rwork[16] = { 0.0f };
    int iwork[8] = { 0 };
    int itype = 2;
    char jobz = 'V';
    char uplo = 'U';
    int n = 3;
    int lda = 3;
    int ldb = 3;
    int lwork = 8;
    int lrwork = 16;
    int liwork = 8;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_chegvd_cblas_call, 0, sizeof(g_chegvd_cblas_call));
    g_chegvd_cblas_rc = 319;
    vtable.ext_ops[FB_OP_CHEGVD][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_chegvd_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CHEGVD);

    thunk = (fb_chegvd_fortran_slot_fn)vtable.ext_ops[FB_OP_CHEGVD][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CHEGVD CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, work, &lwork, rwork, &lrwork, iwork, &liwork, &info);
    if (info != 319 || g_chegvd_cblas_call.called != 1 ||
        g_chegvd_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_chegvd_cblas_call.itype != 2 || g_chegvd_cblas_call.jobz != 'V' ||
        g_chegvd_cblas_call.uplo != FB_UPPER || g_chegvd_cblas_call.n != 3 ||
        g_chegvd_cblas_call.lda != 3 || g_chegvd_cblas_call.ldb != 3) {
        fprintf(stderr, "[FAIL] CHEGVD CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] CHEGVD CBLAS->Fortran thunk maps Fortran UPLO chars into the divide-and-conquer generalized C Hermitian-eigen entry\n");
    return 0;
}

static int check_csygvd_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_chegvd_fn thunk = NULL;
    fb_complex_float_t a[9] = { 0 };
    fb_complex_float_t b[9] = { 0 };
    float w[3] = { 0.0f, 0.0f, 0.0f };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_chegvd_fortran_call, 0, sizeof(g_chegvd_fortran_call));
    vtable.ext_ops[FB_OP_CSYGVD][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_chegvd_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CSYGVD);

    thunk = (fb_chegvd_fn)vtable.ext_ops[FB_OP_CSYGVD][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CSYGVD Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 2, 'V', FB_UPPER, 3, a, 3, b, 3, w);
    if (info != 0 || g_chegvd_fortran_call.query_calls != 1 ||
        g_chegvd_fortran_call.exec_calls != 1 ||
        g_chegvd_fortran_call.exec_lwork != 44 ||
        g_chegvd_fortran_call.exec_lrwork != 21 ||
        g_chegvd_fortran_call.exec_liwork != 17 ||
        w[0] != 10.0f || w[1] != 11.0f || w[2] != 12.0f) {
        fprintf(stderr, "[FAIL] CSYGVD Fortran->CBLAS thunk did not preserve divide-and-conquer generalized complex-symmetric eigen query semantics\n");
        return 1;
    }

    printf("[PASS] CSYGVD Fortran->CBLAS thunk performs the workspace query and forwards divide-and-conquer generalized complex-symmetric eigenvalues\n");
    return 0;
}

static int check_csygvd_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_chegvd_fortran_slot_fn thunk = NULL;
    fb_complex_float_t a[9] = { 0 };
    fb_complex_float_t b[9] = { 0 };
    float w[3] = { 0.0f, 0.0f, 0.0f };
    fb_complex_float_t work[8] = { 0 };
    float rwork[16] = { 0.0f };
    int iwork[8] = { 0 };
    int itype = 2;
    char jobz = 'V';
    char uplo = 'U';
    int n = 3;
    int lda = 3;
    int ldb = 3;
    int lwork = 8;
    int lrwork = 16;
    int liwork = 8;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_chegvd_cblas_call, 0, sizeof(g_chegvd_cblas_call));
    g_chegvd_cblas_rc = 329;
    vtable.ext_ops[FB_OP_CSYGVD][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_chegvd_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CSYGVD);

    thunk = (fb_chegvd_fortran_slot_fn)vtable.ext_ops[FB_OP_CSYGVD][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CSYGVD CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, work, &lwork, rwork, &lrwork, iwork, &liwork, &info);
    if (info != 329 || g_chegvd_cblas_call.called != 1 ||
        g_chegvd_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_chegvd_cblas_call.itype != 2 || g_chegvd_cblas_call.jobz != 'V' ||
        g_chegvd_cblas_call.uplo != FB_UPPER || g_chegvd_cblas_call.n != 3 ||
        g_chegvd_cblas_call.lda != 3 || g_chegvd_cblas_call.ldb != 3) {
        fprintf(stderr, "[FAIL] CSYGVD CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] CSYGVD CBLAS->Fortran thunk maps Fortran UPLO chars into the divide-and-conquer generalized C complex-symmetric eigen entry\n");
    return 0;
}

int main(void)
{
    if (check_ssygv_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_ssygv_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_chegv_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_chegv_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_csygv_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_csygv_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_ssygvx_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_ssygvx_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_chegvx_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_chegvx_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_csygvx_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_csygvx_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_ssygvd_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_ssygvd_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_chegvd_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_chegvd_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_csygvd_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_csygvd_cblas_to_fortran() != 0) {
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}