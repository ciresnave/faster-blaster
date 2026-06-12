#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef int (*fb_ssyevd_fn)(fb_layout_t layout, char jobz, fb_uplo_t uplo, int n,
                            float *a, int lda, float *w);
typedef int (*fb_dsyevd_fn)(fb_layout_t layout, char jobz, fb_uplo_t uplo, int n,
                            double *a, int lda, double *w);
typedef int (*fb_cheevd_fn)(fb_layout_t layout, char jobz, fb_uplo_t uplo, int n,
                            fb_complex_float_t *a, int lda, float *w);
typedef int (*fb_zheevd_fn)(fb_layout_t layout, char jobz, fb_uplo_t uplo, int n,
                            fb_complex_double_t *a, int lda, double *w);

typedef void (*fb_ssyevd_fortran_slot_fn)(char *jobz, char *uplo, int *n,
                                          float *a, int *lda, float *w,
                                          float *work, int *lwork, int *iwork,
                                          int *liwork, int *info);
typedef void (*fb_dsyevd_fortran_slot_fn)(char *jobz, char *uplo, int *n,
                                          double *a, int *lda, double *w,
                                          double *work, int *lwork,
                                          int *iwork, int *liwork, int *info);
typedef void (*fb_cheevd_fortran_slot_fn)(char *jobz, char *uplo, int *n,
                                          fb_complex_float_t *a, int *lda,
                                          float *w, fb_complex_float_t *work,
                                          int *lwork, float *rwork,
                                          int *lrwork, int *iwork,
                                          int *liwork, int *info);
typedef void (*fb_zheevd_fortran_slot_fn)(char *jobz, char *uplo, int *n,
                                          fb_complex_double_t *a, int *lda,
                                          double *w,
                                          fb_complex_double_t *work,
                                          int *lwork, double *rwork,
                                          int *lrwork, int *iwork,
                                          int *liwork, int *info);

static struct {
    int query_calls;
    int exec_calls;
    int query_lwork;
    int exec_lwork;
} g_ssyevd_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    char jobz;
    fb_uplo_t uplo;
    int n;
    int lda;
} g_ssyevd_cblas_call;

static struct {
    int query_calls;
    int exec_calls;
    int query_lwork;
    int exec_lwork;
} g_dsyevd_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    char jobz;
    fb_uplo_t uplo;
    int n;
    int lda;
} g_dsyevd_cblas_call;

static struct {
    int query_calls;
    int exec_calls;
    int query_lwork;
    int exec_lwork;
} g_cheevd_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    char jobz;
    fb_uplo_t uplo;
    int n;
    int lda;
} g_cheevd_cblas_call;

static struct {
    int query_calls;
    int exec_calls;
    int query_lwork;
    int exec_lwork;
} g_zheevd_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    char jobz;
    fb_uplo_t uplo;
    int n;
    int lda;
} g_zheevd_cblas_call;

static int g_ssyevd_cblas_rc = 0;
static int g_dsyevd_cblas_rc = 0;
static int g_cheevd_cblas_rc = 0;
static int g_zheevd_cblas_rc = 0;

static fb_complex_float_t make_cfloat(float real_value)
{
    fb_complex_float_t value = (fb_complex_float_t)0;
    memcpy(&value, &real_value, sizeof(real_value));
    return value;
}

static fb_complex_double_t make_cdouble(double real_value)
{
    fb_complex_double_t value = (fb_complex_double_t)0;
    memcpy(&value, &real_value, sizeof(real_value));
    return value;
}

static void stub_ssyevd_fortran(char *jobz, char *uplo, int *n, float *a,
                                int *lda, float *w, float *work, int *lwork,
                                int *iwork, int *liwork, int *info)
{
    (void)jobz;
    (void)uplo;
    (void)n;
    (void)a;
    (void)lda;
    (void)iwork;
    (void)liwork;
    if (*lwork == -1) {
        g_ssyevd_fortran_call.query_calls += 1;
        g_ssyevd_fortran_call.query_lwork = *lwork;
        work[0] = 33.0f;
        *info = 0;
        return;
    }

    g_ssyevd_fortran_call.exec_calls += 1;
    g_ssyevd_fortran_call.exec_lwork = *lwork;
    w[0] = 7.0f;
    w[1] = 8.0f;
    w[2] = 9.0f;
    *info = 0;
}

static int stub_ssyevd_cblas(fb_layout_t layout, char jobz, fb_uplo_t uplo, int n,
                             float *a, int lda, float *w)
{
    (void)a;
    (void)w;
    g_ssyevd_cblas_call.called += 1;
    g_ssyevd_cblas_call.layout = layout;
    g_ssyevd_cblas_call.jobz = jobz;
    g_ssyevd_cblas_call.uplo = uplo;
    g_ssyevd_cblas_call.n = n;
    g_ssyevd_cblas_call.lda = lda;
    return g_ssyevd_cblas_rc;
}

static void stub_dsyevd_fortran(char *jobz, char *uplo, int *n, double *a,
                                int *lda, double *w, double *work, int *lwork,
                                int *iwork, int *liwork, int *info)
{
    (void)jobz;
    (void)uplo;
    (void)n;
    (void)a;
    (void)lda;
    (void)iwork;
    (void)liwork;
    if (*lwork == -1) {
        g_dsyevd_fortran_call.query_calls += 1;
        g_dsyevd_fortran_call.query_lwork = *lwork;
        work[0] = 43.0;
        *info = 0;
        return;
    }

    g_dsyevd_fortran_call.exec_calls += 1;
    g_dsyevd_fortran_call.exec_lwork = *lwork;
    w[0] = 13.0;
    w[1] = 14.0;
    w[2] = 15.0;
    *info = 0;
}

static int stub_dsyevd_cblas(fb_layout_t layout, char jobz, fb_uplo_t uplo, int n,
                             double *a, int lda, double *w)
{
    (void)a;
    (void)w;
    g_dsyevd_cblas_call.called += 1;
    g_dsyevd_cblas_call.layout = layout;
    g_dsyevd_cblas_call.jobz = jobz;
    g_dsyevd_cblas_call.uplo = uplo;
    g_dsyevd_cblas_call.n = n;
    g_dsyevd_cblas_call.lda = lda;
    return g_dsyevd_cblas_rc;
}

static void stub_cheevd_fortran(char *jobz, char *uplo, int *n,
                                fb_complex_float_t *a, int *lda, float *w,
                                fb_complex_float_t *work, int *lwork,
                                float *rwork, int *lrwork, int *iwork,
                                int *liwork, int *info)
{
    (void)jobz;
    (void)uplo;
    (void)n;
    (void)a;
    (void)lda;
    (void)rwork;
    (void)lrwork;
    (void)iwork;
    (void)liwork;
    if (*lwork == -1) {
        g_cheevd_fortran_call.query_calls += 1;
        g_cheevd_fortran_call.query_lwork = *lwork;
        work[0] = make_cfloat(34.0f);
        *info = 0;
        return;
    }

    g_cheevd_fortran_call.exec_calls += 1;
    g_cheevd_fortran_call.exec_lwork = *lwork;
    w[0] = 10.0f;
    w[1] = 11.0f;
    w[2] = 12.0f;
    *info = 0;
}

static int stub_cheevd_cblas(fb_layout_t layout, char jobz, fb_uplo_t uplo, int n,
                             fb_complex_float_t *a, int lda, float *w)
{
    (void)a;
    (void)w;
    g_cheevd_cblas_call.called += 1;
    g_cheevd_cblas_call.layout = layout;
    g_cheevd_cblas_call.jobz = jobz;
    g_cheevd_cblas_call.uplo = uplo;
    g_cheevd_cblas_call.n = n;
    g_cheevd_cblas_call.lda = lda;
    return g_cheevd_cblas_rc;
}

static void stub_zheevd_fortran(char *jobz, char *uplo, int *n,
                                fb_complex_double_t *a, int *lda, double *w,
                                fb_complex_double_t *work, int *lwork,
                                double *rwork, int *lrwork, int *iwork,
                                int *liwork, int *info)
{
    (void)jobz;
    (void)uplo;
    (void)n;
    (void)a;
    (void)lda;
    (void)rwork;
    (void)lrwork;
    (void)iwork;
    (void)liwork;
    if (*lwork == -1) {
        g_zheevd_fortran_call.query_calls += 1;
        g_zheevd_fortran_call.query_lwork = *lwork;
        work[0] = make_cdouble(44.0);
        *info = 0;
        return;
    }

    g_zheevd_fortran_call.exec_calls += 1;
    g_zheevd_fortran_call.exec_lwork = *lwork;
    w[0] = 16.0;
    w[1] = 17.0;
    w[2] = 18.0;
    *info = 0;
}

static int stub_zheevd_cblas(fb_layout_t layout, char jobz, fb_uplo_t uplo, int n,
                             fb_complex_double_t *a, int lda, double *w)
{
    (void)a;
    (void)w;
    g_zheevd_cblas_call.called += 1;
    g_zheevd_cblas_call.layout = layout;
    g_zheevd_cblas_call.jobz = jobz;
    g_zheevd_cblas_call.uplo = uplo;
    g_zheevd_cblas_call.n = n;
    g_zheevd_cblas_call.lda = lda;
    return g_zheevd_cblas_rc;
}

static int check_ssyevd_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_ssyevd_fn thunk = NULL;
    float a[9] = { 0.0f };
    float w[3] = { 0.0f, 0.0f, 0.0f };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_ssyevd_fortran_call, 0, sizeof(g_ssyevd_fortran_call));

    vtable.ext_ops[FB_OP_SSYEVD][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_ssyevd_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SSYEVD);

    thunk = (fb_ssyevd_fn)vtable.ext_ops[FB_OP_SSYEVD][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SSYEVD Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 'V', FB_LOWER, 3, a, 3, w);
    if (info != 0 || g_ssyevd_fortran_call.query_calls != 1 ||
        g_ssyevd_fortran_call.exec_calls != 1 ||
        g_ssyevd_fortran_call.query_lwork != -1 ||
        g_ssyevd_fortran_call.exec_lwork != 33 ||
        w[0] != 7.0f || w[1] != 8.0f || w[2] != 9.0f) {
        fprintf(stderr, "[FAIL] SSYEVD Fortran->CBLAS thunk did not preserve symmetric EVD query semantics\n");
        return 1;
    }

    printf("[PASS] SSYEVD Fortran->CBLAS thunk performs the workspace query and forwards eigenvalues\n");
    return 0;
}

static int check_ssyevd_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_ssyevd_fortran_slot_fn thunk = NULL;
    float a[9] = { 0.0f };
    float w[3] = { 0.0f, 0.0f, 0.0f };
    float work[8] = { 0.0f };
    int iwork[8] = { 0 };
    char jobz = 'V';
    char uplo = 'L';
    int n = 3;
    int lda = 3;
    int lwork = 8;
    int liwork = 8;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_ssyevd_cblas_call, 0, sizeof(g_ssyevd_cblas_call));
    g_ssyevd_cblas_rc = 241;

    vtable.ext_ops[FB_OP_SSYEVD][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_ssyevd_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SSYEVD);

    thunk = (fb_ssyevd_fortran_slot_fn)vtable.ext_ops[FB_OP_SSYEVD][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SSYEVD CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&jobz, &uplo, &n, a, &lda, w, work, &lwork, iwork, &liwork, &info);
    if (info != 241 || g_ssyevd_cblas_call.called != 1 ||
        g_ssyevd_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_ssyevd_cblas_call.jobz != 'V' || g_ssyevd_cblas_call.uplo != FB_LOWER ||
        g_ssyevd_cblas_call.n != 3 || g_ssyevd_cblas_call.lda != 3) {
        fprintf(stderr, "[FAIL] SSYEVD CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] SSYEVD CBLAS->Fortran thunk maps Fortran UPLO chars into the generic C symmetric-EVD entry\n");
    return 0;
}

static int check_dsyevd_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_dsyevd_fn thunk = NULL;
    double a[9] = { 0.0 };
    double w[3] = { 0.0, 0.0, 0.0 };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dsyevd_fortran_call, 0, sizeof(g_dsyevd_fortran_call));

    vtable.ext_ops[FB_OP_DSYEVD][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_dsyevd_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_DSYEVD);

    thunk = (fb_dsyevd_fn)vtable.ext_ops[FB_OP_DSYEVD][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DSYEVD Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 'V', FB_UPPER, 3, a, 3, w);
    if (info != 0 || g_dsyevd_fortran_call.query_calls != 1 ||
        g_dsyevd_fortran_call.exec_calls != 1 ||
        g_dsyevd_fortran_call.query_lwork != -1 ||
        g_dsyevd_fortran_call.exec_lwork != 43 ||
        w[0] != 13.0 || w[1] != 14.0 || w[2] != 15.0) {
        fprintf(stderr, "[FAIL] DSYEVD Fortran->CBLAS thunk did not preserve double symmetric EVD query semantics\n");
        return 1;
    }

    printf("[PASS] DSYEVD Fortran->CBLAS thunk performs the workspace query and forwards double eigenvalues\n");
    return 0;
}

static int check_dsyevd_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_dsyevd_fortran_slot_fn thunk = NULL;
    double a[9] = { 0.0 };
    double w[3] = { 0.0, 0.0, 0.0 };
    double work[8] = { 0.0 };
    int iwork[8] = { 0 };
    char jobz = 'V';
    char uplo = 'U';
    int n = 3;
    int lda = 3;
    int lwork = 8;
    int liwork = 8;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dsyevd_cblas_call, 0, sizeof(g_dsyevd_cblas_call));
    g_dsyevd_cblas_rc = 251;

    vtable.ext_ops[FB_OP_DSYEVD][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_dsyevd_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_DSYEVD);

    thunk = (fb_dsyevd_fortran_slot_fn)vtable.ext_ops[FB_OP_DSYEVD][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DSYEVD CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&jobz, &uplo, &n, a, &lda, w, work, &lwork, iwork, &liwork, &info);
    if (info != 251 || g_dsyevd_cblas_call.called != 1 ||
        g_dsyevd_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_dsyevd_cblas_call.jobz != 'V' || g_dsyevd_cblas_call.uplo != FB_UPPER ||
        g_dsyevd_cblas_call.n != 3 || g_dsyevd_cblas_call.lda != 3) {
        fprintf(stderr, "[FAIL] DSYEVD CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] DSYEVD CBLAS->Fortran thunk maps Fortran UPLO chars into the generic C double symmetric-EVD entry\n");
    return 0;
}

static int check_cheevd_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_cheevd_fn thunk = NULL;
    fb_complex_float_t a[9] = { 0 };
    float w[3] = { 0.0f, 0.0f, 0.0f };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cheevd_fortran_call, 0, sizeof(g_cheevd_fortran_call));

    vtable.ext_ops[FB_OP_CHEEVD][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_cheevd_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CHEEVD);

    thunk = (fb_cheevd_fn)vtable.ext_ops[FB_OP_CHEEVD][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CHEEVD Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 'V', FB_UPPER, 3, a, 3, w);
    if (info != 0 || g_cheevd_fortran_call.query_calls != 1 ||
        g_cheevd_fortran_call.exec_calls != 1 ||
        g_cheevd_fortran_call.query_lwork != -1 ||
        g_cheevd_fortran_call.exec_lwork != 34 ||
        w[0] != 10.0f || w[1] != 11.0f || w[2] != 12.0f) {
        fprintf(stderr, "[FAIL] CHEEVD Fortran->CBLAS thunk did not preserve Hermitian EVD query semantics\n");
        return 1;
    }

    printf("[PASS] CHEEVD Fortran->CBLAS thunk performs the workspace query and forwards Hermitian eigenvalues\n");
    return 0;
}

static int check_cheevd_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_cheevd_fortran_slot_fn thunk = NULL;
    fb_complex_float_t a[9] = { 0 };
    float w[3] = { 0.0f, 0.0f, 0.0f };
    fb_complex_float_t work[8] = { 0 };
    float rwork[16] = { 0.0f };
    int iwork[8] = { 0 };
    char jobz = 'V';
    char uplo = 'U';
    int n = 3;
    int lda = 3;
    int lwork = 8;
    int lrwork = 16;
    int liwork = 8;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cheevd_cblas_call, 0, sizeof(g_cheevd_cblas_call));
    g_cheevd_cblas_rc = 243;

    vtable.ext_ops[FB_OP_CHEEVD][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_cheevd_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CHEEVD);

    thunk = (fb_cheevd_fortran_slot_fn)vtable.ext_ops[FB_OP_CHEEVD][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CHEEVD CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&jobz, &uplo, &n, a, &lda, w, work, &lwork, rwork, &lrwork, iwork, &liwork, &info);
    if (info != 243 || g_cheevd_cblas_call.called != 1 ||
        g_cheevd_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_cheevd_cblas_call.jobz != 'V' || g_cheevd_cblas_call.uplo != FB_UPPER ||
        g_cheevd_cblas_call.n != 3 || g_cheevd_cblas_call.lda != 3) {
        fprintf(stderr, "[FAIL] CHEEVD CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] CHEEVD CBLAS->Fortran thunk maps Fortran UPLO chars into the generic C Hermitian-EVD entry\n");
    return 0;
}

static int check_zheevd_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_zheevd_fn thunk = NULL;
    fb_complex_double_t a[9] = { 0 };
    double w[3] = { 0.0, 0.0, 0.0 };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zheevd_fortran_call, 0, sizeof(g_zheevd_fortran_call));

    vtable.ext_ops[FB_OP_ZHEEVD][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_zheevd_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_ZHEEVD);

    thunk = (fb_zheevd_fn)vtable.ext_ops[FB_OP_ZHEEVD][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZHEEVD Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 'V', FB_LOWER, 3, a, 3, w);
    if (info != 0 || g_zheevd_fortran_call.query_calls != 1 ||
        g_zheevd_fortran_call.exec_calls != 1 ||
        g_zheevd_fortran_call.query_lwork != -1 ||
        g_zheevd_fortran_call.exec_lwork != 44 ||
        w[0] != 16.0 || w[1] != 17.0 || w[2] != 18.0) {
        fprintf(stderr, "[FAIL] ZHEEVD Fortran->CBLAS thunk did not preserve complex-double Hermitian EVD query semantics\n");
        return 1;
    }

    printf("[PASS] ZHEEVD Fortran->CBLAS thunk performs the workspace query and forwards complex-double Hermitian eigenvalues\n");
    return 0;
}

static int check_zheevd_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_zheevd_fortran_slot_fn thunk = NULL;
    fb_complex_double_t a[9] = { 0 };
    double w[3] = { 0.0, 0.0, 0.0 };
    fb_complex_double_t work[8] = { 0 };
    double rwork[16] = { 0.0 };
    int iwork[8] = { 0 };
    char jobz = 'V';
    char uplo = 'L';
    int n = 3;
    int lda = 3;
    int lwork = 8;
    int lrwork = 16;
    int liwork = 8;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zheevd_cblas_call, 0, sizeof(g_zheevd_cblas_call));
    g_zheevd_cblas_rc = 253;

    vtable.ext_ops[FB_OP_ZHEEVD][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_zheevd_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_ZHEEVD);

    thunk = (fb_zheevd_fortran_slot_fn)vtable.ext_ops[FB_OP_ZHEEVD][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZHEEVD CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&jobz, &uplo, &n, a, &lda, w, work, &lwork, rwork, &lrwork, iwork,
          &liwork, &info);
    if (info != 253 || g_zheevd_cblas_call.called != 1 ||
        g_zheevd_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_zheevd_cblas_call.jobz != 'V' || g_zheevd_cblas_call.uplo != FB_LOWER ||
        g_zheevd_cblas_call.n != 3 || g_zheevd_cblas_call.lda != 3) {
        fprintf(stderr, "[FAIL] ZHEEVD CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] ZHEEVD CBLAS->Fortran thunk maps Fortran UPLO chars into the generic C complex-double Hermitian-EVD entry\n");
    return 0;
}

int main(void)
{
    if (check_ssyevd_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_ssyevd_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_dsyevd_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_dsyevd_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_cheevd_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_cheevd_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_zheevd_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_zheevd_cblas_to_fortran() != 0) {
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}