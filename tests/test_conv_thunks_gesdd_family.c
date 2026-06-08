#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef int (*fb_sgesdd_fn)(fb_layout_t layout, char jobz, int m, int n,
                            float *a, int lda, float *s, float *u, int ldu,
                            float *vt, int ldvt);
typedef int (*fb_cgesdd_fn)(fb_layout_t layout, char jobz, int m, int n,
                            fb_complex_float_t *a, int lda, float *s,
                            fb_complex_float_t *u, int ldu,
                            fb_complex_float_t *vt, int ldvt);

typedef void (*fb_sgesdd_fortran_slot_fn)(char *jobz, int *m, int *n, float *a,
                                          int *lda, float *s, float *u,
                                          int *ldu, float *vt, int *ldvt,
                                          float *work, int *lwork, int *iwork,
                                          int *info);
typedef void (*fb_cgesdd_fortran_slot_fn)(char *jobz, int *m, int *n,
                                          fb_complex_float_t *a, int *lda,
                                          float *s, fb_complex_float_t *u,
                                          int *ldu, fb_complex_float_t *vt,
                                          int *ldvt, fb_complex_float_t *work,
                                          int *lwork, float *rwork,
                                          int *iwork, int *info);

static struct {
    int query_calls;
    int exec_calls;
    int query_lwork;
    int exec_lwork;
} g_sgesdd_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    char jobz;
    int m;
    int n;
    int lda;
    int ldu;
    int ldvt;
} g_sgesdd_cblas_call;

static struct {
    int query_calls;
    int exec_calls;
    int query_lwork;
    int exec_lwork;
} g_cgesdd_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    char jobz;
    int m;
    int n;
    int lda;
    int ldu;
    int ldvt;
} g_cgesdd_cblas_call;

static int g_sgesdd_cblas_rc = 0;
static int g_cgesdd_cblas_rc = 0;

static fb_complex_float_t make_cfloat(float real_value)
{
    fb_complex_float_t value = (fb_complex_float_t)0;
    memcpy(&value, &real_value, sizeof(real_value));
    return value;
}

static void stub_sgesdd_fortran(char *jobz, int *m, int *n, float *a, int *lda,
                                float *s, float *u, int *ldu, float *vt,
                                int *ldvt, float *work, int *lwork,
                                int *iwork, int *info)
{
    (void)jobz;
    (void)m;
    (void)n;
    (void)a;
    (void)lda;
    (void)u;
    (void)ldu;
    (void)vt;
    (void)ldvt;
    (void)iwork;
    if (*lwork == -1) {
        g_sgesdd_fortran_call.query_calls += 1;
        g_sgesdd_fortran_call.query_lwork = *lwork;
        work[0] = 27.0f;
        *info = 0;
        return;
    }

    g_sgesdd_fortran_call.exec_calls += 1;
    g_sgesdd_fortran_call.exec_lwork = *lwork;
    s[0] = 7.0f;
    s[1] = 6.0f;
    s[2] = 5.0f;
    *info = 0;
}

static int stub_sgesdd_cblas(fb_layout_t layout, char jobz, int m, int n,
                             float *a, int lda, float *s, float *u, int ldu,
                             float *vt, int ldvt)
{
    (void)a;
    (void)s;
    (void)u;
    (void)vt;
    g_sgesdd_cblas_call.called += 1;
    g_sgesdd_cblas_call.layout = layout;
    g_sgesdd_cblas_call.jobz = jobz;
    g_sgesdd_cblas_call.m = m;
    g_sgesdd_cblas_call.n = n;
    g_sgesdd_cblas_call.lda = lda;
    g_sgesdd_cblas_call.ldu = ldu;
    g_sgesdd_cblas_call.ldvt = ldvt;
    return g_sgesdd_cblas_rc;
}

static void stub_cgesdd_fortran(char *jobz, int *m, int *n,
                                fb_complex_float_t *a, int *lda, float *s,
                                fb_complex_float_t *u, int *ldu,
                                fb_complex_float_t *vt, int *ldvt,
                                fb_complex_float_t *work, int *lwork,
                                float *rwork, int *iwork, int *info)
{
    (void)jobz;
    (void)m;
    (void)n;
    (void)a;
    (void)lda;
    (void)u;
    (void)ldu;
    (void)vt;
    (void)ldvt;
    (void)rwork;
    (void)iwork;
    if (*lwork == -1) {
        g_cgesdd_fortran_call.query_calls += 1;
        g_cgesdd_fortran_call.query_lwork = *lwork;
        work[0] = make_cfloat(28.0f);
        *info = 0;
        return;
    }

    g_cgesdd_fortran_call.exec_calls += 1;
    g_cgesdd_fortran_call.exec_lwork = *lwork;
    s[0] = 8.0f;
    s[1] = 7.0f;
    s[2] = 6.0f;
    *info = 0;
}

static int stub_cgesdd_cblas(fb_layout_t layout, char jobz, int m, int n,
                             fb_complex_float_t *a, int lda, float *s,
                             fb_complex_float_t *u, int ldu,
                             fb_complex_float_t *vt, int ldvt)
{
    (void)a;
    (void)s;
    (void)u;
    (void)vt;
    g_cgesdd_cblas_call.called += 1;
    g_cgesdd_cblas_call.layout = layout;
    g_cgesdd_cblas_call.jobz = jobz;
    g_cgesdd_cblas_call.m = m;
    g_cgesdd_cblas_call.n = n;
    g_cgesdd_cblas_call.lda = lda;
    g_cgesdd_cblas_call.ldu = ldu;
    g_cgesdd_cblas_call.ldvt = ldvt;
    return g_cgesdd_cblas_rc;
}

static int check_sgesdd_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_sgesdd_fn thunk = NULL;
    float a[9] = { 0.0f };
    float s[3] = { 0.0f, 0.0f, 0.0f };
    float u[9] = { 0.0f };
    float vt[9] = { 0.0f };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sgesdd_fortran_call, 0, sizeof(g_sgesdd_fortran_call));

    vtable.ext_ops[FB_OP_SGESDD][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_sgesdd_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SGESDD);

    thunk = (fb_sgesdd_fn)vtable.ext_ops[FB_OP_SGESDD][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SGESDD Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 'A', 3, 3, a, 3, s, u, 3, vt, 3);
    if (info != 0 || g_sgesdd_fortran_call.query_calls != 1 ||
        g_sgesdd_fortran_call.exec_calls != 1 ||
        g_sgesdd_fortran_call.query_lwork != -1 ||
        g_sgesdd_fortran_call.exec_lwork != 27 ||
        s[0] != 7.0f || s[1] != 6.0f || s[2] != 5.0f) {
        fprintf(stderr, "[FAIL] SGESDD Fortran->CBLAS thunk did not preserve divide-and-conquer SVD query semantics\n");
        return 1;
    }

    printf("[PASS] SGESDD Fortran->CBLAS thunk performs the workspace query and forwards singular values\n");
    return 0;
}

static int check_sgesdd_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_sgesdd_fortran_slot_fn thunk = NULL;
    float a[9] = { 0.0f };
    float s[3] = { 0.0f, 0.0f, 0.0f };
    float u[9] = { 0.0f };
    float vt[9] = { 0.0f };
    float work[8] = { 0.0f };
    int iwork[24] = { 0 };
    char jobz = 'A';
    int m = 3;
    int n = 3;
    int lda = 3;
    int ldu = 3;
    int ldvt = 3;
    int lwork = 8;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sgesdd_cblas_call, 0, sizeof(g_sgesdd_cblas_call));
    g_sgesdd_cblas_rc = 221;

    vtable.ext_ops[FB_OP_SGESDD][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_sgesdd_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SGESDD);

    thunk = (fb_sgesdd_fortran_slot_fn)vtable.ext_ops[FB_OP_SGESDD][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SGESDD CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, iwork, &info);
    if (info != 221 || g_sgesdd_cblas_call.called != 1 ||
        g_sgesdd_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_sgesdd_cblas_call.jobz != 'A' ||
        g_sgesdd_cblas_call.m != 3 || g_sgesdd_cblas_call.n != 3 ||
        g_sgesdd_cblas_call.lda != 3 || g_sgesdd_cblas_call.ldu != 3 ||
        g_sgesdd_cblas_call.ldvt != 3) {
        fprintf(stderr, "[FAIL] SGESDD CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] SGESDD CBLAS->Fortran thunk maps the all-pointer ABI into the generic C divide-and-conquer SVD entry\n");
    return 0;
}

static int check_cgesdd_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_cgesdd_fn thunk = NULL;
    fb_complex_float_t a[9] = { 0 };
    float s[3] = { 0.0f, 0.0f, 0.0f };
    fb_complex_float_t u[9] = { 0 };
    fb_complex_float_t vt[9] = { 0 };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cgesdd_fortran_call, 0, sizeof(g_cgesdd_fortran_call));

    vtable.ext_ops[FB_OP_CGESDD][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_cgesdd_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CGESDD);

    thunk = (fb_cgesdd_fn)vtable.ext_ops[FB_OP_CGESDD][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CGESDD Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 'A', 3, 3, a, 3, s, u, 3, vt, 3);
    if (info != 0 || g_cgesdd_fortran_call.query_calls != 1 ||
        g_cgesdd_fortran_call.exec_calls != 1 ||
        g_cgesdd_fortran_call.query_lwork != -1 ||
        g_cgesdd_fortran_call.exec_lwork != 28 ||
        s[0] != 8.0f || s[1] != 7.0f || s[2] != 6.0f) {
        fprintf(stderr, "[FAIL] CGESDD Fortran->CBLAS thunk did not preserve complex divide-and-conquer SVD query semantics\n");
        return 1;
    }

    printf("[PASS] CGESDD Fortran->CBLAS thunk performs the workspace query and forwards complex singular values\n");
    return 0;
}

static int check_cgesdd_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_cgesdd_fortran_slot_fn thunk = NULL;
    fb_complex_float_t a[9] = { 0 };
    float s[3] = { 0.0f, 0.0f, 0.0f };
    fb_complex_float_t u[9] = { 0 };
    fb_complex_float_t vt[9] = { 0 };
    fb_complex_float_t work[8] = { 0 };
    float rwork[16] = { 0.0f };
    int iwork[24] = { 0 };
    char jobz = 'A';
    int m = 3;
    int n = 3;
    int lda = 3;
    int ldu = 3;
    int ldvt = 3;
    int lwork = 8;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cgesdd_cblas_call, 0, sizeof(g_cgesdd_cblas_call));
    g_cgesdd_cblas_rc = 223;

    vtable.ext_ops[FB_OP_CGESDD][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_cgesdd_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CGESDD);

    thunk = (fb_cgesdd_fortran_slot_fn)vtable.ext_ops[FB_OP_CGESDD][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CGESDD CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, rwork, iwork, &info);
    if (info != 223 || g_cgesdd_cblas_call.called != 1 ||
        g_cgesdd_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_cgesdd_cblas_call.jobz != 'A' ||
        g_cgesdd_cblas_call.m != 3 || g_cgesdd_cblas_call.n != 3 ||
        g_cgesdd_cblas_call.lda != 3 || g_cgesdd_cblas_call.ldu != 3 ||
        g_cgesdd_cblas_call.ldvt != 3) {
        fprintf(stderr, "[FAIL] CGESDD CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] CGESDD CBLAS->Fortran thunk maps the all-pointer ABI into the generic complex divide-and-conquer SVD entry\n");
    return 0;
}

int main(void)
{
    if (check_sgesdd_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_sgesdd_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_cgesdd_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_cgesdd_cblas_to_fortran() != 0) {
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}