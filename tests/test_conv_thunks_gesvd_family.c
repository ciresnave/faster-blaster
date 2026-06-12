#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef int (*fb_sgesvd_fn)(fb_layout_t layout, char jobu, char jobvt, int m,
                            int n, float *a, int lda, float *s, float *u,
                            int ldu, float *vt, int ldvt, float *superb);
typedef int (*fb_dgesvd_fn)(fb_layout_t layout, char jobu, char jobvt, int m,
                            int n, double *a, int lda, double *s, double *u,
                            int ldu, double *vt, int ldvt, double *superb);
typedef int (*fb_cgesvd_fn)(fb_layout_t layout, char jobu, char jobvt, int m,
                            int n, fb_complex_float_t *a, int lda, float *s,
                            fb_complex_float_t *u, int ldu,
                            fb_complex_float_t *vt, int ldvt, float *superb);
typedef int (*fb_zgesvd_fn)(fb_layout_t layout, char jobu, char jobvt, int m,
                            int n, fb_complex_double_t *a, int lda, double *s,
                            fb_complex_double_t *u, int ldu,
                            fb_complex_double_t *vt, int ldvt, double *superb);

typedef void (*fb_sgesvd_fortran_slot_fn)(char *jobu, char *jobvt, int *m,
                                          int *n, float *a, int *lda, float *s,
                                          float *u, int *ldu, float *vt,
                                          int *ldvt, float *work, int *lwork,
                                          int *info);
typedef void (*fb_dgesvd_fortran_slot_fn)(char *jobu, char *jobvt, int *m,
                                          int *n, double *a, int *lda,
                                          double *s, double *u, int *ldu,
                                          double *vt, int *ldvt, double *work,
                                          int *lwork, int *info);
typedef void (*fb_cgesvd_fortran_slot_fn)(char *jobu, char *jobvt, int *m,
                                          int *n, fb_complex_float_t *a,
                                          int *lda, float *s,
                                          fb_complex_float_t *u, int *ldu,
                                          fb_complex_float_t *vt, int *ldvt,
                                          fb_complex_float_t *work,
                                          int *lwork, float *rwork, int *info);
typedef void (*fb_zgesvd_fortran_slot_fn)(char *jobu, char *jobvt, int *m,
                                          int *n, fb_complex_double_t *a,
                                          int *lda, double *s,
                                          fb_complex_double_t *u, int *ldu,
                                          fb_complex_double_t *vt, int *ldvt,
                                          fb_complex_double_t *work,
                                          int *lwork, double *rwork,
                                          int *info);

static struct {
    int query_calls;
    int exec_calls;
    int query_lwork;
    int exec_lwork;
} g_sgesvd_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    char jobu;
    char jobvt;
    int m;
    int n;
    int lda;
    int ldu;
    int ldvt;
    float *superb;
} g_sgesvd_cblas_call;

static struct {
    int query_calls;
    int exec_calls;
    int query_lwork;
    int exec_lwork;
} g_dgesvd_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    char jobu;
    char jobvt;
    int m;
    int n;
    int lda;
    int ldu;
    int ldvt;
    double *superb;
} g_dgesvd_cblas_call;

static struct {
    int query_calls;
    int exec_calls;
    int query_lwork;
    int exec_lwork;
} g_cgesvd_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    char jobu;
    char jobvt;
    int m;
    int n;
    int lda;
    int ldu;
    int ldvt;
    float *superb;
} g_cgesvd_cblas_call;

static struct {
    int query_calls;
    int exec_calls;
    int query_lwork;
    int exec_lwork;
} g_zgesvd_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    char jobu;
    char jobvt;
    int m;
    int n;
    int lda;
    int ldu;
    int ldvt;
    double *superb;
} g_zgesvd_cblas_call;

static int g_sgesvd_cblas_rc = 0;
static int g_dgesvd_cblas_rc = 0;
static int g_cgesvd_cblas_rc = 0;
static int g_zgesvd_cblas_rc = 0;

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

static float cfloat_real(fb_complex_float_t value)
{
    float real_value = 0.0f;
    memcpy(&real_value, &value, sizeof(real_value));
    return real_value;
}

static void stub_sgesvd_fortran(char *jobu, char *jobvt, int *m, int *n,
                                float *a, int *lda, float *s, float *u,
                                int *ldu, float *vt, int *ldvt,
                                float *work, int *lwork, int *info)
{
    (void)jobu;
    (void)jobvt;
    (void)a;
    (void)lda;
    (void)u;
    (void)ldu;
    (void)vt;
    (void)ldvt;
    if (*lwork == -1) {
        g_sgesvd_fortran_call.query_calls += 1;
        g_sgesvd_fortran_call.query_lwork = *lwork;
        work[0] = 25.0f;
        *info = 0;
        return;
    }

    g_sgesvd_fortran_call.exec_calls += 1;
    g_sgesvd_fortran_call.exec_lwork = *lwork;
    s[0] = 5.0f;
    s[1] = 4.0f;
    s[2] = 3.0f;
    work[1] = 9.0f;
    work[2] = 8.0f;
    *info = 0;
}

static int stub_sgesvd_cblas(fb_layout_t layout, char jobu, char jobvt, int m,
                             int n, float *a, int lda, float *s, float *u,
                             int ldu, float *vt, int ldvt, float *superb)
{
    (void)a;
    (void)s;
    (void)u;
    (void)vt;
    g_sgesvd_cblas_call.called += 1;
    g_sgesvd_cblas_call.layout = layout;
    g_sgesvd_cblas_call.jobu = jobu;
    g_sgesvd_cblas_call.jobvt = jobvt;
    g_sgesvd_cblas_call.m = m;
    g_sgesvd_cblas_call.n = n;
    g_sgesvd_cblas_call.lda = lda;
    g_sgesvd_cblas_call.ldu = ldu;
    g_sgesvd_cblas_call.ldvt = ldvt;
    g_sgesvd_cblas_call.superb = superb;
    if (superb) {
        superb[0] = 7.0f;
        superb[1] = 6.0f;
    }
    return g_sgesvd_cblas_rc;
}

static void stub_dgesvd_fortran(char *jobu, char *jobvt, int *m, int *n,
                                double *a, int *lda, double *s, double *u,
                                int *ldu, double *vt, int *ldvt,
                                double *work, int *lwork, int *info)
{
    (void)jobu;
    (void)jobvt;
    (void)a;
    (void)lda;
    (void)u;
    (void)ldu;
    (void)vt;
    (void)ldvt;
    if (*lwork == -1) {
        g_dgesvd_fortran_call.query_calls += 1;
        g_dgesvd_fortran_call.query_lwork = *lwork;
        work[0] = 35.0;
        *info = 0;
        return;
    }

    g_dgesvd_fortran_call.exec_calls += 1;
    g_dgesvd_fortran_call.exec_lwork = *lwork;
    s[0] = 8.0;
    s[1] = 7.0;
    s[2] = 6.0;
    work[1] = 12.0;
    work[2] = 11.0;
    *info = 0;
}

static int stub_dgesvd_cblas(fb_layout_t layout, char jobu, char jobvt, int m,
                             int n, double *a, int lda, double *s, double *u,
                             int ldu, double *vt, int ldvt, double *superb)
{
    (void)a;
    (void)s;
    (void)u;
    (void)vt;
    g_dgesvd_cblas_call.called += 1;
    g_dgesvd_cblas_call.layout = layout;
    g_dgesvd_cblas_call.jobu = jobu;
    g_dgesvd_cblas_call.jobvt = jobvt;
    g_dgesvd_cblas_call.m = m;
    g_dgesvd_cblas_call.n = n;
    g_dgesvd_cblas_call.lda = lda;
    g_dgesvd_cblas_call.ldu = ldu;
    g_dgesvd_cblas_call.ldvt = ldvt;
    g_dgesvd_cblas_call.superb = superb;
    if (superb) {
        superb[0] = 14.0;
        superb[1] = 13.0;
    }
    return g_dgesvd_cblas_rc;
}

static void stub_cgesvd_fortran(char *jobu, char *jobvt, int *m, int *n,
                                fb_complex_float_t *a, int *lda, float *s,
                                fb_complex_float_t *u, int *ldu,
                                fb_complex_float_t *vt, int *ldvt,
                                fb_complex_float_t *work, int *lwork,
                                float *rwork, int *info)
{
    (void)jobu;
    (void)jobvt;
    (void)a;
    (void)lda;
    (void)u;
    (void)ldu;
    (void)vt;
    (void)ldvt;
    if (*lwork == -1) {
        g_cgesvd_fortran_call.query_calls += 1;
        g_cgesvd_fortran_call.query_lwork = *lwork;
        work[0] = make_cfloat(26.0f);
        *info = 0;
        return;
    }

    g_cgesvd_fortran_call.exec_calls += 1;
    g_cgesvd_fortran_call.exec_lwork = *lwork;
    s[0] = 6.0f;
    s[1] = 5.0f;
    s[2] = 4.0f;
    rwork[0] = 11.0f;
    rwork[1] = 10.0f;
    *info = 0;
}

static int stub_cgesvd_cblas(fb_layout_t layout, char jobu, char jobvt, int m,
                             int n, fb_complex_float_t *a, int lda, float *s,
                             fb_complex_float_t *u, int ldu,
                             fb_complex_float_t *vt, int ldvt, float *superb)
{
    (void)a;
    (void)s;
    (void)u;
    (void)vt;
    g_cgesvd_cblas_call.called += 1;
    g_cgesvd_cblas_call.layout = layout;
    g_cgesvd_cblas_call.jobu = jobu;
    g_cgesvd_cblas_call.jobvt = jobvt;
    g_cgesvd_cblas_call.m = m;
    g_cgesvd_cblas_call.n = n;
    g_cgesvd_cblas_call.lda = lda;
    g_cgesvd_cblas_call.ldu = ldu;
    g_cgesvd_cblas_call.ldvt = ldvt;
    g_cgesvd_cblas_call.superb = superb;
    if (superb) {
        superb[0] = 12.0f;
        superb[1] = 13.0f;
    }
    return g_cgesvd_cblas_rc;
}

static void stub_zgesvd_fortran(char *jobu, char *jobvt, int *m, int *n,
                                fb_complex_double_t *a, int *lda, double *s,
                                fb_complex_double_t *u, int *ldu,
                                fb_complex_double_t *vt, int *ldvt,
                                fb_complex_double_t *work, int *lwork,
                                double *rwork, int *info)
{
    (void)jobu;
    (void)jobvt;
    (void)a;
    (void)lda;
    (void)u;
    (void)ldu;
    (void)vt;
    (void)ldvt;
    if (*lwork == -1) {
        g_zgesvd_fortran_call.query_calls += 1;
        g_zgesvd_fortran_call.query_lwork = *lwork;
        work[0] = make_cdouble(36.0);
        *info = 0;
        return;
    }

    g_zgesvd_fortran_call.exec_calls += 1;
    g_zgesvd_fortran_call.exec_lwork = *lwork;
    s[0] = 9.0;
    s[1] = 8.0;
    s[2] = 7.0;
    rwork[0] = 15.0;
    rwork[1] = 14.0;
    *info = 0;
}

static int stub_zgesvd_cblas(fb_layout_t layout, char jobu, char jobvt, int m,
                             int n, fb_complex_double_t *a, int lda, double *s,
                             fb_complex_double_t *u, int ldu,
                             fb_complex_double_t *vt, int ldvt, double *superb)
{
    (void)a;
    (void)s;
    (void)u;
    (void)vt;
    g_zgesvd_cblas_call.called += 1;
    g_zgesvd_cblas_call.layout = layout;
    g_zgesvd_cblas_call.jobu = jobu;
    g_zgesvd_cblas_call.jobvt = jobvt;
    g_zgesvd_cblas_call.m = m;
    g_zgesvd_cblas_call.n = n;
    g_zgesvd_cblas_call.lda = lda;
    g_zgesvd_cblas_call.ldu = ldu;
    g_zgesvd_cblas_call.ldvt = ldvt;
    g_zgesvd_cblas_call.superb = superb;
    if (superb) {
        superb[0] = 16.0;
        superb[1] = 17.0;
    }
    return g_zgesvd_cblas_rc;
}

static int check_sgesvd_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_sgesvd_fn thunk = NULL;
    float a[9] = { 0.0f };
    float s[3] = { 0.0f, 0.0f, 0.0f };
    float u[9] = { 0.0f };
    float vt[9] = { 0.0f };
    float superb[2] = { -1.0f, -1.0f };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sgesvd_fortran_call, 0, sizeof(g_sgesvd_fortran_call));

    vtable.ext_ops[FB_OP_SGESVD][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_sgesvd_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SGESVD);

    thunk = (fb_sgesvd_fn)vtable.ext_ops[FB_OP_SGESVD][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SGESVD Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 'A', 'A', 3, 3, a, 3, s, u, 3, vt, 3, superb);
    if (info != 0 || g_sgesvd_fortran_call.query_calls != 1 ||
        g_sgesvd_fortran_call.exec_calls != 1 ||
        g_sgesvd_fortran_call.query_lwork != -1 ||
        g_sgesvd_fortran_call.exec_lwork != 25 ||
        s[0] != 5.0f || s[1] != 4.0f || s[2] != 3.0f ||
        superb[0] != 9.0f || superb[1] != 8.0f) {
        fprintf(stderr, "[FAIL] SGESVD Fortran->CBLAS thunk did not propagate superb from work scratch\n");
        return 1;
    }

    printf("[PASS] SGESVD Fortran->CBLAS thunk performs the workspace query and propagates superb from work scratch\n");
    return 0;
}

static int check_sgesvd_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_sgesvd_fortran_slot_fn thunk = NULL;
    float a[9] = { 0.0f };
    float s[3] = { 0.0f, 0.0f, 0.0f };
    float u[9] = { 0.0f };
    float vt[9] = { 0.0f };
    float work[8] = { 0.0f };
    char jobu = 'A';
    char jobvt = 'A';
    int m = 3;
    int n = 3;
    int lda = 3;
    int ldu = 3;
    int ldvt = 3;
    int lwork = 8;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sgesvd_cblas_call, 0, sizeof(g_sgesvd_cblas_call));
    g_sgesvd_cblas_rc = 211;

    vtable.ext_ops[FB_OP_SGESVD][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_sgesvd_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SGESVD);

    thunk = (fb_sgesvd_fortran_slot_fn)vtable.ext_ops[FB_OP_SGESVD][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SGESVD CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, &info);
    if (info != 211 || g_sgesvd_cblas_call.called != 1 ||
        g_sgesvd_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_sgesvd_cblas_call.jobu != 'A' || g_sgesvd_cblas_call.jobvt != 'A' ||
        g_sgesvd_cblas_call.m != 3 || g_sgesvd_cblas_call.n != 3 ||
        g_sgesvd_cblas_call.lda != 3 || g_sgesvd_cblas_call.ldu != 3 ||
        g_sgesvd_cblas_call.ldvt != 3 || g_sgesvd_cblas_call.superb == NULL) {
        fprintf(stderr,
                "[FAIL] SGESVD CBLAS->Fortran thunk did not allocate and pass superb scratch (info=%d called=%d layout=%d jobu=%c jobvt=%c m=%d n=%d lda=%d ldu=%d ldvt=%d superb=%p)\n",
                info, g_sgesvd_cblas_call.called, (int)g_sgesvd_cblas_call.layout,
                g_sgesvd_cblas_call.jobu, g_sgesvd_cblas_call.jobvt,
                g_sgesvd_cblas_call.m, g_sgesvd_cblas_call.n,
                g_sgesvd_cblas_call.lda, g_sgesvd_cblas_call.ldu,
                g_sgesvd_cblas_call.ldvt, (void *)g_sgesvd_cblas_call.superb);
        return 1;
    }

    printf("[PASS] SGESVD CBLAS->Fortran thunk allocates the missing superb scratch for the generic C ABI\n");
    return 0;
}

static int check_dgesvd_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_dgesvd_fn thunk = NULL;
    double a[9] = { 0.0 };
    double s[3] = { 0.0, 0.0, 0.0 };
    double u[9] = { 0.0 };
    double vt[9] = { 0.0 };
    double superb[2] = { -1.0, -1.0 };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dgesvd_fortran_call, 0, sizeof(g_dgesvd_fortran_call));

    vtable.ext_ops[FB_OP_DGESVD][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_dgesvd_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_DGESVD);

    thunk = (fb_dgesvd_fn)vtable.ext_ops[FB_OP_DGESVD][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DGESVD Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 'A', 'A', 3, 3, a, 3, s, u, 3, vt, 3,
                 superb);
    if (info != 0 || g_dgesvd_fortran_call.query_calls != 1 ||
        g_dgesvd_fortran_call.exec_calls != 1 ||
        g_dgesvd_fortran_call.query_lwork != -1 ||
        g_dgesvd_fortran_call.exec_lwork != 35 ||
        s[0] != 8.0 || s[1] != 7.0 || s[2] != 6.0 ||
        superb[0] != 12.0 || superb[1] != 11.0) {
        fprintf(stderr, "[FAIL] DGESVD Fortran->CBLAS thunk did not propagate superb from work scratch\n");
        return 1;
    }

    printf("[PASS] DGESVD Fortran->CBLAS thunk performs the workspace query and propagates superb from work scratch\n");
    return 0;
}

static int check_dgesvd_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_dgesvd_fortran_slot_fn thunk = NULL;
    double a[9] = { 0.0 };
    double s[3] = { 0.0, 0.0, 0.0 };
    double u[9] = { 0.0 };
    double vt[9] = { 0.0 };
    double work[8] = { 0.0 };
    char jobu = 'A';
    char jobvt = 'A';
    int m = 3;
    int n = 3;
    int lda = 3;
    int ldu = 3;
    int ldvt = 3;
    int lwork = 8;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dgesvd_cblas_call, 0, sizeof(g_dgesvd_cblas_call));
    g_dgesvd_cblas_rc = 311;

    vtable.ext_ops[FB_OP_DGESVD][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_dgesvd_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_DGESVD);

    thunk = (fb_dgesvd_fortran_slot_fn)vtable.ext_ops[FB_OP_DGESVD][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DGESVD CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork,
          &info);
    if (info != 311 || g_dgesvd_cblas_call.called != 1 ||
        g_dgesvd_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_dgesvd_cblas_call.jobu != 'A' || g_dgesvd_cblas_call.jobvt != 'A' ||
        g_dgesvd_cblas_call.m != 3 || g_dgesvd_cblas_call.n != 3 ||
        g_dgesvd_cblas_call.lda != 3 || g_dgesvd_cblas_call.ldu != 3 ||
        g_dgesvd_cblas_call.ldvt != 3 || g_dgesvd_cblas_call.superb == NULL) {
        fprintf(stderr,
                "[FAIL] DGESVD CBLAS->Fortran thunk did not allocate and pass superb scratch (info=%d called=%d layout=%d jobu=%c jobvt=%c m=%d n=%d lda=%d ldu=%d ldvt=%d superb=%p)\n",
                info, g_dgesvd_cblas_call.called,
                (int)g_dgesvd_cblas_call.layout, g_dgesvd_cblas_call.jobu,
                g_dgesvd_cblas_call.jobvt, g_dgesvd_cblas_call.m,
                g_dgesvd_cblas_call.n, g_dgesvd_cblas_call.lda,
                g_dgesvd_cblas_call.ldu, g_dgesvd_cblas_call.ldvt,
                (void *)g_dgesvd_cblas_call.superb);
        return 1;
    }

    printf("[PASS] DGESVD CBLAS->Fortran thunk allocates the missing superb scratch for the generic double C ABI\n");
    return 0;
}

static int check_cgesvd_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_cgesvd_fn thunk = NULL;
    fb_complex_float_t a[9] = { 0 };
    float s[3] = { 0.0f, 0.0f, 0.0f };
    fb_complex_float_t u[9] = { 0 };
    fb_complex_float_t vt[9] = { 0 };
    float superb[2] = { -1.0f, -1.0f };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cgesvd_fortran_call, 0, sizeof(g_cgesvd_fortran_call));

    vtable.ext_ops[FB_OP_CGESVD][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_cgesvd_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CGESVD);

    thunk = (fb_cgesvd_fn)vtable.ext_ops[FB_OP_CGESVD][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CGESVD Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 'A', 'A', 3, 3, a, 3, s, u, 3, vt, 3, superb);
    if (info != 0 || g_cgesvd_fortran_call.query_calls != 1 ||
        g_cgesvd_fortran_call.exec_calls != 1 ||
        g_cgesvd_fortran_call.query_lwork != -1 ||
        g_cgesvd_fortran_call.exec_lwork != 26 ||
        s[0] != 6.0f || s[1] != 5.0f || s[2] != 4.0f ||
        superb[0] != 11.0f || superb[1] != 10.0f) {
        fprintf(stderr, "[FAIL] CGESVD Fortran->CBLAS thunk did not propagate superb from rwork scratch\n");
        return 1;
    }

    printf("[PASS] CGESVD Fortran->CBLAS thunk performs the workspace query and propagates superb from rwork scratch\n");
    return 0;
}

static int check_cgesvd_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_cgesvd_fortran_slot_fn thunk = NULL;
    fb_complex_float_t a[9] = { 0 };
    float s[3] = { 0.0f, 0.0f, 0.0f };
    fb_complex_float_t u[9] = { 0 };
    fb_complex_float_t vt[9] = { 0 };
    fb_complex_float_t work[8] = { 0 };
    float rwork[8] = { 0.0f };
    char jobu = 'A';
    char jobvt = 'A';
    int m = 3;
    int n = 3;
    int lda = 3;
    int ldu = 3;
    int ldvt = 3;
    int lwork = 8;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cgesvd_cblas_call, 0, sizeof(g_cgesvd_cblas_call));
    g_cgesvd_cblas_rc = 213;

    vtable.ext_ops[FB_OP_CGESVD][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_cgesvd_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CGESVD);

    thunk = (fb_cgesvd_fortran_slot_fn)vtable.ext_ops[FB_OP_CGESVD][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CGESVD CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, rwork, &info);
    if (info != 213 || g_cgesvd_cblas_call.called != 1 ||
        g_cgesvd_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_cgesvd_cblas_call.jobu != 'A' || g_cgesvd_cblas_call.jobvt != 'A' ||
        g_cgesvd_cblas_call.m != 3 || g_cgesvd_cblas_call.n != 3 ||
        g_cgesvd_cblas_call.lda != 3 || g_cgesvd_cblas_call.ldu != 3 ||
        g_cgesvd_cblas_call.ldvt != 3 || g_cgesvd_cblas_call.superb == NULL) {
        fprintf(stderr,
                "[FAIL] CGESVD CBLAS->Fortran thunk did not allocate and pass superb scratch (info=%d called=%d layout=%d jobu=%c jobvt=%c m=%d n=%d lda=%d ldu=%d ldvt=%d superb=%p)\n",
                info, g_cgesvd_cblas_call.called, (int)g_cgesvd_cblas_call.layout,
                g_cgesvd_cblas_call.jobu, g_cgesvd_cblas_call.jobvt,
                g_cgesvd_cblas_call.m, g_cgesvd_cblas_call.n,
                g_cgesvd_cblas_call.lda, g_cgesvd_cblas_call.ldu,
                g_cgesvd_cblas_call.ldvt, (void *)g_cgesvd_cblas_call.superb);
        return 1;
    }

    printf("[PASS] CGESVD CBLAS->Fortran thunk allocates the missing superb scratch for the generic complex C ABI\n");
    return 0;
}

static int check_zgesvd_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_zgesvd_fn thunk = NULL;
    fb_complex_double_t a[9] = { 0 };
    double s[3] = { 0.0, 0.0, 0.0 };
    fb_complex_double_t u[9] = { 0 };
    fb_complex_double_t vt[9] = { 0 };
    double superb[2] = { -1.0, -1.0 };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zgesvd_fortran_call, 0, sizeof(g_zgesvd_fortran_call));

    vtable.ext_ops[FB_OP_ZGESVD][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_zgesvd_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_ZGESVD);

    thunk = (fb_zgesvd_fn)vtable.ext_ops[FB_OP_ZGESVD][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZGESVD Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 'A', 'A', 3, 3, a, 3, s, u, 3, vt, 3,
                 superb);
    if (info != 0 || g_zgesvd_fortran_call.query_calls != 1 ||
        g_zgesvd_fortran_call.exec_calls != 1 ||
        g_zgesvd_fortran_call.query_lwork != -1 ||
        g_zgesvd_fortran_call.exec_lwork != 36 ||
        s[0] != 9.0 || s[1] != 8.0 || s[2] != 7.0 ||
        superb[0] != 15.0 || superb[1] != 14.0) {
        fprintf(stderr, "[FAIL] ZGESVD Fortran->CBLAS thunk did not propagate superb from rwork scratch\n");
        return 1;
    }

    printf("[PASS] ZGESVD Fortran->CBLAS thunk performs the workspace query and propagates superb from rwork scratch\n");
    return 0;
}

static int check_zgesvd_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_zgesvd_fortran_slot_fn thunk = NULL;
    fb_complex_double_t a[9] = { 0 };
    double s[3] = { 0.0, 0.0, 0.0 };
    fb_complex_double_t u[9] = { 0 };
    fb_complex_double_t vt[9] = { 0 };
    fb_complex_double_t work[8] = { 0 };
    double rwork[8] = { 0.0 };
    char jobu = 'A';
    char jobvt = 'A';
    int m = 3;
    int n = 3;
    int lda = 3;
    int ldu = 3;
    int ldvt = 3;
    int lwork = 8;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zgesvd_cblas_call, 0, sizeof(g_zgesvd_cblas_call));
    g_zgesvd_cblas_rc = 313;

    vtable.ext_ops[FB_OP_ZGESVD][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_zgesvd_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_ZGESVD);

    thunk = (fb_zgesvd_fortran_slot_fn)vtable.ext_ops[FB_OP_ZGESVD][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZGESVD CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork,
          rwork, &info);
    if (info != 313 || g_zgesvd_cblas_call.called != 1 ||
        g_zgesvd_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_zgesvd_cblas_call.jobu != 'A' || g_zgesvd_cblas_call.jobvt != 'A' ||
        g_zgesvd_cblas_call.m != 3 || g_zgesvd_cblas_call.n != 3 ||
        g_zgesvd_cblas_call.lda != 3 || g_zgesvd_cblas_call.ldu != 3 ||
        g_zgesvd_cblas_call.ldvt != 3 || g_zgesvd_cblas_call.superb == NULL) {
        fprintf(stderr,
                "[FAIL] ZGESVD CBLAS->Fortran thunk did not allocate and pass superb scratch (info=%d called=%d layout=%d jobu=%c jobvt=%c m=%d n=%d lda=%d ldu=%d ldvt=%d superb=%p)\n",
                info, g_zgesvd_cblas_call.called,
                (int)g_zgesvd_cblas_call.layout, g_zgesvd_cblas_call.jobu,
                g_zgesvd_cblas_call.jobvt, g_zgesvd_cblas_call.m,
                g_zgesvd_cblas_call.n, g_zgesvd_cblas_call.lda,
                g_zgesvd_cblas_call.ldu, g_zgesvd_cblas_call.ldvt,
                (void *)g_zgesvd_cblas_call.superb);
        return 1;
    }

    printf("[PASS] ZGESVD CBLAS->Fortran thunk allocates the missing superb scratch for the generic complex-double C ABI\n");
    return 0;
}

int main(void)
{
    if (check_sgesvd_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_sgesvd_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_dgesvd_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_dgesvd_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_cgesvd_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_cgesvd_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_zgesvd_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_zgesvd_cblas_to_fortran() != 0) {
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}