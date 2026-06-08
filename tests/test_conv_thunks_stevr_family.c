#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef int (*fb_sstevr_fn)(fb_layout_t layout, char jobz, char range, int n,
                            float *d, float *e, float vl, float vu, int il,
                            int iu, float abstol, int *m, float *w, float *z,
                            int ldz, int *isuppz);
typedef int (*fb_dstevr_fn)(fb_layout_t layout, char jobz, char range, int n,
                            double *d, double *e, double vl, double vu, int il,
                            int iu, double abstol, int *m, double *w,
                            double *z, int ldz, int *isuppz);

typedef void (*fb_sstevr_fortran_slot_fn)(char *jobz, char *range, int *n,
                                          float *d, float *e, float *vl,
                                          float *vu, int *il, int *iu,
                                          float *abstol, int *m, float *w,
                                          float *z, int *ldz, int *isuppz,
                                          float *work, int *lwork, int *iwork,
                                          int *liwork, int *info);
typedef void (*fb_dstevr_fortran_slot_fn)(char *jobz, char *range, int *n,
                                          double *d, double *e, double *vl,
                                          double *vu, int *il, int *iu,
                                          double *abstol, int *m, double *w,
                                          double *z, int *ldz, int *isuppz,
                                          double *work, int *lwork,
                                          int *iwork, int *liwork, int *info);

static struct {
    int query_calls;
    int exec_calls;
    int query_lwork;
    int exec_lwork;
    int exec_liwork;
} g_sstevr_fortran_call;

static struct {
    int query_calls;
    int exec_calls;
    int query_lwork;
    int exec_lwork;
    int exec_liwork;
} g_dstevr_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    char jobz;
    char range;
    int n;
    float vl;
    float vu;
    int il;
    int iu;
    float abstol;
    int ldz;
} g_sstevr_cblas_call;

static struct {
    int called;
    fb_layout_t layout;
    char jobz;
    char range;
    int n;
    double vl;
    double vu;
    int il;
    int iu;
    double abstol;
    int ldz;
} g_dstevr_cblas_call;

static int g_sstevr_cblas_rc = 0;
static int g_dstevr_cblas_rc = 0;

static void stub_sstevr_fortran(char *jobz, char *range, int *n, float *d,
                                float *e, float *vl, float *vu, int *il,
                                int *iu, float *abstol, int *m, float *w,
                                float *z, int *ldz, int *isuppz, float *work,
                                int *lwork, int *iwork, int *liwork,
                                int *info)
{
    (void)jobz;
    (void)range;
    (void)n;
    (void)d;
    (void)e;
    (void)vl;
    (void)vu;
    (void)il;
    (void)iu;
    (void)abstol;
    (void)ldz;
    if (*lwork == -1 || *liwork == -1) {
        g_sstevr_fortran_call.query_calls += 1;
        g_sstevr_fortran_call.query_lwork = *lwork;
        work[0] = 33.0f;
        *iwork = 17;
        *info = 0;
        return;
    }

    g_sstevr_fortran_call.exec_calls += 1;
    g_sstevr_fortran_call.exec_lwork = *lwork;
    g_sstevr_fortran_call.exec_liwork = *liwork;
    *m = 2;
    w[0] = 1.5f;
    w[1] = 2.5f;
    z[0] = 11.0f;
    isuppz[0] = 1;
    isuppz[1] = 3;
    *info = 0;
}

static void stub_dstevr_fortran(char *jobz, char *range, int *n, double *d,
                                double *e, double *vl, double *vu, int *il,
                                int *iu, double *abstol, int *m, double *w,
                                double *z, int *ldz, int *isuppz, double *work,
                                int *lwork, int *iwork, int *liwork,
                                int *info)
{
    (void)jobz;
    (void)range;
    (void)n;
    (void)d;
    (void)e;
    (void)vl;
    (void)vu;
    (void)il;
    (void)iu;
    (void)abstol;
    (void)ldz;
    if (*lwork == -1 || *liwork == -1) {
        g_dstevr_fortran_call.query_calls += 1;
        g_dstevr_fortran_call.query_lwork = *lwork;
        work[0] = 34.0;
        *iwork = 19;
        *info = 0;
        return;
    }

    g_dstevr_fortran_call.exec_calls += 1;
    g_dstevr_fortran_call.exec_lwork = *lwork;
    g_dstevr_fortran_call.exec_liwork = *liwork;
    *m = 2;
    w[0] = 3.5;
    w[1] = 4.5;
    z[0] = 21.0;
    isuppz[0] = 2;
    isuppz[1] = 4;
    *info = 0;
}

static int stub_sstevr_cblas(fb_layout_t layout, char jobz, char range, int n,
                             float *d, float *e, float vl, float vu, int il,
                             int iu, float abstol, int *m, float *w, float *z,
                             int ldz, int *isuppz)
{
    (void)d;
    (void)e;
    g_sstevr_cblas_call.called += 1;
    g_sstevr_cblas_call.layout = layout;
    g_sstevr_cblas_call.jobz = jobz;
    g_sstevr_cblas_call.range = range;
    g_sstevr_cblas_call.n = n;
    g_sstevr_cblas_call.vl = vl;
    g_sstevr_cblas_call.vu = vu;
    g_sstevr_cblas_call.il = il;
    g_sstevr_cblas_call.iu = iu;
    g_sstevr_cblas_call.abstol = abstol;
    g_sstevr_cblas_call.ldz = ldz;
    *m = 2;
    w[0] = 5.5f;
    w[1] = 6.5f;
    z[0] = 31.0f;
    isuppz[0] = 1;
    isuppz[1] = 2;
    return g_sstevr_cblas_rc;
}

static int stub_dstevr_cblas(fb_layout_t layout, char jobz, char range, int n,
                             double *d, double *e, double vl, double vu,
                             int il, int iu, double abstol, int *m, double *w,
                             double *z, int ldz, int *isuppz)
{
    (void)d;
    (void)e;
    g_dstevr_cblas_call.called += 1;
    g_dstevr_cblas_call.layout = layout;
    g_dstevr_cblas_call.jobz = jobz;
    g_dstevr_cblas_call.range = range;
    g_dstevr_cblas_call.n = n;
    g_dstevr_cblas_call.vl = vl;
    g_dstevr_cblas_call.vu = vu;
    g_dstevr_cblas_call.il = il;
    g_dstevr_cblas_call.iu = iu;
    g_dstevr_cblas_call.abstol = abstol;
    g_dstevr_cblas_call.ldz = ldz;
    *m = 2;
    w[0] = 7.5;
    w[1] = 8.5;
    z[0] = 41.0;
    isuppz[0] = 2;
    isuppz[1] = 3;
    return g_dstevr_cblas_rc;
}

static int check_sstevr_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_sstevr_fn thunk = NULL;
    float d[3] = { 0.0f, 0.0f, 0.0f };
    float e[2] = { 0.0f, 0.0f };
    float w[3] = { 0.0f, 0.0f, 0.0f };
    float z[9] = { 0.0f };
    int isuppz[6] = { 0, 0, 0, 0, 0, 0 };
    int m = 0;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sstevr_fortran_call, 0, sizeof(g_sstevr_fortran_call));

    vtable.ext_ops[FB_OP_SSTEVR][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_sstevr_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SSTEVR);

    thunk = (fb_sstevr_fn)vtable.ext_ops[FB_OP_SSTEVR][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SSTEVR Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 'V', 'I', 3, d, e, 1.0f, 6.0f, 1, 2,
                 0.25f, &m, w, z, 3, isuppz);
    if (info != 0 || g_sstevr_fortran_call.query_calls != 1 ||
        g_sstevr_fortran_call.exec_calls != 1 ||
        g_sstevr_fortran_call.query_lwork != -1 ||
        g_sstevr_fortran_call.exec_lwork != 33 ||
        g_sstevr_fortran_call.exec_liwork != 17 || m != 2 ||
        w[0] != 1.5f || w[1] != 2.5f || z[0] != 11.0f ||
        isuppz[0] != 1 || isuppz[1] != 3) {
        fprintf(stderr, "[FAIL] SSTEVR Fortran->CBLAS thunk did not preserve tridiagonal RRR query semantics\n");
        return 1;
    }

    printf("[PASS] SSTEVR Fortran->CBLAS thunk performs the workspace queries and forwards tridiagonal RRR outputs\n");
    return 0;
}

static int check_sstevr_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_sstevr_fortran_slot_fn thunk = NULL;
    float d[3] = { 0.0f, 0.0f, 0.0f };
    float e[2] = { 0.0f, 0.0f };
    float w[3] = { 0.0f, 0.0f, 0.0f };
    float z[9] = { 0.0f };
    float work[16] = { 0.0f };
    int iwork[16] = { 0 };
    int isuppz[6] = { 0, 0, 0, 0, 0, 0 };
    char jobz = 'V';
    char range = 'I';
    int n = 3;
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
    memset(&g_sstevr_cblas_call, 0, sizeof(g_sstevr_cblas_call));
    g_sstevr_cblas_rc = 251;

    vtable.ext_ops[FB_OP_SSTEVR][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_sstevr_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SSTEVR);

    thunk = (fb_sstevr_fortran_slot_fn)vtable.ext_ops[FB_OP_SSTEVR][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SSTEVR CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&jobz, &range, &n, d, e, &vl, &vu, &il, &iu, &abstol, &m, w, z,
          &ldz, isuppz, work, &lwork, iwork, &liwork, &info);
    if (info != 251 || g_sstevr_cblas_call.called != 1 ||
        g_sstevr_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_sstevr_cblas_call.jobz != 'V' || g_sstevr_cblas_call.range != 'I' ||
        g_sstevr_cblas_call.n != 3 || g_sstevr_cblas_call.vl != 1.0f ||
        g_sstevr_cblas_call.vu != 6.0f || g_sstevr_cblas_call.il != 1 ||
        g_sstevr_cblas_call.iu != 2 || g_sstevr_cblas_call.abstol != 0.25f ||
        g_sstevr_cblas_call.ldz != 3 || m != 2 || w[0] != 5.5f ||
        w[1] != 6.5f || z[0] != 31.0f || isuppz[0] != 1 ||
        isuppz[1] != 2) {
        fprintf(stderr, "[FAIL] SSTEVR CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] SSTEVR CBLAS->Fortran thunk maps tridiagonal RRR arguments into the C entry\n");
    return 0;
}

static int check_dstevr_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_dstevr_fn thunk = NULL;
    double d[3] = { 0.0, 0.0, 0.0 };
    double e[2] = { 0.0, 0.0 };
    double w[3] = { 0.0, 0.0, 0.0 };
    double z[9] = { 0.0 };
    int isuppz[6] = { 0, 0, 0, 0, 0, 0 };
    int m = 0;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dstevr_fortran_call, 0, sizeof(g_dstevr_fortran_call));

    vtable.ext_ops[FB_OP_DSTEVR][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_dstevr_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_DSTEVR);

    thunk = (fb_dstevr_fn)vtable.ext_ops[FB_OP_DSTEVR][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DSTEVR Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 'V', 'I', 3, d, e, 2.0, 7.0, 1, 2, 0.5,
                 &m, w, z, 3, isuppz);
    if (info != 0 || g_dstevr_fortran_call.query_calls != 1 ||
        g_dstevr_fortran_call.exec_calls != 1 ||
        g_dstevr_fortran_call.query_lwork != -1 ||
        g_dstevr_fortran_call.exec_lwork != 34 ||
        g_dstevr_fortran_call.exec_liwork != 19 || m != 2 ||
        w[0] != 3.5 || w[1] != 4.5 || z[0] != 21.0 ||
        isuppz[0] != 2 || isuppz[1] != 4) {
        fprintf(stderr, "[FAIL] DSTEVR Fortran->CBLAS thunk did not preserve double-precision tridiagonal RRR query semantics\n");
        return 1;
    }

    printf("[PASS] DSTEVR Fortran->CBLAS thunk performs the workspace queries and forwards double-precision tridiagonal RRR outputs\n");
    return 0;
}

static int check_dstevr_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_dstevr_fortran_slot_fn thunk = NULL;
    double d[3] = { 0.0, 0.0, 0.0 };
    double e[2] = { 0.0, 0.0 };
    double w[3] = { 0.0, 0.0, 0.0 };
    double z[9] = { 0.0 };
    double work[16] = { 0.0 };
    int iwork[16] = { 0 };
    int isuppz[6] = { 0, 0, 0, 0, 0, 0 };
    char jobz = 'V';
    char range = 'I';
    int n = 3;
    double vl = 2.0;
    double vu = 7.0;
    int il = 1;
    int iu = 2;
    double abstol = 0.5;
    int m = 0;
    int ldz = 3;
    int lwork = 16;
    int liwork = 16;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dstevr_cblas_call, 0, sizeof(g_dstevr_cblas_call));
    g_dstevr_cblas_rc = 253;

    vtable.ext_ops[FB_OP_DSTEVR][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_dstevr_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_DSTEVR);

    thunk = (fb_dstevr_fortran_slot_fn)vtable.ext_ops[FB_OP_DSTEVR][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DSTEVR CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&jobz, &range, &n, d, e, &vl, &vu, &il, &iu, &abstol, &m, w, z,
          &ldz, isuppz, work, &lwork, iwork, &liwork, &info);
    if (info != 253 || g_dstevr_cblas_call.called != 1 ||
        g_dstevr_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_dstevr_cblas_call.jobz != 'V' || g_dstevr_cblas_call.range != 'I' ||
        g_dstevr_cblas_call.n != 3 || g_dstevr_cblas_call.vl != 2.0 ||
        g_dstevr_cblas_call.vu != 7.0 || g_dstevr_cblas_call.il != 1 ||
        g_dstevr_cblas_call.iu != 2 || g_dstevr_cblas_call.abstol != 0.5 ||
        g_dstevr_cblas_call.ldz != 3 || m != 2 || w[0] != 7.5 ||
        w[1] != 8.5 || z[0] != 41.0 || isuppz[0] != 2 ||
        isuppz[1] != 3) {
        fprintf(stderr, "[FAIL] DSTEVR CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] DSTEVR CBLAS->Fortran thunk maps double-precision tridiagonal RRR arguments into the C entry\n");
    return 0;
}

int main(void)
{
    if (check_sstevr_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_sstevr_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_dstevr_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_dstevr_cblas_to_fortran() != 0) {
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}