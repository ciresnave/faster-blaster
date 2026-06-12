#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef int (*fb_sstevx_fn)(fb_layout_t layout, char jobz, char range, int n,
                            float *d, float *e, float vl, float vu, int il,
                            int iu, float abstol, int *m, float *w, float *z,
                            int ldz, int *ifail);
typedef int (*fb_dstevx_fn)(fb_layout_t layout, char jobz, char range, int n,
                            double *d, double *e, double vl, double vu,
                            int il, int iu, double abstol, int *m, double *w,
                            double *z, int ldz, int *ifail);

typedef void (*fb_sstevx_fortran_slot_fn)(char *jobz, char *range, int *n,
                                          float *d, float *e, float *vl,
                                          float *vu, int *il, int *iu,
                                          float *abstol, int *m, float *w,
                                          float *z, int *ldz, float *work,
                                          int *iwork, int *ifail, int *info);
typedef void (*fb_dstevx_fortran_slot_fn)(char *jobz, char *range, int *n,
                                          double *d, double *e, double *vl,
                                          double *vu, int *il, int *iu,
                                          double *abstol, int *m, double *w,
                                          double *z, int *ldz, double *work,
                                          int *iwork, int *ifail, int *info);

static struct {
    int exec_calls;
    int saw_work;
    int saw_iwork;
} g_sstevx_fortran_call;

static struct {
    int exec_calls;
    int saw_work;
    int saw_iwork;
} g_dstevx_fortran_call;

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
} g_sstevx_cblas_call;

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
} g_dstevx_cblas_call;

static int g_sstevx_cblas_rc = 0;
static int g_dstevx_cblas_rc = 0;

static void stub_sstevx_fortran(char *jobz, char *range, int *n, float *d,
                                float *e, float *vl, float *vu, int *il,
                                int *iu, float *abstol, int *m, float *w,
                                float *z, int *ldz, float *work, int *iwork,
                                int *ifail, int *info)
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
    g_sstevx_fortran_call.exec_calls += 1;
    g_sstevx_fortran_call.saw_work = (work != NULL);
    g_sstevx_fortran_call.saw_iwork = (iwork != NULL);
    *m = 2;
    w[0] = 9.5f;
    w[1] = 10.5f;
    z[0] = 51.0f;
    ifail[0] = 1;
    ifail[1] = 0;
    *info = 0;
}

static void stub_dstevx_fortran(char *jobz, char *range, int *n, double *d,
                                double *e, double *vl, double *vu, int *il,
                                int *iu, double *abstol, int *m, double *w,
                                double *z, int *ldz, double *work, int *iwork,
                                int *ifail, int *info)
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
    g_dstevx_fortran_call.exec_calls += 1;
    g_dstevx_fortran_call.saw_work = (work != NULL);
    g_dstevx_fortran_call.saw_iwork = (iwork != NULL);
    *m = 2;
    w[0] = 11.5;
    w[1] = 12.5;
    z[0] = 61.0;
    ifail[0] = 2;
    ifail[1] = 0;
    *info = 0;
}

static int stub_sstevx_cblas(fb_layout_t layout, char jobz, char range, int n,
                             float *d, float *e, float vl, float vu, int il,
                             int iu, float abstol, int *m, float *w, float *z,
                             int ldz, int *ifail)
{
    (void)d;
    (void)e;
    g_sstevx_cblas_call.called += 1;
    g_sstevx_cblas_call.layout = layout;
    g_sstevx_cblas_call.jobz = jobz;
    g_sstevx_cblas_call.range = range;
    g_sstevx_cblas_call.n = n;
    g_sstevx_cblas_call.vl = vl;
    g_sstevx_cblas_call.vu = vu;
    g_sstevx_cblas_call.il = il;
    g_sstevx_cblas_call.iu = iu;
    g_sstevx_cblas_call.abstol = abstol;
    g_sstevx_cblas_call.ldz = ldz;
    *m = 2;
    w[0] = 13.5f;
    w[1] = 14.5f;
    z[0] = 71.0f;
    ifail[0] = 3;
    ifail[1] = 0;
    return g_sstevx_cblas_rc;
}

static int stub_dstevx_cblas(fb_layout_t layout, char jobz, char range, int n,
                             double *d, double *e, double vl, double vu,
                             int il, int iu, double abstol, int *m, double *w,
                             double *z, int ldz, int *ifail)
{
    (void)d;
    (void)e;
    g_dstevx_cblas_call.called += 1;
    g_dstevx_cblas_call.layout = layout;
    g_dstevx_cblas_call.jobz = jobz;
    g_dstevx_cblas_call.range = range;
    g_dstevx_cblas_call.n = n;
    g_dstevx_cblas_call.vl = vl;
    g_dstevx_cblas_call.vu = vu;
    g_dstevx_cblas_call.il = il;
    g_dstevx_cblas_call.iu = iu;
    g_dstevx_cblas_call.abstol = abstol;
    g_dstevx_cblas_call.ldz = ldz;
    *m = 2;
    w[0] = 15.5;
    w[1] = 16.5;
    z[0] = 81.0;
    ifail[0] = 4;
    ifail[1] = 0;
    return g_dstevx_cblas_rc;
}

static int check_sstevx_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_sstevx_fn thunk = NULL;
    float d[3] = { 0.0f, 0.0f, 0.0f };
    float e[2] = { 0.0f, 0.0f };
    float w[3] = { 0.0f, 0.0f, 0.0f };
    float z[9] = { 0.0f };
    int ifail[3] = { 0, 0, 0 };
    int m = 0;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sstevx_fortran_call, 0, sizeof(g_sstevx_fortran_call));

    vtable.ext_ops[FB_OP_SSTEVX][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_sstevx_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SSTEVX);

    thunk = (fb_sstevx_fn)vtable.ext_ops[FB_OP_SSTEVX][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SSTEVX Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 'V', 'I', 3, d, e, 1.0f, 6.0f, 1, 2,
                 0.25f, &m, w, z, 3, ifail);
    if (info != 0 || g_sstevx_fortran_call.exec_calls != 1 ||
        !g_sstevx_fortran_call.saw_work || !g_sstevx_fortran_call.saw_iwork ||
        m != 2 || w[0] != 9.5f || w[1] != 10.5f || z[0] != 51.0f ||
        ifail[0] != 1) {
        fprintf(stderr, "[FAIL] SSTEVX Fortran->CBLAS thunk did not allocate fixed workspaces and forward selective tridiagonal outputs\n");
        return 1;
    }

    printf("[PASS] SSTEVX Fortran->CBLAS thunk allocates fixed workspaces and forwards selective tridiagonal eigen outputs\n");
    return 0;
}

static int check_sstevx_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_sstevx_fortran_slot_fn thunk = NULL;
    float d[3] = { 0.0f, 0.0f, 0.0f };
    float e[2] = { 0.0f, 0.0f };
    float w[3] = { 0.0f, 0.0f, 0.0f };
    float z[9] = { 0.0f };
    float work[16] = { 0.0f };
    int iwork[16] = { 0 };
    int ifail[3] = { 0, 0, 0 };
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
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sstevx_cblas_call, 0, sizeof(g_sstevx_cblas_call));
    g_sstevx_cblas_rc = 257;

    vtable.ext_ops[FB_OP_SSTEVX][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_sstevx_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SSTEVX);

    thunk = (fb_sstevx_fortran_slot_fn)vtable.ext_ops[FB_OP_SSTEVX][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SSTEVX CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&jobz, &range, &n, d, e, &vl, &vu, &il, &iu, &abstol, &m, w, z,
          &ldz, work, iwork, ifail, &info);
    if (info != 257 || g_sstevx_cblas_call.called != 1 ||
        g_sstevx_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_sstevx_cblas_call.jobz != 'V' || g_sstevx_cblas_call.range != 'I' ||
        g_sstevx_cblas_call.n != 3 || g_sstevx_cblas_call.vl != 1.0f ||
        g_sstevx_cblas_call.vu != 6.0f || g_sstevx_cblas_call.il != 1 ||
        g_sstevx_cblas_call.iu != 2 || g_sstevx_cblas_call.abstol != 0.25f ||
        g_sstevx_cblas_call.ldz != 3 || m != 2 || w[0] != 13.5f ||
        w[1] != 14.5f || z[0] != 71.0f || ifail[0] != 3) {
        fprintf(stderr, "[FAIL] SSTEVX CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] SSTEVX CBLAS->Fortran thunk maps selective tridiagonal eigen arguments into the C entry\n");
    return 0;
}

static int check_dstevx_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_dstevx_fn thunk = NULL;
    double d[3] = { 0.0, 0.0, 0.0 };
    double e[2] = { 0.0, 0.0 };
    double w[3] = { 0.0, 0.0, 0.0 };
    double z[9] = { 0.0 };
    int ifail[3] = { 0, 0, 0 };
    int m = 0;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dstevx_fortran_call, 0, sizeof(g_dstevx_fortran_call));

    vtable.ext_ops[FB_OP_DSTEVX][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_dstevx_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_DSTEVX);

    thunk = (fb_dstevx_fn)vtable.ext_ops[FB_OP_DSTEVX][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DSTEVX Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 'V', 'I', 3, d, e, 2.0, 7.0, 1, 2, 0.5,
                 &m, w, z, 3, ifail);
    if (info != 0 || g_dstevx_fortran_call.exec_calls != 1 ||
        !g_dstevx_fortran_call.saw_work || !g_dstevx_fortran_call.saw_iwork ||
        m != 2 || w[0] != 11.5 || w[1] != 12.5 || z[0] != 61.0 ||
        ifail[0] != 2) {
        fprintf(stderr, "[FAIL] DSTEVX Fortran->CBLAS thunk did not allocate fixed workspaces and forward double-precision selective tridiagonal outputs\n");
        return 1;
    }

    printf("[PASS] DSTEVX Fortran->CBLAS thunk allocates fixed workspaces and forwards double-precision selective tridiagonal eigen outputs\n");
    return 0;
}

static int check_dstevx_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_dstevx_fortran_slot_fn thunk = NULL;
    double d[3] = { 0.0, 0.0, 0.0 };
    double e[2] = { 0.0, 0.0 };
    double w[3] = { 0.0, 0.0, 0.0 };
    double z[9] = { 0.0 };
    double work[16] = { 0.0 };
    int iwork[16] = { 0 };
    int ifail[3] = { 0, 0, 0 };
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
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dstevx_cblas_call, 0, sizeof(g_dstevx_cblas_call));
    g_dstevx_cblas_rc = 259;

    vtable.ext_ops[FB_OP_DSTEVX][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_dstevx_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_DSTEVX);

    thunk = (fb_dstevx_fortran_slot_fn)vtable.ext_ops[FB_OP_DSTEVX][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DSTEVX CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&jobz, &range, &n, d, e, &vl, &vu, &il, &iu, &abstol, &m, w, z,
          &ldz, work, iwork, ifail, &info);
    if (info != 259 || g_dstevx_cblas_call.called != 1 ||
        g_dstevx_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_dstevx_cblas_call.jobz != 'V' || g_dstevx_cblas_call.range != 'I' ||
        g_dstevx_cblas_call.n != 3 || g_dstevx_cblas_call.vl != 2.0 ||
        g_dstevx_cblas_call.vu != 7.0 || g_dstevx_cblas_call.il != 1 ||
        g_dstevx_cblas_call.iu != 2 || g_dstevx_cblas_call.abstol != 0.5 ||
        g_dstevx_cblas_call.ldz != 3 || m != 2 || w[0] != 15.5 ||
        w[1] != 16.5 || z[0] != 81.0 || ifail[0] != 4) {
        fprintf(stderr, "[FAIL] DSTEVX CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] DSTEVX CBLAS->Fortran thunk maps double-precision selective tridiagonal eigen arguments into the C entry\n");
    return 0;
}

int main(void)
{
    if (check_sstevx_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_sstevx_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_dstevx_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_dstevx_cblas_to_fortran() != 0) {
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}