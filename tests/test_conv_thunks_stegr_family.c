#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef int (*fb_sstegr_fn)(fb_layout_t layout, char jobz, char range, int n,
                            float *d, float *e, float vl, float vu, int il,
                            int iu, float abstol, int *m, float *w, float *z,
                            int ldz, int *isuppz);
typedef int (*fb_dstegr_fn)(fb_layout_t layout, char jobz, char range, int n,
                            double *d, double *e, double vl, double vu,
                            int il, int iu, double abstol, int *m, double *w,
                            double *z, int ldz, int *isuppz);

typedef void (*fb_sstegr_fortran_slot_fn)(char *jobz, char *range, int *n,
                                          float *d, float *e, float *vl,
                                          float *vu, int *il, int *iu,
                                          float *abstol, int *m, float *w,
                                          float *z, int *ldz, int *isuppz,
                                          float *work, int *lwork, int *iwork,
                                          int *liwork, int *info);
typedef void (*fb_dstegr_fortran_slot_fn)(char *jobz, char *range, int *n,
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
    int exec_ldz;
} g_sstegr_fortran_call;

static struct {
    int query_calls;
    int exec_calls;
    int query_lwork;
    int exec_lwork;
    int exec_liwork;
    int exec_ldz;
} g_dstegr_fortran_call;

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
} g_sstegr_cblas_call;

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
} g_dstegr_cblas_call;

static int g_sstegr_cblas_rc = 0;
static int g_dstegr_cblas_rc = 0;

static void stub_sstegr_fortran(char *jobz, char *range, int *n, float *d,
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
    if (*lwork == -1 || *liwork == -1) {
        g_sstegr_fortran_call.query_calls += 1;
        g_sstegr_fortran_call.query_lwork = *lwork;
        work[0] = 33.0f;
        *iwork = 17;
        *info = 0;
        return;
    }

    g_sstegr_fortran_call.exec_calls += 1;
    g_sstegr_fortran_call.exec_lwork = *lwork;
    g_sstegr_fortran_call.exec_liwork = *liwork;
    g_sstegr_fortran_call.exec_ldz = *ldz;
    *m = 2;
    w[0] = 1.5f;
    w[1] = 2.5f;
    z[0] = 11.0f;
    z[1] = 12.0f;
    z[2] = 13.0f;
    z[3] = 21.0f;
    z[4] = 22.0f;
    z[5] = 23.0f;
    isuppz[0] = 1;
    isuppz[1] = 3;
    *info = 0;
}

static void stub_dstegr_fortran(char *jobz, char *range, int *n, double *d,
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
    if (*lwork == -1 || *liwork == -1) {
        g_dstegr_fortran_call.query_calls += 1;
        g_dstegr_fortran_call.query_lwork = *lwork;
        work[0] = 34.0;
        *iwork = 19;
        *info = 0;
        return;
    }

    g_dstegr_fortran_call.exec_calls += 1;
    g_dstegr_fortran_call.exec_lwork = *lwork;
    g_dstegr_fortran_call.exec_liwork = *liwork;
    g_dstegr_fortran_call.exec_ldz = *ldz;
    *m = 2;
    w[0] = 3.5;
    w[1] = 4.5;
    z[0] = 31.0;
    z[1] = 32.0;
    z[2] = 33.0;
    z[3] = 41.0;
    z[4] = 42.0;
    z[5] = 43.0;
    isuppz[0] = 2;
    isuppz[1] = 4;
    *info = 0;
}

static int stub_sstegr_cblas(fb_layout_t layout, char jobz, char range, int n,
                             float *d, float *e, float vl, float vu, int il,
                             int iu, float abstol, int *m, float *w, float *z,
                             int ldz, int *isuppz)
{
    (void)d;
    (void)e;
    g_sstegr_cblas_call.called += 1;
    g_sstegr_cblas_call.layout = layout;
    g_sstegr_cblas_call.jobz = jobz;
    g_sstegr_cblas_call.range = range;
    g_sstegr_cblas_call.n = n;
    g_sstegr_cblas_call.vl = vl;
    g_sstegr_cblas_call.vu = vu;
    g_sstegr_cblas_call.il = il;
    g_sstegr_cblas_call.iu = iu;
    g_sstegr_cblas_call.abstol = abstol;
    g_sstegr_cblas_call.ldz = ldz;
    *m = 2;
    w[0] = 5.5f;
    w[1] = 6.5f;
    z[0] = 51.0f;
    isuppz[0] = 1;
    isuppz[1] = 2;
    return g_sstegr_cblas_rc;
}

static int stub_dstegr_cblas(fb_layout_t layout, char jobz, char range, int n,
                             double *d, double *e, double vl, double vu,
                             int il, int iu, double abstol, int *m, double *w,
                             double *z, int ldz, int *isuppz)
{
    (void)d;
    (void)e;
    g_dstegr_cblas_call.called += 1;
    g_dstegr_cblas_call.layout = layout;
    g_dstegr_cblas_call.jobz = jobz;
    g_dstegr_cblas_call.range = range;
    g_dstegr_cblas_call.n = n;
    g_dstegr_cblas_call.vl = vl;
    g_dstegr_cblas_call.vu = vu;
    g_dstegr_cblas_call.il = il;
    g_dstegr_cblas_call.iu = iu;
    g_dstegr_cblas_call.abstol = abstol;
    g_dstegr_cblas_call.ldz = ldz;
    *m = 2;
    w[0] = 7.5;
    w[1] = 8.5;
    z[0] = 61.0;
    isuppz[0] = 2;
    isuppz[1] = 3;
    return g_dstegr_cblas_rc;
}

static int check_sstegr_fortran_to_cblas(void)
{
    static const float expected_z[9] = {
        11.0f, 21.0f, 0.0f,
        12.0f, 22.0f, 0.0f,
        13.0f, 23.0f, 0.0f
    };
    fb_backend_vtable_t vtable;
    fb_sstegr_fn thunk = NULL;
    float d[3] = { 0.0f, 0.0f, 0.0f };
    float e[2] = { 0.0f, 0.0f };
    float w[3] = { 0.0f, 0.0f, 0.0f };
    float z[9] = { 0.0f };
    int isuppz[6] = { 0, 0, 0, 0, 0, 0 };
    int m = 0;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sstegr_fortran_call, 0, sizeof(g_sstegr_fortran_call));

    vtable.ext_ops[FB_OP_SSTEGR][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_sstegr_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SSTEGR);

    thunk = (fb_sstegr_fn)vtable.ext_ops[FB_OP_SSTEGR][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SSTEGR Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 'V', 'I', 3, d, e, 1.0f, 6.0f, 1, 2,
                 0.25f, &m, w, z, 3, isuppz);
    if (info != 0 || g_sstegr_fortran_call.query_calls != 1 ||
        g_sstegr_fortran_call.exec_calls != 1 ||
        g_sstegr_fortran_call.query_lwork != -1 ||
        g_sstegr_fortran_call.exec_lwork != 33 ||
        g_sstegr_fortran_call.exec_liwork != 17 ||
        g_sstegr_fortran_call.exec_ldz != 3 || m != 2 ||
        w[0] != 1.5f || w[1] != 2.5f ||
        memcmp(z, expected_z, sizeof(expected_z)) != 0 ||
        isuppz[0] != 1 || isuppz[1] != 3) {
        fprintf(stderr, "[FAIL] SSTEGR Fortran->CBLAS thunk did not preserve query semantics or row-major vector export\n");
        return 1;
    }

    printf("[PASS] SSTEGR Fortran->CBLAS thunk performs the workspace queries and exports row-major eigenvectors\n");
    return 0;
}

static int check_sstegr_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_sstegr_fortran_slot_fn thunk = NULL;
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
    memset(&g_sstegr_cblas_call, 0, sizeof(g_sstegr_cblas_call));
    g_sstegr_cblas_rc = 281;

    vtable.ext_ops[FB_OP_SSTEGR][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_sstegr_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SSTEGR);

    thunk = (fb_sstegr_fortran_slot_fn)vtable.ext_ops[FB_OP_SSTEGR][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SSTEGR CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&jobz, &range, &n, d, e, &vl, &vu, &il, &iu, &abstol, &m, w, z,
          &ldz, isuppz, work, &lwork, iwork, &liwork, &info);
    if (info != 281 || g_sstegr_cblas_call.called != 1 ||
        g_sstegr_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_sstegr_cblas_call.jobz != 'V' || g_sstegr_cblas_call.range != 'I' ||
        g_sstegr_cblas_call.n != 3 || g_sstegr_cblas_call.vl != 1.0f ||
        g_sstegr_cblas_call.vu != 6.0f || g_sstegr_cblas_call.il != 1 ||
        g_sstegr_cblas_call.iu != 2 || g_sstegr_cblas_call.abstol != 0.25f ||
        g_sstegr_cblas_call.ldz != 3 || m != 2 || w[0] != 5.5f ||
        w[1] != 6.5f || z[0] != 51.0f || isuppz[0] != 1 ||
        isuppz[1] != 2) {
        fprintf(stderr, "[FAIL] SSTEGR CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] SSTEGR CBLAS->Fortran thunk maps tridiagonal MRRR arguments into the C entry\n");
    return 0;
}

static int check_dstegr_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_dstegr_fn thunk = NULL;
    double d[3] = { 0.0, 0.0, 0.0 };
    double e[2] = { 0.0, 0.0 };
    double w[3] = { 0.0, 0.0, 0.0 };
    double z[9] = { 0.0 };
    int isuppz[6] = { 0, 0, 0, 0, 0, 0 };
    int m = 0;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dstegr_fortran_call, 0, sizeof(g_dstegr_fortran_call));

    vtable.ext_ops[FB_OP_DSTEGR][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_dstegr_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_DSTEGR);

    thunk = (fb_dstegr_fn)vtable.ext_ops[FB_OP_DSTEGR][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DSTEGR Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, 'V', 'I', 3, d, e, 2.0, 7.0, 1, 2, 0.5,
                 &m, w, z, 3, isuppz);
    if (info != 0 || g_dstegr_fortran_call.query_calls != 1 ||
        g_dstegr_fortran_call.exec_calls != 1 ||
        g_dstegr_fortran_call.query_lwork != -1 ||
        g_dstegr_fortran_call.exec_lwork != 34 ||
        g_dstegr_fortran_call.exec_liwork != 19 ||
        g_dstegr_fortran_call.exec_ldz != 3 || m != 2 ||
        w[0] != 3.5 || w[1] != 4.5 || z[0] != 31.0 ||
        isuppz[0] != 2 || isuppz[1] != 4) {
        fprintf(stderr, "[FAIL] DSTEGR Fortran->CBLAS thunk did not preserve double-precision query semantics\n");
        return 1;
    }

    printf("[PASS] DSTEGR Fortran->CBLAS thunk performs the workspace queries and forwards double-precision tridiagonal outputs\n");
    return 0;
}

static int check_dstegr_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_dstegr_fortran_slot_fn thunk = NULL;
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
    memset(&g_dstegr_cblas_call, 0, sizeof(g_dstegr_cblas_call));
    g_dstegr_cblas_rc = 283;

    vtable.ext_ops[FB_OP_DSTEGR][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_dstegr_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_DSTEGR);

    thunk = (fb_dstegr_fortran_slot_fn)vtable.ext_ops[FB_OP_DSTEGR][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DSTEGR CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&jobz, &range, &n, d, e, &vl, &vu, &il, &iu, &abstol, &m, w, z,
          &ldz, isuppz, work, &lwork, iwork, &liwork, &info);
    if (info != 283 || g_dstegr_cblas_call.called != 1 ||
        g_dstegr_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_dstegr_cblas_call.jobz != 'V' || g_dstegr_cblas_call.range != 'I' ||
        g_dstegr_cblas_call.n != 3 || g_dstegr_cblas_call.vl != 2.0 ||
        g_dstegr_cblas_call.vu != 7.0 || g_dstegr_cblas_call.il != 1 ||
        g_dstegr_cblas_call.iu != 2 || g_dstegr_cblas_call.abstol != 0.5 ||
        g_dstegr_cblas_call.ldz != 3 || m != 2 || w[0] != 7.5 ||
        w[1] != 8.5 || z[0] != 61.0 || isuppz[0] != 2 ||
        isuppz[1] != 3) {
        fprintf(stderr, "[FAIL] DSTEGR CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] DSTEGR CBLAS->Fortran thunk maps double-precision tridiagonal MRRR arguments into the C entry\n");
    return 0;
}

int main(void)
{
    if (check_sstegr_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_sstegr_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_dstegr_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_dstegr_cblas_to_fortran() != 0) {
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}