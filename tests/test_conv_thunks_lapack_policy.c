#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

static struct {
    int query_calls;
    int exec_calls;
    char jobvl;
    char jobvr;
    float *vl;
    float *vr;
    int ldvl;
    int ldvr;
} g_sgeev_calls;

static struct {
    int query_calls;
    int exec_calls;
    char jobu;
    char jobvt;
    float *u;
    float *vt;
    int ldu;
    int ldvt;
} g_sgesvd_calls;

static struct {
    int query_calls;
    int exec_calls;
    char jobz;
    float *u;
    float *vt;
    int ldu;
    int ldvt;
    int *query_iwork;
    int *exec_iwork;
} g_sgesdd_calls;

static int g_sstev_cblas_calls = 0;

typedef void (*fb_sstev_legacy_c2f_thunk_fn)(int layout, char jobz, int n,
                                             float *d, float *e, float *z,
                                             int ldz, float *work, int *info);

static void stub_sgeev_fortran(char *jobvl, char *jobvr, int *n, float *a,
                               int *lda, float *wr, float *wi, float *vl,
                               int *ldvl, float *vr, int *ldvr, float *work,
                               int *lwork, int *info)
{
    (void)n;
    (void)a;
    (void)lda;
    (void)wr;
    (void)wi;

    if (*lwork == -1) {
        g_sgeev_calls.query_calls += 1;
        *work = 24.0f;
        *info = 0;
        return;
    }

    g_sgeev_calls.exec_calls += 1;
    g_sgeev_calls.jobvl = *jobvl;
    g_sgeev_calls.jobvr = *jobvr;
    g_sgeev_calls.vl = vl;
    g_sgeev_calls.vr = vr;
    g_sgeev_calls.ldvl = *ldvl;
    g_sgeev_calls.ldvr = *ldvr;
    *info = 0;
}

static void stub_sgesvd_fortran(char *jobu, char *jobvt, int *m, int *n,
                                float *a, int *lda, float *s, float *u,
                                int *ldu, float *vt, int *ldvt, float *work,
                                int *lwork, int *info)
{
    (void)m;
    (void)n;
    (void)a;
    (void)lda;
    (void)s;

    if (*lwork == -1) {
        g_sgesvd_calls.query_calls += 1;
        *work = 40.0f;
        *info = 0;
        return;
    }

    g_sgesvd_calls.exec_calls += 1;
    g_sgesvd_calls.jobu = *jobu;
    g_sgesvd_calls.jobvt = *jobvt;
    g_sgesvd_calls.u = u;
    g_sgesvd_calls.vt = vt;
    g_sgesvd_calls.ldu = *ldu;
    g_sgesvd_calls.ldvt = *ldvt;
    *info = 0;
}

static void stub_sgesdd_fortran(char *jobz, int *m, int *n, float *a,
                                int *lda, float *s, float *u, int *ldu,
                                float *vt, int *ldvt, float *work,
                                int *lwork, int *iwork, int *info)
{
    (void)m;
    (void)n;
    (void)a;
    (void)lda;
    (void)s;

    if (*lwork == -1) {
        g_sgesdd_calls.query_calls += 1;
        g_sgesdd_calls.query_iwork = iwork;
        *work = 56.0f;
        *info = 0;
        return;
    }

    g_sgesdd_calls.exec_calls += 1;
    g_sgesdd_calls.jobz = *jobz;
    g_sgesdd_calls.u = u;
    g_sgesdd_calls.vt = vt;
    g_sgesdd_calls.ldu = *ldu;
    g_sgesdd_calls.ldvt = *ldvt;
    g_sgesdd_calls.exec_iwork = iwork;
    *info = 0;
}

static void stub_sstev_cblas(int layout, char jobz, int n, float *d, float *e,
                             float *z, int ldz, float *work, int *info)
{
    (void)layout;
    (void)jobz;
    (void)n;
    (void)d;
    (void)e;
    (void)z;
    (void)ldz;
    (void)work;
    g_sstev_cblas_calls += 1;
    if (info) {
        *info = 0;
    }
}

static int check_sgeev_policy(void)
{
    fb_backend_vtable_t vtable;
    fb_sgeev_fn thunk = NULL;
    float a[16] = { 0.0f };
    float wr[4] = { 0.0f };
    float wi[4] = { 0.0f };
    float vl[16] = { 0.0f };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sgeev_calls, 0, sizeof(g_sgeev_calls));

    vtable.ext_ops[FB_OP_SGEEV][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_sgeev_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SGEEV);

    thunk = (fb_sgeev_fn)vtable.ext_ops[FB_OP_SGEEV][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SGEEV Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    if (thunk(FB_LAYOUT_ROW_MAJOR, 'V', 'N', 4, a, 4, wr, wi, vl, 4, NULL, 1) != -1 ||
        g_sgeev_calls.query_calls != 0 || g_sgeev_calls.exec_calls != 0) {
        fprintf(stderr, "[FAIL] SGEEV accepted unsupported row-major layout\n");
        return 1;
    }

    if (thunk(FB_LAYOUT_COL_MAJOR, 'V', 'N', 4, a, 4, wr, wi, vl, 4, NULL, 1) != 0 ||
        g_sgeev_calls.query_calls != 1 || g_sgeev_calls.exec_calls != 1 ||
        g_sgeev_calls.jobvl != 'V' || g_sgeev_calls.jobvr != 'N' ||
        g_sgeev_calls.vl != vl || g_sgeev_calls.vr != NULL ||
        g_sgeev_calls.ldvl != 4 || g_sgeev_calls.ldvr != 1) {
        fprintf(stderr, "[FAIL] SGEEV did not preserve caller mode and outputs\n");
        return 1;
    }

    if (thunk(FB_LAYOUT_COL_MAJOR, 'V', 'N', 4, a, 4, wr, wi, NULL, 1, NULL, 1) != -1 ||
        g_sgeev_calls.query_calls != 1 || g_sgeev_calls.exec_calls != 1) {
        fprintf(stderr, "[FAIL] SGEEV accepted missing VL buffer for vector job\n");
        return 1;
    }

    printf("[PASS] SGEEV preserves job flags and rejects unsupported layout/buffers\n");
    return 0;
}

static int check_sgesvd_policy(void)
{
    fb_backend_vtable_t vtable;
    fb_sgesvd_fn thunk = NULL;
    float a[24] = { 0.0f };
    float s[4] = { 0.0f };
    float u[24] = { 0.0f };
    float vt[24] = { 0.0f };
    float superb[4] = { 1.0f, 1.0f, 1.0f, 1.0f };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sgesvd_calls, 0, sizeof(g_sgesvd_calls));

    vtable.ext_ops[FB_OP_SGESVD][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_sgesvd_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SGESVD);

    thunk = (fb_sgesvd_fn)vtable.ext_ops[FB_OP_SGESVD][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SGESVD Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    if (thunk(FB_LAYOUT_ROW_MAJOR, 'A', 'S', 6, 4, a, 6, s, u, 6, vt, 4, superb) != -1 ||
        g_sgesvd_calls.query_calls != 0 || g_sgesvd_calls.exec_calls != 0) {
        fprintf(stderr, "[FAIL] SGESVD accepted unsupported row-major layout\n");
        return 1;
    }

    if (thunk(FB_LAYOUT_COL_MAJOR, 'A', 'S', 6, 4, a, 6, s, u, 6, vt, 4, superb) != 0 ||
        g_sgesvd_calls.query_calls != 1 || g_sgesvd_calls.exec_calls != 1 ||
        g_sgesvd_calls.jobu != 'A' || g_sgesvd_calls.jobvt != 'S' ||
        g_sgesvd_calls.u != u || g_sgesvd_calls.vt != vt ||
        g_sgesvd_calls.ldu != 6 || g_sgesvd_calls.ldvt != 4) {
        fprintf(stderr, "[FAIL] SGESVD did not preserve caller job/output contract\n");
        return 1;
    }

    if (thunk(FB_LAYOUT_COL_MAJOR, 'A', 'S', 6, 4, a, 6, s, NULL, 1, vt, 4, superb) != -1 ||
        g_sgesvd_calls.query_calls != 1 || g_sgesvd_calls.exec_calls != 1) {
        fprintf(stderr, "[FAIL] SGESVD accepted missing U buffer for explicit output job\n");
        return 1;
    }

    printf("[PASS] SGESVD preserves SVD job modes and rejects unsupported layout/buffers\n");
    return 0;
}

static int check_sgesdd_policy(void)
{
    fb_backend_vtable_t vtable;
    fb_sgesdd_fn thunk = NULL;
    float a[24] = { 0.0f };
    float s[4] = { 0.0f };
    float u[24] = { 0.0f };
    float vt[24] = { 0.0f };

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sgesdd_calls, 0, sizeof(g_sgesdd_calls));

    vtable.ext_ops[FB_OP_SGESDD][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_sgesdd_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SGESDD);

    thunk = (fb_sgesdd_fn)vtable.ext_ops[FB_OP_SGESDD][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SGESDD Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    if (thunk(FB_LAYOUT_ROW_MAJOR, 'A', 6, 4, a, 6, s, u, 6, vt, 4) != -1 ||
        g_sgesdd_calls.query_calls != 0 || g_sgesdd_calls.exec_calls != 0) {
        fprintf(stderr, "[FAIL] SGESDD accepted unsupported row-major layout\n");
        return 1;
    }

    if (thunk(FB_LAYOUT_COL_MAJOR, 'A', 6, 4, a, 6, s, u, 6, vt, 4) != 0 ||
        g_sgesdd_calls.query_calls != 1 || g_sgesdd_calls.exec_calls != 1 ||
        g_sgesdd_calls.jobz != 'A' || g_sgesdd_calls.u != u ||
        g_sgesdd_calls.vt != vt || g_sgesdd_calls.ldu != 6 ||
        g_sgesdd_calls.ldvt != 4 || g_sgesdd_calls.query_iwork == NULL ||
        g_sgesdd_calls.exec_iwork == NULL) {
        fprintf(stderr, "[FAIL] SGESDD did not preserve jobz/output contract or iwork setup\n");
        return 1;
    }

    if (thunk(FB_LAYOUT_COL_MAJOR, 'A', 6, 4, a, 6, s, NULL, 1, vt, 4) != -1 ||
        g_sgesdd_calls.query_calls != 1 || g_sgesdd_calls.exec_calls != 1) {
        fprintf(stderr, "[FAIL] SGESDD accepted missing U buffer for non-N jobz\n");
        return 1;
    }

    printf("[PASS] SGESDD preserves jobz and allocates iwork during query/execute\n");
    return 0;
}

static int check_sstev_legacy_layout_rejection(void)
{
    fb_backend_vtable_t vtable;
    fb_sstev_legacy_c2f_thunk_fn thunk = NULL;
    float d[4] = { 0.0f };
    float e[4] = { 0.0f };
    float z[16] = { 0.0f };
    float work[8] = { 0.0f };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    g_sstev_cblas_calls = 0;

    vtable.ext_ops[FB_OP_SSTEV][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_sstev_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SSTEV);

    thunk = (fb_sstev_legacy_c2f_thunk_fn)vtable.ext_ops[FB_OP_SSTEV][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SSTEV legacy CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(FB_LAYOUT_ROW_MAJOR, 'N', 4, d, e, z, 4, work, &info);
    if (info != -1 || g_sstev_cblas_calls != 0) {
        fprintf(stderr, "[FAIL] SSTEV legacy thunk accepted unsupported row-major layout\n");
        return 1;
    }

    printf("[PASS] Legacy SSTEV CBLAS->Fortran thunk rejects unsupported row-major layout\n");
    return 0;
}

int main(void)
{
    if (check_sgeev_policy() != 0) {
        return 1;
    }
    if (check_sgesvd_policy() != 0) {
        return 1;
    }
    if (check_sgesdd_policy() != 0) {
        return 1;
    }
    if (check_sstev_legacy_layout_rejection() != 0) {
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}