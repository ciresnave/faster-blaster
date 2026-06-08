#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef int (*fb_sstebz_fn)(char range, char order, int n, float vl, float vu,
                            int il, int iu, float abstol, const float *d,
                            const float *e, int *m, int *nsplit, float *w,
                            int *iblock, int *isplit);
typedef int (*fb_dstebz_fn)(char range, char order, int n, double vl, double vu,
                            int il, int iu, double abstol, const double *d,
                            const double *e, int *m, int *nsplit, double *w,
                            int *iblock, int *isplit);

typedef void (*fb_sstebz_fortran_slot_fn)(char *range, char *order, int *n,
                                          float *vl, float *vu, int *il,
                                          int *iu, float *abstol, float *d,
                                          float *e, int *m, int *nsplit,
                                          float *w, int *iblock, int *isplit,
                                          float *work, int *iwork, int *info);
typedef void (*fb_dstebz_fortran_slot_fn)(char *range, char *order, int *n,
                                          double *vl, double *vu, int *il,
                                          int *iu, double *abstol, double *d,
                                          double *e, int *m, int *nsplit,
                                          double *w, int *iblock, int *isplit,
                                          double *work, int *iwork, int *info);

static struct {
    int exec_calls;
    int saw_work;
    int saw_iwork;
} g_sstebz_fortran_call;

static struct {
    int exec_calls;
    int saw_work;
    int saw_iwork;
} g_dstebz_fortran_call;

static struct {
    int called;
    char range;
    char order;
    int n;
    float vl;
    float vu;
    int il;
    int iu;
    float abstol;
} g_sstebz_cblas_call;

static struct {
    int called;
    char range;
    char order;
    int n;
    double vl;
    double vu;
    int il;
    int iu;
    double abstol;
} g_dstebz_cblas_call;

static int g_sstebz_cblas_rc = 0;
static int g_dstebz_cblas_rc = 0;

static void stub_sstebz_fortran(char *range, char *order, int *n, float *vl,
                                float *vu, int *il, int *iu, float *abstol,
                                float *d, float *e, int *m, int *nsplit,
                                float *w, int *iblock, int *isplit, float *work,
                                int *iwork, int *info)
{
    (void)range;
    (void)order;
    (void)n;
    (void)vl;
    (void)vu;
    (void)il;
    (void)iu;
    (void)abstol;
    (void)d;
    (void)e;
    g_sstebz_fortran_call.exec_calls += 1;
    g_sstebz_fortran_call.saw_work = (work != NULL);
    g_sstebz_fortran_call.saw_iwork = (iwork != NULL);
    *m = 2;
    *nsplit = 1;
    w[0] = 17.5f;
    w[1] = 18.5f;
    iblock[0] = 1;
    isplit[0] = 3;
    *info = 0;
}

static void stub_dstebz_fortran(char *range, char *order, int *n, double *vl,
                                double *vu, int *il, int *iu, double *abstol,
                                double *d, double *e, int *m, int *nsplit,
                                double *w, int *iblock, int *isplit,
                                double *work, int *iwork, int *info)
{
    (void)range;
    (void)order;
    (void)n;
    (void)vl;
    (void)vu;
    (void)il;
    (void)iu;
    (void)abstol;
    (void)d;
    (void)e;
    g_dstebz_fortran_call.exec_calls += 1;
    g_dstebz_fortran_call.saw_work = (work != NULL);
    g_dstebz_fortran_call.saw_iwork = (iwork != NULL);
    *m = 2;
    *nsplit = 1;
    w[0] = 19.5;
    w[1] = 20.5;
    iblock[0] = 2;
    isplit[0] = 4;
    *info = 0;
}

static int stub_sstebz_cblas(char range, char order, int n, float vl, float vu,
                             int il, int iu, float abstol, const float *d,
                             const float *e, int *m, int *nsplit, float *w,
                             int *iblock, int *isplit)
{
    (void)d;
    (void)e;
    g_sstebz_cblas_call.called += 1;
    g_sstebz_cblas_call.range = range;
    g_sstebz_cblas_call.order = order;
    g_sstebz_cblas_call.n = n;
    g_sstebz_cblas_call.vl = vl;
    g_sstebz_cblas_call.vu = vu;
    g_sstebz_cblas_call.il = il;
    g_sstebz_cblas_call.iu = iu;
    g_sstebz_cblas_call.abstol = abstol;
    *m = 2;
    *nsplit = 1;
    w[0] = 21.5f;
    w[1] = 22.5f;
    iblock[0] = 3;
    isplit[0] = 5;
    return g_sstebz_cblas_rc;
}

static int stub_dstebz_cblas(char range, char order, int n, double vl,
                             double vu, int il, int iu, double abstol,
                             const double *d, const double *e, int *m,
                             int *nsplit, double *w, int *iblock, int *isplit)
{
    (void)d;
    (void)e;
    g_dstebz_cblas_call.called += 1;
    g_dstebz_cblas_call.range = range;
    g_dstebz_cblas_call.order = order;
    g_dstebz_cblas_call.n = n;
    g_dstebz_cblas_call.vl = vl;
    g_dstebz_cblas_call.vu = vu;
    g_dstebz_cblas_call.il = il;
    g_dstebz_cblas_call.iu = iu;
    g_dstebz_cblas_call.abstol = abstol;
    *m = 2;
    *nsplit = 1;
    w[0] = 23.5;
    w[1] = 24.5;
    iblock[0] = 4;
    isplit[0] = 6;
    return g_dstebz_cblas_rc;
}

static int check_sstebz_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_sstebz_fn thunk = NULL;
    float d[3] = { 0.0f, 0.0f, 0.0f };
    float e[2] = { 0.0f, 0.0f };
    float w[3] = { 0.0f, 0.0f, 0.0f };
    int iblock[3] = { 0, 0, 0 };
    int isplit[3] = { 0, 0, 0 };
    int m = 0;
    int nsplit = 0;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sstebz_fortran_call, 0, sizeof(g_sstebz_fortran_call));

    vtable.ext_ops[FB_OP_SSTEBZ][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_sstebz_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SSTEBZ);

    thunk = (fb_sstebz_fn)vtable.ext_ops[FB_OP_SSTEBZ][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SSTEBZ Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk('I', 'B', 3, 1.0f, 6.0f, 1, 2, 0.25f, d, e, &m, &nsplit, w,
                 iblock, isplit);
    if (info != 0 || g_sstebz_fortran_call.exec_calls != 1 ||
        !g_sstebz_fortran_call.saw_work || !g_sstebz_fortran_call.saw_iwork ||
        m != 2 || nsplit != 1 || w[0] != 17.5f || w[1] != 18.5f ||
        iblock[0] != 1 || isplit[0] != 3) {
        fprintf(stderr, "[FAIL] SSTEBZ Fortran->CBLAS thunk did not allocate fixed workspaces and forward tridiagonal interval outputs\n");
        return 1;
    }

    printf("[PASS] SSTEBZ Fortran->CBLAS thunk allocates fixed workspaces and forwards tridiagonal interval outputs\n");
    return 0;
}

static int check_sstebz_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_sstebz_fortran_slot_fn thunk = NULL;
    float d[3] = { 0.0f, 0.0f, 0.0f };
    float e[2] = { 0.0f, 0.0f };
    float w[3] = { 0.0f, 0.0f, 0.0f };
    float work[16] = { 0.0f };
    int iwork[16] = { 0 };
    int iblock[3] = { 0, 0, 0 };
    int isplit[3] = { 0, 0, 0 };
    char range = 'I';
    char order = 'B';
    int n = 3;
    float vl = 1.0f;
    float vu = 6.0f;
    int il = 1;
    int iu = 2;
    float abstol = 0.25f;
    int m = 0;
    int nsplit = 0;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sstebz_cblas_call, 0, sizeof(g_sstebz_cblas_call));
    g_sstebz_cblas_rc = 261;

    vtable.ext_ops[FB_OP_SSTEBZ][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_sstebz_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SSTEBZ);

    thunk = (fb_sstebz_fortran_slot_fn)vtable.ext_ops[FB_OP_SSTEBZ][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SSTEBZ CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&range, &order, &n, &vl, &vu, &il, &iu, &abstol, d, e, &m, &nsplit,
          w, iblock, isplit, work, iwork, &info);
    if (info != 261 || g_sstebz_cblas_call.called != 1 ||
        g_sstebz_cblas_call.range != 'I' || g_sstebz_cblas_call.order != 'B' ||
        g_sstebz_cblas_call.n != 3 || g_sstebz_cblas_call.vl != 1.0f ||
        g_sstebz_cblas_call.vu != 6.0f || g_sstebz_cblas_call.il != 1 ||
        g_sstebz_cblas_call.iu != 2 || g_sstebz_cblas_call.abstol != 0.25f ||
        m != 2 || nsplit != 1 || w[0] != 21.5f || w[1] != 22.5f ||
        iblock[0] != 3 || isplit[0] != 5) {
        fprintf(stderr, "[FAIL] SSTEBZ CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] SSTEBZ CBLAS->Fortran thunk maps tridiagonal interval arguments into the C entry\n");
    return 0;
}

static int check_dstebz_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_dstebz_fn thunk = NULL;
    double d[3] = { 0.0, 0.0, 0.0 };
    double e[2] = { 0.0, 0.0 };
    double w[3] = { 0.0, 0.0, 0.0 };
    int iblock[3] = { 0, 0, 0 };
    int isplit[3] = { 0, 0, 0 };
    int m = 0;
    int nsplit = 0;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dstebz_fortran_call, 0, sizeof(g_dstebz_fortran_call));

    vtable.ext_ops[FB_OP_DSTEBZ][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_dstebz_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_DSTEBZ);

    thunk = (fb_dstebz_fn)vtable.ext_ops[FB_OP_DSTEBZ][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DSTEBZ Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk('V', 'E', 3, 2.0, 7.0, 1, 2, 0.5, d, e, &m, &nsplit, w,
                 iblock, isplit);
    if (info != 0 || g_dstebz_fortran_call.exec_calls != 1 ||
        !g_dstebz_fortran_call.saw_work || !g_dstebz_fortran_call.saw_iwork ||
        m != 2 || nsplit != 1 || w[0] != 19.5 || w[1] != 20.5 ||
        iblock[0] != 2 || isplit[0] != 4) {
        fprintf(stderr, "[FAIL] DSTEBZ Fortran->CBLAS thunk did not allocate fixed workspaces and forward double-precision tridiagonal interval outputs\n");
        return 1;
    }

    printf("[PASS] DSTEBZ Fortran->CBLAS thunk allocates fixed workspaces and forwards double-precision tridiagonal interval outputs\n");
    return 0;
}

static int check_dstebz_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_dstebz_fortran_slot_fn thunk = NULL;
    double d[3] = { 0.0, 0.0, 0.0 };
    double e[2] = { 0.0, 0.0 };
    double w[3] = { 0.0, 0.0, 0.0 };
    double work[16] = { 0.0 };
    int iwork[16] = { 0 };
    int iblock[3] = { 0, 0, 0 };
    int isplit[3] = { 0, 0, 0 };
    char range = 'V';
    char order = 'E';
    int n = 3;
    double vl = 2.0;
    double vu = 7.0;
    int il = 1;
    int iu = 2;
    double abstol = 0.5;
    int m = 0;
    int nsplit = 0;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dstebz_cblas_call, 0, sizeof(g_dstebz_cblas_call));
    g_dstebz_cblas_rc = 263;

    vtable.ext_ops[FB_OP_DSTEBZ][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_dstebz_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_DSTEBZ);

    thunk = (fb_dstebz_fortran_slot_fn)vtable.ext_ops[FB_OP_DSTEBZ][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DSTEBZ CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&range, &order, &n, &vl, &vu, &il, &iu, &abstol, d, e, &m, &nsplit,
          w, iblock, isplit, work, iwork, &info);
    if (info != 263 || g_dstebz_cblas_call.called != 1 ||
        g_dstebz_cblas_call.range != 'V' || g_dstebz_cblas_call.order != 'E' ||
        g_dstebz_cblas_call.n != 3 || g_dstebz_cblas_call.vl != 2.0 ||
        g_dstebz_cblas_call.vu != 7.0 || g_dstebz_cblas_call.il != 1 ||
        g_dstebz_cblas_call.iu != 2 || g_dstebz_cblas_call.abstol != 0.5 ||
        m != 2 || nsplit != 1 || w[0] != 23.5 || w[1] != 24.5 ||
        iblock[0] != 4 || isplit[0] != 6) {
        fprintf(stderr, "[FAIL] DSTEBZ CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] DSTEBZ CBLAS->Fortran thunk maps double-precision tridiagonal interval arguments into the C entry\n");
    return 0;
}

int main(void)
{
    if (check_sstebz_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_sstebz_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_dstebz_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_dstebz_cblas_to_fortran() != 0) {
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}