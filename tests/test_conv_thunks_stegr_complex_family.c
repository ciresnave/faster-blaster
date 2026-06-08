#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef int (*fb_cstegr_fn)(fb_layout_t layout, char jobz, char range, int n,
                            float *d, float *e, float vl, float vu, int il,
                            int iu, float abstol, int *m, float *w,
                            fb_complex_float_t *z, int ldz, int *isuppz);
typedef int (*fb_zstegr_fn)(fb_layout_t layout, char jobz, char range, int n,
                            double *d, double *e, double vl, double vu,
                            int il, int iu, double abstol, int *m, double *w,
                            fb_complex_double_t *z, int ldz, int *isuppz);

typedef void (*fb_cstegr_fortran_slot_fn)(char *jobz, char *range, int *n,
                                          float *d, float *e, float *vl,
                                          float *vu, int *il, int *iu,
                                          float *abstol, int *m, float *w,
                                          fb_complex_float_t *z, int *ldz,
                                          int *isuppz, float *work,
                                          int *lwork, int *iwork,
                                          int *liwork, int *info);
typedef void (*fb_zstegr_fortran_slot_fn)(char *jobz, char *range, int *n,
                                          double *d, double *e, double *vl,
                                          double *vu, int *il, int *iu,
                                          double *abstol, int *m, double *w,
                                          fb_complex_double_t *z, int *ldz,
                                          int *isuppz, double *work,
                                          int *lwork, int *iwork,
                                          int *liwork, int *info);

static fb_complex_float_t make_cfloat(float real_value)
{
    fb_complex_float_t value;

    __real__ value = real_value;
    __imag__ value = 0.0f;
    return value;
}

static fb_complex_double_t make_cdouble(double real_value)
{
    fb_complex_double_t value;

    __real__ value = real_value;
    __imag__ value = 0.0;
    return value;
}

static float cfloat_real(fb_complex_float_t value)
{
    return __real__ value;
}

static double cdouble_real(fb_complex_double_t value)
{
    return __real__ value;
}

static int cfloat_reals_match(const fb_complex_float_t *values,
                              const float *expected, int count)
{
    int index = 0;

    for (index = 0; index < count; ++index) {
        if (cfloat_real(values[index]) != expected[index]) {
            return 0;
        }
    }
    return 1;
}

static int cdouble_reals_match(const fb_complex_double_t *values,
                               const double *expected, int count)
{
    int index = 0;

    for (index = 0; index < count; ++index) {
        if (cdouble_real(values[index]) != expected[index]) {
            return 0;
        }
    }
    return 1;
}

static struct {
    int query_calls;
    int exec_calls;
    int query_lwork;
    int exec_lwork;
    int exec_liwork;
    int exec_ldz;
} g_cstegr_fortran_call;

static struct {
    int query_calls;
    int exec_calls;
    int query_lwork;
    int exec_lwork;
    int exec_liwork;
    int exec_ldz;
} g_zstegr_fortran_call;

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
} g_cstegr_cblas_call;

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
} g_zstegr_cblas_call;

static int g_cstegr_cblas_rc = 0;
static int g_zstegr_cblas_rc = 0;

static void stub_cstegr_fortran(char *jobz, char *range, int *n, float *d,
                                float *e, float *vl, float *vu, int *il,
                                int *iu, float *abstol, int *m, float *w,
                                fb_complex_float_t *z, int *ldz, int *isuppz,
                                float *work, int *lwork, int *iwork,
                                int *liwork, int *info)
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
        g_cstegr_fortran_call.query_calls += 1;
        g_cstegr_fortran_call.query_lwork = *lwork;
        work[0] = 33.0f;
        *iwork = 17;
        *info = 0;
        return;
    }

    g_cstegr_fortran_call.exec_calls += 1;
    g_cstegr_fortran_call.exec_lwork = *lwork;
    g_cstegr_fortran_call.exec_liwork = *liwork;
    g_cstegr_fortran_call.exec_ldz = *ldz;
    *m = 2;
    w[0] = 1.5f;
    w[1] = 2.5f;
    z[0] = make_cfloat(11.0f);
    z[1] = make_cfloat(12.0f);
    z[2] = make_cfloat(13.0f);
    z[3] = make_cfloat(21.0f);
    z[4] = make_cfloat(22.0f);
    z[5] = make_cfloat(23.0f);
    isuppz[0] = 1;
    isuppz[1] = 3;
    *info = 0;
}

static void stub_zstegr_fortran(char *jobz, char *range, int *n, double *d,
                                double *e, double *vl, double *vu, int *il,
                                int *iu, double *abstol, int *m, double *w,
                                fb_complex_double_t *z, int *ldz,
                                int *isuppz, double *work, int *lwork,
                                int *iwork, int *liwork, int *info)
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
        g_zstegr_fortran_call.query_calls += 1;
        g_zstegr_fortran_call.query_lwork = *lwork;
        work[0] = 34.0;
        *iwork = 19;
        *info = 0;
        return;
    }

    g_zstegr_fortran_call.exec_calls += 1;
    g_zstegr_fortran_call.exec_lwork = *lwork;
    g_zstegr_fortran_call.exec_liwork = *liwork;
    g_zstegr_fortran_call.exec_ldz = *ldz;
    *m = 2;
    w[0] = 3.5;
    w[1] = 4.5;
    z[0] = make_cdouble(31.0);
    z[1] = make_cdouble(32.0);
    z[2] = make_cdouble(33.0);
    z[3] = make_cdouble(41.0);
    z[4] = make_cdouble(42.0);
    z[5] = make_cdouble(43.0);
    isuppz[0] = 2;
    isuppz[1] = 4;
    *info = 0;
}

static int stub_cstegr_cblas(fb_layout_t layout, char jobz, char range, int n,
                             float *d, float *e, float vl, float vu, int il,
                             int iu, float abstol, int *m, float *w,
                             fb_complex_float_t *z, int ldz, int *isuppz)
{
    (void)d;
    (void)e;
    g_cstegr_cblas_call.called += 1;
    g_cstegr_cblas_call.layout = layout;
    g_cstegr_cblas_call.jobz = jobz;
    g_cstegr_cblas_call.range = range;
    g_cstegr_cblas_call.n = n;
    g_cstegr_cblas_call.vl = vl;
    g_cstegr_cblas_call.vu = vu;
    g_cstegr_cblas_call.il = il;
    g_cstegr_cblas_call.iu = iu;
    g_cstegr_cblas_call.abstol = abstol;
    g_cstegr_cblas_call.ldz = ldz;
    *m = 2;
    w[0] = 5.5f;
    w[1] = 6.5f;
    z[0] = make_cfloat(51.0f);
    isuppz[0] = 1;
    isuppz[1] = 2;
    return g_cstegr_cblas_rc;
}

static int stub_zstegr_cblas(fb_layout_t layout, char jobz, char range, int n,
                             double *d, double *e, double vl, double vu,
                             int il, int iu, double abstol, int *m, double *w,
                             fb_complex_double_t *z, int ldz, int *isuppz)
{
    (void)d;
    (void)e;
    g_zstegr_cblas_call.called += 1;
    g_zstegr_cblas_call.layout = layout;
    g_zstegr_cblas_call.jobz = jobz;
    g_zstegr_cblas_call.range = range;
    g_zstegr_cblas_call.n = n;
    g_zstegr_cblas_call.vl = vl;
    g_zstegr_cblas_call.vu = vu;
    g_zstegr_cblas_call.il = il;
    g_zstegr_cblas_call.iu = iu;
    g_zstegr_cblas_call.abstol = abstol;
    g_zstegr_cblas_call.ldz = ldz;
    *m = 2;
    w[0] = 7.5;
    w[1] = 8.5;
    z[0] = make_cdouble(61.0);
    isuppz[0] = 2;
    isuppz[1] = 3;
    return g_zstegr_cblas_rc;
}

static int check_cstegr_fortran_to_cblas(void)
{
    static const float expected_z[9] = {
        11.0f, 21.0f, 0.0f,
        12.0f, 22.0f, 0.0f,
        13.0f, 23.0f, 0.0f
    };
    fb_backend_vtable_t vtable;
    fb_cstegr_fn thunk = NULL;
    float d[3] = { 0.0f, 0.0f, 0.0f };
    float e[2] = { 0.0f, 0.0f };
    float w[3] = { 0.0f, 0.0f, 0.0f };
    fb_complex_float_t z[9] = { make_cfloat(0.0f) };
    int isuppz[6] = { 0, 0, 0, 0, 0, 0 };
    int m = 0;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cstegr_fortran_call, 0, sizeof(g_cstegr_fortran_call));

    vtable.ext_ops[FB_OP_CSTEGR][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_cstegr_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CSTEGR);

    thunk = (fb_cstegr_fn)vtable.ext_ops[FB_OP_CSTEGR][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CSTEGR Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 'V', 'I', 3, d, e, 1.0f, 6.0f, 1, 2,
                 0.25f, &m, w, z, 3, isuppz);
    if (info != 0 || g_cstegr_fortran_call.query_calls != 1 ||
        g_cstegr_fortran_call.exec_calls != 1 ||
        g_cstegr_fortran_call.query_lwork != -1 ||
        g_cstegr_fortran_call.exec_lwork != 33 ||
        g_cstegr_fortran_call.exec_liwork != 17 ||
        g_cstegr_fortran_call.exec_ldz != 3 || m != 2 ||
        w[0] != 1.5f || w[1] != 2.5f ||
        !cfloat_reals_match(z, expected_z, 9) ||
        isuppz[0] != 1 || isuppz[1] != 3) {
        fprintf(stderr, "[FAIL] CSTEGR Fortran->CBLAS thunk did not preserve query semantics or row-major complex vector export\n");
        return 1;
    }

    printf("[PASS] CSTEGR Fortran->CBLAS thunk performs the workspace queries and exports row-major complex eigenvectors\n");
    return 0;
}

static int check_cstegr_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_cstegr_fortran_slot_fn thunk = NULL;
    float d[3] = { 0.0f, 0.0f, 0.0f };
    float e[2] = { 0.0f, 0.0f };
    float w[3] = { 0.0f, 0.0f, 0.0f };
    fb_complex_float_t z[9] = { make_cfloat(0.0f) };
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
    memset(&g_cstegr_cblas_call, 0, sizeof(g_cstegr_cblas_call));
    g_cstegr_cblas_rc = 297;

    vtable.ext_ops[FB_OP_CSTEGR][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_cstegr_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CSTEGR);

    thunk = (fb_cstegr_fortran_slot_fn)vtable.ext_ops[FB_OP_CSTEGR][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CSTEGR CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&jobz, &range, &n, d, e, &vl, &vu, &il, &iu, &abstol, &m, w, z,
          &ldz, isuppz, work, &lwork, iwork, &liwork, &info);
    if (info != 297 || g_cstegr_cblas_call.called != 1 ||
        g_cstegr_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_cstegr_cblas_call.jobz != 'V' || g_cstegr_cblas_call.range != 'I' ||
        g_cstegr_cblas_call.n != 3 || g_cstegr_cblas_call.vl != 1.0f ||
        g_cstegr_cblas_call.vu != 6.0f || g_cstegr_cblas_call.il != 1 ||
        g_cstegr_cblas_call.iu != 2 ||
        g_cstegr_cblas_call.abstol != 0.25f ||
        g_cstegr_cblas_call.ldz != 3 || m != 2 || w[0] != 5.5f ||
        w[1] != 6.5f || cfloat_real(z[0]) != 51.0f || isuppz[0] != 1 ||
        isuppz[1] != 2) {
        fprintf(stderr, "[FAIL] CSTEGR CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] CSTEGR CBLAS->Fortran thunk maps tridiagonal complex MRRR arguments into the C entry\n");
    return 0;
}

static int check_zstegr_fortran_to_cblas(void)
{
    static const double expected_z[9] = {
        31.0, 41.0, 0.0,
        32.0, 42.0, 0.0,
        33.0, 43.0, 0.0
    };
    fb_backend_vtable_t vtable;
    fb_zstegr_fn thunk = NULL;
    double d[3] = { 0.0, 0.0, 0.0 };
    double e[2] = { 0.0, 0.0 };
    double w[3] = { 0.0, 0.0, 0.0 };
    fb_complex_double_t z[9] = { make_cdouble(0.0) };
    int isuppz[6] = { 0, 0, 0, 0, 0, 0 };
    int m = 0;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zstegr_fortran_call, 0, sizeof(g_zstegr_fortran_call));

    vtable.ext_ops[FB_OP_ZSTEGR][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_zstegr_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_ZSTEGR);

    thunk = (fb_zstegr_fn)vtable.ext_ops[FB_OP_ZSTEGR][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZSTEGR Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 'V', 'I', 3, d, e, 2.0, 7.0, 1, 2,
                 0.5, &m, w, z, 3, isuppz);
    if (info != 0 || g_zstegr_fortran_call.query_calls != 1 ||
        g_zstegr_fortran_call.exec_calls != 1 ||
        g_zstegr_fortran_call.query_lwork != -1 ||
        g_zstegr_fortran_call.exec_lwork != 34 ||
        g_zstegr_fortran_call.exec_liwork != 19 ||
        g_zstegr_fortran_call.exec_ldz != 3 || m != 2 ||
        w[0] != 3.5 || w[1] != 4.5 ||
        !cdouble_reals_match(z, expected_z, 9) ||
        isuppz[0] != 2 || isuppz[1] != 4) {
        fprintf(stderr, "[FAIL] ZSTEGR Fortran->CBLAS thunk did not preserve double-precision complex query semantics or row-major export\n");
        return 1;
    }

    printf("[PASS] ZSTEGR Fortran->CBLAS thunk performs the workspace queries and exports double-precision row-major complex eigenvectors\n");
    return 0;
}

static int check_zstegr_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_zstegr_fortran_slot_fn thunk = NULL;
    double d[3] = { 0.0, 0.0, 0.0 };
    double e[2] = { 0.0, 0.0 };
    double w[3] = { 0.0, 0.0, 0.0 };
    fb_complex_double_t z[9] = { make_cdouble(0.0) };
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
    memset(&g_zstegr_cblas_call, 0, sizeof(g_zstegr_cblas_call));
    g_zstegr_cblas_rc = 299;

    vtable.ext_ops[FB_OP_ZSTEGR][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_zstegr_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_ZSTEGR);

    thunk = (fb_zstegr_fortran_slot_fn)vtable.ext_ops[FB_OP_ZSTEGR][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZSTEGR CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&jobz, &range, &n, d, e, &vl, &vu, &il, &iu, &abstol, &m, w, z,
          &ldz, isuppz, work, &lwork, iwork, &liwork, &info);
    if (info != 299 || g_zstegr_cblas_call.called != 1 ||
        g_zstegr_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_zstegr_cblas_call.jobz != 'V' || g_zstegr_cblas_call.range != 'I' ||
        g_zstegr_cblas_call.n != 3 || g_zstegr_cblas_call.vl != 2.0 ||
        g_zstegr_cblas_call.vu != 7.0 || g_zstegr_cblas_call.il != 1 ||
        g_zstegr_cblas_call.iu != 2 || g_zstegr_cblas_call.abstol != 0.5 ||
        g_zstegr_cblas_call.ldz != 3 || m != 2 || w[0] != 7.5 ||
        w[1] != 8.5 || cdouble_real(z[0]) != 61.0 || isuppz[0] != 2 ||
        isuppz[1] != 3) {
        fprintf(stderr, "[FAIL] ZSTEGR CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] ZSTEGR CBLAS->Fortran thunk maps double-precision complex MRRR arguments into the C entry\n");
    return 0;
}

int main(void)
{
    if (check_cstegr_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_cstegr_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_zstegr_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_zstegr_cblas_to_fortran() != 0) {
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}