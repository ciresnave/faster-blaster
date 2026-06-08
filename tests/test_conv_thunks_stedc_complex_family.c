#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef int (*fb_cstedc_fn)(fb_layout_t layout, char compz, int n, float *d,
                            float *e, fb_complex_float_t *z, int ldz);
typedef int (*fb_zstedc_fn)(fb_layout_t layout, char compz, int n, double *d,
                            double *e, fb_complex_double_t *z, int ldz);

typedef void (*fb_cstedc_fortran_slot_fn)(char *compz, int *n, float *d,
                                          float *e, fb_complex_float_t *z,
                                          int *ldz, fb_complex_float_t *work,
                                          int *lwork, float *rwork,
                                          int *lrwork, int *iwork,
                                          int *liwork, int *info);
typedef void (*fb_zstedc_fortran_slot_fn)(char *compz, int *n, double *d,
                                          double *e, fb_complex_double_t *z,
                                          int *ldz, fb_complex_double_t *work,
                                          int *lwork, double *rwork,
                                          int *lrwork, int *iwork,
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
    int query_lrwork;
    int exec_lwork;
    int exec_lrwork;
    int exec_liwork;
    int exec_ldz;
    fb_complex_float_t z_in[6];
} g_cstedc_fortran_call;

static struct {
    int query_calls;
    int exec_calls;
    int query_lwork;
    int query_lrwork;
    int exec_lwork;
    int exec_lrwork;
    int exec_liwork;
    int exec_ldz;
} g_zstedc_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    char compz;
    int n;
    int ldz;
} g_cstedc_cblas_call;

static struct {
    int called;
    fb_layout_t layout;
    char compz;
    int n;
    int ldz;
} g_zstedc_cblas_call;

static int g_cstedc_cblas_rc = 0;
static int g_zstedc_cblas_rc = 0;

static void stub_cstedc_fortran(char *compz, int *n, float *d, float *e,
                                fb_complex_float_t *z, int *ldz,
                                fb_complex_float_t *work, int *lwork,
                                float *rwork, int *lrwork, int *iwork,
                                int *liwork, int *info)
{
    (void)n;
    if (*lwork == -1 || *lrwork == -1 || *liwork == -1) {
        g_cstedc_fortran_call.query_calls += 1;
        g_cstedc_fortran_call.query_lwork = *lwork;
        g_cstedc_fortran_call.query_lrwork = *lrwork;
        work[0] = make_cfloat(35.0f);
        rwork[0] = 45.0f;
        *iwork = 21;
        *info = 0;
        return;
    }

    g_cstedc_fortran_call.exec_calls += 1;
    g_cstedc_fortran_call.exec_lwork = *lwork;
    g_cstedc_fortran_call.exec_lrwork = *lrwork;
    g_cstedc_fortran_call.exec_liwork = *liwork;
    g_cstedc_fortran_call.exec_ldz = *ldz;
    memcpy(g_cstedc_fortran_call.z_in, z, sizeof(g_cstedc_fortran_call.z_in));
    d[0] = 1.5f;
    e[0] = 2.5f;
    z[0] = make_cfloat(101.0f);
    z[1] = make_cfloat(102.0f);
    z[2] = make_cfloat(103.0f);
    z[3] = make_cfloat(201.0f);
    z[4] = make_cfloat(202.0f);
    z[5] = make_cfloat(203.0f);
    z[6] = make_cfloat(301.0f);
    z[7] = make_cfloat(302.0f);
    z[8] = make_cfloat(303.0f);
    *info = 0;
    (void)compz;
}

static void stub_zstedc_fortran(char *compz, int *n, double *d, double *e,
                                fb_complex_double_t *z, int *ldz,
                                fb_complex_double_t *work, int *lwork,
                                double *rwork, int *lrwork, int *iwork,
                                int *liwork, int *info)
{
    (void)n;
    if (*lwork == -1 || *lrwork == -1 || *liwork == -1) {
        g_zstedc_fortran_call.query_calls += 1;
        g_zstedc_fortran_call.query_lwork = *lwork;
        g_zstedc_fortran_call.query_lrwork = *lrwork;
        work[0] = make_cdouble(36.0);
        rwork[0] = 46.0;
        *iwork = 23;
        *info = 0;
        return;
    }

    g_zstedc_fortran_call.exec_calls += 1;
    g_zstedc_fortran_call.exec_lwork = *lwork;
    g_zstedc_fortran_call.exec_lrwork = *lrwork;
    g_zstedc_fortran_call.exec_liwork = *liwork;
    g_zstedc_fortran_call.exec_ldz = *ldz;
    d[0] = 3.5;
    e[0] = 4.5;
    z[0] = make_cdouble(401.0);
    z[1] = make_cdouble(402.0);
    z[2] = make_cdouble(403.0);
    z[3] = make_cdouble(501.0);
    z[4] = make_cdouble(502.0);
    z[5] = make_cdouble(503.0);
    z[6] = make_cdouble(601.0);
    z[7] = make_cdouble(602.0);
    z[8] = make_cdouble(603.0);
    *info = 0;
    (void)compz;
}

static int stub_cstedc_cblas(fb_layout_t layout, char compz, int n, float *d,
                             float *e, fb_complex_float_t *z, int ldz)
{
    g_cstedc_cblas_call.called += 1;
    g_cstedc_cblas_call.layout = layout;
    g_cstedc_cblas_call.compz = compz;
    g_cstedc_cblas_call.n = n;
    g_cstedc_cblas_call.ldz = ldz;
    d[0] = 5.5f;
    e[0] = 6.5f;
    z[0] = make_cfloat(701.0f);
    return g_cstedc_cblas_rc;
}

static int stub_zstedc_cblas(fb_layout_t layout, char compz, int n, double *d,
                             double *e, fb_complex_double_t *z, int ldz)
{
    g_zstedc_cblas_call.called += 1;
    g_zstedc_cblas_call.layout = layout;
    g_zstedc_cblas_call.compz = compz;
    g_zstedc_cblas_call.n = n;
    g_zstedc_cblas_call.ldz = ldz;
    d[0] = 7.5;
    e[0] = 8.5;
    z[0] = make_cdouble(801.0);
    return g_zstedc_cblas_rc;
}

static int check_cstedc_fortran_to_cblas(void)
{
    static const float expected_input[6] = {
        11.0f, 21.0f, 31.0f,
        12.0f, 22.0f, 32.0f
    };
    static const float expected_z[9] = {
        101.0f, 201.0f, 301.0f,
        102.0f, 202.0f, 302.0f,
        103.0f, 203.0f, 303.0f
    };
    fb_backend_vtable_t vtable;
    fb_cstedc_fn thunk = NULL;
    float d[3] = { 0.0f, 0.0f, 0.0f };
    float e[2] = { 0.0f, 0.0f };
    fb_complex_float_t z[9] = {
        make_cfloat(11.0f), make_cfloat(12.0f), make_cfloat(13.0f),
        make_cfloat(21.0f), make_cfloat(22.0f), make_cfloat(23.0f),
        make_cfloat(31.0f), make_cfloat(32.0f), make_cfloat(33.0f)
    };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cstedc_fortran_call, 0, sizeof(g_cstedc_fortran_call));

    vtable.ext_ops[FB_OP_CSTEDC][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_cstedc_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CSTEDC);

    thunk = (fb_cstedc_fn)vtable.ext_ops[FB_OP_CSTEDC][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CSTEDC Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 'V', 3, d, e, z, 3);
    if (info != 0 || g_cstedc_fortran_call.query_calls != 1 ||
        g_cstedc_fortran_call.exec_calls != 1 ||
        g_cstedc_fortran_call.query_lwork != -1 ||
        g_cstedc_fortran_call.query_lrwork != -1 ||
        g_cstedc_fortran_call.exec_lwork != 35 ||
        g_cstedc_fortran_call.exec_lrwork != 45 ||
        g_cstedc_fortran_call.exec_liwork != 21 ||
        g_cstedc_fortran_call.exec_ldz != 3 ||
        !cfloat_reals_match(g_cstedc_fortran_call.z_in, expected_input, 6) ||
        d[0] != 1.5f || e[0] != 2.5f ||
        !cfloat_reals_match(z, expected_z, 9)) {
        fprintf(stderr, "[FAIL] CSTEDC Fortran->CBLAS thunk did not preserve query sizes or row-major complex vectors\n");
        return 1;
    }

    printf("[PASS] CSTEDC Fortran->CBLAS thunk performs the workspace queries and round-trips row-major complex eigenvectors\n");
    return 0;
}

static int check_cstedc_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_cstedc_fortran_slot_fn thunk = NULL;
    float d[3] = { 0.0f, 0.0f, 0.0f };
    float e[2] = { 0.0f, 0.0f };
    fb_complex_float_t z[9] = { make_cfloat(0.0f) };
    fb_complex_float_t work[8] = { make_cfloat(0.0f) };
    float rwork[8] = { 0.0f };
    int iwork[8] = { 0 };
    char compz = 'I';
    int n = 3;
    int ldz = 3;
    int lwork = 8;
    int lrwork = 8;
    int liwork = 8;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cstedc_cblas_call, 0, sizeof(g_cstedc_cblas_call));
    g_cstedc_cblas_rc = 293;

    vtable.ext_ops[FB_OP_CSTEDC][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_cstedc_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CSTEDC);

    thunk = (fb_cstedc_fortran_slot_fn)vtable.ext_ops[FB_OP_CSTEDC][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CSTEDC CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&compz, &n, d, e, z, &ldz, work, &lwork, rwork, &lrwork, iwork,
          &liwork, &info);
    if (info != 293 || g_cstedc_cblas_call.called != 1 ||
        g_cstedc_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_cstedc_cblas_call.compz != 'I' || g_cstedc_cblas_call.n != 3 ||
        g_cstedc_cblas_call.ldz != 3 || d[0] != 5.5f || e[0] != 6.5f ||
        cfloat_real(z[0]) != 701.0f) {
        fprintf(stderr, "[FAIL] CSTEDC CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] CSTEDC CBLAS->Fortran thunk maps complex divide-and-conquer arguments into the C entry\n");
    return 0;
}

static int check_zstedc_fortran_to_cblas(void)
{
    static const double expected_z[9] = {
        401.0, 501.0, 601.0,
        402.0, 502.0, 602.0,
        403.0, 503.0, 603.0
    };
    fb_backend_vtable_t vtable;
    fb_zstedc_fn thunk = NULL;
    double d[3] = { 0.0, 0.0, 0.0 };
    double e[2] = { 0.0, 0.0 };
    fb_complex_double_t z[9] = { make_cdouble(0.0) };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zstedc_fortran_call, 0, sizeof(g_zstedc_fortran_call));

    vtable.ext_ops[FB_OP_ZSTEDC][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_zstedc_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_ZSTEDC);

    thunk = (fb_zstedc_fn)vtable.ext_ops[FB_OP_ZSTEDC][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZSTEDC Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 'I', 3, d, e, z, 3);
    if (info != 0 || g_zstedc_fortran_call.query_calls != 1 ||
        g_zstedc_fortran_call.exec_calls != 1 ||
        g_zstedc_fortran_call.query_lwork != -1 ||
        g_zstedc_fortran_call.query_lrwork != -1 ||
        g_zstedc_fortran_call.exec_lwork != 36 ||
        g_zstedc_fortran_call.exec_lrwork != 46 ||
        g_zstedc_fortran_call.exec_liwork != 23 ||
        g_zstedc_fortran_call.exec_ldz != 3 ||
        d[0] != 3.5 || e[0] != 4.5 ||
        !cdouble_reals_match(z, expected_z, 9)) {
        fprintf(stderr, "[FAIL] ZSTEDC Fortran->CBLAS thunk did not export generated row-major complex eigenvectors correctly\n");
        return 1;
    }

    printf("[PASS] ZSTEDC Fortran->CBLAS thunk performs the workspace queries and exports generated row-major complex eigenvectors\n");
    return 0;
}

static int check_zstedc_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_zstedc_fortran_slot_fn thunk = NULL;
    double d[3] = { 0.0, 0.0, 0.0 };
    double e[2] = { 0.0, 0.0 };
    fb_complex_double_t z[9] = { make_cdouble(0.0) };
    fb_complex_double_t work[8] = { make_cdouble(0.0) };
    double rwork[8] = { 0.0 };
    int iwork[8] = { 0 };
    char compz = 'V';
    int n = 3;
    int ldz = 3;
    int lwork = 8;
    int lrwork = 8;
    int liwork = 8;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zstedc_cblas_call, 0, sizeof(g_zstedc_cblas_call));
    g_zstedc_cblas_rc = 295;

    vtable.ext_ops[FB_OP_ZSTEDC][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_zstedc_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_ZSTEDC);

    thunk = (fb_zstedc_fortran_slot_fn)vtable.ext_ops[FB_OP_ZSTEDC][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZSTEDC CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&compz, &n, d, e, z, &ldz, work, &lwork, rwork, &lrwork, iwork,
          &liwork, &info);
    if (info != 295 || g_zstedc_cblas_call.called != 1 ||
        g_zstedc_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_zstedc_cblas_call.compz != 'V' || g_zstedc_cblas_call.n != 3 ||
        g_zstedc_cblas_call.ldz != 3 || d[0] != 7.5 || e[0] != 8.5 ||
        cdouble_real(z[0]) != 801.0) {
        fprintf(stderr, "[FAIL] ZSTEDC CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] ZSTEDC CBLAS->Fortran thunk maps double-precision complex divide-and-conquer arguments into the C entry\n");
    return 0;
}

int main(void)
{
    if (check_cstedc_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_cstedc_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_zstedc_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_zstedc_cblas_to_fortran() != 0) {
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}