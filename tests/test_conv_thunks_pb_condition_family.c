#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef int (*fb_spbcon_fn)(fb_layout_t layout, fb_uplo_t uplo, int n, int kd,
                            const float *ab, int ldab, float anorm,
                            float *rcond);
typedef int (*fb_dpbcon_fn)(fb_layout_t layout, fb_uplo_t uplo, int n, int kd,
                            const double *ab, int ldab, double anorm,
                            double *rcond);
typedef int (*fb_cpbcon_fn)(fb_layout_t layout, fb_uplo_t uplo, int n, int kd,
                            const fb_complex_float_t *ab, int ldab,
                            float anorm, float *rcond);
typedef int (*fb_zpbcon_fn)(fb_layout_t layout, fb_uplo_t uplo, int n, int kd,
                            const fb_complex_double_t *ab, int ldab,
                            double anorm, double *rcond);

typedef void (*fb_spbcon_fortran_slot_fn)(char *uplo, int *n, int *kd,
                                          float *ab, int *ldab, float *anorm,
                                          float *rcond, float *work,
                                          int *iwork, int *info);
typedef void (*fb_dpbcon_fortran_slot_fn)(char *uplo, int *n, int *kd,
                                          double *ab, int *ldab,
                                          double *anorm, double *rcond,
                                          double *work, int *iwork,
                                          int *info);
typedef void (*fb_cpbcon_fortran_slot_fn)(char *uplo, int *n, int *kd,
                                          fb_complex_float_t *ab, int *ldab,
                                          float *anorm, float *rcond,
                                          fb_complex_float_t *work,
                                          float *rwork, int *info);
typedef void (*fb_zpbcon_fortran_slot_fn)(char *uplo, int *n, int *kd,
                                          fb_complex_double_t *ab, int *ldab,
                                          double *anorm, double *rcond,
                                          fb_complex_double_t *work,
                                          double *rwork, int *info);

static struct {
    int calls;
    char uplo;
    int n;
    int kd;
    int ldab;
    float anorm;
    int work_seen;
    int aux_seen;
    float ab_snapshot[8];
} g_spbcon_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    fb_uplo_t uplo;
    int n;
    int kd;
    int ldab;
    float anorm;
    const float *ab;
    float *rcond;
} g_spbcon_cblas_call;

static struct {
    int calls;
    char uplo;
    int n;
    int kd;
    int ldab;
    double anorm;
    int work_seen;
    int aux_seen;
    double ab_snapshot[8];
} g_dpbcon_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    fb_uplo_t uplo;
    int n;
    int kd;
    int ldab;
    double anorm;
    const double *ab;
    double *rcond;
} g_dpbcon_cblas_call;

static struct {
    int calls;
    char uplo;
    int n;
    int kd;
    int ldab;
    float anorm;
    int work_seen;
    int aux_seen;
    float ab_real_snapshot[8];
} g_cpbcon_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    fb_uplo_t uplo;
    int n;
    int kd;
    int ldab;
    float anorm;
    const fb_complex_float_t *ab;
    float *rcond;
} g_cpbcon_cblas_call;

static struct {
    int calls;
    char uplo;
    int n;
    int kd;
    int ldab;
    double anorm;
    int work_seen;
    int aux_seen;
    double ab_real_snapshot[8];
} g_zpbcon_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    fb_uplo_t uplo;
    int n;
    int kd;
    int ldab;
    double anorm;
    const fb_complex_double_t *ab;
    double *rcond;
} g_zpbcon_cblas_call;

static fb_complex_float_t make_cfloat(float real_value)
{
    fb_complex_float_t value = (fb_complex_float_t)0;
    memcpy(&value, &real_value, sizeof(real_value));
    return value;
}

static float cfloat_real(fb_complex_float_t value)
{
    float real_value = 0.0f;
    memcpy(&real_value, &value, sizeof(real_value));
    return real_value;
}

static fb_complex_double_t make_cdouble(double real_value)
{
    fb_complex_double_t value = (fb_complex_double_t)0;
    memcpy(&value, &real_value, sizeof(real_value));
    return value;
}

static double cdouble_real(fb_complex_double_t value)
{
    double real_value = 0.0;
    memcpy(&real_value, &value, sizeof(real_value));
    return real_value;
}

static void stub_spbcon_fortran(char *uplo, int *n, int *kd, float *ab,
                                int *ldab, float *anorm, float *rcond,
                                float *work, int *iwork, int *info)
{
    int index = 0;

    g_spbcon_fortran_call.calls += 1;
    g_spbcon_fortran_call.uplo = *uplo;
    g_spbcon_fortran_call.n = *n;
    g_spbcon_fortran_call.kd = *kd;
    g_spbcon_fortran_call.ldab = *ldab;
    g_spbcon_fortran_call.anorm = *anorm;
    g_spbcon_fortran_call.work_seen = (work != NULL);
    g_spbcon_fortran_call.aux_seen = (iwork != NULL);
    for (index = 0; index < (*ldab * *n); ++index) {
        g_spbcon_fortran_call.ab_snapshot[index] = ab[index];
    }
    *rcond = 0.25f;
    *info = 0;
}

static int stub_spbcon_cblas(fb_layout_t layout, fb_uplo_t uplo, int n, int kd,
                             const float *ab, int ldab, float anorm,
                             float *rcond)
{
    g_spbcon_cblas_call.called += 1;
    g_spbcon_cblas_call.layout = layout;
    g_spbcon_cblas_call.uplo = uplo;
    g_spbcon_cblas_call.n = n;
    g_spbcon_cblas_call.kd = kd;
    g_spbcon_cblas_call.ldab = ldab;
    g_spbcon_cblas_call.anorm = anorm;
    g_spbcon_cblas_call.ab = ab;
    g_spbcon_cblas_call.rcond = rcond;
    *rcond = 0.5f;
    return 151;
}

static void stub_dpbcon_fortran(char *uplo, int *n, int *kd, double *ab,
                                int *ldab, double *anorm, double *rcond,
                                double *work, int *iwork, int *info)
{
    int index = 0;

    g_dpbcon_fortran_call.calls += 1;
    g_dpbcon_fortran_call.uplo = *uplo;
    g_dpbcon_fortran_call.n = *n;
    g_dpbcon_fortran_call.kd = *kd;
    g_dpbcon_fortran_call.ldab = *ldab;
    g_dpbcon_fortran_call.anorm = *anorm;
    g_dpbcon_fortran_call.work_seen = (work != NULL);
    g_dpbcon_fortran_call.aux_seen = (iwork != NULL);
    for (index = 0; index < (*ldab * *n); ++index) {
        g_dpbcon_fortran_call.ab_snapshot[index] = ab[index];
    }
    *rcond = 0.35;
    *info = 0;
}

static int stub_dpbcon_cblas(fb_layout_t layout, fb_uplo_t uplo, int n, int kd,
                             const double *ab, int ldab, double anorm,
                             double *rcond)
{
    g_dpbcon_cblas_call.called += 1;
    g_dpbcon_cblas_call.layout = layout;
    g_dpbcon_cblas_call.uplo = uplo;
    g_dpbcon_cblas_call.n = n;
    g_dpbcon_cblas_call.kd = kd;
    g_dpbcon_cblas_call.ldab = ldab;
    g_dpbcon_cblas_call.anorm = anorm;
    g_dpbcon_cblas_call.ab = ab;
    g_dpbcon_cblas_call.rcond = rcond;
    *rcond = 0.6;
    return 152;
}

static void stub_cpbcon_fortran(char *uplo, int *n, int *kd,
                                fb_complex_float_t *ab, int *ldab,
                                float *anorm, float *rcond,
                                fb_complex_float_t *work, float *rwork,
                                int *info)
{
    int index = 0;

    g_cpbcon_fortran_call.calls += 1;
    g_cpbcon_fortran_call.uplo = *uplo;
    g_cpbcon_fortran_call.n = *n;
    g_cpbcon_fortran_call.kd = *kd;
    g_cpbcon_fortran_call.ldab = *ldab;
    g_cpbcon_fortran_call.anorm = *anorm;
    g_cpbcon_fortran_call.work_seen = (work != NULL);
    g_cpbcon_fortran_call.aux_seen = (rwork != NULL);
    for (index = 0; index < (*ldab * *n); ++index) {
        g_cpbcon_fortran_call.ab_real_snapshot[index] = cfloat_real(ab[index]);
    }
    *rcond = 0.75f;
    *info = 0;
}

static int stub_cpbcon_cblas(fb_layout_t layout, fb_uplo_t uplo, int n, int kd,
                             const fb_complex_float_t *ab, int ldab,
                             float anorm, float *rcond)
{
    g_cpbcon_cblas_call.called += 1;
    g_cpbcon_cblas_call.layout = layout;
    g_cpbcon_cblas_call.uplo = uplo;
    g_cpbcon_cblas_call.n = n;
    g_cpbcon_cblas_call.kd = kd;
    g_cpbcon_cblas_call.ldab = ldab;
    g_cpbcon_cblas_call.anorm = anorm;
    g_cpbcon_cblas_call.ab = ab;
    g_cpbcon_cblas_call.rcond = rcond;
    *rcond = 1.25f;
    return 153;
}

static void stub_zpbcon_fortran(char *uplo, int *n, int *kd,
                                fb_complex_double_t *ab, int *ldab,
                                double *anorm, double *rcond,
                                fb_complex_double_t *work, double *rwork,
                                int *info)
{
    int index = 0;

    g_zpbcon_fortran_call.calls += 1;
    g_zpbcon_fortran_call.uplo = *uplo;
    g_zpbcon_fortran_call.n = *n;
    g_zpbcon_fortran_call.kd = *kd;
    g_zpbcon_fortran_call.ldab = *ldab;
    g_zpbcon_fortran_call.anorm = *anorm;
    g_zpbcon_fortran_call.work_seen = (work != NULL);
    g_zpbcon_fortran_call.aux_seen = (rwork != NULL);
    for (index = 0; index < (*ldab * *n); ++index) {
        g_zpbcon_fortran_call.ab_real_snapshot[index] = cdouble_real(ab[index]);
    }
    *rcond = 0.85;
    *info = 0;
}

static int stub_zpbcon_cblas(fb_layout_t layout, fb_uplo_t uplo, int n, int kd,
                             const fb_complex_double_t *ab, int ldab,
                             double anorm, double *rcond)
{
    g_zpbcon_cblas_call.called += 1;
    g_zpbcon_cblas_call.layout = layout;
    g_zpbcon_cblas_call.uplo = uplo;
    g_zpbcon_cblas_call.n = n;
    g_zpbcon_cblas_call.kd = kd;
    g_zpbcon_cblas_call.ldab = ldab;
    g_zpbcon_cblas_call.anorm = anorm;
    g_zpbcon_cblas_call.ab = ab;
    g_zpbcon_cblas_call.rcond = rcond;
    *rcond = 1.35;
    return 154;
}

static int check_spbcon_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_spbcon_fn thunk = NULL;
    float ab[8] = { 10.0f, 11.0f, 12.0f, 13.0f, 20.0f, 21.0f, 22.0f, 23.0f };
    float expected_snapshot[8] = { 0.0f, 20.0f, 11.0f, 21.0f, 12.0f, 22.0f, 13.0f, 23.0f };
    float rcond = 0.0f;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_spbcon_fortran_call, 0, sizeof(g_spbcon_fortran_call));

    vtable.ext_ops[FB_OP_SPBCON][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_spbcon_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SPBCON);

    thunk = (fb_spbcon_fn)vtable.ext_ops[FB_OP_SPBCON][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SPBCON Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, FB_UPPER, 4, 1, ab, 4, 19.0f, &rcond);
    if (info != 0 || g_spbcon_fortran_call.calls != 1 ||
        g_spbcon_fortran_call.uplo != 'U' || g_spbcon_fortran_call.n != 4 ||
        g_spbcon_fortran_call.kd != 1 || g_spbcon_fortran_call.ldab != 2 ||
        g_spbcon_fortran_call.anorm != 19.0f ||
        !g_spbcon_fortran_call.work_seen || !g_spbcon_fortran_call.aux_seen ||
        memcmp(g_spbcon_fortran_call.ab_snapshot, expected_snapshot,
               sizeof(expected_snapshot)) != 0 ||
        rcond != 0.25f) {
        fprintf(stderr, "[FAIL] SPBCON Fortran->CBLAS thunk did not preserve row-major band condition-estimation semantics\n");
        return 1;
    }

    printf("[PASS] SPBCON Fortran->CBLAS thunk transposes row-major band storage and allocates fixed real scratch\n");
    return 0;
}

static int check_spbcon_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_spbcon_fortran_slot_fn thunk = NULL;
    float ab[6] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
    float anorm = 7.0f;
    float rcond = 0.0f;
    float work[9] = { 0.0f };
    int iwork[3] = { 0, 0, 0 };
    char uplo = 'L';
    int n = 3;
    int kd = 1;
    int ldab = 2;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_spbcon_cblas_call, 0, sizeof(g_spbcon_cblas_call));

    vtable.ext_ops[FB_OP_SPBCON][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_spbcon_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SPBCON);

    thunk = (fb_spbcon_fortran_slot_fn)vtable.ext_ops[FB_OP_SPBCON][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SPBCON CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&uplo, &n, &kd, ab, &ldab, &anorm, &rcond, work, iwork, &info);
    if (info != 151 || g_spbcon_cblas_call.called != 1 ||
        g_spbcon_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_spbcon_cblas_call.uplo != FB_LOWER ||
        g_spbcon_cblas_call.n != 3 || g_spbcon_cblas_call.kd != 1 ||
        g_spbcon_cblas_call.ldab != 2 || g_spbcon_cblas_call.anorm != 7.0f ||
        g_spbcon_cblas_call.ab != ab || g_spbcon_cblas_call.rcond != &rcond ||
        rcond != 0.5f) {
        fprintf(stderr, "[FAIL] SPBCON CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] SPBCON CBLAS->Fortran thunk maps the all-pointer ABI into the generic real band-condition entry\n");
    return 0;
}

static int check_dpbcon_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_dpbcon_fn thunk = NULL;
    double ab[8] = { 10.0, 11.0, 12.0, 13.0, 20.0, 21.0, 22.0, 23.0 };
    double expected_snapshot[8] = { 0.0, 20.0, 11.0, 21.0, 12.0, 22.0, 13.0, 23.0 };
    double rcond = 0.0;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dpbcon_fortran_call, 0, sizeof(g_dpbcon_fortran_call));

    vtable.ext_ops[FB_OP_DPBCON][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_dpbcon_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_DPBCON);

    thunk = (fb_dpbcon_fn)vtable.ext_ops[FB_OP_DPBCON][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DPBCON Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, FB_UPPER, 4, 1, ab, 4, 19.0, &rcond);
    if (info != 0 || g_dpbcon_fortran_call.calls != 1 ||
        g_dpbcon_fortran_call.uplo != 'U' || g_dpbcon_fortran_call.n != 4 ||
        g_dpbcon_fortran_call.kd != 1 || g_dpbcon_fortran_call.ldab != 2 ||
        g_dpbcon_fortran_call.anorm != 19.0 ||
        !g_dpbcon_fortran_call.work_seen || !g_dpbcon_fortran_call.aux_seen ||
        memcmp(g_dpbcon_fortran_call.ab_snapshot, expected_snapshot,
               sizeof(expected_snapshot)) != 0 ||
        rcond != 0.35) {
        fprintf(stderr, "[FAIL] DPBCON Fortran->CBLAS thunk did not preserve row-major band condition-estimation semantics\n");
        return 1;
    }

    printf("[PASS] DPBCON Fortran->CBLAS thunk transposes row-major band storage and allocates fixed real scratch\n");
    return 0;
}

static int check_dpbcon_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_dpbcon_fortran_slot_fn thunk = NULL;
    double ab[6] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    double anorm = 7.0;
    double rcond = 0.0;
    double work[9] = { 0.0 };
    int iwork[3] = { 0, 0, 0 };
    char uplo = 'L';
    int n = 3;
    int kd = 1;
    int ldab = 2;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dpbcon_cblas_call, 0, sizeof(g_dpbcon_cblas_call));

    vtable.ext_ops[FB_OP_DPBCON][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_dpbcon_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_DPBCON);

    thunk = (fb_dpbcon_fortran_slot_fn)vtable.ext_ops[FB_OP_DPBCON][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DPBCON CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&uplo, &n, &kd, ab, &ldab, &anorm, &rcond, work, iwork, &info);
    if (info != 152 || g_dpbcon_cblas_call.called != 1 ||
        g_dpbcon_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_dpbcon_cblas_call.uplo != FB_LOWER ||
        g_dpbcon_cblas_call.n != 3 || g_dpbcon_cblas_call.kd != 1 ||
        g_dpbcon_cblas_call.ldab != 2 || g_dpbcon_cblas_call.anorm != 7.0 ||
        g_dpbcon_cblas_call.ab != ab || g_dpbcon_cblas_call.rcond != &rcond ||
        rcond != 0.6) {
        fprintf(stderr, "[FAIL] DPBCON CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] DPBCON CBLAS->Fortran thunk maps the all-pointer ABI into the generic double band-condition entry\n");
    return 0;
}

static int check_cpbcon_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_cpbcon_fn thunk = NULL;
    fb_complex_float_t ab[8] = {
        make_cfloat(30.0f), make_cfloat(31.0f), make_cfloat(32.0f), make_cfloat(33.0f),
        make_cfloat(40.0f), make_cfloat(41.0f), make_cfloat(42.0f), make_cfloat(43.0f)
    };
    float expected_snapshot[8] = { 30.0f, 40.0f, 31.0f, 41.0f, 32.0f, 42.0f, 33.0f, 0.0f };
    float rcond = 0.0f;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cpbcon_fortran_call, 0, sizeof(g_cpbcon_fortran_call));

    vtable.ext_ops[FB_OP_CPBCON][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_cpbcon_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CPBCON);

    thunk = (fb_cpbcon_fn)vtable.ext_ops[FB_OP_CPBCON][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CPBCON Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, FB_LOWER, 4, 1, ab, 4, 29.0f, &rcond);
    if (info != 0 || g_cpbcon_fortran_call.calls != 1 ||
        g_cpbcon_fortran_call.uplo != 'L' || g_cpbcon_fortran_call.n != 4 ||
        g_cpbcon_fortran_call.kd != 1 || g_cpbcon_fortran_call.ldab != 2 ||
        g_cpbcon_fortran_call.anorm != 29.0f ||
        !g_cpbcon_fortran_call.work_seen || !g_cpbcon_fortran_call.aux_seen ||
        memcmp(g_cpbcon_fortran_call.ab_real_snapshot, expected_snapshot,
               sizeof(expected_snapshot)) != 0 ||
        rcond != 0.75f) {
        fprintf(stderr, "[FAIL] CPBCON Fortran->CBLAS thunk did not preserve complex row-major band condition-estimation semantics\n");
        return 1;
    }

    printf("[PASS] CPBCON Fortran->CBLAS thunk transposes row-major Hermitian band storage and allocates complex scratch\n");
    return 0;
}

static int check_cpbcon_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_cpbcon_fortran_slot_fn thunk = NULL;
    fb_complex_float_t ab[6] = {
        make_cfloat(1.0f), make_cfloat(2.0f), make_cfloat(3.0f),
        make_cfloat(4.0f), make_cfloat(5.0f), make_cfloat(6.0f)
    };
    float anorm = 9.0f;
    float rcond = 0.0f;
    fb_complex_float_t work[6];
    float rwork[3] = { 0.0f, 0.0f, 0.0f };
    char uplo = 'U';
    int n = 3;
    int kd = 1;
    int ldab = 2;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cpbcon_cblas_call, 0, sizeof(g_cpbcon_cblas_call));
    memset(work, 0, sizeof(work));

    vtable.ext_ops[FB_OP_CPBCON][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_cpbcon_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CPBCON);

    thunk = (fb_cpbcon_fortran_slot_fn)vtable.ext_ops[FB_OP_CPBCON][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CPBCON CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&uplo, &n, &kd, ab, &ldab, &anorm, &rcond, work, rwork, &info);
    if (info != 153 || g_cpbcon_cblas_call.called != 1 ||
        g_cpbcon_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_cpbcon_cblas_call.uplo != FB_UPPER ||
        g_cpbcon_cblas_call.n != 3 || g_cpbcon_cblas_call.kd != 1 ||
        g_cpbcon_cblas_call.ldab != 2 || g_cpbcon_cblas_call.anorm != 9.0f ||
        g_cpbcon_cblas_call.ab != ab || g_cpbcon_cblas_call.rcond != &rcond ||
        rcond != 1.25f) {
        fprintf(stderr, "[FAIL] CPBCON CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] CPBCON CBLAS->Fortran thunk maps the all-pointer ABI into the generic complex band-condition entry\n");
    return 0;
}

static int check_zpbcon_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_zpbcon_fn thunk = NULL;
    fb_complex_double_t ab[8] = {
        make_cdouble(30.0), make_cdouble(31.0), make_cdouble(32.0), make_cdouble(33.0),
        make_cdouble(40.0), make_cdouble(41.0), make_cdouble(42.0), make_cdouble(43.0)
    };
    double expected_snapshot[8] = { 30.0, 40.0, 31.0, 41.0, 32.0, 42.0, 33.0, 0.0 };
    double rcond = 0.0;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zpbcon_fortran_call, 0, sizeof(g_zpbcon_fortran_call));

    vtable.ext_ops[FB_OP_ZPBCON][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_zpbcon_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_ZPBCON);

    thunk = (fb_zpbcon_fn)vtable.ext_ops[FB_OP_ZPBCON][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZPBCON Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, FB_LOWER, 4, 1, ab, 4, 29.0, &rcond);
    if (info != 0 || g_zpbcon_fortran_call.calls != 1 ||
        g_zpbcon_fortran_call.uplo != 'L' || g_zpbcon_fortran_call.n != 4 ||
        g_zpbcon_fortran_call.kd != 1 || g_zpbcon_fortran_call.ldab != 2 ||
        g_zpbcon_fortran_call.anorm != 29.0 ||
        !g_zpbcon_fortran_call.work_seen || !g_zpbcon_fortran_call.aux_seen ||
        memcmp(g_zpbcon_fortran_call.ab_real_snapshot, expected_snapshot,
               sizeof(expected_snapshot)) != 0 ||
        rcond != 0.85) {
        fprintf(stderr, "[FAIL] ZPBCON Fortran->CBLAS thunk did not preserve complex row-major band condition-estimation semantics\n");
        return 1;
    }

    printf("[PASS] ZPBCON Fortran->CBLAS thunk transposes row-major Hermitian band storage and allocates complex scratch\n");
    return 0;
}

static int check_zpbcon_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_zpbcon_fortran_slot_fn thunk = NULL;
    fb_complex_double_t ab[6] = {
        make_cdouble(1.0), make_cdouble(2.0), make_cdouble(3.0),
        make_cdouble(4.0), make_cdouble(5.0), make_cdouble(6.0)
    };
    double anorm = 9.0;
    double rcond = 0.0;
    fb_complex_double_t work[6];
    double rwork[3] = { 0.0, 0.0, 0.0 };
    char uplo = 'U';
    int n = 3;
    int kd = 1;
    int ldab = 2;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zpbcon_cblas_call, 0, sizeof(g_zpbcon_cblas_call));
    memset(work, 0, sizeof(work));

    vtable.ext_ops[FB_OP_ZPBCON][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_zpbcon_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_ZPBCON);

    thunk = (fb_zpbcon_fortran_slot_fn)vtable.ext_ops[FB_OP_ZPBCON][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZPBCON CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&uplo, &n, &kd, ab, &ldab, &anorm, &rcond, work, rwork, &info);
    if (info != 154 || g_zpbcon_cblas_call.called != 1 ||
        g_zpbcon_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_zpbcon_cblas_call.uplo != FB_UPPER ||
        g_zpbcon_cblas_call.n != 3 || g_zpbcon_cblas_call.kd != 1 ||
        g_zpbcon_cblas_call.ldab != 2 || g_zpbcon_cblas_call.anorm != 9.0 ||
        g_zpbcon_cblas_call.ab != ab || g_zpbcon_cblas_call.rcond != &rcond ||
        rcond != 1.35) {
        fprintf(stderr, "[FAIL] ZPBCON CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] ZPBCON CBLAS->Fortran thunk maps the all-pointer ABI into the generic complex-double band-condition entry\n");
    return 0;
}

int main(void)
{
    if (check_spbcon_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_spbcon_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_dpbcon_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_dpbcon_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_cpbcon_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_cpbcon_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_zpbcon_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_zpbcon_cblas_to_fortran() != 0) {
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}