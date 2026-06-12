#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef int (*fb_spbequ_fn)(fb_layout_t layout, fb_uplo_t uplo, int n, int kd,
                            const float *ab, int ldab, float *s,
                            float *scond, float *amax);
typedef int (*fb_dpbequ_fn)(fb_layout_t layout, fb_uplo_t uplo, int n, int kd,
                            const double *ab, int ldab, double *s,
                            double *scond, double *amax);
typedef int (*fb_cpbequ_fn)(fb_layout_t layout, fb_uplo_t uplo, int n, int kd,
                            const fb_complex_float_t *ab, int ldab, float *s,
                            float *scond, float *amax);
typedef int (*fb_zpbequ_fn)(fb_layout_t layout, fb_uplo_t uplo, int n, int kd,
                            const fb_complex_double_t *ab, int ldab,
                            double *s, double *scond, double *amax);

typedef void (*fb_spbequ_fortran_slot_fn)(char *uplo, int *n, int *kd,
                                          float *ab, int *ldab, float *s,
                                          float *scond, float *amax,
                                          int *info);
typedef void (*fb_dpbequ_fortran_slot_fn)(char *uplo, int *n, int *kd,
                                          double *ab, int *ldab, double *s,
                                          double *scond, double *amax,
                                          int *info);
typedef void (*fb_cpbequ_fortran_slot_fn)(char *uplo, int *n, int *kd,
                                          fb_complex_float_t *ab, int *ldab,
                                          float *s, float *scond,
                                          float *amax, int *info);
typedef void (*fb_zpbequ_fortran_slot_fn)(char *uplo, int *n, int *kd,
                                          fb_complex_double_t *ab, int *ldab,
                                          double *s, double *scond,
                                          double *amax, int *info);

static struct {
    int calls;
    char uplo;
    int n;
    int kd;
    int ldab;
    float ab_snapshot[8];
} g_spbequ_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    fb_uplo_t uplo;
    int n;
    int kd;
    int ldab;
    const float *ab;
    float *s;
    float *scond;
    float *amax;
} g_spbequ_cblas_call;

static struct {
    int calls;
    char uplo;
    int n;
    int kd;
    int ldab;
    double ab_snapshot[8];
} g_dpbequ_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    fb_uplo_t uplo;
    int n;
    int kd;
    int ldab;
    const double *ab;
    double *s;
    double *scond;
    double *amax;
} g_dpbequ_cblas_call;

static struct {
    int calls;
    char uplo;
    int n;
    int kd;
    int ldab;
    float ab_real_snapshot[8];
} g_cpbequ_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    fb_uplo_t uplo;
    int n;
    int kd;
    int ldab;
    const fb_complex_float_t *ab;
    float *s;
    float *scond;
    float *amax;
} g_cpbequ_cblas_call;

static struct {
    int calls;
    char uplo;
    int n;
    int kd;
    int ldab;
    double ab_real_snapshot[8];
} g_zpbequ_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    fb_uplo_t uplo;
    int n;
    int kd;
    int ldab;
    const fb_complex_double_t *ab;
    double *s;
    double *scond;
    double *amax;
} g_zpbequ_cblas_call;

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

static void stub_spbequ_fortran(char *uplo, int *n, int *kd, float *ab,
                                int *ldab, float *s, float *scond,
                                float *amax, int *info)
{
    int index = 0;

    g_spbequ_fortran_call.calls += 1;
    g_spbequ_fortran_call.uplo = *uplo;
    g_spbequ_fortran_call.n = *n;
    g_spbequ_fortran_call.kd = *kd;
    g_spbequ_fortran_call.ldab = *ldab;
    for (index = 0; index < (*ldab * *n); ++index) {
        g_spbequ_fortran_call.ab_snapshot[index] = ab[index];
    }
    for (index = 0; index < *n; ++index) {
        s[index] = (float)(index + 1);
    }
    *scond = 0.25f;
    *amax = 23.0f;
    *info = 0;
}

static int stub_spbequ_cblas(fb_layout_t layout, fb_uplo_t uplo, int n, int kd,
                             const float *ab, int ldab, float *s,
                             float *scond, float *amax)
{
    int index = 0;

    g_spbequ_cblas_call.called += 1;
    g_spbequ_cblas_call.layout = layout;
    g_spbequ_cblas_call.uplo = uplo;
    g_spbequ_cblas_call.n = n;
    g_spbequ_cblas_call.kd = kd;
    g_spbequ_cblas_call.ldab = ldab;
    g_spbequ_cblas_call.ab = ab;
    g_spbequ_cblas_call.s = s;
    g_spbequ_cblas_call.scond = scond;
    g_spbequ_cblas_call.amax = amax;
    for (index = 0; index < n; ++index) {
        s[index] = (float)(10 + index);
    }
    *scond = 0.5f;
    *amax = 31.0f;
    return 141;
}

static void stub_dpbequ_fortran(char *uplo, int *n, int *kd, double *ab,
                                int *ldab, double *s, double *scond,
                                double *amax, int *info)
{
    int index = 0;

    g_dpbequ_fortran_call.calls += 1;
    g_dpbequ_fortran_call.uplo = *uplo;
    g_dpbequ_fortran_call.n = *n;
    g_dpbequ_fortran_call.kd = *kd;
    g_dpbequ_fortran_call.ldab = *ldab;
    for (index = 0; index < (*ldab * *n); ++index) {
        g_dpbequ_fortran_call.ab_snapshot[index] = ab[index];
    }
    for (index = 0; index < *n; ++index) {
        s[index] = (double)(index + 1);
    }
    *scond = 0.35;
    *amax = 24.0;
    *info = 0;
}

static int stub_dpbequ_cblas(fb_layout_t layout, fb_uplo_t uplo, int n, int kd,
                             const double *ab, int ldab, double *s,
                             double *scond, double *amax)
{
    int index = 0;

    g_dpbequ_cblas_call.called += 1;
    g_dpbequ_cblas_call.layout = layout;
    g_dpbequ_cblas_call.uplo = uplo;
    g_dpbequ_cblas_call.n = n;
    g_dpbequ_cblas_call.kd = kd;
    g_dpbequ_cblas_call.ldab = ldab;
    g_dpbequ_cblas_call.ab = ab;
    g_dpbequ_cblas_call.s = s;
    g_dpbequ_cblas_call.scond = scond;
    g_dpbequ_cblas_call.amax = amax;
    for (index = 0; index < n; ++index) {
        s[index] = (double)(10 + index);
    }
    *scond = 0.6;
    *amax = 32.0;
    return 142;
}

static void stub_cpbequ_fortran(char *uplo, int *n, int *kd,
                                fb_complex_float_t *ab, int *ldab,
                                float *s, float *scond, float *amax,
                                int *info)
{
    int index = 0;

    g_cpbequ_fortran_call.calls += 1;
    g_cpbequ_fortran_call.uplo = *uplo;
    g_cpbequ_fortran_call.n = *n;
    g_cpbequ_fortran_call.kd = *kd;
    g_cpbequ_fortran_call.ldab = *ldab;
    for (index = 0; index < (*ldab * *n); ++index) {
        g_cpbequ_fortran_call.ab_real_snapshot[index] = cfloat_real(ab[index]);
    }
    for (index = 0; index < *n; ++index) {
        s[index] = (float)(20 + index);
    }
    *scond = 0.75f;
    *amax = 43.0f;
    *info = 0;
}

static int stub_cpbequ_cblas(fb_layout_t layout, fb_uplo_t uplo, int n, int kd,
                             const fb_complex_float_t *ab, int ldab, float *s,
                             float *scond, float *amax)
{
    int index = 0;

    g_cpbequ_cblas_call.called += 1;
    g_cpbequ_cblas_call.layout = layout;
    g_cpbequ_cblas_call.uplo = uplo;
    g_cpbequ_cblas_call.n = n;
    g_cpbequ_cblas_call.kd = kd;
    g_cpbequ_cblas_call.ldab = ldab;
    g_cpbequ_cblas_call.ab = ab;
    g_cpbequ_cblas_call.s = s;
    g_cpbequ_cblas_call.scond = scond;
    g_cpbequ_cblas_call.amax = amax;
    for (index = 0; index < n; ++index) {
        s[index] = (float)(30 + index);
    }
    *scond = 1.25f;
    *amax = 53.0f;
    return 143;
}

static void stub_zpbequ_fortran(char *uplo, int *n, int *kd,
                                fb_complex_double_t *ab, int *ldab,
                                double *s, double *scond, double *amax,
                                int *info)
{
    int index = 0;

    g_zpbequ_fortran_call.calls += 1;
    g_zpbequ_fortran_call.uplo = *uplo;
    g_zpbequ_fortran_call.n = *n;
    g_zpbequ_fortran_call.kd = *kd;
    g_zpbequ_fortran_call.ldab = *ldab;
    for (index = 0; index < (*ldab * *n); ++index) {
        g_zpbequ_fortran_call.ab_real_snapshot[index] = cdouble_real(ab[index]);
    }
    for (index = 0; index < *n; ++index) {
        s[index] = (double)(20 + index);
    }
    *scond = 0.85;
    *amax = 44.0;
    *info = 0;
}

static int stub_zpbequ_cblas(fb_layout_t layout, fb_uplo_t uplo, int n, int kd,
                             const fb_complex_double_t *ab, int ldab,
                             double *s, double *scond, double *amax)
{
    int index = 0;

    g_zpbequ_cblas_call.called += 1;
    g_zpbequ_cblas_call.layout = layout;
    g_zpbequ_cblas_call.uplo = uplo;
    g_zpbequ_cblas_call.n = n;
    g_zpbequ_cblas_call.kd = kd;
    g_zpbequ_cblas_call.ldab = ldab;
    g_zpbequ_cblas_call.ab = ab;
    g_zpbequ_cblas_call.s = s;
    g_zpbequ_cblas_call.scond = scond;
    g_zpbequ_cblas_call.amax = amax;
    for (index = 0; index < n; ++index) {
        s[index] = (double)(30 + index);
    }
    *scond = 1.35;
    *amax = 54.0;
    return 144;
}

static int check_spbequ_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_spbequ_fn thunk = NULL;
    float ab[8] = { 10.0f, 11.0f, 12.0f, 13.0f, 20.0f, 21.0f, 22.0f, 23.0f };
    float s[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float scond = 0.0f;
    float amax = 0.0f;
    float expected_snapshot[8] = { 0.0f, 20.0f, 11.0f, 21.0f, 12.0f, 22.0f, 13.0f, 23.0f };
    float expected_s[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_spbequ_fortran_call, 0, sizeof(g_spbequ_fortran_call));

    vtable.ext_ops[FB_OP_SPBEQU][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_spbequ_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SPBEQU);

    thunk = (fb_spbequ_fn)vtable.ext_ops[FB_OP_SPBEQU][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SPBEQU Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, FB_UPPER, 4, 1, ab, 4, s, &scond, &amax);
    if (info != 0 || g_spbequ_fortran_call.calls != 1 ||
        g_spbequ_fortran_call.uplo != 'U' || g_spbequ_fortran_call.n != 4 ||
        g_spbequ_fortran_call.kd != 1 || g_spbequ_fortran_call.ldab != 2 ||
        memcmp(g_spbequ_fortran_call.ab_snapshot, expected_snapshot,
               sizeof(expected_snapshot)) != 0 ||
        memcmp(s, expected_s, sizeof(expected_s)) != 0 ||
        scond != 0.25f || amax != 23.0f) {
        fprintf(stderr, "[FAIL] SPBEQU Fortran->CBLAS thunk did not preserve row-major band equilibration semantics\n");
        return 1;
    }

    printf("[PASS] SPBEQU Fortran->CBLAS thunk transposes row-major symmetric band storage and preserves scalar outputs\n");
    return 0;
}

static int check_spbequ_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_spbequ_fortran_slot_fn thunk = NULL;
    float ab[6] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
    float s[3] = { 0.0f, 0.0f, 0.0f };
    float scond = 0.0f;
    float amax = 0.0f;
    char uplo = 'L';
    int n = 3;
    int kd = 1;
    int ldab = 2;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_spbequ_cblas_call, 0, sizeof(g_spbequ_cblas_call));

    vtable.ext_ops[FB_OP_SPBEQU][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_spbequ_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SPBEQU);

    thunk = (fb_spbequ_fortran_slot_fn)vtable.ext_ops[FB_OP_SPBEQU][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SPBEQU CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&uplo, &n, &kd, ab, &ldab, s, &scond, &amax, &info);
    if (info != 141 || g_spbequ_cblas_call.called != 1 ||
        g_spbequ_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_spbequ_cblas_call.uplo != FB_LOWER ||
        g_spbequ_cblas_call.n != 3 || g_spbequ_cblas_call.kd != 1 ||
        g_spbequ_cblas_call.ldab != 2 || g_spbequ_cblas_call.ab != ab ||
        g_spbequ_cblas_call.s != s || g_spbequ_cblas_call.scond != &scond ||
        g_spbequ_cblas_call.amax != &amax ||
        s[0] != 10.0f || s[1] != 11.0f || s[2] != 12.0f ||
        scond != 0.5f || amax != 31.0f) {
        fprintf(stderr, "[FAIL] SPBEQU CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] SPBEQU CBLAS->Fortran thunk maps the all-pointer ABI into the generic C band-equilibration entry\n");
    return 0;
}

static int check_dpbequ_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_dpbequ_fn thunk = NULL;
    double ab[8] = { 10.0, 11.0, 12.0, 13.0, 20.0, 21.0, 22.0, 23.0 };
    double s[4] = { 0.0, 0.0, 0.0, 0.0 };
    double scond = 0.0;
    double amax = 0.0;
    double expected_snapshot[8] = { 0.0, 20.0, 11.0, 21.0, 12.0, 22.0, 13.0, 23.0 };
    double expected_s[4] = { 1.0, 2.0, 3.0, 4.0 };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dpbequ_fortran_call, 0, sizeof(g_dpbequ_fortran_call));

    vtable.ext_ops[FB_OP_DPBEQU][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_dpbequ_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_DPBEQU);

    thunk = (fb_dpbequ_fn)vtable.ext_ops[FB_OP_DPBEQU][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DPBEQU Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, FB_UPPER, 4, 1, ab, 4, s, &scond, &amax);
    if (info != 0 || g_dpbequ_fortran_call.calls != 1 ||
        g_dpbequ_fortran_call.uplo != 'U' || g_dpbequ_fortran_call.n != 4 ||
        g_dpbequ_fortran_call.kd != 1 || g_dpbequ_fortran_call.ldab != 2 ||
        memcmp(g_dpbequ_fortran_call.ab_snapshot, expected_snapshot,
               sizeof(expected_snapshot)) != 0 ||
        memcmp(s, expected_s, sizeof(expected_s)) != 0 ||
        scond != 0.35 || amax != 24.0) {
        fprintf(stderr, "[FAIL] DPBEQU Fortran->CBLAS thunk did not preserve row-major band equilibration semantics\n");
        return 1;
    }

    printf("[PASS] DPBEQU Fortran->CBLAS thunk transposes row-major symmetric band storage and preserves scalar outputs\n");
    return 0;
}

static int check_dpbequ_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_dpbequ_fortran_slot_fn thunk = NULL;
    double ab[6] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    double s[3] = { 0.0, 0.0, 0.0 };
    double scond = 0.0;
    double amax = 0.0;
    char uplo = 'L';
    int n = 3;
    int kd = 1;
    int ldab = 2;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_dpbequ_cblas_call, 0, sizeof(g_dpbequ_cblas_call));

    vtable.ext_ops[FB_OP_DPBEQU][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_dpbequ_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_DPBEQU);

    thunk = (fb_dpbequ_fortran_slot_fn)vtable.ext_ops[FB_OP_DPBEQU][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] DPBEQU CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&uplo, &n, &kd, ab, &ldab, s, &scond, &amax, &info);
    if (info != 142 || g_dpbequ_cblas_call.called != 1 ||
        g_dpbequ_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_dpbequ_cblas_call.uplo != FB_LOWER ||
        g_dpbequ_cblas_call.n != 3 || g_dpbequ_cblas_call.kd != 1 ||
        g_dpbequ_cblas_call.ldab != 2 || g_dpbequ_cblas_call.ab != ab ||
        g_dpbequ_cblas_call.s != s || g_dpbequ_cblas_call.scond != &scond ||
        g_dpbequ_cblas_call.amax != &amax ||
        s[0] != 10.0 || s[1] != 11.0 || s[2] != 12.0 ||
        scond != 0.6 || amax != 32.0) {
        fprintf(stderr, "[FAIL] DPBEQU CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] DPBEQU CBLAS->Fortran thunk maps the all-pointer ABI into the generic C double band-equilibration entry\n");
    return 0;
}

static int check_cpbequ_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_cpbequ_fn thunk = NULL;
    fb_complex_float_t ab[8] = {
        make_cfloat(30.0f), make_cfloat(31.0f), make_cfloat(32.0f), make_cfloat(33.0f),
        make_cfloat(40.0f), make_cfloat(41.0f), make_cfloat(42.0f), make_cfloat(43.0f)
    };
    float s[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float scond = 0.0f;
    float amax = 0.0f;
    float expected_snapshot[8] = { 30.0f, 40.0f, 31.0f, 41.0f, 32.0f, 42.0f, 33.0f, 0.0f };
    float expected_s[4] = { 20.0f, 21.0f, 22.0f, 23.0f };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cpbequ_fortran_call, 0, sizeof(g_cpbequ_fortran_call));

    vtable.ext_ops[FB_OP_CPBEQU][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_cpbequ_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CPBEQU);

    thunk = (fb_cpbequ_fn)vtable.ext_ops[FB_OP_CPBEQU][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CPBEQU Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, FB_LOWER, 4, 1, ab, 4, s, &scond, &amax);
    if (info != 0 || g_cpbequ_fortran_call.calls != 1 ||
        g_cpbequ_fortran_call.uplo != 'L' || g_cpbequ_fortran_call.n != 4 ||
        g_cpbequ_fortran_call.kd != 1 || g_cpbequ_fortran_call.ldab != 2 ||
        memcmp(g_cpbequ_fortran_call.ab_real_snapshot, expected_snapshot,
               sizeof(expected_snapshot)) != 0 ||
        memcmp(s, expected_s, sizeof(expected_s)) != 0 ||
        scond != 0.75f || amax != 43.0f) {
        fprintf(stderr, "[FAIL] CPBEQU Fortran->CBLAS thunk did not preserve complex row-major band equilibration semantics\n");
        return 1;
    }

    printf("[PASS] CPBEQU Fortran->CBLAS thunk transposes row-major Hermitian band storage and preserves real scalar outputs\n");
    return 0;
}

static int check_cpbequ_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_cpbequ_fortran_slot_fn thunk = NULL;
    fb_complex_float_t ab[6] = {
        make_cfloat(1.0f), make_cfloat(2.0f), make_cfloat(3.0f),
        make_cfloat(4.0f), make_cfloat(5.0f), make_cfloat(6.0f)
    };
    float s[3] = { 0.0f, 0.0f, 0.0f };
    float scond = 0.0f;
    float amax = 0.0f;
    char uplo = 'U';
    int n = 3;
    int kd = 1;
    int ldab = 2;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cpbequ_cblas_call, 0, sizeof(g_cpbequ_cblas_call));

    vtable.ext_ops[FB_OP_CPBEQU][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_cpbequ_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CPBEQU);

    thunk = (fb_cpbequ_fortran_slot_fn)vtable.ext_ops[FB_OP_CPBEQU][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CPBEQU CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&uplo, &n, &kd, ab, &ldab, s, &scond, &amax, &info);
    if (info != 143 || g_cpbequ_cblas_call.called != 1 ||
        g_cpbequ_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_cpbequ_cblas_call.uplo != FB_UPPER ||
        g_cpbequ_cblas_call.n != 3 || g_cpbequ_cblas_call.kd != 1 ||
        g_cpbequ_cblas_call.ldab != 2 || g_cpbequ_cblas_call.ab != ab ||
        g_cpbequ_cblas_call.s != s || g_cpbequ_cblas_call.scond != &scond ||
        g_cpbequ_cblas_call.amax != &amax ||
        s[0] != 30.0f || s[1] != 31.0f || s[2] != 32.0f ||
        scond != 1.25f || amax != 53.0f) {
        fprintf(stderr, "[FAIL] CPBEQU CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] CPBEQU CBLAS->Fortran thunk maps the all-pointer ABI into the generic complex band-equilibration entry\n");
    return 0;
}

static int check_zpbequ_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_zpbequ_fn thunk = NULL;
    fb_complex_double_t ab[8] = {
        make_cdouble(30.0), make_cdouble(31.0), make_cdouble(32.0), make_cdouble(33.0),
        make_cdouble(40.0), make_cdouble(41.0), make_cdouble(42.0), make_cdouble(43.0)
    };
    double s[4] = { 0.0, 0.0, 0.0, 0.0 };
    double scond = 0.0;
    double amax = 0.0;
    double expected_snapshot[8] = { 30.0, 40.0, 31.0, 41.0, 32.0, 42.0, 33.0, 0.0 };
    double expected_s[4] = { 20.0, 21.0, 22.0, 23.0 };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zpbequ_fortran_call, 0, sizeof(g_zpbequ_fortran_call));

    vtable.ext_ops[FB_OP_ZPBEQU][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_zpbequ_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_ZPBEQU);

    thunk = (fb_zpbequ_fn)vtable.ext_ops[FB_OP_ZPBEQU][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZPBEQU Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, FB_LOWER, 4, 1, ab, 4, s, &scond, &amax);
    if (info != 0 || g_zpbequ_fortran_call.calls != 1 ||
        g_zpbequ_fortran_call.uplo != 'L' || g_zpbequ_fortran_call.n != 4 ||
        g_zpbequ_fortran_call.kd != 1 || g_zpbequ_fortran_call.ldab != 2 ||
        memcmp(g_zpbequ_fortran_call.ab_real_snapshot, expected_snapshot,
               sizeof(expected_snapshot)) != 0 ||
        memcmp(s, expected_s, sizeof(expected_s)) != 0 ||
        scond != 0.85 || amax != 44.0) {
        fprintf(stderr, "[FAIL] ZPBEQU Fortran->CBLAS thunk did not preserve complex row-major band equilibration semantics\n");
        return 1;
    }

    printf("[PASS] ZPBEQU Fortran->CBLAS thunk transposes row-major Hermitian band storage and preserves real scalar outputs\n");
    return 0;
}

static int check_zpbequ_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_zpbequ_fortran_slot_fn thunk = NULL;
    fb_complex_double_t ab[6] = {
        make_cdouble(1.0), make_cdouble(2.0), make_cdouble(3.0),
        make_cdouble(4.0), make_cdouble(5.0), make_cdouble(6.0)
    };
    double s[3] = { 0.0, 0.0, 0.0 };
    double scond = 0.0;
    double amax = 0.0;
    char uplo = 'U';
    int n = 3;
    int kd = 1;
    int ldab = 2;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_zpbequ_cblas_call, 0, sizeof(g_zpbequ_cblas_call));

    vtable.ext_ops[FB_OP_ZPBEQU][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_zpbequ_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_ZPBEQU);

    thunk = (fb_zpbequ_fortran_slot_fn)vtable.ext_ops[FB_OP_ZPBEQU][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] ZPBEQU CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&uplo, &n, &kd, ab, &ldab, s, &scond, &amax, &info);
    if (info != 144 || g_zpbequ_cblas_call.called != 1 ||
        g_zpbequ_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_zpbequ_cblas_call.uplo != FB_UPPER ||
        g_zpbequ_cblas_call.n != 3 || g_zpbequ_cblas_call.kd != 1 ||
        g_zpbequ_cblas_call.ldab != 2 || g_zpbequ_cblas_call.ab != ab ||
        g_zpbequ_cblas_call.s != s || g_zpbequ_cblas_call.scond != &scond ||
        g_zpbequ_cblas_call.amax != &amax ||
        s[0] != 30.0 || s[1] != 31.0 || s[2] != 32.0 ||
        scond != 1.35 || amax != 54.0) {
        fprintf(stderr, "[FAIL] ZPBEQU CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] ZPBEQU CBLAS->Fortran thunk maps the all-pointer ABI into the generic complex-double band-equilibration entry\n");
    return 0;
}

int main(void)
{
    if (check_spbequ_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_spbequ_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_dpbequ_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_dpbequ_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_cpbequ_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_cpbequ_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_zpbequ_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_zpbequ_cblas_to_fortran() != 0) {
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}