#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef int (*fb_spbequ_fn)(fb_layout_t layout, fb_uplo_t uplo, int n, int kd,
                            const float *ab, int ldab, float *s,
                            float *scond, float *amax);
typedef int (*fb_cpbequ_fn)(fb_layout_t layout, fb_uplo_t uplo, int n, int kd,
                            const fb_complex_float_t *ab, int ldab, float *s,
                            float *scond, float *amax);

typedef void (*fb_spbequ_fortran_slot_fn)(char *uplo, int *n, int *kd,
                                          float *ab, int *ldab, float *s,
                                          float *scond, float *amax,
                                          int *info);
typedef void (*fb_cpbequ_fortran_slot_fn)(char *uplo, int *n, int *kd,
                                          fb_complex_float_t *ab, int *ldab,
                                          float *s, float *scond,
                                          float *amax, int *info);

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

int main(void)
{
    if (check_spbequ_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_spbequ_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_cpbequ_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_cpbequ_cblas_to_fortran() != 0) {
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}