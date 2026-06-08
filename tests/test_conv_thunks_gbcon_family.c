#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef int (*fb_sgbcon_fn)(fb_layout_t layout, char norm, int n, int kl,
                            int ku, const float *ab, int ldab,
                            const int *ipiv, float anorm, float *rcond);
typedef int (*fb_cgbcon_fn)(fb_layout_t layout, char norm, int n, int kl,
                            int ku, const fb_complex_float_t *ab, int ldab,
                            const int *ipiv, float anorm, float *rcond);

typedef void (*fb_sgbcon_fortran_slot_fn)(char *norm, int *n, int *kl,
                                          int *ku, float *ab, int *ldab,
                                          int *ipiv, float *anorm,
                                          float *rcond, float *work,
                                          int *iwork, int *info);
typedef void (*fb_cgbcon_fortran_slot_fn)(char *norm, int *n, int *kl,
                                          int *ku, fb_complex_float_t *ab,
                                          int *ldab, int *ipiv, float *anorm,
                                          float *rcond,
                                          fb_complex_float_t *work,
                                          int *iwork, int *info);

static struct {
    int calls;
    char norm;
    int n;
    int kl;
    int ku;
    int ldab;
    float anorm;
    int work_seen;
    int aux_seen;
    int ipiv_snapshot[4];
    float ab_snapshot[16];
} g_sgbcon_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    char norm;
    int n;
    int kl;
    int ku;
    int ldab;
    float anorm;
    const float *ab;
    const int *ipiv;
    float *rcond;
} g_sgbcon_cblas_call;

static struct {
    int calls;
    char norm;
    int n;
    int kl;
    int ku;
    int ldab;
    float anorm;
    int work_seen;
    int aux_seen;
    int ipiv_snapshot[4];
    float ab_real_snapshot[16];
} g_cgbcon_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    char norm;
    int n;
    int kl;
    int ku;
    int ldab;
    float anorm;
    const fb_complex_float_t *ab;
    const int *ipiv;
    float *rcond;
} g_cgbcon_cblas_call;

static fb_complex_float_t make_cfloat(float real_value)
{
    fb_complex_float_t value;

    __real__ value = real_value;
    __imag__ value = 0.0f;
    return value;
}

static float cfloat_real(fb_complex_float_t value)
{
    return __real__ value;
}

static void stub_sgbcon_fortran(char *norm, int *n, int *kl, int *ku,
                                float *ab, int *ldab, int *ipiv,
                                float *anorm, float *rcond, float *work,
                                int *iwork, int *info)
{
    int index = 0;

    g_sgbcon_fortran_call.calls += 1;
    g_sgbcon_fortran_call.norm = *norm;
    g_sgbcon_fortran_call.n = *n;
    g_sgbcon_fortran_call.kl = *kl;
    g_sgbcon_fortran_call.ku = *ku;
    g_sgbcon_fortran_call.ldab = *ldab;
    g_sgbcon_fortran_call.anorm = *anorm;
    g_sgbcon_fortran_call.work_seen = (work != NULL);
    g_sgbcon_fortran_call.aux_seen = (iwork != NULL);
    memcpy(g_sgbcon_fortran_call.ipiv_snapshot, ipiv,
           sizeof(g_sgbcon_fortran_call.ipiv_snapshot));
    for (index = 0; index < (*ldab * *n); ++index) {
        g_sgbcon_fortran_call.ab_snapshot[index] = ab[index];
    }
    *rcond = 0.25f;
    *info = 0;
}

static int stub_sgbcon_cblas(fb_layout_t layout, char norm, int n, int kl,
                             int ku, const float *ab, int ldab,
                             const int *ipiv, float anorm, float *rcond)
{
    g_sgbcon_cblas_call.called += 1;
    g_sgbcon_cblas_call.layout = layout;
    g_sgbcon_cblas_call.norm = norm;
    g_sgbcon_cblas_call.n = n;
    g_sgbcon_cblas_call.kl = kl;
    g_sgbcon_cblas_call.ku = ku;
    g_sgbcon_cblas_call.ldab = ldab;
    g_sgbcon_cblas_call.anorm = anorm;
    g_sgbcon_cblas_call.ab = ab;
    g_sgbcon_cblas_call.ipiv = ipiv;
    g_sgbcon_cblas_call.rcond = rcond;
    *rcond = 0.5f;
    return 171;
}

static void stub_cgbcon_fortran(char *norm, int *n, int *kl, int *ku,
                                fb_complex_float_t *ab, int *ldab,
                                int *ipiv, float *anorm, float *rcond,
                                fb_complex_float_t *work, int *iwork,
                                int *info)
{
    int index = 0;

    g_cgbcon_fortran_call.calls += 1;
    g_cgbcon_fortran_call.norm = *norm;
    g_cgbcon_fortran_call.n = *n;
    g_cgbcon_fortran_call.kl = *kl;
    g_cgbcon_fortran_call.ku = *ku;
    g_cgbcon_fortran_call.ldab = *ldab;
    g_cgbcon_fortran_call.anorm = *anorm;
    g_cgbcon_fortran_call.work_seen = (work != NULL);
    g_cgbcon_fortran_call.aux_seen = (iwork != NULL);
    memcpy(g_cgbcon_fortran_call.ipiv_snapshot, ipiv,
           sizeof(g_cgbcon_fortran_call.ipiv_snapshot));
    for (index = 0; index < (*ldab * *n); ++index) {
        g_cgbcon_fortran_call.ab_real_snapshot[index] = cfloat_real(ab[index]);
    }
    *rcond = 0.75f;
    *info = 0;
}

static int stub_cgbcon_cblas(fb_layout_t layout, char norm, int n, int kl,
                             int ku, const fb_complex_float_t *ab, int ldab,
                             const int *ipiv, float anorm, float *rcond)
{
    g_cgbcon_cblas_call.called += 1;
    g_cgbcon_cblas_call.layout = layout;
    g_cgbcon_cblas_call.norm = norm;
    g_cgbcon_cblas_call.n = n;
    g_cgbcon_cblas_call.kl = kl;
    g_cgbcon_cblas_call.ku = ku;
    g_cgbcon_cblas_call.ldab = ldab;
    g_cgbcon_cblas_call.anorm = anorm;
    g_cgbcon_cblas_call.ab = ab;
    g_cgbcon_cblas_call.ipiv = ipiv;
    g_cgbcon_cblas_call.rcond = rcond;
    *rcond = 1.25f;
    return 173;
}

static int check_sgbcon_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_sgbcon_fn thunk = NULL;
    float ab[16] = {
        0.0f, 0.0f, 130.0f, 140.0f,
        0.0f, 220.0f, 230.0f, 240.0f,
        310.0f, 320.0f, 330.0f, 340.0f,
        410.0f, 420.0f, 430.0f, 0.0f
    };
    int ipiv[4] = { 1, 2, 3, 4 };
    static const float expected_snapshot[16] = {
        0.0f, 0.0f, 310.0f, 410.0f,
        0.0f, 220.0f, 320.0f, 420.0f,
        130.0f, 230.0f, 330.0f, 430.0f,
        140.0f, 240.0f, 340.0f, 0.0f
    };
    float rcond = 0.0f;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sgbcon_fortran_call, 0, sizeof(g_sgbcon_fortran_call));

    vtable.ext_ops[FB_OP_SGBCON][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_sgbcon_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SGBCON);

    thunk = (fb_sgbcon_fn)vtable.ext_ops[FB_OP_SGBCON][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SGBCON Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, '1', 4, 1, 1, ab, 4, ipiv, 19.0f,
                 &rcond);
    if (info != 0 || g_sgbcon_fortran_call.calls != 1 ||
        g_sgbcon_fortran_call.norm != '1' || g_sgbcon_fortran_call.n != 4 ||
        g_sgbcon_fortran_call.kl != 1 || g_sgbcon_fortran_call.ku != 1 ||
        g_sgbcon_fortran_call.ldab != 4 ||
        g_sgbcon_fortran_call.anorm != 19.0f ||
        !g_sgbcon_fortran_call.work_seen || !g_sgbcon_fortran_call.aux_seen ||
        memcmp(g_sgbcon_fortran_call.ipiv_snapshot, ipiv,
               sizeof(ipiv)) != 0 ||
        memcmp(g_sgbcon_fortran_call.ab_snapshot, expected_snapshot,
               sizeof(expected_snapshot)) != 0 ||
        rcond != 0.25f) {
        fprintf(stderr, "[FAIL] SGBCON Fortran->CBLAS thunk did not preserve general-band factor-storage semantics\n");
        return 1;
    }

    printf("[PASS] SGBCON Fortran->CBLAS thunk transposes expanded row-major GBTRF storage and allocates fixed real scratch\n");
    return 0;
}

static int check_sgbcon_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_sgbcon_fortran_slot_fn thunk = NULL;
    float ab[12] = {
        0.0f, 0.0f, 130.0f,
        0.0f, 220.0f, 230.0f,
        310.0f, 320.0f, 330.0f,
        410.0f, 420.0f, 0.0f
    };
    int ipiv[3] = { 2, 1, 3 };
    float anorm = 7.0f;
    float rcond = 0.0f;
    float work[9] = { 0.0f };
    int iwork[3] = { 0, 0, 0 };
    char norm = 'I';
    int n = 3;
    int kl = 1;
    int ku = 1;
    int ldab = 4;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sgbcon_cblas_call, 0, sizeof(g_sgbcon_cblas_call));

    vtable.ext_ops[FB_OP_SGBCON][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_sgbcon_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SGBCON);

    thunk = (fb_sgbcon_fortran_slot_fn)vtable.ext_ops[FB_OP_SGBCON][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SGBCON CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&norm, &n, &kl, &ku, ab, &ldab, ipiv, &anorm, &rcond, work, iwork,
          &info);
    if (info != 171 || g_sgbcon_cblas_call.called != 1 ||
        g_sgbcon_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_sgbcon_cblas_call.norm != 'I' || g_sgbcon_cblas_call.n != 3 ||
        g_sgbcon_cblas_call.kl != 1 || g_sgbcon_cblas_call.ku != 1 ||
        g_sgbcon_cblas_call.ldab != 4 || g_sgbcon_cblas_call.anorm != 7.0f ||
        g_sgbcon_cblas_call.ab != ab || g_sgbcon_cblas_call.ipiv != ipiv ||
        g_sgbcon_cblas_call.rcond != &rcond || rcond != 0.5f) {
        fprintf(stderr, "[FAIL] SGBCON CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] SGBCON CBLAS->Fortran thunk maps the all-pointer ABI into the general band-condition entry\n");
    return 0;
}

static int check_cgbcon_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_cgbcon_fn thunk = NULL;
    fb_complex_float_t ab[16] = {
        make_cfloat(0.0f), make_cfloat(0.0f), make_cfloat(530.0f), make_cfloat(540.0f),
        make_cfloat(0.0f), make_cfloat(620.0f), make_cfloat(630.0f), make_cfloat(640.0f),
        make_cfloat(710.0f), make_cfloat(720.0f), make_cfloat(730.0f), make_cfloat(740.0f),
        make_cfloat(810.0f), make_cfloat(820.0f), make_cfloat(830.0f), make_cfloat(0.0f)
    };
    int ipiv[4] = { 4, 2, 3, 1 };
    static const float expected_snapshot[16] = {
        0.0f, 0.0f, 710.0f, 810.0f,
        0.0f, 620.0f, 720.0f, 820.0f,
        530.0f, 630.0f, 730.0f, 830.0f,
        540.0f, 640.0f, 740.0f, 0.0f
    };
    float rcond = 0.0f;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cgbcon_fortran_call, 0, sizeof(g_cgbcon_fortran_call));

    vtable.ext_ops[FB_OP_CGBCON][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_cgbcon_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CGBCON);

    thunk = (fb_cgbcon_fn)vtable.ext_ops[FB_OP_CGBCON][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CGBCON Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, 'O', 4, 1, 1, ab, 4, ipiv, 29.0f,
                 &rcond);
    if (info != 0 || g_cgbcon_fortran_call.calls != 1 ||
        g_cgbcon_fortran_call.norm != 'O' || g_cgbcon_fortran_call.n != 4 ||
        g_cgbcon_fortran_call.kl != 1 || g_cgbcon_fortran_call.ku != 1 ||
        g_cgbcon_fortran_call.ldab != 4 ||
        g_cgbcon_fortran_call.anorm != 29.0f ||
        !g_cgbcon_fortran_call.work_seen || !g_cgbcon_fortran_call.aux_seen ||
        memcmp(g_cgbcon_fortran_call.ipiv_snapshot, ipiv,
               sizeof(ipiv)) != 0 ||
        memcmp(g_cgbcon_fortran_call.ab_real_snapshot, expected_snapshot,
               sizeof(expected_snapshot)) != 0 ||
        rcond != 0.75f) {
        fprintf(stderr, "[FAIL] CGBCON Fortran->CBLAS thunk did not preserve complex general-band factor-storage semantics\n");
        return 1;
    }

    printf("[PASS] CGBCON Fortran->CBLAS thunk transposes expanded row-major GBTRF storage and allocates fixed complex scratch\n");
    return 0;
}

static int check_cgbcon_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_cgbcon_fortran_slot_fn thunk = NULL;
    fb_complex_float_t ab[12] = {
        make_cfloat(0.0f), make_cfloat(0.0f), make_cfloat(530.0f),
        make_cfloat(0.0f), make_cfloat(620.0f), make_cfloat(630.0f),
        make_cfloat(710.0f), make_cfloat(720.0f), make_cfloat(730.0f),
        make_cfloat(810.0f), make_cfloat(820.0f), make_cfloat(0.0f)
    };
    int ipiv[3] = { 3, 1, 2 };
    float anorm = 9.0f;
    float rcond = 0.0f;
    fb_complex_float_t work[6];
    int iwork[3] = { 0, 0, 0 };
    char norm = 'M';
    int n = 3;
    int kl = 1;
    int ku = 1;
    int ldab = 4;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cgbcon_cblas_call, 0, sizeof(g_cgbcon_cblas_call));
    memset(work, 0, sizeof(work));

    vtable.ext_ops[FB_OP_CGBCON][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_cgbcon_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CGBCON);

    thunk = (fb_cgbcon_fortran_slot_fn)vtable.ext_ops[FB_OP_CGBCON][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CGBCON CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&norm, &n, &kl, &ku, ab, &ldab, ipiv, &anorm, &rcond, work, iwork,
          &info);
    if (info != 173 || g_cgbcon_cblas_call.called != 1 ||
        g_cgbcon_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_cgbcon_cblas_call.norm != 'M' || g_cgbcon_cblas_call.n != 3 ||
        g_cgbcon_cblas_call.kl != 1 || g_cgbcon_cblas_call.ku != 1 ||
        g_cgbcon_cblas_call.ldab != 4 || g_cgbcon_cblas_call.anorm != 9.0f ||
        g_cgbcon_cblas_call.ab != ab || g_cgbcon_cblas_call.ipiv != ipiv ||
        g_cgbcon_cblas_call.rcond != &rcond || rcond != 1.25f) {
        fprintf(stderr, "[FAIL] CGBCON CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] CGBCON CBLAS->Fortran thunk maps the all-pointer ABI into the general complex band-condition entry\n");
    return 0;
}

int main(void)
{
    if (check_sgbcon_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_sgbcon_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_cgbcon_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_cgbcon_cblas_to_fortran() != 0) {
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}