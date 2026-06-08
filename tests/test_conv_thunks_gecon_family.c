#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef int (*fb_sgecon_fn)(fb_layout_t layout, char norm, int n, const float *a,
                            int lda, float anorm, float *rcond);
typedef int (*fb_cgecon_fn)(fb_layout_t layout, char norm, int n,
                            const fb_complex_float_t *a, int lda,
                            float anorm, float *rcond);

typedef void (*fb_sgecon_fortran_slot_fn)(char *norm, int *n, float *a,
                                          int *lda, float *anorm, float *rcond,
                                          float *work, int *iwork, int *info);
typedef void (*fb_cgecon_fortran_slot_fn)(char *norm, int *n,
                                          fb_complex_float_t *a, int *lda,
                                          float *anorm, float *rcond,
                                          fb_complex_float_t *work,
                                          float *rwork, int *info);

static struct {
    int calls;
    char norm;
    int n;
    int lda;
    float anorm;
    const float *a;
} g_sgecon_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    char norm;
    int n;
    int lda;
    const float *a;
    float anorm;
    float *rcond;
} g_sgecon_cblas_call;

static struct {
    int calls;
    char norm;
    int n;
    int lda;
    float anorm;
    const fb_complex_float_t *a;
} g_cgecon_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    char norm;
    int n;
    int lda;
    const fb_complex_float_t *a;
    float anorm;
    float *rcond;
} g_cgecon_cblas_call;

static int g_sgecon_cblas_rc = 0;
static int g_cgecon_cblas_rc = 0;

static fb_complex_float_t make_cfloat(float real_value)
{
    fb_complex_float_t value = (fb_complex_float_t)0;
    memcpy(&value, &real_value, sizeof(real_value));
    return value;
}

static void stub_sgecon_fortran(char *norm, int *n, const float *a, int *lda,
                                float *anorm, float *rcond, float *work,
                                int *iwork, int *info)
{
    (void)work;
    (void)iwork;
    g_sgecon_fortran_call.calls += 1;
    g_sgecon_fortran_call.norm = *norm;
    g_sgecon_fortran_call.n = *n;
    g_sgecon_fortran_call.lda = *lda;
    g_sgecon_fortran_call.anorm = *anorm;
    g_sgecon_fortran_call.a = a;
    *rcond = 0.25f;
    *info = 0;
}

static int stub_sgecon_cblas(fb_layout_t layout, char norm, int n,
                             const float *a, int lda, float anorm,
                             float *rcond)
{
    g_sgecon_cblas_call.called += 1;
    g_sgecon_cblas_call.layout = layout;
    g_sgecon_cblas_call.norm = norm;
    g_sgecon_cblas_call.n = n;
    g_sgecon_cblas_call.lda = lda;
    g_sgecon_cblas_call.a = a;
    g_sgecon_cblas_call.anorm = anorm;
    g_sgecon_cblas_call.rcond = rcond;
    return g_sgecon_cblas_rc;
}

static void stub_cgecon_fortran(char *norm, int *n, const fb_complex_float_t *a,
                                int *lda, float *anorm, float *rcond,
                                fb_complex_float_t *work, float *rwork,
                                int *info)
{
    (void)work;
    (void)rwork;
    g_cgecon_fortran_call.calls += 1;
    g_cgecon_fortran_call.norm = *norm;
    g_cgecon_fortran_call.n = *n;
    g_cgecon_fortran_call.lda = *lda;
    g_cgecon_fortran_call.anorm = *anorm;
    g_cgecon_fortran_call.a = a;
    *rcond = 0.5f;
    *info = 0;
}

static int stub_cgecon_cblas(fb_layout_t layout, char norm, int n,
                             const fb_complex_float_t *a, int lda, float anorm,
                             float *rcond)
{
    g_cgecon_cblas_call.called += 1;
    g_cgecon_cblas_call.layout = layout;
    g_cgecon_cblas_call.norm = norm;
    g_cgecon_cblas_call.n = n;
    g_cgecon_cblas_call.lda = lda;
    g_cgecon_cblas_call.a = a;
    g_cgecon_cblas_call.anorm = anorm;
    g_cgecon_cblas_call.rcond = rcond;
    return g_cgecon_cblas_rc;
}

static int check_sgecon_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_sgecon_fn thunk = NULL;
    float a[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    float rcond = 0.0f;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sgecon_fortran_call, 0, sizeof(g_sgecon_fortran_call));

    vtable.ext_ops[FB_OP_SGECON][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_sgecon_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SGECON);

    thunk = (fb_sgecon_fn)vtable.ext_ops[FB_OP_SGECON][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SGECON Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, '1', 2, a, 2, 7.5f, &rcond);
    if (info != 0 || g_sgecon_fortran_call.calls != 1 ||
        g_sgecon_fortran_call.norm != '1' || g_sgecon_fortran_call.n != 2 ||
        g_sgecon_fortran_call.lda != 2 || g_sgecon_fortran_call.a != a ||
        g_sgecon_fortran_call.anorm != 7.5f || rcond != 0.25f) {
        fprintf(stderr, "[FAIL] SGECON Fortran->CBLAS thunk did not preserve condition-estimation inputs\n");
        return 1;
    }

    printf("[PASS] SGECON Fortran->CBLAS thunk forwards norm/anorm and returns the estimated reciprocal condition number\n");
    return 0;
}

static int check_sgecon_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_sgecon_fortran_slot_fn thunk = NULL;
    float a[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    float anorm = 7.5f;
    float rcond = 0.0f;
    float work[8] = { 0.0f };
    int iwork[2] = { 0, 0 };
    char norm = 'I';
    int n = 2;
    int lda = 2;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_sgecon_cblas_call, 0, sizeof(g_sgecon_cblas_call));
    g_sgecon_cblas_rc = 181;

    vtable.ext_ops[FB_OP_SGECON][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_sgecon_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SGECON);

    thunk = (fb_sgecon_fortran_slot_fn)vtable.ext_ops[FB_OP_SGECON][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SGECON CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&norm, &n, a, &lda, &anorm, &rcond, work, iwork, &info);
    if (info != 181 || g_sgecon_cblas_call.called != 1 ||
        g_sgecon_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_sgecon_cblas_call.norm != 'I' || g_sgecon_cblas_call.n != 2 ||
        g_sgecon_cblas_call.lda != 2 || g_sgecon_cblas_call.a != a ||
        g_sgecon_cblas_call.anorm != 7.5f || g_sgecon_cblas_call.rcond != &rcond) {
        fprintf(stderr, "[FAIL] SGECON CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] SGECON CBLAS->Fortran thunk maps the all-pointer ABI into the generic C condition-estimation entry\n");
    return 0;
}

static int check_cgecon_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_cgecon_fn thunk = NULL;
    fb_complex_float_t a[4] = {
        make_cfloat(1.0f), make_cfloat(2.0f),
        make_cfloat(3.0f), make_cfloat(4.0f)
    };
    float rcond = 0.0f;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cgecon_fortran_call, 0, sizeof(g_cgecon_fortran_call));

    vtable.ext_ops[FB_OP_CGECON][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_cgecon_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CGECON);

    thunk = (fb_cgecon_fn)vtable.ext_ops[FB_OP_CGECON][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CGECON Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_COL_MAJOR, '1', 2, a, 2, 8.5f, &rcond);
    if (info != 0 || g_cgecon_fortran_call.calls != 1 ||
        g_cgecon_fortran_call.norm != '1' || g_cgecon_fortran_call.n != 2 ||
        g_cgecon_fortran_call.lda != 2 || g_cgecon_fortran_call.a != a ||
        g_cgecon_fortran_call.anorm != 8.5f || rcond != 0.5f) {
        fprintf(stderr, "[FAIL] CGECON Fortran->CBLAS thunk did not preserve complex condition-estimation inputs\n");
        return 1;
    }

    printf("[PASS] CGECON Fortran->CBLAS thunk forwards complex norm/anorm and returns the estimated reciprocal condition number\n");
    return 0;
}

static int check_cgecon_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_cgecon_fortran_slot_fn thunk = NULL;
    fb_complex_float_t a[4] = {
        make_cfloat(1.0f), make_cfloat(2.0f),
        make_cfloat(3.0f), make_cfloat(4.0f)
    };
    float anorm = 8.5f;
    float rcond = 0.0f;
    fb_complex_float_t work[4] = { 0 };
    float rwork[4] = { 0.0f };
    char norm = 'O';
    int n = 2;
    int lda = 2;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cgecon_cblas_call, 0, sizeof(g_cgecon_cblas_call));
    g_cgecon_cblas_rc = 183;

    vtable.ext_ops[FB_OP_CGECON][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_cgecon_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CGECON);

    thunk = (fb_cgecon_fortran_slot_fn)vtable.ext_ops[FB_OP_CGECON][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CGECON CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&norm, &n, a, &lda, &anorm, &rcond, work, rwork, &info);
    if (info != 183 || g_cgecon_cblas_call.called != 1 ||
        g_cgecon_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_cgecon_cblas_call.norm != 'O' || g_cgecon_cblas_call.n != 2 ||
        g_cgecon_cblas_call.lda != 2 || g_cgecon_cblas_call.a != a ||
        g_cgecon_cblas_call.anorm != 8.5f || g_cgecon_cblas_call.rcond != &rcond) {
        fprintf(stderr, "[FAIL] CGECON CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] CGECON CBLAS->Fortran thunk maps the all-pointer ABI into the generic complex condition-estimation entry\n");
    return 0;
}

int main(void)
{
    if (check_sgecon_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_sgecon_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_cgecon_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_cgecon_cblas_to_fortran() != 0) {
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}