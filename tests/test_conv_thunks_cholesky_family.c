#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

typedef void (*fb_spotrf_fortran_slot_fn)(char *uplo, int *n, float *a,
                                          int *lda, int *info);
typedef void (*fb_cpotri_fortran_slot_fn)(char *uplo, int *n,
                                          fb_complex_float_t *a, int *lda,
                                          int *info);
typedef void (*fb_spotrs_fortran_slot_fn)(char *uplo, int *n, int *nrhs,
                                          float *a, int *lda, float *b,
                                          int *ldb, int *info);
typedef void (*fb_cposv_fortran_slot_fn)(char *uplo, int *n, int *nrhs,
                                         fb_complex_float_t *a, int *lda,
                                         fb_complex_float_t *b, int *ldb,
                                         int *info);

static struct {
    int called;
    char uplo;
    int n;
    int lda;
    float *a;
} g_spotrf_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    fb_uplo_t uplo;
    int n;
    int lda;
    float *a;
} g_spotrf_cblas_call;

static struct {
    int called;
    char uplo;
    int n;
    int lda;
    fb_complex_float_t *a;
} g_cpotri_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    fb_uplo_t uplo;
    int n;
    int lda;
    fb_complex_float_t *a;
} g_cpotri_cblas_call;

static struct {
    int called;
    char uplo;
    int n;
    int nrhs;
    int lda;
    int ldb;
    const float *a;
    float b_snapshot[6];
} g_spotrs_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    fb_uplo_t uplo;
    int n;
    int nrhs;
    int lda;
    int ldb;
    const float *a;
    float *b;
} g_spotrs_cblas_call;

static struct {
    int called;
    char uplo;
    int n;
    int nrhs;
    int lda;
    int ldb;
    const fb_complex_float_t *a;
    fb_complex_float_t b_snapshot[6];
} g_cposv_fortran_call;

static struct {
    int called;
    fb_layout_t layout;
    fb_uplo_t uplo;
    int n;
    int nrhs;
    int lda;
    int ldb;
    const fb_complex_float_t *a;
    fb_complex_float_t *b;
} g_cposv_cblas_call;

static void stub_spotrf_fortran(char *uplo, int *n, float *a, int *lda,
                                int *info)
{
    g_spotrf_fortran_call.called += 1;
    g_spotrf_fortran_call.uplo = *uplo;
    g_spotrf_fortran_call.n = *n;
    g_spotrf_fortran_call.lda = *lda;
    g_spotrf_fortran_call.a = a;
    *info = 17;
}

static int stub_spotrf_cblas(const fb_layout_t layout, const fb_uplo_t uplo,
                             const int n, float *a, const int lda)
{
    g_spotrf_cblas_call.called += 1;
    g_spotrf_cblas_call.layout = layout;
    g_spotrf_cblas_call.uplo = uplo;
    g_spotrf_cblas_call.n = n;
    g_spotrf_cblas_call.lda = lda;
    g_spotrf_cblas_call.a = a;
    return 19;
}

static void stub_cpotri_fortran(char *uplo, int *n, fb_complex_float_t *a,
                                int *lda, int *info)
{
    g_cpotri_fortran_call.called += 1;
    g_cpotri_fortran_call.uplo = *uplo;
    g_cpotri_fortran_call.n = *n;
    g_cpotri_fortran_call.lda = *lda;
    g_cpotri_fortran_call.a = a;
    *info = 23;
}

static int stub_cpotri_cblas(const fb_layout_t layout, const fb_uplo_t uplo,
                             const int n, fb_complex_float_t *a,
                             const int lda)
{
    g_cpotri_cblas_call.called += 1;
    g_cpotri_cblas_call.layout = layout;
    g_cpotri_cblas_call.uplo = uplo;
    g_cpotri_cblas_call.n = n;
    g_cpotri_cblas_call.lda = lda;
    g_cpotri_cblas_call.a = a;
    return 29;
}

static void stub_spotrs_fortran(char *uplo, int *n, int *nrhs, float *a,
                                int *lda, float *b, int *ldb, int *info)
{
    int col = 0;
    int row = 0;

    g_spotrs_fortran_call.called += 1;
    g_spotrs_fortran_call.uplo = *uplo;
    g_spotrs_fortran_call.n = *n;
    g_spotrs_fortran_call.nrhs = *nrhs;
    g_spotrs_fortran_call.lda = *lda;
    g_spotrs_fortran_call.ldb = *ldb;
    g_spotrs_fortran_call.a = a;

    for (col = 0; col < *nrhs; ++col) {
        for (row = 0; row < *n; ++row) {
            g_spotrs_fortran_call.b_snapshot[(col * (*n)) + row] =
                b[(col * (*ldb)) + row];
            b[(col * (*ldb)) + row] = (float)(100 + (10 * row) + col);
        }
    }

    *info = 0;
}

static int stub_spotrs_cblas(const fb_layout_t layout, const fb_uplo_t uplo,
                             const int n, const int nrhs, const float *a,
                             const int lda, float *b, const int ldb)
{
    g_spotrs_cblas_call.called += 1;
    g_spotrs_cblas_call.layout = layout;
    g_spotrs_cblas_call.uplo = uplo;
    g_spotrs_cblas_call.n = n;
    g_spotrs_cblas_call.nrhs = nrhs;
    g_spotrs_cblas_call.lda = lda;
    g_spotrs_cblas_call.ldb = ldb;
    g_spotrs_cblas_call.a = a;
    g_spotrs_cblas_call.b = b;
    return 31;
}

static void stub_cposv_fortran(char *uplo, int *n, int *nrhs,
                               fb_complex_float_t *a, int *lda,
                               fb_complex_float_t *b, int *ldb, int *info)
{
    int col = 0;
    int row = 0;
    float real_part = 0.0f;

    g_cposv_fortran_call.called += 1;
    g_cposv_fortran_call.uplo = *uplo;
    g_cposv_fortran_call.n = *n;
    g_cposv_fortran_call.nrhs = *nrhs;
    g_cposv_fortran_call.lda = *lda;
    g_cposv_fortran_call.ldb = *ldb;
    g_cposv_fortran_call.a = a;

    for (col = 0; col < *nrhs; ++col) {
        for (row = 0; row < *n; ++row) {
            g_cposv_fortran_call.b_snapshot[(col * (*n)) + row] =
                b[(col * (*ldb)) + row];
            real_part = (float)(200 + (10 * row) + col);
            memcpy(&b[(col * (*ldb)) + row], &real_part, sizeof(real_part));
        }
    }

    *info = 0;
}

static int stub_cposv_cblas(const fb_layout_t layout, const fb_uplo_t uplo,
                            const int n, const int nrhs,
                            fb_complex_float_t *a, const int lda,
                            fb_complex_float_t *b, const int ldb)
{
    g_cposv_cblas_call.called += 1;
    g_cposv_cblas_call.layout = layout;
    g_cposv_cblas_call.uplo = uplo;
    g_cposv_cblas_call.n = n;
    g_cposv_cblas_call.nrhs = nrhs;
    g_cposv_cblas_call.lda = lda;
    g_cposv_cblas_call.ldb = ldb;
    g_cposv_cblas_call.a = a;
    g_cposv_cblas_call.b = b;
    return 37;
}

static int check_spotrf_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_spotrf_fn thunk = NULL;
    float a[16] = { 0.0f };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_spotrf_fortran_call, 0, sizeof(g_spotrf_fortran_call));

    vtable.ext_ops[FB_OP_SPOTRF][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_spotrf_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SPOTRF);

    thunk = (fb_spotrf_fn)vtable.ext_ops[FB_OP_SPOTRF][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SPOTRF Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, FB_UPPER, 4, a, 4);
    if (info != 17 || g_spotrf_fortran_call.called != 1 ||
        g_spotrf_fortran_call.uplo != 'L' || g_spotrf_fortran_call.n != 4 ||
        g_spotrf_fortran_call.lda != 4 || g_spotrf_fortran_call.a != a) {
        fprintf(stderr, "[FAIL] SPOTRF Fortran->CBLAS thunk did not remap UPLO correctly\n");
        return 1;
    }

    printf("[PASS] SPOTRF Fortran->CBLAS thunk remaps symmetric UPLO for row-major\n");
    return 0;
}

static int check_spotrf_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_spotrf_fortran_slot_fn thunk = NULL;
    float a[16] = { 0.0f };
    int n = 4;
    int lda = 4;
    int info = 0;
    char uplo = 'u';

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_spotrf_cblas_call, 0, sizeof(g_spotrf_cblas_call));

    vtable.ext_ops[FB_OP_SPOTRF][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_spotrf_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SPOTRF);

    thunk = (fb_spotrf_fortran_slot_fn)vtable.ext_ops[FB_OP_SPOTRF][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SPOTRF CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&uplo, &n, a, &lda, &info);
    if (info != 19 || g_spotrf_cblas_call.called != 1 ||
        g_spotrf_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_spotrf_cblas_call.uplo != FB_UPPER || g_spotrf_cblas_call.n != 4 ||
        g_spotrf_cblas_call.lda != 4 || g_spotrf_cblas_call.a != a) {
        fprintf(stderr, "[FAIL] SPOTRF CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] SPOTRF CBLAS->Fortran thunk maps UPLO into the C ABI\n");
    return 0;
}

static int check_cpotri_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_cpotri_fn thunk = NULL;
    fb_complex_float_t a[16];
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cpotri_fortran_call, 0, sizeof(g_cpotri_fortran_call));
    memset(a, 0, sizeof(a));

    vtable.ext_ops[FB_OP_CPOTRI][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_cpotri_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CPOTRI);

    thunk = (fb_cpotri_fn)vtable.ext_ops[FB_OP_CPOTRI][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CPOTRI Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, FB_LOWER, 4, a, 4);
    if (info != 23 || g_cpotri_fortran_call.called != 1 ||
        g_cpotri_fortran_call.uplo != 'U' || g_cpotri_fortran_call.n != 4 ||
        g_cpotri_fortran_call.lda != 4 || g_cpotri_fortran_call.a != a) {
        fprintf(stderr, "[FAIL] CPOTRI Fortran->CBLAS thunk did not remap Hermitian UPLO correctly\n");
        return 1;
    }

    printf("[PASS] CPOTRI Fortran->CBLAS thunk remaps Hermitian UPLO for row-major\n");
    return 0;
}

static int check_cpotri_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_cpotri_fortran_slot_fn thunk = NULL;
    fb_complex_float_t a[16];
    int n = 4;
    int lda = 4;
    int info = 0;
    char uplo = 'L';

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cpotri_cblas_call, 0, sizeof(g_cpotri_cblas_call));
    memset(a, 0, sizeof(a));

    vtable.ext_ops[FB_OP_CPOTRI][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_cpotri_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CPOTRI);

    thunk = (fb_cpotri_fortran_slot_fn)vtable.ext_ops[FB_OP_CPOTRI][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CPOTRI CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&uplo, &n, a, &lda, &info);
    if (info != 29 || g_cpotri_cblas_call.called != 1 ||
        g_cpotri_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_cpotri_cblas_call.uplo != FB_LOWER || g_cpotri_cblas_call.n != 4 ||
        g_cpotri_cblas_call.lda != 4 || g_cpotri_cblas_call.a != a) {
        fprintf(stderr, "[FAIL] CPOTRI CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] CPOTRI CBLAS->Fortran thunk maps Hermitian UPLO into the C ABI\n");
    return 0;
}

static int check_spotrs_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_spotrs_fn thunk = NULL;
    float a[9] = { 0.0f };
    float b[6] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_spotrs_fortran_call, 0, sizeof(g_spotrs_fortran_call));

    vtable.ext_ops[FB_OP_SPOTRS][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_spotrs_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_SPOTRS);

    thunk = (fb_spotrs_fn)vtable.ext_ops[FB_OP_SPOTRS][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SPOTRS Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, FB_UPPER, 3, 2, a, 3, b, 2);
    if (info != 0 || g_spotrs_fortran_call.called != 1 ||
        g_spotrs_fortran_call.uplo != 'L' || g_spotrs_fortran_call.n != 3 ||
        g_spotrs_fortran_call.nrhs != 2 || g_spotrs_fortran_call.ldb != 3 ||
        g_spotrs_fortran_call.b_snapshot[0] != 1.0f ||
        g_spotrs_fortran_call.b_snapshot[1] != 3.0f ||
        g_spotrs_fortran_call.b_snapshot[2] != 5.0f ||
        g_spotrs_fortran_call.b_snapshot[3] != 2.0f ||
        g_spotrs_fortran_call.b_snapshot[4] != 4.0f ||
        g_spotrs_fortran_call.b_snapshot[5] != 6.0f ||
        b[0] != 100.0f || b[1] != 101.0f || b[2] != 110.0f ||
        b[3] != 111.0f || b[4] != 120.0f || b[5] != 121.0f) {
        fprintf(stderr, "[FAIL] SPOTRS Fortran->CBLAS thunk did not translate RHS correctly\n");
        return 1;
    }

    printf("[PASS] SPOTRS Fortran->CBLAS thunk remaps UPLO and transposes row-major RHS\n");
    return 0;
}

static int check_spotrs_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_spotrs_fortran_slot_fn thunk = NULL;
    float a[9] = { 0.0f };
    float b[6] = { 0.0f };
    int n = 3;
    int nrhs = 2;
    int lda = 3;
    int ldb = 3;
    int info = 0;
    char uplo = 'l';

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_spotrs_cblas_call, 0, sizeof(g_spotrs_cblas_call));

    vtable.ext_ops[FB_OP_SPOTRS][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_spotrs_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_SPOTRS);

    thunk = (fb_spotrs_fortran_slot_fn)vtable.ext_ops[FB_OP_SPOTRS][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] SPOTRS CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&uplo, &n, &nrhs, a, &lda, b, &ldb, &info);
    if (info != 31 || g_spotrs_cblas_call.called != 1 ||
        g_spotrs_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_spotrs_cblas_call.uplo != FB_LOWER || g_spotrs_cblas_call.n != 3 ||
        g_spotrs_cblas_call.nrhs != 2 || g_spotrs_cblas_call.lda != 3 ||
        g_spotrs_cblas_call.ldb != 3 || g_spotrs_cblas_call.a != a ||
        g_spotrs_cblas_call.b != b) {
        fprintf(stderr, "[FAIL] SPOTRS CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] SPOTRS CBLAS->Fortran thunk maps UPLO into the C solve ABI\n");
    return 0;
}

static int check_cposv_fortran_to_cblas(void)
{
    fb_backend_vtable_t vtable;
    fb_cposv_fn thunk = NULL;
    fb_complex_float_t a[9];
    fb_complex_float_t b[6];
    float real_parts[6] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
    float actual = 0.0f;
    int index = 0;
    int info = 0;

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cposv_fortran_call, 0, sizeof(g_cposv_fortran_call));
    memset(a, 0, sizeof(a));
    memset(b, 0, sizeof(b));
    for (index = 0; index < 6; ++index) {
        memcpy(&b[index], &real_parts[index], sizeof(real_parts[index]));
    }

    vtable.ext_ops[FB_OP_CPOSV][FB_CONV_FORTRAN] =
        (fb_generic_fn)(void (*)(void))stub_cposv_fortran;
    fb_install_conv_thunks(&vtable, FB_OP_CPOSV);

    thunk = (fb_cposv_fn)vtable.ext_ops[FB_OP_CPOSV][FB_CONV_CBLAS];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CPOSV Fortran->CBLAS thunk was not installed\n");
        return 1;
    }

    info = thunk(FB_LAYOUT_ROW_MAJOR, FB_LOWER, 3, 2, a, 3, b, 2);
    memcpy(&actual, &g_cposv_fortran_call.b_snapshot[0], sizeof(actual));
    if (info != 0 || g_cposv_fortran_call.called != 1 ||
        g_cposv_fortran_call.uplo != 'U' || g_cposv_fortran_call.n != 3 ||
        g_cposv_fortran_call.nrhs != 2 || g_cposv_fortran_call.ldb != 3 ||
        actual != 1.0f) {
        fprintf(stderr, "[FAIL] CPOSV Fortran->CBLAS thunk did not translate complex RHS correctly\n");
        return 1;
    }

    memcpy(&actual, &b[0], sizeof(actual));
    if (actual != 200.0f) {
        fprintf(stderr, "[FAIL] CPOSV Fortran->CBLAS thunk did not copy complex results back to row-major storage\n");
        return 1;
    }

    printf("[PASS] CPOSV Fortran->CBLAS thunk remaps UPLO and transposes complex row-major RHS\n");
    return 0;
}

static int check_cposv_cblas_to_fortran(void)
{
    fb_backend_vtable_t vtable;
    fb_cposv_fortran_slot_fn thunk = NULL;
    fb_complex_float_t a[9];
    fb_complex_float_t b[6];
    int n = 3;
    int nrhs = 2;
    int lda = 3;
    int ldb = 3;
    int info = 0;
    char uplo = 'U';

    memset(&vtable, 0, sizeof(vtable));
    memset(&g_cposv_cblas_call, 0, sizeof(g_cposv_cblas_call));
    memset(a, 0, sizeof(a));
    memset(b, 0, sizeof(b));

    vtable.ext_ops[FB_OP_CPOSV][FB_CONV_CBLAS] =
        (fb_generic_fn)(void (*)(void))stub_cposv_cblas;
    fb_install_conv_thunks(&vtable, FB_OP_CPOSV);

    thunk = (fb_cposv_fortran_slot_fn)vtable.ext_ops[FB_OP_CPOSV][FB_CONV_FORTRAN];
    if (!thunk) {
        fprintf(stderr, "[FAIL] CPOSV CBLAS->Fortran thunk was not installed\n");
        return 1;
    }

    thunk(&uplo, &n, &nrhs, a, &lda, b, &ldb, &info);
    if (info != 37 || g_cposv_cblas_call.called != 1 ||
        g_cposv_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||
        g_cposv_cblas_call.uplo != FB_UPPER || g_cposv_cblas_call.n != 3 ||
        g_cposv_cblas_call.nrhs != 2 || g_cposv_cblas_call.lda != 3 ||
        g_cposv_cblas_call.ldb != 3 || g_cposv_cblas_call.a != a ||
        g_cposv_cblas_call.b != b) {
        fprintf(stderr, "[FAIL] CPOSV CBLAS->Fortran thunk delegated incorrectly\n");
        return 1;
    }

    printf("[PASS] CPOSV CBLAS->Fortran thunk maps UPLO into the complex solve ABI\n");
    return 0;
}

int main(void)
{
    if (check_spotrf_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_spotrf_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_cpotri_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_cpotri_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_spotrs_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_spotrs_cblas_to_fortran() != 0) {
        return 1;
    }
    if (check_cposv_fortran_to_cblas() != 0) {
        return 1;
    }
    if (check_cposv_cblas_to_fortran() != 0) {
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}