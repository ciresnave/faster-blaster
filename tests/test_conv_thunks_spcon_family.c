#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

#define DEFINE_SPCON_REAL_TESTS(SUFFIX, TYPE, OP_ID, BASE)                      \
typedef int (*fb_##SUFFIX##_cblas_fn)(fb_layout_t layout, fb_uplo_t uplo,      \
                                      int n, const TYPE *ap, const int *ipiv,   \
                                      TYPE anorm, TYPE *rcond);                 \
typedef void (*fb_##SUFFIX##_fortran_fn)(char *uplo, int *n, const TYPE *ap,    \
                                         const int *ipiv, const TYPE *anorm,    \
                                         TYPE *rcond, TYPE *work, int *iwork,   \
                                         int *info);                            \
static struct {                                                                   \
    int called;                                                                   \
    char uplo;                                                                    \
    int n;                                                                        \
    TYPE ap_vals[6];                                                              \
    int ipiv_vals[3];                                                             \
    TYPE anorm;                                                                   \
    TYPE *rcond;                                                                  \
    int work_nonnull;                                                             \
    int iwork_nonnull;                                                            \
} g_##SUFFIX##_fortran_call;                                                      \
static struct {                                                                   \
    int called;                                                                   \
    fb_layout_t layout;                                                           \
    fb_uplo_t uplo;                                                               \
    int n;                                                                        \
    const TYPE *ap;                                                               \
    const int *ipiv;                                                              \
    TYPE anorm;                                                                   \
    TYPE *rcond;                                                                  \
} g_##SUFFIX##_cblas_call;                                                        \
static void stub_##SUFFIX##_fortran(char *uplo, int *n, const TYPE *ap,         \
                                    const int *ipiv, const TYPE *anorm,         \
                                    TYPE *rcond, TYPE *work, int *iwork,        \
                                    int *info)                                  \
{                                                                                 \
    size_t elems = (size_t)(*n * (*n + 1)) / 2U;                                 \
    g_##SUFFIX##_fortran_call.called += 1;                                        \
    g_##SUFFIX##_fortran_call.uplo = *uplo;                                       \
    g_##SUFFIX##_fortran_call.n = *n;                                             \
    memset(g_##SUFFIX##_fortran_call.ap_vals, 0, sizeof(g_##SUFFIX##_fortran_call.ap_vals)); \
    if (elems > 0) {                                                              \
        memcpy(g_##SUFFIX##_fortran_call.ap_vals, ap, elems * sizeof(TYPE));      \
    }                                                                             \
    memset(g_##SUFFIX##_fortran_call.ipiv_vals, 0, sizeof(g_##SUFFIX##_fortran_call.ipiv_vals)); \
    if (*n > 0) {                                                                 \
        memcpy(g_##SUFFIX##_fortran_call.ipiv_vals, ipiv, (size_t)(*n) * sizeof(int)); \
    }                                                                             \
    g_##SUFFIX##_fortran_call.anorm = *anorm;                                     \
    g_##SUFFIX##_fortran_call.rcond = rcond;                                      \
    g_##SUFFIX##_fortran_call.work_nonnull = (work != NULL);                      \
    g_##SUFFIX##_fortran_call.iwork_nonnull = (iwork != NULL);                    \
    *rcond = (TYPE)((BASE) + 1);                                                  \
    *info = (BASE) + 2;                                                           \
}                                                                                 \
static int stub_##SUFFIX##_cblas(fb_layout_t layout, fb_uplo_t uplo, int n,     \
                                 const TYPE *ap, const int *ipiv, TYPE anorm,   \
                                 TYPE *rcond)                                   \
{                                                                                 \
    g_##SUFFIX##_cblas_call.called += 1;                                          \
    g_##SUFFIX##_cblas_call.layout = layout;                                      \
    g_##SUFFIX##_cblas_call.uplo = uplo;                                          \
    g_##SUFFIX##_cblas_call.n = n;                                                \
    g_##SUFFIX##_cblas_call.ap = ap;                                              \
    g_##SUFFIX##_cblas_call.ipiv = ipiv;                                          \
    g_##SUFFIX##_cblas_call.anorm = anorm;                                        \
    g_##SUFFIX##_cblas_call.rcond = rcond;                                        \
    *rcond = (TYPE)((BASE) + 3);                                                  \
    return (BASE) + 4;                                                            \
}                                                                                 \
static int check_##SUFFIX##_fortran_to_cblas(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                          \
    TYPE ap_row[6] = { (TYPE)11, (TYPE)12, (TYPE)13, (TYPE)22, (TYPE)23, (TYPE)33 }; \
    TYPE expected_col[6] = { (TYPE)11, (TYPE)12, (TYPE)22, (TYPE)13, (TYPE)23, (TYPE)33 }; \
    int ipiv[3] = { 1, -2, 3 };                                                   \
    TYPE rcond = (TYPE)0;                                                         \
    int info = 0;                                                                 \
    memset(&vtable, 0, sizeof(vtable));                                           \
    memset(&g_##SUFFIX##_fortran_call, 0, sizeof(g_##SUFFIX##_fortran_call));     \
    vtable.ext_ops[OP_ID][FB_CONV_FORTRAN] =                                      \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_fortran;                   \
    fb_install_conv_thunks(&vtable, OP_ID);                                       \
    thunk = (fb_##SUFFIX##_cblas_fn)vtable.ext_ops[OP_ID][FB_CONV_CBLAS];         \
    if (!thunk) {                                                                 \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk was not installed\n"); \
        return 1;                                                                 \
    }                                                                             \
    info = thunk(FB_LAYOUT_ROW_MAJOR, FB_UPPER, 3, ap_row, ipiv, (TYPE)8, &rcond); \
    if (info != (BASE) + 2 ||                                                     \
        g_##SUFFIX##_fortran_call.called != 1 ||                                  \
        g_##SUFFIX##_fortran_call.uplo != 'U' ||                                  \
        g_##SUFFIX##_fortran_call.n != 3 ||                                       \
        memcmp(g_##SUFFIX##_fortran_call.ap_vals, expected_col, sizeof(expected_col)) != 0 || \
        memcmp(g_##SUFFIX##_fortran_call.ipiv_vals, ipiv, sizeof(ipiv)) != 0 ||   \
        g_##SUFFIX##_fortran_call.anorm != (TYPE)8 ||                             \
        g_##SUFFIX##_fortran_call.rcond != &rcond ||                              \
        !g_##SUFFIX##_fortran_call.work_nonnull ||                                \
        !g_##SUFFIX##_fortran_call.iwork_nonnull ||                               \
        rcond != (TYPE)((BASE) + 1)) {                                            \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not route packed SPCON inputs correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk routes packed SPCON inputs\n"); \
    return 0;                                                                     \
}                                                                                 \
static int check_##SUFFIX##_cblas_to_fortran(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                        \
    char uplo = 'L';                                                              \
    int n = 2;                                                                    \
    TYPE ap[3] = { (TYPE)7, (TYPE)8, (TYPE)9 };                                  \
    int ipiv[2] = { 1, -2 };                                                      \
    TYPE anorm = (TYPE)5;                                                         \
    TYPE rcond = (TYPE)0;                                                         \
    TYPE work[6] = { 0 };                                                         \
    int iwork[2] = { 0, 0 };                                                      \
    int info = -999;                                                              \
    memset(&vtable, 0, sizeof(vtable));                                           \
    memset(&g_##SUFFIX##_cblas_call, 0, sizeof(g_##SUFFIX##_cblas_call));         \
    vtable.ext_ops[OP_ID][FB_CONV_CBLAS] =                                        \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_cblas;                     \
    fb_install_conv_thunks(&vtable, OP_ID);                                       \
    thunk = (fb_##SUFFIX##_fortran_fn)vtable.ext_ops[OP_ID][FB_CONV_FORTRAN];     \
    if (!thunk) {                                                                 \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk was not installed\n"); \
        return 1;                                                                 \
    }                                                                             \
    thunk(&uplo, &n, ap, ipiv, &anorm, &rcond, work, iwork, &info);              \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                    \
        g_##SUFFIX##_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||                  \
        g_##SUFFIX##_cblas_call.uplo != FB_LOWER ||                               \
        g_##SUFFIX##_cblas_call.n != 2 ||                                         \
        g_##SUFFIX##_cblas_call.ap != ap ||                                       \
        g_##SUFFIX##_cblas_call.ipiv != ipiv ||                                   \
        g_##SUFFIX##_cblas_call.anorm != (TYPE)5 ||                               \
        g_##SUFFIX##_cblas_call.rcond != &rcond ||                                \
        info != (BASE) + 4 ||                                                     \
        rcond != (TYPE)((BASE) + 3)) {                                            \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not propagate SPCON info correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk propagates SPCON info\n"); \
    return 0;                                                                     \
}

#define DEFINE_SPCON_COMPLEX_TESTS(SUFFIX, CTYPE, RTYPE, OP_ID, BASE)            \
typedef int (*fb_##SUFFIX##_cblas_fn)(fb_layout_t layout, fb_uplo_t uplo,      \
                                      int n, const CTYPE *ap, const int *ipiv,  \
                                      RTYPE anorm, RTYPE *rcond);               \
typedef void (*fb_##SUFFIX##_fortran_fn)(char *uplo, int *n, CTYPE *ap,        \
                                         int *ipiv, RTYPE *anorm, RTYPE *rcond, \
                                         CTYPE *work, int *info);               \
static struct {                                                                   \
    int called;                                                                   \
    char uplo;                                                                    \
    int n;                                                                        \
    CTYPE *ap;                                                                    \
    int ipiv_vals[3];                                                             \
    RTYPE anorm;                                                                  \
    RTYPE *rcond;                                                                 \
    int work_nonnull;                                                             \
} g_##SUFFIX##_fortran_call;                                                      \
static struct {                                                                   \
    int called;                                                                   \
    fb_layout_t layout;                                                           \
    fb_uplo_t uplo;                                                               \
    int n;                                                                        \
    const CTYPE *ap;                                                              \
    const int *ipiv;                                                              \
    RTYPE anorm;                                                                  \
    RTYPE *rcond;                                                                 \
} g_##SUFFIX##_cblas_call;                                                        \
static void stub_##SUFFIX##_fortran(char *uplo, int *n, CTYPE *ap,             \
                                    int *ipiv, RTYPE *anorm, RTYPE *rcond,     \
                                    CTYPE *work, int *info)                    \
{                                                                                 \
    g_##SUFFIX##_fortran_call.called += 1;                                        \
    g_##SUFFIX##_fortran_call.uplo = *uplo;                                       \
    g_##SUFFIX##_fortran_call.n = *n;                                             \
    g_##SUFFIX##_fortran_call.ap = ap;                                            \
    memset(g_##SUFFIX##_fortran_call.ipiv_vals, 0, sizeof(g_##SUFFIX##_fortran_call.ipiv_vals)); \
    if (*n > 0) {                                                                 \
        memcpy(g_##SUFFIX##_fortran_call.ipiv_vals, ipiv, (size_t)(*n) * sizeof(int)); \
    }                                                                             \
    g_##SUFFIX##_fortran_call.anorm = *anorm;                                     \
    g_##SUFFIX##_fortran_call.rcond = rcond;                                      \
    g_##SUFFIX##_fortran_call.work_nonnull = (work != NULL);                      \
    *rcond = (RTYPE)((BASE) + 1);                                                 \
    *info = (BASE) + 2;                                                           \
}                                                                                 \
static int stub_##SUFFIX##_cblas(fb_layout_t layout, fb_uplo_t uplo, int n,     \
                                 const CTYPE *ap, const int *ipiv, RTYPE anorm, \
                                 RTYPE *rcond)                                  \
{                                                                                 \
    g_##SUFFIX##_cblas_call.called += 1;                                          \
    g_##SUFFIX##_cblas_call.layout = layout;                                      \
    g_##SUFFIX##_cblas_call.uplo = uplo;                                          \
    g_##SUFFIX##_cblas_call.n = n;                                                \
    g_##SUFFIX##_cblas_call.ap = ap;                                              \
    g_##SUFFIX##_cblas_call.ipiv = ipiv;                                          \
    g_##SUFFIX##_cblas_call.anorm = anorm;                                        \
    g_##SUFFIX##_cblas_call.rcond = rcond;                                        \
    *rcond = (RTYPE)((BASE) + 3);                                                 \
    return (BASE) + 4;                                                            \
}                                                                                 \
static int check_##SUFFIX##_fortran_to_cblas(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                          \
    CTYPE ap[3] = { 0 };                                                          \
    int ipiv[2] = { 1, -2 };                                                      \
    RTYPE rcond = (RTYPE)0;                                                       \
    int info = 0;                                                                 \
    memset(&vtable, 0, sizeof(vtable));                                           \
    memset(&g_##SUFFIX##_fortran_call, 0, sizeof(g_##SUFFIX##_fortran_call));     \
    vtable.ext_ops[OP_ID][FB_CONV_FORTRAN] =                                      \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_fortran;                   \
    fb_install_conv_thunks(&vtable, OP_ID);                                       \
    thunk = (fb_##SUFFIX##_cblas_fn)vtable.ext_ops[OP_ID][FB_CONV_CBLAS];         \
    if (!thunk) {                                                                 \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk was not installed\n"); \
        return 1;                                                                 \
    }                                                                             \
    info = thunk(FB_LAYOUT_COL_MAJOR, FB_UPPER, 2, ap, ipiv, (RTYPE)8, &rcond);  \
    if (info != (BASE) + 2 ||                                                     \
        g_##SUFFIX##_fortran_call.called != 1 ||                                  \
        g_##SUFFIX##_fortran_call.uplo != 'U' ||                                  \
        g_##SUFFIX##_fortran_call.n != 2 ||                                       \
        g_##SUFFIX##_fortran_call.ap != ap ||                                     \
        memcmp(g_##SUFFIX##_fortran_call.ipiv_vals, ipiv, sizeof(ipiv)) != 0 ||   \
        g_##SUFFIX##_fortran_call.anorm != (RTYPE)8 ||                            \
        g_##SUFFIX##_fortran_call.rcond != &rcond ||                              \
        !g_##SUFFIX##_fortran_call.work_nonnull ||                                \
        rcond != (RTYPE)((BASE) + 1)) {                                           \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not route complex SPCON inputs correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk routes complex SPCON inputs\n"); \
    return 0;                                                                     \
}                                                                                 \
static int check_##SUFFIX##_cblas_to_fortran(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                        \
    char uplo = 'L';                                                              \
    int n = 2;                                                                    \
    CTYPE ap[3] = { 0 };                                                          \
    int ipiv[2] = { 1, -2 };                                                      \
    RTYPE anorm = (RTYPE)5;                                                       \
    RTYPE rcond = (RTYPE)0;                                                       \
    CTYPE work[2] = { 0 };                                                        \
    int info = -999;                                                              \
    memset(&vtable, 0, sizeof(vtable));                                           \
    memset(&g_##SUFFIX##_cblas_call, 0, sizeof(g_##SUFFIX##_cblas_call));         \
    vtable.ext_ops[OP_ID][FB_CONV_CBLAS] =                                        \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_cblas;                     \
    fb_install_conv_thunks(&vtable, OP_ID);                                       \
    thunk = (fb_##SUFFIX##_fortran_fn)vtable.ext_ops[OP_ID][FB_CONV_FORTRAN];     \
    if (!thunk) {                                                                 \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk was not installed\n"); \
        return 1;                                                                 \
    }                                                                             \
    thunk(&uplo, &n, ap, ipiv, &anorm, &rcond, work, &info);                     \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                    \
        g_##SUFFIX##_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||                  \
        g_##SUFFIX##_cblas_call.uplo != FB_LOWER ||                               \
        g_##SUFFIX##_cblas_call.n != 2 ||                                         \
        g_##SUFFIX##_cblas_call.ap != ap ||                                       \
        g_##SUFFIX##_cblas_call.ipiv != ipiv ||                                   \
        g_##SUFFIX##_cblas_call.anorm != (RTYPE)5 ||                              \
        g_##SUFFIX##_cblas_call.rcond != &rcond ||                                \
        info != (BASE) + 4 ||                                                     \
        rcond != (RTYPE)((BASE) + 3)) {                                           \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not propagate complex SPCON info correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk propagates complex SPCON info\n"); \
    return 0;                                                                     \
}

DEFINE_SPCON_REAL_TESTS(sspcon, float, FB_OP_SSPCON, 100)
DEFINE_SPCON_REAL_TESTS(dspcon, double, FB_OP_DSPCON, 300)
DEFINE_SPCON_COMPLEX_TESTS(cspcon, fb_complex_float_t, float, FB_OP_CSPCON, 500)
DEFINE_SPCON_COMPLEX_TESTS(zspcon, fb_complex_double_t, double, FB_OP_ZSPCON, 700)

int main(void)
{
    int status = 0;

    status |= check_sspcon_fortran_to_cblas();
    status |= check_sspcon_cblas_to_fortran();
    status |= check_dspcon_fortran_to_cblas();
    status |= check_dspcon_cblas_to_fortran();
    status |= check_cspcon_fortran_to_cblas();
    status |= check_cspcon_cblas_to_fortran();
    status |= check_zspcon_fortran_to_cblas();
    status |= check_zspcon_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}