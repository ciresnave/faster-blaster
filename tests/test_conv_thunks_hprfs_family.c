#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

static fb_complex_float_t make_cfloat(float real_value)
{
    fb_complex_float_t value = (fb_complex_float_t)0;
    __real__ value = real_value;
    __imag__ value = 0.0f;
    return value;
}

static float cfloat_real(fb_complex_float_t value)
{
    return __real__ value;
}

static fb_complex_double_t make_cdouble(double real_value)
{
    fb_complex_double_t value = (fb_complex_double_t)0;
    __real__ value = real_value;
    __imag__ value = 0.0;
    return value;
}

static double cdouble_real(fb_complex_double_t value)
{
    return __real__ value;
}

#define DEFINE_HPRFS_COMPLEX_TESTS(SUFFIX, CTYPE, RTYPE, OP_ID, BASE, MAKE, REAL_PART) \
typedef int (*fb_##SUFFIX##_cblas_fn)(fb_layout_t layout, fb_uplo_t uplo,      \
                                      int n, int nrhs, const CTYPE *ap,        \
                                      const CTYPE *afp, const int *ipiv,       \
                                      const CTYPE *b, int ldb, CTYPE *x,       \
                                      int ldx, RTYPE *ferr, RTYPE *berr);      \
typedef void (*fb_##SUFFIX##_fortran_fn)(char *uplo, int *n, int *nrhs,        \
                                         CTYPE *ap, CTYPE *afp, int *ipiv,     \
                                         CTYPE *b, int *ldb, CTYPE *x,         \
                                         int *ldx, RTYPE *ferr, RTYPE *berr,   \
                                         CTYPE *work, RTYPE *rwork, int *info); \
static struct {                                                                   \
    int called;                                                                   \
    char uplo;                                                                    \
    int n;                                                                        \
    int nrhs;                                                                     \
    RTYPE ap_snapshot[6];                                                         \
    RTYPE afp_snapshot[6];                                                        \
    int ipiv_vals[3];                                                             \
    RTYPE b_snapshot[6];                                                          \
    RTYPE x_snapshot[6];                                                          \
    int ldb;                                                                      \
    int ldx;                                                                      \
    RTYPE *ferr;                                                                  \
    RTYPE *berr;                                                                  \
    int work_nonnull;                                                             \
    int rwork_nonnull;                                                            \
} g_##SUFFIX##_fortran_call;                                                      \
static struct {                                                                   \
    int called;                                                                   \
    fb_layout_t layout;                                                           \
    fb_uplo_t uplo;                                                               \
    int n;                                                                        \
    int nrhs;                                                                     \
    const CTYPE *ap;                                                              \
    const CTYPE *afp;                                                             \
    const int *ipiv;                                                              \
    const CTYPE *b;                                                               \
    int ldb;                                                                      \
    CTYPE *x;                                                                     \
    int ldx;                                                                      \
    RTYPE *ferr;                                                                  \
    RTYPE *berr;                                                                  \
} g_##SUFFIX##_cblas_call;                                                        \
static void stub_##SUFFIX##_fortran(char *uplo, int *n, int *nrhs, CTYPE *ap,  \
                                    CTYPE *afp, int *ipiv, CTYPE *b, int *ldb,\
                                    CTYPE *x, int *ldx, RTYPE *ferr, RTYPE *berr,\
                                    CTYPE *work, RTYPE *rwork, int *info)     \
{                                                                                 \
    size_t packed_len = (size_t)(*n * (*n + 1)) / 2U;                            \
    size_t mat_elems = (size_t)(*n) * (size_t)(*nrhs);                           \
    size_t index = 0;                                                             \
    g_##SUFFIX##_fortran_call.called += 1;                                        \
    g_##SUFFIX##_fortran_call.uplo = *uplo;                                       \
    g_##SUFFIX##_fortran_call.n = *n;                                             \
    g_##SUFFIX##_fortran_call.nrhs = *nrhs;                                       \
    memset(g_##SUFFIX##_fortran_call.ap_snapshot, 0, sizeof(g_##SUFFIX##_fortran_call.ap_snapshot)); \
    memset(g_##SUFFIX##_fortran_call.afp_snapshot, 0, sizeof(g_##SUFFIX##_fortran_call.afp_snapshot)); \
    memset(g_##SUFFIX##_fortran_call.b_snapshot, 0, sizeof(g_##SUFFIX##_fortran_call.b_snapshot)); \
    memset(g_##SUFFIX##_fortran_call.x_snapshot, 0, sizeof(g_##SUFFIX##_fortran_call.x_snapshot)); \
    for (index = 0; index < packed_len; ++index) {                               \
        g_##SUFFIX##_fortran_call.ap_snapshot[index] = REAL_PART(ap[index]);     \
        g_##SUFFIX##_fortran_call.afp_snapshot[index] = REAL_PART(afp[index]);   \
    }                                                                             \
    for (index = 0; index < mat_elems; ++index) {                                \
        g_##SUFFIX##_fortran_call.b_snapshot[index] = REAL_PART(b[index]);       \
        g_##SUFFIX##_fortran_call.x_snapshot[index] = REAL_PART(x[index]);       \
    }                                                                             \
    memset(g_##SUFFIX##_fortran_call.ipiv_vals, 0, sizeof(g_##SUFFIX##_fortran_call.ipiv_vals)); \
    if (*n > 0) {                                                                 \
        memcpy(g_##SUFFIX##_fortran_call.ipiv_vals, ipiv, (size_t)(*n) * sizeof(int)); \
    }                                                                             \
    g_##SUFFIX##_fortran_call.ldb = *ldb;                                         \
    g_##SUFFIX##_fortran_call.ldx = *ldx;                                         \
    g_##SUFFIX##_fortran_call.ferr = ferr;                                        \
    g_##SUFFIX##_fortran_call.berr = berr;                                        \
    g_##SUFFIX##_fortran_call.work_nonnull = (work != NULL);                      \
    g_##SUFFIX##_fortran_call.rwork_nonnull = (rwork != NULL);                    \
    x[0] = MAKE((RTYPE)((BASE) + 1));                                             \
    x[1] = MAKE((RTYPE)((BASE) + 2));                                             \
    x[2] = MAKE((RTYPE)((BASE) + 3));                                             \
    ferr[0] = (RTYPE)((BASE) + 4);                                                \
    berr[0] = (RTYPE)((BASE) + 5);                                                \
    *info = (BASE) + 6;                                                           \
}                                                                                 \
static int stub_##SUFFIX##_cblas(fb_layout_t layout, fb_uplo_t uplo, int n,     \
                                 int nrhs, const CTYPE *ap, const CTYPE *afp,   \
                                 const int *ipiv, const CTYPE *b, int ldb,      \
                                 CTYPE *x, int ldx, RTYPE *ferr, RTYPE *berr)   \
{                                                                                 \
    g_##SUFFIX##_cblas_call.called += 1;                                          \
    g_##SUFFIX##_cblas_call.layout = layout;                                      \
    g_##SUFFIX##_cblas_call.uplo = uplo;                                          \
    g_##SUFFIX##_cblas_call.n = n;                                                \
    g_##SUFFIX##_cblas_call.nrhs = nrhs;                                          \
    g_##SUFFIX##_cblas_call.ap = ap;                                              \
    g_##SUFFIX##_cblas_call.afp = afp;                                            \
    g_##SUFFIX##_cblas_call.ipiv = ipiv;                                          \
    g_##SUFFIX##_cblas_call.b = b;                                                \
    g_##SUFFIX##_cblas_call.ldb = ldb;                                            \
    g_##SUFFIX##_cblas_call.x = x;                                                \
    g_##SUFFIX##_cblas_call.ldx = ldx;                                            \
    g_##SUFFIX##_cblas_call.ferr = ferr;                                          \
    g_##SUFFIX##_cblas_call.berr = berr;                                          \
    ferr[0] = (RTYPE)((BASE) + 7);                                                \
    berr[0] = (RTYPE)((BASE) + 8);                                                \
    return (BASE) + 9;                                                            \
}                                                                                 \
static int check_##SUFFIX##_fortran_to_cblas(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                          \
    CTYPE ap_row[6] = { MAKE((RTYPE)11), MAKE((RTYPE)12), MAKE((RTYPE)13), MAKE((RTYPE)22), MAKE((RTYPE)23), MAKE((RTYPE)33) }; \
    CTYPE afp_row[6] = { MAKE((RTYPE)41), MAKE((RTYPE)42), MAKE((RTYPE)43), MAKE((RTYPE)52), MAKE((RTYPE)53), MAKE((RTYPE)63) }; \
    RTYPE expected_ap[6] = { (RTYPE)11, (RTYPE)12, (RTYPE)22, (RTYPE)13, (RTYPE)23, (RTYPE)33 }; \
    RTYPE expected_afp[6] = { (RTYPE)41, (RTYPE)42, (RTYPE)52, (RTYPE)43, (RTYPE)53, (RTYPE)63 }; \
    int ipiv[3] = { 1, -2, 3 };                                                   \
    CTYPE b_row[3] = { MAKE((RTYPE)1), MAKE((RTYPE)2), MAKE((RTYPE)3) };         \
    CTYPE x_row[3] = { MAKE((RTYPE)4), MAKE((RTYPE)5), MAKE((RTYPE)6) };         \
    RTYPE expected_b[3] = { (RTYPE)1, (RTYPE)2, (RTYPE)3 };                      \
    RTYPE expected_x[3] = { (RTYPE)4, (RTYPE)5, (RTYPE)6 };                      \
    RTYPE expected_x_after[3] = { (RTYPE)((BASE) + 1), (RTYPE)((BASE) + 2), (RTYPE)((BASE) + 3) }; \
    RTYPE ferr[1] = { 0 };                                                        \
    RTYPE berr[1] = { 0 };                                                        \
    int info = 0;                                                                 \
    size_t index = 0;                                                             \
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
    info = thunk(FB_LAYOUT_ROW_MAJOR, FB_UPPER, 3, 1, ap_row, afp_row, ipiv, b_row, 1, x_row, 1, ferr, berr); \
    if (info != (BASE) + 6 ||                                                     \
        g_##SUFFIX##_fortran_call.called != 1 ||                                  \
        g_##SUFFIX##_fortran_call.uplo != 'U' ||                                  \
        g_##SUFFIX##_fortran_call.n != 3 ||                                       \
        g_##SUFFIX##_fortran_call.nrhs != 1 ||                                    \
        memcmp(g_##SUFFIX##_fortran_call.ipiv_vals, ipiv, sizeof(ipiv)) != 0 ||   \
        g_##SUFFIX##_fortran_call.ldb != 3 ||                                     \
        g_##SUFFIX##_fortran_call.ldx != 3 ||                                     \
        g_##SUFFIX##_fortran_call.ferr != ferr ||                                 \
        g_##SUFFIX##_fortran_call.berr != berr ||                                 \
        !g_##SUFFIX##_fortran_call.work_nonnull ||                                \
        !g_##SUFFIX##_fortran_call.rwork_nonnull) {                               \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not route packed HPRFS calls\n"); \
        return 1;                                                                 \
    }                                                                             \
    for (index = 0; index < 6; ++index) {                                        \
        if (g_##SUFFIX##_fortran_call.ap_snapshot[index] != expected_ap[index] || \
            g_##SUFFIX##_fortran_call.afp_snapshot[index] != expected_afp[index]) { \
            fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not translate packed HPRFS storage correctly\n"); \
            return 1;                                                             \
        }                                                                         \
    }                                                                             \
    for (index = 0; index < 3; ++index) {                                        \
        if (g_##SUFFIX##_fortran_call.b_snapshot[index] != expected_b[index] ||  \
            g_##SUFFIX##_fortran_call.x_snapshot[index] != expected_x[index] ||  \
            REAL_PART(x_row[index]) != expected_x_after[index]) {                \
            fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not translate packed HPRFS vectors correctly\n"); \
            return 1;                                                             \
        }                                                                         \
    }                                                                             \
    if (ferr[0] != (RTYPE)((BASE) + 4) || berr[0] != (RTYPE)((BASE) + 5)) {      \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not propagate HPRFS error estimates correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk routes packed HPRFS inputs\n"); \
    return 0;                                                                     \
}                                                                                 \
static int check_##SUFFIX##_cblas_to_fortran(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                        \
    char uplo = 'L';                                                              \
    int n = 1;                                                                    \
    int nrhs = 1;                                                                 \
    CTYPE ap[1] = { MAKE((RTYPE)9) };                                             \
    CTYPE afp[1] = { MAKE((RTYPE)10) };                                           \
    int ipiv[1] = { 1 };                                                          \
    CTYPE b[1] = { MAKE((RTYPE)11) };                                             \
    CTYPE x[1] = { MAKE((RTYPE)12) };                                             \
    int ldb = 1;                                                                  \
    int ldx = 1;                                                                  \
    RTYPE ferr[1] = { 0 };                                                        \
    RTYPE berr[1] = { 0 };                                                        \
    CTYPE work[2] = { MAKE((RTYPE)0), MAKE((RTYPE)0) };                          \
    RTYPE rwork[1] = { 0 };                                                       \
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
    thunk(&uplo, &n, &nrhs, ap, afp, ipiv, b, &ldb, x, &ldx, ferr, berr, work, rwork, &info); \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                    \
        g_##SUFFIX##_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||                  \
        g_##SUFFIX##_cblas_call.uplo != FB_LOWER ||                               \
        g_##SUFFIX##_cblas_call.n != 1 ||                                         \
        g_##SUFFIX##_cblas_call.nrhs != 1 ||                                      \
        g_##SUFFIX##_cblas_call.ap != ap ||                                       \
        g_##SUFFIX##_cblas_call.afp != afp ||                                     \
        g_##SUFFIX##_cblas_call.ipiv != ipiv ||                                   \
        g_##SUFFIX##_cblas_call.b != b ||                                         \
        g_##SUFFIX##_cblas_call.ldb != 1 ||                                       \
        g_##SUFFIX##_cblas_call.x != x ||                                         \
        g_##SUFFIX##_cblas_call.ldx != 1 ||                                       \
        g_##SUFFIX##_cblas_call.ferr != ferr ||                                   \
        g_##SUFFIX##_cblas_call.berr != berr ||                                   \
        info != (BASE) + 9 || ferr[0] != (RTYPE)((BASE) + 7) || berr[0] != (RTYPE)((BASE) + 8)) { \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not propagate HPRFS info correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk propagates HPRFS info\n"); \
    return 0;                                                                     \
}

DEFINE_HPRFS_COMPLEX_TESTS(chprfs, fb_complex_float_t, float, FB_OP_CHPRFS, 100, make_cfloat, cfloat_real)
DEFINE_HPRFS_COMPLEX_TESTS(zhprfs, fb_complex_double_t, double, FB_OP_ZHPRFS, 300, make_cdouble, cdouble_real)

int main(void)
{
    int status = 0;

    status |= check_chprfs_fortran_to_cblas();
    status |= check_chprfs_cblas_to_fortran();
    status |= check_zhprfs_fortran_to_cblas();
    status |= check_zhprfs_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}