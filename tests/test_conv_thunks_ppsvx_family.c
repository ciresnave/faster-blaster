#include <math.h>
#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

static fb_complex_float_t make_cfloat(float real_value, float imag_value)
{
    fb_complex_float_t value = (fb_complex_float_t)0;
    __real__ value = real_value;
    __imag__ value = imag_value;
    return value;
}

static float cfloat_real(fb_complex_float_t value)
{
    return __real__ value;
}

static float cfloat_imag(fb_complex_float_t value)
{
    return __imag__ value;
}

static fb_complex_double_t make_cdouble(double real_value, double imag_value)
{
    fb_complex_double_t value = (fb_complex_double_t)0;
    __real__ value = real_value;
    __imag__ value = imag_value;
    return value;
}

static double cdouble_real(fb_complex_double_t value)
{
    return __real__ value;
}

static double cdouble_imag(fb_complex_double_t value)
{
    return __imag__ value;
}

#define DEFINE_PPSVX_REAL_TESTS(SUFFIX, TYPE, OP_ID, BASE)                      \
typedef int (*fb_##SUFFIX##_cblas_fn)(fb_layout_t layout, char fact,           \
                                      fb_uplo_t uplo, int n, int nrhs,         \
                                      TYPE *ap, TYPE *afp, char *equed,        \
                                      TYPE *s, TYPE *b, int ldb, TYPE *x,      \
                                      int ldx, TYPE *rcond, TYPE *ferr,        \
                                      TYPE *berr);                             \
typedef void (*fb_##SUFFIX##_fortran_fn)(char *fact, char *uplo, int *n,      \
                                         int *nrhs, TYPE *ap, TYPE *afp,       \
                                         char *equed, TYPE *s, TYPE *b,        \
                                         int *ldb, TYPE *x, int *ldx,          \
                                         TYPE *rcond, TYPE *ferr, TYPE *berr,  \
                                         TYPE *work, int *iwork, int *info);   \
static struct {                                                                  \
    int called;                                                                  \
    char fact;                                                                   \
    char uplo;                                                                   \
    int n;                                                                       \
    int nrhs;                                                                    \
    TYPE ap_snapshot[6];                                                         \
    TYPE afp_snapshot[6];                                                        \
    TYPE b_snapshot[6];                                                          \
    int ldb;                                                                     \
    int ldx;                                                                     \
    char *equed;                                                                 \
    TYPE *s;                                                                     \
    TYPE *rcond;                                                                 \
    TYPE *ferr;                                                                  \
    TYPE *berr;                                                                  \
    int work_nonnull;                                                            \
    int iwork_nonnull;                                                           \
} g_##SUFFIX##_fortran_call;                                                     \
static struct {                                                                  \
    int called;                                                                  \
    fb_layout_t layout;                                                          \
    char fact;                                                                   \
    fb_uplo_t uplo;                                                              \
    int n;                                                                       \
    int nrhs;                                                                    \
    TYPE *ap;                                                                    \
    TYPE *afp;                                                                   \
    char *equed;                                                                 \
    TYPE *s;                                                                     \
    TYPE *b;                                                                     \
    int ldb;                                                                     \
    TYPE *x;                                                                     \
    int ldx;                                                                     \
    TYPE *rcond;                                                                 \
    TYPE *ferr;                                                                  \
    TYPE *berr;                                                                  \
} g_##SUFFIX##_cblas_call;                                                       \
static void stub_##SUFFIX##_fortran(                                             \
    char *fact, char *uplo, int *n, int *nrhs, TYPE *ap, TYPE *afp,            \
    char *equed, TYPE *s, TYPE *b, int *ldb, TYPE *x, int *ldx,                \
    TYPE *rcond, TYPE *ferr, TYPE *berr, TYPE *work, int *iwork, int *info)    \
{                                                                                \
    g_##SUFFIX##_fortran_call.called += 1;                                       \
    g_##SUFFIX##_fortran_call.fact = *fact;                                      \
    g_##SUFFIX##_fortran_call.uplo = *uplo;                                      \
    g_##SUFFIX##_fortran_call.n = *n;                                            \
    g_##SUFFIX##_fortran_call.nrhs = *nrhs;                                      \
    memcpy(g_##SUFFIX##_fortran_call.ap_snapshot, ap, sizeof(g_##SUFFIX##_fortran_call.ap_snapshot)); \
    memcpy(g_##SUFFIX##_fortran_call.afp_snapshot, afp, sizeof(g_##SUFFIX##_fortran_call.afp_snapshot)); \
    memcpy(g_##SUFFIX##_fortran_call.b_snapshot, b, sizeof(g_##SUFFIX##_fortran_call.b_snapshot)); \
    g_##SUFFIX##_fortran_call.ldb = *ldb;                                        \
    g_##SUFFIX##_fortran_call.ldx = *ldx;                                        \
    g_##SUFFIX##_fortran_call.equed = equed;                                     \
    g_##SUFFIX##_fortran_call.s = s;                                             \
    g_##SUFFIX##_fortran_call.rcond = rcond;                                     \
    g_##SUFFIX##_fortran_call.ferr = ferr;                                       \
    g_##SUFFIX##_fortran_call.berr = berr;                                       \
    g_##SUFFIX##_fortran_call.work_nonnull = (work != NULL);                     \
    g_##SUFFIX##_fortran_call.iwork_nonnull = (iwork != NULL);                   \
    *equed = 'Y';                                                                \
    s[0] = (TYPE)((BASE) + 1);                                                   \
    s[1] = (TYPE)((BASE) + 2);                                                   \
    s[2] = (TYPE)((BASE) + 3);                                                   \
    *rcond = (TYPE)((BASE) + 4);                                                 \
    ferr[0] = (TYPE)((BASE) + 5);                                                \
    ferr[1] = (TYPE)((BASE) + 6);                                                \
    berr[0] = (TYPE)((BASE) + 7);                                                \
    berr[1] = (TYPE)((BASE) + 8);                                                \
    ap[0] = (TYPE)31; ap[1] = (TYPE)32; ap[2] = (TYPE)33; ap[3] = (TYPE)34; ap[4] = (TYPE)35; ap[5] = (TYPE)36; \
    afp[0] = (TYPE)41; afp[1] = (TYPE)42; afp[2] = (TYPE)43; afp[3] = (TYPE)44; afp[4] = (TYPE)45; afp[5] = (TYPE)46; \
    x[0] = (TYPE)51; x[1] = (TYPE)52; x[2] = (TYPE)53; x[3] = (TYPE)54; x[4] = (TYPE)55; x[5] = (TYPE)56; \
    *info = (BASE) + 9;                                                          \
}                                                                                \
static int stub_##SUFFIX##_cblas(                                                \
    fb_layout_t layout, char fact, fb_uplo_t uplo, int n, int nrhs, TYPE *ap,  \
    TYPE *afp, char *equed, TYPE *s, TYPE *b, int ldb, TYPE *x, int ldx,       \
    TYPE *rcond, TYPE *ferr, TYPE *berr)                                        \
{                                                                                \
    g_##SUFFIX##_cblas_call.called += 1;                                         \
    g_##SUFFIX##_cblas_call.layout = layout;                                     \
    g_##SUFFIX##_cblas_call.fact = fact;                                         \
    g_##SUFFIX##_cblas_call.uplo = uplo;                                         \
    g_##SUFFIX##_cblas_call.n = n;                                               \
    g_##SUFFIX##_cblas_call.nrhs = nrhs;                                         \
    g_##SUFFIX##_cblas_call.ap = ap;                                             \
    g_##SUFFIX##_cblas_call.afp = afp;                                           \
    g_##SUFFIX##_cblas_call.equed = equed;                                       \
    g_##SUFFIX##_cblas_call.s = s;                                               \
    g_##SUFFIX##_cblas_call.b = b;                                               \
    g_##SUFFIX##_cblas_call.ldb = ldb;                                           \
    g_##SUFFIX##_cblas_call.x = x;                                               \
    g_##SUFFIX##_cblas_call.ldx = ldx;                                           \
    g_##SUFFIX##_cblas_call.rcond = rcond;                                       \
    g_##SUFFIX##_cblas_call.ferr = ferr;                                         \
    g_##SUFFIX##_cblas_call.berr = berr;                                         \
    x[0] = (TYPE)((BASE) + 10);                                                  \
    *rcond = (TYPE)((BASE) + 11);                                                \
    ferr[0] = (TYPE)((BASE) + 12);                                               \
    berr[0] = (TYPE)((BASE) + 13);                                               \
    return (BASE) + 14;                                                          \
}                                                                                \
static int check_##SUFFIX##_fortran_to_cblas(void)                               \
{                                                                                \
    fb_backend_vtable_t vtable;                                                  \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                         \
    TYPE ap_row[6] = { (TYPE)11, (TYPE)12, (TYPE)13, (TYPE)22, (TYPE)23, (TYPE)33 }; \
    TYPE afp_row[6] = { 0, 0, 0, 0, 0, 0 };                                     \
    TYPE zero_afp[6] = { 0, 0, 0, 0, 0, 0 };                                    \
    TYPE expected_ap_in[6] = { (TYPE)11, (TYPE)12, (TYPE)22, (TYPE)13, (TYPE)23, (TYPE)33 }; \
    TYPE expected_b_in[6] = { (TYPE)1, (TYPE)3, (TYPE)5, (TYPE)2, (TYPE)4, (TYPE)6 }; \
    TYPE expected_ap_out[6] = { (TYPE)31, (TYPE)32, (TYPE)34, (TYPE)33, (TYPE)35, (TYPE)36 }; \
    TYPE expected_afp_out[6] = { (TYPE)41, (TYPE)42, (TYPE)44, (TYPE)43, (TYPE)45, (TYPE)46 }; \
    TYPE expected_x_out[6] = { (TYPE)51, (TYPE)54, (TYPE)52, (TYPE)55, (TYPE)53, (TYPE)56 }; \
    TYPE b_row[6] = { (TYPE)1, (TYPE)2, (TYPE)3, (TYPE)4, (TYPE)5, (TYPE)6 };  \
    TYPE original_b[6] = { (TYPE)1, (TYPE)2, (TYPE)3, (TYPE)4, (TYPE)5, (TYPE)6 }; \
    TYPE x_row[6] = { 0, 0, 0, 0, 0, 0 };                                       \
    TYPE s[3] = { 0, 0, 0 };                                                    \
    TYPE rcond = 0;                                                              \
    TYPE ferr[2] = { 0, 0 };                                                    \
    TYPE berr[2] = { 0, 0 };                                                    \
    char equed = 'N';                                                            \
    int info = 0;                                                                \
    memset(&vtable, 0, sizeof(vtable));                                          \
    memset(&g_##SUFFIX##_fortran_call, 0, sizeof(g_##SUFFIX##_fortran_call));    \
    vtable.ext_ops[OP_ID][FB_CONV_FORTRAN] =                                     \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_fortran;                  \
    fb_install_conv_thunks(&vtable, OP_ID);                                      \
    thunk = (fb_##SUFFIX##_cblas_fn)vtable.ext_ops[OP_ID][FB_CONV_CBLAS];        \
    if (!thunk) {                                                                \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk was not installed\n"); \
        return 1;                                                                \
    }                                                                            \
    info = thunk(FB_LAYOUT_ROW_MAJOR, 'E', FB_UPPER, 3, 2, ap_row, afp_row, &equed, s, b_row, 2, x_row, 2, &rcond, ferr, berr); \
    if (info != (BASE) + 9 ||                                                    \
        g_##SUFFIX##_fortran_call.called != 1 ||                                 \
        g_##SUFFIX##_fortran_call.fact != 'E' ||                                 \
        g_##SUFFIX##_fortran_call.uplo != 'U' ||                                 \
        g_##SUFFIX##_fortran_call.n != 3 ||                                      \
        g_##SUFFIX##_fortran_call.nrhs != 2 ||                                   \
        memcmp(g_##SUFFIX##_fortran_call.ap_snapshot, expected_ap_in, sizeof(expected_ap_in)) != 0 || \
        memcmp(g_##SUFFIX##_fortran_call.afp_snapshot, zero_afp, sizeof(zero_afp)) != 0 || \
        memcmp(g_##SUFFIX##_fortran_call.b_snapshot, expected_b_in, sizeof(expected_b_in)) != 0 || \
        g_##SUFFIX##_fortran_call.ldb != 3 ||                                    \
        g_##SUFFIX##_fortran_call.ldx != 3 ||                                    \
        g_##SUFFIX##_fortran_call.equed != &equed ||                             \
        g_##SUFFIX##_fortran_call.s != s ||                                      \
        g_##SUFFIX##_fortran_call.rcond != &rcond ||                             \
        g_##SUFFIX##_fortran_call.ferr != ferr ||                                \
        g_##SUFFIX##_fortran_call.berr != berr ||                                \
        !g_##SUFFIX##_fortran_call.work_nonnull ||                               \
        !g_##SUFFIX##_fortran_call.iwork_nonnull ||                              \
        equed != 'Y' ||                                                          \
        memcmp(ap_row, expected_ap_out, sizeof(expected_ap_out)) != 0 ||        \
        memcmp(afp_row, expected_afp_out, sizeof(expected_afp_out)) != 0 ||     \
        memcmp(b_row, original_b, sizeof(original_b)) != 0 ||                   \
        memcmp(x_row, expected_x_out, sizeof(expected_x_out)) != 0 ||           \
        s[0] != (TYPE)((BASE) + 1) || s[1] != (TYPE)((BASE) + 2) || s[2] != (TYPE)((BASE) + 3) || \
        rcond != (TYPE)((BASE) + 4) || ferr[0] != (TYPE)((BASE) + 5) || ferr[1] != (TYPE)((BASE) + 6) || \
        berr[0] != (TYPE)((BASE) + 7) || berr[1] != (TYPE)((BASE) + 8)) {      \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not route packed PPSVX inputs correctly\n"); \
        return 1;                                                                \
    }                                                                            \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk routes packed PPSVX inputs\n"); \
    return 0;                                                                    \
}                                                                                \
static int check_##SUFFIX##_cblas_to_fortran(void)                               \
{                                                                                \
    fb_backend_vtable_t vtable;                                                  \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                       \
    char fact = 'F';                                                             \
    char uplo = 'L';                                                             \
    char equed = 'Y';                                                            \
    int n = 2;                                                                   \
    int nrhs = 1;                                                                \
    TYPE ap[3] = { (TYPE)9, (TYPE)10, (TYPE)11 };                               \
    TYPE afp[3] = { (TYPE)12, (TYPE)13, (TYPE)14 };                             \
    TYPE s[2] = { (TYPE)15, (TYPE)16 };                                          \
    TYPE b[2] = { (TYPE)17, (TYPE)18 };                                          \
    TYPE x[2] = { 0, 0 };                                                        \
    int ldb = 2;                                                                 \
    int ldx = 2;                                                                 \
    TYPE rcond = 0;                                                              \
    TYPE ferr[1] = { 0 };                                                        \
    TYPE berr[1] = { 0 };                                                        \
    TYPE work[6] = { 0, 0, 0, 0, 0, 0 };                                         \
    int iwork[2] = { 0, 0 };                                                     \
    int info = -999;                                                             \
    memset(&vtable, 0, sizeof(vtable));                                          \
    memset(&g_##SUFFIX##_cblas_call, 0, sizeof(g_##SUFFIX##_cblas_call));        \
    vtable.ext_ops[OP_ID][FB_CONV_CBLAS] =                                       \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_cblas;                    \
    fb_install_conv_thunks(&vtable, OP_ID);                                      \
    thunk = (fb_##SUFFIX##_fortran_fn)vtable.ext_ops[OP_ID][FB_CONV_FORTRAN];    \
    if (!thunk) {                                                                \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk was not installed\n"); \
        return 1;                                                                \
    }                                                                            \
    thunk(&fact, &uplo, &n, &nrhs, ap, afp, &equed, s, b, &ldb, x, &ldx, &rcond, ferr, berr, work, iwork, &info); \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                   \
        g_##SUFFIX##_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||                 \
        g_##SUFFIX##_cblas_call.fact != 'F' ||                                   \
        g_##SUFFIX##_cblas_call.uplo != FB_LOWER ||                              \
        g_##SUFFIX##_cblas_call.n != 2 ||                                        \
        g_##SUFFIX##_cblas_call.nrhs != 1 ||                                     \
        g_##SUFFIX##_cblas_call.ap != ap ||                                      \
        g_##SUFFIX##_cblas_call.afp != afp ||                                    \
        g_##SUFFIX##_cblas_call.equed != &equed ||                               \
        g_##SUFFIX##_cblas_call.s != s ||                                        \
        g_##SUFFIX##_cblas_call.b != b ||                                        \
        g_##SUFFIX##_cblas_call.ldb != 2 ||                                      \
        g_##SUFFIX##_cblas_call.x != x ||                                        \
        g_##SUFFIX##_cblas_call.ldx != 2 ||                                      \
        g_##SUFFIX##_cblas_call.rcond != &rcond ||                               \
        g_##SUFFIX##_cblas_call.ferr != ferr ||                                  \
        g_##SUFFIX##_cblas_call.berr != berr ||                                  \
        info != (BASE) + 14 || x[0] != (TYPE)((BASE) + 10) ||                   \
        rcond != (TYPE)((BASE) + 11) || ferr[0] != (TYPE)((BASE) + 12) ||       \
        berr[0] != (TYPE)((BASE) + 13)) {                                        \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not propagate PPSVX info correctly\n"); \
        return 1;                                                                \
    }                                                                            \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk propagates PPSVX info\n"); \
    return 0;                                                                    \
}

#define DEFINE_PPSVX_COMPLEX_TESTS(SUFFIX, CTYPE, RTYPE, OP_ID, BASE, MAKE, REAL_PART, IMAG_PART, TOL) \
typedef int (*fb_##SUFFIX##_cblas_fn)(fb_layout_t layout, char fact,           \
                                      fb_uplo_t uplo, int n, int nrhs,         \
                                      CTYPE *ap, CTYPE *afp, char *equed,      \
                                      RTYPE *s, CTYPE *b, int ldb, CTYPE *x,   \
                                      int ldx, RTYPE *rcond, RTYPE *ferr,      \
                                      RTYPE *berr);                            \
typedef void (*fb_##SUFFIX##_fortran_fn)(char *fact, char *uplo, int *n,      \
                                         int *nrhs, CTYPE *ap, CTYPE *afp,     \
                                         char *equed, RTYPE *s, CTYPE *b,      \
                                         int *ldb, CTYPE *x, int *ldx,         \
                                         RTYPE *rcond, RTYPE *ferr, RTYPE *berr,\
                                         CTYPE *work, RTYPE *rwork, int *info);\
static struct {                                                                  \
    int called;                                                                  \
    char fact;                                                                   \
    char uplo;                                                                   \
    int n;                                                                       \
    int nrhs;                                                                    \
    CTYPE *ap;                                                                   \
    CTYPE *afp;                                                                  \
    char *equed;                                                                 \
    RTYPE *s;                                                                    \
    CTYPE *b;                                                                    \
    int ldb;                                                                     \
    CTYPE *x;                                                                    \
    int ldx;                                                                     \
    RTYPE *rcond;                                                                \
    RTYPE *ferr;                                                                 \
    RTYPE *berr;                                                                 \
    int work_nonnull;                                                            \
    int rwork_nonnull;                                                           \
} g_##SUFFIX##_fortran_call;                                                     \
static struct {                                                                  \
    int called;                                                                  \
    fb_layout_t layout;                                                          \
    char fact;                                                                   \
    fb_uplo_t uplo;                                                              \
    int n;                                                                       \
    int nrhs;                                                                    \
    CTYPE *ap;                                                                   \
    CTYPE *afp;                                                                  \
    char *equed;                                                                 \
    RTYPE *s;                                                                    \
    CTYPE *b;                                                                    \
    int ldb;                                                                     \
    CTYPE *x;                                                                    \
    int ldx;                                                                     \
    RTYPE *rcond;                                                                \
    RTYPE *ferr;                                                                 \
    RTYPE *berr;                                                                 \
} g_##SUFFIX##_cblas_call;                                                       \
static void stub_##SUFFIX##_fortran(                                             \
    char *fact, char *uplo, int *n, int *nrhs, CTYPE *ap, CTYPE *afp,          \
    char *equed, RTYPE *s, CTYPE *b, int *ldb, CTYPE *x, int *ldx,             \
    RTYPE *rcond, RTYPE *ferr, RTYPE *berr, CTYPE *work, RTYPE *rwork, int *info) \
{                                                                                \
    g_##SUFFIX##_fortran_call.called += 1;                                       \
    g_##SUFFIX##_fortran_call.fact = *fact;                                      \
    g_##SUFFIX##_fortran_call.uplo = *uplo;                                      \
    g_##SUFFIX##_fortran_call.n = *n;                                            \
    g_##SUFFIX##_fortran_call.nrhs = *nrhs;                                      \
    g_##SUFFIX##_fortran_call.ap = ap;                                           \
    g_##SUFFIX##_fortran_call.afp = afp;                                         \
    g_##SUFFIX##_fortran_call.equed = equed;                                     \
    g_##SUFFIX##_fortran_call.s = s;                                             \
    g_##SUFFIX##_fortran_call.b = b;                                             \
    g_##SUFFIX##_fortran_call.ldb = *ldb;                                        \
    g_##SUFFIX##_fortran_call.x = x;                                             \
    g_##SUFFIX##_fortran_call.ldx = *ldx;                                        \
    g_##SUFFIX##_fortran_call.rcond = rcond;                                     \
    g_##SUFFIX##_fortran_call.ferr = ferr;                                       \
    g_##SUFFIX##_fortran_call.berr = berr;                                       \
    g_##SUFFIX##_fortran_call.work_nonnull = (work != NULL);                     \
    g_##SUFFIX##_fortran_call.rwork_nonnull = (rwork != NULL);                   \
    *equed = 'N';                                                                \
    s[0] = (RTYPE)((BASE) + 1);                                                  \
    *rcond = (RTYPE)((BASE) + 2);                                                \
    ferr[0] = (RTYPE)((BASE) + 3);                                               \
    berr[0] = (RTYPE)((BASE) + 4);                                               \
    x[0] = MAKE((RTYPE)((BASE) + 5), (RTYPE)((BASE) + 6));                      \
    *info = (BASE) + 7;                                                          \
}                                                                                \
static int stub_##SUFFIX##_cblas(                                                \
    fb_layout_t layout, char fact, fb_uplo_t uplo, int n, int nrhs, CTYPE *ap, \
    CTYPE *afp, char *equed, RTYPE *s, CTYPE *b, int ldb, CTYPE *x, int ldx,   \
    RTYPE *rcond, RTYPE *ferr, RTYPE *berr)                                     \
{                                                                                \
    g_##SUFFIX##_cblas_call.called += 1;                                         \
    g_##SUFFIX##_cblas_call.layout = layout;                                     \
    g_##SUFFIX##_cblas_call.fact = fact;                                         \
    g_##SUFFIX##_cblas_call.uplo = uplo;                                         \
    g_##SUFFIX##_cblas_call.n = n;                                               \
    g_##SUFFIX##_cblas_call.nrhs = nrhs;                                         \
    g_##SUFFIX##_cblas_call.ap = ap;                                             \
    g_##SUFFIX##_cblas_call.afp = afp;                                           \
    g_##SUFFIX##_cblas_call.equed = equed;                                       \
    g_##SUFFIX##_cblas_call.s = s;                                               \
    g_##SUFFIX##_cblas_call.b = b;                                               \
    g_##SUFFIX##_cblas_call.ldb = ldb;                                           \
    g_##SUFFIX##_cblas_call.x = x;                                               \
    g_##SUFFIX##_cblas_call.ldx = ldx;                                           \
    g_##SUFFIX##_cblas_call.rcond = rcond;                                       \
    g_##SUFFIX##_cblas_call.ferr = ferr;                                         \
    g_##SUFFIX##_cblas_call.berr = berr;                                         \
    x[0] = MAKE((RTYPE)((BASE) + 8), (RTYPE)((BASE) + 9));                      \
    *rcond = (RTYPE)((BASE) + 10);                                               \
    ferr[0] = (RTYPE)((BASE) + 11);                                              \
    berr[0] = (RTYPE)((BASE) + 12);                                              \
    return (BASE) + 13;                                                          \
}                                                                                \
static int check_##SUFFIX##_fortran_to_cblas(void)                               \
{                                                                                \
    fb_backend_vtable_t vtable;                                                  \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                         \
    CTYPE ap[1] = { MAKE((RTYPE)1, (RTYPE)0) };                                 \
    CTYPE afp[1] = { MAKE((RTYPE)2, (RTYPE)0) };                                \
    CTYPE b[1] = { MAKE((RTYPE)3, (RTYPE)0) };                                  \
    CTYPE x[1] = { MAKE((RTYPE)0, (RTYPE)0) };                                  \
    RTYPE s[1] = { 0 };                                                          \
    RTYPE rcond = 0;                                                             \
    RTYPE ferr[1] = { 0 };                                                       \
    RTYPE berr[1] = { 0 };                                                       \
    char equed = 'Y';                                                            \
    int info = 0;                                                                \
    memset(&vtable, 0, sizeof(vtable));                                          \
    memset(&g_##SUFFIX##_fortran_call, 0, sizeof(g_##SUFFIX##_fortran_call));    \
    vtable.ext_ops[OP_ID][FB_CONV_FORTRAN] =                                     \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_fortran;                  \
    fb_install_conv_thunks(&vtable, OP_ID);                                      \
    thunk = (fb_##SUFFIX##_cblas_fn)vtable.ext_ops[OP_ID][FB_CONV_CBLAS];        \
    if (!thunk) {                                                                \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk was not installed\n"); \
        return 1;                                                                \
    }                                                                            \
    info = thunk(FB_LAYOUT_COL_MAJOR, 'N', FB_UPPER, 1, 1, ap, afp, &equed, s, b, 1, x, 1, &rcond, ferr, berr); \
    if (info != (BASE) + 7 ||                                                    \
        g_##SUFFIX##_fortran_call.called != 1 ||                                 \
        g_##SUFFIX##_fortran_call.fact != 'N' ||                                 \
        g_##SUFFIX##_fortran_call.uplo != 'U' ||                                 \
        g_##SUFFIX##_fortran_call.n != 1 ||                                      \
        g_##SUFFIX##_fortran_call.nrhs != 1 ||                                   \
        g_##SUFFIX##_fortran_call.ap != ap ||                                    \
        g_##SUFFIX##_fortran_call.afp != afp ||                                  \
        g_##SUFFIX##_fortran_call.equed != &equed ||                             \
        g_##SUFFIX##_fortran_call.s != s ||                                      \
        g_##SUFFIX##_fortran_call.b != b ||                                      \
        g_##SUFFIX##_fortran_call.ldb != 1 ||                                    \
        g_##SUFFIX##_fortran_call.x != x ||                                      \
        g_##SUFFIX##_fortran_call.ldx != 1 ||                                    \
        g_##SUFFIX##_fortran_call.rcond != &rcond ||                             \
        g_##SUFFIX##_fortran_call.ferr != ferr ||                                \
        g_##SUFFIX##_fortran_call.berr != berr ||                                \
        !g_##SUFFIX##_fortran_call.work_nonnull ||                               \
        !g_##SUFFIX##_fortran_call.rwork_nonnull ||                              \
        equed != 'N' || fabs((double)(s[0] - (RTYPE)((BASE) + 1))) > (TOL) ||   \
        fabs((double)(rcond - (RTYPE)((BASE) + 2))) > (TOL) ||                   \
        fabs((double)(ferr[0] - (RTYPE)((BASE) + 3))) > (TOL) ||                 \
        fabs((double)(berr[0] - (RTYPE)((BASE) + 4))) > (TOL) ||                 \
        fabs((double)(REAL_PART(x[0]) - (RTYPE)((BASE) + 5))) > (TOL) ||         \
        fabs((double)(IMAG_PART(x[0]) - (RTYPE)((BASE) + 6))) > (TOL)) {         \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not route complex PPSVX inputs correctly\n"); \
        return 1;                                                                \
    }                                                                            \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk routes complex PPSVX inputs\n"); \
    return 0;                                                                    \
}                                                                                \
static int check_##SUFFIX##_cblas_to_fortran(void)                               \
{                                                                                \
    fb_backend_vtable_t vtable;                                                  \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                       \
    char fact = 'F';                                                             \
    char uplo = 'L';                                                             \
    char equed = 'Y';                                                            \
    int n = 1;                                                                   \
    int nrhs = 1;                                                                \
    CTYPE ap[1] = { MAKE((RTYPE)1, (RTYPE)0) };                                 \
    CTYPE afp[1] = { MAKE((RTYPE)2, (RTYPE)0) };                                \
    RTYPE s[1] = { (RTYPE)3 };                                                   \
    CTYPE b[1] = { MAKE((RTYPE)4, (RTYPE)0) };                                  \
    CTYPE x[1] = { MAKE((RTYPE)0, (RTYPE)0) };                                  \
    int ldb = 1;                                                                 \
    int ldx = 1;                                                                 \
    RTYPE rcond = 0;                                                             \
    RTYPE ferr[1] = { 0 };                                                       \
    RTYPE berr[1] = { 0 };                                                       \
    CTYPE work[2] = { MAKE((RTYPE)0, (RTYPE)0), MAKE((RTYPE)0, (RTYPE)0) };    \
    RTYPE rwork[1] = { 0 };                                                      \
    int info = -999;                                                             \
    memset(&vtable, 0, sizeof(vtable));                                          \
    memset(&g_##SUFFIX##_cblas_call, 0, sizeof(g_##SUFFIX##_cblas_call));        \
    vtable.ext_ops[OP_ID][FB_CONV_CBLAS] =                                       \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_cblas;                    \
    fb_install_conv_thunks(&vtable, OP_ID);                                      \
    thunk = (fb_##SUFFIX##_fortran_fn)vtable.ext_ops[OP_ID][FB_CONV_FORTRAN];    \
    if (!thunk) {                                                                \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk was not installed\n"); \
        return 1;                                                                \
    }                                                                            \
    thunk(&fact, &uplo, &n, &nrhs, ap, afp, &equed, s, b, &ldb, x, &ldx, &rcond, ferr, berr, work, rwork, &info); \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                   \
        g_##SUFFIX##_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||                 \
        g_##SUFFIX##_cblas_call.fact != 'F' ||                                   \
        g_##SUFFIX##_cblas_call.uplo != FB_LOWER ||                              \
        g_##SUFFIX##_cblas_call.n != 1 ||                                        \
        g_##SUFFIX##_cblas_call.nrhs != 1 ||                                     \
        g_##SUFFIX##_cblas_call.ap != ap ||                                      \
        g_##SUFFIX##_cblas_call.afp != afp ||                                    \
        g_##SUFFIX##_cblas_call.equed != &equed ||                               \
        g_##SUFFIX##_cblas_call.s != s ||                                        \
        g_##SUFFIX##_cblas_call.b != b ||                                        \
        g_##SUFFIX##_cblas_call.ldb != 1 ||                                      \
        g_##SUFFIX##_cblas_call.x != x ||                                        \
        g_##SUFFIX##_cblas_call.ldx != 1 ||                                      \
        g_##SUFFIX##_cblas_call.rcond != &rcond ||                               \
        g_##SUFFIX##_cblas_call.ferr != ferr ||                                  \
        g_##SUFFIX##_cblas_call.berr != berr ||                                  \
        info != (BASE) + 13 ||                                                   \
        fabs((double)(REAL_PART(x[0]) - (RTYPE)((BASE) + 8))) > (TOL) ||        \
        fabs((double)(IMAG_PART(x[0]) - (RTYPE)((BASE) + 9))) > (TOL) ||        \
        fabs((double)(rcond - (RTYPE)((BASE) + 10))) > (TOL) ||                 \
        fabs((double)(ferr[0] - (RTYPE)((BASE) + 11))) > (TOL) ||               \
        fabs((double)(berr[0] - (RTYPE)((BASE) + 12))) > (TOL)) {               \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not propagate complex PPSVX info correctly\n"); \
        return 1;                                                                \
    }                                                                            \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk propagates complex PPSVX info\n"); \
    return 0;                                                                    \
}

DEFINE_PPSVX_REAL_TESTS(sppsvx, float, FB_OP_SPPSVX, 100)
DEFINE_PPSVX_REAL_TESTS(dppsvx, double, FB_OP_DPPSVX, 300)
DEFINE_PPSVX_COMPLEX_TESTS(cppsvx, fb_complex_float_t, float, FB_OP_CPPSVX, 500, make_cfloat, cfloat_real, cfloat_imag, 1.0e-5)
DEFINE_PPSVX_COMPLEX_TESTS(zppsvx, fb_complex_double_t, double, FB_OP_ZPPSVX, 700, make_cdouble, cdouble_real, cdouble_imag, 1.0e-12)

int main(void)
{
    int status = 0;

    status |= check_sppsvx_fortran_to_cblas();
    status |= check_sppsvx_cblas_to_fortran();
    status |= check_dppsvx_fortran_to_cblas();
    status |= check_dppsvx_cblas_to_fortran();
    status |= check_cppsvx_fortran_to_cblas();
    status |= check_cppsvx_cblas_to_fortran();
    status |= check_zppsvx_fortran_to_cblas();
    status |= check_zppsvx_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}