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

#define DEFINE_PTSVX_REAL_TESTS(SUFFIX, TYPE, OP_ID, BASE)                       \
typedef int (*fb_##SUFFIX##_cblas_fn)(fb_layout_t layout, char fact, int n,    \
                                      int nrhs, TYPE *d, TYPE *e, TYPE *df,    \
                                      TYPE *ef, TYPE *b, int ldb, TYPE *x,     \
                                      int ldx, TYPE *rcond, TYPE *ferr,        \
                                      TYPE *berr);                             \
typedef void (*fb_##SUFFIX##_fortran_fn)(char *fact, int *n, int *nrhs,       \
                                         TYPE *d, TYPE *e, TYPE *df, TYPE *ef, \
                                         TYPE *b, int *ldb, TYPE *x, int *ldx, \
                                         TYPE *rcond, TYPE *ferr, TYPE *berr,  \
                                         TYPE *work, int *info);               \
static struct {                                                                   \
    int called;                                                                   \
    char fact;                                                                    \
    int n;                                                                        \
    int nrhs;                                                                     \
    TYPE d_snapshot[2];                                                           \
    TYPE e_snapshot[1];                                                           \
    TYPE *df;                                                                     \
    TYPE *ef;                                                                     \
    TYPE b_snapshot[4];                                                           \
    int ldb;                                                                      \
    int ldx;                                                                      \
    TYPE *rcond;                                                                  \
    TYPE *ferr;                                                                   \
    TYPE *berr;                                                                   \
    int work_nonnull;                                                             \
} g_##SUFFIX##_fortran_call;                                                      \
static struct {                                                                   \
    int called;                                                                   \
    fb_layout_t layout;                                                           \
    char fact;                                                                    \
    int n;                                                                        \
    int nrhs;                                                                     \
    TYPE *d;                                                                      \
    TYPE *e;                                                                      \
    TYPE *df;                                                                     \
    TYPE *ef;                                                                     \
    TYPE *b;                                                                      \
    int ldb;                                                                      \
    TYPE *x;                                                                      \
    int ldx;                                                                      \
    TYPE *rcond;                                                                  \
    TYPE *ferr;                                                                   \
    TYPE *berr;                                                                   \
} g_##SUFFIX##_cblas_call;                                                        \
static void stub_##SUFFIX##_fortran(char *fact, int *n, int *nrhs, TYPE *d,    \
                                    TYPE *e, TYPE *df, TYPE *ef, TYPE *b,      \
                                    int *ldb, TYPE *x, int *ldx, TYPE *rcond,  \
                                    TYPE *ferr, TYPE *berr, TYPE *work,        \
                                    int *info)                                  \
{                                                                                 \
    g_##SUFFIX##_fortran_call.called += 1;                                        \
    g_##SUFFIX##_fortran_call.fact = *fact;                                       \
    g_##SUFFIX##_fortran_call.n = *n;                                             \
    g_##SUFFIX##_fortran_call.nrhs = *nrhs;                                       \
    memcpy(g_##SUFFIX##_fortran_call.d_snapshot, d, (size_t)(*n) * sizeof(TYPE)); \
    memcpy(g_##SUFFIX##_fortran_call.e_snapshot, e, (size_t)((*n) - 1) * sizeof(TYPE)); \
    g_##SUFFIX##_fortran_call.df = df;                                            \
    g_##SUFFIX##_fortran_call.ef = ef;                                            \
    memcpy(g_##SUFFIX##_fortran_call.b_snapshot, b, (size_t)(*ldb) * (size_t)(*nrhs) * sizeof(TYPE)); \
    g_##SUFFIX##_fortran_call.ldb = *ldb;                                         \
    g_##SUFFIX##_fortran_call.ldx = *ldx;                                         \
    g_##SUFFIX##_fortran_call.rcond = rcond;                                      \
    g_##SUFFIX##_fortran_call.ferr = ferr;                                        \
    g_##SUFFIX##_fortran_call.berr = berr;                                        \
    g_##SUFFIX##_fortran_call.work_nonnull = (work != NULL);                      \
    df[0] = (TYPE)((BASE) + 1);                                                   \
    df[1] = (TYPE)((BASE) + 2);                                                   \
    ef[0] = (TYPE)((BASE) + 3);                                                   \
    x[0] = (TYPE)((BASE) + 4);                                                    \
    x[1] = (TYPE)((BASE) + 5);                                                    \
    x[2] = (TYPE)((BASE) + 6);                                                    \
    x[3] = (TYPE)((BASE) + 7);                                                    \
    *rcond = (TYPE)((BASE) + 8);                                                  \
    ferr[0] = (TYPE)((BASE) + 9);                                                 \
    berr[0] = (TYPE)((BASE) + 10);                                                \
    *info = (BASE) + 11;                                                          \
}                                                                                 \
static int stub_##SUFFIX##_cblas(fb_layout_t layout, char fact, int n, int nrhs,\
                                 TYPE *d, TYPE *e, TYPE *df, TYPE *ef,         \
                                 TYPE *b, int ldb, TYPE *x, int ldx,           \
                                 TYPE *rcond, TYPE *ferr, TYPE *berr)          \
{                                                                                 \
    g_##SUFFIX##_cblas_call.called += 1;                                          \
    g_##SUFFIX##_cblas_call.layout = layout;                                      \
    g_##SUFFIX##_cblas_call.fact = fact;                                          \
    g_##SUFFIX##_cblas_call.n = n;                                                \
    g_##SUFFIX##_cblas_call.nrhs = nrhs;                                          \
    g_##SUFFIX##_cblas_call.d = d;                                                \
    g_##SUFFIX##_cblas_call.e = e;                                                \
    g_##SUFFIX##_cblas_call.df = df;                                              \
    g_##SUFFIX##_cblas_call.ef = ef;                                              \
    g_##SUFFIX##_cblas_call.b = b;                                                \
    g_##SUFFIX##_cblas_call.ldb = ldb;                                            \
    g_##SUFFIX##_cblas_call.x = x;                                                \
    g_##SUFFIX##_cblas_call.ldx = ldx;                                            \
    g_##SUFFIX##_cblas_call.rcond = rcond;                                        \
    g_##SUFFIX##_cblas_call.ferr = ferr;                                          \
    g_##SUFFIX##_cblas_call.berr = berr;                                          \
    df[0] = (TYPE)((BASE) + 12);                                                  \
    ef[0] = (TYPE)((BASE) + 13);                                                  \
    x[0] = (TYPE)((BASE) + 14);                                                   \
    *rcond = (TYPE)((BASE) + 15);                                                 \
    ferr[0] = (TYPE)((BASE) + 16);                                                \
    berr[0] = (TYPE)((BASE) + 17);                                                \
    return (BASE) + 18;                                                           \
}                                                                                 \
static int check_##SUFFIX##_fortran_to_cblas(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                          \
    TYPE d[2] = { (TYPE)1, (TYPE)2 };                                             \
    TYPE e[1] = { (TYPE)3 };                                                      \
    TYPE df[2] = { 0, 0 };                                                        \
    TYPE ef[1] = { 0 };                                                           \
    TYPE b[4] = { (TYPE)11, (TYPE)12, (TYPE)13, (TYPE)14 };                      \
    TYPE b_original[4] = { (TYPE)11, (TYPE)12, (TYPE)13, (TYPE)14 };             \
    TYPE expected_b_in[4] = { (TYPE)11, (TYPE)13, (TYPE)12, (TYPE)14 };          \
    TYPE x[4] = { 0, 0, 0, 0 };                                                   \
    TYPE expected_x_out[4] = { (TYPE)((BASE) + 4), (TYPE)((BASE) + 6), (TYPE)((BASE) + 5), (TYPE)((BASE) + 7) }; \
    TYPE rcond = 0;                                                               \
    TYPE ferr[1] = { 0 };                                                         \
    TYPE berr[1] = { 0 };                                                         \
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
    info = thunk(FB_LAYOUT_ROW_MAJOR, 'N', 2, 2, d, e, df, ef, b, 2, x, 2,      \
                 &rcond, ferr, berr);                                             \
    if (info != (BASE) + 11 ||                                                    \
        g_##SUFFIX##_fortran_call.called != 1 ||                                  \
        g_##SUFFIX##_fortran_call.fact != 'N' ||                                  \
        g_##SUFFIX##_fortran_call.n != 2 ||                                       \
        g_##SUFFIX##_fortran_call.nrhs != 2 ||                                    \
        g_##SUFFIX##_fortran_call.d_snapshot[0] != (TYPE)1 ||                    \
        g_##SUFFIX##_fortran_call.e_snapshot[0] != (TYPE)3 ||                    \
        g_##SUFFIX##_fortran_call.df != df ||                                     \
        g_##SUFFIX##_fortran_call.ef != ef ||                                     \
        memcmp(g_##SUFFIX##_fortran_call.b_snapshot, expected_b_in, sizeof(expected_b_in)) != 0 || \
        g_##SUFFIX##_fortran_call.ldb != 2 ||                                     \
        g_##SUFFIX##_fortran_call.ldx != 2 ||                                     \
        g_##SUFFIX##_fortran_call.rcond != &rcond ||                              \
        g_##SUFFIX##_fortran_call.ferr != ferr ||                                 \
        g_##SUFFIX##_fortran_call.berr != berr ||                                 \
        !g_##SUFFIX##_fortran_call.work_nonnull ||                                \
        memcmp(b, b_original, sizeof(b_original)) != 0 ||                         \
        memcmp(x, expected_x_out, sizeof(expected_x_out)) != 0 ||                 \
        df[0] != (TYPE)((BASE) + 1) || df[1] != (TYPE)((BASE) + 2) ||             \
        ef[0] != (TYPE)((BASE) + 3) || rcond != (TYPE)((BASE) + 8) ||             \
        ferr[0] != (TYPE)((BASE) + 9) || berr[0] != (TYPE)((BASE) + 10)) {       \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not route PTSVX inputs correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk routes PTSVX inputs\n"); \
    return 0;                                                                     \
}                                                                                 \
static int check_##SUFFIX##_cblas_to_fortran(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                        \
    char fact = 'F';                                                              \
    int n = 2;                                                                    \
    int nrhs = 1;                                                                 \
    TYPE d[2] = { (TYPE)1, (TYPE)2 };                                             \
    TYPE e[1] = { (TYPE)3 };                                                      \
    TYPE df[2] = { (TYPE)4, (TYPE)5 };                                            \
    TYPE ef[1] = { (TYPE)6 };                                                     \
    TYPE b[2] = { (TYPE)7, (TYPE)8 };                                             \
    int ldb = 2;                                                                  \
    TYPE x[2] = { 0, 0 };                                                         \
    int ldx = 2;                                                                  \
    TYPE rcond = 0;                                                               \
    TYPE ferr[1] = { 0 };                                                         \
    TYPE berr[1] = { 0 };                                                         \
    TYPE work[2] = { 0, 0 };                                                      \
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
    thunk(&fact, &n, &nrhs, d, e, df, ef, b, &ldb, x, &ldx, &rcond, ferr, berr, work, &info); \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                    \
        g_##SUFFIX##_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||                  \
        g_##SUFFIX##_cblas_call.fact != 'F' ||                                    \
        g_##SUFFIX##_cblas_call.n != 2 ||                                         \
        g_##SUFFIX##_cblas_call.nrhs != 1 ||                                      \
        g_##SUFFIX##_cblas_call.d != d ||                                         \
        g_##SUFFIX##_cblas_call.e != e ||                                         \
        g_##SUFFIX##_cblas_call.df != df ||                                       \
        g_##SUFFIX##_cblas_call.ef != ef ||                                       \
        g_##SUFFIX##_cblas_call.b != b ||                                         \
        g_##SUFFIX##_cblas_call.ldb != 2 ||                                       \
        g_##SUFFIX##_cblas_call.x != x ||                                         \
        g_##SUFFIX##_cblas_call.ldx != 2 ||                                       \
        g_##SUFFIX##_cblas_call.rcond != &rcond ||                                \
        g_##SUFFIX##_cblas_call.ferr != ferr ||                                   \
        g_##SUFFIX##_cblas_call.berr != berr ||                                   \
        info != (BASE) + 18 || df[0] != (TYPE)((BASE) + 12) ||                    \
        ef[0] != (TYPE)((BASE) + 13) || x[0] != (TYPE)((BASE) + 14) ||            \
        rcond != (TYPE)((BASE) + 15) || ferr[0] != (TYPE)((BASE) + 16) ||        \
        berr[0] != (TYPE)((BASE) + 17)) {                                         \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not propagate PTSVX info correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk propagates PTSVX info\n"); \
    return 0;                                                                     \
}

#define DEFINE_PTSVX_COMPLEX_TESTS(SUFFIX, CTYPE, REAL_TYPE, OP_ID, BASE, MAKE, REAL_PART, IMAG_PART) \
typedef int (*fb_##SUFFIX##_cblas_fn)(fb_layout_t layout, char fact, int n,    \
                                      int nrhs, REAL_TYPE *d, CTYPE *e,        \
                                      REAL_TYPE *df, CTYPE *ef, CTYPE *b,      \
                                      int ldb, CTYPE *x, int ldx,              \
                                      REAL_TYPE *rcond, REAL_TYPE *ferr,       \
                                      REAL_TYPE *berr);                        \
typedef void (*fb_##SUFFIX##_fortran_fn)(char *fact, int *n, int *nrhs,       \
                                         REAL_TYPE *d, CTYPE *e, REAL_TYPE *df,\
                                         CTYPE *ef, CTYPE *b, int *ldb,        \
                                         CTYPE *x, int *ldx, REAL_TYPE *rcond, \
                                         REAL_TYPE *ferr, REAL_TYPE *berr,     \
                                         CTYPE *work, REAL_TYPE *rwork,        \
                                         int *info);                           \
static struct {                                                                   \
    int called;                                                                   \
    char fact;                                                                    \
    int n;                                                                        \
    int nrhs;                                                                     \
    REAL_TYPE d_snapshot[2];                                                      \
    REAL_TYPE e_real_snapshot[1];                                                 \
    REAL_TYPE e_imag_snapshot[1];                                                 \
    REAL_TYPE *df;                                                                \
    CTYPE *ef;                                                                    \
    REAL_TYPE b_real_snapshot[4];                                                 \
    REAL_TYPE b_imag_snapshot[4];                                                 \
    int ldb;                                                                      \
    int ldx;                                                                      \
    REAL_TYPE *rcond;                                                             \
    REAL_TYPE *ferr;                                                              \
    REAL_TYPE *berr;                                                              \
    int work_nonnull;                                                             \
    int rwork_nonnull;                                                            \
} g_##SUFFIX##_fortran_call;                                                      \
static struct {                                                                   \
    int called;                                                                   \
    fb_layout_t layout;                                                           \
    char fact;                                                                    \
    int n;                                                                        \
    int nrhs;                                                                     \
    REAL_TYPE *d;                                                                 \
    CTYPE *e;                                                                     \
    REAL_TYPE *df;                                                                \
    CTYPE *ef;                                                                    \
    CTYPE *b;                                                                     \
    int ldb;                                                                      \
    CTYPE *x;                                                                     \
    int ldx;                                                                      \
    REAL_TYPE *rcond;                                                             \
    REAL_TYPE *ferr;                                                              \
    REAL_TYPE *berr;                                                              \
} g_##SUFFIX##_cblas_call;                                                        \
static void stub_##SUFFIX##_fortran(char *fact, int *n, int *nrhs,             \
                                    REAL_TYPE *d, CTYPE *e, REAL_TYPE *df,     \
                                    CTYPE *ef, CTYPE *b, int *ldb, CTYPE *x,   \
                                    int *ldx, REAL_TYPE *rcond, REAL_TYPE *ferr,\
                                    REAL_TYPE *berr, CTYPE *work,              \
                                    REAL_TYPE *rwork, int *info)               \
{                                                                                 \
    size_t index = 0;                                                             \
    g_##SUFFIX##_fortran_call.called += 1;                                        \
    g_##SUFFIX##_fortran_call.fact = *fact;                                       \
    g_##SUFFIX##_fortran_call.n = *n;                                             \
    g_##SUFFIX##_fortran_call.nrhs = *nrhs;                                       \
    memcpy(g_##SUFFIX##_fortran_call.d_snapshot, d, (size_t)(*n) * sizeof(REAL_TYPE)); \
    g_##SUFFIX##_fortran_call.e_real_snapshot[0] = REAL_PART(e[0]);               \
    g_##SUFFIX##_fortran_call.e_imag_snapshot[0] = IMAG_PART(e[0]);               \
    g_##SUFFIX##_fortran_call.df = df;                                            \
    g_##SUFFIX##_fortran_call.ef = ef;                                            \
    for (index = 0; index < (size_t)(*ldb) * (size_t)(*nrhs); ++index) {         \
        g_##SUFFIX##_fortran_call.b_real_snapshot[index] = REAL_PART(b[index]);  \
        g_##SUFFIX##_fortran_call.b_imag_snapshot[index] = IMAG_PART(b[index]);  \
    }                                                                             \
    g_##SUFFIX##_fortran_call.ldb = *ldb;                                         \
    g_##SUFFIX##_fortran_call.ldx = *ldx;                                         \
    g_##SUFFIX##_fortran_call.rcond = rcond;                                      \
    g_##SUFFIX##_fortran_call.ferr = ferr;                                        \
    g_##SUFFIX##_fortran_call.berr = berr;                                        \
    g_##SUFFIX##_fortran_call.work_nonnull = (work != NULL);                      \
    g_##SUFFIX##_fortran_call.rwork_nonnull = (rwork != NULL);                    \
    df[0] = (REAL_TYPE)((BASE) + 1);                                              \
    df[1] = (REAL_TYPE)((BASE) + 2);                                              \
    ef[0] = MAKE((REAL_TYPE)((BASE) + 3), (REAL_TYPE)((BASE) + 30));             \
    x[0] = MAKE((REAL_TYPE)((BASE) + 4), (REAL_TYPE)((BASE) + 40));              \
    x[1] = MAKE((REAL_TYPE)((BASE) + 5), (REAL_TYPE)((BASE) + 50));              \
    x[2] = MAKE((REAL_TYPE)((BASE) + 6), (REAL_TYPE)((BASE) + 60));              \
    x[3] = MAKE((REAL_TYPE)((BASE) + 7), (REAL_TYPE)((BASE) + 70));              \
    *rcond = (REAL_TYPE)((BASE) + 8);                                             \
    ferr[0] = (REAL_TYPE)((BASE) + 9);                                            \
    berr[0] = (REAL_TYPE)((BASE) + 10);                                           \
    *info = (BASE) + 11;                                                          \
}                                                                                 \
static int stub_##SUFFIX##_cblas(fb_layout_t layout, char fact, int n, int nrhs,\
                                 REAL_TYPE *d, CTYPE *e, REAL_TYPE *df,        \
                                 CTYPE *ef, CTYPE *b, int ldb, CTYPE *x,       \
                                 int ldx, REAL_TYPE *rcond, REAL_TYPE *ferr,   \
                                 REAL_TYPE *berr)                              \
{                                                                                 \
    g_##SUFFIX##_cblas_call.called += 1;                                          \
    g_##SUFFIX##_cblas_call.layout = layout;                                      \
    g_##SUFFIX##_cblas_call.fact = fact;                                          \
    g_##SUFFIX##_cblas_call.n = n;                                                \
    g_##SUFFIX##_cblas_call.nrhs = nrhs;                                          \
    g_##SUFFIX##_cblas_call.d = d;                                                \
    g_##SUFFIX##_cblas_call.e = e;                                                \
    g_##SUFFIX##_cblas_call.df = df;                                              \
    g_##SUFFIX##_cblas_call.ef = ef;                                              \
    g_##SUFFIX##_cblas_call.b = b;                                                \
    g_##SUFFIX##_cblas_call.ldb = ldb;                                            \
    g_##SUFFIX##_cblas_call.x = x;                                                \
    g_##SUFFIX##_cblas_call.ldx = ldx;                                            \
    g_##SUFFIX##_cblas_call.rcond = rcond;                                        \
    g_##SUFFIX##_cblas_call.ferr = ferr;                                          \
    g_##SUFFIX##_cblas_call.berr = berr;                                          \
    df[0] = (REAL_TYPE)((BASE) + 12);                                             \
    ef[0] = MAKE((REAL_TYPE)((BASE) + 13), (REAL_TYPE)((BASE) + 130));           \
    x[0] = MAKE((REAL_TYPE)((BASE) + 14), (REAL_TYPE)((BASE) + 140));            \
    *rcond = (REAL_TYPE)((BASE) + 15);                                            \
    ferr[0] = (REAL_TYPE)((BASE) + 16);                                           \
    berr[0] = (REAL_TYPE)((BASE) + 17);                                           \
    return (BASE) + 18;                                                           \
}                                                                                 \
static int check_##SUFFIX##_fortran_to_cblas(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                          \
    REAL_TYPE d[2] = { (REAL_TYPE)1, (REAL_TYPE)2 };                             \
    CTYPE e[1] = { MAKE((REAL_TYPE)3, (REAL_TYPE)30) };                          \
    REAL_TYPE df[2] = { 0, 0 };                                                   \
    CTYPE ef[1] = { MAKE((REAL_TYPE)0, (REAL_TYPE)0) };                          \
    CTYPE b[4] = { MAKE((REAL_TYPE)11, (REAL_TYPE)110), MAKE((REAL_TYPE)12, (REAL_TYPE)120), MAKE((REAL_TYPE)13, (REAL_TYPE)130), MAKE((REAL_TYPE)14, (REAL_TYPE)140) }; \
    CTYPE b_original[4] = { MAKE((REAL_TYPE)11, (REAL_TYPE)110), MAKE((REAL_TYPE)12, (REAL_TYPE)120), MAKE((REAL_TYPE)13, (REAL_TYPE)130), MAKE((REAL_TYPE)14, (REAL_TYPE)140) }; \
    CTYPE x[4] = { MAKE((REAL_TYPE)0, (REAL_TYPE)0), MAKE((REAL_TYPE)0, (REAL_TYPE)0), MAKE((REAL_TYPE)0, (REAL_TYPE)0), MAKE((REAL_TYPE)0, (REAL_TYPE)0) }; \
    REAL_TYPE rcond = 0;                                                          \
    REAL_TYPE ferr[1] = { 0 };                                                    \
    REAL_TYPE berr[1] = { 0 };                                                    \
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
    info = thunk(FB_LAYOUT_ROW_MAJOR, 'N', 2, 2, d, e, df, ef, b, 2, x, 2,      \
                 &rcond, ferr, berr);                                             \
    if (info != (BASE) + 11 ||                                                    \
        g_##SUFFIX##_fortran_call.called != 1 ||                                  \
        g_##SUFFIX##_fortran_call.fact != 'N' ||                                  \
        g_##SUFFIX##_fortran_call.n != 2 ||                                       \
        g_##SUFFIX##_fortran_call.nrhs != 2 ||                                    \
        g_##SUFFIX##_fortran_call.d_snapshot[0] != (REAL_TYPE)1 ||               \
        g_##SUFFIX##_fortran_call.e_real_snapshot[0] != (REAL_TYPE)3 ||          \
        g_##SUFFIX##_fortran_call.e_imag_snapshot[0] != (REAL_TYPE)30 ||         \
        g_##SUFFIX##_fortran_call.df != df ||                                     \
        g_##SUFFIX##_fortran_call.ef != ef ||                                     \
        g_##SUFFIX##_fortran_call.b_real_snapshot[0] != (REAL_TYPE)11 ||         \
        g_##SUFFIX##_fortran_call.b_real_snapshot[1] != (REAL_TYPE)13 ||         \
        g_##SUFFIX##_fortran_call.ldb != 2 ||                                     \
        g_##SUFFIX##_fortran_call.ldx != 2 ||                                     \
        g_##SUFFIX##_fortran_call.rcond != &rcond ||                              \
        g_##SUFFIX##_fortran_call.ferr != ferr ||                                 \
        g_##SUFFIX##_fortran_call.berr != berr ||                                 \
        !g_##SUFFIX##_fortran_call.work_nonnull ||                                \
        !g_##SUFFIX##_fortran_call.rwork_nonnull ||                               \
        REAL_PART(b[0]) != REAL_PART(b_original[0]) ||                            \
        IMAG_PART(b[0]) != IMAG_PART(b_original[0]) ||                            \
        REAL_PART(x[0]) != (REAL_TYPE)((BASE) + 4) ||                             \
        IMAG_PART(x[0]) != (REAL_TYPE)((BASE) + 40) ||                            \
        REAL_PART(x[1]) != (REAL_TYPE)((BASE) + 6) ||                             \
        IMAG_PART(x[1]) != (REAL_TYPE)((BASE) + 60) ||                            \
        REAL_PART(x[2]) != (REAL_TYPE)((BASE) + 5) ||                             \
        IMAG_PART(x[2]) != (REAL_TYPE)((BASE) + 50) ||                            \
        df[0] != (REAL_TYPE)((BASE) + 1) || df[1] != (REAL_TYPE)((BASE) + 2) ||  \
        REAL_PART(ef[0]) != (REAL_TYPE)((BASE) + 3) ||                            \
        IMAG_PART(ef[0]) != (REAL_TYPE)((BASE) + 30) ||                           \
        rcond != (REAL_TYPE)((BASE) + 8) || ferr[0] != (REAL_TYPE)((BASE) + 9) || \
        berr[0] != (REAL_TYPE)((BASE) + 10)) {                                    \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not route complex PTSVX inputs correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk routes complex PTSVX inputs\n"); \
    return 0;                                                                     \
}                                                                                 \
static int check_##SUFFIX##_cblas_to_fortran(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                        \
    char fact = 'F';                                                              \
    int n = 2;                                                                    \
    int nrhs = 1;                                                                 \
    REAL_TYPE d[2] = { (REAL_TYPE)1, (REAL_TYPE)2 };                             \
    CTYPE e[1] = { MAKE((REAL_TYPE)3, (REAL_TYPE)30) };                          \
    REAL_TYPE df[2] = { (REAL_TYPE)4, (REAL_TYPE)5 };                            \
    CTYPE ef[1] = { MAKE((REAL_TYPE)6, (REAL_TYPE)60) };                         \
    CTYPE b[2] = { MAKE((REAL_TYPE)7, (REAL_TYPE)70), MAKE((REAL_TYPE)8, (REAL_TYPE)80) }; \
    int ldb = 2;                                                                  \
    CTYPE x[2] = { MAKE((REAL_TYPE)0, (REAL_TYPE)0), MAKE((REAL_TYPE)0, (REAL_TYPE)0) }; \
    int ldx = 2;                                                                  \
    REAL_TYPE rcond = 0;                                                          \
    REAL_TYPE ferr[1] = { 0 };                                                    \
    REAL_TYPE berr[1] = { 0 };                                                    \
    CTYPE work[4] = { MAKE((REAL_TYPE)0, (REAL_TYPE)0), MAKE((REAL_TYPE)0, (REAL_TYPE)0), MAKE((REAL_TYPE)0, (REAL_TYPE)0), MAKE((REAL_TYPE)0, (REAL_TYPE)0) }; \
    REAL_TYPE rwork[2] = { 0, 0 };                                                \
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
    thunk(&fact, &n, &nrhs, d, e, df, ef, b, &ldb, x, &ldx, &rcond, ferr, berr, work, rwork, &info); \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                    \
        g_##SUFFIX##_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||                  \
        g_##SUFFIX##_cblas_call.fact != 'F' ||                                    \
        g_##SUFFIX##_cblas_call.n != 2 ||                                         \
        g_##SUFFIX##_cblas_call.nrhs != 1 ||                                      \
        g_##SUFFIX##_cblas_call.d != d ||                                         \
        g_##SUFFIX##_cblas_call.e != e ||                                         \
        g_##SUFFIX##_cblas_call.df != df ||                                       \
        g_##SUFFIX##_cblas_call.ef != ef ||                                       \
        g_##SUFFIX##_cblas_call.b != b ||                                         \
        g_##SUFFIX##_cblas_call.ldb != 2 ||                                       \
        g_##SUFFIX##_cblas_call.x != x ||                                         \
        g_##SUFFIX##_cblas_call.ldx != 2 ||                                       \
        g_##SUFFIX##_cblas_call.rcond != &rcond ||                                \
        g_##SUFFIX##_cblas_call.ferr != ferr ||                                   \
        g_##SUFFIX##_cblas_call.berr != berr ||                                   \
        info != (BASE) + 18 || df[0] != (REAL_TYPE)((BASE) + 12) ||               \
        REAL_PART(ef[0]) != (REAL_TYPE)((BASE) + 13) ||                           \
        IMAG_PART(ef[0]) != (REAL_TYPE)((BASE) + 130) ||                          \
        REAL_PART(x[0]) != (REAL_TYPE)((BASE) + 14) ||                            \
        IMAG_PART(x[0]) != (REAL_TYPE)((BASE) + 140) ||                           \
        rcond != (REAL_TYPE)((BASE) + 15) || ferr[0] != (REAL_TYPE)((BASE) + 16) || \
        berr[0] != (REAL_TYPE)((BASE) + 17)) {                                    \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not propagate complex PTSVX info correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk propagates complex PTSVX info\n"); \
    return 0;                                                                     \
}

DEFINE_PTSVX_REAL_TESTS(sptsvx, float, FB_OP_SPTSVX, 100)
DEFINE_PTSVX_REAL_TESTS(dptsvx, double, FB_OP_DPTSVX, 300)
DEFINE_PTSVX_COMPLEX_TESTS(cptsvx, fb_complex_float_t, float, FB_OP_CPTSVX, 500, make_cfloat, cfloat_real, cfloat_imag)
DEFINE_PTSVX_COMPLEX_TESTS(zptsvx, fb_complex_double_t, double, FB_OP_ZPTSVX, 700, make_cdouble, cdouble_real, cdouble_imag)

int main(void)
{
    int status = 0;

    status |= check_sptsvx_fortran_to_cblas();
    status |= check_sptsvx_cblas_to_fortran();
    status |= check_dptsvx_fortran_to_cblas();
    status |= check_dptsvx_cblas_to_fortran();
    status |= check_cptsvx_fortran_to_cblas();
    status |= check_cptsvx_cblas_to_fortran();
    status |= check_zptsvx_fortran_to_cblas();
    status |= check_zptsvx_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}