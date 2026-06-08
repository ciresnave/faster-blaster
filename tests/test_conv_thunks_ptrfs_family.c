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

#define DEFINE_PTRFS_REAL_TESTS(SUFFIX, TYPE, OP_ID, BASE)                       \
typedef int (*fb_##SUFFIX##_cblas_fn)(fb_layout_t layout, char uplo,           \
                                      char trans, char diag, int n, int nrhs,  \
                                      const TYPE *ap, const TYPE *b, int ldb,  \
                                      const TYPE *x, int ldx, TYPE *ferr,      \
                                      TYPE *berr);                             \
typedef void (*fb_##SUFFIX##_fortran_fn)(char *uplo, char *trans, char *diag, \
                                         int *n, int *nrhs, const TYPE *ap,    \
                                         const TYPE *b, int *ldb, TYPE *x,     \
                                         int *ldx, TYPE *ferr, TYPE *berr,     \
                                         TYPE *work, int *info);               \
static struct {                                                                   \
    int called;                                                                   \
    char uplo;                                                                    \
    char trans;                                                                   \
    char diag;                                                                    \
    int n;                                                                        \
    int nrhs;                                                                     \
    TYPE ap_snapshot[6];                                                          \
    TYPE b_snapshot[6];                                                           \
    TYPE x_snapshot[6];                                                           \
    int ldb;                                                                      \
    int ldx;                                                                      \
    TYPE *ferr;                                                                   \
    TYPE *berr;                                                                   \
    int work_nonnull;                                                             \
} g_##SUFFIX##_fortran_call;                                                      \
static struct {                                                                   \
    int called;                                                                   \
    fb_layout_t layout;                                                           \
    char uplo;                                                                    \
    char trans;                                                                   \
    char diag;                                                                    \
    int n;                                                                        \
    int nrhs;                                                                     \
    const TYPE *ap;                                                               \
    const TYPE *b;                                                                \
    int ldb;                                                                      \
    const TYPE *x;                                                                \
    int ldx;                                                                      \
    TYPE *ferr;                                                                   \
    TYPE *berr;                                                                   \
} g_##SUFFIX##_cblas_call;                                                        \
static void stub_##SUFFIX##_fortran(char *uplo, char *trans, char *diag,       \
                                    int *n, int *nrhs, const TYPE *ap,         \
                                    const TYPE *b, int *ldb, TYPE *x,          \
                                    int *ldx, TYPE *ferr, TYPE *berr,          \
                                    TYPE *work, int *info)                     \
{                                                                                 \
    g_##SUFFIX##_fortran_call.called += 1;                                        \
    g_##SUFFIX##_fortran_call.uplo = *uplo;                                       \
    g_##SUFFIX##_fortran_call.trans = *trans;                                     \
    g_##SUFFIX##_fortran_call.diag = *diag;                                       \
    g_##SUFFIX##_fortran_call.n = *n;                                             \
    g_##SUFFIX##_fortran_call.nrhs = *nrhs;                                       \
    memcpy(g_##SUFFIX##_fortran_call.ap_snapshot, ap, (size_t)((*n) * ((*n) + 1) / 2) * sizeof(TYPE)); \
    memcpy(g_##SUFFIX##_fortran_call.b_snapshot, b, (size_t)(*ldb) * (size_t)(*nrhs) * sizeof(TYPE)); \
    memcpy(g_##SUFFIX##_fortran_call.x_snapshot, x, (size_t)(*ldx) * (size_t)(*nrhs) * sizeof(TYPE)); \
    g_##SUFFIX##_fortran_call.ldb = *ldb;                                         \
    g_##SUFFIX##_fortran_call.ldx = *ldx;                                         \
    g_##SUFFIX##_fortran_call.ferr = ferr;                                        \
    g_##SUFFIX##_fortran_call.berr = berr;                                        \
    g_##SUFFIX##_fortran_call.work_nonnull = (work != NULL);                      \
    ferr[0] = (TYPE)((BASE) + 1);                                                 \
    berr[0] = (TYPE)((BASE) + 2);                                                 \
    *info = (BASE) + 3;                                                           \
}                                                                                 \
static int stub_##SUFFIX##_cblas(fb_layout_t layout, char uplo, char trans,     \
                                 char diag, int n, int nrhs, const TYPE *ap,   \
                                 const TYPE *b, int ldb, const TYPE *x,        \
                                 int ldx, TYPE *ferr, TYPE *berr)              \
{                                                                                 \
    g_##SUFFIX##_cblas_call.called += 1;                                          \
    g_##SUFFIX##_cblas_call.layout = layout;                                      \
    g_##SUFFIX##_cblas_call.uplo = uplo;                                          \
    g_##SUFFIX##_cblas_call.trans = trans;                                        \
    g_##SUFFIX##_cblas_call.diag = diag;                                          \
    g_##SUFFIX##_cblas_call.n = n;                                                \
    g_##SUFFIX##_cblas_call.nrhs = nrhs;                                          \
    g_##SUFFIX##_cblas_call.ap = ap;                                              \
    g_##SUFFIX##_cblas_call.b = b;                                                \
    g_##SUFFIX##_cblas_call.ldb = ldb;                                            \
    g_##SUFFIX##_cblas_call.x = x;                                                \
    g_##SUFFIX##_cblas_call.ldx = ldx;                                            \
    g_##SUFFIX##_cblas_call.ferr = ferr;                                          \
    g_##SUFFIX##_cblas_call.berr = berr;                                          \
    ferr[0] = (TYPE)((BASE) + 4);                                                 \
    berr[0] = (TYPE)((BASE) + 5);                                                 \
    return (BASE) + 6;                                                            \
}                                                                                 \
static int check_##SUFFIX##_fortran_to_cblas(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                          \
    TYPE ap[6] = { (TYPE)11, (TYPE)12, (TYPE)13, (TYPE)22, (TYPE)23, (TYPE)33 }; \
    TYPE expected_ap[6] = { (TYPE)11, (TYPE)12, (TYPE)22, (TYPE)13, (TYPE)23, (TYPE)33 }; \
    TYPE b[6] = { (TYPE)1, (TYPE)2, (TYPE)3, (TYPE)4, (TYPE)5, (TYPE)6 };       \
    TYPE expected_b[6] = { (TYPE)1, (TYPE)3, (TYPE)5, (TYPE)2, (TYPE)4, (TYPE)6 }; \
    TYPE x[6] = { (TYPE)7, (TYPE)8, (TYPE)9, (TYPE)10, (TYPE)11, (TYPE)12 };    \
    TYPE expected_x[6] = { (TYPE)7, (TYPE)9, (TYPE)11, (TYPE)8, (TYPE)10, (TYPE)12 }; \
    TYPE ap_original[6] = { (TYPE)11, (TYPE)12, (TYPE)13, (TYPE)22, (TYPE)23, (TYPE)33 }; \
    TYPE b_original[6] = { (TYPE)1, (TYPE)2, (TYPE)3, (TYPE)4, (TYPE)5, (TYPE)6 }; \
    TYPE x_original[6] = { (TYPE)7, (TYPE)8, (TYPE)9, (TYPE)10, (TYPE)11, (TYPE)12 }; \
    TYPE ferr[1] = { 0 };                                                          \
    TYPE berr[1] = { 0 };                                                          \
    int info = 0;                                                                  \
    memset(&vtable, 0, sizeof(vtable));                                            \
    memset(&g_##SUFFIX##_fortran_call, 0, sizeof(g_##SUFFIX##_fortran_call));      \
    vtable.ext_ops[OP_ID][FB_CONV_FORTRAN] =                                       \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_fortran;                    \
    fb_install_conv_thunks(&vtable, OP_ID);                                        \
    thunk = (fb_##SUFFIX##_cblas_fn)vtable.ext_ops[OP_ID][FB_CONV_CBLAS];          \
    if (!thunk) {                                                                  \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk was not installed\n"); \
        return 1;                                                                  \
    }                                                                              \
    info = thunk(FB_LAYOUT_ROW_MAJOR, 'U', 'N', 'N', 3, 2, ap, b, 2, x, 2, ferr, berr); \
    if (info != (BASE) + 3 ||                                                      \
        g_##SUFFIX##_fortran_call.called != 1 ||                                   \
        g_##SUFFIX##_fortran_call.uplo != 'U' ||                                   \
        g_##SUFFIX##_fortran_call.trans != 'N' ||                                  \
        g_##SUFFIX##_fortran_call.diag != 'N' ||                                   \
        g_##SUFFIX##_fortran_call.n != 3 ||                                        \
        g_##SUFFIX##_fortran_call.nrhs != 2 ||                                     \
        memcmp(g_##SUFFIX##_fortran_call.ap_snapshot, expected_ap, sizeof(expected_ap)) != 0 || \
        memcmp(g_##SUFFIX##_fortran_call.b_snapshot, expected_b, sizeof(expected_b)) != 0 || \
        memcmp(g_##SUFFIX##_fortran_call.x_snapshot, expected_x, sizeof(expected_x)) != 0 || \
        g_##SUFFIX##_fortran_call.ldb != 3 ||                                      \
        g_##SUFFIX##_fortran_call.ldx != 3 ||                                      \
        g_##SUFFIX##_fortran_call.ferr != ferr ||                                  \
        g_##SUFFIX##_fortran_call.berr != berr ||                                  \
        !g_##SUFFIX##_fortran_call.work_nonnull ||                                 \
        memcmp(ap, ap_original, sizeof(ap_original)) != 0 ||                      \
        memcmp(b, b_original, sizeof(b_original)) != 0 ||                         \
        memcmp(x, x_original, sizeof(x_original)) != 0 ||                         \
        ferr[0] != (TYPE)((BASE) + 1) || berr[0] != (TYPE)((BASE) + 2)) {        \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not route real PTRFS inputs correctly\n"); \
        return 1;                                                                  \
    }                                                                              \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk routes real PTRFS inputs\n"); \
    return 0;                                                                      \
}                                                                                  \
static int check_##SUFFIX##_cblas_to_fortran(void)                                 \
{                                                                                  \
    fb_backend_vtable_t vtable;                                                    \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                         \
    char uplo = 'L';                                                               \
    char trans = 'T';                                                              \
    char diag = 'U';                                                               \
    int n = 2;                                                                     \
    int nrhs = 1;                                                                  \
    TYPE ap[3] = { (TYPE)9, (TYPE)10, (TYPE)11 };                                 \
    TYPE b[2] = { (TYPE)12, (TYPE)13 };                                            \
    int ldb = 2;                                                                   \
    TYPE x[2] = { (TYPE)14, (TYPE)15 };                                            \
    int ldx = 2;                                                                   \
    TYPE ferr[1] = { 0 };                                                          \
    TYPE berr[1] = { 0 };                                                          \
    TYPE work[2] = { 0, 0 };                                                       \
    int info = -999;                                                               \
    memset(&vtable, 0, sizeof(vtable));                                            \
    memset(&g_##SUFFIX##_cblas_call, 0, sizeof(g_##SUFFIX##_cblas_call));          \
    vtable.ext_ops[OP_ID][FB_CONV_CBLAS] =                                         \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_cblas;                      \
    fb_install_conv_thunks(&vtable, OP_ID);                                        \
    thunk = (fb_##SUFFIX##_fortran_fn)vtable.ext_ops[OP_ID][FB_CONV_FORTRAN];     \
    if (!thunk) {                                                                  \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk was not installed\n"); \
        return 1;                                                                  \
    }                                                                              \
    thunk(&uplo, &trans, &diag, &n, &nrhs, ap, b, &ldb, x, &ldx, ferr, berr, work, &info); \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                     \
        g_##SUFFIX##_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||                   \
        g_##SUFFIX##_cblas_call.uplo != 'L' ||                                     \
        g_##SUFFIX##_cblas_call.trans != 'T' ||                                    \
        g_##SUFFIX##_cblas_call.diag != 'U' ||                                     \
        g_##SUFFIX##_cblas_call.n != 2 ||                                          \
        g_##SUFFIX##_cblas_call.nrhs != 1 ||                                       \
        g_##SUFFIX##_cblas_call.ap != ap ||                                        \
        g_##SUFFIX##_cblas_call.b != b ||                                          \
        g_##SUFFIX##_cblas_call.ldb != 2 ||                                        \
        g_##SUFFIX##_cblas_call.x != x ||                                          \
        g_##SUFFIX##_cblas_call.ldx != 2 ||                                        \
        g_##SUFFIX##_cblas_call.ferr != ferr ||                                    \
        g_##SUFFIX##_cblas_call.berr != berr ||                                    \
        info != (BASE) + 6 || ferr[0] != (TYPE)((BASE) + 4) ||                    \
        berr[0] != (TYPE)((BASE) + 5)) {                                           \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not propagate real PTRFS info correctly\n"); \
        return 1;                                                                  \
    }                                                                              \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk propagates real PTRFS info\n"); \
    return 0;                                                                      \
}

#define DEFINE_PTRFS_COMPLEX_TESTS(SUFFIX, CTYPE, REAL_TYPE, OP_ID, BASE, MAKE, REAL_PART, IMAG_PART) \
typedef int (*fb_##SUFFIX##_cblas_fn)(fb_layout_t layout, fb_uplo_t uplo,      \
                                      int n, int nrhs, const REAL_TYPE *d,     \
                                      const CTYPE *e, const REAL_TYPE *df,     \
                                      const CTYPE *ef, const CTYPE *b, int ldb,\
                                      const CTYPE *x, int ldx, REAL_TYPE *ferr,\
                                      REAL_TYPE *berr);                        \
typedef void (*fb_##SUFFIX##_fortran_fn)(char *uplo, int *n, int *nrhs,       \
                                         REAL_TYPE *d, CTYPE *e, REAL_TYPE *df,\
                                         CTYPE *ef, CTYPE *b, int *ldb,        \
                                         CTYPE *x, int *ldx, REAL_TYPE *ferr,  \
                                         REAL_TYPE *berr, CTYPE *work,         \
                                         REAL_TYPE *rwork, int *info);         \
static struct {                                                                   \
    int called;                                                                   \
    char uplo;                                                                    \
    int n;                                                                        \
    int nrhs;                                                                     \
    REAL_TYPE d_snapshot[2];                                                      \
    REAL_TYPE e_real_snapshot[1];                                                 \
    REAL_TYPE e_imag_snapshot[1];                                                 \
    REAL_TYPE df_snapshot[2];                                                     \
    REAL_TYPE ef_real_snapshot[1];                                                \
    REAL_TYPE ef_imag_snapshot[1];                                                \
    REAL_TYPE b_real_snapshot[4];                                                 \
    REAL_TYPE b_imag_snapshot[4];                                                 \
    REAL_TYPE x_real_snapshot[4];                                                 \
    REAL_TYPE x_imag_snapshot[4];                                                 \
    int ldb;                                                                      \
    int ldx;                                                                      \
    REAL_TYPE *ferr;                                                              \
    REAL_TYPE *berr;                                                              \
    int work_nonnull;                                                             \
    int rwork_nonnull;                                                            \
} g_##SUFFIX##_fortran_call;                                                      \
static struct {                                                                   \
    int called;                                                                   \
    fb_layout_t layout;                                                           \
    fb_uplo_t uplo;                                                               \
    int n;                                                                        \
    int nrhs;                                                                     \
    const REAL_TYPE *d;                                                           \
    const CTYPE *e;                                                               \
    const REAL_TYPE *df;                                                          \
    const CTYPE *ef;                                                              \
    const CTYPE *b;                                                               \
    int ldb;                                                                      \
    const CTYPE *x;                                                               \
    int ldx;                                                                      \
    REAL_TYPE *ferr;                                                              \
    REAL_TYPE *berr;                                                              \
} g_##SUFFIX##_cblas_call;                                                        \
static void stub_##SUFFIX##_fortran(char *uplo, int *n, int *nrhs,             \
                                    REAL_TYPE *d, CTYPE *e, REAL_TYPE *df,     \
                                    CTYPE *ef, CTYPE *b, int *ldb, CTYPE *x,   \
                                    int *ldx, REAL_TYPE *ferr, REAL_TYPE *berr,\
                                    CTYPE *work, REAL_TYPE *rwork, int *info)  \
{                                                                                 \
    size_t index = 0;                                                             \
    g_##SUFFIX##_fortran_call.called += 1;                                        \
    g_##SUFFIX##_fortran_call.uplo = *uplo;                                       \
    g_##SUFFIX##_fortran_call.n = *n;                                             \
    g_##SUFFIX##_fortran_call.nrhs = *nrhs;                                       \
    memcpy(g_##SUFFIX##_fortran_call.d_snapshot, d, (size_t)(*n) * sizeof(REAL_TYPE)); \
    memcpy(g_##SUFFIX##_fortran_call.df_snapshot, df, (size_t)(*n) * sizeof(REAL_TYPE)); \
    g_##SUFFIX##_fortran_call.e_real_snapshot[0] = REAL_PART(e[0]);               \
    g_##SUFFIX##_fortran_call.e_imag_snapshot[0] = IMAG_PART(e[0]);               \
    g_##SUFFIX##_fortran_call.ef_real_snapshot[0] = REAL_PART(ef[0]);             \
    g_##SUFFIX##_fortran_call.ef_imag_snapshot[0] = IMAG_PART(ef[0]);             \
    for (index = 0; index < (size_t)(*ldb) * (size_t)(*nrhs); ++index) {         \
        g_##SUFFIX##_fortran_call.b_real_snapshot[index] = REAL_PART(b[index]);  \
        g_##SUFFIX##_fortran_call.b_imag_snapshot[index] = IMAG_PART(b[index]);  \
        g_##SUFFIX##_fortran_call.x_real_snapshot[index] = REAL_PART(x[index]);  \
        g_##SUFFIX##_fortran_call.x_imag_snapshot[index] = IMAG_PART(x[index]);  \
    }                                                                             \
    g_##SUFFIX##_fortran_call.ldb = *ldb;                                         \
    g_##SUFFIX##_fortran_call.ldx = *ldx;                                         \
    g_##SUFFIX##_fortran_call.ferr = ferr;                                        \
    g_##SUFFIX##_fortran_call.berr = berr;                                        \
    g_##SUFFIX##_fortran_call.work_nonnull = (work != NULL);                      \
    g_##SUFFIX##_fortran_call.rwork_nonnull = (rwork != NULL);                    \
    ferr[0] = (REAL_TYPE)((BASE) + 1);                                            \
    berr[0] = (REAL_TYPE)((BASE) + 2);                                            \
    *info = (BASE) + 3;                                                           \
}                                                                                 \
static int stub_##SUFFIX##_cblas(fb_layout_t layout, fb_uplo_t uplo, int n,    \
                                 int nrhs, const REAL_TYPE *d, const CTYPE *e, \
                                 const REAL_TYPE *df, const CTYPE *ef,         \
                                 const CTYPE *b, int ldb, const CTYPE *x,      \
                                 int ldx, REAL_TYPE *ferr, REAL_TYPE *berr)    \
{                                                                                 \
    g_##SUFFIX##_cblas_call.called += 1;                                          \
    g_##SUFFIX##_cblas_call.layout = layout;                                      \
    g_##SUFFIX##_cblas_call.uplo = uplo;                                          \
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
    g_##SUFFIX##_cblas_call.ferr = ferr;                                          \
    g_##SUFFIX##_cblas_call.berr = berr;                                          \
    ferr[0] = (REAL_TYPE)((BASE) + 4);                                            \
    berr[0] = (REAL_TYPE)((BASE) + 5);                                            \
    return (BASE) + 6;                                                            \
}                                                                                 \
static int check_##SUFFIX##_fortran_to_cblas(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                          \
    REAL_TYPE d[2] = { (REAL_TYPE)1, (REAL_TYPE)2 };                             \
    CTYPE e[1] = { MAKE((REAL_TYPE)3, (REAL_TYPE)30) };                          \
    REAL_TYPE df[2] = { (REAL_TYPE)4, (REAL_TYPE)5 };                            \
    CTYPE ef[1] = { MAKE((REAL_TYPE)6, (REAL_TYPE)60) };                         \
    CTYPE b[4] = { MAKE((REAL_TYPE)11, (REAL_TYPE)110), MAKE((REAL_TYPE)12, (REAL_TYPE)120), MAKE((REAL_TYPE)13, (REAL_TYPE)130), MAKE((REAL_TYPE)14, (REAL_TYPE)140) }; \
    CTYPE x[4] = { MAKE((REAL_TYPE)21, (REAL_TYPE)210), MAKE((REAL_TYPE)22, (REAL_TYPE)220), MAKE((REAL_TYPE)23, (REAL_TYPE)230), MAKE((REAL_TYPE)24, (REAL_TYPE)240) }; \
    CTYPE b_original[4] = { MAKE((REAL_TYPE)11, (REAL_TYPE)110), MAKE((REAL_TYPE)12, (REAL_TYPE)120), MAKE((REAL_TYPE)13, (REAL_TYPE)130), MAKE((REAL_TYPE)14, (REAL_TYPE)140) }; \
    CTYPE x_original[4] = { MAKE((REAL_TYPE)21, (REAL_TYPE)210), MAKE((REAL_TYPE)22, (REAL_TYPE)220), MAKE((REAL_TYPE)23, (REAL_TYPE)230), MAKE((REAL_TYPE)24, (REAL_TYPE)240) }; \
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
    info = thunk(FB_LAYOUT_ROW_MAJOR, FB_LOWER, 2, 2, d, e, df, ef, b, 2, x, 2, ferr, berr); \
    if (info != (BASE) + 3 ||                                                     \
        g_##SUFFIX##_fortran_call.called != 1 ||                                  \
        g_##SUFFIX##_fortran_call.uplo != 'L' ||                                  \
        g_##SUFFIX##_fortran_call.n != 2 ||                                       \
        g_##SUFFIX##_fortran_call.nrhs != 2 ||                                    \
        g_##SUFFIX##_fortran_call.d_snapshot[0] != (REAL_TYPE)1 ||               \
        g_##SUFFIX##_fortran_call.df_snapshot[0] != (REAL_TYPE)4 ||              \
        g_##SUFFIX##_fortran_call.e_real_snapshot[0] != (REAL_TYPE)3 ||          \
        g_##SUFFIX##_fortran_call.ef_real_snapshot[0] != (REAL_TYPE)6 ||         \
        g_##SUFFIX##_fortran_call.b_real_snapshot[0] != (REAL_TYPE)11 ||         \
        g_##SUFFIX##_fortran_call.b_real_snapshot[1] != (REAL_TYPE)13 ||         \
        g_##SUFFIX##_fortran_call.x_real_snapshot[0] != (REAL_TYPE)21 ||         \
        g_##SUFFIX##_fortran_call.x_real_snapshot[1] != (REAL_TYPE)23 ||         \
        g_##SUFFIX##_fortran_call.ldb != 2 ||                                     \
        g_##SUFFIX##_fortran_call.ldx != 2 ||                                     \
        g_##SUFFIX##_fortran_call.ferr != ferr ||                                 \
        g_##SUFFIX##_fortran_call.berr != berr ||                                 \
        !g_##SUFFIX##_fortran_call.work_nonnull ||                                \
        !g_##SUFFIX##_fortran_call.rwork_nonnull ||                               \
        REAL_PART(b[0]) != REAL_PART(b_original[0]) ||                            \
        REAL_PART(x[0]) != REAL_PART(x_original[0]) ||                            \
        ferr[0] != (REAL_TYPE)((BASE) + 1) || berr[0] != (REAL_TYPE)((BASE) + 2)) { \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not route complex PTRFS inputs correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk routes complex PTRFS inputs\n"); \
    return 0;                                                                     \
}                                                                                 \
static int check_##SUFFIX##_cblas_to_fortran(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                        \
    char uplo = 'U';                                                              \
    int n = 2;                                                                    \
    int nrhs = 1;                                                                 \
    REAL_TYPE d[2] = { (REAL_TYPE)1, (REAL_TYPE)2 };                             \
    CTYPE e[1] = { MAKE((REAL_TYPE)3, (REAL_TYPE)30) };                          \
    REAL_TYPE df[2] = { (REAL_TYPE)4, (REAL_TYPE)5 };                            \
    CTYPE ef[1] = { MAKE((REAL_TYPE)6, (REAL_TYPE)60) };                         \
    CTYPE b[2] = { MAKE((REAL_TYPE)7, (REAL_TYPE)70), MAKE((REAL_TYPE)8, (REAL_TYPE)80) }; \
    int ldb = 2;                                                                  \
    CTYPE x[2] = { MAKE((REAL_TYPE)9, (REAL_TYPE)90), MAKE((REAL_TYPE)10, (REAL_TYPE)100) }; \
    int ldx = 2;                                                                  \
    REAL_TYPE ferr[1] = { 0 };                                                    \
    REAL_TYPE berr[1] = { 0 };                                                    \
    CTYPE work[2] = { MAKE((REAL_TYPE)0, (REAL_TYPE)0), MAKE((REAL_TYPE)0, (REAL_TYPE)0) }; \
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
    thunk(&uplo, &n, &nrhs, d, e, df, ef, b, &ldb, x, &ldx, ferr, berr, work, rwork, &info); \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                    \
        g_##SUFFIX##_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||                  \
        g_##SUFFIX##_cblas_call.uplo != FB_UPPER ||                               \
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
        g_##SUFFIX##_cblas_call.ferr != ferr ||                                   \
        g_##SUFFIX##_cblas_call.berr != berr ||                                   \
        info != (BASE) + 6 || ferr[0] != (REAL_TYPE)((BASE) + 4) ||              \
        berr[0] != (REAL_TYPE)((BASE) + 5)) {                                     \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not propagate complex PTRFS info correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk propagates complex PTRFS info\n"); \
    return 0;                                                                     \
}

DEFINE_PTRFS_REAL_TESTS(sptrfs, float, FB_OP_SPTRFS, 100)
DEFINE_PTRFS_REAL_TESTS(dptrfs, double, FB_OP_DPTRFS, 300)
DEFINE_PTRFS_COMPLEX_TESTS(cptrfs, fb_complex_float_t, float, FB_OP_CPTRFS, 500, make_cfloat, cfloat_real, cfloat_imag)
DEFINE_PTRFS_COMPLEX_TESTS(zptrfs, fb_complex_double_t, double, FB_OP_ZPTRFS, 700, make_cdouble, cdouble_real, cdouble_imag)

int main(void)
{
    int status = 0;

    status |= check_sptrfs_fortran_to_cblas();
    status |= check_sptrfs_cblas_to_fortran();
    status |= check_dptrfs_fortran_to_cblas();
    status |= check_dptrfs_cblas_to_fortran();
    status |= check_cptrfs_fortran_to_cblas();
    status |= check_cptrfs_cblas_to_fortran();
    status |= check_zptrfs_fortran_to_cblas();
    status |= check_zptrfs_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}