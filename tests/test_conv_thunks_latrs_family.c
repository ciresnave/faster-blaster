#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

static fb_complex_float_t make_cf32(float real_value, float imag_value)
{
    fb_complex_float_t value;

    __real__ value = real_value;
    __imag__ value = imag_value;
    return value;
}

static fb_complex_double_t make_cf64(double real_value, double imag_value)
{
    fb_complex_double_t value;

    __real__ value = real_value;
    __imag__ value = imag_value;
    return value;
}

static int cf32_eq(fb_complex_float_t lhs, fb_complex_float_t rhs)
{
    return __real__ lhs == __real__ rhs && __imag__ lhs == __imag__ rhs;
}

static int cf64_eq(fb_complex_double_t lhs, fb_complex_double_t rhs)
{
    return __real__ lhs == __real__ rhs && __imag__ lhs == __imag__ rhs;
}

#define DEFINE_REAL_LATRS_TESTS(SUFFIX, TYPE, OP_ID, BASE)                     \
typedef int (*fb_##SUFFIX##_cblas_fn)(fb_layout_t layout, char uplo,           \
                                      char trans, char diag, char normin,      \
                                      int n, TYPE *a, int lda, TYPE *x,        \
                                      TYPE *scale, TYPE *cnorm);               \
typedef void (*fb_##SUFFIX##_fortran_fn)(char *uplo, char *trans, char *diag,  \
                                         char *normin, int *n, TYPE *a,        \
                                         int *lda, TYPE *x, TYPE *scale,       \
                                         TYPE *cnorm, int *info);              \
static struct {                                                                 \
    int called;                                                                 \
    char uplo;                                                                  \
    char trans;                                                                 \
    char diag;                                                                  \
    char normin;                                                                \
    int n;                                                                      \
    int lda;                                                                    \
    TYPE *a;                                                                    \
    TYPE *x;                                                                    \
    TYPE *scale;                                                                \
    TYPE *cnorm;                                                                \
    TYPE a_snapshot[4];                                                         \
    TYPE cnorm_snapshot[2];                                                     \
} g_##SUFFIX##_fortran_call;                                                    \
static struct {                                                                 \
    int called;                                                                 \
    fb_layout_t layout;                                                         \
    char uplo;                                                                  \
    char trans;                                                                 \
    char diag;                                                                  \
    char normin;                                                                \
    int n;                                                                      \
    int lda;                                                                    \
    TYPE *a;                                                                    \
    TYPE *x;                                                                    \
    TYPE *scale;                                                                \
    TYPE *cnorm;                                                                \
} g_##SUFFIX##_cblas_call;                                                      \
static void stub_##SUFFIX##_fortran(char *uplo, char *trans, char *diag,       \
                                    char *normin, int *n, TYPE *a, int *lda,  \
                                    TYPE *x, TYPE *scale, TYPE *cnorm,        \
                                    int *info)                                 \
{                                                                               \
    int idx;                                                                    \
    g_##SUFFIX##_fortran_call.called += 1;                                      \
    g_##SUFFIX##_fortran_call.uplo = *uplo;                                     \
    g_##SUFFIX##_fortran_call.trans = *trans;                                   \
    g_##SUFFIX##_fortran_call.diag = *diag;                                     \
    g_##SUFFIX##_fortran_call.normin = *normin;                                 \
    g_##SUFFIX##_fortran_call.n = *n;                                           \
    g_##SUFFIX##_fortran_call.lda = *lda;                                       \
    g_##SUFFIX##_fortran_call.a = a;                                            \
    g_##SUFFIX##_fortran_call.x = x;                                            \
    g_##SUFFIX##_fortran_call.scale = scale;                                    \
    g_##SUFFIX##_fortran_call.cnorm = cnorm;                                    \
    for (idx = 0; idx < 4; ++idx) {                                             \
        g_##SUFFIX##_fortran_call.a_snapshot[idx] = a[idx];                     \
    }                                                                           \
    g_##SUFFIX##_fortran_call.cnorm_snapshot[0] = cnorm[0];                     \
    g_##SUFFIX##_fortran_call.cnorm_snapshot[1] = cnorm[1];                     \
    x[0] = (TYPE)((BASE) + 10);                                                 \
    x[1] = (TYPE)((BASE) + 11);                                                 \
    *scale = (TYPE)((BASE) + 20);                                               \
    *info = 0;                                                                  \
}                                                                               \
static int stub_##SUFFIX##_cblas(fb_layout_t layout, char uplo, char trans,    \
                                 char diag, char normin, int n, TYPE *a,       \
                                 int lda, TYPE *x, TYPE *scale,               \
                                 TYPE *cnorm)                                  \
{                                                                               \
    g_##SUFFIX##_cblas_call.called += 1;                                        \
    g_##SUFFIX##_cblas_call.layout = layout;                                    \
    g_##SUFFIX##_cblas_call.uplo = uplo;                                        \
    g_##SUFFIX##_cblas_call.trans = trans;                                      \
    g_##SUFFIX##_cblas_call.diag = diag;                                        \
    g_##SUFFIX##_cblas_call.normin = normin;                                    \
    g_##SUFFIX##_cblas_call.n = n;                                              \
    g_##SUFFIX##_cblas_call.lda = lda;                                          \
    g_##SUFFIX##_cblas_call.a = a;                                              \
    g_##SUFFIX##_cblas_call.x = x;                                              \
    g_##SUFFIX##_cblas_call.scale = scale;                                      \
    g_##SUFFIX##_cblas_call.cnorm = cnorm;                                      \
    x[0] = (TYPE)((BASE) + 30);                                                 \
    x[1] = (TYPE)((BASE) + 31);                                                 \
    *scale = (TYPE)((BASE) + 40);                                               \
    cnorm[0] = (TYPE)((BASE) + 50);                                             \
    cnorm[1] = (TYPE)((BASE) + 51);                                             \
    return (BASE) + 60;                                                         \
}                                                                               \
static int check_##SUFFIX##_fortran_to_cblas(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                        \
    TYPE a[4] = { (TYPE)1, (TYPE)2, (TYPE)3, (TYPE)4 };                        \
    TYPE x[2] = { (TYPE)5, (TYPE)6 };                                           \
    TYPE scale = (TYPE)7;                                                       \
    TYPE cnorm[2] = { (TYPE)8, (TYPE)9 };                                       \
    int status = 0;                                                             \
    memset(&vtable, 0, sizeof(vtable));                                         \
    memset(&g_##SUFFIX##_fortran_call, 0, sizeof(g_##SUFFIX##_fortran_call));   \
    vtable.ext_ops[OP_ID][FB_CONV_FORTRAN] =                                    \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_fortran;                 \
    fb_install_conv_thunks(&vtable, OP_ID);                                     \
    thunk = (fb_##SUFFIX##_cblas_fn)vtable.ext_ops[OP_ID][FB_CONV_CBLAS];       \
    if (!thunk) {                                                               \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk was not installed\n"); \
        return 1;                                                               \
    }                                                                           \
    status = thunk(FB_LAYOUT_ROW_MAJOR, 'U', 'T', 'N', 'Y', 2, a, 2, x, &scale, cnorm); \
    if (status != 0 || g_##SUFFIX##_fortran_call.called != 1 ||                \
        g_##SUFFIX##_fortran_call.uplo != 'U' ||                                \
        g_##SUFFIX##_fortran_call.trans != 'T' ||                               \
        g_##SUFFIX##_fortran_call.diag != 'N' ||                                \
        g_##SUFFIX##_fortran_call.normin != 'Y' ||                              \
        g_##SUFFIX##_fortran_call.n != 2 ||                                     \
        g_##SUFFIX##_fortran_call.lda != 2 ||                                   \
        g_##SUFFIX##_fortran_call.a == a ||                                     \
        g_##SUFFIX##_fortran_call.x != x ||                                     \
        g_##SUFFIX##_fortran_call.scale != &scale ||                            \
        g_##SUFFIX##_fortran_call.cnorm != cnorm ||                             \
        g_##SUFFIX##_fortran_call.a_snapshot[0] != (TYPE)1 ||                   \
        g_##SUFFIX##_fortran_call.a_snapshot[1] != (TYPE)3 ||                   \
        g_##SUFFIX##_fortran_call.a_snapshot[2] != (TYPE)2 ||                   \
        g_##SUFFIX##_fortran_call.a_snapshot[3] != (TYPE)4 ||                   \
        g_##SUFFIX##_fortran_call.cnorm_snapshot[0] != (TYPE)8 ||               \
        g_##SUFFIX##_fortran_call.cnorm_snapshot[1] != (TYPE)9 ||               \
        a[0] != (TYPE)1 || a[1] != (TYPE)2 || a[2] != (TYPE)3 || a[3] != (TYPE)4 || \
        x[0] != (TYPE)((BASE) + 10) || x[1] != (TYPE)((BASE) + 11) ||           \
        scale != (TYPE)((BASE) + 20)) {                                         \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not preserve the row-major triangular matrix copy or precomputed cnorm path\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk preserves row-major triangular coordinates and forwards precomputed cnorm\n"); \
    return 0;                                                                   \
}                                                                               \
static int check_##SUFFIX##_cblas_to_fortran(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                      \
    char uplo = 'L';                                                            \
    char trans = 'N';                                                           \
    char diag = 'U';                                                            \
    char normin = 'N';                                                          \
    int n = 2;                                                                  \
    int lda = 2;                                                                \
    int info = -999;                                                            \
    TYPE a[4] = { (TYPE)1, (TYPE)2, (TYPE)3, (TYPE)4 };                        \
    TYPE x[2] = { (TYPE)0, (TYPE)0 };                                           \
    TYPE scale = (TYPE)0;                                                       \
    TYPE cnorm[2] = { (TYPE)0, (TYPE)0 };                                       \
    memset(&vtable, 0, sizeof(vtable));                                         \
    memset(&g_##SUFFIX##_cblas_call, 0, sizeof(g_##SUFFIX##_cblas_call));       \
    vtable.ext_ops[OP_ID][FB_CONV_CBLAS] =                                      \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_cblas;                   \
    fb_install_conv_thunks(&vtable, OP_ID);                                     \
    thunk = (fb_##SUFFIX##_fortran_fn)vtable.ext_ops[OP_ID][FB_CONV_FORTRAN];   \
    if (!thunk) {                                                               \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk was not installed\n"); \
        return 1;                                                               \
    }                                                                           \
    thunk(&uplo, &trans, &diag, &normin, &n, a, &lda, x, &scale, cnorm, &info);\
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                  \
        g_##SUFFIX##_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||                \
        g_##SUFFIX##_cblas_call.uplo != 'L' ||                                  \
        g_##SUFFIX##_cblas_call.trans != 'N' ||                                 \
        g_##SUFFIX##_cblas_call.diag != 'U' ||                                  \
        g_##SUFFIX##_cblas_call.normin != 'N' ||                                \
        g_##SUFFIX##_cblas_call.n != 2 ||                                       \
        g_##SUFFIX##_cblas_call.lda != 2 ||                                     \
        g_##SUFFIX##_cblas_call.a != a ||                                       \
        g_##SUFFIX##_cblas_call.x != x ||                                       \
        g_##SUFFIX##_cblas_call.scale != &scale ||                              \
        g_##SUFFIX##_cblas_call.cnorm != cnorm ||                               \
        x[0] != (TYPE)((BASE) + 30) || x[1] != (TYPE)((BASE) + 31) ||           \
        scale != (TYPE)((BASE) + 40) ||                                         \
        cnorm[0] != (TYPE)((BASE) + 50) || cnorm[1] != (TYPE)((BASE) + 51) ||  \
        info != (BASE) + 60) {                                                  \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not forward the LATRS flags, cnorm, or info correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk forwards LATRS flags plus cnorm/scale/info through the column-major C entry\n"); \
    return 0;                                                                   \
}

#define DEFINE_COMPLEX_LATRS_TESTS(SUFFIX, CTYPE, RTYPE, OP_ID, MAKE_FN, EQ_FN, BASE) \
typedef int (*fb_##SUFFIX##_cblas_fn)(fb_layout_t layout, char uplo,           \
                                      char trans, char diag, char normin,      \
                                      int n, CTYPE *a, int lda, CTYPE *x,      \
                                      RTYPE *scale, RTYPE *cnorm);             \
typedef void (*fb_##SUFFIX##_fortran_fn)(char *uplo, char *trans, char *diag,  \
                                         char *normin, int *n, CTYPE *a,       \
                                         int *lda, CTYPE *x, RTYPE *scale,     \
                                         RTYPE *cnorm, int *info);             \
static struct {                                                                 \
    int called;                                                                 \
    char uplo;                                                                  \
    char trans;                                                                 \
    char diag;                                                                  \
    char normin;                                                                \
    int n;                                                                      \
    int lda;                                                                    \
    CTYPE *a;                                                                   \
    CTYPE *x;                                                                   \
    RTYPE *scale;                                                               \
    RTYPE *cnorm;                                                               \
    CTYPE a_snapshot[4];                                                        \
    RTYPE cnorm_snapshot[2];                                                    \
} g_##SUFFIX##_fortran_call;                                                    \
static struct {                                                                 \
    int called;                                                                 \
    fb_layout_t layout;                                                         \
    char uplo;                                                                  \
    char trans;                                                                 \
    char diag;                                                                  \
    char normin;                                                                \
    int n;                                                                      \
    int lda;                                                                    \
    CTYPE *a;                                                                   \
    CTYPE *x;                                                                   \
    RTYPE *scale;                                                               \
    RTYPE *cnorm;                                                               \
} g_##SUFFIX##_cblas_call;                                                      \
static void stub_##SUFFIX##_fortran(char *uplo, char *trans, char *diag,       \
                                    char *normin, int *n, CTYPE *a, int *lda, \
                                    CTYPE *x, RTYPE *scale, RTYPE *cnorm,     \
                                    int *info)                                 \
{                                                                               \
    int idx;                                                                    \
    g_##SUFFIX##_fortran_call.called += 1;                                      \
    g_##SUFFIX##_fortran_call.uplo = *uplo;                                     \
    g_##SUFFIX##_fortran_call.trans = *trans;                                   \
    g_##SUFFIX##_fortran_call.diag = *diag;                                     \
    g_##SUFFIX##_fortran_call.normin = *normin;                                 \
    g_##SUFFIX##_fortran_call.n = *n;                                           \
    g_##SUFFIX##_fortran_call.lda = *lda;                                       \
    g_##SUFFIX##_fortran_call.a = a;                                            \
    g_##SUFFIX##_fortran_call.x = x;                                            \
    g_##SUFFIX##_fortran_call.scale = scale;                                    \
    g_##SUFFIX##_fortran_call.cnorm = cnorm;                                    \
    for (idx = 0; idx < 4; ++idx) {                                             \
        g_##SUFFIX##_fortran_call.a_snapshot[idx] = a[idx];                     \
    }                                                                           \
    g_##SUFFIX##_fortran_call.cnorm_snapshot[0] = cnorm[0];                     \
    g_##SUFFIX##_fortran_call.cnorm_snapshot[1] = cnorm[1];                     \
    x[0] = MAKE_FN((RTYPE)((BASE) + 10), (RTYPE)((BASE) + 60));                \
    x[1] = MAKE_FN((RTYPE)((BASE) + 11), (RTYPE)((BASE) + 61));                \
    *scale = (RTYPE)((BASE) + 20);                                              \
    *info = 0;                                                                  \
}                                                                               \
static int stub_##SUFFIX##_cblas(fb_layout_t layout, char uplo, char trans,    \
                                 char diag, char normin, int n, CTYPE *a,      \
                                 int lda, CTYPE *x, RTYPE *scale,             \
                                 RTYPE *cnorm)                                 \
{                                                                               \
    g_##SUFFIX##_cblas_call.called += 1;                                        \
    g_##SUFFIX##_cblas_call.layout = layout;                                    \
    g_##SUFFIX##_cblas_call.uplo = uplo;                                        \
    g_##SUFFIX##_cblas_call.trans = trans;                                      \
    g_##SUFFIX##_cblas_call.diag = diag;                                        \
    g_##SUFFIX##_cblas_call.normin = normin;                                    \
    g_##SUFFIX##_cblas_call.n = n;                                              \
    g_##SUFFIX##_cblas_call.lda = lda;                                          \
    g_##SUFFIX##_cblas_call.a = a;                                              \
    g_##SUFFIX##_cblas_call.x = x;                                              \
    g_##SUFFIX##_cblas_call.scale = scale;                                      \
    g_##SUFFIX##_cblas_call.cnorm = cnorm;                                      \
    x[0] = MAKE_FN((RTYPE)((BASE) + 30), (RTYPE)((BASE) + 80));                \
    x[1] = MAKE_FN((RTYPE)((BASE) + 31), (RTYPE)((BASE) + 81));                \
    *scale = (RTYPE)((BASE) + 40);                                              \
    cnorm[0] = (RTYPE)((BASE) + 50);                                            \
    cnorm[1] = (RTYPE)((BASE) + 51);                                            \
    return (BASE) + 60;                                                         \
}                                                                               \
static int check_##SUFFIX##_fortran_to_cblas(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                        \
    CTYPE a[4];                                                                 \
    CTYPE x[2];                                                                 \
    RTYPE scale = (RTYPE)7;                                                     \
    RTYPE cnorm[2] = { (RTYPE)8, (RTYPE)9 };                                    \
    int status = 0;                                                             \
    a[0] = MAKE_FN((RTYPE)1, (RTYPE)11);                                        \
    a[1] = MAKE_FN((RTYPE)2, (RTYPE)12);                                        \
    a[2] = MAKE_FN((RTYPE)3, (RTYPE)13);                                        \
    a[3] = MAKE_FN((RTYPE)4, (RTYPE)14);                                        \
    x[0] = MAKE_FN((RTYPE)5, (RTYPE)15);                                        \
    x[1] = MAKE_FN((RTYPE)6, (RTYPE)16);                                        \
    memset(&vtable, 0, sizeof(vtable));                                         \
    memset(&g_##SUFFIX##_fortran_call, 0, sizeof(g_##SUFFIX##_fortran_call));   \
    vtable.ext_ops[OP_ID][FB_CONV_FORTRAN] =                                    \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_fortran;                 \
    fb_install_conv_thunks(&vtable, OP_ID);                                     \
    thunk = (fb_##SUFFIX##_cblas_fn)vtable.ext_ops[OP_ID][FB_CONV_CBLAS];       \
    if (!thunk) {                                                               \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk was not installed\n"); \
        return 1;                                                               \
    }                                                                           \
    status = thunk(FB_LAYOUT_ROW_MAJOR, 'U', 'C', 'N', 'Y', 2, a, 2, x, &scale, cnorm); \
    if (status != 0 || g_##SUFFIX##_fortran_call.called != 1 ||                \
        g_##SUFFIX##_fortran_call.uplo != 'U' ||                                \
        g_##SUFFIX##_fortran_call.trans != 'C' ||                               \
        g_##SUFFIX##_fortran_call.diag != 'N' ||                                \
        g_##SUFFIX##_fortran_call.normin != 'Y' ||                              \
        g_##SUFFIX##_fortran_call.n != 2 ||                                     \
        g_##SUFFIX##_fortran_call.lda != 2 ||                                   \
        g_##SUFFIX##_fortran_call.a == a ||                                     \
        g_##SUFFIX##_fortran_call.x != x ||                                     \
        g_##SUFFIX##_fortran_call.scale != &scale ||                            \
        g_##SUFFIX##_fortran_call.cnorm != cnorm ||                             \
        !EQ_FN(g_##SUFFIX##_fortran_call.a_snapshot[0],                         \
               MAKE_FN((RTYPE)1, (RTYPE)11)) ||                                 \
        !EQ_FN(g_##SUFFIX##_fortran_call.a_snapshot[1],                         \
               MAKE_FN((RTYPE)3, (RTYPE)13)) ||                                 \
        !EQ_FN(g_##SUFFIX##_fortran_call.a_snapshot[2],                         \
               MAKE_FN((RTYPE)2, (RTYPE)12)) ||                                 \
        !EQ_FN(g_##SUFFIX##_fortran_call.a_snapshot[3],                         \
               MAKE_FN((RTYPE)4, (RTYPE)14)) ||                                 \
        g_##SUFFIX##_fortran_call.cnorm_snapshot[0] != (RTYPE)8 ||              \
        g_##SUFFIX##_fortran_call.cnorm_snapshot[1] != (RTYPE)9 ||              \
        !EQ_FN(a[0], MAKE_FN((RTYPE)1, (RTYPE)11)) ||                           \
        !EQ_FN(a[1], MAKE_FN((RTYPE)2, (RTYPE)12)) ||                           \
        !EQ_FN(a[2], MAKE_FN((RTYPE)3, (RTYPE)13)) ||                           \
        !EQ_FN(a[3], MAKE_FN((RTYPE)4, (RTYPE)14)) ||                           \
        !EQ_FN(x[0], MAKE_FN((RTYPE)((BASE) + 10), (RTYPE)((BASE) + 60))) ||    \
        !EQ_FN(x[1], MAKE_FN((RTYPE)((BASE) + 11), (RTYPE)((BASE) + 61))) ||    \
        scale != (RTYPE)((BASE) + 20)) {                                        \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not preserve the row-major complex triangular copy or precomputed cnorm path\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk preserves row-major complex triangular coordinates and forwards precomputed cnorm\n"); \
    return 0;                                                                   \
}                                                                               \
static int check_##SUFFIX##_cblas_to_fortran(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                      \
    char uplo = 'L';                                                            \
    char trans = 'N';                                                           \
    char diag = 'U';                                                            \
    char normin = 'N';                                                          \
    int n = 2;                                                                  \
    int lda = 2;                                                                \
    int info = -999;                                                            \
    CTYPE a[4];                                                                 \
    CTYPE x[2];                                                                 \
    RTYPE scale = (RTYPE)0;                                                     \
    RTYPE cnorm[2] = { (RTYPE)0, (RTYPE)0 };                                    \
    a[0] = MAKE_FN((RTYPE)1, (RTYPE)11);                                        \
    a[1] = MAKE_FN((RTYPE)2, (RTYPE)12);                                        \
    a[2] = MAKE_FN((RTYPE)3, (RTYPE)13);                                        \
    a[3] = MAKE_FN((RTYPE)4, (RTYPE)14);                                        \
    x[0] = MAKE_FN((RTYPE)0, (RTYPE)0);                                         \
    x[1] = MAKE_FN((RTYPE)0, (RTYPE)0);                                         \
    memset(&vtable, 0, sizeof(vtable));                                         \
    memset(&g_##SUFFIX##_cblas_call, 0, sizeof(g_##SUFFIX##_cblas_call));       \
    vtable.ext_ops[OP_ID][FB_CONV_CBLAS] =                                      \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_cblas;                   \
    fb_install_conv_thunks(&vtable, OP_ID);                                     \
    thunk = (fb_##SUFFIX##_fortran_fn)vtable.ext_ops[OP_ID][FB_CONV_FORTRAN];   \
    if (!thunk) {                                                               \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk was not installed\n"); \
        return 1;                                                               \
    }                                                                           \
    thunk(&uplo, &trans, &diag, &normin, &n, a, &lda, x, &scale, cnorm, &info);\
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                  \
        g_##SUFFIX##_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||                \
        g_##SUFFIX##_cblas_call.uplo != 'L' ||                                  \
        g_##SUFFIX##_cblas_call.trans != 'N' ||                                 \
        g_##SUFFIX##_cblas_call.diag != 'U' ||                                  \
        g_##SUFFIX##_cblas_call.normin != 'N' ||                                \
        g_##SUFFIX##_cblas_call.n != 2 ||                                       \
        g_##SUFFIX##_cblas_call.lda != 2 ||                                     \
        g_##SUFFIX##_cblas_call.a != a ||                                       \
        g_##SUFFIX##_cblas_call.x != x ||                                       \
        g_##SUFFIX##_cblas_call.scale != &scale ||                              \
        g_##SUFFIX##_cblas_call.cnorm != cnorm ||                               \
        !EQ_FN(x[0], MAKE_FN((RTYPE)((BASE) + 30), (RTYPE)((BASE) + 80))) ||    \
        !EQ_FN(x[1], MAKE_FN((RTYPE)((BASE) + 31), (RTYPE)((BASE) + 81))) ||    \
        scale != (RTYPE)((BASE) + 40) ||                                        \
        cnorm[0] != (RTYPE)((BASE) + 50) ||                                     \
        cnorm[1] != (RTYPE)((BASE) + 51) ||                                     \
        info != (BASE) + 60) {                                                  \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not forward the complex LATRS flags, cnorm, or info correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk forwards complex LATRS flags plus cnorm/scale/info through the column-major C entry\n"); \
    return 0;                                                                   \
}

DEFINE_REAL_LATRS_TESTS(slatrs, float, FB_OP_SLATRS, 100)
DEFINE_REAL_LATRS_TESTS(dlatrs, double, FB_OP_DLATRS, 300)
DEFINE_COMPLEX_LATRS_TESTS(clatrs, fb_complex_float_t, float, FB_OP_CLATRS,
                           make_cf32, cf32_eq, 500)
DEFINE_COMPLEX_LATRS_TESTS(zlatrs, fb_complex_double_t, double, FB_OP_ZLATRS,
                           make_cf64, cf64_eq, 700)

int main(void)
{
    int status = 0;

    status |= check_slatrs_fortran_to_cblas();
    status |= check_slatrs_cblas_to_fortran();
    status |= check_dlatrs_fortran_to_cblas();
    status |= check_dlatrs_cblas_to_fortran();
    status |= check_clatrs_fortran_to_cblas();
    status |= check_clatrs_cblas_to_fortran();
    status |= check_zlatrs_fortran_to_cblas();
    status |= check_zlatrs_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}