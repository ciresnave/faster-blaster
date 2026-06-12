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

#define DEFINE_REAL_LATBS_TESTS(SUFFIX, TYPE, OP_ID, BASE)                     \
typedef int (*fb_##SUFFIX##_cblas_fn)(fb_layout_t layout, char uplo,           \
                                      char trans, char diag, char normin,      \
                                      int n, int kd, TYPE *ab, int ldab,       \
                                      TYPE *x, TYPE *scale, TYPE *cnorm);      \
typedef void (*fb_##SUFFIX##_fortran_fn)(char *uplo, char *trans, char *diag,  \
                                         char *normin, int *n, int *kd,       \
                                         TYPE *ab, int *ldab, TYPE *x,        \
                                         TYPE *scale, TYPE *cnorm, int *info);\
static struct {                                                                 \
    int called;                                                                 \
    char uplo;                                                                  \
    char trans;                                                                 \
    char diag;                                                                  \
    char normin;                                                                \
    int n;                                                                      \
    int kd;                                                                     \
    int ldab;                                                                   \
    TYPE *ab;                                                                   \
    TYPE *x;                                                                    \
    TYPE *scale;                                                                \
    TYPE *cnorm;                                                                \
    TYPE ab_snapshot[8];                                                        \
    TYPE cnorm_snapshot[4];                                                     \
} g_##SUFFIX##_fortran_call;                                                    \
static struct {                                                                 \
    int called;                                                                 \
    fb_layout_t layout;                                                         \
    char uplo;                                                                  \
    char trans;                                                                 \
    char diag;                                                                  \
    char normin;                                                                \
    int n;                                                                      \
    int kd;                                                                     \
    int ldab;                                                                   \
    TYPE *ab;                                                                   \
    TYPE *x;                                                                    \
    TYPE *scale;                                                                \
    TYPE *cnorm;                                                                \
} g_##SUFFIX##_cblas_call;                                                      \
static void stub_##SUFFIX##_fortran(char *uplo, char *trans, char *diag,       \
                                    char *normin, int *n, int *kd, TYPE *ab,  \
                                    int *ldab, TYPE *x, TYPE *scale,          \
                                    TYPE *cnorm, int *info)                   \
{                                                                               \
    int index;                                                                  \
    g_##SUFFIX##_fortran_call.called += 1;                                      \
    g_##SUFFIX##_fortran_call.uplo = *uplo;                                     \
    g_##SUFFIX##_fortran_call.trans = *trans;                                   \
    g_##SUFFIX##_fortran_call.diag = *diag;                                     \
    g_##SUFFIX##_fortran_call.normin = *normin;                                 \
    g_##SUFFIX##_fortran_call.n = *n;                                           \
    g_##SUFFIX##_fortran_call.kd = *kd;                                         \
    g_##SUFFIX##_fortran_call.ldab = *ldab;                                     \
    g_##SUFFIX##_fortran_call.ab = ab;                                          \
    g_##SUFFIX##_fortran_call.x = x;                                            \
    g_##SUFFIX##_fortran_call.scale = scale;                                    \
    g_##SUFFIX##_fortran_call.cnorm = cnorm;                                    \
    for (index = 0; index < (*ldab * *n); ++index) {                           \
        g_##SUFFIX##_fortran_call.ab_snapshot[index] = ab[index];               \
    }                                                                           \
    for (index = 0; index < *n; ++index) {                                      \
        g_##SUFFIX##_fortran_call.cnorm_snapshot[index] = cnorm[index];         \
    }                                                                           \
    x[0] = (TYPE)((BASE) + 10);                                                 \
    x[1] = (TYPE)((BASE) + 11);                                                 \
    x[2] = (TYPE)((BASE) + 12);                                                 \
    x[3] = (TYPE)((BASE) + 13);                                                 \
    *scale = (TYPE)((BASE) + 20);                                               \
    *info = 0;                                                                  \
}                                                                               \
static int stub_##SUFFIX##_cblas(fb_layout_t layout, char uplo, char trans,    \
                                 char diag, char normin, int n, int kd,        \
                                 TYPE *ab, int ldab, TYPE *x, TYPE *scale,    \
                                 TYPE *cnorm)                                 \
{                                                                               \
    g_##SUFFIX##_cblas_call.called += 1;                                        \
    g_##SUFFIX##_cblas_call.layout = layout;                                    \
    g_##SUFFIX##_cblas_call.uplo = uplo;                                        \
    g_##SUFFIX##_cblas_call.trans = trans;                                      \
    g_##SUFFIX##_cblas_call.diag = diag;                                        \
    g_##SUFFIX##_cblas_call.normin = normin;                                    \
    g_##SUFFIX##_cblas_call.n = n;                                              \
    g_##SUFFIX##_cblas_call.kd = kd;                                            \
    g_##SUFFIX##_cblas_call.ldab = ldab;                                        \
    g_##SUFFIX##_cblas_call.ab = ab;                                            \
    g_##SUFFIX##_cblas_call.x = x;                                              \
    g_##SUFFIX##_cblas_call.scale = scale;                                      \
    g_##SUFFIX##_cblas_call.cnorm = cnorm;                                      \
    x[0] = (TYPE)((BASE) + 30);                                                 \
    x[1] = (TYPE)((BASE) + 31);                                                 \
    x[2] = (TYPE)((BASE) + 32);                                                 \
    x[3] = (TYPE)((BASE) + 33);                                                 \
    *scale = (TYPE)((BASE) + 40);                                               \
    cnorm[0] = (TYPE)((BASE) + 50);                                             \
    cnorm[1] = (TYPE)((BASE) + 51);                                             \
    cnorm[2] = (TYPE)((BASE) + 52);                                             \
    cnorm[3] = (TYPE)((BASE) + 53);                                             \
    return (BASE) + 60;                                                         \
}                                                                               \
static int check_##SUFFIX##_fortran_to_cblas(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                        \
    TYPE ab[8] = { (TYPE)10, (TYPE)11, (TYPE)12, (TYPE)13,                     \
                   (TYPE)20, (TYPE)21, (TYPE)22, (TYPE)23 };                   \
    TYPE x[4] = { (TYPE)1, (TYPE)2, (TYPE)3, (TYPE)4 };                        \
    TYPE scale = (TYPE)5;                                                       \
    TYPE cnorm[4] = { (TYPE)6, (TYPE)7, (TYPE)8, (TYPE)9 };                    \
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
    status = thunk(FB_LAYOUT_ROW_MAJOR, 'U', 'T', 'N', 'Y', 4, 1, ab, 4, x, &scale, cnorm); \
    if (status != 0 || g_##SUFFIX##_fortran_call.called != 1 ||                \
        g_##SUFFIX##_fortran_call.uplo != 'U' ||                                \
        g_##SUFFIX##_fortran_call.trans != 'T' ||                               \
        g_##SUFFIX##_fortran_call.diag != 'N' ||                                \
        g_##SUFFIX##_fortran_call.normin != 'Y' ||                              \
        g_##SUFFIX##_fortran_call.n != 4 ||                                     \
        g_##SUFFIX##_fortran_call.kd != 1 ||                                    \
        g_##SUFFIX##_fortran_call.ldab != 2 ||                                  \
        g_##SUFFIX##_fortran_call.ab == ab ||                                   \
        g_##SUFFIX##_fortran_call.x != x ||                                     \
        g_##SUFFIX##_fortran_call.scale != &scale ||                            \
        g_##SUFFIX##_fortran_call.cnorm != cnorm ||                             \
        g_##SUFFIX##_fortran_call.ab_snapshot[0] != (TYPE)0 ||                  \
        g_##SUFFIX##_fortran_call.ab_snapshot[1] != (TYPE)20 ||                 \
        g_##SUFFIX##_fortran_call.ab_snapshot[2] != (TYPE)11 ||                 \
        g_##SUFFIX##_fortran_call.ab_snapshot[3] != (TYPE)21 ||                 \
        g_##SUFFIX##_fortran_call.ab_snapshot[4] != (TYPE)12 ||                 \
        g_##SUFFIX##_fortran_call.ab_snapshot[5] != (TYPE)22 ||                 \
        g_##SUFFIX##_fortran_call.ab_snapshot[6] != (TYPE)13 ||                 \
        g_##SUFFIX##_fortran_call.ab_snapshot[7] != (TYPE)23 ||                 \
        g_##SUFFIX##_fortran_call.cnorm_snapshot[0] != (TYPE)6 ||               \
        g_##SUFFIX##_fortran_call.cnorm_snapshot[1] != (TYPE)7 ||               \
        g_##SUFFIX##_fortran_call.cnorm_snapshot[2] != (TYPE)8 ||               \
        g_##SUFFIX##_fortran_call.cnorm_snapshot[3] != (TYPE)9 ||               \
        ab[0] != (TYPE)10 || ab[1] != (TYPE)11 || ab[2] != (TYPE)12 ||         \
        ab[3] != (TYPE)13 || ab[4] != (TYPE)20 || ab[5] != (TYPE)21 ||         \
        ab[6] != (TYPE)22 || ab[7] != (TYPE)23 ||                               \
        x[0] != (TYPE)((BASE) + 10) || x[1] != (TYPE)((BASE) + 11) ||           \
        x[2] != (TYPE)((BASE) + 12) || x[3] != (TYPE)((BASE) + 13) ||           \
        scale != (TYPE)((BASE) + 20)) {                                         \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not preserve the row-major triangular band copy or precomputed cnorm path\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk transposes row-major triangular band storage and forwards precomputed cnorm\n"); \
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
    int n = 4;                                                                  \
    int kd = 1;                                                                 \
    int ldab = 2;                                                               \
    int info = -999;                                                            \
    TYPE ab[8] = { (TYPE)1, (TYPE)2, (TYPE)3, (TYPE)4,                         \
                   (TYPE)5, (TYPE)6, (TYPE)7, (TYPE)8 };                       \
    TYPE x[4] = { (TYPE)0, (TYPE)0, (TYPE)0, (TYPE)0 };                        \
    TYPE scale = (TYPE)0;                                                       \
    TYPE cnorm[4] = { (TYPE)0, (TYPE)0, (TYPE)0, (TYPE)0 };                    \
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
    thunk(&uplo, &trans, &diag, &normin, &n, &kd, ab, &ldab, x, &scale, cnorm, &info); \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                  \
        g_##SUFFIX##_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||                \
        g_##SUFFIX##_cblas_call.uplo != 'L' ||                                  \
        g_##SUFFIX##_cblas_call.trans != 'N' ||                                 \
        g_##SUFFIX##_cblas_call.diag != 'U' ||                                  \
        g_##SUFFIX##_cblas_call.normin != 'N' ||                                \
        g_##SUFFIX##_cblas_call.n != 4 ||                                       \
        g_##SUFFIX##_cblas_call.kd != 1 ||                                      \
        g_##SUFFIX##_cblas_call.ldab != 2 ||                                    \
        g_##SUFFIX##_cblas_call.ab != ab ||                                     \
        g_##SUFFIX##_cblas_call.x != x ||                                       \
        g_##SUFFIX##_cblas_call.scale != &scale ||                              \
        g_##SUFFIX##_cblas_call.cnorm != cnorm ||                               \
        x[0] != (TYPE)((BASE) + 30) || x[1] != (TYPE)((BASE) + 31) ||           \
        x[2] != (TYPE)((BASE) + 32) || x[3] != (TYPE)((BASE) + 33) ||           \
        scale != (TYPE)((BASE) + 40) ||                                         \
        cnorm[0] != (TYPE)((BASE) + 50) || cnorm[1] != (TYPE)((BASE) + 51) ||  \
        cnorm[2] != (TYPE)((BASE) + 52) || cnorm[3] != (TYPE)((BASE) + 53) ||  \
        info != (BASE) + 60) {                                                  \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not forward the LATBS flags, cnorm, or info correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk forwards LATBS flags plus cnorm/scale/info through the column-major C entry\n"); \
    return 0;                                                                   \
}

#define DEFINE_COMPLEX_LATBS_TESTS(SUFFIX, CTYPE, RTYPE, OP_ID, MAKE_FN, EQ_FN, BASE) \
typedef int (*fb_##SUFFIX##_cblas_fn)(fb_layout_t layout, char uplo,           \
                                      char trans, char diag, char normin,      \
                                      int n, int kd, CTYPE *ab, int ldab,      \
                                      CTYPE *x, RTYPE *scale, RTYPE *cnorm);   \
typedef void (*fb_##SUFFIX##_fortran_fn)(char *uplo, char *trans, char *diag,  \
                                         char *normin, int *n, int *kd,       \
                                         CTYPE *ab, int *ldab, CTYPE *x,      \
                                         RTYPE *scale, RTYPE *cnorm, int *info);\
static struct {                                                                 \
    int called;                                                                 \
    char uplo;                                                                  \
    char trans;                                                                 \
    char diag;                                                                  \
    char normin;                                                                \
    int n;                                                                      \
    int kd;                                                                     \
    int ldab;                                                                   \
    CTYPE *ab;                                                                  \
    CTYPE *x;                                                                   \
    RTYPE *scale;                                                               \
    RTYPE *cnorm;                                                               \
    CTYPE ab_snapshot[8];                                                       \
    RTYPE cnorm_snapshot[4];                                                    \
} g_##SUFFIX##_fortran_call;                                                    \
static struct {                                                                 \
    int called;                                                                 \
    fb_layout_t layout;                                                         \
    char uplo;                                                                  \
    char trans;                                                                 \
    char diag;                                                                  \
    char normin;                                                                \
    int n;                                                                      \
    int kd;                                                                     \
    int ldab;                                                                   \
    CTYPE *ab;                                                                  \
    CTYPE *x;                                                                   \
    RTYPE *scale;                                                               \
    RTYPE *cnorm;                                                               \
} g_##SUFFIX##_cblas_call;                                                      \
static void stub_##SUFFIX##_fortran(char *uplo, char *trans, char *diag,       \
                                    char *normin, int *n, int *kd, CTYPE *ab, \
                                    int *ldab, CTYPE *x, RTYPE *scale,        \
                                    RTYPE *cnorm, int *info)                  \
{                                                                               \
    int index;                                                                  \
    g_##SUFFIX##_fortran_call.called += 1;                                      \
    g_##SUFFIX##_fortran_call.uplo = *uplo;                                     \
    g_##SUFFIX##_fortran_call.trans = *trans;                                   \
    g_##SUFFIX##_fortran_call.diag = *diag;                                     \
    g_##SUFFIX##_fortran_call.normin = *normin;                                 \
    g_##SUFFIX##_fortran_call.n = *n;                                           \
    g_##SUFFIX##_fortran_call.kd = *kd;                                         \
    g_##SUFFIX##_fortran_call.ldab = *ldab;                                     \
    g_##SUFFIX##_fortran_call.ab = ab;                                          \
    g_##SUFFIX##_fortran_call.x = x;                                            \
    g_##SUFFIX##_fortran_call.scale = scale;                                    \
    g_##SUFFIX##_fortran_call.cnorm = cnorm;                                    \
    for (index = 0; index < (*ldab * *n); ++index) {                           \
        g_##SUFFIX##_fortran_call.ab_snapshot[index] = ab[index];               \
    }                                                                           \
    for (index = 0; index < *n; ++index) {                                      \
        g_##SUFFIX##_fortran_call.cnorm_snapshot[index] = cnorm[index];         \
    }                                                                           \
    x[0] = MAKE_FN((RTYPE)((BASE) + 10), (RTYPE)((BASE) + 60));                \
    x[1] = MAKE_FN((RTYPE)((BASE) + 11), (RTYPE)((BASE) + 61));                \
    x[2] = MAKE_FN((RTYPE)((BASE) + 12), (RTYPE)((BASE) + 62));                \
    x[3] = MAKE_FN((RTYPE)((BASE) + 13), (RTYPE)((BASE) + 63));                \
    *scale = (RTYPE)((BASE) + 20);                                              \
    *info = 0;                                                                  \
}                                                                               \
static int stub_##SUFFIX##_cblas(fb_layout_t layout, char uplo, char trans,    \
                                 char diag, char normin, int n, int kd,        \
                                 CTYPE *ab, int ldab, CTYPE *x, RTYPE *scale, \
                                 RTYPE *cnorm)                                \
{                                                                               \
    g_##SUFFIX##_cblas_call.called += 1;                                        \
    g_##SUFFIX##_cblas_call.layout = layout;                                    \
    g_##SUFFIX##_cblas_call.uplo = uplo;                                        \
    g_##SUFFIX##_cblas_call.trans = trans;                                      \
    g_##SUFFIX##_cblas_call.diag = diag;                                        \
    g_##SUFFIX##_cblas_call.normin = normin;                                    \
    g_##SUFFIX##_cblas_call.n = n;                                              \
    g_##SUFFIX##_cblas_call.kd = kd;                                            \
    g_##SUFFIX##_cblas_call.ldab = ldab;                                        \
    g_##SUFFIX##_cblas_call.ab = ab;                                            \
    g_##SUFFIX##_cblas_call.x = x;                                              \
    g_##SUFFIX##_cblas_call.scale = scale;                                      \
    g_##SUFFIX##_cblas_call.cnorm = cnorm;                                      \
    x[0] = MAKE_FN((RTYPE)((BASE) + 30), (RTYPE)((BASE) + 80));                \
    x[1] = MAKE_FN((RTYPE)((BASE) + 31), (RTYPE)((BASE) + 81));                \
    x[2] = MAKE_FN((RTYPE)((BASE) + 32), (RTYPE)((BASE) + 82));                \
    x[3] = MAKE_FN((RTYPE)((BASE) + 33), (RTYPE)((BASE) + 83));                \
    *scale = (RTYPE)((BASE) + 40);                                              \
    cnorm[0] = (RTYPE)((BASE) + 50);                                            \
    cnorm[1] = (RTYPE)((BASE) + 51);                                            \
    cnorm[2] = (RTYPE)((BASE) + 52);                                            \
    cnorm[3] = (RTYPE)((BASE) + 53);                                            \
    return (BASE) + 60;                                                         \
}                                                                               \
static int check_##SUFFIX##_fortran_to_cblas(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                        \
    CTYPE ab[8];                                                                \
    CTYPE x[4];                                                                 \
    RTYPE scale = (RTYPE)5;                                                     \
    RTYPE cnorm[4] = { (RTYPE)6, (RTYPE)7, (RTYPE)8, (RTYPE)9 };               \
    int status = 0;                                                             \
    ab[0] = MAKE_FN((RTYPE)10, (RTYPE)110);                                     \
    ab[1] = MAKE_FN((RTYPE)11, (RTYPE)111);                                     \
    ab[2] = MAKE_FN((RTYPE)12, (RTYPE)112);                                     \
    ab[3] = MAKE_FN((RTYPE)13, (RTYPE)113);                                     \
    ab[4] = MAKE_FN((RTYPE)20, (RTYPE)120);                                     \
    ab[5] = MAKE_FN((RTYPE)21, (RTYPE)121);                                     \
    ab[6] = MAKE_FN((RTYPE)22, (RTYPE)122);                                     \
    ab[7] = MAKE_FN((RTYPE)23, (RTYPE)123);                                     \
    x[0] = MAKE_FN((RTYPE)1, (RTYPE)101);                                       \
    x[1] = MAKE_FN((RTYPE)2, (RTYPE)102);                                       \
    x[2] = MAKE_FN((RTYPE)3, (RTYPE)103);                                       \
    x[3] = MAKE_FN((RTYPE)4, (RTYPE)104);                                       \
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
    status = thunk(FB_LAYOUT_ROW_MAJOR, 'U', 'C', 'N', 'Y', 4, 1, ab, 4, x, &scale, cnorm); \
    if (status != 0 || g_##SUFFIX##_fortran_call.called != 1 ||                \
        g_##SUFFIX##_fortran_call.uplo != 'U' ||                                \
        g_##SUFFIX##_fortran_call.trans != 'C' ||                               \
        g_##SUFFIX##_fortran_call.diag != 'N' ||                                \
        g_##SUFFIX##_fortran_call.normin != 'Y' ||                              \
        g_##SUFFIX##_fortran_call.n != 4 ||                                     \
        g_##SUFFIX##_fortran_call.kd != 1 ||                                    \
        g_##SUFFIX##_fortran_call.ldab != 2 ||                                  \
        g_##SUFFIX##_fortran_call.ab == ab ||                                   \
        g_##SUFFIX##_fortran_call.x != x ||                                     \
        g_##SUFFIX##_fortran_call.scale != &scale ||                            \
        g_##SUFFIX##_fortran_call.cnorm != cnorm ||                             \
        !EQ_FN(g_##SUFFIX##_fortran_call.ab_snapshot[0],                        \
               MAKE_FN((RTYPE)0, (RTYPE)0)) ||                                  \
        !EQ_FN(g_##SUFFIX##_fortran_call.ab_snapshot[1],                        \
               MAKE_FN((RTYPE)20, (RTYPE)120)) ||                               \
        !EQ_FN(g_##SUFFIX##_fortran_call.ab_snapshot[2],                        \
               MAKE_FN((RTYPE)11, (RTYPE)111)) ||                               \
        !EQ_FN(g_##SUFFIX##_fortran_call.ab_snapshot[3],                        \
               MAKE_FN((RTYPE)21, (RTYPE)121)) ||                               \
        !EQ_FN(g_##SUFFIX##_fortran_call.ab_snapshot[4],                        \
               MAKE_FN((RTYPE)12, (RTYPE)112)) ||                               \
        !EQ_FN(g_##SUFFIX##_fortran_call.ab_snapshot[5],                        \
               MAKE_FN((RTYPE)22, (RTYPE)122)) ||                               \
        !EQ_FN(g_##SUFFIX##_fortran_call.ab_snapshot[6],                        \
               MAKE_FN((RTYPE)13, (RTYPE)113)) ||                               \
        !EQ_FN(g_##SUFFIX##_fortran_call.ab_snapshot[7],                        \
               MAKE_FN((RTYPE)23, (RTYPE)123)) ||                               \
        g_##SUFFIX##_fortran_call.cnorm_snapshot[0] != (RTYPE)6 ||              \
        g_##SUFFIX##_fortran_call.cnorm_snapshot[1] != (RTYPE)7 ||              \
        g_##SUFFIX##_fortran_call.cnorm_snapshot[2] != (RTYPE)8 ||              \
        g_##SUFFIX##_fortran_call.cnorm_snapshot[3] != (RTYPE)9 ||              \
        !EQ_FN(ab[0], MAKE_FN((RTYPE)10, (RTYPE)110)) ||                        \
        !EQ_FN(ab[1], MAKE_FN((RTYPE)11, (RTYPE)111)) ||                        \
        !EQ_FN(ab[2], MAKE_FN((RTYPE)12, (RTYPE)112)) ||                        \
        !EQ_FN(ab[3], MAKE_FN((RTYPE)13, (RTYPE)113)) ||                        \
        !EQ_FN(ab[4], MAKE_FN((RTYPE)20, (RTYPE)120)) ||                        \
        !EQ_FN(ab[5], MAKE_FN((RTYPE)21, (RTYPE)121)) ||                        \
        !EQ_FN(ab[6], MAKE_FN((RTYPE)22, (RTYPE)122)) ||                        \
        !EQ_FN(ab[7], MAKE_FN((RTYPE)23, (RTYPE)123)) ||                        \
        !EQ_FN(x[0], MAKE_FN((RTYPE)((BASE) + 10), (RTYPE)((BASE) + 60))) ||    \
        !EQ_FN(x[1], MAKE_FN((RTYPE)((BASE) + 11), (RTYPE)((BASE) + 61))) ||    \
        !EQ_FN(x[2], MAKE_FN((RTYPE)((BASE) + 12), (RTYPE)((BASE) + 62))) ||    \
        !EQ_FN(x[3], MAKE_FN((RTYPE)((BASE) + 13), (RTYPE)((BASE) + 63))) ||    \
        scale != (RTYPE)((BASE) + 20)) {                                        \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not preserve the row-major complex triangular band copy or precomputed cnorm path\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk transposes row-major complex triangular band storage and forwards precomputed cnorm\n"); \
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
    int n = 4;                                                                  \
    int kd = 1;                                                                 \
    int ldab = 2;                                                               \
    int info = -999;                                                            \
    CTYPE ab[8];                                                                \
    CTYPE x[4];                                                                 \
    RTYPE scale = (RTYPE)0;                                                     \
    RTYPE cnorm[4] = { (RTYPE)0, (RTYPE)0, (RTYPE)0, (RTYPE)0 };               \
    ab[0] = MAKE_FN((RTYPE)1, (RTYPE)101);                                      \
    ab[1] = MAKE_FN((RTYPE)2, (RTYPE)102);                                      \
    ab[2] = MAKE_FN((RTYPE)3, (RTYPE)103);                                      \
    ab[3] = MAKE_FN((RTYPE)4, (RTYPE)104);                                      \
    ab[4] = MAKE_FN((RTYPE)5, (RTYPE)105);                                      \
    ab[5] = MAKE_FN((RTYPE)6, (RTYPE)106);                                      \
    ab[6] = MAKE_FN((RTYPE)7, (RTYPE)107);                                      \
    ab[7] = MAKE_FN((RTYPE)8, (RTYPE)108);                                      \
    x[0] = MAKE_FN((RTYPE)0, (RTYPE)0);                                         \
    x[1] = MAKE_FN((RTYPE)0, (RTYPE)0);                                         \
    x[2] = MAKE_FN((RTYPE)0, (RTYPE)0);                                         \
    x[3] = MAKE_FN((RTYPE)0, (RTYPE)0);                                         \
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
    thunk(&uplo, &trans, &diag, &normin, &n, &kd, ab, &ldab, x, &scale, cnorm, &info); \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                  \
        g_##SUFFIX##_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||                \
        g_##SUFFIX##_cblas_call.uplo != 'L' ||                                  \
        g_##SUFFIX##_cblas_call.trans != 'N' ||                                 \
        g_##SUFFIX##_cblas_call.diag != 'U' ||                                  \
        g_##SUFFIX##_cblas_call.normin != 'N' ||                                \
        g_##SUFFIX##_cblas_call.n != 4 ||                                       \
        g_##SUFFIX##_cblas_call.kd != 1 ||                                      \
        g_##SUFFIX##_cblas_call.ldab != 2 ||                                    \
        g_##SUFFIX##_cblas_call.ab != ab ||                                     \
        g_##SUFFIX##_cblas_call.x != x ||                                       \
        g_##SUFFIX##_cblas_call.scale != &scale ||                              \
        g_##SUFFIX##_cblas_call.cnorm != cnorm ||                               \
        !EQ_FN(x[0], MAKE_FN((RTYPE)((BASE) + 30), (RTYPE)((BASE) + 80))) ||    \
        !EQ_FN(x[1], MAKE_FN((RTYPE)((BASE) + 31), (RTYPE)((BASE) + 81))) ||    \
        !EQ_FN(x[2], MAKE_FN((RTYPE)((BASE) + 32), (RTYPE)((BASE) + 82))) ||    \
        !EQ_FN(x[3], MAKE_FN((RTYPE)((BASE) + 33), (RTYPE)((BASE) + 83))) ||    \
        scale != (RTYPE)((BASE) + 40) ||                                        \
        cnorm[0] != (RTYPE)((BASE) + 50) ||                                     \
        cnorm[1] != (RTYPE)((BASE) + 51) ||                                     \
        cnorm[2] != (RTYPE)((BASE) + 52) ||                                     \
        cnorm[3] != (RTYPE)((BASE) + 53) ||                                     \
        info != (BASE) + 60) {                                                  \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not forward the complex LATBS flags, cnorm, or info correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk forwards complex LATBS flags plus cnorm/scale/info through the column-major C entry\n"); \
    return 0;                                                                   \
}

DEFINE_REAL_LATBS_TESTS(slatbs, float, FB_OP_SLATBS, 100)
DEFINE_REAL_LATBS_TESTS(dlatbs, double, FB_OP_DLATBS, 300)
DEFINE_COMPLEX_LATBS_TESTS(clatbs, fb_complex_float_t, float, FB_OP_CLATBS,
                           make_cf32, cf32_eq, 500)
DEFINE_COMPLEX_LATBS_TESTS(zlatbs, fb_complex_double_t, double, FB_OP_ZLATBS,
                           make_cf64, cf64_eq, 700)

int main(void)
{
    int status = 0;

    status |= check_slatbs_fortran_to_cblas();
    status |= check_slatbs_cblas_to_fortran();
    status |= check_dlatbs_fortran_to_cblas();
    status |= check_dlatbs_cblas_to_fortran();
    status |= check_clatbs_fortran_to_cblas();
    status |= check_clatbs_cblas_to_fortran();
    status |= check_zlatbs_fortran_to_cblas();
    status |= check_zlatbs_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}