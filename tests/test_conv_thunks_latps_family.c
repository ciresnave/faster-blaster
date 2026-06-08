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

#define DEFINE_REAL_LATPS_TESTS(SUFFIX, TYPE, OP_ID, BASE)                     \
typedef int (*fb_##SUFFIX##_cblas_fn)(fb_layout_t layout, char uplo,           \
                                      char trans, char diag, char normin,      \
                                      int n, TYPE *ap, TYPE *x, TYPE *scale,  \
                                      TYPE *cnorm);                            \
typedef void (*fb_##SUFFIX##_fortran_fn)(char *uplo, char *trans, char *diag,  \
                                         char *normin, int *n, TYPE *ap,      \
                                         TYPE *x, TYPE *scale, TYPE *cnorm,   \
                                         int *info);                          \
static struct {                                                                 \
    int called;                                                                 \
    char uplo;                                                                  \
    char trans;                                                                 \
    char diag;                                                                  \
    char normin;                                                                \
    int n;                                                                      \
    TYPE *ap;                                                                   \
    TYPE *x;                                                                    \
    TYPE *scale;                                                                \
    TYPE *cnorm;                                                                \
    TYPE ap_snapshot[6];                                                        \
    TYPE cnorm_snapshot[3];                                                     \
} g_##SUFFIX##_fortran_call;                                                    \
static struct {                                                                 \
    int called;                                                                 \
    fb_layout_t layout;                                                         \
    char uplo;                                                                  \
    char trans;                                                                 \
    char diag;                                                                  \
    char normin;                                                                \
    int n;                                                                      \
    TYPE *ap;                                                                   \
    TYPE *x;                                                                    \
    TYPE *scale;                                                                \
    TYPE *cnorm;                                                                \
} g_##SUFFIX##_cblas_call;                                                      \
static void stub_##SUFFIX##_fortran(char *uplo, char *trans, char *diag,       \
                                    char *normin, int *n, TYPE *ap, TYPE *x,  \
                                    TYPE *scale, TYPE *cnorm, int *info)      \
{                                                                               \
    int idx;                                                                    \
    g_##SUFFIX##_fortran_call.called += 1;                                      \
    g_##SUFFIX##_fortran_call.uplo = *uplo;                                     \
    g_##SUFFIX##_fortran_call.trans = *trans;                                   \
    g_##SUFFIX##_fortran_call.diag = *diag;                                     \
    g_##SUFFIX##_fortran_call.normin = *normin;                                 \
    g_##SUFFIX##_fortran_call.n = *n;                                           \
    g_##SUFFIX##_fortran_call.ap = ap;                                          \
    g_##SUFFIX##_fortran_call.x = x;                                            \
    g_##SUFFIX##_fortran_call.scale = scale;                                    \
    g_##SUFFIX##_fortran_call.cnorm = cnorm;                                    \
    for (idx = 0; idx < 6; ++idx) {                                             \
        g_##SUFFIX##_fortran_call.ap_snapshot[idx] = ap[idx];                   \
    }                                                                           \
    for (idx = 0; idx < 3; ++idx) {                                             \
        g_##SUFFIX##_fortran_call.cnorm_snapshot[idx] = cnorm[idx];             \
    }                                                                           \
    x[0] = (TYPE)((BASE) + 10);                                                 \
    x[1] = (TYPE)((BASE) + 11);                                                 \
    x[2] = (TYPE)((BASE) + 12);                                                 \
    *scale = (TYPE)((BASE) + 20);                                               \
    *info = 0;                                                                  \
}                                                                               \
static int stub_##SUFFIX##_cblas(fb_layout_t layout, char uplo, char trans,    \
                                 char diag, char normin, int n, TYPE *ap,      \
                                 TYPE *x, TYPE *scale, TYPE *cnorm)           \
{                                                                               \
    g_##SUFFIX##_cblas_call.called += 1;                                        \
    g_##SUFFIX##_cblas_call.layout = layout;                                    \
    g_##SUFFIX##_cblas_call.uplo = uplo;                                        \
    g_##SUFFIX##_cblas_call.trans = trans;                                      \
    g_##SUFFIX##_cblas_call.diag = diag;                                        \
    g_##SUFFIX##_cblas_call.normin = normin;                                    \
    g_##SUFFIX##_cblas_call.n = n;                                              \
    g_##SUFFIX##_cblas_call.ap = ap;                                            \
    g_##SUFFIX##_cblas_call.x = x;                                              \
    g_##SUFFIX##_cblas_call.scale = scale;                                      \
    g_##SUFFIX##_cblas_call.cnorm = cnorm;                                      \
    x[0] = (TYPE)((BASE) + 30);                                                 \
    x[1] = (TYPE)((BASE) + 31);                                                 \
    x[2] = (TYPE)((BASE) + 32);                                                 \
    *scale = (TYPE)((BASE) + 40);                                               \
    cnorm[0] = (TYPE)((BASE) + 50);                                             \
    cnorm[1] = (TYPE)((BASE) + 51);                                             \
    cnorm[2] = (TYPE)((BASE) + 52);                                             \
    return (BASE) + 60;                                                         \
}                                                                               \
static int check_##SUFFIX##_fortran_to_cblas(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                        \
    TYPE ap[6] = { (TYPE)1, (TYPE)2, (TYPE)3, (TYPE)4, (TYPE)5, (TYPE)6 };    \
    TYPE x[3] = { (TYPE)7, (TYPE)8, (TYPE)9 };                                  \
    TYPE scale = (TYPE)10;                                                      \
    TYPE cnorm[3] = { (TYPE)11, (TYPE)12, (TYPE)13 };                           \
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
    status = thunk(FB_LAYOUT_ROW_MAJOR, 'U', 'T', 'N', 'Y', 3, ap, x, &scale, cnorm); \
    if (status != 0 || g_##SUFFIX##_fortran_call.called != 1 ||                \
        g_##SUFFIX##_fortran_call.uplo != 'U' ||                                \
        g_##SUFFIX##_fortran_call.trans != 'T' ||                               \
        g_##SUFFIX##_fortran_call.diag != 'N' ||                                \
        g_##SUFFIX##_fortran_call.normin != 'Y' ||                              \
        g_##SUFFIX##_fortran_call.n != 3 ||                                     \
        g_##SUFFIX##_fortran_call.ap == ap ||                                   \
        g_##SUFFIX##_fortran_call.x != x ||                                     \
        g_##SUFFIX##_fortran_call.scale != &scale ||                            \
        g_##SUFFIX##_fortran_call.cnorm != cnorm ||                             \
        g_##SUFFIX##_fortran_call.ap_snapshot[0] != (TYPE)1 ||                  \
        g_##SUFFIX##_fortran_call.ap_snapshot[1] != (TYPE)2 ||                  \
        g_##SUFFIX##_fortran_call.ap_snapshot[2] != (TYPE)4 ||                  \
        g_##SUFFIX##_fortran_call.ap_snapshot[3] != (TYPE)3 ||                  \
        g_##SUFFIX##_fortran_call.ap_snapshot[4] != (TYPE)5 ||                  \
        g_##SUFFIX##_fortran_call.ap_snapshot[5] != (TYPE)6 ||                  \
        g_##SUFFIX##_fortran_call.cnorm_snapshot[0] != (TYPE)11 ||              \
        g_##SUFFIX##_fortran_call.cnorm_snapshot[1] != (TYPE)12 ||              \
        g_##SUFFIX##_fortran_call.cnorm_snapshot[2] != (TYPE)13 ||              \
        ap[0] != (TYPE)1 || ap[1] != (TYPE)2 || ap[2] != (TYPE)3 || ap[3] != (TYPE)4 || \
        ap[4] != (TYPE)5 || ap[5] != (TYPE)6 ||                                 \
        x[0] != (TYPE)((BASE) + 10) || x[1] != (TYPE)((BASE) + 11) ||           \
        x[2] != (TYPE)((BASE) + 12) || scale != (TYPE)((BASE) + 20)) {         \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not translate row-major packed storage or preserve precomputed cnorm correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk translates row-major packed storage and forwards precomputed cnorm\n"); \
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
    int n = 3;                                                                  \
    int info = -999;                                                            \
    TYPE ap[6] = { (TYPE)1, (TYPE)2, (TYPE)3, (TYPE)4, (TYPE)5, (TYPE)6 };    \
    TYPE x[3] = { (TYPE)0, (TYPE)0, (TYPE)0 };                                  \
    TYPE scale = (TYPE)0;                                                       \
    TYPE cnorm[3] = { (TYPE)0, (TYPE)0, (TYPE)0 };                              \
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
    thunk(&uplo, &trans, &diag, &normin, &n, ap, x, &scale, cnorm, &info);     \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                  \
        g_##SUFFIX##_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||                \
        g_##SUFFIX##_cblas_call.uplo != 'L' ||                                  \
        g_##SUFFIX##_cblas_call.trans != 'N' ||                                 \
        g_##SUFFIX##_cblas_call.diag != 'U' ||                                  \
        g_##SUFFIX##_cblas_call.normin != 'N' ||                                \
        g_##SUFFIX##_cblas_call.n != 3 ||                                       \
        g_##SUFFIX##_cblas_call.ap != ap ||                                     \
        g_##SUFFIX##_cblas_call.x != x ||                                       \
        g_##SUFFIX##_cblas_call.scale != &scale ||                              \
        g_##SUFFIX##_cblas_call.cnorm != cnorm ||                               \
        x[0] != (TYPE)((BASE) + 30) || x[1] != (TYPE)((BASE) + 31) ||           \
        x[2] != (TYPE)((BASE) + 32) || scale != (TYPE)((BASE) + 40) ||         \
        cnorm[0] != (TYPE)((BASE) + 50) || cnorm[1] != (TYPE)((BASE) + 51) ||   \
        cnorm[2] != (TYPE)((BASE) + 52) || info != (BASE) + 60) {              \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not forward the packed LATPS flags, cnorm, or info correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk forwards packed LATPS flags plus cnorm/scale/info through the column-major C entry\n"); \
    return 0;                                                                   \
}

#define DEFINE_COMPLEX_LATPS_TESTS(SUFFIX, CTYPE, RTYPE, OP_ID, MAKE_FN, EQ_FN, BASE) \
typedef int (*fb_##SUFFIX##_cblas_fn)(fb_layout_t layout, char uplo,           \
                                      char trans, char diag, char normin,      \
                                      int n, CTYPE *ap, CTYPE *x, RTYPE *scale,\
                                      RTYPE *cnorm);                           \
typedef void (*fb_##SUFFIX##_fortran_fn)(char *uplo, char *trans, char *diag,  \
                                         char *normin, int *n, CTYPE *ap,     \
                                         CTYPE *x, RTYPE *scale, RTYPE *cnorm,\
                                         int *info);                          \
static struct {                                                                 \
    int called;                                                                 \
    char uplo;                                                                  \
    char trans;                                                                 \
    char diag;                                                                  \
    char normin;                                                                \
    int n;                                                                      \
    CTYPE *ap;                                                                  \
    CTYPE *x;                                                                   \
    RTYPE *scale;                                                               \
    RTYPE *cnorm;                                                               \
    CTYPE ap_snapshot[6];                                                       \
    RTYPE cnorm_snapshot[3];                                                    \
} g_##SUFFIX##_fortran_call;                                                    \
static struct {                                                                 \
    int called;                                                                 \
    fb_layout_t layout;                                                         \
    char uplo;                                                                  \
    char trans;                                                                 \
    char diag;                                                                  \
    char normin;                                                                \
    int n;                                                                      \
    CTYPE *ap;                                                                  \
    CTYPE *x;                                                                   \
    RTYPE *scale;                                                               \
    RTYPE *cnorm;                                                               \
} g_##SUFFIX##_cblas_call;                                                      \
static void stub_##SUFFIX##_fortran(char *uplo, char *trans, char *diag,       \
                                    char *normin, int *n, CTYPE *ap, CTYPE *x,\
                                    RTYPE *scale, RTYPE *cnorm, int *info)    \
{                                                                               \
    int idx;                                                                    \
    g_##SUFFIX##_fortran_call.called += 1;                                      \
    g_##SUFFIX##_fortran_call.uplo = *uplo;                                     \
    g_##SUFFIX##_fortran_call.trans = *trans;                                   \
    g_##SUFFIX##_fortran_call.diag = *diag;                                     \
    g_##SUFFIX##_fortran_call.normin = *normin;                                 \
    g_##SUFFIX##_fortran_call.n = *n;                                           \
    g_##SUFFIX##_fortran_call.ap = ap;                                          \
    g_##SUFFIX##_fortran_call.x = x;                                            \
    g_##SUFFIX##_fortran_call.scale = scale;                                    \
    g_##SUFFIX##_fortran_call.cnorm = cnorm;                                    \
    for (idx = 0; idx < 6; ++idx) {                                             \
        g_##SUFFIX##_fortran_call.ap_snapshot[idx] = ap[idx];                   \
    }                                                                           \
    for (idx = 0; idx < 3; ++idx) {                                             \
        g_##SUFFIX##_fortran_call.cnorm_snapshot[idx] = cnorm[idx];             \
    }                                                                           \
    x[0] = MAKE_FN((RTYPE)((BASE) + 10), (RTYPE)((BASE) + 60));                \
    x[1] = MAKE_FN((RTYPE)((BASE) + 11), (RTYPE)((BASE) + 61));                \
    x[2] = MAKE_FN((RTYPE)((BASE) + 12), (RTYPE)((BASE) + 62));                \
    *scale = (RTYPE)((BASE) + 20);                                              \
    *info = 0;                                                                  \
}                                                                               \
static int stub_##SUFFIX##_cblas(fb_layout_t layout, char uplo, char trans,    \
                                 char diag, char normin, int n, CTYPE *ap,     \
                                 CTYPE *x, RTYPE *scale, RTYPE *cnorm)        \
{                                                                               \
    g_##SUFFIX##_cblas_call.called += 1;                                        \
    g_##SUFFIX##_cblas_call.layout = layout;                                    \
    g_##SUFFIX##_cblas_call.uplo = uplo;                                        \
    g_##SUFFIX##_cblas_call.trans = trans;                                      \
    g_##SUFFIX##_cblas_call.diag = diag;                                        \
    g_##SUFFIX##_cblas_call.normin = normin;                                    \
    g_##SUFFIX##_cblas_call.n = n;                                              \
    g_##SUFFIX##_cblas_call.ap = ap;                                            \
    g_##SUFFIX##_cblas_call.x = x;                                              \
    g_##SUFFIX##_cblas_call.scale = scale;                                      \
    g_##SUFFIX##_cblas_call.cnorm = cnorm;                                      \
    x[0] = MAKE_FN((RTYPE)((BASE) + 30), (RTYPE)((BASE) + 80));                \
    x[1] = MAKE_FN((RTYPE)((BASE) + 31), (RTYPE)((BASE) + 81));                \
    x[2] = MAKE_FN((RTYPE)((BASE) + 32), (RTYPE)((BASE) + 82));                \
    *scale = (RTYPE)((BASE) + 40);                                              \
    cnorm[0] = (RTYPE)((BASE) + 50);                                            \
    cnorm[1] = (RTYPE)((BASE) + 51);                                            \
    cnorm[2] = (RTYPE)((BASE) + 52);                                            \
    return (BASE) + 60;                                                         \
}                                                                               \
static int check_##SUFFIX##_fortran_to_cblas(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                        \
    CTYPE ap[6];                                                                \
    CTYPE x[3];                                                                 \
    RTYPE scale = (RTYPE)10;                                                    \
    RTYPE cnorm[3] = { (RTYPE)11, (RTYPE)12, (RTYPE)13 };                       \
    int status = 0;                                                             \
    ap[0] = MAKE_FN((RTYPE)1, (RTYPE)11);                                       \
    ap[1] = MAKE_FN((RTYPE)2, (RTYPE)12);                                       \
    ap[2] = MAKE_FN((RTYPE)3, (RTYPE)13);                                       \
    ap[3] = MAKE_FN((RTYPE)4, (RTYPE)14);                                       \
    ap[4] = MAKE_FN((RTYPE)5, (RTYPE)15);                                       \
    ap[5] = MAKE_FN((RTYPE)6, (RTYPE)16);                                       \
    x[0] = MAKE_FN((RTYPE)7, (RTYPE)17);                                        \
    x[1] = MAKE_FN((RTYPE)8, (RTYPE)18);                                        \
    x[2] = MAKE_FN((RTYPE)9, (RTYPE)19);                                        \
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
    status = thunk(FB_LAYOUT_ROW_MAJOR, 'U', 'C', 'N', 'Y', 3, ap, x, &scale, cnorm); \
    if (status != 0 || g_##SUFFIX##_fortran_call.called != 1 ||                \
        g_##SUFFIX##_fortran_call.uplo != 'U' ||                                \
        g_##SUFFIX##_fortran_call.trans != 'C' ||                               \
        g_##SUFFIX##_fortran_call.diag != 'N' ||                                \
        g_##SUFFIX##_fortran_call.normin != 'Y' ||                              \
        g_##SUFFIX##_fortran_call.n != 3 ||                                     \
        g_##SUFFIX##_fortran_call.ap == ap ||                                   \
        g_##SUFFIX##_fortran_call.x != x ||                                     \
        g_##SUFFIX##_fortran_call.scale != &scale ||                            \
        g_##SUFFIX##_fortran_call.cnorm != cnorm ||                             \
        !EQ_FN(g_##SUFFIX##_fortran_call.ap_snapshot[0],                        \
               MAKE_FN((RTYPE)1, (RTYPE)11)) ||                                 \
        !EQ_FN(g_##SUFFIX##_fortran_call.ap_snapshot[1],                        \
               MAKE_FN((RTYPE)2, (RTYPE)12)) ||                                 \
        !EQ_FN(g_##SUFFIX##_fortran_call.ap_snapshot[2],                        \
               MAKE_FN((RTYPE)4, (RTYPE)14)) ||                                 \
        !EQ_FN(g_##SUFFIX##_fortran_call.ap_snapshot[3],                        \
               MAKE_FN((RTYPE)3, (RTYPE)13)) ||                                 \
        !EQ_FN(g_##SUFFIX##_fortran_call.ap_snapshot[4],                        \
               MAKE_FN((RTYPE)5, (RTYPE)15)) ||                                 \
        !EQ_FN(g_##SUFFIX##_fortran_call.ap_snapshot[5],                        \
               MAKE_FN((RTYPE)6, (RTYPE)16)) ||                                 \
        g_##SUFFIX##_fortran_call.cnorm_snapshot[0] != (RTYPE)11 ||             \
        g_##SUFFIX##_fortran_call.cnorm_snapshot[1] != (RTYPE)12 ||             \
        g_##SUFFIX##_fortran_call.cnorm_snapshot[2] != (RTYPE)13 ||             \
        !EQ_FN(ap[0], MAKE_FN((RTYPE)1, (RTYPE)11)) ||                          \
        !EQ_FN(ap[1], MAKE_FN((RTYPE)2, (RTYPE)12)) ||                          \
        !EQ_FN(ap[2], MAKE_FN((RTYPE)3, (RTYPE)13)) ||                          \
        !EQ_FN(ap[3], MAKE_FN((RTYPE)4, (RTYPE)14)) ||                          \
        !EQ_FN(ap[4], MAKE_FN((RTYPE)5, (RTYPE)15)) ||                          \
        !EQ_FN(ap[5], MAKE_FN((RTYPE)6, (RTYPE)16)) ||                          \
        !EQ_FN(x[0], MAKE_FN((RTYPE)((BASE) + 10), (RTYPE)((BASE) + 60))) ||    \
        !EQ_FN(x[1], MAKE_FN((RTYPE)((BASE) + 11), (RTYPE)((BASE) + 61))) ||    \
        !EQ_FN(x[2], MAKE_FN((RTYPE)((BASE) + 12), (RTYPE)((BASE) + 62))) ||    \
        scale != (RTYPE)((BASE) + 20)) {                                        \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not translate row-major complex packed storage or preserve precomputed cnorm correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk translates row-major complex packed storage and forwards precomputed cnorm\n"); \
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
    int n = 3;                                                                  \
    int info = -999;                                                            \
    CTYPE ap[6];                                                                \
    CTYPE x[3];                                                                 \
    RTYPE scale = (RTYPE)0;                                                     \
    RTYPE cnorm[3] = { (RTYPE)0, (RTYPE)0, (RTYPE)0 };                          \
    ap[0] = MAKE_FN((RTYPE)1, (RTYPE)11);                                       \
    ap[1] = MAKE_FN((RTYPE)2, (RTYPE)12);                                       \
    ap[2] = MAKE_FN((RTYPE)3, (RTYPE)13);                                       \
    ap[3] = MAKE_FN((RTYPE)4, (RTYPE)14);                                       \
    ap[4] = MAKE_FN((RTYPE)5, (RTYPE)15);                                       \
    ap[5] = MAKE_FN((RTYPE)6, (RTYPE)16);                                       \
    x[0] = MAKE_FN((RTYPE)0, (RTYPE)0);                                         \
    x[1] = MAKE_FN((RTYPE)0, (RTYPE)0);                                         \
    x[2] = MAKE_FN((RTYPE)0, (RTYPE)0);                                         \
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
    thunk(&uplo, &trans, &diag, &normin, &n, ap, x, &scale, cnorm, &info);     \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                  \
        g_##SUFFIX##_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||                \
        g_##SUFFIX##_cblas_call.uplo != 'L' ||                                  \
        g_##SUFFIX##_cblas_call.trans != 'N' ||                                 \
        g_##SUFFIX##_cblas_call.diag != 'U' ||                                  \
        g_##SUFFIX##_cblas_call.normin != 'N' ||                                \
        g_##SUFFIX##_cblas_call.n != 3 ||                                       \
        g_##SUFFIX##_cblas_call.ap != ap ||                                     \
        g_##SUFFIX##_cblas_call.x != x ||                                       \
        g_##SUFFIX##_cblas_call.scale != &scale ||                              \
        g_##SUFFIX##_cblas_call.cnorm != cnorm ||                               \
        !EQ_FN(x[0], MAKE_FN((RTYPE)((BASE) + 30), (RTYPE)((BASE) + 80))) ||    \
        !EQ_FN(x[1], MAKE_FN((RTYPE)((BASE) + 31), (RTYPE)((BASE) + 81))) ||    \
        !EQ_FN(x[2], MAKE_FN((RTYPE)((BASE) + 32), (RTYPE)((BASE) + 82))) ||    \
        scale != (RTYPE)((BASE) + 40) ||                                        \
        cnorm[0] != (RTYPE)((BASE) + 50) ||                                     \
        cnorm[1] != (RTYPE)((BASE) + 51) ||                                     \
        cnorm[2] != (RTYPE)((BASE) + 52) ||                                     \
        info != (BASE) + 60) {                                                  \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not forward the complex packed LATPS flags, cnorm, or info correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk forwards complex packed LATPS flags plus cnorm/scale/info through the column-major C entry\n"); \
    return 0;                                                                   \
}

DEFINE_REAL_LATPS_TESTS(slatps, float, FB_OP_SLATPS, 100)
DEFINE_REAL_LATPS_TESTS(dlatps, double, FB_OP_DLATPS, 300)
DEFINE_COMPLEX_LATPS_TESTS(clatps, fb_complex_float_t, float, FB_OP_CLATPS,
                           make_cf32, cf32_eq, 500)
DEFINE_COMPLEX_LATPS_TESTS(zlatps, fb_complex_double_t, double, FB_OP_ZLATPS,
                           make_cf64, cf64_eq, 700)

int main(void)
{
    int status = 0;

    status |= check_slatps_fortran_to_cblas();
    status |= check_slatps_cblas_to_fortran();
    status |= check_dlatps_fortran_to_cblas();
    status |= check_dlatps_cblas_to_fortran();
    status |= check_clatps_fortran_to_cblas();
    status |= check_clatps_cblas_to_fortran();
    status |= check_zlatps_fortran_to_cblas();
    status |= check_zlatps_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}