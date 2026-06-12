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

#define DEFINE_HBGVD_TESTS(SUFFIX, TYPE, REAL_TYPE, OP_ID, BASE, MAKE, REAL_PART) \
typedef int (*fb_##SUFFIX##_cblas_fn)(fb_layout_t layout, char jobz,           \
                                      fb_uplo_t uplo, int n, int ka, int kb,   \
                                      TYPE *ab, int ldab, TYPE *bb, int ldbb,  \
                                      REAL_TYPE *w, TYPE *z, int ldz);         \
typedef void (*fb_##SUFFIX##_fortran_fn)(char *jobz, char *uplo, int *n, int *ka, \
                                         int *kb, TYPE *ab, int *ldab, TYPE *bb,  \
                                         int *ldbb, REAL_TYPE *w, TYPE *z, int *ldz,\
                                         TYPE *work, int *lwork, REAL_TYPE *rwork, \
                                         int *lrwork, int *iwork, int *liwork, int *info);\
static struct {                                                                   \
    int query_calls;                                                              \
    int called;                                                                   \
    char jobz;                                                                    \
    char uplo;                                                                    \
    int n;                                                                        \
    int ka;                                                                       \
    int kb;                                                                       \
    REAL_TYPE ab_snapshot[6];                                                     \
    REAL_TYPE bb_snapshot[6];                                                     \
    int ldab;                                                                     \
    int ldbb;                                                                     \
    REAL_TYPE *w;                                                                 \
    TYPE *z;                                                                      \
    int ldz;                                                                      \
    int work_nonnull;                                                             \
    int lwork;                                                                    \
    int rwork_nonnull;                                                            \
    int lrwork;                                                                   \
    int iwork_nonnull;                                                            \
    int liwork;                                                                   \
} g_##SUFFIX##_fortran_call;                                                      \
static struct {                                                                   \
    int called;                                                                   \
    fb_layout_t layout;                                                           \
    char jobz;                                                                    \
    fb_uplo_t uplo;                                                               \
    int n;                                                                        \
    int ka;                                                                       \
    int kb;                                                                       \
    TYPE *ab;                                                                     \
    int ldab;                                                                     \
    TYPE *bb;                                                                     \
    int ldbb;                                                                     \
    REAL_TYPE *w;                                                                 \
    TYPE *z;                                                                      \
    int ldz;                                                                      \
} g_##SUFFIX##_cblas_call;                                                        \
static void stub_##SUFFIX##_fortran(char *jobz, char *uplo, int *n, int *ka,   \
                                    int *kb, TYPE *ab, int *ldab, TYPE *bb,    \
                                    int *ldbb, REAL_TYPE *w, TYPE *z, int *ldz,\
                                    TYPE *work, int *lwork, REAL_TYPE *rwork,  \
                                    int *lrwork, int *iwork, int *liwork,      \
                                    int *info)                                 \
{                                                                                 \
    size_t index = 0;                                                             \
    if (*lwork == -1 || *lrwork == -1 || *liwork == -1) {                        \
        g_##SUFFIX##_fortran_call.query_calls += 1;                               \
        *work = MAKE((REAL_TYPE)34);                                              \
        *rwork = (REAL_TYPE)21;                                                   \
        *iwork = 18;                                                              \
        *info = 0;                                                                \
        return;                                                                   \
    }                                                                             \
    for (index = 0; index < 6; ++index) {                                        \
        g_##SUFFIX##_fortran_call.ab_snapshot[index] = REAL_PART(ab[index]);     \
        g_##SUFFIX##_fortran_call.bb_snapshot[index] = REAL_PART(bb[index]);     \
    }                                                                             \
    g_##SUFFIX##_fortran_call.called += 1;                                        \
    g_##SUFFIX##_fortran_call.jobz = *jobz;                                       \
    g_##SUFFIX##_fortran_call.uplo = *uplo;                                       \
    g_##SUFFIX##_fortran_call.n = *n;                                             \
    g_##SUFFIX##_fortran_call.ka = *ka;                                           \
    g_##SUFFIX##_fortran_call.kb = *kb;                                           \
    g_##SUFFIX##_fortran_call.ldab = *ldab;                                       \
    g_##SUFFIX##_fortran_call.ldbb = *ldbb;                                       \
    g_##SUFFIX##_fortran_call.w = w;                                              \
    g_##SUFFIX##_fortran_call.z = z;                                              \
    g_##SUFFIX##_fortran_call.ldz = *ldz;                                         \
    g_##SUFFIX##_fortran_call.work_nonnull = (work != NULL);                      \
    g_##SUFFIX##_fortran_call.lwork = *lwork;                                     \
    g_##SUFFIX##_fortran_call.rwork_nonnull = (rwork != NULL);                    \
    g_##SUFFIX##_fortran_call.lrwork = *lrwork;                                   \
    g_##SUFFIX##_fortran_call.iwork_nonnull = (iwork != NULL);                    \
    g_##SUFFIX##_fortran_call.liwork = *liwork;                                   \
    ab[0] = MAKE((REAL_TYPE)0); ab[1] = MAKE((REAL_TYPE)41); ab[2] = MAKE((REAL_TYPE)42); \
    ab[3] = MAKE((REAL_TYPE)52); ab[4] = MAKE((REAL_TYPE)53); ab[5] = MAKE((REAL_TYPE)63); \
    bb[0] = MAKE((REAL_TYPE)0); bb[1] = MAKE((REAL_TYPE)71); bb[2] = MAKE((REAL_TYPE)72); \
    bb[3] = MAKE((REAL_TYPE)82); bb[4] = MAKE((REAL_TYPE)83); bb[5] = MAKE((REAL_TYPE)93); \
    w[0] = (REAL_TYPE)((BASE) + 1); w[1] = (REAL_TYPE)((BASE) + 2); w[2] = (REAL_TYPE)((BASE) + 3); \
    z[0] = MAKE((REAL_TYPE)101); z[1] = MAKE((REAL_TYPE)201); z[2] = MAKE((REAL_TYPE)301); \
    z[3] = MAKE((REAL_TYPE)102); z[4] = MAKE((REAL_TYPE)202); z[5] = MAKE((REAL_TYPE)302); \
    z[6] = MAKE((REAL_TYPE)103); z[7] = MAKE((REAL_TYPE)203); z[8] = MAKE((REAL_TYPE)303); \
    *info = 0;                                                                    \
}                                                                                 \
static int stub_##SUFFIX##_cblas(fb_layout_t layout, char jobz, fb_uplo_t uplo,\
                                 int n, int ka, int kb, TYPE *ab, int ldab,    \
                                 TYPE *bb, int ldbb, REAL_TYPE *w, TYPE *z,    \
                                 int ldz)                                      \
{                                                                                 \
    g_##SUFFIX##_cblas_call.called += 1;                                          \
    g_##SUFFIX##_cblas_call.layout = layout;                                      \
    g_##SUFFIX##_cblas_call.jobz = jobz;                                          \
    g_##SUFFIX##_cblas_call.uplo = uplo;                                          \
    g_##SUFFIX##_cblas_call.n = n;                                                \
    g_##SUFFIX##_cblas_call.ka = ka;                                              \
    g_##SUFFIX##_cblas_call.kb = kb;                                              \
    g_##SUFFIX##_cblas_call.ab = ab;                                              \
    g_##SUFFIX##_cblas_call.ldab = ldab;                                          \
    g_##SUFFIX##_cblas_call.bb = bb;                                              \
    g_##SUFFIX##_cblas_call.ldbb = ldbb;                                          \
    g_##SUFFIX##_cblas_call.w = w;                                                \
    g_##SUFFIX##_cblas_call.z = z;                                                \
    g_##SUFFIX##_cblas_call.ldz = ldz;                                            \
    w[0] = (REAL_TYPE)((BASE) + 4);                                               \
    z[0] = MAKE((REAL_TYPE)((BASE) + 5));                                         \
    return (BASE) + 6;                                                            \
}                                                                                 \
static int check_##SUFFIX##_fortran_to_cblas(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                          \
    TYPE ab[6] = { MAKE((REAL_TYPE)0), MAKE((REAL_TYPE)12), MAKE((REAL_TYPE)23), MAKE((REAL_TYPE)11), MAKE((REAL_TYPE)22), MAKE((REAL_TYPE)33) }; \
    TYPE bb[6] = { MAKE((REAL_TYPE)0), MAKE((REAL_TYPE)45), MAKE((REAL_TYPE)56), MAKE((REAL_TYPE)44), MAKE((REAL_TYPE)55), MAKE((REAL_TYPE)66) }; \
    REAL_TYPE expected_ab_in[6] = { (REAL_TYPE)0, (REAL_TYPE)11, (REAL_TYPE)12, (REAL_TYPE)22, (REAL_TYPE)23, (REAL_TYPE)33 }; \
    REAL_TYPE expected_bb_in[6] = { (REAL_TYPE)0, (REAL_TYPE)44, (REAL_TYPE)45, (REAL_TYPE)55, (REAL_TYPE)56, (REAL_TYPE)66 }; \
    REAL_TYPE expected_ab_out[6] = { (REAL_TYPE)0, (REAL_TYPE)42, (REAL_TYPE)53, (REAL_TYPE)41, (REAL_TYPE)52, (REAL_TYPE)63 }; \
    REAL_TYPE expected_bb_out[6] = { (REAL_TYPE)0, (REAL_TYPE)72, (REAL_TYPE)83, (REAL_TYPE)71, (REAL_TYPE)82, (REAL_TYPE)93 }; \
    REAL_TYPE w[3] = { 0, 0, 0 };                                                 \
    TYPE z[9] = { 0 };                                                            \
    REAL_TYPE expected_z[9] = { (REAL_TYPE)101, (REAL_TYPE)102, (REAL_TYPE)103, (REAL_TYPE)201, (REAL_TYPE)202, (REAL_TYPE)203, (REAL_TYPE)301, (REAL_TYPE)302, (REAL_TYPE)303 }; \
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
    info = thunk(FB_LAYOUT_ROW_MAJOR, 'V', FB_UPPER, 3, 1, 1, ab, 3, bb, 3, w, z, 3); \
    if (info != 0 || g_##SUFFIX##_fortran_call.query_calls != 1 ||               \
        g_##SUFFIX##_fortran_call.called != 1 ||                                  \
        g_##SUFFIX##_fortran_call.jobz != 'V' ||                                  \
        g_##SUFFIX##_fortran_call.uplo != 'U' ||                                  \
        g_##SUFFIX##_fortran_call.n != 3 || g_##SUFFIX##_fortran_call.ka != 1 ||  \
        g_##SUFFIX##_fortran_call.kb != 1 ||                                      \
        g_##SUFFIX##_fortran_call.ldab != 2 || g_##SUFFIX##_fortran_call.ldbb != 2 || \
        g_##SUFFIX##_fortran_call.w != w || g_##SUFFIX##_fortran_call.ldz != 3 ||  \
        !g_##SUFFIX##_fortran_call.work_nonnull || g_##SUFFIX##_fortran_call.lwork != 34 || \
        !g_##SUFFIX##_fortran_call.rwork_nonnull || g_##SUFFIX##_fortran_call.lrwork != 21 || \
        !g_##SUFFIX##_fortran_call.iwork_nonnull || g_##SUFFIX##_fortran_call.liwork != 18) { \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not route HBGVD calls\n"); \
        return 1;                                                                  \
    }                                                                              \
    for (index = 0; index < 6; ++index) {                                         \
        if (g_##SUFFIX##_fortran_call.ab_snapshot[index] != expected_ab_in[index] || \
            g_##SUFFIX##_fortran_call.bb_snapshot[index] != expected_bb_in[index] || \
            REAL_PART(ab[index]) != expected_ab_out[index] ||                     \
            REAL_PART(bb[index]) != expected_bb_out[index]) {                     \
            fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not translate HBGVD band storage correctly\n"); \
            return 1;                                                              \
        }                                                                          \
    }                                                                              \
    for (index = 0; index < 3; ++index) {                                         \
        if (w[index] != (REAL_TYPE)((BASE) + 1 + (int)index) ||                   \
            REAL_PART(z[index]) != expected_z[index] ||                           \
            REAL_PART(z[index + 3]) != expected_z[index + 3] ||                   \
            REAL_PART(z[index + 6]) != expected_z[index + 6]) {                   \
            fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not copy back HBGVD eigen data correctly\n"); \
            return 1;                                                              \
        }                                                                          \
    }                                                                              \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk translates A/B Hermitian band storage, allocates work, and copies back eigenvectors\n"); \
    return 0;                                                                      \
}                                                                                  \
static int check_##SUFFIX##_cblas_to_fortran(void)                                 \
{                                                                                  \
    fb_backend_vtable_t vtable;                                                    \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                         \
    char jobz = 'V';                                                               \
    char uplo = 'L';                                                               \
    int n = 2;                                                                     \
    int ka = 1;                                                                    \
    int kb = 1;                                                                    \
    TYPE ab[4] = { MAKE((REAL_TYPE)4), MAKE((REAL_TYPE)5), MAKE((REAL_TYPE)6), MAKE((REAL_TYPE)7) }; \
    TYPE bb[4] = { MAKE((REAL_TYPE)8), MAKE((REAL_TYPE)9), MAKE((REAL_TYPE)10), MAKE((REAL_TYPE)11) }; \
    int ldab = 2;                                                                  \
    int ldbb = 2;                                                                  \
    REAL_TYPE w[2] = { 0, 0 };                                                    \
    TYPE z[4] = { 0 };                                                             \
    int ldz = 2;                                                                   \
    TYPE work[16] = { 0 };                                                         \
    int lwork = 16;                                                                \
    REAL_TYPE rwork[16] = { 0 };                                                   \
    int lrwork = 16;                                                               \
    int iwork[12] = { 0 };                                                         \
    int liwork = 12;                                                               \
    int info = -999;                                                               \
    memset(&vtable, 0, sizeof(vtable));                                            \
    memset(&g_##SUFFIX##_cblas_call, 0, sizeof(g_##SUFFIX##_cblas_call));          \
    vtable.ext_ops[OP_ID][FB_CONV_CBLAS] =                                         \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_cblas;                      \
    fb_install_conv_thunks(&vtable, OP_ID);                                        \
    thunk = (fb_##SUFFIX##_fortran_fn)vtable.ext_ops[OP_ID][FB_CONV_FORTRAN];      \
    if (!thunk) {                                                                  \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk was not installed\n"); \
        return 1;                                                                  \
    }                                                                              \
    thunk(&jobz, &uplo, &n, &ka, &kb, ab, &ldab, bb, &ldbb, w, z, &ldz, work, &lwork, rwork, &lrwork, iwork, &liwork, &info); \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                     \
        g_##SUFFIX##_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||                   \
        g_##SUFFIX##_cblas_call.jobz != 'V' ||                                     \
        g_##SUFFIX##_cblas_call.uplo != FB_LOWER ||                                \
        g_##SUFFIX##_cblas_call.n != 2 || g_##SUFFIX##_cblas_call.ka != 1 ||       \
        g_##SUFFIX##_cblas_call.kb != 1 ||                                         \
        g_##SUFFIX##_cblas_call.ab != ab || g_##SUFFIX##_cblas_call.ldab != 2 ||   \
        g_##SUFFIX##_cblas_call.bb != bb || g_##SUFFIX##_cblas_call.ldbb != 2 ||   \
        g_##SUFFIX##_cblas_call.w != w || g_##SUFFIX##_cblas_call.z != z ||        \
        g_##SUFFIX##_cblas_call.ldz != 2 || info != (BASE) + 6 ||                  \
        w[0] != (REAL_TYPE)((BASE) + 4) || REAL_PART(z[0]) != (REAL_TYPE)((BASE) + 5)) { \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not forward HBGVD arguments correctly\n"); \
        return 1;                                                                  \
    }                                                                              \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk forwards HBGVD arguments\n"); \
    return 0;                                                                      \
}

DEFINE_HBGVD_TESTS(chbgvd, fb_complex_float_t, float, FB_OP_CHBGVD, 100, make_cfloat, cfloat_real)
DEFINE_HBGVD_TESTS(zhbgvd, fb_complex_double_t, double, FB_OP_ZHBGVD, 300, make_cdouble, cdouble_real)

int main(void)
{
    int status = 0;

    status |= check_chbgvd_fortran_to_cblas();
    status |= check_chbgvd_cblas_to_fortran();
    status |= check_zhbgvd_fortran_to_cblas();
    status |= check_zhbgvd_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}