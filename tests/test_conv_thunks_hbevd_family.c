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

#define DEFINE_HBEVD_TESTS(SUFFIX, TYPE, REAL_TYPE, OP_ID, BASE, MAKE, REAL_PART) \
typedef int (*fb_##SUFFIX##_cblas_fn)(fb_layout_t layout, char jobz,            \
                                      fb_uplo_t uplo, int n, int kd, TYPE *ab,  \
                                      int ldab, REAL_TYPE *w, TYPE *z, int ldz); \
typedef void (*fb_##SUFFIX##_fortran_fn)(char *jobz, char *uplo, int *n, int *kd, \
                                         TYPE *ab, int *ldab, REAL_TYPE *w, TYPE *z, \
                                         int *ldz, TYPE *work, int *lwork,       \
                                         REAL_TYPE *rwork, int *lrwork,          \
                                         int *iwork, int *liwork, int *info);    \
static struct {                                                                     \
    int query_calls;                                                                \
    int calls;                                                                      \
    char jobz;                                                                      \
    char uplo;                                                                      \
    int n;                                                                          \
    int kd;                                                                         \
    REAL_TYPE ab_snapshot[6];                                                       \
    int ldab;                                                                       \
    REAL_TYPE *w;                                                                   \
    int ldz;                                                                        \
    int work_nonnull;                                                               \
    int lwork;                                                                      \
    int rwork_nonnull;                                                              \
    int lrwork;                                                                     \
    int iwork_nonnull;                                                              \
    int liwork;                                                                     \
} g_##SUFFIX##_fortran_call;                                                        \
static struct {                                                                     \
    int called;                                                                     \
    fb_layout_t layout;                                                             \
    char jobz;                                                                      \
    fb_uplo_t uplo;                                                                 \
    int n;                                                                          \
    int kd;                                                                         \
    TYPE *ab;                                                                       \
    int ldab;                                                                       \
    REAL_TYPE *w;                                                                   \
    TYPE *z;                                                                        \
    int ldz;                                                                        \
} g_##SUFFIX##_cblas_call;                                                          \
static void stub_##SUFFIX##_fortran(char *jobz, char *uplo, int *n, int *kd,     \
                                    TYPE *ab, int *ldab, REAL_TYPE *w, TYPE *z,  \
                                    int *ldz, TYPE *work, int *lwork,            \
                                    REAL_TYPE *rwork, int *lrwork, int *iwork,   \
                                    int *liwork, int *info)                       \
{                                                                                   \
    size_t index = 0;                                                               \
    if (*lwork == -1 || *lrwork == -1 || *liwork == -1) {                          \
        g_##SUFFIX##_fortran_call.query_calls += 1;                                 \
        *work = MAKE((REAL_TYPE)34, (REAL_TYPE)0);                                  \
        *rwork = (REAL_TYPE)21;                                                     \
        *iwork = 18;                                                                \
        *info = 0;                                                                  \
        return;                                                                     \
    }                                                                               \
    for (index = 0; index < 6; ++index) {                                          \
        g_##SUFFIX##_fortran_call.ab_snapshot[index] = REAL_PART(ab[index]);       \
    }                                                                               \
    g_##SUFFIX##_fortran_call.calls += 1;                                           \
    g_##SUFFIX##_fortran_call.jobz = *jobz;                                         \
    g_##SUFFIX##_fortran_call.uplo = *uplo;                                         \
    g_##SUFFIX##_fortran_call.n = *n;                                               \
    g_##SUFFIX##_fortran_call.kd = *kd;                                             \
    g_##SUFFIX##_fortran_call.ldab = *ldab;                                         \
    g_##SUFFIX##_fortran_call.w = w;                                                \
    g_##SUFFIX##_fortran_call.ldz = *ldz;                                           \
    g_##SUFFIX##_fortran_call.work_nonnull = (work != NULL);                        \
    g_##SUFFIX##_fortran_call.lwork = *lwork;                                       \
    g_##SUFFIX##_fortran_call.rwork_nonnull = (rwork != NULL);                      \
    g_##SUFFIX##_fortran_call.lrwork = *lrwork;                                     \
    g_##SUFFIX##_fortran_call.iwork_nonnull = (iwork != NULL);                      \
    g_##SUFFIX##_fortran_call.liwork = *liwork;                                     \
    w[0] = (REAL_TYPE)((BASE) + 1); w[1] = (REAL_TYPE)((BASE) + 2); w[2] = (REAL_TYPE)((BASE) + 3); \
    z[0] = MAKE((REAL_TYPE)101, (REAL_TYPE)1); z[1] = MAKE((REAL_TYPE)201, (REAL_TYPE)4); z[2] = MAKE((REAL_TYPE)301, (REAL_TYPE)7); \
    z[3] = MAKE((REAL_TYPE)102, (REAL_TYPE)2); z[4] = MAKE((REAL_TYPE)202, (REAL_TYPE)5); z[5] = MAKE((REAL_TYPE)302, (REAL_TYPE)8); \
    z[6] = MAKE((REAL_TYPE)103, (REAL_TYPE)3); z[7] = MAKE((REAL_TYPE)203, (REAL_TYPE)6); z[8] = MAKE((REAL_TYPE)303, (REAL_TYPE)9); \
    *info = 0;                                                                      \
}                                                                                   \
static int stub_##SUFFIX##_cblas(fb_layout_t layout, char jobz, fb_uplo_t uplo,  \
                                 int n, int kd, TYPE *ab, int ldab, REAL_TYPE *w, \
                                 TYPE *z, int ldz)                                \
{                                                                                   \
    g_##SUFFIX##_cblas_call.called += 1;                                           \
    g_##SUFFIX##_cblas_call.layout = layout;                                       \
    g_##SUFFIX##_cblas_call.jobz = jobz;                                           \
    g_##SUFFIX##_cblas_call.uplo = uplo;                                           \
    g_##SUFFIX##_cblas_call.n = n;                                                 \
    g_##SUFFIX##_cblas_call.kd = kd;                                               \
    g_##SUFFIX##_cblas_call.ab = ab;                                               \
    g_##SUFFIX##_cblas_call.ldab = ldab;                                           \
    g_##SUFFIX##_cblas_call.w = w;                                                 \
    g_##SUFFIX##_cblas_call.z = z;                                                 \
    g_##SUFFIX##_cblas_call.ldz = ldz;                                             \
    w[0] = (REAL_TYPE)((BASE) + 4);                                                \
    z[0] = MAKE((REAL_TYPE)((BASE) + 5), (REAL_TYPE)((BASE) + 15));               \
    return (BASE) + 6;                                                             \
}                                                                                   \
static int check_##SUFFIX##_fortran_to_cblas(void)                                 \
{                                                                                   \
    fb_backend_vtable_t vtable;                                                    \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                           \
    TYPE ab[6] = { MAKE((REAL_TYPE)0, (REAL_TYPE)0), MAKE((REAL_TYPE)12, (REAL_TYPE)2), MAKE((REAL_TYPE)23, (REAL_TYPE)3), \
                   MAKE((REAL_TYPE)11, (REAL_TYPE)1), MAKE((REAL_TYPE)22, (REAL_TYPE)4), MAKE((REAL_TYPE)33, (REAL_TYPE)6) }; \
    REAL_TYPE expected_ab_in[6] = { (REAL_TYPE)0, (REAL_TYPE)11, (REAL_TYPE)12,   \
                                    (REAL_TYPE)22, (REAL_TYPE)23, (REAL_TYPE)33 }; \
    REAL_TYPE w[3] = { 0, 0, 0 };                                                  \
    TYPE z[9] = { 0 };                                                             \
    REAL_TYPE expected_z[9] = { (REAL_TYPE)101, (REAL_TYPE)102, (REAL_TYPE)103,   \
                                (REAL_TYPE)201, (REAL_TYPE)202, (REAL_TYPE)203,   \
                                (REAL_TYPE)301, (REAL_TYPE)302, (REAL_TYPE)303 }; \
    int info = 0;                                                                  \
    size_t index = 0;                                                              \
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
    info = thunk(FB_LAYOUT_ROW_MAJOR, 'V', FB_UPPER, 3, 1, ab, 3, w, z, 3);       \
    if (info != 0 || g_##SUFFIX##_fortran_call.query_calls != 1 ||                \
        g_##SUFFIX##_fortran_call.calls != 1 ||                                   \
        g_##SUFFIX##_fortran_call.jobz != 'V' ||                                  \
        g_##SUFFIX##_fortran_call.uplo != 'U' ||                                  \
        g_##SUFFIX##_fortran_call.n != 3 || g_##SUFFIX##_fortran_call.kd != 1 ||  \
        g_##SUFFIX##_fortran_call.ldab != 2 ||                                    \
        g_##SUFFIX##_fortran_call.w != w || g_##SUFFIX##_fortran_call.ldz != 3 || \
        !g_##SUFFIX##_fortran_call.work_nonnull || g_##SUFFIX##_fortran_call.lwork != 34 || \
        !g_##SUFFIX##_fortran_call.rwork_nonnull || g_##SUFFIX##_fortran_call.lrwork != 21 || \
        !g_##SUFFIX##_fortran_call.iwork_nonnull || g_##SUFFIX##_fortran_call.liwork != 18) { \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not route HBEVD calls\n"); \
        return 1;                                                                  \
    }                                                                              \
    for (index = 0; index < 6; ++index) {                                         \
        if (g_##SUFFIX##_fortran_call.ab_snapshot[index] != expected_ab_in[index]) { \
            fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not translate HBEVD band storage correctly\n"); \
            return 1;                                                              \
        }                                                                          \
    }                                                                              \
    for (index = 0; index < 3; ++index) {                                         \
        if (w[index] != (REAL_TYPE)((BASE) + 1 + (int)index) ||                   \
            REAL_PART(z[index]) != expected_z[index] ||                           \
            REAL_PART(z[index + 3]) != expected_z[index + 3] ||                   \
            REAL_PART(z[index + 6]) != expected_z[index + 6]) {                   \
            fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not copy back HBEVD eigen data correctly\n"); \
            return 1;                                                              \
        }                                                                          \
    }                                                                              \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk queries work, translates Hermitian band storage, and copies back eigenvectors\n"); \
    return 0;                                                                      \
}                                                                                  \
static int check_##SUFFIX##_cblas_to_fortran(void)                                 \
{                                                                                   \
    fb_backend_vtable_t vtable;                                                    \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                         \
    char jobz = 'N';                                                               \
    char uplo = 'L';                                                               \
    int n = 2;                                                                     \
    int kd = 1;                                                                    \
    TYPE ab[4] = { MAKE((REAL_TYPE)4, (REAL_TYPE)1), MAKE((REAL_TYPE)5, (REAL_TYPE)2), MAKE((REAL_TYPE)6, (REAL_TYPE)3), MAKE((REAL_TYPE)7, (REAL_TYPE)4) }; \
    int ldab = 2;                                                                  \
    REAL_TYPE w[2] = { 0, 0 };                                                     \
    TYPE z[1] = { MAKE((REAL_TYPE)0, (REAL_TYPE)0) };                             \
    int ldz = 1;                                                                   \
    TYPE work[34];                                                                 \
    int lwork = 34;                                                                \
    REAL_TYPE rwork[21] = { 0 };                                                   \
    int lrwork = 21;                                                               \
    int iwork[18] = { 0 };                                                         \
    int liwork = 18;                                                               \
    int info = -999;                                                               \
    memset(work, 0, sizeof(work));                                                 \
    memset(&vtable, 0, sizeof(vtable));                                            \
    memset(&g_##SUFFIX##_cblas_call, 0, sizeof(g_##SUFFIX##_cblas_call));         \
    vtable.ext_ops[OP_ID][FB_CONV_CBLAS] =                                         \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_cblas;                      \
    fb_install_conv_thunks(&vtable, OP_ID);                                        \
    thunk = (fb_##SUFFIX##_fortran_fn)vtable.ext_ops[OP_ID][FB_CONV_FORTRAN];      \
    if (!thunk) {                                                                  \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk was not installed\n"); \
        return 1;                                                                  \
    }                                                                              \
    thunk(&jobz, &uplo, &n, &kd, ab, &ldab, w, z, &ldz, work, &lwork, rwork, &lrwork, iwork, &liwork, &info); \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                     \
        g_##SUFFIX##_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||                   \
        g_##SUFFIX##_cblas_call.jobz != 'N' ||                                     \
        g_##SUFFIX##_cblas_call.uplo != FB_LOWER ||                                \
        g_##SUFFIX##_cblas_call.n != 2 || g_##SUFFIX##_cblas_call.kd != 1 ||       \
        g_##SUFFIX##_cblas_call.ab != ab || g_##SUFFIX##_cblas_call.ldab != 2 ||   \
        g_##SUFFIX##_cblas_call.w != w || g_##SUFFIX##_cblas_call.z != z ||        \
        g_##SUFFIX##_cblas_call.ldz != 1 || info != (BASE) + 6 ||                  \
        w[0] != (REAL_TYPE)((BASE) + 4) || REAL_PART(z[0]) != (REAL_TYPE)((BASE) + 5)) { \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not forward HBEVD arguments correctly\n"); \
        return 1;                                                                  \
    }                                                                              \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk forwards HBEVD arguments\n"); \
    return 0;                                                                      \
}

DEFINE_HBEVD_TESTS(chbevd, fb_complex_float_t, float, FB_OP_CHBEVD, 100, make_cfloat, cfloat_real)
DEFINE_HBEVD_TESTS(zhbevd, fb_complex_double_t, double, FB_OP_ZHBEVD, 300, make_cdouble, cdouble_real)

int main(void)
{
    int status = 0;

    status |= check_chbevd_fortran_to_cblas();
    status |= check_chbevd_cblas_to_fortran();
    status |= check_zhbevd_fortran_to_cblas();
    status |= check_zhbevd_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}
