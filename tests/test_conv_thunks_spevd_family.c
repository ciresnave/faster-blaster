#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

#define DEFINE_SPEVD_TESTS(SUFFIX, TYPE, OP_ID, BASE)                            \
typedef int (*fb_##SUFFIX##_cblas_fn)(fb_layout_t layout, char jobz,          \
                                      fb_uplo_t uplo, int n, TYPE *ap,        \
                                      TYPE *w, TYPE *z, int ldz);             \
typedef void (*fb_##SUFFIX##_fortran_fn)(char *jobz, char *uplo, int *n,      \
                                         TYPE *ap, TYPE *w, TYPE *z, int *ldz,\
                                         TYPE *work, int *lwork, int *iwork,  \
                                         int *liwork, int *info);             \
static struct {                                                                   \
    int query_calls;                                                              \
    int calls;                                                                    \
    char jobz;                                                                    \
    char uplo;                                                                    \
    int n;                                                                        \
    TYPE ap_snapshot[6];                                                          \
    TYPE *w;                                                                      \
    TYPE *z;                                                                      \
    int ldz;                                                                      \
    int work_nonnull;                                                             \
    int lwork;                                                                    \
    int iwork_nonnull;                                                            \
    int liwork;                                                                   \
} g_##SUFFIX##_fortran_call;                                                      \
static struct {                                                                   \
    int called;                                                                   \
    fb_layout_t layout;                                                           \
    char jobz;                                                                    \
    fb_uplo_t uplo;                                                               \
    int n;                                                                        \
    TYPE *ap;                                                                     \
    TYPE *w;                                                                      \
    TYPE *z;                                                                      \
    int ldz;                                                                      \
} g_##SUFFIX##_cblas_call;                                                        \
static void stub_##SUFFIX##_fortran(char *jobz, char *uplo, int *n, TYPE *ap,  \
                                    TYPE *w, TYPE *z, int *ldz, TYPE *work,    \
                                    int *lwork, int *iwork, int *liwork,       \
                                    int *info)                                 \
{                                                                                 \
    if (*lwork == -1 || *liwork == -1) {                                         \
        g_##SUFFIX##_fortran_call.query_calls += 1;                              \
        *work = (TYPE)34;                                                        \
        *iwork = 18;                                                             \
        *info = 0;                                                               \
        return;                                                                  \
    }                                                                            \
    memset(g_##SUFFIX##_fortran_call.ap_snapshot, 0, sizeof(g_##SUFFIX##_fortran_call.ap_snapshot)); \
    memcpy(g_##SUFFIX##_fortran_call.ap_snapshot, ap, (size_t)(*n * (*n + 1) / 2) * sizeof(TYPE)); \
    g_##SUFFIX##_fortran_call.calls += 1;                                        \
    g_##SUFFIX##_fortran_call.jobz = *jobz;                                      \
    g_##SUFFIX##_fortran_call.uplo = *uplo;                                      \
    g_##SUFFIX##_fortran_call.n = *n;                                            \
    g_##SUFFIX##_fortran_call.w = w;                                             \
    g_##SUFFIX##_fortran_call.z = z;                                             \
    g_##SUFFIX##_fortran_call.ldz = *ldz;                                        \
    g_##SUFFIX##_fortran_call.work_nonnull = (work != NULL);                     \
    g_##SUFFIX##_fortran_call.lwork = *lwork;                                    \
    g_##SUFFIX##_fortran_call.iwork_nonnull = (iwork != NULL);                   \
    g_##SUFFIX##_fortran_call.liwork = *liwork;                                  \
    ap[0] = (TYPE)41; ap[1] = (TYPE)42; ap[2] = (TYPE)52;                       \
    ap[3] = (TYPE)43; ap[4] = (TYPE)53; ap[5] = (TYPE)63;                       \
    w[0] = (TYPE)((BASE) + 1); w[1] = (TYPE)((BASE) + 2); w[2] = (TYPE)((BASE) + 3); \
    z[0] = (TYPE)101; z[1] = (TYPE)201; z[2] = (TYPE)301;                       \
    z[3] = (TYPE)102; z[4] = (TYPE)202; z[5] = (TYPE)302;                       \
    z[6] = (TYPE)103; z[7] = (TYPE)203; z[8] = (TYPE)303;                       \
    *info = 0;                                                                   \
}                                                                                 \
static int stub_##SUFFIX##_cblas(fb_layout_t layout, char jobz, fb_uplo_t uplo, \
                                 int n, TYPE *ap, TYPE *w, TYPE *z, int ldz)   \
{                                                                                 \
    g_##SUFFIX##_cblas_call.called += 1;                                          \
    g_##SUFFIX##_cblas_call.layout = layout;                                      \
    g_##SUFFIX##_cblas_call.jobz = jobz;                                          \
    g_##SUFFIX##_cblas_call.uplo = uplo;                                          \
    g_##SUFFIX##_cblas_call.n = n;                                                \
    g_##SUFFIX##_cblas_call.ap = ap;                                              \
    g_##SUFFIX##_cblas_call.w = w;                                                \
    g_##SUFFIX##_cblas_call.z = z;                                                \
    g_##SUFFIX##_cblas_call.ldz = ldz;                                            \
    w[0] = (TYPE)((BASE) + 4);                                                    \
    z[0] = (TYPE)((BASE) + 5);                                                    \
    return (BASE) + 6;                                                            \
}                                                                                 \
static int check_##SUFFIX##_fortran_to_cblas(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                          \
    TYPE ap_row[6] = { (TYPE)11, (TYPE)12, (TYPE)13, (TYPE)22, (TYPE)23, (TYPE)33 }; \
    TYPE expected_ap_in[6] = { (TYPE)11, (TYPE)12, (TYPE)22, (TYPE)13, (TYPE)23, (TYPE)33 }; \
    TYPE expected_ap_out[6] = { (TYPE)41, (TYPE)42, (TYPE)43, (TYPE)52, (TYPE)53, (TYPE)63 }; \
    TYPE w[3] = { 0, 0, 0 };                                                      \
    TYPE z[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };                                   \
    TYPE expected_z[9] = { (TYPE)101, (TYPE)102, (TYPE)103, (TYPE)201, (TYPE)202, (TYPE)203, (TYPE)301, (TYPE)302, (TYPE)303 }; \
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
    info = thunk(FB_LAYOUT_ROW_MAJOR, 'V', FB_UPPER, 3, ap_row, w, z, 3);        \
    if (info != 0 || g_##SUFFIX##_fortran_call.query_calls != 1 ||               \
        g_##SUFFIX##_fortran_call.calls != 1 ||                                  \
        g_##SUFFIX##_fortran_call.jobz != 'V' ||                                 \
        g_##SUFFIX##_fortran_call.uplo != 'U' ||                                 \
        g_##SUFFIX##_fortran_call.n != 3 ||                                      \
        memcmp(g_##SUFFIX##_fortran_call.ap_snapshot, expected_ap_in, sizeof(expected_ap_in)) != 0 || \
        g_##SUFFIX##_fortran_call.w != w ||                                      \
        g_##SUFFIX##_fortran_call.ldz != 3 ||                                    \
        !g_##SUFFIX##_fortran_call.work_nonnull ||                               \
        g_##SUFFIX##_fortran_call.lwork != 34 ||                                 \
        !g_##SUFFIX##_fortran_call.iwork_nonnull ||                              \
        g_##SUFFIX##_fortran_call.liwork != 18 ||                                \
        memcmp(ap_row, expected_ap_out, sizeof(expected_ap_out)) != 0 ||         \
        w[0] != (TYPE)((BASE) + 1) || w[1] != (TYPE)((BASE) + 2) ||             \
        w[2] != (TYPE)((BASE) + 3) || memcmp(z, expected_z, sizeof(expected_z)) != 0) { \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not translate packed SPEVD state correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk queries work, translates packed storage, and copies back eigenvectors\n"); \
    return 0;                                                                     \
}                                                                                 \
static int check_##SUFFIX##_cblas_to_fortran(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                        \
    char jobz = 'N';                                                              \
    char uplo = 'L';                                                              \
    int n = 2;                                                                    \
    TYPE ap[3] = { (TYPE)7, (TYPE)8, (TYPE)9 };                                  \
    TYPE w[2] = { 0, 0 };                                                         \
    TYPE z[1] = { 0 };                                                            \
    int ldz = 1;                                                                  \
    TYPE work[34] = { 0 };                                                        \
    int lwork = 34;                                                               \
    int iwork[18] = { 0 };                                                        \
    int liwork = 18;                                                              \
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
    thunk(&jobz, &uplo, &n, ap, w, z, &ldz, work, &lwork, iwork, &liwork, &info); \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                   \
        g_##SUFFIX##_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||                 \
        g_##SUFFIX##_cblas_call.jobz != 'N' ||                                   \
        g_##SUFFIX##_cblas_call.uplo != FB_LOWER ||                              \
        g_##SUFFIX##_cblas_call.n != 2 ||                                        \
        g_##SUFFIX##_cblas_call.ap != ap ||                                      \
        g_##SUFFIX##_cblas_call.w != w ||                                        \
        g_##SUFFIX##_cblas_call.z != z ||                                        \
        g_##SUFFIX##_cblas_call.ldz != 1 || info != (BASE) + 6 ||               \
        w[0] != (TYPE)((BASE) + 4) || z[0] != (TYPE)((BASE) + 5)) {             \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not forward SPEVD arguments correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk forwards SPEVD arguments\n"); \
    return 0;                                                                     \
}

DEFINE_SPEVD_TESTS(sspevd, float, FB_OP_SSPEVD, 100)
DEFINE_SPEVD_TESTS(dspevd, double, FB_OP_DSPEVD, 300)

int main(void)
{
    int status = 0;

    status |= check_sspevd_fortran_to_cblas();
    status |= check_sspevd_cblas_to_fortran();
    status |= check_dspevd_fortran_to_cblas();
    status |= check_dspevd_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}