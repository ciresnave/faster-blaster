#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

#define DEFINE_PPRFS_REAL_TESTS(SUFFIX, TYPE, OP_ID, BASE)                       \
typedef int (*fb_##SUFFIX##_cblas_fn)(fb_layout_t layout, fb_uplo_t uplo,      \
                                      int n, int nrhs, const TYPE *ap,         \
                                      const TYPE *afp, const TYPE *b, int ldb, \
                                      TYPE *x, int ldx, TYPE *ferr, TYPE *berr);\
typedef void (*fb_##SUFFIX##_fortran_fn)(char *uplo, int *n, int *nrhs, TYPE *ap,\
                                         TYPE *afp, TYPE *b, int *ldb, TYPE *x,\
                                         int *ldx, TYPE *ferr, TYPE *berr, TYPE *work,\
                                         int *iwork, int *info);               \
static struct {                                                                    \
    int called;                                                                    \
    char uplo;                                                                      \
    int n;                                                                          \
    int nrhs;                                                                       \
    TYPE ap_snapshot[6];                                                            \
    TYPE afp_snapshot[6];                                                           \
    TYPE b_snapshot[4];                                                             \
    TYPE x_snapshot[4];                                                             \
    int ldb;                                                                        \
    int ldx;                                                                        \
    TYPE *ferr;                                                                     \
    TYPE *berr;                                                                     \
    int work_nonnull;                                                               \
    int iwork_nonnull;                                                              \
} g_##SUFFIX##_fortran_call;                                                       \
static struct {                                                                    \
    int called;                                                                    \
    fb_layout_t layout;                                                            \
    fb_uplo_t uplo;                                                                \
    int n;                                                                          \
    int nrhs;                                                                       \
    const TYPE *ap;                                                                 \
    const TYPE *afp;                                                                \
    const TYPE *b;                                                                  \
    int ldb;                                                                        \
    TYPE *x;                                                                        \
    int ldx;                                                                        \
    TYPE *ferr;                                                                     \
    TYPE *berr;                                                                     \
} g_##SUFFIX##_cblas_call;                                                         \
static void stub_##SUFFIX##_fortran(char *uplo, int *n, int *nrhs, TYPE *ap,    \
                                    TYPE *afp, TYPE *b, int *ldb, TYPE *x,      \
                                    int *ldx, TYPE *ferr, TYPE *berr,           \
                                    TYPE *work, int *iwork, int *info)          \
{                                                                                 \
    size_t packed_len = (size_t)(*n * (*n + 1)) / 2U;                            \
    size_t mat_elems = (size_t)(*n) * (size_t)(*nrhs);                           \
    g_##SUFFIX##_fortran_call.called += 1;                                        \
    g_##SUFFIX##_fortran_call.uplo = *uplo;                                       \
    g_##SUFFIX##_fortran_call.n = *n;                                             \
    g_##SUFFIX##_fortran_call.nrhs = *nrhs;                                       \
    memset(g_##SUFFIX##_fortran_call.ap_snapshot, 0, sizeof(g_##SUFFIX##_fortran_call.ap_snapshot)); \
    memset(g_##SUFFIX##_fortran_call.afp_snapshot, 0, sizeof(g_##SUFFIX##_fortran_call.afp_snapshot)); \
    memset(g_##SUFFIX##_fortran_call.b_snapshot, 0, sizeof(g_##SUFFIX##_fortran_call.b_snapshot)); \
    memset(g_##SUFFIX##_fortran_call.x_snapshot, 0, sizeof(g_##SUFFIX##_fortran_call.x_snapshot)); \
    memcpy(g_##SUFFIX##_fortran_call.ap_snapshot, ap, packed_len * sizeof(TYPE)); \
    memcpy(g_##SUFFIX##_fortran_call.afp_snapshot, afp, packed_len * sizeof(TYPE)); \
    memcpy(g_##SUFFIX##_fortran_call.b_snapshot, b, mat_elems * sizeof(TYPE));    \
    memcpy(g_##SUFFIX##_fortran_call.x_snapshot, x, mat_elems * sizeof(TYPE));    \
    g_##SUFFIX##_fortran_call.ldb = *ldb;                                         \
    g_##SUFFIX##_fortran_call.ldx = *ldx;                                         \
    g_##SUFFIX##_fortran_call.ferr = ferr;                                        \
    g_##SUFFIX##_fortran_call.berr = berr;                                        \
    g_##SUFFIX##_fortran_call.work_nonnull = (work != NULL);                      \
    g_##SUFFIX##_fortran_call.iwork_nonnull = (iwork != NULL);                    \
    x[0] = (TYPE)((BASE) + 1);                                                    \
    x[1] = (TYPE)((BASE) + 2);                                                    \
    x[2] = (TYPE)((BASE) + 3);                                                    \
    x[3] = (TYPE)((BASE) + 4);                                                    \
    ferr[0] = (TYPE)((BASE) + 5);                                                 \
    berr[0] = (TYPE)((BASE) + 6);                                                 \
    *info = (BASE) + 7;                                                           \
}                                                                                 \
static int stub_##SUFFIX##_cblas(fb_layout_t layout, fb_uplo_t uplo, int n,     \
                                 int nrhs, const TYPE *ap, const TYPE *afp,     \
                                 const TYPE *b, int ldb, TYPE *x, int ldx,      \
                                 TYPE *ferr, TYPE *berr)                        \
{                                                                                 \
    g_##SUFFIX##_cblas_call.called += 1;                                          \
    g_##SUFFIX##_cblas_call.layout = layout;                                      \
    g_##SUFFIX##_cblas_call.uplo = uplo;                                          \
    g_##SUFFIX##_cblas_call.n = n;                                                \
    g_##SUFFIX##_cblas_call.nrhs = nrhs;                                          \
    g_##SUFFIX##_cblas_call.ap = ap;                                              \
    g_##SUFFIX##_cblas_call.afp = afp;                                            \
    g_##SUFFIX##_cblas_call.b = b;                                                \
    g_##SUFFIX##_cblas_call.ldb = ldb;                                            \
    g_##SUFFIX##_cblas_call.x = x;                                                \
    g_##SUFFIX##_cblas_call.ldx = ldx;                                            \
    g_##SUFFIX##_cblas_call.ferr = ferr;                                          \
    g_##SUFFIX##_cblas_call.berr = berr;                                          \
    x[0] = (TYPE)((BASE) + 8);                                                    \
    x[1] = (TYPE)((BASE) + 9);                                                    \
    ferr[0] = (TYPE)((BASE) + 10);                                                \
    berr[0] = (TYPE)((BASE) + 11);                                                \
    return (BASE) + 12;                                                           \
}                                                                                 \
static int check_##SUFFIX##_fortran_to_cblas(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                          \
    TYPE ap_row[6] = { (TYPE)11, (TYPE)12, (TYPE)13, (TYPE)22, (TYPE)23, (TYPE)33 }; \
    TYPE afp_row[6] = { (TYPE)41, (TYPE)42, (TYPE)43, (TYPE)52, (TYPE)53, (TYPE)63 }; \
    TYPE expected_ap[6] = { (TYPE)11, (TYPE)12, (TYPE)13, 0, 0, 0 };             \
    TYPE expected_afp[6] = { (TYPE)41, (TYPE)42, (TYPE)43, 0, 0, 0 };            \
    TYPE b_row[4] = { (TYPE)1, (TYPE)2, (TYPE)3, (TYPE)4 };                      \
    TYPE x_row[4] = { (TYPE)5, (TYPE)6, (TYPE)7, (TYPE)8 };                      \
    TYPE expected_b[4] = { (TYPE)1, (TYPE)3, (TYPE)2, (TYPE)4 };                 \
    TYPE expected_x[4] = { (TYPE)5, (TYPE)7, (TYPE)6, (TYPE)8 };                 \
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
    info = thunk(FB_LAYOUT_ROW_MAJOR, FB_UPPER, 2, 2, ap_row, afp_row, b_row, 2, x_row, 2, ferr, berr); \
    if (info != (BASE) + 7 ||                                                     \
        g_##SUFFIX##_fortran_call.called != 1 ||                                  \
        g_##SUFFIX##_fortran_call.uplo != 'U' ||                                  \
        g_##SUFFIX##_fortran_call.n != 2 ||                                       \
        g_##SUFFIX##_fortran_call.nrhs != 2 ||                                    \
        memcmp(g_##SUFFIX##_fortran_call.ap_snapshot, expected_ap, sizeof(expected_ap)) != 0 || \
        memcmp(g_##SUFFIX##_fortran_call.afp_snapshot, expected_afp, sizeof(expected_afp)) != 0 || \
        memcmp(g_##SUFFIX##_fortran_call.b_snapshot, expected_b, sizeof(expected_b)) != 0 || \
        memcmp(g_##SUFFIX##_fortran_call.x_snapshot, expected_x, sizeof(expected_x)) != 0 || \
        g_##SUFFIX##_fortran_call.ldb != 2 ||                                     \
        g_##SUFFIX##_fortran_call.ldx != 2 ||                                     \
        g_##SUFFIX##_fortran_call.ferr != ferr ||                                 \
        g_##SUFFIX##_fortran_call.berr != berr ||                                 \
        !g_##SUFFIX##_fortran_call.work_nonnull ||                                \
        !g_##SUFFIX##_fortran_call.iwork_nonnull ||                               \
        x_row[0] != (TYPE)((BASE) + 1) || x_row[1] != (TYPE)((BASE) + 3) ||      \
        x_row[2] != (TYPE)((BASE) + 2) || x_row[3] != (TYPE)((BASE) + 4) ||      \
        ferr[0] != (TYPE)((BASE) + 5) || berr[0] != (TYPE)((BASE) + 6)) {        \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not route packed PPRFS inputs correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk routes packed PPRFS inputs\n"); \
    return 0;                                                                     \
}                                                                                 \
static int check_##SUFFIX##_cblas_to_fortran(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                        \
    char uplo = 'L';                                                              \
    int n = 1;                                                                    \
    int nrhs = 1;                                                                 \
    TYPE ap[1] = { (TYPE)9 };                                                     \
    TYPE afp[1] = { (TYPE)10 };                                                   \
    TYPE b[1] = { (TYPE)11 };                                                     \
    TYPE x[1] = { (TYPE)12 };                                                     \
    int ldb = 1;                                                                  \
    int ldx = 1;                                                                  \
    TYPE ferr[1] = { 0 };                                                         \
    TYPE berr[1] = { 0 };                                                         \
    TYPE work[2] = { 0 };                                                         \
    int iwork[1] = { 0 };                                                         \
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
    thunk(&uplo, &n, &nrhs, ap, afp, b, &ldb, x, &ldx, ferr, berr, work, iwork, &info); \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                    \
        g_##SUFFIX##_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||                  \
        g_##SUFFIX##_cblas_call.uplo != FB_LOWER ||                               \
        g_##SUFFIX##_cblas_call.n != 1 ||                                         \
        g_##SUFFIX##_cblas_call.nrhs != 1 ||                                      \
        g_##SUFFIX##_cblas_call.ap != ap ||                                       \
        g_##SUFFIX##_cblas_call.afp != afp ||                                     \
        g_##SUFFIX##_cblas_call.b != b ||                                         \
        g_##SUFFIX##_cblas_call.ldb != 1 ||                                       \
        g_##SUFFIX##_cblas_call.x != x ||                                         \
        g_##SUFFIX##_cblas_call.ldx != 1 ||                                       \
        g_##SUFFIX##_cblas_call.ferr != ferr ||                                   \
        g_##SUFFIX##_cblas_call.berr != berr ||                                   \
        info != (BASE) + 12 || x[0] != (TYPE)((BASE) + 8) ||                     \
        ferr[0] != (TYPE)((BASE) + 10) || berr[0] != (TYPE)((BASE) + 11)) {      \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not propagate PPRFS info correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk propagates PPRFS info\n"); \
    return 0;                                                                     \
}

#define DEFINE_PPRFS_COMPLEX_TESTS(SUFFIX, CTYPE, RTYPE, OP_ID, BASE)           \
typedef int (*fb_##SUFFIX##_cblas_fn)(fb_layout_t layout, fb_uplo_t uplo,      \
                                      int n, int nrhs, const CTYPE *ap,        \
                                      const CTYPE *afp, const CTYPE *b,        \
                                      int ldb, CTYPE *x, int ldx,              \
                                      RTYPE *ferr, RTYPE *berr);               \
typedef void (*fb_##SUFFIX##_fortran_fn)(char *uplo, int *n, int *nrhs,        \
                                         CTYPE *ap, CTYPE *afp, CTYPE *b,      \
                                         int *ldb, CTYPE *x, int *ldx,         \
                                         RTYPE *ferr, RTYPE *berr, CTYPE *work,\
                                         RTYPE *rwork, int *info);             \
static struct {                                                                    \
    int called;                                                                    \
    char uplo;                                                                      \
    int n;                                                                          \
    int nrhs;                                                                       \
    CTYPE *ap;                                                                      \
    CTYPE *afp;                                                                     \
    CTYPE *b;                                                                       \
    CTYPE *x;                                                                       \
    int ldb;                                                                        \
    int ldx;                                                                        \
    RTYPE *ferr;                                                                    \
    RTYPE *berr;                                                                    \
    int work_nonnull;                                                               \
    int rwork_nonnull;                                                              \
} g_##SUFFIX##_fortran_call;                                                       \
static struct {                                                                    \
    int called;                                                                    \
    fb_layout_t layout;                                                            \
    fb_uplo_t uplo;                                                                \
    int n;                                                                          \
    int nrhs;                                                                       \
    const CTYPE *ap;                                                                \
    const CTYPE *afp;                                                               \
    const CTYPE *b;                                                                 \
    int ldb;                                                                        \
    CTYPE *x;                                                                       \
    int ldx;                                                                        \
    RTYPE *ferr;                                                                    \
    RTYPE *berr;                                                                    \
} g_##SUFFIX##_cblas_call;                                                         \
static void stub_##SUFFIX##_fortran(char *uplo, int *n, int *nrhs, CTYPE *ap,   \
                                    CTYPE *afp, CTYPE *b, int *ldb, CTYPE *x,  \
                                    int *ldx, RTYPE *ferr, RTYPE *berr,        \
                                    CTYPE *work, RTYPE *rwork, int *info)      \
{                                                                                 \
    g_##SUFFIX##_fortran_call.called += 1;                                        \
    g_##SUFFIX##_fortran_call.uplo = *uplo;                                       \
    g_##SUFFIX##_fortran_call.n = *n;                                             \
    g_##SUFFIX##_fortran_call.nrhs = *nrhs;                                       \
    g_##SUFFIX##_fortran_call.ap = ap;                                            \
    g_##SUFFIX##_fortran_call.afp = afp;                                          \
    g_##SUFFIX##_fortran_call.b = b;                                              \
    g_##SUFFIX##_fortran_call.x = x;                                              \
    g_##SUFFIX##_fortran_call.ldb = *ldb;                                         \
    g_##SUFFIX##_fortran_call.ldx = *ldx;                                         \
    g_##SUFFIX##_fortran_call.ferr = ferr;                                        \
    g_##SUFFIX##_fortran_call.berr = berr;                                        \
    g_##SUFFIX##_fortran_call.work_nonnull = (work != NULL);                      \
    g_##SUFFIX##_fortran_call.rwork_nonnull = (rwork != NULL);                    \
    x[0] = (CTYPE)0;                                                              \
    x[1] = (CTYPE)0;                                                              \
    ferr[0] = (RTYPE)((BASE) + 1);                                                \
    berr[0] = (RTYPE)((BASE) + 2);                                                \
    *info = (BASE) + 3;                                                           \
}                                                                                 \
static int stub_##SUFFIX##_cblas(fb_layout_t layout, fb_uplo_t uplo, int n,     \
                                 int nrhs, const CTYPE *ap, const CTYPE *afp,   \
                                 const CTYPE *b, int ldb, CTYPE *x, int ldx,    \
                                 RTYPE *ferr, RTYPE *berr)                      \
{                                                                                 \
    g_##SUFFIX##_cblas_call.called += 1;                                          \
    g_##SUFFIX##_cblas_call.layout = layout;                                      \
    g_##SUFFIX##_cblas_call.uplo = uplo;                                          \
    g_##SUFFIX##_cblas_call.n = n;                                                \
    g_##SUFFIX##_cblas_call.nrhs = nrhs;                                          \
    g_##SUFFIX##_cblas_call.ap = ap;                                              \
    g_##SUFFIX##_cblas_call.afp = afp;                                            \
    g_##SUFFIX##_cblas_call.b = b;                                                \
    g_##SUFFIX##_cblas_call.ldb = ldb;                                            \
    g_##SUFFIX##_cblas_call.x = x;                                                \
    g_##SUFFIX##_cblas_call.ldx = ldx;                                            \
    g_##SUFFIX##_cblas_call.ferr = ferr;                                          \
    g_##SUFFIX##_cblas_call.berr = berr;                                          \
    ferr[0] = (RTYPE)((BASE) + 4);                                                \
    berr[0] = (RTYPE)((BASE) + 5);                                                \
    return (BASE) + 6;                                                            \
}                                                                                 \
static int check_##SUFFIX##_fortran_to_cblas(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                          \
    CTYPE ap[3] = { 0 };                                                          \
    CTYPE afp[3] = { 0 };                                                         \
    CTYPE b[2] = { 0 };                                                           \
    CTYPE x[2] = { 0 };                                                           \
    RTYPE ferr[1] = { 0 };                                                        \
    RTYPE berr[1] = { 0 };                                                        \
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
    info = thunk(FB_LAYOUT_COL_MAJOR, FB_UPPER, 1, 1, ap, afp, b, 1, x, 1, ferr, berr);\
    if (info != (BASE) + 3 ||                                                     \
        g_##SUFFIX##_fortran_call.called != 1 ||                                  \
        g_##SUFFIX##_fortran_call.uplo != 'U' ||                                  \
        g_##SUFFIX##_fortran_call.n != 1 ||                                       \
        g_##SUFFIX##_fortran_call.nrhs != 1 ||                                    \
        g_##SUFFIX##_fortran_call.ap != ap ||                                     \
        g_##SUFFIX##_fortran_call.afp != afp ||                                   \
        g_##SUFFIX##_fortran_call.b != b ||                                       \
        g_##SUFFIX##_fortran_call.x != x ||                                       \
        g_##SUFFIX##_fortran_call.ldb != 1 ||                                     \
        g_##SUFFIX##_fortran_call.ldx != 1 ||                                     \
        g_##SUFFIX##_fortran_call.ferr != ferr ||                                 \
        g_##SUFFIX##_fortran_call.berr != berr ||                                 \
        !g_##SUFFIX##_fortran_call.work_nonnull ||                                \
        !g_##SUFFIX##_fortran_call.rwork_nonnull ||                               \
        ferr[0] != (RTYPE)((BASE) + 1) || berr[0] != (RTYPE)((BASE) + 2)) {      \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not route complex PPRFS inputs correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk routes complex PPRFS inputs\n"); \
    return 0;                                                                     \
}                                                                                 \
static int check_##SUFFIX##_cblas_to_fortran(void)                                \
{                                                                                 \
    fb_backend_vtable_t vtable;                                                   \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                        \
    char uplo = 'L';                                                              \
    int n = 1;                                                                    \
    int nrhs = 1;                                                                 \
    CTYPE ap[1] = { 0 };                                                          \
    CTYPE afp[1] = { 0 };                                                         \
    CTYPE b[1] = { 0 };                                                           \
    CTYPE x[1] = { 0 };                                                           \
    int ldb = 1;                                                                  \
    int ldx = 1;                                                                  \
    RTYPE ferr[1] = { 0 };                                                        \
    RTYPE berr[1] = { 0 };                                                        \
    CTYPE work[1] = { 0 };                                                        \
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
    thunk(&uplo, &n, &nrhs, ap, afp, b, &ldb, x, &ldx, ferr, berr, work, rwork, &info);\
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                    \
        g_##SUFFIX##_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||                  \
        g_##SUFFIX##_cblas_call.uplo != FB_LOWER ||                               \
        g_##SUFFIX##_cblas_call.n != 1 ||                                         \
        g_##SUFFIX##_cblas_call.nrhs != 1 ||                                      \
        g_##SUFFIX##_cblas_call.ap != ap ||                                       \
        g_##SUFFIX##_cblas_call.afp != afp ||                                     \
        g_##SUFFIX##_cblas_call.b != b ||                                         \
        g_##SUFFIX##_cblas_call.ldb != 1 ||                                       \
        g_##SUFFIX##_cblas_call.x != x ||                                         \
        g_##SUFFIX##_cblas_call.ldx != 1 ||                                       \
        g_##SUFFIX##_cblas_call.ferr != ferr ||                                   \
        g_##SUFFIX##_cblas_call.berr != berr ||                                   \
        info != (BASE) + 6 || ferr[0] != (RTYPE)((BASE) + 4) ||                  \
        berr[0] != (RTYPE)((BASE) + 5)) {                                         \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not propagate complex PPRFS info correctly\n"); \
        return 1;                                                                 \
    }                                                                             \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk propagates complex PPRFS info\n"); \
    return 0;                                                                     \
}

DEFINE_PPRFS_REAL_TESTS(spprfs, float, FB_OP_SPPRFS, 100)
DEFINE_PPRFS_REAL_TESTS(dpprfs, double, FB_OP_DPPRFS, 300)
DEFINE_PPRFS_COMPLEX_TESTS(cpprfs, fb_complex_float_t, float, FB_OP_CPPRFS, 500)
DEFINE_PPRFS_COMPLEX_TESTS(zpprfs, fb_complex_double_t, double, FB_OP_ZPPRFS, 700)

int main(void)
{
    int status = 0;

    status |= check_spprfs_fortran_to_cblas();
    status |= check_spprfs_cblas_to_fortran();
    status |= check_dpprfs_fortran_to_cblas();
    status |= check_dpprfs_cblas_to_fortran();
    status |= check_cpprfs_fortran_to_cblas();
    status |= check_cpprfs_cblas_to_fortran();
    status |= check_zpprfs_fortran_to_cblas();
    status |= check_zpprfs_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}