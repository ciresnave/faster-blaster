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

#define DEFINE_REAL_TZRZF_TESTS(SUFFIX, TYPE, OP_ID, BASE, QUERY_LWORK)        \
typedef int (*fb_##SUFFIX##_cblas_fn)(fb_layout_t layout, int m, int n,        \
                                      TYPE *a, int lda, TYPE *tau);            \
typedef void (*fb_##SUFFIX##_fortran_fn)(int *m, int *n, TYPE *a, int *lda,    \
                                         TYPE *tau, TYPE *work, int *lwork,    \
                                         int *info);                            \
static struct {                                                                 \
    int query_calls;                                                            \
    int exec_calls;                                                             \
    int m;                                                                      \
    int n;                                                                      \
    int lda;                                                                    \
    int query_lwork;                                                            \
    int exec_lwork;                                                             \
    TYPE a_snapshot[8];                                                         \
    TYPE tau_snapshot[2];                                                       \
} g_##SUFFIX##_fortran_call;                                                    \
static struct {                                                                 \
    int called;                                                                 \
    fb_layout_t layout;                                                         \
    int m;                                                                      \
    int n;                                                                      \
    int lda;                                                                    \
    TYPE *a;                                                                    \
    TYPE *tau;                                                                  \
} g_##SUFFIX##_cblas_call;                                                      \
static void stub_##SUFFIX##_fortran(int *m, int *n, TYPE *a, int *lda,         \
                                    TYPE *tau, TYPE *work, int *lwork,         \
                                    int *info)                                  \
{                                                                               \
    int row = 0;                                                                \
    int col = 0;                                                                \
    if (*lwork == -1) {                                                         \
        g_##SUFFIX##_fortran_call.query_calls += 1;                             \
        g_##SUFFIX##_fortran_call.m = *m;                                       \
        g_##SUFFIX##_fortran_call.n = *n;                                       \
        g_##SUFFIX##_fortran_call.lda = *lda;                                   \
        g_##SUFFIX##_fortran_call.query_lwork = *lwork;                         \
        a[0] = (TYPE)-999;                                                      \
        tau[0] = (TYPE)-777;                                                    \
        work[0] = (TYPE)(QUERY_LWORK);                                          \
        *info = 0;                                                              \
        return;                                                                 \
    }                                                                           \
    g_##SUFFIX##_fortran_call.exec_calls += 1;                                  \
    g_##SUFFIX##_fortran_call.m = *m;                                           \
    g_##SUFFIX##_fortran_call.n = *n;                                           \
    g_##SUFFIX##_fortran_call.lda = *lda;                                       \
    g_##SUFFIX##_fortran_call.exec_lwork = *lwork;                              \
    for (col = 0; col < *n; ++col) {                                            \
        for (row = 0; row < *m; ++row) {                                        \
            g_##SUFFIX##_fortran_call.a_snapshot[(col * (*m)) + row] =          \
                a[(col * (*lda)) + row];                                        \
            a[(col * (*lda)) + row] =                                           \
                (TYPE)((BASE) + (10 * row) + col);                              \
        }                                                                       \
    }                                                                           \
    for (row = 0; row < *m; ++row) {                                            \
        g_##SUFFIX##_fortran_call.tau_snapshot[row] = tau[row];                 \
        tau[row] = (TYPE)((BASE) + 200 + row);                                  \
    }                                                                           \
    *info = 0;                                                                  \
}                                                                               \
static int stub_##SUFFIX##_cblas(fb_layout_t layout, int m, int n, TYPE *a,    \
                                 int lda, TYPE *tau)                            \
{                                                                               \
    g_##SUFFIX##_cblas_call.called += 1;                                        \
    g_##SUFFIX##_cblas_call.layout = layout;                                    \
    g_##SUFFIX##_cblas_call.m = m;                                              \
    g_##SUFFIX##_cblas_call.n = n;                                              \
    g_##SUFFIX##_cblas_call.lda = lda;                                          \
    g_##SUFFIX##_cblas_call.a = a;                                              \
    g_##SUFFIX##_cblas_call.tau = tau;                                          \
    a[0] = (TYPE)((BASE) + 300);                                                \
    tau[0] = (TYPE)((BASE) + 400);                                              \
    tau[1] = (TYPE)((BASE) + 401);                                              \
    return (BASE) + 500;                                                        \
}                                                                               \
static int check_##SUFFIX##_fortran_to_cblas(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                        \
    TYPE a[8] = { 0 };                                                          \
    TYPE tau[2] = { 0, 0 };                                                     \
    int row = 0;                                                                \
    int col = 0;                                                                \
    int info = 0;                                                               \
    for (row = 0; row < 2; ++row) {                                             \
        for (col = 0; col < 4; ++col) {                                         \
            a[(row * 4) + col] = (TYPE)((row * 4) + col + 1);                  \
        }                                                                       \
    }                                                                           \
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
    info = thunk(FB_LAYOUT_ROW_MAJOR, 2, 4, a, 4, tau);                        \
    if (info != 0 || g_##SUFFIX##_fortran_call.query_calls != 1 ||              \
        g_##SUFFIX##_fortran_call.exec_calls != 1 ||                            \
        g_##SUFFIX##_fortran_call.m != 2 || g_##SUFFIX##_fortran_call.n != 4 || \
        g_##SUFFIX##_fortran_call.lda != 2 ||                                   \
        g_##SUFFIX##_fortran_call.query_lwork != -1 ||                          \
        g_##SUFFIX##_fortran_call.exec_lwork != (QUERY_LWORK)) {                \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not preserve TZRZF workspace-query metadata\n"); \
        return 1;                                                               \
    }                                                                           \
    for (col = 0; col < 4; ++col) {                                             \
        for (row = 0; row < 2; ++row) {                                         \
            TYPE expected_input = (TYPE)((row * 4) + col + 1);                  \
            TYPE expected_output = (TYPE)((BASE) + (10 * row) + col);           \
            if (g_##SUFFIX##_fortran_call.a_snapshot[(col * 2) + row] !=        \
                    expected_input ||                                            \
                a[(row * 4) + col] != expected_output) {                        \
                fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not round-trip row-major TZRZF matrix data correctly\n"); \
                return 1;                                                       \
            }                                                                   \
        }                                                                       \
    }                                                                           \
    for (row = 0; row < 2; ++row) {                                             \
        if (g_##SUFFIX##_fortran_call.tau_snapshot[row] != (TYPE)0 ||           \
            tau[row] != (TYPE)((BASE) + 200 + row)) {                           \
            fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not restore TZRZF TAU output correctly after the query pass\n"); \
            return 1;                                                           \
        }                                                                       \
    }                                                                           \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk performs the workspace query and preserves row-major TZRZF state\n"); \
    return 0;                                                                   \
}                                                                               \
static int check_##SUFFIX##_cblas_to_fortran(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                      \
    int m = 2;                                                                  \
    int n = 4;                                                                  \
    int lda = 2;                                                                \
    int lwork = 3;                                                              \
    int info = 0;                                                               \
    TYPE a[8] = { 0 };                                                          \
    TYPE tau[2] = { 0, 0 };                                                     \
    TYPE work[3] = { 0, 0, 0 };                                                 \
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
    thunk(&m, &n, a, &lda, tau, work, &lwork, &info);                           \
    if (info != ((BASE) + 500) || g_##SUFFIX##_cblas_call.called != 1 ||        \
        g_##SUFFIX##_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||                \
        g_##SUFFIX##_cblas_call.m != 2 || g_##SUFFIX##_cblas_call.n != 4 ||     \
        g_##SUFFIX##_cblas_call.lda != 2 ||                                     \
        g_##SUFFIX##_cblas_call.a != a || g_##SUFFIX##_cblas_call.tau != tau || \
        a[0] != (TYPE)((BASE) + 300) ||                                         \
        tau[0] != (TYPE)((BASE) + 400) ||                                       \
        tau[1] != (TYPE)((BASE) + 401)) {                                       \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not delegate the no-work TZRZF LAPACKE entry correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk maps the all-pointer ABI into the no-work TZRZF LAPACKE entry\n"); \
    return 0;                                                                   \
}

#define DEFINE_COMPLEX_TZRZF_TESTS(SUFFIX, CTYPE, REALTYPE, OP_ID, MAKE_FN, EQ_FN, BASE, QUERY_LWORK) \
typedef int (*fb_##SUFFIX##_cblas_fn)(fb_layout_t layout, int m, int n,        \
                                      CTYPE *a, int lda, CTYPE *tau);          \
typedef void (*fb_##SUFFIX##_fortran_fn)(int *m, int *n, CTYPE *a, int *lda,   \
                                         CTYPE *tau, CTYPE *work, int *lwork,  \
                                         int *info);                            \
static struct {                                                                 \
    int query_calls;                                                            \
    int exec_calls;                                                             \
    int m;                                                                      \
    int n;                                                                      \
    int lda;                                                                    \
    int query_lwork;                                                            \
    int exec_lwork;                                                             \
    CTYPE a_snapshot[8];                                                        \
    CTYPE tau_snapshot[2];                                                      \
} g_##SUFFIX##_fortran_call;                                                    \
static struct {                                                                 \
    int called;                                                                 \
    fb_layout_t layout;                                                         \
    int m;                                                                      \
    int n;                                                                      \
    int lda;                                                                    \
    CTYPE *a;                                                                   \
    CTYPE *tau;                                                                 \
} g_##SUFFIX##_cblas_call;                                                      \
static void stub_##SUFFIX##_fortran(int *m, int *n, CTYPE *a, int *lda,        \
                                    CTYPE *tau, CTYPE *work, int *lwork,       \
                                    int *info)                                  \
{                                                                               \
    int row = 0;                                                                \
    int col = 0;                                                                \
    if (*lwork == -1) {                                                         \
        g_##SUFFIX##_fortran_call.query_calls += 1;                             \
        g_##SUFFIX##_fortran_call.m = *m;                                       \
        g_##SUFFIX##_fortran_call.n = *n;                                       \
        g_##SUFFIX##_fortran_call.lda = *lda;                                   \
        g_##SUFFIX##_fortran_call.query_lwork = *lwork;                         \
        a[0] = MAKE_FN((REALTYPE)-999, (REALTYPE)-1999);                        \
        tau[0] = MAKE_FN((REALTYPE)-777, (REALTYPE)-1777);                      \
        work[0] = MAKE_FN((REALTYPE)(QUERY_LWORK),                              \
                          (REALTYPE)(QUERY_LWORK + 100));                       \
        *info = 0;                                                              \
        return;                                                                 \
    }                                                                           \
    g_##SUFFIX##_fortran_call.exec_calls += 1;                                  \
    g_##SUFFIX##_fortran_call.m = *m;                                           \
    g_##SUFFIX##_fortran_call.n = *n;                                           \
    g_##SUFFIX##_fortran_call.lda = *lda;                                       \
    g_##SUFFIX##_fortran_call.exec_lwork = *lwork;                              \
    for (col = 0; col < *n; ++col) {                                            \
        for (row = 0; row < *m; ++row) {                                        \
            g_##SUFFIX##_fortran_call.a_snapshot[(col * (*m)) + row] =          \
                a[(col * (*lda)) + row];                                        \
            a[(col * (*lda)) + row] =                                           \
                MAKE_FN((REALTYPE)((BASE) + (10 * row) + col),                  \
                        (REALTYPE)((BASE) + 50 + (10 * row) + col));            \
        }                                                                       \
    }                                                                           \
    for (row = 0; row < *m; ++row) {                                            \
        g_##SUFFIX##_fortran_call.tau_snapshot[row] = tau[row];                 \
        tau[row] = MAKE_FN((REALTYPE)((BASE) + 200 + row),                      \
                           (REALTYPE)((BASE) + 250 + row));                     \
    }                                                                           \
    *info = 0;                                                                  \
}                                                                               \
static int stub_##SUFFIX##_cblas(fb_layout_t layout, int m, int n, CTYPE *a,   \
                                 int lda, CTYPE *tau)                           \
{                                                                               \
    g_##SUFFIX##_cblas_call.called += 1;                                        \
    g_##SUFFIX##_cblas_call.layout = layout;                                    \
    g_##SUFFIX##_cblas_call.m = m;                                              \
    g_##SUFFIX##_cblas_call.n = n;                                              \
    g_##SUFFIX##_cblas_call.lda = lda;                                          \
    g_##SUFFIX##_cblas_call.a = a;                                              \
    g_##SUFFIX##_cblas_call.tau = tau;                                          \
    a[0] = MAKE_FN((REALTYPE)((BASE) + 300), (REALTYPE)((BASE) + 350));        \
    tau[0] = MAKE_FN((REALTYPE)((BASE) + 400), (REALTYPE)((BASE) + 450));      \
    tau[1] = MAKE_FN((REALTYPE)((BASE) + 401), (REALTYPE)((BASE) + 451));      \
    return (BASE) + 500;                                                        \
}                                                                               \
static int check_##SUFFIX##_fortran_to_cblas(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                        \
    CTYPE a[8];                                                                 \
    CTYPE tau[2];                                                               \
    int row = 0;                                                                \
    int col = 0;                                                                \
    int info = 0;                                                               \
    for (row = 0; row < 2; ++row) {                                             \
        for (col = 0; col < 4; ++col) {                                         \
            a[(row * 4) + col] =                                                \
                MAKE_FN((REALTYPE)((row * 4) + col + 1),                        \
                        (REALTYPE)((row * 4) + col + 11));                      \
        }                                                                       \
    }                                                                           \
    tau[0] = MAKE_FN((REALTYPE)0, (REALTYPE)0);                                 \
    tau[1] = MAKE_FN((REALTYPE)0, (REALTYPE)0);                                 \
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
    info = thunk(FB_LAYOUT_ROW_MAJOR, 2, 4, a, 4, tau);                        \
    if (info != 0 || g_##SUFFIX##_fortran_call.query_calls != 1 ||              \
        g_##SUFFIX##_fortran_call.exec_calls != 1 ||                            \
        g_##SUFFIX##_fortran_call.m != 2 || g_##SUFFIX##_fortran_call.n != 4 || \
        g_##SUFFIX##_fortran_call.lda != 2 ||                                   \
        g_##SUFFIX##_fortran_call.query_lwork != -1 ||                          \
        g_##SUFFIX##_fortran_call.exec_lwork != (QUERY_LWORK)) {                \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not preserve complex TZRZF workspace-query metadata\n"); \
        return 1;                                                               \
    }                                                                           \
    for (col = 0; col < 4; ++col) {                                             \
        for (row = 0; row < 2; ++row) {                                         \
            CTYPE expected_input =                                              \
                MAKE_FN((REALTYPE)((row * 4) + col + 1),                        \
                        (REALTYPE)((row * 4) + col + 11));                      \
            CTYPE expected_output =                                             \
                MAKE_FN((REALTYPE)((BASE) + (10 * row) + col),                  \
                        (REALTYPE)((BASE) + 50 + (10 * row) + col));            \
            if (!EQ_FN(g_##SUFFIX##_fortran_call.a_snapshot[(col * 2) + row],   \
                       expected_input) ||                                        \
                !EQ_FN(a[(row * 4) + col], expected_output)) {                  \
                fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not round-trip complex row-major TZRZF matrix data correctly\n"); \
                return 1;                                                       \
            }                                                                   \
        }                                                                       \
    }                                                                           \
    for (row = 0; row < 2; ++row) {                                             \
        CTYPE expected_tau =                                                    \
            MAKE_FN((REALTYPE)((BASE) + 200 + row),                             \
                    (REALTYPE)((BASE) + 250 + row));                            \
        if (!EQ_FN(g_##SUFFIX##_fortran_call.tau_snapshot[row],                 \
                   MAKE_FN((REALTYPE)0, (REALTYPE)0)) ||                        \
            !EQ_FN(tau[row], expected_tau)) {                                   \
            fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not restore complex TZRZF TAU output correctly after the query pass\n"); \
            return 1;                                                           \
        }                                                                       \
    }                                                                           \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk performs the workspace query and preserves complex row-major TZRZF state\n"); \
    return 0;                                                                   \
}                                                                               \
static int check_##SUFFIX##_cblas_to_fortran(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                      \
    int m = 2;                                                                  \
    int n = 4;                                                                  \
    int lda = 2;                                                                \
    int lwork = 3;                                                              \
    int info = 0;                                                               \
    CTYPE a[8];                                                                 \
    CTYPE tau[2];                                                               \
    CTYPE work[3];                                                              \
    int index = 0;                                                              \
    for (index = 0; index < 8; ++index) {                                       \
        a[index] = MAKE_FN((REALTYPE)0, (REALTYPE)0);                           \
    }                                                                           \
    tau[0] = MAKE_FN((REALTYPE)0, (REALTYPE)0);                                 \
    tau[1] = MAKE_FN((REALTYPE)0, (REALTYPE)0);                                 \
    work[0] = MAKE_FN((REALTYPE)0, (REALTYPE)0);                                \
    work[1] = MAKE_FN((REALTYPE)0, (REALTYPE)0);                                \
    work[2] = MAKE_FN((REALTYPE)0, (REALTYPE)0);                                \
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
    thunk(&m, &n, a, &lda, tau, work, &lwork, &info);                           \
    if (info != ((BASE) + 500) || g_##SUFFIX##_cblas_call.called != 1 ||        \
        g_##SUFFIX##_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||                \
        g_##SUFFIX##_cblas_call.m != 2 || g_##SUFFIX##_cblas_call.n != 4 ||     \
        g_##SUFFIX##_cblas_call.lda != 2 ||                                     \
        g_##SUFFIX##_cblas_call.a != a || g_##SUFFIX##_cblas_call.tau != tau || \
        !EQ_FN(a[0], MAKE_FN((REALTYPE)((BASE) + 300),                          \
                             (REALTYPE)((BASE) + 350))) ||                      \
        !EQ_FN(tau[0], MAKE_FN((REALTYPE)((BASE) + 400),                        \
                               (REALTYPE)((BASE) + 450))) ||                    \
        !EQ_FN(tau[1], MAKE_FN((REALTYPE)((BASE) + 401),                        \
                               (REALTYPE)((BASE) + 451)))) {                    \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not delegate the complex no-work TZRZF LAPACKE entry correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk maps the all-pointer ABI into the complex no-work TZRZF LAPACKE entry\n"); \
    return 0;                                                                   \
}

DEFINE_REAL_TZRZF_TESTS(stzrzf, float, FB_OP_STZRZF, 100, 15)
DEFINE_REAL_TZRZF_TESTS(dtzrzf, double, FB_OP_DTZRZF, 300, 16)
DEFINE_COMPLEX_TZRZF_TESTS(ctzrzf, fb_complex_float_t, float, FB_OP_CTZRZF,
                           make_cf32, cf32_eq, 500, 17)
DEFINE_COMPLEX_TZRZF_TESTS(ztzrzf, fb_complex_double_t, double, FB_OP_ZTZRZF,
                           make_cf64, cf64_eq, 700, 18)

int main(void)
{
    int status = 0;

    status |= check_stzrzf_fortran_to_cblas();
    status |= check_stzrzf_cblas_to_fortran();
    status |= check_dtzrzf_fortran_to_cblas();
    status |= check_dtzrzf_cblas_to_fortran();
    status |= check_ctzrzf_fortran_to_cblas();
    status |= check_ctzrzf_cblas_to_fortran();
    status |= check_ztzrzf_fortran_to_cblas();
    status |= check_ztzrzf_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}