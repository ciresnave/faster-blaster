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

#define DEFINE_REAL_LACPY_TESTS(SUFFIX, TYPE, OP_ID, BASE)                     \
typedef int (*fb_##SUFFIX##_cblas_fn)(fb_layout_t layout, char uplo, int m,    \
                                      int n, const TYPE *a, int lda, TYPE *b, \
                                      int ldb);                                \
typedef void (*fb_##SUFFIX##_fortran_fn)(char *uplo, int *m, int *n,          \
                                         const TYPE *a, int *lda, TYPE *b,    \
                                         int *ldb);                           \
static struct {                                                                 \
    int called;                                                                 \
    char uplo;                                                                  \
    int m;                                                                      \
    int n;                                                                      \
    int lda;                                                                    \
    int ldb;                                                                    \
    const TYPE *a;                                                              \
    TYPE *b;                                                                    \
    TYPE a_snapshot[6];                                                         \
} g_##SUFFIX##_fortran_call;                                                    \
static struct {                                                                 \
    int called;                                                                 \
    fb_layout_t layout;                                                         \
    char uplo;                                                                  \
    int m;                                                                      \
    int n;                                                                      \
    int lda;                                                                    \
    int ldb;                                                                    \
    const TYPE *a;                                                              \
    TYPE *b;                                                                    \
} g_##SUFFIX##_cblas_call;                                                      \
static void stub_##SUFFIX##_fortran(char *uplo, int *m, int *n,                \
                                    const TYPE *a, int *lda, TYPE *b,         \
                                    int *ldb)                                  \
{                                                                               \
    TYPE out_snapshot[6] = { (TYPE)((BASE) + 11), (TYPE)0, (TYPE)0,           \
                             (TYPE)((BASE) + 12), (TYPE)((BASE) + 13),         \
                             (TYPE)((BASE) + 14) };                            \
    int index;                                                                  \
    g_##SUFFIX##_fortran_call.called += 1;                                      \
    g_##SUFFIX##_fortran_call.uplo = *uplo;                                     \
    g_##SUFFIX##_fortran_call.m = *m;                                           \
    g_##SUFFIX##_fortran_call.n = *n;                                           \
    g_##SUFFIX##_fortran_call.lda = *lda;                                       \
    g_##SUFFIX##_fortran_call.ldb = *ldb;                                       \
    g_##SUFFIX##_fortran_call.a = a;                                            \
    g_##SUFFIX##_fortran_call.b = b;                                            \
    for (index = 0; index < 6; ++index) {                                       \
        g_##SUFFIX##_fortran_call.a_snapshot[index] = a[index];                 \
    }                                                                           \
    memcpy(b, out_snapshot, sizeof(out_snapshot));                              \
}                                                                               \
static int stub_##SUFFIX##_cblas(fb_layout_t layout, char uplo, int m, int n,  \
                                 const TYPE *a, int lda, TYPE *b, int ldb)     \
{                                                                               \
    g_##SUFFIX##_cblas_call.called += 1;                                        \
    g_##SUFFIX##_cblas_call.layout = layout;                                    \
    g_##SUFFIX##_cblas_call.uplo = uplo;                                        \
    g_##SUFFIX##_cblas_call.m = m;                                              \
    g_##SUFFIX##_cblas_call.n = n;                                              \
    g_##SUFFIX##_cblas_call.lda = lda;                                          \
    g_##SUFFIX##_cblas_call.ldb = ldb;                                          \
    g_##SUFFIX##_cblas_call.a = a;                                              \
    g_##SUFFIX##_cblas_call.b = b;                                              \
    b[0] = (TYPE)((BASE) + 21);                                                 \
    b[4] = (TYPE)((BASE) + 22);                                                 \
    b[5] = (TYPE)((BASE) + 23);                                                 \
    return (BASE) + 99;                                                         \
}                                                                               \
static int check_##SUFFIX##_fortran_to_cblas(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                        \
    TYPE a[6] = { (TYPE)1, (TYPE)2, (TYPE)3, (TYPE)4, (TYPE)5, (TYPE)6 };      \
    TYPE b[6] = { (TYPE)-1, (TYPE)-1, (TYPE)-1, (TYPE)-1, (TYPE)-1, (TYPE)-1 };\
    TYPE expected_a_col[6] = { (TYPE)1, (TYPE)3, (TYPE)5,                      \
                               (TYPE)2, (TYPE)4, (TYPE)6 };                    \
    TYPE expected_b_row[6] = { (TYPE)((BASE) + 11), (TYPE)((BASE) + 12),       \
                               (TYPE)0, (TYPE)((BASE) + 13),                   \
                               (TYPE)0, (TYPE)((BASE) + 14) };                 \
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
    if (thunk(FB_LAYOUT_ROW_MAJOR, 'U', 3, 2, a, 2, b, 2) != 0 ||              \
        g_##SUFFIX##_fortran_call.called != 1 ||                                \
        g_##SUFFIX##_fortran_call.uplo != 'U' ||                                \
        g_##SUFFIX##_fortran_call.m != 3 ||                                     \
        g_##SUFFIX##_fortran_call.n != 2 ||                                     \
        g_##SUFFIX##_fortran_call.lda != 3 ||                                   \
        g_##SUFFIX##_fortran_call.ldb != 3 ||                                   \
        g_##SUFFIX##_fortran_call.a == a ||                                     \
        g_##SUFFIX##_fortran_call.b == b ||                                     \
        memcmp(g_##SUFFIX##_fortran_call.a_snapshot, expected_a_col,            \
               sizeof(expected_a_col)) != 0 ||                                  \
        memcmp(b, expected_b_row, sizeof(expected_b_row)) != 0) {              \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not preserve row-major matrix-copy staging without flipping uplo\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk stages row-major matrices through col-major storage without flipping uplo\n"); \
    return 0;                                                                   \
}                                                                               \
static int check_##SUFFIX##_cblas_to_fortran(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                      \
    char uplo = 'L';                                                            \
    int m = 3;                                                                  \
    int n = 2;                                                                  \
    int lda = 3;                                                                \
    int ldb = 3;                                                                \
    TYPE a[6] = { (TYPE)1, (TYPE)3, (TYPE)5, (TYPE)2, (TYPE)4, (TYPE)6 };      \
    TYPE b[6] = { (TYPE)-1, (TYPE)-1, (TYPE)-1, (TYPE)-1, (TYPE)-1, (TYPE)-1 };\
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
    thunk(&uplo, &m, &n, a, &lda, b, &ldb);                                     \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                  \
        g_##SUFFIX##_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||                \
        g_##SUFFIX##_cblas_call.uplo != 'L' ||                                  \
        g_##SUFFIX##_cblas_call.m != 3 ||                                       \
        g_##SUFFIX##_cblas_call.n != 2 ||                                       \
        g_##SUFFIX##_cblas_call.lda != 3 ||                                     \
        g_##SUFFIX##_cblas_call.ldb != 3 ||                                     \
        g_##SUFFIX##_cblas_call.a != a ||                                       \
        g_##SUFFIX##_cblas_call.b != b ||                                       \
        b[0] != (TYPE)((BASE) + 21) ||                                          \
        b[4] != (TYPE)((BASE) + 22) ||                                          \
        b[5] != (TYPE)((BASE) + 23)) {                                          \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not forward col-major matrix-copy arguments correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk forwards col-major matrix-copy arguments and ignores the C int return\n"); \
    return 0;                                                                   \
}

#define DEFINE_COMPLEX_LACPY_TESTS(SUFFIX, CTYPE, OP_ID, MAKE_FN, EQ_FN, BASE) \
typedef int (*fb_##SUFFIX##_cblas_fn)(fb_layout_t layout, char uplo, int m,    \
                                      int n, const CTYPE *a, int lda, CTYPE *b,\
                                      int ldb);                                 \
typedef void (*fb_##SUFFIX##_fortran_fn)(char *uplo, int *m, int *n,           \
                                         const CTYPE *a, int *lda, CTYPE *b,   \
                                         int *ldb);                            \
static struct {                                                                 \
    int called;                                                                 \
    char uplo;                                                                  \
    int m;                                                                      \
    int n;                                                                      \
    int lda;                                                                    \
    int ldb;                                                                    \
    const CTYPE *a;                                                             \
    CTYPE *b;                                                                   \
    CTYPE a_snapshot[6];                                                        \
} g_##SUFFIX##_fortran_call;                                                    \
static struct {                                                                 \
    int called;                                                                 \
    fb_layout_t layout;                                                         \
    char uplo;                                                                  \
    int m;                                                                      \
    int n;                                                                      \
    int lda;                                                                    \
    int ldb;                                                                    \
    const CTYPE *a;                                                             \
    CTYPE *b;                                                                   \
} g_##SUFFIX##_cblas_call;                                                      \
static void stub_##SUFFIX##_fortran(char *uplo, int *m, int *n,                \
                                    const CTYPE *a, int *lda, CTYPE *b,       \
                                    int *ldb)                                  \
{                                                                               \
    CTYPE out_snapshot[6];                                                      \
    int index;                                                                  \
    out_snapshot[0] = MAKE_FN((BASE) + 11, (BASE) + 111);                      \
    out_snapshot[1] = MAKE_FN(0, 0);                                            \
    out_snapshot[2] = MAKE_FN(0, 0);                                            \
    out_snapshot[3] = MAKE_FN((BASE) + 12, (BASE) + 112);                      \
    out_snapshot[4] = MAKE_FN((BASE) + 13, (BASE) + 113);                      \
    out_snapshot[5] = MAKE_FN((BASE) + 14, (BASE) + 114);                      \
    g_##SUFFIX##_fortran_call.called += 1;                                      \
    g_##SUFFIX##_fortran_call.uplo = *uplo;                                     \
    g_##SUFFIX##_fortran_call.m = *m;                                           \
    g_##SUFFIX##_fortran_call.n = *n;                                           \
    g_##SUFFIX##_fortran_call.lda = *lda;                                       \
    g_##SUFFIX##_fortran_call.ldb = *ldb;                                       \
    g_##SUFFIX##_fortran_call.a = a;                                            \
    g_##SUFFIX##_fortran_call.b = b;                                            \
    for (index = 0; index < 6; ++index) {                                       \
        g_##SUFFIX##_fortran_call.a_snapshot[index] = a[index];                 \
    }                                                                           \
    memcpy(b, out_snapshot, sizeof(out_snapshot));                              \
}                                                                               \
static int stub_##SUFFIX##_cblas(fb_layout_t layout, char uplo, int m, int n,  \
                                 const CTYPE *a, int lda, CTYPE *b, int ldb)   \
{                                                                               \
    g_##SUFFIX##_cblas_call.called += 1;                                        \
    g_##SUFFIX##_cblas_call.layout = layout;                                    \
    g_##SUFFIX##_cblas_call.uplo = uplo;                                        \
    g_##SUFFIX##_cblas_call.m = m;                                              \
    g_##SUFFIX##_cblas_call.n = n;                                              \
    g_##SUFFIX##_cblas_call.lda = lda;                                          \
    g_##SUFFIX##_cblas_call.ldb = ldb;                                          \
    g_##SUFFIX##_cblas_call.a = a;                                              \
    g_##SUFFIX##_cblas_call.b = b;                                              \
    b[0] = MAKE_FN((BASE) + 21, (BASE) + 121);                                 \
    b[4] = MAKE_FN((BASE) + 22, (BASE) + 122);                                 \
    b[5] = MAKE_FN((BASE) + 23, (BASE) + 123);                                 \
    return (BASE) + 99;                                                         \
}                                                                               \
static int check_##SUFFIX##_fortran_to_cblas(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                        \
    CTYPE a[6];                                                                 \
    CTYPE b[6];                                                                 \
    CTYPE expected_a_col[6];                                                    \
    CTYPE expected_b_row[6];                                                    \
    int index;                                                                  \
    a[0] = MAKE_FN(1, 101);                                                     \
    a[1] = MAKE_FN(2, 102);                                                     \
    a[2] = MAKE_FN(3, 103);                                                     \
    a[3] = MAKE_FN(4, 104);                                                     \
    a[4] = MAKE_FN(5, 105);                                                     \
    a[5] = MAKE_FN(6, 106);                                                     \
    for (index = 0; index < 6; ++index) {                                       \
        b[index] = MAKE_FN(-1, -1);                                             \
    }                                                                           \
    expected_a_col[0] = MAKE_FN(1, 101);                                        \
    expected_a_col[1] = MAKE_FN(3, 103);                                        \
    expected_a_col[2] = MAKE_FN(5, 105);                                        \
    expected_a_col[3] = MAKE_FN(2, 102);                                        \
    expected_a_col[4] = MAKE_FN(4, 104);                                        \
    expected_a_col[5] = MAKE_FN(6, 106);                                        \
    expected_b_row[0] = MAKE_FN((BASE) + 11, (BASE) + 111);                    \
    expected_b_row[1] = MAKE_FN((BASE) + 12, (BASE) + 112);                    \
    expected_b_row[2] = MAKE_FN(0, 0);                                          \
    expected_b_row[3] = MAKE_FN((BASE) + 13, (BASE) + 113);                    \
    expected_b_row[4] = MAKE_FN(0, 0);                                          \
    expected_b_row[5] = MAKE_FN((BASE) + 14, (BASE) + 114);                    \
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
    if (thunk(FB_LAYOUT_ROW_MAJOR, 'U', 3, 2, a, 2, b, 2) != 0 ||              \
        g_##SUFFIX##_fortran_call.called != 1 ||                                \
        g_##SUFFIX##_fortran_call.uplo != 'U' ||                                \
        g_##SUFFIX##_fortran_call.m != 3 ||                                     \
        g_##SUFFIX##_fortran_call.n != 2 ||                                     \
        g_##SUFFIX##_fortran_call.lda != 3 ||                                   \
        g_##SUFFIX##_fortran_call.ldb != 3 ||                                   \
        g_##SUFFIX##_fortran_call.a == a ||                                     \
        g_##SUFFIX##_fortran_call.b == b ||                                     \
        !EQ_FN(g_##SUFFIX##_fortran_call.a_snapshot[0], expected_a_col[0]) ||   \
        !EQ_FN(g_##SUFFIX##_fortran_call.a_snapshot[1], expected_a_col[1]) ||   \
        !EQ_FN(g_##SUFFIX##_fortran_call.a_snapshot[2], expected_a_col[2]) ||   \
        !EQ_FN(g_##SUFFIX##_fortran_call.a_snapshot[3], expected_a_col[3]) ||   \
        !EQ_FN(g_##SUFFIX##_fortran_call.a_snapshot[4], expected_a_col[4]) ||   \
        !EQ_FN(g_##SUFFIX##_fortran_call.a_snapshot[5], expected_a_col[5]) ||   \
        !EQ_FN(b[0], expected_b_row[0]) || !EQ_FN(b[1], expected_b_row[1]) ||   \
        !EQ_FN(b[2], expected_b_row[2]) || !EQ_FN(b[3], expected_b_row[3]) ||   \
        !EQ_FN(b[4], expected_b_row[4]) || !EQ_FN(b[5], expected_b_row[5])) {   \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not preserve row-major complex matrix-copy staging without flipping uplo\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk stages row-major complex matrices through col-major storage without flipping uplo\n"); \
    return 0;                                                                   \
}                                                                               \
static int check_##SUFFIX##_cblas_to_fortran(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                      \
    char uplo = 'L';                                                            \
    int m = 3;                                                                  \
    int n = 2;                                                                  \
    int lda = 3;                                                                \
    int ldb = 3;                                                                \
    CTYPE a[6];                                                                 \
    CTYPE b[6];                                                                 \
    int index;                                                                  \
    a[0] = MAKE_FN(1, 101);                                                     \
    a[1] = MAKE_FN(3, 103);                                                     \
    a[2] = MAKE_FN(5, 105);                                                     \
    a[3] = MAKE_FN(2, 102);                                                     \
    a[4] = MAKE_FN(4, 104);                                                     \
    a[5] = MAKE_FN(6, 106);                                                     \
    for (index = 0; index < 6; ++index) {                                       \
        b[index] = MAKE_FN(-1, -1);                                             \
    }                                                                           \
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
    thunk(&uplo, &m, &n, a, &lda, b, &ldb);                                     \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                  \
        g_##SUFFIX##_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||                \
        g_##SUFFIX##_cblas_call.uplo != 'L' ||                                  \
        g_##SUFFIX##_cblas_call.m != 3 ||                                       \
        g_##SUFFIX##_cblas_call.n != 2 ||                                       \
        g_##SUFFIX##_cblas_call.lda != 3 ||                                     \
        g_##SUFFIX##_cblas_call.ldb != 3 ||                                     \
        g_##SUFFIX##_cblas_call.a != a ||                                       \
        g_##SUFFIX##_cblas_call.b != b ||                                       \
        !EQ_FN(b[0], MAKE_FN((BASE) + 21, (BASE) + 121)) ||                    \
        !EQ_FN(b[4], MAKE_FN((BASE) + 22, (BASE) + 122)) ||                    \
        !EQ_FN(b[5], MAKE_FN((BASE) + 23, (BASE) + 123))) {                    \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not forward col-major complex matrix-copy arguments correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk forwards col-major complex matrix-copy arguments and ignores the C int return\n"); \
    return 0;                                                                   \
}

DEFINE_REAL_LACPY_TESTS(slacpy, float, FB_OP_SLACPY, 100)
DEFINE_REAL_LACPY_TESTS(dlacpy, double, FB_OP_DLACPY, 300)
DEFINE_COMPLEX_LACPY_TESTS(clacpy, fb_complex_float_t, FB_OP_CLACPY,
                           make_cf32, cf32_eq, 500)
DEFINE_COMPLEX_LACPY_TESTS(zlacpy, fb_complex_double_t, FB_OP_ZLACPY,
                           make_cf64, cf64_eq, 700)

int main(void)
{
    int status = 0;

    status |= check_slacpy_fortran_to_cblas();
    status |= check_slacpy_cblas_to_fortran();
    status |= check_dlacpy_fortran_to_cblas();
    status |= check_dlacpy_cblas_to_fortran();
    status |= check_clacpy_fortran_to_cblas();
    status |= check_clacpy_cblas_to_fortran();
    status |= check_zlacpy_fortran_to_cblas();
    status |= check_zlacpy_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}