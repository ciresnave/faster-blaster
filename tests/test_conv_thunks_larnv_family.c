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

#define DEFINE_LARNV_REAL_TESTS(SUFFIX, TYPE, OP_ID, BASE)                     \
typedef int (*fb_##SUFFIX##_cblas_fn)(int idist, int *iseed, int n, TYPE *x);  \
typedef void (*fb_##SUFFIX##_fortran_fn)(int *idist, int *iseed, int *n,       \
                                         TYPE *x);                             \
static struct {                                                                 \
    int called;                                                                 \
    int idist;                                                                  \
    int *iseed;                                                                 \
    int n;                                                                      \
    TYPE *x;                                                                    \
} g_##SUFFIX##_fortran_call;                                                    \
static struct {                                                                 \
    int called;                                                                 \
    int idist;                                                                  \
    int *iseed;                                                                 \
    int n;                                                                      \
    TYPE *x;                                                                    \
} g_##SUFFIX##_cblas_call;                                                      \
static void stub_##SUFFIX##_fortran(int *idist, int *iseed, int *n, TYPE *x)   \
{                                                                               \
    g_##SUFFIX##_fortran_call.called += 1;                                      \
    g_##SUFFIX##_fortran_call.idist = *idist;                                   \
    g_##SUFFIX##_fortran_call.iseed = iseed;                                    \
    g_##SUFFIX##_fortran_call.n = *n;                                           \
    g_##SUFFIX##_fortran_call.x = x;                                            \
    iseed[0] = (BASE) + 1;                                                      \
    iseed[1] = (BASE) + 2;                                                      \
    x[0] = (TYPE)((BASE) + 3);                                                  \
    x[1] = (TYPE)((BASE) + 4);                                                  \
}                                                                               \
static int stub_##SUFFIX##_cblas(int idist, int *iseed, int n, TYPE *x)        \
{                                                                               \
    g_##SUFFIX##_cblas_call.called += 1;                                        \
    g_##SUFFIX##_cblas_call.idist = idist;                                      \
    g_##SUFFIX##_cblas_call.iseed = iseed;                                      \
    g_##SUFFIX##_cblas_call.n = n;                                              \
    g_##SUFFIX##_cblas_call.x = x;                                              \
    iseed[0] = (BASE) + 10;                                                     \
    iseed[1] = (BASE) + 11;                                                     \
    x[0] = (TYPE)((BASE) + 12);                                                 \
    x[1] = (TYPE)((BASE) + 13);                                                 \
    return (BASE) + 99;                                                         \
}                                                                               \
static int check_##SUFFIX##_fortran_to_cblas(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                        \
    int iseed[4] = { 1, 2, 3, 4 };                                              \
    TYPE x[2] = { (TYPE)0, (TYPE)0 };                                           \
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
    if (thunk(5, iseed, 2, x) != 0 ||                                           \
        g_##SUFFIX##_fortran_call.called != 1 ||                                \
        g_##SUFFIX##_fortran_call.idist != 5 ||                                 \
        g_##SUFFIX##_fortran_call.iseed != iseed ||                             \
        g_##SUFFIX##_fortran_call.n != 2 ||                                     \
        g_##SUFFIX##_fortran_call.x != x ||                                     \
        iseed[0] != (BASE) + 1 || iseed[1] != (BASE) + 2 ||                     \
        x[0] != (TYPE)((BASE) + 3) || x[1] != (TYPE)((BASE) + 4)) {             \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward LARNV inputs or outputs correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards LARNV inputs and returns success\n"); \
    return 0;                                                                   \
}                                                                               \
static int check_##SUFFIX##_cblas_to_fortran(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                      \
    int idist = 2;                                                              \
    int n = 2;                                                                  \
    int iseed[4] = { 5, 6, 7, 8 };                                              \
    TYPE x[2] = { (TYPE)0, (TYPE)0 };                                           \
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
    thunk(&idist, iseed, &n, x);                                                \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                  \
        g_##SUFFIX##_cblas_call.idist != 2 ||                                   \
        g_##SUFFIX##_cblas_call.iseed != iseed ||                               \
        g_##SUFFIX##_cblas_call.n != 2 ||                                       \
        g_##SUFFIX##_cblas_call.x != x ||                                       \
        iseed[0] != (BASE) + 10 || iseed[1] != (BASE) + 11 ||                   \
        x[0] != (TYPE)((BASE) + 12) || x[1] != (TYPE)((BASE) + 13)) {           \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference LARNV inputs or propagate outputs correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences LARNV inputs and ignores the C int return\n"); \
    return 0;                                                                   \
}

#define DEFINE_LARNV_COMPLEX_TESTS(SUFFIX, CTYPE, OP_ID, MAKE_FN, EQ_FN, BASE) \
typedef int (*fb_##SUFFIX##_cblas_fn)(int idist, int *iseed, int n, CTYPE *x); \
typedef void (*fb_##SUFFIX##_fortran_fn)(int *idist, int *iseed, int *n,       \
                                         CTYPE *x);                            \
static struct {                                                                 \
    int called;                                                                 \
    int idist;                                                                  \
    int *iseed;                                                                 \
    int n;                                                                      \
    CTYPE *x;                                                                   \
} g_##SUFFIX##_fortran_call;                                                    \
static struct {                                                                 \
    int called;                                                                 \
    int idist;                                                                  \
    int *iseed;                                                                 \
    int n;                                                                      \
    CTYPE *x;                                                                   \
} g_##SUFFIX##_cblas_call;                                                      \
static void stub_##SUFFIX##_fortran(int *idist, int *iseed, int *n, CTYPE *x)  \
{                                                                               \
    g_##SUFFIX##_fortran_call.called += 1;                                      \
    g_##SUFFIX##_fortran_call.idist = *idist;                                   \
    g_##SUFFIX##_fortran_call.iseed = iseed;                                    \
    g_##SUFFIX##_fortran_call.n = *n;                                           \
    g_##SUFFIX##_fortran_call.x = x;                                            \
    iseed[0] = (BASE) + 1;                                                      \
    iseed[1] = (BASE) + 2;                                                      \
    x[0] = MAKE_FN((BASE) + 3, (BASE) + 13);                                    \
    x[1] = MAKE_FN((BASE) + 4, (BASE) + 14);                                    \
}                                                                               \
static int stub_##SUFFIX##_cblas(int idist, int *iseed, int n, CTYPE *x)       \
{                                                                               \
    g_##SUFFIX##_cblas_call.called += 1;                                        \
    g_##SUFFIX##_cblas_call.idist = idist;                                      \
    g_##SUFFIX##_cblas_call.iseed = iseed;                                      \
    g_##SUFFIX##_cblas_call.n = n;                                              \
    g_##SUFFIX##_cblas_call.x = x;                                              \
    iseed[0] = (BASE) + 10;                                                     \
    iseed[1] = (BASE) + 11;                                                     \
    x[0] = MAKE_FN((BASE) + 12, (BASE) + 22);                                   \
    x[1] = MAKE_FN((BASE) + 13, (BASE) + 23);                                   \
    return (BASE) + 99;                                                         \
}                                                                               \
static int check_##SUFFIX##_fortran_to_cblas(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                        \
    int iseed[4] = { 1, 2, 3, 4 };                                              \
    CTYPE x[2] = { MAKE_FN(0, 0), MAKE_FN(0, 0) };                             \
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
    if (thunk(4, iseed, 2, x) != 0 ||                                           \
        g_##SUFFIX##_fortran_call.called != 1 ||                                \
        g_##SUFFIX##_fortran_call.idist != 4 ||                                 \
        g_##SUFFIX##_fortran_call.iseed != iseed ||                             \
        g_##SUFFIX##_fortran_call.n != 2 ||                                     \
        g_##SUFFIX##_fortran_call.x != x ||                                     \
        iseed[0] != (BASE) + 1 || iseed[1] != (BASE) + 2 ||                     \
        !EQ_FN(x[0], MAKE_FN((BASE) + 3, (BASE) + 13)) ||                       \
        !EQ_FN(x[1], MAKE_FN((BASE) + 4, (BASE) + 14))) {                       \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward complex LARNV inputs or outputs correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards complex LARNV inputs and returns success\n"); \
    return 0;                                                                   \
}                                                                               \
static int check_##SUFFIX##_cblas_to_fortran(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                      \
    int idist = 5;                                                              \
    int n = 2;                                                                  \
    int iseed[4] = { 5, 6, 7, 8 };                                              \
    CTYPE x[2] = { MAKE_FN(0, 0), MAKE_FN(0, 0) };                             \
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
    thunk(&idist, iseed, &n, x);                                                \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                  \
        g_##SUFFIX##_cblas_call.idist != 5 ||                                   \
        g_##SUFFIX##_cblas_call.iseed != iseed ||                               \
        g_##SUFFIX##_cblas_call.n != 2 ||                                       \
        g_##SUFFIX##_cblas_call.x != x ||                                       \
        iseed[0] != (BASE) + 10 || iseed[1] != (BASE) + 11 ||                   \
        !EQ_FN(x[0], MAKE_FN((BASE) + 12, (BASE) + 22)) ||                      \
        !EQ_FN(x[1], MAKE_FN((BASE) + 13, (BASE) + 23))) {                      \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference complex LARNV inputs or propagate outputs correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences complex LARNV inputs and ignores the C int return\n"); \
    return 0;                                                                   \
}

DEFINE_LARNV_REAL_TESTS(slarnv, float, FB_OP_SLARNV, 100)
DEFINE_LARNV_REAL_TESTS(dlarnv, double, FB_OP_DLARNV, 300)
DEFINE_LARNV_COMPLEX_TESTS(clarnv, fb_complex_float_t, FB_OP_CLARNV,
                           make_cf32, cf32_eq, 500)
DEFINE_LARNV_COMPLEX_TESTS(zlarnv, fb_complex_double_t, FB_OP_ZLARNV,
                           make_cf64, cf64_eq, 700)

int main(void)
{
    int status = 0;

    status |= check_slarnv_fortran_to_cblas();
    status |= check_slarnv_cblas_to_fortran();
    status |= check_dlarnv_fortran_to_cblas();
    status |= check_dlarnv_cblas_to_fortran();
    status |= check_clarnv_fortran_to_cblas();
    status |= check_clarnv_cblas_to_fortran();
    status |= check_zlarnv_fortran_to_cblas();
    status |= check_zlarnv_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}