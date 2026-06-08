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

#define DEFINE_LASCL_REAL_TESTS(SUFFIX, TYPE, OP_ID, BASE)                     \
typedef int (*fb_##SUFFIX##_cblas_fn)(char type, int kl, int ku, TYPE cfrom,   \
                                      TYPE cto, int m, int n, TYPE *a, int lda);\
typedef void (*fb_##SUFFIX##_fortran_fn)(char *type, int *kl, int *ku,         \
                                         TYPE *cfrom, TYPE *cto, int *m, int *n,\
                                         TYPE *a, int *lda, int *info);         \
static struct {                                                                 \
    int called;                                                                 \
    char type;                                                                  \
    int kl;                                                                     \
    int ku;                                                                     \
    TYPE cfrom;                                                                 \
    TYPE cto;                                                                   \
    int m;                                                                      \
    int n;                                                                      \
    TYPE *a;                                                                    \
    int lda;                                                                    \
} g_##SUFFIX##_fortran_call;                                                    \
static struct {                                                                 \
    int called;                                                                 \
    char type;                                                                  \
    int kl;                                                                     \
    int ku;                                                                     \
    TYPE cfrom;                                                                 \
    TYPE cto;                                                                   \
    int m;                                                                      \
    int n;                                                                      \
    TYPE *a;                                                                    \
    int lda;                                                                    \
} g_##SUFFIX##_cblas_call;                                                      \
static void stub_##SUFFIX##_fortran(char *type, int *kl, int *ku, TYPE *cfrom, \
                                    TYPE *cto, int *m, int *n, TYPE *a,        \
                                    int *lda, int *info)                       \
{                                                                               \
    g_##SUFFIX##_fortran_call.called += 1;                                      \
    g_##SUFFIX##_fortran_call.type = *type;                                     \
    g_##SUFFIX##_fortran_call.kl = *kl;                                         \
    g_##SUFFIX##_fortran_call.ku = *ku;                                         \
    g_##SUFFIX##_fortran_call.cfrom = *cfrom;                                   \
    g_##SUFFIX##_fortran_call.cto = *cto;                                       \
    g_##SUFFIX##_fortran_call.m = *m;                                           \
    g_##SUFFIX##_fortran_call.n = *n;                                           \
    g_##SUFFIX##_fortran_call.a = a;                                            \
    g_##SUFFIX##_fortran_call.lda = *lda;                                       \
    a[0] = (TYPE)((BASE) + 1);                                                  \
    a[(size_t)(*lda) + 1] = (TYPE)((BASE) + 2);                                 \
    *info = (BASE) + 3;                                                         \
}                                                                               \
static int stub_##SUFFIX##_cblas(char type, int kl, int ku, TYPE cfrom,        \
                                 TYPE cto, int m, int n, TYPE *a, int lda)     \
{                                                                               \
    g_##SUFFIX##_cblas_call.called += 1;                                        \
    g_##SUFFIX##_cblas_call.type = type;                                        \
    g_##SUFFIX##_cblas_call.kl = kl;                                            \
    g_##SUFFIX##_cblas_call.ku = ku;                                            \
    g_##SUFFIX##_cblas_call.cfrom = cfrom;                                      \
    g_##SUFFIX##_cblas_call.cto = cto;                                          \
    g_##SUFFIX##_cblas_call.m = m;                                              \
    g_##SUFFIX##_cblas_call.n = n;                                              \
    g_##SUFFIX##_cblas_call.a = a;                                              \
    g_##SUFFIX##_cblas_call.lda = lda;                                          \
    a[0] = (TYPE)((BASE) + 4);                                                  \
    a[(size_t)lda + 1] = (TYPE)((BASE) + 5);                                    \
    return (BASE) + 6;                                                          \
}                                                                               \
static int check_##SUFFIX##_fortran_to_cblas(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                        \
    TYPE a[4] = { (TYPE)11, (TYPE)12, (TYPE)13, (TYPE)14 };                     \
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
    if (thunk('G', 1, 2, (TYPE)7, (TYPE)8, 2, 2, a, 2) != (BASE) + 3 ||         \
        g_##SUFFIX##_fortran_call.called != 1 ||                                \
        g_##SUFFIX##_fortran_call.type != 'G' ||                                \
        g_##SUFFIX##_fortran_call.kl != 1 ||                                    \
        g_##SUFFIX##_fortran_call.ku != 2 ||                                    \
        g_##SUFFIX##_fortran_call.cfrom != (TYPE)7 ||                           \
        g_##SUFFIX##_fortran_call.cto != (TYPE)8 ||                             \
        g_##SUFFIX##_fortran_call.m != 2 ||                                     \
        g_##SUFFIX##_fortran_call.n != 2 ||                                     \
        g_##SUFFIX##_fortran_call.a != a ||                                     \
        g_##SUFFIX##_fortran_call.lda != 2 ||                                   \
        a[0] != (TYPE)((BASE) + 1) ||                                           \
        a[3] != (TYPE)((BASE) + 2) ||                                           \
        a[1] != (TYPE)12 || a[2] != (TYPE)13) {                                 \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward LASCL inputs correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards LASCL inputs\n"); \
    return 0;                                                                   \
}                                                                               \
static int check_##SUFFIX##_cblas_to_fortran(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                      \
    char type = 'G';                                                            \
    int kl = 1;                                                                 \
    int ku = 2;                                                                 \
    TYPE cfrom = (TYPE)7;                                                       \
    TYPE cto = (TYPE)8;                                                         \
    int m = 2;                                                                  \
    int n = 2;                                                                  \
    int lda = 2;                                                                \
    int info = -1;                                                              \
    TYPE a[4] = { (TYPE)21, (TYPE)22, (TYPE)23, (TYPE)24 };                     \
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
    thunk(&type, &kl, &ku, &cfrom, &cto, &m, &n, a, &lda, &info);              \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                  \
        g_##SUFFIX##_cblas_call.type != 'G' ||                                  \
        g_##SUFFIX##_cblas_call.kl != 1 ||                                      \
        g_##SUFFIX##_cblas_call.ku != 2 ||                                      \
        g_##SUFFIX##_cblas_call.cfrom != (TYPE)7 ||                             \
        g_##SUFFIX##_cblas_call.cto != (TYPE)8 ||                               \
        g_##SUFFIX##_cblas_call.m != 2 ||                                       \
        g_##SUFFIX##_cblas_call.n != 2 ||                                       \
        g_##SUFFIX##_cblas_call.a != a ||                                       \
        g_##SUFFIX##_cblas_call.lda != 2 ||                                     \
        info != (BASE) + 6 ||                                                   \
        a[0] != (TYPE)((BASE) + 4) ||                                           \
        a[3] != (TYPE)((BASE) + 5) ||                                           \
        a[1] != (TYPE)22 || a[2] != (TYPE)23) {                                 \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference LASCL inputs correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences LASCL inputs\n"); \
    return 0;                                                                   \
}

#define DEFINE_LASCL_COMPLEX_TESTS(SUFFIX, CTYPE, RTYPE, OP_ID, MAKE_FN, EQ_FN, RBASE) \
typedef int (*fb_##SUFFIX##_cblas_fn)(char type, int kl, int ku, RTYPE cfrom,           \
                                      RTYPE cto, int m, int n, CTYPE *a, int lda);       \
typedef void (*fb_##SUFFIX##_fortran_fn)(char *type, int *kl, int *ku, RTYPE *cfrom,     \
                                         RTYPE *cto, int *m, int *n, CTYPE *a, int *lda,  \
                                         int *info);                                       \
static struct {                                                                            \
    int called;                                                                            \
    char type;                                                                             \
    int kl;                                                                                \
    int ku;                                                                                \
    RTYPE cfrom;                                                                           \
    RTYPE cto;                                                                             \
    int m;                                                                                 \
    int n;                                                                                 \
    CTYPE *a;                                                                              \
    int lda;                                                                               \
} g_##SUFFIX##_fortran_call;                                                               \
static struct {                                                                            \
    int called;                                                                            \
    char type;                                                                             \
    int kl;                                                                                \
    int ku;                                                                                \
    RTYPE cfrom;                                                                           \
    RTYPE cto;                                                                             \
    int m;                                                                                 \
    int n;                                                                                 \
    CTYPE *a;                                                                              \
    int lda;                                                                               \
} g_##SUFFIX##_cblas_call;                                                                 \
static void stub_##SUFFIX##_fortran(char *type, int *kl, int *ku, RTYPE *cfrom, RTYPE *cto,\
                                    int *m, int *n, CTYPE *a, int *lda, int *info)         \
{                                                                                          \
    g_##SUFFIX##_fortran_call.called += 1;                                                 \
    g_##SUFFIX##_fortran_call.type = *type;                                                \
    g_##SUFFIX##_fortran_call.kl = *kl;                                                    \
    g_##SUFFIX##_fortran_call.ku = *ku;                                                    \
    g_##SUFFIX##_fortran_call.cfrom = *cfrom;                                              \
    g_##SUFFIX##_fortran_call.cto = *cto;                                                  \
    g_##SUFFIX##_fortran_call.m = *m;                                                      \
    g_##SUFFIX##_fortran_call.n = *n;                                                      \
    g_##SUFFIX##_fortran_call.a = a;                                                       \
    g_##SUFFIX##_fortran_call.lda = *lda;                                                  \
    a[0] = MAKE_FN((RBASE) + 1, (RBASE) + 11);                                             \
    a[(size_t)(*lda) + 1] = MAKE_FN((RBASE) + 2, (RBASE) + 12);                            \
    *info = (RBASE) + 3;                                                                    \
}                                                                                          \
static int stub_##SUFFIX##_cblas(char type, int kl, int ku, RTYPE cfrom, RTYPE cto,       \
                                 int m, int n, CTYPE *a, int lda)                          \
{                                                                                          \
    g_##SUFFIX##_cblas_call.called += 1;                                                   \
    g_##SUFFIX##_cblas_call.type = type;                                                   \
    g_##SUFFIX##_cblas_call.kl = kl;                                                       \
    g_##SUFFIX##_cblas_call.ku = ku;                                                       \
    g_##SUFFIX##_cblas_call.cfrom = cfrom;                                                 \
    g_##SUFFIX##_cblas_call.cto = cto;                                                     \
    g_##SUFFIX##_cblas_call.m = m;                                                         \
    g_##SUFFIX##_cblas_call.n = n;                                                         \
    g_##SUFFIX##_cblas_call.a = a;                                                         \
    g_##SUFFIX##_cblas_call.lda = lda;                                                     \
    a[0] = MAKE_FN((RBASE) + 4, (RBASE) + 14);                                             \
    a[(size_t)lda + 1] = MAKE_FN((RBASE) + 5, (RBASE) + 15);                               \
    return (RBASE) + 6;                                                                    \
}                                                                                          \
static int check_##SUFFIX##_fortran_to_cblas(void)                                         \
{                                                                                          \
    fb_backend_vtable_t vtable;                                                            \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                                   \
    CTYPE a[4] = { MAKE_FN(11, 12), MAKE_FN(21, 22), MAKE_FN(31, 32), MAKE_FN(41, 42) };  \
    memset(&vtable, 0, sizeof(vtable));                                                    \
    memset(&g_##SUFFIX##_fortran_call, 0, sizeof(g_##SUFFIX##_fortran_call));              \
    vtable.ext_ops[OP_ID][FB_CONV_FORTRAN] =                                               \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_fortran;                            \
    fb_install_conv_thunks(&vtable, OP_ID);                                                \
    thunk = (fb_##SUFFIX##_cblas_fn)vtable.ext_ops[OP_ID][FB_CONV_CBLAS];                  \
    if (!thunk) {                                                                          \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk was not installed\n"); \
        return 1;                                                                          \
    }                                                                                      \
    if (thunk('G', 1, 2, (RTYPE)7, (RTYPE)8, 2, 2, a, 2) != (RBASE) + 3 ||                 \
        g_##SUFFIX##_fortran_call.called != 1 ||                                           \
        g_##SUFFIX##_fortran_call.type != 'G' ||                                           \
        g_##SUFFIX##_fortran_call.kl != 1 ||                                               \
        g_##SUFFIX##_fortran_call.ku != 2 ||                                               \
        g_##SUFFIX##_fortran_call.cfrom != (RTYPE)7 ||                                     \
        g_##SUFFIX##_fortran_call.cto != (RTYPE)8 ||                                       \
        g_##SUFFIX##_fortran_call.m != 2 ||                                                \
        g_##SUFFIX##_fortran_call.n != 2 ||                                                \
        g_##SUFFIX##_fortran_call.a != a ||                                                \
        g_##SUFFIX##_fortran_call.lda != 2 ||                                              \
        !EQ_FN(a[0], MAKE_FN((RBASE) + 1, (RBASE) + 11)) ||                                \
        !EQ_FN(a[3], MAKE_FN((RBASE) + 2, (RBASE) + 12)) ||                                \
        !EQ_FN(a[1], MAKE_FN(21, 22)) ||                                                   \
        !EQ_FN(a[2], MAKE_FN(31, 32))) {                                                   \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward LASCL inputs correctly\n"); \
        return 1;                                                                          \
    }                                                                                      \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards LASCL inputs\n");         \
    return 0;                                                                              \
}                                                                                          \
static int check_##SUFFIX##_cblas_to_fortran(void)                                         \
{                                                                                          \
    fb_backend_vtable_t vtable;                                                            \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                                 \
    char type = 'G';                                                                       \
    int kl = 1;                                                                            \
    int ku = 2;                                                                            \
    RTYPE cfrom = (RTYPE)7;                                                                \
    RTYPE cto = (RTYPE)8;                                                                  \
    int m = 2;                                                                             \
    int n = 2;                                                                             \
    int lda = 2;                                                                           \
    int info = -1;                                                                         \
    CTYPE a[4] = { MAKE_FN(51, 52), MAKE_FN(61, 62), MAKE_FN(71, 72), MAKE_FN(81, 82) };  \
    memset(&vtable, 0, sizeof(vtable));                                                    \
    memset(&g_##SUFFIX##_cblas_call, 0, sizeof(g_##SUFFIX##_cblas_call));                  \
    vtable.ext_ops[OP_ID][FB_CONV_CBLAS] =                                                 \
        (fb_generic_fn)(void (*)(void))stub_##SUFFIX##_cblas;                              \
    fb_install_conv_thunks(&vtable, OP_ID);                                                \
    thunk = (fb_##SUFFIX##_fortran_fn)vtable.ext_ops[OP_ID][FB_CONV_FORTRAN];              \
    if (!thunk) {                                                                          \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk was not installed\n"); \
        return 1;                                                                          \
    }                                                                                      \
    thunk(&type, &kl, &ku, &cfrom, &cto, &m, &n, a, &lda, &info);                          \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                             \
        g_##SUFFIX##_cblas_call.type != 'G' ||                                             \
        g_##SUFFIX##_cblas_call.kl != 1 ||                                                 \
        g_##SUFFIX##_cblas_call.ku != 2 ||                                                 \
        g_##SUFFIX##_cblas_call.cfrom != (RTYPE)7 ||                                       \
        g_##SUFFIX##_cblas_call.cto != (RTYPE)8 ||                                         \
        g_##SUFFIX##_cblas_call.m != 2 ||                                                  \
        g_##SUFFIX##_cblas_call.n != 2 ||                                                  \
        g_##SUFFIX##_cblas_call.a != a ||                                                  \
        g_##SUFFIX##_cblas_call.lda != 2 ||                                                \
        info != (RBASE) + 6 ||                                                             \
        !EQ_FN(a[0], MAKE_FN((RBASE) + 4, (RBASE) + 14)) ||                                \
        !EQ_FN(a[3], MAKE_FN((RBASE) + 5, (RBASE) + 15)) ||                                \
        !EQ_FN(a[1], MAKE_FN(61, 62)) ||                                                   \
        !EQ_FN(a[2], MAKE_FN(71, 72))) {                                                   \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference LASCL inputs correctly\n"); \
        return 1;                                                                          \
    }                                                                                      \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences LASCL inputs\n");     \
    return 0;                                                                              \
}

DEFINE_LASCL_REAL_TESTS(slascl, float, FB_OP_SLASCL, 100)
DEFINE_LASCL_REAL_TESTS(dlascl, double, FB_OP_DLASCL, 300)
DEFINE_LASCL_COMPLEX_TESTS(clascl, fb_complex_float_t, float, FB_OP_CLASCL, make_cf32, cf32_eq, 500)
DEFINE_LASCL_COMPLEX_TESTS(zlascl, fb_complex_double_t, double, FB_OP_ZLASCL, make_cf64, cf64_eq, 700)

int main(void)
{
    int status = 0;

    status |= check_slascl_fortran_to_cblas();
    status |= check_slascl_cblas_to_fortran();
    status |= check_dlascl_fortran_to_cblas();
    status |= check_dlascl_cblas_to_fortran();
    status |= check_clascl_fortran_to_cblas();
    status |= check_clascl_cblas_to_fortran();
    status |= check_zlascl_fortran_to_cblas();
    status |= check_zlascl_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}