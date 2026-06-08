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

#define DEFINE_REAL_LATRD_TESTS(SUFFIX, TYPE, OP_ID, BASE)                     \
typedef int (*fb_##SUFFIX##_cblas_fn)(fb_layout_t layout, char uplo, int n,    \
                                      int nb, TYPE *a, int lda, TYPE *e,       \
                                      TYPE *tau, TYPE *w, int ldw);            \
typedef void (*fb_##SUFFIX##_fortran_fn)(char *uplo, int *n, int *nb, TYPE *a, \
                                         int *lda, TYPE *e, TYPE *tau, TYPE *w, \
                                         int *ldw);                            \
static struct {                                                                 \
    int called;                                                                 \
    char uplo;                                                                  \
    int n;                                                                      \
    int nb;                                                                     \
    int lda;                                                                    \
    int ldw;                                                                    \
    TYPE *a;                                                                    \
    TYPE *w;                                                                    \
    TYPE a_snapshot[9];                                                         \
} g_##SUFFIX##_fortran_call;                                                    \
static struct {                                                                 \
    int called;                                                                 \
    fb_layout_t layout;                                                         \
    char uplo;                                                                  \
    int n;                                                                      \
    int nb;                                                                     \
    int lda;                                                                    \
    int ldw;                                                                    \
    TYPE *a;                                                                    \
    TYPE *e;                                                                    \
    TYPE *tau;                                                                  \
    TYPE *w;                                                                    \
} g_##SUFFIX##_cblas_call;                                                      \
static void stub_##SUFFIX##_fortran(char *uplo, int *n, int *nb, TYPE *a,      \
                                    int *lda, TYPE *e, TYPE *tau, TYPE *w,     \
                                    int *ldw)                                   \
{                                                                               \
    size_t idx;                                                                 \
    g_##SUFFIX##_fortran_call.called += 1;                                      \
    g_##SUFFIX##_fortran_call.uplo = *uplo;                                     \
    g_##SUFFIX##_fortran_call.n = *n;                                           \
    g_##SUFFIX##_fortran_call.nb = *nb;                                         \
    g_##SUFFIX##_fortran_call.lda = *lda;                                       \
    g_##SUFFIX##_fortran_call.ldw = *ldw;                                       \
    g_##SUFFIX##_fortran_call.a = a;                                            \
    g_##SUFFIX##_fortran_call.w = w;                                            \
    for (idx = 0; idx < 9; ++idx) {                                             \
        g_##SUFFIX##_fortran_call.a_snapshot[idx] = a[idx];                     \
    }                                                                           \
    a[0] = (TYPE)((BASE) + 10);                                                 \
    a[5] = (TYPE)((BASE) + 11);                                                 \
    w[0] = (TYPE)((BASE) + 20);                                                 \
    w[1] = (TYPE)((BASE) + 21);                                                 \
    w[3] = (TYPE)((BASE) + 22);                                                 \
    w[5] = (TYPE)((BASE) + 23);                                                 \
    e[0] = (TYPE)((BASE) + 30);                                                 \
    e[1] = (TYPE)((BASE) + 31);                                                 \
    tau[0] = (TYPE)((BASE) + 40);                                               \
    tau[1] = (TYPE)((BASE) + 41);                                               \
}                                                                               \
static int stub_##SUFFIX##_cblas(fb_layout_t layout, char uplo, int n, int nb, \
                                 TYPE *a, int lda, TYPE *e, TYPE *tau, TYPE *w, \
                                 int ldw)                                       \
{                                                                               \
    g_##SUFFIX##_cblas_call.called += 1;                                        \
    g_##SUFFIX##_cblas_call.layout = layout;                                    \
    g_##SUFFIX##_cblas_call.uplo = uplo;                                        \
    g_##SUFFIX##_cblas_call.n = n;                                              \
    g_##SUFFIX##_cblas_call.nb = nb;                                            \
    g_##SUFFIX##_cblas_call.lda = lda;                                          \
    g_##SUFFIX##_cblas_call.ldw = ldw;                                          \
    g_##SUFFIX##_cblas_call.a = a;                                              \
    g_##SUFFIX##_cblas_call.e = e;                                              \
    g_##SUFFIX##_cblas_call.tau = tau;                                          \
    g_##SUFFIX##_cblas_call.w = w;                                              \
    a[0] = (TYPE)((BASE) + 300);                                                \
    e[0] = (TYPE)((BASE) + 330);                                                \
    tau[0] = (TYPE)((BASE) + 340);                                              \
    w[0] = (TYPE)((BASE) + 320);                                                \
    return (BASE) + 500;                                                        \
}                                                                               \
static int check_##SUFFIX##_fortran_to_cblas(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                        \
    TYPE a[9] = { (TYPE)1, (TYPE)2, (TYPE)3, (TYPE)4, (TYPE)5, (TYPE)6,        \
                  (TYPE)7, (TYPE)8, (TYPE)9 };                                 \
    TYPE e[3] = { (TYPE)-1, (TYPE)-2, (TYPE)-3 };                              \
    TYPE tau[3] = { (TYPE)-4, (TYPE)-5, (TYPE)-6 };                            \
    TYPE w[6] = { (TYPE)0, (TYPE)0, (TYPE)0, (TYPE)0, (TYPE)0, (TYPE)0 };     \
    int info = 0;                                                               \
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
    info = thunk(FB_LAYOUT_ROW_MAJOR, 'U', 3, 2, a, 3, e, tau, w, 2);          \
    if (info != 0 || g_##SUFFIX##_fortran_call.called != 1 ||                  \
        g_##SUFFIX##_fortran_call.uplo != 'L' ||                                \
        g_##SUFFIX##_fortran_call.n != 3 ||                                     \
        g_##SUFFIX##_fortran_call.nb != 2 ||                                    \
        g_##SUFFIX##_fortran_call.lda != 3 ||                                   \
        g_##SUFFIX##_fortran_call.ldw != 3 ||                                   \
        g_##SUFFIX##_fortran_call.a == a ||                                     \
        g_##SUFFIX##_fortran_call.w == w ||                                     \
        g_##SUFFIX##_fortran_call.a_snapshot[0] != (TYPE)1 ||                  \
        g_##SUFFIX##_fortran_call.a_snapshot[1] != (TYPE)4 ||                  \
        g_##SUFFIX##_fortran_call.a_snapshot[2] != (TYPE)7 ||                  \
        g_##SUFFIX##_fortran_call.a_snapshot[3] != (TYPE)2 ||                  \
        g_##SUFFIX##_fortran_call.a_snapshot[5] != (TYPE)8 ||                  \
        a[0] != (TYPE)((BASE) + 10) || a[7] != (TYPE)((BASE) + 11) ||           \
        w[0] != (TYPE)((BASE) + 20) || w[1] != (TYPE)((BASE) + 22) ||           \
        w[2] != (TYPE)((BASE) + 21) || w[3] != (TYPE)0 ||                       \
        w[4] != (TYPE)0 || w[5] != (TYPE)((BASE) + 23) ||                       \
        e[0] != (TYPE)((BASE) + 30) || e[1] != (TYPE)((BASE) + 31) ||           \
        e[2] != (TYPE)-3 || tau[0] != (TYPE)((BASE) + 40) ||                    \
        tau[1] != (TYPE)((BASE) + 41) || tau[2] != (TYPE)-6) {                  \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not transpose row-major A/W buffers and copy back LATRD outputs correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk flips UPLO, uses col-major temporaries, and copies A/W/E/TAU back\n"); \
    return 0;                                                                   \
}                                                                               \
static int check_##SUFFIX##_cblas_to_fortran(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                      \
    char uplo = 'l';                                                            \
    int n = 3;                                                                  \
    int nb = 2;                                                                 \
    int lda = 3;                                                                \
    int ldw = 3;                                                                \
    TYPE a[9] = { (TYPE)0, (TYPE)0, (TYPE)0, (TYPE)0, (TYPE)0, (TYPE)0,        \
                  (TYPE)0, (TYPE)0, (TYPE)0 };                                 \
    TYPE e[3] = { (TYPE)0, (TYPE)0, (TYPE)0 };                                 \
    TYPE tau[3] = { (TYPE)0, (TYPE)0, (TYPE)0 };                               \
    TYPE w[9] = { (TYPE)0, (TYPE)0, (TYPE)0, (TYPE)0, (TYPE)0, (TYPE)0,        \
                  (TYPE)0, (TYPE)0, (TYPE)0 };                                 \
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
    thunk(&uplo, &n, &nb, a, &lda, e, tau, w, &ldw);                           \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                  \
        g_##SUFFIX##_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||                \
        g_##SUFFIX##_cblas_call.uplo != 'L' ||                                  \
        g_##SUFFIX##_cblas_call.n != 3 ||                                       \
        g_##SUFFIX##_cblas_call.nb != 2 ||                                      \
        g_##SUFFIX##_cblas_call.lda != 3 ||                                     \
        g_##SUFFIX##_cblas_call.ldw != 3 ||                                     \
        g_##SUFFIX##_cblas_call.a != a ||                                       \
        g_##SUFFIX##_cblas_call.e != e ||                                       \
        g_##SUFFIX##_cblas_call.tau != tau ||                                   \
        g_##SUFFIX##_cblas_call.w != w ||                                       \
        a[0] != (TYPE)((BASE) + 300) || e[0] != (TYPE)((BASE) + 330) ||         \
        tau[0] != (TYPE)((BASE) + 340) || w[0] != (TYPE)((BASE) + 320)) {      \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not delegate the all-pointer LATRD ABI into the LAPACKE-style entry correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk normalizes UPLO and forwards A/W/E/TAU directly into the LAPACKE-style entry\n"); \
    return 0;                                                                   \
}

#define DEFINE_COMPLEX_LATRD_TESTS(SUFFIX, CTYPE, REALTYPE, OP_ID, MAKE_FN, EQ_FN, BASE) \
typedef int (*fb_##SUFFIX##_cblas_fn)(fb_layout_t layout, char uplo, int n,    \
                                      int nb, CTYPE *a, int lda, REALTYPE *e,  \
                                      CTYPE *tau, CTYPE *w, int ldw);          \
typedef void (*fb_##SUFFIX##_fortran_fn)(char *uplo, int *n, int *nb, CTYPE *a, \
                                         int *lda, REALTYPE *e, CTYPE *tau,    \
                                         CTYPE *w, int *ldw);                  \
static struct {                                                                 \
    int called;                                                                 \
    char uplo;                                                                  \
    int n;                                                                      \
    int nb;                                                                     \
    int lda;                                                                    \
    int ldw;                                                                    \
    CTYPE *a;                                                                   \
    CTYPE *w;                                                                   \
    CTYPE a_snapshot[9];                                                        \
} g_##SUFFIX##_fortran_call;                                                    \
static struct {                                                                 \
    int called;                                                                 \
    fb_layout_t layout;                                                         \
    char uplo;                                                                  \
    int n;                                                                      \
    int nb;                                                                     \
    int lda;                                                                    \
    int ldw;                                                                    \
    CTYPE *a;                                                                   \
    REALTYPE *e;                                                                \
    CTYPE *tau;                                                                 \
    CTYPE *w;                                                                   \
} g_##SUFFIX##_cblas_call;                                                      \
static void stub_##SUFFIX##_fortran(char *uplo, int *n, int *nb, CTYPE *a,     \
                                    int *lda, REALTYPE *e, CTYPE *tau, CTYPE *w, \
                                    int *ldw)                                   \
{                                                                               \
    size_t idx;                                                                 \
    g_##SUFFIX##_fortran_call.called += 1;                                      \
    g_##SUFFIX##_fortran_call.uplo = *uplo;                                     \
    g_##SUFFIX##_fortran_call.n = *n;                                           \
    g_##SUFFIX##_fortran_call.nb = *nb;                                         \
    g_##SUFFIX##_fortran_call.lda = *lda;                                       \
    g_##SUFFIX##_fortran_call.ldw = *ldw;                                       \
    g_##SUFFIX##_fortran_call.a = a;                                            \
    g_##SUFFIX##_fortran_call.w = w;                                            \
    for (idx = 0; idx < 9; ++idx) {                                             \
        g_##SUFFIX##_fortran_call.a_snapshot[idx] = a[idx];                     \
    }                                                                           \
    a[0] = MAKE_FN((REALTYPE)((BASE) + 10), (REALTYPE)((BASE) + 60));          \
    a[5] = MAKE_FN((REALTYPE)((BASE) + 11), (REALTYPE)((BASE) + 61));          \
    w[0] = MAKE_FN((REALTYPE)((BASE) + 20), (REALTYPE)((BASE) + 70));          \
    w[1] = MAKE_FN((REALTYPE)((BASE) + 21), (REALTYPE)((BASE) + 71));          \
    w[3] = MAKE_FN((REALTYPE)((BASE) + 22), (REALTYPE)((BASE) + 72));          \
    w[5] = MAKE_FN((REALTYPE)((BASE) + 23), (REALTYPE)((BASE) + 73));          \
    e[0] = (REALTYPE)((BASE) + 30);                                             \
    e[1] = (REALTYPE)((BASE) + 31);                                             \
    tau[0] = MAKE_FN((REALTYPE)((BASE) + 40), (REALTYPE)((BASE) + 80));        \
    tau[1] = MAKE_FN((REALTYPE)((BASE) + 41), (REALTYPE)((BASE) + 81));        \
}                                                                               \
static int stub_##SUFFIX##_cblas(fb_layout_t layout, char uplo, int n, int nb, \
                                 CTYPE *a, int lda, REALTYPE *e, CTYPE *tau,   \
                                 CTYPE *w, int ldw)                             \
{                                                                               \
    g_##SUFFIX##_cblas_call.called += 1;                                        \
    g_##SUFFIX##_cblas_call.layout = layout;                                    \
    g_##SUFFIX##_cblas_call.uplo = uplo;                                        \
    g_##SUFFIX##_cblas_call.n = n;                                              \
    g_##SUFFIX##_cblas_call.nb = nb;                                            \
    g_##SUFFIX##_cblas_call.lda = lda;                                          \
    g_##SUFFIX##_cblas_call.ldw = ldw;                                          \
    g_##SUFFIX##_cblas_call.a = a;                                              \
    g_##SUFFIX##_cblas_call.e = e;                                              \
    g_##SUFFIX##_cblas_call.tau = tau;                                          \
    g_##SUFFIX##_cblas_call.w = w;                                              \
    a[0] = MAKE_FN((REALTYPE)((BASE) + 300), (REALTYPE)((BASE) + 350));        \
    e[0] = (REALTYPE)((BASE) + 330);                                            \
    tau[0] = MAKE_FN((REALTYPE)((BASE) + 340), (REALTYPE)((BASE) + 390));      \
    w[0] = MAKE_FN((REALTYPE)((BASE) + 320), (REALTYPE)((BASE) + 370));        \
    return (BASE) + 500;                                                        \
}                                                                               \
static int check_##SUFFIX##_fortran_to_cblas(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                        \
    CTYPE a[9];                                                                 \
    REALTYPE e[3] = { (REALTYPE)-1, (REALTYPE)-2, (REALTYPE)-3 };              \
    CTYPE tau[3];                                                               \
    CTYPE w[6];                                                                 \
    int info = 0;                                                               \
    a[0] = MAKE_FN((REALTYPE)1, (REALTYPE)11);                                  \
    a[1] = MAKE_FN((REALTYPE)2, (REALTYPE)12);                                  \
    a[2] = MAKE_FN((REALTYPE)3, (REALTYPE)13);                                  \
    a[3] = MAKE_FN((REALTYPE)4, (REALTYPE)14);                                  \
    a[4] = MAKE_FN((REALTYPE)5, (REALTYPE)15);                                  \
    a[5] = MAKE_FN((REALTYPE)6, (REALTYPE)16);                                  \
    a[6] = MAKE_FN((REALTYPE)7, (REALTYPE)17);                                  \
    a[7] = MAKE_FN((REALTYPE)8, (REALTYPE)18);                                  \
    a[8] = MAKE_FN((REALTYPE)9, (REALTYPE)19);                                  \
    tau[0] = MAKE_FN((REALTYPE)-4, (REALTYPE)-14);                              \
    tau[1] = MAKE_FN((REALTYPE)-5, (REALTYPE)-15);                              \
    tau[2] = MAKE_FN((REALTYPE)-6, (REALTYPE)-16);                              \
    w[0] = MAKE_FN((REALTYPE)0, (REALTYPE)0);                                   \
    w[1] = MAKE_FN((REALTYPE)0, (REALTYPE)0);                                   \
    w[2] = MAKE_FN((REALTYPE)0, (REALTYPE)0);                                   \
    w[3] = MAKE_FN((REALTYPE)0, (REALTYPE)0);                                   \
    w[4] = MAKE_FN((REALTYPE)0, (REALTYPE)0);                                   \
    w[5] = MAKE_FN((REALTYPE)0, (REALTYPE)0);                                   \
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
    info = thunk(FB_LAYOUT_ROW_MAJOR, 'U', 3, 2, a, 3, e, tau, w, 2);          \
    if (info != 0 || g_##SUFFIX##_fortran_call.called != 1 ||                  \
        g_##SUFFIX##_fortran_call.uplo != 'L' ||                                \
        g_##SUFFIX##_fortran_call.n != 3 ||                                     \
        g_##SUFFIX##_fortran_call.nb != 2 ||                                    \
        g_##SUFFIX##_fortran_call.lda != 3 ||                                   \
        g_##SUFFIX##_fortran_call.ldw != 3 ||                                   \
        g_##SUFFIX##_fortran_call.a == a ||                                     \
        g_##SUFFIX##_fortran_call.w == w ||                                     \
        !EQ_FN(g_##SUFFIX##_fortran_call.a_snapshot[0],                         \
               MAKE_FN((REALTYPE)1, (REALTYPE)11)) ||                           \
        !EQ_FN(g_##SUFFIX##_fortran_call.a_snapshot[1],                         \
               MAKE_FN((REALTYPE)4, (REALTYPE)14)) ||                           \
        !EQ_FN(g_##SUFFIX##_fortran_call.a_snapshot[2],                         \
               MAKE_FN((REALTYPE)7, (REALTYPE)17)) ||                           \
        !EQ_FN(g_##SUFFIX##_fortran_call.a_snapshot[3],                         \
               MAKE_FN((REALTYPE)2, (REALTYPE)12)) ||                           \
        !EQ_FN(g_##SUFFIX##_fortran_call.a_snapshot[5],                         \
               MAKE_FN((REALTYPE)8, (REALTYPE)18)) ||                           \
        !EQ_FN(a[0], MAKE_FN((REALTYPE)((BASE) + 10),                           \
                             (REALTYPE)((BASE) + 60))) ||                       \
        !EQ_FN(a[7], MAKE_FN((REALTYPE)((BASE) + 11),                           \
                             (REALTYPE)((BASE) + 61))) ||                       \
        !EQ_FN(w[0], MAKE_FN((REALTYPE)((BASE) + 20),                           \
                             (REALTYPE)((BASE) + 70))) ||                       \
        !EQ_FN(w[1], MAKE_FN((REALTYPE)((BASE) + 22),                           \
                             (REALTYPE)((BASE) + 72))) ||                       \
        !EQ_FN(w[2], MAKE_FN((REALTYPE)((BASE) + 21),                           \
                             (REALTYPE)((BASE) + 71))) ||                       \
        !EQ_FN(w[3], MAKE_FN((REALTYPE)0, (REALTYPE)0)) ||                      \
        !EQ_FN(w[4], MAKE_FN((REALTYPE)0, (REALTYPE)0)) ||                      \
        !EQ_FN(w[5], MAKE_FN((REALTYPE)((BASE) + 23),                           \
                             (REALTYPE)((BASE) + 73))) ||                       \
        e[0] != (REALTYPE)((BASE) + 30) || e[1] != (REALTYPE)((BASE) + 31) ||   \
        e[2] != (REALTYPE)-3 ||                                                  \
        !EQ_FN(tau[0], MAKE_FN((REALTYPE)((BASE) + 40),                         \
                               (REALTYPE)((BASE) + 80))) ||                     \
        !EQ_FN(tau[1], MAKE_FN((REALTYPE)((BASE) + 41),                         \
                               (REALTYPE)((BASE) + 81))) ||                     \
        !EQ_FN(tau[2], MAKE_FN((REALTYPE)-6, (REALTYPE)-16))) {                 \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not transpose row-major complex A/W buffers and copy back LATRD outputs correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk flips UPLO, uses col-major temporaries, and copies complex A/W/E/TAU back\n"); \
    return 0;                                                                   \
}                                                                               \
static int check_##SUFFIX##_cblas_to_fortran(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                      \
    char uplo = 'l';                                                            \
    int n = 3;                                                                  \
    int nb = 2;                                                                 \
    int lda = 3;                                                                \
    int ldw = 3;                                                                \
    CTYPE a[9];                                                                 \
    REALTYPE e[3] = { (REALTYPE)0, (REALTYPE)0, (REALTYPE)0 };                 \
    CTYPE tau[3];                                                               \
    CTYPE w[9];                                                                 \
    a[0] = MAKE_FN((REALTYPE)0, (REALTYPE)0);                                   \
    a[1] = MAKE_FN((REALTYPE)0, (REALTYPE)0);                                   \
    a[2] = MAKE_FN((REALTYPE)0, (REALTYPE)0);                                   \
    a[3] = MAKE_FN((REALTYPE)0, (REALTYPE)0);                                   \
    a[4] = MAKE_FN((REALTYPE)0, (REALTYPE)0);                                   \
    a[5] = MAKE_FN((REALTYPE)0, (REALTYPE)0);                                   \
    a[6] = MAKE_FN((REALTYPE)0, (REALTYPE)0);                                   \
    a[7] = MAKE_FN((REALTYPE)0, (REALTYPE)0);                                   \
    a[8] = MAKE_FN((REALTYPE)0, (REALTYPE)0);                                   \
    tau[0] = MAKE_FN((REALTYPE)0, (REALTYPE)0);                                 \
    tau[1] = MAKE_FN((REALTYPE)0, (REALTYPE)0);                                 \
    tau[2] = MAKE_FN((REALTYPE)0, (REALTYPE)0);                                 \
    w[0] = MAKE_FN((REALTYPE)0, (REALTYPE)0);                                   \
    w[1] = MAKE_FN((REALTYPE)0, (REALTYPE)0);                                   \
    w[2] = MAKE_FN((REALTYPE)0, (REALTYPE)0);                                   \
    w[3] = MAKE_FN((REALTYPE)0, (REALTYPE)0);                                   \
    w[4] = MAKE_FN((REALTYPE)0, (REALTYPE)0);                                   \
    w[5] = MAKE_FN((REALTYPE)0, (REALTYPE)0);                                   \
    w[6] = MAKE_FN((REALTYPE)0, (REALTYPE)0);                                   \
    w[7] = MAKE_FN((REALTYPE)0, (REALTYPE)0);                                   \
    w[8] = MAKE_FN((REALTYPE)0, (REALTYPE)0);                                   \
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
    thunk(&uplo, &n, &nb, a, &lda, e, tau, w, &ldw);                           \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                  \
        g_##SUFFIX##_cblas_call.layout != FB_LAYOUT_COL_MAJOR ||                \
        g_##SUFFIX##_cblas_call.uplo != 'L' ||                                  \
        g_##SUFFIX##_cblas_call.n != 3 ||                                       \
        g_##SUFFIX##_cblas_call.nb != 2 ||                                      \
        g_##SUFFIX##_cblas_call.lda != 3 ||                                     \
        g_##SUFFIX##_cblas_call.ldw != 3 ||                                     \
        g_##SUFFIX##_cblas_call.a != a ||                                       \
        g_##SUFFIX##_cblas_call.e != e ||                                       \
        g_##SUFFIX##_cblas_call.tau != tau ||                                   \
        g_##SUFFIX##_cblas_call.w != w ||                                       \
        !EQ_FN(a[0], MAKE_FN((REALTYPE)((BASE) + 300),                          \
                             (REALTYPE)((BASE) + 350))) ||                      \
        e[0] != (REALTYPE)((BASE) + 330) ||                                     \
        !EQ_FN(tau[0], MAKE_FN((REALTYPE)((BASE) + 340),                        \
                               (REALTYPE)((BASE) + 390))) ||                    \
        !EQ_FN(w[0], MAKE_FN((REALTYPE)((BASE) + 320),                          \
                             (REALTYPE)((BASE) + 370)))) {                      \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not delegate the all-pointer complex LATRD ABI into the LAPACKE-style entry correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk normalizes UPLO and forwards complex A/W/E/TAU directly into the LAPACKE-style entry\n"); \
    return 0;                                                                   \
}

DEFINE_REAL_LATRD_TESTS(slatrd, float, FB_OP_SLATRD, 100)
DEFINE_REAL_LATRD_TESTS(dlatrd, double, FB_OP_DLATRD, 300)
DEFINE_COMPLEX_LATRD_TESTS(clatrd, fb_complex_float_t, float, FB_OP_CLATRD,
                           make_cf32, cf32_eq, 500)
DEFINE_COMPLEX_LATRD_TESTS(zlatrd, fb_complex_double_t, double, FB_OP_ZLATRD,
                           make_cf64, cf64_eq, 700)

int main(void)
{
    int status = 0;

    status |= check_slatrd_fortran_to_cblas();
    status |= check_slatrd_cblas_to_fortran();
    status |= check_dlatrd_fortran_to_cblas();
    status |= check_dlatrd_cblas_to_fortran();
    status |= check_clatrd_fortran_to_cblas();
    status |= check_clatrd_cblas_to_fortran();
    status |= check_zlatrd_fortran_to_cblas();
    status |= check_zlatrd_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}