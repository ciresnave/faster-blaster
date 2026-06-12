#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

#define DEFINE_LAQR0_TESTS(SUFFIX, TYPE, OP_ID, BASE)                          \
typedef int (*fb_##SUFFIX##_cblas_fn)(int wantt, int wantz, int n, int ilo,    \
                                      int ihi, TYPE *h, int ldh, TYPE *wr,     \
                                      TYPE *wi, int iloz, int ihiz, TYPE *z,   \
                                      int ldz, TYPE *work, int lwork);         \
typedef void (*fb_##SUFFIX##_fortran_fn)(int *wantt, int *wantz, int *n,       \
                                         int *ilo, int *ihi, TYPE *h, int *ldh,\
                                         TYPE *wr, TYPE *wi, int *iloz,        \
                                         int *ihiz, TYPE *z, int *ldz,         \
                                         TYPE *work, int *lwork, int *info);   \
static struct {                                                                 \
    int called;                                                                 \
    int wantt;                                                                  \
    int wantz;                                                                  \
    int n;                                                                      \
    int ilo;                                                                    \
    int ihi;                                                                    \
    TYPE *h;                                                                    \
    int ldh;                                                                    \
    TYPE *wr;                                                                   \
    TYPE *wi;                                                                   \
    int iloz;                                                                   \
    int ihiz;                                                                   \
    TYPE *z;                                                                    \
    int ldz;                                                                    \
    TYPE *work;                                                                 \
    int lwork;                                                                  \
} g_##SUFFIX##_fortran_call;                                                    \
static struct {                                                                 \
    int called;                                                                 \
    int wantt;                                                                  \
    int wantz;                                                                  \
    int n;                                                                      \
    int ilo;                                                                    \
    int ihi;                                                                    \
    TYPE *h;                                                                    \
    int ldh;                                                                    \
    TYPE *wr;                                                                   \
    TYPE *wi;                                                                   \
    int iloz;                                                                   \
    int ihiz;                                                                   \
    TYPE *z;                                                                    \
    int ldz;                                                                    \
    TYPE *work;                                                                 \
    int lwork;                                                                  \
} g_##SUFFIX##_cblas_call;                                                      \
static void stub_##SUFFIX##_fortran(int *wantt, int *wantz, int *n, int *ilo,  \
                                    int *ihi, TYPE *h, int *ldh, TYPE *wr,    \
                                    TYPE *wi, int *iloz, int *ihiz, TYPE *z,  \
                                    int *ldz, TYPE *work, int *lwork,         \
                                    int *info)                                \
{                                                                               \
    g_##SUFFIX##_fortran_call.called += 1;                                      \
    g_##SUFFIX##_fortran_call.wantt = *wantt;                                   \
    g_##SUFFIX##_fortran_call.wantz = *wantz;                                   \
    g_##SUFFIX##_fortran_call.n = *n;                                           \
    g_##SUFFIX##_fortran_call.ilo = *ilo;                                       \
    g_##SUFFIX##_fortran_call.ihi = *ihi;                                       \
    g_##SUFFIX##_fortran_call.h = h;                                            \
    g_##SUFFIX##_fortran_call.ldh = *ldh;                                       \
    g_##SUFFIX##_fortran_call.wr = wr;                                          \
    g_##SUFFIX##_fortran_call.wi = wi;                                          \
    g_##SUFFIX##_fortran_call.iloz = *iloz;                                     \
    g_##SUFFIX##_fortran_call.ihiz = *ihiz;                                     \
    g_##SUFFIX##_fortran_call.z = z;                                            \
    g_##SUFFIX##_fortran_call.ldz = *ldz;                                       \
    g_##SUFFIX##_fortran_call.work = work;                                      \
    g_##SUFFIX##_fortran_call.lwork = *lwork;                                   \
    wr[0] = (TYPE)((BASE) + 1);                                                 \
    wi[0] = (TYPE)((BASE) + 2);                                                 \
    if (*wantz != 0) {                                                          \
        z[0] = (TYPE)((BASE) + 3);                                              \
    }                                                                           \
    work[0] = (TYPE)((BASE) + 4);                                               \
    *info = (BASE) + 5;                                                         \
}                                                                               \
static int stub_##SUFFIX##_cblas(int wantt, int wantz, int n, int ilo, int ihi,\
                                 TYPE *h, int ldh, TYPE *wr, TYPE *wi,        \
                                 int iloz, int ihiz, TYPE *z, int ldz,        \
                                 TYPE *work, int lwork)                       \
{                                                                               \
    g_##SUFFIX##_cblas_call.called += 1;                                        \
    g_##SUFFIX##_cblas_call.wantt = wantt;                                      \
    g_##SUFFIX##_cblas_call.wantz = wantz;                                      \
    g_##SUFFIX##_cblas_call.n = n;                                              \
    g_##SUFFIX##_cblas_call.ilo = ilo;                                          \
    g_##SUFFIX##_cblas_call.ihi = ihi;                                          \
    g_##SUFFIX##_cblas_call.h = h;                                              \
    g_##SUFFIX##_cblas_call.ldh = ldh;                                          \
    g_##SUFFIX##_cblas_call.wr = wr;                                            \
    g_##SUFFIX##_cblas_call.wi = wi;                                            \
    g_##SUFFIX##_cblas_call.iloz = iloz;                                        \
    g_##SUFFIX##_cblas_call.ihiz = ihiz;                                        \
    g_##SUFFIX##_cblas_call.z = z;                                              \
    g_##SUFFIX##_cblas_call.ldz = ldz;                                          \
    g_##SUFFIX##_cblas_call.work = work;                                        \
    g_##SUFFIX##_cblas_call.lwork = lwork;                                      \
    wr[0] = (TYPE)((BASE) + 6);                                                 \
    wi[0] = (TYPE)((BASE) + 7);                                                 \
    if (wantz != 0) {                                                           \
        z[0] = (TYPE)((BASE) + 8);                                              \
    }                                                                           \
    work[0] = (TYPE)((BASE) + 9);                                               \
    return (BASE) + 10;                                                         \
}                                                                               \
static int check_##SUFFIX##_fortran_to_cblas(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                        \
    TYPE h[4] = { (TYPE)1, (TYPE)2, (TYPE)3, (TYPE)4 };                        \
    TYPE wr[2] = { (TYPE)0, (TYPE)0 };                                          \
    TYPE wi[2] = { (TYPE)0, (TYPE)0 };                                          \
    TYPE z[4] = { (TYPE)11, (TYPE)12, (TYPE)13, (TYPE)14 };                    \
    TYPE work[4] = { (TYPE)0, (TYPE)0, (TYPE)0, (TYPE)0 };                     \
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
    if (thunk(1, 1, 2, 1, 2, h, 2, wr, wi, 1, 2, z, 2, work, 4) != (BASE) + 5 ||\
        g_##SUFFIX##_fortran_call.called != 1 ||                                \
        g_##SUFFIX##_fortran_call.wantt != 1 ||                                 \
        g_##SUFFIX##_fortran_call.wantz != 1 ||                                 \
        g_##SUFFIX##_fortran_call.n != 2 ||                                     \
        g_##SUFFIX##_fortran_call.ilo != 1 ||                                   \
        g_##SUFFIX##_fortran_call.ihi != 2 ||                                   \
        g_##SUFFIX##_fortran_call.h != h ||                                     \
        g_##SUFFIX##_fortran_call.ldh != 2 ||                                   \
        g_##SUFFIX##_fortran_call.wr != wr ||                                   \
        g_##SUFFIX##_fortran_call.wi != wi ||                                   \
        g_##SUFFIX##_fortran_call.iloz != 1 ||                                  \
        g_##SUFFIX##_fortran_call.ihiz != 2 ||                                  \
        g_##SUFFIX##_fortran_call.z != z ||                                     \
        g_##SUFFIX##_fortran_call.ldz != 2 ||                                   \
        g_##SUFFIX##_fortran_call.work != work ||                               \
        g_##SUFFIX##_fortran_call.lwork != 4 ||                                 \
        wr[0] != (TYPE)((BASE) + 1) ||                                          \
        wi[0] != (TYPE)((BASE) + 2) ||                                          \
        z[0] != (TYPE)((BASE) + 3) ||                                           \
        work[0] != (TYPE)((BASE) + 4)) {                                        \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward LAQR0 inputs or return info correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards LAQR0 inputs and returns info\n"); \
    return 0;                                                                   \
}                                                                               \
static int check_##SUFFIX##_cblas_to_fortran(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                      \
    int wantt = 1;                                                              \
    int wantz = 1;                                                              \
    int n = 2;                                                                  \
    int ilo = 1;                                                                \
    int ihi = 2;                                                                \
    int ldh = 2;                                                                \
    int iloz = 1;                                                               \
    int ihiz = 2;                                                               \
    int ldz = 2;                                                                \
    int lwork = 4;                                                              \
    int info = -1;                                                              \
    TYPE h[4] = { (TYPE)21, (TYPE)22, (TYPE)23, (TYPE)24 };                    \
    TYPE wr[2] = { (TYPE)0, (TYPE)0 };                                          \
    TYPE wi[2] = { (TYPE)0, (TYPE)0 };                                          \
    TYPE z[4] = { (TYPE)31, (TYPE)32, (TYPE)33, (TYPE)34 };                    \
    TYPE work[4] = { (TYPE)0, (TYPE)0, (TYPE)0, (TYPE)0 };                     \
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
    thunk(&wantt, &wantz, &n, &ilo, &ihi, h, &ldh, wr, wi, &iloz, &ihiz, z,    \
          &ldz, work, &lwork, &info);                                           \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                  \
        g_##SUFFIX##_cblas_call.wantt != 1 ||                                   \
        g_##SUFFIX##_cblas_call.wantz != 1 ||                                   \
        g_##SUFFIX##_cblas_call.n != 2 ||                                       \
        g_##SUFFIX##_cblas_call.ilo != 1 ||                                     \
        g_##SUFFIX##_cblas_call.ihi != 2 ||                                     \
        g_##SUFFIX##_cblas_call.h != h ||                                       \
        g_##SUFFIX##_cblas_call.ldh != 2 ||                                     \
        g_##SUFFIX##_cblas_call.wr != wr ||                                     \
        g_##SUFFIX##_cblas_call.wi != wi ||                                     \
        g_##SUFFIX##_cblas_call.iloz != 1 ||                                    \
        g_##SUFFIX##_cblas_call.ihiz != 2 ||                                    \
        g_##SUFFIX##_cblas_call.z != z ||                                       \
        g_##SUFFIX##_cblas_call.ldz != 2 ||                                     \
        g_##SUFFIX##_cblas_call.work != work ||                                 \
        g_##SUFFIX##_cblas_call.lwork != 4 ||                                   \
        info != (BASE) + 10 ||                                                  \
        wr[0] != (TYPE)((BASE) + 6) ||                                          \
        wi[0] != (TYPE)((BASE) + 7) ||                                          \
        z[0] != (TYPE)((BASE) + 8) ||                                           \
        work[0] != (TYPE)((BASE) + 9)) {                                        \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference LAQR0 inputs or store info correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences LAQR0 inputs and stores info\n"); \
    return 0;                                                                   \
}

DEFINE_LAQR0_TESTS(slaqr0, float, FB_OP_SLAQR0, 100)
DEFINE_LAQR0_TESTS(dlaqr0, double, FB_OP_DLAQR0, 300)

int main(void)
{
    int status = 0;

    status |= check_slaqr0_fortran_to_cblas();
    status |= check_slaqr0_cblas_to_fortran();
    status |= check_dlaqr0_fortran_to_cblas();
    status |= check_dlaqr0_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}