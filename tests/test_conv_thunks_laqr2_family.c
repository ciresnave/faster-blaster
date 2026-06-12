#include <stdio.h>
#include <string.h>

#include "../src/core/conv_thunks.h"

#define DEFINE_LAQR2_TESTS(SUFFIX, TYPE, OP_ID, BASE)                          \
typedef int (*fb_##SUFFIX##_cblas_fn)(int wantt, int wantz, int n, int ktop,   \
                                      int kbot, int nw, TYPE *h, int ldh,      \
                                      int iloz, int ihiz, TYPE *z, int ldz,    \
                                      int *ns, int *nd, TYPE *sr, TYPE *si,    \
                                      TYPE *v, int ldv, int nh, TYPE *t,       \
                                      int ldt, int nv, TYPE *wv, int ldwv,     \
                                      TYPE *work, int lwork);                  \
typedef void (*fb_##SUFFIX##_fortran_fn)(int *wantt, int *wantz, int *n,       \
                                         int *ktop, int *kbot, int *nw,       \
                                         TYPE *h, int *ldh, int *iloz,        \
                                         int *ihiz, TYPE *z, int *ldz,        \
                                         int *ns, int *nd, TYPE *sr, TYPE *si, \
                                         TYPE *v, int *ldv, int *nh, TYPE *t, \
                                         int *ldt, int *nv, TYPE *wv,         \
                                         int *ldwv, TYPE *work, int *lwork);  \
static struct {                                                                 \
    int called;                                                                 \
    int wantt;                                                                  \
    int wantz;                                                                  \
    int n;                                                                      \
    int ktop;                                                                   \
    int kbot;                                                                   \
    int nw;                                                                     \
    TYPE *h;                                                                    \
    int ldh;                                                                    \
    int iloz;                                                                   \
    int ihiz;                                                                   \
    TYPE *z;                                                                    \
    int ldz;                                                                    \
    int *ns;                                                                    \
    int *nd;                                                                    \
    TYPE *sr;                                                                   \
    TYPE *si;                                                                   \
    TYPE *v;                                                                    \
    int ldv;                                                                    \
    int nh;                                                                     \
    TYPE *t;                                                                    \
    int ldt;                                                                    \
    int nv;                                                                     \
    TYPE *wv;                                                                   \
    int ldwv;                                                                   \
    TYPE *work;                                                                 \
    int lwork;                                                                  \
} g_##SUFFIX##_fortran_call;                                                    \
static struct {                                                                 \
    int called;                                                                 \
    int wantt;                                                                  \
    int wantz;                                                                  \
    int n;                                                                      \
    int ktop;                                                                   \
    int kbot;                                                                   \
    int nw;                                                                     \
    TYPE *h;                                                                    \
    int ldh;                                                                    \
    int iloz;                                                                   \
    int ihiz;                                                                   \
    TYPE *z;                                                                    \
    int ldz;                                                                    \
    int *ns;                                                                    \
    int *nd;                                                                    \
    TYPE *sr;                                                                   \
    TYPE *si;                                                                   \
    TYPE *v;                                                                    \
    int ldv;                                                                    \
    int nh;                                                                     \
    TYPE *t;                                                                    \
    int ldt;                                                                    \
    int nv;                                                                     \
    TYPE *wv;                                                                   \
    int ldwv;                                                                   \
    TYPE *work;                                                                 \
    int lwork;                                                                  \
} g_##SUFFIX##_cblas_call;                                                      \
static void stub_##SUFFIX##_fortran(int *wantt, int *wantz, int *n,            \
                                    int *ktop, int *kbot, int *nw, TYPE *h,   \
                                    int *ldh, int *iloz, int *ihiz, TYPE *z,  \
                                    int *ldz, int *ns, int *nd, TYPE *sr,     \
                                    TYPE *si, TYPE *v, int *ldv, int *nh,     \
                                    TYPE *t, int *ldt, int *nv, TYPE *wv,     \
                                    int *ldwv, TYPE *work, int *lwork)        \
{                                                                               \
    g_##SUFFIX##_fortran_call.called += 1;                                      \
    g_##SUFFIX##_fortran_call.wantt = *wantt;                                   \
    g_##SUFFIX##_fortran_call.wantz = *wantz;                                   \
    g_##SUFFIX##_fortran_call.n = *n;                                           \
    g_##SUFFIX##_fortran_call.ktop = *ktop;                                     \
    g_##SUFFIX##_fortran_call.kbot = *kbot;                                     \
    g_##SUFFIX##_fortran_call.nw = *nw;                                         \
    g_##SUFFIX##_fortran_call.h = h;                                            \
    g_##SUFFIX##_fortran_call.ldh = *ldh;                                       \
    g_##SUFFIX##_fortran_call.iloz = *iloz;                                     \
    g_##SUFFIX##_fortran_call.ihiz = *ihiz;                                     \
    g_##SUFFIX##_fortran_call.z = z;                                            \
    g_##SUFFIX##_fortran_call.ldz = *ldz;                                       \
    g_##SUFFIX##_fortran_call.ns = ns;                                          \
    g_##SUFFIX##_fortran_call.nd = nd;                                          \
    g_##SUFFIX##_fortran_call.sr = sr;                                          \
    g_##SUFFIX##_fortran_call.si = si;                                          \
    g_##SUFFIX##_fortran_call.v = v;                                            \
    g_##SUFFIX##_fortran_call.ldv = *ldv;                                       \
    g_##SUFFIX##_fortran_call.nh = *nh;                                         \
    g_##SUFFIX##_fortran_call.t = t;                                            \
    g_##SUFFIX##_fortran_call.ldt = *ldt;                                       \
    g_##SUFFIX##_fortran_call.nv = *nv;                                         \
    g_##SUFFIX##_fortran_call.wv = wv;                                          \
    g_##SUFFIX##_fortran_call.ldwv = *ldwv;                                     \
    g_##SUFFIX##_fortran_call.work = work;                                      \
    g_##SUFFIX##_fortran_call.lwork = *lwork;                                   \
    *ns = (BASE) + 1;                                                           \
    *nd = (BASE) + 2;                                                           \
    sr[0] = (TYPE)((BASE) + 3);                                                 \
    si[0] = (TYPE)((BASE) + 4);                                                 \
    v[0] = (TYPE)((BASE) + 5);                                                  \
    t[0] = (TYPE)((BASE) + 6);                                                  \
    wv[0] = (TYPE)((BASE) + 7);                                                 \
    z[0] = (TYPE)((BASE) + 8);                                                  \
    work[0] = (TYPE)((BASE) + 9);                                               \
}                                                                               \
static int stub_##SUFFIX##_cblas(int wantt, int wantz, int n, int ktop,        \
                                 int kbot, int nw, TYPE *h, int ldh, int iloz, \
                                 int ihiz, TYPE *z, int ldz, int *ns, int *nd, \
                                 TYPE *sr, TYPE *si, TYPE *v, int ldv, int nh, \
                                 TYPE *t, int ldt, int nv, TYPE *wv, int ldwv, \
                                 TYPE *work, int lwork)                        \
{                                                                               \
    g_##SUFFIX##_cblas_call.called += 1;                                        \
    g_##SUFFIX##_cblas_call.wantt = wantt;                                      \
    g_##SUFFIX##_cblas_call.wantz = wantz;                                      \
    g_##SUFFIX##_cblas_call.n = n;                                              \
    g_##SUFFIX##_cblas_call.ktop = ktop;                                        \
    g_##SUFFIX##_cblas_call.kbot = kbot;                                        \
    g_##SUFFIX##_cblas_call.nw = nw;                                            \
    g_##SUFFIX##_cblas_call.h = h;                                              \
    g_##SUFFIX##_cblas_call.ldh = ldh;                                          \
    g_##SUFFIX##_cblas_call.iloz = iloz;                                        \
    g_##SUFFIX##_cblas_call.ihiz = ihiz;                                        \
    g_##SUFFIX##_cblas_call.z = z;                                              \
    g_##SUFFIX##_cblas_call.ldz = ldz;                                          \
    g_##SUFFIX##_cblas_call.ns = ns;                                            \
    g_##SUFFIX##_cblas_call.nd = nd;                                            \
    g_##SUFFIX##_cblas_call.sr = sr;                                            \
    g_##SUFFIX##_cblas_call.si = si;                                            \
    g_##SUFFIX##_cblas_call.v = v;                                              \
    g_##SUFFIX##_cblas_call.ldv = ldv;                                          \
    g_##SUFFIX##_cblas_call.nh = nh;                                            \
    g_##SUFFIX##_cblas_call.t = t;                                              \
    g_##SUFFIX##_cblas_call.ldt = ldt;                                          \
    g_##SUFFIX##_cblas_call.nv = nv;                                            \
    g_##SUFFIX##_cblas_call.wv = wv;                                            \
    g_##SUFFIX##_cblas_call.ldwv = ldwv;                                        \
    g_##SUFFIX##_cblas_call.work = work;                                        \
    g_##SUFFIX##_cblas_call.lwork = lwork;                                      \
    *ns = (BASE) + 10;                                                          \
    *nd = (BASE) + 11;                                                          \
    sr[0] = (TYPE)((BASE) + 12);                                                \
    si[0] = (TYPE)((BASE) + 13);                                                \
    v[0] = (TYPE)((BASE) + 14);                                                 \
    t[0] = (TYPE)((BASE) + 15);                                                 \
    wv[0] = (TYPE)((BASE) + 16);                                                \
    z[0] = (TYPE)((BASE) + 17);                                                 \
    work[0] = (TYPE)((BASE) + 18);                                              \
    return (BASE) + 99;                                                         \
}                                                                               \
static int check_##SUFFIX##_fortran_to_cblas(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_cblas_fn thunk = NULL;                                        \
    int ns = -1;                                                                \
    int nd = -1;                                                                \
    TYPE h[4] = { (TYPE)1, (TYPE)2, (TYPE)3, (TYPE)4 };                        \
    TYPE z[4] = { (TYPE)21, (TYPE)22, (TYPE)23, (TYPE)24 };                    \
    TYPE sr[2] = { (TYPE)0, (TYPE)0 };                                          \
    TYPE si[2] = { (TYPE)0, (TYPE)0 };                                          \
    TYPE v[4] = { (TYPE)0, (TYPE)0, (TYPE)0, (TYPE)0 };                        \
    TYPE t[4] = { (TYPE)0, (TYPE)0, (TYPE)0, (TYPE)0 };                        \
    TYPE wv[4] = { (TYPE)0, (TYPE)0, (TYPE)0, (TYPE)0 };                       \
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
    if (thunk(1, 1, 2, 1, 2, 2, h, 2, 1, 2, z, 2, &ns, &nd, sr, si, v, 2, 2,   \
              t, 2, 2, wv, 2, work, 4) != 0 ||                                 \
        g_##SUFFIX##_fortran_call.called != 1 ||                                \
        g_##SUFFIX##_fortran_call.wantt != 1 ||                                 \
        g_##SUFFIX##_fortran_call.wantz != 1 ||                                 \
        g_##SUFFIX##_fortran_call.n != 2 ||                                     \
        g_##SUFFIX##_fortran_call.ktop != 1 ||                                  \
        g_##SUFFIX##_fortran_call.kbot != 2 ||                                  \
        g_##SUFFIX##_fortran_call.nw != 2 ||                                    \
        g_##SUFFIX##_fortran_call.h != h ||                                     \
        g_##SUFFIX##_fortran_call.ldh != 2 ||                                   \
        g_##SUFFIX##_fortran_call.iloz != 1 ||                                  \
        g_##SUFFIX##_fortran_call.ihiz != 2 ||                                  \
        g_##SUFFIX##_fortran_call.z != z ||                                     \
        g_##SUFFIX##_fortran_call.ldz != 2 ||                                   \
        g_##SUFFIX##_fortran_call.ns != &ns ||                                  \
        g_##SUFFIX##_fortran_call.nd != &nd ||                                  \
        g_##SUFFIX##_fortran_call.sr != sr ||                                   \
        g_##SUFFIX##_fortran_call.si != si ||                                   \
        g_##SUFFIX##_fortran_call.v != v ||                                     \
        g_##SUFFIX##_fortran_call.ldv != 2 ||                                   \
        g_##SUFFIX##_fortran_call.nh != 2 ||                                    \
        g_##SUFFIX##_fortran_call.t != t ||                                     \
        g_##SUFFIX##_fortran_call.ldt != 2 ||                                   \
        g_##SUFFIX##_fortran_call.nv != 2 ||                                    \
        g_##SUFFIX##_fortran_call.wv != wv ||                                   \
        g_##SUFFIX##_fortran_call.ldwv != 2 ||                                  \
        g_##SUFFIX##_fortran_call.work != work ||                               \
        g_##SUFFIX##_fortran_call.lwork != 4 ||                                 \
        ns != (BASE) + 1 || nd != (BASE) + 2 ||                                 \
        sr[0] != (TYPE)((BASE) + 3) || si[0] != (TYPE)((BASE) + 4) ||           \
        v[0] != (TYPE)((BASE) + 5) || t[0] != (TYPE)((BASE) + 6) ||            \
        wv[0] != (TYPE)((BASE) + 7) || z[0] != (TYPE)((BASE) + 8) ||           \
        work[0] != (TYPE)((BASE) + 9)) {                                        \
        fprintf(stderr, "[FAIL] " #SUFFIX " Fortran->CBLAS thunk did not forward LAQR2 inputs or outputs correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " Fortran->CBLAS thunk forwards LAQR2 inputs and returns success\n"); \
    return 0;                                                                   \
}                                                                               \
static int check_##SUFFIX##_cblas_to_fortran(void)                              \
{                                                                               \
    fb_backend_vtable_t vtable;                                                 \
    fb_##SUFFIX##_fortran_fn thunk = NULL;                                      \
    int wantt = 1;                                                              \
    int wantz = 1;                                                              \
    int n = 2;                                                                  \
    int ktop = 1;                                                               \
    int kbot = 2;                                                               \
    int nw = 2;                                                                 \
    int ldh = 2;                                                                \
    int iloz = 1;                                                               \
    int ihiz = 2;                                                               \
    int ldz = 2;                                                                \
    int ns = -1;                                                                \
    int nd = -1;                                                                \
    int ldv = 2;                                                                \
    int nh = 2;                                                                 \
    int ldt = 2;                                                                \
    int nv = 2;                                                                 \
    int ldwv = 2;                                                               \
    int lwork = 4;                                                              \
    TYPE h[4] = { (TYPE)11, (TYPE)12, (TYPE)13, (TYPE)14 };                    \
    TYPE z[4] = { (TYPE)31, (TYPE)32, (TYPE)33, (TYPE)34 };                    \
    TYPE sr[2] = { (TYPE)0, (TYPE)0 };                                          \
    TYPE si[2] = { (TYPE)0, (TYPE)0 };                                          \
    TYPE v[4] = { (TYPE)0, (TYPE)0, (TYPE)0, (TYPE)0 };                        \
    TYPE t[4] = { (TYPE)0, (TYPE)0, (TYPE)0, (TYPE)0 };                        \
    TYPE wv[4] = { (TYPE)0, (TYPE)0, (TYPE)0, (TYPE)0 };                       \
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
    thunk(&wantt, &wantz, &n, &ktop, &kbot, &nw, h, &ldh, &iloz, &ihiz, z,     \
          &ldz, &ns, &nd, sr, si, v, &ldv, &nh, t, &ldt, &nv, wv, &ldwv,       \
          work, &lwork);                                                        \
    if (g_##SUFFIX##_cblas_call.called != 1 ||                                  \
        g_##SUFFIX##_cblas_call.wantt != 1 ||                                   \
        g_##SUFFIX##_cblas_call.wantz != 1 ||                                   \
        g_##SUFFIX##_cblas_call.n != 2 ||                                       \
        g_##SUFFIX##_cblas_call.ktop != 1 ||                                    \
        g_##SUFFIX##_cblas_call.kbot != 2 ||                                    \
        g_##SUFFIX##_cblas_call.nw != 2 ||                                      \
        g_##SUFFIX##_cblas_call.h != h ||                                       \
        g_##SUFFIX##_cblas_call.ldh != 2 ||                                     \
        g_##SUFFIX##_cblas_call.iloz != 1 ||                                    \
        g_##SUFFIX##_cblas_call.ihiz != 2 ||                                    \
        g_##SUFFIX##_cblas_call.z != z ||                                       \
        g_##SUFFIX##_cblas_call.ldz != 2 ||                                     \
        g_##SUFFIX##_cblas_call.ns != &ns ||                                    \
        g_##SUFFIX##_cblas_call.nd != &nd ||                                    \
        g_##SUFFIX##_cblas_call.sr != sr ||                                     \
        g_##SUFFIX##_cblas_call.si != si ||                                     \
        g_##SUFFIX##_cblas_call.v != v ||                                       \
        g_##SUFFIX##_cblas_call.ldv != 2 ||                                     \
        g_##SUFFIX##_cblas_call.nh != 2 ||                                      \
        g_##SUFFIX##_cblas_call.t != t ||                                       \
        g_##SUFFIX##_cblas_call.ldt != 2 ||                                     \
        g_##SUFFIX##_cblas_call.nv != 2 ||                                      \
        g_##SUFFIX##_cblas_call.wv != wv ||                                     \
        g_##SUFFIX##_cblas_call.ldwv != 2 ||                                    \
        g_##SUFFIX##_cblas_call.work != work ||                                 \
        g_##SUFFIX##_cblas_call.lwork != 4 ||                                   \
        ns != (BASE) + 10 || nd != (BASE) + 11 ||                               \
        sr[0] != (TYPE)((BASE) + 12) || si[0] != (TYPE)((BASE) + 13) ||         \
        v[0] != (TYPE)((BASE) + 14) || t[0] != (TYPE)((BASE) + 15) ||          \
        wv[0] != (TYPE)((BASE) + 16) || z[0] != (TYPE)((BASE) + 17) ||         \
        work[0] != (TYPE)((BASE) + 18)) {                                       \
        fprintf(stderr, "[FAIL] " #SUFFIX " CBLAS->Fortran thunk did not dereference LAQR2 inputs or propagate outputs correctly\n"); \
        return 1;                                                               \
    }                                                                           \
    printf("[PASS] " #SUFFIX " CBLAS->Fortran thunk dereferences LAQR2 inputs and ignores the C int return\n"); \
    return 0;                                                                   \
}

DEFINE_LAQR2_TESTS(slaqr2, float, FB_OP_SLAQR2, 100)
DEFINE_LAQR2_TESTS(dlaqr2, double, FB_OP_DLAQR2, 300)

int main(void)
{
    int status = 0;

    status |= check_slaqr2_fortran_to_cblas();
    status |= check_slaqr2_cblas_to_fortran();
    status |= check_dlaqr2_fortran_to_cblas();
    status |= check_dlaqr2_cblas_to_fortran();

    if (status != 0) {
        fprintf(stderr, "Result: FAIL\n");
        return 1;
    }

    printf("Result: PASS\n");
    return 0;
}