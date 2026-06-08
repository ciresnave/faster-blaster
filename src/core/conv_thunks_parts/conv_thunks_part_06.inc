
#define DEFINE_LAPACK_LAHRD_COMPLEX_THUNKS(OP, CTYPE, OP_ID)                   \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const int n, const int k, const int nb, CTYPE *a, const int lda,           \
    CTYPE *tau, CTYPE *t, const int ldt, CTYPE *y, const int ldy) {            \
  int n_ = n;                                                                  \
  int k_ = k;                                                                  \
  int nb_ = nb;                                                                \
  int lda_ = lda;                                                              \
  int ldt_ = ldt;                                                              \
  int ldy_ = ldy;                                                              \
  if (!g_fortran_fn[OP_ID]) return -1;                                         \
  if (((n > 0 || nb > 0) && (!a || !tau || !t || !y))) return -1;             \
  ((void (*)(int *, int *, int *, CTYPE *, int *, CTYPE *, CTYPE *, int *,     \
             CTYPE *, int *))g_fortran_fn[OP_ID])(                             \
      &n_, &k_, &nb_, a, &lda_, tau, t, &ldt_, y, &ldy_);                      \
  return 0;                                                                    \
}                                                                              \
static void thunk_##OP##_cblas_to_fortran(                                     \
    int *n, int *k, int *nb, CTYPE *a, int *lda, CTYPE *tau, CTYPE *t,         \
    int *ldt, CTYPE *y, int *ldy) {                                            \
  if (!n || !k || !nb || !lda || !ldt || !ldy) return;                        \
  if (!g_cblas_fn[OP_ID]) return;                                              \
  if (((*n > 0 || *nb > 0) && (!a || !tau || !t || !y))) return;               \
  (void)((int (*)(int, int, int, CTYPE *, int, CTYPE *, CTYPE *, int, CTYPE *, \
                  int))g_cblas_fn[OP_ID])(                                     \
      *n, *k, *nb, a, *lda, tau, t, *ldt, y, *ldy);                            \
}

#define DEFINE_LAPACK_LAQR1_THUNKS(OP, TYPE, OP_ID)                            \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const int n, TYPE *h, const int ldh, const TYPE sr1, const TYPE si1,       \
    const TYPE sr2, const TYPE si2, TYPE *v) {                                 \
  int n_ = n;                                                                  \
  int ldh_ = ldh;                                                              \
  TYPE sr1_ = sr1;                                                             \
  TYPE si1_ = si1;                                                             \
  TYPE sr2_ = sr2;                                                             \
  TYPE si2_ = si2;                                                             \
  if (!g_fortran_fn[OP_ID]) return -1;                                         \
  if ((n > 0 && !h) || !v) return -1;                                          \
  ((void (*)(int *, TYPE *, int *, TYPE *, TYPE *, TYPE *, TYPE *, TYPE *))    \
       g_fortran_fn[OP_ID])(&n_, h, &ldh_, &sr1_, &si1_, &sr2_, &si2_, v);    \
  return 0;                                                                    \
}                                                                              \
static void thunk_##OP##_cblas_to_fortran(                                     \
    int *n, TYPE *h, int *ldh, TYPE *sr1, TYPE *si1, TYPE *sr2, TYPE *si2,     \
    TYPE *v) {                                                                 \
  if (!g_cblas_fn[OP_ID]) return;                                              \
  if (!n || !ldh || !sr1 || !si1 || !sr2 || !si2 || !v) return;                \
  if (*n > 0 && !h) return;                                                    \
  (void)((int (*)(int, TYPE *, int, TYPE, TYPE, TYPE, TYPE, TYPE *))           \
             g_cblas_fn[OP_ID])(*n, h, *ldh, *sr1, *si1, *sr2, *si2, v);       \
}

#define DEFINE_LAPACK_LAQR0_THUNKS(OP, TYPE, OP_ID)                            \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const int wantt, const int wantz, const int n, const int ilo,              \
    const int ihi, TYPE *h, const int ldh, TYPE *wr, TYPE *wi,                 \
    const int iloz, const int ihiz, TYPE *z, const int ldz, TYPE *work,        \
    const int lwork) {                                                         \
  int wantt_ = wantt;                                                          \
  int wantz_ = wantz;                                                          \
  int n_ = n;                                                                  \
  int ilo_ = ilo;                                                              \
  int ihi_ = ihi;                                                              \
  int ldh_ = ldh;                                                              \
  int iloz_ = iloz;                                                            \
  int ihiz_ = ihiz;                                                            \
  int ldz_ = ldz;                                                              \
  int lwork_ = lwork;                                                          \
  int info_ = 0;                                                               \
  if (!g_fortran_fn[OP_ID]) return -1;                                         \
  if ((n > 0 && (!h || !wr || !wi)) || !work) return -1;                       \
  if (wantz && n > 0 && !z) return -1;                                         \
  ((void (*)(int *, int *, int *, int *, int *, TYPE *, int *, TYPE *,         \
             TYPE *, int *, int *, TYPE *, int *, TYPE *, int *, int *))       \
       g_fortran_fn[OP_ID])(                                                   \
      &wantt_, &wantz_, &n_, &ilo_, &ihi_, h, &ldh_, wr, wi, &iloz_, &ihiz_,   \
      z, &ldz_, work, &lwork_, &info_);                                        \
  return info_;                                                                \
}                                                                              \
static void thunk_##OP##_cblas_to_fortran(                                     \
    int *wantt, int *wantz, int *n, int *ilo, int *ihi, TYPE *h, int *ldh,     \
    TYPE *wr, TYPE *wi, int *iloz, int *ihiz, TYPE *z, int *ldz, TYPE *work,   \
    int *lwork, int *info) {                                                   \
  int info_ = -1;                                                              \
  if (!wantt || !wantz || !n || !ilo || !ihi || !ldh || !iloz || !ihiz ||      \
      !ldz || !lwork || !info) {                                               \
    return;                                                                    \
  }                                                                            \
  if (!g_cblas_fn[OP_ID]) {                                                    \
    *info = -1;                                                                \
    return;                                                                    \
  }                                                                            \
  if (((*n > 0 && (!h || !wr || !wi)) || !work)) {                             \
    *info = -1;                                                                \
    return;                                                                    \
  }                                                                            \
  if (*wantz && *n > 0 && !z) {                                                \
    *info = -1;                                                                \
    return;                                                                    \
  }                                                                            \
  info_ = ((int (*)(int, int, int, int, int, TYPE *, int, TYPE *, TYPE *,      \
                    int, int, TYPE *, int, TYPE *, int))g_cblas_fn[OP_ID])(    \
      *wantt, *wantz, *n, *ilo, *ihi, h, *ldh, wr, wi, *iloz, *ihiz, z, *ldz,  \
      work, *lwork);                                                           \
  *info = info_;                                                               \
}

#define DEFINE_LAPACK_LAQR2_THUNKS(OP, TYPE, OP_ID)                            \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const int wantt, const int wantz, const int n, const int ktop,             \
    const int kbot, const int nw, TYPE *h, const int ldh, const int iloz,      \
    const int ihiz, TYPE *z, const int ldz, int *ns, int *nd, TYPE *sr,        \
    TYPE *si, TYPE *v, const int ldv, const int nh, TYPE *t, const int ldt,    \
    const int nv, TYPE *wv, const int ldwv, TYPE *work, const int lwork) {     \
  int wantt_ = wantt;                                                          \
  int wantz_ = wantz;                                                          \
  int n_ = n;                                                                  \
  int ktop_ = ktop;                                                            \
  int kbot_ = kbot;                                                            \
  int nw_ = nw;                                                                \
  int ldh_ = ldh;                                                              \
  int iloz_ = iloz;                                                            \
  int ihiz_ = ihiz;                                                            \
  int ldz_ = ldz;                                                              \
  int ldv_ = ldv;                                                              \
  int nh_ = nh;                                                                \
  int ldt_ = ldt;                                                              \
  int nv_ = nv;                                                                \
  int ldwv_ = ldwv;                                                            \
  int lwork_ = lwork;                                                          \
  if (!g_fortran_fn[OP_ID]) return -1;                                         \
  if (!ns || !nd || !sr || !si || !v || !t || !wv || !work) return -1;         \
  if (n > 0 && !h) return -1;                                                  \
  if (wantz && n > 0 && !z) return -1;                                         \
  ((void (*)(int *, int *, int *, int *, int *, int *, TYPE *, int *,          \
             int *, int *, TYPE *, int *, int *, int *, TYPE *, TYPE *,        \
             TYPE *, int *, int *, TYPE *, int *, int *, TYPE *, int *,        \
             TYPE *, int *))g_fortran_fn[OP_ID])(                              \
      &wantt_, &wantz_, &n_, &ktop_, &kbot_, &nw_, h, &ldh_, &iloz_, &ihiz_,   \
      z, &ldz_, ns, nd, sr, si, v, &ldv_, &nh_, t, &ldt_, &nv_, wv, &ldwv_,    \
      work, &lwork_);                                                          \
  return 0;                                                                    \
}                                                                              \
static void thunk_##OP##_cblas_to_fortran(                                     \
    int *wantt, int *wantz, int *n, int *ktop, int *kbot, int *nw, TYPE *h,    \
    int *ldh, int *iloz, int *ihiz, TYPE *z, int *ldz, int *ns, int *nd,       \
    TYPE *sr, TYPE *si, TYPE *v, int *ldv, int *nh, TYPE *t, int *ldt,         \
    int *nv, TYPE *wv, int *ldwv, TYPE *work, int *lwork) {                    \
  if (!g_cblas_fn[OP_ID]) return;                                              \
  if (!wantt || !wantz || !n || !ktop || !kbot || !nw || !ldh || !iloz ||      \
      !ihiz || !ldz || !ns || !nd || !ldv || !nh || !ldt || !nv || !ldwv ||    \
      !lwork || !sr || !si || !v || !t || !wv || !work) {                      \
    return;                                                                    \
  }                                                                            \
  if (*n > 0 && !h) return;                                                    \
  if (*wantz && *n > 0 && !z) return;                                          \
  (void)((int (*)(int, int, int, int, int, int, TYPE *, int, int, int,         \
                  TYPE *, int, int *, int *, TYPE *, TYPE *, TYPE *, int,      \
                  int, TYPE *, int, int, TYPE *, int, TYPE *, int))            \
             g_cblas_fn[OP_ID])(                                               \
      *wantt, *wantz, *n, *ktop, *kbot, *nw, h, *ldh, *iloz, *ihiz, z, *ldz,   \
      ns, nd, sr, si, v, *ldv, *nh, t, *ldt, *nv, wv, *ldwv, work, *lwork);    \
}

#define DEFINE_LAPACK_LAQP2_REAL_THUNKS(OP, TYPE, OP_ID)                       \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const int m, const int n, const int offset, TYPE *a, const int lda,        \
    int *jpvt, TYPE *tau, TYPE *vn1, TYPE *vn2, TYPE *work) {                  \
  int m_ = m;                                                                  \
  int n_ = n;                                                                  \
  int offset_ = offset;                                                        \
  int lda_ = lda;                                                              \
  if (!g_fortran_fn[OP_ID]) return -1;                                         \
  if ((m > 0 && n > 0 && !a) || (n > 0 && (!jpvt || !tau || !vn1 || !vn2 ||   \
                                           !work))) {                          \
    return -1;                                                                 \
  }                                                                            \
  ((void (*)(int *, int *, int *, TYPE *, int *, int *, TYPE *, TYPE *,        \
             TYPE *, TYPE *))g_fortran_fn[OP_ID])(                             \
      &m_, &n_, &offset_, a, &lda_, jpvt, tau, vn1, vn2, work);                \
  return 0;                                                                    \
}                                                                              \
static void thunk_##OP##_cblas_to_fortran(                                     \
    int *m, int *n, int *offset, TYPE *a, int *lda, int *jpvt, TYPE *tau,     \
    TYPE *vn1, TYPE *vn2, TYPE *work) {                                        \
  if (!g_cblas_fn[OP_ID]) return;                                              \
  if (!m || !n || !offset || !lda) return;                                     \
  if ((*m > 0 && *n > 0 && !a) ||                                              \
      (*n > 0 && (!jpvt || !tau || !vn1 || !vn2 || !work))) {                  \
    return;                                                                    \
  }                                                                            \
  (void)((int (*)(int, int, int, TYPE *, int, int *, TYPE *, TYPE *, TYPE *,   \
                  TYPE *))g_cblas_fn[OP_ID])(                                  \
      *m, *n, *offset, a, *lda, jpvt, tau, vn1, vn2, work);                    \
}

#define DEFINE_LAPACK_LAQP2_COMPLEX_THUNKS(OP, CTYPE, RTYPE, OP_ID)            \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const int m, const int n, const int offset, CTYPE *a, const int lda,       \
    int *jpvt, CTYPE *tau, RTYPE *vn1, RTYPE *vn2, CTYPE *work) {              \
  int m_ = m;                                                                  \
  int n_ = n;                                                                  \
  int offset_ = offset;                                                        \
  int lda_ = lda;                                                              \
  if (!g_fortran_fn[OP_ID]) return -1;                                         \
  if ((m > 0 && n > 0 && !a) || (n > 0 && (!jpvt || !tau || !vn1 || !vn2 ||   \
                                           !work))) {                          \
    return -1;                                                                 \
  }                                                                            \
  ((void (*)(int *, int *, int *, CTYPE *, int *, int *, CTYPE *, RTYPE *,    \
             RTYPE *, CTYPE *))g_fortran_fn[OP_ID])(                           \
      &m_, &n_, &offset_, a, &lda_, jpvt, tau, vn1, vn2, work);                \
  return 0;                                                                    \
}                                                                              \
static void thunk_##OP##_cblas_to_fortran(                                     \
    int *m, int *n, int *offset, CTYPE *a, int *lda, int *jpvt, CTYPE *tau,    \
    RTYPE *vn1, RTYPE *vn2, CTYPE *work) {                                     \
  if (!g_cblas_fn[OP_ID]) return;                                              \
  if (!m || !n || !offset || !lda) return;                                     \
  if ((*m > 0 && *n > 0 && !a) ||                                              \
      (*n > 0 && (!jpvt || !tau || !vn1 || !vn2 || !work))) {                  \
    return;                                                                    \
  }                                                                            \
  (void)((int (*)(int, int, int, CTYPE *, int, int *, CTYPE *, RTYPE *,        \
                  RTYPE *, CTYPE *))g_cblas_fn[OP_ID])(                        \
      *m, *n, *offset, a, *lda, jpvt, tau, vn1, vn2, work);                    \
}

#define DEFINE_LAPACK_LAQPS_REAL_THUNKS(OP, TYPE, OP_ID)                       \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const int m, const int n, const int offset, const int nb, int *kb,         \
    TYPE *a, const int lda, int *jpvt, TYPE *tau, TYPE *vn1, TYPE *vn2,        \
    TYPE *auxv, TYPE *f, const int ldf) {                                      \
  int m_ = m;                                                                  \
  int n_ = n;                                                                  \
  int offset_ = offset;                                                        \
  int nb_ = nb;                                                                \
  int lda_ = lda;                                                              \
  int ldf_ = ldf;                                                              \
  if (!g_fortran_fn[OP_ID]) return -1;                                         \
  if (!kb) return -1;                                                          \
  if ((m > 0 && n > 0 && !a) ||                                                \
      (n > 0 && (!jpvt || !tau || !vn1 || !vn2 || !auxv || !f))) {            \
    return -1;                                                                 \
  }                                                                            \
  ((void (*)(int *, int *, int *, int *, int *, TYPE *, int *, int *, TYPE *,  \
             TYPE *, TYPE *, TYPE *, TYPE *, int *))g_fortran_fn[OP_ID])(      \
      &m_, &n_, &offset_, &nb_, kb, a, &lda_, jpvt, tau, vn1, vn2, auxv, f,    \
      &ldf_);                                                                  \
  return 0;                                                                    \
}                                                                              \
static void thunk_##OP##_cblas_to_fortran(                                     \
    int *m, int *n, int *offset, int *nb, int *kb, TYPE *a, int *lda,         \
    int *jpvt, TYPE *tau, TYPE *vn1, TYPE *vn2, TYPE *auxv, TYPE *f,           \
    int *ldf) {                                                                \
  if (!g_cblas_fn[OP_ID]) return;                                              \
  if (!m || !n || !offset || !nb || !kb || !lda || !ldf) return;              \
  if ((*m > 0 && *n > 0 && !a) ||                                              \
      (*n > 0 && (!jpvt || !tau || !vn1 || !vn2 || !auxv || !f))) {           \
    return;                                                                    \
  }                                                                            \
  (void)((int (*)(int, int, int, int, int *, TYPE *, int, int *, TYPE *,       \
                  TYPE *, TYPE *, TYPE *, TYPE *, int))g_cblas_fn[OP_ID])(     \
      *m, *n, *offset, *nb, kb, a, *lda, jpvt, tau, vn1, vn2, auxv, f, *ldf);  \
}

#define DEFINE_LAPACK_LAQPS_COMPLEX_THUNKS(OP, CTYPE, RTYPE, OP_ID)            \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const int m, const int n, const int offset, const int nb, int *kb,         \
    CTYPE *a, const int lda, int *jpvt, CTYPE *tau, RTYPE *vn1, RTYPE *vn2,    \
    CTYPE *auxv, CTYPE *f, const int ldf) {                                    \
  int m_ = m;                                                                  \
  int n_ = n;                                                                  \
  int offset_ = offset;                                                        \
  int nb_ = nb;                                                                \
  int lda_ = lda;                                                              \
  int ldf_ = ldf;                                                              \
  if (!g_fortran_fn[OP_ID]) return -1;                                         \
  if (!kb) return -1;                                                          \
  if ((m > 0 && n > 0 && !a) ||                                                \
      (n > 0 && (!jpvt || !tau || !vn1 || !vn2 || !auxv || !f))) {            \
    return -1;                                                                 \
  }                                                                            \
  ((void (*)(int *, int *, int *, int *, int *, CTYPE *, int *, int *,         \
             CTYPE *, RTYPE *, RTYPE *, CTYPE *, CTYPE *, int *))              \
       g_fortran_fn[OP_ID])(                                                   \
      &m_, &n_, &offset_, &nb_, kb, a, &lda_, jpvt, tau, vn1, vn2, auxv, f,    \
      &ldf_);                                                                  \
  return 0;                                                                    \
}                                                                              \
static void thunk_##OP##_cblas_to_fortran(                                     \
    int *m, int *n, int *offset, int *nb, int *kb, CTYPE *a, int *lda,        \
    int *jpvt, CTYPE *tau, RTYPE *vn1, RTYPE *vn2, CTYPE *auxv, CTYPE *f,      \
    int *ldf) {                                                                \
  if (!g_cblas_fn[OP_ID]) return;                                              \
  if (!m || !n || !offset || !nb || !kb || !lda || !ldf) return;              \
  if ((*m > 0 && *n > 0 && !a) ||                                              \
      (*n > 0 && (!jpvt || !tau || !vn1 || !vn2 || !auxv || !f))) {           \
    return;                                                                    \
  }                                                                            \
  (void)((int (*)(int, int, int, int, int *, CTYPE *, int, int *, CTYPE *,     \
                  RTYPE *, RTYPE *, CTYPE *, CTYPE *, int))g_cblas_fn[OP_ID])( \
      *m, *n, *offset, *nb, kb, a, *lda, jpvt, tau, vn1, vn2, auxv, f, *ldf);  \
}

#define DEFINE_LAPACK_LARNV_REAL_THUNKS(OP, TYPE, OP_ID)                       \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const int idist, int *iseed, const int n, TYPE *x) {                       \
  int idist_ = idist;                                                          \
  int n_ = n;                                                                  \
  if (!g_fortran_fn[OP_ID]) return -1;                                         \
  if (!iseed || (n > 0 && !x)) return -1;                                      \
  ((void (*)(int *, int *, int *, TYPE *))g_fortran_fn[OP_ID])(                \
      &idist_, iseed, &n_, x);                                                 \
  return 0;                                                                    \
}                                                                              \
static void thunk_##OP##_cblas_to_fortran(                                     \
    int *idist, int *iseed, int *n, TYPE *x) {                                 \
  if (!g_cblas_fn[OP_ID]) return;                                              \
  if (!idist || !iseed || !n || (*n > 0 && !x)) return;                        \
  (void)((int (*)(int, int *, int, TYPE *))g_cblas_fn[OP_ID])(                 \
      *idist, iseed, *n, x);                                                   \
}

#define DEFINE_LAPACK_LARNV_COMPLEX_THUNKS(OP, CTYPE, OP_ID)                   \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const int idist, int *iseed, const int n, CTYPE *x) {                      \
  int idist_ = idist;                                                          \
  int n_ = n;                                                                  \
  if (!g_fortran_fn[OP_ID]) return -1;                                         \
  if (!iseed || (n > 0 && !x)) return -1;                                      \
  ((void (*)(int *, int *, int *, CTYPE *))g_fortran_fn[OP_ID])(               \
      &idist_, iseed, &n_, x);                                                 \
  return 0;                                                                    \
}                                                                              \
static void thunk_##OP##_cblas_to_fortran(                                     \
    int *idist, int *iseed, int *n, CTYPE *x) {                                \
  if (!g_cblas_fn[OP_ID]) return;                                              \
  if (!idist || !iseed || !n || (*n > 0 && !x)) return;                        \
  (void)((int (*)(int, int *, int, CTYPE *))g_cblas_fn[OP_ID])(                \
      *idist, iseed, *n, x);                                                   \
}

#define DEFINE_LAPACK_LARGE_THUNKS(OP, TYPE, OP_ID)                            \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const int n, TYPE *a, const int lda, int *iseed, TYPE *work) {             \
  int n_ = n;                                                                  \
  int lda_ = lda;                                                              \
  int info_ = 0;                                                               \
  if (!g_fortran_fn[OP_ID]) return -1;                                         \
  if (!iseed || (n > 0 && (!a || !work))) return -1;                           \
  ((void (*)(int *, TYPE *, int *, int *, TYPE *, int *))g_fortran_fn[OP_ID])( \
      &n_, a, &lda_, iseed, work, &info_);                                     \
  return info_;                                                                \
}                                                                              \
static void thunk_##OP##_cblas_to_fortran(                                     \
    int *n, TYPE *a, int *lda, int *iseed, TYPE *work, int *info) {            \
  int info_ = -1;                                                              \
  if (!n || !lda || !iseed || !info) return;                                   \
  if (!g_cblas_fn[OP_ID]) {                                                    \
    *info = -1;                                                                \
    return;                                                                    \
  }                                                                            \
  if (*n > 0 && (!a || !work)) {                                               \
    *info = -1;                                                                \
    return;                                                                    \
  }                                                                            \
  info_ = ((int (*)(int, TYPE *, int, int *, TYPE *))g_cblas_fn[OP_ID])(       \
      *n, a, *lda, iseed, work);                                               \
  *info = info_;                                                               \
}

#define DEFINE_LAPACK_LARGV_COMPLEX_THUNKS(OP, CTYPE, RTYPE, OP_ID)            \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const int n, CTYPE *x, const int incx, CTYPE *y, const int incy,           \
    RTYPE *c, const int incc) {                                                \
  int n_ = n;                                                                  \
  int incx_ = incx;                                                            \
  int incy_ = incy;                                                            \
  int incc_ = incc;                                                            \
  if (!g_fortran_fn[OP_ID]) return -1;                                         \
  if (n > 0 && (!x || !y || !c)) return -1;                                    \
  if (n > 0 && (incx <= 0 || incy <= 0 || incc <= 0)) return -1;               \
  ((void (*)(int *, CTYPE *, int *, CTYPE *, int *, RTYPE *, int *))           \
       g_fortran_fn[OP_ID])(                                                   \
      &n_, x, &incx_, y, &incy_, c, &incc_);                                   \
  return 0;                                                                    \
}                                                                              \
static void thunk_##OP##_cblas_to_fortran(                                     \
    int *n, CTYPE *x, int *incx, CTYPE *y, int *incy, RTYPE *c, int *incc) {  \
  if (!g_cblas_fn[OP_ID]) return;                                              \
  if (!n || !incx || !incy || !incc) return;                                   \
  if (*n > 0 && (!x || !y || !c)) return;                                      \
  if (*n > 0 && (*incx <= 0 || *incy <= 0 || *incc <= 0)) return;              \
  (void)((int (*)(int, CTYPE *, int, CTYPE *, int, RTYPE *, int))              \
      g_cblas_fn[OP_ID])(*n, x, *incx, y, *incy, c, *incc);                    \
}

#define DEFINE_LAPACK_LARRA_THUNKS(OP, TYPE, OP_ID)                            \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const int n, const TYPE *d, TYPE *e, TYPE *e2, int *spltpnt,               \
    const TYPE rtol1, const TYPE rtol2) {                                      \
  int n_ = n;                                                                   \
  TYPE rtol1_ = rtol1;                                                          \
  TYPE rtol2_ = rtol2;                                                          \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (n > 0 && (!d || !e || !e2 || !spltpnt)) return -1;                       \
  ((void (*)(int *, TYPE *, TYPE *, TYPE *, int *, TYPE *, TYPE *))            \
       g_fortran_fn[OP_ID])(                                                   \
      &n_, (TYPE *)d, e, e2, spltpnt, &rtol1_, &rtol2_);                       \
  return 0;                                                                     \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    int *n, TYPE *d, TYPE *e, TYPE *e2, int *spltpnt, TYPE *rtol1,             \
    TYPE *rtol2) {                                                              \
  if (!g_cblas_fn[OP_ID]) return;                                               \
  if (!n || !rtol1 || !rtol2) return;                                           \
  if (*n > 0 && (!d || !e || !e2 || !spltpnt)) return;                         \
  (void)((int (*)(int, const TYPE *, TYPE *, TYPE *, int *, TYPE, TYPE))       \
      g_cblas_fn[OP_ID])(*n, d, e, e2, spltpnt, *rtol1, *rtol2);               \
}

#define DEFINE_LAPACK_LARRB_THUNKS(OP, TYPE, OP_ID)                            \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const int n, TYPE *d, TYPE *lld, const int ifirst, const int ilast,        \
    const TYPE rtol1, const TYPE rtol2, const int offset, TYPE *w,             \
    TYPE *wgap, TYPE *werr, TYPE *work, int *iwork, const TYPE pivmin,         \
    const TYPE spdiam, const int twist) {                                      \
  int n_ = n;                                                                   \
  int ifirst_ = ifirst;                                                         \
  int ilast_ = ilast;                                                           \
  TYPE rtol1_ = rtol1;                                                          \
  TYPE rtol2_ = rtol2;                                                          \
  int offset_ = offset;                                                         \
  TYPE pivmin_ = pivmin;                                                        \
  TYPE spdiam_ = spdiam;                                                        \
  int twist_ = twist;                                                           \
  int info_ = 0;                                                                \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (n > 0 && (!d || !lld || !w || !wgap || !werr || !work || !iwork)) {      \
    return -1;                                                                  \
  }                                                                             \
  ((void (*)(int *, TYPE *, TYPE *, int *, int *, TYPE *, TYPE *, int *,       \
             TYPE *, TYPE *, TYPE *, TYPE *, int *, TYPE *, TYPE *, int *,     \
             int *))g_fortran_fn[OP_ID])(                                       \
      &n_, d, lld, &ifirst_, &ilast_, &rtol1_, &rtol2_, &offset_, w, wgap,     \
      werr, work, iwork, &pivmin_, &spdiam_, &twist_, &info_);                  \
  return info_;                                                                 \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    int *n, TYPE *d, TYPE *lld, int *ifirst, int *ilast, TYPE *rtol1,          \
    TYPE *rtol2, int *offset, TYPE *w, TYPE *wgap, TYPE *werr, TYPE *work,     \
    int *iwork, TYPE *pivmin, TYPE *spdiam, int *twist, int *info) {           \
  int info_ = -1;                                                               \
  if (!n || !ifirst || !ilast || !rtol1 || !rtol2 || !offset || !pivmin ||     \
      !spdiam || !twist || !info) {                                             \
    return;                                                                     \
  }                                                                             \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    *info = -1;                                                                 \
    return;                                                                     \
  }                                                                             \
  if (*n > 0 && (!d || !lld || !w || !wgap || !werr || !work || !iwork)) {     \
    *info = -1;                                                                 \
    return;                                                                     \
  }                                                                             \
  info_ = ((int (*)(int, const TYPE *, const TYPE *, int, int, TYPE, TYPE,     \
                    int, TYPE *, TYPE *, TYPE *, TYPE *, int *, TYPE, TYPE,    \
                    int))g_cblas_fn[OP_ID])(                                    \
      *n, d, lld, *ifirst, *ilast, *rtol1, *rtol2, *offset, w, wgap, werr,     \
      work, iwork, *pivmin, *spdiam, *twist);                                   \
  *info = info_;                                                                \
}

#define DEFINE_LAPACK_LARRC_THUNKS(OP, TYPE, OP_ID)                            \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const char jobt, const int n, const TYPE vl, const TYPE vu,                \
    const TYPE *d, const TYPE *e, const TYPE pivmin, int *eigcnt, int *lcnt,   \
    int *rcnt) {                                                                \
  char jobt_ = jobt;                                                            \
  int n_ = n;                                                                   \
  TYPE vl_ = vl;                                                                \
  TYPE vu_ = vu;                                                                \
  TYPE pivmin_ = pivmin;                                                        \
  int info_ = 0;                                                                \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (!eigcnt || !lcnt || !rcnt) return -1;                                     \
  if (n > 0 && (!d || !e)) return -1;                                           \
  ((void (*)(char *, int *, TYPE *, TYPE *, TYPE *, TYPE *, TYPE *, int *,     \
             int *, int *, int *))g_fortran_fn[OP_ID])(                         \
      &jobt_, &n_, &vl_, &vu_, (TYPE *)d, (TYPE *)e, &pivmin_, eigcnt, lcnt,   \
      rcnt, &info_);                                                             \
  return info_;                                                                 \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    char *jobt, int *n, TYPE *vl, TYPE *vu, TYPE *d, TYPE *e, TYPE *pivmin,    \
    int *eigcnt, int *lcnt, int *rcnt, int *info) {                             \
  int info_ = -1;                                                                \
  if (!jobt || !n || !vl || !vu || !pivmin || !eigcnt || !lcnt || !rcnt ||     \
      !info) {                                                                   \
    return;                                                                      \
  }                                                                              \
  if (!g_cblas_fn[OP_ID]) {                                                      \
    *info = -1;                                                                  \
    return;                                                                      \
  }                                                                              \
  if (*n > 0 && (!d || !e)) {                                                    \
    *info = -1;                                                                  \
    return;                                                                      \
  }                                                                              \
  info_ = ((int (*)(char, int, TYPE, TYPE, const TYPE *, const TYPE *, TYPE,   \
                    int *, int *, int *))g_cblas_fn[OP_ID])(                    \
      *jobt, *n, *vl, *vu, d, e, *pivmin, eigcnt, lcnt, rcnt);                  \
  *info = info_;                                                                 \
}

#define DEFINE_LAPACK_LARRD_THUNKS(OP, TYPE, OP_ID)                            \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const char range, const char order, const int n, const TYPE vl,            \
    const TYPE vu, const int il, const int iu, TYPE *gers, const TYPE reltol,  \
    const TYPE *d, const TYPE *e, const TYPE *e2, const TYPE pivmin,           \
    const int nsplit, const int *isplit, int *m, TYPE *w, TYPE *werr,          \
    TYPE *wl, TYPE *wu, int *iblock, int *indexw, TYPE *work, int *iwork) {    \
  char range_ = range;                                                          \
  char order_ = order;                                                          \
  int n_ = n;                                                                   \
  TYPE vl_ = vl;                                                                \
  TYPE vu_ = vu;                                                                \
  int il_ = il;                                                                 \
  int iu_ = iu;                                                                 \
  TYPE reltol_ = reltol;                                                        \
  TYPE pivmin_ = pivmin;                                                        \
  int nsplit_ = nsplit;                                                         \
  int info_ = 0;                                                                \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (!m || !wl || !wu) return -1;                                              \
  if (n > 0 && (!gers || !d || !e || !e2 || !isplit || !w || !werr ||         \
                !iblock || !indexw || !work || !iwork)) {                       \
    return -1;                                                                  \
  }                                                                             \
  ((void (*)(char *, char *, int *, TYPE *, TYPE *, int *, int *, TYPE *,      \
             TYPE *, TYPE *, TYPE *, TYPE *, TYPE *, int *, int *, int *,      \
             TYPE *, TYPE *, TYPE *, TYPE *, int *, int *, TYPE *, int *,      \
             int *))                                                            \
       g_fortran_fn[OP_ID])(                                                    \
      &range_, &order_, &n_, &vl_, &vu_, &il_, &iu_, gers, &reltol_,           \
      (TYPE *)d, (TYPE *)e, (TYPE *)e2, &pivmin_, &nsplit_, (int *)isplit,     \
      m, w, werr, wl, wu, iblock, indexw, work, iwork, &info_);                \
  return info_;                                                                 \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    char *range, char *order, int *n, TYPE *vl, TYPE *vu, int *il, int *iu,    \
    TYPE *gers, TYPE *reltol, TYPE *d, TYPE *e, TYPE *e2, TYPE *pivmin,        \
    int *nsplit, int *isplit, int *m, TYPE *w, TYPE *werr, TYPE *wl, TYPE *wu, \
    int *iblock, int *indexw, TYPE *work, int *iwork, int *info) {             \
  int info_ = -1;                                                               \
  if (!range || !order || !n || !vl || !vu || !il || !iu || !reltol ||         \
      !pivmin || !nsplit || !m || !wl || !wu || !info) {                        \
    return;                                                                     \
  }                                                                             \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    *info = -1;                                                                 \
    return;                                                                     \
  }                                                                             \
  if (*n > 0 && (!gers || !d || !e || !e2 || !isplit || !w || !werr ||         \
                 !iblock || !indexw || !work || !iwork)) {                      \
    *info = -1;                                                                 \
    return;                                                                     \
  }                                                                             \
  info_ = ((int (*)(char, char, int, TYPE, TYPE, int, int, TYPE *, TYPE,       \
                    const TYPE *, const TYPE *, const TYPE *, TYPE, int,        \
                    const int *, int *, TYPE *, TYPE *, TYPE *, TYPE *, int *,  \
                    int *, TYPE *, int *))g_cblas_fn[OP_ID])(                   \
      *range, *order, *n, *vl, *vu, *il, *iu, gers, *reltol, d, e, e2,         \
      *pivmin, *nsplit, isplit, m, w, werr, wl, wu, iblock, indexw, work,       \
      iwork);                                                                   \
  *info = info_;                                                                \
}

#define DEFINE_LAPACK_LARRE_THUNKS(OP, TYPE, OP_ID)                            \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const char range, const int n, TYPE *vl, TYPE *vu, const int il,           \
    const int iu, TYPE *d, TYPE *e, TYPE *e2, const TYPE rtol1,                \
    const TYPE rtol2, const TYPE spltol, int *nsplit, int *isplit, int *m,     \
    TYPE *w, TYPE *werr, TYPE *wgap, int *iblock, int *indexw, TYPE *gers,     \
    TYPE *pivmin, TYPE *work, int *iwork) {                                     \
  char range_ = range;                                                          \
  int n_ = n;                                                                   \
  int il_ = il;                                                                 \
  int iu_ = iu;                                                                 \
  TYPE rtol1_ = rtol1;                                                          \
  TYPE rtol2_ = rtol2;                                                          \
  TYPE spltol_ = spltol;                                                        \
  int info_ = 0;                                                                \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (!vl || !vu || !nsplit || !m || !pivmin) return -1;                       \
  if (n > 0 && (!d || !e || !e2 || !isplit || !w || !werr || !wgap ||         \
                !iblock || !indexw || !gers || !work || !iwork)) {             \
    return -1;                                                                  \
  }                                                                             \
  ((void (*)(char *, int *, TYPE *, TYPE *, int *, int *, TYPE *, TYPE *,      \
             TYPE *, TYPE *, TYPE *, TYPE *, int *, int *, int *, TYPE *,      \
             TYPE *, TYPE *, int *, int *, TYPE *, TYPE *, TYPE *, int *,      \
             int *))                                                            \
       g_fortran_fn[OP_ID])(                                                    \
      &range_, &n_, vl, vu, &il_, &iu_, d, e, e2, &rtol1_, &rtol2_,            \
      &spltol_, nsplit, isplit, m, w, werr, wgap, iblock, indexw, gers,        \
      pivmin, work, iwork, &info_);                                             \
  return info_;                                                                 \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    char *range, int *n, TYPE *vl, TYPE *vu, int *il, int *iu, TYPE *d,        \
    TYPE *e, TYPE *e2, TYPE *rtol1, TYPE *rtol2, TYPE *spltol, int *nsplit,    \
    int *isplit, int *m, TYPE *w, TYPE *werr, TYPE *wgap, int *iblock,         \
    int *indexw, TYPE *gers, TYPE *pivmin, TYPE *work, int *iwork, int *info) {\
  int info_ = -1;                                                               \
  if (!range || !n || !vl || !vu || !il || !iu || !rtol1 || !rtol2 ||          \
      !spltol || !nsplit || !m || !pivmin || !info) {                           \
    return;                                                                     \
  }                                                                             \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    *info = -1;                                                                 \
    return;                                                                     \
  }                                                                             \
  if (*n > 0 && (!d || !e || !e2 || !isplit || !w || !werr || !wgap ||         \
                 !iblock || !indexw || !gers || !work || !iwork)) {            \
    *info = -1;                                                                 \
    return;                                                                     \
  }                                                                             \
  info_ = ((int (*)(char, int, TYPE *, TYPE *, int, int, TYPE *, TYPE *,       \
                    TYPE *, TYPE, TYPE, TYPE, int *, int *, int *, TYPE *,     \
                    TYPE *, TYPE *, int *, int *, TYPE *, TYPE *, TYPE *,      \
                    int *))g_cblas_fn[OP_ID])(                                  \
      *range, *n, vl, vu, *il, *iu, d, e, e2, *rtol1, *rtol2, *spltol,         \
      nsplit, isplit, m, w, werr, wgap, iblock, indexw, gers, pivmin, work,    \
      iwork);                                                                   \
  *info = info_;                                                                \
}

#define DEFINE_LAPACK_LARRF_THUNKS(OP, TYPE, OP_ID)                            \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const int n, const TYPE *d, const TYPE *l, const TYPE *ld,                 \
    const int clstrt, const int clend, const TYPE *w, TYPE *wgap,              \
    const TYPE *werr, const TYPE spdiam, const TYPE clgapl,                    \
    const TYPE clgapr, const TYPE pivmin, TYPE *sigma, TYPE *dplus,            \
    TYPE *lplus, TYPE *work) {                                                  \
  int n_ = n;                                                                   \
  int clstrt_ = clstrt;                                                         \
  int clend_ = clend;                                                           \
  TYPE spdiam_ = spdiam;                                                        \
  TYPE clgapl_ = clgapl;                                                        \
  TYPE clgapr_ = clgapr;                                                        \
  TYPE pivmin_ = pivmin;                                                        \
  int info_ = 0;                                                                \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (!sigma) return -1;                                                        \
  if (n > 0 && (!d || !l || !ld || !w || !wgap || !werr || !dplus ||          \
                !lplus || !work)) {                                             \
    return -1;                                                                  \
  }                                                                             \
  ((void (*)(int *, TYPE *, TYPE *, TYPE *, int *, int *, TYPE *, TYPE *,      \
             TYPE *, TYPE *, TYPE *, TYPE *, TYPE *, TYPE *, TYPE *, TYPE *,   \
             TYPE *, int *))g_fortran_fn[OP_ID])(                              \
      &n_, (TYPE *)d, (TYPE *)l, (TYPE *)ld, &clstrt_, &clend_, (TYPE *)w,     \
      wgap, (TYPE *)werr, &spdiam_, &clgapl_, &clgapr_, &pivmin_, sigma,       \
      dplus, lplus, work, &info_);                                              \
  return info_;                                                                 \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    int *n, TYPE *d, TYPE *l, TYPE *ld, int *clstrt, int *clend, TYPE *w,      \
    TYPE *wgap, TYPE *werr, TYPE *spdiam, TYPE *clgapl, TYPE *clgapr,          \
    TYPE *pivmin, TYPE *sigma, TYPE *dplus, TYPE *lplus, TYPE *work,           \
    int *info) {                                                                \
  int info_ = -1;                                                               \
  if (!n || !clstrt || !clend || !spdiam || !clgapl || !clgapr || !pivmin ||   \
      !sigma || !info) {                                                        \
    return;                                                                     \
  }                                                                             \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    *info = -1;                                                                 \
    return;                                                                     \
  }                                                                             \
  if (*n > 0 && (!d || !l || !ld || !w || !wgap || !werr || !dplus ||         \
                 !lplus || !work)) {                                            \
    *info = -1;                                                                 \
    return;                                                                     \
  }                                                                             \
  info_ = ((int (*)(int, const TYPE *, const TYPE *, const TYPE *, int, int,   \
                    const TYPE *, TYPE *, const TYPE *, TYPE, TYPE, TYPE,      \
                    TYPE, TYPE *, TYPE *, TYPE *, TYPE *))g_cblas_fn[OP_ID])(   \
      *n, d, l, ld, *clstrt, *clend, w, wgap, werr, *spdiam, *clgapl,          \
      *clgapr, *pivmin, sigma, dplus, lplus, work);                             \
  *info = info_;                                                                \
}

#define DEFINE_LAPACK_LARRJ_THUNKS(OP, TYPE, OP_ID)                            \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const int n, const TYPE *d, const TYPE *e2, const int ifirst,              \
    const int ilast, const TYPE rtol, const int offset, TYPE *w, TYPE *werr,   \
    TYPE *work, int *iwork, const TYPE pivmin, const TYPE spdiam) {            \
  int n_ = n;                                                                   \
  int ifirst_ = ifirst;                                                         \
  int ilast_ = ilast;                                                           \
  TYPE rtol_ = rtol;                                                            \
  int offset_ = offset;                                                         \
  TYPE pivmin_ = pivmin;                                                        \
  TYPE spdiam_ = spdiam;                                                        \
  int info_ = 0;                                                                \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (n > 0 && (!d || !e2 || !w || !werr || !work || !iwork)) return -1;      \
  ((void (*)(int *, TYPE *, TYPE *, int *, int *, TYPE *, int *, TYPE *,       \
             TYPE *, TYPE *, int *, TYPE *, TYPE *, int *))                    \
       g_fortran_fn[OP_ID])(                                                    \
      &n_, (TYPE *)d, (TYPE *)e2, &ifirst_, &ilast_, &rtol_, &offset_, w,      \
      werr, work, iwork, &pivmin_, &spdiam_, &info_);                           \
  return info_;                                                                 \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    int *n, TYPE *d, TYPE *e2, int *ifirst, int *ilast, TYPE *rtol,            \
    int *offset, TYPE *w, TYPE *werr, TYPE *work, int *iwork, TYPE *pivmin,    \
    TYPE *spdiam, int *info) {                                                  \
  int info_ = -1;                                                               \
  if (!n || !ifirst || !ilast || !rtol || !offset || !pivmin || !spdiam ||     \
      !info) {                                                                  \
    return;                                                                     \
  }                                                                             \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    *info = -1;                                                                 \
    return;                                                                     \
  }                                                                             \
  if (*n > 0 && (!d || !e2 || !w || !werr || !work || !iwork)) {              \
    *info = -1;                                                                 \
    return;                                                                     \
  }                                                                             \
  info_ = ((int (*)(int, const TYPE *, const TYPE *, int, int, TYPE, int,      \
                    TYPE *, TYPE *, TYPE *, int *, TYPE, TYPE))                \
      g_cblas_fn[OP_ID])(*n, d, e2, *ifirst, *ilast, *rtol, *offset, w, werr,  \
                         work, iwork, *pivmin, *spdiam);                        \
  *info = info_;                                                                \
}

#define DEFINE_LAPACK_LARRK_THUNKS(OP, TYPE, OP_ID)                            \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const int n, const int iw, const TYPE gl, const TYPE gu, const TYPE *d,    \
    const TYPE *e2, const TYPE pivmin, const TYPE reltol, TYPE *w,             \
    TYPE *werr) {                                                               \
  int n_ = n;                                                                   \
  int iw_ = iw;                                                                 \
  TYPE gl_ = gl;                                                                \
  TYPE gu_ = gu;                                                                \
  TYPE pivmin_ = pivmin;                                                        \
  TYPE reltol_ = reltol;                                                        \
  int info_ = 0;                                                                \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (!w || !werr) return -1;                                                   \
  if (n > 0 && (!d || !e2)) return -1;                                          \
  ((void (*)(int *, int *, TYPE *, TYPE *, TYPE *, TYPE *, TYPE *, TYPE *,     \
             TYPE *, TYPE *, int *))g_fortran_fn[OP_ID])(                      \
      &n_, &iw_, &gl_, &gu_, (TYPE *)d, (TYPE *)e2, &pivmin_, &reltol_,        \
      w, werr, &info_);                                                         \
  return info_;                                                                 \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    int *n, int *iw, TYPE *gl, TYPE *gu, TYPE *d, TYPE *e2, TYPE *pivmin,      \
    TYPE *reltol, TYPE *w, TYPE *werr, int *info) {                             \
  int info_ = -1;                                                               \
  if (!n || !iw || !gl || !gu || !pivmin || !reltol || !w || !werr || !info) { \
    return;                                                                     \
  }                                                                             \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    *info = -1;                                                                 \
    return;                                                                     \
  }                                                                             \
  if (*n > 0 && (!d || !e2)) {                                                  \
    *info = -1;                                                                 \
    return;                                                                     \
  }                                                                             \
  info_ = ((int (*)(int, int, TYPE, TYPE, const TYPE *, const TYPE *, TYPE,    \
                    TYPE, TYPE *, TYPE *))g_cblas_fn[OP_ID])(                  \
      *n, *iw, *gl, *gu, d, e2, *pivmin, *reltol, w, werr);                    \
  *info = info_;                                                                \
}

#define DEFINE_LAPACK_LARRR_THUNKS(OP, TYPE, OP_ID)                            \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const int n, const TYPE *d, TYPE *e) {                                      \
  int n_ = n;                                                                   \
  int info_ = 0;                                                                \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (n > 0 && (!d || !e)) return -1;                                           \
  ((void (*)(int *, TYPE *, TYPE *, int *))g_fortran_fn[OP_ID])(                \
      &n_, (TYPE *)d, e, &info_);                                               \
  return info_;                                                                 \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    int *n, TYPE *d, TYPE *e, int *info) {                                      \
  int info_ = -1;                                                               \
  if (!n || !info) {                                                            \
    return;                                                                     \
  }                                                                             \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    *info = -1;                                                                 \
    return;                                                                     \
  }                                                                             \
  if (*n > 0 && (!d || !e)) {                                                   \
    *info = -1;                                                                 \
    return;                                                                     \
  }                                                                             \
  info_ = ((int (*)(int, const TYPE *, TYPE *))g_cblas_fn[OP_ID])(*n, d, e);   \
  *info = info_;                                                                \
}

#define DEFINE_LAPACK_LARRV_THUNKS(OP, TYPE, OP_ID)                            \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const int n, const TYPE vl, const TYPE vu, TYPE *d, TYPE *l,               \
    const TYPE pivmin, const int *isplit, const int m, const int dol,          \
    const int dou, const TYPE minrgp, const TYPE rtol1, const TYPE rtol2,      \
    TYPE *w, TYPE *werr, TYPE *wgap, const int *iblock, const int *indexw,     \
    const TYPE *gers, TYPE *z, const int ldz, int *isuppz, TYPE *work,         \
    int *iwork) {                                                               \
  int n_ = n;                                                                   \
  TYPE vl_ = vl;                                                                \
  TYPE vu_ = vu;                                                                \
  TYPE pivmin_ = pivmin;                                                        \
  int m_ = m;                                                                   \
  int dol_ = dol;                                                               \
  int dou_ = dou;                                                               \
  TYPE minrgp_ = minrgp;                                                        \
  TYPE rtol1_ = rtol1;                                                          \
  TYPE rtol2_ = rtol2;                                                          \
  int ldz_ = ldz;                                                               \
  int info_ = 0;                                                                \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (!isuppz) return -1;                                                       \
  if (n > 0 && (!d || !l || !isplit || !w || !werr || !wgap || !iblock ||     \
                !indexw || !gers || !z || !work || !iwork)) {                  \
    return -1;                                                                  \
  }                                                                             \
  ((void (*)(int *, TYPE *, TYPE *, TYPE *, TYPE *, TYPE *, int *, int *,      \
             int *, int *, TYPE *, TYPE *, TYPE *, TYPE *, TYPE *, TYPE *,     \
             int *, int *, TYPE *, TYPE *, int *, int *, TYPE *, int *,        \
             int *))g_fortran_fn[OP_ID])(                                       \
      &n_, &vl_, &vu_, d, l, &pivmin_, (int *)isplit, &m_, &dol_, &dou_,       \
      &minrgp_, &rtol1_, &rtol2_, w, werr, wgap, (int *)iblock,                \
      (int *)indexw, (TYPE *)gers, z, &ldz_, isuppz, work, iwork, &info_);     \
  return info_;                                                                 \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    int *n, TYPE *vl, TYPE *vu, TYPE *d, TYPE *l, TYPE *pivmin, int *isplit,   \
    int *m, int *dol, int *dou, TYPE *minrgp, TYPE *rtol1, TYPE *rtol2,        \
    TYPE *w, TYPE *werr, TYPE *wgap, int *iblock, int *indexw, TYPE *gers,     \
    TYPE *z, int *ldz, int *isuppz, TYPE *work, int *iwork, int *info) {       \
  int info_ = -1;                                                               \
  if (!n || !vl || !vu || !pivmin || !m || !dol || !dou || !minrgp ||          \
      !rtol1 || !rtol2 || !ldz || !isuppz || !info) {                           \
    return;                                                                     \
  }                                                                             \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    *info = -1;                                                                 \
    return;                                                                     \
  }                                                                             \
  if (*n > 0 && (!d || !l || !isplit || !w || !werr || !wgap || !iblock ||     \
                 !indexw || !gers || !z || !work || !iwork)) {                 \
    *info = -1;                                                                 \
    return;                                                                     \
  }                                                                             \
  info_ = ((int (*)(int, TYPE, TYPE, TYPE *, TYPE *, TYPE, const int *, int,   \
                    int, int, TYPE, TYPE, TYPE, TYPE *, TYPE *, TYPE *,       \
                    const int *, const int *, const TYPE *, TYPE *, int,       \
                    int *, TYPE *, int *))g_cblas_fn[OP_ID])(                   \
      *n, *vl, *vu, d, l, *pivmin, isplit, *m, *dol, *dou, *minrgp, *rtol1,    \
      *rtol2, w, werr, wgap, iblock, indexw, gers, z, *ldz, isuppz, work,      \
      iwork);                                                                   \
  *info = info_;                                                                \
}

#define DEFINE_LAPACK_LARTG_REAL_THUNKS(OP, TYPE, OP_ID)                       \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const TYPE f, const TYPE g, TYPE *cs, TYPE *sn, TYPE *r) {                 \
  TYPE f_ = f;                                                                  \
  TYPE g_ = g;                                                                  \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (!cs || !sn || !r) return -1;                                              \
  ((void (*)(TYPE *, TYPE *, TYPE *, TYPE *, TYPE *))g_fortran_fn[OP_ID])(     \
      &f_, &g_, cs, sn, r);                                                     \
  return 0;                                                                     \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    TYPE *f, TYPE *g, TYPE *cs, TYPE *sn, TYPE *r) {                            \
  if (!g_cblas_fn[OP_ID]) return;                                               \
  if (!f || !g || !cs || !sn || !r) return;                                     \
  (void)((int (*)(TYPE, TYPE, TYPE *, TYPE *, TYPE *))g_cblas_fn[OP_ID])(      \
      *f, *g, cs, sn, r);                                                       \
}

#define DEFINE_LAPACK_LARTG_COMPLEX_THUNKS(OP, CTYPE, RTYPE, OP_ID)            \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const CTYPE f, const CTYPE g, RTYPE *cs, CTYPE *sn, CTYPE *r) {            \
  CTYPE f_ = f;                                                                  \
  CTYPE g_ = g;                                                                  \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (!cs || !sn || !r) return -1;                                              \
  ((void (*)(CTYPE *, CTYPE *, RTYPE *, CTYPE *, CTYPE *))g_fortran_fn[OP_ID])(\
      &f_, &g_, cs, sn, r);                                                     \
  return 0;                                                                     \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    CTYPE *f, CTYPE *g, RTYPE *cs, CTYPE *sn, CTYPE *r) {                       \
  if (!g_cblas_fn[OP_ID]) return;                                               \
  if (!f || !g || !cs || !sn || !r) return;                                     \
  (void)((int (*)(CTYPE, CTYPE, RTYPE *, CTYPE *, CTYPE *))                    \
      g_cblas_fn[OP_ID])(*f, *g, cs, sn, r);                                    \
}

#define DEFINE_LAPACK_LARTV_REAL_THUNKS(OP, TYPE, OP_ID)                       \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const int n, TYPE *x, const int incx, TYPE *y, const int incy, TYPE *c,    \
    TYPE *s, const int incc) {                                                  \
  int n_ = n;                                                                   \
  int incx_ = incx;                                                             \
  int incy_ = incy;                                                             \
  int incc_ = incc;                                                             \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (n > 0 && (!x || !y || !c || !s)) return -1;                               \
  if (n > 0 && (incx <= 0 || incy <= 0 || incc <= 0)) return -1;                \
  ((void (*)(int *, TYPE *, int *, TYPE *, int *, TYPE *, TYPE *, int *))      \
       g_fortran_fn[OP_ID])(                                                    \
      &n_, x, &incx_, y, &incy_, c, s, &incc_);                                \
  return 0;                                                                     \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    int *n, TYPE *x, int *incx, TYPE *y, int *incy, TYPE *c, TYPE *s,          \
    int *incc) {                                                                \
  if (!g_cblas_fn[OP_ID]) return;                                               \
  if (!n || !incx || !incy || !incc) return;                                    \
  if (*n > 0 && (!x || !y || !c || !s)) return;                                 \
  if (*n > 0 && (*incx <= 0 || *incy <= 0 || *incc <= 0)) return;               \
  (void)((int (*)(int, TYPE *, int, TYPE *, int, TYPE *, TYPE *, int))         \
      g_cblas_fn[OP_ID])(*n, x, *incx, y, *incy, c, s, *incc);                  \
}

#define DEFINE_LAPACK_LARTV_COMPLEX_THUNKS(OP, CTYPE, RTYPE, OP_ID)            \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const int n, CTYPE *x, const int incx, CTYPE *y, const int incy, RTYPE *c, \
    CTYPE *s, const int incc) {                                                 \
  int n_ = n;                                                                   \
  int incx_ = incx;                                                             \
  int incy_ = incy;                                                             \
  int incc_ = incc;                                                             \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (n > 0 && (!x || !y || !c || !s)) return -1;                               \
  if (n > 0 && (incx <= 0 || incy <= 0 || incc <= 0)) return -1;                \
  ((void (*)(int *, CTYPE *, int *, CTYPE *, int *, RTYPE *, CTYPE *, int *))  \
       g_fortran_fn[OP_ID])(                                                    \
      &n_, x, &incx_, y, &incy_, c, s, &incc_);                                \
  return 0;                                                                     \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    int *n, CTYPE *x, int *incx, CTYPE *y, int *incy, RTYPE *c, CTYPE *s,      \
    int *incc) {                                                                \
  if (!g_cblas_fn[OP_ID]) return;                                               \
  if (!n || !incx || !incy || !incc) return;                                    \
  if (*n > 0 && (!x || !y || !c || !s)) return;                                 \
  if (*n > 0 && (*incx <= 0 || *incy <= 0 || *incc <= 0)) return;               \
  (void)((int (*)(int, CTYPE *, int, CTYPE *, int, RTYPE *, CTYPE *, int))     \
      g_cblas_fn[OP_ID])(*n, x, *incx, y, *incy, c, s, *incc);                  \
}

#define DEFINE_LAPACK_LAS2_THUNKS(OP, TYPE, OP_ID)                             \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const TYPE a, const TYPE b, const TYPE c, TYPE *rt1, TYPE *rt2) {          \
  TYPE a_ = a;                                                                  \
  TYPE b_ = b;                                                                  \
  TYPE c_ = c;                                                                  \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (!rt1 || !rt2) return -1;                                                  \
  ((void (*)(TYPE *, TYPE *, TYPE *, TYPE *, TYPE *))g_fortran_fn[OP_ID])(     \
      &a_, &b_, &c_, rt1, rt2);                                                 \
  return 0;                                                                     \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    TYPE *a, TYPE *b, TYPE *c, TYPE *rt1, TYPE *rt2) {                          \
  if (!g_cblas_fn[OP_ID]) return;                                               \
  if (!a || !b || !c || !rt1 || !rt2) return;                                   \
  (void)((int (*)(TYPE, TYPE, TYPE, TYPE *, TYPE *))g_cblas_fn[OP_ID])(        \
      *a, *b, *c, rt1, rt2);                                                    \
}

#define DEFINE_LAPACK_LASCL_REAL_THUNKS(OP, TYPE, OP_ID)                       \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const char type, const int kl, const int ku, const TYPE cfrom,             \
    const TYPE cto, const int m, const int n, TYPE *a, const int lda) {        \
  char type_ = type;                                                            \
  int kl_ = kl;                                                                 \
  int ku_ = ku;                                                                 \
  TYPE cfrom_ = cfrom;                                                          \
  TYPE cto_ = cto;                                                              \
  int m_ = m;                                                                   \
  int n_ = n;                                                                   \
  int lda_ = lda;                                                               \
  int info_ = 0;                                                                \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if ((m > 0 || n > 0) && !a) return -1;                                        \
  ((void (*)(char *, int *, int *, TYPE *, TYPE *, int *, int *, TYPE *, int *,\
             int *))g_fortran_fn[OP_ID])(                                       \
      &type_, &kl_, &ku_, &cfrom_, &cto_, &m_, &n_, a, &lda_, &info_);         \
  return info_;                                                                 \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    char *type, int *kl, int *ku, TYPE *cfrom, TYPE *cto, int *m, int *n,      \
    TYPE *a, int *lda, int *info) {                                             \
  int info_ = -1;                                                               \
  if (!type || !kl || !ku || !cfrom || !cto || !m || !n || !lda || !info) {    \
    return;                                                                     \
  }                                                                             \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    *info = -1;                                                                 \
    return;                                                                     \
  }                                                                             \
  if ((*m > 0 || *n > 0) && !a) {                                               \
    *info = -1;                                                                 \
    return;                                                                     \
  }                                                                             \
  info_ = ((int (*)(char, int, int, TYPE, TYPE, int, int, TYPE *, int))        \
      g_cblas_fn[OP_ID])(*type, *kl, *ku, *cfrom, *cto, *m, *n, a, *lda);      \
  *info = info_;                                                                \
}

#define DEFINE_LAPACK_LASCL_COMPLEX_THUNKS(OP, CTYPE, RTYPE, OP_ID)            \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const char type, const int kl, const int ku, const RTYPE cfrom,            \
    const RTYPE cto, const int m, const int n, CTYPE *a, const int lda) {      \
  char type_ = type;                                                            \
  int kl_ = kl;                                                                 \
  int ku_ = ku;                                                                 \
  RTYPE cfrom_ = cfrom;                                                         \
  RTYPE cto_ = cto;                                                             \
  int m_ = m;                                                                   \
  int n_ = n;                                                                   \
  int lda_ = lda;                                                               \
  int info_ = 0;                                                                \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if ((m > 0 || n > 0) && !a) return -1;                                        \
  ((void (*)(char *, int *, int *, RTYPE *, RTYPE *, int *, int *, CTYPE *,    \
             int *, int *))g_fortran_fn[OP_ID])(                               \
      &type_, &kl_, &ku_, &cfrom_, &cto_, &m_, &n_, a, &lda_, &info_);         \
  return info_;                                                                 \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    char *type, int *kl, int *ku, RTYPE *cfrom, RTYPE *cto, int *m, int *n,    \
    CTYPE *a, int *lda, int *info) {                                            \
  int info_ = -1;                                                               \
  if (!type || !kl || !ku || !cfrom || !cto || !m || !n || !lda || !info) {    \
    return;                                                                     \
  }                                                                             \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    *info = -1;                                                                 \
    return;                                                                     \
  }                                                                             \
  if ((*m > 0 || *n > 0) && !a) {                                               \
    *info = -1;                                                                 \
    return;                                                                     \
  }                                                                             \
  info_ = ((int (*)(char, int, int, RTYPE, RTYPE, int, int, CTYPE *, int))     \
      g_cblas_fn[OP_ID])(*type, *kl, *ku, *cfrom, *cto, *m, *n, a, *lda);      \
  *info = info_;                                                                \
}

#define DEFINE_LAPACK_LASD0_THUNKS(OP, TYPE, OP_ID)                            \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const int n, const int sqre, TYPE *d, const TYPE *e, TYPE *u,              \
    const int ldu, TYPE *vt, const int ldvt, const int smlsiz, int *iwork,     \
    TYPE *work) {                                                               \
  int n_ = n;                                                                   \
  int sqre_ = sqre;                                                             \
  int ldu_ = ldu;                                                               \
  int ldvt_ = ldvt;                                                             \
  int smlsiz_ = smlsiz;                                                         \
  int info_ = 0;                                                                \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (n > 0 && (!d || !u || !vt || !iwork || !work)) return -1;                \
  if ((n > 1 || sqre != 0) && !e) return -1;                                    \
  ((void (*)(int *, int *, TYPE *, TYPE *, TYPE *, int *, TYPE *, int *,        \
             int *, int *, TYPE *, int *))g_fortran_fn[OP_ID])(                 \
      &n_, &sqre_, d, (TYPE *)e, u, &ldu_, vt, &ldvt_, &smlsiz_, iwork, work,   \
      &info_);                                                                  \
  return info_;                                                                 \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    int *n, int *sqre, TYPE *d, TYPE *e, TYPE *u, int *ldu, TYPE *vt,          \
    int *ldvt, int *smlsiz, int *iwork, TYPE *work, int *info) {               \
  int info_ = -1;                                                               \
  if (!n || !sqre || !ldu || !ldvt || !smlsiz || !info) return;                 \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    *info = -1;                                                                 \
    return;                                                                     \
  }                                                                             \
  if (*n > 0 && (!d || !u || !vt || !iwork || !work)) {                        \
    *info = -1;                                                                 \
    return;                                                                     \
  }                                                                             \
  if ((*n > 1 || *sqre != 0) && !e) {                                           \
    *info = -1;                                                                 \
    return;                                                                     \
  }                                                                             \
  info_ = ((int (*)(int, int, TYPE *, const TYPE *, TYPE *, int, TYPE *, int,   \
                    int, int *, TYPE *))g_cblas_fn[OP_ID])(                     \
      *n, *sqre, d, e, u, *ldu, vt, *ldvt, *smlsiz, iwork, work);               \
  *info = info_;                                                                \
}

#define DEFINE_LAPACK_LASD1_THUNKS(OP, TYPE, OP_ID)                            \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const int nl, const int nr, const int sqre, TYPE *d, const TYPE alpha,     \
    const TYPE beta, TYPE *u, const int ldu, TYPE *vt, const int ldvt,          \
    int *idxq, int *iwork, TYPE *work) {                                        \
  int nl_ = nl;                                                                  \
  int nr_ = nr;                                                                  \
  int sqre_ = sqre;                                                              \
  TYPE alpha_ = alpha;                                                           \
  TYPE beta_ = beta;                                                             \
  int ldu_ = ldu;                                                                \
  int ldvt_ = ldvt;                                                              \
  int info_ = 0;                                                                 \
  if (!g_fortran_fn[OP_ID]) return -1;                                           \
  if (nl + nr > 0 && (!d || !u || !vt || !idxq || !iwork || !work)) return -1; \
  ((void (*)(int *, int *, int *, TYPE *, TYPE *, TYPE *, TYPE *, int *,        \
             TYPE *, int *, int *, int *, TYPE *, int *))g_fortran_fn[OP_ID])(  \
      &nl_, &nr_, &sqre_, d, &alpha_, &beta_, u, &ldu_, vt, &ldvt_, idxq,       \
      iwork, work, &info_);                                                      \
  return info_;                                                                  \
}                                                                                \
static void thunk_##OP##_cblas_to_fortran(                                       \
    int *nl, int *nr, int *sqre, TYPE *d, TYPE *alpha, TYPE *beta, TYPE *u,     \
    int *ldu, TYPE *vt, int *ldvt, int *idxq, int *iwork, TYPE *work,           \
    int *info) {                                                                  \
  int info_ = -1;                                                                 \
  if (!nl || !nr || !sqre || !alpha || !beta || !ldu || !ldvt || !info) return; \
  if (!g_cblas_fn[OP_ID]) {                                                       \
    *info = -1;                                                                   \
    return;                                                                       \
  }                                                                               \
  if (*nl + *nr > 0 && (!d || !u || !vt || !idxq || !iwork || !work)) {         \
    *info = -1;                                                                   \
    return;                                                                       \
  }                                                                               \
  info_ = ((int (*)(int, int, int, TYPE *, TYPE, TYPE, TYPE *, int, TYPE *,      \
                    int, int *, int *, TYPE *))g_cblas_fn[OP_ID])(               \
      *nl, *nr, *sqre, d, *alpha, *beta, u, *ldu, vt, *ldvt, idxq, iwork, work); \
  *info = info_;                                                                  \
}

#define DEFINE_LAPACK_LASD2_THUNKS(OP, TYPE, OP_ID)                              \
static int thunk_##OP##_fortran_to_cblas(                                        \
    const int nl, const int nr, const int sqre, int *k, TYPE *d, TYPE *z,       \
    const TYPE alpha, const TYPE beta, TYPE *u, const int ldu, TYPE *vt,         \
    const int ldvt, TYPE *dsigma, TYPE *u2, const int ldu2, TYPE *vt2,           \
    const int ldvt2, int *idxp, int *idx, int *idxc, int *idxq, int *coltyp) {  \
  int nl_ = nl;                                                                   \
  int nr_ = nr;                                                                   \
  int sqre_ = sqre;                                                               \
  TYPE alpha_ = alpha;                                                            \
  TYPE beta_ = beta;                                                              \
  int ldu_ = ldu;                                                                 \
  int ldvt_ = ldvt;                                                               \
  int ldu2_ = ldu2;                                                               \
  int ldvt2_ = ldvt2;                                                             \
  int info_ = 0;                                                                  \
  if (!g_fortran_fn[OP_ID]) return -1;                                            \
  if (!k) return -1;                                                              \
  if (nl + nr > 0 && (!d || !z || !u || !vt || !dsigma || !u2 || !vt2 || !idxp ||\
                      !idx || !idxc || !idxq || !coltyp)) return -1;             \
  ((void (*)(int *, int *, int *, int *, TYPE *, TYPE *, TYPE *, TYPE *, TYPE *, \
             int *, TYPE *, int *, TYPE *, TYPE *, int *, TYPE *, int *, int *,  \
             int *, int *, int *, int *, int *))g_fortran_fn[OP_ID])(            \
      &nl_, &nr_, &sqre_, k, d, z, &alpha_, &beta_, u, &ldu_, vt, &ldvt_,        \
      dsigma, u2, &ldu2_, vt2, &ldvt2_, idxp, idx, idxc, idxq, coltyp, &info_);  \
  return info_;                                                                   \
}                                                                                 \
static void thunk_##OP##_cblas_to_fortran(                                        \
    int *nl, int *nr, int *sqre, int *k, TYPE *d, TYPE *z, TYPE *alpha,          \
    TYPE *beta, TYPE *u, int *ldu, TYPE *vt, int *ldvt, TYPE *dsigma, TYPE *u2,  \
    int *ldu2, TYPE *vt2, int *ldvt2, int *idxp, int *idx, int *idxc,            \
    int *idxq, int *coltyp, int *info) {                                          \
  int info_ = -1;                                                                  \
  if (!nl || !nr || !sqre || !k || !alpha || !beta || !ldu || !ldvt || !ldu2 ||  \
      !ldvt2 || !info) return;                                                     \
  if (!g_cblas_fn[OP_ID]) {                                                        \
    *info = -1;                                                                    \
    return;                                                                        \
  }                                                                                \
  if (*nl + *nr > 0 && (!d || !z || !u || !vt || !dsigma || !u2 || !vt2 ||       \
                        !idxp || !idx || !idxc || !idxq || !coltyp)) {            \
    *info = -1;                                                                    \
    return;                                                                        \
  }                                                                                \
  info_ = ((int (*)(int, int, int, int *, TYPE *, TYPE *, TYPE, TYPE, TYPE *,     \
                    int, TYPE *, int, TYPE *, TYPE *, int, TYPE *, int, int *,    \
                    int *, int *, int *, int *))g_cblas_fn[OP_ID])(               \
      *nl, *nr, *sqre, k, d, z, *alpha, *beta, u, *ldu, vt, *ldvt, dsigma, u2,    \
      *ldu2, vt2, *ldvt2, idxp, idx, idxc, idxq, coltyp);                         \
  *info = info_;                                                                   \
}

#define DEFINE_LAPACK_LASD3_THUNKS(OP, TYPE, OP_ID)                               \
static int thunk_##OP##_fortran_to_cblas(                                         \
    const int nl, const int nr, const int sqre, const int k, TYPE *d, TYPE *q,   \
    const int ldq, TYPE *dsigma, TYPE *u, const int ldu, TYPE *u2,               \
    const int ldu2, TYPE *vt, const int ldvt, TYPE *vt2, const int ldvt2,        \
    int *idxc, int *ctot, TYPE *z) {                                              \
  int nl_ = nl;                                                                    \
  int nr_ = nr;                                                                    \
  int sqre_ = sqre;                                                                \
  int k_ = k;                                                                      \
  int ldq_ = ldq;                                                                  \
  int ldu_ = ldu;                                                                  \
  int ldu2_ = ldu2;                                                                \
  int ldvt_ = ldvt;                                                                \
  int ldvt2_ = ldvt2;                                                              \
  int info_ = 0;                                                                   \
  if (!g_fortran_fn[OP_ID]) return -1;                                             \
  if (k > 0 && (!d || !q || !dsigma || !u || !u2 || !vt || !vt2 || !idxc ||      \
                !ctot || !z)) return -1;                                           \
  ((void (*)(int *, int *, int *, int *, TYPE *, TYPE *, int *, TYPE *, TYPE *,   \
             int *, TYPE *, int *, TYPE *, int *, TYPE *, int *, int *, int *,    \
             TYPE *, int *))g_fortran_fn[OP_ID])(                                  \
      &nl_, &nr_, &sqre_, &k_, d, q, &ldq_, dsigma, u, &ldu_, u2, &ldu2_, vt,     \
      &ldvt_, vt2, &ldvt2_, idxc, ctot, z, &info_);                               \
  return info_;                                                                    \
}                                                                                  \
static void thunk_##OP##_cblas_to_fortran(                                         \
    int *nl, int *nr, int *sqre, int *k, TYPE *d, TYPE *q, int *ldq,              \
    TYPE *dsigma, TYPE *u, int *ldu, TYPE *u2, int *ldu2, TYPE *vt, int *ldvt,    \
    TYPE *vt2, int *ldvt2, int *idxc, int *ctot, TYPE *z, int *info) {            \
  int info_ = -1;                                                                   \
  if (!nl || !nr || !sqre || !k || !ldq || !ldu || !ldu2 || !ldvt || !ldvt2 ||    \
      !info) return;                                                                \
  if (!g_cblas_fn[OP_ID]) {                                                         \
    *info = -1;                                                                     \
    return;                                                                         \
  }                                                                                 \
  if (*k > 0 && (!d || !q || !dsigma || !u || !u2 || !vt || !vt2 || !idxc ||      \
                 !ctot || !z)) {                                                    \
    *info = -1;                                                                     \
    return;                                                                         \
  }                                                                                 \
  info_ = ((int (*)(int, int, int, int, TYPE *, TYPE *, int, TYPE *, TYPE *, int, \
                    TYPE *, int, TYPE *, int, TYPE *, int, int *, int *, TYPE *)) \
      g_cblas_fn[OP_ID])(*nl, *nr, *sqre, *k, d, q, *ldq, dsigma, u, *ldu, u2,    \
                         *ldu2, vt, *ldvt, vt2, *ldvt2, idxc, ctot, z);            \
  *info = info_;                                                                    \
}

#define DEFINE_LAPACK_LASD4_THUNKS(OP, TYPE, OP_ID)                               \
static int thunk_##OP##_fortran_to_cblas(                                         \
    const int n, TYPE *d, TYPE *z, TYPE *delta, const TYPE rho, TYPE *sigma,     \
    TYPE *work) {                                                                  \
  int n_ = n;                                                                      \
  TYPE rho_ = rho;                                                                 \
  int info_ = 0;                                                                   \
  if (!g_fortran_fn[OP_ID]) return -1;                                             \
  if (!sigma) return -1;                                                           \
  if (n > 0 && (!d || !z || !delta || !work)) return -1;                          \
  ((void (*)(int *, TYPE *, TYPE *, TYPE *, TYPE *, TYPE *, TYPE *, int *))       \
      g_fortran_fn[OP_ID])(&n_, d, z, delta, &rho_, sigma, work, &info_);         \
  return info_;                                                                    \
}                                                                                  \
static void thunk_##OP##_cblas_to_fortran(                                         \
    int *n, TYPE *d, TYPE *z, TYPE *delta, TYPE *rho, TYPE *sigma, TYPE *work,   \
    int *info) {                                                                    \
  int info_ = -1;                                                                   \
  if (!n || !rho || !sigma || !info) return;                                       \
  if (!g_cblas_fn[OP_ID]) {                                                         \
    *info = -1;                                                                     \
    return;                                                                         \
  }                                                                                 \
  if (*n > 0 && (!d || !z || !delta || !work)) {                                   \
    *info = -1;                                                                     \
    return;                                                                         \
  }                                                                                 \
  info_ = ((int (*)(int, TYPE *, TYPE *, TYPE *, TYPE, TYPE *, TYPE *))           \
      g_cblas_fn[OP_ID])(*n, d, z, delta, *rho, sigma, work);                     \
  *info = info_;                                                                    \
}

#define DEFINE_LAPACK_LASD5_THUNKS(OP, TYPE, OP_ID)                               \
static int thunk_##OP##_fortran_to_cblas(                                         \
    const int i, TYPE *d, TYPE *z, TYPE *delta, const TYPE rho, TYPE *dsigma,    \
    TYPE *work) {                                                                  \
  int i_ = i;                                                                      \
  TYPE rho_ = rho;                                                                 \
  if (!g_fortran_fn[OP_ID]) return -1;                                             \
  if (!dsigma) return -1;                                                          \
  if (!d || !z || !delta || !work) return -1;                                     \
  ((void (*)(int *, TYPE *, TYPE *, TYPE *, TYPE *, TYPE *, TYPE *))              \
      g_fortran_fn[OP_ID])(&i_, d, z, delta, &rho_, dsigma, work);                \
  return 0;                                                                        \
}                                                                                  \
static void thunk_##OP##_cblas_to_fortran(                                         \
    int *i, TYPE *d, TYPE *z, TYPE *delta, TYPE *rho, TYPE *dsigma, TYPE *work) { \
  if (!g_cblas_fn[OP_ID]) return;                                                  \
  if (!i || !rho || !dsigma) return;                                               \
  if (!d || !z || !delta || !work) return;                                         \
  (void)((int (*)(int, TYPE *, TYPE *, TYPE *, TYPE, TYPE *, TYPE *))             \
      g_cblas_fn[OP_ID])(*i, d, z, delta, *rho, dsigma, work);                    \
}

#define DEFINE_LAPACK_LASD6_THUNKS(OP, TYPE, OP_ID)                               \
static int thunk_##OP##_fortran_to_cblas(                                         \
    const int icompq, const int nl, const int nr, const int sqre, TYPE *d,       \
    TYPE *vf, TYPE *vl, TYPE *alpha, TYPE *beta, int *idxq, int *perm,           \
    int *givptr, int *givcol, const int ldgcol, TYPE *givnum, const int ldgnum,  \
    TYPE *poles, TYPE *difl, TYPE *difr, TYPE *z, int *k, TYPE *c, TYPE *s,      \
    TYPE *work, int *iwork) {                                                     \
  int icompq_ = icompq;                                                            \
  int nl_ = nl;                                                                    \
  int nr_ = nr;                                                                    \
  int sqre_ = sqre;                                                                \
  int ldgcol_ = ldgcol;                                                            \
  int ldgnum_ = ldgnum;                                                            \
  int info_ = 0;                                                                   \
  if (!g_fortran_fn[OP_ID]) return -1;                                             \
  if (!alpha || !beta || !givptr || !k || !c || !s) return -1;                    \
  if (nl + nr > 0 && (!d || !vf || !vl || !idxq || !perm || !givcol || !givnum ||\
                      !poles || !difl || !difr || !z || !work || !iwork)) {      \
    return -1;                                                                     \
  }                                                                                \
  ((void (*)(int *, int *, int *, int *, TYPE *, TYPE *, TYPE *, TYPE *, TYPE *,  \
             int *, int *, int *, int *, int *, TYPE *, int *, TYPE *, TYPE *,    \
             TYPE *, TYPE *, int *, TYPE *, TYPE *, TYPE *, int *, int *))       \
      g_fortran_fn[OP_ID])(&icompq_, &nl_, &nr_, &sqre_, d, vf, vl, alpha, beta, \
                           idxq, perm, givptr, givcol, &ldgcol_, givnum,          \
                           &ldgnum_, poles, difl, difr, z, k, c, s, work, iwork,  \
                           &info_);                                                \
  return info_;                                                                    \
}                                                                                  \
static void thunk_##OP##_cblas_to_fortran(                                         \
    int *icompq, int *nl, int *nr, int *sqre, TYPE *d, TYPE *vf, TYPE *vl,       \
    TYPE *alpha, TYPE *beta, int *idxq, int *perm, int *givptr, int *givcol,     \
    int *ldgcol, TYPE *givnum, int *ldgnum, TYPE *poles, TYPE *difl,             \
    TYPE *difr, TYPE *z, int *k, TYPE *c, TYPE *s, TYPE *work, int *iwork,       \
    int *info) {                                                                    \
  int info_ = -1;                                                                   \
  if (!icompq || !nl || !nr || !sqre || !alpha || !beta || !givptr || !ldgcol || \
      !ldgnum || !k || !c || !s || !info) return;                                  \
  if (!g_cblas_fn[OP_ID]) {                                                         \
    *info = -1;                                                                     \
    return;                                                                         \
  }                                                                                 \
  if (*nl + *nr > 0 && (!d || !vf || !vl || !idxq || !perm || !givcol ||          \
                        !givnum || !poles || !difl || !difr || !z || !work ||     \
                        !iwork)) {                                                  \
    *info = -1;                                                                     \
    return;                                                                         \
  }                                                                                 \
  info_ = ((int (*)(int, int, int, int, TYPE *, TYPE *, TYPE *, TYPE *, TYPE *,   \
                    int *, int *, int *, int *, int, TYPE *, int, TYPE *, TYPE *, \
                    TYPE *, TYPE *, int *, TYPE *, TYPE *, TYPE *, int *))        \
      g_cblas_fn[OP_ID])(*icompq, *nl, *nr, *sqre, d, vf, vl, alpha, beta, idxq,  \
                         perm, givptr, givcol, *ldgcol, givnum, *ldgnum, poles,   \
                         difl, difr, z, k, c, s, work, iwork);                     \
  *info = info_;                                                                    \
}

#define DEFINE_LAPACK_LASD7_THUNKS(OP, TYPE, OP_ID)                               \
static int thunk_##OP##_fortran_to_cblas(                                         \
    const int icompq, const int nl, const int nr, const int sqre, int *k,        \
    TYPE *d, TYPE *z, TYPE *zw, TYPE *vf, TYPE *vfw, TYPE *vl, TYPE *vlw,        \
    const TYPE alpha, const TYPE beta, TYPE *dsigma, int *idx, int *idxp,        \
    int *idxq, int *perm, int *givptr, int *givcol, const int ldgcol,            \
    TYPE *givnum, const int ldgnum, TYPE *c, TYPE *s) {                           \
  int icompq_ = icompq;                                                            \
  int nl_ = nl;                                                                    \
  int nr_ = nr;                                                                    \
  int sqre_ = sqre;                                                                \
  TYPE alpha_ = alpha;                                                             \
  TYPE beta_ = beta;                                                               \
  int ldgcol_ = ldgcol;                                                            \
  int ldgnum_ = ldgnum;                                                            \
  int info_ = 0;                                                                   \
  if (!g_fortran_fn[OP_ID]) return -1;                                             \
  if (!k || !givptr || !c || !s) return -1;                                        \
  if (nl + nr > 0 && (!d || !z || !zw || !vf || !vfw || !vl || !vlw || !dsigma ||\
                      !idx || !idxp || !idxq || !perm || !givcol || !givnum)) {  \
    return -1;                                                                     \
  }                                                                                \
  ((void (*)(int *, int *, int *, int *, int *, TYPE *, TYPE *, TYPE *, TYPE *,   \
             TYPE *, TYPE *, TYPE *, TYPE *, TYPE *, TYPE *, int *, int *, int *, \
             int *, int *, int *, int *, TYPE *, int *, TYPE *, TYPE *, int *))   \
      g_fortran_fn[OP_ID])(&icompq_, &nl_, &nr_, &sqre_, k, d, z, zw, vf, vfw,    \
                           vl, vlw, &alpha_, &beta_, dsigma, idx, idxp, idxq,     \
                           perm, givptr, givcol, &ldgcol_, givnum, &ldgnum_, c,   \
                           s, &info_);                                             \
  return info_;                                                                    \
}                                                                                  \
static void thunk_##OP##_cblas_to_fortran(                                         \
    int *icompq, int *nl, int *nr, int *sqre, int *k, TYPE *d, TYPE *z,          \
    TYPE *zw, TYPE *vf, TYPE *vfw, TYPE *vl, TYPE *vlw, TYPE *alpha, TYPE *beta, \
    TYPE *dsigma, int *idx, int *idxp, int *idxq, int *perm, int *givptr,        \
    int *givcol, int *ldgcol, TYPE *givnum, int *ldgnum, TYPE *c, TYPE *s,       \
    int *info) {                                                                    \
  int info_ = -1;                                                                   \
  if (!icompq || !nl || !nr || !sqre || !k || !alpha || !beta || !givptr ||      \
      !ldgcol || !ldgnum || !c || !s || !info) return;                             \
  if (!g_cblas_fn[OP_ID]) {                                                         \
    *info = -1;                                                                     \
    return;                                                                         \
  }                                                                                 \
  if (*nl + *nr > 0 && (!d || !z || !zw || !vf || !vfw || !vl || !vlw ||         \
                        !dsigma || !idx || !idxp || !idxq || !perm || !givcol || \
                        !givnum)) {                                                \
    *info = -1;                                                                     \
    return;                                                                         \
  }                                                                                 \
  info_ = ((int (*)(int, int, int, int, int *, TYPE *, TYPE *, TYPE *, TYPE *,    \
                    TYPE *, TYPE *, TYPE *, TYPE, TYPE, TYPE *, int *, int *,     \
                    int *, int *, int *, int *, int, TYPE *, int, TYPE *, TYPE *))\
      g_cblas_fn[OP_ID])(*icompq, *nl, *nr, *sqre, k, d, z, zw, vf, vfw, vl, vlw,\
                         *alpha, *beta, dsigma, idx, idxp, idxq, perm, givptr,    \
                         givcol, *ldgcol, givnum, *ldgnum, c, s);                  \
  *info = info_;                                                                    \
}

#define DEFINE_LAPACK_LASD8_THUNKS(OP, TYPE, OP_ID)                               \
static int thunk_##OP##_fortran_to_cblas(                                         \
    const int icompq, const int k, TYPE *d, TYPE *z, TYPE *vf, TYPE *vl,         \
    TYPE *difl, TYPE *difr, const int lddifr, TYPE *dsigma, TYPE *work) {        \
  int icompq_ = icompq;                                                            \
  int k_ = k;                                                                      \
  int lddifr_ = lddifr;                                                            \
  int info_ = 0;                                                                   \
  if (!g_fortran_fn[OP_ID]) return -1;                                             \
  if (!d || !z || !vf || !vl || !difl || !difr || !dsigma || !work) return -1;   \
  ((void (*)(int *, int *, TYPE *, TYPE *, TYPE *, TYPE *, TYPE *, TYPE *, int *, \
             TYPE *, TYPE *, int *))g_fortran_fn[OP_ID])(                         \
      &icompq_, &k_, d, z, vf, vl, difl, difr, &lddifr_, dsigma, work, &info_);   \
  return info_;                                                                    \
}                                                                                  \
static void thunk_##OP##_cblas_to_fortran(                                         \
    int *icompq, int *k, TYPE *d, TYPE *z, TYPE *vf, TYPE *vl, TYPE *difl,       \
    TYPE *difr, int *lddifr, TYPE *dsigma, TYPE *work, int *info) {              \
  int info_ = -1;                                                                   \
  if (!icompq || !k || !lddifr || !info) return;                                   \
  if (!g_cblas_fn[OP_ID]) {                                                         \
    *info = -1;                                                                     \
    return;                                                                         \
  }                                                                                 \
  if (!d || !z || !vf || !vl || !difl || !difr || !dsigma || !work) {             \
    *info = -1;                                                                     \
    return;                                                                         \
  }                                                                                 \
  info_ = ((int (*)(int, int, TYPE *, TYPE *, TYPE *, TYPE *, TYPE *, TYPE *, int,\
                    TYPE *, TYPE *))g_cblas_fn[OP_ID])(                            \
      *icompq, *k, d, z, vf, vl, difl, difr, *lddifr, dsigma, work);               \
  *info = info_;                                                                    \
}

#define DEFINE_LAPACK_LASDA_THUNKS(OP, TYPE, OP_ID)                               \
static int thunk_##OP##_fortran_to_cblas(                                         \
    const int icompq, const int smlsiz, const int n, const int sqre, TYPE *d,    \
    TYPE *e, TYPE *u, const int ldu, TYPE *vt, const int ldvt, int *k,           \
    TYPE *difl, TYPE *difr, TYPE *z, TYPE *zw, TYPE *work, int *iwork) {         \
  int icompq_ = icompq;                                                            \
  int smlsiz_ = smlsiz;                                                            \
  int n_ = n;                                                                      \
  int sqre_ = sqre;                                                                \
  int ldu_ = ldu;                                                                  \
  int ldvt_ = ldvt;                                                                \
  int info_ = 0;                                                                   \
  if (!g_fortran_fn[OP_ID]) return -1;                                             \
  if (!k) return -1;                                                               \
  if (n > 0 && (!d || !e || !u || !vt || !difl || !difr || !z || !zw || !work || \
                !iwork)) {                                                         \
    return -1;                                                                     \
  }                                                                                \
  ((void (*)(int *, int *, int *, int *, TYPE *, TYPE *, TYPE *, int *, TYPE *,   \
             int *, int *, TYPE *, TYPE *, TYPE *, TYPE *, TYPE *, int *, int *))\
      g_fortran_fn[OP_ID])(&icompq_, &smlsiz_, &n_, &sqre_, d, e, u, &ldu_, vt,   \
                           &ldvt_, k, difl, difr, z, zw, work, iwork, &info_);    \
  return info_;                                                                    \
}                                                                                  \
static void thunk_##OP##_cblas_to_fortran(                                         \
    int *icompq, int *smlsiz, int *n, int *sqre, TYPE *d, TYPE *e, TYPE *u,      \
    int *ldu, TYPE *vt, int *ldvt, int *k, TYPE *difl, TYPE *difr, TYPE *z,      \
    TYPE *zw, TYPE *work, int *iwork, int *info) {                                \
  int info_ = -1;                                                                  \
  if (!icompq || !smlsiz || !n || !sqre || !ldu || !ldvt || !k || !info) return; \
  if (!g_cblas_fn[OP_ID]) {                                                        \
    *info = -1;                                                                    \
    return;                                                                        \
  }                                                                                \
  if (*n > 0 && (!d || !e || !u || !vt || !difl || !difr || !z || !zw || !work ||\
                 !iwork)) {                                                        \
    *info = -1;                                                                    \
    return;                                                                        \
  }                                                                                \
  info_ = ((int (*)(int, int, int, int, TYPE *, TYPE *, TYPE *, int, TYPE *, int, \
                    int *, TYPE *, TYPE *, TYPE *, TYPE *, TYPE *, int *))        \
      g_cblas_fn[OP_ID])(*icompq, *smlsiz, *n, *sqre, d, e, u, *ldu, vt, *ldvt,   \
                         k, difl, difr, z, zw, work, iwork);                       \
  *info = info_;                                                                    \
}

#define DEFINE_LAPACK_LASDQ_THUNKS(OP, TYPE, OP_ID)                               \
static int thunk_##OP##_fortran_to_cblas(                                         \
    const char uplo, const int sqre, const int n, const int ncvt, const int nru, \
    const int ncc, TYPE *d, TYPE *e, TYPE *vt, const int ldvt, TYPE *u,          \
    const int ldu, TYPE *c, const int ldc, TYPE *work) {                          \
  char uplo_ = uplo;                                                               \
  int sqre_ = sqre;                                                                \
  int n_ = n;                                                                      \
  int ncvt_ = ncvt;                                                                \
  int nru_ = nru;                                                                  \
  int ncc_ = ncc;                                                                  \
  int ldvt_ = ldvt;                                                                \
  int ldu_ = ldu;                                                                  \
  int ldc_ = ldc;                                                                  \
  int info_ = 0;                                                                   \
  if (!g_fortran_fn[OP_ID]) return -1;                                             \
  if ((n > 0) && (!d || !e)) return -1;                                            \
  if (ncvt > 0 && !vt) return -1;                                                  \
  if (nru > 0 && !u) return -1;                                                    \
  if (ncc > 0 && !c) return -1;                                                    \
  if (!work) return -1;                                                            \
  ((void (*)(char *, int *, int *, int *, int *, int *, TYPE *, TYPE *, TYPE *,   \
             int *, TYPE *, int *, TYPE *, int *, TYPE *, int *))                 \
      g_fortran_fn[OP_ID])(&uplo_, &sqre_, &n_, &ncvt_, &nru_, &ncc_, d, e, vt,   \
                           &ldvt_, u, &ldu_, c, &ldc_, work, &info_);             \
  return info_;                                                                    \
}                                                                                  \
static void thunk_##OP##_cblas_to_fortran(                                         \
    char *uplo, int *sqre, int *n, int *ncvt, int *nru, int *ncc, TYPE *d,       \
    TYPE *e, TYPE *vt, int *ldvt, TYPE *u, int *ldu, TYPE *c, int *ldc,          \
    TYPE *work, int *info) {                                                       \
  int info_ = -1;                                                                  \
  if (!uplo || !sqre || !n || !ncvt || !nru || !ncc || !ldvt || !ldu || !ldc ||  \
      !info) return;                                                               \
  if (!g_cblas_fn[OP_ID]) {                                                        \
    *info = -1;                                                                    \
    return;                                                                        \
  }                                                                                \
  if ((*n > 0) && (!d || !e)) {                                                    \
    *info = -1;                                                                    \
    return;                                                                        \
  }                                                                                \
  if (*ncvt > 0 && !vt) {                                                          \
    *info = -1;                                                                    \
    return;                                                                        \
  }                                                                                \
  if (*nru > 0 && !u) {                                                            \
    *info = -1;                                                                    \
    return;                                                                        \
  }                                                                                \
  if (*ncc > 0 && !c) {                                                            \
    *info = -1;                                                                    \
    return;                                                                        \
  }                                                                                \
  if (!work) {                                                                     \
    *info = -1;                                                                    \
    return;                                                                        \
  }                                                                                \
  info_ = ((int (*)(char, int, int, int, int, int, TYPE *, TYPE *, TYPE *, int,  \
                    TYPE *, int, TYPE *, int, TYPE *))g_cblas_fn[OP_ID])(        \
      *uplo, *sqre, *n, *ncvt, *nru, *ncc, d, e, vt, *ldvt, u, *ldu, c, *ldc,    \
      work);                                                                       \
  *info = info_;                                                                    \
}

#define DEFINE_LAPACK_LASDT_THUNKS(OP, OP_ID)                                     \
static int thunk_##OP##_fortran_to_cblas(                                         \
    const int n, int *lvl, int *lnode, int *inode, int *ndiml, int *ndimr,       \
    const int msub) {                                                              \
  int n_ = n;                                                                      \
  int msub_ = msub;                                                                \
  if (!g_fortran_fn[OP_ID]) return -1;                                             \
  if (!lvl || !lnode) return -1;                                                   \
  if (n > 0 && (!inode || !ndiml || !ndimr)) return -1;                           \
  ((void (*)(int *, int *, int *, int *, int *, int *, int *))g_fortran_fn[OP_ID])(\
      &n_, lvl, lnode, inode, ndiml, ndimr, &msub_);                              \
  return 0;                                                                        \
}                                                                                  \
static void thunk_##OP##_cblas_to_fortran(                                         \
    int *n, int *lvl, int *lnode, int *inode, int *ndiml, int *ndimr,             \
    int *msub) {                                                                   \
  if (!g_cblas_fn[OP_ID]) return;                                                  \
  if (!n || !lvl || !lnode || !msub) return;                                       \
  if (*n > 0 && (!inode || !ndiml || !ndimr)) return;                             \
  (void)((int (*)(int, int *, int *, int *, int *, int *, int))                   \
      g_cblas_fn[OP_ID])(*n, lvl, lnode, inode, ndiml, ndimr, *msub);             \
}

#define DEFINE_LAPACK_LASET_REAL_THUNKS(OP, TYPE, OP_ID)                          \
static int thunk_##OP##_fortran_to_cblas(                                         \
    const char uplo, const int m, const int n, const TYPE alpha, const TYPE beta, \
    TYPE *a, const int lda) {                                                      \
  char uplo_ = uplo;                                                               \
  int m_ = m;                                                                      \
  int n_ = n;                                                                      \
  int lda_ = lda;                                                                  \
  TYPE alpha_ = alpha;                                                             \
  TYPE beta_ = beta;                                                               \
  if (!g_fortran_fn[OP_ID]) return -1;                                             \
  if ((m > 0 || n > 0) && !a) return -1;                                           \
  ((void (*)(char *, int *, int *, TYPE *, TYPE *, TYPE *, int *))                \
      g_fortran_fn[OP_ID])(&uplo_, &m_, &n_, &alpha_, &beta_, a, &lda_);         \
  return 0;                                                                        \
}                                                                                  \
static void thunk_##OP##_cblas_to_fortran(                                         \
    char *uplo, int *m, int *n, TYPE *alpha, TYPE *beta, TYPE *a, int *lda) {    \
  if (!g_cblas_fn[OP_ID]) return;                                                  \
  if (!uplo || !m || !n || !alpha || !beta || !lda) return;                       \
  if ((*m > 0 || *n > 0) && !a) return;                                            \
  (void)((int (*)(char, int, int, TYPE, TYPE, TYPE *, int))                       \
      g_cblas_fn[OP_ID])(*uplo, *m, *n, *alpha, *beta, a, *lda);                  \
}

#define DEFINE_LAPACK_LASET_COMPLEX_THUNKS(OP, TYPE, OP_ID)                       \
static int thunk_##OP##_fortran_to_cblas(                                         \
    const char uplo, const int m, const int n, const TYPE alpha, const TYPE beta, \
    TYPE *a, const int lda) {                                                      \
  char uplo_ = uplo;                                                               \
  int m_ = m;                                                                      \
  int n_ = n;                                                                      \
  int lda_ = lda;                                                                  \
  TYPE alpha_ = alpha;                                                             \
  TYPE beta_ = beta;                                                               \
  if (!g_fortran_fn[OP_ID]) return -1;                                             \
  if ((m > 0 || n > 0) && !a) return -1;                                           \
  ((void (*)(char *, int *, int *, TYPE *, TYPE *, TYPE *, int *))                \
      g_fortran_fn[OP_ID])(&uplo_, &m_, &n_, &alpha_, &beta_, a, &lda_);         \
  return 0;                                                                        \
}                                                                                  \
static void thunk_##OP##_cblas_to_fortran(                                         \
    char *uplo, int *m, int *n, TYPE *alpha, TYPE *beta, TYPE *a, int *lda) {    \
  if (!g_cblas_fn[OP_ID]) return;                                                  \
  if (!uplo || !m || !n || !alpha || !beta || !lda) return;                       \
  if ((*m > 0 || *n > 0) && !a) return;                                            \
  (void)((int (*)(char, int, int, TYPE, TYPE, TYPE *, int))                       \
      g_cblas_fn[OP_ID])(*uplo, *m, *n, *alpha, *beta, a, *lda);                  \
}

#define DEFINE_LAPACK_LASQ1_THUNKS(OP, TYPE, OP_ID)                               \
static int thunk_##OP##_fortran_to_cblas(                                         \
    const int n, TYPE *d, TYPE *e, TYPE *work) {                                  \
  int n_ = n;                                                                      \
  int info_ = 0;                                                                   \
  if (!g_fortran_fn[OP_ID]) return -1;                                             \
  if (n > 0 && (!d || !e || !work)) return -1;                                    \
  ((void (*)(int *, TYPE *, TYPE *, TYPE *, int *))g_fortran_fn[OP_ID])(         \
      &n_, d, e, work, &info_);                                                    \
  return info_;                                                                    \
}                                                                                  \
static void thunk_##OP##_cblas_to_fortran(                                         \
    int *n, TYPE *d, TYPE *e, TYPE *work, int *info) {                            \
  int info_ = -1;                                                                  \
  if (!n || !info) return;                                                         \
  if (!g_cblas_fn[OP_ID]) {                                                        \
    *info = -1;                                                                    \
    return;                                                                        \
  }                                                                                \
  if (*n > 0 && (!d || !e || !work)) {                                             \
    *info = -1;                                                                    \
    return;                                                                        \
  }                                                                                \
  info_ = ((int (*)(int, TYPE *, TYPE *, TYPE *))g_cblas_fn[OP_ID])(             \
      *n, d, e, work);                                                             \
  *info = info_;                                                                   \
}

#define DEFINE_LAPACK_LASQ2_THUNKS(OP, TYPE, OP_ID)                               \
static int thunk_##OP##_fortran_to_cblas(                                         \
    const int n, TYPE *z) {                                                       \
  int n_ = n;                                                                      \
  int info_ = 0;                                                                   \
  if (!g_fortran_fn[OP_ID]) return -1;                                             \
  if (n > 0 && !z) return -1;                                                      \
  ((void (*)(int *, TYPE *, int *))g_fortran_fn[OP_ID])(&n_, z, &info_);         \
  return info_;                                                                    \
}                                                                                  \
static void thunk_##OP##_cblas_to_fortran(                                         \
    int *n, TYPE *z, int *info) {                                                  \
  int info_ = -1;                                                                   \
  if (!n || !info) return;                                                         \
  if (!g_cblas_fn[OP_ID]) {                                                        \
    *info = -1;                                                                    \
    return;                                                                        \
  }                                                                                \
  if (*n > 0 && !z) {                                                              \
    *info = -1;                                                                    \
    return;                                                                        \
  }                                                                                \
  info_ = ((int (*)(int, TYPE *))g_cblas_fn[OP_ID])(*n, z);                      \
  *info = info_;                                                                   \
}

#define DEFINE_LAPACK_LASQ3_THUNKS(OP, TYPE, OP_ID)                               \
static int thunk_##OP##_fortran_to_cblas(                                         \
    const int i0, const int n0, TYPE *z, const int pp, TYPE *dmin, TYPE *sigma,  \
    TYPE *desig, TYPE *qmax, int *nfail, int *iter, int *ndiv, const int ieee,   \
    int *ttype, TYPE *dmin1, TYPE *dmin2, const TYPE dn, const TYPE dn1,         \
    const TYPE dn2, TYPE *g, const TYPE tau) {                                    \
  int i0_ = i0;                                                                    \
  int n0_ = n0;                                                                    \
  int pp_ = pp;                                                                    \
  int ieee_ = ieee;                                                                \
  TYPE dn_ = dn;                                                                   \
  TYPE dn1_ = dn1;                                                                 \
  TYPE dn2_ = dn2;                                                                 \
  TYPE tau_ = tau;                                                                 \
  if (!g_fortran_fn[OP_ID]) return -1;                                             \
  if (!dmin || !sigma || !desig || !qmax || !nfail || !iter || !ndiv ||          \
      !ttype || !dmin1 || !dmin2 || !g) return -1;                                \
  if ((i0 > 0 || n0 > 0) && !z) return -1;                                        \
  ((void (*)(int *, int *, TYPE *, int *, TYPE *, TYPE *, TYPE *, TYPE *,         \
             int *, int *, int *, int *, int *, TYPE *, TYPE *, TYPE *, TYPE *,   \
             TYPE *, TYPE *, TYPE *))g_fortran_fn[OP_ID])(                        \
      &i0_, &n0_, z, &pp_, dmin, sigma, desig, qmax, nfail, iter, ndiv, &ieee_,  \
      ttype, dmin1, dmin2, &dn_, &dn1_, &dn2_, g, &tau_);                         \
  return 0;                                                                        \
}                                                                                  \
static void thunk_##OP##_cblas_to_fortran(                                         \
    int *i0, int *n0, TYPE *z, int *pp, TYPE *dmin, TYPE *sigma, TYPE *desig,    \
    TYPE *qmax, int *nfail, int *iter, int *ndiv, int *ieee, int *ttype,         \
    TYPE *dmin1, TYPE *dmin2, TYPE *dn, TYPE *dn1, TYPE *dn2, TYPE *g,           \
    TYPE *tau) {                                                                   \
  if (!g_cblas_fn[OP_ID]) return;                                                  \
  if (!i0 || !n0 || !pp || !dmin || !sigma || !desig || !qmax || !nfail ||       \
      !iter || !ndiv || !ieee || !ttype || !dmin1 || !dmin2 || !dn || !dn1 ||    \
      !dn2 || !g || !tau) return;                                                  \
  if ((*i0 > 0 || *n0 > 0) && !z) return;                                          \
  (void)((int (*)(int, int, TYPE *, int, TYPE *, TYPE *, TYPE *, TYPE *,          \
                  int *, int *, int *, int, int *, TYPE *, TYPE *, TYPE, TYPE,    \
                  TYPE, TYPE *, TYPE))g_cblas_fn[OP_ID])(                         \
      *i0, *n0, z, *pp, dmin, sigma, desig, qmax, nfail, iter, ndiv, *ieee,      \
      ttype, dmin1, dmin2, *dn, *dn1, *dn2, g, *tau);                             \
}

#define DEFINE_LAPACK_LASQ4_THUNKS(OP, TYPE, OP_ID)                               \
static int thunk_##OP##_fortran_to_cblas(                                         \
    const int i0, const int n0, TYPE *z, const int pp, const int n0in,           \
    const TYPE dmin, const TYPE dmin1, const TYPE dmin2, const TYPE dn,          \
    const TYPE dn1, const TYPE dn2, TYPE *tau, int *ttype, TYPE *g) {            \
  int i0_ = i0;                                                                    \
  int n0_ = n0;                                                                    \
  int pp_ = pp;                                                                    \
  int n0in_ = n0in;                                                                \
  TYPE dmin_ = dmin;                                                               \
  TYPE dmin1_ = dmin1;                                                             \
  TYPE dmin2_ = dmin2;                                                             \
  TYPE dn_ = dn;                                                                   \
  TYPE dn1_ = dn1;                                                                 \
  TYPE dn2_ = dn2;                                                                 \
  if (!g_fortran_fn[OP_ID]) return -1;                                             \
  if (!tau || !ttype || !g) return -1;                                             \
  if ((i0 > 0 || n0 > 0) && !z) return -1;                                        \
  ((void (*)(int *, int *, TYPE *, int *, int *, TYPE *, TYPE *, TYPE *, TYPE *,  \
             TYPE *, TYPE *, TYPE *, int *, TYPE *))g_fortran_fn[OP_ID])(         \
      &i0_, &n0_, z, &pp_, &n0in_, &dmin_, &dmin1_, &dmin2_, &dn_, &dn1_,        \
      &dn2_, tau, ttype, g);                                                       \
  return 0;                                                                        \
}                                                                                  \
static void thunk_##OP##_cblas_to_fortran(                                         \
    int *i0, int *n0, TYPE *z, int *pp, int *n0in, TYPE *dmin, TYPE *dmin1,      \
    TYPE *dmin2, TYPE *dn, TYPE *dn1, TYPE *dn2, TYPE *tau, int *ttype,          \
    TYPE *g) {                                                                     \
  if (!g_cblas_fn[OP_ID]) return;                                                  \
  if (!i0 || !n0 || !pp || !n0in || !dmin || !dmin1 || !dmin2 || !dn ||          \
      !dn1 || !dn2 || !tau || !ttype || !g) return;                               \
  if ((*i0 > 0 || *n0 > 0) && !z) return;                                          \
  (void)((int (*)(int, int, TYPE *, int, int, TYPE, TYPE, TYPE, TYPE, TYPE,       \
                  TYPE, TYPE *, int *, TYPE *))g_cblas_fn[OP_ID])(                \
      *i0, *n0, z, *pp, *n0in, *dmin, *dmin1, *dmin2, *dn, *dn1, *dn2, tau,      \
      ttype, g);                                                                   \
}

#define DEFINE_LAPACK_LASQ5_THUNKS(OP, TYPE, OP_ID)                               \
static int thunk_##OP##_fortran_to_cblas(                                         \
    const int i0, const int n0, TYPE *z, const int pp, const TYPE tau,           \
    TYPE *sigma, TYPE *dmin, TYPE *dmin1, TYPE *dmin2, TYPE *dn, TYPE *dnm1,     \
    TYPE *dnm2, const int ieee) {                                                  \
  int i0_ = i0;                                                                    \
  int n0_ = n0;                                                                    \
  int pp_ = pp;                                                                    \
  TYPE tau_ = tau;                                                                 \
  int ieee_ = ieee;                                                                \
  if (!g_fortran_fn[OP_ID]) return -1;                                             \
  if (!sigma || !dmin || !dmin1 || !dmin2 || !dn || !dnm1 || !dnm2) return -1;  \
  if ((i0 > 0 || n0 > 0) && !z) return -1;                                        \
  ((void (*)(int *, int *, TYPE *, int *, TYPE *, TYPE *, TYPE *, TYPE *, TYPE *, \
             TYPE *, TYPE *, TYPE *, int *))g_fortran_fn[OP_ID])(                 \
      &i0_, &n0_, z, &pp_, &tau_, sigma, dmin, dmin1, dmin2, dn, dnm1, dnm2,    \
      &ieee_);                                                                     \
  return 0;                                                                        \
}                                                                                  \
static void thunk_##OP##_cblas_to_fortran(                                         \
    int *i0, int *n0, TYPE *z, int *pp, TYPE *tau, TYPE *sigma, TYPE *dmin,      \
    TYPE *dmin1, TYPE *dmin2, TYPE *dn, TYPE *dnm1, TYPE *dnm2, int *ieee) {     \
  if (!g_cblas_fn[OP_ID]) return;                                                  \
  if (!i0 || !n0 || !pp || !tau || !sigma || !dmin || !dmin1 || !dmin2 ||        \
      !dn || !dnm1 || !dnm2 || !ieee) return;                                      \
  if ((*i0 > 0 || *n0 > 0) && !z) return;                                          \
  (void)((int (*)(int, int, TYPE *, int, TYPE, TYPE *, TYPE *, TYPE *, TYPE *,    \
                  TYPE *, TYPE *, TYPE *, int))g_cblas_fn[OP_ID])(                \
      *i0, *n0, z, *pp, *tau, sigma, dmin, dmin1, dmin2, dn, dnm1, dnm2, *ieee); \
}

#define DEFINE_LAPACK_LASQ6_THUNKS(OP, TYPE, OP_ID)                               \
static int thunk_##OP##_fortran_to_cblas(                                         \
    const int i0, const int n0, TYPE *z, const int pp) {                         \
  int i0_ = i0;                                                                    \
  int n0_ = n0;                                                                    \
  int pp_ = pp;                                                                    \
  if (!g_fortran_fn[OP_ID]) return -1;                                             \
  if ((i0 > 0 || n0 > 0) && !z) return -1;                                        \
  return ((int (*)(int *, int *, TYPE *, int *))g_fortran_fn[OP_ID])(            \
      &i0_, &n0_, z, &pp_);                                                        \
}                                                                                  \
static int thunk_##OP##_cblas_to_fortran(                                          \
    int *i0, int *n0, TYPE *z, int *pp) {                                          \
  if (!g_cblas_fn[OP_ID]) return -1;                                               \
  if (!i0 || !n0 || !pp) return -1;                                                \
  if ((*i0 > 0 || *n0 > 0) && !z) return -1;                                      \
  return ((int (*)(int, int, TYPE *, int))g_cblas_fn[OP_ID])(                    \
      *i0, *n0, z, *pp);                                                           \
}

#define DEFINE_LAPACK_LASR_THUNKS(OP, TYPE, OP_ID)                                \
static int thunk_##OP##_fortran_to_cblas(                                         \
    const char side, const char pivot, const char direct, const int m,           \
    const int n, const TYPE *c, const TYPE *s, TYPE *a, const int lda) {        \
  char side_ = side;                                                               \
  char pivot_ = pivot;                                                             \
  char direct_ = direct;                                                           \
  int m_ = m;                                                                      \
  int n_ = n;                                                                      \
  int lda_ = lda;                                                                  \
  if (!g_fortran_fn[OP_ID]) return -1;                                             \
  if (m > 0 && n > 0 && (!c || !s || !a)) return -1;                              \
  ((void (*)(const char *, const char *, const char *, const int *, const int *,  \
             const TYPE *, const TYPE *, TYPE *, const int *))g_fortran_fn[OP_ID])(\
      &side_, &pivot_, &direct_, &m_, &n_, c, s, a, &lda_);                       \
  return 0;                                                                        \
}                                                                                  \
static void thunk_##OP##_cblas_to_fortran(                                         \
    const char *side, const char *pivot, const char *direct, const int *m,       \
    const int *n, const TYPE *c, const TYPE *s, TYPE *a, const int *lda) {       \
  if (!g_cblas_fn[OP_ID]) return;                                                  \
  if (!side || !pivot || !direct || !m || !n || !lda) return;                     \
  if (*m > 0 && *n > 0 && (!c || !s || !a)) return;                               \
  (void)((int (*)(char, char, char, int, int, const TYPE *, const TYPE *, TYPE *, \
                  int))g_cblas_fn[OP_ID])(                                        \
      *side, *pivot, *direct, *m, *n, c, s, a, *lda);                             \
}

#define DEFINE_LAPACK_LARZ_THUNKS(OP, TYPE, OP_ID)                             \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const char side, const int m, const int n, const int l, TYPE *v,           \
    const int incv, TYPE *tau, TYPE *c, const int ldc, TYPE *work) {           \
  char side_ = side;                                                           \
  int m_ = m;                                                                  \
  int n_ = n;                                                                  \
  int l_ = l;                                                                  \
  int incv_ = incv;                                                            \
  int ldc_ = ldc;                                                              \
  if (!g_fortran_fn[OP_ID]) return -1;                                         \
  if (!tau) return -1;                                                         \
  if ((m > 0 || n > 0 || l > 0) && (!v || !c || !work)) return -1;             \
  ((void (*)(char *, int *, int *, int *, TYPE *, int *, TYPE *, TYPE *,       \
             int *, TYPE *))g_fortran_fn[OP_ID])(                              \
      &side_, &m_, &n_, &l_, v, &incv_, tau, c, &ldc_, work);                  \
  return 0;                                                                    \
}                                                                              \
static void thunk_##OP##_cblas_to_fortran(                                     \
    char *side, int *m, int *n, int *l, TYPE *v, int *incv, TYPE *tau,         \
    TYPE *c, int *ldc, TYPE *work) {                                           \
  if (!g_cblas_fn[OP_ID]) return;                                              \
  if (!side || !m || !n || !l || !incv || !tau || !ldc) return;                \
  if ((*m > 0 || *n > 0 || *l > 0) && (!v || !c || !work)) return;             \
  (void)((int (*)(char, int, int, int, TYPE *, int, TYPE *, TYPE *, int,       \
                  TYPE *))g_cblas_fn[OP_ID])(                                  \
      *side, *m, *n, *l, v, *incv, tau, c, *ldc, work);                        \
}

#define DEFINE_LAPACK_LARZB_THUNKS(OP, TYPE, OP_ID)                            \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const char side, const char trans, const char direct, const char storev,   \
    const int m, const int n, const int k, const int l, TYPE *v,               \
    const int ldv, TYPE *t, const int ldt, TYPE *c, const int ldc,             \
    TYPE *work, const int ldwork) {                                            \
  char side_ = side;                                                           \
  char trans_ = trans;                                                         \
  char direct_ = direct;                                                       \
  char storev_ = storev;                                                       \
  int m_ = m;                                                                  \
  int n_ = n;                                                                  \
  int k_ = k;                                                                  \
  int l_ = l;                                                                  \
  int ldv_ = ldv;                                                              \
  int ldt_ = ldt;                                                              \
  int ldc_ = ldc;                                                              \
  int ldwork_ = ldwork;                                                        \
  if (!g_fortran_fn[OP_ID]) return -1;                                         \
  if ((m > 0 || n > 0 || k > 0 || l > 0) && (!v || !t || !c || !work)) {      \
    return -1;                                                                 \
  }                                                                            \
  ((void (*)(char *, char *, char *, char *, int *, int *, int *, int *,       \
             TYPE *, int *, TYPE *, int *, TYPE *, int *, TYPE *, int *))      \
       g_fortran_fn[OP_ID])(                                                   \
      &side_, &trans_, &direct_, &storev_, &m_, &n_, &k_, &l_,                \
      v, &ldv_, t, &ldt_, c, &ldc_, work, &ldwork_);                           \
  return 0;                                                                    \
}                                                                              \
static void thunk_##OP##_cblas_to_fortran(                                     \
    char *side, char *trans, char *direct, char *storev, int *m, int *n,       \
    int *k, int *l, TYPE *v, int *ldv, TYPE *t, int *ldt, TYPE *c, int *ldc,  \
    TYPE *work, int *ldwork) {                                                 \
  if (!g_cblas_fn[OP_ID]) return;                                              \
  if (!side || !trans || !direct || !storev || !m || !n || !k || !l ||        \
      !ldv || !ldt || !ldc || !ldwork) return;                                 \
  if ((*m > 0 || *n > 0 || *k > 0 || *l > 0) && (!v || !t || !c || !work)) {  \
    return;                                                                    \
  }                                                                            \
  (void)((int (*)(char, char, char, char, int, int, int, int, TYPE *, int,    \
                  TYPE *, int, TYPE *, int, TYPE *, int))g_cblas_fn[OP_ID])(  \
      *side, *trans, *direct, *storev, *m, *n, *k, *l, v, *ldv,               \
      t, *ldt, c, *ldc, work, *ldwork);                                        \
}

#define DEFINE_LAPACK_LARZT_THUNKS(OP, TYPE, OP_ID)                            \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const char direct, const char storev, const int n, const int k, TYPE *v,   \
    const int ldv, TYPE *tau, TYPE *t, const int ldt) {                        \
  char direct_ = direct;                                                       \
  char storev_ = storev;                                                       \
  int n_ = n;                                                                  \
  int k_ = k;                                                                  \
  int ldv_ = ldv;                                                              \
  int ldt_ = ldt;                                                              \
  if (!g_fortran_fn[OP_ID]) return -1;                                         \
  if ((n > 0 || k > 0) && (!v || !tau || !t)) return -1;                       \
  ((void (*)(char *, char *, int *, int *, TYPE *, int *, TYPE *, TYPE *,      \
             int *))g_fortran_fn[OP_ID])(                                      \
      &direct_, &storev_, &n_, &k_, v, &ldv_, tau, t, &ldt_);                 \
  return 0;                                                                    \
}                                                                              \
static void thunk_##OP##_cblas_to_fortran(                                     \
    char *direct, char *storev, int *n, int *k, TYPE *v, int *ldv, TYPE *tau, \
    TYPE *t, int *ldt) {                                                       \
  if (!g_cblas_fn[OP_ID]) return;                                              \
  if (!direct || !storev || !n || !k || !ldv || !ldt) return;                  \
  if ((*n > 0 || *k > 0) && (!v || !tau || !t)) return;                        \
  (void)((int (*)(char, char, int, int, TYPE *, int, TYPE *, TYPE *, int))     \
             g_cblas_fn[OP_ID])(                                               \
      *direct, *storev, *n, *k, v, *ldv, tau, t, *ldt);                        \
}

#define DEFINE_LAPACK_LARFT_REAL_THUNKS(OP, TYPE, OP_ID)                       \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const char direct, const char storev,            \
    const int n, const int k, const TYPE *v, const int ldv, const TYPE *tau,   \
    TYPE *t, const int ldt) {                                                  \
  char direct_ = direct;                                                       \
  char storev_ = storev;                                                       \
  int n_ = n;                                                                  \
  int k_ = k;                                                                  \
  int nrows_v = fb_lapack_job_is(storev, 'C') ? n :                           \
                (fb_lapack_job_is(storev, 'R') ? k : 1);                       \
  int ncols_v = fb_lapack_job_is(storev, 'C') ? k :                           \
                (fb_lapack_job_is(storev, 'R') ? n : 1);                       \
  int ldv_ = (nrows_v > 0) ? nrows_v : 1;                                      \
  int ldt_ = (k > 0) ? k : 1;                                                  \
  int info = 0;                                                                \
  size_t v_elems = (size_t)ldv_ * (size_t)((ncols_v > 0) ? ncols_v : 1);       \
  size_t t_elems = (size_t)ldt_ * (size_t)((k > 0) ? k : 1);                   \
  TYPE *v_col = NULL;                                                          \
  TYPE *t_col = NULL;                                                          \
  if (!g_fortran_fn[OP_ID]) return -1;                                         \
  if ((k > 0 && (!tau || !t)) || (nrows_v > 0 && ncols_v > 0 && !v)) return -1; \
  if (fb_lapack_layout_is_col_major(layout)) {                                 \
    ((void (*)(char *, char *, int *, int *, const TYPE *, int *,              \
               const TYPE *, TYPE *, int *, int *))g_fortran_fn[OP_ID])(       \
        &direct_, &storev_, &n_, &k_, v, (int *)&ldv, tau, t, (int *)&ldt, &info); \
    return info;                                                               \
  }                                                                            \
  if (layout != FB_LAYOUT_ROW_MAJOR) return -1;                                \
  if (ldt < k) return -10;                                                     \
  if (ldv < ncols_v) return -7;                                                \
  v_col = (TYPE *)malloc(v_elems * sizeof(TYPE));                               \
  t_col = (TYPE *)malloc(t_elems * sizeof(TYPE));                               \
  if (!v_col || !t_col) {                                                      \
    free(v_col);                                                               \
    free(t_col);                                                               \
    return -101;                                                               \
  }                                                                            \
  if ((nrows_v > 0 && ncols_v > 0 &&                                           \
       fb_copy_matrix_to_col_major(v, layout, nrows_v, ncols_v, ldv,          \
                                   v_col, ldv_, sizeof(TYPE)) != 0)) {        \
    free(v_col);                                                               \
    free(t_col);                                                               \
    return -1;                                                                 \
  }                                                                            \
  ((void (*)(char *, char *, int *, int *, const TYPE *, int *,               \
             const TYPE *, TYPE *, int *, int *))g_fortran_fn[OP_ID])(         \
      &direct_, &storev_, &n_, &k_, v_col, &ldv_, tau, t_col, &ldt_, &info);  \
  if (info != 0) {                                                             \
    free(v_col);                                                               \
    free(t_col);                                                               \
    return info;                                                               \
  }                                                                            \
  if (k > 0 &&                                                                 \
      fb_copy_matrix_from_col_major(t_col, k, k, ldt_, layout,                \
                                    t, ldt, sizeof(TYPE)) != 0) {             \
    free(v_col);                                                               \
    free(t_col);                                                               \
    return -1;                                                                 \
  }                                                                            \
  free(v_col);                                                                 \
  free(t_col);                                                                 \
  return 0;                                                                    \
}                                                                              \
static void thunk_##OP##_cblas_to_fortran(                                     \
    char *direct, char *storev, int *n, int *k, TYPE *v, int *ldv,            \
    TYPE *tau, TYPE *t, int *ldt, int *info) {                                 \
  int nrows_v = 0;                                                             \
  int ncols_v = 0;                                                             \
  if (!g_cblas_fn[OP_ID]) {                                                    \
    if (info) *info = -1;                                                      \
    return;                                                                    \
  }                                                                            \
  if (!direct || !storev || !n || !k || !ldv || !ldt || !info) {              \
    return;                                                                    \
  }                                                                            \
  nrows_v = fb_lapack_job_is(*storev, 'C') ? *n :                              \
            (fb_lapack_job_is(*storev, 'R') ? *k : 1);                         \
  ncols_v = fb_lapack_job_is(*storev, 'C') ? *k :                              \
            (fb_lapack_job_is(*storev, 'R') ? *n : 1);                         \
  if ((*k > 0 && (!tau || !t)) || (nrows_v > 0 && ncols_v > 0 && !v)) {       \
    *info = -1;                                                                \
    return;                                                                    \
  }                                                                            \
  *info = ((int (*)(fb_layout_t, char, char, int, int, const TYPE *, int,     \
                    const TYPE *, TYPE *, int))g_cblas_fn[OP_ID])(             \
      FB_LAYOUT_COL_MAJOR, *direct, *storev, *n, *k, v, *ldv, tau, t, *ldt);  \
}

#define DEFINE_LAPACK_LARFT_COMPLEX_THUNKS(OP, TYPE, OP_ID)                    \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const char direct, const char storev,            \
    const int n, const int k, const TYPE *v, const int ldv, const TYPE *tau,   \
    TYPE *t, const int ldt) {                                                  \
  char direct_ = direct;                                                       \
  char storev_ = storev;                                                       \
  int n_ = n;                                                                  \
  int k_ = k;                                                                  \
  int nrows_v = fb_lapack_job_is(storev, 'C') ? n :                           \
                (fb_lapack_job_is(storev, 'R') ? k : 1);                       \
  int ncols_v = fb_lapack_job_is(storev, 'C') ? k :                           \
                (fb_lapack_job_is(storev, 'R') ? n : 1);                       \
  int ldv_ = (nrows_v > 0) ? nrows_v : 1;                                      \
  int ldt_ = (k > 0) ? k : 1;                                                  \
  size_t v_elems = (size_t)ldv_ * (size_t)((ncols_v > 0) ? ncols_v : 1);       \
  size_t t_elems = (size_t)ldt_ * (size_t)((k > 0) ? k : 1);                   \
  TYPE *v_col = NULL;                                                          \
  TYPE *t_col = NULL;                                                          \
  if (!g_fortran_fn[OP_ID]) return -1;                                         \
  if ((k > 0 && (!tau || !t)) || (nrows_v > 0 && ncols_v > 0 && !v)) return -1; \
  if (fb_lapack_layout_is_col_major(layout)) {                                 \
    ((void (*)(char *, char *, int *, int *, const TYPE *, int *,              \
               const TYPE *, TYPE *, int *))g_fortran_fn[OP_ID])(              \
        &direct_, &storev_, &n_, &k_, v, (int *)&ldv, tau, t, (int *)&ldt);   \
    return 0;                                                                  \
  }                                                                            \
  if (layout != FB_LAYOUT_ROW_MAJOR) return -1;                                \
  if (ldt < k) return -10;                                                     \
  if (ldv < ncols_v) return -7;                                                \
  v_col = (TYPE *)malloc(v_elems * sizeof(TYPE));                               \
  t_col = (TYPE *)malloc(t_elems * sizeof(TYPE));                               \
  if (!v_col || !t_col) {                                                      \
    free(v_col);                                                               \
    free(t_col);                                                               \
    return -101;                                                               \
  }                                                                            \
  if ((nrows_v > 0 && ncols_v > 0 &&                                           \
       fb_copy_matrix_to_col_major(v, layout, nrows_v, ncols_v, ldv,          \
                                   v_col, ldv_, sizeof(TYPE)) != 0)) {        \
    free(v_col);                                                               \
    free(t_col);                                                               \
    return -1;                                                                 \
  }                                                                            \
  ((void (*)(char *, char *, int *, int *, const TYPE *, int *,               \
             const TYPE *, TYPE *, int *))g_fortran_fn[OP_ID])(                \
      &direct_, &storev_, &n_, &k_, v_col, &ldv_, tau, t_col, &ldt_);         \
  if (k > 0 &&                                                                 \
      fb_copy_matrix_from_col_major(t_col, k, k, ldt_, layout,                \
                                    t, ldt, sizeof(TYPE)) != 0) {             \
    free(v_col);                                                               \
    free(t_col);                                                               \
    return -1;                                                                 \
  }                                                                            \
  free(v_col);                                                                 \
  free(t_col);                                                                 \
  return 0;                                                                    \
}                                                                              \
static void thunk_##OP##_cblas_to_fortran(                                     \
    char *direct, char *storev, int *n, int *k, TYPE *v, int *ldv,            \
    TYPE *tau, TYPE *t, int *ldt) {                                            \
  int nrows_v = 0;                                                             \
  int ncols_v = 0;                                                             \
  if (!g_cblas_fn[OP_ID]) return;                                              \
  if (!direct || !storev || !n || !k || !ldv || !ldt) return;                  \
  nrows_v = fb_lapack_job_is(*storev, 'C') ? *n :                              \
            (fb_lapack_job_is(*storev, 'R') ? *k : 1);                         \
  ncols_v = fb_lapack_job_is(*storev, 'C') ? *k :                              \
            (fb_lapack_job_is(*storev, 'R') ? *n : 1);                         \
  if ((*k > 0 && (!tau || !t)) || (nrows_v > 0 && ncols_v > 0 && !v)) {       \
    return;                                                                    \
  }                                                                            \
  (void)((int (*)(fb_layout_t, char, char, int, int, const TYPE *, int,       \
                  const TYPE *, TYPE *, int))g_cblas_fn[OP_ID])(               \
      FB_LAYOUT_COL_MAJOR, *direct, *storev, *n, *k, v, *ldv, tau, t, *ldt);  \
}

#define DEFINE_LAPACK_LARFX_REAL_THUNKS(OP, TYPE, OP_ID)                       \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const char side, const int m, const int n,       \
    const TYPE *v, const TYPE tau, TYPE *c, const int ldc, TYPE *work) {       \
  char side_ = side;                                                            \
  int m_ = m;                                                                   \
  int n_ = n;                                                                   \
  int ldc_ = (m > 0) ? m : 1;                                                   \
  int info = 0;                                                                 \
  TYPE tau_ = tau;                                                              \
  size_t c_elems = (size_t)ldc_ * (size_t)((n > 0) ? n : 1);                   \
  TYPE *c_col = NULL;                                                           \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if ((m > 0 || n > 0) && (!v || !c || !work)) return -1;                      \
  if (fb_lapack_layout_is_col_major(layout)) {                                  \
    ((void (*)(char *, int *, int *, TYPE *, TYPE *, TYPE *, int *, TYPE *,    \
               int *))g_fortran_fn[OP_ID])(                                     \
        &side_, &m_, &n_, (TYPE *)v, &tau_, c, &ldc_, work, &info);            \
    return info;                                                                \
  }                                                                             \
  if (layout != FB_LAYOUT_ROW_MAJOR) return -1;                                 \
  if (ldc < n) return -8;                                                       \
  c_col = (TYPE *)malloc(c_elems * sizeof(TYPE));                               \
  if (!c_col) return -101;                                                      \
  if (m > 0 && n > 0 &&                                                         \
      fb_copy_matrix_to_col_major(c, layout, m, n, ldc,                        \
                                  c_col, ldc_, sizeof(TYPE)) != 0) {           \
    free(c_col);                                                                \
    return -1;                                                                  \
  }                                                                             \
  ((void (*)(char *, int *, int *, TYPE *, TYPE *, TYPE *, int *, TYPE *,      \
             int *))g_fortran_fn[OP_ID])(                                       \
      &side_, &m_, &n_, (TYPE *)v, &tau_, c_col, &ldc_, work, &info);          \
  if (m > 0 && n > 0 &&                                                         \
      fb_copy_matrix_from_col_major(c_col, m, n, ldc_, layout,                 \
                                    c, ldc, sizeof(TYPE)) != 0) {              \
    free(c_col);                                                                \
    return -1;                                                                  \
  }                                                                             \
  free(c_col);                                                                  \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    char *side, int *m, int *n, TYPE *v, TYPE *tau, TYPE *c, int *ldc,         \
    TYPE *work, int *info) {                                                    \
  TYPE tau_ = (TYPE)0;                                                          \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!side || !m || !n || !tau || !ldc || !info) {                            \
    return;                                                                     \
  }                                                                             \
  if ((*m > 0 || *n > 0) && (!v || !c || !work)) {                             \
    *info = -1;                                                                 \
    return;                                                                     \
  }                                                                             \
  tau_ = *tau;                                                                  \
  *info = ((int (*)(fb_layout_t, char, int, int, const TYPE *, TYPE, TYPE *,   \
                    int, TYPE *))g_cblas_fn[OP_ID])(                            \
      FB_LAYOUT_COL_MAJOR, *side, *m, *n, v, tau_, c, *ldc, work);             \
}

#define DEFINE_LAPACK_LARFX_COMPLEX_THUNKS(OP, TYPE, OP_ID)                    \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const char side, const int m, const int n,       \
    const TYPE *v, const TYPE tau, TYPE *c, const int ldc, TYPE *work) {       \
  char side_ = side;                                                            \
  char trans_ = 'N';                                                            \
  int m_ = m;                                                                   \
  int n_ = n;                                                                   \
  int k_ = fb_lapack_job_is(side, 'L') ? m :                                   \
            (fb_lapack_job_is(side, 'R') ? n : 0);                              \
  int incv_ = 1;                                                                \
  int ldc_ = (m > 0) ? m : 1;                                                   \
  TYPE tau_ = tau;                                                              \
  size_t c_elems = (size_t)ldc_ * (size_t)((n > 0) ? n : 1);                   \
  TYPE *c_col = NULL;                                                           \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (!fb_lapack_job_is(side, 'L') && !fb_lapack_job_is(side, 'R')) return -1; \
  if ((m > 0 || n > 0) && (!v || !c || !work)) return -1;                      \
  if (fb_lapack_layout_is_col_major(layout)) {                                  \
    ((void (*)(char *, char *, int *, int *, int *, TYPE *, int *, TYPE *,     \
               TYPE *, int *, TYPE *))g_fortran_fn[OP_ID])(                     \
        &side_, &trans_, &m_, &n_, &k_, (TYPE *)v, &incv_, &tau_, c, &ldc_,    \
        work);                                                                  \
    return 0;                                                                   \
  }                                                                             \
  if (layout != FB_LAYOUT_ROW_MAJOR) return -1;                                 \
  if (ldc < n) return -8;                                                       \
  c_col = (TYPE *)malloc(c_elems * sizeof(TYPE));                               \
  if (!c_col) return -101;                                                      \
  if (m > 0 && n > 0 &&                                                         \
      fb_copy_matrix_to_col_major(c, layout, m, n, ldc,                        \
                                  c_col, ldc_, sizeof(TYPE)) != 0) {           \
    free(c_col);                                                                \
    return -1;                                                                  \
  }                                                                             \
  ((void (*)(char *, char *, int *, int *, int *, TYPE *, int *, TYPE *,       \
             TYPE *, int *, TYPE *))g_fortran_fn[OP_ID])(                       \
      &side_, &trans_, &m_, &n_, &k_, (TYPE *)v, &incv_, &tau_, c_col, &ldc_,  \
      work);                                                                    \
  if (m > 0 && n > 0 &&                                                         \
      fb_copy_matrix_from_col_major(c_col, m, n, ldc_, layout,                 \
                                    c, ldc, sizeof(TYPE)) != 0) {              \
    free(c_col);                                                                \
    return -1;                                                                  \
  }                                                                             \
  free(c_col);                                                                  \
  return 0;                                                                     \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    char *side, char *trans, int *m, int *n, int *k, TYPE *v, int *incv,       \
    TYPE *tau, TYPE *c, int *ldc, TYPE *work) {                                 \
  int vec_len = 0;                                                              \
  TYPE tau_ = (TYPE)0;                                                          \
  TYPE *v_copy = NULL;                                                          \
  TYPE *v_arg = v;                                                              \
  if (!g_cblas_fn[OP_ID]) return;                                               \
  if (!side || !trans || !m || !n || !k || !incv || !tau || !ldc) return;      \
  if (*m <= 0 || *n <= 0 || *k <= 0) return;                                    \
  if (!fb_lapack_job_is(*side, 'L') && !fb_lapack_job_is(*side, 'R')) return;  \
  if (!fb_lapack_job_is(*trans, 'N') && !fb_lapack_job_is(*trans, 'C')) return;\
  if (*incv <= 0 || !v || !c || !work) return;                                  \
  vec_len = fb_lapack_job_is(*side, 'L') ? *m : *n;                             \
  tau_ = *tau;                                                                  \
  if (fb_lapack_job_is(*trans, 'C')) {                                          \
    __imag__ tau_ = -__imag__ tau_;                                             \
  }                                                                             \
  if (*incv != 1) {                                                             \
    v_copy = (TYPE *)malloc((size_t)vec_len * sizeof(TYPE));                    \
    if (!v_copy) return;                                                        \
    if (fb_copy_strided_vector(v, vec_len, *incv, v_copy, sizeof(TYPE)) != 0) {\
      free(v_copy);                                                             \
      return;                                                                   \
    }                                                                           \
    v_arg = v_copy;                                                             \
  }                                                                             \
  (void)((int (*)(fb_layout_t, char, int, int, const TYPE *, TYPE, TYPE *,     \
                  int, TYPE *))g_cblas_fn[OP_ID])(                              \
      FB_LAYOUT_COL_MAJOR, *side, *m, *n, v_arg, tau_, c, *ldc, work);         \
  free(v_copy);                                                                 \
}

#define DEFINE_LAPACK_LAEXC_THUNKS(OP, TYPE, OP_ID)                            \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const int wantq, const int n, TYPE *t, const int ldt, TYPE *q,             \
    const int ldq, const int j1, const int n1, const int n2, TYPE *work) {    \
  int wantq_ = wantq;                                                          \
  int n_ = n;                                                                  \
  int ldt_ = ldt;                                                              \
  int ldq_ = ldq;                                                              \
  int j1_ = j1;                                                                \
  int n1_ = n1;                                                                \
  int n2_ = n2;                                                                \
  int info_ = 0;                                                               \
  if (!g_fortran_fn[OP_ID]) return -1;                                         \
  if ((n > 0 && !t) || (wantq != 0 && n > 0 && !q) || (n > 0 && !work)) {     \
    return -1;                                                                 \
  }                                                                            \
  ((void (*)(int *, int *, TYPE *, int *, TYPE *, int *, int *, int *, int *,  \
             TYPE *, int *))g_fortran_fn[OP_ID])(                              \
      &wantq_, &n_, t, &ldt_, q, &ldq_, &j1_, &n1_, &n2_, work, &info_);       \
  return info_;                                                                \
}                                                                              \
static void thunk_##OP##_cblas_to_fortran(                                     \
    int *wantq, int *n, TYPE *t, int *ldt, TYPE *q, int *ldq, int *j1,         \
    int *n1, int *n2, TYPE *work, int *info) {                                 \
  int info_ = -1;                                                              \
  if (!wantq || !n || !ldt || !ldq || !j1 || !n1 || !n2 || !info) return;     \
  if (!g_cblas_fn[OP_ID]) {                                                    \
    *info = -1;                                                                \
    return;                                                                    \
  }                                                                            \
  if ((*n > 0 && !t) || (*wantq != 0 && *n > 0 && !q) || (*n > 0 && !work)) { \
    *info = -1;                                                                \
    return;                                                                    \
  }                                                                            \
  info_ = ((int (*)(int, int, TYPE *, int, TYPE *, int, int, int, int, TYPE *))\
               g_cblas_fn[OP_ID])(*wantq, *n, t, *ldt, q, *ldq, *j1, *n1,      \
                                  *n2, work);                                  \
  *info = info_;                                                               \
}

#define DEFINE_LAPACK_LAESY_THUNKS(OP, CTYPE, OP_ID)                           \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const CTYPE a, const CTYPE b, const CTYPE c, CTYPE *rt1, CTYPE *rt2,       \
    CTYPE *evscal, CTYPE *cs1, CTYPE *sn1) {                                   \
  CTYPE a_ = a;                                                                \
  CTYPE b_ = b;                                                                \
  CTYPE c_ = c;                                                                \
  if (!g_fortran_fn[OP_ID]) return -1;                                         \
  if (!rt1 || !rt2 || !evscal || !cs1 || !sn1) return -1;                      \
  ((void (*)(CTYPE *, CTYPE *, CTYPE *, CTYPE *, CTYPE *, CTYPE *, CTYPE *,    \
             CTYPE *))g_fortran_fn[OP_ID])(                                    \
      &a_, &b_, &c_, rt1, rt2, evscal, cs1, sn1);                              \
  return 0;                                                                    \
}                                                                              \
static void thunk_##OP##_cblas_to_fortran(                                     \
    CTYPE *a, CTYPE *b, CTYPE *c, CTYPE *rt1, CTYPE *rt2, CTYPE *evscal,       \
    CTYPE *cs1, CTYPE *sn1) {                                                  \
  if (!g_cblas_fn[OP_ID]) return;                                              \
  if (!a || !b || !c || !rt1 || !rt2 || !evscal || !cs1 || !sn1) return;       \
  (void)((int (*)(CTYPE, CTYPE, CTYPE, CTYPE *, CTYPE *, CTYPE *, CTYPE *,     \
                  CTYPE *))g_cblas_fn[OP_ID])(                                 \
      *a, *b, *c, rt1, rt2, evscal, cs1, sn1);                                 \
}

#define DEFINE_LAPACK_LAIC1_THUNKS(OP, TYPE, OP_ID)                            \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const int job, const int j, TYPE *x, const TYPE sest, TYPE *w,             \
    const TYPE gamma, TYPE *sestpr, TYPE *s, TYPE *c__) {                      \
  int job_ = job;                                                              \
  int j_ = j;                                                                  \
  TYPE sest_ = sest;                                                           \
  TYPE gamma_ = gamma;                                                         \
  if (!g_fortran_fn[OP_ID]) return -1;                                         \
  if ((j > 0 && (!x || !w)) || !sestpr || !s || !c__) return -1;               \
  ((void (*)(int *, int *, TYPE *, TYPE *, TYPE *, TYPE *, TYPE *, TYPE *,     \
             TYPE *))g_fortran_fn[OP_ID])(                                     \
      &job_, &j_, x, &sest_, w, &gamma_, sestpr, s, c__);                      \
  return 0;                                                                    \
}                                                                              \
static void thunk_##OP##_cblas_to_fortran(                                     \
    int *job, int *j, TYPE *x, TYPE *sest, TYPE *w, TYPE *gamma,               \
    TYPE *sestpr, TYPE *s, TYPE *c__) {                                        \
  if (!g_cblas_fn[OP_ID]) return;                                              \
  if (!job || !j || !sest || !gamma || !sestpr || !s || !c__) return;          \
  if ((*j > 0 && (!x || !w))) return;                                          \
  (void)((int (*)(int, int, TYPE *, TYPE, TYPE *, TYPE, TYPE *, TYPE *, TYPE *))\
             g_cblas_fn[OP_ID])(*job, *j, x, *sest, w, *gamma, sestpr, s, c__);\
}

#define DEFINE_LAPACK_LAIC1_COMPLEX_THUNKS(OP, CTYPE, RTYPE, OP_ID)            \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const int job, const int j, CTYPE *x, const RTYPE sest, CTYPE *w,          \
    const CTYPE gamma, RTYPE *sestpr, CTYPE *s, CTYPE *c__) {                  \
  int job_ = job;                                                              \
  int j_ = j;                                                                  \
  RTYPE sest_ = sest;                                                          \
  CTYPE gamma_ = gamma;                                                        \
  if (!g_fortran_fn[OP_ID]) return -1;                                         \
  if ((j > 0 && (!x || !w)) || !sestpr || !s || !c__) return -1;               \
  ((void (*)(int *, int *, CTYPE *, RTYPE *, CTYPE *, CTYPE *, RTYPE *,        \
             CTYPE *, CTYPE *))g_fortran_fn[OP_ID])(                           \
      &job_, &j_, x, &sest_, w, &gamma_, sestpr, s, c__);                      \
  return 0;                                                                    \
}                                                                              \
static void thunk_##OP##_cblas_to_fortran(                                     \
    int *job, int *j, CTYPE *x, RTYPE *sest, CTYPE *w, CTYPE *gamma,           \
    RTYPE *sestpr, CTYPE *s, CTYPE *c__) {                                     \
  if (!g_cblas_fn[OP_ID]) return;                                              \
  if (!job || !j || !sest || !gamma || !sestpr || !s || !c__) return;          \
  if ((*j > 0 && (!x || !w))) return;                                          \
  (void)((int (*)(int, int, CTYPE *, RTYPE, CTYPE *, CTYPE, RTYPE *, CTYPE *,  \
                  CTYPE *))g_cblas_fn[OP_ID])(                                 \
      *job, *j, x, *sest, w, *gamma, sestpr, s, c__);                          \
}

#define DEFINE_LAPACK_LALN2_THUNKS(OP, TYPE, OP_ID)                            \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const int ltrans, const int na, const int nw, const TYPE smin,             \
    const TYPE ca, TYPE *a, const int lda, const TYPE d1, const TYPE d2,       \
    TYPE *b, const int ldb, const TYPE wr, const TYPE wi, TYPE *x,             \
    const int ldx, TYPE *scale, TYPE *xnorm) {                                 \
  int ltrans_ = ltrans;                                                        \
  int na_ = na;                                                                \
  int nw_ = nw;                                                                \
  int lda_ = lda;                                                              \
  int ldb_ = ldb;                                                              \
  int ldx_ = ldx;                                                              \
  int info_ = 0;                                                               \
  TYPE smin_ = smin;                                                           \
  TYPE ca_ = ca;                                                               \
  TYPE d1_ = d1;                                                               \
  TYPE d2_ = d2;                                                               \
  TYPE wr_ = wr;                                                               \
  TYPE wi_ = wi;                                                               \
  if (!g_fortran_fn[OP_ID]) return -1;                                         \
  if ((na > 0 && (!a || !b || !x)) || !scale || !xnorm) return -1;             \
  ((void (*)(int *, int *, int *, TYPE *, TYPE *, TYPE *, int *, TYPE *,       \
             TYPE *, TYPE *, int *, TYPE *, TYPE *, TYPE *, int *, TYPE *,     \
             TYPE *, int *))g_fortran_fn[OP_ID])(                              \
      &ltrans_, &na_, &nw_, &smin_, &ca_, a, &lda_, &d1_, &d2_, b, &ldb_,      \
      &wr_, &wi_, x, &ldx_, scale, xnorm, &info_);                             \
  return info_;                                                                \
}                                                                              \
static void thunk_##OP##_cblas_to_fortran(                                     \
    int *ltrans, int *na, int *nw, TYPE *smin, TYPE *ca, TYPE *a, int *lda,    \
    TYPE *d1, TYPE *d2, TYPE *b, int *ldb, TYPE *wr, TYPE *wi, TYPE *x,        \
    int *ldx, TYPE *scale, TYPE *xnorm, int *info) {                           \
  int info_ = -1;                                                              \
  if (!ltrans || !na || !nw || !smin || !ca || !lda || !d1 || !d2 || !ldb ||  \
      !wr || !wi || !ldx || !scale || !xnorm || !info) {                       \
    return;                                                                    \
  }                                                                            \
  if (!g_cblas_fn[OP_ID]) {                                                    \
    *info = -1;                                                                \
    return;                                                                    \
  }                                                                            \
  if ((*na > 0 && (!a || !b || !x))) {                                         \
    *info = -1;                                                                \
    return;                                                                    \
  }                                                                            \
  info_ = ((int (*)(int, int, int, TYPE, TYPE, TYPE *, int, TYPE, TYPE,        \
                    TYPE *, int, TYPE, TYPE, TYPE *, int, TYPE *, TYPE *))     \
               g_cblas_fn[OP_ID])(*ltrans, *na, *nw, *smin, *ca, a, *lda,      \
                                  *d1, *d2, b, *ldb, *wr, *wi, x, *ldx,        \
                                  scale, xnorm);                               \
  *info = info_;                                                               \
}
