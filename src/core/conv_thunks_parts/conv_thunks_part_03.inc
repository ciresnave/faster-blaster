
#define DEFINE_LAPACK_PT_SVX_COMPLEX_THUNKS(OP, MAT_TYPE, REAL_TYPE, OP_ID)    \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const char fact, const int n, const int nrhs,    \
    REAL_TYPE *d, MAT_TYPE *e, REAL_TYPE *df, MAT_TYPE *ef, MAT_TYPE *b,       \
    const int ldb, MAT_TYPE *x, const int ldx, REAL_TYPE *rcond,               \
    REAL_TYPE *ferr, REAL_TYPE *berr) {                                         \
  int n_ = n, nrhs_ = nrhs, ldb_ = (n > 0) ? n : 1, ldx_ = (n > 0) ? n : 1,   \
      info = 0;                                                                 \
  int work_len = (n > 0) ? (2 * n) : 1;                                         \
  int rwork_len = (n > 0) ? n : 1;                                              \
  size_t b_elems = (size_t)ldb_ * (size_t)((nrhs > 0) ? nrhs : 1);             \
  size_t x_elems = (size_t)ldx_ * (size_t)((nrhs > 0) ? nrhs : 1);             \
  MAT_TYPE *b_col = NULL;                                                       \
  MAT_TYPE *x_col = NULL;                                                       \
  MAT_TYPE *b_arg = b;                                                          \
  MAT_TYPE *x_arg = x;                                                          \
  MAT_TYPE *work = NULL;                                                        \
  REAL_TYPE *rwork = NULL;                                                      \
  if (!g_fortran_fn[OP_ID] || !rcond || !ferr || !berr) return -1;             \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if ((n > 0 && (!d || !df)) || (n > 1 && (!e || !ef)) ||                    \
        (n > 0 && nrhs > 0 && (!b || !x)) || ldb < nrhs || ldx < nrhs) {      \
      return -1;                                                                \
    }                                                                           \
    b_col = (MAT_TYPE *)malloc(b_elems * sizeof(MAT_TYPE));                     \
    x_col = (MAT_TYPE *)malloc(x_elems * sizeof(MAT_TYPE));                     \
    if (!b_col || !x_col) {                                                     \
      free(b_col);                                                              \
      free(x_col);                                                              \
      return -101;                                                              \
    }                                                                           \
    if (fb_copy_matrix_to_col_major(b, layout, n, nrhs, ldb,                   \
                                    b_col, ldb_, sizeof(MAT_TYPE)) != 0) {     \
      free(b_col);                                                              \
      free(x_col);                                                              \
      return -1;                                                                \
    }                                                                           \
    b_arg = b_col;                                                              \
    x_arg = x_col;                                                              \
  } else if (fb_lapack_layout_is_col_major(layout)) {                           \
    ldb_ = ldb;                                                                 \
    ldx_ = ldx;                                                                 \
    if ((n > 0 && (!d || !df)) || (n > 1 && (!e || !ef)) ||                    \
        (n > 0 && nrhs > 0 && (!b || !x))) {                                   \
      return -1;                                                                \
    }                                                                           \
  } else {                                                                      \
    return -1;                                                                  \
  }                                                                             \
  work = (MAT_TYPE *)malloc((size_t)work_len * sizeof(MAT_TYPE));              \
  rwork = (REAL_TYPE *)malloc((size_t)rwork_len * sizeof(REAL_TYPE));          \
  if (!work || !rwork) {                                                        \
    free(b_col);                                                                \
    free(x_col);                                                                \
    free(work);                                                                 \
    free(rwork);                                                                \
    return -101;                                                                \
  }                                                                             \
  ((void (*)(char *, int *, int *, REAL_TYPE *, MAT_TYPE *, REAL_TYPE *,       \
             MAT_TYPE *, MAT_TYPE *, int *, MAT_TYPE *, int *, REAL_TYPE *,    \
             REAL_TYPE *, REAL_TYPE *, MAT_TYPE *, REAL_TYPE *, int *))        \
      g_fortran_fn[OP_ID])((char *)&fact, &n_, &nrhs_, d, e, df, ef, b_arg,    \
                           &ldb_, x_arg, &ldx_, rcond, ferr, berr, work, rwork,\
                           &info);                                              \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if (fb_copy_matrix_from_col_major(x_col, n, nrhs, ldx_, layout,            \
                                      x, ldx, sizeof(MAT_TYPE)) != 0) {        \
      free(b_col);                                                              \
      free(x_col);                                                              \
      free(work);                                                               \
      free(rwork);                                                              \
      return -1;                                                                \
    }                                                                           \
    free(b_col);                                                                \
    free(x_col);                                                                \
  }                                                                             \
  free(work);                                                                   \
  free(rwork);                                                                  \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(char *fact, int *n, int *nrhs,       \
                                          REAL_TYPE *d, MAT_TYPE *e,           \
                                          REAL_TYPE *df, MAT_TYPE *ef,         \
                                          MAT_TYPE *b, int *ldb, MAT_TYPE *x,  \
                                          int *ldx, REAL_TYPE *rcond,          \
                                          REAL_TYPE *ferr, REAL_TYPE *berr,    \
                                          MAT_TYPE *work, REAL_TYPE *rwork,    \
                                          int *info) {                          \
  (void)work;                                                                   \
  (void)rwork;                                                                  \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!fact || !n || !nrhs || !ldb || !ldx || !rcond || !ferr || !berr) {     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, char, int, int, REAL_TYPE *, MAT_TYPE *,    \
                     REAL_TYPE *, MAT_TYPE *, MAT_TYPE *, int, MAT_TYPE *,     \
                     int, REAL_TYPE *, REAL_TYPE *, REAL_TYPE *))g_cblas_fn[OP_ID])(\
      FB_LAYOUT_COL_MAJOR, *fact, *n, *nrhs, d, e, df, ef, b, *ldb, x, *ldx,   \
      rcond, ferr, berr);                                                       \
  if (info) *info = rc;                                                         \
}

#define DEFINE_LAPACK_PTRFS_REAL_THUNKS(OP, REAL_TYPE, OP_ID)                  \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const char uplo, const char trans,               \
    const char diag, const int n, const int nrhs, const REAL_TYPE *ap,         \
    const REAL_TYPE *b, const int ldb, const REAL_TYPE *x, const int ldx,      \
    REAL_TYPE *ferr, REAL_TYPE *berr) {                                         \
  int n_ = n, nrhs_ = nrhs, ldb_ = (n > 0) ? n : 1, ldx_ = (n > 0) ? n : 1,   \
      info = 0;                                                                 \
  int work_len = (n > 0) ? n : 1;                                               \
  size_t packed_len = (size_t)((n > 0) ? n : 1) *                               \
                      (size_t)(((n + 1) > 0) ? (n + 1) : 2) / 2U;              \
  size_t b_elems = (size_t)ldb_ * (size_t)((nrhs > 0) ? nrhs : 1);             \
  size_t x_elems = (size_t)ldx_ * (size_t)((nrhs > 0) ? nrhs : 1);             \
  REAL_TYPE *ap_col = NULL;                                                     \
  REAL_TYPE *b_col = NULL;                                                      \
  REAL_TYPE *x_col = NULL;                                                      \
  const REAL_TYPE *ap_arg = ap;                                                 \
  const REAL_TYPE *b_arg = b;                                                   \
  const REAL_TYPE *x_arg = x;                                                   \
  REAL_TYPE *work = NULL;                                                       \
  if (!g_fortran_fn[OP_ID] || !ferr || !berr) return -1;                       \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if ((n > 0 && !ap) || (n > 0 && nrhs > 0 && (!b || !x)) ||                 \
        ldb < nrhs || ldx < nrhs) {                                             \
      return -1;                                                                \
    }                                                                           \
    ap_col = (REAL_TYPE *)malloc(packed_len * sizeof(REAL_TYPE));               \
    b_col = (REAL_TYPE *)malloc(b_elems * sizeof(REAL_TYPE));                   \
    x_col = (REAL_TYPE *)malloc(x_elems * sizeof(REAL_TYPE));                   \
    if (!ap_col || !b_col || !x_col) {                                          \
      free(ap_col);                                                             \
      free(b_col);                                                              \
      free(x_col);                                                              \
      return -101;                                                              \
    }                                                                           \
    if (fb_copy_triangular_packed_to_col_major(ap, layout, uplo, diag, n,      \
                                               ap_col, sizeof(REAL_TYPE)) != 0 ||\
        fb_copy_matrix_to_col_major(b, layout, n, nrhs, ldb,                   \
                                    b_col, ldb_, sizeof(REAL_TYPE)) != 0 ||    \
        fb_copy_matrix_to_col_major(x, layout, n, nrhs, ldx,                   \
                                    x_col, ldx_, sizeof(REAL_TYPE)) != 0) {    \
      free(ap_col);                                                             \
      free(b_col);                                                              \
      free(x_col);                                                              \
      return -1;                                                                \
    }                                                                           \
    ap_arg = ap_col;                                                            \
    b_arg = b_col;                                                              \
    x_arg = x_col;                                                              \
  } else if (fb_lapack_layout_is_col_major(layout)) {                           \
    ldb_ = ldb;                                                                 \
    ldx_ = ldx;                                                                 \
    if ((n > 0 && !ap) || (n > 0 && nrhs > 0 && (!b || !x))) {                 \
      return -1;                                                                \
    }                                                                           \
  } else {                                                                      \
    return -1;                                                                  \
  }                                                                             \
  work = (REAL_TYPE *)malloc((size_t)work_len * sizeof(REAL_TYPE));            \
  if (!work) {                                                                  \
    free(ap_col);                                                               \
    free(b_col);                                                                \
    free(x_col);                                                                \
    return -101;                                                                \
  }                                                                             \
  ((void (*)(char *, char *, char *, int *, int *, const REAL_TYPE *,          \
             const REAL_TYPE *, int *, REAL_TYPE *, int *, REAL_TYPE *,        \
             REAL_TYPE *, REAL_TYPE *, int *))g_fortran_fn[OP_ID])(            \
      (char *)&uplo, (char *)&trans, (char *)&diag, &n_, &nrhs_, ap_arg,       \
      b_arg, &ldb_, (REAL_TYPE *)x_arg, &ldx_, ferr, berr, work, &info);       \
  free(ap_col);                                                                 \
  free(b_col);                                                                  \
  free(x_col);                                                                  \
  free(work);                                                                   \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    char *uplo, char *trans, char *diag, int *n, int *nrhs, REAL_TYPE *ap,     \
    REAL_TYPE *b, int *ldb, REAL_TYPE *x, int *ldx, REAL_TYPE *ferr,           \
    REAL_TYPE *berr, REAL_TYPE *work, int *info) {                              \
  (void)work;                                                                   \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!uplo || !trans || !diag || !n || !nrhs || !ldb || !ldx || !ferr || !berr) {\
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, char, char, char, int, int, const REAL_TYPE *,\
                     const REAL_TYPE *, int, const REAL_TYPE *, int,           \
                     REAL_TYPE *, REAL_TYPE *))g_cblas_fn[OP_ID])(             \
      FB_LAYOUT_COL_MAJOR, *uplo, *trans, *diag, *n, *nrhs, ap, b, *ldb, x,   \
      *ldx, ferr, berr);                                                        \
  if (info) *info = rc;                                                         \
}

#define DEFINE_LAPACK_PTRFS_COMPLEX_THUNKS(OP, MAT_TYPE, REAL_TYPE, OP_ID)     \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const fb_uplo_t uplo, const int n,               \
    const int nrhs, const REAL_TYPE *d, const MAT_TYPE *e, const REAL_TYPE *df,\
    const MAT_TYPE *ef, const MAT_TYPE *b, const int ldb, const MAT_TYPE *x,   \
    const int ldx, REAL_TYPE *ferr, REAL_TYPE *berr) {                          \
  char uplo_char = 'U';                                                         \
  int n_ = n, nrhs_ = nrhs, ldb_ = (n > 0) ? n : 1, ldx_ = (n > 0) ? n : 1,   \
      info = 0;                                                                 \
  int work_len = (n > 0) ? n : 1;                                               \
  int rwork_len = (n > 0) ? n : 1;                                              \
  size_t b_elems = (size_t)ldb_ * (size_t)((nrhs > 0) ? nrhs : 1);             \
  size_t x_elems = (size_t)ldx_ * (size_t)((nrhs > 0) ? nrhs : 1);             \
  MAT_TYPE *b_col = NULL;                                                       \
  MAT_TYPE *x_col = NULL;                                                       \
  const MAT_TYPE *b_arg = b;                                                    \
  const MAT_TYPE *x_arg = x;                                                    \
  MAT_TYPE *work = NULL;                                                        \
  REAL_TYPE *rwork = NULL;                                                      \
  if (!g_fortran_fn[OP_ID] || !ferr || !berr) return -1;                       \
  if (fb_lapack_uplo_to_char(uplo, &uplo_char) != 0) return -1;                \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if ((n > 0 && (!d || !df)) || (n > 1 && (!e || !ef)) ||                    \
        (n > 0 && nrhs > 0 && (!b || !x)) || ldb < nrhs || ldx < nrhs) {      \
      return -1;                                                                \
    }                                                                           \
    b_col = (MAT_TYPE *)malloc(b_elems * sizeof(MAT_TYPE));                     \
    x_col = (MAT_TYPE *)malloc(x_elems * sizeof(MAT_TYPE));                     \
    if (!b_col || !x_col) {                                                     \
      free(b_col);                                                              \
      free(x_col);                                                              \
      return -101;                                                              \
    }                                                                           \
    if (fb_copy_matrix_to_col_major(b, layout, n, nrhs, ldb,                   \
                                    b_col, ldb_, sizeof(MAT_TYPE)) != 0 ||     \
        fb_copy_matrix_to_col_major(x, layout, n, nrhs, ldx,                   \
                                    x_col, ldx_, sizeof(MAT_TYPE)) != 0) {     \
      free(b_col);                                                              \
      free(x_col);                                                              \
      return -1;                                                                \
    }                                                                           \
    b_arg = b_col;                                                              \
    x_arg = x_col;                                                              \
  } else if (fb_lapack_layout_is_col_major(layout)) {                           \
    ldb_ = ldb;                                                                 \
    ldx_ = ldx;                                                                 \
    if ((n > 0 && (!d || !df)) || (n > 1 && (!e || !ef)) ||                    \
        (n > 0 && nrhs > 0 && (!b || !x))) {                                   \
      return -1;                                                                \
    }                                                                           \
  } else {                                                                      \
    return -1;                                                                  \
  }                                                                             \
  work = (MAT_TYPE *)malloc((size_t)work_len * sizeof(MAT_TYPE));              \
  rwork = (REAL_TYPE *)malloc((size_t)rwork_len * sizeof(REAL_TYPE));          \
  if (!work || !rwork) {                                                        \
    free(b_col);                                                                \
    free(x_col);                                                                \
    free(work);                                                                 \
    free(rwork);                                                                \
    return -101;                                                                \
  }                                                                             \
  ((void (*)(char *, int *, int *, REAL_TYPE *, MAT_TYPE *, REAL_TYPE *,       \
             MAT_TYPE *, MAT_TYPE *, int *, MAT_TYPE *, int *, REAL_TYPE *,    \
             REAL_TYPE *, MAT_TYPE *, REAL_TYPE *, int *))g_fortran_fn[OP_ID])(\
      &uplo_char, &n_, &nrhs_, (REAL_TYPE *)d, (MAT_TYPE *)e, (REAL_TYPE *)df, \
      (MAT_TYPE *)ef, (MAT_TYPE *)b_arg, &ldb_, (MAT_TYPE *)x_arg, &ldx_,      \
      ferr, berr, work, rwork, &info);                                          \
  free(b_col);                                                                  \
  free(x_col);                                                                  \
  free(work);                                                                   \
  free(rwork);                                                                  \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    char *uplo, int *n, int *nrhs, REAL_TYPE *d, MAT_TYPE *e, REAL_TYPE *df,   \
    MAT_TYPE *ef, MAT_TYPE *b, int *ldb, MAT_TYPE *x, int *ldx,                \
    REAL_TYPE *ferr, REAL_TYPE *berr, MAT_TYPE *work, REAL_TYPE *rwork,        \
    int *info) {                                                                 \
  fb_uplo_t uplo_ = FB_UPPER;                                                   \
  (void)work;                                                                   \
  (void)rwork;                                                                  \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!uplo || !n || !nrhs || !ldb || !ldx || !ferr || !berr ||                \
      fb_lapack_char_to_uplo(*uplo, &uplo_) != 0) {                            \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, fb_uplo_t, int, int, const REAL_TYPE *,      \
                     const MAT_TYPE *, const REAL_TYPE *, const MAT_TYPE *,    \
                     const MAT_TYPE *, int, const MAT_TYPE *, int, REAL_TYPE *,\
                     REAL_TYPE *))g_cblas_fn[OP_ID])(                          \
      FB_LAYOUT_COL_MAJOR, uplo_, *n, *nrhs, d, e, df, ef, b, *ldb, x, *ldx,   \
      ferr, berr);                                                              \
  if (info) *info = rc;                                                         \
}

#define DEFINE_COMPLEX_ROTG_THUNKS(OP, MAT_TYPE, REAL_TYPE, OP_ID)             \
static void thunk_##OP##_fortran_to_cblas(MAT_TYPE *a, MAT_TYPE *b,            \
                                          REAL_TYPE *c, MAT_TYPE *s) {          \
  if (!g_fortran_fn[OP_ID]) return;                                             \
  ((void (*)(MAT_TYPE *, MAT_TYPE *, REAL_TYPE *, MAT_TYPE *))                 \
    g_fortran_fn[OP_ID])(a, b, c, s);                                         \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(MAT_TYPE *a, MAT_TYPE *b,            \
                                          REAL_TYPE *c, MAT_TYPE *s) {          \
  if (!g_cblas_fn[OP_ID]) return;                                               \
  ((void (*)(MAT_TYPE *, MAT_TYPE *, REAL_TYPE *, MAT_TYPE *))                 \
    g_cblas_fn[OP_ID])(a, b, c, s);                                           \
}

#define DEFINE_RSCL_REAL_THUNKS(OP, REAL_TYPE, OP_ID)                          \
static void thunk_##OP##_fortran_to_cblas(int n, REAL_TYPE sa,                \
                      REAL_TYPE *sx, int incx) {           \
  int n_ = n;                                                                   \
  int incx_ = incx;                                                             \
  REAL_TYPE sa_ = sa;                                                           \
  if (!g_fortran_fn[OP_ID]) return;                                             \
  ((void (*)(int *, REAL_TYPE *, REAL_TYPE *, int *))g_fortran_fn[OP_ID])(     \
    &n_, &sa_, sx, &incx_);                                                   \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(int *n, REAL_TYPE *sa,              \
                      REAL_TYPE *sx, int *incx) {          \
  if (!g_cblas_fn[OP_ID]) return;                                               \
  ((void (*)(int, REAL_TYPE, REAL_TYPE *, int))g_cblas_fn[OP_ID])(             \
    *n, *sa, sx, *incx);                                                      \
}

#define DEFINE_RSCL_COMPLEX_THUNKS(OP, MAT_TYPE, REAL_TYPE, OP_ID)             \
static void thunk_##OP##_fortran_to_cblas(int n, REAL_TYPE sa,                \
                      MAT_TYPE *sx, int incx) {            \
  int n_ = n;                                                                   \
  int incx_ = incx;                                                             \
  REAL_TYPE sa_ = sa;                                                           \
  if (!g_fortran_fn[OP_ID]) return;                                             \
  ((void (*)(int *, REAL_TYPE *, MAT_TYPE *, int *))g_fortran_fn[OP_ID])(      \
    &n_, &sa_, sx, &incx_);                                                   \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(int *n, REAL_TYPE *sa,              \
                      MAT_TYPE *sx, int *incx) {           \
  if (!g_cblas_fn[OP_ID]) return;                                               \
  ((void (*)(int, REAL_TYPE, MAT_TYPE *, int))g_cblas_fn[OP_ID])(              \
    *n, *sa, sx, *incx);                                                      \
}

#define DEFINE_SBEV_REAL_THUNKS(OP, TYPE, OP_ID)                                \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const char jobz, const fb_uplo_t uplo,           \
    const int n, const int kd, TYPE *ab, const int ldab, TYPE *w, TYPE *z,     \
    const int ldz) {                                                            \
  char jobz_ = jobz;                                                            \
  char uplo_char = 'U';                                                         \
  int n_ = n;                                                                   \
  int kd_ = kd;                                                                 \
  int ldab_ = (kd >= 0) ? (kd + 1) : 1;                                         \
  int ldz_ = (n > 0) ? n : 1;                                                   \
  int vectors_requested = fb_lapack_job_requests_vectors(jobz_);                \
  int work_size = (n > 1) ? (3 * n - 2) : 1;                                   \
  size_t ab_elems = (size_t)ldab_ * (size_t)((n > 0) ? n : 1);                 \
  size_t z_elems = (size_t)ldz_ * (size_t)((n > 0) ? n : 1);                   \
  TYPE *ab_col = NULL;                                                          \
  TYPE *z_col = NULL;                                                           \
  TYPE *work = NULL;                                                            \
  TYPE dummy_z = (TYPE)0;                                                       \
  TYPE *ab_arg = ab;                                                            \
  TYPE *z_arg = vectors_requested ? z : &dummy_z;                               \
  int info = 0;                                                                 \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (fb_lapack_uplo_to_char(uplo, &uplo_char) != 0 || !w) return -1;          \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if ((n > 0 && !ab) || ldab < n) return -1;                                 \
    if (vectors_requested && (!z || ldz < n)) return -1;                       \
    ab_col = (TYPE *)malloc(ab_elems * sizeof(TYPE));                           \
    if (!ab_col) return -101;                                                   \
    if (fb_copy_sym_band_to_col_major(ab, layout, uplo, n, kd, ldab,           \
                                      ab_col, ldab_, sizeof(TYPE)) != 0) {      \
      free(ab_col);                                                             \
      return -1;                                                                \
    }                                                                           \
    ab_arg = ab_col;                                                            \
    if (vectors_requested) {                                                    \
      z_col = (TYPE *)malloc(z_elems * sizeof(TYPE));                           \
      if (!z_col) {                                                             \
        free(ab_col);                                                           \
        return -101;                                                            \
      }                                                                         \
      z_arg = z_col;                                                            \
    } else {                                                                    \
      z_arg = &dummy_z;                                                         \
      ldz_ = 1;                                                                 \
    }                                                                           \
  } else if (fb_lapack_layout_is_col_major(layout)) {                           \
    ldab_ = ldab;                                                               \
    if (n > 0 && !ab) return -1;                                                \
    if (vectors_requested) {                                                    \
      if (!z || ldz < n) return -1;                                             \
      z_arg = z;                                                                \
      ldz_ = ldz;                                                               \
    } else {                                                                    \
      z_arg = &dummy_z;                                                         \
      ldz_ = 1;                                                                 \
    }                                                                           \
  } else {                                                                      \
    return -1;                                                                  \
  }                                                                             \
  work = (TYPE *)malloc((size_t)work_size * sizeof(TYPE));                     \
  if (!work) {                                                                  \
    free(ab_col);                                                               \
    free(z_col);                                                                \
    return -101;                                                                \
  }                                                                             \
  ((void (*)(char *, char *, int *, int *, TYPE *, int *, TYPE *, TYPE *,      \
             int *, TYPE *, int *))g_fortran_fn[OP_ID])(                       \
      &jobz_, &uplo_char, &n_, &kd_, ab_arg, &ldab_, w, z_arg, &ldz_, work,    \
      &info);                                                                   \
  if (layout == FB_LAYOUT_ROW_MAJOR && vectors_requested && info == 0) {       \
    if (fb_copy_matrix_from_col_major(z_col, n, n, ldz_, layout,               \
                                      z, ldz, sizeof(TYPE)) != 0) {             \
      free(ab_col);                                                             \
      free(z_col);                                                              \
      free(work);                                                               \
      return -1;                                                                \
    }                                                                           \
  }                                                                             \
  free(ab_col);                                                                 \
  free(z_col);                                                                  \
  free(work);                                                                   \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    char *jobz, char *uplo, int *n, int *kd, TYPE *ab, int *ldab, TYPE *w,     \
    TYPE *z, int *ldz, TYPE *work, int *info) {                                 \
  fb_uplo_t uplo_enum = FB_UPPER;                                               \
  TYPE dummy_z = (TYPE)0;                                                       \
  TYPE *z_ptr = z ? z : &dummy_z;                                               \
  int ldz_value = (z && ldz) ? *ldz : 1;                                        \
  (void)work;                                                                   \
  if (!jobz || !uplo || !n || !kd || !ldab || !w ||                            \
      fb_lapack_char_to_uplo(*uplo, &uplo_enum) != 0) {                        \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (fb_lapack_job_requests_vectors(*jobz) && (!z || !ldz)) {                 \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, char, fb_uplo_t, int, int, TYPE *, int,      \
                     TYPE *, TYPE *, int))g_cblas_fn[OP_ID])(                   \
      FB_LAYOUT_COL_MAJOR, *jobz, uplo_enum, *n, *kd, ab, *ldab, w, z_ptr,     \
      ldz_value);                                                               \
  if (info) *info = rc;                                                         \
}

#define DEFINE_HBEV_COMPLEX_THUNKS(OP, CTYPE, RTYPE, OP_ID)                    \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const char jobz, const fb_uplo_t uplo,           \
    const int n, const int kd, CTYPE *ab, const int ldab, RTYPE *w, CTYPE *z,  \
    const int ldz) {                                                            \
  char jobz_ = jobz;                                                            \
  char uplo_char = 'U';                                                         \
  int n_ = n;                                                                   \
  int kd_ = kd;                                                                 \
  int ldab_ = (kd >= 0) ? (kd + 1) : 1;                                         \
  int ldz_ = (n > 0) ? n : 1;                                                   \
  int vectors_requested = fb_lapack_job_requests_vectors(jobz_);                \
  int work_size = (n > 0) ? n : 1;                                              \
  int rwork_size = (n > 1) ? (3 * n - 2) : 1;                                  \
  size_t ab_elems = (size_t)ldab_ * (size_t)((n > 0) ? n : 1);                 \
  size_t z_elems = (size_t)ldz_ * (size_t)((n > 0) ? n : 1);                   \
  CTYPE *ab_col = NULL;                                                         \
  CTYPE *z_col = NULL;                                                          \
  CTYPE *work = NULL;                                                           \
  RTYPE *rwork = NULL;                                                          \
  CTYPE dummy_z = (CTYPE)0;                                                     \
  CTYPE *ab_arg = ab;                                                           \
  CTYPE *z_arg = vectors_requested ? z : &dummy_z;                              \
  int info = 0;                                                                 \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (fb_lapack_uplo_to_char(uplo, &uplo_char) != 0 || !w) return -1;          \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if ((n > 0 && !ab) || ldab < n) return -1;                                 \
    if (vectors_requested && (!z || ldz < n)) return -1;                       \
    ab_col = (CTYPE *)malloc(ab_elems * sizeof(CTYPE));                         \
    if (!ab_col) return -101;                                                   \
    if (fb_copy_sym_band_to_col_major(ab, layout, uplo, n, kd, ldab,           \
                                      ab_col, ldab_, sizeof(CTYPE)) != 0) {    \
      free(ab_col);                                                             \
      return -1;                                                                \
    }                                                                           \
    ab_arg = ab_col;                                                            \
    if (vectors_requested) {                                                    \
      z_col = (CTYPE *)malloc(z_elems * sizeof(CTYPE));                         \
      if (!z_col) {                                                             \
        free(ab_col);                                                           \
        return -101;                                                            \
      }                                                                         \
      z_arg = z_col;                                                            \
    } else {                                                                    \
      z_arg = &dummy_z;                                                         \
      ldz_ = 1;                                                                 \
    }                                                                           \
  } else if (fb_lapack_layout_is_col_major(layout)) {                           \
    ldab_ = ldab;                                                               \
    if (n > 0 && !ab) return -1;                                                \
    if (vectors_requested) {                                                    \
      if (!z || ldz < n) return -1;                                             \
      z_arg = z;                                                                \
      ldz_ = ldz;                                                               \
    } else {                                                                    \
      z_arg = &dummy_z;                                                         \
      ldz_ = 1;                                                                 \
    }                                                                           \
  } else {                                                                      \
    return -1;                                                                  \
  }                                                                             \
  work = (CTYPE *)malloc((size_t)work_size * sizeof(CTYPE));                    \
  rwork = (RTYPE *)malloc((size_t)rwork_size * sizeof(RTYPE));                  \
  if (!work || !rwork) {                                                        \
    free(ab_col);                                                               \
    free(z_col);                                                                \
    free(work);                                                                 \
    free(rwork);                                                                \
    return -101;                                                                \
  }                                                                             \
  ((void (*)(char *, char *, int *, int *, CTYPE *, int *, RTYPE *, CTYPE *,   \
             int *, CTYPE *, RTYPE *, int *))g_fortran_fn[OP_ID])(             \
      &jobz_, &uplo_char, &n_, &kd_, ab_arg, &ldab_, w, z_arg, &ldz_, work,    \
      rwork, &info);                                                            \
  if (layout == FB_LAYOUT_ROW_MAJOR && vectors_requested && info == 0) {       \
    if (fb_copy_matrix_from_col_major(z_col, n, n, ldz_, layout,               \
                                      z, ldz, sizeof(CTYPE)) != 0) {           \
      free(ab_col);                                                             \
      free(z_col);                                                              \
      free(work);                                                               \
      free(rwork);                                                              \
      return -1;                                                                \
    }                                                                           \
  }                                                                             \
  free(ab_col);                                                                 \
  free(z_col);                                                                  \
  free(work);                                                                   \
  free(rwork);                                                                  \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    char *jobz, char *uplo, int *n, int *kd, CTYPE *ab, int *ldab, RTYPE *w,   \
    CTYPE *z, int *ldz, CTYPE *work, RTYPE *rwork, int *info) {                \
  fb_uplo_t uplo_enum = FB_UPPER;                                               \
  CTYPE dummy_z = (CTYPE)0;                                                     \
  CTYPE *z_ptr = z ? z : &dummy_z;                                              \
  int ldz_value = (z && ldz) ? *ldz : 1;                                        \
  (void)work;                                                                   \
  (void)rwork;                                                                  \
  if (!jobz || !uplo || !n || !kd || !ldab || !w ||                            \
      fb_lapack_char_to_uplo(*uplo, &uplo_enum) != 0) {                        \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (fb_lapack_job_requests_vectors(*jobz) && (!z || !ldz)) {                 \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, char, fb_uplo_t, int, int, CTYPE *, int,     \
                     RTYPE *, CTYPE *, int))g_cblas_fn[OP_ID])(                \
      FB_LAYOUT_COL_MAJOR, *jobz, uplo_enum, *n, *kd, ab, *ldab, w, z_ptr,     \
      ldz_value);                                                               \
  if (info) *info = rc;                                                         \
}

#define DEFINE_HBEVD_COMPLEX_THUNKS(OP, CTYPE, RTYPE, OP_ID, QUERY_FN)         \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const char jobz, const fb_uplo_t uplo,           \
    const int n, const int kd, CTYPE *ab, const int ldab, RTYPE *w, CTYPE *z,  \
    const int ldz) {                                                            \
  char jobz_ = jobz;                                                            \
  char uplo_char = 'U';                                                         \
  int n_ = n;                                                                   \
  int kd_ = kd;                                                                 \
  int ldab_ = (kd >= 0) ? (kd + 1) : 1;                                         \
  int ldz_ = (n > 0) ? n : 1;                                                   \
  int vectors_requested = fb_lapack_job_requests_vectors(jobz_);                \
  int lwork_ = -1, lrwork_ = -1, liwork_ = -1, info = 0;                       \
  CTYPE work_query = (CTYPE)0;                                                  \
  RTYPE rwork_query = (RTYPE)0;                                                 \
  int iwork_query = 0;                                                          \
  size_t ab_elems = (size_t)ldab_ * (size_t)((n > 0) ? n : 1);                 \
  size_t z_elems = (size_t)ldz_ * (size_t)((n > 0) ? n : 1);                   \
  CTYPE *ab_col = NULL;                                                         \
  CTYPE *z_col = NULL;                                                          \
  CTYPE *work = NULL;                                                           \
  RTYPE *rwork = NULL;                                                          \
  int *iwork = NULL;                                                            \
  CTYPE dummy_z = (CTYPE)0;                                                     \
  CTYPE *ab_arg = ab;                                                           \
  CTYPE *z_arg = vectors_requested ? z : &dummy_z;                              \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (fb_lapack_uplo_to_char(uplo, &uplo_char) != 0 || !w) return -1;          \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if ((n > 0 && !ab) || ldab < n) return -1;                                 \
    if (vectors_requested && (!z || ldz < n)) return -1;                       \
    ab_col = (CTYPE *)malloc(ab_elems * sizeof(CTYPE));                         \
    if (!ab_col) return -101;                                                   \
    if (fb_copy_sym_band_to_col_major(ab, layout, uplo, n, kd, ldab,           \
                                      ab_col, ldab_, sizeof(CTYPE)) != 0) {    \
      free(ab_col);                                                             \
      return -1;                                                                \
    }                                                                           \
    ab_arg = ab_col;                                                            \
    if (vectors_requested) {                                                    \
      z_col = (CTYPE *)malloc(z_elems * sizeof(CTYPE));                         \
      if (!z_col) {                                                             \
        free(ab_col);                                                           \
        return -101;                                                            \
      }                                                                         \
      z_arg = z_col;                                                            \
    } else {                                                                    \
      z_arg = &dummy_z;                                                         \
      ldz_ = 1;                                                                 \
    }                                                                           \
  } else if (fb_lapack_layout_is_col_major(layout)) {                           \
    ldab_ = ldab;                                                               \
    if (n > 0 && !ab) return -1;                                                \
    if (vectors_requested) {                                                    \
      if (!z || ldz < n) return -1;                                             \
      z_arg = z;                                                                \
      ldz_ = ldz;                                                               \
    } else {                                                                    \
      z_arg = &dummy_z;                                                         \
      ldz_ = 1;                                                                 \
    }                                                                           \
  } else {                                                                      \
    return -1;                                                                  \
  }                                                                             \
  ((void (*)(char *, char *, int *, int *, CTYPE *, int *, RTYPE *, CTYPE *,   \
             int *, CTYPE *, int *, RTYPE *, int *, int *, int *, int *))      \
      g_fortran_fn[OP_ID])(                                                     \
      &jobz_, &uplo_char, &n_, &kd_, ab_arg, &ldab_, w, z_arg, &ldz_,          \
      &work_query, &lwork_, &rwork_query, &lrwork_, &iwork_query, &liwork_,    \
      &info);                                                                   \
  if (info != 0) {                                                              \
    free(ab_col);                                                               \
    free(z_col);                                                                \
    return info;                                                                \
  }                                                                             \
  lwork_ = QUERY_FN(work_query);                                                \
  lrwork_ = (rwork_query > (RTYPE)1) ? (int)rwork_query : 1;                   \
  liwork_ = fb_query_size_from_int(iwork_query);                                \
  work = (CTYPE *)malloc((size_t)lwork_ * sizeof(CTYPE));                       \
  rwork = (RTYPE *)malloc((size_t)lrwork_ * sizeof(RTYPE));                     \
  iwork = (int *)malloc((size_t)liwork_ * sizeof(int));                         \
  if (!work || !rwork || !iwork) {                                              \
    free(ab_col);                                                               \
    free(z_col);                                                                \
    free(work);                                                                 \
    free(rwork);                                                                \
    free(iwork);                                                                \
    return -101;                                                                \
  }                                                                             \
  ((void (*)(char *, char *, int *, int *, CTYPE *, int *, RTYPE *, CTYPE *,   \
             int *, CTYPE *, int *, RTYPE *, int *, int *, int *, int *))      \
      g_fortran_fn[OP_ID])(                                                     \
      &jobz_, &uplo_char, &n_, &kd_, ab_arg, &ldab_, w, z_arg, &ldz_,          \
      work, &lwork_, rwork, &lrwork_, iwork, &liwork_, &info);                 \
  if (layout == FB_LAYOUT_ROW_MAJOR && vectors_requested && info == 0) {       \
    if (fb_copy_matrix_from_col_major(z_col, n, n, ldz_, layout,               \
                                      z, ldz, sizeof(CTYPE)) != 0) {            \
      free(ab_col);                                                             \
      free(z_col);                                                              \
      free(work);                                                               \
      free(rwork);                                                              \
      free(iwork);                                                              \
      return -1;                                                                \
    }                                                                           \
  }                                                                             \
  free(ab_col);                                                                 \
  free(z_col);                                                                  \
  free(work);                                                                   \
  free(rwork);                                                                  \
  free(iwork);                                                                  \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    char *jobz, char *uplo, int *n, int *kd, CTYPE *ab, int *ldab, RTYPE *w,   \
    CTYPE *z, int *ldz, CTYPE *work, int *lwork, RTYPE *rwork, int *lrwork,    \
    int *iwork, int *liwork, int *info) {                                       \
  fb_uplo_t uplo_enum = FB_UPPER;                                               \
  CTYPE dummy_z = (CTYPE)0;                                                     \
  CTYPE *z_ptr = z ? z : &dummy_z;                                              \
  int ldz_value = (z && ldz) ? *ldz : 1;                                        \
  (void)work;                                                                   \
  (void)lwork;                                                                  \
  (void)rwork;                                                                  \
  (void)lrwork;                                                                 \
  (void)iwork;                                                                  \
  (void)liwork;                                                                 \
  if (!jobz || !uplo || !n || !kd || !ldab || !w ||                            \
      fb_lapack_char_to_uplo(*uplo, &uplo_enum) != 0) {                        \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (fb_lapack_job_requests_vectors(*jobz) && (!z || !ldz)) {                 \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, char, fb_uplo_t, int, int, CTYPE *, int,     \
                     RTYPE *, CTYPE *, int))g_cblas_fn[OP_ID])(                 \
      FB_LAYOUT_COL_MAJOR, *jobz, uplo_enum, *n, *kd, ab, *ldab, w, z_ptr,     \
      ldz_value);                                                               \
  if (info) *info = rc;                                                         \
}

#define DEFINE_SPEV_REAL_THUNKS(OP, TYPE, OP_ID)                                \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const char jobz, const fb_uplo_t uplo,           \
    const int n, TYPE *ap, TYPE *w, TYPE *z, const int ldz) {                  \
  char jobz_ = jobz;                                                            \
  char uplo_char = 'U';                                                         \
  int n_ = n;                                                                   \
  int ldz_ = (n > 0) ? n : 1;                                                   \
  int vectors_requested = fb_lapack_job_requests_vectors(jobz_);                \
  int work_size = (n > 0) ? (3 * n) : 1;                                        \
  size_t packed_len = (size_t)((n > 0) ? n : 1) *                               \
                      (size_t)(((n + 1) > 0) ? (n + 1) : 2) / 2U;              \
  size_t z_elems = (size_t)ldz_ * (size_t)((n > 0) ? n : 1);                   \
  TYPE *ap_col = NULL;                                                          \
  TYPE *z_col = NULL;                                                           \
  TYPE *work = NULL;                                                            \
  TYPE dummy_z = (TYPE)0;                                                       \
  TYPE *ap_arg = ap;                                                            \
  TYPE *z_arg = vectors_requested ? z : &dummy_z;                               \
  int info = 0;                                                                 \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (fb_lapack_uplo_to_char(uplo, &uplo_char) != 0 || !w) return -1;          \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if (n > 0 && !ap) return -1;                                                \
    if (vectors_requested && (!z || ldz < n)) return -1;                       \
    ap_col = (TYPE *)malloc(packed_len * sizeof(TYPE));                         \
    if (!ap_col) return -101;                                                   \
    if (fb_copy_triangular_packed_to_col_major(ap, layout, uplo_char, 'N', n,  \
                                               ap_col, sizeof(TYPE)) != 0) {    \
      free(ap_col);                                                             \
      return -1;                                                                \
    }                                                                           \
    ap_arg = ap_col;                                                            \
    if (vectors_requested) {                                                    \
      z_col = (TYPE *)malloc(z_elems * sizeof(TYPE));                           \
      if (!z_col) {                                                             \
        free(ap_col);                                                           \
        return -101;                                                            \
      }                                                                         \
      z_arg = z_col;                                                            \
    } else {                                                                    \
      z_arg = &dummy_z;                                                         \
      ldz_ = 1;                                                                 \
    }                                                                           \
  } else if (fb_lapack_layout_is_col_major(layout)) {                           \
    if (n > 0 && !ap) return -1;                                                \
    if (vectors_requested) {                                                    \
      if (!z || ldz < n) return -1;                                             \
      z_arg = z;                                                                \
      ldz_ = ldz;                                                               \
    } else {                                                                    \
      z_arg = &dummy_z;                                                         \
      ldz_ = 1;                                                                 \
    }                                                                           \
  } else {                                                                      \
    return -1;                                                                  \
  }                                                                             \
  work = (TYPE *)malloc((size_t)work_size * sizeof(TYPE));                     \
  if (!work) {                                                                  \
    free(ap_col);                                                               \
    free(z_col);                                                                \
    return -101;                                                                \
  }                                                                             \
  ((void (*)(char *, char *, int *, TYPE *, TYPE *, TYPE *, int *, TYPE *,     \
             int *))g_fortran_fn[OP_ID])(                                      \
      &jobz_, &uplo_char, &n_, ap_arg, w, z_arg, &ldz_, work, &info);          \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if (fb_copy_triangular_packed_from_col_major(ap_col, layout, uplo_char,    \
                           'N', n, ap, sizeof(TYPE)) != 0 || \
        (vectors_requested &&                                                   \
         fb_copy_matrix_from_col_major(z_col, n, n, ldz_, layout, z, ldz,      \
                                       sizeof(TYPE)) != 0)) {                   \
      free(ap_col);                                                             \
      free(z_col);                                                              \
      free(work);                                                               \
      return -1;                                                                \
    }                                                                           \
  }                                                                             \
  free(ap_col);                                                                 \
  free(z_col);                                                                  \
  free(work);                                                                   \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    char *jobz, char *uplo, int *n, TYPE *ap, TYPE *w, TYPE *z, int *ldz,      \
    TYPE *work, int *info) {                                                    \
  fb_uplo_t uplo_enum = FB_UPPER;                                               \
  TYPE dummy_z = (TYPE)0;                                                       \
  TYPE *z_ptr = z ? z : &dummy_z;                                               \
  int ldz_value = (z && ldz) ? *ldz : 1;                                        \
  (void)work;                                                                   \
  if (!jobz || !uplo || !n || !w ||                                             \
      fb_lapack_char_to_uplo(*uplo, &uplo_enum) != 0) {                        \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (fb_lapack_job_requests_vectors(*jobz) && (!z || !ldz)) {                 \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, char, fb_uplo_t, int, TYPE *, TYPE *,        \
                     TYPE *, int))g_cblas_fn[OP_ID])(                           \
      FB_LAYOUT_COL_MAJOR, *jobz, uplo_enum, *n, ap, w, z_ptr, ldz_value);     \
  if (info) *info = rc;                                                         \
}

#define DEFINE_SPEVD_REAL_THUNKS(OP, TYPE, OP_ID, QUERY_FN)                     \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const char jobz, const fb_uplo_t uplo,           \
    const int n, TYPE *ap, TYPE *w, TYPE *z, const int ldz) {                  \
  char jobz_ = jobz;                                                            \
  char uplo_char = 'U';                                                         \
  int n_ = n;                                                                   \
  int ldz_ = (n > 0) ? n : 1;                                                   \
  int vectors_requested = fb_lapack_job_requests_vectors(jobz_);                \
  int lwork_ = -1;                                                              \
  int liwork_ = -1;                                                             \
  TYPE work_query = (TYPE)0;                                                    \
  int iwork_query = 0;                                                          \
  size_t packed_len = (size_t)((n > 0) ? n : 1) *                               \
                      (size_t)(((n + 1) > 0) ? (n + 1) : 2) / 2U;              \
  size_t z_elems = (size_t)ldz_ * (size_t)((n > 0) ? n : 1);                   \
  TYPE *ap_col = NULL;                                                          \
  TYPE *z_col = NULL;                                                           \
  TYPE *work = NULL;                                                            \
  int *iwork = NULL;                                                            \
  TYPE dummy_z = (TYPE)0;                                                       \
  TYPE *ap_arg = ap;                                                            \
  TYPE *z_arg = vectors_requested ? z : &dummy_z;                               \
  int info = 0;                                                                 \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (fb_lapack_uplo_to_char(uplo, &uplo_char) != 0 || !w) return -1;          \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if (n > 0 && !ap) return -1;                                                \
    if (vectors_requested && (!z || ldz < n)) return -1;                       \
    ap_col = (TYPE *)malloc(packed_len * sizeof(TYPE));                         \
    if (!ap_col) return -101;                                                   \
    if (fb_copy_triangular_packed_to_col_major(ap, layout, uplo_char, 'N', n,  \
                                               ap_col, sizeof(TYPE)) != 0) {    \
      free(ap_col);                                                             \
      return -1;                                                                \
    }                                                                           \
    ap_arg = ap_col;                                                            \
    if (vectors_requested) {                                                    \
      z_col = (TYPE *)malloc(z_elems * sizeof(TYPE));                           \
      if (!z_col) {                                                             \
        free(ap_col);                                                           \
        return -101;                                                            \
      }                                                                         \
      z_arg = z_col;                                                            \
    } else {                                                                    \
      z_arg = &dummy_z;                                                         \
      ldz_ = 1;                                                                 \
    }                                                                           \
  } else if (fb_lapack_layout_is_col_major(layout)) {                           \
    if (n > 0 && !ap) return -1;                                                \
    if (vectors_requested) {                                                    \
      if (!z || ldz < n) return -1;                                             \
      z_arg = z;                                                                \
      ldz_ = ldz;                                                               \
    } else {                                                                    \
      z_arg = &dummy_z;                                                         \
      ldz_ = 1;                                                                 \
    }                                                                           \
  } else {                                                                      \
    return -1;                                                                  \
  }                                                                             \
  ((void (*)(char *, char *, int *, TYPE *, TYPE *, TYPE *, int *, TYPE *,     \
          int *, int *, int *, int *))g_fortran_fn[OP_ID])(                 \
      &jobz_, &uplo_char, &n_, ap_arg, w, z_arg, &ldz_, &work_query, &lwork_,  \
      &iwork_query, &liwork_, &info);                                           \
  if (info != 0) {                                                              \
    free(ap_col);                                                               \
    free(z_col);                                                                \
    return info;                                                                \
  }                                                                             \
  lwork_ = QUERY_FN(work_query);                                                \
  liwork_ = fb_query_size_from_int(iwork_query);                                \
  work = (TYPE *)malloc((size_t)lwork_ * sizeof(TYPE));                         \
  iwork = (int *)malloc((size_t)liwork_ * sizeof(int));                         \
  if (!work || !iwork) {                                                        \
    free(ap_col);                                                               \
    free(z_col);                                                                \
    free(work);                                                                 \
    free(iwork);                                                                \
    return -101;                                                                \
  }                                                                             \
  ((void (*)(char *, char *, int *, TYPE *, TYPE *, TYPE *, int *, TYPE *,     \
          int *, int *, int *, int *))g_fortran_fn[OP_ID])(                 \
      &jobz_, &uplo_char, &n_, ap_arg, w, z_arg, &ldz_, work, &lwork_, iwork,  \
      &liwork_, &info);                                                         \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if (fb_copy_triangular_packed_from_col_major(ap_col, layout, uplo_char,    \
                                                 'N', n, ap, sizeof(TYPE)) != 0 || \
        (vectors_requested &&                                                   \
         fb_copy_matrix_from_col_major(z_col, n, n, ldz_, layout, z, ldz,      \
                                       sizeof(TYPE)) != 0)) {                   \
      free(ap_col);                                                             \
      free(z_col);                                                              \
      free(work);                                                               \
      free(iwork);                                                              \
      return -1;                                                                \
    }                                                                           \
  }                                                                             \
  free(ap_col);                                                                 \
  free(z_col);                                                                  \
  free(work);                                                                   \
  free(iwork);                                                                  \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    char *jobz, char *uplo, int *n, TYPE *ap, TYPE *w, TYPE *z, int *ldz,      \
    TYPE *work, int *lwork, int *iwork, int *liwork, int *info) {              \
  fb_uplo_t uplo_enum = FB_UPPER;                                               \
  TYPE dummy_z = (TYPE)0;                                                       \
  TYPE *z_ptr = z ? z : &dummy_z;                                               \
  int ldz_value = (z && ldz) ? *ldz : 1;                                        \
  (void)work;                                                                   \
  (void)lwork;                                                                  \
  (void)iwork;                                                                  \
  (void)liwork;                                                                 \
  if (!jobz || !uplo || !n || !w ||                                             \
      fb_lapack_char_to_uplo(*uplo, &uplo_enum) != 0) {                        \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (fb_lapack_job_requests_vectors(*jobz) && (!z || !ldz)) {                 \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, char, fb_uplo_t, int, TYPE *, TYPE *,        \
                     TYPE *, int))g_cblas_fn[OP_ID])(                           \
      FB_LAYOUT_COL_MAJOR, *jobz, uplo_enum, *n, ap, w, z_ptr, ldz_value);     \
  if (info) *info = rc;                                                         \
}

#define DEFINE_SPEVX_REAL_THUNKS(OP, TYPE, OP_ID)                               \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const char jobz, const char range,               \
    const fb_uplo_t uplo, const int n, TYPE *ap, const TYPE vl, const TYPE vu, \
    const int il, const int iu, const TYPE abstol, int *m, TYPE *w, TYPE *z,   \
    const int ldz, int *ifail) {                                                \
  char jobz_ = jobz, range_ = range;                                            \
  char uplo_char = 'U';                                                         \
  int n_ = n, il_ = il, iu_ = iu;                                               \
  int z_cols = ((range == 'A' || range == 'a' || range == 'V' || range == 'v') \
                    ? ((n > 0) ? n : 1)                                         \
                    : ((range == 'I' || range == 'i')                           \
                           ? (((iu >= il) && n > 0) ? (iu - il + 1) : 1)        \
                           : 1));                                               \
  int ldz_ = (n > 0) ? n : 1;                                                   \
  int vectors_requested = fb_lapack_job_requests_vectors(jobz_);                \
  int work_size = (n > 0) ? (8 * n) : 1;                                        \
  int iwork_size = (n > 0) ? (5 * n) : 1;                                       \
  TYPE vl_ = vl, vu_ = vu, abstol_ = abstol;                                    \
  size_t packed_len = (size_t)((n > 0) ? n : 1) *                               \
                      (size_t)(((n + 1) > 0) ? (n + 1) : 2) / 2U;              \
  size_t z_elems = (size_t)ldz_ * (size_t)((z_cols > 0) ? z_cols : 1);         \
  TYPE *ap_col = NULL;                                                          \
  TYPE *z_col = NULL;                                                           \
  TYPE *work = NULL;                                                            \
  int *iwork = NULL;                                                            \
  TYPE dummy_z = (TYPE)0;                                                       \
  TYPE *ap_arg = ap;                                                            \
  TYPE *z_arg = z ? z : &dummy_z;                                               \
  int dummy_ifail = 0;                                                          \
  int *ifail_ptr = ifail ? ifail : &dummy_ifail;                                \
  int info = 0;                                                                 \
  if (!m || !w || !g_fortran_fn[OP_ID]) return -1;                              \
  if (fb_lapack_uplo_to_char(uplo, &uplo_char) != 0) return -1;                \
  if (vectors_requested && (!z || !ifail)) return -1;                          \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if (n > 0 && !ap) return -1;                                                \
    if (vectors_requested && ldz < z_cols) return -1;                           \
    ap_col = (TYPE *)malloc(packed_len * sizeof(TYPE));                         \
    if (!ap_col) return -101;                                                   \
    if (fb_copy_triangular_packed_to_col_major(ap, layout, uplo_char, 'N', n,  \
                                               ap_col, sizeof(TYPE)) != 0) {    \
      free(ap_col);                                                             \
      return -1;                                                                \
    }                                                                           \
    ap_arg = ap_col;                                                            \
    if (vectors_requested) {                                                    \
      z_col = (TYPE *)malloc(z_elems * sizeof(TYPE));                           \
      if (!z_col) {                                                             \
        free(ap_col);                                                           \
        return -101;                                                            \
      }                                                                         \
      z_arg = z_col;                                                            \
    } else {                                                                    \
      z_arg = &dummy_z;                                                         \
      ldz_ = 1;                                                                 \
    }                                                                           \
  } else if (fb_lapack_layout_is_col_major(layout)) {                           \
    if (n > 0 && !ap) return -1;                                                \
    if (vectors_requested) {                                                    \
      if (!z || ldz < n) return -1;                                             \
      z_arg = z;                                                                \
      ldz_ = ldz;                                                               \
    } else {                                                                    \
      z_arg = &dummy_z;                                                         \
      ldz_ = 1;                                                                 \
    }                                                                           \
  } else {                                                                      \
    return -1;                                                                  \
  }                                                                             \
  work = (TYPE *)malloc((size_t)work_size * sizeof(TYPE));                     \
  iwork = (int *)malloc((size_t)iwork_size * sizeof(int));                     \
  if (!work || !iwork) {                                                        \
    free(ap_col);                                                               \
    free(z_col);                                                                \
    free(work);                                                                 \
    free(iwork);                                                                \
    return -101;                                                                \
  }                                                                             \
  ((void (*)(char *, char *, char *, int *, TYPE *, TYPE *, TYPE *, int *,     \
             int *, TYPE *, int *, TYPE *, TYPE *, int *, TYPE *, int *,       \
             int *, int *))g_fortran_fn[OP_ID])(                               \
      &jobz_, &range_, &uplo_char, &n_, ap_arg, &vl_, &vu_, &il_, &iu_,        \
      &abstol_, m, w, z_arg, &ldz_, work, iwork, ifail_ptr, &info);            \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if (fb_copy_triangular_packed_from_col_major(ap_col, layout, uplo_char,    \
                                                 'N', n, ap, sizeof(TYPE)) != 0 || \
        (vectors_requested &&                                                   \
         fb_copy_matrix_from_col_major(z_col, n, z_cols, ldz_, layout, z,      \
                                       ldz, sizeof(TYPE)) != 0)) {              \
      free(ap_col);                                                             \
      free(z_col);                                                              \
      free(work);                                                               \
      free(iwork);                                                              \
      return -1;                                                                \
    }                                                                           \
  }                                                                             \
  free(ap_col);                                                                 \
  free(z_col);                                                                  \
  free(work);                                                                   \
  free(iwork);                                                                  \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    char *jobz, char *range, char *uplo, int *n, TYPE *ap, TYPE *vl, TYPE *vu, \
    int *il, int *iu, TYPE *abstol, int *m, TYPE *w, TYPE *z, int *ldz,        \
    TYPE *work, int *iwork, int *ifail, int *info) {                            \
  fb_uplo_t uplo_enum = FB_UPPER;                                               \
  TYPE dummy_z = (TYPE)0;                                                       \
  TYPE *z_ptr = z ? z : &dummy_z;                                               \
  int ldz_value = (z && ldz) ? *ldz : 1;                                        \
  int dummy_ifail = 0;                                                          \
  int *ifail_ptr = ifail ? ifail : &dummy_ifail;                                \
  (void)work;                                                                   \
  (void)iwork;                                                                  \
  if (!jobz || !range || !uplo || !n || !vl || !vu || !il || !iu ||           \
      !abstol || !m || !w ||                                                    \
      fb_lapack_char_to_uplo(*uplo, &uplo_enum) != 0) {                        \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (fb_lapack_job_requests_vectors(*jobz) && (!z || !ldz || !ifail)) {       \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, char, char, fb_uplo_t, int, TYPE *, TYPE,    \
                     TYPE, int, int, TYPE, int *, TYPE *, TYPE *, int,         \
                     int *))g_cblas_fn[OP_ID])(                                \
      FB_LAYOUT_COL_MAJOR, *jobz, *range, uplo_enum, *n, ap, *vl, *vu, *il,   \
      *iu, *abstol, m, w, z_ptr, ldz_value, ifail_ptr);                        \
  if (info) *info = rc;                                                         \
}

#define DEFINE_SBEVD_REAL_THUNKS(OP, TYPE, OP_ID)                               \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const char jobz, const fb_uplo_t uplo,           \
    const int n, const int kd, TYPE *ab, const int ldab, TYPE *w, TYPE *z,     \
    const int ldz) {                                                            \
  char jobz_ = jobz;                                                            \
  char uplo_char = 'U';                                                         \
  int n_ = n;                                                                   \
  int kd_ = kd;                                                                 \
  int ldab_ = (kd >= 0) ? (kd + 1) : 1;                                         \
  int ldz_ = (n > 0) ? n : 1;                                                   \
  int vectors_requested = fb_lapack_job_requests_vectors(jobz_);                \
  int lwork_ = (n <= 1) ? 1 : (vectors_requested ? ((5 * n) + 1 + (2 * n * n)) \
                                                    : (2 * n));                 \
  int liwork_ = (n <= 1) ? 1 : (vectors_requested ? ((5 * n) + 3) : 1);        \
  size_t ab_elems = (size_t)ldab_ * (size_t)((n > 0) ? n : 1);                 \
  size_t z_elems = (size_t)ldz_ * (size_t)((n > 0) ? n : 1);                   \
  TYPE *ab_col = NULL;                                                          \
  TYPE *z_col = NULL;                                                           \
  TYPE *work = NULL;                                                            \
  int *iwork = NULL;                                                            \
  TYPE dummy_z = (TYPE)0;                                                       \
  TYPE *ab_arg = ab;                                                            \
  TYPE *z_arg = vectors_requested ? z : &dummy_z;                               \
  int info = 0;                                                                 \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (fb_lapack_uplo_to_char(uplo, &uplo_char) != 0 || !w) return -1;          \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if ((n > 0 && !ab) || ldab < n) return -1;                                 \
    if (vectors_requested && (!z || ldz < n)) return -1;                       \
    ab_col = (TYPE *)malloc(ab_elems * sizeof(TYPE));                           \
    if (!ab_col) return -101;                                                   \
    if (fb_copy_sym_band_to_col_major(ab, layout, uplo, n, kd, ldab,           \
                                      ab_col, ldab_, sizeof(TYPE)) != 0) {      \
      free(ab_col);                                                             \
      return -1;                                                                \
    }                                                                           \
    ab_arg = ab_col;                                                            \
    if (vectors_requested) {                                                    \
      z_col = (TYPE *)malloc(z_elems * sizeof(TYPE));                           \
      if (!z_col) {                                                             \
        free(ab_col);                                                           \
        return -101;                                                            \
      }                                                                         \
      z_arg = z_col;                                                            \
    } else {                                                                    \
      z_arg = &dummy_z;                                                         \
      ldz_ = 1;                                                                 \
    }                                                                           \
  } else if (fb_lapack_layout_is_col_major(layout)) {                           \
    ldab_ = ldab;                                                               \
    if (n > 0 && !ab) return -1;                                                \
    if (vectors_requested) {                                                    \
      if (!z || ldz < n) return -1;                                             \
      z_arg = z;                                                                \
      ldz_ = ldz;                                                               \
    } else {                                                                    \
      z_arg = &dummy_z;                                                         \
      ldz_ = 1;                                                                 \
    }                                                                           \
  } else {                                                                      \
    return -1;                                                                  \
  }                                                                             \
  work = (TYPE *)malloc((size_t)lwork_ * sizeof(TYPE));                        \
  iwork = (int *)malloc((size_t)liwork_ * sizeof(int));                        \
  if (!work || !iwork) {                                                        \
    free(ab_col);                                                               \
    free(z_col);                                                                \
    free(work);                                                                 \
    free(iwork);                                                                \
    return -101;                                                                \
  }                                                                             \
  ((void (*)(char *, char *, int *, int *, TYPE *, int *, TYPE *, TYPE *,      \
             int *, TYPE *, int *, int *, int *, int *))g_fortran_fn[OP_ID])(  \
      &jobz_, &uplo_char, &n_, &kd_, ab_arg, &ldab_, w, z_arg, &ldz_, work,    \
      &lwork_, iwork, &liwork_, &info);                                         \
  if (layout == FB_LAYOUT_ROW_MAJOR && vectors_requested && info == 0) {       \
    if (fb_copy_matrix_from_col_major(z_col, n, n, ldz_, layout,               \
                                      z, ldz, sizeof(TYPE)) != 0) {             \
      free(ab_col);                                                             \
      free(z_col);                                                              \
      free(work);                                                               \
      free(iwork);                                                              \
      return -1;                                                                \
    }                                                                           \
  }                                                                             \
  free(ab_col);                                                                 \
  free(z_col);                                                                  \
  free(work);                                                                   \
  free(iwork);                                                                  \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    char *jobz, char *uplo, int *n, int *kd, TYPE *ab, int *ldab, TYPE *w,     \
    TYPE *z, int *ldz, TYPE *work, int *lwork, int *iwork, int *liwork,        \
    int *info) {                                                                \
  fb_uplo_t uplo_enum = FB_UPPER;                                               \
  TYPE dummy_z = (TYPE)0;                                                       \
  TYPE *z_ptr = z ? z : &dummy_z;                                               \
  int ldz_value = (z && ldz) ? *ldz : 1;                                        \
  (void)work;                                                                   \
  (void)lwork;                                                                  \
  (void)iwork;                                                                  \
  (void)liwork;                                                                 \
  if (!jobz || !uplo || !n || !kd || !ldab || !w ||                            \
      fb_lapack_char_to_uplo(*uplo, &uplo_enum) != 0) {                        \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (fb_lapack_job_requests_vectors(*jobz) && (!z || !ldz)) {                 \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, char, fb_uplo_t, int, int, TYPE *, int,      \
                     TYPE *, TYPE *, int))g_cblas_fn[OP_ID])(                   \
      FB_LAYOUT_COL_MAJOR, *jobz, uplo_enum, *n, *kd, ab, *ldab, w, z_ptr,     \
      ldz_value);                                                               \
  if (info) *info = rc;                                                         \
}

#define DEFINE_SBEVX_REAL_THUNKS(OP, TYPE, OP_ID)                               \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const char jobz, const char range,               \
    const fb_uplo_t uplo, const int n, const int kd, TYPE *ab, const int ldab, \
    TYPE *q, const int ldq, const TYPE vl, const TYPE vu, const int il,        \
    const int iu, const TYPE abstol, int *m, TYPE *w, TYPE *z, const int ldz,  \
    int *ifail) {                                                               \
  char jobz_ = jobz, range_ = range;                                            \
  char uplo_char = 'U';                                                         \
  int n_ = n, kd_ = kd, il_ = il, iu_ = iu;                                     \
  int ldab_ = (kd >= 0) ? (kd + 1) : 1;                                         \
  int ldq_ = (n > 0) ? n : 1;                                                   \
  int z_cols = ((range == 'A' || range == 'a' || range == 'V' ||               \
                 range == 'v')                                                  \
                    ? ((n > 0) ? n : 1)                                         \
                    : ((range == 'I' || range == 'i')                           \
                           ? (((iu >= il) && n > 0) ? (iu - il + 1) : 1)        \
                           : 1));                                               \
  int ldz_ = (n > 0) ? n : 1;                                                   \
  int vectors_requested = fb_lapack_job_requests_vectors(jobz_);                \
  int work_size = (n > 0) ? (7 * n) : 1;                                        \
  int iwork_size = (n > 0) ? (5 * n) : 1;                                       \
  TYPE vl_ = vl, vu_ = vu, abstol_ = abstol;                                    \
  size_t ab_elems = (size_t)ldab_ * (size_t)((n > 0) ? n : 1);                 \
  size_t q_elems = (size_t)ldq_ * (size_t)((n > 0) ? n : 1);                   \
  size_t z_elems = (size_t)ldz_ * (size_t)((z_cols > 0) ? z_cols : 1);         \
  TYPE *ab_col = NULL;                                                          \
  TYPE *q_col = NULL;                                                           \
  TYPE *z_col = NULL;                                                           \
  TYPE *work = NULL;                                                            \
  int *iwork = NULL;                                                            \
  TYPE dummy_q = (TYPE)0;                                                       \
  TYPE dummy_z = (TYPE)0;                                                       \
  TYPE *ab_arg = ab;                                                            \
  TYPE *q_arg = q ? q : &dummy_q;                                               \
  TYPE *z_arg = z ? z : &dummy_z;                                               \
  int dummy_ifail = 0;                                                          \
  int *ifail_ptr = ifail ? ifail : &dummy_ifail;                                \
  int info = 0;                                                                 \
  if (!m || !w || !g_fortran_fn[OP_ID]) return -1;                              \
  if (fb_lapack_uplo_to_char(uplo, &uplo_char) != 0) return -1;                \
  if (vectors_requested && (!q || !z || !ifail)) return -1;                    \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if ((n > 0 && !ab) || ldab < n) return -1;                                 \
    if (vectors_requested && (ldq < n || ldz < z_cols)) return -1;             \
    ab_col = (TYPE *)malloc(ab_elems * sizeof(TYPE));                           \
    if (vectors_requested) {                                                    \
      q_col = (TYPE *)malloc(q_elems * sizeof(TYPE));                           \
      z_col = (TYPE *)malloc(z_elems * sizeof(TYPE));                           \
    }                                                                           \
    if (!ab_col || (vectors_requested && (!q_col || !z_col))) {                \
      free(ab_col);                                                             \
      free(q_col);                                                              \
      free(z_col);                                                              \
      return -101;                                                              \
    }                                                                           \
    if (fb_copy_sym_band_to_col_major(ab, layout, uplo, n, kd, ldab,           \
                                      ab_col, ldab_, sizeof(TYPE)) != 0) {      \
      free(ab_col);                                                             \
      free(q_col);                                                              \
      free(z_col);                                                              \
      return -1;                                                                \
    }                                                                           \
    ab_arg = ab_col;                                                            \
    if (vectors_requested) {                                                    \
      q_arg = q_col;                                                            \
      z_arg = z_col;                                                            \
    } else {                                                                    \
      q_arg = &dummy_q;                                                         \
      z_arg = &dummy_z;                                                         \
      ldq_ = 1;                                                                 \
      ldz_ = 1;                                                                 \
    }                                                                           \
  } else if (fb_lapack_layout_is_col_major(layout)) {                           \
    ldab_ = ldab;                                                               \
    if (n > 0 && !ab) return -1;                                                \
    if (vectors_requested) {                                                    \
      if (ldq < n || ldz < n) return -1;                                        \
      ldq_ = ldq;                                                               \
      ldz_ = ldz;                                                               \
      q_arg = q;                                                                \
      z_arg = z;                                                                \
    } else {                                                                    \
      q_arg = &dummy_q;                                                         \
      z_arg = &dummy_z;                                                         \
      ldq_ = 1;                                                                 \
      ldz_ = 1;                                                                 \
    }                                                                           \
  } else {                                                                      \
    return -1;                                                                  \
  }                                                                             \
  work = (TYPE *)malloc((size_t)work_size * sizeof(TYPE));                     \
  iwork = (int *)malloc((size_t)iwork_size * sizeof(int));                     \
  if (!work || !iwork) {                                                        \
    free(ab_col);                                                               \
    free(q_col);                                                                \
    free(z_col);                                                                \
    free(work);                                                                 \
    free(iwork);                                                                \
    return -101;                                                                \
  }                                                                             \
  ((void (*)(char *, char *, char *, int *, int *, TYPE *, int *, TYPE *,      \
             int *, TYPE *, TYPE *, int *, int *, TYPE *, int *, TYPE *,       \
             TYPE *, int *, TYPE *, int *, int *, int *))g_fortran_fn[OP_ID])( \
      &jobz_, &range_, &uplo_char, &n_, &kd_, ab_arg, &ldab_, q_arg, &ldq_,    \
      &vl_, &vu_, &il_, &iu_, &abstol_, m, w, z_arg, &ldz_, work, iwork,       \
      ifail_ptr, &info);                                                        \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if (fb_copy_sym_band_from_col_major(ab_col, layout, uplo, n, kd, ldab_,    \
                                        ab, ldab, sizeof(TYPE)) != 0 ||         \
        (vectors_requested &&                                                   \
         fb_copy_matrix_from_col_major(q_col, n, n, ldq_, layout, q, ldq,      \
                                       sizeof(TYPE)) != 0) ||                   \
        (vectors_requested &&                                                   \
         fb_copy_matrix_from_col_major(z_col, n, z_cols, ldz_, layout, z,      \
                                       ldz, sizeof(TYPE)) != 0)) {              \
      free(ab_col);                                                             \
      free(q_col);                                                              \
      free(z_col);                                                              \
      free(work);                                                               \
      free(iwork);                                                              \
      return -1;                                                                \
    }                                                                           \
  }                                                                             \
  free(ab_col);                                                                 \
  free(q_col);                                                                  \
  free(z_col);                                                                  \
  free(work);                                                                   \
  free(iwork);                                                                  \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    char *jobz, char *range, char *uplo, int *n, int *kd, TYPE *ab, int *ldab, \
    TYPE *q, int *ldq, TYPE *vl, TYPE *vu, int *il, int *iu, TYPE *abstol,    \
    int *m, TYPE *w, TYPE *z, int *ldz, TYPE *work, int *iwork, int *ifail,    \
    int *info) {                                                                \
  fb_uplo_t uplo_enum = FB_UPPER;                                               \
  TYPE dummy_q = (TYPE)0;                                                       \
  TYPE dummy_z = (TYPE)0;                                                       \
  TYPE *q_ptr = q ? q : &dummy_q;                                               \
  TYPE *z_ptr = z ? z : &dummy_z;                                               \
  int ldq_value = (q && ldq) ? *ldq : 1;                                        \
  int ldz_value = (z && ldz) ? *ldz : 1;                                        \
  int dummy_ifail = 0;                                                          \
  int *ifail_ptr = ifail ? ifail : &dummy_ifail;                                \
  (void)work;                                                                   \
  (void)iwork;                                                                  \
  if (!jobz || !range || !uplo || !n || !kd || !ldab || !vl || !vu ||          \
      !il || !iu || !abstol || !m || !w ||                                      \
      fb_lapack_char_to_uplo(*uplo, &uplo_enum) != 0) {                        \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (fb_lapack_job_requests_vectors(*jobz) &&                                 \
      (!q || !ldq || !z || !ldz || !ifail)) {                                  \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, char, char, fb_uplo_t, int, int, TYPE *,     \
                     int, TYPE *, int, TYPE, TYPE, int, int, TYPE, int *,      \
                     TYPE *, TYPE *, int, int *))g_cblas_fn[OP_ID])(           \
      FB_LAYOUT_COL_MAJOR, *jobz, *range, uplo_enum, *n, *kd, ab, *ldab,       \
      q_ptr, ldq_value, *vl, *vu, *il, *iu, *abstol, m, w, z_ptr, ldz_value,   \
      ifail_ptr);                                                               \
  if (info) *info = rc;                                                         \
}

#define DEFINE_SBGV_REAL_THUNKS(OP, TYPE, OP_ID)                                \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const char jobz, const fb_uplo_t uplo,           \
    const int n, const int ka, const int kb, TYPE *ab, const int ldab,         \
    TYPE *bb, const int ldbb, TYPE *w, TYPE *z, const int ldz) {               \
  char jobz_ = jobz;                                                            \
  char uplo_char = 'U';                                                         \
  int n_ = n;                                                                   \
  int ka_ = ka;                                                                 \
  int kb_ = kb;                                                                 \
  int ldab_ = (ka >= 0) ? (ka + 1) : 1;                                         \
  int ldbb_ = (kb >= 0) ? (kb + 1) : 1;                                         \
  int ldz_ = (n > 0) ? n : 1;                                                   \
  int vectors_requested = fb_lapack_job_requests_vectors(jobz_);                \
  int work_len = (n > 0) ? (3 * n) : 1;                                         \
  size_t ab_elems = (size_t)ldab_ * (size_t)((n > 0) ? n : 1);                 \
  size_t bb_elems = (size_t)ldbb_ * (size_t)((n > 0) ? n : 1);                 \
  size_t z_elems = (size_t)ldz_ * (size_t)((n > 0) ? n : 1);                   \
  TYPE *ab_col = NULL;                                                          \
  TYPE *bb_col = NULL;                                                          \
  TYPE *z_col = NULL;                                                           \
  TYPE *work = NULL;                                                            \
  TYPE dummy_z = (TYPE)0;                                                       \
  TYPE *ab_arg = ab;                                                            \
  TYPE *bb_arg = bb;                                                            \
  TYPE *z_arg = vectors_requested ? z : &dummy_z;                               \
  int info = 0;                                                                 \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (fb_lapack_uplo_to_char(uplo, &uplo_char) != 0 || !w) return -1;          \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if ((n > 0 && (!ab || !bb)) || ldab < n || ldbb < n) return -1;            \
    if (vectors_requested && (!z || ldz < n)) return -1;                       \
    ab_col = (TYPE *)malloc(ab_elems * sizeof(TYPE));                           \
    bb_col = (TYPE *)malloc(bb_elems * sizeof(TYPE));                           \
    if (vectors_requested) {                                                    \
      z_col = (TYPE *)malloc(z_elems * sizeof(TYPE));                           \
    }                                                                           \
    if (!ab_col || !bb_col || (vectors_requested && !z_col)) {                 \
      free(ab_col);                                                             \
      free(bb_col);                                                             \
      free(z_col);                                                              \
      return -101;                                                              \
    }                                                                           \
    if (fb_copy_sym_band_to_col_major(ab, layout, uplo, n, ka, ldab,           \
                                      ab_col, ldab_, sizeof(TYPE)) != 0 ||      \
        fb_copy_sym_band_to_col_major(bb, layout, uplo, n, kb, ldbb,           \
                                      bb_col, ldbb_, sizeof(TYPE)) != 0) {      \
      free(ab_col);                                                             \
      free(bb_col);                                                             \
      free(z_col);                                                              \
      return -1;                                                                \
    }                                                                           \
    ab_arg = ab_col;                                                            \
    bb_arg = bb_col;                                                            \
    if (vectors_requested) {                                                    \
      z_arg = z_col;                                                            \
    } else {                                                                    \
      z_arg = &dummy_z;                                                         \
      ldz_ = 1;                                                                 \
    }                                                                           \
  } else if (fb_lapack_layout_is_col_major(layout)) {                           \
    ldab_ = ldab;                                                               \
    ldbb_ = ldbb;                                                               \
    if (n > 0 && (!ab || !bb)) return -1;                                       \
    if (vectors_requested) {                                                    \
      if (!z || ldz < n) return -1;                                             \
      z_arg = z;                                                                \
      ldz_ = ldz;                                                               \
    } else {                                                                    \
      z_arg = &dummy_z;                                                         \
      ldz_ = 1;                                                                 \
    }                                                                           \
  } else {                                                                      \
    return -1;                                                                  \
  }                                                                             \
  work = (TYPE *)malloc((size_t)work_len * sizeof(TYPE));                      \
  if (!work) {                                                                  \
    free(ab_col);                                                               \
    free(bb_col);                                                               \
    free(z_col);                                                                \
    return -101;                                                                \
  }                                                                             \
  ((void (*)(char *, char *, int *, int *, int *, TYPE *, int *, TYPE *,       \
             int *, TYPE *, TYPE *, int *, TYPE *, int *))g_fortran_fn[OP_ID])(\
      &jobz_, &uplo_char, &n_, &ka_, &kb_, ab_arg, &ldab_, bb_arg, &ldbb_, w,  \
      z_arg, &ldz_, work, &info);                                               \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if (fb_copy_sym_band_from_col_major(ab_col, layout, uplo, n, ka, ldab_,    \
                                        ab, ldab, sizeof(TYPE)) != 0 ||         \
        fb_copy_sym_band_from_col_major(bb_col, layout, uplo, n, kb, ldbb_,    \
                                        bb, ldbb, sizeof(TYPE)) != 0 ||         \
        (vectors_requested &&                                                   \
         fb_copy_matrix_from_col_major(z_col, n, n, ldz_, layout, z, ldz,      \
                                       sizeof(TYPE)) != 0)) {                   \
      free(ab_col);                                                             \
      free(bb_col);                                                             \
      free(z_col);                                                              \
      free(work);                                                               \
      return -1;                                                                \
    }                                                                           \
  }                                                                             \
  free(ab_col);                                                                 \
  free(bb_col);                                                                 \
  free(z_col);                                                                  \
  free(work);                                                                   \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    char *jobz, char *uplo, int *n, int *ka, int *kb, TYPE *ab, int *ldab,     \
    TYPE *bb, int *ldbb, TYPE *w, TYPE *z, int *ldz, TYPE *work, int *info) {  \
  fb_uplo_t uplo_enum = FB_UPPER;                                               \
  TYPE dummy_z = (TYPE)0;                                                       \
  TYPE *z_ptr = z ? z : &dummy_z;                                               \
  int ldz_value = (z && ldz) ? *ldz : 1;                                        \
  (void)work;                                                                   \
  if (!jobz || !uplo || !n || !ka || !kb || !ldab || !ldbb || !w ||            \
      fb_lapack_char_to_uplo(*uplo, &uplo_enum) != 0) {                        \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (fb_lapack_job_requests_vectors(*jobz) && (!z || !ldz)) {                 \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, char, fb_uplo_t, int, int, int, TYPE *,      \
                     int, TYPE *, int, TYPE *, TYPE *, int))g_cblas_fn[OP_ID])(\
      FB_LAYOUT_COL_MAJOR, *jobz, uplo_enum, *n, *ka, *kb, ab, *ldab, bb,      \
      *ldbb, w, z_ptr, ldz_value);                                              \
  if (info) *info = rc;                                                         \
}

#define DEFINE_HBGV_COMPLEX_THUNKS(OP, CTYPE, RTYPE, OP_ID)                     \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const char jobz, const fb_uplo_t uplo,           \
    const int n, const int ka, const int kb, CTYPE *ab, const int ldab,        \
    CTYPE *bb, const int ldbb, RTYPE *w, CTYPE *z, const int ldz) {            \
  char jobz_ = jobz;                                                            \
  char uplo_char = 'U';                                                         \
  int n_ = n;                                                                   \
  int ka_ = ka;                                                                 \
  int kb_ = kb;                                                                 \
  int ldab_ = (ka >= 0) ? (ka + 1) : 1;                                         \
  int ldbb_ = (kb >= 0) ? (kb + 1) : 1;                                         \
  int ldz_ = (n > 0) ? n : 1;                                                   \
  int vectors_requested = fb_lapack_job_requests_vectors(jobz_);                \
  int work_len = (n > 0) ? n : 1;                                               \
  int rwork_len = (n > 0) ? (3 * n) : 1;                                        \
  size_t ab_elems = (size_t)ldab_ * (size_t)((n > 0) ? n : 1);                 \
  size_t bb_elems = (size_t)ldbb_ * (size_t)((n > 0) ? n : 1);                 \
  size_t z_elems = (size_t)ldz_ * (size_t)((n > 0) ? n : 1);                   \
  CTYPE *ab_col = NULL;                                                         \
  CTYPE *bb_col = NULL;                                                         \
  CTYPE *z_col = NULL;                                                          \
  CTYPE *work = NULL;                                                           \
  RTYPE *rwork = NULL;                                                          \
  CTYPE dummy_z = (CTYPE)0;                                                     \
  CTYPE *ab_arg = ab;                                                           \
  CTYPE *bb_arg = bb;                                                           \
  CTYPE *z_arg = vectors_requested ? z : &dummy_z;                              \
  int info = 0;                                                                 \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (fb_lapack_uplo_to_char(uplo, &uplo_char) != 0 || !w) return -1;          \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if ((n > 0 && (!ab || !bb)) || ldab < n || ldbb < n) return -1;            \
    if (vectors_requested && (!z || ldz < n)) return -1;                       \
    ab_col = (CTYPE *)malloc(ab_elems * sizeof(CTYPE));                         \
    bb_col = (CTYPE *)malloc(bb_elems * sizeof(CTYPE));                         \
    if (vectors_requested) {                                                    \
      z_col = (CTYPE *)malloc(z_elems * sizeof(CTYPE));                         \
    }                                                                           \
    if (!ab_col || !bb_col || (vectors_requested && !z_col)) {                 \
      free(ab_col);                                                             \
      free(bb_col);                                                             \
      free(z_col);                                                              \
      return -101;                                                              \
    }                                                                           \
    if (fb_copy_sym_band_to_col_major(ab, layout, uplo, n, ka, ldab,           \
                                      ab_col, ldab_, sizeof(CTYPE)) != 0 ||    \
        fb_copy_sym_band_to_col_major(bb, layout, uplo, n, kb, ldbb,           \
                                      bb_col, ldbb_, sizeof(CTYPE)) != 0) {    \
      free(ab_col);                                                             \
      free(bb_col);                                                             \
      free(z_col);                                                              \
      return -1;                                                                \
    }                                                                           \
    ab_arg = ab_col;                                                            \
    bb_arg = bb_col;                                                            \
    if (vectors_requested) {                                                    \
      z_arg = z_col;                                                            \
    } else {                                                                    \
      z_arg = &dummy_z;                                                         \
      ldz_ = 1;                                                                 \
    }                                                                           \
  } else if (fb_lapack_layout_is_col_major(layout)) {                           \
    ldab_ = ldab;                                                               \
    ldbb_ = ldbb;                                                               \
    if (n > 0 && (!ab || !bb)) return -1;                                       \
    if (vectors_requested) {                                                    \
      if (!z || ldz < n) return -1;                                             \
      z_arg = z;                                                                \
      ldz_ = ldz;                                                               \
    } else {                                                                    \
      z_arg = &dummy_z;                                                         \
      ldz_ = 1;                                                                 \
    }                                                                           \
  } else {                                                                      \
    return -1;                                                                  \
  }                                                                             \
  work = (CTYPE *)malloc((size_t)work_len * sizeof(CTYPE));                     \
  rwork = (RTYPE *)malloc((size_t)rwork_len * sizeof(RTYPE));                   \
  if (!work || !rwork) {                                                        \
    free(ab_col);                                                               \
    free(bb_col);                                                               \
    free(z_col);                                                                \
    free(work);                                                                 \
    free(rwork);                                                                \
    return -101;                                                                \
  }                                                                             \
  ((void (*)(char *, char *, int *, int *, int *, CTYPE *, int *, CTYPE *,     \
             int *, RTYPE *, CTYPE *, int *, CTYPE *, RTYPE *, int *))         \
      g_fortran_fn[OP_ID])(&jobz_, &uplo_char, &n_, &ka_, &kb_, ab_arg,       \
                           &ldab_, bb_arg, &ldbb_, w, z_arg, &ldz_, work,      \
                           rwork, &info);                                       \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if (fb_copy_sym_band_from_col_major(ab_col, layout, uplo, n, ka, ldab_,    \
                                        ab, ldab, sizeof(CTYPE)) != 0 ||       \
        fb_copy_sym_band_from_col_major(bb_col, layout, uplo, n, kb, ldbb_,    \
                                        bb, ldbb, sizeof(CTYPE)) != 0 ||       \
        (vectors_requested &&                                                   \
         fb_copy_matrix_from_col_major(z_col, n, n, ldz_, layout, z, ldz,      \
                                       sizeof(CTYPE)) != 0)) {                  \
      free(ab_col);                                                             \
      free(bb_col);                                                             \
      free(z_col);                                                              \
      free(work);                                                               \
      free(rwork);                                                              \
      return -1;                                                                \
    }                                                                           \
  }                                                                             \
  free(ab_col);                                                                 \
  free(bb_col);                                                                 \
  free(z_col);                                                                  \
  free(work);                                                                   \
  free(rwork);                                                                  \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    char *jobz, char *uplo, int *n, int *ka, int *kb, CTYPE *ab, int *ldab,    \
    CTYPE *bb, int *ldbb, RTYPE *w, CTYPE *z, int *ldz, CTYPE *work,           \
    RTYPE *rwork, int *info) {                                                  \
  fb_uplo_t uplo_enum = FB_UPPER;                                               \
  CTYPE dummy_z = (CTYPE)0;                                                     \
  CTYPE *z_ptr = z ? z : &dummy_z;                                              \
  int ldz_value = (z && ldz) ? *ldz : 1;                                        \
  (void)work;                                                                   \
  (void)rwork;                                                                  \
  if (!jobz || !uplo || !n || !ka || !kb || !ldab || !ldbb || !w ||            \
      fb_lapack_char_to_uplo(*uplo, &uplo_enum) != 0) {                        \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (fb_lapack_job_requests_vectors(*jobz) && (!z || !ldz)) {                 \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, char, fb_uplo_t, int, int, int, CTYPE *,     \
                     int, CTYPE *, int, RTYPE *, CTYPE *, int))g_cblas_fn[OP_ID])(\
      FB_LAYOUT_COL_MAJOR, *jobz, uplo_enum, *n, *ka, *kb, ab, *ldab, bb,      \
      *ldbb, w, z_ptr, ldz_value);                                              \
  if (info) *info = rc;                                                         \
}

#define DEFINE_HBGVD_COMPLEX_THUNKS(OP, CTYPE, RTYPE, OP_ID, QUERY_FN)         \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const char jobz, const fb_uplo_t uplo,           \
    const int n, const int ka, const int kb, CTYPE *ab, const int ldab,        \
    CTYPE *bb, const int ldbb, RTYPE *w, CTYPE *z, const int ldz) {            \
  char jobz_ = jobz;                                                            \
  char uplo_char = 'U';                                                         \
  int n_ = n;                                                                   \
  int ka_ = ka;                                                                 \
  int kb_ = kb;                                                                 \
  int ldab_ = (ka >= 0) ? (ka + 1) : 1;                                         \
  int ldbb_ = (kb >= 0) ? (kb + 1) : 1;                                         \
  int ldz_ = (n > 0) ? n : 1;                                                   \
  int vectors_requested = fb_lapack_job_requests_vectors(jobz_);                \
  int lwork_ = -1, lrwork_ = -1, liwork_ = -1, info = 0;                       \
  CTYPE work_query = (CTYPE)0;                                                  \
  RTYPE rwork_query = (RTYPE)0;                                                 \
  int iwork_query = 0;                                                          \
  size_t ab_elems = (size_t)ldab_ * (size_t)((n > 0) ? n : 1);                 \
  size_t bb_elems = (size_t)ldbb_ * (size_t)((n > 0) ? n : 1);                 \
  size_t z_elems = (size_t)ldz_ * (size_t)((n > 0) ? n : 1);                   \
  CTYPE *ab_col = NULL;                                                         \
  CTYPE *bb_col = NULL;                                                         \
  CTYPE *z_col = NULL;                                                          \
  CTYPE *work = NULL;                                                           \
  RTYPE *rwork = NULL;                                                          \
  int *iwork = NULL;                                                            \
  CTYPE dummy_z = (CTYPE)0;                                                     \
  CTYPE *ab_arg = ab;                                                           \
  CTYPE *bb_arg = bb;                                                           \
  CTYPE *z_arg = vectors_requested ? z : &dummy_z;                              \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (fb_lapack_uplo_to_char(uplo, &uplo_char) != 0 || !w) return -1;          \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if ((n > 0 && (!ab || !bb)) || ldab < n || ldbb < n) return -1;            \
    if (vectors_requested && (!z || ldz < n)) return -1;                       \
    ab_col = (CTYPE *)malloc(ab_elems * sizeof(CTYPE));                         \
    bb_col = (CTYPE *)malloc(bb_elems * sizeof(CTYPE));                         \
    if (vectors_requested) {                                                    \
      z_col = (CTYPE *)malloc(z_elems * sizeof(CTYPE));                         \
    }                                                                           \
    if (!ab_col || !bb_col || (vectors_requested && !z_col)) {                 \
      free(ab_col);                                                             \
      free(bb_col);                                                             \
      free(z_col);                                                              \
      return -101;                                                              \
    }                                                                           \
    if (fb_copy_sym_band_to_col_major(ab, layout, uplo, n, ka, ldab,           \
                                      ab_col, ldab_, sizeof(CTYPE)) != 0 ||    \
        fb_copy_sym_band_to_col_major(bb, layout, uplo, n, kb, ldbb,           \
                                      bb_col, ldbb_, sizeof(CTYPE)) != 0) {    \
      free(ab_col);                                                             \
      free(bb_col);                                                             \
      free(z_col);                                                              \
      return -1;                                                                \
    }                                                                           \
    ab_arg = ab_col;                                                            \
    bb_arg = bb_col;                                                            \
    if (vectors_requested) {                                                    \
      z_arg = z_col;                                                            \
    } else {                                                                    \
      z_arg = &dummy_z;                                                         \
      ldz_ = 1;                                                                 \
    }                                                                           \
  } else if (fb_lapack_layout_is_col_major(layout)) {                           \
    ldab_ = ldab;                                                               \
    ldbb_ = ldbb;                                                               \
    if (n > 0 && (!ab || !bb)) return -1;                                       \
    if (vectors_requested) {                                                    \
      if (!z || ldz < n) return -1;                                             \
      z_arg = z;                                                                \
      ldz_ = ldz;                                                               \
    } else {                                                                    \
      z_arg = &dummy_z;                                                         \
      ldz_ = 1;                                                                 \
    }                                                                           \
  } else {                                                                      \
    return -1;                                                                  \
  }                                                                             \
  ((void (*)(char *, char *, int *, int *, int *, CTYPE *, int *, CTYPE *,     \
             int *, RTYPE *, CTYPE *, int *, CTYPE *, int *, RTYPE *, int *,   \
             int *, int *, int *))g_fortran_fn[OP_ID])(                        \
      &jobz_, &uplo_char, &n_, &ka_, &kb_, ab_arg, &ldab_, bb_arg, &ldbb_, w,  \
      z_arg, &ldz_, &work_query, &lwork_, &rwork_query, &lrwork_, &iwork_query, \
      &liwork_, &info);                                                         \
  if (info != 0) {                                                              \
    free(ab_col);                                                               \
    free(bb_col);                                                               \
    free(z_col);                                                                \
    return info;                                                                \
  }                                                                             \
  lwork_ = QUERY_FN(work_query);                                                \
  lrwork_ = (rwork_query > (RTYPE)1) ? (int)rwork_query : 1;                   \
  liwork_ = fb_query_size_from_int(iwork_query);                                \
  work = (CTYPE *)malloc((size_t)lwork_ * sizeof(CTYPE));                       \
  rwork = (RTYPE *)malloc((size_t)lrwork_ * sizeof(RTYPE));                     \
  iwork = (int *)malloc((size_t)liwork_ * sizeof(int));                         \
  if (!work || !rwork || !iwork) {                                              \
    free(ab_col);                                                               \
    free(bb_col);                                                               \
    free(z_col);                                                                \
    free(work);                                                                 \
    free(rwork);                                                                \
    free(iwork);                                                                \
    return -101;                                                                \
  }                                                                             \
  ((void (*)(char *, char *, int *, int *, int *, CTYPE *, int *, CTYPE *,     \
             int *, RTYPE *, CTYPE *, int *, CTYPE *, int *, RTYPE *, int *,   \
             int *, int *, int *))g_fortran_fn[OP_ID])(                        \
      &jobz_, &uplo_char, &n_, &ka_, &kb_, ab_arg, &ldab_, bb_arg, &ldbb_, w,  \
      z_arg, &ldz_, work, &lwork_, rwork, &lrwork_, iwork, &liwork_, &info);   \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if (fb_copy_sym_band_from_col_major(ab_col, layout, uplo, n, ka, ldab_,    \
                                        ab, ldab, sizeof(CTYPE)) != 0 ||       \
        fb_copy_sym_band_from_col_major(bb_col, layout, uplo, n, kb, ldbb_,    \
                                        bb, ldbb, sizeof(CTYPE)) != 0 ||       \
        (vectors_requested &&                                                   \
         fb_copy_matrix_from_col_major(z_col, n, n, ldz_, layout, z, ldz,      \
                                       sizeof(CTYPE)) != 0)) {                  \
      free(ab_col);                                                             \
      free(bb_col);                                                             \
      free(z_col);                                                              \
      free(work);                                                               \
      free(rwork);                                                              \
      free(iwork);                                                              \
      return -1;                                                                \
    }                                                                           \
  }                                                                             \
  free(ab_col);                                                                 \
  free(bb_col);                                                                 \
  free(z_col);                                                                  \
  free(work);                                                                   \
  free(rwork);                                                                  \
  free(iwork);                                                                  \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    char *jobz, char *uplo, int *n, int *ka, int *kb, CTYPE *ab, int *ldab,    \
    CTYPE *bb, int *ldbb, RTYPE *w, CTYPE *z, int *ldz, CTYPE *work,           \
    int *lwork, RTYPE *rwork, int *lrwork, int *iwork, int *liwork, int *info) {\
  fb_uplo_t uplo_enum = FB_UPPER;                                               \
  CTYPE dummy_z = (CTYPE)0;                                                     \
  CTYPE *z_ptr = z ? z : &dummy_z;                                              \
  int ldz_value = (z && ldz) ? *ldz : 1;                                        \
  (void)work;                                                                   \
  (void)lwork;                                                                  \
  (void)rwork;                                                                  \
  (void)lrwork;                                                                 \
  (void)iwork;                                                                  \
  (void)liwork;                                                                 \
  if (!jobz || !uplo || !n || !ka || !kb || !ldab || !ldbb || !w ||            \
      fb_lapack_char_to_uplo(*uplo, &uplo_enum) != 0) {                        \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (fb_lapack_job_requests_vectors(*jobz) && (!z || !ldz)) {                 \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, char, fb_uplo_t, int, int, int, CTYPE *,     \
                     int, CTYPE *, int, RTYPE *, CTYPE *, int))g_cblas_fn[OP_ID])(\
      FB_LAYOUT_COL_MAJOR, *jobz, uplo_enum, *n, *ka, *kb, ab, *ldab, bb,      \
      *ldbb, w, z_ptr, ldz_value);                                              \
  if (info) *info = rc;                                                         \
}

#define DEFINE_HBGVX_COMPLEX_THUNKS(OP, CTYPE, RTYPE, OP_ID)                    \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const char jobz, const char range,               \
    const fb_uplo_t uplo, const int n, const int ka, const int kb,             \
    CTYPE *ab, const int ldab, CTYPE *bb, const int ldbb, CTYPE *q,            \
    const int ldq, const RTYPE vl, const RTYPE vu, const int il, const int iu, \
    const RTYPE abstol, int *m, RTYPE *w, CTYPE *z, const int ldz, int *ifail) {\
  char jobz_ = jobz, range_ = range;                                            \
  char uplo_char = 'U';                                                         \
  int n_ = n, ka_ = ka, kb_ = kb, il_ = il, iu_ = iu;                           \
  int ldab_ = (ka >= 0) ? (ka + 1) : 1;                                         \
  int ldbb_ = (kb >= 0) ? (kb + 1) : 1;                                         \
  int ldq_ = (n > 0) ? n : 1;                                                   \
  int ldz_ = (n > 0) ? n : 1;                                                   \
  int vectors_requested = fb_lapack_job_requests_vectors(jobz_);                \
  int work_len = (n > 0) ? n : 1;                                               \
  int rwork_len = (n > 0) ? (7 * n) : 1;                                        \
  int iwork_len = (n > 0) ? (5 * n) : 1;                                        \
  RTYPE vl_ = vl, vu_ = vu, abstol_ = abstol;                                   \
  size_t ab_elems = (size_t)ldab_ * (size_t)((n > 0) ? n : 1);                 \
  size_t bb_elems = (size_t)ldbb_ * (size_t)((n > 0) ? n : 1);                 \
  size_t q_elems = (size_t)ldq_ * (size_t)((n > 0) ? n : 1);                   \
  size_t z_elems = (size_t)ldz_ * (size_t)((n > 0) ? n : 1);                   \
  CTYPE *ab_col = NULL;                                                         \
  CTYPE *bb_col = NULL;                                                         \
  CTYPE *q_col = NULL;                                                          \
  CTYPE *z_col = NULL;                                                          \
  CTYPE *work = NULL;                                                           \
  RTYPE *rwork = NULL;                                                          \
  int *iwork = NULL;                                                            \
  CTYPE dummy_q = (CTYPE)0;                                                     \
  CTYPE dummy_z = (CTYPE)0;                                                     \
  CTYPE *ab_arg = ab;                                                           \
  CTYPE *bb_arg = bb;                                                           \
  CTYPE *q_arg = q ? q : &dummy_q;                                              \
  CTYPE *z_arg = z ? z : &dummy_z;                                              \
  int dummy_ifail = 0;                                                          \
  int *ifail_ptr = ifail ? ifail : &dummy_ifail;                                \
  int info = 0;                                                                 \
  if (!m || !w || !g_fortran_fn[OP_ID]) return -1;                              \
  if (fb_lapack_uplo_to_char(uplo, &uplo_char) != 0) return -1;                \
  if (vectors_requested && (!q || !z || !ifail)) return -1;                    \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if ((n > 0 && (!ab || !bb)) || ldab < n || ldbb < n) return -1;            \
    if (vectors_requested && (ldq < n || ldz < n)) return -1;                  \
    ab_col = (CTYPE *)malloc(ab_elems * sizeof(CTYPE));                         \
    bb_col = (CTYPE *)malloc(bb_elems * sizeof(CTYPE));                         \
    if (vectors_requested) {                                                    \
      q_col = (CTYPE *)malloc(q_elems * sizeof(CTYPE));                         \
      z_col = (CTYPE *)malloc(z_elems * sizeof(CTYPE));                         \
    }                                                                           \
    if (!ab_col || !bb_col || (vectors_requested && (!q_col || !z_col))) {     \
      free(ab_col);                                                             \
      free(bb_col);                                                             \
      free(q_col);                                                              \
      free(z_col);                                                              \
      return -101;                                                              \
    }                                                                           \
    if (fb_copy_sym_band_to_col_major(ab, layout, uplo, n, ka, ldab,           \
                                      ab_col, ldab_, sizeof(CTYPE)) != 0 ||    \
        fb_copy_sym_band_to_col_major(bb, layout, uplo, n, kb, ldbb,           \
                                      bb_col, ldbb_, sizeof(CTYPE)) != 0) {    \
      free(ab_col);                                                             \
      free(bb_col);                                                             \
      free(q_col);                                                              \
      free(z_col);                                                              \
      return -1;                                                                \
    }                                                                           \
    ab_arg = ab_col;                                                            \
    bb_arg = bb_col;                                                            \
    if (vectors_requested) {                                                    \
      q_arg = q_col;                                                            \
      z_arg = z_col;                                                            \
    } else {                                                                    \
      q_arg = &dummy_q;                                                         \
      z_arg = &dummy_z;                                                         \
      ldq_ = 1;                                                                 \
      ldz_ = 1;                                                                 \
    }                                                                           \
  } else if (fb_lapack_layout_is_col_major(layout)) {                           \
    ldab_ = ldab;                                                               \
    ldbb_ = ldbb;                                                               \
    if (n > 0 && (!ab || !bb)) return -1;                                       \
    if (vectors_requested) {                                                    \
      if (ldq < n || ldz < n) return -1;                                        \
      ldq_ = ldq;                                                               \
      ldz_ = ldz;                                                               \
      q_arg = q;                                                                \
      z_arg = z;                                                                \
    } else {                                                                    \
      q_arg = &dummy_q;                                                         \
      z_arg = &dummy_z;                                                         \
      ldq_ = 1;                                                                 \
      ldz_ = 1;                                                                 \
    }                                                                           \
  } else {                                                                      \
    return -1;                                                                  \
  }                                                                             \
  work = (CTYPE *)malloc((size_t)work_len * sizeof(CTYPE));                     \
  rwork = (RTYPE *)malloc((size_t)rwork_len * sizeof(RTYPE));                   \
  iwork = (int *)malloc((size_t)iwork_len * sizeof(int));                       \
  if (!work || !rwork || !iwork) {                                              \
    free(ab_col);                                                               \
    free(bb_col);                                                               \
    free(q_col);                                                                \
    free(z_col);                                                                \
    free(work);                                                                 \
    free(rwork);                                                                \
    free(iwork);                                                                \
    return -101;                                                                \
  }                                                                             \
  ((void (*)(char *, char *, char *, int *, int *, int *, CTYPE *, int *,      \
             CTYPE *, int *, CTYPE *, int *, RTYPE *, RTYPE *, int *, int *,   \
             RTYPE *, int *, RTYPE *, CTYPE *, int *, CTYPE *, RTYPE *, int *, \
             int *, int *))g_fortran_fn[OP_ID])(                               \
      &jobz_, &range_, &uplo_char, &n_, &ka_, &kb_, ab_arg, &ldab_, bb_arg,    \
      &ldbb_, q_arg, &ldq_, &vl_, &vu_, &il_, &iu_, &abstol_, m, w, z_arg,     \
      &ldz_, work, rwork, iwork, ifail_ptr, &info);                             \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if (fb_copy_sym_band_from_col_major(ab_col, layout, uplo, n, ka, ldab_,    \
                                        ab, ldab, sizeof(CTYPE)) != 0 ||       \
        fb_copy_sym_band_from_col_major(bb_col, layout, uplo, n, kb, ldbb_,    \
                                        bb, ldbb, sizeof(CTYPE)) != 0 ||       \
        (vectors_requested &&                                                   \
         fb_copy_matrix_from_col_major(q_col, n, n, ldq_, layout, q, ldq,      \
                                       sizeof(CTYPE)) != 0) ||                 \
        (vectors_requested &&                                                   \
         fb_copy_matrix_from_col_major(z_col, n, n, ldz_, layout, z, ldz,      \
                                       sizeof(CTYPE)) != 0)) {                  \
      free(ab_col);                                                             \
      free(bb_col);                                                             \
      free(q_col);                                                              \
      free(z_col);                                                              \
      free(work);                                                               \
      free(rwork);                                                              \
      free(iwork);                                                              \
      return -1;                                                                \
    }                                                                           \
  }                                                                             \
  free(ab_col);                                                                 \
  free(bb_col);                                                                 \
  free(q_col);                                                                  \
  free(z_col);                                                                  \
  free(work);                                                                   \
  free(rwork);                                                                  \
  free(iwork);                                                                  \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    char *jobz, char *range, char *uplo, int *n, int *ka, int *kb, CTYPE *ab,  \
    int *ldab, CTYPE *bb, int *ldbb, CTYPE *q, int *ldq, RTYPE *vl, RTYPE *vu, \
    int *il, int *iu, RTYPE *abstol, int *m, RTYPE *w, CTYPE *z, int *ldz,     \
    CTYPE *work, RTYPE *rwork, int *iwork, int *ifail, int *info) {            \
  fb_uplo_t uplo_enum = FB_UPPER;                                               \
  CTYPE dummy_q = (CTYPE)0;                                                     \
  CTYPE dummy_z = (CTYPE)0;                                                     \
  CTYPE *q_ptr = q ? q : &dummy_q;                                              \
  CTYPE *z_ptr = z ? z : &dummy_z;                                              \
  int ldq_value = (q && ldq) ? *ldq : 1;                                        \
  int ldz_value = (z && ldz) ? *ldz : 1;                                        \
  int dummy_ifail = 0;                                                          \
  int *ifail_ptr = ifail ? ifail : &dummy_ifail;                                \
  (void)work;                                                                   \
  (void)rwork;                                                                  \
  (void)iwork;                                                                  \
  if (!jobz || !range || !uplo || !n || !ka || !kb || !ldab || !ldbb || !vl || \
      !vu || !il || !iu || !abstol || !m || !w ||                               \
      fb_lapack_char_to_uplo(*uplo, &uplo_enum) != 0) {                        \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (fb_lapack_job_requests_vectors(*jobz) &&                                 \
      (!q || !ldq || !z || !ldz || !ifail)) {                                  \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, char, char, fb_uplo_t, int, int, int, CTYPE *,\
                     int, CTYPE *, int, CTYPE *, int, RTYPE, RTYPE, int, int,  \
                     RTYPE, int *, RTYPE *, CTYPE *, int, int *))g_cblas_fn[OP_ID])(\
      FB_LAYOUT_COL_MAJOR, *jobz, *range, uplo_enum, *n, *ka, *kb, ab, *ldab,  \
      bb, *ldbb, q_ptr, ldq_value, *vl, *vu, *il, *iu, *abstol, m, w, z_ptr,   \
      ldz_value, ifail_ptr);                                                    \
  if (info) *info = rc;                                                         \
}

#define DEFINE_SBGVD_REAL_THUNKS(OP, TYPE, OP_ID)                               \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const char jobz, const fb_uplo_t uplo,           \
    const int n, const int ka, const int kb, TYPE *ab, const int ldab,         \
    TYPE *bb, const int ldbb, TYPE *w, TYPE *z, const int ldz) {               \
  char jobz_ = jobz;                                                            \
  char uplo_char = 'U';                                                         \
  int n_ = n;                                                                   \
  int ka_ = ka;                                                                 \
  int kb_ = kb;                                                                 \
  int ldab_ = (ka >= 0) ? (ka + 1) : 1;                                         \
  int ldbb_ = (kb >= 0) ? (kb + 1) : 1;                                         \
  int ldz_ = (n > 0) ? n : 1;                                                   \
  int vectors_requested = fb_lapack_job_requests_vectors(jobz_);                \
  int lwork_ = (n <= 1) ? 1 : (vectors_requested ? (1 + (6 * n) + (2 * n * n)) \
                                                    : (3 * n));                 \
  int liwork_ = (n <= 1) ? 1 : (vectors_requested ? (3 + (5 * n)) : 1);       \
  size_t ab_elems = (size_t)ldab_ * (size_t)((n > 0) ? n : 1);                 \
  size_t bb_elems = (size_t)ldbb_ * (size_t)((n > 0) ? n : 1);                 \
  size_t z_elems = (size_t)ldz_ * (size_t)((n > 0) ? n : 1);                   \
  TYPE *ab_col = NULL;                                                          \
  TYPE *bb_col = NULL;                                                          \
  TYPE *z_col = NULL;                                                           \
  TYPE *work = NULL;                                                            \
  int *iwork = NULL;                                                            \
  TYPE dummy_z = (TYPE)0;                                                       \
  TYPE *ab_arg = ab;                                                            \
  TYPE *bb_arg = bb;                                                            \
  TYPE *z_arg = vectors_requested ? z : &dummy_z;                               \
  int info = 0;                                                                 \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (fb_lapack_uplo_to_char(uplo, &uplo_char) != 0 || !w) return -1;          \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if ((n > 0 && (!ab || !bb)) || ldab < n || ldbb < n) return -1;            \
    if (vectors_requested && (!z || ldz < n)) return -1;                       \
    ab_col = (TYPE *)malloc(ab_elems * sizeof(TYPE));                           \
    bb_col = (TYPE *)malloc(bb_elems * sizeof(TYPE));                           \
    if (vectors_requested) {                                                    \
      z_col = (TYPE *)malloc(z_elems * sizeof(TYPE));                           \
    }                                                                           \
    if (!ab_col || !bb_col || (vectors_requested && !z_col)) {                 \
      free(ab_col);                                                             \
      free(bb_col);                                                             \
      free(z_col);                                                              \
      return -101;                                                              \
    }                                                                           \
    if (fb_copy_sym_band_to_col_major(ab, layout, uplo, n, ka, ldab,           \
                                      ab_col, ldab_, sizeof(TYPE)) != 0 ||      \
        fb_copy_sym_band_to_col_major(bb, layout, uplo, n, kb, ldbb,           \
                                      bb_col, ldbb_, sizeof(TYPE)) != 0) {      \
      free(ab_col);                                                             \
      free(bb_col);                                                             \
      free(z_col);                                                              \
      return -1;                                                                \
    }                                                                           \
    ab_arg = ab_col;                                                            \
    bb_arg = bb_col;                                                            \
    if (vectors_requested) {                                                    \
      z_arg = z_col;                                                            \
    } else {                                                                    \
      z_arg = &dummy_z;                                                         \
      ldz_ = 1;                                                                 \
    }                                                                           \
  } else if (fb_lapack_layout_is_col_major(layout)) {                           \
    ldab_ = ldab;                                                               \
    ldbb_ = ldbb;                                                               \
    if (n > 0 && (!ab || !bb)) return -1;                                       \
    if (vectors_requested) {                                                    \
      if (!z || ldz < n) return -1;                                             \
      z_arg = z;                                                                \
      ldz_ = ldz;                                                               \
    } else {                                                                    \
      z_arg = &dummy_z;                                                         \
      ldz_ = 1;                                                                 \
    }                                                                           \
  } else {                                                                      \
    return -1;                                                                  \
  }                                                                             \
  work = (TYPE *)malloc((size_t)lwork_ * sizeof(TYPE));                        \
  iwork = (int *)malloc((size_t)liwork_ * sizeof(int));                        \
  if (!work || !iwork) {                                                        \
    free(ab_col);                                                               \
    free(bb_col);                                                               \
    free(z_col);                                                                \
    free(work);                                                                 \
    free(iwork);                                                                \
    return -101;                                                                \
  }                                                                             \
  ((void (*)(char *, char *, int *, int *, int *, TYPE *, int *, TYPE *,       \
             int *, TYPE *, TYPE *, int *, TYPE *, int *, int *, int *,        \
             int *))g_fortran_fn[OP_ID])(                                      \
      &jobz_, &uplo_char, &n_, &ka_, &kb_, ab_arg, &ldab_, bb_arg, &ldbb_, w,  \
      z_arg, &ldz_, work, &lwork_, iwork, &liwork_, &info);                    \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if (fb_copy_sym_band_from_col_major(ab_col, layout, uplo, n, ka, ldab_,    \
                                        ab, ldab, sizeof(TYPE)) != 0 ||         \
        fb_copy_sym_band_from_col_major(bb_col, layout, uplo, n, kb, ldbb_,    \
                                        bb, ldbb, sizeof(TYPE)) != 0 ||         \
        (vectors_requested &&                                                   \
         fb_copy_matrix_from_col_major(z_col, n, n, ldz_, layout, z, ldz,      \
                                       sizeof(TYPE)) != 0)) {                   \
      free(ab_col);                                                             \
      free(bb_col);                                                             \
      free(z_col);                                                              \
      free(work);                                                               \
      free(iwork);                                                              \
      return -1;                                                                \
    }                                                                           \
  }                                                                             \
  free(ab_col);                                                                 \
  free(bb_col);                                                                 \
  free(z_col);                                                                  \
  free(work);                                                                   \
  free(iwork);                                                                  \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    char *jobz, char *uplo, int *n, int *ka, int *kb, TYPE *ab, int *ldab,     \
    TYPE *bb, int *ldbb, TYPE *w, TYPE *z, int *ldz, TYPE *work, int *lwork,   \
    int *iwork, int *liwork, int *info) {                                       \
  fb_uplo_t uplo_enum = FB_UPPER;                                               \
  TYPE dummy_z = (TYPE)0;                                                       \
  TYPE *z_ptr = z ? z : &dummy_z;                                               \
  int ldz_value = (z && ldz) ? *ldz : 1;                                        \
  (void)work;                                                                   \
  (void)lwork;                                                                  \
  (void)iwork;                                                                  \
  (void)liwork;                                                                 \
  if (!jobz || !uplo || !n || !ka || !kb || !ldab || !ldbb || !w ||            \
      fb_lapack_char_to_uplo(*uplo, &uplo_enum) != 0) {                        \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (fb_lapack_job_requests_vectors(*jobz) && (!z || !ldz)) {                 \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, char, fb_uplo_t, int, int, int, TYPE *,      \
                     int, TYPE *, int, TYPE *, TYPE *, int))g_cblas_fn[OP_ID])(\
      FB_LAYOUT_COL_MAJOR, *jobz, uplo_enum, *n, *ka, *kb, ab, *ldab, bb,      \
      *ldbb, w, z_ptr, ldz_value);                                              \
  if (info) *info = rc;                                                         \
}

#define DEFINE_SBGVX_REAL_THUNKS(OP, TYPE, OP_ID)                               \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const char jobz, const char range,               \
    const fb_uplo_t uplo, const int n, const int ka, const int kb, TYPE *ab,   \
    const int ldab, TYPE *bb, const int ldbb, TYPE *q, const int ldq,          \
    const TYPE vl, const TYPE vu, const int il, const int iu, const TYPE abstol,\
    int *m, TYPE *w, TYPE *z, const int ldz, int *ifail) {                     \
  char jobz_ = jobz, range_ = range;                                            \
  char uplo_char = 'U';                                                         \
  int n_ = n, ka_ = ka, kb_ = kb, il_ = il, iu_ = iu;                           \
  int ldab_ = (ka >= 0) ? (ka + 1) : 1;                                         \
  int ldbb_ = (kb >= 0) ? (kb + 1) : 1;                                         \
  int ldq_ = (n > 0) ? n : 1;                                                   \
  int ldz_ = (n > 0) ? n : 1;                                                   \
  int vectors_requested = fb_lapack_job_requests_vectors(jobz_);                \
  int work_size = (n > 0) ? (7 * n) : 1;                                        \
  int iwork_size = (n > 0) ? (5 * n) : 1;                                       \
  TYPE vl_ = vl, vu_ = vu, abstol_ = abstol;                                    \
  size_t ab_elems = (size_t)ldab_ * (size_t)((n > 0) ? n : 1);                 \
  size_t bb_elems = (size_t)ldbb_ * (size_t)((n > 0) ? n : 1);                 \
  size_t q_elems = (size_t)ldq_ * (size_t)((n > 0) ? n : 1);                   \
  size_t z_elems = (size_t)ldz_ * (size_t)((n > 0) ? n : 1);                   \
  TYPE *ab_col = NULL;                                                          \
  TYPE *bb_col = NULL;                                                          \
  TYPE *q_col = NULL;                                                           \
  TYPE *z_col = NULL;                                                           \
  TYPE *work = NULL;                                                            \
  int *iwork = NULL;                                                            \
  TYPE dummy_q = (TYPE)0;                                                       \
  TYPE dummy_z = (TYPE)0;                                                       \
  TYPE *ab_arg = ab;                                                            \
  TYPE *bb_arg = bb;                                                            \
  TYPE *q_arg = q ? q : &dummy_q;                                               \
  TYPE *z_arg = z ? z : &dummy_z;                                               \
  int dummy_ifail = 0;                                                          \
  int *ifail_ptr = ifail ? ifail : &dummy_ifail;                                \
  int info = 0;                                                                 \
  if (!m || !w || !g_fortran_fn[OP_ID]) return -1;                              \
  if (fb_lapack_uplo_to_char(uplo, &uplo_char) != 0) return -1;                \
  if (vectors_requested && (!q || !z || !ifail)) return -1;                    \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if ((n > 0 && (!ab || !bb)) || ldab < n || ldbb < n) return -1;            \
    if (vectors_requested && (ldq < n || ldz < n)) return -1;                  \
    ab_col = (TYPE *)malloc(ab_elems * sizeof(TYPE));                           \
    bb_col = (TYPE *)malloc(bb_elems * sizeof(TYPE));                           \
    if (vectors_requested) {                                                    \
      q_col = (TYPE *)malloc(q_elems * sizeof(TYPE));                           \
      z_col = (TYPE *)malloc(z_elems * sizeof(TYPE));                           \
    }                                                                           \
    if (!ab_col || !bb_col || (vectors_requested && (!q_col || !z_col))) {     \
      free(ab_col);                                                             \
      free(bb_col);                                                             \
      free(q_col);                                                              \
      free(z_col);                                                              \
      return -101;                                                              \
    }                                                                           \
    if (fb_copy_sym_band_to_col_major(ab, layout, uplo, n, ka, ldab,           \
                                      ab_col, ldab_, sizeof(TYPE)) != 0 ||      \
        fb_copy_sym_band_to_col_major(bb, layout, uplo, n, kb, ldbb,           \
                                      bb_col, ldbb_, sizeof(TYPE)) != 0) {      \
      free(ab_col);                                                             \
      free(bb_col);                                                             \
      free(q_col);                                                              \
      free(z_col);                                                              \
      return -1;                                                                \
    }                                                                           \
    ab_arg = ab_col;                                                            \
    bb_arg = bb_col;                                                            \
    if (vectors_requested) {                                                    \
      q_arg = q_col;                                                            \
      z_arg = z_col;                                                            \
    } else {                                                                    \
      q_arg = &dummy_q;                                                         \
      z_arg = &dummy_z;                                                         \
      ldq_ = 1;                                                                 \
      ldz_ = 1;                                                                 \
    }                                                                           \
  } else if (fb_lapack_layout_is_col_major(layout)) {                           \
    ldab_ = ldab;                                                               \
    ldbb_ = ldbb;                                                               \
    if (n > 0 && (!ab || !bb)) return -1;                                       \
    if (vectors_requested) {                                                    \
      if (ldq < n || ldz < n) return -1;                                        \
      ldq_ = ldq;                                                               \
      ldz_ = ldz;                                                               \
      q_arg = q;                                                                \
      z_arg = z;                                                                \
    } else {                                                                    \
      q_arg = &dummy_q;                                                         \
      z_arg = &dummy_z;                                                         \
      ldq_ = 1;                                                                 \
      ldz_ = 1;                                                                 \
    }                                                                           \
  } else {                                                                      \
    return -1;                                                                  \
  }                                                                             \
  work = (TYPE *)malloc((size_t)work_size * sizeof(TYPE));                     \
  iwork = (int *)malloc((size_t)iwork_size * sizeof(int));                     \
  if (!work || !iwork) {                                                        \
    free(ab_col);                                                               \
    free(bb_col);                                                               \
    free(q_col);                                                                \
    free(z_col);                                                                \
    free(work);                                                                 \
    free(iwork);                                                                \
    return -101;                                                                \
  }                                                                             \
  ((void (*)(char *, char *, char *, int *, int *, int *, TYPE *, int *,       \
             TYPE *, int *, TYPE *, int *, TYPE *, TYPE *, int *, int *,       \
             TYPE *, int *, TYPE *, TYPE *, int *, TYPE *, int *, int *,       \
          int *))g_fortran_fn[OP_ID])(                                       \
      &jobz_, &range_, &uplo_char, &n_, &ka_, &kb_, ab_arg, &ldab_, bb_arg,    \
      &ldbb_, q_arg, &ldq_, &vl_, &vu_, &il_, &iu_, &abstol_, m, w, z_arg,     \
      &ldz_, work, iwork, ifail_ptr, &info);                                    \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if (fb_copy_sym_band_from_col_major(ab_col, layout, uplo, n, ka, ldab_,    \
                                        ab, ldab, sizeof(TYPE)) != 0 ||         \
        fb_copy_sym_band_from_col_major(bb_col, layout, uplo, n, kb, ldbb_,    \
                                        bb, ldbb, sizeof(TYPE)) != 0 ||         \
        (vectors_requested &&                                                   \
         fb_copy_matrix_from_col_major(q_col, n, n, ldq_, layout, q, ldq,      \
                                       sizeof(TYPE)) != 0) ||                   \
        (vectors_requested &&                                                   \
         fb_copy_matrix_from_col_major(z_col, n, n, ldz_, layout, z, ldz,      \
                                       sizeof(TYPE)) != 0)) {                   \
      free(ab_col);                                                             \
      free(bb_col);                                                             \
      free(q_col);                                                              \
      free(z_col);                                                              \
      free(work);                                                               \
      free(iwork);                                                              \
      return -1;                                                                \
    }                                                                           \
  }                                                                             \
  free(ab_col);                                                                 \
  free(bb_col);                                                                 \
  free(q_col);                                                                  \
  free(z_col);                                                                  \
  free(work);                                                                   \
  free(iwork);                                                                  \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    char *jobz, char *range, char *uplo, int *n, int *ka, int *kb, TYPE *ab,   \
    int *ldab, TYPE *bb, int *ldbb, TYPE *q, int *ldq, TYPE *vl, TYPE *vu,     \
    int *il, int *iu, TYPE *abstol, int *m, TYPE *w, TYPE *z, int *ldz,        \
    TYPE *work, int *iwork, int *ifail, int *info) {                            \
  fb_uplo_t uplo_enum = FB_UPPER;                                               \
  TYPE dummy_q = (TYPE)0;                                                       \
  TYPE dummy_z = (TYPE)0;                                                       \
  TYPE *q_ptr = q ? q : &dummy_q;                                               \
  TYPE *z_ptr = z ? z : &dummy_z;                                               \
  int ldq_value = (q && ldq) ? *ldq : 1;                                        \
  int ldz_value = (z && ldz) ? *ldz : 1;                                        \
  int dummy_ifail = 0;                                                          \
  int *ifail_ptr = ifail ? ifail : &dummy_ifail;                                \
  (void)work;                                                                   \
  (void)iwork;                                                                  \
  if (!jobz || !range || !uplo || !n || !ka || !kb || !ldab || !ldbb || !ldq ||\
      !vl || !vu || !il || !iu || !abstol || !m || !w ||                       \
      fb_lapack_char_to_uplo(*uplo, &uplo_enum) != 0) {                        \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (fb_lapack_job_requests_vectors(*jobz) &&                                 \
      (!q || !z || !ldz || !ifail)) {                                           \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, char, char, fb_uplo_t, int, int, int, TYPE *,\
                     int, TYPE *, int, TYPE *, int, TYPE, TYPE, int, int,      \
                     TYPE, int *, TYPE *, TYPE *, int, int *))g_cblas_fn[OP_ID])(\
      FB_LAYOUT_COL_MAJOR, *jobz, *range, uplo_enum, *n, *ka, *kb, ab, *ldab,  \
      bb, *ldbb, q_ptr, ldq_value, *vl, *vu, *il, *iu, *abstol, m, w, z_ptr,   \
      ldz_value, ifail_ptr);                                                    \
  if (info) *info = rc;                                                         \
}

#define DEFINE_SPGV_REAL_THUNKS(OP, TYPE, OP_ID)                                \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const int itype, const char jobz,                \
    const fb_uplo_t uplo, const int n, TYPE *ap, TYPE *bp, TYPE *w, TYPE *z,   \
    const int ldz) {                                                            \
  char jobz_ = jobz;                                                            \
  char uplo_char = 'U';                                                         \
  int itype_ = itype;                                                           \
  int n_ = n;                                                                   \
  int ldz_ = (n > 0) ? n : 1;                                                   \
  int vectors_requested = fb_lapack_job_requests_vectors(jobz_);                \
  int work_len = (n > 0) ? (3 * n) : 1;                                         \
  size_t packed_len = (size_t)((n > 0) ? n : 1) *                               \
                      (size_t)(((n + 1) > 0) ? (n + 1) : 2) / 2U;              \
  size_t z_elems = (size_t)ldz_ * (size_t)((n > 0) ? n : 1);                   \
  TYPE *ap_col = NULL;                                                          \
  TYPE *bp_col = NULL;                                                          \
  TYPE *z_col = NULL;                                                           \
  TYPE *work = NULL;                                                            \
  TYPE dummy_z = (TYPE)0;                                                       \
  TYPE *ap_arg = ap;                                                            \
  TYPE *bp_arg = bp;                                                            \
  TYPE *z_arg = vectors_requested ? z : &dummy_z;                               \
  int info = 0;                                                                 \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (fb_lapack_uplo_to_char(uplo, &uplo_char) != 0 || !w) return -1;          \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if ((n > 0 && (!ap || !bp))) return -1;                                     \
    if (vectors_requested && (!z || ldz < n)) return -1;                       \
    ap_col = (TYPE *)malloc(packed_len * sizeof(TYPE));                         \
    bp_col = (TYPE *)malloc(packed_len * sizeof(TYPE));                         \
    if (!ap_col || !bp_col) {                                                   \
      free(ap_col);                                                             \
      free(bp_col);                                                             \
      return -101;                                                              \
    }                                                                           \
    if (fb_copy_triangular_packed_to_col_major(ap, layout, uplo_char, 'N', n,  \
                                               ap_col, sizeof(TYPE)) != 0 ||    \
        fb_copy_triangular_packed_to_col_major(bp, layout, uplo_char, 'N', n,  \
                                               bp_col, sizeof(TYPE)) != 0) {    \
      free(ap_col);                                                             \
      free(bp_col);                                                             \
      return -1;                                                                \
    }                                                                           \
    ap_arg = ap_col;                                                            \
    bp_arg = bp_col;                                                            \
    if (vectors_requested) {                                                    \
      z_col = (TYPE *)malloc(z_elems * sizeof(TYPE));                           \
      if (!z_col) {                                                             \
        free(ap_col);                                                           \
        free(bp_col);                                                           \
        return -101;                                                            \
      }                                                                         \
      z_arg = z_col;                                                            \
    } else {                                                                    \
      z_arg = &dummy_z;                                                         \
      ldz_ = 1;                                                                 \
    }                                                                           \
  } else if (fb_lapack_layout_is_col_major(layout)) {                           \
    if (n > 0 && (!ap || !bp)) return -1;                                       \
    if (vectors_requested) {                                                    \
      if (!z || ldz < n) return -1;                                             \
      z_arg = z;                                                                \
      ldz_ = ldz;                                                               \
    } else {                                                                    \
      z_arg = &dummy_z;                                                         \
      ldz_ = 1;                                                                 \
    }                                                                           \
  } else {                                                                      \
    return -1;                                                                  \
  }                                                                             \
  work = (TYPE *)malloc((size_t)work_len * sizeof(TYPE));                      \
  if (!work) {                                                                  \
    free(ap_col);                                                               \
    free(bp_col);                                                               \
    free(z_col);                                                                \
    return -101;                                                                \
  }                                                                             \
  ((void (*)(int *, char *, char *, int *, TYPE *, TYPE *, TYPE *, TYPE *,     \
             int *, TYPE *, int *))g_fortran_fn[OP_ID])(                       \
      &itype_, &jobz_, &uplo_char, &n_, ap_arg, bp_arg, w, z_arg, &ldz_, work, \
      &info);                                                                   \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if (fb_copy_triangular_packed_from_col_major(ap_col, layout, uplo_char,    \
                                                 'N', n, ap, sizeof(TYPE)) != 0 || \
        fb_copy_triangular_packed_from_col_major(bp_col, layout, uplo_char,    \
                                                 'N', n, bp, sizeof(TYPE)) != 0 || \
        (vectors_requested &&                                                   \
         fb_copy_matrix_from_col_major(z_col, n, n, ldz_, layout, z, ldz,      \
                                       sizeof(TYPE)) != 0)) {                   \
      free(ap_col);                                                             \
      free(bp_col);                                                             \
      free(z_col);                                                              \
      free(work);                                                               \
      return -1;                                                                \
    }                                                                           \
  }                                                                             \
  free(ap_col);                                                                 \
  free(bp_col);                                                                 \
  free(z_col);                                                                  \
  free(work);                                                                   \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    int *itype, char *jobz, char *uplo, int *n, TYPE *ap, TYPE *bp, TYPE *w,   \
    TYPE *z, int *ldz, TYPE *work, int *info) {                                 \
  fb_uplo_t uplo_enum = FB_UPPER;                                               \
  TYPE dummy_z = (TYPE)0;                                                       \
  TYPE *z_ptr = z ? z : &dummy_z;                                               \
  int ldz_value = (z && ldz) ? *ldz : 1;                                        \
  (void)work;                                                                   \
  if (!itype || !jobz || !uplo || !n || !w ||                                   \
      fb_lapack_char_to_uplo(*uplo, &uplo_enum) != 0) {                        \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (fb_lapack_job_requests_vectors(*jobz) && (!z || !ldz)) {                 \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, int, char, fb_uplo_t, int, TYPE *, TYPE *,   \
                     TYPE *, TYPE *, int))g_cblas_fn[OP_ID])(                   \
      FB_LAYOUT_COL_MAJOR, *itype, *jobz, uplo_enum, *n, ap, bp, w, z_ptr,     \
      ldz_value);                                                               \
  if (info) *info = rc;                                                         \
}

#define DEFINE_SPGVD_REAL_THUNKS(OP, TYPE, OP_ID, QUERY_FN)                     \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const int itype, const char jobz,                \
    const fb_uplo_t uplo, const int n, TYPE *ap, TYPE *bp, TYPE *w, TYPE *z,   \
    const int ldz) {                                                            \
  char jobz_ = jobz;                                                            \
  char uplo_char = 'U';                                                         \
  int itype_ = itype;                                                           \
  int n_ = n;                                                                   \
  int ldz_ = (n > 0) ? n : 1;                                                   \
  int vectors_requested = fb_lapack_job_requests_vectors(jobz_);                \
  int lwork_ = -1;                                                              \
  int liwork_ = -1;                                                             \
  TYPE work_query = (TYPE)0;                                                    \
  int iwork_query = 0;                                                          \
  size_t packed_len = (size_t)((n > 0) ? n : 1) *                               \
                      (size_t)(((n + 1) > 0) ? (n + 1) : 2) / 2U;              \
  size_t z_elems = (size_t)ldz_ * (size_t)((n > 0) ? n : 1);                   \
  TYPE *ap_col = NULL;                                                          \
  TYPE *bp_col = NULL;                                                          \
  TYPE *z_col = NULL;                                                           \
  TYPE *work = NULL;                                                            \
  int *iwork = NULL;                                                            \
  TYPE dummy_z = (TYPE)0;                                                       \
  TYPE *ap_arg = ap;                                                            \
  TYPE *bp_arg = bp;                                                            \
  TYPE *z_arg = vectors_requested ? z : &dummy_z;                               \
  int info = 0;                                                                 \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (fb_lapack_uplo_to_char(uplo, &uplo_char) != 0 || !w) return -1;          \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if ((n > 0 && (!ap || !bp))) return -1;                                     \
    if (vectors_requested && (!z || ldz < n)) return -1;                       \
    ap_col = (TYPE *)malloc(packed_len * sizeof(TYPE));                         \
    bp_col = (TYPE *)malloc(packed_len * sizeof(TYPE));                         \
    if (!ap_col || !bp_col) {                                                   \
      free(ap_col);                                                             \
      free(bp_col);                                                             \
      return -101;                                                              \
    }                                                                           \
    if (fb_copy_triangular_packed_to_col_major(ap, layout, uplo_char, 'N', n,  \
                                               ap_col, sizeof(TYPE)) != 0 ||    \
        fb_copy_triangular_packed_to_col_major(bp, layout, uplo_char, 'N', n,  \
                                               bp_col, sizeof(TYPE)) != 0) {    \
      free(ap_col);                                                             \
      free(bp_col);                                                             \
      return -1;                                                                \
    }                                                                           \
    ap_arg = ap_col;                                                            \
    bp_arg = bp_col;                                                            \
    if (vectors_requested) {                                                    \
      z_col = (TYPE *)malloc(z_elems * sizeof(TYPE));                           \
      if (!z_col) {                                                             \
        free(ap_col);                                                           \
        free(bp_col);                                                           \
        return -101;                                                            \
      }                                                                         \
      z_arg = z_col;                                                            \
    } else {                                                                    \
      z_arg = &dummy_z;                                                         \
      ldz_ = 1;                                                                 \
    }                                                                           \
  } else if (fb_lapack_layout_is_col_major(layout)) {                           \
    if (n > 0 && (!ap || !bp)) return -1;                                       \
    if (vectors_requested) {                                                    \
      if (!z || ldz < n) return -1;                                             \
      z_arg = z;                                                                \
      ldz_ = ldz;                                                               \
    } else {                                                                    \
      z_arg = &dummy_z;                                                         \
      ldz_ = 1;                                                                 \
    }                                                                           \
  } else {                                                                      \
    return -1;                                                                  \
  }                                                                             \
  ((void (*)(int *, char *, char *, int *, TYPE *, TYPE *, TYPE *, TYPE *,     \
          int *, TYPE *, int *, int *, int *, int *))g_fortran_fn[OP_ID])(  \
      &itype_, &jobz_, &uplo_char, &n_, ap_arg, bp_arg, w, z_arg, &ldz_,       \
      &work_query, &lwork_, &iwork_query, &liwork_, &info);                    \
  if (info != 0) {                                                              \
    free(ap_col);                                                               \
    free(bp_col);                                                               \
    free(z_col);                                                                \
    return info;                                                                \
  }                                                                             \
  lwork_ = QUERY_FN(work_query);                                                \
  liwork_ = fb_query_size_from_int(iwork_query);                                \
  work = (TYPE *)malloc((size_t)lwork_ * sizeof(TYPE));                         \
  iwork = (int *)malloc((size_t)liwork_ * sizeof(int));                         \
  if (!work || !iwork) {                                                        \
    free(ap_col);                                                               \
    free(bp_col);                                                               \
    free(z_col);                                                                \
    free(work);                                                                 \
    free(iwork);                                                                \
    return -101;                                                                \
  }                                                                             \
  ((void (*)(int *, char *, char *, int *, TYPE *, TYPE *, TYPE *, TYPE *,     \
          int *, TYPE *, int *, int *, int *, int *))g_fortran_fn[OP_ID])(  \
      &itype_, &jobz_, &uplo_char, &n_, ap_arg, bp_arg, w, z_arg, &ldz_,       \
      work, &lwork_, iwork, &liwork_, &info);                                  \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if (fb_copy_triangular_packed_from_col_major(ap_col, layout, uplo_char,    \
                                                 'N', n, ap, sizeof(TYPE)) != 0 || \
        fb_copy_triangular_packed_from_col_major(bp_col, layout, uplo_char,    \
                                                 'N', n, bp, sizeof(TYPE)) != 0 || \
        (vectors_requested &&                                                   \
         fb_copy_matrix_from_col_major(z_col, n, n, ldz_, layout, z, ldz,      \
                                       sizeof(TYPE)) != 0)) {                   \
      free(ap_col);                                                             \
      free(bp_col);                                                             \
      free(z_col);                                                              \
      free(work);                                                               \
      free(iwork);                                                              \
      return -1;                                                                \
    }                                                                           \
  }                                                                             \
  free(ap_col);                                                                 \
  free(bp_col);                                                                 \
  free(z_col);                                                                  \
  free(work);                                                                   \
  free(iwork);                                                                  \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    int *itype, char *jobz, char *uplo, int *n, TYPE *ap, TYPE *bp, TYPE *w,   \
    TYPE *z, int *ldz, TYPE *work, int *lwork, int *iwork, int *liwork,        \
    int *info) {                                                                \
  fb_uplo_t uplo_enum = FB_UPPER;                                               \
  TYPE dummy_z = (TYPE)0;                                                       \
  TYPE *z_ptr = z ? z : &dummy_z;                                               \
  int ldz_value = (z && ldz) ? *ldz : 1;                                        \
  (void)work;                                                                   \
  (void)lwork;                                                                  \
  (void)iwork;                                                                  \
  (void)liwork;                                                                 \
  if (!itype || !jobz || !uplo || !n || !w ||                                   \
      fb_lapack_char_to_uplo(*uplo, &uplo_enum) != 0) {                        \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (fb_lapack_job_requests_vectors(*jobz) && (!z || !ldz)) {                 \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, int, char, fb_uplo_t, int, TYPE *, TYPE *,   \
                     TYPE *, TYPE *, int))g_cblas_fn[OP_ID])(                   \
      FB_LAYOUT_COL_MAJOR, *itype, *jobz, uplo_enum, *n, ap, bp, w, z_ptr,     \
      ldz_value);                                                               \
  if (info) *info = rc;                                                         \
}
