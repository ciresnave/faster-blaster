
#define DEFINE_LAPACK_LACPY_THUNKS(OP, TYPE, OP_ID)                            \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, char uplo, int m, int n, const TYPE *a,          \
    int lda, TYPE *b, int ldb) {                                                \
  int m_ = m;                                                                  \
  int n_ = n;                                                                  \
  int lda_ = lda;                                                              \
  int ldb_ = ldb;                                                              \
  TYPE *a_col = NULL;                                                          \
  TYPE *b_col = NULL;                                                          \
  size_t rows = (size_t)((m > 0) ? m : 1);                                     \
  size_t cols = (size_t)((n > 0) ? n : 1);                                     \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if ((m > 0 && n > 0 && !a) || (m > 0 && n > 0 && !b)) return -1;            \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                         \
    if (lda < n || ldb < n) return -1;                                         \
    a_col = (TYPE *)malloc(rows * cols * sizeof(TYPE));                        \
    if (!a_col) return -101;                                                   \
    b_col = (TYPE *)malloc(rows * cols * sizeof(TYPE));                        \
    if (!b_col) {                                                              \
      free(a_col);                                                             \
      return -101;                                                             \
    }                                                                          \
    if (m > 0 && n > 0 &&                                                       \
        fb_copy_matrix_to_col_major(a, layout, m, n, lda,                      \
                                    a_col, m, sizeof(TYPE)) != 0) {            \
      free(b_col);                                                             \
      free(a_col);                                                             \
      return -1;                                                               \
    }                                                                          \
    ((void (*)(const char *, const int *, const int *, const TYPE *,           \
               const int *, TYPE *, const int *))g_fortran_fn[OP_ID])(         \
        &uplo, &m_, &n_, a_col, &m_, b_col, &m_);                              \
    if (m > 0 && n > 0 &&                                                       \
        fb_copy_matrix_from_col_major(b_col, m, n, m, layout,                  \
                                      b, ldb, sizeof(TYPE)) != 0) {            \
      free(b_col);                                                             \
      free(a_col);                                                             \
      return -1;                                                               \
    }                                                                          \
    free(b_col);                                                               \
    free(a_col);                                                               \
    return 0;                                                                  \
  }                                                                            \
  if (!fb_lapack_layout_is_col_major(layout)) return -1;                       \
  ((void (*)(const char *, const int *, const int *, const TYPE *,             \
             const int *, TYPE *, const int *))g_fortran_fn[OP_ID])(           \
      &uplo, &m_, &n_, a, &lda_, b, &ldb_);                                    \
  return 0;                                                                    \
}                                                                              \
static void thunk_##OP##_cblas_to_fortran(                                     \
    char *uplo, int *m, int *n, const TYPE *a, int *lda, TYPE *b, int *ldb) { \
  if (!g_cblas_fn[OP_ID]) return;                                              \
  if (!uplo || !m || !n || !lda || !ldb) return;                               \
  if ((*m > 0 && *n > 0 && !a) || (*m > 0 && *n > 0 && !b)) return;           \
  (void)((int (*)(fb_layout_t, char, int, int, const TYPE *, int, TYPE *, int))\
             g_cblas_fn[OP_ID])(FB_LAYOUT_COL_MAJOR, *uplo, *m, *n, a, *lda,   \
                                b, *ldb);                                      \
}

#define DEFINE_LAPACK_LASRT_THUNKS(OP, TYPE, OP_ID)                            \
static int thunk_##OP##_fortran_to_cblas(char id, int n, TYPE *d) {            \
  int info = 0;                                                                \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  ((void (*)(const char *, const int *, TYPE *, int *))g_fortran_fn[OP_ID])(   \
      &id, &n, d, &info);                                                      \
  return info;                                                                 \
}                                                                              \
static void thunk_##OP##_cblas_to_fortran(char *id, int *n, TYPE *d,           \
                                          int *info) {                         \
  if (!g_cblas_fn[OP_ID]) {                                                    \
    if (info) *info = -1;                                                      \
    return;                                                                    \
  }                                                                            \
  if (!id || !n || !info) return;                                              \
  *info = ((int (*)(char, int, TYPE *))g_cblas_fn[OP_ID])(*id, *n, d);         \
}

#define DEFINE_LAPACK_LASWP_THUNKS(OP, TYPE, OP_ID)                            \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const int n, TYPE *A, const int lda,             \
    const int k1, const int k2, const int *ipiv, const int incx) {             \
  int n_ = n;                                                                  \
  int lda_ = lda;                                                              \
  int k1_ = k1;                                                                \
  int k2_ = k2;                                                                \
  int incx_ = incx;                                                            \
  int a_rows = (k2 > 0) ? k2 : 1;                                              \
  TYPE *a_col = NULL;                                                          \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (!ipiv) return -1;                                                        \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                         \
    int i = 0;                                                                 \
    if (lda < n) return -1;                                                    \
    if (k2_ >= k1_) {                                                          \
      int abs_incx = (incx_ >= 0) ? incx_ : -incx_;                            \
      for (i = k1_; i <= k2_; ++i) {                                           \
        int pivot_index = k1_ + ((i - k1_) * abs_incx) - 1;                    \
        if (pivot_index >= 0 && ipiv[pivot_index] > a_rows) {                  \
          a_rows = ipiv[pivot_index];                                           \
        }                                                                      \
      }                                                                        \
    }                                                                          \
    if (a_rows < 1) a_rows = 1;                                                \
    if (n > 0 && !A) return -1;                                                \
    a_col = (TYPE *)malloc((size_t)a_rows * (size_t)((n > 0) ? n : 1) *       \
                           sizeof(TYPE));                                      \
    if (!a_col) return -101;                                                   \
    if (n > 0 &&                                                                \
        fb_copy_matrix_to_col_major(A, layout, a_rows, n, lda,                 \
                                    a_col, a_rows, sizeof(TYPE)) != 0) {       \
      free(a_col);                                                             \
      return -1;                                                               \
    }                                                                          \
    ((void (*)(const int *, TYPE *, const int *, const int *, const int *,     \
               const int *, const int *))g_fortran_fn[OP_ID])(                 \
        &n_, a_col, &a_rows, &k1_, &k2_, ipiv, &incx_);                        \
    if (n > 0 &&                                                                \
        fb_copy_matrix_from_col_major(a_col, a_rows, n, a_rows, layout,        \
                                      A, lda, sizeof(TYPE)) != 0) {            \
      free(a_col);                                                             \
      return -1;                                                               \
    }                                                                          \
    free(a_col);                                                               \
    return 0;                                                                  \
  }                                                                            \
  if (!fb_lapack_layout_is_col_major(layout)) {                                \
    return -1;                                                                 \
  }                                                                            \
  if (n > 0 && !A) return -1;                                                  \
  ((void (*)(const int *, TYPE *, const int *, const int *, const int *,       \
             const int *, const int *))g_fortran_fn[OP_ID])(                   \
      &n_, A, &lda_, &k1_, &k2_, ipiv, &incx_);                                \
  return 0;                                                                    \
}                                                                              \
static void thunk_##OP##_cblas_to_fortran(                                     \
    int *n, TYPE *A, int *lda, int *k1, int *k2, int *ipiv, int *incx) {       \
  if (!g_cblas_fn[OP_ID]) return;                                              \
  if (!n || !lda || !k1 || !k2 || !ipiv || !incx || (*n > 0 && !A)) return;    \
  (void)((int (*)(fb_layout_t, int, TYPE *, int, int, int, const int *, int))  \
             g_cblas_fn[OP_ID])(FB_LAYOUT_COL_MAJOR, *n, A, *lda, *k1, *k2,    \
                                ipiv, *incx);                                  \
}

#define DEFINE_LAPACK_LATBS_REAL_THUNKS(OP, TYPE, OP_ID)                       \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, char uplo, char trans, char diag, char normin,   \
    const int n, const int kd, TYPE *ab, const int ldab, TYPE *x,              \
    TYPE *scale, TYPE *cnorm) {                                                \
  fb_uplo_t uplo_ = FB_UPPER;                                                  \
  int n_ = n, kd_ = kd, ldab_ = (kd >= 0) ? (kd + 1) : 1, info = 0;           \
  size_t ab_elems = (size_t)ldab_ * (size_t)((n > 0) ? n : 1);                \
  TYPE *ab_arg = ab;                                                           \
  TYPE *ab_col = NULL;                                                         \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (fb_lapack_char_to_uplo(uplo, &uplo_) != 0 ||                             \
      n < 0 || kd < 0 || !x || !scale || !cnorm) return -1;                   \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                         \
    if ((n > 0 && !ab) || ldab < n) return -1;                                \
    ab_col = (TYPE *)malloc(ab_elems * sizeof(TYPE));                          \
    if (!ab_col) return -101;                                                  \
    if (fb_copy_sym_band_to_col_major(ab, layout, uplo_, n, kd, ldab,         \
                                      ab_col, ldab_, sizeof(TYPE)) != 0) {    \
      free(ab_col);                                                            \
      return -1;                                                               \
    }                                                                          \
    ab_arg = ab_col;                                                           \
  } else if (!fb_lapack_layout_is_col_major(layout)) {                         \
    return -1;                                                                 \
  } else if ((n > 0 && !ab) || ldab < (kd + 1)) {                             \
    return -1;                                                                 \
  } else {                                                                     \
    ldab_ = ldab;                                                              \
  }                                                                            \
  ((void (*)(char *, char *, char *, char *, int *, int *, TYPE *, int *,     \
             TYPE *, TYPE *, TYPE *, int *))g_fortran_fn[OP_ID])(              \
      &uplo, &trans, &diag, &normin, &n_, &kd_, ab_arg, &ldab_, x, scale,      \
      cnorm, &info);                                                           \
  free(ab_col);                                                                \
  return info;                                                                 \
}                                                                              \
static void thunk_##OP##_cblas_to_fortran(                                     \
    char *uplo, char *trans, char *diag, char *normin, int *n, int *kd,        \
    TYPE *ab, int *ldab, TYPE *x, TYPE *scale, TYPE *cnorm, int *info) {       \
  if (!g_cblas_fn[OP_ID]) {                                                    \
    if (info) *info = -1;                                                      \
    return;                                                                    \
  }                                                                            \
  if (!uplo || !trans || !diag || !normin || !n || !kd || !ldab || !x ||      \
      !scale || !cnorm || !info) {                                             \
    if (info) *info = -1;                                                      \
    return;                                                                    \
  }                                                                            \
  *info = ((int (*)(fb_layout_t, char, char, char, char, int, int, TYPE *,    \
                    int, TYPE *, TYPE *, TYPE *))g_cblas_fn[OP_ID])(           \
      FB_LAYOUT_COL_MAJOR, *uplo, *trans, *diag, *normin, *n, *kd, ab, *ldab, \
      x, scale, cnorm);                                                        \
}

#define DEFINE_LAPACK_LATBS_COMPLEX_THUNKS(OP, CTYPE, RTYPE, OP_ID)           \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, char uplo, char trans, char diag, char normin,   \
    const int n, const int kd, CTYPE *ab, const int ldab, CTYPE *x,            \
    RTYPE *scale, RTYPE *cnorm) {                                              \
  fb_uplo_t uplo_ = FB_UPPER;                                                  \
  int n_ = n, kd_ = kd, ldab_ = (kd >= 0) ? (kd + 1) : 1, info = 0;           \
  size_t ab_elems = (size_t)ldab_ * (size_t)((n > 0) ? n : 1);                \
  CTYPE *ab_arg = ab;                                                          \
  CTYPE *ab_col = NULL;                                                        \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (fb_lapack_char_to_uplo(uplo, &uplo_) != 0 ||                             \
      n < 0 || kd < 0 || !x || !scale || !cnorm) return -1;                   \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                         \
    if ((n > 0 && !ab) || ldab < n) return -1;                                \
    ab_col = (CTYPE *)malloc(ab_elems * sizeof(CTYPE));                        \
    if (!ab_col) return -101;                                                  \
    if (fb_copy_sym_band_to_col_major(ab, layout, uplo_, n, kd, ldab,         \
                                      ab_col, ldab_, sizeof(CTYPE)) != 0) {   \
      free(ab_col);                                                            \
      return -1;                                                               \
    }                                                                          \
    ab_arg = ab_col;                                                           \
  } else if (!fb_lapack_layout_is_col_major(layout)) {                         \
    return -1;                                                                 \
  } else if ((n > 0 && !ab) || ldab < (kd + 1)) {                             \
    return -1;                                                                 \
  } else {                                                                     \
    ldab_ = ldab;                                                              \
  }                                                                            \
  ((void (*)(char *, char *, char *, char *, int *, int *, CTYPE *, int *,    \
             CTYPE *, RTYPE *, RTYPE *, int *))g_fortran_fn[OP_ID])(           \
      &uplo, &trans, &diag, &normin, &n_, &kd_, ab_arg, &ldab_, x, scale,      \
      cnorm, &info);                                                           \
  free(ab_col);                                                                \
  return info;                                                                 \
}                                                                              \
static void thunk_##OP##_cblas_to_fortran(                                     \
    char *uplo, char *trans, char *diag, char *normin, int *n, int *kd,        \
    CTYPE *ab, int *ldab, CTYPE *x, RTYPE *scale, RTYPE *cnorm, int *info) {   \
  if (!g_cblas_fn[OP_ID]) {                                                    \
    if (info) *info = -1;                                                      \
    return;                                                                    \
  }                                                                            \
  if (!uplo || !trans || !diag || !normin || !n || !kd || !ldab || !x ||      \
      !scale || !cnorm || !info) {                                             \
    if (info) *info = -1;                                                      \
    return;                                                                    \
  }                                                                            \
  *info = ((int (*)(fb_layout_t, char, char, char, char, int, int, CTYPE *,   \
                    int, CTYPE *, RTYPE *, RTYPE *))g_cblas_fn[OP_ID])(        \
      FB_LAYOUT_COL_MAJOR, *uplo, *trans, *diag, *normin, *n, *kd, ab, *ldab, \
      x, scale, cnorm);                                                        \
}

#define DEFINE_GECON_REAL_FORTRAN_TO_CBLAS(OP, TYPE, OP_ID)                    \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const char norm, const int n, const TYPE *A,     \
    const int lda, const TYPE anorm, TYPE *rcond) {                            \
  if (!fb_lapack_layout_is_col_major(layout)) return -1;                       \
  char norm_ = norm;                                                           \
  int n_ = n, lda_ = lda, info = 0;                                            \
  int work_size = (n > 0) ? (4 * n) : 1;                                       \
  int iwork_size = (n > 0) ? n : 1;                                            \
  TYPE anorm_ = anorm;                                                         \
  TYPE *work = NULL;                                                           \
  int *iwork = NULL;                                                           \
  if (!g_fortran_fn[OP_ID]) return -1;                                         \
  work = (TYPE *)malloc((size_t)work_size * sizeof(TYPE));                     \
  iwork = (int *)malloc((size_t)iwork_size * sizeof(int));                     \
  if (!work || !iwork) {                                                       \
    free(work);                                                                \
    free(iwork);                                                               \
    return -101;                                                               \
  }                                                                            \
  ((void (*)(char *, int *, const TYPE *, int *, TYPE *, TYPE *, TYPE *,       \
             int *, int *))g_fortran_fn[OP_ID])(                               \
      &norm_, &n_, A, &lda_, &anorm_, rcond, work, iwork, &info);              \
  free(work);                                                                  \
  free(iwork);                                                                 \
  return info;                                                                 \
}

#define DEFINE_GECON_COMPLEX_FORTRAN_TO_CBLAS(OP, CTYPE, RTYPE, OP_ID)         \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const char norm, const int n, const CTYPE *A,    \
    const int lda, const RTYPE anorm, RTYPE *rcond) {                          \
  if (!fb_lapack_layout_is_col_major(layout)) return -1;                       \
  char norm_ = norm;                                                           \
  int n_ = n, lda_ = lda, info = 0;                                            \
  int work_size = (n > 0) ? (2 * n) : 1;                                       \
  CTYPE *work = NULL;                                                          \
  RTYPE *rwork = NULL;                                                         \
  RTYPE anorm_ = anorm;                                                        \
  if (!g_fortran_fn[OP_ID]) return -1;                                         \
  work = (CTYPE *)malloc((size_t)work_size * sizeof(CTYPE));                   \
  rwork = (RTYPE *)malloc((size_t)work_size * sizeof(RTYPE));                  \
  if (!work || !rwork) {                                                       \
    free(work);                                                                \
    free(rwork);                                                               \
    return -101;                                                               \
  }                                                                            \
  ((void (*)(char *, int *, const CTYPE *, int *, RTYPE *, RTYPE *, CTYPE *,   \
             RTYPE *, int *))g_fortran_fn[OP_ID])(                             \
      &norm_, &n_, A, &lda_, &anorm_, rcond, work, rwork, &info);              \
  free(work);                                                                  \
  free(rwork);                                                                 \
  return info;                                                                 \
}

#define DEFINE_GECON_REAL_CBLAS_TO_FORTRAN(OP, TYPE, OP_ID)                    \
static void thunk_##OP##_cblas_to_fortran(                                     \
    char *norm, int *n, TYPE *A, int *lda, TYPE *anorm, TYPE *rcond,          \
    TYPE *work, int *iwork, int *info) {                                       \
  (void)work;                                                                   \
  (void)iwork;                                                                  \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, char, int, const TYPE *, int, TYPE, TYPE *))\
      g_cblas_fn[OP_ID])(FB_LAYOUT_COL_MAJOR, *norm, *n, A, *lda,             \
                          *anorm, rcond);                                       \
  if (info) *info = rc;                                                         \
}

#define DEFINE_GECON_COMPLEX_CBLAS_TO_FORTRAN(OP, CTYPE, RTYPE, OP_ID)        \
static void thunk_##OP##_cblas_to_fortran(                                     \
    char *norm, int *n, CTYPE *A, int *lda, RTYPE *anorm, RTYPE *rcond,       \
    CTYPE *work, RTYPE *rwork, int *info) {                                    \
  (void)work;                                                                   \
  (void)rwork;                                                                  \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, char, int, const CTYPE *, int, RTYPE,       \
                     RTYPE *))g_cblas_fn[OP_ID])(FB_LAYOUT_COL_MAJOR, *norm,   \
                                                 *n, A, *lda, *anorm, rcond);  \
  if (info) *info = rc;                                                         \
}

#define DEFINE_GEEQU_REAL_FORTRAN_TO_CBLAS(OP, TYPE, OP_ID)                    \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const int m, const int n, const TYPE *A,         \
    const int lda, TYPE *r, TYPE *c, TYPE *rowcnd, TYPE *colcnd,               \
    TYPE *amax) {                                                               \
  if (!fb_lapack_layout_is_col_major(layout)) return -1;                        \
  int m_ = m, n_ = n, lda_ = lda, info = 0;                                     \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  ((void (*)(int *, int *, const TYPE *, int *, TYPE *, TYPE *, TYPE *,         \
             TYPE *, TYPE *, int *))g_fortran_fn[OP_ID])(                       \
      &m_, &n_, A, &lda_, r, c, rowcnd, colcnd, amax, &info);                  \
  return info;                                                                  \
}

#define DEFINE_GEEQU_COMPLEX_FORTRAN_TO_CBLAS(OP, CTYPE, RTYPE, OP_ID)         \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const int m, const int n, const CTYPE *A,        \
    const int lda, RTYPE *r, RTYPE *c, RTYPE *rowcnd, RTYPE *colcnd,           \
    RTYPE *amax) {                                                              \
  if (!fb_lapack_layout_is_col_major(layout)) return -1;                       \
  int m_ = m, n_ = n, lda_ = lda, info = 0;                                    \
  if (!g_fortran_fn[OP_ID]) return -1;                                         \
  ((void (*)(int *, int *, const CTYPE *, int *, RTYPE *, RTYPE *, RTYPE *,    \
             RTYPE *, RTYPE *, int *))g_fortran_fn[OP_ID])(                    \
      &m_, &n_, A, &lda_, r, c, rowcnd, colcnd, amax, &info);                  \
  return info;                                                                 \
}

#define DEFINE_GEEQU_REAL_CBLAS_TO_FORTRAN(OP, TYPE, OP_ID)                    \
static void thunk_##OP##_cblas_to_fortran(                                     \
    int *m, int *n, TYPE *A, int *lda, TYPE *r, TYPE *c, TYPE *rowcnd,        \
    TYPE *colcnd, TYPE *amax, int *info) {                                     \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, int, int, const TYPE *, int, TYPE *, TYPE *,\
                     TYPE *, TYPE *, TYPE *))g_cblas_fn[OP_ID])(               \
      FB_LAYOUT_COL_MAJOR, *m, *n, A, *lda, r, c, rowcnd, colcnd, amax);      \
  if (info) *info = rc;                                                         \
}

#define DEFINE_GEEQU_COMPLEX_CBLAS_TO_FORTRAN(OP, CTYPE, RTYPE, OP_ID)        \
static void thunk_##OP##_cblas_to_fortran(                                     \
    int *m, int *n, CTYPE *A, int *lda, RTYPE *r, RTYPE *c, RTYPE *rowcnd,    \
    RTYPE *colcnd, RTYPE *amax, int *info) {                                   \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, int, int, const CTYPE *, int, RTYPE *,      \
                     RTYPE *, RTYPE *, RTYPE *, RTYPE *))g_cblas_fn[OP_ID])(  \
      FB_LAYOUT_COL_MAJOR, *m, *n, A, *lda, r, c, rowcnd, colcnd, amax);      \
  if (info) *info = rc;                                                         \
}

#define DEFINE_POCON_REAL_FORTRAN_TO_CBLAS(OP, TYPE, OP_ID)                    \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const fb_uplo_t uplo, const int n,               \
    const TYPE *a, const int lda, const TYPE anorm, TYPE *rcond,               \
    TYPE *work, int *iwork, int *info) {                                        \
  char uplo_char = 'U';                                                         \
  int n_ = n, lda_ = lda, info_local = 0;                                       \
  TYPE anorm_local = anorm;                                                     \
  if (!fb_lapack_layout_is_col_major(layout)) return -1;                       \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (fb_lapack_uplo_to_char(uplo, &uplo_char) != 0) return -1;                \
  ((void (*)(char *, int *, const TYPE *, int *, TYPE *, TYPE *, TYPE *,       \
             int *, int *))g_fortran_fn[OP_ID])(                               \
      &uplo_char, &n_, a, &lda_, &anorm_local, rcond, work, iwork,             \
      (info != NULL) ? info : &info_local);                                     \
  return (info != NULL) ? *info : info_local;                                   \
}

#define DEFINE_POCON_REAL_CBLAS_TO_FORTRAN(OP, TYPE, OP_ID)                    \
static void thunk_##OP##_cblas_to_fortran(                                     \
    char *uplo, int *n, TYPE *a, int *lda, TYPE *anorm, TYPE *rcond,           \
    TYPE *work, int *iwork, int *info) {                                        \
  fb_uplo_t uplo_ = FB_UPPER;                                                   \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!uplo || !n || !anorm || !rcond ||                                        \
      fb_lapack_char_to_uplo(*uplo, &uplo_) != 0) {                            \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, fb_uplo_t, int, const TYPE *, int, TYPE,     \
                     TYPE *, TYPE *, int *, int *))g_cblas_fn[OP_ID])(         \
      FB_LAYOUT_COL_MAJOR, uplo_, *n, a, *lda, *anorm, rcond, work, iwork,     \
      info);                                                                    \
  if (info) *info = rc;                                                         \
}

#define DEFINE_POCON_COMPLEX_FORTRAN_TO_CBLAS(OP, CTYPE, RTYPE, OP_ID)        \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const fb_uplo_t uplo, const int n,               \
    const CTYPE *a, const int lda, const RTYPE anorm, RTYPE *rcond,            \
    RTYPE *rwork, int *info) {                                                  \
  char uplo_char = 'U';                                                         \
  int n_ = n, lda_ = lda, info_local = 0;                                       \
  RTYPE anorm_local = anorm;                                                    \
  int work_size = (n > 0) ? (2 * n) : 1;                                        \
  CTYPE *work = NULL;                                                           \
  if (!fb_lapack_layout_is_col_major(layout)) return -1;                       \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (fb_lapack_uplo_to_char(uplo, &uplo_char) != 0) return -1;                \
  work = (CTYPE *)malloc((size_t)work_size * sizeof(CTYPE));                   \
  if (!work) return -101;                                                       \
  ((void (*)(char *, int *, const CTYPE *, int *, RTYPE *, RTYPE *, CTYPE *,   \
             RTYPE *, int *))g_fortran_fn[OP_ID])(                             \
      &uplo_char, &n_, a, &lda_, &anorm_local, rcond, work, rwork,             \
      (info != NULL) ? info : &info_local);                                     \
  free(work);                                                                   \
  return (info != NULL) ? *info : info_local;                                   \
}

#define DEFINE_POCON_COMPLEX_CBLAS_TO_FORTRAN(OP, CTYPE, RTYPE, OP_ID)        \
static void thunk_##OP##_cblas_to_fortran(                                     \
    char *uplo, int *n, CTYPE *a, int *lda, RTYPE *anorm, RTYPE *rcond,        \
    CTYPE *work, RTYPE *rwork, int *info) {                                     \
  fb_uplo_t uplo_ = FB_UPPER;                                                   \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!uplo || !n || !anorm || !rcond ||                                        \
      fb_lapack_char_to_uplo(*uplo, &uplo_) != 0) {                            \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, fb_uplo_t, int, const CTYPE *, int, RTYPE,   \
                     RTYPE *, RTYPE *, int *))g_cblas_fn[OP_ID])(              \
      FB_LAYOUT_COL_MAJOR, uplo_, *n, a, *lda, *anorm, rcond, rwork, info);    \
  if (info) *info = rc;                                                         \
  (void)work;                                                                   \
}

#define DEFINE_POEQU_REAL_FORTRAN_TO_CBLAS(OP, TYPE, OP_ID)                    \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const int n, const TYPE *a, const int lda,       \
    TYPE *s, TYPE *scond, TYPE *amax, int *info) {                              \
  int n_ = n, lda_ = lda, info_local = 0;                                       \
  if (!fb_lapack_layout_is_col_major(layout)) return -1;                       \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  ((void (*)(int *, const TYPE *, int *, TYPE *, TYPE *, TYPE *, int *))       \
      g_fortran_fn[OP_ID])(&n_, a, &lda_, s, scond, amax,                      \
                           (info != NULL) ? info : &info_local);               \
  return (info != NULL) ? *info : info_local;                                   \
}

#define DEFINE_POEQU_REAL_CBLAS_TO_FORTRAN(OP, TYPE, OP_ID)                    \
static void thunk_##OP##_cblas_to_fortran(                                     \
    int *n, TYPE *a, int *lda, TYPE *s, TYPE *scond, TYPE *amax, int *info) {  \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!n || !s || !scond || !amax) {                                            \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, int, const TYPE *, int, TYPE *, TYPE *,      \
                     TYPE *, int *))g_cblas_fn[OP_ID])(                        \
      FB_LAYOUT_COL_MAJOR, *n, a, *lda, s, scond, amax, info);                 \
  if (info) *info = rc;                                                         \
}

#define DEFINE_POEQU_COMPLEX_FORTRAN_TO_CBLAS(OP, CTYPE, RTYPE, OP_ID)        \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const int n, const CTYPE *a, const int lda,      \
    RTYPE *s, RTYPE *scond, RTYPE *amax, int *info) {                           \
  int n_ = n, lda_ = lda, info_local = 0;                                       \
  if (!fb_lapack_layout_is_col_major(layout)) return -1;                       \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  ((void (*)(int *, const CTYPE *, int *, RTYPE *, RTYPE *, RTYPE *, int *))   \
      g_fortran_fn[OP_ID])(&n_, a, &lda_, s, scond, amax,                      \
                           (info != NULL) ? info : &info_local);               \
  return (info != NULL) ? *info : info_local;                                   \
}

#define DEFINE_POEQU_COMPLEX_CBLAS_TO_FORTRAN(OP, CTYPE, RTYPE, OP_ID)        \
static void thunk_##OP##_cblas_to_fortran(                                     \
    int *n, CTYPE *a, int *lda, RTYPE *s, RTYPE *scond, RTYPE *amax, int *info) { \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!n || !s || !scond || !amax) {                                            \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, int, const CTYPE *, int, RTYPE *, RTYPE *,   \
                     RTYPE *, int *))g_cblas_fn[OP_ID])(                       \
      FB_LAYOUT_COL_MAJOR, *n, a, *lda, s, scond, amax, info);                 \
  if (info) *info = rc;                                                         \
}

#define DEFINE_GEBAK_REAL_FORTRAN_TO_CBLAS(OP, TYPE, OP_ID)                    \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, char job, char side, const int n,                \
    const int ilo, const int ihi, const TYPE *scale, const int m, TYPE *V,     \
    const int ldv) {                                                            \
  if (!fb_lapack_layout_is_col_major(layout)) return -1;                       \
  char job_ = job, side_ = side;                                                \
  int n_ = n, ilo_ = ilo, ihi_ = ihi, m_ = m, ldv_ = ldv, info = 0;            \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  ((void (*)(char *, char *, int *, int *, int *, TYPE *, int *, TYPE *,       \
             int *, int *))g_fortran_fn[OP_ID])(                                \
      &job_, &side_, &n_, &ilo_, &ihi_, (TYPE *)scale, &m_, V, &ldv_, &info); \
  return info;                                                                  \
}

#define DEFINE_GEBAK_COMPLEX_FORTRAN_TO_CBLAS(OP, CTYPE, RTYPE, OP_ID)         \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, char job, char side, const int n,                \
    const int ilo, const int ihi, const RTYPE *scale, const int m, CTYPE *V,   \
    const int ldv) {                                                            \
  if (!fb_lapack_layout_is_col_major(layout)) return -1;                       \
  char job_ = job, side_ = side;                                                \
  int n_ = n, ilo_ = ilo, ihi_ = ihi, m_ = m, ldv_ = ldv, info = 0;            \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  ((void (*)(char *, char *, int *, int *, int *, RTYPE *, int *, CTYPE *,     \
             int *, int *))g_fortran_fn[OP_ID])(                                \
      &job_, &side_, &n_, &ilo_, &ihi_, (RTYPE *)scale, &m_, V, &ldv_, &info);\
  return info;                                                                  \
}

#define DEFINE_GEBAK_REAL_CBLAS_TO_FORTRAN(OP, TYPE, OP_ID)                    \
static void thunk_##OP##_cblas_to_fortran(                                     \
    char *job, char *side, int *n, int *ilo, int *ihi, TYPE *scale, int *m,   \
    TYPE *V, int *ldv, int *info) {                                             \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, char, char, int, int, int, const TYPE *,     \
                     int, TYPE *, int))g_cblas_fn[OP_ID])(                      \
      FB_LAYOUT_COL_MAJOR, *job, *side, *n, *ilo, *ihi, scale, *m, V, *ldv);  \
  if (info) *info = rc;                                                         \
}

#define DEFINE_GEBAK_COMPLEX_CBLAS_TO_FORTRAN(OP, CTYPE, RTYPE, OP_ID)         \
static void thunk_##OP##_cblas_to_fortran(                                     \
    char *job, char *side, int *n, int *ilo, int *ihi, RTYPE *scale, int *m,  \
    CTYPE *V, int *ldv, int *info) {                                            \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, char, char, int, int, int, const RTYPE *,    \
                     int, CTYPE *, int))g_cblas_fn[OP_ID])(                     \
      FB_LAYOUT_COL_MAJOR, *job, *side, *n, *ilo, *ihi, scale, *m, V, *ldv);  \
  if (info) *info = rc;                                                         \
}

#define DEFINE_GEBAL_REAL_FORTRAN_TO_CBLAS(OP, TYPE, OP_ID)                    \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, char job, const int n, TYPE *A, const int lda,   \
    int *ilo, int *ihi, TYPE *scale) {                                          \
  if (!fb_lapack_layout_is_col_major(layout)) return -1;                       \
  char job_ = job;                                                              \
  int n_ = n, lda_ = lda, info = 0;                                             \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  ((void (*)(char *, int *, TYPE *, int *, int *, int *, TYPE *, int *))       \
      g_fortran_fn[OP_ID])(&job_, &n_, A, &lda_, ilo, ihi, scale, &info);      \
  return info;                                                                  \
}

#define DEFINE_GEBAL_COMPLEX_FORTRAN_TO_CBLAS(OP, CTYPE, RTYPE, OP_ID)         \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, char job, const int n, CTYPE *A, const int lda,  \
    int *ilo, int *ihi, RTYPE *scale) {                                         \
  if (!fb_lapack_layout_is_col_major(layout)) return -1;                       \
  char job_ = job;                                                              \
  int n_ = n, lda_ = lda, info = 0;                                             \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  ((void (*)(char *, int *, CTYPE *, int *, int *, int *, RTYPE *, int *))     \
      g_fortran_fn[OP_ID])(&job_, &n_, A, &lda_, ilo, ihi, scale, &info);      \
  return info;                                                                  \
}

#define DEFINE_GEBAL_REAL_CBLAS_TO_FORTRAN(OP, TYPE, OP_ID)                    \
static void thunk_##OP##_cblas_to_fortran(                                     \
    char *job, int *n, TYPE *A, int *lda, int *ilo, int *ihi, TYPE *scale,    \
    int *info) {                                                                \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, char, int, TYPE *, int, int *, int *, TYPE *))\
      g_cblas_fn[OP_ID])(FB_LAYOUT_COL_MAJOR, *job, *n, A, *lda,               \
                          ilo, ihi, scale);                                     \
  if (info) *info = rc;                                                         \
}

#define DEFINE_GEBAL_COMPLEX_CBLAS_TO_FORTRAN(OP, CTYPE, RTYPE, OP_ID)         \
static void thunk_##OP##_cblas_to_fortran(                                     \
    char *job, int *n, CTYPE *A, int *lda, int *ilo, int *ihi, RTYPE *scale,  \
    int *info) {                                                                \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, char, int, CTYPE *, int, int *, int *,       \
                     RTYPE *))g_cblas_fn[OP_ID])(FB_LAYOUT_COL_MAJOR, *job,    \
                                                 *n, A, *lda, ilo, ihi, scale);\
  if (info) *info = rc;                                                         \
}

#define DEFINE_GESDD_REAL_FORTRAN_TO_CBLAS(OP, TYPE, OP_ID, QUERY_FN)          \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, char jobz, const int m, const int n, TYPE *A,    \
    const int lda, TYPE *s, TYPE *U, const int ldu, TYPE *VT,                  \
    const int ldvt) {                                                          \
  if (!fb_lapack_layout_is_col_major(layout)) return -1;                       \
  char jobz_ = jobz;                                                           \
  int m_ = m, n_ = n, lda_ = lda, ldu_ = U ? ldu : 1,                          \
      ldvt_ = VT ? ldvt : 1, lwork_ = -1, info = 0;                            \
  TYPE work_query = (TYPE)0;                                                   \
  TYPE *work = NULL;                                                           \
  TYPE dummy_u = (TYPE)0;                                                      \
  TYPE dummy_vt = (TYPE)0;                                                     \
  TYPE *u_ptr = U ? U : &dummy_u;                                              \
  TYPE *vt_ptr = VT ? VT : &dummy_vt;                                          \
  int minmn = (m < n) ? m : n;                                                 \
  int iwork_size = (minmn > 1) ? (8 * minmn) : 1;                              \
  int *iwork = NULL;                                                           \
  if (!fb_lapack_job_is_none(jobz_) && (U == NULL || VT == NULL)) return -1;   \
  if (!g_fortran_fn[OP_ID]) return -1;                                         \
  iwork = (int *)malloc((size_t)iwork_size * sizeof(int));                     \
  if (!iwork) return -101;                                                     \
  ((void (*)(char *, int *, int *, TYPE *, int *, TYPE *, TYPE *, int *,       \
             TYPE *, int *, TYPE *, int *, int *, int *))g_fortran_fn[OP_ID])( \
      &jobz_, &m_, &n_, A, &lda_, s, u_ptr, &ldu_, vt_ptr, &ldvt_,             \
      &work_query, &lwork_, iwork, &info);                                     \
  if (info != 0) {                                                             \
    free(iwork);                                                               \
    return info;                                                               \
  }                                                                            \
  lwork_ = QUERY_FN(work_query);                                               \
  work = (TYPE *)malloc((size_t)lwork_ * sizeof(TYPE));                        \
  if (!work || !iwork) {                                                       \
    free(work);                                                                \
    free(iwork);                                                               \
    return -101;                                                               \
  }                                                                            \
  ((void (*)(char *, int *, int *, TYPE *, int *, TYPE *, TYPE *, int *,       \
             TYPE *, int *, TYPE *, int *, int *, int *))g_fortran_fn[OP_ID])( \
      &jobz_, &m_, &n_, A, &lda_, s, u_ptr, &ldu_, vt_ptr, &ldvt_,             \
      work, &lwork_, iwork, &info);                                            \
  free(work);                                                                  \
  free(iwork);                                                                 \
  return info;                                                                 \
}

#define DEFINE_GESDD_COMPLEX_FORTRAN_TO_CBLAS(OP, CTYPE, RTYPE, OP_ID, QUERY_FN) \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, char jobz, const int m, const int n, CTYPE *A,   \
    const int lda, RTYPE *s, CTYPE *U, const int ldu, CTYPE *VT,               \
    const int ldvt) {                                                          \
  if (!fb_lapack_layout_is_col_major(layout)) return -1;                       \
  char jobz_ = jobz;                                                           \
  int m_ = m, n_ = n, lda_ = lda, ldu_ = U ? ldu : 1,                          \
      ldvt_ = VT ? ldvt : 1, lwork_ = -1, info = 0;                            \
  CTYPE work_query = (CTYPE)0;                                                 \
  RTYPE dummy_rwork = (RTYPE)0;                                                \
  CTYPE *work = NULL;                                                          \
  RTYPE *rwork = NULL;                                                         \
  int minmn = (m < n) ? m : n;                                                 \
  int rwork_size = (minmn > 1) ? (5 * minmn) : 1;                              \
  int iwork_size = (minmn > 1) ? (8 * minmn) : 1;                              \
  int *iwork = NULL;                                                           \
  CTYPE dummy_u = (CTYPE)0;                                                    \
  CTYPE dummy_vt = (CTYPE)0;                                                   \
  CTYPE *u_ptr = U ? U : &dummy_u;                                             \
  CTYPE *vt_ptr = VT ? VT : &dummy_vt;                                         \
  if (!fb_lapack_job_is_none(jobz_) && (U == NULL || VT == NULL)) return -1;   \
  if (!g_fortran_fn[OP_ID]) return -1;                                         \
  iwork = (int *)malloc((size_t)iwork_size * sizeof(int));                     \
  if (!iwork) return -101;                                                     \
  ((void (*)(char *, int *, int *, CTYPE *, int *, RTYPE *, CTYPE *, int *,    \
             CTYPE *, int *, CTYPE *, int *, RTYPE *, int *, int *))           \
      g_fortran_fn[OP_ID])(&jobz_, &m_, &n_, A, &lda_, s, u_ptr, &ldu_,        \
                           vt_ptr, &ldvt_, &work_query, &lwork_,               \
                           &dummy_rwork, iwork, &info);                        \
  if (info != 0) {                                                             \
    free(iwork);                                                               \
    return info;                                                               \
  }                                                                            \
  lwork_ = QUERY_FN(work_query);                                               \
  work = (CTYPE *)malloc((size_t)lwork_ * sizeof(CTYPE));                      \
  rwork = (RTYPE *)malloc((size_t)rwork_size * sizeof(RTYPE));                 \
  if (!work || !rwork || !iwork) {                                             \
    free(work);                                                                \
    free(rwork);                                                               \
    free(iwork);                                                               \
    return -101;                                                               \
  }                                                                            \
  ((void (*)(char *, int *, int *, CTYPE *, int *, RTYPE *, CTYPE *, int *,    \
             CTYPE *, int *, CTYPE *, int *, RTYPE *, int *, int *))           \
      g_fortran_fn[OP_ID])(&jobz_, &m_, &n_, A, &lda_, s, u_ptr, &ldu_,        \
                           vt_ptr, &ldvt_, work, &lwork_, rwork, iwork,        \
                           &info);                                             \
  free(work);                                                                  \
  free(rwork);                                                                 \
  free(iwork);                                                                 \
  return info;                                                                 \
}

#define DEFINE_GESDD_REAL_CBLAS_TO_FORTRAN(OP, TYPE, OP_ID)                    \
static void thunk_##OP##_cblas_to_fortran(                                     \
    char *jobz, int *m, int *n, TYPE *A, int *lda, TYPE *s, TYPE *U,          \
    int *ldu, TYPE *VT, int *ldvt, TYPE *work, int *lwork, int *iwork,        \
    int *info) {                                                                \
  (void)work;                                                                   \
  (void)lwork;                                                                  \
  (void)iwork;                                                                  \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, char, int, int, TYPE *, int, TYPE *,        \
                     TYPE *, int, TYPE *, int))g_cblas_fn[OP_ID])(            \
      FB_LAYOUT_COL_MAJOR, *jobz, *m, *n, A, *lda, s, U, *ldu, VT, *ldvt);    \
  if (info) *info = rc;                                                         \
}

#define DEFINE_GESDD_COMPLEX_CBLAS_TO_FORTRAN(OP, CTYPE, RTYPE, OP_ID)        \
static void thunk_##OP##_cblas_to_fortran(                                     \
    char *jobz, int *m, int *n, CTYPE *A, int *lda, RTYPE *s, CTYPE *U,       \
    int *ldu, CTYPE *VT, int *ldvt, CTYPE *work, int *lwork, RTYPE *rwork,    \
    int *iwork, int *info) {                                                    \
  (void)work;                                                                   \
  (void)lwork;                                                                  \
  (void)rwork;                                                                  \
  (void)iwork;                                                                  \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, char, int, int, CTYPE *, int, RTYPE *,      \
                     CTYPE *, int, CTYPE *, int))g_cblas_fn[OP_ID])(          \
      FB_LAYOUT_COL_MAJOR, *jobz, *m, *n, A, *lda, s, U, *ldu, VT, *ldvt);    \
  if (info) *info = rc;                                                         \
}

#define DEFINE_SYGV_REAL_FORTRAN_TO_CBLAS(OP, TYPE, OP_ID, QUERY_FN)           \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const int itype, char jobz, const fb_uplo_t uplo,\
    const int n, TYPE *A, const int lda, TYPE *B, const int ldb, TYPE *w) {    \
  char jobz_ = jobz;                                                           \
  char uplo_ = fb_lapack_uplo_from_layout(layout, uplo);                       \
  int itype_ = itype, n_ = n, lda_ = lda, ldb_ = ldb, lwork_ = -1, info = 0;   \
  TYPE work_query = (TYPE)0;                                                   \
  TYPE *work = NULL;                                                           \
  if (!g_fortran_fn[OP_ID]) return -1;                                         \
  ((void (*)(int *, char *, char *, int *, TYPE *, int *, TYPE *, int *,       \
             TYPE *, TYPE *, int *, int *))g_fortran_fn[OP_ID])(               \
      &itype_, &jobz_, &uplo_, &n_, A, &lda_, B, &ldb_, w, &work_query,        \
      &lwork_, &info);                                                         \
  if (info != 0) return info;                                                  \
  lwork_ = QUERY_FN(work_query);                                               \
  work = (TYPE *)malloc((size_t)lwork_ * sizeof(TYPE));                        \
  if (!work) return -101;                                                      \
  ((void (*)(int *, char *, char *, int *, TYPE *, int *, TYPE *, int *,       \
             TYPE *, TYPE *, int *, int *))g_fortran_fn[OP_ID])(               \
      &itype_, &jobz_, &uplo_, &n_, A, &lda_, B, &ldb_, w, work, &lwork_,      \
      &info);                                                                  \
  free(work);                                                                  \
  return info;                                                                 \
}

#define DEFINE_HEGV_COMPLEX_FORTRAN_TO_CBLAS(OP, CTYPE, RTYPE, OP_ID, QUERY_FN) \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const int itype, char jobz, const fb_uplo_t uplo,\
    const int n, CTYPE *A, const int lda, CTYPE *B, const int ldb, RTYPE *w) { \
  char jobz_ = jobz;                                                           \
  char uplo_ = fb_lapack_uplo_from_layout(layout, uplo);                       \
  int itype_ = itype, n_ = n, lda_ = lda, ldb_ = ldb, lwork_ = -1, info = 0;   \
  CTYPE work_query = (CTYPE)0;                                                 \
  RTYPE rwork_query = (RTYPE)0;                                                \
  CTYPE *work = NULL;                                                          \
  RTYPE *rwork = NULL;                                                         \
  int rwork_size = (n > 1) ? (3 * n) : 1;                                      \
  if (!g_fortran_fn[OP_ID]) return -1;                                         \
  ((void (*)(int *, char *, char *, int *, CTYPE *, int *, CTYPE *, int *,     \
             RTYPE *, CTYPE *, int *, RTYPE *, int *))g_fortran_fn[OP_ID])(    \
      &itype_, &jobz_, &uplo_, &n_, A, &lda_, B, &ldb_, w, &work_query,        \
      &lwork_, &rwork_query, &info);                                           \
  if (info != 0) return info;                                                  \
  lwork_ = QUERY_FN(work_query);                                               \
  work = (CTYPE *)malloc((size_t)lwork_ * sizeof(CTYPE));                      \
  rwork = (RTYPE *)malloc((size_t)rwork_size * sizeof(RTYPE));                 \
  if (!work || !rwork) {                                                       \
    free(work);                                                                \
    free(rwork);                                                               \
    return -101;                                                               \
  }                                                                            \
  ((void (*)(int *, char *, char *, int *, CTYPE *, int *, CTYPE *, int *,     \
             RTYPE *, CTYPE *, int *, RTYPE *, int *))g_fortran_fn[OP_ID])(    \
      &itype_, &jobz_, &uplo_, &n_, A, &lda_, B, &ldb_, w, work, &lwork_,      \
      rwork, &info);                                                           \
  free(work);                                                                  \
  free(rwork);                                                                 \
  return info;                                                                 \
}

#define DEFINE_SYGV_REAL_CBLAS_TO_FORTRAN(OP, TYPE, OP_ID)                     \
static void thunk_##OP##_cblas_to_fortran(                                     \
    int *itype, char *jobz, char *uplo, int *n, TYPE *A, int *lda, TYPE *B,   \
    int *ldb, TYPE *w, TYPE *work, int *lwork, int *info) {                    \
  fb_uplo_t uplo_enum = FB_UPPER;                                               \
  (void)work;                                                                   \
  (void)lwork;                                                                  \
  if (fb_lapack_char_to_uplo(*uplo, &uplo_enum) != 0) {                        \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, int, char, fb_uplo_t, int, TYPE *, int,     \
                     TYPE *, int, TYPE *))g_cblas_fn[OP_ID])(                 \
      FB_LAYOUT_COL_MAJOR, *itype, *jobz, uplo_enum, *n, A, *lda, B, *ldb, w);\
  if (info) *info = rc;                                                         \
}

#define DEFINE_HEGV_COMPLEX_CBLAS_TO_FORTRAN(OP, CTYPE, RTYPE, OP_ID)         \
static void thunk_##OP##_cblas_to_fortran(                                     \
    int *itype, char *jobz, char *uplo, int *n, CTYPE *A, int *lda, CTYPE *B, \
    int *ldb, RTYPE *w, CTYPE *work, int *lwork, RTYPE *rwork, int *info) {   \
  fb_uplo_t uplo_enum = FB_UPPER;                                               \
  (void)work;                                                                   \
  (void)lwork;                                                                  \
  (void)rwork;                                                                  \
  if (fb_lapack_char_to_uplo(*uplo, &uplo_enum) != 0) {                        \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, int, char, fb_uplo_t, int, CTYPE *, int,    \
                     CTYPE *, int, RTYPE *))g_cblas_fn[OP_ID])(               \
      FB_LAYOUT_COL_MAJOR, *itype, *jobz, uplo_enum, *n, A, *lda, B, *ldb, w);\
  if (info) *info = rc;                                                         \
}

#define DEFINE_SYEVR_REAL_FORTRAN_TO_CBLAS(OP, TYPE, OP_ID, QUERY_FN)         \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const char jobz, const char range,               \
    const fb_uplo_t uplo, const int n, TYPE *A, const int lda, const TYPE vl,  \
    const TYPE vu, const int il, const int iu, const TYPE abstol, int *m,      \
    TYPE *w, TYPE *Z, const int ldz, int *isuppz) {                            \
  char jobz_ = jobz, range_ = range;                                           \
  char uplo_ = fb_lapack_uplo_from_layout(layout, uplo);                       \
  int n_ = n, lda_ = lda, il_ = il, iu_ = iu;                                  \
  int ldz_ = (Z != NULL) ? ldz : 1;                                            \
  int lwork_ = -1, liwork_ = -1, info = 0;                                     \
  TYPE vl_ = vl, vu_ = vu, abstol_ = abstol;                                   \
  TYPE work_query = (TYPE)0;                                                   \
  int iwork_query = 0;                                                         \
  TYPE dummy_z = (TYPE)0;                                                      \
  TYPE *work = NULL;                                                           \
  TYPE *z_ptr = Z ? Z : &dummy_z;                                              \
  int *iwork = NULL;                                                           \
  int dummy_isuppz[2] = { 0, 0 };                                              \
  int *isuppz_ptr = isuppz ? isuppz : dummy_isuppz;                            \
  if (!m || !w) return -1;                                                     \
  if (fb_lapack_job_requests_vectors(jobz_) && Z == NULL) return -1;           \
  if (fb_lapack_job_requests_vectors(jobz_) && isuppz == NULL) return -1;      \
  if (!g_fortran_fn[OP_ID]) return -1;                                         \
  ((void (*)(char *, char *, char *, int *, TYPE *, int *, TYPE *, TYPE *, int *,\
             int *, TYPE *, int *, TYPE *, TYPE *, int *, int *, TYPE *, int *,\
             int *, int *, int *))g_fortran_fn[OP_ID])(                        \
      &jobz_, &range_, &uplo_, &n_, A, &lda_, &vl_, &vu_, &il_, &iu_,          \
      &abstol_, m, w, z_ptr, &ldz_, isuppz_ptr, &work_query, &lwork_,          \
      &iwork_query, &liwork_, &info);                                          \
  if (info != 0) return info;                                                  \
  lwork_ = QUERY_FN(work_query);                                               \
  liwork_ = fb_query_size_from_int(iwork_query);                               \
  work = (TYPE *)malloc((size_t)lwork_ * sizeof(TYPE));                        \
  iwork = (int *)malloc((size_t)liwork_ * sizeof(int));                        \
  if (!work || !iwork) {                                                       \
    free(work);                                                                \
    free(iwork);                                                               \
    return -101;                                                               \
  }                                                                            \
  ((void (*)(char *, char *, char *, int *, TYPE *, int *, TYPE *, TYPE *, int *,\
             int *, TYPE *, int *, TYPE *, TYPE *, int *, int *, TYPE *, int *,\
             int *, int *, int *))g_fortran_fn[OP_ID])(                        \
      &jobz_, &range_, &uplo_, &n_, A, &lda_, &vl_, &vu_, &il_, &iu_,          \
      &abstol_, m, w, z_ptr, &ldz_, isuppz_ptr, work, &lwork_, iwork,          \
      &liwork_, &info);                                                        \
  free(work);                                                                  \
  free(iwork);                                                                 \
  return info;                                                                 \
}

#define DEFINE_HEEVR_COMPLEX_FORTRAN_TO_CBLAS(OP, CTYPE, RTYPE, OP_ID, CQUERY_FN, RQUERY_FN) \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const char jobz, const char range,               \
    const fb_uplo_t uplo, const int n, CTYPE *A, const int lda, const RTYPE vl,\
    const RTYPE vu, const int il, const int iu, const RTYPE abstol, int *m,    \
    RTYPE *w, CTYPE *Z, const int ldz, int *isuppz) {                          \
  char jobz_ = jobz, range_ = range;                                           \
  char uplo_ = fb_lapack_uplo_from_layout(layout, uplo);                       \
  int n_ = n, lda_ = lda, il_ = il, iu_ = iu;                                  \
  int ldz_ = (Z != NULL) ? ldz : 1;                                            \
  int lwork_ = -1, lrwork_ = -1, liwork_ = -1, info = 0;                       \
  RTYPE vl_ = vl, vu_ = vu, abstol_ = abstol;                                  \
  CTYPE work_query = (CTYPE)0;                                                 \
  RTYPE rwork_query = (RTYPE)0;                                                \
  int iwork_query = 0;                                                         \
  CTYPE dummy_z = (CTYPE)0;                                                    \
  CTYPE *work = NULL;                                                          \
  CTYPE *z_ptr = Z ? Z : &dummy_z;                                             \
  RTYPE *rwork = NULL;                                                         \
  int *iwork = NULL;                                                           \
  int dummy_isuppz[2] = { 0, 0 };                                              \
  int *isuppz_ptr = isuppz ? isuppz : dummy_isuppz;                            \
  if (!m || !w) return -1;                                                     \
  if (fb_lapack_job_requests_vectors(jobz_) && Z == NULL) return -1;           \
  if (fb_lapack_job_requests_vectors(jobz_) && isuppz == NULL) return -1;      \
  if (!g_fortran_fn[OP_ID]) return -1;                                         \
  ((void (*)(char *, char *, char *, int *, CTYPE *, int *, RTYPE *, RTYPE *,  \
             int *, int *, RTYPE *, int *, RTYPE *, CTYPE *, int *, int *,     \
             CTYPE *, int *, RTYPE *, int *, int *, int *, int *))             \
      g_fortran_fn[OP_ID])(&jobz_, &range_, &uplo_, &n_, A, &lda_, &vl_,       \
                           &vu_, &il_, &iu_, &abstol_, m, w, z_ptr, &ldz_,     \
                           isuppz_ptr, &work_query, &lwork_, &rwork_query,     \
                           &lrwork_, &iwork_query, &liwork_, &info);           \
  if (info != 0) return info;                                                  \
  lwork_ = CQUERY_FN(work_query);                                              \
  lrwork_ = RQUERY_FN(rwork_query);                                            \
  liwork_ = fb_query_size_from_int(iwork_query);                               \
  work = (CTYPE *)malloc((size_t)lwork_ * sizeof(CTYPE));                      \
  rwork = (RTYPE *)malloc((size_t)lrwork_ * sizeof(RTYPE));                    \
  iwork = (int *)malloc((size_t)liwork_ * sizeof(int));                        \
  if (!work || !rwork || !iwork) {                                             \
    free(work);                                                                \
    free(rwork);                                                               \
    free(iwork);                                                               \
    return -101;                                                               \
  }                                                                            \
  ((void (*)(char *, char *, char *, int *, CTYPE *, int *, RTYPE *, RTYPE *,  \
             int *, int *, RTYPE *, int *, RTYPE *, CTYPE *, int *, int *,     \
             CTYPE *, int *, RTYPE *, int *, int *, int *, int *))             \
      g_fortran_fn[OP_ID])(&jobz_, &range_, &uplo_, &n_, A, &lda_, &vl_,       \
                           &vu_, &il_, &iu_, &abstol_, m, w, z_ptr, &ldz_,     \
                           isuppz_ptr, work, &lwork_, rwork, &lrwork_, iwork,  \
                           &liwork_, &info);                                   \
  free(work);                                                                  \
  free(rwork);                                                                 \
  free(iwork);                                                                 \
  return info;                                                                 \
}

#define DEFINE_SYEVR_REAL_CBLAS_TO_FORTRAN(OP, TYPE, OP_ID)                   \
static void thunk_##OP##_cblas_to_fortran(                                     \
    char *jobz, char *range, char *uplo, int *n, TYPE *A, int *lda, TYPE *vl, \
    TYPE *vu, int *il, int *iu, TYPE *abstol, int *m, TYPE *w, TYPE *Z,       \
    int *ldz, int *isuppz, TYPE *work, int *lwork, int *iwork, int *liwork,   \
    int *info) {                                                               \
  fb_uplo_t uplo_enum = FB_UPPER;                                              \
  TYPE dummy_z = (TYPE)0;                                                      \
  TYPE *z_ptr = Z ? Z : &dummy_z;                                              \
  int ldz_value = Z ? *ldz : 1;                                                \
  int dummy_isuppz[2] = { 0, 0 };                                              \
  int *isuppz_ptr = isuppz ? isuppz : dummy_isuppz;                            \
  (void)work;                                                                  \
  (void)lwork;                                                                 \
  (void)iwork;                                                                 \
  (void)liwork;                                                                \
  if (fb_lapack_char_to_uplo(*uplo, &uplo_enum) != 0) {                        \
    if (info) *info = -1;                                                      \
    return;                                                                    \
  }                                                                            \
  if (fb_lapack_job_requests_vectors(*jobz) && Z == NULL) {                    \
    if (info) *info = -1;                                                      \
    return;                                                                    \
  }                                                                            \
  if (fb_lapack_job_requests_vectors(*jobz) && isuppz == NULL) {               \
    if (info) *info = -1;                                                      \
    return;                                                                    \
  }                                                                            \
  if (!g_cblas_fn[OP_ID]) {                                                    \
    if (info) *info = -1;                                                      \
    return;                                                                    \
  }                                                                            \
  int rc = ((int (*)(fb_layout_t, char, char, fb_uplo_t, int, TYPE *, int,    \
                     TYPE, TYPE, int, int, TYPE, int *, TYPE *, TYPE *, int,  \
                     int *))g_cblas_fn[OP_ID])(                                \
      FB_LAYOUT_COL_MAJOR, *jobz, *range, uplo_enum, *n, A, *lda, *vl, *vu,   \
      *il, *iu, *abstol, m, w, z_ptr, ldz_value, isuppz_ptr);                  \
  if (info) *info = rc;                                                        \
}

#define DEFINE_HEEVR_COMPLEX_CBLAS_TO_FORTRAN(OP, CTYPE, RTYPE, OP_ID)        \
static void thunk_##OP##_cblas_to_fortran(                                     \
    char *jobz, char *range, char *uplo, int *n, CTYPE *A, int *lda,          \
    RTYPE *vl, RTYPE *vu, int *il, int *iu, RTYPE *abstol, int *m, RTYPE *w,  \
    CTYPE *Z, int *ldz, int *isuppz, CTYPE *work, int *lwork, RTYPE *rwork,   \
    int *lrwork, int *iwork, int *liwork, int *info) {                         \
  fb_uplo_t uplo_enum = FB_UPPER;                                              \
  CTYPE dummy_z = (CTYPE)0;                                                    \
  CTYPE *z_ptr = Z ? Z : &dummy_z;                                             \
  int ldz_value = Z ? *ldz : 1;                                                \
  int dummy_isuppz[2] = { 0, 0 };                                              \
  int *isuppz_ptr = isuppz ? isuppz : dummy_isuppz;                            \
  (void)work;                                                                  \
  (void)lwork;                                                                 \
  (void)rwork;                                                                 \
  (void)lrwork;                                                                \
  (void)iwork;                                                                 \
  (void)liwork;                                                                \
  if (fb_lapack_char_to_uplo(*uplo, &uplo_enum) != 0) {                        \
    if (info) *info = -1;                                                      \
    return;                                                                    \
  }                                                                            \
  if (fb_lapack_job_requests_vectors(*jobz) && Z == NULL) {                    \
    if (info) *info = -1;                                                      \
    return;                                                                    \
  }                                                                            \
  if (fb_lapack_job_requests_vectors(*jobz) && isuppz == NULL) {               \
    if (info) *info = -1;                                                      \
    return;                                                                    \
  }                                                                            \
  if (!g_cblas_fn[OP_ID]) {                                                    \
    if (info) *info = -1;                                                      \
    return;                                                                    \
  }                                                                            \
  int rc = ((int (*)(fb_layout_t, char, char, fb_uplo_t, int, CTYPE *, int,   \
                     RTYPE, RTYPE, int, int, RTYPE, int *, RTYPE *, CTYPE *,  \
                     int, int *))g_cblas_fn[OP_ID])(                           \
      FB_LAYOUT_COL_MAJOR, *jobz, *range, uplo_enum, *n, A, *lda, *vl, *vu,   \
      *il, *iu, *abstol, m, w, z_ptr, ldz_value, isuppz_ptr);                  \
  if (info) *info = rc;                                                        \
}

#define DEFINE_SYEVX_REAL_FORTRAN_TO_CBLAS(OP, TYPE, OP_ID, QUERY_FN)         \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const char jobz, const char range,               \
    const fb_uplo_t uplo, const int n, TYPE *A, const int lda, const TYPE vl,  \
    const TYPE vu, const int il, const int iu, const TYPE abstol, int *m,      \
    TYPE *w, TYPE *Z, const int ldz, int *ifail) {                              \
  char jobz_ = jobz, range_ = range;                                           \
  char uplo_ = fb_lapack_uplo_from_layout(layout, uplo);                       \
  int n_ = n, lda_ = lda, il_ = il, iu_ = iu;                                  \
  int ldz_ = (Z != NULL) ? ldz : 1;                                            \
  int lwork_ = -1, info = 0;                                                   \
  int iwork_size = (n > 0) ? (5 * n) : 1;                                      \
  TYPE vl_ = vl, vu_ = vu, abstol_ = abstol;                                   \
  TYPE work_query = (TYPE)0;                                                   \
  TYPE dummy_z = (TYPE)0;                                                      \
  TYPE *work = NULL;                                                           \
  TYPE *z_ptr = Z ? Z : &dummy_z;                                              \
  int *iwork = NULL;                                                           \
  int dummy_ifail = 0;                                                         \
  int *ifail_ptr = ifail ? ifail : &dummy_ifail;                               \
  if (!m || !w) return -1;                                                     \
  if (fb_lapack_job_requests_vectors(jobz_) && Z == NULL) return -1;           \
  if (fb_lapack_job_requests_vectors(jobz_) && ifail == NULL) return -1;       \
  if (!g_fortran_fn[OP_ID]) return -1;                                         \
  iwork = (int *)malloc((size_t)iwork_size * sizeof(int));                     \
  if (!iwork) return -101;                                                     \
  ((void (*)(char *, char *, char *, int *, TYPE *, int *, TYPE *, TYPE *, int *,\
             int *, TYPE *, int *, TYPE *, TYPE *, int *, TYPE *, int *, int *,\
             int *, int *))g_fortran_fn[OP_ID])(                               \
      &jobz_, &range_, &uplo_, &n_, A, &lda_, &vl_, &vu_, &il_, &iu_,          \
      &abstol_, m, w, z_ptr, &ldz_, &work_query, &lwork_, iwork, ifail_ptr,    \
      &info);                                                                  \
  if (info != 0) {                                                             \
    free(iwork);                                                               \
    return info;                                                               \
  }                                                                            \
  lwork_ = QUERY_FN(work_query);                                               \
  work = (TYPE *)malloc((size_t)lwork_ * sizeof(TYPE));                        \
  if (!work) {                                                                 \
    free(iwork);                                                               \
    return -101;                                                               \
  }                                                                            \
  ((void (*)(char *, char *, char *, int *, TYPE *, int *, TYPE *, TYPE *, int *,\
             int *, TYPE *, int *, TYPE *, TYPE *, int *, TYPE *, int *, int *,\
             int *, int *))g_fortran_fn[OP_ID])(                               \
      &jobz_, &range_, &uplo_, &n_, A, &lda_, &vl_, &vu_, &il_, &iu_,          \
      &abstol_, m, w, z_ptr, &ldz_, work, &lwork_, iwork, ifail_ptr, &info);   \
  free(work);                                                                  \
  free(iwork);                                                                 \
  return info;                                                                 \
}

#define DEFINE_HEEVX_COMPLEX_FORTRAN_TO_CBLAS(OP, CTYPE, RTYPE, OP_ID, QUERY_FN) \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const char jobz, const char range,               \
    const fb_uplo_t uplo, const int n, CTYPE *A, const int lda, const RTYPE vl,\
    const RTYPE vu, const int il, const int iu, const RTYPE abstol, int *m,    \
    RTYPE *w, CTYPE *Z, const int ldz, int *ifail) {                           \
  char jobz_ = jobz, range_ = range;                                           \
  char uplo_ = fb_lapack_uplo_from_layout(layout, uplo);                       \
  int n_ = n, lda_ = lda, il_ = il, iu_ = iu;                                  \
  int ldz_ = (Z != NULL) ? ldz : 1;                                            \
  int lwork_ = -1, info = 0;                                                   \
  int rwork_size = (n > 0) ? (7 * n) : 1;                                      \
  int iwork_size = (n > 0) ? (5 * n) : 1;                                      \
  RTYPE vl_ = vl, vu_ = vu, abstol_ = abstol;                                  \
  CTYPE work_query = (CTYPE)0;                                                 \
  CTYPE dummy_z = (CTYPE)0;                                                    \
  CTYPE *work = NULL;                                                          \
  CTYPE *z_ptr = Z ? Z : &dummy_z;                                             \
  RTYPE *rwork = NULL;                                                         \
  int *iwork = NULL;                                                           \
  int dummy_ifail = 0;                                                         \
  int *ifail_ptr = ifail ? ifail : &dummy_ifail;                               \
  if (!m || !w) return -1;                                                     \
  if (fb_lapack_job_requests_vectors(jobz_) && Z == NULL) return -1;           \
  if (fb_lapack_job_requests_vectors(jobz_) && ifail == NULL) return -1;       \
  if (!g_fortran_fn[OP_ID]) return -1;                                         \
  rwork = (RTYPE *)malloc((size_t)rwork_size * sizeof(RTYPE));                 \
  iwork = (int *)malloc((size_t)iwork_size * sizeof(int));                     \
  if (!rwork || !iwork) {                                                      \
    free(rwork);                                                               \
    free(iwork);                                                               \
    return -101;                                                               \
  }                                                                            \
  ((void (*)(char *, char *, char *, int *, CTYPE *, int *, RTYPE *, RTYPE *,  \
             int *, int *, RTYPE *, int *, RTYPE *, CTYPE *, int *, CTYPE *,   \
             int *, RTYPE *, int *, int *, int *))g_fortran_fn[OP_ID])(        \
      &jobz_, &range_, &uplo_, &n_, A, &lda_, &vl_, &vu_, &il_, &iu_,          \
      &abstol_, m, w, z_ptr, &ldz_, &work_query, &lwork_, rwork, iwork,        \
      ifail_ptr, &info);                                                       \
  if (info != 0) {                                                             \
    free(rwork);                                                               \
    free(iwork);                                                               \
    return info;                                                               \
  }                                                                            \
  lwork_ = QUERY_FN(work_query);                                               \
  work = (CTYPE *)malloc((size_t)lwork_ * sizeof(CTYPE));                      \
  if (!work) {                                                                 \
    free(rwork);                                                               \
    free(iwork);                                                               \
    return -101;                                                               \
  }                                                                            \
  ((void (*)(char *, char *, char *, int *, CTYPE *, int *, RTYPE *, RTYPE *,  \
             int *, int *, RTYPE *, int *, RTYPE *, CTYPE *, int *, CTYPE *,   \
             int *, RTYPE *, int *, int *, int *))g_fortran_fn[OP_ID])(        \
      &jobz_, &range_, &uplo_, &n_, A, &lda_, &vl_, &vu_, &il_, &iu_,          \
      &abstol_, m, w, z_ptr, &ldz_, work, &lwork_, rwork, iwork, ifail_ptr,    \
      &info);                                                                  \
  free(work);                                                                  \
  free(rwork);                                                                 \
  free(iwork);                                                                 \
  return info;                                                                 \
}

#define DEFINE_SYEVX_REAL_CBLAS_TO_FORTRAN(OP, TYPE, OP_ID)                   \
static void thunk_##OP##_cblas_to_fortran(                                     \
    char *jobz, char *range, char *uplo, int *n, TYPE *A, int *lda, TYPE *vl, \
    TYPE *vu, int *il, int *iu, TYPE *abstol, int *m, TYPE *w, TYPE *Z,       \
    int *ldz, TYPE *work, int *lwork, int *iwork, int *ifail, int *info) {    \
  fb_uplo_t uplo_enum = FB_UPPER;                                               \
  TYPE dummy_z = (TYPE)0;                                                      \
  TYPE *z_ptr = Z ? Z : &dummy_z;                                              \
  int ldz_value = Z ? *ldz : 1;                                                \
  int dummy_ifail = 0;                                                         \
  int *ifail_ptr = ifail ? ifail : &dummy_ifail;                               \
  (void)work;                                                                   \
  (void)lwork;                                                                  \
  (void)iwork;                                                                  \
  if (fb_lapack_char_to_uplo(*uplo, &uplo_enum) != 0) {                        \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (fb_lapack_job_requests_vectors(*jobz) && Z == NULL) {                    \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (fb_lapack_job_requests_vectors(*jobz) && ifail == NULL) {                \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, char, char, fb_uplo_t, int, TYPE *, int,    \
                     TYPE, TYPE, int, int, TYPE, int *, TYPE *, TYPE *, int,  \
                     int *))g_cblas_fn[OP_ID])(                                \
      FB_LAYOUT_COL_MAJOR, *jobz, *range, uplo_enum, *n, A, *lda, *vl, *vu,   \
      *il, *iu, *abstol, m, w, z_ptr, ldz_value, ifail_ptr);                   \
  if (info) *info = rc;                                                         \
}

#define DEFINE_HEEVX_COMPLEX_CBLAS_TO_FORTRAN(OP, CTYPE, RTYPE, OP_ID)        \
static void thunk_##OP##_cblas_to_fortran(                                     \
    char *jobz, char *range, char *uplo, int *n, CTYPE *A, int *lda,          \
    RTYPE *vl, RTYPE *vu, int *il, int *iu, RTYPE *abstol, int *m, RTYPE *w,  \
    CTYPE *Z, int *ldz, CTYPE *work, int *lwork, RTYPE *rwork, int *iwork,    \
    int *ifail, int *info) {                                                   \
  fb_uplo_t uplo_enum = FB_UPPER;                                               \
  CTYPE dummy_z = (CTYPE)0;                                                    \
  CTYPE *z_ptr = Z ? Z : &dummy_z;                                             \
  int ldz_value = Z ? *ldz : 1;                                                \
  int dummy_ifail = 0;                                                         \
  int *ifail_ptr = ifail ? ifail : &dummy_ifail;                               \
  (void)work;                                                                   \
  (void)lwork;                                                                  \
  (void)rwork;                                                                  \
  (void)iwork;                                                                  \
  if (fb_lapack_char_to_uplo(*uplo, &uplo_enum) != 0) {                        \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (fb_lapack_job_requests_vectors(*jobz) && Z == NULL) {                    \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (fb_lapack_job_requests_vectors(*jobz) && ifail == NULL) {                \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, char, char, fb_uplo_t, int, CTYPE *, int,   \
                     RTYPE, RTYPE, int, int, RTYPE, int *, RTYPE *, CTYPE *,  \
                     int, int *))g_cblas_fn[OP_ID])(                           \
      FB_LAYOUT_COL_MAJOR, *jobz, *range, uplo_enum, *n, A, *lda, *vl, *vu,   \
      *il, *iu, *abstol, m, w, z_ptr, ldz_value, ifail_ptr);                   \
  if (info) *info = rc;                                                         \
}

#define DEFINE_SYGVX_REAL_FORTRAN_TO_CBLAS(OP, TYPE, OP_ID, QUERY_FN)         \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const int itype, char jobz, char range,          \
    const fb_uplo_t uplo, const int n, TYPE *A, const int lda, TYPE *B,        \
    const int ldb, const TYPE vl, const TYPE vu, const int il, const int iu,   \
    const TYPE abstol, int *m, TYPE *w, TYPE *Z, const int ldz, int *ifail) {  \
  char jobz_ = jobz, range_ = range;                                           \
  char uplo_ = fb_lapack_uplo_from_layout(layout, uplo);                       \
  int itype_ = itype, n_ = n, lda_ = lda, ldb_ = ldb, il_ = il, iu_ = iu;      \
  int ldz_ = (Z != NULL) ? ldz : 1;                                            \
  int lwork_ = -1, info = 0;                                                   \
  int iwork_size = (n > 0) ? (5 * n) : 1;                                      \
  TYPE vl_ = vl, vu_ = vu, abstol_ = abstol;                                   \
  TYPE work_query = (TYPE)0;                                                   \
  TYPE dummy_z = (TYPE)0;                                                      \
  TYPE *work = NULL;                                                           \
  TYPE *z_ptr = Z ? Z : &dummy_z;                                              \
  int *iwork = NULL;                                                           \
  int dummy_ifail = 0;                                                         \
  int *ifail_ptr = ifail ? ifail : &dummy_ifail;                               \
  if (!m || !w) return -1;                                                     \
  if (fb_lapack_job_requests_vectors(jobz_) && Z == NULL) return -1;           \
  if (fb_lapack_job_requests_vectors(jobz_) && ifail == NULL) return -1;       \
  if (!g_fortran_fn[OP_ID]) return -1;                                         \
  iwork = (int *)malloc((size_t)iwork_size * sizeof(int));                     \
  if (!iwork) return -101;                                                     \
  ((void (*)(int *, char *, char *, char *, int *, TYPE *, int *, TYPE *, int *,\
             TYPE *, TYPE *, int *, int *, TYPE *, int *, TYPE *, TYPE *, int *,\
             TYPE *, int *, int *, int *, int *))g_fortran_fn[OP_ID])(         \
      &itype_, &jobz_, &range_, &uplo_, &n_, A, &lda_, B, &ldb_, &vl_, &vu_,  \
      &il_, &iu_, &abstol_, m, w, z_ptr, &ldz_, &work_query, &lwork_, iwork,   \
      ifail_ptr, &info);                                                       \
  if (info != 0) {                                                             \
    free(iwork);                                                               \
    return info;                                                               \
  }                                                                            \
  lwork_ = QUERY_FN(work_query);                                               \
  work = (TYPE *)malloc((size_t)lwork_ * sizeof(TYPE));                        \
  if (!work) {                                                                 \
    free(iwork);                                                               \
    return -101;                                                               \
  }                                                                            \
  ((void (*)(int *, char *, char *, char *, int *, TYPE *, int *, TYPE *, int *,\
             TYPE *, TYPE *, int *, int *, TYPE *, int *, TYPE *, TYPE *, int *,\
             TYPE *, int *, int *, int *, int *))g_fortran_fn[OP_ID])(         \
      &itype_, &jobz_, &range_, &uplo_, &n_, A, &lda_, B, &ldb_, &vl_, &vu_,  \
      &il_, &iu_, &abstol_, m, w, z_ptr, &ldz_, work, &lwork_, iwork,          \
      ifail_ptr, &info);                                                       \
  free(work);                                                                  \
  free(iwork);                                                                 \
  return info;                                                                 \
}

#define DEFINE_HEGVX_COMPLEX_FORTRAN_TO_CBLAS(OP, CTYPE, RTYPE, OP_ID, QUERY_FN) \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const int itype, char jobz, char range,          \
    const fb_uplo_t uplo, const int n, CTYPE *A, const int lda, CTYPE *B,      \
    const int ldb, const RTYPE vl, const RTYPE vu, const int il, const int iu, \
    const RTYPE abstol, int *m, RTYPE *w, CTYPE *Z, const int ldz, int *ifail) {\
  char jobz_ = jobz, range_ = range;                                           \
  char uplo_ = fb_lapack_uplo_from_layout(layout, uplo);                       \
  int itype_ = itype, n_ = n, lda_ = lda, ldb_ = ldb, il_ = il, iu_ = iu;      \
  int ldz_ = (Z != NULL) ? ldz : 1;                                            \
  int lwork_ = -1, info = 0;                                                   \
  int rwork_size = (n > 0) ? (7 * n) : 1;                                      \
  int iwork_size = (n > 0) ? (5 * n) : 1;                                      \
  RTYPE vl_ = vl, vu_ = vu, abstol_ = abstol;                                  \
  CTYPE work_query = (CTYPE)0;                                                 \
  CTYPE dummy_z = (CTYPE)0;                                                    \
  CTYPE *work = NULL;                                                          \
  CTYPE *z_ptr = Z ? Z : &dummy_z;                                             \
  RTYPE *rwork = NULL;                                                         \
  int *iwork = NULL;                                                           \
  int dummy_ifail = 0;                                                         \
  int *ifail_ptr = ifail ? ifail : &dummy_ifail;                               \
  if (!m || !w) return -1;                                                     \
  if (fb_lapack_job_requests_vectors(jobz_) && Z == NULL) return -1;           \
  if (fb_lapack_job_requests_vectors(jobz_) && ifail == NULL) return -1;       \
  if (!g_fortran_fn[OP_ID]) return -1;                                         \
  rwork = (RTYPE *)malloc((size_t)rwork_size * sizeof(RTYPE));                 \
  iwork = (int *)malloc((size_t)iwork_size * sizeof(int));                     \
  if (!rwork || !iwork) {                                                      \
    free(rwork);                                                               \
    free(iwork);                                                               \
    return -101;                                                               \
  }                                                                            \
  ((void (*)(int *, char *, char *, char *, int *, CTYPE *, int *, CTYPE *, int *,\
             RTYPE *, RTYPE *, int *, int *, RTYPE *, int *, RTYPE *, CTYPE *, \
          int *, CTYPE *, int *, RTYPE *, int *, int *, int *))            \
      g_fortran_fn[OP_ID])(&itype_, &jobz_, &range_, &uplo_, &n_, A, &lda_,   \
                           B, &ldb_, &vl_, &vu_, &il_, &iu_, &abstol_, m, w,  \
                           z_ptr, &ldz_, &work_query, &lwork_, rwork, iwork,   \
                           ifail_ptr, &info);                                  \
  if (info != 0) {                                                             \
    free(rwork);                                                               \
    free(iwork);                                                               \
    return info;                                                               \
  }                                                                            \
  lwork_ = QUERY_FN(work_query);                                               \
  work = (CTYPE *)malloc((size_t)lwork_ * sizeof(CTYPE));                      \
  if (!work) {                                                                 \
    free(rwork);                                                               \
    free(iwork);                                                               \
    return -101;                                                               \
  }                                                                            \
  ((void (*)(int *, char *, char *, char *, int *, CTYPE *, int *, CTYPE *, int *,\
             RTYPE *, RTYPE *, int *, int *, RTYPE *, int *, RTYPE *, CTYPE *, \
          int *, CTYPE *, int *, RTYPE *, int *, int *, int *))            \
      g_fortran_fn[OP_ID])(&itype_, &jobz_, &range_, &uplo_, &n_, A, &lda_,   \
                           B, &ldb_, &vl_, &vu_, &il_, &iu_, &abstol_, m, w,  \
                           z_ptr, &ldz_, work, &lwork_, rwork, iwork,          \
                           ifail_ptr, &info);                                  \
  free(work);                                                                  \
  free(rwork);                                                                 \
  free(iwork);                                                                 \
  return info;                                                                 \
}

#define DEFINE_SYGVX_REAL_CBLAS_TO_FORTRAN(OP, TYPE, OP_ID)                    \
static void thunk_##OP##_cblas_to_fortran(                                     \
    int *itype, char *jobz, char *range, char *uplo, int *n, TYPE *A,         \
    int *lda, TYPE *B, int *ldb, TYPE *vl, TYPE *vu, int *il, int *iu,        \
    TYPE *abstol, int *m, TYPE *w, TYPE *Z, int *ldz, TYPE *work, int *lwork, \
    int *iwork, int *ifail, int *info) {                                       \
  fb_uplo_t uplo_enum = FB_UPPER;                                               \
  TYPE dummy_z = (TYPE)0;                                                      \
  TYPE *z_ptr = Z ? Z : &dummy_z;                                              \
  int ldz_value = Z ? *ldz : 1;                                                \
  int dummy_ifail = 0;                                                         \
  int *ifail_ptr = ifail ? ifail : &dummy_ifail;                               \
  (void)work;                                                                   \
  (void)lwork;                                                                  \
  (void)iwork;                                                                  \
  if (fb_lapack_char_to_uplo(*uplo, &uplo_enum) != 0) {                        \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (fb_lapack_job_requests_vectors(*jobz) && Z == NULL) {                    \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (fb_lapack_job_requests_vectors(*jobz) && ifail == NULL) {                \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, int, char, char, fb_uplo_t, int, TYPE *,    \
                     int, TYPE *, int, TYPE, TYPE, int, int, TYPE, int *,     \
                     TYPE *, TYPE *, int, int *))g_cblas_fn[OP_ID])(          \
      FB_LAYOUT_COL_MAJOR, *itype, *jobz, *range, uplo_enum, *n, A, *lda, B,  \
      *ldb, *vl, *vu, *il, *iu, *abstol, m, w, z_ptr, ldz_value, ifail_ptr);   \
  if (info) *info = rc;                                                         \
}

#define DEFINE_HEGVX_COMPLEX_CBLAS_TO_FORTRAN(OP, CTYPE, RTYPE, OP_ID)        \
static void thunk_##OP##_cblas_to_fortran(                                     \
    int *itype, char *jobz, char *range, char *uplo, int *n, CTYPE *A,        \
    int *lda, CTYPE *B, int *ldb, RTYPE *vl, RTYPE *vu, int *il, int *iu,     \
    RTYPE *abstol, int *m, RTYPE *w, CTYPE *Z, int *ldz, CTYPE *work,         \
    int *lwork, RTYPE *rwork, int *iwork, int *ifail, int *info) {            \
  fb_uplo_t uplo_enum = FB_UPPER;                                               \
  CTYPE dummy_z = (CTYPE)0;                                                    \
  CTYPE *z_ptr = Z ? Z : &dummy_z;                                             \
  int ldz_value = Z ? *ldz : 1;                                                \
  int dummy_ifail = 0;                                                         \
  int *ifail_ptr = ifail ? ifail : &dummy_ifail;                               \
  (void)work;                                                                   \
  (void)lwork;                                                                  \
  (void)rwork;                                                                  \
  (void)iwork;                                                                  \
  if (fb_lapack_char_to_uplo(*uplo, &uplo_enum) != 0) {                        \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (fb_lapack_job_requests_vectors(*jobz) && Z == NULL) {                    \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (fb_lapack_job_requests_vectors(*jobz) && ifail == NULL) {                \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, int, char, char, fb_uplo_t, int, CTYPE *,   \
                     int, CTYPE *, int, RTYPE, RTYPE, int, int, RTYPE, int *, \
                     RTYPE *, CTYPE *, int, int *))g_cblas_fn[OP_ID])(        \
      FB_LAYOUT_COL_MAJOR, *itype, *jobz, *range, uplo_enum, *n, A, *lda, B,  \
      *ldb, *vl, *vu, *il, *iu, *abstol, m, w, z_ptr, ldz_value, ifail_ptr);   \
  if (info) *info = rc;                                                         \
}

#define DEFINE_SYGVD_REAL_FORTRAN_TO_CBLAS(OP, TYPE, OP_ID, QUERY_FN)         \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const int itype, char jobz, const fb_uplo_t uplo,\
    const int n, TYPE *A, const int lda, TYPE *B, const int ldb, TYPE *w) {    \
  char jobz_ = jobz;                                                            \
  char uplo_ = fb_lapack_uplo_from_layout(layout, uplo);                        \
  int itype_ = itype, n_ = n, lda_ = lda, ldb_ = ldb;                           \
  int lwork_ = -1, liwork_ = -1, info = 0;                                      \
  TYPE work_query = (TYPE)0;                                                    \
  int iwork_query = 0;                                                          \
  TYPE *work = NULL;                                                            \
  int *iwork = NULL;                                                            \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  ((void (*)(int *, char *, char *, int *, TYPE *, int *, TYPE *, int *,       \
             TYPE *, TYPE *, int *, int *, int *, int *))g_fortran_fn[OP_ID])(\
      &itype_, &jobz_, &uplo_, &n_, A, &lda_, B, &ldb_, w, &work_query,        \
      &lwork_, &iwork_query, &liwork_, &info);                                  \
  if (info != 0) return info;                                                   \
  lwork_ = QUERY_FN(work_query);                                                \
  liwork_ = fb_query_size_from_int(iwork_query);                                \
  work = (TYPE *)malloc((size_t)lwork_ * sizeof(TYPE));                         \
  iwork = (int *)malloc((size_t)liwork_ * sizeof(int));                         \
  if (!work || !iwork) {                                                        \
    free(work);                                                                 \
    free(iwork);                                                                \
    return -101;                                                                \
  }                                                                             \
  ((void (*)(int *, char *, char *, int *, TYPE *, int *, TYPE *, int *,       \
             TYPE *, TYPE *, int *, int *, int *, int *))g_fortran_fn[OP_ID])(\
      &itype_, &jobz_, &uplo_, &n_, A, &lda_, B, &ldb_, w, work, &lwork_,      \
      iwork, &liwork_, &info);                                                  \
  free(work);                                                                   \
  free(iwork);                                                                  \
  return info;                                                                  \
}

#define DEFINE_SYGVD_REAL_CBLAS_TO_FORTRAN(OP, TYPE, OP_ID)                    \
static void thunk_##OP##_cblas_to_fortran(                                     \
    int *itype, char *jobz, char *uplo, int *n, TYPE *A, int *lda, TYPE *B,   \
    int *ldb, TYPE *w, TYPE *work, int *lwork, int *iwork, int *liwork,       \
    int *info) {                                                                \
  fb_uplo_t uplo_enum = FB_UPPER;                                               \
  (void)work;                                                                   \
  (void)lwork;                                                                  \
  (void)iwork;                                                                  \
  (void)liwork;                                                                 \
  if (fb_lapack_char_to_uplo(*uplo, &uplo_enum) != 0) {                        \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, int, char, fb_uplo_t, int, TYPE *, int,     \
                     TYPE *, int, TYPE *))g_cblas_fn[OP_ID])(                 \
      FB_LAYOUT_COL_MAJOR, *itype, *jobz, uplo_enum, *n, A, *lda, B, *ldb, w);\
  if (info) *info = rc;                                                         \
}

#define DEFINE_HEGVD_COMPLEX_CBLAS_TO_FORTRAN(OP, CTYPE, RTYPE, OP_ID)        \
static void thunk_##OP##_cblas_to_fortran(                                     \
    int *itype, char *jobz, char *uplo, int *n, CTYPE *A, int *lda, CTYPE *B, \
    int *ldb, RTYPE *w, CTYPE *work, int *lwork, RTYPE *rwork, int *lrwork,   \
    int *iwork, int *liwork, int *info) {                                       \
  fb_uplo_t uplo_enum = FB_UPPER;                                               \
  (void)work;                                                                   \
  (void)lwork;                                                                  \
  (void)rwork;                                                                  \
  (void)lrwork;                                                                 \
  (void)iwork;                                                                  \
  (void)liwork;                                                                 \
  if (fb_lapack_char_to_uplo(*uplo, &uplo_enum) != 0) {                        \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, int, char, fb_uplo_t, int, CTYPE *, int,    \
                     CTYPE *, int, RTYPE *))g_cblas_fn[OP_ID])(               \
      FB_LAYOUT_COL_MAJOR, *itype, *jobz, uplo_enum, *n, A, *lda, B, *ldb, w);\
  if (info) *info = rc;                                                         \
}

DEFINE_SYEV_REAL_FORTRAN_TO_CBLAS(ssyev, float, FB_OP_SSYEV,
                                  fb_query_size_from_float)
DEFINE_SYEV_REAL_FORTRAN_TO_CBLAS(dsyev, double, FB_OP_DSYEV,
                                  fb_query_size_from_double)
DEFINE_HEEV_COMPLEX_FORTRAN_TO_CBLAS(cheev, fb_complex_float_t, float,
                                     FB_OP_CHEEV, fb_query_size_from_cfloat)
DEFINE_HEEV_COMPLEX_FORTRAN_TO_CBLAS(zheev, fb_complex_double_t, double,
                                     FB_OP_ZHEEV, fb_query_size_from_cdouble)
DEFINE_SYEV_REAL_CBLAS_TO_FORTRAN(ssyev, float, FB_OP_SSYEV)
DEFINE_SYEV_REAL_CBLAS_TO_FORTRAN(dsyev, double, FB_OP_DSYEV)
DEFINE_SYEVR_REAL_FORTRAN_TO_CBLAS(ssyevr, float, FB_OP_SSYEVR,
                                   fb_query_size_from_float)
DEFINE_SYEVR_REAL_FORTRAN_TO_CBLAS(dsyevr, double, FB_OP_DSYEVR,
                                   fb_query_size_from_double)
DEFINE_HEEVR_COMPLEX_FORTRAN_TO_CBLAS(cheevr, fb_complex_float_t, float,
                                      FB_OP_CHEEVR,
                                      fb_query_size_from_cfloat,
                                      fb_query_size_from_float)
DEFINE_HEEVR_COMPLEX_FORTRAN_TO_CBLAS(zheevr, fb_complex_double_t, double,
                                      FB_OP_ZHEEVR,
                                      fb_query_size_from_cdouble,
                                      fb_query_size_from_double)
DEFINE_SYEVR_REAL_CBLAS_TO_FORTRAN(ssyevr, float, FB_OP_SSYEVR)
DEFINE_SYEVR_REAL_CBLAS_TO_FORTRAN(dsyevr, double, FB_OP_DSYEVR)
DEFINE_HEEVR_COMPLEX_CBLAS_TO_FORTRAN(cheevr, fb_complex_float_t, float,
                                      FB_OP_CHEEVR)
DEFINE_HEEVR_COMPLEX_CBLAS_TO_FORTRAN(zheevr, fb_complex_double_t, double,
                                      FB_OP_ZHEEVR)
DEFINE_SYEVX_REAL_FORTRAN_TO_CBLAS(ssyevx, float, FB_OP_SSYEVX,
                                   fb_query_size_from_float)
DEFINE_SYEVX_REAL_FORTRAN_TO_CBLAS(dsyevx, double, FB_OP_DSYEVX,
                                   fb_query_size_from_double)
DEFINE_HEEVX_COMPLEX_FORTRAN_TO_CBLAS(cheevx, fb_complex_float_t, float,
                                      FB_OP_CHEEVX,
                                      fb_query_size_from_cfloat)
DEFINE_HEEVX_COMPLEX_FORTRAN_TO_CBLAS(zheevx, fb_complex_double_t, double,
                                      FB_OP_ZHEEVX,
                                      fb_query_size_from_cdouble)
DEFINE_SYEVX_REAL_CBLAS_TO_FORTRAN(ssyevx, float, FB_OP_SSYEVX)
DEFINE_SYEVX_REAL_CBLAS_TO_FORTRAN(dsyevx, double, FB_OP_DSYEVX)
DEFINE_HEEVX_COMPLEX_CBLAS_TO_FORTRAN(cheevx, fb_complex_float_t, float,
                                      FB_OP_CHEEVX)
DEFINE_HEEVX_COMPLEX_CBLAS_TO_FORTRAN(zheevx, fb_complex_double_t, double,
                                      FB_OP_ZHEEVX)
DEFINE_SYEVD_REAL_CBLAS_TO_FORTRAN(ssyevd, float, FB_OP_SSYEVD)
DEFINE_SYEVD_REAL_CBLAS_TO_FORTRAN(dsyevd, double, FB_OP_DSYEVD)
DEFINE_HEEVD_COMPLEX_CBLAS_TO_FORTRAN(cheevd, fb_complex_float_t, float,
                                      FB_OP_CHEEVD)
DEFINE_HEEVD_COMPLEX_CBLAS_TO_FORTRAN(zheevd, fb_complex_double_t, double,
                                      FB_OP_ZHEEVD)
DEFINE_HEEV_COMPLEX_CBLAS_TO_FORTRAN(cheev, fb_complex_float_t, float,
                                     FB_OP_CHEEV)
DEFINE_HEEV_COMPLEX_CBLAS_TO_FORTRAN(zheev, fb_complex_double_t, double,
                                     FB_OP_ZHEEV)

DEFINE_SYEVD_REAL_FORTRAN_TO_CBLAS(ssyevd, float, FB_OP_SSYEVD,
                                   fb_query_size_from_float)
DEFINE_SYEVD_REAL_FORTRAN_TO_CBLAS(dsyevd, double, FB_OP_DSYEVD,
                                   fb_query_size_from_double)
DEFINE_HEEVD_COMPLEX_FORTRAN_TO_CBLAS(cheevd, fb_complex_float_t, float,
                                      FB_OP_CHEEVD, fb_query_size_from_cfloat)
DEFINE_HEEVD_COMPLEX_FORTRAN_TO_CBLAS(zheevd, fb_complex_double_t, double,
                                      FB_OP_ZHEEVD,
                                      fb_query_size_from_cdouble)
DEFINE_HEGVD_COMPLEX_FORTRAN_TO_CBLAS(chegvd, fb_complex_float_t, float,
                                      FB_OP_CHEGVD,
                                      fb_query_size_from_cfloat)
DEFINE_HEGVD_COMPLEX_FORTRAN_TO_CBLAS(zhegvd, fb_complex_double_t, double,
                                      FB_OP_ZHEGVD,
                                      fb_query_size_from_cdouble)
DEFINE_HEGVD_COMPLEX_FORTRAN_TO_CBLAS(csygvd, fb_complex_float_t, float,
                                      FB_OP_CSYGVD,
                                      fb_query_size_from_cfloat)
DEFINE_HEGVD_COMPLEX_FORTRAN_TO_CBLAS(zsygvd, fb_complex_double_t, double,
                                      FB_OP_ZSYGVD,
                                      fb_query_size_from_cdouble)
DEFINE_SYGVD_REAL_FORTRAN_TO_CBLAS(ssygvd, float, FB_OP_SSYGVD,
                                   fb_query_size_from_float)
DEFINE_SYGVD_REAL_FORTRAN_TO_CBLAS(dsygvd, double, FB_OP_DSYGVD,
                                   fb_query_size_from_double)

DEFINE_LAPACK_LU_SOLVE_THUNKS(sgesv, float, FB_OP_SGESV)
DEFINE_LAPACK_LU_SOLVE_THUNKS(dgesv, double, FB_OP_DGESV)
DEFINE_LAPACK_LU_SOLVE_THUNKS(cgesv, fb_complex_float_t, FB_OP_CGESV)
DEFINE_LAPACK_LU_SOLVE_THUNKS(zgesv, fb_complex_double_t, FB_OP_ZGESV)

DEFINE_LAPACK_LU_FACTOR_THUNKS(sgetrf, float, FB_OP_SGETRF)
DEFINE_LAPACK_LU_FACTOR_THUNKS(dgetrf, double, FB_OP_DGETRF)
DEFINE_LAPACK_LU_FACTOR_THUNKS(cgetrf, fb_complex_float_t, FB_OP_CGETRF)
DEFINE_LAPACK_LU_FACTOR_THUNKS(zgetrf, fb_complex_double_t, FB_OP_ZGETRF)

DEFINE_LAPACK_LU_FACTORED_SOLVE_THUNKS(sgetrs, float, FB_OP_SGETRS)
DEFINE_LAPACK_LU_FACTORED_SOLVE_THUNKS(dgetrs, double, FB_OP_DGETRS)
DEFINE_LAPACK_LU_FACTORED_SOLVE_THUNKS(cgetrs, fb_complex_float_t, FB_OP_CGETRS)
DEFINE_LAPACK_LU_FACTORED_SOLVE_THUNKS(zgetrs, fb_complex_double_t, FB_OP_ZGETRS)

DEFINE_LAPACK_MATRIX_TAU_NOQUERY_THUNKS(sgelq2, float, FB_OP_SGELQ2, m)
DEFINE_LAPACK_MATRIX_TAU_NOQUERY_THUNKS(dgelq2, double, FB_OP_DGELQ2, m)
DEFINE_LAPACK_MATRIX_TAU_NOQUERY_THUNKS(cgelq2, fb_complex_float_t,
                                        FB_OP_CGELQ2, m)
DEFINE_LAPACK_MATRIX_TAU_NOQUERY_THUNKS(zgelq2, fb_complex_double_t,
                                        FB_OP_ZGELQ2, m)

DEFINE_LAPACK_MATRIX_TAU_NOQUERY_THUNKS(sgeql2, float, FB_OP_SGEQL2, n)
DEFINE_LAPACK_MATRIX_TAU_NOQUERY_THUNKS(dgeql2, double, FB_OP_DGEQL2, n)
DEFINE_LAPACK_MATRIX_TAU_NOQUERY_THUNKS(cgeql2, fb_complex_float_t,
                                        FB_OP_CGEQL2, n)
DEFINE_LAPACK_MATRIX_TAU_NOQUERY_THUNKS(zgeql2, fb_complex_double_t,
                                        FB_OP_ZGEQL2, n)

DEFINE_LAPACK_MATRIX_TAU_NOQUERY_THUNKS(sgerq2, float, FB_OP_SGERQ2, m)
DEFINE_LAPACK_MATRIX_TAU_NOQUERY_THUNKS(dgerq2, double, FB_OP_DGERQ2, m)
DEFINE_LAPACK_MATRIX_TAU_NOQUERY_THUNKS(cgerq2, fb_complex_float_t,
                                        FB_OP_CGERQ2, m)
DEFINE_LAPACK_MATRIX_TAU_NOQUERY_THUNKS(zgerq2, fb_complex_double_t,
                                        FB_OP_ZGERQ2, m)

DEFINE_LAPACK_MATRIX_TAU_QUERY_THUNKS(sgeqrf, float, FB_OP_SGEQRF,
                                      fb_query_size_from_float)
DEFINE_LAPACK_MATRIX_TAU_QUERY_THUNKS(dgeqrf, double, FB_OP_DGEQRF,
                                      fb_query_size_from_double)
DEFINE_LAPACK_MATRIX_TAU_QUERY_THUNKS(cgeqrf, fb_complex_float_t,
                                      FB_OP_CGEQRF,
                                      fb_query_size_from_cfloat)
DEFINE_LAPACK_MATRIX_TAU_QUERY_THUNKS(zgeqrf, fb_complex_double_t,
                                      FB_OP_ZGEQRF,
                                      fb_query_size_from_cdouble)

DEFINE_LAPACK_MATRIX_TAU_QUERY_THUNKS(sgelqf, float, FB_OP_SGELQF,
                                      fb_query_size_from_float)
DEFINE_LAPACK_MATRIX_TAU_QUERY_THUNKS(dgelqf, double, FB_OP_DGELQF,
                                      fb_query_size_from_double)
DEFINE_LAPACK_MATRIX_TAU_QUERY_THUNKS(cgelqf, fb_complex_float_t,
                                      FB_OP_CGELQF,
                                      fb_query_size_from_cfloat)
DEFINE_LAPACK_MATRIX_TAU_QUERY_THUNKS(zgelqf, fb_complex_double_t,
                                      FB_OP_ZGELQF,
                                      fb_query_size_from_cdouble)

DEFINE_LAPACK_MATRIX_TAU_QUERY_THUNKS(sgeqlf, float, FB_OP_SGEQLF,
                                      fb_query_size_from_float)
DEFINE_LAPACK_MATRIX_TAU_QUERY_THUNKS(dgeqlf, double, FB_OP_DGEQLF,
                                      fb_query_size_from_double)
DEFINE_LAPACK_MATRIX_TAU_QUERY_THUNKS(cgeqlf, fb_complex_float_t,
                                      FB_OP_CGEQLF,
                                      fb_query_size_from_cfloat)
DEFINE_LAPACK_MATRIX_TAU_QUERY_THUNKS(zgeqlf, fb_complex_double_t,
                                      FB_OP_ZGEQLF,
                                      fb_query_size_from_cdouble)

DEFINE_LAPACK_MATRIX_TAU_QUERY_THUNKS(sgerqf, float, FB_OP_SGERQF,
                                      fb_query_size_from_float)
DEFINE_LAPACK_MATRIX_TAU_QUERY_THUNKS(dgerqf, double, FB_OP_DGERQF,
                                      fb_query_size_from_double)
DEFINE_LAPACK_MATRIX_TAU_QUERY_THUNKS(cgerqf, fb_complex_float_t,
                                      FB_OP_CGERQF,
                                      fb_query_size_from_cfloat)
DEFINE_LAPACK_MATRIX_TAU_QUERY_THUNKS(zgerqf, fb_complex_double_t,
                                      FB_OP_ZGERQF,
                                      fb_query_size_from_cdouble)
DEFINE_LAPACK_MATRIX_TAU_QUERY_THUNKS(stzrzf, float, FB_OP_STZRZF,
                                      fb_query_size_from_float)
DEFINE_LAPACK_MATRIX_TAU_QUERY_THUNKS(dtzrzf, double, FB_OP_DTZRZF,
                                      fb_query_size_from_double)
DEFINE_LAPACK_MATRIX_TAU_QUERY_THUNKS(ctzrzf, fb_complex_float_t,
                                      FB_OP_CTZRZF,
                                      fb_query_size_from_cfloat)
DEFINE_LAPACK_MATRIX_TAU_QUERY_THUNKS(ztzrzf, fb_complex_double_t,
                                      FB_OP_ZTZRZF,
                                      fb_query_size_from_cdouble)

DEFINE_LAPACK_MATRIX_K_TAU_QUERY_THUNKS(sorgqr, float, FB_OP_SORGQR,
                                        fb_query_size_from_float)
DEFINE_LAPACK_MATRIX_K_TAU_QUERY_THUNKS(dorgqr, double, FB_OP_DORGQR,
                                        fb_query_size_from_double)
DEFINE_LAPACK_MATRIX_K_TAU_QUERY_THUNKS(cungqr, fb_complex_float_t,
                                        FB_OP_CUNGQR,
                                        fb_query_size_from_cfloat)
DEFINE_LAPACK_MATRIX_K_TAU_QUERY_THUNKS(zungqr, fb_complex_double_t,
                                        FB_OP_ZUNGQR,
                                        fb_query_size_from_cdouble)

DEFINE_LAPACK_MATRIX_K_TAU_QUERY_THUNKS(sorglq, float, FB_OP_SORGLQ,
                                        fb_query_size_from_float)
DEFINE_LAPACK_MATRIX_K_TAU_QUERY_THUNKS(dorglq, double, FB_OP_DORGLQ,
                                        fb_query_size_from_double)
DEFINE_LAPACK_MATRIX_K_TAU_QUERY_THUNKS(cunglq, fb_complex_float_t,
                                        FB_OP_CUNGLQ,
                                        fb_query_size_from_cfloat)
DEFINE_LAPACK_MATRIX_K_TAU_QUERY_THUNKS(zunglq, fb_complex_double_t,
                                        FB_OP_ZUNGLQ,
                                        fb_query_size_from_cdouble)

DEFINE_LAPACK_MATRIX_K_TAU_QUERY_THUNKS(sorgql, float, FB_OP_SORGQL,
                                        fb_query_size_from_float)
DEFINE_LAPACK_MATRIX_K_TAU_QUERY_THUNKS(dorgql, double, FB_OP_DORGQL,
                                        fb_query_size_from_double)
DEFINE_LAPACK_MATRIX_K_TAU_QUERY_THUNKS(cungql, fb_complex_float_t,
                                        FB_OP_CUNGQL,
                                        fb_query_size_from_cfloat)
DEFINE_LAPACK_MATRIX_K_TAU_QUERY_THUNKS(zungql, fb_complex_double_t,
                                        FB_OP_ZUNGQL,
                                        fb_query_size_from_cdouble)

DEFINE_LAPACK_MATRIX_K_TAU_QUERY_THUNKS(sorgrq, float, FB_OP_SORGRQ,
                                        fb_query_size_from_float)
DEFINE_LAPACK_MATRIX_K_TAU_QUERY_THUNKS(dorgrq, double, FB_OP_DORGRQ,
                                        fb_query_size_from_double)
DEFINE_LAPACK_MATRIX_K_TAU_QUERY_THUNKS(cungrq, fb_complex_float_t,
                                        FB_OP_CUNGRQ,
                                        fb_query_size_from_cfloat)
DEFINE_LAPACK_MATRIX_K_TAU_QUERY_THUNKS(zungrq, fb_complex_double_t,
                                        FB_OP_ZUNGRQ,
                                        fb_query_size_from_cdouble)

DEFINE_LAPACK_APPLY_Q_THUNKS(sormqr, float, FB_OP_SORMQR,
                             fb_query_size_from_float)
DEFINE_LAPACK_APPLY_Q_THUNKS(dormqr, double, FB_OP_DORMQR,
                             fb_query_size_from_double)
DEFINE_LAPACK_APPLY_Q_THUNKS(cunmqr, fb_complex_float_t, FB_OP_CUNMQR,
                             fb_query_size_from_cfloat)
DEFINE_LAPACK_APPLY_Q_THUNKS(zunmqr, fb_complex_double_t, FB_OP_ZUNMQR,
                             fb_query_size_from_cdouble)

DEFINE_LAPACK_APPLY_Q_THUNKS(sormlq, float, FB_OP_SORMLQ,
                             fb_query_size_from_float)
DEFINE_LAPACK_APPLY_Q_THUNKS(dormlq, double, FB_OP_DORMLQ,
                             fb_query_size_from_double)
DEFINE_LAPACK_APPLY_Q_THUNKS(cunmlq, fb_complex_float_t, FB_OP_CUNMLQ,
                             fb_query_size_from_cfloat)
DEFINE_LAPACK_APPLY_Q_THUNKS(zunmlq, fb_complex_double_t, FB_OP_ZUNMLQ,
                             fb_query_size_from_cdouble)

DEFINE_LAPACK_APPLY_Q_THUNKS(sormql, float, FB_OP_SORMQL,
                             fb_query_size_from_float)
DEFINE_LAPACK_APPLY_Q_THUNKS(dormql, double, FB_OP_DORMQL,
                             fb_query_size_from_double)
DEFINE_LAPACK_APPLY_Q_THUNKS(cunmql, fb_complex_float_t, FB_OP_CUNMQL,
                             fb_query_size_from_cfloat)
DEFINE_LAPACK_APPLY_Q_THUNKS(zunmql, fb_complex_double_t, FB_OP_ZUNMQL,
                             fb_query_size_from_cdouble)

DEFINE_LAPACK_APPLY_Q_THUNKS(sormrq, float, FB_OP_SORMRQ,
                             fb_query_size_from_float)
DEFINE_LAPACK_APPLY_Q_THUNKS(dormrq, double, FB_OP_DORMRQ,
                             fb_query_size_from_double)
DEFINE_LAPACK_APPLY_Q_THUNKS(cunmrq, fb_complex_float_t, FB_OP_CUNMRQ,
                             fb_query_size_from_cfloat)
DEFINE_LAPACK_APPLY_Q_THUNKS(zunmrq, fb_complex_double_t, FB_OP_ZUNMRQ,
                             fb_query_size_from_cdouble)

DEFINE_LAPACK_RZ_APPLY_THUNKS(sormr3, float, FB_OP_SORMR3)
DEFINE_LAPACK_RZ_APPLY_THUNKS(dormr3, double, FB_OP_DORMR3)
DEFINE_LAPACK_RZ_APPLY_THUNKS(cunmr3, fb_complex_float_t, FB_OP_CUNMR3)
DEFINE_LAPACK_RZ_APPLY_THUNKS(zunmr3, fb_complex_double_t, FB_OP_ZUNMR3)

DEFINE_LAPACK_RZ_APPLY_QUERY_THUNKS(sormrz, float, FB_OP_SORMRZ,
                                    fb_query_size_from_float)
DEFINE_LAPACK_RZ_APPLY_QUERY_THUNKS(dormrz, double, FB_OP_DORMRZ,
                                    fb_query_size_from_double)
DEFINE_LAPACK_RZ_APPLY_QUERY_THUNKS(cunmrz, fb_complex_float_t,
                                    FB_OP_CUNMRZ,
                                    fb_query_size_from_cfloat)
DEFINE_LAPACK_RZ_APPLY_QUERY_THUNKS(zunmrz, fb_complex_double_t,
                                    FB_OP_ZUNMRZ,
                                    fb_query_size_from_cdouble)

DEFINE_LAPACK_BIDIAG_GENERATOR_THUNKS(sorgbr, float, FB_OP_SORGBR,
                                      fb_query_size_from_float)
DEFINE_LAPACK_BIDIAG_GENERATOR_THUNKS(dorgbr, double, FB_OP_DORGBR,
                                      fb_query_size_from_double)
DEFINE_LAPACK_BIDIAG_GENERATOR_THUNKS(cungbr, fb_complex_float_t,
                                      FB_OP_CUNGBR,
                                      fb_query_size_from_cfloat)
DEFINE_LAPACK_BIDIAG_GENERATOR_THUNKS(zungbr, fb_complex_double_t,
                                      FB_OP_ZUNGBR,
                                      fb_query_size_from_cdouble)

DEFINE_LAPACK_BIDIAG_APPLY_THUNKS(sormbr, float, FB_OP_SORMBR,
                                  fb_query_size_from_float)
DEFINE_LAPACK_BIDIAG_APPLY_THUNKS(dormbr, double, FB_OP_DORMBR,
                                  fb_query_size_from_double)
DEFINE_LAPACK_BIDIAG_APPLY_THUNKS(cunmbr, fb_complex_float_t,
                                  FB_OP_CUNMBR,
                                  fb_query_size_from_cfloat)
DEFINE_LAPACK_BIDIAG_APPLY_THUNKS(zunmbr, fb_complex_double_t,
                                  FB_OP_ZUNMBR,
                                  fb_query_size_from_cdouble)

DEFINE_LAPACK_UPLO_GENERAL_RHS_THUNKS(sposv, float, FB_OP_SPOSV)
DEFINE_LAPACK_UPLO_GENERAL_RHS_THUNKS(dposv, double, FB_OP_DPOSV)
DEFINE_LAPACK_UPLO_GENERAL_RHS_THUNKS(cposv, fb_complex_float_t, FB_OP_CPOSV)
DEFINE_LAPACK_UPLO_GENERAL_RHS_THUNKS(zposv, fb_complex_double_t, FB_OP_ZPOSV)

DEFINE_LAPACK_UPLO_PIVOTED_QUERY_SOLVE_THUNKS(ssysv, float, FB_OP_SSYSV,
                                              fb_query_size_from_float)
DEFINE_LAPACK_UPLO_PIVOTED_QUERY_SOLVE_THUNKS(dsysv, double, FB_OP_DSYSV,
                                              fb_query_size_from_double)
DEFINE_LAPACK_UPLO_PIVOTED_QUERY_SOLVE_THUNKS(chesv, fb_complex_float_t,
                                              FB_OP_CHESV,
                                              fb_query_size_from_cfloat)
DEFINE_LAPACK_UPLO_PIVOTED_QUERY_SOLVE_THUNKS(zhesv, fb_complex_double_t,
                                              FB_OP_ZHESV,
                                              fb_query_size_from_cdouble)

DEFINE_LAPACK_UPLO_INFO_MATRIX_THUNKS(spotrf, float, FB_OP_SPOTRF)
DEFINE_LAPACK_UPLO_INFO_MATRIX_THUNKS(dpotrf, double, FB_OP_DPOTRF)
DEFINE_LAPACK_UPLO_INFO_MATRIX_THUNKS(cpotrf, fb_complex_float_t, FB_OP_CPOTRF)
DEFINE_LAPACK_UPLO_INFO_MATRIX_THUNKS(zpotrf, fb_complex_double_t, FB_OP_ZPOTRF)

DEFINE_LAPACK_UPLO_INFO_MATRIX_THUNKS(spotri, float, FB_OP_SPOTRI)
DEFINE_LAPACK_UPLO_INFO_MATRIX_THUNKS(dpotri, double, FB_OP_DPOTRI)
DEFINE_LAPACK_UPLO_INFO_MATRIX_THUNKS(cpotri, fb_complex_float_t, FB_OP_CPOTRI)
DEFINE_LAPACK_UPLO_INFO_MATRIX_THUNKS(zpotri, fb_complex_double_t, FB_OP_ZPOTRI)
DEFINE_LAPACK_UPLO_INFO_MATRIX_THUNKS(slauu2, float, FB_OP_SLAUU2)
DEFINE_LAPACK_UPLO_INFO_MATRIX_THUNKS(dlauu2, double, FB_OP_DLAUU2)
DEFINE_LAPACK_UPLO_INFO_MATRIX_THUNKS(clauu2, fb_complex_float_t, FB_OP_CLAUU2)
DEFINE_LAPACK_UPLO_INFO_MATRIX_THUNKS(zlauu2, fb_complex_double_t, FB_OP_ZLAUU2)
DEFINE_LAPACK_UPLO_INFO_MATRIX_THUNKS(slauum, float, FB_OP_SLAUUM)
DEFINE_LAPACK_UPLO_INFO_MATRIX_THUNKS(dlauum, double, FB_OP_DLAUUM)
DEFINE_LAPACK_UPLO_INFO_MATRIX_THUNKS(clauum, fb_complex_float_t, FB_OP_CLAUUM)
DEFINE_LAPACK_UPLO_INFO_MATRIX_THUNKS(zlauum, fb_complex_double_t, FB_OP_ZLAUUM)
DEFINE_LAPACK_LA_GBAMV_REAL_THUNKS(sla_gbamv, float, FB_OP_SLA_GBAMV)
DEFINE_LAPACK_LA_GBAMV_REAL_THUNKS(dla_gbamv, double, FB_OP_DLA_GBAMV)
DEFINE_LAPACK_LA_GBAMV_COMPLEX_THUNKS(cla_gbamv, fb_complex_float_t, float,
                                      FB_OP_CLA_GBAMV)
DEFINE_LAPACK_LA_GBAMV_COMPLEX_THUNKS(zla_gbamv, fb_complex_double_t, double,
                                      FB_OP_ZLA_GBAMV)

DEFINE_LAPACK_UPLO_GENERAL_RHS_THUNKS(spotrs, float, FB_OP_SPOTRS)
DEFINE_LAPACK_UPLO_GENERAL_RHS_THUNKS(dpotrs, double, FB_OP_DPOTRS)
DEFINE_LAPACK_UPLO_GENERAL_RHS_THUNKS(cpotrs, fb_complex_float_t, FB_OP_CPOTRS)
DEFINE_LAPACK_UPLO_GENERAL_RHS_THUNKS(zpotrs, fb_complex_double_t, FB_OP_ZPOTRS)

DEFINE_LAPACK_PB_EQU_THUNKS(spbequ, float, float, FB_OP_SPBEQU)
DEFINE_LAPACK_PB_EQU_THUNKS(dpbequ, double, double, FB_OP_DPBEQU)
DEFINE_LAPACK_PB_EQU_THUNKS(cpbequ, fb_complex_float_t, float, FB_OP_CPBEQU)
DEFINE_LAPACK_PB_EQU_THUNKS(zpbequ, fb_complex_double_t, double, FB_OP_ZPBEQU)
DEFINE_LAPACK_PP_EQU_THUNKS(sppequ, float, float, FB_OP_SPPEQU)
DEFINE_LAPACK_PP_EQU_THUNKS(dppequ, double, double, FB_OP_DPPEQU)
DEFINE_LAPACK_PP_EQU_THUNKS(cppequ, fb_complex_float_t, float, FB_OP_CPPEQU)
DEFINE_LAPACK_PP_EQU_THUNKS(zppequ, fb_complex_double_t, double, FB_OP_ZPPEQU)

DEFINE_LAPACK_GB_EQU_THUNKS(sgbequ, float, float, FB_OP_SGBEQU)
DEFINE_LAPACK_GB_EQU_THUNKS(dgbequ, double, double, FB_OP_DGBEQU)
DEFINE_LAPACK_GB_EQU_THUNKS(cgbequ, fb_complex_float_t, float, FB_OP_CGBEQU)
DEFINE_LAPACK_GB_EQU_THUNKS(zgbequ, fb_complex_double_t, double, FB_OP_ZGBEQU)

DEFINE_LAPACK_GB_FACTOR_THUNKS(sgbtrf, float, FB_OP_SGBTRF)
DEFINE_LAPACK_GB_FACTOR_THUNKS(dgbtrf, double, FB_OP_DGBTRF)
DEFINE_LAPACK_GB_FACTOR_THUNKS(cgbtrf, fb_complex_float_t, FB_OP_CGBTRF)
DEFINE_LAPACK_GB_FACTOR_THUNKS(zgbtrf, fb_complex_double_t, FB_OP_ZGBTRF)

DEFINE_LAPACK_GB_FACTORED_SOLVE_THUNKS(sgbtrs, float, FB_OP_SGBTRS)
DEFINE_LAPACK_GB_FACTORED_SOLVE_THUNKS(dgbtrs, double, FB_OP_DGBTRS)
DEFINE_LAPACK_GB_FACTORED_SOLVE_THUNKS(cgbtrs, fb_complex_float_t,
                                       FB_OP_CGBTRS)
DEFINE_LAPACK_GB_FACTORED_SOLVE_THUNKS(zgbtrs, fb_complex_double_t,
                                       FB_OP_ZGBTRS)

DEFINE_LAPACK_GB_DIRECT_SOLVE_THUNKS(sgbsv, float, FB_OP_SGBSV)
DEFINE_LAPACK_GB_DIRECT_SOLVE_THUNKS(dgbsv, double, FB_OP_DGBSV)
DEFINE_LAPACK_GB_DIRECT_SOLVE_THUNKS(cgbsv, fb_complex_float_t, FB_OP_CGBSV)
DEFINE_LAPACK_GB_DIRECT_SOLVE_THUNKS(zgbsv, fb_complex_double_t, FB_OP_ZGBSV)

DEFINE_LAPACK_PB_CON_REAL_THUNKS(spbcon, float, FB_OP_SPBCON)
DEFINE_LAPACK_PB_CON_REAL_THUNKS(dpbcon, double, FB_OP_DPBCON)
DEFINE_LAPACK_PB_CON_COMPLEX_THUNKS(cpbcon, fb_complex_float_t, float,
                                    FB_OP_CPBCON)
DEFINE_LAPACK_PB_CON_COMPLEX_THUNKS(zpbcon, fb_complex_double_t, double,
                                    FB_OP_ZPBCON)

DEFINE_LAPACK_GB_CON_REAL_THUNKS(sgbcon, float, FB_OP_SGBCON)
DEFINE_LAPACK_GB_CON_REAL_THUNKS(dgbcon, double, FB_OP_DGBCON)
DEFINE_LAPACK_GB_CON_COMPLEX_THUNKS(cgbcon, fb_complex_float_t, float,
                                    FB_OP_CGBCON)
DEFINE_LAPACK_GB_CON_COMPLEX_THUNKS(zgbcon, fb_complex_double_t, double,
                                    FB_OP_ZGBCON)
DEFINE_LAPACK_PP_CON_REAL_THUNKS(sppcon, float, FB_OP_SPPCON)
DEFINE_LAPACK_PP_CON_REAL_THUNKS(dppcon, double, FB_OP_DPPCON)
DEFINE_LAPACK_PP_CON_COMPLEX_THUNKS(cppcon, fb_complex_float_t, float,
                                    FB_OP_CPPCON)
DEFINE_LAPACK_PP_CON_COMPLEX_THUNKS(zppcon, fb_complex_double_t, double,
                                    FB_OP_ZPPCON)
DEFINE_LAPACK_SP_CON_REAL_THUNKS(sspcon, float, FB_OP_SSPCON)
DEFINE_LAPACK_SP_CON_REAL_THUNKS(dspcon, double, FB_OP_DSPCON)
DEFINE_LAPACK_SP_CON_COMPLEX_THUNKS(cspcon, fb_complex_float_t, float,
                                    FB_OP_CSPCON)
DEFINE_LAPACK_SP_CON_COMPLEX_THUNKS(zspcon, fb_complex_double_t, double,
                                    FB_OP_ZSPCON)
DEFINE_LAPACK_HP_CON_COMPLEX_THUNKS(chpcon, fb_complex_float_t, float,
                                    FB_OP_CHPCON)
DEFINE_LAPACK_HP_CON_COMPLEX_THUNKS(zhpcon, fb_complex_double_t, double,
                                    FB_OP_ZHPCON)

DEFINE_LAPACK_PB_SOLVE_THUNKS(spbsv, float, FB_OP_SPBSV)
DEFINE_LAPACK_PB_SOLVE_THUNKS(dpbsv, double, FB_OP_DPBSV)
DEFINE_LAPACK_PB_SOLVE_THUNKS(cpbsv, fb_complex_float_t, FB_OP_CPBSV)
DEFINE_LAPACK_PB_SOLVE_THUNKS(zpbsv, fb_complex_double_t, FB_OP_ZPBSV)
DEFINE_LAPACK_PP_SOLVE_THUNKS(sppsv, float, FB_OP_SPPSV)
DEFINE_LAPACK_PP_SOLVE_THUNKS(dppsv, double, FB_OP_DPPSV)
DEFINE_LAPACK_PP_SOLVE_THUNKS(cppsv, fb_complex_float_t, FB_OP_CPPSV)
DEFINE_LAPACK_PP_SOLVE_THUNKS(zppsv, fb_complex_double_t, FB_OP_ZPPSV)
DEFINE_LAPACK_HP_SOLVE_THUNKS(chpsv, fb_complex_float_t, FB_OP_CHPSV)
DEFINE_LAPACK_HP_SOLVE_THUNKS(zhpsv, fb_complex_double_t, FB_OP_ZHPSV)
DEFINE_LAPACK_PP_SVX_REAL_THUNKS(sppsvx, float, FB_OP_SPPSVX)
DEFINE_LAPACK_PP_SVX_REAL_THUNKS(dppsvx, double, FB_OP_DPPSVX)
DEFINE_LAPACK_PP_SVX_COMPLEX_THUNKS(cppsvx, fb_complex_float_t, float,
                                    FB_OP_CPPSVX)
DEFINE_LAPACK_PP_SVX_COMPLEX_THUNKS(zppsvx, fb_complex_double_t, double,
                                    FB_OP_ZPPSVX)
DEFINE_LAPACK_PP_FACTOR_THUNKS(spptrf, float, FB_OP_SPPTRF)
DEFINE_LAPACK_PP_FACTOR_THUNKS(dpptrf, double, FB_OP_DPPTRF)
DEFINE_LAPACK_PP_FACTOR_THUNKS(cpptrf, fb_complex_float_t, FB_OP_CPPTRF)
DEFINE_LAPACK_PP_FACTOR_THUNKS(zpptrf, fb_complex_double_t, FB_OP_ZPPTRF)
DEFINE_LAPACK_PP_FACTOR_THUNKS(spptri, float, FB_OP_SPPTRI)
DEFINE_LAPACK_PP_FACTOR_THUNKS(dpptri, double, FB_OP_DPPTRI)
DEFINE_LAPACK_PP_FACTOR_THUNKS(cpptri, fb_complex_float_t, FB_OP_CPPTRI)
DEFINE_LAPACK_PP_FACTOR_THUNKS(zpptri, fb_complex_double_t, FB_OP_ZPPTRI)
DEFINE_LAPACK_PP_FACTOR_THUNKS(chptrf, fb_complex_float_t, FB_OP_CHPTRF)
DEFINE_LAPACK_PP_FACTOR_THUNKS(zhptrf, fb_complex_double_t, FB_OP_ZHPTRF)
DEFINE_LAPACK_PP_FACTOR_THUNKS(chptri, fb_complex_float_t, FB_OP_CHPTRI)
DEFINE_LAPACK_PP_FACTOR_THUNKS(zhptri, fb_complex_double_t, FB_OP_ZHPTRI)
DEFINE_LAPACK_PP_FACTORED_SOLVE_THUNKS(spptrs, float, FB_OP_SPPTRS)
DEFINE_LAPACK_PP_FACTORED_SOLVE_THUNKS(dpptrs, double, FB_OP_DPPTRS)
DEFINE_LAPACK_PP_FACTORED_SOLVE_THUNKS(cpptrs, fb_complex_float_t, FB_OP_CPPTRS)
DEFINE_LAPACK_PP_FACTORED_SOLVE_THUNKS(zpptrs, fb_complex_double_t, FB_OP_ZPPTRS)
DEFINE_LAPACK_HP_FACTORED_SOLVE_THUNKS(chptrs, fb_complex_float_t, FB_OP_CHPTRS)
DEFINE_LAPACK_HP_FACTORED_SOLVE_THUNKS(zhptrs, fb_complex_double_t, FB_OP_ZHPTRS)
DEFINE_LAPACK_PT_CON_REAL_THUNKS(sptcon, float, FB_OP_SPTCON)
DEFINE_LAPACK_PT_CON_REAL_THUNKS(dptcon, double, FB_OP_DPTCON)
DEFINE_LAPACK_PT_CON_COMPLEX_THUNKS(cptcon, fb_complex_float_t, float,
                                    FB_OP_CPTCON)
DEFINE_LAPACK_PT_CON_COMPLEX_THUNKS(zptcon, fb_complex_double_t, double,
                                    FB_OP_ZPTCON)
DEFINE_LAPACK_PTRFS_REAL_THUNKS(sptrfs, float, FB_OP_SPTRFS)
DEFINE_LAPACK_PTRFS_REAL_THUNKS(dptrfs, double, FB_OP_DPTRFS)
DEFINE_LAPACK_PTRFS_COMPLEX_THUNKS(cptrfs, fb_complex_float_t, float,
                                   FB_OP_CPTRFS)
DEFINE_LAPACK_PTRFS_COMPLEX_THUNKS(zptrfs, fb_complex_double_t, double,
                                   FB_OP_ZPTRFS)
DEFINE_LAPACK_PT_SOLVE_REAL_THUNKS(sptsv, float, FB_OP_SPTSV)
DEFINE_LAPACK_PT_SOLVE_REAL_THUNKS(dptsv, double, FB_OP_DPTSV)
DEFINE_LAPACK_PT_SOLVE_COMPLEX_THUNKS(cptsv, fb_complex_float_t, float,
                                      FB_OP_CPTSV)
DEFINE_LAPACK_PT_SOLVE_COMPLEX_THUNKS(zptsv, fb_complex_double_t, double,
                                      FB_OP_ZPTSV)
DEFINE_LAPACK_PT_SVX_REAL_THUNKS(sptsvx, float, FB_OP_SPTSVX)
DEFINE_LAPACK_PT_SVX_REAL_THUNKS(dptsvx, double, FB_OP_DPTSVX)
DEFINE_LAPACK_PT_SVX_COMPLEX_THUNKS(cptsvx, fb_complex_float_t, float,
                                    FB_OP_CPTSVX)
DEFINE_LAPACK_PT_SVX_COMPLEX_THUNKS(zptsvx, fb_complex_double_t, double,
                                    FB_OP_ZPTSVX)
DEFINE_LAPACK_PT_FACTOR_REAL_THUNKS(spttrf, float, FB_OP_SPTTRF)
DEFINE_LAPACK_PT_FACTOR_REAL_THUNKS(dpttrf, double, FB_OP_DPTTRF)
DEFINE_LAPACK_PT_FACTOR_COMPLEX_THUNKS(cpttrf, fb_complex_float_t, float,
                                       FB_OP_CPTTRF)
DEFINE_LAPACK_PT_FACTOR_COMPLEX_THUNKS(zpttrf, fb_complex_double_t, double,
                                       FB_OP_ZPTTRF)
DEFINE_LAPACK_PT_FACTORED_SOLVE_REAL_THUNKS(spttrs, float, FB_OP_SPTTRS)
DEFINE_LAPACK_PT_FACTORED_SOLVE_REAL_THUNKS(dpttrs, double, FB_OP_DPTTRS)
DEFINE_LAPACK_PT_FACTORED_SOLVE_COMPLEX_THUNKS(cpttrs, fb_complex_float_t,
                                               float, FB_OP_CPTTRS)
DEFINE_LAPACK_PT_FACTORED_SOLVE_COMPLEX_THUNKS(zpttrs, fb_complex_double_t,
                                               double, FB_OP_ZPTTRS)
DEFINE_COMPLEX_ROTG_THUNKS(crotg, fb_complex_float_t, float, FB_OP_CROTG)
DEFINE_COMPLEX_ROTG_THUNKS(zrotg, fb_complex_double_t, double, FB_OP_ZROTG)
DEFINE_RSCL_REAL_THUNKS(srscl, float, FB_OP_SRSCL)
DEFINE_RSCL_REAL_THUNKS(drscl, double, FB_OP_DRSCL)
DEFINE_RSCL_COMPLEX_THUNKS(csrscl, fb_complex_float_t, float, FB_OP_CSRSCL)
DEFINE_RSCL_COMPLEX_THUNKS(zdrscl, fb_complex_double_t, double, FB_OP_ZDRSCL)
DEFINE_SPEV_REAL_THUNKS(sspev, float, FB_OP_SSPEV)
DEFINE_SPEV_REAL_THUNKS(dspev, double, FB_OP_DSPEV)
DEFINE_SPEVD_REAL_THUNKS(sspevd, float, FB_OP_SSPEVD, fb_query_size_from_float)
DEFINE_SPEVD_REAL_THUNKS(dspevd, double, FB_OP_DSPEVD, fb_query_size_from_double)
DEFINE_SPEVX_REAL_THUNKS(sspevx, float, FB_OP_SSPEVX)
DEFINE_SPEVX_REAL_THUNKS(dspevx, double, FB_OP_DSPEVX)
DEFINE_SPGV_REAL_THUNKS(sspgv, float, FB_OP_SSPGV)
DEFINE_SPGV_REAL_THUNKS(dspgv, double, FB_OP_DSPGV)
DEFINE_SPGVD_REAL_THUNKS(sspgvd, float, FB_OP_SSPGVD, fb_query_size_from_float)
DEFINE_SPGVD_REAL_THUNKS(dspgvd, double, FB_OP_DSPGVD, fb_query_size_from_double)
DEFINE_SPGVX_REAL_THUNKS(sspgvx, float, FB_OP_SSPGVX)
DEFINE_SPGVX_REAL_THUNKS(dspgvx, double, FB_OP_DSPGVX)
DEFINE_HPEV_COMPLEX_THUNKS(chpev, fb_complex_float_t, float, FB_OP_CHPEV)
DEFINE_HPEV_COMPLEX_THUNKS(zhpev, fb_complex_double_t, double, FB_OP_ZHPEV)
DEFINE_HPEVD_COMPLEX_THUNKS(chpevd, fb_complex_float_t, float, FB_OP_CHPEVD,
                            fb_query_size_from_cfloat)
DEFINE_HPEVD_COMPLEX_THUNKS(zhpevd, fb_complex_double_t, double, FB_OP_ZHPEVD,
                            fb_query_size_from_cdouble)
DEFINE_HPEVX_COMPLEX_THUNKS(chpevx, fb_complex_float_t, float, FB_OP_CHPEVX)
DEFINE_HPEVX_COMPLEX_THUNKS(zhpevx, fb_complex_double_t, double, FB_OP_ZHPEVX)
DEFINE_HPGV_COMPLEX_THUNKS(chpgv, fb_complex_float_t, float, FB_OP_CHPGV)
DEFINE_HPGV_COMPLEX_THUNKS(zhpgv, fb_complex_double_t, double, FB_OP_ZHPGV)
DEFINE_HPGVD_COMPLEX_THUNKS(chpgvd, fb_complex_float_t, float, FB_OP_CHPGVD,
                            fb_query_size_from_cfloat)
DEFINE_HPGVD_COMPLEX_THUNKS(zhpgvd, fb_complex_double_t, double, FB_OP_ZHPGVD,
                            fb_query_size_from_cdouble)
DEFINE_HPGVX_COMPLEX_THUNKS(chpgvx, fb_complex_float_t, float, FB_OP_CHPGVX)
DEFINE_HPGVX_COMPLEX_THUNKS(zhpgvx, fb_complex_double_t, double, FB_OP_ZHPGVX)
DEFINE_SBEV_REAL_THUNKS(ssbev, float, FB_OP_SSBEV)
DEFINE_SBEV_REAL_THUNKS(dsbev, double, FB_OP_DSBEV)
DEFINE_HBEV_COMPLEX_THUNKS(chbev, fb_complex_float_t, float, FB_OP_CHBEV)
DEFINE_HBEV_COMPLEX_THUNKS(zhbev, fb_complex_double_t, double, FB_OP_ZHBEV)
DEFINE_HBEVD_COMPLEX_THUNKS(chbevd, fb_complex_float_t, float, FB_OP_CHBEVD,
                            fb_query_size_from_cfloat)
DEFINE_HBEVD_COMPLEX_THUNKS(zhbevd, fb_complex_double_t, double, FB_OP_ZHBEVD,
                            fb_query_size_from_cdouble)
DEFINE_SBEVD_REAL_THUNKS(ssbevd, float, FB_OP_SSBEVD)
DEFINE_SBEVD_REAL_THUNKS(dsbevd, double, FB_OP_DSBEVD)
DEFINE_SBEVX_REAL_THUNKS(ssbevx, float, FB_OP_SSBEVX)
DEFINE_SBEVX_REAL_THUNKS(dsbevx, double, FB_OP_DSBEVX)
DEFINE_SBGV_REAL_THUNKS(ssbgv, float, FB_OP_SSBGV)
DEFINE_SBGV_REAL_THUNKS(dsbgv, double, FB_OP_DSBGV)
DEFINE_HBGV_COMPLEX_THUNKS(chbgv, fb_complex_float_t, float, FB_OP_CHBGV)
DEFINE_HBGV_COMPLEX_THUNKS(zhbgv, fb_complex_double_t, double, FB_OP_ZHBGV)
DEFINE_SBGVD_REAL_THUNKS(ssbgvd, float, FB_OP_SSBGVD)
DEFINE_SBGVD_REAL_THUNKS(dsbgvd, double, FB_OP_DSBGVD)
DEFINE_HBGVD_COMPLEX_THUNKS(chbgvd, fb_complex_float_t, float, FB_OP_CHBGVD,
                            fb_query_size_from_cfloat)
DEFINE_HBGVD_COMPLEX_THUNKS(zhbgvd, fb_complex_double_t, double, FB_OP_ZHBGVD,
                            fb_query_size_from_cdouble)
DEFINE_SBGVX_REAL_THUNKS(ssbgvx, float, FB_OP_SSBGVX)
DEFINE_SBGVX_REAL_THUNKS(dsbgvx, double, FB_OP_DSBGVX)
DEFINE_HBGVX_COMPLEX_THUNKS(chbgvx, fb_complex_float_t, float, FB_OP_CHBGVX)
DEFINE_HBGVX_COMPLEX_THUNKS(zhbgvx, fb_complex_double_t, double, FB_OP_ZHBGVX)

DEFINE_LAPACK_PB_SVX_REAL_THUNKS(spbsvx, float, FB_OP_SPBSVX)
DEFINE_LAPACK_PB_SVX_REAL_THUNKS(dpbsvx, double, FB_OP_DPBSVX)
DEFINE_LAPACK_PB_SVX_COMPLEX_THUNKS(cpbsvx, fb_complex_float_t, float,
                                    FB_OP_CPBSVX)
DEFINE_LAPACK_PB_SVX_COMPLEX_THUNKS(zpbsvx, fb_complex_double_t, double,
                                    FB_OP_ZPBSVX)

DEFINE_LAPACK_GB_SVX_REAL_THUNKS(sgbsvx, float, FB_OP_SGBSVX)
DEFINE_LAPACK_GB_SVX_REAL_THUNKS(dgbsvx, double, FB_OP_DGBSVX)
DEFINE_LAPACK_GB_SVX_COMPLEX_THUNKS(cgbsvx, fb_complex_float_t, float,
                                    FB_OP_CGBSVX)
DEFINE_LAPACK_GB_SVX_COMPLEX_THUNKS(zgbsvx, fb_complex_double_t, double,
                                    FB_OP_ZGBSVX)
DEFINE_LAPACK_GB_REFINE_REAL_THUNKS(sgbrfs, float, FB_OP_SGBRFS)
DEFINE_LAPACK_GB_REFINE_REAL_THUNKS(dgbrfs, double, FB_OP_DGBRFS)
DEFINE_LAPACK_GB_REFINE_COMPLEX_THUNKS(cgbrfs, fb_complex_float_t, float,
                                       FB_OP_CGBRFS)
DEFINE_LAPACK_GB_REFINE_COMPLEX_THUNKS(zgbrfs, fb_complex_double_t, double,
                                       FB_OP_ZGBRFS)

DEFINE_LAPACK_PB_FACTOR_THUNKS(spbtrf, float, FB_OP_SPBTRF)
DEFINE_LAPACK_PB_FACTOR_THUNKS(dpbtrf, double, FB_OP_DPBTRF)
DEFINE_LAPACK_PB_FACTOR_THUNKS(ssbtrf, float, FB_OP_SSBTRF)
DEFINE_LAPACK_PB_FACTOR_THUNKS(dsbtrf, double, FB_OP_DSBTRF)
DEFINE_LAPACK_PB_FACTOR_THUNKS(cpbtrf, fb_complex_float_t, FB_OP_CPBTRF)
DEFINE_LAPACK_PB_FACTOR_THUNKS(zpbtrf, fb_complex_double_t, FB_OP_ZPBTRF)

DEFINE_LAPACK_PB_FACTORED_SOLVE_THUNKS(spbtrs, float, FB_OP_SPBTRS)
DEFINE_LAPACK_PB_FACTORED_SOLVE_THUNKS(dpbtrs, double, FB_OP_DPBTRS)
DEFINE_LAPACK_PB_FACTORED_SOLVE_THUNKS(cpbtrs, fb_complex_float_t, FB_OP_CPBTRS)
DEFINE_LAPACK_PB_FACTORED_SOLVE_THUNKS(zpbtrs, fb_complex_double_t, FB_OP_ZPBTRS)

DEFINE_LAPACK_PB_REFINE_REAL_THUNKS(spbrfs, float, FB_OP_SPBRFS)
DEFINE_LAPACK_PB_REFINE_REAL_THUNKS(dpbrfs, double, FB_OP_DPBRFS)
DEFINE_LAPACK_PB_REFINE_COMPLEX_THUNKS(cpbrfs, fb_complex_float_t, float,
                                       FB_OP_CPBRFS)
DEFINE_LAPACK_PB_REFINE_COMPLEX_THUNKS(zpbrfs, fb_complex_double_t, double,
                                       FB_OP_ZPBRFS)
DEFINE_LAPACK_PP_REFINE_REAL_THUNKS(spprfs, float, FB_OP_SPPRFS)
DEFINE_LAPACK_PP_REFINE_REAL_THUNKS(dpprfs, double, FB_OP_DPPRFS)
DEFINE_LAPACK_PP_REFINE_COMPLEX_THUNKS(cpprfs, fb_complex_float_t, float,
                                       FB_OP_CPPRFS)
DEFINE_LAPACK_PP_REFINE_COMPLEX_THUNKS(zpprfs, fb_complex_double_t, double,
                                       FB_OP_ZPPRFS)
DEFINE_LAPACK_HP_REFINE_COMPLEX_THUNKS(chprfs, fb_complex_float_t, float,
                                       FB_OP_CHPRFS)
DEFINE_LAPACK_HP_REFINE_COMPLEX_THUNKS(zhprfs, fb_complex_double_t, double,
                                       FB_OP_ZHPRFS)

DEFINE_GELS_REAL_FORTRAN_TO_CBLAS(sgels, float, FB_OP_SGELS,
                                  fb_query_size_from_float)
DEFINE_GELS_REAL_FORTRAN_TO_CBLAS(dgels, double, FB_OP_DGELS,
                                  fb_query_size_from_double)
DEFINE_GELS_COMPLEX_FORTRAN_TO_CBLAS(cgels, fb_complex_float_t, FB_OP_CGELS,
                                     fb_query_size_from_cfloat)
DEFINE_GELS_COMPLEX_FORTRAN_TO_CBLAS(zgels, fb_complex_double_t, FB_OP_ZGELS,
                                     fb_query_size_from_cdouble)
DEFINE_GELS_REAL_CBLAS_TO_FORTRAN(sgels, float, FB_OP_SGELS)
DEFINE_GELS_REAL_CBLAS_TO_FORTRAN(dgels, double, FB_OP_DGELS)
DEFINE_GELS_COMPLEX_CBLAS_TO_FORTRAN(cgels, fb_complex_float_t, FB_OP_CGELS)
DEFINE_GELS_COMPLEX_CBLAS_TO_FORTRAN(zgels, fb_complex_double_t, FB_OP_ZGELS)

DEFINE_GESVD_REAL_FORTRAN_TO_CBLAS(sgesvd, float, FB_OP_SGESVD,
                                   fb_query_size_from_float)
DEFINE_GESVD_REAL_FORTRAN_TO_CBLAS(dgesvd, double, FB_OP_DGESVD,
                                   fb_query_size_from_double)
DEFINE_GESVD_COMPLEX_FORTRAN_TO_CBLAS(cgesvd, fb_complex_float_t, float,
                                      FB_OP_CGESVD,
                                      fb_query_size_from_cfloat)
DEFINE_GESVD_COMPLEX_FORTRAN_TO_CBLAS(zgesvd, fb_complex_double_t, double,
                                      FB_OP_ZGESVD,
                                      fb_query_size_from_cdouble)
DEFINE_GESVD_REAL_CBLAS_TO_FORTRAN(sgesvd, float, FB_OP_SGESVD)
DEFINE_GESVD_REAL_CBLAS_TO_FORTRAN(dgesvd, double, FB_OP_DGESVD)
DEFINE_GESVD_COMPLEX_CBLAS_TO_FORTRAN(cgesvd, fb_complex_float_t, float,
                                      FB_OP_CGESVD)
DEFINE_GESVD_COMPLEX_CBLAS_TO_FORTRAN(zgesvd, fb_complex_double_t, double,
                                      FB_OP_ZGESVD)

DEFINE_GEBD2_REAL_FORTRAN_TO_CBLAS(sgebd2, float, FB_OP_SGEBD2)
DEFINE_GEBD2_REAL_FORTRAN_TO_CBLAS(dgebd2, double, FB_OP_DGEBD2)
DEFINE_GEBD2_COMPLEX_FORTRAN_TO_CBLAS(cgebd2, fb_complex_float_t, float,
                                      FB_OP_CGEBD2)
DEFINE_GEBD2_COMPLEX_FORTRAN_TO_CBLAS(zgebd2, fb_complex_double_t, double,
                                      FB_OP_ZGEBD2)
DEFINE_GEBD2_REAL_CBLAS_TO_FORTRAN(sgebd2, float, FB_OP_SGEBD2)
DEFINE_GEBD2_REAL_CBLAS_TO_FORTRAN(dgebd2, double, FB_OP_DGEBD2)
DEFINE_GEBD2_COMPLEX_CBLAS_TO_FORTRAN(cgebd2, fb_complex_float_t, float,
                                      FB_OP_CGEBD2)
DEFINE_GEBD2_COMPLEX_CBLAS_TO_FORTRAN(zgebd2, fb_complex_double_t, double,
                                      FB_OP_ZGEBD2)

DEFINE_LAPACK_LATRD_REAL_THUNKS(slatrd, float, FB_OP_SLATRD)
DEFINE_LAPACK_LATRD_REAL_THUNKS(dlatrd, double, FB_OP_DLATRD)
DEFINE_LAPACK_LATRD_COMPLEX_THUNKS(clatrd, fb_complex_float_t, float,
                                   FB_OP_CLATRD)
DEFINE_LAPACK_LATRD_COMPLEX_THUNKS(zlatrd, fb_complex_double_t, double,
                                   FB_OP_ZLATRD)

DEFINE_LAPACK_LATDF_THUNKS(slatdf, float, float, FB_OP_SLATDF)
DEFINE_LAPACK_LATDF_THUNKS(dlatdf, double, double, FB_OP_DLATDF)
DEFINE_LAPACK_LATDF_THUNKS(clatdf, fb_complex_float_t, float, FB_OP_CLATDF)
DEFINE_LAPACK_LATDF_THUNKS(zlatdf, fb_complex_double_t, double, FB_OP_ZLATDF)
DEFINE_LAPACK_LASSQ_REAL_THUNKS(slassq, float, FB_OP_SLASSQ)
DEFINE_LAPACK_LASSQ_REAL_THUNKS(dlassq, double, FB_OP_DLASSQ)
DEFINE_LAPACK_LASSQ_COMPLEX_THUNKS(classq, fb_complex_float_t, float,
                                   FB_OP_CLASSQ)
DEFINE_LAPACK_LASSQ_COMPLEX_THUNKS(zlassq, fb_complex_double_t, double,
                                   FB_OP_ZLASSQ)
DEFINE_LAPACK_LAEV2_THUNKS(slaev2, float, FB_OP_SLAEV2)
DEFINE_LAPACK_LAEV2_THUNKS(dlaev2, double, FB_OP_DLAEV2)
DEFINE_LAPACK_LASV2_THUNKS(slasv2, float, FB_OP_SLASV2)
DEFINE_LAPACK_LASV2_THUNKS(dlasv2, double, FB_OP_DLASV2)
DEFINE_LAPACK_LAEIN_REAL_THUNKS(slaein, float, FB_OP_SLAEIN)
DEFINE_LAPACK_LAEIN_REAL_THUNKS(dlaein, double, FB_OP_DLAEIN)
DEFINE_LAPACK_LAEIN_COMPLEX_THUNKS(claein, fb_complex_float_t, float,
                                   FB_OP_CLAEIN)
DEFINE_LAPACK_LAEIN_COMPLEX_THUNKS(zlaein, fb_complex_double_t, double,
                                   FB_OP_ZLAEIN)
DEFINE_LAPACK_LAHQR_REAL_THUNKS(slahqr, float, FB_OP_SLAHQR)
DEFINE_LAPACK_LAHQR_REAL_THUNKS(dlahqr, double, FB_OP_DLAHQR)
DEFINE_LAPACK_LAHQR_COMPLEX_THUNKS(clahqr, fb_complex_float_t, FB_OP_CLAHQR)
DEFINE_LAPACK_LAHQR_COMPLEX_THUNKS(zlahqr, fb_complex_double_t, FB_OP_ZLAHQR)
DEFINE_LAPACK_LAHRD_REAL_THUNKS(slahrd, float, FB_OP_SLAHRD)
DEFINE_LAPACK_LAHRD_REAL_THUNKS(dlahrd, double, FB_OP_DLAHRD)
DEFINE_LAPACK_LAHRD_COMPLEX_THUNKS(clahrd, fb_complex_float_t, FB_OP_CLAHRD)
DEFINE_LAPACK_LAHRD_COMPLEX_THUNKS(zlahrd, fb_complex_double_t, FB_OP_ZLAHRD)
DEFINE_LAPACK_LAQR0_THUNKS(slaqr0, float, FB_OP_SLAQR0)
DEFINE_LAPACK_LAQR0_THUNKS(dlaqr0, double, FB_OP_DLAQR0)
DEFINE_LAPACK_LAQR1_THUNKS(slaqr1, float, FB_OP_SLAQR1)
DEFINE_LAPACK_LAQR1_THUNKS(dlaqr1, double, FB_OP_DLAQR1)
DEFINE_LAPACK_LAQR2_THUNKS(slaqr2, float, FB_OP_SLAQR2)
DEFINE_LAPACK_LAQR2_THUNKS(dlaqr2, double, FB_OP_DLAQR2)
DEFINE_LAPACK_LAQP2_REAL_THUNKS(slaqp2, float, FB_OP_SLAQP2)
DEFINE_LAPACK_LAQP2_REAL_THUNKS(dlaqp2, double, FB_OP_DLAQP2)
DEFINE_LAPACK_LAQP2_COMPLEX_THUNKS(claqp2, fb_complex_float_t, float,
                                   FB_OP_CLAQP2)
DEFINE_LAPACK_LAQP2_COMPLEX_THUNKS(zlaqp2, fb_complex_double_t, double,
                                   FB_OP_ZLAQP2)
DEFINE_LAPACK_LAQPS_REAL_THUNKS(slaqps, float, FB_OP_SLAQPS)
DEFINE_LAPACK_LAQPS_REAL_THUNKS(dlaqps, double, FB_OP_DLAQPS)
DEFINE_LAPACK_LAQPS_COMPLEX_THUNKS(claqps, fb_complex_float_t, float,
                                   FB_OP_CLAQPS)
DEFINE_LAPACK_LAQPS_COMPLEX_THUNKS(zlaqps, fb_complex_double_t, double,
                                   FB_OP_ZLAQPS)
DEFINE_LAPACK_LARFB_THUNKS(slarfb, float, FB_OP_SLARFB)
DEFINE_LAPACK_LARFB_THUNKS(dlarfb, double, FB_OP_DLARFB)
DEFINE_LAPACK_LARFB_THUNKS(clarfb, fb_complex_float_t, FB_OP_CLARFB)
DEFINE_LAPACK_LARFB_THUNKS(zlarfb, fb_complex_double_t, FB_OP_ZLARFB)
DEFINE_LAPACK_LARNV_REAL_THUNKS(slarnv, float, FB_OP_SLARNV)
DEFINE_LAPACK_LARNV_REAL_THUNKS(dlarnv, double, FB_OP_DLARNV)
DEFINE_LAPACK_LARNV_COMPLEX_THUNKS(clarnv, fb_complex_float_t, FB_OP_CLARNV)
DEFINE_LAPACK_LARNV_COMPLEX_THUNKS(zlarnv, fb_complex_double_t, FB_OP_ZLARNV)
DEFINE_LAPACK_LARFT_REAL_THUNKS(slarft, float, FB_OP_SLARFT)
DEFINE_LAPACK_LARFT_REAL_THUNKS(dlarft, double, FB_OP_DLARFT)
DEFINE_LAPACK_LARFT_COMPLEX_THUNKS(clarft, fb_complex_float_t, FB_OP_CLARFT)
DEFINE_LAPACK_LARFT_COMPLEX_THUNKS(zlarft, fb_complex_double_t, FB_OP_ZLARFT)
DEFINE_LAPACK_LARFX_REAL_THUNKS(slarfx, float, FB_OP_SLARFX)
DEFINE_LAPACK_LARFX_REAL_THUNKS(dlarfx, double, FB_OP_DLARFX)
DEFINE_LAPACK_LARFX_COMPLEX_THUNKS(clarfx, fb_complex_float_t, FB_OP_CLARFX)
DEFINE_LAPACK_LARFX_COMPLEX_THUNKS(zlarfx, fb_complex_double_t, FB_OP_ZLARFX)
DEFINE_LAPACK_LARGE_THUNKS(slarge, float, FB_OP_SLARGE)
DEFINE_LAPACK_LARGE_THUNKS(dlarge, double, FB_OP_DLARGE)
DEFINE_LAPACK_LARGV_COMPLEX_THUNKS(clargv, fb_complex_float_t, float,
                                   FB_OP_CLARGV)
DEFINE_LAPACK_LARGV_COMPLEX_THUNKS(zlargv, fb_complex_double_t, double,
                                   FB_OP_ZLARGV)
DEFINE_LAPACK_LARRA_THUNKS(slarra, float, FB_OP_SLARRA)
DEFINE_LAPACK_LARRA_THUNKS(dlarra, double, FB_OP_DLARRA)
DEFINE_LAPACK_LARRB_THUNKS(slarrb, float, FB_OP_SLARRB)
DEFINE_LAPACK_LARRB_THUNKS(dlarrb, double, FB_OP_DLARRB)
DEFINE_LAPACK_LARRC_THUNKS(slarrc, float, FB_OP_SLARRC)
DEFINE_LAPACK_LARRC_THUNKS(dlarrc, double, FB_OP_DLARRC)
DEFINE_LAPACK_LARRD_THUNKS(slarrd, float, FB_OP_SLARRD)
DEFINE_LAPACK_LARRD_THUNKS(dlarrd, double, FB_OP_DLARRD)
DEFINE_LAPACK_LARRE_THUNKS(slarre, float, FB_OP_SLARRE)
DEFINE_LAPACK_LARRE_THUNKS(dlarre, double, FB_OP_DLARRE)
DEFINE_LAPACK_LARRF_THUNKS(slarrf, float, FB_OP_SLARRF)
DEFINE_LAPACK_LARRF_THUNKS(dlarrf, double, FB_OP_DLARRF)
DEFINE_LAPACK_LARRJ_THUNKS(slarrj, float, FB_OP_SLARRJ)
DEFINE_LAPACK_LARRJ_THUNKS(dlarrj, double, FB_OP_DLARRJ)
DEFINE_LAPACK_LARRK_THUNKS(slarrk, float, FB_OP_SLARRK)
DEFINE_LAPACK_LARRK_THUNKS(dlarrk, double, FB_OP_DLARRK)
DEFINE_LAPACK_LARRR_THUNKS(slarrr, float, FB_OP_SLARRR)
DEFINE_LAPACK_LARRR_THUNKS(dlarrr, double, FB_OP_DLARRR)
DEFINE_LAPACK_LARRV_THUNKS(slarrv, float, FB_OP_SLARRV)
DEFINE_LAPACK_LARRV_THUNKS(dlarrv, double, FB_OP_DLARRV)
DEFINE_LAPACK_LARTG_REAL_THUNKS(slartg, float, FB_OP_SLARTG)
DEFINE_LAPACK_LARTG_REAL_THUNKS(dlartg, double, FB_OP_DLARTG)
DEFINE_LAPACK_LARTG_COMPLEX_THUNKS(clartg, fb_complex_float_t, float, FB_OP_CLARTG)
DEFINE_LAPACK_LARTG_COMPLEX_THUNKS(zlartg, fb_complex_double_t, double, FB_OP_ZLARTG)
DEFINE_LAPACK_LARTV_REAL_THUNKS(slartv, float, FB_OP_SLARTV)
DEFINE_LAPACK_LARTV_REAL_THUNKS(dlartv, double, FB_OP_DLARTV)
DEFINE_LAPACK_LARTV_COMPLEX_THUNKS(clartv, fb_complex_float_t, float, FB_OP_CLARTV)
DEFINE_LAPACK_LARTV_COMPLEX_THUNKS(zlartv, fb_complex_double_t, double, FB_OP_ZLARTV)
DEFINE_LAPACK_LARZ_THUNKS(slarz, float, FB_OP_SLARZ)
DEFINE_LAPACK_LARZ_THUNKS(dlarz, double, FB_OP_DLARZ)
DEFINE_LAPACK_LARZ_THUNKS(clarz, fb_complex_float_t, FB_OP_CLARZ)
DEFINE_LAPACK_LARZ_THUNKS(zlarz, fb_complex_double_t, FB_OP_ZLARZ)
DEFINE_LAPACK_LARZB_THUNKS(slarzb, float, FB_OP_SLARZB)
DEFINE_LAPACK_LARZB_THUNKS(dlarzb, double, FB_OP_DLARZB)
DEFINE_LAPACK_LARZB_THUNKS(clarzb, fb_complex_float_t, FB_OP_CLARZB)
DEFINE_LAPACK_LARZB_THUNKS(zlarzb, fb_complex_double_t, FB_OP_ZLARZB)
DEFINE_LAPACK_LARZT_THUNKS(slarzt, float, FB_OP_SLARZT)
DEFINE_LAPACK_LARZT_THUNKS(dlarzt, double, FB_OP_DLARZT)
DEFINE_LAPACK_LARZT_THUNKS(clarzt, fb_complex_float_t, FB_OP_CLARZT)
DEFINE_LAPACK_LARZT_THUNKS(zlarzt, fb_complex_double_t, FB_OP_ZLARZT)
DEFINE_LAPACK_LAS2_THUNKS(slas2, float, FB_OP_SLAS2)
DEFINE_LAPACK_LAS2_THUNKS(dlas2, double, FB_OP_DLAS2)
DEFINE_LAPACK_LASCL_REAL_THUNKS(slascl, float, FB_OP_SLASCL)
DEFINE_LAPACK_LASCL_REAL_THUNKS(dlascl, double, FB_OP_DLASCL)
DEFINE_LAPACK_LASCL_COMPLEX_THUNKS(clascl, fb_complex_float_t, float, FB_OP_CLASCL)
DEFINE_LAPACK_LASCL_COMPLEX_THUNKS(zlascl, fb_complex_double_t, double, FB_OP_ZLASCL)
DEFINE_LAPACK_LASD0_THUNKS(slasd0, float, FB_OP_SLASD0)
DEFINE_LAPACK_LASD0_THUNKS(dlasd0, double, FB_OP_DLASD0)
DEFINE_LAPACK_LASD1_THUNKS(slasd1, float, FB_OP_SLASD1)
DEFINE_LAPACK_LASD1_THUNKS(dlasd1, double, FB_OP_DLASD1)
DEFINE_LAPACK_LASD2_THUNKS(slasd2, float, FB_OP_SLASD2)
DEFINE_LAPACK_LASD2_THUNKS(dlasd2, double, FB_OP_DLASD2)
DEFINE_LAPACK_LASD3_THUNKS(slasd3, float, FB_OP_SLASD3)
DEFINE_LAPACK_LASD3_THUNKS(dlasd3, double, FB_OP_DLASD3)
DEFINE_LAPACK_LASD4_THUNKS(slasd4, float, FB_OP_SLASD4)
DEFINE_LAPACK_LASD4_THUNKS(dlasd4, double, FB_OP_DLASD4)
DEFINE_LAPACK_LASD5_THUNKS(slasd5, float, FB_OP_SLASD5)
DEFINE_LAPACK_LASD5_THUNKS(dlasd5, double, FB_OP_DLASD5)
DEFINE_LAPACK_LASD6_THUNKS(slasd6, float, FB_OP_SLASD6)
DEFINE_LAPACK_LASD6_THUNKS(dlasd6, double, FB_OP_DLASD6)
DEFINE_LAPACK_LASD7_THUNKS(slasd7, float, FB_OP_SLASD7)
DEFINE_LAPACK_LASD7_THUNKS(dlasd7, double, FB_OP_DLASD7)
DEFINE_LAPACK_LASD8_THUNKS(slasd8, float, FB_OP_SLASD8)
DEFINE_LAPACK_LASD8_THUNKS(dlasd8, double, FB_OP_DLASD8)
DEFINE_LAPACK_LASDA_THUNKS(slasda, float, FB_OP_SLASDA)
DEFINE_LAPACK_LASDA_THUNKS(dlasda, double, FB_OP_DLASDA)
DEFINE_LAPACK_LASDQ_THUNKS(slasdq, float, FB_OP_SLASDQ)
DEFINE_LAPACK_LASDQ_THUNKS(dlasdq, double, FB_OP_DLASDQ)
DEFINE_LAPACK_LASDT_THUNKS(slasdt, FB_OP_SLASDT)
DEFINE_LAPACK_LASDT_THUNKS(dlasdt, FB_OP_DLASDT)
DEFINE_LAPACK_LASET_REAL_THUNKS(slaset, float, FB_OP_SLASET)
DEFINE_LAPACK_LASET_REAL_THUNKS(dlaset, double, FB_OP_DLASET)
DEFINE_LAPACK_LASET_COMPLEX_THUNKS(claset, fb_complex_float_t, FB_OP_CLASET)
DEFINE_LAPACK_LASET_COMPLEX_THUNKS(zlaset, fb_complex_double_t, FB_OP_ZLASET)
DEFINE_LAPACK_LASQ1_THUNKS(slasq1, float, FB_OP_SLASQ1)
DEFINE_LAPACK_LASQ1_THUNKS(dlasq1, double, FB_OP_DLASQ1)
DEFINE_LAPACK_LASQ2_THUNKS(slasq2, float, FB_OP_SLASQ2)
DEFINE_LAPACK_LASQ2_THUNKS(dlasq2, double, FB_OP_DLASQ2)
DEFINE_LAPACK_LASQ3_THUNKS(slasq3, float, FB_OP_SLASQ3)
DEFINE_LAPACK_LASQ3_THUNKS(dlasq3, double, FB_OP_DLASQ3)
DEFINE_LAPACK_LASQ4_THUNKS(slasq4, float, FB_OP_SLASQ4)
DEFINE_LAPACK_LASQ4_THUNKS(dlasq4, double, FB_OP_DLASQ4)
DEFINE_LAPACK_LASQ5_THUNKS(slasq5, float, FB_OP_SLASQ5)
DEFINE_LAPACK_LASQ5_THUNKS(dlasq5, double, FB_OP_DLASQ5)
DEFINE_LAPACK_LASQ6_THUNKS(slasq6, float, FB_OP_SLASQ6)
DEFINE_LAPACK_LASQ6_THUNKS(dlasq6, double, FB_OP_DLASQ6)
DEFINE_LAPACK_LASR_THUNKS(slasr, float, FB_OP_SLASR)
DEFINE_LAPACK_LASR_THUNKS(dlasr, double, FB_OP_DLASR)
DEFINE_LAPACK_LAEXC_THUNKS(slaexc, float, FB_OP_SLAEXC)
DEFINE_LAPACK_LAEXC_THUNKS(dlaexc, double, FB_OP_DLAEXC)
DEFINE_LAPACK_LAEXC_THUNKS(claexc, fb_complex_float_t, FB_OP_CLAEXC)
DEFINE_LAPACK_LAEXC_THUNKS(zlaexc, fb_complex_double_t, FB_OP_ZLAEXC)
DEFINE_LAPACK_LAESY_THUNKS(claesy, fb_complex_float_t, FB_OP_CLAESY)
DEFINE_LAPACK_LAESY_THUNKS(zlaesy, fb_complex_double_t, FB_OP_ZLAESY)
DEFINE_LAPACK_LAIC1_THUNKS(slaic1, float, FB_OP_SLAIC1)
DEFINE_LAPACK_LAIC1_THUNKS(dlaic1, double, FB_OP_DLAIC1)
DEFINE_LAPACK_LAIC1_COMPLEX_THUNKS(claic1, fb_complex_float_t, float,
                                   FB_OP_CLAIC1)
DEFINE_LAPACK_LAIC1_COMPLEX_THUNKS(zlaic1, fb_complex_double_t, double,
                                   FB_OP_ZLAIC1)
DEFINE_LAPACK_LALN2_THUNKS(slaln2, float, FB_OP_SLALN2)
DEFINE_LAPACK_LALN2_THUNKS(dlaln2, double, FB_OP_DLALN2)
DEFINE_LAPACK_LACPY_THUNKS(slacpy, float, FB_OP_SLACPY)
DEFINE_LAPACK_LACPY_THUNKS(dlacpy, double, FB_OP_DLACPY)
DEFINE_LAPACK_LACPY_THUNKS(clacpy, fb_complex_float_t, FB_OP_CLACPY)
DEFINE_LAPACK_LACPY_THUNKS(zlacpy, fb_complex_double_t, FB_OP_ZLACPY)
DEFINE_LAPACK_LASRT_THUNKS(slasrt, float, FB_OP_SLASRT)
DEFINE_LAPACK_LASRT_THUNKS(dlasrt, double, FB_OP_DLASRT)
DEFINE_LAPACK_LASWP_THUNKS(slaswp, float, FB_OP_SLASWP)
DEFINE_LAPACK_LASWP_THUNKS(dlaswp, double, FB_OP_DLASWP)
DEFINE_LAPACK_LASWP_THUNKS(claswp, fb_complex_float_t, FB_OP_CLASWP)
DEFINE_LAPACK_LASWP_THUNKS(zlaswp, fb_complex_double_t, FB_OP_ZLASWP)
DEFINE_LAPACK_LATBS_REAL_THUNKS(slatbs, float, FB_OP_SLATBS)
DEFINE_LAPACK_LATBS_REAL_THUNKS(dlatbs, double, FB_OP_DLATBS)
DEFINE_LAPACK_LATBS_COMPLEX_THUNKS(clatbs, fb_complex_float_t, float,
                                   FB_OP_CLATBS)
DEFINE_LAPACK_LATBS_COMPLEX_THUNKS(zlatbs, fb_complex_double_t, double,
                                   FB_OP_ZLATBS)
DEFINE_LAPACK_LATPS_REAL_THUNKS(slatps, float, FB_OP_SLATPS)
DEFINE_LAPACK_LATPS_REAL_THUNKS(dlatps, double, FB_OP_DLATPS)
DEFINE_LAPACK_LATPS_COMPLEX_THUNKS(clatps, fb_complex_float_t, float,
                                   FB_OP_CLATPS)
DEFINE_LAPACK_LATPS_COMPLEX_THUNKS(zlatps, fb_complex_double_t, double,
                                   FB_OP_ZLATPS)
DEFINE_LAPACK_LATRS_REAL_THUNKS(slatrs, float, FB_OP_SLATRS)
DEFINE_LAPACK_LATRS_REAL_THUNKS(dlatrs, double, FB_OP_DLATRS)
DEFINE_LAPACK_LATRS_COMPLEX_THUNKS(clatrs, fb_complex_float_t, float,
                                   FB_OP_CLATRS)
DEFINE_LAPACK_LATRS_COMPLEX_THUNKS(zlatrs, fb_complex_double_t, double,
                                   FB_OP_ZLATRS)

#undef DEFINE_LAPACK_LATRD_REAL_THUNKS
#undef DEFINE_LAPACK_LATRD_COMPLEX_THUNKS
#undef DEFINE_LAPACK_LATDF_THUNKS
#undef DEFINE_LAPACK_LASSQ_REAL_THUNKS
#undef DEFINE_LAPACK_LASSQ_COMPLEX_THUNKS
#undef DEFINE_LAPACK_LAEV2_THUNKS
#undef DEFINE_LAPACK_LA_GBAMV_REAL_THUNKS
#undef DEFINE_LAPACK_LA_GBAMV_COMPLEX_THUNKS
#undef DEFINE_LAPACK_LASV2_THUNKS
#undef DEFINE_LAPACK_LAEIN_REAL_THUNKS
#undef DEFINE_LAPACK_LAEIN_COMPLEX_THUNKS
#undef DEFINE_LAPACK_LAHQR_REAL_THUNKS
#undef DEFINE_LAPACK_LAHQR_COMPLEX_THUNKS
#undef DEFINE_LAPACK_LAHRD_REAL_THUNKS
#undef DEFINE_LAPACK_LAHRD_COMPLEX_THUNKS
#undef DEFINE_LAPACK_LAQR0_THUNKS
#undef DEFINE_LAPACK_LAQR1_THUNKS
#undef DEFINE_LAPACK_LAQR2_THUNKS
#undef DEFINE_LAPACK_LAQP2_REAL_THUNKS
#undef DEFINE_LAPACK_LAQP2_COMPLEX_THUNKS
#undef DEFINE_LAPACK_LAQPS_REAL_THUNKS
#undef DEFINE_LAPACK_LAQPS_COMPLEX_THUNKS
#undef DEFINE_LAPACK_LARFB_THUNKS
#undef DEFINE_LAPACK_LARNV_REAL_THUNKS
#undef DEFINE_LAPACK_LARNV_COMPLEX_THUNKS
#undef DEFINE_LAPACK_LARFT_REAL_THUNKS
#undef DEFINE_LAPACK_LARFT_COMPLEX_THUNKS
#undef DEFINE_LAPACK_LARFX_REAL_THUNKS
#undef DEFINE_LAPACK_LARFX_COMPLEX_THUNKS
#undef DEFINE_LAPACK_LARGE_THUNKS
#undef DEFINE_LAPACK_LARGV_COMPLEX_THUNKS
#undef DEFINE_LAPACK_LARRA_THUNKS
#undef DEFINE_LAPACK_LARRB_THUNKS
#undef DEFINE_LAPACK_LARRC_THUNKS
#undef DEFINE_LAPACK_LARRD_THUNKS
#undef DEFINE_LAPACK_LARRE_THUNKS
#undef DEFINE_LAPACK_LARRF_THUNKS
#undef DEFINE_LAPACK_LARRJ_THUNKS
#undef DEFINE_LAPACK_LARRK_THUNKS
#undef DEFINE_LAPACK_LARRR_THUNKS
#undef DEFINE_LAPACK_LARRV_THUNKS
#undef DEFINE_LAPACK_LARTG_REAL_THUNKS
#undef DEFINE_LAPACK_LARTG_COMPLEX_THUNKS
#undef DEFINE_LAPACK_LARTV_REAL_THUNKS
#undef DEFINE_LAPACK_LARTV_COMPLEX_THUNKS
#undef DEFINE_LAPACK_LAS2_THUNKS
#undef DEFINE_LAPACK_LASCL_REAL_THUNKS
#undef DEFINE_LAPACK_LASCL_COMPLEX_THUNKS
#undef DEFINE_LAPACK_LASD0_THUNKS
#undef DEFINE_LAPACK_LASD1_THUNKS
#undef DEFINE_LAPACK_LASD2_THUNKS
#undef DEFINE_LAPACK_LASD3_THUNKS
#undef DEFINE_LAPACK_LASD4_THUNKS
#undef DEFINE_LAPACK_LASD5_THUNKS
#undef DEFINE_LAPACK_LASD6_THUNKS
#undef DEFINE_LAPACK_LASD7_THUNKS
#undef DEFINE_LAPACK_LASD8_THUNKS
#undef DEFINE_LAPACK_LASDA_THUNKS
#undef DEFINE_LAPACK_LASDQ_THUNKS
#undef DEFINE_LAPACK_LASDT_THUNKS
#undef DEFINE_LAPACK_LASET_REAL_THUNKS
#undef DEFINE_LAPACK_LASET_COMPLEX_THUNKS
#undef DEFINE_LAPACK_LASQ1_THUNKS
#undef DEFINE_LAPACK_LASQ2_THUNKS
#undef DEFINE_LAPACK_LASQ3_THUNKS
#undef DEFINE_LAPACK_LASQ4_THUNKS
#undef DEFINE_LAPACK_LASQ5_THUNKS
#undef DEFINE_LAPACK_LASQ6_THUNKS
#undef DEFINE_LAPACK_LASR_THUNKS
#undef DEFINE_LAPACK_LARZ_THUNKS
#undef DEFINE_LAPACK_LARZB_THUNKS
#undef DEFINE_LAPACK_LARZT_THUNKS
#undef DEFINE_LAPACK_LAEXC_THUNKS
#undef DEFINE_LAPACK_LAESY_THUNKS
#undef DEFINE_LAPACK_LAIC1_THUNKS
#undef DEFINE_LAPACK_LAIC1_COMPLEX_THUNKS
#undef DEFINE_LAPACK_LALN2_THUNKS
#undef DEFINE_LAPACK_LACPY_THUNKS
#undef DEFINE_LAPACK_LASRT_THUNKS
#undef DEFINE_LAPACK_LASWP_THUNKS
#undef DEFINE_LAPACK_LATBS_REAL_THUNKS
#undef DEFINE_LAPACK_LATBS_COMPLEX_THUNKS
#undef DEFINE_LAPACK_LATPS_REAL_THUNKS
#undef DEFINE_LAPACK_LATPS_COMPLEX_THUNKS
#undef DEFINE_LAPACK_LATRS_REAL_THUNKS
#undef DEFINE_LAPACK_LATRS_COMPLEX_THUNKS
#undef DEFINE_LAPACK_PP_SVX_REAL_THUNKS
#undef DEFINE_LAPACK_PP_SVX_COMPLEX_THUNKS
#undef DEFINE_LAPACK_PP_FACTOR_THUNKS
#undef DEFINE_LAPACK_PP_FACTORED_SOLVE_THUNKS
#undef DEFINE_LAPACK_HP_FACTORED_SOLVE_THUNKS
#undef DEFINE_LAPACK_PP_REFINE_REAL_THUNKS
#undef DEFINE_LAPACK_PP_REFINE_COMPLEX_THUNKS
#undef DEFINE_LAPACK_HP_REFINE_COMPLEX_THUNKS
#undef DEFINE_LAPACK_PT_CON_REAL_THUNKS
#undef DEFINE_LAPACK_PT_CON_COMPLEX_THUNKS
#undef DEFINE_LAPACK_PTRFS_REAL_THUNKS
#undef DEFINE_LAPACK_PTRFS_COMPLEX_THUNKS
#undef DEFINE_COMPLEX_ROTG_THUNKS
#undef DEFINE_RSCL_REAL_THUNKS
#undef DEFINE_RSCL_COMPLEX_THUNKS
#undef DEFINE_SPEV_REAL_THUNKS
#undef DEFINE_SPEVD_REAL_THUNKS
#undef DEFINE_SPEVX_REAL_THUNKS
#undef DEFINE_SPGV_REAL_THUNKS
#undef DEFINE_SPGVD_REAL_THUNKS
#undef DEFINE_SPGVX_REAL_THUNKS
#undef DEFINE_HPEV_COMPLEX_THUNKS
#undef DEFINE_HPEVD_COMPLEX_THUNKS
#undef DEFINE_HPEVX_COMPLEX_THUNKS
#undef DEFINE_HPGV_COMPLEX_THUNKS
#undef DEFINE_HPGVD_COMPLEX_THUNKS
#undef DEFINE_HPGVX_COMPLEX_THUNKS
#undef DEFINE_SBEV_REAL_THUNKS
#undef DEFINE_HBEV_COMPLEX_THUNKS
#undef DEFINE_HBEVD_COMPLEX_THUNKS
#undef DEFINE_SBEVD_REAL_THUNKS
#undef DEFINE_SBEVX_REAL_THUNKS
#undef DEFINE_SBGV_REAL_THUNKS
#undef DEFINE_HBGV_COMPLEX_THUNKS
#undef DEFINE_SBGVD_REAL_THUNKS
#undef DEFINE_HBGVD_COMPLEX_THUNKS
#undef DEFINE_SBGVX_REAL_THUNKS
#undef DEFINE_HBGVX_COMPLEX_THUNKS
#undef DEFINE_LAPACK_PT_SOLVE_REAL_THUNKS
#undef DEFINE_LAPACK_PT_SOLVE_COMPLEX_THUNKS
#undef DEFINE_LAPACK_PT_SVX_REAL_THUNKS
#undef DEFINE_LAPACK_PT_SVX_COMPLEX_THUNKS
#undef DEFINE_LAPACK_PT_FACTOR_REAL_THUNKS
#undef DEFINE_LAPACK_PT_FACTOR_COMPLEX_THUNKS
#undef DEFINE_LAPACK_PT_FACTORED_SOLVE_REAL_THUNKS
#undef DEFINE_LAPACK_PT_FACTORED_SOLVE_COMPLEX_THUNKS
#undef DEFINE_LAPACK_GB_REFINE_REAL_THUNKS
#undef DEFINE_LAPACK_GB_REFINE_COMPLEX_THUNKS

DEFINE_GEBRD_REAL_FORTRAN_TO_CBLAS(sgebrd, float, FB_OP_SGEBRD,
                                   fb_query_size_from_float)
DEFINE_GEBRD_REAL_FORTRAN_TO_CBLAS(dgebrd, double, FB_OP_DGEBRD,
                                   fb_query_size_from_double)
DEFINE_GEBRD_COMPLEX_FORTRAN_TO_CBLAS(cgebrd, fb_complex_float_t, float,
                                      FB_OP_CGEBRD,
                                      fb_query_size_from_cfloat)
DEFINE_GEBRD_COMPLEX_FORTRAN_TO_CBLAS(zgebrd, fb_complex_double_t, double,
                                      FB_OP_ZGEBRD,
                                      fb_query_size_from_cdouble)
DEFINE_GEBRD_REAL_CBLAS_TO_FORTRAN(sgebrd, float, FB_OP_SGEBRD)
DEFINE_GEBRD_REAL_CBLAS_TO_FORTRAN(dgebrd, double, FB_OP_DGEBRD)
DEFINE_GEBRD_COMPLEX_CBLAS_TO_FORTRAN(cgebrd, fb_complex_float_t, float,
                                      FB_OP_CGEBRD)
DEFINE_GEBRD_COMPLEX_CBLAS_TO_FORTRAN(zgebrd, fb_complex_double_t, double,
                                      FB_OP_ZGEBRD)

DEFINE_GECON_REAL_FORTRAN_TO_CBLAS(sgecon, float, FB_OP_SGECON)
DEFINE_GECON_REAL_FORTRAN_TO_CBLAS(dgecon, double, FB_OP_DGECON)
DEFINE_GECON_COMPLEX_FORTRAN_TO_CBLAS(cgecon, fb_complex_float_t, float,
                                      FB_OP_CGECON)
DEFINE_GECON_COMPLEX_FORTRAN_TO_CBLAS(zgecon, fb_complex_double_t, double,
                                      FB_OP_ZGECON)
DEFINE_GECON_REAL_CBLAS_TO_FORTRAN(sgecon, float, FB_OP_SGECON)
DEFINE_GECON_REAL_CBLAS_TO_FORTRAN(dgecon, double, FB_OP_DGECON)
DEFINE_GECON_COMPLEX_CBLAS_TO_FORTRAN(cgecon, fb_complex_float_t, float,
                                      FB_OP_CGECON)
DEFINE_GECON_COMPLEX_CBLAS_TO_FORTRAN(zgecon, fb_complex_double_t, double,
                                      FB_OP_ZGECON)
DEFINE_GEBAK_REAL_FORTRAN_TO_CBLAS(sgebak, float, FB_OP_SGEBAK)
DEFINE_GEBAK_REAL_FORTRAN_TO_CBLAS(dgebak, double, FB_OP_DGEBAK)
DEFINE_GEBAK_COMPLEX_FORTRAN_TO_CBLAS(cgebak, fb_complex_float_t, float,
                                      FB_OP_CGEBAK)
DEFINE_GEBAK_COMPLEX_FORTRAN_TO_CBLAS(zgebak, fb_complex_double_t, double,
                                      FB_OP_ZGEBAK)
