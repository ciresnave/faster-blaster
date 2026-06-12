
#define DEFINE_LAPACK_LU_FACTOR_THUNKS(OP, TYPE, OP_ID)                        \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const int m, const int n, TYPE *A,               \
    const int lda, int *ipiv) {                                                \
  if (!fb_lapack_layout_is_col_major(layout)) return -1;                       \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  int m_ = m, n_ = n, lda_ = lda, info = 0;                                    \
  ((void (*)(int *, int *, TYPE *, int *, int *, int *))g_fortran_fn[OP_ID])(  \
      &m_, &n_, A, &lda_, ipiv, &info);                                        \
  return info;                                                                 \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    int *m, int *n, TYPE *A, int *lda, int *ipiv, int *info) {                 \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, int, int, TYPE *, int, int *))               \
      g_cblas_fn[OP_ID])(FB_LAYOUT_COL_MAJOR, *m, *n, A, *lda, ipiv);          \
  if (info) *info = rc;                                                         \
}

#define DEFINE_LAPACK_LU_FACTORED_SOLVE_THUNKS(OP, TYPE, OP_ID)                \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const fb_transpose_t trans, const int n,         \
    const int nrhs, const TYPE *A, const int lda, const int *ipiv, TYPE *B,    \
    const int ldb) {                                                            \
  char trans_char = 'N';                                                        \
  if (!fb_lapack_layout_is_col_major(layout)) return -1;                       \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (fb_lapack_transpose_to_char(trans, &trans_char) != 0) return -1;         \
  int n_ = n, nrhs_ = nrhs, lda_ = lda, ldb_ = ldb, info = 0;                  \
  ((void (*)(char *, int *, int *, TYPE *, int *, int *, TYPE *, int *,        \
             int *))g_fortran_fn[OP_ID])(&trans_char, &n_, &nrhs_, (TYPE *)A,  \
                                         &lda_, (int *)ipiv, B, &ldb_, &info); \
  return info;                                                                 \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    char *trans, int *n, int *nrhs, TYPE *A, int *lda, int *ipiv, TYPE *B,     \
    int *ldb, int *info) {                                                      \
  fb_transpose_t trans_ = FB_NO_TRANS;                                          \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!trans || fb_lapack_char_to_transpose(*trans, &trans_) != 0) {           \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, fb_transpose_t, int, int, const TYPE *, int, \
                     const int *, TYPE *, int))g_cblas_fn[OP_ID])(              \
      FB_LAYOUT_COL_MAJOR, trans_, *n, *nrhs, A, *lda, ipiv, B, *ldb);         \
  if (info) *info = rc;                                                         \
}

#define DEFINE_LAPACK_MATRIX_TAU_QUERY_THUNKS(OP, TYPE, OP_ID, QUERY_FN)      \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const int m, const int n, TYPE *A,               \
    const int lda, TYPE *tau) {                                                 \
  int m_ = m, n_ = n, lda_ = (m > 0) ? m : 1, lwork_ = -1, info = 0;           \
  int tau_len = (m < n) ? m : n;                                                \
  int tau_alloc = (tau_len > 0) ? tau_len : 1;                                  \
  size_t a_elems = (size_t)lda_ * (size_t)((n > 0) ? n : 1);                   \
  TYPE work_query = (TYPE)0;                                                    \
  TYPE *a_col = NULL;                                                           \
  TYPE *tau_work = NULL;                                                        \
  TYPE *work = NULL;                                                            \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  a_col = (TYPE *)malloc(a_elems * sizeof(TYPE));                               \
  tau_work = (TYPE *)malloc((size_t)tau_alloc * sizeof(TYPE));                  \
  if (!a_col || !tau_work) {                                                    \
    free(a_col);                                                                \
    free(tau_work);                                                             \
    return -101;                                                                \
  }                                                                             \
  if (fb_copy_matrix_to_col_major(A, layout, m, n, lda,                        \
                                  a_col, lda_, sizeof(TYPE)) != 0) {           \
    free(a_col);                                                                \
    free(tau_work);                                                             \
    return -1;                                                                  \
  }                                                                             \
  memset(tau_work, 0, (size_t)tau_alloc * sizeof(TYPE));                        \
  ((void (*)(int *, int *, TYPE *, int *, TYPE *, TYPE *, int *, int *))       \
      g_fortran_fn[OP_ID])(&m_, &n_, a_col, &lda_, tau_work, &work_query,      \
                           &lwork_, &info);                                     \
  if (info != 0) {                                                              \
    free(a_col);                                                                \
    free(tau_work);                                                             \
    return info;                                                                \
  }                                                                             \
  lwork_ = QUERY_FN(work_query);                                                \
  work = (TYPE *)malloc((size_t)lwork_ * sizeof(TYPE));                         \
  if (!work) {                                                                  \
    free(a_col);                                                                \
    free(tau_work);                                                             \
    return -101;                                                                \
  }                                                                             \
  if (fb_copy_matrix_to_col_major(A, layout, m, n, lda,                        \
                                  a_col, lda_, sizeof(TYPE)) != 0) {           \
    free(a_col);                                                                \
    free(tau_work);                                                             \
    free(work);                                                                 \
    return -1;                                                                  \
  }                                                                             \
  memset(tau_work, 0, (size_t)tau_alloc * sizeof(TYPE));                        \
  ((void (*)(int *, int *, TYPE *, int *, TYPE *, TYPE *, int *, int *))       \
      g_fortran_fn[OP_ID])(&m_, &n_, a_col, &lda_, tau_work, work,             \
                           &lwork_, &info);                                     \
  if (fb_copy_matrix_from_col_major(a_col, m, n, lda_, layout,                 \
                                    A, lda, sizeof(TYPE)) != 0) {              \
    free(a_col);                                                                \
    free(tau_work);                                                             \
    free(work);                                                                 \
    return -1;                                                                  \
  }                                                                             \
  if (tau && tau_len > 0) {                                                     \
    memcpy(tau, tau_work, (size_t)tau_len * sizeof(TYPE));                      \
  }                                                                             \
  free(a_col);                                                                  \
  free(tau_work);                                                               \
  free(work);                                                                   \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    int *m, int *n, TYPE *A, int *lda, TYPE *tau, TYPE *work,                  \
    int *lwork, int *info) {                                                    \
  (void)work;                                                                   \
  (void)lwork;                                                                  \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, int, int, TYPE *, int, TYPE *))              \
      g_cblas_fn[OP_ID])(FB_LAYOUT_COL_MAJOR, *m, *n, A, *lda, tau);           \
  if (info) *info = rc;                                                         \
}

#define DEFINE_LAPACK_MATRIX_K_TAU_QUERY_THUNKS(OP, TYPE, OP_ID, QUERY_FN)    \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const int m, const int n, const int k,           \
    TYPE *A, const int lda, const TYPE *tau) {                                  \
  int m_ = m, n_ = n, k_ = k, lda_ = (m > 0) ? m : 1, lwork_ = -1, info = 0;   \
  int tau_alloc = (k > 0) ? k : 1;                                              \
  size_t a_elems = (size_t)lda_ * (size_t)((n > 0) ? n : 1);                   \
  TYPE work_query = (TYPE)0;                                                    \
  TYPE *a_col = NULL;                                                           \
  TYPE *tau_copy = NULL;                                                        \
  TYPE *work = NULL;                                                            \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (k > 0 && !tau) return -1;                                                 \
  a_col = (TYPE *)malloc(a_elems * sizeof(TYPE));                               \
  tau_copy = (TYPE *)malloc((size_t)tau_alloc * sizeof(TYPE));                  \
  if (!a_col || !tau_copy) {                                                    \
    free(a_col);                                                                \
    free(tau_copy);                                                             \
    return -101;                                                                \
  }                                                                             \
  if (fb_copy_matrix_to_col_major(A, layout, m, n, lda,                        \
                                  a_col, lda_, sizeof(TYPE)) != 0) {           \
    free(a_col);                                                                \
    free(tau_copy);                                                             \
    return -1;                                                                  \
  }                                                                             \
  memset(tau_copy, 0, (size_t)tau_alloc * sizeof(TYPE));                        \
  if (k > 0) {                                                                  \
    memcpy(tau_copy, tau, (size_t)k * sizeof(TYPE));                            \
  }                                                                             \
  ((void (*)(int *, int *, int *, TYPE *, int *, TYPE *, TYPE *, int *, int *))\
      g_fortran_fn[OP_ID])(&m_, &n_, &k_, a_col, &lda_, tau_copy,              \
                           &work_query, &lwork_, &info);                        \
  if (info != 0) {                                                              \
    free(a_col);                                                                \
    free(tau_copy);                                                             \
    return info;                                                                \
  }                                                                             \
  lwork_ = QUERY_FN(work_query);                                                \
  work = (TYPE *)malloc((size_t)lwork_ * sizeof(TYPE));                         \
  if (!work) {                                                                  \
    free(a_col);                                                                \
    free(tau_copy);                                                             \
    return -101;                                                                \
  }                                                                             \
  if (fb_copy_matrix_to_col_major(A, layout, m, n, lda,                        \
                                  a_col, lda_, sizeof(TYPE)) != 0) {           \
    free(a_col);                                                                \
    free(tau_copy);                                                             \
    free(work);                                                                 \
    return -1;                                                                  \
  }                                                                             \
  memset(tau_copy, 0, (size_t)tau_alloc * sizeof(TYPE));                        \
  if (k > 0) {                                                                  \
    memcpy(tau_copy, tau, (size_t)k * sizeof(TYPE));                            \
  }                                                                             \
  ((void (*)(int *, int *, int *, TYPE *, int *, TYPE *, TYPE *, int *, int *))\
      g_fortran_fn[OP_ID])(&m_, &n_, &k_, a_col, &lda_, tau_copy,              \
                           work, &lwork_, &info);                               \
  if (fb_copy_matrix_from_col_major(a_col, m, n, lda_, layout,                 \
                                    A, lda, sizeof(TYPE)) != 0) {              \
    free(a_col);                                                                \
    free(tau_copy);                                                             \
    free(work);                                                                 \
    return -1;                                                                  \
  }                                                                             \
  free(a_col);                                                                  \
  free(tau_copy);                                                               \
  free(work);                                                                   \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    int *m, int *n, int *k, TYPE *A, int *lda, TYPE *tau, TYPE *work,         \
    int *lwork, int *info) {                                                    \
  (void)work;                                                                   \
  (void)lwork;                                                                  \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, int, int, int, TYPE *, int, const TYPE *))   \
      g_cblas_fn[OP_ID])(FB_LAYOUT_COL_MAJOR, *m, *n, *k, A, *lda, tau);       \
  if (info) *info = rc;                                                         \
}

#define DEFINE_LAPACK_MATRIX_TAU_NOQUERY_THUNKS(OP, TYPE, OP_ID, WORK_LEN_EXPR) \
static int thunk_##OP##_fortran_to_cblas(                                        \
    const fb_layout_t layout, const int m, const int n, TYPE *A,                 \
    const int lda, TYPE *tau) {                                                   \
  int m_ = m, n_ = n, lda_ = (m > 0) ? m : 1, info = 0;                          \
  int tau_len = (m < n) ? m : n;                                                  \
  int tau_alloc = (tau_len > 0) ? tau_len : 1;                                    \
  int work_len = ((WORK_LEN_EXPR) > 0) ? (WORK_LEN_EXPR) : 1;                     \
  size_t a_elems = (size_t)lda_ * (size_t)((n > 0) ? n : 1);                     \
  TYPE *a_col = NULL;                                                              \
  TYPE *tau_work = NULL;                                                           \
  TYPE *work = NULL;                                                               \
  if (!g_fortran_fn[OP_ID]) return -1;                                             \
  if ((m > 0 && n > 0 && !A) || (tau_len > 0 && !tau)) return -1;                 \
  a_col = (TYPE *)malloc(a_elems * sizeof(TYPE));                                  \
  tau_work = (TYPE *)malloc((size_t)tau_alloc * sizeof(TYPE));                     \
  work = (TYPE *)malloc((size_t)work_len * sizeof(TYPE));                          \
  if (!a_col || !tau_work || !work) {                                              \
    free(a_col);                                                                   \
    free(tau_work);                                                                \
    free(work);                                                                    \
    return -101;                                                                   \
  }                                                                                \
  if (fb_copy_matrix_to_col_major(A, layout, m, n, lda,                           \
                                  a_col, lda_, sizeof(TYPE)) != 0) {             \
    free(a_col);                                                                   \
    free(tau_work);                                                                \
    free(work);                                                                    \
    return -1;                                                                     \
  }                                                                                \
  memset(tau_work, 0, (size_t)tau_alloc * sizeof(TYPE));                           \
  ((void (*)(int *, int *, TYPE *, int *, TYPE *, TYPE *, int *))                \
      g_fortran_fn[OP_ID])(&m_, &n_, a_col, &lda_, tau_work, work, &info);        \
  if (fb_copy_matrix_from_col_major(a_col, m, n, lda_, layout,                    \
                                    A, lda, sizeof(TYPE)) != 0) {                 \
    free(a_col);                                                                   \
    free(tau_work);                                                                \
    free(work);                                                                    \
    return -1;                                                                     \
  }                                                                                \
  if (tau && tau_len > 0) {                                                        \
    memcpy(tau, tau_work, (size_t)tau_len * sizeof(TYPE));                         \
  }                                                                                \
  free(a_col);                                                                     \
  free(tau_work);                                                                  \
  free(work);                                                                      \
  return info;                                                                     \
}                                                                                  \
static void thunk_##OP##_cblas_to_fortran(                                         \
    int *m, int *n, TYPE *A, int *lda, TYPE *tau, TYPE *work, int *info) {       \
  (void)work;                                                                      \
  if (!g_cblas_fn[OP_ID]) {                                                        \
    if (info) *info = -1;                                                          \
    return;                                                                        \
  }                                                                                \
  int rc = ((int (*)(fb_layout_t, int, int, TYPE *, int, TYPE *))                 \
      g_cblas_fn[OP_ID])(FB_LAYOUT_COL_MAJOR, *m, *n, A, *lda, tau);              \
  if (info) *info = rc;                                                            \
}

#define DEFINE_LAPACK_APPLY_Q_THUNKS(OP, TYPE, OP_ID, QUERY_FN)               \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const fb_side_t side,                            \
    const fb_transpose_t trans, const int m, const int n, const int k,         \
    const TYPE *A, const int lda, const TYPE *tau, TYPE *C, const int ldc) {   \
  char side_char = 'L';                                                         \
  char trans_char = 'N';                                                        \
  int m_ = m, n_ = n, k_ = k;                                              \
  int a_rows = 0, lda_ = 1, ldc_ = (m > 0) ? m : 1;                            \
  int lwork_ = -1, info = 0, tau_alloc = (k > 0) ? k : 1;                      \
  size_t a_elems = 0, c_elems = 0;                                              \
  TYPE work_query = (TYPE)0;                                                    \
  TYPE *a_col = NULL;                                                           \
  TYPE *tau_copy = NULL;                                                        \
  TYPE *c_col = NULL;                                                           \
  TYPE *work = NULL;                                                            \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (fb_lapack_side_to_char(side, &side_char) != 0 ||                         \
      fb_lapack_transpose_to_char(trans, &trans_char) != 0) return -1;         \
  a_rows = (side == FB_LEFT) ? m : n;                                           \
  if ((a_rows > 0 && k > 0 && !A) || (k > 0 && !tau) ||                        \
      (m > 0 && n > 0 && !C)) return -1;                                       \
  lda_ = (a_rows > 0) ? a_rows : 1;                                             \
  a_elems = (size_t)lda_ * (size_t)((k > 0) ? k : 1);                          \
  c_elems = (size_t)ldc_ * (size_t)((n > 0) ? n : 1);                          \
  a_col = (TYPE *)malloc(a_elems * sizeof(TYPE));                               \
  tau_copy = (TYPE *)malloc((size_t)tau_alloc * sizeof(TYPE));                  \
  c_col = (TYPE *)malloc(c_elems * sizeof(TYPE));                               \
  if (!a_col || !tau_copy || !c_col) {                                          \
    free(a_col);                                                                \
    free(tau_copy);                                                             \
    free(c_col);                                                                \
    return -101;                                                                \
  }                                                                             \
  if ((a_rows > 0 && k > 0 &&                                                   \
       fb_copy_matrix_to_col_major(A, layout, a_rows, k, lda,                  \
                                   a_col, lda_, sizeof(TYPE)) != 0) ||         \
      (m > 0 && n > 0 &&                                                        \
       fb_copy_matrix_to_col_major(C, layout, m, n, ldc,                       \
                                   c_col, ldc_, sizeof(TYPE)) != 0)) {         \
    free(a_col);                                                                \
    free(tau_copy);                                                             \
    free(c_col);                                                                \
    return -1;                                                                  \
  }                                                                             \
  memset(tau_copy, 0, (size_t)tau_alloc * sizeof(TYPE));                        \
  if (k > 0) {                                                                  \
    memcpy(tau_copy, tau, (size_t)k * sizeof(TYPE));                            \
  }                                                                             \
  ((void (*)(char *, char *, int *, int *, int *, TYPE *, int *, TYPE *,       \
             TYPE *, int *, TYPE *, int *, int *))g_fortran_fn[OP_ID])(        \
      &side_char, &trans_char, &m_, &n_, &k_, a_col, &lda_, tau_copy, c_col,  \
      &ldc_, &work_query, &lwork_, &info);                                      \
  if (info != 0) {                                                              \
    free(a_col);                                                                \
    free(tau_copy);                                                             \
    free(c_col);                                                                \
    return info;                                                                \
  }                                                                             \
  lwork_ = QUERY_FN(work_query);                                                \
  work = (TYPE *)malloc((size_t)lwork_ * sizeof(TYPE));                         \
  if (!work) {                                                                  \
    free(a_col);                                                                \
    free(tau_copy);                                                             \
    free(c_col);                                                                \
    return -101;                                                                \
  }                                                                             \
  if ((a_rows > 0 && k > 0 &&                                                   \
       fb_copy_matrix_to_col_major(A, layout, a_rows, k, lda,                  \
                                   a_col, lda_, sizeof(TYPE)) != 0) ||         \
      (m > 0 && n > 0 &&                                                        \
       fb_copy_matrix_to_col_major(C, layout, m, n, ldc,                       \
                                   c_col, ldc_, sizeof(TYPE)) != 0)) {         \
    free(a_col);                                                                \
    free(tau_copy);                                                             \
    free(c_col);                                                                \
    free(work);                                                                 \
    return -1;                                                                  \
  }                                                                             \
  memset(tau_copy, 0, (size_t)tau_alloc * sizeof(TYPE));                        \
  if (k > 0) {                                                                  \
    memcpy(tau_copy, tau, (size_t)k * sizeof(TYPE));                            \
  }                                                                             \
  ((void (*)(char *, char *, int *, int *, int *, TYPE *, int *, TYPE *,       \
             TYPE *, int *, TYPE *, int *, int *))g_fortran_fn[OP_ID])(        \
      &side_char, &trans_char, &m_, &n_, &k_, a_col, &lda_, tau_copy, c_col,  \
      &ldc_, work, &lwork_, &info);                                             \
  if (m > 0 && n > 0 &&                                                         \
      fb_copy_matrix_from_col_major(c_col, m, n, ldc_, layout,                 \
                                    C, ldc, sizeof(TYPE)) != 0) {              \
    free(a_col);                                                                \
    free(tau_copy);                                                             \
    free(c_col);                                                                \
    free(work);                                                                 \
    return -1;                                                                  \
  }                                                                             \
  free(a_col);                                                                  \
  free(tau_copy);                                                               \
  free(c_col);                                                                  \
  free(work);                                                                   \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    char *side, char *trans, int *m, int *n, int *k, TYPE *A, int *lda,       \
    TYPE *tau, TYPE *C, int *ldc, TYPE *work, int *lwork, int *info) {         \
  fb_side_t side_;                                                              \
  fb_transpose_t trans_;                                                        \
  (void)work;                                                                   \
  (void)lwork;                                                                  \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (fb_lapack_char_to_side(*side, &side_) != 0 ||                            \
      fb_lapack_char_to_transpose(*trans, &trans_) != 0) {                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, fb_side_t, fb_transpose_t, int, int, int,   \
                     const TYPE *, int, const TYPE *, TYPE *, int))            \
      g_cblas_fn[OP_ID])(FB_LAYOUT_COL_MAJOR, side_, trans_, *m, *n, *k,       \
                          A, *lda, tau, C, *ldc);                               \
  if (info) *info = rc;                                                         \
}

#define DEFINE_LAPACK_RZ_APPLY_THUNKS(OP, TYPE, OP_ID)                        \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const fb_side_t side,                            \
    const fb_transpose_t trans, const int m, const int n, const int k,         \
    const int l, const TYPE *A, const int lda, const TYPE *tau, TYPE *C,       \
    const int ldc) {                                                            \
  char side_char = 'L';                                                         \
  char trans_char = 'N';                                                        \
  int a_rows = (k > 0) ? k : 1;                                                 \
  int m_ = m, n_ = n, k_ = k, l_ = l;                                        \
  int a_cols = (side == FB_LEFT) ? m : n;                                       \
  int lda_ = (k > 0) ? k : 1;                                                   \
  int ldc_ = (m > 0) ? m : 1;                                                   \
  int info = 0, tau_alloc = (k > 0) ? k : 1;                                   \
  int work_len = (side == FB_LEFT) ? ((n > 0) ? n : 1) :                       \
                                     ((m > 0) ? m : 1);                        \
  size_t a_elems = (size_t)lda_ * (size_t)((a_cols > 0) ? a_cols : 1);         \
  size_t c_elems = (size_t)ldc_ * (size_t)((n > 0) ? n : 1);                   \
  TYPE *a_col = NULL;                                                           \
  TYPE *tau_copy = NULL;                                                        \
  TYPE *c_col = NULL;                                                           \
  TYPE *work = NULL;                                                            \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (fb_lapack_side_to_char(side, &side_char) != 0 ||                         \
      fb_lapack_transpose_to_char(trans, &trans_char) != 0) return -1;         \
  if ((k > 0 && a_cols > 0 && !A) || (k > 0 && !tau) ||                        \
      (m > 0 && n > 0 && !C)) return -1;                                       \
  a_col = (TYPE *)malloc(a_elems * sizeof(TYPE));                               \
  tau_copy = (TYPE *)malloc((size_t)tau_alloc * sizeof(TYPE));                  \
  c_col = (TYPE *)malloc(c_elems * sizeof(TYPE));                               \
  work = (TYPE *)malloc((size_t)work_len * sizeof(TYPE));                       \
  if (!a_col || !tau_copy || !c_col || !work) {                                 \
    free(a_col);                                                                \
    free(tau_copy);                                                             \
    free(c_col);                                                                \
    free(work);                                                                 \
    return -101;                                                                \
  }                                                                             \
  if ((k > 0 && a_cols > 0 &&                                                   \
       fb_copy_matrix_to_col_major(A, layout, k, a_cols, lda,                  \
                                   a_col, lda_, sizeof(TYPE)) != 0) ||         \
      (m > 0 && n > 0 &&                                                        \
       fb_copy_matrix_to_col_major(C, layout, m, n, ldc,                       \
                                   c_col, ldc_, sizeof(TYPE)) != 0)) {         \
    free(a_col);                                                                \
    free(tau_copy);                                                             \
    free(c_col);                                                                \
    free(work);                                                                 \
    return -1;                                                                  \
  }                                                                             \
  memset(tau_copy, 0, (size_t)tau_alloc * sizeof(TYPE));                        \
  if (k > 0) {                                                                  \
    memcpy(tau_copy, tau, (size_t)k * sizeof(TYPE));                            \
  }                                                                             \
  ((void (*)(char *, char *, int *, int *, int *, int *, TYPE *, int *,       \
             TYPE *, TYPE *, int *, TYPE *, int *))g_fortran_fn[OP_ID])(      \
      &side_char, &trans_char, &m_, &n_, &k_, &l_, a_col, &lda_, tau_copy, c_col,\
      &ldc_, work, &info);                                                      \
  if (m > 0 && n > 0 &&                                                         \
      fb_copy_matrix_from_col_major(c_col, m, n, ldc_, layout,                 \
                                    C, ldc, sizeof(TYPE)) != 0) {              \
    free(a_col);                                                                \
    free(tau_copy);                                                             \
    free(c_col);                                                                \
    free(work);                                                                 \
    return -1;                                                                  \
  }                                                                             \
  free(a_col);                                                                  \
  free(tau_copy);                                                               \
  free(c_col);                                                                  \
  free(work);                                                                   \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    char *side, char *trans, int *m, int *n, int *k, int *l, TYPE *A,         \
    int *lda, TYPE *tau, TYPE *C, int *ldc, TYPE *work, int *info) {          \
  fb_side_t side_;                                                              \
  fb_transpose_t trans_;                                                        \
  (void)work;                                                                   \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (fb_lapack_char_to_side(*side, &side_) != 0 ||                            \
      fb_lapack_char_to_transpose(*trans, &trans_) != 0) {                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, fb_side_t, fb_transpose_t, int, int, int,   \
                     int, const TYPE *, int, const TYPE *, TYPE *, int))      \
      g_cblas_fn[OP_ID])(FB_LAYOUT_COL_MAJOR, side_, trans_, *m, *n, *k, *l,   \
                          A, *lda, tau, C, *ldc);                               \
  if (info) *info = rc;                                                         \
}

#define DEFINE_LAPACK_RZ_APPLY_QUERY_THUNKS(OP, TYPE, OP_ID, QUERY_FN)        \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const fb_side_t side,                            \
    const fb_transpose_t trans, const int m, const int n, const int k,         \
    const int l, const TYPE *A, const int lda, const TYPE *tau, TYPE *C,       \
    const int ldc) {                                                            \
  char side_char = 'L';                                                         \
  char trans_char = 'N';                                                        \
  int a_rows = (k > 0) ? k : 1;                                                 \
  int m_ = m, n_ = n, k_ = k, l_ = l;                                        \
  int a_cols = (side == FB_LEFT) ? m : n;                                       \
  int lda_ = (k > 0) ? k : 1;                                                   \
  int ldc_ = (m > 0) ? m : 1;                                                   \
  int info = 0, tau_alloc = (k > 0) ? k : 1;                                   \
  int lwork_ = -1;                                                              \
  size_t a_elems = (size_t)lda_ * (size_t)((a_cols > 0) ? a_cols : 1);         \
  size_t c_elems = (size_t)ldc_ * (size_t)((n > 0) ? n : 1);                   \
  TYPE work_query = (TYPE)0;                                                    \
  TYPE *a_col = NULL;                                                           \
  TYPE *tau_copy = NULL;                                                        \
  TYPE *c_col = NULL;                                                           \
  TYPE *work = NULL;                                                            \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (fb_lapack_side_to_char(side, &side_char) != 0 ||                         \
      fb_lapack_transpose_to_char(trans, &trans_char) != 0) return -1;         \
  if ((k > 0 && a_cols > 0 && !A) || (k > 0 && !tau) ||                        \
      (m > 0 && n > 0 && !C)) return -1;                                       \
  a_col = (TYPE *)malloc(a_elems * sizeof(TYPE));                               \
  tau_copy = (TYPE *)malloc((size_t)tau_alloc * sizeof(TYPE));                  \
  c_col = (TYPE *)malloc(c_elems * sizeof(TYPE));                               \
  if (!a_col || !tau_copy || !c_col) {                                          \
    free(a_col);                                                                \
    free(tau_copy);                                                             \
    free(c_col);                                                                \
    return -101;                                                                \
  }                                                                             \
  if ((k > 0 && a_cols > 0 &&                                                   \
       fb_copy_matrix_to_col_major(A, layout, a_rows, a_cols, lda,             \
                                   a_col, lda_, sizeof(TYPE)) != 0) ||         \
      (m > 0 && n > 0 &&                                                        \
       fb_copy_matrix_to_col_major(C, layout, m, n, ldc,                       \
                                   c_col, ldc_, sizeof(TYPE)) != 0)) {         \
    free(a_col);                                                                \
    free(tau_copy);                                                             \
    free(c_col);                                                                \
    return -1;                                                                  \
  }                                                                             \
  memset(tau_copy, 0, (size_t)tau_alloc * sizeof(TYPE));                        \
  if (k > 0) {                                                                  \
    memcpy(tau_copy, tau, (size_t)k * sizeof(TYPE));                            \
  }                                                                             \
  ((void (*)(char *, char *, int *, int *, int *, int *, TYPE *, int *,       \
             TYPE *, TYPE *, int *, TYPE *, int *, int *))                    \
      g_fortran_fn[OP_ID])(                                                     \
      &side_char, &trans_char, &m_, &n_, &k_, &l_, a_col, &lda_, tau_copy, c_col,\
      &ldc_, &work_query, &lwork_, &info);                                      \
  if (info != 0) {                                                              \
    free(a_col);                                                                \
    free(tau_copy);                                                             \
    free(c_col);                                                                \
    return info;                                                                \
  }                                                                             \
  lwork_ = QUERY_FN(work_query);                                                \
  work = (TYPE *)malloc((size_t)lwork_ * sizeof(TYPE));                         \
  if (!work) {                                                                  \
    free(a_col);                                                                \
    free(tau_copy);                                                             \
    free(c_col);                                                                \
    return -101;                                                                \
  }                                                                             \
  if ((k > 0 && a_cols > 0 &&                                                   \
       fb_copy_matrix_to_col_major(A, layout, a_rows, a_cols, lda,             \
                                   a_col, lda_, sizeof(TYPE)) != 0) ||         \
      (m > 0 && n > 0 &&                                                        \
       fb_copy_matrix_to_col_major(C, layout, m, n, ldc,                       \
                                   c_col, ldc_, sizeof(TYPE)) != 0)) {         \
    free(a_col);                                                                \
    free(tau_copy);                                                             \
    free(c_col);                                                                \
    free(work);                                                                 \
    return -1;                                                                  \
  }                                                                             \
  memset(tau_copy, 0, (size_t)tau_alloc * sizeof(TYPE));                        \
  if (k > 0) {                                                                  \
    memcpy(tau_copy, tau, (size_t)k * sizeof(TYPE));                            \
  }                                                                             \
  ((void (*)(char *, char *, int *, int *, int *, int *, TYPE *, int *,       \
             TYPE *, TYPE *, int *, TYPE *, int *, int *))                    \
      g_fortran_fn[OP_ID])(                                                     \
      &side_char, &trans_char, &m_, &n_, &k_, &l_, a_col, &lda_, tau_copy, c_col,\
      &ldc_, work, &lwork_, &info);                                             \
  if (m > 0 && n > 0 &&                                                         \
      fb_copy_matrix_from_col_major(c_col, m, n, ldc_, layout,                 \
                                    C, ldc, sizeof(TYPE)) != 0) {              \
    free(a_col);                                                                \
    free(tau_copy);                                                             \
    free(c_col);                                                                \
    free(work);                                                                 \
    return -1;                                                                  \
  }                                                                             \
  free(a_col);                                                                  \
  free(tau_copy);                                                               \
  free(c_col);                                                                  \
  free(work);                                                                   \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    char *side, char *trans, int *m, int *n, int *k, int *l, TYPE *A,         \
    int *lda, TYPE *tau, TYPE *C, int *ldc, TYPE *work, int *lwork,           \
    int *info) {                                                                \
  fb_side_t side_;                                                              \
  fb_transpose_t trans_;                                                        \
  (void)work;                                                                   \
  (void)lwork;                                                                  \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (fb_lapack_char_to_side(*side, &side_) != 0 ||                            \
      fb_lapack_char_to_transpose(*trans, &trans_) != 0) {                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, fb_side_t, fb_transpose_t, int, int, int,   \
                     int, const TYPE *, int, const TYPE *, TYPE *, int))      \
      g_cblas_fn[OP_ID])(FB_LAYOUT_COL_MAJOR, side_, trans_, *m, *n, *k, *l,   \
                          A, *lda, tau, C, *ldc);                               \
  if (info) *info = rc;                                                         \
}

#define DEFINE_LAPACK_LARFB_THUNKS(OP, TYPE, OP_ID)                            \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const fb_side_t side,                            \
    const fb_transpose_t trans, const char direct, const char storev,          \
    const int m, const int n, const int k, const TYPE *v, const int ldv,       \
    const TYPE *t, const int ldt, TYPE *c, const int ldc) {                    \
  char side_char = 'L';                                                        \
  char trans_char = 'N';                                                       \
  char direct_char = direct;                                                   \
  char storev_char = storev;                                                   \
  char uplo_char = 'L';                                                        \
  int forward = 0;                                                             \
  int backward = 0;                                                            \
  int col = 0;                                                                 \
  int row_storev = 0;                                                          \
  int left = (side == FB_LEFT);                                                \
  int right = (side == FB_RIGHT);                                              \
  int nrows_v = 0;                                                             \
  int ncols_v = 0;                                                             \
  int m_ = m;                                                                  \
  int n_ = n;                                                                  \
  int k_ = k;                                                                  \
  int ldv_ = 1;                                                                \
  int ldt_ = (k > 0) ? k : 1;                                                  \
  int ldc_ = (m > 0) ? m : 1;                                                  \
  int ldwork_ = left ? ((n > 0) ? n : 1) : ((m > 0) ? m : 1);                 \
  size_t v_elems = 0;                                                          \
  size_t t_elems = 0;                                                          \
  size_t c_elems = 0;                                                          \
  size_t work_elems = 0;                                                       \
  TYPE *v_col = NULL;                                                          \
  TYPE *t_col = NULL;                                                          \
  TYPE *c_col = NULL;                                                          \
  TYPE *work = NULL;                                                           \
  if (!g_fortran_fn[OP_ID]) return -1;                                         \
  if (!left && !right) return -1;                                              \
  if (fb_lapack_side_to_char(side, &side_char) != 0 ||                         \
      fb_lapack_transpose_to_char(trans, &trans_char) != 0) return -1;         \
  forward = fb_lapack_job_is(direct, 'F');                                     \
  backward = fb_lapack_job_is(direct, 'B');                                    \
  col = fb_lapack_job_is(storev, 'C');                                         \
  row_storev = fb_lapack_job_is(storev, 'R');                                  \
  if ((!forward && !backward) || (!col && !row_storev)) return -1;             \
  uplo_char = ((forward && col) || (!forward && !col)) ? 'L' : 'U';            \
  nrows_v = (col && left) ? m : ((col && right) ? n : k);                      \
  ncols_v = (!col && left) ? m : ((!col && right) ? n : k);                    \
  if ((m > 0 && n > 0 && !c) ||                                                \
      (k > 0 && ((nrows_v > 0 && ncols_v > 0 && !v) || !t))) return -1;       \
  ldv_ = (nrows_v > 0) ? nrows_v : 1;                                          \
  v_elems = (size_t)ldv_ * (size_t)((ncols_v > 0) ? ncols_v : 1);              \
  t_elems = (size_t)ldt_ * (size_t)((k > 0) ? k : 1);                          \
  c_elems = (size_t)ldc_ * (size_t)((n > 0) ? n : 1);                          \
  work_elems = (size_t)ldwork_ * (size_t)((k > 0) ? k : 1);                    \
  work = (TYPE *)malloc(work_elems * sizeof(TYPE));                            \
  if (!work) return -101;                                                      \
  if (fb_lapack_layout_is_col_major(layout)) {                                 \
    ((void (*)(char *, char *, char *, char *, int *, int *, int *,            \
               TYPE *, int *, TYPE *, int *, TYPE *, int *, TYPE *,            \
               int *))g_fortran_fn[OP_ID])(                                    \
        &side_char, &trans_char, &direct_char, &storev_char,                   \
        &m_, &n_, &k_, (TYPE *)v, (int *)&ldv, (TYPE *)t, (int *)&ldt,         \
        c, (int *)&ldc, work, &ldwork_);                                       \
    free(work);                                                                 \
    return 0;                                                                   \
  }                                                                            \
  if (layout != FB_LAYOUT_ROW_MAJOR) {                                         \
    free(work);                                                                 \
    return -1;                                                                  \
  }                                                                            \
  if ((m > 0 && ldc < n) || (k > 0 && ldt < k) ||                              \
      (ncols_v > 0 && ldv < ncols_v) ||                                        \
      ((col && k > nrows_v) || (row_storev && k > ncols_v))) {                 \
    free(work);                                                                 \
    return -1;                                                                  \
  }                                                                            \
  v_col = (TYPE *)malloc(v_elems * sizeof(TYPE));                               \
  t_col = (TYPE *)malloc(t_elems * sizeof(TYPE));                               \
  c_col = (TYPE *)malloc(c_elems * sizeof(TYPE));                               \
  if (!v_col || !t_col || !c_col) {                                             \
    free(v_col);                                                                 \
    free(t_col);                                                                 \
    free(c_col);                                                                 \
    free(work);                                                                  \
    return -101;                                                                 \
  }                                                                             \
  if ((k > 0 && nrows_v > 0 && ncols_v > 0 &&                                   \
       fb_copy_row_major_trapezoid_to_col_major(v, direct_char, uplo_char,     \
                                                'U', nrows_v, ncols_v, ldv,     \
                                                v_col, ldv_, sizeof(TYPE)) != 0) || \
      (k > 0 &&                                                                   \
       fb_copy_matrix_to_col_major(t, layout, k, k, ldt,                        \
                                   t_col, ldt_, sizeof(TYPE)) != 0) ||          \
      (m > 0 && n > 0 &&                                                         \
       fb_copy_matrix_to_col_major(c, layout, m, n, ldc,                        \
                                   c_col, ldc_, sizeof(TYPE)) != 0)) {          \
    free(v_col);                                                                 \
    free(t_col);                                                                 \
    free(c_col);                                                                 \
    free(work);                                                                  \
    return -1;                                                                   \
  }                                                                              \
  ((void (*)(char *, char *, char *, char *, int *, int *, int *,              \
             TYPE *, int *, TYPE *, int *, TYPE *, int *, TYPE *,              \
             int *))g_fortran_fn[OP_ID])(                                       \
      &side_char, &trans_char, &direct_char, &storev_char,                      \
      &m_, &n_, &k_, v_col, &ldv_, t_col, &ldt_, c_col, &ldc_,                 \
      work, &ldwork_);                                                          \
  if (m > 0 && n > 0 &&                                                          \
      fb_copy_matrix_from_col_major(c_col, m, n, ldc_, layout,                 \
                                    c, ldc, sizeof(TYPE)) != 0) {              \
    free(v_col);                                                                 \
    free(t_col);                                                                 \
    free(c_col);                                                                 \
    free(work);                                                                  \
    return -1;                                                                   \
  }                                                                              \
  free(v_col);                                                                   \
  free(t_col);                                                                   \
  free(c_col);                                                                   \
  free(work);                                                                    \
  return 0;                                                                      \
}                                                                                \
static void thunk_##OP##_cblas_to_fortran(                                      \
    char *side, char *trans, char *direct, char *storev, int *m, int *n,       \
    int *k, TYPE *v, int *ldv, TYPE *t, int *ldt, TYPE *c, int *ldc,          \
    TYPE *work, int *ldwork, int *info) {                                       \
  fb_side_t side_ = FB_LEFT;                                                     \
  fb_transpose_t trans_ = FB_NO_TRANS;                                           \
  int left = 0;                                                                  \
  int col = 0;                                                                   \
  int row_storev = 0;                                                            \
  int nrows_v = 0;                                                               \
  int ncols_v = 0;                                                               \
  int rc = -1;                                                                   \
  (void)work;                                                                    \
  (void)ldwork;                                                                  \
  if (!side || !trans || !direct || !storev || !m || !n || !k ||                \
      !ldv || !ldt || !ldc || !info) {                                           \
    return;                                                                      \
  }                                                                              \
  if (!g_cblas_fn[OP_ID]) {                                                      \
    *info = -1;                                                                  \
    return;                                                                      \
  }                                                                              \
  if (fb_lapack_char_to_side(*side, &side_) != 0 ||                              \
      fb_lapack_char_to_transpose(*trans, &trans_) != 0) {                       \
    *info = -1;                                                                  \
    return;                                                                      \
  }                                                                              \
  left = (side_ == FB_LEFT);                                                     \
  col = fb_lapack_job_is(*storev, 'C');                                          \
  row_storev = fb_lapack_job_is(*storev, 'R');                                   \
  if (!col && !row_storev) {                                                     \
    *info = -1;                                                                  \
    return;                                                                      \
  }                                                                              \
  nrows_v = (col && left) ? *m : ((col && !left) ? *n : *k);                    \
  ncols_v = (!col && left) ? *m : ((!col && !left) ? *n : *k);                  \
  if ((*m > 0 && *n > 0 && !c) ||                                                \
      (*k > 0 && ((nrows_v > 0 && ncols_v > 0 && !v) || !t))) {                 \
    *info = -1;                                                                  \
    return;                                                                      \
  }                                                                              \
  rc = ((int (*)(fb_layout_t, fb_side_t, fb_transpose_t, char, char,            \
                 int, int, int, const TYPE *, int, const TYPE *, int,           \
                 TYPE *, int))g_cblas_fn[OP_ID])(                                \
      FB_LAYOUT_COL_MAJOR, side_, trans_, *direct, *storev,                      \
      *m, *n, *k, v, *ldv, t, *ldt, c, *ldc);                                    \
  *info = rc;                                                                     \
}

#define DEFINE_LAPACK_BIDIAG_GENERATOR_THUNKS(OP, TYPE, OP_ID, QUERY_FN)      \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, char vect, const int m, const int n,             \
    const int k, TYPE *A, const int lda, const TYPE *tau) {                    \
  int m_ = m, n_ = n, k_ = k, lda_ = (m > 0) ? m : 1, lwork_ = -1, info = 0;   \
  int tau_alloc = (k > 0) ? k : 1;                                              \
  size_t a_elems = (size_t)lda_ * (size_t)((n > 0) ? n : 1);                   \
  TYPE work_query = (TYPE)0;                                                    \
  TYPE *a_col = NULL;                                                           \
  TYPE *tau_copy = NULL;                                                        \
  TYPE *work = NULL;                                                            \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if ((m > 0 && n > 0 && !A) || (k > 0 && !tau)) return -1;                    \
  a_col = (TYPE *)malloc(a_elems * sizeof(TYPE));                               \
  tau_copy = (TYPE *)malloc((size_t)tau_alloc * sizeof(TYPE));                  \
  if (!a_col || !tau_copy) {                                                    \
    free(a_col);                                                                \
    free(tau_copy);                                                             \
    return -101;                                                                \
  }                                                                             \
  if (m > 0 && n > 0 &&                                                         \
      fb_copy_matrix_to_col_major(A, layout, m, n, lda,                        \
                                  a_col, lda_, sizeof(TYPE)) != 0) {           \
    free(a_col);                                                                \
    free(tau_copy);                                                             \
    return -1;                                                                  \
  }                                                                             \
  memset(tau_copy, 0, (size_t)tau_alloc * sizeof(TYPE));                        \
  if (k > 0) {                                                                  \
    memcpy(tau_copy, tau, (size_t)k * sizeof(TYPE));                            \
  }                                                                             \
  ((void (*)(char *, int *, int *, int *, TYPE *, int *, TYPE *, TYPE *,       \
             int *, int *))g_fortran_fn[OP_ID])(                               \
      &vect, &m_, &n_, &k_, a_col, &lda_, tau_copy, &work_query, &lwork_,      \
      &info);                                                                   \
  if (info != 0) {                                                              \
    free(a_col);                                                                \
    free(tau_copy);                                                             \
    return info;                                                                \
  }                                                                             \
  lwork_ = QUERY_FN(work_query);                                                \
  work = (TYPE *)malloc((size_t)lwork_ * sizeof(TYPE));                         \
  if (!work) {                                                                  \
    free(a_col);                                                                \
    free(tau_copy);                                                             \
    return -101;                                                                \
  }                                                                             \
  if (m > 0 && n > 0 &&                                                         \
      fb_copy_matrix_to_col_major(A, layout, m, n, lda,                        \
                                  a_col, lda_, sizeof(TYPE)) != 0) {           \
    free(a_col);                                                                \
    free(tau_copy);                                                             \
    free(work);                                                                 \
    return -1;                                                                  \
  }                                                                             \
  memset(tau_copy, 0, (size_t)tau_alloc * sizeof(TYPE));                        \
  if (k > 0) {                                                                  \
    memcpy(tau_copy, tau, (size_t)k * sizeof(TYPE));                            \
  }                                                                             \
  ((void (*)(char *, int *, int *, int *, TYPE *, int *, TYPE *, TYPE *,       \
             int *, int *))g_fortran_fn[OP_ID])(                               \
      &vect, &m_, &n_, &k_, a_col, &lda_, tau_copy, work, &lwork_, &info);     \
  if (m > 0 && n > 0 &&                                                         \
      fb_copy_matrix_from_col_major(a_col, m, n, lda_, layout,                 \
                                    A, lda, sizeof(TYPE)) != 0) {              \
    free(a_col);                                                                \
    free(tau_copy);                                                             \
    free(work);                                                                 \
    return -1;                                                                  \
  }                                                                             \
  free(a_col);                                                                  \
  free(tau_copy);                                                               \
  free(work);                                                                   \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    char *vect, int *m, int *n, int *k, TYPE *A, int *lda,                     \
    TYPE *tau, TYPE *work, int *lwork, int *info) {                             \
  (void)work;                                                                   \
  (void)lwork;                                                                  \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, char, int, int, int, TYPE *, int,           \
                     const TYPE *))g_cblas_fn[OP_ID])(                          \
      FB_LAYOUT_COL_MAJOR, *vect, *m, *n, *k, A, *lda, tau);                   \
  if (info) *info = rc;                                                         \
}

#define DEFINE_LAPACK_BIDIAG_APPLY_THUNKS(OP, TYPE, OP_ID, QUERY_FN)          \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, char vect, char side, char trans,                \
    const int m, const int n, const int k, const TYPE *A, const int lda,       \
    const TYPE *tau, TYPE *C, const int ldc) {                                  \
  fb_side_t side_enum;                                                          \
  int m_ = m, n_ = n, k_ = k;                                              \
  int a_rows = 0, a_cols = 0, lda_ = 1, ldc_ = (m > 0) ? m : 1;                \
  int lwork_ = -1, info = 0, tau_alloc = (k > 0) ? k : 1;                      \
  size_t a_elems = 0, c_elems = 0;                                              \
  TYPE work_query = (TYPE)0;                                                    \
  TYPE *a_col = NULL;                                                           \
  TYPE *tau_copy = NULL;                                                        \
  TYPE *c_col = NULL;                                                           \
  TYPE *work = NULL;                                                            \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (fb_lapack_char_to_side(side, &side_enum) != 0 ||                         \
      fb_lapack_bidiag_apply_a_dims(vect, side_enum, m, n, k,                  \
                                    &a_rows, &a_cols) != 0) return -1;         \
  if ((a_rows > 0 && a_cols > 0 && !A) || (k > 0 && !tau) ||                   \
      (m > 0 && n > 0 && !C)) return -1;                                       \
  lda_ = (a_rows > 0) ? a_rows : 1;                                             \
  a_elems = (size_t)lda_ * (size_t)((a_cols > 0) ? a_cols : 1);                \
  c_elems = (size_t)ldc_ * (size_t)((n > 0) ? n : 1);                          \
  a_col = (TYPE *)malloc(a_elems * sizeof(TYPE));                               \
  tau_copy = (TYPE *)malloc((size_t)tau_alloc * sizeof(TYPE));                  \
  c_col = (TYPE *)malloc(c_elems * sizeof(TYPE));                               \
  if (!a_col || !tau_copy || !c_col) {                                          \
    free(a_col);                                                                \
    free(tau_copy);                                                             \
    free(c_col);                                                                \
    return -101;                                                                \
  }                                                                             \
  if ((a_rows > 0 && a_cols > 0 &&                                              \
       fb_copy_matrix_to_col_major(A, layout, a_rows, a_cols, lda,             \
                                   a_col, lda_, sizeof(TYPE)) != 0) ||         \
      (m > 0 && n > 0 &&                                                        \
       fb_copy_matrix_to_col_major(C, layout, m, n, ldc,                       \
                                   c_col, ldc_, sizeof(TYPE)) != 0)) {         \
    free(a_col);                                                                \
    free(tau_copy);                                                             \
    free(c_col);                                                                \
    return -1;                                                                  \
  }                                                                             \
  memset(tau_copy, 0, (size_t)tau_alloc * sizeof(TYPE));                        \
  if (k > 0) {                                                                  \
    memcpy(tau_copy, tau, (size_t)k * sizeof(TYPE));                            \
  }                                                                             \
  ((void (*)(char *, char *, char *, int *, int *, int *, TYPE *, int *,       \
             TYPE *, TYPE *, int *, TYPE *, int *, int *))                     \
      g_fortran_fn[OP_ID])(                                                     \
      &vect, &side, &trans, &m_, &n_, &k_, a_col, &lda_, tau_copy, c_col, &ldc_, \
      &work_query, &lwork_, &info);                                             \
  if (info != 0) {                                                              \
    free(a_col);                                                                \
    free(tau_copy);                                                             \
    free(c_col);                                                                \
    return info;                                                                \
  }                                                                             \
  lwork_ = QUERY_FN(work_query);                                                \
  work = (TYPE *)malloc((size_t)lwork_ * sizeof(TYPE));                         \
  if (!work) {                                                                  \
    free(a_col);                                                                \
    free(tau_copy);                                                             \
    free(c_col);                                                                \
    return -101;                                                                \
  }                                                                             \
  if ((a_rows > 0 && a_cols > 0 &&                                              \
       fb_copy_matrix_to_col_major(A, layout, a_rows, a_cols, lda,             \
                                   a_col, lda_, sizeof(TYPE)) != 0) ||         \
      (m > 0 && n > 0 &&                                                        \
       fb_copy_matrix_to_col_major(C, layout, m, n, ldc,                       \
                                   c_col, ldc_, sizeof(TYPE)) != 0)) {         \
    free(a_col);                                                                \
    free(tau_copy);                                                             \
    free(c_col);                                                                \
    free(work);                                                                 \
    return -1;                                                                  \
  }                                                                             \
  memset(tau_copy, 0, (size_t)tau_alloc * sizeof(TYPE));                        \
  if (k > 0) {                                                                  \
    memcpy(tau_copy, tau, (size_t)k * sizeof(TYPE));                            \
  }                                                                             \
  ((void (*)(char *, char *, char *, int *, int *, int *, TYPE *, int *,       \
             TYPE *, TYPE *, int *, TYPE *, int *, int *))                     \
      g_fortran_fn[OP_ID])(                                                     \
      &vect, &side, &trans, &m_, &n_, &k_, a_col, &lda_, tau_copy, c_col, &ldc_, \
      work, &lwork_, &info);                                                    \
  if (m > 0 && n > 0 &&                                                         \
      fb_copy_matrix_from_col_major(c_col, m, n, ldc_, layout,                 \
                                    C, ldc, sizeof(TYPE)) != 0) {              \
    free(a_col);                                                                \
    free(tau_copy);                                                             \
    free(c_col);                                                                \
    free(work);                                                                 \
    return -1;                                                                  \
  }                                                                             \
  free(a_col);                                                                  \
  free(tau_copy);                                                               \
  free(c_col);                                                                  \
  free(work);                                                                   \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    char *vect, char *side, char *trans, int *m, int *n, int *k,               \
    TYPE *A, int *lda, TYPE *tau, TYPE *C, int *ldc,                           \
    TYPE *work, int *lwork, int *info) {                                        \
  (void)work;                                                                   \
  (void)lwork;                                                                  \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, char, char, char, int, int, int,            \
                     const TYPE *, int, const TYPE *, TYPE *, int))            \
      g_cblas_fn[OP_ID])(FB_LAYOUT_COL_MAJOR, *vect, *side, *trans,            \
                          *m, *n, *k, A, *lda, tau, C, *ldc);                   \
  if (info) *info = rc;                                                         \
}

#define DEFINE_LAPACK_UPLO_INFO_MATRIX_THUNKS(OP, TYPE, OP_ID)                 \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const fb_uplo_t uplo, const int n, TYPE *A,      \
    const int lda) {                                                            \
  char uplo_char = fb_lapack_uplo_from_layout(layout, uplo);                   \
  int n_ = n, lda_ = lda, info = 0;                                             \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  ((void (*)(char *, int *, TYPE *, int *, int *))g_fortran_fn[OP_ID])(        \
      &uplo_char, &n_, A, &lda_, &info);                                       \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    char *uplo, int *n, TYPE *A, int *lda, int *info) {                        \
  fb_uplo_t uplo_ = FB_UPPER;                                                   \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!uplo || fb_lapack_char_to_uplo(*uplo, &uplo_) != 0) {                   \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, fb_uplo_t, int, TYPE *, int))                \
      g_cblas_fn[OP_ID])(FB_LAYOUT_COL_MAJOR, uplo_, *n, A, *lda);             \
  if (info) *info = rc;                                                         \
}

#define DEFINE_LAPACK_UPLO_GENERAL_RHS_THUNKS(OP, TYPE, OP_ID)                 \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const fb_uplo_t uplo, const int n,               \
    const int nrhs, TYPE *A, const int lda, TYPE *B, const int ldb) {          \
  char uplo_char = fb_lapack_uplo_from_layout(layout, uplo);                   \
  int n_ = n, nrhs_ = nrhs, lda_ = lda, ldb_ = ldb, info = 0;                  \
  TYPE *b_arg = B;                                                              \
  TYPE *b_col = NULL;                                                           \
  int b_col_ld = (n > 0) ? n : 1;                                               \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    size_t b_col_elems = (size_t)b_col_ld * (size_t)((nrhs > 0) ? nrhs : 1);   \
    if (nrhs > 0 && ldb < nrhs) return -1;                                      \
    b_col = (TYPE *)malloc(b_col_elems * sizeof(TYPE));                         \
    if (!b_col) return -101;                                                    \
    if (fb_copy_row_major_to_col_major(B, n, nrhs, ldb, b_col, b_col_ld,       \
                                       sizeof(TYPE)) != 0) {                    \
      free(b_col);                                                              \
      return -1;                                                                \
    }                                                                           \
    b_arg = b_col;                                                              \
    ldb_ = b_col_ld;                                                            \
  } else if (!fb_lapack_layout_is_col_major(layout)) {                          \
    return -1;                                                                  \
  }                                                                             \
  ((void (*)(char *, int *, int *, TYPE *, int *, TYPE *, int *, int *))       \
      g_fortran_fn[OP_ID])(&uplo_char, &n_, &nrhs_, A, &lda_, b_arg, &ldb_,    \
                           &info);                                              \
  if (b_col) {                                                                  \
    if (info == 0 &&                                                             \
        fb_copy_col_major_to_row_major(b_col, n, nrhs, ldb_, B, ldb,           \
                                       sizeof(TYPE)) != 0) {                    \
      free(b_col);                                                              \
      return -1;                                                                \
    }                                                                           \
    free(b_col);                                                                \
  }                                                                             \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    char *uplo, int *n, int *nrhs, TYPE *A, int *lda, TYPE *B, int *ldb,       \
    int *info) {                                                                \
  fb_uplo_t uplo_ = FB_UPPER;                                                   \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!uplo || fb_lapack_char_to_uplo(*uplo, &uplo_) != 0) {                   \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, fb_uplo_t, int, int, TYPE *, int, TYPE *,    \
                     int))g_cblas_fn[OP_ID])(FB_LAYOUT_COL_MAJOR, uplo_, *n,    \
                                             *nrhs, A, *lda, B, *ldb);         \
  if (info) *info = rc;                                                         \
}

#define DEFINE_LAPACK_PB_EQU_THUNKS(OP, MAT_TYPE, REAL_TYPE, OP_ID)           \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const fb_uplo_t uplo, const int n,               \
    const int kd, const MAT_TYPE *ab, const int ldab, REAL_TYPE *s,            \
    REAL_TYPE *scond, REAL_TYPE *amax) {                                        \
  char uplo_char = 'U';                                                         \
  int n_ = n, kd_ = kd, ldab_ = (kd >= 0) ? (kd + 1) : 1, info = 0;           \
  size_t ab_elems = (size_t)ldab_ * (size_t)((n > 0) ? n : 1);                 \
  MAT_TYPE *ab_col = NULL;                                                      \
  const MAT_TYPE *ab_arg = ab;                                                  \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (fb_lapack_uplo_to_char(uplo, &uplo_char) != 0) return -1;                \
  if (!s || !scond || !amax) return -1;                                         \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if ((n > 0 && !ab) || ldab < n) return -1;                                 \
    ab_col = (MAT_TYPE *)malloc(ab_elems * sizeof(MAT_TYPE));                   \
    if (!ab_col) return -101;                                                   \
    if (fb_copy_sym_band_to_col_major(ab, layout, uplo, n, kd, ldab,           \
                                      ab_col, ldab_, sizeof(MAT_TYPE)) != 0) { \
      free(ab_col);                                                             \
      return -1;                                                                \
    }                                                                           \
    ab_arg = ab_col;                                                            \
  } else if (fb_lapack_layout_is_col_major(layout)) {                           \
    ldab_ = ldab;                                                               \
    if (n > 0 && !ab) return -1;                                                \
  } else {                                                                      \
    return -1;                                                                  \
  }                                                                             \
  ((void (*)(char *, int *, int *, const MAT_TYPE *, int *, REAL_TYPE *,      \
             REAL_TYPE *, REAL_TYPE *, int *))g_fortran_fn[OP_ID])(            \
      &uplo_char, &n_, &kd_, ab_arg, &ldab_, s, scond, amax, &info);           \
  free(ab_col);                                                                 \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    char *uplo, int *n, int *kd, MAT_TYPE *ab, int *ldab, REAL_TYPE *s,       \
    REAL_TYPE *scond, REAL_TYPE *amax, int *info) {                             \
  fb_uplo_t uplo_ = FB_UPPER;                                                   \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!uplo || fb_lapack_char_to_uplo(*uplo, &uplo_) != 0) {                   \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, fb_uplo_t, int, int, const MAT_TYPE *, int, \
                     REAL_TYPE *, REAL_TYPE *, REAL_TYPE *))g_cblas_fn[OP_ID])(\
      FB_LAYOUT_COL_MAJOR, uplo_, *n, *kd, ab, *ldab, s, scond, amax);         \
  if (info) *info = rc;                                                         \
}

#define DEFINE_LAPACK_PP_EQU_THUNKS(OP, MAT_TYPE, REAL_TYPE, OP_ID)            \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const fb_uplo_t uplo, const int n,               \
    const MAT_TYPE *ap, REAL_TYPE *s, REAL_TYPE *scond, REAL_TYPE *amax) {     \
  char uplo_char = 'U';                                                         \
  int n_ = n, info = 0;                                                         \
  size_t packed_len = (size_t)((n > 0) ? n : 1) *                               \
                      (size_t)(((n + 1) > 0) ? (n + 1) : 2) / 2U;              \
  MAT_TYPE *ap_col = NULL;                                                      \
  const MAT_TYPE *ap_arg = ap;                                                  \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (fb_lapack_uplo_to_char(uplo, &uplo_char) != 0) return -1;                \
  if (!s || !scond || !amax) return -1;                                         \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if (n > 0 && !ap) return -1;                                                \
    ap_col = (MAT_TYPE *)malloc(packed_len * sizeof(MAT_TYPE));                 \
    if (!ap_col) return -101;                                                   \
    if (fb_copy_triangular_packed_to_col_major(ap, layout, uplo_char, 'N', n,  \
                                               ap_col, sizeof(MAT_TYPE)) != 0) {\
      free(ap_col);                                                             \
      return -1;                                                                \
    }                                                                           \
    ap_arg = ap_col;                                                            \
  } else if (fb_lapack_layout_is_col_major(layout)) {                           \
    if (n > 0 && !ap) return -1;                                                \
  } else {                                                                      \
    return -1;                                                                  \
  }                                                                             \
  ((void (*)(char *, int *, const MAT_TYPE *, REAL_TYPE *, REAL_TYPE *,        \
             REAL_TYPE *, int *))g_fortran_fn[OP_ID])(                         \
      &uplo_char, &n_, ap_arg, s, scond, amax, &info);                          \
  free(ap_col);                                                                 \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    char *uplo, int *n, MAT_TYPE *ap, REAL_TYPE *s, REAL_TYPE *scond,          \
    REAL_TYPE *amax, int *info) {                                               \
  fb_uplo_t uplo_ = FB_UPPER;                                                   \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!uplo || !n || !s || !scond || !amax ||                                   \
      fb_lapack_char_to_uplo(*uplo, &uplo_) != 0) {                            \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, fb_uplo_t, int, const MAT_TYPE *,            \
                     REAL_TYPE *, REAL_TYPE *, REAL_TYPE *))g_cblas_fn[OP_ID])(\
      FB_LAYOUT_COL_MAJOR, uplo_, *n, ap, s, scond, amax);                     \
  if (info) *info = rc;                                                         \
}

#define DEFINE_LAPACK_GB_EQU_THUNKS(OP, MAT_TYPE, REAL_TYPE, OP_ID)            \
static int thunk_##OP##_fortran_to_cblas(                                       \
    const fb_layout_t layout, const int m, const int n, const int kl,          \
    const int ku, const MAT_TYPE *ab, const int ldab, REAL_TYPE *r,            \
    REAL_TYPE *c, REAL_TYPE *rowcnd, REAL_TYPE *colcnd,                        \
    REAL_TYPE *amax) {                                                          \
  int m_ = m, n_ = n, kl_ = kl, ku_ = ku;                                       \
  int ldab_ = ((kl >= 0) && (ku >= 0)) ? (kl + ku + 1) : 1;                    \
  int info = 0;                                                                 \
  size_t ab_elems = (size_t)ldab_ * (size_t)((n > 0) ? n : 1);                 \
  MAT_TYPE *ab_col = NULL;                                                      \
  const MAT_TYPE *ab_arg = ab;                                                  \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (!r || !c || !rowcnd || !colcnd || !amax) return -1;                      \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if ((n > 0 && !ab) || ldab < n) return -1;                                 \
    ab_col = (MAT_TYPE *)malloc(ab_elems * sizeof(MAT_TYPE));                   \
    if (!ab_col) return -101;                                                   \
    if (fb_copy_general_band_to_col_major(ab, layout, m, n, kl, ku, ldab,      \
                                          ab_col, ldab_,                        \
                                          sizeof(MAT_TYPE)) != 0) {             \
      free(ab_col);                                                             \
      return -1;                                                                \
    }                                                                           \
    ab_arg = ab_col;                                                            \
  } else if (fb_lapack_layout_is_col_major(layout)) {                           \
    ldab_ = ldab;                                                               \
    if (n > 0 && !ab) return -1;                                                \
  } else {                                                                      \
    return -1;                                                                  \
  }                                                                             \
  ((void (*)(int *, int *, int *, int *, const MAT_TYPE *, int *,              \
             REAL_TYPE *, REAL_TYPE *, REAL_TYPE *, REAL_TYPE *,               \
             REAL_TYPE *, int *))g_fortran_fn[OP_ID])(                         \
      &m_, &n_, &kl_, &ku_, ab_arg, &ldab_, r, c, rowcnd, colcnd, amax,        \
      &info);                                                                   \
  free(ab_col);                                                                 \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    int *m, int *n, int *kl, int *ku, MAT_TYPE *ab, int *ldab, REAL_TYPE *r,   \
    REAL_TYPE *c, REAL_TYPE *rowcnd, REAL_TYPE *colcnd,                        \
    REAL_TYPE *amax, int *info) {                                               \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, int, int, int, int, const MAT_TYPE *, int,   \
                     REAL_TYPE *, REAL_TYPE *, REAL_TYPE *, REAL_TYPE *,       \
                     REAL_TYPE *))g_cblas_fn[OP_ID])(                          \
      FB_LAYOUT_COL_MAJOR, *m, *n, *kl, *ku, ab, *ldab, r, c, rowcnd,          \
      colcnd, amax);                                                            \
  if (info) *info = rc;                                                         \
}

#define DEFINE_LAPACK_GB_FACTOR_THUNKS(OP, MAT_TYPE, OP_ID)                    \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const int m, const int n, const int kl,         \
    const int ku, MAT_TYPE *ab, const int ldab, int *ipiv) {                  \
  int m_ = m, n_ = n, kl_ = kl, ku_ = ku;                                      \
  int ldab_ = ((kl >= 0) && (ku >= 0)) ? (2 * kl + ku + 1) : 1;               \
  int info = 0;                                                                \
  size_t ab_elems = (size_t)ldab_ * (size_t)((n > 0) ? n : 1);                \
  MAT_TYPE *ab_col = NULL;                                                     \
  MAT_TYPE *ab_arg = ab;                                                       \
  if (!g_fortran_fn[OP_ID]) return -1;                                         \
  if (!ipiv) return -1;                                                        \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                         \
    if ((n > 0 && !ab) || ldab < n) return -1;                                \
    ab_col = (MAT_TYPE *)malloc(ab_elems * sizeof(MAT_TYPE));                  \
    if (!ab_col) return -101;                                                  \
    if (fb_copy_general_band_to_col_major(ab, layout, m, n, kl, kl + ku,      \
                                          ldab, ab_col, ldab_,                 \
                                          sizeof(MAT_TYPE)) != 0) {            \
      free(ab_col);                                                            \
      return -1;                                                               \
    }                                                                          \
    ab_arg = ab_col;                                                           \
  } else if (fb_lapack_layout_is_col_major(layout)) {                          \
    ldab_ = ldab;                                                              \
    if (n > 0 && !ab) return -1;                                               \
  } else {                                                                     \
    return -1;                                                                 \
  }                                                                            \
  ((void (*)(int *, int *, int *, int *, MAT_TYPE *, int *, int *, int *))    \
      g_fortran_fn[OP_ID])(&m_, &n_, &kl_, &ku_, ab_arg, &ldab_, ipiv, &info);\
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                         \
    if (fb_copy_general_band_from_col_major(ab_col, m, n, kl, kl + ku, ldab_, \
                                            layout, ab, ldab,                  \
                                            sizeof(MAT_TYPE)) != 0) {          \
      free(ab_col);                                                            \
      return -1;                                                               \
    }                                                                          \
    free(ab_col);                                                              \
  }                                                                            \
  return info;                                                                 \
}                                                                              \
static void thunk_##OP##_cblas_to_fortran(                                     \
    int *m, int *n, int *kl, int *ku, MAT_TYPE *ab, int *ldab, int *ipiv,     \
    int *info) {                                                               \
  if (!g_cblas_fn[OP_ID]) {                                                    \
    if (info) *info = -1;                                                      \
    return;                                                                    \
  }                                                                            \
  if (!m || !n || !kl || !ku || !ldab || !ipiv || (*n > 0 && !ab)) {          \
    if (info) *info = -1;                                                      \
    return;                                                                    \
  }                                                                            \
  int rc = ((int (*)(fb_layout_t, int, int, int, int, MAT_TYPE *, int, int *))\
      g_cblas_fn[OP_ID])(FB_LAYOUT_COL_MAJOR, *m, *n, *kl, *ku, ab, *ldab,    \
                         ipiv);                                                \
  if (info) *info = rc;                                                        \
}

#define DEFINE_LAPACK_GB_FACTORED_SOLVE_THUNKS(OP, MAT_TYPE, OP_ID)           \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const char trans, const int n, const int kl,    \
    const int ku, const int nrhs, const MAT_TYPE *ab, const int ldab,         \
    const int *ipiv, MAT_TYPE *b, const int ldb) {                            \
  char trans_char = trans;                                                     \
  int n_ = n, kl_ = kl, ku_ = ku, nrhs_ = nrhs;                                \
  int ldab_ = ((kl >= 0) && (ku >= 0)) ? (2 * kl + ku + 1) : 1;               \
  int ldb_ = (n > 0) ? n : 1, info = 0;                                        \
  size_t ab_elems = (size_t)ldab_ * (size_t)((n > 0) ? n : 1);                \
  size_t b_elems = (size_t)ldb_ * (size_t)((nrhs > 0) ? nrhs : 1);            \
  MAT_TYPE *ab_col = NULL;                                                     \
  MAT_TYPE *b_col = NULL;                                                      \
  const MAT_TYPE *ab_arg = ab;                                                 \
  MAT_TYPE *b_arg = b;                                                         \
  if (!g_fortran_fn[OP_ID]) return -1;                                         \
  if (!ipiv) return -1;                                                        \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                         \
    if ((n > 0 && !ab) || (n > 0 && nrhs > 0 && !b) || ldab < n || ldb < nrhs) {\
      return -1;                                                               \
    }                                                                          \
    ab_col = (MAT_TYPE *)malloc(ab_elems * sizeof(MAT_TYPE));                  \
    b_col = (MAT_TYPE *)malloc(b_elems * sizeof(MAT_TYPE));                    \
    if (!ab_col || !b_col) {                                                   \
      free(ab_col);                                                            \
      free(b_col);                                                             \
      return -101;                                                             \
    }                                                                          \
    if (fb_copy_general_band_to_col_major(ab, layout, n, n, kl, kl + ku,      \
                                          ldab, ab_col, ldab_,                 \
                                          sizeof(MAT_TYPE)) != 0 ||            \
        fb_copy_matrix_to_col_major(b, layout, n, nrhs, ldb,                  \
                                    b_col, ldb_, sizeof(MAT_TYPE)) != 0) {     \
      free(ab_col);                                                            \
      free(b_col);                                                             \
      return -1;                                                               \
    }                                                                          \
    ab_arg = ab_col;                                                           \
    b_arg = b_col;                                                             \
  } else if (fb_lapack_layout_is_col_major(layout)) {                          \
    ldab_ = ldab;                                                              \
    ldb_ = ldb;                                                                \
    if ((n > 0 && !ab) || (n > 0 && nrhs > 0 && !b)) {                        \
      return -1;                                                               \
    }                                                                          \
  } else {                                                                     \
    return -1;                                                                 \
  }                                                                            \
  ((void (*)(char *, int *, int *, int *, int *, const MAT_TYPE *, int *,     \
             const int *, MAT_TYPE *, int *, int *))g_fortran_fn[OP_ID])(     \
      &trans_char, &n_, &kl_, &ku_, &nrhs_, ab_arg, &ldab_, ipiv, b_arg,      \
      &ldb_, &info);                                                           \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                         \
    if (fb_copy_matrix_from_col_major(b_col, n, nrhs, ldb_, layout,           \
                                      b, ldb, sizeof(MAT_TYPE)) != 0) {       \
      free(ab_col);                                                            \
      free(b_col);                                                             \
      return -1;                                                               \
    }                                                                          \
    free(ab_col);                                                              \
    free(b_col);                                                               \
  }                                                                            \
  return info;                                                                 \
}                                                                              \
static void thunk_##OP##_cblas_to_fortran(                                     \
    char *trans, int *n, int *kl, int *ku, int *nrhs, MAT_TYPE *ab,           \
    int *ldab, int *ipiv, MAT_TYPE *b, int *ldb, int *info) {                 \
  if (!g_cblas_fn[OP_ID]) {                                                    \
    if (info) *info = -1;                                                      \
    return;                                                                    \
  }                                                                            \
  if (!trans || !n || !kl || !ku || !nrhs || !ldab || !ipiv || !ldb ||        \
      (*n > 0 && !ab) || (*n > 0 && *nrhs > 0 && !b)) {                       \
    if (info) *info = -1;                                                      \
    return;                                                                    \
  }                                                                            \
  int rc =                                                                     \
      ((int (*)(fb_layout_t, char, int, int, int, int, const MAT_TYPE *, int, \
                const int *, MAT_TYPE *, int))g_cblas_fn[OP_ID])(             \
          FB_LAYOUT_COL_MAJOR, *trans, *n, *kl, *ku, *nrhs, ab, *ldab, ipiv, \
          b, *ldb);                                                            \
  if (info) *info = rc;                                                        \
}

#define DEFINE_LAPACK_GB_DIRECT_SOLVE_THUNKS(OP, MAT_TYPE, OP_ID)             \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const int n, const int kl, const int ku,        \
    const int nrhs, MAT_TYPE *ab, const int ldab, int *ipiv, MAT_TYPE *b,     \
    const int ldb) {                                                           \
  int n_ = n, kl_ = kl, ku_ = ku, nrhs_ = nrhs;                                \
  int ldab_ = ((kl >= 0) && (ku >= 0)) ? (2 * kl + ku + 1) : 1;               \
  int ldb_ = (n > 0) ? n : 1, info = 0;                                        \
  size_t ab_elems = (size_t)ldab_ * (size_t)((n > 0) ? n : 1);                \
  size_t b_elems = (size_t)ldb_ * (size_t)((nrhs > 0) ? nrhs : 1);            \
  MAT_TYPE *ab_col = NULL;                                                     \
  MAT_TYPE *b_col = NULL;                                                      \
  MAT_TYPE *ab_arg = ab;                                                       \
  MAT_TYPE *b_arg = b;                                                         \
  if (!g_fortran_fn[OP_ID]) return -1;                                         \
  if (!ipiv) return -1;                                                        \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                         \
    if ((n > 0 && !ab) || (n > 0 && nrhs > 0 && !b) || ldab < n || ldb < nrhs) {\
      return -1;                                                               \
    }                                                                          \
    ab_col = (MAT_TYPE *)malloc(ab_elems * sizeof(MAT_TYPE));                  \
    b_col = (MAT_TYPE *)malloc(b_elems * sizeof(MAT_TYPE));                    \
    if (!ab_col || !b_col) {                                                   \
      free(ab_col);                                                            \
      free(b_col);                                                             \
      return -101;                                                             \
    }                                                                          \
    if (fb_copy_general_band_to_col_major(ab, layout, n, n, kl, kl + ku,      \
                                          ldab, ab_col, ldab_,                 \
                                          sizeof(MAT_TYPE)) != 0 ||            \
        fb_copy_matrix_to_col_major(b, layout, n, nrhs, ldb,                  \
                                    b_col, ldb_, sizeof(MAT_TYPE)) != 0) {     \
      free(ab_col);                                                            \
      free(b_col);                                                             \
      return -1;                                                               \
    }                                                                          \
    ab_arg = ab_col;                                                           \
    b_arg = b_col;                                                             \
  } else if (fb_lapack_layout_is_col_major(layout)) {                          \
    ldab_ = ldab;                                                              \
    ldb_ = ldb;                                                                \
    if ((n > 0 && !ab) || (n > 0 && nrhs > 0 && !b)) {                        \
      return -1;                                                               \
    }                                                                          \
  } else {                                                                     \
    return -1;                                                                 \
  }                                                                            \
  ((void (*)(int *, int *, int *, int *, MAT_TYPE *, int *, int *, MAT_TYPE *,\
             int *, int *))g_fortran_fn[OP_ID])(                              \
      &n_, &kl_, &ku_, &nrhs_, ab_arg, &ldab_, ipiv, b_arg, &ldb_, &info);    \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                         \
    if (fb_copy_general_band_from_col_major(ab_col, n, n, kl, kl + ku, ldab_, \
                                            layout, ab, ldab,                  \
                                            sizeof(MAT_TYPE)) != 0 ||          \
        fb_copy_matrix_from_col_major(b_col, n, nrhs, ldb_, layout,           \
                                      b, ldb, sizeof(MAT_TYPE)) != 0) {       \
      free(ab_col);                                                            \
      free(b_col);                                                             \
      return -1;                                                               \
    }                                                                          \
    free(ab_col);                                                              \
    free(b_col);                                                               \
  }                                                                            \
  return info;                                                                 \
}                                                                              \
static void thunk_##OP##_cblas_to_fortran(                                     \
    int *n, int *kl, int *ku, int *nrhs, MAT_TYPE *ab, int *ldab, int *ipiv,  \
    MAT_TYPE *b, int *ldb, int *info) {                                        \
  if (!g_cblas_fn[OP_ID]) {                                                    \
    if (info) *info = -1;                                                      \
    return;                                                                    \
  }                                                                            \
  if (!n || !kl || !ku || !nrhs || !ldab || !ipiv || !ldb ||                  \
      (*n > 0 && !ab) || (*n > 0 && *nrhs > 0 && !b)) {                       \
    if (info) *info = -1;                                                      \
    return;                                                                    \
  }                                                                            \
  int rc =                                                                     \
      ((int (*)(fb_layout_t, int, int, int, int, MAT_TYPE *, int, int *,      \
                MAT_TYPE *, int))g_cblas_fn[OP_ID])(                          \
          FB_LAYOUT_COL_MAJOR, *n, *kl, *ku, *nrhs, ab, *ldab, ipiv, b,       \
          *ldb);                                                               \
  if (info) *info = rc;                                                        \
}

#define DEFINE_LAPACK_PB_CON_REAL_THUNKS(OP, REAL_TYPE, OP_ID)                \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const fb_uplo_t uplo, const int n,               \
    const int kd, const REAL_TYPE *ab, const int ldab, const REAL_TYPE anorm,  \
    REAL_TYPE *rcond) {                                                         \
  char uplo_char = 'U';                                                         \
  int n_ = n, kd_ = kd, ldab_ = (kd >= 0) ? (kd + 1) : 1, info = 0;           \
  int work_len = (n > 0) ? (3 * n) : 1;                                         \
  int iwork_len = (n > 0) ? n : 1;                                              \
  size_t ab_elems = (size_t)ldab_ * (size_t)((n > 0) ? n : 1);                 \
  REAL_TYPE *ab_col = NULL;                                                     \
  const REAL_TYPE *ab_arg = ab;                                                 \
  REAL_TYPE *work = NULL;                                                       \
  int *iwork = NULL;                                                            \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (fb_lapack_uplo_to_char(uplo, &uplo_char) != 0 || !rcond) return -1;      \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if ((n > 0 && !ab) || ldab < n) return -1;                                 \
    ab_col = (REAL_TYPE *)malloc(ab_elems * sizeof(REAL_TYPE));                 \
    if (!ab_col) return -101;                                                   \
    if (fb_copy_sym_band_to_col_major(ab, layout, uplo, n, kd, ldab,           \
                                      ab_col, ldab_, sizeof(REAL_TYPE)) != 0) {\
      free(ab_col);                                                             \
      return -1;                                                                \
    }                                                                           \
    ab_arg = ab_col;                                                            \
  } else if (fb_lapack_layout_is_col_major(layout)) {                           \
    ldab_ = ldab;                                                               \
    if (n > 0 && !ab) return -1;                                                \
  } else {                                                                      \
    return -1;                                                                  \
  }                                                                             \
  work = (REAL_TYPE *)malloc((size_t)work_len * sizeof(REAL_TYPE));            \
  iwork = (int *)malloc((size_t)iwork_len * sizeof(int));                      \
  if (!work || !iwork) {                                                        \
    free(ab_col);                                                               \
    free(work);                                                                 \
    free(iwork);                                                                \
    return -101;                                                                \
  }                                                                             \
  ((void (*)(char *, int *, int *, const REAL_TYPE *, int *, const REAL_TYPE *,\
             REAL_TYPE *, REAL_TYPE *, int *, int *))g_fortran_fn[OP_ID])(     \
      &uplo_char, &n_, &kd_, ab_arg, &ldab_, &anorm, rcond, work, iwork,       \
      &info);                                                                   \
  free(ab_col);                                                                 \
  free(work);                                                                   \
  free(iwork);                                                                  \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    char *uplo, int *n, int *kd, REAL_TYPE *ab, int *ldab, REAL_TYPE *anorm,   \
    REAL_TYPE *rcond, REAL_TYPE *work, int *iwork, int *info) {                \
  fb_uplo_t uplo_ = FB_UPPER;                                                   \
  (void)work;                                                                   \
  (void)iwork;                                                                  \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!uplo || !anorm || !rcond ||                                              \
      fb_lapack_char_to_uplo(*uplo, &uplo_) != 0) {                            \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, fb_uplo_t, int, int, const REAL_TYPE *, int, \
                     REAL_TYPE, REAL_TYPE *))g_cblas_fn[OP_ID])(               \
      FB_LAYOUT_COL_MAJOR, uplo_, *n, *kd, ab, *ldab, *anorm, rcond);          \
  if (info) *info = rc;                                                         \
}

#define DEFINE_LAPACK_PB_CON_COMPLEX_THUNKS(OP, MAT_TYPE, REAL_TYPE, OP_ID)   \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const fb_uplo_t uplo, const int n,               \
    const int kd, const MAT_TYPE *ab, const int ldab, const REAL_TYPE anorm,   \
    REAL_TYPE *rcond) {                                                         \
  char uplo_char = 'U';                                                         \
  int n_ = n, kd_ = kd, ldab_ = (kd >= 0) ? (kd + 1) : 1, info = 0;           \
  int work_len = (n > 0) ? (2 * n) : 1;                                         \
  int rwork_len = (n > 0) ? n : 1;                                              \
  size_t ab_elems = (size_t)ldab_ * (size_t)((n > 0) ? n : 1);                 \
  MAT_TYPE *ab_col = NULL;                                                      \
  const MAT_TYPE *ab_arg = ab;                                                  \
  MAT_TYPE *work = NULL;                                                        \
  REAL_TYPE *rwork = NULL;                                                      \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (fb_lapack_uplo_to_char(uplo, &uplo_char) != 0 || !rcond) return -1;      \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if ((n > 0 && !ab) || ldab < n) return -1;                                 \
    ab_col = (MAT_TYPE *)malloc(ab_elems * sizeof(MAT_TYPE));                   \
    if (!ab_col) return -101;                                                   \
    if (fb_copy_sym_band_to_col_major(ab, layout, uplo, n, kd, ldab,           \
                                      ab_col, ldab_, sizeof(MAT_TYPE)) != 0) { \
      free(ab_col);                                                             \
      return -1;                                                                \
    }                                                                           \
    ab_arg = ab_col;                                                            \
  } else if (fb_lapack_layout_is_col_major(layout)) {                           \
    ldab_ = ldab;                                                               \
    if (n > 0 && !ab) return -1;                                                \
  } else {                                                                      \
    return -1;                                                                  \
  }                                                                             \
  work = (MAT_TYPE *)malloc((size_t)work_len * sizeof(MAT_TYPE));              \
  rwork = (REAL_TYPE *)malloc((size_t)rwork_len * sizeof(REAL_TYPE));          \
  if (!work || !rwork) {                                                        \
    free(ab_col);                                                               \
    free(work);                                                                 \
    free(rwork);                                                                \
    return -101;                                                                \
  }                                                                             \
  ((void (*)(char *, int *, int *, const MAT_TYPE *, int *, const REAL_TYPE *, \
             REAL_TYPE *, MAT_TYPE *, REAL_TYPE *, int *))g_fortran_fn[OP_ID])(\
      &uplo_char, &n_, &kd_, ab_arg, &ldab_, &anorm, rcond, work, rwork,       \
      &info);                                                                   \
  free(ab_col);                                                                 \
  free(work);                                                                   \
  free(rwork);                                                                  \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    char *uplo, int *n, int *kd, MAT_TYPE *ab, int *ldab, REAL_TYPE *anorm,    \
    REAL_TYPE *rcond, MAT_TYPE *work, REAL_TYPE *rwork, int *info) {           \
  fb_uplo_t uplo_ = FB_UPPER;                                                   \
  (void)work;                                                                   \
  (void)rwork;                                                                  \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!uplo || !anorm || !rcond ||                                              \
      fb_lapack_char_to_uplo(*uplo, &uplo_) != 0) {                            \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, fb_uplo_t, int, int, const MAT_TYPE *, int,  \
                     REAL_TYPE, REAL_TYPE *))g_cblas_fn[OP_ID])(               \
      FB_LAYOUT_COL_MAJOR, uplo_, *n, *kd, ab, *ldab, *anorm, rcond);          \
  if (info) *info = rc;                                                         \
}

#define DEFINE_LAPACK_PP_CON_REAL_THUNKS(OP, REAL_TYPE, OP_ID)                 \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const fb_uplo_t uplo, const int n,               \
    const REAL_TYPE *ap, const REAL_TYPE anorm, REAL_TYPE *rcond) {            \
  char uplo_char = 'U';                                                         \
  int n_ = n, info = 0;                                                         \
  int work_len = (n > 0) ? (3 * n) : 1;                                         \
  int iwork_len = (n > 0) ? n : 1;                                              \
  size_t packed_len = (size_t)((n > 0) ? n : 1) *                               \
                      (size_t)(((n + 1) > 0) ? (n + 1) : 2) / 2U;              \
  REAL_TYPE *ap_col = NULL;                                                     \
  const REAL_TYPE *ap_arg = ap;                                                 \
  REAL_TYPE *work = NULL;                                                       \
  int *iwork = NULL;                                                            \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (fb_lapack_uplo_to_char(uplo, &uplo_char) != 0 || !rcond) return -1;      \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if (n > 0 && !ap) return -1;                                                \
    ap_col = (REAL_TYPE *)malloc(packed_len * sizeof(REAL_TYPE));               \
    if (!ap_col) return -101;                                                   \
    if (fb_copy_triangular_packed_to_col_major(ap, layout, uplo_char, 'N', n,  \
                                               ap_col, sizeof(REAL_TYPE)) != 0) {\
      free(ap_col);                                                             \
      return -1;                                                                \
    }                                                                           \
    ap_arg = ap_col;                                                            \
  } else if (fb_lapack_layout_is_col_major(layout)) {                           \
    if (n > 0 && !ap) return -1;                                                \
  } else {                                                                      \
    return -1;                                                                  \
  }                                                                             \
  work = (REAL_TYPE *)malloc((size_t)work_len * sizeof(REAL_TYPE));            \
  iwork = (int *)malloc((size_t)iwork_len * sizeof(int));                      \
  if (!work || !iwork) {                                                        \
    free(ap_col);                                                               \
    free(work);                                                                 \
    free(iwork);                                                                \
    return -101;                                                                \
  }                                                                             \
  ((void (*)(char *, int *, const REAL_TYPE *, const REAL_TYPE *, REAL_TYPE *, \
             REAL_TYPE *, int *, int *))g_fortran_fn[OP_ID])(                  \
      &uplo_char, &n_, ap_arg, &anorm, rcond, work, iwork, &info);             \
  free(ap_col);                                                                 \
  free(work);                                                                   \
  free(iwork);                                                                  \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    char *uplo, int *n, REAL_TYPE *ap, REAL_TYPE *anorm, REAL_TYPE *rcond,     \
    REAL_TYPE *work, int *iwork, int *info) {                                  \
  fb_uplo_t uplo_ = FB_UPPER;                                                   \
  (void)work;                                                                   \
  (void)iwork;                                                                  \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!uplo || !n || !anorm || !rcond ||                                        \
      fb_lapack_char_to_uplo(*uplo, &uplo_) != 0) {                            \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, fb_uplo_t, int, const REAL_TYPE *, REAL_TYPE,\
                     REAL_TYPE *))g_cblas_fn[OP_ID])(                           \
      FB_LAYOUT_COL_MAJOR, uplo_, *n, ap, *anorm, rcond);                      \
  if (info) *info = rc;                                                         \
}

#define DEFINE_LAPACK_PP_CON_COMPLEX_THUNKS(OP, MAT_TYPE, REAL_TYPE, OP_ID)    \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const fb_uplo_t uplo, const int n,               \
    const MAT_TYPE *ap, const REAL_TYPE anorm, REAL_TYPE *rcond) {             \
  char uplo_char = 'U';                                                         \
  int n_ = n, info = 0;                                                         \
  REAL_TYPE anorm_local = anorm;                                                \
  int work_len = (n > 0) ? n : 1;                                               \
  int rwork_len = (n > 0) ? n : 1;                                              \
  size_t packed_len = (size_t)((n > 0) ? n : 1) *                               \
                      (size_t)(((n + 1) > 0) ? (n + 1) : 2) / 2U;              \
  MAT_TYPE *ap_col = NULL;                                                      \
  const MAT_TYPE *ap_arg = ap;                                                  \
  MAT_TYPE *work = NULL;                                                        \
  REAL_TYPE *rwork = NULL;                                                      \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (fb_lapack_uplo_to_char(uplo, &uplo_char) != 0 || !rcond) return -1;      \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if (n > 0 && !ap) return -1;                                                \
    ap_col = (MAT_TYPE *)malloc(packed_len * sizeof(MAT_TYPE));                 \
    if (!ap_col) return -101;                                                   \
    if (fb_copy_triangular_packed_to_col_major(ap, layout, uplo_char, 'N', n,  \
                                               ap_col, sizeof(MAT_TYPE)) != 0) {\
      free(ap_col);                                                             \
      return -1;                                                                \
    }                                                                           \
    ap_arg = ap_col;                                                            \
  } else if (fb_lapack_layout_is_col_major(layout)) {                           \
    if (n > 0 && !ap) return -1;                                                \
  } else {                                                                      \
    return -1;                                                                  \
  }                                                                             \
  work = (MAT_TYPE *)malloc((size_t)work_len * sizeof(MAT_TYPE));              \
  rwork = (REAL_TYPE *)malloc((size_t)rwork_len * sizeof(REAL_TYPE));          \
  if (!work || !rwork) {                                                        \
    free(ap_col);                                                               \
    free(work);                                                                 \
    free(rwork);                                                                \
    return -101;                                                                \
  }                                                                             \
  ((void (*)(char *, int *, MAT_TYPE *, REAL_TYPE *, REAL_TYPE *, MAT_TYPE *,  \
             REAL_TYPE *, int *))g_fortran_fn[OP_ID])(                         \
      &uplo_char, &n_, (MAT_TYPE *)ap_arg, &anorm_local, rcond, work, rwork, &info); \
  free(ap_col);                                                                 \
  free(work);                                                                   \
  free(rwork);                                                                  \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    char *uplo, int *n, MAT_TYPE *ap, REAL_TYPE *anorm, REAL_TYPE *rcond,      \
    MAT_TYPE *work, REAL_TYPE *rwork, int *info) {                             \
  fb_uplo_t uplo_ = FB_UPPER;                                                   \
  (void)work;                                                                   \
  (void)rwork;                                                                  \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!uplo || !n || !anorm || !rcond ||                                        \
      fb_lapack_char_to_uplo(*uplo, &uplo_) != 0) {                            \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, fb_uplo_t, int, const MAT_TYPE *, REAL_TYPE, \
                     REAL_TYPE *))g_cblas_fn[OP_ID])(                           \
      FB_LAYOUT_COL_MAJOR, uplo_, *n, ap, *anorm, rcond);                      \
  if (info) *info = rc;                                                         \
}

#define DEFINE_LAPACK_SP_CON_REAL_THUNKS(OP, REAL_TYPE, OP_ID)                 \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const fb_uplo_t uplo, const int n,               \
    const REAL_TYPE *ap, const int *ipiv, const REAL_TYPE anorm,               \
    REAL_TYPE *rcond) {                                                         \
  char uplo_char = 'U';                                                         \
  int n_ = n, info = 0;                                                         \
  int work_len = (n > 0) ? (3 * n) : 1;                                         \
  int iwork_len = (n > 0) ? n : 1;                                              \
  size_t packed_len = (size_t)((n > 0) ? n : 1) *                               \
                      (size_t)(((n + 1) > 0) ? (n + 1) : 2) / 2U;              \
  REAL_TYPE *ap_col = NULL;                                                     \
  const REAL_TYPE *ap_arg = ap;                                                 \
  REAL_TYPE *work = NULL;                                                       \
  int *iwork = NULL;                                                            \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (fb_lapack_uplo_to_char(uplo, &uplo_char) != 0 || !ipiv || !rcond)        \
    return -1;                                                                  \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if (n > 0 && !ap) return -1;                                                \
    ap_col = (REAL_TYPE *)malloc(packed_len * sizeof(REAL_TYPE));               \
    if (!ap_col) return -101;                                                   \
    if (fb_copy_triangular_packed_to_col_major(ap, layout, uplo_char, 'N', n,  \
                                               ap_col, sizeof(REAL_TYPE)) != 0) {\
      free(ap_col);                                                             \
      return -1;                                                                \
    }                                                                           \
    ap_arg = ap_col;                                                            \
  } else if (fb_lapack_layout_is_col_major(layout)) {                           \
    if (n > 0 && !ap) return -1;                                                \
  } else {                                                                      \
    return -1;                                                                  \
  }                                                                             \
  work = (REAL_TYPE *)malloc((size_t)work_len * sizeof(REAL_TYPE));            \
  iwork = (int *)malloc((size_t)iwork_len * sizeof(int));                      \
  if (!work || !iwork) {                                                        \
    free(ap_col);                                                               \
    free(work);                                                                 \
    free(iwork);                                                                \
    return -101;                                                                \
  }                                                                             \
  ((void (*)(char *, int *, const REAL_TYPE *, const int *, const REAL_TYPE *, \
             REAL_TYPE *, REAL_TYPE *, int *, int *))g_fortran_fn[OP_ID])(     \
      &uplo_char, &n_, ap_arg, ipiv, &anorm, rcond, work, iwork, &info);       \
  free(ap_col);                                                                 \
  free(work);                                                                   \
  free(iwork);                                                                  \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    char *uplo, int *n, REAL_TYPE *ap, int *ipiv, REAL_TYPE *anorm,            \
    REAL_TYPE *rcond, REAL_TYPE *work, int *iwork, int *info) {                \
  fb_uplo_t uplo_ = FB_UPPER;                                                   \
  (void)work;                                                                   \
  (void)iwork;                                                                  \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!uplo || !n || !ipiv || !anorm || !rcond ||                               \
      fb_lapack_char_to_uplo(*uplo, &uplo_) != 0) {                            \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, fb_uplo_t, int, const REAL_TYPE *,           \
                     const int *, REAL_TYPE, REAL_TYPE *))g_cblas_fn[OP_ID])(  \
      FB_LAYOUT_COL_MAJOR, uplo_, *n, ap, ipiv, *anorm, rcond);                \
  if (info) *info = rc;                                                         \
}

#define DEFINE_LAPACK_SP_CON_COMPLEX_THUNKS(OP, MAT_TYPE, REAL_TYPE, OP_ID)   \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const fb_uplo_t uplo, const int n,               \
    const MAT_TYPE *ap, const int *ipiv, const REAL_TYPE anorm,                \
    REAL_TYPE *rcond) {                                                         \
  char uplo_char = 'U';                                                         \
  int n_ = n, info = 0;                                                         \
  REAL_TYPE anorm_local = anorm;                                                \
  int work_len = (n > 0) ? n : 1;                                               \
  size_t packed_len = (size_t)((n > 0) ? n : 1) *                               \
                      (size_t)(((n + 1) > 0) ? (n + 1) : 2) / 2U;              \
  MAT_TYPE *ap_col = NULL;                                                      \
  const MAT_TYPE *ap_arg = ap;                                                  \
  MAT_TYPE *work = NULL;                                                        \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (fb_lapack_uplo_to_char(uplo, &uplo_char) != 0 || !ipiv || !rcond)        \
    return -1;                                                                  \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if (n > 0 && !ap) return -1;                                                \
    ap_col = (MAT_TYPE *)malloc(packed_len * sizeof(MAT_TYPE));                 \
    if (!ap_col) return -101;                                                   \
    if (fb_copy_triangular_packed_to_col_major(ap, layout, uplo_char, 'N', n,  \
                                               ap_col, sizeof(MAT_TYPE)) != 0) {\
      free(ap_col);                                                             \
      return -1;                                                                \
    }                                                                           \
    ap_arg = ap_col;                                                            \
  } else if (fb_lapack_layout_is_col_major(layout)) {                           \
    if (n > 0 && !ap) return -1;                                                \
  } else {                                                                      \
    return -1;                                                                  \
  }                                                                             \
  work = (MAT_TYPE *)malloc((size_t)work_len * sizeof(MAT_TYPE));              \
  if (!work) {                                                                  \
    free(ap_col);                                                               \
    return -101;                                                                \
  }                                                                             \
  ((void (*)(char *, int *, MAT_TYPE *, int *, REAL_TYPE *, REAL_TYPE *,       \
             MAT_TYPE *, int *))g_fortran_fn[OP_ID])(                          \
      &uplo_char, &n_, (MAT_TYPE *)ap_arg, (int *)ipiv, &anorm_local, rcond, work,   \
      &info);                                                                   \
  free(ap_col);                                                                 \
  free(work);                                                                   \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    char *uplo, int *n, MAT_TYPE *ap, int *ipiv, REAL_TYPE *anorm,             \
    REAL_TYPE *rcond, MAT_TYPE *work, int *info) {                             \
  fb_uplo_t uplo_ = FB_UPPER;                                                   \
  (void)work;                                                                   \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!uplo || !n || !ipiv || !anorm || !rcond ||                               \
      fb_lapack_char_to_uplo(*uplo, &uplo_) != 0) {                            \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, fb_uplo_t, int, const MAT_TYPE *,            \
                     const int *, REAL_TYPE, REAL_TYPE *))g_cblas_fn[OP_ID])(  \
      FB_LAYOUT_COL_MAJOR, uplo_, *n, ap, ipiv, *anorm, rcond);                \
  if (info) *info = rc;                                                         \
}

#define DEFINE_LAPACK_HP_CON_COMPLEX_THUNKS(OP, MAT_TYPE, REAL_TYPE, OP_ID)   \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const fb_uplo_t uplo, const int n,               \
    const MAT_TYPE *ap, const int *ipiv, const REAL_TYPE anorm,                \
    REAL_TYPE *rcond) {                                                         \
  char uplo_char = 'U';                                                         \
  int n_ = n, info = 0;                                                         \
  REAL_TYPE anorm_local = anorm;                                                \
  int work_len = (n > 0) ? (2 * n) : 1;                                         \
  int rwork_len = (n > 0) ? n : 1;                                              \
  size_t packed_len = (size_t)((n > 0) ? n : 1) *                               \
                      (size_t)(((n + 1) > 0) ? (n + 1) : 2) / 2U;              \
  MAT_TYPE *ap_col = NULL;                                                      \
  const MAT_TYPE *ap_arg = ap;                                                  \
  MAT_TYPE *work = NULL;                                                        \
  REAL_TYPE *rwork = NULL;                                                      \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (fb_lapack_uplo_to_char(uplo, &uplo_char) != 0 || !ipiv || !rcond)        \
    return -1;                                                                  \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if (n > 0 && !ap) return -1;                                                \
    ap_col = (MAT_TYPE *)malloc(packed_len * sizeof(MAT_TYPE));                 \
    if (!ap_col) return -101;                                                   \
    if (fb_copy_triangular_packed_to_col_major(ap, layout, uplo_char, 'N', n,  \
                                               ap_col, sizeof(MAT_TYPE)) != 0) {\
      free(ap_col);                                                             \
      return -1;                                                                \
    }                                                                           \
    ap_arg = ap_col;                                                            \
  } else if (fb_lapack_layout_is_col_major(layout)) {                           \
    if (n > 0 && !ap) return -1;                                                \
  } else {                                                                      \
    return -1;                                                                  \
  }                                                                             \
  work = (MAT_TYPE *)malloc((size_t)work_len * sizeof(MAT_TYPE));              \
  rwork = (REAL_TYPE *)malloc((size_t)rwork_len * sizeof(REAL_TYPE));          \
  if (!work || !rwork) {                                                        \
    free(ap_col);                                                               \
    free(work);                                                                 \
    free(rwork);                                                                \
    return -101;                                                                \
  }                                                                             \
  ((void (*)(char *, int *, MAT_TYPE *, int *, REAL_TYPE *, REAL_TYPE *,       \
             MAT_TYPE *, REAL_TYPE *, int *))g_fortran_fn[OP_ID])(             \
      &uplo_char, &n_, (MAT_TYPE *)ap_arg, (int *)ipiv, &anorm_local, rcond, work,   \
      rwork, &info);                                                            \
  free(ap_col);                                                                 \
  free(work);                                                                   \
  free(rwork);                                                                  \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    char *uplo, int *n, MAT_TYPE *ap, int *ipiv, REAL_TYPE *anorm,             \
    REAL_TYPE *rcond, MAT_TYPE *work, REAL_TYPE *rwork, int *info) {           \
  fb_uplo_t uplo_ = FB_UPPER;                                                   \
  (void)work;                                                                   \
  (void)rwork;                                                                  \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!uplo || !n || !ipiv || !anorm || !rcond ||                               \
      fb_lapack_char_to_uplo(*uplo, &uplo_) != 0) {                            \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, fb_uplo_t, int, const MAT_TYPE *,            \
                     const int *, REAL_TYPE, REAL_TYPE *))g_cblas_fn[OP_ID])(  \
      FB_LAYOUT_COL_MAJOR, uplo_, *n, ap, ipiv, *anorm, rcond);                \
  if (info) *info = rc;                                                         \
}

#define DEFINE_LAPACK_PT_CON_REAL_THUNKS(OP, REAL_TYPE, OP_ID)                 \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const int n, const REAL_TYPE *d,                 \
    const REAL_TYPE *e, const REAL_TYPE anorm, REAL_TYPE *rcond) {             \
  int n_ = n;                                                                   \
  int info = 0;                                                                 \
  int work_len = (n > 0) ? n : 1;                                               \
  REAL_TYPE *work = NULL;                                                       \
  if (!g_fortran_fn[OP_ID] || !rcond) return -1;                                \
  if (!(layout == FB_LAYOUT_ROW_MAJOR || fb_lapack_layout_is_col_major(layout))) { \
    return -1;                                                                  \
  }                                                                             \
  if ((n > 0 && !d) || (n > 1 && !e)) return -1;                               \
  work = (REAL_TYPE *)malloc((size_t)work_len * sizeof(REAL_TYPE));            \
  if (!work) return -101;                                                       \
  ((void (*)(int *, const REAL_TYPE *, const REAL_TYPE *, const REAL_TYPE *,   \
             REAL_TYPE *, REAL_TYPE *, int *))g_fortran_fn[OP_ID])(            \
      &n_, d, e, &anorm, rcond, work, &info);                                  \
  free(work);                                                                   \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    int *n, REAL_TYPE *d, REAL_TYPE *e, REAL_TYPE *anorm, REAL_TYPE *rcond,    \
    REAL_TYPE *work, int *info) {                                               \
  (void)work;                                                                   \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!n || !anorm || !rcond) {                                                 \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, int, const REAL_TYPE *, const REAL_TYPE *,   \
                     REAL_TYPE, REAL_TYPE *))g_cblas_fn[OP_ID])(                \
      FB_LAYOUT_COL_MAJOR, *n, d, e, *anorm, rcond);                           \
  if (info) *info = rc;                                                         \
}

#define DEFINE_LAPACK_PT_CON_COMPLEX_THUNKS(OP, MAT_E_TYPE, REAL_TYPE, OP_ID)  \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const int n, const REAL_TYPE *d,                 \
    const MAT_E_TYPE *e, const REAL_TYPE anorm, REAL_TYPE *rcond) {            \
  int n_ = n;                                                                   \
  int info = 0;                                                                 \
  int work_len = (n > 0) ? n : 1;                                               \
  REAL_TYPE *work = NULL;                                                       \
  if (!g_fortran_fn[OP_ID] || !rcond) return -1;                                \
  if (!(layout == FB_LAYOUT_ROW_MAJOR || fb_lapack_layout_is_col_major(layout))) { \
    return -1;                                                                  \
  }                                                                             \
  if ((n > 0 && !d) || (n > 1 && !e)) return -1;                               \
  work = (REAL_TYPE *)malloc((size_t)work_len * sizeof(REAL_TYPE));            \
  if (!work) return -101;                                                       \
  ((void (*)(int *, const REAL_TYPE *, const MAT_E_TYPE *, const REAL_TYPE *,  \
             REAL_TYPE *, REAL_TYPE *, int *))g_fortran_fn[OP_ID])(            \
      &n_, d, e, &anorm, rcond, work, &info);                                  \
  free(work);                                                                   \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    int *n, REAL_TYPE *d, MAT_E_TYPE *e, REAL_TYPE *anorm, REAL_TYPE *rcond,   \
    REAL_TYPE *work, int *info) {                                               \
  (void)work;                                                                   \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!n || !anorm || !rcond) {                                                 \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, int, const REAL_TYPE *, const MAT_E_TYPE *,  \
                     REAL_TYPE, REAL_TYPE *))g_cblas_fn[OP_ID])(                \
      FB_LAYOUT_COL_MAJOR, *n, d, e, *anorm, rcond);                           \
  if (info) *info = rc;                                                         \
}

#define DEFINE_LAPACK_PT_FACTOR_REAL_THUNKS(OP, REAL_TYPE, OP_ID)              \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const int n, REAL_TYPE *d, REAL_TYPE *e) {       \
  int n_ = n;                                                                   \
  int info = 0;                                                                 \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (!(layout == FB_LAYOUT_ROW_MAJOR || fb_lapack_layout_is_col_major(layout))) { \
    return -1;                                                                  \
  }                                                                             \
  if ((n > 0 && !d) || (n > 1 && !e)) return -1;                               \
  ((void (*)(int *, REAL_TYPE *, REAL_TYPE *, int *))g_fortran_fn[OP_ID])(     \
      &n_, d, e, &info);                                                        \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(int *n, REAL_TYPE *d, REAL_TYPE *e,  \
                                          int *info) {                          \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!n) {                                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, int, REAL_TYPE *, REAL_TYPE *))             \
      g_cblas_fn[OP_ID])(FB_LAYOUT_COL_MAJOR, *n, d, e);                       \
  if (info) *info = rc;                                                         \
}

#define DEFINE_LAPACK_PT_FACTOR_COMPLEX_THUNKS(OP, MAT_TYPE, REAL_TYPE, OP_ID) \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const int n, REAL_TYPE *d, MAT_TYPE *e) {        \
  int n_ = n;                                                                   \
  int info = 0;                                                                 \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (!(layout == FB_LAYOUT_ROW_MAJOR || fb_lapack_layout_is_col_major(layout))) { \
    return -1;                                                                  \
  }                                                                             \
  if ((n > 0 && !d) || (n > 1 && !e)) return -1;                               \
  ((void (*)(int *, REAL_TYPE *, MAT_TYPE *, int *))g_fortran_fn[OP_ID])(      \
      &n_, d, e, &info);                                                        \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(int *n, REAL_TYPE *d, MAT_TYPE *e,   \
                                          int *info) {                          \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!n) {                                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, int, REAL_TYPE *, MAT_TYPE *))              \
      g_cblas_fn[OP_ID])(FB_LAYOUT_COL_MAJOR, *n, d, e);                       \
  if (info) *info = rc;                                                         \
}

#define DEFINE_LAPACK_PT_FACTORED_SOLVE_REAL_THUNKS(OP, REAL_TYPE, OP_ID)      \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const fb_uplo_t uplo, const int n,               \
    const int nrhs, const REAL_TYPE *d, const REAL_TYPE *e, REAL_TYPE *b,      \
    const int ldb) {                                                            \
  char uplo_char = 'U';                                                         \
  int n_ = n, nrhs_ = nrhs, ldb_ = (n > 0) ? n : 1, info = 0;                  \
  size_t b_elems = (size_t)ldb_ * (size_t)((nrhs > 0) ? nrhs : 1);             \
  REAL_TYPE *b_col = NULL;                                                      \
  REAL_TYPE *b_arg = b;                                                         \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (fb_lapack_uplo_to_char(uplo, &uplo_char) != 0) return -1;                \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if ((n > 0 && !d) || (n > 1 && !e) || (n > 0 && nrhs > 0 && !b) ||        \
        ldb < nrhs) {                                                           \
      return -1;                                                                \
    }                                                                           \
    b_col = (REAL_TYPE *)malloc(b_elems * sizeof(REAL_TYPE));                   \
    if (!b_col) return -101;                                                    \
    if (fb_copy_matrix_to_col_major(b, layout, n, nrhs, ldb,                   \
                                    b_col, ldb_, sizeof(REAL_TYPE)) != 0) {    \
      free(b_col);                                                              \
      return -1;                                                                \
    }                                                                           \
    b_arg = b_col;                                                              \
  } else if (fb_lapack_layout_is_col_major(layout)) {                           \
    ldb_ = ldb;                                                                 \
    if ((n > 0 && !d) || (n > 1 && !e) || (n > 0 && nrhs > 0 && !b)) {        \
      return -1;                                                                \
    }                                                                           \
  } else {                                                                      \
    return -1;                                                                  \
  }                                                                             \
  ((void (*)(char *, int *, int *, const REAL_TYPE *, const REAL_TYPE *,       \
             REAL_TYPE *, int *, int *))g_fortran_fn[OP_ID])(                  \
      &uplo_char, &n_, &nrhs_, d, e, b_arg, &ldb_, &info);                     \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if (fb_copy_matrix_from_col_major(b_col, n, nrhs, ldb_, layout,            \
                                      b, ldb, sizeof(REAL_TYPE)) != 0) {       \
      free(b_col);                                                              \
      return -1;                                                                \
    }                                                                           \
    free(b_col);                                                                \
  }                                                                             \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(char *uplo, int *n, int *nrhs,       \
                                          REAL_TYPE *d, REAL_TYPE *e,           \
                                          REAL_TYPE *b, int *ldb, int *info) {  \
  fb_uplo_t uplo_ = FB_UPPER;                                                   \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!uplo || !n || !nrhs || !ldb ||                                           \
      fb_lapack_char_to_uplo(*uplo, &uplo_) != 0) {                            \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, fb_uplo_t, int, int, const REAL_TYPE *,      \
                     const REAL_TYPE *, REAL_TYPE *, int))g_cblas_fn[OP_ID])(  \
      FB_LAYOUT_COL_MAJOR, uplo_, *n, *nrhs, d, e, b, *ldb);                   \
  if (info) *info = rc;                                                         \
}

#define DEFINE_LAPACK_PT_FACTORED_SOLVE_COMPLEX_THUNKS(OP, MAT_TYPE, REAL_TYPE, OP_ID) \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const fb_uplo_t uplo, const int n,               \
    const int nrhs, const REAL_TYPE *d, const MAT_TYPE *e, MAT_TYPE *b,        \
    const int ldb) {                                                            \
  char uplo_char = 'U';                                                         \
  int n_ = n, nrhs_ = nrhs, ldb_ = (n > 0) ? n : 1, info = 0;                  \
  size_t b_elems = (size_t)ldb_ * (size_t)((nrhs > 0) ? nrhs : 1);             \
  MAT_TYPE *b_col = NULL;                                                       \
  MAT_TYPE *b_arg = b;                                                          \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (fb_lapack_uplo_to_char(uplo, &uplo_char) != 0) return -1;                \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if ((n > 0 && !d) || (n > 1 && !e) || (n > 0 && nrhs > 0 && !b) ||        \
        ldb < nrhs) {                                                           \
      return -1;                                                                \
    }                                                                           \
    b_col = (MAT_TYPE *)malloc(b_elems * sizeof(MAT_TYPE));                     \
    if (!b_col) return -101;                                                    \
    if (fb_copy_matrix_to_col_major(b, layout, n, nrhs, ldb,                   \
                                    b_col, ldb_, sizeof(MAT_TYPE)) != 0) {     \
      free(b_col);                                                              \
      return -1;                                                                \
    }                                                                           \
    b_arg = b_col;                                                              \
  } else if (fb_lapack_layout_is_col_major(layout)) {                           \
    ldb_ = ldb;                                                                 \
    if ((n > 0 && !d) || (n > 1 && !e) || (n > 0 && nrhs > 0 && !b)) {        \
      return -1;                                                                \
    }                                                                           \
  } else {                                                                      \
    return -1;                                                                  \
  }                                                                             \
  ((void (*)(char *, int *, int *, const REAL_TYPE *, const MAT_TYPE *,        \
             MAT_TYPE *, int *, int *))g_fortran_fn[OP_ID])(                   \
      &uplo_char, &n_, &nrhs_, d, e, b_arg, &ldb_, &info);                     \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if (fb_copy_matrix_from_col_major(b_col, n, nrhs, ldb_, layout,            \
                                      b, ldb, sizeof(MAT_TYPE)) != 0) {        \
      free(b_col);                                                              \
      return -1;                                                                \
    }                                                                           \
    free(b_col);                                                                \
  }                                                                             \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(char *uplo, int *n, int *nrhs,       \
                                          REAL_TYPE *d, MAT_TYPE *e,           \
                                          MAT_TYPE *b, int *ldb, int *info) {   \
  fb_uplo_t uplo_ = FB_UPPER;                                                   \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!uplo || !n || !nrhs || !ldb ||                                           \
      fb_lapack_char_to_uplo(*uplo, &uplo_) != 0) {                            \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, fb_uplo_t, int, int, const REAL_TYPE *,      \
                     const MAT_TYPE *, MAT_TYPE *, int))g_cblas_fn[OP_ID])(    \
      FB_LAYOUT_COL_MAJOR, uplo_, *n, *nrhs, d, e, b, *ldb);                   \
  if (info) *info = rc;                                                         \
}

#define DEFINE_LAPACK_PT_SOLVE_REAL_THUNKS(OP, REAL_TYPE, OP_ID)               \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const int n, const int nrhs, REAL_TYPE *d,       \
    REAL_TYPE *e, REAL_TYPE *b, const int ldb) {                               \
  int n_ = n, nrhs_ = nrhs, ldb_ = (n > 0) ? n : 1, info = 0;                  \
  size_t b_elems = (size_t)ldb_ * (size_t)((nrhs > 0) ? nrhs : 1);             \
  REAL_TYPE *b_col = NULL;                                                      \
  REAL_TYPE *b_arg = b;                                                         \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if ((n > 0 && !d) || (n > 1 && !e) || (n > 0 && nrhs > 0 && !b) ||        \
        ldb < nrhs) {                                                           \
      return -1;                                                                \
    }                                                                           \
    b_col = (REAL_TYPE *)malloc(b_elems * sizeof(REAL_TYPE));                   \
    if (!b_col) return -101;                                                    \
    if (fb_copy_matrix_to_col_major(b, layout, n, nrhs, ldb,                   \
                                    b_col, ldb_, sizeof(REAL_TYPE)) != 0) {    \
      free(b_col);                                                              \
      return -1;                                                                \
    }                                                                           \
    b_arg = b_col;                                                              \
  } else if (fb_lapack_layout_is_col_major(layout)) {                           \
    ldb_ = ldb;                                                                 \
    if ((n > 0 && !d) || (n > 1 && !e) || (n > 0 && nrhs > 0 && !b)) {        \
      return -1;                                                                \
    }                                                                           \
  } else {                                                                      \
    return -1;                                                                  \
  }                                                                             \
  ((void (*)(int *, int *, REAL_TYPE *, REAL_TYPE *, REAL_TYPE *, int *,       \
             int *))g_fortran_fn[OP_ID])(                                      \
      &n_, &nrhs_, d, e, b_arg, &ldb_, &info);                                 \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if (fb_copy_matrix_from_col_major(b_col, n, nrhs, ldb_, layout,            \
                                      b, ldb, sizeof(REAL_TYPE)) != 0) {       \
      free(b_col);                                                              \
      return -1;                                                                \
    }                                                                           \
    free(b_col);                                                                \
  }                                                                             \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(int *n, int *nrhs, REAL_TYPE *d,     \
                                          REAL_TYPE *e, REAL_TYPE *b, int *ldb, \
                                          int *info) {                          \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!n || !nrhs || !ldb) {                                                    \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, int, int, REAL_TYPE *, REAL_TYPE *,         \
                     REAL_TYPE *, int))g_cblas_fn[OP_ID])(                     \
      FB_LAYOUT_COL_MAJOR, *n, *nrhs, d, e, b, *ldb);                          \
  if (info) *info = rc;                                                         \
}

#define DEFINE_LAPACK_PT_SOLVE_COMPLEX_THUNKS(OP, MAT_TYPE, REAL_TYPE, OP_ID)  \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const int n, const int nrhs, REAL_TYPE *d,       \
    MAT_TYPE *e, MAT_TYPE *b, const int ldb) {                                 \
  int n_ = n, nrhs_ = nrhs, ldb_ = (n > 0) ? n : 1, info = 0;                  \
  size_t b_elems = (size_t)ldb_ * (size_t)((nrhs > 0) ? nrhs : 1);             \
  MAT_TYPE *b_col = NULL;                                                       \
  MAT_TYPE *b_arg = b;                                                          \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if ((n > 0 && !d) || (n > 1 && !e) || (n > 0 && nrhs > 0 && !b) ||        \
        ldb < nrhs) {                                                           \
      return -1;                                                                \
    }                                                                           \
    b_col = (MAT_TYPE *)malloc(b_elems * sizeof(MAT_TYPE));                     \
    if (!b_col) return -101;                                                    \
    if (fb_copy_matrix_to_col_major(b, layout, n, nrhs, ldb,                   \
                                    b_col, ldb_, sizeof(MAT_TYPE)) != 0) {     \
      free(b_col);                                                              \
      return -1;                                                                \
    }                                                                           \
    b_arg = b_col;                                                              \
  } else if (fb_lapack_layout_is_col_major(layout)) {                           \
    ldb_ = ldb;                                                                 \
    if ((n > 0 && !d) || (n > 1 && !e) || (n > 0 && nrhs > 0 && !b)) {        \
      return -1;                                                                \
    }                                                                           \
  } else {                                                                      \
    return -1;                                                                  \
  }                                                                             \
  ((void (*)(int *, int *, REAL_TYPE *, MAT_TYPE *, MAT_TYPE *, int *,        \
             int *))g_fortran_fn[OP_ID])(                                      \
      &n_, &nrhs_, d, e, b_arg, &ldb_, &info);                                 \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if (fb_copy_matrix_from_col_major(b_col, n, nrhs, ldb_, layout,            \
                                      b, ldb, sizeof(MAT_TYPE)) != 0) {        \
      free(b_col);                                                              \
      return -1;                                                                \
    }                                                                           \
    free(b_col);                                                                \
  }                                                                             \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(int *n, int *nrhs, REAL_TYPE *d,     \
                                          MAT_TYPE *e, MAT_TYPE *b, int *ldb,  \
                                          int *info) {                          \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!n || !nrhs || !ldb) {                                                    \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, int, int, REAL_TYPE *, MAT_TYPE *,          \
                     MAT_TYPE *, int))g_cblas_fn[OP_ID])(                      \
      FB_LAYOUT_COL_MAJOR, *n, *nrhs, d, e, b, *ldb);                          \
  if (info) *info = rc;                                                         \
}

#define DEFINE_LAPACK_PT_SVX_REAL_THUNKS(OP, REAL_TYPE, OP_ID)                 \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const char fact, const int n, const int nrhs,    \
    REAL_TYPE *d, REAL_TYPE *e, REAL_TYPE *df, REAL_TYPE *ef, REAL_TYPE *b,    \
    const int ldb, REAL_TYPE *x, const int ldx, REAL_TYPE *rcond,              \
    REAL_TYPE *ferr, REAL_TYPE *berr) {                                         \
  int n_ = n, nrhs_ = nrhs, ldb_ = (n > 0) ? n : 1, ldx_ = (n > 0) ? n : 1,   \
      info = 0;                                                                 \
  int work_len = (n > 0) ? n : 1;                                               \
  size_t b_elems = (size_t)ldb_ * (size_t)((nrhs > 0) ? nrhs : 1);             \
  size_t x_elems = (size_t)ldx_ * (size_t)((nrhs > 0) ? nrhs : 1);             \
  REAL_TYPE *b_col = NULL;                                                      \
  REAL_TYPE *x_col = NULL;                                                      \
  REAL_TYPE *b_arg = b;                                                         \
  REAL_TYPE *x_arg = x;                                                         \
  REAL_TYPE *work = NULL;                                                       \
  if (!g_fortran_fn[OP_ID] || !rcond || !ferr || !berr) return -1;             \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if ((n > 0 && (!d || !df)) || (n > 1 && (!e || !ef)) ||                    \
        (n > 0 && nrhs > 0 && (!b || !x)) || ldb < nrhs || ldx < nrhs) {      \
      return -1;                                                                \
    }                                                                           \
    b_col = (REAL_TYPE *)malloc(b_elems * sizeof(REAL_TYPE));                   \
    x_col = (REAL_TYPE *)malloc(x_elems * sizeof(REAL_TYPE));                   \
    if (!b_col || !x_col) {                                                     \
      free(b_col);                                                              \
      free(x_col);                                                              \
      return -101;                                                              \
    }                                                                           \
    if (fb_copy_matrix_to_col_major(b, layout, n, nrhs, ldb,                   \
                                    b_col, ldb_, sizeof(REAL_TYPE)) != 0) {    \
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
  work = (REAL_TYPE *)malloc((size_t)work_len * sizeof(REAL_TYPE));            \
  if (!work) {                                                                  \
    free(b_col);                                                                \
    free(x_col);                                                                \
    return -101;                                                                \
  }                                                                             \
  ((void (*)(char *, int *, int *, REAL_TYPE *, REAL_TYPE *, REAL_TYPE *,      \
             REAL_TYPE *, REAL_TYPE *, int *, REAL_TYPE *, int *, REAL_TYPE *, \
             REAL_TYPE *, REAL_TYPE *, REAL_TYPE *, int *))g_fortran_fn[OP_ID])(\
      (char *)&fact, &n_, &nrhs_, d, e, df, ef, b_arg, &ldb_, x_arg, &ldx_,    \
      rcond, ferr, berr, work, &info);                                          \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if (fb_copy_matrix_from_col_major(x_col, n, nrhs, ldx_, layout,            \
                                      x, ldx, sizeof(REAL_TYPE)) != 0) {       \
      free(b_col);                                                              \
      free(x_col);                                                              \
      free(work);                                                               \
      return -1;                                                                \
    }                                                                           \
    free(b_col);                                                                \
    free(x_col);                                                                \
  }                                                                             \
  free(work);                                                                   \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(char *fact, int *n, int *nrhs,       \
                                          REAL_TYPE *d, REAL_TYPE *e,          \
                                          REAL_TYPE *df, REAL_TYPE *ef,        \
                                          REAL_TYPE *b, int *ldb, REAL_TYPE *x,\
                                          int *ldx, REAL_TYPE *rcond,          \
                                          REAL_TYPE *ferr, REAL_TYPE *berr,    \
                                          REAL_TYPE *work, int *info) {         \
  (void)work;                                                                   \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!fact || !n || !nrhs || !ldb || !ldx || !rcond || !ferr || !berr) {     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, char, int, int, REAL_TYPE *, REAL_TYPE *,   \
                     REAL_TYPE *, REAL_TYPE *, REAL_TYPE *, int, REAL_TYPE *,  \
                     int, REAL_TYPE *, REAL_TYPE *, REAL_TYPE *))g_cblas_fn[OP_ID])(\
      FB_LAYOUT_COL_MAJOR, *fact, *n, *nrhs, d, e, df, ef, b, *ldb, x, *ldx,   \
      rcond, ferr, berr);                                                       \
  if (info) *info = rc;                                                         \
}
