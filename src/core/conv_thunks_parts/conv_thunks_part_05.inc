
#define DEFINE_LAPACK_GB_SVX_COMPLEX_THUNKS(OP, MAT_TYPE, REAL_TYPE, OP_ID)   \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const char fact, const char trans, const int n, \
    const int kl, const int ku, const int nrhs, MAT_TYPE *ab, const int ldab, \
    MAT_TYPE *afb, const int ldafb, int *ipiv, char *equed, REAL_TYPE *r,     \
    REAL_TYPE *c, MAT_TYPE *b, const int ldb, MAT_TYPE *x, const int ldx,     \
    REAL_TYPE *rcond, REAL_TYPE *ferr, REAL_TYPE *berr,                        \
    REAL_TYPE *rpvgrw) {                                                        \
  int compact_rows = (kl >= 0 && ku >= 0) ? (kl + ku + 1) : 1;                \
  int fact_rows = (kl >= 0 && ku >= 0) ? (2 * kl + ku + 1) : 1;               \
  int n_ = n, kl_ = kl, ku_ = ku, nrhs_ = nrhs, ldab_ = fact_rows,            \
      ldafb_ = fact_rows, ldb_ = (n > 0) ? n : 1, ldx_ = (n > 0) ? n : 1,     \
      info = 0;                                                                 \
  int work_len = (n > 0) ? (2 * n) : 1;                                        \
  int rwork_len = (n > 1) ? n : 2;                                             \
  int fact_is_f = (fact == 'F' || fact == 'f');                                \
  int fact_is_e = (fact == 'E' || fact == 'e');                                \
  int fact_is_n = (fact == 'N' || fact == 'n');                                \
  int equed_has_scaling = 0;                                                    \
  size_t ab_elems = (size_t)fact_rows * (size_t)((n > 0) ? n : 1);            \
  size_t afb_elems = (size_t)fact_rows * (size_t)((n > 0) ? n : 1);           \
  size_t b_elems = (size_t)ldb_ * (size_t)((nrhs > 0) ? nrhs : 1);            \
  size_t x_elems = (size_t)ldx_ * (size_t)((nrhs > 0) ? nrhs : 1);            \
  MAT_TYPE *ab_col = NULL;                                                      \
  MAT_TYPE *afb_col = NULL;                                                     \
  MAT_TYPE *b_col = NULL;                                                       \
  MAT_TYPE *x_col = NULL;                                                       \
  MAT_TYPE *ab_arg = ab;                                                        \
  MAT_TYPE *afb_arg = afb;                                                      \
  MAT_TYPE *b_arg = b;                                                          \
  MAT_TYPE *x_arg = x;                                                          \
  MAT_TYPE *work = NULL;                                                        \
  REAL_TYPE *rwork = NULL;                                                      \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (!ipiv || !equed || !r || !c || !rcond || !ferr || !berr || !rpvgrw) {   \
    return -1;                                                                  \
  }                                                                             \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if ((n > 0 && (!ab || !afb || !ipiv)) || (n > 0 && nrhs > 0 && (!b || !x))||\
        ldab < n || ldafb < n || ldb < nrhs || ldx < nrhs) {                  \
      return -1;                                                                \
    }                                                                           \
    ab_col = (MAT_TYPE *)malloc(ab_elems * sizeof(MAT_TYPE));                  \
    afb_col = (MAT_TYPE *)calloc(afb_elems, sizeof(MAT_TYPE));                 \
    b_col = (MAT_TYPE *)malloc(b_elems * sizeof(MAT_TYPE));                    \
    x_col = (MAT_TYPE *)malloc(x_elems * sizeof(MAT_TYPE));                    \
    if (!ab_col || !afb_col || !b_col || !x_col) {                             \
      free(ab_col);                                                             \
      free(afb_col);                                                            \
      free(b_col);                                                              \
      free(x_col);                                                              \
      return -101;                                                              \
    }                                                                           \
    if (fb_copy_general_band_compact_row_to_expanded_col_major(                \
            ab, n, n, kl, ku, ldab, ab_col, fact_rows, sizeof(MAT_TYPE)) != 0 ||\
        (fact_is_f &&                                                           \
         fb_copy_general_band_to_col_major(afb, layout, n, n, kl, kl + ku,    \
                                           ldafb, afb_col, fact_rows,          \
                                           sizeof(MAT_TYPE)) != 0) ||          \
        fb_copy_matrix_to_col_major(b, layout, n, nrhs, ldb,                  \
                                    b_col, ldb_, sizeof(MAT_TYPE)) != 0) {     \
      free(ab_col);                                                             \
      free(afb_col);                                                            \
      free(b_col);                                                              \
      free(x_col);                                                              \
      return -1;                                                                \
    }                                                                           \
    ab_arg = ab_col;                                                            \
    afb_arg = afb_col;                                                          \
    b_arg = b_col;                                                              \
    x_arg = x_col;                                                              \
  } else if (fb_lapack_layout_is_col_major(layout)) {                           \
    ldab_ = ldab;                                                               \
    ldafb_ = ldafb;                                                             \
    ldb_ = ldb;                                                                 \
    ldx_ = ldx;                                                                 \
    if ((n > 0 && (!ab || !afb || !ipiv)) ||                                   \
        (n > 0 && nrhs > 0 && (!b || !x))) {                                   \
      return -1;                                                                \
    }                                                                           \
  } else {                                                                      \
    return -1;                                                                  \
  }                                                                             \
  work = (MAT_TYPE *)calloc((size_t)work_len, sizeof(MAT_TYPE));               \
  rwork = (REAL_TYPE *)calloc((size_t)rwork_len, sizeof(REAL_TYPE));           \
  if (!work || !rwork) {                                                        \
    free(ab_col);                                                               \
    free(afb_col);                                                              \
    free(b_col);                                                                \
    free(x_col);                                                                \
    free(work);                                                                 \
    free(rwork);                                                                \
    return -101;                                                                \
  }                                                                             \
  ((void (*)(char *, char *, int *, int *, int *, int *, MAT_TYPE *, int *,   \
             MAT_TYPE *, int *, int *, char *, REAL_TYPE *, REAL_TYPE *,      \
             MAT_TYPE *, int *, MAT_TYPE *, int *, REAL_TYPE *, REAL_TYPE *,  \
             REAL_TYPE *, MAT_TYPE *, REAL_TYPE *, int *))g_fortran_fn[OP_ID])(\
      (char *)&fact, (char *)&trans, &n_, &kl_, &ku_, &nrhs_, ab_arg, &ldab_, \
      afb_arg, &ldafb_, ipiv, equed, r, c, b_arg, &ldb_, x_arg, &ldx_, rcond, \
      ferr, berr, work, rwork, &info);                                          \
  *rpvgrw = (rwork_len > 1) ? rwork[1] : rwork[0];                             \
  equed_has_scaling = (*equed == 'R' || *equed == 'r' || *equed == 'C' ||     \
                       *equed == 'c' || *equed == 'B' || *equed == 'b');      \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if ((fact_is_e && equed_has_scaling &&                                      \
         fb_copy_general_band_compact_row_from_expanded_col_major(             \
             ab_col, n, n, kl, ku, fact_rows, ab, ldab, sizeof(MAT_TYPE)) != 0)||\
        ((fact_is_e || fact_is_n) &&                                            \
         fb_copy_general_band_from_col_major(afb_col, n, n, kl, kl + ku,       \
                                             fact_rows, layout, afb, ldafb,     \
                                             sizeof(MAT_TYPE)) != 0) ||        \
        (fact_is_f && equed_has_scaling &&                                      \
         fb_copy_matrix_from_col_major(b_col, n, nrhs, ldb_, layout,          \
                                       b, ldb, sizeof(MAT_TYPE)) != 0) ||      \
        fb_copy_matrix_from_col_major(x_col, n, nrhs, ldx_, layout,           \
                                      x, ldx, sizeof(MAT_TYPE)) != 0) {        \
      free(ab_col);                                                             \
      free(afb_col);                                                            \
      free(b_col);                                                              \
      free(x_col);                                                              \
      free(work);                                                               \
      free(rwork);                                                              \
      return -1;                                                                \
    }                                                                           \
    free(ab_col);                                                               \
    free(afb_col);                                                              \
    free(b_col);                                                                \
    free(x_col);                                                                \
  }                                                                             \
  free(work);                                                                   \
  free(rwork);                                                                  \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    char *fact, char *trans, int *n, int *kl, int *ku, int *nrhs,             \
    MAT_TYPE *ab, int *ldab, MAT_TYPE *afb, int *ldafb, int *ipiv,            \
    char *equed, REAL_TYPE *r, REAL_TYPE *c, MAT_TYPE *b, int *ldb,           \
    MAT_TYPE *x, int *ldx, REAL_TYPE *rcond, REAL_TYPE *ferr,                 \
    REAL_TYPE *berr, MAT_TYPE *work, REAL_TYPE *rwork, int *info) {           \
  REAL_TYPE rpvgrw = (REAL_TYPE)0;                                              \
  (void)work;                                                                   \
  (void)rwork;                                                                  \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!fact || !trans || !ipiv || !equed || !r || !c || !rcond || !ferr ||    \
      !berr) {                                                                  \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, char, char, int, int, int, int,             \
                     MAT_TYPE *, int, MAT_TYPE *, int, int *, char *,         \
                     REAL_TYPE *, REAL_TYPE *, MAT_TYPE *, int, MAT_TYPE *,   \
                     int, REAL_TYPE *, REAL_TYPE *, REAL_TYPE *,              \
                     REAL_TYPE *))g_cblas_fn[OP_ID])(                          \
      FB_LAYOUT_COL_MAJOR, *fact, *trans, *n, *kl, *ku, *nrhs, ab, *ldab,     \
      afb, *ldafb, ipiv, equed, r, c, b, *ldb, x, *ldx, rcond, ferr, berr,    \
      &rpvgrw);                                                                 \
  if (info) *info = rc;                                                         \
}

#define DEFINE_LAPACK_PB_REFINE_REAL_THUNKS(OP, REAL_TYPE, OP_ID)             \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const fb_uplo_t uplo, const int n,               \
    const int kd, const int nrhs, const REAL_TYPE *ab, const int ldab,         \
    const REAL_TYPE *afb, const int ldafb, const REAL_TYPE *b, const int ldb,  \
    REAL_TYPE *x, const int ldx, REAL_TYPE *ferr, REAL_TYPE *berr) {           \
  char uplo_char = 'U';                                                         \
  int n_ = n, kd_ = kd, nrhs_ = nrhs, ldab_ = (kd >= 0) ? (kd + 1) : 1,        \
      ldafb_ = (kd >= 0) ? (kd + 1) : 1, ldb_ = (n > 0) ? n : 1,               \
      ldx_ = (n > 0) ? n : 1, info = 0;                                         \
  int work_len = (n > 0) ? (3 * n) : 1;                                         \
  int iwork_len = (n > 0) ? n : 1;                                              \
  size_t ab_elems = (size_t)ldab_ * (size_t)((n > 0) ? n : 1);                 \
  size_t afb_elems = (size_t)ldafb_ * (size_t)((n > 0) ? n : 1);               \
  size_t b_elems = (size_t)ldb_ * (size_t)((nrhs > 0) ? nrhs : 1);             \
  size_t x_elems = (size_t)ldx_ * (size_t)((nrhs > 0) ? nrhs : 1);             \
  REAL_TYPE *ab_col = NULL;                                                     \
  REAL_TYPE *afb_col = NULL;                                                    \
  REAL_TYPE *b_col = NULL;                                                      \
  REAL_TYPE *x_col = NULL;                                                      \
  const REAL_TYPE *ab_arg = ab;                                                 \
  const REAL_TYPE *afb_arg = afb;                                               \
  const REAL_TYPE *b_arg = b;                                                   \
  REAL_TYPE *x_arg = x;                                                         \
  REAL_TYPE *work = NULL;                                                       \
  int *iwork = NULL;                                                            \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (fb_lapack_uplo_to_char(uplo, &uplo_char) != 0 || !ferr || !berr) return -1;\
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if ((n > 0 && (!ab || !afb)) || (n > 0 && nrhs > 0 && (!b || !x)) ||       \
        ldab < n || ldafb < n || ldb < nrhs || ldx < nrhs) {                   \
      return -1;                                                                \
    }                                                                           \
    ab_col = (REAL_TYPE *)malloc(ab_elems * sizeof(REAL_TYPE));                 \
    afb_col = (REAL_TYPE *)malloc(afb_elems * sizeof(REAL_TYPE));               \
    b_col = (REAL_TYPE *)malloc(b_elems * sizeof(REAL_TYPE));                   \
    x_col = (REAL_TYPE *)malloc(x_elems * sizeof(REAL_TYPE));                   \
    if (!ab_col || !afb_col || !b_col || !x_col) {                              \
      free(ab_col);                                                             \
      free(afb_col);                                                            \
      free(b_col);                                                              \
      free(x_col);                                                              \
      return -101;                                                              \
    }                                                                           \
    if (fb_copy_sym_band_to_col_major(ab, layout, uplo, n, kd, ldab,          \
                                      ab_col, ldab_, sizeof(REAL_TYPE)) != 0 ||\
        fb_copy_sym_band_to_col_major(afb, layout, uplo, n, kd, ldafb,        \
                                      afb_col, ldafb_, sizeof(REAL_TYPE)) != 0 ||\
        fb_copy_matrix_to_col_major(b, layout, n, nrhs, ldb,                  \
                                    b_col, ldb_, sizeof(REAL_TYPE)) != 0 ||    \
        fb_copy_matrix_to_col_major(x, layout, n, nrhs, ldx,                  \
                                    x_col, ldx_, sizeof(REAL_TYPE)) != 0) {    \
      free(ab_col);                                                             \
      free(afb_col);                                                            \
      free(b_col);                                                              \
      free(x_col);                                                              \
      return -1;                                                                \
    }                                                                           \
    ab_arg = ab_col;                                                            \
    afb_arg = afb_col;                                                          \
    b_arg = b_col;                                                              \
    x_arg = x_col;                                                              \
  } else if (fb_lapack_layout_is_col_major(layout)) {                           \
    ldab_ = ldab;                                                               \
    ldafb_ = ldafb;                                                             \
    ldb_ = ldb;                                                                 \
    ldx_ = ldx;                                                                 \
    if ((n > 0 && (!ab || !afb)) || (n > 0 && nrhs > 0 && (!b || !x))) {      \
      return -1;                                                                \
    }                                                                           \
  } else {                                                                      \
    return -1;                                                                  \
  }                                                                             \
  work = (REAL_TYPE *)malloc((size_t)work_len * sizeof(REAL_TYPE));            \
  iwork = (int *)malloc((size_t)iwork_len * sizeof(int));                      \
  if (!work || !iwork) {                                                        \
    free(ab_col);                                                               \
    free(afb_col);                                                              \
    free(b_col);                                                                \
    free(x_col);                                                                \
    free(work);                                                                 \
    free(iwork);                                                                \
    return -101;                                                                \
  }                                                                             \
  ((void (*)(char *, int *, int *, int *, const REAL_TYPE *, int *,            \
             const REAL_TYPE *, int *, const REAL_TYPE *, int *, REAL_TYPE *,  \
             int *, REAL_TYPE *, REAL_TYPE *, REAL_TYPE *, int *, int *))      \
      g_fortran_fn[OP_ID])(                                                     \
      &uplo_char, &n_, &kd_, &nrhs_, ab_arg, &ldab_, afb_arg, &ldafb_,         \
      b_arg, &ldb_, x_arg, &ldx_, ferr, berr, work, iwork, &info);             \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if (fb_copy_matrix_from_col_major(x_col, n, nrhs, ldx_, layout,           \
                                      x, ldx, sizeof(REAL_TYPE)) != 0) {       \
      free(ab_col);                                                             \
      free(afb_col);                                                            \
      free(b_col);                                                              \
      free(x_col);                                                              \
      free(work);                                                               \
      free(iwork);                                                              \
      return -1;                                                                \
    }                                                                           \
    free(ab_col);                                                               \
    free(afb_col);                                                              \
    free(b_col);                                                                \
    free(x_col);                                                                \
  }                                                                             \
  free(work);                                                                   \
  free(iwork);                                                                  \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    char *uplo, int *n, int *kd, int *nrhs, REAL_TYPE *ab, int *ldab,          \
    REAL_TYPE *afb, int *ldafb, REAL_TYPE *b, int *ldb, REAL_TYPE *x,          \
    int *ldx, REAL_TYPE *ferr, REAL_TYPE *berr, REAL_TYPE *work, int *iwork,   \
    int *info) {                                                                \
  fb_uplo_t uplo_ = FB_UPPER;                                                   \
  (void)work;                                                                   \
  (void)iwork;                                                                  \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!uplo || !ferr || !berr || fb_lapack_char_to_uplo(*uplo, &uplo_) != 0) {\
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, fb_uplo_t, int, int, int, const REAL_TYPE *,\
                     int, const REAL_TYPE *, int, const REAL_TYPE *, int,      \
                     REAL_TYPE *, int, REAL_TYPE *, REAL_TYPE *))              \
      g_cblas_fn[OP_ID])(FB_LAYOUT_COL_MAJOR, uplo_, *n, *kd, *nrhs,           \
                         ab, *ldab, afb, *ldafb, b, *ldb, x, *ldx,             \
                         ferr, berr);                                           \
  if (info) *info = rc;                                                         \
}

#define DEFINE_LAPACK_PB_REFINE_COMPLEX_THUNKS(OP, MAT_TYPE, REAL_TYPE, OP_ID) \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const fb_uplo_t uplo, const int n,               \
    const int kd, const int nrhs, const MAT_TYPE *ab, const int ldab,          \
    const MAT_TYPE *afb, const int ldafb, const MAT_TYPE *b, const int ldb,    \
    MAT_TYPE *x, const int ldx, REAL_TYPE *ferr, REAL_TYPE *berr) {            \
  char uplo_char = 'U';                                                         \
  int n_ = n, kd_ = kd, nrhs_ = nrhs, ldab_ = (kd >= 0) ? (kd + 1) : 1,        \
      ldafb_ = (kd >= 0) ? (kd + 1) : 1, ldb_ = (n > 0) ? n : 1,               \
      ldx_ = (n > 0) ? n : 1, info = 0;                                         \
  int work_len = (n > 0) ? (2 * n) : 1;                                         \
  int rwork_len = (n > 0) ? n : 1;                                              \
  size_t ab_elems = (size_t)ldab_ * (size_t)((n > 0) ? n : 1);                 \
  size_t afb_elems = (size_t)ldafb_ * (size_t)((n > 0) ? n : 1);               \
  size_t b_elems = (size_t)ldb_ * (size_t)((nrhs > 0) ? nrhs : 1);             \
  size_t x_elems = (size_t)ldx_ * (size_t)((nrhs > 0) ? nrhs : 1);             \
  MAT_TYPE *ab_col = NULL;                                                      \
  MAT_TYPE *afb_col = NULL;                                                     \
  MAT_TYPE *b_col = NULL;                                                       \
  MAT_TYPE *x_col = NULL;                                                       \
  const MAT_TYPE *ab_arg = ab;                                                  \
  const MAT_TYPE *afb_arg = afb;                                                \
  const MAT_TYPE *b_arg = b;                                                    \
  MAT_TYPE *x_arg = x;                                                          \
  MAT_TYPE *work = NULL;                                                        \
  REAL_TYPE *rwork = NULL;                                                      \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (fb_lapack_uplo_to_char(uplo, &uplo_char) != 0 || !ferr || !berr) return -1;\
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if ((n > 0 && (!ab || !afb)) || (n > 0 && nrhs > 0 && (!b || !x)) ||       \
        ldab < n || ldafb < n || ldb < nrhs || ldx < nrhs) {                   \
      return -1;                                                                \
    }                                                                           \
    ab_col = (MAT_TYPE *)malloc(ab_elems * sizeof(MAT_TYPE));                   \
    afb_col = (MAT_TYPE *)malloc(afb_elems * sizeof(MAT_TYPE));                 \
    b_col = (MAT_TYPE *)malloc(b_elems * sizeof(MAT_TYPE));                     \
    x_col = (MAT_TYPE *)malloc(x_elems * sizeof(MAT_TYPE));                     \
    if (!ab_col || !afb_col || !b_col || !x_col) {                              \
      free(ab_col);                                                             \
      free(afb_col);                                                            \
      free(b_col);                                                              \
      free(x_col);                                                              \
      return -101;                                                              \
    }                                                                           \
    if (fb_copy_sym_band_to_col_major(ab, layout, uplo, n, kd, ldab,          \
                                      ab_col, ldab_, sizeof(MAT_TYPE)) != 0 || \
        fb_copy_sym_band_to_col_major(afb, layout, uplo, n, kd, ldafb,        \
                                      afb_col, ldafb_, sizeof(MAT_TYPE)) != 0 ||\
        fb_copy_matrix_to_col_major(b, layout, n, nrhs, ldb,                  \
                                    b_col, ldb_, sizeof(MAT_TYPE)) != 0 ||     \
        fb_copy_matrix_to_col_major(x, layout, n, nrhs, ldx,                  \
                                    x_col, ldx_, sizeof(MAT_TYPE)) != 0) {     \
      free(ab_col);                                                             \
      free(afb_col);                                                            \
      free(b_col);                                                              \
      free(x_col);                                                              \
      return -1;                                                                \
    }                                                                           \
    ab_arg = ab_col;                                                            \
    afb_arg = afb_col;                                                          \
    b_arg = b_col;                                                              \
    x_arg = x_col;                                                              \
  } else if (fb_lapack_layout_is_col_major(layout)) {                           \
    ldab_ = ldab;                                                               \
    ldafb_ = ldafb;                                                             \
    ldb_ = ldb;                                                                 \
    ldx_ = ldx;                                                                 \
    if ((n > 0 && (!ab || !afb)) || (n > 0 && nrhs > 0 && (!b || !x))) {      \
      return -1;                                                                \
    }                                                                           \
  } else {                                                                      \
    return -1;                                                                  \
  }                                                                             \
  work = (MAT_TYPE *)malloc((size_t)work_len * sizeof(MAT_TYPE));              \
  rwork = (REAL_TYPE *)malloc((size_t)rwork_len * sizeof(REAL_TYPE));          \
  if (!work || !rwork) {                                                        \
    free(ab_col);                                                               \
    free(afb_col);                                                              \
    free(b_col);                                                                \
    free(x_col);                                                                \
    free(work);                                                                 \
    free(rwork);                                                                \
    return -101;                                                                \
  }                                                                             \
  ((void (*)(char *, int *, int *, int *, const MAT_TYPE *, int *,             \
             const MAT_TYPE *, int *, const MAT_TYPE *, int *, MAT_TYPE *,     \
             int *, REAL_TYPE *, REAL_TYPE *, MAT_TYPE *, REAL_TYPE *, int *)) \
      g_fortran_fn[OP_ID])(                                                     \
      &uplo_char, &n_, &kd_, &nrhs_, ab_arg, &ldab_, afb_arg, &ldafb_,         \
      b_arg, &ldb_, x_arg, &ldx_, ferr, berr, work, rwork, &info);             \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if (fb_copy_matrix_from_col_major(x_col, n, nrhs, ldx_, layout,           \
                                      x, ldx, sizeof(MAT_TYPE)) != 0) {        \
      free(ab_col);                                                             \
      free(afb_col);                                                            \
      free(b_col);                                                              \
      free(x_col);                                                              \
      free(work);                                                               \
      free(rwork);                                                              \
      return -1;                                                                \
    }                                                                           \
    free(ab_col);                                                               \
    free(afb_col);                                                              \
    free(b_col);                                                                \
    free(x_col);                                                                \
  }                                                                             \
  free(work);                                                                   \
  free(rwork);                                                                  \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    char *uplo, int *n, int *kd, int *nrhs, MAT_TYPE *ab, int *ldab,           \
    MAT_TYPE *afb, int *ldafb, MAT_TYPE *b, int *ldb, MAT_TYPE *x, int *ldx,  \
    REAL_TYPE *ferr, REAL_TYPE *berr, MAT_TYPE *work, REAL_TYPE *rwork,        \
    int *info) {                                                                \
  fb_uplo_t uplo_ = FB_UPPER;                                                   \
  (void)work;                                                                   \
  (void)rwork;                                                                  \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!uplo || !ferr || !berr || fb_lapack_char_to_uplo(*uplo, &uplo_) != 0) {\
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, fb_uplo_t, int, int, int, const MAT_TYPE *, \
                     int, const MAT_TYPE *, int, const MAT_TYPE *, int,        \
                     MAT_TYPE *, int, REAL_TYPE *, REAL_TYPE *))               \
      g_cblas_fn[OP_ID])(FB_LAYOUT_COL_MAJOR, uplo_, *n, *kd, *nrhs,           \
                         ab, *ldab, afb, *ldafb, b, *ldb, x, *ldx,             \
                         ferr, berr);                                           \
  if (info) *info = rc;                                                         \
}

#define DEFINE_LAPACK_PP_REFINE_REAL_THUNKS(OP, REAL_TYPE, OP_ID)              \
static int thunk_##OP##_fortran_to_cblas(                                       \
    const fb_layout_t layout, const fb_uplo_t uplo, const int n,                \
    const int nrhs, const REAL_TYPE *ap, const REAL_TYPE *afp,                  \
    const REAL_TYPE *b, const int ldb, REAL_TYPE *x, const int ldx,             \
    REAL_TYPE *ferr, REAL_TYPE *berr) {                                          \
  char uplo_char = 'U';                                                          \
  int n_ = n, nrhs_ = nrhs, ldb_ = (n > 0) ? n : 1, ldx_ = (n > 0) ? n : 1,    \
      info = 0;                                                                  \
  int work_len = (n > 0) ? n : 1;                                                \
  int iwork_len = (n > 0) ? n : 1;                                               \
  size_t packed_len = (size_t)((n > 0) ? n : 1) *                                \
                      (size_t)(((n + 1) > 0) ? (n + 1) : 2) / 2U;               \
  size_t b_elems = (size_t)ldb_ * (size_t)((nrhs > 0) ? nrhs : 1);              \
  size_t x_elems = (size_t)ldx_ * (size_t)((nrhs > 0) ? nrhs : 1);              \
  REAL_TYPE *ap_col = NULL;                                                      \
  REAL_TYPE *afp_col = NULL;                                                     \
  REAL_TYPE *b_col = NULL;                                                       \
  REAL_TYPE *x_col = NULL;                                                       \
  const REAL_TYPE *ap_arg = ap;                                                  \
  const REAL_TYPE *afp_arg = afp;                                                \
  const REAL_TYPE *b_arg = b;                                                    \
  REAL_TYPE *x_arg = x;                                                          \
  REAL_TYPE *work = NULL;                                                        \
  int *iwork = NULL;                                                             \
  if (!g_fortran_fn[OP_ID]) return -1;                                           \
  if (fb_lapack_uplo_to_char(uplo, &uplo_char) != 0 || !ferr || !berr) return -1;\
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                           \
    if ((n > 0 && (!ap || !afp)) || (n > 0 && nrhs > 0 && (!b || !x)) ||        \
        ldb < nrhs || ldx < nrhs) {                                              \
      return -1;                                                                 \
    }                                                                            \
    ap_col = (REAL_TYPE *)malloc(packed_len * sizeof(REAL_TYPE));                \
    afp_col = (REAL_TYPE *)malloc(packed_len * sizeof(REAL_TYPE));               \
    b_col = (REAL_TYPE *)malloc(b_elems * sizeof(REAL_TYPE));                    \
    x_col = (REAL_TYPE *)malloc(x_elems * sizeof(REAL_TYPE));                    \
    if (!ap_col || !afp_col || !b_col || !x_col) {                               \
      free(ap_col); free(afp_col); free(b_col); free(x_col);                     \
      return -101;                                                               \
    }                                                                            \
    if (fb_copy_triangular_packed_to_col_major(ap, layout, uplo_char, 'N', n,   \
                                               ap_col, sizeof(REAL_TYPE)) != 0 ||\
        fb_copy_triangular_packed_to_col_major(afp, layout, uplo_char, 'N', n,  \
                                               afp_col, sizeof(REAL_TYPE)) != 0 ||\
        fb_copy_matrix_to_col_major(b, layout, n, nrhs, ldb,                    \
                                    b_col, ldb_, sizeof(REAL_TYPE)) != 0 ||     \
        fb_copy_matrix_to_col_major(x, layout, n, nrhs, ldx,                    \
                                    x_col, ldx_, sizeof(REAL_TYPE)) != 0) {     \
      free(ap_col); free(afp_col); free(b_col); free(x_col);                     \
      return -1;                                                                 \
    }                                                                            \
    ap_arg = ap_col; afp_arg = afp_col; b_arg = b_col; x_arg = x_col;            \
  } else if (fb_lapack_layout_is_col_major(layout)) {                            \
    ldb_ = ldb;                                                                  \
    ldx_ = ldx;                                                                  \
    if ((n > 0 && (!ap || !afp)) || (n > 0 && nrhs > 0 && (!b || !x))) {       \
      return -1;                                                                 \
    }                                                                            \
  } else {                                                                       \
    return -1;                                                                   \
  }                                                                              \
  work = (REAL_TYPE *)malloc((size_t)work_len * sizeof(REAL_TYPE));              \
  iwork = (int *)malloc((size_t)iwork_len * sizeof(int));                        \
  if (!work || !iwork) {                                                         \
    free(ap_col); free(afp_col); free(b_col); free(x_col); free(work); free(iwork);\
    return -101;                                                                 \
  }                                                                              \
  ((void (*)(char *, int *, int *, REAL_TYPE *, REAL_TYPE *, REAL_TYPE *, int *,\
             REAL_TYPE *, int *, REAL_TYPE *, REAL_TYPE *, REAL_TYPE *, int *,  \
             int *))g_fortran_fn[OP_ID])(                                        \
      &uplo_char, &n_, &nrhs_, (REAL_TYPE *)ap_arg, (REAL_TYPE *)afp_arg,       \
      (REAL_TYPE *)b_arg, &ldb_, x_arg, &ldx_, ferr, berr, work, iwork, &info); \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                           \
    if (fb_copy_matrix_from_col_major(x_col, n, nrhs, ldx_, layout,             \
                                      x, ldx, sizeof(REAL_TYPE)) != 0) {        \
      free(ap_col); free(afp_col); free(b_col); free(x_col); free(work); free(iwork);\
      return -1;                                                                 \
    }                                                                            \
    free(ap_col); free(afp_col); free(b_col); free(x_col);                       \
  }                                                                              \
  free(work); free(iwork);                                                       \
  return info;                                                                   \
}                                                                                \
static void thunk_##OP##_cblas_to_fortran(                                       \
    char *uplo, int *n, int *nrhs, REAL_TYPE *ap, REAL_TYPE *afp,              \
    REAL_TYPE *b, int *ldb, REAL_TYPE *x, int *ldx, REAL_TYPE *ferr,           \
    REAL_TYPE *berr, REAL_TYPE *work, int *iwork, int *info) {                 \
  fb_uplo_t uplo_ = FB_UPPER;                                                    \
  (void)work; (void)iwork;                                                       \
  if (!g_cblas_fn[OP_ID]) { if (info) *info = -1; return; }                     \
  if (!uplo || !ferr || !berr || fb_lapack_char_to_uplo(*uplo, &uplo_) != 0) { \
    if (info) *info = -1;                                                        \
    return;                                                                      \
  }                                                                              \
  int rc = ((int (*)(fb_layout_t, fb_uplo_t, int, int, const REAL_TYPE *,      \
                     const REAL_TYPE *, const REAL_TYPE *, int, REAL_TYPE *,    \
                     int, REAL_TYPE *, REAL_TYPE *))g_cblas_fn[OP_ID])(         \
      FB_LAYOUT_COL_MAJOR, uplo_, *n, *nrhs, ap, afp, b, *ldb, x, *ldx, ferr, berr);\
  if (info) *info = rc;                                                          \
}

#define DEFINE_LAPACK_PP_REFINE_COMPLEX_THUNKS(OP, MAT_TYPE, REAL_TYPE, OP_ID) \
static int thunk_##OP##_fortran_to_cblas(                                       \
    const fb_layout_t layout, const fb_uplo_t uplo, const int n,                \
    const int nrhs, const MAT_TYPE *ap, const MAT_TYPE *afp,                    \
    const MAT_TYPE *b, const int ldb, MAT_TYPE *x, const int ldx,               \
    REAL_TYPE *ferr, REAL_TYPE *berr) {                                          \
  char uplo_char = 'U';                                                          \
  int n_ = n, nrhs_ = nrhs, ldb_ = (n > 0) ? n : 1, ldx_ = (n > 0) ? n : 1,    \
      info = 0;                                                                  \
  int work_len = (n > 0) ? n : 1;                                                \
  int rwork_len = (n > 0) ? n : 1;                                               \
  size_t packed_len = (size_t)((n > 0) ? n : 1) *                                \
                      (size_t)(((n + 1) > 0) ? (n + 1) : 2) / 2U;               \
  size_t b_elems = (size_t)ldb_ * (size_t)((nrhs > 0) ? nrhs : 1);              \
  size_t x_elems = (size_t)ldx_ * (size_t)((nrhs > 0) ? nrhs : 1);              \
  MAT_TYPE *ap_col = NULL;                                                       \
  MAT_TYPE *afp_col = NULL;                                                      \
  MAT_TYPE *b_col = NULL;                                                        \
  MAT_TYPE *x_col = NULL;                                                        \
  const MAT_TYPE *ap_arg = ap;                                                   \
  const MAT_TYPE *afp_arg = afp;                                                 \
  const MAT_TYPE *b_arg = b;                                                     \
  MAT_TYPE *x_arg = x;                                                           \
  MAT_TYPE *work = NULL;                                                         \
  REAL_TYPE *rwork = NULL;                                                       \
  if (!g_fortran_fn[OP_ID]) return -1;                                           \
  if (fb_lapack_uplo_to_char(uplo, &uplo_char) != 0 || !ferr || !berr) return -1;\
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                           \
    if ((n > 0 && (!ap || !afp)) || (n > 0 && nrhs > 0 && (!b || !x)) ||        \
        ldb < nrhs || ldx < nrhs) {                                              \
      return -1;                                                                 \
    }                                                                            \
    ap_col = (MAT_TYPE *)malloc(packed_len * sizeof(MAT_TYPE));                  \
    afp_col = (MAT_TYPE *)malloc(packed_len * sizeof(MAT_TYPE));                 \
    b_col = (MAT_TYPE *)malloc(b_elems * sizeof(MAT_TYPE));                      \
    x_col = (MAT_TYPE *)malloc(x_elems * sizeof(MAT_TYPE));                      \
    if (!ap_col || !afp_col || !b_col || !x_col) {                               \
      free(ap_col); free(afp_col); free(b_col); free(x_col);                     \
      return -101;                                                               \
    }                                                                            \
    if (fb_copy_triangular_packed_to_col_major(ap, layout, uplo_char, 'N', n,   \
                                               ap_col, sizeof(MAT_TYPE)) != 0 || \
        fb_copy_triangular_packed_to_col_major(afp, layout, uplo_char, 'N', n,  \
                                               afp_col, sizeof(MAT_TYPE)) != 0 ||\
        fb_copy_matrix_to_col_major(b, layout, n, nrhs, ldb,                    \
                                    b_col, ldb_, sizeof(MAT_TYPE)) != 0 ||      \
        fb_copy_matrix_to_col_major(x, layout, n, nrhs, ldx,                    \
                                    x_col, ldx_, sizeof(MAT_TYPE)) != 0) {      \
      free(ap_col); free(afp_col); free(b_col); free(x_col);                     \
      return -1;                                                                 \
    }                                                                            \
    ap_arg = ap_col; afp_arg = afp_col; b_arg = b_col; x_arg = x_col;            \
  } else if (fb_lapack_layout_is_col_major(layout)) {                            \
    ldb_ = ldb;                                                                  \
    ldx_ = ldx;                                                                  \
    if ((n > 0 && (!ap || !afp)) || (n > 0 && nrhs > 0 && (!b || !x))) {       \
      return -1;                                                                 \
    }                                                                            \
  } else {                                                                       \
    return -1;                                                                   \
  }                                                                              \
  work = (MAT_TYPE *)malloc((size_t)work_len * sizeof(MAT_TYPE));                \
  rwork = (REAL_TYPE *)malloc((size_t)rwork_len * sizeof(REAL_TYPE));            \
  if (!work || !rwork) {                                                         \
    free(ap_col); free(afp_col); free(b_col); free(x_col); free(work); free(rwork);\
    return -101;                                                                 \
  }                                                                              \
  ((void (*)(char *, int *, int *, MAT_TYPE *, MAT_TYPE *, MAT_TYPE *, int *,   \
             MAT_TYPE *, int *, REAL_TYPE *, REAL_TYPE *, MAT_TYPE *,           \
             REAL_TYPE *, int *))g_fortran_fn[OP_ID])(                           \
      &uplo_char, &n_, &nrhs_, (MAT_TYPE *)ap_arg, (MAT_TYPE *)afp_arg,         \
      (MAT_TYPE *)b_arg, &ldb_, x_arg, &ldx_, ferr, berr, work, rwork, &info);  \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                           \
    if (fb_copy_matrix_from_col_major(x_col, n, nrhs, ldx_, layout,             \
                                      x, ldx, sizeof(MAT_TYPE)) != 0) {         \
      free(ap_col); free(afp_col); free(b_col); free(x_col); free(work); free(rwork);\
      return -1;                                                                 \
    }                                                                            \
    free(ap_col); free(afp_col); free(b_col); free(x_col);                       \
  }                                                                              \
  free(work); free(rwork);                                                       \
  return info;                                                                   \
}                                                                                \
static void thunk_##OP##_cblas_to_fortran(                                       \
    char *uplo, int *n, int *nrhs, MAT_TYPE *ap, MAT_TYPE *afp,                \
    MAT_TYPE *b, int *ldb, MAT_TYPE *x, int *ldx, REAL_TYPE *ferr,             \
    REAL_TYPE *berr, MAT_TYPE *work, REAL_TYPE *rwork, int *info) {            \
  fb_uplo_t uplo_ = FB_UPPER;                                                    \
  (void)work; (void)rwork;                                                       \
  if (!g_cblas_fn[OP_ID]) { if (info) *info = -1; return; }                     \
  if (!uplo || !ferr || !berr || fb_lapack_char_to_uplo(*uplo, &uplo_) != 0) { \
    if (info) *info = -1;                                                        \
    return;                                                                      \
  }                                                                              \
  int rc = ((int (*)(fb_layout_t, fb_uplo_t, int, int, const MAT_TYPE *,       \
                     const MAT_TYPE *, const MAT_TYPE *, int, MAT_TYPE *,       \
                     int, REAL_TYPE *, REAL_TYPE *))g_cblas_fn[OP_ID])(         \
      FB_LAYOUT_COL_MAJOR, uplo_, *n, *nrhs, ap, afp, b, *ldb, x, *ldx, ferr, berr);\
  if (info) *info = rc;                                                          \
}

#define DEFINE_LAPACK_HP_REFINE_COMPLEX_THUNKS(OP, MAT_TYPE, REAL_TYPE, OP_ID) \
static int thunk_##OP##_fortran_to_cblas(                                       \
    const fb_layout_t layout, const fb_uplo_t uplo, const int n,                \
    const int nrhs, const MAT_TYPE *ap, const MAT_TYPE *afp, const int *ipiv,   \
    const MAT_TYPE *b, const int ldb, MAT_TYPE *x, const int ldx,               \
    REAL_TYPE *ferr, REAL_TYPE *berr) {                                          \
  char uplo_char = 'U';                                                          \
  int n_ = n, nrhs_ = nrhs, ldb_ = (n > 0) ? n : 1, ldx_ = (n > 0) ? n : 1,    \
      info = 0;                                                                  \
  int work_len = (n > 0) ? (2 * n) : 1;                                         \
  int rwork_len = (n > 0) ? n : 1;                                              \
  size_t packed_len = (size_t)((n > 0) ? n : 1) *                                \
                      (size_t)(((n + 1) > 0) ? (n + 1) : 2) / 2U;               \
  size_t b_elems = (size_t)ldb_ * (size_t)((nrhs > 0) ? nrhs : 1);              \
  size_t x_elems = (size_t)ldx_ * (size_t)((nrhs > 0) ? nrhs : 1);              \
  MAT_TYPE *ap_col = NULL;                                                       \
  MAT_TYPE *afp_col = NULL;                                                      \
  MAT_TYPE *b_col = NULL;                                                        \
  MAT_TYPE *x_col = NULL;                                                        \
  const MAT_TYPE *ap_arg = ap;                                                   \
  const MAT_TYPE *afp_arg = afp;                                                 \
  const MAT_TYPE *b_arg = b;                                                     \
  MAT_TYPE *x_arg = x;                                                           \
  MAT_TYPE *work = NULL;                                                         \
  REAL_TYPE *rwork = NULL;                                                       \
  if (!g_fortran_fn[OP_ID]) return -1;                                           \
  if (fb_lapack_uplo_to_char(uplo, &uplo_char) != 0 || !ipiv || !ferr || !berr) \
    return -1;                                                                   \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                           \
    if ((n > 0 && (!ap || !afp)) || (n > 0 && nrhs > 0 && (!b || !x)) ||        \
        ldb < nrhs || ldx < nrhs) {                                              \
      return -1;                                                                 \
    }                                                                            \
    ap_col = (MAT_TYPE *)malloc(packed_len * sizeof(MAT_TYPE));                  \
    afp_col = (MAT_TYPE *)malloc(packed_len * sizeof(MAT_TYPE));                 \
    b_col = (MAT_TYPE *)malloc(b_elems * sizeof(MAT_TYPE));                      \
    x_col = (MAT_TYPE *)malloc(x_elems * sizeof(MAT_TYPE));                      \
    if (!ap_col || !afp_col || !b_col || !x_col) {                               \
      free(ap_col); free(afp_col); free(b_col); free(x_col);                     \
      return -101;                                                               \
    }                                                                            \
    if (fb_copy_triangular_packed_to_col_major(ap, layout, uplo_char, 'N', n,   \
                                               ap_col, sizeof(MAT_TYPE)) != 0 || \
        fb_copy_triangular_packed_to_col_major(afp, layout, uplo_char, 'N', n,  \
                                               afp_col, sizeof(MAT_TYPE)) != 0 ||\
        fb_copy_matrix_to_col_major(b, layout, n, nrhs, ldb,                    \
                                    b_col, ldb_, sizeof(MAT_TYPE)) != 0 ||      \
        fb_copy_matrix_to_col_major(x, layout, n, nrhs, ldx,                    \
                                    x_col, ldx_, sizeof(MAT_TYPE)) != 0) {      \
      free(ap_col); free(afp_col); free(b_col); free(x_col);                     \
      return -1;                                                                 \
    }                                                                            \
    ap_arg = ap_col; afp_arg = afp_col; b_arg = b_col; x_arg = x_col;            \
  } else if (fb_lapack_layout_is_col_major(layout)) {                            \
    ldb_ = ldb;                                                                  \
    ldx_ = ldx;                                                                  \
    if ((n > 0 && (!ap || !afp)) || (n > 0 && nrhs > 0 && (!b || !x))) {       \
      return -1;                                                                 \
    }                                                                            \
  } else {                                                                       \
    return -1;                                                                   \
  }                                                                              \
  work = (MAT_TYPE *)malloc((size_t)work_len * sizeof(MAT_TYPE));                \
  rwork = (REAL_TYPE *)malloc((size_t)rwork_len * sizeof(REAL_TYPE));            \
  if (!work || !rwork) {                                                         \
    free(ap_col); free(afp_col); free(b_col); free(x_col); free(work); free(rwork);\
    return -101;                                                                 \
  }                                                                              \
  ((void (*)(char *, int *, int *, MAT_TYPE *, MAT_TYPE *, int *, MAT_TYPE *,   \
             int *, MAT_TYPE *, int *, REAL_TYPE *, REAL_TYPE *, MAT_TYPE *,    \
             REAL_TYPE *, int *))g_fortran_fn[OP_ID])(                           \
      &uplo_char, &n_, &nrhs_, (MAT_TYPE *)ap_arg, (MAT_TYPE *)afp_arg,         \
      (int *)ipiv, (MAT_TYPE *)b_arg, &ldb_, x_arg, &ldx_, ferr, berr, work,    \
      rwork, &info);                                                             \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                           \
    if (fb_copy_matrix_from_col_major(x_col, n, nrhs, ldx_, layout,             \
                                      x, ldx, sizeof(MAT_TYPE)) != 0) {         \
      free(ap_col); free(afp_col); free(b_col); free(x_col); free(work); free(rwork);\
      return -1;                                                                 \
    }                                                                            \
    free(ap_col); free(afp_col); free(b_col); free(x_col);                       \
  }                                                                              \
  free(work); free(rwork);                                                       \
  return info;                                                                   \
}                                                                                \
static void thunk_##OP##_cblas_to_fortran(                                       \
    char *uplo, int *n, int *nrhs, MAT_TYPE *ap, MAT_TYPE *afp, int *ipiv,      \
    MAT_TYPE *b, int *ldb, MAT_TYPE *x, int *ldx, REAL_TYPE *ferr,             \
    REAL_TYPE *berr, MAT_TYPE *work, REAL_TYPE *rwork, int *info) {            \
  fb_uplo_t uplo_ = FB_UPPER;                                                    \
  (void)work; (void)rwork;                                                       \
  if (!g_cblas_fn[OP_ID]) { if (info) *info = -1; return; }                     \
  if (!uplo || !ipiv || !ferr || !berr ||                                        \
      fb_lapack_char_to_uplo(*uplo, &uplo_) != 0) {                             \
    if (info) *info = -1;                                                        \
    return;                                                                      \
  }                                                                              \
  int rc = ((int (*)(fb_layout_t, fb_uplo_t, int, int, const MAT_TYPE *,        \
                     const MAT_TYPE *, const int *, const MAT_TYPE *, int,      \
                     MAT_TYPE *, int, REAL_TYPE *, REAL_TYPE *))g_cblas_fn[OP_ID])(\
      FB_LAYOUT_COL_MAJOR, uplo_, *n, *nrhs, ap, afp, ipiv, b, *ldb, x, *ldx,  \
      ferr, berr);                                                               \
  if (info) *info = rc;                                                          \
}

#define DEFINE_LAPACK_UPLO_PIVOTED_QUERY_SOLVE_THUNKS(OP, TYPE, OP_ID, QUERY_FN) \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const char uplo, const int n,                    \
    const int nrhs, TYPE *A, const int lda, int *ipiv, TYPE *B,                \
    const int ldb) {                                                            \
  fb_uplo_t uplo_enum = FB_UPPER;                                               \
  char uplo_char = 'U';                                                         \
  int n_ = n, nrhs_ = nrhs, lda_ = (n > 0) ? n : 1,                            \
      ldb_ = (n > 0) ? n : 1, lwork_ = -1, info = 0;                           \
  size_t a_elems = (size_t)lda_ * (size_t)((n > 0) ? n : 1);                   \
  size_t b_elems = (size_t)ldb_ * (size_t)((nrhs > 0) ? nrhs : 1);             \
  TYPE work_query = (TYPE)0;                                                    \
  TYPE *a_col = NULL;                                                           \
  TYPE *b_col = NULL;                                                           \
  TYPE *work = NULL;                                                            \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (fb_lapack_char_to_uplo(uplo, &uplo_enum) != 0) return -1;                \
  uplo_char = fb_lapack_uplo_from_layout(layout, uplo_enum);                   \
  a_col = (TYPE *)malloc(a_elems * sizeof(TYPE));                               \
  b_col = (TYPE *)malloc(b_elems * sizeof(TYPE));                               \
  if (!a_col || !b_col) {                                                       \
    free(a_col);                                                                \
    free(b_col);                                                                \
    return -101;                                                                \
  }                                                                             \
  if (fb_copy_matrix_to_col_major(A, layout, n, n, lda,                        \
                                  a_col, lda_, sizeof(TYPE)) != 0 ||           \
      fb_copy_matrix_to_col_major(B, layout, n, nrhs, ldb,                     \
                                  b_col, ldb_, sizeof(TYPE)) != 0) {           \
    free(a_col);                                                                \
    free(b_col);                                                                \
    return -1;                                                                  \
  }                                                                             \
  ((void (*)(char *, int *, int *, TYPE *, int *, int *, TYPE *, int *,        \
             TYPE *, int *, int *))g_fortran_fn[OP_ID])(                       \
      &uplo_char, &n_, &nrhs_, a_col, &lda_, ipiv, b_col, &ldb_,               \
      &work_query, &lwork_, &info);                                             \
  if (info != 0) {                                                              \
    free(a_col);                                                                \
    free(b_col);                                                                \
    return info;                                                                \
  }                                                                             \
  lwork_ = QUERY_FN(work_query);                                                \
  work = (TYPE *)malloc((size_t)lwork_ * sizeof(TYPE));                         \
  if (!work) {                                                                  \
    free(a_col);                                                                \
    free(b_col);                                                                \
    return -101;                                                                \
  }                                                                             \
  if (fb_copy_matrix_to_col_major(A, layout, n, n, lda,                        \
                                  a_col, lda_, sizeof(TYPE)) != 0 ||           \
      fb_copy_matrix_to_col_major(B, layout, n, nrhs, ldb,                     \
                                  b_col, ldb_, sizeof(TYPE)) != 0) {           \
    free(a_col);                                                                \
    free(b_col);                                                                \
    free(work);                                                                 \
    return -1;                                                                  \
  }                                                                             \
  ((void (*)(char *, int *, int *, TYPE *, int *, int *, TYPE *, int *,        \
             TYPE *, int *, int *))g_fortran_fn[OP_ID])(                       \
      &uplo_char, &n_, &nrhs_, a_col, &lda_, ipiv, b_col, &ldb_,               \
      work, &lwork_, &info);                                                    \
  if (fb_copy_matrix_from_col_major(a_col, n, n, lda_, layout,                 \
                                    A, lda, sizeof(TYPE)) != 0 ||              \
      fb_copy_matrix_from_col_major(b_col, n, nrhs, ldb_, layout,              \
                                    B, ldb, sizeof(TYPE)) != 0) {              \
    free(a_col);                                                                \
    free(b_col);                                                                \
    free(work);                                                                 \
    return -1;                                                                  \
  }                                                                             \
  free(a_col);                                                                  \
  free(b_col);                                                                  \
  free(work);                                                                   \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    char *uplo, int *n, int *nrhs, TYPE *A, int *lda, int *ipiv, TYPE *B,      \
    int *ldb, TYPE *work, int *lwork, int *info) {                              \
  fb_uplo_t uplo_enum = FB_UPPER;                                               \
  char uplo_char = 'U';                                                         \
  (void)work;                                                                   \
  (void)lwork;                                                                  \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!uplo || fb_lapack_char_to_uplo(*uplo, &uplo_enum) != 0) {               \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  uplo_char = (uplo_enum == FB_UPPER) ? 'U' : 'L';                             \
  int rc = ((int (*)(fb_layout_t, char, int, int, TYPE *, int, int *, TYPE *,  \
                     int))g_cblas_fn[OP_ID])(                                   \
      FB_LAYOUT_COL_MAJOR, uplo_char, *n, *nrhs, A, *lda, ipiv, B, *ldb);      \
  if (info) *info = rc;                                                         \
}

#define DEFINE_SYEV_REAL_FORTRAN_TO_CBLAS(OP, TYPE, OP_ID, QUERY_FN)           \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const char jobz, const fb_uplo_t uplo,           \
    const int n, TYPE *A, const int lda, TYPE *w) {                            \
  char jobz_ = jobz;                                                           \
  char uplo_ = fb_lapack_uplo_from_layout(layout, uplo);                       \
  int n_ = n, lda_ = lda, lwork_ = -1, info = 0;                               \
  TYPE work_query = (TYPE)0;                                                   \
  TYPE *work = NULL;                                                           \
  if (!g_fortran_fn[OP_ID]) return -1;                                         \
  ((void (*)(char *, char *, int *, TYPE *, int *, TYPE *, TYPE *, int *,      \
             int *))g_fortran_fn[OP_ID])(&jobz_, &uplo_, &n_, A, &lda_, w,     \
                                         &work_query, &lwork_, &info);          \
  if (info != 0) return info;                                                  \
  lwork_ = QUERY_FN(work_query);                                               \
  work = (TYPE *)malloc((size_t)lwork_ * sizeof(TYPE));                        \
  if (!work) return -101;                                                      \
  ((void (*)(char *, char *, int *, TYPE *, int *, TYPE *, TYPE *, int *,      \
             int *))g_fortran_fn[OP_ID])(&jobz_, &uplo_, &n_, A, &lda_, w,     \
                                         work, &lwork_, &info);                \
  free(work);                                                                  \
  return info;                                                                 \
}

#define DEFINE_HEEV_COMPLEX_FORTRAN_TO_CBLAS(OP, CTYPE, RTYPE, OP_ID, QUERY_FN) \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const char jobz, const fb_uplo_t uplo,           \
    const int n, CTYPE *A, const int lda, RTYPE *w) {                          \
  char jobz_ = jobz;                                                           \
  char uplo_ = fb_lapack_uplo_from_layout(layout, uplo);                       \
  int n_ = n, lda_ = lda, lwork_ = -1, info = 0;                               \
  CTYPE work_query = (CTYPE)0;                                                 \
  RTYPE rwork_query = (RTYPE)0;                                                \
  CTYPE *work = NULL;                                                          \
  RTYPE *rwork = NULL;                                                         \
  int rwork_size = (n > 1) ? (3 * n) : 1;                                      \
  if (!g_fortran_fn[OP_ID]) return -1;                                         \
  ((void (*)(char *, char *, int *, CTYPE *, int *, RTYPE *, CTYPE *, int *,   \
             RTYPE *, int *))g_fortran_fn[OP_ID])(&jobz_, &uplo_, &n_, A,      \
                                                  &lda_, w, &work_query,        \
                                                  &lwork_, &rwork_query, &info);\
  if (info != 0) return info;                                                  \
  lwork_ = QUERY_FN(work_query);                                               \
  work = (CTYPE *)malloc((size_t)lwork_ * sizeof(CTYPE));                      \
  rwork = (RTYPE *)malloc((size_t)rwork_size * sizeof(RTYPE));                 \
  if (!work || !rwork) {                                                       \
    free(work);                                                                \
    free(rwork);                                                               \
    return -101;                                                               \
  }                                                                            \
  ((void (*)(char *, char *, int *, CTYPE *, int *, RTYPE *, CTYPE *, int *,   \
             RTYPE *, int *))g_fortran_fn[OP_ID])(&jobz_, &uplo_, &n_, A,      \
                                                  &lda_, w, work, &lwork_,      \
                                                  rwork, &info);               \
  free(work);                                                                  \
  free(rwork);                                                                 \
  return info;                                                                 \
}

#define DEFINE_SYEV_REAL_CBLAS_TO_FORTRAN(OP, TYPE, OP_ID)                     \
static void thunk_##OP##_cblas_to_fortran(                                     \
    char *jobz, char *uplo, int *n, TYPE *A, int *lda, TYPE *w, TYPE *work,   \
    int *lwork, int *info) {                                                    \
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
  int rc = ((int (*)(fb_layout_t, char, fb_uplo_t, int, TYPE *, int, TYPE *)) \
      g_cblas_fn[OP_ID])(FB_LAYOUT_COL_MAJOR, *jobz, uplo_enum, *n, A, *lda,  \
                          w);                                                   \
  if (info) *info = rc;                                                         \
}

#define DEFINE_HEEV_COMPLEX_CBLAS_TO_FORTRAN(OP, CTYPE, RTYPE, OP_ID)         \
static void thunk_##OP##_cblas_to_fortran(                                     \
    char *jobz, char *uplo, int *n, CTYPE *A, int *lda, RTYPE *w, CTYPE *work,\
    int *lwork, RTYPE *rwork, int *info) {                                      \
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
  int rc = ((int (*)(fb_layout_t, char, fb_uplo_t, int, CTYPE *, int, RTYPE *))\
      g_cblas_fn[OP_ID])(FB_LAYOUT_COL_MAJOR, *jobz, uplo_enum, *n, A, *lda,  \
                          w);                                                   \
  if (info) *info = rc;                                                         \
}

#define DEFINE_SYEVD_REAL_FORTRAN_TO_CBLAS(OP, TYPE, OP_ID, QUERY_FN)          \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const char jobz, const fb_uplo_t uplo,           \
    const int n, TYPE *A, const int lda, TYPE *w) {                            \
  char jobz_ = jobz;                                                           \
  char uplo_ = fb_lapack_uplo_from_layout(layout, uplo);                       \
  int n_ = n, lda_ = lda, lwork_ = -1, liwork_ = -1, info = 0;                 \
  TYPE work_query = (TYPE)0;                                                   \
  int iwork_query = 0;                                                         \
  TYPE *work = NULL;                                                           \
  int *iwork = NULL;                                                           \
  if (!g_fortran_fn[OP_ID]) return -1;                                         \
  ((void (*)(char *, char *, int *, TYPE *, int *, TYPE *, TYPE *, int *,      \
             int *, int *, int *))g_fortran_fn[OP_ID])(&jobz_, &uplo_, &n_, A, \
                                                        &lda_, w, &work_query,  \
                                                        &lwork_, &iwork_query,  \
                                                        &liwork_, &info);      \
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
  ((void (*)(char *, char *, int *, TYPE *, int *, TYPE *, TYPE *, int *,      \
             int *, int *, int *))g_fortran_fn[OP_ID])(&jobz_, &uplo_, &n_, A, \
                                                        &lda_, w, work,         \
                                                        &lwork_, iwork,         \
                                                        &liwork_, &info);      \
  free(work);                                                                  \
  free(iwork);                                                                 \
  return info;                                                                 \
}

#define DEFINE_HEEVD_COMPLEX_FORTRAN_TO_CBLAS(OP, CTYPE, RTYPE, OP_ID, QUERY_FN) \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const char jobz, const fb_uplo_t uplo,           \
    const int n, CTYPE *A, const int lda, RTYPE *w) {                          \
  char jobz_ = jobz;                                                           \
  char uplo_ = fb_lapack_uplo_from_layout(layout, uplo);                       \
  int n_ = n, lda_ = lda;                                                      \
  int lwork_ = -1, lrwork_ = -1, liwork_ = -1, info = 0;                       \
  CTYPE work_query = (CTYPE)0;                                                 \
  RTYPE rwork_query = (RTYPE)0;                                                \
  int iwork_query = 0;                                                         \
  CTYPE *work = NULL;                                                          \
  RTYPE *rwork = NULL;                                                         \
  int *iwork = NULL;                                                           \
  if (!g_fortran_fn[OP_ID]) return -1;                                         \
  ((void (*)(char *, char *, int *, CTYPE *, int *, RTYPE *, CTYPE *, int *,   \
             RTYPE *, int *, int *, int *, int *))g_fortran_fn[OP_ID])(        \
      &jobz_, &uplo_, &n_, A, &lda_, w, &work_query, &lwork_,                  \
      &rwork_query, &lrwork_, &iwork_query, &liwork_, &info);                  \
  if (info != 0) return info;                                                  \
  lwork_ = QUERY_FN(work_query);                                               \
  lrwork_ = (rwork_query > (RTYPE)1) ? (int)rwork_query : 1;                   \
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
  ((void (*)(char *, char *, int *, CTYPE *, int *, RTYPE *, CTYPE *, int *,   \
             RTYPE *, int *, int *, int *, int *))g_fortran_fn[OP_ID])(        \
      &jobz_, &uplo_, &n_, A, &lda_, w, work, &lwork_, rwork, &lrwork_,        \
      iwork, &liwork_, &info);                                                 \
  free(work);                                                                  \
  free(rwork);                                                                 \
  free(iwork);                                                                 \
  return info;                                                                 \
}

#define DEFINE_SYEVD_REAL_CBLAS_TO_FORTRAN(OP, TYPE, OP_ID)                    \
static void thunk_##OP##_cblas_to_fortran(                                     \
    char *jobz, char *uplo, int *n, TYPE *A, int *lda, TYPE *w, TYPE *work,   \
    int *lwork, int *iwork, int *liwork, int *info) {                          \
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
  int rc = ((int (*)(fb_layout_t, char, fb_uplo_t, int, TYPE *, int, TYPE *)) \
      g_cblas_fn[OP_ID])(FB_LAYOUT_COL_MAJOR, *jobz, uplo_enum, *n, A, *lda,  \
                          w);                                                   \
  if (info) *info = rc;                                                         \
}

#define DEFINE_HEEVD_COMPLEX_CBLAS_TO_FORTRAN(OP, CTYPE, RTYPE, OP_ID)        \
static void thunk_##OP##_cblas_to_fortran(                                     \
    char *jobz, char *uplo, int *n, CTYPE *A, int *lda, RTYPE *w, CTYPE *work,\
    int *lwork, RTYPE *rwork, int *lrwork, int *iwork, int *liwork,           \
    int *info) {                                                                \
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
  int rc = ((int (*)(fb_layout_t, char, fb_uplo_t, int, CTYPE *, int, RTYPE *))\
      g_cblas_fn[OP_ID])(FB_LAYOUT_COL_MAJOR, *jobz, uplo_enum, *n, A, *lda,  \
                          w);                                                   \
  if (info) *info = rc;                                                         \
}

#define DEFINE_HEGVD_COMPLEX_FORTRAN_TO_CBLAS(OP, CTYPE, RTYPE, OP_ID, QUERY_FN) \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const int itype, char jobz, const fb_uplo_t uplo,\
    const int n, CTYPE *A, const int lda, CTYPE *B, const int ldb, RTYPE *w) { \
  char jobz_ = jobz;                                                           \
  char uplo_ = fb_lapack_uplo_from_layout(layout, uplo);                       \
  int itype_ = itype, n_ = n, lda_ = lda, ldb_ = ldb;                          \
  int lwork_ = -1, lrwork_ = -1, liwork_ = -1, info = 0;                       \
  CTYPE work_query = (CTYPE)0;                                                 \
  RTYPE rwork_query = (RTYPE)0;                                                \
  int iwork_query = 0;                                                         \
  CTYPE *work = NULL;                                                          \
  RTYPE *rwork = NULL;                                                         \
  int *iwork = NULL;                                                           \
  if (!g_fortran_fn[OP_ID]) return -1;                                         \
  ((void (*)(int *, char *, char *, int *, CTYPE *, int *, CTYPE *, int *,     \
             RTYPE *, CTYPE *, int *, RTYPE *, int *, int *, int *, int *))   \
      g_fortran_fn[OP_ID])(&itype_, &jobz_, &uplo_, &n_, A, &lda_, B, &ldb_,   \
                           w, &work_query, &lwork_, &rwork_query, &lrwork_,    \
                           &iwork_query, &liwork_, &info);                     \
  if (info != 0) return info;                                                  \
  lwork_ = QUERY_FN(work_query);                                               \
  lrwork_ = (rwork_query > (RTYPE)1) ? (int)rwork_query : 1;                   \
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
  ((void (*)(int *, char *, char *, int *, CTYPE *, int *, CTYPE *, int *,     \
             RTYPE *, CTYPE *, int *, RTYPE *, int *, int *, int *, int *))   \
      g_fortran_fn[OP_ID])(&itype_, &jobz_, &uplo_, &n_, A, &lda_, B, &ldb_,   \
                           w, work, &lwork_, rwork, &lrwork_, iwork,           \
                           &liwork_, &info);                                   \
  free(work);                                                                  \
  free(rwork);                                                                 \
  free(iwork);                                                                 \
  return info;                                                                 \
}

#define DEFINE_GEEV_REAL_FORTRAN_TO_CBLAS(OP, TYPE, OP_ID, QUERY_FN)           \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const char jobvl, const char jobvr, const int n, \
    TYPE *A, const int lda, TYPE *wr, TYPE *wi, TYPE *VL, const int ldvl,      \
    TYPE *VR, const int ldvr) {                                                \
  if (!fb_lapack_layout_is_col_major(layout)) return -1;                       \
  char jobvl_ = jobvl;                                                         \
  char jobvr_ = jobvr;                                                         \
  if (fb_lapack_job_requests_vectors(jobvl_) && VL == NULL) return -1;         \
  if (fb_lapack_job_requests_vectors(jobvr_) && VR == NULL) return -1;         \
  int n_ = n, lda_ = lda, ldvl_ = ldvl, ldvr_ = ldvr, lwork_ = -1, info = 0;   \
  TYPE work_query = (TYPE)0;                                                   \
  TYPE *work = NULL;                                                           \
  if (!g_fortran_fn[OP_ID]) return -1;                                         \
  ((void (*)(char *, char *, int *, TYPE *, int *, TYPE *, TYPE *, TYPE *,     \
             int *, TYPE *, int *, TYPE *, int *, int *))g_fortran_fn[OP_ID])( \
      &jobvl_, &jobvr_, &n_, A, &lda_, wr, wi, VL, &ldvl_, VR, &ldvr_,         \
      &work_query, &lwork_, &info);                                            \
  if (info != 0) return info;                                                  \
  lwork_ = QUERY_FN(work_query);                                               \
  work = (TYPE *)malloc((size_t)lwork_ * sizeof(TYPE));                        \
  if (!work) return -101;                                                      \
  ((void (*)(char *, char *, int *, TYPE *, int *, TYPE *, TYPE *, TYPE *,     \
             int *, TYPE *, int *, TYPE *, int *, int *))g_fortran_fn[OP_ID])( \
      &jobvl_, &jobvr_, &n_, A, &lda_, wr, wi, VL, &ldvl_, VR, &ldvr_,         \
      work, &lwork_, &info);                                                   \
  free(work);                                                                  \
  return info;                                                                 \
}

#define DEFINE_GEEV_COMPLEX_FORTRAN_TO_CBLAS(OP, CTYPE, RTYPE, OP_ID, QUERY_FN) \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const char jobvl, const char jobvr, const int n, \
    CTYPE *A, const int lda, CTYPE *w, CTYPE *VL, const int ldvl,              \
    CTYPE *VR, const int ldvr) {                                               \
  if (!fb_lapack_layout_is_col_major(layout)) return -1;                       \
  char jobvl_ = jobvl;                                                         \
  char jobvr_ = jobvr;                                                         \
  if (fb_lapack_job_requests_vectors(jobvl_) && VL == NULL) return -1;         \
  if (fb_lapack_job_requests_vectors(jobvr_) && VR == NULL) return -1;         \
  int n_ = n, lda_ = lda, ldvl_ = ldvl, ldvr_ = ldvr, lwork_ = -1, info = 0;   \
  CTYPE work_query = (CTYPE)0;                                                 \
  RTYPE rwork_query = (RTYPE)0;                                                \
  CTYPE *work = NULL;                                                          \
  RTYPE *rwork = NULL;                                                         \
  int rwork_size = (n > 1) ? (2 * n) : 1;                                      \
  if (!g_fortran_fn[OP_ID]) return -1;                                         \
  ((void (*)(char *, char *, int *, CTYPE *, int *, CTYPE *, CTYPE *, int *,   \
             CTYPE *, int *, CTYPE *, int *, RTYPE *, int *))                  \
      g_fortran_fn[OP_ID])(&jobvl_, &jobvr_, &n_, A, &lda_, w, VL, &ldvl_,     \
                           VR, &ldvr_, &work_query, &lwork_, &rwork_query,      \
                           &info);                                             \
  if (info != 0) return info;                                                  \
  lwork_ = QUERY_FN(work_query);                                               \
  work = (CTYPE *)malloc((size_t)lwork_ * sizeof(CTYPE));                      \
  rwork = (RTYPE *)malloc((size_t)rwork_size * sizeof(RTYPE));                 \
  if (!work || !rwork) {                                                       \
    free(work);                                                                \
    free(rwork);                                                               \
    return -101;                                                               \
  }                                                                            \
  ((void (*)(char *, char *, int *, CTYPE *, int *, CTYPE *, CTYPE *, int *,   \
             CTYPE *, int *, CTYPE *, int *, RTYPE *, int *))                  \
      g_fortran_fn[OP_ID])(&jobvl_, &jobvr_, &n_, A, &lda_, w, VL, &ldvl_,     \
                           VR, &ldvr_, work, &lwork_, rwork, &info);           \
  free(work);                                                                  \
  free(rwork);                                                                 \
  return info;                                                                 \
}

#define DEFINE_GEEV_REAL_CBLAS_TO_FORTRAN(OP, TYPE, OP_ID)                    \
static void thunk_##OP##_cblas_to_fortran(                                     \
    char *jobvl, char *jobvr, int *n, TYPE *A, int *lda, TYPE *wr, TYPE *wi,  \
    TYPE *VL, int *ldvl, TYPE *VR, int *ldvr, TYPE *work, int *lwork,         \
    int *info) {                                                                \
  (void)work;                                                                   \
  (void)lwork;                                                                  \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, char, char, int, TYPE *, int, TYPE *,       \
                     TYPE *, TYPE *, int, TYPE *, int))g_cblas_fn[OP_ID])(    \
      FB_LAYOUT_COL_MAJOR, *jobvl, *jobvr, *n, A, *lda, wr, wi, VL, *ldvl,    \
      VR, *ldvr);                                                               \
  if (info) *info = rc;                                                         \
}

#define DEFINE_GEEV_COMPLEX_CBLAS_TO_FORTRAN(OP, CTYPE, RTYPE, OP_ID)         \
static void thunk_##OP##_cblas_to_fortran(                                     \
    char *jobvl, char *jobvr, int *n, CTYPE *A, int *lda, CTYPE *w, CTYPE *VL,\
    int *ldvl, CTYPE *VR, int *ldvr, CTYPE *work, int *lwork, RTYPE *rwork,   \
    int *info) {                                                                \
  (void)work;                                                                   \
  (void)lwork;                                                                  \
  (void)rwork;                                                                  \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, char, char, int, CTYPE *, int, CTYPE *,     \
                     CTYPE *, int, CTYPE *, int))g_cblas_fn[OP_ID])(          \
      FB_LAYOUT_COL_MAJOR, *jobvl, *jobvr, *n, A, *lda, w, VL, *ldvl, VR,     \
      *ldvr);                                                                   \
  if (info) *info = rc;                                                         \
}

#define DEFINE_GESVD_REAL_FORTRAN_TO_CBLAS(OP, TYPE, OP_ID, QUERY_FN)          \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const char jobu, const char jobvt,               \
    const int m, const int n, TYPE *A, const int lda, TYPE *s, TYPE *U,        \
    const int ldu, TYPE *VT, const int ldvt, TYPE *superb) {                   \
  if (!fb_lapack_layout_is_col_major(layout)) return -1;                       \
  char jobu_ = jobu, jobvt_ = jobvt;                                           \
  int m_ = m, n_ = n, lda_ = lda, ldu_ = U ? ldu : 1,                          \
      ldvt_ = VT ? ldvt : 1, lwork_ = -1, info = 0;                            \
  TYPE work_query = (TYPE)0;                                                   \
  TYPE *work = NULL;                                                           \
  TYPE dummy_u = (TYPE)0;                                                      \
  TYPE dummy_vt = (TYPE)0;                                                     \
  TYPE *u_ptr = U ? U : &dummy_u;                                              \
  TYPE *vt_ptr = VT ? VT : &dummy_vt;                                          \
  int minmn = (m < n) ? m : n;                                                 \
  if (fb_lapack_svd_job_requires_explicit_output(jobu_) && U == NULL)          \
    return -1;                                                                 \
  if (fb_lapack_svd_job_requires_explicit_output(jobvt_) && VT == NULL)        \
    return -1;                                                                 \
  if (superb != NULL && minmn > 0) {                                           \
    memset(superb, 0, (size_t)minmn * sizeof(TYPE));                           \
  }                                                                            \
  if (!g_fortran_fn[OP_ID]) return -1;                                         \
  ((void (*)(char *, char *, int *, int *, TYPE *, int *, TYPE *, TYPE *,      \
             int *, TYPE *, int *, TYPE *, int *, int *))g_fortran_fn[OP_ID])( \
      &jobu_, &jobvt_, &m_, &n_, A, &lda_, s, u_ptr, &ldu_, vt_ptr,            \
      &ldvt_, &work_query, &lwork_, &info);                                    \
  if (info != 0) return info;                                                  \
  lwork_ = QUERY_FN(work_query);                                               \
  work = (TYPE *)malloc((size_t)lwork_ * sizeof(TYPE));                        \
  if (!work) return -101;                                                      \
  ((void (*)(char *, char *, int *, int *, TYPE *, int *, TYPE *, TYPE *,      \
             int *, TYPE *, int *, TYPE *, int *, int *))g_fortran_fn[OP_ID])( \
      &jobu_, &jobvt_, &m_, &n_, A, &lda_, s, u_ptr, &ldu_, vt_ptr,            \
      &ldvt_, work, &lwork_, &info);                                           \
  if (info == 0 && superb != NULL && minmn > 1) {                              \
    int i;                                                                      \
    for (i = 0; i < (minmn - 1); ++i) {                                         \
      superb[i] = work[i + 1];                                                  \
    }                                                                           \
  }                                                                             \
  free(work);                                                                  \
  return info;                                                                 \
}

#define DEFINE_GESVD_COMPLEX_FORTRAN_TO_CBLAS(OP, CTYPE, RTYPE, OP_ID, QUERY_FN) \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const char jobu, const char jobvt,               \
    const int m, const int n, CTYPE *A, const int lda, RTYPE *s, CTYPE *U,     \
    const int ldu, CTYPE *VT, const int ldvt, RTYPE *superb) {                 \
  if (!fb_lapack_layout_is_col_major(layout)) return -1;                       \
  char jobu_ = jobu, jobvt_ = jobvt;                                           \
  int m_ = m, n_ = n, lda_ = lda, ldu_ = U ? ldu : 1,                          \
      ldvt_ = VT ? ldvt : 1, lwork_ = -1, info = 0;                            \
  CTYPE work_query = (CTYPE)0;                                                 \
  RTYPE dummy_rwork = (RTYPE)0;                                                \
  CTYPE *work = NULL;                                                          \
  RTYPE *rwork = NULL;                                                         \
  CTYPE dummy_u = (CTYPE)0;                                                    \
  CTYPE dummy_vt = (CTYPE)0;                                                   \
  CTYPE *u_ptr = U ? U : &dummy_u;                                             \
  CTYPE *vt_ptr = VT ? VT : &dummy_vt;                                         \
  int minmn = (m < n) ? m : n;                                                 \
  int rwork_size = (minmn > 1) ? (5 * minmn) : 1;                              \
  if (fb_lapack_svd_job_requires_explicit_output(jobu_) && U == NULL)          \
    return -1;                                                                 \
  if (fb_lapack_svd_job_requires_explicit_output(jobvt_) && VT == NULL)        \
    return -1;                                                                 \
  if (superb != NULL && minmn > 0) {                                           \
    memset(superb, 0, (size_t)minmn * sizeof(RTYPE));                          \
  }                                                                            \
  if (!g_fortran_fn[OP_ID]) return -1;                                         \
  ((void (*)(char *, char *, int *, int *, CTYPE *, int *, RTYPE *, CTYPE *,  \
             int *, CTYPE *, int *, CTYPE *, int *, RTYPE *, int *))          \
      g_fortran_fn[OP_ID])(&jobu_, &jobvt_, &m_, &n_, A, &lda_, s, u_ptr,      \
                           &ldu_, vt_ptr, &ldvt_, &work_query, &lwork_,        \
                           &dummy_rwork, &info);                               \
  if (info != 0) return info;                                                  \
  lwork_ = QUERY_FN(work_query);                                               \
  work = (CTYPE *)malloc((size_t)lwork_ * sizeof(CTYPE));                      \
  rwork = (RTYPE *)malloc((size_t)rwork_size * sizeof(RTYPE));                 \
  if (!work || !rwork) {                                                       \
    free(work);                                                                \
    free(rwork);                                                               \
    return -101;                                                               \
  }                                                                            \
  ((void (*)(char *, char *, int *, int *, CTYPE *, int *, RTYPE *, CTYPE *,  \
             int *, CTYPE *, int *, CTYPE *, int *, RTYPE *, int *))          \
      g_fortran_fn[OP_ID])(&jobu_, &jobvt_, &m_, &n_, A, &lda_, s, u_ptr,      \
                           &ldu_, vt_ptr, &ldvt_, work, &lwork_, rwork,        \
                           &info);                                             \
  if (info == 0 && superb != NULL && minmn > 1) {                              \
    int i;                                                                      \
    for (i = 0; i < (minmn - 1); ++i) {                                         \
      superb[i] = rwork[i];                                                     \
    }                                                                           \
  }                                                                             \
  free(work);                                                                  \
  free(rwork);                                                                 \
  return info;                                                                 \
}

#define DEFINE_GESVD_REAL_CBLAS_TO_FORTRAN(OP, TYPE, OP_ID)                    \
static void thunk_##OP##_cblas_to_fortran(                                     \
    char *jobu, char *jobvt, int *m, int *n, TYPE *A, int *lda, TYPE *s,      \
    TYPE *U, int *ldu, TYPE *VT, int *ldvt, TYPE *work, int *lwork,           \
    int *info) {                                                                \
  int minmn = (*m < *n) ? *m : *n;                                              \
  int superb_size = (minmn > 1) ? (minmn - 1) : 1;                              \
  TYPE *superb = NULL;                                                          \
  (void)work;                                                                   \
  (void)lwork;                                                                  \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  superb = (TYPE *)malloc((size_t)superb_size * sizeof(TYPE));                 \
  if (!superb) {                                                                \
    if (info) *info = -101;                                                     \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, char, char, int, int, TYPE *, int, TYPE *,  \
                     TYPE *, int, TYPE *, int, TYPE *))g_cblas_fn[OP_ID])(    \
      FB_LAYOUT_COL_MAJOR, *jobu, *jobvt, *m, *n, A, *lda, s, U, *ldu, VT,    \
      *ldvt, superb);                                                           \
  free(superb);                                                                 \
  if (info) *info = rc;                                                         \
}

#define DEFINE_GESVD_COMPLEX_CBLAS_TO_FORTRAN(OP, CTYPE, RTYPE, OP_ID)        \
static void thunk_##OP##_cblas_to_fortran(                                     \
    char *jobu, char *jobvt, int *m, int *n, CTYPE *A, int *lda, RTYPE *s,    \
    CTYPE *U, int *ldu, CTYPE *VT, int *ldvt, CTYPE *work, int *lwork,        \
    RTYPE *rwork, int *info) {                                                  \
  int minmn = (*m < *n) ? *m : *n;                                              \
  int superb_size = (minmn > 1) ? (minmn - 1) : 1;                              \
  RTYPE *superb = NULL;                                                         \
  (void)work;                                                                   \
  (void)lwork;                                                                  \
  (void)rwork;                                                                  \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  superb = (RTYPE *)malloc((size_t)superb_size * sizeof(RTYPE));               \
  if (!superb) {                                                                \
    if (info) *info = -101;                                                     \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, char, char, int, int, CTYPE *, int, RTYPE *,\
                     CTYPE *, int, CTYPE *, int, RTYPE *))g_cblas_fn[OP_ID])( \
      FB_LAYOUT_COL_MAJOR, *jobu, *jobvt, *m, *n, A, *lda, s, U, *ldu, VT,    \
      *ldvt, superb);                                                           \
  free(superb);                                                                 \
  if (info) *info = rc;                                                         \
}

#define DEFINE_GELS_REAL_FORTRAN_TO_CBLAS(OP, TYPE, OP_ID, QUERY_FN)          \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, char trans, const int m, const int n,            \
    const int nrhs, TYPE *A, const int lda, TYPE *B, const int ldb) {          \
  if (!fb_lapack_layout_is_col_major(layout)) return -1;                       \
  char trans_ = trans;                                                          \
  int m_ = m, n_ = n, nrhs_ = nrhs, lda_ = lda, ldb_ = ldb;                    \
  int lwork_ = -1, info = 0;                                                    \
  TYPE work_query = (TYPE)0;                                                    \
  TYPE *work = NULL;                                                            \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  ((void (*)(char *, int *, int *, int *, TYPE *, int *, TYPE *, int *,        \
             TYPE *, int *, int *))g_fortran_fn[OP_ID])(                       \
      &trans_, &m_, &n_, &nrhs_, A, &lda_, B, &ldb_, &work_query, &lwork_,     \
      &info);                                                                   \
  if (info != 0) return info;                                                   \
  lwork_ = QUERY_FN(work_query);                                                \
  work = (TYPE *)malloc((size_t)lwork_ * sizeof(TYPE));                         \
  if (!work) return -101;                                                       \
  ((void (*)(char *, int *, int *, int *, TYPE *, int *, TYPE *, int *,        \
             TYPE *, int *, int *))g_fortran_fn[OP_ID])(                       \
      &trans_, &m_, &n_, &nrhs_, A, &lda_, B, &ldb_, work, &lwork_, &info);    \
  free(work);                                                                   \
  return info;                                                                  \
}

#define DEFINE_GELS_COMPLEX_FORTRAN_TO_CBLAS(OP, CTYPE, OP_ID, QUERY_FN)      \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, char trans, const int m, const int n,            \
    const int nrhs, CTYPE *A, const int lda, CTYPE *B, const int ldb) {        \
  if (!fb_lapack_layout_is_col_major(layout)) return -1;                       \
  char trans_ = trans;                                                          \
  int m_ = m, n_ = n, nrhs_ = nrhs, lda_ = lda, ldb_ = ldb;                    \
  int lwork_ = -1, info = 0;                                                    \
  CTYPE work_query = (CTYPE)0;                                                  \
  CTYPE *work = NULL;                                                           \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  ((void (*)(char *, int *, int *, int *, CTYPE *, int *, CTYPE *, int *,      \
             CTYPE *, int *, int *))g_fortran_fn[OP_ID])(                      \
      &trans_, &m_, &n_, &nrhs_, A, &lda_, B, &ldb_, &work_query, &lwork_,     \
      &info);                                                                   \
  if (info != 0) return info;                                                   \
  lwork_ = QUERY_FN(work_query);                                                \
  work = (CTYPE *)malloc((size_t)lwork_ * sizeof(CTYPE));                       \
  if (!work) return -101;                                                       \
  ((void (*)(char *, int *, int *, int *, CTYPE *, int *, CTYPE *, int *,      \
             CTYPE *, int *, int *))g_fortran_fn[OP_ID])(                      \
      &trans_, &m_, &n_, &nrhs_, A, &lda_, B, &ldb_, work, &lwork_, &info);    \
  free(work);                                                                   \
  return info;                                                                  \
}

#define DEFINE_GELS_REAL_CBLAS_TO_FORTRAN(OP, TYPE, OP_ID)                    \
static void thunk_##OP##_cblas_to_fortran(                                     \
    char *trans, int *m, int *n, int *nrhs, TYPE *A, int *lda, TYPE *B,       \
    int *ldb, TYPE *work, int *lwork, int *info) {                             \
  (void)work;                                                                   \
  (void)lwork;                                                                  \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, char, int, int, int, TYPE *, int, TYPE *,   \
                     int))g_cblas_fn[OP_ID])(                                  \
      FB_LAYOUT_COL_MAJOR, *trans, *m, *n, *nrhs, A, *lda, B, *ldb);          \
  if (info) *info = rc;                                                         \
}

#define DEFINE_GELS_COMPLEX_CBLAS_TO_FORTRAN(OP, CTYPE, OP_ID)                \
static void thunk_##OP##_cblas_to_fortran(                                     \
    char *trans, int *m, int *n, int *nrhs, CTYPE *A, int *lda, CTYPE *B,     \
    int *ldb, CTYPE *work, int *lwork, int *info) {                            \
  (void)work;                                                                   \
  (void)lwork;                                                                  \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, char, int, int, int, CTYPE *, int, CTYPE *, \
                     int))g_cblas_fn[OP_ID])(                                  \
      FB_LAYOUT_COL_MAJOR, *trans, *m, *n, *nrhs, A, *lda, B, *ldb);          \
  if (info) *info = rc;                                                         \
}

#define DEFINE_GEBRD_REAL_FORTRAN_TO_CBLAS(OP, TYPE, OP_ID, QUERY_FN)          \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const int m, const int n, TYPE *A,               \
    const int lda, TYPE *d, TYPE *e, TYPE *tauq, TYPE *taup) {                 \
  if (!fb_lapack_layout_is_col_major(layout)) return -1;                       \
  int m_ = m, n_ = n, lda_ = lda, lwork_ = -1, info = 0;                       \
  TYPE work_query = (TYPE)0;                                                   \
  TYPE *work = NULL;                                                           \
  if (!g_fortran_fn[OP_ID]) return -1;                                         \
  ((void (*)(int *, int *, TYPE *, int *, TYPE *, TYPE *, TYPE *, TYPE *,      \
             TYPE *, int *, int *))g_fortran_fn[OP_ID])(                       \
      &m_, &n_, A, &lda_, d, e, tauq, taup, &work_query, &lwork_, &info);      \
  if (info != 0) return info;                                                  \
  lwork_ = QUERY_FN(work_query);                                               \
  work = (TYPE *)malloc((size_t)lwork_ * sizeof(TYPE));                        \
  if (!work) return -101;                                                      \
  ((void (*)(int *, int *, TYPE *, int *, TYPE *, TYPE *, TYPE *, TYPE *,      \
             TYPE *, int *, int *))g_fortran_fn[OP_ID])(                       \
      &m_, &n_, A, &lda_, d, e, tauq, taup, work, &lwork_, &info);             \
  free(work);                                                                  \
  return info;                                                                 \
}

#define DEFINE_GEBRD_COMPLEX_FORTRAN_TO_CBLAS(OP, CTYPE, RTYPE, OP_ID, QUERY_FN) \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const int m, const int n, CTYPE *A,             \
    const int lda, RTYPE *d, RTYPE *e, CTYPE *tauq, CTYPE *taup) {             \
  if (!fb_lapack_layout_is_col_major(layout)) return -1;                       \
  int m_ = m, n_ = n, lda_ = lda, lwork_ = -1, info = 0;                       \
  CTYPE work_query = (CTYPE)0;                                                 \
  CTYPE *work = NULL;                                                          \
  if (!g_fortran_fn[OP_ID]) return -1;                                         \
  ((void (*)(int *, int *, CTYPE *, int *, RTYPE *, RTYPE *, CTYPE *, CTYPE *, \
             CTYPE *, int *, int *))g_fortran_fn[OP_ID])(                      \
      &m_, &n_, A, &lda_, d, e, tauq, taup, &work_query, &lwork_, &info);      \
  if (info != 0) return info;                                                  \
  lwork_ = QUERY_FN(work_query);                                               \
  work = (CTYPE *)malloc((size_t)lwork_ * sizeof(CTYPE));                      \
  if (!work) return -101;                                                      \
  ((void (*)(int *, int *, CTYPE *, int *, RTYPE *, RTYPE *, CTYPE *, CTYPE *, \
             CTYPE *, int *, int *))g_fortran_fn[OP_ID])(                      \
      &m_, &n_, A, &lda_, d, e, tauq, taup, work, &lwork_, &info);             \
  free(work);                                                                  \
  return info;                                                                 \
}

#define DEFINE_GEBRD_REAL_CBLAS_TO_FORTRAN(OP, TYPE, OP_ID)                    \
static void thunk_##OP##_cblas_to_fortran(                                     \
    int *m, int *n, TYPE *A, int *lda, TYPE *d, TYPE *e, TYPE *tauq,          \
    TYPE *taup, TYPE *work, int *lwork, int *info) {                           \
  (void)work;                                                                   \
  (void)lwork;                                                                  \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, int, int, TYPE *, int, TYPE *, TYPE *,      \
                     TYPE *, TYPE *))g_cblas_fn[OP_ID])(                       \
      FB_LAYOUT_COL_MAJOR, *m, *n, A, *lda, d, e, tauq, taup);                \
  if (info) *info = rc;                                                         \
}

#define DEFINE_GEBRD_COMPLEX_CBLAS_TO_FORTRAN(OP, CTYPE, RTYPE, OP_ID)        \
static void thunk_##OP##_cblas_to_fortran(                                     \
    int *m, int *n, CTYPE *A, int *lda, RTYPE *d, RTYPE *e, CTYPE *tauq,      \
    CTYPE *taup, CTYPE *work, int *lwork, int *info) {                         \
  (void)work;                                                                   \
  (void)lwork;                                                                  \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, int, int, CTYPE *, int, RTYPE *, RTYPE *,   \
                     CTYPE *, CTYPE *))g_cblas_fn[OP_ID])(                     \
      FB_LAYOUT_COL_MAJOR, *m, *n, A, *lda, d, e, tauq, taup);                \
  if (info) *info = rc;                                                         \
}

#define DEFINE_GEBD2_REAL_FORTRAN_TO_CBLAS(OP, TYPE, OP_ID)                    \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const int m, const int n, TYPE *A,               \
    const int lda, TYPE *d, TYPE *e, TYPE *tauq, TYPE *taup) {                 \
  if (!fb_lapack_layout_is_col_major(layout)) return -1;                       \
  int m_ = m, n_ = n, lda_ = lda, info = 0;                                     \
  int work_size = (m > n) ? m : n;                                             \
  TYPE *work = NULL;                                                           \
  if (work_size < 1) work_size = 1;                                            \
  if (!g_fortran_fn[OP_ID]) return -1;                                         \
  work = (TYPE *)malloc((size_t)work_size * sizeof(TYPE));                     \
  if (!work) return -101;                                                      \
  ((void (*)(int *, int *, TYPE *, int *, TYPE *, TYPE *, TYPE *, TYPE *,      \
             TYPE *, int *))g_fortran_fn[OP_ID])(                              \
      &m_, &n_, A, &lda_, d, e, tauq, taup, work, &info);                      \
  free(work);                                                                  \
  return info;                                                                 \
}

#define DEFINE_GEBD2_COMPLEX_FORTRAN_TO_CBLAS(OP, CTYPE, RTYPE, OP_ID)         \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const int m, const int n, CTYPE *A,              \
    const int lda, RTYPE *d, RTYPE *e, CTYPE *tauq, CTYPE *taup) {             \
  if (!fb_lapack_layout_is_col_major(layout)) return -1;                       \
  int m_ = m, n_ = n, lda_ = lda, info = 0;                                     \
  int work_size = (m > n) ? m : n;                                             \
  CTYPE *work = NULL;                                                          \
  if (work_size < 1) work_size = 1;                                            \
  if (!g_fortran_fn[OP_ID]) return -1;                                         \
  work = (CTYPE *)malloc((size_t)work_size * sizeof(CTYPE));                   \
  if (!work) return -101;                                                      \
  ((void (*)(int *, int *, CTYPE *, int *, RTYPE *, RTYPE *, CTYPE *, CTYPE *, \
             CTYPE *, int *))g_fortran_fn[OP_ID])(                             \
      &m_, &n_, A, &lda_, d, e, tauq, taup, work, &info);                      \
  free(work);                                                                  \
  return info;                                                                 \
}

#define DEFINE_GEBD2_REAL_CBLAS_TO_FORTRAN(OP, TYPE, OP_ID)                    \
static void thunk_##OP##_cblas_to_fortran(                                     \
    int *m, int *n, TYPE *A, int *lda, TYPE *d, TYPE *e, TYPE *tauq,          \
    TYPE *taup, TYPE *work, int *info) {                                       \
  (void)work;                                                                   \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, int, int, TYPE *, int, TYPE *, TYPE *,      \
                     TYPE *, TYPE *))g_cblas_fn[OP_ID])(                       \
      FB_LAYOUT_COL_MAJOR, *m, *n, A, *lda, d, e, tauq, taup);                \
  if (info) *info = rc;                                                         \
}

#define DEFINE_GEBD2_COMPLEX_CBLAS_TO_FORTRAN(OP, CTYPE, RTYPE, OP_ID)        \
static void thunk_##OP##_cblas_to_fortran(                                     \
    int *m, int *n, CTYPE *A, int *lda, RTYPE *d, RTYPE *e, CTYPE *tauq,      \
    CTYPE *taup, CTYPE *work, int *info) {                                     \
  (void)work;                                                                   \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, int, int, CTYPE *, int, RTYPE *, RTYPE *,   \
                     CTYPE *, CTYPE *))g_cblas_fn[OP_ID])(                     \
      FB_LAYOUT_COL_MAJOR, *m, *n, A, *lda, d, e, tauq, taup);                \
  if (info) *info = rc;                                                         \
}

#define DEFINE_LAPACK_LATRD_REAL_THUNKS(OP, TYPE, OP_ID)                       \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, char uplo, const int n, const int nb, TYPE *A,   \
    const int lda, TYPE *e, TYPE *tau, TYPE *W, const int ldw) {               \
  fb_uplo_t uplo_enum = FB_UPPER;                                               \
  char uplo_char = 'U';                                                         \
  int n_ = n, nb_ = nb;                                                         \
  int lda_ = (n > 0) ? n : 1, ldw_ = (n > 0) ? n : 1;                          \
  int panel_len = 0, panel_alloc = 1, w_cols = (nb > 0) ? nb : 1;             \
  size_t a_elems = 0, w_elems = 0;                                              \
  TYPE *a_col = NULL;                                                           \
  TYPE *e_copy = NULL;                                                          \
  TYPE *tau_copy = NULL;                                                        \
  TYPE *w_col = NULL;                                                           \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (fb_lapack_char_to_uplo(uplo, &uplo_enum) != 0) return -1;                \
  if (n < 0 || nb < 0) return -1;                                               \
  uplo_char = fb_lapack_uplo_from_layout(layout, uplo_enum);                   \
  panel_len = (n > 1 && nb > 0) ? ((nb < (n - 1)) ? nb : (n - 1)) : 0;        \
  panel_alloc = (panel_len > 0) ? panel_len : 1;                               \
  if ((n > 0 && !A) || (panel_len > 0 && (!e || !tau)) ||                      \
      (n > 0 && nb > 0 && !W)) return -1;                                      \
  a_elems = (size_t)lda_ * (size_t)((n > 0) ? n : 1);                          \
  w_elems = (size_t)ldw_ * (size_t)w_cols;                                      \
  a_col = (TYPE *)malloc(a_elems * sizeof(TYPE));                               \
  e_copy = (TYPE *)malloc((size_t)panel_alloc * sizeof(TYPE));                  \
  tau_copy = (TYPE *)malloc((size_t)panel_alloc * sizeof(TYPE));                \
  w_col = (TYPE *)malloc(w_elems * sizeof(TYPE));                               \
  if (!a_col || !e_copy || !tau_copy || !w_col) {                               \
    free(a_col);                                                                \
    free(e_copy);                                                               \
    free(tau_copy);                                                             \
    free(w_col);                                                                \
    return -101;                                                                \
  }                                                                             \
  if (n > 0 &&                                                                  \
      fb_copy_matrix_to_col_major(A, layout, n, n, lda,                        \
                                  a_col, lda_, sizeof(TYPE)) != 0) {          \
    free(a_col);                                                                \
    free(e_copy);                                                               \
    free(tau_copy);                                                             \
    free(w_col);                                                                \
    return -1;                                                                  \
  }                                                                             \
  memset(e_copy, 0, (size_t)panel_alloc * sizeof(TYPE));                        \
  memset(tau_copy, 0, (size_t)panel_alloc * sizeof(TYPE));                      \
  memset(w_col, 0, w_elems * sizeof(TYPE));                                     \
  ((void (*)(char *, int *, int *, TYPE *, int *, TYPE *, TYPE *, TYPE *,     \
             int *))g_fortran_fn[OP_ID])(                                      \
      &uplo_char, &n_, &nb_, a_col, &lda_, e_copy, tau_copy, w_col, &ldw_);    \
  if (n > 0 &&                                                                  \
      fb_copy_matrix_from_col_major(a_col, n, n, lda_, layout,                 \
                                    A, lda, sizeof(TYPE)) != 0) {             \
    free(a_col);                                                                \
    free(e_copy);                                                               \
    free(tau_copy);                                                             \
    free(w_col);                                                                \
    return -1;                                                                  \
  }                                                                             \
  if (n > 0 && nb > 0 &&                                                        \
      fb_copy_matrix_from_col_major(w_col, n, nb, ldw_, layout,                \
                                    W, ldw, sizeof(TYPE)) != 0) {            \
    free(a_col);                                                                \
    free(e_copy);                                                               \
    free(tau_copy);                                                             \
    free(w_col);                                                                \
    return -1;                                                                  \
  }                                                                             \
  if (panel_len > 0) {                                                          \
    memcpy(e, e_copy, (size_t)panel_len * sizeof(TYPE));                        \
    memcpy(tau, tau_copy, (size_t)panel_len * sizeof(TYPE));                    \
  }                                                                             \
  free(a_col);                                                                  \
  free(e_copy);                                                                 \
  free(tau_copy);                                                               \
  free(w_col);                                                                  \
  return 0;                                                                     \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    char *uplo, int *n, int *nb, TYPE *A, int *lda, TYPE *e, TYPE *tau,        \
    TYPE *W, int *ldw) {                                                        \
  fb_uplo_t uplo_enum = FB_UPPER;                                               \
  char uplo_char = 'U';                                                         \
  if (!g_cblas_fn[OP_ID]) return;                                               \
  if (!uplo || fb_lapack_char_to_uplo(*uplo, &uplo_enum) != 0) return;         \
  uplo_char = (uplo_enum == FB_UPPER) ? 'U' : 'L';                             \
  (void)((int (*)(fb_layout_t, char, int, int, TYPE *, int, TYPE *, TYPE *,   \
                  TYPE *, int))g_cblas_fn[OP_ID])(                             \
      FB_LAYOUT_COL_MAJOR, uplo_char, *n, *nb, A, *lda, e, tau, W, *ldw);      \
}

#define DEFINE_LAPACK_LATRD_COMPLEX_THUNKS(OP, CTYPE, RTYPE, OP_ID)            \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, char uplo, const int n, const int nb, CTYPE *A,  \
    const int lda, RTYPE *e, CTYPE *tau, CTYPE *W, const int ldw) {            \
  fb_uplo_t uplo_enum = FB_UPPER;                                               \
  char uplo_char = 'U';                                                         \
  int n_ = n, nb_ = nb;                                                         \
  int lda_ = (n > 0) ? n : 1, ldw_ = (n > 0) ? n : 1;                          \
  int panel_len = 0, panel_alloc = 1, w_cols = (nb > 0) ? nb : 1;             \
  size_t a_elems = 0, w_elems = 0;                                              \
  CTYPE *a_col = NULL;                                                          \
  RTYPE *e_copy = NULL;                                                         \
  CTYPE *tau_copy = NULL;                                                       \
  CTYPE *w_col = NULL;                                                          \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (fb_lapack_char_to_uplo(uplo, &uplo_enum) != 0) return -1;                \
  if (n < 0 || nb < 0) return -1;                                               \
  uplo_char = fb_lapack_uplo_from_layout(layout, uplo_enum);                   \
  panel_len = (n > 1 && nb > 0) ? ((nb < (n - 1)) ? nb : (n - 1)) : 0;        \
  panel_alloc = (panel_len > 0) ? panel_len : 1;                               \
  if ((n > 0 && !A) || (panel_len > 0 && (!e || !tau)) ||                      \
      (n > 0 && nb > 0 && !W)) return -1;                                      \
  a_elems = (size_t)lda_ * (size_t)((n > 0) ? n : 1);                          \
  w_elems = (size_t)ldw_ * (size_t)w_cols;                                      \
  a_col = (CTYPE *)malloc(a_elems * sizeof(CTYPE));                             \
  e_copy = (RTYPE *)malloc((size_t)panel_alloc * sizeof(RTYPE));                \
  tau_copy = (CTYPE *)malloc((size_t)panel_alloc * sizeof(CTYPE));              \
  w_col = (CTYPE *)malloc(w_elems * sizeof(CTYPE));                             \
  if (!a_col || !e_copy || !tau_copy || !w_col) {                               \
    free(a_col);                                                                \
    free(e_copy);                                                               \
    free(tau_copy);                                                             \
    free(w_col);                                                                \
    return -101;                                                                \
  }                                                                             \
  if (n > 0 &&                                                                  \
      fb_copy_matrix_to_col_major(A, layout, n, n, lda,                        \
                                  a_col, lda_, sizeof(CTYPE)) != 0) {         \
    free(a_col);                                                                \
    free(e_copy);                                                               \
    free(tau_copy);                                                             \
    free(w_col);                                                                \
    return -1;                                                                  \
  }                                                                             \
  memset(e_copy, 0, (size_t)panel_alloc * sizeof(RTYPE));                       \
  memset(tau_copy, 0, (size_t)panel_alloc * sizeof(CTYPE));                     \
  memset(w_col, 0, w_elems * sizeof(CTYPE));                                    \
  ((void (*)(char *, int *, int *, CTYPE *, int *, RTYPE *, CTYPE *, CTYPE *, \
             int *))g_fortran_fn[OP_ID])(                                      \
      &uplo_char, &n_, &nb_, a_col, &lda_, e_copy, tau_copy, w_col, &ldw_);    \
  if (n > 0 &&                                                                  \
      fb_copy_matrix_from_col_major(a_col, n, n, lda_, layout,                 \
                                    A, lda, sizeof(CTYPE)) != 0) {            \
    free(a_col);                                                                \
    free(e_copy);                                                               \
    free(tau_copy);                                                             \
    free(w_col);                                                                \
    return -1;                                                                  \
  }                                                                             \
  if (n > 0 && nb > 0 &&                                                        \
      fb_copy_matrix_from_col_major(w_col, n, nb, ldw_, layout,                \
                                    W, ldw, sizeof(CTYPE)) != 0) {           \
    free(a_col);                                                                \
    free(e_copy);                                                               \
    free(tau_copy);                                                             \
    free(w_col);                                                                \
    return -1;                                                                  \
  }                                                                             \
  if (panel_len > 0) {                                                          \
    memcpy(e, e_copy, (size_t)panel_len * sizeof(RTYPE));                       \
    memcpy(tau, tau_copy, (size_t)panel_len * sizeof(CTYPE));                   \
  }                                                                             \
  free(a_col);                                                                  \
  free(e_copy);                                                                 \
  free(tau_copy);                                                               \
  free(w_col);                                                                  \
  return 0;                                                                     \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    char *uplo, int *n, int *nb, CTYPE *A, int *lda, RTYPE *e, CTYPE *tau,     \
    CTYPE *W, int *ldw) {                                                       \
  fb_uplo_t uplo_enum = FB_UPPER;                                               \
  char uplo_char = 'U';                                                         \
  if (!g_cblas_fn[OP_ID]) return;                                               \
  if (!uplo || fb_lapack_char_to_uplo(*uplo, &uplo_enum) != 0) return;         \
  uplo_char = (uplo_enum == FB_UPPER) ? 'U' : 'L';                             \
  (void)((int (*)(fb_layout_t, char, int, int, CTYPE *, int, RTYPE *,         \
                  CTYPE *, CTYPE *, int))g_cblas_fn[OP_ID])(                   \
      FB_LAYOUT_COL_MAJOR, uplo_char, *n, *nb, A, *lda, e, tau, W, *ldw);      \
}

#define DEFINE_LAPACK_LATDF_THUNKS(OP, ZTYPE, RTYPE, OP_ID)                    \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const int ijob, const int n, ZTYPE *Z,           \
    const int ldz, ZTYPE *rhs, RTYPE *rdsum, RTYPE *rdscal,                    \
    int *ipiv, int *jpiv) {                                                    \
  int ijob_ = ijob, n_ = n, ldz_ = ldz;                                        \
  ZTYPE *z_arg = Z;                                                            \
  ZTYPE *z_col = NULL;                                                         \
  int z_col_ld = (n > 0) ? n : 1;                                              \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (n < 0) return -1;                                                        \
  if ((n > 0 && (!Z || !rhs)) || !rdsum || !rdscal || !ipiv || !jpiv) {       \
    return -1;                                                                 \
  }                                                                            \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                         \
    size_t z_col_elems = (size_t)z_col_ld * (size_t)((n > 0) ? n : 1);        \
    if (n > 0 && ldz < n) return -1;                                           \
    z_col = (ZTYPE *)malloc(z_col_elems * sizeof(ZTYPE));                      \
    if (!z_col) return -101;                                                   \
    if (n > 0 &&                                                                \
        fb_copy_matrix_to_col_major(Z, layout, n, n, ldz,                      \
                                    z_col, z_col_ld, sizeof(ZTYPE)) != 0) {   \
      free(z_col);                                                             \
      return -1;                                                               \
    }                                                                          \
    z_arg = z_col;                                                             \
    ldz_ = z_col_ld;                                                           \
  } else if (!fb_lapack_layout_is_col_major(layout)) {                         \
    return -1;                                                                 \
  } else if (n > 0 && ldz < n) {                                               \
    return -1;                                                                 \
  }                                                                            \
  ((void (*)(int *, int *, ZTYPE *, int *, ZTYPE *, RTYPE *, RTYPE *,         \
             int *, int *))g_fortran_fn[OP_ID])(                               \
      &ijob_, &n_, z_arg, &ldz_, rhs, rdsum, rdscal, ipiv, jpiv);              \
  if (z_col) {                                                                 \
    if (n > 0 &&                                                                \
        fb_copy_matrix_from_col_major(z_col, n, n, ldz_, layout,               \
                                      Z, ldz, sizeof(ZTYPE)) != 0) {          \
      free(z_col);                                                             \
      return -1;                                                               \
    }                                                                          \
    free(z_col);                                                               \
  }                                                                            \
  return 0;                                                                    \
}                                                                              \
static void thunk_##OP##_cblas_to_fortran(                                     \
    int *ijob, int *n, ZTYPE *Z, int *ldz, ZTYPE *rhs, RTYPE *rdsum,           \
    RTYPE *rdscal, int *ipiv, int *jpiv) {                                     \
  if (!g_cblas_fn[OP_ID]) return;                                              \
  (void)((int (*)(fb_layout_t, int, int, ZTYPE *, int, ZTYPE *, RTYPE *,      \
                  RTYPE *, int *, int *))g_cblas_fn[OP_ID])(                   \
      FB_LAYOUT_COL_MAJOR, *ijob, *n, Z, *ldz, rhs, rdsum, rdscal, ipiv, jpiv);\
}

#define DEFINE_LAPACK_LATRS_REAL_THUNKS(OP, TYPE, OP_ID)                       \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, char uplo, char trans, char diag, char normin,   \
    const int n, TYPE *A, const int lda, TYPE *x, TYPE *scale, TYPE *cnorm) {  \
  int n_ = n, lda_ = lda, info = 0;                                            \
  TYPE *a_arg = A;                                                             \
  TYPE *a_col = NULL;                                                          \
  int a_col_ld = (n > 0) ? n : 1;                                              \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (n < 0 || !x || !scale || !cnorm) return -1;                              \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                         \
    size_t a_col_elems = (size_t)a_col_ld * (size_t)((n > 0) ? n : 1);        \
    if (n > 0 && (!A || lda < n)) return -1;                                   \
    a_col = (TYPE *)malloc(a_col_elems * sizeof(TYPE));                        \
    if (!a_col) return -101;                                                   \
    if (n > 0 &&                                                                \
        fb_copy_matrix_to_col_major(A, layout, n, n, lda,                      \
                                    a_col, a_col_ld, sizeof(TYPE)) != 0) {    \
      free(a_col);                                                             \
      return -1;                                                               \
    }                                                                          \
    a_arg = a_col;                                                             \
    lda_ = a_col_ld;                                                           \
  } else if (!fb_lapack_layout_is_col_major(layout)) {                         \
    return -1;                                                                 \
  } else if (n > 0 && (!A || lda < n)) {                                       \
    return -1;                                                                 \
  }                                                                            \
  ((void (*)(char *, char *, char *, char *, int *, TYPE *, int *, TYPE *,    \
             TYPE *, TYPE *, int *))g_fortran_fn[OP_ID])(                      \
      &uplo, &trans, &diag, &normin, &n_, a_arg, &lda_, x, scale, cnorm,       \
      &info);                                                                  \
  free(a_col);                                                                 \
  return info;                                                                 \
}                                                                              \
static void thunk_##OP##_cblas_to_fortran(                                     \
    char *uplo, char *trans, char *diag, char *normin, int *n, TYPE *A,        \
    int *lda, TYPE *x, TYPE *scale, TYPE *cnorm, int *info) {                  \
  if (!g_cblas_fn[OP_ID]) {                                                    \
    if (info) *info = -1;                                                      \
    return;                                                                    \
  }                                                                            \
  if (!uplo || !trans || !diag || !normin || !n || !lda || !x || !scale ||    \
      !cnorm || !info) {                                                       \
    if (info) *info = -1;                                                      \
    return;                                                                    \
  }                                                                            \
  *info = ((int (*)(fb_layout_t, char, char, char, char, int, TYPE *, int,   \
                    TYPE *, TYPE *, TYPE *))g_cblas_fn[OP_ID])(                \
      FB_LAYOUT_COL_MAJOR, *uplo, *trans, *diag, *normin, *n, A, *lda, x,      \
      scale, cnorm);                                                           \
}

#define DEFINE_LAPACK_LATRS_COMPLEX_THUNKS(OP, CTYPE, RTYPE, OP_ID)           \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, char uplo, char trans, char diag, char normin,   \
    const int n, CTYPE *A, const int lda, CTYPE *x, RTYPE *scale,              \
    RTYPE *cnorm) {                                                            \
  int n_ = n, lda_ = lda, info = 0;                                            \
  CTYPE *a_arg = A;                                                            \
  CTYPE *a_col = NULL;                                                         \
  int a_col_ld = (n > 0) ? n : 1;                                              \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (n < 0 || !x || !scale || !cnorm) return -1;                              \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                         \
    size_t a_col_elems = (size_t)a_col_ld * (size_t)((n > 0) ? n : 1);        \
    if (n > 0 && (!A || lda < n)) return -1;                                   \
    a_col = (CTYPE *)malloc(a_col_elems * sizeof(CTYPE));                      \
    if (!a_col) return -101;                                                   \
    if (n > 0 &&                                                                \
        fb_copy_matrix_to_col_major(A, layout, n, n, lda,                      \
                                    a_col, a_col_ld, sizeof(CTYPE)) != 0) {   \
      free(a_col);                                                             \
      return -1;                                                               \
    }                                                                          \
    a_arg = a_col;                                                             \
    lda_ = a_col_ld;                                                           \
  } else if (!fb_lapack_layout_is_col_major(layout)) {                         \
    return -1;                                                                 \
  } else if (n > 0 && (!A || lda < n)) {                                       \
    return -1;                                                                 \
  }                                                                            \
  ((void (*)(char *, char *, char *, char *, int *, CTYPE *, int *, CTYPE *,  \
             RTYPE *, RTYPE *, int *))g_fortran_fn[OP_ID])(                    \
      &uplo, &trans, &diag, &normin, &n_, a_arg, &lda_, x, scale, cnorm,       \
      &info);                                                                  \
  free(a_col);                                                                 \
  return info;                                                                 \
}                                                                              \
static void thunk_##OP##_cblas_to_fortran(                                     \
    char *uplo, char *trans, char *diag, char *normin, int *n, CTYPE *A,       \
    int *lda, CTYPE *x, RTYPE *scale, RTYPE *cnorm, int *info) {               \
  if (!g_cblas_fn[OP_ID]) {                                                    \
    if (info) *info = -1;                                                      \
    return;                                                                    \
  }                                                                            \
  if (!uplo || !trans || !diag || !normin || !n || !lda || !x || !scale ||    \
      !cnorm || !info) {                                                       \
    if (info) *info = -1;                                                      \
    return;                                                                    \
  }                                                                            \
  *info = ((int (*)(fb_layout_t, char, char, char, char, int, CTYPE *, int,  \
                    CTYPE *, RTYPE *, RTYPE *))g_cblas_fn[OP_ID])(             \
      FB_LAYOUT_COL_MAJOR, *uplo, *trans, *diag, *normin, *n, A, *lda, x,      \
      scale, cnorm);                                                           \
}

#define DEFINE_LAPACK_LATPS_REAL_THUNKS(OP, TYPE, OP_ID)                       \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, char uplo, char trans, char diag, char normin,   \
    const int n, TYPE *ap, TYPE *x, TYPE *scale, TYPE *cnorm) {                \
  int n_ = n, info = 0;                                                        \
  TYPE *ap_arg = ap;                                                           \
  TYPE *ap_col = NULL;                                                         \
  size_t packed_len = (size_t)((n > 0) ? n : 1) *                             \
                      (size_t)(((n + 1) > 0) ? (n + 1) : 2) / 2U;             \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (n < 0 || !x || !scale || !cnorm) return -1;                              \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                         \
    if (n > 0 && !ap) return -1;                                               \
    ap_col = (TYPE *)malloc(packed_len * sizeof(TYPE));                        \
    if (!ap_col) return -101;                                                  \
    if (fb_copy_triangular_packed_to_col_major(ap, layout, uplo, diag, n,      \
                                               ap_col, sizeof(TYPE)) != 0) {   \
      free(ap_col);                                                            \
      return -1;                                                               \
    }                                                                          \
    ap_arg = ap_col;                                                           \
  } else if (!fb_lapack_layout_is_col_major(layout)) {                         \
    return -1;                                                                 \
  } else if (n > 0 && !ap) {                                                   \
    return -1;                                                                 \
  }                                                                            \
  ((void (*)(char *, char *, char *, char *, int *, TYPE *, TYPE *, TYPE *,   \
             TYPE *, int *))g_fortran_fn[OP_ID])(                              \
      &uplo, &trans, &diag, &normin, &n_, ap_arg, x, scale, cnorm, &info);     \
  free(ap_col);                                                                \
  return info;                                                                 \
}                                                                              \
static void thunk_##OP##_cblas_to_fortran(                                     \
    char *uplo, char *trans, char *diag, char *normin, int *n, TYPE *ap,       \
    TYPE *x, TYPE *scale, TYPE *cnorm, int *info) {                            \
  if (!g_cblas_fn[OP_ID]) {                                                    \
    if (info) *info = -1;                                                      \
    return;                                                                    \
  }                                                                            \
  if (!uplo || !trans || !diag || !normin || !n || !x || !scale || !cnorm ||  \
      !info) {                                                                 \
    if (info) *info = -1;                                                      \
    return;                                                                    \
  }                                                                            \
  *info = ((int (*)(fb_layout_t, char, char, char, char, int, TYPE *, TYPE *, \
                    TYPE *, TYPE *))g_cblas_fn[OP_ID])(                        \
      FB_LAYOUT_COL_MAJOR, *uplo, *trans, *diag, *normin, *n, ap, x, scale,    \
      cnorm);                                                                  \
}

#define DEFINE_LAPACK_LATPS_COMPLEX_THUNKS(OP, CTYPE, RTYPE, OP_ID)           \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, char uplo, char trans, char diag, char normin,   \
    const int n, CTYPE *ap, CTYPE *x, RTYPE *scale, RTYPE *cnorm) {            \
  int n_ = n, info = 0;                                                        \
  CTYPE *ap_arg = ap;                                                          \
  CTYPE *ap_col = NULL;                                                        \
  size_t packed_len = (size_t)((n > 0) ? n : 1) *                             \
                      (size_t)(((n + 1) > 0) ? (n + 1) : 2) / 2U;             \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (n < 0 || !x || !scale || !cnorm) return -1;                              \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                         \
    if (n > 0 && !ap) return -1;                                               \
    ap_col = (CTYPE *)malloc(packed_len * sizeof(CTYPE));                      \
    if (!ap_col) return -101;                                                  \
    if (fb_copy_triangular_packed_to_col_major(ap, layout, uplo, diag, n,      \
                                               ap_col, sizeof(CTYPE)) != 0) {  \
      free(ap_col);                                                            \
      return -1;                                                               \
    }                                                                          \
    ap_arg = ap_col;                                                           \
  } else if (!fb_lapack_layout_is_col_major(layout)) {                         \
    return -1;                                                                 \
  } else if (n > 0 && !ap) {                                                   \
    return -1;                                                                 \
  }                                                                            \
  ((void (*)(char *, char *, char *, char *, int *, CTYPE *, CTYPE *,         \
             RTYPE *, RTYPE *, int *))g_fortran_fn[OP_ID])(                    \
      &uplo, &trans, &diag, &normin, &n_, ap_arg, x, scale, cnorm, &info);     \
  free(ap_col);                                                                \
  return info;                                                                 \
}                                                                              \
static void thunk_##OP##_cblas_to_fortran(                                     \
    char *uplo, char *trans, char *diag, char *normin, int *n, CTYPE *ap,      \
    CTYPE *x, RTYPE *scale, RTYPE *cnorm, int *info) {                         \
  if (!g_cblas_fn[OP_ID]) {                                                    \
    if (info) *info = -1;                                                      \
    return;                                                                    \
  }                                                                            \
  if (!uplo || !trans || !diag || !normin || !n || !x || !scale || !cnorm ||  \
      !info) {                                                                 \
    if (info) *info = -1;                                                      \
    return;                                                                    \
  }                                                                            \
  *info = ((int (*)(fb_layout_t, char, char, char, char, int, CTYPE *,       \
                    CTYPE *, RTYPE *, RTYPE *))g_cblas_fn[OP_ID])(             \
      FB_LAYOUT_COL_MAJOR, *uplo, *trans, *diag, *normin, *n, ap, x, scale,    \
      cnorm);                                                                  \
}

#define DEFINE_LAPACK_LASSQ_REAL_THUNKS(OP, TYPE, OP_ID)                       \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const int n, TYPE *x, const int incx, TYPE *scale, TYPE *sumsq) {          \
  int n_ = n;                                                                  \
  int incx_ = incx;                                                            \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if ((n > 0 && !x) || !scale || !sumsq) return -1;                            \
  ((void (*)(int *, TYPE *, int *, TYPE *, TYPE *))g_fortran_fn[OP_ID])(       \
      &n_, x, &incx_, scale, sumsq);                                           \
  return 0;                                                                    \
}                                                                              \
static void thunk_##OP##_cblas_to_fortran(                                     \
    int *n, TYPE *x, int *incx, TYPE *scale, TYPE *sumsq) {                    \
  if (!g_cblas_fn[OP_ID]) return;                                              \
  if (!n || !incx || (*n > 0 && !x) || !scale || !sumsq) return;               \
  (void)((int (*)(int, TYPE *, int, TYPE *, TYPE *))g_cblas_fn[OP_ID])(        \
      *n, x, *incx, scale, sumsq);                                             \
}

#define DEFINE_LAPACK_LASSQ_COMPLEX_THUNKS(OP, CTYPE, RTYPE, OP_ID)            \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const int n, CTYPE *x, const int incx, RTYPE *scale, RTYPE *sumsq) {       \
  int n_ = n;                                                                  \
  int incx_ = incx;                                                            \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if ((n > 0 && !x) || !scale || !sumsq) return -1;                            \
  ((void (*)(int *, CTYPE *, int *, RTYPE *, RTYPE *))g_fortran_fn[OP_ID])(    \
      &n_, x, &incx_, scale, sumsq);                                           \
  return 0;                                                                    \
}                                                                              \
static void thunk_##OP##_cblas_to_fortran(                                     \
    int *n, CTYPE *x, int *incx, RTYPE *scale, RTYPE *sumsq) {                 \
  if (!g_cblas_fn[OP_ID]) return;                                              \
  if (!n || !incx || (*n > 0 && !x) || !scale || !sumsq) return;               \
  (void)((int (*)(int, CTYPE *, int, RTYPE *, RTYPE *))g_cblas_fn[OP_ID])(     \
      *n, x, *incx, scale, sumsq);                                             \
}

#define DEFINE_LAPACK_LAEV2_THUNKS(OP, TYPE, OP_ID)                            \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const TYPE a, const TYPE b, const TYPE c, TYPE *rt1, TYPE *rt2,            \
    TYPE *cs1, TYPE *sn1) {                                                    \
  TYPE a_ = a;                                                                 \
  TYPE b_ = b;                                                                 \
  TYPE c_ = c;                                                                 \
  if (!g_fortran_fn[OP_ID]) return -1;                                         \
  if (!rt1 || !rt2 || !cs1 || !sn1) return -1;                                 \
  ((void (*)(TYPE *, TYPE *, TYPE *, TYPE *, TYPE *, TYPE *, TYPE *))          \
       g_fortran_fn[OP_ID])(&a_, &b_, &c_, rt1, rt2, cs1, sn1);                \
  return 0;                                                                    \
}                                                                              \
static void thunk_##OP##_cblas_to_fortran(                                     \
    TYPE *a, TYPE *b, TYPE *c, TYPE *rt1, TYPE *rt2, TYPE *cs1, TYPE *sn1) {  \
  if (!g_cblas_fn[OP_ID]) return;                                              \
  if (!a || !b || !c || !rt1 || !rt2 || !cs1 || !sn1) return;                  \
  (void)((int (*)(TYPE, TYPE, TYPE, TYPE *, TYPE *, TYPE *, TYPE *))           \
             g_cblas_fn[OP_ID])(*a, *b, *c, rt1, rt2, cs1, sn1);               \
}

#define DEFINE_LAPACK_LASV2_THUNKS(OP, TYPE, OP_ID)                            \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const TYPE f, const TYPE g, const TYPE h, TYPE *ssmin, TYPE *ssmax,        \
    TYPE *snr, TYPE *csr, TYPE *sn, TYPE *cs) {                                \
  TYPE f_ = f;                                                                  \
  TYPE g_ = g;                                                                  \
  TYPE h_ = h;                                                                  \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (!ssmin || !ssmax || !snr || !csr || !sn || !cs) return -1;               \
  ((void (*)(TYPE *, TYPE *, TYPE *, TYPE *, TYPE *, TYPE *, TYPE *, TYPE *,    \
             TYPE *))g_fortran_fn[OP_ID])(                                      \
      &f_, &g_, &h_, ssmin, ssmax, snr, csr, sn, cs);                          \
  return 0;                                                                     \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    TYPE *f, TYPE *g, TYPE *h, TYPE *ssmin, TYPE *ssmax, TYPE *snr,            \
    TYPE *csr, TYPE *sn, TYPE *cs) {                                            \
  if (!g_cblas_fn[OP_ID]) return;                                               \
  if (!f || !g || !h || !ssmin || !ssmax || !snr || !csr || !sn || !cs) return;\
  (void)((int (*)(TYPE, TYPE, TYPE, TYPE *, TYPE *, TYPE *, TYPE *, TYPE *,     \
                  TYPE *))g_cblas_fn[OP_ID])(                                   \
      *f, *g, *h, ssmin, ssmax, snr, csr, sn, cs);                              \
}

#define DEFINE_LAPACK_LA_GBAMV_REAL_THUNKS(OP, TYPE, OP_ID)                    \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const int trans, const int m, const int n, const int kl, const int ku,     \
    const TYPE alpha, TYPE *ab, const int ldab, TYPE *x, const int incx,       \
    const TYPE beta, TYPE *y, const int incy) {                                \
  int trans_ = trans;                                                           \
  int m_ = m;                                                                   \
  int n_ = n;                                                                   \
  int kl_ = kl;                                                                 \
  int ku_ = ku;                                                                 \
  TYPE alpha_ = alpha;                                                          \
  int ldab_ = ldab;                                                             \
  int incx_ = incx;                                                             \
  TYPE beta_ = beta;                                                            \
  int incy_ = incy;                                                             \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (!y) return -1;                                                            \
  if ((m > 0 || n > 0) && (!ab || !x)) return -1;                              \
  ((void (*)(int *, int *, int *, int *, int *, TYPE *, TYPE *, int *, TYPE *, \
             int *, TYPE *, TYPE *, int *))g_fortran_fn[OP_ID])(               \
      &trans_, &m_, &n_, &kl_, &ku_, &alpha_, ab, &ldab_, x, &incx_, &beta_,  \
      y, &incy_);                                                               \
  return 0;                                                                     \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    int *trans, int *m, int *n, int *kl, int *ku, TYPE *alpha, TYPE *ab,       \
    int *ldab, TYPE *x, int *incx, TYPE *beta, TYPE *y, int *incy) {           \
  if (!g_cblas_fn[OP_ID]) return;                                               \
  if (!trans || !m || !n || !kl || !ku || !alpha || !ldab || !incx || !beta || \
      !y || !incy) return;                                                      \
  if ((*m > 0 || *n > 0) && (!ab || !x)) return;                               \
  (void)((int (*)(int, int, int, int, int, TYPE, TYPE *, int, TYPE *, int,     \
                  TYPE, TYPE *, int))g_cblas_fn[OP_ID])(                        \
      *trans, *m, *n, *kl, *ku, *alpha, ab, *ldab, x, *incx, *beta, y, *incy); \
}

#define DEFINE_LAPACK_LA_GBAMV_COMPLEX_THUNKS(OP, CTYPE, RTYPE, OP_ID)         \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const int trans, const int m, const int n, const int kl, const int ku,     \
    const RTYPE alpha, CTYPE *ab, const int ldab, CTYPE *x, const int incx,    \
    const RTYPE beta, RTYPE *y, const int incy) {                               \
  int trans_ = trans;                                                           \
  int m_ = m;                                                                   \
  int n_ = n;                                                                   \
  int kl_ = kl;                                                                 \
  int ku_ = ku;                                                                 \
  RTYPE alpha_ = alpha;                                                         \
  int ldab_ = ldab;                                                             \
  int incx_ = incx;                                                             \
  RTYPE beta_ = beta;                                                           \
  int incy_ = incy;                                                             \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (!y) return -1;                                                            \
  if ((m > 0 || n > 0) && (!ab || !x)) return -1;                              \
  ((void (*)(int *, int *, int *, int *, int *, RTYPE *, CTYPE *, int *,       \
             CTYPE *, int *, RTYPE *, RTYPE *, int *))g_fortran_fn[OP_ID])(    \
      &trans_, &m_, &n_, &kl_, &ku_, &alpha_, ab, &ldab_, x, &incx_, &beta_,  \
      y, &incy_);                                                               \
  return 0;                                                                     \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    int *trans, int *m, int *n, int *kl, int *ku, RTYPE *alpha, CTYPE *ab,     \
    int *ldab, CTYPE *x, int *incx, RTYPE *beta, RTYPE *y, int *incy) {        \
  if (!g_cblas_fn[OP_ID]) return;                                               \
  if (!trans || !m || !n || !kl || !ku || !alpha || !ldab || !incx || !beta || \
      !y || !incy) return;                                                      \
  if ((*m > 0 || *n > 0) && (!ab || !x)) return;                               \
  (void)((int (*)(int, int, int, int, int, RTYPE, CTYPE *, int, CTYPE *, int,  \
                  RTYPE, RTYPE *, int))g_cblas_fn[OP_ID])(                      \
      *trans, *m, *n, *kl, *ku, *alpha, ab, *ldab, x, *incx, *beta, y, *incy); \
}

#define DEFINE_LAPACK_LAEIN_REAL_THUNKS(OP, TYPE, OP_ID)                       \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const int rightv, const int noinit, const int n, TYPE *h, const int ldh,   \
    const TYPE wr, const TYPE wi, TYPE *vr, TYPE *vi, TYPE *b, const int ldb,  \
    TYPE *work, const TYPE eps3, const TYPE smlnum, const TYPE bignum) {       \
  int rightv_ = rightv;                                                        \
  int noinit_ = noinit;                                                        \
  int n_ = n;                                                                  \
  int ldh_ = ldh;                                                              \
  int ldb_ = ldb;                                                              \
  int info_ = 0;                                                               \
  TYPE wr_ = wr;                                                               \
  TYPE wi_ = wi;                                                               \
  TYPE eps3_ = eps3;                                                           \
  TYPE smlnum_ = smlnum;                                                       \
  TYPE bignum_ = bignum;                                                       \
  if (!g_fortran_fn[OP_ID]) return -1;                                         \
  if ((n > 0 && (!h || !vr || !vi || !b || !work))) return -1;                 \
  ((void (*)(int *, int *, int *, TYPE *, int *, TYPE *, TYPE *, TYPE *,       \
             TYPE *, TYPE *, int *, TYPE *, TYPE *, TYPE *, TYPE *, int *))    \
       g_fortran_fn[OP_ID])(                                                   \
      &rightv_, &noinit_, &n_, h, &ldh_, &wr_, &wi_, vr, vi, b, &ldb_, work,   \
      &eps3_, &smlnum_, &bignum_, &info_);                                     \
  return info_;                                                                \
}                                                                              \
static void thunk_##OP##_cblas_to_fortran(                                     \
    int *rightv, int *noinit, int *n, TYPE *h, int *ldh, TYPE *wr, TYPE *wi,   \
    TYPE *vr, TYPE *vi, TYPE *b, int *ldb, TYPE *work, TYPE *eps3,             \
    TYPE *smlnum, TYPE *bignum, int *info) {                                   \
  int info_ = -1;                                                              \
  if (!rightv || !noinit || !n || !ldh || !wr || !wi || !ldb || !eps3 ||      \
      !smlnum || !bignum || !info) {                                           \
    return;                                                                    \
  }                                                                            \
  if (!g_cblas_fn[OP_ID]) {                                                    \
    *info = -1;                                                                \
    return;                                                                    \
  }                                                                            \
  if ((*n > 0 && (!h || !vr || !vi || !b || !work))) {                         \
    *info = -1;                                                                \
    return;                                                                    \
  }                                                                            \
  info_ = ((int (*)(int, int, int, TYPE *, int, TYPE, TYPE, TYPE *, TYPE *,    \
                    TYPE *, int, TYPE *, TYPE, TYPE, TYPE))g_cblas_fn[OP_ID])( \
      *rightv, *noinit, *n, h, *ldh, *wr, *wi, vr, vi, b, *ldb, work, *eps3,   \
      *smlnum, *bignum);                                                       \
  *info = info_;                                                               \
}

#define DEFINE_LAPACK_LAEIN_COMPLEX_THUNKS(OP, CTYPE, RTYPE, OP_ID)            \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const int rightv, const int noinit, const int n, CTYPE *h, const int ldh,  \
    const CTYPE w, CTYPE *v, CTYPE *b, const int ldb, RTYPE *rwork,            \
    const RTYPE eps3, const RTYPE smlnum) {                                    \
  int rightv_ = rightv;                                                        \
  int noinit_ = noinit;                                                        \
  int n_ = n;                                                                  \
  int ldh_ = ldh;                                                              \
  int ldb_ = ldb;                                                              \
  int info_ = 0;                                                               \
  CTYPE w_ = w;                                                                \
  RTYPE eps3_ = eps3;                                                          \
  RTYPE smlnum_ = smlnum;                                                      \
  if (!g_fortran_fn[OP_ID]) return -1;                                         \
  if ((n > 0 && (!h || !v || !b || !rwork))) return -1;                        \
  ((void (*)(int *, int *, int *, CTYPE *, int *, CTYPE *, CTYPE *, CTYPE *,   \
             int *, RTYPE *, RTYPE *, RTYPE *, int *))g_fortran_fn[OP_ID])(    \
      &rightv_, &noinit_, &n_, h, &ldh_, &w_, v, b, &ldb_, rwork, &eps3_,      \
      &smlnum_, &info_);                                                       \
  return info_;                                                                \
}                                                                              \
static void thunk_##OP##_cblas_to_fortran(                                     \
    int *rightv, int *noinit, int *n, CTYPE *h, int *ldh, CTYPE *w, CTYPE *v,  \
    CTYPE *b, int *ldb, RTYPE *rwork, RTYPE *eps3, RTYPE *smlnum,              \
    int *info) {                                                               \
  int info_ = -1;                                                              \
  if (!rightv || !noinit || !n || !ldh || !w || !ldb || !rwork || !eps3 ||    \
      !smlnum || !info) {                                                      \
    return;                                                                    \
  }                                                                            \
  if (!g_cblas_fn[OP_ID]) {                                                    \
    *info = -1;                                                                \
    return;                                                                    \
  }                                                                            \
  if ((*n > 0 && (!h || !v || !b || !rwork))) {                                \
    *info = -1;                                                                \
    return;                                                                    \
  }                                                                            \
  info_ = ((int (*)(int, int, int, CTYPE *, int, CTYPE, CTYPE *, CTYPE *,      \
                    int, RTYPE *, RTYPE, RTYPE))g_cblas_fn[OP_ID])(            \
      *rightv, *noinit, *n, h, *ldh, *w, v, b, *ldb, rwork, *eps3, *smlnum);   \
  *info = info_;                                                               \
}

#define DEFINE_LAPACK_LAHQR_REAL_THUNKS(OP, TYPE, OP_ID)                       \
static int thunk_##OP##_fortran_to_cblas(                                      \
    char job, char compz, const int n, const int ilo, const int ihi, TYPE *h,  \
    const int ldh, TYPE *wr, TYPE *wi, const int iloz, const int ihiz,         \
    TYPE *z, const int ldz) {                                                  \
  char job_ = job;                                                             \
  char compz_ = compz;                                                         \
  int n_ = n;                                                                  \
  int ilo_ = ilo;                                                              \
  int ihi_ = ihi;                                                              \
  int ldh_ = ldh;                                                              \
  int iloz_ = iloz;                                                            \
  int ihiz_ = ihiz;                                                            \
  int ldz_ = ldz;                                                              \
  int info_ = 0;                                                               \
  if (!g_fortran_fn[OP_ID]) return -1;                                         \
  if ((n > 0 && (!h || !wr || !wi || !z))) return -1;                          \
  ((void (*)(char *, char *, int *, int *, int *, TYPE *, int *, TYPE *,       \
             TYPE *, int *, int *, TYPE *, int *, int *))g_fortran_fn[OP_ID])( \
      &job_, &compz_, &n_, &ilo_, &ihi_, h, &ldh_, wr, wi, &iloz_, &ihiz_, z,  \
      &ldz_, &info_);                                                          \
  return info_;                                                                \
}                                                                              \
static void thunk_##OP##_cblas_to_fortran(                                     \
    char *job, char *compz, int *n, int *ilo, int *ihi, TYPE *h, int *ldh,     \
    TYPE *wr, TYPE *wi, int *iloz, int *ihiz, TYPE *z, int *ldz,               \
    int *info) {                                                               \
  int info_ = -1;                                                              \
  if (!job || !compz || !n || !ilo || !ihi || !ldh || !iloz || !ihiz || !ldz ||\
      !info) {                                                                 \
    return;                                                                    \
  }                                                                            \
  if (!g_cblas_fn[OP_ID]) {                                                    \
    *info = -1;                                                                \
    return;                                                                    \
  }                                                                            \
  if ((*n > 0 && (!h || !wr || !wi || !z))) {                                  \
    *info = -1;                                                                \
    return;                                                                    \
  }                                                                            \
  info_ = ((int (*)(char, char, int, int, int, TYPE *, int, TYPE *, TYPE *,    \
                    int, int, TYPE *, int))g_cblas_fn[OP_ID])(                 \
      *job, *compz, *n, *ilo, *ihi, h, *ldh, wr, wi, *iloz, *ihiz, z, *ldz);   \
  *info = info_;                                                               \
}

#define DEFINE_LAPACK_LAHQR_COMPLEX_THUNKS(OP, CTYPE, OP_ID)                   \
static int thunk_##OP##_fortran_to_cblas(                                      \
    char job, char compz, const int n, const int ilo, const int ihi, CTYPE *h, \
    const int ldh, CTYPE *w, const int iloz, const int ihiz, CTYPE *z,         \
    const int ldz) {                                                           \
  char job_ = job;                                                             \
  char compz_ = compz;                                                         \
  int n_ = n;                                                                  \
  int ilo_ = ilo;                                                              \
  int ihi_ = ihi;                                                              \
  int ldh_ = ldh;                                                              \
  int iloz_ = iloz;                                                            \
  int ihiz_ = ihiz;                                                            \
  int ldz_ = ldz;                                                              \
  int info_ = 0;                                                               \
  if (!g_fortran_fn[OP_ID]) return -1;                                         \
  if ((n > 0 && (!h || !w || !z))) return -1;                                  \
  ((void (*)(char *, char *, int *, int *, int *, CTYPE *, int *, CTYPE *,     \
             int *, int *, CTYPE *, int *, int *))g_fortran_fn[OP_ID])(        \
      &job_, &compz_, &n_, &ilo_, &ihi_, h, &ldh_, w, &iloz_, &ihiz_, z,       \
      &ldz_, &info_);                                                          \
  return info_;                                                                \
}                                                                              \
static void thunk_##OP##_cblas_to_fortran(                                     \
    char *job, char *compz, int *n, int *ilo, int *ihi, CTYPE *h, int *ldh,    \
    CTYPE *w, int *iloz, int *ihiz, CTYPE *z, int *ldz, int *info) {           \
  int info_ = -1;                                                              \
  if (!job || !compz || !n || !ilo || !ihi || !ldh || !iloz || !ihiz || !ldz ||\
      !info) {                                                                 \
    return;                                                                    \
  }                                                                            \
  if (!g_cblas_fn[OP_ID]) {                                                    \
    *info = -1;                                                                \
    return;                                                                    \
  }                                                                            \
  if ((*n > 0 && (!h || !w || !z))) {                                          \
    *info = -1;                                                                \
    return;                                                                    \
  }                                                                            \
  info_ = ((int (*)(char, char, int, int, int, CTYPE *, int, CTYPE *, int,     \
                    int, CTYPE *, int))g_cblas_fn[OP_ID])(                     \
      *job, *compz, *n, *ilo, *ihi, h, *ldh, w, *iloz, *ihiz, z, *ldz);        \
  *info = info_;                                                               \
}

#define DEFINE_LAPACK_LAHRD_REAL_THUNKS(OP, TYPE, OP_ID)                       \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const int n, const int k, const int nb, TYPE *a, const int lda, TYPE *tau, \
    TYPE *t, const int ldt, TYPE *y, const int ldy) {                          \
  int n_ = n;                                                                  \
  int k_ = k;                                                                  \
  int nb_ = nb;                                                                \
  int lda_ = lda;                                                              \
  int ldt_ = ldt;                                                              \
  int ldy_ = ldy;                                                              \
  int info_ = 0;                                                               \
  if (!g_fortran_fn[OP_ID]) return -1;                                         \
  if (((n > 0 || nb > 0) && (!a || !tau || !t || !y))) return -1;             \
  ((void (*)(int *, int *, int *, TYPE *, int *, TYPE *, TYPE *, int *,        \
             TYPE *, int *, int *))g_fortran_fn[OP_ID])(                       \
      &n_, &k_, &nb_, a, &lda_, tau, t, &ldt_, y, &ldy_, &info_);              \
  return info_;                                                                \
}                                                                              \
static void thunk_##OP##_cblas_to_fortran(                                     \
    int *n, int *k, int *nb, TYPE *a, int *lda, TYPE *tau, TYPE *t, int *ldt,  \
    TYPE *y, int *ldy, int *info) {                                            \
  int info_ = -1;                                                              \
  if (!n || !k || !nb || !lda || !ldt || !ldy || !info) return;               \
  if (!g_cblas_fn[OP_ID]) {                                                    \
    *info = -1;                                                                \
    return;                                                                    \
  }                                                                            \
  if (((*n > 0 || *nb > 0) && (!a || !tau || !t || !y))) {                     \
    *info = -1;                                                                \
    return;                                                                    \
  }                                                                            \
  info_ = ((int (*)(int, int, int, TYPE *, int, TYPE *, TYPE *, int, TYPE *,   \
                    int))g_cblas_fn[OP_ID])(                                   \
      *n, *k, *nb, a, *lda, tau, t, *ldt, y, *ldy);                            \
  *info = info_;                                                               \
}
