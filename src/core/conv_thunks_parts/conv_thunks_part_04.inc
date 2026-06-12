
#define DEFINE_SPGVX_REAL_THUNKS(OP, TYPE, OP_ID)                               \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const int itype, const char jobz,                \
    const char range, const fb_uplo_t uplo, const int n, TYPE *ap, TYPE *bp,   \
    const TYPE vl, const TYPE vu, const int il, const int iu,                  \
    const TYPE abstol, int *m, TYPE *w, TYPE *z, const int ldz, int *ifail) {  \
  char jobz_ = jobz, range_ = range;                                            \
  char uplo_char = 'U';                                                         \
  int itype_ = itype, n_ = n, il_ = il, iu_ = iu;                               \
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
  TYPE *bp_col = NULL;                                                          \
  TYPE *z_col = NULL;                                                           \
  int *iwork = NULL;                                                            \
  TYPE dummy_z = (TYPE)0;                                                       \
  TYPE *ap_arg = ap;                                                            \
  TYPE *bp_arg = bp;                                                            \
  TYPE *z_arg = z ? z : &dummy_z;                                               \
  int dummy_ifail = 0;                                                          \
  int *ifail_ptr = ifail ? ifail : &dummy_ifail;                                \
  int info = 0;                                                                 \
  if (!m || !w || !g_fortran_fn[OP_ID]) return -1;                              \
  if (fb_lapack_uplo_to_char(uplo, &uplo_char) != 0) return -1;                \
  if (vectors_requested && (!z || !ifail)) return -1;                          \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if ((n > 0 && (!ap || !bp))) return -1;                                     \
    if (vectors_requested && ldz < z_cols) return -1;                           \
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
  iwork = (int *)malloc((size_t)iwork_size * sizeof(int));                     \
  if (!iwork) {                                                                 \
    free(ap_col);                                                               \
    free(bp_col);                                                               \
    free(z_col);                                                                \
    free(iwork);                                                                \
    return -101;                                                                \
  }                                                                             \
  ((void (*)(int *, char *, char *, char *, int *, TYPE *, TYPE *, TYPE *,     \
             TYPE *, int *, int *, TYPE *, int *, TYPE *, TYPE *, int *,       \
             TYPE *, int *, int *, int *))g_fortran_fn[OP_ID])(                \
      &itype_, &jobz_, &range_, &uplo_char, &n_, ap_arg, bp_arg, &vl_, &vu_,   \
      &il_, &iu_, &abstol_, m, w, z_arg, &ldz_, NULL, iwork, ifail_ptr,        \
      &info);                                                                   \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if (fb_copy_triangular_packed_from_col_major(ap_col, layout, uplo_char,    \
                                                 'N', n, ap, sizeof(TYPE)) != 0 || \
        fb_copy_triangular_packed_from_col_major(bp_col, layout, uplo_char,    \
                                                 'N', n, bp, sizeof(TYPE)) != 0 || \
        (vectors_requested &&                                                   \
         fb_copy_matrix_from_col_major(z_col, n, z_cols, ldz_, layout, z,      \
                                       ldz, sizeof(TYPE)) != 0)) {              \
      free(ap_col);                                                             \
      free(bp_col);                                                             \
      free(z_col);                                                              \
      free(iwork);                                                              \
      return -1;                                                                \
    }                                                                           \
  }                                                                             \
  free(ap_col);                                                                 \
  free(bp_col);                                                                 \
  free(z_col);                                                                  \
  free(iwork);                                                                  \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    int *itype, char *jobz, char *range, char *uplo, int *n, TYPE *ap,         \
    TYPE *bp, TYPE *vl, TYPE *vu, int *il, int *iu, TYPE *abstol, int *m,      \
    TYPE *w, TYPE *z, int *ldz, TYPE *work, int *iwork, int *ifail, int *info) { \
  fb_uplo_t uplo_enum = FB_UPPER;                                               \
  TYPE dummy_z = (TYPE)0;                                                       \
  TYPE *z_ptr = z ? z : &dummy_z;                                               \
  int ldz_value = (z && ldz) ? *ldz : 1;                                        \
  int dummy_ifail = 0;                                                          \
  int *ifail_ptr = ifail ? ifail : &dummy_ifail;                                \
  (void)work;                                                                   \
  (void)iwork;                                                                  \
  if (!itype || !jobz || !range || !uplo || !n || !vl || !vu || !il || !iu ||  \
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
  int rc = ((int (*)(fb_layout_t, int, char, char, fb_uplo_t, int, TYPE *,     \
                     TYPE *, TYPE, TYPE, int, int, TYPE, int *, TYPE *,        \
                     TYPE *, int, int *))g_cblas_fn[OP_ID])(                    \
      FB_LAYOUT_COL_MAJOR, *itype, *jobz, *range, uplo_enum, *n, ap, bp,       \
      *vl, *vu, *il, *iu, *abstol, m, w, z_ptr, ldz_value, ifail_ptr);         \
  if (info) *info = rc;                                                         \
}

#define DEFINE_HPGV_COMPLEX_THUNKS(OP, CTYPE, RTYPE, OP_ID)                    \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const int itype, const char jobz,                \
    const fb_uplo_t uplo, const int n, CTYPE *ap, CTYPE *bp, RTYPE *w,         \
    CTYPE *z, const int ldz) {                                                  \
  char jobz_ = jobz;                                                            \
  char uplo_char = 'U';                                                         \
  int itype_ = itype, n_ = n;                                                   \
  int ldz_ = (n > 0) ? n : 1;                                                   \
  int vectors_requested = fb_lapack_job_requests_vectors(jobz_);                \
  int work_size = (n > 0) ? (2 * n - 1) : 1;                                   \
  int rwork_size = (n > 0) ? (3 * n - 2) : 1;                                  \
  size_t packed_len = (size_t)((n > 0) ? n : 1) *                              \
                      (size_t)(((n + 1) > 0) ? (n + 1) : 2) / 2U;              \
  size_t z_elems = (size_t)ldz_ * (size_t)((n > 0) ? n : 1);                   \
  CTYPE *ap_col = NULL;                                                         \
  CTYPE *bp_col = NULL;                                                         \
  CTYPE *z_col = NULL;                                                          \
  CTYPE *work = NULL;                                                           \
  RTYPE *rwork = NULL;                                                          \
  CTYPE dummy_z = (CTYPE)0;                                                     \
  CTYPE *ap_arg = ap;                                                           \
  CTYPE *bp_arg = bp;                                                           \
  CTYPE *z_arg = z ? z : &dummy_z;                                              \
  int info = 0;                                                                 \
  if (!w || !g_fortran_fn[OP_ID]) return -1;                                    \
  if (fb_lapack_uplo_to_char(uplo, &uplo_char) != 0) return -1;                \
  if (vectors_requested && !z) return -1;                                       \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if (n > 0 && (!ap || !bp)) return -1;                                       \
    if (vectors_requested && ldz < n) return -1;                                \
    ap_col = (CTYPE *)malloc(packed_len * sizeof(CTYPE));                       \
    bp_col = (CTYPE *)malloc(packed_len * sizeof(CTYPE));                       \
    if (!ap_col || !bp_col) {                                                   \
      free(ap_col);                                                             \
      free(bp_col);                                                             \
      return -101;                                                              \
    }                                                                           \
    if (fb_copy_triangular_packed_to_col_major(ap, layout, uplo_char, 'N', n,  \
                                               ap_col, sizeof(CTYPE)) != 0 ||  \
        fb_copy_triangular_packed_to_col_major(bp, layout, uplo_char, 'N', n,  \
                                               bp_col, sizeof(CTYPE)) != 0) {  \
      free(ap_col);                                                             \
      free(bp_col);                                                             \
      return -1;                                                                \
    }                                                                           \
    ap_arg = ap_col;                                                            \
    bp_arg = bp_col;                                                            \
    if (vectors_requested) {                                                    \
      z_col = (CTYPE *)malloc(z_elems * sizeof(CTYPE));                         \
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
  work = (CTYPE *)malloc((size_t)work_size * sizeof(CTYPE));                    \
  rwork = (RTYPE *)malloc((size_t)rwork_size * sizeof(RTYPE));                  \
  if (!work || !rwork) {                                                        \
    free(ap_col);                                                               \
    free(bp_col);                                                               \
    free(z_col);                                                                \
    free(work);                                                                 \
    free(rwork);                                                                \
    return -101;                                                                \
  }                                                                             \
  ((void (*)(int *, char *, char *, int *, CTYPE *, CTYPE *, RTYPE *, CTYPE *, \
             int *, CTYPE *, RTYPE *, int *))g_fortran_fn[OP_ID])(             \
      &itype_, &jobz_, &uplo_char, &n_, ap_arg, bp_arg, w, z_arg, &ldz_,       \
      work, rwork, &info);                                                      \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if (fb_copy_triangular_packed_from_col_major(ap_col, layout, uplo_char,    \
                                                 'N', n, ap, sizeof(CTYPE)) != 0 || \
        fb_copy_triangular_packed_from_col_major(bp_col, layout, uplo_char,    \
                                                 'N', n, bp, sizeof(CTYPE)) != 0 || \
        (vectors_requested &&                                                   \
         fb_copy_matrix_from_col_major(z_col, n, n, ldz_, layout, z, ldz,      \
                                       sizeof(CTYPE)) != 0)) {                  \
      free(ap_col);                                                             \
      free(bp_col);                                                             \
      free(z_col);                                                              \
      free(work);                                                               \
      free(rwork);                                                              \
      return -1;                                                                \
    }                                                                           \
  }                                                                             \
  free(ap_col);                                                                 \
  free(bp_col);                                                                 \
  free(z_col);                                                                  \
  free(work);                                                                   \
  free(rwork);                                                                  \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    int *itype, char *jobz, char *uplo, int *n, CTYPE *ap, CTYPE *bp,          \
    RTYPE *w, CTYPE *z, int *ldz, CTYPE *work, RTYPE *rwork, int *info) {      \
  fb_uplo_t uplo_enum = FB_UPPER;                                               \
  CTYPE dummy_z = (CTYPE)0;                                                     \
  CTYPE *z_ptr = z ? z : &dummy_z;                                              \
  int ldz_value = (z && ldz) ? *ldz : 1;                                        \
  (void)work;                                                                   \
  (void)rwork;                                                                  \
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
  int rc = ((int (*)(fb_layout_t, int, char, fb_uplo_t, int, CTYPE *, CTYPE *, \
                     RTYPE *, CTYPE *, int))g_cblas_fn[OP_ID])(                \
      FB_LAYOUT_COL_MAJOR, *itype, *jobz, uplo_enum, *n, ap, bp, w, z_ptr,     \
      ldz_value);                                                               \
  if (info) *info = rc;                                                         \
}

#define DEFINE_HPEV_COMPLEX_THUNKS(OP, CTYPE, RTYPE, OP_ID)                    \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const char jobz, const fb_uplo_t uplo,           \
    const int n, CTYPE *ap, RTYPE *w, CTYPE *z, const int ldz) {               \
  char jobz_ = jobz;                                                            \
  char uplo_char = 'U';                                                         \
  int n_ = n;                                                                   \
  int ldz_ = (n > 0) ? n : 1;                                                   \
  int vectors_requested = fb_lapack_job_requests_vectors(jobz_);                \
  int work_size = (n > 0) ? (2 * n - 1) : 1;                                   \
  int rwork_size = (n > 0) ? (3 * n - 2) : 1;                                  \
  size_t packed_len = (size_t)((n > 0) ? n : 1) *                              \
                      (size_t)(((n + 1) > 0) ? (n + 1) : 2) / 2U;              \
  size_t z_elems = (size_t)ldz_ * (size_t)((n > 0) ? n : 1);                   \
  CTYPE *ap_col = NULL;                                                         \
  CTYPE *z_col = NULL;                                                          \
  CTYPE *work = NULL;                                                           \
  RTYPE *rwork = NULL;                                                          \
  CTYPE dummy_z = (CTYPE)0;                                                     \
  CTYPE *ap_arg = ap;                                                           \
  CTYPE *z_arg = z ? z : &dummy_z;                                              \
  int info = 0;                                                                 \
  if (!w || !g_fortran_fn[OP_ID]) return -1;                                    \
  if (fb_lapack_uplo_to_char(uplo, &uplo_char) != 0) return -1;                \
  if (vectors_requested && !z) return -1;                                       \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if (n > 0 && !ap) return -1;                                                \
    if (vectors_requested && ldz < n) return -1;                                \
    ap_col = (CTYPE *)malloc(packed_len * sizeof(CTYPE));                       \
    if (!ap_col) return -101;                                                   \
    if (fb_copy_triangular_packed_to_col_major(ap, layout, uplo_char, 'N', n,  \
                                               ap_col, sizeof(CTYPE)) != 0) {  \
      free(ap_col);                                                             \
      return -1;                                                                \
    }                                                                           \
    ap_arg = ap_col;                                                            \
    if (vectors_requested) {                                                    \
      z_col = (CTYPE *)malloc(z_elems * sizeof(CTYPE));                         \
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
  work = (CTYPE *)malloc((size_t)work_size * sizeof(CTYPE));                    \
  rwork = (RTYPE *)malloc((size_t)rwork_size * sizeof(RTYPE));                  \
  if (!work || !rwork) {                                                        \
    free(ap_col);                                                               \
    free(z_col);                                                                \
    free(work);                                                                 \
    free(rwork);                                                                \
    return -101;                                                                \
  }                                                                             \
  ((void (*)(char *, char *, int *, CTYPE *, RTYPE *, CTYPE *, int *, CTYPE *, \
             RTYPE *, int *))g_fortran_fn[OP_ID])(                             \
      &jobz_, &uplo_char, &n_, ap_arg, w, z_arg, &ldz_, work, rwork, &info);   \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if (fb_copy_triangular_packed_from_col_major(ap_col, layout, uplo_char,    \
                                                 'N', n, ap, sizeof(CTYPE)) != 0 || \
        (vectors_requested &&                                                   \
         fb_copy_matrix_from_col_major(z_col, n, n, ldz_, layout, z, ldz,      \
                                       sizeof(CTYPE)) != 0)) {                  \
      free(ap_col);                                                             \
      free(z_col);                                                              \
      free(work);                                                               \
      free(rwork);                                                              \
      return -1;                                                                \
    }                                                                           \
  }                                                                             \
  free(ap_col);                                                                 \
  free(z_col);                                                                  \
  free(work);                                                                   \
  free(rwork);                                                                  \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    char *jobz, char *uplo, int *n, CTYPE *ap, RTYPE *w, CTYPE *z, int *ldz,   \
    CTYPE *work, RTYPE *rwork, int *info) {                                     \
  fb_uplo_t uplo_enum = FB_UPPER;                                               \
  CTYPE dummy_z = (CTYPE)0;                                                     \
  CTYPE *z_ptr = z ? z : &dummy_z;                                              \
  int ldz_value = (z && ldz) ? *ldz : 1;                                        \
  (void)work;                                                                   \
  (void)rwork;                                                                  \
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
  int rc = ((int (*)(fb_layout_t, char, fb_uplo_t, int, CTYPE *, RTYPE *,      \
                     CTYPE *, int))g_cblas_fn[OP_ID])(                         \
      FB_LAYOUT_COL_MAJOR, *jobz, uplo_enum, *n, ap, w, z_ptr, ldz_value);     \
  if (info) *info = rc;                                                         \
}

#define DEFINE_HPEVD_COMPLEX_THUNKS(OP, CTYPE, RTYPE, OP_ID, QUERY_FN)         \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const char jobz, const fb_uplo_t uplo,           \
    const int n, CTYPE *ap, RTYPE *w, CTYPE *z, const int ldz) {               \
  char jobz_ = jobz;                                                            \
  char uplo_char = 'U';                                                         \
  int n_ = n;                                                                   \
  int ldz_ = (n > 0) ? n : 1;                                                   \
  int vectors_requested = fb_lapack_job_requests_vectors(jobz_);                \
  int lwork_ = -1, lrwork_ = -1, liwork_ = -1, info = 0;                       \
  CTYPE work_query = (CTYPE)0;                                                  \
  RTYPE rwork_query = (RTYPE)0;                                                 \
  int iwork_query = 0;                                                          \
  size_t packed_len = (size_t)((n > 0) ? n : 1) *                              \
                      (size_t)(((n + 1) > 0) ? (n + 1) : 2) / 2U;              \
  size_t z_elems = (size_t)ldz_ * (size_t)((n > 0) ? n : 1);                   \
  CTYPE *ap_col = NULL;                                                         \
  CTYPE *z_col = NULL;                                                          \
  CTYPE *work = NULL;                                                           \
  RTYPE *rwork = NULL;                                                          \
  int *iwork = NULL;                                                            \
  CTYPE dummy_z = (CTYPE)0;                                                     \
  CTYPE *ap_arg = ap;                                                           \
  CTYPE *z_arg = vectors_requested ? z : &dummy_z;                              \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (fb_lapack_uplo_to_char(uplo, &uplo_char) != 0 || !w) return -1;          \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if (n > 0 && !ap) return -1;                                                \
    if (vectors_requested && (!z || ldz < n)) return -1;                       \
    ap_col = (CTYPE *)malloc(packed_len * sizeof(CTYPE));                       \
    if (!ap_col) return -101;                                                   \
    if (fb_copy_triangular_packed_to_col_major(ap, layout, uplo_char, 'N', n,  \
                                               ap_col, sizeof(CTYPE)) != 0) {  \
      free(ap_col);                                                             \
      return -1;                                                                \
    }                                                                           \
    ap_arg = ap_col;                                                            \
    if (vectors_requested) {                                                    \
      z_col = (CTYPE *)malloc(z_elems * sizeof(CTYPE));                         \
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
  ((void (*)(char *, char *, int *, CTYPE *, RTYPE *, CTYPE *, int *, CTYPE *, \
             int *, RTYPE *, int *, int *, int *, int *))g_fortran_fn[OP_ID])( \
      &jobz_, &uplo_char, &n_, ap_arg, w, z_arg, &ldz_, &work_query, &lwork_,  \
      &rwork_query, &lrwork_, &iwork_query, &liwork_, &info);                  \
  if (info != 0) {                                                              \
    free(ap_col);                                                               \
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
    free(ap_col);                                                               \
    free(z_col);                                                                \
    free(work);                                                                 \
    free(rwork);                                                                \
    free(iwork);                                                                \
    return -101;                                                                \
  }                                                                             \
  ((void (*)(char *, char *, int *, CTYPE *, RTYPE *, CTYPE *, int *, CTYPE *, \
             int *, RTYPE *, int *, int *, int *, int *))g_fortran_fn[OP_ID])( \
      &jobz_, &uplo_char, &n_, ap_arg, w, z_arg, &ldz_, work, &lwork_, rwork,  \
      &lrwork_, iwork, &liwork_, &info);                                        \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if (fb_copy_triangular_packed_from_col_major(ap_col, layout, uplo_char,    \
                                                 'N', n, ap, sizeof(CTYPE)) != 0 || \
        (vectors_requested &&                                                   \
         fb_copy_matrix_from_col_major(z_col, n, n, ldz_, layout, z, ldz,      \
                                       sizeof(CTYPE)) != 0)) {                  \
      free(ap_col);                                                             \
      free(z_col);                                                              \
      free(work);                                                               \
      free(rwork);                                                              \
      free(iwork);                                                              \
      return -1;                                                                \
    }                                                                           \
  }                                                                             \
  free(ap_col);                                                                 \
  free(z_col);                                                                  \
  free(work);                                                                   \
  free(rwork);                                                                  \
  free(iwork);                                                                  \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    char *jobz, char *uplo, int *n, CTYPE *ap, RTYPE *w, CTYPE *z, int *ldz,   \
    CTYPE *work, int *lwork, RTYPE *rwork, int *lrwork, int *iwork,            \
    int *liwork, int *info) {                                                   \
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
  int rc = ((int (*)(fb_layout_t, char, fb_uplo_t, int, CTYPE *, RTYPE *,      \
                     CTYPE *, int))g_cblas_fn[OP_ID])(                         \
      FB_LAYOUT_COL_MAJOR, *jobz, uplo_enum, *n, ap, w, z_ptr, ldz_value);     \
  if (info) *info = rc;                                                         \
}

#define DEFINE_HPEVX_COMPLEX_THUNKS(OP, CTYPE, RTYPE, OP_ID)                   \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const char jobz, const char range,               \
    const fb_uplo_t uplo, const int n, CTYPE *ap, const RTYPE vl,              \
    const RTYPE vu, const int il, const int iu, const RTYPE abstol, int *m,    \
    RTYPE *w, CTYPE *z, const int ldz, int *ifail) {                           \
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
  int work_size = (n > 0) ? (2 * n) : 1;                                        \
  int rwork_size = (n > 0) ? (7 * n) : 1;                                       \
  int iwork_size = (n > 0) ? (5 * n) : 1;                                       \
  RTYPE vl_ = vl, vu_ = vu, abstol_ = abstol;                                   \
  size_t packed_len = (size_t)((n > 0) ? n : 1) *                               \
                      (size_t)(((n + 1) > 0) ? (n + 1) : 2) / 2U;              \
  size_t z_elems = (size_t)ldz_ * (size_t)((z_cols > 0) ? z_cols : 1);         \
  CTYPE *ap_col = NULL;                                                         \
  CTYPE *z_col = NULL;                                                          \
  CTYPE *work = NULL;                                                           \
  RTYPE *rwork = NULL;                                                          \
  int *iwork = NULL;                                                            \
  CTYPE dummy_z = (CTYPE)0;                                                     \
  CTYPE *ap_arg = ap;                                                           \
  CTYPE *z_arg = z ? z : &dummy_z;                                              \
  int dummy_ifail = 0;                                                          \
  int *ifail_ptr = ifail ? ifail : &dummy_ifail;                                \
  int info = 0;                                                                 \
  if (!m || !w || !g_fortran_fn[OP_ID]) return -1;                              \
  if (fb_lapack_uplo_to_char(uplo, &uplo_char) != 0) return -1;                \
  if (vectors_requested && (!z || !ifail)) return -1;                          \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if (n > 0 && !ap) return -1;                                                \
    if (vectors_requested && ldz < z_cols) return -1;                           \
    ap_col = (CTYPE *)malloc(packed_len * sizeof(CTYPE));                       \
    if (!ap_col) return -101;                                                   \
    if (fb_copy_triangular_packed_to_col_major(ap, layout, uplo_char, 'N', n,  \
                                               ap_col, sizeof(CTYPE)) != 0) {  \
      free(ap_col);                                                             \
      return -1;                                                                \
    }                                                                           \
    ap_arg = ap_col;                                                            \
    if (vectors_requested) {                                                    \
      z_col = (CTYPE *)malloc(z_elems * sizeof(CTYPE));                         \
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
  work = (CTYPE *)malloc((size_t)work_size * sizeof(CTYPE));                    \
  rwork = (RTYPE *)malloc((size_t)rwork_size * sizeof(RTYPE));                  \
  iwork = (int *)malloc((size_t)iwork_size * sizeof(int));                      \
  if (!work || !rwork || !iwork) {                                              \
    free(ap_col);                                                               \
    free(z_col);                                                                \
    free(work);                                                                 \
    free(rwork);                                                                \
    free(iwork);                                                                \
    return -101;                                                                \
  }                                                                             \
  ((void (*)(char *, char *, char *, int *, CTYPE *, RTYPE *, RTYPE *, int *,  \
             int *, RTYPE *, int *, RTYPE *, CTYPE *, int *, CTYPE *, RTYPE *, \
             int *, int *, int *))g_fortran_fn[OP_ID])(                        \
      &jobz_, &range_, &uplo_char, &n_, ap_arg, &vl_, &vu_, &il_, &iu_,        \
      &abstol_, m, w, z_arg, &ldz_, work, rwork, iwork, ifail_ptr, &info);     \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if (fb_copy_triangular_packed_from_col_major(ap_col, layout, uplo_char,    \
                                                 'N', n, ap, sizeof(CTYPE)) != 0 || \
        (vectors_requested &&                                                   \
         fb_copy_matrix_from_col_major(z_col, n, z_cols, ldz_, layout, z,      \
                                       ldz, sizeof(CTYPE)) != 0)) {             \
      free(ap_col);                                                             \
      free(z_col);                                                              \
      free(work);                                                               \
      free(rwork);                                                              \
      free(iwork);                                                              \
      return -1;                                                                \
    }                                                                           \
  }                                                                             \
  free(ap_col);                                                                 \
  free(z_col);                                                                  \
  free(work);                                                                   \
  free(rwork);                                                                  \
  free(iwork);                                                                  \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    char *jobz, char *range, char *uplo, int *n, CTYPE *ap, RTYPE *vl,         \
    RTYPE *vu, int *il, int *iu, RTYPE *abstol, int *m, RTYPE *w, CTYPE *z,    \
    int *ldz, CTYPE *work, RTYPE *rwork, int *iwork, int *ifail, int *info) {  \
  fb_uplo_t uplo_enum = FB_UPPER;                                               \
  CTYPE dummy_z = (CTYPE)0;                                                     \
  CTYPE *z_ptr = z ? z : &dummy_z;                                              \
  int ldz_value = (z && ldz) ? *ldz : 1;                                        \
  int dummy_ifail = 0;                                                          \
  int *ifail_ptr = ifail ? ifail : &dummy_ifail;                                \
  (void)work;                                                                   \
  (void)rwork;                                                                  \
  (void)iwork;                                                                  \
  if (!jobz || !range || !uplo || !n || !vl || !vu || !il || !iu || !abstol || \
      !m || !w || fb_lapack_char_to_uplo(*uplo, &uplo_enum) != 0) {            \
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
  int rc = ((int (*)(fb_layout_t, char, char, fb_uplo_t, int, CTYPE *, RTYPE,  \
                     RTYPE, int, int, RTYPE, int *, RTYPE *, CTYPE *, int,     \
                     int *))g_cblas_fn[OP_ID])(                                \
      FB_LAYOUT_COL_MAJOR, *jobz, *range, uplo_enum, *n, ap, *vl, *vu, *il,    \
      *iu, *abstol, m, w, z_ptr, ldz_value, ifail_ptr);                        \
  if (info) *info = rc;                                                         \
}

#define DEFINE_HPGVD_COMPLEX_THUNKS(OP, CTYPE, RTYPE, OP_ID, QUERY_FN)         \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const int itype, const char jobz,                \
    const fb_uplo_t uplo, const int n, CTYPE *ap, CTYPE *bp, RTYPE *w,         \
    CTYPE *z, const int ldz) {                                                  \
  char jobz_ = jobz;                                                            \
  char uplo_char = 'U';                                                         \
  int itype_ = itype, n_ = n;                                                   \
  int ldz_ = (n > 0) ? n : 1;                                                   \
  int vectors_requested = fb_lapack_job_requests_vectors(jobz_);                \
  int lwork_ = -1, lrwork_ = -1, liwork_ = -1, info = 0;                       \
  CTYPE work_query = (CTYPE)0;                                                  \
  RTYPE rwork_query = (RTYPE)0;                                                 \
  int iwork_query = 0;                                                          \
  size_t packed_len = (size_t)((n > 0) ? n : 1) *                              \
                      (size_t)(((n + 1) > 0) ? (n + 1) : 2) / 2U;              \
  size_t z_elems = (size_t)ldz_ * (size_t)((n > 0) ? n : 1);                   \
  CTYPE *ap_col = NULL;                                                         \
  CTYPE *bp_col = NULL;                                                         \
  CTYPE *z_col = NULL;                                                          \
  CTYPE *work = NULL;                                                           \
  RTYPE *rwork = NULL;                                                          \
  int *iwork = NULL;                                                            \
  CTYPE dummy_z = (CTYPE)0;                                                     \
  CTYPE *ap_arg = ap;                                                           \
  CTYPE *bp_arg = bp;                                                           \
  CTYPE *z_arg = vectors_requested ? z : &dummy_z;                              \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (fb_lapack_uplo_to_char(uplo, &uplo_char) != 0 || !w) return -1;          \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if (n > 0 && (!ap || !bp)) return -1;                                       \
    if (vectors_requested && (!z || ldz < n)) return -1;                       \
    ap_col = (CTYPE *)malloc(packed_len * sizeof(CTYPE));                       \
    bp_col = (CTYPE *)malloc(packed_len * sizeof(CTYPE));                       \
    if (!ap_col || !bp_col) {                                                   \
      free(ap_col);                                                             \
      free(bp_col);                                                             \
      return -101;                                                              \
    }                                                                           \
    if (fb_copy_triangular_packed_to_col_major(ap, layout, uplo_char, 'N', n,  \
                                               ap_col, sizeof(CTYPE)) != 0 ||  \
        fb_copy_triangular_packed_to_col_major(bp, layout, uplo_char, 'N', n,  \
                                               bp_col, sizeof(CTYPE)) != 0) {  \
      free(ap_col);                                                             \
      free(bp_col);                                                             \
      return -1;                                                                \
    }                                                                           \
    ap_arg = ap_col;                                                            \
    bp_arg = bp_col;                                                            \
    if (vectors_requested) {                                                    \
      z_col = (CTYPE *)malloc(z_elems * sizeof(CTYPE));                         \
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
  ((void (*)(int *, char *, char *, int *, CTYPE *, CTYPE *, RTYPE *, CTYPE *, \
             int *, CTYPE *, int *, RTYPE *, int *, int *, int *, int *))     \
      g_fortran_fn[OP_ID])(&itype_, &jobz_, &uplo_char, &n_, ap_arg, bp_arg,   \
                           w, z_arg, &ldz_, &work_query, &lwork_,              \
                           &rwork_query, &lrwork_, &iwork_query, &liwork_,     \
                           &info);                                              \
  if (info != 0) {                                                              \
    free(ap_col);                                                               \
    free(bp_col);                                                               \
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
    free(ap_col);                                                               \
    free(bp_col);                                                               \
    free(z_col);                                                                \
    free(work);                                                                 \
    free(rwork);                                                                \
    free(iwork);                                                                \
    return -101;                                                                \
  }                                                                             \
  ((void (*)(int *, char *, char *, int *, CTYPE *, CTYPE *, RTYPE *, CTYPE *, \
             int *, CTYPE *, int *, RTYPE *, int *, int *, int *, int *))     \
      g_fortran_fn[OP_ID])(&itype_, &jobz_, &uplo_char, &n_, ap_arg, bp_arg,   \
                           w, z_arg, &ldz_, work, &lwork_, rwork, &lrwork_,    \
                           iwork, &liwork_, &info);                             \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if (fb_copy_triangular_packed_from_col_major(ap_col, layout, uplo_char,    \
                                                 'N', n, ap, sizeof(CTYPE)) != 0 || \
        fb_copy_triangular_packed_from_col_major(bp_col, layout, uplo_char,    \
                                                 'N', n, bp, sizeof(CTYPE)) != 0 || \
        (vectors_requested &&                                                   \
         fb_copy_matrix_from_col_major(z_col, n, n, ldz_, layout, z, ldz,      \
                                       sizeof(CTYPE)) != 0)) {                  \
      free(ap_col);                                                             \
      free(bp_col);                                                             \
      free(z_col);                                                              \
      free(work);                                                               \
      free(rwork);                                                              \
      free(iwork);                                                              \
      return -1;                                                                \
    }                                                                           \
  }                                                                             \
  free(ap_col);                                                                 \
  free(bp_col);                                                                 \
  free(z_col);                                                                  \
  free(work);                                                                   \
  free(rwork);                                                                  \
  free(iwork);                                                                  \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    int *itype, char *jobz, char *uplo, int *n, CTYPE *ap, CTYPE *bp,          \
    RTYPE *w, CTYPE *z, int *ldz, CTYPE *work, int *lwork, RTYPE *rwork,       \
    int *lrwork, int *iwork, int *liwork, int *info) {                         \
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
  int rc = ((int (*)(fb_layout_t, int, char, fb_uplo_t, int, CTYPE *, CTYPE *, \
                     RTYPE *, CTYPE *, int))g_cblas_fn[OP_ID])(                \
      FB_LAYOUT_COL_MAJOR, *itype, *jobz, uplo_enum, *n, ap, bp, w, z_ptr,     \
      ldz_value);                                                               \
  if (info) *info = rc;                                                         \
}

#define DEFINE_HPGVX_COMPLEX_THUNKS(OP, CTYPE, RTYPE, OP_ID)                   \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const int itype, const char jobz,                \
    const char range, const fb_uplo_t uplo, const int n, CTYPE *ap,            \
    CTYPE *bp, const RTYPE vl, const RTYPE vu, const int il, const int iu,     \
    const RTYPE abstol, int *m, RTYPE *w, CTYPE *z, const int ldz,             \
    int *ifail) {                                                               \
  char jobz_ = jobz, range_ = range;                                            \
  char uplo_char = 'U';                                                         \
  int itype_ = itype, n_ = n, il_ = il, iu_ = iu;                               \
  int z_cols = ((range == 'A' || range == 'a' || range == 'V' || range == 'v') \
                    ? ((n > 0) ? n : 1)                                         \
                    : ((range == 'I' || range == 'i')                           \
                           ? (((iu >= il) && n > 0) ? (iu - il + 1) : 1)        \
                           : 1));                                               \
  int ldz_ = (n > 0) ? n : 1;                                                   \
  int vectors_requested = fb_lapack_job_requests_vectors(jobz_);                \
  int work_size = (n > 0) ? (2 * n) : 1;                                        \
  int rwork_size = (n > 0) ? (7 * n) : 1;                                       \
  int iwork_size = (n > 0) ? (5 * n) : 1;                                       \
  RTYPE vl_ = vl, vu_ = vu, abstol_ = abstol;                                   \
  size_t packed_len = (size_t)((n > 0) ? n : 1) *                               \
                      (size_t)(((n + 1) > 0) ? (n + 1) : 2) / 2U;              \
  size_t z_elems = (size_t)ldz_ * (size_t)((z_cols > 0) ? z_cols : 1);         \
  CTYPE *ap_col = NULL;                                                         \
  CTYPE *bp_col = NULL;                                                         \
  CTYPE *z_col = NULL;                                                          \
  CTYPE *work = NULL;                                                           \
  RTYPE *rwork = NULL;                                                          \
  int *iwork = NULL;                                                            \
  CTYPE dummy_z = (CTYPE)0;                                                     \
  CTYPE *ap_arg = ap;                                                           \
  CTYPE *bp_arg = bp;                                                           \
  CTYPE *z_arg = z ? z : &dummy_z;                                              \
  int dummy_ifail = 0;                                                          \
  int *ifail_ptr = ifail ? ifail : &dummy_ifail;                                \
  int info = 0;                                                                 \
  if (!m || !w || !g_fortran_fn[OP_ID]) return -1;                              \
  if (fb_lapack_uplo_to_char(uplo, &uplo_char) != 0) return -1;                \
  if (vectors_requested && (!z || !ifail)) return -1;                          \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if (n > 0 && (!ap || !bp)) return -1;                                       \
    if (vectors_requested && ldz < z_cols) return -1;                           \
    ap_col = (CTYPE *)malloc(packed_len * sizeof(CTYPE));                       \
    bp_col = (CTYPE *)malloc(packed_len * sizeof(CTYPE));                       \
    if (!ap_col || !bp_col) {                                                   \
      free(ap_col);                                                             \
      free(bp_col);                                                             \
      return -101;                                                              \
    }                                                                           \
    if (fb_copy_triangular_packed_to_col_major(ap, layout, uplo_char, 'N', n,  \
                                               ap_col, sizeof(CTYPE)) != 0 ||  \
        fb_copy_triangular_packed_to_col_major(bp, layout, uplo_char, 'N', n,  \
                                               bp_col, sizeof(CTYPE)) != 0) {  \
      free(ap_col);                                                             \
      free(bp_col);                                                             \
      return -1;                                                                \
    }                                                                           \
    ap_arg = ap_col;                                                            \
    bp_arg = bp_col;                                                            \
    if (vectors_requested) {                                                    \
      z_col = (CTYPE *)malloc(z_elems * sizeof(CTYPE));                         \
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
  work = (CTYPE *)malloc((size_t)work_size * sizeof(CTYPE));                    \
  rwork = (RTYPE *)malloc((size_t)rwork_size * sizeof(RTYPE));                  \
  iwork = (int *)malloc((size_t)iwork_size * sizeof(int));                      \
  if (!work || !rwork || !iwork) {                                              \
    free(ap_col);                                                               \
    free(bp_col);                                                               \
    free(z_col);                                                                \
    free(work);                                                                 \
    free(rwork);                                                                \
    free(iwork);                                                                \
    return -101;                                                                \
  }                                                                             \
  ((void (*)(int *, char *, char *, char *, int *, CTYPE *, CTYPE *, RTYPE *,  \
             RTYPE *, int *, int *, RTYPE *, int *, RTYPE *, CTYPE *, int *,   \
             CTYPE *, RTYPE *, int *, int *, int *))g_fortran_fn[OP_ID])(      \
      &itype_, &jobz_, &range_, &uplo_char, &n_, ap_arg, bp_arg, &vl_, &vu_,   \
      &il_, &iu_, &abstol_, m, w, z_arg, &ldz_, work, rwork, iwork, ifail_ptr,  \
      &info);                                                                   \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if (fb_copy_triangular_packed_from_col_major(ap_col, layout, uplo_char,    \
                                                 'N', n, ap, sizeof(CTYPE)) != 0 || \
        fb_copy_triangular_packed_from_col_major(bp_col, layout, uplo_char,    \
                                                 'N', n, bp, sizeof(CTYPE)) != 0 || \
        (vectors_requested &&                                                   \
         fb_copy_matrix_from_col_major(z_col, n, z_cols, ldz_, layout, z,      \
                                       ldz, sizeof(CTYPE)) != 0)) {             \
      free(ap_col);                                                             \
      free(bp_col);                                                             \
      free(z_col);                                                              \
      free(work);                                                               \
      free(rwork);                                                              \
      free(iwork);                                                              \
      return -1;                                                                \
    }                                                                           \
  }                                                                             \
  free(ap_col);                                                                 \
  free(bp_col);                                                                 \
  free(z_col);                                                                  \
  free(work);                                                                   \
  free(rwork);                                                                  \
  free(iwork);                                                                  \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    int *itype, char *jobz, char *range, char *uplo, int *n, CTYPE *ap,        \
    CTYPE *bp, RTYPE *vl, RTYPE *vu, int *il, int *iu, RTYPE *abstol, int *m,  \
    RTYPE *w, CTYPE *z, int *ldz, CTYPE *work, RTYPE *rwork, int *iwork,       \
    int *ifail, int *info) {                                                    \
  fb_uplo_t uplo_enum = FB_UPPER;                                               \
  CTYPE dummy_z = (CTYPE)0;                                                     \
  CTYPE *z_ptr = z ? z : &dummy_z;                                              \
  int ldz_value = (z && ldz) ? *ldz : 1;                                        \
  int dummy_ifail = 0;                                                          \
  int *ifail_ptr = ifail ? ifail : &dummy_ifail;                                \
  (void)work;                                                                   \
  (void)rwork;                                                                  \
  (void)iwork;                                                                  \
  if (!itype || !jobz || !range || !uplo || !n || !vl || !vu || !il || !iu ||  \
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
  int rc = ((int (*)(fb_layout_t, int, char, char, fb_uplo_t, int, CTYPE *,    \
                     CTYPE *, RTYPE, RTYPE, int, int, RTYPE, int *, RTYPE *,   \
                     CTYPE *, int, int *))g_cblas_fn[OP_ID])(                  \
      FB_LAYOUT_COL_MAJOR, *itype, *jobz, *range, uplo_enum, *n, ap, bp,       \
      *vl, *vu, *il, *iu, *abstol, m, w, z_ptr, ldz_value, ifail_ptr);         \
  if (info) *info = rc;                                                         \
}

#define DEFINE_LAPACK_GB_CON_REAL_THUNKS(OP, REAL_TYPE, OP_ID)                \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const char norm, const int n, const int kl,      \
    const int ku, const REAL_TYPE *ab, const int ldab, const int *ipiv,        \
    const REAL_TYPE anorm, REAL_TYPE *rcond) {                                  \
  char norm_char = norm;                                                        \
  int n_ = n, kl_ = kl, ku_ = ku;                                               \
  int ldab_ = (kl >= 0 && ku >= 0) ? (2 * kl + ku + 1) : 1;                    \
  int info = 0;                                                                 \
  int work_len = (n > 0) ? (3 * n) : 1;                                         \
  int iwork_len = (n > 0) ? n : 1;                                              \
  size_t ab_elems = (size_t)ldab_ * (size_t)((n > 0) ? n : 1);                 \
  REAL_TYPE *ab_col = NULL;                                                     \
  const REAL_TYPE *ab_arg = ab;                                                 \
  REAL_TYPE *work = NULL;                                                       \
  int *iwork = NULL;                                                            \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (!ipiv || !rcond) return -1;                                               \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if ((n > 0 && !ab) || ldab < n) return -1;                                 \
    ab_col = (REAL_TYPE *)malloc(ab_elems * sizeof(REAL_TYPE));                 \
    if (!ab_col) return -101;                                                   \
    if (fb_copy_general_band_to_col_major(ab, layout, n, n, kl, kl + ku,       \
                                          ldab, ab_col, ldab_,                  \
                                          sizeof(REAL_TYPE)) != 0) {            \
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
  ((void (*)(char *, int *, int *, int *, const REAL_TYPE *, int *,            \
             const int *, const REAL_TYPE *, REAL_TYPE *, REAL_TYPE *,         \
             int *, int *))g_fortran_fn[OP_ID])(                               \
      &norm_char, &n_, &kl_, &ku_, ab_arg, &ldab_, ipiv, &anorm, rcond, work,  \
      iwork, &info);                                                            \
  free(ab_col);                                                                 \
  free(work);                                                                   \
  free(iwork);                                                                  \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    char *norm, int *n, int *kl, int *ku, REAL_TYPE *ab, int *ldab, int *ipiv,\
    REAL_TYPE *anorm, REAL_TYPE *rcond, REAL_TYPE *work, int *iwork,          \
    int *info) {                                                                \
  (void)work;                                                                   \
  (void)iwork;                                                                  \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!norm || !ipiv || !anorm || !rcond) {                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, char, int, int, int, const REAL_TYPE *, int, \
                     const int *, REAL_TYPE, REAL_TYPE *))g_cblas_fn[OP_ID])(  \
      FB_LAYOUT_COL_MAJOR, *norm, *n, *kl, *ku, ab, *ldab, ipiv, *anorm,      \
      rcond);                                                                   \
  if (info) *info = rc;                                                         \
}

#define DEFINE_LAPACK_GB_CON_COMPLEX_THUNKS(OP, MAT_TYPE, REAL_TYPE, OP_ID)   \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const char norm, const int n, const int kl,      \
    const int ku, const MAT_TYPE *ab, const int ldab, const int *ipiv,         \
    const REAL_TYPE anorm, REAL_TYPE *rcond) {                                  \
  char norm_char = norm;                                                        \
  int n_ = n, kl_ = kl, ku_ = ku;                                               \
  int ldab_ = (kl >= 0 && ku >= 0) ? (2 * kl + ku + 1) : 1;                    \
  int info = 0;                                                                 \
  int work_len = (n > 0) ? (2 * n) : 1;                                         \
  int iwork_len = (n > 0) ? n : 1;                                              \
  size_t ab_elems = (size_t)ldab_ * (size_t)((n > 0) ? n : 1);                 \
  MAT_TYPE *ab_col = NULL;                                                      \
  const MAT_TYPE *ab_arg = ab;                                                  \
  MAT_TYPE *work = NULL;                                                        \
  int *iwork = NULL;                                                            \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (!ipiv || !rcond) return -1;                                               \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if ((n > 0 && !ab) || ldab < n) return -1;                                 \
    ab_col = (MAT_TYPE *)malloc(ab_elems * sizeof(MAT_TYPE));                   \
    if (!ab_col) return -101;                                                   \
    if (fb_copy_general_band_to_col_major(ab, layout, n, n, kl, kl + ku,       \
                                          ldab, ab_col, ldab_,                  \
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
  work = (MAT_TYPE *)malloc((size_t)work_len * sizeof(MAT_TYPE));              \
  iwork = (int *)malloc((size_t)iwork_len * sizeof(int));                      \
  if (!work || !iwork) {                                                        \
    free(ab_col);                                                               \
    free(work);                                                                 \
    free(iwork);                                                                \
    return -101;                                                                \
  }                                                                             \
  ((void (*)(char *, int *, int *, int *, const MAT_TYPE *, int *,             \
             const int *, const REAL_TYPE *, REAL_TYPE *, MAT_TYPE *,          \
             int *, int *))g_fortran_fn[OP_ID])(                               \
      &norm_char, &n_, &kl_, &ku_, ab_arg, &ldab_, ipiv, &anorm, rcond, work,  \
      iwork, &info);                                                            \
  free(ab_col);                                                                 \
  free(work);                                                                   \
  free(iwork);                                                                  \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    char *norm, int *n, int *kl, int *ku, MAT_TYPE *ab, int *ldab, int *ipiv, \
    REAL_TYPE *anorm, REAL_TYPE *rcond, MAT_TYPE *work, int *iwork, int *info) {\
  (void)work;                                                                   \
  (void)iwork;                                                                  \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!norm || !ipiv || !anorm || !rcond) {                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, char, int, int, int, const MAT_TYPE *, int,  \
                     const int *, REAL_TYPE, REAL_TYPE *))g_cblas_fn[OP_ID])(  \
      FB_LAYOUT_COL_MAJOR, *norm, *n, *kl, *ku, ab, *ldab, ipiv, *anorm,      \
      rcond);                                                                   \
  if (info) *info = rc;                                                         \
}

#define DEFINE_LAPACK_PB_SOLVE_THUNKS(OP, MAT_TYPE, OP_ID)                    \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const fb_uplo_t uplo, const int n,               \
    const int kd, const int nrhs, MAT_TYPE *ab, const int ldab,                \
    MAT_TYPE *b, const int ldb) {                                               \
  char uplo_char = 'U';                                                         \
  int n_ = n, kd_ = kd, nrhs_ = nrhs, ldab_ = (kd >= 0) ? (kd + 1) : 1,        \
      ldb_ = (n > 0) ? n : 1, info = 0;                                         \
  size_t ab_elems = (size_t)ldab_ * (size_t)((n > 0) ? n : 1);                 \
  size_t b_elems = (size_t)ldb_ * (size_t)((nrhs > 0) ? nrhs : 1);             \
  MAT_TYPE *ab_col = NULL;                                                      \
  MAT_TYPE *b_col = NULL;                                                       \
  MAT_TYPE *ab_arg = ab;                                                        \
  MAT_TYPE *b_arg = b;                                                          \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (fb_lapack_uplo_to_char(uplo, &uplo_char) != 0) return -1;                \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if ((n > 0 && !ab) || (n > 0 && nrhs > 0 && !b) || ldab < n || ldb < nrhs) {\
      return -1;                                                                \
    }                                                                           \
    ab_col = (MAT_TYPE *)malloc(ab_elems * sizeof(MAT_TYPE));                   \
    b_col = (MAT_TYPE *)malloc(b_elems * sizeof(MAT_TYPE));                     \
    if (!ab_col || !b_col) {                                                    \
      free(ab_col);                                                             \
      free(b_col);                                                              \
      return -101;                                                              \
    }                                                                           \
    if (fb_copy_sym_band_to_col_major(ab, layout, uplo, n, kd, ldab,          \
                                      ab_col, ldab_, sizeof(MAT_TYPE)) != 0 || \
        fb_copy_matrix_to_col_major(b, layout, n, nrhs, ldb,                   \
                                    b_col, ldb_, sizeof(MAT_TYPE)) != 0) {     \
      free(ab_col);                                                             \
      free(b_col);                                                              \
      return -1;                                                                \
    }                                                                           \
    ab_arg = ab_col;                                                            \
    b_arg = b_col;                                                              \
  } else if (fb_lapack_layout_is_col_major(layout)) {                           \
    ldab_ = ldab;                                                               \
    ldb_ = ldb;                                                                 \
    if ((n > 0 && !ab) || (n > 0 && nrhs > 0 && !b)) {                         \
      return -1;                                                                \
    }                                                                           \
  } else {                                                                      \
    return -1;                                                                  \
  }                                                                             \
  ((void (*)(char *, int *, int *, int *, MAT_TYPE *, int *, MAT_TYPE *,       \
             int *, int *))g_fortran_fn[OP_ID])(                               \
      &uplo_char, &n_, &kd_, &nrhs_, ab_arg, &ldab_, b_arg, &ldb_, &info);     \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if (fb_copy_sym_band_from_col_major(ab_col, layout, uplo, n, kd, ldab_,   \
                                        ab, ldab, sizeof(MAT_TYPE)) != 0 ||    \
        fb_copy_matrix_from_col_major(b_col, n, nrhs, ldb_, layout,            \
                                      b, ldb, sizeof(MAT_TYPE)) != 0) {        \
      free(ab_col);                                                             \
      free(b_col);                                                              \
      return -1;                                                                \
    }                                                                           \
    free(ab_col);                                                               \
    free(b_col);                                                                \
  }                                                                             \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    char *uplo, int *n, int *kd, int *nrhs, MAT_TYPE *ab, int *ldab,           \
    MAT_TYPE *b, int *ldb, int *info) {                                         \
  fb_uplo_t uplo_ = FB_UPPER;                                                   \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!uplo || fb_lapack_char_to_uplo(*uplo, &uplo_) != 0) {                   \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, fb_uplo_t, int, int, int, MAT_TYPE *, int,  \
                     MAT_TYPE *, int))g_cblas_fn[OP_ID])(                      \
      FB_LAYOUT_COL_MAJOR, uplo_, *n, *kd, *nrhs, ab, *ldab, b, *ldb);         \
  if (info) *info = rc;                                                         \
}

#define DEFINE_LAPACK_PP_SOLVE_THUNKS(OP, MAT_TYPE, OP_ID)                     \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const fb_uplo_t uplo, const int n,               \
    const int nrhs, MAT_TYPE *ap, MAT_TYPE *b, const int ldb) {                \
  char uplo_char = 'U';                                                         \
  int n_ = n, nrhs_ = nrhs, ldb_ = (n > 0) ? n : 1, info = 0;                  \
  size_t packed_len = (size_t)((n > 0) ? n : 1) *                               \
                      (size_t)(((n + 1) > 0) ? (n + 1) : 2) / 2U;              \
  size_t b_elems = (size_t)ldb_ * (size_t)((nrhs > 0) ? nrhs : 1);             \
  MAT_TYPE *ap_col = NULL;                                                      \
  MAT_TYPE *b_col = NULL;                                                       \
  MAT_TYPE *ap_arg = ap;                                                        \
  MAT_TYPE *b_arg = b;                                                          \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (fb_lapack_uplo_to_char(uplo, &uplo_char) != 0) return -1;                \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if ((n > 0 && !ap) || (n > 0 && nrhs > 0 && !b) || ldb < nrhs) {          \
      return -1;                                                                \
    }                                                                           \
    ap_col = (MAT_TYPE *)malloc(packed_len * sizeof(MAT_TYPE));                 \
    b_col = (MAT_TYPE *)malloc(b_elems * sizeof(MAT_TYPE));                     \
    if (!ap_col || !b_col) {                                                    \
      free(ap_col);                                                             \
      free(b_col);                                                              \
      return -101;                                                              \
    }                                                                           \
    if (fb_copy_triangular_packed_to_col_major(ap, layout, uplo_char, 'N', n,  \
                                               ap_col, sizeof(MAT_TYPE)) != 0 ||\
        fb_copy_matrix_to_col_major(b, layout, n, nrhs, ldb,                   \
                                    b_col, ldb_, sizeof(MAT_TYPE)) != 0) {     \
      free(ap_col);                                                             \
      free(b_col);                                                              \
      return -1;                                                                \
    }                                                                           \
    ap_arg = ap_col;                                                            \
    b_arg = b_col;                                                              \
  } else if (fb_lapack_layout_is_col_major(layout)) {                           \
    ldb_ = ldb;                                                                 \
    if ((n > 0 && !ap) || (n > 0 && nrhs > 0 && !b)) {                         \
      return -1;                                                                \
    }                                                                           \
  } else {                                                                      \
    return -1;                                                                  \
  }                                                                             \
  ((void (*)(char *, int *, int *, MAT_TYPE *, MAT_TYPE *, int *, int *))      \
      g_fortran_fn[OP_ID])(&uplo_char, &n_, &nrhs_, ap_arg, b_arg, &ldb_, &info);\
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if (fb_copy_triangular_packed_from_col_major(ap_col, layout, uplo_char, 'N',\
                                                 n, ap, sizeof(MAT_TYPE)) != 0 ||\
        fb_copy_matrix_from_col_major(b_col, n, nrhs, ldb_, layout,            \
                                      b, ldb, sizeof(MAT_TYPE)) != 0) {        \
      free(ap_col);                                                             \
      free(b_col);                                                              \
      return -1;                                                                \
    }                                                                           \
    free(ap_col);                                                               \
    free(b_col);                                                                \
  }                                                                             \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    char *uplo, int *n, int *nrhs, MAT_TYPE *ap, MAT_TYPE *b, int *ldb,       \
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
  int rc = ((int (*)(fb_layout_t, fb_uplo_t, int, int, MAT_TYPE *, MAT_TYPE *,\
                     int))g_cblas_fn[OP_ID])(                                   \
      FB_LAYOUT_COL_MAJOR, uplo_, *n, *nrhs, ap, b, *ldb);                     \
  if (info) *info = rc;                                                         \
}

#define DEFINE_LAPACK_HP_SOLVE_THUNKS(OP, MAT_TYPE, OP_ID)                     \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const fb_uplo_t uplo, const int n,               \
    const int nrhs, MAT_TYPE *ap, const int *ipiv, MAT_TYPE *b,                \
    const int ldb) {                                                            \
  char uplo_char = 'U';                                                         \
  int n_ = n, nrhs_ = nrhs, ldb_ = (n > 0) ? n : 1, info = 0;                  \
  size_t packed_len = (size_t)((n > 0) ? n : 1) *                               \
                      (size_t)(((n + 1) > 0) ? (n + 1) : 2) / 2U;              \
  size_t b_elems = (size_t)ldb_ * (size_t)((nrhs > 0) ? nrhs : 1);             \
  MAT_TYPE *ap_col = NULL;                                                      \
  MAT_TYPE *b_col = NULL;                                                       \
  MAT_TYPE *ap_arg = ap;                                                        \
  MAT_TYPE *b_arg = b;                                                          \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (fb_lapack_uplo_to_char(uplo, &uplo_char) != 0 || !ipiv) return -1;       \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if ((n > 0 && !ap) || (n > 0 && nrhs > 0 && !b) || ldb < nrhs) {          \
      return -1;                                                                \
    }                                                                           \
    ap_col = (MAT_TYPE *)malloc(packed_len * sizeof(MAT_TYPE));                 \
    b_col = (MAT_TYPE *)malloc(b_elems * sizeof(MAT_TYPE));                     \
    if (!ap_col || !b_col) {                                                    \
      free(ap_col);                                                             \
      free(b_col);                                                              \
      return -101;                                                              \
    }                                                                           \
    if (fb_copy_triangular_packed_to_col_major(ap, layout, uplo_char, 'N', n,  \
                                               ap_col, sizeof(MAT_TYPE)) != 0 ||\
        fb_copy_matrix_to_col_major(b, layout, n, nrhs, ldb,                   \
                                    b_col, ldb_, sizeof(MAT_TYPE)) != 0) {     \
      free(ap_col);                                                             \
      free(b_col);                                                              \
      return -1;                                                                \
    }                                                                           \
    ap_arg = ap_col;                                                            \
    b_arg = b_col;                                                              \
  } else if (fb_lapack_layout_is_col_major(layout)) {                           \
    ldb_ = ldb;                                                                 \
    if ((n > 0 && !ap) || (n > 0 && nrhs > 0 && !b)) {                         \
      return -1;                                                                \
    }                                                                           \
  } else {                                                                      \
    return -1;                                                                  \
  }                                                                             \
  ((void (*)(char *, int *, int *, MAT_TYPE *, int *, MAT_TYPE *, int *,       \
             int *))g_fortran_fn[OP_ID])(                                       \
      &uplo_char, &n_, &nrhs_, ap_arg, (int *)ipiv, b_arg, &ldb_, &info);      \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if (fb_copy_triangular_packed_from_col_major(ap_col, layout, uplo_char,    \
                                                 'N', n, ap, sizeof(MAT_TYPE)) != 0 ||\
        fb_copy_matrix_from_col_major(b_col, n, nrhs, ldb_, layout,            \
                                      b, ldb, sizeof(MAT_TYPE)) != 0) {        \
      free(ap_col);                                                             \
      free(b_col);                                                              \
      return -1;                                                                \
    }                                                                           \
    free(ap_col);                                                               \
    free(b_col);                                                                \
  }                                                                             \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    char *uplo, int *n, int *nrhs, MAT_TYPE *ap, int *ipiv, MAT_TYPE *b,       \
    int *ldb, int *info) {                                                      \
  fb_uplo_t uplo_ = FB_UPPER;                                                   \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!uplo || !n || !nrhs || !ipiv || !ldb ||                                  \
      fb_lapack_char_to_uplo(*uplo, &uplo_) != 0) {                            \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, fb_uplo_t, int, int, MAT_TYPE *,             \
                     const int *, MAT_TYPE *, int))g_cblas_fn[OP_ID])(         \
      FB_LAYOUT_COL_MAJOR, uplo_, *n, *nrhs, ap, ipiv, b, *ldb);               \
  if (info) *info = rc;                                                         \
}

#define DEFINE_LAPACK_PP_FACTORED_SOLVE_THUNKS(OP, MAT_TYPE, OP_ID)            \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const fb_uplo_t uplo, const int n,               \
    const int nrhs, const MAT_TYPE *ap, MAT_TYPE *b, const int ldb) {          \
  char uplo_char = 'U';                                                         \
  int n_ = n, nrhs_ = nrhs, ldb_ = (n > 0) ? n : 1, info = 0;                  \
  size_t packed_len = (size_t)((n > 0) ? n : 1) *                               \
                      (size_t)(((n + 1) > 0) ? (n + 1) : 2) / 2U;              \
  size_t b_elems = (size_t)ldb_ * (size_t)((nrhs > 0) ? nrhs : 1);             \
  MAT_TYPE *ap_col = NULL;                                                      \
  MAT_TYPE *b_col = NULL;                                                       \
  const MAT_TYPE *ap_arg = ap;                                                  \
  MAT_TYPE *b_arg = b;                                                          \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (fb_lapack_uplo_to_char(uplo, &uplo_char) != 0) return -1;                \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if ((n > 0 && !ap) || (n > 0 && nrhs > 0 && !b) || ldb < nrhs) {          \
      return -1;                                                                \
    }                                                                           \
    ap_col = (MAT_TYPE *)malloc(packed_len * sizeof(MAT_TYPE));                 \
    b_col = (MAT_TYPE *)malloc(b_elems * sizeof(MAT_TYPE));                     \
    if (!ap_col || !b_col) {                                                    \
      free(ap_col);                                                             \
      free(b_col);                                                              \
      return -101;                                                              \
    }                                                                           \
    if (fb_copy_triangular_packed_to_col_major(ap, layout, uplo_char, 'N', n,  \
                                               ap_col, sizeof(MAT_TYPE)) != 0 ||\
        fb_copy_matrix_to_col_major(b, layout, n, nrhs, ldb,                   \
                                    b_col, ldb_, sizeof(MAT_TYPE)) != 0) {     \
      free(ap_col);                                                             \
      free(b_col);                                                              \
      return -1;                                                                \
    }                                                                           \
    ap_arg = ap_col;                                                            \
    b_arg = b_col;                                                              \
  } else if (fb_lapack_layout_is_col_major(layout)) {                           \
    ldb_ = ldb;                                                                 \
    if ((n > 0 && !ap) || (n > 0 && nrhs > 0 && !b)) {                         \
      return -1;                                                                \
    }                                                                           \
  } else {                                                                      \
    return -1;                                                                  \
  }                                                                             \
  ((void (*)(char *, int *, int *, const MAT_TYPE *, MAT_TYPE *, int *,        \
             int *))g_fortran_fn[OP_ID])(                                       \
      &uplo_char, &n_, &nrhs_, ap_arg, b_arg, &ldb_, &info);                   \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if (fb_copy_matrix_from_col_major(b_col, n, nrhs, ldb_, layout,            \
                                      b, ldb, sizeof(MAT_TYPE)) != 0) {        \
      free(ap_col);                                                             \
      free(b_col);                                                              \
      return -1;                                                                \
    }                                                                           \
    free(ap_col);                                                               \
    free(b_col);                                                                \
  }                                                                             \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    char *uplo, int *n, int *nrhs, const MAT_TYPE *ap, MAT_TYPE *b, int *ldb, \
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
  int rc = ((int (*)(fb_layout_t, fb_uplo_t, int, int, const MAT_TYPE *,      \
                     MAT_TYPE *, int))g_cblas_fn[OP_ID])(                      \
      FB_LAYOUT_COL_MAJOR, uplo_, *n, *nrhs, ap, b, *ldb);                     \
  if (info) *info = rc;                                                         \
}

#define DEFINE_LAPACK_HP_FACTORED_SOLVE_THUNKS(OP, MAT_TYPE, OP_ID)            \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const fb_uplo_t uplo, const int n,               \
    const int nrhs, const MAT_TYPE *ap, const int *ipiv, MAT_TYPE *b,          \
    const int ldb) {                                                            \
  char uplo_char = 'U';                                                         \
  int n_ = n, nrhs_ = nrhs, ldb_ = (n > 0) ? n : 1, info = 0;                  \
  size_t packed_len = (size_t)((n > 0) ? n : 1) *                               \
                      (size_t)(((n + 1) > 0) ? (n + 1) : 2) / 2U;              \
  size_t b_elems = (size_t)ldb_ * (size_t)((nrhs > 0) ? nrhs : 1);             \
  MAT_TYPE *ap_col = NULL;                                                      \
  MAT_TYPE *b_col = NULL;                                                       \
  const MAT_TYPE *ap_arg = ap;                                                  \
  MAT_TYPE *b_arg = b;                                                          \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (fb_lapack_uplo_to_char(uplo, &uplo_char) != 0 || !ipiv) return -1;       \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if ((n > 0 && !ap) || (n > 0 && nrhs > 0 && !b) || ldb < nrhs) {          \
      return -1;                                                                \
    }                                                                           \
    ap_col = (MAT_TYPE *)malloc(packed_len * sizeof(MAT_TYPE));                 \
    b_col = (MAT_TYPE *)malloc(b_elems * sizeof(MAT_TYPE));                     \
    if (!ap_col || !b_col) {                                                    \
      free(ap_col);                                                             \
      free(b_col);                                                              \
      return -101;                                                              \
    }                                                                           \
    if (fb_copy_triangular_packed_to_col_major(ap, layout, uplo_char, 'N', n,  \
                                               ap_col, sizeof(MAT_TYPE)) != 0 ||\
        fb_copy_matrix_to_col_major(b, layout, n, nrhs, ldb,                   \
                                    b_col, ldb_, sizeof(MAT_TYPE)) != 0) {     \
      free(ap_col);                                                             \
      free(b_col);                                                              \
      return -1;                                                                \
    }                                                                           \
    ap_arg = ap_col;                                                            \
    b_arg = b_col;                                                              \
  } else if (fb_lapack_layout_is_col_major(layout)) {                           \
    ldb_ = ldb;                                                                 \
    if ((n > 0 && !ap) || (n > 0 && nrhs > 0 && !b)) {                         \
      return -1;                                                                \
    }                                                                           \
  } else {                                                                      \
    return -1;                                                                  \
  }                                                                             \
  ((void (*)(char *, int *, int *, const MAT_TYPE *, const int *, MAT_TYPE *,  \
             int *, int *))g_fortran_fn[OP_ID])(                               \
      &uplo_char, &n_, &nrhs_, ap_arg, ipiv, b_arg, &ldb_, &info);             \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if (fb_copy_matrix_from_col_major(b_col, n, nrhs, ldb_, layout,            \
                                      b, ldb, sizeof(MAT_TYPE)) != 0) {        \
      free(ap_col);                                                             \
      free(b_col);                                                              \
      return -1;                                                                \
    }                                                                           \
    free(ap_col);                                                               \
    free(b_col);                                                                \
  }                                                                             \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    char *uplo, int *n, int *nrhs, const MAT_TYPE *ap, const int *ipiv,        \
    MAT_TYPE *b, int *ldb, int *info) {                                         \
  fb_uplo_t uplo_ = FB_UPPER;                                                   \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!uplo || !n || !nrhs || !ipiv || !ldb ||                                  \
      fb_lapack_char_to_uplo(*uplo, &uplo_) != 0) {                            \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, fb_uplo_t, int, int, const MAT_TYPE *,       \
                     const int *, MAT_TYPE *, int))g_cblas_fn[OP_ID])(         \
      FB_LAYOUT_COL_MAJOR, uplo_, *n, *nrhs, ap, ipiv, b, *ldb);               \
  if (info) *info = rc;                                                         \
}

#define DEFINE_LAPACK_PB_FACTOR_THUNKS(OP, MAT_TYPE, OP_ID)                   \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const fb_uplo_t uplo, const int n,               \
    const int kd, MAT_TYPE *ab, const int ldab) {                              \
  char uplo_char = 'U';                                                         \
  int n_ = n, kd_ = kd, ldab_ = (kd >= 0) ? (kd + 1) : 1, info = 0;            \
  size_t ab_elems = (size_t)ldab_ * (size_t)((n > 0) ? n : 1);                 \
  MAT_TYPE *ab_col = NULL;                                                      \
  MAT_TYPE *ab_arg = ab;                                                        \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (fb_lapack_uplo_to_char(uplo, &uplo_char) != 0) return -1;                \
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
  ((void (*)(char *, int *, int *, MAT_TYPE *, int *, int *))                  \
      g_fortran_fn[OP_ID])(&uplo_char, &n_, &kd_, ab_arg, &ldab_, &info);      \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if (fb_copy_sym_band_from_col_major(ab_col, layout, uplo, n, kd, ldab_,    \
                                        ab, ldab, sizeof(MAT_TYPE)) != 0) {    \
      free(ab_col);                                                             \
      return -1;                                                                \
    }                                                                           \
    free(ab_col);                                                               \
  }                                                                             \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    char *uplo, int *n, int *kd, MAT_TYPE *ab, int *ldab, int *info) {         \
  fb_uplo_t uplo_ = FB_UPPER;                                                   \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!uplo || fb_lapack_char_to_uplo(*uplo, &uplo_) != 0) {                   \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, fb_uplo_t, int, int, MAT_TYPE *, int))      \
      g_cblas_fn[OP_ID])(FB_LAYOUT_COL_MAJOR, uplo_, *n, *kd, ab, *ldab);      \
  if (info) *info = rc;                                                         \
}

#define DEFINE_LAPACK_PP_FACTOR_THUNKS(OP, MAT_TYPE, OP_ID)                    \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const fb_uplo_t uplo, const int n,               \
    MAT_TYPE *ap) {                                                             \
  char uplo_char = 'U';                                                         \
  int n_ = n, info = 0;                                                         \
  size_t packed_len = (size_t)((n > 0) ? n : 1) *                               \
                      (size_t)(((n + 1) > 0) ? (n + 1) : 2) / 2U;              \
  MAT_TYPE *ap_col = NULL;                                                      \
  MAT_TYPE *ap_arg = ap;                                                        \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (fb_lapack_uplo_to_char(uplo, &uplo_char) != 0) return -1;                \
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
  ((void (*)(char *, int *, MAT_TYPE *, int *))g_fortran_fn[OP_ID])(           \
      &uplo_char, &n_, ap_arg, &info);                                          \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if (fb_copy_triangular_packed_from_col_major(ap_col, layout, uplo_char,    \
                                                 'N', n, ap,                    \
                                                 sizeof(MAT_TYPE)) != 0) {      \
      free(ap_col);                                                             \
      return -1;                                                                \
    }                                                                           \
    free(ap_col);                                                               \
  }                                                                             \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    char *uplo, int *n, MAT_TYPE *ap, int *info) {                              \
  fb_uplo_t uplo_ = FB_UPPER;                                                   \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!uplo || !n || fb_lapack_char_to_uplo(*uplo, &uplo_) != 0) {             \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, fb_uplo_t, int, MAT_TYPE *))                 \
      g_cblas_fn[OP_ID])(FB_LAYOUT_COL_MAJOR, uplo_, *n, ap);                   \
  if (info) *info = rc;                                                         \
}

#define DEFINE_LAPACK_PB_FACTORED_SOLVE_THUNKS(OP, MAT_TYPE, OP_ID)           \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const fb_uplo_t uplo, const int n,               \
    const int kd, const int nrhs, const MAT_TYPE *ab, const int ldab,          \
    MAT_TYPE *b, const int ldb) {                                               \
  char uplo_char = 'U';                                                         \
  int n_ = n, kd_ = kd, nrhs_ = nrhs, ldab_ = (kd >= 0) ? (kd + 1) : 1,        \
      ldb_ = (n > 0) ? n : 1, info = 0;                                         \
  size_t ab_elems = (size_t)ldab_ * (size_t)((n > 0) ? n : 1);                 \
  size_t b_elems = (size_t)ldb_ * (size_t)((nrhs > 0) ? nrhs : 1);             \
  MAT_TYPE *ab_col = NULL;                                                      \
  MAT_TYPE *b_col = NULL;                                                       \
  const MAT_TYPE *ab_arg = ab;                                                  \
  MAT_TYPE *b_arg = b;                                                          \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (fb_lapack_uplo_to_char(uplo, &uplo_char) != 0) return -1;                \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if ((n > 0 && !ab) || (n > 0 && nrhs > 0 && !b) || ldab < n || ldb < nrhs) {\
      return -1;                                                                \
    }                                                                           \
    ab_col = (MAT_TYPE *)malloc(ab_elems * sizeof(MAT_TYPE));                   \
    b_col = (MAT_TYPE *)malloc(b_elems * sizeof(MAT_TYPE));                     \
    if (!ab_col || !b_col) {                                                    \
      free(ab_col);                                                             \
      free(b_col);                                                              \
      return -101;                                                              \
    }                                                                           \
    if (fb_copy_sym_band_to_col_major(ab, layout, uplo, n, kd, ldab,           \
                                      ab_col, ldab_, sizeof(MAT_TYPE)) != 0 || \
        fb_copy_matrix_to_col_major(b, layout, n, nrhs, ldb,                   \
                                    b_col, ldb_, sizeof(MAT_TYPE)) != 0) {     \
      free(ab_col);                                                             \
      free(b_col);                                                              \
      return -1;                                                                \
    }                                                                           \
    ab_arg = ab_col;                                                            \
    b_arg = b_col;                                                              \
  } else if (fb_lapack_layout_is_col_major(layout)) {                           \
    ldab_ = ldab;                                                               \
    ldb_ = ldb;                                                                 \
    if ((n > 0 && !ab) || (n > 0 && nrhs > 0 && !b)) {                         \
      return -1;                                                                \
    }                                                                           \
  } else {                                                                      \
    return -1;                                                                  \
  }                                                                             \
  ((void (*)(char *, int *, int *, int *, const MAT_TYPE *, int *,            \
             MAT_TYPE *, int *, int *))g_fortran_fn[OP_ID])(                   \
      &uplo_char, &n_, &kd_, &nrhs_, ab_arg, &ldab_, b_arg, &ldb_, &info);     \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if (fb_copy_matrix_from_col_major(b_col, n, nrhs, ldb_, layout,            \
                                      b, ldb, sizeof(MAT_TYPE)) != 0) {        \
      free(ab_col);                                                             \
      free(b_col);                                                              \
      return -1;                                                                \
    }                                                                           \
    free(ab_col);                                                               \
    free(b_col);                                                                \
  }                                                                             \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    char *uplo, int *n, int *kd, int *nrhs, MAT_TYPE *ab, int *ldab,           \
    MAT_TYPE *b, int *ldb, int *info) {                                         \
  fb_uplo_t uplo_ = FB_UPPER;                                                   \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!uplo || fb_lapack_char_to_uplo(*uplo, &uplo_) != 0) {                   \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, fb_uplo_t, int, int, int, const MAT_TYPE *, \
                     int, MAT_TYPE *, int))g_cblas_fn[OP_ID])(                 \
      FB_LAYOUT_COL_MAJOR, uplo_, *n, *kd, *nrhs, ab, *ldab, b, *ldb);         \
  if (info) *info = rc;                                                         \
}

#define DEFINE_LAPACK_PB_SVX_REAL_THUNKS(OP, REAL_TYPE, OP_ID)                \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const char fact, const fb_uplo_t uplo,           \
    const int n, const int kd, const int nrhs, REAL_TYPE *ab, const int ldab, \
    REAL_TYPE *afb, const int ldafb, char *equed, REAL_TYPE *s, REAL_TYPE *b, \
    const int ldb, REAL_TYPE *x, const int ldx, REAL_TYPE *rcond,             \
    REAL_TYPE *ferr, REAL_TYPE *berr) {                                         \
  char uplo_char = 'U';                                                         \
  int n_ = n, kd_ = kd, nrhs_ = nrhs, ldab_ = (kd >= 0) ? (kd + 1) : 1,        \
      ldafb_ = (kd >= 0) ? (kd + 1) : 1, ldb_ = (n > 0) ? n : 1,               \
      ldx_ = (n > 0) ? n : 1, info = 0;                                         \
  int work_len = (n > 0) ? (3 * n) : 1;                                         \
  int iwork_len = (n > 0) ? n : 1;                                              \
  int fact_is_f = (fact == 'F' || fact == 'f');                                 \
  int fact_is_e = (fact == 'E' || fact == 'e');                                 \
  int fact_is_n = (fact == 'N' || fact == 'n');                                 \
  size_t ab_elems = (size_t)ldab_ * (size_t)((n > 0) ? n : 1);                 \
  size_t afb_elems = (size_t)ldafb_ * (size_t)((n > 0) ? n : 1);               \
  size_t b_elems = (size_t)ldb_ * (size_t)((nrhs > 0) ? nrhs : 1);             \
  size_t x_elems = (size_t)ldx_ * (size_t)((nrhs > 0) ? nrhs : 1);             \
  REAL_TYPE *ab_col = NULL;                                                     \
  REAL_TYPE *afb_col = NULL;                                                    \
  REAL_TYPE *b_col = NULL;                                                      \
  REAL_TYPE *x_col = NULL;                                                      \
  REAL_TYPE *ab_arg = ab;                                                       \
  REAL_TYPE *afb_arg = afb;                                                     \
  REAL_TYPE *b_arg = b;                                                         \
  REAL_TYPE *x_arg = x;                                                         \
  REAL_TYPE *work = NULL;                                                       \
  int *iwork = NULL;                                                            \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (fb_lapack_uplo_to_char(uplo, &uplo_char) != 0 || !equed || !s ||         \
      !rcond || !ferr || !berr) {                                               \
    return -1;                                                                  \
  }                                                                             \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if ((n > 0 && (!ab || !afb)) || (n > 0 && nrhs > 0 && (!b || !x)) ||       \
        ldab < n || ldafb < n || ldb < nrhs || ldx < nrhs) {                   \
      return -1;                                                                \
    }                                                                           \
    ab_col = (REAL_TYPE *)malloc(ab_elems * sizeof(REAL_TYPE));                 \
    afb_col = (REAL_TYPE *)calloc(afb_elems, sizeof(REAL_TYPE));                \
    b_col = (REAL_TYPE *)malloc(b_elems * sizeof(REAL_TYPE));                   \
    x_col = (REAL_TYPE *)malloc(x_elems * sizeof(REAL_TYPE));                   \
    if (!ab_col || !afb_col || !b_col || !x_col) {                              \
      free(ab_col);                                                             \
      free(afb_col);                                                            \
      free(b_col);                                                              \
      free(x_col);                                                              \
      return -101;                                                              \
    }                                                                           \
    if (fb_copy_sym_band_to_col_major(ab, layout, uplo, n, kd, ldab,           \
                                      ab_col, ldab_, sizeof(REAL_TYPE)) != 0 ||\
        (fact_is_f &&                                                             \
         fb_copy_sym_band_to_col_major(afb, layout, uplo, n, kd, ldafb,        \
                                       afb_col, ldafb_, sizeof(REAL_TYPE)) != 0) ||\
        fb_copy_matrix_to_col_major(b, layout, n, nrhs, ldb,                   \
                                    b_col, ldb_, sizeof(REAL_TYPE)) != 0) {     \
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
  ((void (*)(char *, char *, int *, int *, int *, REAL_TYPE *, int *,         \
             REAL_TYPE *, int *, char *, REAL_TYPE *, REAL_TYPE *, int *,      \
             REAL_TYPE *, int *, REAL_TYPE *, REAL_TYPE *, REAL_TYPE *,        \
             REAL_TYPE *, int *, int *))g_fortran_fn[OP_ID])(                  \
      (char *)&fact, &uplo_char, &n_, &kd_, &nrhs_, ab_arg, &ldab_, afb_arg,   \
      &ldafb_, equed, s, b_arg, &ldb_, x_arg, &ldx_, rcond, ferr, berr, work,  \
      iwork, &info);                                                            \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if ((fact_is_e && (*equed == 'Y' || *equed == 'y') &&                      \
         fb_copy_sym_band_from_col_major(ab_col, layout, uplo, n, kd, ldab_,   \
                                         ab, ldab, sizeof(REAL_TYPE)) != 0) || \
        ((fact_is_e || fact_is_n) &&                                            \
         fb_copy_sym_band_from_col_major(afb_col, layout, uplo, n, kd,         \
                                         ldafb_, afb, ldafb,                    \
                                         sizeof(REAL_TYPE)) != 0) ||           \
        fb_copy_matrix_from_col_major(b_col, n, nrhs, ldb_, layout,            \
                                      b, ldb, sizeof(REAL_TYPE)) != 0 ||       \
        fb_copy_matrix_from_col_major(x_col, n, nrhs, ldx_, layout,            \
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
    char *fact, char *uplo, int *n, int *kd, int *nrhs, REAL_TYPE *ab,         \
    int *ldab, REAL_TYPE *afb, int *ldafb, char *equed, REAL_TYPE *s,          \
    REAL_TYPE *b, int *ldb, REAL_TYPE *x, int *ldx, REAL_TYPE *rcond,          \
    REAL_TYPE *ferr, REAL_TYPE *berr, REAL_TYPE *work, int *iwork,             \
    int *info) {                                                                 \
  (void)work;                                                                   \
  (void)iwork;                                                                  \
  fb_uplo_t uplo_ = FB_UPPER;                                                   \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!fact || !uplo || !equed || !s || !rcond || !ferr || !berr ||            \
      fb_lapack_char_to_uplo(*uplo, &uplo_) != 0) {                            \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, char, fb_uplo_t, int, int, int,             \
                     REAL_TYPE *, int, REAL_TYPE *, int, char *, REAL_TYPE *, \
                     REAL_TYPE *, int, REAL_TYPE *, int, REAL_TYPE *,         \
                     REAL_TYPE *, REAL_TYPE *))g_cblas_fn[OP_ID])(            \
      FB_LAYOUT_COL_MAJOR, *fact, uplo_, *n, *kd, *nrhs, ab, *ldab, afb,      \
      *ldafb, equed, s, b, *ldb, x, *ldx, rcond, ferr, berr);                 \
  if (info) *info = rc;                                                         \
}

#define DEFINE_LAPACK_PB_SVX_COMPLEX_THUNKS(OP, MAT_TYPE, REAL_TYPE, OP_ID)   \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const char fact, const fb_uplo_t uplo,           \
    const int n, const int kd, const int nrhs, MAT_TYPE *ab, const int ldab,  \
    MAT_TYPE *afb, const int ldafb, char *equed, REAL_TYPE *s, MAT_TYPE *b,   \
    const int ldb, MAT_TYPE *x, const int ldx, REAL_TYPE *rcond,              \
    REAL_TYPE *ferr, REAL_TYPE *berr) {                                         \
  char uplo_char = 'U';                                                         \
  int n_ = n, kd_ = kd, nrhs_ = nrhs, ldab_ = (kd >= 0) ? (kd + 1) : 1,        \
      ldafb_ = (kd >= 0) ? (kd + 1) : 1, ldb_ = (n > 0) ? n : 1,               \
      ldx_ = (n > 0) ? n : 1, info = 0;                                         \
  int work_len = (n > 0) ? (2 * n) : 1;                                         \
  int rwork_len = (n > 0) ? n : 1;                                              \
  int fact_is_f = (fact == 'F' || fact == 'f');                                 \
  int fact_is_e = (fact == 'E' || fact == 'e');                                 \
  int fact_is_n = (fact == 'N' || fact == 'n');                                 \
  size_t ab_elems = (size_t)ldab_ * (size_t)((n > 0) ? n : 1);                 \
  size_t afb_elems = (size_t)ldafb_ * (size_t)((n > 0) ? n : 1);               \
  size_t b_elems = (size_t)ldb_ * (size_t)((nrhs > 0) ? nrhs : 1);             \
  size_t x_elems = (size_t)ldx_ * (size_t)((nrhs > 0) ? nrhs : 1);             \
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
  if (fb_lapack_uplo_to_char(uplo, &uplo_char) != 0 || !equed || !s ||         \
      !rcond || !ferr || !berr) {                                               \
    return -1;                                                                  \
  }                                                                             \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if ((n > 0 && (!ab || !afb)) || (n > 0 && nrhs > 0 && (!b || !x)) ||       \
        ldab < n || ldafb < n || ldb < nrhs || ldx < nrhs) {                   \
      return -1;                                                                \
    }                                                                           \
    ab_col = (MAT_TYPE *)malloc(ab_elems * sizeof(MAT_TYPE));                   \
    afb_col = (MAT_TYPE *)calloc(afb_elems, sizeof(MAT_TYPE));                  \
    b_col = (MAT_TYPE *)malloc(b_elems * sizeof(MAT_TYPE));                     \
    x_col = (MAT_TYPE *)malloc(x_elems * sizeof(MAT_TYPE));                     \
    if (!ab_col || !afb_col || !b_col || !x_col) {                              \
      free(ab_col);                                                             \
      free(afb_col);                                                            \
      free(b_col);                                                              \
      free(x_col);                                                              \
      return -101;                                                              \
    }                                                                           \
    if (fb_copy_sym_band_to_col_major(ab, layout, uplo, n, kd, ldab,           \
                                      ab_col, ldab_, sizeof(MAT_TYPE)) != 0 || \
        (fact_is_f &&                                                             \
         fb_copy_sym_band_to_col_major(afb, layout, uplo, n, kd, ldafb,        \
                                       afb_col, ldafb_, sizeof(MAT_TYPE)) != 0) ||\
        fb_copy_matrix_to_col_major(b, layout, n, nrhs, ldb,                   \
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
  ((void (*)(char *, char *, int *, int *, int *, MAT_TYPE *, int *,          \
             MAT_TYPE *, int *, char *, REAL_TYPE *, MAT_TYPE *, int *,        \
             MAT_TYPE *, int *, REAL_TYPE *, REAL_TYPE *, REAL_TYPE *,         \
             MAT_TYPE *, REAL_TYPE *, int *))g_fortran_fn[OP_ID])(             \
      (char *)&fact, &uplo_char, &n_, &kd_, &nrhs_, ab_arg, &ldab_, afb_arg,   \
      &ldafb_, equed, s, b_arg, &ldb_, x_arg, &ldx_, rcond, ferr, berr, work,  \
      rwork, &info);                                                            \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if ((fact_is_e && (*equed == 'Y' || *equed == 'y') &&                      \
         fb_copy_sym_band_from_col_major(ab_col, layout, uplo, n, kd, ldab_,   \
                                         ab, ldab, sizeof(MAT_TYPE)) != 0) ||  \
        ((fact_is_e || fact_is_n) &&                                            \
         fb_copy_sym_band_from_col_major(afb_col, layout, uplo, n, kd,         \
                                         ldafb_, afb, ldafb,                    \
                                         sizeof(MAT_TYPE)) != 0) ||            \
        fb_copy_matrix_from_col_major(b_col, n, nrhs, ldb_, layout,            \
                                      b, ldb, sizeof(MAT_TYPE)) != 0 ||        \
        fb_copy_matrix_from_col_major(x_col, n, nrhs, ldx_, layout,            \
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
    char *fact, char *uplo, int *n, int *kd, int *nrhs, MAT_TYPE *ab,          \
    int *ldab, MAT_TYPE *afb, int *ldafb, char *equed, REAL_TYPE *s,           \
    MAT_TYPE *b, int *ldb, MAT_TYPE *x, int *ldx, REAL_TYPE *rcond,            \
    REAL_TYPE *ferr, REAL_TYPE *berr, MAT_TYPE *work, REAL_TYPE *rwork,        \
    int *info) {                                                                 \
  (void)work;                                                                   \
  (void)rwork;                                                                  \
  fb_uplo_t uplo_ = FB_UPPER;                                                   \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!fact || !uplo || !equed || !s || !rcond || !ferr || !berr ||            \
      fb_lapack_char_to_uplo(*uplo, &uplo_) != 0) {                            \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, char, fb_uplo_t, int, int, int,             \
                     MAT_TYPE *, int, MAT_TYPE *, int, char *, REAL_TYPE *,   \
                     MAT_TYPE *, int, MAT_TYPE *, int, REAL_TYPE *,           \
                     REAL_TYPE *, REAL_TYPE *))g_cblas_fn[OP_ID])(            \
      FB_LAYOUT_COL_MAJOR, *fact, uplo_, *n, *kd, *nrhs, ab, *ldab, afb,      \
      *ldafb, equed, s, b, *ldb, x, *ldx, rcond, ferr, berr);                 \
  if (info) *info = rc;                                                         \
}

#define DEFINE_LAPACK_PP_SVX_REAL_THUNKS(OP, REAL_TYPE, OP_ID)                 \
static int thunk_##OP##_fortran_to_cblas(                                       \
    const fb_layout_t layout, const char fact, const fb_uplo_t uplo,            \
    const int n, const int nrhs, REAL_TYPE *ap, REAL_TYPE *afb, char *equed,    \
    REAL_TYPE *s, REAL_TYPE *b, const int ldb, REAL_TYPE *x, const int ldx,     \
    REAL_TYPE *rcond, REAL_TYPE *ferr, REAL_TYPE *berr) {                        \
  char uplo_char = 'U';                                                          \
  int n_ = n, nrhs_ = nrhs, ldb_ = (n > 0) ? n : 1, ldx_ = (n > 0) ? n : 1,    \
      info = 0;                                                                  \
  int work_len = (n > 0) ? (3 * n) : 1;                                         \
  int iwork_len = (n > 0) ? n : 1;                                              \
  int fact_is_f = (fact == 'F' || fact == 'f');                                 \
  int fact_is_e = (fact == 'E' || fact == 'e');                                 \
  int fact_is_n = (fact == 'N' || fact == 'n');                                 \
  int equed_has_scaling = 0;                                                     \
  size_t packed_len = (size_t)((n > 0) ? n : 1) *                               \
                      (size_t)(((n + 1) > 0) ? (n + 1) : 2) / 2U;              \
  size_t b_elems = (size_t)ldb_ * (size_t)((nrhs > 0) ? nrhs : 1);             \
  size_t x_elems = (size_t)ldx_ * (size_t)((nrhs > 0) ? nrhs : 1);             \
  REAL_TYPE *ap_col = NULL;                                                      \
  REAL_TYPE *afb_col = NULL;                                                     \
  REAL_TYPE *b_col = NULL;                                                       \
  REAL_TYPE *x_col = NULL;                                                       \
  REAL_TYPE *ap_arg = ap;                                                        \
  REAL_TYPE *afb_arg = afb;                                                      \
  REAL_TYPE *b_arg = b;                                                          \
  REAL_TYPE *x_arg = x;                                                          \
  REAL_TYPE *work = NULL;                                                        \
  int *iwork = NULL;                                                             \
  if (!g_fortran_fn[OP_ID]) return -1;                                           \
  if (fb_lapack_uplo_to_char(uplo, &uplo_char) != 0 || !equed || !s || !rcond ||\
      !ferr || !berr) return -1;                                                 \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                           \
    if ((n > 0 && (!ap || !afb)) || (n > 0 && nrhs > 0 && (!b || !x)) ||       \
        ldb < nrhs || ldx < nrhs) {                                              \
      return -1;                                                                 \
    }                                                                            \
    ap_col = (REAL_TYPE *)malloc(packed_len * sizeof(REAL_TYPE));                \
    afb_col = (REAL_TYPE *)calloc(packed_len, sizeof(REAL_TYPE));                \
    b_col = (REAL_TYPE *)malloc(b_elems * sizeof(REAL_TYPE));                    \
    x_col = (REAL_TYPE *)malloc(x_elems * sizeof(REAL_TYPE));                    \
    if (!ap_col || !afb_col || !b_col || !x_col) {                               \
      free(ap_col); free(afb_col); free(b_col); free(x_col);                     \
      return -101;                                                               \
    }                                                                            \
    if (fb_copy_triangular_packed_to_col_major(ap, layout, uplo_char, 'N', n,   \
                                               ap_col, sizeof(REAL_TYPE)) != 0 ||\
        (fact_is_f &&                                                             \
         fb_copy_triangular_packed_to_col_major(afb, layout, uplo_char, 'N', n, \
                                                afb_col, sizeof(REAL_TYPE)) != 0) ||\
        fb_copy_matrix_to_col_major(b, layout, n, nrhs, ldb,                    \
                                    b_col, ldb_, sizeof(REAL_TYPE)) != 0) {     \
      free(ap_col); free(afb_col); free(b_col); free(x_col);                     \
      return -1;                                                                 \
    }                                                                            \
    ap_arg = ap_col;                                                             \
    afb_arg = afb_col;                                                           \
    b_arg = b_col;                                                               \
    x_arg = x_col;                                                               \
  } else if (fb_lapack_layout_is_col_major(layout)) {                            \
    ldb_ = ldb;                                                                  \
    ldx_ = ldx;                                                                  \
    if ((n > 0 && (!ap || !afb)) || (n > 0 && nrhs > 0 && (!b || !x))) {       \
      return -1;                                                                 \
    }                                                                            \
  } else {                                                                       \
    return -1;                                                                   \
  }                                                                              \
  work = (REAL_TYPE *)calloc((size_t)work_len, sizeof(REAL_TYPE));              \
  iwork = (int *)calloc((size_t)iwork_len, sizeof(int));                        \
  if (!work || !iwork) {                                                         \
    free(ap_col); free(afb_col); free(b_col); free(x_col); free(work); free(iwork);\
    return -101;                                                                 \
  }                                                                              \
  ((void (*)(char *, char *, int *, int *, REAL_TYPE *, REAL_TYPE *, char *,    \
             REAL_TYPE *, REAL_TYPE *, int *, REAL_TYPE *, int *, REAL_TYPE *,  \
             REAL_TYPE *, REAL_TYPE *, REAL_TYPE *, int *, int *))             \
      g_fortran_fn[OP_ID])((char *)&fact, &uplo_char, &n_, &nrhs_, ap_arg,      \
                           afb_arg, equed, s, b_arg, &ldb_, x_arg, &ldx_,       \
                           rcond, ferr, berr, work, iwork, &info);              \
  equed_has_scaling = (*equed == 'Y' || *equed == 'y');                          \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                           \
    if ((fact_is_e && equed_has_scaling &&                                       \
         fb_copy_triangular_packed_from_col_major(ap_col, layout, uplo_char, 'N',\
                                                  n, ap, sizeof(REAL_TYPE)) != 0) ||\
        ((fact_is_e || fact_is_n) &&                                             \
         fb_copy_triangular_packed_from_col_major(afb_col, layout, uplo_char, 'N',\
                                                  n, afb, sizeof(REAL_TYPE)) != 0) ||\
        (fact_is_f && equed_has_scaling &&                                       \
         fb_copy_matrix_from_col_major(b_col, n, nrhs, ldb_, layout,            \
                                       b, ldb, sizeof(REAL_TYPE)) != 0) ||      \
        fb_copy_matrix_from_col_major(x_col, n, nrhs, ldx_, layout,             \
                                      x, ldx, sizeof(REAL_TYPE)) != 0) {        \
      free(ap_col); free(afb_col); free(b_col); free(x_col); free(work); free(iwork);\
      return -1;                                                                 \
    }                                                                            \
    free(ap_col); free(afb_col); free(b_col); free(x_col);                       \
  }                                                                              \
  free(work); free(iwork);                                                       \
  return info;                                                                   \
}                                                                                \
static void thunk_##OP##_cblas_to_fortran(                                       \
    char *fact, char *uplo, int *n, int *nrhs, REAL_TYPE *ap, REAL_TYPE *afb,   \
    char *equed, REAL_TYPE *s, REAL_TYPE *b, int *ldb, REAL_TYPE *x, int *ldx, \
    REAL_TYPE *rcond, REAL_TYPE *ferr, REAL_TYPE *berr, REAL_TYPE *work,        \
    int *iwork, int *info) {                                                     \
  fb_uplo_t uplo_ = FB_UPPER;                                                    \
  (void)work;                                                                    \
  (void)iwork;                                                                   \
  if (!g_cblas_fn[OP_ID]) {                                                      \
    if (info) *info = -1;                                                        \
    return;                                                                      \
  }                                                                              \
  if (!fact || !uplo || !equed || !s || !rcond || !ferr || !berr ||             \
      fb_lapack_char_to_uplo(*uplo, &uplo_) != 0) {                             \
    if (info) *info = -1;                                                        \
    return;                                                                      \
  }                                                                              \
  int rc = ((int (*)(fb_layout_t, char, fb_uplo_t, int, int, REAL_TYPE *,      \
                     REAL_TYPE *, char *, REAL_TYPE *, REAL_TYPE *, int,        \
                     REAL_TYPE *, int, REAL_TYPE *, REAL_TYPE *, REAL_TYPE *))  \
      g_cblas_fn[OP_ID])(FB_LAYOUT_COL_MAJOR, *fact, uplo_, *n, *nrhs, ap,      \
                         afb, equed, s, b, *ldb, x, *ldx, rcond, ferr, berr);   \
  if (info) *info = rc;                                                          \
}

#define DEFINE_LAPACK_PP_SVX_COMPLEX_THUNKS(OP, MAT_TYPE, REAL_TYPE, OP_ID)    \
static int thunk_##OP##_fortran_to_cblas(                                       \
    const fb_layout_t layout, const char fact, const fb_uplo_t uplo,            \
    const int n, const int nrhs, MAT_TYPE *ap, MAT_TYPE *afb, char *equed,      \
    REAL_TYPE *s, MAT_TYPE *b, const int ldb, MAT_TYPE *x, const int ldx,       \
    REAL_TYPE *rcond, REAL_TYPE *ferr, REAL_TYPE *berr) {                        \
  char uplo_char = 'U';                                                          \
  int n_ = n, nrhs_ = nrhs, ldb_ = (n > 0) ? n : 1, ldx_ = (n > 0) ? n : 1,    \
      info = 0;                                                                  \
  int work_len = (n > 0) ? (2 * n) : 1;                                         \
  int rwork_len = (n > 0) ? n : 1;                                              \
  int fact_is_f = (fact == 'F' || fact == 'f');                                 \
  int fact_is_e = (fact == 'E' || fact == 'e');                                 \
  int fact_is_n = (fact == 'N' || fact == 'n');                                 \
  int equed_has_scaling = 0;                                                     \
  size_t packed_len = (size_t)((n > 0) ? n : 1) *                               \
                      (size_t)(((n + 1) > 0) ? (n + 1) : 2) / 2U;              \
  size_t b_elems = (size_t)ldb_ * (size_t)((nrhs > 0) ? nrhs : 1);             \
  size_t x_elems = (size_t)ldx_ * (size_t)((nrhs > 0) ? nrhs : 1);             \
  MAT_TYPE *ap_col = NULL;                                                       \
  MAT_TYPE *afb_col = NULL;                                                      \
  MAT_TYPE *b_col = NULL;                                                        \
  MAT_TYPE *x_col = NULL;                                                        \
  MAT_TYPE *ap_arg = ap;                                                         \
  MAT_TYPE *afb_arg = afb;                                                       \
  MAT_TYPE *b_arg = b;                                                           \
  MAT_TYPE *x_arg = x;                                                           \
  MAT_TYPE *work = NULL;                                                         \
  REAL_TYPE *rwork = NULL;                                                       \
  if (!g_fortran_fn[OP_ID]) return -1;                                           \
  if (fb_lapack_uplo_to_char(uplo, &uplo_char) != 0 || !equed || !s || !rcond ||\
      !ferr || !berr) return -1;                                                 \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                           \
    if ((n > 0 && (!ap || !afb)) || (n > 0 && nrhs > 0 && (!b || !x)) ||       \
        ldb < nrhs || ldx < nrhs) {                                              \
      return -1;                                                                 \
    }                                                                            \
    ap_col = (MAT_TYPE *)malloc(packed_len * sizeof(MAT_TYPE));                  \
    afb_col = (MAT_TYPE *)calloc(packed_len, sizeof(MAT_TYPE));                  \
    b_col = (MAT_TYPE *)malloc(b_elems * sizeof(MAT_TYPE));                      \
    x_col = (MAT_TYPE *)malloc(x_elems * sizeof(MAT_TYPE));                      \
    if (!ap_col || !afb_col || !b_col || !x_col) {                               \
      free(ap_col); free(afb_col); free(b_col); free(x_col);                     \
      return -101;                                                               \
    }                                                                            \
    if (fb_copy_triangular_packed_to_col_major(ap, layout, uplo_char, 'N', n,   \
                                               ap_col, sizeof(MAT_TYPE)) != 0 || \
        (fact_is_f &&                                                             \
         fb_copy_triangular_packed_to_col_major(afb, layout, uplo_char, 'N', n, \
                                                afb_col, sizeof(MAT_TYPE)) != 0) ||\
        fb_copy_matrix_to_col_major(b, layout, n, nrhs, ldb,                    \
                                    b_col, ldb_, sizeof(MAT_TYPE)) != 0) {      \
      free(ap_col); free(afb_col); free(b_col); free(x_col);                     \
      return -1;                                                                 \
    }                                                                            \
    ap_arg = ap_col;                                                             \
    afb_arg = afb_col;                                                           \
    b_arg = b_col;                                                               \
    x_arg = x_col;                                                               \
  } else if (fb_lapack_layout_is_col_major(layout)) {                            \
    ldb_ = ldb;                                                                  \
    ldx_ = ldx;                                                                  \
    if ((n > 0 && (!ap || !afb)) || (n > 0 && nrhs > 0 && (!b || !x))) {       \
      return -1;                                                                 \
    }                                                                            \
  } else {                                                                       \
    return -1;                                                                   \
  }                                                                              \
  work = (MAT_TYPE *)calloc((size_t)work_len, sizeof(MAT_TYPE));                \
  rwork = (REAL_TYPE *)calloc((size_t)rwork_len, sizeof(REAL_TYPE));            \
  if (!work || !rwork) {                                                         \
    free(ap_col); free(afb_col); free(b_col); free(x_col); free(work); free(rwork);\
    return -101;                                                                 \
  }                                                                              \
  ((void (*)(char *, char *, int *, int *, MAT_TYPE *, MAT_TYPE *, char *,      \
             REAL_TYPE *, MAT_TYPE *, int *, MAT_TYPE *, int *, REAL_TYPE *,    \
             REAL_TYPE *, REAL_TYPE *, MAT_TYPE *, REAL_TYPE *, int *))         \
      g_fortran_fn[OP_ID])((char *)&fact, &uplo_char, &n_, &nrhs_, ap_arg,      \
                           afb_arg, equed, s, b_arg, &ldb_, x_arg, &ldx_,       \
                           rcond, ferr, berr, work, rwork, &info);              \
  equed_has_scaling = (*equed == 'Y' || *equed == 'y');                          \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                           \
    if ((fact_is_e && equed_has_scaling &&                                       \
         fb_copy_triangular_packed_from_col_major(ap_col, layout, uplo_char, 'N',\
                                                  n, ap, sizeof(MAT_TYPE)) != 0) ||\
        ((fact_is_e || fact_is_n) &&                                             \
         fb_copy_triangular_packed_from_col_major(afb_col, layout, uplo_char, 'N',\
                                                  n, afb, sizeof(MAT_TYPE)) != 0) ||\
        (fact_is_f && equed_has_scaling &&                                       \
         fb_copy_matrix_from_col_major(b_col, n, nrhs, ldb_, layout,            \
                                       b, ldb, sizeof(MAT_TYPE)) != 0) ||       \
        fb_copy_matrix_from_col_major(x_col, n, nrhs, ldx_, layout,             \
                                      x, ldx, sizeof(MAT_TYPE)) != 0) {         \
      free(ap_col); free(afb_col); free(b_col); free(x_col); free(work); free(rwork);\
      return -1;                                                                 \
    }                                                                            \
    free(ap_col); free(afb_col); free(b_col); free(x_col);                       \
  }                                                                              \
  free(work); free(rwork);                                                       \
  return info;                                                                   \
}                                                                                \
static void thunk_##OP##_cblas_to_fortran(                                       \
    char *fact, char *uplo, int *n, int *nrhs, MAT_TYPE *ap, MAT_TYPE *afb,     \
    char *equed, REAL_TYPE *s, MAT_TYPE *b, int *ldb, MAT_TYPE *x, int *ldx,   \
    REAL_TYPE *rcond, REAL_TYPE *ferr, REAL_TYPE *berr, MAT_TYPE *work,         \
    REAL_TYPE *rwork, int *info) {                                               \
  fb_uplo_t uplo_ = FB_UPPER;                                                    \
  (void)work;                                                                    \
  (void)rwork;                                                                   \
  if (!g_cblas_fn[OP_ID]) {                                                      \
    if (info) *info = -1;                                                        \
    return;                                                                      \
  }                                                                              \
  if (!fact || !uplo || !equed || !s || !rcond || !ferr || !berr ||             \
      fb_lapack_char_to_uplo(*uplo, &uplo_) != 0) {                             \
    if (info) *info = -1;                                                        \
    return;                                                                      \
  }                                                                              \
  int rc = ((int (*)(fb_layout_t, char, fb_uplo_t, int, int, MAT_TYPE *,       \
                     MAT_TYPE *, char *, REAL_TYPE *, MAT_TYPE *, int,          \
                     MAT_TYPE *, int, REAL_TYPE *, REAL_TYPE *, REAL_TYPE *))   \
      g_cblas_fn[OP_ID])(FB_LAYOUT_COL_MAJOR, *fact, uplo_, *n, *nrhs, ap,      \
                         afb, equed, s, b, *ldb, x, *ldx, rcond, ferr, berr);   \
  if (info) *info = rc;                                                          \
}

#define DEFINE_LAPACK_GB_SVX_REAL_THUNKS(OP, REAL_TYPE, OP_ID)                \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const char fact, const char trans, const int n, \
    const int kl, const int ku, const int nrhs, REAL_TYPE *ab, const int ldab,\
    REAL_TYPE *afb, const int ldafb, int *ipiv, char *equed, REAL_TYPE *r,    \
    REAL_TYPE *c, REAL_TYPE *b, const int ldb, REAL_TYPE *x, const int ldx,   \
    REAL_TYPE *rcond, REAL_TYPE *ferr, REAL_TYPE *berr,                        \
    REAL_TYPE *rpvgrw) {                                                        \
  int compact_rows = (kl >= 0 && ku >= 0) ? (kl + ku + 1) : 1;                \
  int fact_rows = (kl >= 0 && ku >= 0) ? (2 * kl + ku + 1) : 1;               \
  int n_ = n, kl_ = kl, ku_ = ku, nrhs_ = nrhs, ldab_ = fact_rows,            \
      ldafb_ = fact_rows, ldb_ = (n > 0) ? n : 1, ldx_ = (n > 0) ? n : 1,     \
      info = 0;                                                                 \
  int work_len = (n > 0) ? (3 * n) : 1;                                        \
  int iwork_len = (n > 0) ? n : 1;                                             \
  int fact_is_f = (fact == 'F' || fact == 'f');                                \
  int fact_is_e = (fact == 'E' || fact == 'e');                                \
  int fact_is_n = (fact == 'N' || fact == 'n');                                \
  int equed_has_scaling = 0;                                                    \
  size_t ab_elems = (size_t)fact_rows * (size_t)((n > 0) ? n : 1);            \
  size_t afb_elems = (size_t)fact_rows * (size_t)((n > 0) ? n : 1);           \
  size_t b_elems = (size_t)ldb_ * (size_t)((nrhs > 0) ? nrhs : 1);            \
  size_t x_elems = (size_t)ldx_ * (size_t)((nrhs > 0) ? nrhs : 1);            \
  REAL_TYPE *ab_col = NULL;                                                     \
  REAL_TYPE *afb_col = NULL;                                                    \
  REAL_TYPE *b_col = NULL;                                                      \
  REAL_TYPE *x_col = NULL;                                                      \
  REAL_TYPE *ab_arg = ab;                                                       \
  REAL_TYPE *afb_arg = afb;                                                     \
  REAL_TYPE *b_arg = b;                                                         \
  REAL_TYPE *x_arg = x;                                                         \
  REAL_TYPE *work = NULL;                                                       \
  int *iwork = NULL;                                                            \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (!ipiv || !equed || !r || !c || !rcond || !ferr || !berr || !rpvgrw) {   \
    return -1;                                                                  \
  }                                                                             \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if ((n > 0 && (!ab || !afb || !ipiv)) || (n > 0 && nrhs > 0 && (!b || !x))||\
        ldab < n || ldafb < n || ldb < nrhs || ldx < nrhs) {                  \
      return -1;                                                                \
    }                                                                           \
    ab_col = (REAL_TYPE *)malloc(ab_elems * sizeof(REAL_TYPE));                \
    afb_col = (REAL_TYPE *)calloc(afb_elems, sizeof(REAL_TYPE));               \
    b_col = (REAL_TYPE *)malloc(b_elems * sizeof(REAL_TYPE));                  \
    x_col = (REAL_TYPE *)malloc(x_elems * sizeof(REAL_TYPE));                  \
    if (!ab_col || !afb_col || !b_col || !x_col) {                             \
      free(ab_col);                                                             \
      free(afb_col);                                                            \
      free(b_col);                                                              \
      free(x_col);                                                              \
      return -101;                                                              \
    }                                                                           \
    if (fb_copy_general_band_compact_row_to_expanded_col_major(                \
            ab, n, n, kl, ku, ldab, ab_col, fact_rows, sizeof(REAL_TYPE)) != 0||\
        (fact_is_f &&                                                           \
         fb_copy_general_band_to_col_major(afb, layout, n, n, kl, kl + ku,    \
                                           ldafb, afb_col, fact_rows,          \
                                           sizeof(REAL_TYPE)) != 0) ||         \
        fb_copy_matrix_to_col_major(b, layout, n, nrhs, ldb,                  \
                                    b_col, ldb_, sizeof(REAL_TYPE)) != 0) {    \
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
  work = (REAL_TYPE *)calloc((size_t)work_len, sizeof(REAL_TYPE));             \
  iwork = (int *)calloc((size_t)iwork_len, sizeof(int));                       \
  if (!work || !iwork) {                                                        \
    free(ab_col);                                                               \
    free(afb_col);                                                              \
    free(b_col);                                                                \
    free(x_col);                                                                \
    free(work);                                                                 \
    free(iwork);                                                                \
    return -101;                                                                \
  }                                                                             \
  ((void (*)(char *, char *, int *, int *, int *, int *, REAL_TYPE *, int *,  \
             REAL_TYPE *, int *, int *, char *, REAL_TYPE *, REAL_TYPE *,     \
             REAL_TYPE *, int *, REAL_TYPE *, int *, REAL_TYPE *, REAL_TYPE *,\
             REAL_TYPE *, REAL_TYPE *, int *, int *))g_fortran_fn[OP_ID])(    \
      (char *)&fact, (char *)&trans, &n_, &kl_, &ku_, &nrhs_, ab_arg, &ldab_, \
      afb_arg, &ldafb_, ipiv, equed, r, c, b_arg, &ldb_, x_arg, &ldx_, rcond, \
      ferr, berr, work, iwork, &info);                                          \
  *rpvgrw = work[0];                                                            \
  equed_has_scaling = (*equed == 'R' || *equed == 'r' || *equed == 'C' ||     \
                       *equed == 'c' || *equed == 'B' || *equed == 'b');      \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if ((fact_is_e && equed_has_scaling &&                                      \
         fb_copy_general_band_compact_row_from_expanded_col_major(             \
             ab_col, n, n, kl, ku, fact_rows, ab, ldab, sizeof(REAL_TYPE)) != 0)||\
        ((fact_is_e || fact_is_n) &&                                            \
         fb_copy_general_band_from_col_major(afb_col, n, n, kl, kl + ku,       \
                                             fact_rows, layout, afb, ldafb,     \
                                             sizeof(REAL_TYPE)) != 0) ||       \
        (fact_is_f && equed_has_scaling &&                                      \
         fb_copy_matrix_from_col_major(b_col, n, nrhs, ldb_, layout,          \
                                       b, ldb, sizeof(REAL_TYPE)) != 0) ||     \
        fb_copy_matrix_from_col_major(x_col, n, nrhs, ldx_, layout,           \
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
    char *fact, char *trans, int *n, int *kl, int *ku, int *nrhs,             \
    REAL_TYPE *ab, int *ldab, REAL_TYPE *afb, int *ldafb, int *ipiv,          \
    char *equed, REAL_TYPE *r, REAL_TYPE *c, REAL_TYPE *b, int *ldb,          \
    REAL_TYPE *x, int *ldx, REAL_TYPE *rcond, REAL_TYPE *ferr,                \
    REAL_TYPE *berr, REAL_TYPE *work, int *iwork, int *info) {                \
  REAL_TYPE rpvgrw = (REAL_TYPE)0;                                              \
  (void)work;                                                                   \
  (void)iwork;                                                                  \
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
                     REAL_TYPE *, int, REAL_TYPE *, int, int *, char *,       \
                     REAL_TYPE *, REAL_TYPE *, REAL_TYPE *, int, REAL_TYPE *, \
                     int, REAL_TYPE *, REAL_TYPE *, REAL_TYPE *,              \
                     REAL_TYPE *))g_cblas_fn[OP_ID])(                          \
      FB_LAYOUT_COL_MAJOR, *fact, *trans, *n, *kl, *ku, *nrhs, ab, *ldab,     \
      afb, *ldafb, ipiv, equed, r, c, b, *ldb, x, *ldx, rcond, ferr, berr,    \
      &rpvgrw);                                                                 \
  if (info) *info = rc;                                                         \
}
