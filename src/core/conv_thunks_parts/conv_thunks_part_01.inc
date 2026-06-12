                                         float *work, int *info) {
  if (fb_lapack_reject_non_col_major_cblas_layout(layout, info))
    return;
  if (!g_fortran_fn[FB_OP_SSTEV]) {
    if (info)
      *info = -1;
    return;
  }
  int ldz_ = ldz;
  ((void (*)(char *, int *, float *, float *, float *, int *, float *,
             int *))g_fortran_fn[FB_OP_SSTEV])(&jobz, &n, d, e, z, &ldz_, work,
                                               info);
}

static void thunk_dstev_fortran_to_cblas(char *jobz, int *n, double *d,
                                         double *e, double *z, int *ldz,
                                         double *work, int *info) {
  if (!g_cblas_fn[FB_OP_DSTEV]) {
    if (info)
      *info = -1;
    return;
  }
  ((void (*)(int, char, int, double *, double *, double *, int, double *,
             int *))g_cblas_fn[FB_OP_DSTEV])(102, *jobz, *n, d, e, z, *ldz,
                                             work, info);
}

static void thunk_dstev_cblas_to_fortran(int layout, char jobz, int n,
                                         double *d, double *e, double *z,
                                         int ldz, double *work, int *info) {
  if (fb_lapack_reject_non_col_major_cblas_layout(layout, info))
    return;
  if (!g_fortran_fn[FB_OP_DSTEV]) {
    if (info)
      *info = -1;
    return;
  }
  int ldz_ = ldz;
  ((void (*)(char *, int *, double *, double *, double *, int *, double *,
             int *))g_fortran_fn[FB_OP_DSTEV])(&jobz, &n, d, e, z, &ldz_, work,
                                               info);
}

#define DEFINE_STEVR_REAL_THUNKS(OP, TYPE, OP_ID, QUERY_FN)                   \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const char jobz, const char range,               \
    const int n, TYPE *d, TYPE *e, const TYPE vl, const TYPE vu,               \
    const int il, const int iu, const TYPE abstol, int *m, TYPE *w, TYPE *z,  \
    const int ldz, int *isuppz) {                                              \
  char jobz_ = jobz, range_ = range;                                           \
  int n_ = n, il_ = il, iu_ = iu;                                              \
  int ldz_ = (z != NULL) ? ldz : 1;                                            \
  int lwork_ = -1, liwork_ = -1, info = 0;                                     \
  TYPE vl_ = vl, vu_ = vu, abstol_ = abstol;                                   \
  TYPE work_query = (TYPE)0;                                                   \
  int iwork_query = 0;                                                         \
  TYPE dummy_z = (TYPE)0;                                                      \
  TYPE *work = NULL;                                                           \
  TYPE *z_ptr = z ? z : &dummy_z;                                              \
  int *iwork = NULL;                                                           \
  int dummy_isuppz[2] = { 0, 0 };                                              \
  int *isuppz_ptr = isuppz ? isuppz : dummy_isuppz;                            \
  (void)layout;                                                                \
  if (!m || !w) return -1;                                                     \
  if (fb_lapack_job_requests_vectors(jobz_) && z == NULL) return -1;           \
  if (fb_lapack_job_requests_vectors(jobz_) && isuppz == NULL) return -1;      \
  if (!g_fortran_fn[OP_ID]) return -1;                                         \
  ((void (*)(char *, char *, int *, TYPE *, TYPE *, TYPE *, TYPE *, int *,     \
             int *, TYPE *, int *, TYPE *, TYPE *, int *, int *, TYPE *,       \
             int *, int *, int *, int *))g_fortran_fn[OP_ID])(                 \
      &jobz_, &range_, &n_, d, e, &vl_, &vu_, &il_, &iu_, &abstol_, m, w,      \
      z_ptr, &ldz_, isuppz_ptr, &work_query, &lwork_, &iwork_query, &liwork_,  \
      &info);                                                                   \
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
  ((void (*)(char *, char *, int *, TYPE *, TYPE *, TYPE *, TYPE *, int *,     \
             int *, TYPE *, int *, TYPE *, TYPE *, int *, int *, TYPE *,       \
             int *, int *, int *, int *))g_fortran_fn[OP_ID])(                 \
      &jobz_, &range_, &n_, d, e, &vl_, &vu_, &il_, &iu_, &abstol_, m, w,      \
      z_ptr, &ldz_, isuppz_ptr, work, &lwork_, iwork, &liwork_, &info);        \
  free(work);                                                                  \
  free(iwork);                                                                 \
  return info;                                                                 \
}                                                                              \
static void thunk_##OP##_cblas_to_fortran(                                     \
    char *jobz, char *range, int *n, TYPE *d, TYPE *e, TYPE *vl, TYPE *vu,    \
    int *il, int *iu, TYPE *abstol, int *m, TYPE *w, TYPE *z, int *ldz,       \
    int *isuppz, TYPE *work, int *lwork, int *iwork, int *liwork, int *info) {\
  TYPE dummy_z = (TYPE)0;                                                      \
  TYPE *z_ptr = z ? z : &dummy_z;                                              \
  int ldz_value = z ? *ldz : 1;                                                \
  int dummy_isuppz[2] = { 0, 0 };                                              \
  int *isuppz_ptr = isuppz ? isuppz : dummy_isuppz;                            \
  (void)work;                                                                  \
  (void)lwork;                                                                 \
  (void)iwork;                                                                 \
  (void)liwork;                                                                \
  if (fb_lapack_job_requests_vectors(*jobz) && z == NULL) {                    \
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
  int rc = ((int (*)(fb_layout_t, char, char, int, TYPE *, TYPE *, TYPE, TYPE, \
                     int, int, TYPE, int *, TYPE *, TYPE *, int, int *))       \
      g_cblas_fn[OP_ID])(FB_LAYOUT_COL_MAJOR, *jobz, *range, *n, d, e, *vl,   \
                          *vu, *il, *iu, *abstol, m, w, z_ptr, ldz_value,      \
                          isuppz_ptr);                                         \
  if (info) *info = rc;                                                        \
}

DEFINE_STEVR_REAL_THUNKS(sstevr, float, FB_OP_SSTEVR, fb_query_size_from_float)
DEFINE_STEVR_REAL_THUNKS(dstevr, double, FB_OP_DSTEVR, fb_query_size_from_double)

#undef DEFINE_STEVR_REAL_THUNKS

#define DEFINE_STEDC_REAL_THUNKS(OP, TYPE, OP_ID, QUERY_FN)                   \
static int thunk_##OP##_fortran_to_cblas(                                     \
    const fb_layout_t layout, const char compz, const int n, TYPE *d,         \
    TYPE *e, TYPE *z, const int ldz) {                                        \
  char compz_ = compz;                                                        \
  int n_ = n;                                                                 \
  int ldz_query = (z != NULL) ? ldz : 1;                                      \
  int ldz_exec = ldz_query;                                                   \
  int lwork_ = -1, liwork_ = -1, info = 0;                                    \
  int row = 0, col = 0;                                                       \
  int vectors_needed = fb_lapack_job_produces_vectors(compz_);                \
  int transpose_input = fb_lapack_job_requests_vectors(compz_);               \
  int z_dim = (n > 0) ? n : 1;                                                \
  TYPE work_query = (TYPE)0;                                                  \
  int iwork_query = 0;                                                        \
  TYPE dummy_z = (TYPE)0;                                                     \
  TYPE *work = NULL;                                                          \
  TYPE *z_query = z ? z : &dummy_z;                                           \
  TYPE *z_exec = z ? z : &dummy_z;                                            \
  int *iwork = NULL;                                                          \
  if (!d || !e) return -1;                                                    \
  if (vectors_needed && z == NULL) return -1;                                 \
  if (!g_fortran_fn[OP_ID]) return -1;                                        \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                        \
    if (ldz < n) return -7;                                                   \
    ldz_query = z_dim;                                                        \
    ldz_exec = z_dim;                                                         \
    if (!vectors_needed) {                                                    \
      z_query = &dummy_z;                                                     \
      z_exec = &dummy_z;                                                      \
    }                                                                         \
  } else if (!fb_lapack_layout_is_col_major(layout)) {                        \
    return -1;                                                                \
  } else if (!vectors_needed && z == NULL) {                                  \
    ldz_query = 1;                                                            \
    ldz_exec = 1;                                                             \
  }                                                                           \
  ((void (*)(char *, int *, TYPE *, TYPE *, TYPE *, int *, TYPE *, int *,     \
             int *, int *, int *))g_fortran_fn[OP_ID])(                       \
      &compz_, &n_, d, e, z_query, &ldz_query, &work_query, &lwork_,         \
      &iwork_query, &liwork_, &info);                                         \
  if (info != 0) return info;                                                 \
  lwork_ = QUERY_FN(work_query);                                              \
  liwork_ = fb_query_size_from_int(iwork_query);                              \
  work = (TYPE *)malloc((size_t)lwork_ * sizeof(TYPE));                       \
  iwork = (int *)malloc((size_t)liwork_ * sizeof(int));                       \
  if (!work || !iwork) {                                                      \
    free(work);                                                               \
    free(iwork);                                                              \
    return -101;                                                              \
  }                                                                           \
  if (layout == FB_LAYOUT_ROW_MAJOR && vectors_needed) {                      \
    z_exec = (TYPE *)malloc((size_t)z_dim * (size_t)z_dim * sizeof(TYPE));    \
    if (!z_exec) {                                                            \
      free(work);                                                             \
      free(iwork);                                                            \
      return -101;                                                            \
    }                                                                         \
    if (transpose_input) {                                                    \
      for (row = 0; row < n; ++row) {                                         \
        for (col = 0; col < n; ++col) {                                       \
          z_exec[(size_t)row + (size_t)col * (size_t)ldz_exec] =              \
              z[(size_t)row * (size_t)ldz + (size_t)col];                     \
        }                                                                     \
      }                                                                       \
    }                                                                         \
  }                                                                           \
  ((void (*)(char *, int *, TYPE *, TYPE *, TYPE *, int *, TYPE *, int *,     \
             int *, int *, int *))g_fortran_fn[OP_ID])(                       \
      &compz_, &n_, d, e, z_exec, &ldz_exec, work, &lwork_, iwork, &liwork_, \
      &info);                                                                 \
  if (info == 0 && layout == FB_LAYOUT_ROW_MAJOR && vectors_needed) {         \
    for (row = 0; row < n; ++row) {                                           \
      for (col = 0; col < n; ++col) {                                         \
        z[(size_t)row * (size_t)ldz + (size_t)col] =                          \
            z_exec[(size_t)row + (size_t)col * (size_t)ldz_exec];             \
      }                                                                       \
    }                                                                         \
  }                                                                           \
  free(work);                                                                 \
  free(iwork);                                                                \
  if (layout == FB_LAYOUT_ROW_MAJOR && vectors_needed) free(z_exec);          \
  return info;                                                                \
}                                                                             \
static void thunk_##OP##_cblas_to_fortran(char *compz, int *n, TYPE *d,       \
                                          TYPE *e, TYPE *z, int *ldz,         \
                                          TYPE *work, int *lwork, int *iwork, \
                                          int *liwork, int *info) {           \
  TYPE dummy_z = (TYPE)0;                                                     \
  TYPE *z_ptr = z ? z : &dummy_z;                                             \
  int ldz_value = (z != NULL && ldz != NULL) ? *ldz : 1;                      \
  (void)work;                                                                 \
  (void)lwork;                                                                \
  (void)iwork;                                                                \
  (void)liwork;                                                               \
  if (!compz || !n || !d || !e) {                                             \
    if (info) *info = -1;                                                     \
    return;                                                                   \
  }                                                                           \
  if (fb_lapack_job_produces_vectors(*compz) && (!z || !ldz)) {               \
    if (info) *info = -1;                                                     \
    return;                                                                   \
  }                                                                           \
  if (!g_cblas_fn[OP_ID]) {                                                   \
    if (info) *info = -1;                                                     \
    return;                                                                   \
  }                                                                           \
  int rc = ((int (*)(fb_layout_t, char, int, TYPE *, TYPE *, TYPE *, int))    \
                g_cblas_fn[OP_ID])(FB_LAYOUT_COL_MAJOR, *compz, *n, d, e,     \
                                   z_ptr, ldz_value);                         \
  if (info) *info = rc;                                                       \
}

DEFINE_STEDC_REAL_THUNKS(sstedc, float, FB_OP_SSTEDC, fb_query_size_from_float)
DEFINE_STEDC_REAL_THUNKS(dstedc, double, FB_OP_DSTEDC, fb_query_size_from_double)

#undef DEFINE_STEDC_REAL_THUNKS

#define DEFINE_STEDC_COMPLEX_THUNKS(OP, CTYPE, RTYPE, OP_ID, CQUERY_FN,       \
                                    RQUERY_FN)                                 \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const char compz, const int n, RTYPE *d,         \
    RTYPE *e, CTYPE *z, const int ldz) {                                        \
  char compz_ = compz;                                                          \
  int n_ = n;                                                                   \
  int ldz_query = (z != NULL) ? ldz : 1;                                        \
  int ldz_exec = ldz_query;                                                     \
  int lwork_ = -1, lrwork_ = -1, liwork_ = -1, info = 0;                       \
  int row = 0, col = 0;                                                         \
  int vectors_needed = fb_lapack_job_produces_vectors(compz_);                  \
  int transpose_input = fb_lapack_job_requests_vectors(compz_);                 \
  int z_dim = (n > 0) ? n : 1;                                                  \
  CTYPE work_query = (CTYPE)0;                                                  \
  RTYPE rwork_query = (RTYPE)0;                                                 \
  int iwork_query = 0;                                                          \
  CTYPE dummy_z = (CTYPE)0;                                                     \
  CTYPE *work = NULL;                                                           \
  CTYPE *z_query = z ? z : &dummy_z;                                            \
  CTYPE *z_exec = z ? z : &dummy_z;                                             \
  RTYPE *rwork = NULL;                                                          \
  int *iwork = NULL;                                                            \
  if (!d || !e) return -1;                                                      \
  if (vectors_needed && z == NULL) return -1;                                   \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if (ldz < n) return -7;                                                     \
    ldz_query = z_dim;                                                          \
    ldz_exec = z_dim;                                                           \
    if (!vectors_needed) {                                                      \
      z_query = &dummy_z;                                                       \
      z_exec = &dummy_z;                                                        \
    }                                                                           \
  } else if (!fb_lapack_layout_is_col_major(layout)) {                          \
    return -1;                                                                  \
  } else if (!vectors_needed && z == NULL) {                                    \
    ldz_query = 1;                                                              \
    ldz_exec = 1;                                                               \
  }                                                                             \
  ((void (*)(char *, int *, RTYPE *, RTYPE *, CTYPE *, int *, CTYPE *, int *,  \
             RTYPE *, int *, int *, int *, int *))g_fortran_fn[OP_ID])(        \
      &compz_, &n_, d, e, z_query, &ldz_query, &work_query, &lwork_,           \
      &rwork_query, &lrwork_, &iwork_query, &liwork_, &info);                  \
  if (info != 0) return info;                                                   \
  lwork_ = CQUERY_FN(work_query);                                               \
  lrwork_ = RQUERY_FN(rwork_query);                                             \
  liwork_ = fb_query_size_from_int(iwork_query);                                \
  work = (CTYPE *)malloc((size_t)lwork_ * sizeof(CTYPE));                       \
  rwork = (RTYPE *)malloc((size_t)lrwork_ * sizeof(RTYPE));                     \
  iwork = (int *)malloc((size_t)liwork_ * sizeof(int));                         \
  if (!work || !rwork || !iwork) {                                              \
    free(work);                                                                 \
    free(rwork);                                                                \
    free(iwork);                                                                \
    return -101;                                                                \
  }                                                                             \
  if (layout == FB_LAYOUT_ROW_MAJOR && vectors_needed) {                        \
    z_exec = (CTYPE *)malloc((size_t)z_dim * (size_t)z_dim * sizeof(CTYPE));    \
    if (!z_exec) {                                                              \
      free(work);                                                               \
      free(rwork);                                                              \
      free(iwork);                                                              \
      return -101;                                                              \
    }                                                                           \
    if (transpose_input) {                                                      \
      for (row = 0; row < n; ++row) {                                           \
        for (col = 0; col < n; ++col) {                                         \
          z_exec[(size_t)row + (size_t)col * (size_t)ldz_exec] =                \
              z[(size_t)row * (size_t)ldz + (size_t)col];                       \
        }                                                                       \
      }                                                                         \
    }                                                                           \
  }                                                                             \
  ((void (*)(char *, int *, RTYPE *, RTYPE *, CTYPE *, int *, CTYPE *, int *,  \
             RTYPE *, int *, int *, int *, int *))g_fortran_fn[OP_ID])(        \
      &compz_, &n_, d, e, z_exec, &ldz_exec, work, &lwork_, rwork,             \
      &lrwork_, iwork, &liwork_, &info);                                        \
  if (info == 0 && layout == FB_LAYOUT_ROW_MAJOR && vectors_needed) {           \
    for (row = 0; row < n; ++row) {                                             \
      for (col = 0; col < n; ++col) {                                           \
        z[(size_t)row * (size_t)ldz + (size_t)col] =                            \
            z_exec[(size_t)row + (size_t)col * (size_t)ldz_exec];               \
      }                                                                         \
    }                                                                           \
  }                                                                             \
  free(work);                                                                   \
  free(rwork);                                                                  \
  free(iwork);                                                                  \
  if (layout == FB_LAYOUT_ROW_MAJOR && vectors_needed) free(z_exec);            \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(char *compz, int *n, RTYPE *d,       \
                                          RTYPE *e, CTYPE *z, int *ldz,        \
                                          CTYPE *work, int *lwork,             \
                                          RTYPE *rwork, int *lrwork,           \
                                          int *iwork, int *liwork,             \
                                          int *info) {                         \
  CTYPE dummy_z = (CTYPE)0;                                                     \
  CTYPE *z_ptr = z ? z : &dummy_z;                                              \
  int ldz_value = (z != NULL && ldz != NULL) ? *ldz : 1;                        \
  (void)work;                                                                   \
  (void)lwork;                                                                  \
  (void)rwork;                                                                  \
  (void)lrwork;                                                                 \
  (void)iwork;                                                                  \
  (void)liwork;                                                                 \
  if (!compz || !n || !d || !e) {                                               \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (fb_lapack_job_produces_vectors(*compz) && (!z || !ldz)) {                 \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, char, int, RTYPE *, RTYPE *, CTYPE *, int))  \
                g_cblas_fn[OP_ID])(FB_LAYOUT_COL_MAJOR, *compz, *n, d, e,      \
                                   z_ptr, ldz_value);                           \
  if (info) *info = rc;                                                         \
}

DEFINE_STEDC_COMPLEX_THUNKS(cstedc, fb_complex_float_t, float, FB_OP_CSTEDC,
                            fb_query_size_from_cfloat,
                            fb_query_size_from_float)
DEFINE_STEDC_COMPLEX_THUNKS(zstedc, fb_complex_double_t, double, FB_OP_ZSTEDC,
                            fb_query_size_from_cdouble,
                            fb_query_size_from_double)

#undef DEFINE_STEDC_COMPLEX_THUNKS

#define DEFINE_STEGR_REAL_THUNKS(OP, TYPE, OP_ID, QUERY_FN)                   \
static int thunk_##OP##_fortran_to_cblas(                                     \
    const fb_layout_t layout, const char jobz, const char range,              \
    const int n, TYPE *d, TYPE *e, const TYPE vl, const TYPE vu,              \
    const int il, const int iu, const TYPE abstol, int *m, TYPE *w, TYPE *z, \
    const int ldz, int *isuppz) {                                             \
  char jobz_ = jobz, range_ = range;                                          \
  int n_ = n, il_ = il, iu_ = iu;                                             \
  int ldz_query = (z != NULL) ? ldz : 1;                                      \
  int ldz_exec = ldz_query;                                                   \
  int lwork_ = -1, liwork_ = -1, info = 0;                                    \
  int row = 0, col = 0;                                                       \
  int vectors_requested = fb_lapack_job_requests_vectors(jobz_);              \
  int z_dim = (n > 0) ? n : 1;                                                \
  TYPE vl_ = vl, vu_ = vu, abstol_ = abstol;                                  \
  TYPE work_query = (TYPE)0;                                                  \
  int iwork_query = 0;                                                        \
  TYPE dummy_z = (TYPE)0;                                                     \
  TYPE *work = NULL;                                                          \
  TYPE *z_query = z ? z : &dummy_z;                                           \
  TYPE *z_exec = z ? z : &dummy_z;                                            \
  int *iwork = NULL;                                                          \
  int dummy_isuppz[2] = { 0, 0 };                                             \
  int *isuppz_ptr = isuppz ? isuppz : dummy_isuppz;                           \
  if (!m || !w) return -1;                                                    \
  if (vectors_requested && z == NULL) return -1;                              \
  if (vectors_requested && isuppz == NULL) return -1;                         \
  if (!g_fortran_fn[OP_ID]) return -1;                                        \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                        \
    if ((vectors_requested && ldz < z_dim) || ldz < 1) return -15;           \
    ldz_query = z_dim;                                                        \
    ldz_exec = z_dim;                                                         \
    if (!vectors_requested) {                                                 \
      z_query = &dummy_z;                                                     \
      z_exec = &dummy_z;                                                      \
    }                                                                         \
  } else if (!fb_lapack_layout_is_col_major(layout)) {                        \
    return -1;                                                                \
  } else if (!vectors_requested && z == NULL) {                               \
    ldz_query = 1;                                                            \
    ldz_exec = 1;                                                             \
  }                                                                           \
  ((void (*)(char *, char *, int *, TYPE *, TYPE *, TYPE *, TYPE *, int *,    \
             int *, TYPE *, int *, TYPE *, TYPE *, int *, int *, TYPE *,      \
             int *, int *, int *, int *))g_fortran_fn[OP_ID])(                \
      &jobz_, &range_, &n_, d, e, &vl_, &vu_, &il_, &iu_, &abstol_, m, w,     \
      z_query, &ldz_query, isuppz_ptr, &work_query, &lwork_, &iwork_query,    \
      &liwork_, &info);                                                       \
  if (info != 0) return info;                                                 \
  lwork_ = QUERY_FN(work_query);                                              \
  liwork_ = fb_query_size_from_int(iwork_query);                              \
  work = (TYPE *)malloc((size_t)lwork_ * sizeof(TYPE));                       \
  iwork = (int *)malloc((size_t)liwork_ * sizeof(int));                       \
  if (!work || !iwork) {                                                      \
    free(work);                                                               \
    free(iwork);                                                              \
    return -101;                                                              \
  }                                                                           \
  if (layout == FB_LAYOUT_ROW_MAJOR && vectors_requested) {                   \
    z_exec = (TYPE *)malloc((size_t)z_dim * (size_t)z_dim * sizeof(TYPE));    \
    if (!z_exec) {                                                            \
      free(work);                                                             \
      free(iwork);                                                            \
      return -101;                                                            \
    }                                                                         \
  }                                                                           \
  ((void (*)(char *, char *, int *, TYPE *, TYPE *, TYPE *, TYPE *, int *,    \
             int *, TYPE *, int *, TYPE *, TYPE *, int *, int *, TYPE *,      \
             int *, int *, int *, int *))g_fortran_fn[OP_ID])(                \
      &jobz_, &range_, &n_, d, e, &vl_, &vu_, &il_, &iu_, &abstol_, m, w,     \
      z_exec, &ldz_exec, isuppz_ptr, work, &lwork_, iwork, &liwork_, &info);  \
  if (info == 0 && layout == FB_LAYOUT_ROW_MAJOR && vectors_requested) {      \
    for (row = 0; row < n; ++row) {                                           \
      for (col = 0; col < *m; ++col) {                                        \
        z[(size_t)row * (size_t)ldz + (size_t)col] =                          \
            z_exec[(size_t)row + (size_t)col * (size_t)ldz_exec];             \
      }                                                                       \
    }                                                                         \
  }                                                                           \
  free(work);                                                                 \
  free(iwork);                                                                \
  if (layout == FB_LAYOUT_ROW_MAJOR && vectors_requested) free(z_exec);       \
  return info;                                                                \
}                                                                             \
static void thunk_##OP##_cblas_to_fortran(                                    \
    char *jobz, char *range, int *n, TYPE *d, TYPE *e, TYPE *vl, TYPE *vu,   \
    int *il, int *iu, TYPE *abstol, int *m, TYPE *w, TYPE *z, int *ldz,      \
    int *isuppz, TYPE *work, int *lwork, int *iwork, int *liwork, int *info) {\
  TYPE dummy_z = (TYPE)0;                                                     \
  TYPE *z_ptr = z ? z : &dummy_z;                                             \
  int ldz_value = z ? *ldz : 1;                                               \
  int dummy_isuppz[2] = { 0, 0 };                                             \
  int *isuppz_ptr = isuppz ? isuppz : dummy_isuppz;                           \
  (void)work;                                                                 \
  (void)lwork;                                                                \
  (void)iwork;                                                                \
  (void)liwork;                                                               \
  if (fb_lapack_job_requests_vectors(*jobz) && z == NULL) {                   \
    if (info) *info = -1;                                                     \
    return;                                                                   \
  }                                                                           \
  if (fb_lapack_job_requests_vectors(*jobz) && isuppz == NULL) {              \
    if (info) *info = -1;                                                     \
    return;                                                                   \
  }                                                                           \
  if (!g_cblas_fn[OP_ID]) {                                                   \
    if (info) *info = -1;                                                     \
    return;                                                                   \
  }                                                                           \
  int rc = ((int (*)(fb_layout_t, char, char, int, TYPE *, TYPE *, TYPE,      \
                     TYPE, int, int, TYPE, int *, TYPE *, TYPE *, int,       \
                     int *))g_cblas_fn[OP_ID])(                               \
      FB_LAYOUT_COL_MAJOR, *jobz, *range, *n, d, e, *vl, *vu, *il, *iu,      \
      *abstol, m, w, z_ptr, ldz_value, isuppz_ptr);                           \
  if (info) *info = rc;                                                       \
}

DEFINE_STEGR_REAL_THUNKS(sstegr, float, FB_OP_SSTEGR, fb_query_size_from_float)
DEFINE_STEGR_REAL_THUNKS(dstegr, double, FB_OP_DSTEGR, fb_query_size_from_double)

#undef DEFINE_STEGR_REAL_THUNKS

#define DEFINE_STEGR_COMPLEX_THUNKS(OP, CTYPE, RTYPE, OP_ID, QUERY_FN)        \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const char jobz, const char range,               \
    const int n, RTYPE *d, RTYPE *e, const RTYPE vl, const RTYPE vu,           \
    const int il, const int iu, const RTYPE abstol, int *m, RTYPE *w,          \
    CTYPE *z, const int ldz, int *isuppz) {                                     \
  char jobz_ = jobz, range_ = range;                                            \
  int n_ = n, il_ = il, iu_ = iu;                                               \
  int ldz_query = (z != NULL) ? ldz : 1;                                        \
  int ldz_exec = ldz_query;                                                     \
  int lwork_ = -1, liwork_ = -1, info = 0;                                      \
  int row = 0, col = 0;                                                         \
  int vectors_requested = fb_lapack_job_requests_vectors(jobz_);                \
  int z_dim = (n > 0) ? n : 1;                                                  \
  RTYPE vl_ = vl, vu_ = vu, abstol_ = abstol;                                   \
  RTYPE work_query = (RTYPE)0;                                                  \
  int iwork_query = 0;                                                          \
  CTYPE dummy_z = (CTYPE)0;                                                     \
  RTYPE *work = NULL;                                                           \
  CTYPE *z_query = z ? z : &dummy_z;                                            \
  CTYPE *z_exec = z ? z : &dummy_z;                                             \
  int *iwork = NULL;                                                            \
  int dummy_isuppz[2] = { 0, 0 };                                               \
  int *isuppz_ptr = isuppz ? isuppz : dummy_isuppz;                             \
  if (!m || !w) return -1;                                                      \
  if (vectors_requested && z == NULL) return -1;                                \
  if (vectors_requested && isuppz == NULL) return -1;                           \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                          \
    if ((vectors_requested && ldz < z_dim) || ldz < 1) return -15;             \
    ldz_query = z_dim;                                                          \
    ldz_exec = z_dim;                                                           \
    if (!vectors_requested) {                                                   \
      z_query = &dummy_z;                                                       \
      z_exec = &dummy_z;                                                        \
    }                                                                           \
  } else if (!fb_lapack_layout_is_col_major(layout)) {                          \
    return -1;                                                                  \
  } else if (!vectors_requested && z == NULL) {                                 \
    ldz_query = 1;                                                              \
    ldz_exec = 1;                                                               \
  }                                                                             \
  ((void (*)(char *, char *, int *, RTYPE *, RTYPE *, RTYPE *, RTYPE *, int *, \
             int *, RTYPE *, int *, RTYPE *, CTYPE *, int *, int *, RTYPE *,   \
             int *, int *, int *, int *))g_fortran_fn[OP_ID])(                 \
      &jobz_, &range_, &n_, d, e, &vl_, &vu_, &il_, &iu_, &abstol_, m, w,      \
      z_query, &ldz_query, isuppz_ptr, &work_query, &lwork_, &iwork_query,     \
      &liwork_, &info);                                                         \
  if (info != 0) return info;                                                   \
  lwork_ = QUERY_FN(work_query);                                                \
  liwork_ = fb_query_size_from_int(iwork_query);                                \
  work = (RTYPE *)malloc((size_t)lwork_ * sizeof(RTYPE));                       \
  iwork = (int *)malloc((size_t)liwork_ * sizeof(int));                         \
  if (!work || !iwork) {                                                        \
    free(work);                                                                 \
    free(iwork);                                                                \
    return -101;                                                                \
  }                                                                             \
  if (layout == FB_LAYOUT_ROW_MAJOR && vectors_requested) {                     \
    z_exec = (CTYPE *)malloc((size_t)z_dim * (size_t)z_dim * sizeof(CTYPE));    \
    if (!z_exec) {                                                              \
      free(work);                                                               \
      free(iwork);                                                              \
      return -101;                                                              \
    }                                                                           \
  }                                                                             \
  ((void (*)(char *, char *, int *, RTYPE *, RTYPE *, RTYPE *, RTYPE *, int *, \
             int *, RTYPE *, int *, RTYPE *, CTYPE *, int *, int *, RTYPE *,   \
             int *, int *, int *, int *))g_fortran_fn[OP_ID])(                 \
      &jobz_, &range_, &n_, d, e, &vl_, &vu_, &il_, &iu_, &abstol_, m, w,      \
      z_exec, &ldz_exec, isuppz_ptr, work, &lwork_, iwork, &liwork_, &info);   \
  if (info == 0 && layout == FB_LAYOUT_ROW_MAJOR && vectors_requested) {       \
    for (row = 0; row < n; ++row) {                                             \
      for (col = 0; col < *m; ++col) {                                          \
        z[(size_t)row * (size_t)ldz + (size_t)col] =                            \
            z_exec[(size_t)row + (size_t)col * (size_t)ldz_exec];               \
      }                                                                         \
    }                                                                           \
  }                                                                             \
  free(work);                                                                   \
  free(iwork);                                                                  \
  if (layout == FB_LAYOUT_ROW_MAJOR && vectors_requested) free(z_exec);         \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    char *jobz, char *range, int *n, RTYPE *d, RTYPE *e, RTYPE *vl,            \
    RTYPE *vu, int *il, int *iu, RTYPE *abstol, int *m, RTYPE *w,              \
    CTYPE *z, int *ldz, int *isuppz, RTYPE *work, int *lwork, int *iwork,      \
    int *liwork, int *info) {                                                   \
  CTYPE dummy_z = (CTYPE)0;                                                     \
  CTYPE *z_ptr = z ? z : &dummy_z;                                              \
  int ldz_value = z ? *ldz : 1;                                                 \
  int dummy_isuppz[2] = { 0, 0 };                                               \
  int *isuppz_ptr = isuppz ? isuppz : dummy_isuppz;                             \
  (void)work;                                                                   \
  (void)lwork;                                                                  \
  (void)iwork;                                                                  \
  (void)liwork;                                                                 \
  if (fb_lapack_job_requests_vectors(*jobz) && z == NULL) {                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (fb_lapack_job_requests_vectors(*jobz) && isuppz == NULL) {                \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, char, char, int, RTYPE *, RTYPE *, RTYPE,    \
                     RTYPE, int, int, RTYPE, int *, RTYPE *, CTYPE *, int,     \
                     int *))g_cblas_fn[OP_ID])(                                 \
      FB_LAYOUT_COL_MAJOR, *jobz, *range, *n, d, e, *vl, *vu, *il, *iu,        \
      *abstol, m, w, z_ptr, ldz_value, isuppz_ptr);                             \
  if (info) *info = rc;                                                         \
}

DEFINE_STEGR_COMPLEX_THUNKS(cstegr, fb_complex_float_t, float, FB_OP_CSTEGR,
                            fb_query_size_from_float)
DEFINE_STEGR_COMPLEX_THUNKS(zstegr, fb_complex_double_t, double, FB_OP_ZSTEGR,
                            fb_query_size_from_double)

#undef DEFINE_STEGR_COMPLEX_THUNKS

#define DEFINE_STEVX_REAL_THUNKS(OP, TYPE, OP_ID)                             \
static int thunk_##OP##_fortran_to_cblas(                                     \
    const fb_layout_t layout, const char jobz, const char range,              \
    const int n, TYPE *d, TYPE *e, const TYPE vl, const TYPE vu,              \
    const int il, const int iu, const TYPE abstol, int *m, TYPE *w, TYPE *z, \
    const int ldz, int *ifail) {                                              \
  char jobz_ = jobz, range_ = range;                                          \
  int n_ = n, il_ = il, iu_ = iu;                                             \
  int ldz_ = (z != NULL) ? ldz : 1;                                           \
  int work_size = (n > 0) ? (5 * n) : 1;                                      \
  TYPE vl_ = vl, vu_ = vu, abstol_ = abstol;                                  \
  TYPE dummy_z = (TYPE)0;                                                     \
  TYPE *work = NULL;                                                          \
  TYPE *z_ptr = z ? z : &dummy_z;                                             \
  int *iwork = NULL;                                                          \
  int dummy_ifail = 0;                                                        \
  int *ifail_ptr = ifail ? ifail : &dummy_ifail;                              \
  int info = 0;                                                               \
  (void)layout;                                                               \
  if (!m || !w) return -1;                                                    \
  if (fb_lapack_job_requests_vectors(jobz_) && z == NULL) return -1;          \
  if (fb_lapack_job_requests_vectors(jobz_) && ifail == NULL) return -1;      \
  if (!g_fortran_fn[OP_ID]) return -1;                                        \
  work = (TYPE *)malloc((size_t)work_size * sizeof(TYPE));                    \
  iwork = (int *)malloc((size_t)work_size * sizeof(int));                     \
  if (!work || !iwork) {                                                      \
    free(work);                                                               \
    free(iwork);                                                              \
    return -101;                                                              \
  }                                                                           \
  ((void (*)(char *, char *, int *, TYPE *, TYPE *, TYPE *, TYPE *, int *,    \
             int *, TYPE *, int *, TYPE *, TYPE *, int *, TYPE *, int *,      \
             int *, int *))g_fortran_fn[OP_ID])(                              \
      &jobz_, &range_, &n_, d, e, &vl_, &vu_, &il_, &iu_, &abstol_, m, w,     \
      z_ptr, &ldz_, work, iwork, ifail_ptr, &info);                           \
  free(work);                                                                 \
  free(iwork);                                                                \
  return info;                                                                \
}                                                                             \
static void thunk_##OP##_cblas_to_fortran(                                    \
    char *jobz, char *range, int *n, TYPE *d, TYPE *e, TYPE *vl, TYPE *vu,   \
    int *il, int *iu, TYPE *abstol, int *m, TYPE *w, TYPE *z, int *ldz,      \
    TYPE *work, int *iwork, int *ifail, int *info) {                          \
  TYPE dummy_z = (TYPE)0;                                                     \
  TYPE *z_ptr = z ? z : &dummy_z;                                             \
  int ldz_value = z ? *ldz : 1;                                               \
  int dummy_ifail = 0;                                                        \
  int *ifail_ptr = ifail ? ifail : &dummy_ifail;                              \
  (void)work;                                                                 \
  (void)iwork;                                                                \
  if (fb_lapack_job_requests_vectors(*jobz) && z == NULL) {                   \
    if (info) *info = -1;                                                     \
    return;                                                                   \
  }                                                                           \
  if (fb_lapack_job_requests_vectors(*jobz) && ifail == NULL) {               \
    if (info) *info = -1;                                                     \
    return;                                                                   \
  }                                                                           \
  if (!g_cblas_fn[OP_ID]) {                                                   \
    if (info) *info = -1;                                                     \
    return;                                                                   \
  }                                                                           \
  int rc = ((int (*)(fb_layout_t, char, char, int, TYPE *, TYPE *, TYPE,      \
                     TYPE, int, int, TYPE, int *, TYPE *, TYPE *, int,        \
                     int *))g_cblas_fn[OP_ID])(                               \
      FB_LAYOUT_COL_MAJOR, *jobz, *range, *n, d, e, *vl, *vu, *il, *iu,      \
      *abstol, m, w, z_ptr, ldz_value, ifail_ptr);                            \
  if (info) *info = rc;                                                       \
}

DEFINE_STEVX_REAL_THUNKS(sstevx, float, FB_OP_SSTEVX)
DEFINE_STEVX_REAL_THUNKS(dstevx, double, FB_OP_DSTEVX)

#undef DEFINE_STEVX_REAL_THUNKS

#define DEFINE_STEBZ_REAL_THUNKS(OP, TYPE, OP_ID)                             \
static int thunk_##OP##_fortran_to_cblas(                                     \
    const char range, const char order, const int n, const TYPE vl,           \
    const TYPE vu, const int il, const int iu, const TYPE abstol,             \
    const TYPE *d, const TYPE *e, int *m, int *nsplit, TYPE *w, int *iblock, \
    int *isplit) {                                                             \
  char range_ = range, order_ = order;                                        \
  int n_ = n, il_ = il, iu_ = iu;                                             \
  int work_size = (n > 0) ? (4 * n) : 1;                                      \
  int iwork_size = (n > 0) ? (3 * n) : 1;                                     \
  TYPE vl_ = vl, vu_ = vu, abstol_ = abstol;                                  \
  TYPE *work = NULL;                                                          \
  int *iwork = NULL;                                                          \
  int info = 0;                                                               \
  if (!m || !nsplit || !w || !iblock || !isplit) return -1;                  \
  if (!g_fortran_fn[OP_ID]) return -1;                                        \
  work = (TYPE *)malloc((size_t)work_size * sizeof(TYPE));                    \
  iwork = (int *)malloc((size_t)iwork_size * sizeof(int));                    \
  if (!work || !iwork) {                                                      \
    free(work);                                                               \
    free(iwork);                                                              \
    return -101;                                                              \
  }                                                                           \
  ((void (*)(char *, char *, int *, TYPE *, TYPE *, int *, int *, TYPE *,     \
             TYPE *, TYPE *, int *, int *, TYPE *, int *, int *, TYPE *,      \
             int *, int *))g_fortran_fn[OP_ID])(                              \
      &range_, &order_, &n_, &vl_, &vu_, &il_, &iu_, &abstol_, (TYPE *)d,     \
      (TYPE *)e, m, nsplit, w, iblock, isplit, work, iwork, &info);           \
  free(work);                                                                 \
  free(iwork);                                                                \
  return info;                                                                \
}                                                                             \
static void thunk_##OP##_cblas_to_fortran(                                    \
    char *range, char *order, int *n, TYPE *vl, TYPE *vu, int *il, int *iu,  \
    TYPE *abstol, TYPE *d, TYPE *e, int *m, int *nsplit, TYPE *w,             \
    int *iblock, int *isplit, TYPE *work, int *iwork, int *info) {            \
  (void)work;                                                                 \
  (void)iwork;                                                                \
  if (!g_cblas_fn[OP_ID]) {                                                   \
    if (info) *info = -1;                                                     \
    return;                                                                   \
  }                                                                           \
  int rc = ((int (*)(char, char, int, TYPE, TYPE, int, int, TYPE,             \
                     const TYPE *, const TYPE *, int *, int *, TYPE *,        \
                     int *, int *))g_cblas_fn[OP_ID])(                        \
      *range, *order, *n, *vl, *vu, *il, *iu, *abstol, d, e, m, nsplit, w,   \
      iblock, isplit);                                                        \
  if (info) *info = rc;                                                       \
}

DEFINE_STEBZ_REAL_THUNKS(sstebz, float, FB_OP_SSTEBZ)
DEFINE_STEBZ_REAL_THUNKS(dstebz, double, FB_OP_DSTEBZ)

#undef DEFINE_STEBZ_REAL_THUNKS

#define DEFINE_STEIN_REAL_THUNKS(OP, TYPE, OP_ID)                             \
static int thunk_##OP##_fortran_to_cblas(                                     \
    const fb_layout_t layout, const int n, const TYPE *d, const TYPE *e,      \
    const int m, const TYPE *w, const int *iblock, const int *isplit,         \
    TYPE *z, const int ldz, int *ifail) {                                     \
  int work_size = (n > 0) ? (5 * n) : 1;                                      \
  int iwork_size = (n > 0) ? n : 1;                                           \
  int z_rows = (n > 0) ? n : 1;                                               \
  int z_cols = (m > 0) ? m : 1;                                               \
  int ldz_col = z_rows;                                                       \
  TYPE *work = NULL;                                                          \
  TYPE *z_col = z;                                                            \
  int *iwork = NULL;                                                          \
  int info = 0;                                                               \
  int row = 0;                                                                \
  int col = 0;                                                                \
  if (!d || !e || !w || !iblock || !isplit || !z || !ifail) return -1;       \
  if (!g_fortran_fn[OP_ID]) return -1;                                        \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                        \
    if (ldz < z_cols) return -10;                                             \
    z_col = (TYPE *)malloc((size_t)z_rows * (size_t)z_cols * sizeof(TYPE));   \
    if (!z_col) return -101;                                                  \
  } else if (layout != FB_LAYOUT_COL_MAJOR) {                                 \
    return -1;                                                                \
  }                                                                           \
  work = (TYPE *)malloc((size_t)work_size * sizeof(TYPE));                    \
  iwork = (int *)malloc((size_t)iwork_size * sizeof(int));                    \
  if (!work || !iwork) {                                                      \
    free(work);                                                               \
    free(iwork);                                                              \
    if (layout == FB_LAYOUT_ROW_MAJOR) free(z_col);                           \
    return -101;                                                              \
  }                                                                           \
  ((void (*)(int *, TYPE *, TYPE *, int *, TYPE *, int *, int *, TYPE *,      \
             int *, TYPE *, int *, int *, int *))g_fortran_fn[OP_ID])(        \
      (int *)&n, (TYPE *)d, (TYPE *)e, (int *)&m, (TYPE *)w, (int *)iblock,   \
      (int *)isplit, z_col, &ldz_col, work, iwork, ifail, &info);             \
  if (info == 0 && layout == FB_LAYOUT_ROW_MAJOR) {                           \
    for (row = 0; row < n; ++row) {                                           \
      for (col = 0; col < m; ++col) {                                         \
        z[(size_t)row * (size_t)ldz + (size_t)col] =                          \
            z_col[(size_t)row + (size_t)col * (size_t)ldz_col];               \
      }                                                                       \
    }                                                                         \
  }                                                                           \
  free(work);                                                                 \
  free(iwork);                                                                \
  if (layout == FB_LAYOUT_ROW_MAJOR) free(z_col);                             \
  return info;                                                                \
}                                                                             \
static void thunk_##OP##_cblas_to_fortran(                                    \
    int *n, TYPE *d, TYPE *e, int *m, TYPE *w, int *iblock, int *isplit,     \
    TYPE *z, int *ldz, TYPE *work, int *iwork, int *ifail, int *info) {      \
  (void)work;                                                                 \
  (void)iwork;                                                                \
  if (!d || !e || !w || !iblock || !isplit || !z || !ifail) {                \
    if (info) *info = -1;                                                     \
    return;                                                                   \
  }                                                                           \
  if (!g_cblas_fn[OP_ID]) {                                                   \
    if (info) *info = -1;                                                     \
    return;                                                                   \
  }                                                                           \
  int rc = ((int (*)(fb_layout_t, int, const TYPE *, const TYPE *, int,       \
                     const TYPE *, const int *, const int *, TYPE *, int,     \
                     int *))g_cblas_fn[OP_ID])(                               \
      FB_LAYOUT_COL_MAJOR, *n, d, e, *m, w, iblock, isplit, z, *ldz, ifail); \
  if (info) *info = rc;                                                       \
}

DEFINE_STEIN_REAL_THUNKS(sstein, float, FB_OP_SSTEIN)
DEFINE_STEIN_REAL_THUNKS(dstein, double, FB_OP_DSTEIN)

#undef DEFINE_STEIN_REAL_THUNKS

#define DEFINE_STEIN_COMPLEX_THUNKS(OP, CTYPE, RTYPE, OP_ID)                  \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const int n, const RTYPE *d, const RTYPE *e,    \
    const int m, const RTYPE *w, const int *iblock, const int *isplit,        \
    CTYPE *z, const int ldz, int *ifail) {                                     \
  int work_size = (n > 0) ? (5 * n) : 1;                                       \
  int iwork_size = (n > 0) ? n : 1;                                            \
  int z_rows = (n > 0) ? n : 1;                                                \
  int z_cols = (m > 0) ? m : 1;                                                \
  int ldz_col = z_rows;                                                        \
  RTYPE *work = NULL;                                                          \
  CTYPE *z_col = z;                                                            \
  int *iwork = NULL;                                                           \
  int info = 0;                                                                \
  int row = 0;                                                                 \
  int col = 0;                                                                 \
  if (!d || !e || !w || !iblock || !isplit || !z || !ifail) return -1;        \
  if (!g_fortran_fn[OP_ID]) return -1;                                         \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                         \
    if (ldz < z_cols) return -10;                                              \
    z_col = (CTYPE *)malloc((size_t)z_rows * (size_t)z_cols * sizeof(CTYPE));  \
    if (!z_col) return -101;                                                   \
  } else if (layout != FB_LAYOUT_COL_MAJOR) {                                  \
    return -1;                                                                 \
  }                                                                            \
  work = (RTYPE *)malloc((size_t)work_size * sizeof(RTYPE));                   \
  iwork = (int *)malloc((size_t)iwork_size * sizeof(int));                     \
  if (!work || !iwork) {                                                       \
    free(work);                                                                \
    free(iwork);                                                               \
    if (layout == FB_LAYOUT_ROW_MAJOR) free(z_col);                            \
    return -101;                                                               \
  }                                                                            \
  ((void (*)(int *, RTYPE *, RTYPE *, int *, RTYPE *, int *, int *, CTYPE *,  \
             int *, RTYPE *, int *, int *, int *))g_fortran_fn[OP_ID])(       \
      (int *)&n, (RTYPE *)d, (RTYPE *)e, (int *)&m, (RTYPE *)w, (int *)iblock,\
      (int *)isplit, z_col, &ldz_col, work, iwork, ifail, &info);              \
  if (info == 0 && layout == FB_LAYOUT_ROW_MAJOR) {                            \
    for (row = 0; row < n; ++row) {                                            \
      for (col = 0; col < m; ++col) {                                          \
        z[(size_t)row * (size_t)ldz + (size_t)col] =                           \
            z_col[(size_t)row + (size_t)col * (size_t)ldz_col];                \
      }                                                                        \
    }                                                                          \
  }                                                                            \
  free(work);                                                                  \
  free(iwork);                                                                 \
  if (layout == FB_LAYOUT_ROW_MAJOR) free(z_col);                              \
  return info;                                                                 \
}                                                                              \
static void thunk_##OP##_cblas_to_fortran(                                     \
    int *n, RTYPE *d, RTYPE *e, int *m, RTYPE *w, int *iblock, int *isplit,   \
    CTYPE *z, int *ldz, RTYPE *work, int *iwork, int *ifail, int *info) {     \
  (void)work;                                                                  \
  (void)iwork;                                                                 \
  if (!d || !e || !w || !iblock || !isplit || !z || !ifail) {                 \
    if (info) *info = -1;                                                      \
    return;                                                                    \
  }                                                                            \
  if (!g_cblas_fn[OP_ID]) {                                                    \
    if (info) *info = -1;                                                      \
    return;                                                                    \
  }                                                                            \
  int rc = ((int (*)(fb_layout_t, int, const RTYPE *, const RTYPE *, int,     \
                     const RTYPE *, const int *, const int *, CTYPE *, int,    \
                     int *))g_cblas_fn[OP_ID])(                                \
      FB_LAYOUT_COL_MAJOR, *n, d, e, *m, w, iblock, isplit, z, *ldz, ifail);  \
  if (info) *info = rc;                                                        \
}

DEFINE_STEIN_COMPLEX_THUNKS(cstein, fb_complex_float_t, float, FB_OP_CSTEIN)
DEFINE_STEIN_COMPLEX_THUNKS(zstein, fb_complex_double_t, double, FB_OP_ZSTEIN)

#undef DEFINE_STEIN_COMPLEX_THUNKS

#define DEFINE_STEQR_REAL_THUNKS(OP, TYPE, OP_ID)                            \
static int thunk_##OP##_fortran_to_cblas(                                    \
    const fb_layout_t layout, const char compz, const int n, TYPE *d,        \
    TYPE *e, TYPE *z, const int ldz) {                                       \
  char compz_ = compz;                                                       \
  int n_ = n;                                                                \
  int work_size =                                                            \
      fb_lapack_job_is_none(compz_) ? 1 : ((n > 0) ? (2 * n - 2) : 1);      \
  int vectors_needed = fb_lapack_job_produces_vectors(compz_);               \
  int transpose_input = fb_lapack_job_requests_vectors(compz_);              \
  int z_dim = (n > 0) ? n : 1;                                               \
  int ldz_col = z_dim;                                                       \
  TYPE dummy_z = (TYPE)0;                                                    \
  TYPE *work = NULL;                                                         \
  TYPE *z_col = z ? z : &dummy_z;                                            \
  int info = 0;                                                              \
  int row = 0;                                                               \
  int col = 0;                                                               \
  if (!d || !e) return -1;                                                   \
  if (vectors_needed && z == NULL) return -1;                                \
  if (!g_fortran_fn[OP_ID]) return -1;                                       \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                       \
    if (ldz < n) return -7;                                                  \
    if (vectors_needed) {                                                    \
      z_col = (TYPE *)malloc((size_t)z_dim * (size_t)z_dim * sizeof(TYPE));  \
      if (!z_col) return -101;                                               \
      if (transpose_input) {                                                 \
        for (row = 0; row < n; ++row) {                                      \
          for (col = 0; col < n; ++col) {                                    \
            z_col[(size_t)row + (size_t)col * (size_t)ldz_col] =             \
                z[(size_t)row * (size_t)ldz + (size_t)col];                  \
          }                                                                  \
        }                                                                    \
      }                                                                      \
    } else {                                                                 \
      z_col = &dummy_z;                                                      \
      ldz_col = 1;                                                           \
    }                                                                        \
  } else if (!fb_lapack_layout_is_col_major(layout)) {                       \
    return -1;                                                               \
  } else if (!vectors_needed && z == NULL) {                                 \
    ldz_col = 1;                                                             \
  } else {                                                                   \
    ldz_col = ldz;                                                           \
  }                                                                          \
  work = (TYPE *)malloc((size_t)work_size * sizeof(TYPE));                   \
  if (!work) {                                                               \
    if (layout == FB_LAYOUT_ROW_MAJOR && vectors_needed) free(z_col);        \
    return -101;                                                             \
  }                                                                          \
  ((void (*)(char *, int *, TYPE *, TYPE *, TYPE *, int *, TYPE *, int *))   \
       g_fortran_fn[OP_ID])(&compz_, &n_, d, e, z_col, &ldz_col, work,       \
                            &info);                                          \
  if (info == 0 && layout == FB_LAYOUT_ROW_MAJOR && vectors_needed) {        \
    for (row = 0; row < n; ++row) {                                          \
      for (col = 0; col < n; ++col) {                                        \
        z[(size_t)row * (size_t)ldz + (size_t)col] =                         \
            z_col[(size_t)row + (size_t)col * (size_t)ldz_col];              \
      }                                                                      \
    }                                                                        \
  }                                                                          \
  free(work);                                                                \
  if (layout == FB_LAYOUT_ROW_MAJOR && vectors_needed) free(z_col);          \
  return info;                                                               \
}                                                                            \
static void thunk_##OP##_cblas_to_fortran(char *compz, int *n, TYPE *d,      \
                                          TYPE *e, TYPE *z, int *ldz,        \
                                          TYPE *work, int *info) {           \
  TYPE dummy_z = (TYPE)0;                                                    \
  TYPE *z_ptr = z ? z : &dummy_z;                                            \
  int vectors_needed = compz ? fb_lapack_job_produces_vectors(*compz) : 0;   \
  int ldz_value = (z != NULL && ldz != NULL) ? *ldz : 1;                     \
  (void)work;                                                                \
  if (!compz || !n || !d || !e) {                                            \
    if (info) *info = -1;                                                    \
    return;                                                                  \
  }                                                                          \
  if (vectors_needed && (!z || !ldz)) {                                      \
    if (info) *info = -1;                                                    \
    return;                                                                  \
  }                                                                          \
  if (!g_cblas_fn[OP_ID]) {                                                  \
    if (info) *info = -1;                                                    \
    return;                                                                  \
  }                                                                          \
  int rc = ((int (*)(fb_layout_t, char, int, TYPE *, TYPE *, TYPE *, int))   \
                g_cblas_fn[OP_ID])(FB_LAYOUT_COL_MAJOR, *compz, *n, d, e,    \
                                   z_ptr, ldz_value);                        \
  if (info) *info = rc;                                                      \
}

DEFINE_STEQR_REAL_THUNKS(ssteqr, float, FB_OP_SSTEQR)
DEFINE_STEQR_REAL_THUNKS(dsteqr, double, FB_OP_DSTEQR)
DEFINE_STEQR_REAL_THUNKS(spteqr, float, FB_OP_SPTEQR)
DEFINE_STEQR_REAL_THUNKS(dpteqr, double, FB_OP_DPTEQR)

#undef DEFINE_STEQR_REAL_THUNKS

#define DEFINE_STEQR_COMPLEX_THUNKS(OP, CTYPE, RTYPE, OP_ID)                  \
static int thunk_##OP##_fortran_to_cblas(                                     \
    const fb_layout_t layout, const char compz, const int n, RTYPE *d,        \
    RTYPE *e, CTYPE *z, const int ldz) {                                      \
  char compz_ = compz;                                                        \
  int n_ = n;                                                                 \
  int work_size =                                                             \
      fb_lapack_job_is_none(compz_) ? 1 : ((n > 0) ? (2 * n - 2) : 1);       \
  int vectors_needed = fb_lapack_job_produces_vectors(compz_);                \
  int transpose_input = fb_lapack_job_requests_vectors(compz_);               \
  int z_dim = (n > 0) ? n : 1;                                                \
  int ldz_col = z_dim;                                                        \
  CTYPE dummy_z = (CTYPE)0;                                                   \
  CTYPE *z_col = z ? z : &dummy_z;                                            \
  RTYPE *work = NULL;                                                         \
  int info = 0;                                                               \
  int row = 0;                                                                \
  int col = 0;                                                                \
  if (!d || !e) return -1;                                                    \
  if (vectors_needed && z == NULL) return -1;                                 \
  if (!g_fortran_fn[OP_ID]) return -1;                                        \
  if (layout == FB_LAYOUT_ROW_MAJOR) {                                        \
    if (ldz < n) return -7;                                                   \
    if (vectors_needed) {                                                     \
      z_col = (CTYPE *)malloc((size_t)z_dim * (size_t)z_dim * sizeof(CTYPE)); \
      if (!z_col) return -101;                                                \
      if (transpose_input) {                                                  \
        for (row = 0; row < n; ++row) {                                       \
          for (col = 0; col < n; ++col) {                                     \
            z_col[(size_t)row + (size_t)col * (size_t)ldz_col] =              \
                z[(size_t)row * (size_t)ldz + (size_t)col];                   \
          }                                                                   \
        }                                                                     \
      }                                                                       \
    } else {                                                                  \
      z_col = &dummy_z;                                                       \
      ldz_col = 1;                                                            \
    }                                                                         \
  } else if (!fb_lapack_layout_is_col_major(layout)) {                        \
    return -1;                                                                \
  } else if (!vectors_needed && z == NULL) {                                  \
    ldz_col = 1;                                                              \
  } else {                                                                    \
    ldz_col = ldz;                                                            \
  }                                                                           \
  work = (RTYPE *)malloc((size_t)work_size * sizeof(RTYPE));                  \
  if (!work) {                                                                \
    if (layout == FB_LAYOUT_ROW_MAJOR && vectors_needed) free(z_col);         \
    return -101;                                                              \
  }                                                                           \
  ((void (*)(char *, int *, RTYPE *, RTYPE *, CTYPE *, int *, RTYPE *, int *))\
       g_fortran_fn[OP_ID])(&compz_, &n_, d, e, z_col, &ldz_col, work, &info);\
  if (info == 0 && layout == FB_LAYOUT_ROW_MAJOR && vectors_needed) {         \
    for (row = 0; row < n; ++row) {                                           \
      for (col = 0; col < n; ++col) {                                         \
        z[(size_t)row * (size_t)ldz + (size_t)col] =                          \
            z_col[(size_t)row + (size_t)col * (size_t)ldz_col];               \
      }                                                                       \
    }                                                                         \
  }                                                                           \
  free(work);                                                                 \
  if (layout == FB_LAYOUT_ROW_MAJOR && vectors_needed) free(z_col);           \
  return info;                                                                \
}                                                                             \
static void thunk_##OP##_cblas_to_fortran(char *compz, int *n, RTYPE *d,      \
                                          RTYPE *e, CTYPE *z, int *ldz,       \
                                          RTYPE *work, int *info) {           \
  CTYPE dummy_z = (CTYPE)0;                                                   \
  CTYPE *z_ptr = z ? z : &dummy_z;                                            \
  int ldz_value = (z != NULL && ldz != NULL) ? *ldz : 1;                      \
  (void)work;                                                                 \
  if (!compz || !n || !d || !e) {                                             \
    if (info) *info = -1;                                                     \
    return;                                                                   \
  }                                                                           \
  if (fb_lapack_job_produces_vectors(*compz) && (!z || !ldz)) {               \
    if (info) *info = -1;                                                     \
    return;                                                                   \
  }                                                                           \
  if (!g_cblas_fn[OP_ID]) {                                                   \
    if (info) *info = -1;                                                     \
    return;                                                                   \
  }                                                                           \
  int rc = ((int (*)(fb_layout_t, char, int, RTYPE *, RTYPE *, CTYPE *, int)) \
                g_cblas_fn[OP_ID])(FB_LAYOUT_COL_MAJOR, *compz, *n, d, e,     \
                                   z_ptr, ldz_value);                         \
  if (info) *info = rc;                                                       \
}

DEFINE_STEQR_COMPLEX_THUNKS(csteqr, fb_complex_float_t, float, FB_OP_CSTEQR)
DEFINE_STEQR_COMPLEX_THUNKS(zsteqr, fb_complex_double_t, double, FB_OP_ZSTEQR)
DEFINE_STEQR_COMPLEX_THUNKS(cpteqr, fb_complex_float_t, float, FB_OP_CPTEQR)
DEFINE_STEQR_COMPLEX_THUNKS(zpteqr, fb_complex_double_t, double, FB_OP_ZPTEQR)

#undef DEFINE_STEQR_COMPLEX_THUNKS

#define DEFINE_STERF_REAL_THUNKS(OP, TYPE, OP_ID)                             \
static int thunk_##OP##_fortran_to_cblas(const int n, TYPE *d, TYPE *e) {     \
  int n_ = n;                                                                 \
  int info = 0;                                                               \
  if (!d || !e) return -1;                                                    \
  if (!g_fortran_fn[OP_ID]) return -1;                                        \
  ((void (*)(int *, TYPE *, TYPE *, int *))g_fortran_fn[OP_ID])(              \
      &n_, d, e, &info);                                                      \
  return info;                                                                \
}                                                                             \
static void thunk_##OP##_cblas_to_fortran(int *n, TYPE *d, TYPE *e,           \
                                          int *info) {                        \
  if (!d || !e) {                                                             \
    if (info) *info = -1;                                                     \
    return;                                                                   \
  }                                                                           \
  if (!g_cblas_fn[OP_ID]) {                                                   \
    if (info) *info = -1;                                                     \
    return;                                                                   \
  }                                                                           \
  int rc = ((int (*)(int, TYPE *, TYPE *))g_cblas_fn[OP_ID])(*n, d, e);      \
  if (info) *info = rc;                                                       \
}

DEFINE_STERF_REAL_THUNKS(ssterf, float, FB_OP_SSTERF)
DEFINE_STERF_REAL_THUNKS(dsterf, double, FB_OP_DSTERF)

#undef DEFINE_STERF_REAL_THUNKS

/* ---- SGEES / DGEES  (FB_OP_SGEES=221, FB_OP_DGEES=222) ----------------- */

static void thunk_sgees_fortran_to_cblas(char *jobvs, char *sort, void *select,
                                         int *n, float *a, int *lda, int *sdim,
                                         float *wr, float *wi, float *vs,
                                         int *ldvs, float *work, int *lwork,
                                         int *bwork, int *info) {
  if (!g_cblas_fn[FB_OP_SGEES]) {
    if (info)
      *info = -1;
    return;
  }
  /* CBLAS: (int layout, char jobvs, char sort, void *select, int n,
   *         float *a, int lda, int *sdim, float *wr, float *wi,
   *         float *vs, int ldvs, float *work, int lwork, int *bwork, int *info)
   */
  ((void (*)(int, char, char, void *, int, float *, int, int *, float *,
             float *, float *, int, float *, int, int *,
             int *))g_cblas_fn[FB_OP_SGEES])(102, *jobvs, *sort, select, *n, a,
                                             *lda, sdim, wr, wi, vs, *ldvs,
                                             work, *lwork, bwork, info);
}

static void thunk_sgees_cblas_to_fortran(int layout, char jobvs, char sort,
                                         void *select, int n, float *a, int lda,
                                         int *sdim, float *wr, float *wi,
                                         float *vs, int ldvs, float *work,
                                         int lwork, int *bwork, int *info) {
  if (fb_lapack_reject_non_col_major_cblas_layout(layout, info))
    return;
  if (!g_fortran_fn[FB_OP_SGEES]) {
    if (info)
      *info = -1;
    return;
  }
  int lda_ = lda, ldvs_ = ldvs, lwork_ = lwork;
  ((void (*)(char *, char *, void *, int *, float *, int *, int *, float *,
             float *, float *, int *, float *, int *, int *,
             int *))g_fortran_fn[FB_OP_SGEES])(&jobvs, &sort, select, &n, a,
                                               &lda_, sdim, wr, wi, vs, &ldvs_,
                                               work, &lwork_, bwork, info);
}

static void thunk_dgees_fortran_to_cblas(char *jobvs, char *sort, void *select,
                                         int *n, double *a, int *lda, int *sdim,
                                         double *wr, double *wi, double *vs,
                                         int *ldvs, double *work, int *lwork,
                                         int *bwork, int *info) {
  if (!g_cblas_fn[FB_OP_DGEES]) {
    if (info)
      *info = -1;
    return;
  }
  ((void (*)(int, char, char, void *, int, double *, int, int *, double *,
             double *, double *, int, double *, int, int *,
             int *))g_cblas_fn[FB_OP_DGEES])(102, *jobvs, *sort, select, *n, a,
                                             *lda, sdim, wr, wi, vs, *ldvs,
                                             work, *lwork, bwork, info);
}

static void thunk_dgees_cblas_to_fortran(int layout, char jobvs, char sort,
                                         void *select, int n, double *a,
                                         int lda, int *sdim, double *wr,
                                         double *wi, double *vs, int ldvs,
                                         double *work, int lwork, int *bwork,
                                         int *info) {
  if (fb_lapack_reject_non_col_major_cblas_layout(layout, info))
    return;
  if (!g_fortran_fn[FB_OP_DGEES]) {
    if (info)
      *info = -1;
    return;
  }
  int lda_ = lda, ldvs_ = ldvs, lwork_ = lwork;
  ((void (*)(char *, char *, void *, int *, double *, int *, int *, double *,
             double *, double *, int *, double *, int *, int *,
             int *))g_fortran_fn[FB_OP_DGEES])(&jobvs, &sort, select, &n, a,
                                               &lda_, sdim, wr, wi, vs, &ldvs_,
                                               work, &lwork_, bwork, info);
}

static void thunk_cgees_fortran_to_cblas(char *jobvs, char *sort, void *select,
                                         int *n, fb_complex_float_t *a,
                                         int *lda, int *sdim,
                                         fb_complex_float_t *w,
                                         fb_complex_float_t *vs, int *ldvs,
                                         fb_complex_float_t *work, int *lwork,
                                         float *rwork, int *bwork, int *info) {
  if (!g_cblas_fn[FB_OP_CGEES]) {
    if (info)
      *info = -1;
    return;
  }
  ((void (*)(int, char, char, void *, int, fb_complex_float_t *, int, int *,
             fb_complex_float_t *, fb_complex_float_t *, int,
             fb_complex_float_t *, int, float *, int *,
             int *))g_cblas_fn[FB_OP_CGEES])(102, *jobvs, *sort, select, *n, a,
                                             *lda, sdim, w, vs, *ldvs, work,
                                             *lwork, rwork, bwork, info);
}

static void thunk_cgees_cblas_to_fortran(
    int layout, char jobvs, char sort, void *select, int n,
    fb_complex_float_t *a, int lda, int *sdim, fb_complex_float_t *w,
    fb_complex_float_t *vs, int ldvs, fb_complex_float_t *work, int lwork,
    float *rwork, int *bwork, int *info) {
  if (fb_lapack_reject_non_col_major_cblas_layout(layout, info))
    return;
  if (!g_fortran_fn[FB_OP_CGEES]) {
    if (info)
      *info = -1;
    return;
  }
  int lda_ = lda, ldvs_ = ldvs, lwork_ = lwork;
  ((void (*)(char *, char *, void *, int *, fb_complex_float_t *, int *, int *,
             fb_complex_float_t *, fb_complex_float_t *, int *,
             fb_complex_float_t *, int *, float *, int *,
             int *))g_fortran_fn[FB_OP_CGEES])(&jobvs, &sort, select, &n, a,
                                               &lda_, sdim, w, vs, &ldvs_,
                                               work, &lwork_, rwork, bwork,
                                               info);
}

static void thunk_zgees_fortran_to_cblas(
    char *jobvs, char *sort, void *select, int *n, fb_complex_double_t *a,
    int *lda, int *sdim, fb_complex_double_t *w, fb_complex_double_t *vs,
    int *ldvs, fb_complex_double_t *work, int *lwork, double *rwork,
    int *bwork, int *info) {
  if (!g_cblas_fn[FB_OP_ZGEES]) {
    if (info)
      *info = -1;
    return;
  }
  ((void (*)(int, char, char, void *, int, fb_complex_double_t *, int, int *,
             fb_complex_double_t *, fb_complex_double_t *, int,
             fb_complex_double_t *, int, double *, int *,
             int *))g_cblas_fn[FB_OP_ZGEES])(102, *jobvs, *sort, select, *n, a,
                                             *lda, sdim, w, vs, *ldvs, work,
                                             *lwork, rwork, bwork, info);
}

static void thunk_zgees_cblas_to_fortran(
    int layout, char jobvs, char sort, void *select, int n,
    fb_complex_double_t *a, int lda, int *sdim, fb_complex_double_t *w,
    fb_complex_double_t *vs, int ldvs, fb_complex_double_t *work, int lwork,
    double *rwork, int *bwork, int *info) {
  if (fb_lapack_reject_non_col_major_cblas_layout(layout, info))
    return;
  if (!g_fortran_fn[FB_OP_ZGEES]) {
    if (info)
      *info = -1;
    return;
  }
  int lda_ = lda, ldvs_ = ldvs, lwork_ = lwork;
  ((void (*)(char *, char *, void *, int *, fb_complex_double_t *, int *,
             int *, fb_complex_double_t *, fb_complex_double_t *, int *,
             fb_complex_double_t *, int *, double *, int *,
             int *))g_fortran_fn[FB_OP_ZGEES])(&jobvs, &sort, select, &n, a,
                                               &lda_, sdim, w, vs, &ldvs_,
                                               work, &lwork_, rwork, bwork,
                                               info);
}

static void thunk_sgeesx_fortran_to_cblas(char *jobvs, char *sort,
                                          void *select, char *sense, int *n,
                                          float *a, int *lda, int *sdim,
                                          float *wr, float *wi, float *vs,
                                          int *ldvs, float *rconde,
                                          float *rcondv, float *work,
                                          int *lwork, int *iwork, int *liwork,
                                          int *bwork, int *info) {
  if (!g_cblas_fn[FB_OP_SGEESX]) {
    if (info) *info = -1;
    return;
  }
  ((void (*)(int, char, char, void *, char, int, float *, int, int *, float *,
             float *, float *, int, float *, float *))g_cblas_fn[FB_OP_SGEESX])(
      102, *jobvs, *sort, select, *sense, *n, a, *lda, sdim, wr, wi, vs,
      *ldvs, rconde, rcondv);
}

static void thunk_sgeesx_cblas_to_fortran(int layout, char jobvs, char sort,
                                          void *select, char sense, int n,
                                          float *a, int lda, int *sdim,
                                          float *wr, float *wi, float *vs,
                                          int ldvs, float *rconde,
                                          float *rcondv, float *work,
                                          int lwork, int *iwork, int liwork,
                                          int *bwork, int *info) {
  (void)work;
  (void)lwork;
  (void)iwork;
  (void)liwork;
  if (fb_lapack_reject_non_col_major_cblas_layout(layout, info)) return;
  if (!g_fortran_fn[FB_OP_SGEESX]) {
    if (info) *info = -1;
    return;
  }
  int lda_ = lda, ldvs_ = ldvs, lwork_ = -1, liwork_ = -1, info_ = 0;
  float work_query = 0.0f;
  int iwork_query = 0;
  int *bwork_ = bwork;
  int *alloc_bwork = NULL;
  if (bwork_ == NULL) {
    alloc_bwork = (int *)calloc((size_t)((n > 1) ? n : 1), sizeof(int));
    if (alloc_bwork == NULL) {
      if (info) *info = -101;
      return;
    }
    bwork_ = alloc_bwork;
  }
  ((void (*)(char *, char *, void *, char *, int *, float *, int *, int *,
             float *, float *, float *, int *, float *, float *, float *,
             int *, int *, int *, int *, int *))g_fortran_fn[FB_OP_SGEESX])(
      &jobvs, &sort, select, &sense, &n, a, &lda_, sdim, wr, wi, vs, &ldvs_,
      rconde, rcondv, &work_query, &lwork_, &iwork_query, &liwork_, bwork_,
      &info_);
  if (info_) {
    if (info) *info = info_;
    free(alloc_bwork);
    return;
  }
  lwork_ = (lwork_ > 0) ? lwork_ : 1;
  liwork_ = (liwork_ > 0) ? liwork_ : 1;
  float *work_ = (float *)malloc((size_t)lwork_ * sizeof(float));
  int *iwork_ = (int *)malloc((size_t)liwork_ * sizeof(int));
  if (!work_ || !iwork_) {
    free(work_);
    free(iwork_);
    free(alloc_bwork);
    if (info) *info = -101;
    return;
  }
  ((void (*)(char *, char *, void *, char *, int *, float *, int *, int *,
             float *, float *, float *, int *, float *, float *, float *,
             int *, int *, int *, int *, int *))g_fortran_fn[FB_OP_SGEESX])(
      &jobvs, &sort, select, &sense, &n, a, &lda_, sdim, wr, wi, vs, &ldvs_,
      rconde, rcondv, work_, &lwork_, iwork_, &liwork_, bwork_, &info_);
  if (info) *info = info_;
  free(work_);
  free(iwork_);
  free(alloc_bwork);
}

static void thunk_dgeesx_fortran_to_cblas(char *jobvs, char *sort,
                                          void *select, char *sense, int *n,
                                          double *a, int *lda, int *sdim,
                                          double *wr, double *wi, double *vs,
                                          int *ldvs, double *rconde,
                                          double *rcondv, double *work,
                                          int *lwork, int *iwork,
                                          int *liwork, int *bwork, int *info) {
  if (!g_cblas_fn[FB_OP_DGEESX]) {
    if (info) *info = -1;
    return;
  }
  ((void (*)(int, char, char, void *, char, int, double *, int, int *,
             double *, double *, double *, int, double *, double *))
      g_cblas_fn[FB_OP_DGEESX])(102, *jobvs, *sort, select, *sense, *n, a,
                                *lda, sdim, wr, wi, vs, *ldvs, rconde,
                                rcondv);
}

static void thunk_dgeesx_cblas_to_fortran(int layout, char jobvs, char sort,
                                          void *select, char sense, int n,
                                          double *a, int lda, int *sdim,
                                          double *wr, double *wi, double *vs,
                                          int ldvs, double *rconde,
                                          double *rcondv, double *work,
                                          int lwork, int *iwork, int liwork,
                                          int *bwork, int *info) {
  (void)work;
  (void)lwork;
  (void)iwork;
  (void)liwork;
  if (fb_lapack_reject_non_col_major_cblas_layout(layout, info)) return;
  if (!g_fortran_fn[FB_OP_DGEESX]) {
    if (info) *info = -1;
    return;
  }
  int lda_ = lda, ldvs_ = ldvs, lwork_ = -1, liwork_ = -1, info_ = 0;
  double work_query = 0.0;
  int iwork_query = 0;
  int *bwork_ = bwork;
  int *alloc_bwork = NULL;
  if (bwork_ == NULL) {
    alloc_bwork = (int *)calloc((size_t)((n > 1) ? n : 1), sizeof(int));
    if (alloc_bwork == NULL) {
      if (info) *info = -101;
      return;
    }
    bwork_ = alloc_bwork;
  }
  ((void (*)(char *, char *, void *, char *, int *, double *, int *, int *,
             double *, double *, double *, int *, double *, double *,
             double *, int *, int *, int *, int *, int *))g_fortran_fn[FB_OP_DGEESX])(
      &jobvs, &sort, select, &sense, &n, a, &lda_, sdim, wr, wi, vs, &ldvs_,
      rconde, rcondv, &work_query, &lwork_, &iwork_query, &liwork_, bwork_,
      &info_);
  if (info_) {
    if (info) *info = info_;
    free(alloc_bwork);
    return;
  }
  lwork_ = (lwork_ > 0) ? lwork_ : 1;
  liwork_ = (liwork_ > 0) ? liwork_ : 1;
  double *work_ = (double *)malloc((size_t)lwork_ * sizeof(double));
  int *iwork_ = (int *)malloc((size_t)liwork_ * sizeof(int));
  if (!work_ || !iwork_) {
    free(work_);
    free(iwork_);
    free(alloc_bwork);
    if (info) *info = -101;
    return;
  }
  ((void (*)(char *, char *, void *, char *, int *, double *, int *, int *,
             double *, double *, double *, int *, double *, double *,
             double *, int *, int *, int *, int *, int *))g_fortran_fn[FB_OP_DGEESX])(
      &jobvs, &sort, select, &sense, &n, a, &lda_, sdim, wr, wi, vs, &ldvs_,
      rconde, rcondv, work_, &lwork_, iwork_, &liwork_, bwork_, &info_);
  if (info) *info = info_;
  free(work_);
  free(iwork_);
  free(alloc_bwork);
}

static void thunk_cgeesx_fortran_to_cblas(char *jobvs, char *sort,
                                          void *select, char *sense, int *n,
                                          fb_complex_float_t *a, int *lda,
                                          int *sdim, fb_complex_float_t *w,
                                          fb_complex_float_t *vs, int *ldvs,
                                          float *rconde, float *rcondv,
                                          fb_complex_float_t *work,
                                          int *lwork, float *rwork,
                                          int *bwork, int *info) {
  if (!g_cblas_fn[FB_OP_CGEESX]) {
    if (info) *info = -1;
    return;
  }
  ((void (*)(int, char, char, void *, char, int, fb_complex_float_t *, int,
             int *, fb_complex_float_t *, fb_complex_float_t *, int, float *,
             float *))g_cblas_fn[FB_OP_CGEESX])(102, *jobvs, *sort, select,
                                                *sense, *n, a, *lda, sdim, w,
                                                vs, *ldvs, rconde, rcondv);
}

static void thunk_cgeesx_cblas_to_fortran(int layout, char jobvs, char sort,
                                          void *select, char sense, int n,
                                          fb_complex_float_t *a, int lda,
                                          int *sdim, fb_complex_float_t *w,
                                          fb_complex_float_t *vs, int ldvs,
                                          float *rconde, float *rcondv,
                                          fb_complex_float_t *work,
                                          int lwork, float *rwork,
                                          int *bwork, int *info) {
  if (fb_lapack_reject_non_col_major_cblas_layout(layout, info)) return;
  if (!g_fortran_fn[FB_OP_CGEESX]) {
    if (info) *info = -1;
    return;
  }
  int lda_ = lda, ldvs_ = ldvs, lwork_ = -1, info_ = 0;
  float *rwork_ = rwork;
  int *bwork_ = bwork;
  int *alloc_bwork = NULL;
  float *alloc_rwork = NULL;
  if (rwork_ == NULL) {
    alloc_rwork = (float *)calloc((size_t)((n > 1) ? n : 1), sizeof(float));
    if (alloc_rwork == NULL) {
      if (info) *info = -101;
      return;
    }
    rwork_ = alloc_rwork;
  }
  if (bwork_ == NULL) {
    alloc_bwork = (int *)calloc((size_t)((n > 1) ? n : 1), sizeof(int));
    if (alloc_bwork == NULL) {
      free(alloc_rwork);
      if (info) *info = -101;
      return;
    }
    bwork_ = alloc_bwork;
  }
  ((void (*)(char *, char *, void *, char *, int *, fb_complex_float_t *,
             int *, int *, fb_complex_float_t *, fb_complex_float_t *, int *,
             float *, float *, fb_complex_float_t *, int *, float *, int *,
             int *))g_fortran_fn[FB_OP_CGEESX])(
      &jobvs, &sort, select, &sense, &n, a, &lda_, sdim, w, vs, &ldvs_,
      rconde, rcondv, work, &lwork_, rwork_, bwork_, &info_);
  if (info) *info = info_;
  free(alloc_rwork);
  free(alloc_bwork);
}

static void thunk_zgeesx_fortran_to_cblas(char *jobvs, char *sort,
                                          void *select, char *sense, int *n,
                                          fb_complex_double_t *a, int *lda,
                                          int *sdim, fb_complex_double_t *w,
                                          fb_complex_double_t *vs, int *ldvs,
                                          double *rconde, double *rcondv,
                                          fb_complex_double_t *work,
                                          int *lwork, double *rwork,
                                          int *bwork, int *info) {
  if (!g_cblas_fn[FB_OP_ZGEESX]) {
    if (info) *info = -1;
    return;
  }
  ((void (*)(int, char, char, void *, char, int, fb_complex_double_t *, int,
             int *, fb_complex_double_t *, fb_complex_double_t *, int,
             double *, double *))g_cblas_fn[FB_OP_ZGEESX])(102, *jobvs,
                                                            *sort, select,
                                                            *sense, *n, a,
                                                            *lda, sdim, w,
                                                            vs, *ldvs,
                                                            rconde, rcondv);
}

static void thunk_zgeesx_cblas_to_fortran(int layout, char jobvs, char sort,
                                          void *select, char sense, int n,
                                          fb_complex_double_t *a, int lda,
                                          int *sdim, fb_complex_double_t *w,
                                          fb_complex_double_t *vs, int ldvs,
                                          double *rconde, double *rcondv,
                                          fb_complex_double_t *work,
                                          int lwork, double *rwork,
                                          int *bwork, int *info) {
  if (fb_lapack_reject_non_col_major_cblas_layout(layout, info)) return;
  if (!g_fortran_fn[FB_OP_ZGEESX]) {
    if (info) *info = -1;
    return;
  }
  int lda_ = lda, ldvs_ = ldvs, lwork_ = -1, info_ = 0;
  double *rwork_ = rwork;
  int *bwork_ = bwork;
  int *alloc_bwork = NULL;
  double *alloc_rwork = NULL;
  if (rwork_ == NULL) {
    alloc_rwork = (double *)calloc((size_t)((n > 1) ? n : 1), sizeof(double));
    if (alloc_rwork == NULL) {
      if (info) *info = -101;
      return;
    }
    rwork_ = alloc_rwork;
  }
  if (bwork_ == NULL) {
    alloc_bwork = (int *)calloc((size_t)((n > 1) ? n : 1), sizeof(int));
    if (alloc_bwork == NULL) {
      free(alloc_rwork);
      if (info) *info = -101;
      return;
    }
    bwork_ = alloc_bwork;
  }
  ((void (*)(char *, char *, void *, char *, int *, fb_complex_double_t *,
             int *, int *, fb_complex_double_t *, fb_complex_double_t *, int *,
             double *, double *, fb_complex_double_t *, int *, double *, int *,
             int *))g_fortran_fn[FB_OP_ZGEESX])(
      &jobvs, &sort, select, &sense, &n, a, &lda_, sdim, w, vs, &ldvs_,
      rconde, rcondv, work, &lwork_, rwork_, bwork_, &info_);
  if (info) *info = info_;
  free(alloc_rwork);
  free(alloc_bwork);
}

static void thunk_sbdsdc_fortran_to_cblas(char *uplo, char *compq, int *n,
                                          float *d, float *e, float *u,
                                          int *ldu, float *vt, int *ldvt,
                                          float *q, int *iq, float *work,
                                          int *iwork, int *info) {
  if (!g_cblas_fn[FB_OP_SBDSDC]) {
    if (info) *info = -1;
    return;
  }
  int rc = ((int (*)(int, char, char, int, float *, float *, float *, int,
                     float *, int, float *, int *))g_cblas_fn[FB_OP_SBDSDC])(
      102, *uplo, *compq, *n, d, e, u, *ldu, vt, *ldvt, q, iq);
  if (info) *info = rc;
}

static void thunk_dbdsdc_fortran_to_cblas(char *uplo, char *compq, int *n,
                                          double *d, double *e, double *u,
                                          int *ldu, double *vt, int *ldvt,
                                          double *q, int *iq, double *work,
                                          int *iwork, int *info) {
  if (!g_cblas_fn[FB_OP_DBDSDC]) {
    if (info) *info = -1;
    return;
  }
  int rc = ((int (*)(int, char, char, int, double *, double *, double *, int,
                     double *, int, double *, int *))g_cblas_fn[FB_OP_DBDSDC])(
      102, *uplo, *compq, *n, d, e, u, *ldu, vt, *ldvt, q, iq);
  if (info) *info = rc;
}

static void thunk_cbdsdc_fortran_to_cblas(char *uplo, char *compq, int *n,
                                          float *d, float *e,
                                          fb_complex_float_t *u, int *ldu,
                                          fb_complex_float_t *vt, int *ldvt,
                                          float *q, int *iq, float *work,
                                          int *iwork, int *info) {
  if (!g_cblas_fn[FB_OP_CBDSDC]) {
    if (info) *info = -1;
    return;
  }
  int rc = ((int (*)(int, char, char, int, float *, float *,
                     fb_complex_float_t *, int, fb_complex_float_t *, int,
                     float *, int *))g_cblas_fn[FB_OP_CBDSDC])(
      102, *uplo, *compq, *n, d, e, u, *ldu, vt, *ldvt, q, iq);
  if (info) *info = rc;
}

static void thunk_zbdsdc_fortran_to_cblas(char *uplo, char *compq, int *n,
                                          double *d, double *e,
                                          fb_complex_double_t *u, int *ldu,
                                          fb_complex_double_t *vt, int *ldvt,
                                          double *q, int *iq, double *work,
                                          int *iwork, int *info) {
  if (!g_cblas_fn[FB_OP_ZBDSDC]) {
    if (info) *info = -1;
    return;
  }
  int rc = ((int (*)(int, char, char, int, double *, double *,
                     fb_complex_double_t *, int, fb_complex_double_t *, int,
                     double *, int *))g_cblas_fn[FB_OP_ZBDSDC])(
      102, *uplo, *compq, *n, d, e, u, *ldu, vt, *ldvt, q, iq);
  if (info) *info = rc;
}

static void thunk_sbdsdc_cblas_to_fortran(int layout, char uplo, char compq,
                                          int n, float *d, float *e, float *u,
                                          int ldu, float *vt, int ldvt,
                                          float *q, int *iq, float *work,
                                          int *iwork, int *info) {
  (void)work;
  (void)iwork;
  if (fb_lapack_reject_non_col_major_cblas_layout(layout, info)) return;
  if (!g_fortran_fn[FB_OP_SBDSDC]) {
    if (info) *info = -1;
    return;
  }
  int ldu_ = ldu, ldvt_ = ldvt;
  ((void (*)(char *, char *, int *, float *, float *, float *, int *, float *,
             int *, float *, int *, float *, int *, int *))g_fortran_fn[FB_OP_SBDSDC])(
      &uplo, &compq, &n, d, e, u, &ldu_, vt, &ldvt_, q, iq, work, iwork, info);
}

static void thunk_dbdsdc_cblas_to_fortran(int layout, char uplo, char compq,
                                          int n, double *d, double *e,
                                          double *u, int ldu, double *vt,
                                          int ldvt, double *q, int *iq,
                                          double *work, int *iwork, int *info) {
  (void)work;
  (void)iwork;
  if (fb_lapack_reject_non_col_major_cblas_layout(layout, info)) return;
  if (!g_fortran_fn[FB_OP_DBDSDC]) {
    if (info) *info = -1;
    return;
  }
  int ldu_ = ldu, ldvt_ = ldvt;
  ((void (*)(char *, char *, int *, double *, double *, double *, int *,
             double *, int *, double *, int *, double *, int *, int *))g_fortran_fn[FB_OP_DBDSDC])(
      &uplo, &compq, &n, d, e, u, &ldu_, vt, &ldvt_, q, iq, work, iwork, info);
}

static void thunk_cbdsdc_cblas_to_fortran(int layout, char uplo, char compq,
                                          int n, float *d, float *e,
                                          fb_complex_float_t *u, int ldu,
                                          fb_complex_float_t *vt, int ldvt,
                                          float *q, int *iq, float *work,
                                          int *iwork, int *info) {
  (void)work;
  (void)iwork;
  if (fb_lapack_reject_non_col_major_cblas_layout(layout, info)) return;
  if (!g_fortran_fn[FB_OP_CBDSDC]) {
    if (info) *info = -1;
    return;
  }
  int ldu_ = ldu, ldvt_ = ldvt;
  ((void (*)(char *, char *, int *, float *, float *, fb_complex_float_t *,
             int *, fb_complex_float_t *, int *, float *, int *, float *,
             int *, int *))g_fortran_fn[FB_OP_CBDSDC])(&uplo, &compq, &n, d,
                                                        e, u, &ldu_, vt,
                                                        &ldvt_, q, iq, work,
                                                        iwork, info);
}

static void thunk_zbdsdc_cblas_to_fortran(int layout, char uplo, char compq,
                                          int n, double *d, double *e,
                                          fb_complex_double_t *u, int ldu,
                                          fb_complex_double_t *vt, int ldvt,
                                          double *q, int *iq, double *work,
                                          int *iwork, int *info) {
  (void)work;
  (void)iwork;
  if (fb_lapack_reject_non_col_major_cblas_layout(layout, info)) return;
  if (!g_fortran_fn[FB_OP_ZBDSDC]) {
    if (info) *info = -1;
    return;
  }
  int ldu_ = ldu, ldvt_ = ldvt;
  ((void (*)(char *, char *, int *, double *, double *, fb_complex_double_t *,
             int *, fb_complex_double_t *, int *, double *, int *, double *,
             int *, int *))g_fortran_fn[FB_OP_ZBDSDC])(&uplo, &compq, &n, d,
                                                        e, u, &ldu_, vt,
                                                        &ldvt_, q, iq, work,
                                                        iwork, info);
}

static void thunk_sbdsqr_fortran_to_cblas(char *uplo, int *n, int *ncvt,
                                          int *nru, int *ncc, float *d,
                                          float *e, float *vt, int *ldvt,
                                          float *u, int *ldu, float *c,
                                          int *ldc, float *work, int *info) {
  if (!g_cblas_fn[FB_OP_SBDSQR]) {
    if (info) *info = -1;
    return;
  }
  int rc = ((int (*)(int, char, int, int, int, int, float *, float *, float *,
                     int, float *, int, float *, int))g_cblas_fn[FB_OP_SBDSQR])(
      102, *uplo, *n, *ncvt, *nru, *ncc, d, e, vt, *ldvt, u, *ldu, c, *ldc);
  if (info) *info = rc;
}

static void thunk_dbdsqr_fortran_to_cblas(char *uplo, int *n, int *ncvt,
                                          int *nru, int *ncc, double *d,
                                          double *e, double *vt, int *ldvt,
                                          double *u, int *ldu, double *c,
                                          int *ldc, double *work, int *info) {
  if (!g_cblas_fn[FB_OP_DBDSQR]) {
    if (info) *info = -1;
    return;
  }
  int rc = ((int (*)(int, char, int, int, int, int, double *, double *,
                     double *, int, double *, int, double *, int))
      g_cblas_fn[FB_OP_DBDSQR])(102, *uplo, *n, *ncvt, *nru, *ncc, d, e, vt,
                                *ldvt, u, *ldu, c, *ldc);
  if (info) *info = rc;
}

static void thunk_cbdsqr_fortran_to_cblas(char *uplo, int *n, int *ncvt,
                                          int *nru, int *ncc, float *d,
                                          float *e, fb_complex_float_t *vt,
                                          int *ldvt, fb_complex_float_t *u,
                                          int *ldu, fb_complex_float_t *c,
                                          int *ldc, float *rwork, int *info) {
  if (!g_cblas_fn[FB_OP_CBDSQR]) {
    if (info) *info = -1;
    return;
  }
  int rc = ((int (*)(int, char, int, int, int, int, float *, float *,
                     fb_complex_float_t *, int, fb_complex_float_t *, int,
                     fb_complex_float_t *, int))g_cblas_fn[FB_OP_CBDSQR])(
      102, *uplo, *n, *ncvt, *nru, *ncc, d, e, vt, *ldvt, u, *ldu, c, *ldc);
  if (info) *info = rc;
}

static void thunk_zbdsqr_fortran_to_cblas(char *uplo, int *n, int *ncvt,
                                          int *nru, int *ncc, double *d,
                                          double *e, fb_complex_double_t *vt,
                                          int *ldvt, fb_complex_double_t *u,
                                          int *ldu, fb_complex_double_t *c,
                                          int *ldc, double *rwork, int *info) {
  if (!g_cblas_fn[FB_OP_ZBDSQR]) {
    if (info) *info = -1;
    return;
  }
  int rc = ((int (*)(int, char, int, int, int, int, double *, double *,
                     fb_complex_double_t *, int, fb_complex_double_t *, int,
                     fb_complex_double_t *, int))g_cblas_fn[FB_OP_ZBDSQR])(
      102, *uplo, *n, *ncvt, *nru, *ncc, d, e, vt, *ldvt, u, *ldu, c, *ldc);
  if (info) *info = rc;
}

static void thunk_sbdsqr_cblas_to_fortran(int layout, char uplo, int n,
                                          int ncvt, int nru, int ncc, float *d,
                                          float *e, float *vt, int ldvt,
                                          float *u, int ldu, float *c, int ldc,
                                          float *work, int *info) {
  if (fb_lapack_reject_non_col_major_cblas_layout(layout, info)) return;
  if (!g_fortran_fn[FB_OP_SBDSQR]) {
    if (info) *info = -1;
    return;
  }
  int ldvt_ = ldvt, ldu_ = ldu, ldc_ = ldc;
  ((void (*)(char *, int *, int *, int *, int *, float *, float *, float *,
             int *, float *, int *, float *, int *, float *, int *))
      g_fortran_fn[FB_OP_SBDSQR])(&uplo, &n, &ncvt, &nru, &ncc, d, e, vt,
                                  &ldvt_, u, &ldu_, c, &ldc_, work, info);
}

static void thunk_dbdsqr_cblas_to_fortran(int layout, char uplo, int n,
                                          int ncvt, int nru, int ncc,
                                          double *d, double *e, double *vt,
                                          int ldvt, double *u, int ldu,
                                          double *c, int ldc, double *work,
                                          int *info) {
  if (fb_lapack_reject_non_col_major_cblas_layout(layout, info)) return;
  if (!g_fortran_fn[FB_OP_DBDSQR]) {
    if (info) *info = -1;
    return;
  }
  int ldvt_ = ldvt, ldu_ = ldu, ldc_ = ldc;
  ((void (*)(char *, int *, int *, int *, int *, double *, double *, double *,
             int *, double *, int *, double *, int *, double *, int *))
      g_fortran_fn[FB_OP_DBDSQR])(&uplo, &n, &ncvt, &nru, &ncc, d, e, vt,
                                  &ldvt_, u, &ldu_, c, &ldc_, work, info);
}

static void thunk_cbdsqr_cblas_to_fortran(int layout, char uplo, int n,
                                          int ncvt, int nru, int ncc, float *d,
                                          float *e, fb_complex_float_t *vt,
                                          int ldvt, fb_complex_float_t *u,
                                          int ldu, fb_complex_float_t *c,
                                          int ldc, float *rwork, int *info) {
  if (fb_lapack_reject_non_col_major_cblas_layout(layout, info)) return;
  if (!g_fortran_fn[FB_OP_CBDSQR]) {
    if (info) *info = -1;
    return;
  }
  int ldvt_ = ldvt, ldu_ = ldu, ldc_ = ldc;
  ((void (*)(char *, int *, int *, int *, int *, float *, float *,
             fb_complex_float_t *, int *, fb_complex_float_t *, int *,
             fb_complex_float_t *, int *, float *, int *))
      g_fortran_fn[FB_OP_CBDSQR])(&uplo, &n, &ncvt, &nru, &ncc, d, e, vt,
                                  &ldvt_, u, &ldu_, c, &ldc_, rwork, info);
}

static void thunk_zbdsqr_cblas_to_fortran(int layout, char uplo, int n,
                                          int ncvt, int nru, int ncc,
                                          double *d, double *e,
                                          fb_complex_double_t *vt, int ldvt,
                                          fb_complex_double_t *u, int ldu,
                                          fb_complex_double_t *c, int ldc,
                                          double *rwork, int *info) {
  if (fb_lapack_reject_non_col_major_cblas_layout(layout, info)) return;
  if (!g_fortran_fn[FB_OP_ZBDSQR]) {
    if (info) *info = -1;
    return;
  }
  int ldvt_ = ldvt, ldu_ = ldu, ldc_ = ldc;
  ((void (*)(char *, int *, int *, int *, int *, double *, double *,
             fb_complex_double_t *, int *, fb_complex_double_t *, int *,
             fb_complex_double_t *, int *, double *, int *))
      g_fortran_fn[FB_OP_ZBDSQR])(&uplo, &n, &ncvt, &nru, &ncc, d, e, vt,
                                  &ldvt_, u, &ldu_, c, &ldc_, rwork, info);
}

static void thunk_sgeevx_fortran_to_cblas(char *balanc, char *jobvl,
                                          char *jobvr, char *sense, int *n,
                                          float *a, int *lda, float *wr,
                                          float *wi, float *vl, int *ldvl,
                                          float *vr, int *ldvr, int *ilo,
                                          int *ihi, float *scale,
                                          float *abnrm, float *rconde,
                                          float *rcondv, float *work,
                                          int *lwork, int *iwork, int *info) {
  if (!g_cblas_fn[FB_OP_SGEEVX]) {
    if (info) *info = -1;
    return;
  }
  int rc = ((int (*)(int, char, char, char, char, int, float *, int, float *,
                     float *, float *, int, float *, int, int *, int *,
                     float *, float *, float *, float *))g_cblas_fn[FB_OP_SGEEVX])(
      102, *balanc, *jobvl, *jobvr, *sense, *n, a, *lda, wr, wi, vl, *ldvl, vr,
      *ldvr, ilo, ihi, scale, abnrm, rconde, rcondv);
  if (info) *info = rc;
}

static void thunk_dgeevx_fortran_to_cblas(char *balanc, char *jobvl,
                                          char *jobvr, char *sense, int *n,
                                          double *a, int *lda, double *wr,
                                          double *wi, double *vl, int *ldvl,
                                          double *vr, int *ldvr, int *ilo,
                                          int *ihi, double *scale,
                                          double *abnrm, double *rconde,
                                          double *rcondv, double *work,
                                          int *lwork, int *iwork, int *info) {
  if (!g_cblas_fn[FB_OP_DGEEVX]) {
    if (info) *info = -1;
    return;
  }
  int rc = ((int (*)(int, char, char, char, char, int, double *, int, double *,
                     double *, double *, int, double *, int, int *, int *,
                     double *, double *, double *, double *))
      g_cblas_fn[FB_OP_DGEEVX])(102, *balanc, *jobvl, *jobvr, *sense, *n, a,
                                *lda, wr, wi, vl, *ldvl, vr, *ldvr, ilo, ihi,
                                scale, abnrm, rconde, rcondv);
  if (info) *info = rc;
}

static void thunk_cgeevx_fortran_to_cblas(char *balanc, char *jobvl,
                                          char *jobvr, char *sense, int *n,
                                          fb_complex_float_t *a, int *lda,
                                          fb_complex_float_t *w,
                                          fb_complex_float_t *vl, int *ldvl,
                                          fb_complex_float_t *vr, int *ldvr,
                                          int *ilo, int *ihi, float *scale,
                                          float *abnrm, float *rconde,
                                          float *rcondv,
                                          fb_complex_float_t *work,
                                          int *lwork, float *rwork,
                                          int *info) {
  if (!g_cblas_fn[FB_OP_CGEEVX]) {
    if (info) *info = -1;
    return;
  }
  int rc = ((int (*)(int, char, char, char, char, int, fb_complex_float_t *,
                     int, fb_complex_float_t *, fb_complex_float_t *, int,
                     fb_complex_float_t *, int, int *, int *, float *, float *,
                     float *, float *))g_cblas_fn[FB_OP_CGEEVX])(
      102, *balanc, *jobvl, *jobvr, *sense, *n, a, *lda, w, vl, *ldvl, vr,
      *ldvr, ilo, ihi, scale, abnrm, rconde, rcondv);
  if (info) *info = rc;
}

static void thunk_zgeevx_fortran_to_cblas(char *balanc, char *jobvl,
                                          char *jobvr, char *sense, int *n,
                                          fb_complex_double_t *a, int *lda,
                                          fb_complex_double_t *w,
                                          fb_complex_double_t *vl, int *ldvl,
                                          fb_complex_double_t *vr, int *ldvr,
                                          int *ilo, int *ihi, double *scale,
                                          double *abnrm, double *rconde,
                                          double *rcondv,
                                          fb_complex_double_t *work,
                                          int *lwork, double *rwork,
                                          int *info) {
  if (!g_cblas_fn[FB_OP_ZGEEVX]) {
    if (info) *info = -1;
    return;
  }
  int rc = ((int (*)(int, char, char, char, char, int, fb_complex_double_t *,
                     int, fb_complex_double_t *, fb_complex_double_t *, int,
                     fb_complex_double_t *, int, int *, int *, double *,
                     double *, double *, double *))g_cblas_fn[FB_OP_ZGEEVX])(
      102, *balanc, *jobvl, *jobvr, *sense, *n, a, *lda, w, vl, *ldvl, vr,
      *ldvr, ilo, ihi, scale, abnrm, rconde, rcondv);
  if (info) *info = rc;
}

static void thunk_sgeevx_cblas_to_fortran(int layout, char balanc, char jobvl,
                                          char jobvr, char sense, int n,
                                          float *a, int lda, float *wr,
                                          float *wi, float *vl, int ldvl,
                                          float *vr, int ldvr, int *ilo,
                                          int *ihi, float *scale,
                                          float *abnrm, float *rconde,
                                          float *rcondv, float *work,
                                          int lwork, int *iwork, int *info) {
  if (fb_lapack_reject_non_col_major_cblas_layout(layout, info)) return;
  if (!g_fortran_fn[FB_OP_SGEEVX]) {
    if (info) *info = -1;
    return;
  }
  int lda_ = lda, ldvl_ = ldvl, ldvr_ = ldvr, lwork_ = lwork;
  ((void (*)(char *, char *, char *, char *, int *, float *, int *, float *,
             float *, float *, int *, float *, int *, int *, int *, float *,
             float *, float *, float *, float *, int *, int *, int *))
      g_fortran_fn[FB_OP_SGEEVX])(&balanc, &jobvl, &jobvr, &sense, &n, a,
                                  &lda_, wr, wi, vl, &ldvl_, vr, &ldvr_, ilo,
                                  ihi, scale, abnrm, rconde, rcondv, work,
                                  &lwork_, iwork, info);
}

static void thunk_dgeevx_cblas_to_fortran(int layout, char balanc, char jobvl,
                                          char jobvr, char sense, int n,
                                          double *a, int lda, double *wr,
                                          double *wi, double *vl, int ldvl,
                                          double *vr, int ldvr, int *ilo,
                                          int *ihi, double *scale,
                                          double *abnrm, double *rconde,
                                          double *rcondv, double *work,
                                          int lwork, int *iwork, int *info) {
  if (fb_lapack_reject_non_col_major_cblas_layout(layout, info)) return;
  if (!g_fortran_fn[FB_OP_DGEEVX]) {
    if (info) *info = -1;
    return;
  }
  int lda_ = lda, ldvl_ = ldvl, ldvr_ = ldvr, lwork_ = lwork;
  ((void (*)(char *, char *, char *, char *, int *, double *, int *, double *,
             double *, double *, int *, double *, int *, int *, int *,
             double *, double *, double *, double *, double *, int *, int *,
             int *))g_fortran_fn[FB_OP_DGEEVX])(&balanc, &jobvl, &jobvr,
                                                &sense, &n, a, &lda_, wr, wi,
                                                vl, &ldvl_, vr, &ldvr_, ilo,
                                                ihi, scale, abnrm, rconde,
                                                rcondv, work, &lwork_, iwork,
                                                info);
}

static void thunk_cgeevx_cblas_to_fortran(
    int layout, char balanc, char jobvl, char jobvr, char sense, int n,
    fb_complex_float_t *a, int lda, fb_complex_float_t *w,
    fb_complex_float_t *vl, int ldvl, fb_complex_float_t *vr, int ldvr,
    int *ilo, int *ihi, float *scale, float *abnrm, float *rconde,
    float *rcondv, fb_complex_float_t *work, int lwork, float *rwork,
    int *info) {
  if (fb_lapack_reject_non_col_major_cblas_layout(layout, info)) return;
  if (!g_fortran_fn[FB_OP_CGEEVX]) {
    if (info) *info = -1;
    return;
  }
  int lda_ = lda, ldvl_ = ldvl, ldvr_ = ldvr, lwork_ = lwork;
  ((void (*)(char *, char *, char *, char *, int *, fb_complex_float_t *,
             int *, fb_complex_float_t *, fb_complex_float_t *, int *,
             fb_complex_float_t *, int *, int *, int *, float *, float *,
             float *, float *, fb_complex_float_t *, int *, float *, int *))
      g_fortran_fn[FB_OP_CGEEVX])(&balanc, &jobvl, &jobvr, &sense, &n, a,
                                  &lda_, w, vl, &ldvl_, vr, &ldvr_, ilo, ihi,
                                  scale, abnrm, rconde, rcondv, work, &lwork_,
                                  rwork, info);
}

static void thunk_zgeevx_cblas_to_fortran(
    int layout, char balanc, char jobvl, char jobvr, char sense, int n,
    fb_complex_double_t *a, int lda, fb_complex_double_t *w,
    fb_complex_double_t *vl, int ldvl, fb_complex_double_t *vr, int ldvr,
    int *ilo, int *ihi, double *scale, double *abnrm, double *rconde,
    double *rcondv, fb_complex_double_t *work, int lwork, double *rwork,
    int *info) {
  if (fb_lapack_reject_non_col_major_cblas_layout(layout, info)) return;
  if (!g_fortran_fn[FB_OP_ZGEEVX]) {
    if (info) *info = -1;
    return;
  }
  int lda_ = lda, ldvl_ = ldvl, ldvr_ = ldvr, lwork_ = lwork;
  ((void (*)(char *, char *, char *, char *, int *, fb_complex_double_t *,
             int *, fb_complex_double_t *, fb_complex_double_t *, int *,
             fb_complex_double_t *, int *, int *, int *, double *, double *,
             double *, double *, fb_complex_double_t *, int *, double *,
             int *))g_fortran_fn[FB_OP_ZGEEVX])(&balanc, &jobvl, &jobvr,
                                                &sense, &n, a, &lda_, w, vl,
                                                &ldvl_, vr, &ldvr_, ilo, ihi,
                                                scale, abnrm, rconde, rcondv,
                                                work, &lwork_, rwork, info);
}

static void thunk_sgelsd_fortran_to_cblas(int *m, int *n, int *nrhs,
                                          float *a, int *lda, float *b,
                                          int *ldb, float *s, float *rcond,
                                          int *rank, float *work, int *lwork,
                                          int *iwork, int *info) {
  (void)work;
  (void)lwork;
  (void)iwork;
  if (!g_cblas_fn[FB_OP_SGELSD]) {
    if (info) *info = -1;
    return;
  }
  int rc = ((int (*)(int, int, int, int, float *, int, float *, int, float *,
                     float, int *))g_cblas_fn[FB_OP_SGELSD])(
      102, *m, *n, *nrhs, a, *lda, b, *ldb, s, *rcond, rank);
  if (info) *info = rc;
}

static void thunk_sgelsd_cblas_to_fortran(int layout, int m, int n, int nrhs,
                                          float *a, int lda, float *b, int ldb,
                                          float *s, float rcond, int *rank,
                                          int *info) {
  if (fb_lapack_reject_non_col_major_cblas_layout(layout, info)) return;
  if (!g_fortran_fn[FB_OP_SGELSD]) {
    if (info) *info = -1;
    return;
  }
  int lda_ = lda, ldb_ = ldb, lwork_ = -1, info_ = 0;
  float rcond_ = rcond;
  float work_query = 0.0f;
  int iwork_query = 0;
  ((void (*)(int *, int *, int *, float *, int *, float *, int *, float *,
             float *, int *, float *, int *, int *, int *))
      g_fortran_fn[FB_OP_SGELSD])(&m, &n, &nrhs, a, &lda_, b, &ldb_, s,
                                  &rcond_, rank, &work_query, &lwork_,
                                  &iwork_query, &info_);
  if (info_ != 0) {
    if (info) *info = info_;
    return;
  }
  lwork_ = (work_query > 1.0f) ? (int)work_query : 1;
  int iwork_n = (iwork_query > 1) ? iwork_query : 1;
  float *work = (float *)malloc((size_t)lwork_ * sizeof(float));
  int *iwork = (int *)malloc((size_t)iwork_n * sizeof(int));
  if (!work || !iwork) {
    free(work);
    free(iwork);
    if (info) *info = -101;
    return;
  }
  ((void (*)(int *, int *, int *, float *, int *, float *, int *, float *,
             float *, int *, float *, int *, int *, int *))
      g_fortran_fn[FB_OP_SGELSD])(&m, &n, &nrhs, a, &lda_, b, &ldb_, s,
                                  &rcond_, rank, work, &lwork_, iwork,
                                  &info_);
  free(work);
  free(iwork);
  if (info) *info = info_;
}

static void thunk_sgelss_fortran_to_cblas(int *m, int *n, int *nrhs,
                                          float *a, int *lda, float *b,
                                          int *ldb, float *s, float *rcond,
                                          int *rank, float *work, int *lwork,
                                          int *info) {
  (void)work;
  (void)lwork;
  if (!g_cblas_fn[FB_OP_SGELSS]) {
    if (info) *info = -1;
    return;
  }
  int rc = ((int (*)(int, int, int, int, float *, int, float *, int, float *,
                     float, int *))g_cblas_fn[FB_OP_SGELSS])(
      102, *m, *n, *nrhs, a, *lda, b, *ldb, s, *rcond, rank);
  if (info) *info = rc;
}

static void thunk_dgelss_fortran_to_cblas(int *m, int *n, int *nrhs,
                                          double *a, int *lda, double *b,
                                          int *ldb, double *s, double *rcond,
                                          int *rank, double *work, int *lwork,
                                          int *info) {
  (void)work;
  (void)lwork;
  if (!g_cblas_fn[FB_OP_DGELSS]) {
    if (info) *info = -1;
    return;
  }
  int rc = ((int (*)(int, int, int, int, double *, int, double *, int,
                     double *, double, int *))g_cblas_fn[FB_OP_DGELSS])(
      102, *m, *n, *nrhs, a, *lda, b, *ldb, s, *rcond, rank);
  if (info) *info = rc;
}

static void thunk_cgelss_fortran_to_cblas(int *m, int *n, int *nrhs,
                                          fb_complex_float_t *a, int *lda,
                                          fb_complex_float_t *b, int *ldb,
                                          float *s, float *rcond, int *rank,
                                          fb_complex_float_t *work,
                                          int *lwork, float *rwork, int *info) {
  (void)work;
  (void)lwork;
  (void)rwork;
  if (!g_cblas_fn[FB_OP_CGELSS]) {
    if (info) *info = -1;
    return;
  }
  int rc = ((int (*)(int, int, int, int, fb_complex_float_t *, int,
                     fb_complex_float_t *, int, float *, float, int *))
      g_cblas_fn[FB_OP_CGELSS])(102, *m, *n, *nrhs, a, *lda, b, *ldb, s,
                                *rcond, rank);
  if (info) *info = rc;
}

static void thunk_zgelss_fortran_to_cblas(int *m, int *n, int *nrhs,
                                          fb_complex_double_t *a, int *lda,
                                          fb_complex_double_t *b, int *ldb,
                                          double *s, double *rcond, int *rank,
                                          fb_complex_double_t *work,
                                          int *lwork, double *rwork,
                                          int *info) {
  (void)work;
  (void)lwork;
  (void)rwork;
  if (!g_cblas_fn[FB_OP_ZGELSS]) {
    if (info) *info = -1;
    return;
  }
  int rc = ((int (*)(int, int, int, int, fb_complex_double_t *, int,
                     fb_complex_double_t *, int, double *, double, int *))
      g_cblas_fn[FB_OP_ZGELSS])(102, *m, *n, *nrhs, a, *lda, b, *ldb, s,
                                *rcond, rank);
  if (info) *info = rc;
}

static void thunk_sgelss_cblas_to_fortran(int layout, int m, int n, int nrhs,
                                          float *a, int lda, float *b, int ldb,
                                          float *s, float rcond, int *rank,
                                          int *info) {
  if (fb_lapack_reject_non_col_major_cblas_layout(layout, info)) return;
  if (!g_fortran_fn[FB_OP_SGELSS]) {
    if (info) *info = -1;
    return;
  }
  int lda_ = lda, ldb_ = ldb, lwork_ = -1, info_ = 0;
  float rcond_ = rcond;
  float work_query = 0.0f;
  ((void (*)(int *, int *, int *, float *, int *, float *, int *, float *,
             float *, int *, float *, int *, int *))g_fortran_fn[FB_OP_SGELSS])(
      &m, &n, &nrhs, a, &lda_, b, &ldb_, s, &rcond_, rank, &work_query,
      &lwork_, &info_);
  if (info_ != 0) {
    if (info) *info = info_;
    return;
  }
  lwork_ = (work_query > 1.0f) ? (int)work_query : 1;
  float *work = (float *)malloc((size_t)lwork_ * sizeof(float));
  if (!work) {
    if (info) *info = -101;
    return;
  }
  ((void (*)(int *, int *, int *, float *, int *, float *, int *, float *,
             float *, int *, float *, int *, int *))g_fortran_fn[FB_OP_SGELSS])(
      &m, &n, &nrhs, a, &lda_, b, &ldb_, s, &rcond_, rank, work, &lwork_,
      &info_);
  free(work);
  if (info) *info = info_;
}

static void thunk_dgelss_cblas_to_fortran(int layout, int m, int n, int nrhs,
                                          double *a, int lda, double *b,
                                          int ldb, double *s, double rcond,
                                          int *rank, int *info) {
  if (fb_lapack_reject_non_col_major_cblas_layout(layout, info)) return;
  if (!g_fortran_fn[FB_OP_DGELSS]) {
    if (info) *info = -1;
    return;
  }
  int lda_ = lda, ldb_ = ldb, lwork_ = -1, info_ = 0;
  double rcond_ = rcond;
  double work_query = 0.0;
  ((void (*)(int *, int *, int *, double *, int *, double *, int *, double *,
             double *, int *, double *, int *, int *))g_fortran_fn[FB_OP_DGELSS])(
      &m, &n, &nrhs, a, &lda_, b, &ldb_, s, &rcond_, rank, &work_query,
      &lwork_, &info_);
  if (info_ != 0) {
    if (info) *info = info_;
    return;
  }
  lwork_ = (work_query > 1.0) ? (int)work_query : 1;
  double *work = (double *)malloc((size_t)lwork_ * sizeof(double));
  if (!work) {
    if (info) *info = -101;
    return;
  }
  ((void (*)(int *, int *, int *, double *, int *, double *, int *, double *,
             double *, int *, double *, int *, int *))g_fortran_fn[FB_OP_DGELSS])(
      &m, &n, &nrhs, a, &lda_, b, &ldb_, s, &rcond_, rank, work, &lwork_,
      &info_);
  free(work);
  if (info) *info = info_;
}

static void thunk_cgelss_cblas_to_fortran(int layout, int m, int n, int nrhs,
                                          fb_complex_float_t *a, int lda,
                                          fb_complex_float_t *b, int ldb,
                                          float *s, float rcond, int *rank,
                                          int *info) {
  if (fb_lapack_reject_non_col_major_cblas_layout(layout, info)) return;
  if (!g_fortran_fn[FB_OP_CGELSS]) {
    if (info) *info = -1;
    return;
  }
  int lda_ = lda, ldb_ = ldb, lwork_ = -1, info_ = 0;
  float rcond_ = rcond;
  fb_complex_float_t work_query = 0.0f;
  float *rwork = (float *)calloc((size_t)((m < n ? m : n) > 0 ? (m < n ? m : n) : 1), sizeof(float));
  if (!rwork) {
    if (info) *info = -101;
    return;
  }
  ((void (*)(int *, int *, int *, fb_complex_float_t *, int *,
             fb_complex_float_t *, int *, float *, float *, int *,
             fb_complex_float_t *, int *, float *, int *))
      g_fortran_fn[FB_OP_CGELSS])(&m, &n, &nrhs, a, &lda_, b, &ldb_, s,
                                  &rcond_, rank, &work_query, &lwork_, rwork,
                                  &info_);
  if (info_ != 0) {
    free(rwork);
    if (info) *info = info_;
    return;
  }
  lwork_ = fb_query_size_from_cfloat(work_query);
  fb_complex_float_t *work = (fb_complex_float_t *)malloc((size_t)lwork_ * sizeof(fb_complex_float_t));
  if (!work) {
    free(rwork);
    if (info) *info = -101;
    return;
  }
  ((void (*)(int *, int *, int *, fb_complex_float_t *, int *,
             fb_complex_float_t *, int *, float *, float *, int *,
             fb_complex_float_t *, int *, float *, int *))
      g_fortran_fn[FB_OP_CGELSS])(&m, &n, &nrhs, a, &lda_, b, &ldb_, s,
                                  &rcond_, rank, work, &lwork_, rwork, &info_);
  free(work);
  free(rwork);
  if (info) *info = info_;
}

static void thunk_zgelss_cblas_to_fortran(int layout, int m, int n, int nrhs,
                                          fb_complex_double_t *a, int lda,
                                          fb_complex_double_t *b, int ldb,
                                          double *s, double rcond, int *rank,
                                          int *info) {
  if (fb_lapack_reject_non_col_major_cblas_layout(layout, info)) return;
  if (!g_fortran_fn[FB_OP_ZGELSS]) {
    if (info) *info = -1;
    return;
  }
  int lda_ = lda, ldb_ = ldb, lwork_ = -1, info_ = 0;
  double rcond_ = rcond;
  fb_complex_double_t work_query = 0.0;
  double *rwork = (double *)calloc((size_t)((m < n ? m : n) > 0 ? (m < n ? m : n) : 1), sizeof(double));
  if (!rwork) {
    if (info) *info = -101;
    return;
  }
  ((void (*)(int *, int *, int *, fb_complex_double_t *, int *,
             fb_complex_double_t *, int *, double *, double *, int *,
             fb_complex_double_t *, int *, double *, int *))
      g_fortran_fn[FB_OP_ZGELSS])(&m, &n, &nrhs, a, &lda_, b, &ldb_, s,
                                  &rcond_, rank, &work_query, &lwork_, rwork,
                                  &info_);
  if (info_ != 0) {
    free(rwork);
    if (info) *info = info_;
    return;
  }
  lwork_ = fb_query_size_from_cdouble(work_query);
  fb_complex_double_t *work = (fb_complex_double_t *)malloc((size_t)lwork_ * sizeof(fb_complex_double_t));
  if (!work) {
    free(rwork);
    if (info) *info = -101;
    return;
  }
  ((void (*)(int *, int *, int *, fb_complex_double_t *, int *,
             fb_complex_double_t *, int *, double *, double *, int *,
             fb_complex_double_t *, int *, double *, int *))
      g_fortran_fn[FB_OP_ZGELSS])(&m, &n, &nrhs, a, &lda_, b, &ldb_, s,
                                  &rcond_, rank, work, &lwork_, rwork, &info_);
  free(work);
  free(rwork);
  if (info) *info = info_;
}

/* ---- SGTSVX / DGTSVX  (FB_OP_SGTSVX=397, FB_OP_DGTSVX=398) ------------ */

static int thunk_sgtsvx_fortran_to_cblas(
    const fb_layout_t layout, const char fact, const char trans, const int n,
    const int nrhs, const float *dl, const float *d, const float *du,
    float *dlf, float *df, float *duf, float *du2, int *ipiv, const float *b,
    const int ldb, float *x, const int ldx, float *rcond, float *ferr,
    float *berr) {
  int n_ = n;
  int nrhs_ = nrhs;
  int ldb_ = (n > 0) ? n : 1;
  int ldx_ = (n > 0) ? n : 1;
  int info = 0;
  int work_len = (n > 0) ? (3 * n) : 1;
  int iwork_len = (n > 0) ? n : 1;
  size_t b_elems = (size_t)ldb_ * (size_t)((nrhs > 0) ? nrhs : 1);
  size_t x_elems = (size_t)ldx_ * (size_t)((nrhs > 0) ? nrhs : 1);
  float *b_col = NULL;
  float *x_col = NULL;
  float *work = NULL;
  int *iwork = NULL;

  if (!g_fortran_fn[FB_OP_SGTSVX]) return -1;
  if (!dl || !d || !du || !dlf || !df || !duf || !du2 || !ipiv || !b || !x ||
      !rcond || !ferr || !berr) {
    return -1;
  }

  if (layout == FB_LAYOUT_ROW_MAJOR) {
    if (ldb < nrhs || ldx < nrhs) return -15;
    b_col = (float *)malloc(b_elems * sizeof(float));
    x_col = (float *)malloc(x_elems * sizeof(float));
    if (!b_col || !x_col) {
      free(b_col);
      free(x_col);
      return -101;
    }
    if (fb_copy_matrix_to_col_major(b, layout, n, nrhs, ldb, b_col, ldb_,
                                    sizeof(float)) != 0) {
      free(b_col);
      free(x_col);
      return -1;
    }
  } else if (fb_lapack_layout_is_col_major(layout)) {
    ldb_ = ldb;
    ldx_ = ldx;
    b_col = (float *)b;
    x_col = x;
  } else {
    return -1;
  }

  work = (float *)calloc((size_t)work_len, sizeof(float));
  iwork = (int *)calloc((size_t)iwork_len, sizeof(int));
  if (!work || !iwork) {
    free(work);
    free(iwork);
    if (layout == FB_LAYOUT_ROW_MAJOR) {
      free(b_col);
      free(x_col);
    }
    return -101;
  }

  ((void (*)(char *, char *, int *, int *, const float *, const float *,
             const float *, float *, float *, float *, float *, int *, float *,
             int *, float *, int *, float *, float *, float *, float *, int *,
             int *))g_fortran_fn[FB_OP_SGTSVX])((char *)&fact, (char *)&trans,
                                                &n_, &nrhs_, dl, d, du, dlf,
                                                df, duf, du2, ipiv, b_col,
                                                &ldb_, x_col, &ldx_, rcond,
                                                ferr, berr, work, iwork,
                                                &info);

  if (info == 0 && layout == FB_LAYOUT_ROW_MAJOR &&
      fb_copy_matrix_from_col_major(x_col, n, nrhs, ldx_, layout, x, ldx,
                                    sizeof(float)) != 0) {
    info = -1;
  }

  free(work);
  free(iwork);
  if (layout == FB_LAYOUT_ROW_MAJOR) {
    free(b_col);
    free(x_col);
  }
  return info;
}

static void thunk_sgtsvx_cblas_to_fortran(
    char *fact, char *trans, int *n, int *nrhs, float *dl, float *d, float *du,
    float *dlf, float *df, float *duf, float *du2, int *ipiv, float *b,
    int *ldb, float *x, int *ldx, float *rcond, float *ferr, float *berr,
    float *work, int *iwork, int *info) {
  int rc;
  (void)work;
  (void)iwork;
  if (!fact || !trans || !n || !nrhs || !dl || !d || !du || !dlf || !df ||
      !duf || !du2 || !ipiv || !b || !ldb || !x || !ldx || !rcond || !ferr ||
      !berr) {
    if (info) *info = -1;
    return;
  }
  if (!g_cblas_fn[FB_OP_SGTSVX]) {
    if (info) *info = -1;
    return;
  }
  rc = ((int (*)(fb_layout_t, char, char, int, int, const float *,
                 const float *, const float *, float *, float *, float *,
                 float *, int *, const float *, int, float *, int, float *,
                 float *, float *))g_cblas_fn[FB_OP_SGTSVX])(
      FB_LAYOUT_COL_MAJOR, *fact, *trans, *n, *nrhs, dl, d, du, dlf, df, duf,
      du2, ipiv, b, *ldb, x, *ldx, rcond, ferr, berr);
  if (info) *info = rc;
}

static int thunk_dgtsvx_fortran_to_cblas(
    const fb_layout_t layout, const char fact, const char trans, const int n,
    const int nrhs, const double *dl, const double *d, const double *du,
    double *dlf, double *df, double *duf, double *du2, int *ipiv,
    const double *b, const int ldb, double *x, const int ldx, double *rcond,
    double *ferr, double *berr) {
  int n_ = n;
  int nrhs_ = nrhs;
  int ldb_ = (n > 0) ? n : 1;
  int ldx_ = (n > 0) ? n : 1;
  int info = 0;
  int work_len = (n > 0) ? (3 * n) : 1;
  int iwork_len = (n > 0) ? n : 1;
  size_t b_elems = (size_t)ldb_ * (size_t)((nrhs > 0) ? nrhs : 1);
  size_t x_elems = (size_t)ldx_ * (size_t)((nrhs > 0) ? nrhs : 1);
  double *b_col = NULL;
  double *x_col = NULL;
  double *work = NULL;
  int *iwork = NULL;

  if (!g_fortran_fn[FB_OP_DGTSVX]) return -1;
  if (!dl || !d || !du || !dlf || !df || !duf || !du2 || !ipiv || !b || !x ||
      !rcond || !ferr || !berr) {
    return -1;
  }

  if (layout == FB_LAYOUT_ROW_MAJOR) {
    if (ldb < nrhs || ldx < nrhs) return -15;
    b_col = (double *)malloc(b_elems * sizeof(double));
    x_col = (double *)malloc(x_elems * sizeof(double));
    if (!b_col || !x_col) {
      free(b_col);
      free(x_col);
      return -101;
    }
    if (fb_copy_matrix_to_col_major(b, layout, n, nrhs, ldb, b_col, ldb_,
                                    sizeof(double)) != 0) {
      free(b_col);
      free(x_col);
      return -1;
    }
  } else if (fb_lapack_layout_is_col_major(layout)) {
    ldb_ = ldb;
    ldx_ = ldx;
    b_col = (double *)b;
    x_col = x;
  } else {
    return -1;
  }

  work = (double *)calloc((size_t)work_len, sizeof(double));
  iwork = (int *)calloc((size_t)iwork_len, sizeof(int));
  if (!work || !iwork) {
    free(work);
    free(iwork);
    if (layout == FB_LAYOUT_ROW_MAJOR) {
      free(b_col);
      free(x_col);
    }
    return -101;
  }

  ((void (*)(char *, char *, int *, int *, const double *, const double *,
             const double *, double *, double *, double *, double *, int *,
             double *, int *, double *, int *, double *, double *, double *,
             double *, int *, int *))g_fortran_fn[FB_OP_DGTSVX])(
      (char *)&fact, (char *)&trans, &n_, &nrhs_, dl, d, du, dlf, df, duf,
      du2, ipiv, b_col, &ldb_, x_col, &ldx_, rcond, ferr, berr, work, iwork,
      &info);

  if (info == 0 && layout == FB_LAYOUT_ROW_MAJOR &&
      fb_copy_matrix_from_col_major(x_col, n, nrhs, ldx_, layout, x, ldx,
                                    sizeof(double)) != 0) {
    info = -1;
  }

  free(work);
  free(iwork);
  if (layout == FB_LAYOUT_ROW_MAJOR) {
    free(b_col);
    free(x_col);
  }
  return info;
}

static void thunk_dgtsvx_cblas_to_fortran(
    char *fact, char *trans, int *n, int *nrhs, double *dl, double *d,
    double *du, double *dlf, double *df, double *duf, double *du2, int *ipiv,
    double *b, int *ldb, double *x, int *ldx, double *rcond, double *ferr,
    double *berr, double *work, int *iwork, int *info) {
  int rc;
  (void)work;
  (void)iwork;
  if (!fact || !trans || !n || !nrhs || !dl || !d || !du || !dlf || !df ||
      !duf || !du2 || !ipiv || !b || !ldb || !x || !ldx || !rcond || !ferr ||
      !berr) {
    if (info) *info = -1;
    return;
  }
  if (!g_cblas_fn[FB_OP_DGTSVX]) {
    if (info) *info = -1;
    return;
  }
  rc = ((int (*)(fb_layout_t, char, char, int, int, const double *,
                 const double *, const double *, double *, double *, double *,
                 double *, int *, const double *, int, double *, int, double *,
                 double *, double *))g_cblas_fn[FB_OP_DGTSVX])(
      FB_LAYOUT_COL_MAJOR, *fact, *trans, *n, *nrhs, dl, d, du, dlf, df, duf,
      du2, ipiv, b, *ldb, x, *ldx, rcond, ferr, berr);
  if (info) *info = rc;
}

static int thunk_cgtsvx_fortran_to_cblas(
    const fb_layout_t layout, const char fact, const char trans, const int n,
    const int nrhs, const fb_complex_float_t *dl,
    const fb_complex_float_t *d, const fb_complex_float_t *du,
    fb_complex_float_t *dlf, fb_complex_float_t *df, fb_complex_float_t *duf,
    fb_complex_float_t *du2, int *ipiv, const fb_complex_float_t *b,
    const int ldb, fb_complex_float_t *x, const int ldx, float *rcond,
    float *ferr, float *berr) {
  int n_ = n;
  int nrhs_ = nrhs;
  int ldb_ = (n > 0) ? n : 1;
  int ldx_ = (n > 0) ? n : 1;
  int info = 0;
  int work_len = (n > 0) ? (2 * n) : 1;
  int rwork_len = (n > 1) ? n : 2;
  size_t b_elems = (size_t)ldb_ * (size_t)((nrhs > 0) ? nrhs : 1);
  size_t x_elems = (size_t)ldx_ * (size_t)((nrhs > 0) ? nrhs : 1);
  fb_complex_float_t *b_col = NULL;
  fb_complex_float_t *x_col = NULL;
  fb_complex_float_t *work = NULL;
  float *rwork = NULL;

  if (!g_fortran_fn[FB_OP_CGTSVX]) return -1;
  if (!dl || !d || !du || !dlf || !df || !duf || !du2 || !ipiv || !b || !x ||
      !rcond || !ferr || !berr) {
    return -1;
  }

  if (layout == FB_LAYOUT_ROW_MAJOR) {
    if (ldb < nrhs || ldx < nrhs) return -15;
    b_col = (fb_complex_float_t *)malloc(b_elems * sizeof(fb_complex_float_t));
    x_col = (fb_complex_float_t *)malloc(x_elems * sizeof(fb_complex_float_t));
    if (!b_col || !x_col) {
      free(b_col);
      free(x_col);
      return -101;
    }
    if (fb_copy_matrix_to_col_major(b, layout, n, nrhs, ldb, b_col, ldb_,
                                    sizeof(fb_complex_float_t)) != 0) {
      free(b_col);
      free(x_col);
      return -1;
    }
  } else if (fb_lapack_layout_is_col_major(layout)) {
    ldb_ = ldb;
    ldx_ = ldx;
    b_col = (fb_complex_float_t *)b;
    x_col = x;
  } else {
    return -1;
  }

  work = (fb_complex_float_t *)calloc((size_t)work_len,
                                      sizeof(fb_complex_float_t));
  rwork = (float *)calloc((size_t)rwork_len, sizeof(float));
  if (!work || !rwork) {
    free(work);
    free(rwork);
    if (layout == FB_LAYOUT_ROW_MAJOR) {
      free(b_col);
      free(x_col);
    }
    return -101;
  }

  ((void (*)(char *, char *, int *, int *, const fb_complex_float_t *,
             const fb_complex_float_t *, const fb_complex_float_t *,
             fb_complex_float_t *, fb_complex_float_t *, fb_complex_float_t *,
             fb_complex_float_t *, int *, fb_complex_float_t *, int *,
             fb_complex_float_t *, int *, float *, float *, float *,
             fb_complex_float_t *, float *, int *))g_fortran_fn[FB_OP_CGTSVX])(
      (char *)&fact, (char *)&trans, &n_, &nrhs_, dl, d, du, dlf, df, duf,
      du2, ipiv, b_col, &ldb_, x_col, &ldx_, rcond, ferr, berr, work, rwork,
      &info);

  if (info == 0 && layout == FB_LAYOUT_ROW_MAJOR &&
      fb_copy_matrix_from_col_major(x_col, n, nrhs, ldx_, layout, x, ldx,
                                    sizeof(fb_complex_float_t)) != 0) {
    info = -1;
  }

  free(work);
  free(rwork);
  if (layout == FB_LAYOUT_ROW_MAJOR) {
    free(b_col);
    free(x_col);
  }
  return info;
}

static void thunk_cgtsvx_cblas_to_fortran(
    char *fact, char *trans, int *n, int *nrhs, fb_complex_float_t *dl,
    fb_complex_float_t *d, fb_complex_float_t *du, fb_complex_float_t *dlf,
    fb_complex_float_t *df, fb_complex_float_t *duf, fb_complex_float_t *du2,
    int *ipiv, fb_complex_float_t *b, int *ldb, fb_complex_float_t *x,
    int *ldx, float *rcond, float *ferr, float *berr,
    fb_complex_float_t *work, float *rwork, int *info) {
  int rc;
  (void)work;
  (void)rwork;
  if (!fact || !trans || !n || !nrhs || !dl || !d || !du || !dlf || !df ||
      !duf || !du2 || !ipiv || !b || !ldb || !x || !ldx || !rcond || !ferr ||
      !berr) {
    if (info) *info = -1;
    return;
  }
  if (!g_cblas_fn[FB_OP_CGTSVX]) {
    if (info) *info = -1;
    return;
  }
  rc = ((int (*)(fb_layout_t, char, char, int, int,
                 const fb_complex_float_t *, const fb_complex_float_t *,
                 const fb_complex_float_t *, fb_complex_float_t *,
                 fb_complex_float_t *, fb_complex_float_t *,
                 fb_complex_float_t *, int *, const fb_complex_float_t *, int,
                 fb_complex_float_t *, int, float *, float *, float *))
            g_cblas_fn[FB_OP_CGTSVX])(
      FB_LAYOUT_COL_MAJOR, *fact, *trans, *n, *nrhs, dl, d, du, dlf, df, duf,
      du2, ipiv, b, *ldb, x, *ldx, rcond, ferr, berr);
  if (info) *info = rc;
}

static int thunk_zgtsvx_fortran_to_cblas(
    const fb_layout_t layout, const char fact, const char trans, const int n,
    const int nrhs, const fb_complex_double_t *dl,
    const fb_complex_double_t *d, const fb_complex_double_t *du,
    fb_complex_double_t *dlf, fb_complex_double_t *df,
    fb_complex_double_t *duf, fb_complex_double_t *du2, int *ipiv,
    const fb_complex_double_t *b, const int ldb, fb_complex_double_t *x,
    const int ldx, double *rcond, double *ferr, double *berr) {
  int n_ = n;
  int nrhs_ = nrhs;
  int ldb_ = (n > 0) ? n : 1;
  int ldx_ = (n > 0) ? n : 1;
  int info = 0;
  int work_len = (n > 0) ? (2 * n) : 1;
  int rwork_len = (n > 1) ? n : 2;
  size_t b_elems = (size_t)ldb_ * (size_t)((nrhs > 0) ? nrhs : 1);
  size_t x_elems = (size_t)ldx_ * (size_t)((nrhs > 0) ? nrhs : 1);
  fb_complex_double_t *b_col = NULL;
  fb_complex_double_t *x_col = NULL;
  fb_complex_double_t *work = NULL;
  double *rwork = NULL;

  if (!g_fortran_fn[FB_OP_ZGTSVX]) return -1;
  if (!dl || !d || !du || !dlf || !df || !duf || !du2 || !ipiv || !b || !x ||
      !rcond || !ferr || !berr) {
    return -1;
  }

  if (layout == FB_LAYOUT_ROW_MAJOR) {
    if (ldb < nrhs || ldx < nrhs) return -15;
    b_col = (fb_complex_double_t *)malloc(b_elems * sizeof(fb_complex_double_t));
    x_col = (fb_complex_double_t *)malloc(x_elems * sizeof(fb_complex_double_t));
    if (!b_col || !x_col) {
      free(b_col);
      free(x_col);
      return -101;
    }
    if (fb_copy_matrix_to_col_major(b, layout, n, nrhs, ldb, b_col, ldb_,
                                    sizeof(fb_complex_double_t)) != 0) {
      free(b_col);
      free(x_col);
      return -1;
    }
  } else if (fb_lapack_layout_is_col_major(layout)) {
    ldb_ = ldb;
    ldx_ = ldx;
    b_col = (fb_complex_double_t *)b;
    x_col = x;
  } else {
    return -1;
  }

  work = (fb_complex_double_t *)calloc((size_t)work_len,
                                       sizeof(fb_complex_double_t));
  rwork = (double *)calloc((size_t)rwork_len, sizeof(double));
  if (!work || !rwork) {
    free(work);
    free(rwork);
    if (layout == FB_LAYOUT_ROW_MAJOR) {
      free(b_col);
      free(x_col);
    }
    return -101;
  }

  ((void (*)(char *, char *, int *, int *, const fb_complex_double_t *,
             const fb_complex_double_t *, const fb_complex_double_t *,
             fb_complex_double_t *, fb_complex_double_t *,
             fb_complex_double_t *, fb_complex_double_t *, int *,
             fb_complex_double_t *, int *, fb_complex_double_t *, int *,
             double *, double *, double *, fb_complex_double_t *, double *,
             int *))g_fortran_fn[FB_OP_ZGTSVX])(
      (char *)&fact, (char *)&trans, &n_, &nrhs_, dl, d, du, dlf, df, duf,
      du2, ipiv, b_col, &ldb_, x_col, &ldx_, rcond, ferr, berr, work, rwork,
      &info);

  if (info == 0 && layout == FB_LAYOUT_ROW_MAJOR &&
      fb_copy_matrix_from_col_major(x_col, n, nrhs, ldx_, layout, x, ldx,
                                    sizeof(fb_complex_double_t)) != 0) {
    info = -1;
  }

  free(work);
  free(rwork);
  if (layout == FB_LAYOUT_ROW_MAJOR) {
    free(b_col);
    free(x_col);
  }
  return info;
}

static void thunk_zgtsvx_cblas_to_fortran(
    char *fact, char *trans, int *n, int *nrhs, fb_complex_double_t *dl,
    fb_complex_double_t *d, fb_complex_double_t *du, fb_complex_double_t *dlf,
    fb_complex_double_t *df, fb_complex_double_t *duf,
    fb_complex_double_t *du2, int *ipiv, fb_complex_double_t *b, int *ldb,
    fb_complex_double_t *x, int *ldx, double *rcond, double *ferr,
    double *berr, fb_complex_double_t *work, double *rwork, int *info) {
  int rc;
  (void)work;
  (void)rwork;
  if (!fact || !trans || !n || !nrhs || !dl || !d || !du || !dlf || !df ||
      !duf || !du2 || !ipiv || !b || !ldb || !x || !ldx || !rcond || !ferr ||
      !berr) {
    if (info) *info = -1;
    return;
  }
  if (!g_cblas_fn[FB_OP_ZGTSVX]) {
    if (info) *info = -1;
    return;
  }
  rc = ((int (*)(fb_layout_t, char, char, int, int,
                 const fb_complex_double_t *, const fb_complex_double_t *,
                 const fb_complex_double_t *, fb_complex_double_t *,
                 fb_complex_double_t *, fb_complex_double_t *,
                 fb_complex_double_t *, int *, const fb_complex_double_t *,
                 int, fb_complex_double_t *, int, double *, double *,
                 double *))g_cblas_fn[FB_OP_ZGTSVX])(
      FB_LAYOUT_COL_MAJOR, *fact, *trans, *n, *nrhs, dl, d, du, dlf, df, duf,
      du2, ipiv, b, *ldb, x, *ldx, rcond, ferr, berr);
  if (info) *info = rc;
}

/* ---- ORG2L/UNG2L  unblocked Q-generator -------------------------------- */

#define DEFINE_LAPACK_UNBLOCKED_MATRIX_K_TAU_THUNKS(OP, TYPE, OP_ID,          \
                                                   WORK_LEN_EXPR)             \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const int m, const int n, const int k,           \
    TYPE *A, const int lda, const TYPE *tau) {                                 \
  int m_ = m, n_ = n, k_ = k, lda_ = (m > 0) ? m : 1, info = 0;               \
  int tau_alloc = (k > 0) ? k : 1;                                             \
  int work_len = ((WORK_LEN_EXPR) > 0) ? (WORK_LEN_EXPR) : 1;                 \
  size_t a_elems = (size_t)lda_ * (size_t)((n > 0) ? n : 1);                  \
  TYPE *a_col = NULL;                                                           \
  TYPE *tau_copy = NULL;                                                        \
  TYPE *work = NULL;                                                            \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if ((m > 0 && n > 0 && !A) || (k > 0 && !tau)) return -1;                    \
  a_col = (TYPE *)malloc(a_elems * sizeof(TYPE));                               \
  tau_copy = (TYPE *)malloc((size_t)tau_alloc * sizeof(TYPE));                  \
  work = (TYPE *)malloc((size_t)work_len * sizeof(TYPE));                       \
  if (!a_col || !tau_copy || !work) {                                           \
    free(a_col);                                                                \
    free(tau_copy);                                                             \
    free(work);                                                                 \
    return -101;                                                                \
  }                                                                             \
  if (m > 0 && n > 0 &&                                                         \
      fb_copy_matrix_to_col_major(A, layout, m, n, lda,                        \
                                  a_col, lda_, sizeof(TYPE)) != 0) {          \
    free(a_col);                                                                \
    free(tau_copy);                                                             \
    free(work);                                                                 \
    return -1;                                                                  \
  }                                                                             \
  memset(tau_copy, 0, (size_t)tau_alloc * sizeof(TYPE));                        \
  if (k > 0) {                                                                  \
    memcpy(tau_copy, tau, (size_t)k * sizeof(TYPE));                            \
  }                                                                             \
  ((void (*)(int *, int *, int *, TYPE *, int *, TYPE *, TYPE *, int *))      \
      g_fortran_fn[OP_ID])(                                                     \
      &m_, &n_, &k_, a_col, &lda_, tau_copy, work, &info);                     \
  if (m > 0 && n > 0 &&                                                         \
      fb_copy_matrix_from_col_major(a_col, m, n, lda_, layout,                 \
                                    A, lda, sizeof(TYPE)) != 0) {             \
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
    int *info) {                                                                \
  (void)work;                                                                   \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, int, int, int, TYPE *, int, const TYPE *))  \
      g_cblas_fn[OP_ID])(FB_LAYOUT_COL_MAJOR, *m, *n, *k, A, *lda, tau);       \
  if (info) *info = rc;                                                         \
}

DEFINE_LAPACK_UNBLOCKED_MATRIX_K_TAU_THUNKS(sorg2l, float, FB_OP_SORG2L, n)
DEFINE_LAPACK_UNBLOCKED_MATRIX_K_TAU_THUNKS(dorg2l, double, FB_OP_DORG2L, n)
DEFINE_LAPACK_UNBLOCKED_MATRIX_K_TAU_THUNKS(cung2l, fb_complex_float_t,
                                            FB_OP_CUNG2L, n)
DEFINE_LAPACK_UNBLOCKED_MATRIX_K_TAU_THUNKS(zung2l, fb_complex_double_t,
                                            FB_OP_ZUNG2L, n)
DEFINE_LAPACK_UNBLOCKED_MATRIX_K_TAU_THUNKS(sorgl2, float, FB_OP_SORGL2, m)
DEFINE_LAPACK_UNBLOCKED_MATRIX_K_TAU_THUNKS(dorgl2, double, FB_OP_DORGL2, m)
DEFINE_LAPACK_UNBLOCKED_MATRIX_K_TAU_THUNKS(cungl2, fb_complex_float_t,
                                            FB_OP_CUNGL2, m)
DEFINE_LAPACK_UNBLOCKED_MATRIX_K_TAU_THUNKS(zungl2, fb_complex_double_t,
                                            FB_OP_ZUNGL2, m)
DEFINE_LAPACK_UNBLOCKED_MATRIX_K_TAU_THUNKS(sorgr2, float, FB_OP_SORGR2, m)
DEFINE_LAPACK_UNBLOCKED_MATRIX_K_TAU_THUNKS(dorgr2, double, FB_OP_DORGR2, m)
DEFINE_LAPACK_UNBLOCKED_MATRIX_K_TAU_THUNKS(cungr2, fb_complex_float_t,
                                            FB_OP_CUNGR2, m)
DEFINE_LAPACK_UNBLOCKED_MATRIX_K_TAU_THUNKS(zungr2, fb_complex_double_t,
                                            FB_OP_ZUNGR2, m)

#undef DEFINE_LAPACK_UNBLOCKED_MATRIX_K_TAU_THUNKS

/* ---- LATRZ  unblocked RZ factorization helper -------------------------- */

#define DEFINE_LAPACK_LATRZ_THUNKS(OP, TYPE, OP_ID)                           \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const int m, const int n, const int l,          \
    TYPE *A, const int lda, TYPE *tau) {                                       \
  int m_ = m, n_ = n, l_ = l, lda_ = (m > 0) ? m : 1;                          \
  int tau_len = (m > 0) ? m : 0;                                               \
  int tau_alloc = (tau_len > 0) ? tau_len : 1;                                 \
  int work_len = (m > 0) ? m : 1;                                              \
  size_t a_elems = (size_t)lda_ * (size_t)((n > 0) ? n : 1);                   \
  TYPE *a_col = NULL;                                                           \
  TYPE *tau_copy = NULL;                                                        \
  TYPE *work = NULL;                                                            \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if ((m > 0 && n > 0 && !A) || (m > 0 && !tau)) return -1;                    \
  a_col = (TYPE *)malloc(a_elems * sizeof(TYPE));                               \
  tau_copy = (TYPE *)malloc((size_t)tau_alloc * sizeof(TYPE));                  \
  work = (TYPE *)malloc((size_t)work_len * sizeof(TYPE));                       \
  if (!a_col || !tau_copy || !work) {                                           \
    free(a_col);                                                                \
    free(tau_copy);                                                             \
    free(work);                                                                 \
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
  ((void (*)(int *, int *, int *, TYPE *, int *, TYPE *, TYPE *))              \
      g_fortran_fn[OP_ID])(&m_, &n_, &l_, a_col, &lda_, tau_copy, work);       \
  if (m > 0 && n > 0 &&                                                         \
      fb_copy_matrix_from_col_major(a_col, m, n, lda_, layout,                 \
                                    A, lda, sizeof(TYPE)) != 0) {              \
    free(a_col);                                                                \
    free(tau_copy);                                                             \
    free(work);                                                                 \
    return -1;                                                                  \
  }                                                                             \
  if (tau_len > 0) {                                                            \
    memcpy(tau, tau_copy, (size_t)tau_len * sizeof(TYPE));                      \
  }                                                                             \
  free(a_col);                                                                  \
  free(tau_copy);                                                               \
  free(work);                                                                   \
  return 0;                                                                     \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    int *m, int *n, int *l, TYPE *A, int *lda, TYPE *tau, TYPE *work) {        \
  (void)work;                                                                   \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    return;                                                                     \
  }                                                                             \
  (void)((int (*)(fb_layout_t, int, int, int, TYPE *, int, TYPE *))            \
      g_cblas_fn[OP_ID])(FB_LAYOUT_COL_MAJOR, *m, *n, *l, A, *lda, tau);       \
}

DEFINE_LAPACK_LATRZ_THUNKS(slatrz, float, FB_OP_SLATRZ)
DEFINE_LAPACK_LATRZ_THUNKS(dlatrz, double, FB_OP_DLATRZ)
DEFINE_LAPACK_LATRZ_THUNKS(clatrz, fb_complex_float_t, FB_OP_CLATRZ)
DEFINE_LAPACK_LATRZ_THUNKS(zlatrz, fb_complex_double_t, FB_OP_ZLATRZ)

#undef DEFINE_LAPACK_LATRZ_THUNKS

/* ---- ORGTR/UNGTR  tridiagonal-reduction generator ---------------------- */

#define DEFINE_LAPACK_TRIDIAG_GENERATOR_THUNKS(OP, TYPE, OP_ID, QUERY_FN)     \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, char uplo, const int n, TYPE *A,                 \
    const int lda, const TYPE *tau) {                                          \
  fb_uplo_t uplo_enum = FB_UPPER;                                               \
  char uplo_char = 'U';                                                         \
  int n_ = n, lda_ = (n > 0) ? n : 1, lwork_ = -1, info = 0;                   \
  int tau_len = (n > 1) ? (n - 1) : 0, tau_alloc = (tau_len > 0) ? tau_len : 1;\
  size_t a_elems = 0;                                                           \
  TYPE work_query = (TYPE)0;                                                    \
  TYPE *a_col = NULL;                                                           \
  TYPE *tau_copy = NULL;                                                        \
  TYPE *work = NULL;                                                            \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (fb_lapack_char_to_uplo(uplo, &uplo_enum) != 0) return -1;                \
  uplo_char = fb_lapack_uplo_from_layout(layout, uplo_enum);                   \
  if ((n > 0 && !A) || (tau_len > 0 && !tau)) return -1;                       \
  a_elems = (size_t)lda_ * (size_t)((n > 0) ? n : 1);                          \
  a_col = (TYPE *)malloc(a_elems * sizeof(TYPE));                               \
  tau_copy = (TYPE *)malloc((size_t)tau_alloc * sizeof(TYPE));                  \
  if (!a_col || !tau_copy) {                                                    \
    free(a_col);                                                                \
    free(tau_copy);                                                             \
    return -101;                                                                \
  }                                                                             \
  if (n > 0 &&                                                                  \
      fb_copy_matrix_to_col_major(A, layout, n, n, lda,                        \
                                  a_col, lda_, sizeof(TYPE)) != 0) {          \
    free(a_col);                                                                \
    free(tau_copy);                                                             \
    return -1;                                                                  \
  }                                                                             \
  memset(tau_copy, 0, (size_t)tau_alloc * sizeof(TYPE));                        \
  if (tau_len > 0) {                                                            \
    memcpy(tau_copy, tau, (size_t)tau_len * sizeof(TYPE));                      \
  }                                                                             \
  ((void (*)(char *, int *, TYPE *, int *, TYPE *, TYPE *, int *, int *))     \
      g_fortran_fn[OP_ID])(                                                     \
      &uplo_char, &n_, a_col, &lda_, tau_copy, &work_query, &lwork_, &info);   \
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
  if (n > 0 &&                                                                  \
      fb_copy_matrix_to_col_major(A, layout, n, n, lda,                        \
                                  a_col, lda_, sizeof(TYPE)) != 0) {          \
    free(a_col);                                                                \
    free(tau_copy);                                                             \
    free(work);                                                                 \
    return -1;                                                                  \
  }                                                                             \
  memset(tau_copy, 0, (size_t)tau_alloc * sizeof(TYPE));                        \
  if (tau_len > 0) {                                                            \
    memcpy(tau_copy, tau, (size_t)tau_len * sizeof(TYPE));                      \
  }                                                                             \
  ((void (*)(char *, int *, TYPE *, int *, TYPE *, TYPE *, int *, int *))     \
      g_fortran_fn[OP_ID])(                                                     \
      &uplo_char, &n_, a_col, &lda_, tau_copy, work, &lwork_, &info);          \
  if (n > 0 &&                                                                  \
      fb_copy_matrix_from_col_major(a_col, n, n, lda_, layout,                 \
                                    A, lda, sizeof(TYPE)) != 0) {             \
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
    char *uplo, int *n, TYPE *A, int *lda, TYPE *tau, TYPE *work,              \
    int *lwork, int *info) {                                                    \
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
  int rc = ((int (*)(fb_layout_t, char, int, TYPE *, int, const TYPE *))      \
      g_cblas_fn[OP_ID])(FB_LAYOUT_COL_MAJOR, uplo_char, *n, A, *lda, tau);    \
  if (info) *info = rc;                                                         \
}

DEFINE_LAPACK_TRIDIAG_GENERATOR_THUNKS(sorgtr, float, FB_OP_SORGTR,
                                       fb_query_size_from_float)
DEFINE_LAPACK_TRIDIAG_GENERATOR_THUNKS(dorgtr, double, FB_OP_DORGTR,
                                       fb_query_size_from_double)
DEFINE_LAPACK_TRIDIAG_GENERATOR_THUNKS(cungtr, fb_complex_float_t,
                                       FB_OP_CUNGTR,
                                       fb_query_size_from_cfloat)
DEFINE_LAPACK_TRIDIAG_GENERATOR_THUNKS(zungtr, fb_complex_double_t,
                                       FB_OP_ZUNGTR,
                                       fb_query_size_from_cdouble)

#undef DEFINE_LAPACK_TRIDIAG_GENERATOR_THUNKS

/* ---- ORGHR/UNGHR  Hessenberg generator --------------------------------- */

#define DEFINE_LAPACK_HESSENBERG_GENERATOR_THUNKS(OP, TYPE, OP_ID, QUERY_FN)  \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const int n, const int ilo, const int ihi,       \
    TYPE *A, const int lda, const TYPE *tau) {                                 \
  int n_ = n, ilo_ = ilo, ihi_ = ihi, lda_ = (n > 0) ? n : 1;                 \
  int lwork_ = -1, info = 0, tau_len = 0, tau_alloc = 1;                       \
  size_t a_elems = 0;                                                           \
  TYPE work_query = (TYPE)0;                                                    \
  TYPE *a_col = NULL;                                                           \
  TYPE *tau_copy = NULL;                                                        \
  TYPE *work = NULL;                                                            \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  tau_len = (ihi > ilo) ? (ihi - ilo) : 0;                                      \
  tau_alloc = (tau_len > 0) ? tau_len : 1;                                      \
  if ((n > 0 && !A) || (tau_len > 0 && !tau)) return -1;                       \
  a_elems = (size_t)lda_ * (size_t)((n > 0) ? n : 1);                          \
  a_col = (TYPE *)malloc(a_elems * sizeof(TYPE));                               \
  tau_copy = (TYPE *)malloc((size_t)tau_alloc * sizeof(TYPE));                  \
  if (!a_col || !tau_copy) {                                                    \
    free(a_col);                                                                \
    free(tau_copy);                                                             \
    return -101;                                                                \
  }                                                                             \
  if (n > 0 &&                                                                  \
      fb_copy_matrix_to_col_major(A, layout, n, n, lda,                        \
                                  a_col, lda_, sizeof(TYPE)) != 0) {          \
    free(a_col);                                                                \
    free(tau_copy);                                                             \
    return -1;                                                                  \
  }                                                                             \
  memset(tau_copy, 0, (size_t)tau_alloc * sizeof(TYPE));                        \
  if (tau_len > 0) {                                                            \
    memcpy(tau_copy, tau, (size_t)tau_len * sizeof(TYPE));                      \
  }                                                                             \
  ((void (*)(int *, int *, int *, TYPE *, int *, TYPE *, TYPE *, int *,       \
             int *))g_fortran_fn[OP_ID])(                                      \
      &n_, &ilo_, &ihi_, a_col, &lda_, tau_copy, &work_query, &lwork_, &info); \
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
  if (n > 0 &&                                                                  \
      fb_copy_matrix_to_col_major(A, layout, n, n, lda,                        \
                                  a_col, lda_, sizeof(TYPE)) != 0) {          \
    free(a_col);                                                                \
    free(tau_copy);                                                             \
    free(work);                                                                 \
    return -1;                                                                  \
  }                                                                             \
  memset(tau_copy, 0, (size_t)tau_alloc * sizeof(TYPE));                        \
  if (tau_len > 0) {                                                            \
    memcpy(tau_copy, tau, (size_t)tau_len * sizeof(TYPE));                      \
  }                                                                             \
  ((void (*)(int *, int *, int *, TYPE *, int *, TYPE *, TYPE *, int *,       \
             int *))g_fortran_fn[OP_ID])(                                      \
      &n_, &ilo_, &ihi_, a_col, &lda_, tau_copy, work, &lwork_, &info);        \
  if (n > 0 &&                                                                  \
      fb_copy_matrix_from_col_major(a_col, n, n, lda_, layout,                 \
                                    A, lda, sizeof(TYPE)) != 0) {             \
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
    int *n, int *ilo, int *ihi, TYPE *A, int *lda, TYPE *tau,                  \
    TYPE *work, int *lwork, int *info) {                                        \
  (void)work;                                                                   \
  (void)lwork;                                                                  \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, int, int, int, TYPE *, int, const TYPE *))  \
      g_cblas_fn[OP_ID])(FB_LAYOUT_COL_MAJOR, *n, *ilo, *ihi, A, *lda, tau);   \
  if (info) *info = rc;                                                         \
}

DEFINE_LAPACK_HESSENBERG_GENERATOR_THUNKS(sorghr, float, FB_OP_SORGHR,
                                          fb_query_size_from_float)
DEFINE_LAPACK_HESSENBERG_GENERATOR_THUNKS(dorghr, double, FB_OP_DORGHR,
                                          fb_query_size_from_double)
DEFINE_LAPACK_HESSENBERG_GENERATOR_THUNKS(cunghr, fb_complex_float_t,
                                          FB_OP_CUNGHR,
                                          fb_query_size_from_cfloat)
DEFINE_LAPACK_HESSENBERG_GENERATOR_THUNKS(zunghr, fb_complex_double_t,
                                          FB_OP_ZUNGHR,
                                          fb_query_size_from_cdouble)

#undef DEFINE_LAPACK_HESSENBERG_GENERATOR_THUNKS

/* ---- GEHRD Hessenberg reduction --------------------------------------- */

#define DEFINE_LAPACK_HESSENBERG_REDUCTION_NWORK_THUNKS(OP, TYPE, OP_ID)      \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const int n, const int ilo, const int ihi,       \
    TYPE *A, const int lda, TYPE *tau) {                                       \
  int n_ = n, ilo_ = ilo, ihi_ = ihi, lda_ = (n > 0) ? n : 1, info = 0;       \
  int tau_len = 0, tau_alloc = 1, work_len = (n > 0) ? n : 1;                 \
  size_t a_elems = 0;                                                           \
  TYPE *a_col = NULL;                                                           \
  TYPE *tau_copy = NULL;                                                        \
  TYPE *work = NULL;                                                            \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  tau_len = (ihi > ilo) ? (ihi - ilo) : 0;                                      \
  tau_alloc = (tau_len > 0) ? tau_len : 1;                                      \
  if ((n > 0 && !A) || (tau_len > 0 && !tau)) return -1;                       \
  a_elems = (size_t)lda_ * (size_t)((n > 0) ? n : 1);                          \
  a_col = (TYPE *)malloc(a_elems * sizeof(TYPE));                               \
  tau_copy = (TYPE *)malloc((size_t)tau_alloc * sizeof(TYPE));                  \
  work = (TYPE *)malloc((size_t)work_len * sizeof(TYPE));                       \
  if (!a_col || !tau_copy || !work) {                                           \
    free(a_col);                                                                \
    free(tau_copy);                                                             \
    free(work);                                                                 \
    return -101;                                                                \
  }                                                                             \
  if (n > 0 &&                                                                  \
      fb_copy_matrix_to_col_major(A, layout, n, n, lda,                        \
                                  a_col, lda_, sizeof(TYPE)) != 0) {          \
    free(a_col);                                                                \
    free(tau_copy);                                                             \
    free(work);                                                                 \
    return -1;                                                                  \
  }                                                                             \
  memset(tau_copy, 0, (size_t)tau_alloc * sizeof(TYPE));                        \
  ((void (*)(int *, int *, int *, TYPE *, int *, TYPE *, TYPE *, int *))      \
      g_fortran_fn[OP_ID])(                                                     \
      &n_, &ilo_, &ihi_, a_col, &lda_, tau_copy, work, &info);                 \
  if (n > 0 &&                                                                  \
      fb_copy_matrix_from_col_major(a_col, n, n, lda_, layout,                 \
                                    A, lda, sizeof(TYPE)) != 0) {             \
    free(a_col);                                                                \
    free(tau_copy);                                                             \
    free(work);                                                                 \
    return -1;                                                                  \
  }                                                                             \
  if (tau_len > 0) {                                                            \
    memcpy(tau, tau_copy, (size_t)tau_len * sizeof(TYPE));                      \
  }                                                                             \
  free(a_col);                                                                  \
  free(tau_copy);                                                               \
  free(work);                                                                   \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    int *n, int *ilo, int *ihi, TYPE *A, int *lda, TYPE *tau,                  \
    TYPE *work, int *info) {                                                    \
  (void)work;                                                                   \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, int, int, int, TYPE *, int, TYPE *))        \
      g_cblas_fn[OP_ID])(FB_LAYOUT_COL_MAJOR, *n, *ilo, *ihi, A, *lda, tau);   \
  if (info) *info = rc;                                                         \
}

DEFINE_LAPACK_HESSENBERG_REDUCTION_NWORK_THUNKS(sgehd2, float, FB_OP_SGEHD2)
DEFINE_LAPACK_HESSENBERG_REDUCTION_NWORK_THUNKS(dgehd2, double, FB_OP_DGEHD2)
DEFINE_LAPACK_HESSENBERG_REDUCTION_NWORK_THUNKS(cgehd2, fb_complex_float_t,
                                                FB_OP_CGEHD2)
DEFINE_LAPACK_HESSENBERG_REDUCTION_NWORK_THUNKS(zgehd2, fb_complex_double_t,
                                                FB_OP_ZGEHD2)

#undef DEFINE_LAPACK_HESSENBERG_REDUCTION_NWORK_THUNKS

#define DEFINE_LAPACK_HESSENBERG_REDUCTION_QUERY_THUNKS(OP, TYPE, OP_ID,     \
                                                       QUERY_FN)             \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const int n, const int ilo, const int ihi,       \
    TYPE *A, const int lda, TYPE *tau) {                                       \
  int n_ = n, ilo_ = ilo, ihi_ = ihi, lda_ = (n > 0) ? n : 1;                 \
  int lwork_ = -1, info = 0, tau_len = 0, tau_alloc = 1;                       \
  size_t a_elems = 0;                                                           \
  TYPE work_query = (TYPE)0;                                                    \
  TYPE *a_col = NULL;                                                           \
  TYPE *tau_copy = NULL;                                                        \
  TYPE *work = NULL;                                                            \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  tau_len = (ihi > ilo) ? (ihi - ilo) : 0;                                      \
  tau_alloc = (tau_len > 0) ? tau_len : 1;                                      \
  if ((n > 0 && !A) || (tau_len > 0 && !tau)) return -1;                       \
  a_elems = (size_t)lda_ * (size_t)((n > 0) ? n : 1);                          \
  a_col = (TYPE *)malloc(a_elems * sizeof(TYPE));                               \
  tau_copy = (TYPE *)malloc((size_t)tau_alloc * sizeof(TYPE));                  \
  if (!a_col || !tau_copy) {                                                    \
    free(a_col);                                                                \
    free(tau_copy);                                                             \
    return -101;                                                                \
  }                                                                             \
  if (n > 0 &&                                                                  \
      fb_copy_matrix_to_col_major(A, layout, n, n, lda,                        \
                                  a_col, lda_, sizeof(TYPE)) != 0) {          \
    free(a_col);                                                                \
    free(tau_copy);                                                             \
    return -1;                                                                  \
  }                                                                             \
  memset(tau_copy, 0, (size_t)tau_alloc * sizeof(TYPE));                        \
  ((void (*)(int *, int *, int *, TYPE *, int *, TYPE *, TYPE *, int *,       \
             int *))g_fortran_fn[OP_ID])(                                      \
      &n_, &ilo_, &ihi_, a_col, &lda_, tau_copy, &work_query, &lwork_, &info); \
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
  if (n > 0 &&                                                                  \
      fb_copy_matrix_to_col_major(A, layout, n, n, lda,                        \
                                  a_col, lda_, sizeof(TYPE)) != 0) {          \
    free(a_col);                                                                \
    free(tau_copy);                                                             \
    free(work);                                                                 \
    return -1;                                                                  \
  }                                                                             \
  memset(tau_copy, 0, (size_t)tau_alloc * sizeof(TYPE));                        \
  ((void (*)(int *, int *, int *, TYPE *, int *, TYPE *, TYPE *, int *,       \
             int *))g_fortran_fn[OP_ID])(                                      \
      &n_, &ilo_, &ihi_, a_col, &lda_, tau_copy, work, &lwork_, &info);        \
  if (n > 0 &&                                                                  \
      fb_copy_matrix_from_col_major(a_col, n, n, lda_, layout,                 \
                                    A, lda, sizeof(TYPE)) != 0) {             \
    free(a_col);                                                                \
    free(tau_copy);                                                             \
    free(work);                                                                 \
    return -1;                                                                  \
  }                                                                             \
  if (tau_len > 0) {                                                            \
    memcpy(tau, tau_copy, (size_t)tau_len * sizeof(TYPE));                      \
  }                                                                             \
  free(a_col);                                                                  \
  free(tau_copy);                                                               \
  free(work);                                                                   \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    int *n, int *ilo, int *ihi, TYPE *A, int *lda, TYPE *tau,                  \
    TYPE *work, int *lwork, int *info) {                                        \
  (void)work;                                                                   \
  (void)lwork;                                                                  \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, int, int, int, TYPE *, int, TYPE *))        \
      g_cblas_fn[OP_ID])(FB_LAYOUT_COL_MAJOR, *n, *ilo, *ihi, A, *lda, tau);   \
  if (info) *info = rc;                                                         \
}

DEFINE_LAPACK_HESSENBERG_REDUCTION_QUERY_THUNKS(sgehrd, float, FB_OP_SGEHRD,
                                                fb_query_size_from_float)
DEFINE_LAPACK_HESSENBERG_REDUCTION_QUERY_THUNKS(dgehrd, double, FB_OP_DGEHRD,
                                                fb_query_size_from_double)
DEFINE_LAPACK_HESSENBERG_REDUCTION_QUERY_THUNKS(cgehrd, fb_complex_float_t,
                                                FB_OP_CGEHRD,
                                                fb_query_size_from_cfloat)
DEFINE_LAPACK_HESSENBERG_REDUCTION_QUERY_THUNKS(zgehrd, fb_complex_double_t,
                                                FB_OP_ZGEHRD,
                                                fb_query_size_from_cdouble)

#undef DEFINE_LAPACK_HESSENBERG_REDUCTION_QUERY_THUNKS

/* ---- ORMHR/UNMHR  Hessenberg reflector application --------------------- */

#define DEFINE_LAPACK_HESSENBERG_APPLY_THUNKS(OP, TYPE, OP_ID, QUERY_FN)      \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, char side, char trans,                           \
    const int m, const int n, const int ilo, const int ihi,                    \
    const TYPE *A, const int lda, const TYPE *tau, TYPE *C, const int ldc) {   \
  fb_side_t side_enum;                                                          \
  int m_ = m, n_ = n, ilo_ = ilo, ihi_ = ihi;                                   \
  int nq = 0, lda_ = 1, ldc_ = (m > 0) ? m : 1, lwork_ = -1, info = 0;         \
  int tau_len = 0, tau_alloc = 1, layout_info = 0;                             \
  size_t a_elems = 0, c_elems = 0;                                              \
  TYPE work_query = (TYPE)0;                                                    \
  TYPE *a_col = NULL;                                                           \
  TYPE *tau_copy = NULL;                                                        \
  TYPE *c_col = NULL;                                                           \
  TYPE *work = NULL;                                                            \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  if (fb_lapack_reject_non_col_major_cblas_layout(layout, &layout_info))       \
    return layout_info;                                                         \
  if (fb_lapack_char_to_side(side, &side_enum) != 0) return -1;                \
  nq = (side_enum == FB_LEFT) ? m : n;                                          \
  tau_len = (ihi > ilo) ? (ihi - ilo) : 0;                                      \
  tau_alloc = (tau_len > 0) ? tau_len : 1;                                      \
  if ((nq > 0 && !A) || (tau_len > 0 && !tau) || (m > 0 && n > 0 && !C))      \
    return -1;                                                                  \
  lda_ = (nq > 0) ? nq : 1;                                                     \
  a_elems = (size_t)lda_ * (size_t)((nq > 0) ? nq : 1);                        \
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
  if ((nq > 0 &&                                                                 \
       fb_copy_matrix_to_col_major(A, layout, nq, nq, lda,                     \
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
  if (tau_len > 0) {                                                            \
    memcpy(tau_copy, tau, (size_t)tau_len * sizeof(TYPE));                      \
  }                                                                             \
  ((void (*)(char *, char *, int *, int *, int *, int *, TYPE *, int *,        \
             TYPE *, TYPE *, int *, TYPE *, int *, int *))                     \
      g_fortran_fn[OP_ID])(                                                     \
      &side, &trans, &m_, &n_, &ilo_, &ihi_, a_col, &lda_, tau_copy, c_col,    \
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
  if ((nq > 0 &&                                                                 \
       fb_copy_matrix_to_col_major(A, layout, nq, nq, lda,                     \
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
  if (tau_len > 0) {                                                            \
    memcpy(tau_copy, tau, (size_t)tau_len * sizeof(TYPE));                      \
  }                                                                             \
  ((void (*)(char *, char *, int *, int *, int *, int *, TYPE *, int *,        \
             TYPE *, TYPE *, int *, TYPE *, int *, int *))                     \
      g_fortran_fn[OP_ID])(                                                     \
      &side, &trans, &m_, &n_, &ilo_, &ihi_, a_col, &lda_, tau_copy, c_col,    \
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
    char *side, char *trans, int *m, int *n, int *ilo, int *ihi,               \
    TYPE *A, int *lda, TYPE *tau, TYPE *C, int *ldc,                           \
    TYPE *work, int *lwork, int *info) {                                        \
  (void)work;                                                                   \
  (void)lwork;                                                                  \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, char, char, int, int, int, int,              \
                     const TYPE *, int, const TYPE *, TYPE *, int))            \
      g_cblas_fn[OP_ID])(FB_LAYOUT_COL_MAJOR, *side, *trans, *m, *n,           \
                          *ilo, *ihi, A, *lda, tau, C, *ldc);                   \
  if (info) *info = rc;                                                         \
}

DEFINE_LAPACK_HESSENBERG_APPLY_THUNKS(sormhr, float, FB_OP_SORMHR,
                                      fb_query_size_from_float)
DEFINE_LAPACK_HESSENBERG_APPLY_THUNKS(dormhr, double, FB_OP_DORMHR,
                                      fb_query_size_from_double)
DEFINE_LAPACK_HESSENBERG_APPLY_THUNKS(cunmhr, fb_complex_float_t,
                                      FB_OP_CUNMHR,
                                      fb_query_size_from_cfloat)
DEFINE_LAPACK_HESSENBERG_APPLY_THUNKS(zunmhr, fb_complex_double_t,
                                      FB_OP_ZUNMHR,
                                      fb_query_size_from_cdouble)

#undef DEFINE_LAPACK_HESSENBERG_APPLY_THUNKS

/* ---- DSPRFS  (FB_OP_DSPRFS=1076) --------------------------------------- */

static void thunk_dsprfs_fortran_to_cblas(char *uplo, int *n, int *nrhs,
                                          double *ap, double *afp, int *ipiv,
                                          double *b, int *ldb, double *x,
                                          int *ldx, double *ferr, double *berr,
                                          double *work, int *iwork, int *info) {
  if (!g_cblas_fn[FB_OP_DSPRFS]) {
    if (info)
      *info = -1;
    return;
  }
  /* CBLAS: (int layout, char uplo, int n, int nrhs,
   *         double *ap, double *afp, int *ipiv,
   *         double *b, int ldb, double *x, int ldx,
   *         double *ferr, double *berr, double *work, int *iwork, int *info) */
  ((void (*)(int, char, int, int, double *, double *, int *, double *, int,
             double *, int, double *, double *, double *, int *,
             int *))g_cblas_fn[FB_OP_DSPRFS])(102, *uplo, *n, *nrhs, ap, afp,
                                              ipiv, b, *ldb, x, *ldx, ferr,
                                              berr, work, iwork, info);
}

static void thunk_dsprfs_cblas_to_fortran(int layout, char uplo, int n,
                                          int nrhs, double *ap, double *afp,
                                          int *ipiv, double *b, int ldb,
                                          double *x, int ldx, double *ferr,
                                          double *berr, double *work,
                                          int *iwork, int *info) {
  if (fb_lapack_reject_non_col_major_cblas_layout(layout, info))
    return;
  if (!g_fortran_fn[FB_OP_DSPRFS]) {
    if (info)
      *info = -1;
    return;
  }
  int ldb_ = ldb, ldx_ = ldx;
  ((void (*)(char *, int *, int *, double *, double *, int *, double *, int *,
             double *, int *, double *, double *, double *, int *,
             int *))g_fortran_fn[FB_OP_DSPRFS])(&uplo, &n, &nrhs, ap, afp, ipiv,
                                                b, &ldb_, x, &ldx_, ferr, berr,
                                                work, iwork, info);
}

/* ---- STGSEN/DTGSEN  (FB_OP_STGSEN=1225, FB_OP_DTGSEN=1226) ------------- */

static void thunk_stgsen_fortran_to_cblas(
    int *ijob, int *wantq, int *wantz, const int *select, int *n, float *a,
    int *lda, float *b, int *ldb, float *alphar, float *alphai, float *beta,
    float *q, int *ldq, float *z, int *ldz, int *m, float *pl, float *pr,
    float *dif, float *work, int *lwork, int *iwork, int *liwork, int *info) {
  if (!g_cblas_fn[FB_OP_STGSEN]) {
    if (info)
      *info = -1;
    return;
  }
  ((void (*)(int, int, int, int, const int *, int, float *, int, float *, int,
             float *, float *, float *, float *, int, float *, int, int *,
             float *, float *, float *, float *, int, int *, int,
             int *))g_cblas_fn[FB_OP_STGSEN])(
      102, *ijob, *wantq, *wantz, select, *n, a, *lda, b, *ldb, alphar, alphai,
      beta, q, *ldq, z, *ldz, m, pl, pr, dif, work, *lwork, iwork, *liwork,
      info);
}

static void thunk_stgsen_cblas_to_fortran(
    int layout, int ijob, int wantq, int wantz, const int *select, int n,
    float *a, int lda, float *b, int ldb, float *alphar, float *alphai,
    float *beta, float *q, int ldq, float *z, int ldz, int *m, float *pl,
    float *pr, float *dif, float *work, int lwork, int *iwork, int liwork,
    int *info) {
  if (fb_lapack_reject_non_col_major_cblas_layout(layout, info))
    return;
  if (!g_fortran_fn[FB_OP_STGSEN]) {
    if (info)
      *info = -1;
    return;
  }
  int lda_ = lda, ldb_ = ldb, ldq_ = ldq, ldz_ = ldz, lwork_ = lwork,
      liwork_ = liwork;
  ((void (*)(int *, int *, int *, const int *, int *, float *, int *, float *,
             int *, float *, float *, float *, float *, int *, float *, int *,
             int *, float *, float *, float *, float *, int *, int *, int *,
             int *))g_fortran_fn[FB_OP_STGSEN])(
      &ijob, &wantq, &wantz, select, &n, a, &lda_, b, &ldb_, alphar, alphai,
      beta, q, &ldq_, z, &ldz_, m, pl, pr, dif, work, &lwork_, iwork, &liwork_,
      info);
}

static void thunk_dtgsen_fortran_to_cblas(
    int *ijob, int *wantq, int *wantz, const int *select, int *n, double *a,
    int *lda, double *b, int *ldb, double *alphar, double *alphai, double *beta,
    double *q, int *ldq, double *z, int *ldz, int *m, double *pl, double *pr,
    double *dif, double *work, int *lwork, int *iwork, int *liwork, int *info) {
  if (!g_cblas_fn[FB_OP_DTGSEN]) {
    if (info)
      *info = -1;
    return;
  }
  ((void (*)(int, int, int, int, const int *, int, double *, int, double *, int,
             double *, double *, double *, double *, int, double *, int, int *,
             double *, double *, double *, double *, int, int *, int,
             int *))g_cblas_fn[FB_OP_DTGSEN])(
      102, *ijob, *wantq, *wantz, select, *n, a, *lda, b, *ldb, alphar, alphai,
      beta, q, *ldq, z, *ldz, m, pl, pr, dif, work, *lwork, iwork, *liwork,
      info);
}

static void thunk_dtgsen_cblas_to_fortran(
    int layout, int ijob, int wantq, int wantz, const int *select, int n,
    double *a, int lda, double *b, int ldb, double *alphar, double *alphai,
    double *beta, double *q, int ldq, double *z, int ldz, int *m, double *pl,
    double *pr, double *dif, double *work, int lwork, int *iwork, int liwork,
    int *info) {
  if (fb_lapack_reject_non_col_major_cblas_layout(layout, info))
    return;
  if (!g_fortran_fn[FB_OP_DTGSEN]) {
    if (info)
      *info = -1;
    return;
  }
  int lda_ = lda, ldb_ = ldb, ldq_ = ldq, ldz_ = ldz, lwork_ = lwork,
      liwork_ = liwork;
  ((void (*)(int *, int *, int *, const int *, int *, double *, int *, double *,
             int *, double *, double *, double *, double *, int *, double *,
             int *, int *, double *, double *, double *, double *, int *, int *,
             int *, int *))g_fortran_fn[FB_OP_DTGSEN])(
      &ijob, &wantq, &wantz, select, &n, a, &lda_, b, &ldb_, alphar, alphai,
      beta, q, &ldq_, z, &ldz_, m, pl, pr, dif, work, &lwork_, iwork, &liwork_,
      info);
}

/* ---- CTGSEN/ZTGSEN  (FB_OP_CTGSEN=1227, FB_OP_ZTGSEN=1228) ------------- */
/* Complex arrays are typed void* — only the pointer value is forwarded.    */

static void thunk_ctgsen_fortran_to_cblas(
    int *ijob, int *wantq, int *wantz, const int *select, int *n, void *a,
    int *lda, void *b, int *ldb, void *alpha, void *beta, void *q, int *ldq,
    void *z, int *ldz, int *m, float *pl, float *pr, float *dif, void *work,
    int *lwork, int *iwork, int *liwork, int *info) {
  if (!g_cblas_fn[FB_OP_CTGSEN]) {
    if (info)
      *info = -1;
    return;
  }
  ((void (*)(int, int, int, int, const int *, int, void *, int, void *, int,
             void *, void *, void *, int, void *, int, int *, float *, float *,
             float *, void *, int, int *, int, int *))g_cblas_fn[FB_OP_CTGSEN])(
      102, *ijob, *wantq, *wantz, select, *n, a, *lda, b, *ldb, alpha, beta, q,
      *ldq, z, *ldz, m, pl, pr, dif, work, *lwork, iwork, *liwork, info);
}

static void thunk_ctgsen_cblas_to_fortran(
    int layout, int ijob, int wantq, int wantz, const int *select, int n,
    void *a, int lda, void *b, int ldb, void *alpha, void *beta, void *q,
    int ldq, void *z, int ldz, int *m, float *pl, float *pr, float *dif,
    void *work, int lwork, int *iwork, int liwork, int *info) {
  if (fb_lapack_reject_non_col_major_cblas_layout(layout, info))
    return;
  if (!g_fortran_fn[FB_OP_CTGSEN]) {
    if (info)
      *info = -1;
    return;
  }
  int lda_ = lda, ldb_ = ldb, ldq_ = ldq, ldz_ = ldz, lwork_ = lwork,
      liwork_ = liwork;
  int ldb_tmp = ldb_; /* avoid -Wshadow with ldb_ used below */
  (void)ldb_tmp;
  ((void (*)(int *, int *, int *, const int *, int *, void *, int *, void *,
             int *, void *, void *, void *, int *, void *, int *, int *,
             float *, float *, float *, void *, int *, int *, int *,
             int *))g_fortran_fn[FB_OP_CTGSEN])(
      &ijob, &wantq, &wantz, select, &n, a, &lda_, b, &ldb_, alpha, beta, q,
      &ldq_, z, &ldz_, m, pl, pr, dif, work, &lwork_, iwork, &liwork_, info);
}

static void thunk_ztgsen_fortran_to_cblas(
    int *ijob, int *wantq, int *wantz, const int *select, int *n, void *a,
    int *lda, void *b, int *ldb, void *alpha, void *beta, void *q, int *ldq,
    void *z, int *ldz, int *m, double *pl, double *pr, double *dif, void *work,
    int *lwork, int *iwork, int *liwork, int *info) {
  if (!g_cblas_fn[FB_OP_ZTGSEN]) {
    if (info)
      *info = -1;
    return;
  }
  ((void (*)(int, int, int, int, const int *, int, void *, int, void *, int,
             void *, void *, void *, int, void *, int, int *, double *,
             double *, double *, void *, int, int *, int,
             int *))g_cblas_fn[FB_OP_ZTGSEN])(
      102, *ijob, *wantq, *wantz, select, *n, a, *lda, b, *ldb, alpha, beta, q,
      *ldq, z, *ldz, m, pl, pr, dif, work, *lwork, iwork, *liwork, info);
}

static void thunk_ztgsen_cblas_to_fortran(
    int layout, int ijob, int wantq, int wantz, const int *select, int n,
    void *a, int lda, void *b, int ldb, void *alpha, void *beta, void *q,
    int ldq, void *z, int ldz, int *m, double *pl, double *pr, double *dif,
    void *work, int lwork, int *iwork, int liwork, int *info) {
  if (fb_lapack_reject_non_col_major_cblas_layout(layout, info))
    return;
  if (!g_fortran_fn[FB_OP_ZTGSEN]) {
    if (info)
      *info = -1;
    return;
  }
  int lda_ = lda, ldb_ = ldb, ldq_ = ldq, ldz_ = ldz, lwork_ = lwork,
      liwork_ = liwork;
  int ldb_tmp = ldb_;
  (void)ldb_tmp;
  ((void (*)(int *, int *, int *, const int *, int *, void *, int *, void *,
             int *, void *, void *, void *, int *, void *, int *, int *,
             double *, double *, double *, void *, int *, int *, int *,
             int *))g_fortran_fn[FB_OP_ZTGSEN])(
      &ijob, &wantq, &wantz, select, &n, a, &lda_, b, &ldb_, alpha, beta, q,
      &ldq_, z, &ldz_, m, pl, pr, dif, work, &lwork_, iwork, &liwork_, info);
}

/* ---- STGSJA/DTGSJA  (FB_OP_STGSJA=1229, FB_OP_DTGSJA=1230) ------------ */

static void thunk_stgsja_fortran_to_cblas(
    char *jobu, char *jobv, char *jobq, int *m, int *p, int *n, int *k, int *l,
    float *a, int *lda, float *b, int *ldb, float *tola, float *tolb,
    float *alpha, float *beta, float *u, int *ldu, float *v, int *ldv, float *q,
    int *ldq, float *work, int *ncycle, int *info) {
  if (!g_cblas_fn[FB_OP_STGSJA]) {
    if (info)
      *info = -1;
    return;
  }
  /* CBLAS: (int layout, char jobu, char jobv, char jobq,
   *         int m, int p, int n, int k, int l,
   *         float *a, int lda, float *b, int ldb,
   *         float tola, float tolb, float *alpha, float *beta,
   *         float *u, int ldu, float *v, int ldv,
   *         float *q, int ldq, float *work, int *ncycle, int *info) */
  ((void (*)(int, char, char, char, int, int, int, int, int, float *, int,
             float *, int, float, float, float *, float *, float *, int,
             float *, int, float *, int, float *, int *,
             int *))g_cblas_fn[FB_OP_STGSJA])(
      102, *jobu, *jobv, *jobq, *m, *p, *n, *k, *l, a, *lda, b, *ldb, *tola,
      *tolb, alpha, beta, u, *ldu, v, *ldv, q, *ldq, work, ncycle, info);
}

static void thunk_stgsja_cblas_to_fortran(
    int layout, char jobu, char jobv, char jobq, int m, int p, int n, int k,
    int l, float *a, int lda, float *b, int ldb, float tola, float tolb,
    float *alpha, float *beta, float *u, int ldu, float *v, int ldv, float *q,
    int ldq, float *work, int *ncycle, int *info) {
  if (fb_lapack_reject_non_col_major_cblas_layout(layout, info))
    return;
  if (!g_fortran_fn[FB_OP_STGSJA]) {
    if (info)
      *info = -1;
    return;
  }
  int lda_ = lda, ldb_ = ldb, ldu_ = ldu, ldv_ = ldv, ldq_ = ldq;
  ((void (*)(char *, char *, char *, int *, int *, int *, int *, int *, float *,
             int *, float *, int *, float *, float *, float *, float *, float *,
             int *, float *, int *, float *, int *, float *, int *,
             int *))g_fortran_fn[FB_OP_STGSJA])(
      &jobu, &jobv, &jobq, &m, &p, &n, &k, &l, a, &lda_, b, &ldb_, &tola, &tolb,
      alpha, beta, u, &ldu_, v, &ldv_, q, &ldq_, work, ncycle, info);
}

static void thunk_dtgsja_fortran_to_cblas(
    char *jobu, char *jobv, char *jobq, int *m, int *p, int *n, int *k, int *l,
    double *a, int *lda, double *b, int *ldb, double *tola, double *tolb,
    double *alpha, double *beta, double *u, int *ldu, double *v, int *ldv,
    double *q, int *ldq, double *work, int *ncycle, int *info) {
  if (!g_cblas_fn[FB_OP_DTGSJA]) {
    if (info)
      *info = -1;
    return;
  }
  ((void (*)(int, char, char, char, int, int, int, int, int, double *, int,
             double *, int, double, double, double *, double *, double *, int,
             double *, int, double *, int, double *, int *,
             int *))g_cblas_fn[FB_OP_DTGSJA])(
      102, *jobu, *jobv, *jobq, *m, *p, *n, *k, *l, a, *lda, b, *ldb, *tola,
      *tolb, alpha, beta, u, *ldu, v, *ldv, q, *ldq, work, ncycle, info);
}

static void thunk_dtgsja_cblas_to_fortran(
    int layout, char jobu, char jobv, char jobq, int m, int p, int n, int k,
    int l, double *a, int lda, double *b, int ldb, double tola, double tolb,
    double *alpha, double *beta, double *u, int ldu, double *v, int ldv,
    double *q, int ldq, double *work, int *ncycle, int *info) {
  if (fb_lapack_reject_non_col_major_cblas_layout(layout, info))
    return;
  if (!g_fortran_fn[FB_OP_DTGSJA]) {
    if (info)
      *info = -1;
    return;
  }
  int lda_ = lda, ldb_ = ldb, ldu_ = ldu, ldv_ = ldv, ldq_ = ldq;
  ((void (*)(char *, char *, char *, int *, int *, int *, int *, int *,
             double *, int *, double *, int *, double *, double *, double *,
             double *, double *, int *, double *, int *, double *, int *,
             double *, int *, int *))g_fortran_fn[FB_OP_DTGSJA])(
      &jobu, &jobv, &jobq, &m, &p, &n, &k, &l, a, &lda_, b, &ldb_, &tola, &tolb,
      alpha, beta, u, &ldu_, v, &ldv_, q, &ldq_, work, ncycle, info);
}

/* ---- CTGSJA/ZTGSJA  (FB_OP_CTGSJA=1231, FB_OP_ZTGSJA=1232) ------------ */
/* Complex matrices as void*; tola/tolb/alpha/beta are real (float/double). */

static void thunk_ctgsja_fortran_to_cblas(
    char *jobu, char *jobv, char *jobq, int *m, int *p, int *n, int *k, int *l,
    void *a, int *lda, void *b, int *ldb, float *tola, float *tolb,
    float *alpha, float *beta, void *u, int *ldu, void *v, int *ldv, void *q,
    int *ldq, void *work, int *ncycle, int *info) {
  if (!g_cblas_fn[FB_OP_CTGSJA]) {
    if (info)
      *info = -1;
    return;
  }
  ((void (*)(int, char, char, char, int, int, int, int, int, void *, int,
             void *, int, float, float, float *, float *, void *, int, void *,
             int, void *, int, void *, int *, int *))g_cblas_fn[FB_OP_CTGSJA])(
      102, *jobu, *jobv, *jobq, *m, *p, *n, *k, *l, a, *lda, b, *ldb, *tola,
      *tolb, alpha, beta, u, *ldu, v, *ldv, q, *ldq, work, ncycle, info);
}

static void thunk_ctgsja_cblas_to_fortran(
    int layout, char jobu, char jobv, char jobq, int m, int p, int n, int k,
    int l, void *a, int lda, void *b, int ldb, float tola, float tolb,
    float *alpha, float *beta, void *u, int ldu, void *v, int ldv, void *q,
    int ldq, void *work, int *ncycle, int *info) {
  if (fb_lapack_reject_non_col_major_cblas_layout(layout, info))
    return;
  if (!g_fortran_fn[FB_OP_CTGSJA]) {
    if (info)
      *info = -1;
    return;
  }
  int lda_ = lda, ldb_ = ldb, ldu_ = ldu, ldv_ = ldv, ldq_ = ldq;
  ((void (*)(char *, char *, char *, int *, int *, int *, int *, int *, void *,
             int *, void *, int *, float *, float *, float *, float *, void *,
             int *, void *, int *, void *, int *, void *, int *,
             int *))g_fortran_fn[FB_OP_CTGSJA])(
      &jobu, &jobv, &jobq, &m, &p, &n, &k, &l, a, &lda_, b, &ldb_, &tola, &tolb,
      alpha, beta, u, &ldu_, v, &ldv_, q, &ldq_, work, ncycle, info);
}

static void thunk_ztgsja_fortran_to_cblas(
    char *jobu, char *jobv, char *jobq, int *m, int *p, int *n, int *k, int *l,
    void *a, int *lda, void *b, int *ldb, double *tola, double *tolb,
    double *alpha, double *beta, void *u, int *ldu, void *v, int *ldv, void *q,
    int *ldq, void *work, int *ncycle, int *info) {
  if (!g_cblas_fn[FB_OP_ZTGSJA]) {
    if (info)
      *info = -1;
    return;
  }
  ((void (*)(int, char, char, char, int, int, int, int, int, void *, int,
             void *, int, double, double, double *, double *, void *, int,
             void *, int, void *, int, void *, int *,
             int *))g_cblas_fn[FB_OP_ZTGSJA])(
      102, *jobu, *jobv, *jobq, *m, *p, *n, *k, *l, a, *lda, b, *ldb, *tola,
      *tolb, alpha, beta, u, *ldu, v, *ldv, q, *ldq, work, ncycle, info);
}

static void thunk_ztgsja_cblas_to_fortran(
    int layout, char jobu, char jobv, char jobq, int m, int p, int n, int k,
    int l, void *a, int lda, void *b, int ldb, double tola, double tolb,
    double *alpha, double *beta, void *u, int ldu, void *v, int ldv, void *q,
    int ldq, void *work, int *ncycle, int *info) {
  if (fb_lapack_reject_non_col_major_cblas_layout(layout, info))
    return;
  if (!g_fortran_fn[FB_OP_ZTGSJA]) {
    if (info)
      *info = -1;
    return;
  }
  int lda_ = lda, ldb_ = ldb, ldu_ = ldu, ldv_ = ldv, ldq_ = ldq;
  ((void (*)(char *, char *, char *, int *, int *, int *, int *, int *, void *,
             int *, void *, int *, double *, double *, double *, double *,
             void *, int *, void *, int *, void *, int *, void *, int *,
             int *))g_fortran_fn[FB_OP_ZTGSJA])(
      &jobu, &jobv, &jobq, &m, &p, &n, &k, &l, a, &lda_, b, &ldb_, &tola, &tolb,
      alpha, beta, u, &ldu_, v, &ldv_, q, &ldq_, work, ncycle, info);
}

/* ---- Spectral LAPACK thunks -------------------------------------------- */

/* These bridge underscore-only LAPACK exports into the simplified
 * LAPACKE-style signatures stored in fb_backend_vtable_t. Template selection
 * is driven by ABI shape plus policy dimensions: layout handling, workspace
 * query behavior, and whether mode bits are forwarded or constrained. No
 * template may silently discard caller intent; unsupported combinations must
 * fail explicitly instead of being rewritten to an easier subset. For
 * row-major symmetric/Hermitian inputs we flip UPLO because row-major upper
 * data is seen as column-major lower data by the Fortran kernels.
 *
 * Generic rule: reuse one thunk template per exact ABI shape. Only add a
 * bespoke family macro when that shape also needs extra layout or workspace
 * policy. Generated LAPACK C bridges are column-major only unless a template
 * explicitly translates row-major inputs. */

#define DEFINE_LAPACK_LU_SOLVE_THUNKS(OP, TYPE, OP_ID)                         \
static int thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const int n, const int nrhs, TYPE *A,            \
    const int lda, int *ipiv, TYPE *B, const int ldb) {                        \
  if (layout != FB_LAYOUT_COL_MAJOR) return -1;                                \
  if (!g_fortran_fn[OP_ID]) return -1;                                          \
  int n_ = n, nrhs_ = nrhs, lda_ = lda, ldb_ = ldb, info = 0;                  \
  ((void (*)(int *, int *, TYPE *, int *, int *, TYPE *, int *, int *))        \
      g_fortran_fn[OP_ID])(&n_, &nrhs_, A, &lda_, ipiv, B, &ldb_, &info);      \
  return info;                                                                  \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    int *n, int *nrhs, TYPE *A, int *lda, int *ipiv, TYPE *B, int *ldb,        \
    int *info) {                                                                \
  if (!g_cblas_fn[OP_ID]) {                                                     \
    if (info) *info = -1;                                                       \
    return;                                                                     \
  }                                                                             \
  int rc = ((int (*)(fb_layout_t, int, int, TYPE *, int, int *, TYPE *, int))  \
      g_cblas_fn[OP_ID])(FB_LAYOUT_COL_MAJOR, *n, *nrhs, A, *lda, ipiv, B, *ldb); \
  if (info) *info = rc;                                                         \
}
