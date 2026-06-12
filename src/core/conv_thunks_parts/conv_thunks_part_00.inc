/**
 * @file conv_thunks.c
 * @brief Cross-convention calling-convention thunks for BLAS/LAPACK operations.
 *
 * ## Design
 *
 * One static thunk function is generated per (operation, direction) pair for
 * every standard BLAS operation that differs between calling conventions.
 *
 * Two arrays are exposed:
 *   - k_cblas_to_fortran_thunks[op]   — wraps CBLAS impl, exposes Fortran ABI
 *   - k_fortran_to_cblas_thunks[op]   — wraps Fortran impl, exposes CBLAS ABI
 *
 * Backing function pointers (g_cblas_fn / g_fortran_fn) are filled by
 * fb_install_conv_thunks() which is called from vtable_autofill.c Strategy 5.
 *
 * ## Key difference: CBLAS vs Fortran
 *
 *   CBLAS:   pass-by-value for integer and real scalar IN arguments
 *   Fortran: pass-by-pointer for ALL arguments (integers, scalars, arrays)
 *
 * Complex scalars are already passed by pointer in CBLAS (as const void *), so
 * complex-typed operations only need to convert the integer parameters.
 *
 * ## Note on multi-backend support
 *
 * The global backing arrays (g_cblas_fn, g_fortran_fn) hold one function
 * pointer per operation, limiting full thunk support to one active backend at
 * a time.  Multi-backend JIT trampolines are a planned enhancement.
 *
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

/* ============================================================================
 * Global backing pointers
 *
 * g_cblas_fn[op]    — CBLAS-convention implementation (used by →Fortran thunks)
 * g_fortran_fn[op]  — Fortran-convention implementation (used by →CBLAS thunks)
 * ========================================================================== */
static fb_generic_fn g_cblas_fn   [FB_JUDGE_MAX_OPERATIONS];
static fb_generic_fn g_fortran_fn [FB_JUDGE_MAX_OPERATIONS];

static char fb_lapack_uplo_from_layout(fb_layout_t layout, fb_uplo_t uplo) {
  char uplo_char = (uplo == FB_LOWER) ? 'L' : 'U';
  if (layout == FB_LAYOUT_ROW_MAJOR) {
    uplo_char = (uplo_char == 'U') ? 'L' : 'U';
  }
  return uplo_char;
}

static int fb_lapack_layout_is_col_major(fb_layout_t layout) {
  return layout == FB_LAYOUT_COL_MAJOR;
}

static int fb_lapack_reject_non_col_major_cblas_layout(int layout, int *info) {
  if (layout == FB_LAYOUT_COL_MAJOR) {
    return 0;
  }
  if (info) {
    *info = -1;
  }
  return 1;
}

static int fb_lapack_job_is(char job, char upper_code) {
  char lower_code = (char)(upper_code + ('a' - 'A'));
  return (job == upper_code) || (job == lower_code);
}

static int fb_lapack_job_requests_vectors(char job) {
  return fb_lapack_job_is(job, 'V');
}

static int fb_lapack_job_produces_vectors(char job) {
  return fb_lapack_job_requests_vectors(job) || fb_lapack_job_is(job, 'I');
}

static int fb_lapack_svd_job_requires_explicit_output(char job) {
  return fb_lapack_job_is(job, 'A') || fb_lapack_job_is(job, 'S');
}

static int fb_lapack_job_is_none(char job) {
  return fb_lapack_job_is(job, 'N');
}

static int fb_lapack_transpose_to_char(fb_transpose_t trans, char *trans_char) {
  if (!trans_char) {
    return -1;
  }

  switch (trans) {
  case FB_NO_TRANS:
    *trans_char = 'N';
    return 0;
  case FB_TRANS:
    *trans_char = 'T';
    return 0;
  case FB_CONJ_TRANS:
    *trans_char = 'C';
    return 0;
  default:
    return -1;
  }
}

static int fb_lapack_char_to_transpose(char trans_char,
                                       fb_transpose_t *trans) {
  if (!trans) {
    return -1;
  }

  if (fb_lapack_job_is(trans_char, 'N')) {
    *trans = FB_NO_TRANS;
    return 0;
  }
  if (fb_lapack_job_is(trans_char, 'T')) {
    *trans = FB_TRANS;
    return 0;
  }
  if (fb_lapack_job_is(trans_char, 'C')) {
    *trans = FB_CONJ_TRANS;
    return 0;
  }

  return -1;
}

static int fb_lapack_side_to_char(fb_side_t side, char *side_char) {
  if (!side_char) {
    return -1;
  }

  switch (side) {
  case FB_LEFT:
    *side_char = 'L';
    return 0;
  case FB_RIGHT:
    *side_char = 'R';
    return 0;
  default:
    return -1;
  }
}

static int fb_lapack_char_to_side(char side_char, fb_side_t *side) {
  if (!side) {
    return -1;
  }

  if (fb_lapack_job_is(side_char, 'L')) {
    *side = FB_LEFT;
    return 0;
  }
  if (fb_lapack_job_is(side_char, 'R')) {
    *side = FB_RIGHT;
    return 0;
  }

  return -1;
}

static int fb_lapack_bidiag_apply_a_dims(char vect_char, fb_side_t side,
                                         int m, int n, int k,
                                         int *rows, int *cols) {
  int nq = 0;
  int min_nq_k = 0;

  if (!rows || !cols) {
    return -1;
  }
  if ((side != FB_LEFT) && (side != FB_RIGHT)) {
    return -1;
  }

  nq = (side == FB_LEFT) ? m : n;
  min_nq_k = (nq < k) ? nq : k;

  if (fb_lapack_job_is(vect_char, 'Q')) {
    *rows = nq;
    *cols = min_nq_k;
    return 0;
  }
  if (fb_lapack_job_is(vect_char, 'P')) {
    *rows = min_nq_k;
    *cols = nq;
    return 0;
  }

  return -1;
}

static int fb_lapack_char_to_uplo(char uplo_char, fb_uplo_t *uplo) {
  if (!uplo) {
    return -1;
  }

  if (fb_lapack_job_is(uplo_char, 'U')) {
    *uplo = FB_UPPER;
    return 0;
  }
  if (fb_lapack_job_is(uplo_char, 'L')) {
    *uplo = FB_LOWER;
    return 0;
  }

  return -1;
}

static int fb_lapack_uplo_to_char(fb_uplo_t uplo, char *uplo_char) {
  if (!uplo_char) {
    return -1;
  }

  switch (uplo) {
  case FB_UPPER:
    *uplo_char = 'U';
    return 0;
  case FB_LOWER:
    *uplo_char = 'L';
    return 0;
  default:
    return -1;
  }
}

static int fb_copy_row_major_to_col_major(const void *src, int rows, int cols,
                                          int src_ld, void *dst, int dst_ld,
                                          size_t elem_size) {
  const unsigned char *src_bytes = (const unsigned char *)src;
  unsigned char *dst_bytes = (unsigned char *)dst;
  int row = 0;
  int col = 0;

  if (!src || !dst || rows < 0 || cols < 0 || elem_size == 0) {
    return -1;
  }
  if ((rows > 0 && dst_ld < rows) || (cols > 0 && src_ld < cols)) {
    return -1;
  }

  for (row = 0; row < rows; ++row) {
    for (col = 0; col < cols; ++col) {
      memcpy(dst_bytes + (((size_t)col * (size_t)dst_ld) + (size_t)row) * elem_size,
             src_bytes + (((size_t)row * (size_t)src_ld) + (size_t)col) * elem_size,
             elem_size);
    }
  }

  return 0;
}

static int fb_copy_col_major_to_row_major(const void *src, int rows, int cols,
                                          int src_ld, void *dst, int dst_ld,
                                          size_t elem_size) {
  const unsigned char *src_bytes = (const unsigned char *)src;
  unsigned char *dst_bytes = (unsigned char *)dst;
  int row = 0;
  int col = 0;

  if (!src || !dst || rows < 0 || cols < 0 || elem_size == 0) {
    return -1;
  }
  if ((cols > 0 && dst_ld < cols) || (rows > 0 && src_ld < rows)) {
    return -1;
  }

  for (row = 0; row < rows; ++row) {
    for (col = 0; col < cols; ++col) {
      memcpy(dst_bytes + (((size_t)row * (size_t)dst_ld) + (size_t)col) * elem_size,
             src_bytes + (((size_t)col * (size_t)src_ld) + (size_t)row) * elem_size,
             elem_size);
    }
  }

  return 0;
}

static int fb_copy_col_major_to_col_major(const void *src, int rows, int cols,
                                          int src_ld, void *dst, int dst_ld,
                                          size_t elem_size) {
  const unsigned char *src_bytes = (const unsigned char *)src;
  unsigned char *dst_bytes = (unsigned char *)dst;
  int row = 0;
  int col = 0;

  if (!src || !dst || rows < 0 || cols < 0 || elem_size == 0) {
    return -1;
  }
  if ((rows > 0 && src_ld < rows) || (rows > 0 && dst_ld < rows)) {
    return -1;
  }

  for (col = 0; col < cols; ++col) {
    for (row = 0; row < rows; ++row) {
      memcpy(dst_bytes + (((size_t)col * (size_t)dst_ld) + (size_t)row) * elem_size,
             src_bytes + (((size_t)col * (size_t)src_ld) + (size_t)row) * elem_size,
             elem_size);
    }
  }

  return 0;
}

static int fb_copy_matrix_to_col_major(const void *src, fb_layout_t layout,
                                       int rows, int cols, int src_ld,
                                       void *dst, int dst_ld,
                                       size_t elem_size) {
  if (layout == FB_LAYOUT_ROW_MAJOR) {
    return fb_copy_row_major_to_col_major(src, rows, cols, src_ld,
                                          dst, dst_ld, elem_size);
  }
  if (fb_lapack_layout_is_col_major(layout)) {
    return fb_copy_col_major_to_col_major(src, rows, cols, src_ld,
                                          dst, dst_ld, elem_size);
  }

  return -1;
}

static int fb_copy_matrix_from_col_major(const void *src, int rows, int cols,
                                         int src_ld, fb_layout_t layout,
                                         void *dst, int dst_ld,
                                         size_t elem_size) {
  if (layout == FB_LAYOUT_ROW_MAJOR) {
    return fb_copy_col_major_to_row_major(src, rows, cols, src_ld,
                                          dst, dst_ld, elem_size);
  }
  if (fb_lapack_layout_is_col_major(layout)) {
    return fb_copy_col_major_to_col_major(src, rows, cols, src_ld,
                                          dst, dst_ld, elem_size);
  }

  return -1;
}

static int fb_copy_strided_vector(const void *src, int count, int inc,
                                  void *dst, size_t elem_size) {
  const unsigned char *src_bytes = (const unsigned char *)src;
  unsigned char *dst_bytes = (unsigned char *)dst;
  int index = 0;

  if (!src || !dst || count < 0 || inc <= 0 || elem_size == 0) {
    return -1;
  }

  for (index = 0; index < count; ++index) {
    memcpy(dst_bytes + ((size_t)index * elem_size),
           src_bytes + ((size_t)index * (size_t)inc * elem_size),
           elem_size);
  }

  return 0;
}

static int fb_copy_row_major_triangular_to_col_major(const void *src,
                                                     char uplo, char diag,
                                                     int n, int src_ld,
                                                     void *dst, int dst_ld,
                                                     size_t elem_size) {
  const unsigned char *src_bytes = (const unsigned char *)src;
  unsigned char *dst_bytes = (unsigned char *)dst;
  int upper = fb_lapack_job_is(uplo, 'U');
  int lower = fb_lapack_job_is(uplo, 'L');
  int unit = fb_lapack_job_is(diag, 'U');
  int non_unit = fb_lapack_job_is(diag, 'N');
  int st = unit ? 1 : 0;
  int row = 0;
  int col = 0;

  if (!src || !dst || n < 0 || elem_size == 0) {
    return -1;
  }
  if (!upper && !lower) {
    return -1;
  }
  if (!unit && !non_unit) {
    return -1;
  }
  if ((n > 0 && src_ld < n) || (n > 0 && dst_ld < n)) {
    return -1;
  }

  for (col = 0; col < n; ++col) {
    for (row = 0; row < n; ++row) {
      int in_region = upper ? (row + st <= col) : (row >= col + st);
      size_t src_idx = 0;
      size_t dst_idx = 0;

      if (!in_region) {
        continue;
      }

      src_idx = ((size_t)row * (size_t)src_ld) + (size_t)col;
      dst_idx = ((size_t)col * (size_t)dst_ld) + (size_t)row;
      memcpy(dst_bytes + (dst_idx * elem_size),
             src_bytes + (src_idx * elem_size), elem_size);
    }
  }

  return 0;
}

static int fb_copy_row_major_trapezoid_to_col_major(const void *src,
                                                    char direct, char uplo,
                                                    char diag, int rows,
                                                    int cols, int src_ld,
                                                    void *dst, int dst_ld,
                                                    size_t elem_size) {
  const unsigned char *src_bytes = (const unsigned char *)src;
  unsigned char *dst_bytes = (unsigned char *)dst;
  int front = fb_lapack_job_is(direct, 'F');
  int back = fb_lapack_job_is(direct, 'B');
  int lower = fb_lapack_job_is(uplo, 'L');
  int upper = fb_lapack_job_is(uplo, 'U');
  int unit = fb_lapack_job_is(diag, 'U');
  int non_unit = fb_lapack_job_is(diag, 'N');
  int tri_src_offset = 0;
  int tri_dst_offset = 0;
  int tri_n = (rows < cols) ? rows : cols;
  int rect_src_offset = -1;
  int rect_dst_offset = -1;
  int rect_rows = (rows > cols) ? (rows - cols) : rows;
  int rect_cols = (cols > rows) ? (cols - rows) : cols;

  if (!src || !dst || rows < 0 || cols < 0 || elem_size == 0) {
    return -1;
  }
  if ((!front && !back) || (!lower && !upper) || (!unit && !non_unit)) {
    return -1;
  }
  if ((cols > 0 && src_ld < cols) || (rows > 0 && dst_ld < rows)) {
    return -1;
  }

  if (front) {
    if (lower && rows > cols) {
      rect_src_offset = tri_n * src_ld;
      rect_dst_offset = tri_n;
    } else if (!lower && cols > rows) {
      rect_src_offset = tri_n;
      rect_dst_offset = tri_n * dst_ld;
    }
  } else {
    if (rows > cols) {
      tri_src_offset = rect_rows * src_ld;
      tri_dst_offset = rect_rows;
      if (!lower) {
        rect_src_offset = 0;
        rect_dst_offset = 0;
      }
    } else if (cols > rows) {
      tri_src_offset = rect_cols;
      tri_dst_offset = rect_cols * dst_ld;
      if (lower) {
        rect_src_offset = 0;
        rect_dst_offset = 0;
      }
    }
  }

  if (rect_src_offset >= 0 && rect_dst_offset >= 0 &&
      rect_rows > 0 && rect_cols > 0) {
    if (fb_copy_row_major_to_col_major(
            src_bytes + ((size_t)rect_src_offset * elem_size),
            rect_rows, rect_cols, src_ld,
            dst_bytes + ((size_t)rect_dst_offset * elem_size),
            dst_ld, elem_size) != 0) {
      return -1;
    }
  }

  if (tri_n > 0) {
    if (fb_copy_row_major_triangular_to_col_major(
            src_bytes + ((size_t)tri_src_offset * elem_size),
            uplo, diag, tri_n, src_ld,
            dst_bytes + ((size_t)tri_dst_offset * elem_size),
            dst_ld, elem_size) != 0) {
      return -1;
    }
  }

  return 0;
}

static int fb_copy_triangular_packed_to_col_major(const void *src,
                                                  fb_layout_t layout,
                                                  char uplo, char diag,
                                                  int n, void *dst,
                                                  size_t elem_size) {
  const unsigned char *src_bytes = (const unsigned char *)src;
  unsigned char *dst_bytes = (unsigned char *)dst;
  int upper = fb_lapack_job_is(uplo, 'U');
  int lower = fb_lapack_job_is(uplo, 'L');
  int unit = fb_lapack_job_is(diag, 'U');
  int non_unit = fb_lapack_job_is(diag, 'N');
  int st = 0;
  int i = 0;
  int j = 0;
  size_t packed_len = 0;

  if (!src || !dst || n < 0 || elem_size == 0) {
    return -1;
  }
  if (!upper && !lower) {
    return -1;
  }
  if (!unit && !non_unit) {
    return -1;
  }

  packed_len = (size_t)(n > 0 ? n : 1) * (size_t)(n + 1 > 0 ? (n + 1) : 2) / 2;
  memset(dst_bytes, 0, packed_len * elem_size);

  if (fb_lapack_layout_is_col_major(layout)) {
    memcpy(dst_bytes, src_bytes, packed_len * elem_size);
    return 0;
  }
  if (layout != FB_LAYOUT_ROW_MAJOR) {
    return -1;
  }

  st = unit ? 1 : 0;
  if (upper) {
    for (j = 0; j < n - st; ++j) {
      for (i = j + st; i < n; ++i) {
        size_t dst_idx = (size_t)j + (((size_t)i + 1U) * (size_t)i) / 2U;
        size_t src_idx = ((size_t)j * (size_t)(2 * n - j + 1)) / 2U +
                         (size_t)(i - j);
        memcpy(dst_bytes + (dst_idx * elem_size),
               src_bytes + (src_idx * elem_size), elem_size);
      }
    }
    return 0;
  }

  for (j = st; j < n; ++j) {
    for (i = 0; i < j + 1 - st; ++i) {
      size_t dst_idx = (size_t)(j - i) +
                       ((size_t)i * (size_t)(2 * n - i + 1)) / 2U;
      size_t src_idx = (((size_t)j + 1U) * (size_t)j) / 2U + (size_t)i;
      memcpy(dst_bytes + (dst_idx * elem_size),
             src_bytes + (src_idx * elem_size), elem_size);
    }
  }

  return 0;
}

static int fb_copy_triangular_packed_from_col_major(const void *src,
                                                    fb_layout_t layout,
                                                    char uplo, char diag,
                                                    int n, void *dst,
                                                    size_t elem_size) {
  const unsigned char *src_bytes = (const unsigned char *)src;
  unsigned char *dst_bytes = (unsigned char *)dst;
  int upper = fb_lapack_job_is(uplo, 'U');
  int lower = fb_lapack_job_is(uplo, 'L');
  int unit = fb_lapack_job_is(diag, 'U');
  int non_unit = fb_lapack_job_is(diag, 'N');
  int st = 0;
  int i = 0;
  int j = 0;
  size_t packed_len = 0;

  if (!src || !dst || n < 0 || elem_size == 0) {
    return -1;
  }
  if (!upper && !lower) {
    return -1;
  }
  if (!unit && !non_unit) {
    return -1;
  }

  packed_len = (size_t)(n > 0 ? n : 1) * (size_t)(n + 1 > 0 ? (n + 1) : 2) / 2;
  memset(dst_bytes, 0, packed_len * elem_size);

  if (fb_lapack_layout_is_col_major(layout)) {
    memcpy(dst_bytes, src_bytes, packed_len * elem_size);
    return 0;
  }
  if (layout != FB_LAYOUT_ROW_MAJOR) {
    return -1;
  }

  st = unit ? 1 : 0;
  if (upper) {
    for (j = 0; j < n - st; ++j) {
      for (i = j + st; i < n; ++i) {
        size_t src_idx = (size_t)j + (((size_t)i + 1U) * (size_t)i) / 2U;
        size_t dst_idx = ((size_t)j * (size_t)(2 * n - j + 1)) / 2U +
                         (size_t)(i - j);
        memcpy(dst_bytes + (dst_idx * elem_size),
               src_bytes + (src_idx * elem_size), elem_size);
      }
    }
    return 0;
  }

  for (j = st; j < n; ++j) {
    for (i = 0; i < j + 1 - st; ++i) {
      size_t src_idx = (size_t)(j - i) +
                       ((size_t)i * (size_t)(2 * n - i + 1)) / 2U;
      size_t dst_idx = (((size_t)j + 1U) * (size_t)j) / 2U + (size_t)i;
      memcpy(dst_bytes + (dst_idx * elem_size),
             src_bytes + (src_idx * elem_size), elem_size);
    }
  }

  return 0;
}

static int fb_copy_general_band_to_col_major(const void *src, fb_layout_t layout,
                                             int m, int n, int kl, int ku,
                                             int src_ld, void *dst, int dst_ld,
                                             size_t elem_size) {
  const unsigned char *src_bytes = (const unsigned char *)src;
  unsigned char *dst_bytes = (unsigned char *)dst;
  int i = 0;
  int j = 0;

  if (!src || !dst || m < 0 || n < 0 || kl < 0 || ku < 0 || elem_size == 0) {
    return -1;
  }
  if (dst_ld < (kl + ku + 1)) {
    return -1;
  }

  memset(dst_bytes, 0, (size_t)dst_ld * (size_t)((n > 0) ? n : 1) * elem_size);

  if (layout == FB_LAYOUT_ROW_MAJOR) {
    if (src_ld < n) {
      return -1;
    }

    for (j = 0; j < n; ++j) {
      int i_start = ku - j;
      int i_end = m + ku - j;

      if (i_start < 0) {
        i_start = 0;
      }
      if (i_end > (kl + ku + 1)) {
        i_end = kl + ku + 1;
      }
      if (i_end > dst_ld) {
        i_end = dst_ld;
      }

      for (i = i_start; i < i_end; ++i) {
        memcpy(dst_bytes + ((((size_t)j * (size_t)dst_ld) + (size_t)i) * elem_size),
               src_bytes + ((((size_t)i * (size_t)src_ld) + (size_t)j) * elem_size),
               elem_size);
      }
    }

    return 0;
  }

  if (fb_lapack_layout_is_col_major(layout)) {
    return fb_copy_col_major_to_col_major(src, kl + ku + 1, n, src_ld,
                                          dst, dst_ld, elem_size);
  }

  return -1;
}

static int fb_copy_sym_band_to_col_major(const void *src, fb_layout_t layout,
                                         fb_uplo_t uplo, int n, int kd,
                                         int src_ld, void *dst, int dst_ld,
                                         size_t elem_size) {
  int kl = 0;
  int ku = 0;

  switch (uplo) {
  case FB_UPPER:
    ku = kd;
    break;
  case FB_LOWER:
    kl = kd;
    break;
  default:
    return -1;
  }

  return fb_copy_general_band_to_col_major(src, layout, n, n, kl, ku,
                                           src_ld, dst, dst_ld, elem_size);
}

static int fb_copy_general_band_from_col_major(const void *src, int m, int n,
                                               int kl, int ku, int src_ld,
                                               fb_layout_t layout, void *dst,
                                               int dst_ld, size_t elem_size) {
  const unsigned char *src_bytes = (const unsigned char *)src;
  unsigned char *dst_bytes = (unsigned char *)dst;
  int i = 0;
  int j = 0;

  if (!src || !dst || m < 0 || n < 0 || kl < 0 || ku < 0 || elem_size == 0) {
    return -1;
  }
  if (src_ld < (kl + ku + 1)) {
    return -1;
  }

  if (layout == FB_LAYOUT_ROW_MAJOR) {
    if (dst_ld < n) {
      return -1;
    }

    memset(dst_bytes, 0,
           (size_t)(kl + ku + 1) * (size_t)((dst_ld > 0) ? dst_ld : 1) * elem_size);

    for (j = 0; j < n; ++j) {
      int i_start = ku - j;
      int i_end = m + ku - j;

      if (i_start < 0) {
        i_start = 0;
      }
      if (i_end > (kl + ku + 1)) {
        i_end = kl + ku + 1;
      }
      if (i_end > src_ld) {
        i_end = src_ld;
      }

      for (i = i_start; i < i_end; ++i) {
        memcpy(dst_bytes + ((((size_t)i * (size_t)dst_ld) + (size_t)j) * elem_size),
               src_bytes + ((((size_t)j * (size_t)src_ld) + (size_t)i) * elem_size),
               elem_size);
      }
    }

    return 0;
  }

  if (fb_lapack_layout_is_col_major(layout)) {
    return fb_copy_col_major_to_col_major(src, kl + ku + 1, n, src_ld,
                                          dst, dst_ld, elem_size);
  }

  return -1;
}

static int fb_copy_general_band_compact_row_to_expanded_col_major(
    const void *src, int m, int n, int kl, int ku, int src_ld, void *dst,
    int dst_ld, size_t elem_size) {
  const unsigned char *src_bytes = (const unsigned char *)src;
  unsigned char *dst_bytes = (unsigned char *)dst;
  int compact_rows = kl + ku + 1;
  int i = 0;
  int j = 0;

  if (!src || !dst || m < 0 || n < 0 || kl < 0 || ku < 0 || elem_size == 0) {
    return -1;
  }
  if (src_ld < n || dst_ld < (2 * kl + ku + 1)) {
    return -1;
  }

  memset(dst_bytes, 0,
         (size_t)dst_ld * (size_t)((n > 0) ? n : 1) * elem_size);

  for (j = 0; j < n; ++j) {
    int i_start = ku - j;
    int i_end = m + ku - j;

    if (i_start < 0) {
      i_start = 0;
    }
    if (i_end > compact_rows) {
      i_end = compact_rows;
    }

    for (i = i_start; i < i_end; ++i) {
      memcpy(dst_bytes + ((((size_t)j * (size_t)dst_ld) + (size_t)(i + kl)) *
                          elem_size),
             src_bytes + ((((size_t)i * (size_t)src_ld) + (size_t)j) *
                          elem_size),
             elem_size);
    }
  }

  return 0;
}

static int fb_copy_general_band_compact_row_from_expanded_col_major(
    const void *src, int m, int n, int kl, int ku, int src_ld, void *dst,
    int dst_ld, size_t elem_size) {
  const unsigned char *src_bytes = (const unsigned char *)src;
  unsigned char *dst_bytes = (unsigned char *)dst;
  int compact_rows = kl + ku + 1;
  int i = 0;
  int j = 0;

  if (!src || !dst || m < 0 || n < 0 || kl < 0 || ku < 0 || elem_size == 0) {
    return -1;
  }
  if (src_ld < (2 * kl + ku + 1) || dst_ld < n) {
    return -1;
  }

  memset(dst_bytes, 0,
         (size_t)compact_rows * (size_t)((dst_ld > 0) ? dst_ld : 1) *
             elem_size);

  for (j = 0; j < n; ++j) {
    int i_start = ku - j;
    int i_end = m + ku - j;

    if (i_start < 0) {
      i_start = 0;
    }
    if (i_end > compact_rows) {
      i_end = compact_rows;
    }

    for (i = i_start; i < i_end; ++i) {
      memcpy(dst_bytes + ((((size_t)i * (size_t)dst_ld) + (size_t)j) *
                          elem_size),
             src_bytes + ((((size_t)j * (size_t)src_ld) + (size_t)(i + kl)) *
                          elem_size),
             elem_size);
    }
  }

  return 0;
}

static int fb_copy_sym_band_from_col_major(const void *src, fb_layout_t layout,
                                           fb_uplo_t uplo, int n, int kd,
                                           int src_ld, void *dst, int dst_ld,
                                           size_t elem_size) {
  int kl = 0;
  int ku = 0;

  switch (uplo) {
  case FB_UPPER:
    ku = kd;
    break;
  case FB_LOWER:
    kl = kd;
    break;
  default:
    return -1;
  }

  return fb_copy_general_band_from_col_major(src, n, n, kl, ku, src_ld,
                                             layout, dst, dst_ld, elem_size);
}

static int fb_query_size_from_int(int query_size) {
  return (query_size > 1) ? query_size : 1;
}

static int fb_query_size_from_float(float query_size) {
  return (query_size > 1.0f) ? (int)query_size : 1;
}

static int fb_query_size_from_double(double query_size) {
  return (query_size > 1.0) ? (int)query_size : 1;
}

static int fb_query_size_from_cfloat(fb_complex_float_t query_size) {
  float real_part = 0.0f;
  memcpy(&real_part, &query_size, sizeof(real_part));
  return (real_part > 1.0f) ? (int)real_part : 1;
}

static int fb_query_size_from_cdouble(fb_complex_double_t query_size) {
  double real_part = 0.0;
  memcpy(&real_part, &query_size, sizeof(real_part));
  return (real_part > 1.0) ? (int)real_part : 1;
}

/* ============================================================================
 * Thunk helper macros
 *
 * These generate a pair of static thunk functions for operations that follow
 * the common integer + real-scalar pattern:
 *   cblas_OP (N, ALPHA, X, INCX, Y, INCY)
 *   OP_      (N*, ALPHA*, X, INCX*, Y, INCY*)
 * ========================================================================== */

/** AXPY family: (n, alpha:scalar, x, incx, y, incy) */
#define DEFINE_AXPY_THUNKS(OP, TYPE, OP_ID)                                    \
static void thunk_##OP##_cblas_to_fortran(                                      \
    int n, TYPE alpha, const TYPE *x, int incx, TYPE *y, int incy) {            \
    int  n_ = n, incx_ = incx, incy_ = incy;                                   \
    TYPE a_ = alpha;                                                             \
    if (!g_fortran_fn[OP_ID]) return;                                           \
    ((void(*)(const int*, const TYPE*, const TYPE*,                             \
              const int*, TYPE*, const int*))                                    \
     g_fortran_fn[OP_ID])(&n_, &a_, x, &incx_, y, &incy_);                     \
}                                                                               \
static void thunk_##OP##_fortran_to_cblas(                                      \
    const int *n, const TYPE *alpha, const TYPE *x,                             \
    const int *incx, TYPE *y, const int *incy) {                                \
    if (!g_cblas_fn[OP_ID]) return;                                             \
    ((void(*)(int, TYPE, const TYPE*, int, TYPE*, int))                         \
     g_cblas_fn[OP_ID])(*n, *alpha, x, *incx, y, *incy);                       \
}

/** SCAL family: (n, alpha:scalar, x, incx) */
#define DEFINE_SCAL_THUNKS(OP, TYPE, OP_ID)                                    \
static void thunk_##OP##_cblas_to_fortran(                                      \
    int n, TYPE alpha, TYPE *x, int incx) {                                     \
    int  n_ = n, incx_ = incx;                                                  \
    TYPE a_ = alpha;                                                             \
    if (!g_fortran_fn[OP_ID]) return;                                           \
    ((void(*)(const int*, const TYPE*, TYPE*, const int*))                      \
     g_fortran_fn[OP_ID])(&n_, &a_, x, &incx_);                                \
}                                                                               \
static void thunk_##OP##_fortran_to_cblas(                                      \
    const int *n, const TYPE *alpha, TYPE *x, const int *incx) {                \
    if (!g_cblas_fn[OP_ID]) return;                                             \
    ((void(*)(int, TYPE, TYPE*, int))                                            \
     g_cblas_fn[OP_ID])(*n, *alpha, x, *incx);                                 \
}

/** NRM2/ASUM family: (n, x, incx) → scalar RETURN (same register ABI) */
#define DEFINE_NRM2_THUNKS(OP, RTYPE, TYPE, OP_ID)                             \
static RTYPE thunk_##OP##_cblas_to_fortran(                                     \
    int n, const TYPE *x, int incx) {                                           \
    int n_ = n, incx_ = incx;                                                   \
    if (!g_fortran_fn[OP_ID]) return (RTYPE)0;                                  \
    return ((RTYPE(*)(const int*, const TYPE*, const int*))                     \
     g_fortran_fn[OP_ID])(&n_, x, &incx_);                                     \
}                                                                               \
static RTYPE thunk_##OP##_fortran_to_cblas(                                     \
    const int *n, const TYPE *x, const int *incx) {                             \
    if (!g_cblas_fn[OP_ID]) return (RTYPE)0;                                    \
    return ((RTYPE(*)(int, const TYPE*, int))                                   \
     g_cblas_fn[OP_ID])(*n, x, *incx);                                         \
}

  /* Alias-return family with corrected direction naming: used for newer op IDs
   * whose dedicated focused tests exercise the slot orientation directly. */
  #define DEFINE_NRM2_ALIAS_THUNKS(OP, RTYPE, TYPE, OP_ID)                        \
  static RTYPE thunk_##OP##_fortran_to_cblas(                                      \
    int n, const TYPE *x, int incx) {                                            \
    int n_ = n, incx_ = incx;                                                    \
    if (!g_fortran_fn[OP_ID]) return (RTYPE)0;                                   \
    return ((RTYPE(*)(const int*, const TYPE*, const int*))                      \
     g_fortran_fn[OP_ID])(&n_, x, &incx_);                                       \
  }                                                                                \
  static RTYPE thunk_##OP##_cblas_to_fortran(                                      \
    const int *n, const TYPE *x, const int *incx) {                              \
    if (!g_cblas_fn[OP_ID]) return (RTYPE)0;                                     \
    return ((RTYPE(*)(int, const TYPE*, int))                                    \
     g_cblas_fn[OP_ID])(*n, x, *incx);                                           \
  }

/** DOT family: (n, x, incx, y, incy) → scalar RETURN */
#define DEFINE_DOT_THUNKS(OP, TYPE, OP_ID)                                     \
static TYPE thunk_##OP##_cblas_to_fortran(                                      \
    int n, const TYPE *x, int incx, const TYPE *y, int incy) {                  \
    int n_ = n, incx_ = incx, incy_ = incy;                                    \
    if (!g_fortran_fn[OP_ID]) return (TYPE)0;                                   \
    return ((TYPE(*)(const int*, const TYPE*, const int*,                       \
                     const TYPE*, const int*))                                   \
     g_fortran_fn[OP_ID])(&n_, x, &incx_, y, &incy_);                          \
}                                                                               \
static TYPE thunk_##OP##_fortran_to_cblas(                                      \
    const int *n, const TYPE *x, const int *incx,                               \
    const TYPE *y, const int *incy) {                                            \
    if (!g_cblas_fn[OP_ID]) return (TYPE)0;                                     \
    return ((TYPE(*)(int, const TYPE*, int, const TYPE*, int))                  \
     g_cblas_fn[OP_ID])(*n, x, *incx, y, *incy);                               \
}

/** COPY/SWAP family: (n, x, incx, y, incy) — no scalar */
#define DEFINE_COPY_THUNKS(OP, TYPE, OP_ID)                                    \
static void thunk_##OP##_cblas_to_fortran(                                      \
    int n, const TYPE *x, int incx, TYPE *y, int incy) {                        \
    int n_ = n, incx_ = incx, incy_ = incy;                                    \
    if (!g_fortran_fn[OP_ID]) return;                                           \
    ((void(*)(const int*, const TYPE*, const int*, TYPE*, const int*))          \
     g_fortran_fn[OP_ID])(&n_, x, &incx_, y, &incy_);                          \
}                                                                               \
static void thunk_##OP##_fortran_to_cblas(                                      \
    const int *n, const TYPE *x, const int *incx, TYPE *y, const int *incy) {  \
    if (!g_cblas_fn[OP_ID]) return;                                             \
    ((void(*)(int, const TYPE*, int, TYPE*, int))                               \
     g_cblas_fn[OP_ID])(*n, x, *incx, y, *incy);                               \
}

/* Integer-only pointer conversion for complex AXPY (alpha is const void *,
 * same in CBLAS and Fortran — only n/incx/incy need pointer wrapping). */
#define DEFINE_CAXPY_THUNKS(OP, OP_ID)                                         \
static void thunk_##OP##_cblas_to_fortran(                                      \
    int n, const void *alpha, const void *x, int incx, void *y, int incy) {    \
    int n_ = n, incx_ = incx, incy_ = incy;                                    \
    if (!g_fortran_fn[OP_ID]) return;                                           \
    ((void(*)(const int*, const void*, const void*,                             \
              const int*, void*, const int*))                                    \
     g_fortran_fn[OP_ID])(&n_, alpha, x, &incx_, y, &incy_);                   \
}                                                                               \
static void thunk_##OP##_fortran_to_cblas(                                      \
    const int *n, const void *alpha, const void *x,                             \
    const int *incx, void *y, const int *incy) {                                \
    if (!g_cblas_fn[OP_ID]) return;                                             \
    ((void(*)(int, const void*, const void*, int, void*, int))                  \
     g_cblas_fn[OP_ID])(*n, alpha, x, *incx, y, *incy);                        \
}

/* ============================================================================
 * Thunk instantiation — BLAS Level 1
 * ========================================================================== */

DEFINE_AXPY_THUNKS(saxpy, float,  FB_OP_SAXPY)
DEFINE_AXPY_THUNKS(daxpy, double, FB_OP_DAXPY)
DEFINE_CAXPY_THUNKS(caxpy, FB_OP_CAXPY)
DEFINE_CAXPY_THUNKS(zaxpy, FB_OP_ZAXPY)

DEFINE_SCAL_THUNKS(sscal, float,  FB_OP_SSCAL)
DEFINE_SCAL_THUNKS(dscal, double, FB_OP_DSCAL)

DEFINE_COPY_THUNKS(scopy, float,  FB_OP_SCOPY)
DEFINE_COPY_THUNKS(dcopy, double, FB_OP_DCOPY)
DEFINE_COPY_THUNKS(sswap, float,  FB_OP_SSWAP)
DEFINE_COPY_THUNKS(dswap, double, FB_OP_DSWAP)

DEFINE_NRM2_THUNKS(snrm2,  float,  float,  FB_OP_SNRM2)
DEFINE_NRM2_THUNKS(dnrm2,  double, double, FB_OP_DNRM2)
DEFINE_NRM2_THUNKS(scnrm2, float,  float,  FB_OP_SCNRM2)
DEFINE_NRM2_THUNKS(dznrm2, double, double, FB_OP_DZNRM2)
DEFINE_NRM2_ALIAS_THUNKS(cnrm2,  float,  fb_complex_float_t,  FB_OP_CNRM2)
DEFINE_NRM2_ALIAS_THUNKS(znrm2,  double, fb_complex_double_t, FB_OP_ZNRM2)
DEFINE_NRM2_THUNKS(sasum,  float,  float,  FB_OP_SASUM)
DEFINE_NRM2_THUNKS(dasum,  double, double, FB_OP_DASUM)
DEFINE_NRM2_THUNKS(scasum, float,  float,  FB_OP_SCASUM)
DEFINE_NRM2_THUNKS(dzasum, double, double, FB_OP_DZASUM)

DEFINE_DOT_THUNKS(sdot, float,  FB_OP_SDOT)
DEFINE_DOT_THUNKS(ddot, double, FB_OP_DDOT)


/** ROT family: (n, x, incx, y, incy, c, s) */
#define DEFINE_ROT_THUNKS(OP, TYPE, OP_ID)                                      \
static void thunk_##OP##_cblas_to_fortran(                                      \
    int n, TYPE *x, int incx, TYPE *y, int incy, TYPE c, TYPE s) {              \
    int n_ = n, incx_ = incx, incy_ = incy;                                    \
    TYPE c_ = c, s_ = s;                                                        \
    if (!g_fortran_fn[OP_ID]) return;                                           \
    ((void(*)(const int*, TYPE*, const int*, TYPE*, const int*,                 \
              const TYPE*, const TYPE*))                                         \
     g_fortran_fn[OP_ID])(&n_, x, &incx_, y, &incy_, &c_, &s_);               \
}                                                                               \
static void thunk_##OP##_fortran_to_cblas(                                      \
    const int *n, TYPE *x, const int *incx, TYPE *y, const int *incy,          \
    const TYPE *c, const TYPE *s) {                                             \
    if (!g_cblas_fn[OP_ID]) return;                                             \
    ((void(*)(int, TYPE*, int, TYPE*, int, TYPE, TYPE))                         \
     g_cblas_fn[OP_ID])(*n, x, *incx, y, *incy, *c, *s);                      \
}

/** IAMAX family: (n, x, incx) → int */
#define DEFINE_IAMAX_THUNKS(OP, TYPE, OP_ID)                                   \
static int thunk_##OP##_cblas_to_fortran(                                       \
    int n, const TYPE *x, int incx) {                                           \
    int n_ = n, incx_ = incx;                                                   \
    if (!g_fortran_fn[OP_ID]) return 0;                                         \
    return ((int(*)(const int*, const TYPE*, const int*))                       \
     g_fortran_fn[OP_ID])(&n_, x, &incx_);                                     \
}                                                                               \
static int thunk_##OP##_fortran_to_cblas(                                       \
    const int *n, const TYPE *x, const int *incx) {                             \
    if (!g_cblas_fn[OP_ID]) return 0;                                           \
    return ((int(*)(int, const TYPE*, int))                                     \
     g_cblas_fn[OP_ID])(*n, x, *incx);                                         \
}

/* ============================================================================
 * Thunk helper macros — BLAS Level 2
 * ========================================================================== */

/** GEMV: (Order, Trans, M, N, alpha, A, lda, X, incx, beta, Y, incy) */
#define DEFINE_GEMV_THUNKS(OP, TYPE, OP_ID)                                    \
static void thunk_##OP##_cblas_to_fortran(                                      \
    int Order, int Trans, int M, int N, TYPE alpha,                             \
    const TYPE *A, int lda, const TYPE *X, int incx,                            \
    TYPE beta, TYPE *Y, int incy) {                                             \
    if (!g_fortran_fn[OP_ID]) return;                                           \
    int Order_=Order, Trans_=Trans, M_=M, N_=N;                                \
    TYPE alpha_=alpha, beta_=beta;                                              \
    int lda_=lda, incx_=incx, incy_=incy;                                      \
    ((void(*)(const int*, const int*, const int*, const int*,                   \
              const TYPE*, const TYPE*, const int*,                              \
              const TYPE*, const int*, const TYPE*, TYPE*, const int*))          \
     g_fortran_fn[OP_ID])(                                                      \
         &Order_, &Trans_, &M_, &N_,                                            \
         &alpha_, A, &lda_, X, &incx_, &beta_, Y, &incy_);                     \
}                                                                               \
static void thunk_##OP##_fortran_to_cblas(                                      \
    const int *Order, const int *Trans, const int *M, const int *N,             \
    const TYPE *alpha, const TYPE *A, const int *lda,                           \
    const TYPE *X, const int *incx, const TYPE *beta, TYPE *Y,                  \
    const int *incy) {                                                          \
    if (!g_cblas_fn[OP_ID]) return;                                             \
    ((void(*)(int, int, int, int, TYPE, const TYPE*, int,                       \
              const TYPE*, int, TYPE, TYPE*, int))                               \
     g_cblas_fn[OP_ID])(                                                        \
         *Order, *Trans, *M, *N,                                                \
         *alpha, A, *lda, X, *incx, *beta, Y, *incy);                          \
}

  /** GBMV: (Order, Trans, M, N, KL, KU, alpha, A, lda, X, incx, beta, Y, incy) */
  #define DEFINE_GBMV_THUNKS(OP, TYPE, OP_ID)                                    \
  static void thunk_##OP##_fortran_to_cblas(                                      \
    int Order, int Trans, int M, int N, int KL, int KU, TYPE alpha,            \
    const TYPE *A, int lda, const TYPE *X, int incx, TYPE beta, TYPE *Y,        \
    int incy) {                                                                 \
    if (!g_fortran_fn[OP_ID]) return;                                           \
    int Order_ = Order, Trans_ = Trans, M_ = M, N_ = N;                         \
    int KL_ = KL, KU_ = KU, lda_ = lda, incx_ = incx, incy_ = incy;             \
    TYPE alpha_ = alpha, beta_ = beta;                                          \
    ((void(*)(const int*, const int*, const int*, const int*,                   \
          const int*, const int*, const TYPE*, const TYPE*,                 \
          const int*, const TYPE*, const int*, const TYPE*,                 \
          TYPE*, const int*))                                               \
     g_fortran_fn[OP_ID])(                                                      \
       &Order_, &Trans_, &M_, &N_, &KL_, &KU_, &alpha_, A, &lda_, X,         \
       &incx_, &beta_, Y, &incy_);                                            \
  }                                                                               \
  static void thunk_##OP##_cblas_to_fortran(                                      \
    const int *Order, const int *Trans, const int *M, const int *N,            \
    const int *KL, const int *KU, const TYPE *alpha, const TYPE *A,            \
    const int *lda, const TYPE *X, const int *incx, const TYPE *beta,          \
    TYPE *Y, const int *incy) {                                                 \
    if (!g_cblas_fn[OP_ID]) return;                                             \
    ((void(*)(int, int, int, int, int, int, TYPE, const TYPE*, int,             \
          const TYPE*, int, TYPE, TYPE*, int))                              \
     g_cblas_fn[OP_ID])(                                                        \
       *Order, *Trans, *M, *N, *KL, *KU, *alpha, A, *lda, X, *incx,          \
       *beta, Y, *incy);                                                      \
  }

  /** CGBMV/ZGBMV: complex alpha/beta and buffers are pointer-based. */
  #define DEFINE_CGBMV_THUNKS(OP, OP_ID)                                         \
  static void thunk_##OP##_fortran_to_cblas(                                      \
    int Order, int Trans, int M, int N, int KL, int KU, const void *alpha,     \
    const void *A, int lda, const void *X, int incx, const void *beta,         \
    void *Y, int incy) {                                                        \
    if (!g_fortran_fn[OP_ID]) return;                                           \
    int Order_ = Order, Trans_ = Trans, M_ = M, N_ = N;                         \
    int KL_ = KL, KU_ = KU, lda_ = lda, incx_ = incx, incy_ = incy;             \
    ((void(*)(const int*, const int*, const int*, const int*,                   \
          const int*, const int*, const void*, const void*,                 \
          const int*, const void*, const int*, const void*,                 \
          void*, const int*))                                               \
     g_fortran_fn[OP_ID])(                                                      \
       &Order_, &Trans_, &M_, &N_, &KL_, &KU_, alpha, A, &lda_, X,           \
       &incx_, beta, Y, &incy_);                                              \
  }                                                                               \
  static void thunk_##OP##_cblas_to_fortran(                                      \
    const int *Order, const int *Trans, const int *M, const int *N,            \
    const int *KL, const int *KU, const void *alpha, const void *A,            \
    const int *lda, const void *X, const int *incx, const void *beta,          \
    void *Y, const int *incy) {                                                 \
    if (!g_cblas_fn[OP_ID]) return;                                             \
    ((void(*)(int, int, int, int, int, int, const void*, const void*, int,      \
          const void*, int, const void*, void*, int))                       \
     g_cblas_fn[OP_ID])(                                                        \
       *Order, *Trans, *M, *N, *KL, *KU, alpha, A, *lda, X, *incx,           \
       beta, Y, *incy);                                                       \
  }

  /** CGEMV/ZGEMV: complex alpha/beta and buffers are pointer-based. */
  #define DEFINE_CGEMV_THUNKS(OP, OP_ID)                                         \
  static void thunk_##OP##_fortran_to_cblas(                                      \
    int Order, int Trans, int M, int N, const void *alpha,                      \
    const void *A, int lda, const void *X, int incx,                            \
    const void *beta, void *Y, int incy) {                                      \
    if (!g_fortran_fn[OP_ID]) return;                                           \
    int Order_ = Order, Trans_ = Trans, M_ = M, N_ = N;                         \
    int lda_ = lda, incx_ = incx, incy_ = incy;                                 \
    ((void(*)(const int*, const int*, const int*, const int*,                   \
          const void*, const void*, const int*, const void*,                \
          const int*, const void*, void*, const int*))                      \
     g_fortran_fn[OP_ID])(                                                      \
       &Order_, &Trans_, &M_, &N_, alpha, A, &lda_, X, &incx_, beta, Y,      \
       &incy_);                                                               \
  }                                                                               \
  static void thunk_##OP##_cblas_to_fortran(                                      \
    const int *Order, const int *Trans, const int *M, const int *N,            \
    const void *alpha, const void *A, const int *lda, const void *X,           \
    const int *incx, const void *beta, void *Y, const int *incy) {             \
    if (!g_cblas_fn[OP_ID]) return;                                             \
    ((void(*)(int, int, int, int, const void*, const void*, int,                \
          const void*, int, const void*, void*, int))                       \
     g_cblas_fn[OP_ID])(                                                        \
       *Order, *Trans, *M, *N, alpha, A, *lda, X, *incx, beta, Y, *incy);    \
  }

  /** GBMVX: layout-aware general band matrix-vector multiply */
  #define DEFINE_GBMVX_THUNKS(OP, TYPE, OP_ID)                                   \
  static void thunk_##OP##_fortran_to_cblas(                                      \
    const fb_layout_t layout, const char trans, const int m, const int n,      \
    const int kl, const int ku, const TYPE alpha, TYPE *ab, const int ldab,    \
    TYPE *x, const int incx, const TYPE beta, TYPE *y, const int incy) {       \
    char trans_ = trans;                                                        \
    int m_ = m, n_ = n, kl_ = kl, ku_ = ku, ldab_ = ldab;                      \
    int incx_ = incx, incy_ = incy;                                             \
    int band_ld = (kl >= 0 && ku >= 0) ? (kl + ku + 1) : 1;                    \
    size_t ab_elems = (size_t)band_ld * (size_t)((n > 0) ? n : 1);             \
    TYPE alpha_ = alpha, beta_ = beta;                                          \
    TYPE *ab_col = NULL;                                                        \
    TYPE *ab_arg = ab;                                                          \
    if (!g_fortran_fn[OP_ID]) return;                                           \
    if (m < 0 || n < 0 || kl < 0 || ku < 0) return;                            \
    if (layout == FB_LAYOUT_ROW_MAJOR) {                                        \
      if ((n > 0 && !ab) || ldab < n) return;                                \
      ab_col = (TYPE *)malloc(ab_elems * sizeof(TYPE));                       \
      if (!ab_col) return;                                                    \
      if (fb_copy_general_band_to_col_major(ab, layout, m, n, kl, ku, ldab,  \
                          ab_col, band_ld,                  \
                          sizeof(TYPE)) != 0) {             \
        free(ab_col);                                                       \
        return;                                                             \
      }                                                                       \
      ab_arg = ab_col;                                                        \
      ldab_ = band_ld;                                                        \
    } else if (fb_lapack_layout_is_col_major(layout)) {                         \
      if ((n > 0 && !ab) || ldab < band_ld) return;                          \
    } else {                                                                    \
      return;                                                                 \
    }                                                                           \
    ((void (*)(char *, int *, int *, int *, int *, TYPE *, TYPE *, int *,      \
           TYPE *, int *, TYPE *, TYPE *, int *))                          \
      g_fortran_fn[OP_ID])(                                                   \
        &trans_, &m_, &n_, &kl_, &ku_, &alpha_, ab_arg, &ldab_, x,         \
        &incx_, &beta_, y, &incy_);                                        \
    free(ab_col);                                                               \
  }                                                                               \
  static void thunk_##OP##_cblas_to_fortran(                                      \
    char *trans, int *m, int *n, int *kl, int *ku, TYPE *alpha, TYPE *ab,      \
    int *ldab, TYPE *x, int *incx, TYPE *beta, TYPE *y, int *incy) {           \
    if (!g_cblas_fn[OP_ID] || !trans || !m || !n || !kl || !ku || !alpha ||    \
      !ldab || !incx || !beta || !incy) return;                               \
    ((void (*)(fb_layout_t, char, int, int, int, int, TYPE, TYPE *, int,       \
           TYPE *, int, TYPE, TYPE *, int))g_cblas_fn[OP_ID])(             \
      FB_LAYOUT_COL_MAJOR, *trans, *m, *n, *kl, *ku, *alpha, ab, *ldab, x,   \
      *incx, *beta, y, *incy);                                                \
  }

/** GER: (Order, M, N, alpha, X, incx, Y, incy, A, lda) */
#define DEFINE_GER_THUNKS(OP, TYPE, OP_ID)                                     \
static void thunk_##OP##_cblas_to_fortran(                                      \
    int Order, int M, int N, TYPE alpha,                                        \
    const TYPE *X, int incx, const TYPE *Y, int incy, TYPE *A, int lda) {       \
    if (!g_fortran_fn[OP_ID]) return;                                           \
    int Order_=Order, M_=M, N_=N, incx_=incx, incy_=incy, lda_=lda;           \
    TYPE alpha_=alpha;                                                          \
    ((void(*)(const int*, const int*, const int*, const TYPE*,                  \
              const TYPE*, const int*, const TYPE*, const int*,                 \
              TYPE*, const int*))                                                \
     g_fortran_fn[OP_ID])(                                                      \
         &Order_, &M_, &N_, &alpha_,                                            \
         X, &incx_, Y, &incy_, A, &lda_);                                      \
}                                                                               \
static void thunk_##OP##_fortran_to_cblas(                                      \
    const int *Order, const int *M, const int *N, const TYPE *alpha,            \
    const TYPE *X, const int *incx, const TYPE *Y, const int *incy,             \
    TYPE *A, const int *lda) {                                                  \
    if (!g_cblas_fn[OP_ID]) return;                                             \
    ((void(*)(int, int, int, TYPE, const TYPE*, int, const TYPE*, int,          \
              TYPE*, int))                                                       \
     g_cblas_fn[OP_ID])(                                                        \
         *Order, *M, *N, *alpha, X, *incx, Y, *incy, A, *lda);                 \
}

  /** CGERC/ZGERC/CGERU/ZGERU: complex alpha and buffers are pointer-based. */
  #define DEFINE_CGER_THUNKS(OP, OP_ID)                                          \
  static void thunk_##OP##_fortran_to_cblas(                                      \
    int Order, int M, int N, const void *alpha,                                 \
    const void *X, int incx, const void *Y, int incy, void *A, int lda) {      \
    if (!g_fortran_fn[OP_ID]) return;                                           \
    int Order_ = Order, M_ = M, N_ = N;                                         \
    int incx_ = incx, incy_ = incy, lda_ = lda;                                 \
    ((void(*)(const int*, const int*, const int*, const void*,                  \
          const void*, const int*, const void*, const int*,                 \
          void*, const int*))                                               \
     g_fortran_fn[OP_ID])(                                                      \
       &Order_, &M_, &N_, alpha, X, &incx_, Y, &incy_, A, &lda_);            \
  }                                                                               \
  static void thunk_##OP##_cblas_to_fortran(                                      \
    const int *Order, const int *M, const int *N, const void *alpha,           \
    const void *X, const int *incx, const void *Y, const int *incy,            \
    void *A, const int *lda) {                                                  \
    if (!g_cblas_fn[OP_ID]) return;                                             \
    ((void(*)(int, int, int, const void*, const void*, int,                     \
          const void*, int, void*, int))                                    \
     g_cblas_fn[OP_ID])(                                                        \
       *Order, *M, *N, alpha, X, *incx, Y, *incy, A, *lda);                  \
  }

/** SYR: (Order, Uplo, N, alpha, X, incx, A, lda) */
#define DEFINE_SYR_THUNKS(OP, TYPE, OP_ID)                                     \
static void thunk_##OP##_cblas_to_fortran(                                      \
    int Order, int Uplo, int N, TYPE alpha,                                     \
    const TYPE *X, int incx, TYPE *A, int lda) {                                \
    if (!g_fortran_fn[OP_ID]) return;                                           \
    int Order_=Order, Uplo_=Uplo, N_=N, incx_=incx, lda_=lda;                 \
    TYPE alpha_=alpha;                                                          \
    ((void(*)(const int*, const int*, const int*, const TYPE*,                  \
              const TYPE*, const int*, TYPE*, const int*))                       \
     g_fortran_fn[OP_ID])(                                                      \
         &Order_, &Uplo_, &N_, &alpha_, X, &incx_, A, &lda_);                  \
}                                                                               \
static void thunk_##OP##_fortran_to_cblas(                                      \
    const int *Order, const int *Uplo, const int *N, const TYPE *alpha,         \
    const TYPE *X, const int *incx, TYPE *A, const int *lda) {                  \
    if (!g_cblas_fn[OP_ID]) return;                                             \
    ((void(*)(int, int, int, TYPE, const TYPE*, int, TYPE*, int))               \
     g_cblas_fn[OP_ID])(*Order, *Uplo, *N, *alpha, X, *incx, A, *lda);         \
}

  /** CSYR/ZSYR: alpha and data buffers are already pointer-based. */
  #define DEFINE_CSYR_THUNKS(OP, OP_ID)                                          \
  static void thunk_##OP##_fortran_to_cblas(                                      \
    int Order, int Uplo, int N, const void *alpha,                              \
    const void *X, int incx, void *A, int lda) {                                \
    if (!g_fortran_fn[OP_ID]) return;                                           \
    int Order_ = Order, Uplo_ = Uplo, N_ = N;                                   \
    int incx_ = incx, lda_ = lda;                                               \
    ((void(*)(const int*, const int*, const int*, const void*,                  \
          const void*, const int*, void*, const int*))                      \
     g_fortran_fn[OP_ID])(                                                      \
       &Order_, &Uplo_, &N_, alpha, X, &incx_, A, &lda_);                     \
  }                                                                               \
  static void thunk_##OP##_cblas_to_fortran(                                      \
    const int *Order, const int *Uplo, const int *N, const void *alpha,         \
    const void *X, const int *incx, void *A, const int *lda) {                  \
    if (!g_cblas_fn[OP_ID]) return;                                             \
    ((void(*)(int, int, int, const void*, const void*, int, void*, int))        \
     g_cblas_fn[OP_ID])(*Order, *Uplo, *N, alpha, X, *incx, A, *lda);          \
  }

/** SYR2: (Order, Uplo, N, alpha, X, incx, Y, incy, A, lda) */
#define DEFINE_SYR2_THUNKS(OP, TYPE, OP_ID)                                    \
static void thunk_##OP##_cblas_to_fortran(                                      \
    int Order, int Uplo, int N, TYPE alpha,                                     \
    const TYPE *X, int incx, const TYPE *Y, int incy, TYPE *A, int lda) {       \
    if (!g_fortran_fn[OP_ID]) return;                                           \
    int Order_=Order, Uplo_=Uplo, N_=N, incx_=incx, incy_=incy, lda_=lda;     \
    TYPE alpha_=alpha;                                                          \
    ((void(*)(const int*, const int*, const int*, const TYPE*,                  \
              const TYPE*, const int*, const TYPE*, const int*,                 \
              TYPE*, const int*))                                                \
     g_fortran_fn[OP_ID])(                                                      \
         &Order_, &Uplo_, &N_, &alpha_,                                         \
         X, &incx_, Y, &incy_, A, &lda_);                                      \
}                                                                               \
static void thunk_##OP##_fortran_to_cblas(                                      \
    const int *Order, const int *Uplo, const int *N, const TYPE *alpha,         \
    const TYPE *X, const int *incx, const TYPE *Y, const int *incy,             \
    TYPE *A, const int *lda) {                                                  \
    if (!g_cblas_fn[OP_ID]) return;                                             \
    ((void(*)(int, int, int, TYPE, const TYPE*, int, const TYPE*, int,          \
              TYPE*, int))                                                       \
     g_cblas_fn[OP_ID])(*Order, *Uplo, *N, *alpha, X, *incx, Y, *incy, A, *lda); \
}

  /** CSYR2/ZSYR2: alpha and data buffers are already pointer-based. */
  #define DEFINE_CSYR2_THUNKS(OP, OP_ID)                                         \
  static void thunk_##OP##_fortran_to_cblas(                                      \
    int Order, int Uplo, int N, const void *alpha,                              \
    const void *X, int incx, const void *Y, int incy,                           \
    void *A, int lda) {                                                         \
    if (!g_fortran_fn[OP_ID]) return;                                           \
    int Order_ = Order, Uplo_ = Uplo, N_ = N;                                   \
    int incx_ = incx, incy_ = incy, lda_ = lda;                                 \
    ((void(*)(const int*, const int*, const int*, const void*,                  \
          const void*, const int*, const void*, const int*,                \
          void*, const int*))                                               \
     g_fortran_fn[OP_ID])(                                                      \
       &Order_, &Uplo_, &N_, alpha, X, &incx_, Y, &incy_, A, &lda_);         \
  }                                                                               \
  static void thunk_##OP##_cblas_to_fortran(                                      \
    const int *Order, const int *Uplo, const int *N, const void *alpha,         \
    const void *X, const int *incx, const void *Y, const int *incy,             \
    void *A, const int *lda) {                                                  \
    if (!g_cblas_fn[OP_ID]) return;                                             \
    ((void(*)(int, int, int, const void*, const void*, int,                     \
          const void*, int, void*, int))                                    \
     g_cblas_fn[OP_ID])(*Order, *Uplo, *N, alpha, X, *incx, Y, *incy, A, *lda); \
  }

  /** SBMV: (Order, Uplo, N, K, alpha, A, lda, X, incx, beta, Y, incy) */
  #define DEFINE_SBMV_THUNKS(OP, TYPE, OP_ID)                                    \
  static void thunk_##OP##_fortran_to_cblas(                                      \
    int Order, int Uplo, int N, int K, TYPE alpha,                              \
    const TYPE *A, int lda, const TYPE *X, int incx,                            \
    TYPE beta, TYPE *Y, int incy) {                                             \
    if (!g_fortran_fn[OP_ID]) return;                                           \
    int Order_ = Order, Uplo_ = Uplo, N_ = N, K_ = K;                           \
    int lda_ = lda, incx_ = incx, incy_ = incy;                                 \
    TYPE alpha_ = alpha, beta_ = beta;                                          \
    ((void(*)(const int*, const int*, const int*, const int*,                   \
          const TYPE*, const TYPE*, const int*, const TYPE*,                \
          const int*, const TYPE*, TYPE*, const int*))                      \
     g_fortran_fn[OP_ID])(                                                      \
       &Order_, &Uplo_, &N_, &K_, &alpha_, A, &lda_, X, &incx_, &beta_, Y,   \
       &incy_);                                                               \
  }                                                                               \
  static void thunk_##OP##_cblas_to_fortran(                                      \
    const int *Order, const int *Uplo, const int *N, const int *K,             \
    const TYPE *alpha, const TYPE *A, const int *lda, const TYPE *X,           \
    const int *incx, const TYPE *beta, TYPE *Y, const int *incy) {             \
    if (!g_cblas_fn[OP_ID]) return;                                             \
    ((void(*)(int, int, int, int, TYPE, const TYPE*, int, const TYPE*, int,     \
          TYPE, TYPE*, int))                                                \
     g_cblas_fn[OP_ID])(                                                        \
       *Order, *Uplo, *N, *K, *alpha, A, *lda, X, *incx, *beta, Y, *incy);   \
  }

  /** SPMV: (Order, Uplo, N, alpha, AP, X, incx, beta, Y, incy) */
  #define DEFINE_SPMV_THUNKS(OP, TYPE, OP_ID)                                    \
  static void thunk_##OP##_fortran_to_cblas(                                      \
    int Order, int Uplo, int N, TYPE alpha, const TYPE *AP,                     \
    const TYPE *X, int incx, TYPE beta, TYPE *Y, int incy) {                    \
    if (!g_fortran_fn[OP_ID]) return;                                           \
    int Order_ = Order, Uplo_ = Uplo, N_ = N;                                   \
    int incx_ = incx, incy_ = incy;                                             \
    TYPE alpha_ = alpha, beta_ = beta;                                          \
    ((void(*)(const int*, const int*, const int*, const TYPE*,                  \
          const TYPE*, const TYPE*, const int*, const TYPE*,                \
          TYPE*, const int*))                                               \
     g_fortran_fn[OP_ID])(                                                      \
       &Order_, &Uplo_, &N_, &alpha_, AP, X, &incx_, &beta_, Y, &incy_);     \
  }                                                                               \
  static void thunk_##OP##_cblas_to_fortran(                                      \
    const int *Order, const int *Uplo, const int *N, const TYPE *alpha,        \
    const TYPE *AP, const TYPE *X, const int *incx, const TYPE *beta,          \
    TYPE *Y, const int *incy) {                                                 \
    if (!g_cblas_fn[OP_ID]) return;                                             \
    ((void(*)(int, int, int, TYPE, const TYPE*, const TYPE*, int, TYPE,         \
          TYPE*, int))                                                      \
     g_cblas_fn[OP_ID])(                                                        \
       *Order, *Uplo, *N, *alpha, AP, X, *incx, *beta, Y, *incy);            \
  }

/** SPR: (Order, Uplo, N, alpha, X, incx, AP) */
#define DEFINE_SPR_THUNKS(OP, TYPE, OP_ID)                                     \
static void thunk_##OP##_fortran_to_cblas(                                      \
    int Order, int Uplo, int N, TYPE alpha, const TYPE *X, int incx,           \
    TYPE *AP) {                                                                 \
    if (!g_fortran_fn[OP_ID]) return;                                           \
    int Order_ = Order, Uplo_ = Uplo, N_ = N, incx_ = incx;                    \
    TYPE alpha_ = alpha;                                                        \
    ((void(*)(const int*, const int*, const int*, const TYPE*,                  \
              const TYPE*, const int*, TYPE*))                                  \
     g_fortran_fn[OP_ID])(                                                      \
         &Order_, &Uplo_, &N_, &alpha_, X, &incx_, AP);                        \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    const int *Order, const int *Uplo, const int *N, const TYPE *alpha,        \
    const TYPE *X, const int *incx, TYPE *AP) {                                 \
    if (!g_cblas_fn[OP_ID]) return;                                             \
    ((void(*)(int, int, int, TYPE, const TYPE*, int, TYPE*))                    \
     g_cblas_fn[OP_ID])(*Order, *Uplo, *N, *alpha, X, *incx, AP);              \
}

/** SPR2: (Order, Uplo, N, alpha, X, incx, Y, incy, AP) */
#define DEFINE_SPR2_THUNKS(OP, TYPE, OP_ID)                                    \
static void thunk_##OP##_fortran_to_cblas(                                      \
    int Order, int Uplo, int N, TYPE alpha, const TYPE *X, int incx,           \
    const TYPE *Y, int incy, TYPE *AP) {                                        \
    if (!g_fortran_fn[OP_ID]) return;                                           \
    int Order_ = Order, Uplo_ = Uplo, N_ = N;                                   \
    int incx_ = incx, incy_ = incy;                                             \
    TYPE alpha_ = alpha;                                                        \
    ((void(*)(const int*, const int*, const int*, const TYPE*,                  \
              const TYPE*, const int*, const TYPE*, const int*,                 \
              TYPE*))                                                           \
     g_fortran_fn[OP_ID])(                                                      \
         &Order_, &Uplo_, &N_, &alpha_, X, &incx_, Y, &incy_, AP);             \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
    const int *Order, const int *Uplo, const int *N, const TYPE *alpha,        \
    const TYPE *X, const int *incx, const TYPE *Y, const int *incy,            \
    TYPE *AP) {                                                                 \
    if (!g_cblas_fn[OP_ID]) return;                                             \
    ((void(*)(int, int, int, TYPE, const TYPE*, int, const TYPE*, int,          \
              TYPE*))                                                           \
     g_cblas_fn[OP_ID])(*Order, *Uplo, *N, *alpha, X, *incx, Y, *incy, AP);    \
}

/** SYMV: (Order, Uplo, N, alpha, A, lda, X, incx, beta, Y, incy) */
#define DEFINE_SYMV_THUNKS(OP, TYPE, OP_ID)                                    \
static void thunk_##OP##_cblas_to_fortran(                                      \
    int Order, int Uplo, int N, TYPE alpha,                                     \
    const TYPE *A, int lda, const TYPE *X, int incx,                            \
    TYPE beta, TYPE *Y, int incy) {                                             \
    if (!g_fortran_fn[OP_ID]) return;                                           \
    int Order_=Order, Uplo_=Uplo, N_=N, lda_=lda, incx_=incx, incy_=incy;     \
    TYPE alpha_=alpha, beta_=beta;                                              \
    ((void(*)(const int*, const int*, const int*, const TYPE*,                  \
              const TYPE*, const int*, const TYPE*, const int*,                 \
              const TYPE*, TYPE*, const int*))                                   \
     g_fortran_fn[OP_ID])(                                                      \
         &Order_, &Uplo_, &N_, &alpha_,                                         \
         A, &lda_, X, &incx_, &beta_, Y, &incy_);                              \
}                                                                               \
static void thunk_##OP##_fortran_to_cblas(                                      \
    const int *Order, const int *Uplo, const int *N, const TYPE *alpha,         \
    const TYPE *A, const int *lda, const TYPE *X, const int *incx,              \
    const TYPE *beta, TYPE *Y, const int *incy) {                               \
    if (!g_cblas_fn[OP_ID]) return;                                             \
    ((void(*)(int, int, int, TYPE, const TYPE*, int, const TYPE*, int,          \
              TYPE, TYPE*, int))                                                 \
     g_cblas_fn[OP_ID])(                                                        \
         *Order, *Uplo, *N, *alpha, A, *lda, X, *incx, *beta, Y, *incy);       \
}

  /** CSYMV/ZSYMV: alpha/beta and data buffers are already pointer-based. */
  #define DEFINE_CSYMV_THUNKS(OP, OP_ID)                                         \
  static void thunk_##OP##_fortran_to_cblas(                                      \
    int Order, int Uplo, int N, const void *alpha,                              \
    const void *A, int lda, const void *X, int incx,                            \
    const void *beta, void *Y, int incy) {                                      \
    if (!g_fortran_fn[OP_ID]) return;                                           \
    int Order_ = Order, Uplo_ = Uplo, N_ = N;                                   \
    int lda_ = lda, incx_ = incx, incy_ = incy;                                 \
    ((void(*)(const int*, const int*, const int*, const void*,                  \
          const void*, const int*, const void*, const int*,                \
          const void*, void*, const int*))                                  \
     g_fortran_fn[OP_ID])(                                                      \
       &Order_, &Uplo_, &N_, alpha, A, &lda_, X, &incx_, beta, Y, &incy_);   \
  }                                                                               \
  static void thunk_##OP##_cblas_to_fortran(                                      \
    const int *Order, const int *Uplo, const int *N, const void *alpha,         \
    const void *A, const int *lda, const void *X, const int *incx,              \
    const void *beta, void *Y, const int *incy) {                               \
    if (!g_cblas_fn[OP_ID]) return;                                             \
    ((void(*)(int, int, int, const void*, const void*, int,                     \
          const void*, int, const void*, void*, int))                       \
     g_cblas_fn[OP_ID])(                                                        \
       *Order, *Uplo, *N, alpha, A, *lda, X, *incx, beta, Y, *incy);         \
  }

  /** CHEMV/ZHEMV: by-value complex scalars, pointer-based arrays. */
  #define DEFINE_HEMV_THUNKS(OP, TYPE, OP_ID)                                    \
  static void thunk_##OP##_fortran_to_cblas(                                      \
    int Order, int Uplo, int N, TYPE alpha,                                     \
    const TYPE *A, int lda, const TYPE *X, int incx,                            \
    TYPE beta, TYPE *Y, int incy) {                                             \
    if (!g_fortran_fn[OP_ID]) return;                                           \
    int Order_ = Order, Uplo_ = Uplo, N_ = N;                                   \
    int lda_ = lda, incx_ = incx, incy_ = incy;                                 \
    TYPE alpha_ = alpha, beta_ = beta;                                          \
    ((void(*)(const int*, const int*, const int*, const TYPE*,                  \
          const TYPE*, const int*, const TYPE*, const int*,                \
          const TYPE*, TYPE*, const int*))                                  \
     g_fortran_fn[OP_ID])(                                                      \
       &Order_, &Uplo_, &N_, &alpha_, A, &lda_, X, &incx_, &beta_, Y,        \
       &incy_);                                                               \
  }                                                                               \
  static void thunk_##OP##_cblas_to_fortran(                                      \
    const int *Order, const int *Uplo, const int *N, const TYPE *alpha,         \
    const TYPE *A, const int *lda, const TYPE *X, const int *incx,              \
    const TYPE *beta, TYPE *Y, const int *incy) {                               \
    if (!g_cblas_fn[OP_ID]) return;                                             \
    ((void(*)(int, int, int, TYPE, const TYPE*, int, const TYPE*, int,          \
          TYPE, TYPE*, int))                                                \
     g_cblas_fn[OP_ID])(*Order, *Uplo, *N, *alpha, A, *lda, X, *incx, *beta,   \
              Y, *incy);                                              \
  }

  /** CHPMV/ZHPMV: by-value complex scalars, packed Hermitian storage. */
  #define DEFINE_HPMV_THUNKS(OP, TYPE, OP_ID)                                    \
  static void thunk_##OP##_fortran_to_cblas(                                      \
    int Order, int Uplo, int N, TYPE alpha,                                     \
    const TYPE *AP, const TYPE *X, int incx,                                    \
    TYPE beta, TYPE *Y, int incy) {                                             \
    if (!g_fortran_fn[OP_ID]) return;                                           \
    int Order_ = Order, Uplo_ = Uplo, N_ = N;                                   \
    int incx_ = incx, incy_ = incy;                                             \
    TYPE alpha_ = alpha, beta_ = beta;                                          \
    ((void(*)(const int*, const int*, const int*, const TYPE*,                  \
          const TYPE*, const TYPE*, const int*, const TYPE*,                \
          TYPE*, const int*))                                               \
     g_fortran_fn[OP_ID])(                                                      \
       &Order_, &Uplo_, &N_, &alpha_, AP, X, &incx_, &beta_, Y, &incy_);      \
  }                                                                               \
  static void thunk_##OP##_cblas_to_fortran(                                      \
    const int *Order, const int *Uplo, const int *N, const TYPE *alpha,         \
    const TYPE *AP, const TYPE *X, const int *incx, const TYPE *beta,           \
    TYPE *Y, const int *incy) {                                                 \
    if (!g_cblas_fn[OP_ID]) return;                                             \
    ((void(*)(int, int, int, TYPE, const TYPE*, const TYPE*, int,               \
          TYPE, TYPE*, int))                                                \
     g_cblas_fn[OP_ID])(*Order, *Uplo, *N, *alpha, AP, X, *incx, *beta, Y,     \
                        *incy);                                                 \
  }

  /** CHBMV/ZHBMV: by-value complex scalars, Hermitian band storage. */
  #define DEFINE_HBMV_THUNKS(OP, TYPE, OP_ID)                                    \
  static void thunk_##OP##_fortran_to_cblas(                                      \
    int Order, int Uplo, int N, int K, TYPE alpha,                              \
    const TYPE *A, int lda, const TYPE *X, int incx,                            \
    TYPE beta, TYPE *Y, int incy) {                                             \
    if (!g_fortran_fn[OP_ID]) return;                                           \
    int Order_ = Order, Uplo_ = Uplo, N_ = N, K_ = K, lda_ = lda;               \
    int incx_ = incx, incy_ = incy;                                             \
    TYPE alpha_ = alpha, beta_ = beta;                                          \
    ((void(*)(const int*, const int*, const int*, const int*, const TYPE*,      \
          const TYPE*, const int*, const TYPE*, const int*, const TYPE*,    \
          TYPE*, const int*))                                               \
     g_fortran_fn[OP_ID])(                                                      \
       &Order_, &Uplo_, &N_, &K_, &alpha_, A, &lda_, X, &incx_, &beta_, Y,     \
       &incy_);                                                                 \
  }                                                                               \
  static void thunk_##OP##_cblas_to_fortran(                                      \
    const int *Order, const int *Uplo, const int *N, const int *K,              \
    const TYPE *alpha, const TYPE *A, const int *lda, const TYPE *X,            \
    const int *incx, const TYPE *beta, TYPE *Y, const int *incy) {              \
    if (!g_cblas_fn[OP_ID]) return;                                             \
    ((void(*)(int, int, int, int, TYPE, const TYPE*, int, const TYPE*, int,     \
          TYPE, TYPE*, int))                                                \
     g_cblas_fn[OP_ID])(*Order, *Uplo, *N, *K, *alpha, A, *lda, X, *incx,       \
                        *beta, Y, *incy);                                       \
  }

  /** CHER/ZHER: real alpha, full Hermitian rank-1 update. */
  #define DEFINE_HER_THUNKS(OP, REAL_TYPE, COMPLEX_TYPE, OP_ID)                  \
  static void thunk_##OP##_fortran_to_cblas(                                      \
    int Order, int Uplo, int N, REAL_TYPE alpha,                                \
    const COMPLEX_TYPE *X, int incx, COMPLEX_TYPE *A, int lda) {                \
    if (!g_fortran_fn[OP_ID]) return;                                           \
    int Order_ = Order, Uplo_ = Uplo, N_ = N;                                   \
    int incx_ = incx, lda_ = lda;                                               \
    REAL_TYPE alpha_ = alpha;                                                   \
    ((void(*)(const int*, const int*, const int*, const REAL_TYPE*,             \
          const COMPLEX_TYPE*, const int*, COMPLEX_TYPE*, const int*))      \
     g_fortran_fn[OP_ID])(                                                      \
       &Order_, &Uplo_, &N_, &alpha_, X, &incx_, A, &lda_);                    \
  }                                                                               \
  static void thunk_##OP##_cblas_to_fortran(                                      \
    const int *Order, const int *Uplo, const int *N, const REAL_TYPE *alpha,    \
    const COMPLEX_TYPE *X, const int *incx, COMPLEX_TYPE *A, const int *lda) {  \
    if (!g_cblas_fn[OP_ID]) return;                                             \
    ((void(*)(int, int, int, REAL_TYPE, const COMPLEX_TYPE*, int,               \
          COMPLEX_TYPE*, int))                                               \
     g_cblas_fn[OP_ID])(*Order, *Uplo, *N, *alpha, X, *incx, A, *lda);         \
  }

  /** CHER2/ZHER2: by-value complex alpha, full Hermitian rank-2 update. */
  #define DEFINE_HER2_THUNKS(OP, TYPE, OP_ID)                                    \
  static void thunk_##OP##_fortran_to_cblas(                                      \
    int Order, int Uplo, int N, TYPE alpha,                                     \
    const TYPE *X, int incx, const TYPE *Y, int incy, TYPE *A, int lda) {       \
    if (!g_fortran_fn[OP_ID]) return;                                           \
    int Order_ = Order, Uplo_ = Uplo, N_ = N;                                   \
    int incx_ = incx, incy_ = incy, lda_ = lda;                                 \
    TYPE alpha_ = alpha;                                                        \
    ((void(*)(const int*, const int*, const int*, const TYPE*,                  \
          const TYPE*, const int*, const TYPE*, const int*, TYPE*,          \
          const int*))                                                      \
     g_fortran_fn[OP_ID])(                                                      \
       &Order_, &Uplo_, &N_, &alpha_, X, &incx_, Y, &incy_, A, &lda_);         \
  }                                                                               \
  static void thunk_##OP##_cblas_to_fortran(                                      \
    const int *Order, const int *Uplo, const int *N, const TYPE *alpha,         \
    const TYPE *X, const int *incx, const TYPE *Y, const int *incy,             \
    TYPE *A, const int *lda) {                                                  \
    if (!g_cblas_fn[OP_ID]) return;                                             \
    ((void(*)(int, int, int, TYPE, const TYPE*, int, const TYPE*, int,          \
          TYPE*, int))                                                       \
     g_cblas_fn[OP_ID])(*Order, *Uplo, *N, *alpha, X, *incx, Y, *incy, A,      \
                        *lda);                                                  \
  }

  /** CHPR/ZHPR: real alpha, packed Hermitian rank-1 update. */
  #define DEFINE_HPR_THUNKS(OP, REAL_TYPE, COMPLEX_TYPE, OP_ID)                  \
  static void thunk_##OP##_fortran_to_cblas(                                      \
    int Order, int Uplo, int N, REAL_TYPE alpha,                                \
    const COMPLEX_TYPE *X, int incx, COMPLEX_TYPE *AP) {                        \
    if (!g_fortran_fn[OP_ID]) return;                                           \
    int Order_ = Order, Uplo_ = Uplo, N_ = N, incx_ = incx;                     \
    REAL_TYPE alpha_ = alpha;                                                   \
    ((void(*)(const int*, const int*, const int*, const REAL_TYPE*,             \
          const COMPLEX_TYPE*, const int*, COMPLEX_TYPE*))                  \
     g_fortran_fn[OP_ID])(                                                      \
       &Order_, &Uplo_, &N_, &alpha_, X, &incx_, AP);                          \
  }                                                                               \
  static void thunk_##OP##_cblas_to_fortran(                                      \
    const int *Order, const int *Uplo, const int *N, const REAL_TYPE *alpha,    \
    const COMPLEX_TYPE *X, const int *incx, COMPLEX_TYPE *AP) {                 \
    if (!g_cblas_fn[OP_ID]) return;                                             \
    ((void(*)(int, int, int, REAL_TYPE, const COMPLEX_TYPE*, int,               \
          COMPLEX_TYPE*))                                                    \
     g_cblas_fn[OP_ID])(*Order, *Uplo, *N, *alpha, X, *incx, AP);              \
  }

  /** CHPR2/ZHPR2: by-value complex alpha, packed Hermitian rank-2 update. */
  #define DEFINE_HPR2_THUNKS(OP, TYPE, OP_ID)                                    \
  static void thunk_##OP##_fortran_to_cblas(                                      \
    int Order, int Uplo, int N, TYPE alpha,                                     \
    const TYPE *X, int incx, const TYPE *Y, int incy, TYPE *AP) {               \
    if (!g_fortran_fn[OP_ID]) return;                                           \
    int Order_ = Order, Uplo_ = Uplo, N_ = N;                                   \
    int incx_ = incx, incy_ = incy;                                             \
    TYPE alpha_ = alpha;                                                        \
    ((void(*)(const int*, const int*, const int*, const TYPE*,                  \
          const TYPE*, const int*, const TYPE*, const int*, TYPE*))        \
     g_fortran_fn[OP_ID])(                                                      \
       &Order_, &Uplo_, &N_, &alpha_, X, &incx_, Y, &incy_, AP);               \
  }                                                                               \
  static void thunk_##OP##_cblas_to_fortran(                                      \
    const int *Order, const int *Uplo, const int *N, const TYPE *alpha,         \
    const TYPE *X, const int *incx, const TYPE *Y, const int *incy, TYPE *AP) { \
    if (!g_cblas_fn[OP_ID]) return;                                             \
    ((void(*)(int, int, int, TYPE, const TYPE*, int, const TYPE*, int,          \
          TYPE*))                                                            \
     g_cblas_fn[OP_ID])(*Order, *Uplo, *N, *alpha, X, *incx, Y, *incy, AP);    \
  }

/** TBMV/TBSV: (Order, Uplo, Trans, Diag, N, K, A, lda, X, incx) */
#define DEFINE_TBMV_THUNKS(OP, TYPE, OP_ID)                                    \
static void thunk_##OP##_fortran_to_cblas(                                      \
  int Order, int Uplo, int Trans, int Diag, int N, int K,                       \
  const TYPE *A, int lda, TYPE *X, int incx) {                                  \
  if (!g_fortran_fn[OP_ID]) return;                                             \
  int Order_=Order, Uplo_=Uplo, Trans_=Trans, Diag_=Diag;                       \
  int N_=N, K_=K, lda_=lda, incx_=incx;                                         \
  ((void(*)(const int*, const int*, const int*, const int*, const int*,         \
        const int*, const TYPE*, const int*, TYPE*, const int*))                \
   g_fortran_fn[OP_ID])(                                                        \
     &Order_, &Uplo_, &Trans_, &Diag_, &N_, &K_, A, &lda_, X, &incx_);         \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
  const int *Order, const int *Uplo, const int *Trans, const int *Diag,         \
  const int *N, const int *K, const TYPE *A, const int *lda,                    \
  TYPE *X, const int *incx) {                                                   \
  if (!g_cblas_fn[OP_ID]) return;                                               \
  ((void(*)(int, int, int, int, int, int, const TYPE*, int, TYPE*, int))        \
   g_cblas_fn[OP_ID])(                                                          \
     *Order, *Uplo, *Trans, *Diag, *N, *K, A, *lda, X, *incx);                 \
}

/** TPMV/TPSV: (Order, Uplo, Trans, Diag, N, AP, X, incx) */
#define DEFINE_TPMV_THUNKS(OP, TYPE, OP_ID)                                    \
static void thunk_##OP##_fortran_to_cblas(                                      \
  int Order, int Uplo, int Trans, int Diag, int N,                              \
  const TYPE *AP, TYPE *X, int incx) {                                          \
  if (!g_fortran_fn[OP_ID]) return;                                             \
  int Order_=Order, Uplo_=Uplo, Trans_=Trans, Diag_=Diag;                       \
  int N_=N, incx_=incx;                                                         \
  ((void(*)(const int*, const int*, const int*, const int*, const int*,         \
        const TYPE*, TYPE*, const int*))                                        \
   g_fortran_fn[OP_ID])(                                                        \
     &Order_, &Uplo_, &Trans_, &Diag_, &N_, AP, X, &incx_);                    \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
  const int *Order, const int *Uplo, const int *Trans, const int *Diag,         \
  const int *N, const TYPE *AP, TYPE *X, const int *incx) {                     \
  if (!g_cblas_fn[OP_ID]) return;                                               \
  ((void(*)(int, int, int, int, int, const TYPE*, TYPE*, int))                  \
   g_cblas_fn[OP_ID])(                                                          \
     *Order, *Uplo, *Trans, *Diag, *N, AP, X, *incx);                          \
}

/** TRMV/TRSV: (Order, Uplo, Trans, Diag, N, A, lda, X, incx) */
#define DEFINE_TRMV_THUNKS(OP, TYPE, OP_ID)                                    \
static void thunk_##OP##_fortran_to_cblas(                                      \
  int Order, int Uplo, int Trans, int Diag, int N,                            \
  const TYPE *A, int lda, TYPE *X, int incx) {                                \
  if (!g_fortran_fn[OP_ID]) return;                                           \
  int Order_=Order, Uplo_=Uplo, Trans_=Trans, Diag_=Diag, N_=N;             \
  int lda_=lda, incx_=incx;                                                   \
  ((void(*)(const int*, const int*, const int*, const int*, const int*,       \
        const TYPE*, const int*, TYPE*, const int*))                       \
   g_fortran_fn[OP_ID])(                                                      \
     &Order_, &Uplo_, &Trans_, &Diag_, &N_, A, &lda_, X, &incx_);          \
}                                                                               \
static void thunk_##OP##_cblas_to_fortran(                                      \
  const int *Order, const int *Uplo, const int *Trans, const int *Diag,       \
  const int *N, const TYPE *A, const int *lda, TYPE *X, const int *incx) {    \
  if (!g_cblas_fn[OP_ID]) return;                                             \
  ((void(*)(int, int, int, int, int, const TYPE*, int, TYPE*, int))           \
   g_cblas_fn[OP_ID])(                                                        \
     *Order, *Uplo, *Trans, *Diag, *N, A, *lda, X, *incx);                 \
}

/* ============================================================================
 * Thunk helper macros — BLAS Level 3
 * ========================================================================== */

/** CGEMM/ZGEMM: alpha/beta are const void* (complex) */
#define DEFINE_CGEMM_THUNKS(OP, OP_ID)                                         \
static void thunk_##OP##_cblas_to_fortran(                                      \
    int Order, int TransA, int TransB, int M, int N, int K,                     \
    const void *alpha, const void *A, int lda,                                  \
                       const void *B, int ldb,                                  \
    const void *beta, void *C, int ldc) {                                       \
    if (!g_fortran_fn[OP_ID]) return;                                           \
    int Order_=Order, TransA_=TransA, TransB_=TransB;                          \
    int M_=M, N_=N, K_=K, lda_=lda, ldb_=ldb, ldc_=ldc;                       \
    ((void(*)(const int*, const int*, const int*,                               \
              const int*, const int*, const int*,                               \
              const void*, const void*, const int*,                              \
                           const void*, const int*,                              \
              const void*, void*, const int*))                                   \
     g_fortran_fn[OP_ID])(                                                      \
         &Order_, &TransA_, &TransB_,                                           \
         &M_, &N_, &K_,                                                         \
         alpha, A, &lda_, B, &ldb_, beta, C, &ldc_);                           \
}                                                                               \
static void thunk_##OP##_fortran_to_cblas(                                      \
    const int *Order, const int *TransA, const int *TransB,                     \
    const int *M, const int *N, const int *K,                                   \
    const void *alpha, const void *A, const int *lda,                           \
                       const void *B, const int *ldb,                           \
    const void *beta, void *C, const int *ldc) {                                \
    if (!g_cblas_fn[OP_ID]) return;                                             \
    ((void(*)(int, int, int, int, int, int,                                     \
              const void*, const void*, int, const void*, int,                  \
              const void*, void*, int))                                          \
     g_cblas_fn[OP_ID])(                                                        \
         *Order, *TransA, *TransB,                                              \
         *M, *N, *K,                                                            \
         alpha, A, *lda, B, *ldb, beta, C, *ldc);                              \
}

/** SYMM: (Order, Side, Uplo, M, N, alpha, A, lda, B, ldb, beta, C, ldc) */
#define DEFINE_SYMM_THUNKS(OP, TYPE, OP_ID)                                    \
static void thunk_##OP##_cblas_to_fortran(                                      \
    int Order, int Side, int Uplo, int M, int N,                                \
    TYPE alpha, const TYPE *A, int lda, const TYPE *B, int ldb,                 \
    TYPE beta, TYPE *C, int ldc) {                                              \
    if (!g_fortran_fn[OP_ID]) return;                                           \
    int Order_=Order, Side_=Side, Uplo_=Uplo, M_=M, N_=N;                     \
    TYPE alpha_=alpha, beta_=beta;                                              \
    int lda_=lda, ldb_=ldb, ldc_=ldc;                                          \
    ((void(*)(const int*, const int*, const int*, const int*, const int*,       \
              const TYPE*, const TYPE*, const int*,                              \
                            const TYPE*, const int*,                             \
              const TYPE*, TYPE*, const int*))                                   \
     g_fortran_fn[OP_ID])(                                                      \
         &Order_, &Side_, &Uplo_, &M_, &N_,                                    \
         &alpha_, A, &lda_, B, &ldb_, &beta_, C, &ldc_);                       \
}                                                                               \
static void thunk_##OP##_fortran_to_cblas(                                      \
    const int *Order, const int *Side, const int *Uplo,                         \
    const int *M, const int *N,                                                 \
    const TYPE *alpha, const TYPE *A, const int *lda,                           \
                       const TYPE *B, const int *ldb,                           \
    const TYPE *beta, TYPE *C, const int *ldc) {                                \
    if (!g_cblas_fn[OP_ID]) return;                                             \
    ((void(*)(int, int, int, int, int,                                          \
              TYPE, const TYPE*, int, const TYPE*, int,                         \
              TYPE, TYPE*, int))                                                 \
     g_cblas_fn[OP_ID])(                                                        \
         *Order, *Side, *Uplo, *M, *N,                                         \
         *alpha, A, *lda, B, *ldb, *beta, C, *ldc);                            \
}

    /** CSYMM/ZSYMM: alpha/beta and matrix buffers are pointer-based. */
    #define DEFINE_CSYMM_THUNKS(OP, OP_ID)                                         \
    static void thunk_##OP##_fortran_to_cblas(                                      \
      int Order, int Side, int Uplo, int M, int N,                                \
      const void *alpha, const void *A, int lda,                                  \
                 const void *B, int ldb,                                  \
      const void *beta, void *C, int ldc) {                                       \
      if (!g_fortran_fn[OP_ID]) return;                                           \
      int Order_ = Order, Side_ = Side, Uplo_ = Uplo;                             \
      int M_ = M, N_ = N, lda_ = lda, ldb_ = ldb, ldc_ = ldc;                     \
      ((void(*)(const int*, const int*, const int*, const int*, const int*,       \
            const void*, const void*, const int*,                              \
                   const void*, const int*,                              \
            const void*, void*, const int*))                                   \
       g_fortran_fn[OP_ID])(                                                      \
         &Order_, &Side_, &Uplo_, &M_, &N_,                                     \
         alpha, A, &lda_, B, &ldb_, beta, C, &ldc_);                            \
    }                                                                               \
    static void thunk_##OP##_cblas_to_fortran(                                      \
      const int *Order, const int *Side, const int *Uplo,                         \
      const int *M, const int *N,                                                 \
      const void *alpha, const void *A, const int *lda,                           \
                 const void *B, const int *ldb,                           \
      const void *beta, void *C, const int *ldc) {                                \
      if (!g_cblas_fn[OP_ID]) return;                                             \
      ((void(*)(int, int, int, int, int,                                          \
            const void*, const void*, int, const void*, int,                  \
            const void*, void*, int))                                          \
       g_cblas_fn[OP_ID])(                                                        \
         *Order, *Side, *Uplo, *M, *N,                                          \
         alpha, A, *lda, B, *ldb, beta, C, *ldc);                               \
    }

/** SYRK: (Order, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc) */
#define DEFINE_SYRK_THUNKS(OP, TYPE, OP_ID)                                    \
static void thunk_##OP##_cblas_to_fortran(                                      \
    int Order, int Uplo, int Trans, int N, int K,                               \
    TYPE alpha, const TYPE *A, int lda, TYPE beta, TYPE *C, int ldc) {          \
    if (!g_fortran_fn[OP_ID]) return;                                           \
    int Order_=Order, Uplo_=Uplo, Trans_=Trans, N_=N, K_=K;                   \
    TYPE alpha_=alpha, beta_=beta;                                              \
    int lda_=lda, ldc_=ldc;                                                     \
    ((void(*)(const int*, const int*, const int*, const int*, const int*,       \
              const TYPE*, const TYPE*, const int*,                              \
              const TYPE*, TYPE*, const int*))                                   \
     g_fortran_fn[OP_ID])(                                                      \
         &Order_, &Uplo_, &Trans_, &N_, &K_,                                   \
         &alpha_, A, &lda_, &beta_, C, &ldc_);                                 \
}                                                                               \
static void thunk_##OP##_fortran_to_cblas(                                      \
    const int *Order, const int *Uplo, const int *Trans,                        \
    const int *N, const int *K,                                                 \
    const TYPE *alpha, const TYPE *A, const int *lda,                           \
    const TYPE *beta, TYPE *C, const int *ldc) {                                \
    if (!g_cblas_fn[OP_ID]) return;                                             \
    ((void(*)(int, int, int, int, int,                                          \
              TYPE, const TYPE*, int, TYPE, TYPE*, int))                         \
     g_cblas_fn[OP_ID])(                                                        \
         *Order, *Uplo, *Trans, *N, *K,                                        \
         *alpha, A, *lda, *beta, C, *ldc);                                     \
}

    /** CSYRK/ZSYRK: alpha/beta and matrix buffers are pointer-based. */
    #define DEFINE_CSYRK_THUNKS(OP, OP_ID)                                         \
    static void thunk_##OP##_fortran_to_cblas(                                      \
      int Order, int Uplo, int Trans, int N, int K,                               \
      const void *alpha, const void *A, int lda,                                  \
      const void *beta, void *C, int ldc) {                                       \
      if (!g_fortran_fn[OP_ID]) return;                                           \
      int Order_ = Order, Uplo_ = Uplo, Trans_ = Trans;                           \
      int N_ = N, K_ = K, lda_ = lda, ldc_ = ldc;                                 \
      ((void(*)(const int*, const int*, const int*, const int*, const int*,       \
            const void*, const void*, const int*,                              \
            const void*, void*, const int*))                                   \
       g_fortran_fn[OP_ID])(                                                      \
         &Order_, &Uplo_, &Trans_, &N_, &K_,                                    \
         alpha, A, &lda_, beta, C, &ldc_);                                      \
    }                                                                               \
    static void thunk_##OP##_cblas_to_fortran(                                      \
      const int *Order, const int *Uplo, const int *Trans,                        \
      const int *N, const int *K,                                                 \
      const void *alpha, const void *A, const int *lda,                           \
      const void *beta, void *C, const int *ldc) {                                \
      if (!g_cblas_fn[OP_ID]) return;                                             \
      ((void(*)(int, int, int, int, int,                                          \
            const void*, const void*, int, const void*, void*, int))          \
       g_cblas_fn[OP_ID])(                                                        \
         *Order, *Uplo, *Trans, *N, *K,                                         \
         alpha, A, *lda, beta, C, *ldc);                                        \
    }

  /** HERK: (Order, Uplo, Trans, N, K, alpha_real, A, lda, beta_real, C, ldc) */
  #define DEFINE_HERK_THUNKS(OP, REAL_TYPE, COMPLEX_TYPE, OP_ID)                 \
  static void thunk_##OP##_fortran_to_cblas(                                      \
    int Order, int Uplo, int Trans, int N, int K,                               \
    REAL_TYPE alpha, const COMPLEX_TYPE *A, int lda,                            \
    REAL_TYPE beta, COMPLEX_TYPE *C, int ldc) {                                 \
    if (!g_fortran_fn[OP_ID]) return;                                           \
    int Order_=Order, Uplo_=Uplo, Trans_=Trans, N_=N, K_=K;                     \
    REAL_TYPE alpha_=alpha, beta_=beta;                                         \
    int lda_=lda, ldc_=ldc;                                                     \
    ((void(*)(const int*, const int*, const int*, const int*, const int*,       \
          const REAL_TYPE*, const COMPLEX_TYPE*, const int*,                \
          const REAL_TYPE*, COMPLEX_TYPE*, const int*))                     \
     g_fortran_fn[OP_ID])(                                                      \
       &Order_, &Uplo_, &Trans_, &N_, &K_,                                    \
       &alpha_, A, &lda_, &beta_, C, &ldc_);                                  \
  }                                                                               \
  static void thunk_##OP##_cblas_to_fortran(                                      \
    const int *Order, const int *Uplo, const int *Trans,                        \
    const int *N, const int *K,                                                 \
    const REAL_TYPE *alpha, const COMPLEX_TYPE *A, const int *lda,              \
    const REAL_TYPE *beta, COMPLEX_TYPE *C, const int *ldc) {                   \
    if (!g_cblas_fn[OP_ID]) return;                                             \
    ((void(*)(int, int, int, int, int,                                          \
          REAL_TYPE, const COMPLEX_TYPE*, int,                              \
          REAL_TYPE, COMPLEX_TYPE*, int))                                   \
     g_cblas_fn[OP_ID])(                                                        \
       *Order, *Uplo, *Trans, *N, *K,                                         \
       *alpha, A, *lda, *beta, C, *ldc);                                      \
  }

/** SYR2K: (Order, Uplo, Trans, N, K, alpha, A, lda, B, ldb, beta, C, ldc) */
#define DEFINE_SYR2K_THUNKS(OP, TYPE, OP_ID)                                   \
static void thunk_##OP##_cblas_to_fortran(                                      \
    int Order, int Uplo, int Trans, int N, int K,                               \
    TYPE alpha, const TYPE *A, int lda, const TYPE *B, int ldb,                 \
    TYPE beta, TYPE *C, int ldc) {                                              \
    if (!g_fortran_fn[OP_ID]) return;                                           \
    int Order_=Order, Uplo_=Uplo, Trans_=Trans, N_=N, K_=K;                   \
    TYPE alpha_=alpha, beta_=beta;                                              \
    int lda_=lda, ldb_=ldb, ldc_=ldc;                                          \
    ((void(*)(const int*, const int*, const int*, const int*, const int*,       \
              const TYPE*, const TYPE*, const int*,                              \
                            const TYPE*, const int*,                             \
              const TYPE*, TYPE*, const int*))                                   \
     g_fortran_fn[OP_ID])(                                                      \
         &Order_, &Uplo_, &Trans_, &N_, &K_,                                   \
         &alpha_, A, &lda_, B, &ldb_, &beta_, C, &ldc_);                       \
}                                                                               \
static void thunk_##OP##_fortran_to_cblas(                                      \
    const int *Order, const int *Uplo, const int *Trans,                        \
    const int *N, const int *K,                                                 \
    const TYPE *alpha, const TYPE *A, const int *lda,                           \
                       const TYPE *B, const int *ldb,                           \
    const TYPE *beta, TYPE *C, const int *ldc) {                                \
    if (!g_cblas_fn[OP_ID]) return;                                             \
    ((void(*)(int, int, int, int, int,                                          \
              TYPE, const TYPE*, int, const TYPE*, int,                         \
              TYPE, TYPE*, int))                                                 \
     g_cblas_fn[OP_ID])(                                                        \
         *Order, *Uplo, *Trans, *N, *K,                                        \
         *alpha, A, *lda, B, *ldb, *beta, C, *ldc);                            \
}

    /** CSYR2K/ZSYR2K: alpha/beta and matrix buffers are pointer-based. */
    #define DEFINE_CSYR2K_THUNKS(OP, OP_ID)                                        \
    static void thunk_##OP##_fortran_to_cblas(                                      \
      int Order, int Uplo, int Trans, int N, int K,                               \
      const void *alpha, const void *A, int lda,                                  \
      const void *B, int ldb, const void *beta, void *C, int ldc) {               \
      if (!g_fortran_fn[OP_ID]) return;                                           \
      int Order_ = Order, Uplo_ = Uplo, Trans_ = Trans;                           \
      int N_ = N, K_ = K, lda_ = lda, ldb_ = ldb, ldc_ = ldc;                     \
      ((void(*)(const int*, const int*, const int*, const int*, const int*,       \
            const void*, const void*, const int*,                              \
            const void*, const int*, const void*, void*, const int*))          \
       g_fortran_fn[OP_ID])(                                                      \
         &Order_, &Uplo_, &Trans_, &N_, &K_,                                    \
         alpha, A, &lda_, B, &ldb_, beta, C, &ldc_);                            \
    }                                                                               \
    static void thunk_##OP##_cblas_to_fortran(                                      \
      const int *Order, const int *Uplo, const int *Trans,                        \
      const int *N, const int *K,                                                 \
      const void *alpha, const void *A, const int *lda,                           \
      const void *B, const int *ldb, const void *beta, void *C,                   \
      const int *ldc) {                                                           \
      if (!g_cblas_fn[OP_ID]) return;                                             \
      ((void(*)(int, int, int, int, int,                                          \
            const void*, const void*, int, const void*, int,                  \
            const void*, void*, int))                                          \
       g_cblas_fn[OP_ID])(                                                        \
         *Order, *Uplo, *Trans, *N, *K,                                         \
         alpha, A, *lda, B, *ldb, beta, C, *ldc);                               \
    }

  /** HER2K: (Order, Uplo, Trans, N, K, alpha_complex, A, lda, B, ldb, beta_real, C, ldc) */
  #define DEFINE_HER2K_THUNKS(OP, COMPLEX_TYPE, REAL_TYPE, OP_ID)                \
  static void thunk_##OP##_fortran_to_cblas(                                      \
    int Order, int Uplo, int Trans, int N, int K,                               \
    COMPLEX_TYPE alpha, const COMPLEX_TYPE *A, int lda,                         \
    const COMPLEX_TYPE *B, int ldb,                                             \
    REAL_TYPE beta, COMPLEX_TYPE *C, int ldc) {                                 \
    if (!g_fortran_fn[OP_ID]) return;                                           \
    int Order_=Order, Uplo_=Uplo, Trans_=Trans, N_=N, K_=K;                     \
    COMPLEX_TYPE alpha_=alpha;                                                  \
    REAL_TYPE beta_=beta;                                                       \
    int lda_=lda, ldb_=ldb, ldc_=ldc;                                           \
    ((void(*)(const int*, const int*, const int*, const int*, const int*,       \
          const COMPLEX_TYPE*, const COMPLEX_TYPE*, const int*,             \
          const COMPLEX_TYPE*, const int*,                                  \
          const REAL_TYPE*, COMPLEX_TYPE*, const int*))                     \
     g_fortran_fn[OP_ID])(                                                      \
       &Order_, &Uplo_, &Trans_, &N_, &K_,                                    \
       &alpha_, A, &lda_, B, &ldb_, &beta_, C, &ldc_);                        \
  }                                                                               \
  static void thunk_##OP##_cblas_to_fortran(                                      \
    const int *Order, const int *Uplo, const int *Trans,                        \
    const int *N, const int *K,                                                 \
    const COMPLEX_TYPE *alpha, const COMPLEX_TYPE *A, const int *lda,           \
    const COMPLEX_TYPE *B, const int *ldb,                                      \
    const REAL_TYPE *beta, COMPLEX_TYPE *C, const int *ldc) {                   \
    if (!g_cblas_fn[OP_ID]) return;                                             \
    ((void(*)(int, int, int, int, int,                                          \
          COMPLEX_TYPE, const COMPLEX_TYPE*, int,                           \
          const COMPLEX_TYPE*, int,                                         \
          REAL_TYPE, COMPLEX_TYPE*, int))                                   \
     g_cblas_fn[OP_ID])(                                                        \
       *Order, *Uplo, *Trans, *N, *K,                                         \
       *alpha, A, *lda, B, *ldb, *beta, C, *ldc);                             \
  }

/** TRMM/TRSM: (Order, Side, Uplo, Trans, Diag, M, N, alpha, A, lda, B, ldb) */
#define DEFINE_TRMM_THUNKS(OP, TYPE, OP_ID)                                    \
static void thunk_##OP##_cblas_to_fortran(                                      \
    int Order, int Side, int Uplo, int Trans, int Diag,                         \
    int M, int N, TYPE alpha, const TYPE *A, int lda, TYPE *B, int ldb) {       \
    if (!g_fortran_fn[OP_ID]) return;                                           \
    int Order_=Order, Side_=Side, Uplo_=Uplo, Trans_=Trans, Diag_=Diag;       \
    int M_=M, N_=N, lda_=lda, ldb_=ldb;                                        \
    TYPE alpha_=alpha;                                                          \
    ((void(*)(const int*, const int*, const int*, const int*, const int*,       \
              const int*, const int*, const TYPE*,                               \
              const TYPE*, const int*, TYPE*, const int*))                       \
     g_fortran_fn[OP_ID])(                                                      \
         &Order_, &Side_, &Uplo_, &Trans_, &Diag_,                             \
         &M_, &N_, &alpha_, A, &lda_, B, &ldb_);                               \
}                                                                               \
static void thunk_##OP##_fortran_to_cblas(                                      \
    const int *Order, const int *Side, const int *Uplo,                         \
    const int *Trans, const int *Diag,                                          \
    const int *M, const int *N, const TYPE *alpha,                              \
    const TYPE *A, const int *lda, TYPE *B, const int *ldb) {                   \
    if (!g_cblas_fn[OP_ID]) return;                                             \
    ((void(*)(int, int, int, int, int, int, int,                                \
              TYPE, const TYPE*, int, TYPE*, int))                               \
     g_cblas_fn[OP_ID])(                                                        \
         *Order, *Side, *Uplo, *Trans, *Diag,                                  \
         *M, *N, *alpha, A, *lda, B, *ldb);                                    \
}

    /** CTRMM/ZTRMM/CTRSM/ZTRSM: complex alpha and matrix buffers are pointer-based. */
    #define DEFINE_CTRMM_THUNKS(OP, OP_ID)                                         \
    static void thunk_##OP##_fortran_to_cblas(                                      \
      int Order, int Side, int Uplo, int Trans, int Diag,                         \
      int M, int N, const void *alpha, const void *A, int lda,                    \
      void *B, int ldb) {                                                         \
      if (!g_fortran_fn[OP_ID]) return;                                           \
      int Order_ = Order, Side_ = Side, Uplo_ = Uplo;                             \
      int Trans_ = Trans, Diag_ = Diag, M_ = M, N_ = N;                           \
      int lda_ = lda, ldb_ = ldb;                                                 \
      ((void(*)(const int*, const int*, const int*, const int*, const int*,       \
            const int*, const int*, const void*, const void*, const int*,      \
            void*, const int*))                                                \
       g_fortran_fn[OP_ID])(                                                      \
         &Order_, &Side_, &Uplo_, &Trans_, &Diag_,                              \
         &M_, &N_, alpha, A, &lda_, B, &ldb_);                                  \
    }                                                                               \
    static void thunk_##OP##_cblas_to_fortran(                                      \
      const int *Order, const int *Side, const int *Uplo,                         \
      const int *Trans, const int *Diag,                                          \
      const int *M, const int *N, const void *alpha,                              \
      const void *A, const int *lda, void *B, const int *ldb) {                   \
      if (!g_cblas_fn[OP_ID]) return;                                             \
      ((void(*)(int, int, int, int, int, int, int,                                \
            const void*, const void*, int, void*, int))                        \
       g_cblas_fn[OP_ID])(                                                        \
         *Order, *Side, *Uplo, *Trans, *Diag,                                   \
         *M, *N, alpha, A, *lda, B, *ldb);                                      \
    }

/* ============================================================================
 * Thunk instantiation — BLAS Level 1 completion (rot, iamax)
 * ========================================================================== */

DEFINE_ROT_THUNKS(srot, float,  FB_OP_SROT)
DEFINE_ROT_THUNKS(drot, double, FB_OP_DROT)
DEFINE_IAMAX_THUNKS(isamax, float,  FB_OP_ISAMAX)
DEFINE_IAMAX_THUNKS(idamax, double, FB_OP_IDAMAX)

/* ============================================================================
 * Thunk instantiation — BLAS Level 2
 * ========================================================================== */

DEFINE_GBMV_THUNKS(sgbmv, float, FB_OP_SGBMV)
DEFINE_GBMV_THUNKS(dgbmv, double, FB_OP_DGBMV)
DEFINE_CGBMV_THUNKS(cgbmv, FB_OP_CGBMV)
DEFINE_CGBMV_THUNKS(zgbmv, FB_OP_ZGBMV)
DEFINE_GEMV_THUNKS(sgemv, float,  FB_OP_SGEMV)
DEFINE_GEMV_THUNKS(dgemv, double, FB_OP_DGEMV)
DEFINE_CGEMV_THUNKS(cgemv, FB_OP_CGEMV)
DEFINE_CGEMV_THUNKS(zgemv, FB_OP_ZGEMV)
DEFINE_GBMVX_THUNKS(sgbmvx, float, FB_OP_SGBMVX)
DEFINE_GBMVX_THUNKS(dgbmvx, double, FB_OP_DGBMVX)
DEFINE_GBMVX_THUNKS(cgbmvx, fb_complex_float_t, FB_OP_CGBMVX)
DEFINE_GBMVX_THUNKS(zgbmvx, fb_complex_double_t, FB_OP_ZGBMVX)
DEFINE_GER_THUNKS(sger,   float,  FB_OP_SGER)
DEFINE_GER_THUNKS(dger,   double, FB_OP_DGER)
DEFINE_CGER_THUNKS(cgerc, FB_OP_CGERC)
DEFINE_CGER_THUNKS(zgerc, FB_OP_ZGERC)
DEFINE_CGER_THUNKS(cgeru, FB_OP_CGERU)
DEFINE_CGER_THUNKS(zgeru, FB_OP_ZGERU)
DEFINE_SYR_THUNKS(ssyr,   float,  FB_OP_SSYR)
DEFINE_SYR_THUNKS(dsyr,   double, FB_OP_DSYR)
DEFINE_CSYR_THUNKS(csyr, FB_OP_CSYR)
DEFINE_CSYR_THUNKS(zsyr, FB_OP_ZSYR)
DEFINE_SYR2_THUNKS(ssyr2, float,  FB_OP_SSYR2)
DEFINE_SYR2_THUNKS(dsyr2, double, FB_OP_DSYR2)
DEFINE_CSYR2_THUNKS(csyr2, FB_OP_CSYR2)
DEFINE_CSYR2_THUNKS(zsyr2, FB_OP_ZSYR2)
DEFINE_TBMV_THUNKS(stbmv, float, FB_OP_STBMV)
DEFINE_TBMV_THUNKS(dtbmv, double, FB_OP_DTBMV)
DEFINE_TBMV_THUNKS(ctbmv, fb_complex_float_t, FB_OP_CTBMV)
DEFINE_TBMV_THUNKS(ztbmv, fb_complex_double_t, FB_OP_ZTBMV)
DEFINE_TBMV_THUNKS(stbsv, float, FB_OP_STBSV)
DEFINE_TBMV_THUNKS(dtbsv, double, FB_OP_DTBSV)
DEFINE_TBMV_THUNKS(ctbsv, fb_complex_float_t, FB_OP_CTBSV)
DEFINE_TBMV_THUNKS(ztbsv, fb_complex_double_t, FB_OP_ZTBSV)
DEFINE_TPMV_THUNKS(stpmv, float, FB_OP_STPMV)
DEFINE_TPMV_THUNKS(dtpmv, double, FB_OP_DTPMV)
DEFINE_TPMV_THUNKS(ctpmv, fb_complex_float_t, FB_OP_CTPMV)
DEFINE_TPMV_THUNKS(ztpmv, fb_complex_double_t, FB_OP_ZTPMV)
DEFINE_TPMV_THUNKS(stpsv, float, FB_OP_STPSV)
DEFINE_TPMV_THUNKS(dtpsv, double, FB_OP_DTPSV)
DEFINE_TPMV_THUNKS(ctpsv, fb_complex_float_t, FB_OP_CTPSV)
DEFINE_TPMV_THUNKS(ztpsv, fb_complex_double_t, FB_OP_ZTPSV)
DEFINE_HBMV_THUNKS(chbmv, fb_complex_float_t, FB_OP_CHBMV)
DEFINE_HBMV_THUNKS(zhbmv, fb_complex_double_t, FB_OP_ZHBMV)
DEFINE_HBMV_THUNKS(csbmv, fb_complex_float_t, FB_OP_CSBMV)
DEFINE_HBMV_THUNKS(zsbmv, fb_complex_double_t, FB_OP_ZSBMV)
DEFINE_HPMV_THUNKS(chpmv, fb_complex_float_t, FB_OP_CHPMV)
DEFINE_HPMV_THUNKS(zhpmv, fb_complex_double_t, FB_OP_ZHPMV)
DEFINE_HEMV_THUNKS(chemv, fb_complex_float_t, FB_OP_CHEMV)
DEFINE_HEMV_THUNKS(zhemv, fb_complex_double_t, FB_OP_ZHEMV)
DEFINE_HER_THUNKS(cher, float, fb_complex_float_t, FB_OP_CHER)
DEFINE_HER_THUNKS(zher, double, fb_complex_double_t, FB_OP_ZHER)
DEFINE_HER2_THUNKS(cher2, fb_complex_float_t, FB_OP_CHER2)
DEFINE_HER2_THUNKS(zher2, fb_complex_double_t, FB_OP_ZHER2)
DEFINE_HPR_THUNKS(chpr, float, fb_complex_float_t, FB_OP_CHPR)
DEFINE_HPR_THUNKS(zhpr, double, fb_complex_double_t, FB_OP_ZHPR)
DEFINE_HPR2_THUNKS(chpr2, fb_complex_float_t, FB_OP_CHPR2)
DEFINE_HPR2_THUNKS(zhpr2, fb_complex_double_t, FB_OP_ZHPR2)
DEFINE_SBMV_THUNKS(ssbmv, float, FB_OP_SSBMV)
DEFINE_SBMV_THUNKS(dsbmv, double, FB_OP_DSBMV)
DEFINE_SPMV_THUNKS(sspmv, float, FB_OP_SSPMV)
DEFINE_SPMV_THUNKS(dspmv, double, FB_OP_DSPMV)
DEFINE_SPR_THUNKS(sspr, float, FB_OP_SSPR)
DEFINE_SPR_THUNKS(dspr, double, FB_OP_DSPR)
DEFINE_SPR2_THUNKS(sspr2, float, FB_OP_SSPR2)
DEFINE_SPR2_THUNKS(dspr2, double, FB_OP_DSPR2)
DEFINE_SYMV_THUNKS(ssymv, float,  FB_OP_SSYMV)
DEFINE_SYMV_THUNKS(dsymv, double, FB_OP_DSYMV)
DEFINE_CSYMV_THUNKS(csymv, FB_OP_CSYMV)
DEFINE_CSYMV_THUNKS(zsymv, FB_OP_ZSYMV)
DEFINE_TRMV_THUNKS(strmv, float,  FB_OP_STRMV)
DEFINE_TRMV_THUNKS(dtrmv, double, FB_OP_DTRMV)
DEFINE_TRMV_THUNKS(ctrmv, fb_complex_float_t, FB_OP_CTRMV)
DEFINE_TRMV_THUNKS(ztrmv, fb_complex_double_t, FB_OP_ZTRMV)
DEFINE_TRMV_THUNKS(strsv, float,  FB_OP_STRSV)
DEFINE_TRMV_THUNKS(dtrsv, double, FB_OP_DTRSV)
DEFINE_TRMV_THUNKS(ctrsv, fb_complex_float_t, FB_OP_CTRSV)
DEFINE_TRMV_THUNKS(ztrsv, fb_complex_double_t, FB_OP_ZTRSV)

/* ============================================================================
 * Thunk instantiation — BLAS Level 3 completion (complex + L3 variants)
 * ========================================================================== */

DEFINE_CGEMM_THUNKS(cgemm, FB_OP_CGEMM)
DEFINE_CGEMM_THUNKS(zgemm, FB_OP_ZGEMM)
DEFINE_SYMM_THUNKS(ssymm,   float,  FB_OP_SSYMM)
DEFINE_SYMM_THUNKS(dsymm,   double, FB_OP_DSYMM)
DEFINE_CSYMM_THUNKS(csymm, FB_OP_CSYMM)
DEFINE_CSYMM_THUNKS(zsymm, FB_OP_ZSYMM)
DEFINE_CSYMM_THUNKS(chemm, FB_OP_CHEMM)
DEFINE_CSYMM_THUNKS(zhemm, FB_OP_ZHEMM)
DEFINE_SYRK_THUNKS(ssyrk,   float,  FB_OP_SSYRK)
DEFINE_SYRK_THUNKS(dsyrk,   double, FB_OP_DSYRK)
DEFINE_CSYRK_THUNKS(csyrk, FB_OP_CSYRK)
DEFINE_CSYRK_THUNKS(zsyrk, FB_OP_ZSYRK)
DEFINE_HERK_THUNKS(cherk, float, fb_complex_float_t, FB_OP_CHERK)
DEFINE_HERK_THUNKS(zherk, double, fb_complex_double_t, FB_OP_ZHERK)
DEFINE_SYR2K_THUNKS(ssyr2k, float,  FB_OP_SSYR2K)
DEFINE_SYR2K_THUNKS(dsyr2k, double, FB_OP_DSYR2K)
DEFINE_CSYR2K_THUNKS(csyr2k, FB_OP_CSYR2K)
DEFINE_CSYR2K_THUNKS(zsyr2k, FB_OP_ZSYR2K)
DEFINE_HER2K_THUNKS(cher2k, fb_complex_float_t, float, FB_OP_CHER2K)
DEFINE_HER2K_THUNKS(zher2k, fb_complex_double_t, double, FB_OP_ZHER2K)
DEFINE_TRMM_THUNKS(strmm,   float,  FB_OP_STRMM)
DEFINE_TRMM_THUNKS(dtrmm,   double, FB_OP_DTRMM)
DEFINE_CTRMM_THUNKS(ctrmm, FB_OP_CTRMM)
DEFINE_CTRMM_THUNKS(ztrmm, FB_OP_ZTRMM)
DEFINE_TRMM_THUNKS(strsm,   float,  FB_OP_STRSM)
DEFINE_TRMM_THUNKS(dtrsm,   double, FB_OP_DTRSM)
DEFINE_CTRMM_THUNKS(ctrsm, FB_OP_CTRSM)
DEFINE_CTRMM_THUNKS(ztrsm, FB_OP_ZTRSM)

/* ============================================================================
 * BLAS Level 3 — SGEMM / DGEMM
 *
 * Full signature (CBLAS order_/transA/transB enums are int-compatible):
 *   sgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
 * Fortran convention wraps all scalar + integer args as pointers.
 * ========================================================================== */

static void thunk_sgemm_cblas_to_fortran(
    int Order, int TransA, int TransB,
    int M, int N, int K,
    float alpha, const float *A, int lda,
                 const float *B, int ldb,
    float beta, float *C, int ldc)
{
    if (!g_fortran_fn[FB_OP_SGEMM]) return;
    int Order_  = Order,  TransA_ = TransA, TransB_ = TransB;
    int M_ = M, N_ = N, K_ = K;
    float alpha_ = alpha, beta_ = beta;
    int lda_ = lda, ldb_ = ldb, ldc_ = ldc;
    ((void(*)(const int*, const int*, const int*,
              const int*, const int*, const int*,
              const float*, const float*, const int*,
                            const float*, const int*,
              const float*, float*, const int*))
     g_fortran_fn[FB_OP_SGEMM])(
         &Order_, &TransA_, &TransB_,
         &M_, &N_, &K_,
         &alpha_, A, &lda_, B, &ldb_, &beta_, C, &ldc_);
}

static void thunk_sgemm_fortran_to_cblas(
    const int *Order, const int *TransA, const int *TransB,
    const int *M,     const int *N,      const int *K,
    const float *alpha, const float *A, const int *lda,
                        const float *B, const int *ldb,
    const float *beta, float *C, const int *ldc)
{
    if (!g_cblas_fn[FB_OP_SGEMM]) return;
    ((void(*)(int, int, int, int, int, int,
              float, const float*, int, const float*, int,
              float, float*, int))
     g_cblas_fn[FB_OP_SGEMM])(
         *Order, *TransA, *TransB,
         *M, *N, *K,
         *alpha, A, *lda, B, *ldb, *beta, C, *ldc);
}

static void thunk_dgemm_cblas_to_fortran(
    int Order, int TransA, int TransB,
    int M, int N, int K,
    double alpha, const double *A, int lda,
                  const double *B, int ldb,
    double beta, double *C, int ldc)
{
    if (!g_fortran_fn[FB_OP_DGEMM]) return;
    int Order_  = Order,  TransA_ = TransA, TransB_ = TransB;
    int M_ = M, N_ = N, K_ = K;
    double alpha_ = alpha, beta_ = beta;
    int lda_ = lda, ldb_ = ldb, ldc_ = ldc;
    ((void(*)(const int*, const int*, const int*,
              const int*, const int*, const int*,
              const double*, const double*, const int*,
                             const double*, const int*,
              const double*, double*, const int*))
     g_fortran_fn[FB_OP_DGEMM])(
         &Order_, &TransA_, &TransB_,
         &M_, &N_, &K_,
         &alpha_, A, &lda_, B, &ldb_, &beta_, C, &ldc_);
}

static void thunk_dgemm_fortran_to_cblas(
    const int *Order, const int *TransA, const int *TransB,
    const int *M,     const int *N,      const int *K,
    const double *alpha, const double *A, const int *lda,
                         const double *B, const int *ldb,
    const double *beta, double *C, const int *ldc)
{
    if (!g_cblas_fn[FB_OP_DGEMM]) return;
    ((void(*)(int, int, int, int, int, int,
              double, const double*, int, const double*, int,
              double, double*, int))
     g_cblas_fn[FB_OP_DGEMM])(
         *Order, *TransA, *TransB,
         *M, *N, *K,
         *alpha, A, *lda, B, *ldb, *beta, C, *ldc);
}

/* ============================================================================
 * LAPACK-operation thunks
 *
 * Each entry covers one LAPACK driver or computational routine that the
 * reference DLL exports with Fortran ABI (*_ suffix, all-pointer) but
 * callers may request via CBLAS-style ABI (scalars by value, layout first).
 *
 * Convention mapping:
 *   Fortran:  all arguments are pointers, including chars and ints.
 *   CBLAS:    'layout' (int) is prepended; char and int scalars are by value.
 *             Array/output arguments remain pointers in both conventions.
 *
 * Complex arrays are typed 'void *' in both thunk signatures; the calling
 * code casts and only the raw pointer value is forwarded, so the type tag
 * is irrelevant here.
 * ========================================================================== */

/* ---- SSTEV / DSTEV  (FB_OP_SSTEV=1119, FB_OP_DSTEV=1120) --------------- */

static void thunk_sstev_fortran_to_cblas(char *jobz, int *n, float *d, float *e,
                                         float *z, int *ldz, float *work,
                                         int *info) {
  if (!g_cblas_fn[FB_OP_SSTEV]) {
    if (info)
      *info = -1;
    return;
  }
  /* CBLAS: (int layout, char jobz, int n, float *d, float *e,
   *         float *z, int ldz, float *work, int *info) */
  ((void (*)(int, char, int, float *, float *, float *, int, float *,
             int *))g_cblas_fn[FB_OP_SSTEV])(102 /*LAPACK_COL_MAJOR*/, *jobz,
                                             *n, d, e, z, *ldz, work, info);
}

static void thunk_sstev_cblas_to_fortran(int layout, char jobz, int n, float *d,
                                         float *e, float *z, int ldz,
