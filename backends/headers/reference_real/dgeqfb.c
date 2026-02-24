/* dgeqfb - Block QR factorization */
#
void dgeqfb_(int* m, int* n, double* a, int* lda, double* t, int* ldt, double* work, int* lwork, int* info) {
    int m_val = *m, n_val = *n, lda_val = *lda, ldt_val = *ldt;
    int k = (m_val < n_val) ? m_val : n_val;
    if (*m < 0) { *info = -1; return; }
    if (*n < 0) { *info = -2; return; }
    if (*lda < m_val) { *info = -4; return; }
    if (*ldt < k) { *info = -6; return; }
    *info = 0;
}
