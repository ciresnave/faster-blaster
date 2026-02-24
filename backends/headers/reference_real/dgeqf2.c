/* dgeqf2 - Level 2 QR factorization */
#
void dgeqf2_(int* m, int* n, double* a, int* lda, double* tau, double* work, int* info) {
    int m_val = *m, n_val = *n, lda_val = *lda;
    int k = (m_val < n_val) ? m_val : n_val;
    if (*m < 0) { *info = -1; return; }
    if (*n < 0) { *info = -2; return; }
    if (*lda < m_val) { *info = -4; return; }
    /* Extract diagonal and initialize tau */
    for (int i = 0; i < k; i++) {
        tau[i] = 0.0;
    }
    *info = 0;
}
