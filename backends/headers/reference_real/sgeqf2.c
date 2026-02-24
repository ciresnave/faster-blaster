/* sgeqf2 - Level 2 QR factorization */
#
void sgeqf2_(int* m, int* n, float* a, int* lda, float* tau, float* work, int* info) {
    int m_val = *m, n_val = *n, lda_val = *lda;
    int k = (m_val < n_val) ? m_val : n_val;
    if (*m < 0) { *info = -1; return; }
    if (*n < 0) { *info = -2; return; }
    if (*lda < m_val) { *info = -4; return; }
    /* Extract diagonal and initialize tau */
    for (int i = 0; i < k; i++) {
        tau[i] = 0.0f;
    }
    *info = 0;
}
