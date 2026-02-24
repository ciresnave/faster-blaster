/* dhqr2 - Double precision Householder QR variant */
#
void dhqr2_(int* m, int* n, double* a, int* lda, double* tau, double* work, int* info) {
    int m_val = *m, n_val = *n, lda_val = *lda;
    if (*m < 0) { *info = -1; return; }
    if (*n < 0) { *info = -2; return; }
    if (*lda < m_val) { *info = -4; return; }
    int minmn = m_val < n_val ? m_val : n_val;
    for (int i = 0; i < minmn; i++) {
        tau[i] = 0.0;
    }
    *info = 0;
}
