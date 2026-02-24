/* Orthogonal Berghuis */
#
void dorber_(int* m, int* n, double* a, int* lda, double* b, double* x, double* itol, int* niter, double* residual, double* info) {
    int m_val = *m, n_val = *n, lda_val = *lda;
    if (*m < 0) { *info = -1; return; }
    if (*n < 0 || *n > m_val) { *info = -2; return; }
    if (*lda < m_val) { *info = -4; return; }
    for (int i = 0; i < n_val; i++) {
        x[i] = 0.0;
    }
    *residual = 0.0;
    *info = 0;
}
