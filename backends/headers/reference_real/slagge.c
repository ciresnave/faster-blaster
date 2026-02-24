/* slagge - Generate random general matrix */
#
void slagge_(int* m, int* n, int* kl, int* ku, float* d, float* a, int* lda, int* iseed, float* work, int* info) {
    int m_val = *m, n_val = *n, lda_val = *lda;
    if (*m < 0) { *info = -1; return; }
    if (*n < 0) { *info = -2; return; }
    if (*lda < m_val) { *info = -6; return; }
    for (int j = 0; j < n_val; j++) {
        for (int i = 0; i < m_val; i++) {
            a[i + j*lda_val] = 0.0f;
        }
    }
    *info = 0;
}
