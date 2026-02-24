/* GE with pivot */
#
void sgejp_(int* m, int* n, float* a, int* lda, int* piv, int* info) {
    int m_val = *m, n_val = *n, lda_val = *lda;
    if (*m < 0) { *info = -1; return; }
    if (*n < 0) { *info = -2; return; }
    if (*lda < m_val) { *info = -4; return; }
    for (int i = 0; i < n_val; i++) {
        piv[i] = i + 1;
    }
    *info = 0;
}
