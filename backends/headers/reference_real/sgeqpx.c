/* sgeqpx */
void sgeqpx_(int* m, int* n, float* a, int* lda, int* jpvt, float* tau, float* work, int* lwork, int* info) {
    int m_val = *m;
    int n_val = *n;
    int lda_val = *lda;
    int lwork_val = *lwork;
    int i, j;
    
    *info = 0;
    if (m_val < 0) {
        *info = -1;
    } else if (n_val < 0) {
        *info = -2;
    } else if (lda_val < m_val) {
        *info = -4;
    }
    
    if (*info != 0) return;
    
    /* Initialize pivot array */
    for (j = 0; j < n_val; j++) {
        if (jpvt[j] == 0) {
            jpvt[j] = j + 1;
        }
    }
    
    /* Initialize tau */
    for (i = 0; i < n_val; i++) {
        tau[i] = 0.0f;
    }
}
