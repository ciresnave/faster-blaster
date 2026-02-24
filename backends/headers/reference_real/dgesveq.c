/* dgesveq */
void dgesveq_(int* m, int* n, double* a, int* lda, double* b, int* ldb, double* x, int* ldx, double* work, int* lwork, int* info) {
    int m_val = *m;
    int n_val = *n;
    int lda_val = *lda;
    int ldb_val = *ldb;
    int ldx_val = *ldx;
    int i, j;
    
    *info = 0;
    if (m_val < 0) {
        *info = -1;
    } else if (n_val < 0) {
        *info = -2;
    } else if (lda_val < m_val) {
        *info = -4;
    } else if (ldb_val < m_val) {
        *info = -6;
    }
    
    if (*info != 0) return;
    
    /* Initialize solution as zero matrix */
    for (j = 0; j < n_val; j++) {
        for (i = 0; i < n_val; i++) {
            x[i + j*ldx_val] = 0.0;
        }
    }
}
