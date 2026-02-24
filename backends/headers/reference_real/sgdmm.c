/* sgdmm - General diagonal matrix multiplication */
#
void sgdmm_(int* m, int* n, float* d, float* a, int* lda, float* result, int* ldresult, int* info) {
    int m_val = *m, n_val = *n, lda_val = *lda, ldresult_val = *ldresult;
    if (*m < 0) { *info = -1; return; }
    if (*n < 0) { *info = -2; return; }
    if (*lda < m_val) { *info = -5; return; }
    if (*ldresult < m_val) { *info = -7; return; }
    for (int j = 0; j < n_val; j++) {
        for (int i = 0; i < m_val; i++) {
            result[i + j*ldresult_val] = d[i] * a[i + j*lda_val];
        }
    }
    *info = 0;
}
