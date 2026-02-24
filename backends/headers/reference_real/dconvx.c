/* dconvx - Double precision convex combination */
#
void dconvx_(int* m, int* n, double* a, int* lda, double* b, int* ldb, double* alpha, double* result, int* ldresult, int* info) {
    int m_val = *m, n_val = *n, lda_val = *lda, ldb_val = *ldb, ldresult_val = *ldresult;
    if (*m < 0) { *info = -1; return; }
    if (*n < 0) { *info = -2; return; }
    if (*lda < m_val) { *info = -4; return; }
    if (*ldb < m_val) { *info = -6; return; }
    if (*ldresult < m_val) { *info = -9; return; }
    for (int j = 0; j < n_val; j++) {
        for (int i = 0; i < m_val; i++) {
            result[i + j*ldresult_val] = (*alpha)*a[i + j*lda_val] + (1.0 - *alpha)*b[i + j*ldb_val];
        }
    }
    *info = 0;
}
