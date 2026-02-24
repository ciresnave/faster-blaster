/* dlage - Form matrix A from L and U factors */
#
void dlage_(int* m, int* n, double* l, int* ldl, double* u, int* ldu, double* a, int* lda, int* info) {
    int m_val = *m, n_val = *n, ldl_val = *ldl, ldu_val = *ldu, lda_val = *lda;
    if (*m < 0) { *info = -1; return; }
    if (*n < 0) { *info = -2; return; }
    if (*ldl < m_val) { *info = -4; return; }
    if (*ldu < m_val) { *info = -6; return; }
    if (*lda < m_val) { *info = -8; return; }
    /* Form A = L * U */
    for (int j = 0; j < n_val; j++) {
        for (int i = 0; i < m_val; i++) {
            a[i + j*lda_val] = 0.0;
            for (int k = 0; k < m_val && k < n_val; k++) {
                a[i + j*lda_val] += l[i + k*ldl_val] * u[k + j*ldu_val];
            }
        }
    }
    *info = 0;
}
