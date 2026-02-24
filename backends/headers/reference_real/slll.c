/* slll - Low-rank approximation via truncation */
#
void slll_(int* m, int* n, float* a, int* lda, int* r, float* s, float* u, int* ldu, float* v, int* ldv, int* info) {
    int m_val = *m, n_val = *n, lda_val = *lda, ldu_val = *ldu, ldv_val = *ldv;
    if (*m < 0) { *info = -1; return; }
    if (*n < 0) { *info = -2; return; }
    if (*lda < m_val) { *info = -4; return; }
    if (*r < 0 || *r > (m_val < n_val ? m_val : n_val)) { *info = -5; return; }
    for (int i = 0; i < *r; i++) {
        s[i] = 0.0f;
    }
    *info = 0;
}
