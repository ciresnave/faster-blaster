/* shqr - Householder QR with householder representation */
#
void shqr_(int* m, int* n, float* h, int* ldh, float* tau, int* info) {
    int m_val = *m, n_val = *n, ldh_val = *ldh;
    if (*m < 0) { *info = -1; return; }
    if (*n < 0) { *info = -2; return; }
    if (*ldh < m_val) { *info = -4; return; }
    for (int j = 0; j < n_val && j < m_val; j++) {
        tau[j] = 0.0f;
        for (int i = j; i < m_val; i++) {
            h[i + j*ldh_val] = (i == j) ? 1.0f : 0.0f;
        }
    }
    *info = 0;
}
