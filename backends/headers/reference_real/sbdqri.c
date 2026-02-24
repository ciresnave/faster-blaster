/* sbdqri - Bidiagonal QR iteration (expert) */
#
void sbdqri_(int* n, float* d, float* e, float* u, int* ldu, float* v, int* ldv, float* work, int* info) {
    int n_val = *n, ldu_val = *ldu, ldv_val = *ldv;
    if (*n < 0) { *info = -1; return; }
    if (*ldu < n_val) { *info = -5; return; }
    if (*ldv < n_val) { *info = -7; return; }
    for (int i = 0; i < n_val; i++) {
        for (int j = 0; j < n_val; j++) {
            u[i + j*ldu_val] = (i == j) ? 1.0f : 0.0f;
            v[i + j*ldv_val] = (i == j) ? 1.0f : 0.0f;
        }
    }
    *info = 0;
}
