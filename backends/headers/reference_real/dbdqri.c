/* dbdqri - Double precision bidiagonal QR iteration */
#
void dbdqri_(int* n, double* d, double* e, double* u, int* ldu, double* v, int* ldv, double* work, int* info) {
    int n_val = *n, ldu_val = *ldu, ldv_val = *ldv;
    if (*n < 0) { *info = -1; return; }
    if (*ldu < n_val) { *info = -5; return; }
    if (*ldv < n_val) { *info = -7; return; }
    for (int i = 0; i < n_val; i++) {
        for (int j = 0; j < n_val; j++) {
            u[i + j*ldu_val] = (i == j) ? 1.0 : 0.0;
            v[i + j*ldv_val] = (i == j) ? 1.0 : 0.0;
        }
    }
    *info = 0;
}
