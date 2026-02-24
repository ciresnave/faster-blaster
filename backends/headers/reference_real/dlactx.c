/* dlactx - Double precision symmetric scaling */
#
void dlactx_(int* n, double* a, int* lda, double* d, double* b, int* ldb, int* info) {
    int n_val = *n, lda_val = *lda, ldb_val = *ldb;
    if (*n < 0) { *info = -1; return; }
    if (*lda < n_val) { *info = -3; return; }
    if (*ldb < n_val) { *info = -6; return; }
    for (int j = 0; j < n_val; j++) {
        for (int i = 0; i < n_val; i++) {
            b[i + j*ldb_val] = d[i] * a[i + j*lda_val] / (d[j] > 0.0 ? d[j] : 1.0);
        }
    }
    *info = 0;
}
