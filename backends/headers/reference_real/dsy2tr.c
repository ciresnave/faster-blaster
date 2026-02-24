/* dsy2tr - Double precision symmetric to triangular storage */
#
void dsy2tr_(char* uplo, int* n, double* a, int* lda, double* b, int* ldb, int* info) {
    int n_val = *n, lda_val = *lda, ldb_val = *ldb;
    if (*uplo != 'U' && *uplo != 'L') { *info = -1; return; }
    if (*n < 0) { *info = -2; return; }
    if (*lda < n_val) { *info = -4; return; }
    if (*ldb < n_val) { *info = -6; return; }
    if (*uplo == 'U') {
        for (int j = 0; j < n_val; j++) {
            for (int i = 0; i <= j; i++) {
                b[i + j*ldb_val] = a[i + j*lda_val];
            }
        }
    } else {
        for (int j = 0; j < n_val; j++) {
            for (int i = j; i < n_val; i++) {
                b[i + j*ldb_val] = a[i + j*lda_val];
            }
        }
    }
    *info = 0;
}
