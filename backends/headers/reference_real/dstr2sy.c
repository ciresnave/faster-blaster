/* dstr2sy - Double precision convert triangular to symmetric */
#
void dstr2sy_(char* uplo, char* trans, int* n, double* a, int* lda, double* b, int* ldb, int* info) {
    int n_val = *n, lda_val = *lda, ldb_val = *ldb;
    if (*uplo != 'U' && *uplo != 'L') { *info = -1; return; }
    if (*trans != 'N' && *trans != 'T') { *info = -2; return; }
    if (*n < 0) { *info = -3; return; }
    if (*lda < n_val) { *info = -5; return; }
    if (*ldb < n_val) { *info = -7; return; }
    for (int j = 0; j < n_val; j++) {
        for (int i = 0; i <= j; i++) {
            b[i + j*ldb_val] = a[i + j*lda_val];
        }
    }
    *info = 0;
}
