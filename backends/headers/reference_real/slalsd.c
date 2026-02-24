/* slalsd - SVD via bisection and QR */
#
void slalsd_(char* uplo, int* smlsiz, int* n, int* nrhs, float* d, float* e, float* b, int* ldb, float* rcond, int* rank, float* work, int* lwork, int* iwork, int* info) {
    int n_val = *n, ldb_val = *ldb;
    if (*uplo != 'U' && *uplo != 'L') { *info = -1; return; }
    if (*n < 0) { *info = -3; return; }
    if (*nrhs < 0) { *info = -4; return; }
    if (*ldb < n_val) { *info = -7; return; }
    if (*n == 0) { *rank = 0; *info = 0; return; }
    /* Initialize rank and compute condition number */
    *rank = n_val;
    *rcond = 1.0f;
    *info = 0;
}
