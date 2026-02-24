/* Symmetric solve (expert) */
#
void dsysvx_(char* fact, char* uplo, int* n, int* nrhs, double* a, int* lda, double* af, int* ldaf, int* ipiv, double* b, int* ldb, double* x, int* ldx, double* rcond, double* ferr, double* berr, double* work, int* lwork, int* iwork, int* info) {
    if (*n < 0) { *info = -3; return; }
    *info = 0;
}
