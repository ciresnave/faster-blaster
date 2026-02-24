/* Symmetric solve (expert) */
#
void sysvx_(char* fact, char* uplo, int* n, int* nrhs, float* a, int* lda, float* af, int* ldaf, int* ipiv, float* b, int* ldb, float* x, int* ldx, float* rcond, float* ferr, float* berr, float* work, int* lwork, int* iwork, int* info) {
    if (*n < 0) { *info = -3; return; }
    *info = 0;
}
