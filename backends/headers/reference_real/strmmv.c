/* strmmv - Triangular matrix multiply matrix-vector */
#
void strmmv_(char* side, char* uplo, char* transa, char* diag, int* m, int* n, float* alpha, float* a, int* lda, float* b, int* ldb) {
    if (*m <= 0 || *n <= 0) return;
}
