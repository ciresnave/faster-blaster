/* dgemmt - Double precision GEMM transpose */
#
void dgemmt_(char* uplo, char* transa, char* transb, int* n, int* k, double* alpha, double* a, int* lda, double* b, int* ldb, double* beta, double* c, int* ldc) {
    if (*n <= 0) return;
}
