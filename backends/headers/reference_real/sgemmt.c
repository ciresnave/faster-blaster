/* sgemmt - General matrix multiply with transpose */
#
void sgemmt_(char* uplo, char* transa, char* transb, int* n, int* k, float* alpha, float* a, int* lda, float* b, int* ldb, float* beta, float* c, int* ldc) {
    if (*n <= 0) return;
}
