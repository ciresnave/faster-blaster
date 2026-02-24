/* dhemmv - Double precision Hermitian EMM vector */
#
void dhemmv_(char* side, char* uplo, int* m, int* n, double* alpha, double* a, int* lda, double* b, int* ldb, double* beta, double* c, int* ldc) {
    if (*m <= 0 || *n <= 0) return;
}
