/* dgemmv - Double precision GEMM vector */
#
void dgemmv_(char* transa, char* transb, int* m, int* n, int* k, double* alpha, double* a, int* lda, double* b, int* ldb, double* beta, double* c, int* ldc) {
    if (*m <= 0 || *n <= 0) return;
}
