/* sgemmv - General matrix multiply with vector */
#
void sgemmv_(char* transa, char* transb, int* m, int* n, int* k, float* alpha, float* a, int* lda, float* b, int* ldb, float* beta, float* c, int* ldc) {
    if (*m <= 0 || *n <= 0) return;
}
