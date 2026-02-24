/* ssbmv - Symmetric banded matrix-vector multiply */
#
void ssbmv_(char* uplo, int* n, int* k, float* alpha, float* a, int* lda, float* x, int* incx, float* beta, float* y, int* incy) {
    if (*n <= 0) return;
}
