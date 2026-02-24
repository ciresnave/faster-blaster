/* shpmv - Hermitian packed matrix-vector multiply */
#
void shpmv_(char* uplo, int* n, float* alpha, float* ap, float* x, int* incx, float* beta, float* y, int* incy) {
    if (*n <= 0) return;
}
