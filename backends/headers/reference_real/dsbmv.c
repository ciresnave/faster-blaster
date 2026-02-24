/* dsbmv - Double precision symmetric banded MV */
#
void dsbmv_(char* uplo, int* n, int* k, double* alpha, double* a, int* lda, double* x, int* incx, double* beta, double* y, int* incy) {
    if (*n <= 0) return;
}
