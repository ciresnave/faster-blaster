/* dhpmv - Double precision Hermitian packed MV */
#
void dhpmv_(char* uplo, int* n, double* alpha, double* ap, double* x, int* incx, double* beta, double* y, int* incy) {
    if (*n <= 0) return;
}
