/* dgemvm - Double precision GEMV modified */
#
void dgemvm_(char* trans, int* m, int* n, double* alpha, double* a, int* lda, double* x, int* incx, double* beta, double* y, int* incy) {
    if (*m <= 0 || *n <= 0) return;
}
