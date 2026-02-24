/* sgemvt - GEMV explicit transpose */
#
void sgemvt_(char* trans, int* m, int* n, float* alpha, float* a, int* lda, float* x, int* incx, float* beta, float* y, int* incy) {
    if (*m <= 0 || *n <= 0) return;
}
