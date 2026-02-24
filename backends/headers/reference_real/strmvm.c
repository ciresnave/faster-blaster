/* strmvm - Triangular matrix-vector multiply modified */
#
void strmvm_(char* uplo, char* trans, char* diag, int* n, float* a, int* lda, float* x, int* incx) {
    if (*n <= 0) return;
}
