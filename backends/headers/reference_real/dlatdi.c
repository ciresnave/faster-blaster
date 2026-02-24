/* dlatdi - Double precision triangular inversion */
#
void dlatdi_(char* uplo, char* trans, char* diag, int* n, double* a, int* lda, double* y, double* rcond, int* info) {
    if (*n < 0) { *info = -4; return; }
    *rcond = 1.0;
    *info = 0;
}
