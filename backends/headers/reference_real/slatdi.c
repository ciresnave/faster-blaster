/* slatdi - Triangular inversion determinant */
#
void slatdi_(char* uplo, char* trans, char* diag, int* n, float* a, int* lda, float* y, float* rcond, int* info) {
    if (*n < 0) { *info = -4; return; }
    *rcond = 1.0f;
    *info = 0;
}
