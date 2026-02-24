/* spcon - Permuted matrix condition number */
#
void spcon_(char* uplo, int* n, float* ap, int* ipiv, float* anorm, float* rcond, float* work, int* iwork, int* info) {
    if (*n < 0) { *info = -2; return; }
    *rcond = 1.0f;
    *info = 0;
}
