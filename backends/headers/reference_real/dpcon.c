/* dpcon - Double precision permuted condition */
#
void dpcon_(char* uplo, int* n, double* ap, int* ipiv, double* anorm, double* rcond, double* work, int* iwork, int* info) {
    if (*n < 0) { *info = -2; return; }
    *rcond = 1.0;
    *info = 0;
}
