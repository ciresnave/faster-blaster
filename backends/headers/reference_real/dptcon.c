/* dptcon - Double precision packed triangular condition */
#
void dptcon_(char* uplo, int* n, double* ap, double* diag, double* anorm, double* rcond, double* work, int* info) {
    if (*n < 0) { *info = -2; return; }
    *rcond = 1.0;
    *info = 0;
}
