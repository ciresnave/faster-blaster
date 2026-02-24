/* dtpcon - Double precision packed triangular condition */
#
void dtpcon_(char* norm, char* uplo, char* diag, int* n, double* ap, double* rcond, double* work, int* iwork, int* info) {
    if (*n < 0) { *info = -4; return; }
    *rcond = 1.0;
    *info = 0;
}
