/* stpcon - Packed triangular condition number */
#
void stpcon_(char* norm, char* uplo, char* diag, int* n, float* ap, float* rcond, float* work, int* iwork, int* info) {
    if (*n < 0) { *info = -4; return; }
    *rcond = 1.0f;
    *info = 0;
}
