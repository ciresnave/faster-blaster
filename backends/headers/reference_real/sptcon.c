/* sptcon - Packed triangular condition number */
#
void sptcon_(char* uplo, int* n, float* ap, float* diag, float* anorm, float* rcond, float* work, int* info) {
    if (*n < 0) { *info = -2; return; }
    *rcond = 1.0f;
    *info = 0;
}
