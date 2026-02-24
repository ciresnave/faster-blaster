/* dtbcon - Double precision banded triangular condition */
#
void dtbcon_(char* norm, char* uplo, char* diag, int* n, int* kd, double* ab, int* ldab, double* rcond, double* work, int* iwork, int* info) {
    if (*n < 0) { *info = -4; return; }
    *rcond = 1.0;
    *info = 0;
}
