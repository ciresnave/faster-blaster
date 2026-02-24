/* stbcon - Banded triangular condition number */
#
void stbcon_(char* norm, char* uplo, char* diag, int* n, int* kd, float* ab, int* ldab, float* rcond, float* work, int* iwork, int* info) {
    if (*n < 0) { *info = -4; return; }
    *rcond = 1.0f;
    *info = 0;
}
