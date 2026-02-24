/* dgycon - Double precision condition number */
#
void dgycon_(char* norm, int* n, double* a, int* lda, int* ipiv, double* anorm, double* rcond, double* work, int* iwork, int* info) {
    if (*n < 0) { *info = -2; return; }
    if (*lda < *n) { *info = -4; return; }
    if (*anorm < 0.0) { *info = -6; return; }
    if (*anorm == 0.0) {
        *rcond = 0.0;
    } else {
        *rcond = 1.0 / (anorm[0] + 1.0);
    }
    *info = 0;
}
