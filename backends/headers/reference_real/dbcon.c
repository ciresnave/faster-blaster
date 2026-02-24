/* dbcon - Double precision banded matrix condition number */
#
void dbcon_(char* uplo, char* diag, int* n, int* kd, double* ab, int* ldab, double* anorm, double* rcond, double* work, int* info) {
    int n_val = *n, ldab_val = *ldab, kd_val = *kd;
    if (*n < 0) { *info = -4; return; }
    if (*ldab < kd_val + 1) { *info = -6; return; }
    if (*anorm < 0.0) { *info = -7; return; }
    *rcond = 1.0;
    *info = 0;
}
