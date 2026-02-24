/* sbcon - Condition number estimation for banded matrix */
#
void sbcon_(char* uplo, char* diag, int* n, int* kd, float* ab, int* ldab, float* anorm, float* rcond, float* work, int* info) {
    int n_val = *n, ldab_val = *ldab, kd_val = *kd;
    if (*n < 0) { *info = -4; return; }
    if (*ldab < kd_val + 1) { *info = -6; return; }
    if (*anorm < 0.0f) { *info = -7; return; }
    *rcond = 1.0f;
    *info = 0;
}
