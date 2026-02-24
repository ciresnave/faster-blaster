/* sgycon - General matrix condition number */
#
void sgycon_(char* norm, int* n, float* a, int* lda, int* ipiv, float* anorm, float* rcond, float* work, int* iwork, int* info) {
    if (*n < 0) { *info = -2; return; }
    if (*lda < *n) { *info = -4; return; }
    if (*anorm < 0.0f) { *info = -6; return; }
    if (*anorm == 0.0f) {
        *rcond = 0.0f;
    } else {
        *rcond = 1.0f / (anorm[0] + 1.0f);
    }
    *info = 0;
}
