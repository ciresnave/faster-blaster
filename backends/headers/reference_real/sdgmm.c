/* sdgmm - Diagonal matrix multiplication */
#
void sdgmm_(char* side, int* m, int* n, float* a, int* lda, float* x, int* incx, float* b, int* ldb, int* info) {
    if (*m < 0 || *n < 0) { *info = -2; return; }
    if (*lda < *m) { *info = -4; return; }
    if (*ldb < *m) { *info = -8; return; }
    *info = 0;
}
