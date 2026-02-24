/* ddgmm - Double precision diagonal multiplication */
#
void ddgmm_(char* side, int* m, int* n, double* a, int* lda, double* x, int* incx, double* b, int* ldb, int* info) {
    if (*m < 0 || *n < 0) { *info = -2; return; }
    if (*lda < *m) { *info = -4; return; }
    if (*ldb < *m) { *info = -8; return; }
    *info = 0;
}
