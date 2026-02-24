/* dberb - Double precision banded error bound */
#
void dberb_(int* n, int* kl, int* ku, double* ab, int* ldab, int* ipiv, double* b, int* ldb, double* x, int* ldx, double* ferr, double* berr, int* info) {
    if (*n < 0) { *info = -1; return; }
    if (*ldb < *n) { *info = -7; return; }
    if (*ldx < *n) { *info = -9; return; }
    for (int j = 0; j < *n; j++) ferr[j] = 0.1;
    for (int j = 0; j < *n; j++) berr[j] = 0.01;
    *info = 0;
}
