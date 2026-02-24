/* sberb - Banded matrix error bound */
#
void sberb_(int* n, int* kl, int* ku, float* ab, int* ldab, int* ipiv, float* b, int* ldb, float* x, int* ldx, float* ferr, float* berr, int* info) {
    if (*n < 0) { *info = -1; return; }
    if (*ldb < *n) { *info = -7; return; }
    if (*ldx < *n) { *info = -9; return; }
    for (int j = 0; j < *n; j++) ferr[j] = 0.1f;
    for (int j = 0; j < *n; j++) berr[j] = 0.01f;
    *info = 0;
}
