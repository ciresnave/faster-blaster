/* sbdsvd - Bidiagonal SVD */
#
void sbdsvd_(int* n, int* nrhs, float* d, float* e, float* u, int* ldu, float* vt, int* ldvt, float* b, int* ldb, float* work, int* info) {
    if (*n < 0) { *info = -1; return; }
    *info = 0;
}
