/* dbdsvd - Double precision bidiagonal SVD */
#
void dbdsvd_(int* n, int* nrhs, double* d, double* e, double* u, int* ldu, double* vt, int* ldvt, double* b, int* ldb, double* work, int* info) {
    if (*n < 0) { *info = -1; return; }
    *info = 0;
}
