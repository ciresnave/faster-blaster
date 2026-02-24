/* sbdsdc - Bidiagonal SVD divide-and-conquer */
#
void sbdsdc_(char* uplo, char* compq, int* n, float* d, float* e, float* u, int* ldu, float* vt, int* ldvt, float* q, int* iq, float* work, int* iwork, int* info) {
    if (*n < 0) { *info = -3; return; }
    *info = 0;
}
