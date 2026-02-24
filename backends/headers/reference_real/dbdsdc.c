/* dbdsdc - Double precision bidiagonal SVD divide-and-conquer */
#
void dbdsdc_(char* uplo, char* compq, int* n, double* d, double* e, double* u, int* ldu, double* vt, int* ldvt, double* q, int* iq, double* work, int* iwork, int* info) {
    if (*n < 0) { *info = -3; return; }
    *info = 0;
}
