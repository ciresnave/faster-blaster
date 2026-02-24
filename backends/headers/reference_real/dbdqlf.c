/* dbdqlf - Double precision bidiagonal QL factorization */
#
void dbdqlf_(int* n, double* d, double* e, double* q, int* ldq, double* work, int* info) {
    if (*n < 0) { *info = -1; return; }
    *info = 0;
}
