/* sbdqlf - Bidiagonal QL factorization */
#
void sbdqlf_(int* n, float* d, float* e, float* q, int* ldq, float* work, int* info) {
    if (*n < 0) { *info = -1; return; }
    *info = 0;
}
