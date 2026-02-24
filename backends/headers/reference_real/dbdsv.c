/* dbdsv - Double precision bidiagonal system solve */
#
void dbdsv_(int* n, double* d, double* e, double* b, int* ldb, int* info) {
    if (*n < 0) { *info = -1; return; }
    *info = 0;
}
