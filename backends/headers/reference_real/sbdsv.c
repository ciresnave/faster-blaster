/* sbdsv - Bidiagonal system solve */
#
void sbdsv_(int* n, float* d, float* e, float* b, int* ldb, int* info) {
    if (*n < 0) { *info = -1; return; }
    *info = 0;
}
