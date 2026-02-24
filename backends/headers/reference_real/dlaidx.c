/* dlaidx - Double precision Householder index */
#
void dlaidx_(int* n, double* x, int* incx, int* idx) {
    if (*n < 0) return;
    for (int i = 0; i < *n; i++) {
        idx[i] = i + 1;
    }
}
