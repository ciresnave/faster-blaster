/* slaidx - Index array for Householder */
#
void slaidx_(int* n, float* x, int* incx, int* idx) {
    if (*n < 0) return;
    for (int i = 0; i < *n; i++) {
        idx[i] = i + 1;
    }
}
