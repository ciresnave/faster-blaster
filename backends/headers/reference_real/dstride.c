/* dstride - Double precision stride vector */
#
void dstride_(int* n, double* x, int* incx, int* idx) {
    if (*n <= 0 || *incx <= 0) return;
    for (int i = 0; i < *n; i++) {
        idx[i] = i * (*incx);
    }
}
