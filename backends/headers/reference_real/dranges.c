/* dranges - Double precision range statistics */
#
void dranges_(int* n, double* x, int* incx, double* minval, double* maxval, double* range) {
    if (*n <= 0) return;
    *minval = x[0];
    *maxval = x[0];
    for (int i = 1; i < *n; i++) {
        double val = x[i * (*incx)];
        if (val < *minval) *minval = val;
        if (val > *maxval) *maxval = val;
    }
    *range = *maxval - *minval;
}
