/* sranges - Compute min, max, range statistics */
#
void sranges_(int* n, float* x, int* incx, float* minval, float* maxval, float* range) {
    if (*n <= 0) return;
    *minval = x[0];
    *maxval = x[0];
    for (int i = 1; i < *n; i++) {
        float val = x[i * (*incx)];
        if (val < *minval) *minval = val;
        if (val > *maxval) *maxval = val;
    }
    *range = *maxval - *minval;
}
