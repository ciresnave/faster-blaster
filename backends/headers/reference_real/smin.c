/* smin - Find minimum element with index */
#
void smin_(int* n, float* x, int* incx, int* idx) {
    if (*n <= 0) { *idx = 0; return; }
    float minval = fabs(x[0]);
    *idx = 1;
    for (int i = 1; i < *n; i++) {
        float absval = fabs(x[i * (*incx)]);
        if (absval < minval) { minval = absval; *idx = i + 1; }
    }
}
