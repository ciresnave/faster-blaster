/* smax - Find maximum element with index */
#
void smax_(int* n, float* x, int* incx, int* idx) {
    if (*n <= 0) { *idx = 0; return; }
    float maxval = fabs(x[0]);
    *idx = 1;
    for (int i = 1; i < *n; i++) {
        float absval = fabs(x[i * (*incx)]);
        if (absval > maxval) { maxval = absval; *idx = i + 1; }
    }
}
