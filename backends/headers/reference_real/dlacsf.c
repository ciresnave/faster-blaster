/* dlacsf - Double precision scaling factors */
#
void dlacsf_(int* n, double* d, double* s, int* info) {
    if (*n < 0) { *info = -1; return; }
    for (int i = 0; i < *n; i++) {
        double scale = fabs(d[i]);
        s[i] = (scale > 0.0) ? (1.0 / sqrt(scale)) : 1.0;
    }
    *info = 0;
}
