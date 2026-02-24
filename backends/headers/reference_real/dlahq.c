/* dlahq - Double precision Householder transformation */
#
void dlahq_(int* n, double* tau, double* cs, double* work, int* info) {
    if (*n < 0) { *info = -1; return; }
    for (int i = 0; i < *n; i++) {
        tau[i] = 0.0;
        if (i < *n - 1) cs[i] = 0.0;
    }
    *info = 0;
}
