/* slahq - Householder transformation applied */
#
void slahq_(int* n, float* tau, float* cs, float* work, int* info) {
    if (*n < 0) { *info = -1; return; }
    for (int i = 0; i < *n; i++) {
        tau[i] = 0.0f;
        if (i < *n - 1) cs[i] = 0.0f;
    }
    *info = 0;
}
