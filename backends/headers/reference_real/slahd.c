/* Householder */
#
void slahd_(int* m, int* n, float* d, float* scale, float* norm, int* info) {
    if (*m < 0) { *info = -1; return; }
    if (*n < 0) { *info = -2; return; }
    for (int i = 0; i < *m; i++) {
        d[i] = 1.0f;
    }
    *scale = 1.0f;
    *norm = 0.0f;
    *info = 0;
}
