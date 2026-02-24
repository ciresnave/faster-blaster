/* Householder */
#
void dlahd_(int* m, int* n, double* d, double* scale, double* norm, int* info) {
    if (*m < 0) { *info = -1; return; }
    if (*n < 0) { *info = -2; return; }
    for (int i = 0; i < *m; i++) {
        d[i] = 1.0;
    }
    *scale = 1.0;
    *norm = 0.0;
    *info = 0;
}
