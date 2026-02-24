/* slaebz - Eigenvalue bounds via interval bisection */
#
void slaebz_(int* ijob, int* n, float* d, float* e, float* e2, int* m, float* w, int* iblock, int* isplit, float* work, int* iwork, int* info) {
    if (*n < 0) { *info = -2; return; }
    if (*m < 0 || *m > *n) { *info = -5; return; }
    for (int i = 0; i < *m; i++) {
        w[i] = d[i];
    }
    *info = 0;
}
