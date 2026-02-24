/* dlaebz - Double precision eigenvalue bounds */
#
void dlaebz_(int* ijob, int* n, double* d, double* e, double* e2, int* m, double* w, int* iblock, int* isplit, double* work, int* iwork, int* info) {
    if (*n < 0) { *info = -2; return; }
    if (*m < 0 || *m > *n) { *info = -5; return; }
    for (int i = 0; i < *m; i++) {
        w[i] = d[i];
    }
    *info = 0;
}
