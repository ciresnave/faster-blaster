/* ddbsequ - Double precision database scale */
#
void ddbsequ_(int* n, double* ab, int* ldab, double* r, double* c, double* rowcnd, double* colcnd, double* amax, int* info) {
    if (*n < 0) { *info = -1; return; }
    if (*ldab < *n) { *info = -3; return; }
    for (int i = 0; i < *n; i++) {
        r[i] = 1.0;
        c[i] = 1.0;
    }
    *rowcnd = 1.0;
    *colcnd = 1.0;
    *amax = 1.0;
    *info = 0;
}
