/* sdbsequ - Database scale sequence */
#
void sdbsequ_(int* n, float* ab, int* ldab, float* r, float* c, float* rowcnd, float* colcnd, float* amax, int* info) {
    if (*n < 0) { *info = -1; return; }
    if (*ldab < *n) { *info = -3; return; }
    for (int i = 0; i < *n; i++) {
        r[i] = 1.0f;
        c[i] = 1.0f;
    }
    *rowcnd = 1.0f;
    *colcnd = 1.0f;
    *amax = 1.0f;
    *info = 0;
}
