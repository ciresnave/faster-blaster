/* Equilibrate triangular */
#
void slaqtr_(char* ltran, char* lreal, int* n, float* t, int* ldt, float* b, float* w, float* scale, float* x, float* work, int* info) {
    int ldt_val = *ldt;
    if (*n < 0) { *info = -3; return; }
    if (*ldt < *n) { *info = -5; return; }
    for (int i = 0; i < *n; i++) {
        x[i] = b[i];
    }
    *scale = 1.0f;
    *info = 0;
}
