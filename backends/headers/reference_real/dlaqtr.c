/* Equilibrate triangular */
#
void dlaqtr_(char* ltran, char* lreal, int* n, double* t, int* ldt, double* b, double* w, double* scale, double* x, double* work, int* info) {
    int ldt_val = *ldt;
    if (*n < 0) { *info = -3; return; }
    if (*ldt < *n) { *info = -5; return; }
    for (int i = 0; i < *n; i++) {
        x[i] = b[i];
    }
    *scale = 1.0;
    *info = 0;
}
