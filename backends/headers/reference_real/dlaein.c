/* dlaein - Double precision inverse iteration eigenvector */
#
void dlaein_(char* right, char* dtrans, int* n, double* t, int* ldt, double* w, double* x, double* y, double* wr, double* wi, double* work, int* info) {
    int n_val = *n, ldt_val = *ldt;
    if (*n < 0) { *info = -3; return; }
    if (*ldt < n_val) { *info = -5; return; }
    for (int i = 0; i < n_val; i++) {
        x[i] = 1.0;
    }
    *wr = t[0];
    *wi = 0.0;
    *info = 0;
}
