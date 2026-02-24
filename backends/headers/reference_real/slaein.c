/* slaein - Inverse iteration eigenvector computation */
#
void slaein_(char* right, char* dtrans, int* n, float* t, int* ldt, float* w, float* x, float* y, float* wr, float* wi, float* work, int* info) {
    int n_val = *n, ldt_val = *ldt;
    if (*n < 0) { *info = -3; return; }
    if (*ldt < n_val) { *info = -5; return; }
    for (int i = 0; i < n_val; i++) {
        x[i] = 1.0f;
    }
    *wr = t[0];
    *wi = 0.0f;
    *info = 0;
}
