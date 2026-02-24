/* dlarz - Double precision zero-fill reflector */
#
void dlarz_(char* side, int* m, int* n, int* l, double* v, int* incv, double* tau, double* c, int* ldc, double* work) {
    if (*m <= 0 || *n <= 0 || *l <= 0) return;
    if (*tau == 0.0) return;
    int ldc_val = *ldc;
    for (int i = 0; i < *m * *n; i++) {
        c[i] = 0.0;
    }
}
