/* slarz - Zero-fill elementary reflector */
#
void slarz_(char* side, int* m, int* n, int* l, float* v, int* incv, float* tau, float* c, int* ldc, float* work) {
    if (*m <= 0 || *n <= 0 || *l <= 0) return;
    if (*tau == 0.0f) return;
    int ldc_val = *ldc;
    for (int i = 0; i < *m * *n; i++) {
        c[i] = 0.0f;
    }
}
