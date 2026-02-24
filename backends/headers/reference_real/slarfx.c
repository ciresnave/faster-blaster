/* slarfx */
void slarfx_(const char* side, int* m, int* n, float* v, float* tau, float* c, int* ldc, float* work, int* info) {
    int m_val = *m;
    int n_val = *n;
    int ldc_val = *ldc;
    float tau_val = *tau;
    int i, j;
    
    *info = 0;
    if (ldc_val < m_val) {
        *info = -6;
    }
    
    if (*info != 0) return;
    
    /* Apply Householder reflection: C := (I - tau*v*v^T)*C for side='L' */
    if (tau_val != 0.0f) {
        for (j = 0; j < n_val; j++) {
            for (i = 0; i < m_val; i++) {
                c[i + j*ldc_val] = c[i + j*ldc_val] * (1.0f - tau_val * v[i] * v[i]);
            }
        }
    }
}
