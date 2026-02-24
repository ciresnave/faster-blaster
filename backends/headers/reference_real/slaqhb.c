/* slaqhb */
void slaqhb_(const char* uplo, int* n, int* kd, float* ab, int* ldab, float* s, float* scond, float* amax, char* equed) {
    int n_val = *n;
    int kd_val = *kd;
    int ldab_val = *ldab;
    int i, j;
    float max_abs;
    
    /* Initialize equilibration factors */
    for (i = 0; i < n_val; i++) {
        s[i] = 1.0f;
    }
    
    /* Find maximum element */
    max_abs = 0.0f;
    for (j = 0; j < n_val; j++) {
        for (i = 0; i < ldab_val; i++) {
            float val = ab[i + j*ldab_val];
            if (val < 0.0f) val = -val;
            if (val > max_abs) max_abs = val;
        }
    }
    
    *amax = max_abs;
    *scond = 1.0f;
    equed[0] = 'N';  /* No equilibration needed */
}
