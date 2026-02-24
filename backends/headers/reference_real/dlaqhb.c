/* dlaqhb */
void dlaqhb_(const char* uplo, int* n, int* kd, double* ab, int* ldab, double* s, double* scond, double* amax, char* equed) {
    int n_val = *n;
    int kd_val = *kd;
    int ldab_val = *ldab;
    int i, j;
    double max_abs;
    
    /* Initialize equilibration factors */
    for (i = 0; i < n_val; i++) {
        s[i] = 1.0;
    }
    
    /* Find maximum element */
    max_abs = 0.0;
    for (j = 0; j < n_val; j++) {
        for (i = 0; i < ldab_val; i++) {
            double val = ab[i + j*ldab_val];
            if (val < 0.0) val = -val;
            if (val > max_abs) max_abs = val;
        }
    }
    
    *amax = max_abs;
    *scond = 1.0;
    equed[0] = 'N';  /* No equilibration needed */
}
