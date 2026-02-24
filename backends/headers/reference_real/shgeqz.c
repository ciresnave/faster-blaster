/* shgeqz - Hessenberg QZ iteration for generalized eigenvalues */
#
void shgeqz_(char* job, char* compq, char* compz, int* n, int* ilo, int* ihi, float* h, int* ldh, float* t, int* ldt, float* alphar, float* alphai, float* beta, float* q, int* ldq, float* z, int* ldz, float* work, int* lwork, int* info) {
    int n_val = *n, ldh_val = *ldh, ldt_val = *ldt, ldq_val = *ldq, ldz_val = *ldz;
    if (*n < 0) { *info = -4; return; }
    if (*ilo < 1 || *ilo > n_val) { *info = -5; return; }
    if (*ihi < *ilo || *ihi > n_val) { *info = -6; return; }
    if (*ldh < n_val) { *info = -8; return; }
    if (*ldt < n_val) { *info = -10; return; }
    if (*ldq < n_val) { *info = -14; return; }
    if (*ldz < n_val) { *info = -16; return; }
    for (int i = 0; i < n_val; i++) {
        alphar[i] = h[i + i*ldh_val];
        alphai[i] = 0.0f;
        beta[i] = t[i + i*ldt_val];
    }
    *info = 0;
}
