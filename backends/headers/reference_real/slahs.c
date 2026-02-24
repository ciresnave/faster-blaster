/* slahs - Householder similarity transformation */
#
void slahs_(char* side, char* trans, int* m, int* n, float* h, int* ldh, float* tau, float* c, int* ldc, int* info) {
    int m_val = *m, n_val = *n, ldh_val = *ldh, ldc_val = *ldc;
    if (*m < 0) { *info = -3; return; }
    if (*n < 0) { *info = -4; return; }
    if (*ldh < m_val) { *info = -6; return; }
    if (*ldc < m_val) { *info = -9; return; }
    *info = 0;
}
