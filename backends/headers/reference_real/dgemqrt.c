/* Multiply QR tall-skin QR */
#
void dgemqrt_(char* side, char* trans, int* m, int* n, int* k, int* nb, double* v, int* ldv, double* t, int* ldt, double* c, int* ldc, double* work, int* info) {
    int m_val = *m, n_val = *n, k_val = *k, nb_val = *nb;
    int ldv_val = *ldv, ldt_val = *ldt, ldc_val = *ldc;
    if (*side != 'L' && *side != 'R') { *info = -1; return; }
    if (*trans != 'N' && *trans != 'T') { *info = -2; return; }
    if (*m < 0) { *info = -3; return; }
    if (*n < 0) { *info = -4; return; }
    if (*ldc < m_val) { *info = -11; return; }
    *info = 0;
}
