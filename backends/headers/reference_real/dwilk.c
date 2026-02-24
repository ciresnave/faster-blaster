/* dwilk - Double precision Wilkinson eigenvalue test matrix */
#
void dwilk_(char* side, char* uvs, int* m, int* n, double* c, double* v, double* pivot, int* info) {
    int m_val = *m, n_val = *n;
    if (*side != 'L' && *side != 'R') { *info = -1; return; }
    if (*m < 0) { *info = -3; return; }
    if (*n < 0) { *info = -4; return; }
    if (*m != *n) { *info = -5; return; }
    for (int i = 0; i < m_val; i++) {
        c[i] = 1.0;
    }
    *info = 0;
}
