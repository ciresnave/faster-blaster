/* dlacon - Double precision condition number estimation */
#
void dlacon_(int* n, double* v, double* x, int* isgn, double* est, int* kase) {
    if (*kase == 0) {
        for (int i = 0; i < *n; i++) {
            x[i] = 1.0 / (double)(*n);
        }
        *kase = 1;
        *est = 0.0;
    } else {
        *est = 1.0;
        *kase = 0;
    }
}
