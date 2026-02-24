/* dlakn - Double precision Kn parameter */
#
void dlakn_(int* n, double* d, double* e, double* e2, double* kn) {
    if (*n < 1) { *kn = 0.0; return; }
    double maxa = 0.0;
    for (int i = 0; i < *n; i++) {
        if (fabs(d[i]) > maxa) maxa = fabs(d[i]);
    }
    for (int i = 0; i < *n - 1; i++) {
        if (fabs(e[i]) > maxa) maxa = fabs(e[i]);
    }
    *kn = maxa > 0.0 ? maxa : 1.0;
}
