/* slakn - Compute Kn for eigenvalue bisection */
#
void slakn_(int* n, float* d, float* e, float* e2, float* kn) {
    if (*n < 1) { *kn = 0.0f; return; }
    float maxa = 0.0f;
    for (int i = 0; i < *n; i++) {
        if (fabs(d[i]) > maxa) maxa = fabs(d[i]);
    }
    for (int i = 0; i < *n - 1; i++) {
        if (fabs(e[i]) > maxa) maxa = fabs(e[i]);
    }
    *kn = maxa > 0.0f ? maxa : 1.0f;
}
