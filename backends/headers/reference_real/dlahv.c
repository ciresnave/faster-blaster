/* dlahv - Double precision Householder vector */
#
void dlahv_(int* n, double* x, double* tau, double* c, double* s) {
    if (*n <= 0) return;
    double xnorm = 0.0;
    for (int i = 1; i < *n; i++) {
        xnorm += x[i] * x[i];
    }
    xnorm = sqrt(xnorm);
    if (xnorm <= 0.0) {
        *tau = 0.0;
        *c = 1.0;
        *s = 0.0;
    } else {
        double alpha = x[0];
        double r = sqrt(alpha*alpha + xnorm*xnorm);
        *c = alpha / r;
        *s = xnorm / r;
        *tau = (*s) / (1.0 + *c);
    }
}
