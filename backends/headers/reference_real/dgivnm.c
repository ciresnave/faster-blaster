/* dgivnm - Double precision Givens rotation with norm */
#
void dgivnm_(double* a, double* b, double* c, double* s, double* r) {
    double abs_a = fabs(*a);
    double abs_b = fabs(*b);
    double rho = abs_a > abs_b ? abs_a : abs_b;
    if (rho > 0.0) {
        *r = rho * sqrt((*a/rho)*(*a/rho) + (*b/rho)*(*b/rho));
    } else {
        *r = 0.0;
    }
    if (*r > 0.0) {
        *c = *a / *r;
        *s = *b / *r;
    } else {
        *c = 1.0;
        *s = 0.0;
    }
}
