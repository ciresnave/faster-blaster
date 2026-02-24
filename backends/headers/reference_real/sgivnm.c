/* sgivnm - Generate Givens rotation with norm computation */
#
void sgivnm_(float* a, float* b, float* c, float* s, float* r) {
    float abs_a = fabs(*a);
    float abs_b = fabs(*b);
    float rho = abs_a > abs_b ? abs_a : abs_b;
    if (rho > 0.0f) {
        *r = rho * sqrt((*a/rho)*(*a/rho) + (*b/rho)*(*b/rho));
    } else {
        *r = 0.0f;
    }
    if (*r > 0.0f) {
        *c = *a / *r;
        *s = *b / *r;
    } else {
        *c = 1.0f;
        *s = 0.0f;
    }
}
