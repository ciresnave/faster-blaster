/* Reciprocal condition */
#
void dlarc_(char* type, int* n, double* a, double* b, double* c, double* s) {
    double r = sqrt((*a)*(*a) + (*b)*(*b));
    if (r > 0.0) {
        *c = (*a) / r;
        *s = (*b) / r;
    } else {
        *c = 1.0;
        *s = 0.0;
    }
}
