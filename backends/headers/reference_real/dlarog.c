/* dlarog - Double precision Givens generator */
#
void dlarog_(double* x, double* y, double* c, double* s) {
    double r = sqrt((*x)*(*x) + (*y)*(*y));
    if (r == 0.0) {
        *c = 1.0;
        *s = 0.0;
    } else {
        *c = (*x) / r;
        *s = (*y) / r;
    }
}
