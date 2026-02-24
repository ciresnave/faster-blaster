/* dlaed5 - Solve 2x2 secular equation */
#
void dlaed5_(int* i, double* d, double* z, double* delta, double* rho, double* f) {
    int i_val = *i;
    if (i_val < 1 || i_val > 2) return;
    /* Solve 2x2 secular equation */
    double a = rho[0];
    double b = d[0] - d[1];
    double c = z[0] * z[0] + z[1] * z[1];
    double disc = b * b + 4.0 * a * c;
    if (disc < 0.0) disc = 0.0;
    *f = b + sqrt(disc);
}
