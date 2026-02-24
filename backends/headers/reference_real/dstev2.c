/* dstev2 - Direct 2x2 tridiagonal eigenvalue solver */

#

/* DSTEV2 - Compute eigenvalues and eigenvector of 2x2 tridiagonal matrix (double precision)
   For tridiagonal matrix T = [[a, b], [b, c]]
   Returns eigenvalues and eigenvector components
 */
void dstev2_(const double *a, const double *b, const double *c,
             double *w1, double *w2, double *cs1, double *sn1, int *info)
{
    *info = 0;
    double a_val = *a;
    double b_val = *b;
    double c_val = *c;

    /* Eigenvalues of tridiagonal [a, b; b, c] */
    double trace = a_val + c_val;
    double det = a_val * c_val - b_val * b_val;

    double disc = trace * trace / 4.0 - det;
    if (disc < 0.0) disc = 0.0;

    double sqrt_disc = sqrt(disc);
    *w1 = trace / 2.0 + sqrt_disc;
    *w2 = trace / 2.0 - sqrt_disc;

    if (*w1 < *w2) {
        double tmp = *w1;
        *w1 = *w2;
        *w2 = tmp;
    }

    /* Eigenvector for w1 */
    double denom = a_val - *w1;
    if (fabs(denom) > 1e-15) {
        double norm = sqrt(b_val * b_val + denom * denom);
        if (norm > 0.0) {
            *cs1 = denom / norm;
            *sn1 = b_val / norm;
        } else {
            *cs1 = 1.0;
            *sn1 = 0.0;
        }
    } else if (fabs(b_val) > 1e-15) {
        double norm = sqrt(b_val * b_val + (c_val - *w1) * (c_val - *w1));
        if (norm > 0.0) {
            *cs1 = b_val / norm;
            *sn1 = (c_val - *w1) / norm;
        } else {
            *cs1 = 1.0;
            *sn1 = 0.0;
        }
    } else {
        *cs1 = 1.0;
        *sn1 = 0.0;
    }
}
