/* dsyev2 - Direct 2x2 symmetric eigenvalue solver */

#

/* DSYEV2 - Compute eigenvalues and eigenvectors of 2x2 symmetric matrix (double precision)
   A: Upper triangle of 2x2 matrix [[a, b], [b, c]]
   B: B parameter (off-diagonal)
   C: C parameter
   W1, W2: Output eigenvalues
   CS1, SN1: Cosine and sine of rotation
   INFO: Status
 */
void dsyev2_(const double *a, const double *b, const double *c,
             double *w1, double *w2, double *cs1, double *sn1, int *info)
{
    *info = 0;
    double a_val = *a;
    double b_val = *b;
    double c_val = *c;

    /* For 2x2 symmetric matrix, eigenvalues are roots of:
       det([[a-lambda, b], [b, c-lambda]]) = 0
       (a-lambda)(c-lambda) - b^2 = 0
       lambda^2 - (a+c)lambda + ac - b^2 = 0
    */

    double trace = a_val + c_val;
    double det = a_val * c_val - b_val * b_val;

    /* Discriminant */
    double disc = trace * trace / 4.0 - det;
    if (disc < 0.0) disc = 0.0;

    double sqrt_disc = sqrt(disc);
    *w1 = trace / 2.0 + sqrt_disc;
    *w2 = trace / 2.0 - sqrt_disc;

    /* Ensure w1 >= w2 */
    if (*w1 < *w2) {
        double tmp = *w1;
        *w1 = *w2;
        *w2 = tmp;
    }

    /* Compute eigenvector for larger eigenvalue */
    /* [a - w1, b; b, c - w1] * v1 = 0 */
    double diag_diff = a_val - *w1;

    if (fabs(b_val) > 1e-15) {
        /* Use (b, c - w1) as eigenvector */
        double norm = sqrt(b_val * b_val + (c_val - *w1) * (c_val - *w1));
        if (norm > 0.0) {
            *cs1 = b_val / norm;
            *sn1 = (c_val - *w1) / norm;
        } else {
            *cs1 = 1.0;
            *sn1 = 0.0;
        }
    } else if (fabs(diag_diff) > 1e-15) {
        /* Use (a - w1, b) as eigenvector */
        double norm = sqrt(diag_diff * diag_diff + b_val * b_val);
        if (norm > 0.0) {
            *cs1 = diag_diff / norm;
            *sn1 = b_val / norm;
        } else {
            *cs1 = 1.0;
            *sn1 = 0.0;
        }
    } else {
        *cs1 = 1.0;
        *sn1 = 0.0;
    }
}
