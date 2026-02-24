/* ssyev2 - Direct 2x2 symmetric eigenvalue solver */

#

/* SSYEV2 - Compute eigenvalues and eigenvectors of 2x2 symmetric matrix
   A: Upper triangle of 2x2 matrix [[a, b], [b, c]]
   B: B parameter (off-diagonal)
   C: C parameter
   W1, W2: Output eigenvalues
   CS1, SN1: Cosine and sine of rotation
   INFO: Status
 */
void ssyev2_(const float *a, const float *b, const float *c,
             float *w1, float *w2, float *cs1, float *sn1, int *info)
{
    *info = 0;
    float a_val = *a;
    float b_val = *b;
    float c_val = *c;

    /* For 2x2 symmetric matrix, eigenvalues are roots of:
       det([[a-lambda, b], [b, c-lambda]]) = 0
       (a-lambda)(c-lambda) - b^2 = 0
       lambda^2 - (a+c)lambda + ac - b^2 = 0
    */

    float trace = a_val + c_val;
    float det = a_val * c_val - b_val * b_val;

    /* Discriminant */
    float disc = trace * trace / 4.0f - det;
    if (disc < 0.0f) disc = 0.0f;

    float sqrt_disc = sqrtf(disc);
    *w1 = trace / 2.0f + sqrt_disc;
    *w2 = trace / 2.0f - sqrt_disc;

    /* Ensure w1 >= w2 */
    if (*w1 < *w2) {
        float tmp = *w1;
        *w1 = *w2;
        *w2 = tmp;
    }

    /* Compute eigenvector for larger eigenvalue */
    /* [a - w1, b; b, c - w1] * v1 = 0 */
    float diag_diff = a_val - *w1;

    if (fabsf(b_val) > 1e-10f) {
        /* Use (b, c - w1) as eigenvector */
        float norm = sqrtf(b_val * b_val + (c_val - *w1) * (c_val - *w1));
        if (norm > 0.0f) {
            *cs1 = b_val / norm;
            *sn1 = (c_val - *w1) / norm;
        } else {
            *cs1 = 1.0f;
            *sn1 = 0.0f;
        }
    } else if (fabsf(diag_diff) > 1e-10f) {
        /* Use (a - w1, b) as eigenvector */
        float norm = sqrtf(diag_diff * diag_diff + b_val * b_val);
        if (norm > 0.0f) {
            *cs1 = diag_diff / norm;
            *sn1 = b_val / norm;
        } else {
            *cs1 = 1.0f;
            *sn1 = 0.0f;
        }
    } else {
        *cs1 = 1.0f;
        *sn1 = 0.0f;
    }
}
