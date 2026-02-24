/* sstev2 - Direct 2x2 tridiagonal eigenvalue solver */

#

/* SSTEV2 - Compute eigenvalues and eigenvector of 2x2 tridiagonal matrix
   For tridiagonal matrix T = [[a, b], [b, c]]
   Returns eigenvalues and eigenvector components
 */
void sstev2_(const float *a, const float *b, const float *c,
             float *w1, float *w2, float *cs1, float *sn1, int *info)
{
    *info = 0;
    float a_val = *a;
    float b_val = *b;
    float c_val = *c;

    /* Eigenvalues of tridiagonal [a, b; b, c] */
    float trace = a_val + c_val;
    float det = a_val * c_val - b_val * b_val;

    float disc = trace * trace / 4.0f - det;
    if (disc < 0.0f) disc = 0.0f;

    float sqrt_disc = sqrtf(disc);
    *w1 = trace / 2.0f + sqrt_disc;
    *w2 = trace / 2.0f - sqrt_disc;

    if (*w1 < *w2) {
        float tmp = *w1;
        *w1 = *w2;
        *w2 = tmp;
    }

    /* Eigenvector for w1 */
    float denom = a_val - *w1;
    if (fabsf(denom) > 1e-10f) {
        float norm = sqrtf(b_val * b_val + denom * denom);
        if (norm > 0.0f) {
            *cs1 = denom / norm;
            *sn1 = b_val / norm;
        } else {
            *cs1 = 1.0f;
            *sn1 = 0.0f;
        }
    } else if (fabsf(b_val) > 1e-10f) {
        float norm = sqrtf(b_val * b_val + (c_val - *w1) * (c_val - *w1));
        if (norm > 0.0f) {
            *cs1 = b_val / norm;
            *sn1 = (c_val - *w1) / norm;
        } else {
            *cs1 = 1.0f;
            *sn1 = 0.0f;
        }
    } else {
        *cs1 = 1.0f;
        *sn1 = 0.0f;
    }
}
