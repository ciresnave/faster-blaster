/* dlaswp - apply row interchanges (double precision) */
void dlaswp_(const int *n, double *A, const int *lda, const int *k1, const int *k2,
             const int *ipiv, const int *incx)
{
    int nn = *n;
    int lda_val = *lda;
    int k1_val = *k1;
    int k2_val = *k2;
    int incx_val = *incx;
    
    if (incx_val > 0) {
        // Forward pass
        for (int k = k1_val - 1; k < k2_val; k++) {
            int kp = ipiv[k] - 1;  // Convert from 1-based to 0-based
            if (kp != k) {
                // Swap row k and row kp
                for (int j = 0; j < nn; j++) {
                    double temp = A[k + j * lda_val];
                    A[k + j * lda_val] = A[kp + j * lda_val];
                    A[kp + j * lda_val] = temp;
                }
            }
        }
    } else if (incx_val < 0) {
        // Backward pass
        for (int k = k2_val - 1; k >= k1_val - 1; k--) {
            int kp = ipiv[k] - 1;  // Convert from 1-based to 0-based
            if (kp != k) {
                // Swap row k and row kp
                for (int j = 0; j < nn; j++) {
                    double temp = A[k + j * lda_val];
                    A[k + j * lda_val] = A[kp + j * lda_val];
                    A[kp + j * lda_val] = temp;
                }
            }
        }
    }
}
