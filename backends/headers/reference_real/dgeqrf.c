#

void dgeqrf_(const int *m, const int *n, double *A, const int *lda,
             double *tau, double *work, const int *lwork, int *info)
{
    int i, j, k, minmn;
    double alpha, norm, sigma, inv_sigma, beta;
    
    *info = 0;
    if (*m <= 0 || *n <= 0) return;
    
    minmn = (*m < *n) ? *m : *n;
    
    for (k = 0; k < minmn; k++) {
        /* Compute Householder reflection for column k */
        alpha = A[k + k * *lda];
        
        /* Compute norm of A[k+1:m, k] */
        norm = 0.0;
        for (i = k + 1; i < *m; i++) {
            norm += A[i + k * *lda] * A[i + k * *lda];
        }
        norm = sqrt(norm);
        
        /* Compute Householder vector and tau */
        if (norm == 0.0 && alpha >= 0.0) {
            tau[k] = 0.0;
            continue;
        }
        
        sigma = (alpha < 0.0) ? sqrt(alpha * alpha + norm * norm)
                              : -sqrt(alpha * alpha + norm * norm);
        
        tau[k] = (sigma - alpha) / sigma;
        inv_sigma = 1.0 / (alpha - sigma);
        
        /* Scale Householder vector */
        for (i = k + 1; i < *m; i++) {
            A[i + k * *lda] *= inv_sigma;
        }
        A[k + k * *lda] = sigma;
        
        /* Apply Householder transformation to trailing submatrix */
        for (j = k + 1; j < *n; j++) {
            beta = A[k + j * *lda];
            for (i = k + 1; i < *m; i++) {
                beta += A[i + k * *lda] * A[i + j * *lda];
            }
            beta *= tau[k];
            
            A[k + j * *lda] -= beta;
            for (i = k + 1; i < *m; i++) {
                A[i + j * *lda] -= beta * A[i + k * *lda];
            }
        }
    }
}
