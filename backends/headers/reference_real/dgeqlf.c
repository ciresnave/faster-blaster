#

void dgeqlf_(const int *m, const int *n, double *A, const int *lda,
             double *tau, double *work, const int *lwork, int *info)
{
    int i, j, k, minmn;
    double alpha, norm, sigma, inv_sigma, beta;
    
    *info = 0;
    if (*m <= 0 || *n <= 0) return;
    
    minmn = (*m < *n) ? *m : *n;
    
    /* QL factorization: process bottom-right corner */
    for (k = minmn - 1; k >= 0; k--) {
        int row = *m - minmn + k;
        int col = *n - minmn + k;
        
        alpha = A[row + col * *lda];
        
        /* Compute norm of A[0:row, col] */
        norm = 0.0;
        for (i = 0; i < row; i++) {
            norm += A[i + col * *lda] * A[i + col * *lda];
        }
        norm = sqrt(norm);
        
        if (norm == 0.0 && alpha >= 0.0) {
            tau[k] = 0.0;
            continue;
        }
        
        sigma = (alpha < 0.0) ? sqrt(alpha * alpha + norm * norm)
                              : -sqrt(alpha * alpha + norm * norm);
        
        tau[k] = (sigma - alpha) / sigma;
        inv_sigma = 1.0 / (alpha - sigma);
        
        /* Scale Householder vector */
        for (i = 0; i < row; i++) {
            A[i + col * *lda] *= inv_sigma;
        }
        A[row + col * *lda] = sigma;
        
        /* Apply to columns to the left */
        for (j = 0; j < col; j++) {
            beta = A[row + j * *lda];
            for (i = 0; i < row; i++) {
                beta += A[i + col * *lda] * A[i + j * *lda];
            }
            beta *= tau[k];
            
            A[row + j * *lda] -= beta;
            for (i = 0; i < row; i++) {
                A[i + j * *lda] -= beta * A[i + col * *lda];
            }
        }
    }
}
