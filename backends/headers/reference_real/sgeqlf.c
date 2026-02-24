#

void sgeqlf_(const int *m, const int *n, float *A, const int *lda,
             float *tau, float *work, const int *lwork, int *info)
{
    int i, j, k, minmn;
    float alpha, norm, sigma, inv_sigma, beta;
    
    *info = 0;
    if (*m <= 0 || *n <= 0) return;
    
    minmn = (*m < *n) ? *m : *n;
    
    /* QL factorization: process bottom-right corner */
    for (k = minmn - 1; k >= 0; k--) {
        int row = *m - minmn + k;
        int col = *n - minmn + k;
        
        alpha = A[row + col * *lda];
        
        /* Compute norm of A[0:row, col] */
        norm = 0.0f;
        for (i = 0; i < row; i++) {
            norm += A[i + col * *lda] * A[i + col * *lda];
        }
        norm = sqrtf(norm);
        
        if (norm == 0.0f && alpha >= 0.0f) {
            tau[k] = 0.0f;
            continue;
        }
        
        sigma = (alpha < 0.0f) ? sqrtf(alpha * alpha + norm * norm)
                               : -sqrtf(alpha * alpha + norm * norm);
        
        tau[k] = (sigma - alpha) / sigma;
        inv_sigma = 1.0f / (alpha - sigma);
        
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
