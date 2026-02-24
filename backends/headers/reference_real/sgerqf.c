#

void sgerqf_(const int *m, const int *n, float *A, const int *lda,
             float *tau, float *work, const int *lwork, int *info)
{
    int i, j, k, minmn;
    float alpha, norm, sigma, inv_sigma, beta;
    
    *info = 0;
    if (*m <= 0 || *n <= 0) return;
    
    minmn = (*m < *n) ? *m : *n;
    
    /* RQ factorization: process bottom-right corner, Householder in rows */
    for (k = minmn - 1; k >= 0; k--) {
        int row = *m - minmn + k;
        int col = *n - minmn + k;
        
        alpha = A[row + col * *lda];
        
        /* Compute norm of A[row, 0:col] */
        norm = 0.0f;
        for (j = 0; j < col; j++) {
            norm += A[row + j * *lda] * A[row + j * *lda];
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
        
        /* Normalize Householder vector */
        for (j = 0; j < col; j++) {
            A[row + j * *lda] *= inv_sigma;
        }
        A[row + col * *lda] = sigma;
        
        /* Apply to rows above */
        for (i = 0; i < row; i++) {
            beta = A[i + col * *lda];
            for (j = 0; j < col; j++) {
                beta += A[i + j * *lda] * A[row + j * *lda];
            }
            beta *= tau[k];
            
            A[i + col * *lda] -= beta;
            for (j = 0; j < col; j++) {
                A[i + j * *lda] -= beta * A[row + j * *lda];
            }
        }
    }
}
