#

void sgelqf_(const int *m, const int *n, float *A, const int *lda,
             float *tau, float *work, const int *lwork, int *info)
{
    int i, j, k, minmn;
    float alpha, norm, sigma, inv_sigma, beta;
    
    *info = 0;
    if (*m <= 0 || *n <= 0) return;
    
    minmn = (*m < *n) ? *m : *n;
    
    /* LQ factorization: Householder reflectors stored in rows */
    for (k = 0; k < minmn; k++) {
        /* Householder vector in row k, starting at column k */
        alpha = A[k + k * *lda];
        
        /* Compute norm of A[k, k+1:n] */
        norm = 0.0f;
        for (j = k + 1; j < *n; j++) {
            norm += A[k + j * *lda] * A[k + j * *lda];
        }
        norm = sqrtf(norm);
        
        /* Compute Householder tau */
        if (norm == 0.0f && alpha >= 0.0f) {
            tau[k] = 0.0f;
            continue;
        }
        
        sigma = (alpha < 0.0f) ? sqrtf(alpha * alpha + norm * norm)
                               : -sqrtf(alpha * alpha + norm * norm);
        
        tau[k] = (sigma - alpha) / sigma;
        inv_sigma = 1.0f / (alpha - sigma);
        
        /* Normalize Householder vector */
        for (j = k + 1; j < *n; j++) {
            A[k + j * *lda] *= inv_sigma;
        }
        A[k + k * *lda] = sigma;
        
        /* Apply to rows below */
        for (i = k + 1; i < *m; i++) {
            beta = A[i + k * *lda];
            for (j = k + 1; j < *n; j++) {
                beta += A[i + j * *lda] * A[k + j * *lda];
            }
            beta *= tau[k];
            
            A[i + k * *lda] -= beta;
            for (j = k + 1; j < *n; j++) {
                A[i + j * *lda] -= beta * A[k + j * *lda];
            }
        }
    }
}
