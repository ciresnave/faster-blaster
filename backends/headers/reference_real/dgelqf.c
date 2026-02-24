#

void dgelqf_(const int *m, const int *n, double *A, const int *lda,
             double *tau, double *work, const int *lwork, int *info)
{
    int i, j, k, minmn;
    double alpha, norm, sigma, inv_sigma, beta;
    
    *info = 0;
    if (*m <= 0 || *n <= 0) return;
    
    minmn = (*m < *n) ? *m : *n;
    
    /* LQ factorization: Householder reflectors stored in rows */
    for (k = 0; k < minmn; k++) {
        /* Householder vector in row k, starting at column k */
        alpha = A[k + k * *lda];
        
        /* Compute norm of A[k, k+1:n] */
        norm = 0.0;
        for (j = k + 1; j < *n; j++) {
            norm += A[k + j * *lda] * A[k + j * *lda];
        }
        norm = sqrt(norm);
        
        /* Compute Householder tau */
        if (norm == 0.0 && alpha >= 0.0) {
            tau[k] = 0.0;
            continue;
        }
        
        sigma = (alpha < 0.0) ? sqrt(alpha * alpha + norm * norm)
                              : -sqrt(alpha * alpha + norm * norm);
        
        tau[k] = (sigma - alpha) / sigma;
        inv_sigma = 1.0 / (alpha - sigma);
        
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
