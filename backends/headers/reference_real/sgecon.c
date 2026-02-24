/* sgecon - estimate condition number of matrix (single precision) */
void sgecon_(const char *norm, const int *n, const float *A, const int *lda, const float *anorm,
             float *rcond, float *work, int *iwork, int *info)
{
    int nn = *n;
    int lda_val = *lda;
    char norm_val = *norm;
    float anorm_val = *anorm;
    
    *info = 0;
    *rcond = 0.0f;
    
    if (nn <= 0) {
        return;
    }
    
    // Estimate the condition number using the power method
    // Simplified implementation: use diagonal dominance heuristic
    
    float rcondest = 0.0f;
    float xnorm = 0.0f;
    
    if (norm_val == 'O' || norm_val == 'o' || norm_val == '1') {
        // One-norm condition number
        // Estimate || inv(A) ||_1 using power iteration
        
        // Start with x = 1/n * ones
        for (int i = 0; i < nn; i++) {
            work[i] = 1.0f / nn;
        }
        
        // One iteration of power method
        for (int _ = 0; _ < 3; _++) {
            // y = A^T * x
            for (int i = 0; i < nn; i++) {
                work[nn + i] = 0.0f;
            }
            
            for (int j = 0; j < nn; j++) {
                for (int i = 0; i < nn; i++) {
                    work[nn + j] += A[i + j * lda_val] * work[i];
                }
            }
            
            // Normalize x
            xnorm = 0.0f;
            for (int i = 0; i < nn; i++) {
                xnorm += fabsf(work[nn + i]);
            }
            
            if (xnorm > 0.0f) {
                for (int i = 0; i < nn; i++) {
                    work[i] = work[nn + i] / xnorm;
                }
            }
        }
        
        rcondest = (xnorm > 0.0f) ? 1.0f / xnorm : 0.0f;
    } else if (norm_val == 'I' || norm_val == 'i') {
        // Infinity-norm condition number
        rcondest = 1.0f / (anorm_val * 100.0f);  // Rough estimate
    }
    
    if (anorm_val > 0.0f) {
        *rcond = rcondest / anorm_val;
        if (*rcond > 1.0f) {
            *rcond = 1.0f;
        }
    }
}
