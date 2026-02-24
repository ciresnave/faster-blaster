/* strcon - estimate condition number of triangular matrix (single precision) */
void strcon_(const char *norm, const char *uplo, const char *diag, const int *n, const float *A,
             const int *lda, float *rcond, float *work, int *iwork, int *info)
{
    int nn = *n;
    int lda_val = *lda;
    char norm_val = *norm;
    char uplo_val = *uplo;
    char diag_val = *diag;
    
    *info = 0;
    *rcond = 0.0f;
    
    if (nn <= 0) {
        return;
    }
    
    // For triangular matrices, condition number is easier to estimate
    // || T ||_norm * || T^-1 ||_norm
    
    float anorm = 0.0f;
    
    if (norm_val == 'O' || norm_val == 'o' || norm_val == '1') {
        // One-norm: max column sum
        for (int j = 0; j < nn; j++) {
            float colsum = 0.0f;
            
            if (uplo_val == 'U' || uplo_val == 'u') {
                for (int i = 0; i <= j && i < nn; i++) {
                    colsum += fabsf(A[i + j * lda_val]);
                }
            } else {
                for (int i = j; i < nn; i++) {
                    colsum += fabsf(A[i + j * lda_val]);
                }
            }
            
            anorm = (anorm > colsum) ? anorm : colsum;
        }
    } else if (norm_val == 'I' || norm_val == 'i') {
        // Infinity-norm: max row sum
        for (int i = 0; i < nn; i++) {
            float rowsum = 0.0f;
            
            if (uplo_val == 'U' || uplo_val == 'u') {
                for (int j = i; j < nn; j++) {
                    rowsum += fabsf(A[i + j * lda_val]);
                }
            } else {
                for (int j = 0; j <= i && j < nn; j++) {
                    rowsum += fabsf(A[i + j * lda_val]);
                }
            }
            
            anorm = (anorm > rowsum) ? anorm : rowsum;
        }
    }
    
    // Estimate || T^-1 ||_norm using power iteration (simplified)
    float ainv = 0.0f;
    
    // Very rough estimate: diagonal dominance heuristic
    if (diag_val == 'N' || diag_val == 'n') {
        // Non-unit diagonal: use diagonal elements
        float mindiag = 1e10f, maxdiag = 0.0f;
        for (int i = 0; i < nn; i++) {
            float diag_elem = fabsf(A[i + i * lda_val]);
            mindiag = (mindiag < diag_elem) ? mindiag : diag_elem;
            maxdiag = (maxdiag > diag_elem) ? maxdiag : diag_elem;
        }
        
        if (mindiag > 0.0f) {
            ainv = 1.0f / mindiag;
        }
    } else {
        // Unit diagonal: norm of inverse is at least 1.0
        ainv = 1.0f;
    }
    
    if (anorm > 0.0f && ainv > 0.0f) {
        *rcond = 1.0f / (anorm * ainv);
        if (*rcond > 1.0f) {
            *rcond = 1.0f;
        }
    }
}
