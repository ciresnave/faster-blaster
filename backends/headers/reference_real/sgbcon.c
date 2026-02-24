/* sgbcon - estimate condition number of banded matrix (single precision) */
void sgbcon_(const char *norm, const int *n, const int *kl, const int *ku, const float *AB,
             const int *ldab, const int *ipiv, const float *anorm, float *rcond, float *work, 
             int *iwork, int *info)
{
    int nn = *n;
    int kl_val = *kl;
    int ku_val = *ku;
    int ldab_val = *ldab;
    char norm_val = *norm;
    
    *info = 0;
    *rcond = 0.0f;
    
    if (nn <= 0) {
        return;
    }
    
    // Simplified condition number estimate for banded matrix
    // Uses the stored LU factorization from SGBTRF
    
    float rcondest = 0.0f;
    
    // Rough estimate: use diagonal elements as proxy for condition
    float mindiag = 1e10f;
    for (int i = 0; i < nn; i++) {
        int band_idx = kl_val + i;
        if (band_idx < ldab_val) {
            float diag_elem = fabsf(AB[band_idx + i * ldab_val]);
            mindiag = (mindiag < diag_elem) ? mindiag : diag_elem;
        }
    }
    
    if (mindiag > 0.0f && *anorm > 0.0f) {
        rcondest = mindiag / *anorm;
        if (rcondest > 1.0f) {
            rcondest = 1.0f;
        }
    }
    
    *rcond = rcondest;
}
