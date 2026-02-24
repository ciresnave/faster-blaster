/* dgbcon - estimate condition number of banded matrix (double precision) */
void dgbcon_(const char *norm, const int *n, const int *kl, const int *ku, const double *AB,
             const int *ldab, const int *ipiv, const double *anorm, double *rcond, double *work, 
             int *iwork, int *info)
{
    int nn = *n;
    int kl_val = *kl;
    int ku_val = *ku;
    int ldab_val = *ldab;
    char norm_val = *norm;
    
    *info = 0;
    *rcond = 0.0;
    
    if (nn <= 0) {
        return;
    }
    
    // Simplified condition number estimate for banded matrix
    // Uses the stored LU factorization from DGBTRF
    
    double rcondest = 0.0;
    
    // Rough estimate: use diagonal elements as proxy for condition
    double mindiag = 1e10;
    for (int i = 0; i < nn; i++) {
        int band_idx = kl_val + i;
        if (band_idx < ldab_val) {
            double diag_elem = fabs(AB[band_idx + i * ldab_val]);
            mindiag = (mindiag < diag_elem) ? mindiag : diag_elem;
        }
    }
    
    if (mindiag > 0.0 && *anorm > 0.0) {
        rcondest = mindiag / *anorm;
        if (rcondest > 1.0) {
            rcondest = 1.0;
        }
    }
    
    *rcond = rcondest;
}
