/* dgbtrs - solve banded system using LU factorization (double precision) */
void dgbtrs_(const char *trans, const int *n, const int *kl, const int *ku, const int *nrhs,
             const double *AB, const int *ldab, const int *ipiv, double *B, const int *ldb, int *info)
{
    int nn = *n;
    int kl_val = *kl;
    int ku_val = *ku;
    int nrhs_val = *nrhs;
    int ldab_val = *ldab;
    int ldb_val = *ldb;
    char trans_val = *trans;
    
    *info = 0;
    
    // Apply row interchanges from LU factorization
    for (int i = 0; i < nn; i++) {
        int kp = ipiv[i] - 1;  // Convert to 0-based
        if (kp != i) {
            for (int j = 0; j < nrhs_val; j++) {
                double temp = B[i + j * ldb_val];
                B[i + j * ldb_val] = B[kp + j * ldb_val];
                B[kp + j * ldb_val] = temp;
            }
        }
    }
    
    if (trans_val == 'N' || trans_val == 'n') {
        // Solve L*y = b: forward substitution
        for (int i = 0; i < nn; i++) {
            int i_start = (i - kl_val >= 0) ? i - kl_val : 0;
            
            for (int j = 0; j < nrhs_val; j++) {
                for (int k = i_start; k < i; k++) {
                    int band_idx = kl_val + k - i;
                    if (band_idx >= 0) {
                        B[i + j * ldb_val] -= AB[band_idx + i * ldab_val] * B[k + j * ldb_val];
                    }
                }
            }
        }
        
        // Solve U*x = y: backward substitution
        for (int i = nn - 1; i >= 0; i--) {
            int i_end = (i + ku_val < nn) ? i + ku_val : nn - 1;
            double diag = AB[kl_val + i * ldab_val];
            
            for (int j = 0; j < nrhs_val; j++) {
                for (int k = i + 1; k <= i_end; k++) {
                    int band_idx = kl_val + i - k;
                    if (band_idx >= 0) {
                        B[i + j * ldb_val] -= AB[band_idx + k * ldab_val] * B[k + j * ldb_val];
                    }
                }
                B[i + j * ldb_val] /= diag;
            }
        }
    } else {
        // Solve U^T*y = b: forward substitution
        for (int i = 0; i < nn; i++) {
            int i_end = (i + ku_val < nn) ? i + ku_val : nn - 1;
            double diag = AB[kl_val + i * ldab_val];
            
            for (int j = 0; j < nrhs_val; j++) {
                B[i + j * ldb_val] /= diag;
                
                for (int k = i + 1; k <= i_end; k++) {
                    int band_idx = kl_val + i - k;
                    if (band_idx >= 0) {
                        B[k + j * ldb_val] -= AB[band_idx + k * ldab_val] * B[i + j * ldb_val];
                    }
                }
            }
        }
        
        // Solve L^T*x = y: backward substitution
        for (int i = nn - 1; i >= 0; i--) {
            int i_start = (i - kl_val >= 0) ? i - kl_val : 0;
            
            for (int j = 0; j < nrhs_val; j++) {
                for (int k = i_start; k < i; k++) {
                    int band_idx = kl_val + k - i;
                    if (band_idx >= 0) {
                        B[i + j * ldb_val] -= AB[band_idx + i * ldab_val] * B[k + j * ldb_val];
                    }
                }
            }
        }
    }
}
