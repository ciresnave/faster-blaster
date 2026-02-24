/* sgbtrf - banded LU factorization with partial pivoting (single precision) */
void sgbtrf_(const int *m, const int *n, const int *kl, const int *ku, float *AB, 
             const int *ldab, int *ipiv, int *info)
{
    int mm = *m;
    int nn = *n;
    int kl_val = *kl;
    int ku_val = *ku;
    int ldab_val = *ldab;
    int minmn = (mm < nn) ? mm : nn;
    
    *info = 0;
    
    // Process each column
    for (int j = 0; j < minmn; j++) {
        // Row range for this column: max(0, j-kl_val) to min(mm-1, j+ku_val)
        int irow_start = (j - kl_val >= 0) ? j - kl_val : 0;
        int irow_end = (j + ku_val < mm) ? j + ku_val : mm - 1;
        
        // Find pivot: max element in column j starting from row j
        int pivot_row = j;
        float pivot_val = fabsf(AB[kl_val + j + j * ldab_val]);
        
        for (int i = j + 1; i <= irow_end; i++) {
            int band_idx = kl_val + j - i;  // Band storage row index
            if (band_idx >= 0) {
                float val = fabsf(AB[band_idx + i * ldab_val]);
                if (val > pivot_val) {
                    pivot_val = val;
                    pivot_row = i;
                }
            }
        }
        
        ipiv[j] = pivot_row + 1;  // 1-based
        
        // Check for singularity
        if (pivot_val == 0.0f) {
            *info = j + 1;
            return;
        }
        
        // Swap rows if needed
        if (pivot_row != j) {
            int i1_start = (j - ku_val >= 0) ? j - ku_val : 0;
            int i1_end = (j + kl_val < nn) ? j + kl_val : nn - 1;
            
            for (int i = i1_start; i <= i1_end; i++) {
                float temp = AB[kl_val + j - i + i * ldab_val];
                AB[kl_val + j - i + i * ldab_val] = AB[kl_val + pivot_row - i + i * ldab_val];
                AB[kl_val + pivot_row - i + i * ldab_val] = temp;
            }
        }
        
        // Scale L factors
        float diag = AB[kl_val + j + j * ldab_val];
        for (int i = j + 1; i <= irow_end && i < mm; i++) {
            int band_idx = kl_val + j - i;
            if (band_idx >= 0) {
                AB[band_idx + i * ldab_val] /= diag;
            }
        }
        
        // Update trailing submatrix
        for (int jj = j + 1; jj < nn; jj++) {
            int jrow_start = (jj - kl_val >= 0) ? jj - kl_val : 0;
            int jrow_end = (jj + ku_val < mm) ? jj + ku_val : mm - 1;
            
            int i_overlap_start = (j + 1 > jrow_start) ? j + 1 : jrow_start;
            int i_overlap_end = (irow_end < jrow_end) ? irow_end : jrow_end;
            
            for (int i = i_overlap_start; i <= i_overlap_end && i < mm; i++) {
                int band_i = kl_val + j - i;
                int band_j = kl_val + jj - i;
                if (band_i >= 0 && band_j >= 0) {
                    AB[band_j + i * ldab_val] -= AB[band_i + i * ldab_val] * AB[kl_val + jj - j + j * ldab_val];
                }
            }
        }
    }
}
