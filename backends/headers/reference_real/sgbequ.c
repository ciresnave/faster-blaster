/* sgbequ - compute scaling factors for banded matrix (single precision) */
void sgbequ_(const int *m, const int *n, const int *kl, const int *ku, const float *AB,
             const int *ldab, float *r, float *c, float *rowcnd, float *colcnd, float *amax, int *info)
{
    int mm = *m;
    int nn = *n;
    int kl_val = *kl;
    int ku_val = *ku;
    int ldab_val = *ldab;
    
    *info = 0;
    *amax = 0.0f;
    
    // Initialize scaling factors to 1.0
    for (int i = 0; i < mm; i++) {
        r[i] = 1.0f;
    }
    for (int j = 0; j < nn; j++) {
        c[j] = 1.0f;
    }
    
    // Compute max element and scaling factors
    for (int j = 0; j < nn; j++) {
        int jrow_start = (j - ku_val >= 0) ? j - ku_val : 0;
        int jrow_end = (j + kl_val < mm) ? j + kl_val : mm - 1;
        
        float cj = 0.0f;
        for (int i = jrow_start; i <= jrow_end; i++) {
            int band_idx = kl_val + j - i;
            if (band_idx >= 0) {
                float aij = fabsf(AB[band_idx + i * ldab_val]);
                *amax = (*amax > aij) ? *amax : aij;
                
                if (aij > 0.0f) {
                    float row_scale = r[i];
                    r[i] = (row_scale > aij) ? row_scale : aij;
                    cj = (cj > aij) ? cj : aij;
                }
            }
        }
        
        c[j] = (cj > 0.0f) ? cj : 1.0f;
    }
    
    // Compute inverse scaling factors
    for (int i = 0; i < mm; i++) {
        if (r[i] > 0.0f) {
            r[i] = 1.0f / r[i];
        } else {
            *info = i + 1;
        }
    }
    
    for (int j = 0; j < nn; j++) {
        if (c[j] > 0.0f) {
            c[j] = 1.0f / c[j];
        } else {
            *info = mm + j + 1;
        }
    }
    
    // Compute row and column condition numbers
    float rmin = 1e10f, rmax = 0.0f;
    for (int i = 0; i < mm; i++) {
        if (r[i] > 0.0f) {
            rmin = (rmin < r[i]) ? rmin : r[i];
            rmax = (rmax > r[i]) ? rmax : r[i];
        }
    }
    *rowcnd = (rmin > 0.0f) ? rmin / rmax : 0.0f;
    
    float cmin = 1e10f, cmax = 0.0f;
    for (int j = 0; j < nn; j++) {
        if (c[j] > 0.0f) {
            cmin = (cmin < c[j]) ? cmin : c[j];
            cmax = (cmax > c[j]) ? cmax : c[j];
        }
    }
    *colcnd = (cmin > 0.0f) ? cmin / cmax : 0.0f;
}
