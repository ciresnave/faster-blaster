/* dgbequ - compute scaling factors for banded matrix (double precision) */
void dgbequ_(const int *m, const int *n, const int *kl, const int *ku, const double *AB,
             const int *ldab, double *r, double *c, double *rowcnd, double *colcnd, double *amax, int *info)
{
    int mm = *m;
    int nn = *n;
    int kl_val = *kl;
    int ku_val = *ku;
    int ldab_val = *ldab;
    
    *info = 0;
    *amax = 0.0;
    
    // Initialize scaling factors to 1.0
    for (int i = 0; i < mm; i++) {
        r[i] = 1.0;
    }
    for (int j = 0; j < nn; j++) {
        c[j] = 1.0;
    }
    
    // Compute max element and scaling factors
    for (int j = 0; j < nn; j++) {
        int jrow_start = (j - ku_val >= 0) ? j - ku_val : 0;
        int jrow_end = (j + kl_val < mm) ? j + kl_val : mm - 1;
        
        double cj = 0.0;
        for (int i = jrow_start; i <= jrow_end; i++) {
            int band_idx = kl_val + j - i;
            if (band_idx >= 0) {
                double aij = fabs(AB[band_idx + i * ldab_val]);
                *amax = (*amax > aij) ? *amax : aij;
                
                if (aij > 0.0) {
                    double row_scale = r[i];
                    r[i] = (row_scale > aij) ? row_scale : aij;
                    cj = (cj > aij) ? cj : aij;
                }
            }
        }
        
        c[j] = (cj > 0.0) ? cj : 1.0;
    }
    
    // Compute inverse scaling factors
    for (int i = 0; i < mm; i++) {
        if (r[i] > 0.0) {
            r[i] = 1.0 / r[i];
        } else {
            *info = i + 1;
        }
    }
    
    for (int j = 0; j < nn; j++) {
        if (c[j] > 0.0) {
            c[j] = 1.0 / c[j];
        } else {
            *info = mm + j + 1;
        }
    }
    
    // Compute row and column condition numbers
    double rmin = 1e10, rmax = 0.0;
    for (int i = 0; i < mm; i++) {
        if (r[i] > 0.0) {
            rmin = (rmin < r[i]) ? rmin : r[i];
            rmax = (rmax > r[i]) ? rmax : r[i];
        }
    }
    *rowcnd = (rmin > 0.0) ? rmin / rmax : 0.0;
    
    double cmin = 1e10, cmax = 0.0;
    for (int j = 0; j < nn; j++) {
        if (c[j] > 0.0) {
            cmin = (cmin < c[j]) ? cmin : c[j];
            cmax = (cmax > c[j]) ? cmax : c[j];
        }
    }
    *colcnd = (cmin > 0.0) ? cmin / cmax : 0.0;
}
