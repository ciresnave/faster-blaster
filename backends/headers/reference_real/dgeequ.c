/* dgeequ - compute scaling factors for general matrix (double precision) */
void dgeequ_(const int *m, const int *n, const double *A, const int *lda, double *r, double *c,
             double *rowcnd, double *colcnd, double *amax, int *info)
{
    int mm = *m;
    int nn = *n;
    int lda_val = *lda;
    
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
        double cj = 0.0;
        for (int i = 0; i < mm; i++) {
            double aij = fabs(A[i + j * lda_val]);
            *amax = (*amax > aij) ? *amax : aij;
            
            if (aij > 0.0) {
                double row_scale = r[i];
                r[i] = (row_scale > aij) ? row_scale : aij;
                cj = (cj > aij) ? cj : aij;
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
