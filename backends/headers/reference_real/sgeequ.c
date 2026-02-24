/* sgeequ - compute scaling factors for general matrix (single precision) */
void sgeequ_(const int *m, const int *n, const float *A, const int *lda, float *r, float *c,
             float *rowcnd, float *colcnd, float *amax, int *info)
{
    int mm = *m;
    int nn = *n;
    int lda_val = *lda;
    
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
        float cj = 0.0f;
        for (int i = 0; i < mm; i++) {
            float aij = fabsf(A[i + j * lda_val]);
            *amax = (*amax > aij) ? *amax : aij;
            
            if (aij > 0.0f) {
                float row_scale = r[i];
                r[i] = (row_scale > aij) ? row_scale : aij;
                cj = (cj > aij) ? cj : aij;
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
