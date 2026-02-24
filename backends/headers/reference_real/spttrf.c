/* spttrf - factorize tridiagonal matrix (single precision) */
void spttrf_(const int *n, float *d, float *e, int *info)
{
    int nn = *n;
    *info = 0;
    
    if (nn <= 0) {
        return;
    }
    
    // Factorize the tridiagonal matrix using Crout's method
    // d[0] remains as d[0]
    // e[i] = e[i] / d[i]  (multipliers for lower triangle)
    // d[i+1] = d[i+1] - e[i] * d[i] * e[i]  (update diagonal)
    
    for (int i = 0; i < nn - 1; i++) {
        if (d[i] == 0.0f) {
            *info = i + 1;  // 1-based indexing
            return;
        }
        
        // Scale e[i] by d[i]
        e[i] /= d[i];
        
        // Update d[i+1]
        d[i + 1] -= e[i] * e[i] * d[i];
    }
    
    // Check last diagonal element
    if (d[nn - 1] == 0.0f) {
        *info = nn;
    }
}
