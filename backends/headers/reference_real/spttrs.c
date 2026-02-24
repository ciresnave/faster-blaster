/* spttrs - solve tridiagonal system using factorization (single precision) */
void spttrs_(const int *n, const int *nrhs, const float *d, const float *e, float *B, 
             const int *ldb, int *info)
{
    int nn = *n;
    int nrhs_val = *nrhs;
    int ldb_val = *ldb;
    
    *info = 0;
    
    if (nn <= 0 || nrhs_val <= 0) {
        return;
    }
    
    // Solve L*y = b using forward substitution
    // Since L is unit lower triangular (with 1s on diagonal),
    // we only need to apply the subdiagonal multipliers
    for (int j = 0; j < nrhs_val; j++) {
        for (int i = 1; i < nn; i++) {
            B[i + j * ldb_val] -= e[i - 1] * B[i - 1 + j * ldb_val];
        }
    }
    
    // Solve D*z = y: diagonal scaling
    for (int j = 0; j < nrhs_val; j++) {
        for (int i = 0; i < nn; i++) {
            if (d[i] != 0.0f) {
                B[i + j * ldb_val] /= d[i];
            }
        }
    }
    
    // Solve L^T*x = z using backward substitution
    for (int j = 0; j < nrhs_val; j++) {
        for (int i = nn - 2; i >= 0; i--) {
            B[i + j * ldb_val] -= e[i] * B[i + 1 + j * ldb_val];
        }
    }
}
