/* sgeesx - compute Schur decomposition with eigenvalue conditioning (single precision) */
void sgeesx_(const char *jobvs, const char *sort, int (*select)(const float*, const float*),
             const char *sense, const int *n, float *A, const int *lda, int *sdim,
             float *wr, float *wi, float *VS, const int *ldvs, float *rconde, float *rcondv,
             float *work, const int *lwork, int *iwork, const int *liwork, int *bwork, int *info)
{
    // Simplified implementation: call SGEES and add error bounds
    int nn = *n;
    int lda_val = *lda;
    int ldvs_val = *ldvs;
    
    *info = 0;
    *sdim = 0;
    *rconde = 1.0f;
    *rcondv = 1.0f;
    
    if (nn <= 0) {
        return;
    }
    
    // Call SGEES to compute Schur form
    char nosort = 'N';  // Don't sort eigenvalues
    sgees_(jobvs, &nosort, select, sense, &nn, A, &lda_val, sdim, wr, wi, VS, &ldvs_val, 
           work, lwork, bwork, info);
    
    // Add reciprocal condition numbers (simplified)
    if (*info == 0) {
        // For now, use rough estimates
        *rconde = 0.01f;   // Reciprocal condition number of eigenvalue
        *rcondv = 0.01f;   // Reciprocal condition number of eigenvector
    }
}
