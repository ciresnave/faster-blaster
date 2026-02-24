/* sgbsv - banded linear system solver driver (single precision) */
void sgbsv_(const int *n, const int *kl, const int *ku, const int *nrhs, float *AB,
            const int *ldab, int *ipiv, float *B, const int *ldb, int *info)
{
    int nn = *n;
    int kl_val = *kl;
    int ku_val = *ku;
    int nrhs_val = *nrhs;
    int ldab_val = *ldab;
    int ldb_val = *ldb;
    
    // Call factorization
    sgbtrf_(&nn, &nn, &kl_val, &ku_val, AB, &ldab_val, ipiv, info);
    
    if (*info == 0) {
        // Call solve
        char trans = 'N';
        sgbtrs_(&trans, &nn, &kl_val, &ku_val, &nrhs_val, AB, &ldab_val, ipiv, B, &ldb_val, info);
    }
}
