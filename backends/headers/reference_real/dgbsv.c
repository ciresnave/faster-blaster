/* dgbsv - banded linear system solver driver (double precision) */
void dgbsv_(const int *n, const int *kl, const int *ku, const int *nrhs, double *AB,
            const int *ldab, int *ipiv, double *B, const int *ldb, int *info)
{
    int nn = *n;
    int kl_val = *kl;
    int ku_val = *ku;
    int nrhs_val = *nrhs;
    int ldab_val = *ldab;
    int ldb_val = *ldb;
    
    // Call factorization
    dgbtrf_(&nn, &nn, &kl_val, &ku_val, AB, &ldab_val, ipiv, info);
    
    if (*info == 0) {
        // Call solve
        char trans = 'N';
        dgbtrs_(&trans, &nn, &kl_val, &ku_val, &nrhs_val, AB, &ldab_val, ipiv, B, &ldb_val, info);
    }
}
