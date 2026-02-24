/* slagtm - Tridiagonal matrix-vector multiplication */
#
void slagtm_(char* trans, int* n, int* nrhs, float* alpha, float* dl, float* d, float* du, float* x, int* ldx, float* beta, float* b, int* ldb) {
    int n_val = *n, nrhs_val = *nrhs, ldx_val = *ldx, ldb_val = *ldb;
    for (int j = 0; j < nrhs_val; j++) {
        for (int i = 0; i < n_val; i++) {
            float sum = 0.0f;
            if (i > 0) sum += dl[i-1] * x[(i-1) + j*ldx_val];
            sum += d[i] * x[i + j*ldx_val];
            if (i < n_val - 1) sum += du[i] * x[(i+1) + j*ldx_val];
            b[i + j*ldb_val] = (*alpha) * sum + (*beta) * b[i + j*ldb_val];
        }
    }
}
