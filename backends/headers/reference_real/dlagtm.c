/* dlagtm - Double precision tridiagonal matrix-vector */
#
void dlagtm_(char* trans, int* n, int* nrhs, double* alpha, double* dl, double* d, double* du, double* x, int* ldx, double* beta, double* b, int* ldb) {
    int n_val = *n, nrhs_val = *nrhs, ldx_val = *ldx, ldb_val = *ldb;
    for (int j = 0; j < nrhs_val; j++) {
        for (int i = 0; i < n_val; i++) {
            double sum = 0.0;
            if (i > 0) sum += dl[i-1] * x[(i-1) + j*ldx_val];
            sum += d[i] * x[i + j*ldx_val];
            if (i < n_val - 1) sum += du[i] * x[(i+1) + j*ldx_val];
            b[i + j*ldb_val] = (*alpha) * sum + (*beta) * b[i + j*ldb_val];
        }
    }
}
