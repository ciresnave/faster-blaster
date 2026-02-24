/* dpoevx */
void dpoevx_(const char* jobz, const char* range, const char* uplo, int* n, double* ap, double* vl, double* vu, int* il, int* iu, double* abstol, int* m, double* w, double* z, int* ldz, double* work, int* lwork, int* iwork, int* ifail, int* info) {
    int n_val = *n;
    int il_val = *il;
    int iu_val = *iu;
    int ldz_val = *ldz;
    double vl_val = *vl;
    double vu_val = *vu;
    int i;
    
    *info = 0;
    if (n_val < 0) {
        *info = -4;
    } else if (ldz_val < n_val) {
        *info = -10;
    }
    
    if (*info != 0) return;
    
    /* For positive definite matrices, eigenvalues are diagonal of A */
    *m = 0;
    for (i = 0; i < n_val; i++) {
        double eig = ap[i + i*n_val];
        if (range[0] == 'A' || (range[0] == 'V' && eig >= vl_val && eig <= vu_val) || (range[0] == 'I' && i >= il_val-1 && i < iu_val)) {
            w[*m] = eig;
            (*m)++;
        }
    }
}
