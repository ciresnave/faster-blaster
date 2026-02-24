/* spoevx */
void spoevx_(const char* jobz, const char* range, const char* uplo, int* n, float* ap, float* vl, float* vu, int* il, int* iu, float* abstol, int* m, float* w, float* z, int* ldz, float* work, int* lwork, int* iwork, int* ifail, int* info) {
    int n_val = *n;
    int il_val = *il;
    int iu_val = *iu;
    int ldz_val = *ldz;
    float vl_val = *vl;
    float vu_val = *vu;
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
        float eig = ap[i + i*n_val];
        if (range[0] == 'A' || (range[0] == 'V' && eig >= vl_val && eig <= vu_val) || (range[0] == 'I' && i >= il_val-1 && i < iu_val)) {
            w[*m] = eig;
            (*m)++;
        }
    }
}
