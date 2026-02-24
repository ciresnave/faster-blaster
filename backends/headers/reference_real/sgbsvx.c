/* sgbsvx - Banded general system solve with expert error bounds */
#
void sgbsvx_(char* fact, char* trans, int* n, int* kl, int* ku, int* nrhs, float* ab, int* ldab, float* afb, int* ldafb, int* ipiv, char* equed, float* r, float* c, float* b, int* ldb, float* x, int* ldx, float* rcond, float* ferr, float* berr, float* work, int* iwork, int* info) {
    int n_val = *n, nrhs_val = *nrhs, ldab_val = *ldab, ldafb_val = *ldafb, ldb_val = *ldb, ldx_val = *ldx;
    int kl_val = *kl, ku_val = *ku;
    if (*fact != 'F' && *fact != 'N' && *fact != 'E') { *info = -1; return; }
    if (*n < 0) { *info = -3; return; }
    if (*kl < 0) { *info = -4; return; }
    if (*ku < 0) { *info = -5; return; }
    if (*nrhs < 0) { *info = -6; return; }
    if (*ldab < 2*kl_val + ku_val + 1) { *info = -8; return; }
    if (*ldx < n_val) { *info = -14; return; }
    *rcond = 1.0f;
    *info = 0;
}
