#
#include <string.h>

/**
 * SGEES - Compute Schur decomposition of a matrix
 *
 * Computes the Schur decomposition (A = Z*T*Z^T) where T is in Schur form
 * and Z is orthogonal. Optionally computes eigenvalues.
 */
void sgees_(const char *jobz, const char *sort, int (*select)(float, float),
            const int *n, float *A, const int *lda, int *sdim, float *wr, float *wi,
            float *z, const int *ldz, float *work, const int *lwork, int *bwork, int *info)
{
    int i, j, iinfo;
    int lwork_gehrd, lwork_hseqr, lwork_needed;
    float *work_ptr;
    
    *info = 0;
    *sdim = 0;
    
    /* Validate inputs */
    if (*n < 0) { *info = -4; return; }
    if (*lda < (*n > 1 ? *n : 1)) { *info = -6; return; }
    if (*ldz < (*n > 1 ? *n : 1)) { *info = -10; return; }
    
    /* Quick return */
    if (*n == 0) {
        if (*lwork == -1) work[0] = 1.0f;
        return;
    }
    
    if (*n == 1) {
        if (jobz[0] == 'V') z[0] = 1.0f;
        wr[0] = A[0];
        wi[0] = 0.0f;
        *sdim = 1;
        if (*lwork == -1) work[0] = 1.0f;
        return;
    }
    
    /* Determine workspace needed */
    lwork_gehrd = *n;
    lwork_hseqr = *n;
    lwork_needed = 2 * *n + 32 * *n;  /* Approximate for both GEHRD and HSEQR */
    
    if (*lwork == -1) {
        work[0] = (float)lwork_needed;
        return;
    }
    
    if (*lwork < lwork_needed) { *info = -12; return; }
    
    /* Step 1: Reduce to Hessenberg form A = Z*H*Z^T */
    int ilo = 1, ihi = *n;
    float *tau = work;
    work_ptr = work + *n;
    int lwork_red = *lwork - *n;
    
    sgehrd_(n, &ilo, &ihi, A, lda, tau, work_ptr, &lwork_red, &iinfo);
    if (iinfo != 0) {
        *info = iinfo;
        return;
    }
    
    /* Initialize Z if needed */
    if (jobz[0] == 'V') {
        memset(z, 0, *ldz * *n * sizeof(float));
        for (i = 0; i < *n; i++) z[i * *ldz + i] = 1.0f;
    }
    
    /* Step 2: Apply implicit QR iteration to upper Hessenberg matrix */
    char job_schur = 'S';
    char compz_arg = (jobz[0] == 'V') ? 'I' : 'N';
    lwork_red = *lwork - (work_ptr - work);
    
    shseqr_(&job_schur, &compz_arg, n, &ilo, &ihi, A, lda, wr, wi, 
            z, ldz, work_ptr, &lwork_red, &iinfo);
    if (iinfo != 0) {
        *info = iinfo + 1;
        return;
    }
    
    /* Count selected eigenvalues if sorting requested */
    if (sort[0] != 'N' && select != NULL) {
        *sdim = 0;
        for (i = 0; i < *n; i++) {
            if (select(wr[i], wi[i])) {
                *sdim = *sdim + 1;
                bwork[i] = 1;
            } else {
                bwork[i] = 0;
            }
        }
    } else {
        *sdim = *n;
    }
}
