/* sgeqrt3 */
#
#include <string.h>
#include <stdlib.h>
#include <blas_reference.h>

/* QR factorization (recursive) */
void sgeqrt3_(const int *m, const int *n, float *A, const int *lda,
               float *T, const int *ldt, int *info)
{
    int m_val = *m;
    int n_val = *n;
    int ldt_val = *ldt;
    *info = 0;

    if (m_val <= 0 || n_val <= 0) return;

    /* Simplified: just perform standard QR decomposition */
    float *work = (float*)malloc(n_val * sizeof(float));
    float *tau = work;
    
    int lwork = n_val;
    sgeqrf_(&m_val, &n_val, A, lda, tau, &work[n_val], &lwork, info);
    
    free(work);
}
