/* QR tall-skinny */
#
#include <string.h>
#include <blas_reference.h>

/* QR factorization (tall-skinny) */
void dgeqrt_(const int *m, const int *n, const int *nb, double *A, const int *lda,
              double *T, const int *ldt, double *work, int *info)
{
    int m_val = *m;
    int n_val = *n;
    int nb_val = *nb;
    int ldt_val = *ldt;
    *info = 0;

    if (m_val <= 0 || n_val <= 0 || nb_val <= 0) return;

    /* Simplified: just perform standard QR decomposition */
    double *tau = work;
    
    dgeqrf_(&m_val, &n_val, A, lda, tau, &work[n_val], NULL, info);
}
