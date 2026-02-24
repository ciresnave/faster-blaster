#
#include <string.h>
#include <blas_reference.h>

/* Blocked LU factorization */
void sgetfb_(const int *m, const int *n, const int *nb, float *A, const int *lda,
              int *ipiv, int *info)
{
    int m_val = *m;
    int n_val = *n;
    int nb_val = *nb;
    int lda_val = *lda;
    *info = 0;

    if (m_val <= 0 || n_val <= 0 || nb_val <= 0) return;

    /* Use SGETRF for blocks */
    int k = (m_val < n_val) ? m_val : n_val;
    
    for (int i = 0; i < k; i += nb_val) {
        int ib = (i + nb_val < k) ? nb_val : (k - i);
        
        /* Factor block */
        sgetrf_(&m_val, &ib, &A[i * lda_val], &lda_val, &ipiv[i], info);
        if (*info != 0) return;
    }
}
