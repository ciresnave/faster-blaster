/* Equilibrate Cholesky */
#
#include <float.h>
#include <string.h>
#include <blas_reference.h>

/* Positive definite matrix equilibration */
void dpoequ_(const int *n, const double *A, const int *lda, double *s,
              double *scond, double *amax, int *info)
{
    int n_val = *n;
    int lda_val = *lda;
    *info = 0;
    *amax = 0.0;

    if (n_val <= 0) {
        *scond = 1.0;
        return;
    }

    /* For diagonal matrix, scaling factors based on sqrt of diagonal */
    double smin = DBL_MAX;
    double smax = 0.0;

    for (int i = 0; i < n_val; i++) {
        s[i] = A[i + i * lda_val];
        *amax = fmax(*amax, s[i]);
        
        if (s[i] <= 0.0) {
            *info = i + 1;
            return;
        }
        
        if (s[i] > 0.0) {
            smin = fmin(smin, s[i]);
            smax = fmax(smax, s[i]);
        }
    }

    /* Compute scaling factors */
    for (int i = 0; i < n_val; i++) {
        s[i] = 1.0 / sqrt(s[i]);
    }
    
    *scond = sqrt(smin / smax);
}
