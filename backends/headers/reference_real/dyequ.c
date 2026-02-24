/* Equilibrate symmetric */
#
#include <float.h>
#include <string.h>
#include <blas_reference.h>

/* Symmetric matrix equilibration - compute row scaling factors */
void dyequ_(const char *uplo, const int *n, const double *A, const int *lda,
             double *s, double *scond, double *amax, int *info)
{
    int n_val = *n;
    int lda_val = *lda;
    *info = 0;
    *amax = 0.0;

    if (n_val <= 0) {
        *scond = 1.0;
        return;
    }

    int is_upper = (*uplo == 'U' || *uplo == 'u');
    double smin = DBL_MAX;
    double smax = 0.0;

    /* For symmetric matrix, find max in each row/column */
    for (int i = 0; i < n_val; i++) {
        s[i] = 0.0;
        
        if (is_upper) {
            /* Upper triangle */
            for (int j = 0; j <= i; j++) {
                s[i] = fmax(s[i], fabs(A[j + i * lda_val]));
            }
        } else {
            /* Lower triangle */
            for (int j = i; j < n_val; j++) {
                s[i] = fmax(s[i], fabs(A[j + i * lda_val]));
            }
        }
        
        *amax = fmax(*amax, s[i]);
        
        if (s[i] > 0.0) {
            smin = fmin(smin, s[i]);
            smax = fmax(smax, s[i]);
        }
    }

    /* Compute scaling factors */
    if (*amax == 0.0) {
        *scond = 1.0;
    } else {
        for (int i = 0; i < n_val; i++) {
            if (s[i] > 0.0) {
                s[i] = 1.0 / sqrt(s[i]);
            } else {
                s[i] = 1.0;
            }
        }
        
        *scond = (smin == 0.0 || smax == 0.0) ? 1.0 : smin / smax;
    }
}
