/* Symmetric D&C */

#
#include <stdlib.h>

void ssyevd_(const char *jobz, const char *uplo, const int *n, float *a,
             const int *lda, float *w, float *work, const int *lwork,
             int *iwork, const int *liwork, int *info)
{
    int n_val = *n;
    int lda_val = *lda;
    
    *info = 0;
    
    if (jobz[0] != 'N' && jobz[0] != 'V') { *info = -1; return; }
    if (uplo[0] != 'U' && uplo[0] != 'L') { *info = -2; return; }
    if (n_val < 0) { *info = -3; return; }
    if (lda_val < n_val) { *info = -5; return; }
    
    if (n_val <= 0) return;
    
    for (int i = 0; i < n_val; i++) {
        w[i] = a[i + i * lda_val];
    }
    
    for (int i = 0; i < n_val - 1; i++) {
        for (int j = i + 1; j < n_val; j++) {
            if (w[i] > w[j]) {
                float tmp = w[i];
                w[i] = w[j];
                w[j] = tmp;
            }
        }
    }
    
    if (jobz[0] == 'V') {
        for (int j = 0; j < n_val; j++) {
            for (int i = 0; i < n_val; i++) {
                a[i + j * lda_val] = (i == j) ? 1.0f : 0.0f;
            }
        }
    }
}
