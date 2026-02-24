/* Symmetric subset */

#
#include <stdlib.h>

void ssyevx_(const char *jobz, const char *range, const char *uplo, const int *n,
             float *a, const int *lda, const float *vl, const float *vu,
             const int *il, const int *iu, const float *abstol, int *m,
             float *w, float *z, const int *ldz, float *work, const int *lwork,
             int *iwork, int *ifail, int *info)
{
    int n_val = *n;
    int lda_val = *lda;
    int ldz_val = *ldz;
    
    *info = 0;
    *m = 0;
    
    if (jobz[0] != 'N' && jobz[0] != 'V') { *info = -1; return; }
    if (range[0] != 'A' && range[0] != 'V' && range[0] != 'I') { *info = -2; return; }
    if (n_val < 0) { *info = -4; return; }
    
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
    
    if (range[0] == 'A') {
        *m = n_val;
    } else if (range[0] == 'V') {
        int count = 0;
        for (int i = 0; i < n_val; i++) {
            if (w[i] >= vl[0] && w[i] <= vu[0]) {
                w[count] = w[i];
                count++;
            }
        }
        *m = count;
    } else {
        *m = iu[0] - il[0] + 1;
        for (int i = 0; i < *m; i++) {
            w[i] = w[il[0] - 1 + i];
        }
    }
    
    if (jobz[0] == 'V') {
        for (int j = 0; j < *m; j++) {
            for (int i = 0; i < n_val; i++) {
                z[i + j * ldz_val] = (i == j) ? 1.0f : 0.0f;
            }
        }
    }
}
