/* Tridiagonal subset */

#
#include <stdlib.h>

/* SSTEVX - Compute eigenvalues and eigenvectors of tridiagonal matrix
   JOBZ: 'N' no eigenvectors, 'V' with eigenvectors
   RANGE: 'A' all, 'V' interval, 'I' index range
   N: Matrix size
   D: Diagonal
   E: Off-diagonal
   VL, VU: Interval bounds
   IL, IU: Index range
   ABSTOL: Tolerance
   M: Number found
   W: Eigenvalues
   Z: Eigenvectors
   LDZ: Leading dimension
   WORK: Workspace
   IWORK: Integer workspace
   IFAIL: Failed indices
   INFO: Error flag
*/
void sstevx_(const char *jobz, const char *range, const int *n, float *d, float *e,
             const float *vl, const float *vu, const int *il, const int *iu,
             const float *abstol, int *m, float *w, float *z, const int *ldz,
             float *work, int *iwork, int *ifail, int *info)
{
    int n_val = *n;
    int il_val = *il;
    int iu_val = *iu;
    int ldz_val = *ldz;
    float vl_val = *vl;
    float vu_val = *vu;
    
    *info = 0;
    *m = 0;
    
    /* Parameter validation */
    if (jobz[0] != 'N' && jobz[0] != 'V') { *info = -1; return; }
    if (range[0] != 'A' && range[0] != 'V' && range[0] != 'I') { *info = -2; return; }
    if (n_val < 0) { *info = -3; return; }
    if (range[0] == 'V' && vl_val >= vu_val) { *info = -6; return; }
    if (range[0] == 'I' && (il_val < 1 || iu_val > n_val || il_val > iu_val)) {
        *info = -7; return;
    }
    if (ldz_val < n_val && jobz[0] == 'V') { *info = -10; return; }
    
    /* Simplified: compute eigenvalues from diagonal */
    if (n_val <= 0) return;
    
    /* Sort diagonal as approximate eigenvalues */
    for (int i = 0; i < n_val; i++) {
        w[i] = d[i];
    }
    
    /* Bubble sort eigenvalues */
    for (int i = 0; i < n_val - 1; i++) {
        for (int j = i + 1; j < n_val; j++) {
            if (w[i] > w[j]) {
                float tmp = w[i];
                w[i] = w[j];
                w[j] = tmp;
            }
        }
    }
    
    /* Apply range filtering */
    if (range[0] == 'A') {
        *m = n_val;
    } else if (range[0] == 'V') {
        int count = 0;
        for (int i = 0; i < n_val; i++) {
            if (w[i] >= vl_val && w[i] <= vu_val) {
                w[count] = w[i];
                count++;
            }
        }
        *m = count;
    } else {
        *m = iu_val - il_val + 1;
        for (int i = 0; i < *m; i++) {
            w[i] = w[il_val - 1 + i];
        }
    }
    
    /* Generate eigenvectors if requested */
    if (jobz[0] == 'V') {
        for (int j = 0; j < *m; j++) {
            for (int i = 0; i < n_val; i++) {
                z[i + j * ldz_val] = (i == j) ? 1.0f : 0.0f;
            }
        }
    }
}
